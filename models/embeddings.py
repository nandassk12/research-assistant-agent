"""
models/embeddings.py

Handles embedding generation and persistent vector storage for the
Personal Research Assistant Agent using Sentence Transformers and ChromaDB.
"""

import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR: str = "./chroma_db"
CHROMA_COLLECTION_NAME: str = "research_papers"
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
RETRIEVER_TOP_K: int = 5

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Create and return a HuggingFaceEmbeddings instance backed by the
    all-MiniLM-L6-v2 Sentence Transformer model.

    Returns
    -------
    HuggingFaceEmbeddings
        A LangChain embedding object ready to encode text.

    Raises
    ------
    RuntimeError
        If the model cannot be loaded.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Loaded embedding model: %s", EMBEDDING_MODEL_NAME)
        return embeddings
    except Exception as exc:
        logger.error("Failed to load embedding model '%s': %s", EMBEDDING_MODEL_NAME, exc)
        raise RuntimeError(f"Could not initialise embeddings: {exc}") from exc


def get_vectorstore() -> Chroma:
    """
    Create or open a persistent ChromaDB vector store.

    The store is saved to ``./chroma_db`` and uses the collection name
    ``research_papers``.

    Returns
    -------
    Chroma
        A LangChain Chroma vector store instance.

    Raises
    ------
    RuntimeError
        If the vector store cannot be created or opened.
    """
    try:
        vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_PERSIST_DIR,
        )
        logger.info(
            "Opened ChromaDB collection '%s' at '%s'",
            CHROMA_COLLECTION_NAME,
            CHROMA_PERSIST_DIR,
        )
        return vectorstore
    except Exception as exc:
        logger.error("Failed to initialise ChromaDB vector store: %s", exc)
        raise RuntimeError(f"Could not initialise vector store: {exc}") from exc


def add_papers_to_store(papers: List[Dict[str, Any]]) -> int:
    """
    Chunk and upsert a list of research papers into the persistent vector store.

    Each paper dictionary **must** contain the following keys:

    * ``title``   – Paper title (str)
    * ``abstract``– Paper abstract / full text to embed (str)
    * ``authors`` – Author list (str or list)
    * ``year``    – Publication year (str or int)
    * ``url``     – Source URL (str)
    * ``source``  – Provider identifier, e.g. ``"arxiv"`` (str)

    Papers that already exist in the store (detected by matching ``url``
    stored in chunk metadata) are silently skipped.

    Parameters
    ----------
    papers : list of dict
        List of paper metadata dictionaries as described above.

    Returns
    -------
    int
        Number of new papers whose chunks were added to the store.

    Raises
    ------
    ValueError
        If ``papers`` is not a list or contains non-dict items.
    RuntimeError
        If the vector store cannot be accessed.
    """
    if not isinstance(papers, list):
        raise ValueError("`papers` must be a list of dicts.")

    try:
        vectorstore = get_vectorstore()
    except RuntimeError as exc:
        raise RuntimeError("Cannot add papers – vector store unavailable.") from exc

    # -----------------------------------------------------------------------
    # Retrieve URLs already present in the collection to avoid duplicates.
    # -----------------------------------------------------------------------
    try:
        existing_data = vectorstore.get(include=["metadatas"])
        existing_urls: set = {
            meta.get("url", "")
            for meta in (existing_data.get("metadatas") or [])
            if meta
        }
    except Exception as exc:
        logger.warning("Could not fetch existing metadata; duplicate check skipped: %s", exc)
        existing_urls = set()

    # -----------------------------------------------------------------------
    # Text splitter shared across all papers.
    # -----------------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    documents: List[Document] = []
    papers_added: int = 0

    for paper in papers:
        if not isinstance(paper, dict):
            logger.warning("Skipping non-dict item in papers list: %r", paper)
            continue

        url: str = str(paper.get("url", "")).strip()
        title: str = str(paper.get("title", "Untitled")).strip()

        # Skip if this paper URL is already indexed.
        if url and url in existing_urls:
            logger.debug("Paper already in store, skipping: %s", title)
            continue

        abstract: str = str(paper.get("abstract", "")).strip()
        if not abstract:
            logger.warning("Paper '%s' has no abstract; skipping.", title)
            continue

        authors: str = (
            ", ".join(paper["authors"])
            if isinstance(paper.get("authors"), list)
            else str(paper.get("authors", "Unknown"))
        )

        metadata: Dict[str, Any] = {
            "title": title,
            "authors": authors,
            "year": str(paper.get("year", "")),
            "url": url,
            "source": str(paper.get("source", "")),
        }

        # Prepend the title so every chunk carries context.
        full_text: str = f"Title: {title}\n\n{abstract}"
        chunks: List[Document] = splitter.create_documents(
            texts=[full_text],
            metadatas=[metadata],
        )

        documents.extend(chunks)
        papers_added += 1
        logger.debug("Prepared %d chunk(s) for '%s'", len(chunks), title)

    # -----------------------------------------------------------------------
    # Batch-add all new chunks in one call for efficiency.
    # -----------------------------------------------------------------------
    if documents:
        try:
            vectorstore.add_documents(documents)
            logger.info(
                "Added %d paper(s) (%d chunk(s)) to the vector store.",
                papers_added,
                len(documents),
            )
        except Exception as exc:
            logger.error("Failed to add documents to vector store: %s", exc)
            raise RuntimeError(f"Vector store write failed: {exc}") from exc

    return papers_added


def get_retriever(
    vectorstore: Optional[Chroma] = None,
) -> Any:
    """
    Return a similarity-based retriever that fetches the top-k most relevant
    chunks from the vector store.

    Parameters
    ----------
    vectorstore : Chroma, optional
        An existing Chroma instance to wrap.  If ``None`` (default) a new
        instance is created via :func:`get_vectorstore`.

    Returns
    -------
    VectorStoreRetriever
        A LangChain retriever configured for ``k=5`` similarity search.

    Raises
    ------
    RuntimeError
        If the vector store cannot be accessed.
    """
    try:
        store: Chroma = vectorstore if vectorstore is not None else get_vectorstore()
        retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_TOP_K},
        )
        logger.info("Created retriever with k=%d", RETRIEVER_TOP_K)
        return retriever
    except Exception as exc:
        logger.error("Failed to create retriever: %s", exc)
        raise RuntimeError(f"Could not create retriever: {exc}") from exc
# add at bottom temporarily
if __name__ == "__main__":
    vs = get_vectorstore()
    print("ChromaDB connected ✅")
    r = get_retriever()
    print("Retriever ready ✅")