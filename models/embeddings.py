"""
models/embeddings.py

Handles embedding generation and persistent vector storage for the
Personal Research Assistant Agent using Sentence Transformers and ChromaDB.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple

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
FULLTEXT_CHUNK_SIZE: int = 1000
FULLTEXT_CHUNK_OVERLAP: int = 100
RETRIEVER_TOP_K: int = 5

# Standard academic section headings used for section-aware splitting
SECTION_PATTERNS: List[str] = [
    "Abstract", "Introduction", "Related Work", "Background",
    "Methodology", "Method", "Approach", "Experiments",
    "Results", "Evaluation", "Discussion", "Conclusion", "References",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def split_by_sections(
    text: str,
    section_names: List[str],
) -> List[Tuple[str, str]]:
    """
    Split *text* into ``(section_name, section_text)`` tuples by detecting
    standard academic headings.

    Detection is line-based: a line that matches a section name (case-insensitive,
    with optional numbering) is treated as a heading.  If no sections are
    detected the whole text is returned as a single ``("full_text", text)`` tuple.

    Parameters
    ----------
    text : str
        Raw paper text (e.g. from HTML extraction or PDF).
    section_names : list of str
        Section heading strings to search for.

    Returns
    -------
    list of (str, str)
        ``[(section_name, section_text), ...]`` preserving document order.
    """
    if not text or not text.strip():
        return []

    # Build a combined pattern: optional leading number + section name + optional colon
    pattern = re.compile(
        r"^(?:\d+\.?\s+)?({names})\s*:?\s*$".format(
            names="|".join(re.escape(s) for s in section_names)
        ),
        re.IGNORECASE | re.MULTILINE,
    )

    lines = text.splitlines(keepends=True)
    sections: List[Tuple[str, str]] = []
    current_name = "full_text"
    current_lines: List[str] = []

    for line in lines:
        m = pattern.match(line.strip())
        if m:
            # Save the previous section (if it has content)
            body = "".join(current_lines).strip()
            if body:
                sections.append((current_name, body))
            # Start a new section
            current_name = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Flush the last section
    body = "".join(current_lines).strip()
    if body:
        sections.append((current_name, body))

    # Fallback: no headings matched
    if not sections:
        return [("full_text", text)]

    return sections


def _chunk_paper(
    paper: Dict[str, Any],
    authors_str: str,
) -> List[Document]:
    """
    Convert a single paper dict into a list of LangChain :class:`Document`
    objects suitable for insertion into the vector store.

    If the paper contains a ``full_text`` field the text is split
    section-by-section with 1000-char chunks; otherwise the abstract is
    chunked with the original 500-char splitter.

    Parameters
    ----------
    paper : dict
        Paper metadata dict (must contain at least ``title``).
    authors_str : str
        Pre-formatted author string.

    Returns
    -------
    list of Document
        Ready-to-embed LangChain documents with metadata.
    """
    title        = str(paper.get("title", "Untitled")).strip()
    abstract     = str(paper.get("abstract", "")).strip()
    full_text    = str(paper.get("full_text", "")).strip()
    content_type = str(paper.get("content_type", "abstract"))

    base_metadata: Dict[str, Any] = {
        "title":        title,
        "authors":      authors_str,
        "year":         str(paper.get("year", "")),
        "url":          str(paper.get("url", "")),
        "source":       str(paper.get("source", "")),
        "citations":    str(paper.get("citations", "0")),
        "content_type": content_type,
    }

    documents: List[Document] = []

    if full_text:
        # ── Section-aware chunking for full-text papers ──────────────────────
        section_splitter = RecursiveCharacterTextSplitter(
            chunk_size=FULLTEXT_CHUNK_SIZE,
            chunk_overlap=FULLTEXT_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        sections = split_by_sections(full_text, SECTION_PATTERNS)
        for section_name, section_text in sections:
            if len(section_text.strip()) < 50:
                continue
            section_metadata = {**base_metadata, "section": section_name}
            prefixed = f"[Section: {section_name}]\n{section_text}"
            try:
                chunks = section_splitter.create_documents(
                    texts=[prefixed],
                    metadatas=[section_metadata],
                )
                documents.extend(chunks)
            except Exception as exc:
                logger.warning(
                    "_chunk_paper: failed to chunk section '%s' of '%s': %s",
                    section_name, title, exc,
                )
    else:
        # ── Fallback: abstract-only chunking ──────────────────────────────
        abstract_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        abstract_text = f"Title: {title}\n\nAbstract: {abstract}"
        abstract_metadata = {**base_metadata, "section": "abstract"}
        try:
            chunks = abstract_splitter.create_documents(
                texts=[abstract_text],
                metadatas=[abstract_metadata],
            )
            documents.extend(chunks)
        except Exception as exc:
            logger.warning(
                "_chunk_paper: failed to chunk abstract of '%s': %s", title, exc
            )

    return documents


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

        url: str   = str(paper.get("url", "")).strip()
        title: str = str(paper.get("title", "Untitled")).strip()

        # Skip if this paper URL is already indexed.
        if url and url in existing_urls:
            logger.debug("Paper already in store, skipping: %s", title)
            continue

        abstract: str = str(paper.get("abstract", "")).strip()
        full_text: str = str(paper.get("full_text", "")).strip()
        if not abstract and not full_text:
            logger.warning("Paper '%s' has no abstract or full text; skipping.", title)
            continue

        authors_str: str = (
            ", ".join(paper["authors"])
            if isinstance(paper.get("authors"), list)
            else str(paper.get("authors", "Unknown"))
        )

        # Section-aware chunking via helper
        try:
            paper_docs = _chunk_paper(paper, authors_str)
        except Exception as exc:
            logger.error("_chunk_paper failed for '%s': %s", title, exc)
            continue

        if paper_docs:
            documents.extend(paper_docs)
            papers_added += 1
            logger.debug(
                "Prepared %d chunk(s) for '%s' (content_type=%s)",
                len(paper_docs), title, paper.get("content_type", "abstract"),
            )

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