"""
src/chatbot.py

Context-aware conversational RAG chatbot for the Personal Research Assistant Agent.

The chat chain is primed with the full research session context (query, summary,
top papers) so the assistant acts as a true research companion that knows exactly
which papers were found and what was discovered.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

from models.llm import get_llm
from models.embeddings import get_retriever

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
MEMORY_KEY: str = "chat_history"
OUTPUT_KEY: str = "answer"
CHAIN_VERBOSE: bool = False
MAX_CONTEXT_PAPERS: int = 5        # top N papers included in system prompt
SUMMARY_PREVIEW_CHARS: int = 800   # chars of summary shown to LLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_papers_for_prompt(papers: List[Dict[str, Any]]) -> str:
    """
    Format the top :data:`MAX_CONTEXT_PAPERS` papers as a numbered list
    suitable for embedding in a system prompt.

    Parameters
    ----------
    papers : list of dict
        Ranked paper dicts.

    Returns
    -------
    str
        Multi-line numbered list string.
    """
    if not papers:
        return "  (no papers available)"

    lines: List[str] = []
    for i, paper in enumerate(papers[:MAX_CONTEXT_PAPERS], 1):
        title = str(paper.get("title", "Untitled"))
        authors_raw = paper.get("authors", [])
        if isinstance(authors_raw, list):
            authors = ", ".join(authors_raw[:3]) + (" et al." if len(authors_raw) > 3 else "")
        else:
            authors = str(authors_raw)
        year = str(paper.get("year", "N/A"))
        lines.append(f"  {i}. {title} — {authors} ({year})")
    return "\n".join(lines)


def _build_system_prompt(
    query: str,
    summary: str,
    papers: List[Dict[str, Any]],
) -> str:
    """
    Build the system prompt that grounds the chatbot in the current research session.

    Parameters
    ----------
    query : str
        Original research query.
    summary : str
        LLM-generated research summary.
    papers : list of dict
        Ranked paper list from the agent pipeline.

    Returns
    -------
    str
        Full system prompt string.
    """
    papers_text = _format_papers_for_prompt(papers)
    summary_preview = (summary or "")[:SUMMARY_PREVIEW_CHARS]
    paper_count = len(papers) if papers else 0

    return f"""You are an expert research assistant. You have just analyzed \
{paper_count} academic papers about: '{query}'

RESEARCH SUMMARY:
{summary_preview}

PAPERS ANALYZED:
{papers_text}

YOUR ROLE:
- Answer questions specifically about these papers
- Compare methodologies and findings across papers
- Suggest which papers to read first based on the user's needs
- Explain technical concepts that appear in the papers
- Always cite specific paper titles when answering
- If asked something not covered in the papers, say so honestly

Always ground your answers in the actual papers found. Be concise and helpful."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_chat_chain(
    query: str = "",
    summary: str = "",
    papers: Optional[List[Dict[str, Any]]] = None,
) -> ConversationalRetrievalChain:
    """
    Build a context-aware ConversationalRetrievalChain for the current session.

    The chain is primed with a system prompt containing the research query,
    a preview of the generated summary, and the top papers, so the assistant
    can answer follow-up questions as a true research companion.

    Parameters
    ----------
    query : str, optional
        The original research query from this session.
    summary : str, optional
        The LLM-generated research summary report.
    papers : list of dict, optional
        Ranked paper dicts from the agent pipeline.

    Returns
    -------
    ConversationalRetrievalChain
        Ready-to-use chain with session context baked into the system prompt.

    Raises
    ------
    RuntimeError
        If the chain cannot be constructed.
    """
    papers = papers or []

    try:
        llm = get_llm()
        retriever = get_retriever()
        memory = ConversationBufferMemory(
            memory_key=MEMORY_KEY,
            return_messages=True,
            output_key=OUTPUT_KEY,
        )

        # Build a context-rich system prompt
        system_prompt = _build_system_prompt(query, summary, papers)

        # Combine system context with the retrieved document context
        condense_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            HumanMessagePromptTemplate.from_template("{question}"),
        ])

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=CHAIN_VERBOSE,
            condense_question_prompt=condense_prompt,
        )

        logger.info(
            "get_chat_chain: chain created for query '%s' with %d papers.",
            query[:60], len(papers),
        )
        return chain

    except Exception as exc:
        logger.error("get_chat_chain: failed to create chain — %s", exc)
        raise RuntimeError(f"Could not build chat chain: {exc}") from exc


def chat(
    chain: ConversationalRetrievalChain,
    user_message: str,
    papers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Send a user message to the RAG chain and return a grounded answer.

    Extracts source metadata and identifies suggested papers most relevant
    to the question for display in the UI.

    Parameters
    ----------
    chain : ConversationalRetrievalChain
        Chain instance created by :func:`get_chat_chain`.
    user_message : str
        User's follow-up question.
    papers : list of dict, optional
        Full paper list (used to populate ``suggested_papers``).

    Returns
    -------
    dict
        Keys:
        * ``answer``           – str, LLM response grounded in retrieved chunks.
        * ``sources``          – list of dicts with ``title``, ``url``.
        * ``suggested_papers`` – list of 2–3 paper title strings most relevant
                                 to the question (matched by keyword overlap).
    """
    papers = papers or []

    if not isinstance(user_message, str) or not user_message.strip():
        logger.warning("chat: received empty user message.")
        return {"answer": "Please enter a question.", "sources": [], "suggested_papers": []}

    try:
        response: Dict[str, Any] = chain.invoke({"question": user_message.strip()})

        answer: str = str(response.get(OUTPUT_KEY, "")).strip()
        if not answer:
            answer = "I could not find a relevant answer in the stored papers."

        # ── Extract unique source references ───────────────────────────────
        source_docs: List[Any] = response.get("source_documents", [])
        seen_urls: set = set()
        sources: List[Dict[str, str]] = []
        for doc in source_docs:
            meta: Dict[str, Any] = getattr(doc, "metadata", {}) or {}
            url: str = str(meta.get("url", "")).strip()
            title: str = str(meta.get("title", "Unknown")).strip()
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({"title": title, "url": url})

        # ── Suggested papers (simple keyword overlap) ──────────────────────
        suggested_papers: List[str] = []
        if papers:
            query_lower = user_message.lower()
            scored = []
            for p in papers:
                title = str(p.get("title", ""))
                abstract = str(p.get("abstract", ""))
                text = (title + " " + abstract).lower()
                # Count how many query words appear in the paper text
                overlap = sum(1 for word in query_lower.split() if len(word) > 3 and word in text)
                scored.append((overlap, title))
            scored.sort(reverse=True)
            suggested_papers = [t for _, t in scored[:3] if t]

        logger.info("chat: %d source(s), %d suggestion(s).", len(sources), len(suggested_papers))
        return {"answer": answer, "sources": sources, "suggested_papers": suggested_papers}

    except Exception as exc:
        logger.error("chat: chain invocation failed — %s", exc)
        return {
            "answer": (
                "Sorry, I encountered an error while searching the papers. "
                "Please try rephrasing your question."
            ),
            "sources": [],
            "suggested_papers": [],
        }


def reset_chat(chain: ConversationalRetrievalChain) -> None:
    """
    Clear the conversation memory of a chat chain.

    Parameters
    ----------
    chain : ConversationalRetrievalChain
        Chain instance whose memory should be reset.
    """
    try:
        chain.memory.clear()
        logger.info("reset_chat: conversation memory cleared.")
    except Exception as exc:
        logger.warning("reset_chat: could not clear memory — %s", exc)
