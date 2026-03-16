"""
src/chatbot.py

Context-aware research chat using direct Groq LLM calls.

Replaces the brittle ConversationalRetrievalChain approach with a plain
dict-based context object and explicit message construction.  This avoids
all LangChain chain compatibility issues while retaining full RAG grounding.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from models.llm import get_llm
from models.embeddings import get_retriever

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_HISTORY_MESSAGES: int = 6      # keep last N turns to avoid token overflow
MAX_PAPERS_IN_PROMPT: int = 5      # top N papers listed in system prompt
SUMMARY_PREVIEW_CHARS: int = 600   # chars of summary embedded in prompt
CHUNKS_PREVIEW_CHARS: int = 1500   # chars of retrieved excerpt text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_papers_for_prompt(papers: List[Dict[str, Any]]) -> str:
    """
    Format the top :data:`MAX_PAPERS_IN_PROMPT` papers as a numbered list
    for embedding in the system prompt.

    Parameters
    ----------
    papers : list of dict
        Ranked paper dicts from the agent pipeline.

    Returns
    -------
    str
        Multi-line numbered list string.
    """
    if not papers:
        return "  (no papers available)"
    lines: List[str] = []
    for i, paper in enumerate(papers[:MAX_PAPERS_IN_PROMPT], 1):
        title = str(paper.get("title", "Untitled"))
        authors_raw = paper.get("authors", [])
        if isinstance(authors_raw, list):
            au = ", ".join(authors_raw[:3]) + (" et al." if len(authors_raw) > 3 else "")
        else:
            au = str(authors_raw)
        year = str(paper.get("year", "N/A"))
        lines.append(f"  {i}. {title} — {au} ({year})")
    return "\n".join(lines)


def _build_history_messages(history: List[Dict[str, str]]) -> List:
    """
    Convert the stored history list into LangChain message objects.

    Only the last :data:`MAX_HISTORY_MESSAGES` messages are included to
    prevent token overflow.

    Parameters
    ----------
    history : list of dict
        Each dict has ``role`` (``"user"`` or ``"assistant"``) and
        ``content`` keys.

    Returns
    -------
    list
        List of :class:`~langchain_core.messages.HumanMessage` /
        :class:`~langchain_core.messages.AIMessage` objects.
    """
    recent = history[-MAX_HISTORY_MESSAGES:]
    messages = []
    for msg in recent:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_chat_chain(
    query: str = "",
    summary: str = "",
    papers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build and return a research-session context dict.

    This replaces the old ``ConversationalRetrievalChain`` with a plain dict
    that stores session state.  No LangChain chain is constructed here —
    the LLM is called directly inside :func:`chat`.

    Parameters
    ----------
    query : str, optional
        The original research query for this session.
    summary : str, optional
        LLM-generated research summary from the agent pipeline.
    papers : list of dict, optional
        Ranked paper list from the agent pipeline.

    Returns
    -------
    dict
        Context dict with keys: ``query``, ``summary``, ``papers``,
        ``history``.
    """
    context: Dict[str, Any] = {
        "query": query,
        "summary": summary,
        "papers": papers or [],
        "history": [],
    }
    logger.info(
        "get_chat_chain: context created for query '%s' with %d papers.",
        query[:60], len(context["papers"]),
    )
    return context


def chat(
    context: Dict[str, Any],
    user_message: str,
) -> Dict[str, Any]:
    """
    Answer a follow-up question using direct LLM invocation + RAG retrieval.

    Workflow
    --------
    1. Retrieve relevant paper chunks from ChromaDB.
    2. Build a system prompt containing research context, top papers, and
       retrieved excerpts.
    3. Prepend recent history and append the user message.
    4. Call the Groq LLM directly (no chain).
    5. Update session history in ``context``.
    6. Return answer, sources, and updated context.

    Parameters
    ----------
    context : dict
        Session context dict created by :func:`get_chat_chain`.
    user_message : str
        User's follow-up question.

    Returns
    -------
    dict
        Keys:
        * ``answer``  – str, LLM response.
        * ``sources`` – list of dicts with ``title`` and ``url``.
        * ``context`` – updated context dict (caller must store this).
    """
    if not isinstance(user_message, str) or not user_message.strip():
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "context": context,
        }

    query      = context.get("query", "")
    summary    = context.get("summary", "")
    papers     = context.get("papers", [])
    history    = context.get("history", [])

    # ── Step 1: retrieve relevant chunks ─────────────────────────────────────
    docs = []
    sources: List[Dict[str, str]] = []
    try:
        retriever = get_retriever()
        docs = retriever.invoke(user_message.strip())
    except Exception as exc:
        logger.warning("chat: retriever failed — %s", exc)

    # Deduplicate sources from metadata
    seen_urls: set = set()
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        url   = str(meta.get("url", "")).strip()
        title = str(meta.get("title", "Unknown")).strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append({"title": title, "url": url})

    # ── Step 2: build system prompt ───────────────────────────────────────────
    papers_text = _format_papers_for_prompt(papers)
    chunks_text = "\n\n".join(
        getattr(doc, "page_content", "") for doc in docs
    )[:CHUNKS_PREVIEW_CHARS]
    summary_preview = (summary or "")[:SUMMARY_PREVIEW_CHARS]

    system_prompt = f"""You are an expert research assistant analyzing \
{len(papers)} papers about: '{query}'

RESEARCH CONTEXT:
{summary_preview}

PAPERS ANALYZED:
{papers_text}

RELEVANT PAPER EXCERPTS:
{chunks_text if chunks_text else "(no excerpts retrieved)"}

Always cite specific paper titles when answering.
Be concise and helpful.
If the question is not covered in the papers, say so honestly."""

    # ── Step 3: assemble message list ─────────────────────────────────────────
    messages = (
        [SystemMessage(content=system_prompt)]
        + _build_history_messages(history)
        + [HumanMessage(content=user_message.strip())]
    )

    # ── Step 4: call LLM ──────────────────────────────────────────────────────
    try:
        llm    = get_llm()
        resp   = llm.invoke(messages)
        answer = str(resp.content).strip() if resp and resp.content else ""
        if not answer:
            answer = "I could not find a relevant answer in the stored papers."
    except Exception as exc:
        logger.error("chat: LLM call failed — %s", exc)
        answer = (
            "Sorry, I encountered an error while generating an answer. "
            "Please try rephrasing your question."
        )

    # ── Step 5: update history ────────────────────────────────────────────────
    context["history"].append({"role": "user",      "content": user_message.strip()})
    context["history"].append({"role": "assistant", "content": answer})

    logger.info("chat: answered (%d chars), %d source(s).", len(answer), len(sources))
    return {"answer": answer, "sources": sources, "context": context}


def reset_chat(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clear the conversation history in the context dict.

    Parameters
    ----------
    context : dict
        Session context dict returned by :func:`get_chat_chain`.

    Returns
    -------
    dict
        The same context dict with ``history`` set to ``[]``.
    """
    try:
        context["history"] = []
        logger.info("reset_chat: history cleared.")
    except Exception as exc:
        logger.warning("reset_chat: failed to clear history — %s", exc)
    return context
