"""
models/llm.py

LLM provider and LLM-powered helper functions for the
Personal Research Assistant Agent.

All LLM functionality is centralised here and accessed by src/agent.py.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
GROQ_MODEL: str = "llama-3.3-70b-versatile"
LLM_TEMPERATURE: float = 0.3
LLM_MAX_TOKENS: int = 2048          # raised to fit the summary report
SUMMARIZE_TOP_N: int = 10           # max papers sent to the summariser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM singleton
# ---------------------------------------------------------------------------

def get_llm() -> ChatGroq:
    """
    Create and return a ChatGroq LLM instance.

    Returns
    -------
    ChatGroq
        Configured Groq language model client.

    Raises
    ------
    ValueError
        If ``GROQ_API_KEY`` is not set in the environment.
    """
    api_key: str = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

    return ChatGroq(
        api_key=api_key,
        model_name=GROQ_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


# ---------------------------------------------------------------------------
# LLM-powered helpers
# ---------------------------------------------------------------------------

def validate_keywords(keywords: List[str], original_query: str) -> Dict[str, Any]:
    """
    Ask the LLM to rate and refine a set of extracted search keywords.

    Sends a validation prompt to Groq and parses the JSON response.  If the
    LLM call fails or the response cannot be parsed, returns a safe fallback
    dict so that the agent pipeline can continue uninterrupted.

    Parameters
    ----------
    keywords : list of str
        Keywords extracted by the preprocessor.
    original_query : str
        The raw user query for context.

    Returns
    -------
    dict
        Keys:
        * ``confidence``       – int (1–10), how well keywords match the query.
        * ``refined_keywords`` – list[str], improved search terms from the LLM.
        * ``reason``           – str, one-line explanation from the LLM.

    Notes
    -----
    Fallback (on any error):
    ``{"confidence": 7, "refined_keywords": keywords, "reason": "fallback"}``
    """
    fallback: Dict[str, Any] = {
        "confidence": 7,
        "refined_keywords": keywords,
        "reason": "fallback",
    }

    if not keywords and not original_query.strip():
        logger.warning("validate_keywords: both keywords and query are empty.")
        return fallback

    prompt: str = (
        "You are a research keyword validator.\n"
        f"Original query: {original_query}\n"
        f"Extracted keywords: {keywords}\n\n"
        "Task: Rate how well these keywords represent the research query. "
        "Also provide 3-5 refined search terms.\n\n"
        "Respond in this EXACT JSON format (no extra text, no markdown):\n"
        "{\n"
        '  "confidence": <int 1-10>,\n'
        '  "refined_keywords": [<list of strings>],\n'
        '  "reason": "<one line explanation>"\n'
        "}"
    )

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        raw_text: str = (
            response.content if hasattr(response, "content") else str(response)
        ).strip()

        # Extract JSON block — tolerate extra prose before/after
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not json_match:
            logger.warning("validate_keywords: no JSON found in LLM response.")
            return fallback

        data: Dict[str, Any] = json.loads(json_match.group())

        confidence: int = int(data.get("confidence", 7))
        refined: List[str] = data.get("refined_keywords", keywords)
        reason: str = str(data.get("reason", ""))

        # Sanity-clamp confidence to [1, 10]
        confidence = max(1, min(10, confidence))

        logger.info(
            "validate_keywords: confidence=%d, refined=%s", confidence, refined
        )
        return {
            "confidence": confidence,
            "refined_keywords": refined if isinstance(refined, list) else keywords,
            "reason": reason,
        }

    except json.JSONDecodeError as exc:
        logger.warning("validate_keywords: JSON parse error — %s", exc)
    except Exception as exc:
        logger.error("validate_keywords: LLM call failed — %s", exc)

    return fallback


def summarize_papers(papers: List[Dict[str, Any]], query: str) -> str:
    """
    Generate a structured research summary report using the LLM.

    Sends up to :data:`SUMMARIZE_TOP_N` papers to Groq and asks it to
    produce a formatted research summary report.

    Parameters
    ----------
    papers : list of dict
        Ranked paper dicts (each with title, abstract, authors, year, url).
    query : str
        The original user query — included in the report header.

    Returns
    -------
    str
        Formatted research summary report.  On LLM failure, returns a
        human-readable fallback error string.
    """
    if not papers:
        return "No papers available to summarise."

    # Trim to top N and format for the prompt
    top_papers: List[Dict[str, Any]] = papers[:SUMMARIZE_TOP_N]
    paper_count: int = len(top_papers)

    papers_text: str = ""
    for i, paper in enumerate(top_papers, 1):
        authors = paper.get("authors", [])
        author_str = (
            ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            if isinstance(authors, list)
            else str(authors)
        )
        papers_text += (
            f"{i}. Title: {paper.get('title', 'N/A')}\n"
            f"   Authors: {author_str}\n"
            f"   Year: {paper.get('year', 'N/A')}\n"
            f"   URL: {paper.get('url', 'N/A')}\n"
            f"   Abstract: {str(paper.get('abstract', ''))[:400]}\n\n"
        )

    prompt: str = (
        f"You are an expert research analyst. Analyse these {paper_count} papers "
        f"related to the query: '{query}'\n\n"
        f"{papers_text}\n"
        "Generate a comprehensive research summary report in this EXACT format "
        "(preserve all headers exactly):\n\n"
        "RESEARCH SUMMARY REPORT\n"
        "========================\n"
        f"Query: {query}\n"
        f"Papers Analyzed: {paper_count}\n\n"
        "OVERVIEW\n"
        "--------\n"
        "[Write 2-3 paragraphs summarising what these papers collectively cover]\n\n"
        "KEY FINDINGS\n"
        "------------\n"
        "• [finding 1]\n"
        "• [finding 2]\n"
        "• [finding 3]\n"
        "• [finding 4]\n"
        "• [finding 5]\n\n"
        "TOP PAPERS\n"
        "----------\n"
        "[For each paper: Number. Title — Authors (Year)\\n   Summary: 1-2 lines\\n   Link: url]\n\n"
        "KEY TAKEAWAYS\n"
        "-------------\n"
        "[Describe common conclusions and patterns across the papers]\n\n"
        "RECOMMENDATIONS\n"
        "---------------\n"
        "[Which papers to read first and why]\n"
    )

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        summary: str = (
            response.content if hasattr(response, "content") else str(response)
        ).strip()

        if not summary:
            raise ValueError("LLM returned an empty response.")

        logger.info("summarize_papers: summary generated (%d chars).", len(summary))
        return summary

    except Exception as exc:
        logger.error("summarize_papers: LLM call failed — %s", exc)
        # Graceful fallback — plain-text report built from paper metadata
        fallback_lines = [
            "RESEARCH SUMMARY REPORT",
            "========================",
            f"Query: {query}",
            f"Papers Analyzed: {paper_count}",
            "",
            "NOTE: Automated summary unavailable (LLM error). Raw paper list below.",
            "",
            "TOP PAPERS",
            "----------",
        ]
        for i, paper in enumerate(top_papers, 1):
            fallback_lines.append(
                f"{i}. {paper.get('title', 'N/A')} "
                f"({paper.get('year', 'N/A')}) — {paper.get('url', '')}"
            )
        return "\n".join(fallback_lines)
