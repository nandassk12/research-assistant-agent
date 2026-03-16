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
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
GROQ_MODEL: str = "llama-3.3-70b-versatile"
LLM_TEMPERATURE: float = 0.3
LLM_MAX_TOKENS: int = 4096          # large enough for a full 10-section report
SUMMARIZE_TOP_N: int = 15           # max papers sent to the summariser

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
    Generate a structured, 10-section academic literature-review report.

    Sends up to :data:`SUMMARIZE_TOP_N` papers to Groq and instructs it to
    synthesise insights across all papers rather than summarise them
    individually.

    Parameters
    ----------
    papers : list of dict
        Ranked paper dicts (each with title, abstract, authors, year,
        url, source).
    query : str
        The original user query — embedded in the report header.

    Returns
    -------
    str
        Full structured report.  On LLM failure a plain-text fallback
        listing paper titles and URLs is returned instead.
    """
    if not papers:
        return "No papers available to summarise."

    top_papers: List[Dict[str, Any]] = papers[:SUMMARIZE_TOP_N]
    paper_count: int = len(top_papers)

    # ── Step 1: dynamic metadata ──────────────────────────────────────────────
    years: List[int] = [
        int(p["year"]) for p in papers
        if p.get("year") and str(p.get("year")).isdigit()
    ]
    min_year = min(years) if years else "N/A"
    max_year = max(years) if years else "N/A"
    coverage = f"{min_year} – {max_year}"

    source_map: Dict[str, str] = {
        "arxiv":            "arXiv",
        "semantic_scholar": "Semantic Scholar",
        "openalex":         "OpenAlex",
        "pubmed":           "PubMed",
        "core":             "CORE",
    }
    unique_sources: List[str] = []
    seen: set = set()
    for p in papers:
        src = str(p.get("source", "")).lower()
        for key, label in source_map.items():
            if key in src and label not in seen:
                unique_sources.append(label)
                seen.add(label)
    sources_str = " · ".join(unique_sources) if unique_sources else "arXiv · Semantic Scholar"

    # ── Step 2: format paper list for prompt ──────────────────────────────────
    papers_text: str = ""
    for i, paper in enumerate(top_papers, 1):
        authors = paper.get("authors", [])
        if isinstance(authors, list):
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
        else:
            author_str = str(authors)

        papers_text += f"""
Paper {i}:
Title: {paper.get('title', 'N/A')}
Authors: {author_str}
Year: {paper.get('year', 'N/A')}
URL: {paper.get('url', 'N/A')}
Abstract: {str(paper.get('abstract', ''))[:400]}
---"""

    # ── Step 3: system prompt ─────────────────────────────────────────────────
    system_prompt = """You are a research synthesis assistant.

Your task is NOT to summarize papers individually.
Synthesize insights across all papers and produce
a structured academic literature review.

RULES:
- Write in academic literature-review style
- Identify common themes across papers
- Group methods into approach families with table
- Avoid generic summaries — be specific
- Highlight limitations and research gaps
- Add inline citations like [Author et al., Year]
- If papers disagree explain the disagreement
- Output clean markdown only
- Each section header on its OWN line
- Always use ## for headers
- Always use --- as section dividers
- No icons or emojis anywhere in output"""

    # ── Step 4: user prompt with exact 9-section template ─────────────────
    user_prompt = f"""User Query:
{query}

Retrieved Papers:
{papers_text}

Generate the research summary using EXACTLY this structure:

---
## RESEARCH SUMMARY REPORT

**Query:** {query}
**Sources:** {sources_str}
**Coverage:** {coverage}

---
## 1. OVERVIEW

[2-3 paragraphs about the research problem,
why it matters, overall direction of the field]

---
## 2. FIELD LANDSCAPE

[How the field evolved over time.
Early approaches → modern approaches → emerging trends.
Include inline citations like [Author et al., Year]]

---
## 3. MAIN APPROACH FAMILIES

| Approach | Description | Strengths | Limitations |
|----------|-------------|-----------|-------------|
[Fill with 3-5 approach families from papers]

---
## 4. KEY FINDINGS

- Finding 1 [Citation]
- Finding 2 [Citation]
- Finding 3 [Citation]
- Finding 4 [Citation]
- Finding 5 [Citation]

---
## 5. TOP PAPERS

**Title — Authors (Year)**
Contribution: [main contribution in 1-2 lines]
Link: [url]

[repeat for each top paper]

---
## 6. COMMON CHALLENGES

- **Challenge 1:** explanation
- **Challenge 2:** explanation
- **Challenge 3:** explanation

---
## 7. RESEARCH GAPS

- **Gap 1:** explanation
- **Gap 2:** explanation
- **Gap 3:** explanation

---
## 8. FUTURE RESEARCH DIRECTIONS

- **Direction 1:** explanation
- **Direction 2:** explanation
- **Direction 3:** explanation

---
## 9. KEY TAKEAWAY

[2-3 paragraph synthesis of current state,
what works, what doesn't, what comes next]
"""

    # ── Step 5: call LLM ──────────────────────────────────────────────────────
    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        summary: str = (
            response.content if hasattr(response, "content") else str(response)
        ).strip()

        if not summary:
            raise ValueError("LLM returned an empty response.")

        logger.info("summarize_papers: report generated (%d chars).", len(summary))
        return summary

    except Exception as exc:
        logger.error("summarize_papers: LLM call failed — %s", exc)

        # ── Step 6: plain-text fallback ───────────────────────────────────────
        fallback_lines = [
            "RESEARCH SUMMARY REPORT",
            "========================",
            f"Query: {query}",
            f"Papers Analyzed: {paper_count}",
            f"Sources: {sources_str}",
            f"Coverage: {coverage}",
            "",
            "NOTE: Automated synthesis unavailable (LLM error). Paper list below.",
            "",
            "TOP PAPERS",
            "----------",
        ]
        for i, paper in enumerate(top_papers, 1):
            authors = paper.get("authors", [])
            author_str = (
                ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else "")
                if isinstance(authors, list) else str(authors)
            )
            fallback_lines.append(
                f"{i}. {paper.get('title', 'N/A')} "
                f"— {author_str} ({paper.get('year', 'N/A')}) "
                f"| {paper.get('url', '')}"
            )
        return "\n".join(fallback_lines)
