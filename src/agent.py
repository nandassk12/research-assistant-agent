"""
src/agent.py

Main orchestrator for the Personal Research Assistant Agent.

Coordinates all components — preprocessing, fetching, scraping,
deduplication, ranking, vector storage, and LLM summarisation —
to transform a raw user query into a structured research report.
"""

import logging
from typing import Any, Dict, List, Optional

from models.llm import get_llm, validate_keywords, summarize_papers
from utils.input_handler import process_input
from models.embeddings import add_papers_to_store, get_retriever
from utils.preprocessor import preprocess_query
from utils.deduplicator import deduplicate_papers
from utils.ranker import rank_papers
from src.fetcher import fetch_papers
from src.scraper import scrape_papers

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Minimum number of fetched papers before scraper fallback is triggered
MIN_FETCH_RESULTS: int = 5

# Minimum LLM confidence score to trust refined keywords
MIN_KEYWORD_CONFIDENCE: int = 6

# CS / AI / ML topic-type keywords (lowercase for matching)
CS_AI_ML_KEYWORDS: List[str] = [
    "machine learning", "deep learning", "neural network",
    "artificial intelligence", "nlp", "computer vision",
    "reinforcement learning", "transformer", "llm", "bert",
    "classification", "regression", "clustering", "cnn", "rnn",
    "lstm", "gan", "diffusion", "embedding", "vector", "model",
    "dataset", "training", "inference", "pytorch", "tensorflow",
]

# Error message returned when no relevant papers survive ranking
NO_PAPERS_MESSAGE: str = (
    "No relevant papers found. Please try rephrasing your query."
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _determine_topic_type(query: str) -> str:
    """
    Classify a query as computer-science / AI / ML or general research.

    Checks whether any keyword from :data:`CS_AI_ML_KEYWORDS` appears
    (case-insensitive) in the query string.

    Parameters
    ----------
    query : str
        Raw or preprocessed user query.

    Returns
    -------
    str
        ``"cs_ai_ml"`` if a CS/AI/ML keyword is detected, else ``"general"``.
    """
    try:
        lower_query: str = query.lower()
        for keyword in CS_AI_ML_KEYWORDS:
            if keyword in lower_query:
                logger.debug("Topic type 'cs_ai_ml' matched keyword: '%s'", keyword)
                return "cs_ai_ml"
        return "general"
    except Exception as exc:
        logger.warning("_determine_topic_type: unexpected error — %s", exc)
        return "general"


def _build_error_result(query: str, message: str) -> Dict[str, Any]:
    """Return a standardised error result dict."""
    return {
        "success": False,
        "query": query,
        "papers": [],
        "summary": message,
        "paper_count": 0,
        "search_terms": [],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(query: str, max_results: int = 20) -> Dict[str, Any]:
    """
    Execute the full 8-step Research Assistant Agent pipeline.

    Workflow
    --------
    1. **Preprocess** — extract intent & keywords from the raw query.
    2. **Validate keywords** — LLM rates and optionally refines keywords.
    3. **Fetch papers** — topic-aware arXiv / Semantic Scholar fetching;
       falls back to web scraping if results are insufficient.
    4. **Deduplicate** — multi-level duplicate removal.
    5. **Rank** — TF-IDF cosine similarity ranking; returns error if empty.
    6. **Store** — add ranked papers to the ChromaDB vector store.
    7. **Summarise** — LLM generates a structured research report.
    8. **Return** — structured result dict.

    Parameters
    ----------
    query : str
        Raw user research query.
    max_results : int, optional
        Target maximum number of papers to fetch (default 20).

    Returns
    -------
    dict
        Keys:
        * ``success``      – bool
        * ``query``        – str, original query
        * ``papers``       – list of ranked paper dicts
        * ``summary``      – str, LLM-generated research report
        * ``paper_count``  – int, number of papers returned
        * ``search_terms`` – list[str], terms used for searching
    """
    if not isinstance(query, str) or not query.strip():
        logger.warning("run_agent: received empty query.")
        return _build_error_result(query, "Query must be a non-empty string.")

    original_query: str = query.strip()
    logger.info("run_agent: starting pipeline for query: '%s'", original_query)

    # ── STEP 1: Preprocess query ───────────────────────────────────────────
    try:
        preprocessed: Dict[str, Any] = preprocess_query(original_query)
        search_terms: List[str] = preprocessed.get("search_terms") or []
        logger.info("Step 1 complete — %d search term(s) extracted.", len(search_terms))
    except Exception as exc:
        logger.error("Step 1 (preprocess) failed: %s", exc)
        search_terms = []

    # Fallback: use original query if no terms extracted
    effective_search_terms: List[str] = search_terms if search_terms else [original_query]

    # ── STEP 2: Validate keywords with LLM ────────────────────────────────
    try:
        validation: Dict[str, Any] = validate_keywords(
            effective_search_terms, original_query
        )
        confidence: int = validation.get("confidence", 7)
        refined_keywords: List[str] = validation.get("refined_keywords", effective_search_terms)
        logger.info(
            "Step 2 complete — confidence=%d, refined=%s", confidence, refined_keywords
        )

        if confidence >= MIN_KEYWORD_CONFIDENCE:
            # Use LLM-refined keywords joined as a single search string
            search_query: str = " ".join(refined_keywords) if refined_keywords else original_query
            final_search_terms: List[str] = refined_keywords
        else:
            # Low confidence — fall back to original query
            logger.info(
                "Step 2: confidence %d < %d threshold; using original query.",
                confidence, MIN_KEYWORD_CONFIDENCE,
            )
            search_query = original_query
            final_search_terms = [original_query]

    except Exception as exc:
        logger.error("Step 2 (validate keywords) failed: %s", exc)
        search_query = original_query
        final_search_terms = effective_search_terms

    # ── STEP 3: Fetch papers ───────────────────────────────────────────────
    try:
        from src.scraper import enrich_papers_with_fulltext

        topic_type: str = _determine_topic_type(original_query)
        logger.info("Step 3: topic_type='%s', search_query='%s'", topic_type, search_query)

        # Fetch from all 5 API sources
        fetched_papers: List[Dict[str, Any]] = fetch_papers(
            search_query, max_results=max_results
        )
        logger.info("Step 3: fetch_papers returned %d result(s).", len(fetched_papers))

        # Always enrich with full text where available
        try:
            fetched_papers = enrich_papers_with_fulltext(fetched_papers)
        except Exception as enrich_exc:
            logger.warning("Step 3: full-text enrichment failed — %s", enrich_exc)

        # Log source diversity breakdown
        source_counts: Dict[str, int] = {}
        for p in fetched_papers:
            src = str(p.get("source", "unknown"))
            source_counts[src] = source_counts.get(src, 0) + 1
        logger.info("Step 3: source breakdown — %s", source_counts)

        # Only scrape as fallback if truly not enough results
        scraped_papers: List[Dict[str, Any]] = []
        if len(fetched_papers) < MIN_FETCH_RESULTS:
            logger.info(
                "Step 3: fewer than %d results — triggering scraper fallback.",
                MIN_FETCH_RESULTS,
            )
            scraped_papers = scrape_papers(search_query, max_results=max_results)
            fetched_papers += scraped_papers
            logger.info("Step 3: scrape_papers added %d result(s).", len(scraped_papers))

        combined_papers: List[Dict[str, Any]] = fetched_papers
        logger.info("Step 3 complete — combined total: %d paper(s).", len(combined_papers))

    except Exception as exc:
        logger.error("Step 3 (fetch) failed: %s", exc)
        return _build_error_result(
            original_query,
            f"Paper fetching failed: {exc}. Please try again.",
        )

    if not combined_papers:
        logger.warning("Step 3: no papers found for query '%s'.", original_query)
        return _build_error_result(original_query, NO_PAPERS_MESSAGE)

    # ── STEP 4: Deduplicate ────────────────────────────────────────────────
    try:
        deduplicated_papers: List[Dict[str, Any]] = deduplicate_papers(combined_papers)
        logger.info(
            "Step 4 complete — %d paper(s) after deduplication (was %d).",
            len(deduplicated_papers), len(combined_papers),
        )
    except Exception as exc:
        logger.error("Step 4 (deduplicate) failed: %s", exc)
        deduplicated_papers = combined_papers  # continue with duplicates rather than crash

    # ── STEP 5: Rank papers ────────────────────────────────────────────────
    try:
        ranked_papers: List[Dict[str, Any]] = rank_papers(
            deduplicated_papers, original_query
        )
        logger.info("Step 5 complete — %d paper(s) after ranking.", len(ranked_papers))
    except Exception as exc:
        logger.error("Step 5 (rank) failed: %s", exc)
        ranked_papers = []

    if not ranked_papers:
        logger.warning("Step 5: ranking produced no results.")
        return _build_error_result(original_query, NO_PAPERS_MESSAGE)

    # ── STEP 6: Add to vector store ────────────────────────────────────────
    try:
        added_count: int = add_papers_to_store(ranked_papers)
        logger.info("Step 6 complete — %d new paper(s) added to vector store.", added_count)
    except Exception as exc:
        # Non-fatal: log and continue — the agent can still return results
        logger.error("Step 6 (vector store) failed (non-fatal): %s", exc)

    # ── STEP 7: Generate summary ───────────────────────────────────────────
    try:
        summary: str = summarize_papers(ranked_papers, original_query)
        logger.info("Step 7 complete — summary generated (%d chars).", len(summary))
    except Exception as exc:
        logger.error("Step 7 (summarise) failed: %s", exc)
        summary = f"Summary generation failed: {exc}"

    # ── STEP 8: Return result ──────────────────────────────────────────────
    result: Dict[str, Any] = {
        "success": True,
        "query": original_query,
        "papers": ranked_papers,
        "summary": summary,
        "paper_count": len(ranked_papers),
        "search_terms": final_search_terms,
    }

    logger.info(
        "run_agent: pipeline complete — %d paper(s) returned for query '%s'.",
        len(ranked_papers), original_query,
    )
    return result


def run_agent_with_input(
    text: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    file_name: Optional[str] = None,
    max_results: int = 20,
) -> Dict[str, Any]:
    """
    New primary entry point for the Streamlit UI.

    Delegates input normalisation to :func:`utils.input_handler.process_input`,
    then forwards the derived ``search_query`` to the existing
    :func:`run_agent` pipeline.  The returned dict is the standard agent
    result enriched with an ``input_type`` key.

    Parameters
    ----------
    text : str, optional
        Raw text entered by the user (short query, long paragraph, or code).
    file_bytes : bytes, optional
        Uploaded file content (PDF or DOCX).
    file_name : str, optional
        Name of the uploaded file used for type detection.
    max_results : int, optional
        Maximum number of papers to fetch (default 20).

    Returns
    -------
    dict
        Standard :func:`run_agent` result dict plus:
        * ``input_type`` – str, the detected input type.
    """
    try:
        processed: Dict[str, Any] = process_input(
            text=text,
            file_bytes=file_bytes,
            file_name=file_name,
        )

        # Surface any input-processing errors immediately
        if processed.get("error"):
            logger.warning(
                "run_agent_with_input: input processing error — %s",
                processed["error"],
            )
            return {
                "success": False,
                "query": text or file_name or "",
                "papers": [],
                "summary": processed["error"],
                "paper_count": 0,
                "search_terms": [],
                "input_type": processed.get("input_type", "unknown"),
            }

        search_query: str = processed.get("search_query", "").strip()
        if not search_query:
            return {
                "success": False,
                "query": text or file_name or "",
                "papers": [],
                "summary": "Could not derive a search query from the provided input.",
                "paper_count": 0,
                "search_terms": [],
                "input_type": processed.get("input_type", "unknown"),
            }

        logger.info(
            "run_agent_with_input: routing '%s' input to run_agent with query '%s'.",
            processed.get("input_type"), search_query[:80],
        )

        result: Dict[str, Any] = run_agent(search_query, max_results=max_results)
        result["input_type"] = processed.get("input_type", "short_query")
        return result

    except Exception as exc:
        logger.error("run_agent_with_input: unexpected error — %s", exc)
        return {
            "success": False,
            "query": text or file_name or "",
            "papers": [],
            "summary": f"Unexpected error: {exc}",
            "paper_count": 0,
            "search_terms": [],
            "input_type": "unknown",
        }
