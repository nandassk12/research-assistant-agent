"""
src/fetcher.py

Fetches research papers from arXiv and Semantic Scholar APIs.

arXiv results use the official ``arxiv`` Python library.
Semantic Scholar results use the public Graph API via ``requests``.
``fetch_papers`` is the primary entry-point for the agent.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import arxiv
import requests

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# ── arXiv ─────────────────────────────────────────────────────────────────
ARXIV_MAX_RETRY: int = 3
ARXIV_RETRY_DELAY: float = 1.0          # seconds between retries
ARXIV_SORT_BY = arxiv.SortCriterion.Relevance

# ── Semantic Scholar ───────────────────────────────────────────────────────
SS_API_URL: str = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_FIELDS: str = "title,abstract,authors,year,externalIds,url"
SS_TIMEOUT: int = 15                     # seconds
SS_INITIAL_BACKOFF: float = 2.0         # seconds for first 429 retry
SS_MAX_RETRY: int = 4
SS_BACKOFF_FACTOR: float = 2.0          # exponential multiplier

# ── Combined fetch thresholds ──────────────────────────────────────────────
ARXIV_MIN_RESULTS_BEFORE_SS: int = 5   # if arXiv returns fewer, also use SS

# ── HTTP headers ───────────────────────────────────────────────────────────
REQUEST_HEADERS: Dict[str, str] = {
    "User-Agent": "ResearchAssistantAgent/1.0 (educational project; python-requests)"
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_str(value: Any, fallback: str = "") -> str:
    """Return ``str(value).strip()`` or *fallback* if value is falsy."""
    try:
        return str(value).strip() if value else fallback
    except Exception:
        return fallback


def _normalise_ss_paper(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a raw Semantic Scholar API paper dict to the canonical paper format.

    Returns ``None`` if the paper has no title (unusable record).
    """
    title: str = _safe_str(raw.get("title"))
    if not title:
        return None

    # Authors – API returns list of {"authorId":…, "name":…}
    authors: List[str] = [
        _safe_str(a.get("name")) for a in (raw.get("authors") or [])
        if a.get("name")
    ]

    # Year
    year: str = _safe_str(raw.get("year"))

    # URL – prefer the provided url field, fall back to DOI or ArXiv ID
    url: str = _safe_str(raw.get("url"))
    if not url:
        ext_ids: dict = raw.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv")
        doi = ext_ids.get("DOI")
        if arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"
        elif doi:
            url = f"https://doi.org/{doi}"

    paper_id: str = _safe_str(raw.get("paperId"))

    return {
        "title": title,
        "abstract": _safe_str(raw.get("abstract")),
        "authors": authors,
        "year": year,
        "url": url,
        "source": "semantic_scholar",
        "id": paper_id,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_arxiv(query: str, max_results: int = 15) -> List[Dict[str, Any]]:
    """
    Fetch research papers from arXiv using the official ``arxiv`` library.

    Retries up to :data:`ARXIV_MAX_RETRY` times on failure, with a
    :data:`ARXIV_RETRY_DELAY`-second pause between attempts.

    Parameters
    ----------
    query : str
        Search query string.
    max_results : int, optional
        Maximum number of results to return (default 15).

    Returns
    -------
    list of dict
        Each dict contains:
        ``title``, ``abstract``, ``authors``, ``year``, ``url``,
        ``source`` (``"arxiv"``), ``id``.
        Returns an empty list on failure.
    """
    if not isinstance(query, str) or not query.strip():
        logger.warning("fetch_arxiv: empty query received.")
        return []

    papers: List[Dict[str, Any]] = []

    for attempt in range(1, ARXIV_MAX_RETRY + 1):
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query.strip(),
                max_results=max_results,
                sort_by=ARXIV_SORT_BY,
            )
            for result in client.results(search):
                papers.append({
                    "title": _safe_str(result.title),
                    "abstract": _safe_str(result.summary),
                    "authors": [_safe_str(a.name) for a in (result.authors or [])],
                    "year": str(result.published.year) if result.published else "",
                    "url": _safe_str(result.entry_id),
                    "source": "arxiv",
                    "id": _safe_str(result.get_short_id()),
                })
            logger.info("fetch_arxiv: retrieved %d paper(s) for query '%s'.", len(papers), query)
            return papers

        except Exception as exc:
            logger.warning(
                "fetch_arxiv attempt %d/%d failed: %s", attempt, ARXIV_MAX_RETRY, exc
            )
            if attempt < ARXIV_MAX_RETRY:
                time.sleep(ARXIV_RETRY_DELAY)

    logger.error("fetch_arxiv: all %d retries exhausted for query '%s'.", ARXIV_MAX_RETRY, query)
    return []


def fetch_semantic_scholar(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch research papers from the Semantic Scholar Graph API.

    Handles HTTP 429 (rate-limit) responses with exponential back-off up to
    :data:`SS_MAX_RETRY` attempts.  Timeout and connection errors are caught
    and logged; an empty list is returned rather than raising.

    Parameters
    ----------
    query : str
        Search query string.
    max_results : int, optional
        Maximum number of results to request from the API (default 10).

    Returns
    -------
    list of dict
        Each dict contains:
        ``title``, ``abstract``, ``authors``, ``year``, ``url``,
        ``source`` (``"semantic_scholar"``), ``id``.
        Returns an empty list on failure.
    """
    if not isinstance(query, str) or not query.strip():
        logger.warning("fetch_semantic_scholar: empty query received.")
        return []

    params: Dict[str, Any] = {
        "query": query.strip(),
        "limit": max_results,
        "fields": SS_FIELDS,
    }

    backoff: float = SS_INITIAL_BACKOFF

    for attempt in range(1, SS_MAX_RETRY + 1):
        try:
            response = requests.get(
                SS_API_URL,
                params=params,
                headers=REQUEST_HEADERS,
                timeout=SS_TIMEOUT,
            )

            if response.status_code == 429:
                logger.warning(
                    "fetch_semantic_scholar: rate-limited (429). "
                    "Waiting %.1fs before retry %d/%d.",
                    backoff, attempt, SS_MAX_RETRY,
                )
                time.sleep(backoff)
                backoff *= SS_BACKOFF_FACTOR
                continue

            response.raise_for_status()
            data: Dict[str, Any] = response.json()
            raw_papers: List[Dict] = data.get("data") or []

            papers: List[Dict[str, Any]] = []
            for raw in raw_papers:
                normalised = _normalise_ss_paper(raw)
                if normalised:
                    papers.append(normalised)

            logger.info(
                "fetch_semantic_scholar: retrieved %d paper(s) for query '%s'.",
                len(papers), query,
            )
            return papers

        except requests.exceptions.Timeout:
            logger.warning(
                "fetch_semantic_scholar attempt %d/%d: request timed out.",
                attempt, SS_MAX_RETRY,
            )
        except requests.exceptions.ConnectionError as exc:
            logger.warning(
                "fetch_semantic_scholar attempt %d/%d: connection error — %s",
                attempt, SS_MAX_RETRY, exc,
            )
        except requests.exceptions.HTTPError as exc:
            logger.error("fetch_semantic_scholar: HTTP error — %s", exc)
            return []
        except Exception as exc:
            logger.error("fetch_semantic_scholar: unexpected error — %s", exc)
            return []

        if attempt < SS_MAX_RETRY:
            time.sleep(backoff)
            backoff *= SS_BACKOFF_FACTOR

    logger.error(
        "fetch_semantic_scholar: all %d retries exhausted for query '%s'.",
        SS_MAX_RETRY, query,
    )
    return []


def fetch_papers(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Primary paper-fetching entry-point for the Research Assistant Agent.

    Strategy
    --------
    1. Fetch from arXiv (``max_results // 2 + 5`` results).
    2. **Always** also fetch from Semantic Scholar (remaining budget).
    3. If arXiv returned fewer than :data:`ARXIV_MIN_RESULTS_BEFORE_SS`
       papers, the Semantic Scholar quota is increased to ``max_results``.
    4. Combine both lists (de-duplication is handled by the caller via
       ``utils.deduplicator``).

    Parameters
    ----------
    query : str
        Search query string.
    max_results : int, optional
        Target combined result count (default 20).

    Returns
    -------
    list of dict
        Combined list of paper dicts from all sources.
        Never raises — returns ``[]`` on complete failure.
    """
    if not isinstance(query, str) or not query.strip():
        logger.warning("fetch_papers: empty query received.")
        return []

    arxiv_limit: int = max(max_results // 2 + 5, 5)
    ss_limit: int = max(max_results // 2, 5)

    logger.info(
        "fetch_papers: querying arXiv (limit=%d) + Semantic Scholar (limit=%d).",
        arxiv_limit, ss_limit,
    )

    arxiv_papers: List[Dict[str, Any]] = fetch_arxiv(query, max_results=arxiv_limit)

    # If arXiv results are thin, use full budget for Semantic Scholar
    if len(arxiv_papers) < ARXIV_MIN_RESULTS_BEFORE_SS:
        ss_limit = max_results
        logger.info(
            "fetch_papers: arXiv returned only %d results; expanding SS limit to %d.",
            len(arxiv_papers), ss_limit,
        )

    ss_papers: List[Dict[str, Any]] = fetch_semantic_scholar(query, max_results=ss_limit)

    combined: List[Dict[str, Any]] = arxiv_papers + ss_papers
    logger.info(
        "fetch_papers: combined total = %d paper(s) "
        "(arXiv=%d, SemanticScholar=%d).",
        len(combined), len(arxiv_papers), len(ss_papers),
    )
    return combined
