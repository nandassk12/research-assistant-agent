"""
src/fetcher.py

Fetches research papers from arXiv and Semantic Scholar APIs.

arXiv results use the official ``arxiv`` Python library.
Semantic Scholar results use the public Graph API via ``requests``.
``fetch_papers`` is the primary entry-point for the agent.
"""

import logging
import os
import time
import xml.etree.ElementTree as ET
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

# ── OpenAlex ───────────────────────────────────────────────────────────────
OPENALEX_API_URL: str = "https://api.openalex.org/works"
OPENALEX_FIELDS: str = (
    "title,abstract_inverted_index,authorships,"
    "publication_year,doi,primary_location,cited_by_count,concepts"
)
OPENALEX_DELAY: float = 1.0

# ── PubMed ─────────────────────────────────────────────────────────────────
PUBMED_SEARCH_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL: str  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_DELAY: float = 0.5

# ── CORE ───────────────────────────────────────────────────────────────────
CORE_API_URL: str = "https://api.core.ac.uk/v3/search/works"
CORE_API_KEY: str = os.getenv("CORE_API_KEY", "")

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


def fetch_openalex(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch open-access research papers from the OpenAlex API.

    Decodes the ``abstract_inverted_index`` format returned by the API
    into a plain text abstract string.

    Parameters
    ----------
    query : str
        Search query string.
    max_results : int, optional
        Maximum number of results to request (default 10).

    Returns
    -------
    list of dict
        Each dict contains: ``title``, ``abstract``, ``authors``, ``year``,
        ``url``, ``source`` (``"openalex"``), ``id``, ``citations``,
        ``topics``.  Returns ``[]`` on any failure.
    """
    if not isinstance(query, str) or not query.strip():
        logger.warning("fetch_openalex: empty query received.")
        return []

    try:
        params: Dict[str, Any] = {
            "search":   query.strip(),
            "per_page": max_results,
            "filter":   "is_oa:true",
            "select":   OPENALEX_FIELDS,
        }
        resp = requests.get(
            OPENALEX_API_URL,
            params=params,
            headers=REQUEST_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

        papers: List[Dict[str, Any]] = []
        for item in results:
            # ── Title ────────────────────────────────────────────────────────
            title = _safe_str(item.get("title"))
            if not title:
                continue

            # ── Abstract — decode inverted index ──────────────────────────
            abstract = ""
            inv = item.get("abstract_inverted_index") or {}
            if inv:
                try:
                    pos_word: Dict[int, str] = {}
                    for word, positions in inv.items():
                        for pos in positions:
                            pos_word[pos] = word
                    abstract = " ".join(pos_word[i] for i in sorted(pos_word))
                except Exception:
                    abstract = ""

            # ── Authors ─────────────────────────────────────────────────────
            authors: List[str] = []
            for ship in item.get("authorships") or []:
                name = (ship.get("author") or {}).get("display_name", "")
                if name:
                    authors.append(_safe_str(name))

            # ── Year ────────────────────────────────────────────────────────
            year = _safe_str(item.get("publication_year"))

            # ── URL: prefer landing page, fall back to DOI ───────────────
            loc = item.get("primary_location") or {}
            url = _safe_str(loc.get("landing_page_url"))
            if not url:
                doi = _safe_str(item.get("doi"))
                url = f"https://doi.org/{doi}" if doi else ""

            # ── Topics from concepts ────────────────────────────────────────
            topics: List[str] = [
                _safe_str(c.get("display_name"))
                for c in (item.get("concepts") or [])[:5]
                if c.get("display_name")
            ]

            papers.append({
                "title":     title,
                "abstract":  abstract,
                "authors":   authors,
                "year":      year,
                "url":       url,
                "source":    "openalex",
                "id":        _safe_str(item.get("id")),
                "citations": item.get("cited_by_count", 0),
                "topics":    topics,
            })

        time.sleep(OPENALEX_DELAY)
        logger.info("fetch_openalex: retrieved %d paper(s) for query '%s'.", len(papers), query)
        return papers

    except Exception as exc:
        logger.error("fetch_openalex: failed — %s", exc)
        return []


def fetch_pubmed(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch research papers from NCBI PubMed via the Entrez eUtils API.

    Two-step process: search for PMIDs then fetch full XML records.

    Parameters
    ----------
    query : str
        Search query string.
    max_results : int, optional
        Maximum number of results to request (default 10).

    Returns
    -------
    list of dict
        Each dict contains: ``title``, ``abstract``, ``authors``, ``year``,
        ``url``, ``source`` (``"pubmed"``), ``id``.
        Returns ``[]`` on any failure.
    """
    if not isinstance(query, str) or not query.strip():
        logger.warning("fetch_pubmed: empty query received.")
        return []

    try:
        # ── Step 1: search for PubMed IDs ─────────────────────────────────
        search_resp = requests.get(
            PUBMED_SEARCH_URL,
            params={
                "db":      "pubmed",
                "term":    query.strip(),
                "retmax":  max_results,
                "retmode": "json",
            },
            headers=REQUEST_HEADERS,
            timeout=15,
        )
        search_resp.raise_for_status()
        id_list: List[str] = (
            search_resp.json()
            .get("esearchresult", {})
            .get("idlist", [])
        )
        if not id_list:
            logger.info("fetch_pubmed: no IDs found for query '%s'.", query)
            return []

        time.sleep(PUBMED_DELAY)

        # ── Step 2: fetch full XML records ────────────────────────────────
        fetch_resp = requests.get(
            PUBMED_FETCH_URL,
            params={
                "db":      "pubmed",
                "id":      ",".join(id_list),
                "retmode": "xml",
                "rettype": "abstract",
            },
            headers=REQUEST_HEADERS,
            timeout=20,
        )
        fetch_resp.raise_for_status()

        # ── Step 3: parse XML ─────────────────────────────────────────────
        root = ET.fromstring(fetch_resp.content)
        papers: List[Dict[str, Any]] = []

        for article in root.findall(".//PubmedArticle"):
            try:
                med = article.find("MedlineCitation")
                if med is None:
                    continue
                art = med.find("Article")
                if art is None:
                    continue

                # Title
                title_el = art.find("ArticleTitle")
                title = _safe_str(title_el.text if title_el is not None else "")
                if not title:
                    continue

                # Abstract
                abstract_el = art.find(".//AbstractText")
                abstract = _safe_str(abstract_el.text if abstract_el is not None else "")

                # Authors
                authors: List[str] = []
                for au in art.findall(".//Author"):
                    last  = au.findtext("LastName", "").strip()
                    first = au.findtext("ForeName", "").strip()
                    name  = f"{first} {last}".strip()
                    if name:
                        authors.append(name)

                # Year
                year = ""
                pub_date = art.find(".//PubDate")
                if pub_date is not None:
                    year = _safe_str(pub_date.findtext("Year", ""))

                # PMID
                pmid_el = med.find("PMID")
                pmid    = _safe_str(pmid_el.text if pmid_el is not None else "")
                url     = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

                papers.append({
                    "title":    title,
                    "abstract": abstract,
                    "authors":  authors,
                    "year":     year,
                    "url":      url,
                    "source":   "pubmed",
                    "id":       pmid,
                })
            except Exception as parse_exc:
                logger.warning("fetch_pubmed: skipped a record — %s", parse_exc)
                continue

        logger.info("fetch_pubmed: retrieved %d paper(s) for query '%s'.", len(papers), query)
        return papers

    except Exception as exc:
        logger.error("fetch_pubmed: failed — %s", exc)
        return []


def fetch_core(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch open-access papers from the CORE API (v3).

    Requires a ``CORE_API_KEY`` environment variable.  If the key is absent
    a warning is logged and an empty list is returned immediately.

    Parameters
    ----------
    query : str
        Search query string.
    max_results : int, optional
        Maximum number of results to request (default 10).

    Returns
    -------
    list of dict
        Each dict contains: ``title``, ``abstract``, ``authors``, ``year``,
        ``url``, ``source`` (``"core"``), ``id``, ``full_text``.
        Returns ``[]`` on any failure or missing API key.
    """
    if not isinstance(query, str) or not query.strip():
        logger.warning("fetch_core: empty query received.")
        return []

    if not CORE_API_KEY:
        logger.warning(
            "fetch_core: CORE_API_KEY not set — skipping CORE source. "
            "Add CORE_API_KEY to your .env file to enable it."
        )
        return []

    try:
        resp = requests.get(
            CORE_API_URL,
            params={"q": query.strip(), "limit": max_results},
            headers={
                **REQUEST_HEADERS,
                "Authorization": f"Bearer {CORE_API_KEY}",
            },
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

        papers: List[Dict[str, Any]] = []
        for item in results:
            title = _safe_str(item.get("title"))
            if not title:
                continue

            # Authors — CORE returns a list of author name strings
            raw_authors = item.get("authors") or []
            if isinstance(raw_authors, list):
                authors: List[str] = [
                    _safe_str(a.get("name") if isinstance(a, dict) else a)
                    for a in raw_authors if a
                ]
            else:
                authors = []

            papers.append({
                "title":     title,
                "abstract":  _safe_str(item.get("abstract")),
                "authors":   authors,
                "year":      _safe_str(item.get("yearPublished")),
                "url":       _safe_str(item.get("downloadUrl")),
                "source":    "core",
                "id":        _safe_str(item.get("id")),
                "full_text": _safe_str(item.get("fullText")),
            })

        logger.info("fetch_core: retrieved %d paper(s) for query '%s'.", len(papers), query)
        return papers

    except Exception as exc:
        logger.error("fetch_core: failed — %s", exc)
        return []


def fetch_papers(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Primary paper-fetching entry-point for the Research Assistant Agent.

    Queries all five sources in parallel-ish sequence and combines results.
    De-duplication is handled downstream by ``utils.deduplicator``.

    Sources
    -------
    * arXiv          — primary CS/AI/ML source
    * Semantic Scholar — cross-domain citation graph
    * OpenAlex       — open-access metadata
    * PubMed         — biomedical literature
    * CORE           — open-access full-text (requires API key)

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

    arxiv_papers    = fetch_arxiv(query,            max_results // 3 + 5)
    ss_papers       = fetch_semantic_scholar(query, max_results // 4)
    openalex_papers = fetch_openalex(query,         max_results // 4)
    pubmed_papers   = fetch_pubmed(query,           max_results // 5)
    core_papers     = fetch_core(query,             max_results // 5)

    combined: List[Dict[str, Any]] = (
        arxiv_papers + ss_papers + openalex_papers + pubmed_papers + core_papers
    )

    logger.info(
        "fetch_papers: sources — arXiv=%d SS=%d OpenAlex=%d PubMed=%d CORE=%d total=%d",
        len(arxiv_papers), len(ss_papers), len(openalex_papers),
        len(pubmed_papers), len(core_papers), len(combined),
    )
    return combined
