"""
src/scraper.py

BeautifulSoup-based web scraper used as a last-resort fallback when the
arXiv and Semantic Scholar APIs are unavailable or return insufficient results.

IMPORTANT
---------
Web scraping is fragile — the target sites may change their HTML structure
at any time.  The scraper is intentionally defensive: every function silently
returns an empty list rather than raising an exception.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# ── arXiv scrape ───────────────────────────────────────────────────────────
ARXIV_SEARCH_URL: str = "https://arxiv.org/search/"
ARXIV_SCRAPE_DELAY: float = 1.5         # seconds between page requests

# ── Semantic Scholar scrape ────────────────────────────────────────────────
SS_SEARCH_URL: str = "https://www.semanticscholar.org/search"
SS_SCRAPE_DELAY: float = 2.0

# ── HTTP settings ────────────────────────────────────────────────────────
REQUEST_TIMEOUT: int = 20               # seconds
REQUEST_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── Parser ───────────────────────────────────────────────────────────────
BS4_PARSER: str = "html.parser"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_text(element: Optional[Any], fallback: str = "") -> str:
    """Extract and clean text from a BeautifulSoup element safely."""
    try:
        return element.get_text(separator=" ", strip=True) if element else fallback
    except Exception:
        return fallback


def _get_page(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[BeautifulSoup]:
    """
    Perform an HTTP GET and return a BeautifulSoup object, or ``None`` on error.

    Parameters
    ----------
    url : str
        Target URL.
    params : dict, optional
        Query-string parameters.

    Returns
    -------
    BeautifulSoup or None
    """
    try:
        response = requests.get(
            url,
            params=params,
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return BeautifulSoup(response.text, BS4_PARSER)
    except requests.exceptions.Timeout:
        logger.warning("_get_page: request timed out for URL '%s'.", url)
    except requests.exceptions.ConnectionError as exc:
        logger.warning("_get_page: connection error for '%s': %s", url, exc)
    except requests.exceptions.HTTPError as exc:
        logger.warning("_get_page: HTTP error for '%s': %s", url, exc)
    except Exception as exc:
        logger.error("_get_page: unexpected error for '%s': %s", url, exc)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_arxiv_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Scrape arXiv search results using BeautifulSoup.

    Targets ``https://arxiv.org/search/`` with ``searchtype=all``.

    Parameters
    ----------
    query : str
        Research topic or keyword string.
    max_results : int, optional
        Maximum number of papers to return (default 10).

    Returns
    -------
    list of dict
        Each dict contains:
        ``title``, ``abstract``, ``authors``, ``year``, ``url``,
        ``source`` (``"arxiv_scrape"``), ``id``.
        Returns an empty list on any failure.
    """
    if not isinstance(query, str) or not query.strip():
        logger.warning("scrape_arxiv_search: empty query received.")
        return []

    params: Dict[str, Any] = {
        "searchtype": "all",
        "query": query.strip(),
        "start": 0,
    }

    logger.info("scrape_arxiv_search: fetching '%s'.", query)
    soup = _get_page(ARXIV_SEARCH_URL, params=params)
    if soup is None:
        return []

    papers: List[Dict[str, Any]] = []

    try:
        # arXiv wraps each result in <li class="arxiv-result">
        results = soup.find_all("li", class_="arxiv-result")

        for item in results[:max_results]:
            try:
                # Title
                title_tag = item.find("p", class_="title")
                title: str = _safe_text(title_tag)

                # Abstract
                abstract_tag = item.find("span", class_="abstract-full")
                if not abstract_tag:
                    abstract_tag = item.find("p", class_="abstract")
                abstract: str = _safe_text(abstract_tag)

                # Authors
                authors_tag = item.find("p", class_="authors")
                authors: List[str] = []
                if authors_tag:
                    authors = [
                        a.get_text(strip=True)
                        for a in authors_tag.find_all("a")
                    ]

                # URL and ID
                link_tag = item.find("p", class_="list-title")
                url: str = ""
                paper_id: str = ""
                if link_tag:
                    a_tag = link_tag.find("a", href=True)
                    if a_tag:
                        url = a_tag["href"].strip()
                        paper_id = url.split("/abs/")[-1] if "/abs/" in url else url

                # Year – from the submission date span
                date_tag = item.find("p", class_="is-size-7")
                year: str = ""
                if date_tag:
                    date_text = date_tag.get_text()
                    # Look for a 4-digit year
                    import re
                    year_match = re.search(r"\b(19|20)\d{2}\b", date_text)
                    if year_match:
                        year = year_match.group()

                if title:
                    papers.append({
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "year": year,
                        "url": url,
                        "source": "arxiv_scrape",
                        "id": paper_id,
                    })

            except Exception as exc:
                logger.debug("scrape_arxiv_search: error parsing result item — %s", exc)
                continue

    except Exception as exc:
        logger.error("scrape_arxiv_search: parsing failed — %s", exc)
        return []

    logger.info("scrape_arxiv_search: scraped %d paper(s).", len(papers))
    time.sleep(ARXIV_SCRAPE_DELAY)
    return papers


def scrape_semantic_scholar(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Scrape Semantic Scholar search results using BeautifulSoup.

    Targets ``https://www.semanticscholar.org/search`` with ``sort=Relevance``.

    .. note::
       Semantic Scholar's search page is heavily JavaScript-rendered.  This
       scraper extracts whatever metadata is present in the initial HTML
       (typically title and partial metadata only).  Use
       :func:`src.fetcher.fetch_semantic_scholar` for richer results.

    Parameters
    ----------
    query : str
        Research topic or keyword string.
    max_results : int, optional
        Maximum number of papers to return (default 10).

    Returns
    -------
    list of dict
        Each dict contains:
        ``title``, ``abstract``, ``authors``, ``year``, ``url``,
        ``source`` (``"semantic_scholar_scrape"``), ``id``.
        Returns an empty list on any failure.
    """
    if not isinstance(query, str) or not query.strip():
        logger.warning("scrape_semantic_scholar: empty query received.")
        return []

    params: Dict[str, Any] = {
        "q": query.strip(),
        "sort": "Relevance",
    }

    logger.info("scrape_semantic_scholar: fetching '%s'.", query)
    soup = _get_page(SS_SEARCH_URL, params=params)
    if soup is None:
        return []

    papers: List[Dict[str, Any]] = []

    try:
        # Semantic Scholar renders result cards – attempt best-effort extraction
        # from whatever static HTML is present
        result_cards = soup.find_all("div", {"data-test-id": "paper-row"})

        if not result_cards:
            # Fallback: try generic article / result container selectors
            result_cards = (
                soup.find_all("article") or
                soup.find_all("div", class_=lambda c: c and "result" in c.lower())
            )

        for card in result_cards[:max_results]:
            try:
                # Title
                title_tag = (
                    card.find("h2") or
                    card.find("h3") or
                    card.find(attrs={"data-test-id": "title"})
                )
                title: str = _safe_text(title_tag)
                if not title:
                    continue

                # Abstract
                abstract_tag = card.find("div", class_=lambda c: c and "abstract" in c.lower())
                abstract: str = _safe_text(abstract_tag)

                # Authors
                authors: List[str] = []
                author_tags = card.find_all("span", class_=lambda c: c and "author" in c.lower())
                if author_tags:
                    authors = [_safe_text(a) for a in author_tags if _safe_text(a)]

                # Year
                import re
                card_text = card.get_text()
                year: str = ""
                year_match = re.search(r"\b(19|20)\d{2}\b", card_text)
                if year_match:
                    year = year_match.group()

                # URL
                url: str = ""
                paper_id: str = ""
                link_tag = card.find("a", href=True)
                if link_tag:
                    href = link_tag["href"].strip()
                    if href.startswith("/"):
                        url = f"https://www.semanticscholar.org{href}"
                    else:
                        url = href
                    paper_id = href.split("/")[-1]

                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": year,
                    "url": url,
                    "source": "semantic_scholar_scrape",
                    "id": paper_id,
                })

            except Exception as exc:
                logger.debug("scrape_semantic_scholar: error parsing card — %s", exc)
                continue

    except Exception as exc:
        logger.error("scrape_semantic_scholar: parsing failed — %s", exc)
        return []

    logger.info("scrape_semantic_scholar: scraped %d paper(s).", len(papers))
    time.sleep(SS_SCRAPE_DELAY)
    return papers


def scrape_papers(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Fallback scraping entry-point for the Research Assistant Agent.

    Tries arXiv scraping first; if it returns no results, falls back to
    Semantic Scholar scraping.  Both sources are combined when results from
    arXiv are present.

    Parameters
    ----------
    query : str
        Research topic or keyword string.
    max_results : int, optional
        Maximum combined results to return (default 10).

    Returns
    -------
    list of dict
        Combined scraped paper list.  Never raises — returns ``[]`` on
        complete failure.
    """
    if not isinstance(query, str) or not query.strip():
        logger.warning("scrape_papers: empty query received.")
        return []

    logger.info("scrape_papers: starting fallback scraping for '%s'.", query)

    arxiv_results: List[Dict[str, Any]] = scrape_arxiv_search(query, max_results=max_results)

    ss_results: List[Dict[str, Any]] = []
    if not arxiv_results:
        logger.info("scrape_papers: arXiv scrape returned nothing; trying Semantic Scholar.")
        ss_results = scrape_semantic_scholar(query, max_results=max_results)
    else:
        # Optionally pull a few extra from SS even when arXiv succeeds
        remaining: int = max(max_results - len(arxiv_results), 0)
        if remaining > 0:
            ss_results = scrape_semantic_scholar(query, max_results=remaining)

    combined: List[Dict[str, Any]] = arxiv_results + ss_results
    logger.info(
        "scrape_papers: total scraped = %d (arXiv=%d, SemanticScholar=%d).",
        len(combined), len(arxiv_results), len(ss_results),
    )
    return combined
