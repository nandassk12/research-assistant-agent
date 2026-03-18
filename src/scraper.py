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
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import fitz  # PyMuPDF
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

# ── Full-text extraction ──────────────────────────────────────────────────
ARXIV_HTML_URL: str = "https://arxiv.org/html/{paper_id}"
UNPAYWALL_URL: str  = "https://api.unpaywall.org/v2/{doi}"
UNPAYWALL_EMAIL: str = "research.agent@example.com"
FULLTEXT_MAX_CHARS: int = 15_000
PDF_MAX_BYTES: int = 5 * 1024 * 1024      # 5 MB
PDF_MAX_PAGES: int = 10
ENRICH_DELAY: float = 1.0                 # seconds between papers

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


# ---------------------------------------------------------------------------
# Full-text extraction helpers
# ---------------------------------------------------------------------------

def scrape_arxiv_fulltext(paper_id: str) -> str:
    """
    Scrape the full text of an arXiv paper from its HTML rendering.

    Fetches ``https://arxiv.org/html/{paper_id}``, extracts the main
    ``<article>`` body, strips navigation/figure/reference noise, and
    returns up to :data:`FULLTEXT_MAX_CHARS` characters of clean prose.

    Parameters
    ----------
    paper_id : str
        arXiv paper ID (e.g. ``"2401.12345"`` or ``"2401.12345v2"``).

    Returns
    -------
    str
        Cleaned full text, or ``""`` on any failure.
    """
    if not paper_id or not str(paper_id).strip():
        return ""
    try:
        url = ARXIV_HTML_URL.format(paper_id=str(paper_id).strip())
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, BS4_PARSER)

        # Target the main article body
        article = soup.find("article")
        if not article:
            logger.debug("scrape_arxiv_fulltext: no <article> tag found for '%s'.", paper_id)
            return ""

        # Remove noisy sub-trees
        for tag in article.find_all(["nav", "header", "footer", "figure"]):
            tag.decompose()

        # Remove the references section (everything from first References heading)
        for heading in article.find_all(re.compile(r"^h[1-6]$")):
            if re.search(r"\bReferences\b", heading.get_text(), re.IGNORECASE):
                for sibling in list(heading.find_next_siblings()):
                    sibling.decompose()
                heading.decompose()
                break

        text = article.get_text(separator=" ", strip=True)
        # Collapse excessive whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        logger.info(
            "scrape_arxiv_fulltext: extracted %d chars for paper '%s'.",
            len(text), paper_id,
        )
        return text[:FULLTEXT_MAX_CHARS]

    except Exception as exc:
        logger.warning("scrape_arxiv_fulltext: failed for '%s' — %s", paper_id, exc)
        return ""


def download_and_extract_pdf(pdf_url: str) -> str:
    """
    Download a PDF from *pdf_url* and extract its text using PyMuPDF.

    Only the first :data:`PDF_MAX_PAGES` pages are processed, and the
    download is aborted if the response exceeds :data:`PDF_MAX_BYTES`.

    Parameters
    ----------
    pdf_url : str
        Direct URL to the PDF file.

    Returns
    -------
    str
        Extracted text (up to :data:`FULLTEXT_MAX_CHARS` chars),
        or ``""`` on any failure.
    """
    if not pdf_url or not str(pdf_url).strip():
        return ""
    try:
        resp = requests.get(
            pdf_url,
            headers=REQUEST_HEADERS,
            timeout=30,
            stream=True,
        )
        resp.raise_for_status()

        # Stream download with size guard
        chunks: List[bytes] = []
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=65_536):
            downloaded += len(chunk)
            if downloaded > PDF_MAX_BYTES:
                logger.warning(
                    "download_and_extract_pdf: PDF exceeds %d MB limit, aborting '%s'.",
                    PDF_MAX_BYTES // (1024 * 1024), pdf_url,
                )
                return ""
            chunks.append(chunk)
        pdf_bytes = b"".join(chunks)

        # Extract text via PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text: List[str] = []
        for page_num in range(min(len(doc), PDF_MAX_PAGES)):
            try:
                pages_text.append(doc[page_num].get_text())
            except Exception:
                continue
        doc.close()

        text = " ".join(pages_text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        logger.info(
            "download_and_extract_pdf: extracted %d chars from '%s'.",
            len(text), pdf_url,
        )
        return text[:FULLTEXT_MAX_CHARS]

    except Exception as exc:
        logger.warning("download_and_extract_pdf: failed for '%s' — %s", pdf_url, exc)
        return ""


def fetch_unpaywall(doi: str) -> str:
    """
    Attempt to retrieve the open-access PDF URL for a DOI via Unpaywall,
    then extract its text.

    Parameters
    ----------
    doi : str
        Digital Object Identifier for the paper (e.g. ``"10.1234/example"``).

    Returns
    -------
    str
        Extracted full text, or ``""`` if no OA PDF is available or on failure.
    """
    if not doi or not str(doi).strip():
        return ""
    try:
        url = UNPAYWALL_URL.format(doi=str(doi).strip())
        resp = requests.get(
            url,
            params={"email": UNPAYWALL_EMAIL},
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        best_oa = data.get("best_oa_location") or {}
        pdf_url = str(best_oa.get("url_for_pdf") or "").strip()

        if not pdf_url:
            logger.debug("fetch_unpaywall: no PDF URL for DOI '%s'.", doi)
            return ""

        logger.info("fetch_unpaywall: found PDF for DOI '%s' at '%s'.", doi, pdf_url)
        return download_and_extract_pdf(pdf_url)

    except Exception as exc:
        logger.warning("fetch_unpaywall: failed for DOI '%s' — %s", doi, exc)
        return ""


def enrich_papers_with_fulltext(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Attempt to attach full text to each paper dict in-place.

    Strategy per paper
    ------------------
    1. **arXiv** papers with an ``id`` → :func:`scrape_arxiv_fulltext`.
    2. Any paper with a ``doi`` field → :func:`fetch_unpaywall` (PDF).
    3. **CORE** papers already containing ``full_text`` → marked directly.

    Papers that cannot be enriched are returned unchanged with
    ``content_type`` set to ``"abstract"``.

    Parameters
    ----------
    papers : list of dict
        Paper dicts from the fetcher / scraper pipeline.

    Returns
    -------
    list of dict
        Same list with ``full_text`` and ``content_type`` fields added.
    """
    total = len(papers)
    enriched_count = 0

    for i, paper in enumerate(papers):
        source = str(paper.get("source", "")).lower()
        got_fulltext = False

        try:
            # ── 1. arXiv HTML full text ───────────────────────────────────────
            if "arxiv" in source and paper.get("id"):
                ft = scrape_arxiv_fulltext(paper["id"])
                if ft:
                    paper["full_text"]    = ft
                    paper["content_type"] = "full_text"
                    got_fulltext = True

            # ── 2. Unpaywall PDF (DOI-based) ────────────────────────────────
            doi = str(paper.get("doi", "")).strip()
            if doi and not got_fulltext:
                ft = fetch_unpaywall(doi)
                if ft:
                    paper["full_text"]    = ft
                    paper["content_type"] = "full_text"
                    got_fulltext = True

            # ── 3. CORE papers may already carry full_text ──────────────────
            if source == "core" and paper.get("full_text"):
                paper["content_type"] = "full_text"
                got_fulltext = True

        except Exception as exc:
            logger.warning("enrich_papers_with_fulltext: error on paper %d — %s", i, exc)

        # Default content type if nothing was found
        if not got_fulltext:
            paper.setdefault("content_type", "abstract")
        else:
            enriched_count += 1

        logger.info(
            "enrich_papers_with_fulltext: enriched %d/%d papers.",
            enriched_count, total,
        )
        if i < total - 1:           # skip delay after last paper
            time.sleep(ENRICH_DELAY)

    logger.info(
        "enrich_papers_with_fulltext: complete — %d/%d papers have full text.",
        enriched_count, total,
    )
    return papers


# ---------------------------------------------------------------------------
# Existing public API (kept intact)
# ---------------------------------------------------------------------------

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

    # Full text enrichment disabled
    # arXiv HTML only exists for recent papers
    # causing 404 errors for older papers
    return combined
