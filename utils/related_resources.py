"""
utils/related_resources.py

Fetches related datasets from Papers With Code (PWC) and related GitHub
repositories for the Personal Research Assistant Agent.
"""

import logging
import os
import time
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
PWC_BASE_URL: str = "https://paperswithcode.com/api/v1"
GITHUB_API_URL: str = "https://api.github.com/search/repositories"
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
REQUEST_TIMEOUT: int = 10
DELAY: float = 0.5

REQUEST_HEADERS: Dict[str, str] = {
    "User-Agent": "ResearchAssistantAgent/1.0"
}

GITHUB_HEADERS: Dict[str, str] = {
    "User-Agent": "ResearchAssistantAgent/1.0",
    "Accept": "application/vnd.github.v3+json"
}
if GITHUB_TOKEN:
    GITHUB_HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Search Functions
# ---------------------------------------------------------------------------


def search_github_repos(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search GitHub for code repositories related to the query.

    Appends " machine learning" to the query to narrow results, returning
    the most-starred matching repos.

    Parameters
    ----------
    query : str
        The base search string.
    max_results : int, default=5
        Maximum number of repository results to return.

    Returns
    -------
    list of dict
        A list of repository dictionaries with keys:
        ``name``, ``full_name``, ``description``, ``stars``, ``language``,
        ``url``, ``topics``. Returns an empty list `[]` on failure.
    """
    if not query.strip():
        return []

    # LLM keywords are already clean
    # Use directly as GitHub search query
    search_query = query.strip()
    params = {
        "q": search_query,
        "sort": "stars",
        "order": "desc",
        "per_page": max_results,
    }

    try:
        response = requests.get(
            GITHUB_API_URL, params=params, headers=GITHUB_HEADERS, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        items = data.get("items", [])
        repos: List[Dict[str, Any]] = []

        for item in items:
            repos.append({
                "name": str(item.get("name", "")),
                "full_name": str(item.get("full_name", "")),
                "description": str(item.get("description", "")),
                "stars": int(item.get("stargazers_count", 0)),
                "language": str(item.get("language") or "Unknown"),
                "url": str(item.get("html_url", "")),
                "topics": item.get("topics", [])
            })

        logger.info("search_github_repos: retrieved %d repos for '%s'", len(repos), query)
        return repos

    except Exception as exc:
        logger.error("search_github_repos: failed for query '%s' — %s", query, exc)
        return []


def get_related_resources(
    query: str,
    max_repos: int = 5
) -> Dict[str, Any]:
    """
    Fetch related GitHub repositories for a query.

    Parameters
    ----------
    query : str
        The primary research query.
    max_repos : int, default=5
        Maximum number of repositories to retrieve.

    Returns
    -------
    dict
        A dictionary containing:
        - ``repos``: List of repository dicts
        - ``query``: The query used for searches
    """
    logger.info("get_related_resources: fetching resources for '%s'", query)


    # 3. Fetch GitHub repos
    repos = search_github_repos(query, max_results=max_repos)

    total_results = len(repos)
    logger.info("get_related_resources: completed, %d total resources found.", total_results)

    return {
        "repos": repos,
        "query": query
    }
