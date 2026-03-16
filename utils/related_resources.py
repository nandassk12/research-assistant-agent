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

def search_datasets(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search PapersWithCode for datasets related to the query.

    Uses the PWC REST API ``/datasets/`` endpoint.

    Parameters
    ----------
    query : str
        The search string for datasets.
    max_results : int, default=5
        Maximum number of dataset results to return.

    Returns
    -------
    list of dict
        A list of dataset dictionaries with keys:
        ``name``, ``full_name``, ``url``, ``description``, ``paper_count``.
        Returns an empty list `[]` on failure.
    """
    if not query.strip():
        return []

    url = f"{PWC_BASE_URL}/datasets/"
    params = {
        "q": query,
        "page": 1,
    }

    try:
        response = requests.get(
            url, params=params, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        datasets: List[Dict[str, Any]] = []

        for item in results:
            if len(datasets) >= max_results:
                break

            name = str(item.get("name", ""))
            if not name:
                continue

            datasets.append({
                "name": name,
                "full_name": str(item.get("full_name", name)),
                "url": f"https://paperswithcode.com/dataset/{name}",
                "description": str(item.get("description", "")),
                "paper_count": int(item.get("num_papers", 0))  # PWC API uses num_papers usually
            })

        logger.info("search_datasets: retrieved %d datasets for '%s'", len(datasets), query)
        return datasets

    except Exception as exc:
        logger.error("search_datasets: failed for query '%s' — %s", query, exc)
        return []


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

    search_query = f"{query} machine learning"
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
    max_datasets: int = 5,
    max_repos: int = 5
) -> Dict[str, Any]:
    """
    Orchestrate fetching of both related datasets and GitHub repositories.

    Parameters
    ----------
    query : str
        The primary research query.
    max_datasets : int, default=5
        Maximum number of datasets to retrieve.
    max_repos : int, default=5
        Maximum number of repositories to retrieve.

    Returns
    -------
    dict
        A dictionary containing:
        - ``datasets``: List of dataset dicts
        - ``repos``: List of repository dicts
        - ``query``: The query used for searches
    """
    logger.info("get_related_resources: fetching resources for '%s'", query)

    # 1. Fetch datasets
    datasets = search_datasets(query, max_results=max_datasets)

    # 2. Prevent rapid API thrashing
    time.sleep(DELAY)

    # 3. Fetch GitHub repos
    repos = search_github_repos(query, max_results=max_repos)

    total_results = len(datasets) + len(repos)
    logger.info("get_related_resources: completed, %d total resources found.", total_results)

    return {
        "datasets": datasets,
        "repos": repos,
        "query": query
    }
