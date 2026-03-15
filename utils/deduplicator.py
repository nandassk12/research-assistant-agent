"""
utils/deduplicator.py

Multi-level duplicate detection for research paper lists.

Applies three progressively fuzzier deduplication passes to produce a
clean, unique set of papers before they are stored or presented to the user.
"""

import logging
from typing import List, Dict, Any

from fuzzywuzzy import fuzz

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Minimum fuzzy-title similarity (0–100) to consider two papers duplicates.
FUZZY_TITLE_THRESHOLD: int = 90

# Metadata keys used for duplicate detection
URL_KEY: str = "url"
ID_KEY: str = "id"
TITLE_KEY: str = "title"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deduplicate_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate research papers from a list using three detection levels.

    **Level 1 — Exact URL / ID match**
        Papers sharing the same non-empty ``url`` or ``id`` value are
        collapsed to the first occurrence.

    **Level 2 — Exact title match (case-insensitive)**
        Papers whose normalised titles are identical are collapsed.

    **Level 3 — Fuzzy title match**
        Any remaining pair of papers whose titles score ≥
        :data:`FUZZY_TITLE_THRESHOLD` on ``fuzz.token_sort_ratio`` are
        considered duplicates; the later entry is discarded.

    Parameters
    ----------
    papers : list of dict
        Each dict should contain at least a ``title`` key. ``url`` and
        ``id`` keys are used when present.

    Returns
    -------
    list of dict
        Deduplicated paper list retaining the first occurrence of each unique
        paper.

    Raises
    ------
    ValueError
        If ``papers`` is not a list.
    """
    if not isinstance(papers, list):
        raise ValueError("`papers` must be a list of dicts.")

    original_count: int = len(papers)

    if original_count == 0:
        logger.info("deduplicate_papers: received empty list, nothing to do.")
        return []

    # ── Level 1: Exact URL / ID deduplication ─────────────────────────────
    seen_urls: set = set()
    seen_ids: set = set()
    after_l1: List[Dict[str, Any]] = []

    for paper in papers:
        if not isinstance(paper, dict):
            logger.warning("Skipping non-dict item: %r", paper)
            continue

        url: str = str(paper.get(URL_KEY, "")).strip()
        pid: str = str(paper.get(ID_KEY, "")).strip()

        url_dup = url and url in seen_urls
        id_dup  = pid and pid in seen_ids

        if url_dup or id_dup:
            continue  # duplicate — discard

        if url:
            seen_urls.add(url)
        if pid:
            seen_ids.add(pid)

        after_l1.append(paper)

    removed_l1: int = original_count - len(after_l1)
    logger.info("Level 1 (exact URL/ID): removed %d duplicate(s).", removed_l1)

    # ── Level 2: Exact title deduplication (case-insensitive) ─────────────
    seen_titles: set = set()
    after_l2: List[Dict[str, Any]] = []

    for paper in after_l1:
        normalised_title: str = str(paper.get(TITLE_KEY, "")).strip().lower()

        if normalised_title and normalised_title in seen_titles:
            continue

        if normalised_title:
            seen_titles.add(normalised_title)

        after_l2.append(paper)

    removed_l2: int = len(after_l1) - len(after_l2)
    logger.info("Level 2 (exact title): removed %d duplicate(s).", removed_l2)

    # ── Level 3: Fuzzy title deduplication ────────────────────────────────
    after_l3: List[Dict[str, Any]] = []

    for candidate in after_l2:
        candidate_title: str = str(candidate.get(TITLE_KEY, "")).strip()
        is_duplicate: bool = False

        for accepted in after_l3:
            accepted_title: str = str(accepted.get(TITLE_KEY, "")).strip()

            try:
                score: int = fuzz.token_sort_ratio(candidate_title, accepted_title)
            except Exception as exc:
                logger.warning("fuzz.token_sort_ratio failed: %s", exc)
                score = 0

            if score >= FUZZY_TITLE_THRESHOLD:
                is_duplicate = True
                logger.debug(
                    "Fuzzy dup (score=%d): '%s' ≈ '%s'",
                    score, candidate_title, accepted_title,
                )
                break

        if not is_duplicate:
            after_l3.append(candidate)

    removed_l3: int = len(after_l2) - len(after_l3)
    logger.info("Level 3 (fuzzy title ≥%d%%): removed %d duplicate(s).", FUZZY_TITLE_THRESHOLD, removed_l3)

    total_removed: int = removed_l1 + removed_l2 + removed_l3
    logger.info(
        "Deduplication complete: %d → %d papers (%d duplicate(s) removed).",
        original_count, len(after_l3), total_removed,
    )

    return after_l3
