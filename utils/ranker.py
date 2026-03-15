"""
utils/ranker.py

Hybrid BM25 + Semantic Similarity ranker for research papers.

Combines sparse keyword retrieval (BM25Okapi) with dense semantic similarity
(Sentence Transformers) to produce a robust relevance score for each paper.
The threshold for filtering is intentionally low (0.05) so that semantically
relevant papers are not discarded due to vocabulary mismatch.
"""

import logging
import re
import string
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
BM25_WEIGHT: float = 0.4
SEMANTIC_WEIGHT: float = 0.6
MIN_RELEVANCE_SCORE: float = 0.05
MAX_PAPERS_RETURNED: int = 20
MIN_PAPERS_THRESHOLD: int = 3
TITLE_KEY: str = "title"
ABSTRACT_KEY: str = "abstract"
RELEVANCE_KEY: str = "relevance_score"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singleton
# ---------------------------------------------------------------------------
_embedding_model: Optional[SentenceTransformer] = None


def _get_embedding_model() -> SentenceTransformer:
    """
    Return the SentenceTransformer model, loading it on the first call.

    Subsequent calls return the cached instance to avoid repeated disk I/O
    and model initialisation overhead.

    Returns
    -------
    SentenceTransformer
        Loaded embedding model ready for encoding.

    Raises
    ------
    RuntimeError
        If the model cannot be loaded.
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Loaded SentenceTransformer model: %s", EMBEDDING_MODEL)
        except Exception as exc:
            logger.error("Failed to load SentenceTransformer '%s': %s", EMBEDDING_MODEL, exc)
            raise RuntimeError(f"Embedding model unavailable: {exc}") from exc
    return _embedding_model


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """
    Tokenise a string for BM25 indexing.

    Lowercases the input, removes punctuation, splits on whitespace, and
    filters empty tokens.

    Parameters
    ----------
    text : str
        Raw text to tokenise.

    Returns
    -------
    list of str
        Clean token list.
    """
    if not isinstance(text, str):
        return []
    try:
        # Lowercase and strip punctuation
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Split and discard empty tokens
        tokens = [t for t in text.split() if t]
        return tokens
    except Exception as exc:
        logger.warning("_tokenize: error tokenising text — %s", exc)
        return []


def _build_corpus_text(paper: Dict[str, Any]) -> str:
    """
    Combine a paper's title and abstract into a single corpus string.

    Parameters
    ----------
    paper : dict
        Paper metadata dict with optional ``title`` and ``abstract`` keys.

    Returns
    -------
    str
        Concatenated title and abstract, whitespace-stripped.
    """
    try:
        title: str = str(paper.get(TITLE_KEY, "")).strip()
        abstract: str = str(paper.get(ABSTRACT_KEY, "")).strip()
        return f"{title} {abstract}".strip()
    except Exception as exc:
        logger.warning("_build_corpus_text: error building text — %s", exc)
        return ""


def _compute_bm25_scores(corpus: List[str], query: str) -> np.ndarray:
    """
    Compute normalised BM25 scores for each document in *corpus*.

    Tokenises both the corpus and the query, initialises :class:`BM25Okapi`,
    and returns scores normalised to [0, 1] via :class:`MinMaxScaler`.

    Parameters
    ----------
    corpus : list of str
        Pre-built text strings, one per paper.
    query : str
        User search/research query.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(len(corpus),)`` with scores in [0, 1].
        Returns a zero array if all scores are zero or on any error.
    """
    n: int = len(corpus)
    zero_scores: np.ndarray = np.zeros(n, dtype=float)

    if not corpus or not query.strip():
        return zero_scores

    try:
        tokenized_corpus: List[List[str]] = [_tokenize(text) for text in corpus]
        tokenized_query: List[str] = _tokenize(query)

        if not tokenized_query:
            logger.warning("_compute_bm25_scores: query tokenised to empty list.")
            return zero_scores

        bm25 = BM25Okapi(tokenized_corpus)
        raw_scores: np.ndarray = np.array(bm25.get_scores(tokenized_query), dtype=float)

        # Edge case: all scores are zero — nothing to normalise
        if raw_scores.max() == 0.0:
            logger.debug("_compute_bm25_scores: all BM25 scores are zero.")
            return zero_scores

        # MinMaxScaler expects shape (n, 1)
        scaler = MinMaxScaler()
        normalised: np.ndarray = scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()
        return normalised

    except Exception as exc:
        logger.error("_compute_bm25_scores: failed — %s", exc)
        return zero_scores


def _compute_semantic_scores(corpus: List[str], query: str) -> np.ndarray:
    """
    Compute cosine similarity scores between *query* and each corpus text.

    Uses the lazy-loaded :data:`EMBEDDING_MODEL` SentenceTransformer to
    encode the query and all corpus texts in a single batch, then computes
    pairwise cosine similarity.

    Parameters
    ----------
    corpus : list of str
        Pre-built text strings, one per paper.
    query : str
        User search/research query.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(len(corpus),)`` with cosine similarities
        clipped to [0, 1].  Returns a zero array on any error.
    """
    n: int = len(corpus)
    zero_scores: np.ndarray = np.zeros(n, dtype=float)

    if not corpus or not query.strip():
        return zero_scores

    try:
        model = _get_embedding_model()

        # Encode query and corpus in one call for efficiency
        all_texts: List[str] = [query] + corpus
        embeddings: np.ndarray = model.encode(
            all_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalised → dot product = cosine sim
            show_progress_bar=False,
        )

        query_emb: np.ndarray = embeddings[0]          # shape (dim,)
        corpus_embs: np.ndarray = embeddings[1:]       # shape (n, dim)

        # Cosine similarity = dot product (embeddings are already normalised)
        scores: np.ndarray = corpus_embs @ query_emb   # shape (n,)

        # Clip negatives to 0 (near-orthogonal / irrelevant texts)
        scores = np.clip(scores, 0.0, 1.0)
        return scores

    except Exception as exc:
        logger.error("_compute_semantic_scores: failed — %s", exc)
        return zero_scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_papers(papers: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Rank research papers using a hybrid BM25 + semantic similarity score.

    Scoring formula
    ---------------
    ``final_score = (BM25_WEIGHT × bm25_score) + (SEMANTIC_WEIGHT × semantic_score)``

    Both component scores are normalised to [0, 1] before combining.
    Semantic similarity is weighted more heavily (:data:`SEMANTIC_WEIGHT` = 0.6)
    because meaning matters more than exact keyword overlap for research papers.

    Post-scoring pipeline
    ---------------------
    1. Attach ``relevance_score`` to each paper dict.
    2. Discard papers scoring below :data:`MIN_RELEVANCE_SCORE`.
    3. If fewer than :data:`MIN_PAPERS_THRESHOLD` survive, return ``[]``.
    4. Sort descending and return at most :data:`MAX_PAPERS_RETURNED` papers.

    Parameters
    ----------
    papers : list of dict
        Paper metadata dicts (must contain ``title`` and ``abstract`` keys).
    query : str
        User search/research query.

    Returns
    -------
    list of dict
        Papers enriched with ``relevance_score``, sorted highest-first.
        Returns ``[]`` on complete failure or when too few papers qualify.

    Raises
    ------
    ValueError
        If ``papers`` is not a list.
    """
    if not isinstance(papers, list):
        raise ValueError("`papers` must be a list of dicts.")

    if not papers:
        logger.info("rank_papers: empty paper list received.")
        return []

    if not isinstance(query, str) or not query.strip():
        logger.warning("rank_papers: empty query; cannot rank.")
        return []

    try:
        # ── Filter valid dicts and build corpus ───────────────────────────
        valid_papers: List[Dict[str, Any]] = [p for p in papers if isinstance(p, dict)]
        corpus: List[str] = [_build_corpus_text(p) for p in valid_papers]

        # Remove papers with no usable text
        non_empty_idx = [i for i, text in enumerate(corpus) if text.strip()]
        if not non_empty_idx:
            logger.warning("rank_papers: all papers have empty title and abstract.")
            return []

        valid_papers = [valid_papers[i] for i in non_empty_idx]
        corpus = [corpus[i] for i in non_empty_idx]

        logger.info("rank_papers: ranking %d paper(s) for query '%s'.", len(valid_papers), query)

        # ── Compute component scores ──────────────────────────────────────
        bm25_scores: np.ndarray = _compute_bm25_scores(corpus, query)
        semantic_scores: np.ndarray = _compute_semantic_scores(corpus, query)

        # ── Hybrid combination ────────────────────────────────────────────
        final_scores: np.ndarray = (
            BM25_WEIGHT * bm25_scores + SEMANTIC_WEIGHT * semantic_scores
        )

        # ── Attach scores and filter ──────────────────────────────────────
        scored_papers: List[Dict[str, Any]] = []
        for paper, score in zip(valid_papers, final_scores):
            enriched = {**paper, RELEVANCE_KEY: round(float(score), 6)}
            if score >= MIN_RELEVANCE_SCORE:
                scored_papers.append(enriched)

        logger.info(
            "rank_papers: %d paper(s) passed the %.2f threshold (from %d).",
            len(scored_papers), MIN_RELEVANCE_SCORE, len(valid_papers),
        )

        # ── Minimum viable result check ───────────────────────────────────
        if len(scored_papers) < MIN_PAPERS_THRESHOLD:
            logger.warning(
                "rank_papers: only %d paper(s) met threshold (minimum %d); returning [].",
                len(scored_papers), MIN_PAPERS_THRESHOLD,
            )
            return []

        # ── Sort and truncate ─────────────────────────────────────────────
        scored_papers.sort(key=lambda p: p[RELEVANCE_KEY], reverse=True)
        result = scored_papers[:MAX_PAPERS_RETURNED]

        logger.info("rank_papers: returning %d ranked paper(s).", len(result))
        return result

    except Exception as exc:
        logger.error("rank_papers: unexpected error — %s", exc)
        return []
