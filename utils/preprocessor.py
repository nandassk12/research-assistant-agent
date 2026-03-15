"""
utils/preprocessor.py

Natural-language query pre-processing for the Personal Research Assistant Agent.

Combines spaCy (NER + noun chunks) for intent extraction and KeyBERT for
keyword extraction, producing a unified structure consumed by downstream
search and retrieval components.
"""

import logging
from typing import Dict, List, Any

# ── spaCy ─────────────────────────────────────────────────────────────────────
import spacy

# ── KeyBERT ───────────────────────────────────────────────────────────────────
from keybert import KeyBERT

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
SPACY_MODEL: str = "en_core_web_sm"

# Entity labels to capture (extend as needed)
ENTITY_LABELS: tuple = ("ORG", "PRODUCT", "GPE", "PERSON", "WORK_OF_ART", "EVENT", "NORP", "FAC")

# KeyBERT settings
KEYPHRASE_NGRAM_RANGE: tuple = (2, 3)
DEFAULT_TOP_N: int = 5
KEYBERT_STOP_WORDS: str = "english"
KEYBERT_DIVERSITY: float = 0.5   # MMR diversity (0 = max relevance, 1 = max diversity)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (avoid repeated model loads)
# ---------------------------------------------------------------------------
_nlp = None
_kw_model = None


def _get_nlp() -> spacy.language.Language:
    """Return the spaCy language model, loading it on first call."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(SPACY_MODEL)
            logger.info("Loaded spaCy model: %s", SPACY_MODEL)
        except OSError as exc:
            logger.error("spaCy model '%s' not found. Run: python -m spacy download %s", SPACY_MODEL, SPACY_MODEL)
            raise RuntimeError(f"spaCy model '{SPACY_MODEL}' unavailable: {exc}") from exc
    return _nlp


def _get_kw_model() -> KeyBERT:
    """Return the KeyBERT model, loading it on first call."""
    global _kw_model
    if _kw_model is None:
        try:
            _kw_model = KeyBERT()
            logger.info("Loaded KeyBERT model.")
        except Exception as exc:
            logger.error("Failed to load KeyBERT: %s", exc)
            raise RuntimeError(f"KeyBERT unavailable: {exc}") from exc
    return _kw_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_intent(query: str) -> Dict[str, Any]:
    """
    Extract semantic intent from a natural-language query using spaCy.

    Parses named entities, noun chunks, and identifies the main subject
    (first noun-chunk or root token fallback).

    Parameters
    ----------
    query : str
        Raw user query string.

    Returns
    -------
    dict
        Keys:
        * ``entities``   – list[str] of named entity texts matching
                           :data:`ENTITY_LABELS`.
        * ``noun_chunks``– list[str] of noun phrases detected by spaCy.
        * ``subject``    – str, the main topic/subject of the query.

    Raises
    ------
    RuntimeError
        If the spaCy model cannot be loaded.
    """
    if not isinstance(query, str) or not query.strip():
        logger.warning("extract_intent received empty or non-string query.")
        return {"entities": [], "noun_chunks": [], "subject": ""}

    try:
        nlp = _get_nlp()
        doc = nlp(query.strip())

        entities: List[str] = [
            ent.text for ent in doc.ents if ent.label_ in ENTITY_LABELS
        ]

        noun_chunks: List[str] = [chunk.text for chunk in doc.noun_chunks]

        # Subject heuristic: first noun-chunk, else root token, else empty string
        if noun_chunks:
            subject: str = noun_chunks[0]
        else:
            root_tokens = [tok for tok in doc if tok.dep_ == "ROOT"]
            subject = root_tokens[0].text if root_tokens else query.strip()

        logger.debug("Intent extracted — entities: %s | noun_chunks: %s | subject: '%s'",
                     entities, noun_chunks, subject)

        return {
            "entities": entities,
            "noun_chunks": noun_chunks,
            "subject": subject,
        }

    except RuntimeError:
        raise
    except Exception as exc:
        logger.error("Unexpected error in extract_intent: %s", exc)
        return {"entities": [], "noun_chunks": [], "subject": ""}


def extract_keywords(query: str, top_n: int = DEFAULT_TOP_N) -> List[str]:
    """
    Extract the most relevant keywords / keyphrases from a query using KeyBERT.

    Uses Maximal Marginal Relevance (MMR) to balance relevance with diversity.

    Parameters
    ----------
    query : str
        Raw user query string.
    top_n : int, optional
        Number of keywords to extract (default: :data:`DEFAULT_TOP_N`).

    Returns
    -------
    list of str
        Extracted keyword strings, ranked by relevance (highest first).

    Raises
    ------
    RuntimeError
        If the KeyBERT model cannot be loaded.
    """
    if not isinstance(query, str) or not query.strip():
        logger.warning("extract_keywords received empty or non-string query.")
        return []

    if not isinstance(top_n, int) or top_n < 1:
        logger.warning("Invalid top_n=%r; defaulting to %d.", top_n, DEFAULT_TOP_N)
        top_n = DEFAULT_TOP_N

    try:
        kw_model = _get_kw_model()
        results = kw_model.extract_keywords(
            query.strip(),
            keyphrase_ngram_range=KEYPHRASE_NGRAM_RANGE,
            stop_words=KEYBERT_STOP_WORDS,
            top_n=top_n,
            use_mmr=True,
            diversity=KEYBERT_DIVERSITY,
        )
        keywords: List[str] = [kw for kw, _score in results]
        logger.debug("Keywords extracted: %s", keywords)
        return keywords

    except RuntimeError:
        raise
    except Exception as exc:
        logger.error("Unexpected error in extract_keywords: %s", exc)
        return []


def preprocess_query(query: str) -> Dict[str, Any]:
    """
    Full preprocessing pipeline: intent extraction + keyword extraction.

    Combines the outputs of :func:`extract_intent` and
    :func:`extract_keywords` into a single structure that can be forwarded
    to the search and retrieval layers.

    Parameters
    ----------
    query : str
        Raw user query string.

    Returns
    -------
    dict
        Keys:
        * ``original_query`` – str, the unmodified input query.
        * ``intent``         – dict (output of :func:`extract_intent`).
        * ``keywords``       – list[str] (output of :func:`extract_keywords`).
        * ``search_terms``   – list[str], deduplicated union of noun_chunks,
                               entities, and keywords for downstream search.
    """
    if not isinstance(query, str) or not query.strip():
        logger.warning("preprocess_query received empty or non-string query.")
        return {
            "original_query": query,
            "intent": {"entities": [], "noun_chunks": [], "subject": ""},
            "keywords": [],
            "search_terms": [],
        }

    try:
        intent: Dict[str, Any] = extract_intent(query)
        keywords: List[str] = extract_keywords(query)

        # Build a deduplicated list of search terms (order preserved)
        seen: set = set()
        search_terms: List[str] = []
        for term in (intent.get("noun_chunks", []) +
                     intent.get("entities", []) +
                     keywords):
            normalised = term.lower().strip()
            if normalised and normalised not in seen:
                seen.add(normalised)
                search_terms.append(term)

        result: Dict[str, Any] = {
            "original_query": query,
            "intent": intent,
            "keywords": keywords,
            "search_terms": search_terms,
        }

        logger.info("Preprocessed query — %d search term(s) generated.", len(search_terms))
        return result

    except Exception as exc:
        logger.error("Unexpected error in preprocess_query: %s", exc)
        return {
            "original_query": query,
            "intent": {"entities": [], "noun_chunks": [], "subject": ""},
            "keywords": [],
            "search_terms": [],
        }
