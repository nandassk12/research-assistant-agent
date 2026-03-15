"""
utils/input_handler.py

Multi-format input detection and pre-processing for the
Personal Research Assistant Agent.

Accepts plain text, long paragraphs, PDF bytes, DOCX bytes, and code
snippets, and converts them all into a clean ``search_query`` string that
the agent pipeline can consume.
"""

import io
import logging
import re
from typing import Any, Dict, List, Optional

import fitz                          # PyMuPDF
import spacy
from docx import Document as DocxDocument
from keybert import KeyBERT

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
SHORT_QUERY_MAX_CHARS: int = 200
KEYBERT_TOP_N: int = 5
KEYBERT_NGRAM_RANGE: tuple = (1, 2)
KEYBERT_STOP_WORDS: str = "english"
SUMMARIZE_MAX_SENTENCES: int = 5
SPACY_MODEL: str = "en_core_web_sm"

# Code-detection indicators (substrings checked against input text)
CODE_INDICATORS: List[str] = [
    "def ", "import ", "class ", "for ", "while ",
    "if __name__", "#", "{", "}", "=>",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_nlp = None
_kw_model = None


def _get_nlp() -> spacy.language.Language:
    """Load and cache the spaCy model."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(SPACY_MODEL)
            logger.info("Loaded spaCy model: %s", SPACY_MODEL)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{SPACY_MODEL}' not found. "
                f"Run: python -m spacy download {SPACY_MODEL}"
            ) from exc
    return _nlp


def _get_kw_model() -> KeyBERT:
    """Load and cache the KeyBERT model."""
    global _kw_model
    if _kw_model is None:
        try:
            _kw_model = KeyBERT()
            logger.info("Loaded KeyBERT model.")
        except Exception as exc:
            raise RuntimeError(f"KeyBERT unavailable: {exc}") from exc
    return _kw_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_input_type(
    text: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    file_name: Optional[str] = None,
) -> str:
    """
    Classify the type of input provided by the user.

    Detection priority
    ------------------
    1. ``file_name`` ends with ``.pdf``  → ``"pdf"``
    2. ``file_name`` ends with ``.docx`` → ``"docx"``
    3. ``text`` contains code indicators → ``"code"``
    4. ``len(text) > 200``               → ``"long_text"``
    5. Otherwise                         → ``"short_query"``

    Parameters
    ----------
    text : str, optional
        Raw string input from the user.
    file_bytes : bytes, optional
        Raw file content (PDF or DOCX).
    file_name : str, optional
        Original filename, used to infer file type.

    Returns
    -------
    str
        One of: ``"pdf"``, ``"docx"``, ``"code"``, ``"long_text"``,
        ``"short_query"``.
    """
    try:
        if file_name:
            name_lower = file_name.strip().lower()
            if name_lower.endswith(".pdf"):
                return "pdf"
            if name_lower.endswith(".docx"):
                return "docx"

        if text:
            for indicator in CODE_INDICATORS:
                if indicator in text:
                    return "code"
            if len(text) > SHORT_QUERY_MAX_CHARS:
                return "long_text"

        return "short_query"

    except Exception as exc:
        logger.warning("detect_input_type: error — %s", exc)
        return "short_query"


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract plain text from a PDF supplied as raw bytes using PyMuPDF.

    Parameters
    ----------
    file_bytes : bytes
        Raw PDF file content.

    Returns
    -------
    str
        Concatenated text from all pages, cleaned of excessive whitespace.
        Returns an empty string on any error.
    """
    if not file_bytes:
        return ""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text: List[str] = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                pages_text.append(page_text.strip())
        doc.close()
        full_text = "\n\n".join(pages_text)
        # Collapse runs of 3+ newlines into two
        full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()
        logger.info("extract_text_from_pdf: extracted %d chars.", len(full_text))
        return full_text
    except Exception as exc:
        logger.error("extract_text_from_pdf: failed — %s", exc)
        return ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract plain text from a DOCX file supplied as raw bytes.

    Parameters
    ----------
    file_bytes : bytes
        Raw DOCX file content.

    Returns
    -------
    str
        Text from all paragraphs joined by newlines.
        Returns an empty string on any error.
    """
    if not file_bytes:
        return ""
    try:
        buffer = io.BytesIO(file_bytes)
        doc = DocxDocument(buffer)
        paragraphs: List[str] = [
            para.text.strip()
            for para in doc.paragraphs
            if para.text.strip()
        ]
        full_text = "\n".join(paragraphs)
        logger.info("extract_text_from_docx: extracted %d chars.", len(full_text))
        return full_text
    except Exception as exc:
        logger.error("extract_text_from_docx: failed — %s", exc)
        return ""


def summarize_long_text(text: str, max_sentences: int = SUMMARIZE_MAX_SENTENCES) -> str:
    """
    Extractive summarisation of a long text using spaCy + KeyBERT.

    Splits the text into sentences with spaCy, then uses KeyBERT to score
    each sentence's relevance and returns the top *max_sentences*.

    Parameters
    ----------
    text : str
        Long input text to summarise.
    max_sentences : int, optional
        Maximum number of sentences to return (default :data:`SUMMARIZE_MAX_SENTENCES`).

    Returns
    -------
    str
        Joined string of the most relevant sentences.
        Falls back to the first *max_sentences* sentences on any error.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    try:
        nlp = _get_nlp()
        doc = nlp(text[:100_000])          # spaCy has a default 1M char limit; guard anyway
        sentences: List[str] = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if not sentences:
            return text[:500]

        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        # Score each sentence using KeyBERT similarity to the full text
        kw_model = _get_kw_model()
        scored: List[tuple] = []
        for sent in sentences:
            try:
                result = kw_model.extract_keywords(
                    sent,
                    keyphrase_ngram_range=(1, 1),
                    stop_words=KEYBERT_STOP_WORDS,
                    top_n=1,
                )
                score = result[0][1] if result else 0.0
            except Exception:
                score = 0.0
            scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_sentences: List[str] = [s for _, s in scored[:max_sentences]]

        # Restore original order by position in sentence list
        order_map = {sent: i for i, sent in enumerate(sentences)}
        top_sentences.sort(key=lambda s: order_map.get(s, 0))

        summary = " ".join(top_sentences)
        logger.info("summarize_long_text: summary is %d chars.", len(summary))
        return summary

    except Exception as exc:
        logger.error("summarize_long_text: failed — %s", exc)
        # Graceful fallback: return first max_sentences-worth of raw text
        try:
            nlp = _get_nlp()
            doc = nlp(text[:10_000])
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            return " ".join(sents[:max_sentences])
        except Exception:
            return text[:500]


def extract_search_query(text: str) -> str:
    """
    Derive a concise search query string from long or code input.

    For **long text / documents**
        Uses KeyBERT to extract the top 5 keyphrases and joins them.

    For **code snippets**
        Extracts import module names, ``def`` function names, and ``class``
        names to build a meaningful programmatic query.

    Parameters
    ----------
    text : str
        Pre-extracted text (from PDF, DOCX, or long input).

    Returns
    -------
    str
        Clean search query string.  Falls back to the first 200 characters
        of the text on any error.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # ── Code path ──────────────────────────────────────────────────────────
    is_code = any(ind in text for ind in CODE_INDICATORS)
    if is_code:
        try:
            terms: List[str] = []

            # import statements → module names
            imports = re.findall(r"^import\s+([\w.]+)", text, re.MULTILINE)
            imports += re.findall(r"^from\s+([\w.]+)\s+import", text, re.MULTILINE)
            terms.extend(imports[:5])

            # def function names
            funcs = re.findall(r"\bdef\s+(\w+)\s*\(", text)
            terms.extend(funcs[:3])

            # class names
            classes = re.findall(r"\bclass\s+(\w+)", text)
            terms.extend(classes[:3])

            if terms:
                query = " ".join(dict.fromkeys(terms))  # deduplicate, preserve order
                logger.info("extract_search_query (code): '%s'", query)
                return query
        except Exception as exc:
            logger.warning("extract_search_query (code): regex failed — %s", exc)

    # ── Long text / document path ──────────────────────────────────────────
    try:
        kw_model = _get_kw_model()
        results = kw_model.extract_keywords(
            text[:5_000],                   # KeyBERT is slow on very long strings
            keyphrase_ngram_range=KEYBERT_NGRAM_RANGE,
            stop_words=KEYBERT_STOP_WORDS,
            top_n=KEYBERT_TOP_N,
        )
        keywords: List[str] = [kw for kw, _ in results]
        query = " ".join(keywords)
        logger.info("extract_search_query (long_text): '%s'", query)
        return query
    except Exception as exc:
        logger.error("extract_search_query: KeyBERT failed — %s", exc)
        return text[:200].strip()


def process_input(
    text: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    file_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry point — detect, extract, and prepare any supported input type.

    Detects the input type, extracts clean text, and derives a search query
    ready to be passed to :func:`src.agent.run_agent`.

    Parameters
    ----------
    text : str, optional
        Raw text entered by the user.
    file_bytes : bytes, optional
        Uploaded file content (PDF or DOCX).
    file_name : str, optional
        Name of the uploaded file (used for type detection).

    Returns
    -------
    dict
        Keys:
        * ``input_type``    – str, detected type.
        * ``original_text`` – str, first 500 chars of the raw input.
        * ``search_query``  – str, the clean query to send to the agent.
        * ``display_text``  – str, human-readable description for the UI.
        * ``error``         – str or None.
    """
    error: Optional[str] = None
    original_text: str = ""
    search_query: str = ""
    display_text: str = ""

    try:
        input_type: str = detect_input_type(
            text=text, file_bytes=file_bytes, file_name=file_name
        )
        logger.info("process_input: detected type '%s'.", input_type)

        # ── Extract raw text from file types ──────────────────────────────
        if input_type == "pdf":
            if not file_bytes:
                return _error_result("pdf", "No PDF bytes provided.")
            raw_text = extract_text_from_pdf(file_bytes)
            if not raw_text:
                return _error_result("pdf", "Could not extract text from the PDF.")
            original_text = raw_text[:500]
            search_query = extract_search_query(raw_text)
            display_text = f"PDF document: {file_name or 'uploaded file'}"

        elif input_type == "docx":
            if not file_bytes:
                return _error_result("docx", "No DOCX bytes provided.")
            raw_text = extract_text_from_docx(file_bytes)
            if not raw_text:
                return _error_result("docx", "Could not extract text from the DOCX file.")
            original_text = raw_text[:500]
            search_query = extract_search_query(raw_text)
            display_text = f"Word document: {file_name or 'uploaded file'}"

        elif input_type == "long_text":
            raw_text = text or ""
            original_text = raw_text[:500]
            search_query = extract_search_query(raw_text)
            display_text = f"Long text input ({len(raw_text)} characters)"

        elif input_type == "code":
            raw_text = text or ""
            original_text = raw_text[:500]
            search_query = extract_search_query(raw_text)
            display_text = "Code snippet"

        else:  # short_query
            raw_text = (text or "").strip()
            original_text = raw_text
            search_query = raw_text
            display_text = raw_text

        if not search_query:
            error = "Could not derive a search query from the provided input."
            logger.warning("process_input: empty search_query for type '%s'.", input_type)

        logger.info(
            "process_input: type='%s', search_query='%s'", input_type, search_query[:80]
        )

        return {
            "input_type": input_type,
            "original_text": original_text,
            "search_query": search_query,
            "display_text": display_text,
            "error": error,
        }

    except Exception as exc:
        logger.error("process_input: unexpected error — %s", exc)
        return _error_result("unknown", str(exc))


def _error_result(input_type: str, message: str) -> Dict[str, Any]:
    """Return a standardised error dict from process_input."""
    return {
        "input_type": input_type,
        "original_text": "",
        "search_query": "",
        "display_text": "",
        "error": message,
    }
