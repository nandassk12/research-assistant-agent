# 🔬 Personal Research Assistant Agent

> An AI-powered academic research pipeline that autonomously fetches, ranks, deduplicates, embeds, and summarizes scientific papers — with a conversational chatbot grounded in retrieved literature.

---

## Overview

This project is a fully autonomous research agent that takes a natural language query (e.g., *"phishing URL detection machine learning"*) and returns a structured, academic-grade research report by:

1. Extracting search intent using NLP and BERT-based keyword models
2. Fetching papers from **5 academic APIs** (arXiv, Semantic Scholar, OpenAlex, PubMed, CORE)
3. Deduplicating results with fuzzy string matching
4. Ranking papers with TF-IDF + Cosine Similarity
5. Enriching papers with full-text content where available
6. Embedding chunks into a persistent ChromaDB vector store
7. Generating a structured research report via Groq LLM (LLaMA-3)
8. Enabling follow-up Q&A through a conversational RAG chatbot

---

## Project Structure

```
research-assistant-agent/
├── src/
│   ├── agent.py              # Orchestrator — rule-based decision logic
│   ├── chatbot.py            # Conversational RAG chatbot with memory
│   ├── fetcher.py            # API fetchers: arXiv, Semantic Scholar, OpenAlex, PubMed, CORE
│   └── scraper.py            # BeautifulSoup scraper + full-text enrichment
├── models/
│   ├── llm.py                # Groq LLM: keyword validation + summarization
│   └── embeddings.py         # Sentence Transformers + ChromaDB + section-aware chunking
├── utils/
│   ├── preprocessor.py       # spaCy NER + KeyBERT keyword extraction
│   ├── ranker.py             # TF-IDF + BM25 hybrid ranking
│   ├── deduplicator.py       # 3-level fuzzy deduplication
│   ├── exporter.py           # PDF report export (fpdf2)
│   ├── input_handler.py      # Multi-format input: text, PDF, DOCX
│   ├── memory.py             # SQLite persistent search memory
│   └── related_resources.py  # GitHub repository suggestions
├── data/
│   └── sample_queries.txt
├── app.py                    # Streamlit UI — 3-column layout
├── .env.example
├── requirements.txt
├── docker-compose.yml
└── Dockerfile
```

---

## Progress & Changelog

This project went from zero to a fully working research agent in **3 days**. Here's how it actually got built.

---

### ✅ Phase 1 — Getting the Core Pipeline Working *(Mar 15, 2026)*

The goal of phase 1 was simple: take a text query in, get ranked papers out. No UI, no fancy features — just prove the pipeline works end to end.

**Started with the skeleton.** Laid out the full folder structure upfront (`src/`, `models/`, `utils/`) so every module had a clear home before any code was written. This made wiring everything together later much cleaner.

**First real decision: how to handle the LLM.** Rather than sending every query straight to Groq, the pipeline was designed to use lightweight local models first. spaCy handles named entity recognition and tokenization, KeyBERT extracts the most semantically relevant keyphrases using BERT embeddings — and only then does Groq get involved to validate whether those keywords actually match what the user meant. This keeps LLM calls to exactly 2 per query (validation + summarization) no matter how complex the pipeline gets.

**Embeddings and vector storage.** Sentence Transformers (`all-MiniLM-L6-v2`) was chosen for encoding because it's fast, runs fully locally, and produces strong semantic vectors for academic text. ChromaDB handles persistent storage — so if you search the same topic twice, the papers are already embedded and retrieval is instant.

**Ranking was always going to be hybrid.** A single TF-IDF score isn't enough for academic queries — technical terms like "YOLO" or "BERT" need exact-match boosting that dense embeddings alone miss. So BM25 (probabilistic term frequency ranking) was added alongside TF-IDF cosine similarity from the start, giving the ranker both lexical precision and semantic coverage.

**Fetching and fallback.** arXiv and Semantic Scholar were the first two sources wired up, with BeautifulSoup scraping as a fallback when API results came back sparse. The initial chatbot used LangChain's `ConversationBufferMemory` to hold conversation history, with responses grounded in the ChromaDB-retrieved chunks.

---

### ✅ Phase 2 — Making It Usable *(Mar 15, 2026)*

Phase 1 proved the pipeline works. Phase 2 was about making it actually usable by a human.

**The summarization prompt needed real work.** The first LLM output was a wall of text. The prompt was completely restructured to force a specific output format — Overview, Key Findings, Top Papers with one-line summaries, Takeaways, and Recommendations. This turned the LLM from a text dumper into something that actually reads like a research brief.

**Users shouldn't need to type queries.** An input handler was added that accepts PDF and DOCX uploads alongside plain text. You can drop in a paper you're reading and the agent will find related work automatically. fpdf2 handles the reverse direction — exporting the generated report as a clean downloadable PDF.

**First Streamlit UI.** The initial frontend was intentionally simple: a search bar, a spinner, a report display, and a chat box below. The focus was on getting the data flow right rather than making it look good. That came in phase 3.

---

### ✅ Phase 3 — Scaling Up & Full Polish *(Mar 17, 2026)*

Phase 3 was the biggest push — expanding from 2 paper sources to 5, adding full-text extraction, rebuilding the chatbot, and completing the Streamlit UI.

**The paper source problem.** Two APIs wasn't enough. A query about medical NLP would return plenty from arXiv but almost nothing useful from Semantic Scholar alone. Three new sources were integrated: **OpenAlex** (open-access filter, returns citation counts and concept tags), **PubMed** (via NCBI E-utilities, parsed from XML — no API key needed), and **CORE** (which uniquely provides actual full-text for many papers directly in the API response). Each source required completely different parsing logic and rate limiting.

**Abstracts aren't enough for good RAG.** This was the insight that drove the biggest architectural change. When the chatbot only has abstracts to work with, answers to questions like "what dataset did they use?" or "what was their training setup?" are basically guesses. Full-text extraction was added at three levels: scraping the HTML version of arXiv papers, downloading open-access PDFs via Unpaywall and extracting text with PyMuPDF, and using CORE's native full-text field. Papers with full text are tagged `content_type: full_text`; others stay as `content_type: abstract`.

**Section-aware chunking changed retrieval quality completely.** Instead of blindly splitting full text every 1000 characters, the pipeline now detects section headings (Introduction, Methodology, Results, Conclusion, etc.) and chunks within each section. Every chunk is prefixed with its section name and stored with section metadata in ChromaDB. When you ask "what methodology did they use?", the retriever finds methodology-section chunks specifically — not random paragraphs from the middle of the paper.

**The chatbot needed a rebuild.** LangChain's chain abstraction was masking errors and making debugging painful. The chatbot was rewritten with direct Groq API calls, giving full control over prompt construction, history injection, and error handling. The behaviour is identical from the user's side but far more reliable under the hood.

**Persistence across sessions.** SQLite was added via `utils/memory.py` so search history survives app restarts. The sidebar loads from `research_memory.db` on startup, and clicking any past query restores the result either from the in-memory session cache (instant) or by re-running the agent (with a spinner).

**The UI became a proper product.** The Streamlit frontend was rebuilt as a 3-column layout: sidebar for search history, main panel for results, right panel for the chat. Results are split across 4 tabs — Relevant Papers (sortable/filterable table with relevance bars), Report (structured summary), Insights (3 Plotly charts: year distribution, source breakdown, top 10 by relevance), and Repositories (auto-fetched GitHub repos related to the query topic).

---

## Architecture & Workflow


## Core ML Techniques & AI Stack

### 1. Large Language Models (LLMs)
- **Groq API (LLaMA-3.3-70B-versatile):** Powers two critical steps — keyword confidence validation (scoring 1-10 how well extracted keywords match user intent) and structured academic report generation. Deliberately limited to exactly 2 LLM calls to keep latency low and costs minimal. All other intelligence is handled by lightweight local models.

### 2. Dense Vector Embeddings & Semantic Search
- **Sentence-Transformers (`all-MiniLM-L6-v2`):** Encodes paper chunks into 384-dimensional dense vectors capturing semantic meaning beyond keyword overlap. Used for RAG retrieval where cosine similarity between the query vector and stored document vectors surfaces the most contextually relevant paper chunks.
- **ChromaDB (persistent):** Local vector database stored in `./chroma_db`. Before embedding, checks if a paper's ID already exists to avoid re-embedding on repeated queries. Retrieves top-5 most relevant chunks (k=5) per query.

### 3. Sparse Retrieval & Lexical Search
- **BM25 (`rank-bm25`):** A probabilistic ranking function (Best Match 25) that scores documents based on term frequency saturation and inverse document frequency. Works especially well for technical queries with specific terminology like model names or dataset names. Runs in parallel with TF-IDF for hybrid ranking.
- **TF-IDF & Cosine Similarity (scikit-learn):** Builds a term-frequency matrix across all fetched paper titles and abstracts, then computes cosine similarity between the query vector and each paper. Papers scoring below 0.1 are discarded as irrelevant. Returns top 20 ranked papers. The hybrid of BM25 + TF-IDF ensures both lexical precision and statistical coverage.

### 4. Keyword Extraction & NLP Processing
- **KeyBERT:** Uses BERT embeddings to extract keyphrases from the user's query by finding n-grams most similar to the query's overall embedding. Produces more semantically meaningful keywords than simple TF-IDF-based extraction. Output feeds directly into API search queries.
- **spaCy (`en_core_web_sm`):** Performs Named Entity Recognition (NER) to identify proper nouns, organizations, and technical terms in the query. Also handles tokenization and part-of-speech tagging for intent classification — lightweight, runs entirely locally with no API calls.

### 5. Section-Aware Chunking (Key Architecture Upgrade)
- For papers with full-text available, the pipeline splits content by detected section headings (Abstract, Introduction, Methodology, Results, Conclusion, etc.) rather than blindly splitting by character count. Each chunk is prefixed with its section name (e.g., `[Section: Methodology]`) and stored with section metadata in ChromaDB. This means when a user asks "what methodology was used?", the retriever can surface methodology-section chunks specifically rather than random text fragments. Full-text chunks use size 1000 with overlap 100; abstract-only uses size 500 with overlap 50.

### 6. Full-Text Extraction Pipeline
- **arXiv HTML scraping:** Fetches the HTML version of arXiv papers (`arxiv.org/html/{id}`), strips navigation and reference sections, and returns up to 15,000 characters of clean body text.
- **Unpaywall API + PyMuPDF (fitz):** For papers with a DOI, queries Unpaywall to find open-access PDF links. Downloads and extracts text from the first 10 pages using PyMuPDF, limited to 5MB and 15,000 chars.
- **CORE native full-text:** CORE API returns `fullText` directly for many papers — stored as-is without scraping.
- This enrichment step upgrades papers from `content_type: abstract` to `content_type: full_text`, substantially improving RAG retrieval quality.

### 7. Fuzzy String Matching & Deduplication
- **Three-level deduplication pipeline:** Level 1 removes exact DOI/ID matches. Level 2 removes exact title string matches. Level 3 uses `fuzzywuzzy` (Levenshtein distance) with a 90% similarity threshold to catch near-duplicate titles like "Detecting Phishing URLs Using ML" vs "Phishing URL Detection with Machine Learning". This is critical because the same paper often appears across arXiv, Semantic Scholar, and OpenAlex with slightly different metadata.

### 8. Retrieval-Augmented Generation (RAG)
- The full pipeline is a RAG architecture: ChromaDB acts as the external knowledge base, Sentence Transformers handle the retrieval encoding, and the Groq LLM generates grounded responses. For the chatbot, each user question triggers a fresh ChromaDB retrieval, and the retrieved chunks are injected into the LLM context window alongside the conversation history — ensuring answers are always grounded in the actual fetched papers, not the LLM's parametric memory.

### 9. Conversational Memory
- **ConversationBufferMemory (LangChain):** Stores the full message history in memory for the current session, allowing the chatbot to handle multi-turn follow-ups like "explain the methodology in paper 2" after previously discussing paper 1. The chatbot was rebuilt with direct LLM calls (bypassing LangChain chain abstraction) for more reliable control over prompt construction and error handling.

### 10. Persistent Search Memory (SQLite)
- Search history is persisted to `research_memory.db` using SQLite via `utils/memory.py`. Unlike in-memory session state, this survives app restarts. The last 20 queries are saved to `search_history.txt` as well. On clicking a history entry, the app first checks the in-memory session cache before re-running the agent pipeline.

---

## Paper Sources

| Source | Method | Notes |
|---|---|---|
| **arXiv** | `arxiv` Python library | CS/AI/ML first; full-text via HTML scraping |
| **Semantic Scholar** | REST API | General academic; abstract + citations |
| **OpenAlex** | REST API | Open access filter; includes citation counts + concepts |
| **PubMed** | NCBI E-utilities (XML) | Life sciences focus; no API key required |
| **CORE** | REST API | Full-text available for many papers; requires free API key |
| **BeautifulSoup** | Web scraping | Fallback when API results < 5 |

---

## Tech Stack

| Category | Tool |
|---|---|
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Framework | LangChain + LangChain-Community |
| Vector DB | ChromaDB (persistent local) |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Keyword Extraction | KeyBERT |
| NLP | spaCy `en_core_web_sm` |
| Sparse Retrieval | BM25 (`rank-bm25`) + TF-IDF (scikit-learn) |
| Deduplication | fuzzywuzzy + python-Levenshtein |
| Full-Text PDF | PyMuPDF (fitz) |
| Frontend | Streamlit + Plotly |
| PDF Export | fpdf2 |
| Persistent Memory | SQLite |
| Environment | python-dotenv |
| Containerization | Docker + Docker Compose |

---

## Environment Setup

```bash
# Clone and enter the project
git clone https://github.com/your-username/research-assistant-agent
cd research-assistant-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Configure environment variables
cp .env.example .env
# Edit .env and add your API keys
```

**.env.example:**
```
GROQ_API_KEY=your_groq_key_here
CORE_API_KEY=your_core_key_here   # Free at https://core.ac.uk/api-keys/register
```

---

## Running the App

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

**Sample queries to try:**
```
phishing URL detection machine learning
data augmentation techniques for imbalanced datasets
transformer models for text classification
federated learning privacy preserving
object detection YOLO real time
```

---

## Docker Deployment

```bash
# Build
docker build -t research-assistant .

# Run
docker run -d \
  --name research-assistant \
  -p 8501:8501 \
  --env-file .env \
  --restart always \
  research-assistant

# Or with Docker Compose
docker-compose up -d
```

App accessible at `http://your-server-ip:8501`

---

## Features

**Search & Fetch**
- Natural language query input or PDF/DOCX document upload
- Parallel fetching from 5 academic APIs with retry + exponential backoff
- Automatic full-text enrichment via arXiv HTML, Unpaywall, and CORE

**Analysis & Ranking**
- Hybrid BM25 + TF-IDF ranking with relevance score threshold
- 3-level fuzzy deduplication across all sources
- Section-aware chunking for full-text papers

**Report & Export**
- Structured research summary: Overview, Key Findings, Top Papers, Takeaways, Recommendations
- Downloadable PDF export
- Related GitHub repositories tab

**Visualizations (Insights Tab)**
- Papers by year distribution (bar chart)
- Source breakdown (donut chart)
- Top 10 papers by relevance score (horizontal bar)

**Chat**
- RAG-grounded conversational chatbot
- Quick-question chips for common follow-ups
- Full conversation history within session

**Persistence**
- SQLite search memory across sessions
- Persistent ChromaDB — no re-embedding for repeated queries
- Search history sidebar with one-click restore

## Known Limitations

- PubMed is limited to biomedical domains; queries on pure CS topics will return few results from this source.
- CORE requires a free API key. Without it the source is skipped silently.
- Full-text extraction is best-effort: not all papers have open-access PDFs, and HTML parsing quality varies by publisher.
- ChromaDB runs locally and is not shared across deployments. On a fresh VPS deploy, the vector store starts empty and is rebuilt on first use.
- Groq rate limits apply on the free tier. High-volume use may hit limits on the summarization call.
- The BeautifulSoup scraper is a last-resort fallback and may break if source sites change their HTML structure.
