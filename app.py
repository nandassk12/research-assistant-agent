"""
app.py

Streamlit UI for the Personal Research Assistant Agent.
Phase 8 — Full production UI with Search, Papers, and Chat tabs.
"""

import logging

import streamlit as st

from src.agent import run_agent_with_input
from src.chatbot import get_chat_chain, chat, reset_chat
from utils.exporter import export_to_pdf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Assistant Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — premium dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Base ───────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ─────────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ── Section headings ───────────────────────────────────────────────────── */
h1 { color: #e0e0ff !important; font-weight: 700 !important; }
h2, h3 { color: #c0c0ee !important; font-weight: 600 !important; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122a 0%, #1a1a3e 100%) !important;
    border-right: 1px solid rgba(100,100,255,0.2);
}
[data-testid="stSidebar"] * { color: #d0d0ff !important; }

/* ── Tab bar ────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    color: #a0a0cc !important;
    font-weight: 500;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #ffffff !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 15px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(79,70,229,0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79,70,229,0.6);
}
.stButton > button:active { transform: translateY(0px); }

/* ── Inputs ─────────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(100,100,255,0.3) !important;
    border-radius: 10px !important;
    color: #e0e0ff !important;
    font-size: 15px !important;
    transition: border 0.2s ease;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border: 1px solid rgba(100,100,255,0.8) !important;
    box-shadow: 0 0 0 2px rgba(79,70,229,0.2) !important;
}

/* ── Paper card ─────────────────────────────────────────────────────────── */
.paper-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(100,100,255,0.2);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: all 0.25s ease;
}
.paper-card:hover {
    border-color: rgba(100,100,255,0.5);
    background: rgba(255,255,255,0.07);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(79,70,229,0.15);
}
.paper-title {
    font-size: 17px;
    font-weight: 700;
    color: #b0b8ff;
    margin-bottom: 6px;
}
.paper-meta {
    font-size: 13px;
    color: #8080b0;
    margin-bottom: 10px;
}

/* ── Source badge ───────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-arxiv { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-ss    { background: rgba(34,197,94,0.2);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.badge-scrape{ background: rgba(234,179,8,0.2);  color: #facc15; border: 1px solid rgba(234,179,8,0.3); }
.badge-other { background: rgba(99,102,241,0.2); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.3); }

/* ── Summary box ────────────────────────────────────────────────────────── */
.summary-box {
    background: rgba(79,70,229,0.08);
    border: 1px solid rgba(79,70,229,0.25);
    border-radius: 14px;
    padding: 24px 28px;
    font-size: 14px;
    line-height: 1.8;
    color: #d0d0ee;
    white-space: pre-wrap;
    font-family: 'Inter', sans-serif;
}

/* ── Search term tags ───────────────────────────────────────────────────── */
.tag {
    display: inline-block;
    background: rgba(79,70,229,0.2);
    border: 1px solid rgba(79,70,229,0.4);
    color: #a5b4fc;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 500;
    margin: 3px;
}

/* ── Chat bubbles ───────────────────────────────────────────────────────── */
.chat-user {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 8px 0 8px 20%;
    font-size: 14px;
    line-height: 1.6;
}
.chat-assistant {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(100,100,255,0.2);
    color: #d0d0ee;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 18px;
    margin: 8px 20% 8px 0;
    font-size: 14px;
    line-height: 1.6;
}
.chat-source {
    font-size: 11px;
    color: #6060a0;
    margin-top: 6px;
    padding-left: 4px;
}

/* ── Divider ───────────────────────────────────────────────────────────── */
hr { border-color: rgba(100,100,255,0.15) !important; }

/* ── Download button ────────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #0d9488) !important;
    box-shadow: 0 4px 15px rgba(5,150,105,0.4) !important;
}
.stDownloadButton > button:hover {
    box-shadow: 0 6px 20px rgba(5,150,105,0.6) !important;
}

/* ── Metric cards ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(100,100,255,0.2);
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricValue"] { color: #a5b4fc !important; }

/* ── Expander ───────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 8px !important;
    color: #a0a0cc !important;
}

/* ── Slider ─────────────────────────────────────────────────────────────── */
.stSlider > div > div > div { background: rgba(79,70,229,0.6) !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    defaults = {
        "result": None,
        "chat_chain": None,
        "chat_history": [],
        "search_history": [],
        "processing": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _source_badge(source: str) -> str:
    """Return an HTML badge string for a given source label."""
    s = str(source).lower()
    if "arxiv_scrape" in s:
        return '<span class="badge badge-scrape">arXiv (scraped)</span>'
    if "arxiv" in s:
        return '<span class="badge badge-arxiv">arXiv</span>'
    if "semantic" in s:
        return '<span class="badge badge-ss">Semantic Scholar</span>'
    return f'<span class="badge badge-other">{source}</span>'


def _run_search(query_text: str = None, file_bytes: bytes = None, file_name: str = None) -> None:
    """Execute the agent pipeline and store result in session state."""
    st.session_state.processing = True
    try:
        result = run_agent_with_input(
            text=query_text,
            file_bytes=file_bytes,
            file_name=file_name,
            max_results=st.session_state.get("max_results", 20),
        )
        st.session_state.result = result

        # Save to search history (keep last 5)
        label = query_text or file_name or "Upload"
        if label and label not in [h["label"] for h in st.session_state.search_history]:
            st.session_state.search_history.insert(0, {
                "label": label[:60],
                "text": query_text,
                "file_bytes": file_bytes,
                "file_name": file_name,
            })
            st.session_state.search_history = st.session_state.search_history[:5]

        # Initialise chat chain after a successful search (pass full research context)
        if result.get("success"):
            st.session_state.chat_chain = None   # reset old chain
            st.session_state.chat_history = []
            try:
                st.session_state.chat_chain = get_chat_chain(
                    query=result.get("query", ""),
                    summary=result.get("summary", ""),
                    papers=result.get("papers", []),
                )
            except Exception as exc:
                logger.warning("Chat chain init failed: %s", exc)

    except Exception as exc:
        logger.error("_run_search error: %s", exc)
        st.session_state.result = {
            "success": False,
            "summary": f"An unexpected error occurred: {exc}",
            "papers": [],
            "paper_count": 0,
            "search_terms": [],
            "query": query_text or "",
        }
    finally:
        st.session_state.processing = False


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔬 Research Assistant")
    st.markdown("*Powered by LangChain + Groq LLaMA 3*")
    st.divider()

    # Settings
    st.markdown("### ⚙️ Settings")
    max_results = st.slider("Max Papers to Fetch", 10, 50, 20, step=10)
    st.session_state.max_results = max_results

    st.divider()

    # Search history
    st.markdown("### 🕘 Recent Searches")
    if not st.session_state.search_history:
        st.caption("No searches yet.")
    else:
        for item in st.session_state.search_history:
            if st.button(f"🔄 {item['label']}", key=f"hist_{item['label']}", use_container_width=True):
                with st.spinner("🔍 Re-running search…"):
                    _run_search(
                        query_text=item["text"],
                        file_bytes=item["file_bytes"],
                        file_name=item["file_name"],
                    )
                st.rerun()

    st.divider()

    # About
    st.markdown("### ℹ️ About")
    st.caption("Built with LangChain, ChromaDB, Groq LLaMA 3")
    st.caption("Sources: arXiv + Semantic Scholar")
    st.caption("Hybrid BM25 + Semantic ranking")


# ---------------------------------------------------------------------------
# MAIN AREA — 3 tabs
# ---------------------------------------------------------------------------
tab_search, tab_papers, tab_chat = st.tabs([
    "🔍 Search", "📚 Papers", "💬 Chat"
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — SEARCH
# ═══════════════════════════════════════════════════════════════════════════
with tab_search:
    # Hero header
    st.markdown("""
    <div style='text-align:center; padding: 30px 0 16px;'>
        <h1 style='font-size:2.6rem; background: linear-gradient(135deg,#818cf8,#c084fc);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                   margin-bottom:6px;'>
            🔬 Personal Research Assistant
        </h1>
        <p style='color:#8080b0; font-size:16px; margin:0;'>
            Search academic papers and get AI-powered research summaries
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Input row ──────────────────────────────────────────────────────────
    col_text, col_file = st.columns([3, 2], gap="large")

    with col_text:
        query_text = st.text_area(
            "Research Query",
            placeholder="Enter research topic, paste an abstract, or write a long question…",
            height=120,
            key="query_input",
        )

    with col_file:
        uploaded_file = st.file_uploader(
            "Or upload a document",
            type=["pdf", "docx"],
            help="Upload a PDF or Word document to extract and search its content.",
        )

    # ── Search button ──────────────────────────────────────────────────────
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        search_clicked = st.button(
            "🔍 Search Papers",
            use_container_width=True,
            disabled=st.session_state.processing,
        )

    if search_clicked:
        has_text = query_text and query_text.strip()
        has_file = uploaded_file is not None

        if not has_text and not has_file:
            st.warning("⚠️ Please enter a query or upload a document.")
        else:
            file_bytes = uploaded_file.read() if has_file else None
            file_name  = uploaded_file.name  if has_file else None

            with st.spinner("🔍 Searching academic databases…"):
                _run_search(
                    query_text=query_text if has_text else None,
                    file_bytes=file_bytes,
                    file_name=file_name,
                )
            st.rerun()

    # ── Results ────────────────────────────────────────────────────────────
    result = st.session_state.result

    if result is not None:
        st.divider()

        if not result.get("success", False):
            st.error(f"❌ {result.get('summary', 'Unknown error occurred.')}")

        else:
            # Metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("📄 Papers Found",  result.get("paper_count", 0))
            m2.metric("🔑 Search Terms",  len(result.get("search_terms", [])))
            input_type = result.get("input_type", "short_query")
            m3.metric("📥 Input Type", input_type.replace("_", " ").title())

            st.success(f"✅ Found **{result['paper_count']}** relevant papers")

            # Input type badge (only when not a plain short query)
            if input_type != "short_query":
                st.info(f"📄 Input processed as: **{input_type.replace('_', ' ').title()}**")

            # Search terms as tags
            terms = result.get("search_terms", [])
            if terms:
                st.markdown("**🏷️ Search terms used:**")
                tags_html = " ".join(f'<span class="tag">{t}</span>' for t in terms[:10])
                st.markdown(tags_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Summary
            st.markdown("### 📋 Research Summary Report")
            summary = result.get("summary", "")
            st.markdown(
                f'<div class="summary-box">{summary}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Download PDF
            dl_col, _ = st.columns([1, 3])
            with dl_col:
                try:
                    pdf_bytes = export_to_pdf(
                        summary=summary,
                        query=result.get("query", ""),
                        papers=result.get("papers", []),
                    )
                    if pdf_bytes:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_bytes,
                            file_name="research_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                except Exception as exc:
                    logger.warning("PDF export failed: %s", exc)
                    st.caption("⚠️ PDF export unavailable.")

            # Related searches
            if terms:
                st.markdown("### 🔗 Related Searches")
                rel_cols = st.columns(min(3, len(terms)))
                for i, term in enumerate(terms[:3]):
                    with rel_cols[i]:
                        if st.button(f"🔍 {term}", key=f"rel_{i}", use_container_width=True):
                            with st.spinner(f"Searching for '{term}'…"):
                                _run_search(query_text=term)
                            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — PAPERS
# ═══════════════════════════════════════════════════════════════════════════
with tab_papers:
    result = st.session_state.result

    if result is None or not result.get("success"):
        st.markdown("""
        <div style='text-align:center; padding:60px 0; color:#5050a0;'>
            <div style='font-size:4rem;'>📚</div>
            <h3 style='color:#5050a0;'>No papers yet</h3>
            <p>Run a search in the <b>🔍 Search</b> tab to see papers here.</p>
        </div>""", unsafe_allow_html=True)
    else:
        papers = result.get("papers", [])
        st.markdown(f"### 📚 {len(papers)} Papers Found")

        # Sort options
        sort_col, _ = st.columns([1, 3])
        with sort_col:
            sort_by = st.selectbox(
                "Sort by",
                ["Relevance Score", "Year (Newest)", "Year (Oldest)"],
                key="sort_papers",
            )

        if sort_by == "Year (Newest)":
            papers = sorted(papers, key=lambda p: str(p.get("year", "0")), reverse=True)
        elif sort_by == "Year (Oldest)":
            papers = sorted(papers, key=lambda p: str(p.get("year", "9999")))

        st.divider()

        for i, paper in enumerate(papers, 1):
            title   = paper.get("title", "Untitled")
            authors = paper.get("authors", [])
            year    = paper.get("year", "N/A")
            url     = paper.get("url", "")
            abstract= paper.get("abstract", "No abstract available.")
            source  = paper.get("source", "")
            score   = paper.get("relevance_score", 0.0)

            author_str = (
                ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
                if isinstance(authors, list) else str(authors)
            )

            # Card header
            st.markdown(f"""
            <div class="paper-card">
                <div class="paper-title">#{i} &nbsp; {title}</div>
                <div class="paper-meta">
                    👤 {author_str or "Unknown"}  &nbsp;|&nbsp;
                    📅 {year}  &nbsp;|&nbsp;
                    {_source_badge(source)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Relevance bar + expander sit outside the HTML div
            score_col, link_col = st.columns([4, 1])
            with score_col:
                st.progress(min(float(score), 1.0), text=f"Relevance: {score:.3f}")
            with link_col:
                if url:
                    st.link_button("🔗 Open", url, use_container_width=True)

            with st.expander("📄 Abstract"):
                st.markdown(f"<p style='color:#c0c0e0; line-height:1.7; font-size:14px;'>{abstract}</p>",
                            unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom:6px;'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — CHAT
# ═══════════════════════════════════════════════════════════════════════════
with tab_chat:
    result = st.session_state.result

    if result is None or not result.get("success"):
        st.markdown("""
        <div style='text-align:center; padding:60px 0; color:#5050a0;'>
            <div style='font-size:4rem;'>💬</div>
            <h3 style='color:#5050a0;'>Chat unavailable</h3>
            <p>Run a search first to enable paper chat.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("### 💬 Chat with Your Research Papers")
        st.caption("Ask follow-up questions — answers are grounded in the fetched papers only.")

        # ── Context banner ─────────────────────────────────────────────────
        active_query = result.get("query", "")
        active_count = result.get("paper_count", 0)
        active_terms = result.get("search_terms", [])
        st.markdown(f"""
        <div style='
            background: rgba(79,70,229,0.12);
            border: 1px solid rgba(79,70,229,0.3);
            border-radius: 12px;
            padding: 14px 20px;
            margin-bottom: 16px;
        '>
            <span style='color:#a5b4fc; font-weight:600; font-size:14px;'>🔬 Active Research Session</span><br>
            <span style='color:#8080b0; font-size:13px;'>Query: <b style="color:#c0c0ee;">{active_query[:80]}</b></span>
            &nbsp;|&nbsp;
            <span style='color:#8080b0; font-size:13px;'><b style="color:#c0c0ee;">{active_count}</b> papers loaded</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Clear chat button ──────────────────────────────────────────────
        clear_col, _ = st.columns([1, 4])
        with clear_col:
            if st.button("🗑️ Clear Chat", key="clear_chat", use_container_width=True):
                st.session_state.chat_history = []
                if st.session_state.chat_chain:
                    try:
                        reset_chat(st.session_state.chat_chain)
                    except Exception as exc:
                        logger.warning("reset_chat failed: %s", exc)
                st.rerun()

        st.divider()

        # Chat history display
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div style='text-align:center; padding:30px; color:#5050a0;'>
                    <p>💡 Try asking: "What are the key findings?" or
                    "Which paper should I read first?"</p>
                </div>""", unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(
                            f'<div class="chat-user">🧑 {msg["content"]}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="chat-assistant">🤖 {msg["content"]}</div>',
                            unsafe_allow_html=True,
                        )
                        # Show sources
                        sources = msg.get("sources", [])
                        if sources:
                            src_lines = " &nbsp;|&nbsp; ".join(
                                f'<a href="{s["url"]}" target="_blank" '
                                f'style="color:#6060c0; font-size:11px;">'
                                f'📄 {s["title"][:50]}</a>'
                                for s in sources[:3] if s.get("url")
                            )
                            if src_lines:
                                st.markdown(
                                    f'<div class="chat-source">📎 Sources: {src_lines}</div>',
                                    unsafe_allow_html=True,
                                )
                        # Show suggested papers
                        suggested = msg.get("suggested_papers", [])
                        if suggested:
                            sugg_html = " ".join(
                                f'<span class="tag">📄 {t[:45]}</span>'
                                for t in suggested
                            )
                            st.markdown(
                                f'<div class="chat-source" style="margin-top:6px;">💡 Related: {sugg_html}</div>',
                                unsafe_allow_html=True,
                            )

        st.divider()

        # Chat input
        chat_col, send_col = st.columns([5, 1])
        with chat_col:
            user_input = st.text_input(
                "Your question",
                placeholder="Ask a question about the papers…",
                label_visibility="collapsed",
                key="chat_input",
            )
        with send_col:
            send_clicked = st.button("Send ➤", use_container_width=True, key="send_chat")

        if send_clicked and user_input and user_input.strip():
            chain = st.session_state.chat_chain
            if chain is None:
                st.warning("Chat is not ready. Please run a search first.")
            else:
                # Append user message
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input.strip(),
                })

                with st.spinner("🤔 Thinking…"):
                    try:
                        response = chat(
                            chain,
                            user_input.strip(),
                            papers=result.get("papers", []),
                        )
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response.get("answer", "Sorry, I could not find an answer."),
                            "sources": response.get("sources", []),
                            "suggested_papers": response.get("suggested_papers", []),
                        })
                    except Exception as exc:
                        logger.error("Chat error: %s", exc)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "Sorry, an error occurred. Please try again.",
                            "sources": [],
                            "suggested_papers": [],
                        })
                st.rerun()
