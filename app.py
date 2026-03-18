"""
app.py
Complete rewrite — Personal Research Assistant Agent
Three-column layout: Sidebar + Search/Results (Main) + Chat (Right)
"""

import logging
import os

import streamlit as st

from src.agent import run_agent_with_input
from src.chatbot import get_chat_chain, chat, reset_chat
from utils.exporter import export_to_pdf
from utils.related_resources import get_related_resources
from utils.memory import (
    save_search,
    get_search,
    get_recent_queries,
    clear_history,
    get_db_stats
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config & UI hiding
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "result": None,
    "chat_chain": None,
    "chat_history": [],
    "selected_paper_idx": None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# THREE COLUMN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
sidebar_col, main_col, chat_col = st.columns([1, 3, 2], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with sidebar_col:
    st.markdown("### 🔬 Research Assistant")
    st.divider()
    if st.button(
        "🔍 New Search",
        use_container_width=True,
        type="primary"
    ):
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.selected_paper_idx = None
        st.session_state.chat_chain = None
        st.rerun()
    
    st.divider()
    st.markdown("#### Recent Searches")
    
    recent_queries = get_recent_queries(5)
    
    if not recent_queries:
        st.caption("No searches yet.")
    else:
        for query in recent_queries:
            label = query[:25] + ("..." if len(query) > 25 else "")
            if st.button(
                label,
                key=f"hist_{query}",
                use_container_width=True
            ):
                # Restore full result from SQLite instantly!
                cached_result = get_search(query)
                if cached_result:
                    st.session_state.result = cached_result
                    st.session_state.chat_history = []
                    st.session_state.selected_paper_idx = None
                    # Reinitialize chat with cached context
                    if cached_result.get("success"):
                        try:
                            st.session_state.chat_chain = get_chat_chain(
                                query=cached_result["query"],
                                summary=cached_result["summary"],
                                papers=cached_result["papers"]
                            )
                        except Exception:
                            st.session_state.chat_chain = None
                    st.rerun()
                else:
                    # Fallback re-run if not in DB
                    with st.spinner("Loading..."):
                        new_result = run_agent_with_input(
                            text=query,
                            max_results=30
                        )
                        save_search(query, new_result)
                        st.session_state.result = new_result
                        st.session_state.chat_history = []
                        st.session_state.selected_paper_idx = None
                    st.rerun()
        
        st.divider()
        
        # DB stats
        stats = get_db_stats()
        if stats:
            st.caption(
                f"📊 {stats.get('total_searches', 0)} saved · "
                f"{stats.get('db_size_kb', 0):.1f} KB"
            )
        
        # Clear button
        if st.button(
            "🗑️ Clear History",
            use_container_width=True,
            key="clear_hist"
        ):
            clear_history()
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN SEARCH PANEL
# ══════════════════════════════════════════════════════════════════════════════
with main_col:
    result = st.session_state.result
    
    if result is None:
        st.markdown("# Research Assistant")
        st.caption("AI-powered academic research")
        st.markdown("")
        
        with st.container(border=True):
            query_input = st.text_area(
                "Research Query",
                placeholder="Describe your research topic...",
                height=100,
                label_visibility="hidden"
            )
            
            search_col, upload_col = st.columns([3, 1])
            with search_col:
                search_btn = st.button("Search", type="primary", use_container_width=True)
            with upload_col:
                uploaded_file = st.file_uploader(
                    "Upload",
                    type=["pdf", "docx"],
                    label_visibility="collapsed"
                )
            if uploaded_file:
                st.caption(f"📎 Attached: {uploaded_file.name}")
                
        # On search click
        if search_btn:
            has_input = (query_input and query_input.strip()) or uploaded_file
            if not has_input:
                st.warning("⚠️ Please enter a query or upload a document.")
            else:
                text = query_input.strip() if query_input else None
                file_bytes = uploaded_file.read() if uploaded_file else None
                file_name = uploaded_file.name if uploaded_file else None
                
                with st.spinner("Searching..."):
                    try:
                        res = run_agent_with_input(
                            text=text,
                            file_bytes=file_bytes,
                            file_name=file_name,
                            max_results=30
                        )
                        
                        st.session_state.result = res
                        st.session_state.selected_paper_idx = None
                        st.session_state.chat_history = []
                        
                        # Save to both file and memory cache
                        history_key = file_name if file_name else (text or "")[:40]
                        save_search(history_key, res)
                        
                        if res.get("success"):
                            st.session_state.chat_chain = get_chat_chain(
                                query=res["query"],
                                summary=res["summary"],
                                papers=res["papers"]
                            )
                    except Exception as exc:
                        logger.error("Search failed: %s", exc)
                        st.error("Search failed. Check logs.")
                st.rerun()
    else:
        if result.get("success"):
            # ════════════════════════════
            # RESULT HEADER
            # ════════════════════════════
            title_col, export_col = st.columns([5, 1])

            with title_col:
                st.markdown(f"## {result['query']}")
                terms = result.get('search_terms', [])
                if terms:
                    st.caption(", ".join(terms[:4]))

            with export_col:
                try:
                    pdf_bytes = export_to_pdf(
                        summary=result['summary'],
                        query=result['query'],
                        papers=result['papers']
                    )
                    if pdf_bytes:
                        st.download_button(
                            "Export",
                            pdf_bytes,
                            "report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                except Exception:
                    pass

            st.divider()

            # ════════════════════════════
            # TABS
            # ════════════════════════════
            tab1, tab2, tab3, tab4 = st.tabs([
                "Relevant Papers",
                "Report",
                "Insights",
                "Repositories"
            ])

            # ════════════════════════════
            # TAB 1 — RELEVANT PAPERS
            # ════════════════════════════
            with tab1:
                papers = result['papers']
                
                # Relevance bar chart (mini)
                st.caption(f"RELEVANCE DISTRIBUTION ({len(papers)} PAPERS)")
                
                import plotly.graph_objects as go
                scores = [p.get('relevance_score', 0) for p in papers]
                
                fig = go.Figure(go.Bar(
                    y=scores,
                    marker=dict(
                        color=scores,
                        colorscale=[[0,'#1a472a'],[1,'#2ea043']]
                    ),
                    width=0.8
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=80,
                    margin=dict(l=0,r=0,t=0,b=0),
                    showlegend=False,
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False, showticklabels=False)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # Filter + Sort row
                f1, f2 = st.columns([3, 1])
                with f1:
                    paper_filter = st.text_input(
                        "",
                        placeholder="🔍 Search papers...",
                        label_visibility="collapsed",
                        key="paper_filter"
                    )
                with f2:
                    sort_opt = st.selectbox(
                        "",
                        ["Relevance", "Newest", "Oldest"],
                        label_visibility="collapsed",
                        key="sort_opt"
                    )
                
                # Table header
                h1, h2, h3, h4 = st.columns([0.8, 1, 1, 7])
                for col, label in zip(
                    [h1, h2, h3, h4],
                    ["REL.", "SOURCE", "CIT/YR", "TITLE"]
                ):
                    with col:
                        st.markdown(
                            f"<span style='color:#8b949e; font-size:11px; font-weight:600;'>{label}</span>",
                            unsafe_allow_html=True
                        )
                st.markdown(
                    "<div style='border-bottom: 1px solid #21262d; margin-top: 4px; margin-bottom: 4px;'></div>",
                    unsafe_allow_html=True
                )
                
                # Sort logic
                papers_show = papers.copy()
                if sort_opt == "Newest":
                    papers_show.sort(
                        key=lambda p: int(p.get('year') or 0),
                        reverse=True
                    )
                elif sort_opt == "Oldest":
                    papers_show.sort(
                        key=lambda p: int(p.get('year') or 0)
                    )
                else:
                    papers_show.sort(
                        key=lambda p: p.get('relevance_score', 0),
                        reverse=True
                    )
                
                # Filter
                if paper_filter:
                    papers_show = [
                        p for p in papers_show
                        if paper_filter.lower() in p.get('title','').lower()
                        or paper_filter.lower() in str(p.get('authors','')).lower()
                    ]
                
                # Paper rows
                for idx, paper in enumerate(papers_show):
                    score = paper.get('relevance_score', 0)
                    year = paper.get('year', 'N/A')
                    source = paper.get('source', '')
                    title = paper.get('title', 'Untitled')
                    authors = paper.get('authors', [])
                    url = paper.get('url', '')
                    abstract = paper.get('abstract', '')
                    citations = paper.get('citations', '')
                    
                    if isinstance(authors, list) and authors:
                        author_str = (
                            f"{authors[0]}, …, {authors[-1]}"
                            if len(authors) > 2
                            else ", ".join(authors)
                        )
                    else:
                        author_str = str(authors)

                    c1, c2, c3, c4 = st.columns([0.8, 1, 1, 7])
                    
                    with c1:
                        # Relevance score badge (green)
                        st.markdown(
                            f"<span style='background:#14532d; color:#86efac; "
                            f"padding:2px 7px; border-radius:4px; "
                            f"font-size:11px; font-weight:bold;'>"
                            f"{score:.2f}</span>",
                            unsafe_allow_html=True
                        )
                    
                    with c2:
                        # Source badge
                        source_labels = {
                            'arxiv': ('ARXIV', '#1d4ed8', '#93c5fd'),
                            'arxiv_scrape': ('ARXIV', '#1d4ed8', '#93c5fd'),
                            'semantic_scholar': ('S2', '#7c3aed', '#c4b5fd'),
                            'semantic_scholar_scrape': ('S2', '#7c3aed', '#c4b5fd'),
                            'openalex': ('OA', '#059669', '#6ee7b7'),
                            'pubmed': ('PM', '#dc2626', '#fca5a5'),
                            'core': ('CORE', '#d97706', '#fcd34d'),
                        }
                        src_info = source_labels.get(
                            source.lower(),
                            ('OTHER', '#475569', '#cbd5e1')
                        )
                        src_label, bg_color, text_color = src_info
                        st.markdown(
                            f"<span style='background:{bg_color}; "
                            f"color:{text_color}; "
                            f"padding:2px 7px; border-radius:4px; "
                            f"font-size:11px; font-weight:bold;'>"
                            f"{src_label}</span>",
                            unsafe_allow_html=True
                        )
                    
                    with c3:
                        # Citations or year as fallback
                        display_val = str(citations) if citations else str(year)
                        st.markdown(
                            f"<span style='background:#21262d; color:#8b949e; "
                            f"padding:2px 7px; border-radius:4px; "
                            f"font-size:11px;'>{display_val}</span>",
                            unsafe_allow_html=True
                        )
                    
                    with c4:
                        title_col, link_col = st.columns([10, 1])
                        with title_col:
                            if st.button(
                                title[:70] + ("…" if len(title) > 70 else ""),
                                key=f"paper_{idx}",
                                use_container_width=True
                            ):
                                st.session_state.selected_paper_idx = (
                                    None if st.session_state.selected_paper_idx == idx
                                    else idx
                                )
                                st.rerun()
                            
                            st.markdown(
                                f"<span style='color:#8b949e; font-size:11px;'>"
                                f"{author_str}</span>",
                                unsafe_allow_html=True
                            )
                        with link_col:
                            if url:
                                st.link_button("↗", url)
                    
                    # Abstract panel
                    if st.session_state.selected_paper_idx == idx:
                        with st.container(border=True):
                            st.markdown("**Abstract**")
                            st.markdown(
                                f"<p style='color:#c9d1d9; font-size:13px; "
                                f"line-height:1.6;'>"
                                f"{abstract[:600]}"
                                f"{'…' if len(abstract)>600 else ''}</p>",
                                unsafe_allow_html=True
                            )
                            if url:
                                st.link_button("View Paper →", url)
                    
                    st.markdown(
                        "<hr style='margin:2px 0; border-color:#21262d;'>",
                        unsafe_allow_html=True
                    )

            # ════════════════════════════
            # TAB 2 — REPORT
            # ════════════════════════════
            with tab2:
                summary = result.get('summary', '')
                sections = summary.split('\n## ')
                
                if len(sections) <= 1:
                    with st.container(border=True):
                        st.markdown(summary)
                else:
                    header = sections[0].replace(
                        '## RESEARCH SUMMARY REPORT', ''
                    ).strip()
                    
                    with st.container(border=True):
                        st.markdown("## RESEARCH SUMMARY REPORT")
                        st.markdown(header)
                    
                    st.markdown("")
                    
                    for section in sections[1:]:
                        if not section.strip():
                            continue
                        lines = section.split('\n')
                        section_title = lines[0].strip()
                        section_content = '\n'.join(lines[1:]).strip()
                        if section_title.upper() == "RESEARCH SUMMARY REPORT":
                            st.markdown(f"## {section_title}")
                            st.markdown(section_content)
                        else:
                            st.markdown(f"**{section_title.title()}**")
                            st.markdown(section_content)
                            st.markdown(
                                "<hr style='margin:2px 0; border-color:#21262d;'>",
                                unsafe_allow_html=True
                            )

            # ════════════════════════════
            # TAB 3 — INSIGHTS
            # ════════════════════════════
            with tab3:
                import plotly.graph_objects as go
                from collections import Counter
                
                papers = result['papers']
                
                # ── Chart 1: Year Distribution ──
                st.caption("PAPERS BY YEAR")
                
                year_counts = Counter(
                    str(p.get('year', 'Unknown'))
                    for p in papers
                    if p.get('year')
                )
                years_sorted = sorted(year_counts.keys())
                counts = [year_counts[y] for y in years_sorted]
                
                fig_year = go.Figure(go.Bar(
                    x=years_sorted,
                    y=counts,
                    marker=dict(color='#7c3aed'),
                    text=counts,
                    textposition='outside',
                    textfont=dict(color='#8b949e', size=11)
                ))
                fig_year.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#8b949e', size=11),
                    height=220,
                    margin=dict(l=0, r=0, t=20, b=0),
                    xaxis=dict(
                        gridcolor='#21262d',
                        tickfont=dict(color='#8b949e')
                    ),
                    yaxis=dict(
                        gridcolor='#21262d',
                        tickfont=dict(color='#8b949e')
                    )
                )
                st.plotly_chart(fig_year, use_container_width=True)
                
                st.divider()
                
                # ── Chart 2: Source Breakdown ──
                st.caption("SOURCE BREAKDOWN")
                
                source_labels = {
                    'arxiv': 'arXiv',
                    'semantic_scholar': 'Semantic Scholar',
                    'openalex': 'OpenAlex',
                    'pubmed': 'PubMed',
                    'core': 'CORE',
                    'arxiv_scrape': 'arXiv (scraped)',
                    'semantic_scholar_scrape': 'S2 (scraped)'
                }
                
                src_counts = Counter()
                for p in papers:
                    src = p.get('source', 'unknown')
                    label = source_labels.get(src, src)
                    src_counts[label] += 1
                
                fig_src = go.Figure(go.Pie(
                    labels=list(src_counts.keys()),
                    values=list(src_counts.values()),
                    hole=0.5,
                    marker=dict(
                        colors=[
                            '#7c3aed', '#2ea043', '#1d4ed8',
                            '#dc2626', '#d97706', '#0891b2', '#475569'
                        ]
                    ),
                    textfont=dict(color='#c9d1d9', size=12)
                ))
                fig_src.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#8b949e'),
                    height=280,
                    margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(
                        font=dict(color='#8b949e', size=11),
                        bgcolor='rgba(0,0,0,0)'
                    )
                )
                st.plotly_chart(fig_src, use_container_width=True)
                
                st.divider()
                
                # ── Chart 3: Top Papers by Relevance ──
                st.caption("TOP 10 PAPERS BY RELEVANCE")
                
                top10 = sorted(
                    papers,
                    key=lambda p: p.get('relevance_score', 0),
                    reverse=True
                )[:10]
                
                titles_short = [
                    p.get('title', '')[:35] + '…'
                    if len(p.get('title', '')) > 35
                    else p.get('title', '')
                    for p in top10
                ]
                rel_scores = [p.get('relevance_score', 0) for p in top10]
                
                fig_rel = go.Figure(go.Bar(
                    x=rel_scores,
                    y=titles_short,
                    orientation='h',
                    marker=dict(
                        color=rel_scores,
                        colorscale=[[0, '#1a1f6e'], [1, '#7c3aed']]
                    ),
                    text=[f"{s:.2f}" for s in rel_scores],
                    textposition='outside',
                    textfont=dict(color='#8b949e', size=10)
                ))
                fig_rel.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#8b949e', size=11),
                    height=350,
                    margin=dict(l=0, r=60, t=10, b=0),
                    xaxis=dict(
                        gridcolor='#21262d',
                        range=[0, 1],
                        tickfont=dict(color='#8b949e')
                    ),
                    yaxis=dict(
                        gridcolor='#21262d',
                        tickfont=dict(color='#8b949e', size=10)
                    )
                )
                st.plotly_chart(fig_rel, use_container_width=True)
                
                st.divider()
                
                # ── Stats summary ──
                st.caption("QUICK STATS")
                
                s1, s2, s3, s4 = st.columns(4)
                
                all_years = [
                    int(p.get('year', 0))
                    for p in papers
                    if str(p.get('year', '')).isdigit()
                ]
                
                s1.metric(
                    "Total Papers",
                    len(papers)
                )
                s2.metric(
                    "Avg Relevance",
                    f"{sum(rel_scores)/len(rel_scores):.2f}"
                    if rel_scores else "N/A"
                )
                s3.metric(
                    "Year Range",
                    f"{min(all_years)}–{max(all_years)}"
                    if all_years else "N/A"
                )
                s4.metric(
                    "Sources",
                    len(src_counts)
                )

            # ════════════════════════════
            # TAB 4 — REPOSITORIES
            # ════════════════════════════
            with tab4:
                # Check cache first
                from utils.memory import save_repos, get_repos
                
                with st.spinner("Finding repositories..."):
                    # Use LLM refined search terms for better results
                    search_terms = result.get('search_terms', [])
                    repo_query = (
                        " ".join(search_terms[:2])
                        if search_terms
                        else result.get('query', '')
                    )
                    
                    # Cache by original query
                    cache_key = result.get('query', '')
                    cached_repos = get_repos(cache_key)
                    
                    if cached_repos:
                        repos = cached_repos
                    else:
                        resources = get_related_resources(
                            query=repo_query,
                            max_repos=8
                        )
                        repos = resources.get('repos', [])
                        if repos:
                            save_repos(cache_key, repos)
                
                if repos:
                    st.caption(f"{len(repos)} repositories found")
                    st.divider()
                    
                    for repo in repos:
                        with st.container(border=True):
                            rc1, rc2 = st.columns([5, 1])
                            
                            with rc1:
                                st.markdown(f"**{repo.get('name', 'N/A')}**")
                                
                                if repo.get('description'):
                                    st.markdown(
                                        f"<span style='color:#8b949e; font-size:13px;'>"
                                        f"{repo['description'][:120]}"
                                        f"{'…' if len(repo.get('description',''))>120 else ''}"
                                        f"</span>",
                                        unsafe_allow_html=True
                                    )
                                
                                meta = []
                                if repo.get('stars'):
                                    meta.append(f"⭐ {repo['stars']:,}")
                                if repo.get('language'):
                                    meta.append(f"🔵 {repo['language']}")
                                if repo.get('topics'):
                                    topics = repo['topics'][:3]
                                    meta.append(" · ".join(topics))
                                
                                if meta:
                                    st.caption(" · ".join(meta))
                            
                            with rc2:
                                if repo.get('url'):
                                    st.link_button(
                                        "View →",
                                        repo['url'],
                                        use_container_width=True
                                    )
                else:
                    st.markdown(
                        "<div style='text-align:center; "
                        "padding:40px; color:#8b949e;'>"
                        "No repositories found for this query."
                        "</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.error(f"❌ {result.get('summary', 'Error occurred')}")


# ══════════════════════════════════════════════════════════════════════════════
# CHAT PANEL
# ══════════════════════════════════════════════════════════════════════════════
with chat_col:
    result = st.session_state.result
    
    if result is None or not result.get("success"):
        st.markdown("### Research Chat")
        st.divider()
        st.markdown(
            "<div style='text-align:center; "
            "padding:60px 20px; color:#8b949e;'>"
            "🔍 Search for papers<br>to start chatting"
            "</div>",
            unsafe_allow_html=True
        )
    
    else:
        
        # ── Header ──
        ch1, ch2 = st.columns([4, 1])
        with ch1:
            st.markdown("### Research Chat")
        with ch2:
            if st.button(
                "Clear",
                key="clear_chat",
                use_container_width=True
            ):
                if st.session_state.chat_chain:
                    st.session_state.chat_chain = reset_chat(
                        st.session_state.chat_chain
                    )
                st.session_state.chat_history = []
                st.rerun()
        
        # ── Context banner ──
        with st.container(border=True):
            st.markdown(
                f"**{result.get('query','')[:50]}"
                f"{'…' if len(result.get('query',''))>50 else ''}**"
            )
            st.caption(
                f"{result.get('paper_count', 0)} papers in context"
            )
        
        st.divider()
        
        # ── Quick chips ──
        st.caption("QUICK QUESTIONS")
        
        chips = [
            "Which paper should I read first?",
            "What methods were used?",
            "What are the key limitations?",
            "Compare the approaches",
            "What datasets were used?"
        ]
        
        for i, chip in enumerate(chips):
            if st.button(
                chip,
                key=f"chip_{i}",
                use_container_width=True
            ):
                chain = st.session_state.chat_chain
                if chain:
                    try:
                        with st.spinner("Thinking..."):
                            response = chat(chain, chip)
                        st.session_state.chat_chain = response["context"]
                        st.session_state.chat_history.append(
                            {"role": "user", "content": chip}
                        )
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response.get("answer", ""),
                            "sources": response.get("sources", [])
                        })
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Chat error: {exc}")
        
        st.divider()
        
        # ── Chat history ──
        chat_container = st.container(height=380)
        with chat_container:
            
            if not st.session_state.chat_history:
                st.markdown(
                    "<div style='text-align:center; "
                    "padding:40px 20px; color:#8b949e; "
                    "font-size:13px;'>"
                    "Ask a question about the papers above"
                    "</div>",
                    unsafe_allow_html=True
                )
            
            for message in st.session_state.chat_history:
                
                if message["role"] == "user":
                    st.markdown(
                        f'<div style="display:flex; '
                        f'justify-content:flex-end; '
                        f'margin:8px 0;">'
                        f'<div style="background:#7c3aed; '
                        f'color:white; '
                        f'padding:8px 12px; '
                        f'border-radius:16px 16px 4px 16px; '
                        f'max-width:85%; '
                        f'font-size:13px; '
                        f'line-height:1.5;">'
                        f'{message["content"]}'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )
                
                else:
                    st.markdown(
                        f'<div style="display:flex; '
                        f'justify-content:flex-start; '
                        f'margin:8px 0;">'
                        f'<div style="background:#161b22; '
                        f'color:#c9d1d9; '
                        f'padding:8px 12px; '
                        f'border-radius:16px 16px 16px 4px; '
                        f'max-width:85%; '
                        f'font-size:13px; '
                        f'line-height:1.5; '
                        f'border:1px solid #21262d;">'
                        f'{message["content"]}'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )
                    
                    # Sources below assistant message
                    sources = message.get("sources", [])
                    for src in sources[:2]:
                        if src.get("title"):
                            st.markdown(
                                f"<div style='font-size:11px; "
                                f"color:#8b949e; "
                                f"margin:2px 0 6px 4px;'>"
                                f"📎 {src['title'][:45]}"
                                f"{'…' if len(src.get('title',''))>45 else ''}"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                    
                    st.markdown(
                        "<div style='clear:both; margin-bottom:4px;'></div>",
                        unsafe_allow_html=True
                    )
        
        # ── Chat input ──
        user_input = st.chat_input(
            "Ask about the papers...",
            key="chat_input"
        )
        
        if user_input and user_input.strip():
            chain = st.session_state.chat_chain
            if chain:
                try:
                    with st.spinner("Thinking..."):
                        response = chat(chain, user_input.strip())
                    st.session_state.chat_chain = response["context"]
                    st.session_state.chat_history.append(
                        {"role": "user", "content": user_input.strip()}
                    )
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response.get(
                            "answer",
                            "Sorry I could not find an answer."
                        ),
                        "sources": response.get("sources", [])
                    })
                except Exception as exc:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Sorry an error occurred: {exc}",
                        "sources": []
                    })
            else:
                st.warning("Chat not ready. Please search first.")
            st.rerun()
