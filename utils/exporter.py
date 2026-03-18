"""
utils/exporter.py

PDF export utility for the Personal Research Assistant Agent.

Converts research summary reports and paper metadata into a clean,
downloadable PDF using the fpdf2 library.  All text is handled through
fpdf2's built-in core fonts (Helvetica / Courier) to guarantee Unicode
compatibility without requiring external font files.
"""

import io
import logging
from datetime import datetime
from typing import Any, Dict, List

from fpdf import FPDF

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
PDF_TITLE: str = "Research Assistant Agent"
PDF_AUTHOR: str = "Research Assistant Agent"
PDF_FONT_FAMILY: str = "Helvetica"
PDF_MONO_FONT: str = "Courier"
PDF_PAGE_MARGIN: float = 15.0          # mm
PDF_TITLE_FONT_SIZE: int = 18
PDF_SECTION_FONT_SIZE: int = 13
PDF_BODY_FONT_SIZE: int = 10
PDF_SMALL_FONT_SIZE: int = 9
PDF_LINE_HEIGHT: float = 6.0           # mm per line
PDF_SECTION_SPACING: float = 4.0       # mm before section headers
PDF_DATE_FORMAT: str = "%B %d, %Y"    # e.g. "March 15, 2026"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ResearchPDF(FPDF):
    """FPDF subclass with a branded header and automatic page numbering footer."""

    def header(self) -> None:
        self.set_font(PDF_FONT_FAMILY, style="B", size=9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, PDF_TITLE, align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font(PDF_FONT_FAMILY, style="I", size=8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def _safe_text(value: Any, fallback: str = "N/A") -> str:
    """Coerce *value* to a clean ASCII-safe string for PDF rendering."""
    try:
        text = str(value).strip() if value else fallback
        # Replace common unicode dashes/quotes that fpdf core fonts cannot render
        replacements = {
            "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u2022": "*", "\u2026": "...",
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        # Strip any remaining non-latin1 characters (built-in fpdf fonts are latin-1)
        text = text.encode("latin-1", errors="replace").decode("latin-1")
        return text
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _render_table(pdf: FPDF, rows: list, width: float) -> None:
    if not rows:
        return
    
    num_cols = max(len(row) for row in rows)
    if num_cols == 0:
        return
    
    col_width = width / num_cols
    
    for i, row in enumerate(rows):
        is_header = (i == 0)
        
        # Calculate row height based on content
        row_height = 7
        
        x_start = pdf.get_x()
        y_start = pdf.get_y()
        
        for j, cell in enumerate(row):
            # Pad row if fewer cells
            text = cell if j < len(row) else ""
            
            pdf.set_xy(x_start + j * col_width, y_start)
            
            if is_header:
                pdf.set_font("Helvetica", style="B", size=9)
                pdf.set_fill_color(40, 40, 50)
                fill = True
            else:
                pdf.set_font("Helvetica", style="", size=9)
                pdf.set_fill_color(25, 25, 35)
                fill = True
            
            pdf.set_draw_color(70, 70, 80)
            pdf.cell(
                col_width, row_height,
                text[:40] + ("..." if len(text) > 40 else ""),
                border=1,
                fill=fill,
                new_x="RIGHT",
                new_y="TOP"
            )
        
        pdf.ln(row_height)
    
    pdf.set_fill_color(255, 255, 255)
    pdf.set_draw_color(0, 0, 0)

def export_to_pdf(
    summary: str,
    query: str,
    papers: List[Dict[str, Any]],
) -> bytes:
    """
    Generate a formatted PDF research report and return it as raw bytes.

    The PDF contains:

    1. **Header band** — report title, query, and generation date.
    2. **Summary section** — the full LLM-generated research summary.
    3. **References section** — one entry per paper with title, authors,
       year, and URL.

    Parameters
    ----------
    summary : str
        Full research summary text (typically from ``summarize_papers``).
    query : str
        The original user research query shown at the top of the report.
    papers : list of dict
        Ranked paper dicts with keys: ``title``, ``authors``, ``year``,
        ``url``.

    Returns
    -------
    bytes
        Raw PDF bytes suitable for a Streamlit ``st.download_button``.
        Returns ``b""`` on any failure so the caller can handle gracefully.
    """
    if not isinstance(papers, list):
        papers = []

    try:
        pdf = _ResearchPDF(orientation="P", unit="mm", format="A4")
        pdf.set_margins(PDF_PAGE_MARGIN, PDF_PAGE_MARGIN, PDF_PAGE_MARGIN)
        pdf.set_auto_page_break(auto=True, margin=PDF_PAGE_MARGIN)
        pdf.set_author(PDF_AUTHOR)
        pdf.set_title(_safe_text(query))
        pdf.add_page()

        effective_width: float = pdf.w - 2 * PDF_PAGE_MARGIN

        # ── Report title ───────────────────────────────────────────────────
        pdf.set_font(PDF_FONT_FAMILY, style="B", size=PDF_TITLE_FONT_SIZE)
        pdf.cell(0, 10, PDF_TITLE, align="C", new_x="LMARGIN", new_y="NEXT")

        # ── Query ──────────────────────────────────────────────────────────
        pdf.set_font(PDF_FONT_FAMILY, style="", size=PDF_BODY_FONT_SIZE)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(
            0, PDF_LINE_HEIGHT,
            f"Query: {_safe_text(query)}",
            new_x="LMARGIN", new_y="NEXT",
        )

        # ── Date ───────────────────────────────────────────────────────────
        date_str: str = datetime.now().strftime(PDF_DATE_FORMAT)
        pdf.multi_cell(
            0, PDF_LINE_HEIGHT,
            f"Generated: {date_str}",
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.set_text_color(0, 0, 0)

        # ── Divider ────────────────────────────────────────────────────────
        pdf.ln(2)
        pdf.set_draw_color(180, 180, 180)
        pdf.line(PDF_PAGE_MARGIN, pdf.get_y(), pdf.w - PDF_PAGE_MARGIN, pdf.get_y())
        pdf.ln(4)

        # ── Summary section ────────────────────────────────────────────────
        pdf.set_font(PDF_FONT_FAMILY, style="B", size=PDF_SECTION_FONT_SIZE)
        pdf.cell(0, 8, "Research Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        pdf.set_font(PDF_FONT_FAMILY, style="", size=PDF_BODY_FONT_SIZE)
        clean_summary: str = _safe_text(summary, fallback="No summary available.")
        
        # Collect all table rows first, then render as table
        table_rows = []
        in_table = False
        
        # Render line by line to preserve the report's section formatting
        for line in clean_summary.splitlines():
            line = line.strip()
            
            # Skip unwanted lines
            if not line or line == "N/A" or line == "---":
                if line == "---":
                    pdf.ln(2)
                continue
            
            # Section headers
            if line.startswith("## "):
                title = line.replace("## ", "").strip()
                title = title.replace("**", "")
                pdf.ln(4)
                pdf.set_font(PDF_FONT_FAMILY, style="B", size=12)
                pdf.multi_cell(effective_width, 7, title,
                    new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)
                pdf.set_font(PDF_FONT_FAMILY, style="", size=PDF_BODY_FONT_SIZE)
                continue
            
            if line.startswith("|"):
                # Skip separator rows
                stripped = line.replace("|","").replace("-","").replace(" ","").replace(":","")
                if not stripped:
                    continue
                cells = [c.strip().replace("**","") for c in line.split("|") if c.strip()]
                if cells:
                    table_rows.append(cells)
                in_table = True
                continue
            
            # When table ends render it
            if in_table and not line.startswith("|"):
                in_table = False
                if table_rows:
                    _render_table(pdf, table_rows, effective_width)
                    table_rows = []
                    pdf.ln(3)
            
            # Bullet points
            if line.startswith("- "):
                content = line[2:].strip().replace("**", "")
                pdf.set_font(PDF_FONT_FAMILY, style="", size=PDF_BODY_FONT_SIZE)
                pdf.multi_cell(effective_width, 6, f"  \u2022  {content}",
                    new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1)
                continue
            
            # Regular text (remove bold markers)
            clean_line = line.replace("**", "")
            
            # Clickable link handling
            if clean_line.startswith("Link: http"):
                url = clean_line.replace("Link: ", "").strip()
                pdf.set_font("Helvetica", style="U", size=9)
                pdf.set_text_color(30, 100, 200)
                pdf.cell(
                    0, 6,
                    url[:80] + ("..." if len(url) > 80 else ""),
                    link=url,
                    new_x="LMARGIN",
                    new_y="NEXT"
                )
                pdf.set_text_color(0, 0, 0)
                pdf.set_font("Helvetica", style="", size=10)
                pdf.ln(1)
                continue
            
            if clean_line:
                pdf.set_font(PDF_FONT_FAMILY, style="", size=PDF_BODY_FONT_SIZE)
                pdf.multi_cell(effective_width, 6, clean_line,
                    new_x="LMARGIN", new_y="NEXT")

        # After loop ends check if table still pending:
        if table_rows:
            _render_table(pdf, table_rows, effective_width)

        # ── Divider ────────────────────────────────────────────────────────
        pdf.ln(3)
        pdf.line(PDF_PAGE_MARGIN, pdf.get_y(), pdf.w - PDF_PAGE_MARGIN, pdf.get_y())
        pdf.ln(4)

        # ── References section ─────────────────────────────────────────────
        pdf.set_font(PDF_FONT_FAMILY, style="B", size=PDF_SECTION_FONT_SIZE)
        pdf.cell(0, 8, "References", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        if not papers:
            pdf.set_font(PDF_FONT_FAMILY, style="I", size=PDF_BODY_FONT_SIZE)
            pdf.cell(0, PDF_LINE_HEIGHT, "No papers available.", new_x="LMARGIN", new_y="NEXT")
        else:
            for idx, paper in enumerate(papers, 1):
                try:
                    title: str = _safe_text(paper.get("title"), "Untitled")
                    year: str = _safe_text(paper.get("year"), "")
                    url: str = _safe_text(paper.get("url"), "")

                    raw_authors = paper.get("authors", [])
                    if isinstance(raw_authors, list):
                        author_list = [_safe_text(a) for a in raw_authors[:5]]
                        if len(raw_authors) > 5:
                            author_list.append("et al.")
                        authors: str = ", ".join(author_list)
                    else:
                        authors = _safe_text(raw_authors)

                    # Paper number + title
                    pdf.set_font(PDF_FONT_FAMILY, style="B", size=PDF_BODY_FONT_SIZE)
                    pdf.multi_cell(
                        effective_width, PDF_LINE_HEIGHT,
                        f"{idx}. {title}",
                        new_x="LMARGIN", new_y="NEXT",
                    )

                    # Authors + year
                    pdf.set_font(PDF_FONT_FAMILY, style="", size=PDF_SMALL_FONT_SIZE)
                    pdf.set_text_color(60, 60, 60)
                    meta_line = f"   {authors}"
                    if year:
                        meta_line += f" ({year})"
                    pdf.multi_cell(
                        effective_width, PDF_LINE_HEIGHT,
                        meta_line,
                        new_x="LMARGIN", new_y="NEXT",
                    )

                    # URL
                    if url:
                        pdf.set_font(PDF_MONO_FONT, style="", size=PDF_SMALL_FONT_SIZE)
                        pdf.set_text_color(30, 100, 200)
                        pdf.multi_cell(
                            effective_width, PDF_LINE_HEIGHT,
                            f"   {url}",
                            new_x="LMARGIN", new_y="NEXT",
                        )

                    pdf.set_text_color(0, 0, 0)
                    pdf.ln(2)

                except Exception as paper_exc:
                    logger.warning("export_to_pdf: skipped paper %d — %s", idx, paper_exc)
                    continue

        # ── Serialise to bytes ─────────────────────────────────────────────
        pdf_bytes: bytes = bytes(pdf.output())
        logger.info("export_to_pdf: PDF generated (%d bytes, %d page(s)).",
                    len(pdf_bytes), pdf.page)
        return pdf_bytes

    except Exception as exc:
        logger.error("export_to_pdf: PDF generation failed — %s", exc)
        return b""
