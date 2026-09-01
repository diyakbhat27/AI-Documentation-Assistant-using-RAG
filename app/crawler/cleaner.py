"""
app/crawler/cleaner.py
─────────────────────────────────────────────────────────────────────────────
Converts raw HTML documentation pages into clean Markdown text.

Strategy:
  1. Parse HTML with BeautifulSoup.
  2. Remove boilerplate: <nav>, <footer>, <header>, sidebar divs, cookie
     banners, search bars, and all <script> / <style> tags.
  3. Locate the main article content region.
  4. Keep: headings (h1–h6), <pre>/<code>, <table>, lists, paragraphs.
  5. Extract page <title> for metadata.
  6. Convert cleaned HTML → Markdown via markdownify.
  7. Strip excessive blank lines (max 2 consecutive).

Output per page:
  {"url": "...", "title": "Creating Tables", "markdown": "# Creating Tables\n\n..."}
─────────────────────────────────────────────────────────────────────────────
"""
import re
from typing import Optional

from bs4 import BeautifulSoup, Tag
import markdownify


# ── CSS selectors for boilerplate to remove ──────────────────────────────────

_REMOVE_SELECTORS = [
    # Structural boilerplate
    "nav",
    "footer",
    "header",

    # Docusaurus sidebar elements
    ".theme-doc-sidebar-container",
    "[class*='docSidebarContainer']",
    "aside",

    # Table of contents / breadcrumb / pagination
    ".table-of-contents",
    "[class*='tocCollapsible']",
    ".breadcrumbs",
    ".pagination-nav",

    # Cookie banners
    "[class*='cookie']",
    "[id*='cookie']",
    "[class*='consent']",

    # Search bars
    "[class*='search']",
    "[role='search']",

    # Accessibility skip links / announcements
    "[class*='skipToContent']",
    "[class*='announcementBar']",

    # Scripts and styles — always remove
    "script",
    "style",
]


# ── Custom markdownify converter ─────────────────────────────────────────────

class _DocConverter(markdownify.MarkdownConverter):
    """Keeps code blocks intact with language annotations."""

    def convert_pre(self, el: Tag, text: str, convert_as_inline: bool = False, **kwargs) -> str:
        # Detect language class (e.g. language-ts, language-python)
        code_tag = el.find("code")
        lang = ""
        if code_tag and code_tag.get("class"):
            for cls in code_tag.get("class", []):
                if cls.startswith("language-"):
                    lang = cls.replace("language-", "")
                    break
        raw = code_tag.get_text() if code_tag else el.get_text()
        return f"\n```{lang}\n{raw.strip()}\n```\n\n"


def _to_markdown(html_fragment: str) -> str:
    """Convert an HTML fragment to Markdown."""
    return _DocConverter(
        heading_style=markdownify.ATX,
        bullets="-",
        strip=["img"],  # skip images (no value in text RAG)
    ).convert(html_fragment)


# ── Post-processing ──────────────────────────────────────────────────────────

def _clean_markdown(text: str) -> str:
    """Collapse excessive blank lines (max 2 consecutive) and strip edges."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def clean_page(url: str, html: str) -> Optional[dict[str, str]]:
    """
    Given a raw HTML string and its source URL, return a cleaned dict with
    the page title and full markdown content, or None if no content found.

    Args:
        url:  The page URL (used for logging and metadata).
        html: Raw HTML string fetched by fetcher.py.

    Returns:
        {"url": ..., "title": ..., "markdown": ...}  or None on failure.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as exc:
        print(f"[cleaner] Parse error for {url}: {exc}")
        return None

    # ── Remove boilerplate elements ─────────────────────────────────────────
    for selector in _REMOVE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    # ── Extract page <title> ────────────────────────────────────────────────
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        # Docusaurus titles often have " | SiteName" suffix — strip it
        title = title_tag.string.strip().split("|")[0].strip()
    else:
        # Fallback: use <h1> text or last URL segment
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else url.rstrip("/").split("/")[-1]

    # ── Locate main content region ──────────────────────────────────────────
    # Docusaurus wraps article content in <article> or .theme-doc-markdown
    article = (
        soup.find("article")
        or soup.find(class_="theme-doc-markdown")
        or soup.find("main")
    )
    if not article:
        print(f"[cleaner] WARN: No content region found for {url}")
        return None

    # ── Convert to Markdown ─────────────────────────────────────────────────
    md_raw = _to_markdown(str(article))
    md = _clean_markdown(md_raw)

    if len(md) < 50:  # Skip effectively empty pages
        print(f"[cleaner] WARN: Skipping near-empty page {url}")
        return None

    word_count = len(md.split())
    print(f"[cleaner] OK {url}  title={title!r}  words={word_count}")

    return {
        "url": url,
        "title": title,
        "markdown": md,
    }
