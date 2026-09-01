"""
app/crawler/sitemap.py
─────────────────────────────────────────────────────────────────────────────
Discovers all documentation page URLs from the LioranDB sitemap.xml.

• Fetches sitemap via httpx.
• Parses XML using Python's built-in xml.etree.ElementTree.
• Extracts all <loc> values and optional <lastmod> dates.
• Filters out non-page URLs (blog, download, markdown-page, category indexes,
  images, feeds, and asset files).
• Returns a clean list of dicts: [{"url": "...", "lastmod": "..." | None}]
─────────────────────────────────────────────────────────────────────────────
"""
import re
import xml.etree.ElementTree as ET
from typing import Optional

import httpx

from app.core.settings import settings

# ── Sitemap XML namespace ─────────────────────────────────────────────────────
# The sitemaps.org schema wraps every element in this namespace.
_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# ── URL patterns to skip ─────────────────────────────────────────────────────
# Navigation / category pages and non-content paths.
_SKIP_PATTERNS = re.compile(
    r"/blog|/download|/markdown-page|/docs/category/",
    re.IGNORECASE,
)

# File extensions that indicate non-page resources (images, feeds, assets).
_ASSET_EXTENSIONS = re.compile(
    r"\.(png|jpg|jpeg|gif|svg|webp|ico|pdf|xml|rss|atom|json|css|js|woff2?)$",
    re.IGNORECASE,
)


def get_doc_urls() -> list[dict[str, Optional[str]]]:
    """
    Fetch sitemap.xml and return a deduplicated, sorted list of documentation
    page URLs with their last-modified dates.

    Returns:
        List of dicts, each with:
            - "url"     (str):           Absolute page URL.
            - "lastmod" (str | None):    ISO date string, or None if absent.

    Raises:
        RuntimeError: If the sitemap cannot be fetched or parsed.
    """
    sitemap_url = f"{settings.DOCS_BASE_URL}/sitemap.xml"
    print(f"[sitemap] Fetching sitemap from {sitemap_url} ...")

    try:
        resp = httpx.get(sitemap_url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"[sitemap] Could not fetch sitemap: {exc}") from exc

    # ── Parse XML ────────────────────────────────────────────────────────────
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        raise RuntimeError(f"[sitemap] Could not parse sitemap XML: {exc}") from exc

    # ── Extract <loc> and <lastmod> from each <url> entry ────────────────────
    seen: set[str] = set()
    results: list[dict[str, Optional[str]]] = []

    for url_element in root.findall("sm:url", _NS):
        loc_tag = url_element.find("sm:loc", _NS)
        if loc_tag is None or not loc_tag.text:
            continue

        url = loc_tag.text.strip()

        # Skip duplicates
        if url in seen:
            continue
        seen.add(url)

        # Skip non-page URLs
        if _SKIP_PATTERNS.search(url):
            continue
        if _ASSET_EXTENSIONS.search(url):
            continue

        # Extract optional lastmod
        lastmod_tag = url_element.find("sm:lastmod", _NS)
        lastmod = lastmod_tag.text.strip() if lastmod_tag is not None and lastmod_tag.text else None

        results.append({"url": url, "lastmod": lastmod})

    # Sort by URL for deterministic ordering
    results.sort(key=lambda entry: entry["url"])

    print(f"[sitemap] Found {len(results)} documentation URLs to crawl.")
    return results
