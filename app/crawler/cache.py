"""
app/crawler/cache.py
─────────────────────────────────────────────────────────────────────────────
Disk cache layer for cleaned documentation pages.

Files are stored as:
    cache/
    ├── page_<md5>.md        # Cleaned markdown (filename = md5 hash of URL)
    └── metadata.json        # URL → {title, last_updated, filename} map

Cache behaviour:
    • URL in metadata.json + lastmod unchanged → skip, use cached file.
    • URL is new                               → crawl, clean, save.
    • URL exists but lastmod changed           → re-crawl, overwrite.
─────────────────────────────────────────────────────────────────────────────
"""
import hashlib
import json
import os
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

_CACHE_DIR = "./cache"
_METADATA_FILE = os.path.join(_CACHE_DIR, "metadata.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _url_to_filename(url: str) -> str:
    """Deterministic filename from URL: page_<md5>.md"""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    return f"page_{url_hash}.md"


def _load_metadata() -> dict:
    """Load the metadata.json file, or return empty dict if it doesn't exist."""
    if os.path.isfile(_METADATA_FILE):
        with open(_METADATA_FILE, "r", encoding="utf-8") as fh:
            try:
                return json.load(fh)
            except json.JSONDecodeError:
                return {}
    return {}


def _save_metadata(metadata: dict) -> None:
    """Persist the metadata dict to metadata.json."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_METADATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)


# ── Public API ────────────────────────────────────────────────────────────────

def needs_crawl(url: str, lastmod: Optional[str]) -> bool:
    """
    Check whether a URL needs to be (re-)crawled.

    Returns True if:
        • URL is not in the cache metadata, OR
        • URL exists but lastmod has changed, OR
        • The cached markdown file is missing from disk.
    """
    metadata = _load_metadata()

    if url not in metadata:
        return True

    entry = metadata[url]

    # lastmod changed → re-crawl
    if entry.get("last_updated") != lastmod:
        return True

    # Cached file missing from disk → re-crawl
    filepath = os.path.join(_CACHE_DIR, entry["filename"])
    if not os.path.isfile(filepath):
        return True

    return False


def save_page(url: str, title: str, markdown: str, lastmod: Optional[str]) -> None:
    """
    Save a cleaned page to the disk cache and update metadata.json.

    Args:
        url:      The page's source URL.
        title:    Extracted page title.
        markdown: Cleaned markdown content.
        lastmod:  Last-modified date from sitemap, or None.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)

    filename = _url_to_filename(url)
    filepath = os.path.join(_CACHE_DIR, filename)

    # Write the markdown file
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(markdown)

    # Update metadata
    metadata = _load_metadata()
    metadata[url] = {
        "title": title,
        "last_updated": lastmod,
        "filename": filename,
    }
    _save_metadata(metadata)

    print(f"[cache] Saved {filename}  <-- {url}")


def load_page(url: str) -> Optional[dict[str, str]]:
    """
    Load a cached page from disk.

    Returns:
        {"url": ..., "title": ..., "markdown": ...}  or None if not cached.
    """
    metadata = _load_metadata()

    if url not in metadata:
        return None

    entry = metadata[url]
    filepath = os.path.join(_CACHE_DIR, entry["filename"])

    if not os.path.isfile(filepath):
        return None

    with open(filepath, "r", encoding="utf-8") as fh:
        markdown = fh.read()

    return {
        "url": url,
        "title": entry["title"],
        "markdown": markdown,
    }


def load_all_cached() -> list[dict[str, str]]:
    """
    Load all cached pages from disk.

    Returns:
        List of {"url": ..., "title": ..., "markdown": ...} dicts.
    """
    metadata = _load_metadata()
    pages = []

    for url, entry in metadata.items():
        filepath = os.path.join(_CACHE_DIR, entry["filename"])
        if not os.path.isfile(filepath):
            continue
        with open(filepath, "r", encoding="utf-8") as fh:
            markdown = fh.read()
        pages.append({
            "url": url,
            "title": entry["title"],
            "markdown": markdown,
        })

    return pages
