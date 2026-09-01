"""
app/crawler/fetcher.py
─────────────────────────────────────────────────────────────────────────────
Synchronous page downloader using httpx.

• Concurrency: sequential (simple and reliable for ~40 pages).
• Timeout: 10 seconds per request.
• Retry: up to 3 attempts with exponential backoff (1s, 2s, 4s).
• On permanent failure after retries: logs the URL and skips — never crashes.
• Returns a dict mapping URL → HTML string.
─────────────────────────────────────────────────────────────────────────────
"""
import time
from typing import Optional

import httpx


# ── Configuration ─────────────────────────────────────────────────────────────

_TIMEOUT_SECONDS = 10        # per-request timeout
_MAX_RETRIES = 3             # total attempts per URL
_BACKOFF_BASE = 1            # first retry waits 1s, then 2s, then 4s

_USER_AGENT = "LioranDB-AI-Assistant/1.0 (documentation crawler)"


# ── Single-page fetch with retry ─────────────────────────────────────────────

def _fetch_one(
    client: httpx.Client,
    url: str,
) -> Optional[tuple[str, str]]:
    """
    Download a single URL with retry + exponential backoff.

    Returns:
        (url, html_string) on success, or None on failure.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.get(url, follow_redirects=True)
            resp.raise_for_status()
            print(f"[fetcher] OK {url}")
            return (url, resp.text)

        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            wait = _BACKOFF_BASE * (2 ** (attempt - 1))  # 1s, 2s, 4s
            if attempt < _MAX_RETRIES:
                print(
                    f"[fetcher] WARN: Attempt {attempt}/{_MAX_RETRIES} "
                    f"failed for {url}: {exc}  — retrying in {wait}s"
                )
                time.sleep(wait)
            else:
                print(
                    f"[fetcher] ERROR: All {_MAX_RETRIES} attempts failed "
                    f"for {url}: {exc}  — skipping."
                )
                return None


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_all_pages(urls: list[str]) -> dict[str, str]:
    """
    Download all given URLs sequentially with retry.

    Args:
        urls: List of absolute page URL strings to fetch.

    Returns:
        Dict mapping URL → HTML string (only successful fetches).
    """
    results: dict[str, str] = {}

    with httpx.Client(
        timeout=httpx.Timeout(_TIMEOUT_SECONDS),
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        for url in urls:
            result = _fetch_one(client, url)
            if result is not None:
                results[result[0]] = result[1]

    print(f"[fetcher] Done — {len(results)}/{len(urls)} pages fetched.")
    return results
