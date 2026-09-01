"""
app/chunking/chunker.py
─────────────────────────────────────────────────────────────────────────────
Phase 3: Chunking Pipeline

Transforms cached markdown files into retrieval-optimized chunks. Each chunk
is self-contained and carries enough metadata to be cited as a source.

Pipeline:
  Step 3.1 — Document Parser
    • Read every .md file from cache/
    • Load title + source_url from metadata.json
    • Parse markdown to extract section, code block, and paragraph boundaries

  Step 3.2 — Smart Chunking Logic
    • Chunk size: 800 chars, overlap: 150 chars
    • Never split inside a code block
    • Prefer heading boundaries (## / ###)
    • Prefer paragraph boundaries (double newlines)
    • Hard limit fallback at 800 ± 200 char tolerance

  Step 3.3 — Metadata Attachment
    • Each chunk gets a uuid4 id, text, source_url, page_title,
      section_heading, and char_count

  Step 3.4 — Chunk Validation
    • No empty text, no duplicate IDs, source_url and page_title populated
    • No chunk exceeds 1200 characters
    • Warning if total chunk count < 50
─────────────────────────────────────────────────────────────────────────────
"""
import json
import os
import re
import uuid
from typing import List, Optional

from app.core.settings import settings


# ── Constants ─────────────────────────────────────────────────────────────────

CHUNK_SIZE: int = settings.CHUNK_SIZE        # 800
CHUNK_OVERLAP: int = settings.CHUNK_OVERLAP  # 150
MAX_CHUNK_CHARS: int = 1200                  # Validation ceiling
TOLERANCE: int = 200                         # Boundary search window


# ── Chunk data structure ──────────────────────────────────────────────────────

def _make_chunk(
    text: str,
    source_url: str,
    page_title: str,
    section_heading: str,
) -> dict:
    """Create a single chunk metadata dict (Step 3.3)."""
    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "source_url": source_url,
        "page_title": page_title,
        "section_heading": section_heading,
        "char_count": len(text),
    }


# ── Step 3.1 — Document Parser ───────────────────────────────────────────────

def _load_cache_metadata(cache_dir: str) -> dict:
    """Load metadata.json from the cache directory."""
    metadata_path = os.path.join(cache_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"[chunker] metadata.json not found in {cache_dir}"
        )
    with open(metadata_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_all_pages(cache_dir: str) -> List[dict]:
    """
    Read every .md file from cache/ and pair it with metadata.

    Returns:
        List of dicts with keys: source_url, page_title, markdown
    """
    metadata = _load_cache_metadata(cache_dir)

    # Build filename → (url, title) lookup
    filename_to_meta: dict[str, tuple[str, str]] = {}
    for url, entry in metadata.items():
        filename_to_meta[entry["filename"]] = (url, entry["title"])

    pages: List[dict] = []

    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".md"):
            continue

        filepath = os.path.join(cache_dir, fname)
        with open(filepath, "r", encoding="utf-8") as fh:
            markdown = fh.read()

        if fname in filename_to_meta:
            url, title = filename_to_meta[fname]
        else:
            # Orphan .md file — skip with warning
            print(f"[chunker] WARN: {fname} has no metadata entry, skipping")
            continue

        pages.append({
            "source_url": url,
            "page_title": title,
            "markdown": markdown,
        })

    print(f"[chunker] Loaded {len(pages)} markdown files from cache")
    return pages


# ── Markdown structural parsing ──────────────────────────────────────────────

def _find_code_block_ranges(text: str) -> List[tuple[int, int]]:
    """
    Find all (start, end) character offsets of fenced code blocks (``` pairs).
    These regions must never be split.
    """
    ranges: List[tuple[int, int]] = []
    pattern = re.compile(r"^```", re.MULTILINE)
    matches = list(pattern.finditer(text))

    # Pair up opening and closing fences
    i = 0
    while i < len(matches) - 1:
        start = matches[i].start()
        end = matches[i + 1].end()
        ranges.append((start, end))
        i += 2

    return ranges


def _is_inside_code_block(pos: int, code_ranges: List[tuple[int, int]]) -> bool:
    """Check if a character position falls inside any code block."""
    for start, end in code_ranges:
        if start <= pos <= end:
            return True
    return False


def _find_code_block_end(pos: int, code_ranges: List[tuple[int, int]]) -> int:
    """If pos is inside a code block, return the end of that block."""
    for start, end in code_ranges:
        if start <= pos <= end:
            return end
    return pos


def _extract_sections(markdown: str) -> List[tuple[str, str, int]]:
    """
    Parse markdown into sections delimited by ## or ### headings.

    Returns:
        List of (heading_text, section_body, char_offset) tuples.
        The first section may have heading="" if content precedes any heading.
    """
    pattern = re.compile(r"^(#{2,3} .+)$", re.MULTILINE)
    matches = list(pattern.finditer(markdown))

    sections: List[tuple[str, str, int]] = []

    if not matches:
        # No headings — entire document is one section
        return [("", markdown.strip(), 0)]

    # Text before the first heading
    pre_heading = markdown[: matches[0].start()].strip()
    if pre_heading:
        sections.append(("", pre_heading, 0))

    for i, match in enumerate(matches):
        heading = match.group(1).lstrip("#").strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[body_start:body_end].strip()
        if body:
            sections.append((heading, body, match.start()))

    return sections


# ── Step 3.2 — Smart Chunking Logic ──────────────────────────────────────────

def _find_best_split_point(
    text: str,
    target: int,
    code_ranges: List[tuple[int, int]],
) -> int:
    """
    Find the best split point near `target` position in `text`.

    Priority order:
      1. Never split inside a code block — extend to code block end
      2. Heading boundary (## or ###) within ±TOLERANCE
      3. Paragraph boundary (double newline) within ±TOLERANCE
      4. Hard cut at target
    """
    # If target is inside a code block, extend past it
    if _is_inside_code_block(target, code_ranges):
        return _find_code_block_end(target, code_ranges)

    search_start = max(0, target - TOLERANCE)
    search_end = min(len(text), target + TOLERANCE)
    search_window = text[search_start:search_end]

    # Look for heading boundary in the window
    heading_match = None
    for m in re.finditer(r"\n(?=#{2,3} )", search_window):
        candidate = search_start + m.start()
        # Prefer the split closest to target
        if heading_match is None or abs(candidate - target) < abs(heading_match - target):
            # Make sure this split point is NOT inside a code block
            if not _is_inside_code_block(candidate, code_ranges):
                heading_match = candidate

    if heading_match is not None:
        return heading_match

    # Look for paragraph boundary (double newline) in the window
    para_match = None
    for m in re.finditer(r"\n\n", search_window):
        candidate = search_start + m.start()
        if para_match is None or abs(candidate - target) < abs(para_match - target):
            if not _is_inside_code_block(candidate, code_ranges):
                para_match = candidate

    if para_match is not None:
        return para_match + 1  # Split after the first newline

    # Hard fallback — split at target (but not inside a code block)
    return target


def _split_oversized_chunk(chunk_dict: dict) -> List[dict]:
    """
    If a chunk exceeds MAX_CHUNK_CHARS, force-split it into smaller pieces.
    This handles runaway code blocks that can't be avoided during initial chunking.
    """
    text = chunk_dict["text"]
    if len(text) <= MAX_CHUNK_CHARS:
        return [chunk_dict]

    # Force split at paragraph, newline, or hard cut
    pieces: List[dict] = []
    pos = 0
    while pos < len(text):
        end = min(pos + CHUNK_SIZE, len(text))
        if end >= len(text):
            piece = text[pos:].strip()
            if piece:
                pieces.append(_make_chunk(
                    text=piece,
                    source_url=chunk_dict["source_url"],
                    page_title=chunk_dict["page_title"],
                    section_heading=chunk_dict["section_heading"],
                ))
            break

        # Try paragraph boundary
        search_start = max(pos, end - TOLERANCE)
        search_end = min(len(text), end + TOLERANCE)
        window = text[search_start:search_end]

        split_at = None
        for m in re.finditer(r"\n\n", window):
            candidate = search_start + m.start() + 1
            if split_at is None or abs(candidate - end) < abs(split_at - end):
                split_at = candidate

        if split_at is None:
            # Try single newline
            for m in re.finditer(r"\n", window):
                candidate = search_start + m.start() + 1
                if split_at is None or abs(candidate - end) < abs(split_at - end):
                    split_at = candidate

        if split_at is None or split_at <= pos:
            split_at = end

        piece = text[pos:split_at].strip()
        if piece:
            pieces.append(_make_chunk(
                text=piece,
                source_url=chunk_dict["source_url"],
                page_title=chunk_dict["page_title"],
                section_heading=chunk_dict["section_heading"],
            ))

        next_pos = split_at - CHUNK_OVERLAP
        if next_pos <= pos:
            next_pos = split_at
        pos = next_pos

    return pieces


def _smart_chunk_text(
    text: str,
    section_heading: str,
    source_url: str,
    page_title: str,
) -> List[dict]:
    """
    Split a text block into overlapping chunks using smart boundary detection.

    Returns:
        List of chunk metadata dicts.
    """
    if not text.strip():
        return []

    code_ranges = _find_code_block_ranges(text)
    chunks: List[dict] = []
    pos = 0
    text_len = len(text)

    while pos < text_len:
        # Determine end of this chunk
        end = pos + CHUNK_SIZE

        if end >= text_len:
            # Last chunk -- take everything remaining
            chunk_text = text[pos:].strip()
            if chunk_text:
                chunk = _make_chunk(
                    text=chunk_text,
                    source_url=source_url,
                    page_title=page_title,
                    section_heading=section_heading,
                )
                chunks.extend(_split_oversized_chunk(chunk))
            break

        # Find the best split point
        split_at = _find_best_split_point(text, end, code_ranges)

        # Safety: ensure we always advance past pos
        if split_at <= pos:
            split_at = end

        chunk_text = text[pos:split_at].strip()
        if chunk_text:
            chunk = _make_chunk(
                text=chunk_text,
                source_url=source_url,
                page_title=page_title,
                section_heading=section_heading,
            )
            # Handle oversized chunks from code block extension
            chunks.extend(_split_oversized_chunk(chunk))

        # Overlap: next chunk starts CHUNK_OVERLAP chars before the split
        next_pos = split_at - CHUNK_OVERLAP
        if next_pos <= pos:
            next_pos = split_at  # Avoid infinite loop on tiny chunks

        pos = next_pos

    return chunks


def _chunk_single_page(page: dict) -> List[dict]:
    """
    Convert a single page dict into a list of chunk metadata dicts.

    Args:
        page: Dict with keys source_url, page_title, markdown
    """
    markdown = page["markdown"]
    source_url = page["source_url"]
    page_title = page["page_title"]

    sections = _extract_sections(markdown)
    all_chunks: List[dict] = []

    for heading, body, _ in sections:
        # Prepend heading to section body for context
        section_text = f"{heading}\n\n{body}".strip() if heading else body

        if len(section_text) <= CHUNK_SIZE:
            # Small section — single chunk
            chunk = _make_chunk(
                text=section_text,
                source_url=source_url,
                page_title=page_title,
                section_heading=heading,
            )
            all_chunks.append(chunk)
        else:
            # Large section — smart split
            sub_chunks = _smart_chunk_text(
                text=section_text,
                section_heading=heading,
                source_url=source_url,
                page_title=page_title,
            )
            all_chunks.extend(sub_chunks)

    return all_chunks


# ── Step 3.4 — Chunk Validation ──────────────────────────────────────────────

class ChunkValidationError(Exception):
    """Raised when chunk validation fails."""
    pass


def _validate_chunks(chunks: List[dict]) -> None:
    """
    Assert all Phase 3 exit criteria.

    Raises ChunkValidationError on any failure.
    """
    errors: List[str] = []
    seen_ids: set = set()

    for i, chunk in enumerate(chunks):
        label = f"Chunk {i} (id={chunk.get('id', 'MISSING')[:12]}...)"

        # No empty or whitespace-only text
        if not chunk.get("text", "").strip():
            errors.append(f"{label}: empty or whitespace-only text")

        # No duplicate IDs
        cid = chunk.get("id", "")
        if cid in seen_ids:
            errors.append(f"{label}: duplicate id")
        seen_ids.add(cid)

        # source_url populated
        if not chunk.get("source_url", "").strip():
            errors.append(f"{label}: missing source_url")

        # page_title populated
        if not chunk.get("page_title", "").strip():
            errors.append(f"{label}: missing page_title")

        # No chunk exceeds MAX_CHUNK_CHARS (catches runaway code blocks)
        char_count = chunk.get("char_count", 0)
        if char_count > MAX_CHUNK_CHARS:
            errors.append(
                f"{label}: exceeds {MAX_CHUNK_CHARS} chars "
                f"(actual: {char_count})"
            )

    if errors:
        error_report = "\n".join(f"  * {e}" for e in errors)
        raise ChunkValidationError(
            f"[chunker] Validation FAILED with {len(errors)} error(s):\n"
            f"{error_report}"
        )

    # Warning: suspiciously low count
    if len(chunks) < 50:
        print(
            f"[chunker] WARNING: Only {len(chunks)} chunks produced. "
            f"Expected 100-500. Something may have gone wrong."
        )

    print(f"[chunker] [OK] Validation passed -- {len(chunks)} chunks, 0 errors")


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_pages_from_cache(
    cache_dir: Optional[str] = None,
) -> List[dict]:
    """
    Phase 3 entry point: read cached markdown, chunk, validate, return.

    Args:
        cache_dir: Path to cache directory. Defaults to settings.CACHE_DIR.

    Returns:
        List of validated chunk dicts, each with keys:
        id, text, source_url, page_title, section_heading, char_count
    """
    cache_dir = cache_dir or settings.CACHE_DIR

    print("=" * 60)
    print("  Phase 3: Chunking Pipeline")
    print("=" * 60)

    # Step 3.1 — Load all cached pages
    pages = _load_all_pages(cache_dir)
    if not pages:
        print("[chunker] No cached pages found. Run the crawler first.")
        return []

    # Step 3.2 + 3.3 — Chunk each page with metadata
    all_chunks: List[dict] = []
    for page in pages:
        page_chunks = _chunk_single_page(page)
        all_chunks.extend(page_chunks)
        print(
            f"[chunker] {page['page_title']:<40s} -> "
            f"{len(page_chunks):>3d} chunks"
        )

    print(f"\n[chunker] Total chunks produced: {len(all_chunks)}")

    # Step 3.4 — Validate
    _validate_chunks(all_chunks)

    return all_chunks


# ── Backward-compatible Chunk dataclass ───────────────────────────────────────
# The retriever.py accesses .text, .url, .title, .heading, .chunk_index
# on chunk objects. This dataclass maintains that interface.

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Backward-compatible chunk object for retriever.py."""
    text: str
    url: str
    title: str
    heading: str = ""
    chunk_index: int = 0


def chunk_pages(pages: list) -> list:
    """
    Legacy adapter: accepts a list of cleaned page dicts and returns Chunk
    objects compatible with both the Phase 3 schema and retriever.py.

    Each input page dict should have keys: url, title, markdown
    Returns list of Chunk dataclass instances.
    """
    all_chunks: list = []
    global_idx = 0

    for page in pages:
        page_data = {
            "source_url": page.get("url", "") if isinstance(page, dict) else getattr(page, "url", ""),
            "page_title": page.get("title", "") if isinstance(page, dict) else getattr(page, "title", ""),
            "markdown": page.get("markdown", "") if isinstance(page, dict) else getattr(page, "markdown", ""),
        }
        page_chunks = _chunk_single_page(page_data)

        for chunk_dict in page_chunks:
            chunk_obj = Chunk(
                text=chunk_dict["text"],
                url=chunk_dict["source_url"],
                title=chunk_dict["page_title"],
                heading=chunk_dict["section_heading"],
                chunk_index=global_idx,
            )
            all_chunks.append(chunk_obj)
            global_idx += 1

        print(
            f"[chunker] {page_data['source_url']} -> "
            f"{len(page_chunks)} chunks"
        )

    print(f"[chunker] Total chunks: {len(all_chunks)}")

    # Convert to dicts for validation, then return Chunk objects
    validation_dicts = [
        {
            "id": str(uuid.uuid4()),
            "text": c.text,
            "source_url": c.url,
            "page_title": c.title,
            "section_heading": c.heading,
            "char_count": len(c.text),
        }
        for c in all_chunks
    ]
    _validate_chunks(validation_dicts)

    return all_chunks

