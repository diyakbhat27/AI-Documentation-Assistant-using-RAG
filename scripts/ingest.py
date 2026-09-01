"""
scripts/ingest.py
─────────────────────────────────────────────────────────────────────────────
Full ingestion pipeline: crawl -> clean -> chunk -> embed -> index

Run from the project root:
    python scripts/ingest.py              # Full pipeline (Phases 2-4)
    python scripts/ingest.py --phase3     # Phase 3 only: chunk cached pages
    python scripts/ingest.py --phase4     # Phase 4 only: embed + index cached chunks
    python scripts/ingest.py --force      # Force re-index even if data exists
    python scripts/ingest.py --test       # Run a test search query after indexing

Phase 4 flow:
  1. Run sitemap discovery
  2. Fetch and cache all pages (skip cached)
  3. Load all chunks from cache (Phase 3)
  4. Check if ChromaDB already has data
     -> If yes: skip embedding, print count, exit
     -> If no: proceed
  5. Embed all chunks in batches of 64
  6. Insert into ChromaDB in batches of 100
  7. Print final collection.count()
─────────────────────────────────────────────────────────────────────────────
"""
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.settings import settings


def run_phase3_only():
    """Run Phase 3 (Chunking Pipeline) in isolation on cached pages."""
    from app.chunking.chunker import chunk_pages_from_cache

    chunks = chunk_pages_from_cache()

    print("\n" + "=" * 60)
    print(f"  Phase 3 Complete!")
    print(f"    Chunks produced : {len(chunks)}")
    print(f"    Cache directory : {settings.CACHE_DIR}")
    print("=" * 60)

    return chunks


def run_phase4_only(force: bool = False):
    """
    Run Phase 4 (Embed + Index) in isolation.

    Loads chunks from cache via Phase 3, then embeds and indexes them.
    """
    from app.chunking.chunker import chunk_pages_from_cache
    from app.rag.retriever import build_index, index_size

    # Phase 3: Chunk from cache
    chunks = chunk_pages_from_cache()
    if not chunks:
        print("[ingest] No chunks produced. Aborting Phase 4.")
        return

    # Phase 4: Embed + Index
    print("\n" + "=" * 60)
    print("  Phase 4: Embeddings & ChromaDB Indexing")
    print("=" * 60)

    final_count = build_index(chunks, force=force)

    print("\n" + "=" * 60)
    print(f"  Phase 4 Complete!")
    print(f"    Chunks processed : {len(chunks)}")
    print(f"    ChromaDB count   : {final_count}")
    print(f"    Embedding        : {settings.EMBEDDING_PROVIDER}")
    print(f"    Vector store     : {settings.CHROMA_PERSIST_DIR}")
    print("=" * 60)

    return final_count


def run_test_search():
    """Run a manual test search query to verify the index works."""
    from app.rag.retriever import retrieve, index_size

    count = index_size()
    if count == 0:
        print("[test] ChromaDB is empty. Run ingest first.")
        return

    print("\n" + "=" * 60)
    print("  Test Search")
    print("=" * 60)
    print(f"[test] ChromaDB has {count} documents.")

    test_query = "How do I create an index in LioranDB?"
    print(f"[test] Query: {test_query!r}")

    results = retrieve(test_query, top_k=5)

    print(f"[test] Got {len(results)} results:\n")
    for i, chunk in enumerate(results, 1):
        print(f"  --- Result {i} (score={chunk.score}) ---")
        print(f"  Title:   {chunk.title}")
        print(f"  Heading: {chunk.heading}")
        print(f"  URL:     {chunk.url}")
        print(f"  Text:    {chunk.text[:150]}...")
        print()

    print("=" * 60)


def main(force: bool = False):
    """Full ingestion pipeline: crawl -> clean -> chunk -> embed -> index."""
    from app.crawler.sitemap import get_doc_urls
    from app.crawler.fetcher import fetch_all_pages
    from app.crawler.cleaner import clean_page
    from app.chunking.chunker import chunk_pages_from_cache
    from app.rag.retriever import build_index, index_size

    print("=" * 60)
    print("  LioranDB AI Assistant -- Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Discover URLs
    urls = get_doc_urls()
    url_strings = [entry["url"] for entry in urls]

    # Step 2: Download pages (with caching)
    html_map = fetch_all_pages(url_strings)

    # Step 3: Clean HTML -> Markdown (handled by fetcher + cache)
    pages_cleaned = 0
    for url, html in html_map.items():
        page = clean_page(url, html)
        if page:
            pages_cleaned += 1

    print(f"\n[ingest] [x] Cleaned {pages_cleaned} pages.")

    # Phase 3: Chunk from cache
    chunks = chunk_pages_from_cache()
    if not chunks:
        print("[ingest] No chunks produced. Aborting.")
        return

    # Phase 4: Check if already indexed, embed + index
    print("\n" + "=" * 60)
    print("  Phase 4: Embeddings & ChromaDB Indexing")
    print("=" * 60)

    final_count = build_index(chunks, force=force)

    print("\n" + "=" * 60)
    print(f"  [x] Ingestion complete!")
    print(f"    Pages crawled  : {pages_cleaned}")
    print(f"    Chunks indexed : {final_count}")
    print(f"    Embedding      : {settings.EMBEDDING_PROVIDER}")
    print(f"    Vector store   : {settings.CHROMA_PERSIST_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    force_flag = "--force" in sys.argv

    if "--phase3" in sys.argv:
        run_phase3_only()
    elif "--phase4" in sys.argv:
        run_phase4_only(force=force_flag)
    elif "--test" in sys.argv:
        run_test_search()
    else:
        main(force=force_flag)
