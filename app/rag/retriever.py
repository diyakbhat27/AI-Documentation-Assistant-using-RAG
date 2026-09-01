"""
app/rag/retriever.py
─────────────────────────────────────────────────────────────────────────────
Phase 4: ChromaDB Vector Store Manager

Manages the ChromaDB persistent vector store:
  * build_index()  -- embed all chunks and upsert into ChromaDB
  * search()       -- embed a query and return top-K matching chunks
  * retrieve()     -- high-level search (legacy compatibility)
  * index_size()   -- return current collection count

ChromaDB is used in persistent mode (chroma_db/ on disk) so the index
survives server restarts without re-embedding.

Phase 4 behavior:
  * If collection already has data on startup -> skip re-indexing
  * Embed in batches of 64 (memory safe)
  * Insert into ChromaDB in batches of 100
  * Verify count matches expected after indexing
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.settings import settings
from app.rag.embedder import get_embedder, EMBED_BATCH_SIZE


# ── Constants ─────────────────────────────────────────────────────────────────

CHROMA_INSERT_BATCH: int = 100   # Batch size for ChromaDB upserts


# ── Dataclass for a retrieved result ─────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single chunk retrieved from the vector store with its score."""
    id: str             # ChromaDB document ID
    text: str
    url: str
    title: str
    heading: str
    score: float        # Distance score (lower = more similar for cosine)


# ── ChromaDB client (module-level singleton) ──────────────────────────────────

_chroma_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    """
    Initialize ChromaDB PersistentClient and get/create the collection.

    Uses singleton pattern -- client and collection are created once and reused.
    The collection uses cosine similarity for HNSW index.
    """
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    _chroma_client = chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    _collection = _chroma_client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )

    count = _collection.count()
    if count > 0:
        print(
            f"[retriever] ChromaDB collection '{settings.CHROMA_COLLECTION_NAME}' "
            f"loaded with {count} existing documents."
        )
    else:
        print(
            f"[retriever] ChromaDB collection '{settings.CHROMA_COLLECTION_NAME}' "
            f"created (empty)."
        )

    return _collection


# ── Index building (Phase 4) ──────────────────────────────────────────────────

class IndexingError(Exception):
    """Raised when indexing fails verification."""
    pass


def build_index(chunks: list, force: bool = False) -> int:
    """
    Embed all chunks and upsert them into ChromaDB.

    Phase 4 flow:
      1. Check if ChromaDB already has data -> skip if yes (unless force=True)
      2. Embed all chunks in batches of 64
      3. Insert into ChromaDB in batches of 100
      4. Verify final count matches expected

    Args:
        chunks:  List of chunk objects/dicts from chunker.py.
                 Supports both Chunk dataclass (.text, .url) and
                 Phase 3 dicts (id, text, source_url, page_title).
        force:   If True, re-index even if data exists.

    Returns:
        Final collection count.

    Raises:
        IndexingError: If indexing completes but count is 0.
    """
    collection = _get_collection()
    existing_count = collection.count()

    # Step 4.3: Skip if already indexed
    if existing_count > 0 and not force:
        print(
            f"[retriever] ChromaDB already has {existing_count} documents. "
            f"Skipping re-indexing. Use force=True to re-index."
        )
        return existing_count

    if force and existing_count > 0:
        print(f"[retriever] Force re-index requested. Clearing {existing_count} existing documents...")
        # Delete all existing documents
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
        print(f"[retriever] Cleared collection.")

    # Normalize chunks to a common format
    total = len(chunks)
    print(f"[retriever] Indexing {total} chunks into ChromaDB ...")

    embedder = get_embedder()

    # Step 4.3, bullet 5: Embed all chunks in batches of 64
    print(f"[retriever] Embedding {total} chunks (batch size={EMBED_BATCH_SIZE}) ...")

    all_texts: List[str] = []
    all_ids: List[str] = []
    all_metadatas: List[dict] = []

    for chunk in chunks:
        # Support both Chunk dataclass and Phase 3 dicts
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            chunk_id = chunk.get("id", "")
            url = chunk.get("source_url", chunk.get("url", ""))
            title = chunk.get("page_title", chunk.get("title", ""))
            heading = chunk.get("section_heading", chunk.get("heading", ""))
        else:
            text = getattr(chunk, "text", "")
            chunk_id = getattr(chunk, "id", "")
            url = getattr(chunk, "url", getattr(chunk, "source_url", ""))
            title = getattr(chunk, "title", getattr(chunk, "page_title", ""))
            heading = getattr(chunk, "heading", getattr(chunk, "section_heading", ""))

        # Generate ID if missing
        if not chunk_id:
            import hashlib
            idx = getattr(chunk, "chunk_index", len(all_ids))
            chunk_id = hashlib.md5(f"{url}::{idx}".encode()).hexdigest()

        all_texts.append(text)
        all_ids.append(chunk_id)
        all_metadatas.append({
            "url": url,
            "title": title,
            "heading": heading,
            "source_url": url,
            "page_title": title,
            "section_heading": heading,
        })

    # Embed in batches of 64
    print(f"[retriever] Generating embeddings ...")
    all_embeddings = embedder.embed_chunks(all_texts)
    print(f"[retriever] Embeddings generated: {len(all_embeddings)} vectors.")

    # Step 4.3, bullet 6: Insert into ChromaDB in batches of 100
    for batch_start in range(0, total, CHROMA_INSERT_BATCH):
        batch_end = min(batch_start + CHROMA_INSERT_BATCH, total)

        batch_ids = all_ids[batch_start:batch_end]
        batch_texts = all_texts[batch_start:batch_end]
        batch_embeddings = all_embeddings[batch_start:batch_end]
        batch_metadatas = all_metadatas[batch_start:batch_end]

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )

        batch_num = batch_start // CHROMA_INSERT_BATCH + 1
        total_batches = (total + CHROMA_INSERT_BATCH - 1) // CHROMA_INSERT_BATCH
        print(
            f"[retriever] Upserted batch {batch_num}/{total_batches} "
            f"({len(batch_ids)} chunks)"
        )

    # Step 4.4: Persistence verification
    final_count = collection.count()
    print(f"[retriever] [OK] Index built -- {final_count} total documents in ChromaDB.")

    if final_count == 0:
        raise IndexingError(
            "Indexing failed -- no chunks were stored in ChromaDB. "
            "Check embedder and chunk data."
        )

    if final_count != total:
        print(
            f"[retriever] WARNING: Expected {total} documents but "
            f"ChromaDB reports {final_count}. Some chunks may have "
            f"duplicate IDs."
        )

    return final_count


# ── Search / Query retrieval (Phase 4) ────────────────────────────────────────

def search(
    query_embedding: List[float],
    top_k: int = 5,
) -> List[RetrievedChunk]:
    """
    Search ChromaDB for the top-K most similar chunks to the query embedding.

    Args:
        query_embedding: Pre-computed embedding vector for the query.
        top_k:           Number of results to return (default 5).

    Returns:
        List of RetrievedChunk ordered by relevance (most relevant first).
    """
    collection = _get_collection()
    count = collection.count()

    if count == 0:
        print("[retriever] WARNING: ChromaDB is empty. Run ingest.py first.")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    retrieved: List[RetrievedChunk] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
        retrieved.append(
            RetrievedChunk(
                id=chunk_id,
                text=doc,
                url=meta.get("url", meta.get("source_url", "")),
                title=meta.get("title", meta.get("page_title", "")),
                heading=meta.get("heading", meta.get("section_heading", "")),
                score=round(float(dist), 4),
            )
        )

    return retrieved


def retrieve(query: str, top_k: int | None = None) -> List[RetrievedChunk]:
    """
    High-level search: embed the query string and return top-K chunks.

    This is the primary interface used by the generator. It handles
    embedding the query internally.

    Args:
        query: The user's natural-language question.
        top_k: Number of results to return. Defaults to settings.TOP_K_RESULTS.

    Returns:
        List of RetrievedChunk ordered by relevance (most relevant first).
    """
    k = top_k or settings.TOP_K_RESULTS
    embedder = get_embedder()

    # Use the embedder's query-specific method
    query_vector = embedder.embed_query(query)

    return search(query_embedding=query_vector, top_k=k)


# ── Utility ───────────────────────────────────────────────────────────────────

def index_size() -> int:
    """Return the number of documents currently in the ChromaDB collection."""
    try:
        return _get_collection().count()
    except Exception:
        return 0


def collection_is_populated() -> bool:
    """Check if the ChromaDB collection already has data (skip re-indexing)."""
    try:
        return _get_collection().count() > 0
    except Exception:
        return False
