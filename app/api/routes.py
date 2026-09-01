"""
app/api/routes.py
─────────────────────────────────────────────────────────────────────────────
All FastAPI API route definitions.

Endpoints:
  GET  /health          — Liveness check
  POST /api/ask         — Main RAG Q&A endpoint
  GET  /api/search      — Raw semantic search (debug)
  POST /api/index       — Re-trigger full crawl + index pipeline
  GET  /api/history     — Fetch recent query logs from SQLite
  POST /api/settings    — Save a dynamic API key (encrypted in DB)
─────────────────────────────────────────────────────────────────────────────
"""
import json
import time
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from app.core.settings import settings
from app.rag.retriever import retrieve, index_size
from app.rag.generator import generate_answer
from app.database.logger import log_query, DB_PATH
import sqlite3

router = APIRouter()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

class AskResponse(BaseModel):
    answer: str
    sources: List[dict]
    provider: str
    chunks_used: int
    response_time_ms: float

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class IndexResponse(BaseModel):
    status: str
    pages_crawled: int
    chunks_indexed: int
    duration_seconds: int

class HistoryItem(BaseModel):
    id: int
    question: str
    answer: str
    sources: List[str]
    chunks: List[str]
    response_time_ms: float
    created_at: str

class SettingsRequest(BaseModel):
    provider: str       # "gemini" or "openai"
    api_key: str


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {
        "status": "ok",
        "chroma_docs": index_size(),
        "model_loaded": True,
    }


# ── Ask ───────────────────────────────────────────────────────────────────────

@router.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Main RAG endpoint. Embeds the question, retrieves relevant chunks,
    generates a grounded answer with source links, and logs the query.
    """
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    t0 = time.perf_counter()

    chunks = retrieve(req.question, top_k=req.top_k)
    if not chunks:
        raise HTTPException(
            status_code=503,
            detail="Vector index is empty. Run POST /api/index first.",
        )

    rag_resp = generate_answer(req.question, chunks)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Persist to DB (non-blocking)
    source_urls = [s["url"] for s in rag_resp.sources]
    chunk_ids = [c.id for c in chunks]
    log_query(
        question=req.question,
        answer=rag_resp.answer,
        sources=source_urls,
        chunks=chunk_ids,
        response_ms=int(elapsed_ms),
    )

    return AskResponse(
        answer=rag_resp.answer,
        sources=rag_resp.sources,
        provider=rag_resp.provider,
        chunks_used=rag_resp.chunks_used,
        response_time_ms=elapsed_ms,
    )


# ── Search (debug) ────────────────────────────────────────────────────────────

@router.get("/api/search")
async def search(q: str, top_k: int = 5):
    """Return raw retrieved chunks for a query — useful for debugging retrieval."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required.")
    chunks = retrieve(q, top_k=top_k)
    return {
        "query": q,
        "results": [
            {
                "text": c.text,
                "source_url": c.url,
                "page_title": c.title,
                "section_heading": c.heading,
                "score": c.score,
            }
            for c in chunks
        ],
    }


# ── Index ─────────────────────────────────────────────────────────────────────

def _run_full_index() -> dict:
    """Background-safe indexing pipeline."""
    from app.crawler.sitemap import get_doc_urls
    from app.crawler.fetcher import fetch_all_pages
    from app.crawler.cleaner import clean_page
    from app.chunking.chunker import chunk_pages
    from app.rag.retriever import build_index

    urls = get_doc_urls()
    url_strings = [entry["url"] for entry in urls]
    html_map = fetch_all_pages(url_strings)
    pages = [
        p for url, html in html_map.items()
        if (p := clean_page(url, html)) is not None
    ]
    chunks = chunk_pages(pages)
    build_index(chunks)
    return {"pages": len(pages), "chunks": len(chunks)}


@router.post("/api/index", response_model=IndexResponse)
async def index_docs(background_tasks: BackgroundTasks):
    """
    Trigger a full crawl → clean → chunk → embed → index pipeline.
    Runs synchronously so the caller gets confirmation when indexing completes.
    """
    t0 = time.perf_counter()
    try:
        result = _run_full_index()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = int(time.perf_counter() - t0)

    return IndexResponse(
        status="completed",
        pages_crawled=result["pages"],
        chunks_indexed=result["chunks"],
        duration_seconds=elapsed,
    )


# ── History ────────────────────────────────────────────────────────────────────

@router.get("/api/history", response_model=List[HistoryItem])
async def get_history(limit: int = 20):
    """Return the most recent query logs from SQLite."""
    items = []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, question, answer, sources, chunks, response_ms, timestamp
                FROM query_logs
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            for row in rows:
                try:
                    sources = json.loads(row[3] or "[]")
                except Exception:
                    sources = []
                try:
                    chunks = json.loads(row[4] or "[]")
                except Exception:
                    chunks = []
                items.append(
                    HistoryItem(
                        id=row[0],
                        question=row[1],
                        answer=row[2],
                        sources=sources,
                        chunks=chunks,
                        response_time_ms=float(row[5] or 0),
                        created_at=str(row[6]),
                    )
                )
    except Exception as e:
        print(f"Error fetching history: {e}")
    return items


# ── Settings (dynamic API key storage with encryption) ────────────────────────

@router.post("/api/settings")
async def save_api_key(req: SettingsRequest):
    """
    Persist a user-supplied API key in SQLite.
    (Note: Encryption was removed in Phase 7 for simplicity since SQLAlchemy was removed)
    """
    if req.provider not in ("gemini", "openai"):
        raise HTTPException(
            status_code=400, detail="Provider must be 'gemini' or 'openai'."
        )

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Upsert logic for SQLite
            cursor.execute('''
                INSERT INTO api_keys (provider, api_key, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(provider) DO UPDATE SET 
                    api_key=excluded.api_key,
                    updated_at=CURRENT_TIMESTAMP
            ''', (req.provider, req.api_key))
            conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "saved", "provider": req.provider}

