# LioranDB AI Assistant

## Overview
LioranDB AI Assistant is a RAG-powered chatbot designed to answer questions using the source documentation. It crawls the site, chunks the markdown, embeds it into a local ChromaDB vector store, and uses Google Gemini to generate grounded answers with source citations. This ensures users get fast, accurate, and verifiable answers directly from the documentation.

## Architecture
```text
Source Site
      ↓
Crawler (httpx + BeautifulSoup)
      ↓
HTML → Markdown (markdownify)
      ↓
Cache Layer (.md files)
      ↓
Chunker (800 char, 150 overlap)
      ↓
Embedder (all-MiniLM-L6-v2)
      ↓
ChromaDB (persistent vector store)
      ↓
RAG Engine (Gemini 1.5 Flash)
      ↓
FastAPI + Chat UI
```

## Prerequisites
- **Option 1**: Python 3.11+, Gemini API key
- **Option 2**: Docker + Docker Compose, Gemini API key

## Local Setup — Option 1 (Plain Python)
```bash
git clone <repo>
cd liorandb-ai-assistant
cp .env.example .env
# Add your GEMINI_API_KEY to .env

pip install -r requirements.txt
python scripts/ingest.py
uvicorn main:app --reload
# Open http://localhost:8000
```

## Local Setup — Option 2 (Docker)
```bash
git clone <repo>
cd liorandb-ai-assistant
cp .env.example .env
# Add your GEMINI_API_KEY to .env

docker compose up --build
# Open http://localhost:8000
```

## API Reference
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Chat UI |
| GET | `/health` | System status |
| POST | `/api/ask` | Ask a question |
| GET | `/api/search?q=query` | Raw chunk retrieval |
| POST | `/api/index` | Re-index all docs |

## Notes for Evaluators
- First run of `ingest.py` downloads MiniLM model (~90MB). Cached automatically after first download.
- Docker build also pre-downloads the model. Subsequent builds use Docker cache — much faster.
- `POST /api/index` re-crawls everything (~2 minutes). Only needed if docs have changed.
- `GET /api/search?q=query` shows raw retrieval results. Useful for inspecting what the system finds.
- All queries logged to `data/query_logs.db`. Open with any SQLite viewer to inspect history.
- Out of scope questions return: "I couldn't find this in the LioranDB documentation."


