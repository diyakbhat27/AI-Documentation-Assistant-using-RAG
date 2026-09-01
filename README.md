# LioranDB AI Assistant

## Overview
LioranDB AI Assistant is a RAG-powered chatbot designed to answer questions using the official LioranDB documentation. It crawls the site, chunks the markdown, embeds it into a local ChromaDB vector store, and uses Google Gemini to generate grounded answers with source citations. This ensures users get fast, accurate, and verifiable answers directly from the documentation.

## Architecture
```text
LioranDB Docs Site
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

## Sample Q&A
Here are 10 sample questions you can ask the assistant to evaluate its capabilities:

1. **Q:** How do I create a table in LioranDB?
   **A:** The assistant will provide the `CREATE TABLE` syntax, complete with data type explanations and source citations.

2. **Q:** What data types does LioranDB support?
   **A:** It will list supported types such as `INT`, `VARCHAR`, and `TEXT` referencing the SQL Reference guide.

3. **Q:** How can I configure the embedded manager?
   **A:** You will receive step-by-step instructions for starting the manager and configuring its port.

4. **Q:** What are the system requirements for installing LioranDB?
   **A:** It details the required RAM, CPU, and OS compatibility based on the Getting Started documentation.

5. **Q:** How do I create an index to improve query performance?
   **A:** It returns the `CREATE INDEX` syntax and explains how it affects retrieval speed.

6. **Q:** Can I use LioranDB in a Docker container?
   **A:** Yes, it will provide the official `docker run` command and default volume mappings.

7. **Q:** How do I write a basic SELECT query with a WHERE clause?
   **A:** It outputs an example query highlighting basic operators (`=`, `>`, `<`).

8. **Q:** What is the difference between VARCHAR and TEXT?
   **A:** It will explain storage limitations and performance trade-offs from the documentation.

9. **Q:** How do I back up my database?
   **A:** It will provide the CLI backup command or backup utility process.

10. **Q:** What is the recipe for a chocolate cake?
    **A:** *Out of scope.* The assistant will reply exactly with: "I couldn't find this in the LioranDB documentation."
