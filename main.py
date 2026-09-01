"""
main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application entry point.

• Mounts /static folder for CSS and JS assets.
• Registers Jinja2 templates pointing to app/templates/.
• Includes API router from app/api/routes.py.
• GET /  → serves the chat UI (index.html).
• GET /health → returns system status JSON.
─────────────────────────────────────────────────────────────────────────────
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.settings import settings
from app.api.routes import router


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run one-time setup on startup, cleanup on shutdown."""
    # Ensure runtime directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)

    # Initialize database tables (creates them if they don't exist)
    from app.database.logger import init_db
    init_db()

    # Validate required environment variables — fail fast
    settings.validate()

    print(
        f"[startup] LioranDB AI Assistant ready  "
        f"| LLM={settings.LLM_PROVIDER}  "
        f"| Embedding model={settings.EMBEDDING_MODEL}"
    )
    yield
    # Shutdown (nothing to clean up)


# ── Application ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LioranDB AI Documentation Assistant",
    description=(
        "RAG-powered assistant that answers questions about LioranDB "
        "using the official documentation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Static files (CSS, JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Jinja2 templates (chat UI)
templates = Jinja2Templates(directory="app/templates")

# API routes
app.include_router(router)


# ── Chat UI ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    """Serve the single-page chat interface."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "LioranDB AI Assistant",
        },
    )


# ── Run directly ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
