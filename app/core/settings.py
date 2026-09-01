"""
app/core/settings.py
─────────────────────────────────────────────────────────────────────────────
Centralized configuration loader.

• Reads .env once via python-dotenv.
• Validates that the required API key is present at startup.
• Exposes a single `settings` object imported everywhere — zero hardcoded
  values anywhere else in the codebase.
─────────────────────────────────────────────────────────────────────────────
"""
import os
import sys
from dotenv import load_dotenv

# Load .env file from the project root
load_dotenv()


class Settings:
    """Application-wide configuration, populated from environment variables."""

    # ── LLM Provider ────────────────────────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")

    # ── Gemini ──────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_LLM_MODEL: str = os.getenv("GEMINI_LLM_MODEL", "gemini-1.5-flash")
    GEMINI_EMBEDDING_MODEL: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"
    )

    # ── OpenAI ──────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    # ── Groq (free tier) ────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_LLM_MODEL: str = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")

    # ── Embedding Provider ──────────────────────────────────────────────────
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
    LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── ChromaDB Persistence ────────────────────────────────────────────────
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "liorandb_docs"
    )

    # ── Chunking ────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # ── Retrieval ───────────────────────────────────────────────────────────
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

    # ── Cache ───────────────────────────────────────────────────────────────
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./cache")

    # ── Docs / Sitemap ──────────────────────────────────────────────────────
    DOCS_BASE_URL: str = os.getenv(
        "DOCS_BASE_URL", "https://db.lioransolutions.com"
    )
    SITEMAP_URL: str = os.getenv(
        "SITEMAP_URL", "https://db.lioransolutions.com/sitemap.xml"
    )

    # ── SQLite Database ─────────────────────────────────────────────────────
    SQLITE_DB: str = os.getenv("SQLITE_DB", "./data/query_logs.db")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/query_logs.db")
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", "")

    def validate(self) -> None:
        """
        Fail fast with a clear, human-readable error if the required
        API key is missing for the chosen LLM provider.

        Called once on FastAPI startup so misconfigurations are caught
        immediately — not buried in a cryptic traceback minutes later.
        """
        provider = self.LLM_PROVIDER.lower()

        if provider == "gemini":
            key = self.GEMINI_API_KEY.strip()
            if not key or "REPLACE" in key:
                print(
                    "\n"
                    "╔══════════════════════════════════════════════════════════╗\n"
                    "║  ERROR: GEMINI_API_KEY is not set or is a placeholder.  ║\n"
                    "║                                                          ║\n"
                    "║  1. Open your .env file                                  ║\n"
                    "║  2. Set GEMINI_API_KEY=<your-key>                        ║\n"
                    "║  3. Get a free key: https://aistudio.google.com/app/apikey║\n"
                    "╚══════════════════════════════════════════════════════════╝\n",
                    file=sys.stderr,
                )
                raise SystemExit(1)

        elif provider == "groq":
            key = self.GROQ_API_KEY.strip()
            if not key or "REPLACE" in key:
                print(
                    "\n"
                    "╔══════════════════════════════════════════════════════════╗\n"
                    "║  ERROR: GROQ_API_KEY is not set.                        ║\n"
                    "║                                                          ║\n"
                    "║  1. Open your .env file                                  ║\n"
                    "║  2. Set GROQ_API_KEY=<your-key>                          ║\n"
                    "║  3. Get a free key: https://console.groq.com/keys       ║\n"
                    "╚══════════════════════════════════════════════════════════╝\n",
                    file=sys.stderr,
                )
                raise SystemExit(1)

        elif provider == "openai":
            key = self.OPENAI_API_KEY.strip()
            if not key or "REPLACE" in key:
                print(
                    "\n"
                    "╔══════════════════════════════════════════════════════════╗\n"
                    "║  ERROR: OPENAI_API_KEY is not set.                      ║\n"
                    "║                                                          ║\n"
                    "║  1. Open your .env file                                  ║\n"
                    "║  2. Set OPENAI_API_KEY=<your-key>                        ║\n"
                    "╚══════════════════════════════════════════════════════════╝\n",
                    file=sys.stderr,
                )
                raise SystemExit(1)


# Single shared instance used throughout the app
settings = Settings()
