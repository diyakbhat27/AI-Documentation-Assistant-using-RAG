"""
app/database/logger.py
─────────────────────────────────────────────────────────────────────────────
Phase 7: SQLite Logging

Minimal SQLite query logger with no ORM dependencies.
Logs all RAG queries directly into a local SQLite database.
All failures are caught silently to never disrupt the user experience.
─────────────────────────────────────────────────────────────────────────────
"""
import os
import json
import sqlite3
from typing import List, Optional

# ── Setup ──────────────────────────────────────────────────────────────────────

# Ensure data directory exists on import
os.makedirs("data", exist_ok=True)

DB_PATH = "data/query_logs.db"

def init_db() -> None:
    """Create the query_logs table if it doesn't already exist."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS query_logs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    question    TEXT NOT NULL,
                    answer      TEXT NOT NULL,
                    sources     TEXT,        -- JSON-encoded list of source URLs
                    chunks      TEXT,        -- JSON-encoded list of retrieved chunk IDs
                    response_ms INTEGER,
                    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            ''')
            # Optional: recreate api_keys table for /api/settings if needed
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT UNIQUE NOT NULL,
                    api_key TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            ''')
            conn.commit()
    except Exception as exc:
        print(f"[logger] Failed to initialize DB: {exc}")


def log_query(
    question: str,
    answer: str,
    sources: List[str],
    chunks: List[str],
    response_ms: int,
) -> None:
    """
    Persist a completed RAG query to the query_logs table.
    Failures are caught and printed so they don't break the API response.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO query_logs (question, answer, sources, chunks, response_ms)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                question,
                answer,
                json.dumps(sources),
                json.dumps(chunks),
                response_ms
            ))
            conn.commit()
    except Exception as exc:
        print(f"[logger] Failed to log query: {exc}")

# Initialize table on import
init_db()
