"""
scripts/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Runs 10 representative test questions through the live RAG pipeline and
writes the full outputs (answer + sources + latency) to tests/test_results.md.

Run from the project root (server must be running):
    python scripts/evaluate.py

OR against a remote URL:
    BASE_URL=https://your-app.onrender.com python scripts/evaluate.py
─────────────────────────────────────────────────────────────────────────────
"""
import os
import sys
import json
import time
from datetime import datetime

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "test_results.md",
)

# ── 10 evaluation questions ────────────────────────────────────────────────────

QUESTIONS = [
    "What is LioranDB and what problem does it solve?",
    "How do I install and get started with LioranDB using the embedded mode?",
    "How do I create a collection and insert documents in LioranDB?",
    "What query and update operators are supported in LioranDB?",
    "How do transactions work in LioranDB? Are they ACID-compliant?",
    "How do I set up the LioranDB server and CLI tools?",
    "How does replication work in LioranDB?",
    "How do I use the LioranDB TypeScript driver to connect to a server?",
    "How do I integrate LioranDB with a NestJS application?",
    "What encryption options are available in LioranDB?",
]


def ask(question: str) -> dict:
    resp = requests.post(
        f"{BASE_URL}/api/ask",
        json={"question": question},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    print(f"[evaluate] Running {len(QUESTIONS)} test questions against {BASE_URL} ...\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    lines = [
        "# LioranDB AI Assistant — Evaluation Results\n",
        f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  \n",
        f"**Base URL:** {BASE_URL}  \n",
        f"**Total questions:** {len(QUESTIONS)}\n",
        "\n---\n",
    ]

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"[{i}/{len(QUESTIONS)}] {question}")
        try:
            t0 = time.perf_counter()
            result = ask(question)
            elapsed = round((time.perf_counter() - t0) * 1000, 1)

            answer = result.get("answer", "")
            sources = result.get("sources", [])
            provider = result.get("provider", "")
            api_ms = result.get("response_time_ms", elapsed)

            source_lines = "\n".join(
                f"  - [{s.get('title', s.get('url', ''))}]({s.get('url', '')})"
                for s in sources
            )

            lines += [
                f"## Q{i}. {question}\n",
                f"**Provider:** `{provider}` | **Response time:** {api_ms} ms\n",
                "\n### Answer\n",
                answer,
                "\n\n### Sources\n",
                source_lines or "_No sources returned._",
                "\n\n---\n",
            ]
            print(f"    [x] {api_ms} ms\n")

        except Exception as exc:
            lines += [
                f"## Q{i}. {question}\n",
                f"**ERROR:** {exc}\n\n---\n",
            ]
            print(f"    [FAILED] ERROR: {exc}\n")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[evaluate] [x] Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
