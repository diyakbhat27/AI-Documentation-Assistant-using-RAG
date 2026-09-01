"""
app/rag/generator.py
─────────────────────────────────────────────────────────────────────────────
Multi-provider LLM answer generator for the RAG pipeline.

Supported providers (set via LLM_PROVIDER in .env):
  • "gemini" — Google Gemini API (gemini-1.5-flash default)
  • "openai" — OpenAI Chat Completions (gpt-4o-mini default)
  • "groq"  — Groq API (llama-3.3-70b-versatile default, FREE)

Flow:
  1. retrieve() fetches the top-K relevant chunks.
  2. build_prompt() assembles the system + user prompt with context.
  3. The LLM generates a grounded answer with inline citations.
  4. Source URLs are deduplicated and returned alongside the answer.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.core.settings import settings
from app.rag.retriever import RetrievedChunk


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    answer: str
    sources: List[dict]     # [{"url": ..., "title": ...}, ...]
    provider: str
    chunks_used: int


# ── Prompt builder ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a documentation assistant for LioranDB.

Answer the user's question using ONLY the documentation
context provided below. Do not use any outside knowledge.

Rules:
- If the answer is clearly in the context, answer completely and concisely
- Include relevant code examples exactly as they appear in the context
- If the answer is NOT in the context, respond with exactly:
  "I couldn't find this in the LioranDB documentation."
- Always end your response with a Sources section listing
  only the URLs you actually used to answer
- Format code using markdown code blocks with the language specified
- Do not make up information not present in the context"""

def _build_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        block = (
            f"[{i}] Page: {chunk.title} | Section: {chunk.heading}\n"
            f"{chunk.text}\n"
            f"Source: {chunk.url}"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)
    return f"Documentation Context:\n{context}\n\nQuestion: {question}"


# ── Gemini generator ──────────────────────────────────────────────────────────

def _generate_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    try:
        response = client.models.generate_content(
            model=settings.GEMINI_LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )
        return response.text.strip()
    except Exception as e:
        import traceback
        print(f"[generator] Gemini error: {e}")
        traceback.print_exc()
        return "Sorry, I encountered an error generating a response. Please try again."


# ── OpenAI generator ──────────────────────────────────────────────────────────

def _generate_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = client.chat.completions.create(
            model=settings.OPENAI_LLM_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[generator] OpenAI error: {e}")
        return "Sorry, I encountered an error generating a response. Please try again."


# ── Groq generator (free tier, OpenAI-compatible API) ─────────────────────────

def _generate_groq(prompt: str) -> str:
    import httpx

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.GROQ_LLM_MODEL,
                "messages": messages,
                "temperature": 0.2,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        import traceback
        print(f"[generator] Groq error: {e}")
        traceback.print_exc()
        return "Sorry, I encountered an error generating a response. Please try again."


# ── Public generate function ──────────────────────────────────────────────────

def generate_answer(question: str, chunks: List[RetrievedChunk]) -> RAGResponse:
    """
    Generate a grounded answer for `question` using the retrieved `chunks`.

    Args:
        question: The user's natural-language question.
        chunks:   Top-K chunks returned by retriever.retrieve().

    Returns:
        RAGResponse with the answer text, deduplicated sources, and metadata.
    """
    prompt = _build_prompt(question, chunks)
    provider = settings.LLM_PROVIDER.lower()

    if provider == "gemini":
        answer = _generate_gemini(prompt)
    elif provider == "openai":
        answer = _generate_openai(prompt)
    elif provider == "groq":
        answer = _generate_groq(prompt)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}. Choose 'gemini', 'openai', or 'groq'."
        )

    # Deduplicate source URLs (preserve order)
    seen_urls: set[str] = set()
    sources: List[dict] = []
    for chunk in chunks:
        if chunk.url and chunk.url not in seen_urls:
            seen_urls.add(chunk.url)
            sources.append({"url": chunk.url, "title": chunk.title})

    return RAGResponse(
        answer=answer,
        sources=sources,
        provider=provider,
        chunks_used=len(chunks),
    )
