"""
app/rag/embedder.py
─────────────────────────────────────────────────────────────────────────────
Phase 4: Embedding Service

Multi-provider embedding service with singleton model loading.

Supported providers (set via EMBEDDING_PROVIDER in .env):
  * "local"  -- sentence-transformers all-MiniLM-L6-v2 (offline, no API key)
  * "gemini" -- Google text-embedding-004 API
  * "openai" -- OpenAI text-embedding-3-small API

Key design decisions:
  * Model is loaded ONCE at first call via singleton pattern
  * Never reloaded per request -- would be extremely slow
  * Batch size capped at 64 for memory safety on free-tier hosts
  * Two clean public methods: embed_chunks() and embed_query()

Usage:
    from app.rag.embedder import get_embedder
    embedder = get_embedder()
    vectors = embedder.embed_chunks(["chunk 1", "chunk 2"])
    query_vec = embedder.embed_query("how do I create an index?")
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.core.settings import settings


# ── Constants ─────────────────────────────────────────────────────────────────

EMBED_BATCH_SIZE: int = 64   # Memory safe for Render free tier


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """Base class for all embedding providers."""

    @abstractmethod
    def embed_chunks(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embed for indexing. Processes texts in batches of EMBED_BATCH_SIZE.

        Args:
            texts: List of chunk text strings.

        Returns:
            List of embedding vectors (one per input text).
        """

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Single embed for search queries.

        Args:
            text: The user's search query.

        Returns:
            Single embedding vector.
        """

    # Legacy aliases for backward compatibility
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Legacy alias for embed_chunks."""
        return self.embed_chunks(texts)

    def embed_one(self, text: str) -> List[float]:
        """Legacy alias for embed_query."""
        return self.embed_query(text)


# ── Local (sentence-transformers) ─────────────────────────────────────────────

class LocalEmbedder(BaseEmbedder):
    """
    Runs all-MiniLM-L6-v2 (or any sentence-transformers model) on CPU.

    The model is loaded once in __init__ and reused for all subsequent calls.
    Batch encoding uses EMBED_BATCH_SIZE to stay within RAM limits.
    """

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        model_name = settings.LOCAL_EMBEDDING_MODEL
        print(f"[embedder] Loading local model: {model_name} ...")
        self._model = SentenceTransformer(model_name)
        print(f"[embedder] Local model ready ({model_name}).")

    def embed_chunks(self, texts: List[str]) -> List[List[float]]:
        """Batch embed with EMBED_BATCH_SIZE for memory safety."""
        all_vectors: List[List[float]] = []
        total = len(texts)

        for batch_start in range(0, total, EMBED_BATCH_SIZE):
            batch = texts[batch_start : batch_start + EMBED_BATCH_SIZE]
            vectors = self._model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=EMBED_BATCH_SIZE,
            )
            all_vectors.extend(v.tolist() for v in vectors)

        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        vector = self._model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector[0].tolist()


# ── Google Gemini ─────────────────────────────────────────────────────────────

class GeminiEmbedder(BaseEmbedder):
    """Uses Google's text-embedding-004 model via the Gemini API."""

    def __init__(self) -> None:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self._genai = genai
        self._model = settings.GEMINI_EMBEDDING_MODEL
        print(f"[embedder] Gemini embedder ready (model={self._model}).")

    def embed_chunks(self, texts: List[str]) -> List[List[float]]:
        """Embed texts one at a time using retrieval_document task type."""
        results: List[List[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            for text in batch:
                resp = self._genai.embed_content(
                    model=self._model,
                    content=text,
                    task_type="retrieval_document",
                )
                results.append(resp["embedding"])
        return results

    def embed_query(self, text: str) -> List[float]:
        """Use retrieval_query task type for query-side embeddings."""
        resp = self._genai.embed_content(
            model=self._model,
            content=text,
            task_type="retrieval_query",
        )
        return resp["embedding"]


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIEmbedder(BaseEmbedder):
    """Uses OpenAI's text-embedding-3-small model."""

    def __init__(self) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.OPENAI_EMBEDDING_MODEL
        print(f"[embedder] OpenAI embedder ready (model={self._model}).")

    def embed_chunks(self, texts: List[str]) -> List[List[float]]:
        """Batch embed using OpenAI's batch API."""
        results: List[List[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            results.extend(item.embedding for item in resp.data)
        return results

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        resp = self._client.embeddings.create(model=self._model, input=[text])
        return resp.data[0].embedding


# ── Singleton factory ─────────────────────────────────────────────────────────

_embedder_instance: BaseEmbedder | None = None


def get_embedder() -> BaseEmbedder:
    """
    Return the singleton embedder for the configured EMBEDDING_PROVIDER.

    The model is loaded once on first call and reused for all subsequent
    requests. This avoids the multi-second model load on every request.
    """
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance

    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider == "local":
        _embedder_instance = LocalEmbedder()
    elif provider == "gemini":
        _embedder_instance = GeminiEmbedder()
    elif provider == "openai":
        _embedder_instance = OpenAIEmbedder()
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER={provider!r}. "
            "Choose 'local', 'gemini', or 'openai'."
        )
    return _embedder_instance
