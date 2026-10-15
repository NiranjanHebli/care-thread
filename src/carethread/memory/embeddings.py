"""Embedding service supporting local FastEmbed (BAAI/bge-small-en-v1.5) and HashEmbedder.

Dense text embeddings using FastEmbed or deterministic hash generator:
- 384-dimensional dense vectors
- FastEmbed with local ONNX model
- Deterministic HashEmbedder for fast offline unit tests
- Batch computation with configurable batch size
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

EMBEDDING_DIM: int = 384
DEFAULT_MODEL: str = "BAAI/bge-small-en-v1.5"


@runtime_checkable
class Embedder(Protocol):
    """Protocol for text embedding models."""

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic hash-based embedder for fast, reproducible unit tests.

    Produces normalized unit vectors of dimension 384 without downloading weights.
    """

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self._dim

        # Produce pseudo-random deterministic floats from SHA-256 rounds
        vector: list[float] = []
        seed = text.encode("utf-8")
        block_idx = 0

        while len(vector) < self._dim:
            h = hashlib.sha256(seed + block_idx.to_bytes(4, "big")).digest()
            for i in range(0, len(h), 4):
                if len(vector) >= self._dim:
                    break
                val = int.from_bytes(h[i : i + 4], "big", signed=True)
                vector.append(float(val) / (2**31))
            block_idx += 1

        # Normalize to unit length (L2 norm = 1.0) for cosine similarity
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 1e-9:
            vector = [x / norm for x in vector]
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class FastEmbedder:
    """Local ONNX embeddings using fastembed (BAAI/bge-small-en-v1.5)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dimension: int = EMBEDDING_DIM,
    ) -> None:
        self._model_name = model_name
        self._dim = dimension
        self._model = None

    @property
    def dimension(self) -> int:
        return self._dim

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            logger.info(f"Loading FastEmbed model '{self._model_name}'...")
            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        batch_results = self.embed_batch([text])
        return batch_results[0] if batch_results else [0.0] * self._dim

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            model = self._get_model()
            embeddings_iter = model.embed(texts)
            results: list[list[float]] = []
            for emb in embeddings_iter:
                # Convert numpy array to list of floats
                results.append([float(x) for x in emb])
            return results
        except Exception as e:
            logger.warning(f"FastEmbed failed ({e}); falling back to deterministic HashEmbedder.")
            fallback = HashEmbedder(dimension=self._dim)
            return fallback.embed_batch(texts)


_EMBEDDER_INSTANCE: Embedder | None = None


def get_embedder(mode: str | None = None) -> Embedder:
    """Returns the configured embedding engine.

    Modes:
    - 'hash': Fast deterministic hash embedder (used for unit tests and CI).
    - 'fastembed': Local ONNX bge-small-en-v1.5 model.
    - 'auto': Checks CT_EMBED_MODE or CT_LLM_MODE env vars.
    """
    resolved_mode = mode or os.getenv("CT_EMBED_MODE", "").strip().lower()

    if not resolved_mode:
        llm_mode = os.getenv("CT_LLM_MODE", "").strip().lower()
        if llm_mode in ("mock", "test"):
            resolved_mode = "hash"
        else:
            resolved_mode = "auto"

    if resolved_mode in ("hash", "test", "mock"):
        return HashEmbedder()

    # FastEmbed with singleton caching
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        _EMBEDDER_INSTANCE = FastEmbedder()
    return _EMBEDDER_INSTANCE
