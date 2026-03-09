# src/liagent/agent/embedder.py
"""Embedding provider abstraction: MLX local -> OpenAI API -> graceful degradation."""

from __future__ import annotations

import hashlib
import re
import time
from typing import Protocol, runtime_checkable

import numpy as np

from ..logging import get_logger

_log = get_logger("embedder")


# --- Protocol -----------------------------------------------------------------

@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...
    @property
    def dimensions(self) -> int: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


# --- Query type classification ------------------------------------------------

_EXACT_PATTERNS = [
    re.compile(r"^[A-Z]{1,5}$"),                          # Tickers: AAPL, MSFT
    re.compile(r"^[A-Z][A-Z0-9_]{3,}$"),                  # Error codes: ERR_CONNECTION
    re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}"),               # UUIDs
    re.compile(r"^[A-Z0-9_./-]+$"),                        # Pure identifier tokens
]

_SEMANTIC_INDICATORS = [
    "how", "what", "why", "explain", "configure", "setup",
    "describe", "discuss", "previously", "remember",
    "earlier", "plan", "design",
]


def classify_query_type(query: str) -> str:
    """Classify query as 'exact', 'semantic', or 'mixed'.

    exact:    tickers, error codes, UUIDs, identifiers
    semantic: natural language questions, references to past context
    mixed:    default when both signals present or neither dominant
    """
    q = query.strip()
    if not q:
        return "mixed"

    exact_signal = any(p.search(q) for p in _EXACT_PATTERNS)
    semantic_signal = any(ind in q.lower() for ind in _SEMANTIC_INDICATORS)

    if exact_signal and not semantic_signal:
        return "exact"
    if semantic_signal and not exact_signal:
        return "semantic"
    return "mixed"



# --- MLX Embedder -------------------------------------------------------------

class MLXEmbedder:
    """Local MLX embedding model (bge-small-zh-v1.5, 512 dims, ~130MB).

    Lazy-loaded on first encode(). Not thread-safe; callers must synchronize.
    """

    DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

    def __init__(self, model_path: str | None = None):
        self._model_path = model_path or self.DEFAULT_MODEL
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._dimensions = 512

    @property
    def model_name(self) -> str:
        return self._model_path.split("/")[-1] if "/" in self._model_path else self._model_path

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _load(self):
        if self._loaded:
            return
        try:
            import mlx.core as mx
            from transformers import AutoTokenizer, AutoModel
            _log.event("mlx_embedder_loading", model=self._model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModel.from_pretrained(self._model_path, trust_remote_code=True)
            self._loaded = True
            _log.event("mlx_embedder_loaded", model=self._model_path)
        except Exception as e:
            _log.error("mlx_embedder_load_failed", e)
            raise

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        import torch
        encoded = self._tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        with torch.no_grad():
            output = self._model(**encoded)
        embeddings = output.last_hidden_state[:, 0, :].numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return embeddings / norms


# --- OpenAI Embedder ----------------------------------------------------------

class OpenAIEmbedder:
    """OpenAI text-embedding-3-small API fallback."""

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        import os
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._dimensions = 1536

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        import httpx
        resp = httpx.post(
            "https://api.openai.com/v1/embeddings",
            json={"input": texts, "model": self._model},
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        vecs = [np.array(d["embedding"], dtype=np.float32) for d in data]
        if vecs and len(vecs[0]) != self._dimensions:
            _log.warning(
                f"openai_dimension_mismatch: expected {self._dimensions}, "
                f"got {len(vecs[0])}"
            )
        return np.stack(vecs)


# --- EmbedderChain ------------------------------------------------------------

class EmbedderChain:
    """Try providers in order. First success wins. All fail -> returns None."""

    def __init__(self, providers: list | None = None):
        self._providers: list = providers or []
        self._active: EmbeddingProvider | None = None
        self._last_success_ts: float = 0.0
        self._last_failure_ts: float = 0.0

    @property
    def model_name(self) -> str:
        return self._active.model_name if self._active else "none"

    @property
    def dimensions(self) -> int:
        return self._active.dimensions if self._active else 0

    def encode(self, texts: list[str]) -> np.ndarray | None:
        if self._active is not None:
            try:
                result = self._active.encode(texts)
                self._last_success_ts = time.time()
                return result
            except Exception as e:
                _log.warning(f"embedder_active_failed: {self._active.model_name}: {e}")
                self._active = None
                self._last_failure_ts = time.time()

        for p in self._providers:
            try:
                result = p.encode(texts)
                self._active = p
                self._last_success_ts = time.time()
                _log.event("embedder_fallback_success", provider=p.model_name)
                return result
            except Exception as e:
                _log.warning(f"embedder_provider_failed: {p.model_name}: {e}")
                continue

        self._last_failure_ts = time.time()
        _log.warning("embedder_all_failed, returning None")
        return None
