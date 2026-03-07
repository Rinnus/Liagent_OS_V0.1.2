"""Abstract base classes for LLM and TTS backends."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class _MissingNumpy:
    """Lazy numpy placeholder so imports work in minimal test environments."""

    ndarray = Any

    def __getattr__(self, _name: str) -> Any:
        raise ModuleNotFoundError(
            "numpy is required for audio array operations in liagent.engine"
        )


try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal CI env only
    np = _MissingNumpy()  # type: ignore[assignment]


class LLMBackend(ABC):
    """Unified interface for language model inference."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream-generate text from messages. Yields token strings."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    def unload(self):
        """Release model resources."""
        pass


class TTSBackend(ABC):
    """Unified interface for text-to-speech."""

    @abstractmethod
    async def synthesize(self, text: str) -> np.ndarray:
        """Generate audio waveform from text. Returns float32 numpy array."""
        ...

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        """Stream audio chunks. Default: yield single synthesize() result."""
        yield await self.synthesize(text)

    def set_speaker(self, name: str):
        """Switch speaker identity (for embedding-based TTS). Optional override."""
        pass

    def unload(self):
        pass
