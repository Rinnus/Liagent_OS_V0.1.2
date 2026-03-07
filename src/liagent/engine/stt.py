"""Speech-to-text engine with local MLX and API backends."""

import io
import os
import wave
from typing import Any


class _MissingNumpy:
    """Lazy numpy placeholder so module import works in minimal test envs."""

    ndarray = Any

    def __getattr__(self, _name: str) -> Any:
        raise ModuleNotFoundError(
            "numpy is required for STT waveform array handling in liagent.engine.stt"
        )


try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal CI env only
    np = _MissingNumpy()  # type: ignore[assignment]

from .runtime import run_mlx_serialized


def _normalize_api_language(raw: str | None) -> str | None:
    value = (raw or "").strip().lower()
    if not value or value == "auto":
        return None
    if value in {"zh", "chinese", "zh-cn", "mandarin"}:
        return "zh"
    if value in {"en", "english", "en-us", "en-gb"}:
        return "en"
    return value


class STTEngine:
    def __init__(
        self,
        model: str,
        language: str = "auto",
        *,
        backend: str = "local",
        api_base_url: str = "",
        api_key: str = "",
        api_model: str = "",
    ):
        self.model_path = model
        self.language = language
        self.backend = str(backend or "local").strip().lower()
        if self.backend not in {"local", "api"}:
            self.backend = "local"
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.api_model = api_model
        self.timeout_sec = max(
            3.0, float(os.environ.get("LIAGENT_STT_TIMEOUT_SEC", "25"))
        )
        self.sample_rate_hz = int(os.environ.get("LIAGENT_STT_SAMPLE_RATE", "16000"))
        self._model = None
        self._api_client = None

    def _ensure_loaded(self):
        """Lazy-load local ASR model on first use."""
        if self._model is None:
            from mlx_audio.stt.utils import load_model

            self._model = load_model(self.model_path)

    def _ensure_api_client(self):
        if self._api_client is None:
            if not self.api_key:
                raise ValueError("STT API backend requires api_key")
            from openai import AsyncOpenAI

            kwargs = {"api_key": self.api_key}
            if self.api_base_url:
                kwargs["base_url"] = self.api_base_url
            self._api_client = AsyncOpenAI(**kwargs)
        return self._api_client

    @staticmethod
    def _to_wav_bytes(audio: np.ndarray, sample_rate_hz: int) -> bytes:
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return b""
        pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate_hz)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    async def _transcribe_api(self, audio: str | np.ndarray, language: str | None) -> str:
        model = str(self.api_model or "").strip()
        if not model:
            raise ValueError("STT API backend requires api_model")

        client = self._ensure_api_client()
        lang = _normalize_api_language(language or self.language)

        if isinstance(audio, str):
            with open(audio, "rb") as fh:
                resp = await client.audio.transcriptions.create(
                    model=model,
                    file=fh,
                    language=lang,
                )
        else:
            wav_bytes = self._to_wav_bytes(audio, self.sample_rate_hz)
            if not wav_bytes:
                return ""
            fh = io.BytesIO(wav_bytes)
            fh.name = "audio.wav"
            resp = await client.audio.transcriptions.create(
                model=model,
                file=fh,
                language=lang,
            )

        text = getattr(resp, "text", "")
        if not text and isinstance(resp, dict):
            text = resp.get("text", "")
        return str(text or "").strip()

    async def _transcribe_local(self, audio: str | np.ndarray, language: str | None) -> str:
        lang = language or self.language

        def _do():
            import mlx.core as mx

            mx.clear_cache()
            try:
                self._ensure_loaded()
                result = self._model.generate(audio, language=lang)
                return result.text.strip()
            finally:
                mx.clear_cache()

        return await run_mlx_serialized(_do, timeout_sec=self.timeout_sec)

    def unload(self):
        if self.backend == "local" and self._model is not None:
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass
        self._model = None
        self._api_client = None

    async def transcribe(self, audio: str | np.ndarray, language: str | None = None) -> str:
        """Transcribe audio file path or numpy array to text."""
        if self.backend == "api":
            return await self._transcribe_api(audio, language)
        return await self._transcribe_local(audio, language)
