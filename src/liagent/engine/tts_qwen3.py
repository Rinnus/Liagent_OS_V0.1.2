"""Qwen3-TTS-CustomVoice backend — the sole local TTS engine.

Uses preset speaker voices (Vivian, Serena, Ryan, etc.) via
generate_custom_voice(). No speaker embedding extraction needed.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import numpy as np

from .base import TTSBackend
from .runtime import run_mlx_serialized
from ..logging import get_logger

_log = get_logger("tts_qwen3")


def _clean_for_tts(text: str) -> str:
    """Clean text for TTS — keep prosody-relevant punctuation, remove noise."""
    import re
    from .tts_utils import clean_text_for_tts

    # Phase 1: strip emoji, markdown, special brackets (from tts_local)
    text = clean_text_for_tts(text)

    # Phase 2: normalize punctuation for stable synthesis
    # Keep ! ? . , and em dash because they help prosody.
    text = re.sub(r"[!]{2,}", "!", text)          # collapse repeated exclamation
    text = re.sub(r"[?]{2,}", "?", text)          # collapse repeated question mark
    text = re.sub(r"[.]{3,}", ",", text)          # ellipsis -> comma (unstable prosody)
    text = re.sub(r"[~～]+", "", text)             # remove tildes (causes pitch wobble)
    text = re.sub(r";", ",", text)                # semicolons -> comma
    text = re.sub(r":", ",", text)                # colons -> comma
    text = re.sub(r"[""\"'']+", "", text)          # strip quotes
    text = re.sub(r"[,]{2,}", ",", text)          # collapse multiple commas
    text = re.sub(r"\s+", " ", text).strip()

    # Ensure ending has sentence-final punctuation
    text = text.rstrip(", ")
    if text and text[-1] not in ".!?":
        text += "."
    return text

# Available preset speakers in CustomVoice model
PRESET_SPEAKERS = [
    "serena", "vivian", "uncle_fu", "ryan", "aiden",
    "ono_anna", "sohee", "eric", "dylan",
]
DEFAULT_SPEAKER = "serena"


class Qwen3TTS(TTSBackend):
    """Qwen3-TTS-CustomVoice via mlx-audio with preset speakers."""

    def __init__(
        self,
        model_path: str,
        *,
        speaker_name: str = DEFAULT_SPEAKER,
        language: str = "zh",
        speed: float = 1.0,
        temperature: float = 0.3,
        top_k: int = 20,
        top_p: float = 0.8,
        repetition_penalty: float = 1.05,
    ):
        self.model_path = model_path
        self.model = None
        self.speaker_name = self._normalize_speaker(speaker_name)
        self.language = str(language or "zh").strip().lower()
        self.speed = max(0.5, min(2.0, float(speed or 1.0)))
        # TTS temperature is NOT like LLM temperature — values above 0.5 cause
        # acoustic token sampling instability → noise bursts + timbre drift
        self.temperature = max(0.05, min(0.5, float(temperature or 0.3)))
        self.top_k = max(1, min(100, int(top_k or 20)))
        self.top_p = max(0.1, min(0.95, float(top_p or 0.8)))
        self.repetition_penalty = max(1.0, min(2.0, float(repetition_penalty or 1.05)))
        self.sample_rate: int = 24000
        self._loaded = False

        self.timeout_sec = max(4.0, float(os.environ.get("LIAGENT_TTS_TIMEOUT_SEC", "15")))
        self.warmup_timeout_sec = max(
            self.timeout_sec,
            float(os.environ.get("LIAGENT_TTS_WARMUP_TIMEOUT_SEC", "60")),
        )

    # ── Loading ──────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self.model is not None:
            self._loaded = True
            return
        from mlx_audio.tts.utils import load_model
        import transformers.utils.logging as tf_logging

        # Suppress harmless transformers warnings during load:
        # - "model of type qwen3_tts to instantiate a model of type ." (AutoConfig unknown type)
        # - "incorrect regex pattern... fix_mistral_regex" (false positive on Qwen2 tokenizer)
        prev_verbosity = tf_logging.get_verbosity()
        tf_logging.set_verbosity_error()
        try:
            self.model = load_model(self.model_path)
        finally:
            tf_logging.set_verbosity(prev_verbosity)
        sr = int(getattr(self.model, "sample_rate", 0) or 0)
        if sr > 0:
            self.sample_rate = sr
        self._loaded = True
        _log.trace("qwen3_tts_loaded", model_path=self.model_path,
                   sample_rate=self.sample_rate, speaker=self.speaker_name)

    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False

    # ── Speaker ───────────────────────────────────────────────────

    @staticmethod
    def _normalize_speaker(name: str) -> str:
        name = str(name or DEFAULT_SPEAKER).strip().lower()
        if name not in PRESET_SPEAKERS:
            _log.warning(f"unknown speaker '{name}', falling back to '{DEFAULT_SPEAKER}'")
            return DEFAULT_SPEAKER
        return name

    def set_speaker(self, name: str):
        """Switch to a different preset speaker."""
        self.speaker_name = self._normalize_speaker(name)

    # ── Synthesis ────────────────────────────────────────────────

    async def synthesize(self, text: str) -> np.ndarray:
        text = _clean_for_tts(text)
        if not text:
            return np.array([], dtype=np.float32)

        timeout = self.timeout_sec if self.is_loaded() else self.warmup_timeout_sec
        # Snapshot params at call time — prevents mid-generation speaker switch
        speaker = self.speaker_name
        language = self.language
        temperature = self.temperature
        top_k = self.top_k
        top_p = self.top_p
        repetition_penalty = self.repetition_penalty

        def _generate() -> np.ndarray:
            import mlx.core as mx

            self._ensure_loaded()
            mx.clear_cache()
            try:
                results = list(self.model.generate_custom_voice(
                    text=text,
                    speaker=speaker,
                    language=language,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                ))

                audio_segments: list[np.ndarray] = []
                for item in results:
                    arr = self._extract_audio(item)
                    if arr is not None and arr.size > 0:
                        audio_segments.append(arr)

                if not audio_segments:
                    raise RuntimeError("qwen3-tts returned no audio")

                merged = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
                return self._normalize_audio(merged)
            finally:
                mx.clear_cache()

        return await run_mlx_serialized(_generate, timeout_sec=timeout)

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        """Streaming synthesis using Qwen3-TTS stream=True.

        The model processes the full text internally but yields audio chunks
        every ~2s, preserving global prosody. Uses Queue-bridge pattern:
        MLX thread pushes chunks → asyncio consumer yields them.
        """
        text = _clean_for_tts(text)
        if not text:
            return

        import asyncio
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

        timeout = self.timeout_sec if self.is_loaded() else self.warmup_timeout_sec
        # Snapshot params at call time — prevents mid-generation speaker switch
        speaker = self.speaker_name
        language = self.language
        temperature = self.temperature
        top_k = self.top_k
        top_p = self.top_p
        repetition_penalty = self.repetition_penalty

        from .runtime import _RUNTIME_GUARD, _MLX_EXECUTOR, _MLX_LOCK

        with _RUNTIME_GUARD:
            executor = _MLX_EXECUTOR
            lock = _MLX_LOCK

        def _stream_worker():
            import mlx.core as mx

            with lock:
                self._ensure_loaded()
                mx.clear_cache()
                try:
                    for item in self.model.generate_custom_voice(
                        text=text,
                        speaker=speaker,
                        language=language,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        stream=True,
                        streaming_interval=1.5,
                    ):
                        arr = self._extract_audio(item)
                        if arr is not None and arr.size > 0:
                            chunk = self._normalize_audio(arr)
                            if chunk.size > 0:
                                loop.call_soon_threadsafe(queue.put_nowait, chunk)
                finally:
                    mx.clear_cache()
                    loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        fut = loop.run_in_executor(executor, _stream_worker)

        try:
            while True:
                item = await asyncio.wait_for(queue.get(), timeout=timeout)
                if item is None:
                    break
                yield item
        except asyncio.TimeoutError:
            fut.cancel()
            raise TimeoutError(f"TTS stream timeout ({timeout:.1f}s)")
        finally:
            if not fut.done():
                try:
                    await asyncio.wait_for(fut, timeout=5)
                except Exception:
                    pass

    # ── Audio helpers ────────────────────────────────────────────

    @staticmethod
    def _extract_audio(item) -> np.ndarray | None:
        """Pull float32 numpy array from a generation result."""
        for candidate in (
            getattr(item, "audio", None),
            getattr(getattr(item, "output", None), "audio", None),
            item,
        ):
            if candidate is None:
                continue
            try:
                arr = np.asarray(candidate, dtype=np.float32).reshape(-1)
                if arr.size > 0:
                    return arr
            except Exception:
                continue
        return None

    @staticmethod
    def _normalize_audio(audio: np.ndarray) -> np.ndarray:
        """Normalize loudness to near-unity peak."""
        if audio.size == 0:
            return audio.astype(np.float32, copy=False)
        x = audio.astype(np.float32, copy=False)
        peak = float(np.max(np.abs(x)))
        if peak <= 1e-8:
            return np.array([], dtype=np.float32)
        gain = min(12.0, 0.92 / peak)
        if gain > 0:
            x = x * gain
        return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
