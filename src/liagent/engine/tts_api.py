"""API-based TTS engine — supports OpenAI-compatible and ElevenLabs endpoints."""

import asyncio
import io

import numpy as np

from .base import TTSBackend
from .tts_utils import clean_text_for_tts


class ApiTTS(TTSBackend):
    """TTS backend that routes to OpenAI-compatible or ElevenLabs API
    based on the base_url.
    """

    sample_rate: int = 24000

    def __init__(self, base_url: str, api_key: str, model: str, voice: str = "alloy"):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self._loaded = True

        # Detect provider from URL
        self._provider = "elevenlabs" if "elevenlabs" in self.base_url.lower() else "openai"

        if self._provider == "openai":
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(base_url=self.base_url, api_key=api_key)
        else:
            self.client = None  # ElevenLabs uses httpx directly

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self):
        self._loaded = False

    def list_voices(self) -> list[str]:
        if self._provider == "elevenlabs":
            return [self.voice]  # voice is a voice_id
        return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def set_voice(self, voice: str):
        self.voice = voice

    async def synthesize(self, text: str) -> np.ndarray:
        text = clean_text_for_tts(text)
        if not text:
            return np.array([], dtype=np.float32)

        if self._provider == "elevenlabs":
            return await self._synthesize_elevenlabs(text)
        return await self._synthesize_openai(text)

    async def _synthesize_openai(self, text: str) -> np.ndarray:
        response = await self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="pcm",
        )
        # OpenAI PCM: 24kHz, 16-bit signed LE mono
        pcm_bytes = response.content
        self.sample_rate = 24000
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio

    async def _synthesize_elevenlabs(self, text: str) -> np.ndarray:
        import httpx

        url = f"{self.base_url}/text-to-speech/{self.voice}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        body = {
            "text": text,
            "model_id": self.model,
        }
        params = {"output_format": "pcm_24000"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=body, headers=headers, params=params)
            resp.raise_for_status()
            pcm_bytes = resp.content

        # ElevenLabs pcm_24000: 24kHz, 16-bit signed LE mono
        self.sample_rate = 24000
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
