"""TTS utility functions — text cleaning, sentence splitting, chunk building.

Extracted from tts_local.py (deprecated Kokoro backend) so that active
modules (tts_qwen3, tts_api, voice_chat, cli) do not depend on a
deprecated module.
"""

import re
from typing import Literal


def clean_text_for_tts(text: str) -> str:
    """Clean text for TTS while keeping punctuation that helps prosody."""
    text = re.sub(
        r"[\U0001F600-\U0001F64F"
        r"\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF"
        r"\U0000FE00-\U0000FE0F"
        r"\U0001F900-\U0001F9FF"
        r"\U0001FA00-\U0001FA6F"
        r"\U0001FA70-\U0001FAFF"
        r"\U0000200D"
        r"\U00002B50\U00002B55"
        r"\U0000231A-\U0000231B"
        r"\U000023E9-\U000023F3"
        r"\U000023F8-\U000023FA]+",
        "",
        text,
    )
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[\[\]\(\)\{\}<>@#$%^&*_+=\\|/`~]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str, max_chunk_chars: int = 220) -> list[str]:
    """Split long text into sentence chunks for low-latency playback."""
    if not text:
        return []

    parts = [p for p in re.split(r"(?<=[.!?\n])", text) if p.strip()]
    if not parts:
        return []

    result: list[str] = []
    chunk = ""
    for part in parts:
        if len(chunk) + len(part) > max_chunk_chars and chunk:
            result.append(chunk.strip())
            chunk = part
        else:
            chunk += part
    if chunk.strip():
        result.append(chunk.strip())
    return result


def build_tts_chunks(
    text: str,
    *,
    chunk_strategy: Literal["oneshot", "smart_chunk"] = "smart_chunk",
    max_chunk_chars: int = 220,
) -> list[str]:
    cleaned = clean_text_for_tts(text)
    if not cleaned:
        return []
    if chunk_strategy == "oneshot" or len(cleaned) <= max_chunk_chars:
        return [cleaned]
    return split_sentences(cleaned, max_chunk_chars=max_chunk_chars)
