"""Text cleaning utilities for LLM output post-processing."""

import re

_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U0001f900-\U0001f9ff"  # supplemental
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\U0000fe0f"             # variation selector-16
    "\U0000200d"             # ZWJ
    "]+",
)

# Qwen3 thinking tags that may leak through
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_FILLER_OPENINGS = ("okay,", "okay!", "sure,", "sure!", "no problem,", "no problem!")
_TOOL_ARTIFACT_RE = re.compile(
    r"\[\s*tool_call\s*\]\s*[^\n\r]*",
    re.IGNORECASE,
)


def clean_output(text: str) -> str:
    """Post-process LLM output: strip emoji, thinking tags, leading filler."""
    text = _THINK_RE.sub("", text)
    text = _TOOL_ARTIFACT_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    for filler in _FILLER_OPENINGS:
        if text.startswith(filler):
            text = text[len(filler):]
            break
    return text.strip()
