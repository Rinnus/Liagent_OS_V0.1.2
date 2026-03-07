"""Shared helpers for the tools package."""

import re

_SANITIZE_RE = re.compile(r"</?[a-zA-Z_][^>]*>")


def _sanitize(text: str) -> str:
    """Strip XML-like tags that could be interpreted as tool_call instructions."""
    return _SANITIZE_RE.sub("", text)
