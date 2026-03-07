"""Unified failure classification for tool execution and policy errors."""
from __future__ import annotations

import re
from enum import Enum


class FailureKind(str, Enum):
    """Inherits str so FailureKind.TIMEOUT == "timeout" for backward compat."""
    TIMEOUT = "timeout"
    POLICY_BUDGET = "policy_budget"
    POLICY_ALLOWLIST = "policy_allowlist"
    POLICY_DEDUP = "policy_dedup"
    BAD_ARGS = "bad_args"
    AUTH = "auth"
    PROVIDER = "provider"
    RATE_LIMIT = "rate_limit"


RECOVERY_STRATEGY: dict[FailureKind, str] = {
    FailureKind.TIMEOUT:          "fallback_chain",
    FailureKind.POLICY_BUDGET:    "force_answer",
    FailureKind.POLICY_ALLOWLIST: "suggest_alternative",
    FailureKind.POLICY_DEDUP:     "skip",
    FailureKind.BAD_ARGS:         "fix_args_and_retry",
    FailureKind.AUTH:             "skip_with_explanation",
    FailureKind.PROVIDER:         "fallback_chain",
    FailureKind.RATE_LIMIT:       "fallback_chain",
}

FALLBACK_ELIGIBLE: frozenset[FailureKind] = frozenset({
    FailureKind.TIMEOUT, FailureKind.PROVIDER, FailureKind.RATE_LIMIT,
})

_ARG_SIGNALS = re.compile(
    r"missing.*argument|required.*parameter|invalid.*arg|TypeError", re.I
)


def classify_error(err_type: str, message: str) -> FailureKind:
    if err_type == "timeout":
        return FailureKind.TIMEOUT
    msg_lower = message.lower()
    if "429" in message or "rate limit" in msg_lower:
        return FailureKind.RATE_LIMIT
    if "401" in message or "403" in message or "unauthorized" in msg_lower or "forbidden" in msg_lower:
        return FailureKind.AUTH
    if _ARG_SIGNALS.search(message):
        return FailureKind.BAD_ARGS
    m = re.search(r"\b(4\d{2}|5\d{2})\b", message)
    if m:
        code = int(m.group(1))
        if 400 <= code < 500:
            return FailureKind.BAD_ARGS
        if code >= 500:
            return FailureKind.PROVIDER
    return FailureKind.PROVIDER
