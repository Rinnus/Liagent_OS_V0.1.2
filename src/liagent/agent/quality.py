"""Lightweight response quality checks."""

from __future__ import annotations

import re

_XML_RE = re.compile(r"</?(tool_call|observation|observation_json|think)[^>]*>", re.IGNORECASE)
_FILLERS = ("okay,", "okay!", "sure,", "sure!", "no problem,", "no problem!")


_COPOUT_PATTERNS = (
    "please check directly",
    "please visit directly",
    "you should check",
    "you may search",
    "unable to fetch",
    "could not find",
    "no relevant information found in search results",
    "not found in search results",
    "needs further analysis",
    "needs further verification",
)


def detect_copout(answer: str) -> bool:
    """Detect if answer is a cop-out that deflects to external sources."""
    text = (answer or "").lower()
    if any(p in text for p in _COPOUT_PATTERNS):
        return True

    # Imperative deflection patterns like:
    # "please check Apple official site", "visit official website",
    # "monitor ... authorized retailers".
    if re.search(
        r"\b(?:please\s+)?(?:check|visit|monitor)\b.*\b"
        r"(?:official\s+site|official\s+website|authorized\s+retailers?)\b",
        text,
    ):
        return True
    return False


_PROGRESS_PLACEHOLDER_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"(?:sorry[,，:：]?\s*)?(?:let me|i(?:'| wi)?ll|i need to|i'm going to)\s+(?:check|look|inspect|review|read|open|pull up|call|run|fetch)"
    r"|i(?:'m| am)\s+(?:checking|looking|inspecting|reviewing|reading|opening|fetching)"
    r")",
    re.IGNORECASE,
)

_PROGRESS_PLACEHOLDER_HINTS = (
    "let me check",
    "let me look",
    "let me inspect",
    "let me review",
    "let me call",
    "let me run",
    "i'm checking",
    "i am checking",
)


def detect_progress_placeholder(answer: str) -> bool:
    """Detect progress-only filler replies that should not be shown as final answers."""
    text = str(answer or "").strip()
    if not text:
        return False
    compact = re.sub(r"\s+", " ", text)
    lowered = compact.lower()
    if len(compact) > 120:
        return False
    if not any(h in lowered or h in compact for h in _PROGRESS_PLACEHOLDER_HINTS):
        return False
    if _PROGRESS_PLACEHOLDER_PREFIX_RE.search(compact) is None:
        return False
    # If the answer already contains substantive findings, do not treat it as filler.
    if re.search(r"\b(?:found|result|conclusion|observed)\b", compact, re.IGNORECASE):
        return False
    if _SPECIFIC_DATA_RE.search(compact):
        return False
    return True


# Patterns where the model claims to have performed file/action operations.
# Mapped to the tool(s) that MUST have been called for the claim to be valid.
_ACTION_CLAIM_RULES: list[tuple[tuple[str, ...], set[str]]] = [
    # (claim patterns, required tools)
    (
        ("saved", "written", "generated and saved", "stored", "saved to", "written to", "placed in cwork"),
        {"write_file"},
    ),
    (
        ("executed", "run result", "execution result"),
        {"python_exec"},
    ),
]


def detect_hallucinated_action(answer: str, tools_used: set[str]) -> str:
    """Detect if the answer claims an action that was never actually performed.

    Returns a non-empty string describing the missing tool if hallucination
    is detected, or empty string if no hallucination found.
    """
    text = (answer or "").lower()
    for claim_patterns, required_tools in _ACTION_CLAIM_RULES:
        if any(p in text for p in claim_patterns):
            if not (tools_used & required_tools):
                return ", ".join(sorted(required_tools))
    return ""


_TOOL_FAILURE_CLAIM_RE = re.compile(
    r"(?:"
    r"\btool\s+(?:did(?:n't| not)\s+return|failed\s+to\s+return|did(?:n't| not)\s+produce)\b"
    r"|\b(?:technical|local\s+environment|system)\s+(?:issue|issues|problem|problems|fault|failure)\b"
    r")",
    re.IGNORECASE,
)

_TOOL_PROTOCOL_LEAK_RE = re.compile(r"<(?:tool_call|function_calls|invoke)\b", re.IGNORECASE)
_CONFIRMATION_PENDING_RE = re.compile(
    r"(?:"
    r"\b(?:waiting|awaiting)\s+(?:for\s+)?confirmation\b"
    r"|second\s+confirmation\s+is\s+required"
    r")",
    re.IGNORECASE,
)


def detect_unsourced_tool_failure(answer: str) -> bool:
    """Detect failure explanations that imply a tool/runtime fault."""
    text = str(answer or "").strip()
    if not text:
        return False
    return _TOOL_FAILURE_CLAIM_RE.search(text) is not None


def detect_tool_protocol_leak(answer: str) -> bool:
    """Detect raw tool protocol markers leaking into a user-visible answer."""
    return _TOOL_PROTOCOL_LEAK_RE.search(str(answer or "")) is not None


def detect_confirmation_pending(answer: str) -> bool:
    """Detect answers that explicitly report execution is blocked on confirmation."""
    text = str(answer or "").strip()
    if not text:
        return False
    return _CONFIRMATION_PENDING_RE.search(text) is not None


# ── Unwritten code detection (compliance check, NOT hallucination) ──────

# Language-agnostic: matches all fenced code blocks
_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)\n```", re.DOTALL)

# Strong intent: user clearly wants a file created → +5
_STRONG_FILE_INTENT = (
    "put in", "save to", "store to", "write to",
    "cwork", "folder",
    "create", "write", "build",
)

# Weak intent: ambiguous — could be explanatory → +2
_WEAK_FILE_INTENT = (
    "script", "game", "app", "generate", "write code",
    "create",
)

# Sensitive operations → +5 (audit trail required)
_SENSITIVE_OPS_RE = re.compile(
    r"\b(os\.remove|os\.unlink|shutil\.rmtree|shutil\.move|subprocess\."
    r"|socket\.|exec\(|eval\(|__import__)\b"
)


def _score_code_for_file(block: str) -> int:
    """Score a single code block on how strongly it should be written to a file."""
    score = 0
    lines = block.strip().splitlines()

    # Structure: entry point → almost certainly a standalone script
    if any("if __name__" in ln for ln in lines):
        score += 5

    # Structure: class definitions → module-level code
    class_count = sum(1 for ln in lines if re.match(r"^\s*class\s+\w+", ln))
    score += min(class_count * 3, 9)

    # Structure: multiple functions = real module
    func_count = sum(1 for ln in lines if re.match(r"^\s*(async\s+)?def\s+\w+", ln))
    if func_count >= 2:
        score += 2

    # Content: sensitive operations → must audit
    sensitive_hits = len(_SENSITIVE_OPS_RE.findall(block))
    score += min(sensitive_hits * 5, 10)

    # Content: import density (>3 imports = standalone program)
    import_count = sum(
        1 for ln in lines if re.match(r"^\s*(import |from \S+ import )", ln)
    )
    if import_count >= 3:
        score += 2

    # Volume: +1 per 10 lines
    score += len(lines) // 10

    return score


def detect_unwritten_code(
    answer: str,
    user_input: str,
    tools_used: set[str],
) -> tuple[bool, int, int]:
    """Detect if the response contains code that should have been saved via write_file.

    This is a **compliance** check (output format), not a hallucination check.
    Uses a weighted scoring model across structure, content, intent, and volume.

    Returns (should_retry, total_code_lines, score).
    """
    if "write_file" in tools_used:
        return False, 0, 0

    blocks = _CODE_BLOCK_RE.findall(answer)
    if not blocks:
        return False, 0, 0

    # Code-to-text ratio: if code is <30% of the answer, it's likely explanatory
    total_code_chars = sum(len(b) for b in blocks)
    if len(answer) > 0 and total_code_chars / len(answer) < 0.30:
        return False, 0, 0

    total_lines = 0
    total_score = 0
    for block in blocks:
        total_lines += len(block.strip().splitlines())
        total_score += _score_code_for_file(block)

    # Intent scoring (strong vs weak)
    query_lower = user_input.lower()
    if any(sig in query_lower for sig in _STRONG_FILE_INTENT):
        total_score += 5
    elif any(sig in query_lower for sig in _WEAK_FILE_INTENT):
        total_score += 2

    # Threshold: 5 points triggers write_file requirement
    return total_score >= 5, total_lines, total_score


# ── Data hallucination detection ─────────────────────────────────────

# Real-time topic keywords in user input that require tool-backed data.
# Includes host/runtime status queries (CPU, memory, disk, temperature, latency).
_REALTIME_TOPIC_RE = re.compile(
    r"(stock|share\s*price|price|quote|market|market\s*cap|change|earnings|"
    r"revenue|income|profit|p/e|weather|temperature|exchange\s*rate|"
    r"oil\s*price|gold\s*price|housing\s*price|news|latest|today|current|real\s*time|"
    r"cpu|memory|ram|disk|storage|load|usage|latency|network|uptime|system\s*status)"
)

# Specific data patterns in model output that suggest fabricated numbers.
# Covers finance/news values and local runtime metrics such as GB, ms, cores, and °C.
_SPECIFIC_DATA_RE = re.compile(
    r"(\d+(?:\.\d+)?(?:\s*(?:billion|million|trillion|thousand|hundred)))"
    r"|(\$\s*\d+(?:\.\d+)?)"
    r"|(\d+(?:\.\d+)?%)"
    r"|(\d+(?:\.\d+)?\s*(?:USD|RMB|CNY|EUR|GBP|JPY))"
    r"|(\d+(?:\.\d+)?\s*(?:KB|MB|GB|TB|KiB|MiB|GiB|TiB))"
    r"|(\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds))"
    r"|(\d+(?:\.\d+)?\s*(?:°C|℃|°F))"
    r"|(\d+(?:\.\d+)?\s*(?:cores?))"
)


def detect_unsourced_data(
    answer: str, user_input: str, tools_used: set[str]
) -> bool:
    """Detect if answer cites specific numbers about a real-time topic without tool backing.

    Returns True if the response likely contains hallucinated data.
    """
    # If any data-fetching tool was used, data may be legitimate
    if tools_used & {"web_search", "web_fetch", "python_exec", "system_status"}:
        return False
    # Check if user asked about a real-time topic
    if not _REALTIME_TOPIC_RE.search(user_input):
        return False
    # Check if the answer cites specific numbers
    matches = _SPECIFIC_DATA_RE.findall(answer)
    return len(matches) >= 1




# ── Key metrics cross-validation (stock data safety net) ────────────

# Extract percentage pattern like (2.21%) from text
_ANSWER_PCT_RE = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*%\s*\)")
# Extract market cap in trillion/billion from answer
_ANSWER_MCAP_WY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(trillion|t)\b", re.IGNORECASE)
_ANSWER_MCAP_SY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(billion|b)\b", re.IGNORECASE)
# Extract percentage from stock result
_STOCK_PCT_RE = re.compile(r"\(\s*[+-]?(\d+(?:\.\d+)?)\s*%\s*\)")
# Extract market cap from stock summary
_STOCK_MCAP_WY_RE = re.compile(
    r"(?:market\s*cap|mcap)[^\d]{0,16}\$?\s*(\d+(?:\.\d+)?)\s*(?:trillion|t)\b",
    re.IGNORECASE,
)
_STOCK_MCAP_SY_RE = re.compile(
    r"(?:market\s*cap|mcap)[^\d]{0,16}\$?\s*(\d+(?:\.\d+)?)\s*(?:billion|b)\b",
    re.IGNORECASE,
)


def validate_key_metrics(answer: str, stock_result: str) -> tuple[str, list[str]]:
    """Cross-validate key numbers in answer against stock tool output.

    Checks percentage changes and market cap figures. If answer diverges
    >50% from stock data, replaces with stock value.

    Returns (fixed_answer, list_of_fixes).
    """
    if not answer or not stock_result:
        return answer, []

    fixes: list[str] = []
    fixed = answer

    # 1. Validate percentage change
    stock_pcts = _STOCK_PCT_RE.findall(stock_result)
    if stock_pcts:
        stock_pct = float(stock_pcts[0])
        for m in _ANSWER_PCT_RE.finditer(fixed):
            ans_pct = float(m.group(1))
            if stock_pct > 0.01 and ans_pct > 0.01:
                ratio = max(ans_pct, stock_pct) / min(ans_pct, stock_pct)
                if ratio > 1.5:
                    old = m.group(0)
                    # Preserve the bracket style
                    opener = old[0]
                    closer = ")"
                    new = f"{opener}{stock_pct:.2f}%{closer}"
                    fixed = fixed.replace(old, new, 1)
                    fixes.append(f"change {ans_pct}% -> {stock_pct}%")

    # 2. Validate market cap (trillion)
    stock_mcap_wy = _STOCK_MCAP_WY_RE.findall(stock_result)
    if stock_mcap_wy:
        stock_val = float(stock_mcap_wy[0])
        for m in _ANSWER_MCAP_WY_RE.finditer(fixed):
            ans_val = float(m.group(1))
            if stock_val > 0 and ans_val > 0:
                ratio = max(ans_val, stock_val) / min(ans_val, stock_val)
                if ratio > 1.5:
                    old = m.group(0)
                    suffix = m.group(2).lower()
                    new = f"{stock_val:.2f}T" if suffix == "t" else f"{stock_val:.2f} trillion"
                    fixed = fixed.replace(old, new, 1)
                    fixes.append(f"market cap {ans_val} trillion -> {stock_val} trillion")

    # 3. Validate market cap (billion)
    stock_mcap_sy = _STOCK_MCAP_SY_RE.findall(stock_result)
    if stock_mcap_sy:
        stock_val = float(stock_mcap_sy[0])
        for m in _ANSWER_MCAP_SY_RE.finditer(fixed):
            ans_val = float(m.group(1))
            if stock_val > 0 and ans_val > 0:
                ratio = max(ans_val, stock_val) / min(ans_val, stock_val)
                if ratio > 1.5:
                    old = m.group(0)
                    suffix = m.group(2).lower()
                    new = f"{stock_val:.2f}B" if suffix == "b" else f"{stock_val:.2f} billion"
                    fixed = fixed.replace(old, new, 1)
                    fixes.append(f"market cap {ans_val} billion -> {stock_val} billion")

    return fixed, fixes


# ── Ungrounded numbers detection (answer vs tool observations) ────────

def _extract_numbers(text: str) -> list[float]:
    """Extract numeric values from text using _SPECIFIC_DATA_RE.

    Returns parsed floats with currency/unit suffixes stripped.
    """
    nums: list[float] = []
    for m in _SPECIFIC_DATA_RE.finditer(text):
        raw = m.group(0)
        # Strip non-numeric suffixes/prefixes
        cleaned = re.sub(
            r"[$%USD RMB CNY EUR GBP JPY]", "", raw, flags=re.IGNORECASE
        )
        cleaned = re.sub(
            r"(?:billion|million|trillion)", "", cleaned, flags=re.IGNORECASE
        ).strip()
        cleaned = cleaned.replace(",", "")
        try:
            nums.append(float(cleaned))
        except ValueError:
            continue
    return nums


def _is_trivial(n: float) -> bool:
    """Filter out trivial numbers that are too common to be meaningful.

    Conservative filter: only removes 0-2 (ordinals/bullets) and year-like
    values. Since _SPECIFIC_DATA_RE already requires a data suffix (%, $, units),
    most captured numbers are meaningful data points, e.g. 7%
    as an interest rate should NOT be filtered.
    """
    # Very small integers (ordinals, bullets)
    if n == int(n) and 0 <= n <= 2:
        return True
    # Year-like values
    if n == int(n) and 1900 <= n <= 2100:
        return True
    return False


def detect_ungrounded_numbers(
    answer: str,
    observations: str,
    *,
    min_ungrounded: int = 3,
    tolerance: float = 0.20,
) -> bool:
    """Detect if answer cites numbers not grounded in tool observations.

    Extracts numeric values from both answer and observations, then checks
    whether answer numbers have approximate matches in the observation set.
    Returns True if >= min_ungrounded non-trivial answer numbers have no
    match within the given relative tolerance.
    """
    if not answer or not observations:
        return False

    ans_nums = _extract_numbers(answer)
    obs_nums = _extract_numbers(observations)

    if not ans_nums or not obs_nums:
        return False

    ungrounded = 0
    for a in ans_nums:
        if _is_trivial(a):
            continue
        matched = any(
            abs(a - o) / max(abs(o), 1.0) <= tolerance for o in obs_nums
        )
        if not matched:
            ungrounded += 1

    return ungrounded >= min_ungrounded


def detect_answer_degenerate(text: str, *, min_len: int = 30, min_repeats: int = 3) -> bool:
    """Detect degenerate *answer* text via sliding-window substring repetition.

    Scans for any substring of length >= ``min_len`` that appears
    ``min_repeats`` or more times in *text*.  Unlike ``detect_degenerate_output``
    (which only checks tool_call tags and a fixed prefix), this catches
    organic content loops such as the model repeating the same paragraph.

    Returns True if the answer is degenerate.
    """
    if len(text) < min_len * min_repeats:
        return False
    # Strided sliding window across the full text.
    # Step size = wlen to avoid O(n²) while still covering all regions.
    for wlen in (30, 50, 80):
        if wlen > len(text) // min_repeats:
            continue
        for start in range(0, len(text) - wlen + 1, wlen):
            pat = text[start : start + wlen]
            if text.count(pat) >= min_repeats:
                return True
    return False


_TOOL_CALL_FRAG_RE = re.compile(r"<tool_call>", re.IGNORECASE)
_REASONING_LEAK_LINE_RE = re.compile(
    r"^\s*(?:let me|i need to|i should|i will|i'll|calling tool|searching|"
    r"continuing)",
    re.IGNORECASE,
)
_REASONING_LEAK_KEYWORDS = (
    "let me search",
    "let me fetch",
    "let me call",
    "i need to call",
    "calling tool",
    "i should search",
    "searching for",
    "continuing",
)


def detect_reasoning_leak(text: str, *, min_hits: int = 3) -> bool:
    """Detect repeated planning chatter that should not be shown to users."""
    body = str(text or "").strip()
    if not body:
        return False
    lowered = body.lower()
    keyword_hits = sum(lowered.count(keyword) for keyword in _REASONING_LEAK_KEYWORDS)
    if keyword_hits >= min_hits:
        return True

    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if len(lines) < min_hits:
        return False
    planning_lines = [ln for ln in lines if _REASONING_LEAK_LINE_RE.search(ln)]
    return len(planning_lines) >= min_hits


def detect_degenerate_output(text: str) -> bool:
    """Detect degenerate model output: repetitive tool_call fragments or loops.

    Returns True if the output is degenerate and should not be shown to user.
    """
    body = str(text or "")
    if not body:
        return False
    if detect_reasoning_leak(body):
        return True
    tc_count = len(_TOOL_CALL_FRAG_RE.findall(body))
    if tc_count >= 3:
        return True
    # Check for any long substring (20+ chars) repeated 3+ times
    if len(body) > 100:
        chunk = body[:60]
        if body.count(chunk) >= 3:
            return True
    return False


def quality_fix(answer: str, max_chars: int = 2000) -> tuple[str, dict]:
    """Return fixed answer and quality metadata."""
    raw = answer or ""
    fixed = _XML_RE.sub("", raw).strip()
    for f in _FILLERS:
        if fixed.startswith(f):
            fixed = fixed[len(f) :].strip()
            break
    if len(fixed) > max_chars:
        fixed = fixed[: max_chars - 3].rstrip() + "..."

    issues = []
    if raw != fixed:
        issues.append("normalized_output")
    return fixed, {"issues": issues, "score": 1.0 - min(0.5, len(issues) * 0.1)}


# ── Task outcome estimation (moved from service_tier.py) ─────────

_SENTINEL = object()


def estimate_task_success(
    *,
    answer: str,
    tool_calls: int,
    tool_errors: int,
    policy_blocked: int,
    plan_total_steps: int,
    plan_completed_steps: int,
    tools_used: set[str] | None = None,
    detect_hallucinated_action_fn=_SENTINEL,
) -> tuple[bool, str]:
    """Heuristic task outcome estimation for telemetry."""
    text = (answer or "").strip()
    if not text:
        return False, "empty_answer"

    if detect_tool_protocol_leak(text):
        return False, "tool_protocol_leak"

    if detect_progress_placeholder(text):
        return False, "progress_placeholder"

    if detect_confirmation_pending(text):
        return False, "confirmation_pending"

    if tool_errors == 0 and detect_unsourced_tool_failure(text):
        return False, "unsourced_tool_failure"

    if (
        tool_errors > 0
        and plan_total_steps > 0
        and plan_completed_steps < max(1, (plan_total_steps + 1) // 2)
    ):
        return False, "tool_errors_low_plan_completion"

    if (
        policy_blocked > 0
        and tool_calls > 0
        and policy_blocked >= tool_calls
        and len(text) < 48
    ):
        return False, "all_tools_blocked_short_answer"

    low_conf_markers = ("unable", "cannot", "failed", "sorry", "not sure", "uncertain")
    if (
        any(m in text for m in low_conf_markers)
        and len(text) < 36
        and plan_total_steps > 0
        and plan_completed_steps == 0
    ):
        return False, "fallback_only"

    if tool_calls > 0:
        if detect_copout(text):
            return False, "copout_answer"

    if tools_used is not None:
        _fn = detect_hallucinated_action_fn
        if _fn is _SENTINEL:
            _fn = detect_hallucinated_action
        if _fn is not None and _fn(text, tools_used):
            return False, "hallucinated_action"

    if detect_answer_degenerate(text):
        return False, "degenerate_answer"

    return True, "ok"


def plan_completion_ratio(total_steps: int, completed_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    ratio = float(completed_steps) / float(total_steps)
    return max(0.0, min(1.0, ratio))
