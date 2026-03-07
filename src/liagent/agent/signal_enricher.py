"""Signal Enricher — converts raw deltas into a unified enriched schema.

API deltas (pct_change) get deterministic enrichment (no LLM, confidence=1.0).
Web deltas (content_changed) get LLM extraction (confidence=0.5-0.8).

Enriched delta schema::

    {
        "severity": 0.0-1.0,
        "sentiment": -1.0 to 1.0,
        "event_type": str,
        "key_fact": str,
        "confidence": 0.0-1.0,
        "source": "api" | "web_enriched",
        "raw_delta": dict,
    }
"""

import json
import re
import time

from ..logging import get_logger

_log = get_logger("signal_enricher")

# ── JSON parsing (standalone, copied from planner.py to avoid coupling) ────

_JSON_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def _clean_json_text(raw: str) -> str:
    return _JSON_CTRL_RE.sub("", str(raw or ""))


def _iter_json_object_blobs(text: str) -> list[str]:
    s = str(text or "")
    out: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for idx, ch in enumerate(s):
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(s[start : idx + 1])
                    start = -1
    return out


def _json_object_candidates(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    text_no_think = _THINK_BLOCK_RE.sub("", text).strip()
    cleaned = _clean_json_text(text)
    candidates: list[str] = [text, cleaned]
    if text_no_think and text_no_think != text:
        candidates.append(text_no_think)
        candidates.append(_clean_json_text(text_no_think))
    for block in _JSON_FENCE_RE.findall(text):
        block_clean = _clean_json_text(block).strip()
        if block_clean:
            candidates.append(block_clean)
    for blob in _iter_json_object_blobs(text):
        cleaned_blob = _clean_json_text(blob).strip()
        if cleaned_blob:
            candidates.append(cleaned_blob)
    for blob in _iter_json_object_blobs(cleaned):
        if blob.strip():
            candidates.append(blob.strip())
    seen: set[str] = set()
    uniq: list[str] = []
    for item in candidates:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    return uniq


def _extract_json(raw: str) -> dict | None:
    for candidate in _json_object_candidates(raw):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


# ── LLM prompt for web delta enrichment ───────────────────────────────────

_ENRICH_PROMPT = """\
You are a signal analyst. Given a monitoring factor's web search results,
extract structured event information.

Factor: {factor_name}
Intent: {intent}
Entity: {entity}
Content:
{content}

Output a single JSON object (no markdown fences):
{{
  "severity": <0.0-1.0, how important is this event>,
  "sentiment": <-1.0 to 1.0, negative to positive>,
  "event_type": "<category: earnings, regulatory, product, market, legal, personnel, other>",
  "key_fact": "<one sentence summary of what happened>"
}}"""


# ── SignalEnricher ─────────────────────────────────────────────────────────

class SignalEnricher:
    """Converts raw signal deltas into unified enriched schema.

    API deltas: deterministic, no LLM cost.
    Web deltas: LLM extraction, rate-limited and cached.
    """

    def __init__(self, engine, *, cache_ttl_seconds: int = 3600):
        self.engine = engine
        self._cache_ttl = cache_ttl_seconds
        # Cache: (factor_id, content_hash) → (enriched_delta, timestamp)
        self._cache: dict[tuple[str, str], tuple[dict, float]] = {}
        # Rate limit: (interest_id, factor_id) → last_enrich_timestamp
        self._rate_limit: dict[tuple[str, str], float] = {}
        self._rate_limit_seconds = 60.0

    async def enrich(self, signal: dict) -> dict:
        """Enrich a signal's delta in-place and return the signal."""
        delta = signal.get("delta")
        if not delta:
            return signal

        if "pct_change" in delta:
            enriched = self._enrich_api_delta(delta, signal)
        elif delta.get("type") == "content_changed":
            enriched = await self._enrich_web_delta(delta, signal)
        else:
            return signal

        signal["delta"] = enriched
        return signal

    def _enrich_api_delta(self, delta: dict, context: dict) -> dict:
        """Deterministic enrichment for API/numeric deltas. Free, no LLM."""
        pct = delta["pct_change"]
        severity = min(1.0, abs(pct) / 10.0)
        sentiment = 1.0 if pct > 0 else (-1.0 if pct < 0 else 0.0)

        direction = "rose" if pct > 0 else "fell"
        entity = context.get("entity", context.get("factor_name", "?"))
        key_fact = f"{entity} {direction} {abs(pct):.1f}%"

        return {
            "severity": round(severity, 2),
            "sentiment": sentiment,
            "event_type": "market",
            "key_fact": key_fact,
            "confidence": 1.0,
            "source": "api",
            "raw_delta": delta,
        }

    async def _enrich_web_delta(self, delta: dict, context: dict) -> dict:
        """LLM-based enrichment for web content changes. Rate-limited + cached."""
        factor_id = context.get("factor_id", "")
        interest_id = context.get("interest_id", "")
        content_hash = delta.get("new_hash", "")

        # Check cache
        cache_key = (factor_id, content_hash)
        cached = self._cache.get(cache_key)
        if cached is not None:
            enriched, ts = cached
            if time.monotonic() - ts < self._cache_ttl:
                return enriched

        # Check rate limit
        rl_key = (interest_id, factor_id)
        last_enrich = self._rate_limit.get(rl_key, 0.0)
        now = time.monotonic()
        if now - last_enrich < self._rate_limit_seconds:
            return self._web_fallback(delta)

        self._rate_limit[rl_key] = now

        # Build LLM prompt
        content = delta.get("full_content") or delta.get("snippet", "")
        if not content.strip():
            return self._web_fallback(delta)

        prompt = _ENRICH_PROMPT.format(
            factor_name=context.get("factor_name", ""),
            intent=context.get("intent", ""),
            entity=context.get("entity", ""),
            content=content[:2000],
        )

        try:
            raw = await self.engine.generate_extraction(
                [{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.1,
            )
            parsed = _extract_json(raw)
            if parsed and "severity" in parsed:
                enriched = {
                    "severity": max(0.0, min(1.0, float(parsed.get("severity", 0.3)))),
                    "sentiment": max(-1.0, min(1.0, float(parsed.get("sentiment", 0.0)))),
                    "event_type": str(parsed.get("event_type", "other")),
                    "key_fact": str(parsed.get("key_fact", delta.get("snippet", "")[:100])),
                    "confidence": 0.7,
                    "source": "web_enriched",
                    "raw_delta": delta,
                }
                self._cache[cache_key] = (enriched, now)
                return enriched
        except Exception as e:
            _log.error("enrich_web_llm", e)

        return self._web_fallback(delta)

    @staticmethod
    def _web_fallback(delta: dict) -> dict:
        """Safe fallback when LLM enrichment fails."""
        return {
            "severity": 0.3,
            "sentiment": 0.0,
            "event_type": "other",
            "key_fact": delta.get("snippet", "")[:100] or "Content changed",
            "confidence": 0.3,
            "source": "web_enriched",
            "raw_delta": delta,
        }
