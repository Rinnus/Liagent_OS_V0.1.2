"""Evidence aggregation — structured artifacts, cross-source verification, and synthesis."""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse


# ── Regex patterns for extracting structured data points ──────────────

_MONEY_RE = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|trillion|B|M|T)?",
    re.IGNORECASE,
)
_UNIT_MONEY_RE = re.compile(
    r"(?<![$€£¥\d.,])([\d,]+(?:\.\d+)?)\s*(billion|million|trillion|B|M|T)\b(?:\s*(?:USD|US\$|RMB|CNY|EUR|GBP|JPY))?",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"([\d]+(?:\.\d+)?)\s*%")
_DATE_RE = re.compile(
    r"(?:Q[1-4]\s*\d{4}|FY\s*\d{4}|\d{4}\s*Q[1-4]|\b(?:20\d{2})\b)",
    re.IGNORECASE,
)
_EPS_RE = re.compile(r"EPS\s*(?:of\s*)?\$?([\d]+(?:\.\d+)?)", re.IGNORECASE)

# URL extraction from search result text (shared with tool_orchestrator)
_URL_LINE_RE = re.compile(r"URL:\s*(https?://\S+)", re.IGNORECASE)


# ── Money normalization ───────────────────────────────────────────────

# Multipliers to normalize to millions
_MONEY_UNITS: dict[str, float] = {
    "trillion": 1_000_000, "t": 1_000_000,
    "billion": 1_000, "b": 1_000,
    "million": 1, "m": 1,
}


def _normalize_money(raw: str, unit: str | None) -> float | None:
    """Normalize a money match to millions. Returns None if unparseable."""
    try:
        if not raw:
            return None
        num_match = re.search(r"[\d,]+(?:\.\d+)?", raw)
        if not num_match:
            return None
        num = float(num_match.group(0).replace(",", ""))
        unit_key = (unit or "").strip().lower()
        mult = _MONEY_UNITS.get(unit_key, 1)
        return num * mult
    except (ValueError, TypeError):
        pass
    return None


# ── Domain trust ──────────────────────────────────────────────────────

_DEFAULT_TRUSTED_SUFFIXES = (
    "reuters.com", "bloomberg.com", "finance.yahoo.com",
    "wsj.com", "cnbc.com", "ft.com", "sec.gov",
    "nasdaq.com", "investing.com", "marketwatch.com",
    "eastmoney.com", "10jqka.com.cn", "hexun.com",
)

_trusted_suffixes: tuple[str, ...] | None = None


def _get_trusted_suffixes() -> tuple[str, ...]:
    """Load trusted domain suffixes (cached). Configurable via LIAGENT_TRUSTED_DOMAINS env."""
    global _trusted_suffixes
    if _trusted_suffixes is not None:
        return _trusted_suffixes
    custom = os.environ.get("LIAGENT_TRUSTED_DOMAINS", "").strip()
    if custom:
        _trusted_suffixes = tuple(s.strip().lower() for s in custom.split(",") if s.strip())
    else:
        _trusted_suffixes = _DEFAULT_TRUSTED_SUFFIXES
    return _trusted_suffixes


def _is_trusted_domain(url: str) -> bool:
    """Check if a URL's host matches a trusted domain suffix (boundary-safe)."""
    try:
        host = (urlparse(url).hostname or "").lower()
        if not host:
            return False
        # IDNA: encode+decode to normalize unicode domains
        try:
            host = host.encode("idna").decode("ascii").lower()
        except (UnicodeError, UnicodeDecodeError):
            pass
        for suffix in _get_trusted_suffixes():
            if host == suffix or host.endswith("." + suffix):
                return True
    except Exception:
        pass
    return False


# ── Data extraction ───────────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    """Extract clean domain from a URL."""
    try:
        host = urlparse(url).hostname or ""
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def extract_urls_from_text(text: str) -> list[str]:
    """Extract URLs from text containing 'URL: https://...' lines.

    Public helper shared between brain.py and tool_orchestrator.py.
    """
    return [m.group(1).strip() for m in _URL_LINE_RE.finditer(text)]


def _extract_data_points(text: str) -> list[dict]:
    """Extract monetary values, percentages, dates, and EPS from text."""
    points: list[dict] = []
    seen_spans: set[tuple[int, int]] = set()

    for m in _MONEY_RE.finditer(text):
        # Keep symbol-based raw concise (e.g. "$6.8") to avoid mixing
        # with unit-only phrases in downstream heuristics/tests.
        raw = f"${m.group(1)}" if m.group(0).lstrip().startswith("$") else m.group(0).strip()
        # Groups: (1)USD_num, (2)USD_unit
        usd_unit = m.group(2)
        value_m = _normalize_money(raw, usd_unit)
        pt: dict = {"type": "money", "raw": raw, "pos": m.start()}
        if value_m is not None:
            pt["value_m"] = round(value_m, 2)
        points.append(pt)
        seen_spans.add((m.start(), m.end()))

    for m in _UNIT_MONEY_RE.finditer(text):
        span = (m.start(), m.end())
        if span in seen_spans:
            continue
        raw = m.group(0).strip()
        unit = m.group(2)
        value_m = _normalize_money(raw, unit)
        pt = {"type": "money", "raw": raw, "pos": m.start()}
        if value_m is not None:
            pt["value_m"] = round(value_m, 2)
        points.append(pt)

    for m in _PERCENT_RE.finditer(text):
        val = m.group(1)
        points.append({"type": "percent", "raw": f"{val}%", "value": float(val), "pos": m.start()})

    for m in _EPS_RE.finditer(text):
        val = m.group(1)
        points.append({"type": "eps", "raw": f"EPS ${val}", "value": float(val), "pos": m.start()})

    for m in _DATE_RE.finditer(text):
        points.append({"type": "date", "raw": m.group(0).strip(), "pos": m.start()})

    return points


# ── Conflict detection (money + percent + eps) ────────────────────────

def _find_conflicts(sources: list[dict]) -> list[str]:
    """Compare same-type numeric data across sources; flag significant divergence.

    Thresholds:
    - money (value_m, normalized to millions): >5% relative divergence
    - percent (value): >2 absolute difference
    - eps (value): >2% relative divergence
    """
    conflicts: list[str] = []

    money_by_source: dict[str, list[float]] = {}
    percent_by_source: dict[str, list[float]] = {}
    eps_by_source: dict[str, list[float]] = {}

    for src in sources:
        domain = src["domain"]
        for pt in src["points"]:
            if pt["type"] == "money" and "value_m" in pt:
                money_by_source.setdefault(domain, []).append(pt["value_m"])
            elif pt["type"] == "percent" and "value" in pt:
                percent_by_source.setdefault(domain, []).append(pt["value"])
            elif pt["type"] == "eps" and "value" in pt:
                eps_by_source.setdefault(domain, []).append(pt["value"])

    def _cross_compare(by_source: dict[str, list[float]], label: str,
                       threshold: float, absolute: bool = False) -> list[str]:
        pairs: list[tuple[str, float]] = []
        for domain, vals in by_source.items():
            for v in vals:
                pairs.append((domain, v))
        hits: list[str] = []
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                d1, v1 = pairs[i]
                d2, v2 = pairs[j]
                if d1 == d2:
                    continue
                if absolute:
                    if abs(v1 - v2) > threshold:
                        hits.append(f"{label}: {d1}={v1} vs {d2}={v2}")
                else:
                    avg = (abs(v1) + abs(v2)) / 2
                    if avg > 0 and abs(v1 - v2) / avg > threshold:
                        hits.append(f"{label}: {d1}={v1} vs {d2}={v2}")
        return hits

    conflicts.extend(_cross_compare(eps_by_source, "EPS", 0.02))
    conflicts.extend(_cross_compare(money_by_source, "amount", 0.05))
    conflicts.extend(_cross_compare(percent_by_source, "%", 2.0, absolute=True))

    return conflicts[:3]


# ── Evidence aggregation (for multi-tool queries) ─────────────────────

def aggregate_evidence(
    context_vars: dict[str, str],
    source_urls: list[tuple[str, str]],
) -> str:
    """Build a structured evidence summary from tool results.

    Returns an empty string if fewer than 2 result sources are available,
    or if no meaningful data points are extracted.

    Args:
        context_vars: Dict of context variables from the ReAct loop.
            Only keys ending with ``_result`` (excluding ``_pending_*``) are used.
        source_urls: List of (title, url) tuples from search results.
    """
    # 1. Filter result keys
    result_texts: dict[str, str] = {}
    for k, v in context_vars.items():
        if (
            k.endswith("_result")
            and not k.startswith("_pending_")
            and isinstance(v, str)
            and v.strip()
        ):
            result_texts[k] = v

    if len(result_texts) < 2:
        return ""

    # 2. Build domain lookup from source_urls
    url_domains: list[tuple[str, str]] = []  # (domain, title)
    for title, url in source_urls:
        domain = _extract_domain(url)
        if domain:
            url_domains.append((domain, title or domain))

    # 3. Extract data points per source
    sources: list[dict] = []
    for key, text in result_texts.items():
        # Try to match a domain from urls that appear in the text
        domain = key.replace("_result", "")
        for d, t in url_domains:
            if d in text or t.lower() in text.lower():
                domain = d
                break

        points = _extract_data_points(text)
        if points:
            sources.append({"domain": domain, "points": points, "key": key})

    if not sources:
        return ""

    # 4. Check for cross-source conflicts
    conflicts = _find_conflicts(sources)

    # 5. Build summary (≤300 chars)
    parts: list[str] = []
    # Collect unique dates mentioned
    dates = set()
    for src in sources:
        for pt in src["points"]:
            if pt["type"] == "date":
                dates.add(pt["raw"])

    if dates:
        parts.append(f"data recency: {', '.join(sorted(dates)[:3])}")

    parts.append(f"combined {len(sources)} data sources")

    if conflicts:
        parts.append("[Warning] " + "; ".join(conflicts))

    summary = " | ".join(parts)
    # Enforce 300-char cap
    if len(summary) > 300:
        summary = summary[:297] + "..."

    return f"[Evidence] {summary}"
