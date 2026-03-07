"""Web search tool — safe, read-only DuckDuckGo search with quality scoring,
retry logic, and circuit breaker protection."""

import asyncio
import re
import time
from urllib.parse import urlparse

from . import ToolCapability, tool
from ._utils import _sanitize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VALID_TIMELIMITS = {"d", "w", "m", "y"}

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "for", "to",
    "how", "what", "is", "stock", "price", "latest", "today",
    "please", "help", "me", "can", "you", "check", "search",
})

_QUALITY_THRESHOLD = 0.3
_MIN_RESULTS_FOR_RETRY = 2


def _score_results(query: str, results: list[dict]) -> float:
    """Multi-signal quality score for search results.

    Returns a float in [0.0, 1.0].

    Signals (weights):
      - count:     0.20  —  more results = better coverage
      - content:   0.25  —  longer snippets = richer information
      - diversity: 0.20  —  diverse domains = less SEO spam
      - relevance: 0.35  —  keyword overlap between query and results
    """
    if not results:
        return 0.0

    # --- count signal: 5+ results = 1.0
    count_score = min(len(results) / 5.0, 1.0)

    # --- content signal: average body length, 80+ chars = 1.0
    avg_body = sum(len(r.get("body", "")) for r in results) / len(results)
    content_score = min(avg_body / 80.0, 1.0)

    # --- diversity signal: unique domains / result count
    domains = set()
    for r in results:
        href = r.get("href", "")
        if href:
            try:
                domains.add(urlparse(href).netloc)
            except Exception:
                pass
    diversity_score = (len(domains) / len(results)) if results else 0.0

    # --- relevance signal: fraction of query keywords found in result text
    query_words = {w.lower() for w in query.split() if len(w) > 1}
    if query_words:
        combined_text = " ".join(
            (r.get("title", "") + " " + r.get("body", "")).lower()
            for r in results
        )
        matched = sum(1 for w in query_words if w in combined_text)
        relevance_score = matched / len(query_words)
    else:
        relevance_score = 0.0

    raw = (
        0.20 * count_score
        + 0.25 * content_score
        + 0.20 * diversity_score
        + 0.35 * relevance_score
    )
    return max(0.0, min(1.0, raw))


def _simplify_query(query: str) -> str:
    """Strip stopwords and keep max 3 core keywords for retry.

    If stripping removes everything, return the original query.
    """
    words = query.split()
    core = [w for w in words if w.lower() not in _STOPWORDS]
    if not core:
        return query
    return " ".join(core[:3])


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------
class _SearchCircuitBreaker:
    """Opens after *threshold* consecutive failures; auto-resets after *timeout* seconds."""

    def __init__(self, threshold: int = 3, timeout: float = 60.0):
        self._threshold = threshold
        self._timeout = timeout
        self._failures = 0
        self._opened_at: float | None = None

    def is_open(self) -> bool:
        if self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self._timeout:
                # Timeout elapsed — half-open → allow one attempt
                self._opened_at = None
                self._failures = 0
                return False
            return True
        return False

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None


# Module-level circuit breaker instance
_circuit_breaker = _SearchCircuitBreaker(threshold=3, timeout=60.0)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _format_results(results: list[dict]) -> str:
    """Format results into numbered text lines."""
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = _sanitize(r.get("title", ""))
        body = _sanitize(r.get("body", ""))
        href = r.get("href", "")
        lines.append(f"{i}. {title}")
        lines.append(f"   {body}")
        if href:
            lines.append(f"   URL: {href}")
    return "\n".join(lines)


def _error_msg(kind: str, query: str, detail: str) -> str:
    """Build a SEARCH_ERROR message that never starts with '[' or '{'."""
    return f'SEARCH_ERROR({kind}): Query: "{query}" \u2014 {detail}'


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------
def _validate_web_search(args: dict) -> tuple[bool, str]:
    query = str(args.get("query", "")).strip()
    max_results = int(args.get("max_results", 5) or 5)
    if not query:
        return False, "query is required"
    if len(query) > 200:
        return False, "query is too long"
    if max_results < 1 or max_results > 10:
        return False, "max_results must be in [1, 10]"
    timelimit = args.get("timelimit")
    if timelimit and str(timelimit) not in _VALID_TIMELIMITS:
        return False, f"timelimit must be one of: {', '.join(sorted(_VALID_TIMELIMITS))}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------
@tool(
    name="web_search",
    description=(
        "Search the web via DuckDuckGo. Returns titles, snippets, and URLs.\n"
        "WHEN TO USE: factual questions about current events, prices, news, "
        "people, companies, or anything that may have changed after training.\n"
        "QUERY TIPS: Use 2\u20135 specific English keywords (e.g. 'Tesla Q4 2025 revenue'). "
        "Avoid full sentences. Include the year for time-sensitive topics.\n"
        "TIMELIMIT: 'd' (last day) for live data, 'w' (week) for recent news, "
        "'m' (month) for broader coverage. Omit for evergreen topics."
    ),
    risk_level="low",
    capability=ToolCapability(
        data_classification="public",
        network_access=True,
        filesystem_access=False,
        requires_user_presence=False,
        max_output_chars=3200,
        latency_tier="medium",
        failure_modes=("network_timeout", "rate_limit"),
    ),
    validator=_validate_web_search,
    parameters={
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Refined query keywords, space-separated, English preferred. "
                    "Example: 'Google GOOG stock price 2026'"
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results, default 5",
            },
            "timelimit": {
                "type": "string",
                "description": (
                    "Time filter: d=last day, w=last week, m=last month, y=last year. "
                    "For live data (price/news), prefer d or w."
                ),
            },
        },
        "required": ["query"],
    },
)
async def web_search(
    query: str, max_results: int = 5, timelimit: str | None = None, **kwargs
) -> str:
    """Search the web and return titles, snippets, and URLs.

    Includes automatic retry with simplified query and circuit breaker
    protection against repeated failures.
    """
    # --- Circuit breaker check ---
    if _circuit_breaker.is_open():
        return (
            "SEARCH_UNAVAILABLE: Web search is temporarily unavailable. "
            "Answer using your knowledge."
        )

    max_results = max(1, min(max_results, 10))
    if timelimit and str(timelimit) not in _VALID_TIMELIMITS:
        timelimit = None

    from ddgs import DDGS

    loop = asyncio.get_event_loop()

    def _do_search(q: str) -> list[dict] | str:
        """Execute a single search attempt.  Returns results list or error string."""
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(q, max_results=max_results, timelimit=timelimit))
        except Exception as e:
            return str(e)

    def _search_with_retry() -> str:
        # --- First attempt ---
        outcome = _do_search(query)

        if isinstance(outcome, str):
            # Network / API error
            _circuit_breaker.record_failure()
            return _error_msg(
                "error", query,
                f"{outcome}. Try a simpler query or answer from knowledge."
            )

        results: list[dict] = outcome

        # --- Empty results → retry with simplified query ---
        if not results:
            simplified = _simplify_query(query)
            if simplified != query:
                retry_outcome = _do_search(simplified)
                if isinstance(retry_outcome, list) and retry_outcome:
                    _circuit_breaker.record_success()
                    return _format_results(retry_outcome)
            _circuit_breaker.record_failure()
            return _error_msg(
                "no_results", query,
                "No results found. Try different keywords."
            )

        # --- Quality check → retry if low quality and few results ---
        score = _score_results(query, results)
        if score < _QUALITY_THRESHOLD and len(results) <= _MIN_RESULTS_FOR_RETRY:
            simplified = _simplify_query(query)
            if simplified != query:
                retry_outcome = _do_search(simplified)
                if isinstance(retry_outcome, list) and retry_outcome:
                    retry_score = _score_results(simplified, retry_outcome)
                    if retry_score > score:
                        results = retry_outcome

        _circuit_breaker.record_success()
        return _format_results(results)

    return await loop.run_in_executor(None, _search_with_retry)
