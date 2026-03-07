"""Signal Poller — Layer 0 of the attention system.

Periodically polls EXECUTABLE/PROXY factors and detects significant changes.
All polling is zero-token (API calls only). LLM is never invoked here.

Architecture:
  InterestStore.get_pollable_factors() → per-factor asyncio loops → API calls
  → value comparison → signal_log on significant change → on_signal callback
"""

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Callable, Coroutine

import httpx

from ..logging import get_logger
from .interest import InterestStore

_log = get_logger("signal_poller")


# ── Circuit breaker for DuckDuckGo rate limiting ────────────────────────

class _CircuitBreaker:
    """Simple circuit breaker with half-open probe support."""

    def __init__(self, fail_threshold: int = 3, reset_seconds: float = 120):
        self._fail_threshold = fail_threshold
        self._reset_seconds = reset_seconds
        self._failures = 0
        self._open_until = 0.0

    def record_failure(self):
        self._failures += 1
        if self._failures >= self._fail_threshold:
            self._open_until = time.monotonic() + self._reset_seconds

    def record_success(self):
        self._failures = 0
        self._open_until = 0.0

    def is_open(self) -> bool:
        if self._open_until <= 0.0:
            return False
        if time.monotonic() >= self._open_until:
            self._open_until = 0.0  # half-open: allow one probe
            return False
        return True


_ddgs_breaker = _CircuitBreaker(fail_threshold=3, reset_seconds=120)

# Finnhub API
_FINNHUB_BASE = "https://finnhub.io/api/v1"


_finnhub_key_cache: str | None = None


def _finnhub_key() -> str:
    """Resolve Finnhub API key: config.json > env var. Cached after first call."""
    global _finnhub_key_cache
    if _finnhub_key_cache is not None:
        return _finnhub_key_cache
    key = ""
    try:
        from ..config import AppConfig
        key = AppConfig.load().api_keys.finnhub.strip()
    except Exception:
        pass
    if not key:
        key = os.environ.get("FINNHUB_API_KEY", "").strip()
    _finnhub_key_cache = key
    return key


# ── Data fetchers (zero LLM tokens) ────────────────────────────────────────

async def fetch_stock_quote(symbol: str) -> dict | None:
    """Fetch structured quote from Finnhub. Returns {price, change, change_pct, volume, ...}."""
    key = _finnhub_key()
    if not key:
        return None
    params = {"symbol": symbol.upper(), "token": key}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{_FINNHUB_BASE}/quote", params=params)
            data = resp.json()
        if data.get("c") is None:
            return None
        return {
            "price": data["c"],
            "change": data.get("d", 0) or 0,
            "change_pct": data.get("dp", 0) or 0,
            "open": data.get("o"),
            "high": data.get("h"),
            "low": data.get("l"),
            "prev_close": data.get("pc"),
        }
    except Exception as e:
        _log.error("fetch_stock", e, symbol=symbol)
        return None


async def fetch_web_signal(query: str, max_results: int = 3) -> dict | None:
    """Fetch web search results and return a content hash + snippet."""
    if _ddgs_breaker.is_open():
        return None
    try:
        from ddgs import DDGS
        loop = asyncio.get_event_loop()

        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        results = await loop.run_in_executor(None, _search)
        if not results:
            return None  # legitimate empty results, not a service failure

        _ddgs_breaker.record_success()

        # Build a content fingerprint from titles + snippets
        content = "\n".join(
            f"{r.get('title', '')} {r.get('body', '')}" for r in results
        )
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        snippet = results[0].get("body", "")[:200] if results else ""

        return {
            "content_hash": content_hash,
            "snippet": snippet,
            "full_content": content,
            "result_count": len(results),
        }
    except Exception as e:
        _ddgs_breaker.record_failure()
        _log.error("fetch_web", e, query=query)
        return None


# ── Change detection ────────────────────────────────────────────────────────

def compute_delta(
    source_hint: str,
    prev_json: str | None,
    new_value: dict,
    *,
    fetch_mode: str | None = None,
) -> dict | None:
    """Compare new value with previous. Returns delta dict or None if insignificant.

    For stock/API data: computes absolute and percentage price change.
    For web data: detects content hash change.

    ``fetch_mode`` overrides source_hint routing so that a stock hint
    that fell back to web search still produces a content_changed delta.
    When *None* the mode is auto-detected from the value keys.
    """
    if not prev_json:
        return None  # First poll, no comparison possible

    try:
        prev = json.loads(prev_json)
    except (json.JSONDecodeError, TypeError):
        return None

    # Determine comparison strategy: explicit fetch_mode > value-key detection > source_hint
    use_numeric = False
    if fetch_mode == "api":
        use_numeric = True
    elif fetch_mode == "web":
        use_numeric = False
    elif "price" in new_value:
        use_numeric = True
    elif "content_hash" in new_value:
        use_numeric = False
    elif source_hint.startswith("stock_") or source_hint in (
        "company_profile", "market_cap",
    ):
        use_numeric = True

    if use_numeric:
        # Numeric comparison on price
        prev_price = prev.get("price")
        new_price = new_value.get("price")
        if prev_price and new_price and prev_price > 0:
            abs_change = new_price - prev_price
            pct_change = (abs_change / prev_price) * 100
            return {
                "abs_change": round(abs_change, 4),
                "pct_change": round(pct_change, 2),
                "prev_price": prev_price,
                "new_price": new_price,
            }
        return None

    # Text/web content: hash comparison
    prev_hash = prev.get("content_hash", "")
    new_hash = new_value.get("content_hash", "")
    if prev_hash and new_hash and prev_hash != new_hash:
        return {
            "type": "content_changed",
            "prev_hash": prev_hash,
            "new_hash": new_hash,
            "snippet": new_value.get("snippet", ""),
            "full_content": new_value.get("full_content", ""),
        }

    return None


# Thresholds for triggering a signal event
_STOCK_CHANGE_THRESHOLD_PCT = 2.0  # 2% price change
_ALWAYS_LOG_THRESHOLD_PCT = 0.0    # log all stock polls for history


def is_significant(source_hint: str, delta: dict | None) -> bool:
    """Determine if a delta is significant enough to emit a signal."""
    if delta is None:
        return False

    if "pct_change" in delta:
        return abs(delta["pct_change"]) >= _STOCK_CHANGE_THRESHOLD_PCT

    if delta.get("type") == "content_changed":
        return True  # Any content change in proxy is notable

    return False


# ── SignalPoller ────────────────────────────────────────────────────────────

class SignalPoller:
    """Periodically polls factor data sources and detects significant changes.

    Zero LLM tokens consumed. All sensing is API calls + local math.
    """

    def __init__(
        self,
        store: InterestStore,
        on_signal: Callable[[dict], Coroutine] | None = None,
        enricher: "object | None" = None,
    ):
        self.store = store
        self.on_signal = on_signal
        self.enricher = enricher  # optional SignalEnricher
        self._poll_tasks: dict[str, asyncio.Task] = {}  # factor_id → task
        self._stopped = False

    async def start(self):
        """Load all pollable factors and start polling loops."""
        self._stopped = False
        factors = self.store.get_pollable_factors()
        for f in factors:
            self._start_factor(f)
        _log.event("poller_started", factor_count=len(factors))

    async def stop(self):
        """Stop all polling loops with two-phase graceful shutdown."""
        self._stopped = True
        tasks = list(self._poll_tasks.values())
        if tasks:
            # Phase 1: let in-flight work finish (polls check _stopped each iteration)
            done, pending = await asyncio.wait(tasks, timeout=5)
            # Phase 2: cancel any that didn't finish
            for task in pending:
                task.cancel()
            for task in pending:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._poll_tasks.clear()
        # Prune old signal_log on shutdown
        try:
            self.store.prune_signal_log()
        except Exception:
            pass
        _log.event("poller_stopped")

    def add_interest(self, interest: dict):
        """Start polling for a newly created interest's factors."""
        for f in interest.get("factors", []):
            if f.get("resolution") != "blind":
                factor_with_meta = {
                    **f,
                    "intent": interest.get("intent", ""),
                    "discord_thread_id": interest.get("discord_thread_id"),
                    "query": interest.get("query", ""),
                }
                self._start_factor(factor_with_meta)

    def remove_interest(self, interest_id: str):
        """Stop polling for an interest (pause/archive)."""
        to_remove = [
            fid for fid, task in self._poll_tasks.items()
            if getattr(task, '_interest_id', None) == interest_id
        ]
        for fid in to_remove:
            self._poll_tasks[fid].cancel()
            del self._poll_tasks[fid]

    def _start_factor(self, factor: dict):
        fid = factor["id"]
        if fid in self._poll_tasks:
            return
        task = asyncio.create_task(self._poll_loop(factor))
        task._interest_id = factor.get("interest_id", "")  # type: ignore[attr-defined]
        self._poll_tasks[fid] = task

    async def _poll_loop(self, factor: dict):
        """Polling loop for a single factor."""
        fid = factor["id"]
        interest_id = factor.get("interest_id", "")
        base_interval = factor.get("poll_interval", 300)
        interval = base_interval
        consecutive_failures = 0
        source_hint = factor.get("source_hint", "")
        bound_tool = factor.get("bound_tool")
        _prune_counter = 0

        # Stagger start to avoid thundering herd
        await asyncio.sleep(hash(fid) % min(interval, 30))

        while not self._stopped:
            try:
                value, fetch_mode = await self._fetch(factor, source_hint, bound_tool)
                if value is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        interval = min(interval * 2, base_interval * 4)
                else:
                    consecutive_failures = 0
                    interval = base_interval
                if value is not None:
                    now = datetime.now(timezone.utc).isoformat()
                    value_json = json.dumps(value, ensure_ascii=False)

                    # Compute delta from last stored value
                    prev_json = factor.get("last_value")
                    delta = compute_delta(
                        source_hint, prev_json, value,
                        fetch_mode=fetch_mode,
                    )

                    # Update stored value
                    self.store.update_factor_value(fid, value_json, now)
                    factor["last_value"] = value_json  # update local copy

                    # Log signal if significant
                    if is_significant(source_hint, delta):
                        delta_json = json.dumps(delta, ensure_ascii=False)
                        self.store.record_signal(
                            factor_id=fid,
                            interest_id=interest_id,
                            value_json=value_json,
                            delta_json=delta_json,
                        )
                        _log.event(
                            "signal_detected",
                            factor=factor.get("name", ""),
                            delta=delta,
                        )
                        signal = {
                            "type": "signal",
                            "interest_id": interest_id,
                            "factor_id": fid,
                            "factor_name": factor.get("name", ""),
                            "source_hint": source_hint,
                            "value": value,
                            "delta": delta,
                            "intent": factor.get("intent", ""),
                            "entity": factor.get("entity", ""),
                            "discord_thread_id": factor.get(
                                "discord_thread_id"
                            ),
                        }
                        # Enrich signal before emitting
                        if self.enricher is not None:
                            try:
                                signal = await self.enricher.enrich(signal)
                            except Exception as e:
                                _log.error("signal_enrich", e)
                        # Emit callback
                        if self.on_signal:
                            try:
                                await self.on_signal(signal)
                            except Exception as e:
                                _log.error("signal_callback", e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                _log.error(
                    "poll_error", e,
                    factor=factor.get("name", ""), factor_id=fid,
                )

            # Periodic signal_log pruning (roughly once per hour)
            _prune_counter += 1
            if _prune_counter % 100 == 0:
                try:
                    self.store.prune_signal_log()
                except Exception:
                    pass

            await asyncio.sleep(interval)

    async def _fetch(
        self, factor: dict, source_hint: str, bound_tool: str | None,
    ) -> tuple[dict | None, str | None]:
        """Fetch data for a single factor. Zero tokens.

        Returns ``(value, fetch_mode)`` where *fetch_mode* is ``"api"``
        or ``"web"`` (or ``None`` on failure).  Stock hints try the API
        first then fall back to web search so the signal is never lost.
        """
        entity = factor.get("entity", "")

        # EXECUTABLE: try direct API first for stock hints
        if source_hint in (
            "stock_price", "stock_quote", "stock_volume",
            "stock_metrics", "market_cap",
        ):
            result = await fetch_stock_quote(entity)
            if result is not None:
                return result, "api"
            # API unavailable — fall through to web search fallback
            query = f"{entity} {factor.get('name', '')}".strip()
            result = await fetch_web_signal(query)
            if result is not None:
                return result, "web"
            return None, None

        # PROXY: web search
        if bound_tool == "web_search" or source_hint in (
            "news_search", "company_news", "industry_news",
            "regulatory_news", "financial_earnings", "earnings_calendar",
            "macro_indicator", "interest_rate", "currency_rate",
            "tech_trend", "github_activity", "product_release",
            "social_sentiment", "web_content", "rss_feed",
            "insider_trading", "institutional_holdings",
            "analyst_rating", "credit_rating",
            "sec_filing", "patent_filing", "regulatory_action",
        ):
            query = f"{entity} {factor.get('name', '')}".strip()
            result = await fetch_web_signal(query)
            if result is not None:
                return result, "web"
            return None, None

        return None, None
