#!/usr/bin/env python3
"""Live end-to-end test of the perception loop with real APIs.

Uses:
  - DuckDuckGo web search (no API key needed)
  - Moonshot/Kimi LLM API (from config.json)
  - Finnhub stock API (optional, falls back to web search)

Run:
  ./venv/bin/python tests/manual/live_perception_test.py
"""

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

# Ensure src/ is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from liagent.config import AppConfig
from liagent.engine.engine_manager import EngineManager
from liagent.agent.interest import InterestStore, ResolvedFactor, Resolution
from liagent.agent.signal_poller import (
    SignalPoller, compute_delta, is_significant,
    fetch_stock_quote, fetch_web_signal,
)
from liagent.agent.signal_enricher import SignalEnricher
from liagent.agent.anomaly_detector import AnomalyDetector, evaluate_window, score_signal


# ── Pretty printing ──────────────────────────────────────────────────────

_BOLD = "\033[1m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def banner(text: str):
    print(f"\n{_BOLD}{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}{_RESET}\n")


def step(n: int, text: str):
    print(f"{_CYAN}[Step {n}]{_RESET} {text}")


def ok(text: str):
    print(f"  {_GREEN}✓{_RESET} {text}")


def warn(text: str):
    print(f"  {_YELLOW}!{_RESET} {text}")


def fail(text: str):
    print(f"  {_RED}✗{_RESET} {text}")


def info(key: str, value):
    print(f"  {_DIM}{key}:{_RESET} {value}")


def delta_summary(delta: dict) -> str:
    if "severity" in delta:
        sent_icon = "📈" if delta.get("sentiment", 0) > 0 else "📉" if delta.get("sentiment", 0) < 0 else "➡️"
        return (
            f"{sent_icon} severity={delta['severity']:.2f} "
            f"sentiment={delta.get('sentiment', 0):.2f} "
            f"conf={delta.get('confidence', 0):.2f} "
            f"type={delta.get('event_type', '?')} "
            f"| {delta.get('key_fact', '')}"
        )
    if "pct_change" in delta:
        return f"pct_change={delta['pct_change']:.2f}%"
    if delta.get("type") == "content_changed":
        return f"content_changed | {delta.get('snippet', '')[:80]}"
    return str(delta)[:120]


# ── Main ─────────────────────────────────────────────────────────────────

async def main():
    banner("Live Perception Loop Test")

    # ── 0. Load config & build engine ──
    step(0, "Loading config & building engine (LLM API)...")
    config = AppConfig.load()
    info("LLM backend", config.llm.backend)
    info("LLM model", config.llm.api_model)
    info("LLM base URL", config.llm.api_base_url)

    engine = EngineManager(config, voice_mode=False)
    ok("EngineManager created")

    # Quick sanity: can we call generate_reasoning?
    try:
        test_resp = await engine.generate_reasoning(
            [{"role": "user", "content": "Say 'hello' in one word."}],
            max_tokens=10, temperature=0.0, enable_thinking=False,
        )
        ok(f"LLM API works: {test_resp.strip()!r}")
    except Exception as e:
        fail(f"LLM API failed: {e}")
        return

    # ── 1. Create interest store (temp DB) ──
    step(1, "Creating interest with 2 factors (AAPL stock + regulatory news)...")
    tmp_dir = tempfile.mkdtemp(prefix="liagent_live_")
    store = InterestStore(db_path=Path(tmp_dir) / "live_test.db")

    factors = [
        ResolvedFactor(
            name="AAPL Price",
            source_hint="stock_price",
            entity="AAPL",
            frequency="realtime",
            resolution=Resolution.EXECUTABLE,
            bound_tool="stock",
            reliability=1.0,
            weight=1.0,
        ),
        ResolvedFactor(
            name="Apple Regulatory News",
            source_hint="regulatory_news",
            entity="Apple",
            frequency="hourly",
            resolution=Resolution.PROXY,
            bound_tool="web_search",
            reliability=0.7,
            weight=0.7,
        ),
    ]

    interest = store.create_interest(
        query="Watch AAPL and focus on regulatory risk",
        intent="Monitor AAPL for regulatory risks",
        context={"cost_basis": 200, "position_size": 50},
        factors=factors,
    )
    ok(f"Interest created: id={interest['id']}")
    info("factors", len(interest["factors"]))
    info("coverage", f"{interest['coverage_ratio']:.1%}")

    # ── 2. Real data fetch ──
    step(2, "Fetching real data from APIs...")

    # 2a. Stock quote (Finnhub or web fallback)
    t0 = time.perf_counter()
    stock_result = await fetch_stock_quote("AAPL")
    stock_elapsed = (time.perf_counter() - t0) * 1000

    stock_mode = "api"
    if stock_result is not None:
        ok(f"Finnhub stock API: AAPL=${stock_result['price']:.2f} ({stock_elapsed:.0f}ms)")
    else:
        warn("Finnhub unavailable (no API key?), falling back to web search...")
        stock_mode = "web"
        t0 = time.perf_counter()
        stock_result = await fetch_web_signal("AAPL stock price today")
        stock_elapsed = (time.perf_counter() - t0) * 1000
        if stock_result:
            ok(f"Web fallback: hash={stock_result['content_hash']} ({stock_elapsed:.0f}ms)")
            info("snippet", stock_result["snippet"][:120])
        else:
            fail("Web search also failed!")
            return

    # 2b. Regulatory news (always web)
    t0 = time.perf_counter()
    web_result = await fetch_web_signal("Apple regulatory news SEC investigation 2025 2026")
    web_elapsed = (time.perf_counter() - t0) * 1000

    if web_result:
        ok(f"Web search (regulatory): hash={web_result['content_hash']} ({web_elapsed:.0f}ms)")
        info("snippet", web_result["snippet"][:120])
        info("results", web_result["result_count"])
    else:
        fail("Web search failed for regulatory news!")
        return

    # ── 3. Compute deltas ──
    step(3, "Computing deltas (simulating second poll)...")

    # For stock: simulate prior value as 5% lower (to force a significant delta)
    if stock_mode == "api":
        prior_price = stock_result["price"] * 0.95
        fake_prev_stock = json.dumps({"price": prior_price})
        stock_delta = compute_delta("stock_price", fake_prev_stock, stock_result, fetch_mode="api")
    else:
        fake_prev_web = json.dumps({"content_hash": "fake_old_hash_000"})
        stock_delta = compute_delta("stock_price", fake_prev_web, stock_result, fetch_mode="web")

    if stock_delta:
        ok(f"Stock delta: {delta_summary(stock_delta)}")
        info("significant", is_significant("stock_price", stock_delta))
    else:
        warn("No stock delta (content unchanged)")

    # Web: simulate different prior hash
    fake_prev_reg = json.dumps({"content_hash": "old_regulatory_hash"})
    web_delta = compute_delta("regulatory_news", fake_prev_reg, web_result, fetch_mode="web")
    if web_delta:
        ok(f"Web delta: {delta_summary(web_delta)}")
        info("significant", is_significant("regulatory_news", web_delta))
    else:
        warn("No web delta (content unchanged)")

    # ── 4. Enrichment (real LLM call) ──
    step(4, "Enriching signals via LLM...")
    enricher = SignalEnricher(engine)
    pollable = store.get_pollable_factors()
    stock_factor = next(f for f in pollable if f["source_hint"] == "stock_price")
    web_factor = next(f for f in pollable if f["source_hint"] == "regulatory_news")

    enriched_signals = []

    if stock_delta and is_significant("stock_price", stock_delta):
        stock_signal = {
            "type": "signal",
            "interest_id": interest["id"],
            "factor_id": stock_factor["id"],
            "factor_name": "AAPL Price",
            "source_hint": "stock_price",
            "value": stock_result,
            "delta": stock_delta,
            "intent": "Monitor AAPL for regulatory risks",
            "entity": "AAPL",
        }
        t0 = time.perf_counter()
        stock_signal = await enricher.enrich(stock_signal)
        enrich_elapsed = (time.perf_counter() - t0) * 1000
        ok(f"Stock enriched ({enrich_elapsed:.0f}ms): {delta_summary(stock_signal['delta'])}")
        enriched_signals.append(stock_signal)

    if web_delta and is_significant("regulatory_news", web_delta):
        web_signal = {
            "type": "signal",
            "interest_id": interest["id"],
            "factor_id": web_factor["id"],
            "factor_name": "Apple Regulatory News",
            "source_hint": "regulatory_news",
            "value": web_result,
            "delta": web_delta,
            "intent": "Monitor AAPL for regulatory risks",
            "entity": "Apple",
        }
        t0 = time.perf_counter()
        web_signal = await enricher.enrich(web_signal)
        enrich_elapsed = (time.perf_counter() - t0) * 1000
        ok(f"Web enriched ({enrich_elapsed:.0f}ms): {delta_summary(web_signal['delta'])}")
        enriched_signals.append(web_signal)

    if not enriched_signals:
        warn("No significant signals to process")
        return

    # ── 5. Anomaly detection ──
    step(5, "Running anomaly detection...")

    for sig in enriched_signals:
        score = score_signal(sig)
        info(f"  {sig['factor_name']} score", f"{score:.2f}")

    anomaly = evaluate_window(enriched_signals)
    if anomaly:
        ok(f"ANOMALY DETECTED!")
        info("score", f"{anomaly['score']:.2f}")
        info("factors", anomaly["factor_count"])
        info("signals", anomaly["signal_count"])
        info("summary", anomaly["summary"])
    else:
        warn("No anomaly (below threshold 1.5)")
        info("signals", len(enriched_signals))
        info("hint", "Individual signals will be passed through as low-priority notifications")

    # ── 6. Full async pipeline test ──
    step(6, "Testing async pipeline (ingest → anomaly detector)...")

    anomaly_results = []
    passthrough_results = []

    async def on_anomaly(a):
        anomaly_results.append(a)

    async def on_passthrough(s):
        passthrough_results.append(s)

    detector = AnomalyDetector(
        on_anomaly=on_anomaly,
        on_signal_passthrough=on_passthrough,
        window_seconds=0.5,  # short window for test
    )

    for sig in enriched_signals:
        await detector.ingest(sig)

    # Wait for deferred flush if no urgent signal triggered immediate flush
    await asyncio.sleep(1.0)

    if anomaly_results:
        ok(f"Anomaly callback fired: score={anomaly_results[0]['score']:.2f}")
    elif passthrough_results:
        ok(f"Passthrough callback fired: {len(passthrough_results)} signal(s)")
        for p in passthrough_results:
            info("  →", delta_summary(p.get("delta", {})))
    else:
        warn("No callbacks fired (unexpected)")

    # ── Summary ──
    banner("Results Summary")
    print(f"  Data sources:  Finnhub={'✓' if stock_mode == 'api' else '✗ (web fallback)'}  DuckDuckGo=✓")
    print(f"  LLM enrichment: ✓ ({config.llm.api_model})")
    print(f"  Signals:        {len(enriched_signals)}")
    print(f"  Anomalies:      {len(anomaly_results)}")
    print(f"  Passthroughs:   {len(passthrough_results)}")
    print()

    # Print enriched deltas as JSON for inspection
    print(f"{_DIM}Enriched deltas:{_RESET}")
    for sig in enriched_signals:
        d = sig["delta"]
        # Don't print raw_delta to keep output clean
        clean = {k: v for k, v in d.items() if k != "raw_delta"}
        print(f"  {sig['factor_name']}: {json.dumps(clean, ensure_ascii=False, indent=4)}")

    print(f"\n{_GREEN}Live test complete.{_RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
