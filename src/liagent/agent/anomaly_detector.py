"""Anomaly Detector — Layer 1 of the attention system.

Cross-factor coincidence detection within time windows.
Zero LLM tokens — purely statistical aggregation.

Architecture:
  SignalPoller.on_signal → AnomalyDetector.ingest()
    → buffer signals per interest in a time window
    → when window expires or threshold reached, evaluate
    → if anomaly score ≥ threshold → emit on_anomaly callback
    → otherwise, emit individual signals as low-priority notifications

Key rules:
  1. A single PROXY signal alone never triggers an anomaly (too noisy).
  2. A single EXECUTABLE signal with large delta can trigger solo.
  3. Multiple corroborating signals (any combination) amplify the score.
  4. Time window: signals within WINDOW_SECONDS are grouped.
"""

import asyncio
import time
from typing import Callable, Coroutine

from ..logging import get_logger

_log = get_logger("anomaly_detector")

# ── Configuration ────────────────────────────────────────────────────────

WINDOW_SECONDS = 300  # 5-minute grouping window
ANOMALY_THRESHOLD = 1.5  # weighted score needed to emit anomaly
SOLO_EXECUTABLE_THRESHOLD_PCT = 5.0  # single stock signal ≥5% triggers solo


# ── Scoring ──────────────────────────────────────────────────────────────

def score_signal(signal: dict) -> float:
    """Compute a 0–2.0 contribution score for a single signal.

    Enriched signals (with severity+confidence) use the unified formula.
    Legacy unenriched signals fall back to the original tiered scoring.
    """
    delta = signal.get("delta") or {}

    # ── Enriched path (unified schema from SignalEnricher) ──
    if "severity" in delta and "confidence" in delta:
        return min(2.0, delta["severity"] * delta["confidence"] * 2.0)

    # ── Legacy fallback (unenriched signals) ──
    pct = delta.get("pct_change")
    if pct is not None:
        abs_pct = abs(pct)
        if abs_pct >= 10.0:
            return 2.0
        if abs_pct >= 5.0:
            return 1.5
        if abs_pct >= 3.0:
            return 1.0
        if abs_pct >= 2.0:
            return 0.6
        return 0.3  # below threshold but still detected

    # Content change (proxy) — flat score, needs corroboration
    if delta.get("type") == "content_changed":
        return 0.5

    return 0.2


def evaluate_window(signals: list[dict]) -> dict | None:
    """Evaluate a window of signals from the same interest.

    Returns an anomaly dict if the combined score exceeds threshold,
    or None if signals are below anomaly level.

    Anomaly dict:
    {
        "type": "anomaly",
        "interest_id": str,
        "score": float,
        "signal_count": int,
        "signals": list[dict],  # contributing signals
        "summary": str,         # human-readable description
        "intent": str,
        "discord_thread_id": str | None,
    }
    """
    if not signals:
        return None

    # Compute weighted aggregate score
    total_score = 0.0
    factor_scores: dict[str, float] = {}  # factor_id → best score
    for sig in signals:
        fid = sig.get("factor_id", "")
        s = score_signal(sig)
        # Keep best score per factor (avoid double-counting rapid polls)
        if s > factor_scores.get(fid, 0):
            factor_scores[fid] = s

    total_score = sum(factor_scores.values())

    # Corroboration bonus: multiple distinct factors add 20% each
    n_factors = len(factor_scores)
    if n_factors >= 2:
        total_score *= 1.0 + 0.2 * (n_factors - 1)

    # Solo high-severity trigger (source-agnostic, replaces old EXECUTABLE-only rule)
    if n_factors == 1:
        sole_delta = signals[-1].get("delta") or {}
        # Enriched path: high severity + high confidence → force to threshold
        if sole_delta.get("severity", 0) >= 0.85 and sole_delta.get("confidence", 0) >= 0.7:
            total_score = max(total_score, ANOMALY_THRESHOLD)
        # Legacy path: single EXECUTABLE with huge delta triggers solo
        pct = sole_delta.get("pct_change")
        if pct is not None and abs(pct) >= SOLO_EXECUTABLE_THRESHOLD_PCT:
            total_score = max(total_score, ANOMALY_THRESHOLD)

    if total_score < ANOMALY_THRESHOLD:
        return None

    # Build summary
    parts = []
    for sig in signals:
        delta = sig.get("delta") or {}
        name = sig.get("factor_name", "?")
        # Enriched path: use key_fact if available
        key_fact = delta.get("key_fact")
        if key_fact:
            parts.append(f"{name}: {key_fact}")
            continue
        # Legacy path
        raw = delta.get("raw_delta", delta)
        pct = raw.get("pct_change")
        if pct is not None:
            direction = "+" if pct > 0 else ""
            parts.append(f"{name}: {direction}{pct:.1f}%")
        elif raw.get("type") == "content_changed":
            snippet = raw.get("snippet", "")[:80]
            parts.append(f"{name}: content changed — {snippet}")
        else:
            parts.append(f"{name}: signal detected")

    ref = signals[0]
    return {
        "type": "anomaly",
        "interest_id": ref.get("interest_id", ""),
        "score": round(total_score, 2),
        "signal_count": len(signals),
        "factor_count": n_factors,
        "signals": signals,
        "summary": "; ".join(parts),
        "intent": ref.get("intent", ""),
        "discord_thread_id": ref.get("discord_thread_id"),
    }


# ── AnomalyDetector ─────────────────────────────────────────────────────

class AnomalyDetector:
    """Buffers signals per interest and detects cross-factor anomalies.

    Zero LLM tokens. Purely time-window + weighted scoring.
    """

    def __init__(
        self,
        on_anomaly: Callable[[dict], Coroutine] | None = None,
        on_signal_passthrough: Callable[[dict], Coroutine] | None = None,
        window_seconds: float = WINDOW_SECONDS,
        anomaly_threshold: float = ANOMALY_THRESHOLD,
    ):
        self.on_anomaly = on_anomaly
        self.on_signal_passthrough = on_signal_passthrough
        self._window_seconds = window_seconds
        self._anomaly_threshold = anomaly_threshold

        # interest_id → list of (timestamp, signal_dict)
        self._buffers: dict[str, list[tuple[float, dict]]] = {}
        # interest_id → scheduled flush task
        self._flush_tasks: dict[str, asyncio.Task] = {}

    async def ingest(self, signal: dict):
        """Receive a signal from SignalPoller. Buffer and evaluate."""
        iid = signal.get("interest_id", "")
        if not iid:
            return

        now = time.monotonic()
        buf = self._buffers.setdefault(iid, [])
        buf.append((now, signal))

        # Prune stale entries beyond 2x window
        cutoff = now - self._window_seconds * 2
        self._buffers[iid] = [(t, s) for t, s in buf if t > cutoff]

        # Cleanup stale buffers (interests not updated in 1 hour)
        self._cleanup_stale_buffers(now)

        # Check if we should evaluate immediately (high-urgency signal)
        if self._is_urgent(signal):
            await self._flush(iid)
            return

        # Schedule a deferred flush at window expiry (if not already scheduled)
        if iid not in self._flush_tasks or self._flush_tasks[iid].done():
            self._flush_tasks[iid] = asyncio.create_task(
                self._deferred_flush(iid)
            )

    async def _deferred_flush(self, interest_id: str):
        """Wait for the window to close, then evaluate."""
        await asyncio.sleep(self._window_seconds)
        await self._flush(interest_id)

    async def _flush(self, interest_id: str):
        """Evaluate buffered signals for an interest and emit results."""
        buf = self._buffers.pop(interest_id, [])
        if not buf:
            return

        # Cancel any pending deferred flush
        task = self._flush_tasks.pop(interest_id, None)
        if task and not task.done():
            task.cancel()

        signals = [s for _, s in buf]
        anomaly = evaluate_window(signals)

        if anomaly:
            _log.event(
                "anomaly_detected",
                interest_id=interest_id,
                score=anomaly["score"],
                signal_count=anomaly["signal_count"],
                factor_count=anomaly["factor_count"],
            )
            if self.on_anomaly:
                try:
                    await self.on_anomaly(anomaly)
                except Exception as e:
                    _log.error("anomaly_callback", e)
        else:
            # Pass through individual signals as low-priority notifications
            if self.on_signal_passthrough:
                for sig in signals:
                    try:
                        await self.on_signal_passthrough(sig)
                    except Exception as e:
                        _log.error("signal_passthrough", e)

    def _cleanup_stale_buffers(self, now: float):
        """Remove buffer entries for interests not updated in 1 hour."""
        stale_threshold = 3600.0  # 1 hour
        stale_ids = [
            iid for iid, buf in self._buffers.items()
            if buf and (now - buf[-1][0]) > stale_threshold
        ]
        for iid in stale_ids:
            del self._buffers[iid]
            task = self._flush_tasks.pop(iid, None)
            if task and not task.done():
                task.cancel()

    def _is_urgent(self, signal: dict) -> bool:
        """Check if this signal warrants immediate evaluation.

        Urgent = high enriched severity OR large executable delta.
        """
        delta = signal.get("delta") or {}
        # Enriched path: high severity is urgent
        if delta.get("severity", 0) >= 0.85:
            return True
        # Legacy path: large stock move
        pct = delta.get("pct_change")
        if pct is not None and abs(pct) >= SOLO_EXECUTABLE_THRESHOLD_PCT:
            return True
        return False

    async def shutdown(self):
        """Flush all pending buffers and cancel timers."""
        for task in self._flush_tasks.values():
            task.cancel()
        self._flush_tasks.clear()

        # Flush remaining buffers
        for iid in list(self._buffers.keys()):
            await self._flush(iid)
