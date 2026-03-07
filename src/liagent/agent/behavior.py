"""Proactive intelligence: behavior signal collection, pattern detection, action routing."""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ..logging import StructuredLogger

_log = StructuredLogger("behavior")

# Tool → domain classification (hardcoded, zero-LLM)
TOOL_DOMAIN_MAP: dict[str, str] = {
    "stock": "stock",
    "web_search": "news",
    "web_fetch": "news",
    "python_exec": "coding",
    "run_tests": "coding",
    "lint_code": "coding",
    "verify_syntax": "coding",
}

# Stock ticker regex: uppercase 1-5 letters, context-gated
_TICKER_RE = re.compile(r'\b([A-Z]{1,5})\b')
_TICKER_CONTEXT = re.compile(
    r'(?:price|stock|ticker|quote|shares|market|buy|sell|gain|drop|trade)',
    re.IGNORECASE,
)
_TICKER_STOPWORDS = {
    "API", "GET", "POST", "PUT", "URL", "HTTP", "JSON", "HTML", "CSS",
    "SQL", "RAM", "CPU", "GPU", "LLM", "MLX", "USD", "EUR", "GBP",
    "THE", "AND", "FOR", "NOT", "BUT", "ALL", "ANY", "ARE", "CAN",
}


def extract_topics_from_tool_args(tool_name: str, tool_args: dict) -> list[str]:
    """Zero-LLM entity extraction from tool arguments."""
    topics: list[str] = []

    if tool_name == "stock":
        symbol = tool_args.get("symbol", "") or tool_args.get("ticker", "")
        if symbol:
            topics.append(symbol.upper().strip())

    elif tool_name in ("web_search", "web_fetch"):
        query = tool_args.get("query", "") or tool_args.get("url", "")
        if _TICKER_CONTEXT.search(query):
            for m in _TICKER_RE.finditer(query):
                t = m.group(1)
                if t not in _TICKER_STOPWORDS and len(t) >= 2:
                    topics.append(t)

    return topics


def parse_behavior_signals(raw: str) -> list[dict[str, Any]]:
    """Parse LLM output into validated behavior signals.

    Filters: confidence >= 0.6, max 6 entries, strips markdown fences.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    text = text.strip()

    try:
        entries = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []

    if not isinstance(entries, list):
        return []

    valid = []
    seen_keys: set[str] = set()
    for e in entries:
        if not isinstance(e, dict):
            continue
        key = e.get("key", "")
        if not key or key in seen_keys:
            continue
        confidence = float(e.get("confidence", 0))
        if confidence < 0.6:
            continue
        seen_keys.add(key)
        valid.append({
            "signal_type": e.get("signal_type", "intent"),
            "key": key,
            "domain": e.get("domain", "general"),
            "confidence": confidence,
            "metadata": json.dumps(e.get("metadata", {}), ensure_ascii=False),
        })
        if len(valid) >= 6:
            break

    return valid


class BehaviorSignalStore:
    """Queue-buffered writes to behavior_signals table.

    L0 signals are enqueued in memory and batch-flushed to SQLite
    asynchronously (every flush_threshold records or on explicit flush).
    """

    def __init__(self, db_path: str, *, flush_threshold: int = 10):
        self.db_path = db_path
        self.flush_threshold = flush_threshold
        self._queue: deque[dict[str, Any]] = deque()
        self._seen: set[tuple[str, str, str]] = set()  # (type, key, session_id) dedup

    def record(
        self,
        signal_type: str,
        key: str,
        *,
        domain: str = "",
        source_origin: str = "user",
        metadata: str = "{}",
        session_id: str = "",
        hour: int | None = None,
        weekday: int | None = None,
    ) -> None:
        """Enqueue a signal. Deduplicates by (signal_type, key, session_id) within a session."""
        dedup_key = (signal_type, key, session_id)
        if dedup_key in self._seen:
            return
        self._seen.add(dedup_key)

        now = datetime.now(timezone.utc)
        if hour is None or weekday is None:
            local_now = datetime.now().astimezone()
            if hour is None:
                hour = local_now.hour
            if weekday is None:
                weekday = local_now.weekday()

        self._queue.append({
            "signal_type": signal_type,
            "key": key,
            "domain": domain,
            "source_origin": source_origin,
            "metadata": metadata,
            "session_id": session_id,
            "hour": hour,
            "weekday": weekday,
            "created_at": now.isoformat(),
        })

        if len(self._queue) >= self.flush_threshold:
            self.flush()

    def flush(self) -> int:
        """Write queued signals to SQLite. Returns count written."""
        if not self._queue:
            return 0
        batch = list(self._queue)
        self._queue.clear()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    "INSERT INTO behavior_signals "
                    "(signal_type, key, domain, source_origin, metadata, "
                    "session_id, hour, weekday, created_at) "
                    "VALUES (:signal_type, :key, :domain, :source_origin, "
                    ":metadata, :session_id, :hour, :weekday, :created_at)",
                    batch,
                )
            return len(batch)
        except Exception as e:
            _log.error("behavior_signal_flush", e)
            self._queue.extend(batch)  # re-queue for retry
            return 0

    def reset_session_dedup(self) -> None:
        """Clear per-session dedup set (call at session start)."""
        self._seen.clear()

    def get_signals(
        self, signal_type: str, key: str, *, days: int = 30
    ) -> list[dict[str, Any]]:
        """Read signals for a given type+key within retention window."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM behavior_signals "
                "WHERE signal_type = ? AND key = ? AND created_at > ? "
                "ORDER BY created_at DESC",
                (signal_type, key, cutoff),
            ).fetchall()
        return [dict(r) for r in rows]

    def prune(self, *, days: int = 90) -> int:
        """Delete signals older than days. Returns rows deleted."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM behavior_signals WHERE created_at < ?", (cutoff,)
            )
            return cursor.rowcount


class DomainFeedback:
    """Track per-domain acceptance rates and consecutive-ignore counts."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _ensure_row(self, conn: sqlite3.Connection, domain: str, stype: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO domain_feedback "
            "(domain, suggestion_type, updated_at) VALUES (?, ?, ?)",
            (domain, stype, now),
        )

    def accept_rate(self, domain: str, suggestion_type: str) -> float:
        """Laplace-smoothed: (accepted+1) / (total+2)."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT total_accepted, total_suggested FROM domain_feedback "
                "WHERE domain = ? AND suggestion_type = ?",
                (domain, suggestion_type),
            ).fetchone()
        if not row:
            return 1 / 2  # prior
        accepted, total = row
        return (accepted + 1) / (total + 2)

    def consecutive_ignored(self, domain: str, suggestion_type: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT consecutive_ignored FROM domain_feedback "
                "WHERE domain = ? AND suggestion_type = ?",
                (domain, suggestion_type),
            ).fetchone()
        return row[0] if row else 0

    def record_suggested(self, domain: str, suggestion_type: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_row(conn, domain, suggestion_type)
            conn.execute(
                "UPDATE domain_feedback "
                "SET total_suggested = total_suggested + 1, updated_at = ? "
                "WHERE domain = ? AND suggestion_type = ?",
                (now, domain, suggestion_type),
            )

    def record_accepted(self, domain: str, suggestion_type: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_row(conn, domain, suggestion_type)
            conn.execute(
                "UPDATE domain_feedback "
                "SET total_accepted = total_accepted + 1, "
                "    total_suggested = total_suggested + 1, "
                "    consecutive_ignored = 0, updated_at = ? "
                "WHERE domain = ? AND suggestion_type = ?",
                (now, domain, suggestion_type),
            )

    def record_accepted_outcome(self, domain: str, suggestion_type: str) -> None:
        """Record an accepted suggestion that was already counted by record_suggested()."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_row(conn, domain, suggestion_type)
            conn.execute(
                "UPDATE domain_feedback "
                "SET total_accepted = total_accepted + 1, "
                "    consecutive_ignored = 0, updated_at = ? "
                "WHERE domain = ? AND suggestion_type = ?",
                (now, domain, suggestion_type),
            )

    def record_rejected(self, domain: str, suggestion_type: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_row(conn, domain, suggestion_type)
            conn.execute(
                "UPDATE domain_feedback "
                "SET total_rejected = total_rejected + 1, "
                "    total_suggested = total_suggested + 1, "
                "    consecutive_ignored = 0, updated_at = ? "
                "WHERE domain = ? AND suggestion_type = ?",
                (now, domain, suggestion_type),
            )

    def record_rejected_outcome(self, domain: str, suggestion_type: str) -> None:
        """Record a rejection after the suggestion impression was already counted."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_row(conn, domain, suggestion_type)
            conn.execute(
                "UPDATE domain_feedback "
                "SET total_rejected = total_rejected + 1, "
                "    consecutive_ignored = 0, updated_at = ? "
                "WHERE domain = ? AND suggestion_type = ?",
                (now, domain, suggestion_type),
            )

    def record_ignored(self, domain: str, suggestion_type: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_row(conn, domain, suggestion_type)
            conn.execute(
                "UPDATE domain_feedback "
                "SET total_ignored = total_ignored + 1, "
                "    total_suggested = total_suggested + 1, "
                "    consecutive_ignored = consecutive_ignored + 1, "
                "    updated_at = ? "
                "WHERE domain = ? AND suggestion_type = ?",
                (now, domain, suggestion_type),
            )

    def record_ignored_outcome(self, domain: str, suggestion_type: str) -> None:
        """Record an ignored suggestion after its impressions were already counted."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_row(conn, domain, suggestion_type)
            conn.execute(
                "UPDATE domain_feedback "
                "SET total_ignored = total_ignored + 1, "
                "    consecutive_ignored = consecutive_ignored + 1, "
                "    updated_at = ? "
                "WHERE domain = ? AND suggestion_type = ?",
                (now, domain, suggestion_type),
            )


class BehaviorPatternDetector:
    """Detect recurring behavioral patterns from behavior_signals.

    Detection prefers session-scoped patterns when one session clearly has
    enough repeated evidence. If no single session qualifies, it falls back to
    an aggregate cross-session pattern for the same (signal_type, key, domain).

    Confidence = clamp(base + day_bonus + recency + temporal, 0.0, 1.0)
    """

    def __init__(self, db_path: str, *, lookback_days: int = 30):
        self.db_path = db_path
        self.lookback_days = lookback_days

    def detect(self) -> list[dict[str, Any]]:
        """Return candidate patterns [{pattern_key, domain, confidence, count, distinct_days}]."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        ).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            scoped_rows = conn.execute(
                """
                SELECT signal_type, key, domain,
                       COALESCE(NULLIF(session_id, ''), '') as session_id,
                       COUNT(*) as cnt,
                       COUNT(DISTINCT date(created_at)) as distinct_days,
                       MAX(created_at) as last_seen
                FROM behavior_signals
                WHERE source_origin = 'user'
                  AND created_at > ?
                  AND COALESCE(NULLIF(session_id, ''), '') != ''
                GROUP BY signal_type, key, domain, COALESCE(NULLIF(session_id, ''), '')
                HAVING cnt >= 3 AND COUNT(DISTINCT date(created_at)) >= 2
                """,
                (cutoff,),
            ).fetchall()
            global_rows = conn.execute(
                """
                SELECT signal_type, key, domain,
                       COUNT(*) as cnt,
                       COUNT(DISTINCT date(created_at)) as distinct_days,
                       MAX(created_at) as last_seen
                FROM behavior_signals
                WHERE source_origin = 'user'
                  AND created_at > ?
                GROUP BY signal_type, key, domain
                HAVING cnt >= 3 AND COUNT(DISTINCT date(created_at)) >= 2
                """,
                (cutoff,),
            ).fetchall()

        candidates = []
        covered_keys: set[tuple[str, str, str]] = set()

        for row in scoped_rows:
            row = dict(row)
            session_scope = str(row.get("session_id", "") or "").strip()
            covered_keys.add((row["signal_type"], row["key"], row["domain"]))
            pattern_key = (
                f"{row['signal_type']}:{row['key']}:{session_scope}"
                if session_scope else f"{row['signal_type']}:{row['key']}"
            )
            confidence = self._compute_confidence(
                row["cnt"], row["distinct_days"], row["last_seen"],
                row["signal_type"], row["key"],
                session_id=session_scope or None,
            )
            candidates.append({
                "pattern_key": pattern_key,
                "domain": row["domain"],
                "confidence": confidence,
                "count": row["cnt"],
                "distinct_days": row["distinct_days"],
                "signal_type": row["signal_type"],
                "key": row["key"],
                "target_session_id": row.get("session_id", "") or None,
            })

        for row in global_rows:
            row = dict(row)
            key = (row["signal_type"], row["key"], row["domain"])
            if key in covered_keys:
                continue
            confidence = self._compute_confidence(
                row["cnt"], row["distinct_days"], row["last_seen"],
                row["signal_type"], row["key"],
            )
            candidates.append({
                "pattern_key": f"{row['signal_type']}:{row['key']}",
                "domain": row["domain"],
                "confidence": confidence,
                "count": row["cnt"],
                "distinct_days": row["distinct_days"],
                "signal_type": row["signal_type"],
                "key": row["key"],
                "target_session_id": None,
            })

        return candidates

    def _compute_confidence(
        self,
        count: int,
        distinct_days: int,
        last_seen: str,
        signal_type: str,
        key: str,
        *,
        session_id: str | None = None,
    ) -> float:
        base = min(1.0, count / 10)
        day_bonus = min(0.3, distinct_days * 0.06)

        # Recency: 1.0 if today, decays by 0.1 per day
        try:
            last_dt = datetime.fromisoformat(last_seen)
            days_since = max(0.0, (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400)
        except (ValueError, TypeError):
            days_since = 7.0
        recency = max(0.0, 1.0 - days_since * 0.1)

        temporal = self._temporal_regularity(signal_type, key, session_id=session_id) * 0.2

        raw = base + day_bonus + recency + temporal
        return max(0.0, min(1.0, raw))

    def _temporal_regularity(
        self,
        signal_type: str,
        key: str,
        *,
        session_id: str | None = None,
    ) -> float:
        """Compute regularity score from hour histogram entropy.

        Low entropy (concentrated in few hours) -> high regularity -> score near 1.0.
        High entropy (spread across hours) -> low regularity -> score near 0.0.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        ).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT hour FROM behavior_signals "
                    "WHERE signal_type = ? AND key = ? "
                    "AND source_origin = 'user' AND created_at > ? "
                    "AND COALESCE(NULLIF(session_id, ''), '') = ?",
                    (signal_type, key, cutoff, session_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT hour FROM behavior_signals "
                    "WHERE signal_type = ? AND key = ? "
                    "AND source_origin = 'user' AND created_at > ?",
                    (signal_type, key, cutoff),
                ).fetchall()

        if len(rows) < 3:
            return 0.0

        # Build 24-bin histogram
        hist = [0] * 24
        for (h,) in rows:
            if h is not None and 0 <= h < 24:
                hist[h] += 1

        total = sum(hist)
        if total == 0:
            return 0.0

        # Entropy calculation
        entropy = 0.0
        for count in hist:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # Normalize: max_entropy = log2(24) ≈ 4.585
        max_entropy = math.log2(24)
        regularity = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        return max(0.0, min(1.0, regularity))


@dataclass
class RoutingContext:
    """Context for net-value computation."""
    in_complex_task: bool = False
    in_quiet_hours: bool = False
    today_touch_count: int = 0
    daily_limit: int = 5


def compute_fatigue(today_count: int, daily_limit: int, consecutive_ignored: int) -> float:
    """Fatigue score in [0.0, 1.0]. Higher = more fatigued."""
    base = today_count / max(daily_limit, 1)
    ignore_penalty = consecutive_ignored * 0.1
    return min(1.0, base + ignore_penalty)


def is_quiet_hours(quiet_hours: str) -> bool:
    """Check if current local time falls within quiet hours.

    Format: "HH:MM-HH:MM" (e.g., "23:00-07:00").
    Handles wrap-around (23:00-07:00 = 11pm to 7am).
    Returns False for empty/invalid input.
    """
    if not quiet_hours or not quiet_hours.strip():
        return False
    try:
        parts = quiet_hours.strip().split("-")
        if len(parts) != 2:
            return False
        start_h, start_m = map(int, parts[0].strip().split(":"))
        end_h, end_m = map(int, parts[1].strip().split(":"))
        if not (0 <= start_h <= 23 and 0 <= start_m <= 59):
            return False
        if not (0 <= end_h <= 23 and 0 <= end_m <= 59):
            return False
    except (ValueError, IndexError):
        return False

    now = datetime.now().astimezone()
    cur_minutes = now.hour * 60 + now.minute
    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m

    if start_minutes <= end_minutes:
        # No wrap: e.g., "09:00-17:00"
        return start_minutes <= cur_minutes < end_minutes
    else:
        # Wrap-around: e.g., "23:00-07:00"
        return cur_minutes >= start_minutes or cur_minutes < end_minutes


class ProactiveActionRouter:
    """Net-value decision routing with hysteresis for auto-create boundary.

    Decision cascade:
    1. suppressed -> skip
    2. not authorized -> suggest_only
    3. complex_task or quiet_hours -> defer
    4. read_only & conf >= 0.8 & net >= T_auto -> auto_create
    5. conf >= 0.5 & net >= T_suggest -> suggest
    6. net > 0 -> defer
    7. else -> skip
    """

    ENTER_AUTO = 0.8
    EXIT_AUTO = 0.7

    def __init__(
        self,
        db_path: str,
        *,
        authorization: dict[str, str] | None = None,
        t_auto: float = 0.1,
        t_suggest: float = 0.0,
    ):
        self.db_path = db_path
        self.authorization = authorization or {}
        self.t_auto = t_auto
        self.t_suggest = t_suggest
        self._domain_feedback = DomainFeedback(db_path)
        self._auto_active: set[str] = set()  # pattern_keys in auto-create zone

    def route(self, candidate: dict[str, Any], ctx: RoutingContext) -> str:
        pattern_key = candidate["pattern_key"]
        domain = candidate["domain"]
        confidence = candidate["confidence"]
        is_read_only = candidate.get("is_read_only", False)

        # 1. Suppression check
        if self._is_suppressed(pattern_key):
            return "skip"

        # 2. Authorization check
        auth = self.authorization.get(domain, "suggest")
        if auth != "auto":
            return "suggest_only"

        # 3. Context gates
        if ctx.in_complex_task or ctx.in_quiet_hours:
            return "defer"

        # Compute net value
        net = self._compute_net_value(candidate, ctx)

        # 4. Auto-create with hysteresis (read-only only)
        if is_read_only:
            if pattern_key in self._auto_active:
                # Already in auto zone -- exit only below EXIT_AUTO
                if confidence < self.EXIT_AUTO:
                    self._auto_active.discard(pattern_key)
                else:
                    if net >= self.t_auto:
                        return "auto_create"
            else:
                # Not yet in auto zone -- enter at ENTER_AUTO
                if confidence >= self.ENTER_AUTO and net >= self.t_auto:
                    self._auto_active.add(pattern_key)
                    return "auto_create"

        # 5. Suggest
        if confidence >= 0.5 and net >= self.t_suggest:
            return "suggest"

        # 6. Low positive net -> defer
        if net > 0:
            return "defer"

        return "skip"

    def _compute_net_value(self, candidate: dict, ctx: RoutingContext) -> float:
        domain = candidate["domain"]
        sug_type = candidate.get("suggestion_type", "watch")
        confidence = candidate["confidence"]
        is_read_only = candidate.get("is_read_only", False)

        accept_rate = self._domain_feedback.accept_rate(domain, sug_type)
        benefit = confidence * accept_rate

        interruption = 0.3 if ctx.in_complex_task else 0.0
        fatigue = compute_fatigue(
            ctx.today_touch_count, ctx.daily_limit,
            self._domain_feedback.consecutive_ignored(domain, sug_type),
        )
        risk = 0.0 if is_read_only else 0.3

        cost = interruption + fatigue + risk
        return benefit - cost

    def _is_suppressed(self, pattern_key: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT suppressed_until FROM behavior_suppressions "
                "WHERE pattern_key = ?",
                (pattern_key,),
            ).fetchone()
        if not row:
            return False
        return row[0] > now

    def add_suppression(
        self, pattern_key: str, domain: str, days: int = 30, reason: str = ""
    ) -> None:
        until = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO behavior_suppressions "
                "(pattern_key, domain, suppressed_until, reason) "
                "VALUES (?, ?, ?, ?)",
                (pattern_key, domain, until, reason),
            )

    def clean_expired_suppressions(self) -> int:
        """Remove expired suppressions. Returns count deleted."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM behavior_suppressions WHERE suppressed_until < ?",
                (now,),
            )
            return cursor.rowcount


class PendingSuggestionStore:
    """CRUD for pending_suggestions table."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def add(
        self,
        *,
        pattern_key: str,
        domain: str,
        suggestion_type: str,
        message: str,
        action_json: str,
        confidence: float,
        net_value: float,
        delivery_mode: str = "session",
        target_session_id: str | None = None,
    ) -> bool:
        """Insert a new pending suggestion. Returns False if duplicate active pattern."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO pending_suggestions "
                    "(pattern_key, domain, suggestion_type, message, action_json, "
                    "confidence, net_value, delivery_mode, target_session_id, "
                    "status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                    (pattern_key, domain, suggestion_type, message, action_json,
                     confidence, net_value, delivery_mode, target_session_id, now, now),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def get_pending(
        self, *, max_items: int = 2, session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get top pending session suggestions eligible for chat-time delivery.

        Only returns delivery_mode='session' — auto suggestions are consumed
        exclusively by the bridge loop via get_by_delivery_mode('auto').
        """
        now = datetime.now(timezone.utc).isoformat()
        session_filter = str(session_id or "").strip()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if session_filter:
                rows = conn.execute(
                    "SELECT * FROM pending_suggestions "
                    "WHERE status = 'pending' "
                    "  AND delivery_mode = 'session' "
                    "  AND (target_session_id IS NULL OR target_session_id = '' "
                    "       OR target_session_id = ?) "
                    "  AND (next_eligible_at IS NULL OR next_eligible_at <= ?) "
                    "ORDER BY net_value DESC, created_at ASC "
                    "LIMIT ?",
                    (session_filter, now, max_items),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM pending_suggestions "
                    "WHERE status = 'pending' "
                    "  AND delivery_mode = 'session' "
                    "  AND (target_session_id IS NULL OR target_session_id = '') "
                    "  AND (next_eligible_at IS NULL OR next_eligible_at <= ?) "
                    "ORDER BY net_value DESC, created_at ASC "
                    "LIMIT ?",
                    (now, max_items),
                ).fetchall()
        return [dict(r) for r in rows]

    def mark_shown(self, suggestion_id: int, *, cooldown_sec: float | int | None = None) -> None:
        """Atomic increment of sessions_shown, optionally delaying the next delivery."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            if cooldown_sec is not None and float(cooldown_sec) > 0:
                next_eligible_at = (
                    datetime.now(timezone.utc) + timedelta(seconds=float(cooldown_sec))
                ).isoformat()
                conn.execute(
                    "UPDATE pending_suggestions "
                    "SET sessions_shown = sessions_shown + 1, "
                    "    last_shown_at = ?, next_eligible_at = ?, updated_at = ? "
                    "WHERE id = ?",
                    (now, next_eligible_at, now, suggestion_id),
                )
            else:
                conn.execute(
                    "UPDATE pending_suggestions "
                    "SET sessions_shown = sessions_shown + 1, "
                    "    last_shown_at = ?, updated_at = ? "
                    "WHERE id = ?",
                    (now, now, suggestion_id),
                )

    def get_suggestion(self, suggestion_id: int) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM pending_suggestions WHERE id = ?",
                (suggestion_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_status(self, suggestion_id: int, status: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE pending_suggestions SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, suggestion_id),
            )

    def expire_stale(self, *, max_shown: int = 3) -> list[dict]:
        """Mark suggestions shown >= max_shown times as expired.

        Returns list of expired suggestion dicts (domain, suggestion_type)
        so callers can feed the fatigue model via record_ignored().
        """
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            about_to_expire = conn.execute(
                "SELECT domain, suggestion_type FROM pending_suggestions "
                "WHERE status = 'pending' AND sessions_shown >= ?",
                (max_shown,),
            ).fetchall()
            conn.execute(
                "UPDATE pending_suggestions "
                "SET status = 'expired', updated_at = ? "
                "WHERE status = 'pending' AND sessions_shown >= ?",
                (now, max_shown),
            )
            return [dict(r) for r in about_to_expire]

    def get_by_delivery_mode(
        self, mode: str, *, status: str = "pending", max_items: int = 10,
    ) -> list[dict[str, Any]]:
        """Get suggestions filtered by delivery_mode and status."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM pending_suggestions "
                "WHERE delivery_mode = ? AND status = ? "
                "  AND (next_eligible_at IS NULL OR next_eligible_at <= ?) "
                "ORDER BY net_value DESC, created_at ASC "
                "LIMIT ?",
                (mode, status, now, max_items),
            ).fetchall()
        return [dict(r) for r in rows]

    def try_claim(self, suggestion_id: int) -> bool:
        """Atomically claim a pending suggestion for processing."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE pending_suggestions SET status = 'processing' "
                "WHERE id = ? AND status = 'pending'",
                (suggestion_id,),
            )
            return cursor.rowcount > 0

    def add_simple(
        self, *, message: str, delivery_mode: str = "session",
        domain: str = "general", confidence: float = 0.5,
        action_json: str | dict[str, Any] | None = None,
        pattern_key: str | None = None,
        target_session_id: str | None = None,
    ) -> bool:
        """Convenience method for goal_loop / bridge result delivery."""
        if action_json is None:
            action = json.dumps({"type": "display"}, ensure_ascii=False)
        elif isinstance(action_json, str):
            action = action_json
        else:
            action = json.dumps(action_json, ensure_ascii=False)
        key = pattern_key or (
            f"system:{delivery_mode}:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        )
        return self.add(
            pattern_key=key,
            domain=domain,
            suggestion_type="goal_result" if delivery_mode == "session" else "auto_task",
            message=message, action_json=action,
            confidence=confidence, net_value=confidence * 0.8,
            delivery_mode=delivery_mode,
            target_session_id=target_session_id,
        )

    def transition_status(self, suggestion_id: int, from_status: str, to_status: str) -> bool:
        """Atomically transition status if the current state matches."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE pending_suggestions SET status = ?, updated_at = ? "
                "WHERE id = ? AND status = ?",
                (to_status, now, suggestion_id, from_status),
            )
            return cursor.rowcount > 0
