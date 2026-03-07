"""Interest monitoring — /watch system for proactive data sensing.

Architecture:
  User query → LLM factor decomposition → deterministic resolver → SQLite storage

The resolver classifies each factor as:
  EXECUTABLE — direct tool mapping (structured data, reliable)
  PROXY      — approximated via web_search (semi-structured, lower reliability)
  BLIND      — no data source available (excluded from monitoring)
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

from ..config import db_path as _db_path
from ..logging import get_logger
from ..tools import get_all_tools

_log = get_logger("interest")

DB_PATH = _db_path()


# ── Resolution classification ───────────────────────────────────────────────

class Resolution(str, Enum):
    EXECUTABLE = "executable"
    PROXY = "proxy"
    BLIND = "blind"


# ── Factor Resolver ─────────────────────────────────────────────────────────
#
# Deterministic mapping: source_hint → (tool_name, resolution).
# LLM outputs ideal factors unconstrained; resolver compiles against
# the actually registered tool registry.

# source_hint → tool_name for structured, reliable data sources
_EXECUTABLE_MAP: dict[str, str] = {
    "stock_price": "stock",
    "stock_quote": "stock",
    "stock_volume": "stock",
    "stock_metrics": "stock",
    "company_profile": "stock",
    "market_cap": "stock",
}

# source_hints where web_search cannot provide structured monitoring data
# (only truly unmonitorable real-time feeds remain here;
#  publicly web-searchable hints have moved to PROXY)
_BLIND_HINTS: set[str] = {
    "options_flow",
    "short_interest",
}

_RELIABILITY = {
    Resolution.EXECUTABLE: 1.0,
    Resolution.PROXY: 0.7,
    Resolution.BLIND: 0.0,
}


@dataclass
class ResolvedFactor:
    name: str
    source_hint: str
    entity: str
    frequency: str
    resolution: Resolution
    bound_tool: str | None
    reliability: float
    weight: float


def resolve_factor(
    source_hint: str, *, available_tools: set[str] | None = None,
) -> tuple[Resolution, str | None, float]:
    """Classify a single source_hint into EXECUTABLE/PROXY/BLIND.

    Returns (resolution, bound_tool_name, reliability).
    """
    if available_tools is None:
        available_tools = set(get_all_tools().keys())

    # Direct tool mapping
    tool_name = _EXECUTABLE_MAP.get(source_hint)
    if tool_name and tool_name in available_tools:
        return Resolution.EXECUTABLE, tool_name, _RELIABILITY[Resolution.EXECUTABLE]

    # Known blind spots — structured data with no available API
    if source_hint in _BLIND_HINTS:
        return Resolution.BLIND, None, _RELIABILITY[Resolution.BLIND]

    # Fallback: PROXY via web_search
    if "web_search" in available_tools:
        return Resolution.PROXY, "web_search", _RELIABILITY[Resolution.PROXY]

    return Resolution.BLIND, None, _RELIABILITY[Resolution.BLIND]


def resolve_factors(
    raw_factors: list[dict], *, available_tools: set[str] | None = None,
) -> list[ResolvedFactor]:
    """Resolve a list of LLM-generated factors against the tool registry."""
    available = available_tools if available_tools is not None else set(get_all_tools().keys())
    results = []
    for f in raw_factors:
        source_hint = f.get("source_hint", "")
        resolution, tool_name, reliability = resolve_factor(
            source_hint, available_tools=available,
        )
        results.append(ResolvedFactor(
            name=f.get("name", ""),
            source_hint=source_hint,
            entity=f.get("entity", ""),
            frequency=f.get("frequency", ""),
            resolution=resolution,
            bound_tool=tool_name,
            reliability=reliability,
            weight=f.get("weight", 1.0),
        ))
    return results


# ── Interest Store (SQLite) ─────────────────────────────────────────────────

class InterestStore:
    """SQLite-backed storage for monitoring interests and their factor graphs."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS interests (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                intent TEXT NOT NULL DEFAULT '',
                context_json TEXT NOT NULL DEFAULT '{}',
                discord_thread_id TEXT,
                coverage_ratio REAL NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS interest_factors (
                id TEXT PRIMARY KEY,
                interest_id TEXT NOT NULL REFERENCES interests(id),
                name TEXT NOT NULL,
                source_hint TEXT NOT NULL DEFAULT '',
                entity TEXT NOT NULL DEFAULT '',
                frequency TEXT NOT NULL DEFAULT '',
                resolution TEXT NOT NULL DEFAULT 'blind',
                bound_tool TEXT,
                reliability REAL NOT NULL DEFAULT 0.0,
                weight REAL NOT NULL DEFAULT 1.0,
                poll_interval INTEGER NOT NULL DEFAULT 300,
                last_value TEXT,
                last_checked TEXT
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS interest_factor_edges (
                from_factor_id TEXT NOT NULL,
                to_factor_id TEXT NOT NULL,
                relation TEXT NOT NULL DEFAULT '',
                correlation REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (from_factor_id, to_factor_id)
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS blind_backlog (
                source_hint TEXT PRIMARY KEY,
                requested_count INTEGER NOT NULL DEFAULT 1,
                first_requested TEXT NOT NULL,
                resolved_tool TEXT
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS signal_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                factor_id TEXT NOT NULL,
                interest_id TEXT NOT NULL,
                value_json TEXT NOT NULL DEFAULT '{}',
                delta_json TEXT,
                created_at TEXT NOT NULL
            )""")
            conn.execute("""CREATE INDEX IF NOT EXISTS idx_factors_interest
                ON interest_factors (interest_id)""")
            conn.execute("""CREATE INDEX IF NOT EXISTS idx_signal_factor_time
                ON signal_log (factor_id, created_at DESC)""")

    # ── CRUD ────────────────────────────────────────────────────────────────

    def create_interest(
        self,
        *,
        query: str,
        intent: str = "",
        context: dict | None = None,
        discord_thread_id: str | None = None,
        factors: list[ResolvedFactor] | None = None,
        edges: list[dict] | None = None,
    ) -> dict:
        interest_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        context_json = json.dumps(context or {}, ensure_ascii=False)

        factors = factors or []
        edges = edges or []

        # Coverage ratio
        total = len(factors)
        active = sum(1 for f in factors if f.resolution != Resolution.BLIND)
        coverage = active / total if total > 0 else 0.0

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO interests
                   (id, query, intent, context_json, discord_thread_id,
                    coverage_ratio, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'active', ?)""",
                (interest_id, query, intent, context_json,
                 discord_thread_id, coverage, now),
            )

            # Insert factors
            factor_ids: dict[str, str] = {}
            for f in factors:
                fid = uuid.uuid4().hex[:12]
                factor_ids[f.name] = fid
                interval = _frequency_to_seconds(f.frequency)
                conn.execute(
                    """INSERT INTO interest_factors
                       (id, interest_id, name, source_hint, entity, frequency,
                        resolution, bound_tool, reliability, weight, poll_interval)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (fid, interest_id, f.name, f.source_hint, f.entity,
                     f.frequency, f.resolution.value, f.bound_tool,
                     f.reliability, f.weight, interval),
                )

            # Insert edges
            for edge in edges:
                from_id = factor_ids.get(edge.get("from", ""))
                to_id = factor_ids.get(edge.get("to", ""))
                if from_id and to_id:
                    conn.execute(
                        """INSERT OR IGNORE INTO interest_factor_edges
                           (from_factor_id, to_factor_id, relation, correlation)
                           VALUES (?, ?, ?, ?)""",
                        (from_id, to_id, edge.get("relation", ""), 0.0),
                    )

            # Track blind spots in backlog
            for f in factors:
                if f.resolution == Resolution.BLIND:
                    conn.execute(
                        """INSERT INTO blind_backlog
                             (source_hint, requested_count, first_requested)
                           VALUES (?, 1, ?)
                           ON CONFLICT(source_hint) DO UPDATE SET
                             requested_count = requested_count + 1""",
                        (f.source_hint, now),
                    )

        return self.get_interest(interest_id)

    def get_interest(self, interest_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM interests WHERE id = ?", (interest_id,),
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            try:
                result["context"] = json.loads(result.pop("context_json", "{}"))
            except (json.JSONDecodeError, TypeError):
                result["context"] = {}

            # Attach factors
            factors = conn.execute(
                """SELECT * FROM interest_factors
                   WHERE interest_id = ? ORDER BY weight DESC""",
                (interest_id,),
            ).fetchall()
            result["factors"] = [dict(f) for f in factors]

            # Attach edges
            factor_ids = [f["id"] for f in result["factors"]]
            if factor_ids:
                ph = ",".join("?" * len(factor_ids))
                edges = conn.execute(
                    f"""SELECT * FROM interest_factor_edges
                        WHERE from_factor_id IN ({ph})""",
                    factor_ids,
                ).fetchall()
                result["edges"] = [dict(e) for e in edges]
            else:
                result["edges"] = []

        return result

    def list_interests(self, *, include_archived: bool = False) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if include_archived:
                rows = conn.execute(
                    "SELECT * FROM interests ORDER BY created_at DESC",
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM interests
                       WHERE status != 'archived'
                       ORDER BY created_at DESC""",
                ).fetchall()
        return [dict(r) for r in rows]

    def update_interest(self, interest_id: str, **fields) -> dict | None:
        allowed = {"intent", "status", "discord_thread_id", "coverage_ratio"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_interest(interest_id)
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [interest_id]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE interests SET {set_clause} WHERE id = ?", values,
            )
        return self.get_interest(interest_id)

    def pause_interest(self, interest_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE interests SET status = 'paused' "
                "WHERE id = ? AND status = 'active'",
                (interest_id,),
            )
        return cur.rowcount > 0

    def resume_interest(self, interest_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE interests SET status = 'active' "
                "WHERE id = ? AND status = 'paused'",
                (interest_id,),
            )
        return cur.rowcount > 0

    def archive_interest(self, interest_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE interests SET status = 'archived' "
                "WHERE id = ? AND status != 'archived'",
                (interest_id,),
            )
        return cur.rowcount > 0

    def get_blind_backlog(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM blind_backlog
                   WHERE resolved_tool IS NULL
                   ORDER BY requested_count DESC""",
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Signal tracking ─────────────────────────────────────────────────────

    def update_factor_value(
        self, factor_id: str, value_json: str, checked_at: str,
    ) -> None:
        """Update a factor's last_value and last_checked."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE interest_factors SET last_value = ?, last_checked = ? "
                "WHERE id = ?",
                (value_json, checked_at, factor_id),
            )

    def prune_signal_log(self, days: int = 14):
        """Delete signal_log entries older than *days* to bound table growth."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM signal_log WHERE created_at < ?", (cutoff,))
        except Exception:
            pass

    def record_signal(
        self,
        *,
        factor_id: str,
        interest_id: str,
        value_json: str,
        delta_json: str | None = None,
    ) -> int:
        """Record a signal event in the log. Returns the row id."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO signal_log
                   (factor_id, interest_id, value_json, delta_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (factor_id, interest_id, value_json, delta_json, now),
            )
            return cur.lastrowid

    def get_recent_signals(
        self, interest_id: str, *, limit: int = 20,
    ) -> list[dict]:
        """Get recent signal events for an interest."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT sl.*, if2.name as factor_name
                   FROM signal_log sl
                   JOIN interest_factors if2 ON sl.factor_id = if2.id
                   WHERE sl.interest_id = ?
                   ORDER BY sl.created_at DESC LIMIT ?""",
                (interest_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_pollable_factors(self) -> list[dict]:
        """Get all EXECUTABLE/PROXY factors from active interests, with interest metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT f.*, i.intent, i.discord_thread_id, i.query
                   FROM interest_factors f
                   JOIN interests i ON f.interest_id = i.id
                   WHERE i.status = 'active'
                     AND f.resolution != 'blind'
                   ORDER BY f.poll_interval ASC""",
            ).fetchall()
        return [dict(r) for r in rows]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _frequency_to_seconds(freq: str) -> int:
    """Convert frequency string to poll interval in seconds."""
    mapping = {
        "realtime": 30,
        "1min": 60,
        "5min": 300,
        "15min": 900,
        "hourly": 3600,
        "daily": 86400,
        "weekly": 604800,
        "monthly": 2592000,
        "quarterly": 7776000,
        "event": 3600,
    }
    return mapping.get((freq or "").lower().strip(), 300)


# ── LLM Factor Generation ──────────────────────────────────────────────────

_FACTOR_SYSTEM_PROMPT = """\
You are a factor analyst for a personal monitoring system.
Given the user's monitoring request, decompose it into observable factors
that can be periodically polled from data sources.

Rules:
1. Generate 5-12 factors covering different risk/opportunity dimensions.
2. Each factor needs a source_hint describing the DATA TYPE (not the tool name).
3. Assign weights 0.1-1.0 based on relevance to the user's intent.
4. Define edges (causal relationships) between connected factors.
5. All factors are monitorable. Specialized APIs provide structured data
   for stock/financial hints; all others are monitored via web search
   with LLM enrichment. Use any hint that best describes the data need.

Valid source_hint values:
  stock_price, stock_quote, stock_volume, stock_metrics,
  company_profile, market_cap,
  news_search, company_news, industry_news, regulatory_news,
  financial_earnings, earnings_calendar,
  analyst_rating, credit_rating,
  insider_trading, institutional_holdings, short_interest, options_flow,
  macro_indicator, interest_rate, currency_rate,
  tech_trend, github_activity, product_release,
  social_sentiment, patent_filing, regulatory_action,
  web_content, rss_feed, sec_filing

Output valid JSON only. No markdown fences, no commentary.
{
  "intent": "one-line summary of what user cares about",
  "context": {"key": "value pairs extracted from user input"},
  "factors": [
    {
      "name": "human-readable factor name",
      "source_hint": "from the list above",
      "entity": "specific identifier (ticker, company, URL, etc.)",
      "frequency": "realtime|5min|15min|hourly|daily|weekly|quarterly|event",
      "weight": 0.8
    }
  ],
  "edges": [
    {"from": "factor name", "to": "factor name", "relation": "causal description"}
  ]
}"""


async def create_interest_from_query(
    query: str,
    engine,
    store: InterestStore,
) -> dict:
    """Full pipeline: LLM generates ideal factors → resolver classifies → store persists.

    Returns the complete interest dict with factors and coverage.
    """
    # Step 1: LLM generates ideal factors (unconstrained by tools)
    messages = [
        {"role": "system", "content": _FACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    raw = await engine.generate_reasoning(
        messages, max_tokens=2048, temperature=0.3, enable_thinking=False,
    )

    # Parse JSON (strip markdown fences if LLM adds them)
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, IndexError) as e:
        _log.error("interest", e, action="parse_llm_factors")
        raise ValueError(f"Failed to parse LLM factor output: {e}") from e

    # Step 2: Deterministic resolver — classify against tool registry
    resolved = resolve_factors(data.get("factors", []))

    # Step 3: Persist to SQLite
    interest = store.create_interest(
        query=query,
        intent=data.get("intent", ""),
        context=data.get("context", {}),
        factors=resolved,
        edges=data.get("edges", []),
    )

    return interest


def build_coverage_summary(interest: dict) -> dict:
    """Build a structured coverage summary for Discord embed display."""
    factors = interest.get("factors", [])

    executable = [f for f in factors if f["resolution"] == "executable"]
    proxy = [f for f in factors if f["resolution"] == "proxy"]
    blind = [f for f in factors if f["resolution"] == "blind"]

    total = len(factors)
    active = len(executable) + len(proxy)
    coverage = active / total if total > 0 else 0.0

    return {
        "interest_id": interest["id"],
        "intent": interest.get("intent", ""),
        "context": interest.get("context", {}),
        "executable": [
            {"name": f["name"], "entity": f.get("entity", "")}
            for f in executable
        ],
        "proxy": [
            {"name": f["name"], "entity": f.get("entity", "")}
            for f in proxy
        ],
        "blind": [
            {"name": f["name"], "source_hint": f.get("source_hint", "")}
            for f in blind
        ],
        "coverage_ratio": coverage,
        "total": total,
        "active": active,
    }
