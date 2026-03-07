"""Conversation memory (sliding window) + long-term memory (SQLite)."""

import hashlib
import json
import math
import os
import re
import sqlite3
import time as _time_mod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ..config import db_path as _db_path
from ..logging import get_logger

_log = get_logger("memory")

DB_PATH = _db_path()  # backward-compat module-level alias


def _estimate_chars(msg: dict) -> int:
    """Estimate character cost of a message (content + tool_calls)."""
    n = len(msg.get("content", "") or "")
    for tc in msg.get("tool_calls", []):
        n += len(str(tc))
    return n


# Default char budget ≈ 6000 tokens assuming ~2.5 chars/token average (mixed CJK/ASCII).
_DEFAULT_CHAR_BUDGET = int(os.environ.get("LIAGENT_CONTEXT_CHAR_BUDGET", "16000"))

_HALF_LIVES: dict[str, int | None] = {
    "tool_result": 60, "user_stated": 90, "user_edit": None,
    "llm_extract": 30, "llm_inferred": 20,
}
_RRF_K = 60
_RECALL_WINDOWS: dict[str, tuple[int, int]] = {
    "exact": (5, 20), "semantic": (20, 5), "mixed": (15, 15),
}


@dataclass
class EvidenceChunk:
    """Canonical grounding unit for retrieved memory evidence."""

    evidence_id: str
    source_type: str
    source_ref: str
    line_start: int | None
    line_end: int | None
    snippet: str
    score: float
    retrieved_at: str


def _rrf_score(rank_vec: int, rank_bm25: int, k: int = _RRF_K) -> float:
    """Reciprocal Rank Fusion: score = 1/(k+rank_vec) + 1/(k+rank_bm25)."""
    return 1.0 / (k + rank_vec) + 1.0 / (k + rank_bm25)


def _temporal_decay(score: float, age_days: float, source: str) -> float:
    """Apply source-type exponential decay. user_edit never decays."""
    half_life = _HALF_LIVES.get(source)
    if half_life is None:
        return score
    return score * (0.5 ** (age_days / half_life))


def _mmr_rerank(
    candidates: list[dict], query_vec: np.ndarray,
    lambda_param: float = 0.7, top_n: int = 8,
) -> list[dict]:
    """Maximal Marginal Relevance: balance relevance vs diversity."""
    if not candidates:
        return []

    selected: list[dict] = []
    remaining = list(candidates)

    for _ in range(min(top_n, len(candidates))):
        best_idx = -1
        best_mmr = -float("inf")

        for i, cand in enumerate(remaining):
            vec = cand.get("_vec")
            relevance = cand.get("_rrf_score", 0.0)

            if vec is not None and query_vec is not None:
                # Use cosine similarity as relevance component
                qn = np.linalg.norm(query_vec)
                cn = np.linalg.norm(vec)
                if qn > 0 and cn > 0:
                    relevance = float(np.dot(query_vec, vec) / (qn * cn))

            # Max similarity to already selected
            max_sim = 0.0
            for sel in selected:
                sv = sel.get("_vec")
                if sv is not None and vec is not None:
                    sn = np.linalg.norm(sv)
                    cn2 = np.linalg.norm(vec)
                    if sn > 0 and cn2 > 0:
                        sim = float(np.dot(sv, vec) / (sn * cn2))
                        max_sim = max(max_sim, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))

    return selected


class ConversationMemory:
    def __init__(self, max_turns: int = 30, *, char_budget: int = _DEFAULT_CHAR_BUDGET):
        self.max_turns = max_turns
        self.char_budget = max(4000, char_budget)
        self.messages: list[dict] = []
        self.compressed_context: str = ""
        self._last_compress_turn: int = 0  # cooldown: skip if compressed recently

    def add(self, role: str, content: str, *, tool_calls: list[dict] | None = None):
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)
        self._trim()

    def _trim(self):
        """Drop oldest messages to stay within both message count and char budget."""
        # Hard cap on message count (backward compat)
        max_msgs = self.max_turns * 3
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]
        if not self.messages:
            return
        # Char-budget trim: walk backwards, keep as many messages as budget allows.
        # Always keep at least the most recent message to prevent a single
        # oversized message (e.g. large tool result) from wiping the entire
        # conversation history.
        total = 0
        keep_from = 0
        for i in range(len(self.messages) - 1, -1, -1):
            cost = _estimate_chars(self.messages[i])
            if total + cost > self.char_budget:
                keep_from = i + 1
                break
            total += cost
        # Clamp: never drop ALL messages — keep at least the newest one
        max_keep_from = max(0, len(self.messages) - 1)
        keep_from = min(keep_from, max_keep_from)
        if keep_from > 0:
            self.messages = self.messages[keep_from:]

    def get_messages(self, max_messages: int = 0) -> list[dict]:
        msgs = []
        if self.compressed_context:
            msgs.append({"role": "system", "content": f"[History Summary]\n{self.compressed_context}"})
        tail = self.messages[-max_messages:] if max_messages > 0 else self.messages
        msgs.extend(tail.copy())
        return msgs

    async def compress(self, engine, *, cooldown: int = 6):
        """Compress older messages into a structured summary when context pressure is high.

        Triggers when total chars exceed 80% of char_budget, giving headroom
        before ``_trim()`` starts dropping messages.

        Args:
            engine: EngineManager with generate_extraction() method.
            cooldown: Minimum messages since last compression (avoids repeated LLM calls).
        """
        n = len(self.messages)
        if n < 4:  # minimum messages to have anything worth compressing
            return
        turns_since = n - self._last_compress_turn
        if turns_since < cooldown:
            return
        total_chars = sum(_estimate_chars(m) for m in self.messages)
        if total_chars < self.char_budget * 0.8:
            return  # not yet under pressure
        half = len(self.messages) // 2
        old_msgs = self.messages[:half]
        # Build summarization prompt
        conversation_text = "\n".join(
            f"[{m['role']}] {m['content'][:200]}" for m in old_msgs
        )
        rolling_prefix = self.compressed_context.strip()
        user_payload = conversation_text
        if rolling_prefix:
            user_payload = (
                f"[Previous Summary]\n{rolling_prefix}\n\n"
                f"[New Messages]\n{conversation_text}"
            )
        prompt = [
            {
                "role": "system",
                "content": (
                    "Summarize this conversation segment into a concise rolling context. Format:\n"
                    "FACTS: [key facts, user preferences, constraints]\n"
                    "GOALS: [active goals, pending tasks]\n"
                    "DECISIONS: [decisions made, approaches chosen]\n"
                    "Max 150 words total. Merge with any previous summary."
                ),
            },
            {"role": "user", "content": user_payload},
        ]
        try:
            summary = await engine.generate_extraction(prompt, max_tokens=200, temperature=0.2)
            summary = summary.strip()
            if summary:
                self.compressed_context = summary[:2000]
                self.messages = self.messages[half:]
                self._last_compress_turn = len(self.messages)
        except Exception as exc:
            _log.trace("compression_error", error=str(exc))

    def clear(self):
        self.messages.clear()
        self.compressed_context = ""

    def last_user_message(self) -> str:
        for m in reversed(self.messages):
            if m["role"] == "user":
                return m.get("content", "")
        return ""

    def turn_count(self) -> int:
        """Number of user messages (turns) in the current window."""
        return sum(1 for m in self.messages if m["role"] == "user")


class LongTermMemory:
    """SQLite-backed persistent memory for session summaries and key facts."""

    def __init__(self, db_path: Path = DB_PATH, data_dir: Path | None = None, journal=None):
        self.db_path = db_path
        self.data_dir = data_dir or Path.home() / ".liagent"
        self.facts_md_path = self.data_dir / "facts.md"
        self.journal = journal
        self._event_log = None
        try:
            from ..knowledge.event_log import EventLog
            self._event_log = EventLog()
        except Exception as exc:
            _log.trace("event_log_init_skipped", error=str(exc))
        self._init_db()
        self._embedder = None
        self._init_vector_tables()
        self.sync_facts_from_markdown()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS session_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    turn_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS key_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            cols = {
                row[1] for row in conn.execute("PRAGMA table_info(key_facts)").fetchall()
            }
            if "expires_at" not in cols:
                conn.execute("ALTER TABLE key_facts ADD COLUMN expires_at TEXT")
            if "confidence" not in cols:
                conn.execute("ALTER TABLE key_facts ADD COLUMN confidence REAL DEFAULT 0.7")
            if "source" not in cols:
                conn.execute("ALTER TABLE key_facts ADD COLUMN source TEXT DEFAULT 'llm_extract'")
            if "fact_key" not in cols:
                conn.execute("ALTER TABLE key_facts ADD COLUMN fact_key TEXT DEFAULT ''")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS pending_goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    goal_text TEXT NOT NULL,
                    plan_json TEXT NOT NULL DEFAULT '{}',
                    completed_steps INTEGER NOT NULL DEFAULT 0,
                    total_steps INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            # Add evidence_json column for crash-recoverable step evidence
            goal_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(pending_goals)").fetchall()
            }
            if "evidence_json" not in goal_cols:
                conn.execute("ALTER TABLE pending_goals ADD COLUMN evidence_json TEXT DEFAULT '[]'")
            if "reasoning_summary_json" not in goal_cols:
                conn.execute(
                    "ALTER TABLE pending_goals ADD COLUMN reasoning_summary_json TEXT DEFAULT '{}'"
                )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    tool_used TEXT,
                    feedback TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            # FTS5 index for key_facts (external content, jieba pre-tokenized)
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS key_facts_fts USING fts5(fact_text)"
            )
            # One-time FTS population for existing data
            fts_count = conn.execute(
                "SELECT COUNT(*) FROM key_facts_fts"
            ).fetchone()[0]
            fact_count = conn.execute(
                "SELECT COUNT(*) FROM key_facts"
            ).fetchone()[0]
            if fts_count == 0 and fact_count > 0:
                self._rebuild_fts(conn)
            conn.execute("""CREATE TABLE IF NOT EXISTS user_profile_slots (
                dimension              TEXT PRIMARY KEY,
                value                  TEXT NOT NULL,
                confidence             REAL NOT NULL DEFAULT 0.5,
                source                 TEXT NOT NULL DEFAULT 'implicit',
                locked                 INTEGER NOT NULL DEFAULT 0,
                evidence_count         INTEGER NOT NULL DEFAULT 0,
                candidate_value        TEXT NOT NULL DEFAULT '',
                candidate_evidence_count INTEGER NOT NULL DEFAULT 0,
                updated_at             TEXT NOT NULL
            )""")

            # ── Proactive Intelligence tables ──────────────────────
            conn.execute("""CREATE TABLE IF NOT EXISTS behavior_signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_type     TEXT NOT NULL,
                key             TEXT NOT NULL,
                domain          TEXT DEFAULT '',
                source_origin   TEXT DEFAULT 'user',
                metadata        TEXT DEFAULT '{}' CHECK(json_valid(metadata)),
                session_id      TEXT,
                hour            INTEGER,
                weekday         INTEGER,
                created_at      TEXT NOT NULL
            )""")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bs_key_created "
                "ON behavior_signals(key, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bs_created "
                "ON behavior_signals(created_at)"
            )

            conn.execute("""CREATE TABLE IF NOT EXISTS pending_suggestions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_key     TEXT NOT NULL,
                domain          TEXT NOT NULL,
                suggestion_type TEXT NOT NULL,
                message         TEXT NOT NULL,
                action_json     TEXT NOT NULL CHECK(json_valid(action_json)),
                confidence      REAL NOT NULL,
                net_value       REAL NOT NULL,
                delivery_mode   TEXT DEFAULT 'session',
                sessions_shown  INTEGER DEFAULT 0,
                next_eligible_at TEXT,
                last_shown_at   TEXT,
                status          TEXT DEFAULT 'pending',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            )""")
            suggestion_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(pending_suggestions)").fetchall()
            }
            if "target_session_id" not in suggestion_cols:
                conn.execute("ALTER TABLE pending_suggestions ADD COLUMN target_session_id TEXT")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_pending_active_pattern "
                "ON pending_suggestions(pattern_key) "
                "WHERE status IN ('pending', 'deferred')"
            )

            conn.execute("""CREATE TABLE IF NOT EXISTS behavior_suppressions (
                pattern_key      TEXT PRIMARY KEY,
                domain           TEXT NOT NULL,
                suppressed_until TEXT NOT NULL,
                reason           TEXT DEFAULT ''
            )""")

            conn.execute("""CREATE TABLE IF NOT EXISTS domain_feedback (
                domain             TEXT NOT NULL,
                suggestion_type    TEXT NOT NULL,
                total_suggested    INTEGER DEFAULT 0,
                total_accepted     INTEGER DEFAULT 0,
                total_rejected     INTEGER DEFAULT 0,
                total_ignored      INTEGER DEFAULT 0,
                consecutive_ignored INTEGER DEFAULT 0,
                updated_at         TEXT NOT NULL,
                PRIMARY KEY (domain, suggestion_type)
            )""")

    def _init_vector_tables(self):
        """Create fact_embeddings and retrieval_log tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("""CREATE TABLE IF NOT EXISTS fact_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                model_name TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (fact_id) REFERENCES key_facts(id) ON DELETE CASCADE
            )""")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fact_embeddings_fact_id "
                "ON fact_embeddings(fact_id)"
            )
            conn.execute("""CREATE TABLE IF NOT EXISTS retrieval_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                query_type TEXT NOT NULL,
                fts_hits INTEGER NOT NULL DEFAULT 0,
                vec_hits INTEGER NOT NULL DEFAULT 0,
                fused_count INTEGER NOT NULL DEFAULT 0,
                final_count INTEGER NOT NULL DEFAULT 0,
                latency_ms REAL NOT NULL DEFAULT 0.0,
                embedder_provider TEXT NOT NULL DEFAULT 'none',
                created_at TEXT NOT NULL
            )""")

    def set_embedder(self, embedder):
        """Set the embedding provider (EmbedderChain instance)."""
        self._embedder = embedder

    def _embed_facts(self, conn, facts_with_ids: list[tuple[int, str]]):
        """Generate and store embeddings for facts. Skip if no embedder."""
        if not self._embedder or not facts_with_ids:
            return
        texts = [text for _, text in facts_with_ids]
        vecs = self._embedder.encode(texts)
        if vecs is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        model_name = self._embedder.model_name
        dims = self._embedder.dimensions
        for (fact_id, _), vec in zip(facts_with_ids, vecs):
            blob = vec.astype(np.float32).tobytes()
            # Remove old embedding for this fact if exists
            conn.execute("DELETE FROM fact_embeddings WHERE fact_id = ?", (fact_id,))
            conn.execute(
                "INSERT INTO fact_embeddings (fact_id, embedding, model_name, dimensions, created_at) VALUES (?, ?, ?, ?, ?)",
                (fact_id, blob, model_name, dims, now),
            )

    def prune_old_records(self, days: int = 30, goal_days: int = 7):
        """Delete old rows from diagnostic/session tables to bound growth.

        Args:
            days: Retention period for retrieval_log, session_summaries, user_feedback.
            goal_days: Retention period for resolved pending_goals.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        goal_cutoff = (datetime.now(timezone.utc) - timedelta(days=goal_days)).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM retrieval_log WHERE created_at < ?", (cutoff,))
                conn.execute("DELETE FROM session_summaries WHERE created_at < ?", (cutoff,))
                conn.execute("DELETE FROM user_feedback WHERE created_at < ?", (cutoff,))
                conn.execute(
                    "DELETE FROM pending_goals WHERE status != 'active' AND updated_at < ?",
                    (goal_cutoff,),
                )
        except Exception as exc:
            _log.trace("prune_old_records_error", error=str(exc))

    def close(self):
        """Close the event log connection if open."""
        if self._event_log:
            try:
                self._event_log.close()
            except Exception as exc:
                _log.trace("event_log_close_error", error=str(exc))
            self._event_log = None

    def save_summary(self, session_id: str, summary: str, turn_count: int):
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO session_summaries (session_id, summary, turn_count, created_at) VALUES (?, ?, ?, ?)",
                (session_id, summary, turn_count, now),
            )

    def save_facts(self, facts: list[dict], ttl_days: int = 90):
        """Save or update key facts. Each dict has 'fact' and 'category' keys."""
        now = datetime.now(timezone.utc).isoformat()
        if ttl_days > 0:
            expires_at = (
                datetime.now(timezone.utc) + timedelta(days=ttl_days)
            ).replace(microsecond=0).isoformat()
        else:
            expires_at = None

        with sqlite3.connect(self.db_path) as conn:
            for f in facts:
                fact = f.get("fact", "").strip()
                category = f.get("category", "")
                confidence_raw = f.get("confidence", 0.7)
                source = str(f.get("source", "llm_extract") or "llm_extract")
                if not fact:
                    continue
                try:
                    confidence = float(confidence_raw)
                except (TypeError, ValueError):
                    confidence = 0.7
                confidence = max(0.0, min(1.0, confidence))
                fact_key = self._fact_key(fact, category)
                is_new = conn.execute(
                    "SELECT 1 FROM key_facts WHERE fact = ?", (fact,)
                ).fetchone() is None
                conn.execute(
                    """INSERT INTO key_facts (fact, category, created_at, updated_at, expires_at, confidence, source, fact_key)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(fact) DO UPDATE SET
                           category = excluded.category,
                           updated_at = excluded.updated_at,
                           expires_at = excluded.expires_at,
                           confidence = excluded.confidence,
                           source = excluded.source,
                           fact_key = excluded.fact_key""",
                    (fact, category, now, now, expires_at, confidence, source, fact_key),
                )
                if self.journal:
                    self.journal.fact_learned(
                        fact=fact, category=category, confidence=confidence,
                        source=source, is_new=is_new,
                    )
                # Sync FTS index
                row = conn.execute(
                    "SELECT id FROM key_facts WHERE fact = ?", (fact,)
                ).fetchone()
                if row:
                    tokenized = _tokenize_for_fts(fact)
                    conn.execute(
                        "DELETE FROM key_facts_fts WHERE rowid = ?", (row[0],)
                    )
                    conn.execute(
                        "INSERT INTO key_facts_fts(rowid, fact_text) VALUES (?, ?)",
                        (row[0], tokenized),
                    )
            # Generate embeddings for saved facts
            if self._embedder:
                facts_with_ids = []
                for f in facts:
                    fact_text = f.get("fact", "").strip()
                    if not fact_text:
                        continue
                    row = conn.execute("SELECT id FROM key_facts WHERE fact = ?", (fact_text,)).fetchone()
                    if row:
                        facts_with_ids.append((row[0], fact_text))
                self._embed_facts(conn, facts_with_ids)
        self.sync_facts_to_markdown()
        # Emit change events to the event log for downstream consumers
        if self._event_log:
            from ..knowledge.event_log import ChangeEvent
            for f in facts:
                fact_text = f.get("fact", "").strip()
                if not fact_text:
                    continue
                try:
                    fkey = hashlib.sha256(fact_text.encode()).hexdigest()[:12]
                    self._event_log.append(ChangeEvent(
                        entity_type="fact",
                        entity_id=f"fact-{fkey}",
                        action="upsert",
                        payload={
                            "text": fact_text,
                            "category": f.get("category", "general"),
                            "confidence": float(f.get("confidence", 0.7)),
                        },
                    ))
                except Exception as exc:
                    _log.warning("fact_event_emit_error", error=str(exc))

    def upsert_tool_fact(
        self,
        fact: str,
        fact_key: str,
        category: str = "interest",
        confidence: float = 0.60,
        ttl_days: int = 90,
    ) -> None:
        """Upsert a zero-LLM fact extracted from tool execution.

        If fact_key already exists, bumps confidence by 0.05 (capped at 0.90)
        and updates timestamps.  Otherwise inserts a new row.
        """
        now = datetime.now(timezone.utc).isoformat()
        expires_at = (
            datetime.now(timezone.utc) + timedelta(days=ttl_days)
        ).replace(microsecond=0).isoformat() if ttl_days > 0 else None

        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT confidence FROM key_facts WHERE fact_key = ?",
                (fact_key,),
            ).fetchone()

            if existing:
                new_conf = min(0.90, existing[0] + 0.05)
                conn.execute(
                    "UPDATE key_facts SET confidence = ?, updated_at = ?, "
                    "expires_at = ? WHERE fact_key = ?",
                    (new_conf, now, expires_at, fact_key),
                )
            else:
                conn.execute(
                    "INSERT INTO key_facts "
                    "(fact, category, created_at, updated_at, expires_at, "
                    "confidence, source, fact_key) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'tool_extract', ?)",
                    (fact, category, now, now, expires_at, confidence, fact_key),
                )
                # Sync FTS index for new facts
                row = conn.execute(
                    "SELECT id FROM key_facts WHERE fact_key = ?", (fact_key,)
                ).fetchone()
                if row:
                    tokenized = _tokenize_for_fts(fact)
                    conn.execute(
                        "INSERT INTO key_facts_fts(rowid, fact_text) VALUES (?, ?)",
                        (row[0], tokenized),
                    )

    # ── Execution checkpoint methods (reuse pending_goals table) ──────

    def upsert_checkpoint(
        self,
        session_id: str,
        goal: str,
        plan_steps: list[dict],
        completed_steps: int,
        total_steps: int,
        evidence: list[dict],
        reasoning_summary: dict | None = None,
    ) -> int:
        """Create or update an execution checkpoint in pending_goals."""
        now = datetime.now(timezone.utc).isoformat()
        plan_json = json.dumps(plan_steps, ensure_ascii=False)
        evidence_json = json.dumps(evidence, ensure_ascii=False)
        reasoning_json = json.dumps(reasoning_summary or {}, ensure_ascii=False)
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT id FROM pending_goals WHERE session_id = ? AND status = 'active'",
                (session_id,),
            ).fetchone()
            if existing:
                conn.execute(
                    """UPDATE pending_goals SET
                        goal_text = ?, plan_json = ?, completed_steps = ?,
                        total_steps = ?, evidence_json = ?, reasoning_summary_json = ?,
                        updated_at = ?
                    WHERE id = ?""",
                    (goal, plan_json, completed_steps, total_steps, evidence_json,
                     reasoning_json, now, existing[0]),
                )
                return existing[0]
            else:
                conn.execute(
                    """INSERT INTO pending_goals
                        (session_id, goal_text, plan_json, completed_steps, total_steps,
                         status, created_at, updated_at, evidence_json, reasoning_summary_json)
                    VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)""",
                    (session_id, goal, plan_json, completed_steps, total_steps,
                     now, now, evidence_json, reasoning_json),
                )
                row = conn.execute("SELECT last_insert_rowid()").fetchone()
                return row[0] if row else 0

    def get_active_checkpoint(self, session_id: str = "") -> dict | None:
        """Get the most recent active checkpoint, optionally filtered by session_id."""
        with sqlite3.connect(self.db_path) as conn:
            if session_id:
                row = conn.execute(
                    """SELECT id, session_id, goal_text, plan_json, completed_steps,
                              total_steps, evidence_json, created_at, updated_at,
                              reasoning_summary_json
                    FROM pending_goals WHERE status = 'active' AND session_id = ?
                    ORDER BY updated_at DESC LIMIT 1""",
                    (session_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT id, session_id, goal_text, plan_json, completed_steps,
                              total_steps, evidence_json, created_at, updated_at,
                              reasoning_summary_json
                    FROM pending_goals WHERE status = 'active'
                    ORDER BY updated_at DESC LIMIT 1""",
                ).fetchone()
            if not row:
                return None
            return {
                "id": row[0], "session_id": row[1], "goal_text": row[2],
                "plan_json": row[3], "completed_steps": row[4],
                "total_steps": row[5], "evidence_json": row[6],
                "created_at": row[7], "updated_at": row[8],
                "reasoning_summary_json": row[9] if len(row) > 9 else None,
            }

    def complete_checkpoint(self, checkpoint_id: int) -> None:
        """Mark a checkpoint as completed."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE pending_goals SET status = 'completed', updated_at = ? WHERE id = ?",
                (now, checkpoint_id),
            )

    def abandon_checkpoint(self, checkpoint_id: int) -> None:
        """Mark a checkpoint as abandoned."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE pending_goals SET status = 'abandoned', updated_at = ? WHERE id = ?",
                (now, checkpoint_id),
            )

    def get_recent_summaries(self, limit: int = 5) -> list[str]:
        """Return recent session summaries, oldest first."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT summary FROM session_summaries ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r[0] for r in reversed(rows)]

    def get_all_facts(self, min_confidence: float = 0.65) -> list[dict]:
        """Return up to 20 most recently updated facts."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT fact, category, confidence, source, fact_key, updated_at
                   FROM key_facts
                   WHERE (expires_at IS NULL OR expires_at > ?)
                     AND COALESCE(confidence, 0.7) >= ?
                   ORDER BY updated_at DESC LIMIT 20""",
                (now, float(min_confidence)),
            ).fetchall()
        facts = [
            {
                "fact": r[0],
                "category": r[1],
                "confidence": float(r[2] or 0.7),
                "source": r[3] or "llm_extract",
                "fact_key": r[4] or "",
                "updated_at": r[5],
            }
            for r in rows
        ]
        return self._dedupe_conflicts(facts)

    def get_relevant_facts(self, query: str, limit: int = 8) -> list[dict]:
        """Hybrid vector+BM25 retrieval with RRF fusion, MMR, and temporal decay.

        Falls back to FTS-only when embedder is unavailable.
        """
        t0 = _time_mod.time()

        from liagent.agent.embedder import classify_query_type
        query_type = classify_query_type(query)
        vec_limit, fts_limit = _RECALL_WINDOWS.get(query_type, (15, 15))

        tokens = _tokenize(query)
        if not tokens:
            return self.get_all_facts(min_confidence=0.65)[:limit]

        now = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # FTS recall
            fts_results = self._fts_search(conn, tokens, now, fts_limit)

            # Vector recall (if embedder available)
            vec_results = []
            query_vec = None
            if self._embedder:
                vecs = self._embedder.encode([query])
                if vecs is not None:
                    query_vec = vecs[0]
                    vec_results = self._vec_search(conn, query_vec, now, vec_limit)

            # If no results from either, fall back
            if not fts_results and not vec_results:
                result = self._get_relevant_facts_legacy(query, limit)
                self._log_retrieval(conn, query, query_type, 0, 0, 0, len(result), t0)
                return result

            # RRF fusion
            fused = self._rrf_fuse(fts_results, vec_results)

            # MMR dedup (only if we have vectors)
            if query_vec is not None and any(c.get("_vec") is not None for c in fused):
                mmr_result = _mmr_rerank(fused, query_vec, lambda_param=0.7, top_n=limit * 2)
            else:
                mmr_result = fused[:limit * 2]

            # Temporal decay
            for f in mmr_result:
                updated_at = f.get("updated_at", now)
                try:
                    updated_dt = datetime.fromisoformat(updated_at)
                    age_days = (datetime.now(timezone.utc) - updated_dt).total_seconds() / 86400.0
                except (ValueError, TypeError):
                    age_days = 0.0
                source = f.get("source", "llm_extract")
                f["_rrf_score"] = _temporal_decay(f.get("_rrf_score", 0.0), age_days, source)

            # Sort by decayed score, dedupe, take limit
            mmr_result.sort(key=lambda x: x.get("_rrf_score", 0.0), reverse=True)
            deduped = self._dedupe_conflicts(mmr_result)[:limit]

            # Log retrieval
            self._log_retrieval(conn, query, query_type, len(fts_results), len(vec_results), len(fused), len(deduped), t0)

        # Clean internal fields before returning
        for f in deduped:
            f.pop("_bm25_rank", None)
            f.pop("_semantic_score", None)
            f.pop("_rrf_score", None)
            f.pop("_vec", None)
            f.pop("_fts_rank", None)
            f.pop("_vec_rank", None)
        return deduped

    def get_relevant_evidence(self, query: str, limit: int = 8) -> list[EvidenceChunk]:
        """Return memory retrieval results as evidence chunks with stable refs."""
        facts = self.get_relevant_facts(query, limit=limit)
        now = datetime.now(timezone.utc).isoformat()
        out: list[EvidenceChunk] = []
        for idx, f in enumerate(facts[:limit]):
            ref = f"memory:key_facts:{f.get('fact_key') or idx}"
            evidence_id = f"ev_{idx + 1}"
            confidence = float(f.get("confidence", 0.7) or 0.7)
            semantic = _semantic_score(query, str(f.get("fact", "") or ""))
            score = round(max(0.0, min(1.0, (confidence + semantic) / 2.0)), 4)
            out.append(
                EvidenceChunk(
                    evidence_id=evidence_id,
                    source_type="memory_fact",
                    source_ref=ref,
                    line_start=None,
                    line_end=None,
                    snippet=str(f.get("fact", "") or "")[:280],
                    score=score,
                    retrieved_at=now,
                )
            )
        return out

    def _get_relevant_facts_legacy(self, query: str, limit: int = 8) -> list[dict]:
        """Legacy token-intersection fallback when FTS is unavailable."""
        facts = self.get_all_facts(min_confidence=0.65)
        query_tokens = {t for t in _tokenize(query) if t}
        if not query_tokens:
            return facts[:limit]

        scored = []
        for f in facts:
            fact_tokens = {t for t in _tokenize(f["fact"]) if t}
            lexical = len(query_tokens & fact_tokens)
            semantic = _semantic_score(query, f.get("fact", ""))
            score = lexical + semantic + int(f.get("confidence", 0.7) * 2)
            scored.append((score, f))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for score, f in scored if score > 0][:limit] or facts[:limit]

    def _fts_search(self, conn, tokens: list[str], now: str, limit: int) -> list[dict]:
        """FTS5 BM25 search, returning fact dicts with _fts_rank."""
        fts_query = " OR ".join(f'"{t}"' for t in tokens if t)
        if not fts_query:
            return []
        try:
            rows = conn.execute(
                """SELECT kf.id, kf.fact, kf.category, kf.confidence, kf.source,
                          kf.fact_key, kf.updated_at, bm25(key_facts_fts) AS rank
                   FROM key_facts_fts fts
                   JOIN key_facts kf ON kf.id = fts.rowid
                   WHERE key_facts_fts MATCH ?
                     AND (kf.expires_at IS NULL OR kf.expires_at > ?)
                     AND COALESCE(kf.confidence, 0.7) >= 0.65
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, now, limit),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            _log.warning("fts_search_error", error=str(exc))
            return []

        results = []
        for rank_idx, r in enumerate(rows, 1):
            results.append({
                "id": int(r[0]), "fact": r[1], "category": r[2],
                "confidence": float(r[3] or 0.7), "source": r[4] or "llm_extract",
                "fact_key": r[5] or "", "updated_at": r[6],
                "_bm25_rank": float(r[7] or 0.0),
                "_fts_rank": rank_idx,
            })
        return results

    def _vec_search(self, conn, query_vec: np.ndarray, now: str, limit: int) -> list[dict]:
        """Vector similarity search using stored embeddings."""
        rows = conn.execute(
            """SELECT fe.fact_id, fe.embedding, fe.dimensions,
                      kf.fact, kf.category, kf.confidence, kf.source,
                      kf.fact_key, kf.updated_at
               FROM fact_embeddings fe
               JOIN key_facts kf ON kf.id = fe.fact_id
               WHERE (kf.expires_at IS NULL OR kf.expires_at > ?)
                 AND COALESCE(kf.confidence, 0.7) >= 0.65""",
            (now,),
        ).fetchall()

        if not rows:
            return []

        scored = []
        for r in rows:
            vec = np.frombuffer(r[1], dtype=np.float32)
            qn = np.linalg.norm(query_vec)
            vn = np.linalg.norm(vec)
            if qn > 0 and vn > 0:
                sim = float(np.dot(query_vec, vec) / (qn * vn))
            else:
                sim = 0.0
            scored.append((sim, {
                "id": int(r[0]), "fact": r[3], "category": r[4],
                "confidence": float(r[5] or 0.7), "source": r[6] or "llm_extract",
                "fact_key": r[7] or "", "updated_at": r[8],
                "_vec": vec, "_vec_sim": sim,
            }))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for rank_idx, (sim, d) in enumerate(scored[:limit], 1):
            d["_vec_rank"] = rank_idx
            results.append(d)
        return results

    def _rrf_fuse(self, fts_results: list[dict], vec_results: list[dict]) -> list[dict]:
        """Reciprocal Rank Fusion over FTS and vector results."""
        by_id: dict[int, dict] = {}

        for f in fts_results:
            fid = f["id"]
            by_id[fid] = dict(f)
            by_id[fid]["_fts_rank"] = f.get("_fts_rank", 999)
            by_id[fid]["_vec_rank"] = 999

        for f in vec_results:
            fid = f["id"]
            if fid in by_id:
                by_id[fid]["_vec_rank"] = f.get("_vec_rank", 999)
                by_id[fid]["_vec"] = f.get("_vec")
            else:
                entry = dict(f)
                entry["_fts_rank"] = 999
                entry["_vec_rank"] = f.get("_vec_rank", 999)
                by_id[fid] = entry

        for fid, entry in by_id.items():
            entry["_rrf_score"] = _rrf_score(
                entry.get("_vec_rank", 999),
                entry.get("_fts_rank", 999),
            )

        fused = list(by_id.values())
        fused.sort(key=lambda x: x.get("_rrf_score", 0.0), reverse=True)
        return fused

    def _log_retrieval(self, conn, query: str, query_type: str,
                       fts_hits: int, vec_hits: int, fused_count: int,
                       final_count: int, t0: float):
        """Record retrieval metrics for observability."""
        latency_ms = (_time_mod.time() - t0) * 1000.0
        provider = self._embedder.model_name if self._embedder else "none"
        now = datetime.now(timezone.utc).isoformat()
        try:
            conn.execute(
                """INSERT INTO retrieval_log
                   (query, query_type, fts_hits, vec_hits, fused_count, final_count,
                    latency_ms, embedder_provider, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (query, query_type, fts_hits, vec_hits, fused_count, final_count,
                 latency_ms, provider, now),
            )
        except Exception as exc:
            _log.warning("retrieval_log_error", error=str(exc))

    @staticmethod
    def _fact_key(fact: str, category: str) -> str:
        tokens = _tokenize(fact)
        root = tokens[0] if tokens else fact[:12]
        return f"{category}:{root}".lower()

    @staticmethod
    def _dedupe_conflicts(facts: list[dict]) -> list[dict]:
        # Keep the highest-confidence version under the same fact_key/category.
        best: dict[tuple[str, str], dict] = {}
        for f in facts:
            key = (f.get("category", ""), f.get("fact_key", "") or f.get("fact", ""))
            cur = best.get(key)
            if cur is None or float(f.get("confidence", 0.0)) > float(cur.get("confidence", 0.0)):
                best[key] = f
        # Preserve recency-ish order from original list.
        ordered = []
        seen = set()
        for f in facts:
            key = (f.get("category", ""), f.get("fact_key", "") or f.get("fact", ""))
            if key in seen:
                continue
            if best.get(key) is f:
                ordered.append(f)
                seen.add(key)
        return ordered


    def _rebuild_fts(self, conn):
        """Rebuild FTS5 index from all key_facts rows."""
        conn.execute("DELETE FROM key_facts_fts")
        for row_id, fact in conn.execute("SELECT id, fact FROM key_facts").fetchall():
            conn.execute(
                "INSERT INTO key_facts_fts(rowid, fact_text) VALUES (?, ?)",
                (row_id, _tokenize_for_fts(fact)),
            )

    # --- User feedback ---

    def save_feedback(
        self,
        session_id: str,
        turn_index: int,
        query: str,
        answer: str,
        tool_used: str | None,
        feedback: str,
        source: str = "ui_button",
    ):
        """Record user feedback (positive/negative) for a turn."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO user_feedback
                   (session_id, turn_index, query, answer, tool_used, feedback, source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, turn_index, query, answer[:2000], tool_used, feedback, source, now),
            )

    # --- Memory hygiene ---

    _SOURCE_CONFIDENCE = {
        "tool_result": 0.9,
        "user_stated": 0.85,
        "llm_extract": 0.6,
        "llm_inferred": 0.4,
    }

    def decay_confidence(self, decay_rate: float = 0.02, min_age_days: int = 7):
        """Reduce confidence of facts not updated in min_age_days.

        User-stated and tool-result facts decay at 1/4 the normal rate
        to preserve explicitly remembered information longer.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=min_age_days)
        ).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            # Slow decay for high-trust sources (user_stated, tool_result)
            conn.execute(
                """UPDATE key_facts
                   SET confidence = MAX(0.05, confidence - ?)
                   WHERE updated_at < ?
                     AND COALESCE(confidence, 0.7) > 0.1
                     AND COALESCE(source, 'llm_extract') IN ('user_stated', 'tool_result')""",
                (decay_rate * 0.25, cutoff),
            )
            # Normal decay for LLM-extracted/inferred facts
            conn.execute(
                """UPDATE key_facts
                   SET confidence = MAX(0.05, confidence - ?)
                   WHERE updated_at < ?
                     AND COALESCE(confidence, 0.7) > 0.1
                     AND COALESCE(source, 'llm_extract') NOT IN ('user_stated', 'tool_result')""",
                (decay_rate, cutoff),
            )

    def prune_memory(self, min_confidence: float = 0.15, keep_recent: int = 50):
        """Delete low-confidence or expired facts, keeping at most keep_recent."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            # Delete expired
            cur_exp = conn.execute(
                "DELETE FROM key_facts WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )
            n_expired = cur_exp.rowcount
            # Delete very low confidence
            cur_low = conn.execute(
                "DELETE FROM key_facts WHERE COALESCE(confidence, 0.7) < ?",
                (min_confidence,),
            )
            n_low = cur_low.rowcount
            # Keep only the most recent
            n_excess = 0
            count = conn.execute("SELECT COUNT(*) FROM key_facts").fetchone()[0]
            if count > keep_recent:
                cur_exc = conn.execute(
                    """DELETE FROM key_facts WHERE id NOT IN (
                        SELECT id FROM key_facts ORDER BY updated_at DESC LIMIT ?
                    )""",
                    (keep_recent,),
                )
                n_excess = cur_exc.rowcount
            # Rebuild FTS after pruning
            self._rebuild_fts(conn)
        if self.journal:
            self.journal.memory_pruned(
                expired=n_expired, low_confidence=n_low, excess=n_excess,
            )

    def detect_conflicts(self, new_fact: str, category: str, new_confidence: float):
        """If a conflicting fact exists under the same key, lower the old one's confidence."""
        fact_key = self._fact_key(new_fact, category)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT id, fact, confidence FROM key_facts
                   WHERE fact_key = ? AND fact != ?""",
                (fact_key, new_fact),
            ).fetchall()
            for row_id, old_fact, old_conf in rows:
                # Demote conflicting fact
                old_conf_f = float(old_conf or 0.7)
                demoted = max(0.1, old_conf_f * 0.5)
                conn.execute(
                    "UPDATE key_facts SET confidence = ? WHERE id = ?",
                    (demoted, row_id),
                )
                if self.journal:
                    self.journal.fact_conflict(
                        new_fact=new_fact, old_fact=old_fact,
                        old_confidence=old_conf_f, demoted_confidence=demoted,
                    )

    def apply_source_confidence(self, facts: list[dict]) -> list[dict]:
        """Set initial confidence based on source type for new facts."""
        for f in facts:
            source = str(f.get("source", "llm_extract"))
            default_conf = self._SOURCE_CONFIDENCE.get(source, 0.6)
            if "confidence" not in f or f["confidence"] is None:
                f["confidence"] = default_conf
        return facts

    # --- Markdown sync (facts.md) ---

    def sync_facts_to_markdown(self):
        """Write all key_facts to facts.md for human review/backup."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT fact, category, confidence, source, updated_at FROM key_facts ORDER BY updated_at DESC"
            ).fetchall()
        if not rows:
            return

        self.facts_md_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# LiAgent User Facts\n"]
        for fact, category, confidence, source, updated_at in rows:
            lines.append(f"## {fact}")
            lines.append(f"- **category**: {category or ''}")
            lines.append(f"- **confidence**: {confidence or 0.7}")
            lines.append(f"- **source**: {source or 'llm_extract'}")
            lines.append(f"- **updated_at**: {updated_at or ''}")
            lines.append("")

        self.facts_md_path.write_text("\n".join(lines), encoding="utf-8")

    def sync_facts_from_markdown(self):
        """Read facts.md and merge user edits back into SQLite."""
        if not self.facts_md_path.exists():
            return

        text = self.facts_md_path.read_text(encoding="utf-8")
        blocks = re.split(r"^## ", text, flags=re.MULTILINE)

        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            existing = {
                row[0]: row[1]
                for row in conn.execute("SELECT fact, id FROM key_facts").fetchall()
            }
            for block in blocks[1:]:  # skip header
                parsed = self._parse_fact_md_block(block)
                if not parsed or not parsed.get("fact"):
                    continue
                fact = parsed["fact"]
                category = parsed.get("category", "")
                confidence = parsed.get("confidence", 0.7)
                source = parsed.get("source", "llm_extract")

                if fact in existing:
                    # Update confidence if user edited it
                    conn.execute(
                        """UPDATE key_facts SET confidence = ?, category = ?, updated_at = ?
                           WHERE fact = ?""",
                        (confidence, category, now, fact),
                    )
                else:
                    # New fact from user edit
                    fact_key = self._fact_key(fact, category)
                    conn.execute(
                        """INSERT OR IGNORE INTO key_facts
                           (fact, category, created_at, updated_at, confidence, source, fact_key)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (fact, category, now, now, confidence, "user_edit", fact_key),
                    )
            # Rebuild FTS to include any new/updated facts
            self._rebuild_fts(conn)

    def get_recent_feedback(self, days: int = 7) -> list[dict]:
        """Retrieve recent user feedback entries."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT query, answer, tool_used, feedback, source, created_at "
                "FROM user_feedback WHERE created_at >= ? "
                "ORDER BY created_at DESC LIMIT 50",
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _parse_fact_md_block(block: str) -> dict | None:
        """Parse a single ## block from facts.md."""
        lines = block.strip().split("\n")
        if not lines:
            return None
        result: dict = {"fact": lines[0].strip()}
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("- **category**:"):
                result["category"] = line.split(":", 1)[1].strip()
            elif line.startswith("- **confidence**:"):
                try:
                    result["confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("- **source**:"):
                result["source"] = line.split(":", 1)[1].strip()
        return result if result.get("fact") else None



def _tokenize(text: str) -> list[str]:
    """Segment text using jieba for Chinese + regex for alphanumeric."""
    import jieba
    import re

    tokens: list[str] = []
    for tok in jieba.cut_for_search(text or ""):
        tok = tok.strip().lower()
        if tok and len(tok) > 0 and not tok.isspace():
            tokens.append(tok)

    # Keep the full normalized English phrase as an additional search token
    # so exact phrase queries (e.g. "drink coffee") can match directly.
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if (
        normalized
        and " " in normalized
        and re.fullmatch(r"[a-z0-9][a-z0-9 ._/-]*", normalized)
        and normalized not in tokens
    ):
        tokens.append(normalized)
    return tokens


def _tokenize_for_fts(text: str) -> str:
    """Return space-joined jieba tokens for FTS5 indexing."""
    return " ".join(_tokenize(text))


def _semantic_vector(text: str) -> dict[str, float]:
    """Build a lightweight sparse semantic vector from tokens."""
    vec: dict[str, float] = {}
    for tok in _tokenize(text):
        if len(tok) < 2:
            continue
        vec[tok] = vec.get(tok, 0.0) + 1.0
    if not vec:
        return {}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm <= 0:
        return {}
    return {k: (v / norm) for k, v in vec.items()}


def _semantic_score(query: str, text: str) -> float:
    """Cosine similarity over lightweight sparse vectors, in [0, 1]."""
    qv = _semantic_vector(query)
    tv = _semantic_vector(text)
    if not qv or not tv:
        return 0.0
    if len(qv) > len(tv):
        qv, tv = tv, qv
    dot = 0.0
    for k, v in qv.items():
        dot += v * tv.get(k, 0.0)
    return max(0.0, min(1.0, dot))


class UserProfileStore:
    """CRUD for user_profile_slots. Stateless -- opens a new connection per call."""

    _CREATE_TABLE_SQL = """CREATE TABLE IF NOT EXISTS user_profile_slots (
        dimension              TEXT PRIMARY KEY,
        value                  TEXT NOT NULL,
        confidence             REAL NOT NULL DEFAULT 0.5,
        source                 TEXT NOT NULL DEFAULT 'implicit',
        locked                 INTEGER NOT NULL DEFAULT 0,
        evidence_count         INTEGER NOT NULL DEFAULT 0,
        candidate_value        TEXT NOT NULL DEFAULT '',
        candidate_evidence_count INTEGER NOT NULL DEFAULT 0,
        updated_at             TEXT NOT NULL
    )"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self):
        """Ensure the user_profile_slots table exists (safe for :memory: DBs)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(self._CREATE_TABLE_SQL)

    @staticmethod
    def _row_to_dict(row, cursor) -> dict:
        cols = [d[0] for d in cursor.description]
        return dict(zip(cols, row))

    def get(self, dimension: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(self._CREATE_TABLE_SQL)
            cur = conn.execute(
                "SELECT * FROM user_profile_slots WHERE dimension = ?",
                (dimension,),
            )
            row = cur.fetchone()
            return self._row_to_dict(row, cur) if row else None

    def get_all(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(self._CREATE_TABLE_SQL)
            cur = conn.execute("SELECT * FROM user_profile_slots")
            return [self._row_to_dict(r, cur) for r in cur.fetchall()]

    def upsert(
        self,
        dimension: str,
        value: str,
        *,
        confidence: float = 0.5,
        source: str = "implicit",
        locked: int = 0,
        evidence_count: int = 0,
        candidate_value: str = "",
        candidate_evidence_count: int = 0,
    ):
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO user_profile_slots
                   (dimension, value, confidence, source, locked,
                    evidence_count, candidate_value, candidate_evidence_count, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(dimension) DO UPDATE SET
                       value = excluded.value,
                       confidence = excluded.confidence,
                       source = excluded.source,
                       locked = excluded.locked,
                       evidence_count = excluded.evidence_count,
                       candidate_value = excluded.candidate_value,
                       candidate_evidence_count = excluded.candidate_evidence_count,
                       updated_at = excluded.updated_at""",
                (dimension, value, confidence, source, locked,
                 evidence_count, candidate_value, candidate_evidence_count, now),
            )

    def delete(self, dimension: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM user_profile_slots WHERE dimension = ?",
                (dimension,),
            )

    def delete_all(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM user_profile_slots")

    # ---- Explicit write API ----

    def set_explicit(self, dimension: str, value: str):
        """User-stated preference — always locked, confidence=0.95."""
        if dimension == "domains":
            value = self.normalize_domains(value)
        self.upsert(
            dimension, value,
            confidence=0.95,
            source="user_stated",
            locked=1,
            evidence_count=1,
            candidate_value="",
            candidate_evidence_count=0,
        )

    def forget(self, dimension: str):
        """Delete a specific dimension slot."""
        self.delete(dimension)

    def forget_all(self):
        """Clear all profile slots."""
        self.delete_all()

    # ---- Hysteresis merge logic ----

    _NUDGE = {"strong": 0.16, "moderate": 0.08, "weak": 0.04}

    @staticmethod
    def normalize_domains(raw: str) -> str:
        """lower() -> strip() -> split(',') -> dedup -> sort -> join(',')."""
        if not raw or not raw.strip():
            return ""
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
        seen: set[str] = set()
        deduped: list[str] = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        return ",".join(sorted(deduped))

    # ---- Portrait compilation ----

    def compile_portrait(self) -> str:
        """Compile top-5 profile slots into natural language portrait.

        Hard preferences (locked/user_stated) → "requires"
        Soft preferences (implicit) → "tends to prefer"
        """
        slots = self.get_all()
        # Filter confidence > 0.3, sort by confidence DESC, cap at 5
        slots = [s for s in slots if s["confidence"] > 0.3]
        slots.sort(key=lambda s: s["confidence"], reverse=True)
        slots = slots[:5]

        if not slots:
            return ""

        hard = []
        soft = []
        for s in slots:
            dim, val = s["dimension"], s["value"]
            is_hard = s["locked"] or s["source"] == "user_stated"
            label = f"{dim}: {val}"
            if is_hard:
                hard.append(label)
            else:
                soft.append(label)

        lines = ["[User Profile]"]
        if hard:
            lines.append(
                "The user requires: " + "; ".join(hard) + "."
            )
        if soft:
            lines.append(
                "The user tends to prefer: " + "; ".join(soft) + "."
            )
        return "\n".join(lines) + "\n"

    def merge_implicit(self, dimension: str, value: str, signal_strength: str):
        """Hysteresis merge -- design doc section 4."""
        nudge = self._NUDGE.get(signal_strength, 0.04)

        # Normalize domains
        if dimension == "domains":
            value = self.normalize_domains(value)

        slot = self.get(dimension)

        # Case 4: locked -> skip
        if slot and slot["locked"]:
            return

        # Case 1: new dimension
        if slot is None:
            self.upsert(
                dimension, value,
                confidence=0.3 + nudge,
                source="implicit",
                evidence_count=1,
            )
            return

        conf = slot["confidence"]
        ev = slot["evidence_count"]
        cand_val = slot["candidate_value"]
        cand_count = slot["candidate_evidence_count"]

        # Case 2: same direction
        if value == slot["value"]:
            conf = min(1.0, conf + nudge)
            ev += 1
            self.upsert(
                dimension, value,
                confidence=conf,
                source="implicit",
                evidence_count=ev,
                candidate_value=cand_val,
                candidate_evidence_count=cand_count,
            )
            return

        # Case 3: opposing direction
        conf = max(0.0, conf - 0.15)
        if cand_val == "" or cand_val == value:
            cand_val = value
            cand_count += 1
        else:
            # Different third-party direction -- reset candidate
            cand_val = value
            cand_count = 1

        # Check flip condition
        if conf < 0.3 and cand_count >= 3:
            self.upsert(
                dimension, cand_val,
                confidence=0.3 + nudge,
                source="implicit",
                evidence_count=cand_count,
                candidate_value="",
                candidate_evidence_count=0,
            )
        else:
            self.upsert(
                dimension, slot["value"],
                confidence=conf,
                source="implicit",
                locked=slot["locked"],
                evidence_count=ev,
                candidate_value=cand_val,
                candidate_evidence_count=cand_count,
            )
