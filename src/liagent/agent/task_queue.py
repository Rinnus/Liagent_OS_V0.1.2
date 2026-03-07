"""Autonomous task storage (SQLite) and priority-based execution engine."""

import asyncio
import json
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

from .memory import DB_PATH, ConversationMemory, LongTermMemory
from ..logging import get_logger
from .time_utils import _now_local, _now_local_iso

_log = get_logger("task_queue")

_HEARTBEAT_TOOL_RE = re.compile(r"^Tool:\s*(\S+)", re.MULTILINE)


def _heartbeat_budget_from_prompt(prompt: str):
    """Recover a fail-closed heartbeat budget from the persisted execution prompt."""
    from ..skills.router import BudgetOverride

    match = _HEARTBEAT_TOOL_RE.search(prompt or "")
    tool_name = match.group(1) if match else None
    return BudgetOverride(
        allowed_tools={tool_name} if tool_name else set(),
        max_tool_calls=1,
        timeout_ms=30_000,
    )


# ─── TaskStore ────────────────────────────────────────────────────────────────

class TaskStore:
    """SQLite-backed storage for autonomous tasks and their run history."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS autonomous_tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    trigger_config TEXT NOT NULL DEFAULT '{}',
                    prompt_template TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    priority INTEGER NOT NULL DEFAULT 10,
                    created_at TEXT NOT NULL,
                    next_run_at TEXT,
                    last_run_at TEXT,
                    result_summary TEXT,
                    error_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 2
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS autonomous_task_runs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES autonomous_tasks(id),
                    trigger_event TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    prompt TEXT,
                    result TEXT,
                    error TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT
                )"""
            )
            # Migration: add expires_at to existing DBs
            try:
                conn.execute(
                    "ALTER TABLE autonomous_task_runs ADD COLUMN expires_at TEXT"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists
            # Indexes for common queries
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_tasks_status_trigger
                   ON autonomous_tasks (status, trigger_type)"""
            )
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_runs_task_created
                   ON autonomous_task_runs (task_id, created_at)"""
            )
            # One-time cleanup: soft-delete tasks with unsupported trigger_type
            conn.execute(
                """UPDATE autonomous_tasks SET status = 'deleted'
                   WHERE trigger_type NOT IN ('cron', 'once', 'filewatch', 'heartbeat')
                   AND status != 'deleted'"""
            )

            # --- Goal Autonomy migrations ---
            task_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(autonomous_tasks)").fetchall()
            }
            if "source" not in task_cols:
                conn.execute("ALTER TABLE autonomous_tasks ADD COLUMN source TEXT DEFAULT 'user_manual'")
            if "goal_id" not in task_cols:
                conn.execute("ALTER TABLE autonomous_tasks ADD COLUMN goal_id INTEGER")
            if "dedup_key" not in task_cols:
                conn.execute("ALTER TABLE autonomous_tasks ADD COLUMN dedup_key TEXT")
            if "risk_level" not in task_cols:
                conn.execute("ALTER TABLE autonomous_tasks ADD COLUMN risk_level TEXT DEFAULT 'low'")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_dedup_active "
                "ON autonomous_tasks(dedup_key) "
                "WHERE dedup_key IS NOT NULL AND status IN ('active', 'paused')"
            )

            run_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(autonomous_task_runs)").fetchall()
            }
            if "origin" not in run_cols:
                conn.execute("ALTER TABLE autonomous_task_runs ADD COLUMN origin TEXT DEFAULT 'user'")
            if "goal_id" not in run_cols:
                conn.execute("ALTER TABLE autonomous_task_runs ADD COLUMN goal_id INTEGER")
            if "confidence_label" not in run_cols:
                conn.execute("ALTER TABLE autonomous_task_runs ADD COLUMN confidence_label TEXT")
            if "evidence_json" not in run_cols:
                conn.execute("ALTER TABLE autonomous_task_runs ADD COLUMN evidence_json TEXT")

    # ── Task CRUD ──────────────────────────────────────────────────────────

    def create_task(
        self,
        *,
        name: str,
        trigger_type: str,
        trigger_config: dict,
        prompt_template: str,
        priority: int = 10,
        max_retries: int = 2,
    ) -> dict:
        task_id = uuid.uuid4().hex[:12]
        now = _now_local_iso()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO autonomous_tasks
                   (id, name, trigger_type, trigger_config, prompt_template,
                    status, priority, created_at, max_retries)
                   VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)""",
                (
                    task_id, name, trigger_type,
                    json.dumps(trigger_config, ensure_ascii=False),
                    prompt_template, priority, now, max_retries,
                ),
            )
        return self.get_task(task_id)

    def list_tasks(self, *, include_deleted: bool = False, include_system: bool = True) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if include_deleted:
                rows = conn.execute(
                    "SELECT * FROM autonomous_tasks ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM autonomous_tasks WHERE status != 'deleted' ORDER BY created_at DESC"
                ).fetchall()
        results = [dict(r) for r in rows]
        if not include_system:
            results = [t for t in results if t.get("trigger_type") != "heartbeat"]
        return results

    def get_task(self, task_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM autonomous_tasks WHERE id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        # Parse trigger_config back to dict
        try:
            result["trigger_config"] = json.loads(result.get("trigger_config", "{}"))
        except (json.JSONDecodeError, TypeError):
            result["trigger_config"] = {}
        return result

    def update_task(self, task_id: str, **fields) -> dict | None:
        if not fields:
            return self.get_task(task_id)
        allowed = {
            "name", "trigger_type", "trigger_config", "prompt_template",
            "status", "priority", "next_run_at", "last_run_at",
            "result_summary", "error_count", "max_retries",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_task(task_id)
        # Serialize trigger_config if present
        if "trigger_config" in updates and isinstance(updates["trigger_config"], dict):
            updates["trigger_config"] = json.dumps(updates["trigger_config"], ensure_ascii=False)
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [task_id]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE autonomous_tasks SET {set_clause} WHERE id = ?",
                values,
            )
        return self.get_task(task_id)

    def delete_task(self, task_id: str) -> bool:
        """Soft-delete a task."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE autonomous_tasks SET status = 'deleted' WHERE id = ? AND status != 'deleted'",
                (task_id,),
            )
        return cur.rowcount > 0

    def pause_task(self, task_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE autonomous_tasks SET status = 'paused' WHERE id = ? AND status = 'active'",
                (task_id,),
            )
        return cur.rowcount > 0

    def resume_task(self, task_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE autonomous_tasks SET status = 'active' WHERE id = ? AND status = 'paused'",
                (task_id,),
            )
        return cur.rowcount > 0

    # ── Run tracking ───────────────────────────────────────────────────────

    def create_run(
        self,
        task_id: str,
        *,
        trigger_event: str = "",
        prompt: str = "",
    ) -> dict:
        run_id = uuid.uuid4().hex[:12]
        now = _now_local_iso()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO autonomous_task_runs
                   (id, task_id, trigger_event, status, prompt, created_at)
                   VALUES (?, ?, ?, 'pending', ?, ?)""",
                (run_id, task_id, trigger_event, prompt, now),
            )
        return {"id": run_id, "task_id": task_id, "status": "pending"}

    def update_run(self, run_id: str, **fields) -> None:
        allowed = {
            "status", "result", "error", "started_at", "finished_at", "expires_at",
            "origin", "goal_id", "confidence_label", "evidence_json",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [run_id]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE autonomous_task_runs SET {set_clause} WHERE id = ?",
                values,
            )

    def transition_run(self, run_id: str, from_status: str, to_status: str, **fields) -> bool:
        """Atomic state transition: updates only if current status matches from_status.
        Returns True if transition succeeded, False if run was already in a different state."""
        allowed_extra = {"result", "error", "started_at", "finished_at", "expires_at"}
        extras = {k: v for k, v in fields.items() if k in allowed_extra}
        updates = {"status": to_status, **extras}
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [run_id, from_status]
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                f"UPDATE autonomous_task_runs SET {set_clause} WHERE id = ? AND status = ?",
                values,
            )
        return cur.rowcount > 0

    def get_expired_pending_confirms(self) -> list[str]:
        """Return run IDs that are pending_confirm with expired deadline."""
        now = _now_local_iso()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id FROM autonomous_task_runs "
                "WHERE status = 'pending_confirm' AND expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            ).fetchall()
        return [r[0] for r in rows]

    def get_recent_runs(self, task_id: str, limit: int = 10) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM autonomous_task_runs
                   WHERE task_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (task_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recoverable_runs(self, *, statuses: tuple[str, ...] = ("pending", "confirmed")) -> list[dict]:
        """Return runs that should be re-enqueued after a process restart."""
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""SELECT * FROM autonomous_task_runs
                    WHERE status IN ({placeholders})
                    ORDER BY created_at ASC""",
                statuses,
            ).fetchall()
        return [dict(r) for r in rows]

    def recover_stale_runs(self) -> int:
        """Reset stale running/pending runs from a previous crash.

        - running → error (with crash note) — re-executing may cause duplicates
        - pending → left as-is (safe to re-enqueue later)

        Returns the number of runs recovered.
        """
        now = _now_local_iso()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """UPDATE autonomous_task_runs
                   SET status = 'error', error = 'process_crash_recovery', finished_at = ?
                   WHERE status = 'running'""",
                (now,),
            )
            return cur.rowcount

    def create_task_from_prompt(
        self, *, prompt: str, source: str = "user_manual",
        goal_id: int | None = None, dedup_key: str | None = None,
        risk_level: str = "low", trigger_type: str = "once",
        trigger_config: dict | None = None, priority: int = 10,
    ) -> dict | None:
        """Convenience wrapper for goal/bridge task creation. Returns None if dedup_key exists."""
        if dedup_key:
            with sqlite3.connect(self.db_path) as conn:
                existing = conn.execute(
                    "SELECT id FROM autonomous_tasks "
                    "WHERE dedup_key = ? AND status IN ('active', 'paused')",
                    (dedup_key,),
                ).fetchone()
                if existing:
                    return None
        task = self.create_task(
            name=prompt[:80], trigger_type=trigger_type,
            trigger_config=trigger_config or {}, prompt_template=prompt, priority=priority,
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE autonomous_tasks SET source=?, goal_id=?, dedup_key=?, risk_level=? WHERE id=?",
                (source, goal_id, dedup_key, risk_level, task["id"]),
            )
        return self.get_task(task["id"])

    def get_recent_runs_for_goal(self, goal_id: int, *, limit: int = 10) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM autonomous_task_runs WHERE goal_id = ? ORDER BY created_at DESC LIMIT ?",
                (goal_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_active_tasks_by_trigger(self, trigger_type: str) -> list[dict]:
        """Get all active tasks of a given trigger type."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM autonomous_tasks WHERE trigger_type = ? AND status = 'active'",
                (trigger_type,),
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["trigger_config"] = json.loads(d.get("trigger_config", "{}"))
            except (json.JSONDecodeError, TypeError):
                d["trigger_config"] = {}
            results.append(d)
        return results


# ─── TaskExecutor ─────────────────────────────────────────────────────────────

@dataclass
class _QueueItem:
    """Priority queue item. Lower priority number = higher priority."""
    priority: int
    run_id: str
    task_id: str
    prompt: str
    trigger_event: str = ""
    budget: "BudgetOverride | None" = field(default=None, compare=False)
    goal_id: int | None = field(default=None, compare=False)
    source: str = field(default="user_manual", compare=False)

    def __lt__(self, other: "_QueueItem") -> bool:
        return self.priority < other.priority


class TaskExecutor:
    """Priority-based async executor that runs autonomous tasks through Brain.

    Key properties:
    - Shares the Brain's run_lock to serialize access with user requests
    - Isolates conversation memory per task execution
    - Supports preemption: user requests cancel running background tasks
    - Broadcasts results via on_result callback
    """

    def __init__(
        self,
        brain,
        store: TaskStore,
        run_lock: asyncio.Lock,
        on_result: Callable[[dict], Coroutine] | None = None,
        goal_store=None,
    ):
        self.brain = brain
        self.store = store
        self.run_lock = run_lock
        self.on_result = on_result
        self.goal_store = goal_store
        self._queue: asyncio.PriorityQueue[_QueueItem] = asyncio.PriorityQueue()
        self._current_task: asyncio.Task | None = None
        self._current_run_id: str | None = None
        self._current_task_id: str | None = None
        self._consumer_task: asyncio.Task | None = None
        self._task_swap_lock = asyncio.Lock()
        self._stopped = False

    async def start(self):
        """Start the background consumer loop and recover stale runs from crashes."""
        self._stopped = False
        recovered = self.store.recover_stale_runs()
        if recovered:
            _log.event("stale_runs_recovered", count=recovered)
        replayed = 0
        for run in self.store.get_recoverable_runs():
            task_meta = self.store.get_task(run["task_id"])
            if task_meta is None or task_meta.get("status") != "active":
                self.store.update_run(
                    run["id"],
                    status="error",
                    error="recovery_task_missing",
                    finished_at=_now_local_iso(),
                )
                continue
            budget = None
            trigger_event = str(run.get("trigger_event") or "")
            if trigger_event.startswith("heartbeat:"):
                budget = _heartbeat_budget_from_prompt(str(run.get("prompt") or ""))
            origin = str(run.get("origin") or "")
            if origin == "goal":
                source = "goal"
            elif origin == "system":
                source = "system"
            else:
                source = "user_manual"
            self.enqueue(
                run["id"],
                run["task_id"],
                str(run.get("prompt") or ""),
                priority=int(task_meta.get("priority", 10) or 10),
                trigger_event=trigger_event,
                budget=budget,
                goal_id=run.get("goal_id"),
                source=source,
            )
            replayed += 1
        if replayed:
            _log.event("recoverable_runs_enqueued", count=replayed)
        self._consumer_task = asyncio.create_task(self._consumer_loop())
        _log.event("consumer_loop_started")

    async def stop(self):
        """Stop the consumer loop and cancel any running task."""
        self._stopped = True
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except (asyncio.CancelledError, Exception):
                pass

    def enqueue(self, run_id: str, task_id: str, prompt: str, *,
                priority: int = 10, trigger_event: str = "",
                budget: "BudgetOverride | None" = None,
                goal_id: int | None = None,
                source: str = "user_manual"):
        """Add a task run to the priority queue."""
        item = _QueueItem(
            priority=priority,
            run_id=run_id,
            task_id=task_id,
            prompt=prompt,
            trigger_event=trigger_event,
            budget=budget,
            goal_id=goal_id,
            source=source,
        )
        self._queue.put_nowait(item)
        _log.event("task_enqueued", run_id=run_id, task_id=task_id, priority=priority)

    async def preempt(self):
        """Cancel the currently running background task for user preemption.

        The run is marked as 'preempted' (not pending/error).
        For once tasks, a new run is scheduled after the user interaction completes.
        For cron tasks, the next scheduled trigger will create a new run.
        """
        if self._current_task and not self._current_task.done():
            run_id = self._current_run_id
            task_id = self._current_task_id
            _log.event("task_preempting", run_id=run_id)
            self._current_task.cancel()
            try:
                await asyncio.wait_for(self._current_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
            if run_id:
                self.store.update_run(run_id, status="preempted")

    async def _consumer_loop(self):
        """Background loop that dequeues and executes task runs."""
        while not self._stopped:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            self._current_run_id = item.run_id
            self._current_task_id = item.task_id
            self._current_task = asyncio.create_task(
                self._execute_run(item)
            )
            try:
                await self._current_task
            except asyncio.CancelledError:
                _log.event("task_preempted", run_id=item.run_id)
            except Exception as e:
                _log.error("task_executor", e, action=f"execute_run={item.run_id}")
            finally:
                self._current_task = None
                self._current_run_id = None
                self._current_task_id = None

    async def _execute_run(self, item: _QueueItem):
        """Execute a single task run with isolated memory."""
        run_id = item.run_id
        task_id = item.task_id
        now = _now_local_iso()
        self.store.update_run(run_id, status="running", started_at=now)

        # Isolated conversation memory for the task
        task_memory = ConversationMemory(max_turns=10)
        original_memory = None

        try:
            # Acquire brain lock + swap memory
            async with self.run_lock:
                async with self._task_swap_lock:
                    original_memory = self.brain.memory
                    self.brain.memory = task_memory

                try:
                    # Collect the full response
                    _origin = "goal" if item.goal_id else ("system" if item.source != "user_manual" else "user")
                    result_parts: list[str] = []
                    _quality_meta: dict | None = None
                    async for event in self.brain.run(
                        item.prompt, budget=item.budget,
                        execution_origin=_origin,
                        goal_id=item.goal_id,
                    ):
                        etype = event[0]
                        if etype == "token":
                            result_parts.append(event[1])
                        elif etype == "done":
                            if event[1]:
                                result_parts = [event[1]]
                            if len(event) > 2 and isinstance(event[2], dict):
                                _quality_meta = event[2]
                        elif etype == "error":
                            raise RuntimeError(event[1])
                finally:
                    # Always restore original memory
                    async with self._task_swap_lock:
                        self.brain.memory = original_memory

            result_text = "".join(result_parts).strip()
            finished = _now_local_iso()
            _run_updates: dict[str, Any] = {
                "status": "success", "result": result_text, "finished_at": finished,
            }
            if _quality_meta:
                _run_updates["confidence_label"] = _quality_meta.get("confidence_label", "")
                _run_updates["evidence_json"] = json.dumps(
                    _quality_meta.get("issues", []), ensure_ascii=False,
                )
            self.store.update_run(run_id, **_run_updates)
            # Update task metadata
            summary = result_text[:200] if result_text else ""
            task_updates: dict[str, Any] = {
                "last_run_at": finished, "result_summary": summary, "error_count": 0,
            }
            # Once tasks: mark done after successful execution (not at trigger time)
            task_meta = self.store.get_task(task_id)
            if task_meta and task_meta.get("trigger_type") == "once":
                task_updates["status"] = "done"
            self.store.update_task(task_id, **task_updates)

            # Always persist origin for audit trail
            self.store.update_run(run_id, origin=_origin)

            # Goal feedback: update goal confidence on task completion
            if item.goal_id:
                self.store.update_run(run_id, goal_id=item.goal_id)
                if self.goal_store:
                    try:
                        self.goal_store.adjust_confidence(item.goal_id, delta=0.05)
                        self.goal_store.record_event(
                            item.goal_id, "task_completed",
                            {"task_id": task_id, "run_id": run_id, "summary": summary},
                        )
                    except Exception:
                        pass  # goal feedback must not break task execution

            # Broadcast result — errors here must NOT overwrite the success status
            if self.on_result:
                try:
                    task = self.store.get_task(task_id)
                    await self.on_result({
                        "type": "task_result",
                        "task_id": task_id,
                        "task_name": task["name"] if task else "",
                        "run_id": run_id,
                        "status": "success",
                        "result": result_text,
                        "trigger_event": item.trigger_event or "",
                        "finished_at": finished,
                    })
                except Exception as e:
                    _log.error("task_executor", e, action=f"broadcast_result={run_id}")

        except asyncio.CancelledError:
            # Restore memory if not already restored
            if original_memory is not None:
                async with self._task_swap_lock:
                    if self.brain.memory is task_memory:
                        self.brain.memory = original_memory
            raise

        except Exception as e:
            # Restore memory if not already restored
            if original_memory is not None:
                async with self._task_swap_lock:
                    if self.brain.memory is task_memory:
                        self.brain.memory = original_memory

            error_msg = str(e)
            finished = _now_local_iso()
            self.store.update_run(
                run_id, status="error", error=error_msg, finished_at=finished,
                origin=_origin,
            )

            # Increment error count, auto-pause if exceeded
            task = self.store.get_task(task_id)
            if task:
                new_count = (task.get("error_count", 0) or 0) + 1
                max_retries = task.get("max_retries", 2) or 2
                updates: dict[str, Any] = {"error_count": new_count, "last_run_at": finished}
                if new_count > max_retries:
                    updates["status"] = "paused"
                    _log.warning(
                        f"task_executor: task={task_id} auto-paused after {new_count} errors"
                    )
                self.store.update_task(task_id, **updates)

            # Goal feedback on failure
            if item.goal_id and self.goal_store:
                try:
                    self.goal_store.adjust_confidence(item.goal_id, delta=-0.05)
                    self.goal_store.record_event(
                        item.goal_id, "task_failed",
                        {"task_id": task_id, "run_id": run_id, "error": str(e)[:200]},
                    )
                except Exception:
                    pass

            # Broadcast error — must not raise further
            if self.on_result:
                try:
                    await self.on_result({
                        "type": "task_result",
                        "task_id": task_id,
                        "task_name": task["name"] if task else "",
                        "run_id": run_id,
                        "status": "error",
                        "error": error_msg,
                        "trigger_event": item.trigger_event or "",
                        "finished_at": finished,
                    })
                except Exception as be:
                    _log.error("task_executor", be, action=f"broadcast_error={run_id}")
