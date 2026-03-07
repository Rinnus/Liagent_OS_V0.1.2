"""Knowledge Projector — SQLite event log → Obsidian Vault one-way projection.

SQLite is the source of truth. The vault is a read-only projection view.
Events are replayed from the event log and projected to markdown files
in the vault directory.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .event_log import EventLog, ChangeEvent
from ..logging import get_logger

_log = get_logger("projector")

_DEFAULT_VAULT_DIR = Path.home() / ".liagent" / "vault"
_SAFE_FILENAME_RE = re.compile(r"[^\w\-.]", re.ASCII)


def _safe_filename(name: str, max_len: int = 60) -> str:
    """Convert a name to a safe filename."""
    safe = _SAFE_FILENAME_RE.sub("_", name.strip())
    return safe[:max_len] if safe else "untitled"


class KnowledgeProjector:
    """Project change events from SQLite to an Obsidian-compatible vault.

    Maintains a cursor (last_projected_seq) so incremental projections
    are efficient. Full rebuild is also supported.
    """

    def __init__(
        self,
        event_log: EventLog,
        vault_dir: Path | str = _DEFAULT_VAULT_DIR,
    ):
        self._log = event_log
        self._vault = Path(vault_dir)
        self._vault.mkdir(parents=True, exist_ok=True)
        self._cursor_file = self._vault / ".projector_cursor"
        self._last_seq = self._load_cursor()

    def _load_cursor(self) -> int:
        try:
            return int(self._cursor_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            return 0

    def _save_cursor(self, seq: int) -> None:
        self._cursor_file.write_text(str(seq))
        self._last_seq = seq

    def project_incremental(self) -> int:
        """Replay new events since last cursor and project to vault.

        Returns number of events processed.
        """
        events = self._log.replay(since_seq=self._last_seq)
        if not events:
            return 0

        count = 0
        for evt in events:
            self._project_event(evt)
            count += 1

        new_seq = events[-1].seq
        self._save_cursor(new_seq)
        _log.event("projection_complete", events=count, cursor=new_seq)
        return count

    def project_full_rebuild(self) -> int:
        """Full rebuild: replay all events from seq=0."""
        self._save_cursor(0)
        return self.project_incremental()

    def _project_event(self, event: ChangeEvent) -> None:
        """Route an event to its entity-type handler."""
        handler = {
            "fact": self._project_fact,
            "evidence": self._project_evidence,
            "task": self._project_task,
            "session": self._project_session,
        }.get(event.entity_type)

        if handler:
            try:
                handler(event)
            except Exception as e:
                _log.warning(f"projection_failed: {event.entity_type}/{event.entity_id}: {e}")

    def _project_fact(self, event: ChangeEvent) -> None:
        facts_dir = self._vault / "facts"
        facts_dir.mkdir(exist_ok=True)

        if event.action == "delete":
            target = facts_dir / f"{_safe_filename(event.entity_id)}.md"
            target.unlink(missing_ok=True)
            return

        payload = event.payload or {}
        content = payload.get("text", payload.get("content", ""))
        source = payload.get("source", "")
        category = payload.get("category", "general")
        confidence = payload.get("confidence", "")

        lines = [
            f"# {event.entity_id}",
            "",
            f"**Category:** {category}",
        ]
        if source:
            lines.append(f"**Source:** {source}")
        if confidence:
            lines.append(f"**Confidence:** {confidence}")
        lines.extend(["", content, ""])

        filename = _safe_filename(event.entity_id)
        (facts_dir / f"{filename}.md").write_text("\n".join(lines), encoding="utf-8")

    def _project_evidence(self, event: ChangeEvent) -> None:
        evidence_dir = self._vault / "evidence"
        evidence_dir.mkdir(exist_ok=True)

        if event.action == "delete":
            target = evidence_dir / f"{_safe_filename(event.entity_id)}.md"
            target.unlink(missing_ok=True)
            return

        payload = event.payload or {}
        lines = [
            f"# Evidence: {event.entity_id}",
            "",
            f"**Query:** {payload.get('query', '')}",
            f"**Tool:** {payload.get('tool', '')}",
            f"**URLs:** {', '.join(payload.get('urls', []))}",
            "",
            payload.get("observation", ""),
            "",
        ]

        filename = _safe_filename(event.entity_id)
        (evidence_dir / f"{filename}.md").write_text("\n".join(lines), encoding="utf-8")

    def _project_task(self, event: ChangeEvent) -> None:
        tasks_dir = self._vault / "tasks"
        tasks_dir.mkdir(exist_ok=True)

        payload = event.payload or {}
        status = payload.get("status", "pending")
        lines = [
            f"# Task: {event.entity_id}",
            "",
            f"**Status:** {status}",
            f"**Goal:** {payload.get('goal', '')}",
            "",
        ]
        steps = payload.get("steps", [])
        if steps:
            lines.append("## Steps")
            for i, step in enumerate(steps, 1):
                check = "x" if step.get("status") == "done" else " "
                lines.append(f"- [{check}] {step.get('description', step.get('id', ''))}")
            lines.append("")

        filename = _safe_filename(event.entity_id)
        (tasks_dir / f"{filename}.md").write_text("\n".join(lines), encoding="utf-8")

    def _project_session(self, event: ChangeEvent) -> None:
        sessions_dir = self._vault / "sessions"
        sessions_dir.mkdir(exist_ok=True)

        payload = event.payload or {}
        lines = [
            f"# Session: {event.entity_id}",
            "",
            f"**State:** {payload.get('state', '')}",
            f"**Goal:** {payload.get('goal', '')}",
            f"**Progress:** {payload.get('progress', '')}",
            "",
        ]

        filename = _safe_filename(event.entity_id)
        (sessions_dir / f"{filename}.md").write_text("\n".join(lines), encoding="utf-8")


def project_pending_events(
    *,
    db_path: Path | str | None = None,
    vault_dir: Path | str = _DEFAULT_VAULT_DIR,
) -> int:
    """Project incremental knowledge events using a fresh EventLog handle."""
    elog = EventLog() if db_path is None else EventLog(db_path=db_path)
    try:
        projector = KnowledgeProjector(elog, vault_dir=vault_dir)
        return projector.project_incremental()
    finally:
        elog.close()
