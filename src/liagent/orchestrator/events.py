"""Unified event protocol for multi-agent systems."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentEvent:
    type: str
    payload: Any
    source: str       # "brain", "lead", "sub:0", "sub:1"
    run_id: str
    agent_id: str
    seq: int
    timestamp: float

    def to_legacy_tuple(self) -> tuple:
        """Adapt to existing (type, ...) protocol used by cli.py and web_server.py.

        Passthrough events store the tuple tail as payload (tuple of remaining
        elements).  Reconstruct the original (type, el1, el2, ...) form.
        Research-mode events store dicts which are unpacked by type.
        """
        # Passthrough path: payload is a tuple of remaining elements
        if isinstance(self.payload, tuple):
            return (self.type, *self.payload)
        # Research-mode events with dict payloads
        if self.type == "tool_start" and isinstance(self.payload, dict):
            return ("tool_start", self.payload.get("name", ""), self.payload.get("args", {}))
        if self.type == "tool_result" and isinstance(self.payload, dict):
            return ("tool_result", self.payload.get("name", ""), self.payload.get("result", ""))
        if self.type == "confirmation_required" and isinstance(self.payload, dict):
            p = self.payload
            return ("confirmation_required", p.get("token"), p.get("tool"), p.get("reason"), p.get("brief"))
        return (self.type, self.payload)


class EventSequencer:
    """Monotonic sequence generator (async-safe, single-thread)."""
    def __init__(self, start: int = 0):
        self._seq = start

    def next(self) -> int:
        val = self._seq
        self._seq += 1
        return val


def make_event(
    type: str, payload: Any, *, source: str, run_id: str, agent_id: str,
    sequencer: EventSequencer,
) -> AgentEvent:
    return AgentEvent(
        type=type, payload=payload, source=source,
        run_id=run_id, agent_id=agent_id,
        seq=sequencer.next(), timestamp=time.time(),
    )
