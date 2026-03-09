"""Shared run-control primitives for cancellation-aware execution."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class RunCancellationScope:
    """Cooperative cancellation scope shared across a single foreground run."""

    requested: bool = False
    reason: str = ""
    requested_at: float = 0.0
    _event: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _children: list["RunCancellationScope"] = field(default_factory=list, init=False, repr=False)

    def cancel(self, reason: str = "cancelled") -> None:
        if self.requested:
            return
        self.requested = True
        self.reason = str(reason or "cancelled")
        self.requested_at = time.time()
        self._event.set()
        for child in list(self._children):
            child.cancel(self.reason)

    @property
    def cancelled(self) -> bool:
        return bool(self.requested)

    async def wait_requested(self) -> str:
        if not self.requested:
            await self._event.wait()
        return self.reason or "cancelled"

    def child(self) -> "RunCancellationScope":
        child = RunCancellationScope()
        self._children.append(child)
        if self.requested:
            child.cancel(self.reason or "cancelled")
        return child

    def raise_if_cancelled(self) -> None:
        if self.requested:
            raise asyncio.CancelledError(self.reason or "cancelled")
