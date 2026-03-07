"""Notification channel router — reliable dispatch with dedup, rate-limit, and fallback.

Provides a degradation chain: WebSocket -> CLI -> SystemNotify.
Each channel implements the NotificationChannel protocol.
The ChannelRouter tries channels in order, falling back on failure.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Protocol, runtime_checkable

from ..logging import get_logger

_log = get_logger("notification")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class NotificationChannel(Protocol):
    """Protocol for notification channels."""

    @property
    def channel_type(self) -> str: ...

    async def send(self, message: str, *, priority: str = "normal") -> bool: ...


# ---------------------------------------------------------------------------
# Concrete channels
# ---------------------------------------------------------------------------

class CLIPushChannel:
    """Prints notifications to stdout (optionally via Rich console)."""

    @property
    def channel_type(self) -> str:
        return "cli"

    async def send(self, message: str, *, priority: str = "normal") -> bool:
        try:
            from rich.console import Console
            console = Console()
            prefix = "[bold yellow]Notification[/bold yellow]"
            if priority == "urgent":
                prefix = "[bold red]URGENT[/bold red]"
            console.print(f"{prefix}: {message}")
        except ImportError:
            print(f"[{priority.upper()}] {message}")
        return True


class SystemNotifyChannel:
    """Uses macOS osascript for native desktop notifications."""

    @property
    def channel_type(self) -> str:
        return "system_notify"

    async def send(self, message: str, *, priority: str = "normal") -> bool:
        try:
            escaped = message.replace('"', '\\"')
            cmd = [
                "osascript", "-e",
                f'display notification "{escaped}" with title "LiAgent"',
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return proc.returncode == 0
        except Exception as exc:
            _log.error("system_notify", exc, channel="system_notify")
            return False


class WebSocketChannel:
    """Sends via an active WebSocket connection. Returns False if no connection."""

    def __init__(self, ws=None):
        self._ws = ws

    @property
    def channel_type(self) -> str:
        return "websocket"

    async def send(self, message: str, *, priority: str = "normal") -> bool:
        if self._ws is None:
            return False
        try:
            import json
            payload = json.dumps({
                "type": "notification",
                "message": message,
                "priority": priority,
            })
            await self._ws.send(payload)
            return True
        except Exception as exc:
            _log.error("websocket_send", exc, channel="websocket")
            return False


# ---------------------------------------------------------------------------
# Channel Router
# ---------------------------------------------------------------------------

class ChannelRouter:
    """Dispatches notifications through an ordered list of channels with
    content-hash dedup, sliding-window rate limiting, and automatic fallback.

    Parameters
    ----------
    channels : list
        Ordered list of notification channels (highest priority first).
    dedup_window_sec : float
        Seconds within which identical messages are suppressed.  Default 300.
    rate_limit_per_min : int
        Maximum sends allowed per 60-second sliding window.  Default 10.
    """

    def __init__(
        self,
        channels: list | None = None,
        *,
        dedup_window_sec: float = 300,
        rate_limit_per_min: int = 10,
    ):
        self._channels: list = channels or []
        self._dedup_window_sec = dedup_window_sec
        self._rate_limit_per_min = rate_limit_per_min

        # hash -> timestamp of last send
        self._dedup_cache: dict[str, float] = {}
        # timestamps of successful sends (ascending)
        self._send_timestamps: list[float] = []

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _hash_message(message: str) -> str:
        return hashlib.sha256(message.encode("utf-8")).hexdigest()

    def _is_duplicate(self, msg_hash: str, now: float) -> bool:
        """Return True if the same hash was seen within the dedup window."""
        ts = self._dedup_cache.get(msg_hash)
        if ts is not None and (now - ts) < self._dedup_window_sec:
            return True
        return False

    def _is_rate_limited(self, now: float) -> bool:
        """Return True if we've exceeded the per-minute rate limit."""
        window_start = now - 60.0
        # Prune old entries
        self._send_timestamps = [
            t for t in self._send_timestamps if t > window_start
        ]
        return len(self._send_timestamps) >= self._rate_limit_per_min

    def _clean_dedup_cache(self, now: float) -> None:
        """Remove expired dedup entries."""
        expired = [
            h for h, ts in self._dedup_cache.items()
            if (now - ts) >= self._dedup_window_sec
        ]
        for h in expired:
            del self._dedup_cache[h]

    # ---- public API -------------------------------------------------------

    async def dispatch(self, message: str, *, priority: str = "normal") -> bool:
        """Try to send *message* through the channel chain.

        Returns True if the message was delivered (or dedup-suppressed).
        Returns False if all channels failed or rate-limited.
        """
        if not self._channels:
            return False

        now = time.monotonic()

        # --- dedup check ---
        msg_hash = self._hash_message(message)
        if self._is_duplicate(msg_hash, now):
            _log.event("notification_dedup", hash=msg_hash[:12])
            return True  # already sent recently

        # --- rate-limit check ---
        if self._is_rate_limited(now):
            _log.warning("notification_rate_limited", pending=message[:80])
            return False

        # --- try channels in order ---
        for ch in self._channels:
            try:
                ok = await ch.send(message, priority=priority)
                if ok:
                    # Record successful send
                    self._dedup_cache[msg_hash] = now
                    self._send_timestamps.append(now)
                    self._clean_dedup_cache(now)
                    _log.event(
                        "notification_sent",
                        channel=getattr(ch, "channel_type", "unknown"),
                        priority=priority,
                    )
                    return True
            except Exception as exc:
                _log.error(
                    "notification_channel_error",
                    exc,
                    channel=getattr(ch, "channel_type", "unknown"),
                )
                # Fall through to next channel

        _log.warning("notification_all_channels_failed", message=message[:80])
        return False
