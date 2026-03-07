"""Shared local-timezone helpers for the agent package."""

from datetime import datetime
from zoneinfo import ZoneInfo

# Detect system local timezone, fall back to US Pacific.
try:
    _LOCAL_TZ = datetime.now().astimezone().tzinfo
except Exception:
    _LOCAL_TZ = ZoneInfo("America/Los_Angeles")


def _now_local() -> datetime:
    """Current time in the system local timezone."""
    return datetime.now(tz=_LOCAL_TZ)


def _now_local_iso() -> str:
    """Current local time as ISO 8601 string with timezone offset."""
    return _now_local().isoformat()
