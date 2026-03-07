"""Tool Trust Registry — tracks trust status of MCP servers and tools.

Storage: JSON file with atomic write (tempfile + os.replace).
Single instance per process — callers should NOT create multiple instances
for the same store_path to avoid write conflicts.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, get_args

_log = logging.getLogger(__name__)

TrustStatus = Literal["unknown", "approved", "revoked"]
_VALID_STATUSES = set(get_args(TrustStatus))

_DEFAULT_STORE_PATH = Path.home() / ".liagent" / "tool_trust.json"


class TrustRegistry:
    """Manages trust state for MCP servers.

    Each entry: ``{server_id: {status, source, updated_at}}``.
    """

    def __init__(self, store_path: Path | None = None):
        self._path = store_path or _DEFAULT_STORE_PATH
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    self._data = {}
                else:
                    # Filter out malformed entries (non-dict values)
                    self._data = {k: v for k, v in raw.items() if isinstance(v, dict)}
            except Exception:
                _log.warning("trust_registry: failed to load %s, starting fresh", self._path)
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        """Atomic write: write to temp file in same directory, then os.replace."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._path.parent), suffix=".tmp", prefix=".trust_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, str(self._path))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            _log.error("trust_registry: failed to save: %s", e)
            raise

    def get_status(self, server_id: str) -> TrustStatus:
        entry = self._data.get(server_id)
        if not isinstance(entry, dict):
            return "unknown"
        status = entry.get("status", "unknown")
        return status if status in _VALID_STATUSES else "unknown"

    def get_entry(self, server_id: str) -> dict | None:
        entry = self._data.get(server_id)
        if not isinstance(entry, dict):
            return None
        return dict(entry)

    def set_status(self, server_id: str, status: TrustStatus, *, source: str = ""):
        if status not in _VALID_STATUSES:
            raise ValueError(f"Invalid trust status: {status!r}. Must be one of {_VALID_STATUSES}")
        self._data[server_id] = {
            "status": status,
            "source": source,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def ensure_registered(self, server_id: str, *, source: str = "discovered"):
        """Register a server as unknown if not already tracked. Does NOT overwrite existing entries."""
        if server_id not in self._data:
            self.set_status(server_id, "unknown", source=source)

    def list_by_status(self, status: TrustStatus) -> list[str]:
        return [
            sid for sid, entry in self._data.items()
            if isinstance(entry, dict) and entry.get("status") == status
        ]

    def is_connectable(self, server_id: str) -> bool:
        """Return True only if the server is approved. Unknown and revoked are NOT connectable."""
        return self.get_status(server_id) == "approved"
