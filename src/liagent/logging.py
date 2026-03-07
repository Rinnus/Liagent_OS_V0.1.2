"""Structured logging for LiAgent — JSON event logging with component context."""

import json
import logging
import logging.handlers
import os
import time
from pathlib import Path

# Configurable log directory — override with LIAGENT_LOG_DIR env var.
# File logging is disabled when LIAGENT_LOG_DISABLE=1 (useful for CI/sandbox).
_LOG_DISABLED = os.environ.get("LIAGENT_LOG_DISABLE", "").strip().lower() in {"1", "true", "yes"}

_logger = logging.getLogger("liagent")
_logger.setLevel(logging.DEBUG)
_logger.propagate = False  # prevent duplicate output via root logger

# File handler: lazily initialized on first emit to avoid import-time side effects.
_file_handler: logging.FileHandler | None = None
_file_handler_failed = False


def _ensure_file_handler():
    """Lazily create the file handler on first use."""
    global _file_handler, _file_handler_failed
    if _file_handler is not None or _file_handler_failed or _LOG_DISABLED:
        return
    try:
        log_dir = Path(os.environ.get("LIAGENT_LOG_DIR", "").strip() or (Path.home() / ".liagent" / "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "liagent.jsonl"
        max_bytes = int(os.environ.get("LIAGENT_LOG_MAX_MB", "10")) * 1024 * 1024
        backup_count = int(os.environ.get("LIAGENT_LOG_BACKUPS", "3"))
        _file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8",
        )
        _file_handler.setLevel(logging.DEBUG)
        _logger.addHandler(_file_handler)
    except Exception:
        _file_handler_failed = True  # don't retry on every log call


# Console handler: only warnings+
_console = logging.StreamHandler()
_console.setLevel(logging.WARNING)
_console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
_logger.addHandler(_console)


class StructuredLogger:
    """Structured event logger that outputs JSON lines."""

    def __init__(self, component: str = ""):
        self.component = component

    def _emit(self, level: str, event_type: str, **kwargs):
        _ensure_file_handler()
        record = {
            "ts": time.time(),
            "level": level,
            "component": self.component,
            "event": event_type,
            **kwargs,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        if level == "error":
            _logger.error(line)
        elif level == "warning":
            _logger.warning(line)
        elif level == "debug":
            _logger.debug(line)
        else:
            _logger.info(line)

    def event(self, event_type: str, **kwargs):
        """Log a generic event."""
        self._emit("info", event_type, **kwargs)

    def tool_call(self, tool_name: str, args: dict, status: str, duration_ms: float):
        """Log a tool call with timing."""
        self._emit("info", "tool_call",
                    tool=tool_name, args_preview=str(args)[:200],
                    status=status, duration_ms=round(duration_ms, 1))

    def llm_call(self, model: str, tokens_in: int, tokens_out: int, duration_ms: float):
        """Log an LLM call with token counts."""
        self._emit("info", "llm_call",
                    model=model, tokens_in=tokens_in, tokens_out=tokens_out,
                    duration_ms=round(duration_ms, 1))

    def error(self, component: str, error: Exception | str, **context):
        """Log an error with context (replaces silent `except: pass`)."""
        err_str = str(error) if isinstance(error, Exception) else error
        err_type = type(error).__name__ if isinstance(error, Exception) else "str"
        self._emit("error", "error",
                    error_component=component, error_type=err_type,
                    error_msg=err_str[:500], **context)

    def warning(self, msg: str, **context):
        """Log a warning."""
        self._emit("warning", "warning", msg=msg, **context)

    def trace(self, category: str, **kwargs):
        """Log a real-time agent behavior trace event."""
        self._emit("info", f"trace.{category}", **kwargs)


# Module-level singleton
logger = StructuredLogger("liagent")


def get_logger(component: str) -> StructuredLogger:
    """Get a logger for a specific component."""
    return StructuredLogger(component)
