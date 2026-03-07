"""Shared path security validation for filesystem tools."""

import os
from pathlib import Path


def get_cwork_root() -> Path:
    """Return the effective cwork root for filesystem tools."""
    raw = (
        os.environ.get("LIAGENT_CWORK_DIR")
        or os.environ.get("LIAGENT_CWORK_ROOT")
        or str(Path.home() / "Desktop" / "cwork")
    )
    return Path(raw).expanduser().resolve()


def get_cwork_root_str() -> str:
    return str(get_cwork_root())


def get_cwork_max_depth() -> int:
    raw = str(os.environ.get("LIAGENT_CWORK_MAX_DEPTH", "5") or "5").strip()
    try:
        return max(1, int(raw))
    except Exception:
        return 5


def allow_hidden_cwork_paths() -> bool:
    return str(os.environ.get("LIAGENT_CWORK_ALLOW_HIDDEN", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


_CWORK_ROOT = get_cwork_root()


def _normalize_path(path_str: str) -> str:
    """Auto-complete relative / shorthand paths to absolute cwork paths."""
    cwork_root = get_cwork_root()
    p = path_str.strip()
    # Already absolute and under cwork — pass through
    if p.startswith(str(cwork_root)):
        return p
    # Common user/model typos for the absolute cwork path
    # e.g. "/users/desktop/cwork", "/User/desktop/cwork"
    lowered = p.lower().rstrip("/")
    _CWORK_LOWER = str(cwork_root).lower()
    if lowered == _CWORK_LOWER or lowered.startswith(_CWORK_LOWER + "/"):
        # Case-insensitive match — map to real path
        tail = p[len(str(cwork_root)):] if len(p) > len(str(cwork_root)) else ""
        return str(cwork_root) + tail
    # Strip leading "cwork/" or "cwork" prefix the model often produces
    stripped = p
    for prefix in ("cwork/", "cwork\\", "cwork"):
        if stripped.lower().startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    # If path is absolute but not under cwork, don't rewrite — let validation reject
    if stripped.startswith("/") and stripped != p:
        return p
    # Treat remainder as relative to cwork root
    return str(cwork_root / stripped) if stripped else str(cwork_root)


def _validate_cwork_path(path_str: str) -> tuple[bool, str, Path | None]:
    """Validate that *path_str* resolves inside the cwork sandbox.

    Returns (ok, reason, resolved_path).
    Relative and shorthand paths (e.g. "cwork", "cwork/foo.txt", "foo.txt")
    are auto-completed to absolute paths under _CWORK_ROOT.
    """
    if not path_str or not path_str.strip():
        return False, "path is required", None

    normalized = _normalize_path(path_str)
    cwork_root = get_cwork_root()
    try:
        resolved = Path(normalized).resolve()
    except (OSError, ValueError) as exc:
        return False, f"invalid path: {exc}", None

    # Must be under cwork root
    try:
        resolved.relative_to(cwork_root)
    except ValueError:
        return False, f"path must be under {cwork_root}", None

    # Symlink escape: every parent must also resolve inside cwork
    for parent in resolved.parents:
        if parent == cwork_root or parent == cwork_root.parent:
            break
        try:
            parent.resolve().relative_to(cwork_root)
        except ValueError:
            return False, "symlink escape detected", None

    # Depth limit
    rel = resolved.relative_to(cwork_root)
    max_depth = get_cwork_max_depth()
    if len(rel.parts) > max_depth:
        return False, f"path too deep (max {max_depth} levels)", None

    # No hidden files/directories
    if not allow_hidden_cwork_paths():
        for part in rel.parts:
            if part.startswith("."):
                return False, f"hidden file/directory not allowed: {part}", None

    return True, "ok", resolved
