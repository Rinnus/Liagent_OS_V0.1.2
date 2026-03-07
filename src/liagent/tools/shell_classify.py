"""Shell command classification — 3-tier allowlist with argument validation."""
from __future__ import annotations

from pathlib import Path

from ._path_security import get_cwork_root

# ── Path position specs ─────────────────────────────────────────────
# "all_positional" — every non-flag arg is a path
# "from_index_N"   — positional args starting at index N are paths
# "index_N"        — only positional arg at index N is a path
# "none"           — no positional args are paths

_SAFE_RULES: dict[str, dict] = {
    "ls":    {"path_positions": "all_positional"},
    "cat":   {"path_positions": "all_positional"},
    "head":  {"path_positions": "all_positional"},
    "tail":  {"path_positions": "all_positional"},
    "wc":    {"path_positions": "all_positional"},
    "diff":  {"path_positions": "all_positional"},
    "grep":  {"path_positions": "from_index_1"},
    "rg":    {"path_positions": "from_index_1",
              "blocked_flags": frozenset({"--pre"})},
    "sed":   {"path_positions": "from_index_1",
              "blocked_flags": frozenset({"-i", "--in-place"})},
    "find":  {"path_positions": "index_0",
              "blocked_flags": frozenset({"-exec", "-execdir", "-delete", "-ok"})},
    "pwd":   {"path_positions": "none"},
    "echo":  {"path_positions": "none"},
    "date":  {"path_positions": "none"},
    "which": {"path_positions": "none"},
}

_SAFE_COMPOUND: dict[tuple[str, str], dict] = {
    ("git", "status"):  {"path_positions": "none"},
    ("git", "log"):     {"path_positions": "none"},
    ("git", "diff"):    {"path_positions": "none"},
    ("git", "branch"):  {"path_positions": "none"},
    ("git", "show"):    {"path_positions": "none"},
    ("git", "rev-parse"): {"path_positions": "none"},
    ("git", "ls-files"): {"path_positions": "none"},
    ("pip", "list"):    {"path_positions": "none"},
    ("pip", "show"):    {"path_positions": "none"},
    ("npm", "list"):    {"path_positions": "none"},
    ("node", "--version"): {"path_positions": "none"},
    ("python", "--version"): {"path_positions": "none"},
    ("python3", "--version"): {"path_positions": "none"},
    ("uv", "--version"): {"path_positions": "none"},
    ("pnpm", "--version"): {"path_positions": "none"},
    ("yarn", "--version"): {"path_positions": "none"},
    ("poetry", "--version"): {"path_positions": "none"},
    ("ruff", "--version"): {"path_positions": "none"},
}

_DEV_COMPOUND: dict[tuple[str, str], dict] = {
    ("git", "add"):      {"path_positions": "all_positional"},
    ("git", "commit"):   {"path_positions": "none"},
    ("git", "fetch"):    {"path_positions": "none"},
    ("git", "push"):     {"path_positions": "none"},
    ("git", "pull"):     {"path_positions": "none"},
    ("git", "checkout"): {"path_positions": "all_positional"},
    ("git", "merge"):    {"path_positions": "none"},
    ("git", "rebase"):   {"path_positions": "none"},
    ("git", "stash"):    {"path_positions": "none"},
    ("pip", "install"):  {"path_positions": "none"},
    ("npm", "install"):  {"path_positions": "none"},
    ("npm", "ci"):       {"path_positions": "none"},
    ("npm", "run"):      {"path_positions": "none"},
    ("npm", "test"):     {"path_positions": "none"},
    ("pnpm", "install"): {"path_positions": "none"},
    ("pnpm", "add"):     {"path_positions": "none"},
    ("pnpm", "remove"):  {"path_positions": "none"},
    ("pnpm", "run"):     {"path_positions": "none"},
    ("yarn", "install"): {"path_positions": "none"},
    ("yarn", "add"):     {"path_positions": "none"},
    ("yarn", "remove"):  {"path_positions": "none"},
    ("yarn", "run"):     {"path_positions": "none"},
    ("poetry", "install"): {"path_positions": "none"},
    ("poetry", "add"):   {"path_positions": "none"},
    ("poetry", "remove"): {"path_positions": "none"},
    ("poetry", "run"):   {"path_positions": "none"},
    ("uv", "sync"):      {"path_positions": "none"},
    ("uv", "add"):       {"path_positions": "none"},
    ("uv", "remove"):    {"path_positions": "none"},
    ("uv", "run"):       {"path_positions": "none"},
    ("ruff", "check"):   {"path_positions": "none"},
    ("ruff", "format"):  {"path_positions": "none"},
}

# Safe filesystem operations — all paths validated inside cwork sandbox.
# Path validation (validate_path_arg) already enforces cwork boundary,
# so these are safe to auto-execute without confirmation.
_SAFE_CWORK: dict[str, dict] = {
    "mkdir":   {"path_positions": "all_positional",
                "blocked_flags": frozenset()},
    "cp":      {"path_positions": "all_positional",
                "blocked_flags": frozenset()},
    "mv":      {"path_positions": "all_positional",
                "blocked_flags": frozenset()},
    "touch":   {"path_positions": "all_positional",
                "blocked_flags": frozenset()},
}

_DEV_SIMPLE: dict[str, dict] = {
    "npx":     {"path_positions": "none"},
    "python":  {"path_positions": "all_positional",
                "blocked_flags": frozenset({"-c"})},
    "python3": {"path_positions": "all_positional",
                "blocked_flags": frozenset({"-c"})},
    "pytest":  {"path_positions": "all_positional"},
    "cargo":   {"path_positions": "none"},
    "make":    {"path_positions": "none"},
    "go":      {"path_positions": "none"},
}

_PRIVILEGED: frozenset[str] = frozenset({
    "sudo", "chmod", "chown", "rm", "kill", "killall", "pkill",
    "docker", "apt", "yum",
})

_PRIVILEGED_COMPOUND: frozenset[tuple[str, str]] = frozenset({
    ("brew", "install"),
})

_SCOPED_GRANT_COMMANDS: frozenset[str] = frozenset({
    "uv", "pnpm", "yarn", "poetry", "ruff",
})


def validate_path_arg(arg: str) -> tuple[bool, str]:
    """Validate a single path argument is inside cwork."""
    try:
        cwork_root = get_cwork_root()
        p = Path(arg).expanduser()
        if not p.is_absolute():
            p = cwork_root / p
        resolved = p.resolve(strict=False)

        if not resolved.is_relative_to(cwork_root):
            return False, f"Path escapes cwork: {resolved}"

        # Symlink escape check
        current = cwork_root
        for part in resolved.relative_to(cwork_root).parts:
            current = current / part
            if current.is_symlink():
                link_target = current.resolve()
                if not link_target.is_relative_to(cwork_root):
                    return False, f"Symlink escapes cwork: {current} -> {link_target}"

        return True, ""
    except (OSError, ValueError) as e:
        return False, f"Path validation error: {e}"


def _extract_positional_args(args: list[str]) -> list[str]:
    """Extract non-flag arguments (skip -x and --xxx)."""
    positional = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a.startswith("-"):
            if a in _FLAGS_WITH_VALUE:
                skip_next = True
            continue
        positional.append(a)
    return positional


def _first_scope_arg(cmd: str, args: list[str]) -> str | None:
    """Extract the scope token for commands whose grants should be narrowed."""
    if not args:
        return None

    if cmd in {"python", "python3"}:
        if args[0] == "-m" and len(args) > 1:
            return f"-m:{args[1]}"
        skip_next = False
        for a in args:
            if skip_next:
                skip_next = False
                continue
            if a.startswith("-"):
                if a in _FLAGS_WITH_VALUE:
                    skip_next = True
                continue
            return Path(a).as_posix()
        return None

    if cmd == "npx":
        skip_next = False
        for a in args:
            if skip_next:
                skip_next = False
                continue
            if a.startswith("-"):
                if a in _FLAGS_WITH_VALUE:
                    skip_next = True
                continue
            return a
        return None

    return None


def _looks_like_path(arg: str) -> bool:
    """Check if an argument looks like a filesystem path that needs validation."""
    return (
        arg.startswith("/")
        or arg.startswith("../")
        or arg.startswith("./")
        or arg.startswith("~")
        or arg == ".."
        or "/../" in arg
    )


def validate_argv(cmd: str, args: list[str], rule: dict | None = None) -> tuple[bool, str]:
    """Validate arguments according to command-specific rules."""
    if rule is None:
        rule = _SAFE_RULES.get(cmd, _DEV_SIMPLE.get(cmd, {}))

    blocked = rule.get("blocked_flags", frozenset())
    for a in args:
        if a in blocked:
            return False, f"Flag '{a}' is not allowed for '{cmd}'"

    pos_spec = rule.get("path_positions", "none")
    positional = _extract_positional_args(args)

    if pos_spec == "none":
        return True, ""

    if pos_spec == "all_positional":
        path_args = positional
    elif pos_spec.startswith("from_index_"):
        idx = int(pos_spec.split("_")[-1])
        path_args = positional[idx:]
    elif pos_spec.startswith("index_"):
        idx = int(pos_spec.split("_")[-1])
        path_args = [positional[idx]] if idx < len(positional) else []
    else:
        path_args = []

    for pa in path_args:
        ok, reason = validate_path_arg(pa)
        if not ok:
            return False, reason

    # Secondary scan: validate ALL non-flag args that look like paths,
    # including flag-value args (e.g. grep -f /etc/passwd).
    # This catches paths hidden as flag values that _extract_positional_args skips.
    validated = set(path_args)
    for a in args:
        if not a.startswith("-") and a not in validated and _looks_like_path(a):
            ok, reason = validate_path_arg(a)
            if not ok:
                return False, f"Flag-value path escape: {reason}"

    # Tertiary scan: extract values from --flag=value format
    # e.g. --file=/etc/passwd → validate /etc/passwd
    for a in args:
        if a.startswith("-") and "=" in a:
            val = a.split("=", 1)[1]
            if val and val not in validated and _looks_like_path(val):
                ok, reason = validate_path_arg(val)
                if not ok:
                    return False, f"Flag-value path escape: {reason}"

    return True, ""


def classify_command(argv: list[str]) -> tuple[str, str]:
    """Classify a command into (tier, reason).

    Returns:
        tier: "safe" | "dev" | "privileged" | "denied"
        reason: human-readable explanation
    """
    if not argv:
        return "denied", "Empty command"

    cmd = argv[0]
    sub = argv[1] if len(argv) > 1 else ""
    args = argv[1:]
    compound_args = argv[2:]

    # 1. Check compound commands first (cmd + subcmd)
    compound_key = (cmd, sub)

    if compound_key in _SAFE_COMPOUND:
        rule = _SAFE_COMPOUND[compound_key]
        ok, reason = validate_argv(cmd, compound_args, rule)
        if not ok:
            return "denied", reason
        return "safe", f"{cmd} {sub}"

    if compound_key in _DEV_COMPOUND:
        rule = _DEV_COMPOUND[compound_key]
        ok, reason = validate_argv(cmd, compound_args, rule)
        if not ok:
            return "denied", reason
        return "dev", f"{cmd} {sub}"

    if compound_key in _PRIVILEGED_COMPOUND:
        return "privileged", f"{cmd} {sub}"

    # 2. Check simple commands (cmd only)
    if cmd in _SAFE_RULES:
        rule = _SAFE_RULES[cmd]
        ok, reason = validate_argv(cmd, args, rule)
        if not ok:
            return "denied", reason
        return "safe", cmd

    # 2b. Safe cwork filesystem ops — safe IF all paths validate inside cwork
    if cmd in _SAFE_CWORK:
        rule = _SAFE_CWORK[cmd]
        ok, reason = validate_argv(cmd, args, rule)
        if not ok:
            # Path escapes cwork → hard deny (no escape allowed, even with confirmation)
            return "denied", reason
        return "safe", cmd

    if cmd in _DEV_SIMPLE:
        rule = _DEV_SIMPLE[cmd]
        ok, reason = validate_argv(cmd, args, rule)
        if not ok:
            return "denied", reason
        return "dev", cmd

    if cmd in _PRIVILEGED:
        return "privileged", cmd

    # 3. Unrecognized — default deny
    return "denied", f"Command '{cmd}' not in allowlist"


def grant_key(argv: list[str]) -> str:
    """Generate session grant key for a dev-tier command."""
    if not argv:
        return "shell_exec:dev:unknown"
    scoped_simple = _first_scope_arg(argv[0], argv[1:])
    if scoped_simple:
        return f"shell_exec:dev:{argv[0]}:{scoped_simple}"
    if len(argv) > 1 and argv[0] in _SCOPED_GRANT_COMMANDS and not argv[1].startswith("-"):
        return f"shell_exec:dev:{argv[0]}:{argv[1]}"
    return f"shell_exec:dev:{argv[0]}"


def grant_scope_label(argv: list[str]) -> str:
    """Human-readable grant scope shown in confirmation messages."""
    if not argv:
        return "unknown"
    scoped_simple = _first_scope_arg(argv[0], argv[1:])
    if scoped_simple:
        if scoped_simple.startswith("-m:"):
            return f"{argv[0]} -m {scoped_simple.split(':', 1)[1]}"
        return f"{argv[0]} {scoped_simple}"
    if len(argv) > 1 and argv[0] in _SCOPED_GRANT_COMMANDS and not argv[1].startswith("-"):
        return f"{argv[0]} {argv[1]}"
    return argv[0]
# Flags that consume the next argv token.
_FLAGS_WITH_VALUE: frozenset[str] = frozenset({
    "-n", "-m", "-c", "-f", "-o", "-e", "--name",
    "--message", "--file", "--output", "-p", "--package",
})
