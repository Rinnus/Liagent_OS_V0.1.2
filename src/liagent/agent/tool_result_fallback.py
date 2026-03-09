"""Shared formatting for deterministic tool-result fallback answers."""

from __future__ import annotations

from .quality import quality_fix
from .text_utils import clean_output


def _truncate_inline(text: str, *, limit: int = 120) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _context_lines(tool_name: str, tool_args: dict | None) -> list[str]:
    args = tool_args or {}
    lines: list[str] = []
    if tool_name in {"list_dir", "read_file", "run_tests"}:
        path = str(args.get("path", "") or "").strip()
        if path:
            lines.append(f"Target path: `{_truncate_inline(path)}`")
    if tool_name == "shell_exec":
        command = str(args.get("command", "") or "").strip()
        if command:
            lines.append(f"Command: `{_truncate_inline(command)}`")
    if tool_name == "python_exec":
        code = str(args.get("code", "") or "").strip()
        if code:
            first_line = code.splitlines()[0].strip()
            if first_line:
                lines.append(f"Code entry: `{_truncate_inline(first_line)}`")
    if tool_name == "web_search":
        query = str(args.get("query", "") or "").strip()
        if query:
            lines.append(f"Query: `{_truncate_inline(query)}`")
    if tool_name in {"web_fetch", "browser"}:
        url = str(args.get("url", "") or "").strip()
        if url:
            lines.append(f"Target URL: `{_truncate_inline(url)}`")
    return lines


def _prefix(tool_name: str, *, confirmed: bool) -> str:
    if confirmed:
        return f"Confirmed and executed `{tool_name}`."
    return f"Executed `{tool_name}`."


def _multi_prefix(*, confirmed: bool) -> str:
    if confirmed:
        return "Recent tool calls were confirmed and executed."
    return "Recent tool results were received."


def _tool_display_name(item: dict) -> str:
    requested = str(item.get("requested_tool_name") or item.get("tool_name") or "").strip()
    effective = str(item.get("effective_tool_name") or item.get("tool_name") or "").strip()
    if requested and effective and requested != effective:
        return f"`{requested}` -> `{effective}`"
    return f"`{effective or requested or 'unknown'}`"


def _tool_context_lines(item: dict) -> list[str]:
    requested = str(item.get("requested_tool_name") or item.get("tool_name") or "").strip()
    effective = str(item.get("effective_tool_name") or item.get("tool_name") or "").strip()
    requested_args = item.get("requested_tool_args") if isinstance(item.get("requested_tool_args"), dict) else {}
    effective_args = item.get("effective_tool_args") if isinstance(item.get("effective_tool_args"), dict) else {}
    lines: list[str] = []
    if requested and effective and requested != effective:
        lines.append(f"Requested tool: `{requested}`")
        lines.extend(_context_lines(requested, requested_args))
        lines.append(f"Executed tool: `{effective}`")
        lines.extend(_context_lines(effective, effective_args))
        return lines
    lines.extend(_context_lines(effective or requested, effective_args or requested_args))
    return lines


def _observation_preview(tool_name: str, observation: str) -> str:
    obs = str(observation or "").strip()
    if not obs:
        return "[empty result]"
    lines = [ln.rstrip() for ln in obs.splitlines() if ln.strip()]
    if tool_name == "list_dir":
        if len(lines) <= 1:
            return lines[0] if lines else "[empty result]"
        preview_lines = lines[:8]
        text = "\n".join(preview_lines)
        if len(lines) > len(preview_lines):
            text += "\n...(truncated)"
        return text
    preview = "\n".join(lines[:12]) if lines else obs
    if preview != obs:
        preview += "\n...(truncated)"
    return preview


def _format_multi_tool_results(
    tool_contexts: list[dict],
    *,
    confirmed: bool,
) -> str:
    parts = [_multi_prefix(confirmed=confirmed)]
    for idx, item in enumerate(tool_contexts[:4], start=1):
        tool_name = str(item.get("effective_tool_name") or item.get("tool_name") or "unknown")
        status = str(item.get("status") or "result")
        reason = str(item.get("reason") or "").strip()
        parts.append("")
        if status == "skip":
            parts.append(f"{idx}. {_tool_display_name(item)} (skipped)")
            parts.extend(_tool_context_lines(item))
            if reason:
                parts.append(f"Skip reason: {reason}")
            continue
        if status == "error":
            parts.append(f"{idx}. {_tool_display_name(item)} (error)")
            parts.extend(_tool_context_lines(item))
            if reason:
                parts.append(f"Error type: {reason}")
            parts.append(_observation_preview(tool_name, str(item.get("observation") or "")))
            continue
        parts.append(f"{idx}. {_tool_display_name(item)}")
        parts.extend(_tool_context_lines(item))
        parts.append(_observation_preview(tool_name, str(item.get("observation") or "")))
    return "\n".join(parts)


def _format_list_dir(tool_name: str, observation: str, *, confirmed: bool, tool_args: dict | None) -> str:
    lines = [ln.rstrip() for ln in str(observation or "").splitlines() if ln.strip()]
    if not lines:
        return f"{_prefix(tool_name, confirmed=confirmed)} No directory contents were available to display."
    header = lines[0].strip()
    entries = [ln.strip() for ln in lines[1:]]
    parts = [_prefix(tool_name, confirmed=confirmed)]
    parts.extend(_context_lines(tool_name, tool_args))
    if not entries:
        parts.append(f"Directory `{header}` is currently empty.")
        return "\n".join(parts)
    shown = entries[:20]
    parts.append(f"Directory `{header}` currently shows {len(entries)} entries:")
    parts.extend(f"- {entry}" for entry in shown)
    if len(entries) > len(shown):
        parts.append(f"- ... plus {len(entries) - len(shown)} more")
    return "\n".join(parts)


def _format_read_file(tool_name: str, observation: str, *, confirmed: bool, tool_args: dict | None) -> str:
    obs = str(observation or "").strip()
    if not obs:
        return f"{_prefix(tool_name, confirmed=confirmed)} No readable content was available to display."
    lines = obs.splitlines()
    preview = "\n".join(lines[:40]).strip()
    if preview != obs:
        preview += "\n...(truncated)"
    parts = [_prefix(tool_name, confirmed=confirmed)]
    parts.extend(_context_lines(tool_name, tool_args))
    if preview.startswith("[Page "):
        parts.append("Excerpt from the current page:")
    else:
        parts.append("Excerpt from the file:")
    parts.append(preview)
    return "\n".join(parts)


def _format_run_tests(tool_name: str, observation: str, *, confirmed: bool, tool_args: dict | None) -> str:
    obs = str(observation or "").strip()
    if not obs:
        return f"{_prefix(tool_name, confirmed=confirmed)} No test output was available to display."
    lines = [ln.strip() for ln in obs.splitlines() if ln.strip()]
    summary = next(
        (
            ln for ln in reversed(lines)
            if any(marker in ln for marker in (" passed", " failed", " error", " errors", " skipped", " xfailed"))
        ),
        lines[-1] if lines else obs,
    )
    tail = "\n".join(lines[-12:]) if lines else obs
    parts = [_prefix(tool_name, confirmed=confirmed)]
    parts.extend(_context_lines(tool_name, tool_args))
    parts.append(f"Result summary: {summary}")
    parts.append("")
    parts.append("Recent output:")
    parts.append(tail)
    return "\n".join(parts)


def _format_shell_exec(tool_name: str, observation: str, *, confirmed: bool, tool_args: dict | None) -> str:
    obs = str(observation or "").strip()
    if not obs:
        return f"{_prefix(tool_name, confirmed=confirmed)} No command output was available to display."
    lines = obs.splitlines()
    preview = "\n".join(lines[:30]).strip()
    if preview != obs:
        preview += "\n...(truncated)"
    parts = [_prefix(tool_name, confirmed=confirmed)]
    parts.extend(_context_lines(tool_name, tool_args))
    parts.append("Recent command output:")
    parts.append(preview)
    return "\n".join(parts)


def _format_python_exec(tool_name: str, observation: str, *, confirmed: bool, tool_args: dict | None) -> str:
    obs = str(observation or "").strip()
    if not obs:
        return f"{_prefix(tool_name, confirmed=confirmed)} No Python output was available to display."
    lines = obs.splitlines()
    preview = "\n".join(lines[:30]).strip()
    if preview != obs:
        preview += "\n...(truncated)"
    parts = [_prefix(tool_name, confirmed=confirmed)]
    parts.extend(_context_lines(tool_name, tool_args))
    parts.append("Python output:")
    parts.append(preview)
    return "\n".join(parts)


def format_tool_result_fallback(
    *,
    tool_name: str,
    observation: str,
    tool_args: dict | None = None,
    tool_contexts: list[dict] | None = None,
    execution_ok: bool = True,
    confirmed: bool = False,
    reason: str,
) -> tuple[str, dict]:
    obs = str(observation or "").strip()
    contexts = [
        {
            "tool_name": str(item.get("tool_name") or ""),
            "requested_tool_name": str(item.get("requested_tool_name") or item.get("tool_name") or ""),
            "requested_tool_args": dict(item.get("requested_tool_args") or item.get("tool_args") or {}),
            "effective_tool_name": str(item.get("effective_tool_name") or item.get("tool_name") or ""),
            "effective_tool_args": dict(item.get("effective_tool_args") or item.get("tool_args") or {}),
            "observation": str(item.get("observation") or ""),
            "tool_args": dict(item.get("tool_args") or {}),
            "status": str(item.get("status") or "result"),
            "reason": str(item.get("reason") or ""),
        }
        for item in (tool_contexts or [])
        if str(item.get("tool_name") or item.get("effective_tool_name") or "").strip()
        and (
            str(item.get("observation") or "").strip()
            or str(item.get("reason") or "").strip()
        )
    ]

    if len(contexts) > 1:
        answer = _format_multi_tool_results(contexts, confirmed=confirmed)
    elif contexts:
        tool_name = contexts[0]["effective_tool_name"] or contexts[0]["tool_name"]
        obs = contexts[0]["observation"]
        tool_args = contexts[0]["effective_tool_args"] or contexts[0]["tool_args"]

    if len(contexts) <= 1:
        if tool_name == "list_dir":
            answer = _format_list_dir(tool_name, obs, confirmed=confirmed, tool_args=tool_args)
        elif tool_name == "system_status":
            parts = [_prefix(tool_name, confirmed=confirmed)]
            parts.extend(_context_lines(tool_name, tool_args))
            parts.append("Current system status:")
            parts.append(obs or "[empty result]")
            answer = "\n".join(parts)
        elif tool_name == "read_file":
            answer = _format_read_file(tool_name, obs, confirmed=confirmed, tool_args=tool_args)
        elif tool_name == "run_tests":
            answer = _format_run_tests(tool_name, obs, confirmed=confirmed, tool_args=tool_args)
        elif tool_name == "shell_exec":
            answer = _format_shell_exec(tool_name, obs, confirmed=confirmed, tool_args=tool_args)
        elif tool_name == "python_exec":
            answer = _format_python_exec(tool_name, obs, confirmed=confirmed, tool_args=tool_args)
        elif tool_name in {"web_search", "web_fetch", "browser"}:
            parts = [_prefix(tool_name, confirmed=confirmed)]
            parts.extend(_context_lines(tool_name, tool_args))
            parts.append("Available result:")
            parts.append(obs or "[empty result]")
            answer = "\n".join(parts)
        elif not execution_ok:
            parts = [_prefix(tool_name, confirmed=confirmed)]
            parts.extend(_context_lines(tool_name, tool_args))
            parts.append("Tool returned an error result:")
            parts.append(obs or "[empty result]")
            answer = "\n".join(parts)
        elif obs.startswith("["):
            parts = [_prefix(tool_name, confirmed=confirmed)]
            parts.extend(_context_lines(tool_name, tool_args))
            parts.append("Tool returned the following status:")
            parts.append(obs)
            answer = "\n".join(parts)
        else:
            parts = [_prefix(tool_name, confirmed=confirmed)]
            parts.extend(_context_lines(tool_name, tool_args))
            parts.append("Most recent tool result:")
            parts.append(obs or "[empty result]")
            answer = "\n".join(parts)

    answer = clean_output(answer)
    answer, qmeta = quality_fix(answer)
    issues = list(qmeta.get("issues", []))
    issues.append(f"tool_result_fallback:{reason}")
    qmeta["issues"] = issues
    return answer, qmeta
