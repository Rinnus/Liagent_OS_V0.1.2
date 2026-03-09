"""Tool execution — timeout, retry, degradation, and observation handling."""

import asyncio
import json
from contextlib import contextmanager


def build_tool_degrade_observation(tool_name: str, tool_args: dict, err_text: str) -> str:
    """Build a deterministic fallback observation when tool execution fails."""
    if tool_name == "web_search":
        query = str((tool_args or {}).get("query", "")).strip()
        return (
            f"[Tool degraded] web_search unavailable: {err_text}\n"
            f"query: {query or '(empty)'}\n"
            "Check search arguments and retry, or try different keywords."
        )
    if tool_name == "screenshot":
        return (
            f"[Tool degraded] screenshot unavailable: {err_text}\n"
            "Ask the user for an additional screenshot or text description before continuing."
        )
    if tool_name == "python_exec":
        code_preview = str((tool_args or {}).get("code", ""))[:80]
        return (
            f"[Tool degraded] python_exec unavailable: {err_text}\n"
            f"code: {code_preview or '(empty)'}\n"
            "Explain the code logic directly or provide expected results."
        )
    if tool_name == "web_fetch":
        url = str((tool_args or {}).get("url", "")).strip()
        if not url or "missing" in err_text.lower() or "argument" in err_text.lower():
            return (
                f"[Tool degraded] web_fetch invalid arguments: {err_text}\n"
                "web_fetch requires a full `url` argument. "
                "If you need search results, use web_search instead."
            )
        return (
            f"[Tool degraded] web_fetch unavailable: {err_text}\n"
            f"URL: {url}\n"
            "Check the URL and retry, or use web_search to find related pages."
        )
    if tool_name == "describe_image":
        fpath = str((tool_args or {}).get("path", "")).strip()
        return (
            f"[Tool degraded] describe_image unavailable: {err_text}\n"
            f"path: {fpath or '(empty)'}\n"
            "State clearly that the image cannot be analyzed."
        )
    if tool_name == "list_dir":
        fpath = str((tool_args or {}).get("path", "")).strip()
        return (
            f"[Tool degraded] list_dir unavailable: {err_text}\n"
            f"path: {fpath or '(cwork root)'}\n"
            "State clearly that directory listing failed."
        )
    if tool_name == "read_file":
        fpath = str((tool_args or {}).get("path", "")).strip()
        return (
            f"[Tool degraded] read_file unavailable: {err_text}\n"
            f"path: {fpath or '(empty)'}\n"
            "Try list_dir first to verify the path, or use web_search for the content."
        )
    if tool_name == "write_file":
        fpath = str((tool_args or {}).get("path", "")).strip()
        return (
            f"[Tool degraded] write_file unavailable: {err_text}\n"
            f"path: {fpath or '(empty)'}\n"
            "State that writing failed and suggest manual action."
        )
    return (
        f"[Tool degraded] {tool_name} unavailable: {err_text}\n"
        "Try an alternative approach or different tool to achieve the same goal."
    )


from urllib.parse import urlparse


def _simplify_query(query: str) -> str:
    """Strip common stopwords, keep max 3 keywords."""
    _stops = {"the", "a", "an", "and", "or", "of", "in", "for", "to",
              "how", "what", "is", "stock", "price", "latest", "today",
              "please", "help", "me", "can", "you", "check", "search"}
    words = query.split()
    core = [w for w in words if w.lower() not in _stops]
    return " ".join(core[:3]) if core else query


TOOL_FALLBACK_MAP: dict[str, list[tuple]] = {
    "web_search": [
        ("retry_refined", lambda args: {"query": _simplify_query(args.get("query", ""))}),
    ],
    "web_fetch": [
        ("retry_refined", lambda args: args),
        ("fallback_tool", "web_search", lambda args: {"query": urlparse(args.get("url", "")).hostname or args.get("url", "")}),
    ],
    "stock": [
        ("retry_refined", lambda args: args),
        ("fallback_tool", "web_search", lambda args: {"query": f"{args.get('symbol', '')} stock price today"}),
    ],
}


_LATENCY_TIMEOUT_MULTIPLIER = {"fast": 0.6, "medium": 1.0, "slow": 1.8}
_TIMEOUT_FLOOR = 10.0   # never below 10s
_TIMEOUT_CEIL = 180.0   # never above 3min
_SLOW_TIMEOUT_FLOOR = 45.0  # slow tools at least 45s


def _effective_timeout(tool_def, base_timeout: float) -> float:
    """Compute latency-aware timeout for *tool_def*."""
    latency_tier = getattr(tool_def.capability, "latency_tier", "medium")
    # MCP tools (namespaced with __) with unset default "fast" → treat as medium
    if latency_tier == "fast" and "__" in tool_def.name:
        latency_tier = "medium"
    mult = _LATENCY_TIMEOUT_MULTIPLIER.get(latency_tier, 1.0)
    tier_floor = _SLOW_TIMEOUT_FLOOR if latency_tier == "slow" else _TIMEOUT_FLOOR
    # Tool-declared minimum (e.g. run_tests 125s) takes priority over tier floor
    tool_min = getattr(tool_def.capability, "min_timeout_sec", 0.0) or 0.0
    floor = max(tier_floor, tool_min)
    ceil = max(_TIMEOUT_CEIL, tool_min)  # tool_min can raise ceiling
    return max(floor, min(base_timeout * mult, ceil))


class ToolExecutor:
    """Execute tools with timeout and bounded retries."""

    def __init__(self, tool_policy, retry_count: int, timeout_sec: float,
                 relation_graph=None):
        self.tool_policy = tool_policy
        self.retry_count = retry_count
        self.timeout_sec = timeout_sec
        self.relation_graph = relation_graph  # ToolRelationGraph | None
        self._active_cancel_scope = None

    @contextmanager
    def use_cancel_scope(self, cancel_scope):
        previous = self._active_cancel_scope
        self._active_cancel_scope = cancel_scope
        try:
            yield self
        finally:
            self._active_cancel_scope = previous

    async def _await_tool_call(self, awaitable, *, timeout_sec: float, cancel_scope=None):
        scope = cancel_scope if cancel_scope is not None else self._active_cancel_scope
        if scope is None:
            return await asyncio.wait_for(awaitable, timeout=timeout_sec)

        scope.raise_if_cancelled()
        tool_task = asyncio.create_task(awaitable)
        cancel_task = asyncio.create_task(scope.wait_requested())
        timeout_task = asyncio.create_task(asyncio.sleep(timeout_sec))
        try:
            done, _ = await asyncio.wait(
                {tool_task, cancel_task, timeout_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if tool_task in done:
                return await tool_task
            if cancel_task in done:
                tool_task.cancel()
                await asyncio.gather(tool_task, return_exceptions=True)
                scope.raise_if_cancelled()
                raise asyncio.CancelledError(await cancel_task)
            tool_task.cancel()
            await asyncio.gather(tool_task, return_exceptions=True)
            raise asyncio.TimeoutError()
        finally:
            for pending in (cancel_task, timeout_task):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(cancel_task, timeout_task, return_exceptions=True)

    async def execute(self, tool_def, tool_args: dict, *, cancel_scope=None) -> tuple[str, bool, str]:
        """Execute tool, returning (observation, is_error, error_type)."""
        last_err = ""
        err_type = "exception"
        # E3-1: non-idempotent tools get max_attempts=1 (no retry)
        is_idempotent = getattr(tool_def.capability, "idempotent", True)
        max_attempts = 1 if not is_idempotent else self.retry_count + 1
        # E3-2: latency-aware timeout
        effective_timeout = _effective_timeout(tool_def, self.timeout_sec)
        for _ in range(max_attempts):
            scope = cancel_scope if cancel_scope is not None else self._active_cancel_scope
            if scope is not None:
                scope.raise_if_cancelled()
            try:
                result = await self._await_tool_call(
                    tool_def.func(**tool_args),
                    timeout_sec=effective_timeout,
                    cancel_scope=scope,
                )
                return self.tool_policy.sanitize_output(tool_def, result), False, ""
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                last_err = f"Tool execution timed out (>{effective_timeout:.0f}s)"
                err_type = "timeout"
            except Exception as e:
                last_err = str(e)
                err_type = "exception"
        from .failure_taxonomy import classify_error
        kind = classify_error(err_type, last_err)
        return f"[Tool execution error] {last_err}", True, kind

    async def execute_with_fallback(
        self, tool_def, tool_args: dict, *, get_tool_fn=None, cancel_scope=None,
    ) -> tuple[str, bool, str, str, dict]:
        """Execute tool with fallback chain.

        Returns (observation, is_error, error_type, effective_tool_name, effective_args).
        """
        obs, is_err, err_type = await self.execute(tool_def, tool_args, cancel_scope=cancel_scope)
        if not is_err:
            return obs, False, "", tool_def.name, tool_args

        last_err_type = err_type  # E3-4: track through fallback chain

        # Gate: only attempt fallback for eligible failure kinds
        from .failure_taxonomy import FALLBACK_ELIGIBLE
        if err_type not in FALLBACK_ELIGIBLE:
            return obs, True, err_type, tool_def.name, tool_args

        # Graph-based fallback path (E2)
        if self.relation_graph is not None:
            fallbacks = self.relation_graph.get_fallbacks(tool_def.name)
            for rel in fallbacks:
                transformed_args = rel.transform_args(tool_args)
                if rel.target == tool_def.name:
                    # E3-1: non-idempotent tools NEVER self-retry with same args
                    if not tool_def.capability.idempotent and transformed_args == tool_args:
                        continue
                    # E3-3: timeout + network_timeout → skip same-args self-retry
                    if (last_err_type == "timeout"
                            and "network_timeout" in tool_def.capability.failure_modes
                            and transformed_args == tool_args):
                        continue
                    if rel.allow_same_args or transformed_args != tool_args:
                        obs2, is_err2, err_type2 = await self.execute(
                            tool_def, transformed_args, cancel_scope=cancel_scope,
                        )
                        if is_err2:
                            last_err_type = err_type2
                        if not is_err2:
                            return obs2, False, "", tool_def.name, transformed_args
                elif get_tool_fn:
                    fallback_def = get_tool_fn(rel.target)
                    if fallback_def:
                        obs2, is_err2, err_type2 = await self.execute(
                            fallback_def, transformed_args, cancel_scope=cancel_scope,
                        )
                        if is_err2:
                            last_err_type = err_type2
                        if not is_err2:
                            return obs2, False, "", rel.target, transformed_args
        else:
            # Legacy fallback path (backward compat)
            strategies = TOOL_FALLBACK_MAP.get(tool_def.name, [])
            for strategy in strategies:
                stype = strategy[0]
                if stype == "retry_refined":
                    args_fn = strategy[1]
                    refined_args = args_fn(tool_args)
                    if refined_args != tool_args:
                        obs2, is_err2, err_type2 = await self.execute(
                            tool_def, refined_args, cancel_scope=cancel_scope,
                        )
                        if is_err2:
                            last_err_type = err_type2
                        if not is_err2:
                            return obs2, False, "", tool_def.name, refined_args
                elif stype == "fallback_tool" and get_tool_fn:
                    fallback_name = strategy[1]
                    args_fn = strategy[2]
                    fallback_def = get_tool_fn(fallback_name)
                    if fallback_def:
                        fallback_args = args_fn(tool_args)
                        obs2, is_err2, err_type2 = await self.execute(
                            fallback_def, fallback_args, cancel_scope=cancel_scope,
                        )
                        if is_err2:
                            last_err_type = err_type2
                        if not is_err2:
                            return obs2, False, "", fallback_name, fallback_args

        return obs, True, last_err_type, tool_def.name, tool_args
