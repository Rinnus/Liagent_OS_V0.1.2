"""Shared runtime primitives for stable MLX execution."""

from __future__ import annotations

import asyncio
import atexit
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")

# MLX/Metal command buffer usage is fragile under concurrent encoders.
# We serialize MLX compute through one dedicated worker thread.
_MLX_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_MLX_LOCK = threading.Lock()
_RUNTIME_GUARD = threading.Lock()

_DEFAULT_TIMEOUT_SEC = max(
    1.0, float(os.environ.get("LIAGENT_MLX_TIMEOUT_SEC", "45"))
)


def _shutdown_executor():
    try:
        _MLX_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


atexit.register(_shutdown_executor)


def _reset_runtime_if_current(
    expected_executor: ThreadPoolExecutor,
    expected_lock: threading.Lock,
):
    """Replace executor+lock when current runtime appears wedged."""
    global _MLX_EXECUTOR, _MLX_LOCK
    with _RUNTIME_GUARD:
        if _MLX_EXECUTOR is not expected_executor or _MLX_LOCK is not expected_lock:
            return
        old_executor = _MLX_EXECUTOR
        _MLX_EXECUTOR = ThreadPoolExecutor(max_workers=1)
        _MLX_LOCK = threading.Lock()
        try:
            old_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


async def run_mlx_serialized(
    fn: Callable[[], T],
    *,
    timeout_sec: float | None = None,
) -> T:
    """Run MLX workload in a single-thread serialized lane with recovery timeout."""
    loop = asyncio.get_running_loop()
    with _RUNTIME_GUARD:
        executor = _MLX_EXECUTOR
        lock = _MLX_LOCK

    def _run() -> T:
        with lock:
            return fn()

    fut = loop.run_in_executor(executor, _run)
    limit = _DEFAULT_TIMEOUT_SEC if timeout_sec is None else float(timeout_sec)
    if limit <= 0:
        return await fut
    try:
        return await asyncio.wait_for(fut, timeout=limit)
    except asyncio.TimeoutError as e:
        fut.cancel()
        _reset_runtime_if_current(executor, lock)
        raise TimeoutError(f"MLX execution timeout ({limit:.1f}s)") from e
    except asyncio.CancelledError:
        # Caller cancelled (new turn or Ctrl+C). If the worker keeps running,
        # move future requests to a fresh executor so the app stays responsive.
        if not fut.done():
            fut.cancel()
            _reset_runtime_if_current(executor, lock)
        raise


def shutdown_mlx_runtime(wait: bool = False):
    """Shutdown MLX executor explicitly during app exit."""
    global _MLX_EXECUTOR, _MLX_LOCK
    with _RUNTIME_GUARD:
        old = _MLX_EXECUTOR
        _MLX_EXECUTOR = ThreadPoolExecutor(max_workers=1)
        _MLX_LOCK = threading.Lock()
    try:
        old.shutdown(wait=wait, cancel_futures=True)
    except Exception:
        pass
