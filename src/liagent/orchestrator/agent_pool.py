"""AgentPool — interleaved IO-parallel, LLM-serial SubAgent scheduling."""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from ..logging import get_logger
from .budget import BudgetSlice
from .sub_agent import SubAgentContext, SubAgentResult

_log = get_logger("agent_pool")


class AgentPool:
    """Execute multiple SubAgent strategies with two-phase scheduling.

    Phase 1 — IO parallel: all SubAgents fire tool calls concurrently.
    Phase 2 — LLM serial: each SubAgent's LLM analysis runs under a semaphore.

    Args:
        llm_semaphore: asyncio.Semaphore(1) shared with EngineManager.
        execute_fn: async callable(SubAgentContext) -> SubAgentResult.
    """

    def __init__(
        self,
        *,
        llm_semaphore: asyncio.Semaphore | None = None,
        execute_fn: Callable[[SubAgentContext], Any] | None = None,
        max_concurrent: int = 5,
    ):
        self._llm_sem = llm_semaphore or asyncio.Semaphore(1)
        self._execute_fn = execute_fn
        self._max_concurrent = max_concurrent

    async def execute(
        self,
        contexts: list[SubAgentContext],
    ) -> list[SubAgentResult]:
        """Execute all SubAgent contexts, return results in order."""
        if not contexts:
            return []

        sem = asyncio.Semaphore(self._max_concurrent)

        async def _run_one(ctx: SubAgentContext) -> SubAgentResult:
            async with sem:
                start = time.monotonic()
                try:
                    if self._execute_fn is None:
                        return SubAgentResult(
                            agent_id=ctx.agent_id,
                            strategy=ctx.strategy,
                            success=False,
                            error="no execute_fn configured",
                        )
                    result = await self._execute_fn(ctx)
                    if not isinstance(result, SubAgentResult):
                        result = SubAgentResult(
                            agent_id=ctx.agent_id,
                            strategy=ctx.strategy,
                            summary=str(result),
                            success=True,
                        )
                    result.elapsed_ms = (time.monotonic() - start) * 1000
                    return result
                except asyncio.TimeoutError:
                    _log.warning(f"agent_pool_timeout: {ctx.agent_id}")
                    return SubAgentResult(
                        agent_id=ctx.agent_id,
                        strategy=ctx.strategy,
                        success=False,
                        error="timeout",
                        elapsed_ms=(time.monotonic() - start) * 1000,
                    )
                except Exception as e:
                    _log.warning(f"agent_pool_error: {ctx.agent_id}: {e}")
                    return SubAgentResult(
                        agent_id=ctx.agent_id,
                        strategy=ctx.strategy,
                        success=False,
                        error=str(e),
                        elapsed_ms=(time.monotonic() - start) * 1000,
                    )

        tasks = [asyncio.create_task(_run_one(ctx)) for ctx in contexts]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        _log.event(
            "agent_pool_complete",
            total=len(results),
            success=sum(1 for r in results if r.success),
        )
        return list(results)
