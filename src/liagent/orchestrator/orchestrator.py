"""Orchestrator — intent routing + event formatting + delegate to brain.

Integrates PolicyRouter for intent classification and GroundingGate
for quality checks on research synthesis output.
"""

from __future__ import annotations

import uuid
import inspect
from collections.abc import AsyncIterator

from ..agent.grounding_gate import GroundingGate
from ..agent.policy_router import PolicyRouter
from ..agent.run_control import RunCancellationScope
from ..logging import get_logger
from .events import AgentEvent, EventSequencer, make_event

_log = get_logger("orchestrator")


class Orchestrator:
    """Orchestrator wrapping brain.run() with intent routing and quality gate.

    Responsibilities:
    1. Classify user intent via PolicyRouter
    2. Emit 'dispatch' AgentEvent for UI compatibility
    3. Delegate to brain.run() with routing metadata
    4. Quality-gate final output via GroundingGate
    """

    def __init__(self, engine, brain):
        self.brain = brain
        self._engine = engine
        self._router = PolicyRouter()
        self._gate = GroundingGate(
            experience=getattr(brain, "experience", None),
        )

    async def dispatch(
        self,
        query: str,
        *,
        images: list[str] | None = None,
        low_latency: bool = False,
        session_key: str | None = None,
        cancel_scope: RunCancellationScope | None = None,
    ) -> AsyncIterator[AgentEvent]:
        run_id = str(uuid.uuid4())[:8]
        seq = EventSequencer()

        # Classify intent
        intent = self._router.classify(query, images=images, low_latency=low_latency)
        _log.trace("orchestrator_dispatch",
                    intent_category=intent.category, task_class=intent.task_class,
                    entities=intent.entities, reason=intent.reason)

        yield make_event(
            "dispatch",
            {"mode": intent.category, "query": query,
             "task_class": intent.task_class, "entities": intent.entities},
            source="brain", run_id=run_id, agent_id="brain", sequencer=seq,
        )

        run_kwargs = {
            "images": images,
            "low_latency": low_latency,
            "session_key": session_key,
        }
        try:
            run_sig = inspect.signature(self.brain.run)
            if "cancel_scope" in run_sig.parameters:
                run_kwargs["cancel_scope"] = cancel_scope
        except (TypeError, ValueError):
            pass

        collected_answer = ""
        async for legacy_event in self.brain.run(query, **run_kwargs):
            evt_type = legacy_event[0]
            evt_payload = (
                legacy_event[1]
                if len(legacy_event) == 2
                else legacy_event[1:]
            )
            # Collect answer tokens for quality gate
            if evt_type == "token" and len(legacy_event) >= 2:
                collected_answer += str(legacy_event[1])
            yield make_event(
                evt_type, evt_payload,
                source="brain", run_id=run_id, agent_id="brain", sequencer=seq,
            )

        # Quality gate on final collected answer
        if collected_answer.strip():
            verdict = self._gate.check(answer=collected_answer, query=query)
            if verdict.action != "accept":
                _log.trace("grounding_gate_post",
                           action=verdict.action,
                           reason=verdict.retry_reason)
            if verdict.quality_meta:
                yield make_event(
                    "quality_gate",
                    {"verdict": verdict.action, **verdict.quality_meta},
                    source="orchestrator", run_id=run_id,
                    agent_id="orchestrator", sequencer=seq,
                )

    async def shutdown(self):
        """Graceful shutdown — delegate to brain."""
        if hasattr(self.brain, "shutdown"):
            await self.brain.shutdown()
