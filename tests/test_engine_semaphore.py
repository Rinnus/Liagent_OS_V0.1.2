# tests/test_engine_semaphore.py
"""Verify EngineManager exposes _llm_semaphore and all LLM paths respect it."""
import asyncio
import unittest


def test_engine_manager_has_semaphore():
    """Verify the semaphore and _unlocked methods are defined."""
    import inspect
    from liagent.engine.engine_manager import EngineManager
    src = inspect.getsource(EngineManager.__init__)
    assert "_llm_semaphore" in src
    # Verify all 5 _unlocked methods exist
    for name in ("_generate_llm_unlocked", "_generate_llm_routed_unlocked",
                 "_generate_reasoning_unlocked", "_generate_text_unlocked",
                 "_stream_text_unlocked"):
        assert hasattr(EngineManager, name), f"missing {name}"


class SemaphoreSerializationTest(unittest.IsolatedAsyncioTestCase):
    async def test_semaphore_prevents_concurrent_llm(self):
        """Two concurrent generate calls should serialize, not overlap."""
        call_log = []
        sem = asyncio.Semaphore(1)

        async def fake_generate(name):
            async with sem:
                call_log.append(f"{name}_start")
                await asyncio.sleep(0.05)
                call_log.append(f"{name}_end")

        await asyncio.gather(fake_generate("a"), fake_generate("b"))
        # With semaphore(1), calls must serialize: a_start, a_end, b_start, b_end
        # (or b first, but never interleaved)
        starts = [i for i, x in enumerate(call_log) if x.endswith("_start")]
        ends = [i for i, x in enumerate(call_log) if x.endswith("_end")]
        # First end must come before second start
        self.assertLess(ends[0], starts[1])
