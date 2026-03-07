import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TryGenerateReasoningTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_none_when_locked(self):
        from liagent.engine.engine_manager import EngineManager
        em = MagicMock(spec=EngineManager)
        em._llm_semaphore = asyncio.Semaphore(1)
        await em._llm_semaphore.acquire()  # Lock it

        em._generate_reasoning_unlocked = AsyncMock(return_value="test")
        em.try_generate_reasoning = EngineManager.try_generate_reasoning.__get__(em)

        result = await em.try_generate_reasoning(
            messages=[{"role": "user", "content": "test"}],
            timeout=0.01,
        )
        self.assertIsNone(result)
        em._generate_reasoning_unlocked.assert_not_called()
        em._llm_semaphore.release()

    async def test_returns_result_when_unlocked(self):
        from liagent.engine.engine_manager import EngineManager
        em = MagicMock(spec=EngineManager)
        em._llm_semaphore = asyncio.Semaphore(1)
        em._generate_reasoning_unlocked = AsyncMock(return_value="result text")
        em.try_generate_reasoning = EngineManager.try_generate_reasoning.__get__(em)

        result = await em.try_generate_reasoning(
            messages=[{"role": "user", "content": "test"}],
            timeout=0.05,
        )
        self.assertEqual(result, "result text")
        em._generate_reasoning_unlocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
