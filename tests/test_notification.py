# tests/test_notification.py
"""Tests for notification channel router: reliability, dedup, rate-limit, fallback."""

import asyncio
import unittest

from liagent.agent.notification import ChannelRouter


class _FakeChannel:
    def __init__(self, *, fail=False, channel_type="fake"):
        self._fail = fail
        self._type = channel_type
        self.sent: list[str] = []

    @property
    def channel_type(self):
        return self._type

    async def send(self, message, *, priority="normal"):
        if self._fail:
            return False
        self.sent.append(message)
        return True


class ChannelRouterTests(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_dispatch_sends_to_first_channel(self):
        ch1 = _FakeChannel(channel_type="primary")
        ch2 = _FakeChannel(channel_type="secondary")
        router = ChannelRouter(channels=[ch1, ch2])
        ok = self._run(router.dispatch("hello"))
        self.assertTrue(ok)
        self.assertEqual(ch1.sent, ["hello"])
        self.assertEqual(ch2.sent, [])

    def test_fallback_on_failure(self):
        ch1 = _FakeChannel(fail=True, channel_type="broken")
        ch2 = _FakeChannel(channel_type="backup")
        router = ChannelRouter(channels=[ch1, ch2])
        ok = self._run(router.dispatch("hello"))
        self.assertTrue(ok)
        self.assertEqual(ch2.sent, ["hello"])

    def test_all_fail_returns_false(self):
        ch1 = _FakeChannel(fail=True)
        router = ChannelRouter(channels=[ch1])
        ok = self._run(router.dispatch("hello"))
        self.assertFalse(ok)

    def test_dedup_within_window(self):
        ch = _FakeChannel()
        router = ChannelRouter(channels=[ch], dedup_window_sec=300)
        self._run(router.dispatch("same message"))
        self._run(router.dispatch("same message"))
        self.assertEqual(len(ch.sent), 1)

    def test_different_messages_not_deduped(self):
        ch = _FakeChannel()
        router = ChannelRouter(channels=[ch], dedup_window_sec=300)
        self._run(router.dispatch("message A"))
        self._run(router.dispatch("message B"))
        self.assertEqual(len(ch.sent), 2)

    def test_rate_limit(self):
        ch = _FakeChannel()
        router = ChannelRouter(channels=[ch], rate_limit_per_min=2)
        for i in range(5):
            self._run(router.dispatch(f"msg {i}"))
        self.assertEqual(len(ch.sent), 2)

    def test_empty_channels(self):
        router = ChannelRouter(channels=[])
        ok = self._run(router.dispatch("hello"))
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
