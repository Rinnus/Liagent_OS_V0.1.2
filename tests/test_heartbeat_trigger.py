"""Tests for HeartbeatTrigger integration."""
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

from liagent.agent.triggers import FileWatcher, HeartbeatTrigger
from liagent.agent.heartbeat import HeartbeatConfig, HeartbeatMetrics
from liagent.agent.task_queue import TaskStore


class _AsyncTestCase(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)


class HeartbeatTriggerTests(_AsyncTestCase):

    def _config(self, **kw):
        defaults = dict(active_hours="00:00-23:59", timezone="UTC", cooldown_minutes=30,
                        channels=[], max_actions_per_run=3, dry_run=True,
                        action_allowlist=["notify_user"], instructions="test")
        defaults.update(kw)
        return HeartbeatConfig(**defaults)

    def test_in_active_hours_all_day(self):
        runner = MagicMock()
        config = self._config(active_hours="00:00-23:59")
        trigger = HeartbeatTrigger(runner, config)
        self.assertTrue(trigger._in_active_hours())

    def test_in_active_hours_restricted(self):
        runner = MagicMock()
        config = self._config(active_hours="00:00-00:01", timezone="UTC")
        trigger = HeartbeatTrigger(runner, config)
        # This may or may not be true depending on current time; test that it doesn't crash
        result = trigger._in_active_hours()
        self.assertIsInstance(result, bool)

    def test_start_stop(self):
        runner = AsyncMock()
        config = self._config()
        trigger = HeartbeatTrigger(runner, config, schedule="*/1 * * * *")
        # Just verify start/stop don't raise
        self._run(trigger.start())
        self.assertIsNotNone(trigger._task)
        self._run(trigger.stop())
        self.assertTrue(trigger._stopped)

    def test_overnight_hours_parsing(self):
        runner = MagicMock()
        config = self._config(active_hours="22:00-06:00", timezone="UTC")
        trigger = HeartbeatTrigger(runner, config)
        # Should not crash on overnight range
        result = trigger._in_active_hours()
        self.assertIsInstance(result, bool)

    def test_default_schedule(self):
        runner = MagicMock()
        config = self._config()
        trigger = HeartbeatTrigger(runner, config)
        self.assertEqual(trigger.schedule, "*/30 * * * *")

    def test_custom_schedule(self):
        runner = MagicMock()
        config = self._config()
        trigger = HeartbeatTrigger(runner, config, schedule="*/5 * * * *")
        self.assertEqual(trigger.schedule, "*/5 * * * *")

    def test_invalid_active_hours_defaults_true(self):
        runner = MagicMock()
        config = self._config(active_hours="invalid")
        trigger = HeartbeatTrigger(runner, config)
        # Should default to True on parse failure
        self.assertTrue(trigger._in_active_hours())


class ConfirmationSweepTests(unittest.TestCase):
    def test_sweep_expires_old_pending_confirms(self):
        """Sweep should transition expired pending_confirm runs to expired."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        store = TaskStore(db_path=Path(tmp.name))
        task = store.create_task(
            name="test", trigger_type="heartbeat",
            trigger_config={}, prompt_template="test",
        )
        run = store.create_run(task["id"], trigger_event="test")
        store.update_run(run["id"], status="pending_confirm",
                         expires_at="2020-01-01T00:00:00+00:00")

        expired_ids = store.get_expired_pending_confirms()
        self.assertIn(run["id"], expired_ids)

        ok = store.transition_run(run["id"], "pending_confirm", "expired")
        self.assertTrue(ok)

        expired_ids2 = store.get_expired_pending_confirms()
        self.assertNotIn(run["id"], expired_ids2)
        Path(tmp.name).unlink(missing_ok=True)


class FileWatcherFallbackTests(_AsyncTestCase):
    def test_filewatch_falls_back_to_polling_when_native_start_fails(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        store = TaskStore(db_path=Path(tmp.name))
        watcher = FileWatcher(executor=MagicMock(), store=store, cwork_dir=Path(tempfile.mkdtemp()))

        class _BaseObserver:
            def __init__(self):
                self.daemon = False
                self.scheduled = []

            def schedule(self, handler, path, recursive=True):
                self.scheduled.append((handler, path, recursive))

            def stop(self):
                return None

            def join(self, timeout=None):
                return None

        class _KqueueObserver(_BaseObserver):
            def start(self):
                raise RuntimeError("kqueue unavailable")

        class _PollingObserver(_BaseObserver):
            def start(self):
                return None

        class _NativeObserver(_BaseObserver):
            def start(self):
                raise RuntimeError("native unavailable")

        watchdog_observers = ModuleType("watchdog.observers")
        watchdog_observers.Observer = _NativeObserver
        watchdog_kqueue = ModuleType("watchdog.observers.kqueue")
        watchdog_kqueue.KqueueObserver = _KqueueObserver
        watchdog_polling = ModuleType("watchdog.observers.polling")
        watchdog_polling.PollingObserver = _PollingObserver
        watchdog_events = ModuleType("watchdog.events")

        class _FileSystemEventHandler:
            pass

        watchdog_events.FileSystemEventHandler = _FileSystemEventHandler

        with patch.dict(
            sys.modules,
            {
                "watchdog.observers": watchdog_observers,
                "watchdog.observers.kqueue": watchdog_kqueue,
                "watchdog.observers.polling": watchdog_polling,
                "watchdog.events": watchdog_events,
            },
        ):
            self._run(watcher.start())

        self.assertEqual(watcher._observer_backend, "polling")
        self.assertIsInstance(watcher._observer, _PollingObserver)
        self._run(watcher.stop())
        Path(tmp.name).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
