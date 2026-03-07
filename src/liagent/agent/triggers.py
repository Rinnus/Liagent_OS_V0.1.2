"""Trigger system for autonomous tasks: CronTrigger + FileWatcher + TriggerManager."""

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from ..logging import get_logger
from .time_utils import _LOCAL_TZ, _now_local, _now_local_iso

if TYPE_CHECKING:
    from .task_queue import TaskExecutor, TaskStore

_log = get_logger("triggers")


# ─── CronTrigger ──────────────────────────────────────────────────────────────

class CronTrigger:
    """Schedules a task based on a cron expression using croniter.

    All times use the system local timezone. Each trigger runs as its own
    asyncio.Task, sleeping until the next scheduled time.
    """

    def __init__(
        self,
        task_id: str,
        schedule: str,
        executor: "TaskExecutor",
        store: "TaskStore",
    ):
        self.task_id = task_id
        self.schedule = schedule
        self.executor = executor
        self.store = store
        self._task: asyncio.Task | None = None
        self._stopped = False

    async def start(self):
        self._stopped = False
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stopped = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _loop(self):
        from croniter import croniter

        while not self._stopped:
            try:
                now = _now_local()
                cron = croniter(self.schedule, now)
                next_time = cron.get_next(datetime)
                # Store next_run_at
                self.store.update_task(
                    self.task_id, next_run_at=next_time.isoformat()
                )
                delay = (next_time - now).total_seconds()
                if delay > 0:
                    await asyncio.sleep(delay)

                # Check task is still active
                task = self.store.get_task(self.task_id)
                if not task or task.get("status") != "active":
                    _log.event("cron_stopped", task_id=self.task_id)
                    break

                # Create and enqueue run
                prompt = task.get("prompt_template", "")
                run = self.store.create_run(
                    self.task_id,
                    trigger_event=f"cron:{self.schedule}",
                    prompt=prompt,
                )
                self.executor.enqueue(
                    run["id"],
                    self.task_id,
                    prompt,
                    priority=task.get("priority", 10) or 10,
                    trigger_event=f"cron:{self.schedule}",
                    goal_id=task.get("goal_id"),
                    source=task.get("source", "system"),
                )
                _log.event("cron_triggered", task_id=self.task_id, run_id=run["id"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                _log.error("cron_trigger", e, action=f"task={self.task_id}")
                await asyncio.sleep(60)  # Back off on error


# ─── OnceTrigger ──────────────────────────────────────────────────────────────

class OnceTrigger:
    """One-shot delayed trigger. Fires at fire_at (absolute local time), then marks task as done.

    Accepts either fire_at (datetime) or delay_seconds (int). If fire_at is provided,
    delay is computed as remaining seconds. This enables restart recovery.
    """

    def __init__(
        self,
        task_id: str,
        executor: "TaskExecutor",
        store: "TaskStore",
        *,
        fire_at: datetime | None = None,
        delay_seconds: int = 0,
    ):
        self.task_id = task_id
        self.executor = executor
        self.store = store
        self._task: asyncio.Task | None = None

        if fire_at is not None:
            self.fire_at = fire_at
        else:
            self.fire_at = _now_local() + __import__("datetime").timedelta(seconds=delay_seconds)

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _run(self):
        try:
            now = _now_local()
            remaining = (self.fire_at - now).total_seconds()
            if remaining > 0:
                _log.event("once_waiting", task_id=self.task_id,
                           fire_at=self.fire_at.isoformat(), remaining_s=round(remaining))
                await asyncio.sleep(remaining)
            else:
                _log.event("once_fire_immediate", task_id=self.task_id,
                           fire_at=self.fire_at.isoformat(), overdue_s=round(-remaining))

            task = self.store.get_task(self.task_id)
            if not task or task.get("status") != "active":
                return

            prompt = task.get("prompt_template", "")
            delay_desc = f"{round((self.fire_at - _now_local()).total_seconds())}s" if remaining > 0 else "immediate"
            run = self.store.create_run(
                self.task_id,
                trigger_event=f"once:{self.fire_at.strftime('%H:%M:%S')}",
                prompt=prompt,
            )
            self.executor.enqueue(
                run["id"],
                self.task_id,
                prompt,
                priority=task.get("priority", 10) or 10,
                trigger_event=f"once:{self.fire_at.strftime('%H:%M:%S')}",
                goal_id=task.get("goal_id"),
                source=task.get("source", "system"),
            )
            _log.event("once_triggered", task_id=self.task_id, run_id=run["id"])
            # Note: task is NOT marked done here. The executor marks it done
            # after successful execution, so retry logic works on failure.

        except asyncio.CancelledError:
            pass
        except Exception as e:
            _log.error("once_trigger", e, action=f"task={self.task_id}")


# ─── FileWatcher ──────────────────────────────────────────────────────────────

# Safety whitelist for watched paths
ALLOWED_ROOTS = [
    Path.home() / "Desktop",
    Path.home() / "Documents",
    Path.home() / "Downloads",
]
DENIED_PATTERNS = {".*", "node_modules", "__pycache__", "venv", ".git"}

def _default_cwork_dir() -> Path:
    raw = (
        os.environ.get("LIAGENT_CWORK_DIR")
        or os.environ.get("LIAGENT_CWORK_ROOT")
        or str(Path.home() / "Desktop" / "cwork")
    )
    return Path(raw).expanduser().resolve()


# Preset subdirectories under cwork/
CWORK_DIR = _default_cwork_dir()
PRESET_DIRS: dict[str, str] = {
    "inbox": "This file was just placed in the inbox. Read it and decide how to handle it: {file_path}",
    "summarize": "Summarize the following file in under 300 words: {file_path}",
    "translate": "Translate the following file into English (if already English, translate into Chinese): {file_path}",
    "analyze": "Analyze the following image and provide a detailed description: {file_path}",
    "code-review": "Review the following code file for quality issues and improvement suggestions: {file_path}",
}


def _is_path_allowed(path: Path) -> bool:
    """Check if a path is within allowed roots and not in denied patterns."""
    resolved = path.resolve()
    if not any(
        resolved == root or resolved.is_relative_to(root)
        for root in ALLOWED_ROOTS
    ):
        return False
    for part in resolved.parts:
        for pattern in DENIED_PATTERNS:
            if pattern.startswith(".") and part.startswith("."):
                return False
            if part == pattern:
                return False
    return True


class _FileEventHandler:
    """Watchdog event handler that bridges file events to asyncio.

    Implements debouncing: same file within 5 seconds is ignored.
    """

    def __init__(
        self,
        executor: "TaskExecutor",
        store: "TaskStore",
        loop: asyncio.AbstractEventLoop,
        cwork_dir: Path | None = None,
    ):
        self.executor = executor
        self.store = store
        self.loop = loop
        self.cwork_dir = (
            Path(cwork_dir).expanduser().resolve()
            if cwork_dir
            else _default_cwork_dir()
        )
        self._last_seen: dict[str, float] = {}
        self._debounce_sec = 5.0

    def dispatch(self, event):
        """Called by watchdog from a background thread."""
        if event.is_directory:
            return
        if event.event_type not in ("created", "modified", "moved"):
            return

        src_path = getattr(event, "dest_path", None) or event.src_path
        path = Path(src_path)

        if not _is_path_allowed(path):
            return

        # Debounce
        key = str(path)
        now = time.time()
        if key in self._last_seen and (now - self._last_seen[key]) < self._debounce_sec:
            return
        self._last_seen[key] = now

        # Bridge to async
        self.loop.call_soon_threadsafe(
            asyncio.ensure_future,
            self._handle_file(path),
        )

    async def _handle_file(self, path: Path):
        """Find matching FileWatch task and enqueue a run."""
        try:
            # Determine which cwork subdirectory this belongs to
            subdir = None
            if path.is_relative_to(self.cwork_dir):
                relative = path.relative_to(self.cwork_dir)
                if relative.parts:
                    subdir = relative.parts[0]

            # Find matching filewatch task
            tasks = self.store.get_active_tasks_by_trigger("filewatch")
            matched_task = None
            for task in tasks:
                config = task.get("trigger_config", {})
                watch_dir = config.get("watch_dir", "")
                if subdir and watch_dir.rstrip("/").endswith(subdir):
                    matched_task = task
                    break
                elif str(path).startswith(watch_dir):
                    matched_task = task
                    break

            if not matched_task:
                # Fallback: use inbox task if file is directly in cwork/
                for task in tasks:
                    config = task.get("trigger_config", {})
                    if config.get("watch_dir", "").rstrip("/").endswith("inbox"):
                        matched_task = task
                        break

            if not matched_task:
                _log.event("filewatch_no_match", path=str(path))
                return

            # Build prompt from template
            prompt_template = matched_task.get("prompt_template", "")
            prompt = prompt_template.format(file_path=str(path))

            run = self.store.create_run(
                matched_task["id"],
                trigger_event=f"file:{path.name}",
                prompt=prompt,
            )
            self.executor.enqueue(
                run["id"],
                matched_task["id"],
                prompt,
                priority=matched_task.get("priority", 10) or 10,
                trigger_event=f"file:{path.name}",
                goal_id=matched_task.get("goal_id"),
                source=matched_task.get("source", "system"),
            )
            _log.event("filewatch_triggered", task_id=matched_task["id"], file=path.name)

        except Exception as e:
            _log.error("file_watcher", e, action=f"handle_file={path}")


class FileWatcher:
    """Watches cwork directory for file changes using watchdog."""

    def __init__(
        self,
        executor: "TaskExecutor",
        store: "TaskStore",
        cwork_dir: str | Path | None = None,
    ):
        self.executor = executor
        self.store = store
        self.cwork_dir = (
            Path(cwork_dir).expanduser().resolve()
            if cwork_dir
            else _default_cwork_dir()
        )
        self._observer = None
        self._observer_backend: str | None = None

    def _ensure_cwork_dirs(self):
        """Create cwork directory and preset subdirectories."""
        for subdir in PRESET_DIRS:
            (self.cwork_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _ensure_preset_tasks(self):
        """Insert preset FileWatch tasks if they don't already exist."""
        existing = self.store.get_active_tasks_by_trigger("filewatch")
        existing_dirs = set()
        for t in existing:
            config = t.get("trigger_config", {})
            existing_dirs.add(config.get("watch_dir", ""))

        for subdir, prompt_template in PRESET_DIRS.items():
            watch_dir = str(self.cwork_dir / subdir)
            if watch_dir in existing_dirs:
                continue
            self.store.create_task(
                name=f"FileWatch: {subdir}",
                trigger_type="filewatch",
                trigger_config={"watch_dir": watch_dir, "recursive": False},
                prompt_template=prompt_template,
                priority=5,  # Higher priority than default cron tasks
            )
            _log.event("filewatch_preset_created", subdir=subdir)

    async def start(self):
        """Start watching cwork directory."""
        try:
            from watchdog.observers import Observer
            try:
                from watchdog.observers.kqueue import KqueueObserver
            except ImportError:
                KqueueObserver = None
            from watchdog.observers.polling import PollingObserver
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            _log.warning("file_watcher: watchdog not installed, file watching disabled")
            return

        self._ensure_cwork_dirs()
        self._ensure_preset_tasks()

        loop = asyncio.get_running_loop()
        handler = _FileEventHandler(self.executor, self.store, loop, cwork_dir=self.cwork_dir)

        # Create a proper watchdog handler that delegates to our handler
        class _WatchdogBridge(FileSystemEventHandler):
            def on_any_event(self, event):
                handler.dispatch(event)

        watch_bridge = _WatchdogBridge()

        def _build_observer(observer_cls):
            observer = observer_cls()
            observer.schedule(
                watch_bridge,
                str(self.cwork_dir),
                recursive=True,
            )
            observer.daemon = True
            return observer

        backend_pref = str(os.environ.get("LIAGENT_FILEWATCH_BACKEND", "auto") or "auto").strip().lower()
        observer_choices: list[tuple[str, type]] = []
        if backend_pref == "native":
            observer_choices = [("native", Observer)]
        elif backend_pref == "kqueue":
            if KqueueObserver is None:
                _log.warning("filewatch_kqueue_unavailable")
                observer_choices = [("polling", PollingObserver)]
            else:
                observer_choices = [("kqueue", KqueueObserver)]
        elif backend_pref == "polling":
            observer_choices = [("polling", PollingObserver)]
        else:
            if sys.platform == "darwin":
                if KqueueObserver is not None:
                    observer_choices.append(("kqueue", KqueueObserver))
                observer_choices.append(("polling", PollingObserver))
            observer_choices.append(("native", Observer))

        last_exc = None
        for backend_name, observer_cls in observer_choices:
            try:
                self._observer = _build_observer(observer_cls)
                self._observer.start()
                self._observer_backend = backend_name
                break
            except Exception as exc:
                last_exc = exc
                _log.warning("filewatch_backend_start_failed", backend=backend_name, error=str(exc))
                if self._observer is not None:
                    try:
                        self._observer.stop()
                    except Exception:
                        pass
                    try:
                        self._observer.join(timeout=1)
                    except Exception:
                        pass
                self._observer = None
        if self._observer is None:
            if last_exc is not None:
                raise last_exc
            return
        _log.event(
            "filewatch_started",
            path=str(self.cwork_dir),
            backend=self._observer_backend or "unknown",
        )

    async def stop(self):
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            self._observer_backend = None


# ─── HeartbeatTrigger ─────────────────────────────────────────────────────────

class HeartbeatTrigger:
    """Runs HeartbeatRunner on a cron schedule, respecting active_hours."""

    def __init__(self, runner, config, schedule: str = "*/30 * * * *"):
        self.runner = runner    # HeartbeatRunner instance
        self.config = config    # HeartbeatConfig
        self.schedule = schedule  # cron expression, default every 30 min
        self._task: asyncio.Task | None = None
        self._stopped = False

    async def start(self):
        self._stopped = False
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stopped = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    def _in_active_hours(self) -> bool:
        """Check if current time is within config.active_hours (e.g. '08:00-22:00')."""
        try:
            start_str, end_str = self.config.active_hours.split("-")
            tz = ZoneInfo(self.config.timezone)
            now = datetime.now(tz)
            start_h, start_m = map(int, start_str.split(":"))
            end_h, end_m = map(int, end_str.split(":"))
            current_minutes = now.hour * 60 + now.minute
            start_minutes = start_h * 60 + start_m
            end_minutes = end_h * 60 + end_m
            if start_minutes <= end_minutes:
                return start_minutes <= current_minutes < end_minutes
            else:  # overnight range like 22:00-06:00
                return current_minutes >= start_minutes or current_minutes < end_minutes
        except Exception:
            return True  # Default to active if parsing fails

    async def _loop(self):
        from croniter import croniter

        while not self._stopped:
            try:
                now = _now_local()
                cron = croniter(self.schedule, now)
                next_time = cron.get_next(datetime)
                delay = (next_time - now).total_seconds()
                if delay > 0:
                    await asyncio.sleep(delay)

                if not self._in_active_hours():
                    _log.event("heartbeat_outside_hours",
                               active_hours=self.config.active_hours)
                    continue

                metrics = await self.runner.run()
                _log.event("heartbeat_completed", run_id=metrics.run_id,
                           actions_queued=metrics.actions_queued,
                           llm_invoked=metrics.llm_invoked)
            except asyncio.CancelledError:
                break
            except Exception as e:
                _log.error("heartbeat_trigger", e)
                await asyncio.sleep(60)


# ─── TriggerManager ───────────────────────────────────────────────────────────

class TriggerManager:
    """Manages the lifecycle of all triggers (cron, filewatch, etc.)."""

    def __init__(
        self,
        executor: "TaskExecutor",
        store: "TaskStore",
        *,
        cwork_dir: str | Path | None = None,
    ):
        self.executor = executor
        self.store = store
        self._cwork_dir = cwork_dir
        self._cron_triggers: dict[str, CronTrigger] = {}
        self._once_triggers: dict[str, OnceTrigger] = {}
        self._file_watcher: FileWatcher | None = None
        self._heartbeat_trigger: HeartbeatTrigger | None = None
        self._sweep_task: asyncio.Task | None = None
        self._stopped = False

    async def start(self):
        """Start all triggers based on active tasks in the store."""
        # Start cron triggers
        cron_tasks = self.store.get_active_tasks_by_trigger("cron")
        for task in cron_tasks:
            config = task.get("trigger_config", {})
            schedule = config.get("schedule", "")
            if schedule:
                await self.register_cron(task["id"], schedule)

        # Recover once triggers (restart recovery)
        # Primary source: next_run_at (runtime state, always up-to-date)
        # Fallback: trigger_config.fire_at (creation-time record)
        once_tasks = self.store.get_active_tasks_by_trigger("once")
        once_recovered = 0
        once_missed = 0
        for task in once_tasks:
            fire_at_str = task.get("next_run_at", "")
            if not fire_at_str:
                config = task.get("trigger_config", {})
                fire_at_str = config.get("fire_at", "")
            if not fire_at_str:
                self.store.update_task(task["id"], status="done")
                once_missed += 1
                continue
            try:
                fire_at = datetime.fromisoformat(fire_at_str)
                # Ensure timezone-aware
                if fire_at.tzinfo is None:
                    fire_at = fire_at.replace(tzinfo=_LOCAL_TZ)
                now = _now_local()
                remaining = (fire_at - now).total_seconds()

                if remaining < -3600:
                    # More than 1 hour overdue — mark as missed, don't execute
                    self.store.update_task(task["id"], status="done")
                    _log.event("once_missed", task_id=task["id"],
                               fire_at=fire_at_str, overdue_s=round(-remaining))
                    once_missed += 1
                else:
                    # Still in window (future, or up to 1h overdue) — fire it
                    trigger = OnceTrigger(
                        task["id"], self.executor, self.store, fire_at=fire_at,
                    )
                    self._once_triggers[task["id"]] = trigger
                    await trigger.start()
                    once_recovered += 1
                    _log.event("once_recovered", task_id=task["id"],
                               fire_at=fire_at_str, remaining_s=round(max(0, remaining)))
            except (ValueError, TypeError) as e:
                _log.error("once_recovery", e, task_id=task["id"])
                self.store.update_task(task["id"], status="done")
                once_missed += 1

        # Start file watcher
        self._file_watcher = FileWatcher(self.executor, self.store, cwork_dir=self._cwork_dir)
        await self._file_watcher.start()

        # Start confirmation sweep
        self._sweep_task = asyncio.create_task(self._sweep_expired_confirmations())

        _log.event("triggers_started",
                   cron_count=len(self._cron_triggers),
                   once_recovered=once_recovered,
                   once_missed=once_missed)

    async def stop(self):
        """Stop all triggers."""
        self._stopped = True
        for trigger in self._cron_triggers.values():
            await trigger.stop()
        self._cron_triggers.clear()
        for trigger in self._once_triggers.values():
            await trigger.stop()
        self._once_triggers.clear()
        if self._file_watcher:
            await self._file_watcher.stop()
        if self._heartbeat_trigger:
            await self._heartbeat_trigger.stop()
        if self._sweep_task and not self._sweep_task.done():
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _sweep_expired_confirmations(self):
        """Periodically expire pending_confirm runs that have passed their deadline."""
        while not self._stopped:
            try:
                await asyncio.sleep(60)
                expired_ids = self.store.get_expired_pending_confirms()
                for run_id in expired_ids:
                    ok = self.store.transition_run(run_id, "pending_confirm", "expired",
                                                  finished_at=_now_local_iso())
                    if ok:
                        _log.event("confirmation_expired", run_id=run_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                _log.error("sweep_expired", e)
                await asyncio.sleep(60)

    async def register_heartbeat(self, runner, config, schedule: str = "*/30 * * * *"):
        """Register a heartbeat trigger."""
        if self._heartbeat_trigger:
            await self._heartbeat_trigger.stop()
        self._heartbeat_trigger = HeartbeatTrigger(runner, config, schedule)
        await self._heartbeat_trigger.start()
        _log.event("heartbeat_registered", schedule=schedule,
                   active_hours=config.active_hours)

    async def register_cron(self, task_id: str, schedule: str):
        """Register a new cron trigger for a task."""
        if task_id in self._cron_triggers:
            await self._cron_triggers[task_id].stop()
        trigger = CronTrigger(task_id, schedule, self.executor, self.store)
        self._cron_triggers[task_id] = trigger
        await trigger.start()
        _log.event("cron_registered", task_id=task_id, schedule=schedule)

    async def register_once(self, task_id: str, delay_seconds: int):
        """Register a one-shot delayed trigger and persist fire_at."""
        from datetime import timedelta
        if task_id in self._once_triggers:
            await self._once_triggers[task_id].stop()

        fire_at = _now_local() + timedelta(seconds=delay_seconds)

        # Persist fire_at into trigger_config so it survives restarts
        task = self.store.get_task(task_id)
        if task:
            config = task.get("trigger_config", {}) or {}
            config["fire_at"] = fire_at.isoformat()
            config["delay_seconds"] = delay_seconds
            self.store.update_task(task_id,
                                   trigger_config=config,
                                   next_run_at=fire_at.isoformat())

        trigger = OnceTrigger(task_id, self.executor, self.store, fire_at=fire_at)
        self._once_triggers[task_id] = trigger
        await trigger.start()
        _log.event("once_registered", task_id=task_id,
                   fire_at=fire_at.isoformat(), delay=delay_seconds)

    async def unregister(self, task_id: str):
        """Stop and remove a trigger."""
        if task_id in self._cron_triggers:
            await self._cron_triggers[task_id].stop()
            del self._cron_triggers[task_id]
            _log.event("cron_unregistered", task_id=task_id)
        if task_id in self._once_triggers:
            await self._once_triggers[task_id].stop()
            del self._once_triggers[task_id]

    async def reload_task(self, task_id: str):
        """Reload a single task's trigger (after update/resume)."""
        task = self.store.get_task(task_id)
        if not task:
            await self.unregister(task_id)
            return

        if task.get("status") != "active":
            await self.unregister(task_id)
            return

        if task.get("trigger_type") == "cron":
            config = task.get("trigger_config", {})
            schedule = config.get("schedule", "")
            if schedule:
                await self.register_cron(task_id, schedule)
