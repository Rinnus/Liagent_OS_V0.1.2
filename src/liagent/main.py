"""LiAgent entry point."""

import asyncio
import os
import sys
import warnings
from pathlib import Path

from rich.console import Console

from .ui import shared_state as _shared

console = Console(highlight=False)


def _validate_local_model_paths(config) -> list[str]:
    """Validate required local model paths before engine initialization."""
    missing: list[str] = []
    if config.llm.backend == "local":
        p = Path(config.llm.local_model_path).expanduser()
        if not p.exists():
            missing.append(f"LLM local model not found: {p}")
    if config.tts_enabled and config.tts.backend == "local":
        p = Path(config.tts.local_model_path).expanduser()
        if not p.exists():
            missing.append(f"TTS local model not found: {p}")
        elif "kokoro" in str(p).lower():
            try:
                __import__("misaki")
            except Exception:
                missing.append(
                    "Kokoro dependency missing: misaki (install with `pip install \"misaki[zh]\"`)"
                )
            try:
                __import__("misaki.en")
            except Exception:
                missing.append(
                    "Kokoro dependency missing: misaki.en stack (install with `pip install \"misaki[en]\"`)"
                )
            try:
                __import__("misaki.espeak")
            except Exception:
                missing.append(
                    "Kokoro dependency missing: misaki.espeak/phonemizer (install with `pip install \"misaki[en]\"`)"
                )
            lang = str(getattr(config.tts, "language", "zh") or "zh").strip().lower()
            if lang in {"zh", "chinese", "zh-cn", "mandarin", "auto"}:
                try:
                    __import__("misaki.zh")
                except Exception:
                    missing.append(
                        "Kokoro zh dependency missing: misaki.zh (install with `pip install \"misaki[zh]\"`)"
                    )
    stt_backend = str(getattr(config.stt, "backend", "local") or "local").strip().lower()
    if stt_backend == "local":
        p = Path(config.stt.model).expanduser()
        if not p.exists():
            missing.append(f"STT local model not found: {p}")
    return missing


def _bootstrap_missing_local_models_to_api(config, missing: list[str]) -> list[str]:
    """Convert missing local backends to API backends for first-run usability.

    Returns human-readable migration notes. No-op when nothing is changed.
    """
    notes: list[str] = []

    llm_missing = any(msg.startswith("LLM local model not found:") for msg in missing)
    if llm_missing and config.llm.backend == "local":
        config.llm.backend = "api"
        config.llm.api_base_url = (
            str(config.llm.api_base_url).strip()
            or os.environ.get("LLM_API_BASE_URL", "https://api.openai.com/v1").strip()
        )
        config.llm.api_model = (
            str(config.llm.api_model).strip()
            or os.environ.get("LLM_API_MODEL", "gpt-4o").strip()
        )
        if not str(config.llm.api_key).strip():
            config.llm.api_key = os.environ.get("LLM_API_KEY", "").strip()
        if str(getattr(config, "runtime_mode", "hybrid_balanced") or "").strip().lower() != "local_private":
            config.runtime_mode = "cloud_performance"
        notes.append(f"LLM -> API ({config.llm.api_model})")

    tts_missing = any(msg.startswith("TTS local model not found:") for msg in missing)
    kokoro_dep_missing = any("Kokoro dependency missing" in msg for msg in missing)
    if (
        config.tts_enabled
        and config.tts.backend == "local"
        and (tts_missing or kokoro_dep_missing)
    ):
        config.tts.backend = "api"
        config.tts.api_base_url = (
            str(config.tts.api_base_url).strip()
            or os.environ.get("TTS_API_BASE_URL", "").strip()
            or os.environ.get("LLM_API_BASE_URL", "https://api.openai.com/v1").strip()
        )
        config.tts.api_model = (
            str(config.tts.api_model).strip()
            or os.environ.get("TTS_API_MODEL", "tts-1").strip()
        )
        config.tts.api_voice = (
            str(config.tts.api_voice).strip()
            or os.environ.get("TTS_API_VOICE", "alloy").strip()
        )
        if not str(config.tts.api_key).strip():
            config.tts.api_key = (
                os.environ.get("TTS_API_KEY", "").strip()
                or os.environ.get("LLM_API_KEY", "").strip()
            )
        notes.append(f"TTS -> API ({config.tts.api_model})")

    stt_missing = any(msg.startswith("STT local model not found:") for msg in missing)
    stt_backend = str(getattr(config.stt, "backend", "local") or "local").strip().lower()
    if stt_missing and stt_backend == "local":
        config.stt.backend = "api"
        config.stt.api_base_url = (
            str(getattr(config.stt, "api_base_url", "") or "").strip()
            or os.environ.get("STT_API_BASE_URL", "").strip()
            or os.environ.get("LLM_API_BASE_URL", "https://api.openai.com/v1").strip()
        )
        config.stt.api_model = (
            str(getattr(config.stt, "api_model", "") or "").strip()
            or os.environ.get("STT_API_MODEL", "gpt-4o-mini-transcribe").strip()
        )
        if not str(getattr(config.stt, "api_key", "") or "").strip():
            config.stt.api_key = (
                os.environ.get("STT_API_KEY", "").strip()
                or os.environ.get("LLM_API_KEY", "").strip()
            )
        notes.append(f"STT -> API ({config.stt.api_model})")

    if notes:
        config.save()
    return notes


def main():
    from dotenv import load_dotenv
    load_dotenv()
    if os.environ.get("LIAGENT_SUPPRESS_RESOURCE_TRACKER_WARNING", "1").strip().lower() in {"1", "true", "yes"}:
        warnings.filterwarnings(
            "ignore",
            message=r"resource_tracker: There appear to be \d+ leaked semaphore objects to clean up at shutdown",
            category=UserWarning,
        )
    if os.environ.get("LIAGENT_SUPPRESS_JIEBA_PKG_RESOURCES_WARNING", "1").strip().lower() in {"1", "true", "yes"}:
        warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API\.",
            category=UserWarning,
            module=r"jieba\._compat",
        )

    web_mode = "--web" in sys.argv
    discord_mode = "--discord" in sys.argv
    port = 8080
    host = "127.0.0.1"
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
        if arg == "--host" and i + 1 < len(sys.argv):
            host = sys.argv[i + 1]

    # Discord bot mode — thin client, no model loading
    if discord_mode:
        token = os.environ.get("LIAGENT_DISCORD_TOKEN", "").strip()
        ws_secret = os.environ.get("LIAGENT_WEB_TOKEN", "").strip()
        ws_url = os.environ.get("LIAGENT_WS_URL", "ws://localhost:8080/ws/chat").strip()
        if not token:
            console.print("  [white]LIAGENT_DISCORD_TOKEN not set[/]")
            sys.exit(1)
        from .ui.discord_bot import create_discord_bot
        bot = create_discord_bot(ws_url, ws_secret)
        console.print(f"  [dim]connecting to {ws_url}...[/]")
        bot.run(token)
        return

    console.print("  [dim]loading engines...[/]")

    from .config import AppConfig

    config = AppConfig.load()
    cwork_root = str(Path(config.tasks.cwork_dir).expanduser().resolve())
    os.environ["LIAGENT_CWORK_DIR"] = cwork_root
    os.environ["LIAGENT_CWORK_ROOT"] = cwork_root
    missing_models = _validate_local_model_paths(config)
    if missing_models:
        bootstrap_enabled = os.environ.get(
            "LIAGENT_BOOTSTRAP_API_ON_MISSING", "1"
        ).strip().lower() in {"1", "true", "yes"}
        if bootstrap_enabled:
            notes = _bootstrap_missing_local_models_to_api(config, missing_models)
            if notes:
                console.print("  [dim]local model check failed; switched to API bootstrap mode[/]")
                for note in notes:
                    console.print(f"  [dim]- {note}[/]")
                console.print(
                    "  [dim]tip: open /settings and fill API keys to complete setup[/]"
                )
                missing_models = _validate_local_model_paths(config)
        if missing_models:
            console.print("  [white]model path check failed:[/]")
            for item in missing_models:
                console.print(f"  [dim]- {item}[/]")
            console.print(
                "  [dim]tip: check LIAGENT_MODELS_DIR or update model paths in config.json[/]"
            )
            sys.exit(1)

    try:
        from .engine.engine_manager import EngineManager

        engine = EngineManager(config)
    except Exception as e:
        console.print(f"  [white]load failed:[/] [dim]{e}[/]")
        console.print("  [dim]tip: use /settings to switch to API mode[/]")
        sys.exit(1)

    console.print("  [dim]ready[/]")

    if web_mode:
        from .agent.brain import AgentBrain
        from .ui.web_server import create_app, broadcast_task_result

        brain = AgentBrain(engine)
        app = create_app(engine, brain)

        # Initialize notification router
        try:
            from .agent.notification import ChannelRouter, CLIPushChannel, SystemNotifyChannel
            brain.notification_router = ChannelRouter(
                channels=[CLIPushChannel(), SystemNotifyChannel()]
            )
        except Exception:
            brain.notification_router = None

        # Initialize autonomous task system
        if config.tasks.enabled:
            from .agent.task_queue import TaskStore, TaskExecutor
            from .agent.triggers import TriggerManager
            from .ui.shared_state import _BRAIN_RUN_LOCK

            task_store = TaskStore()
            executor = TaskExecutor(
                brain, task_store, _BRAIN_RUN_LOCK,
                on_result=broadcast_task_result,
            )
            trigger_manager = TriggerManager(executor, task_store, cwork_dir=config.tasks.cwork_dir)

            def _ensure_heartbeat_task(store) -> str | None:
                """Ensure a single system heartbeat task exists, return its id."""
                existing = store.get_active_tasks_by_trigger("heartbeat")
                if existing:
                    return existing[0]["id"]
                task = store.create_task(
                    name="Heartbeat Executor",
                    trigger_type="heartbeat",
                    trigger_config={"system": True},
                    prompt_template="",
                    priority=8,
                )
                return task["id"]

            # Initialize heartbeat if HEARTBEAT.md exists
            _heartbeat_md = Path.home() / ".liagent" / "HEARTBEAT.md"
            if _heartbeat_md.exists():
                try:
                    from .agent.heartbeat import (
                        parse_heartbeat_md, HeartbeatRunner, CursorStore, ExecuteResult,
                    )
                    from .skills.router import BudgetOverride

                    _hb_text = _heartbeat_md.read_text()
                    _hb_config = parse_heartbeat_md(_hb_text)
                    _hb_db = Path.home() / ".liagent" / "heartbeat.db"
                    _hb_store = CursorStore(db_path=_hb_db)

                    # Ensure system heartbeat task
                    _hb_task_id = _ensure_heartbeat_task(task_store)

                    # Build callbacks that bind to TaskStore + TaskExecutor
                    async def _on_hb_execute(action, prompt, allowed_tools):
                        run = task_store.create_run(
                            _hb_task_id,
                            trigger_event=f"heartbeat:{action.action_key}",
                            prompt=prompt,
                        )
                        budget = BudgetOverride(
                            allowed_tools={action.tool_name} if action.tool_name else None,
                            max_tool_calls=1,
                            timeout_ms=30_000,
                        )
                        executor.enqueue(
                            run["id"], _hb_task_id, prompt,
                            priority=8,
                            trigger_event=f"heartbeat:{action.action_key}",
                            budget=budget,
                            source="system",
                        )
                        return ExecuteResult(run_id=run["id"], status="queued")

                    async def _on_hb_needs_confirmation(action, prompt, allowed_tools, reason):
                        run = task_store.create_run(
                            _hb_task_id,
                            trigger_event=f"heartbeat:{action.action_key}",
                            prompt=prompt,
                        )
                        from datetime import timedelta
                        from .agent.time_utils import _now_local
                        expires_at = (_now_local() + timedelta(seconds=300)).isoformat()
                        task_store.update_run(
                            run["id"], status="pending_confirm", expires_at=expires_at,
                        )
                        # Broadcast to Web + Discord via task-push channel
                        await broadcast_task_result({
                            "type": "heartbeat_confirm",
                            "token": run["id"],
                            "action_type": action.tool_name,
                            "action_args": action.tool_args,
                            "description": action.description,
                            "risk_level": action.risk_level,
                            "expires_at": expires_at,
                            "allowed_tools": allowed_tools,
                            "prompt": prompt,
                        })
                        return run["id"]

                    _hb_runner = HeartbeatRunner(
                        config=_hb_config,
                        engine=engine,
                        long_term_memory=brain.long_term,
                        notification_router=getattr(brain, 'notification_router', None),
                        cursor_store=_hb_store,
                        on_action=brain.record_system_activity,
                        pattern_detector=getattr(brain, 'pattern_detector', None),
                        proactive_router=getattr(brain, 'proactive_router', None),
                        suggestion_store=getattr(brain, 'suggestion_store', None),
                        on_execute=_on_hb_execute,
                        on_needs_confirmation=_on_hb_needs_confirmation,
                    )
                    # Use cooldown_minutes as cron interval
                    _hb_schedule = f"*/{_hb_config.cooldown_minutes} * * * *"
                    app.state.heartbeat_runner = _hb_runner
                    app.state.heartbeat_config = _hb_config
                    app.state.heartbeat_schedule = _hb_schedule
                    app.state.heartbeat_task_id = _hb_task_id
                    # Wire anomaly detector → heartbeat pre-check
                    try:
                        from .agent.anomaly_precheck import AnomalyPreCheck
                        import liagent.ui.shared_state as _ws_shared
                        _anomaly_precheck = AnomalyPreCheck()
                        _hb_runner.pre_checks.append(_anomaly_precheck)
                        _ws_shared._anomaly_precheck = _anomaly_precheck
                    except Exception as _wire_exc:
                        console.print(f"  [yellow]anomaly→heartbeat wiring failed: {_wire_exc}[/]")
                    console.print(f"  [dim]heartbeat loaded (schedule: {_hb_schedule}, dry_run: {_hb_config.dry_run})[/]")
                except Exception as e:
                    console.print(f"  [yellow]heartbeat init failed: {e}[/]")

            # Attach to app.state for REST API access
            app.state.task_store = task_store
            app.state.executor = executor
            app.state.trigger_manager = trigger_manager

            # Wire up the create_task tool so Brain can create tasks
            from .tools.task_tool import configure as _configure_task_tool
            _configure_task_tool(task_store, trigger_manager, engine)

            # Start executor + triggers after uvicorn starts
            import contextlib
            from .ui.web_server import _lifespan as _ws_lifespan

            @contextlib.asynccontextmanager
            async def _lifespan(app_instance):
                # Nest web_server's lifespan (signal_poller, anomaly_detector, orchestrator, brain shutdown)
                async with _ws_lifespan(app_instance):
                    await executor.start()
                    await trigger_manager.start()
                    # Start heartbeat trigger
                    if hasattr(app_instance.state, 'heartbeat_runner'):
                        await trigger_manager.register_heartbeat(
                            app_instance.state.heartbeat_runner,
                            app_instance.state.heartbeat_config,
                            app_instance.state.heartbeat_schedule,
                        )
                    console.print("  [dim]autonomous task system ready[/]")
                    # Goal autonomy stores
                    from .agent.goal_store import GoalStore
                    goal_store = GoalStore(brain.long_term.db_path)
                    executor.goal_store = goal_store
                    # Expose to WS handler for suggestion accept → goal/task creation
                    _shared._goal_store = goal_store
                    _shared._task_store = task_store
                    _shared._trigger_mgr = trigger_manager

                    # Start bridge loop
                    from .agent.proactive_bridge import bridge_loop
                    _bridge_task = asyncio.create_task(
                        bridge_loop(
                            suggestion_store=brain.suggestion_store,
                            goal_store=goal_store,
                            task_store=task_store,
                            trigger_mgr=trigger_manager,
                            config=config,
                        )
                    )

                    # Start reflection loop
                    from .agent.goal_loop import reflection_loop
                    _reflection_task = asyncio.create_task(
                        reflection_loop(
                            goal_store=goal_store,
                            group_store=goal_store,
                            engine=engine,
                            pattern_detector=brain.pattern_detector,
                            brain=brain,
                            task_store=task_store,
                            trigger_mgr=trigger_manager,
                            suggestion_store=brain.suggestion_store,
                            config=config,
                        )
                    )
                    try:
                        yield
                    finally:
                        _bridge_task.cancel()
                        _reflection_task.cancel()
                        try:
                            await _bridge_task
                        except asyncio.CancelledError:
                            pass
                        try:
                            await _reflection_task
                        except asyncio.CancelledError:
                            pass
                        await trigger_manager.stop()
                        await executor.stop()

            app.router.lifespan_context = _lifespan
            console.print("  [dim]task system initialized[/]")

        import uvicorn

        visible_host = "localhost" if host == "127.0.0.1" else host
        if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("LIAGENT_WEB_TOKEN", "").strip():
            console.print("  [white]LIAGENT_WEB_TOKEN is required when --host exposes the server beyond localhost[/]")
            sys.exit(1)
        console.print(f"  [white]http://{visible_host}:{port}[/]")
        # Keep lifespan enabled so shutdown hooks can close MCP stdio cleanly.
        lifespan_mode = "on"
        try:
            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level="warning",
                lifespan=lifespan_mode,
                ws="websockets-sansio",
                ws_max_size=10 * 1024 * 1024,  # 10MB for voice messages
            )
        except KeyboardInterrupt:
            # Graceful terminal shutdown on Ctrl+C without noisy traceback.
            pass
        finally:
            # Defensive brain.shutdown() — idempotent (_shutdown_done guard),
            # safe even if lifespan already ran it.
            try:
                import asyncio as _aio
                _aio.run(brain.shutdown())
            except Exception:
                pass
            try:
                engine.shutdown()
            except Exception:
                pass
    else:
        from .ui.cli import run_cli

        try:
            asyncio.run(run_cli(engine))
        finally:
            try:
                engine.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
