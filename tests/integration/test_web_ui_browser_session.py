"""Real-browser regression for session rotation and targeted suggestion routing."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import uvicorn

from liagent.config import AppConfig

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402


class _FakeBrowserMetrics:
    def weekly_summary(self, *, days: int = 7):
        return {
            "days": days,
            "turns": 0,
            "tool_calls": 0,
            "success_rate": 1.0,
            "avg_latency_ms": 0.0,
        }


class _FakeBrowserEngine:
    def __init__(self):
        self.config = AppConfig()
        self.config.save = lambda: None

    def llm_status(self):
        return "llm:ready"

    def tts_status(self):
        return "tts:ready"

    def stt_status(self):
        return "stt:ready"

    def health_check(self):
        return {"status": "ok"}

    def configure_runtime_adapters(self):
        return None


class _FakeBrowserBrain:
    def __init__(self):
        self.feedback_calls: list[dict] = []
        self.clear_calls: list[str | None] = []
        self.suggestion_feedback_calls: list[dict] = []
        self.tool_policy = SimpleNamespace(recent_audit=lambda limit=50: [])
        self.suggestion_store = None
        self.proactive_router = None

    async def record_user_feedback(self, **kwargs):
        self.feedback_calls.append(kwargs)

    async def clear_memory_for_session(self, session_key: str | None = None):
        self.clear_calls.append(session_key)

    async def finalize_session(self, *, session_key: str | None = None):
        return None

    async def resolve_confirmation(self, token: str, approved: bool, force: bool = False, *, session_key: str | None = None):
        return {
            "status": "ok",
            "token": token,
            "approved": approved,
            "force": force,
            "session_key": session_key,
        }

    async def shutdown(self):
        return None


class _BrowserHarness:
    def __init__(self):
        self.engine = _FakeBrowserEngine()
        self.brain = _FakeBrowserBrain()
        self.loop = None
        self.stream_calls: list[dict] = []

    async def fake_stream_events(
        self,
        ws,
        _brain,
        prompt_text: str,
        run_id: str,
        *,
        session_key: str | None = None,
        ws_lock,
        **_kwargs,
    ):
        self.loop = asyncio.get_running_loop()
        self.stream_calls.append(
            {
                "prompt": prompt_text,
                "run_id": run_id,
                "session_key": session_key,
            }
        )
        async with ws_lock:
            await ws.send_json({"type": "run_state", "state": "streaming", "run_id": run_id})
            await ws.send_json({"type": "token", "text": f"reply:{prompt_text}"})
            await ws.send_json(
                {
                    "type": "done",
                    "text": f"reply:{prompt_text}",
                    "run_id": run_id,
                    "quality": {"confidence_label": "high", "confidence_note": "browser-test"},
                }
            )

    async def fake_apply_suggestion_feedback(
        self,
        suggestion_id: int,
        action: str,
        *,
        session_id: str | None = None,
    ):
        self.brain.suggestion_feedback_calls.append(
            {
                "suggestion_id": suggestion_id,
                "action": action,
                "session_id": session_id,
            }
        )


@contextmanager
def _run_test_server():
    from liagent.ui import task_routes, web_server
    import liagent.ui.shared_state as _shared

    harness = _BrowserHarness()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(
        web_server.app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        lifespan="on",
        ws="websockets-sansio",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)

    with ExitStack() as stack:
        stack.enter_context(patch.object(web_server, "_WEB_AUTH_TOKEN", ""))
        stack.enter_context(patch.object(task_routes, "_WEB_AUTH_TOKEN", ""))
        stack.enter_context(patch.object(_shared, "_WEB_AUTH_TOKEN", ""))
        stack.enter_context(patch.object(_shared, "_engine", harness.engine))
        stack.enter_context(patch.object(_shared, "_brain", harness.brain))
        stack.enter_context(patch.object(_shared, "_metrics", _FakeBrowserMetrics()))
        stack.enter_context(patch.object(_shared, "_signal_poller", None))
        stack.enter_context(patch.object(_shared, "_anomaly_detector", None))
        stack.enter_context(patch.object(_shared, "_orchestrator", None))
        stack.enter_context(patch.object(_shared, "_BRAIN_RUN_LOCK", asyncio.Lock()))
        stack.enter_context(patch.object(web_server, "_stream_events", new=harness.fake_stream_events))
        stack.enter_context(
            patch.object(web_server, "_apply_suggestion_feedback", new=harness.fake_apply_suggestion_feedback)
        )
        _shared._chat_clients.clear()
        _shared._chat_client_sessions.clear()
        _shared._http_rate.clear()
        _shared._ws_rate.clear()
        async def _clear_push_clients():
            async with _shared._push_clients_lock:
                _shared._push_clients.clear()
                _shared._push_client_sessions.clear()
        asyncio.run(_clear_push_clients())

        thread.start()
        deadline = time.time() + 10
        while time.time() < deadline:
            if getattr(server, "started", False):
                break
            time.sleep(0.05)
        if not getattr(server, "started", False):
            server.should_exit = True
            thread.join(timeout=5)
            raise RuntimeError("uvicorn test server failed to start")

        try:
            yield f"http://127.0.0.1:{port}", harness
        finally:
            server.should_exit = True
            thread.join(timeout=10)


def test_browser_session_rotation_and_targeted_suggestions():
    from liagent.ui import task_routes

    with _run_test_server() as (base_url, harness):
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(headless=True)
            except PlaywrightError as exc:
                pytest.skip(f"playwright browser unavailable: {exc}")

            page = browser.new_page()
            page.goto(base_url, wait_until="networkidle")
            page.wait_for_selector("#send-btn")
            page.wait_for_function(
                "() => import('/static/js/ws-send.js').then((m) => !!(m.getWs() && m.getWs().readyState === 1))"
            )

            page.evaluate(
                """
                async () => {
                  document.getElementById('user-input').value = 'alpha';
                  const mod = await import('/static/js/renderers/settings-panel.js');
                  mod.sendMessage();
                }
                """
            )

            deadline = time.time() + 5
            while not harness.stream_calls and time.time() < deadline:
                time.sleep(0.05)
            assert harness.stream_calls and harness.stream_calls[0]["prompt"] == "alpha"

            page.wait_for_selector(".feedback-bar", state="attached")
            session_1 = page.evaluate("sessionStorage.getItem('liagent_web_session_key')")
            assert session_1
            page.locator(".msg.assistant").last.hover()
            page.locator(".feedback-bar .fb-btn").first.click()

            deadline = time.time() + 5
            while len(harness.brain.feedback_calls) < 1 and time.time() < deadline:
                time.sleep(0.05)
            assert harness.brain.feedback_calls[0]["session_key"] == session_1

            page.click("#btn-clear")
            page.wait_for_function(
                "(oldKey) => sessionStorage.getItem('liagent_web_session_key') !== oldKey",
                arg=session_1,
            )
            session_2 = page.evaluate("sessionStorage.getItem('liagent_web_session_key')")
            assert session_2 and session_2 != session_1

            old_push = asyncio.run_coroutine_threadsafe(
                task_routes.broadcast_task_result(
                    {
                        "type": "proactive_suggestion",
                        "message": "old-session-suggestion",
                        "suggestion_id": 101,
                        "target_session_id": session_1,
                    }
                ),
                harness.loop,
            ).result(timeout=5)
            assert old_push == 0
            page.wait_for_timeout(300)
            assert page.locator("text=old-session-suggestion").count() == 0

            page.evaluate(
                """
                async () => {
                  document.getElementById('user-input').value = 'beta';
                  const mod = await import('/static/js/renderers/settings-panel.js');
                  mod.sendMessage();
                }
                """
            )

            deadline = time.time() + 5
            while len(harness.stream_calls) < 2 and time.time() < deadline:
                time.sleep(0.05)
            assert harness.stream_calls[-1]["prompt"] == "beta"

            page.wait_for_selector(".feedback-bar", state="attached")

            new_push = asyncio.run_coroutine_threadsafe(
                task_routes.broadcast_task_result(
                    {
                        "type": "proactive_suggestion",
                        "message": "new-session-suggestion",
                        "suggestion_id": 202,
                        "target_session_id": session_2,
                    }
                ),
                harness.loop,
            ).result(timeout=5)
            assert new_push == 1
            page.wait_for_selector("text=new-session-suggestion")
            page.locator("button:has-text('accept')").last.click()

            deadline = time.time() + 5
            while len(harness.brain.suggestion_feedback_calls) < 1 and time.time() < deadline:
                time.sleep(0.05)
            assert harness.brain.clear_calls == [session_1]
            assert harness.stream_calls[0]["session_key"] == session_1
            assert harness.stream_calls[-1]["session_key"] == session_2
            assert harness.brain.suggestion_feedback_calls[-1]["session_id"] == session_2
            assert harness.brain.suggestion_feedback_calls[-1]["suggestion_id"] == 202
            browser.close()
