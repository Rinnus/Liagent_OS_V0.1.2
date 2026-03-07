import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import WebSocketDisconnect

import liagent.ui.shared_state as _shared
from liagent.config import AppConfig


class _FakeChatWS:
    def __init__(self):
        self.sent: list[dict] = []
        self.client = SimpleNamespace(host="127.0.0.1")
        self.headers: dict[str, str] = {}
        self._recv_queue: asyncio.Queue = asyncio.Queue()
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def close(self, code=None, reason=None):
        self.closed = True

    async def receive_json(self):
        item = await self._recv_queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def send_json(self, payload: dict):
        self.sent.append(payload)

    async def push(self, item):
        await self._recv_queue.put(item)


class _FakeLongTermMemory:
    def __init__(self, **kwargs):
        self.db_path = kwargs.get("db_path") or Path(tempfile.gettempdir()) / "websocket-chat-test.db"
        self.feedback_calls: list[dict] = []

    def set_embedder(self, *args, **kwargs):
        return None

    def get_recent_summaries(self, *args, **kwargs):
        return []

    def get_all_facts(self, *args, **kwargs):
        return []

    def get_relevant_facts(self, *args, **kwargs):
        return []

    def get_relevant_evidence(self, *args, **kwargs):
        return []

    def decay_confidence(self, *args, **kwargs):
        return None

    def prune_memory(self, *args, **kwargs):
        return None

    def prune_old_records(self, *args, **kwargs):
        return None

    def save_feedback(self, **kwargs):
        self.feedback_calls.append(kwargs)

    def apply_source_confidence(self, facts):
        return facts

    def detect_conflicts(self, *args, **kwargs):
        return None

    def close(self):
        return None


class _FakeExperience:
    def __init__(self, *args, **kwargs):
        return None

    def match(self, query):
        return None

    def record_outcome(self, *args, **kwargs):
        return None

    def sync_from_markdown(self):
        return None

    def sync_to_markdown(self):
        return None

    async def generate_skill(self, *args, **kwargs):
        return None


class _FakeMetrics:
    def log_turn(self, *args, **kwargs):
        return None

    def log_runtime(self, *args, **kwargs):
        return None


class _RawCallEngine:
    def __init__(self):
        self.config = AppConfig()
        self.config.tts_enabled = False
        self.config.llm.max_tokens = 128
        self.config.llm.temperature = 0.2
        self.config.tool_profile = "research"
        self.config.mcp_discovery.enabled = False
        self.config.mcp_servers = []
        self.tts = None
        self.tool_parser = SimpleNamespace(
            parse=lambda *_a, **_kw: None,
            format_schemas=lambda tools: tools,
        )
        self._stream_responses = [
            ['I will search now.\n', 'web_search({"query": "GOOG stock", "timelimit": "d"})'],
            ["Support is around 300."],
        ]

    async def generate_llm_routed(self, *args, **kwargs):
        yield "OK"

    async def generate_llm(self, *args, **kwargs):
        yield "OK"

    async def generate_text(self, *args, **kwargs):
        yield "OK"

    async def stream_text(self, *args, **kwargs):
        for token in self._stream_responses.pop(0):
            yield token

    async def generate_extraction(self, *args, **kwargs):
        return "[]"

    async def generate_reasoning(self, *args, **kwargs):
        return "OK"


class WebSocketChatTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._orig_brain = _shared._brain
        self._orig_chat_clients = dict(_shared._chat_clients)
        self._orig_chat_sessions = dict(_shared._chat_client_sessions)
        _shared._chat_clients.clear()
        _shared._chat_client_sessions.clear()

    async def asyncTearDown(self):
        _shared._brain = self._orig_brain
        _shared._chat_clients.clear()
        _shared._chat_clients.update(self._orig_chat_clients)
        _shared._chat_client_sessions.clear()
        _shared._chat_client_sessions.update(self._orig_chat_sessions)

    async def _wait_for_sent(self, ws: _FakeChatWS, predicate, *, timeout: float = 1.0):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if predicate(ws.sent):
                return
            await asyncio.sleep(0.01)
        self.fail("timed out waiting for websocket payload")

    async def test_barge_in_is_processed_while_stream_task_is_running(self):
        from liagent.ui import web_server

        ws = _FakeChatWS()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def fake_stream_events(*args, **kwargs):
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with (
            patch.object(web_server, "_stream_events", new=fake_stream_events),
            patch.object(web_server, "_allow_rate", return_value=True),
            patch.object(web_server, "_WEB_AUTH_TOKEN", ""),
        ):
            _shared._brain = SimpleNamespace()
            chat_task = asyncio.create_task(web_server.ws_chat(ws))
            await ws.push({"type": "text", "text": "hello"})
            await asyncio.wait_for(started.wait(), timeout=1.0)
            await ws.push({"type": "barge_in"})
            await ws.push(WebSocketDisconnect())
            await asyncio.wait_for(chat_task, timeout=1.0)

        self.assertTrue(cancelled.is_set())
        self.assertTrue(any(msg.get("type") == "barge_in_ack" for msg in ws.sent))

    async def test_non_browser_ws_chat_is_not_registered_for_background_push(self):
        from liagent.ui import web_server

        ws = _FakeChatWS()
        started = asyncio.Event()

        async def fake_stream_events(*args, **kwargs):
            started.set()
            await asyncio.sleep(60)

        with (
            patch.object(web_server, "_stream_events", new=fake_stream_events),
            patch.object(web_server, "_allow_rate", return_value=True),
            patch.object(web_server, "_WEB_AUTH_TOKEN", ""),
        ):
            _shared._brain = SimpleNamespace()
            chat_task = asyncio.create_task(web_server.ws_chat(ws))
            await ws.push({"type": "text", "text": "hello"})
            await asyncio.wait_for(started.wait(), timeout=1.0)
            self.assertNotIn(ws, _shared._chat_clients)
            chat_task.cancel()
            await asyncio.wait_for(chat_task, timeout=1.0)

    async def test_tool_confirm_is_processed_while_stream_task_is_running(self):
        from liagent.ui import web_server

        ws = _FakeChatWS()
        started = asyncio.Event()
        resolve_confirmation = AsyncMock(return_value={"status": "rejected"})

        async def fake_stream_events(*args, **kwargs):
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                raise

        with (
            patch.object(web_server, "_stream_events", new=fake_stream_events),
            patch.object(web_server, "_allow_rate", return_value=True),
            patch.object(web_server, "_WEB_AUTH_TOKEN", ""),
        ):
            _shared._brain = SimpleNamespace(resolve_confirmation=resolve_confirmation)
            chat_task = asyncio.create_task(web_server.ws_chat(ws))
            await ws.push({"type": "text", "text": "hello"})
            await asyncio.wait_for(started.wait(), timeout=1.0)
            await ws.push(
                {
                    "type": "tool_confirm",
                    "token": "tok-1",
                    "approved": "false",
                    "session_key": "discord:g:1:c:2:u:3",
                }
            )
            await ws.push(WebSocketDisconnect())
            await asyncio.wait_for(chat_task, timeout=1.0)

        resolve_confirmation.assert_awaited_once_with(
            "tok-1",
            approved=False,
            force=False,
            session_key="discord:g:1:c:2:u:3",
        )
        self.assertTrue(any(msg.get("type") == "tool_confirm_result" for msg in ws.sent))

    async def test_clear_resets_target_repl_session_and_acknowledges(self):
        from liagent.ui import web_server

        ws = _FakeChatWS()
        clear_memory_for_session = AsyncMock()
        reset_repl_session = AsyncMock(return_value=True)

        with (
            patch.object(web_server, "_allow_rate", return_value=True),
            patch.object(web_server, "_WEB_AUTH_TOKEN", ""),
            patch("liagent.tools.stateful_repl.reset_repl_session", new=reset_repl_session),
        ):
            _shared._brain = SimpleNamespace(clear_memory_for_session=clear_memory_for_session)
            chat_task = asyncio.create_task(web_server.ws_chat(ws))
            await ws.push({"type": "clear", "session_key": "web:test-session"})
            await ws.push(WebSocketDisconnect())
            await asyncio.wait_for(chat_task, timeout=1.0)

        clear_memory_for_session.assert_awaited_once_with("web:test-session")
        reset_repl_session.assert_awaited_once_with("web:test-session")
        cleared = [msg for msg in ws.sent if msg.get("type") == "cleared"]
        self.assertEqual(len(cleared), 1)
        self.assertEqual(cleared[0]["session_key"], "web:test-session")
        self.assertTrue(cleared[0]["repl_reset"])

    async def test_suggestion_feedback_uses_payload_session_key(self):
        from liagent.ui import web_server

        ws = _FakeChatWS()
        apply_feedback = AsyncMock()

        with (
            patch.object(web_server, "_allow_rate", return_value=True),
            patch.object(web_server, "_WEB_AUTH_TOKEN", ""),
            patch.object(web_server, "_apply_suggestion_feedback", new=apply_feedback),
        ):
            _shared._brain = SimpleNamespace()
            chat_task = asyncio.create_task(web_server.ws_chat(ws))
            await ws.push(
                {
                    "type": "suggestion_feedback",
                    "id": 99,
                    "action": "accept",
                    "session_key": "discord:g:3:c:4:u:5",
                }
            )
            await ws.push(WebSocketDisconnect())
            await asyncio.wait_for(chat_task, timeout=1.0)

        apply_feedback.assert_awaited_once_with(
            99,
            "accept",
            session_id="discord:g:3:c:4:u:5",
        )

    async def test_feedback_uses_payload_session_key(self):
        from liagent.ui import web_server

        ws = _FakeChatWS()
        record_feedback = AsyncMock()

        with (
            patch.object(web_server, "_allow_rate", return_value=True),
            patch.object(web_server, "_WEB_AUTH_TOKEN", ""),
        ):
            _shared._brain = SimpleNamespace(record_user_feedback=record_feedback)
            chat_task = asyncio.create_task(web_server.ws_chat(ws))
            await ws.push(
                {
                    "type": "feedback",
                    "feedback": "positive",
                    "turn_index": 2,
                    "query": "q",
                    "answer": "a",
                    "tool_used": "web_search",
                    "session_key": "web:feedback-session",
                }
            )
            await ws.push(WebSocketDisconnect())
            await asyncio.wait_for(chat_task, timeout=1.0)

        record_feedback.assert_awaited_once_with(
            query="q",
            answer="a",
            tool_used="web_search",
            feedback="positive",
            turn_index=2,
            session_key="web:feedback-session",
        )

    async def test_ws_chat_forwards_sanitized_tool_events_without_leaking_raw_calls(self):
        from liagent.ui import web_server

        ws = _FakeChatWS()

        async def fake_run(*args, **kwargs):
            yield ("token", "I will search now.\n")
            yield ("tool_start", "web_search", {"query": "GOOG stock", "timelimit": "d"})
            yield ("tool_result", "web_search", "1. GOOG overview\nURL: https://example.com")
            yield ("done", "Support is around 300.")

        fake_engine = _RawCallEngine()
        fake_brain = SimpleNamespace(run=fake_run, engine=fake_engine)

        with (
            patch.object(web_server, "_allow_rate", return_value=True),
            patch.object(web_server, "_WEB_AUTH_TOKEN", ""),
            patch.object(_shared, "_brain", fake_brain),
            patch.object(_shared, "_engine", fake_engine),
            patch.object(_shared, "_metrics", _FakeMetrics()),
            patch.object(_shared, "_orchestrator", None),
            patch.object(_shared, "_BRAIN_RUN_LOCK", asyncio.Lock()),
        ):
            chat_task = asyncio.create_task(web_server.ws_chat(ws))
            await ws.push(
                {
                    "type": "text",
                    "text": "check GOOG support",
                    "session_key": "web:raw-call",
                }
            )
            await self._wait_for_sent(
                ws,
                lambda sent: any(msg.get("type") == "done" for msg in sent),
                timeout=5.0,
            )
            await ws.push(WebSocketDisconnect())
            await asyncio.wait_for(chat_task, timeout=1.0)

        token_text = "".join(msg.get("text", "") for msg in ws.sent if msg.get("type") == "token")
        self.assertNotIn('web_search({"query": "GOOG stock", "timelimit": "d"})', token_text)
        self.assertTrue(any(msg.get("type") == "tool_start" and msg.get("name") == "web_search" for msg in ws.sent))
        self.assertTrue(any(msg.get("type") == "tool_result" and msg.get("name") == "web_search" for msg in ws.sent))
        done_messages = [msg for msg in ws.sent if msg.get("type") == "done"]
        self.assertEqual(len(done_messages), 1)
        self.assertIn("Support is around 300.", done_messages[0]["text"])
