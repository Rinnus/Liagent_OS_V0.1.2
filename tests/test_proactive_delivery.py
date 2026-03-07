"""Regression tests for proactive suggestion delivery and shutdown projection."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import liagent.ui.shared_state as _shared


class _FakeWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, payload: dict):
        self.sent.append(payload)


class _FakeSuggestionStore:
    def __init__(self, suggestion: dict | None = None):
        self.suggestion = suggestion or {}
        self.suggestion.setdefault("status", "pending")
        self.mark_shown_calls: list[tuple[int, int | None]] = []
        self.status_updates: list[tuple[int, str]] = []
        self.transitions: list[tuple[int, str, str]] = []

    def mark_shown(self, suggestion_id: int, *, cooldown_sec=None):
        self.mark_shown_calls.append((suggestion_id, cooldown_sec))

    def get_suggestion(self, suggestion_id: int):
        return self.suggestion

    def update_status(self, suggestion_id: int, status: str):
        self.suggestion["status"] = status
        self.status_updates.append((suggestion_id, status))

    def transition_status(self, suggestion_id: int, from_status: str, to_status: str):
        self.transitions.append((suggestion_id, from_status, to_status))
        if self.suggestion.get("status") != from_status:
            return False
        self.suggestion["status"] = to_status
        return True


class _FakeDomainFeedback:
    def __init__(self):
        self.suggested: list[tuple[str, str]] = []
        self.accepted: list[tuple[str, str]] = []
        self.rejected: list[tuple[str, str]] = []

    def record_suggested(self, domain: str, suggestion_type: str):
        self.suggested.append((domain, suggestion_type))

    def record_accepted_outcome(self, domain: str, suggestion_type: str):
        self.accepted.append((domain, suggestion_type))

    def record_rejected_outcome(self, domain: str, suggestion_type: str):
        self.rejected.append((domain, suggestion_type))


class _FakeTaskStore:
    def __init__(self, *, should_fail: bool = False):
        self.should_fail = should_fail
        self.created: list[dict] = []

    def create_task_from_prompt(
        self, *, prompt: str, source: str, goal_id, risk_level: str, dedup_key=None,
    ):
        if self.should_fail:
            raise RuntimeError("boom")
        task = {
            "id": f"task-{len(self.created) + 1}",
            "prompt": prompt,
            "source": source,
            "goal_id": goal_id,
            "risk_level": risk_level,
            "dedup_key": dedup_key,
        }
        self.created.append(task)
        return task


class _FakeGoalStore:
    def __init__(self):
        self.created: list[dict] = []
        self.transitions: list[tuple[int, str]] = []
        self.events: list[tuple[int, str, dict]] = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return len(self.created)

    def transition(self, goal_id: int, state: str):
        self.transitions.append((goal_id, state))

    def record_event(self, goal_id: int, event_type: str, payload: dict):
        self.events.append((goal_id, event_type, payload))


class ProactiveDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._orig_brain = _shared._brain
        self._orig_orchestrator = _shared._orchestrator
        self._orig_signal_poller = _shared._signal_poller
        self._orig_anomaly_detector = _shared._anomaly_detector
        self._orig_goal_store = getattr(_shared, "_goal_store", None)
        self._orig_task_store = getattr(_shared, "_task_store", None)
        self._orig_trigger_mgr = getattr(_shared, "_trigger_mgr", None)
        self._orig_chat_clients = dict(_shared._chat_clients)
        self._orig_chat_sessions = dict(_shared._chat_client_sessions)
        async with _shared._push_clients_lock:
            self._orig_push_clients = set(_shared._push_clients)
            self._orig_push_sessions = dict(_shared._push_client_sessions)
            _shared._push_clients.clear()
            _shared._push_client_sessions.clear()
        _shared._chat_clients.clear()
        _shared._chat_client_sessions.clear()

    async def asyncTearDown(self):
        _shared._brain = self._orig_brain
        _shared._orchestrator = self._orig_orchestrator
        _shared._signal_poller = self._orig_signal_poller
        _shared._anomaly_detector = self._orig_anomaly_detector
        _shared._goal_store = self._orig_goal_store
        _shared._task_store = self._orig_task_store
        _shared._trigger_mgr = self._orig_trigger_mgr
        _shared._chat_clients.clear()
        _shared._chat_clients.update(self._orig_chat_clients)
        _shared._chat_client_sessions.clear()
        _shared._chat_client_sessions.update(self._orig_chat_sessions)
        async with _shared._push_clients_lock:
            _shared._push_clients.clear()
            _shared._push_clients.update(self._orig_push_clients)
            _shared._push_client_sessions.clear()
            _shared._push_client_sessions.update(self._orig_push_sessions)

    async def test_broadcast_pending_suggestion_marks_shown_only_after_delivery(self):
        from liagent.ui.web_server import _broadcast_pending_suggestion

        feedback = _FakeDomainFeedback()
        store = _FakeSuggestionStore()
        brain = SimpleNamespace(
            suggestion_store=store,
            proactive_router=SimpleNamespace(_domain_feedback=feedback),
        )
        suggestion = {
            "id": 7,
            "message": "Check AAPL again",
            "domain": "stock",
            "suggestion_type": "watch",
        }
        with patch("liagent.ui.web_server.broadcast_task_result", new=AsyncMock(return_value=0)):
            delivered = await _broadcast_pending_suggestion(brain, suggestion)
        self.assertFalse(delivered)
        self.assertEqual(store.mark_shown_calls, [])
        self.assertEqual(feedback.suggested, [])

        with patch("liagent.ui.web_server.broadcast_task_result", new=AsyncMock(return_value=2)):
            delivered = await _broadcast_pending_suggestion(brain, suggestion)
        self.assertTrue(delivered)
        self.assertEqual(store.mark_shown_calls, [(7, 300)])
        self.assertEqual(feedback.suggested, [("stock", "watch")])

    async def test_apply_suggestion_feedback_records_accept_only_after_success(self):
        from liagent.ui.web_server import _apply_suggestion_feedback

        feedback = _FakeDomainFeedback()
        suggestion = {
            "id": 9,
            "message": "Run follow-up task",
            "domain": "coding",
            "suggestion_type": "watch",
            "action_json": '{"prompt":"run tests","risk":"low"}',
            "confidence": 0.8,
        }
        store = _FakeSuggestionStore(suggestion)
        trigger_mgr = SimpleNamespace(register_once=AsyncMock())
        _shared._task_store = _FakeTaskStore()
        _shared._goal_store = None
        _shared._trigger_mgr = trigger_mgr
        _shared._brain = SimpleNamespace(
            suggestion_store=store,
            proactive_router=SimpleNamespace(_domain_feedback=feedback),
        )

        await _apply_suggestion_feedback(9, "accept")
        self.assertEqual(store.status_updates, [(9, "accepted")])
        self.assertEqual(feedback.accepted, [("coding", "watch")])
        trigger_mgr.register_once.assert_awaited_once_with("task-1", delay_seconds=0)

    async def test_apply_suggestion_feedback_is_idempotent(self):
        from liagent.ui.web_server import _apply_suggestion_feedback

        feedback = _FakeDomainFeedback()
        suggestion = {
            "id": 12,
            "message": "Run follow-up task",
            "domain": "coding",
            "suggestion_type": "watch",
            "action_json": '{"prompt":"run tests","risk":"low"}',
            "confidence": 0.8,
        }
        store = _FakeSuggestionStore(suggestion)
        trigger_mgr = SimpleNamespace(register_once=AsyncMock())
        task_store = _FakeTaskStore()
        _shared._task_store = task_store
        _shared._goal_store = None
        _shared._trigger_mgr = trigger_mgr
        _shared._brain = SimpleNamespace(
            suggestion_store=store,
            proactive_router=SimpleNamespace(_domain_feedback=feedback),
        )

        await _apply_suggestion_feedback(12, "accept")
        await _apply_suggestion_feedback(12, "accept")
        self.assertEqual(len(task_store.created), 1)
        self.assertEqual(store.status_updates, [(12, "accepted")])
        self.assertEqual(feedback.accepted, [("coding", "watch")])

    async def test_apply_suggestion_feedback_create_goal_preserves_semantics(self):
        from liagent.ui.web_server import _apply_suggestion_feedback

        feedback = _FakeDomainFeedback()
        suggestion = {
            "id": 13,
            "message": "Should I set up ongoing monitoring?",
            "domain": "finance",
            "suggestion_type": "watch",
            "action_json": (
                '{"create_goal": true, "objective": "Track AAPL", '
                '"rationale": "Repeated interest", "idempotency_key": "goal:aapl"}'
            ),
            "confidence": 0.9,
        }
        store = _FakeSuggestionStore(suggestion)
        goal_store = _FakeGoalStore()
        _shared._task_store = _FakeTaskStore()
        _shared._goal_store = goal_store
        _shared._trigger_mgr = SimpleNamespace(register_once=AsyncMock())
        _shared._brain = SimpleNamespace(
            suggestion_store=store,
            proactive_router=SimpleNamespace(_domain_feedback=feedback),
        )

        await _apply_suggestion_feedback(13, "accept")
        self.assertEqual(len(goal_store.created), 1)
        self.assertEqual(goal_store.created[0]["objective"], "Track AAPL")
        self.assertEqual(goal_store.created[0]["idempotency_key"], "goal:aapl")
        self.assertEqual(goal_store.transitions, [(1, "active")])
        self.assertEqual(store.status_updates, [(13, "accepted")])

    async def test_apply_suggestion_feedback_does_not_record_accept_on_failure(self):
        from liagent.ui.web_server import _apply_suggestion_feedback

        feedback = _FakeDomainFeedback()
        suggestion = {
            "id": 11,
            "message": "Run follow-up task",
            "domain": "coding",
            "suggestion_type": "watch",
            "action_json": '{"prompt":"run tests","risk":"low"}',
            "confidence": 0.8,
        }
        store = _FakeSuggestionStore(suggestion)
        _shared._task_store = _FakeTaskStore(should_fail=True)
        _shared._goal_store = None
        _shared._trigger_mgr = SimpleNamespace(register_once=AsyncMock())
        _shared._brain = SimpleNamespace(
            suggestion_store=store,
            proactive_router=SimpleNamespace(_domain_feedback=feedback),
        )

        await _apply_suggestion_feedback(11, "accept")
        self.assertEqual(store.status_updates, [(11, "failed")])
        self.assertEqual(feedback.accepted, [])

    async def test_apply_suggestion_feedback_rejects_mismatched_target_session(self):
        from liagent.ui.web_server import _apply_suggestion_feedback

        store = _FakeSuggestionStore(
            {
                "id": 12,
                "message": "Monitor NVDA?",
                "status": "pending",
                "domain": "stock",
                "suggestion_type": "watch",
                "action_json": "{}",
                "target_session_id": "session-a",
            }
        )
        feedback = _FakeDomainFeedback()
        _shared._brain = SimpleNamespace(
            suggestion_store=store,
            proactive_router=SimpleNamespace(_domain_feedback=feedback),
        )

        await _apply_suggestion_feedback(12, "dismiss", session_id="session-b")

        self.assertEqual(store.suggestion["status"], "pending")
        self.assertEqual(store.transitions, [])
        self.assertEqual(feedback.rejected, [])

    async def test_web_lifespan_projects_after_shutdown(self):
        from liagent.ui.web_server import _lifespan

        call_order: list[str] = []

        async def brain_shutdown():
            call_order.append("brain_shutdown")

        _shared._brain = SimpleNamespace(
            shutdown=brain_shutdown,
            suggestion_store=None,
        )
        _shared._orchestrator = None
        _shared._signal_poller = None
        _shared._anomaly_detector = None

        def project_pending_events():
            call_order.append("project")
            return 0

        with patch("liagent.knowledge.projector.project_pending_events", side_effect=project_pending_events):
            async with _lifespan(SimpleNamespace()):
                await asyncio.sleep(0)

        self.assertEqual(call_order, ["brain_shutdown", "project"])

    async def test_broadcast_task_result_forwards_proactive_suggestion_to_push_and_chat(self):
        from liagent.ui.task_routes import broadcast_task_result

        push_ws = _FakeWS()
        chat_ws = _FakeWS()
        async with _shared._push_clients_lock:
            _shared._push_clients.add(push_ws)
        _shared._chat_clients[chat_ws] = asyncio.Lock()

        payload = {
            "type": "proactive_suggestion",
            "message": "Check TSLA",
            "suggestion_id": 5,
        }
        delivered = await broadcast_task_result(payload)

        self.assertEqual(delivered, 2)
        self.assertEqual(push_ws.sent, [payload])
        self.assertEqual(chat_ws.sent, [payload])


class DiscordPushDispatchTests(unittest.TestCase):
    def test_dispatch_push_message_routes_proactive_suggestion(self):
        from liagent.ui.discord_bot import LiAgentBot

        bot = SimpleNamespace(
            _deliver_push=AsyncMock(),
            _deliver_heartbeat_confirm=AsyncMock(),
            _deliver_proactive_suggestion=AsyncMock(),
        )

        asyncio.run(
            LiAgentBot._dispatch_push_message(
                bot, {"type": "proactive_suggestion", "message": "Check NVDA"},
            )
        )

        bot._deliver_push.assert_not_called()
        bot._deliver_heartbeat_confirm.assert_not_called()
        bot._deliver_proactive_suggestion.assert_awaited_once()

    def test_deliver_proactive_suggestion_uses_target_session_channel(self):
        from liagent.ui.discord_bot import LiAgentBot

        sent_embeds: list[tuple[object, object | None]] = []

        class _Channel:
            async def send(self, *, embed=None, view=None):
                sent_embeds.append((embed, view))

        channel = _Channel()
        bot = SimpleNamespace(
            _push_channel_id=999,
            _session_channels={"discord:g:1:c:22:u:3": 22},
            get_channel=lambda channel_id: channel if channel_id == 22 else None,
            fetch_channel=AsyncMock(return_value=None),
        )

        asyncio.run(
            LiAgentBot._deliver_proactive_suggestion(
                bot,
                {
                    "type": "proactive_suggestion",
                    "message": "Check NVDA",
                    "suggestion_id": 7,
                    "target_session_id": "discord:g:1:c:22:u:3",
                },
            )
        )

        self.assertEqual(len(sent_embeds), 1)
