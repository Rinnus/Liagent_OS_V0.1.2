import asyncio
import unittest

import liagent.ui.shared_state as _shared


class _FakeWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, payload: dict):
        self.sent.append(payload)


class ParseApprovedFlagTests(unittest.TestCase):
    def test_true_values(self):
        from liagent.ui.task_routes import _parse_approved_flag

        for value in (True, 1, "1", "true", "TRUE", "yes", "on"):
            self.assertTrue(_parse_approved_flag(value), value)

    def test_false_values(self):
        from liagent.ui.task_routes import _parse_approved_flag

        for value in (False, 0, "0", "false", "FALSE", "no", "", None):
            self.assertFalse(_parse_approved_flag(value), value)


class BroadcastTaskResultTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._orig_chat_clients = dict(_shared._chat_clients)
        self._orig_chat_sessions = dict(_shared._chat_client_sessions)
        self._orig_push_clients = set(_shared._push_clients)
        self._orig_push_sessions = dict(_shared._push_client_sessions)
        _shared._chat_clients.clear()
        _shared._chat_client_sessions.clear()
        async with _shared._push_clients_lock:
            _shared._push_clients.clear()
            _shared._push_client_sessions.clear()

    async def asyncTearDown(self):
        _shared._chat_clients.clear()
        _shared._chat_clients.update(self._orig_chat_clients)
        _shared._chat_client_sessions.clear()
        _shared._chat_client_sessions.update(self._orig_chat_sessions)
        async with _shared._push_clients_lock:
            _shared._push_clients.clear()
            _shared._push_clients.update(self._orig_push_clients)
            _shared._push_client_sessions.clear()
            _shared._push_client_sessions.update(self._orig_push_sessions)

    async def test_task_result_keeps_error_and_run_id_for_chat_clients(self):
        from liagent.ui.task_routes import broadcast_task_result

        ws = _FakeWS()
        _shared._chat_clients[ws] = asyncio.Lock()

        delivered = await broadcast_task_result(
            {
                "type": "task_result",
                "task_id": "t1",
                "task_name": "heartbeat",
                "run_id": "r1",
                "status": "error",
                "error": "boom",
                "trigger_event": "heartbeat:test",
            }
        )

        self.assertEqual(delivered, 1)
        self.assertEqual(ws.sent[0]["run_id"], "r1")
        self.assertEqual(ws.sent[0]["error"], "boom")

    async def test_targeted_proactive_suggestion_only_reaches_matching_chat_session(self):
        from liagent.ui.task_routes import broadcast_task_result

        push_ws = _FakeWS()
        chat_ws_match = _FakeWS()
        chat_ws_other = _FakeWS()
        async with _shared._push_clients_lock:
            _shared._push_clients.add(push_ws)
        _shared._chat_clients[chat_ws_match] = asyncio.Lock()
        _shared._chat_clients[chat_ws_other] = asyncio.Lock()
        _shared._chat_client_sessions[chat_ws_match] = "session-a"
        _shared._chat_client_sessions[chat_ws_other] = "session-b"

        delivered = await broadcast_task_result(
            {
                "type": "proactive_suggestion",
                "message": "Monitor NVDA?",
                "suggestion_id": 7,
                "target_session_id": "session-a",
            }
        )

        self.assertEqual(delivered, 1)
        self.assertEqual(push_ws.sent, [])
        self.assertEqual(len(chat_ws_match.sent), 1)
        self.assertEqual(chat_ws_other.sent, [])

    async def test_targeted_proactive_suggestion_only_reaches_matching_push_subscription(self):
        from liagent.ui.task_routes import broadcast_task_result

        push_ws_match = _FakeWS()
        push_ws_other = _FakeWS()
        async with _shared._push_clients_lock:
            _shared._push_clients.add(push_ws_match)
            _shared._push_clients.add(push_ws_other)
            _shared._push_client_sessions[push_ws_match] = {"session-a"}
            _shared._push_client_sessions[push_ws_other] = {"session-b"}

        delivered = await broadcast_task_result(
            {
                "type": "proactive_suggestion",
                "message": "Monitor AMD?",
                "suggestion_id": 11,
                "target_session_id": "session-a",
            }
        )

        self.assertEqual(delivered, 1)
        self.assertEqual(len(push_ws_match.sent), 1)
        self.assertEqual(push_ws_other.sent, [])
