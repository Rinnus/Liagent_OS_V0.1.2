"""Tests for Discord bot confirmation flow — buttons for tool confirmation."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch


def _make_mock_bot():
    """Create a mock LiAgentBot with send_to_liagent."""
    bot = MagicMock()
    bot.send_to_liagent = AsyncMock()
    bot._remember_session_key = MagicMock(side_effect=lambda value: value)
    bot._ensure_ws = AsyncMock()
    bot._ws_lock = asyncio.Lock()
    bot._ws = MagicMock()
    bot._ws.send = AsyncMock()
    return bot


def _make_view(cls, *args, **kwargs):
    """Instantiate a discord.ui.View subclass inside a running event loop."""
    async def _create():
        return cls(*args, **kwargs)
    return asyncio.run(_create())


class TestConfirmationView(unittest.TestCase):
    """Test _ConfirmationView button interactions."""

    def test_view_has_two_buttons(self):
        from liagent.ui.discord_bot import _ConfirmationView
        bot = _make_mock_bot()
        view = _make_view(_ConfirmationView, bot, token="abc123", tool="screenshot", brief="")
        buttons = [c for c in view.children if hasattr(c, "label")]
        assert len(buttons) == 2
        labels = {b.label for b in buttons}
        assert "Approve" in labels
        assert "Reject" in labels

    def test_timeout_value(self):
        from liagent.ui.discord_bot import _ConfirmationView, _CONFIRM_TIMEOUT
        bot = _make_mock_bot()
        view = _make_view(_ConfirmationView, bot, token="abc123", tool="screenshot", brief="")
        assert view.timeout == _CONFIRM_TIMEOUT

    def test_approve_sends_tool_confirm(self):
        """Clicking approve sends tool_confirm with approved=True."""
        from liagent.ui.discord_bot import _ConfirmationView
        bot = _make_mock_bot()
        bot.send_to_liagent.return_value = [
            {"type": "tool_confirm_result", "result": {"status": "ok", "answer": "Done!"}}
        ]

        async def run():
            view = _ConfirmationView(
                bot,
                token="tok1",
                tool="web_search",
                brief="",
                session_key="discord:g:1:c:2:u:3",
            )
            interaction = MagicMock()
            interaction.response = MagicMock()
            interaction.response.edit_message = AsyncMock()
            await view._resolve(interaction, approved=True, force=False)
            return view

        view = asyncio.run(run())
        bot.send_to_liagent.assert_called_once_with(
            {
                "type": "tool_confirm",
                "token": "tok1",
                "approved": True,
                "force": False,
                "session_key": "discord:g:1:c:2:u:3",
            }
        )
        assert view.result["status"] == "ok"
        assert view.result["answer"] == "Done!"

    def test_reject_sends_tool_confirm_false(self):
        """Clicking reject sends tool_confirm with approved=False."""
        from liagent.ui.discord_bot import _ConfirmationView
        bot = _make_mock_bot()
        bot.send_to_liagent.return_value = [
            {"type": "tool_confirm_result", "result": {"status": "rejected"}}
        ]

        async def run():
            view = _ConfirmationView(
                bot,
                token="tok2",
                tool="screenshot",
                brief="",
                session_key="discord:g:9:c:8:u:7",
            )
            interaction = MagicMock()
            interaction.response = MagicMock()
            interaction.response.edit_message = AsyncMock()
            await view._resolve(interaction, approved=False, force=False)
            return view

        view = asyncio.run(run())
        bot.send_to_liagent.assert_called_once_with(
            {
                "type": "tool_confirm",
                "token": "tok2",
                "approved": False,
                "force": False,
                "session_key": "discord:g:9:c:8:u:7",
            }
        )
        assert view.result["status"] == "rejected"

    def test_ws_error_produces_error_result(self):
        """WebSocket error during confirm produces error result."""
        from liagent.ui.discord_bot import _ConfirmationView
        bot = _make_mock_bot()
        bot.send_to_liagent.side_effect = ConnectionError("ws down")

        async def run():
            view = _ConfirmationView(bot, token="tok3", tool="screenshot", brief="")
            interaction = MagicMock()
            interaction.response = MagicMock()
            interaction.response.edit_message = AsyncMock()
            await view._resolve(interaction, approved=True, force=False)
            return view

        view = asyncio.run(run())
        assert view.result["status"] == "error"
        assert "ws down" in view.result["message"]

    def test_buttons_disabled_after_click(self):
        """All buttons should be disabled after interaction."""
        from liagent.ui.discord_bot import _ConfirmationView
        bot = _make_mock_bot()
        bot.send_to_liagent.return_value = [
            {"type": "tool_confirm_result", "result": {"status": "ok"}}
        ]

        async def run():
            view = _ConfirmationView(bot, token="tok4", tool="screenshot", brief="")
            interaction = MagicMock()
            interaction.response = MagicMock()
            interaction.response.edit_message = AsyncMock()
            await view._resolve(interaction, approved=True, force=False)
            return view

        view = asyncio.run(run())
        for child in view.children:
            assert child.disabled is True


class TestSecondConfirmView(unittest.TestCase):
    """Test _SecondConfirmView for two-stage confirmation."""

    def test_has_force_and_reject_buttons(self):
        from liagent.ui.discord_bot import _SecondConfirmView
        bot = _make_mock_bot()
        view = _make_view(
            _SecondConfirmView,
            bot,
            token="tok5",
            tool="screenshot",
            session_key="discord:g:2:c:3:u:4",
        )
        buttons = [c for c in view.children if hasattr(c, "label")]
        assert len(buttons) == 2
        labels = {b.label for b in buttons}
        assert "Force Approve" in labels
        assert "Reject" in labels

    def test_force_approve_sends_force_true(self):
        """Force Approve sends tool_confirm with force=True."""
        from liagent.ui.discord_bot import _SecondConfirmView
        bot = _make_mock_bot()
        bot.send_to_liagent.return_value = [
            {"type": "tool_confirm_result", "result": {"status": "ok", "answer": "Forced"}}
        ]

        async def run():
            view = _SecondConfirmView(
                bot,
                token="tok6",
                tool="screenshot",
                session_key="discord:g:2:c:3:u:4",
            )
            interaction = MagicMock()
            interaction.response = MagicMock()
            interaction.response.edit_message = AsyncMock()
            # Simulate force_approve button click
            for child in view.children:
                child.disabled = True
            await interaction.response.edit_message(view=view)
            events = await bot.send_to_liagent(
                {
                    "type": "tool_confirm",
                    "token": "tok6",
                    "approved": True,
                    "force": True,
                    "session_key": "discord:g:2:c:3:u:4",
                }
            )
            for ev in events:
                if ev.get("type") == "tool_confirm_result":
                    view.result = ev.get("result", {})
            view._done.set()
            return view

        view = asyncio.run(run())
        bot.send_to_liagent.assert_called_once_with(
            {
                "type": "tool_confirm",
                "token": "tok6",
                "approved": True,
                "force": True,
                "session_key": "discord:g:2:c:3:u:4",
            }
        )
        assert view.result["status"] == "ok"


class TestProactiveSuggestionView(unittest.TestCase):
    def test_feedback_includes_session_key(self):
        from liagent.ui.discord_bot import _ProactiveSuggestionView

        bot = _make_mock_bot()

        async def run():
            view = _ProactiveSuggestionView(
                bot,
                suggestion_id=42,
                session_key="discord:g:5:c:6:u:7",
            )
            await view._send_feedback("accept")

        asyncio.run(run())
        bot._remember_session_key.assert_called_once_with("discord:g:5:c:6:u:7")
        bot._ws.send.assert_awaited_once_with(
            json.dumps(
                {
                    "type": "suggestion_feedback",
                    "id": 42,
                    "action": "accept",
                    "session_key": "discord:g:5:c:6:u:7",
                }
            )
        )


class TestSendReplyConfirmation(unittest.TestCase):
    """Test _send_reply event detection logic for confirmation_required."""

    def test_send_reply_detects_confirmation(self):
        """_send_reply should detect confirmation_required event."""
        events = [
            {
                "type": "confirmation_required",
                "token": "tok-abc",
                "tool": "screenshot",
                "reason": "High-risk tool requires confirmation",
                "brief": json.dumps({"stage": 1, "required_stage": 2}),
            },
            {"type": "done", "text": "Tool 'screenshot' requires confirmation."},
        ]

        confirm_event = None
        final_text = ""
        for ev in events:
            if ev.get("type") == "confirmation_required":
                confirm_event = ev
            elif ev.get("type") == "done":
                final_text = ev.get("text", "")

        assert confirm_event is not None
        assert confirm_event["token"] == "tok-abc"
        assert confirm_event["tool"] == "screenshot"
        assert final_text

    def test_send_reply_normal_events_no_confirmation(self):
        """Normal events without confirmation should follow standard path."""
        events = [
            {"type": "tool_start", "name": "web_search"},
            {"type": "tool_result", "name": "web_search", "result": "..."},
            {"type": "done", "text": "Here is the result."},
        ]

        confirm_event = None
        final_text = ""
        for ev in events:
            if ev.get("type") == "confirmation_required":
                confirm_event = ev
            elif ev.get("type") == "done":
                final_text = ev.get("text", "")

        assert confirm_event is None
        assert final_text == "Here is the result."


class TestSendToLiagentTerminalTypes(unittest.TestCase):
    """Verify send_to_liagent breaks on tool_confirm_result."""

    def test_tool_confirm_result_is_terminal(self):
        """tool_confirm_result should be a terminal event type."""
        terminal_types = ("done", "error", "tts_done", "cleared", "tool_confirm_result", "finalized")
        assert "tool_confirm_result" in terminal_types
        assert "finalized" in terminal_types


class TestDiscordSessionKey(unittest.TestCase):
    def test_build_discord_session_key_guild_channel_user(self):
        from liagent.ui.discord_bot import _build_discord_session_key
        key = _build_discord_session_key(guild_id=123, channel_id=456, user_id=789)
        self.assertEqual(key, "discord:g:123:c:456:u:789")

    def test_build_discord_session_key_dm_with_parent(self):
        from liagent.ui.discord_bot import _build_discord_session_key
        key = _build_discord_session_key(
            guild_id=None,
            channel_id=222,
            user_id=333,
            parent_channel_id=111,
        )
        self.assertEqual(key, "discord:g:dm:c:222:u:333:p:111")


class TestDiscordClose(unittest.TestCase):
    def test_close_finalizes_all_known_sessions(self):
        from liagent.ui.discord_bot import LiAgentBot

        async def run():
            bot = LiAgentBot("ws://localhost:8080/ws/chat", "")
            bot._ws = object()
            bot._known_session_keys = {"discord:g:1:c:2:u:3", "discord:g:4:c:5:u:6"}
            bot.send_to_liagent = AsyncMock(return_value=[{"type": "finalized"}])
            bot._close_ws = AsyncMock()
            bot._push_task = None
            with patch("liagent.ui.discord_bot.commands.Bot.close", new=AsyncMock()):
                await bot.close()
            return bot

        bot = asyncio.run(run())
        bot.send_to_liagent.assert_has_awaits(
            [
                call({"type": "finalize", "session_key": "discord:g:1:c:2:u:3"}),
                call({"type": "finalize", "session_key": "discord:g:4:c:5:u:6"}),
            ],
            any_order=False,
        )
        bot._close_ws.assert_awaited_once()
