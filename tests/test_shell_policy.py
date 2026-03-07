"""Tests for shell_exec policy integration — tier gating via policy gate."""
import asyncio
import time
import unittest
from unittest.mock import MagicMock


class ShellProfileTests(unittest.TestCase):
    """Test tool_profile gating for shell_exec."""

    def test_research_allows_shell(self):
        from liagent.tools.policy import _TOOL_PROFILE_MAP
        self.assertIn("shell_exec", _TOOL_PROFILE_MAP["research"])

    def test_minimal_blocks_shell(self):
        from liagent.tools.policy import _TOOL_PROFILE_MAP
        self.assertNotIn("shell_exec", _TOOL_PROFILE_MAP["minimal"])

    def test_full_allows_shell(self):
        from liagent.tools.policy import _TOOL_PROFILE_MAP
        self.assertIsNone(_TOOL_PROFILE_MAP["full"])


class ShellGrantKeyTests(unittest.TestCase):
    """Test fine-grained grant key for shell_exec dev commands."""

    def test_grant_key_format(self):
        from liagent.tools.shell_classify import grant_key
        self.assertEqual(grant_key(["git", "commit"]), "shell_exec:dev:git")
        self.assertEqual(grant_key(["pip", "install", "x"]), "shell_exec:dev:pip")
        self.assertEqual(grant_key(["npm", "run", "build"]), "shell_exec:dev:npm")
        self.assertEqual(grant_key(["uv", "sync"]), "shell_exec:dev:uv:sync")
        self.assertEqual(grant_key(["ruff", "check", "src"]), "shell_exec:dev:ruff:check")
        self.assertEqual(grant_key(["python", "-m", "pytest", "tests"]), "shell_exec:dev:python:-m:pytest")
        self.assertEqual(grant_key(["npx", "-y", "playwright", "test"]), "shell_exec:dev:npx:playwright")


class ShellTierGatingTests(unittest.TestCase):
    """Test that policy gate blocks dev/privileged shell commands."""

    def _make_shell_tool_def(self):
        """Get the registered shell_exec tool def."""
        import liagent.tools.shell_exec  # noqa: F401
        from liagent.tools import get_tool
        td = get_tool("shell_exec")
        self.assertIsNotNone(td)
        return td

    def _make_ctx(self, user_input="run a command"):
        from liagent.agent.run_context import RunContext
        ctx = RunContext(user_input=user_input)
        ctx.budget = MagicMock(enable_policy_review=False)
        return ctx

    def _mock_handle_policy_block(self):
        def handler(tool_name, tool_args, reason, full_response, hint_msg):
            return f"[Blocked] {reason}", "", [("policy_blocked", tool_name, reason)]
        return handler

    def _mock_tool_policy(self):
        policy = MagicMock()
        policy.evaluate.return_value = (True, "allowed")
        policy.confirmation_brief.return_value = {"message": "test"}
        policy.audit.return_value = None
        return policy

    def _run_policy_gate(self, command, tool_grants=None, confirmed=False):
        from liagent.agent.policy_gate import evaluate_tool_policy
        td = self._make_shell_tool_def()
        ctx = self._make_ctx()

        async def _test():
            return await evaluate_tool_policy(
                tool_name="shell_exec",
                tool_args={"command": command},
                tool_sig=f"shell_exec({command})",
                full_response=f"<tool_call>shell_exec({command})</tool_call>",
                confirmed=confirmed,
                ctx=ctx,
                tool_policy=self._mock_tool_policy(),
                planner=MagicMock(),
                handle_policy_block_fn=self._mock_handle_policy_block(),
                pending_confirmations={},
                dup_tool_limit=3,
                tool_cache_enabled=False,
                enable_policy_review=False,
                disable_policy_review_in_voice=True,
                tool_grants=tool_grants or {},
            )

        return asyncio.run(_test())

    def test_safe_command_allowed(self):
        """Safe commands (ls, cat) should pass policy gate."""
        decision = self._run_policy_gate("ls -la")
        self.assertTrue(decision.allowed)

    def test_dev_command_blocked_without_grant(self):
        """Dev commands without a session grant should require confirmation."""
        decision = self._run_policy_gate("git commit -m fix")
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.must_return)
        has_confirm = any(ev[0] == "confirmation_required" for ev in decision.events)
        self.assertTrue(has_confirm, f"Expected confirmation_required event, got {decision.events}")

    def test_dev_command_allowed_with_grant(self):
        """Dev commands with a valid session grant should pass."""
        from liagent.tools.shell_classify import grant_key
        gk = grant_key(["git", "commit"])
        grants = {gk: time.time() + 3600}
        decision = self._run_policy_gate("git commit -m fix", tool_grants=grants)
        self.assertTrue(decision.allowed)

    def test_dev_command_blocked_with_expired_grant(self):
        """Dev commands with an expired grant should require confirmation."""
        from liagent.tools.shell_classify import grant_key
        gk = grant_key(["git", "commit"])
        grants = {gk: time.time() - 1}
        decision = self._run_policy_gate("git commit -m fix", tool_grants=grants)
        self.assertFalse(decision.allowed)

    def test_scoped_dev_command_allowed_with_matching_grant(self):
        """Scoped dev commands should honor matching command-family grants only."""
        from liagent.tools.shell_classify import grant_key
        gk = grant_key(["uv", "sync"])
        grants = {gk: time.time() + 3600}
        decision = self._run_policy_gate("uv sync", tool_grants=grants)
        self.assertTrue(decision.allowed)

    def test_scoped_dev_command_blocked_with_other_subcommand_grant(self):
        """Grant for one scoped subcommand must not unlock a different subcommand."""
        from liagent.tools.shell_classify import grant_key
        gk = grant_key(["uv", "sync"])
        grants = {gk: time.time() + 3600}
        decision = self._run_policy_gate("uv run pytest", tool_grants=grants)
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.must_return)

    def test_python_module_grant_is_scoped(self):
        """python -m grants must not unlock other modules."""
        from liagent.tools.shell_classify import grant_key
        gk = grant_key(["python", "-m", "pytest", "tests"])
        grants = {gk: time.time() + 3600}
        allowed = self._run_policy_gate("python -m pytest tests/unit", tool_grants=grants)
        blocked = self._run_policy_gate("python -m http.server", tool_grants=grants)
        self.assertTrue(allowed.allowed)
        self.assertFalse(blocked.allowed)
        self.assertTrue(blocked.must_return)

    def test_npx_package_grant_is_scoped(self):
        """npx grants must not unlock other packages."""
        from liagent.tools.shell_classify import grant_key
        gk = grant_key(["npx", "-y", "playwright", "test"])
        grants = {gk: time.time() + 3600}
        allowed = self._run_policy_gate("npx -y playwright test", tool_grants=grants)
        blocked = self._run_policy_gate("npx -y cowsay hi", tool_grants=grants)
        self.assertTrue(allowed.allowed)
        self.assertFalse(blocked.allowed)
        self.assertTrue(blocked.must_return)

    def test_privileged_command_blocked(self):
        """Privileged commands should always require confirmation."""
        decision = self._run_policy_gate("sudo ls")
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.must_return)

    def test_privileged_two_step(self):
        """Privileged commands should require 2-step confirmation."""
        decision = self._run_policy_gate("rm -rf temp")
        has_confirm = any(ev[0] == "confirmation_required" for ev in decision.events)
        self.assertTrue(has_confirm)

    def test_confirmed_dev_passes(self):
        """Dev commands with confirmed=True should pass."""
        decision = self._run_policy_gate("git commit -m fix", confirmed=True)
        self.assertTrue(decision.allowed)

    def test_confirmed_privileged_passes(self):
        """Privileged commands with confirmed=True should pass."""
        decision = self._run_policy_gate("sudo ls", confirmed=True)
        self.assertTrue(decision.allowed)

    def test_pending_confirmation_has_grant_key(self):
        """Pending confirmation for dev commands should include shell_grant_key."""
        pending = {}

        from liagent.agent.policy_gate import evaluate_tool_policy
        td = self._make_shell_tool_def()
        ctx = self._make_ctx("git commit")

        async def _test():
            return await evaluate_tool_policy(
                tool_name="shell_exec",
                tool_args={"command": "git commit -m fix"},
                tool_sig="shell_exec(git commit -m fix)",
                full_response="<tool_call>shell_exec(git commit)</tool_call>",
                confirmed=False,
                ctx=ctx,
                tool_policy=self._mock_tool_policy(),
                planner=MagicMock(),
                handle_policy_block_fn=self._mock_handle_policy_block(),
                pending_confirmations=pending,
                dup_tool_limit=3,
                tool_cache_enabled=False,
                enable_policy_review=False,
                disable_policy_review_in_voice=True,
                tool_grants={},
            )

        asyncio.run(_test())
        self.assertEqual(len(pending), 1)
        payload = list(pending.values())[0]
        self.assertIn("shell_grant_key", payload)
        self.assertEqual(payload["shell_grant_key"], "shell_exec:dev:git")


class ShellGrantOnConfirmTests(unittest.TestCase):
    """Test that shell-specific grants are created on confirmation."""

    def test_confirmation_returns_shell_grant_key(self):
        """resolve_confirmation should return shell_grant_key from payload."""
        from liagent.agent.confirmation_handler import resolve_confirmation
        import liagent.tools.shell_exec  # noqa: F401
        from liagent.tools import get_tool
        from liagent.agent.tool_executor import ToolExecutor
        from liagent.tools.policy import ToolPolicy
        from liagent.agent.memory import ConversationMemory
        from datetime import datetime, timedelta, timezone

        td = get_tool("shell_exec")
        self.assertIsNotNone(td)

        pending = {
            "abc123": {
                "tool_name": "shell_exec",
                "tool_args": {"command": "echo hello"},
                "created_at": datetime.now(timezone.utc),
                "user_input": "run echo",
                "assistant_tool_call": "",
                "required_stage": 1,
                "stage": 1,
                "pending_reason": "shell_exec dev: echo",
                "shell_grant_key": "shell_exec:dev:echo",
            },
        }

        policy = ToolPolicy(tool_profile="full")
        executor_inst = ToolExecutor(policy, retry_count=0, timeout_sec=10)
        memory = ConversationMemory()

        async def fake_final_answer():
            return "Done.", {}

        async def _test():
            return await resolve_confirmation(
                "abc123", True, False,
                pending_confirmations=pending,
                confirm_ttl=timedelta(minutes=10),
                tool_policy=policy,
                tool_executor=executor_inst,
                memory=memory,
                final_answer_fn=fake_final_answer,
            )

        result = asyncio.run(_test())
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result.get("execution_ok"))
        self.assertEqual(result.get("shell_grant_key"), "shell_exec:dev:echo")
