import tempfile
import unittest
from pathlib import Path

from liagent.agent.memory import LongTermMemory
from liagent.agent.prompt_builder import PromptBuilder

# Trigger built-in tool registration for deterministic prompt content.
from liagent.tools import describe_image as _di  # noqa: F401
from liagent.tools import lint_code as _lc  # noqa: F401
from liagent.tools import list_dir as _ld  # noqa: F401
from liagent.tools import python_exec as _pe  # noqa: F401
from liagent.tools import read_file as _rf  # noqa: F401
from liagent.tools import run_tests as _rt  # noqa: F401
from liagent.tools import screenshot as _sc  # noqa: F401
from liagent.tools import task_tool as _tt  # noqa: F401
from liagent.tools import verify_syntax as _vs  # noqa: F401
from liagent.tools import web_fetch as _wf  # noqa: F401
from liagent.tools import web_search as _ws  # noqa: F401
from liagent.tools import write_file as _wrf  # noqa: F401


class PromptBuilderApiTests(unittest.TestCase):
    def _make_builder(self, td: str) -> PromptBuilder:
        mem = LongTermMemory(Path(td) / "mem.db", data_dir=Path(td) / "data")
        return PromptBuilder(mem)

    def test_api_prompt_omits_inline_tool_schemas(self):
        with tempfile.TemporaryDirectory() as td:
            pb = self._make_builder(td)
            local_prompt = pb.build_system_prompt(query="Check Google stock price")
            api_prompt = pb.build_system_prompt_for_api(
                query="Check Google stock price"
            )

        self.assertIn("Tool schemas are supplied separately", api_prompt)
        self.assertIn("Use native function/tool calling", api_prompt)
        self.assertIn("Do not print `<tool_call>...</tool_call>`", api_prompt)
        self.assertNotIn("**web_search**(", api_prompt)
        self.assertIn("**web_search**(", local_prompt)
        self.assertLess(len(api_prompt), len(local_prompt))

    def test_api_prompt_still_supports_professional_mode(self):
        with tempfile.TemporaryDirectory() as td:
            pb = self._make_builder(td)
            api_prompt = pb.build_system_prompt_for_api(
                query="Write a research report"
            )
        self.assertIn("Professional mode", api_prompt)

    def test_system_prompt_uses_neutral_assistant_name(self):
        with tempfile.TemporaryDirectory() as td:
            pb = self._make_builder(td)
            prompt = pb.build_system_prompt(query="Check Google stock price")
        self.assertIn("You are LiAgent", prompt)
        self.assertNotIn("You are Mia", prompt)

    def test_api_prompt_falls_back_to_inline_schema_for_non_function_protocol(self):
        with tempfile.TemporaryDirectory() as td:
            pb = self._make_builder(td)
            prompt = pb.build_system_prompt_for_api(
                query="Check stock",
                tool_protocol="native_xml",
            )
        self.assertIn("<tool_call>", prompt)
        self.assertIn("**web_search**(", prompt)

    def test_prompt_filters_unavailable_exec_tools(self):
        with tempfile.TemporaryDirectory() as td:
            pb = self._make_builder(td)
            prompt = pb.build_system_prompt_for_api(
                query="Write code",
                tool_profile="full",
                available_tool_names={"web_search", "read_file"},
            )
        self.assertNotIn("python_exec", prompt)
        self.assertNotIn("write_file", prompt)
