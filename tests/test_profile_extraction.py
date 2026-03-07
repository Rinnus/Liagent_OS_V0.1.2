"""Tests for implicit preference extraction in session_finalizer."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from liagent.agent.memory import LongTermMemory, UserProfileStore
from liagent.agent.prompt_builder import PromptBuilder


class PreferenceExtractionPromptTests(unittest.TestCase):
    """Verify the extraction prompt structure."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.pb = PromptBuilder(
            LongTermMemory(
                db_path=Path(self.td.name) / "mem.db",
                data_dir=Path(self.td.name) / "data",
            )
        )

    def tearDown(self):
        self.td.cleanup()

    def test_prompt_structure(self):
        messages = [
            {"role": "user", "content": "Please answer in English."},
            {"role": "assistant", "content": "Understood."},
        ]
        result = self.pb.build_preference_extraction_prompt(messages)
        self.assertEqual(result[0]["role"], "system")
        self.assertIn("dimension", result[0]["content"])
        self.assertIn("signal_strength", result[0]["content"])
        # Must NOT extract timezone
        self.assertIn("NOT extract timezone", result[0]["content"])
        self.assertEqual(result[1]["role"], "user")

    def test_prompt_lists_dimensions(self):
        messages = [{"role": "user", "content": "test"}]
        result = self.pb.build_preference_extraction_prompt(messages)
        system = result[0]["content"]
        for dim in ["language", "response_style", "tone", "domains",
                     "expertise_level", "data_preference"]:
            self.assertIn(dim, system)


class ParsePreferenceOutputTests(unittest.TestCase):
    """Verify parsing of LLM preference extraction output."""

    def test_valid_json_array(self):
        from liagent.agent.session_finalizer import parse_preferences
        raw = '[{"dimension": "language", "value": "zh", "signal_strength": "strong"}]'
        result = parse_preferences(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["dimension"], "language")

    def test_empty_array(self):
        from liagent.agent.session_finalizer import parse_preferences
        self.assertEqual(parse_preferences("[]"), [])

    def test_invalid_json(self):
        from liagent.agent.session_finalizer import parse_preferences
        self.assertEqual(parse_preferences("not json"), [])

    def test_json_with_markdown_fence(self):
        from liagent.agent.session_finalizer import parse_preferences
        raw = '```json\n[{"dimension": "tone", "value": "casual", "signal_strength": "moderate"}]\n```'
        result = parse_preferences(raw)
        self.assertEqual(len(result), 1)

    def test_filters_invalid_entries(self):
        from liagent.agent.session_finalizer import parse_preferences
        raw = '[{"dimension": "language", "value": "zh", "signal_strength": "strong"}, {"bad": "entry"}]'
        result = parse_preferences(raw)
        self.assertEqual(len(result), 1)

    def test_filters_timezone(self):
        from liagent.agent.session_finalizer import parse_preferences
        raw = '[{"dimension": "timezone", "value": "UTC+8", "signal_strength": "strong"}]'
        result = parse_preferences(raw)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
