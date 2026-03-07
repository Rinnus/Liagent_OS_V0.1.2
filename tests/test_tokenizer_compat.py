"""Tokenizer-level regression tests for role='tool' + tool_calls rendering.

These tests verify that each model's chat_template correctly handles the new
structured tool exchange format (role='tool' messages + tool_calls field).
Requires local model files — automatically skipped in CI or environments
where models aren't available.
"""

import unittest
import os
from pathlib import Path

MODELS_DIR = Path(
    os.environ.get("LIAGENT_MODELS_DIR", str(Path.home() / "Desktop" / "liagent-models"))
)


def _load_tokenizer(model_path: Path):
    """Load a tokenizer from a local model path."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)


@unittest.skipUnless(
    (MODELS_DIR / "mlx-community:GLM-4.7-Flash-4bit").exists(),
    "GLM-4.7-Flash model not available locally",
)
class GLM47TokenizerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _load_tokenizer(MODELS_DIR / "mlx-community:GLM-4.7-Flash-4bit")

    def test_tool_role_renders(self):
        """role='tool' messages should be rendered by GLM template."""
        messages = [
            {"role": "user", "content": "What is the weather?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "web_search", "arguments": {"query": "weather today"}}}
            ]},
            {"role": "tool", "content": "Sunny, 25°C"},
        ]
        result = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertIn("Sunny", result)
        self.assertIn("web_search", result)

    def test_enable_thinking_param(self):
        """GLM tokenizer should accept enable_thinking parameter."""
        messages = [{"role": "user", "content": "Hello"}]
        # Should not raise TypeError
        result = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        self.assertIn("Hello", result)

    def test_enable_thinking_false(self):
        """enable_thinking=False should produce different output."""
        messages = [{"role": "user", "content": "Hello"}]
        result_on = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        result_off = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        # They should differ (one ends with <think>, other with </think>)
        self.assertNotEqual(result_on, result_off)


@unittest.skipUnless(
    (MODELS_DIR / "mlx-community:Qwen3-Coder-30B-A3B-Instruct-4bit").exists(),
    "Qwen3-Coder model not available locally",
)
class Qwen3CoderTokenizerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _load_tokenizer(MODELS_DIR / "mlx-community:Qwen3-Coder-30B-A3B-Instruct-4bit")

    def test_tool_role_renders(self):
        """role='tool' messages should be rendered by Qwen3 template."""
        messages = [
            {"role": "user", "content": "Search something"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "web_search", "arguments": {"query": "test"}}}
            ]},
            {"role": "tool", "content": "Search results here"},
        ]
        result = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertIn("Search results here", result)


if __name__ == "__main__":
    unittest.main()
