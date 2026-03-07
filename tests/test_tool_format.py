"""Tests for engine.tool_format module."""

import unittest

from liagent.engine.tool_format import (
    Qwen3CoderFormat,
    JsonInXmlFormat,
    GLM47Format,
    OpenAIFormat,
    RawJsonCallFormat,
    RawKeywordCallFormat,
    CompositeFormat,
    get_parser_for_family,
    get_parser_for_protocol,
    get_default_composite,
)


class Qwen3CoderFormatTests(unittest.TestCase):
    def setUp(self):
        self.parser = Qwen3CoderFormat()

    def test_parse_single_param(self):
        text = "<tool_call><function=web_search><parameter=query>test</parameter></function></tool_call>"
        result = self.parser.parse(text)
        self.assertEqual(result["name"], "web_search")
        self.assertEqual(result["args"]["query"], "test")

    def test_parse_no_match(self):
        self.assertIsNone(self.parser.parse("plain text"))

    def test_strip(self):
        text = "before <tool_call><function=x></function></tool_call> after"
        self.assertEqual(self.parser.strip(text), "before  after")

    def test_lenient_unclosed(self):
        text = "<tool_call><function=search><parameter=q>hello</parameter>"
        result = self.parser.parse_lenient(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "search")


class JsonInXmlFormatTests(unittest.TestCase):
    def setUp(self):
        self.parser = JsonInXmlFormat()

    def test_parse(self):
        text = '<tool_call>{"name": "test", "args": {"x": 1}}</tool_call>'
        result = self.parser.parse(text)
        self.assertEqual(result["name"], "test")

    def test_parse_invalid_json(self):
        text = "<tool_call>{invalid}</tool_call>"
        self.assertIsNone(self.parser.parse(text))

    def test_strip(self):
        text = 'a <tool_call>{"name": "x"}</tool_call> b'
        self.assertNotIn("<tool_call>", self.parser.strip(text))

    def test_lenient_unclosed(self):
        text = '<tool_call>{"name": "test", "args": {"q": "x"}}\n'
        result = self.parser.parse_lenient(text)
        self.assertIsNotNone(result)


class OpenAIFormatTests(unittest.TestCase):
    def setUp(self):
        self.parser = OpenAIFormat()

    def test_parse_function_call(self):
        import json
        text = json.dumps({
            "function_call": {"name": "search", "arguments": '{"query": "test"}'}
        })
        result = self.parser.parse(text)
        self.assertEqual(result["name"], "search")
        self.assertEqual(result["args"]["query"], "test")

    def test_parse_no_match(self):
        self.assertIsNone(self.parser.parse("plain text"))


class RawJsonCallFormatTests(unittest.TestCase):
    def setUp(self):
        self.parser = RawJsonCallFormat()

    def test_parse_raw_call(self):
        text = 'web_search({"query": "GOOG stock", "timelimit": "d"})'
        result = self.parser.parse(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "web_search")
        self.assertEqual(result["args"]["query"], "GOOG stock")

    def test_parse_lenient_embedded_line(self):
        text = 'I will search now.\nweb_search({"query": "GOOG stock"})'
        result = self.parser.parse_lenient(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "web_search")

    def test_strip_raw_call(self):
        text = 'before\nweb_search({"query": "GOOG stock"})\nafter'
        result = self.parser.strip(text)
        self.assertEqual(result, "before\nafter")


class RawKeywordCallFormatTests(unittest.TestCase):
    def setUp(self):
        self.parser = RawKeywordCallFormat()

    def test_parse_keyword_call(self):
        text = 'web_search(query="GOOG stock", timelimit="d")'
        result = self.parser.parse(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "web_search")
        self.assertEqual(result["args"]["timelimit"], "d")

    def test_parse_lenient_keyword_line(self):
        text = 'I will search now.\nweb_search(query="GOOG stock")'
        result = self.parser.parse_lenient(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "web_search")

    def test_strip_keyword_call(self):
        text = 'before\nweb_search(query="GOOG stock")\nafter'
        result = self.parser.strip(text)
        self.assertEqual(result, "before\nafter")


class GLM47FormatTests(unittest.TestCase):
    def setUp(self):
        self.parser = GLM47Format()

    def test_parse_single_param(self):
        text = "<tool_call>web_search<arg_key>query</arg_key><arg_value>test</arg_value></tool_call>"
        result = self.parser.parse(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "web_search")
        self.assertEqual(result["args"]["query"], "test")

    def test_parse_multi_param(self):
        text = (
            "<tool_call>web_search"
            "<arg_key>query</arg_key><arg_value>Google stock</arg_value>"
            "<arg_key>timelimit</arg_key><arg_value>d</arg_value>"
            "</tool_call>"
        )
        result = self.parser.parse(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "web_search")
        self.assertEqual(result["args"]["query"], "Google stock")
        self.assertEqual(result["args"]["timelimit"], "d")

    def test_parse_no_args(self):
        text = "<tool_call>list_tasks</tool_call>"
        result = self.parser.parse(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "list_tasks")
        self.assertEqual(result["args"], {})

    def test_parse_no_match(self):
        self.assertIsNone(self.parser.parse("plain text"))

    def test_strip(self):
        text = "before <tool_call>web_search<arg_key>q</arg_key><arg_value>x</arg_value></tool_call> after"
        result = self.parser.strip(text)
        self.assertNotIn("<tool_call>", result)
        self.assertIn("before", result)
        self.assertIn("after", result)

    def test_lenient_unclosed(self):
        text = "<tool_call>web_search<arg_key>query</arg_key><arg_value>test</arg_value>"
        result = self.parser.parse_lenient(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "web_search")

    def test_format_schemas_passthrough(self):
        tools = [{"type": "function", "function": {"name": "test"}}]
        self.assertEqual(self.parser.format_schemas(tools), tools)

    def test_think_plus_toolcall(self):
        text = (
            "<think>I need to search for stock prices.</think>"
            "<tool_call>stock<arg_key>symbol</arg_key><arg_value>GOOGL</arg_value></tool_call>"
        )
        result = self.parser.parse(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "stock")
        self.assertEqual(result["args"]["symbol"], "GOOGL")


class CompositeFormatTests(unittest.TestCase):
    def test_default_tries_json_then_native(self):
        composite = CompositeFormat()
        # JSON format
        text = '<tool_call>{"name": "a", "args": {}}</tool_call>'
        self.assertEqual(composite.parse(text)["name"], "a")
        # Native format
        text = "<tool_call><function=b><parameter=x>1</parameter></function></tool_call>"
        self.assertEqual(composite.parse(text)["name"], "b")

    def test_strip_removes_both(self):
        composite = CompositeFormat()
        text = '<tool_call>{"x": 1}</tool_call> and <tool_call><function=y></function></tool_call>'
        result = composite.strip(text)
        self.assertNotIn("<tool_call>", result)


class FactoryTests(unittest.TestCase):
    def test_known_families(self):
        p = get_parser_for_family("qwen3-coder")
        self.assertIsInstance(p, Qwen3CoderFormat)
        p = get_parser_for_family("qwen3-vl")
        self.assertIsInstance(p, JsonInXmlFormat)
        p = get_parser_for_family("openai")
        self.assertIsInstance(p, OpenAIFormat)
        p = get_parser_for_family("gemini")
        self.assertIsInstance(p, OpenAIFormat)
        p = get_parser_for_family("claude")
        self.assertIsInstance(p, OpenAIFormat)
        p = get_parser_for_family("glm47")
        self.assertIsInstance(p, GLM47Format)

    def test_unknown_family_returns_composite(self):
        p = get_parser_for_family("unknown-model")
        self.assertIsInstance(p, CompositeFormat)

    def test_default_composite(self):
        c = get_default_composite()
        self.assertIsInstance(c, CompositeFormat)
        self.assertEqual(len(c.parsers), 5)

    def test_protocol_parser_selection(self):
        p = get_parser_for_protocol("openai_function", model_family="deepseek")
        self.assertIsInstance(p, OpenAIFormat)
        p = get_parser_for_protocol("json_xml", model_family="openai")
        self.assertIsInstance(p, JsonInXmlFormat)


if __name__ == "__main__":
    unittest.main()
