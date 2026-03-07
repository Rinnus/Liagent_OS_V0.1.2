import unittest

from liagent.agent.text_utils import clean_output


class TextUtilsTests(unittest.TestCase):
    def test_clean_output_removes_tool_call_artifacts(self):
        raw = "[tool_call] web_fetch]\nBased on the result, the conference will take place in San Francisco."
        out = clean_output(raw)
        self.assertNotIn("tool_call", out.lower())
        self.assertIn("conference will take place in San Francisco", out)


if __name__ == "__main__":
    unittest.main()
