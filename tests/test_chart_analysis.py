"""Tests for chart schema extraction chain."""
import unittest
from liagent.agent.chart_analysis import (
    CHART_ANALYSIS_PROMPT,
    ChartAnalysisResult,
    is_chart_context,
    parse_chart_result,
)


class ChartDetectionTests(unittest.TestCase):
    def test_stock_chart(self):
        self.assertTrue(is_chart_context("show me AAPL stock price chart"))

    def test_revenue_graph(self):
        self.assertTrue(is_chart_context("this revenue graph shows earnings"))

    def test_plain_photo(self):
        self.assertFalse(is_chart_context("describe this photo of a cat"))

    def test_empty(self):
        self.assertFalse(is_chart_context(""))

    def test_bar_chart(self):
        self.assertTrue(is_chart_context("analyze this bar chart"))

    def test_candlestick(self):
        self.assertTrue(is_chart_context("explain the candlestick pattern"))


class ChartParsingTests(unittest.TestCase):
    def test_structured_output(self):
        raw = (
            "Chart type: line chart\n"
            "Axes: X=Date (2023-01 to 2024-01), Y=Revenue ($M)\n"
            "Time window: Jan 2023 to Jan 2024\n"
            "Key data points: Q1=$45M, Q2=$52M, Q3=$48M, Q4=$61M\n"
            "Trends: upward, +35% YoY\n"
            "Conclusions: Revenue grew consistently with Q4 acceleration.\n"
            "Uncertainty: Q3 dip may be seasonal."
        )
        r = parse_chart_result(raw)
        self.assertEqual(r.chart_type, "line chart")
        self.assertIn("Revenue", r.y_axis)
        self.assertIn("upward", r.trends)
        self.assertGreater(len(r.data_points), 0)
        self.assertEqual(r.raw, raw)

    def test_unstructured_fallback(self):
        r = parse_chart_result("Some rambling about colors and layout")
        self.assertEqual(r.chart_type, "unknown")
        self.assertTrue(r.raw)

    def test_partial_output(self):
        raw = "Chart type: bar chart\nKey data points: A=10, B=20"
        r = parse_chart_result(raw)
        self.assertEqual(r.chart_type, "bar chart")
        self.assertGreater(len(r.data_points), 0)

    def test_result_dataclass_fields(self):
        r = ChartAnalysisResult(
            chart_type="pie", x_axis="", y_axis="",
            time_window="", data_points="A=50%",
            trends="", conclusions="", uncertainty="", raw="test",
        )
        self.assertEqual(r.chart_type, "pie")
        self.assertEqual(r.raw, "test")


class ChartPromptTests(unittest.TestCase):
    def test_required_sections(self):
        for kw in [
            "Chart type", "Axes", "Time window", "Key data points",
            "Trends", "Conclusions", "Uncertainty",
        ]:
            self.assertIn(kw, CHART_ANALYSIS_PROMPT)
