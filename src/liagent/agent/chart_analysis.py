"""Chart schema extraction — two-stage detection and structured parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── Chart context detection signals ────────────────────────────────────
_CHART_SIGNALS_EN = frozenset({
    "chart", "graph", "plot", "diagram", "candlestick",
    "histogram", "bar chart", "line chart", "pie chart",
    "scatter", "heatmap", "treemap",
})

_CHART_SIGNALS_ZH = frozenset({
    "K line", "k line", "trend chart", "bar graph", "pie chart zh",
})

# Combined for quick membership check
_ALL_CHART_SIGNALS = _CHART_SIGNALS_EN | _CHART_SIGNALS_ZH


def is_chart_context(user_input: str) -> bool:
    """Detect if user input suggests a chart/graph analysis context."""
    if not user_input:
        return False
    lower = user_input.lower()
    return any(signal in lower for signal in _ALL_CHART_SIGNALS)


# ── Structured extraction prompt ───────────────────────────────────────
CHART_ANALYSIS_PROMPT = (
    "Analyze this chart/graph image in detail. Extract the following structured information:\n\n"
    "1. **Chart type**: What kind of chart is this? (line, bar, candlestick, pie, scatter, etc.)\n"
    "2. **Axes**: What are the X and Y axes? Include labels, units, and ranges.\n"
    "3. **Time window**: What time period does the chart cover?\n"
    "4. **Key data points**: List the most important specific values visible in the chart.\n"
    "5. **Trends**: Describe the overall trends, patterns, and notable changes.\n"
    "6. **Conclusions**: What are the main takeaways from this chart?\n"
    "7. **Uncertainty**: Note any data that is unclear, estimated, or potentially inaccurate.\n\n"
    "Format your response with these exact section headers:\n"
    "Chart type: ...\n"
    "Axes: ...\n"
    "Time window: ...\n"
    "Key data points: ...\n"
    "Trends: ...\n"
    "Conclusions: ...\n"
    "Uncertainty: ..."
)

# Generic prompts (extracted from tool_orchestrator for reuse)
GENERIC_PROMPT_SCREENSHOT = "Please analyze this screenshot and describe what's on screen."
GENERIC_PROMPT_IMAGE = "Please describe this image."


# ── Structured result dataclass ────────────────────────────────────────
@dataclass
class ChartAnalysisResult:
    chart_type: str = "unknown"
    x_axis: str = ""
    y_axis: str = ""
    time_window: str = ""
    data_points: str = ""
    trends: str = ""
    conclusions: str = ""
    uncertainty: str = ""
    raw: str = ""


# Section header patterns (case-insensitive)
_SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"chart\s*type", "chart_type"),
    (r"axes?", "axes"),
    (r"time\s*window", "time_window"),
    (r"key\s*data\s*points?", "data_points"),
    (r"trends?", "trends"),
    (r"conclusions?", "conclusions"),
    (r"uncertainty", "uncertainty"),
]


def parse_chart_result(raw: str) -> ChartAnalysisResult:
    """Parse VLM chart analysis output into structured result.

    Tolerant of format drift — falls back gracefully for unstructured output.
    """
    result = ChartAnalysisResult(raw=raw)

    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in raw.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        matched = False
        for pattern, field_name in _SECTION_PATTERNS:
            # Match "Chart type: value" or "**Chart type**: value"
            m = re.match(
                rf"^\*?\*?{pattern}\*?\*?\s*[:]\s*(.*)",
                stripped,
                re.IGNORECASE,
            )
            if m:
                if current_key and current_lines:
                    sections[current_key] = " ".join(current_lines).strip()
                current_key = field_name
                current_lines = [m.group(1).strip()] if m.group(1).strip() else []
                matched = True
                break

        if not matched and current_key:
            current_lines.append(stripped)

    if current_key and current_lines:
        sections[current_key] = " ".join(current_lines).strip()

    # Map parsed sections to result fields
    result.chart_type = sections.get("chart_type", "unknown")
    axes = sections.get("axes", "")
    if axes:
        if "Y=" in axes or "y=" in axes:
            parts = re.split(r",\s*(?=[YXyx]=)", axes)
            for part in parts:
                part = part.strip()
                if part.upper().startswith("X="):
                    result.x_axis = part[2:].strip()
                elif part.upper().startswith("Y="):
                    result.y_axis = part[2:].strip()
        else:
            result.x_axis = axes
            result.y_axis = axes
    result.time_window = sections.get("time_window", "")
    result.data_points = sections.get("data_points", "")
    result.trends = sections.get("trends", "")
    result.conclusions = sections.get("conclusions", "")
    result.uncertainty = sections.get("uncertainty", "")

    return result
