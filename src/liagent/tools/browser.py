"""Browser automation wrapper — thin layer over MCP Playwright with action-level risk grading."""
from __future__ import annotations

import re

from . import ToolCapability, tool
from ..logging import get_logger

_log = get_logger("browser")

# ── Compile-time fixed mapping: wrapper tool -> (MCP server, MCP tool) ──
_WRAPPER_TO_MCP: dict[str, tuple[str, str]] = {
    "browser_navigate":   ("playwright", "browser_navigate"),
    "browser_screenshot": ("playwright", "browser_take_screenshot"),
    "browser_extract":    ("playwright", "browser_snapshot"),
    "browser_click":      ("playwright", "browser_click"),
    "browser_fill":       ("playwright", "browser_fill_form"),
    "browser_submit":     ("playwright", "browser_press_key"),
}

# ── Dynamic risk upgrade patterns ───────────────────────────────────
_SENSITIVE_URL_RE = re.compile(
    r"(login|signin|sign.in|auth|checkout|payment|billing|account.settings|oauth|signup)",
    re.I,
)
_SENSITIVE_SELECTOR_RE = re.compile(
    r"(password|credit.?card|cvv|ssn|token|secret|pay|submit.*order|confirm.*purchase)",
    re.I,
)

_READ_TOOLS = frozenset({"browser_navigate", "browser_screenshot", "browser_extract"})
_ALWAYS_SENSITIVE = frozenset({"browser_submit"})


def classify_browser_action(tool_name: str, url: str, selector: str) -> str:
    """Classify browser action risk: 'read' | 'write' | 'write_sensitive'."""
    if tool_name in _READ_TOOLS:
        return "read"
    if tool_name in _ALWAYS_SENSITIVE:
        return "write_sensitive"
    # Dynamic upgrade for click/fill
    if _SENSITIVE_URL_RE.search(url or ""):
        return "write_sensitive"
    if _SENSITIVE_SELECTOR_RE.search(selector or ""):
        return "write_sensitive"
    return "write"


async def _call_mcp(wrapper_name: str, user_args: dict) -> str:
    """Call MCP tool via fixed mapping. Server/tool names are constants.

    Applies dynamic risk classification: if the action is 'write_sensitive'
    (login pages, payment forms, etc.), the call is blocked unless the user
    has explicitly confirmed.
    """
    mapping = _WRAPPER_TO_MCP.get(wrapper_name)
    if mapping is None:
        return f"[Error] Unknown browser tool: {wrapper_name}"

    server, mcp_tool = mapping  # compile-time constants

    # Dynamic risk upgrade based on URL/selector patterns
    url = str(user_args.get("url", ""))
    selector = str(user_args.get("element", ""))
    risk = classify_browser_action(wrapper_name, url, selector)
    if risk == "write_sensitive":
        _log.event("browser_sensitive", tool=wrapper_name, url=url[:100])
        return (
            f"[Sensitive Action] Browser action '{wrapper_name}' targets a sensitive context "
            f"(login/payment/auth page). This requires explicit user confirmation. "
            f"Risk classification: {risk}"
        )

    try:
        from .mcp_bridge import get_bridge
        bridge = get_bridge()
        if bridge is None:
            return "[Error] MCP bridge not available. Browser tools require MCP Playwright server."
        reason = getattr(bridge, "server_error", lambda _name: "")(server)
        if reason:
            return f"[Unavailable] Browser MCP server '{server}' unavailable: {reason}"
        return await bridge.call_tool(server, mcp_tool, user_args)
    except Exception as e:
        return f"[Error] Browser action failed: {e}"


# ── Tool registrations ──────────────────────────────────────────────

@tool(
    name="browser_navigate",
    description="Navigate browser to a URL and return page content snapshot.",
    risk_level="medium",
    requires_confirmation=False,
    capability=ToolCapability(network_access=True, max_output_chars=4000, latency_tier="slow"),
    parameters={
        "properties": {
            "url": {"type": "string", "description": "URL to navigate to"},
        },
        "required": ["url"],
    },
)
async def browser_navigate(url: str, **kwargs) -> str:
    return await _call_mcp("browser_navigate", {"url": url})


@tool(
    name="browser_screenshot",
    description="Take a screenshot of the current browser page.",
    risk_level="medium",
    requires_confirmation=False,
    capability=ToolCapability(network_access=True, max_output_chars=2000, latency_tier="slow"),
    parameters={"properties": {}, "required": []},
)
async def browser_screenshot(**kwargs) -> str:
    return await _call_mcp("browser_screenshot", {})


@tool(
    name="browser_extract",
    description="Extract structured text content from the current browser page (accessibility snapshot).",
    risk_level="medium",
    requires_confirmation=False,
    capability=ToolCapability(network_access=True, max_output_chars=4000, latency_tier="medium"),
    parameters={"properties": {}, "required": []},
)
async def browser_extract(**kwargs) -> str:
    return await _call_mcp("browser_extract", {})


@tool(
    name="browser_click",
    description="Click an element on the current page. Requires confirmation or session grant.",
    risk_level="medium",
    requires_confirmation=True,
    capability=ToolCapability(network_access=True, max_output_chars=2000, latency_tier="medium", idempotent=False),
    parameters={
        "properties": {
            "element": {"type": "string", "description": "Element description or ref to click"},
        },
        "required": ["element"],
    },
)
async def browser_click(element: str, **kwargs) -> str:
    return await _call_mcp("browser_click", {"element": element})


@tool(
    name="browser_fill",
    description="Fill a form field on the current page. Requires confirmation or session grant.",
    risk_level="medium",
    requires_confirmation=True,
    capability=ToolCapability(network_access=True, max_output_chars=2000, latency_tier="medium", idempotent=False),
    parameters={
        "properties": {
            "element": {"type": "string", "description": "Form field element to fill"},
            "value": {"type": "string", "description": "Value to enter"},
        },
        "required": ["element", "value"],
    },
)
async def browser_fill(element: str, value: str, **kwargs) -> str:
    return await _call_mcp("browser_fill", {"element": element, "value": value})


@tool(
    name="browser_submit",
    description="Press Enter/submit on the current page. Always requires confirmation (sensitive action).",
    risk_level="high",
    requires_confirmation=True,
    capability=ToolCapability(network_access=True, max_output_chars=2000, latency_tier="medium", idempotent=False),
    parameters={
        "properties": {
            "key": {"type": "string", "description": "Key to press (default: Enter)"},
        },
        "required": [],
    },
)
async def browser_submit(key: str = "Enter", **kwargs) -> str:
    return await _call_mcp("browser_submit", {"key": key})
