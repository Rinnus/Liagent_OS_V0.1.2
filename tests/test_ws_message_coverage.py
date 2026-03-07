"""
Test: WebSocket message type coverage between backend and frontend.

Validates that every message type sent by the Python backend has a
corresponding entry in the frontend's REGISTERED_MESSAGE_TYPES array
(C5 constraint from the frontend modularization design).
"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "src" / "liagent" / "ui" / "static"
_JS = _STATIC / "js"


def _parse_registered_types() -> set[str]:
    """Extract REGISTERED_MESSAGE_TYPES from ws-client.js."""
    text = (_JS / "ws-client.js").read_text()
    m = re.search(
        r"REGISTERED_MESSAGE_TYPES\s*=\s*\[(.*?)\]",
        text,
        re.DOTALL,
    )
    assert m, "REGISTERED_MESSAGE_TYPES not found in ws-client.js"
    return set(re.findall(r"'([a-z_]+)'", m.group(1)))


def _parse_backend_types() -> set[str]:
    """Extract all WS message types sent by web_server.py."""
    text = (_ROOT / "src" / "liagent" / "ui" / "web_server.py").read_text()
    # Match {"type": "xxx"} and {'type': 'xxx'} patterns
    types: set[str] = set()
    for m in re.finditer(r"""["']type["']\s*:\s*["']([a-z_]+)["']""", text):
        types.add(m.group(1))
    # Also check voice_chat.py for types forwarded to WS
    voice_chat = _ROOT / "src" / "liagent" / "voice" / "voice_chat.py"
    if voice_chat.exists():
        vtext = voice_chat.read_text()
        for m in re.finditer(r"""["']type["']\s*:\s*["']([a-z_]+)["']""", vtext):
            types.add(m.group(1))
    # Exclude backend-internal types not sent to frontend WS
    internal = {"audio", "text", "clear", "feedback", "tool_confirm", "barge_in"}
    return types - internal


def _parse_switch_cases() -> set[str]:
    """Extract handled case values from handleWSMessage switch in ws-client.js."""
    text = (_JS / "ws-client.js").read_text()
    return set(re.findall(r"case\s+'([a-z_]+)'", text))


# ── Tests ────────────────────────────────────────────────────────────


def test_registered_types_cover_backend():
    """Every backend message type must appear in REGISTERED_MESSAGE_TYPES."""
    registered = _parse_registered_types()
    backend = _parse_backend_types()
    missing = backend - registered
    assert not missing, (
        f"Backend sends types not in REGISTERED_MESSAGE_TYPES: {sorted(missing)}"
    )


def test_switch_cases_cover_registered():
    """Every REGISTERED_MESSAGE_TYPE should have a switch case or be a known no-op."""
    registered = _parse_registered_types()
    cases = _parse_switch_cases()
    # These types are intentionally handled as no-ops (no case needed)
    known_noops = set()
    uncovered = registered - cases - known_noops
    assert not uncovered, (
        f"REGISTERED_MESSAGE_TYPES without switch case: {sorted(uncovered)}"
    )


def test_no_inline_script_in_index():
    """index.html must not contain the old inline script block."""
    text = (_STATIC / "index.html").read_text()
    assert 'id="inline-hybrid"' not in text, "Inline script block still present"
    assert "window._wsBridge" not in text, "Legacy bridge references still present"


def test_index_line_count():
    """index.html should be under 1400 lines (pure HTML+CSS shell)."""
    lines = (_STATIC / "index.html").read_text().count("\n") + 1
    assert lines <= 1400, f"index.html is {lines} lines (target ≤1400)"


def test_no_ws_client_imports_from_renderers():
    """Renderers must import getWs/wsSend from ws-send.js, not ws-client.js."""
    renderers = _JS / "renderers"
    for f in renderers.glob("*.js"):
        text = f.read_text()
        assert "from '../ws-client.js'" not in text, (
            f"{f.name} still imports from ws-client.js (should use ws-send.js)"
        )


def test_ws_send_has_no_imports():
    """ws-send.js must be a zero-import module to break circular deps."""
    text = (_JS / "ws-send.js").read_text()
    # Check for actual import statements (not the word in comments)
    import_lines = [
        line.strip()
        for line in text.splitlines()
        if re.match(r"^\s*import\s+", line)
    ]
    assert not import_lines, (
        f"ws-send.js must have zero imports, found: {import_lines}"
    )


def test_stores_have_no_renderer_imports():
    """Store modules must never import from renderers/."""
    stores = _JS / "stores"
    for f in stores.glob("*.js"):
        text = f.read_text()
        assert "from '../renderers/" not in text, (
            f"{f.name} imports from renderers/ (C1 violation)"
        )
        assert "from './renderers/" not in text, (
            f"{f.name} imports from renderers/ (C1 violation)"
        )


def test_web_ui_messages_include_explicit_session_key():
    """Web chat sends a stable session_key on text/audio/clear control paths."""
    settings_text = (_JS / "renderers" / "settings-panel.js").read_text()
    message_panel_text = (_JS / "renderers" / "message-panel.js").read_text()
    voice_text = (_JS / "renderers" / "voice-overlay.js").read_text()
    chat_store_text = (_JS / "stores" / "chat-store.js").read_text()
    ws_client_text = (_JS / "ws-client.js").read_text()

    assert "function _buildWebSessionKey()" in chat_store_text
    assert "session_key: getWebSessionKey()" in settings_text
    assert "{ type: 'clear', session_key: getWebSessionKey() }" in settings_text
    assert "session_key: getWebSessionKey()" in voice_text
    assert "from '../stores/chat-store.js'" in voice_text
    assert "session_key: bar.dataset.sessionKey || ''" in message_panel_text
    assert "var suggestionSessionKey = data.target_session_id || getWebSessionKey();" in ws_client_text
    assert "getLastAssistantEl" in ws_client_text
    assert "getLastAssistantRunId" in ws_client_text
