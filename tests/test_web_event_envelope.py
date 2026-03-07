from liagent.orchestrator.events import AgentEvent
from liagent.ui.event_envelope import (
    _attach_event_meta,
    _event_payload_tail,
    _extract_agent_event,
)


def test_extract_agent_event_exposes_envelope_meta():
    event = AgentEvent(
        type="token",
        payload="hello",
        source="sub:0",
        run_id="run123",
        agent_id="worker-0",
        seq=7,
        timestamp=1234.5,
    )
    legacy, meta = _extract_agent_event(event)
    assert legacy == ("token", "hello")
    assert meta["event_source"] == "sub:0"
    assert meta["agent_id"] == "worker-0"
    assert meta["event_seq"] == 7
    assert meta["event_ts"] == 1234.5
    assert meta["event_run_id"] == "run123"


def test_attach_event_meta_sets_defaults_without_overriding_existing_fields():
    payload = {"type": "token", "event_source": "frontend"}
    merged = _attach_event_meta(
        payload,
        event_type="token",
        event_meta={"event_source": "brain", "agent_id": "a1", "event_seq": 1},
    )
    assert merged["event_type"] == "token"
    assert merged["event_source"] == "frontend"
    assert merged["agent_id"] == "a1"
    assert merged["event_seq"] == 1


def test_event_payload_tail_for_single_and_multi_field_legacy_tuples():
    assert _event_payload_tail(("done",)) is None
    assert _event_payload_tail(("done", "ok")) == "ok"
    assert _event_payload_tail(("policy_blocked", "web_search", "rate limited")) == (
        "web_search",
        "rate limited",
    )
