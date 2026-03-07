import time
from liagent.orchestrator.events import AgentEvent, make_event, EventSequencer

def test_agent_event_to_legacy_tuple():
    e = AgentEvent(type="token", payload="hello", source="brain",
                   run_id="r1", agent_id="a1", seq=0, timestamp=1.0)
    assert e.to_legacy_tuple() == ("token", "hello")

def test_agent_event_frozen():
    e = AgentEvent(type="done", payload="ok", source="lead",
                   run_id="r1", agent_id="a1", seq=1, timestamp=1.0)
    try:
        e.type = "token"
        assert False, "should be frozen"
    except AttributeError:
        pass

def test_tool_start_legacy_tuple_packs_args():
    """tool_start events have (type, name, args) in legacy format."""
    e = AgentEvent(type="tool_start",
                   payload={"name": "web_search", "args": {"query": "test"}},
                   source="sub:0", run_id="r1", agent_id="s0", seq=2, timestamp=1.0)
    legacy = e.to_legacy_tuple()
    assert legacy[0] == "tool_start"
    assert legacy[1] == "web_search"
    assert legacy[2] == {"query": "test"}

def test_tool_result_legacy_tuple():
    e = AgentEvent(type="tool_result",
                   payload={"name": "web_search", "result": "data..."},
                   source="sub:0", run_id="r1", agent_id="s0", seq=3, timestamp=1.0)
    legacy = e.to_legacy_tuple()
    assert legacy[0] == "tool_result"
    assert legacy[1] == "web_search"
    assert legacy[2] == "data..."

def test_event_sequencer_monotonic():
    seq = EventSequencer()
    assert seq.next() == 0
    assert seq.next() == 1
    assert seq.next() == 2

def test_confirmation_required_legacy_tuple():
    e = AgentEvent(type="confirmation_required",
                   payload={"token": "abc123", "tool": "execute_code",
                            "reason": "code execution", "brief": "Run code?"},
                   source="brain", run_id="r1", agent_id="a1", seq=4, timestamp=1.0)
    legacy = e.to_legacy_tuple()
    assert legacy[0] == "confirmation_required"
    assert legacy[1] == "abc123"
    assert legacy[2] == "execute_code"
    assert legacy[3] == "code execution"
    assert legacy[4] == "Run code?"

def test_passthrough_tuple_payload_roundtrip():
    """3-element legacy tuples (policy_blocked, step_check) survive the round-trip."""
    # Orchestrator passthrough stores tuple tail as payload
    e = AgentEvent(type="policy_blocked",
                   payload=("web_search", "rate limited"),
                   source="brain", run_id="r1", agent_id="a1", seq=5, timestamp=1.0)
    legacy = e.to_legacy_tuple()
    assert legacy == ("policy_blocked", "web_search", "rate limited")


def test_passthrough_2elem_payload_unchanged():
    """2-element legacy tuples store scalar payload (not a tuple)."""
    e = AgentEvent(type="token", payload="hello",
                   source="brain", run_id="r1", agent_id="a1", seq=6, timestamp=1.0)
    legacy = e.to_legacy_tuple()
    assert legacy == ("token", "hello")


def test_make_event_auto_timestamp():
    seq = EventSequencer()
    before = time.time()
    e = make_event("token", "hi", source="brain", run_id="r1", agent_id="a1", sequencer=seq)
    after = time.time()
    assert e.seq == 0
    assert before <= e.timestamp <= after
