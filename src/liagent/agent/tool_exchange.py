"""Unified tool exchange — assistant(tool_calls) + tool(observation) memory writes.

Centralizes the pattern of recording a tool call and its observation into
conversation memory, handling VLM vs Coder format differences.
"""

import json

from .tool_parsing import sanitize_observation, strip_any_tool_call


def append_tool_exchange(
    memory,
    *,
    assistant_content: str,
    tool_name: str,
    tool_args: dict,
    observation: str,
    hint: str = "",
    use_structured: bool = True,
    evidence_step_id: str = "",
):
    """Write assistant(tool_call) + tool(observation) into conversation memory.

    Args:
        memory: ConversationMemory instance.
        assistant_content: Visible text from the assistant (may contain residual XML).
        tool_name: Name of the tool called.
        tool_args: Arguments passed to the tool.
        observation: Raw tool output (will be sanitized).
        hint: Optional instruction appended to the observation message.
        use_structured: True = Coder path (tool_calls field, content/tool_calls mutually exclusive).
                        False = VLM path (tool call embedded in content as XML).
    """
    safe_obs = sanitize_observation(observation)
    if evidence_step_id:
        safe_obs = f"[[evidence:{evidence_step_id}]] {safe_obs}"

    if use_structured:
        # Coder path: tool_calls field on assistant message, content only has visible text.
        # Strip any residual <tool_call> XML from content to prevent GLM template duplication.
        clean_content = strip_any_tool_call(assistant_content).strip()
        if not clean_content:
            clean_content = f"Calling tool {tool_name}."
        memory.add("assistant", clean_content, tool_calls=[{
            "name": tool_name,
            "arguments": tool_args,
        }])
    else:
        # VLM path: tool call embedded in content (VLM has no tool_calls support).
        tc_xml = (
            f'<tool_call>{{"name": "{tool_name}", '
            f'"args": {json.dumps(tool_args, ensure_ascii=False)}}}</tool_call>'
        )
        memory.add("assistant", f"{assistant_content}\n{tc_xml}")

    # Observation as role='tool'
    obs_content = safe_obs
    if hint:
        obs_content = f"{safe_obs}\n{hint}"
    memory.add("tool", obs_content)
