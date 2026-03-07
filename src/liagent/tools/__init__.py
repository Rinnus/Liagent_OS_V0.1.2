"""Tool registry for LiAgent."""

from dataclasses import dataclass, field
from typing import Callable, Literal


@dataclass
class ToolCapability:
    """Capability manifest for a tool."""

    data_classification: Literal["public", "internal", "sensitive"] = "public"
    network_access: bool = False
    filesystem_access: bool = False
    requires_user_presence: bool = False
    max_output_chars: int = 1200
    # E2: extended capability profile
    cost_tier: Literal["free", "low", "medium", "high"] = "free"
    latency_tier: Literal["fast", "medium", "slow"] = "fast"
    idempotent: bool = True
    failure_modes: tuple[str, ...] = ()  # immutable, hashable
    min_timeout_sec: float = 0.0  # tool-declared minimum outer timeout (0 = use tier default)


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict  # JSON Schema style
    func: Callable
    risk_level: Literal["low", "medium", "high"] = "low"
    requires_confirmation: bool = False
    validator: Callable[[dict], tuple[bool, str]] | None = None
    capability: ToolCapability = field(default_factory=ToolCapability)

    def to_native_schema(self) -> dict:
        """Convert to Qwen3-Coder native tool schema for tokenizer.apply_chat_template(tools=...)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def schema_text(self) -> str:
        """Readable tool description for the system prompt."""
        params = ", ".join(
            f'{k}: {v.get("type", "string")}' + (f' ({v.get("description", "")})' if v.get("description") else "")
            for k, v in self.parameters.get("properties", {}).items()
        )
        risk = f"risk={self.risk_level}"
        confirm = ", confirm" if self.requires_confirmation else ""
        cap = self.capability
        caps = []
        if cap.network_access:
            caps.append("net")
        if cap.filesystem_access:
            caps.append("fs")
        caps.append(f"class={cap.data_classification}")
        caps.append(f"max_out={cap.max_output_chars}")
        if cap.cost_tier != "free":
            caps.append(f"cost={cap.cost_tier}")
        if cap.latency_tier != "fast":
            caps.append(f"latency={cap.latency_tier}")
        if not cap.idempotent:
            caps.append("non-idempotent")
        caps_text = ", ".join(caps)
        return f"- **{self.name}**({params}): {self.description} [{risk}{confirm}, {caps_text}]"


# Global registry
_TOOLS: dict[str, ToolDef] = {}


def tool(
    name: str,
    description: str,
    parameters: dict | None = None,
    *,
    risk_level: Literal["low", "medium", "high"] = "low",
    requires_confirmation: bool = False,
    validator: Callable[[dict], tuple[bool, str]] | None = None,
    capability: ToolCapability | None = None,
):
    """Decorator to register a tool function."""
    if parameters is None:
        parameters = {"properties": {}}

    def decorator(func: Callable):
        td = ToolDef(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
            risk_level=risk_level,
            requires_confirmation=requires_confirmation,
            validator=validator,
            capability=capability or ToolCapability(),
        )
        _TOOLS[name] = td
        return func

    return decorator


def get_all_tools() -> dict[str, ToolDef]:
    return _TOOLS.copy()


def get_tool(name: str) -> ToolDef | None:
    return _TOOLS.get(name)


def get_native_tool_schemas() -> list[dict]:
    """Return all registered tools as native schemas for Qwen3-Coder."""
    return [t.to_native_schema() for t in _TOOLS.values()]
