"""Configuration management for LiAgent."""

import json
import os
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def db_path() -> Path:
    """Canonical SQLite database path — override with LIAGENT_DB_PATH env var."""
    custom = os.environ.get("LIAGENT_DB_PATH", "").strip()
    if custom:
        return Path(custom)
    return PROJECT_ROOT / "liagent.db"

# Unified model base directory — override with LIAGENT_MODELS_DIR env var
MODELS_BASE_DIR = Path(
    os.environ.get("LIAGENT_MODELS_DIR", str(Path.home() / "Desktop" / "liagent-models"))
)
DEFAULT_MODELS_DIR = MODELS_BASE_DIR if MODELS_BASE_DIR.exists() else MODELS_DIR
CONFIG_PATH = PROJECT_ROOT / "config.json"

# Current config schema version — bump when format changes
CONFIG_VERSION = 3

RuntimeMode = Literal["local_private", "hybrid_balanced", "cloud_performance"]


@dataclass
class LLMConfig:
    backend: str = "local"  # "local" | "api"
    model_family: str = "glm47"  # glm47 | qwen3-coder | qwen3-vl | llama | deepseek | openai | gemini | claude
    tool_protocol: str = "auto"  # auto | openai_function | native_xml | json_xml
    # Local mode
    local_model_path: str = str(DEFAULT_MODELS_DIR / "Qwen3-VL-4B-Instruct-MLX-4bit")
    # API mode (OpenAI-compatible)
    api_base_url: str = ""
    api_key: str = ""
    api_model: str = ""
    api_cache_mode: str = "implicit"  # off | implicit | explicit
    api_cache_policy: str = "tiered"  # flat | tiered
    api_cache_ttl_sec: int = 600
    api_cache_ttl_static_sec: int = 3600
    api_cache_ttl_memory_sec: int = 900
    api_cache_min_prefix_chars: int = 400
    # Generation params
    max_tokens: int = 2048
    temperature: float = 0.3


@dataclass
class TTSConfig:
    backend: str = "local"  # "local" | "api"
    tts_engine: str = "qwen3"  # qwen3 | api
    # Local mode
    local_model_path: str = str(DEFAULT_MODELS_DIR / "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")
    language: str = "zh"
    voice: str = ""  # legacy field, kept for config compat
    speed: float = 1.0
    speaker_name: str = "serena"  # preset speaker from CustomVoice model
    temperature: float = 0.3
    top_k: int = 20
    top_p: float = 0.8
    repetition_penalty: float = 1.05
    voice_profile: str = "base_default"  # legacy field, kept for config compat
    speaker: str = ""  # legacy alias, kept for config compat
    chunk_strategy: Literal["oneshot", "smart_chunk"] = "smart_chunk"
    max_chunk_chars: int = 220
    ref_audio: str = ""   # legacy field
    ref_text: str = ""    # legacy field
    # API mode
    api_base_url: str = ""
    api_key: str = ""
    api_model: str = ""
    api_voice: str = "alloy"


@dataclass
class STTConfig:
    backend: str = "local"  # "local" | "api"
    model: str = str(DEFAULT_MODELS_DIR / "Qwen3-ASR-0.6B-4bit")
    language: str = "auto"  # "auto", "chinese", "english", etc.
    api_base_url: str = ""
    api_key: str = ""
    api_model: str = ""


@dataclass
class ApiKeysConfig:
    """Third-party API keys used by the attention/signal system."""
    finnhub: str = ""  # https://finnhub.io/register — free tier: 60 req/min

@dataclass
class TaskConfig:
    """Configuration for the autonomous task system."""
    enabled: bool = True
    cwork_dir: str = str(Path.home() / "Desktop" / "cwork")
    max_concurrent_tasks: int = 1
    default_priority: int = 10
    default_max_retries: int = 2


@dataclass
class RoutingConfig:
    """Capability router configuration."""

    enabled: bool = True
    long_context_local_soft_tokens: int = 6000
    long_context_local_hard_tokens: int = 12000
    private_cloud_override: bool = False
    code_delegate_enabled: bool = True


@dataclass
class SandboxConfig:
    """Tool execution sandbox configuration."""

    enabled: bool = False
    mode: Literal["off", "non_main", "all"] = "off"
    image: str = "liagent-sandbox:latest"
    network_default: Literal["off", "on"] = "off"
    docker_bin: str = "docker"
    workspace_mount: str = str(Path.home() / "Desktop" / "cwork")
    cpu_limit: float = 1.5
    memory_limit_mb: int = 1024
    pids_limit: int = 256


@dataclass
class BudgetConfig:
    """Session-level token/cost guardrails."""

    session_soft_tokens: int = 60000
    session_hard_tokens: int = 120000
    session_soft_usd: float = 1.5
    session_hard_usd: float = 3.0


@dataclass
class MCPDiscoveryConfig:
    """Local MCP server auto-discovery and hot-reload settings."""

    enabled: bool = True  # trust registry protects against unknown tools
    dirs: list[str] = field(default_factory=lambda: [
        str(Path.home() / ".liagent" / "mcp.d"),
        str(Path.home() / ".config" / "modelcontextprotocol"),
    ])
    hot_reload_sec: int = 120
    tool_timeout_sec: float = 30.0
    max_response_bytes: int = 1_000_000


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP (Model Context Protocol) server."""
    name: str = ""                    # Namespace prefix, e.g. "github"
    command: str = ""                 # Executable path, e.g. "npx"
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    risk_level: str = "medium"        # Default risk level for all tools from this server
    network_access: bool = True
    filesystem_access: bool = False
    enabled: bool = True
    wrapper_only: bool = False        # If True, tools are NOT registered — only callable via wrapper


@dataclass
class ProactiveConfig:
    """Configuration for proactive intelligence system."""
    authorization: dict[str, str] = field(default_factory=lambda: {
        "stock": "auto",
        "news": "auto",
        "coding": "suggest",
        "general": "suggest",
    })
    quiet_hours: str = "23:00-07:00"
    timezone: str = "local"
    daily_touch_limit: int = 5
    same_category_cooldown_hours: int = 4
    suppression_days: int = 30
    # Goal autonomy settings
    max_active_goals: int = 5
    max_new_goals_per_day: int = 3
    max_goal_tasks_per_day: int = 20
    reflection_budget_tokens_per_day: int = 5000
    goal_auto_retire_days: int = 14
    reflection_interval_min: int = 30
    reflection_interval_max: int = 120
    bridge_scan_interval_sec: int = 30

    @classmethod
    def load(cls, path: Path | None = None) -> "ProactiveConfig":
        """Load from YAML file. Returns defaults if file missing."""
        if path is None:
            path = Path.home() / ".liagent" / "proactive.yaml"
        if not path.exists():
            return cls()
        try:
            import yaml
            data = yaml.safe_load(path.read_text()) or {}
            auth = data.get("authorization", None)
            defaults = cls()
            if auth and isinstance(auth, dict):
                merged_auth = dict(defaults.authorization)
                merged_auth.update(auth)
            else:
                merged_auth = defaults.authorization
            return cls(
                authorization=merged_auth,
                quiet_hours=str(data.get("quiet_hours", defaults.quiet_hours)),
                timezone=str(data.get("timezone", defaults.timezone)),
                daily_touch_limit=int(data.get("daily_touch_limit", defaults.daily_touch_limit)),
                same_category_cooldown_hours=int(data.get("same_category_cooldown_hours", defaults.same_category_cooldown_hours)),
                suppression_days=int(data.get("suppression_days", defaults.suppression_days)),
            )
        except Exception:
            return cls()


@dataclass
class AppConfig:
    config_version: int = CONFIG_VERSION
    runtime_mode: RuntimeMode = "hybrid_balanced"
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)
    mcp_discovery: MCPDiscoveryConfig = field(default_factory=MCPDiscoveryConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    api_keys: ApiKeysConfig = field(default_factory=ApiKeysConfig)
    tool_profile: Literal["minimal", "research", "full"] = "research"
    repl_mode: Literal["off", "sandboxed", "trusted_local"] = "sandboxed"
    tts_enabled: bool = True
    voice_mode: bool = False
    show_thinking: bool = False       # UI: whether to display <think> content
    enable_thinking: bool = True      # Model: whether to generate <think> reasoning chain
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)

    @classmethod
    def _migrate(cls, data: dict) -> dict:
        """Auto-migrate old config formats to current version."""
        version = data.get("config_version", 1)
        if version < 2:
            # v1 → v2: add model_family, tts_engine, config_version
            llm = data.get("llm", {})
            if "model_family" not in llm:
                llm["model_family"] = "glm47"
            if "api_cache_mode" not in llm:
                llm["api_cache_mode"] = "implicit"
            if "api_cache_ttl_sec" not in llm:
                llm["api_cache_ttl_sec"] = 600
            tts = data.get("tts", {})
            if "tts_engine" not in tts:
                tts["tts_engine"] = "api" if tts.get("backend") == "api" else "qwen3"
            data["config_version"] = 2
        if version < 3:
            data.setdefault("runtime_mode", "hybrid_balanced")
            data.setdefault("routing", {})
            data.setdefault("sandbox", {})
            data.setdefault("budget", {})
            data.setdefault("mcp_discovery", {})
            data["config_version"] = 3
        llm = data.get("llm", {})
        if "tool_protocol" not in llm:
            llm["tool_protocol"] = "auto"
        if "api_cache_mode" not in llm:
            llm["api_cache_mode"] = "implicit"
        if "api_cache_policy" not in llm:
            llm["api_cache_policy"] = "tiered"
        if "api_cache_ttl_sec" not in llm:
            llm["api_cache_ttl_sec"] = 600
        if "api_cache_ttl_static_sec" not in llm:
            llm["api_cache_ttl_static_sec"] = 3600
        if "api_cache_ttl_memory_sec" not in llm:
            llm["api_cache_ttl_memory_sec"] = 900
        if "api_cache_min_prefix_chars" not in llm:
            llm["api_cache_min_prefix_chars"] = 400
        data["llm"] = llm
        stt = data.get("stt", {})
        if "backend" not in stt:
            stt["backend"] = "local"
        data["stt"] = stt
        # Migrate kokoro → qwen3
        tts = data.get("tts", {})
        if tts.get("tts_engine") == "kokoro":
            tts["tts_engine"] = "qwen3"
        data["tts"] = tts
        # Ensure mcp_servers key exists (added in v2+)
        if "mcp_servers" not in data:
            data["mcp_servers"] = []
        if "runtime_mode" not in data:
            data["runtime_mode"] = "hybrid_balanced"
        if "routing" not in data:
            data["routing"] = {}
        if "sandbox" not in data:
            data["sandbox"] = {}
        if "budget" not in data:
            data["budget"] = {}
        if "mcp_discovery" not in data:
            data["mcp_discovery"] = {}
        return data

    def save(self):
        CONFIG_PATH.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False)
        )

    @classmethod
    def load(cls) -> "AppConfig":
        if not CONFIG_PATH.exists():
            cfg = cls()
            cfg.save()
            return cfg
        data = json.loads(CONFIG_PATH.read_text())
        data = cls._migrate(data)

        llm_data = data.get("llm", {})
        tts_data_raw = data.get("tts", {})
        stt_data = data.get("stt", {})
        allowed_tts_fields = {f.name for f in fields(TTSConfig)}
        tts_data = {k: v for k, v in tts_data_raw.items() if k in allowed_tts_fields}

        # Backward compatibility: ensure speed is present
        if "speed" not in tts_data:
            tts_data["speed"] = 1.0
        # Migrate old Kokoro/Base model path to Qwen3-TTS-CustomVoice
        local_path = str(tts_data.get("local_model_path", "") or "")
        qwen3_default = str(DEFAULT_MODELS_DIR / "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")
        if not local_path or "Kokoro" in local_path or "Base-4bit" in local_path:
            tts_data["local_model_path"] = qwen3_default
        # Parse MCP server configs
        mcp_raw = data.get("mcp_servers", [])
        allowed_mcp_fields = {f.name for f in fields(MCPServerConfig)}
        mcp_servers = []
        for entry in mcp_raw:
            if isinstance(entry, dict) and entry.get("name"):
                filtered = {k: v for k, v in entry.items() if k in allowed_mcp_fields}
                mcp_servers.append(MCPServerConfig(**filtered))

        # Parse task config
        tasks_raw = data.get("tasks", {})
        allowed_task_fields = {f.name for f in fields(TaskConfig)}
        tasks_data = {k: v for k, v in tasks_raw.items() if k in allowed_task_fields} if isinstance(tasks_raw, dict) else {}

        # Parse runtime mode + policy configs
        runtime_mode = str(data.get("runtime_mode", "hybrid_balanced") or "hybrid_balanced").strip().lower()
        if runtime_mode not in {"local_private", "hybrid_balanced", "cloud_performance"}:
            runtime_mode = "hybrid_balanced"

        routing_raw = data.get("routing", {})
        allowed_routing_fields = {f.name for f in fields(RoutingConfig)}
        routing_data = (
            {k: v for k, v in routing_raw.items() if k in allowed_routing_fields}
            if isinstance(routing_raw, dict)
            else {}
        )

        sandbox_raw = data.get("sandbox", {})
        allowed_sandbox_fields = {f.name for f in fields(SandboxConfig)}
        sandbox_data = (
            {k: v for k, v in sandbox_raw.items() if k in allowed_sandbox_fields}
            if isinstance(sandbox_raw, dict)
            else {}
        )

        budget_raw = data.get("budget", {})
        allowed_budget_fields = {f.name for f in fields(BudgetConfig)}
        budget_data = (
            {k: v for k, v in budget_raw.items() if k in allowed_budget_fields}
            if isinstance(budget_raw, dict)
            else {}
        )

        mcp_discovery_raw = data.get("mcp_discovery", {})
        allowed_mcp_discovery_fields = {f.name for f in fields(MCPDiscoveryConfig)}
        mcp_discovery_data = (
            {k: v for k, v in mcp_discovery_raw.items() if k in allowed_mcp_discovery_fields}
            if isinstance(mcp_discovery_raw, dict)
            else {}
        )

        api_keys_raw = data.get("api_keys", {})
        allowed_api_keys_fields = {f.name for f in fields(ApiKeysConfig)}
        api_keys_data = (
            {k: v for k, v in api_keys_raw.items() if k in allowed_api_keys_fields}
            if isinstance(api_keys_raw, dict)
            else {}
        )

        repl_mode = str(data.get("repl_mode", "sandboxed") or "sandboxed").strip().lower()
        if repl_mode not in {"off", "sandboxed", "trusted_local"}:
            repl_mode = "sandboxed"

        # Parse proactive config (nested dataclass with special authorization merge)
        proactive_raw = data.get("proactive", {})
        if isinstance(proactive_raw, dict) and proactive_raw:
            defaults = ProactiveConfig()
            auth = proactive_raw.get("authorization", None)
            if auth and isinstance(auth, dict):
                merged_auth = dict(defaults.authorization)
                merged_auth.update(auth)
            else:
                merged_auth = defaults.authorization
            allowed_proactive_fields = {f.name for f in fields(ProactiveConfig)}
            proactive_data = {k: v for k, v in proactive_raw.items()
                             if k in allowed_proactive_fields and k != "authorization"}
            proactive = ProactiveConfig(authorization=merged_auth, **proactive_data)
        else:
            proactive = ProactiveConfig()

        return cls(
            runtime_mode=runtime_mode,
            llm=LLMConfig(**llm_data),
            tts=TTSConfig(**tts_data),
            stt=STTConfig(**stt_data),
            mcp_servers=mcp_servers,
            mcp_discovery=MCPDiscoveryConfig(**mcp_discovery_data),
            tasks=TaskConfig(**tasks_data),
            routing=RoutingConfig(**routing_data),
            sandbox=SandboxConfig(**sandbox_data),
            budget=BudgetConfig(**budget_data),
            api_keys=ApiKeysConfig(**api_keys_data),
            proactive=proactive,
            tool_profile=data.get("tool_profile", "research"),
            repl_mode=repl_mode,
            tts_enabled=data.get("tts_enabled", True),
            voice_mode=data.get("voice_mode", False),
            show_thinking=data.get("show_thinking", False),
            enable_thinking=data.get("enable_thinking", True),
        )
