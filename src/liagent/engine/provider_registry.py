"""Provider capability registry for OpenAI-compatible API backends."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderProfile:
    """Normalized provider capability + preset metadata."""

    name: str
    label: str
    api_base_url: str
    api_model: str
    model_family: str = "openai"
    tool_protocol: str = "openai_function"
    aliases: tuple[str, ...] = ()
    preset_visible: bool = True
    supports_tools: bool = True
    supports_parallel_tool_calls: bool = True
    supports_tool_choice: bool = True
    supports_stream_usage: bool = True
    max_tokens_field: str = "max_tokens"
    temperature_policy: str = "any"  # any | fixed | clamp
    temperature_fixed: float | None = None
    temperature_min: float | None = None
    temperature_max: float | None = None


_BASE_REGISTRY: dict[str, ProviderProfile] = {
    "openai": ProviderProfile(
        name="openai",
        label="OpenAI",
        api_base_url="https://api.openai.com/v1",
        api_model="gpt-4o",
        model_family="openai",
        aliases=("api.openai.com", "openai", "gpt-", " o1", " o3", " o4"),
    ),
    "gemini": ProviderProfile(
        name="gemini",
        label="Gemini (OpenAI-compatible)",
        api_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_model="gemini-3.0-flash",
        model_family="openai",
        aliases=("generativelanguage.googleapis.com", "gemini"),
        supports_parallel_tool_calls=False,
        supports_tool_choice=False,
        supports_stream_usage=False,
    ),
    "deepseek": ProviderProfile(
        name="deepseek",
        label="DeepSeek",
        api_base_url="https://api.deepseek.com/v1",
        api_model="deepseek-chat",
        model_family="deepseek",
        aliases=("api.deepseek.com", "deepseek"),
        supports_parallel_tool_calls=False,
        supports_tool_choice=False,
        supports_stream_usage=False,
    ),
    "moonshot": ProviderProfile(
        name="moonshot",
        label="Moonshot (Kimi)",
        api_base_url="https://api.moonshot.cn/v1",
        api_model="kimi-k2.5",
        model_family="openai",
        aliases=("api.moonshot.cn", "api.moonshot.ai", "moonshot", "moonshotai", "kimi"),
        temperature_policy="fixed",
        temperature_fixed=1.0,
    ),
    "openrouter": ProviderProfile(
        name="openrouter",
        label="OpenRouter",
        api_base_url="https://openrouter.ai/api/v1",
        api_model="openai/gpt-4o-mini",
        model_family="openai",
        aliases=("openrouter.ai",),
    ),
    "ollama": ProviderProfile(
        name="ollama",
        label="Ollama (local API)",
        api_base_url="http://127.0.0.1:11434/v1",
        api_model="llama3.1",
        model_family="llama",
        aliases=("127.0.0.1:11434", "localhost:11434", "ollama"),
    ),
    # Not a default UI preset because official Anthropic endpoint is not OpenAI-compatible.
    "claude": ProviderProfile(
        name="claude",
        label="Claude",
        api_base_url="",
        api_model="claude-3-5-sonnet",
        model_family="openai",
        aliases=("anthropic", "claude"),
        preset_visible=False,
        supports_parallel_tool_calls=False,
        supports_tool_choice=False,
        supports_stream_usage=False,
        max_tokens_field="max_completion_tokens",
    ),
}


_PRESET_ORDER = ("openai", "gemini", "deepseek", "moonshot", "openrouter", "ollama")
_PROVIDER_MATCH_ORDER = ("moonshot", "gemini", "deepseek", "claude", "ollama", "openrouter", "openai")


def _safe_json_env(name: str) -> dict:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _to_aliases(raw) -> tuple[str, ...]:
    if isinstance(raw, (list, tuple)):
        out = [str(v).strip().lower() for v in raw if str(v).strip()]
    elif isinstance(raw, str):
        out = [part.strip().lower() for part in raw.replace(";", ",").split(",") if part.strip()]
    else:
        out = []
    return tuple(dict.fromkeys(out))


def _coerce_profile_from_env(
    key: str,
    payload: dict,
    base: ProviderProfile,
) -> ProviderProfile:
    aliases = _to_aliases(payload.get("aliases", base.aliases))
    if key not in aliases:
        aliases = tuple(dict.fromkeys((key, *aliases)))
    fixed_raw = payload.get("temperature_fixed", base.temperature_fixed)
    min_raw = payload.get("temperature_min", base.temperature_min)
    max_raw = payload.get("temperature_max", base.temperature_max)
    return ProviderProfile(
        name=key,
        label=str(payload.get("label", base.label) or base.label),
        api_base_url=str(payload.get("api_base_url", base.api_base_url) or base.api_base_url),
        api_model=str(payload.get("api_model", base.api_model) or base.api_model),
        model_family=str(payload.get("model_family", base.model_family) or base.model_family).strip().lower() or "openai",
        tool_protocol=str(payload.get("tool_protocol", base.tool_protocol) or base.tool_protocol).strip().lower() or "openai_function",
        aliases=aliases,
        preset_visible=bool(payload.get("preset_visible", base.preset_visible)),
        supports_tools=bool(payload.get("supports_tools", base.supports_tools)),
        supports_parallel_tool_calls=bool(
            payload.get("supports_parallel_tool_calls", base.supports_parallel_tool_calls)
        ),
        supports_tool_choice=bool(payload.get("supports_tool_choice", base.supports_tool_choice)),
        supports_stream_usage=bool(payload.get("supports_stream_usage", base.supports_stream_usage)),
        max_tokens_field=str(payload.get("max_tokens_field", base.max_tokens_field) or base.max_tokens_field),
        temperature_policy=str(payload.get("temperature_policy", base.temperature_policy) or base.temperature_policy).strip().lower() or "any",
        temperature_fixed=float(fixed_raw) if fixed_raw is not None else None,
        temperature_min=float(min_raw) if min_raw is not None else None,
        temperature_max=float(max_raw) if max_raw is not None else None,
    )


def get_provider_registry() -> dict[str, ProviderProfile]:
    """Return merged provider registry (built-ins + env extension)."""
    reg = dict(_BASE_REGISTRY)
    extra = _safe_json_env("LIAGENT_API_PROVIDER_REGISTRY_JSON")
    for raw_key, payload in extra.items():
        if not isinstance(raw_key, str) or not isinstance(payload, dict):
            continue
        key = raw_key.strip().lower()
        if not key:
            continue
        extends = str(payload.get("extends", "openai") or "openai").strip().lower()
        base = reg.get(extends, reg["openai"])
        reg[key] = _coerce_profile_from_env(key, payload, base)
    if "openai" not in reg:
        reg["openai"] = _BASE_REGISTRY["openai"]
    return reg


def get_provider_profile(provider: str) -> ProviderProfile:
    key = str(provider or "").strip().lower() or "openai"
    reg = get_provider_registry()
    return reg.get(key, reg["openai"])


def _match_alias(text: str, provider: ProviderProfile) -> bool:
    hay = str(text or "").lower()
    if not hay:
        return False
    return any(alias and alias in hay for alias in provider.aliases)


def infer_api_provider(api_model: str, api_base_url: str) -> str:
    """Infer provider key from model/base URL."""
    model_hay = str(api_model or "").strip().lower()
    url_hay = str(api_base_url or "").strip().lower()
    combined = f"{model_hay} {url_hay}".strip()

    reg = get_provider_registry()
    ordered = [k for k in _PROVIDER_MATCH_ORDER if k in reg]
    ordered.extend(k for k in reg if k not in ordered)

    # Model prefix/vendor should win over gateway host (e.g., OpenRouter + moonshotai/*).
    for key in ordered:
        if key in {"openrouter", "openai", "ollama"}:
            continue
        if _match_alias(model_hay, reg[key]):
            return key
    for key in ordered:
        if _match_alias(combined, reg[key]):
            return key
    return "openai"


def infer_api_model_family(api_model: str, api_base_url: str) -> str:
    """Infer tool-call parsing family for API models."""
    hay = f"{api_model} {api_base_url}".strip().lower()
    if not hay:
        return "openai"
    if "deepseek" in hay:
        return "deepseek"
    if "llama" in hay or "meta-" in hay:
        return "llama"
    if "qwen" in hay:
        return "qwen3-vl"
    provider = infer_api_provider(api_model, api_base_url)
    fam = str(get_provider_profile(provider).model_family or "openai").strip().lower()
    return fam or "openai"


def infer_api_tool_protocol(api_model: str, api_base_url: str) -> str:
    """Infer tool protocol for API models."""
    provider = infer_api_provider(api_model, api_base_url)
    protocol = str(get_provider_profile(provider).tool_protocol or "openai_function").strip().lower()
    if protocol not in {"openai_function", "native_xml", "json_xml"}:
        return "openai_function"
    return protocol


def list_provider_presets(*, include_custom: bool = False) -> list[dict[str, object]]:
    """Provider presets for CLI/Web setting UIs."""
    reg = get_provider_registry()
    ordered = [k for k in _PRESET_ORDER if k in reg and reg[k].preset_visible]
    ordered.extend(sorted(k for k, p in reg.items() if p.preset_visible and k not in ordered))
    out: list[dict[str, object]] = []
    for key in ordered:
        p = reg[key]
        out.append(
            {
                "id": p.name,
                "label": p.label,
                "api_base_url": p.api_base_url,
                "api_model": p.api_model,
                "model_family": p.model_family,
                "tool_protocol": p.tool_protocol,
                "aliases": list(p.aliases),
                "temperature_policy": p.temperature_policy,
                "temperature_fixed": p.temperature_fixed,
                "temperature_min": p.temperature_min,
                "temperature_max": p.temperature_max,
            }
        )
    if include_custom:
        out.append(
            {
                "id": "custom",
                "label": "Custom",
                "api_base_url": "",
                "api_model": "",
                "model_family": "openai",
                "tool_protocol": "openai_function",
                "aliases": [],
                "temperature_policy": "any",
                "temperature_fixed": None,
                "temperature_min": None,
                "temperature_max": None,
            }
        )
    return out


def provider_presets_map() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for item in list_provider_presets(include_custom=False):
        key = str(item.get("id", "")).strip().lower()
        if not key:
            continue
        out[key] = {
            "label": str(item.get("label", key)),
            "api_base_url": str(item.get("api_base_url", "")),
            "api_model": str(item.get("api_model", "")),
            "model_family": str(item.get("model_family", "openai") or "openai"),
            "tool_protocol": str(item.get("tool_protocol", "openai_function") or "openai_function"),
            "temperature_policy": str(item.get("temperature_policy", "any") or "any"),
        }
    return out
