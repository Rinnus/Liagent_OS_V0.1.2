"""LLM usage cost estimation helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PricingRates:
    prompt_per_mtok: float
    completion_per_mtok: float
    cached_prompt_per_mtok: float
    cache_write_per_mtok: float


_DEFAULT_PROVIDER_RATES: dict[str, PricingRates] = {
    # Conservative defaults; override via env for exact billing.
    "openai": PricingRates(
        prompt_per_mtok=2.50,
        completion_per_mtok=10.00,
        cached_prompt_per_mtok=1.25,
        cache_write_per_mtok=2.50,
    ),
    "gemini": PricingRates(
        prompt_per_mtok=0.50,
        completion_per_mtok=3.00,
        cached_prompt_per_mtok=0.05,
        cache_write_per_mtok=0.50,
    ),
    "deepseek": PricingRates(
        prompt_per_mtok=0.60,
        completion_per_mtok=2.40,
        cached_prompt_per_mtok=0.12,
        cache_write_per_mtok=0.60,
    ),
    "claude": PricingRates(
        prompt_per_mtok=3.00,
        completion_per_mtok=15.00,
        cached_prompt_per_mtok=0.30,
        cache_write_per_mtok=3.00,
    ),
    # Conservative placeholder defaults for Kimi/Moonshot.
    # Override via LIAGENT_COST_MOONSHOT_* or model-prefix overrides for exact billing.
    "moonshot": PricingRates(
        prompt_per_mtok=1.00,
        completion_per_mtok=4.00,
        cached_prompt_per_mtok=0.20,
        cache_write_per_mtok=1.00,
    ),
    # OpenRouter model pricing varies by routed provider/model.
    "openrouter": PricingRates(
        prompt_per_mtok=2.50,
        completion_per_mtok=10.00,
        cached_prompt_per_mtok=1.25,
        cache_write_per_mtok=2.50,
    ),
}


def _safe_float_env(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _safe_rates(raw: dict, fallback: PricingRates) -> PricingRates:
    return PricingRates(
        prompt_per_mtok=float(raw.get("prompt_per_mtok", fallback.prompt_per_mtok)),
        completion_per_mtok=float(raw.get("completion_per_mtok", fallback.completion_per_mtok)),
        cached_prompt_per_mtok=float(raw.get("cached_prompt_per_mtok", fallback.cached_prompt_per_mtok)),
        cache_write_per_mtok=float(raw.get("cache_write_per_mtok", fallback.cache_write_per_mtok)),
    )


def _model_overrides() -> dict[str, PricingRates]:
    raw = str(os.environ.get("LIAGENT_COST_MODEL_OVERRIDES_JSON", "") or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, PricingRates] = {}
    for key, val in parsed.items():
        if not isinstance(key, str) or not isinstance(val, dict):
            continue
        base = _DEFAULT_PROVIDER_RATES.get("openai")
        out[key.strip().lower()] = _safe_rates(val, base)
    return out


def resolve_pricing_rates(provider: str, model: str) -> PricingRates:
    provider_key = str(provider or "openai").strip().lower() or "openai"
    fallback = _DEFAULT_PROVIDER_RATES.get(provider_key, _DEFAULT_PROVIDER_RATES["openai"])

    provider_tag = provider_key.upper().replace("-", "_")
    rates = PricingRates(
        prompt_per_mtok=_safe_float_env(
            f"LIAGENT_COST_{provider_tag}_PROMPT_PER_MTOK",
            fallback.prompt_per_mtok,
        ),
        completion_per_mtok=_safe_float_env(
            f"LIAGENT_COST_{provider_tag}_COMPLETION_PER_MTOK",
            fallback.completion_per_mtok,
        ),
        cached_prompt_per_mtok=_safe_float_env(
            f"LIAGENT_COST_{provider_tag}_CACHED_PROMPT_PER_MTOK",
            fallback.cached_prompt_per_mtok,
        ),
        cache_write_per_mtok=_safe_float_env(
            f"LIAGENT_COST_{provider_tag}_CACHE_WRITE_PER_MTOK",
            fallback.cache_write_per_mtok,
        ),
    )

    global_prompt = _safe_float_env("LIAGENT_COST_DEFAULT_PROMPT_PER_MTOK", rates.prompt_per_mtok)
    global_completion = _safe_float_env(
        "LIAGENT_COST_DEFAULT_COMPLETION_PER_MTOK", rates.completion_per_mtok
    )
    global_cached_prompt = _safe_float_env(
        "LIAGENT_COST_DEFAULT_CACHED_PROMPT_PER_MTOK", rates.cached_prompt_per_mtok
    )
    global_cache_write = _safe_float_env(
        "LIAGENT_COST_DEFAULT_CACHE_WRITE_PER_MTOK", rates.cache_write_per_mtok
    )
    rates = PricingRates(
        prompt_per_mtok=global_prompt if os.environ.get("LIAGENT_COST_DEFAULT_PROMPT_PER_MTOK") else rates.prompt_per_mtok,
        completion_per_mtok=global_completion if os.environ.get("LIAGENT_COST_DEFAULT_COMPLETION_PER_MTOK") else rates.completion_per_mtok,
        cached_prompt_per_mtok=global_cached_prompt if os.environ.get("LIAGENT_COST_DEFAULT_CACHED_PROMPT_PER_MTOK") else rates.cached_prompt_per_mtok,
        cache_write_per_mtok=global_cache_write if os.environ.get("LIAGENT_COST_DEFAULT_CACHE_WRITE_PER_MTOK") else rates.cache_write_per_mtok,
    )

    model_key = str(model or "").strip().lower()
    if model_key:
        overrides = _model_overrides()
        for prefix, override_rates in overrides.items():
            if model_key.startswith(prefix):
                rates = override_rates
                break
    return rates


def estimate_usage_cost_usd(
    *,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    rates = resolve_pricing_rates(provider, model)

    prompt = max(0, int(prompt_tokens or 0))
    completion = max(0, int(completion_tokens or 0))
    cached_prompt = max(0, min(prompt, int(cached_prompt_tokens or 0)))
    cache_write = max(0, int(cache_write_tokens or 0))
    uncached_prompt = max(0, prompt - cached_prompt)

    return round(
        (uncached_prompt * rates.prompt_per_mtok)
        / 1_000_000.0
        + (cached_prompt * rates.cached_prompt_per_mtok) / 1_000_000.0
        + (cache_write * rates.cache_write_per_mtok) / 1_000_000.0
        + (completion * rates.completion_per_mtok) / 1_000_000.0,
        8,
    )
