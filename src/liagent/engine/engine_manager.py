"""Engine manager — creates, holds, and hot-swaps all backends."""

import asyncio
import gc
import os
import resource
import time
from collections.abc import AsyncIterator
from pathlib import Path

from ..config import AppConfig, LLMConfig, TTSConfig, MODELS_BASE_DIR
from ..logging import get_logger
from .base import LLMBackend, TTSBackend
from .cost_model import estimate_usage_cost_usd
from .provider_registry import (
    infer_api_model_family,
    infer_api_provider,
    infer_api_tool_protocol,
)
from .runtime import shutdown_mlx_runtime
from .stt import STTEngine
from .tool_format import get_parser_for_protocol, ToolCallParser

_log = get_logger("engine_manager")


class EngineManager:
    def __init__(self, config: AppConfig, *, voice_mode: bool = False):
        self.config = config
        self._start_time = time.time()
        self.config.llm = self._normalize_llm_config(self.config.llm)
        self.config.tts = self._normalize_tts_config(self.config.tts)
        self.llm: LLMBackend = self._create_llm(config.llm)
        # TTS: initialize when enabled in config (model is lazy-loaded on first use)
        self.tts: TTSBackend | None = (
            self._create_tts(self.config.tts) if config.tts_enabled else None
        )
        self.stt: STTEngine = self._create_stt(config.stt)
        self._llm_fallback: LLMBackend | None = None
        self._reasoning_llm: LLMBackend | None = None
        self.tool_parser: ToolCallParser = self._select_tool_parser(config.llm)
        self._llm_semaphore = asyncio.Semaphore(1)
        self._last_llm_usage: dict[str, int | str | bool | float] = {}
        self._cumulative_llm_usage: dict[str, int | str | bool | float] = {
            "provider": "",
            "model": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "cache_write_tokens": 0,
            "cache_hit_ratio": 0.0,
            "estimated_cost_usd": 0.0,
            "estimated": False,
        }
        self.configure_runtime_adapters()

    def configure_runtime_adapters(self):
        """Sync runtime-side adapters (sandbox, etc.) from current app config."""
        try:
            from ..tools.sandbox_runtime import configure_from_app_config

            configure_from_app_config(self.config)
        except Exception as exc:
            _log.warning(f"failed to configure runtime adapters: {exc}")

    # --- Factory methods ---

    @staticmethod
    def _normalize_tts_config(cfg: TTSConfig) -> TTSConfig:
        if cfg.backend == "api":
            cfg.voice_profile = "api_voice"
            cfg.api_voice = str(cfg.api_voice or "").strip() or "alloy"
            return cfg

        # Default to Qwen3-TTS-CustomVoice
        local_path = str(cfg.local_model_path or "").strip()
        qwen3_default = MODELS_BASE_DIR / "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
        if not local_path or "Kokoro" in local_path or "Base" in local_path:
            _log.info(
                f"TTS model path normalized: '{local_path}' -> '{qwen3_default}'"
            )
            cfg.local_model_path = str(qwen3_default)

        raw_lang = str(cfg.language or "zh").strip().lower()
        if raw_lang in {"chinese", "zh-cn", "mandarin"}:
            cfg.language = "zh"
        elif raw_lang in {"english", "en-us"}:
            cfg.language = "en"
        elif raw_lang in {"british", "en-gb"}:
            cfg.language = "en-gb"
        else:
            cfg.language = raw_lang or "zh"

        # Speaker name — must be a preset from CustomVoice model
        from .tts_qwen3 import DEFAULT_SPEAKER
        raw_speaker = str(getattr(cfg, "speaker_name", "") or "").strip().lower()
        cfg.speaker_name = raw_speaker if raw_speaker else DEFAULT_SPEAKER

        try:
            speed = float(getattr(cfg, "speed", 1.0))
        except (TypeError, ValueError):
            speed = 1.0
        cfg.speed = max(0.5, min(2.0, speed))

        # Clamp TTS generation params to safe ranges — values outside these
        # cause noise bursts, timbre drift, or degenerate output
        try:
            cfg.temperature = max(0.05, min(0.5, float(cfg.temperature or 0.3)))
        except (TypeError, ValueError):
            cfg.temperature = 0.3
        try:
            cfg.top_p = max(0.1, min(0.95, float(cfg.top_p or 0.8)))
        except (TypeError, ValueError):
            cfg.top_p = 0.8
        try:
            cfg.top_k = max(1, min(100, int(cfg.top_k or 20)))
        except (TypeError, ValueError):
            cfg.top_k = 20
        try:
            cfg.repetition_penalty = max(1.0, min(2.0, float(cfg.repetition_penalty or 1.05)))
        except (TypeError, ValueError):
            cfg.repetition_penalty = 1.05

        return cfg

    @staticmethod
    def _select_tool_parser(cfg: LLMConfig) -> ToolCallParser:
        return get_parser_for_protocol(
            str(getattr(cfg, "tool_protocol", "auto") or "auto"),
            model_family=str(getattr(cfg, "model_family", "") or ""),
        )

    @staticmethod
    def _normalize_llm_config(cfg: LLMConfig) -> LLMConfig:
        """Normalize LLM backend/family fields for stable runtime behavior."""
        cfg.backend = str(getattr(cfg, "backend", "local") or "local").strip().lower()
        if cfg.backend not in {"local", "api"}:
            cfg.backend = "local"
        cfg.model_family = str(getattr(cfg, "model_family", "glm47") or "glm47").strip().lower()
        cfg.tool_protocol = str(getattr(cfg, "tool_protocol", "auto") or "auto").strip().lower()
        if cfg.tool_protocol not in {"auto", "openai_function", "native_xml", "json_xml"}:
            cfg.tool_protocol = "auto"
        cfg.api_cache_mode = str(getattr(cfg, "api_cache_mode", "implicit") or "implicit").strip().lower()
        if cfg.api_cache_mode not in {"off", "implicit", "explicit"}:
            cfg.api_cache_mode = "implicit"
        cfg.api_cache_policy = str(getattr(cfg, "api_cache_policy", "tiered") or "tiered").strip().lower()
        if cfg.api_cache_policy not in {"flat", "tiered"}:
            cfg.api_cache_policy = "tiered"
        try:
            cfg.api_cache_ttl_sec = max(60, int(getattr(cfg, "api_cache_ttl_sec", 600) or 600))
        except (TypeError, ValueError):
            cfg.api_cache_ttl_sec = 600
        try:
            cfg.api_cache_ttl_static_sec = max(
                cfg.api_cache_ttl_sec,
                int(getattr(cfg, "api_cache_ttl_static_sec", 3600) or 3600),
            )
        except (TypeError, ValueError):
            cfg.api_cache_ttl_static_sec = max(cfg.api_cache_ttl_sec, 3600)
        try:
            cfg.api_cache_ttl_memory_sec = max(
                60,
                min(
                    cfg.api_cache_ttl_static_sec,
                    int(getattr(cfg, "api_cache_ttl_memory_sec", 900) or 900),
                ),
            )
        except (TypeError, ValueError):
            cfg.api_cache_ttl_memory_sec = min(cfg.api_cache_ttl_static_sec, 900)
        try:
            cfg.api_cache_min_prefix_chars = max(
                120,
                int(getattr(cfg, "api_cache_min_prefix_chars", 400) or 400),
            )
        except (TypeError, ValueError):
            cfg.api_cache_min_prefix_chars = 400

        if cfg.backend == "api" and cfg.model_family in {"", "auto", "glm47"}:
            cfg.model_family = infer_api_model_family(
                str(getattr(cfg, "api_model", "") or ""),
                str(getattr(cfg, "api_base_url", "") or ""),
            )
        elif not cfg.model_family:
            cfg.model_family = "glm47"

        if cfg.backend == "api":
            if cfg.tool_protocol in {"", "auto"}:
                cfg.tool_protocol = infer_api_tool_protocol(
                    str(getattr(cfg, "api_model", "") or ""),
                    str(getattr(cfg, "api_base_url", "") or ""),
                )
        elif cfg.tool_protocol in {"", "auto"}:
            fam = str(cfg.model_family or "").strip().lower()
            cfg.tool_protocol = "json_xml" if fam in {"qwen3-vl", "llama", "deepseek"} else "native_xml"

        return cfg

    @staticmethod
    def _create_llm(cfg: LLMConfig) -> LLMBackend:
        if cfg.backend == "api":
            from .vlm_api import ApiVLM

            return ApiVLM(
                cfg.api_base_url,
                cfg.api_key,
                cfg.api_model,
                cache_mode=cfg.api_cache_mode,
                cache_policy=cfg.api_cache_policy,
                cache_ttl_sec=cfg.api_cache_ttl_sec,
                cache_ttl_static_sec=cfg.api_cache_ttl_static_sec,
                cache_ttl_memory_sec=cfg.api_cache_ttl_memory_sec,
                cache_min_prefix_chars=cfg.api_cache_min_prefix_chars,
            )
        else:
            from .vlm_local import LocalVLM

            return LocalVLM(cfg.local_model_path)

    @staticmethod
    def _create_tts(cfg: TTSConfig) -> TTSBackend:
        cfg = EngineManager._normalize_tts_config(cfg)
        if cfg.backend == "api":
            from .tts_api import ApiTTS

            return ApiTTS(cfg.api_base_url, cfg.api_key, cfg.api_model, cfg.api_voice)
        else:
            from .tts_qwen3 import Qwen3TTS

            return Qwen3TTS(
                cfg.local_model_path,
                speaker_name=cfg.speaker_name,
                language=cfg.language,
                speed=cfg.speed,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty,
            )

    @staticmethod
    def _create_stt(cfg) -> STTEngine:
        backend = str(getattr(cfg, "backend", "local") or "local").strip().lower()
        if backend not in {"local", "api"}:
            backend = "local"
        model = str(getattr(cfg, "model", "") or "")
        language = str(getattr(cfg, "language", "auto") or "auto")
        api_base_url = str(getattr(cfg, "api_base_url", "") or "")
        api_key = str(getattr(cfg, "api_key", "") or "")
        api_model = str(getattr(cfg, "api_model", "") or "")
        return STTEngine(
            model=model,
            language=language,
            backend=backend,
            api_base_url=api_base_url,
            api_key=api_key,
            api_model=api_model,
        )

    # --- Hot-swap ---

    def switch_llm(self, new_cfg: LLMConfig):
        """Unload current LLM and load a new one."""
        new_cfg = self._normalize_llm_config(new_cfg)
        if self.llm:
            self.llm.unload()
            del self.llm
            gc.collect()
        if self._llm_fallback:
            self._llm_fallback.unload()
            self._llm_fallback = None
        self.llm = self._create_llm(new_cfg)
        self.tool_parser = self._select_tool_parser(new_cfg)
        self.config.llm = new_cfg
        self.config.save()

    def switch_tts(self, new_cfg: TTSConfig):
        """Unload current TTS and load a new one."""
        new_cfg = self._normalize_tts_config(new_cfg)
        if self.tts:
            self.tts.unload()
            del self.tts
            gc.collect()
            self.tts = None
        if self.config.tts_enabled:
            self.tts = self._create_tts(new_cfg)
        self.config.tts = new_cfg
        self.config.save()

    def set_tts_enabled(self, enabled: bool):
        """Enable/disable TTS and keep runtime instance in sync."""
        target = bool(enabled)
        self.config.tts_enabled = target
        if not target:
            if self.tts:
                self.tts.unload()
                del self.tts
                gc.collect()
                self.tts = None
            self.config.save()
            return

        if self.tts is None:
            try:
                self.tts = self._create_tts(self.config.tts)
            except Exception:
                self.config.tts_enabled = False
                self.config.save()
                raise
        self.config.save()

    def switch_stt(self, new_cfg):
        """Unload current STT and load a new one."""
        if self.stt:
            try:
                self.stt.unload()
            except Exception:
                pass
        self.stt = self._create_stt(new_cfg)
        self.config.stt = new_cfg
        self.config.save()

    def _can_fallback_api(self) -> bool:
        cfg = self.config.llm
        return (
            cfg.backend == "local"
            and bool(cfg.api_base_url and cfg.api_key and cfg.api_model)
        )

    def _runtime_mode(self) -> str:
        mode = str(getattr(self.config, "runtime_mode", "hybrid_balanced") or "hybrid_balanced").strip().lower()
        if mode not in {"local_private", "hybrid_balanced", "cloud_performance"}:
            return "hybrid_balanced"
        return mode

    def _ensure_fallback_llm(self) -> LLMBackend:
        if self._llm_fallback is None:
            from .vlm_api import ApiVLM

            cfg = self.config.llm
            self._llm_fallback = ApiVLM(
                cfg.api_base_url,
                cfg.api_key,
                cfg.api_model,
                cache_mode=cfg.api_cache_mode,
                cache_policy=cfg.api_cache_policy,
                cache_ttl_sec=cfg.api_cache_ttl_sec,
                cache_ttl_static_sec=cfg.api_cache_ttl_static_sec,
                cache_ttl_memory_sec=cfg.api_cache_ttl_memory_sec,
                cache_min_prefix_chars=cfg.api_cache_min_prefix_chars,
            )
        return self._llm_fallback

    @staticmethod
    def _extract_backend_usage(
        backend: LLMBackend | None,
    ) -> dict[str, int | str | bool | float]:
        if backend is None:
            return {}
        raw = getattr(backend, "last_usage", None)
        if not isinstance(raw, dict):
            return {}
        prompt_tokens = int(raw.get("prompt_tokens", 0) or 0)
        completion_tokens = int(raw.get("completion_tokens", 0) or 0)
        total_tokens = int(raw.get("total_tokens", 0) or 0)
        cached_prompt_tokens = int(raw.get("cached_prompt_tokens", 0) or 0)
        cache_write_tokens = int(raw.get("cache_write_tokens", 0) or 0)
        cached_prompt_tokens = max(0, min(prompt_tokens, cached_prompt_tokens))
        cache_hit_ratio = (
            float(cached_prompt_tokens) / float(prompt_tokens)
            if prompt_tokens > 0
            else 0.0
        )
        out = {
            "provider": str(raw.get("provider", "") or ""),
            "model": str(raw.get("model", "") or ""),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "cache_write_tokens": max(0, cache_write_tokens),
            "cache_hit_ratio": round(float(raw.get("cache_hit_ratio", cache_hit_ratio) or cache_hit_ratio), 4),
            "estimated": bool(raw.get("estimated", False)),
        }
        if out["total_tokens"] <= 0:
            return {}
        return out

    def get_last_llm_usage(self) -> dict[str, int | str | bool | float]:
        return dict(self._last_llm_usage)

    def get_cumulative_llm_usage(self) -> dict[str, int | str | bool | float]:
        return dict(self._cumulative_llm_usage)

    def reset_llm_usage_counters(self):
        self._last_llm_usage = {}
        self._cumulative_llm_usage = {
            "provider": "",
            "model": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "cache_write_tokens": 0,
            "cache_hit_ratio": 0.0,
            "estimated_cost_usd": 0.0,
            "estimated": False,
        }

    def _accumulate_llm_usage(self, usage: dict[str, int | str | bool | float]):
        if not usage:
            return
        p = int(usage.get("prompt_tokens", 0) or 0)
        c = int(usage.get("completion_tokens", 0) or 0)
        t = int(usage.get("total_tokens", 0) or 0)
        cp = int(usage.get("cached_prompt_tokens", 0) or 0)
        cw = int(usage.get("cache_write_tokens", 0) or 0)
        cp = max(0, min(p, cp))
        run_cost = float(usage.get("estimated_cost_usd", 0.0) or 0.0)
        if t <= 0:
            return
        self._cumulative_llm_usage["provider"] = str(usage.get("provider", "") or "")
        self._cumulative_llm_usage["model"] = str(usage.get("model", "") or "")
        self._cumulative_llm_usage["prompt_tokens"] = int(
            self._cumulative_llm_usage.get("prompt_tokens", 0) or 0
        ) + p
        self._cumulative_llm_usage["completion_tokens"] = int(
            self._cumulative_llm_usage.get("completion_tokens", 0) or 0
        ) + c
        self._cumulative_llm_usage["total_tokens"] = int(
            self._cumulative_llm_usage.get("total_tokens", 0) or 0
        ) + t
        self._cumulative_llm_usage["cached_prompt_tokens"] = int(
            self._cumulative_llm_usage.get("cached_prompt_tokens", 0) or 0
        ) + cp
        self._cumulative_llm_usage["cache_write_tokens"] = int(
            self._cumulative_llm_usage.get("cache_write_tokens", 0) or 0
        ) + max(0, cw)
        cum_prompt = int(self._cumulative_llm_usage.get("prompt_tokens", 0) or 0)
        cum_cached = int(self._cumulative_llm_usage.get("cached_prompt_tokens", 0) or 0)
        self._cumulative_llm_usage["cache_hit_ratio"] = round(
            (float(cum_cached) / float(cum_prompt)) if cum_prompt > 0 else 0.0,
            4,
        )
        self._cumulative_llm_usage["estimated_cost_usd"] = round(
            float(self._cumulative_llm_usage.get("estimated_cost_usd", 0.0) or 0.0)
            + max(0.0, run_cost),
            8,
        )
        self._cumulative_llm_usage["estimated"] = bool(
            self._cumulative_llm_usage.get("estimated", False)
            or usage.get("estimated", False)
        )

    def _estimate_usage_cost(self, usage: dict[str, int | str | bool | float]) -> float:
        if not usage:
            return 0.0
        provider = str(usage.get("provider", "") or "").strip().lower()
        if provider in {"", "local"}:
            return 0.0
        model = str(usage.get("model", "") or "").strip()
        if not model:
            model = str(getattr(self.config.llm, "api_model", "") or "").strip()
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        cached_prompt_tokens = int(usage.get("cached_prompt_tokens", 0) or 0)
        cache_write_tokens = int(usage.get("cache_write_tokens", 0) or 0)
        if prompt_tokens <= 0 and completion_tokens <= 0 and cache_write_tokens <= 0:
            return 0.0
        return estimate_usage_cost_usd(
            provider=provider or "openai",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            cache_write_tokens=cache_write_tokens,
        )

    def _capture_backend_usage(self, backend: LLMBackend | None):
        self._last_llm_usage = self._extract_backend_usage(backend)
        if self._last_llm_usage:
            self._last_llm_usage["estimated_cost_usd"] = self._estimate_usage_cost(
                self._last_llm_usage
            )
        self._accumulate_llm_usage(self._last_llm_usage)

    @staticmethod
    def _estimate_input_tokens(messages: list[dict], tools: list[dict] | None = None) -> int:
        chars = 0
        for msg in messages or []:
            chars += len(str(msg.get("content", "") or ""))
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                chars += len(str(tool_calls))
        if tools:
            chars += len(str(tools))
        return max(1, (chars + 3) // 4)

    def _project_api_cost_usd(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        tools: list[dict] | None = None,
    ) -> float:
        cfg = self.config.llm
        if not self._can_fallback_api():
            return 0.0
        prompt_tokens = self._estimate_input_tokens(messages, tools=tools)
        completion_tokens = max(128, min(int(max_tokens or 1), max(256, prompt_tokens // 2)))
        return estimate_usage_cost_usd(
            provider=infer_api_provider(
                str(getattr(cfg, "api_model", "") or ""),
                str(getattr(cfg, "api_base_url", "") or ""),
            ),
            model=str(getattr(cfg, "api_model", "") or ""),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=0,
            cache_write_tokens=0,
        )

    async def _generate_llm_unlocked(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        disable_thinking: bool = False,
    ) -> AsyncIterator[str]:
        """Generate with local-first strategy and optional API fallback."""
        safe_max_tokens = max(1, int(max_tokens or 1))
        try:
            safe_temperature = float(temperature)
        except (TypeError, ValueError):
            safe_temperature = 0.3
        if safe_temperature <= 0:
            safe_temperature = 0.1

        _gen_kwargs: dict = dict(
            images=images, max_tokens=safe_max_tokens,
            temperature=safe_temperature, tools=tools,
        )
        if disable_thinking:
            _gen_kwargs["disable_thinking"] = True

        runtime_mode = self._runtime_mode()
        if runtime_mode == "cloud_performance" and self._can_fallback_api():
            try:
                fallback = self._ensure_fallback_llm()
                async for token in fallback.generate(messages, **_gen_kwargs):
                    yield token
                self._capture_backend_usage(fallback)
                return
            except Exception as exc:
                _log.warning(f"cloud-performance API-first failed, fallback to local: {exc}")

        try:
            async for token in self.llm.generate(messages, **_gen_kwargs):
                yield token
            self._capture_backend_usage(self.llm)
            return
        except Exception:
            if runtime_mode == "local_private" or not self._can_fallback_api():
                raise

        fallback = self._ensure_fallback_llm()
        async for token in fallback.generate(messages, **_gen_kwargs):
            yield token
        self._capture_backend_usage(fallback)

    async def generate_llm(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        async with self._llm_semaphore:
            async for token in self._generate_llm_unlocked(
                messages, images, max_tokens, temperature, tools=tools
            ):
                yield token

    async def _generate_llm_routed_unlocked(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        *,
        service_tier: str = "standard_chat",
        planning_enabled: bool = False,
    ) -> AsyncIterator[str]:
        """Active model routing: choose backend based on task complexity.

        Gated by LIAGENT_ACTIVE_MODEL_ROUTING env var (default: false).
        When disabled, falls through to generate_llm().
        """
        active_routing = os.environ.get(
            "LIAGENT_ACTIVE_MODEL_ROUTING", "false"
        ).strip().lower() in {"1", "true", "yes"}

        if not active_routing or not self._can_fallback_api():
            async for token in self._generate_llm_unlocked(
                messages,
                images=images,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
            ):
                yield token
            return

        # Routing decision: deep_task + planning → prefer API; else prefer local
        prefer_api = service_tier == "deep_task" and planning_enabled
        projected_api_cost = self._project_api_cost_usd(
            messages,
            max_tokens=max_tokens,
            tools=tools,
        )
        soft_cost_cap = max(
            0.0,
            float(os.environ.get("LIAGENT_ROUTE_SOFT_API_COST_USD", "0.020")),
        )
        hard_cost_cap = max(
            soft_cost_cap,
            float(os.environ.get("LIAGENT_ROUTE_HARD_API_COST_USD", "0.060")),
        )
        if prefer_api and projected_api_cost >= hard_cost_cap:
            _log.trace(
                "routed_llm",
                route="force_local_by_cost",
                projected_api_cost=projected_api_cost,
                hard_cost_cap=hard_cost_cap,
                service_tier=service_tier,
            )
            prefer_api = False
        elif prefer_api and projected_api_cost >= soft_cost_cap:
            # Under moderate cost pressure, only deep tasks keep API preference.
            if service_tier != "deep_task":
                prefer_api = False
            _log.trace(
                "routed_llm",
                route="soft_cost_pressure",
                projected_api_cost=projected_api_cost,
                soft_cost_cap=soft_cost_cap,
                service_tier=service_tier,
            )
        safe_max_tokens = max(1, int(max_tokens or 1))
        try:
            safe_temperature = float(temperature)
        except (TypeError, ValueError):
            safe_temperature = 0.3
        if safe_temperature <= 0:
            safe_temperature = 0.1

        if prefer_api:
            # Try API first, fallback to local
            try:
                fallback = self._ensure_fallback_llm()
                async for token in fallback.generate(
                    messages, images=images,
                    max_tokens=safe_max_tokens, temperature=safe_temperature,
                    tools=tools,
                ):
                    yield token
                self._capture_backend_usage(fallback)
                return
            except Exception as e:
                _log.warning(f"API LLM failed, falling back to local: {e}")
            # Fallback to local
            async for token in self.llm.generate(
                messages, images=images,
                max_tokens=safe_max_tokens, temperature=safe_temperature,
                tools=tools,
            ):
                yield token
            self._capture_backend_usage(self.llm)
        else:
            # Try local first, fallback to API
            try:
                async for token in self.llm.generate(
                    messages, images=images,
                    max_tokens=safe_max_tokens, temperature=safe_temperature,
                    tools=tools,
                ):
                    yield token
                self._capture_backend_usage(self.llm)
                return
            except Exception as e:
                _log.warning(f"Local LLM failed, falling back to API: {e}")
            fallback = self._ensure_fallback_llm()
            async for token in fallback.generate(
                messages, images=images,
                max_tokens=safe_max_tokens, temperature=safe_temperature,
                tools=tools,
            ):
                yield token
            self._capture_backend_usage(fallback)

    async def generate_llm_routed(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        *,
        service_tier: str = "standard_chat",
        planning_enabled: bool = False,
    ) -> AsyncIterator[str]:
        async with self._llm_semaphore:
            async for token in self._generate_llm_routed_unlocked(
                messages, images, max_tokens, temperature, tools,
                service_tier=service_tier, planning_enabled=planning_enabled,
            ):
                yield token

    # --- Reasoning / text-chat model (GLM-4.7-Flash) ---

    def _reasoning_backend_mode(self) -> str:
        """Resolve reasoning backend policy.

        Modes:
        - local_prefer: try local reasoning model first, fallback to LLM/API.
        - api_only: skip local reasoning model and use active LLM path directly.
        """
        raw = str(os.environ.get("LIAGENT_REASONING_BACKEND", "auto") or "auto").strip().lower()
        if raw in {"api_only", "api"}:
            return "api_only"
        if raw in {"local_prefer", "local"}:
            return "local_prefer"
        # Auto policy: API backend implies API-only reasoning path by default.
        backend = str(getattr(self.config.llm, "backend", "local") or "local").strip().lower()
        if backend == "api":
            return "api_only"
        return "local_prefer"

    async def _generate_reasoning_via_llm_unlocked(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None = None,
        disable_thinking: bool = False,
    ) -> str:
        chunks: list[str] = []
        async for token in self._generate_llm_unlocked(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            disable_thinking=disable_thinking,
        ):
            chunks.append(token)
        return "".join(chunks)

    def unload_reasoning_llm(self):
        """Unload reasoning model to free ~16GB memory (e.g. before voice mode)."""
        if self._reasoning_llm is not None:
            self._reasoning_llm.unload()
            self._reasoning_llm = None
            gc.collect()
            _log.trace("reasoning_llm_unloaded", reason="voice_mode")

    def _ensure_reasoning_llm(self) -> "LLMBackend":
        if self._reasoning_llm is None:
            coder_path = os.environ.get(
                "LIAGENT_REASONING_MODEL_PATH",
                str(MODELS_BASE_DIR / "mlx-community:GLM-4.7-Flash-4bit"),
            )
            if not Path(coder_path).exists():
                raise FileNotFoundError(f"Reasoning/text model not found: {coder_path}")
            from .reasoning_local import LocalReasoning

            self._reasoning_llm = LocalReasoning(coder_path)
        return self._reasoning_llm

    async def _generate_reasoning_unlocked(
        self, messages: list[dict], *, max_tokens: int = 4096, temperature: float = 0.4,
        enable_thinking: bool = True,
    ) -> str:
        """Use reasoning model for tasks with API/VLM fallback when unavailable."""
        if self._reasoning_backend_mode() == "api_only":
            # Reasoning-capable API models (e.g. Kimi k2.5, DeepSeek-R1) share
            # max_tokens between internal reasoning and visible output.  Inflate
            # the budget so callers requesting small outputs still get results.
            api_max_tokens = max(max_tokens, min(max_tokens * 4, 4096))
            result = await self._generate_reasoning_via_llm_unlocked(
                messages,
                max_tokens=api_max_tokens,
                temperature=temperature,
            )
            if result and "<think>" in result[:20]:
                idx = result.find("</think>")
                if idx >= 0:
                    result = result[idx + len("</think>"):].strip()
            return result
        try:
            llm = self._ensure_reasoning_llm()
            result = await llm.generate(
                messages, max_tokens=max_tokens, temperature=temperature,
                enable_thinking=enable_thinking,
            )
        except (FileNotFoundError, RuntimeError, AttributeError) as e:
            _log.warning(
                f"reasoning LLM unavailable, falling back to VLM/API: {e}",
                fallback="vlm_or_api",
            )
            result = await self._generate_reasoning_via_llm_unlocked(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        # Strip <think>...</think> block — callers expect clean text/JSON
        if result and "<think>" in result[:20]:
            idx = result.find("</think>")
            if idx >= 0:
                result = result[idx + len("</think>"):].strip()
        return result

    async def generate_reasoning(
        self, messages: list[dict], *, max_tokens: int = 4096, temperature: float = 0.4,
        enable_thinking: bool = True,
    ) -> str:
        async with self._llm_semaphore:
            return await self._generate_reasoning_unlocked(
                messages, max_tokens=max_tokens, temperature=temperature,
                enable_thinking=enable_thinking,
            )

    async def try_generate_reasoning(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.4,
        enable_thinking: bool = True,
        timeout: float = 0.05,
    ) -> str | None:
        """Try to acquire LLM lock and generate reasoning.
        Returns None if lock unavailable within timeout."""
        try:
            await asyncio.wait_for(self._llm_semaphore.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        try:
            return await self._generate_reasoning_unlocked(
                messages, max_tokens=max_tokens, temperature=temperature,
                enable_thinking=enable_thinking,
            )
        finally:
            self._llm_semaphore.release()

    async def generate_extraction(
        self, messages: list[dict], *, max_tokens: int = 256, temperature: float = 0.1,
    ) -> str:
        """Lightweight LLM call for structured extraction. No reasoning inflation.

        Unlike generate_reasoning(), this passes exact max_tokens to the API
        instead of inflating by 4x. Ideal for short JSON extraction tasks.
        """
        async with self._llm_semaphore:
            if self._reasoning_backend_mode() == "api_only":
                result = await self._generate_reasoning_via_llm_unlocked(
                    messages, max_tokens=max_tokens, temperature=temperature,
                    disable_thinking=True,
                )
            else:
                try:
                    llm = self._ensure_reasoning_llm()
                    result = await llm.generate(
                        messages, max_tokens=max_tokens, temperature=temperature,
                        enable_thinking=False,
                    )
                except (FileNotFoundError, RuntimeError, AttributeError) as e:
                    _log.warning(
                        f"reasoning LLM unavailable for extraction, fallback: {e}",
                    )
                    result = await self._generate_reasoning_via_llm_unlocked(
                        messages, max_tokens=max_tokens, temperature=temperature,
                        disable_thinking=True,
                    )
            # Strip <think> tags from API output
            if result and "<think>" in result[:20]:
                idx = result.find("</think>")
                if idx >= 0:
                    result = result[idx + len("</think>"):].strip()
            return result

    async def _generate_text_unlocked(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.4,
        tools: list[dict] | None = None,
        enable_thinking: bool = True,
    ) -> AsyncIterator[str]:
        """Generate text with reasoning model, falling back to 4B VLM on failure.

        Reuses the reasoning model instance to avoid loading twice.
        """
        if self._reasoning_backend_mode() == "api_only":
            async for token in self._generate_llm_unlocked(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
            ):
                yield token
            return
        try:
            llm = self._ensure_reasoning_llm()
            result = await llm.generate(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                enable_thinking=enable_thinking,
            )
            yield result
        except (FileNotFoundError, RuntimeError) as e:
            _log.warning(
                f"reasoning LLM unavailable, falling back to VLM: {e}",
                fallback="vlm",
            )
            async for token in self._generate_llm_unlocked(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
            ):
                yield token

    async def _stream_text_unlocked(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.4,
        tools: list[dict] | None = None,
        enable_thinking: bool = True,
    ) -> AsyncIterator[str]:
        """Stream tokens from reasoning model, falling back to batch generate_text on error."""
        if self._reasoning_backend_mode() == "api_only":
            async for token in self._generate_llm_unlocked(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
            ):
                yield token
            return
        try:
            llm = self._ensure_reasoning_llm()
            async for token in llm.stream_generate(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                enable_thinking=enable_thinking,
            ):
                yield token
        except (FileNotFoundError, RuntimeError, AttributeError) as e:
            _log.warning(
                f"stream_text unavailable, falling back to batch generate_text: {e}",
                fallback="batch",
            )
            async for token in self._generate_text_unlocked(
                messages, max_tokens=max_tokens,
                temperature=temperature, tools=tools,
                enable_thinking=enable_thinking,
            ):
                yield token

    async def stream_text(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.4,
        tools: list[dict] | None = None,
        enable_thinking: bool = True,
    ) -> AsyncIterator[str]:
        async with self._llm_semaphore:
            async for token in self._stream_text_unlocked(
                messages, max_tokens=max_tokens, temperature=temperature,
                tools=tools, enable_thinking=enable_thinking,
            ):
                yield token

    def llm_status(self) -> str:
        cfg = self.config.llm
        if cfg.backend == "api":
            return f"API ({cfg.api_model})"
        return f"Local ({cfg.local_model_path.split('/')[-1]})"

    def tts_status(self) -> str:
        if not self.tts:
            return "Disabled"
        cfg = self.config.tts
        if cfg.backend == "api":
            return f"API ({cfg.api_model}, {cfg.api_voice})"
        speaker = str(getattr(cfg, "speaker_name", "") or "serena")
        lang = str(cfg.language or "zh")
        speed = getattr(cfg, "speed", 1.0)
        try:
            speed_label = f"{float(speed):.2f}x"
        except (TypeError, ValueError):
            speed_label = "1.00x"
        return f"Local (qwen3-tts:{speaker}, {lang}, {speed_label})"

    def stt_status(self) -> str:
        cfg = self.config.stt
        backend = str(getattr(cfg, "backend", "local") or "local").strip().lower()
        if backend == "api":
            return f"API ({getattr(cfg, 'api_model', '') or '-'})"
        model_path = str(getattr(cfg, "model", "") or "")
        model_name = model_path.split("/")[-1] if model_path else "-"
        return f"Local ({model_name})"

    def health_check(self) -> dict:
        """Return current system health metrics."""
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports KB
        if os.uname().sysname == "Darwin":
            rss_mb = rss_bytes / (1024 * 1024)
        else:
            rss_mb = rss_bytes / 1024
        return {
            "memory_rss_mb": round(rss_mb, 1),
            "llm_loaded": self.llm is not None,
            "reasoning_loaded": self._reasoning_llm is not None,
            "tts_loaded": self.tts is not None,
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }

    def maybe_gc(self, threshold_mb: int = 28000):
        """Trigger GC if RSS exceeds threshold (default 28GB for 32GB M5)."""
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Darwin":
            rss_mb = rss_bytes / (1024 * 1024)
        else:
            rss_mb = rss_bytes / 1024
        if rss_mb > threshold_mb:
            _log.warning(f"RSS {rss_mb:.0f}MB exceeds {threshold_mb}MB, triggering GC")
            gc.collect()

    def shutdown(self):
        """Release model/runtime resources during process exit."""
        try:
            if self.llm:
                self.llm.unload()
        except Exception as e:
            _log.error("engine_manager", e, phase="shutdown", component="llm")
        try:
            if self._llm_fallback:
                self._llm_fallback.unload()
                self._llm_fallback = None
        except Exception as e:
            _log.error("engine_manager", e, phase="shutdown", component="llm_fallback")
        try:
            if self._reasoning_llm:
                self._reasoning_llm.unload()
                self._reasoning_llm = None
        except Exception as e:
            _log.error("engine_manager", e, phase="shutdown", component="reasoning_llm")
        try:
            if self.tts:
                self.tts.unload()
        except Exception as e:
            _log.error("engine_manager", e, phase="shutdown", component="tts")
        try:
            if self.stt:
                self.stt.unload()
        except Exception as e:
            _log.error("engine_manager", e, phase="shutdown", component="stt")
        gc.collect()
        shutdown_mlx_runtime(wait=False)
