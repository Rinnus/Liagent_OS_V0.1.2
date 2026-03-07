"""OpenAI-compatible API VLM engine."""

import asyncio
import base64
import json
import os
import random
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path

from ..logging import get_logger
from .base import LLMBackend
from .provider_registry import (
    get_provider_profile,
    infer_api_provider,
)

_log = get_logger("vlm_api")


_TEMP_UNSET = object()


class ApiVLM(LLMBackend):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        cache_mode: str = "implicit",
        cache_policy: str = "tiered",
        cache_ttl_sec: int = 600,
        cache_ttl_static_sec: int | None = None,
        cache_ttl_memory_sec: int | None = None,
        cache_min_prefix_chars: int = 400,
    ):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.base_url = str(base_url or "")
        self.model = str(model or "").strip()
        self._loaded = True
        self._provider = self._infer_provider(self.base_url, self.model)
        self._profile = get_provider_profile(self._provider)
        self._supports_tools = bool(self._profile.supports_tools)
        self._supports_parallel_tool_calls = bool(self._profile.supports_parallel_tool_calls)
        self._supports_tool_choice = bool(self._profile.supports_tool_choice)
        self._supports_stream_usage = bool(self._profile.supports_stream_usage)
        self._max_tokens_field = str(self._profile.max_tokens_field or "max_tokens").strip() or "max_tokens"
        if self._max_tokens_field not in {"max_tokens", "max_completion_tokens"}:
            self._max_tokens_field = "max_tokens"
        self._temperature_policy = str(getattr(self._profile, "temperature_policy", "any") or "any").strip().lower()
        if self._temperature_policy not in {"any", "fixed", "clamp"}:
            self._temperature_policy = "any"
        self._temperature_min = getattr(self._profile, "temperature_min", None)
        self._temperature_max = getattr(self._profile, "temperature_max", None)
        self._default_forced_temperature: float | None = None
        self._forced_temperature_by_scope: dict[tuple[bool, bool, bool], float] = {}
        self._temperature_unsupported_scopes: set[tuple[bool, bool, bool]] = set()
        if self._temperature_policy == "fixed":
            fixed = getattr(self._profile, "temperature_fixed", None)
            if fixed is not None:
                try:
                    self._default_forced_temperature = float(fixed)
                except Exception:
                    self._default_forced_temperature = None
        self._cache_mode = str(cache_mode or "implicit").strip().lower()
        if self._cache_mode not in {"off", "implicit", "explicit"}:
            self._cache_mode = "implicit"
        self._cache_policy = str(cache_policy or "tiered").strip().lower()
        if self._cache_policy not in {"flat", "tiered"}:
            self._cache_policy = "tiered"
        self._cache_ttl_sec = max(60, int(cache_ttl_sec or 600))
        self._cache_ttl_static_sec = max(
            self._cache_ttl_sec,
            int(cache_ttl_static_sec or (self._cache_ttl_sec * 6)),
        )
        self._cache_ttl_memory_sec = max(
            60,
            min(
                self._cache_ttl_static_sec,
                int(cache_ttl_memory_sec or max(self._cache_ttl_sec, 900)),
            ),
        )
        self._cache_min_prefix_chars = max(120, int(cache_min_prefix_chars or 400))
        self._openrouter_base = (
            "openrouter.ai" in self.base_url.lower()
            or self._provider == "openrouter"
        )
        self._anthropic_base = "anthropic" in self.base_url.lower()

        self._retry_attempts = max(1, int(os.environ.get("LIAGENT_API_RETRY_ATTEMPTS", "3")))
        self._retry_backoff_base = max(
            0.05, float(os.environ.get("LIAGENT_API_RETRY_BACKOFF_BASE_SEC", "0.35"))
        )
        self._retry_backoff_max = max(
            self._retry_backoff_base, float(os.environ.get("LIAGENT_API_RETRY_BACKOFF_MAX_SEC", "4.0"))
        )
        self._circuit_threshold = max(
            1, int(os.environ.get("LIAGENT_API_CIRCUIT_BREAKER_THRESHOLD", "4"))
        )
        self._circuit_cooldown_sec = max(
            1.0, float(os.environ.get("LIAGENT_API_CIRCUIT_COOLDOWN_SEC", "12"))
        )
        self._circuit_cooldown_max_sec = max(
            self._circuit_cooldown_sec,
            float(
                os.environ.get(
                    "LIAGENT_API_CIRCUIT_COOLDOWN_MAX_SEC",
                    str(max(120.0, self._circuit_cooldown_sec)),
                )
            ),
        )
        self._failure_streak = 0
        self._circuit_open_until = 0.0
        self.last_usage: dict[str, int | str | bool] = {
            "provider": self._profile.name,
            "model": self.model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "cache_write_tokens": 0,
            "cache_hit_ratio": 0.0,
            "estimated": False,
        }

    @staticmethod
    def _infer_provider(base_url: str, model: str) -> str:
        return infer_api_provider(model, base_url)

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self):
        self._loaded = False

    def _build_messages(
        self, messages: list[dict], images: list[str] | None
    ) -> list[dict]:
        """Convert internal message format to OpenAI API format with images."""
        api_messages: list[dict] = []
        pending_tool_call_ids: list[str] = []
        last_user_idx = max(
            (i for i, msg in enumerate(messages) if msg.get("role") == "user"),
            default=-1,
        )

        def _flush_pending_tool_placeholders() -> None:
            while pending_tool_call_ids:
                tcid = pending_tool_call_ids.pop(0)
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tcid,
                        "content": "[Tool Result]\n(unavailable due to trimmed history)",
                    }
                )

        for idx, m in enumerate(messages):
            role = str(m.get("role", "") or "").strip().lower()
            content = m.get("content", "")
            text_content = str(content or "")

            # Strict providers require tool replies to immediately follow
            # assistant tool_calls. Backfill placeholders before any non-tool
            # message if history lost those tool replies.
            if role != "tool" and pending_tool_call_ids:
                _flush_pending_tool_placeholders()

            if role == "system":
                if text_content.strip():
                    api_messages.append({"role": "system", "content": text_content})
                continue

            if role == "assistant":
                tool_calls = m.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    normalized_calls: list[dict] = []
                    for tc_idx, tc in enumerate(tool_calls):
                        if not isinstance(tc, dict):
                            continue
                        name = str(tc.get("name", "")).strip()
                        args_raw = tc.get("arguments", {})
                        if not name:
                            fn = tc.get("function")
                            if isinstance(fn, dict):
                                name = str(fn.get("name", "")).strip()
                                if "arguments" in fn:
                                    args_raw = fn.get("arguments")
                        if not name:
                            continue
                        if isinstance(args_raw, str):
                            args_json = args_raw
                        else:
                            try:
                                args_json = json.dumps(args_raw or {}, ensure_ascii=False)
                            except Exception:
                                args_json = "{}"
                        call_id = str(tc.get("id", "")).strip() or f"call_{idx}_{tc_idx + 1}"
                        normalized_calls.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": args_json},
                            }
                        )
                    if normalized_calls:
                        pending_tool_call_ids.extend(
                            [str(tc.get("id", "") or "") for tc in normalized_calls if str(tc.get("id", "") or "")]
                        )
                        # Some providers reject empty assistant content. Keep a
                        # compact placeholder while preserving tool_calls context.
                        assistant_payload = {
                            "role": "assistant",
                            "content": text_content if text_content.strip() else "Tool call dispatched.",
                            "tool_calls": normalized_calls,
                        }
                        # Kimi/Moonshot may enforce thinking context for
                        # historical assistant tool-call messages.
                        if self._profile.name == "moonshot":
                            assistant_payload["reasoning_content"] = (
                                text_content if text_content.strip() else "Tool planning and selection."
                            )
                        api_messages.append(assistant_payload)
                        continue
                if text_content.strip():
                    api_messages.append({"role": "assistant", "content": text_content})
                continue

            if role == "tool":
                tool_text = text_content if text_content.strip() else "[Tool Result]\n(empty)"
                if pending_tool_call_ids:
                    tcid = pending_tool_call_ids.pop(0)
                    api_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tcid,
                            "content": tool_text,
                        }
                    )
                else:
                    # Internal memory can contain orphan tool messages. Keep them
                    # as user context for broad compatibility.
                    api_messages.append({"role": "user", "content": f"[Tool Result]\n{tool_text}"})
                continue

            if role != "user":
                continue

            is_last_user = idx == last_user_idx
            if is_last_user and images:
                content_block = [{"type": "text", "text": text_content}]
                for img_path in images:
                    img_data = Path(img_path).read_bytes()
                    b64 = base64.b64encode(img_data).decode()
                    suffix = Path(img_path).suffix.lstrip(".")
                    mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
                    content_block.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}"
                            },
                        }
                    )
                api_messages.append({"role": "user", "content": content_block})
            elif text_content.strip():
                api_messages.append({"role": "user", "content": text_content})
        if pending_tool_call_ids:
            _flush_pending_tool_placeholders()
        return api_messages

    @staticmethod
    def _coerce_dict(value) -> dict:
        if isinstance(value, dict):
            return value
        if value is None:
            return {}
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                maybe = to_dict()
                return maybe if isinstance(maybe, dict) else {}
            except Exception:
                return {}
        if hasattr(value, "__dict__"):
            try:
                raw = vars(value)
            except Exception:
                raw = {}
            return raw if isinstance(raw, dict) else {}
        return {}

    @staticmethod
    def _safe_json_env(name: str) -> dict:
        raw = str(os.environ.get(name, "") or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _message_chars(msg: dict) -> int:
        content = msg.get("content", "")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for item in content:
                if isinstance(item, dict):
                    total += len(str(item.get("text", "") or ""))
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        total += len(str(image_url.get("url", "") or ""))
                else:
                    total += len(str(item or ""))
            return total
        return len(str(content or ""))

    def _select_cache_ttl(self, api_messages: list[dict], tools: list[dict] | None) -> int:
        base_ttl = self._cache_ttl_sec
        if self._cache_mode != "explicit" or self._cache_policy != "tiered":
            return base_ttl

        system_chars = 0
        user_chars = 0
        for msg in api_messages or []:
            role = str(msg.get("role", "") or "").strip().lower()
            chars = self._message_chars(msg)
            if role == "system":
                system_chars += chars
            elif role == "user":
                user_chars += chars
        tools_chars = 0
        if tools:
            try:
                tools_chars = len(json.dumps(tools, ensure_ascii=False))
            except Exception:
                tools_chars = len(str(tools))

        stable_prefix_chars = system_chars + tools_chars
        if stable_prefix_chars >= self._cache_min_prefix_chars:
            return self._cache_ttl_static_sec
        if system_chars >= max(180, self._cache_min_prefix_chars // 2):
            return self._cache_ttl_memory_sec
        if user_chars <= max(120, self._cache_min_prefix_chars // 4):
            return max(base_ttl, self._cache_ttl_memory_sec)
        return base_ttl

    def _build_cache_headers(self, ttl_sec: int) -> dict[str, str]:
        if self._cache_mode != "explicit":
            return {}
        headers: dict[str, str] = {}
        if self._openrouter_base:
            headers["x-openrouter-cache-control"] = f"max-age={int(ttl_sec)}"
        if self._provider == "claude" or self._anthropic_base:
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        headers.update({
            str(k): str(v)
            for k, v in self._safe_json_env("LIAGENT_API_CACHE_EXTRA_HEADERS_JSON").items()
            if str(k).strip()
        })
        return headers

    def _build_cache_body(self, ttl_sec: int) -> dict:
        if self._cache_mode != "explicit":
            return {}
        body: dict = {
            "cache_control": {
                "type": "ephemeral",
                "ttl_seconds": int(ttl_sec),
            }
        }
        extra = self._safe_json_env("LIAGENT_API_CACHE_EXTRA_BODY_JSON")
        if extra:
            body.update(extra)
        return body

    def _build_request_kwargs(
        self,
        *,
        api_messages: list[dict],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None,
        stream: bool,
        disable_thinking: bool = False,
    ) -> dict:
        temp_scope = self._temperature_scope(
            disable_thinking=disable_thinking,
            tools=tools,
            stream=stream,
        )
        temp = self._coerce_temperature_for_request(temperature, scope=temp_scope)
        kwargs = {
            "model": self.model,
            "messages": api_messages,
            self._max_tokens_field: max_tokens,
            "stream": stream,
        }
        if temp is not None:
            kwargs["temperature"] = temp
        if stream and self._supports_stream_usage:
            kwargs["stream_options"] = {"include_usage": True}
        if tools and self._supports_tools:
            kwargs["tools"] = tools
            if self._supports_tool_choice:
                kwargs["tool_choice"] = "auto"
            if not self._supports_parallel_tool_calls:
                kwargs["parallel_tool_calls"] = False
        cache_ttl = self._select_cache_ttl(api_messages, tools)
        cache_headers = self._build_cache_headers(cache_ttl)
        if cache_headers:
            kwargs["extra_headers"] = cache_headers
        cache_body = self._build_cache_body(cache_ttl)
        if cache_body:
            kwargs["extra_body"] = cache_body
        # Disable server-side thinking/reasoning for extraction tasks
        if disable_thinking:
            extra = kwargs.get("extra_body") or {}
            extra["thinking"] = {"type": "disabled"}
            kwargs["extra_body"] = extra
        return kwargs

    @staticmethod
    def _temperature_scope(
        *,
        disable_thinking: bool,
        tools: list[dict] | None,
        stream: bool,
    ) -> tuple[bool, bool, bool]:
        return (
            bool(disable_thinking),
            bool(tools),
            bool(stream),
        )

    @classmethod
    def _temperature_scope_from_kwargs(cls, kwargs: dict) -> tuple[bool, bool, bool]:
        extra_body = kwargs.get("extra_body") or {}
        thinking_cfg = extra_body.get("thinking") if isinstance(extra_body, dict) else None
        disable_thinking = (
            isinstance(thinking_cfg, dict)
            and str(thinking_cfg.get("type", "")).strip().lower() == "disabled"
        )
        return cls._temperature_scope(
            disable_thinking=disable_thinking,
            tools=kwargs.get("tools"),
            stream=bool(kwargs.get("stream")),
        )

    def _get_scope_temperature(self, scope: tuple[bool, bool, bool]) -> object:
        if scope in self._temperature_unsupported_scopes:
            return None
        if scope in self._forced_temperature_by_scope:
            return float(self._forced_temperature_by_scope[scope])
        hinted = self._provider_scope_temperature_hint(scope)
        if hinted is not None:
            return float(hinted)
        if self._default_forced_temperature is not None:
            return float(self._default_forced_temperature)
        return _TEMP_UNSET

    def _provider_scope_temperature_hint(
        self,
        scope: tuple[bool, bool, bool],
    ) -> float | None:
        disable_thinking, _has_tools, _stream = scope
        provider = str(getattr(self._profile, "name", "") or "").strip().lower()
        model = str(self.model or "").strip().lower()
        if provider == "moonshot" and model.startswith("kimi-k2.5"):
            if disable_thinking:
                return 0.6
            return 1.0
        return None

    def _coerce_temperature_for_request(
        self,
        requested: float,
        *,
        scope: tuple[bool, bool, bool],
    ) -> float | None:
        forced = self._get_scope_temperature(scope)
        if forced is None:
            return None
        if forced is not _TEMP_UNSET:
            return float(forced)
        try:
            temp = float(requested)
        except Exception:
            temp = 0.7
        if self._temperature_policy in {"clamp", "fixed"}:
            if self._temperature_min is not None:
                try:
                    temp = max(temp, float(self._temperature_min))
                except Exception:
                    pass
            if self._temperature_max is not None:
                try:
                    temp = min(temp, float(self._temperature_max))
                except Exception:
                    pass
        return temp

    @staticmethod
    def _usage_to_dict(usage_obj) -> dict[str, int]:
        if usage_obj is None:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_prompt_tokens": 0,
                "cache_write_tokens": 0,
            }

        usage = ApiVLM._coerce_dict(usage_obj)
        if not usage:
            usage = {}

        p = int(
            usage.get("prompt_tokens", getattr(usage_obj, "prompt_tokens", 0)) or 0
        )
        c = int(
            usage.get("completion_tokens", getattr(usage_obj, "completion_tokens", 0)) or 0
        )
        t = int(usage.get("total_tokens", getattr(usage_obj, "total_tokens", 0)) or 0)
        if t <= 0:
            t = p + c

        prompt_details = ApiVLM._coerce_dict(
            usage.get("prompt_tokens_details", getattr(usage_obj, "prompt_tokens_details", None))
        )
        input_details = ApiVLM._coerce_dict(
            usage.get("input_tokens_details", getattr(usage_obj, "input_tokens_details", None))
        )
        cached_prompt = int(
            prompt_details.get("cached_tokens", 0)
            or prompt_details.get("cached_prompt_tokens", 0)
            or input_details.get("cached_tokens", 0)
            or input_details.get("cache_read_input_tokens", 0)
            or 0
        )
        cache_write = int(
            prompt_details.get("cache_write_tokens", 0)
            or input_details.get("cache_creation_input_tokens", 0)
            or 0
        )
        return {
            "prompt_tokens": max(0, p),
            "completion_tokens": max(0, c),
            "total_tokens": max(0, t),
            "cached_prompt_tokens": max(0, cached_prompt),
            "cache_write_tokens": max(0, cache_write),
        }

    @staticmethod
    def _estimate_tokens(chars: int) -> int:
        return max(1, (max(0, chars) + 3) // 4)

    def _set_last_usage(
        self,
        usage: dict[str, int] | None,
        *,
        estimated: bool,
        input_chars: int,
        output_chars: int,
    ):
        u = usage or {}
        total = int(u.get("total_tokens", 0) or 0)
        prompt = int(u.get("prompt_tokens", 0) or 0)
        completion = int(u.get("completion_tokens", 0) or 0)
        cached_prompt = int(u.get("cached_prompt_tokens", 0) or 0)
        cache_write = int(u.get("cache_write_tokens", 0) or 0)
        if total <= 0:
            prompt = self._estimate_tokens(input_chars)
            completion = self._estimate_tokens(output_chars)
            total = prompt + completion
            estimated = True
            cached_prompt = 0
            cache_write = 0
        cached_prompt = max(0, min(prompt, cached_prompt))
        cache_hit_ratio = float(cached_prompt) / float(prompt) if prompt > 0 else 0.0
        self.last_usage = {
            "provider": self._profile.name,
            "model": self.model,
            "prompt_tokens": max(0, prompt),
            "completion_tokens": max(0, completion),
            "total_tokens": max(0, total),
            "cached_prompt_tokens": cached_prompt,
            "cache_write_tokens": max(0, cache_write),
            "cache_hit_ratio": round(cache_hit_ratio, 4),
            "estimated": bool(estimated),
        }

    @staticmethod
    def _is_tools_error(exc: Exception) -> bool:
        text = str(exc).lower()
        hints = ("tools", "tool_choice", "parallel_tool_calls", "function call", "function calling")
        return any(h in text for h in hints) and any(k in text for k in ("unsupported", "invalid", "unknown", "not allowed", "not support"))

    @staticmethod
    def _required_temperature_from_error(exc: Exception) -> float | None:
        text = str(exc).lower()
        if "temperature" not in text:
            return None
        patterns = (
            r"only\s+([0-9]+(?:\.[0-9]+)?)\s+is allowed",
            r"temperature[^0-9\-]*must\s+be\s+([0-9]+(?:\.[0-9]+)?)",
            r"temperature[^0-9\-]*should\s+be\s+([0-9]+(?:\.[0-9]+)?)",
            r"temperature[^0-9\-]*expect(?:s|ed)?\s+([0-9]+(?:\.[0-9]+)?)",
        )
        for pat in patterns:
            m = re.search(pat, text)
            if not m:
                continue
            try:
                return float(m.group(1))
            except Exception:
                continue
        return None

    @staticmethod
    def _is_unsupported_param_error(exc: Exception, param: str) -> bool:
        text = str(exc).lower()
        token = str(param or "").strip().lower()
        if not token:
            return False
        if any(
            hint in text
            for hint in (
                "too large",
                "too long",
                "exceed",
                "exceeds",
                "greater than",
                "must be <=",
                "maximum",
            )
        ):
            return False
        return token in text and any(
            k in text
            for k in (
                "unsupported",
                "not support",
                "invalid",
                "unknown",
                "not allowed",
                "unrecognized",
                "unexpected",
            )
        )

    def _switch_max_tokens_field(self, kwargs: dict) -> str:
        current = str(self._max_tokens_field or "max_tokens").strip().lower()
        if current == "max_completion_tokens":
            if "max_completion_tokens" in kwargs:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            self._max_tokens_field = "max_tokens"
            return "switch_to_max_tokens"
        if current == "max_tokens":
            if "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            self._max_tokens_field = "max_completion_tokens"
            return "switch_to_max_completion_tokens"
        return ""

    def _apply_tools_runtime_downgrade(self, action: str):
        if action == "drop_parallel_tool_calls":
            self._supports_parallel_tool_calls = False
            return
        if action == "drop_tool_choice":
            self._supports_tool_choice = False
            return
        if action == "drop_tools":
            self._supports_tools = False
            self._supports_tool_choice = False
            self._supports_parallel_tool_calls = False

    @staticmethod
    def _downgrade_tools_kwargs(kwargs: dict) -> str:
        """Progressive tool-call downgrade for cross-provider compatibility."""
        if "parallel_tool_calls" in kwargs:
            kwargs.pop("parallel_tool_calls", None)
            return "drop_parallel_tool_calls"
        if "tool_choice" in kwargs:
            kwargs.pop("tool_choice", None)
            return "drop_tool_choice"
        if "tools" in kwargs:
            kwargs.pop("tools", None)
            kwargs.pop("tool_choice", None)
            kwargs.pop("parallel_tool_calls", None)
            return "drop_tools"
        return ""

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            if status_code == 429 or 500 <= status_code < 600:
                return True
        text = str(exc).lower()
        retry_hints = (
            "429",
            "rate limit",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "connection reset",
            "peer closed connection",
            "incomplete chunked read",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "internal server error",
            "502",
            "503",
            "504",
        )
        return any(h in text for h in retry_hints)

    def _circuit_open(self) -> bool:
        return time.time() < self._circuit_open_until

    def _record_success(self):
        self._failure_streak = 0
        self._circuit_open_until = 0.0

    def _record_failure(self):
        self._failure_streak += 1
        if self._failure_streak >= self._circuit_threshold:
            # Exponential cooldown after threshold: base * 2^(extra failures), capped.
            extra = max(0, self._failure_streak - self._circuit_threshold)
            cooldown = min(
                self._circuit_cooldown_max_sec,
                self._circuit_cooldown_sec * (2 ** extra),
            )
            self._circuit_open_until = time.time() + cooldown
            _log.warning(
                "api_circuit_open",
                provider=self._profile.name,
                model=self.model,
                failure_streak=self._failure_streak,
                cooldown_sec=round(float(cooldown), 3),
            )

    async def _sleep_backoff(self, attempt: int):
        exp = min(self._retry_backoff_max, self._retry_backoff_base * (2 ** max(0, attempt - 1)))
        jitter = exp * (0.2 + random.random() * 0.4)
        await asyncio.sleep(min(self._retry_backoff_max, exp + jitter))

    async def _create_with_retry(
        self,
        request_kwargs: dict,
        *,
        allow_tools_drop: bool = True,
    ):
        if self._circuit_open():
            raise RuntimeError(
                f"API circuit open for provider={self._profile.name}; retry later."
            )
        kwargs = dict(request_kwargs)
        attempt = 0
        while attempt < self._retry_attempts:
            attempt += 1
            try:
                return await self.client.chat.completions.create(**kwargs), kwargs
            except Exception as e:
                temp_scope = self._temperature_scope_from_kwargs(kwargs)
                required_temp = self._required_temperature_from_error(e)
                if "temperature" in kwargs and required_temp is not None:
                    try:
                        current_temp = float(kwargs.get("temperature"))
                    except Exception:
                        current_temp = required_temp
                    if abs(current_temp - required_temp) > 1e-6:
                        kwargs["temperature"] = required_temp
                        self._temperature_unsupported_scopes.discard(temp_scope)
                        self._forced_temperature_by_scope[temp_scope] = required_temp
                        _log.warning(
                            "api_temperature_runtime_clamp",
                            provider=self._profile.name,
                            model=self.model,
                            previous=current_temp,
                            required=required_temp,
                            error=str(e),
                        )
                        attempt = max(0, attempt - 1)
                        continue

                if "temperature" in kwargs and self._is_unsupported_param_error(e, "temperature"):
                    kwargs.pop("temperature", None)
                    self._forced_temperature_by_scope.pop(temp_scope, None)
                    self._temperature_unsupported_scopes.add(temp_scope)
                    _log.warning(
                        "api_temperature_unsupported_drop",
                        provider=self._profile.name,
                        model=self.model,
                        error=str(e),
                    )
                    attempt = max(0, attempt - 1)
                    continue

                if "stream_options" in kwargs and self._is_unsupported_param_error(e, "stream_options"):
                    kwargs.pop("stream_options", None)
                    self._supports_stream_usage = False
                    _log.warning(
                        "api_stream_options_unsupported_runtime_downgrade",
                        provider=self._profile.name,
                        model=self.model,
                        error=str(e),
                    )
                    attempt = max(0, attempt - 1)
                    continue

                if self._is_unsupported_param_error(e, self._max_tokens_field):
                    switched = self._switch_max_tokens_field(kwargs)
                    if switched:
                        _log.warning(
                            "api_max_tokens_field_runtime_switch",
                            provider=self._profile.name,
                            model=self.model,
                            action=switched,
                            error=str(e),
                        )
                        attempt = max(0, attempt - 1)
                        continue

                if allow_tools_drop and "tools" in kwargs and self._is_tools_error(e):
                    downgrade_action = self._downgrade_tools_kwargs(kwargs)
                    if not downgrade_action:
                        allow_tools_drop = False
                    else:
                        self._apply_tools_runtime_downgrade(downgrade_action)
                        _log.warning(
                            "api_tools_not_supported_progressive_downgrade",
                            provider=self._profile.name,
                            model=self.model,
                            action=downgrade_action,
                            error=str(e),
                        )
                        # Tool-shape downgrades are compatibility fallbacks, not transport retries.
                        attempt = max(0, attempt - 1)
                        continue
                    _log.warning(
                        "api_tools_not_supported_retry_without_tools",
                        provider=self._profile.name,
                        model=self.model,
                        error=str(e),
                    )

                retryable = self._is_retryable_error(e)
                if retryable and attempt < self._retry_attempts:
                    _log.warning(
                        "api_call_retry",
                        provider=self._profile.name,
                        model=self.model,
                        attempt=attempt,
                        error=str(e),
                    )
                    await self._sleep_backoff(attempt)
                    continue
                self._record_failure()
                raise
        raise RuntimeError("API call failed after retry exhaustion")

    @staticmethod
    def _extract_choice_payload(resp) -> tuple[str, list]:
        choices = getattr(resp, "choices", None)
        if not choices:
            return "", []
        msg = getattr(choices[0], "message", None)
        if msg is None and isinstance(choices[0], dict):
            msg = choices[0].get("message")
        if msg is None:
            return "", []
        content = getattr(msg, "content", None)
        tool_calls = getattr(msg, "tool_calls", None)
        if isinstance(msg, dict):
            content = msg.get("content", content)
            tool_calls = msg.get("tool_calls", tool_calls)
        text = ApiVLM._coerce_delta_content(content)
        return text, list(tool_calls or [])

    async def generate(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        api_messages = self._build_messages(messages, images)
        request_kwargs = self._build_request_kwargs(
            api_messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            stream=True,
            disable_thinking=kwargs.get("disable_thinking", False),
        )
        self.last_usage = {
            "provider": self._profile.name,
            "model": self.model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "cache_write_tokens": 0,
            "cache_hit_ratio": 0.0,
            "estimated": False,
        }

        input_chars = len(json.dumps(api_messages, ensure_ascii=False))
        if "tools" in request_kwargs:
            input_chars += len(json.dumps(request_kwargs["tools"], ensure_ascii=False))

        stream_response = None
        used_kwargs = dict(request_kwargs)
        try:
            stream_response, used_kwargs = await self._create_with_retry(
                request_kwargs, allow_tools_drop=True
            )
        except Exception:
            # Final failure after retries; circuit breaker state is already updated.
            raise

        # Some providers may ignore `stream=True` on retries and return a normal
        # completion object. Handle it as a non-stream response instead of failing.
        if not hasattr(stream_response, "__aiter__"):
            _log.warning(
                "api_stream_returned_non_stream_payload",
                provider=self._profile.name,
                model=self.model,
            )
            output_chars = 0
            usage = self._usage_to_dict(getattr(stream_response, "usage", None))
            text, direct_tool_calls = self._extract_choice_payload(stream_response)
            if text:
                output_chars += len(text)
                yield text
            for pos, tc in enumerate(direct_tool_calls, start=1):
                name = ""
                args_raw = ""
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    name = str(fn.get("name", "")).strip()
                    args_raw = str(fn.get("arguments", "") or "")
                else:
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        name = str(getattr(fn, "name", "") or "").strip()
                        args_raw = str(getattr(fn, "arguments", "") or "")
                if not name:
                    continue
                args = self._decode_tool_args(args_raw)
                payload = json.dumps({"name": name, "args": args}, ensure_ascii=False)
                if text or pos > 1:
                    yield "\n"
                output_chars += len(payload) + 20
                yield f"<tool_call>{payload}</tool_call>"
            self._set_last_usage(
                usage,
                estimated=False,
                input_chars=input_chars,
                output_chars=output_chars,
            )
            self._record_success()
            return

        streamed_text = False
        output_chars = 0
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tool_chunks: dict[int, dict[str, str]] = {}
        emitted_text_parts: list[str] = []
        try:
            async for chunk in stream_response:
                usage_obj = getattr(chunk, "usage", None)
                if usage_obj is not None:
                    usage = self._usage_to_dict(usage_obj)
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = choices[0].delta
                content_text = self._coerce_delta_content(getattr(delta, "content", None))
                if content_text:
                    streamed_text = True
                    output_chars += len(content_text)
                    emitted_text_parts.append(content_text)
                    yield content_text
                delta_tool_calls = getattr(delta, "tool_calls", None)
                if not delta_tool_calls:
                    continue
                for tc in delta_tool_calls:
                    idx = int(getattr(tc, "index", 0) or 0)
                    state = tool_chunks.setdefault(idx, {"name": "", "arguments": ""})
                    fn = getattr(tc, "function", None)
                    if fn is None:
                        continue
                    fn_name = getattr(fn, "name", None)
                    if fn_name:
                        state["name"] = str(fn_name)
                    fn_args = getattr(fn, "arguments", None)
                    if fn_args:
                        state["arguments"] += str(fn_args)
        except Exception as e:
            retryable = self._is_retryable_error(e)
            # If we only emitted a very short partial stream, recover via non-stream call.
            should_recover_nonstream = retryable and (
                (not streamed_text) or output_chars < 180
            )
            if should_recover_nonstream:
                _log.warning(
                    "api_stream_failed_fallback_nonstream",
                    provider=self._profile.name,
                    model=self.model,
                    error=str(e),
                )
                non_stream_kwargs = self._build_request_kwargs(
                    api_messages=api_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tools=tools if "tools" in request_kwargs else None,
                    stream=False,
                    disable_thinking=self._temperature_scope_from_kwargs(request_kwargs)[0],
                )
                response, used_kwargs = await self._create_with_retry(
                    non_stream_kwargs, allow_tools_drop=True
                )
                text, direct_tool_calls = self._extract_choice_payload(response)
                usage = self._usage_to_dict(getattr(response, "usage", None))
                if text:
                    append_text = text
                    if emitted_text_parts:
                        partial = "".join(emitted_text_parts)
                        if partial and append_text.startswith(partial):
                            append_text = append_text[len(partial):]
                        elif partial and append_text:
                            # Best-effort dedupe by longest common prefix.
                            limit = min(len(partial), len(append_text))
                            lcp = 0
                            for i in range(limit):
                                if partial[i] != append_text[i]:
                                    break
                                lcp += 1
                            if lcp >= 40:
                                append_text = append_text[lcp:]
                    if append_text:
                        output_chars += len(append_text)
                        yield append_text
                for pos, tc in enumerate(direct_tool_calls, start=1):
                    name = ""
                    args_raw = ""
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        name = str(fn.get("name", "")).strip()
                        args_raw = str(fn.get("arguments", "") or "")
                    else:
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            name = str(getattr(fn, "name", "") or "").strip()
                            args_raw = str(getattr(fn, "arguments", "") or "")
                    if not name:
                        continue
                    args = self._decode_tool_args(args_raw)
                    payload = json.dumps({"name": name, "args": args}, ensure_ascii=False)
                    if text or pos > 1:
                        yield "\n"
                    output_chars += len(payload) + 20
                    yield f"<tool_call>{payload}</tool_call>"
                self._set_last_usage(
                    usage,
                    estimated=False,
                    input_chars=input_chars,
                    output_chars=output_chars,
                )
                self._record_success()
                return
            _log.warning(
                "api_stream_partial_output",
                provider=self._profile.name,
                model=self.model,
                error=str(e),
            )
            self._record_failure()
            self._set_last_usage(
                usage if usage.get("total_tokens", 0) > 0 else None,
                estimated=True,
                input_chars=input_chars,
                output_chars=output_chars,
            )
            return

        for pos, idx in enumerate(sorted(tool_chunks.keys()), start=1):
            state = tool_chunks[idx]
            name = state.get("name", "").strip()
            if not name:
                continue
            args = self._decode_tool_args(state.get("arguments", ""))
            payload = json.dumps({"name": name, "args": args}, ensure_ascii=False)
            if streamed_text or pos > 1:
                yield "\n"
            output_chars += len(payload) + 20
            yield f"<tool_call>{payload}</tool_call>"

        self._set_last_usage(
            usage if usage.get("total_tokens", 0) > 0 else None,
            estimated=usage.get("total_tokens", 0) <= 0,
            input_chars=input_chars,
            output_chars=output_chars,
        )
        self._record_success()

    @staticmethod
    def _decode_tool_args(raw: str) -> dict:
        candidate = str(raw or "").strip()
        if not candidate:
            return {}
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            # Lenient recovery for providers that stream partial argument JSON.
            for end in range(len(candidate), 0, -1):
                if candidate[end - 1] != "}":
                    continue
                try:
                    parsed = json.loads(candidate[:end])
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    continue
        return {}

    @staticmethod
    def _coerce_delta_content(content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
                    continue
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
            return "".join(parts)
        return str(content)
