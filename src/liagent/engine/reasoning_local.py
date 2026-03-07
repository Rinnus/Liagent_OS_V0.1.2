"""Text-only reasoning/chat backend using mlx-lm (GLM-4.7-Flash-4bit).

Lazy-loaded on first call. Used as the primary text chat model, skill generation,
experience analysis, and complex reasoning tasks. Supports native tool calling
via tokenizer's tools= parameter and optional <think> reasoning via enable_thinking.
"""

import asyncio
import gc
import threading
from collections.abc import AsyncIterator

from .base import LLMBackend
from .runtime import run_mlx_serialized, _RUNTIME_GUARD, _MLX_EXECUTOR, _MLX_LOCK

_SENTINEL = object()


class LocalReasoning(LLMBackend):
    """Text-only reasoning/chat model via mlx-lm.

    Lazy-loaded: model weights are loaded on first generate() call.
    Designed for GLM-4.7-Flash (MoE, ~3B active params, 4-bit).
    Supports native tool definitions via tools= parameter and
    optional enable_thinking for <think> reasoning chains.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def _ensure_loaded(self):
        if self.model is not None:
            return
        from mlx_lm import load

        self.model, self.tokenizer = load(self.model_path)

    def _prepare_prompt(
        self, messages: list[dict], tools: list[dict] | None = None, enable_thinking: bool = True
    ) -> str:
        """Apply chat template and return the prompt string.

        Args:
            enable_thinking: Pass to tokenizer for models that support <think> mode
                (e.g. GLM-4.7-Flash). Falls back gracefully if tokenizer doesn't
                accept this parameter (e.g. Qwen3-Coder).
        """
        self._ensure_loaded()
        template_kwargs = dict(
            tokenize=False, add_generation_prompt=True
        )
        if tools:
            template_kwargs["tools"] = tools
        template_kwargs["enable_thinking"] = enable_thinking
        try:
            return self.tokenizer.apply_chat_template(
                messages, **template_kwargs
            )
        except TypeError:
            # Tokenizer doesn't support enable_thinking (e.g. Qwen3-Coder) — fallback
            template_kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(
                messages, **template_kwargs
            )

    async def generate(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.4,
        *,
        tools: list[dict] | None = None,
        enable_thinking: bool = True,
    ) -> str:
        """Generate text-only reasoning result. Returns complete string (non-streaming).

        The `images` parameter is accepted for interface compatibility but ignored.
        When `tools` is provided, it is passed to apply_chat_template() for
        native tool calling support.
        """
        prompt = self._prepare_prompt(messages, tools, enable_thinking=enable_thinking)
        from mlx_lm import generate as lm_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        sampler = make_sampler(temp=max(0.05, float(temperature)))
        logits_processors = make_logits_processors(
            repetition_penalty=1.2, repetition_context_size=40
        )

        def _do():
            return lm_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max(1, int(max_tokens)),
                sampler=sampler,
                logits_processors=logits_processors,
                verbose=False,
            )

        return await run_mlx_serialized(_do, timeout_sec=120)

    async def stream_generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.4,
        tools: list[dict] | None = None,
        enable_thinking: bool = True,
    ) -> AsyncIterator[str]:
        """Stream tokens via Queue-bridge.

        MLX thread pushes tokens to asyncio.Queue, async consumer yields them.
        """
        prompt = self._prepare_prompt(messages, tools, enable_thinking=enable_thinking)
        from mlx_lm import stream_generate as lm_stream_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        sampler = make_sampler(temp=max(0.05, float(temperature)))
        logits_processors = make_logits_processors(
            repetition_penalty=1.2, repetition_context_size=40
        )
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | object] = asyncio.Queue()

        with _RUNTIME_GUARD:
            executor = _MLX_EXECUTOR
            lock = _MLX_LOCK

        def _stream_worker():
            with lock:
                for result in lm_stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max(1, int(max_tokens)),
                    sampler=sampler,
                    logits_processors=logits_processors,
                ):
                    token_text = result.text if hasattr(result, "text") else str(result)
                    loop.call_soon_threadsafe(queue.put_nowait, token_text)
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        fut = loop.run_in_executor(executor, _stream_worker)

        try:
            while True:
                item = await asyncio.wait_for(queue.get(), timeout=120)
                if item is _SENTINEL:
                    break
                yield item
        except asyncio.TimeoutError:
            fut.cancel()
            # Drain remaining items to prevent memory leaks
            try:
                while not queue.empty():
                    queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            raise TimeoutError("MLX stream_generate timeout (120s)")
        finally:
            # Ensure the future completes
            if not fut.done():
                try:
                    await asyncio.wait_for(fut, timeout=5)
                except Exception:
                    pass

    def is_loaded(self) -> bool:
        return self.model is not None

    def unload(self):
        """Release model resources."""
        self.model = None
        self.tokenizer = None
        gc.collect()
