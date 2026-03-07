"""Local VLM engine using mlx-vlm."""

import os
from collections.abc import AsyncIterator

from .base import LLMBackend
from .runtime import run_mlx_serialized


def _patch_transformers_video_processor():
    """Workaround for transformers 5.0.0rc3 bug where
    VIDEO_PROCESSOR_MAPPING_NAMES has None values → TypeError.
    Fix: replace None with empty tuple so 'in' checks work."""
    try:
        import transformers.models.auto.video_processing_auto as vpa

        for key in list(vpa.VIDEO_PROCESSOR_MAPPING_NAMES.keys()):
            if vpa.VIDEO_PROCESSOR_MAPPING_NAMES[key] is None:
                vpa.VIDEO_PROCESSOR_MAPPING_NAMES[key] = ()
    except Exception:
        pass


_patch_transformers_video_processor()


def _safe_temperature(temperature: float | None) -> float:
    """mlx-vlm expects a positive float temperature; never return None."""
    if temperature is None:
        return 0.1
    try:
        t = float(temperature)
    except (TypeError, ValueError):
        return 0.1
    if t <= 0:
        return 0.1
    return t


class LocalVLM(LLMBackend):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.config = None
        self.timeout_sec = max(
            5.0, float(os.environ.get("LIAGENT_LLM_TIMEOUT_SEC", "60"))
        )
        self._loaded = False
        eager_load = (
            os.environ.get("LIAGENT_LLM_EAGER_LOAD", "").strip().lower()
            in {"1", "true", "yes"}
        )
        if eager_load:
            self._ensure_loaded()

    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None and self.processor is not None

    def unload(self):
        if self.model is None and self.processor is None:
            self._loaded = False
            return
        del self.model, self.processor
        self.model = self.processor = None
        self._loaded = False

    def _ensure_loaded(self):
        if self.model is not None and self.processor is not None:
            self._loaded = True
            return
        import logging
        import warnings
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logging.disable(logging.WARNING)
            self.model, self.processor = load(self.model_path)
            logging.disable(logging.NOTSET)
        self.config = load_config(self.model_path)
        self._loaded = True

    async def generate(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        from mlx_vlm import generate as vlm_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        self._ensure_loaded()

        if images:
            # Image call: use mlx-vlm template with system + last user only,
            # so vision tokens align with the actual image features
            img_messages = [messages[0], messages[-1]]  # system + latest user
            formatted_prompt = apply_chat_template(
                self.processor, self.config,
                img_messages,
                num_images=len(images),
            )
        else:
            # Text-only: use tokenizer's native template for full multi-turn
            formatted_prompt = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Run synchronous mlx-vlm generate in thread to avoid blocking
        safe_temp = _safe_temperature(temperature)

        def _do_generate():
            import mlx.core as mx
            try:
                return vlm_generate(
                    self.model,
                    self.processor,
                    formatted_prompt,
                    image=images if images else None,
                    max_tokens=max_tokens,
                    temperature=safe_temp,
                    verbose=False,
                )
            except UnicodeDecodeError:
                # mlx-vlm detokenizer can fail on partial multi-byte sequences;
                # retry with fewer tokens as a fallback
                try:
                    return vlm_generate(
                        self.model,
                        self.processor,
                        formatted_prompt,
                        image=images if images else None,
                        max_tokens=max(256, max_tokens // 2),
                        temperature=safe_temp,
                        verbose=False,
                    )
                except UnicodeDecodeError:
                    # Return a sentinel so caller can yield an error message
                    return None
            finally:
                # Keep all MLX cache operations inside serialized runtime.
                mx.clear_cache()

        output = await run_mlx_serialized(_do_generate, timeout_sec=self.timeout_sec)

        if output is None:
            yield "(encoding error during generation; please retry)"
        else:
            # mlx-vlm generate returns a GenerationResult with .text attribute
            text = output.text if hasattr(output, "text") else str(output)
            yield text
