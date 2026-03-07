"""Settings menu — minimal black/white/gray style."""

from dataclasses import asdict

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from ..config import LLMConfig, TTSConfig
from ..engine.engine_manager import EngineManager
from ..engine.provider_registry import (
    infer_api_provider,
    list_provider_presets,
    provider_presets_map,
)

console = Console()


def _infer_llm_api_provider(api_base_url: str, api_model: str) -> str:
    presets = provider_presets_map()
    provider = infer_api_provider(api_model, api_base_url)
    return provider if provider in presets else "custom"


def settings_menu(engine: EngineManager) -> bool:
    """Show settings menu. Returns True if any engine was changed."""
    changed = False

    while True:
        console.print()

        lines = Text()
        lines.append("  1", style="bold white")
        lines.append(f"  LLM   {engine.llm_status()}\n", style="dim")
        lines.append("  2", style="bold white")
        lines.append(f"  TTS   {engine.tts_status()}\n", style="dim")
        lines.append("  3", style="bold white")
        lines.append("  Voice\n", style="dim")
        lines.append("  4", style="bold white")
        tts_state = "on" if engine.config.tts_enabled else "off"
        lines.append(f"  TTS toggle  [{tts_state}]\n", style="dim")
        lines.append("  5", style="bold white")
        think_show = "on" if engine.config.show_thinking else "off"
        lines.append(f"  Show thinking  [{think_show}]\n", style="dim")
        lines.append("  6", style="bold white")
        think_enable = "on" if engine.config.enable_thinking else "off"
        lines.append(f"  Thinking chain  [{think_enable}]\n", style="dim")
        lines.append("  0", style="bold white")
        lines.append("  Back", style="dim")

        console.print(Panel(lines, border_style="dim", title="[dim]settings[/]", title_align="left", padding=(0, 1)))

        choice = Prompt.ask("[dim]>[/]", choices=["0", "1", "2", "3", "4", "5", "6"], default="0", show_choices=False)

        if choice == "0":
            break
        elif choice == "1":
            changed |= _switch_llm(engine)
        elif choice == "2":
            changed |= _switch_tts(engine)
        elif choice == "3":
            _switch_voice(engine)
        elif choice == "4":
            try:
                engine.set_tts_enabled(not engine.config.tts_enabled)
                console.print(f"  [dim]tts {'on' if engine.config.tts_enabled else 'off'}[/]")
            except Exception as e:
                console.print(f"  [dim]tts error: {e}[/]")
        elif choice == "5":
            engine.config.show_thinking = not engine.config.show_thinking
            engine.config.save()
            console.print(f"  [dim]show thinking {'on' if engine.config.show_thinking else 'off'}[/]")
        elif choice == "6":
            engine.config.enable_thinking = not engine.config.enable_thinking
            engine.config.save()
            console.print(f"  [dim]thinking chain {'on' if engine.config.enable_thinking else 'off'}[/]")

    return changed


def _switch_llm(engine: EngineManager) -> bool:
    mode = Prompt.ask("  [dim]mode[/]", choices=["local", "api"], default=engine.config.llm.backend)
    base = asdict(engine.config.llm)

    if mode == "local":
        path = Prompt.ask("  [dim]model path[/]", default=engine.config.llm.local_model_path)
        family = Prompt.ask(
            "  [dim]model family[/]",
            choices=["glm47", "qwen3-vl", "qwen3-coder", "llama", "deepseek", "openai"],
            default=(engine.config.llm.model_family or "glm47"),
        )
        base.update({"backend": "local", "local_model_path": path, "model_family": family})
        new_cfg = LLMConfig(**base)
    else:
        presets = provider_presets_map()
        preset_catalog = list_provider_presets(include_custom=False)
        provider_choices = [str(item.get("id", "")).strip() for item in preset_catalog if str(item.get("id", "")).strip()]
        if not provider_choices:
            provider_choices = ["openai"]
        provider_choices = list(dict.fromkeys([*provider_choices, "custom"]))
        default_provider = _infer_llm_api_provider(
            engine.config.llm.api_base_url,
            engine.config.llm.api_model,
        )
        if default_provider not in provider_choices:
            default_provider = "custom"
        provider = Prompt.ask(
            "  [dim]api provider preset[/]",
            choices=provider_choices,
            default=default_provider,
        )
        preset = presets.get(provider, {})
        default_base_url = (
            str(engine.config.llm.api_base_url or "").strip()
            or preset.get("api_base_url", "https://api.openai.com/v1")
        )
        default_model = (
            str(engine.config.llm.api_model or "").strip()
            or preset.get("api_model", "gpt-4o")
        )
        default_family = (
            str(engine.config.llm.model_family or "").strip()
            or preset.get("model_family", "openai")
        )
        default_tool_protocol = (
            str(engine.config.llm.tool_protocol or "").strip()
            or preset.get("tool_protocol", "openai_function")
        )

        base_url = Prompt.ask("  [dim]api base url[/]", default=default_base_url)
        api_key = Prompt.ask("  [dim]api key[/]", default=engine.config.llm.api_key or "", password=True)
        model = Prompt.ask("  [dim]model[/]", default=default_model)
        family = Prompt.ask(
            "  [dim]model family[/]",
            choices=["openai", "deepseek", "llama", "qwen3-vl", "gemini", "claude", "glm47"],
            default=default_family,
        )
        cache_mode = Prompt.ask(
            "  [dim]api prompt cache mode[/]",
            choices=["implicit", "explicit", "off"],
            default=(engine.config.llm.api_cache_mode or "implicit"),
        )
        cache_ttl_default = str(getattr(engine.config.llm, "api_cache_ttl_sec", 600) or 600)
        cache_ttl_raw = Prompt.ask("  [dim]api cache ttl sec[/]", default=cache_ttl_default)
        try:
            cache_ttl = max(60, int(cache_ttl_raw))
        except (TypeError, ValueError):
            cache_ttl = 600
        base.update(
            {
                "backend": "api",
                "api_base_url": base_url,
                "api_key": api_key,
                "api_model": model,
                "model_family": family,
                "tool_protocol": default_tool_protocol,
                "api_cache_mode": cache_mode,
                "api_cache_ttl_sec": cache_ttl,
            }
        )
        new_cfg = LLMConfig(**base)

    console.print("  [dim]switching...[/]")
    try:
        engine.switch_llm(new_cfg)
        console.print(f"  [white]llm: {engine.llm_status()}[/]")
        return True
    except Exception as e:
        console.print(f"  [dim]error: {e}[/]")
        return False


def _switch_tts(engine: EngineManager) -> bool:
    mode = Prompt.ask("  [dim]mode[/]", choices=["local", "api"], default=engine.config.tts.backend)
    base = asdict(engine.config.tts)

    if mode == "local":
        from ..engine.tts_qwen3 import PRESET_SPEAKERS, DEFAULT_SPEAKER

        path = Prompt.ask("  [dim]model path[/]", default=engine.config.tts.local_model_path)
        lang = Prompt.ask("  [dim]language (zh/en/en-gb/ja/fr/es/it/pt/hi)[/]", default=engine.config.tts.language or "zh")
        current_speaker = engine.config.tts.speaker_name or DEFAULT_SPEAKER
        speaker_list = ", ".join(PRESET_SPEAKERS)
        speaker = Prompt.ask(f"  [dim]speaker ({speaker_list})[/]", default=current_speaker)
        speed_str = Prompt.ask("  [dim]speed (0.5-2.0)[/]", default=str(engine.config.tts.speed or 1.0))
        chunk_strategy = Prompt.ask(
            "  [dim]chunk strategy[/]",
            choices=["oneshot", "smart_chunk"],
            default=engine.config.tts.chunk_strategy,
        )
        max_chars_str = Prompt.ask("  [dim]max chunk chars[/]", default=str(engine.config.tts.max_chunk_chars))

        try:
            speed = max(0.5, min(2.0, float(speed_str)))
        except ValueError:
            speed = float(engine.config.tts.speed or 1.0)
        try:
            max_chunk_chars = max(80, int(max_chars_str))
        except ValueError:
            max_chunk_chars = engine.config.tts.max_chunk_chars

        speaker = speaker.strip().lower()
        if speaker not in PRESET_SPEAKERS:
            console.print(f"  [dim]unknown speaker '{speaker}', using '{DEFAULT_SPEAKER}'[/]")
            speaker = DEFAULT_SPEAKER

        base.update(
            {
                "backend": "local",
                "local_model_path": path,
                "language": (lang or "zh").strip().lower(),
                "speaker_name": speaker,
                "speed": speed,
                "chunk_strategy": chunk_strategy,
                "max_chunk_chars": max_chunk_chars,
            }
        )
        new_cfg = TTSConfig(**base)
    else:
        base_url = Prompt.ask("  [dim]api base url[/]", default=engine.config.tts.api_base_url or "https://api.openai.com/v1")
        api_key = Prompt.ask("  [dim]api key[/]", default=engine.config.tts.api_key or "", password=True)
        model = Prompt.ask("  [dim]model[/]", default=engine.config.tts.api_model or "tts-1")
        voice = Prompt.ask("  [dim]voice[/]", default=engine.config.tts.api_voice or "alloy")
        base.update(
            {
                "backend": "api",
                "api_base_url": base_url,
                "api_key": api_key,
                "api_model": model,
                "api_voice": voice,
            }
        )
        new_cfg = TTSConfig(**base)

    console.print("  [dim]switching...[/]")
    try:
        engine.switch_tts(new_cfg)
        console.print(f"  [white]tts: {engine.tts_status()}[/]")
        return True
    except Exception as e:
        console.print(f"  [dim]error: {e}[/]")
        return False


def _switch_voice(engine: EngineManager):
    """Lightweight voice switch — changes speaker without reloading model."""
    if not engine.tts:
        console.print("  [dim]tts disabled[/]")
        return

    if engine.config.tts.backend == "local":
        from ..engine.tts_qwen3 import PRESET_SPEAKERS, DEFAULT_SPEAKER

        current_speaker = engine.config.tts.speaker_name or DEFAULT_SPEAKER
        speaker_list = ", ".join(PRESET_SPEAKERS)
        speaker = Prompt.ask(f"  [dim]speaker ({speaker_list})[/]", default=current_speaker)
        speed_str = Prompt.ask("  [dim]speed (0.5-2.0)[/]", default=str(engine.config.tts.speed or 1.0))
        try:
            speed = max(0.5, min(2.0, float(speed_str)))
        except ValueError:
            speed = float(engine.config.tts.speed or 1.0)

        speaker = speaker.strip().lower()
        if speaker not in PRESET_SPEAKERS:
            console.print(f"  [dim]unknown speaker '{speaker}', using '{DEFAULT_SPEAKER}'[/]")
            speaker = DEFAULT_SPEAKER

        # Lightweight switch — no model reload
        if hasattr(engine.tts, "set_speaker"):
            engine.tts.set_speaker(speaker)
        if hasattr(engine.tts, "speed"):
            engine.tts.speed = speed

        engine.config.tts.speaker_name = speaker
        engine.config.tts.speed = speed
        console.print(f"  [white]speaker: {speaker} @ {speed:.2f}x[/]")
    else:
        voice = Prompt.ask("  [dim]voice[/]", default=engine.config.tts.api_voice)
        engine.config.tts.api_voice = voice
        console.print(f"  [white]voice: {voice}[/]")

    engine.config.save()
