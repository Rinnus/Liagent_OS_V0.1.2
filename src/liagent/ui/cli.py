"""Terminal UI — minimal, black/white/gray."""

import asyncio
import json
import sys
import time

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..agent.brain import AgentBrain
from ..engine.engine_manager import EngineManager
from ..engine.tts_utils import build_tts_chunks
from ..voice.audio_player import AudioPlayer
from ..voice.voice_chat import VoiceChat
from .settings import settings_menu

console = Console(highlight=False)
player = AudioPlayer()

# ─── UI primitives ──────────────────────────────────────────────────────────

LOGO = """
  ╷   •  ╭─╮  ╭─╮  ╭─╮  ╭╮╷ ╭╮╷
  │   │  ├─┤  │ ╵  ├─   │╵│ │ │
  ╰── ╰  ╵ ╵  ╰─╯  ╰─╯  ╵ ╰ ╵ ╵
"""


def _bar(label: str, value: str) -> Text:
    t = Text()
    t.append(f"  {label} ", style="dim")
    t.append(value, style="white")
    return t


def _show_banner(engine: EngineManager, voice_mode: bool = False):
    console.print()
    console.print(LOGO, style="bold white", highlight=False)

    status = Text()
    status.append("  llm ", style="dim")
    status.append(engine.llm_status(), style="white")
    status.append("    tts ", style="dim")
    status.append(engine.tts_status(), style="white")
    if voice_mode:
        status.append("    mode ", style="dim")
        status.append("voice", style="bold white")
    console.print(status)

    cmds = Text()
    cmds.append("  /voice", style="white")
    cmds.append("  /settings", style="dim")
    cmds.append("  /tts", style="dim")
    cmds.append("  /clear", style="dim")
    cmds.append("  /quit", style="dim")
    console.print(cmds)
    console.print()


def _print_divider():
    console.print("  " + "─" * (console.width - 4), style="dim")


# ─── Voice mode ─────────────────────────────────────────────────────────────

async def _run_voice_mode(engine: EngineManager, brain: AgentBrain):
    """Continuous voice conversation loop with real-time feedback."""
    console.print()
    console.print("  [white]voice mode[/]  [dim]speak naturally, ctrl+c to exit[/]")
    console.print()

    voice_chat = VoiceChat(engine, brain)

    def ui_callback(event, data=None):
        if event == "listening":
            console.print("  [dim]listening...[/]", end="\r")
        elif event == "hearing":
            console.print("  [white]hearing...[/]   ", end="\r")
        elif event == "status":
            console.print(f"  [dim]{data}[/]        ", end="\r")
        elif event == "thinking":
            console.print(f"\n  [dim]you:[/] {data}")
            console.print("  [dim]thinking...[/]", end="\r")
        elif event == "answer":
            console.print(f"\r  [white]{data}[/]")
            console.print()
        elif event == "speaking":
            pass  # answer already printed

    try:
        await voice_chat.run(ui_callback=ui_callback)
    except (KeyboardInterrupt, asyncio.CancelledError):
        voice_chat.stop()
        console.print("\n  [dim]voice mode ended[/]")


# ─── Text chat handling ─────────────────────────────────────────────────────

async def _handle_input(user_input: str, brain: AgentBrain, engine: EngineManager,
                        orchestrator=None):
    """Process text input through agent, render streaming output."""
    full_answer = ""
    printed_tool = False

    if orchestrator is not None:
        event_source = orchestrator.dispatch(user_input)
    else:
        event_source = brain.run(user_input)

    async for event in event_source:
        # Convert AgentEvent to legacy tuple if needed
        if hasattr(event, "to_legacy_tuple"):
            event = event.to_legacy_tuple()
        etype = event[0]

        if etype == "token":
            token = event[1]
            full_answer += token
            # Print directly for streaming feel
            if not printed_tool:
                print(token, end="", flush=True)

        elif etype == "tool_start":
            name, args = event[1], event[2]
            if full_answer:
                print()  # end partial line
            arg_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else ""
            console.print(f"  [dim]{name}({arg_str})[/]")
            full_answer = ""
            printed_tool = True

        elif etype == "tool_result":
            name, result = event[1], event[2]
            preview = result[:120].replace("\n", " ")
            if len(result) > 120:
                preview += "..."
            console.print(f"  [dim]{preview}[/]")
            console.print()
            printed_tool = False

        elif etype == "llm_usage":
            payload = event[1] if len(event) > 1 else {}
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {"raw": payload}
            provider = payload.get("provider", "-")
            pt = int(payload.get("prompt_tokens", 0) or 0)
            ct = int(payload.get("completion_tokens", 0) or 0)
            tt = int(payload.get("total_tokens", 0) or 0)
            cpt = int(payload.get("cached_prompt_tokens", 0) or 0)
            cost = float(payload.get("estimated_cost_usd", 0.0) or 0.0)
            extra = " (estimated)" if payload.get("estimated") else ""
            budget = ""
            if "turn_budget" in payload:
                budget = (
                    f" | turn {int(payload.get('turn_used', 0) or 0)}"
                    f"/{int(payload.get('turn_budget', 0) or 0)}"
                )
            cache = ""
            if cpt > 0:
                cache = f" | cached={cpt}"
            cost_txt = f" | cost=${cost:.5f}" if cost > 0 else ""
            console.print(
                f"  [dim]llm usage: {provider} p/c/t={pt}/{ct}/{tt}{cache}{cost_txt}{extra}{budget}[/]"
            )

        elif etype == "policy_blocked":
            console.print(f"  [dim]policy blocked: {event[1]} ({event[2]})[/]")

        elif etype == "policy_review":
            console.print(f"  [dim]policy review {event[1]} -> {event[2]}[/]")

        elif etype == "confirmation_required":
            token, tool, reason = event[1], event[2], event[3]
            console.print(f"  [dim]confirmation required: {tool} ({reason})[/]")
            if len(event) > 4 and event[4]:
                console.print(f"  [dim]{event[4]}[/]")
            console.print(
                f"  [white]/confirm {token}[/]  [dim](use --force for second-stage confirm)[/]  [dim]or[/]  [white]/reject {token}[/]"
            )

        elif etype == "done":
            if full_answer and not printed_tool:
                pass  # already streamed
            print()

            # TTS
            if engine.config.tts_enabled and engine.tts and full_answer:
                answer = event[1]
                try:
                    chunks = build_tts_chunks(
                        answer,
                        chunk_strategy=engine.config.tts.chunk_strategy,
                        max_chunk_chars=engine.config.tts.max_chunk_chars,
                    )
                    for idx, chunk in enumerate(chunks):
                        audio = await engine.tts.synthesize(chunk)
                        if audio.size > 0:
                            if idx == 0:
                                player.play(audio)  # non-blocking
                            else:
                                player.play_sync(audio)
                except Exception as e:
                    console.print(f"  [dim]tts error: {e}[/]")

        elif etype == "error":
            print()
            console.print(f"  [dim]error: {event[1]}[/]")


# ─── Main loop ──────────────────────────────────────────────────────────────

async def run_cli(engine: EngineManager):
    """Main CLI event loop."""
    brain = AgentBrain(engine)

    # Create orchestrator for multi-agent routing
    try:
        from ..orchestrator.orchestrator import Orchestrator
        orchestrator = Orchestrator(engine=engine, brain=brain)
    except Exception:
        orchestrator = None

    _show_banner(engine)

    try:
        while True:
            try:
                user_input = console.input("  [bold white]>[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n")
                break

            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd in ("/quit", "/exit", "/q"):
                break
            elif cmd == "/settings":
                settings_menu(engine)
                _show_banner(engine)
                continue
            elif cmd == "/clear":
                await brain.clear_memory()
                console.print("  [dim]cleared[/]")
                continue
            elif cmd == "/tts":
                try:
                    engine.set_tts_enabled(not engine.config.tts_enabled)
                    console.print(f"  [dim]tts {'on' if engine.config.tts_enabled else 'off'}[/]")
                except Exception as e:
                    console.print(f"  [dim]tts error: {e}[/]")
                continue
            elif cmd == "/voice":
                try:
                    await _run_voice_mode(engine, brain)
                except Exception as e:
                    console.print(f"  [dim]voice error: {e}[/]")
                _show_banner(engine)
                continue
            # Text chat
            console.print()
            await _handle_input(user_input, brain, engine, orchestrator=orchestrator)
            console.print()
    finally:
        try:
            await brain.shutdown()
        except Exception:
            pass
        if orchestrator is not None:
            try:
                await orchestrator.shutdown()
            except Exception:
                pass
        # Project knowledge events to vault on shutdown
        try:
            from ..knowledge.projector import project_pending_events
            project_pending_events()
        except Exception:
            pass
