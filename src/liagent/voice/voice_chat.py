"""Voice chat orchestrator — continuous listen → think → speak loop."""

import asyncio

import numpy as np

from ..agent.brain import AgentBrain
from ..engine.engine_manager import EngineManager
from ..engine.tts_utils import build_tts_chunks
from .audio_player import AudioPlayer
from .recorder import MicRecorder


class VoiceChat:
    """Manages the real-time voice conversation loop."""

    def __init__(self, engine: EngineManager, brain: AgentBrain):
        self.engine = engine
        self.brain = brain
        self.recorder = MicRecorder()
        self.player = AudioPlayer()
        self.active = False

    async def run(self, ui_callback=None):
        """Run continuous voice loop until stopped.

        ui_callback(event_type, data):
            ("status", str)       — status text to display
            ("listening", None)   — waiting for speech
            ("hearing", None)     — speech detected
            ("thinking", text)    — STT result, processing
            ("speaking", text)    — TTS playing
            ("answer", text)      — full answer text
        """
        self.active = True
        # Free 30B Coder memory — voice mode only uses 4B VLM
        self.engine.unload_reasoning_llm()

        def _notify(event, data=None):
            if ui_callback:
                ui_callback(event, data)

        while self.active:
            # 1. Listen
            _notify("listening")

            def on_speak():
                self.player.stop()  # Interrupt any ongoing playback
                _notify("hearing")

            audio = await self.recorder.listen_once(on_speaking=on_speak)

            if audio is None or not self.active:
                continue

            # 2. Transcribe
            _notify("status", "transcribing...")
            text = await self.engine.stt.transcribe(audio)

            if not text or not text.strip():
                continue

            _notify("thinking", text)

            # 3. Generate response
            full_answer = ""
            async for event in self.brain.run(text, low_latency=True):
                if not self.active:
                    break
                etype = event[0]
                if etype == "token":
                    full_answer += event[1]
                elif etype == "bridge_tts":
                    # Immediate bridge phrase while 30B processes
                    bridge_text = event[1]
                    if (
                        self.engine.config.tts_enabled
                        and self.engine.tts
                        and bridge_text
                    ):
                        try:
                            bridge_audio = await self.engine.tts.synthesize(bridge_text)
                            if bridge_audio.size > 0:
                                self.player.play_sync(bridge_audio)
                        except Exception:
                            pass
                elif etype == "done":
                    full_answer = event[1]

            if not self.active:
                break

            _notify("answer", full_answer)

            # 4. TTS — streaming if available, fallback to chunked synthesis
            if (
                self.engine.config.tts_enabled
                and self.engine.tts
                and full_answer
            ):
                _notify("speaking", full_answer)
                if hasattr(self.engine.tts, "synthesize_stream"):
                    # Qwen3-TTS: stream full text for consistent prosody
                    try:
                        async for audio_out in self.engine.tts.synthesize_stream(full_answer):
                            if not self.active:
                                break
                            if audio_out.size > 0:
                                self.player.play_sync(audio_out)
                    except Exception:
                        pass
                else:
                    # Fallback: chunk-based synthesis
                    chunks = build_tts_chunks(
                        full_answer,
                        chunk_strategy=self.engine.config.tts.chunk_strategy,
                        max_chunk_chars=self.engine.config.tts.max_chunk_chars,
                    )
                    for chunk in chunks:
                        if not self.active:
                            break
                        if len(chunk.strip()) < 2:
                            continue
                        try:
                            audio_out = await self.engine.tts.synthesize(chunk)
                            if audio_out.size > 0:
                                self.player.play_sync(audio_out)
                        except Exception:
                            break

    def stop(self):
        self.active = False
        self.player.stop()
        self.recorder.stop()
