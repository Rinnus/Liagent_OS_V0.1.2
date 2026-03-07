"""Microphone recorder with VAD-based auto-stop."""

import asyncio
import threading

import numpy as np

from .vad import VAD, SAMPLE_RATE, CHUNK_SAMPLES


class MicRecorder:
    """Records from microphone, auto-detects speech boundaries via VAD."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.vad = VAD(sample_rate=sample_rate)
        self._recording = False

    async def listen_once(
        self, on_speaking: callable = None, on_silence: callable = None
    ) -> np.ndarray | None:
        """Listen for one utterance. Returns audio when speech ends, or None.

        on_speaking(): called when speech detected (for UI feedback)
        on_silence():  called when waiting silently
        """
        import sounddevice as sd

        self.vad.reset()
        self._recording = True
        chunks: list[np.ndarray] = []
        speech_started = False
        result_audio = None

        event = asyncio.Event()

        def callback(indata, frames, time_info, status):
            nonlocal speech_started, result_audio

            chunk = indata[:, 0].copy()
            state = self.vad.process(chunk)

            if state == "speaking":
                chunks.append(chunk)
                if not speech_started:
                    speech_started = True
                    if on_speaking:
                        on_speaking()
            elif state == "end":
                # Speech ended — return collected audio
                if chunks:
                    result_audio = np.concatenate(chunks)
                event.set()
            elif state == "silent" and not speech_started:
                # Still waiting for speech
                pass

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=callback,
        )

        try:
            stream.start()
            # Wait for speech to end, or timeout after 30s
            try:
                await asyncio.wait_for(event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                pass
            stream.stop()
            stream.close()
        except Exception:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        self._recording = False
        return result_audio

    def stop(self):
        self._recording = False
