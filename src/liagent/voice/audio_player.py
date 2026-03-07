"""Non-blocking audio player with interrupt support."""

import threading

import numpy as np


class AudioPlayer:
    """Play audio in a background thread. Supports interruption."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._playing = False
        self._thread: threading.Thread | None = None

    @property
    def is_playing(self) -> bool:
        return self._playing

    def play(self, audio: np.ndarray):
        """Start playing audio in background. Stops any current playback."""
        self.stop()
        if audio.size == 0:
            return
        self._playing = True
        self._thread = threading.Thread(target=self._play, args=(audio,), daemon=True)
        self._thread.start()

    def _play(self, audio: np.ndarray):
        try:
            import sounddevice as sd

            sd.play(audio, samplerate=self.sample_rate)
            sd.wait()
        except Exception:
            pass
        finally:
            self._playing = False

    def stop(self):
        """Interrupt current playback immediately."""
        if self._playing:
            try:
                import sounddevice as sd

                sd.stop()
            except Exception:
                pass
            self._playing = False

    def play_sync(self, audio: np.ndarray):
        """Play audio blocking in the current thread."""
        if audio.size == 0:
            return
        try:
            import sounddevice as sd

            self._playing = True
            sd.play(audio, samplerate=self.sample_rate)
            sd.wait()
        except Exception:
            pass
        finally:
            self._playing = False
