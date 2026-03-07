"""Energy-based Voice Activity Detection — zero dependencies, zero latency."""

import numpy as np

# Tuned for typical MacBook mic
ENERGY_THRESHOLD = 0.015      # RMS threshold to consider "speech"
SILENCE_DURATION = 1.2        # seconds of silence to end utterance
MIN_SPEECH_DURATION = 0.3     # minimum speech to be considered valid
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600          # 100ms chunks


class VAD:
    """Lightweight energy-based voice activity detector."""

    def __init__(
        self,
        threshold: float = ENERGY_THRESHOLD,
        silence_dur: float = SILENCE_DURATION,
        min_speech_dur: float = MIN_SPEECH_DURATION,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.threshold = threshold
        self.silence_samples = int(silence_dur * sample_rate)
        self.min_speech_samples = int(min_speech_dur * sample_rate)
        self.reset()

    def reset(self):
        self.is_speaking = False
        self._silent_count = 0
        self._speech_count = 0

    def process(self, chunk: np.ndarray) -> str:
        """Feed a chunk of audio. Returns: 'silent', 'speaking', or 'end'.

        'end' is returned exactly once when speech stops after silence timeout.
        """
        rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

        if rms >= self.threshold:
            self._silent_count = 0
            self._speech_count += len(chunk)
            if not self.is_speaking:
                self.is_speaking = True
            return "speaking"
        else:
            if self.is_speaking:
                self._silent_count += len(chunk)
                if self._silent_count >= self.silence_samples:
                    valid = self._speech_count >= self.min_speech_samples
                    self.reset()
                    return "end" if valid else "silent"
                return "speaking"  # still within silence tolerance
            return "silent"
