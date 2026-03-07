#!/usr/bin/env python3
"""Run a lightweight TTS consistency benchmark on 20 fixed prompts."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from liagent.config import AppConfig
from liagent.engine.tts_local import LocalTTS

SAMPLE_RATE = 24000

BENCH_TEXTS = [
    "Hello, today we start a voice consistency benchmark.",
    "Please read this sentence in a natural and steady tone.",
    "If you can hear me, continue to the next sentence.",
    "This system should keep the same timbre across multi-turn conversations.",
    "Intonation can vary, but avoid obvious speaker switching artifacts.",
    "We will measure pitch and speech-rate stability.",
    "Please keep sentence boundaries natural and pauses not too long.",
    "The optimization target is stable and predictable output.",
    "Clear expression is critical in complex tasks.",
    "Now continue with the next standard sentence.",
    "Repeated generation of the same text should remain consistent.",
    "Long responses should avoid voice-style drift as much as possible.",
    "If segmentation is needed, use larger chunk sizes.",
    "Please keep a normal speaking rate, neither too fast nor too slow.",
    "We will compute feature metrics for each audio sample.",
    "The final output will be a reproducible benchmark report.",
    "This supports later automated parameter tuning.",
    "Thanks for your cooperation; we are entering the final lines.",
    "Please keep the current pronunciation style.",
    "Benchmark finished. Preparing output.",
]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _estimate_pitch_hz(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    if audio.size < sample_rate // 4:
        return 0.0
    x = audio.astype(np.float64)
    x = x - x.mean()
    energy = np.sqrt(np.mean(x**2)) + 1e-9
    if energy < 1e-4:
        return 0.0
    min_hz, max_hz = 70, 400
    min_lag = sample_rate // max_hz
    max_lag = sample_rate // min_hz
    corr = np.correlate(x, x, mode="full")[len(x) - 1 :]
    window = corr[min_lag:max_lag]
    if window.size == 0:
        return 0.0
    lag = int(np.argmax(window)) + min_lag
    if lag <= 0:
        return 0.0
    return float(sample_rate / lag)


async def run_benchmark() -> dict:
    cfg = AppConfig.load()
    tcfg = cfg.tts
    tts = LocalTTS(
        model_path=tcfg.local_model_path,
        language=tcfg.language,
        temperature=tcfg.temperature,
        ref_audio=tcfg.ref_audio,
        ref_text=tcfg.ref_text,
        top_k=tcfg.top_k,
        top_p=tcfg.top_p,
        repetition_penalty=tcfg.repetition_penalty,
        voice_profile=tcfg.voice_profile,
        speaker=tcfg.speaker,
    )

    embeddings: list[np.ndarray] = []
    pitches: list[float] = []
    rates: list[float] = []
    durations: list[float] = []
    rows: list[dict] = []

    model = tts.model
    has_spk = hasattr(model, "extract_speaker_embedding")
    mx = None
    if has_spk:
        import mlx.core as mlx_core

        mx = mlx_core

    for text in BENCH_TEXTS:
        wav = await tts.synthesize(text)
        duration = wav.size / SAMPLE_RATE if wav.size else 0.0
        durations.append(duration)
        pitch = _estimate_pitch_hz(wav)
        pitches.append(pitch)
        rates.append((len(text) / duration) if duration > 0 else 0.0)

        emb_np = None
        if has_spk and wav.size > 0:
            try:
                emb = model.extract_speaker_embedding(mx.array(wav), sr=SAMPLE_RATE)
                emb_np = np.array(emb).reshape(-1).astype(np.float32)
            except Exception:
                emb_np = None
        if emb_np is not None:
            embeddings.append(emb_np)

        rows.append(
            {
                "text": text,
                "duration_sec": round(duration, 3),
                "pitch_hz": round(pitch, 2),
            }
        )

    speaker_cos = []
    if len(embeddings) >= 2:
        anchor = embeddings[0]
        for emb in embeddings[1:]:
            speaker_cos.append(_cosine(anchor, emb))

    report = {
        "num_texts": len(BENCH_TEXTS),
        "speaker_embedding_available": bool(has_spk and embeddings),
        "speaker_cosine_mean": round(float(np.mean(speaker_cos)) if speaker_cos else 0.0, 4),
        "speaker_cosine_std": round(float(np.std(speaker_cos)) if speaker_cos else 0.0, 4),
        "pitch_mean_hz": round(float(np.mean(pitches)) if pitches else 0.0, 2),
        "pitch_std_hz": round(float(np.std(pitches)) if pitches else 0.0, 2),
        "speech_rate_mean_char_per_sec": round(float(np.mean(rates)) if rates else 0.0, 3),
        "speech_rate_std_char_per_sec": round(float(np.std(rates)) if rates else 0.0, 3),
        "duration_mean_sec": round(float(np.mean(durations)) if durations else 0.0, 3),
        "items": rows,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="", help="Optional JSON output path")
    args = parser.parse_args()

    report = asyncio.run(run_benchmark())
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
