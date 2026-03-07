#!/usr/bin/env python3
"""Small grid search helper for TTS consistency parameters."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
from pathlib import Path

from liagent.config import AppConfig
from liagent.engine.tts_local import LocalTTS

TEXTS = [
    "Hello, this is a parameter tuning test.",
    "We want timbre and prosody to remain stable.",
    "Please keep a natural speaking rate and clear articulation.",
    "This is the final sentence for evaluation.",
]


async def evaluate_one(tts: LocalTTS, texts: list[str]) -> dict:
    import numpy as np

    durs = []
    for text in texts:
        wav = await tts.synthesize(text)
        durs.append(wav.size / 24000 if wav.size else 0.0)
    return {
        "mean_duration": float(np.mean(durs)),
        "std_duration": float(np.std(durs)),
    }


async def main_async(args):
    cfg = AppConfig.load()
    tcfg = cfg.tts

    temps = [float(x) for x in args.temps.split(",")]
    top_ps = [float(x) for x in args.top_ps.split(",")]
    top_ks = [int(x) for x in args.top_ks.split(",")]

    results = []
    for temp, top_p, top_k in itertools.product(temps, top_ps, top_ks):
        tts = LocalTTS(
            model_path=tcfg.local_model_path,
            language=tcfg.language,
            temperature=temp,
            ref_audio=tcfg.ref_audio,
            ref_text=tcfg.ref_text,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=tcfg.repetition_penalty,
            voice_profile=tcfg.voice_profile,
            speaker=tcfg.speaker,
        )
        m = await evaluate_one(tts, TEXTS)
        m.update({"temperature": temp, "top_p": top_p, "top_k": top_k})
        results.append(m)

    results.sort(key=lambda x: (x["std_duration"], x["mean_duration"]))
    out = {"best": results[:5], "all": results}
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temps", default="0.2,0.3,0.4")
    parser.add_argument("--top-ps", default="0.85,0.9,0.95")
    parser.add_argument("--top-ks", default="10,20,30")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
