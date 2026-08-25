"""Generate a synthetic, voice-like WAV so anyone can smoke-test the service
without committing a real person's voice to the repo.

    python scripts/make_sample.py            # -> sample.wav
    python scripts/make_sample.py out.wav 3  # custom path / seconds

It is a ~120 Hz tone with a few harmonics and light noise: enough to pass the
quality gate and read as a low-pitched (male-ish) voice under the mock backend.
For meaningful real predictions, use an actual voice clip (see README).
"""
from __future__ import annotations

import sys
import wave

import numpy as np


def make(path: str = "sample.wav", seconds: float = 2.0, sr: int = 16_000) -> None:
    f0 = 120.0  # fundamental frequency (low-pitched, reads male-ish)
    # Alternate 0.25 s voiced segments with 0.1 s gaps, the way real speech has
    # pauses between syllables. The gaps let the quality gate find a noise floor.
    t_seg = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
    seg = sum(amp * np.sin(2 * np.pi * f0 * k * t_seg)
              for k, amp in enumerate([1.0, 0.5, 0.33, 0.25], start=1))
    gap = np.zeros(int(sr * 0.1))
    parts = []
    while sum(len(p) for p in parts) < sr * seconds:
        parts += [seg, gap]
    signal = np.concatenate(parts)[: int(sr * seconds)]
    signal += 0.001 * np.random.default_rng(0).standard_normal(len(signal))  # noise floor
    signal *= 0.3 / np.max(np.abs(signal))  # normalize headroom

    pcm = (signal * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    print(f"wrote {path} ({seconds}s @ {sr} Hz)")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    make(out, secs)
