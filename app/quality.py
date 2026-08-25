"""Audio quality gate (part of Task 4: degrade gracefully).

The assignment's key requirement: surface an `audio_quality` flag instead of
silently returning bad predictions. We estimate two things with a lightweight,
fully explainable energy analysis (no heavy dependencies):

  * SNR (signal-to-noise ratio) - how far the voice rises above background noise.
  * speech_seconds           - roughly how much actual speech is present.

Then we map those to good / degraded / insufficient. A production system would
swap the energy-based speech detector for a learned VAD (e.g. Silero or
webrtcvad); the interface here stays the same.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import settings

# 30 ms frames with a 10 ms hop is the classic speech-framing choice.
_FRAME_MS = 30
_HOP_MS = 10


@dataclass(frozen=True)
class QualityReport:
    flag: str            # "good" | "degraded" | "insufficient"
    snr_db: float
    speech_seconds: float
    duration_seconds: float


def _frame_energies(x: np.ndarray, sr: int) -> np.ndarray:
    """Return RMS energy for each overlapping frame."""
    frame = int(sr * _FRAME_MS / 1000)
    hop = int(sr * _HOP_MS / 1000)
    if len(x) < frame:
        return np.array([np.sqrt(np.mean(x**2)) + 1e-10]) if len(x) else np.array([1e-10])
    n = 1 + (len(x) - frame) // hop
    energies = np.empty(n, dtype=np.float64)
    for i in range(n):
        seg = x[i * hop : i * hop + frame]
        energies[i] = np.sqrt(np.mean(seg**2))
    return energies + 1e-10  # avoid log(0)


def assess(x: np.ndarray, sr: int) -> QualityReport:
    duration = len(x) / sr if sr else 0.0
    energies = _frame_energies(x, sr)

    # Noise floor = quiet frames (10th pct); speech level = loud frames (90th pct).
    noise_floor = float(np.percentile(energies, 10))
    speech_level = float(np.percentile(energies, 90))
    snr_db = 20.0 * np.log10(speech_level / noise_floor)

    # A frame counts as "speech" if it sits clearly above the noise floor.
    speech_threshold = noise_floor * 3.0
    speech_frames = int(np.sum(energies > speech_threshold))
    speech_seconds = speech_frames * (_HOP_MS / 1000.0)

    flag = _classify(snr_db, speech_seconds)
    return QualityReport(
        flag=flag,
        snr_db=round(snr_db, 2),
        speech_seconds=round(speech_seconds, 3),
        duration_seconds=round(duration, 3),
    )


def _classify(snr_db: float, speech_seconds: float) -> str:
    # Not enough actual speech to infer anything -> refuse to guess.
    if speech_seconds < settings.min_speech_seconds:
        return "insufficient"
    if snr_db >= settings.snr_good_db:
        return "good"
    if snr_db >= settings.snr_degraded_db:
        return "degraded"
    # Speech is present but buried in noise: usable but low-trust.
    return "degraded"
