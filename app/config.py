"""Central configuration.

Every tunable number in the service lives here so there are no "magic numbers"
scattered through the code. Anything can be overridden with an environment
variable, which is how the Docker container gets configured without editing code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


@dataclass(frozen=True)
class Settings:
    # --- Audio ---
    # All audio is resampled to this before anything else. 16 kHz is the
    # standard rate speech models expect (captures frequencies up to 8 kHz,
    # which is where speech information lives).
    target_sample_rate: int = 16_000

    # --- Model backend ---
    # "mock"     -> fast deterministic stub, no heavy download (used for tests/CI)
    # "wav2vec2" -> the real audeering age+gender model (downloaded on first run)
    model_backend: str = _env_str("MODEL_BACKEND", "wav2vec2")
    model_name: str = _env_str(
        "MODEL_NAME", "audeering/wav2vec2-large-robust-24-ft-age-gender"
    )

    # --- Quality gate thresholds ---
    # Below this many seconds of *detected speech* we can't infer reliably.
    min_speech_seconds: float = _env_float("MIN_SPEECH_SECONDS", 0.5)
    # Signal-to-noise ratio (dB): at or above -> "good"; between the two ->
    # "degraded" (we still predict, but trim confidence); the insufficient case
    # is handled by the speech-duration check above.
    snr_good_db: float = _env_float("SNR_GOOD_DB", 15.0)
    snr_degraded_db: float = _env_float("SNR_DEGRADED_DB", 5.0)
    # When audio is "degraded", multiply model confidence by this factor so the
    # service never reports high confidence on noisy input.
    degraded_confidence_factor: float = _env_float("DEGRADED_CONFIDENCE_FACTOR", 0.7)


settings = Settings()
