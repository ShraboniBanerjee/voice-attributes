"""Attribute inference (Task 2).

The model sits behind a small Backend interface with two implementations:

  * MockBackend    - fast, no download. Estimates pitch for a believable gender
                     guess; used by the test suite and for local smoke tests.
  * Wav2Vec2Backend- the real model: audeering's wav2vec2 fine-tuned for age and
                     gender. One model does both tasks and was trained to be
                     robust to noise, which suits the logistics call setting.

Decoupling the model behind an interface means the whole pipeline is testable
without a 1 GB download, and switching models is a one-line config change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import settings


@dataclass(frozen=True)
class RawPrediction:
    gender_probs: dict[str, float]  # e.g. {"female": .., "male": .., "child": ..}
    age_years: float


class Backend:
    name = "base"

    def predict(self, x: np.ndarray, sr: int) -> RawPrediction:  # pragma: no cover
        raise NotImplementedError


class MockBackend(Backend):
    """Deterministic stub. Gender comes from a crude pitch estimate so demos look
    plausible; age is a fixed placeholder (a stub cannot really infer age)."""

    name = "mock"

    def predict(self, x: np.ndarray, sr: int) -> RawPrediction:
        f0 = _estimate_pitch_hz(x, sr)
        # Map fundamental frequency onto a soft male/female probability.
        # ~85-155 Hz typical male, ~165-255 Hz typical female.
        if f0 <= 0:
            male_p = 0.5
        else:
            male_p = float(np.clip((200.0 - f0) / (200.0 - 120.0), 0.05, 0.95))
        female_p = 1.0 - male_p
        return RawPrediction(
            gender_probs={"female": female_p, "male": male_p, "child": 0.0},
            age_years=35.0,  # placeholder - the mock does not infer age
        )


def _estimate_pitch_hz(x: np.ndarray, sr: int) -> float:
    """Very rough autocorrelation pitch estimate over the loudest ~1 s."""
    if len(x) < sr // 20:
        return 0.0
    seg = x[: sr] if len(x) >= sr else x
    seg = seg - np.mean(seg)
    corr = np.correlate(seg, seg, mode="full")[len(seg) - 1 :]
    # Search lags for f0 between 70 Hz and 300 Hz.
    lo, hi = sr // 300, sr // 70
    if hi >= len(corr):
        return 0.0
    lag = lo + int(np.argmax(corr[lo:hi]))
    return sr / lag if lag else 0.0


class Wav2Vec2Backend(Backend):
    """The real audeering wav2vec2 age+gender model (loaded once, at startup)."""

    name = "wav2vec2"

    def __init__(self, model_name: str) -> None:
        # Heavy imports happen only when this backend is actually used, so the
        # mock/test path never needs torch or transformers installed.
        import torch  # noqa: F401
        import torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

        self._torch = torch

        class _Head(nn.Module):
            def __init__(self, config, num_labels):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, num_labels)

            def forward(self, features):
                x = self.dropout(features)
                x = torch.tanh(self.dense(x))
                x = self.dropout(x)
                return self.out_proj(x)

        class _AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = _Head(config, 1)
                self.gender = _Head(config, 3)  # order: female, male, child
                self.init_weights()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values)[0]
                pooled = torch.mean(hidden, dim=1)
                age = self.age(pooled)
                gender = torch.softmax(self.gender(pooled), dim=1)
                return age, gender

        self._processor = Wav2Vec2Processor.from_pretrained(model_name)
        self._model = _AgeGenderModel.from_pretrained(model_name).eval()

    def predict(self, x: np.ndarray, sr: int) -> RawPrediction:
        torch = self._torch
        inputs = self._processor(x, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            age, gender = self._model(inputs["input_values"])
        gender = gender[0].tolist()
        return RawPrediction(
            gender_probs={"female": gender[0], "male": gender[1], "child": gender[2]},
            age_years=float(age[0][0]) * 100.0,  # model outputs age in [0, 1]
        )


def build_backend() -> Backend:
    """Factory: choose the backend from configuration."""
    if settings.model_backend == "mock":
        return MockBackend()
    if settings.model_backend == "wav2vec2":
        return Wav2Vec2Backend(settings.model_name)
    raise ValueError(f"unknown MODEL_BACKEND: {settings.model_backend!r}")
