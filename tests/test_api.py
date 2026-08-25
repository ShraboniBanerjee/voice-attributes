import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.inference import MockBackend

SR = 16_000


def _wav_bytes(seconds=2.0, f0=120.0, sr=SR):
    """A voiced tone with gaps, encoded as a real WAV byte string."""
    out, t_tone = [], np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * f0 * t_tone)
    gap = np.zeros(int(sr * 0.1))
    while sum(len(c) for c in out) < sr * seconds:
        out += [tone, gap]
    x = np.concatenate(out)[: int(sr * seconds)]
    pcm = (x * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@pytest.fixture()
def client(monkeypatch):
    # Force the fast mock backend regardless of environment config.
    monkeypatch.setattr(app_main, "build_backend", lambda: MockBackend())
    with TestClient(app_main.app) as c:
        yield c


def test_healthz_reports_backend(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["backend"] == "mock"


def test_analyze_returns_valid_contract(client):
    r = client.post(
        "/analyze",
        files={"file": ("s.wav", _wav_bytes(), "audio/wav")},
        data={"contact_id": "contact-123"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["contact_id"] == "contact-123"
    assert body["gender"]["prediction"] in ("male", "female", "unknown")
    assert 0.0 <= body["gender"]["confidence"] <= 1.0
    assert body["age_bracket"]["prediction"] in (
        "18-30", "31-45", "46-60", "60+", "unknown",
    )
    assert body["audio_quality"] in ("good", "degraded", "insufficient")
    assert isinstance(body["processing_ms"], int)


def test_low_pitch_reads_as_male_under_mock(client):
    r = client.post("/analyze", files={"file": ("s.wav", _wav_bytes(f0=110), "audio/wav")})
    assert r.json()["gender"]["prediction"] == "male"


def test_generated_contact_id_when_missing(client):
    r = client.post("/analyze", files={"file": ("s.wav", _wav_bytes(), "audio/wav")})
    assert len(r.json()["contact_id"]) > 0


def test_raw_body_upload(client):
    r = client.post(
        "/analyze", content=_wav_bytes(), headers={"content-type": "audio/wav"}
    )
    assert r.status_code == 200


def test_bad_audio_returns_422(client):
    r = client.post("/analyze", files={"file": ("x.wav", b"not real audio", "audio/wav")})
    assert r.status_code == 422
