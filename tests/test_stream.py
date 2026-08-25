import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.inference import MockBackend

SR = 16_000


def _pcm_bytes(seconds=2.0, f0=120.0, sr=SR):
    """Raw 16-bit PCM of a voiced tone with gaps (what a client would stream)."""
    out, t_seg = [], np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
    seg = 0.3 * np.sin(2 * np.pi * f0 * t_seg)
    gap = np.zeros(int(sr * 0.1))
    while sum(len(c) for c in out) < sr * seconds:
        out += [seg, gap]
    x = np.concatenate(out)[: int(sr * seconds)]
    return (x * 32767).astype(np.int16).tobytes()


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(app_main, "build_backend", lambda: MockBackend())
    with TestClient(app_main.app) as c:
        yield c


def test_streaming_emits_progressive_then_final(client):
    pcm = _pcm_bytes(seconds=2.0)
    chunk = SR * 2 // 4  # ~0.5 s of int16 samples -> bytes
    chunk_bytes = chunk * 2

    received = []
    with client.websocket_connect("/stream?contact_id=ws-1") as ws:
        for i in range(0, len(pcm), chunk_bytes):
            ws.send_bytes(pcm[i : i + chunk_bytes])
        ws.send_text("end")
        for _ in range(20):  # drain until we see the final message
            msg = ws.receive_json()
            received.append(msg)
            if msg.get("final"):
                break

    assert received, "expected at least one streamed prediction"
    assert received[-1]["final"] is True
    assert received[-1]["contact_id"] == "ws-1"
    assert received[-1]["gender"]["prediction"] in ("male", "female", "unknown")
