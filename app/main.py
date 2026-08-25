"""HTTP API (Task 3) + ops hooks (Task 4).

* Model is built ONCE at startup and reused (loading per-request would wreck
  latency).
* /analyze accepts a multipart upload or a raw audio body.
* Decode failures return a clean 422 instead of a 500 stack trace.
* Every request logs contact_id, quality, latency, and backend - but never the
  audio bytes (observability without storing PII).
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .audio import AudioDecodeError
from .config import settings
from .inference import build_backend
from .pipeline import analyze, analyze_samples
from .schema import AnalyzeResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("voice-attributes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loading backend=%s model=%s", settings.model_backend, settings.model_name)
    t0 = time.perf_counter()
    app.state.backend = build_backend()
    log.info("backend ready in %d ms", int((time.perf_counter() - t0) * 1000))
    yield
    app.state.backend = None


app = FastAPI(title="Voice Attribute Inference", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz(request: Request):
    backend = getattr(request.app.state, "backend", None)
    return {
        "status": "ok" if backend is not None else "starting",
        "backend": getattr(backend, "name", None),
        "model": settings.model_name if settings.model_backend == "wav2vec2" else None,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(
    request: Request,
    file: UploadFile | None = File(default=None),
    contact_id: str | None = Form(default=None),
):
    # Accept either a multipart file upload or a raw audio body.
    if file is not None:
        raw = await file.read()
    else:
        raw = await request.body()

    cid = contact_id or request.query_params.get("contact_id") or str(uuid.uuid4())

    if not raw:
        return JSONResponse(
            status_code=422,
            content={"error": "no audio payload received", "contact_id": cid},
        )

    backend = request.app.state.backend
    try:
        result = analyze(raw, cid, backend)
    except AudioDecodeError as exc:
        log.warning("decode_failed contact_id=%s reason=%s", cid, exc)
        return JSONResponse(
            status_code=422,
            content={"error": f"could not decode audio: {exc}", "contact_id": cid},
        )
    except Exception:  # last-resort guard so one bad call can't take the service down
        log.exception("inference_failed contact_id=%s", cid)
        return JSONResponse(
            status_code=500,
            content={"error": "internal inference error", "contact_id": cid},
        )
    finally:
        del raw  # do not hold caller audio in memory beyond the request

    log.info(
        "analyzed contact_id=%s gender=%s age=%s quality=%s ms=%d backend=%s",
        result.contact_id,
        result.gender.prediction,
        result.age_bracket.prediction,
        result.audio_quality,
        result.processing_ms,
        backend.name,
    )
    return result


@app.websocket("/stream")
async def stream(ws: WebSocket):
    """Streaming inference (bonus).

    Protocol: the client sends binary messages of raw 16-bit little-endian PCM,
    mono, at 16 kHz. The server accumulates the audio and emits a progressive
    prediction (`"final": false`) about once per second of new audio, then a
    `"final": true` result when the client sends the text "end" or disconnects.

    Raw PCM avoids having to decode a partial compressed container on every
    chunk; a gateway would transcode the call leg to PCM before forwarding here.
    Privacy: the buffer lives only in memory and is dropped when the socket closes.
    """
    await ws.accept()
    backend = ws.app.state.backend
    sr = settings.target_sample_rate
    contact_id = ws.query_params.get("contact_id") or str(uuid.uuid4())
    emit_every = sr  # emit at most once per ~1 s of new audio
    buf = bytearray()
    last_emit = 0
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                buf.extend(msg["bytes"])
                if (len(buf) // 2) - last_emit >= emit_every:
                    last_emit = len(buf) // 2
                    await _emit(ws, bytes(buf), sr, contact_id, backend, final=False)
            elif msg.get("text") and msg["text"].strip().lower() in ("end", "close", "done"):
                break
    except WebSocketDisconnect:
        pass

    if len(buf) >= 2:
        await _emit(ws, bytes(buf), sr, contact_id, backend, final=True)
    del buf
    try:
        await ws.close()
    except Exception:
        pass


async def _emit(ws, pcm_bytes, sr, contact_id, backend, final):
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    result = analyze_samples(x, sr, contact_id, backend)
    payload = result.model_dump()
    payload["final"] = final
    log.info(
        "stream contact_id=%s gender=%s age=%s quality=%s final=%s",
        result.contact_id, result.gender.prediction,
        result.age_bracket.prediction, result.audio_quality, final,
    )
    await ws.send_json(payload)
