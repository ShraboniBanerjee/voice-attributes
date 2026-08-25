"""Audio ingestion (Task 1).

We accept arbitrary uploaded bytes (mp3, opus, ogg, wav, telephone mu-law, ...)
and hand them to ffmpeg, which decodes *any* common codec and returns a
standardized 16 kHz mono 16-bit WAV. Everything after this stage in the pipeline
only ever sees one clean format.

Privacy: the audio is streamed through ffmpeg via in-memory pipes. Nothing is
ever written to disk.
"""
from __future__ import annotations

import io
import subprocess
import wave

import numpy as np

from .config import settings


class AudioDecodeError(Exception):
    """Raised when the input bytes could not be decoded into audio."""


def decode_to_mono_f32(raw: bytes) -> tuple[np.ndarray, int]:
    """Decode arbitrary audio bytes to a mono float32 waveform in [-1, 1].

    Returns (samples, sample_rate). Raises AudioDecodeError on failure.
    """
    if not raw:
        raise AudioDecodeError("empty audio payload")

    sr = settings.target_sample_rate
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",          # read the uploaded bytes from stdin
        "-f", "wav",             # output container: WAV
        "-acodec", "pcm_s16le",  # 16-bit signed PCM
        "-ac", "1",              # mono
        "-ar", str(sr),          # resample to target rate
        "pipe:1",                # write result to stdout
    ]
    try:
        proc = subprocess.run(
            cmd, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30,
        )
    except FileNotFoundError as exc:  # ffmpeg not installed
        raise AudioDecodeError("ffmpeg not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError("audio decode timed out") from exc

    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "ignore").strip()[:200]
        raise AudioDecodeError(f"ffmpeg could not decode input: {detail}")

    samples = _read_wav_bytes(proc.stdout)
    return samples, sr


def _read_wav_bytes(wav_bytes: bytes) -> np.ndarray:
    """Parse a 16-bit PCM mono WAV (as produced by ffmpeg above) to float32."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if not raw:
        raise AudioDecodeError("decoded audio contained no samples")
    ints = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return ints / 32768.0  # scale int16 range to [-1, 1)
