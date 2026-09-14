"""Audio decoding, done once per upload.

The browser records WebM/Opus (that's what MediaRecorder gives us on Chrome and
Firefox), which soundfile and librosa can't open. faster-whisper ships PyAV as a
dependency and exposes a decoder that handles it, so we reuse that rather than
requiring a separate ffmpeg install on the user's PATH.

Both the ASR and the voice-emotion head want the same 16 kHz mono float32
waveform, so decoding lives here and the result is passed to both.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


def decode(data: bytes) -> Optional[np.ndarray]:
    """Decode arbitrary browser-recorded audio to 16 kHz mono float32 in [-1, 1].

    Returns None if the bytes can't be decoded at all, so callers can degrade to
    a text-only reading rather than failing the whole request.
    """
    if not data:
        return None

    try:
        from faster_whisper.audio import decode_audio

        waveform = decode_audio(io.BytesIO(data), sampling_rate=SAMPLE_RATE)
        return np.asarray(waveform, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        log.warning("PyAV decode failed (%s); trying soundfile", exc)

    # Fallback for plain WAV uploads, which soundfile handles without PyAV.
    try:
        import soundfile as sf

        waveform, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        mono = waveform.mean(axis=1)
        if rate != SAMPLE_RATE:
            mono = _resample_linear(mono, rate, SAMPLE_RATE)
        return mono.astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        log.error("Could not decode uploaded audio: %s", exc)
        return None


def _resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resample. Crude, but this path only runs for WAV fallbacks."""
    if src_rate == dst_rate or x.size == 0:
        return x
    duration = x.size / src_rate
    target_len = int(round(duration * dst_rate))
    return np.interp(
        np.linspace(0.0, duration, target_len, endpoint=False),
        np.linspace(0.0, duration, x.size, endpoint=False),
        x,
    ).astype(np.float32)


def duration_seconds(waveform: Optional[np.ndarray]) -> float:
    if waveform is None or waveform.size == 0:
        return 0.0
    return float(waveform.size) / SAMPLE_RATE
