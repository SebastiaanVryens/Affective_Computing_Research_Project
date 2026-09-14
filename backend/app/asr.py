"""Local speech-to-text via faster-whisper.

Nothing leaves the machine. The model is lazy-loaded on first use (the initial
call downloads weights to the HF cache) and then held for the process lifetime.

Segment timings are kept, not thrown away: they're what lets the fusion layer
line a spoken sentence up with the face reading from the same few seconds.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import settings

log = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    text: str


# Below this RMS the clip really is silence (muted mic, wrong input device),
# and retrying without the VAD would only invite hallucinated text.
SILENCE_RMS = 0.002


@dataclass
class Transcript:
    text: str
    segments: List[Segment]
    language: str
    # Whisper's own confidence proxy, averaged over segments. Low values usually
    # mean silence or background noise rather than genuine speech.
    avg_logprob: float
    # Signal level of the clip. Surfaced to the client because "no words
    # appeared" has two very different causes — nothing was said, or nothing was
    # heard — and only this number tells them apart. Defaulted, so it must come
    # after every field that isn't.
    rms: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class SpeechRecognizer:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False

        with self._lock:
            if self._model is not None:
                return True
            if self._load_error is not None:
                return False
            try:
                from faster_whisper import WhisperModel

                device = settings.resolved_device()
                compute = settings.whisper_compute_type
                # int8 is a CPU quantisation; on CUDA it's slower than float16.
                if device == "cuda" and compute == "int8":
                    compute = "float16"

                log.info(
                    "Loading Whisper %s (device=%s, compute=%s)",
                    settings.whisper_model,
                    device,
                    compute,
                )
                self._model = WhisperModel(
                    settings.whisper_model, device=device, compute_type=compute
                )
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)
                log.error("Whisper unavailable: %s", exc)
                return False
        return True

    def transcribe(self, waveform: Optional[np.ndarray]) -> Transcript:
        empty = Transcript(text="", segments=[], language="en", avg_logprob=-10.0)

        if waveform is None or waveform.size == 0 or not self._ensure_loaded():
            return empty

        rms = float(np.sqrt(np.mean(waveform**2))) if waveform.size else 0.0

        try:
            result = self._run(waveform, use_vad=True)

            # The VAD is tuned for continuous speech and is noticeably trigger
            # happy on the short clips the live preview sends — a three-second
            # window caught mid-sentence, or recorded through a quiet laptop mic
            # with gain control disabled, can be discarded wholesale. That shows
            # up as a recording where no words ever appear.
            #
            # So when it returns nothing but the audio clearly *had* signal, try
            # again unfiltered. Genuinely silent input still costs one wasted
            # pass and is caught by the level check instead.
            if result.is_empty and rms > SILENCE_RMS:
                log.info(
                    "VAD discarded everything but the clip has signal "
                    "(rms=%.4f); retrying without it",
                    rms,
                )
                result = self._run(waveform, use_vad=False)

            result.rms = rms
            return result
        except Exception as exc:  # noqa: BLE001
            log.error("Transcription failed: %s", exc)
            empty.rms = rms
            return empty

    def _run(self, waveform: np.ndarray, use_vad: bool) -> Transcript:
        segments, info = self._model.transcribe(
            waveform,
            beam_size=5,
            # Whisper hallucinates fluent nonsense over silence; its own VAD
            # filter is the cheapest defence against a diary full of
            # "Thank you for watching!"
            vad_filter=use_vad,
            vad_parameters={"min_silence_duration_ms": 500} if use_vad else None,
            condition_on_previous_text=False,
        )

        collected: List[Segment] = []
        logprobs: List[float] = []
        for seg in segments:  # generator: this is where work happens
            text = seg.text.strip()
            if not text:
                continue
            collected.append(Segment(start=seg.start, end=seg.end, text=text))
            logprobs.append(seg.avg_logprob)

        return Transcript(
            text=" ".join(s.text for s in collected).strip(),
            segments=collected,
            language=getattr(info, "language", "en"),
            avg_logprob=float(np.mean(logprobs)) if logprobs else -10.0,
        )

    def status(self) -> dict:
        return {
            "loaded": self._model is not None,
            "model": settings.whisper_model,
            "error": self._load_error,
        }


recognizer = SpeechRecognizer()
