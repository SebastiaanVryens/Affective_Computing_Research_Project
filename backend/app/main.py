"""FastAPI sidecar for the Mindscape diary.

Deliberately stateless: the browser owns all diary data (IndexedDB) and this
process only does inference. That keeps the privacy story simple — audio arrives,
gets turned into numbers and text, and is never written to disk.

Endpoints
    GET  /api/health   what's loaded, what device, which fusion weights
    POST /api/analyze  the main one: audio (+ optional face vector) -> reading
    POST /api/text     text-only reading, for typed entries and for testing
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import audio_io
from .asr import recognizer
from .audio_model import voice_model
from .config import settings
from .emotions import EMOTIONS
from . import fusion as fusion_module
from .emotions import blend as blend_vectors, dominant as dominant_of, sentiment as sentiment_of
from .fusion import FusedResult, fuse
from .keywords import aggregate_by_emotion, extract
from .text_model import split_sentences, text_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
log = logging.getLogger("mindscape")

app = FastAPI(title="Mindscape", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TextRequest(BaseModel):
    text: str
    face_vector: Optional[List[float]] = Field(default=None, alias="faceVector")

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": settings.resolved_device(),
        "emotions": EMOTIONS,
        "models": {
            "text": text_model.status(),
            "asr": recognizer.status(),
            "voice_emotion_enabled": settings.enable_voice_emotion,
        },
        "warm": text_model.status()["loaded"] and recognizer.status()["loaded"],
        "fusion_weights": {
            "text": settings.weight_text,
            "face": settings.weight_face,
            "voice": settings.weight_voice,
        },
    }


@app.post("/api/warmup")
def warmup() -> dict:
    """Load the models now, in the background, so the first chunk isn't slow.

    Everything here is lazy-loaded on first use, which costs about nineteen
    seconds. If that lands on the first streamed chunk of a recording, the
    session's opening seconds are silently dropped — the frontend skips chunks
    while one is still in flight — and it looks to the user like live
    transcription simply doesn't work.

    Called by the browser as soon as it sees the sidecar is up, so the loading
    happens while the page is still being read rather than mid-sentence. Returns
    immediately; the work continues on a daemon thread.
    """
    if getattr(app.state, "warming", False):
        return {"status": "already-warming"}
    app.state.warming = True

    def load() -> None:
        try:
            log.info("Warming models…")
            started = time.perf_counter()
            text_model.predict("warming up the text model")
            recognizer.transcribe(np.zeros(16_000, dtype=np.float32))
            # Kicks off its own background thread and returns at once. This is
            # the ~1GB one, so it finishes well after the others; until it does,
            # the voice channel falls back to prosody.
            voice_model.start_loading()
            log.info("Models warm in %.1fs", time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 - warmup must never break startup
            log.warning("Warmup failed (models will load on first use): %s", exc)
        finally:
            app.state.warming = False

    threading.Thread(target=load, daemon=True, name="warmup").start()
    return {"status": "warming"}


@app.post("/api/analyze")
async def analyze(
    audio: UploadFile = File(...),
    # The browser scores faces locally with face-api and sends the session's
    # aggregated vector along, so the video stream itself never leaves the page.
    face_vector: Optional[str] = Form(default=None),
    # Timestamped face readings, so each utterance is fused against the face
    # from that moment rather than against a whole-session average.
    face_timeline: Optional[str] = Form(default=None),
    weights: Optional[str] = Form(default=None),
) -> dict:
    """Read the upload, then hand the heavy work to a worker thread.

    The handoff is not optional. Everything below — decoding, Whisper, the
    transformers heads — is synchronous and CPU-bound, and running it directly
    in an `async def` body executes it *on the event loop*, where it blocks
    every other request in the process. That included `/api/health`, so a single
    slow analyse made the browser report the whole sidecar as offline while the
    server sat there at full CPU.
    """
    raw = await audio.read()
    return await run_in_threadpool(
        _analyze_sync, raw, face_vector, face_timeline, weights
    )


def _analyze_sync(
    raw: bytes,
    face_vector: Optional[str],
    face_timeline: Optional[str],
    weights: Optional[str],
) -> dict:
    started = time.perf_counter()

    waveform = audio_io.decode(raw)
    duration = audio_io.duration_seconds(waveform)

    transcript = recognizer.transcribe(waveform)

    # Per-sentence scoring serves two purposes: it gives keywords something to
    # attribute emotion to, and it surfaces the entry's emotional peak — which is
    # a far better core-memory candidate than the whole-entry average.
    sentences = split_sentences(transcript.text)
    sentence_vectors = text_model.predict_batch(sentences) if sentences else []

    if sentences:
        # Weight each sentence by length so a passing "yeah" doesn't count as
        # much as a paragraph about the thing that actually happened.
        weights_by_len = [max(1, len(s.split())) for s in sentences]
        text_vector = [
            sum(v[i] * w for v, w in zip(sentence_vectors, weights_by_len))
            / sum(weights_by_len)
            for i in range(len(EMOTIONS))
        ]
    else:
        text_vector = None

    voice_vector, voice_source = voice_model.predict(waveform)
    parsed_face = _parse_vector(face_vector)
    timeline = _parse_timeline(face_timeline)
    parsed_weights = _parse_weights(weights)

    # Per-utterance fusion, when the browser sent a face timeline.
    #
    # Fusing once over the whole entry throws away the thing that makes three
    # channels worth having: *co-occurrence*. A face reading averaged over two
    # minutes can't corroborate any particular sentence. Aligned to the segment
    # it belongs to, it can — and agreement between channels in the same few
    # seconds is much stronger evidence than agreement between two averages.
    aligned = _fuse_per_segment(
        transcript.segments, sentence_vectors, sentences, timeline,
        voice_vector if duration > 0 else None, voice_source, parsed_weights,
    )

    if aligned is not None:
        result, segment_readings = aligned
    else:
        segment_readings = []
        result = fuse(
            text_vector=text_vector,
            face_vector=parsed_face,
            voice_vector=voice_vector if duration > 0 else None,
            text_source="meld-text" if text_vector else None,
            face_source="face-api-browser" if parsed_face else None,
            voice_source=voice_source,
            weights=parsed_weights,
        )

    found = extract(sentences, sentence_vectors)
    peak = _peak_sentence(sentences, sentence_vectors)

    return {
        "transcript": {
            "text": transcript.text,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in transcript.segments
            ],
            "language": transcript.language,
            "confidence": round(transcript.avg_logprob, 3),
            # Diagnostics for the commonest failure: no words came back.
            "level": round(transcript.rms, 5),
            "silent": transcript.rms < 0.002,
        },
        "reading": result.to_dict(),
        "keywords": [k.to_dict() for k in found],
        "keywordsByEmotion": aggregate_by_emotion(found),
        "sentences": [
            {"text": s, "vector": [round(p, 4) for p in v]}
            for s, v in zip(sentences, sentence_vectors)
        ],
        "peak": peak,
        "segments": segment_readings,
        "alignment": {
            "perSegment": bool(segment_readings),
            "faceSamples": len(timeline),
        },
        "audio": {
            "durationSeconds": round(duration, 2),
            "decoded": waveform is not None,
        },
        "timing": {"totalMs": round((time.perf_counter() - started) * 1000)},
    }


@app.post("/api/text")
def analyze_text(request: TextRequest) -> dict:
    """Text-only path. Used for typed entries and for scripted evaluation."""
    sentences = split_sentences(request.text)
    sentence_vectors = text_model.predict_batch(sentences) if sentences else []
    text_vector = (
        [
            sum(v[i] for v in sentence_vectors) / len(sentence_vectors)
            for i in range(len(EMOTIONS))
        ]
        if sentence_vectors
        else None
    )

    result = fuse(
        text_vector=text_vector,
        face_vector=request.face_vector,
        text_source="meld-text" if text_vector else None,
        face_source="face-api-browser" if request.face_vector else None,
    )
    found = extract(sentences, sentence_vectors)

    return {
        "reading": result.to_dict(),
        "keywords": [k.to_dict() for k in found],
        "keywordsByEmotion": aggregate_by_emotion(found),
        "peak": _peak_sentence(sentences, sentence_vectors),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_vector(raw: Optional[str]) -> Optional[List[float]]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) == len(EMOTIONS):
            return [float(x) for x in parsed]
    except (json.JSONDecodeError, TypeError, ValueError):
        log.warning("Ignoring malformed face vector")
    return None


def _parse_timeline(raw: Optional[str]) -> List[dict]:
    """Parse the browser's timestamped face readings. Malformed input is dropped."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []
        return [
            s
            for s in parsed
            if isinstance(s, dict)
            and isinstance(s.get("t"), (int, float))
            and isinstance(s.get("v"), list)
            and len(s["v"]) == len(EMOTIONS)
        ]
    except (json.JSONDecodeError, TypeError, ValueError):
        log.warning("Ignoring malformed face timeline")
        return []


def _fuse_per_segment(
    segments,
    sentence_vectors,
    sentences,
    timeline,
    voice_vector,
    voice_source,
    weights,
):
    """Fuse each spoken segment against the face from that same moment.

    Returns (entry_level_result, per_segment_readings), or None when there isn't
    enough to align — no timeline, or no speech segments — in which case the
    caller falls back to whole-entry fusion.

    Text is matched to segments positionally. Whisper's segmentation and our
    sentence splitting don't produce identical boundaries, so this is approximate;
    it's good enough because both are ordered and roughly sentence-sized, and the
    cost of a one-off misalignment is small compared to having no alignment.
    """
    if not timeline or not segments:
        return None

    readings = []
    for i, segment in enumerate(segments):
        # Whisper emits trailing fragments of punctuation ("." on its own) past
        # the end of real speech. Fused against a face timeline those become
        # confident face-only readings of nothing, which skews the entry.
        if len(segment.text.strip(" .,!?-")) < 3:
            continue

        segment_text = sentence_vectors[i] if i < len(sentence_vectors) else None
        segment_face = fusion_module.face_at(timeline, segment.start, segment.end)
        if segment_text is None and segment_face is None:
            continue

        readings.append(
            (
                segment,
                fuse(
                    text_vector=segment_text,
                    face_vector=segment_face,
                    # One voice reading covers the whole clip; the SER model
                    # isn't run per segment because that would multiply the
                    # heaviest inference in the pipeline by the segment count.
                    voice_vector=voice_vector,
                    text_source="meld-text" if segment_text is not None else None,
                    face_source="face-api-aligned" if segment_face is not None else None,
                    voice_source=voice_source,
                    weights=weights,
                ),
            )
        )

    if not readings:
        return None

    # Entry level is the duration-weighted mean of the segment readings, so a
    # long reflective passage counts for more than a two-word aside.
    entry_vector = blend_vectors(
        [
            (r.vector, max(0.5, seg.end - seg.start) * (0.4 + 0.6 * r.certainty))
            for seg, r in readings
        ]
    )

    entry = FusedResult(
        vector=entry_vector,
        dominant=dominant_of(entry_vector),
        sentiment=sentiment_of(entry_vector),
        certainty=fusion_module.certainty(entry_vector),
        # Report the channel breakdown from the most emotionally committed
        # segment: it's the one that explains why the entry reads as it does.
        modalities=max(readings, key=lambda r: r[1].certainty)[1].modalities,
    )

    per_segment = [
        {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text,
            "dominant": r.dominant,
            "certainty": round(r.certainty, 4),
            "vector": [round(p, 4) for p in r.vector],
            "faceAligned": any(m.name == "face" and m.available for m in r.modalities),
        }
        for seg, r in readings
    ]
    return entry, per_segment


def _parse_weights(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _peak_sentence(sentences, vectors) -> Optional[dict]:
    """The single most emotionally committed sentence in the entry.

    Neutral is excluded from the running, because the most confident sentence in
    a calm entry is almost always a confidently neutral one — which makes for a
    useless core memory.
    """
    best = None
    best_score = 0.0
    neutral_idx = EMOTIONS.index("neutral")

    for sentence, vector in zip(sentences, vectors):
        score = max(p for i, p in enumerate(vector) if i != neutral_idx)
        if score > best_score:
            best_score = score
            best = {
                "text": sentence,
                "vector": [round(p, 4) for p in vector],
                "emotion": EMOTIONS[max(range(len(vector)), key=lambda i: vector[i])],
                "score": round(score, 4),
            }
    return best
