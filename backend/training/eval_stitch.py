"""Measure what stitching 3s chunks costs against one full-context pass.

    python -m training.eval_stitch --audio ../data/stitch
    python -m training.eval_stitch --audio ../data/stitch --json ../data/eval_stitch.json

The app currently does both: 3s chunks drive the live preview, and at stop() the
whole recording is re-transcribed from scratch because "Whisper is markedly
better with full context" (capture/session.ts). Dropping that final pass and
keeping only the stitched chunks is what makes an in-browser port viable — a
3-minute entry re-transcribed on WASM blows the client's 180s timeout, whereas
chunks that already ran during recording cost nothing extra at stop.

This script answers whether that trade is affordable. For each recording it runs:

  full      one transcribe() over the whole waveform          (today's saved entry)
  stitched  transcribe() per 3s chunk, texts joined           (the proposed entry)

and reports, per condition:

* **WER** against a reference transcript, if one is supplied
* **the emotion reading the app would actually store** — dominant label and the
  full 7-vector, from the same text_model the sidecar uses

The second is the number that decides this, not WER. The transcript is an
intermediate: nothing in the app displays it in isolation, it feeds the text
channel of fusion. A stitched transcript can lose several points of WER and
still produce an identical emotion reading, in which case the full pass is
buying nothing. It can also drop one negation and flip the label. WER cannot
tell those apart; label agreement can.

Honest caveats, which belong in the write-up alongside any number from here:

* Chunking here is faithful to mic.ts in length (3s) and in seam loss (the live
  recorder restarts between clips and drops a few ms each time, so `--seam-ms`
  is subtracted at every boundary). It is *not* faithful in codec: this decodes
  once and slices the waveform, where the app encodes each clip to WebM/Opus
  separately. Opus at conversational bitrates is transparent enough that this
  should not matter, but it is an untested assumption.
* Whisper's own VAD and its 30s training window mean per-chunk transcription is
  disadvantaged in a way that no amount of stitching recovers. That is the
  effect being measured, so this is the point, but it also means the result is
  specific to `base.en` — `tiny.en` and streaming-first models like Moonshine
  will have a different, probably smaller, gap.
* Sample sizes here will be tiny. Report per-file results, not just the mean.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import audio_io  # noqa: E402
from app.asr import recognizer  # noqa: E402
from app.config import settings  # noqa: E402
from app.emotions import EMOTIONS, dominant  # noqa: E402
from app.text_model import text_model  # noqa: E402

# Mirrors CHUNK_MS in frontend/src/capture/session.ts. If that constant moves,
# this one has to move with it or the measurement stops describing the app.
CHUNK_MS = 3000

# The live recorder stops and restarts between clips, losing a few ms at each
# seam (see the header comment in capture/mic.ts). Modelled, because those lost
# milliseconds land mid-word and are a real part of what stitching costs.
DEFAULT_SEAM_MS = 60

AUDIO_SUFFIXES = {".wav", ".webm", ".m4a", ".mp3", ".ogg", ".flac"}


@dataclass
class Pass:
    """One transcription condition over one recording."""

    name: str
    text: str
    vector: List[float]
    label: str
    seconds: float
    """Wall-clock transcription time, for the realtime factor."""


# --- join strategies -------------------------------------------------------
#
# The first stitch run showed the damage is not random word error. WER did not
# predict label flips at all (an entry flipped at WER 0.044; another held at
# 0.391). What flipped labels was *sentence boundary corruption*: Whisper ends
# every chunk as though it were a complete utterance, so a seam mid-sentence
# becomes a full stop, and text_model then splits and classifies around a
# boundary that is not there. Joining is therefore not a formatting detail —
# it is the whole experiment.

# Phrases Whisper emits when a window is mostly silence. A trailing 3s chunk is
# exactly that case, which is why two of six entries ended with one of these.
HALLUCINATIONS = {
    "thank you", "thank you very much", "thanks for watching", "thanks",
    "bye", "bye bye", "goodbye", "you", "okay", "so", "please subscribe",
    "subtitles by the amara.org community", "transcription by castingwords",
}


def _is_hallucination(chunk: str) -> bool:
    """True if the whole chunk is a known filler phrase.

    Deliberately exact-match on the entire chunk: "you" and "so" are ordinary
    words, and dropping them mid-sentence would cause far more damage than the
    hallucination does. A 3s chunk that transcribes to nothing *but* "You" is a
    different matter — that is silence being filled.
    """
    return " ".join(normalise(chunk)) in HALLUCINATIONS


def join_raw(chunks: List[str]) -> str:
    """What the client does today: accumulate with a space, trust Whisper."""
    return " ".join(c for c in chunks if c)


def join_filtered(chunks: List[str]) -> str:
    """Drop hallucinated chunks, otherwise unchanged."""
    return " ".join(c for c in chunks if c and not _is_hallucination(c))


def join_stripped(chunks: List[str]) -> str:
    """Drop hallucinations, then remove sentence-final punctuation at seams.

    Aggressive: it assumes every chunk boundary is mid-sentence. At 3s chunks
    over speech whose sentences run 4-5s, most boundaries are — but the ones
    that genuinely coincide with a sentence end get merged into their
    neighbour, which is the cost being measured against join_smart.
    """
    kept = [c for c in chunks if c and not _is_hallucination(c)]
    out = [re.sub(r"[.!?]+$", "", c.strip()) for c in kept[:-1]]
    return " ".join(out + ([kept[-1].strip()] if kept else []))


def join_smart(chunks: List[str]) -> str:
    """Drop the seam's full stop only when the next chunk resumes lowercase.

    Whisper capitalises what it believes is a sentence start, so a chunk ending
    "...the whole project." followed by "in the meeting..." is self-evidently a
    false boundary, while one followed by "But hear me..." may well be real.
    Cheaper than it sounds and it keeps the true boundaries that join_stripped
    throws away.
    """
    kept = [c.strip() for c in chunks if c and not _is_hallucination(c)]
    if not kept:
        return ""

    out = []
    for i, chunk in enumerate(kept[:-1]):
        following = kept[i + 1]
        resumes_lowercase = following[:1].islower()
        out.append(re.sub(r"[.!?]+$", "", chunk) if resumes_lowercase else chunk)
    out.append(kept[-1])
    return " ".join(out)


JOIN_STRATEGIES = {
    "raw": join_raw,
    "filtered": join_filtered,
    "stripped": join_stripped,
    "smart": join_smart,
}


def normalise(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace.

    WER is meaningless without a normalisation policy, and the policy has to be
    the same for both conditions. Whisper's punctuation and casing are its own
    inventions — penalising a stitched pass for putting a comma somewhere else
    would measure the wrong thing entirely.
    """
    cleaned = re.sub(r"[^\w\s']", " ", text.lower())
    return cleaned.split()


def word_error_rate(reference: str, hypothesis: str) -> Optional[float]:
    """Levenshtein distance over words, divided by reference length.

    Written out rather than pulling in `jiwer`: it is fifteen lines, and the
    backend's dependency list is already heavy enough that anything avoidable
    should be avoided.
    """
    ref = normalise(reference)
    hyp = normalise(hypothesis)
    if not ref:
        return None

    # Single-row DP; we only ever need the previous row.
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(
                previous[j - 1] if r == h
                else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        previous = current

    return previous[-1] / len(ref)


def chunk_waveform(
    waveform: np.ndarray, chunk_ms: int, seam_ms: int
) -> List[np.ndarray]:
    """Slice into chunk_ms pieces, dropping seam_ms at each boundary."""
    rate = audio_io.SAMPLE_RATE
    step = int(rate * chunk_ms / 1000)
    seam = int(rate * seam_ms / 1000)

    chunks = []
    start = 0
    while start < waveform.size:
        end = min(start + step, waveform.size)
        piece = waveform[start:end]
        # A trailing sliver shorter than the seam is what the recorder would
        # have swallowed at the final restart; nothing survives it.
        if piece.size > seam:
            chunks.append(piece)
        start = end + seam

    return chunks


def score(name: str, text: str, seconds: float) -> Pass:
    """Run one transcript through the app's text model."""
    vector = text_model.predict(text) if text.strip() else [0.0] * len(EMOTIONS)
    return Pass(
        name=name,
        text=text,
        vector=vector,
        label=dominant(vector) if text.strip() else "neutral",
        seconds=seconds,
    )


def cosine(a: List[float], b: List[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom else 0.0


def evaluate(path: Path, seam_ms: int) -> Optional[Dict]:
    """Run both conditions over one recording."""
    waveform = audio_io.decode(path.read_bytes())
    if waveform is None or waveform.size == 0:
        print(f"  !! could not decode {path.name}")
        return None

    duration = waveform.size / audio_io.SAMPLE_RATE
    chunks = chunk_waveform(waveform, CHUNK_MS, seam_ms)

    started = time.perf_counter()
    full = score("full", recognizer.transcribe(waveform).text, time.perf_counter() - started)

    # Transcribe the chunks once and join them four ways. Every strategy then
    # sees byte-identical ASR output, so any difference between them is the
    # joining and nothing else.
    started = time.perf_counter()
    parts = [recognizer.transcribe(c).text.strip() for c in chunks]
    chunk_seconds = time.perf_counter() - started

    passes = [full] + [
        score(name, fn(parts), chunk_seconds) for name, fn in JOIN_STRATEGIES.items()
    ]
    stitched = next(p for p in passes if p.name == "raw")

    # A .txt beside the audio is the reference. Optional: without it the
    # emotion-agreement columns still work, which is most of the point.
    reference_path = path.with_suffix(".txt")
    reference = reference_path.read_text(encoding="utf-8").strip() if reference_path.exists() else None

    result = {
        "file": path.name,
        "durationSeconds": round(duration, 2),
        "chunks": len(chunks),
        "agree": full.label == stitched.label,
        "cosine": round(cosine(full.vector, stitched.vector), 4),
        "conditions": {},
    }

    for p in passes:
        entry = {
            "text": p.text,
            "label": p.label,
            "vector": [round(v, 4) for v in p.vector],
            "seconds": round(p.seconds, 2),
            "realtimeFactor": round(p.seconds / duration, 3) if duration else None,
            # Against the full pass, which is the thing being replaced.
            "agreesWithFull": p.label == full.label,
            "cosineToFull": round(cosine(p.vector, full.vector), 4),
        }
        if reference is not None:
            wer = word_error_rate(reference, p.text)
            entry["wer"] = round(wer, 4) if wer is not None else None
        result["conditions"][p.name] = entry

    return result


def report(results: List[Dict]) -> None:
    names = ["full"] + list(JOIN_STRATEGIES)

    print(f"\n{'=' * 74}\nPer recording (label, vs the full pass)\n{'=' * 74}")
    header = "".join(f"{n:<11}" for n in names)
    print(f"\n  {'file':<14}{header}")
    for r in results:
        cells = ""
        for n in names:
            c = r["conditions"][n]
            mark = "" if n == "full" or c["agreesWithFull"] else "*"
            cells += f"{c['label'] + mark:<11}"
        print(f"  {r['file']:<14}{cells}")
    print("\n  * = disagrees with the full pass, i.e. stitching changed the reading")

    print(f"\n{'=' * 74}\nSummary\n{'=' * 74}")
    print(f"\n  {'strategy':<11}{'agree':<9}{'cosine':<9}{'WER':<9}")
    for n in names:
        conds = [r["conditions"][n] for r in results]
        agree = sum(c["agreesWithFull"] for c in conds)
        cos = np.mean([c["cosineToFull"] for c in conds])
        wers = [c["wer"] for c in conds if c.get("wer") is not None]
        wer_cell = f"{np.mean(wers):.3f}" if wers else "n/a"
        print(f"  {n:<11}{f'{agree}/{len(conds)}':<9}{cos:<9.3f}{wer_cell:<9}")

    print(
        "\n  'agree' and 'cosine' are measured against the full pass, which is\n"
        "  the thing being replaced - not against the intended labels. The\n"
        "  question is whether stitching reproduces today's reading, not\n"
        "  whether today's reading is correct.\n"
        "\n  With a handful of recordings none of these means are significant.\n"
        "  Read the per-file table, and when a label flips, diff the two\n"
        "  transcripts and find out what it dropped."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="directory of recordings; a matching .txt alongside enables WER",
    )
    parser.add_argument(
        "--seam-ms",
        type=int,
        default=DEFAULT_SEAM_MS,
        help=f"ms dropped at each chunk boundary (default {DEFAULT_SEAM_MS})",
    )
    parser.add_argument(
        "--whisper-model",
        help="override settings.whisper_model, e.g. tiny.en. The browser port "
        "would likely run tiny.en on the live chunks, so the gap between it "
        "and base.en here is the other half of the decision.",
    )
    parser.add_argument("--json", type=Path, help="write full results here")
    args = parser.parse_args()

    # Safe to set here: asr.py reads this lazily inside _ensure_loaded(), which
    # has not run yet because nothing has called transcribe().
    if args.whisper_model:
        settings.whisper_model = args.whisper_model

    if not args.audio.is_dir():
        print(f"No such directory: {args.audio}")
        raise SystemExit(1)

    files = sorted(
        p for p in args.audio.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES
    )
    if not files:
        print(f"No audio in {args.audio} (looked for {', '.join(sorted(AUDIO_SUFFIXES))})")
        raise SystemExit(1)

    print(f"Recordings: {len(files)}")
    print(f"Chunk      : {CHUNK_MS}ms, seam {args.seam_ms}ms")
    print(f"Whisper    : {settings.whisper_model} ({settings.whisper_compute_type})")
    print("Loading models (first run downloads weights)...")

    results = []
    for path in files:
        print(f"  {path.name}")
        result = evaluate(path, args.seam_ms)
        if result is not None:
            results.append(result)

    if not results:
        print("Nothing decoded.")
        raise SystemExit(1)

    report(results)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
