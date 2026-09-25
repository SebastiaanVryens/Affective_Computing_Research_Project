"""Dump every channel's per-utterance vector on MELD, aligned for fusion.

    python -m training.dump_channels --split test --source <MELD_data>
    python -m training.dump_channels --split dev  --source <MELD_data>

Each channel has been measured on its own. None of them has been measured
*together*, which is the thing this project actually claims: certainty-weighted
late fusion, cross-modal reinforcement, time-aligned fusion. `eval_fusion.py`
tests those claims and this script gives it the inputs — one file per split
holding, for every utterance, whatever each channel had to say about it.

Vectors are written **raw**, exactly as each channel produces them. Calibration
(temperature, per-class prior bias) is deliberately left to `eval_fusion.py`,
because those corrections must be fitted on dev and applied to test, and baking
them in here would make that impossible to do honestly.

Channels, each through the app's own code path rather than a reimplementation:

* **text** — `app/text_model.py`, so the MELD checkpoint, the general-domain
  ensemble and the context window all match what the sidecar serves. Scored one
  dialogue at a time in utterance order, because `predict_batch` builds each
  item's context from the preceding items in the list it is given; handing it a
  flat list of the whole split would prepend unrelated dialogues.
* **face_api** — the stock head's per-frame output from `eval-faceapi.mjs`,
  mean-pooled over the frames where a face was found. That pooling is what
  `face.ts` does before fusion sees anything.
* **face_trained** — `models/meld-face` over the same speaker crops, pooled the
  same way, so the two face variants differ only in the model.
* **voice** — `app/audio_model.py`'s neural tier. Needs a checkpoint whose head
  actually loads; see the README.

A channel is simply absent from an utterance's entry when it had nothing to say
— no face found, audio too short. `fuse()` already treats absence as distinct
from uncertainty, so the fusion evaluation inherits the real coverage pattern
rather than an idealised one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import BACKEND_ROOT, PROJECT_ROOT  # noqa: E402
from app.emotions import EMOTIONS  # noqa: E402

CSV_DIR = PROJECT_ROOT / "data" / "meld"
FACES_DIR = PROJECT_ROOT / "data" / "meld_faces"
MOSI_FACES_DIR = PROJECT_ROOT / "data" / "mosi_faces"
OUT_DIR = PROJECT_ROOT / "data" / "fusion"
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}

# CMU-MOSI transcripts are 100% uppercase with no punctuation. RoBERTa is
# case-sensitive and was trained on ordinary prose, so feeding them verbatim
# costs the text channel ~20 points of sentiment accuracy (0.683 -> 0.483,
# measured) — which would hand the comparison to face and voice for a reason
# that has nothing to do with either. Lowercasing recovers it and beat
# sentence-casing in the same measurement. Whisper output, which the app
# actually receives, is properly cased, so this normalises *toward* the
# deployment condition rather than away from it.
def normalise_transcript(text: str) -> str:
    return text.lower().strip() if text.isupper() else text.strip()


def key_of(dialogue: int, utterance: int) -> str:
    return f"{dialogue}_{utterance}"


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------


def dump_text(split: str) -> Dict[str, List[float]]:
    from app.text_model import text_model

    df = pd.read_csv(CSV_DIR / f"{split}.csv")
    out: Dict[str, List[float]] = {}
    started = time.time()

    groups = list(df.groupby("Dialogue_ID", sort=False))
    for n, (dialogue, group) in enumerate(groups, start=1):
        # Sort within the dialogue rather than trusting row order: the context
        # window is built from list position, so a shuffled group would prepend
        # the wrong prior turns and quietly cost accuracy without erroring.
        group = group.sort_values("Utterance_ID")
        texts = [str(t) for t in group["Utterance"]]
        vectors = text_model.predict_batch(texts)
        for utterance, vector in zip(group["Utterance_ID"], vectors):
            out[key_of(int(dialogue), int(utterance))] = [float(v) for v in vector]
        if n % 100 == 0:
            rate = n / max(1e-9, time.time() - started)
            print(f"    {n}/{len(groups)} dialogues ({rate:.1f}/s)")

    print(f"  text: {len(out)} utterances in {time.time() - started:.0f}s")
    return out


# --------------------------------------------------------------------------
# face
# --------------------------------------------------------------------------


def pool_frames(per_clip: Dict[str, List[np.ndarray]]) -> Dict[str, List[float]]:
    """Mean-pool a clip's frame vectors and renormalise."""
    out: Dict[str, List[float]] = {}
    for key, frames in per_clip.items():
        if not frames:
            continue
        mean = np.mean(frames, axis=0)
        total = mean.sum()
        if total > 1e-9:
            out[key] = [float(v) for v in (mean / total)]
    return out


def dump_face_api(split: str) -> Dict[str, List[float]]:
    """Re-use the stock head's per-frame output; no inference needed."""
    parts = sorted(FACES_DIR.glob(f"faceapi_{split}.part*.json"))
    merged = FACES_DIR / f"faceapi_{split}.json"
    files = parts if parts else ([merged] if merged.exists() else [])
    if not files:
        print(f"  face_api: no scored frames for {split}; skipping")
        return {}

    per_clip: Dict[str, List[np.ndarray]] = defaultdict(list)
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("emotions") != EMOTIONS:
            raise SystemExit(f"{path}: label order does not match app/emotions.py")
        for row in payload["results"]:
            if row.get("found") and row.get("probs"):
                per_clip[row["utterance_key"]].append(np.asarray(row["probs"], float))

    pooled = pool_frames(per_clip)
    print(f"  face_api: {len(pooled)} clips (from {len(files)} file(s))")
    return pooled


def dump_face_trained(split: str, model_dir: Path, batch_size: int) -> Dict[str, List[float]]:
    import torch
    from transformers import AutoModelForImageClassification

    from training.train_face import FaceFrames, normalisation, softmax

    manifest = FACES_DIR / f"{split}.csv"
    if not manifest.exists() or not (model_dir / "config.json").exists():
        print(f"  face_trained: missing manifest or model; skipping")
        return {}

    results = json.loads((model_dir / "meld_results.json").read_text(encoding="utf-8"))
    size = int(results.get("input_size", 160))
    base = results.get("base_model", "google/mobilenet_v2_1.0_224")

    df = pd.read_csv(manifest)
    df["label"] = df["emotion"].map(LABEL2ID).astype(int)
    mean, std = normalisation(base)
    dataset = FaceFrames(df, mean, std, size, train=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageClassification.from_pretrained(str(model_dir)).eval().to(device)

    per_clip: Dict[str, List[np.ndarray]] = defaultdict(list)
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            batch = torch.stack(
                [dataset[i]["pixel_values"] for i in range(start, stop)]
            ).to(device)
            logits = model(pixel_values=batch).logits.float().cpu().numpy()
            probs = softmax(logits)
            for offset, row in enumerate(range(start, stop)):
                per_clip[df["utterance_key"].iloc[row]].append(probs[offset])
            if (start // batch_size) % 50 == 0 and start:
                print(f"    {start}/{len(dataset)} crops")

    pooled = pool_frames(per_clip)
    print(f"  face_trained: {len(pooled)} clips in {time.time() - started:.0f}s")
    return pooled


# --------------------------------------------------------------------------
# voice
# --------------------------------------------------------------------------


def dump_voice(split: str, source: Path) -> Dict[str, List[float]]:
    from app.audio_io import SAMPLE_RATE
    from app.audio_model import MIN_DURATION_S, VoiceEmotionModel
    from training.eval_voice import load_audio
    from training.prepare_meld_video import index_clips

    voice = VoiceEmotionModel()
    voice._load()  # noqa: SLF001
    if voice._model is None:  # noqa: SLF001
        print(f"  voice: model did not load ({voice._load_error}); skipping")  # noqa: SLF001
        return {}

    clips = index_clips(source)
    df = pd.read_csv(CSV_DIR / f"{split}.csv")
    out: Dict[str, List[float]] = {}
    started = time.time()

    for n, row in enumerate(df.itertuples(index=False), start=1):
        dialogue, utterance = int(row.Dialogue_ID), int(row.Utterance_ID)
        clip = clips.get(f"dia{dialogue}_utt{utterance}.mp4")
        if clip is None:
            continue
        waveform = load_audio(clip)
        if waveform is None or waveform.size < MIN_DURATION_S * SAMPLE_RATE:
            continue
        vector = voice._predict_neural(waveform)  # noqa: SLF001
        if vector is not None:
            out[key_of(dialogue, utterance)] = [float(v) for v in vector]
        if n % 400 == 0:
            print(f"    {n}/{len(df)} -> {len(out)} scored")

    print(f"  voice: {len(out)} utterances in {time.time() - started:.0f}s")
    return out


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--source", type=Path, help="extracted MELD.Raw (needed for voice)")
    parser.add_argument(
        "--channels",
        default="text,face_api,face_trained,voice",
        help="comma-separated subset to dump; others are left out of the file",
    )
    parser.add_argument("--face-model", type=Path, default=BACKEND_ROOT / "models" / "meld-face")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    wanted = {c.strip() for c in args.channels.split(",") if c.strip()}
    df = pd.read_csv(CSV_DIR / f"{args.split}.csv")
    labels = {
        key_of(int(r.Dialogue_ID), int(r.Utterance_ID)): str(r.Emotion)
        for r in df.itertuples(index=False)
    }
    print(f"=== {args.split}: {len(labels)} utterances ===")

    channels: Dict[str, Dict[str, List[float]]] = {}
    if "text" in wanted:
        channels["text"] = dump_text(args.split)
    if "face_api" in wanted:
        channels["face_api"] = dump_face_api(args.split)
    if "face_trained" in wanted:
        channels["face_trained"] = dump_face_trained(
            args.split, args.face_model, args.batch_size
        )
    if "voice" in wanted:
        if args.source is None:
            print("  voice: --source not given; skipping")
        else:
            channels["voice"] = dump_voice(args.split, args.source)

    out = args.out or (OUT_DIR / f"{args.split}_channels.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Merge into one entry per utterance, with a channel simply absent when it
    # had nothing to say. Absence is meaningful here and must not become a
    # uniform vector — fuse() distinguishes the two.
    merged = {}
    for key, emotion in labels.items():
        entry = {"label": emotion}
        for name, vectors in channels.items():
            if key in vectors:
                entry[name] = [round(v, 6) for v in vectors[key]]
        merged[key] = entry

    out.write_text(
        json.dumps({"emotions": EMOTIONS, "split": args.split, "utterances": merged}),
        encoding="utf-8",
    )

    print(f"\nCoverage over {len(labels)} utterances:")
    for name in channels:
        have = sum(1 for e in merged.values() if name in e)
        print(f"  {name:<14} {have:>5} ({100 * have / len(labels):5.1f}%)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
