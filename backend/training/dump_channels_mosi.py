"""Dump every channel's per-clip vector on CMU-MOSI, aligned for fusion.

    python -m training.prepare_mosi --source <MOSI_data> --save-frames
    node scripts/eval-faceapi.mjs ...            # in frontend/, sharded
    python -m training.dump_channels_mosi --source <MOSI_data>

The MELD fusion ablation said adding face and voice makes the reading worse. The
standing objection is that MELD is a hostile test for both: faces below the
detector's size threshold, laugh tracks over the speech, and labels driven by
dialogue content. MOSI removes all three — single speaker, close to camera,
unscripted, clean audio — so this is where the objection gets settled.

Output matches `dump_channels.py` exactly, so `eval_fusion.py` reads it unchanged.
The one difference is the label: MOSI annotates **sentiment**, and the channels
still emit seven-class vectors. That is deliberate rather than a compromise —
`app/emotions.py` already defines the seven-to-three collapse and the app reports
that view, so fusion happens in seven-space exactly as it does in production and
only the *scoring* drops to three. No new mapping is invented here.
"""

from __future__ import annotations

import argparse
import csv
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
from training.dump_channels import normalise_transcript, pool_frames  # noqa: E402

MOSI_FACES_DIR = PROJECT_ROOT / "data" / "mosi_faces"
OUT_DIR = PROJECT_ROOT / "data" / "fusion"


def load_labels(source: Path) -> pd.DataFrame:
    rows = list(
        csv.DictReader(open(source / "label.csv", encoding="utf-8", errors="replace"))
    )
    df = pd.DataFrame(rows)
    df["key"] = df["video_id"].astype(str) + "_" + df["clip_id"].astype(str)
    df["sentiment"] = df["annotation"].str.strip().str.lower()
    df["clip"] = [source / "Raw" / r["video_id"] / f"{r['clip_id']}.mp4" for r in rows]
    return df


def dump_text(df: pd.DataFrame) -> Dict[str, List[float]]:
    from app.text_model import text_model

    out: Dict[str, List[float]] = {}
    started = time.time()

    # Grouped by source video and ordered by clip id, so the context window is
    # built from the same speaker's earlier clips rather than a stranger's —
    # the same reason dump_channels.py scores MELD one dialogue at a time.
    df = df.copy()
    df["_order"] = pd.to_numeric(df["clip_id"], errors="coerce").fillna(0).astype(int)
    groups = list(df.groupby("video_id", sort=False))
    for n, (_, group) in enumerate(groups, start=1):
        group = group.sort_values("_order")
        texts = [normalise_transcript(str(t)) for t in group["text"]]
        for key, vector in zip(group["key"], text_model.predict_batch(texts)):
            out[key] = [float(v) for v in vector]
        if n % 25 == 0:
            print(f"    {n}/{len(groups)} videos")

    print(f"  text: {len(out)} clips in {time.time() - started:.0f}s")
    return out


def dump_face_api() -> Dict[str, List[float]]:
    parts = sorted(MOSI_FACES_DIR.glob("faceapi_clips.part*.json"))
    if not parts:
        print("  face_api: no scored frames; skipping")
        return {}
    per_clip: Dict[str, List[np.ndarray]] = defaultdict(list)
    total = 0
    for path in parts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("emotions") != EMOTIONS:
            raise SystemExit(f"{path}: label order does not match app/emotions.py")
        for row in payload["results"]:
            total += 1
            if row.get("found") and row.get("probs"):
                per_clip[row["utterance_key"]].append(np.asarray(row["probs"], float))
    found = sum(len(v) for v in per_clip.values())
    pooled = pool_frames(per_clip)
    print(f"  face_api: {len(pooled)} clips, {found}/{total} frames detected "
          f"({100 * found / max(1, total):.1f}%)")
    return pooled


def dump_face_trained(model_dir: Path, batch_size: int) -> Dict[str, List[float]]:
    import torch
    from transformers import AutoModelForImageClassification

    from training.train_face import FaceFrames, normalisation, softmax

    manifest = MOSI_FACES_DIR / "clips.csv"
    if not manifest.exists() or not (model_dir / "config.json").exists():
        print("  face_trained: missing manifest or model; skipping")
        return {}

    results = json.loads((model_dir / "meld_results.json").read_text(encoding="utf-8"))
    size = int(results.get("input_size", 160))
    base = results.get("base_model", "google/mobilenet_v2_1.0_224")

    df = pd.read_csv(manifest)
    # FaceFrames wants a `label` column and resolves `path` against its own
    # root; MOSI crops live elsewhere, so point it at this corpus.
    df["label"] = 0
    import training.train_face as tf

    previous_root = tf.FACES_DIR
    tf.FACES_DIR = MOSI_FACES_DIR
    try:
        mean, std = normalisation(base)
        dataset = FaceFrames(df, mean, std, size, train=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = (
            AutoModelForImageClassification.from_pretrained(str(model_dir)).eval().to(device)
        )

        per_clip: Dict[str, List[np.ndarray]] = defaultdict(list)
        started = time.time()
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                stop = min(start + batch_size, len(dataset))
                batch = torch.stack(
                    [dataset[i]["pixel_values"] for i in range(start, stop)]
                ).to(device)
                probs = softmax(model(pixel_values=batch).logits.float().cpu().numpy())
                for offset, row in enumerate(range(start, stop)):
                    per_clip[df["utterance_key"].iloc[row]].append(probs[offset])
    finally:
        tf.FACES_DIR = previous_root

    pooled = pool_frames(per_clip)
    print(f"  face_trained: {len(pooled)} clips in {time.time() - started:.0f}s")
    return pooled


def dump_voice(df: pd.DataFrame) -> Dict[str, List[float]]:
    from app.audio_io import SAMPLE_RATE
    from app.audio_model import MIN_DURATION_S, VoiceEmotionModel
    from training.eval_voice import load_audio

    voice = VoiceEmotionModel()
    voice._load()  # noqa: SLF001
    if voice._model is None:  # noqa: SLF001
        print(f"  voice: did not load ({voice._load_error}); skipping")  # noqa: SLF001
        return {}

    out: Dict[str, List[float]] = {}
    started = time.time()
    for n, row in enumerate(df.itertuples(index=False), start=1):
        clip = Path(row.clip)
        if not clip.exists():
            continue
        waveform = load_audio(clip)
        if waveform is None or waveform.size < MIN_DURATION_S * SAMPLE_RATE:
            continue
        vector = voice._predict_neural(waveform)  # noqa: SLF001
        if vector is not None:
            out[row.key] = [float(v) for v in vector]
        if n % 300 == 0:
            print(f"    {n}/{len(df)} -> {len(out)} scored")

    print(f"  voice: {len(out)} clips in {time.time() - started:.0f}s")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--channels", default="text,face_api,face_trained,voice")
    parser.add_argument("--face-model", type=Path, default=BACKEND_ROOT / "models" / "meld-face")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    wanted = {c.strip() for c in args.channels.split(",") if c.strip()}
    df = load_labels(args.source)
    print(f"=== MOSI: {len(df)} clips ===")
    print("  splits:", dict(df["mode"].value_counts()))
    print("  sentiment:", dict(df["sentiment"].value_counts()))

    channels: Dict[str, Dict[str, List[float]]] = {}
    if "text" in wanted:
        channels["text"] = dump_text(df)
    if "face_api" in wanted:
        channels["face_api"] = dump_face_api()
    if "face_trained" in wanted:
        channels["face_trained"] = dump_face_trained(args.face_model, args.batch_size)
    if "voice" in wanted:
        channels["voice"] = dump_voice(df)

    # MOSI ships its own train/valid/test split; reuse it rather than inventing
    # one, so any calibration fitted on "dev" never touches the test clips.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_map = {"train": "mositrain", "valid": "mosidev", "test": "mositest"}
    for mode, name in split_map.items():
        subset = df[df["mode"] == mode]
        merged = {}
        for row in subset.itertuples(index=False):
            entry = {"label": row.sentiment}
            for channel, vectors in channels.items():
                if row.key in vectors:
                    entry[channel] = [round(v, 6) for v in vectors[row.key]]
            merged[row.key] = entry

        out = OUT_DIR / f"{name}_channels.json"
        out.write_text(
            json.dumps(
                {"emotions": EMOTIONS, "split": name, "taxonomy": "sentiment",
                 "utterances": merged}
            ),
            encoding="utf-8",
        )
        print(f"\n{name}: {len(merged)} clips")
        for channel in channels:
            have = sum(1 for e in merged.values() if channel in e)
            print(f"  {channel:<14} {have:>5} ({100 * have / max(1, len(merged)):5.1f}%)")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
