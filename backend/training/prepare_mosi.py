"""Extract speaker face crops and whole frames from CMU-MOSI.

    python -m training.prepare_mosi --source <MOSI_data> --save-frames

MELD said fusion does not help. The obvious objection is that MELD is a hostile
test for two of the three channels: wide multi-party sitcom shots where the face
falls below the detector's size threshold, laugh tracks and overlapping speech
over the audio, and utterance labels driven by dialogue the face never shows.

CMU-MOSI is the counter-test. It is ~2,200 clips of YouTube monologues — one
person, close to the camera, talking unscripted about their own opinions. That is
structurally the diary this app is built for, so if the visual and acoustic
channels are ever going to carry signal, they should carry it here.

Its labels are **sentiment**, not MELD's seven emotions, which sounds like a
mismatch and is not: `app/emotions.py` already defines the collapse from seven
emotions to three sentiments, and the app reports that view. Evaluating there
needs no new judgement call — the mapping is the project's own.

Everything below reuses `prepare_meld_video.py`: same YuNet detector, same frame
sampling, same speaker-track heuristic, same crop geometry. Only the manifest
differs, so any difference in the results is the data rather than the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import PROJECT_ROOT  # noqa: E402
from training.prepare_meld_video import (  # noqa: E402
    FaceDetector,
    MANIFEST_COLUMNS,
    _init_worker,
    _process_one,
    _process_one_serial,
    download_yunet,
)

OUT_DIR = PROJECT_ROOT / "data" / "mosi_faces"


def load_labels(source: Path) -> pd.DataFrame:
    path = source / "label.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing — point --source at the extracted MOSI download")
    rows = list(csv.DictReader(open(path, encoding="utf-8", errors="replace")))
    df = pd.DataFrame(rows)
    df["clip"] = [
        source / "Raw" / r["video_id"] / f"{r['clip_id']}.mp4" for r in rows
    ]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--trim", type=float, default=0.15)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--min-track", type=int, default=3)
    parser.add_argument("--detector", default="yunet", choices=["yunet", "haar"])
    parser.add_argument("--det-score", type=float, default=0.6)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2))
    )
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError:
        raise SystemExit("opencv-python is required")

    labels = load_labels(args.source)
    if args.limit:
        labels = labels.head(args.limit)
    missing = sum(1 for c in labels["clip"] if not Path(c).exists())
    print(f"{len(labels)} clips listed, {missing} missing on disk")

    if args.detector == "yunet":
        download_yunet()
    detector = FaceDetector(args.detector, args.det_score) if args.workers <= 1 else None
    print(f"Detector: {args.detector} | workers: {args.workers}")

    split_dir = args.output / "clips"
    split_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for row in labels.itertuples(index=False):
        clip = Path(row.clip)
        if not clip.exists():
            continue
        tasks.append(
            (
                clip,
                {
                    # `split` drives the whole-frame directory name; `emotion`
                    # carries MOSI's sentiment annotation rather than one of
                    # MELD's seven, which is what the fusion evaluation expects
                    # to score against for this corpus.
                    "split": "clips",
                    "dialogue_id": row.video_id,
                    "utterance_id": row.clip_id,
                    "emotion": str(row.annotation).strip().lower(),
                },
                args,
                split_dir,
            )
        )

    rows: List[dict] = []
    frame_rows: List[dict] = []
    misses: Dict[str, int] = {}
    started = time.time()
    reported: set[str] = set()

    def absorb(position: int, payload) -> None:
        clip_rows, reason, whole, detail = payload
        if reason:
            if detail and reason not in reported:
                reported.add(reason)
                print(f"  ! {detail}")
            misses[reason] = misses.get(reason, 0) + 1
        rows.extend(clip_rows)
        frame_rows.extend(whole)
        if position % 250 == 0:
            rate = position / max(1e-9, time.time() - started)
            print(f"  {position}/{len(tasks)} clips -> {len(rows)} crops ({rate:.1f}/s)")

    if args.workers > 1:
        import multiprocessing as mp

        with mp.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.detector, args.det_score),
        ) as pool:
            for position, payload in enumerate(
                pool.imap_unordered(_process_one, tasks, chunksize=8), start=1
            ):
                absorb(position, payload)
    else:
        for position, task in enumerate(tasks, start=1):
            absorb(position, _process_one_serial(task, detector))

    crops = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if len(crops):
        crops["utterance_key"] = (
            crops["dialogue_id"].astype(str) + "_" + crops["utterance_id"].astype(str)
        )
    crops.to_csv(args.output / "clips.csv", index=False)

    if args.save_frames and frame_rows:
        whole = pd.DataFrame(frame_rows)
        whole["utterance_key"] = (
            whole["dialogue_id"].astype(str) + "_" + whole["utterance_id"].astype(str)
        )
        whole.to_csv(args.output / "clips_frames.csv", index=False)
        print(f"  {len(whole)} whole frames -> clips_frames.csv")

    covered = crops["utterance_key"].nunique() if len(crops) else 0
    print(f"\n{len(crops)} crops from {covered}/{len(tasks)} clips "
          f"({100 * covered / max(1, len(tasks)):.1f}%)")
    if misses:
        print("  dropped:", ", ".join(f"{k}={v}" for k, v in sorted(misses.items())))

    print("\nCoverage by sentiment:")
    wanted = labels["annotation"].str.strip().str.lower().value_counts()
    have = (
        crops.groupby("emotion")["utterance_key"].nunique() if len(crops) else pd.Series(dtype=int)
    )
    for label in ("positive", "negative", "neutral"):
        total = int(wanted.get(label, 0))
        got = int(have.get(label, 0))
        pct = 100 * got / total if total else 0.0
        print(f"  {label:<9} {got:>5}/{total:<5} ({pct:5.1f}%)")

    (args.output / "prepare_summary.json").write_text(
        json.dumps(
            {
                "dataset": "cmu-mosi",
                "clips": len(tasks),
                "clips_with_faces": int(covered),
                "crops": len(crops),
                "dropped": misses,
                "frames_per_clip": args.frames,
                "crop_size": args.size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
