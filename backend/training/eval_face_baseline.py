"""Score face-api's stock expression head — the baseline the face channel has never had.

    cd frontend
    node scripts/eval-faceapi.mjs --manifest ../data/meld_faces/dev_frames.csv \
        --root ../data/meld_faces --out ../data/meld_faces/faceapi_dev.json
    node scripts/eval-faceapi.mjs --manifest ../data/meld_faces/test_frames.csv \
        --root ../data/meld_faces --out ../data/meld_faces/faceapi_test.json

    cd backend
    python -m training.eval_face_baseline

Every other channel in this project has a table. The face channel has a sentence
in the README — "reads a resting face as slightly sad" — and no measurement
behind it. This script produces the measurement, and it is worth running *before*
training a replacement, because it answers a question training cannot: whether
the stock head's problem is accuracy or calibration.

Those need different fixes, and only one of them is expensive.

- If face-api is *inaccurate*, a trained head is the answer.
- If face-api is accurate but *overconfident*, the answer is a temperature — a
  single scalar, no training, no new weights in the browser. app/fusion.py
  weights each channel by entropy, so an overconfident channel wins weight it
  has not earned, and fixing that is nearly free.

Three sections of output, in increasing order of how specific they are to this
project:

*Standard metrics* (weighted/macro F1) so the number is comparable to FER work.

*Calibration* — ECE with a temperature fitted on dev and applied to test. Fitting
on test would be an oracle result and is reported as such if dev is missing.

*The neutral-drift test*, which is the actual README claim: given a frame MELD
labels neutral, what does face-api say? A head that is merely inaccurate spreads
its errors around. A head with the bias described in the README puts them
somewhere specific.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import BACKEND_ROOT, PROJECT_ROOT  # noqa: E402
from app.emotions import EMOTIONS  # noqa: E402
from training.train_face import (  # noqa: E402
    expected_calibration_error,
    fit_temperature,
    pool_by_clip,
    score,
    softmax,
)

FACES_DIR = PROJECT_ROOT / "data" / "meld_faces"
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}


def have_baseline(path: Path) -> bool:
    """True if a merged file OR its shards exist.

    Checking only `path.exists()` silently downgraded a sharded dev run to "no
    dev file", which swapped the honest dev-fitted temperature for an oracle
    fitted on test — a result that looks the same but means much less.
    """
    return path.exists() or bool(list(path.parent.glob(f"{path.stem}.part*.json")))


def load_baseline(path: Path) -> Tuple[pd.DataFrame, dict]:
    """Read the Node script's output into a frame of *detected* frames only.

    Frames where face-api found nothing are dropped from the scoring set and
    counted separately. That mirrors fusion.py, which treats "no face" as a
    distinct state from "the face looked neutral" — scoring a miss as a neutral
    prediction would quietly credit the head for the frames it gave up on.
    """
    # A sharded run writes <stem>.part0.json, .part1.json, ... Merge them back
    # into one payload so everything downstream is unaware sharding happened.
    parts = sorted(path.parent.glob(f"{path.stem}.part*.json"))

    if not path.exists() and not parts:
        raise SystemExit(
            f"{path} missing. Run the Node scorer first:\n"
            f"  cd frontend && node scripts/eval-faceapi.mjs "
            f"--manifest ../data/meld_faces/{path.stem.replace('faceapi_', '')}_frames.csv "
            f"--root ../data/meld_faces --out ../{path.relative_to(PROJECT_ROOT)}"
        )

    if parts and not path.exists():
        merged = None
        for part in parts:
            chunk = json.loads(part.read_text(encoding="utf-8"))
            if merged is None:
                merged = chunk
            else:
                merged["results"].extend(chunk["results"])
                merged["frames"] += chunk["frames"]
                merged["frames_with_face"] += chunk["frames_with_face"]
        print(f"  merged {len(parts)} shards -> {len(merged['results'])} frames")
        payload = merged
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("emotions") != EMOTIONS:
        raise SystemExit(
            f"{path}: label order {payload.get('emotions')} does not match "
            f"app/emotions.py {EMOTIONS}. The vectors are indexed positionally, "
            "so this would silently score the wrong classes."
        )

    rows = payload["results"]
    detected = [r for r in rows if r.get("found") and r.get("probs")]
    df = pd.DataFrame(
        {
            "utterance_key": [r["utterance_key"] for r in detected],
            "emotion": [r["emotion"] for r in detected],
            "label": [LABEL2ID[r["emotion"]] for r in detected],
            "det_score": [r["det_score"] for r in detected],
        }
    )
    probs = np.array([r["probs"] for r in detected], dtype=np.float64)
    df = df.reset_index(drop=True)
    df["_row"] = np.arange(len(df))

    payload["_probs"] = probs
    payload["_total"] = len(rows)
    payload["_errors"] = sum(1 for r in rows if r.get("error"))
    return df, payload


def coverage_table(df: pd.DataFrame, payload: dict, name: str) -> dict:
    """Detection coverage per class — a bias that exists before any scoring.

    face-api not finding a face is not a neutral outcome for this app: the
    channel simply goes silent, and it goes silent more often for some emotions
    than others.
    """
    total = payload["_total"]
    found = len(df)
    print(f"\n{name}: face detected in {found}/{total} frames ({100 * found / total:.1f}%)")
    if payload["_errors"]:
        print(f"  {payload['_errors']} frames errored during decode/inference")

    counts = df["emotion"].value_counts()
    stats = {}
    for emotion in EMOTIONS:
        n = int(counts.get(emotion, 0))
        stats[emotion] = n
        share = 100 * n / max(1, found)
        print(f"    {emotion:<9} {n:>6} detected frames ({share:4.1f}% of detections)")
    return {"frames": total, "detected": found, "per_class": stats}


def logits_from_probs(probs: np.ndarray) -> np.ndarray:
    """Recover logits so a temperature can be applied.

    face-api hands back a softmax, not logits. log(p) differs from the true
    logits by a per-row constant, and softmax is invariant to exactly that
    constant — so temperature scaling on log(p) is identical to temperature
    scaling on the original logits. Nothing is being approximated here.
    """
    return np.log(np.clip(probs, 1e-12, None))


def neutral_drift(probs: np.ndarray, df: pd.DataFrame) -> dict:
    """Test the README's actual claim, rather than a proxy for it.

    "Reads a resting face as slightly sad" is a statement about the conditional
    distribution P(prediction | truth = neutral), not about accuracy. Two numbers
    settle it: where the mass goes on neutral frames, and how often the head
    prefers sadness to neutral on them.
    """
    mask = df["label"].to_numpy() == LABEL2ID["neutral"]
    if not mask.any():
        return {}

    mean = probs[mask].mean(axis=0)
    preds = probs[mask].argmax(axis=1)

    print(f"\nNeutral-drift test  ({int(mask.sum())} frames MELD labels neutral)")
    print("  mean probability assigned, and how often each class wins:")
    order = np.argsort(-mean)
    for i in order:
        won = float((preds == i).mean())
        flag = "  <- drift" if EMOTIONS[i] != "neutral" and i == order[0] else ""
        print(f"    {EMOTIONS[i]:<9} p={mean[i]:.3f}   argmax {100 * won:5.1f}%{flag}")

    sad_over_neutral = float(
        (probs[mask][:, LABEL2ID["sadness"]] > probs[mask][:, LABEL2ID["neutral"]]).mean()
    )
    print(
        f"  sadness outranks neutral on {100 * sad_over_neutral:.1f}% of true-neutral frames"
    )
    verdict = (
        "confirmed" if EMOTIONS[order[0]] != "neutral" or sad_over_neutral > 0.25 else "not seen"
    )
    print(f"  README's 'resting face reads slightly sad': {verdict}")

    return {
        "n_neutral_frames": int(mask.sum()),
        "mean_probability": {e: round(float(mean[i]), 4) for i, e in enumerate(EMOTIONS)},
        "argmax_share": {
            e: round(float((preds == i).mean()), 4) for i, e in enumerate(EMOTIONS)
        },
        "sadness_over_neutral_rate": round(sad_over_neutral, 4),
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=FACES_DIR / "faceapi_test.json")
    parser.add_argument(
        "--dev",
        type=Path,
        default=FACES_DIR / "faceapi_dev.json",
        help="used only to fit the temperature; without it the fit is an oracle",
    )
    parser.add_argument("--out", type=Path, default=FACES_DIR / "faceapi_baseline.json")
    args = parser.parse_args()

    from sklearn.metrics import classification_report, confusion_matrix

    test_df, test_payload = load_baseline(args.test)
    test_probs = test_payload["_probs"]
    test_labels = test_df["label"].to_numpy()

    print("=" * 68)
    print("face-api stock expression head — MELD test")
    print("=" * 68)
    print(f"  {test_payload['source']}")
    print(f"  backend={test_payload['backend']}  "
          f"input_size={test_payload['input_size']}  "
          f"score_threshold={test_payload['score_threshold']}")
    print(f"  mean inference {test_payload.get('mean_inference_ms', 0):.1f} ms/frame")

    coverage = coverage_table(test_df, test_payload, "Coverage")

    # --- temperature, fitted on dev when we have it ------------------------
    oracle = False
    if have_baseline(args.dev):
        dev_df, dev_payload = load_baseline(args.dev)
        temperature = fit_temperature(
            logits_from_probs(dev_payload["_probs"]), dev_df["label"].to_numpy()
        )
        print(f"\nTemperature fitted on dev ({len(dev_df)} frames): T = {temperature:.3f}")
    else:
        oracle = True
        temperature = fit_temperature(logits_from_probs(test_probs), test_labels)
        print(f"\nNo dev file — temperature fitted on test: T = {temperature:.3f}")
        print("  ! This is an ORACLE number. It is an upper bound on what calibration")
        print("  ! could buy, not a result. Run the Node scorer on dev_frames.csv.")

    calibrated = softmax(logits_from_probs(test_probs) / temperature)

    print("\n--- Metrics ---")
    results: Dict[str, dict] = {}
    results["frame_raw"] = score(test_probs, test_labels, "frame-level (as shipped)")
    results["frame_calibrated"] = score(calibrated, test_labels, "frame-level (T-scaled)")

    clip_raw, clip_labels = pool_by_clip(test_probs, test_df)
    clip_cal, _ = pool_by_clip(calibrated, test_df)
    results["clip_raw"] = score(clip_raw, clip_labels, "clip-level (as shipped)")
    results["clip_calibrated"] = score(clip_cal, clip_labels, "clip-level (T-scaled)")

    counts = np.bincount(test_labels, minlength=len(EMOTIONS)).astype(np.float64)
    prior = np.tile(counts / counts.sum(), (len(clip_labels), 1))
    results["prior"] = score(prior, clip_labels, "prior baseline (clip)")

    print("\nPer-class breakdown, clip-level:")
    report_text = classification_report(
        clip_labels,
        clip_raw.argmax(axis=1),
        labels=list(range(len(EMOTIONS))),
        target_names=EMOTIONS,
        digits=3,
        zero_division=0,
    )
    print(report_text)

    print("Confusion matrix, clip-level (rows = MELD truth, cols = face-api):")
    matrix = confusion_matrix(
        clip_labels, clip_raw.argmax(axis=1), labels=list(range(len(EMOTIONS)))
    )
    print(f"    {'':<9}" + "".join(f"{e[:6]:>8}" for e in EMOTIONS))
    for i, emotion in enumerate(EMOTIONS):
        print(f"    {emotion:<9}" + "".join(f"{v:>8}" for v in matrix[i]))

    drift = neutral_drift(test_probs, test_df)

    # --- what this means for the next step ---------------------------------
    print("\n--- Read-out ---")
    gain = results["frame_raw"]["ece"] - results["frame_calibrated"]["ece"]
    if gain > 0.02:
        print(f"  Calibration alone removes {gain:.3f} of ECE"
              f"{' (oracle)' if oracle else ''}. That is a real improvement to the")
        print("  fusion weighting for the cost of one scalar — worth taking whether or")
        print("  not a trained head follows.")
    else:
        print(f"  Calibration buys little ({gain:+.3f} ECE). The stock head's problem,")
        print("  if it has one, is accuracy rather than confidence.")

    if results["clip_raw"]["macro_f1"] < results["prior"]["macro_f1"] + 0.05:
        print("  Clip-level macro-F1 barely beats predicting the class prior. On this")
        print("  data the stock head carries little signal, which is the strongest")
        print("  case available for training a replacement.")

    args.out.write_text(
        json.dumps(
            {
                "source": test_payload["source"],
                "input_size": test_payload["input_size"],
                "score_threshold": test_payload["score_threshold"],
                "temperature": temperature,
                "temperature_is_oracle": oracle,
                "coverage": coverage,
                "metrics": results,
                "classification_report": report_text,
                "confusion_matrix": matrix.tolist(),
                "neutral_drift": drift,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
