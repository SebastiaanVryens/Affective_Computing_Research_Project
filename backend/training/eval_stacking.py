"""Learned late fusion — can a stacker extract what weighted averaging cannot?

    python -m training.eval_stacking --train mositrain --dev mosidev --test mositest

`eval_fusion.py` found that averaging the channels never beats text alone: the
oracle weighting, tuned directly on test, buys +0.002. The obvious reading is
that the auxiliary channels have nothing to add. That reading is wrong, and the
diagnostic that shows it is this:

    text is wrong on 22.7% of MOSI test clips
    on those clips, face or voice is right 49.4% of the time
    oracle per-clip channel choice: 0.885 accuracy vs text's 0.773

So the information *is* there — +11 points of it. What fails is the *rule*.
A fixed weight vector blends every channel on every clip, which means a channel
that is right one time in three still drags the other two thirds of the time.
Extracting the signal needs a rule that decides *when* to listen, and a linear
blend cannot express that no matter how it is weighted.

This is still late fusion — the channels stay independent models and only their
outputs are combined — but the combiner is learned rather than asserted. It is
also the first thing a reviewer will ask for before accepting "fusion does not
help", so the negative result is not safe to report without it.

Features per utterance: each channel's seven probabilities, its certainty
(``1 - normalised entropy``, the same quantity fusion.py weights by), and an
availability flag, so the stacker can learn to ignore a channel that is missing
rather than being handed a uniform vector it might read as a real opinion.

Trained on train, every choice made on dev, test quoted once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import PROJECT_ROOT  # noqa: E402
from app.emotions import EMOTIONS  # noqa: E402
from training.eval_fusion import (  # noqa: E402
    COLLAPSE,
    SENTIMENTS,
    apply_calibration,
    calibrate,
    load_split,
    score,
    to_sentiment,
)

FUSION_DIR = PROJECT_ROOT / "data" / "fusion"
CHANNELS = ("text", "face", "voice")


def certainty(vector: np.ndarray) -> float:
    """1 - normalised Shannon entropy — fusion.py's own confidence measure."""
    p = np.clip(vector, 1e-12, None)
    entropy = float(-(p * np.log(p)).sum())
    return 1.0 - entropy / float(np.log(len(vector)))


def featurise(
    keys: Sequence[str], channels: Dict[str, Dict[str, Optional[np.ndarray]]]
) -> np.ndarray:
    rows = []
    for key in keys:
        row: List[float] = []
        for name in CHANNELS:
            vector = channels[name][key]
            if vector is None:
                # Uniform *and* flagged absent. Without the flag the stacker
                # cannot tell "no opinion" from "genuinely undecided", which is
                # the distinction fusion.py exists to preserve.
                row.extend([1.0 / len(EMOTIONS)] * len(EMOTIONS))
                row.extend([0.0, 0.0])
            else:
                row.extend(vector.tolist())
                row.extend([certainty(vector), 1.0])
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="mositrain")
    parser.add_argument("--dev", default="mosidev")
    parser.add_argument("--test", default="mositest")
    parser.add_argument("--face", default="face_api", choices=["face_api", "face_trained"])
    parser.add_argument("--out", type=Path, default=FUSION_DIR / "stacking_results.json")
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    tr_keys, tr_y, tr_ch, taxonomy = load_split(args.train, args.face)
    dv_keys, dv_y, dv_ch, _ = load_split(args.dev, args.face)
    te_keys, te_y, te_ch, _ = load_split(args.test, args.face)
    collapse = COLLAPSE if taxonomy == "sentiment" else None
    classes = SENTIMENTS if taxonomy == "sentiment" else EMOTIONS
    print(f"train {len(tr_keys)} / dev {len(dv_keys)} / test {len(te_keys)} | {taxonomy}")

    # Same calibration as eval_fusion, fitted on dev, so the comparison against
    # weighted averaging differs only in the combiner.
    for name in CHANNELS:
        temperature, bias = calibrate(dv_ch[name], dv_keys, dv_y, collapse)
        for split in (tr_ch, dv_ch, te_ch):
            split[name] = apply_calibration(split[name], temperature, bias)

    def collapsed(matrix: np.ndarray) -> np.ndarray:
        return to_sentiment(matrix) if collapse is not None else matrix

    # --- baseline: text alone ------------------------------------------------
    text_test = np.array(
        [te_ch["text"][k] if te_ch["text"][k] is not None else np.ones(7) / 7 for k in te_keys]
    )
    print("\n--- Baseline ---")
    baseline = score(collapsed(text_test), te_y, "text alone (test)")

    # --- stacker -------------------------------------------------------------
    X_tr, X_dv, X_te = (featurise(k, c) for k, c in
                        ((tr_keys, tr_ch), (dv_keys, dv_ch), (te_keys, te_ch)))
    print(f"\nFeatures: {X_tr.shape[1]} per utterance "
          f"({len(CHANNELS)} channels x ({len(EMOTIONS)} probs + certainty + present))")

    print("\n--- Stacker, regularisation chosen on dev ---")
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        model = LogisticRegression(C=C, max_iter=4000, multi_class="multinomial")
        model.fit(X_tr, tr_y)
        f1 = f1_score(dv_y, model.predict(X_dv), average="weighted", zero_division=0)
        print(f"  C={C:<6} dev wF1 {f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, C, model)

    dev_f1, C, model = best
    print(f"  chosen C={C} (dev wF1 {dev_f1:.4f})")

    print("\n--- Test ---")
    probs = model.predict_proba(X_te)
    stacked = score(probs, te_y, f"stacker (C={C})")

    # Text-only stacker: isolates how much of any gain comes from the auxiliary
    # channels rather than from simply re-fitting a classifier on text's output.
    text_cols = list(range(0, len(EMOTIONS) + 2))
    solo = LogisticRegression(C=C, max_iter=4000, multi_class="multinomial")
    solo.fit(X_tr[:, text_cols], tr_y)
    text_stacked = score(solo.predict_proba(X_te[:, text_cols]), te_y, "stacker, text features only")

    print("\n--- Read-out ---")
    gain = stacked["weighted_f1"] - baseline["weighted_f1"]
    attributable = stacked["weighted_f1"] - text_stacked["weighted_f1"]
    print(f"  text alone            {baseline['weighted_f1']:.4f}")
    print(f"  stacker (all)         {stacked['weighted_f1']:.4f}  ({gain:+.4f})")
    print(f"  stacker (text only)   {text_stacked['weighted_f1']:.4f}")
    print(f"  attributable to face+voice: {attributable:+.4f}")
    if gain <= 0:
        print("  Even a learned combiner does not beat the best channel. The")
        print("  complementary information is real but not linearly recoverable.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "taxonomy": taxonomy,
                "face_variant": args.face,
                "C": C,
                "dev_wf1": float(dev_f1),
                "baseline_text": baseline,
                "stacker": stacked,
                "stacker_text_only": text_stacked,
                "gain_vs_text": float(gain),
                "attributable_to_aux": float(attributable),
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
