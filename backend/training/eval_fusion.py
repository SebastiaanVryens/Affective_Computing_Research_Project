"""Does fusing the channels actually beat the best one on its own?

    python -m training.dump_channels --split dev  --source <MELD_data>
    python -m training.dump_channels --split test --source <MELD_data>
    python -m training.eval_fusion

This is the one claim the project rests on and has never tested. The README
argues for certainty-weighted late fusion, for cross-modal reinforcement, and for
aligning readings in time — all of it reasoned from first principles, none of it
measured. Every channel now has a number; the combination does not.

What it reports, in the order the questions actually matter:

*Single channels, as a floor.* Fusion has to beat the best of them. If it does
not, no amount of argument about log-opinion pools rescues it, and the honest
move is to serve the best channel alone.

*Every subset.* Which channels help, which are dead weight. A channel that lowers
the fused score is worth knowing about — `fusion.py` cannot notice that on its
own, because it weights by confidence, not by whether a channel has ever been
right.

*The independence knob, swept.* `MINDSCAPE_INDEPENDENCE=0` is plain weighted
averaging; 1 treats the channels as independent sensors and multiplies. The
default of 0.5 was chosen by argument. This says whether it was a good argument.

*The weights, swept on dev and quoted once on test.* 0.5 / 0.3 / 0.2 predate every
measurement in this repo, and we now know the face channel is roughly half as
accurate as text.

Calibration is fitted here rather than in the dump, on dev, and applied to test —
a per-class prior bias for the voice channel especially, which otherwise cannot
say "neutral" at all and drags any fusion it touches.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import PROJECT_ROOT  # noqa: E402
from app.emotions import EMOTIONS, SENTIMENT_OF  # noqa: E402
from app.fusion import fuse  # noqa: E402
from training.train_face import (  # noqa: E402
    fit_prior_correction,
    fit_temperature,
    score,
    softmax,
)

FUSION_DIR = PROJECT_ROOT / "data" / "fusion"
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}

# Sentiment taxonomy, for corpora annotated that way (CMU-MOSI).
#
# Channels still emit seven-class vectors and fusion still happens in
# seven-space, exactly as in production — only the scoring collapses to three,
# using app/emotions.py's own SENTIMENT_OF table. Nothing is remapped by hand.
SENTIMENTS = ["neutral", "positive", "negative"]
SENTIMENT2ID = {s: i for i, s in enumerate(SENTIMENTS)}
# Column-sum matrix: a 7-vector times this gives the 3-vector.
COLLAPSE = np.zeros((len(EMOTIONS), len(SENTIMENTS)))
for _i, _e in enumerate(EMOTIONS):
    COLLAPSE[_i, SENTIMENT2ID[SENTIMENT_OF[_e]]] = 1.0


def to_sentiment(matrix: np.ndarray) -> np.ndarray:
    """Collapse 7-class probability rows into 3-class sentiment rows."""
    out = matrix @ COLLAPSE
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def load_split(split: str, face_variant: str):
    path = FUSION_DIR / f"{split}_channels.json"
    if not path.exists():
        raise SystemExit(
            f"{path} missing. Run: python -m training.dump_channels --split {split} "
            "--source <MELD_data>"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("emotions") != EMOTIONS:
        raise SystemExit(f"{path}: label order does not match app/emotions.py")

    keys = sorted(payload["utterances"])
    taxonomy = payload.get("taxonomy", "emotion")
    table = SENTIMENT2ID if taxonomy == "sentiment" else LABEL2ID
    labels = np.array([table[payload["utterances"][k]["label"]] for k in keys])

    channels: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    for name, source in (("text", "text"), ("face", face_variant), ("voice", "voice")):
        per_key: Dict[str, Optional[np.ndarray]] = {}
        for k in keys:
            vector = payload["utterances"][k].get(source)
            per_key[k] = np.asarray(vector, dtype=np.float64) if vector else None
        channels[name] = per_key
    return keys, labels, channels, taxonomy


def stack(channel: Dict[str, Optional[np.ndarray]], keys: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (matrix over present keys, boolean mask aligned to keys)."""
    mask = np.array([channel[k] is not None for k in keys])
    present = np.array([channel[k] for k in keys if channel[k] is not None])
    return present, mask


def calibrate(
    dev_channel: Dict[str, Optional[np.ndarray]],
    dev_keys: Sequence[str],
    dev_labels: np.ndarray,
    collapse: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """Fit temperature and per-class bias for one channel on dev.

    Both, because they fix different faults: temperature corrects over- or
    under-confidence, the bias corrects a wrong class prior. The voice head needs
    the second badly (it assigns ~0.01 of its mass to neutral while MELD is 48%
    neutral) and the face head needs the first.

    Fitted jointly rather than in sequence: fitting a temperature and a bias
    independently on the same logits and then applying both double-corrects, and
    on this data collapses the model to always-neutral.

    ``collapse`` is supplied for sentiment-annotated corpora. The parameters stay
    seven-dimensional, because fusion still runs in seven-space, but the
    likelihood being optimised is that of the *collapsed* three-class
    distribution — which is what the labels actually describe. Fitting
    seven-class biases against three-class label indices would silently score the
    wrong columns.
    """
    present, mask = stack(dev_channel, dev_keys)
    if not len(present):
        return 1.0, np.zeros(len(EMOTIONS))
    labels = dev_labels[mask]
    logits = np.log(np.clip(present, 1e-12, None))

    from scipy.optimize import minimize

    index = np.arange(len(labels))

    def nll(params: np.ndarray) -> float:
        temperature = np.exp(params[0])  # keep it positive
        probs = softmax(logits / temperature + params[1:])
        if collapse is not None:
            probs = probs @ collapse
            probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        return float(-np.log(np.clip(probs[index, labels], 1e-12, None)).mean())

    start = np.zeros(1 + len(EMOTIONS))
    result = minimize(nll, start, method="L-BFGS-B")
    temperature = float(np.exp(result.x[0]))
    bias = result.x[1:] - result.x[1:].mean()
    return temperature, bias


def apply_calibration(
    channel: Dict[str, Optional[np.ndarray]], temperature: float, bias: np.ndarray
) -> Dict[str, Optional[np.ndarray]]:
    out: Dict[str, Optional[np.ndarray]] = {}
    for key, vector in channel.items():
        if vector is None:
            out[key] = None
            continue
        logits = np.log(np.clip(vector, 1e-12, None))
        out[key] = softmax((logits / temperature + bias)[None])[0]
    return out


def fuse_all(
    keys: Sequence[str],
    channels: Dict[str, Dict[str, Optional[np.ndarray]]],
    use: Sequence[str],
    weights: Dict[str, float],
    independence: float,
) -> np.ndarray:
    """Run app/fusion.fuse over every utterance and return the fused matrix."""
    from app.config import settings

    previous = settings.fusion_independence
    settings.fusion_independence = independence
    try:
        out = np.zeros((len(keys), len(EMOTIONS)))
        for i, key in enumerate(keys):
            kwargs = {}
            for name in ("text", "face", "voice"):
                vector = channels[name][key] if name in use else None
                kwargs[f"{name}_vector"] = list(vector) if vector is not None else None
                kwargs[f"{name}_source"] = name if vector is not None else None
            result = fuse(weights=weights, **kwargs)
            out[i] = result.vector
        return out
    finally:
        settings.fusion_independence = previous


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--face",
        default="face_api",
        choices=["face_api", "face_trained"],
        help="which face channel to fuse; face_api is what the app ships",
    )
    parser.add_argument("--no-calibrate", action="store_true", help="ablation: raw vectors")
    parser.add_argument("--dev-split", default="dev", help="e.g. mosidev")
    parser.add_argument("--test-split", default="test", help="e.g. mositest")
    parser.add_argument("--out", type=Path, default=FUSION_DIR / "fusion_results.json")
    args = parser.parse_args()

    dev_keys, dev_labels, dev_channels, taxonomy = load_split(args.dev_split, args.face)
    test_keys, test_labels, test_channels, test_taxonomy = load_split(args.test_split, args.face)
    if taxonomy != test_taxonomy:
        raise SystemExit(f"dev is {taxonomy!r} but test is {test_taxonomy!r}")
    classes = SENTIMENTS if taxonomy == "sentiment" else EMOTIONS
    collapse = COLLAPSE if taxonomy == "sentiment" else None

    def report(matrix: np.ndarray, labels: np.ndarray, title: str) -> dict:
        """Score in the corpus's own label space.

        Fusion always happens over seven emotions, exactly as in production; only
        the scoring drops to sentiment, via app/emotions.py's own table.
        """
        return score(to_sentiment(matrix) if collapse is not None else matrix, labels, title)
    print(f"dev {len(dev_keys)} / test {len(test_keys)} utterances | face = {args.face} | taxonomy = {taxonomy}")

    print("\nCoverage (test):")
    for name in ("text", "face", "voice"):
        have = sum(1 for k in test_keys if test_channels[name][k] is not None)
        print(f"  {name:<6} {have:>5} ({100 * have / len(test_keys):5.1f}%)")

    # ---- calibration, fitted on dev -------------------------------------
    calibration: Dict[str, dict] = {}
    if not args.no_calibrate:
        print("\nCalibration fitted on dev (temperature, then per-class bias):")
        for name in ("text", "face", "voice"):
            temperature, bias = calibrate(dev_channels[name], dev_keys, dev_labels, collapse)
            calibration[name] = {"temperature": temperature, "bias": bias.tolist()}
            dev_channels[name] = apply_calibration(dev_channels[name], temperature, bias)
            test_channels[name] = apply_calibration(test_channels[name], temperature, bias)
            top = EMOTIONS[int(np.argmax(bias))]  # bias is always 7-dim
            print(f"  {name:<6} T={temperature:5.2f}  largest bias {top} {bias.max():+.2f}")

    results: Dict[str, dict] = {"face_variant": args.face, "taxonomy": taxonomy,
                                "calibration": calibration}

    # ---- single channels, the floor fusion must clear --------------------
    print("\n--- Single channels (test) ---")
    singles: Dict[str, float] = {}
    for name in ("text", "face", "voice"):
        present, mask = stack(test_channels[name], test_keys)
        if not len(present):
            continue
        # Scored only where the channel spoke, so this is its quality, not its
        # coverage. The subset comparison below is where coverage starts to count.
        m = report(present, test_labels[mask], f"{name} alone (n={mask.sum()})")
        singles[name] = m["weighted_f1"]
        results.setdefault("singles", {})[name] = m

    counts = np.bincount(test_labels, minlength=len(classes)).astype(float)
    prior = np.tile(counts / counts.sum(), (len(test_labels), 1))
    results["prior"] = score(prior, test_labels, "class prior (all utterances)")

    # ---- subsets, at the shipped weights ---------------------------------
    base = {"text": 0.5, "face": 0.3, "voice": 0.2}
    print(f"\n--- Channel subsets, shipped weights {base}, independence 0.5 ---")
    subsets = []
    for size in (1, 2, 3):
        subsets.extend(itertools.combinations(("text", "face", "voice"), size))
    for use in subsets:
        fused = fuse_all(test_keys, test_channels, use, base, 0.5)
        m = report(fused, test_labels, "+".join(use))
        results.setdefault("subsets", {})["+".join(use)] = m

    # ---- independence sweep ----------------------------------------------
    print("\n--- Independence sweep (all three, shipped weights) ---")
    for independence in (0.0, 0.25, 0.5, 0.75, 1.0):
        fused = fuse_all(test_keys, test_channels, ("text", "face", "voice"), base, independence)
        m = report(fused, test_labels, f"independence = {independence}")
        results.setdefault("independence", {})[str(independence)] = m

    # ---- weight sweep, chosen on dev, quoted on test ---------------------
    print("\n--- Weight sweep (chosen on dev) ---")
    grid = [w / 10 for w in range(0, 11)]
    best = None
    for wt in grid:
        for wf in grid:
            wv = round(1.0 - wt - wf, 3)
            if wv < -1e-9 or wv > 1:
                continue
            weights = {"text": wt, "face": wf, "voice": wv}
            fused = fuse_all(dev_keys, dev_channels, ("text", "face", "voice"), weights, 0.5)
            preds = (to_sentiment(fused) if collapse is not None else fused).argmax(axis=1)
            from sklearn.metrics import f1_score

            f1 = f1_score(dev_labels, preds, average="weighted", zero_division=0)
            if best is None or f1 > best[0]:
                best = (f1, weights)

    dev_f1, best_weights = best
    print(f"  best on dev: {best_weights}  (dev wF1 {dev_f1:.4f})")
    fused = fuse_all(test_keys, test_channels, ("text", "face", "voice"), best_weights, 0.5)
    results["tuned"] = report(fused, test_labels, "tuned weights (test)")
    results["tuned_weights"] = best_weights
    results["tuned_dev_wf1"] = float(dev_f1)

    # ---- read-out ---------------------------------------------------------
    print("\n--- Read-out ---")
    best_single = max(singles, key=singles.get) if singles else None
    all_three = results["subsets"].get("text+face+voice", {}).get("weighted_f1", 0.0)
    if best_single:
        gap = all_three - singles[best_single]
        print(f"  best single channel: {best_single} at {singles[best_single]:.4f}")
        print(f"  all three fused:     {all_three:.4f}  ({gap:+.4f})")
        if gap <= 0:
            print("  Fusion does not beat the best channel alone on this data. That is")
            print("  a result, not a bug — report it rather than tuning until it flips.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
