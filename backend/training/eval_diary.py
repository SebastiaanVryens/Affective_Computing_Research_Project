"""Compare the MELD head, the general head, and their ensemble on diary text.

    python -m training.eval_diary
    python -m training.eval_diary --meld-weight 0.35 --json results.json

Reports, per condition:

* accuracy and macro-F1 against the probe set's intended labels
* a confusion matrix
* **calibration** — mean confidence when right vs when wrong

That last one is the number most worth reporting, and the one a plain accuracy
comparison hides. The app's fusion layer weights each modality by its certainty
(see app/fusion.py), so a model that is confidently wrong does more damage than
one that is accurately unsure. A model can lose on accuracy and still be the
better choice here if its confidence tracks its correctness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402
from app.emotions import EMOTIONS, from_label_scores, normalize  # noqa: E402
from training.diary_probe import PROBE, counts  # noqa: E402

# The general model uses different names for three of the seven classes.
LABEL_ALIASES = {"happy": "joy", "sad": "sadness", "angry": "anger"}


def normalise_scores(raw: Sequence[dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in raw:
        label = item["label"].lower().strip()
        out[LABEL_ALIASES.get(label, label)] = float(item["score"])
    return out


def predict_all(pipe, texts: List[str]) -> List[List[float]]:
    raw = pipe(texts, truncation=True, max_length=256)
    return [from_label_scores(normalise_scores(scores)) for scores in raw]


def blend(a: List[List[float]], b: List[List[float]], w: float) -> List[List[float]]:
    return [
        normalize([x * w + y * (1 - w) for x, y in zip(va, vb)])
        for va, vb in zip(a, b)
    ]


def evaluate(name: str, vectors: List[List[float]], gold: List[str]) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score

    predicted = [EMOTIONS[int(np.argmax(v))] for v in vectors]
    confidence = [float(np.max(v)) for v in vectors]
    correct = [p == g for p, g in zip(predicted, gold)]

    accuracy = float(np.mean(correct))
    macro_f1 = f1_score(gold, predicted, labels=EMOTIONS, average="macro", zero_division=0)

    conf_right = [c for c, ok in zip(confidence, correct) if ok]
    conf_wrong = [c for c, ok in zip(confidence, correct) if not ok]

    print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
    print(f"  accuracy   {accuracy:.3f}   ({sum(correct)}/{len(gold)})")
    print(f"  macro-F1   {macro_f1:.3f}")
    print(f"  mean confidence   correct: {np.mean(conf_right) if conf_right else 0:.3f}"
          f"   wrong: {np.mean(conf_wrong) if conf_wrong else 0:.3f}")

    # Positive separation means confidence is informative: the model is more
    # sure when it is right. Near zero or negative means its confidence is
    # noise, and the fusion layer will be misled by it.
    separation = (np.mean(conf_right) if conf_right else 0) - (
        np.mean(conf_wrong) if conf_wrong else 0
    )
    print(f"  calibration separation: {separation:+.3f}"
          f"   {'(confidence is informative)' if separation > 0.05 else '(confidence is NOT informative)'}")

    print("\n  per-class recall:")
    for emotion in EMOTIONS:
        idx = [i for i, g in enumerate(gold) if g == emotion]
        if not idx:
            continue
        hits = sum(correct[i] for i in idx)
        print(f"    {emotion:<9} {hits}/{len(idx)}")

    print("\n  confusion (rows = intended, cols = predicted):")
    matrix = confusion_matrix(gold, predicted, labels=EMOTIONS)
    header = "".join(f"{e[:4]:>6}" for e in EMOTIONS)
    print(f"    {'':<9}{header}")
    for emotion, row in zip(EMOTIONS, matrix):
        cells = "".join(f"{v:>6}" if v else f"{'.':>6}" for v in row)
        print(f"    {emotion:<9}{cells}")

    return {
        "name": name,
        "accuracy": accuracy,
        "macro_f1": float(macro_f1),
        "mean_confidence_correct": float(np.mean(conf_right)) if conf_right else 0.0,
        "mean_confidence_wrong": float(np.mean(conf_wrong)) if conf_wrong else 0.0,
        "calibration_separation": float(separation),
        "predictions": predicted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meld-weight",
        type=float,
        default=settings.ensemble_meld_weight,
        help="ensemble weight on the MELD head (default: from config)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="try ensemble weights 0.0-1.0 and report the curve",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="MELD checkpoint to evaluate (default: the one the app serves)",
    )
    parser.add_argument("--json", type=Path, help="write full results here")
    args = parser.parse_args()

    checkpoint = args.checkpoint or settings.meld_text_model
    if not (checkpoint / "config.json").exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint}. Run: python -m training.train_text"
        )

    from transformers import pipeline

    texts = [t for t, _ in PROBE]
    gold = [g for _, g in PROBE]

    print(f"Diary probe set: {len(PROBE)} sentences")
    print("Class balance:", counts())
    print(f"\nMELD checkpoint : {settings.meld_text_model}")
    print(f"General model   : {settings.fallback_text_model}")

    device = 0 if settings.resolved_device() == "cuda" else -1
    meld_pipe = pipeline(
        "text-classification", model=str(checkpoint), top_k=None, device=device
    )
    general_pipe = pipeline(
        "text-classification", model=settings.fallback_text_model, top_k=None, device=device
    )

    meld_vectors = predict_all(meld_pipe, texts)
    general_vectors = predict_all(general_pipe, texts)
    ensemble_vectors = blend(meld_vectors, general_vectors, args.meld_weight)

    results = [
        evaluate("MELD-finetuned only", meld_vectors, gold),
        evaluate("General pretrained only", general_vectors, gold),
        evaluate(f"Ensemble (MELD w={args.meld_weight:.2f})", ensemble_vectors, gold),
    ]

    if args.sweep:
        print(f"\n{'=' * 74}\nEnsemble weight sweep\n{'=' * 74}")
        print(f"  {'w(MELD)':>8}  {'accuracy':>9}  {'macro-F1':>9}")
        from sklearn.metrics import f1_score

        for w in np.arange(0.0, 1.01, 0.1):
            vectors = blend(meld_vectors, general_vectors, float(w))
            predicted = [EMOTIONS[int(np.argmax(v))] for v in vectors]
            acc = float(np.mean([p == g for p, g in zip(predicted, gold)]))
            f1 = f1_score(gold, predicted, labels=EMOTIONS, average="macro", zero_division=0)
            print(f"  {w:>8.1f}  {acc:>9.3f}  {f1:>9.3f}")
        print(
            "\n  Caution: picking the peak of this curve on the same ~50 sentences "
            "you evaluate on is overfitting. Quote the curve's shape, not its argmax."
        )

    print(f"\n{'=' * 74}\nSummary\n{'=' * 74}")
    print(f"  {'condition':<34}{'acc':>7}{'macroF1':>9}{'calib':>8}")
    for r in results:
        print(
            f"  {r['name']:<34}{r['accuracy']:>7.3f}{r['macro_f1']:>9.3f}"
            f"{r['calibration_separation']:>+8.3f}"
        )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "probe_size": len(PROBE),
                    "class_balance": counts(),
                    "meld_weight": args.meld_weight,
                    "results": results,
                    "gold": gold,
                    "sentences": texts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
