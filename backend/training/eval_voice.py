"""Score both voice tiers on MELD audio — the voice channel's first measurement.

    python -m training.eval_voice --source D:\\MELD.Raw --splits dev,test

The README calls the voice channel "the weak one" and says the prosody fallback
"is *not* a classifier and shouldn't be reported as one". Both claims are
plausible and neither has a number behind it. This produces them.

It scores the two tiers app/audio_model.py actually ships, on the same clips,
through the same decoder the app uses (app/audio_io.decode, i.e. PyAV via
faster-whisper), so the numbers describe the deployed path rather than a
reimplementation of it:

1. the wav2vec2 SER head (~1.2GB, RAVDESS-trained, 8 classes folded to 7)
2. the prosody heuristic (numpy, no model)

The head-to-head is the point. The expensive tier is assumed better and has never
been checked, and there are three distinct outcomes that all matter:

* The neural head wins clearly — then the 1.2GB in the desktop bundle is earned.
* It wins narrowly — then `MINDSCAPE_VOICE_EMOTION=false` is a defensible
  default for a laptop install, and that is a shipping decision, not a caveat.
* It loses — then the app has been paying 1.2GB and a background download to be
  worse than thirty lines of numpy, which is worth knowing before a demo.

Two structural problems are measured rather than argued about:

*The calm/neutral collapse.* SER_TO_MELD sends both "calm" and "neutral" to
neutral, so two of eight source classes pile onto one target class. If that
inflates the neutral prior, it shows up as neutral recall far above precision.

*Acted source data.* RAVDESS is acted emotional speech; MELD is acted
conversational speech. Neither is spontaneous, and the transfer between them is
exactly the domain step the text channel already measured (README, B->C).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import PROJECT_ROOT, settings  # noqa: E402
from app.emotions import EMOTIONS  # noqa: E402
from training.prepare_meld_video import index_clips  # noqa: E402
from training.train_face import (  # noqa: E402
    expected_calibration_error,
    fit_prior_correction,
    fit_temperature,
    score,
    softmax,
)

CSV_DIR = PROJECT_ROOT / "data" / "meld"
OUT_DIR = PROJECT_ROOT / "data" / "meld_voice"
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}


def load_audio(clip: Path) -> Optional[np.ndarray]:
    """Decode a clip's audio track through the app's own decoder."""
    from app.audio_io import decode

    try:
        return decode(clip.read_bytes())
    except Exception:  # noqa: BLE001
        return None


def run_split(
    split: str, clips: Dict[str, Path], args: argparse.Namespace
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """Score every utterance in a split with both tiers."""
    from app.audio_model import VoiceEmotionModel, prosody_vector

    csv_path = CSV_DIR / f"{split}.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} missing. Run: python -m training.prepare_meld")

    utterances = pd.read_csv(csv_path)
    if args.limit:
        utterances = utterances.head(args.limit)

    voice = VoiceEmotionModel()
    if not args.prosody_only:
        print("  loading wav2vec2 SER head (blocking; the app loads it in background)...")
        started = time.time()
        # _load() is the synchronous body the app runs on a daemon thread. An
        # evaluation wants to wait for it rather than silently fall through to
        # prosody and score the wrong tier.
        voice._load()  # noqa: SLF001
        if voice._model is None:  # noqa: SLF001
            raise SystemExit(f"voice model failed to load: {voice._load_error}")  # noqa: SLF001
        print(f"  loaded in {time.time() - started:.1f}s "
              f"({sorted(set(voice._id2label.values()))})")  # noqa: SLF001

    rows: List[dict] = []
    neural: List[np.ndarray] = []
    prosody: List[np.ndarray] = []
    misses: Dict[str, int] = {}
    neural_ms = 0.0

    for position, row in enumerate(utterances.itertuples(index=False), start=1):
        dialogue, utterance = int(row.Dialogue_ID), int(row.Utterance_ID)
        clip = clips.get(f"dia{dialogue}_utt{utterance}.mp4")
        if clip is None:
            misses["no-clip"] = misses.get("no-clip", 0) + 1
            continue

        waveform = load_audio(clip)
        if waveform is None or waveform.size == 0:
            misses["undecodable"] = misses.get("undecodable", 0) + 1
            continue

        from app.audio_io import SAMPLE_RATE
        from app.audio_model import MIN_DURATION_S

        if waveform.size < MIN_DURATION_S * SAMPLE_RATE:
            # The app returns a uniform vector here and tags it
            # "insufficient-audio"; scoring that as a prediction would credit the
            # model for clips it explicitly declined to judge.
            misses["too-short"] = misses.get("too-short", 0) + 1
            continue

        prosody.append(np.asarray(prosody_vector(waveform), dtype=np.float64))

        if args.prosody_only:
            neural.append(np.full(len(EMOTIONS), 1.0 / len(EMOTIONS)))
        else:
            t0 = time.time()
            vector = voice._predict_neural(waveform)  # noqa: SLF001
            neural_ms += (time.time() - t0) * 1000.0
            if vector is None:
                misses["neural-failed"] = misses.get("neural-failed", 0) + 1
                prosody.pop()
                continue
            neural.append(np.asarray(vector, dtype=np.float64))

        rows.append(
            {
                "split": split,
                "utterance_key": f"{dialogue}_{utterance}",
                "emotion": str(row.Emotion),
                "label": LABEL2ID[str(row.Emotion)],
                "duration_s": round(waveform.size / SAMPLE_RATE, 2),
            }
        )

        if position % 200 == 0:
            print(f"    {position}/{len(utterances)} -> {len(rows)} scored")

    df = pd.DataFrame(rows)
    stats = {
        "utterances": len(utterances),
        "scored": len(df),
        "dropped": misses,
        "mean_neural_ms": round(neural_ms / max(1, len(df)), 1),
        "mean_duration_s": round(float(df["duration_s"].mean()), 2) if len(df) else 0.0,
    }
    return df, np.array(neural), np.array(prosody), stats


def neutral_inflation(probs: np.ndarray, labels: np.ndarray, name: str) -> dict:
    """Measure the calm->neutral collapse rather than arguing about it.

    Two of RAVDESS's eight classes map onto MELD's single neutral, so neutral
    receives probability mass from two sources on every clip. If that matters, it
    shows as neutral being over-predicted: recall high, precision low.
    """
    idx = LABEL2ID["neutral"]
    preds = probs.argmax(axis=1)
    predicted_neutral = float((preds == idx).mean())
    actually_neutral = float((labels == idx).mean())
    mean_neutral_mass = float(probs[:, idx].mean())

    print(f"\n  {name}: neutral predicted on {100 * predicted_neutral:.1f}% of clips, "
          f"true rate {100 * actually_neutral:.1f}%")
    print(f"    mean probability mass on neutral: {mean_neutral_mass:.3f}")
    if predicted_neutral > actually_neutral * 1.3:
        print("    -> over-predicted; consistent with the calm/neutral collapse")

    return {
        "predicted_neutral_rate": round(predicted_neutral, 4),
        "true_neutral_rate": round(actually_neutral, 4),
        "mean_neutral_mass": round(mean_neutral_mass, 4),
        "inflation_ratio": round(predicted_neutral / max(1e-9, actually_neutral), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="extracted MELD.Raw")
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--prosody-only",
        action="store_true",
        help="skip the 1.2GB head and score the heuristic alone",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR / "voice_baseline.json")
    args = parser.parse_args()

    from sklearn.metrics import classification_report, confusion_matrix

    if not args.source.is_dir():
        raise SystemExit(f"--source {args.source} is not a directory")

    print(f"Indexing clips under {args.source} ...")
    clips = index_clips(args.source)
    print(f"  found {len(clips)} .mp4 files")
    print(f"  device: {settings.resolved_device()}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    collected = {}
    for split in splits:
        print(f"\n=== {split} ===")
        collected[split] = run_split(split, clips, args)

    # Temperature and per-class bias for each tier, both fitted on dev.
    #
    # The bias matters more than the temperature for a SER head: these are
    # trained on acted corpora where almost every clip is expressive, so they
    # assign almost no mass to neutral, while MELD is ~48% neutral. Temperature
    # cannot fix that — it scales all classes equally — and without the bias a
    # head that ranks emotions well still scores near zero.
    temperatures = {"neural": 1.0, "prosody": 1.0}
    biases = {
        "neural": np.zeros(len(EMOTIONS)),
        "prosody": np.zeros(len(EMOTIONS)),
    }
    if "dev" in collected:
        dev_df, dev_neural, dev_prosody, _ = collected["dev"]
        dev_labels = dev_df["label"].to_numpy()
        if len(dev_df):
            if not args.prosody_only:
                dev_logits = np.log(np.clip(dev_neural, 1e-12, None))
                temperatures["neural"] = fit_temperature(dev_logits, dev_labels)
                biases["neural"] = fit_prior_correction(dev_logits, dev_labels)
            dev_prosody_logits = np.log(np.clip(dev_prosody, 1e-12, None))
            temperatures["prosody"] = fit_temperature(dev_prosody_logits, dev_labels)
            biases["prosody"] = fit_prior_correction(dev_prosody_logits, dev_labels)
            print(f"\nTemperatures fitted on dev: {temperatures}")
            if not args.prosody_only:
                print("Per-class bias fitted on dev (neural):")
                for emotion, b in zip(EMOTIONS, biases["neural"]):
                    print(f"  {emotion:<9} {b:+.3f}")

    report: Dict[str, dict] = {
        "temperatures": temperatures,
        "prior_bias": {k: v.tolist() for k, v in biases.items()},
        "emotions": EMOTIONS,
        "splits": {},
    }

    for split in splits:
        df, neural, prosody, stats = collected[split]
        if not len(df):
            print(f"\n{split}: nothing scored ({stats['dropped']})")
            continue

        labels = df["label"].to_numpy()
        print(f"\n{'=' * 66}")
        print(f"{split}: {stats['scored']}/{stats['utterances']} utterances scored "
              f"(mean {stats['mean_duration_s']}s)")
        if stats["dropped"]:
            print("  dropped:", ", ".join(f"{k}={v}" for k, v in sorted(stats["dropped"].items())))
        print("=" * 66)

        entry: Dict[str, object] = {"stats": stats, "metrics": {}}

        counts = np.bincount(labels, minlength=len(EMOTIONS)).astype(np.float64)
        prior = np.tile(counts / counts.sum(), (len(labels), 1))
        entry["metrics"]["prior"] = score(prior, labels, "prior baseline")

        if not args.prosody_only:
            neural_logits = np.log(np.clip(neural, 1e-12, None))
            entry["metrics"]["neural_raw"] = score(neural, labels, "wav2vec2 SER (as shipped)")
            cal = softmax(neural_logits / temperatures["neural"])
            entry["metrics"]["neural_calibrated"] = score(cal, labels, "wav2vec2 SER (T-scaled)")
            corrected = softmax(neural_logits + biases["neural"])
            entry["metrics"]["neural_prior_corrected"] = score(
                corrected, labels, "wav2vec2 SER (prior-corrected)"
            )
            both = softmax(neural_logits / temperatures["neural"] + biases["neural"])
            entry["metrics"]["neural_prior_and_T"] = score(
                both, labels, "wav2vec2 SER (prior + T)"
            )

        entry["metrics"]["prosody_raw"] = score(prosody, labels, "prosody heuristic")
        cal_p = softmax(np.log(np.clip(prosody, 1e-12, None)) / temperatures["prosody"])
        entry["metrics"]["prosody_calibrated"] = score(cal_p, labels, "prosody (T-scaled)")

        best = neural if not args.prosody_only else prosody
        best_name = "wav2vec2" if not args.prosody_only else "prosody"

        print(f"\nPer-class breakdown ({best_name}):")
        report_text = classification_report(
            labels,
            best.argmax(axis=1),
            labels=list(range(len(EMOTIONS))),
            target_names=EMOTIONS,
            digits=3,
            zero_division=0,
        )
        print(report_text)
        entry["classification_report"] = report_text

        matrix = confusion_matrix(
            labels, best.argmax(axis=1), labels=list(range(len(EMOTIONS)))
        )
        print(f"Confusion ({best_name}; rows = MELD truth):")
        print(f"  {'':<9}" + "".join(f"{e[:6]:>8}" for e in EMOTIONS))
        for i, emotion in enumerate(EMOTIONS):
            print(f"  {emotion:<9}" + "".join(f"{v:>8}" for v in matrix[i]))
        entry["confusion_matrix"] = matrix.tolist()

        entry["neutral_inflation"] = {}
        if not args.prosody_only:
            entry["neutral_inflation"]["neural"] = neutral_inflation(neural, labels, "wav2vec2")
        entry["neutral_inflation"]["prosody"] = neutral_inflation(prosody, labels, "prosody")

        if not args.prosody_only:
            print("\n--- Head to head ---")
            n = entry["metrics"]["neural_raw"]
            p = entry["metrics"]["prosody_raw"]
            delta = n["weighted_f1"] - p["weighted_f1"]
            print(f"  wav2vec2 {n['weighted_f1']:.4f} vs prosody {p['weighted_f1']:.4f} "
                  f"weighted-F1  (delta {delta:+.4f})")
            print(f"  macro-F1  {n['macro_f1']:.4f} vs {p['macro_f1']:.4f}"
                  f"  (delta {n['macro_f1'] - p['macro_f1']:+.4f})")
            print(f"  ECE       {n['ece']:.4f} vs {p['ece']:.4f}  (lower is better)")
            print(f"  cost      ~1.2GB + {stats['mean_neural_ms']}ms/clip  vs  ~0 + negligible")
            if delta < 0.01:
                print("  -> the expensive tier is not earning its download on this data.")

        report["splits"][split] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
