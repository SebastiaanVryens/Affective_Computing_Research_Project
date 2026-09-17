"""Map additional emotion corpora into MELD's label space.

    python -m training.prepare_corpora
    python -m training.prepare_corpora --corpora goemotions,dailydialog --cap 20000

Why this exists: `training/eval_diary.py` shows the MELD-only model scoring 0.532
on diary-register text against 0.578 in-domain. MELD is acted, multi-speaker
sitcom dialogue; the app receives one person narrating their day in retrospect.
The cheapest fix for a domain gap is usually more *varied* data, not a bigger
model — so this pulls in corpora that sit closer to the target register and
projects them all onto the same seven labels.

Corpora, and why each was chosen:

* **GoEmotions** (Reddit comments, 43k). First-person, informal, self-reported
  experience — much closer to diary register than scripted dialogue. Google
  publishes an Ekman grouping of its 27 fine labels that collapses to *exactly*
  MELD's seven, so no judgement calls are needed in the mapping.

* **DailyDialog** (everyday conversations, 87k). Its seven labels are already
  ours, one-for-one. Mundane daily topics — closer to diary subject matter than
  Friends, though still two-person dialogue. 83% of it is "no emotion", so the
  neutral class is aggressively subsampled.

* **dair-ai/emotion** (tweets, 16k). First-person and emotional, but only six
  labels with no neutral and no disgust, and tweet register is its own dialect.
  Off by default; enable with `--corpora ...,tweets` to test whether it helps.

Everything is written to data/corpora/ in the same schema MELD uses, so
train_text.py can mix them without special-casing.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import PROJECT_ROOT  # noqa: E402
from app.emotions import EMOTIONS  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "corpora"

# Google's published Ekman grouping for GoEmotions' 27 fine-grained labels.
# Source: google-research/google-research/goemotions/data/ekman_mapping.json
# Reproduced here rather than fetched so the mapping is visible and auditable —
# it's the single most consequential judgement call in this file.
GOEMOTIONS_EKMAN = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": [
        "joy", "amusement", "approval", "excitement", "gratitude", "love",
        "optimism", "relief", "pride", "admiration", "desire", "caring",
    ],
    "sadness": [
        "sadness", "disappointment", "embarrassment", "grief", "remorse",
    ],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}
FINE_TO_EKMAN = {
    fine: coarse for coarse, fines in GOEMOTIONS_EKMAN.items() for fine in fines
}

# DailyDialog's integer emotion codes. These already are MELD's seven.
DAILYDIALOG_LABELS = {
    0: "neutral", 1: "anger", 2: "disgust", 3: "fear",
    4: "joy", 5: "sadness", 6: "surprise",
}

# dair-ai/emotion. "love" has no MELD counterpart and is folded into joy, which
# is the least-wrong option but is a real approximation worth flagging.
TWEET_LABELS = {
    0: "sadness", 1: "joy", 2: "joy",  # 2 = love -> joy
    3: "anger", 4: "fear", 5: "surprise",
}


def load_goemotions(cap: int | None, cap_mode: str = "proportional") -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(
        "google-research-datasets/go_emotions", "simplified", split="train"
    )
    names = ds.features["labels"].feature.names

    rows = []
    dropped_ambiguous = 0
    for example in ds:
        coarse = {FINE_TO_EKMAN.get(names[i]) for i in example["labels"]}
        coarse.discard(None)
        # GoEmotions is multi-label. An example whose fine labels land in two
        # different Ekman buckets has no single correct answer in our scheme, so
        # it's dropped rather than arbitrarily resolved.
        if len(coarse) != 1:
            dropped_ambiguous += 1
            continue
        rows.append({"text": example["text"].strip(), "Emotion": coarse.pop()})

    df = pd.DataFrame(rows)
    print(f"  dropped {dropped_ambiguous} multi-Ekman examples as ambiguous")
    return _finalize(df, "goemotions", cap, cap_mode)


def load_dailydialog(cap: int | None, cap_mode: str = "proportional") -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("benjaminbeilharz/better_daily_dialog", split="train")
    df = pd.DataFrame(
        {
            "text": [u.strip() for u in ds["utterance"]],
            "Emotion": [DAILYDIALOG_LABELS.get(e) for e in ds["emotion"]],
        }
    )
    return _finalize(df, "dailydialog", cap, cap_mode)


def load_tweets(cap: int | None, cap_mode: str = "proportional") -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("dair-ai/emotion", split="train")
    df = pd.DataFrame(
        {
            "text": [t.strip() for t in ds["text"]],
            "Emotion": [TWEET_LABELS.get(l) for l in ds["label"]],
        }
    )
    return _finalize(df, "tweets", cap, cap_mode)


LOADERS = {
    "goemotions": load_goemotions,
    "dailydialog": load_dailydialog,
    "tweets": load_tweets,
}


def _finalize(
    df: pd.DataFrame, source: str, cap: int | None, cap_mode: str = "proportional"
) -> pd.DataFrame:
    df = df.dropna(subset=["text", "Emotion"])
    df = df[df["Emotion"].isin(EMOTIONS)]
    df = df[df["text"].str.len() >= 4]
    df = df.drop_duplicates(subset=["text"])

    # Neutral dominates every one of these corpora (83% of DailyDialog). Left
    # alone it would drown the rare classes far worse than MELD already does, so
    # cap it at twice the next-largest class — enough to keep neutral learnable
    # without letting it set the prior.
    counts = df["Emotion"].value_counts()
    non_neutral_max = counts.drop("neutral", errors="ignore").max()
    if "neutral" in counts and counts["neutral"] > 2 * non_neutral_max:
        keep = int(2 * non_neutral_max)
        neutral = df[df["Emotion"] == "neutral"].sample(n=keep, random_state=42)
        df = pd.concat([df[df["Emotion"] != "neutral"], neutral])
        print(f"  subsampled neutral {counts['neutral']} -> {keep}")

    if cap and len(df) > cap:
        if cap_mode == "proportional":
            # Stratified cap, so trimming for size doesn't also reshape the balance.
            df = (
                df.groupby("Emotion", group_keys=False)
                .apply(lambda g: g.sample(n=max(1, int(cap * len(g) / len(df))), random_state=42))
            )
        else:
            # Cap each class instead of the corpus.
            #
            # The proportional cap preserves the balance, which sounds neutral and
            # isn't: these corpora exist to help the classes MELD starves, and
            # trimming them proportionally trims the rare classes too. DailyDialog
            # went 87k -> 25k and took fear down to 93 examples — fewer than MELD's
            # own fear count, for a corpus added to fix exactly that.
            #
            # Capping per class keeps every rare example and takes the cut out of
            # joy and neutral, which have thousands to spare. The auxiliary
            # distribution is deliberately no longer the source corpus's; that is
            # the point, and train_text.py's class weighting is computed after the
            # mix, so it follows automatically.
            per_class = max(1, cap // len(EMOTIONS))
            df = df.groupby("Emotion", group_keys=False).apply(
                lambda g: g.sample(n=min(len(g), per_class), random_state=42)
            )
            print(f"  capped per class at {per_class}")
        print(f"  capped to {len(df)} examples")

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df["source"] = source
    # These corpora have no dialogue structure we're preserving, so every row is
    # its own single-utterance "dialogue". That makes train_text.py's context
    # builder a no-op for them, which is correct: prepending an unrelated Reddit
    # comment as "context" would be worse than none.
    df["Dialogue_ID"] = [f"{source}_{i}" for i in range(len(df))]
    df["Utterance_ID"] = 0
    df["Speaker"] = "Speaker"
    df["Utterance"] = df["text"]
    return df[
        ["Utterance", "Emotion", "Dialogue_ID", "Utterance_ID", "Speaker", "source"]
    ]


def report(df: pd.DataFrame, name: str) -> None:
    counts = Counter(df["Emotion"])
    total = len(df)
    print(f"\n{name}: {total} examples")
    for emotion in EMOTIONS:
        n = counts.get(emotion, 0)
        bar = "#" * int(36 * n / max(1, max(counts.values())))
        print(f"  {emotion:<9} {n:>6} ({100 * n / max(1, total):4.1f}%) {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpora",
        default="goemotions,dailydialog",
        help=f"comma-separated from: {', '.join(LOADERS)}",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=25000,
        help="max examples per corpus (0 = no cap)",
    )
    parser.add_argument(
        "--cap-mode",
        default="proportional",
        choices=["proportional", "per-class"],
        help="proportional keeps the source corpus's balance; per-class keeps every "
        "rare example and takes the cut out of joy/neutral. These corpora exist to "
        "feed the classes MELD starves, so per-class is usually what you want — "
        "proportional is the default only because it is what the shipped model used",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = args.cap if args.cap > 0 else None

    frames = []
    for name in [c.strip() for c in args.corpora.split(",") if c.strip()]:
        if name not in LOADERS:
            raise SystemExit(f"Unknown corpus '{name}'. Options: {', '.join(LOADERS)}")
        print(f"\n=== {name} ===")
        df = LOADERS[name](cap, args.cap_mode)
        report(df, name)
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"  -> {OUT_DIR / f'{name}.csv'}")
        frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        report(combined, "COMBINED (auxiliary only, excludes MELD)")
        print(
            "\nNow train with:\n"
            "  .venv\\Scripts\\python.exe -m training.train_text "
            "--corpora meld,goemotions,dailydialog --model roberta-base --device cuda"
        )


if __name__ == "__main__":
    main()
