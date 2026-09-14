"""Fetch and sanity-check the MELD text annotations.

We only need the CSVs, not the 10GB of raw video — the text channel trains on
the utterance transcripts, and those are a few megabytes that live directly in
the declare-lab/MELD repo.

    python -m training.prepare_meld

Writes to data/meld/ at the project root. Pass --source to point at a local
checkout instead if you've already cloned the repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import PROJECT_ROOT  # noqa: E402
from app.emotions import EMOTIONS  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "meld"

SPLITS = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

# The repo has moved this directory around across commits, so try the known
# layouts before giving up and asking the user to clone manually.
REMOTE_BASES = [
    "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/",
    "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD_Dyadic/",
    "https://raw.githubusercontent.com/declare-lab/MELD/main/data/MELD/",
]

REQUIRED_COLUMNS = {"Utterance", "Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID"}


def fetch_split(filename: str, source: str | None) -> pd.DataFrame:
    if source:
        local = Path(source) / filename
        if not local.exists():
            raise FileNotFoundError(f"{local} not found in --source directory")
        print(f"  reading {local}")
        return pd.read_csv(local)

    last_error: Exception | None = None
    for base in REMOTE_BASES:
        url = base + filename
        try:
            print(f"  trying {url}")
            # MELD's CSVs contain stray encoding artefacts from the subtitle
            # extraction; latin-1 reads them without choking where utf-8 fails.
            return pd.read_csv(url, encoding="utf-8", encoding_errors="replace")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(
        f"Could not download {filename} from any known location. "
        f"Clone https://github.com/declare-lab/MELD and pass --source <repo>/data/MELD. "
        f"Last error: {last_error}"
    )


def clean(df: pd.DataFrame, split: str) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{split}: expected columns missing from CSV: {sorted(missing)}")

    df = df.copy()
    df["Emotion"] = df["Emotion"].astype(str).str.strip().str.lower()
    df["Utterance"] = (
        df["Utterance"]
        .astype(str)
        # Subtitle rips are full of these; they'd otherwise become tokens.
        .str.replace("\x92", "'", regex=False)
        .str.replace("\x85", "...", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    unknown = set(df["Emotion"]) - set(EMOTIONS)
    if unknown:
        raise ValueError(f"{split}: unexpected emotion labels {unknown}")

    before = len(df)
    df = df[df["Utterance"].str.len() > 0]
    # Dialogue order matters for the context feature in train_text.py.
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)
    if before != len(df):
        print(f"  dropped {before - len(df)} empty utterances")
    return df


def report(df: pd.DataFrame, split: str) -> None:
    counts = df["Emotion"].value_counts()
    total = len(df)
    print(f"\n{split}: {total} utterances, {df['Dialogue_ID'].nunique()} dialogues")
    for emotion in EMOTIONS:
        n = int(counts.get(emotion, 0))
        bar = "#" * int(40 * n / max(1, counts.max()))
        print(f"  {emotion:<9} {n:>5} ({100 * n / total:4.1f}%) {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", help="Local directory holding the MELD CSVs (skips download)"
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split, filename in SPLITS.items():
        print(f"\n=== {split} ===")
        df = clean(fetch_split(filename, args.source), split)
        out = DATA_DIR / f"{split}.csv"
        df.to_csv(out, index=False)
        report(df, split)
        print(f"  -> {out}")

    print(
        "\nDone. Note the class imbalance above: neutral is roughly half of the "
        "training set, which is why train_text.py uses class-weighted loss and "
        "reports weighted-F1 rather than accuracy."
    )


if __name__ == "__main__":
    main()
