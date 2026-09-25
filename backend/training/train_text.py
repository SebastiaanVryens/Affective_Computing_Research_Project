"""Fine-tune a transformer on MELD utterances for 7-way emotion classification.

    python -m training.prepare_meld          # once
    python -m training.train_text            # CPU-friendly defaults
    python -m training.train_text --model roberta-base --context 2 --device cuda

Two things here are worth defending in a write-up:

*Class-weighted loss.* MELD is severely imbalanced — neutral is ~47% of the
training set, fear and disgust under 3% each. Unweighted cross-entropy produces
a model with respectable accuracy that has simply learned to say "neutral", which
is useless for a diary. Weights are inverse-frequency, so the rare classes carry
proportionally more gradient.

*Dialogue context.* MELD utterances are conversational fragments; "I can't
believe it" is unclassifiable alone. Prepending the previous ``--context``
utterances from the same dialogue is the cheapest large win available on this
dataset, and ablating it (``--context 0``) gives you a clean comparison for the
report.

The headline metric is weighted-F1, which is what the MELD paper reports, so
your numbers are directly comparable to its baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import BACKEND_ROOT, PROJECT_ROOT  # noqa: E402
from app.emotions import EMOTIONS  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "meld"
CORPORA_DIR = PROJECT_ROOT / "data" / "corpora"
DEFAULT_OUTPUT = BACKEND_ROOT / "models" / "meld-text"

LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}


def load_auxiliary(names: List[str], context: int, speakers: bool) -> pd.DataFrame:
    """Load extra corpora produced by training/prepare_corpora.py.

    These are single-utterance rows with unique Dialogue_IDs, so the context
    builder is a no-op on them by construction — prepending an unrelated Reddit
    comment as "prior turn" would be actively misleading.

    Auxiliary data is added to *training only*. Dev and test stay pure MELD, so
    the headline weighted-F1 remains directly comparable to the MELD paper and
    to the single-corpus runs.
    """
    frames = []
    for name in names:
        path = CORPORA_DIR / f"{name}.csv"
        if not path.exists():
            raise SystemExit(
                f"{path} missing. Run: python -m training.prepare_corpora "
                f"--corpora {name}"
            )
        df = pd.read_csv(path)
        df["label"] = df["Emotion"].map(LABEL2ID)
        df["text"] = _with_context(df, context, speakers)
        df["source"] = name
        frames.append(df[["text", "label", "Emotion", "source"]])
        print(f"  + {name}: {len(df)} examples")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_split(split: str, context: int, speakers: bool = False) -> pd.DataFrame:
    path = DATA_DIR / f"{split}.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing. Run: python -m training.prepare_meld")

    df = pd.read_csv(path)
    df["label"] = df["Emotion"].map(LABEL2ID)

    if context <= 0 and not speakers:
        df["text"] = df["Utterance"]
    else:
        df["text"] = _with_context(df, context, speakers)
    return df[["text", "label", "Emotion"]]


def _with_context(df: pd.DataFrame, context: int, speakers: bool) -> pd.Series:
    """Prepend prior turns from the same dialogue, separated by </s>.

    The separator is RoBERTa's own; for a BERT-family model the tokenizer will
    treat it as ordinary text, which still works but less cleanly.

    With ``speakers`` on, each turn is prefixed with who said it. This is
    standard in emotion-recognition-in-conversation work: it lets the model tell
    "the speaker is escalating" from "someone is arguing *with* the speaker",
    which are different emotions from identical words. Note the diary has only
    one speaker, so this helps the MELD benchmark more than it helps the app —
    which is exactly the kind of gap worth reporting.
    """
    out = pd.Series(index=df.index, dtype=object)

    for _, group in df.groupby("Dialogue_ID", sort=False):
        # Sort within the dialogue rather than trusting the file's row order:
        # getting this wrong would silently prepend the *wrong* prior turns,
        # which trains fine and quietly costs accuracy rather than erroring.
        group = group.sort_values("Utterance_ID")

        if speakers:
            turns = [
                f"{str(s).strip()}: {u}"
                for s, u in zip(group["Speaker"], group["Utterance"])
            ]
        else:
            turns = group["Utterance"].tolist()

        for i, row_index in enumerate(group.index):
            prior = turns[max(0, i - context) : i] if context > 0 else []
            out.at[row_index] = (
                " </s> ".join([*prior, turns[i]]) if prior else turns[i]
            )

    # Assigning by index label (not position) keeps every text aligned with its
    # own label even if the caller hands us an unsorted frame.
    return out


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=len(EMOTIONS)).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero on a class absent from a split
    weights = len(labels) / (len(EMOTIONS) * counts)
    return weights


def build_metrics():
    from sklearn.metrics import classification_report, f1_score

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            # MELD's headline number.
            "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
            # Macro treats fear and disgust as equal citizens — the number that
            # actually shows whether the class weighting worked.
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "accuracy": float((preds == labels).mean()),
        }

    return compute, classification_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="distilroberta-base",
        help="distilroberta-base is ~2x faster on CPU; roberta-base scores higher",
    )
    parser.add_argument("--context", type=int, default=2, help="prior turns to prepend")
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--corpora",
        default="meld",
        help="training corpora, comma-separated: meld,goemotions,dailydialog,tweets. "
             "Dev/test always stay pure MELD so the benchmark stays comparable.",
    )
    parser.add_argument(
        "--speaker-tokens",
        action="store_true",
        help="prefix each turn with its speaker (helps MELD; diary has one speaker)",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="ablation: plain cross-entropy, for comparison in the report",
    )
    args = parser.parse_args()

    import torch
    from torch import nn
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    from datasets import Dataset

    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    print(
        f"Device: {'cuda' if use_cuda else 'cpu'} | model: {args.model} | "
        f"context: {args.context} | speakers: {args.speaker_tokens}"
    )

    requested = [c.strip() for c in args.corpora.split(",") if c.strip()]
    if "meld" not in requested:
        raise SystemExit("--corpora must include meld (it supplies dev/test)")
    auxiliary_names = [c for c in requested if c != "meld"]

    splits = {
        s: load_split(s, args.context, args.speaker_tokens)
        for s in ("train", "dev", "test")
    }
    splits["train"]["source"] = "meld"
    print(f"  meld train: {len(splits['train'])} examples")

    if auxiliary_names:
        auxiliary = load_auxiliary(auxiliary_names, args.context, args.speaker_tokens)
        splits["train"] = pd.concat(
            [splits["train"], auxiliary], ignore_index=True
        ).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"  train total: {len(splits['train'])} examples")
    print(f"  dev  (MELD): {len(splits['dev'])}   test (MELD): {len(splits['test'])}")
    print("  train class balance:", dict(splits["train"]["Emotion"].value_counts()))

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    datasets = {
        name: Dataset.from_pandas(df[["text", "label"]], preserve_index=False).map(
            tokenize, batched=True, remove_columns=["text"]
        )
        for name, df in splits.items()
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(EMOTIONS),
        # Baking the label names into the config is what lets app/text_model.py
        # load this checkpoint through a plain pipeline() with no remapping.
        id2label={i: e for i, e in enumerate(EMOTIONS)},
        label2id=LABEL2ID,
    )

    weights = None
    if not args.no_class_weights:
        weights = torch.tensor(
            class_weights(splits["train"]["label"].to_numpy()), dtype=torch.float
        )
        print("\nClass weights (inverse frequency):")
        for emotion, w in zip(EMOTIONS, weights.tolist()):
            print(f"  {emotion:<9} {w:.3f}")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = nn.CrossEntropyLoss(
                weight=weights.to(outputs.logits.device) if weights is not None else None
            )
            loss = loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    compute_metrics, classification_report = build_metrics()

    # transformers 5 dropped `warmup_ratio` in favour of an explicit step count,
    # so derive the same 6% warmup from the actual schedule length. Computed
    # rather than hardcoded so it still tracks if batch size or epochs change.
    steps_per_epoch = math.ceil(len(datasets["train"]) / args.batch_size)
    warmup_steps = max(1, int(0.06 * steps_per_epoch * args.epochs))
    print(f"Schedule: {steps_per_epoch} steps/epoch, {warmup_steps} warmup steps")

    training_args = TrainingArguments(
        output_dir=str(args.output.parent / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=use_cuda,
        use_cpu=not use_cuda,
        report_to=[],
        seed=42,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Dev first, and saved alongside test below.
    #
    # Comparing runs on the test number and then shipping the winner is test-set
    # selection: the reported figure stops being an estimate of held-out
    # performance and becomes the maximum over however many configurations were
    # tried. Dev exists to absorb that. Choose the configuration on `dev_metrics`,
    # then quote `test_metrics` once for the one you chose.
    print("\n=== Dev set (use THIS to choose between runs) ===")
    dev_metrics = trainer.evaluate(datasets["dev"], metric_key_prefix="dev")
    for key, value in dev_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    print("\n=== Test set (quote once, for the chosen run) ===")
    metrics = trainer.evaluate(datasets["test"], metric_key_prefix="test")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    predictions = trainer.predict(datasets["test"])
    preds = np.argmax(predictions.predictions, axis=-1)
    print("\nPer-class breakdown (the table to put in the report):")
    report_text = classification_report(
        splits["test"]["label"].to_numpy(),
        preds,
        target_names=EMOTIONS,
        digits=3,
        zero_division=0,
    )
    print(report_text)

    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    (args.output / "meld_results.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "corpora": requested,
                "context_turns": args.context,
                "speaker_tokens": args.speaker_tokens,
                "class_weighted": not args.no_class_weights,
                "epochs": args.epochs,
                "train_examples": len(splits["train"]),
                "train_class_balance": {
                    str(k): int(v) for k, v in splits["train"]["Emotion"].value_counts().items()
                },
                "dev_metrics": {
                    k: v for k, v in dev_metrics.items() if isinstance(v, (int, float))
                },
                "test_metrics": {
                    k: v for k, v in metrics.items() if isinstance(v, (int, float))
                },
                "classification_report": report_text,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved to {args.output}")
    print("The backend picks this up automatically on next start.")


if __name__ == "__main__":
    main()
