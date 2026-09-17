"""Fine-tune a small CNN on MELD face crops for 7-way emotion classification.

    python -m training.prepare_meld_video --source D:\\MELD.Raw   # once
    python -m training.train_face --device cuda
    python -m training.train_face --model google/mobilenet_v2_1.0_224 --epochs 8

This replaces the only channel in the app that was never trained on anything:
frontend/src/capture/face.ts loads face-api's stock expression head, and the
README's own complaint about it ("reads a resting face as slightly sad") has
never had a number attached to it. Now it can.

Three choices here are specific to how the face vector is *consumed*, and are
worth defending separately from accuracy:

*Calibration is a first-class metric, not a footnote.* app/fusion.py weights each
channel by the entropy of its distribution, so a head that is confidently wrong
does not merely contribute noise — it wins weight while doing it. A model with
better top-1 and worse calibration would make the fused reading worse. We
therefore report expected calibration error alongside F1, fit a temperature on
dev, and save it beside the weights for inference to apply.

*Clip-level metrics, not just frame-level.* The app never classifies one frame:
face.ts averages readings over a window before fusion touches them. Mean-pooling
the frames of an utterance and scoring that is the number that predicts app
behaviour; the frame-level number is the one comparable to other FER papers. Both
are reported, and the gap between them is itself a result.

*Label smoothing on top of class weighting.* The frame labels are weak by
construction — an utterance label stamped onto every sampled frame — so training
the model to be certain about them teaches overconfidence on exactly the frames
that are mislabelled. Smoothing is the cheap correction, and ``--label-smoothing
0`` is the ablation.

The headline metric stays weighted-F1, matching train_text.py and the MELD paper.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import BACKEND_ROOT, PROJECT_ROOT  # noqa: E402
from app.emotions import EMOTIONS  # noqa: E402

FACES_DIR = PROJECT_ROOT / "data" / "meld_faces"
DEFAULT_OUTPUT = BACKEND_ROOT / "models" / "meld-face"

LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_manifest(split: str, args: argparse.Namespace) -> pd.DataFrame:
    path = FACES_DIR / f"{split}.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} missing. Run: python -m training.prepare_meld_video --source <MELD.Raw>"
        )

    df = pd.read_csv(path)
    if not len(df):
        raise SystemExit(f"{path} is empty — no faces were extracted for {split}.")

    before = len(df)
    if args.min_det_score > 0:
        df = df[df["det_score"] >= args.min_det_score]
    if args.unambiguous:
        # Only clips where the speaker heuristic had no serious rival. Costs data
        # and buys label purity; running both is the ablation that says whether
        # the speaker-selection noise actually matters.
        df = df[df["rival_faces"] == 0]
    if before != len(df):
        print(f"  {split}: filtered {before - len(df)} crops ({before} -> {len(df)})")

    df = df.reset_index(drop=True)
    df["label"] = df["emotion"].map(LABEL2ID)
    if df["label"].isna().any():
        bad = sorted(set(df.loc[df["label"].isna(), "emotion"]))
        raise SystemExit(f"{split}: unexpected emotion labels {bad}")
    df["label"] = df["label"].astype(int)
    return df


class FaceFrames:
    """Reads crops off disk and augments them. Deliberately not torchvision.

    The backend's only vision dependency is OpenCV, which prepare_meld_video.py
    already needs for decoding; pulling torchvision in for four augmentations
    would add a CUDA-version-pinned dependency to a project that installs torch
    by hand. These are the four that matter for faces.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        mean: np.ndarray,
        std: np.ndarray,
        size: int,
        train: bool,
        seed: int = 42,
    ) -> None:
        self.paths = [str(FACES_DIR / p) for p in frame["path"]]
        self.labels = frame["label"].to_numpy()
        self.mean = mean.reshape(3, 1, 1)
        self.std = std.reshape(3, 1, 1)
        self.size = size
        self.train = train
        self.seed = seed
        # Seeded lazily, per worker. Note the module is never held as an
        # attribute: Windows spawns dataloader workers rather than forking them,
        # so the dataset gets pickled, and a module object is not picklable.
        self._rng: Optional[np.random.Generator] = None

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            import torch

            info = torch.utils.data.get_worker_info()
            worker = info.id if info is not None else 0
            self._rng = np.random.default_rng(self.seed + 1000 * worker)
        return self._rng

    def _augment(self, image: np.ndarray) -> np.ndarray:
        import cv2

        height, width = image.shape[:2]

        # Random resized crop, 80-100% of the frame. The crops already have a
        # 35% margin around the detector box, so this jitters how tightly the
        # face is framed — which is exactly what varies between the detector
        # used here and face-api's in the browser.
        scale = float(self.rng.uniform(0.8, 1.0))
        side = int(round(min(height, width) * scale))
        x = int(self.rng.integers(0, max(1, width - side + 1)))
        y = int(self.rng.integers(0, max(1, height - side + 1)))
        image = image[y : y + side, x : x + side]

        # Small rotation only. A face is not rotation-invariant past a head
        # tilt, and training on 30-degree rotations teaches a pose the detector
        # would never hand us at inference.
        if self.rng.random() < 0.5:
            angle = float(self.rng.uniform(-12, 12))
            matrix = cv2.getRotationMatrix2D(
                (image.shape[1] / 2.0, image.shape[0] / 2.0), angle, 1.0
            )
            image = cv2.warpAffine(
                image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT_101
            )

        # Horizontal flip. Safe for the seven classes — none of them is
        # lateralised the way, say, a gesture label would be.
        if self.rng.random() < 0.5:
            image = image[:, ::-1]

        # Brightness/contrast. Friends is lit consistently; a webcam is not.
        if self.rng.random() < 0.7:
            alpha = float(self.rng.uniform(0.8, 1.25))  # contrast
            beta = float(self.rng.uniform(-25, 25))  # brightness
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        return image

    def __getitem__(self, index: int) -> Dict:
        import cv2
        import torch

        image = cv2.imread(self.paths[index], cv2.IMREAD_COLOR)
        if image is None:
            # A truncated JPEG shouldn't kill an eight-hour run.
            image = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        if self.train:
            image = self._augment(image)

        if image.shape[0] != self.size or image.shape[1] != self.size:
            interpolation = cv2.INTER_AREA if image.shape[0] > self.size else cv2.INTER_CUBIC
            image = cv2.resize(image, (self.size, self.size), interpolation=interpolation)

        # BGR (OpenCV) -> RGB (every pretrained backbone). Getting this wrong
        # trains and serves fine, and quietly costs several F1 points.
        image = np.ascontiguousarray(image[:, :, ::-1])
        array = image.astype(np.float32).transpose(2, 0, 1) / 255.0
        array = (array - self.mean) / self.std

        return {
            "pixel_values": torch.from_numpy(array),
            "labels": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Standard ECE: mean gap between confidence and accuracy, bin-weighted.

    Reported because fusion.py's entropy weighting reads this model's confidence
    literally. ECE is the number that says whether that reading can be trusted.
    """
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)

    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Single-parameter temperature scaling, fitted by grid search on dev NLL.

    A grid beats LBFGS here for the same reason it does in any one-dimensional
    convex-ish problem: it cannot diverge, needs no gradients, and 400 forward
    evaluations on a dev set this size is instant.
    """
    # The ceiling is 20 rather than the conventional 5 because face-api's stock
    # head saturated the old bound on MELD: its ECE is 0.515 and even T=5 was
    # not enough to cool it. A grid that cannot express the answer silently
    # returns its own edge, so the range has to cover the models actually
    # being measured, not the ones that behave.
    lo, hi = 0.25, 20.0
    grid = np.linspace(lo, hi, 800)
    best_t, best_nll = 1.0, math.inf
    for temperature in grid:
        probs = softmax(logits / temperature)
        nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean()
        if nll < best_nll:
            best_t, best_nll = float(temperature), float(nll)

    # Landing on an endpoint means the optimum is outside the grid, so the
    # returned value is a clamp rather than a fit. Worth saying out loud: it
    # usually means dev is too easy or too small for the fit to mean anything.
    if best_t <= grid[1] or best_t >= grid[-2]:
        print(
            f"  ! temperature hit the search bound ({best_t:.2f} in [{lo}, {hi}]) — "
            "treat it as a clamp, not a fitted value."
        )
    return best_t


def fit_prior_correction(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fit a per-class log-bias on dev — temperature scaling's generalisation.

    A temperature divides every logit equally, so it can make a model more or
    less confident but cannot move probability *between* classes. That is useless
    against a prior mismatch, where a model is fine at ranking but was trained on
    a different class balance than it is being used on.

    The voice head is the case in point: trained on acted speech where nearly
    every clip is expressive, it assigns a mean 0.010 of its mass to neutral,
    while MELD is 48% neutral. It discriminates joy/anger/surprise well above
    chance and still scores 0.098 weighted-F1, purely because it will not say
    "neutral". A bias vector fixes exactly that and nothing else — it shifts the
    operating point, leaving the model's ranking within a clip untouched.

    Fitted by minimising dev NLL over the seven biases. Returns the vector to add
    to ``log(p)`` before re-softmaxing.
    """
    from scipy.optimize import minimize

    n_classes = logits.shape[1]
    index = np.arange(len(labels))

    def nll(bias: np.ndarray) -> float:
        probs = softmax(logits + bias)
        return float(-np.log(np.clip(probs[index, labels], 1e-12, None)).mean())

    result = minimize(nll, np.zeros(n_classes), method="L-BFGS-B")
    bias = np.asarray(result.x, dtype=np.float64)
    # Only differences matter to softmax; centring keeps the numbers readable.
    return bias - bias.mean()


def score(probs: np.ndarray, labels: np.ndarray, title: str) -> dict:
    from sklearn.metrics import f1_score

    preds = probs.argmax(axis=1)
    metrics = {
        "n": int(len(labels)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float((preds == labels).mean()),
        "ece": expected_calibration_error(probs, labels),
        "mean_confidence": float(probs.max(axis=1).mean()),
    }
    print(
        f"  {title:<28} weighted-F1 {metrics['weighted_f1']:.4f}  "
        f"macro-F1 {metrics['macro_f1']:.4f}  acc {metrics['accuracy']:.4f}  "
        f"ECE {metrics['ece']:.4f}"
    )
    return metrics


def pool_by_clip(
    probs: np.ndarray, frame: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean-pool frame probabilities within each utterance.

    This is what the app actually does — face.ts averages its 8 Hz readings over
    a window before the backend fuses them — so this is the metric that predicts
    behaviour in the product rather than performance on a FER benchmark.
    """
    grouped = pd.DataFrame(probs, columns=EMOTIONS)
    grouped["key"] = frame["utterance_key"].to_numpy()
    grouped["label"] = frame["label"].to_numpy()

    pooled = grouped.groupby("key", sort=True).agg(
        {**{e: "mean" for e in EMOTIONS}, "label": "first"}
    )
    matrix = pooled[EMOTIONS].to_numpy(dtype=np.float64)
    matrix = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1e-12, None)
    return matrix, pooled["label"].to_numpy()


def prior_baseline(train: pd.DataFrame, labels: np.ndarray) -> dict:
    """Floor: predict the training class distribution for every input.

    Worth printing because a 7-class problem where neutral is half the data has a
    deceptively high accuracy floor, and a face model that fails to beat this on
    macro-F1 has learned nothing the fusion layer needs.
    """
    counts = np.bincount(train["label"].to_numpy(), minlength=len(EMOTIONS)).astype(np.float64)
    prior = counts / counts.sum()
    return score(np.tile(prior, (len(labels), 1)), labels, "prior baseline (test, clip)")


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=len(EMOTIONS)).astype(np.float64)
    counts[counts == 0] = 1.0
    return len(labels) / (len(EMOTIONS) * counts)


def normalisation(model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Pull the backbone's own mean/std, falling back to ImageNet's."""
    try:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(model_name)
        mean = np.array(getattr(processor, "image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.array(getattr(processor, "image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
        return mean, std
    except Exception as exc:  # noqa: BLE001
        print(f"  (no image processor for {model_name}: {exc}; using ImageNet stats)")
        return (
            np.array([0.485, 0.456, 0.406], dtype=np.float32),
            np.array([0.229, 0.224, 0.225], dtype=np.float32),
        )


def export_onnx(model, size: int, path: Path, mean: np.ndarray, std: np.ndarray) -> Optional[str]:
    """Export for onnxruntime-web, so the browser can run this head.

    The whole point of training this is that it ends up in face.ts, and face.ts
    runs in a tab where the video is never allowed to leave the page. An ONNX
    graph is the only artefact that satisfies both.
    """
    import torch

    model = model.eval().cpu()
    dummy = torch.zeros(1, 3, size, size, dtype=torch.float32)

    class Wrapped(torch.nn.Module):
        """Softmax inside the graph.

        face.ts consumes a probability vector, and doing the softmax in the
        browser is one more place for the train/serve mismatch that already cost
        this project half the text model's confidence once.
        """

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            return torch.softmax(self.inner(pixel_values=pixel_values).logits, dim=-1)

    common = dict(
        input_names=["pixel_values"],
        output_names=["probabilities"],
        dynamic_axes={"pixel_values": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=17,
    )
    try:
        # torch 2.9 flipped torch.onnx.export to the dynamo path by default, and
        # it handles `dynamic_axes` less predictably than the legacy tracer for
        # this graph. Ask for the tracer; fall back if the kwarg doesn't exist.
        try:
            torch.onnx.export(Wrapped(model), (dummy,), str(path), dynamo=False, **common)
        except TypeError:
            torch.onnx.export(Wrapped(model), (dummy,), str(path), **common)
    except Exception as exc:  # noqa: BLE001
        print(f"  ONNX export failed ({exc}) — the PyTorch checkpoint is still saved.")
        return None

    (path.parent / "preprocess.json").write_text(
        json.dumps(
            {
                "input_size": size,
                "layout": "NCHW",
                "channel_order": "RGB",
                "scale": 1.0 / 255.0,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "labels": EMOTIONS,
                "note": (
                    "Crop square around the detector box with 35% margin, resize to "
                    "input_size, convert BGR->RGB, scale, then (x-mean)/std. Output is "
                    "already softmaxed and indexed by `labels`, which matches "
                    "app/emotions.py EMOTIONS order."
                ),
                "fixed_spatial_dims": (
                    "Only the batch axis is dynamic; the graph declares "
                    "[batch, 3, input_size, input_size]. MobileNetV2's TF-style "
                    "padding is derived from the input shape and traces as a "
                    "constant, so height and width are frozen at export. "
                    "onnxruntime rejects any other size outright — verified — so a "
                    "mis-sized crop fails loudly instead of scoring wrongly."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  ONNX -> {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="google/mobilenet_v2_1.0_224",
        help="small enough to run in a browser tab beside the 3D scene; "
        "microsoft/resnet-18 scores a little higher and costs ~4x",
    )
    parser.add_argument("--size", type=int, default=160, help="input resolution")
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-det-score", type=float, default=0.0)
    parser.add_argument(
        "--unambiguous",
        action="store_true",
        help="ablation: keep only clips where speaker selection had no rival face",
    )
    parser.add_argument(
        "--no-class-weights", action="store_true", help="ablation: plain cross-entropy"
    )
    parser.add_argument(
        "--no-onnx", action="store_true", help="skip the browser export"
    )
    args = parser.parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError:
        raise SystemExit(
            "opencv-python is required: .venv\\Scripts\\python.exe -m pip install opencv-python"
        )

    import torch
    from torch import nn
    from transformers import AutoModelForImageClassification, Trainer, TrainingArguments

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    print(
        f"Device: {'cuda' if use_cuda else 'cpu'} | model: {args.model} | "
        f"size: {args.size} | smoothing: {args.label_smoothing}"
    )

    print("\nLoading manifests:")
    frames = {split: load_manifest(split, args) for split in ("train", "dev", "test")}
    for split, df in frames.items():
        print(
            f"  {split:<5} {len(df):>6} crops from {df['utterance_key'].nunique():>5} clips"
        )
    print("  train class balance:", dict(frames["train"]["emotion"].value_counts()))

    mean, std = normalisation(args.model)
    datasets = {
        split: FaceFrames(df, mean, std, args.size, train=(split == "train"))
        for split, df in frames.items()
    }

    model = AutoModelForImageClassification.from_pretrained(
        args.model,
        num_labels=len(EMOTIONS),
        id2label={i: e for i, e in enumerate(EMOTIONS)},
        label2id=LABEL2ID,
        # The backbone's ImageNet head has 1000-odd classes; we are replacing it.
        ignore_mismatched_sizes=True,
    )

    weights = None
    if not args.no_class_weights:
        weights = torch.tensor(
            class_weights(frames["train"]["label"].to_numpy()), dtype=torch.float
        )
        print("\nClass weights (inverse frequency):")
        for emotion, w in zip(EMOTIONS, weights.tolist()):
            print(f"  {emotion:<9} {w:.3f}")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = nn.CrossEntropyLoss(
                weight=weights.to(outputs.logits.device) if weights is not None else None,
                label_smoothing=args.label_smoothing,
            )
            loss = loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    from sklearn.metrics import classification_report, f1_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "accuracy": float((preds == labels).mean()),
        }

    steps_per_epoch = math.ceil(len(datasets["train"]) / args.batch_size)
    warmup_steps = max(1, int(0.06 * steps_per_epoch * args.epochs))
    print(f"\nSchedule: {steps_per_epoch} steps/epoch, {warmup_steps} warmup steps")

    training_args = TrainingArguments(
        output_dir=str(args.output.parent / "checkpoints-face"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=1e-4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=use_cuda,
        use_cpu=not use_cuda,
        dataloader_num_workers=args.workers,
        # Without this the workers respawn each epoch, resetting their RNG and
        # replaying the identical augmentations every time — which quietly turns
        # augmentation off after epoch one.
        dataloader_persistent_workers=args.workers > 0,
        remove_unused_columns=False,
        report_to=[],
        seed=42,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ---- calibration, fitted on dev and applied to test -------------------
    def logits_of(split: str) -> np.ndarray:
        raw = trainer.predict(datasets[split]).predictions
        # Some heads return a tuple; the logits are always first.
        return np.asarray(raw[0] if isinstance(raw, tuple) else raw, dtype=np.float64)

    dev_logits = logits_of("dev")
    dev_labels = frames["dev"]["label"].to_numpy()
    temperature = fit_temperature(dev_logits, dev_labels)
    print(f"\nFitted temperature on dev: T = {temperature:.3f}")
    if temperature > 1.05:
        print("  T > 1 means the raw head was overconfident — exactly what would have")
        print("  let this channel out-weigh text in fusion while being wrong.")

    test_logits = logits_of("test")
    test_labels = frames["test"]["label"].to_numpy()

    print("\n=== Test set ===")
    results: Dict[str, dict] = {}
    raw_frame = softmax(test_logits)
    cal_frame = softmax(test_logits / temperature)

    results["frame_raw"] = score(raw_frame, test_labels, "frame-level (raw)")
    results["frame_calibrated"] = score(cal_frame, test_labels, "frame-level (T-scaled)")

    raw_clip, clip_labels = pool_by_clip(raw_frame, frames["test"])
    cal_clip, _ = pool_by_clip(cal_frame, frames["test"])
    results["clip_raw"] = score(raw_clip, clip_labels, "clip-level (raw)")
    results["clip_calibrated"] = score(cal_clip, clip_labels, "clip-level (T-scaled)")
    results["prior"] = prior_baseline(frames["train"], clip_labels)

    print("\nPer-class breakdown, clip-level (the table for the report):")
    report_text = classification_report(
        clip_labels,
        cal_clip.argmax(axis=1),
        labels=list(range(len(EMOTIONS))),
        target_names=EMOTIONS,
        digits=3,
        zero_division=0,
    )
    print(report_text)

    # ---- save -------------------------------------------------------------
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))

    onnx_path = None
    if not args.no_onnx:
        print("\nExporting for the browser:")
        onnx_path = export_onnx(
            trainer.model, args.size, args.output / "face-emotion.onnx", mean, std
        )

    (args.output / "meld_results.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "input_size": args.size,
                "epochs": args.epochs,
                "class_weighted": not args.no_class_weights,
                "label_smoothing": args.label_smoothing,
                "unambiguous_only": args.unambiguous,
                "temperature": temperature,
                "train_crops": len(frames["train"]),
                "train_clips": int(frames["train"]["utterance_key"].nunique()),
                "test_metrics": results,
                "classification_report": report_text,
                "onnx": onnx_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved to {args.output}")
    print(
        "Nothing serves this yet — face.ts still loads face-api's stock head. "
        "Compare the numbers above against that baseline before rewiring the "
        "capture path, and keep the temperature: fusion reads this confidence."
    )


if __name__ == "__main__":
    main()
