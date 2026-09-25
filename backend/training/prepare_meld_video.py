"""Turn MELD's raw video into a face-crop dataset keyed to the text splits.

    python -m training.prepare_meld           # once: the CSVs
    python -m training.prepare_meld_video --source D:\\MELD.Raw

prepare_meld.py deliberately skips the video ("we only need the CSVs"). For the
*text* channel that was right. For the face channel it is the whole dataset, and
it is the only public corpus that gives us faces in MELD's own seven labels, on
MELD's own train/dev/test boundaries. That last property is what makes an
end-to-end fusion ablation possible: face and text can be scored on the same
held-out utterances instead of on anecdotes.

Three things here deserve scrutiny in a write-up, because each one injects noise:

*Utterance labels applied to frames.* MELD annotates an utterance, not a frame.
A neutral-looking frame inside an angry utterance is simply mislabelled. We
mitigate by sampling from the middle of the clip (expressions peak mid-utterance,
and the edges are contaminated by the neighbouring turns) and by pooling frames
per clip at evaluation time — but the training labels remain weak. Report this.

*Speaker identification.* MELD clips are multi-party Friends scenes; the largest
face is not always the one talking. We link detections into tracks across the
sampled frames and keep the track that is large, central and persistent. It is a
heuristic, and ``--report-multiface`` tells you how often the clip had a second
plausible candidate, which is the honest error bar on it.

*Coverage is class-dependent.* Clips where no face is found are dropped, and that
loss is not uniform across emotions. The per-class coverage table at the end is
therefore part of the result, not diagnostics — a face channel that silently sees
disgust less often than joy has a bias that no amount of training fixes.

Writes data/meld_faces/{split}/*.jpg plus a manifest CSV per split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import BACKEND_ROOT, PROJECT_ROOT  # noqa: E402
from app.emotions import EMOTIONS  # noqa: E402

CSV_DIR = PROJECT_ROOT / "data" / "meld"
OUT_DIR = PROJECT_ROOT / "data" / "meld_faces"

# YuNet: a 337KB face detector that ships in OpenCV's model zoo. Chosen over the
# bundled Haar cascade because Friends is shot with people turned towards each
# other rather than towards the camera, and Haar only reliably finds a frontal
# face — which would bias the dataset towards exactly the posed-looking frames
# we are trying to move away from.
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_PATH = BACKEND_ROOT / "models" / "_detectors" / "face_detection_yunet.onnx"

MANIFEST_COLUMNS = [
    "split",
    "dialogue_id",
    "utterance_id",
    "emotion",
    "clip",
    "path",
    "frame_index",
    "t",
    "det_score",
    "area_frac",
    "centrality",
    "track_frames",
    "rival_faces",
]


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


@dataclass
class Detection:
    """One face in one frame. Box is (x, y, w, h) in pixels."""

    box: Tuple[float, float, float, float]
    score: float
    landmarks: Optional[np.ndarray] = None  # 5x2: eyes, nose, mouth corners

    @property
    def area(self) -> float:
        return max(0.0, self.box[2]) * max(0.0, self.box[3])

    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2.0, y + h / 2.0


def download_yunet() -> Path:
    if YUNET_PATH.exists() and YUNET_PATH.stat().st_size > 0:
        return YUNET_PATH
    YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching face detector -> {YUNET_PATH}")
    try:
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    except Exception as exc:  # noqa: BLE001
        YUNET_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download YuNet from {YUNET_URL} ({exc}). "
            f"Download it manually to {YUNET_PATH}, or pass --detector haar."
        ) from exc
    return YUNET_PATH


class FaceDetector:
    """YuNet if we can get it, Haar if we can't.

    The two are not interchangeable in quality — Haar misses roughly the frames
    YuNet is here for — so which one ran is recorded in the run summary rather
    than left implicit.
    """

    def __init__(self, kind: str, score_threshold: float) -> None:
        import cv2

        self.cv2 = cv2
        self.kind = kind
        self.score_threshold = score_threshold
        self._size: Tuple[int, int] = (0, 0)

        if kind == "yunet":
            self._model = cv2.FaceDetectorYN.create(
                str(download_yunet()),
                "",
                (320, 320),
                score_threshold,
                0.3,  # NMS
                50,  # top_k
            )
        else:
            cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._model = cv2.CascadeClassifier(str(cascade))
            if self._model.empty():
                raise RuntimeError(f"OpenCV cascade failed to load from {cascade}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        height, width = frame.shape[:2]

        if self.kind == "haar":
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            boxes = self._model.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            # Haar reports no confidence, so everything it returns scores 1.0.
            # That makes det_score meaningless in a Haar run; the manifest keeps
            # the column so the two runs stay schema-compatible.
            return [Detection((float(x), float(y), float(w), float(h)), 1.0) for x, y, w, h in boxes]

        if (width, height) != self._size:
            self._model.setInputSize((width, height))
            self._size = (width, height)

        _, raw = self._model.detect(frame)
        if raw is None:
            return []

        found = []
        for row in raw:
            box = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            landmarks = np.array(row[4:14], dtype=np.float32).reshape(5, 2)
            found.append(Detection(box, float(row[14]), landmarks))
        return found


# --------------------------------------------------------------------------
# track selection — "which of these faces is the one talking?"
# --------------------------------------------------------------------------


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    overlap = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


@dataclass
class Track:
    """A face followed across the sampled frames of one clip."""

    frames: List[int] = field(default_factory=list)
    detections: List[Detection] = field(default_factory=list)

    def add(self, frame_index: int, detection: Detection) -> None:
        self.frames.append(frame_index)
        self.detections.append(detection)

    @property
    def last_box(self) -> Tuple[float, float, float, float]:
        return self.detections[-1].box


def link_tracks(per_frame: List[List[Detection]], iou_threshold: float = 0.3) -> List[Track]:
    """Greedy IoU association across sampled frames.

    Sampled frames sit a few hundred milliseconds apart, so a face moves a little
    but rarely leaves its own box — greedy IoU is sufficient and costs nothing.
    A shot change inside the clip breaks the track, which is correct behaviour
    here: after a cut it genuinely is a different view, and we would rather end
    up with two short tracks than one track that silently spans two people.
    """
    tracks: List[Track] = []
    for frame_index, detections in enumerate(per_frame):
        claimed: set[int] = set()
        for detection in sorted(detections, key=lambda d: d.area, reverse=True):
            best, best_iou = None, iou_threshold
            for index, track in enumerate(tracks):
                if index in claimed or track.frames[-1] == frame_index:
                    continue
                overlap = iou(track.last_box, detection.box)
                if overlap >= best_iou:
                    best, best_iou = index, overlap
            if best is None:
                tracks.append(Track())
                tracks[-1].add(frame_index, detection)
            else:
                tracks[best].add(frame_index, detection)
                claimed.add(best)
    return tracks


def track_score(track: Track, shape: Tuple[int, int], n_sampled: int) -> float:
    """How likely this track is to be the speaker.

    Three factors, multiplied so that a track has to satisfy all of them:

    *Size.* The speaker is usually framed larger than whoever is reacting.
    *Centrality.* Friends puts the talker near the middle of the frame; extras
    and half-visible shoulders sit at the edges.
    *Persistence.* A face present in two of eight sampled frames is somebody
    walking past. The speaker is on screen for most of their own utterance.

    Deliberately not mouth motion: that is the principled signal, but estimating
    it from five landmarks at 8 samples per clip was noisier in practice than the
    geometry, and a weak active-speaker signal that *looks* principled is worse
    than an honest heuristic.
    """
    height, width = shape
    mean_area = float(np.mean([d.area for d in track.detections])) / (width * height)

    centers = np.array([d.center for d in track.detections], dtype=np.float32)
    offsets = np.abs(centers - np.array([width / 2.0, height / 2.0], dtype=np.float32))
    # Normalised to [0, 1] where 1 is dead centre.
    centrality = float(
        np.mean(1.0 - np.clip(offsets / np.array([width / 2.0, height / 2.0]), 0, 1).mean(axis=1))
    )

    persistence = len(track.frames) / max(1, n_sampled)
    return mean_area * centrality * persistence


# --------------------------------------------------------------------------
# clip processing
# --------------------------------------------------------------------------


def sample_indices(total_frames: int, n: int, trim: float) -> List[int]:
    """Evenly spaced frame indices from the middle ``1 - 2*trim`` of the clip.

    The edges are trimmed because MELD's clip boundaries are cut on the audio
    turn, not the expression: the opening frames still carry the *previous*
    speaker's reaction and the closing frames have already begun the next one.
    Sampling them would label another person's expression with this utterance.
    """
    if total_frames <= 0:
        return []
    lo = int(total_frames * trim)
    hi = max(lo + 1, int(total_frames * (1.0 - trim)))
    if hi - lo < n:
        return list(range(lo, hi))
    return [int(round(x)) for x in np.linspace(lo, hi - 1, n)]


def crop_face(
    frame: np.ndarray, detection: Detection, margin: float, size: int, cv2
) -> Optional[np.ndarray]:
    """Square crop around the box with ``margin`` padding, resized to ``size``.

    Squared before padding so the aspect ratio never distorts, and clamped to the
    frame afterwards. A crop that ends up mostly off-screen is rejected rather
    than letterboxed — a half-black training image teaches the model about the
    edge of the shot, not about the face.
    """
    height, width = frame.shape[:2]
    x, y, w, h = detection.box
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + margin)

    x0, y0 = cx - side / 2.0, cy - side / 2.0
    x1, y1 = cx + side / 2.0, cy + side / 2.0

    cx0, cy0 = int(max(0, round(x0))), int(max(0, round(y0)))
    cx1, cy1 = int(min(width, round(x1))), int(min(height, round(y1)))
    if cx1 - cx0 < 24 or cy1 - cy0 < 24:
        return None

    kept = ((cx1 - cx0) * (cy1 - cy0)) / max(1.0, side * side)
    if kept < 0.6:
        return None

    crop = frame[cy0:cy1, cx0:cx1]
    # INTER_AREA downscales without the aliasing that INTER_LINEAR leaves on
    # high-frequency detail like eyelashes and teeth.
    interpolation = cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(crop, (size, size), interpolation=interpolation)


@dataclass
class ClipResult:
    rows: List[dict] = field(default_factory=list)
    reason: Optional[str] = None
    # Every sampled frame, written whole and independent of whether a face was
    # found. The baseline evaluation needs these: face-api runs its *own*
    # detector, and scoring it only on frames our detector liked would hand it a
    # pre-filtered test set and call the comparison fair.
    frame_rows: List[dict] = field(default_factory=list)


def process_clip(
    clip: Path,
    detector: FaceDetector,
    args: argparse.Namespace,
    out_dir: Path,
    meta: dict,
) -> ClipResult:
    cv2 = detector.cv2

    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        return ClipResult(reason="unreadable")

    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

        frames: List[np.ndarray] = []
        kept_indices: List[int] = []

        if total > 0:
            wanted = sample_indices(total, args.frames, args.trim)
            if not wanted:
                return ClipResult(reason="empty")
            for index in wanted:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if ok and frame is not None:
                    frames.append(frame)
                    kept_indices.append(index)
        else:
            # Some MELD clips carry a broken frame count. Decoding the whole clip
            # and subsampling afterwards is slower, but it is the difference
            # between dropping those utterances and keeping them.
            decoded: List[np.ndarray] = []
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                decoded.append(frame)
            for index in sample_indices(len(decoded), args.frames, args.trim):
                frames.append(decoded[index])
                kept_indices.append(index)

        if not frames:
            return ClipResult(reason="no-frames")

        # Saved before detection runs, so the set is detector-independent.
        frame_rows: List[dict] = []
        if args.save_frames:
            frame_dir = out_dir.parent / f"{meta['split']}_frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            for position, frame in enumerate(frames):
                height, width = frame.shape[:2]
                if width > args.frame_width:
                    scale = args.frame_width / width
                    frame_out = cv2.resize(
                        frame,
                        (args.frame_width, int(round(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    frame_out = frame
                index = kept_indices[position]
                name = f"dia{meta['dialogue_id']}_utt{meta['utterance_id']}_f{index}.jpg"
                cv2.imwrite(
                    str(frame_dir / name), frame_out, [cv2.IMWRITE_JPEG_QUALITY, args.quality]
                )
                frame_rows.append(
                    {
                        **meta,
                        "path": f"{frame_dir.name}/{name}",
                        "frame_index": index,
                        "t": round(index / fps, 3),
                        "width": frame_out.shape[1],
                        "height": frame_out.shape[0],
                    }
                )

        per_frame = [detector.detect(frame) for frame in frames]
        if not any(per_frame):
            return ClipResult(reason="no-face", frame_rows=frame_rows)

        shape = frames[0].shape[:2]
        tracks = link_tracks(per_frame)
        ranked = sorted(
            tracks, key=lambda t: track_score(t, shape, len(frames)), reverse=True
        )
        speaker = ranked[0]
        if len(speaker.frames) < args.min_track:
            return ClipResult(reason="track-too-short", frame_rows=frame_rows)

        # How many other tracks were within 70% of the winner's score — i.e. how
        # often the speaker choice was a genuine coin-flip. Carried per row so a
        # later run can train on the unambiguous subset and compare.
        best = track_score(speaker, shape, len(frames))
        rivals = sum(
            1
            for t in ranked[1:]
            if best > 0 and track_score(t, shape, len(frames)) / best > 0.7
        )

        rows: List[dict] = []
        for frame_position, detection in zip(speaker.frames, speaker.detections):
            crop = crop_face(frames[frame_position], detection, args.margin, args.size, cv2)
            if crop is None:
                continue

            frame_index = kept_indices[frame_position]
            name = f"dia{meta['dialogue_id']}_utt{meta['utterance_id']}_f{frame_index}.jpg"
            path = out_dir / name
            cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, args.quality])

            height, width = shape
            cx, cy = detection.center
            rows.append(
                {
                    **meta,
                    "clip": clip.name,
                    "path": f"{out_dir.name}/{name}",
                    "frame_index": frame_index,
                    "t": round(frame_index / fps, 3),
                    "det_score": round(detection.score, 4),
                    "area_frac": round(detection.area / (width * height), 5),
                    "centrality": round(
                        1.0
                        - float(
                            np.mean(
                                [
                                    abs(cx - width / 2.0) / (width / 2.0),
                                    abs(cy - height / 2.0) / (height / 2.0),
                                ]
                            )
                        ),
                        4,
                    ),
                    "track_frames": len(speaker.frames),
                    "rival_faces": rivals,
                }
            )

        return ClipResult(
            rows=rows,
            reason=None if rows else "crop-rejected",
            frame_rows=frame_rows,
        )
    finally:
        capture.release()


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# parallel workers
# --------------------------------------------------------------------------

# Per-process state. The detector holds a native cv2 object that cannot be
# pickled, so each worker builds its own once in the initializer rather than
# receiving one through the task queue.
_WORKER: Dict[str, object] = {}


def _init_worker(kind: str, det_score: float) -> None:
    _WORKER["detector"] = FaceDetector(kind, det_score)


def _process_one_serial(task, detector: "FaceDetector"):
    """Same contract as _process_one, with the detector passed in (--workers 1)."""
    clip, meta, args, out_dir = task
    try:
        result = process_clip(clip, detector, args, out_dir, meta)
        return result.rows, result.reason, result.frame_rows, None
    except Exception as exc:  # noqa: BLE001
        return [], f"error:{type(exc).__name__}", [], f"{clip.name}: {exc}"


def _process_one(task):
    """Run one clip in a worker. Returns plain data — ClipResult stays local."""
    clip, meta, args, out_dir = task
    detector = _WORKER["detector"]
    try:
        result = process_clip(clip, detector, args, out_dir, meta)
        return result.rows, result.reason, result.frame_rows, None
    except Exception as exc:  # noqa: BLE001
        return [], f"error:{type(exc).__name__}", [], f"{clip.name}: {exc}"


def index_clips(source: Path) -> Dict[str, Path]:
    """Map ``dia3_utt7.mp4`` -> its path, wherever it sits under ``source``.

    MELD.Raw has been repackaged with different directory names across releases
    (``train_splits``, ``dev_splits_complete``, ``output_repeated_splits_test``),
    so we glob for the filenames instead of hardcoding a layout that only works
    for whichever tarball we happened to test against.
    """
    index: Dict[str, Path] = {}
    for path in source.rglob("*.mp4"):
        index.setdefault(path.name.lower(), path)
    return index


def coverage_report(df: pd.DataFrame, utterances: pd.DataFrame, split: str) -> dict:
    """Per-class coverage — the part of this that is a finding, not a log line."""
    print(f"\n{split} coverage (utterances with a usable face):")
    covered = df.groupby("emotion")["utterance_key"].nunique() if len(df) else pd.Series(dtype=int)
    wanted = utterances["Emotion"].value_counts()

    stats = {}
    for emotion in EMOTIONS:
        total = int(wanted.get(emotion, 0))
        have = int(covered.get(emotion, 0))
        pct = 100.0 * have / total if total else 0.0
        bar = "#" * int(round(pct / 4))
        print(f"  {emotion:<9} {have:>5}/{total:<5} ({pct:5.1f}%) {bar}")
        stats[emotion] = {"utterances": total, "covered": have, "pct": round(pct, 2)}

    spread = [s["pct"] for s in stats.values() if s["utterances"]]
    if spread and max(spread) - min(spread) > 10:
        print(
            f"  ! coverage varies {max(spread) - min(spread):.1f} points across classes — "
            "the face channel sees some emotions less often than others by construction."
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Directory holding MELD's extracted .mp4 clips (searched recursively). "
        "Get it from the MELD.Raw tarball linked in the declare-lab/MELD README.",
    )
    parser.add_argument("--splits", default="train,dev,test")
    parser.add_argument("--frames", type=int, default=8, help="frames sampled per clip")
    parser.add_argument(
        "--trim",
        type=float,
        default=0.15,
        help="fraction of the clip discarded at each end (neighbouring turns bleed in)",
    )
    parser.add_argument(
        "--size", type=int, default=160, help="saved crop size; train at or below this"
    )
    parser.add_argument("--margin", type=float, default=0.35, help="padding around the box")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality")
    parser.add_argument(
        "--min-track",
        type=int,
        default=3,
        help="drop the clip unless the speaker track survives this many sampled frames",
    )
    parser.add_argument(
        "--detector", default="yunet", choices=["yunet", "haar"], help="haar needs no download"
    )
    parser.add_argument("--det-score", type=float, default=0.6)
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="also write every sampled frame whole, for the face-api baseline "
        "(it runs its own detector, so it needs un-pre-filtered input)",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=640,
        help="downscale saved frames to this width; 640 matches the webcam face.ts sees",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 2)),
        help="parallel clip workers; 1 runs in-process. Decode and detection are "
        "CPU-bound and independent per clip, so this is close to linear",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="stop after N clips per split (smoke test)"
    )
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError:
        raise SystemExit(
            "opencv-python is required: .venv\\Scripts\\python.exe -m pip install opencv-python"
        )

    if not args.source.is_dir():
        raise SystemExit(f"--source {args.source} is not a directory")

    print(f"Indexing clips under {args.source} ...")
    clips = index_clips(args.source)
    print(f"  found {len(clips)} .mp4 files")
    if not clips:
        raise SystemExit("No .mp4 files found. Point --source at the extracted MELD.Raw.")

    # Make sure the detector weights are on disk before any worker starts:
    # eight processes racing to download the same file is a corrupt file.
    if args.detector == "yunet":
        download_yunet()

    # Only built in the parent for the serial path; workers build their own.
    detector = FaceDetector(args.detector, args.det_score) if args.workers <= 1 else None
    print(f"Detector: {args.detector} | workers: {args.workers}")
    args.output.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, dict] = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        csv_path = CSV_DIR / f"{split}.csv"
        if not csv_path.exists():
            raise SystemExit(f"{csv_path} missing. Run: python -m training.prepare_meld")

        utterances = pd.read_csv(csv_path)
        if args.limit:
            utterances = utterances.head(args.limit)

        split_dir = args.output / split
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {split}: {len(utterances)} utterances ===")
        rows: List[dict] = []
        frame_rows: List[dict] = []
        misses: Dict[str, int] = {}

        tasks = []
        for row in utterances.itertuples(index=False):
            dialogue, utterance = int(row.Dialogue_ID), int(row.Utterance_ID)
            clip = clips.get(f"dia{dialogue}_utt{utterance}.mp4")
            if clip is None:
                misses["no-clip"] = misses.get("no-clip", 0) + 1
                continue
            tasks.append(
                (
                    clip,
                    {
                        "split": split,
                        "dialogue_id": dialogue,
                        "utterance_id": utterance,
                        "emotion": str(row.Emotion),
                    },
                    args,
                    split_dir,
                )
            )

        started = time.time()
        reported: set[str] = set()

        def absorb(position: int, payload) -> None:
            clip_rows, reason, whole, detail = payload
            if reason:
                # One bad clip shouldn't end a run over 10k of them — but a
                # silent tally of "error:AttributeError" is useless when the
                # cause is systematic, so the first of each kind is printed.
                if detail and reason not in reported:
                    reported.add(reason)
                    print(f"  ! {detail}")
                misses[reason] = misses.get(reason, 0) + 1
            rows.extend(clip_rows)
            frame_rows.extend(whole)
            if position % 250 == 0:
                rate = position / max(1e-9, time.time() - started)
                left = (len(tasks) - position) / max(1e-9, rate)
                print(
                    f"  {position}/{len(tasks)} clips -> {len(rows)} crops  "
                    f"({rate:.1f} clips/s, ~{left / 60:.0f} min left)"
                )

        if args.workers > 1:
            # Video decode and detection are both CPU-bound and wholly
            # independent per clip, so this scales close to linearly until disk
            # write bandwidth becomes the limit.
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

        frame_df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
        if len(frame_df):
            frame_df["utterance_key"] = (
                frame_df["dialogue_id"].astype(str) + "_" + frame_df["utterance_id"].astype(str)
            )

        manifest = args.output / f"{split}.csv"
        frame_df.to_csv(manifest, index=False)

        if args.save_frames:
            whole = pd.DataFrame(frame_rows)
            whole["utterance_key"] = (
                whole["dialogue_id"].astype(str) + "_" + whole["utterance_id"].astype(str)
            )
            whole_path = args.output / f"{split}_frames.csv"
            whole.to_csv(whole_path, index=False)
            print(
                f"  {len(whole)} whole frames from {whole['utterance_key'].nunique()} "
                f"clips -> {whole_path}"
            )

        clips_kept = frame_df["utterance_key"].nunique() if len(frame_df) else 0
        print(f"\n  {len(frame_df)} crops from {clips_kept} clips -> {manifest}")
        if misses:
            print("  dropped:", ", ".join(f"{k}={v}" for k, v in sorted(misses.items())))

        summary[split] = {
            "utterances": len(utterances),
            "clips_with_faces": int(clips_kept),
            "crops": len(frame_df),
            "dropped": misses,
            "coverage": coverage_report(frame_df, utterances, split),
        }

    (args.output / "prepare_summary.json").write_text(
        json.dumps(
            {
                "detector": args.detector,
                "frames_per_clip": args.frames,
                "trim": args.trim,
                "crop_size": args.size,
                "margin": args.margin,
                "min_track": args.min_track,
                "splits": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nDone. Next: python -m training.train_face --device cuda")


if __name__ == "__main__":
    main()
