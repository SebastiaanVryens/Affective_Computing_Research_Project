"""Runtime configuration, overridable by environment variable.

Defaults are tuned for CPU-only inference, because that's the first target.
Flipping ``MINDSCAPE_DEVICE=cuda`` is the only change needed to move the whole
sidecar onto the GPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _meld_checkpoint() -> Path:
    """Where the fine-tuned MELD head lives.

    In a packaged build the weights ship beside the app rather than in
    backend/models, so MINDSCAPE_MODEL_BUNDLE (set by the desktop launcher,
    alongside HF_HOME for the hub models) wins when it points somewhere real.
    Falls through to the dev location so nothing changes when it is unset.
    """
    bundle = os.getenv("MINDSCAPE_MODEL_BUNDLE")
    if bundle:
        candidate = Path(bundle) / "meld-text"
        if candidate.is_dir():
            return candidate
    return BACKEND_ROOT / "models" / "meld-text"


@dataclass
class Settings:
    # --- device ---------------------------------------------------------
    # "cpu" | "cuda" | "auto". "auto" picks cuda when torch can see a GPU.
    device: str = os.getenv("MINDSCAPE_DEVICE", "cpu")

    # --- text emotion ---------------------------------------------------
    # Where train_text.py writes its fine-tuned MELD checkpoint. If this
    # directory exists it wins; otherwise we fall back to the HF model below,
    # so the app is useful before you've trained anything.
    meld_text_model: Path = field(default_factory=_meld_checkpoint)
    fallback_text_model: str = os.getenv(
        "MINDSCAPE_TEXT_MODEL", "j-hartmann/emotion-english-distilroberta-base"
    )

    # Ensemble the MELD checkpoint with the general-domain model above.
    #
    # Rationale, measured rather than assumed: on a diary-style probe set the two
    # agree with intent equally often but fail on *different* sentences, and the
    # MELD model is systematically less confident on calm first-person narration
    # (its training data is acted sitcom dialogue). Averaging recovers the
    # general model's robustness without discarding the MELD-specific tuning.
    #
    # Set false to serve the MELD checkpoint alone — that's the ablation, and
    # training/eval_diary.py reports all three conditions.
    ensemble_text: bool = _env_bool("MINDSCAPE_ENSEMBLE_TEXT", True)
    # Weight on the MELD head; the remainder goes to the general model.
    ensemble_meld_weight: float = _env_float("MINDSCAPE_ENSEMBLE_W", 0.5)

    # --- speech recognition ---------------------------------------------
    # faster-whisper size. "base.en" is the sweet spot on CPU; "small.en" is
    # noticeably better on mumbling but ~3x slower.
    whisper_model: str = os.getenv("MINDSCAPE_WHISPER_MODEL", "base.en")
    # int8 keeps CPU inference tolerable; use "float16" on CUDA.
    whisper_compute_type: str = os.getenv("MINDSCAPE_WHISPER_COMPUTE", "int8")

    # --- voice emotion ---------------------------------------------------
    # The wav2vec2 SER head is the single heaviest thing here (~1.2GB, and slow
    # on CPU). It is lazy-loaded, and when disabled the audio channel falls back
    # to the prosody heuristic in audio_model.py, which costs nothing.
    enable_voice_emotion: bool = _env_bool("MINDSCAPE_VOICE_EMOTION", True)
    voice_emotion_model: str = os.getenv(
        "MINDSCAPE_VOICE_MODEL",
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    )

    # Whether the prosody heuristic's reading is allowed into the fused result
    # when the neural tier is unavailable. Off, on measurement.
    #
    # training/eval_voice.py scored it on 2,487 MELD test clips: it predicts
    # neutral on 100% of them, its per-class numbers are identical to predicting
    # the class prior, choosing the emotion from its non-neutral mass lands at
    # 17.3% against a 16.7% chance rate, and using `charge` to tell an emotional
    # clip from a neutral one gives ROC-AUC 0.474 — no signal, very slightly the
    # wrong way.
    #
    # That alone would argue for ignoring it. What makes it actively harmful is
    # the shape: it puts a mean 0.787 of its mass on neutral, so it is a *peaked*
    # distribution, not a flat one. fusion.py weights each channel by the entropy
    # of its reading, which rewards confidence — so an uninformative channel that
    # is reliably confident earns real weight and drags every entry toward
    # neutral. A uniform vector would be harmless; this is not that.
    #
    # With this off the channel reports `available: false` and fusion
    # renormalises over the channels that did say something, exactly as it
    # already does when no face is visible. The tier that ran is still tagged in
    # the response, so the heuristic stays visible in diagnostics.
    #
    # Set MINDSCAPE_PROSODY_FUSION=true to restore the old behaviour, which is
    # also how to run the ablation. Note the measurement is on MELD — acted,
    # laugh-tracked, ~3.5s per clip — and quiet diary audio may suit prosody
    # better. If you re-measure and it helps there, flip this back.
    prosody_in_fusion: bool = _env_bool("MINDSCAPE_PROSODY_FUSION", False)

    # --- fusion ----------------------------------------------------------
    # Late-fusion weights per modality. Text leads because the MELD-trained head
    # is the best-calibrated of the three; face is a close second but drifts on
    # a resting face; voice is noisiest, so it mostly breaks ties.
    # Measured, not assumed. training/eval_fusion.py swept these on MELD and
    # CMU-MOSI; both corpora put the optimum at roughly 0.9 text / 0.1 face /
    # 0.0 voice, and both put the *shipped* 0.5/0.3/0.2 well below serving text
    # alone (MELD -0.021 wF1, MOSI -0.102).
    #
    # The reason is unequal channel quality rather than anything wrong with the
    # fusion arithmetic. On MOSI, text scores 0.758 weighted-F1, face 0.497 and
    # voice 0.413; giving half the mass to channels roughly half as good dilutes
    # the one that is right. Entropy weighting does not rescue it, because after
    # calibration the weak channels are *well* calibrated — confidently mediocre
    # is exactly what a certainty-weighted pool rewards.
    #
    # Note this governs the committed entry only. The face channel still drives
    # the live world through mood.pushFace(), where it is 99.4% reliable on
    # webcam-style video and doing a different job entirely.
    #
    # Both sweeps actually put the optimum at 0.9 / 0.1 / 0.0. These defaults
    # stop short of that deliberately: a hard zero would make a channel
    # structurally unable to contribute no matter what model is behind it, which
    # is a permanent answer to a question that is still open — the voice channel
    # is already silenced at load time, and a better SER checkpoint is a known
    # next step. Set MINDSCAPE_W_VOICE=0 to reproduce the swept optimum.
    weight_text: float = _env_float("MINDSCAPE_W_TEXT", 0.8)
    weight_face: float = _env_float("MINDSCAPE_W_FACE", 0.15)
    weight_voice: float = _env_float("MINDSCAPE_W_VOICE", 0.05)

    # How independent the three channels are treated as being, in [0, 1].
    #
    # 0   = plain weighted averaging. Agreement between channels cannot raise
    #       confidence above the best single channel. This is the ablation.
    # 1   = fully independent sensors; agreeing channels multiply, which sharpens
    #       aggressively. Overconfident here, because face, voice and words from
    #       one person in one moment share a cause and are plainly correlated.
    # 0.5 = the default. Agreement is rewarded and disagreement penalised, at
    #       roughly half the strength true independence would imply.
    #
    # Now measured, and the answer is 0. The sweep in training/eval_fusion.py is
    # monotonic on both corpora — every step away from plain weighted averaging
    # costs accuracy. On MOSI: 0.744 at independence 0, falling to 0.644 at 1.0,
    # with the old 0.5 default sitting at 0.657. On MELD the spread is smaller
    # (0.0007) but points the same way.
    #
    # The reasoning behind the default was that agreeing channels should sharpen
    # each other. It holds only when the channels are of comparable quality; here
    # a weak channel agreeing with a strong one adds confidence without adding
    # information, and a weak channel disagreeing destroys it.
    fusion_independence: float = _env_float("MINDSCAPE_INDEPENDENCE", 0.0)

    # --- server ----------------------------------------------------------
    cors_origins: list[str] = field(
        default_factory=lambda: os.getenv(
            "MINDSCAPE_CORS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
    )

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def has_meld_checkpoint(self) -> bool:
        return (self.meld_text_model / "config.json").exists()


settings = Settings()
