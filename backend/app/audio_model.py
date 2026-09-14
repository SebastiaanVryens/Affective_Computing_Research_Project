"""Voice emotion from the raw waveform.

Two tiers, because the good model is expensive on CPU:

1. ``wav2vec2`` SER head (RAVDESS-trained, 8 classes) when enabled and loadable.
   Its labels map onto MELD's seven with only one collapse: RAVDESS "calm"
   folds into neutral.
2. A prosody heuristic that needs nothing but numpy. It reads energy, pitch
   variability and voiced-fraction into an arousal/valence guess. It is *not* a
   classifier and shouldn't be reported as one in the write-up — it exists so
   the voice channel still contributes something when the model is off.

The tier that actually ran is reported back in the response as ``source``, so
the UI (and your evaluation) can tell them apart.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple

import numpy as np

from .audio_io import SAMPLE_RATE
from .config import settings
from .emotions import EMOTIONS, from_label_scores, normalize, uniform_vector

log = logging.getLogger(__name__)

# RAVDESS (what the wav2vec2 head was trained on) -> MELD. "calm" has no MELD
# counterpart and is the closest thing RAVDESS has to a resting state.
RAVDESS_TO_MELD = {
    "angry": "anger",
    "calm": "neutral",
    "disgust": "disgust",
    "fearful": "fear",
    "happy": "joy",
    "neutral": "neutral",
    "sad": "sadness",
    "surprised": "surprise",
}

# Below this, there isn't enough speech to say anything about prosody.
MIN_DURATION_S = 0.6


class VoiceEmotionModel:
    def __init__(self) -> None:
        self._model = None
        self._extractor = None
        self._id2label: dict[int, str] = {}
        self._lock = threading.Lock()
        self._loading = False
        self._load_error: Optional[str] = None

    def _ensure_loaded(self) -> bool:
        """True only if the model is ready *right now*.

        Deliberately never blocks. This head is ~1GB and downloads on first use,
        and a live diary chunk arrives every three seconds — if the first one
        waited for that download, it would time out and every chunk behind it
        would queue on the same lock. The symptom is that live transcription
        appears completely dead for the first several minutes.

        So a request that arrives before the model is ready starts the load in
        the background, uses the prosody heuristic for now, and picks up the
        neural head on a later chunk once it's actually available.
        """
        if not settings.enable_voice_emotion:
            return False
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False

        self.start_loading()
        return self._model is not None

    def start_loading(self) -> None:
        """Kick off the background load if it isn't already running or done."""
        if self._model is not None or self._load_error is not None:
            return
        with self._lock:
            if self._loading or self._model is not None:
                return
            self._loading = True
        threading.Thread(target=self._load, daemon=True, name="voice-model").start()

    def _load(self) -> None:
        """Runs on a daemon thread; never raises into a request."""
        try:
            from transformers import (
                AutoFeatureExtractor,
                AutoModelForAudioClassification,
            )

            name = settings.voice_emotion_model
            log.info(
                "Loading voice emotion model %s in background "
                "(first run downloads ~1GB; prosody is used until it lands)",
                name,
            )
            extractor = AutoFeatureExtractor.from_pretrained(name)
            model = AutoModelForAudioClassification.from_pretrained(name)
            model.eval()
            model.to(settings.resolved_device())

            # Published last, and only once fully built: `predict` checks
            # `self._model` without the lock, so a half-initialised model must
            # never be visible to it.
            self._extractor = extractor
            self._id2label = {int(k): v for k, v in model.config.id2label.items()}
            self._model = model
            log.info("Voice emotion model ready: %s", sorted(self._id2label.values()))
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            log.error("Voice emotion model unavailable, using prosody: %s", exc)
        finally:
            self._loading = False

    def predict(self, waveform: Optional[np.ndarray]) -> Tuple[List[float], str]:
        """Return (canonical vector, source tag)."""
        if waveform is None or waveform.size < MIN_DURATION_S * SAMPLE_RATE:
            return uniform_vector(), "insufficient-audio"

        if self._ensure_loaded():
            vector = self._predict_neural(waveform)
            if vector is not None:
                return vector, "wav2vec2-ser"

        return prosody_vector(waveform), "prosody-heuristic"

    def _predict_neural(self, waveform: np.ndarray) -> Optional[List[float]]:
        try:
            import torch

            # The SER head was trained on short clips; on a long diary entry the
            # pooled representation washes out. Score 4s windows and average,
            # which also caps memory on CPU.
            windows = _windows(waveform, window_s=4.0, hop_s=2.0)
            device = settings.resolved_device()
            accumulated = np.zeros(len(EMOTIONS), dtype=np.float64)

            for window in windows:
                inputs = self._extractor(
                    window, sampling_rate=SAMPLE_RATE, return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self._model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                scores: dict[str, float] = {}
                for idx, prob in enumerate(probs):
                    raw_label = self._id2label.get(idx, "").lower()
                    meld_label = RAVDESS_TO_MELD.get(raw_label)
                    if meld_label:
                        scores[meld_label] = scores.get(meld_label, 0.0) + float(prob)
                accumulated += np.asarray(from_label_scores(scores))

            return normalize(accumulated / max(1, len(windows)))
        except Exception as exc:  # noqa: BLE001
            log.error("Voice inference failed, falling back to prosody: %s", exc)
            return None


def _windows(waveform: np.ndarray, window_s: float, hop_s: float) -> List[np.ndarray]:
    size = int(window_s * SAMPLE_RATE)
    hop = int(hop_s * SAMPLE_RATE)
    if waveform.size <= size:
        return [waveform]
    return [waveform[i : i + size] for i in range(0, waveform.size - size + 1, hop)]


# --------------------------------------------------------------------------
# Prosody heuristic
# --------------------------------------------------------------------------


def prosody_features(waveform: np.ndarray) -> dict:
    """Energy, pitch and rhythm descriptors used by the heuristic.

    These are also worth logging per entry in their own right — they're the kind
    of low-level acoustic features MELD's audio baseline uses openSMILE for, and
    they make a reasonable appendix table.
    """
    frame = int(0.025 * SAMPLE_RATE)  # 25 ms
    hop = int(0.010 * SAMPLE_RATE)  # 10 ms
    if waveform.size < frame:
        return {"rms": 0.0, "rms_var": 0.0, "pitch_mean": 0.0, "pitch_var": 0.0, "voiced": 0.0}

    frames = np.lib.stride_tricks.sliding_window_view(waveform, frame)[::hop]
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

    # Voiced frames are the ones meaningfully above the noise floor.
    floor = np.percentile(rms, 20)
    voiced_mask = rms > max(floor * 2.0, 0.01)
    voiced_ratio = float(voiced_mask.mean())

    pitches = _autocorr_pitch(frames[voiced_mask]) if voiced_mask.any() else np.array([])
    pitches = pitches[(pitches > 60) & (pitches < 400)]  # plausible speech F0

    loud = rms[voiced_mask] if voiced_mask.any() else rms
    return {
        "rms": float(np.mean(loud)),
        "rms_var": float(np.std(loud)),
        "pitch_mean": float(np.mean(pitches)) if pitches.size else 0.0,
        "pitch_var": float(np.std(pitches)) if pitches.size else 0.0,
        "voiced": voiced_ratio,
    }


def _autocorr_pitch(frames: np.ndarray) -> np.ndarray:
    """Per-frame F0 by autocorrelation peak. Rough, but cheap and dependency-free."""
    if frames.size == 0:
        return np.array([])
    min_lag = SAMPLE_RATE // 400
    max_lag = SAMPLE_RATE // 60

    out = []
    for frame in frames[:: max(1, len(frames) // 200)]:  # subsample for speed
        frame = frame - frame.mean()
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        segment = corr[min_lag:max_lag]
        if segment.size == 0 or corr[0] <= 1e-9:
            continue
        lag = int(np.argmax(segment)) + min_lag
        if corr[lag] / corr[0] > 0.3:  # require a real periodic peak
            out.append(SAMPLE_RATE / lag)
    return np.asarray(out)


def prosody_vector(waveform: np.ndarray) -> List[float]:
    """Map prosody onto the MELD classes via a soft arousal/valence guess.

    Loud + pitch-variable reads as high arousal; quiet + monotone as low. Valence
    is the genuinely weak axis here — acoustics alone barely separate anger from
    joy — so the heuristic deliberately keeps a lot of mass on neutral and lets
    the text channel resolve the sign.
    """
    f = prosody_features(waveform)
    if f["voiced"] < 0.1:
        return uniform_vector()

    # Normalise into rough 0..1 bands calibrated on conversational speech.
    energy = _band(f["rms"], 0.01, 0.15)
    pitch_range = _band(f["pitch_var"], 5.0, 60.0)
    pitch_height = _band(f["pitch_mean"], 90.0, 240.0)
    dynamics = _band(f["rms_var"], 0.005, 0.08)

    arousal = 0.4 * energy + 0.3 * pitch_range + 0.2 * dynamics + 0.1 * pitch_height

    scores = {
        # Low arousal pulls toward the flat and the heavy.
        "neutral": 0.5 + 0.5 * (1 - abs(arousal - 0.45) * 2),
        "sadness": 0.6 * max(0.0, 0.45 - arousal) * 2,
        # High arousal splits between the loud negative and the bright positive;
        # without lexical content we can't pick, so both get mass.
        "anger": 0.5 * max(0.0, arousal - 0.55) * 2 * energy,
        "joy": 0.5 * max(0.0, arousal - 0.5) * 2 * pitch_range,
        "fear": 0.3 * max(0.0, arousal - 0.6) * 2 * pitch_height,
        "surprise": 0.3 * max(0.0, pitch_range - 0.7) * 2,
        "disgust": 0.05,
    }
    return from_label_scores({k: max(0.0, v) for k, v in scores.items()})


def _band(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(min(1.0, max(0.0, (value - low) / (high - low))))


voice_model = VoiceEmotionModel()
