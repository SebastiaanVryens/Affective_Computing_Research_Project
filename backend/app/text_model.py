"""Text emotion classification over the MELD label space.

Two checkpoints can back this:

* ``backend/models/meld-text`` — what ``training/train_text.py`` produces. Its
  label order already matches ``emotions.EMOTIONS``, so no remapping is needed.
* ``j-hartmann/emotion-english-distilroberta-base`` — the fallback. It happens
  to emit exactly MELD's seven labels (under different index order), so it maps
  cleanly and lets the app run before you've trained anything.

Whichever loads, the public surface is the same: text in, a canonical
seven-element probability vector out.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import List, Optional

from .config import settings
from .emotions import EMOTIONS, from_label_scores, normalize, uniform_vector

log = logging.getLogger(__name__)

# Utterances shorter than this carry almost no lexical emotion signal; the
# classifier will happily hallucinate confident nonsense on "ok" or "hmm", so we
# return uniform and let the face and voice channels carry the moment instead.
MIN_INFORMATIVE_CHARS = 4

# Separator used to join context turns. Must match training/train_text.py.
CONTEXT_SEPARATOR = " </s> "


class TextEmotionModel:
    """Lazily-loaded sequence classifier. Thread-safe to call from FastAPI."""

    def __init__(self) -> None:
        self._pipeline = None
        # Second, general-domain head used for the ensemble. None when the
        # ensemble is off or when there's no MELD checkpoint to pair it with.
        self._general = None
        self._lock = threading.Lock()
        self._source: Optional[str] = None
        self._load_error: Optional[str] = None
        # How many prior sentences the checkpoint expects prepended. Read from
        # the checkpoint itself so inference always matches however it was
        # trained — serving bare sentences to a context-trained model is a
        # silent accuracy loss, not an error.
        self._context_turns = 0

    # -- loading ---------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        if self._pipeline is not None:
            return True
        if self._load_error is not None:
            return False

        with self._lock:
            if self._pipeline is not None:  # won the race while waiting
                return True
            if self._load_error is not None:
                return False
            try:
                from transformers import pipeline

                device = 0 if settings.resolved_device() == "cuda" else -1
                has_meld = settings.has_meld_checkpoint()

                if has_meld:
                    self._context_turns = _read_context_turns(
                        settings.meld_text_model / "meld_results.json"
                    )
                    self._pipeline = pipeline(
                        "text-classification",
                        model=str(settings.meld_text_model),
                        top_k=None,  # return every class, not just argmax
                        device=device,
                    )
                    self._source = "meld-finetuned"
                else:
                    log.warning(
                        "No MELD checkpoint at %s - using pretrained fallback %s. "
                        "Run training/train_text.py for the MELD-trained head.",
                        settings.meld_text_model,
                        settings.fallback_text_model,
                    )
                    self._pipeline = pipeline(
                        "text-classification",
                        model=settings.fallback_text_model,
                        top_k=None,
                        device=device,
                    )
                    self._source = "fallback-pretrained"

                # The general-domain head is loaded as a second opinion only when
                # there's a MELD model to pair it with — on its own it *is* the
                # fallback, and ensembling a model with itself is just overhead.
                if has_meld and settings.ensemble_text:
                    self._general = pipeline(
                        "text-classification",
                        model=settings.fallback_text_model,
                        top_k=None,
                        device=device,
                    )
                    self._source = "meld+general-ensemble"
                    log.info(
                        "Text ensemble ready: MELD w=%.2f, general w=%.2f",
                        settings.ensemble_meld_weight,
                        1 - settings.ensemble_meld_weight,
                    )
                else:
                    log.info("Text emotion model ready (%s)", self._source)
            except Exception as exc:  # noqa: BLE001 - degrade instead of crash
                self._load_error = str(exc)
                log.error("Text emotion model unavailable: %s", exc)
                return False
        return True

    def _blend(self, meld_scores: dict, general_scores: Optional[dict]) -> List[float]:
        """Weighted average of the two heads, in canonical vector space.

        Averaging happens *after* mapping both to the MELD label space, so the
        two models' differing label names never have to be reconciled directly.
        """
        meld_vector = from_label_scores(meld_scores)
        if general_scores is None:
            return meld_vector

        general_vector = from_label_scores(general_scores)
        w = settings.ensemble_meld_weight
        return normalize(
            [m * w + g * (1 - w) for m, g in zip(meld_vector, general_vector)]
        )

    # -- inference -------------------------------------------------------

    def predict(self, text: str) -> List[float]:
        """Return a canonical probability vector for a single utterance."""
        cleaned = _clean(text)
        if len(cleaned) < MIN_INFORMATIVE_CHARS or not self._ensure_loaded():
            return uniform_vector()

        try:
            # transformers returns [[{label, score}, ...]] for a single input
            raw = self._pipeline(cleaned, truncation=True, max_length=256)[0]
            general = (
                self._general(cleaned, truncation=True, max_length=256)[0]
                if self._general is not None
                else None
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Text inference failed: %s", exc)
            return uniform_vector()

        return self._blend(
            {item["label"]: item["score"] for item in raw},
            {item["label"]: item["score"] for item in general} if general else None,
        )

    def predict_batch(self, texts: List[str]) -> List[List[float]]:
        """Per-sentence vectors for one entry, in order.

        Sentences are scored *with* the preceding ones prepended, matching how
        the checkpoint was trained. A diary is a monologue rather than a
        dialogue, so the prior "turns" are the speaker's own earlier sentences —
        not what MELD was built from, but the same shape of input, which is what
        the model actually keys on. Each sentence still contributes only its own
        label, so the peak-sentence and keyword attribution stay per-sentence.
        """
        if not texts or not self._ensure_loaded():
            return [uniform_vector() for _ in texts]

        informative = [
            i for i, t in enumerate(texts) if len(_clean(t)) >= MIN_INFORMATIVE_CHARS
        ]
        results = [uniform_vector() for _ in texts]
        if not informative:
            return results

        contextual = [self._with_context(texts, i) for i in informative]
        try:
            raw = self._pipeline(contextual, truncation=True, max_length=256)
            # The general model was never trained on prepended dialogue turns, so
            # it sees the bare sentence. Feeding it the context format would put
            # it just as far out of domain as the mismatch this fixes.
            general = (
                self._general(
                    [_clean(texts[i]) for i in informative],
                    truncation=True,
                    max_length=256,
                )
                if self._general is not None
                else [None] * len(informative)
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Batched text inference failed: %s", exc)
            return results

        for slot, scores, gen in zip(informative, raw, general):
            results[slot] = self._blend(
                {s["label"]: s["score"] for s in scores},
                {g["label"]: g["score"] for g in gen} if gen else None,
            )
        return results

    def _with_context(self, texts: List[str], index: int) -> str:
        """Build the model input for `texts[index]`, matching training format."""
        current = _clean(texts[index])
        if self._context_turns <= 0 or index == 0:
            return current

        prior = [
            _clean(t)
            for t in texts[max(0, index - self._context_turns) : index]
            if _clean(t)
        ]
        return CONTEXT_SEPARATOR.join([*prior, current]) if prior else current

    # -- introspection ---------------------------------------------------

    def status(self) -> dict:
        return {
            "loaded": self._pipeline is not None,
            "source": self._source,
            "labels": EMOTIONS,
            "contextTurns": self._context_turns,
            "ensemble": self._general is not None,
            "ensembleMeldWeight": settings.ensemble_meld_weight,
            "error": self._load_error,
        }


def _read_context_turns(results_path) -> int:
    """How many context turns the checkpoint was trained with.

    Written by training/train_text.py. A checkpoint without the file (hand-built,
    or from an older run) falls back to 0, which is the safe direction: no
    context is a milder mismatch than the wrong amount of it.
    """
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        turns = int(data.get("context_turns", 0))
        log.info("Checkpoint expects %d context turn(s) at inference", turns)
        return max(0, turns)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read context_turns (%s); serving without context", exc)
        return 0


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def split_sentences(text: str) -> List[str]:
    """Cheap sentence splitter.

    Deliberately not spaCy/NLTK — Whisper output is already lightly punctuated
    and this only feeds per-sentence scoring, where a wrong boundary costs us
    nothing worse than a slightly odd peak sentence.
    """
    parts = re.split(r"(?<=[.!?])\s+|\n+", _clean(text))
    return [p.strip() for p in parts if len(p.strip()) >= MIN_INFORMATIVE_CHARS]


text_model = TextEmotionModel()
