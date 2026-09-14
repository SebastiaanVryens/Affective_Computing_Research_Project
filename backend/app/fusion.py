"""Late fusion of the three modality channels.

Two problems have to be solved here, and they pull in opposite directions.

**Shrugs shouldn't outvote convictions.** A modality with no idea what it's
looking at would otherwise vote at full strength, and three shrugs beat one
confident reading. So each channel's static weight is scaled by its own
certainty — ``1 - normalised Shannon entropy``, near 1 for a peaked distribution
and near 0 for a uniform one. The scheme self-gates: the face channel quietly
stops mattering while you sit with a resting face and takes over the moment your
expression changes.

**Agreement should count for something.** This is what a plain weighted average
cannot express. Averaging is a *linear opinion pool*, and its output is bounded
by its inputs — two independent sensors that both say "joy, 0.6" produce exactly
0.6, no more confident than either alone. That's wrong. Two sensors agreeing is
stronger evidence than one.

So the default is a *logarithmic* opinion pool instead: probabilities are
multiplied rather than added, weights act as exponents, and the total exponent is
allowed to exceed 1 when several channels are genuinely contributing. That is the
independent-evidence assumption, and it makes agreement sharpen the result and
disagreement flatten it — both correct.

How far to trust that independence is the ``independence`` knob. Face, voice and
words from one person in one moment are obviously *not* independent sensors —
they share a cause. So the default is deliberately below the fully-independent
value of 1.0. Set it to 0 to recover exact averaging, which is the ablation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .config import settings
from .emotions import (
    EMOTIONS,
    blend,
    charge,
    complexity,
    components,
    dominant,
    normalize,
    sentiment,
    uniform_vector,
)

# Mixed into every distribution before log-pooling. Without it a single channel
# reporting ~0 for some emotion vetoes that emotion outright no matter what the
# others say — multiplication's failure mode, and a real risk for the rare
# classes (fear, disgust) that the text model already under-predicts.
POOL_FLOOR = 0.02

# Minimum cross-modal agreement for a multi-emotion reading to be reported as a
# genuine blend rather than as disagreement. Below this the channels are telling
# different stories and the spread is noise, not nuance.
BLEND_AGREEMENT_FLOOR = 0.55

# Minimum non-neutral mass before a spread is worth calling a blend at all.
# Below this the entry is essentially calm and the leftover mass is noise.
MIN_BLEND_CHARGE = 0.35

# Share of the non-neutral mass at which one emotion is clearly leading, so an
# uncorroborated spread can still be named after it rather than called unclear.
CLEAR_LEAD_SHARE = 0.5


def certainty(vector: Sequence[float]) -> float:
    """1 - normalised entropy, in [0, 1]."""
    entropy = 0.0
    for p in vector:
        if p > 1e-9:
            entropy -= p * math.log(p)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(EMOTIONS))))


def effective_channels(weights: Sequence[float]) -> float:
    """How many channels are *meaningfully* contributing.

    The inverse participation ratio, ``(Σw)² / Σw²``. One dominant channel gives
    ~1 no matter how many others are nominally present; n equally-weighted
    channels give n. This is what stops a barely-contributing third channel from
    being counted as a full independent witness.
    """
    total = sum(weights)
    sum_squares = sum(w * w for w in weights)
    if sum_squares <= 1e-12:
        return 0.0
    return (total * total) / sum_squares


def face_at(
    timeline: Sequence[dict], start: float, end: float, pad: float = 0.5
) -> Optional[List[float]]:
    """Average face reading over a time window, for aligning to one utterance.

    This is what makes face and speech corroborate *each other* rather than
    corroborating a two-minute average. Saying "I passed" while visibly relieved
    is a different event from saying it flatly, and only a time-aligned reading
    can tell them apart.

    The window is padded on both sides because expression and speech aren't
    synchronous — a reaction often lands slightly before or after the words.

    Returns None when no face was visible during the window, which is a real and
    different state from "the face looked neutral": the fusion layer drops the
    channel entirely rather than treating absence as evidence.
    """
    if not timeline:
        return None

    lo, hi = start - pad, end + pad
    total = [0.0] * len(EMOTIONS)
    count = 0

    for sample in timeline:
        t = sample.get("t")
        vector = sample.get("v")
        if t is None or not vector or len(vector) != len(EMOTIONS):
            continue
        if lo <= t <= hi:
            for i in range(len(EMOTIONS)):
                total[i] += float(vector[i])
            count += 1

    if count == 0:
        return None
    return normalize(total)


def agreement(parts: Sequence[tuple[Sequence[float], float]]) -> float:
    """How much the channels are telling the same story, in [0, 1].

    Weighted mean pairwise Bhattacharyya coefficient, ``Σ √(p·q)`` — 1 for
    identical distributions, 0 for distributions with disjoint support. Chosen
    over cosine similarity because it's the natural overlap measure for
    probability vectors and is symmetric without normalisation games.

    A single channel has nothing to agree with, so it returns 1: pass-through,
    no bonus and no penalty.
    """
    if len(parts) < 2:
        return 1.0

    total = 0.0
    total_weight = 0.0
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            (p, wp), (q, wq) = parts[i], parts[j]
            overlap = sum(math.sqrt(max(0.0, a) * max(0.0, b)) for a, b in zip(p, q))
            pair_weight = wp * wq
            total += min(1.0, overlap) * pair_weight
            total_weight += pair_weight

    return total / total_weight if total_weight > 1e-12 else 1.0


def log_pool(
    parts: Sequence[tuple[Sequence[float], float]], independence: float
) -> List[float]:
    """Weighted geometric mean, sharpened by how many channels agree.

    Mechanically: normalise the weights to sum to 1 (relative influence), then
    scale them all by a sharpening exponent ``tau`` before multiplying.

    ``tau = 1 + independence * (effective_channels - 1)``

    With one channel, tau is 1 and the channel passes through untouched. With two
    equally-weighted channels and ``independence = 1``, tau is 2 — the exact
    product of the two distributions, i.e. treating them as fully independent
    evidence. Values in between hedge, which is the honest position for three
    sensors pointed at the same person.

    The sharpening applies to disagreement too: conflicting channels multiply to
    a flatter distribution than either input, so the fused certainty drops. That
    falls out of the arithmetic rather than being special-cased.
    """
    active = [(v, w) for v, w in parts if w > 1e-9]
    if not active:
        return uniform_vector()

    weights = [w for _, w in active]
    total = sum(weights)
    normalised = [w / total for w in weights]

    # Gate the independence bonus on whether the channels actually agree.
    #
    # Without this the pool manufactures false confidence out of conflict: a
    # smiling face against an angry voice share only their small neutral mass, so
    # multiplying them yields a *confidently* neutral reading that neither
    # channel ever endorsed. But face and voice aren't independent sensors of a
    # latent state — when they disagree it usually means one of them is wrong,
    # not that the truth is their intersection.
    #
    # So the channels are treated as corroborating evidence only to the extent
    # they're saying the same thing. Under real conflict tau falls to 1 and the
    # pool degenerates to a weighted geometric mean, which reports the low
    # certainty that a contradiction deserves.
    consensus = agreement(active)
    tau = 1.0 + independence * max(0.0, effective_channels(weights) - 1.0) * consensus

    # Sum of logs rather than a product of probabilities: seven classes times
    # three channels of small numbers underflows float64 territory quickly, and
    # log-space costs nothing here.
    accumulated = [0.0] * len(EMOTIONS)
    for (vector, _), weight in zip(active, normalised):
        exponent = weight * tau
        for i in range(len(EMOTIONS)):
            smoothed = (1 - POOL_FLOOR) * vector[i] + POOL_FLOOR / len(EMOTIONS)
            accumulated[i] += exponent * math.log(smoothed)

    # Subtract the max before exponentiating — standard log-sum-exp guard.
    peak = max(accumulated)
    geometric = normalize([math.exp(a - peak) for a in accumulated])

    # Finally, pick the *operator* by consensus too, not just its strength.
    #
    # A geometric pool is an AND: it rewards what every channel endorses. That's
    # exactly right for corroboration and exactly wrong for contradiction, where
    # it still prefers whichever class the channels merely tolerate. Damping tau
    # alone doesn't fix that — the bias is in the operator, not its exponent.
    #
    # An arithmetic pool is an OR: conflicting channels produce a broad, visibly
    # unsure distribution. For a diary "it could be joy or anger, I can't tell"
    # is a better answer than a confident compromise nobody argued for.
    #
    # So: corroboration pools multiplicatively, contradiction pools additively,
    # and consensus slides between them.
    if consensus >= 0.999:
        return geometric

    arithmetic = blend(active)
    return normalize(
        [consensus * g + (1 - consensus) * a for g, a in zip(geometric, arithmetic)]
    )


@dataclass
class ModalityReading:
    name: str
    vector: List[float]
    base_weight: float
    # Set when a channel is present but produced nothing usable (no face in
    # frame, audio too short, model unloaded). Recorded rather than silently
    # dropped, because "we couldn't see you" is different from "you looked calm".
    available: bool = True
    source: Optional[str] = None

    @property
    def certainty(self) -> float:
        return certainty(self.vector) if self.available else 0.0

    @property
    def effective_weight(self) -> float:
        return self.base_weight * self.certainty if self.available else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "vector": [round(p, 4) for p in self.vector],
            "dominant": dominant(self.vector) if self.available else None,
            "certainty": round(self.certainty, 4),
            "weight": round(self.effective_weight, 4),
            "available": self.available,
            "source": self.source,
        }


@dataclass
class BlendProfile:
    """How many emotions are in play, and whether that's real or just noise.

    A probability vector alone cannot answer this. ``fear 0.5, sadness 0.5`` is
    the identical vector whether the person felt both at once or the model
    couldn't decide between them — the simplex has no room to distinguish a
    mixture from an uncertainty. Reporting either as "low intensity" is wrong,
    and it's what a lone ``1 - entropy`` score does.

    The way out is that a *single* channel can't tell these apart but three can.
    If face, voice and words independently converge on the same two emotions,
    that's corroboration, and the blend is real. If they're each pointing
    somewhere different, the spread is disagreement. So ``supported`` is decided
    by cross-modal agreement, not by the shape of the fused vector.

    The three numbers are deliberately orthogonal:
      charge      how much is happening      (independent of which/how many)
      complexity  how many emotions          (independent of how strong)
      supported   whether to believe it      (independent of both)
    """

    charge: float
    complexity: float
    components: List[tuple[str, float]]
    supported: bool

    @property
    def is_blend(self) -> bool:
        """Two or more emotions, enough of them to matter, and channels agreeing.

        The charge floor is load-bearing. On a flat "I went to the library" entry
        the residual non-neutral mass is a few percent of noise smeared across
        several classes, which looks exactly like a rich three-way blend to any
        measure that only inspects proportions. Blending *nothing* three ways is
        still nothing.
        """
        return (
            len(self.components) >= 2
            and self.supported
            and self.charge >= MIN_BLEND_CHARGE
        )

    @property
    def label(self) -> str:
        """Human phrasing — 'fear and sadness', or the lead emotion on its own."""
        if not self.components:
            return "neutral"
        if self.charge < MIN_BLEND_CHARGE:
            # Too little going on to name a mixture; the entry is basically calm.
            return "neutral"
        if len(self.components) == 1:
            return self.components[0][0]
        if self.is_blend:
            names = [name for name, _ in self.components[:3]]
            return " and ".join(names) if len(names) == 2 else ", ".join(names)

        # Spread, but uncorroborated. Only call it "unclear" when nothing leads —
        # a clearly dominant emotion with some minor company should be reported
        # as that emotion, not thrown away as confusion.
        leader, share = self.components[0]
        return leader if share >= CLEAR_LEAD_SHARE else "unclear"

    def to_dict(self) -> dict:
        return {
            "charge": round(self.charge, 4),
            "complexity": round(self.complexity, 4),
            "components": [
                {"emotion": name, "share": round(share, 4)}
                for name, share in self.components
            ],
            "isBlend": self.is_blend,
            "supported": self.supported,
            "label": self.label,
        }


@dataclass
class FusedResult:
    vector: List[float]
    dominant: str
    sentiment: str
    certainty: float
    modalities: List[ModalityReading]
    blend: Optional[BlendProfile] = None

    def to_dict(self) -> dict:
        return {
            "vector": [round(p, 4) for p in self.vector],
            "emotions": {e: round(p, 4) for e, p in zip(EMOTIONS, self.vector)},
            "dominant": self.dominant,
            "sentiment": self.sentiment,
            "certainty": round(self.certainty, 4),
            "blend": self.blend.to_dict() if self.blend else None,
            "modalities": [m.to_dict() for m in self.modalities],
        }


def fuse(
    text_vector: Optional[Sequence[float]] = None,
    face_vector: Optional[Sequence[float]] = None,
    voice_vector: Optional[Sequence[float]] = None,
    *,
    text_source: Optional[str] = None,
    face_source: Optional[str] = None,
    voice_source: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> FusedResult:
    w = {
        "text": settings.weight_text,
        "face": settings.weight_face,
        "voice": settings.weight_voice,
    }
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})

    readings = [
        ModalityReading(
            name="text",
            vector=normalize(text_vector) if text_vector else uniform_vector(),
            base_weight=w["text"],
            available=text_vector is not None,
            source=text_source,
        ),
        ModalityReading(
            name="face",
            vector=normalize(face_vector) if face_vector else uniform_vector(),
            base_weight=w["face"],
            available=face_vector is not None,
            source=face_source,
        ),
        ModalityReading(
            name="voice",
            vector=normalize(voice_vector) if voice_vector else uniform_vector(),
            base_weight=w["voice"],
            available=voice_vector is not None,
            source=voice_source,
        ),
    ]

    total_weight = sum(r.effective_weight for r in readings)
    if total_weight <= 1e-6:
        # Every channel either missing or maximally unsure. Rather than emit a
        # uniform vector that would render as muddy grey, report neutral — which
        # is also the honest reading of "nothing detectable happened".
        fused = normalize([1.0 if e == "neutral" else 0.05 for e in EMOTIONS])
    elif settings.fusion_independence > 0:
        fused = log_pool(
            [(r.vector, r.effective_weight) for r in readings],
            independence=settings.fusion_independence,
        )
    else:
        fused = blend((r.vector, r.effective_weight) for r in readings)

    # Whether a spread across emotions is a real blend or just disagreement is
    # decided by the channels, not by the fused vector's shape. With only one
    # channel available there's nothing to corroborate against, so we fall back
    # to that channel's own confidence.
    contributing = [(r.vector, r.effective_weight) for r in readings if r.effective_weight > 0]
    if len(contributing) >= 2:
        supported = agreement(contributing) >= BLEND_AGREEMENT_FLOOR
    elif contributing:
        supported = certainty(contributing[0][0]) >= 0.25
    else:
        supported = False

    return FusedResult(
        vector=fused,
        dominant=dominant(fused),
        sentiment=sentiment(fused),
        certainty=certainty(fused),
        modalities=readings,
        blend=BlendProfile(
            charge=charge(fused),
            complexity=complexity(fused),
            components=components(fused),
            supported=supported,
        ),
    )
