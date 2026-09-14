"""The MELD label space, shared by every model in the sidecar.

The ordering of ``EMOTIONS`` is load-bearing: probability vectors cross the wire
to the browser as bare arrays, and frontend/src/emotions.ts indexes them
positionally. If you reorder this list, reorder that one.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

# MELD's seven emotion labels, in canonical order.
EMOTIONS: List[str] = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
]

EMOTION_INDEX: Dict[str, int] = {e: i for i, e in enumerate(EMOTIONS)}

# MELD also ships a 3-way sentiment label per utterance. We don't predict it,
# but we report it as a derived view because it's a more stable signal than the
# 7-way argmax when the emotion head is uncertain.
SENTIMENT_OF: Dict[str, str] = {
    "neutral": "neutral",
    "joy": "positive",
    "surprise": "positive",
    "sadness": "negative",
    "anger": "negative",
    "fear": "negative",
    "disgust": "negative",
}


def zero_vector() -> List[float]:
    return [0.0] * len(EMOTIONS)


def uniform_vector() -> List[float]:
    return [1.0 / len(EMOTIONS)] * len(EMOTIONS)


def normalize(vector: Sequence[float]) -> List[float]:
    """Clamp negatives away and rescale to sum 1. All-zero input -> uniform."""
    clipped = [max(0.0, float(v)) for v in vector]
    total = sum(clipped)
    if total <= 1e-9:
        return uniform_vector()
    return [v / total for v in clipped]


def from_label_scores(scores: Dict[str, float]) -> List[float]:
    """Build a canonical vector from a {label: score} mapping.

    Labels the caller doesn't provide stay at zero. Unknown labels are ignored,
    which is what lets the label-mapping tables in the model modules be partial.
    """
    vector = zero_vector()
    for label, score in scores.items():
        idx = EMOTION_INDEX.get(label.lower().strip())
        if idx is not None:
            vector[idx] += float(score)
    return normalize(vector)


def dominant(vector: Sequence[float]) -> str:
    return EMOTIONS[max(range(len(EMOTIONS)), key=lambda i: vector[i])]


def sentiment(vector: Sequence[float]) -> str:
    """Collapse the 7-way distribution into MELD's 3-way sentiment view.

    Done by summing probability mass per sentiment class rather than by looking
    up the argmax, so a reading split across sadness/anger/fear still reads as
    clearly negative even when no single emotion wins.
    """
    totals = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for emotion, prob in zip(EMOTIONS, vector):
        totals[SENTIMENT_OF[emotion]] += prob
    return max(totals, key=totals.get)


def as_dict(vector: Sequence[float]) -> Dict[str, float]:
    return {e: round(float(p), 6) for e, p in zip(EMOTIONS, vector)}


def charge(vector: Sequence[float]) -> float:
    """How much emotion is present at all, in [0, 1].

    Simply the non-neutral mass. Deliberately says nothing about *which* emotions
    or how many — that's ``complexity``'s job. Keeping the two apart is the whole
    point: a person feeling fear and grief at once is having an intense
    experience, and any measure that scores them as "unclear" is wrong.
    """
    return 1.0 - float(vector[EMOTION_INDEX["neutral"]])


def complexity(vector: Sequence[float]) -> float:
    """Effective number of emotions being felt, in [1, 6].

    The perplexity (``exp`` of Shannon entropy) of the distribution with neutral
    removed and the remainder renormalised. This reads literally as a count:

        1.0  a single emotion
        2.0  two, held about equally — fear *and* sadness
        6.0  all of them, i.e. no structure at all

    Perplexity is the right tool here because it is invariant to how many classes
    exist: two emotions at 0.5 each give exactly 2.0 whether the label space has
    seven emotions or seventy.
    """
    neutral_index = EMOTION_INDEX["neutral"]
    rest = [max(0.0, v) for i, v in enumerate(vector) if i != neutral_index]
    total = sum(rest)
    if total <= 1e-9:
        return 1.0

    entropy = 0.0
    for value in rest:
        p = value / total
        if p > 1e-9:
            entropy -= p * math.log(p)
    return math.exp(entropy)


def components(
    vector: Sequence[float], floor: float = 0.15
) -> List[tuple[str, float]]:
    """The emotions actually contributing, largest first.

    ``floor`` is a share of the non-neutral mass, not an absolute probability, so
    the threshold means the same thing for a strongly-felt entry and a mild one.
    """
    neutral_index = EMOTION_INDEX["neutral"]
    total = sum(max(0.0, v) for i, v in enumerate(vector) if i != neutral_index)
    if total <= 1e-9:
        return [("neutral", 1.0)]

    found = [
        (EMOTIONS[i], v / total)
        for i, v in enumerate(vector)
        if i != neutral_index and v / total >= floor
    ]
    found.sort(key=lambda pair: pair[1], reverse=True)
    return found or [("neutral", 1.0)]


def blend(parts: Iterable[tuple[Sequence[float], float]]) -> List[float]:
    """Weighted late fusion. ``parts`` is an iterable of (vector, weight)."""
    out = zero_vector()
    for vector, weight in parts:
        if weight <= 0:
            continue
        for i in range(len(EMOTIONS)):
            out[i] += float(vector[i]) * float(weight)
    return normalize(out)
