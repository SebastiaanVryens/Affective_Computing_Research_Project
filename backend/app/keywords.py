"""Keyword extraction, with each keyword tagged by the emotion it was said in.

This is the bit that makes the world feel personal rather than abstract. A day
isn't just "62% sadness" — it's "sadness, and the word was *thesis*". So we
don't just rank keywords by salience: we attribute each one to the emotion of
the sentence it appeared in, and that attribution is what colours the word in
the 3D scene.

YAKE is used when installed (better multi-word phrases); otherwise a built-in
candidate extractor covers it with no extra dependency.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .emotions import EMOTIONS, dominant, normalize, zero_vector

log = logging.getLogger(__name__)

# Function words plus the filler that dominates spoken diary entries. Anything
# here can still appear *inside* a multi-word phrase, it just can't be a keyword
# on its own.
STOPWORDS = frozenset(
    """
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he her here hers herself him himself his how i if in into is isn't it
its itself just let's me more most mustn't my myself no nor not of off on once only or
other ought our ours ourselves out over own same shan't she should shouldn't so some
such than that the their theirs them themselves then there these they this those
through to too under until up very was wasn't we were weren't what when where which
while who whom why with won't would wouldn't you your yours yourself yourselves
also get got go going going-to would-be really quite maybe kind sort lot bit thing
things stuff okay ok yeah yep nope uh um erm hmm like actually basically literally
today tomorrow yesterday day days week month year time
i'm i've i'd i'll you're you've we're we've they're it's that's there's
""".split()
)

# Spoken filler that should never survive into a keyword, even mid-phrase.
FILLER = frozenset({"uh", "um", "erm", "hmm", "mm", "ah", "eh", "like", "you know"})

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'\-]+")


@dataclass
class Keyword:
    text: str
    score: float
    count: int = 1
    # Probability mass over emotions, accumulated from every sentence the
    # keyword appeared in, weighted by how confident that sentence's reading was.
    emotion_vector: List[float] = field(default_factory=zero_vector)

    @property
    def emotion(self) -> str:
        return dominant(self.emotion_vector)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": round(self.score, 4),
            "count": self.count,
            "emotion": self.emotion,
            "emotionVector": [round(p, 4) for p in normalize(self.emotion_vector)],
        }


def extract(
    sentences: Sequence[str],
    sentence_vectors: Sequence[Sequence[float]],
    top_k: int = 12,
) -> List[Keyword]:
    """Rank keywords across an entry and attribute each to an emotion.

    ``sentences`` and ``sentence_vectors`` are parallel: index i's vector is the
    emotion reading for sentence i.
    """
    if not sentences:
        return []

    merged: Dict[str, Keyword] = {}

    for sentence, vector in zip(sentences, sentence_vectors):
        candidates = _candidates(sentence)
        if not candidates:
            continue

        # A flat, uncertain sentence shouldn't stain its keywords with a
        # confident emotion, so attribution is weighted by peak probability.
        confidence = max(vector) if len(vector) else 0.0

        for phrase, local_score in candidates.items():
            key = phrase.lower()
            existing = merged.get(key)
            if existing is None:
                existing = Keyword(text=phrase, score=0.0, count=0)
                merged[key] = existing
            existing.score += local_score
            existing.count += 1
            for i in range(len(EMOTIONS)):
                existing.emotion_vector[i] += vector[i] * confidence

    # Repeating a word across an entry is a strong signal it mattered, but the
    # boost is sublinear so one obsessive mention doesn't crowd everything out.
    for keyword in merged.values():
        keyword.score *= 1.0 + 0.4 * (keyword.count - 1) ** 0.5
        keyword.emotion_vector = normalize(keyword.emotion_vector)

    ranked = sorted(merged.values(), key=lambda k: k.score, reverse=True)
    return _drop_subsumed(ranked)[:top_k]


def _candidates(sentence: str) -> Dict[str, float]:
    """Scored candidate phrases from a single sentence."""
    yake_result = _yake_candidates(sentence)
    if yake_result is not None:
        return yake_result
    return _ngram_candidates(sentence)


def _yake_candidates(sentence: str) -> Dict[str, float] | None:
    try:
        import yake
    except ImportError:
        return None

    try:
        extractor = yake.KeywordExtractor(lan="en", n=2, top=8, dedupLim=0.8)
        # YAKE scores are a cost: lower is better. Invert into a salience.
        out: Dict[str, float] = {}
        for phrase, cost in extractor.extract_keywords(sentence):
            phrase = phrase.strip()
            if _acceptable(phrase):
                out[phrase] = 1.0 / (1.0 + float(cost))
        return out
    except Exception as exc:  # noqa: BLE001
        log.debug("YAKE failed on a sentence, using built-in extractor: %s", exc)
        return None


def _ngram_candidates(sentence: str) -> Dict[str, float]:
    """Contiguous non-stopword runs, capped at bigrams.

    The RAKE intuition: stopwords are phrase delimiters, and what survives
    between them is the content. Longer runs score higher because a two-word
    phrase ("job interview") says more than either word alone.
    """
    tokens = TOKEN_RE.findall(sentence.lower())
    if not tokens:
        return {}

    runs: List[List[str]] = []
    current: List[str] = []
    for token in tokens:
        if token in STOPWORDS or token in FILLER or len(token) < 3:
            if current:
                runs.append(current)
                current = []
        else:
            current.append(token)
    if current:
        runs.append(current)

    out: Dict[str, float] = {}
    for run in runs:
        for size in (2, 1):
            for i in range(len(run) - size + 1):
                phrase = " ".join(run[i : i + size])
                if not _acceptable(phrase):
                    continue
                # Bigrams beat unigrams; rarer (longer) words beat short ones.
                out[phrase] = max(
                    out.get(phrase, 0.0),
                    (1.6 if size == 2 else 1.0) * min(1.5, len(phrase) / 8),
                )
    return out


def _acceptable(phrase: str) -> bool:
    phrase = phrase.strip().lower()
    if len(phrase) < 3 or phrase in FILLER:
        return False
    words = phrase.split()
    if len(words) > 2:
        return False
    # A phrase made entirely of stopwords carries nothing.
    return any(w not in STOPWORDS and len(w) >= 3 for w in words)


def _drop_subsumed(ranked: List[Keyword]) -> List[Keyword]:
    """Remove unigrams already covered by a higher-ranked bigram.

    Without this, "job interview", "job" and "interview" all make the list and
    the world fills with near-duplicate floating words.
    """
    kept: List[Keyword] = []
    seen_words: set[str] = set()
    for keyword in ranked:
        words = set(keyword.text.lower().split())
        if len(words) == 1 and words & seen_words:
            continue
        kept.append(keyword)
        seen_words |= words
    return kept


def aggregate_by_emotion(keywords: Sequence[Keyword]) -> Dict[str, List[str]]:
    """Group keyword text by dominant emotion — the shape the UI renders."""
    grouped: Dict[str, List[str]] = defaultdict(list)
    for keyword in keywords:
        grouped[keyword.emotion].append(keyword.text)
    return dict(grouped)
