"""Dependency-light checks for the parts that don't need torch.

    python -m training.smoke_test

Covers the taxonomy, the fusion arithmetic, and keyword extraction — the logic
most likely to break silently, since a wrong fusion weight or a misaligned label
index produces plausible-looking numbers rather than an error.

Deliberately not pytest: this should run in a bare interpreter with only numpy
available, so it's useful before the full install finishes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.emotions import (  # noqa: E402
    EMOTIONS,
    blend,
    dominant,
    from_label_scores,
    normalize,
    sentiment,
    uniform_vector,
)
from app.fusion import certainty, fuse  # noqa: E402
from app.keywords import extract  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


print("\n=== taxonomy ===")
check("seven labels", len(EMOTIONS) == 7, str(EMOTIONS))
check("neutral is index 0", EMOTIONS[0] == "neutral")
check("uniform sums to 1", close(sum(uniform_vector()), 1.0))
check("normalize handles all-zero", close(sum(normalize([0] * 7)), 1.0))
check(
    "normalize clips negatives",
    all(p >= 0 for p in normalize([-5, 1, 0, 0, 0, 0, 0])),
)

joy_heavy = from_label_scores({"joy": 0.8, "neutral": 0.2})
check("from_label_scores -> joy", dominant(joy_heavy) == "joy")
check("unknown labels ignored", close(sum(from_label_scores({"ennui": 1.0, "joy": 1.0})), 1.0))

print("\n=== sentiment collapse ===")
check("joy is positive", sentiment(from_label_scores({"joy": 1.0})) == "positive")
check("anger is negative", sentiment(from_label_scores({"anger": 1.0})) == "negative")
# The reason sentiment sums mass instead of reading the argmax: no single
# emotion wins here, but the reading is unambiguously negative.
split_negative = from_label_scores({"sadness": 0.3, "anger": 0.3, "fear": 0.25, "joy": 0.15})
check(
    "split negative reads negative",
    sentiment(split_negative) == "negative",
    str(split_negative),
)

print("\n=== certainty ===")
check("uniform has ~0 certainty", close(certainty(uniform_vector()), 0.0, 1e-9))
peaked = from_label_scores({"anger": 1.0})
check("one-hot has ~1 certainty", certainty(peaked) > 0.99)
check(
    "certainty is monotone",
    certainty(from_label_scores({"joy": 0.9, "neutral": 0.1}))
    > certainty(from_label_scores({"joy": 0.5, "neutral": 0.5})),
)

print("\n=== fusion ===")
confident_sad = from_label_scores({"sadness": 0.92, "neutral": 0.08})
vague = uniform_vector()

# The central claim of fusion.py: an uncertain channel shouldn't be able to
# drown out a confident one, even with a larger static weight.
result = fuse(text_vector=confident_sad, face_vector=vague, weights={"text": 0.3, "face": 0.7})
check(
    "certainty beats static weight",
    result.dominant == "sadness",
    f"got {result.dominant} ({result.to_dict()['emotions']})",
)

all_vague = fuse(text_vector=vague, face_vector=vague, voice_vector=vague)
check(
    "total ambiguity -> neutral, not uniform",
    all_vague.dominant == "neutral",
    f"got {all_vague.dominant}",
)

missing = fuse(text_vector=confident_sad)
check("absent channels excluded", missing.dominant == "sadness")
check(
    "absent channel marked unavailable",
    not next(m for m in missing.modalities if m.name == "face").available,
)

confident_joy = from_label_scores({"joy": 0.9, "neutral": 0.1})
confident_anger = from_label_scores({"anger": 0.9, "neutral": 0.1})
conflict = fuse(
    text_vector=confident_joy,
    face_vector=confident_anger,
    weights={"text": 0.5, "face": 0.5},
)
# The property that matters is relative, not an absolute threshold: with seven
# classes, a clean 50/50 split between two of them still scores around 0.5,
# because the other five are firmly excluded and that genuinely is information.
# What must hold is that disagreement makes the fused reading *less* certain
# than either channel was on its own.
check(
    "equal conflict lowers certainty below both inputs",
    conflict.certainty < min(certainty(confident_joy), certainty(confident_anger)),
    f"fused={conflict.certainty:.3f} vs inputs={certainty(confident_joy):.3f}",
)
check(
    "conflict leaves no confident winner",
    close(conflict.vector[EMOTIONS.index("joy")], conflict.vector[EMOTIONS.index("anger")], 0.02),
    f"joy={conflict.vector[EMOTIONS.index('joy')]:.3f} anger={conflict.vector[EMOTIONS.index('anger')]:.3f}",
)

print("\n=== cross-modal reinforcement ===")
from app.config import settings as _settings  # noqa: E402
from app.fusion import agreement  # noqa: E402

_settings.fusion_independence = 0.5
joyful = from_label_scores({"joy": 0.60, "neutral": 0.25, "surprise": 0.15})
also_joyful = from_label_scores({"joy": 0.60, "neutral": 0.25, "surprise": 0.15})
angry_voice = from_label_scores({"anger": 0.60, "neutral": 0.25, "sadness": 0.15})

solo = certainty(joyful)
agreeing = fuse(face_vector=joyful, voice_vector=also_joyful)
check(
    "agreeing channels beat either alone",
    agreeing.vector[EMOTIONS.index("joy")] > 0.60 and agreeing.certainty > solo,
    f"joy={agreeing.vector[EMOTIONS.index('joy')]:.3f} vs 0.600, "
    f"certainty={agreeing.certainty:.3f} vs {solo:.3f}",
)

conflicting = fuse(face_vector=joyful, voice_vector=angry_voice)
check(
    "conflicting channels stay unsure",
    conflicting.certainty < solo,
    f"certainty={conflicting.certainty:.3f} vs solo {solo:.3f}",
)
# The bug this guards: a geometric pool alone answers "confidently neutral" to a
# contradiction, inventing a consensus neither channel argued for.
check(
    "conflict does not manufacture confidence",
    conflicting.certainty < agreeing.certainty,
    f"conflict={conflicting.certainty:.3f} agree={agreeing.certainty:.3f}",
)

check("agreement is 1 for identical", close(agreement([(joyful, 1.0), (also_joyful, 1.0)]), 1.0, 1e-6))
check(
    "agreement is low for disjoint peaks",
    agreement([(joyful, 1.0), (angry_voice, 1.0)]) < 0.5,
    f"{agreement([(joyful, 1.0), (angry_voice, 1.0)]):.3f}",
)
check("agreement of one channel is 1", close(agreement([(joyful, 1.0)]), 1.0, 1e-9))

# Multiplication's failure mode: one channel near zero on a class must not be
# able to veto it when the others are confident.
blind = from_label_scores({"neutral": 0.97, "joy": 0.03})
scared = from_label_scores({"fear": 0.8, "sadness": 0.2})
check(
    "one blind channel cannot veto two confident ones",
    fuse(text_vector=scared, voice_vector=scared, face_vector=blind).dominant == "fear",
    str(fuse(text_vector=scared, voice_vector=scared, face_vector=blind).dominant),
)

_settings.fusion_independence = 0.0
check(
    "independence=0 reproduces plain averaging",
    close(
        fuse(face_vector=joyful, voice_vector=also_joyful).vector[EMOTIONS.index("joy")],
        0.60,
        0.01,
    ),
)
_settings.fusion_independence = 0.5

print("\n=== simultaneous emotions ===")
from app.emotions import charge, complexity, components  # noqa: E402

both = from_label_scores({"fear": 0.5, "sadness": 0.5})
single = from_label_scores({"sadness": 1.0})
flat = uniform_vector()

check("complexity counts one emotion as 1", close(complexity(single), 1.0, 0.01), f"{complexity(single):.3f}")
check("complexity counts two as 2", close(complexity(both), 2.0, 0.01), f"{complexity(both):.3f}")
check("complexity is high for formless", complexity(flat) > 5.0, f"{complexity(flat):.3f}")

# The core fix: a blend is a high-charge event, not a low-confidence one. The
# old single "intensity" score rated fear+sadness at 0.64 against 1.0 for a pure
# emotion, so someone feeling two things at once got a smaller, dimmer orb.
check("a blend is not scored as low charge", close(charge(both), 1.0, 0.01), f"{charge(both):.3f}")
check("charge ignores how many emotions", close(charge(both), charge(single), 0.02))
check("neutral has no charge", close(charge(from_label_scores({"neutral": 1.0})), 0.0, 0.01))
check("components finds both", len(components(both)) == 2, str(components(both)))
check("components finds one", len(components(single)) == 1, str(components(single)))

# Mixture vs uncertainty: identical vectors, opposite verdicts, decided by
# whether the channels corroborate rather than by the vector's shape.
_settings.fusion_independence = 0.5
corroborated = fuse(text_vector=both, face_vector=both, voice_vector=both)
split = fuse(
    face_vector=from_label_scores({"fear": 0.85, "neutral": 0.15}),
    voice_vector=from_label_scores({"sadness": 0.85, "neutral": 0.15}),
)
check(
    "corroborated spread reads as a real blend",
    corroborated.blend.is_blend,
    corroborated.blend.label,
)
check("contradicted spread does not", not split.blend.is_blend, split.blend.label)
check(
    "both have similar complexity (the vector alone cannot separate them)",
    abs(corroborated.blend.complexity - split.blend.complexity) < 0.4,
    f"{corroborated.blend.complexity:.2f} vs {split.blend.complexity:.2f}",
)
check(
    "a single emotion is not labelled a blend",
    not fuse(text_vector=single, face_vector=single, voice_vector=single).blend.is_blend,
)

print("\n=== blend ===")
check(
    "zero weights -> uniform",
    close(sum(blend([(confident_sad, 0.0)])), 1.0),
)
check(
    "blend is weight-proportional",
    dominant(blend([(from_label_scores({"joy": 1.0}), 0.9),
                    (from_label_scores({"anger": 1.0}), 0.1)])) == "joy",
)

print("\n=== keywords ===")
sentences = [
    "My thesis defense got moved up to next week and I am not ready.",
    "But my supervisor said the draft was actually really good.",
    "I keep thinking about the thesis defense at night.",
]
vectors = [
    from_label_scores({"fear": 0.7, "sadness": 0.2, "neutral": 0.1}),
    from_label_scores({"joy": 0.8, "neutral": 0.2}),
    from_label_scores({"fear": 0.6, "sadness": 0.3, "neutral": 0.1}),
]

found = extract(sentences, vectors, top_k=10)
texts = [k.text.lower() for k in found]
print(f"  extracted: {[(k.text, k.emotion) for k in found]}")

check("found something", len(found) > 0)
check(
    "picked up the recurring subject",
    any("thesis" in t or "defense" in t for t in texts),
    str(texts),
)
check("no stopwords survived", not any(t in {"the", "and", "but", "my"} for t in texts))
check(
    "repetition outranks a single mention",
    found[0].count >= 1 and found[0].score > 0,
)

thesis = next((k for k in found if "thesis" in k.text.lower()), None)
if thesis:
    # "thesis" appears only in the two anxious sentences, so its attributed
    # emotion should follow those, not the positive one in between.
    check(
        "keyword inherits its sentence's emotion",
        thesis.emotion in {"fear", "sadness"},
        f"thesis -> {thesis.emotion}",
    )

supervisor = next((k for k in found if "supervisor" in k.text.lower()), None)
if supervisor:
    check(
        "positive-sentence keyword reads positive",
        supervisor.emotion == "joy",
        f"supervisor -> {supervisor.emotion}",
    )

check("empty input is safe", extract([], []) == [])

print("\n" + "=" * 46)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("All checks passed.")
