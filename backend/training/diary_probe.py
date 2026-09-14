"""A small diary-domain probe set, for measuring the MELD -> diary domain shift.

MELD is acted, multi-speaker sitcom dialogue. This app is one person, alone,
narrating their day in retrospect. Those differ in almost every way that matters
to a text classifier: person, tense, turn-taking, and how emotion is signalled
(MELD's emotion is largely *performed at* another character; a diary's is
*reported about* the past).

This set exists because MELD test F1 cannot detect that gap — it measures
in-domain performance by construction. These sentences are written in the
register the app actually receives.

Honest caveats, which belong in the write-up alongside any number computed here:

* ~50 items is small. Treat differences under roughly 10 percentage points as
  suggestive, not significant.
* The labels are the *author's intent*, not consensus annotation. A second
  annotator would disagree on some — emotion labelling is genuinely ambiguous,
  which is also why MELD's own inter-annotator agreement is well below ceiling.
* The items were written before seeing any model's output on them, but by
  someone who had seen the model fail on other sentences. That's a weaker
  guarantee than a true held-out set.

Treat this as a diagnostic instrument, not a benchmark.
"""

from __future__ import annotations

from typing import List, Tuple

# (sentence, intended emotion)
PROBE: List[Tuple[str, str]] = [
    # --- joy -----------------------------------------------------------
    ("I finally submitted the paper and I feel so relieved", "joy"),
    ("We got the results back and I actually passed", "joy"),
    ("I spent the whole evening laughing with my flatmates", "joy"),
    ("The sun came out today and I sat outside for an hour just enjoying it", "joy"),
    ("She said yes to coffee on Thursday", "joy"),
    ("I cooked something properly for the first time in weeks and it was good", "joy"),
    ("My supervisor told me the draft was genuinely strong", "joy"),
    # --- sadness -------------------------------------------------------
    ("I just felt empty walking home in the dark", "sadness"),
    ("I miss my family more than I expected to", "sadness"),
    ("Nobody messaged me all weekend", "sadness"),
    ("I keep thinking about how things used to be before I moved here", "sadness"),
    ("I had to put the phone down because I did not want them to hear me crying", "sadness"),
    ("It has been a long time since I felt like myself", "sadness"),
    ("I sat in the library until closing and got nothing done", "sadness"),
    # --- anger ---------------------------------------------------------
    ("My laptop died right before the deadline and I lost everything", "anger"),
    ("I cannot believe they cancelled it again without telling anyone", "anger"),
    ("He took credit for the whole project in the meeting", "anger"),
    ("I am so tired of being the only one who does any of the work", "anger"),
    ("They changed the requirements for the third time this month", "anger"),
    ("The landlord still has not fixed the heating and it is November", "anger"),
    ("I wasted four hours because someone could not be bothered to reply", "anger"),
    # --- fear ----------------------------------------------------------
    ("I have been dreading this presentation all week", "fear"),
    ("I keep worrying that I am going to run out of money", "fear"),
    ("What if I picked completely the wrong subject to study", "fear"),
    ("The results come out tomorrow and I cannot stop thinking about it", "fear"),
    ("I have to call them back and I have been putting it off for days", "fear"),
    ("I am not sure I can keep this up for another two years", "fear"),
    ("Everyone else seems to know what they are doing and I do not", "fear"),
    # --- surprise ------------------------------------------------------
    ("They announced the grant came through, I did not see that coming", "surprise"),
    ("I opened the door and my sister was just standing there", "surprise"),
    ("Out of nowhere he apologised for the whole thing", "surprise"),
    ("I checked my account and there was far more in it than I thought", "surprise"),
    ("Turns out the deadline was moved and nobody told me until today", "surprise"),
    ("I ran into my old teacher in the supermarket of all places", "surprise"),
    # --- disgust -------------------------------------------------------
    ("The state of the shared kitchen this morning was genuinely revolting", "disgust"),
    ("I could not finish the meal, something about it was off", "disgust"),
    ("The way he talks about his coworkers makes my skin crawl", "disgust"),
    ("There was mould growing behind the fridge and nobody had noticed", "disgust"),
    ("I find the whole way that company operates repulsive", "disgust"),
    # --- neutral -------------------------------------------------------
    ("I woke up, had breakfast, and went to the library", "neutral"),
    ("The weather was grey and I did some reading", "neutral"),
    ("I had two lectures today and then went food shopping", "neutral"),
    ("Spent most of the afternoon answering emails", "neutral"),
    ("I took the bus into town and picked up a parcel", "neutral"),
    ("Nothing much happened, it was a fairly ordinary Tuesday", "neutral"),
    ("I did laundry and tidied my room a bit", "neutral"),
    ("Met with my study group for an hour to go over the notes", "neutral"),
]


def counts() -> dict:
    out: dict = {}
    for _, label in PROBE:
        out[label] = out.get(label, 0) + 1
    return out
