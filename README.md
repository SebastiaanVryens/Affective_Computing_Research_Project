# Affective Computing Project

An emotional diary that builds a persistent 3D inner world out of how you actually
felt, captured from face, voice and words while you talk about your day.

Built on the [MELD](https://github.com/declare-lab/MELD) emotion taxonomy — every
modality votes in the same seven classes, which is what makes fusing them
coherent rather than arbitrary.

```
neutral · joy · sadness · anger · fear · disgust · surprise
```

---

## What it does

Press **Talk about your day** and speak. Three things happen at different speeds:

| Layer | Latency | What runs | Where |
|---|---|---|---|
| Live | ~1 frame | face expression, vocal energy | browser |
| Streamed | ~3.5 s | transcription, text emotion, keywords | local sidecar |
| Commit | on stop | full re-analysis, entry written | local sidecar |

**You don't have to press anything to see transcription** — it streams while you
talk. The button only ends the entry.

Measured on this machine (CPU only, Whisper `base.en`): a warm chunk transcribes
at **0.17× realtime** — 3 s of audio in ~0.75 s. Words land roughly 3.5 s after
you say them, which is the floor: a chunk has to be recorded before it can be
transcribed.

The models cold-load in ~12 s, so the browser fires `POST /api/warmup` the moment
it first sees the sidecar. That matters more than it sounds — without it the load
landed on the *first chunk of a recording*, every chunk behind it queued on the
same lock, and live transcription looked broken for the opening of an entry.

The world reacts to your face **immediately** — sky colour, light, particle
motion — because that channel never leaves the page. Words bloom into the scene
a beat later, as the sidecar finishes each chunk.

### The sky

The background is the live emotion vector spread across the whole screen as one
soft field, showing **distinct colours rather than a blend**. Joy and sadness held
together give you yellow flowing into blue, not the grey-green mud that averaging
yellow and blue produces. Same argument as the two-toned orbs: mixing destroys the
thing worth seeing.

**Only the dominant emotions are drawn, and related ones sit together.** Rendering
all seven is faithful to the vector and looks like a children's book. Two
mechanisms reduce it, because neither is sufficient alone:

*Power sharpening* fades a long tail continuously (a faint 15% third emotion drops
to ~4%), but it preserves ratios, so four emotions sitting close together stay
five colours at any exponent. *Nucleus masking* handles that: sorted descending, an
emotion fades once those ahead of it account for 80% of the mood — the same idea
as nucleus sampling in language models. The colour count then adapts to the shape
of the feeling rather than to a fixed number.

Both are continuous. Two emotions swapping rank have near-equal weights and
therefore near-equal masks, so the swap is invisible — where a hard top-K cutoff
would flicker.

The gradient is ordered by **affective similarity** rather than MELD's array
order, walking the valence/arousal circumplex:

```
neutral → sadness → disgust → anger → fear → surprise → joy
 (calm)   (low arousal, negative) ⟶ (high arousal) ⟶ (positive)
```

Several adjacencies are Plutchik's own blend pairs — fear beside surprise is awe,
surprise beside joy is delight — so where two neighbours appear together, the
gradient between them names a feeling that exists. Under the array order, sadness
sat next to joy and every transition read as an arbitrary jump.

What it renders, after focusing and the clarity desaturation:

| live mood | the sky | colours | clarity |
|---|---|---|---|
| pure joy | `#f5ce4e` throughout | 1 | 0.83 |
| bittersweet (joy + sadness) | `#4977b3` → `#99a089` → `#e9c85e` | 2 | 0.64 |
| fear + sadness | `#4977b3` → `#6c6cbb` → `#8e61c3` | 2 | 0.64 |
| joy-led, real sadness | `#8f988c` → `#e0c56b` | 2 | 0.50 |
| four-way mess | muted `#566e8e` → `#cebd85` | 4 | 0.21 |
| nothing detected | near-greys | — | 0.00 |

The bottom two rows are the point: **the sky is most colourful when it is most
sure.** A flat distribution has no ranking to cut, so focusing cannot reduce it —
but flat is exactly where clarity is lowest, and the desaturation pulls it to grey.
Uncertainty looks uncertain rather than looking like a rainbow.

**The shape holds still; only the colour changes.** Where each emotion sits is
decided by a domain-warped noise field — noise sampled at coordinates displaced by
other noise, which turns round blobs into the long curling forms that read as flow
rather than as clouds. That field drifts by a measured 0.5% over five seconds and
2.6% over thirty: invisible in the moment, enough to avoid looking frozen over
several minutes.

Holding it still is deliberate. If the field slid around, a change of feeling would
arrive as *motion*, and motion competes with colour for attention. Static shape
plus a ~4 second colour time-constant means a mood shift is something you notice
having happened rather than something you watch happen.

**Two layers, both gradients.** The upper sky is how you feel now; the lower is
every entry you have ever saved, darkened so it reads as ground. Today's feeling
sitting on everything that came before is the conceit of the whole world.

That lower layer used to be a single averaged colour, which flattened a whole diary
of distinct feelings into one tint — the same mixing-is-loss problem the rest of
the app exists to avoid. It is now a real gradient through the same field, focused
more gently than the live sky: a diary spanning months genuinely does contain
several emotions, and cutting it to one or two would misreport it.

**An idle world is not grey.** With no face, voice or words arriving, the mood
settles to the lifetime distribution rather than to neutral — so opening the app
shows the world you have actually built, already coloured before the camera has
produced a single reading. A diary of 40% sadness, 32% joy and 18% calm opens onto
a sky running `#3f7fd2` through to `#fed240`. Only a diary with no entries opens
grey, which is honest.

Emotions occupy **fixed slots** rather than a top-N list, so a feeling rising or
fading changes only its own share instead of reshuffling the whole sky. The UI
glass picks up a much fainter version of the same tint.

### The world

When you stop, the session crystallises into a permanent orb in the galaxy,
coloured by its dominant emotion. **The world is never reset.** Watching it fill in
over weeks is the point.

Orbs sit on a golden-angle spiral, and each entry's position is a pure function of
its index *stored on the entry* — so tuning the layout later never relocates
memories you already have.

An entry the fusion layer judged a genuine blend gets a **two-toned orb**: the
primary emotion swept across into the secondary, which is what an Inside Out core
memory looks like and the film's whole argument in one object. Single-emotion orbs
stay one colour, so two tones always mean something.

Any entry can be promoted to a **core memory**, lifting it onto a glowing ring at
the centre. That's a deliberate user action, not an automatic threshold — the app
can tell which moment was most *intense*, but only you know which was *formative*.

**History** shows day / week / month dots, each a pie sliced by that period's
emotion proportions. Days you didn't write show as hollow rings, because the gaps
are part of the record.

Slices and legend are both ordered largest-first, so every dot starts its dominant
emotion at twelve o'clock and the legend reads left-to-right in the same order the
pie runs clockwise. The legend lists only what was actually present, with
percentages — a static key in array order gives every emotion equal billing and
still names Anger and Fear on a week that contained neither.

---

## Privacy

Designed in rather than bolted on:

- Camera frames are scored **in the browser** by face-api. Video never leaves the page.
- Audio goes only to a sidecar **on your own machine**. Whisper is local.
- The backend is **stateless** — audio in, numbers out, nothing written to disk.
- Diary entries live in **IndexedDB** in your browser. Export or wipe from History.
- The **clinical PDF is rendered in the browser too**, via print-to-PDF. Sending
  diary content to the sidecar to render it would have broken the one guarantee
  the app makes, so it does not.

No cloud service sees any of it.

### Sharing with a clinician

History → **Summary for a doctor (PDF)** asks for a name and a period, then
produces a printable report: adherence, day-by-day chart, trend direction,
day-to-day variability, within-day swing, consecutive negative days, time-of-day
pattern, a slope per emotion, anger and disgust as their own tracked dimension,
positive affect and flatness, recording behaviour over time, verbatim quotes from
the most marked entries, quoted passages by subject, recurring themes, and which
channels the summary actually rests on.

Five things about it are deliberate:

**It computes nothing that resembles a screening score** and flags no risk. A
false negative from a face model has no business reassuring anyone, and a false
positive has no business alarming them.

**The dimensions describe what was expressed, never what it means.** Anger and
disgust are tracked together — contempt, which is most of what "hostility" means
clinically, falls between them and MELD has no label for it — but the section is
titled *expressed*, because the distance between "expressed a lot of anger" and
"is aggressive" is the distance between a mood diary and a risk assessment. The
same rule governs the flatness proxies: they are named flatness rather than
anhedonia, since anhedonia is about anticipated pleasure and this data cannot
reach it. Each slope is also fitted per emotion, because fear rising while
sadness falls nets out in the aggregate to "broadly flat" — true about the
average, misleading about the person.

**Subjects are surfaced by quoting, never by scoring.** `state/lexicon.ts` matches
a table of phrases — self-harm, hopelessness, anger, panic, substances, sleep,
isolation, functioning — against the transcripts, and every hit is rendered as
the sentence it appeared in with the matched words marked. There is no score, no
severity, no risk level, and no aggregate across categories, because nothing in
a word list resolves negation, idiom, tense, or someone else's story: "we watched
a film about suicide prevention" matches, and the only design that survives being
wrong about it is one where the reader sees the sentence. Categories with no hits
are omitted rather than printed as "none found" — "none found" reads as
reassurance, and matching cannot support that. Adding a dimension is one entry in
`CATEGORIES`. When self-harm or hopelessness language is present, the export
screen says so before the document is generated and offers crisis resources — the
person handing this to someone should know what it quotes back.

**Every derived figure carries its sample.** A trend needs 7 recorded days, a
variability figure needs 5 adjacent pairs, within-day range needs 3 days with
more than one entry, and a change in recording behaviour needs 3 weeks — below
that the report says so rather than printing a confident number fitted through
four points. Fewer than five entries triggers a caution box. The flatness
thresholds have no validated basis, so they are printed in the section that uses
them rather than buried: a reader who would have drawn the line elsewhere can
only discount the figure if they can see where it was drawn.

**Themes are filtered twice, because keywords are stored.** The backend has two
keyword extractors and they are not equally good: YAKE filters statistically and
never lets "something" or "expected" through, while the built-in RAKE-style
fallback ranks by phrase length and repetition alone. With no notion of how
*ordinary* a word is, the fallback floats the words a person says in every entry
straight to the top of a table labelled "recurring themes" — they recur, but they
are not themes. `keywords.py`'s stoplist now covers that closed class
(indefinite pronouns, light and mental verbs, generic nouns of time and
quantity), and `state/stopwords.ts` applies the same rule again at display time,
since keywords are extracted once and stored in the entry — no backend change
reaches a diary already recorded. Both keep out anything a diary is plausibly
*about*: "alone", "tired", "sleep", "money", "night", "stress". Over-filtering
silently deletes the one word that mattered, which is the worse failure.

**Provenance is a section, not a footnote.** If 80% of entries had the camera off,
the summary is a summary of *words*, and the reader can see that instead of
assuming three channels were always present.

Surprise is counted as neither positive nor negative, despite MELD's own sentiment
mapping calling it positive — a shock and a delight are both surprise, and
counting it as positive would quietly inflate every positive figure in a document
someone might make decisions from.

The raw JSON export is still there for backup and portability.

The report is rendered into the live DOM and printed, rather than into an
iframe. That makes the print lifecycle load-bearing: `window.print()` blocks
until the dialog closes in Chrome and Firefox, so `afterprint` fires while that
call is still on the stack. Registering the listener *after* calling print means
it is attached after the event has already passed, cleanup never runs, and the
next export appends a second report beside the first — both match
`#print-report`, so the PDF comes out containing the whole document twice. The
listener therefore goes on before print, cleanup removes every matching node
rather than just its own, and a re-entrancy flag stops a double-click starting a
second run.

---

## Running it

```powershell
.\run.ps1
```

Starts the sidecar in its own window, waits for it, fires the warmup, opens the
browser, runs the frontend. Flags: `-NoBackend` (face channel only), `-Gpu`.

Or manually, two windows — PowerShell 5.1 has no `&&`, and every command uses the
venv's interpreter explicitly because that's where the packages live:

```powershell
# window 1
cd backend
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000

# window 2
cd frontend
npm run dev
```

Must be `localhost`, not a LAN IP — `getUserMedia` needs a secure context.

### First-time setup

```powershell
cd frontend
npm install
npm run fetch-models          # copies face-api weights out of node_modules

cd ..\backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For CUDA, install torch from the matching index instead — `cu130` has builds
matching torch 2.14:

```powershell
.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps `
    --index-url https://download.pytorch.org/whl/cu130 torch==2.14.0+cu130
```

### Training the text model

The app ships working without this — it falls back to a pretrained checkpoint that
happens to emit exactly MELD's seven labels.

```powershell
cd backend
.venv\Scripts\python.exe -m training.prepare_meld     # CSVs only (~5MB, not video)
.venv\Scripts\python.exe -m training.prepare_corpora  # GoEmotions + DailyDialog
.venv\Scripts\python.exe -m training.train_text --model roberta-base --device cuda `
    --epochs 4 --corpora meld,goemotions,dailydialog
.venv\Scripts\python.exe -m training.eval_diary --sweep
```

The checkpoint lands in `backend/models/meld-text/` and the backend picks it up on
next start. `meld_results.json` beside it has the per-class table.

---

## Results

### Shipped model

`roberta-base`, trained on MELD + GoEmotions + DailyDialog, 2 turns of context,
class-weighted, blended at inference with a general-domain model.
**MELD test weighted-F1 0.607.**

The MELD paper's own text-only baselines sit around 0.55–0.57 (text-CNN 0.550,
bcLSTM 0.564, DialogueRNN ≈ 0.570), so this is at or above published numbers.

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| neutral | 0.817 | 0.651 | 0.725 | 1256 |
| joy | 0.617 | 0.602 | 0.610 | 402 |
| sadness | 0.343 | 0.399 | 0.369 | 208 |
| anger | 0.472 | 0.507 | 0.489 | 345 |
| fear | 0.171 | 0.280 | 0.212 | 50 |
| disgust | 0.239 | 0.309 | 0.269 | 68 |
| surprise | 0.452 | 0.698 | 0.548 | 281 |

Read the precision/recall asymmetry as confirmation the class weighting worked.
Neutral has high precision (0.817) but depressed recall (0.651), while fear,
disgust and surprise all show recall above precision. The model is pushed to *risk*
guessing rare emotions instead of retreating to neutral — the behaviour a diary
needs, and it costs accuracy to get. `--no-class-weights` shows the trade.

### Capacity vs. data domain

Three conditions, each changing exactly one variable. Dev and test stay pure MELD
in every run, so the headline number stays comparable across them and to the paper.

| | Training data | Model | MELD wF1 | MELD macro | Diary acc | Diary macro |
|---|---|---|---|---|---|---|
| **A** | MELD | distilroberta, 4ep | 0.578 | 0.430 | 0.532 | 0.511 |
| **B** | MELD | roberta-base, 8ep | 0.604 | 0.447 | 0.574 | 0.562 |
| **C** | MELD + GoEmotions + DailyDialog | roberta-base, 4ep | **0.607** | **0.460** | **0.617** | **0.605** |
| **C + ensemble** | (as C, blended with general model) | — | — | — | **0.638** | **0.621** |

**The dissociation between B and C is the result worth reporting.** Adding 50k
out-of-domain examples moved MELD test F1 by +0.003 — noise — while moving diary
accuracy by **+4.3 points**. Extra data bought essentially nothing in-domain and a
great deal out-of-domain: direct evidence the remaining gap was *domain*, not model
capacity, and a claim a single-corpus setup cannot make.

B vs A separates the other factor: capacity and training length helped both axes
evenly (+2.6 MELD, +4.2 diary), confirming A was undertrained rather than
mis-specified — its dev curve was still climbing 2.5 points in the final epoch.

Recall on the two failing classes:

| | anger | fear | anger→sadness confusion |
|---|---|---|---|
| A | 1/7 | 1/7 | 4/7 |
| B | 2/7 | 2/7 | 3/7 |
| C | **3/7** | **3/7** | **1/7** |

Repaired by the predicted mechanism: GoEmotions' `anger` bucket folds in
*annoyance* and *disapproval*, exactly the low-arousal register diary anger uses
and sitcom anger does not.

### The domain-shift finding

The most interesting result here, and the one MELD test F1 cannot show — that
metric measures in-domain performance by construction.

`training/eval_diary.py` scores three conditions on a 47-sentence diary-register
probe set (`training/diary_probe.py`): first-person, retrospective, single speaker.

| Condition | Accuracy | Macro-F1 | Calibration |
|---|---|---|---|
| Fine-tuned only (shipped model C) | 0.617 | 0.605 | +0.157 |
| General pretrained only | 0.596 | 0.592 | +0.054 |
| **Ensemble (w=0.5)** | **0.638** | **0.621** | **+0.182** |

**A MELD-only model is worse on diary text than on MELD.** Condition A scored 0.532
here against 0.578 in-domain. Training on *Friends* alone buys in-domain accuracy
and gives some back on the target domain.

**The ensemble is right for a specific, measurable reason.** It beats both on
accuracy and has **three times the general model's calibration separation** — the
gap between mean confidence when right and when wrong. That matters more than
accuracy here, because `app/fusion.py` weights each modality by its own certainty.
A confidently-wrong text channel drags the world's colour off; an accurately-unsure
one steps aside and lets face and voice carry the moment. The weight sweep is flat
from 0.0–0.6 and degrades after, so 0.5 is defensible rather than tuned.

**Anger and fear collapse into sadness.** In condition A, recall on anger was 1/7
for *both* the MELD model and the general model, with → sadness the dominant
confusion in each. The reason is register, not capacity: diary anger reads *"I'm so
tired of being the only one who does any of the work"* — the arousal that marks
anger in speech is absent from retrospective first-person text.

That is a direct argument *for* the multimodal design rather than a limitation of
it: the arousal text loses is exactly what prosody and face still carry.

Caveats that belong in any write-up: 47 items is small (treat sub-10-point gaps as
suggestive), the labels are author intent rather than consensus annotation, and the
probe set was written by someone who had already seen the model fail on other
sentences. A diagnostic instrument, not a benchmark.

### Auxiliary corpora

`training/prepare_corpora.py` maps additional corpora into MELD's label space, with
mappings chosen to need as little judgement as possible:

* **GoEmotions** (Reddit, 25k after capping). Google publishes an Ekman grouping of
  its 27 fine labels that collapses to *exactly* MELD's seven. Multi-label examples
  spanning two Ekman buckets are dropped rather than arbitrarily resolved (3,855).
* **DailyDialog** (everyday conversation, 25k after capping). Its seven labels
  already *are* ours. 83% "no emotion", so neutral is subsampled.
* **dair-ai/emotion** (tweets) is wired up but off by default: six labels, no
  neutral or disgust, and `love` has to fold into `joy`.

Auxiliary rows are single utterances with unique dialogue IDs, so the context
builder is a no-op on them — prepending an unrelated Reddit comment as a "prior
turn" would be worse than no context.

**ISEAR** would be the best fit of all — literally *"describe a situation in which
you felt X"* — but is not currently loadable from the HuggingFace Hub under any of
the usual identifiers. Worth sourcing separately.

---

## The research angles

Each is a deliberate choice with a flag that turns it off.

**Class-weighted loss.** MELD is severely imbalanced: neutral is 47.2% of training,
fear 2.7% and disgust 2.7%. Unweighted cross-entropy yields a model with decent
accuracy that has learned to answer "neutral" — useless for a diary.
`--no-class-weights` gives the comparison.

**Dialogue context.** MELD utterances are conversational fragments; *"I can't
believe it"* is unclassifiable alone. Prepending prior turns is the cheapest large
win on this dataset. `--context 0` ablates it.

**Certainty-weighted late fusion.** Plain averaging lets a modality that has no idea
vote at full strength — three shrugs outvote one confident reading. Each channel's
weight scales by `1 - normalised entropy` of its own output, so the scheme
self-gates: the face channel stops mattering while you sit with a resting face and
takes over the moment your expression changes.

**Cross-modal reinforcement.** Weighted averaging is a *linear opinion pool* and its
output is bounded by its inputs: two channels both saying "joy 0.60" produce exactly
0.60, no more confident than either alone. That's wrong — corroboration is evidence.
The default is a *logarithmic* pool, where channels multiply and the total exponent
may exceed 1:

| scenario | dominant | certainty |
|---|---|---|
| one channel (joy 0.60) | joy | 0.518 |
| two channels agree (both joy 0.60) | joy **0.706** | **0.589** |
| two channels conflict (joy vs anger) | neutral | **0.267** |

`MINDSCAPE_INDEPENDENCE=0` recovers exact averaging — the ablation. The default 0.5
reflects that face, voice and words from one person in one moment share a cause and
are *not* independent sensors.

Two failure modes are guarded, both in the smoke test:

* **Veto.** Multiplication lets one channel reporting ≈0 kill an emotion outright.
  Every distribution is floor-smoothed before pooling, so two channels confident
  about fear still win against a third that can't see it.
* **Manufactured consensus.** A pure geometric pool answers a contradiction with
  whatever class the channels merely *tolerate* — a smiling face against an angry
  voice yields a confident "neutral" neither argued for. So the operator itself is
  chosen by agreement (mean pairwise Bhattacharyya coefficient): corroboration pools
  multiplicatively, contradiction pools additively and reports honest uncertainty.

**Time-aligned fusion.** The browser sends a *timeline* of face readings, not one
session average, and each Whisper segment fuses against the face from that moment
(±0.5 s, since expression and speech aren't synchronous). This is what lets the
channels corroborate each other at all — a face averaged over two minutes cannot
confirm any particular sentence.

Measured on a 15 s entry where the speaker is anxious then relieved:

| | entry reading | per-segment |
|---|---|---|
| session-average face | joy, certainty 0.278 | — |
| time-aligned | joy, certainty 0.204 | sadness → fear → **joy** |

The aligned certainty is *lower*, which is correct: the entry contains three
feelings and a confident single label would be a lie. The gain is the arc.

Concretely: *"Afterwards my supervisor told me I still passed"* scores **joy at
0.692** fused with the relieved face from those seconds — the most confident reading
in the entry. The text channel alone calls that same sentence **sadness**. That one
sentence is the clearest demonstration of why three channels beat one.

**Emotions are never single.** A probability vector over seven classes holds seven
numbers, but it sums to 1, forcing a conflation: `fear 0.5, sadness 0.5` is the
*identical* vector whether both were felt at once or the model couldn't decide.
Worse, a lone `1 - entropy` intensity score rates that blend at 0.644 against 1.0
for a pure emotion — so someone feeling two things intensely got a smaller, dimmer
orb. Backwards.

The fix stops one number carrying three facts:

| measure | question | independent of |
|---|---|---|
| **charge** | how much is happening? | which emotions, how many |
| **complexity** | how many emotions? | how strongly felt |
| **supported** | should we believe it? | both |

`supported` is the interesting one. **A single channel genuinely cannot tell a
mixture from an uncertainty** — the information isn't in the vector. Three can: if
face, voice and words *independently* converge on the same two emotions, that's
corroboration; if they each point elsewhere, it's disagreement. Two readings with
near-identical complexity get opposite verdicts:

| | complexity | verdict |
|---|---|---|
| all three channels say fear + sadness | 2.00 | **real blend** |
| face says fear, voice says sadness | 2.03 | *unclear* |

On real diary text: *"last day at the job… excited but I cried saying goodbye"* →
**sadness and joy**; *"the exam is over, terrified all week, now I feel light"* →
**fear and joy**.

Two gates stop it over-firing. A flat entry's residual few percent smears across
classes and looks exactly like a rich three-way blend to anything inspecting only
proportions — so a blend needs real `charge` behind it. And a spread with a clear
leader is reported as that emotion rather than discarded as "unclear".

---

## Honest limitations

- The **voice channel** is the weak one. The wav2vec2 head is RAVDESS-trained
  (acted, not conversational) and its "calm" class has no MELD counterpart. When
  disabled, the fallback is a **prosody heuristic**, not a classifier — don't report
  it as one. The API tags which tier ran (`"source": "prosody-heuristic"`).
- **face-api's expression net** is trained on posed-ish data and reads a resting
  face as slightly sad. The entropy weighting mitigates this, but the bias is real.
- **Fear at F1 0.212** is weak — 50 test examples against 1,256 neutral.
- **Train/serve format must match.** The checkpoint trains on
  `"prev </s> prev </s> current"`; `app/text_model.py` reads `context_turns` out of
  the checkpoint and reproduces it at inference. Serving bare sentences to a
  context-trained model cost roughly half the model's confidence and produced no
  error at all.
- `--speaker-tokens` helps the MELD benchmark but a diary has one speaker, so expect
  the gain to be in-domain only — a case where benchmark and application diverge.

---

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `MINDSCAPE_DEVICE` | `cpu` | `cuda` or `auto` |
| `MINDSCAPE_WHISPER_MODEL` | `base.en` | `small.en` is better, ~3× slower |
| `MINDSCAPE_VOICE_EMOTION` | `true` | `false` drops to the prosody heuristic |
| `MINDSCAPE_W_TEXT` / `_FACE` / `_VOICE` | `0.5` / `0.3` / `0.2` | fusion weights |
| `MINDSCAPE_INDEPENDENCE` | `0.5` | cross-modal reinforcement; `0` = plain averaging |
| `MINDSCAPE_ENSEMBLE_TEXT` | `true` | blend MELD head with general-domain model |
| `MINDSCAPE_ENSEMBLE_W` | `0.5` | weight on the MELD head within that blend |

---

## Layout

```
frontend/src/
  emotions.ts          the 7-class taxonomy, colours, focusing, gradient order
  capture/
    mood.ts            live mood bus — the seam between capture and world
    face.ts            face-api loop, 8 Hz, in-browser, timestamped
    mic.ts             audio level (60 Hz) + two-recorder chunking
    session.ts         orchestrates live / streamed / commit
  world/
    scene.ts           camera, render loop, bloom
    atmosphere.ts      sky shader, lights, motes — the fast layer
    orbs.ts            memory galaxy + core ring — the permanent layer
    placement.ts       one definition of where an orb goes
    keywords.ts        floating words
  state/
    db.ts              IndexedDB, export/import
    history.ts         day/week/month rollups
    report.ts          the clinician summary's numbers, each with its sample
    lexicon.ts         phrase table per subject — quotes hits, scores nothing
    stopwords.ts       display-time theme filter, mirrors keywords.py's stoplist
  ui/                  HUD and overlay views

backend/app/
  emotions.py          the same taxonomy (label order is load-bearing)
  fusion.py            certainty-weighted late fusion, log pooling, blend profile
  text_model.py        MELD checkpoint + general-model ensemble
  audio_model.py       wav2vec2 SER (background-loaded) + prosody fallback
  asr.py               faster-whisper, with VAD retry and level reporting
  keywords.py          keyword extraction with emotion attribution
backend/training/
  prepare_meld.py      fetch + sanity-check the CSVs
  prepare_corpora.py   map GoEmotions / DailyDialog into MELD's labels
  train_text.py        fine-tune, evaluate, save
  eval_diary.py        the domain-shift measurement
  diary_probe.py       the 47-sentence probe set
  smoke_test.py        dependency-light checks for the fusion arithmetic
```

## Adding 3D assets

Orbs are `IcosahedronGeometry` in `world/orbs.ts`. Swap in a glTF and placement,
colour, two-tone blending and scaling all still work — entry positions are stored
on the entry, so tuning the layout never relocates existing memories.
