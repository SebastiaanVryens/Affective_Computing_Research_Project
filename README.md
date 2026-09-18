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

**An orb's size and glow carry how loudly it was spoken.** An animated entry
swells and brightens; a quiet late-night one stays small and dim.

This is the honest replacement for what the voice channel was supposed to do.
Emotion-from-tone was meant to catch the thing words cannot — *"I'm fine"* said
flatly and *"I'm fine"* said brightly are the same transcript — and measured on
two corpora it does not deliver: it scores at the class prior and drags fusion
down. Loudness delivers exactly that distinction, and has the advantage of being
**measured rather than inferred**. An RMS meter cannot be confidently wrong the
way a classifier can. So the world shows vocal intensity directly, labelled as
what it is, instead of a model's guess about what that intensity meant.

**Ranked against your own diary, never an absolute scale.** A headset and a
laptop array microphone disagree by more than a whisper differs from a shout, and
leaning closer to the desk changes it again — an absolute mapping would mostly
encode which machine you sat at. Each entry is scored by its percentile within
your own history, so your loudest entry is the loudest whatever your hardware
does, and the spread fills the visual range whether you are a quiet talker or a
loud one.

Four details that matter more than they look:

* **Raw values are stored; ranking happens at read time.** Normalising on save
  would freeze each entry against whatever history existed that day.
* **Only frames above the noise floor count**, so a long thinking pause cannot
  quietly average an entry down towards silence.
* **The effect is multiplicative and centred on 1**, so it modulates the existing
  meaning (certainty, duration) rather than competing with it — and an entry with
  no measurement keeps exactly the size it had. A diary written before this
  existed looks unchanged, instead of every orb silently becoming "average".
* **Size and glow say the same thing deliberately.** One cue is ambiguous against
  the orb's own emotion colour; two agreeing cues read instantly.

Promotion to a core memory preserves both, which needed care: those are separate
meshes whose scale and emissive intensity are rewritten every frame, so the vocal
factors ride in `userData` and are re-applied in the update loop rather than baked
in once. Without that, promoting a loud entry would visibly shrink and dim it.

Spectral brightness — a raised, tense voice against a calm one — is captured and
ranked alongside, and currently unused. It is a second axis if loud-and-tense
should ever look different from loud-and-warm.

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

### Training the face model

The face channel is the one piece of this system that was never trained on
anything: `face.ts` loads face-api's stock expression head, and the complaint
about it under *Honest limitations* has never had a number attached to it. These
two scripts fix that, using the half of MELD that `prepare_meld.py` deliberately
skips — the raw video.

MELD's clips are the only public face data in **MELD's own seven labels on MELD's
own splits**. That matters more than the extra images: face and text can finally
be scored on the same held-out utterances, which is what turns the fusion
argument from a worked example into an ablation.

Download `MELD.Raw.tar.gz` (~10GB) from the [MELD
repo](https://github.com/declare-lab/MELD) and extract it anywhere — the script
globs for `dia*_utt*.mp4` rather than assuming a directory layout, because the
tarball has been repackaged more than once.

```powershell
cd backend
.venv\Scripts\python.exe -m pip install opencv-python Pillow
# dev + test first: that's all the baseline measurement needs
.venv\Scripts\python.exe -m training.prepare_meld_video --source D:\MELD.Raw `
    --splits dev,test --save-frames
.venv\Scripts\python.exe -m training.prepare_meld_video --source D:\MELD.Raw --splits train
```

#### Measure before you train

Run this first. It answers a question training cannot: whether the stock head's
problem is **accuracy** or **calibration** — which have very different fixes, and
only one of them is expensive.

```powershell
cd frontend
node scripts/eval-faceapi.mjs --manifest ..\data\meld_faces\dev_frames.csv `
    --root ..\data\meld_faces --out ..\data\meld_faces\faceapi_dev.json
node scripts/eval-faceapi.mjs --manifest ..\data\meld_faces\test_frames.csv `
    --root ..\data\meld_faces --out ..\data\meld_faces\faceapi_test.json

cd ..\backend
.venv\Scripts\python.exe -m training.eval_face_baseline
```

`eval-faceapi.mjs` runs the **same three nets from the same `public/models`
weights the browser fetches**, with `face.ts`'s own `inputSize` and 0.35 score
floor, on whole frames so face-api does its own detection. Scoring it only on
frames *our* detector already approved would hand it a pre-filtered test set and
call the comparison fair. It runs on the WASM backend — slower than
`tfjs-node`, numerically identical, and no native toolchain on Windows.

`eval_face_baseline.py` reports coverage, F1, ECE with a temperature fitted on
dev, a confusion matrix, and a **neutral-drift test** that checks the README's
actual claim: given a frame MELD labels neutral, where does face-api put its
probability mass? An inaccurate head spreads errors around; a biased one puts
them somewhere specific.

If that comes back *accurate but overconfident*, the fix is one scalar — no
training, no new weights in the browser — and `fusion.py`'s entropy weighting
stops handing the face channel weight it hasn't earned. Take that either way.

#### Then train

```powershell
.venv\Scripts\python.exe -m training.train_face --device cuda --epochs 8
```

Extraction samples 8 frames from the middle of each clip, detects faces with
YuNet, links them into tracks and keeps the one that is large, central and
persistent — the speaker, usually. It writes `data/meld_faces/` plus a per-class
**coverage table**, which is a result rather than a log line: clips where no face
is found are dropped, that loss is not uniform across emotions, and a face
channel that sees disgust less often than joy has a bias no amount of training
removes.

Training reports four numbers instead of one, for reasons specific to how the
face vector gets consumed:

- **frame-level** is comparable to other FER work; **clip-level** mean-pools the
  frames of an utterance, which is what `face.ts` actually does before fusion
  sees anything. The gap between them is itself a finding.
- **ECE, before and after temperature scaling.** `fusion.py` weights each channel
  by the entropy of its distribution, so a head that is confidently wrong doesn't
  just add noise — it *wins weight* while doing it. A model with better top-1 and
  worse calibration would make the fused reading worse. The fitted temperature is
  saved beside the weights; inference has to apply it.
- **A prior baseline.** Neutral is half of MELD, so the accuracy floor is
  deceptively high. A face model that fails to beat the prior on macro-F1 has
  learned nothing the fusion layer needs.

`train_face.py` also exports `face-emotion.onnx` with the softmax inside the
graph, plus a `preprocess.json` recording the exact crop/resize/normalise
recipe — that pairing is what keeps a browser-side head from repeating the
train/serve mismatch that once cost the text model half its confidence.

**Nothing serves this yet.** `face.ts` still loads face-api. Measure the trained
head against that baseline before rewiring the capture path — the honest
comparison is the point, and swapping first would throw it away.

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

### Capping the auxiliary corpora by class, not proportionally

`--cap-mode per-class` keeps every example of the rare classes and takes the cut
out of joy and neutral instead. The original proportional cap preserved each
source corpus's balance, which sounds neutral and isn't: these corpora exist to
feed the classes MELD starves, and trimming them proportionally trimmed the rare
classes too. DailyDialog went 87k → 25k and took fear down to **93 examples**,
fewer than MELD's own fear count, for a corpus added to fix exactly that.

Selection is on dev; test is quoted once for the chosen run.

| Run | Aux corpora | dev wF1 | test wF1 | test macro | fear F1 | disgust F1 |
|---|---|---|---|---|---|---|
| C (previous) | proportional, go+dd | — | 0.6068 | 0.4602 | 0.212 | 0.269 |
| D | per-class, go+dd | 0.5987 | 0.6180 | 0.4620 | 0.205 | 0.251 |
| **E (shipped)** | per-class, go+dd+tweets | **0.6031** | **0.6291** | 0.4691 | 0.205 | 0.312 |

**More data does not fix fear, and that is the finding.** Fear's auxiliary
examples went 435 → 652 → **2,585** across these three runs — 5.9× — and its F1
went 0.212 → 0.205 → 0.205. The class weighting is already pushing hard on it
(weight 2.69). MELD's fear is acted sitcom panic across 50 test examples; the
tweets corpus's fear is *"i feel anxious"*. Same label, different phenomenon —
the same domain gap measured in B→C, reappearing **inside a single class**, where
more data cannot cross it.

Read the +0.022 headline gain with that in mind: macro-F1 moved only +0.009, so
almost all of it is neutral improving (0.725 → 0.776). Single seed, and disgust
swung 0.251 → 0.312 between two runs with *identical* disgust data, so treat any
difference under ~0.05 at low support as noise.

### The face channel, measured

Until now this channel had a sentence in *Honest limitations* and no number. It
has both halves of a comparison now: face-api's stock head, and a MobileNetV2
head trained on MELD's own video via `prepare_meld_video.py` + `train_face.py`.
Same clips, same labels, same splits, so the difference is the model.

MELD test, **clip-level** (frames of an utterance mean-pooled, which is what
`face.ts` does before fusion sees anything):

| | weighted-F1 | macro-F1 | accuracy | ECE | fitted T |
|---|---|---|---|---|---|
| face-api (shipped) | 0.3150 | **0.1651** | 0.3286 | 0.3451 | 7.12 |
| trained, unweighted | **0.3462** | 0.1466 | **0.4206** | **0.0980** | 2.25 |
| trained, class-weighted | 0.2675 | 0.1517 | 0.2450 | 0.1145 | 4.48 |
| predict the class prior | 0.3128 | 0.0928 | 0.4813 | 0.0054 | — |

**We did not ship the trained head.** It wins weighted-F1 and accuracy, but it
loses macro-F1 — and this project's own argument for class-weighted text training
says macro is the metric a diary needs: *pushed to risk guessing rare emotions
instead of retreating to neutral*. The trained head's advantage comes almost
entirely from predicting neutral better (recall 0.795), which is the retreat the
text pipeline was designed to avoid. It is also trained on acted sitcom
expression under TV lighting, which is the B→C domain gap again.

Three things here are worth more than the ranking:

**Face-only recognition on MELD saturates around 0.31–0.35 weighted-F1.** Three
independent attempts land in that band, and none beats the class prior on
accuracy. The earlier "face-api is weak" reading was wrong: it is close to the
ceiling of the task as MELD poses it.

**Fear and disgust are unreachable — 0.000 F1 for *both* models**, on 45 and 63
test clips. Not a tuning problem.

**Half the frames never reach the classifier.** face-api detects a face in only
**48.5%** of test frames (9,976/20,568), while YuNet finds a speaker in 94% of the
same clips. `sweep-detector.mjs` shows why, and it is almost purely apparent face
size:

| face width (% of frame) | frames | face-api finds it |
|---|---|---|
| 0–10% | 3,186 | ~4% |
| 10–15% | 4,193 | 23.6% |
| 15–20% | 7,405 | 79.3% |
| 20–30% | 1,999 | 91.3% |

There is a cliff between 10% and 20% of frame width, and MELD's wide two-shots sit
on it. **This is a property of the benchmark, not of the app**: someone at a
laptop spans 25–40% of the frame and lands in the 91% band. It caps any face-only
result on MELD, which is a limitation of the *evaluation*.

Finally, the README's long-standing claim is now quantified. On frames MELD labels
neutral, face-api puts p=0.218 on sadness and **sadness outranks neutral on 31.7%
of them**.

### The voice fallback carries no signal

`training/eval_voice.py` scored the prosody heuristic on 2,487 MELD test clips.

- It predicts **neutral on 100%** of them. Its per-class numbers are identical to
  predicting the class prior.
- Choosing the emotion from its non-neutral mass: **17.3%** against a 16.7%
  chance rate.
- Using `charge` (1 − p(neutral)) to tell an emotional clip from a neutral one:
  **ROC-AUC 0.474** — no signal, marginally the wrong way.

It is not a constant function (224 distinct vectors in 237 clips), but the
variation carries nothing. What makes it *harmful* rather than merely useless is
the shape: a mean **0.787** of its mass sits on neutral, so it is a peaked
distribution, and `fusion.py` weights by entropy — which rewards confidence. An
uninformative channel that is reliably confident earns real weight and drags every
entry toward neutral. With the measured mean vector fused against a weak text
reading of joy, the entry flips to **neutral at 0.719**.

So the prosody tier no longer votes: `predict()` returns `None`, fusion marks the
channel `available: false` and renormalises, and the source tag still reports
which tier ran. `MINDSCAPE_PROSODY_FUSION=true` restores the old behaviour and is
how to run the ablation. Note prosody returns a *uniform* vector on pure tones —
it is harmful specifically when fed speech.

### The neural voice tier was never actually running

The default checkpoint, `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`,
was saved under **transformers 4.8.2** with the old `Wav2Vec2ClassificationHead`
layout (`classifier.dense`, `classifier.output`). Current transformers builds
`projector` + `classifier` instead, so **every head tensor fails to match, is
discarded, and is replaced with a random initialisation.** The 422 wav2vec2
feature-extractor weights load fine; the part that maps features to emotions does
not. `transformers` prints a warning and returns the model. Nothing raises.

This is the same failure class the *Train/serve format must match* note already
describes — a silent quality loss with no error — and `audio_model.py` now refuses
any checkpoint whose `classifier`/`projector` tensors were newly initialised,
naming the cause instead of serving noise.

### A replacement that loads, and what it took to make it usable

`firdhokk/speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53` loads
cleanly (426/426 weights) and emits **exactly MELD's seven classes**, so the
`calm` → `neutral` collapse disappears: the mapping is one-to-one.

Out of the box it scores **0.098** weighted-F1 — six times worse than predicting
the class prior. That number is misleading. The model predicts neutral on **0.8%**
of clips against a 48% true rate, with a mean 0.010 of its mass on the class: it
was trained on acted speech where nearly every clip is expressive. Its *ranking*
is sound, its *operating point* is not.

Restricted to genuinely non-neutral clips it picks the right emotion **27.2%** of
the time against 16.7% chance (joy 34%, anger 38%, surprise 30%). The prosody
heuristic scores 17.3% on that same subproblem — chance. One tier has signal; the
other does not.

A temperature cannot fix a prior mismatch, because it scales every class equally.
A per-class bias fitted on dev can (`fit_prior_correction`), and MELD test then
gives:

| | weighted-F1 | macro-F1 | accuracy | ECE |
|---|---|---|---|---|
| class prior | 0.3115 | 0.0927 | 0.4801 | 0.000 |
| prosody heuristic | 0.3115 | 0.0927 | 0.4801 | 0.320 |
| wav2vec2, as loaded | 0.0981 | 0.1090 | 0.1448 | 0.757 |
| **wav2vec2, prior-corrected** | 0.2954 | **0.1542** | 0.2891 | 0.417 |

The fitted bias is **neutral +6.71**, joy −4.95, anger −3.57 — the size of the
correction is itself the evidence.

**Macro-F1 0.1542 against the prior's 0.0927 is the result**: on the metric this
project argues a diary needs, the voice channel carries signal for the first time,
in the same band as the face channel (0.147–0.165).

**It is still not shipped.** ECE 0.417 means it is badly calibrated, and fusion
weights by entropy — enabling it now would repeat the prosody mistake with a
better model. Two things are needed first: *joint* vector scaling rather than a
temperature and a bias fitted independently (doing both separately double-corrects
and collapses the model to always-neutral), and a refit on diary audio rather than
MELD, since the bias encodes MELD's 48% neutral rate and a diary's is unknown.

### Does fusing the channels actually help? No — and that is the main result

Every channel above was measured alone. The combination — the thing this project
is *for* — had never been. `training/dump_channels.py` writes each channel's
per-utterance vector and `training/eval_fusion.py` scores every subset, sweeps
the independence knob, and tunes the weights on dev.

**MELD test**, weighted-F1, calibrated on dev:

| | wF1 | macro-F1 |
|---|---|---|
| **text alone** | **0.6217** | **0.4405** |
| text + face | 0.6114 | 0.4202 |
| text + voice | 0.6107 | 0.4237 |
| text + face + voice | 0.6005 | 0.3991 |
| class prior | 0.3127 | 0.0928 |

Every channel added makes it worse, monotonically, and the dev-chosen weight
sweep independently zeroed the face channel. On emotional (non-neutral)
utterances it is worse still: adding both channels drops the rate at which the
system says *anything* other than neutral from 69.6% to **55.4%** — the retreat
to neutral this project was built to avoid.

The obvious objection is that MELD is a hostile test: face and voice both score
at the class prior there, and blending a signal-free channel into a good one can
only dilute it.

**So the same experiment was run on CMU-MOSI** — ~2,200 YouTube monologues, one
speaker, close to camera, unscripted. Structurally the diary this app is for.
`training/prepare_mosi.py` and `training/dump_channels_mosi.py` reuse the same
detector, crops, models and fusion code; only the corpus changes. MOSI annotates
sentiment, so scoring collapses seven emotions to three using
`app/emotions.py`'s own `SENTIMENT_OF` table — fusion still runs in seven-space
exactly as in production, and no new mapping is invented.

The domain difference is stark, and it confirms the face-size diagnosis exactly:

| | MELD | CMU-MOSI |
|---|---|---|
| face-api frame detection | 48.5% | **99.4%** |
| YuNet speaker coverage | 93.4% | 99.95% |
| neutral share of labels | 48% | 4% |
| face alone, vs class prior | +0.003 | **+0.104** |

**On MOSI the channels work** — face beats the prior by 10 points of weighted-F1
and 11 of macro, where on MELD it beat it by nothing. And fusion *still* loses:
text alone 0.7582, text+face 0.7120, all three 0.6565.

An oracle weight sweep, tuned directly on test, puts fusion's absolute ceiling at
**0.7601 — +0.002 over text alone**, at weights of 0.9 text / 0.1 face / 0.0
voice.

#### The information is there; the rule cannot reach it

This is not "the auxiliary channels are useless". On MOSI:

* text is wrong on 22.7% of test clips
* on those clips, face or voice is right **49.4%** of the time
* oracle per-clip channel selection scores **0.885** against text's 0.773

That is **+11.2 points** of genuinely complementary information. A fixed weight
vector cannot reach it, because it blends every channel on every clip rather than
deciding *when* to listen.

So `training/eval_stacking.py` replaces the hand-specified rule with a learned
one — still late fusion, still combining only channel outputs, but the combiner
is fitted rather than asserted. Features are each channel's seven probabilities,
its certainty, and an availability flag.

| combiner | MOSI test wF1 | vs text |
|---|---|---|
| weighted averaging (oracle weights) | 0.7601 | +0.002 |
| logistic stacker | 0.7630 | +0.005 |
| MLP (32) | 0.7630 | +0.005 |
| RandomForest (400) | 0.7697 | +0.012 |
| HistGradientBoosting | 0.7508 | −0.007 |

The best recovers about **10% of the available headroom**, and the logistic
stacker's gain decomposes to **+0.0028 attributable to face and voice** — the
rest is refitting on text's own output. (Treat the RandomForest number with
suspicion: it scored *higher* on test than on the 229-clip dev split used to
choose it.)

**The claim that survives**, on two corpora with opposite channel quality:

> Late fusion does not improve on the strongest single channel. On MELD the
> auxiliary channels carry no signal; on MOSI they carry real signal and are
> demonstrably complementary, yet no combiner tested — fixed-weight,
> oracle-weighted, logistic, MLP or tree-ensemble — recovers more than a point of
> it. The complementary information exists at the utterance level but is not
> recoverable from calibrated channel posteriors.

#### What changed as a result

`MINDSCAPE_INDEPENDENCE` now defaults to **0**. The sweep is monotonic on both
corpora: on MOSI, 0.744 at independence 0 falling to 0.644 at 1.0, with the old
0.5 default at 0.657. Cross-modal reinforcement is a good argument that the data
does not support — it holds only when channels are of comparable quality, and a
weak channel agreeing with a strong one adds confidence without adding
information.

Fusion weights move from 0.5 / 0.3 / 0.2 to **0.8 / 0.15 / 0.05**. The old
defaults were measurably worse than serving text alone. Both sweeps put the true
optimum at 0.9 / 0.1 / 0.0; the shipped defaults stop short of a hard zero
deliberately, since that would permanently silence a channel whose model is still
being replaced.

None of this touches the **live** face channel. `mood.pushFace()` drives the sky
and particles at 8 Hz, is 99.4% reliable on webcam-framed video, and is doing a
different job from classifying a committed entry. The measurements above condemn
only the second.

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
  Measured on MELD it carries **no signal at all** (neutral on 100% of clips,
  `charge` ROC-AUC 0.474), so it no longer contributes to fusion — see *The voice
  fallback carries no signal*.
- **The neural voice tier has never worked.** Its default checkpoint loads with a
  randomly initialised classification head (see *The neural voice tier was never
  actually running*), so any voice reading the app has ever produced came either
  from that random head or from the signal-free heuristic. `audio_model.py` now
  refuses such a checkpoint. A working replacement is identified and measured but
  **not enabled**: it needs joint calibration and an in-domain refit first.
- **face-api's expression net** is trained on posed-ish data and reads a resting
  face as slightly sad. Now quantified: on true-neutral MELD frames it puts
  p=0.218 on sadness, and sadness outranks neutral on **31.7%** of them. It is
  also badly calibrated — ECE 0.514 frame-level, fitted temperature **7.12** —
  which matters because the entropy weighting *rewards* confidence rather than
  mitigating it, so this channel has been outvoting the better-calibrated text
  model on certainty it has not earned. A trained replacement exists
  (`train_face.py`) and was **not** adopted: it loses macro-F1, which is the
  metric this project argues a diary needs.
- **The face temperature is not applied at runtime.** T = 7.12 was fitted on
  MELD, where face-api is confidently *wrong*; on easy webcam faces it may be
  confidently right, and that temperature would over-flatten a good signal.
  Refitting on in-domain recordings is the prerequisite for shipping it.
- **Face labels from MELD video are weak by construction.** An utterance carries
  one emotion, and `prepare_meld_video.py` stamps it onto every sampled frame, so
  a neutral-looking frame inside an angry utterance is simply mislabelled.
  Mid-clip sampling and clip-level pooling reduce the damage; they don't remove
  it. Speaker selection is a geometric heuristic on a multi-party sitcom, and
  `rival_faces` in the manifest records how often the choice was close —
  `--unambiguous` trains on the clean subset, which is the ablation that says
  whether it mattered. Friends is also acted: *conversational* rather than
  *posed-peak*, which is the specific failure being corrected, but not
  spontaneous affect.
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
| `MINDSCAPE_PROSODY_FUSION` | `false` | let the prosody heuristic vote in fusion. Off on measurement — it carries no signal but is confidently neutral, and entropy weighting rewards that. `true` restores the old behaviour and runs the ablation |
| `MINDSCAPE_W_TEXT` / `_FACE` / `_VOICE` | `0.8` / `0.15` / `0.05` | fusion weights, swept on MELD and MOSI. The old 0.5/0.3/0.2 scored below serving text alone |
| `MINDSCAPE_INDEPENDENCE` | `0` | cross-modal reinforcement; `0` = plain averaging. Was 0.5; the sweep on both corpora is monotonic and every step above 0 costs accuracy |
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
    mic.ts             audio level (60 Hz) + two-recorder chunking, and the
                       per-session loudness a saved entry carries
    session.ts         orchestrates live / streamed / commit
  ../scripts/
    fetch-models.mjs   copies face-api weights out of node_modules
    eval-faceapi.mjs   scores that stock head in Node, mirroring face.ts exactly
  world/
    scene.ts           camera, render loop, bloom
    atmosphere.ts      sky shader, lights, motes — the fast layer
    orbs.ts            memory galaxy + core ring — the permanent layer
    placement.ts       one definition of where an orb goes
    keywords.ts        floating words
  state/
    db.ts              IndexedDB, export/import
    vocals.ts          per-diary loudness ranking -> orb size and glow
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
  dump_channels.py     per-utterance vectors from every channel (MELD)
  prepare_mosi.py      CMU-MOSI face crops, reusing the MELD extractor
  dump_channels_mosi.py  the same dump for CMU-MOSI
  eval_fusion.py       subsets, independence sweep, weight sweep
  eval_stacking.py     learned late fusion, the control for eval_fusion
  prepare_meld_video.py  MELD's raw video -> speaker face crops (+ whole frames)
  eval_face_baseline.py  score face-api's stock head: F1, ECE, neutral drift
  train_face.py        fine-tune the face head, calibrate, export ONNX
  eval_diary.py        the domain-shift measurement
  diary_probe.py       the 47-sentence probe set
  smoke_test.py        dependency-light checks for the fusion arithmetic
```

## Adding 3D assets

Orbs are `IcosahedronGeometry` in `world/orbs.ts`. Swap in a glTF and placement,
colour, two-tone blending and scaling all still work — entry positions are stored
on the entry, so tuning the layout never relocates existing memories.
