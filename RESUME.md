# Session notes — face & voice channel measurement

Updated 2026-09-17. Untracked scratch file; delete once the work is committed.

The measurement work is **done**. Results are written into README.md; this file
holds the operational detail that doesn't belong there — what's on disk, what was
run, and what's still open.

---

## What changed, and what to do with it

**Ship:**

- `backend/models/exp-e-tweets` — text, MELD test wF1 **0.6291** vs the previous
  0.6068, selected on dev. To adopt it, copy over `backend/models/meld-text`
  (the backend loads that path on next start).
- The prosody fusion fix (`app/audio_model.py` + `settings.prosody_in_fusion`).
  Already in place, default off, reversible with `MINDSCAPE_PROSODY_FUSION=true`.
- The SER load guard in `audio_model.py._load()` — refuses a checkpoint whose
  classification head was randomly initialised. This is the bug fix of the
  session: the shipped voice model had a random head and nothing errored.
- The face-capture error handling and the HMR camera-leak fix in the frontend.

**Do not ship:**

- `backend/models/meld-face` — the trained face head. It beats face-api on
  weighted-F1 (0.3462 vs 0.3150) but **loses macro-F1** (0.1466 vs 0.1651), and
  this project's own argument is that macro is what a diary needs. Nothing loads
  it; no action required to leave it out.
- The face temperature (T = 7.12). Real, but fitted on MELD — refit in-domain
  before applying it at runtime.

---

## Data on disk (regenerable, gitignored)

| Path | What |
|---|---|
| `../MELD_data/train_splits/` etc. | 15,908 mp4 (10.2 GB) — the only copy |
| `data/meld_faces/{train,dev,test}.csv` | 65,378 / 7,324 / 16,912 speaker crops |
| `data/meld_faces/{dev,test}_frames.csv` | whole frames for the face-api baseline |
| `data/meld_faces/faceapi_{dev,test}.part*.json` | baseline shards (merged automatically) |
| `data/meld_faces/faceapi_baseline.json` | scored baseline |
| `data/meld_faces/detector_sweep.json` | inputSize × scoreThreshold sweep |
| `data/meld_voice/voice_prosody.json` | prosody tier scored on dev+test |
| `data/corpora/` | **per-class capped** (fear 112/540, disgust 260/496) |
| `data/corpora_proportional/` | the old proportional CSVs, for reproducing run C |
| `data/corpora_perclass/` | backup of the per-class ones |

All tarballs were deleted after verifying the 15,908 extracted mp4 files, so the
extracted video is now the **only** copy. Re-fetch from
`https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz`
(~50 min download, ~2 min unpack) if it is ever lost.

The broken `ehcalabres` checkpoint was also purged from the HuggingFace cache; it
re-downloads on demand if you want to reproduce the random-head finding. Training
intermediates under `backend/models/checkpoints*` were deleted — note that
`train_text.py` writes them to a shared directory with `save_total_limit=1`, so
each run evicts the previous run's checkpoint.

**`data/corpora/` is shared mutable state.** Swapping in the proportional CSVs to
reproduce run C must be undone afterwards. An interrupted run already left it in
the wrong state once. Adding a `--corpora-dir` flag to `train_text.py` would
remove the hazard.

---

## Running things again

```powershell
# face-api baseline (5 shards in parallel, ~15 min)
cd frontend
node scripts/eval-faceapi.mjs --manifest ..\data\meld_faces\test_frames.csv `
    --root ..\data\meld_faces --out ..\data\meld_faces\faceapi_test.part0.json `
    --shard 0 --shards 5          # ... and shards 1..4
cd ..\backend
.venv\Scripts\python.exe -m training.eval_face_baseline

# train the face head (~35 min)
.venv\Scripts\python.exe -m training.train_face --device cuda --epochs 8 `
    --workers 0 --no-class-weights --lr 1e-4
```

### Two operational traps, both hit during this session

**`--workers 0` is not a typo.** Windows *spawns* dataloader workers, so each one
imports torch and commits ~4 GB. `--workers 3` reserved ~14 GB on a 15.3 GB
machine and died with `paging file is too small`. Worse, each crash orphaned its
children: 13 stranded processes accumulated **42.4 GB of committed memory** with
near-zero working sets, starving every subsequent attempt. Check
`Get-Process python` before blaming anything else. Counter-intuitively
`--workers 0` also ran **faster** — 5.1 it/s vs 1.45.

**Don't run training alongside anything else that imports torch.** That is what
started the spiral.

---

## The voice channel — measured, and not shipped

The default SER checkpoint **never worked**: saved under transformers 4.8.2, its
head does not match the current layout, so every head tensor was randomly
initialised at load. `audio_model.py` now rejects such a checkpoint.

A replacement loads cleanly and emits exactly MELD's seven classes (no `calm`
collapse):

```
MINDSCAPE_VOICE_MODEL=firdhokk/speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53
MINDSCAPE_DEVICE=cuda
python -m training.eval_voice --source "..\..\MELD_data" --splits dev,test
```

MELD test: raw **0.0981** wF1 (it predicts neutral on 0.8% of clips against a 48%
true rate), rising to **0.2954** wF1 / **0.1542** macro after per-class prior
correction — against the prior baseline's 0.0927 macro. On non-neutral clips it is
27.2% correct vs 16.7% chance, where prosody was 17.3%. Real signal, wrong
operating point.

**Not enabled**, and two things must happen first:

1. **Joint vector scaling.** The temperature and the bias are currently fitted
   independently on the same logits; applying both double-corrects and collapses
   the model to always-neutral (the `prior + T` row reproduces the prior baseline
   exactly — it is an artifact, not a result). Fit scale and bias together.
2. **Refit on diary audio.** The bias encodes MELD's 48% neutral rate. A diary's
   neutral rate is unknown and probably different.

Until then the voice channel reports `available: false` and fusion runs on text +
face, which is honest but means two working modalities, not three.

## Plan — what to do next, in order

### 1. Measure whether fusion actually works  ← the biggest gap

The project's central claim has never been tested. The README argues for
certainty-weighted late fusion, cross-modal reinforcement and time-aligned
fusion; all of it is argued, none measured end to end. Every *individual* channel
is now measured, and MELD test carries text, audio and video for the same 2,610
utterances — so every input already exists.

Questions it answers:

- Does fusing beat the best single channel? Text alone is 0.629; fusion has to
  clear that to justify itself.
- Does entropy weighting beat plain averaging? `MINDSCAPE_INDEPENDENCE=0` is the
  ablation.
- What should the weights be? 0.5 / 0.3 / 0.2 were set before anything was
  measured, and face is now known to be roughly half as accurate as text.

How: extend each eval script to dump per-utterance vectors keyed by
`dialogue_id`/`utterance_id`, then add `training/eval_fusion.py` to load all
three and sweep weights and the independence parameter.

Effort: ~half a day. Be prepared for fusion not to beat text alone — that is
still a result, and better found here than by a reviewer.

### 2. Build an in-domain probe set

Every number so far is MELD. `eval_diary.py` covers text with 47 sentences and
has no face or voice equivalent. Record **20–30 short diary entries with
self-labelled emotion**. That unlocks four things currently guessed:

- refit the face temperature in-domain (T = 7.12 is a MELD artifact)
- refit the voice prior bias (it encodes MELD's 48% neutral rate)
- confirm face detection really is ~100% here rather than MELD's 48.5%
- test whether the trained face head transfers at all

Effort: ~1 hour recording + ~2 hours scripting. Highest leverage per hour on this
list, because it converts MELD findings into product decisions.

### 3. Finish the voice channel

- **Joint vector scaling** — fit scale and bias together; the current sequential
  fit double-corrects into always-neutral.
- **Then enable it**, once ECE is acceptable. At 0.417 today, enabling it repeats
  the prosody mistake with a better model.

Effort: ~1 hour + a 15-min rerun. Depends on #2 for the in-domain refit.

### 4. Hygiene

- **Adopt run E** — copy `models/exp-e-tweets` over `models/meld-text`. The
  +0.022 is sitting unused.
- **Seed variance** — 3 seeds on run E, so the gain can be quoted with a range.
- **`--corpora-dir` flag** for `train_text.py`, removing the shared-mutable-state
  hazard that already corrupted `data/corpora/` once.
- **Drop the broken model from `bundling/bundle_models.py`** — the installer
  ships 1.2 GB of weights for a checkpoint that cannot load.

Effort: ~1 hour total.

### 5. Model improvements — deliberately last

B→C says the bottleneck is **domain, not capacity**, so a bigger backbone is
likely wasted effort. Two ideas that fit the evidence:

- **Source ISEAR.** Already named in this README as the best fit — *"describe a
  situation in which you felt X"* is literally the diary task — and not loadable
  from the Hub. Worth sourcing directly.
- **Reconsider the taxonomy.** Fear and disgust are unreachable for *every* model
  tested, on 45 and 63 test clips. Either document them as a limitation, or ask
  whether seven classes is right for a face channel that can only separate
  neutral / joy / anger / surprise.

---

## Still open

- **Seed variance is unmeasured everywhere.** Run E's +0.022 is single-seed;
  disgust swung 0.251 → 0.312 between two face runs with identical data. Two or
  three seeds on run E would say whether the text gain is real.

- **Run C has no dev number** — the re-run was interrupted. Needed only if you
  want the dev column complete for the report.

- **Face detector precision is unmeasured.** The sweep shows `scoreThreshold`
  0.35 → 0.20 recovering ~16 points of detection for no cost, but it measures
  recall only. On MELD there is nearly always a real face; on a cluttered webcam
  view a lower threshold may start firing on lamps. Don't change the shipped
  default without a false-positive check — and note the whole issue is likely moot
  for this app, since faces at a laptop span 25–40% of frame width and sit well
  above the detector's cliff.

- **The face head overfits.** Eval loss rose from epoch 2 even at lr 1e-4;
  `load_best_model_at_end` picked epoch 6. More augmentation or fewer epochs might
  do better, though the ceiling evidence suggests limited headroom.

---

## Not committed

Nothing has been committed. Modified: `.gitignore`, `README.md`,
`backend/requirements.txt`, `backend/app/config.py`, `backend/app/audio_model.py`,
`backend/training/prepare_corpora.py`, `backend/training/train_text.py`,
`frontend/package.json`, `frontend/src/main.ts`, `frontend/src/capture/face.ts`.
New: `backend/training/{prepare_meld_video,train_face,eval_face_baseline,eval_voice}.py`,
`frontend/scripts/{eval-faceapi,sweep-detector}.mjs`.
