/**
 * The live mood bus.
 *
 * This is the seam between "capture" and "world". Capture sources push into it
 * at whatever rate they naturally run (face ~8 Hz, audio ~60 Hz, backend text
 * every few seconds); the 3D scene reads `current()` once per render frame and
 * never waits on anything.
 *
 * Nothing in here does I/O or async work. It's a mutable snapshot of "how does
 * this moment feel", designed to be read 60 times a second for free.
 */

import {
  EMOTIONS,
  GRADIENT_ORDER,
  focus,
  type Emotion,
  type EmotionVector,
  blend,
  dominant,
  normalize,
  ema,
  intensity,
  mixedColor,
  uniformVector,
} from '../emotions';

export interface MoodSnapshot {
  /** Fused, smoothed distribution driving the world's colour. */
  vector: EmotionVector;
  dominant: Emotion;
  /** Emotional clarity in [0,1] — how committed the reading is. */
  clarity: number;
  /** Vocal energy in [0,1], updated per audio frame. Drives motion, not colour. */
  arousal: number;
  /** True while a face is actually in frame. */
  faceVisible: boolean;
  /** True while the mic is picking up speech rather than room tone. */
  speaking: boolean;
  /** Hex colour for the whole mixture — the world's ambient tint. */
  color: string;
  /**
   * Per-emotion weights for the sky, in GRADIENT_ORDER and sharpened so only the
   * dominant two or three claim meaningful area.
   *
   * Still all seven slots rather than a top-N list: the sky treats these as
   * fixed positions, so an emotion rising or fading only changes its own share.
   * A top-N list would reshuffle which slot an emotion occupies and the sky
   * would jump every time the ranking changed. Sharpening achieves the same
   * "show fewer colours" result continuously instead.
   */
  bands: Float32Array;
}

type Listener = (snapshot: MoodSnapshot) => void;

/**
 * How fast each channel's smoothing tracks reality.
 *
 * Face is the fastest because it's the one the user can consciously test
 * ("if I smile, does the world go yellow?") and a sluggish response there reads
 * as broken. Text is slowest — it arrives already aggregated over a sentence,
 * and snapping the whole world on each new sentence is visually violent.
 */
const SMOOTHING = {
  face: 0.28,
  voice: 0.15,
  text: 0.09,
} as const;

/**
 * Weights for the live blend. These deliberately differ from the backend's
 * fusion weights (config.py): this is the *felt* mood driving visuals, where
 * face responsiveness matters more than classification accuracy. The backend's
 * weighting is what gets recorded in the diary.
 */
const LIVE_WEIGHTS = {
  face: 0.45,
  voice: 0.2,
  text: 0.35,
} as const;

/** Face confidence decays once we stop seeing a face, instead of freezing. */
const FACE_STALE_MS = 1200;

/**
 * How hard to concentrate the sky onto its dominant emotions.
 *
 * At 1 the raw distribution renders — all seven at once, a rainbow. 2 is enough
 * to fade out a faint third emotion without flattening a genuine blend; the
 * nucleus below does the work of limiting how many colours appear at all, so
 * this doesn't need to be aggressive. Higher would start erasing real secondary
 * emotions, which is the thing we're trying to show.
 */
const SKY_SHARPENING = 2;

/**
 * Share of the mood the sky's colours must account for.
 *
 * Emotions beyond this fade out, so the number of colours adapts to the shape
 * of the feeling: one clear emotion stays one colour, an even split keeps both,
 * a diffuse spread is cut after the two or three carrying most of it.
 */
const SKY_NUCLEUS = 0.8;

/**
 * What the world looks like before anything has been detected.
 *
 * Deliberately neutral-dominant rather than uniform. A uniform vector is the
 * honest "no information" state numerically, but it renders as all seven
 * emotion bands at once — a rainbow sky meaning "we know nothing", which is
 * both ugly and misleading. Backend fusion already resolves total ignorance to
 * neutral for the same reason (see fusion.py); this keeps the two consistent.
 *
 * This is only the *fallback* resting state. Once the diary has entries in it,
 * main.ts replaces it with the lifetime mood, so opening the app shows the
 * world you have actually accumulated rather than a blank grey one.
 */
function defaultResting(): EmotionVector {
  const out = { ...uniformVector() };
  for (const e of EMOTIONS) out[e] = e === 'neutral' ? 1 : 0.05;
  return normalize(out);
}

class MoodBus {
  private face: EmotionVector = uniformVector();
  private voice: EmotionVector = uniformVector();
  private text: EmotionVector = uniformVector();

  private faceSeenAt = 0;
  private textSeenAt = 0;
  private hasText = false;
  private hasHeardVoice = false;

  private fused: EmotionVector = defaultResting();
  /**
   * Where the mood settles when no channel is reporting — the lifetime mood
   * once there are entries, plain neutral before that.
   */
  private resting: EmotionVector = defaultResting();
  private arousalValue = 0;
  private speakingNow = false;

  private listeners = new Set<Listener>();
  private cachedColor = '#8d93a8';
  private colorDirty = true;
  // Reused across frames: current() runs every render tick, and allocating a
  // fresh array there would hand the GC 60 short-lived objects a second.
  private bandBuffer = new Float32Array(EMOTIONS.length);

  // -- inputs ----------------------------------------------------------

  pushFace(vector: EmotionVector): void {
    this.face = ema(this.face, vector, SMOOTHING.face);
    this.faceSeenAt = performance.now();
    this.recompute();
  }

  /** Called when the detector ran but found no face — decays rather than holds. */
  faceLost(): void {
    this.face = ema(this.face, uniformVector(), 0.06);
    this.recompute();
  }

  pushVoice(vector: EmotionVector): void {
    this.hasHeardVoice = true;
    this.voice = ema(this.voice, vector, SMOOTHING.voice);
    this.recompute();
  }

  pushText(vector: EmotionVector): void {
    this.text = ema(this.text, vector, this.hasText ? SMOOTHING.text : 1);
    this.hasText = true;
    this.textSeenAt = performance.now();
    this.recompute();
  }

  /**
   * Per-audio-frame energy. Called ~60x/sec, so it deliberately skips the
   * fusion recompute and only touches two numbers.
   */
  pushArousal(level: number, speaking: boolean): void {
    this.arousalValue = level;
    this.speakingNow = speaking;
  }

  // -- fusion ----------------------------------------------------------

  private recompute(): void {
    const now = performance.now();
    const faceAge = now - this.faceSeenAt;
    // Linear fade-out of the face channel's authority over FACE_STALE_MS.
    const faceWeight =
      this.faceSeenAt === 0
        ? 0
        : LIVE_WEIGHTS.face * Math.max(0, 1 - faceAge / FACE_STALE_MS);

    // Text stays authoritative for a while after the last utterance — what you
    // said a few seconds ago is still what the moment is about.
    const textWeight = this.hasText
      ? LIVE_WEIGHTS.text * Math.max(0.25, 1 - (now - this.textSeenAt) / 45_000)
      : 0;

    // With nothing contributing, settle back to the resting mood rather than
    // to uniform — which is the accumulated world once the diary has entries.
    this.fused =
      faceWeight + textWeight <= 1e-6 && !this.hasHeardVoice
        ? this.resting
        : blend([
            { vector: this.face, weight: faceWeight },
            { vector: this.voice, weight: LIVE_WEIGHTS.voice },
            { vector: this.text, weight: textWeight },
          ]);
    this.colorDirty = true;
    this.emit();
  }

  // -- outputs ---------------------------------------------------------

  current(): MoodSnapshot {
    if (this.colorDirty) {
      this.cachedColor = mixedColor(this.fused);
      this.colorDirty = false;
    }
    const focused = focus(this.fused, SKY_SHARPENING, SKY_NUCLEUS);
    for (let i = 0; i < GRADIENT_ORDER.length; i++) {
      this.bandBuffer[i] = focused[GRADIENT_ORDER[i]];
    }

    return {
      vector: this.fused,
      bands: this.bandBuffer,
      dominant: dominant(this.fused),
      clarity: intensity(this.fused),
      arousal: this.arousalValue,
      faceVisible: performance.now() - this.faceSeenAt < FACE_STALE_MS,
      speaking: this.speakingNow,
      color: this.cachedColor,
    };
  }

  /**
   * Set the mood the world falls back to when nothing is being detected.
   *
   * Called with the diary's lifetime distribution, so an idle world shows who
   * you have been rather than a neutral grey. Applied immediately when no
   * channel is live, so opening the app is already coloured before the camera
   * has even produced its first reading.
   */
  setResting(vector: EmotionVector): void {
    this.resting = normalize(vector);
    this.recompute();
  }

  /** Raw per-channel state, for the diagnostics panel. */
  channels(): Record<'face' | 'voice' | 'text', EmotionVector> {
    return { face: this.face, voice: this.voice, text: this.text };
  }

  /** Forget the conversation-level signal at the end of a session. */
  resetUtterance(): void {
    this.text = uniformVector();
    this.voice = uniformVector();
    this.hasText = false;
    this.hasHeardVoice = false;
    this.speakingNow = false;
    this.arousalValue = 0;
    this.recompute();
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(): void {
    if (this.listeners.size === 0) return;
    const snapshot = this.current();
    for (const listener of this.listeners) listener(snapshot);
  }
}

export const mood = new MoodBus();
