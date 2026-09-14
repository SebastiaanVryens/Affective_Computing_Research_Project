/**
 * The MELD label space.
 *
 * Everything in this app — face, voice, text — votes in these seven classes.
 * The order here is the canonical order and must match the backend's
 * backend/app/emotions.py EMOTIONS list, because probability vectors are sent
 * over the wire as bare arrays.
 */
export const EMOTIONS = [
  'neutral',
  'joy',
  'sadness',
  'anger',
  'fear',
  'disgust',
  'surprise',
] as const;

export type Emotion = (typeof EMOTIONS)[number];

/**
 * The emotions arranged so that neighbours are *related*, for anything that
 * renders them as a continuous gradient.
 *
 * `EMOTIONS` above is MELD's canonical order — it's the wire format and must not
 * change. But that order is alphabetical-ish accident, so rendering a gradient
 * along it puts sadness next to joy and fear next to disgust, and every
 * transition reads as an arbitrary jump.
 *
 * This ordering instead walks the valence/arousal circumplex, which is roughly
 * how the affect literature (Russell; Plutchik) arranges these:
 *
 *   neutral → sadness → disgust → anger → fear → surprise → joy
 *   (calm)    (low arousal, negative) ⟶ (high arousal) ⟶ (positive)
 *
 * Several adjacencies are Plutchik's own blend pairs — fear beside surprise is
 * awe, surprise beside joy is delight — so where two neighbours do appear
 * together, the gradient between them names a feeling that actually exists.
 */
export const GRADIENT_ORDER: readonly Emotion[] = [
  'neutral',
  'sadness',
  'disgust',
  'anger',
  'fear',
  'surprise',
  'joy',
] as const;

/** A probability distribution over the seven emotions. Always sums to ~1. */
export type EmotionVector = Record<Emotion, number>;

export interface EmotionPalette {
  /** Core hue, used for orb bodies and history dots. */
  base: string;
  /** Brighter variant for emissive highlights and glow. */
  glow: string;
  /** Human-facing name, Inside Out flavoured. */
  label: string;
}

/**
 * Colour identity per emotion. Five of these are the Inside Out cast; surprise
 * and neutral aren't characters in the film, so they get an amber and a soft
 * slate that sit next to the others without competing.
 */
export const PALETTE: Record<Emotion, EmotionPalette> = {
  neutral: { base: '#8d93a8', glow: '#b9c0d4', label: 'Calm' },
  joy: { base: '#ffd23f', glow: '#fff2a8', label: 'Joy' },
  sadness: { base: '#3f7fd2', glow: '#8dc0ff', label: 'Sadness' },
  anger: { base: '#e2453c', glow: '#ff8b76', label: 'Anger' },
  fear: { base: '#9b5de5', glow: '#d3a9ff', label: 'Fear' },
  disgust: { base: '#6bbf4a', glow: '#b6f09a', label: 'Disgust' },
  surprise: { base: '#ff8c42', glow: '#ffc48c', label: 'Surprise' },
};

/** All-zero vector, useful as an accumulator seed. */
export function zeroVector(): EmotionVector {
  return Object.fromEntries(EMOTIONS.map((e) => [e, 0])) as EmotionVector;
}

/** Uniform distribution — what we report when we know nothing at all. */
export function uniformVector(): EmotionVector {
  const p = 1 / EMOTIONS.length;
  return Object.fromEntries(EMOTIONS.map((e) => [e, p])) as EmotionVector;
}

/** Converts the backend's bare array form into a keyed vector. */
export function fromArray(probs: number[]): EmotionVector {
  const v = zeroVector();
  EMOTIONS.forEach((e, i) => {
    v[e] = probs[i] ?? 0;
  });
  return normalize(v);
}

export function toArray(v: EmotionVector): number[] {
  return EMOTIONS.map((e) => v[e]);
}

/** Rescales so the vector sums to 1. Returns uniform if everything is zero. */
export function normalize(v: EmotionVector): EmotionVector {
  const total = EMOTIONS.reduce((sum, e) => sum + Math.max(0, v[e]), 0);
  if (total <= 1e-9) return uniformVector();
  const out = zeroVector();
  for (const e of EMOTIONS) out[e] = Math.max(0, v[e]) / total;
  return out;
}

/** Weighted sum of several vectors, renormalised. Weights need not sum to 1. */
export function blend(parts: Array<{ vector: EmotionVector; weight: number }>): EmotionVector {
  const out = zeroVector();
  for (const { vector, weight } of parts) {
    if (weight <= 0) continue;
    for (const e of EMOTIONS) out[e] += vector[e] * weight;
  }
  return normalize(out);
}

/** Exponential moving average — how the live face signal gets its smoothing. */
export function ema(previous: EmotionVector, next: EmotionVector, alpha: number): EmotionVector {
  const out = zeroVector();
  for (const e of EMOTIONS) out[e] = previous[e] * (1 - alpha) + next[e] * alpha;
  return normalize(out);
}

export function dominant(v: EmotionVector): Emotion {
  let best: Emotion = 'neutral';
  for (const e of EMOTIONS) if (v[e] > v[best]) best = e;
  return best;
}

/**
 * How much emotion is present at all, in [0, 1] — the non-neutral mass.
 *
 * Deliberately says nothing about which emotions or how many. Feeling two things
 * at once is not the same as feeling nothing clearly, and separating "how much"
 * from "how many" is what keeps the two apart. Mirrors backend `charge()`.
 */
export function charge(v: EmotionVector): number {
  return 1 - v.neutral;
}

/**
 * Effective number of emotions being felt, in [1, 6].
 *
 * Perplexity of the distribution with neutral removed — reads literally as a
 * count: 1.0 is a single emotion, 2.0 is two held about equally, 6.0 is no
 * structure at all. Mirrors backend `complexity()`.
 */
export function complexity(v: EmotionVector): number {
  const rest = EMOTIONS.filter((e) => e !== 'neutral').map((e) => Math.max(0, v[e]));
  const total = rest.reduce((sum, p) => sum + p, 0);
  if (total <= 1e-9) return 1;

  let entropy = 0;
  for (const value of rest) {
    const p = value / total;
    if (p > 1e-9) entropy -= p * Math.log(p);
  }
  return Math.exp(entropy);
}

/**
 * Concentrates a distribution onto the few emotions that actually dominate.
 *
 * Rendering all seven at once produces a rainbow — technically faithful to the
 * vector, but it reads as a children's book rather than as a mood, and the two
 * emotions that matter get the same billing as five that don't.
 *
 * Two stages, because neither alone is sufficient:
 *
 * **Power sharpening** raises each weight to `power` and renormalises. It is
 * continuous and order-preserving, so nothing pops and no emotion overtakes
 * another as a side effect. It cleanly kills a long tail — a faint 15% third
 * emotion drops to ~4%. But it preserves *ratios*, so it barely helps when four
 * emotions sit close together: 30/25/20/15 stays five visible colours at any
 * exponent, which is exactly the messy case worth fixing.
 *
 * **Nucleus masking** handles that. Sorted descending, an emotion fades out once
 * the emotions ahead of it already account for `nucleus` of the mass — the same
 * idea as nucleus sampling in language models. It adapts to the shape of the
 * distribution rather than to a fixed count: one clear feeling stays one colour,
 * an even split keeps both, and a diffuse spread is cut off after the few that
 * carry most of it.
 *
 * The fade band keeps that continuous. Two emotions swapping rank have nearly
 * equal weights and therefore nearly equal masks, so a swap is invisible rather
 * than a flicker — which is what a hard top-K cutoff would give you.
 */
export function focus(
  v: EmotionVector,
  power = 2,
  nucleus = 0.8,
  fade = 0.18
): EmotionVector {
  const sharpened = zeroVector();
  for (const e of EMOTIONS) sharpened[e] = Math.pow(Math.max(0, v[e]), power);
  const normalized = normalize(sharpened);

  const ranked = EMOTIONS.map((e) => ({ e, w: normalized[e] })).sort(
    (a, b) => b.w - a.w
  );

  const out = zeroVector();
  let cumulativeBefore = 0;
  for (const { e, w } of ranked) {
    // How far past the nucleus the *preceding* emotions already carry us. Using
    // the running total before this emotion — not including it — is what lets
    // the emotion that crosses the boundary survive rather than being cut in
    // half by its own mass.
    const overflow = (cumulativeBefore - nucleus) / fade;
    const keep = 1 - Math.min(1, Math.max(0, overflow));
    out[e] = w * keep;
    cumulativeBefore += w;
  }
  return normalize(out);
}

/** The emotions actually contributing, largest first. Mirrors backend. */
export function components(
  v: EmotionVector,
  floor = 0.15
): Array<{ emotion: Emotion; share: number }> {
  const rest = EMOTIONS.filter((e) => e !== 'neutral');
  const total = rest.reduce((sum, e) => sum + Math.max(0, v[e]), 0);
  if (total <= 1e-9) return [{ emotion: 'neutral', share: 1 }];

  const found = rest
    .map((emotion) => ({ emotion, share: v[emotion] / total }))
    .filter((c) => c.share >= floor)
    .sort((a, b) => b.share - a.share);

  return found.length > 0 ? found : [{ emotion: 'neutral', share: 1 }];
}

/**
 * How far the moment is from emotional flatness, in [0, 1].
 *
 * Defined as 1 - (normalised Shannon entropy), so a confident single-emotion
 * reading scores near 1 and a "could be anything" reading scores near 0. This
 * is what decides orb brightness and whether an entry is offered as a core
 * memory — it measures emotional *clarity*, not arousal.
 */
export function intensity(v: EmotionVector): number {
  let entropy = 0;
  for (const e of EMOTIONS) {
    const p = v[e];
    if (p > 1e-9) entropy -= p * Math.log(p);
  }
  const maxEntropy = Math.log(EMOTIONS.length);
  return Math.min(1, Math.max(0, 1 - entropy / maxEntropy));
}

/** Same measure, but ignoring neutral — "how much did anything actually happen". */
export function affectiveCharge(v: EmotionVector): number {
  return 1 - v.neutral;
}

/** Parses "#rrggbb" into the 0..1 float triple three.js wants. */
export function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.replace('#', ''), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

/**
 * The single colour that represents a whole mixture — each emotion's hue
 * weighted by its probability. Used for the world's ambient tint and for the
 * fallback colour of a mixed day.
 */
export function mixedColor(v: EmotionVector): string {
  let r = 0;
  let g = 0;
  let b = 0;
  for (const e of EMOTIONS) {
    const [er, eg, eb] = hexToRgb(PALETTE[e].base);
    r += er * v[e];
    g += eg * v[e];
    b += eb * v[e];
  }
  const to255 = (x: number) => Math.round(Math.min(255, Math.max(0, x * 255)));
  return `#${[to255(r), to255(g), to255(b)]
    .map((c) => c.toString(16).padStart(2, '0'))
    .join('')}`;
}

/**
 * A CSS conic-gradient that slices a circle by emotion proportion — the
 * history view's dots. Emotions under `floor` are folded away so a dot with a
 * 0.4% sliver of disgust doesn't render a hairline artifact.
 */
export function conicGradient(v: EmotionVector, floor = 0.04): string {
  // Largest slice first, so every dot starts its dominant emotion at twelve
  // o'clock. Canonical order would put whichever emotion happens to come first
  // in the array at the top regardless of size, which makes a row of dots much
  // harder to read at a glance.
  const slices = EMOTIONS.map((e) => ({ e, p: v[e] }))
    .filter((s) => s.p >= floor)
    .sort((a, b) => b.p - a.p);
  if (slices.length === 0) return PALETTE.neutral.base;
  const total = slices.reduce((sum, s) => sum + s.p, 0);

  const stops: string[] = [];
  let cursor = 0;
  for (const { e, p } of slices) {
    const end = cursor + (p / total) * 360;
    stops.push(`${PALETTE[e].base} ${cursor.toFixed(2)}deg ${end.toFixed(2)}deg`);
    cursor = end;
  }
  return `conic-gradient(from -90deg, ${stops.join(', ')})`;
}
