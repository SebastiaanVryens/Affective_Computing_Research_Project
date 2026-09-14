/**
 * What *kind* of place this is.
 *
 * The motifs in ./motifs.ts decide what stands on the ground. This decides what
 * the ground is — and it is a separate question, because a sea is not a prop.
 * Somebody who talks about the coast should not get a lawn with three pebbles on
 * it; they should get a coast, with the water running out to the horizon and the
 * land ending in a shore. Somebody who talks about the woods should get woods:
 * ground that carries on past the trees, a treeline closing the distance, and no
 * sea at all, because there is no reason for one.
 *
 * So the biome is chosen first, from what the person keeps returning to, and
 * everything the landscape modules do is parameterised by it: whether there is
 * water and how much, how the land ends at its edge, what closes the horizon,
 * how much relief, and what the ground is made of.
 *
 * Two things deliberately do *not* decide the biome:
 *
 * Feelings. Emotion colours this world everywhere — the ground's hue, the sky,
 * the relief, the tint on every prop — but it does not pick the *place*. A rule
 * that turned a bad month into a barren landscape would be telling someone
 * something about themselves, in a language they can't argue with, from a face
 * classifier. Topic is a thing the person said; mood is a thing a model decided.
 * Only the first gets to choose what world they walk into.
 *
 * A single entry. Selection runs on motif *share*, which already requires a
 * theme to recur across entries before it counts at all. One holiday does not
 * flood the world.
 */

import type { MotifPresence } from './motifs';

export type BiomeId = 'coast' | 'forest' | 'alpine' | 'meadow' | 'room' | 'bare';

/** How the island's ground ends at its rim. */
export type Edge = 'shore' | 'continuous';

/** What closes the distance behind it. */
export type Backdrop = 'peaks' | 'treeline' | 'rolling' | 'none';

export interface Biome {
  id: BiomeId;
  label: string;
  /** One line, shown when you arrive at the mind view. */
  blurb: string;

  edge: Edge;
  backdrop: Backdrop;

  /**
   * Sea level, or null for dry ground.
   *
   * Relative to the same datum `elevation` uses, so a level near 0 floods the
   * hollows and a level well below it leaves the ground dry.
   */
  waterLevel: number | null;
  /** How far the water reaches. A pond is not an ocean. */
  waterRadius: number;

  /** Multiplies the relief the diary's emotional charge asks for. */
  reliefScale: number;

  /**
   * How far the ground leans, in world units, from the middle to either edge.
   *
   * The thing that stops every place being a dome on a disc. A tilted ground
   * reads as a hillside, and — where there is a sea — puts the waterline across
   * the view as a shoreline instead of around it as a ring. Zero is a floor.
   */
  tilt: number;

  /**
   * Multiplies how fast the camera circles, on top of what the diary asks for.
   *
   * A room should mostly sit still; open country can turn. See `restlessness`
   * in mindscape.ts for the other half of this.
   */
  orbitScale: number;
  /** Height above which snow lies. Infinity for none. */
  snowAbove: number;

  /** The ground itself, before the diary's colour is mixed into it. */
  ground: {
    base: string;
    dry: string;
    rock: string;
    sand: string;
  };
  /** How much of the ground's colour is emotion rather than earth. */
  tint: number;
}

/**
 * The layouts.
 *
 * Five outdoor places and one indoor one. The set is meant to cover the shapes a
 * spoken diary actually takes rather than to be a catalogue of terrain types —
 * there is no desert and no swamp because nobody's week is one.
 */
export const BIOMES: Record<BiomeId, Biome> = {
  coast: {
    id: 'coast',
    label: 'a coast',
    blurb: 'Your mind is a coast.',
    edge: 'shore',
    backdrop: 'peaks',
    waterLevel: -0.6,
    waterRadius: 200,
    reliefScale: 1,
    tilt: 2.4,
    orbitScale: 1,
    snowAbove: Infinity,
    ground: { base: '#5c6b45', dry: '#8a8155', rock: '#6e717c', sand: '#cbb68c' },
    tint: 0.4,
  },

  forest: {
    id: 'forest',
    label: 'woodland',
    blurb: 'Your mind has grown over.',
    edge: 'continuous',
    backdrop: 'treeline',
    // A pool in the low ground, no more. Woods have water in them; they are not
    // in the water.
    waterLevel: -1.1,
    waterRadius: 20,
    reliefScale: 0.85,
    tilt: 1.0,
    orbitScale: 0.85,
    snowAbove: Infinity,
    ground: { base: '#3f5936', dry: '#5d6b3c', rock: '#5c5f68', sand: '#8d8460' },
    tint: 0.34,
  },

  alpine: {
    id: 'alpine',
    label: 'high ground',
    blurb: 'Your mind is up in the rock and snow.',
    edge: 'continuous',
    backdrop: 'peaks',
    // A tarn, sitting in whatever hollow the relief happens to give it.
    waterLevel: -1.5,
    waterRadius: 18,
    // The one biome that reshapes the ground rather than just recolouring it —
    // but only somewhat. The camera sits at a fixed height above a fixed datum,
    // so relief past roughly 1.2 puts hilltops above the lens and you spend the
    // view inside a slope instead of looking at one.
    reliefScale: 1.15,
    tilt: 2.1,
    orbitScale: 0.75,
    // Low enough that the tops actually catch it. Snow is what sells this biome
    // — far more than the rock colour does — so the line has to sit inside the
    // range of heights the relief above can actually reach.
    snowAbove: 1.9,
    ground: { base: '#5e6168', dry: '#797c83', rock: '#868a95', sand: '#9a9689' },
    // The least emotionally tinted of the biomes. Rock is grey, and a hillside
    // that took the diary's hue as strongly as a meadow does stops reading as
    // stone — which is the one thing this biome has to communicate.
    tint: 0.18,
  },

  meadow: {
    id: 'meadow',
    label: 'open ground',
    blurb: 'Your mind is open ground.',
    edge: 'continuous',
    backdrop: 'rolling',
    waterLevel: null,
    waterRadius: 0,
    reliefScale: 0.5,
    tilt: 0.8,
    orbitScale: 1,
    snowAbove: Infinity,
    ground: { base: '#6a7a41', dry: '#9a9152', rock: '#767a86', sand: '#b8ab84' },
    tint: 0.46,
  },

  room: {
    id: 'room',
    label: 'a room',
    blurb: 'Your mind is a room you spend a lot of time in.',
    edge: 'continuous',
    backdrop: 'none',
    waterLevel: null,
    waterRadius: 0,
    // Flat, because it is a floor. Every other landform term is switched off for
    // this biome in `elevation`.
    reliefScale: 0,
    tilt: 0,
    orbitScale: 0.1,
    snowAbove: Infinity,
    ground: { base: '#7a5c3e', dry: '#8d6d49', rock: '#6a5a4c', sand: '#a08662' },
    tint: 0.22,
  },

  bare: {
    id: 'bare',
    label: 'bare ground',
    blurb: 'Bare ground so far — keep talking and it will grow into something.',
    edge: 'continuous',
    backdrop: 'rolling',
    waterLevel: null,
    waterRadius: 0,
    reliefScale: 0.7,
    tilt: 0.6,
    orbitScale: 0.9,
    snowAbove: Infinity,
    ground: { base: '#5f6552', dry: '#7c7864', rock: '#6e717c', sand: '#a9a289' },
    tint: 0.38,
  },
};

/**
 * Which motifs argue for which place.
 *
 * A motif can vote for more than one — woods and animals both suggest being
 * outdoors, books and work both suggest being indoors — with weights rather than
 * a hard assignment, because most diaries are a mixture and the right answer is
 * whichever reading is strongest overall, not whichever single word won.
 */
const VOTES: Record<BiomeId, Partial<Record<string, number>>> = {
  coast: { sea: 1 },
  forest: { forest: 1, pets: 0.3 },
  alpine: { mountains: 1, sport: 0.25 },
  meadow: { garden: 1, sport: 0.45, pets: 0.4, people: 0.25 },
  // Indoors is the *bundle* — no single motif means "a room", but somebody whose
  // diary is books, work and home is plainly not standing on a hillside.
  room: { study: 0.8, home: 1, city: 0.6, music: 0.6 },
  bare: {},
};

/**
 * How strong the leading reading has to be before it gets to pick the world.
 *
 * Below this, `bare` wins. A landscape asserted from almost nothing is worse
 * than an honest empty one — the whole promise of this view is that it is a
 * picture of what you have actually been saying.
 */
const MIN_CONVICTION = 0.26;

export interface BiomeChoice {
  biome: Biome;
  /** Winning score in [0, 1]. Low means the world is barely committed. */
  conviction: number;
  /** The runner-up, for the variations the blurb mentions. */
  secondary: BiomeId | null;
}

export function chooseBiome(motifs: MotifPresence[]): BiomeChoice {
  const share = new Map(motifs.map((m) => [m.motif.id, m.share]));

  const scored = (Object.keys(VOTES) as BiomeId[])
    .filter((id) => id !== 'bare')
    .map((id) => {
      let score = 0;
      for (const [motifId, weight] of Object.entries(VOTES[id])) {
        score += (share.get(motifId) ?? 0) * (weight ?? 0);
      }
      return { id, score };
    })
    .sort((a, b) => b.score - a.score);

  const best = scored[0];
  if (!best || best.score < MIN_CONVICTION) {
    return { biome: BIOMES.bare, conviction: best?.score ?? 0, secondary: null };
  }

  return {
    biome: BIOMES[best.id],
    conviction: Math.min(1, best.score),
    secondary: scored[1] && scored[1].score > MIN_CONVICTION * 0.6 ? scored[1].id : null,
  };
}

/**
 * The caption: what this place is, and what is in it.
 *
 * Returned in two parts because they are read differently. The headline names
 * the place and is the answer to "why does my world look like this"; the detail
 * lists the themes that produced it, which is the evidence for the headline.
 *
 * It is worth stating in words at all — rather than trusting the person to infer
 * it — because the inference is the whole point of the view. A coastline nobody
 * connects to having talked about the coast is just a coastline.
 */
export function describePlace(
  choice: BiomeChoice,
  motifs: MotifPresence[]
): { headline: string; detail: string } {
  if (motifs.length === 0) {
    return { headline: BIOMES.bare.blurb, detail: '' };
  }

  const named = motifs.slice(0, 3).map((m) => m.motif.label.toLowerCase());
  const list =
    named.length === 1
      ? named[0]
      : `${named.slice(0, -1).join(', ')} and ${named[named.length - 1]}`;

  return {
    headline: choice.biome.blurb,
    detail: choice.biome.id === 'bare' ? `So far: ${list}.` : `Built from ${list}.`,
  };
}
