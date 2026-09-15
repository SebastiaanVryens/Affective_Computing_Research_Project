/**
 * What your landscape is made of.
 *
 * The orbs record *how* you felt. This records what you kept talking *about* —
 * and it is the only input the mindscape has for deciding whether you get a
 * forest, a coastline, or a room full of books. Somebody who spends three months
 * talking about their thesis should walk out onto an island of paper; somebody
 * who talks about the sea should get a sea.
 *
 * Deliberately a hand-written lexicon rather than anything learned. Two reasons.
 *
 * The first is that the alternative would be a second classifier, trained on
 * nothing, guessing at a category space nobody has defined — and when it was
 * wrong, there would be no way to see *why* your landscape grew a mountain. A
 * word list is auditable: the mountain is there because you said "climbing"
 * eleven times, and you can read the line that says so.
 *
 * The second is scope honesty. This is decoration, not measurement. Nothing in
 * the clinical report reads from this file, and nothing here should ever be
 * presented as a finding about a person — it exists so the world looks like it
 * belongs to you, and that is all it is licensed to claim.
 *
 * Terms are matched by loose stem, so a list can stay singular and present-tense
 * and still catch "books", "booking" is *not* matched (see `stems`), "climbed",
 * "swimming". Words are allowed to belong to several motifs — "swim" feeds both
 * the sea and sport, and whichever the person talks about more wins on volume
 * rather than by a tie-break nobody could predict.
 */

import {
  type Emotion,
  type EmotionVector,
  blend,
  dominant,
  uniformVector,
} from '../emotions';
import { isThemeworthy } from '../state/stopwords';
import type { DiaryEntry } from '../state/db';

/** Where on the island a motif wants to stand. */
export type Ground = 'any' | 'low' | 'high' | 'shore';

export interface Motif {
  id: string;
  /** Shown in the world legend, and in the toast when a motif first appears. */
  label: string;
  /** Singular, present-tense stems. See `stems()` for what inflections match. */
  terms: string[];
  prefer: Ground;
  /** Minimum gap between two props of this motif, in world units. */
  spacing: number;
  /** Hard cap on props, so one obsession doesn't tile the whole island. */
  maxProps: number;

  /**
   * What this motif becomes indoors.
   *
   * The room biome is chosen for a diary that is mostly indoor things, but the
   * motifs still fire — so without this the woods put pine trees on the
   * floorboards and the working world stands an office block next to the sofa.
   * Both looked merely odd as abstract shapes and look absurd as real models.
   *
   * Three answers, and which one a motif gets is a judgement about what it
   * *means* rather than about what it looks like:
   *
   *   undefined — place it as usual. It is something that belongs in a room.
   *   a model key — place that instead. Woods indoors are a houseplant, which
   *                 is exactly what somebody who loves the woods keeps on a
   *                 windowsill.
   *   false     — do not place it at all. You are already home; the city is out
   *                 of the window (see view.ts), and neither wants a scale model
   *                 of itself on the rug.
   */
  indoor?: string | false;
}

/**
 * The motif set.
 *
 * Eleven, chosen to cover what spoken diary entries are actually about without
 * the landscape turning into a catalogue. The ordering here is the tie-break
 * when two motifs score identically, so the more visually distinctive ones sit
 * earlier — a tie that resolves to "forest" makes a better island than one that
 * resolves to "city".
 */
export const MOTIFS: Motif[] = [
  {
    id: 'study',
    label: 'Books and papers',
    prefer: 'any',
    spacing: 1.6,
    maxProps: 7,
    terms: `book paper research thesis study read exam library article lecture
      seminar professor supervisor university essay deadline assignment note
      write writing dissertation journal citation chapter revision coursework
      phd degree course homework draft abstract experiment data analysis`.split(/\s+/),
  },
  {
    id: 'forest',
    label: 'Woods',
    prefer: 'high',
    // Wide enough for a broad canopy. The tree models are about as wide as they
    // are tall, so a spacing set for a narrow conifer has them growing through
    // each other.
    spacing: 3.2,
    maxProps: 9,
    indoor: 'room:plant',
    terms: `forest tree wood hike trail nature park camping pine leaf autumn
      moss outdoors wander wilderness mushroom branch spruce birch fern
      countryside meadow field`.split(/\s+/),
  },
  {
    id: 'sea',
    label: 'Water',
    prefer: 'shore',
    spacing: 2.0,
    maxProps: 6,
    indoor: false,
    terms: `beach sea ocean swim sand wave coast shore surf island sail boat
      harbour harbor tide seaside lake river dive fishing pier shell seagull
      saltwater kayak canoe`.split(/\s+/),
  },
  {
    id: 'mountains',
    label: 'High ground',
    prefer: 'high',
    spacing: 4.2,
    maxProps: 3,
    indoor: false,
    terms: `mountain hill climb peak summit snow ski cliff boulder altitude
      valley ridge glacier slope avalanche crag`.split(/\s+/),
  },
  {
    id: 'home',
    label: 'Home',
    prefer: 'any',
    spacing: 2.8,
    maxProps: 4,
    indoor: false,
    terms: `home house flat apartment kitchen bedroom sofa bed cook cleaning
      laundry rent neighbour neighbor tea coffee blanket cosy cozy dinner
      breakfast lunch recipe bake bread window door`.split(/\s+/),
  },
  {
    id: 'city',
    label: 'The working world',
    prefer: 'low',
    spacing: 2.6,
    maxProps: 3,
    indoor: false,
    terms: `city town street traffic train tram bus metro subway commute office
      work meeting job colleague boss shop market cafe bar restaurant downtown
      email project client deadline shift overtime`.split(/\s+/),
  },
  {
    id: 'people',
    label: 'People',
    prefer: 'any',
    spacing: 2.0,
    maxProps: 7,
    terms: `friend family mother mum mom father dad sister brother partner
      girlfriend boyfriend wife husband parent grandma grandpa party talk
      conversation visit together hug love lonely alone miss cousin wedding
      birthday goodbye argument apology`.split(/\s+/),
  },
  {
    id: 'music',
    label: 'Music',
    prefer: 'any',
    spacing: 2.5,
    maxProps: 3,
    terms: `music song guitar piano sing band concert gig album playlist listen
      drum violin choir dance rehearsal melody chord vinyl headphone`.split(/\s+/),
  },
  {
    id: 'sport',
    label: 'Moving',
    prefer: 'low',
    spacing: 3.2,
    maxProps: 2,
    terms: `run gym workout training football basketball tennis bike cycling
      match team practice yoga fitness race exercise swim stretch marathon
      pitch court`.split(/\s+/),
  },
  {
    id: 'garden',
    label: 'Growing things',
    prefer: 'low',
    spacing: 1.1,
    maxProps: 14,
    indoor: 'room:plant',
    terms: `garden flower plant grow seed bloom blossom spring herb tomato
      vegetable gardening greenhouse watering soil rose petal harvest
      allotment`.split(/\s+/),
  },
  {
    id: 'pets',
    label: 'Animals',
    prefer: 'any',
    spacing: 2.6,
    maxProps: 3,
    terms: `dog cat puppy kitten pet vet purr bark horse rabbit hamster paw
      fur kennel leash`.split(/\s+/),
  },
];

/**
 * Term → motif ids.
 *
 * Built once at module load. A term appearing in several motifs lands in several
 * ids, and every one of them scores the mention — a word is allowed to mean two
 * things at once, exactly as the emotion model is.
 */
const TERM_INDEX = ((): Map<string, string[]> => {
  const index = new Map<string, string[]>();
  for (const motif of MOTIFS) {
    for (const term of motif.terms) {
      const existing = index.get(term);
      if (existing) existing.push(motif.id);
      else index.set(term, [motif.id]);
    }
  }
  return index;
})();

/**
 * Candidate stems for a spoken word, cheapest-first.
 *
 * Not a real stemmer, and deliberately not: a proper one (Porter, Snowball) is a
 * dependency and a pile of rules for a gain that does not exist here, because
 * the term lists are hand-written and can simply include any inflection this
 * misses. What matters is that it never *over*-stems into a different word —
 * "booking" must not become "book", or every mention of a dentist appointment
 * grows a library. So -ing and -ed only strip when the remainder is still long
 * enough to be the word it claims to be, and the term lists carry the awkward
 * cases (`read`, `write`) explicitly.
 */
function stems(word: string): string[] {
  const out = [word];
  const n = word.length;

  if (n > 4 && word.endsWith('ies')) out.push(`${word.slice(0, -3)}y`);
  if (n > 4 && word.endsWith('ves')) out.push(`${word.slice(0, -3)}f`);
  if (n > 4 && word.endsWith('es')) out.push(word.slice(0, -2));
  if (n > 3 && word.endsWith('s') && !word.endsWith('ss')) out.push(word.slice(0, -1));
  if (n > 5 && word.endsWith('ing')) {
    out.push(word.slice(0, -3), `${word.slice(0, -3)}e`);
    // "swimming" → "swim": undo the doubled final consonant.
    if (word[n - 4] === word[n - 5]) out.push(word.slice(0, -4));
  }
  if (n > 4 && word.endsWith('ed')) {
    out.push(word.slice(0, -2), word.slice(0, -1));
    if (word[n - 3] === word[n - 4]) out.push(word.slice(0, -3));
  }
  return out;
}

/** Which motifs, if any, a single spoken word belongs to. */
function motifsFor(word: string): string[] | undefined {
  for (const stem of stems(word)) {
    const hit = TERM_INDEX.get(stem);
    if (hit) return hit;
  }
  return undefined;
}

export interface MotifPresence {
  motif: Motif;
  /** Distinct entries that mentioned it. The honest "is this a theme" measure. */
  entryCount: number;
  /** Total weighted mentions. Decides how densely the motif fills its ground. */
  mentions: number;
  /** Share of the total across all present motifs, in [0, 1]. */
  share: number;
  /** How the person felt in the entries that mentioned it — tints the props. */
  vector: EmotionVector;
  emotion: Emotion;
  /** How many props to place. */
  propCount: number;
}

/**
 * Mentions of one motif that a single entry can contribute.
 *
 * Without a cap, one twenty-minute entry about a holiday produces more sea than
 * three months of everything else, and the landscape stops being a picture of
 * the diary and becomes a picture of its longest day.
 */
const MAX_MENTIONS_PER_ENTRY = 5;

/**
 * Weight given to a word the keyword extractor already singled out.
 *
 * The extractor has read the whole entry and decided the word was salient, which
 * is strictly more evidence than the word merely occurring in the transcript. It
 * is not worth much more than that, though — the extractors disagree with each
 * other (see state/stopwords.ts), so this stays a thumb on the scale rather than
 * a separate channel.
 */
const KEYWORD_WEIGHT = 2.5;

/**
 * How many motifs can stand on the ground at once.
 *
 * Five, because this is a room rather than a map. Every motif takes an arc of a
 * circle a dozen units across, and past five those arcs are too narrow to read
 * as separate places — you stop seeing "the woods, and over there the books" and
 * start seeing an evenly-mixed pile of objects, which says nothing.
 *
 * Exported, and applied by the caller rather than here, because this is a limit
 * on *placement* and nothing else. It used to be applied inside `detectMotifs`,
 * which silently imposed it on every other question too — and the one that
 * suffered was the view out of the room's window, which asks "what is the
 * strongest outdoor theme". In an indoor-dominant diary the outdoor theme is
 * precisely the one ranked sixth, so a person who wrote about the sea every
 * weekend got a city skyline. Truncating at the point of use keeps the limit
 * where its reasoning applies.
 */
export const MAX_MOTIFS = 5;

/**
 * Read the landscape's contents out of the diary.
 *
 * Returns *every* motif that clears the recurrence bar, ranked by weight,
 * strongest first — not a top-N. Callers that can only show a few take the first
 * few themselves; see MAX_MOTIFS. A motif mentioned exactly once, in one entry,
 * is dropped for the same reason the report drops a keyword that appears once:
 * it is a word, not a theme, and building a forest out of it would tell the
 * person something about themselves that isn't true.
 */
export function detectMotifs(entries: DiaryEntry[]): MotifPresence[] {
  const tally = new Map<
    string,
    { mentions: number; entries: Set<string>; parts: Array<{ vector: EmotionVector; weight: number }> }
  >();

  const record = (id: string, entry: DiaryEntry, amount: number): void => {
    let row = tally.get(id);
    if (!row) {
      row = { mentions: 0, entries: new Set(), parts: [] };
      tally.set(id, row);
    }
    row.mentions += amount;
    row.entries.add(entry.id);
  };

  for (const entry of entries) {
    // Per-entry sub-tally so MAX_MENTIONS_PER_ENTRY can be applied before the
    // entry is folded into the lifetime figure.
    const local = new Map<string, number>();

    const bump = (id: string, amount: number): void => {
      local.set(id, (local.get(id) ?? 0) + amount);
    };

    for (const word of tokenize(entry.transcript)) {
      const ids = motifsFor(word);
      if (ids) for (const id of ids) bump(id, 1);
    }

    for (const keyword of entry.keywords ?? []) {
      if (!isThemeworthy(keyword.text)) continue;
      for (const word of tokenize(keyword.text)) {
        const ids = motifsFor(word);
        if (ids) for (const id of ids) bump(id, KEYWORD_WEIGHT);
      }
    }

    for (const [id, amount] of local) {
      record(id, entry, Math.min(MAX_MENTIONS_PER_ENTRY, amount));
      tally.get(id)!.parts.push({
        vector: entry.vector,
        // Same certainty floor the rest of the app uses, so an ambiguous entry
        // still colours its motif rather than vanishing from the average.
        weight: 0.4 + 0.6 * entry.certainty,
      });
    }
  }

  const present: MotifPresence[] = [];
  for (const motif of MOTIFS) {
    const row = tally.get(motif.id);
    if (!row) continue;
    if (row.entries.size < 2 && row.mentions < 2) continue;

    const vector = row.parts.length ? blend(row.parts) : uniformVector();
    present.push({
      motif,
      entryCount: row.entries.size,
      mentions: row.mentions,
      share: 0, // filled in below, once the total is known
      vector,
      emotion: dominant(vector),
      propCount: propCountFor(motif, row.entries.size, row.mentions),
    });
  }

  present.sort((a, b) => weightOf(b) - weightOf(a));

  const total = present.reduce((sum, p) => sum + weightOf(p), 0);
  if (total > 0) for (const p of present) p.share = weightOf(p) / total;

  return present;
}

/**
 * How much of the island a motif has earned.
 *
 * Distinct entries count for far more than raw mentions, because they are the
 * thing that separates a recurring part of someone's life from a single evening
 * they happened to describe in detail.
 */
function weightOf(p: MotifPresence): number {
  return p.entryCount * 3 + p.mentions;
}

/**
 * How many props a motif places.
 *
 * Logarithmic on purpose. The interesting transition is 0 → 1 → a few: the first
 * time the woods appear is the moment the world tells you something. Going from
 * forty mentions to eighty is not a moment, and if it were linear it would crowd
 * every other motif off the ground by the second month.
 *
 * The coefficients are small because the ground is small. Three trees in a room
 * is a wood; twenty is a texture you cannot see past.
 */
function propCountFor(motif: Motif, entryCount: number, mentions: number): number {
  const raw = 1 + 1.1 * Math.log2(1 + entryCount) + 0.3 * Math.log2(1 + mentions);
  return Math.max(1, Math.min(motif.maxProps, Math.round(raw)));
}

/** Words, lowercased, punctuation and numerals dropped. */
function tokenize(text: string): string[] {
  if (!text) return [];
  return text
    .toLowerCase()
    .split(/[^a-z']+/)
    .filter((word) => word.length >= 3);
}

/**
 * One-line description of what the landscape is currently showing.
 *
 * Used for the toast after a save and for the empty state, so the connection
 * between "I talked about the sea" and "there is now a sea" is stated once in
 * words rather than left for the person to notice or not.
 */
export function describeLandscape(motifs: MotifPresence[]): string {
  if (motifs.length === 0) {
    return 'Bare ground so far — keep talking and it will grow into something.';
  }
  const named = motifs.slice(0, 3).map((m) => m.motif.label.toLowerCase());
  const list =
    named.length === 1
      ? named[0]
      : `${named.slice(0, -1).join(', ')} and ${named[named.length - 1]}`;
  return `Your mind has grown ${list}.`;
}
