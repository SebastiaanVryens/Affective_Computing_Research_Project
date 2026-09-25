/**
 * How loud an entry was, *for you*.
 *
 * The voice channel was supposed to carry what the words cannot: "I'm fine"
 * said flatly and "I'm fine" said brightly are the same transcript. Measured on
 * two corpora, the emotion-from-tone model does not deliver that — it scores at
 * the class prior and drags fusion down. Loudness does deliver it, and has the
 * advantage of being *measured* rather than inferred: an RMS meter cannot be
 * confidently wrong the way a classifier can.
 *
 * So the orbs encode vocal intensity directly and honestly, instead of a model's
 * guess about what that intensity meant.
 *
 * The whole problem is the normalisation. Absolute microphone levels are
 * incomparable — a headset and a laptop array disagree by more than a whisper
 * differs from a shout, and moving closer to the desk changes it again. An
 * absolute scale would mostly encode which machine you sat at.
 *
 * Rank within your own history sidesteps all of it. Your loudest entry is the
 * loudest whatever your hardware does, and the spread always fills the visual
 * range regardless of whether you are a quiet talker or a loud one.
 */

import type { DiaryEntry } from './db';

/** Below this there isn't enough speech to characterise how it sounded. */
const MIN_SPEAKING_SECONDS = 1.5;

/** Ranks need a population; below this, every entry reads as average. */
const MIN_POPULATION = 4;

export interface VocalRank {
  /** Loudness percentile within this diary, 0 (quietest) to 1 (loudest). */
  loudness: number;
  /** Spectral-brightness percentile — a tense, raised voice ranks high. */
  brightness: number;
}

function percentileRanks(values: number[]): number[] {
  const order = values
    .map((value, index) => ({ value, index }))
    .sort((a, b) => a.value - b.value);

  const ranks = new Array<number>(values.length).fill(0.5);
  // Ties share the mean of the positions they span, so a diary recorded at one
  // steady volume does not get an arbitrary ordering imposed on it.
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j + 1 < order.length && order[j + 1].value === order[i].value) j++;
    const shared = (i + j) / 2 / Math.max(1, order.length - 1);
    for (let k = i; k <= j; k++) ranks[order[k].index] = shared;
    i = j + 1;
  }
  return ranks;
}

/**
 * Rank every entry's voice against the rest of the diary.
 *
 * Entries with no usable vocal measurement are absent from the result rather
 * than defaulted, so callers can leave them at their existing appearance. A
 * diary written entirely before this feature existed therefore looks exactly as
 * it did, instead of every orb silently becoming "average".
 */
export function vocalRanks(entries: DiaryEntry[]): Map<string, VocalRank> {
  const usable = entries.filter(
    (entry) =>
      entry.vocals !== undefined &&
      entry.vocals.speakingSeconds >= MIN_SPEAKING_SECONDS
  );

  const out = new Map<string, VocalRank>();
  if (usable.length < MIN_POPULATION) return out;

  const loudness = percentileRanks(usable.map((e) => e.vocals!.mean));
  const brightness = percentileRanks(usable.map((e) => e.vocals!.brightness));
  usable.forEach((entry, i) => {
    out.set(entry.id, { loudness: loudness[i], brightness: brightness[i] });
  });
  return out;
}

/**
 * Size multiplier for an orb, centred on 1.
 *
 * Deliberately gentle. Loudness is one fact about a memory competing with
 * emotion, certainty and duration for the same visual channel, and a range wide
 * enough to be unmistakable would drown them.
 */
export function loudnessScale(rank: VocalRank | undefined): number {
  if (!rank) return 1;
  return 0.85 + 0.3 * rank.loudness;
}

/**
 * Emissive multiplier, also centred on 1.
 *
 * Driven by loudness rather than spectral brightness, so size and glow say the
 * same thing twice. That redundancy is the point: one cue is ambiguous against
 * the orb's own emotion colour, two agreeing cues read immediately. Spectral
 * brightness is ranked and stored alongside for a future second axis.
 */
export function loudnessGlow(rank: VocalRank | undefined): number {
  if (!rank) return 1;
  return 0.78 + 0.44 * rank.loudness;
}
