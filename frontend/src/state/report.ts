/**
 * Turns the diary into the summary a clinician would actually read.
 *
 * Scope note, because it shapes every decision in this file: this is
 * *self-tracked data*, not a clinical instrument. Nothing here is validated
 * against a diagnostic scale, and none of it should be presented as though it
 * were. So the report deliberately does not compute anything that resembles a
 * screening score, and does not flag "risk" — a false negative from a face
 * model has no business reassuring anyone, and a false positive has no business
 * alarming them.
 *
 * What it does instead is present the things a mood diary is genuinely good
 * for, which clinicians already use paper diaries to get:
 *
 *   - adherence: how often, how consistently, where the gaps are
 *   - trajectory: is this moving, and in which direction
 *   - variability: day-to-day lability, which is clinically meaningful
 *   - persistence: consecutive low days matter more than an average
 *   - diurnal pattern: time-of-day variation
 *   - the person's own words, verbatim
 *   - provenance: how much of this was measured from what
 *
 * Every derived number carries the sample it came from, so a reader can
 * discount a "trend" computed from four days.
 */

import {
  EMOTIONS,
  type Emotion,
  type EmotionVector,
  blend,
  charge,
  dominant,
  zeroVector,
} from '../emotions';
import type { DiaryEntry } from './db';

/**
 * Valence grouping used throughout the report.
 *
 * Note surprise is counted as neither. MELD's own sentiment mapping calls it
 * positive, and the backend follows that for consistency with the dataset — but
 * surprise is genuinely ambivalent (a shock and a delight are both surprise),
 * and treating it as positive would quietly inflate every positive figure in a
 * document someone might make decisions from.
 */
export const NEGATIVE: Emotion[] = ['sadness', 'anger', 'fear', 'disgust'];
export const POSITIVE: Emotion[] = ['joy'];
export const AMBIVALENT: Emotion[] = ['neutral', 'surprise'];

/** Minimum distinct days before a trend line is worth reporting at all. */
const MIN_DAYS_FOR_TREND = 7;
/** Minimum days before day-to-day variability means anything. */
const MIN_DAYS_FOR_VARIABILITY = 5;

export interface DaySummary {
  day: string;
  date: Date;
  vector: EmotionVector;
  dominant: Emotion;
  /** Negative mass minus positive mass, in [-1, 1]. Negative = worse. */
  valence: number;
  charge: number;
  entryCount: number;
  totalSeconds: number;
}

export interface ChannelReliability {
  name: string;
  /** Share of entries where this channel produced a usable reading. */
  availability: number;
  /** Mean certainty when it was available. */
  meanCertainty: number;
}

export interface NotableEntry {
  date: Date;
  emotion: Emotion;
  /** The person's own words. Verbatim — never paraphrased. */
  quote: string;
  score: number;
  isCoreMemory: boolean;
  note?: string;
}

export interface TimeOfDayBand {
  label: string;
  entryCount: number;
  meanValence: number;
}

export interface ClinicalReport {
  name: string;
  generatedAt: Date;
  periodStart: Date;
  periodEnd: Date;
  periodLabel: string;

  // -- adherence -------------------------------------------------------
  entryCount: number;
  daysCovered: number;
  daysInPeriod: number;
  coverage: number;
  longestGapDays: number;
  totalMinutes: number;
  medianEntryMinutes: number;

  // -- profile ---------------------------------------------------------
  overall: EmotionVector;
  negativeShare: number;
  positiveShare: number;
  days: DaySummary[];

  // -- patterns --------------------------------------------------------
  /** Least-squares slope of daily valence, per week. Null if too few days. */
  valenceTrendPerWeek: number | null;
  /** Mean absolute day-to-day change in valence. Null if too few days. */
  variability: number | null;
  /** Longest run of consecutive recorded days with negative dominant affect. */
  longestNegativeRun: number;
  timeOfDay: TimeOfDayBand[];

  // -- content ---------------------------------------------------------
  notable: NotableEntry[];
  coreMemories: NotableEntry[];
  themes: Array<{ text: string; emotion: Emotion; count: number }>;

  // -- provenance ------------------------------------------------------
  channels: ChannelReliability[];
  /** Share of entries that carried a transcript at all. */
  transcribedShare: number;
}

export type Period = 'all' | '30d' | '90d';

export function buildReport(
  allEntries: DiaryEntry[],
  name: string,
  period: Period
): ClinicalReport {
  const now = new Date();
  const cutoff = periodCutoff(period, now);
  const entries = allEntries
    .filter((e) => !cutoff || new Date(e.createdAt) >= cutoff)
    .sort((a, b) => a.createdAt.localeCompare(b.createdAt));

  const periodStart = entries.length
    ? new Date(entries[0].createdAt)
    : cutoff ?? now;
  const periodEnd = entries.length
    ? new Date(entries[entries.length - 1].createdAt)
    : now;

  const days = summariseDays(entries);
  const overall = entries.length
    ? blend(entries.map((e) => ({ vector: e.vector, weight: 0.4 + 0.6 * e.certainty })))
    : zeroVector();

  const daysInPeriod = Math.max(
    1,
    Math.round((startOfDay(periodEnd).getTime() - startOfDay(periodStart).getTime()) / 86_400_000) + 1
  );

  const durations = entries.map((e) => e.durationSeconds).sort((a, b) => a - b);

  return {
    name: name.trim() || 'Not provided',
    generatedAt: now,
    periodStart,
    periodEnd,
    periodLabel: describePeriod(period),

    entryCount: entries.length,
    daysCovered: days.length,
    daysInPeriod,
    coverage: days.length / daysInPeriod,
    longestGapDays: longestGap(days),
    totalMinutes: Math.round(entries.reduce((s, e) => s + e.durationSeconds, 0) / 60),
    medianEntryMinutes: durations.length
      ? Math.round((durations[Math.floor(durations.length / 2)] / 60) * 10) / 10
      : 0,

    overall,
    negativeShare: shareOf(overall, NEGATIVE),
    positiveShare: shareOf(overall, POSITIVE),
    days,

    valenceTrendPerWeek: valenceTrend(days),
    variability: valenceVariability(days),
    longestNegativeRun: longestNegativeRun(days),
    timeOfDay: timeOfDayBands(entries),

    notable: notableEntries(entries),
    coreMemories: entries.filter((e) => e.isCoreMemory).map(toNotable),
    themes: recurringThemes(entries),

    channels: channelReliability(entries),
    transcribedShare: entries.length
      ? entries.filter((e) => e.transcript.trim().length > 0).length / entries.length
      : 0,
  };
}

// ---------------------------------------------------------------------------
// Days
// ---------------------------------------------------------------------------

function summariseDays(entries: DiaryEntry[]): DaySummary[] {
  const byDay = new Map<string, DiaryEntry[]>();
  for (const entry of entries) {
    const list = byDay.get(entry.day);
    if (list) list.push(entry);
    else byDay.set(entry.day, [entry]);
  }

  return [...byDay.entries()]
    .map(([day, group]) => {
      const vector = blend(
        group.map((e) => ({ vector: e.vector, weight: 0.4 + 0.6 * e.certainty }))
      );
      return {
        day,
        date: startOfDay(new Date(group[0].createdAt)),
        vector,
        dominant: dominant(vector),
        valence: shareOf(vector, POSITIVE) - shareOf(vector, NEGATIVE),
        charge: charge(vector),
        entryCount: group.length,
        totalSeconds: group.reduce((s, e) => s + e.durationSeconds, 0),
      };
    })
    .sort((a, b) => a.date.getTime() - b.date.getTime());
}

function shareOf(vector: EmotionVector, group: Emotion[]): number {
  return group.reduce((sum, e) => sum + vector[e], 0);
}

function longestGap(days: DaySummary[]): number {
  let longest = 0;
  for (let i = 1; i < days.length; i++) {
    const gap =
      Math.round((days[i].date.getTime() - days[i - 1].date.getTime()) / 86_400_000) - 1;
    longest = Math.max(longest, gap);
  }
  return longest;
}

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

/**
 * Least-squares slope of daily valence, expressed per week.
 *
 * Returned as null below MIN_DAYS_FOR_TREND rather than as a small number:
 * a slope fitted through four points is noise, and printing it next to the word
 * "trend" in a document a clinician reads would be actively misleading.
 *
 * Regressed against real elapsed days, not against the index of recorded days,
 * so a fortnight's silence doesn't compress into the same x-distance as a day.
 */
function valenceTrend(days: DaySummary[]): number | null {
  if (days.length < MIN_DAYS_FOR_TREND) return null;

  const origin = days[0].date.getTime();
  const points = days.map((d) => ({
    x: (d.date.getTime() - origin) / 86_400_000,
    y: d.valence,
  }));

  const n = points.length;
  const meanX = points.reduce((s, p) => s + p.x, 0) / n;
  const meanY = points.reduce((s, p) => s + p.y, 0) / n;

  let numerator = 0;
  let denominator = 0;
  for (const p of points) {
    numerator += (p.x - meanX) * (p.y - meanY);
    denominator += (p.x - meanX) ** 2;
  }
  if (denominator < 1e-9) return null;
  return (numerator / denominator) * 7;
}

/**
 * Mean absolute day-to-day change in valence — a lability proxy.
 *
 * Only consecutive *calendar* days count. Comparing Monday to the following
 * Friday and calling the difference "day-to-day variability" would overstate
 * instability in exactly the people who record irregularly.
 */
function valenceVariability(days: DaySummary[]): number | null {
  const deltas: number[] = [];
  for (let i = 1; i < days.length; i++) {
    const gapDays = Math.round(
      (days[i].date.getTime() - days[i - 1].date.getTime()) / 86_400_000
    );
    if (gapDays === 1) deltas.push(Math.abs(days[i].valence - days[i - 1].valence));
  }
  if (deltas.length < MIN_DAYS_FOR_VARIABILITY) return null;
  return deltas.reduce((s, d) => s + d, 0) / deltas.length;
}

/** Longest run of consecutive recorded days whose dominant affect was negative. */
function longestNegativeRun(days: DaySummary[]): number {
  let longest = 0;
  let current = 0;
  for (let i = 0; i < days.length; i++) {
    const isNegative = NEGATIVE.includes(days[i].dominant);
    const consecutive =
      i === 0 ||
      Math.round((days[i].date.getTime() - days[i - 1].date.getTime()) / 86_400_000) === 1;
    current = isNegative ? (consecutive ? current + 1 : 1) : 0;
    longest = Math.max(longest, current);
  }
  return longest;
}

const BANDS: Array<{ label: string; from: number; to: number }> = [
  { label: 'Morning (05–12)', from: 5, to: 12 },
  { label: 'Afternoon (12–17)', from: 12, to: 17 },
  { label: 'Evening (17–22)', from: 17, to: 22 },
  { label: 'Night (22–05)', from: 22, to: 5 },
];

function timeOfDayBands(entries: DiaryEntry[]): TimeOfDayBand[] {
  return BANDS.map(({ label, from, to }) => {
    const inBand = entries.filter((e) => {
      const hour = new Date(e.createdAt).getHours();
      return from < to ? hour >= from && hour < to : hour >= from || hour < to;
    });
    const meanValence = inBand.length
      ? inBand.reduce(
          (s, e) => s + shareOf(e.vector, POSITIVE) - shareOf(e.vector, NEGATIVE),
          0
        ) / inBand.length
      : 0;
    return { label, entryCount: inBand.length, meanValence };
  });
}

// ---------------------------------------------------------------------------
// Content
// ---------------------------------------------------------------------------

function toNotable(entry: DiaryEntry): NotableEntry {
  return {
    date: new Date(entry.createdAt),
    emotion: entry.peak?.emotion ?? entry.dominant,
    quote: entry.peak?.text ?? entry.transcript,
    score: entry.peak?.score ?? entry.certainty,
    isCoreMemory: entry.isCoreMemory,
    note: entry.note,
  };
}

/**
 * The entries most worth a clinician's attention.
 *
 * Ranked by emotional charge, then spread across the period rather than taken
 * as a flat top-N: five quotes from one bad week is a misleading portrait of
 * three months. At most two per calendar week.
 */
function notableEntries(entries: DiaryEntry[], limit = 8): NotableEntry[] {
  const withQuotes = entries.filter((e) => (e.peak?.text ?? e.transcript).trim().length > 20);
  const ranked = [...withQuotes].sort(
    (a, b) => (b.peak?.score ?? b.certainty) - (a.peak?.score ?? a.certainty)
  );

  const perWeek = new Map<string, number>();
  const chosen: DiaryEntry[] = [];
  for (const entry of ranked) {
    if (chosen.length >= limit) break;
    const week = weekKey(new Date(entry.createdAt));
    const used = perWeek.get(week) ?? 0;
    if (used >= 2) continue;
    perWeek.set(week, used + 1);
    chosen.push(entry);
  }

  return chosen
    .sort((a, b) => a.createdAt.localeCompare(b.createdAt))
    .map(toNotable);
}

/** Keywords appearing across multiple entries — what the person keeps returning to. */
function recurringThemes(
  entries: DiaryEntry[],
  limit = 14
): Array<{ text: string; emotion: Emotion; count: number }> {
  const merged = new Map<
    string,
    { text: string; score: number; entries: Set<string>; vector: EmotionVector }
  >();

  for (const entry of entries) {
    for (const keyword of entry.keywords ?? []) {
      const key = keyword.text.toLowerCase();
      let record = merged.get(key);
      if (!record) {
        record = { text: keyword.text, score: 0, entries: new Set(), vector: zeroVector() };
        merged.set(key, record);
      }
      record.score += keyword.score;
      record.entries.add(entry.id);
      EMOTIONS.forEach((e, i) => {
        record!.vector[e] += keyword.emotionVector?.[i] ?? 0;
      });
    }
  }

  return [...merged.values()]
    // Appearing once is a word, not a theme.
    .filter((r) => r.entries.size >= 2)
    .sort((a, b) => b.entries.size - a.entries.size || b.score - a.score)
    .slice(0, limit)
    .map((r) => ({ text: r.text, emotion: dominant(r.vector), count: r.entries.size }));
}

// ---------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------

/**
 * How much of this report rests on what.
 *
 * Arguably the most important section for a clinician. A summary built from
 * entries where the camera was off is a summary of *words*, and a reader should
 * be able to see that rather than assume three channels were always present.
 */
function channelReliability(entries: DiaryEntry[]): ChannelReliability[] {
  const names = ['text', 'face', 'voice'];
  return names.map((name) => {
    const readings = entries
      .map((e) => (e.modalities ?? []).find((m) => m.name === name))
      .filter((m): m is NonNullable<typeof m> => Boolean(m));
    const available = readings.filter((m) => m.available);
    return {
      name,
      availability: entries.length ? available.length / entries.length : 0,
      meanCertainty: available.length
        ? available.reduce((s, m) => s + m.certainty, 0) / available.length
        : 0,
    };
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function periodCutoff(period: Period, now: Date): Date | null {
  if (period === 'all') return null;
  const days = period === '30d' ? 30 : 90;
  const cutoff = new Date(now);
  cutoff.setDate(cutoff.getDate() - days);
  return startOfDay(cutoff);
}

function describePeriod(period: Period): string {
  if (period === '30d') return 'Last 30 days';
  if (period === '90d') return 'Last 90 days';
  return 'Complete record';
}

function startOfDay(date: Date): Date {
  const d = new Date(date);
  d.setHours(0, 0, 0, 0);
  return d;
}

function weekKey(date: Date): string {
  const d = new Date(date);
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() - ((d.getDay() + 6) % 7));
  return d.toISOString().slice(0, 10);
}

/** Plain-language reading of the trend slope, with its own hedging built in. */
export function describeTrend(perWeek: number | null): string {
  if (perWeek === null) {
    return `Not enough days recorded to fit a trend (needs at least ${MIN_DAYS_FOR_TREND}).`;
  }
  const magnitude = Math.abs(perWeek);
  if (magnitude < 0.02) return 'Broadly flat across the period.';
  const direction = perWeek > 0 ? 'toward more positive' : 'toward more negative';
  const strength = magnitude < 0.06 ? 'a slight drift' : 'a clear movement';
  return `${strength[0].toUpperCase()}${strength.slice(1)} ${direction} affect (${
    perWeek > 0 ? '+' : ''
  }${(perWeek * 100).toFixed(1)} points per week).`;
}

export function describeVariability(value: number | null): string {
  if (value === null) {
    return `Not enough consecutive days to assess day-to-day variability (needs at least ${MIN_DAYS_FOR_VARIABILITY} adjacent pairs).`;
  }
  if (value < 0.15) return 'Day-to-day mood was relatively stable.';
  if (value < 0.35) return 'Moderate day-to-day variation.';
  return 'Large day-to-day swings.';
}
