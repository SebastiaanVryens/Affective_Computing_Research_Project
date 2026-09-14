/**
 * Rolling entries up into the day / week / month buckets the history view draws
 * as dots.
 *
 * A bucket's emotion vector is a certainty-weighted mean of its entries, not a
 * plain average: three tentative readings shouldn't outvote one unambiguous
 * one. The same weighting is used for the world's lifetime totals, so the dot
 * you see and the colour the world took on came from the same arithmetic.
 */

import {
  type Emotion,
  type EmotionVector,
  blend,
  dominant,
  intensity,
  zeroVector,
} from '../emotions';
import { isThemeworthy } from './stopwords';
import type { DiaryEntry } from './db';

export type Granularity = 'day' | 'week' | 'month';

export interface Bucket {
  /** Sort/lookup key: "2026-09-14", "2026-W37", or "2026-09". */
  key: string;
  /** Short label for the UI. */
  label: string;
  /** First moment of the bucket, for ordering and for the tooltip. */
  start: Date;
  vector: EmotionVector;
  dominant: Emotion;
  clarity: number;
  entryCount: number;
  coreMemoryCount: number;
  /** Total time spent talking in this bucket. */
  totalSeconds: number;
  /** Highest-scoring keywords across the bucket. */
  topKeywords: Array<{ text: string; emotion: string; score: number }>;
  entries: DiaryEntry[];
}

export function bucketKeyFor(date: Date, granularity: Granularity): string {
  const year = date.getFullYear();
  switch (granularity) {
    case 'day':
      return `${year}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
    case 'week':
      return `${isoWeekYear(date)}-W${pad(isoWeekNumber(date))}`;
    case 'month':
      return `${year}-${pad(date.getMonth() + 1)}`;
  }
}

export function groupEntries(
  entries: DiaryEntry[],
  granularity: Granularity
): Bucket[] {
  const byKey = new Map<string, DiaryEntry[]>();
  for (const entry of entries) {
    const key = bucketKeyFor(new Date(entry.createdAt), granularity);
    const existing = byKey.get(key);
    if (existing) existing.push(entry);
    else byKey.set(key, [entry]);
  }

  return [...byKey.entries()]
    .map(([key, group]) => buildBucket(key, group, granularity))
    .sort((a, b) => a.start.getTime() - b.start.getTime());
}

function buildBucket(
  key: string,
  entries: DiaryEntry[],
  granularity: Granularity
): Bucket {
  const start = new Date(
    Math.min(...entries.map((e) => new Date(e.createdAt).getTime()))
  );

  const vector = blend(
    entries.map((entry) => ({
      vector: entry.vector,
      // Same shape as db.accumulate: a floor of 0.4 so a genuinely ambiguous
      // day still counts for something rather than vanishing from the average.
      weight: (0.4 + 0.6 * entry.certainty) * Math.min(3, entry.durationSeconds / 30),
    }))
  );

  return {
    key,
    label: labelFor(start, granularity),
    start,
    vector,
    dominant: dominant(vector),
    clarity: intensity(vector),
    entryCount: entries.length,
    coreMemoryCount: entries.filter((e) => e.isCoreMemory).length,
    totalSeconds: entries.reduce((sum, e) => sum + e.durationSeconds, 0),
    topKeywords: topKeywords(entries),
    entries: entries.sort((a, b) => a.createdAt.localeCompare(b.createdAt)),
  };
}

function topKeywords(
  entries: DiaryEntry[]
): Array<{ text: string; emotion: string; score: number }> {
  const merged = new Map<string, { text: string; emotion: string; score: number }>();
  for (const entry of entries) {
    for (const keyword of entry.keywords ?? []) {
      // Same filter the report uses — a bucket summarised by "something" and
      // "everything" is no more informative here than it is there.
      if (!isThemeworthy(keyword.text)) continue;
      const key = keyword.text.toLowerCase();
      const existing = merged.get(key);
      if (existing) existing.score += keyword.score;
      else merged.set(key, { ...keyword });
    }
  }
  return [...merged.values()].sort((a, b) => b.score - a.score).slice(0, 6);
}

/**
 * Pads a bucket list with empty slots so the history view shows the gaps.
 *
 * The days you didn't write anything are part of the record — a two-week hole
 * is information. Empty buckets render as hollow rings rather than being
 * silently skipped, which would compress time and make the strip a lie.
 */
export function fillGaps(
  buckets: Bucket[],
  granularity: Granularity,
  through: Date = new Date()
): Array<Bucket | { key: string; label: string; start: Date; empty: true }> {
  if (buckets.length === 0) return [];

  const byKey = new Map(buckets.map((b) => [b.key, b]));
  const out: Array<Bucket | { key: string; label: string; start: Date; empty: true }> = [];

  const cursor = startOf(buckets[0].start, granularity);
  const end = startOf(through, granularity);

  // Hard stop: a corrupt date shouldn't spin here forever.
  for (let guard = 0; cursor <= end && guard < 5000; guard++) {
    const key = bucketKeyFor(cursor, granularity);
    const found = byKey.get(key);
    out.push(
      found ?? { key, label: labelFor(cursor, granularity), start: new Date(cursor), empty: true }
    );
    advance(cursor, granularity);
  }
  return out;
}

function startOf(date: Date, granularity: Granularity): Date {
  const d = new Date(date);
  d.setHours(0, 0, 0, 0);
  if (granularity === 'month') d.setDate(1);
  if (granularity === 'week') {
    // ISO weeks start Monday; getDay() puts Sunday at 0.
    const offset = (d.getDay() + 6) % 7;
    d.setDate(d.getDate() - offset);
  }
  return d;
}

function advance(date: Date, granularity: Granularity): void {
  switch (granularity) {
    case 'day':
      date.setDate(date.getDate() + 1);
      break;
    case 'week':
      date.setDate(date.getDate() + 7);
      break;
    case 'month':
      date.setMonth(date.getMonth() + 1);
      break;
  }
}

function labelFor(date: Date, granularity: Granularity): string {
  switch (granularity) {
    case 'day':
      return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    case 'week':
      return `Week of ${date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}`;
    case 'month':
      return date.toLocaleDateString(undefined, { month: 'long', year: 'numeric' });
  }
}

// ISO-8601 week numbering. Worth doing properly: the naive "day of year / 7"
// version drifts and produces a duplicate or missing week every few years.
function isoWeekNumber(date: Date): number {
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  d.setUTCDate(d.getUTCDate() + 4 - (d.getUTCDay() || 7)); // shift to that week's Thursday
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  return Math.ceil(((d.getTime() - yearStart.getTime()) / 86_400_000 + 1) / 7);
}

function isoWeekYear(date: Date): number {
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  d.setUTCDate(d.getUTCDate() + 4 - (d.getUTCDay() || 7));
  return d.getUTCFullYear();
}

function pad(n: number): string {
  return `${n}`.padStart(2, '0');
}

/** Lifetime summary for the header strip. */
export function summarize(entries: DiaryEntry[]): {
  vector: EmotionVector;
  totalEntries: number;
  totalMinutes: number;
  streakDays: number;
} {
  if (entries.length === 0) {
    return { vector: zeroVector(), totalEntries: 0, totalMinutes: 0, streakDays: 0 };
  }

  const vector = blend(
    entries.map((e) => ({ vector: e.vector, weight: 0.4 + 0.6 * e.certainty }))
  );

  const days = new Set(entries.map((e) => e.day));
  let streak = 0;
  const cursor = new Date();
  // Today not being written yet shouldn't break a streak that's still alive,
  // so start counting from today and allow the first day to be missing.
  for (let guard = 0; guard < 3650; guard++) {
    const key = bucketKeyFor(cursor, 'day');
    if (days.has(key)) streak++;
    else if (guard > 0) break;
    cursor.setDate(cursor.getDate() - 1);
  }

  return {
    vector,
    totalEntries: entries.length,
    totalMinutes: Math.round(entries.reduce((s, e) => s + e.durationSeconds, 0) / 60),
    streakDays: streak,
  };
}
