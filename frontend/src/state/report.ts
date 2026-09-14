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
 *
 * The same scope note governs the later sections. Hostility is reported as
 * *observed anger*, never as a prediction of behaviour; flatness is reported as
 * a set of proxies with their thresholds stated, never as anhedonia; and the
 * language sections quote sentences rather than scoring them. See
 * ./lexicon.ts for why that last one is built the way it is.
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
import { CATEGORIES, type LexiconMatch, scanTranscript } from './lexicon';
import { isThemeworthy } from './stopwords';
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

/**
 * The hostility grouping.
 *
 * Anger and disgust together, because contempt — which is most of what clinical
 * "hostility" means — lands between the two in every categorical scheme, and
 * MELD has no label for it. Reporting anger alone would systematically under-
 * count exactly the presentation this section exists to make visible.
 */
export const HOSTILE: Emotion[] = ['anger', 'disgust'];

/** Minimum distinct days before a trend line is worth reporting at all. */
const MIN_DAYS_FOR_TREND = 7;
/** Minimum days before day-to-day variability means anything. */
const MIN_DAYS_FOR_VARIABILITY = 5;
/** Minimum weeks before a change in recording behaviour is worth reporting. */
const MIN_WEEKS_FOR_ENGAGEMENT = 3;
/** Minimum multi-entry days before within-day range means anything. */
const MIN_DAYS_FOR_RANGE = 3;

/**
 * Thresholds for the flatness proxies.
 *
 * These are arbitrary in the sense that no validated cut-off exists for a
 * seven-class expression model, so they are named, exported, and printed in the
 * report rather than buried — a reader who disagrees with where the line sits
 * can see exactly where it was drawn and discount accordingly.
 */
export const LOW_JOY_THRESHOLD = 0.08;
export const FLAT_CHARGE_THRESHOLD = 0.35;

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

/** Per-emotion slope, so movements that cancel out in valence stay visible. */
export interface EmotionTrend {
  emotion: Emotion;
  share: number;
  perWeek: number | null;
}

export interface HostilitySummary {
  /** Anger + disgust mass across the period. */
  share: number;
  perWeek: number | null;
  /** Consecutive recorded days whose dominant affect was anger or disgust. */
  longestRunDays: number;
  daysDominant: number;
  peakDay: { date: Date; share: number } | null;
  /** What the person was talking about on hostility-dominant days. */
  themes: Array<{ text: string; count: number }>;
}

/**
 * Proxies for flattened affect.
 *
 * Named "flatness" rather than "anhedonia" throughout, deliberately. Anhedonia
 * is a symptom with a definition this data cannot satisfy — it is about
 * anticipated and consummatory pleasure, not about how much joy a classifier
 * read off a face. What is measurable here is that little positive affect was
 * expressed and little of anything else was either, which is worth a reader's
 * attention on its own terms without borrowing a clinical word for it.
 */
export interface FlatnessSummary {
  joyShare: number;
  joyPerWeek: number | null;
  neutralShare: number;
  meanCharge: number;
  chargePerWeek: number | null;
  /** Recorded days whose charge fell below FLAT_CHARGE_THRESHOLD. */
  flatDays: number;
  /** Longest consecutive run of recorded days under LOW_JOY_THRESHOLD. */
  longestLowJoyRun: number;
}

/**
 * Swing *within* a single day, which day-to-day variability cannot see.
 *
 * Only days with two or more entries contribute — a range computed from one
 * reading is zero, and averaging those zeros in would make anyone who records
 * once a day look stable by definition.
 */
export interface WithinDaySummary {
  daysWithMultipleEntries: number;
  meanRange: number | null;
  widestDay: { date: Date; range: number; entryCount: number } | null;
}

export interface EngagementWeek {
  weekStart: Date;
  entries: number;
  meanMinutes: number;
}

/**
 * Withdrawal from the diary itself.
 *
 * Disengagement is information — someone recording daily in week one and once
 * in week six has told you something the emotion vectors have not. It is also
 * the figure most easily confounded by ordinary life, so it is reported as
 * behaviour and left uninterpreted.
 */
export interface EngagementSummary {
  weeks: EngagementWeek[];
  /** Change in entries per week, per week. Null below MIN_WEEKS_FOR_ENGAGEMENT. */
  entriesPerWeekTrend: number | null;
  /** Change in minutes per entry, per week. */
  durationTrendPerWeek: number | null;
  daysSinceLastEntry: number;
}

/** One lexicon category's hits, as quotes. Never as a count of anything else. */
export interface LanguageFlagGroup {
  id: string;
  label: string;
  blurb: string;
  offersSupport: boolean;
  entryCount: number;
  dayCount: number;
  firstSeen: Date;
  lastSeen: Date;
  excerpts: Array<{
    date: Date;
    emotion: Emotion;
    match: LexiconMatch;
  }>;
  /** Matches beyond the display cap, so the reader knows the list is partial. */
  omitted: number;
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
  /** One slope per emotion, ranked by share. */
  emotionTrends: EmotionTrend[];
  withinDay: WithinDaySummary;

  // -- dimensions ------------------------------------------------------
  hostility: HostilitySummary;
  flatness: FlatnessSummary;
  engagement: EngagementSummary;

  // -- content ---------------------------------------------------------
  notable: NotableEntry[];
  coreMemories: NotableEntry[];
  themes: Array<{ text: string; emotion: Emotion; count: number }>;
  languageFlags: LanguageFlagGroup[];

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
    emotionTrends: emotionTrends(days, overall),
    withinDay: withinDaySummary(entries),

    hostility: hostilitySummary(days, overall, entries),
    flatness: flatnessSummary(days, overall),
    engagement: engagementSummary(entries, now),

    notable: notableEntries(entries),
    coreMemories: entries.filter((e) => e.isCoreMemory).map(toNotable),
    themes: recurringThemes(entries),
    languageFlags: languageFlags(entries),

    channels: channelReliability(entries),
    transcribedShare: entries.length
      ? entries.filter((e) => e.transcript.trim().length > 0).length / entries.length
      : 0,
  };
}

// ---------------------------------------------------------------------------
// Days
// ---------------------------------------------------------------------------

function groupByDay(entries: DiaryEntry[]): Map<string, DiaryEntry[]> {
  const byDay = new Map<string, DiaryEntry[]>();
  for (const entry of entries) {
    const list = byDay.get(entry.day);
    if (list) list.push(entry);
    else byDay.set(entry.day, [entry]);
  }
  return byDay;
}

function summariseDays(entries: DiaryEntry[]): DaySummary[] {
  return [...groupByDay(entries).entries()]
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
  return dailySlope(days, (d) => d.valence);
}

/**
 * Least-squares slope of some per-day quantity, expressed per week.
 *
 * Shared by every trend in this file so that "per week" means the same thing
 * everywhere: regressed against elapsed days, gated on the same minimum sample,
 * and null rather than a small number when there isn't enough to fit.
 */
function dailySlope(
  days: DaySummary[],
  value: (day: DaySummary) => number,
  minDays = MIN_DAYS_FOR_TREND
): number | null {
  if (days.length < minDays) return null;
  const origin = days[0].date.getTime();
  return slopePerUnit(
    days.map((d) => ({ x: (d.date.getTime() - origin) / 86_400_000, y: value(d) }))
  );
}

/** The regression itself. x in days, result per week. Null if x never varies. */
function slopePerUnit(points: Array<{ x: number; y: number }>): number | null {
  const n = points.length;
  if (n < 2) return null;

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
 * One slope per emotion.
 *
 * Valence alone hides the case that matters most: fear climbing while sadness
 * falls nets out to "broadly flat", which is a true statement about the average
 * and a misleading one about the person. Sorted by share so the emotions
 * actually present lead, rather than the canonical wire order.
 */
function emotionTrends(days: DaySummary[], overall: EmotionVector): EmotionTrend[] {
  return EMOTIONS.map((emotion) => ({
    emotion,
    share: overall[emotion],
    perWeek: dailySlope(days, (d) => d.vector[emotion]),
  })).sort((a, b) => b.share - a.share);
}

/**
 * Widest swing inside a single day, averaged over days that had more than one
 * entry. Days with a single entry are excluded rather than counted as zero.
 */
function withinDaySummary(entries: DiaryEntry[]): WithinDaySummary {
  const byDay = groupByDay(entries);

  const ranges: Array<{ date: Date; range: number; entryCount: number }> = [];
  for (const group of byDay.values()) {
    if (group.length < 2) continue;
    const valences = group.map(
      (e) => shareOf(e.vector, POSITIVE) - shareOf(e.vector, NEGATIVE)
    );
    ranges.push({
      date: startOfDay(new Date(group[0].createdAt)),
      range: Math.max(...valences) - Math.min(...valences),
      entryCount: group.length,
    });
  }

  const widest = ranges.reduce<WithinDaySummary['widestDay']>(
    (best, r) => (best === null || r.range > best.range ? r : best),
    null
  );

  return {
    daysWithMultipleEntries: ranges.length,
    meanRange:
      ranges.length >= MIN_DAYS_FOR_RANGE
        ? ranges.reduce((s, r) => s + r.range, 0) / ranges.length
        : null,
    widestDay: widest,
  };
}

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

/**
 * Observed anger and disgust over time.
 *
 * Every word in this summary describes what was *expressed*, because that is
 * all the data supports. It is worth being blunt about why: a report that
 * turned an expression classifier into a statement about aggression would be
 * making a claim about what someone might do, from evidence that only says how
 * their face looked while they talked about their week.
 */
function hostilitySummary(
  days: DaySummary[],
  overall: EmotionVector,
  entries: DiaryEntry[]
): HostilitySummary {
  const dominantDays = days.filter((d) => HOSTILE.includes(d.dominant));

  const peak = days.reduce<HostilitySummary['peakDay']>((best, d) => {
    const share = shareOf(d.vector, HOSTILE);
    return best === null || share > best.share ? { date: d.date, share } : best;
  }, null);

  const hostileDayKeys = new Set(dominantDays.map((d) => d.day));
  const hostileEntries = entries.filter((e) => hostileDayKeys.has(e.day));

  return {
    share: shareOf(overall, HOSTILE),
    perWeek: dailySlope(days, (d) => shareOf(d.vector, HOSTILE)),
    longestRunDays: longestRun(days, (d) => HOSTILE.includes(d.dominant)),
    daysDominant: dominantDays.length,
    peakDay: peak,
    themes: recurringThemes(hostileEntries, 8).map(({ text, count }) => ({ text, count })),
  };
}

/** The flatness proxies. Thresholds are exported so the report can print them. */
function flatnessSummary(days: DaySummary[], overall: EmotionVector): FlatnessSummary {
  return {
    joyShare: overall.joy,
    joyPerWeek: dailySlope(days, (d) => d.vector.joy),
    neutralShare: overall.neutral,
    meanCharge: days.length
      ? days.reduce((s, d) => s + d.charge, 0) / days.length
      : 0,
    chargePerWeek: dailySlope(days, (d) => d.charge),
    flatDays: days.filter((d) => d.charge < FLAT_CHARGE_THRESHOLD).length,
    longestLowJoyRun: longestRun(days, (d) => d.vector.joy < LOW_JOY_THRESHOLD),
  };
}

/**
 * Recording behaviour, week by week.
 *
 * Weeks with no entries at all are included as zeroes — they are the whole
 * point. Dropping them would turn six weeks of silence into an unbroken line
 * and hide precisely the disengagement this is here to show.
 */
function engagementSummary(entries: DiaryEntry[], now: Date): EngagementSummary {
  if (entries.length === 0) {
    return {
      weeks: [],
      entriesPerWeekTrend: null,
      durationTrendPerWeek: null,
      daysSinceLastEntry: 0,
    };
  }

  const byWeek = new Map<string, DiaryEntry[]>();
  for (const entry of entries) {
    const key = weekKey(new Date(entry.createdAt));
    const list = byWeek.get(key);
    if (list) list.push(entry);
    else byWeek.set(key, [entry]);
  }

  const first = weekStartOf(new Date(entries[0].createdAt));
  const last = weekStartOf(new Date(entries[entries.length - 1].createdAt));
  const weeks: EngagementWeek[] = [];
  for (
    let cursor = new Date(first);
    cursor.getTime() <= last.getTime();
    cursor.setDate(cursor.getDate() + 7)
  ) {
    const group = byWeek.get(cursor.toISOString().slice(0, 10)) ?? [];
    weeks.push({
      weekStart: new Date(cursor),
      entries: group.length,
      meanMinutes: group.length
        ? group.reduce((s, e) => s + e.durationSeconds, 0) / group.length / 60
        : 0,
    });
  }

  // Entry count is regressed per week index; duration per elapsed day, since a
  // single entry has a duration but a week of them has a rate.
  const entriesPerWeekTrend =
    weeks.length >= MIN_WEEKS_FOR_ENGAGEMENT
      ? slopePerUnit(weeks.map((w, i) => ({ x: i * 7, y: w.entries })))
      : null;

  const origin = new Date(entries[0].createdAt).getTime();
  const durationTrendPerWeek =
    entries.length >= MIN_DAYS_FOR_TREND
      ? slopePerUnit(
          entries.map((e) => ({
            x: (new Date(e.createdAt).getTime() - origin) / 86_400_000,
            y: e.durationSeconds / 60,
          }))
        )
      : null;

  const lastEntry = new Date(entries[entries.length - 1].createdAt);
  return {
    weeks,
    entriesPerWeekTrend,
    durationTrendPerWeek,
    daysSinceLastEntry: Math.max(
      0,
      Math.round(
        (startOfDay(now).getTime() - startOfDay(lastEntry).getTime()) / 86_400_000
      )
    ),
  };
}

/** Longest run of consecutive *calendar* days satisfying a predicate. */
function longestRun(days: DaySummary[], predicate: (day: DaySummary) => boolean): number {
  let longest = 0;
  let current = 0;
  for (let i = 0; i < days.length; i++) {
    const consecutive =
      i === 0 ||
      Math.round((days[i].date.getTime() - days[i - 1].date.getTime()) / 86_400_000) === 1;
    current = predicate(days[i]) ? (consecutive ? current + 1 : 1) : 0;
    longest = Math.max(longest, current);
  }
  return longest;
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
  return longestRun(days, (d) => NEGATIVE.includes(d.dominant));
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
      // Dropped here rather than at extraction because stored entries carry
      // whatever their extractor produced at the time. See ./stopwords.ts.
      if (!isThemeworthy(keyword.text)) continue;
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

/** Quotes shown per category before the list is truncated. */
const MAX_EXCERPTS_PER_CATEGORY = 10;

/**
 * Groups the lexicon's hits by category, as dated quotes.
 *
 * Note what is *not* returned: no rate, no per-category score, no total. The
 * counts that are here — how many entries, over how many days, first and last —
 * exist to let a reader see whether something appeared once in March or in
 * every entry for a fortnight, which is the distinction that actually changes
 * how a sentence should be read. Categories with no hits are omitted entirely
 * rather than printed as "none found", because "none found" reads as
 * reassurance and word-matching cannot support that.
 */
function languageFlags(entries: DiaryEntry[]): LanguageFlagGroup[] {
  const hits = new Map<
    string,
    Array<{ entry: DiaryEntry; match: LexiconMatch }>
  >();

  for (const entry of entries) {
    for (const match of scanTranscript(entry.transcript)) {
      const list = hits.get(match.categoryId);
      if (list) list.push({ entry, match });
      else hits.set(match.categoryId, [{ entry, match }]);
    }
  }

  const groups: LanguageFlagGroup[] = [];
  // Iterating CATEGORIES rather than the map keeps the report's section order
  // fixed and independent of whichever category happened to match first.
  for (const category of CATEGORIES) {
    const list = hits.get(category.id);
    if (!list || list.length === 0) continue;

    const chronological = [...list].sort((a, b) =>
      a.entry.createdAt.localeCompare(b.entry.createdAt)
    );
    const shown = chronological.slice(0, MAX_EXCERPTS_PER_CATEGORY);

    groups.push({
      id: category.id,
      label: category.label,
      blurb: category.blurb,
      offersSupport: category.offersSupport ?? false,
      entryCount: new Set(chronological.map((h) => h.entry.id)).size,
      dayCount: new Set(chronological.map((h) => h.entry.day)).size,
      firstSeen: new Date(chronological[0].entry.createdAt),
      lastSeen: new Date(chronological[chronological.length - 1].entry.createdAt),
      excerpts: shown.map(({ entry, match }) => ({
        date: new Date(entry.createdAt),
        emotion: entry.dominant,
        match,
      })),
      omitted: chronological.length - shown.length,
    });
  }

  return groups;
}

/**
 * Whether any entry contains language from a category that should put support
 * resources in front of the person using the app.
 *
 * Takes entries rather than a built report so the caller doesn't have to build
 * one — this is asked before the report exists, on the screen where someone is
 * deciding whether to hand the document to another person.
 */
export function hasSupportLanguage(entries: DiaryEntry[]): boolean {
  const supported = new Set(
    CATEGORIES.filter((c) => c.offersSupport).map((c) => c.id)
  );
  return entries.some((entry) =>
    scanTranscript(entry.transcript).some((m) => supported.has(m.categoryId))
  );
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

/** Monday of the week containing `date`, local time. */
function weekStartOf(date: Date): Date {
  const d = startOfDay(date);
  d.setDate(d.getDate() - ((d.getDay() + 6) % 7));
  return d;
}

function weekKey(date: Date): string {
  return weekStartOf(date).toISOString().slice(0, 10);
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

/**
 * Wording for a single emotion's slope.
 *
 * Kept neutral on purpose — "rising" and "falling", not "worsening" and
 * "improving". More fear is not straightforwardly worse than more sadness, and
 * for someone who arrived numb, more of almost anything may be progress. That
 * judgement belongs to the reader, who knows things this document does not.
 */
export function describeEmotionTrend(perWeek: number | null): string {
  if (perWeek === null) return 'not enough days';
  const magnitude = Math.abs(perWeek);
  if (magnitude < 0.01) return 'steady';
  const direction = perWeek > 0 ? 'rising' : 'falling';
  return `${direction} ${(magnitude * 100).toFixed(1)} pts/wk`;
}

export function describeHostility(h: HostilitySummary): string {
  if (h.daysDominant === 0) {
    return (
      `Anger and disgust together accounted for ${Math.round(h.share * 100)}% of ` +
      'recorded affect, and no single day was dominated by either.'
    );
  }
  const run =
    h.longestRunDays > 1
      ? ` The longest unbroken stretch was ${h.longestRunDays} consecutive recorded days.`
      : '';
  return (
    `Anger and disgust together accounted for ${Math.round(h.share * 100)}% of ` +
    `recorded affect. ${h.daysDominant} recorded ${
      h.daysDominant === 1 ? 'day was' : 'days were'
    } dominated by one of the two.${run}`
  );
}

export function describeFlatness(f: FlatnessSummary, daysCovered: number): string {
  if (daysCovered === 0) return 'No recorded days in this period.';
  const flatShare = f.flatDays / daysCovered;
  const parts = [
    `Joy made up ${Math.round(f.joyShare * 100)}% of recorded affect and neutral ` +
      `${Math.round(f.neutralShare * 100)}%.`,
  ];
  if (f.flatDays > 0) {
    parts.push(
      `${f.flatDays} of ${daysCovered} recorded days (${Math.round(flatShare * 100)}%) ` +
        'carried little emotional content of any kind.'
    );
  }
  if (f.longestLowJoyRun >= 3) {
    parts.push(
      `${f.longestLowJoyRun} consecutive recorded days passed with almost no joy expressed.`
    );
  }
  return parts.join(' ');
}

export function describeEngagement(e: EngagementSummary): string {
  if (e.weeks.length === 0) return 'No entries in this period.';

  const parts: string[] = [];
  if (e.entriesPerWeekTrend === null) {
    parts.push(
      `Too few weeks to say whether recording is changing (needs at least ${MIN_WEEKS_FOR_ENGAGEMENT}).`
    );
  } else if (Math.abs(e.entriesPerWeekTrend) < 0.15) {
    parts.push('Recording frequency was broadly steady across the period.');
  } else {
    const direction = e.entriesPerWeekTrend > 0 ? 'rose' : 'fell';
    parts.push(
      `Recording frequency ${direction} by about ${Math.abs(
        e.entriesPerWeekTrend
      ).toFixed(1)} entries per week, week on week.`
    );
  }

  if (e.durationTrendPerWeek !== null && Math.abs(e.durationTrendPerWeek) >= 0.2) {
    const direction = e.durationTrendPerWeek > 0 ? 'longer' : 'shorter';
    parts.push(
      `Entries also got ${direction} — by roughly ${Math.abs(
        e.durationTrendPerWeek
      ).toFixed(1)} minutes per week.`
    );
  }

  if (e.daysSinceLastEntry >= 7) {
    parts.push(
      `The most recent entry was ${e.daysSinceLastEntry} days before this report was generated.`
    );
  }

  return parts.join(' ');
}

export function describeWithinDay(w: WithinDaySummary): string {
  if (w.meanRange === null) {
    return (
      `Only ${w.daysWithMultipleEntries} recorded ${
        w.daysWithMultipleEntries === 1 ? 'day has' : 'days have'
      } more than one entry, which is too few to describe swing within a day ` +
      `(needs at least ${MIN_DAYS_FOR_RANGE}).`
    );
  }
  const descriptor =
    w.meanRange < 0.2 ? 'narrow' : w.meanRange < 0.45 ? 'moderate' : 'wide';
  return (
    `Across ${w.daysWithMultipleEntries} days with more than one entry, the spread ` +
    `between the best and worst reading of the same day averaged ${w.meanRange.toFixed(
      2
    )} — a ${descriptor} range.`
  );
}
