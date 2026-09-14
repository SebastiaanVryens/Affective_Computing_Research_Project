/**
 * Content categories detected in the transcripts by matching words.
 *
 * Read this paragraph before adding anything here, because the design of this
 * file is a safety decision rather than a technical one.
 *
 * A lexicon detects *language*, not states. "Suicide" appearing in a transcript
 * means the word was said; it does not mean the person is suicidal, and the
 * absence of the word does not mean they are not. This module therefore does
 * exactly one thing: it finds the sentence a term appeared in and hands that
 * sentence, verbatim, to a human reader. It deliberately produces:
 *
 *   - no score
 *   - no severity, level, or band
 *   - no "risk" of any kind
 *   - no aggregation of categories into a single number
 *
 * The reasons are concrete. Nothing here resolves negation ("I don't want to
 * kill myself"), reported speech ("the documentary was about suicide"), idiom
 * ("that meeting killed me"), or tense ("I used to feel that way"). A classifier
 * that scored those uniformly would be wrong in both directions, and the
 * direction that matters is the false negative: a screen that prints "no risk
 * detected" over an entry whose wording it simply failed to match is worse than
 * a report with no such section at all. Quoting the sentence and letting the
 * reader judge is the only form of this that survives being wrong.
 *
 * Adding a category is a table entry — append to CATEGORIES. Keep phrases
 * specific enough that a reader isn't wading through false positives, and
 * remember that every match costs someone attention: the point is to surface
 * the handful of sentences worth reading, not to mark up the whole diary.
 */

/** A phrase to look for. `*` stands in for up to three intervening words. */
type Pattern = string | RegExp;

export interface ConcernCategory {
  id: string;
  /** Section heading in the report. Describes language, never a person. */
  label: string;
  /** One line telling the reader what this is and what it is not. */
  blurb: string;
  patterns: Pattern[];
  /**
   * Whether matching this category should put support resources in front of
   * the person using the app. True only where a wrong guess costs nothing and
   * a missed one might: someone who wrote these words and is fine loses two
   * seconds to a notice they can dismiss.
   */
  offersSupport?: boolean;
}

/**
 * Order matters — it is the order sections appear in the report, and the first
 * categories are the ones a reader with limited time should reach first.
 */
export const CATEGORIES: ConcernCategory[] = [
  {
    id: 'self-harm',
    label: 'Language about self-harm or suicide',
    blurb:
      'Entries containing words associated with suicide or self-harm, quoted in ' +
      'full so the context is visible. Matching is by wording alone: it does not ' +
      'distinguish a statement of intent from a denial, a memory, a figure of ' +
      'speech, or a discussion of someone else.',
    offersSupport: true,
    patterns: [
      'kill myself',
      'killing myself',
      'end my life',
      'ending my life',
      'take my own life',
      'taking my own life',
      'end it all',
      'want to die',
      'wanted to die',
      'wish i was dead',
      'wish i were dead',
      'better off dead',
      'better off without me',
      'suicidal',
      'suicide',
      'self harm',
      'self-harm',
      'harm myself',
      'hurt myself',
      'hurting myself',
      'cut myself',
      'cutting myself',
      'overdose',
      'no reason to live',
      'nothing to live for',
      'not want to be here',
      "don't want to be here",
      "don't want to wake up",
      'never wake up',
      'disappear forever',
      'stop existing',
    ],
  },
  {
    id: 'hopelessness',
    label: 'Language about hopelessness or worthlessness',
    blurb:
      'Kept separate from the category above because the two are clinically ' +
      'distinct and frequently appear without each other. These are ordinary ' +
      'words that everyone uses on a bad day — what a reader is looking for is ' +
      'repetition across weeks, not any single sentence.',
    offersSupport: true,
    patterns: [
      'hopeless',
      'no hope',
      "what's the point",
      'no point',
      'pointless',
      'nothing matters',
      "nothing's going to change",
      'never get better',
      'never gets better',
      'no way out',
      'trapped',
      'worthless',
      'a burden',
      'burden to',
      'hate myself',
      'useless',
      "can't go on",
      'given up',
      'giving up',
      'failed at everything',
      'nobody would notice',
      'no one would notice',
    ],
  },
  {
    id: 'aggression',
    label: 'Language about anger or aggression',
    blurb:
      'Words describing anger, confrontation or destructive impulse. This ' +
      'records what was said about anger — it is not a prediction of behaviour ' +
      'and nothing here distinguishes a described urge from an action.',
    patterns: [
      'lost my temper',
      'lose my temper',
      'lash out',
      'lashed out',
      'snapped at',
      'shouted at',
      'screamed at',
      'yelled at',
      'want to hit',
      'wanted to hit',
      'punch',
      'punched',
      'smash',
      'smashed',
      'threw * across',
      'broke something',
      'broken something',
      'want to hurt',
      'so angry * could',
      'furious',
      'rage',
      'raging',
      'seeing red',
      'blew up at',
      'picked a fight',
      'got in a fight',
      'violent',
    ],
  },
  {
    id: 'panic',
    label: 'Language about panic or acute anxiety',
    blurb:
      'Wording associated with panic and somatic anxiety, which is often more ' +
      'visible in what someone says than in how their face or voice reads.',
    patterns: [
      'panic attack',
      'panicking',
      "can't breathe",
      "couldn't breathe",
      'heart racing',
      'chest tight',
      'tightness in my chest',
      'shaking',
      'hyperventilating',
      'freaking out',
      'spiralling',
      'spiraling',
      'on edge',
      'dread',
      'terrified',
      'overwhelmed',
      "can't calm down",
      'racing thoughts',
    ],
  },
  {
    id: 'substance',
    label: 'Mentions of alcohol or other substances',
    blurb:
      'Substance words appear in diaries for every reason from a glass of wine ' +
      'with dinner to a way of coping. This section only shows that the subject ' +
      'came up, and in what words.',
    patterns: [
      'drinking',
      'drank',
      'drunk',
      'hungover',
      'hangover',
      'too much to drink',
      'blackout',
      'blacked out',
      'bottle of',
      'a few beers',
      'weed',
      'stoned',
      // "high" and "wasted" on their own match "expectations were high" and
      // "wasted the afternoon" far more often than anything meant here, and a
      // category that fires on every other entry gets skipped by the reader.
      'got high',
      'getting high',
      'pills',
      'sleeping pills',
      'painkillers',
      'needed a drink',
    ],
  },
  {
    id: 'sleep',
    label: 'Language about sleep',
    blurb:
      'Sleep is one of the few things a diary reports more reliably than any ' +
      'model can infer, and it moves early — which is why it earns its own ' +
      'section rather than sitting among the themes.',
    patterns: [
      "can't sleep",
      "couldn't sleep",
      'no sleep',
      'barely slept',
      'insomnia',
      'awake all night',
      'up all night',
      'lying awake',
      'woke up at',
      'kept waking',
      'nightmare',
      'nightmares',
      'slept all day',
      'sleeping too much',
      'exhausted',
      'no energy',
    ],
  },
  {
    id: 'isolation',
    label: 'Language about withdrawal or isolation',
    blurb:
      'Words about being alone or pulling away from people. Read alongside the ' +
      'recording-behaviour figures in the engagement section, which measure ' +
      'withdrawal from the diary itself.',
    patterns: [
      'lonely',
      'loneliness',
      'all alone',
      'so alone',
      'no one to talk to',
      'nobody to talk to',
      'no friends',
      'cancelled plans',
      'canceled plans',
      "didn't leave the house",
      'stayed in bed',
      'avoiding people',
      'avoided everyone',
      "haven't seen anyone",
      'ignored my phone',
      'isolated',
      'shut myself',
    ],
  },
  {
    id: 'functioning',
    label: 'Language about concentration and day-to-day functioning',
    blurb:
      'Mentions of difficulty working, concentrating or keeping up with ' +
      'ordinary tasks — the practical cost that is often what actually brings ' +
      "someone to an appointment, and which an emotion model cannot see at all.",
    patterns: [
      "can't concentrate",
      "couldn't concentrate",
      "can't focus",
      "couldn't focus",
      'behind on',
      'falling behind',
      'missed a deadline',
      'missed the deadline',
      "didn't go to work",
      'called in sick',
      'skipped class',
      'skipped work',
      'getting nothing done',
      "can't think straight",
      'brain fog',
      'forgetting things',
      'procrastinating',
    ],
  },
];

export interface LexiconMatch {
  categoryId: string;
  /** The matched text exactly as the person said it. */
  term: string;
  /** The sentence it appeared in, verbatim, trimmed only if very long. */
  quote: string;
  /** Where `term` sits inside `quote`, so the reader can see what matched. */
  highlight: { start: number; end: number };
}

/** Longest quote we will print before trimming around the match. */
const MAX_QUOTE = 240;
/** Matches per category per entry. Two is enough to show context. */
const MAX_PER_CATEGORY = 2;

/**
 * Finds every category's language in one transcript.
 *
 * Capped per category so a single entry that circles the same subject twenty
 * times contributes two quotes rather than twenty — the report is meant to be
 * readable, and the twentieth repetition tells a reader nothing the second
 * didn't.
 */
export function scanTranscript(transcript: string): LexiconMatch[] {
  const text = normalise(transcript);
  if (!text.trim()) return [];

  const sentences = splitSentences(text);
  const out: LexiconMatch[] = [];

  for (const category of CATEGORIES) {
    let found = 0;
    // Deduplicated by sentence: one sentence matching four terms from the same
    // category is one quote, not four copies of itself.
    const usedSentences = new Set<number>();

    for (const pattern of category.patterns) {
      if (found >= MAX_PER_CATEGORY) break;
      const regex = compile(pattern);

      let match: RegExpExecArray | null;
      while ((match = regex.exec(text)) !== null) {
        if (found >= MAX_PER_CATEGORY) break;
        // Zero-length matches would spin here forever.
        if (match[0].length === 0) {
          regex.lastIndex += 1;
          continue;
        }

        const index = sentenceIndexAt(sentences, match.index);
        if (usedSentences.has(index)) continue;
        usedSentences.add(index);
        found += 1;

        out.push(
          buildMatch(category.id, match[0], sentences[index], match.index, match[0].length)
        );
      }
    }
  }

  return out;
}

export function categoryById(id: string): ConcernCategory | undefined {
  return CATEGORIES.find((c) => c.id === id);
}

// ---------------------------------------------------------------------------
// Matching internals
// ---------------------------------------------------------------------------

/** Up to three intervening words, which is what `*` compiles to. */
const GAP = '\\s+(?:\\w+\\s+){0,3}';

const compiled = new Map<string, RegExp>();

function compile(pattern: Pattern): RegExp {
  if (pattern instanceof RegExp) {
    // Cloned with `g` so callers can't hand us a stateful regex, and so the
    // exec loop above terminates.
    return new RegExp(pattern.source, ensureGlobal(pattern.flags));
  }

  const cached = compiled.get(pattern);
  if (cached) {
    cached.lastIndex = 0;
    return cached;
  }

  const tokens = pattern.trim().split(/\s+/);
  let body = '';
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i] === '*') continue;
    if (body) body += tokens[i - 1] === '*' ? GAP : '\\s+';
    body += compileToken(tokens[i]);
  }

  const regex = new RegExp(`\\b${body}\\b`, 'gi');
  compiled.set(pattern, regex);
  return regex;
}

/** Contractions whose stem is not simply the token with "n't" removed. */
const IRREGULAR_NEGATIONS: Record<string, string> = {
  "can't": 'can\\s*not',
  "won't": 'will\\s+not',
  "shan't": 'shall\\s+not',
};

/**
 * Compiles one word, expanding a contracted negative to match both forms.
 *
 * A pattern written "couldn't sleep" has to match a transcript that says "could
 * not sleep", and vice versa — which form appears is a property of the
 * transcriber, not of the person. Whisper contracts most of the time and then
 * doesn't, and writing every phrase twice in the tables above would double their
 * length while still missing whichever variant someone forgot.
 *
 * "can't" and "won't" are handled by name because their stems are "ca" and "wo".
 */
function compileToken(token: string): string {
  const lower = token.toLowerCase();

  const irregular = IRREGULAR_NEGATIONS[lower];
  if (irregular) return `(?:${escapeRegex(token)}|${irregular})`;

  if (lower.length > 3 && lower.endsWith("n't")) {
    return `(?:${escapeRegex(token)}|${escapeRegex(token.slice(0, -3))}\\s+not)`;
  }

  return escapeRegex(token);
}

function ensureGlobal(flags: string): string {
  return flags.includes('g') ? flags : flags + 'g';
}

/** Escapes only true metacharacters — escaping more throws under the u flag. */
function escapeRegex(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Curly apostrophes and collapsed whitespace.
 *
 * Whisper emits U+2019 for apostrophes, so a pattern written "can't" would
 * silently never match a real transcript without this.
 */
function normalise(text: string): string {
  return text.replace(/[‘’ʼ]/g, "'").replace(/\s+/g, ' ');
}

interface Sentence {
  text: string;
  start: number;
  end: number;
}

function splitSentences(text: string): Sentence[] {
  const out: Sentence[] = [];
  // Avoids lookbehind, which is still missing in some browsers this has to
  // run in unmodified.
  const regex = /[^.!?]+[.!?]*/g;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(text)) !== null) {
    if (!match[0].trim()) continue;
    out.push({ text: match[0], start: match.index, end: match.index + match[0].length });
  }
  return out.length > 0 ? out : [{ text, start: 0, end: text.length }];
}

function sentenceIndexAt(sentences: Sentence[], index: number): number {
  for (let i = 0; i < sentences.length; i++) {
    if (index >= sentences[i].start && index < sentences[i].end) return i;
  }
  return sentences.length - 1;
}

/**
 * Builds the quote, trimming a very long sentence around the match rather than
 * from its start — the words either side of the term are the ones that tell a
 * reader whether the match means anything.
 */
function buildMatch(
  categoryId: string,
  term: string,
  sentence: Sentence,
  absoluteIndex: number,
  length: number
): LexiconMatch {
  const leading = sentence.text.length - sentence.text.trimStart().length;
  const trimmed = sentence.text.trim();
  let start = absoluteIndex - sentence.start - leading;
  let quote = trimmed;

  if (quote.length > MAX_QUOTE) {
    const context = Math.max(0, Math.floor((MAX_QUOTE - length) / 2));
    let from = Math.max(0, start - context);
    let to = Math.min(quote.length, start + length + context);
    // Snap to word boundaries so the quote doesn't start mid-word.
    while (from > 0 && /\S/.test(quote[from - 1])) from -= 1;
    while (to < quote.length && /\S/.test(quote[to])) to += 1;

    const prefix = from > 0 ? '… ' : '';
    start = start - from + prefix.length;
    quote = prefix + quote.slice(from, to).trim() + (to < trimmed.length ? ' …' : '');
  }

  // A clamp rather than an assumption: if any of the arithmetic above is off by
  // a character, a wrong highlight is cosmetic but an out-of-range slice would
  // drop the quote entirely.
  start = Math.max(0, Math.min(start, quote.length));
  return {
    categoryId,
    term,
    quote,
    highlight: { start, end: Math.min(quote.length, start + length) },
  };
}
