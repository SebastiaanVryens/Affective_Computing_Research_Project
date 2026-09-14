/**
 * Renders the clinical report and hands it to the browser's print-to-PDF.
 *
 * Why print rather than a PDF library: the diary has never left the browser,
 * and generating the file anywhere else would break that. Printing keeps
 * everything local, adds no dependency to an already-large bundle, produces
 * selectable text rather than a rasterised page, and lets the reader see what
 * they're saving before they save it.
 *
 * Charts are inline SVG for the same reasons — they print at the printer's
 * resolution rather than the screen's, and need no library.
 */

import { PALETTE, type Emotion, type EmotionVector, EMOTIONS } from '../emotions';
import {
  type ClinicalReport,
  type DaySummary,
  type EngagementSummary,
  type LanguageFlagGroup,
  type Period,
  FLAT_CHARGE_THRESHOLD,
  LOW_JOY_THRESHOLD,
  NEGATIVE,
  buildReport,
  describeEmotionTrend,
  describeEngagement,
  describeFlatness,
  describeHostility,
  describeTrend,
  describeVariability,
  describeWithinDay,
} from '../state/report';
import type { LexiconMatch } from '../state/lexicon';
import type { DiaryEntry } from '../state/db';

const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * Guards against a second export starting while one is still open.
 *
 * The button is easy to double-click, and Enter in the name field fires it too.
 * Two overlapping runs would put two report nodes in the DOM, and since the
 * print stylesheet shows everything matching #print-report, the PDF would come
 * out with the whole document twice.
 */
let printing = false;

export function generateReport(
  entries: DiaryEntry[],
  name: string,
  period: Period
): void {
  if (printing) return;
  printing = true;

  // Nothing should be here, but if a previous cleanup was missed the leftover
  // would print alongside this one. Clearing first makes a duplicate
  // impossible regardless of how the previous run ended.
  for (const stale of document.querySelectorAll('#print-report')) stale.remove();

  const report = buildReport(entries, name, period);

  const host = document.createElement('div');
  host.id = 'print-report';
  host.append(
    buildHeader(report),
    buildNotice(report),
    buildAtAGlance(report),
    buildProfile(report),
    buildTimeline(report),
    buildPatterns(report),
    buildHostility(report),
    buildFlatness(report),
    buildEngagement(report),
    buildVoices(report),
    ...buildLanguageSections(report),
    buildThemes(report),
    buildProvenance(report),
    buildMethod()
  );

  document.body.append(host);
  document.body.classList.add('printing');

  let fallbackTimer = 0;

  /**
   * Idempotent, and removes *every* report node rather than just this one.
   *
   * Belt and braces against the duplicate-report bug: if a previous cleanup was
   * ever missed, this sweeps up whatever it left behind instead of adding to it.
   */
  const cleanup = (): void => {
    window.clearTimeout(fallbackTimer);
    window.removeEventListener('afterprint', cleanup);
    for (const node of document.querySelectorAll('#print-report')) node.remove();
    document.body.classList.remove('printing');
    printing = false;
  };

  // Registered *before* print(), which is the whole point.
  //
  // window.print() blocks until the dialog is dismissed in Chrome and Firefox,
  // so `afterprint` fires while that call is still on the stack. Attaching the
  // listener afterwards means it is registered after the event has already
  // passed, cleanup never runs, and the report node stays in the DOM — where
  // the next export appends a second one beside it. Both match #print-report,
  // both print, and the PDF contains the whole document twice.
  window.addEventListener('afterprint', cleanup);

  // Safety net for browsers that have historically not fired afterprint. Long
  // enough that someone reading the print preview is never cut off mid-review.
  fallbackTimer = window.setTimeout(cleanup, 5 * 60_000);

  // Let layout and fonts settle before the dialog snapshots the page; printing
  // synchronously can capture a half-laid-out document.
  window.setTimeout(() => window.print(), 120);
}

// ---------------------------------------------------------------------------
// Sections
// ---------------------------------------------------------------------------

function buildHeader(r: ClinicalReport): HTMLElement {
  const header = el('header', 'rp-header');
  header.append(
    el('h1', 'rp-title', 'Emotional diary summary'),
    keyValue('Name', r.name),
    keyValue(
      'Period',
      `${formatDate(r.periodStart)} — ${formatDate(r.periodEnd)} (${r.periodLabel})`
    ),
    keyValue('Generated', formatDateTime(r.generatedAt))
  );
  return header;
}

/**
 * The notice is not boilerplate and is deliberately the first thing after the
 * header. This document can end up in front of someone making decisions, and it
 * is produced by a consumer app using models that were never validated for
 * clinical use. Saying so plainly is the difference between a useful adjunct and
 * a misleading one.
 */
function buildNotice(r: ClinicalReport): HTMLElement {
  const box = el('section', 'rp-notice');
  box.append(
    el('h2', 'rp-notice-title', 'What this document is'),
    el(
      'p',
      '',
      'A summary of self-recorded diary entries. The emotion labels are produced ' +
        'automatically from facial expression, voice and transcribed speech by ' +
        'general-purpose models that have not been validated for clinical use.'
    ),
    el(
      'p',
      '',
      'It is not a diagnostic instrument, not a screening tool, and contains no ' +
        'clinical scoring. It carries roughly the evidential weight of a paper ' +
        'mood diary that has been counted up — useful as a prompt for ' +
        'conversation and for noticing patterns over time, not as a measurement.'
    ),
    el(
      'p',
      '',
      'Accuracy varies by emotion and is poorest for fear and disgust. See ' +
        '"Method and limitations" at the end before drawing conclusions from ' +
        'any figure here.'
    )
  );

  // A pointer rather than a summary. Someone who reads only the first page
  // should learn that these sections exist and that they are quotations — but
  // putting the quotes themselves up here would lead the document with its
  // most alarming content, out of the context that makes it readable.
  const support = r.languageFlags.filter((g) => g.offersSupport);
  if (support.length > 0) {
    box.append(
      el(
        'p',
        'rp-notice-pointer',
        'This report contains sections quoting the author’s own words on ' +
          `${listOf(support.map((g) => g.label.replace(/^Language about /, '')))}. ` +
          'Those sections are quotations found by matching words, not an ' +
          'assessment — nothing in this document estimates risk, and the absence ' +
          'of a section is not evidence that a subject was never on the ' +
          'author’s mind.'
      )
    );
  }
  return box;
}

function buildAtAGlance(r: ClinicalReport): HTMLElement {
  const section = sectionWith('At a glance');
  const grid = el('div', 'rp-stats');

  grid.append(
    stat(String(r.entryCount), 'entries'),
    stat(`${r.daysCovered}/${r.daysInPeriod}`, 'days recorded'),
    stat(`${Math.round(r.coverage * 100)}%`, 'coverage'),
    stat(`${r.totalMinutes}`, 'minutes total'),
    stat(`${r.medianEntryMinutes}`, 'median entry (min)'),
    stat(String(r.longestGapDays), 'longest gap (days)')
  );
  section.append(grid);

  if (r.entryCount < 5) {
    section.append(
      el(
        'p',
        'rp-caution',
        `Only ${r.entryCount} ${r.entryCount === 1 ? 'entry' : 'entries'} in this ` +
          'period. Everything below describes a very small sample and should be ' +
          'read as anecdote rather than as pattern.'
      )
    );
  }
  return section;
}

function buildProfile(r: ClinicalReport): HTMLElement {
  const section = sectionWith('Emotional profile');

  section.append(
    el(
      'p',
      'rp-lead',
      `Across the period, ${pct(r.negativeShare)} of recorded affect fell into ` +
        `negative categories (sadness, anger, fear, disgust) and ${pct(r.positiveShare)} ` +
        'into joy. The remainder is neutral or surprise, which is counted as neither.'
    )
  );

  const ranked = EMOTIONS.map((e) => ({ emotion: e, share: r.overall[e] }))
    .filter((x) => x.share >= 0.005)
    .sort((a, b) => b.share - a.share);

  const bars = el('div', 'rp-bars');
  for (const { emotion, share } of ranked) {
    const row = el('div', 'rp-bar-row');
    row.append(
      el('span', 'rp-bar-label', PALETTE[emotion].label),
      barTrack(share, PALETTE[emotion].base),
      el('span', 'rp-bar-value', pct(share))
    );
    bars.append(row);
  }
  section.append(bars);
  return section;
}

function buildTimeline(r: ClinicalReport): HTMLElement {
  const section = sectionWith('Day by day');
  if (r.days.length === 0) {
    section.append(el('p', '', 'No entries in this period.'));
    return section;
  }

  section.append(
    el(
      'p',
      'rp-lead',
      'Each bar is one recorded day. Height shows how much emotional content the ' +
        'entry carried; colour shows the dominant emotion. Bars below the centre ' +
        'line are negative-dominant days. Unrecorded days are left blank.'
    ),
    valenceChart(r.days),
    legendFor(r.days)
  );
  return section;
}

function buildPatterns(r: ClinicalReport): HTMLElement {
  const section = sectionWith('Patterns');

  const list = el('dl', 'rp-findings');
  addFinding(list, 'Direction', describeTrend(r.valenceTrendPerWeek));
  addFinding(list, 'Variability', describeVariability(r.variability));
  addFinding(list, 'Within a day', describeWithinDay(r.withinDay));
  addFinding(
    list,
    'Consecutive negative days',
    r.longestNegativeRun === 0
      ? 'No run of consecutive recorded days with negative dominant affect.'
      : `Longest run: ${r.longestNegativeRun} consecutive recorded ${
          r.longestNegativeRun === 1 ? 'day' : 'days'
        } with negative dominant affect.`
  );
  section.append(list);

  // Per-emotion slopes. The aggregate figure above can only move if the net
  // moves, so an emotion rising while another falls reads there as "flat".
  const moving = r.emotionTrends.filter((t) => t.share >= 0.02);
  if (moving.length > 0) {
    section.append(
      el('h3', 'rp-subhead', 'Each emotion separately'),
      el(
        'p',
        'rp-lead',
        'The direction figure above is a net, so two emotions moving opposite ways ' +
          'cancel in it. These are the same slope fitted to each emotion on its ' +
          'own. "Rising" and "falling" describe the numbers, not whether things ' +
          'are getting better.'
      )
    );

    const table = el('table', 'rp-table');
    table.innerHTML =
      '<thead><tr><th>Emotion</th><th>Share</th><th>Direction</th></tr></thead>';
    const body = el('tbody');
    for (const trend of moving) {
      const row = el('tr');
      const nameCell = el('td');
      nameCell.append(
        swatch(PALETTE[trend.emotion].base),
        document.createTextNode(' ' + PALETTE[trend.emotion].label)
      );
      row.append(nameCell, el('td', '', pct(trend.share)), el('td', '', describeEmotionTrend(trend.perWeek)));
      body.append(row);
    }
    table.append(body);
    section.append(table);
  }

  const withEntries = r.timeOfDay.filter((b) => b.entryCount > 0);
  if (withEntries.length > 1) {
    section.append(el('h3', 'rp-subhead', 'Time of day'));
    const table = el('table', 'rp-table');
    table.innerHTML =
      '<thead><tr><th>Band</th><th>Entries</th><th>Mean valence</th></tr></thead>';
    const body = el('tbody');
    for (const band of withEntries) {
      const row = el('tr');
      row.append(
        el('td', '', band.label),
        el('td', '', String(band.entryCount)),
        el('td', '', signed(band.meanValence))
      );
      body.append(row);
    }
    table.append(body);
    section.append(
      table,
      el(
        'p',
        'rp-note',
        'Valence runs from −1 (entirely negative) to +1 (entirely joy). ' +
          'Bands with few entries will be noisy.'
      )
    );
  }
  return section;
}

/**
 * Anger and disgust as their own tracked dimension.
 *
 * The heading says "expressed", and the framing paragraph says what this is
 * not, because the gap between "this person expressed a lot of anger" and "this
 * person is aggressive" is the entire distance between a mood diary and a risk
 * assessment. The data supports the first and says nothing about the second.
 */
function buildHostility(r: ClinicalReport): HTMLElement {
  const section = sectionWith('Anger and disgust expressed');
  const h = r.hostility;

  section.append(
    el('p', 'rp-lead', describeHostility(h)),
    el(
      'p',
      'rp-note',
      'Anger and disgust are counted together because contempt — which is most ' +
        'of what is usually meant by hostility — falls between them and has no ' +
        'label of its own in this scheme. This describes emotion that was ' +
        'expressed while recording. It is not a measure of behaviour towards ' +
        'anyone and carries no implication about what the author did or might do.'
    )
  );

  if (h.peakDay && h.share > 0) {
    const list = el('dl', 'rp-findings');
    addFinding(
      list,
      'Highest day',
      `${formatDate(h.peakDay.date)} — ${pct(h.peakDay.share)} of that day's affect.`
    );
    addFinding(
      list,
      'Direction',
      h.perWeek === null
        ? 'Not enough days recorded to fit a trend.'
        : Math.abs(h.perWeek) < 0.01
          ? 'Broadly steady across the period.'
          : `${h.perWeek > 0 ? 'Rising' : 'Falling'} by ${Math.abs(h.perWeek * 100).toFixed(
              1
            )} points per week.`
    );
    section.append(list);
  }

  if (h.themes.length > 0) {
    section.append(
      el('h3', 'rp-subhead', 'What was being talked about on those days'),
      el(
        'p',
        'rp-note',
        'Keywords from entries recorded on days where anger or disgust dominated. ' +
          'Co-occurrence only — these are subjects that came up, not causes.'
      ),
      keywordRow(h.themes)
    );
  }

  return section;
}

/**
 * The flatness proxies.
 *
 * Every threshold is printed in the section that uses it. They are judgement
 * calls with no validated basis, and a reader who would have drawn the line
 * elsewhere can only discount the figure if they can see where it was drawn.
 */
function buildFlatness(r: ClinicalReport): HTMLElement {
  const section = sectionWith('Positive affect and flatness');

  section.append(
    el('p', 'rp-lead', describeFlatness(r.flatness, r.daysCovered)),
    el(
      'p',
      'rp-note',
      'Two different things are counted here. Little joy is one; little emotion ' +
        'of any kind — a reading close to neutral whatever the subject — is the ' +
        'other, and it is the one a diary is unusually good at showing. ' +
        `A day counts as flat below ${FLAT_CHARGE_THRESHOLD.toFixed(2)} emotional ` +
        `charge, and as low-joy below ${LOW_JOY_THRESHOLD.toFixed(2)}. Both ` +
        'thresholds are judgement calls, not validated cut-offs.'
    )
  );

  const list = el('dl', 'rp-findings');
  addFinding(
    list,
    'Joy over time',
    r.flatness.joyPerWeek === null
      ? 'Not enough days recorded to fit a trend.'
      : Math.abs(r.flatness.joyPerWeek) < 0.01
        ? 'Broadly steady across the period.'
        : `${r.flatness.joyPerWeek > 0 ? 'Rising' : 'Falling'} by ${Math.abs(
            r.flatness.joyPerWeek * 100
          ).toFixed(1)} points per week.`
  );
  addFinding(
    list,
    'Emotional charge',
    `Averaged ${r.flatness.meanCharge.toFixed(2)} per recorded day, where 0 is ` +
      'entirely neutral and 1 is entirely non-neutral.'
  );
  section.append(list);

  return section;
}

/**
 * Recording behaviour over time.
 *
 * Kept strictly descriptive. Someone recording less often may be disengaging,
 * or may be busy, or may simply be better — and a document that guessed between
 * those would be inventing the most consequential sentence in it.
 */
function buildEngagement(r: ClinicalReport): HTMLElement {
  const section = sectionWith('Recording behaviour');
  if (r.engagement.weeks.length === 0) {
    section.append(el('p', '', 'No entries in this period.'));
    return section;
  }

  section.append(
    el('p', 'rp-lead', describeEngagement(r.engagement)),
    engagementChart(r.engagement),
    el(
      'p',
      'rp-note',
      'Bars are entries per week; weeks with none are shown as gaps rather than ' +
        'skipped. How often someone records is not a measure of how they are — ' +
        'people stop for every reason, including getting better — but a change ' +
        'in it is context for everything above.'
    )
  );
  return section;
}

/**
 * One section per matched lexicon category.
 *
 * Each is a list of dated quotations and nothing else. There is no count of
 * severity, no trend line, and no summary sentence interpreting them, because
 * every one of those would be an inference the matching cannot support. The
 * matched words are marked so a reader can see in a glance which hits are
 * idiom, negation, or someone else's story, and discard them.
 */
function buildLanguageSections(r: ClinicalReport): HTMLElement[] {
  return r.languageFlags.map((group) => buildLanguageSection(group));
}

function buildLanguageSection(group: LanguageFlagGroup): HTMLElement {
  const section = sectionWith(group.label);
  if (group.offersSupport) section.classList.add('rp-section-marked');

  const span =
    group.dayCount === 1
      ? `on one day (${formatDate(group.firstSeen)})`
      : `across ${group.dayCount} days, between ${formatDate(group.firstSeen)} and ${formatDate(
          group.lastSeen
        )}`;

  section.append(
    el(
      'p',
      'rp-lead',
      `Found in ${group.entryCount} ${group.entryCount === 1 ? 'entry' : 'entries'} ${span}.`
    ),
    el('p', 'rp-note', group.blurb)
  );

  for (const excerpt of group.excerpts) {
    section.append(markedQuote(excerpt.date, excerpt.emotion, excerpt.match));
  }

  if (group.omitted > 0) {
    section.append(
      el(
        'p',
        'rp-note',
        `${group.omitted} further ${
          group.omitted === 1 ? 'passage' : 'passages'
        } matched this category and are not shown here.`
      )
    );
  }

  return section;
}

function buildVoices(r: ClinicalReport): HTMLElement {
  const section = sectionWith('In their own words');
  if (r.notable.length === 0) {
    section.append(el('p', '', 'No transcribed entries long enough to quote.'));
    return section;
  }

  section.append(
    el(
      'p',
      'rp-lead',
      'The most emotionally marked moments, quoted verbatim and spread across the ' +
        'period rather than taken from whichever week was worst. Transcription is ' +
        'automatic and may contain errors.'
    )
  );

  for (const item of r.notable) {
    section.append(quoteBlock(item.date, item.emotion, item.quote));
  }

  if (r.coreMemories.length > 0) {
    section.append(
      el('h3', 'rp-subhead', 'Marked as significant by the author'),
      el(
        'p',
        'rp-note',
        'These were flagged by hand, not selected automatically — the author ' +
          'considered them formative.'
      )
    );
    for (const item of r.coreMemories) {
      const block = quoteBlock(item.date, item.emotion, item.quote);
      if (item.note) block.append(el('p', 'rp-quote-note', item.note));
      section.append(block);
    }
  }
  return section;
}

function buildThemes(r: ClinicalReport): HTMLElement {
  const section = sectionWith('Recurring themes');
  if (r.themes.length === 0) {
    section.append(
      el('p', '', 'No keyword recurred across two or more entries in this period.')
    );
    return section;
  }

  section.append(
    el(
      'p',
      'rp-lead',
      'Words and phrases appearing in two or more entries, with the emotion they ' +
        'were most often spoken in. Extracted automatically from the transcripts.'
    )
  );

  const table = el('table', 'rp-table');
  table.innerHTML =
    '<thead><tr><th>Theme</th><th>Entries</th><th>Usual emotion</th></tr></thead>';
  const body = el('tbody');
  for (const theme of r.themes) {
    const row = el('tr');
    const emotionCell = el('td');
    emotionCell.append(swatch(PALETTE[theme.emotion].base), document.createTextNode(
      ' ' + PALETTE[theme.emotion].label
    ));
    row.append(el('td', '', theme.text), el('td', '', String(theme.count)), emotionCell);
    body.append(row);
  }
  table.append(body);
  section.append(table);
  return section;
}

function buildProvenance(r: ClinicalReport): HTMLElement {
  const section = sectionWith('What this was measured from');

  section.append(
    el(
      'p',
      'rp-lead',
      'Each entry combines up to three channels. A channel that was unavailable ' +
        '— camera off, too little speech — contributed nothing rather than a ' +
        'neutral guess. Low availability means this summary rests mostly on the ' +
        'channels that remain.'
    )
  );

  const table = el('table', 'rp-table');
  table.innerHTML =
    '<thead><tr><th>Channel</th><th>Available</th><th>Mean confidence</th></tr></thead>';
  const body = el('tbody');
  const labels: Record<string, string> = {
    text: 'Transcribed speech',
    face: 'Facial expression',
    voice: 'Vocal tone',
  };
  for (const channel of r.channels) {
    const row = el('tr');
    row.append(
      el('td', '', labels[channel.name] ?? channel.name),
      el('td', '', pct(channel.availability)),
      el('td', '', pct(channel.meanCertainty))
    );
    body.append(row);
  }
  table.append(body);
  section.append(
    table,
    el(
      'p',
      'rp-note',
      `${pct(r.transcribedShare)} of entries produced a usable transcript.`
    )
  );
  return section;
}

function buildMethod(): HTMLElement {
  const section = sectionWith('Method and limitations');
  const list = el('ul', 'rp-list');

  for (const text of [
    'Emotions are classified into seven categories (neutral, joy, sadness, anger, ' +
      'fear, disgust, surprise) following the MELD dataset convention.',
    'Facial expression is scored in the browser; speech is transcribed locally by ' +
      'Whisper; text emotion comes from a RoBERTa model trained on MELD plus two ' +
      'general-domain corpora. No diary content is transmitted anywhere.',
    'The three channels are combined by weighting each according to its own ' +
      'confidence, so an uncertain channel contributes less than a confident one.',
    'Accuracy is uneven across categories. On held-out data the model is reliable ' +
      'for neutral and joy, moderate for anger, sadness and surprise, and weak for ' +
      'fear and disgust — those two are under-represented in the training data and ' +
      'their figures here should be treated with particular caution.',
    'Automatic emotion recognition is known to under-perform on retrospective, ' +
      'calm first-person speech of exactly the kind a diary contains. Low-arousal ' +
      'anger in particular is frequently misread as sadness.',
    'Entries are self-selected. The person chose when to record and what to say, ' +
      'so this reflects what they chose to express, not their emotional state ' +
      'generally.',
    'Percentages are weighted by how confident each reading was, so they will not ' +
      'match a simple count of entries.',
    'Trend lines are least-squares fits against elapsed days, reported only where ' +
      'at least seven days were recorded, and shown per emotion as well as in ' +
      'aggregate because opposite movements cancel in the aggregate.',
    'Sections quoting language about a subject are produced by matching a fixed ' +
      'list of words and phrases against the transcripts. Matching does not ' +
      'interpret: it cannot tell a statement from a denial, a memory, a joke, a ' +
      'song lyric, or an account of somebody else, which is why those sections ' +
      'print the sentence rather than a count or a score.',
    'That word list is not exhaustive and was not validated against anything. An ' +
      'absent section means no listed phrase was matched in a transcript — it is ' +
      'not evidence that a subject was absent from the author’s mind, and it ' +
      'should never be read as reassurance. Transcription errors alone are enough ' +
      'to lose a match.',
    'Nothing in this document estimates risk, and no section of it should be used ' +
      'in place of asking the person directly.',
  ]) {
    list.append(el('li', '', text));
  }

  section.append(list);
  section.append(
    el(
      'p',
      'rp-note',
      'Generated by Mindscape, a student research project in affective computing. ' +
        'Not a medical device.'
    )
  );
  return section;
}

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------

/**
 * Day-by-day chart: one bar per recorded day, above or below a centre line.
 *
 * Deliberately positioned by real date rather than by index, so gaps in
 * recording appear as gaps. Compressing them out would imply a continuity of
 * observation that isn't there.
 */
function valenceChart(days: DaySummary[]): SVGElement {
  const width = 720;
  const height = 150;
  const mid = height / 2;

  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('class', 'rp-chart');

  const start = days[0].date.getTime();
  const end = days[days.length - 1].date.getTime();
  const span = Math.max(1, end - start);
  const barWidth = Math.max(2, Math.min(14, (width / (span / 86_400_000 + 1)) * 0.7));

  const axis = document.createElementNS(SVG_NS, 'line');
  axis.setAttribute('x1', '0');
  axis.setAttribute('x2', String(width));
  axis.setAttribute('y1', String(mid));
  axis.setAttribute('y2', String(mid));
  axis.setAttribute('class', 'rp-axis');
  svg.append(axis);

  for (const day of days) {
    const x =
      span > 0 ? ((day.date.getTime() - start) / span) * (width - barWidth) : 0;
    // Height encodes emotional charge, direction encodes valence: a strongly
    // felt day is tall whichever way it points, which is the honest reading.
    const magnitude = Math.max(0.06, day.charge) * (mid - 8);
    const negative = NEGATIVE.includes(day.dominant);

    const bar = document.createElementNS(SVG_NS, 'rect');
    bar.setAttribute('x', String(x));
    bar.setAttribute('y', String(negative ? mid : mid - magnitude));
    bar.setAttribute('width', String(barWidth));
    bar.setAttribute('height', String(magnitude));
    bar.setAttribute('fill', PALETTE[day.dominant].base);
    bar.setAttribute('rx', '1.5');
    svg.append(bar);
  }

  for (const [label, y] of [
    ['more positive', 12],
    ['more negative', height - 4],
  ] as const) {
    const text = document.createElementNS(SVG_NS, 'text');
    text.setAttribute('x', '0');
    text.setAttribute('y', String(y));
    text.setAttribute('class', 'rp-chart-label');
    text.textContent = label;
    svg.append(text);
  }

  return svg;
}

/**
 * Entries per week, including the empty ones.
 *
 * Plotted against week index with every intervening week present, so a month of
 * silence is a month of empty slots rather than two adjacent bars.
 */
function engagementChart(engagement: EngagementSummary): SVGElement {
  const width = 720;
  const height = 96;
  const floor = height - 16;

  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('class', 'rp-chart');

  const weeks = engagement.weeks;
  const peak = Math.max(1, ...weeks.map((w) => w.entries));
  const slot = width / weeks.length;
  const barWidth = Math.max(2, Math.min(24, slot * 0.7));

  const axis = document.createElementNS(SVG_NS, 'line');
  axis.setAttribute('x1', '0');
  axis.setAttribute('x2', String(width));
  axis.setAttribute('y1', String(floor));
  axis.setAttribute('y2', String(floor));
  axis.setAttribute('class', 'rp-axis');
  svg.append(axis);

  weeks.forEach((week, i) => {
    if (week.entries === 0) return;
    const barHeight = (week.entries / peak) * (floor - 10);
    const bar = document.createElementNS(SVG_NS, 'rect');
    bar.setAttribute('x', String(i * slot + (slot - barWidth) / 2));
    bar.setAttribute('y', String(floor - barHeight));
    bar.setAttribute('width', String(barWidth));
    bar.setAttribute('height', String(barHeight));
    bar.setAttribute('fill', '#5c6270');
    bar.setAttribute('rx', '1.5');
    svg.append(bar);
  });

  // Only the ends are labelled: a tick per week is unreadable over a long
  // period, and the two dates are what place the shape in time.
  for (const [index, anchor] of [
    [0, 'start'],
    [weeks.length - 1, 'end'],
  ] as const) {
    const label = document.createElementNS(SVG_NS, 'text');
    label.setAttribute('x', String(index === 0 ? 0 : width));
    label.setAttribute('y', String(height - 2));
    label.setAttribute('text-anchor', anchor);
    label.setAttribute('class', 'rp-chart-label');
    label.textContent = formatDate(weeks[index].weekStart);
    svg.append(label);
  }

  const scale = document.createElementNS(SVG_NS, 'text');
  scale.setAttribute('x', '0');
  scale.setAttribute('y', '9');
  scale.setAttribute('class', 'rp-chart-label');
  scale.textContent = `peak ${peak} ${peak === 1 ? 'entry' : 'entries'}/week`;
  svg.append(scale);

  return svg;
}

function keywordRow(themes: Array<{ text: string; count: number }>): HTMLElement {
  const row = el('div', 'rp-chips');
  for (const theme of themes) {
    const chip = el('span', 'rp-chip');
    chip.append(
      document.createTextNode(theme.text),
      el('span', 'rp-chip-count', `×${theme.count}`)
    );
    row.append(chip);
  }
  return row;
}

function legendFor(days: DaySummary[]): HTMLElement {
  const present = new Set(days.map((d) => d.dominant));
  const legend = el('div', 'rp-legend');
  for (const emotion of EMOTIONS) {
    if (!present.has(emotion)) continue;
    const item = el('span', 'rp-legend-item');
    item.append(swatch(PALETTE[emotion].base), document.createTextNode(PALETTE[emotion].label));
    legend.append(item);
  }
  return legend;
}

function barTrack(share: number, color: string): HTMLElement {
  const track = el('span', 'rp-bar-track');
  const fill = el('span', 'rp-bar-fill');
  fill.style.width = `${Math.max(0.5, share * 100)}%`;
  fill.style.background = color;
  track.append(fill);
  return track;
}

// ---------------------------------------------------------------------------
// Small builders
// ---------------------------------------------------------------------------

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className = '',
  text = ''
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function sectionWith(title: string): HTMLElement {
  const section = el('section', 'rp-section');
  section.append(el('h2', 'rp-section-title', title));
  return section;
}

function keyValue(label: string, value: string): HTMLElement {
  const row = el('div', 'rp-kv');
  row.append(el('span', 'rp-kv-key', label), el('span', 'rp-kv-value', value));
  return row;
}

function stat(value: string, label: string): HTMLElement {
  const box = el('div', 'rp-stat');
  box.append(el('span', 'rp-stat-value', value), el('span', 'rp-stat-label', label));
  return box;
}

function addFinding(list: HTMLElement, term: string, detail: string): void {
  list.append(el('dt', '', term), el('dd', '', detail));
}

function quoteBlock(date: Date, emotion: Emotion, quote: string): HTMLElement {
  const block = el('blockquote', 'rp-quote');
  block.style.borderLeftColor = PALETTE[emotion].base;
  block.append(
    el('p', 'rp-quote-text', `“${quote.trim()}”`),
    el('p', 'rp-quote-meta', `${formatDate(date)} · read as ${PALETTE[emotion].label}`)
  );
  return block;
}

/**
 * A quotation with the matched term marked.
 *
 * Built from text nodes rather than innerHTML — this is the one place in the
 * report where the content is a raw transcript with an offset into it, and
 * assembling that as markup would put user text through an HTML parser for no
 * reason. The marking is what lets a reader dismiss a false positive without
 * reading the whole passage twice.
 */
function markedQuote(date: Date, emotion: Emotion, match: LexiconMatch): HTMLElement {
  const block = el('blockquote', 'rp-quote rp-quote-marked');
  block.style.borderLeftColor = PALETTE[emotion].base;

  const text = el('p', 'rp-quote-text');
  const { start, end } = match.highlight;
  const mark = el('mark', 'rp-mark', match.quote.slice(start, end));

  text.append(
    document.createTextNode('“' + match.quote.slice(0, start)),
    mark,
    document.createTextNode(match.quote.slice(end) + '”')
  );

  block.append(
    text,
    el(
      'p',
      'rp-quote-meta',
      `${formatDate(date)} · entry read as ${PALETTE[emotion].label} · matched “${match.term}”`
    )
  );
  return block;
}

/** "a, b and c" — used in the notice's pointer to the quoted sections. */
function listOf(items: string[]): string {
  if (items.length <= 1) return items[0] ?? '';
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

function swatch(color: string): HTMLElement {
  const dot = el('span', 'rp-swatch');
  dot.style.background = color;
  return dot;
}

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function signed(value: number): string {
  return `${value > 0 ? '+' : ''}${value.toFixed(2)}`;
}

function formatDate(date: Date): string {
  return date.toLocaleDateString(undefined, {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  });
}

function formatDateTime(date: Date): string {
  return `${formatDate(date)}, ${date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  })}`;
}

export type { EmotionVector };
