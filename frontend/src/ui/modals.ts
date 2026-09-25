/**
 * The overlay views: history dots, core memories, and model diagnostics.
 *
 * All three render into one modal shell. They're built with direct DOM calls
 * rather than a framework — the app has exactly three views and no shared
 * client state worth reconciling, so a renderer would be more machinery than
 * the problem needs.
 */

import {
  EMOTIONS,
  PALETTE,
  conicGradient,
  type Emotion,
  type EmotionVector,
} from '../emotions';
import type { HealthResponse } from '../api';
import { type Period, hasSupportLanguage } from '../state/report';
import { generateReport } from './report';
import {
  type Bucket,
  type Granularity,
  fillGaps,
  groupEntries,
  summarize,
} from '../state/history';
import {
  type DiaryEntry,
  allEntries,
  deleteEntry,
  exportAll,
  setCoreMemory,
  wipe,
} from '../state/db';

export interface ModalHost {
  root: HTMLElement;
  content: HTMLElement;
  closeButton: HTMLElement;
  backdrop: HTMLElement;
}

export interface ModalCallbacks {
  /** Fired when the diary changes, so the world can rebuild. */
  onDiaryChanged: () => void | Promise<void>;
  onToast: (message: string) => void;
}

export class Modals {
  private granularity: Granularity = 'day';
  private escapeHandler: ((e: KeyboardEvent) => void) | null = null;

  constructor(
    private host: ModalHost,
    private callbacks: ModalCallbacks
  ) {
    host.closeButton.addEventListener('click', () => this.close());
    host.backdrop.addEventListener('click', () => this.close());
  }

  private open(): void {
    this.host.root.hidden = false;
    if (!this.escapeHandler) {
      this.escapeHandler = (e: KeyboardEvent) => {
        if (e.key === 'Escape') this.close();
      };
      document.addEventListener('keydown', this.escapeHandler);
    }
  }

  close(): void {
    this.host.root.hidden = true;
    this.host.content.replaceChildren();
    if (this.escapeHandler) {
      document.removeEventListener('keydown', this.escapeHandler);
      this.escapeHandler = null;
    }
  }

  // -- history ---------------------------------------------------------

  async showHistory(): Promise<void> {
    const entries = await allEntries();
    this.open();
    this.renderHistory(entries);
  }

  private renderHistory(entries: DiaryEntry[]): void {
    const content = this.host.content;
    content.replaceChildren();

    content.append(heading('Your year so far', 'Each dot is one slice of time, split by how it felt.'));

    if (entries.length === 0) {
      content.append(emptyState('Nothing recorded yet. Talk about your day and the first dot appears here.'));
      return;
    }

    const summary = summarize(entries);
    content.append(
      paragraph(
        `${summary.totalEntries} entries · ${summary.totalMinutes} minutes · ` +
          `${summary.streakDays}-day streak`,
        'subtitle'
      )
    );

    content.append(this.buildExportRow(entries));

    // Granularity switcher
    const switcher = document.createElement('div');
    switcher.className = 'granularity';
    for (const level of ['day', 'week', 'month'] as Granularity[]) {
      const button = document.createElement('button');
      button.textContent = level[0].toUpperCase() + level.slice(1);
      button.setAttribute('aria-pressed', String(level === this.granularity));
      button.addEventListener('click', () => {
        this.granularity = level;
        this.renderHistory(entries);
      });
      switcher.appendChild(button);
    }
    content.append(switcher);

    // Dots
    const buckets = groupEntries(entries, this.granularity);
    const padded = fillGaps(buckets, this.granularity);

    const label = document.createElement('p');
    label.className = 'dot-label';
    label.textContent = `${padded[0]?.label ?? ''} — ${padded[padded.length - 1]?.label ?? ''}`;
    content.append(label);

    const grid = document.createElement('div');
    grid.className = 'dot-grid';

    for (const bucket of padded) {
      const dot = document.createElement('button');
      dot.className = 'dot';

      if ('empty' in bucket) {
        dot.classList.add('is-empty');
        dot.title = `${bucket.label} — nothing recorded`;
        dot.disabled = true;
      } else {
        dot.style.background = conicGradient(bucket.vector);
        dot.title = describeBucket(bucket);
        if (bucket.coreMemoryCount > 0) dot.classList.add('has-core');
        dot.addEventListener('click', () => this.renderBucketDetail(bucket, entries));
      }
      grid.appendChild(dot);
    }
    content.append(grid);

    // Legend — doubles as the summary of the whole period on screen.
    content.append(buildLegend(summary.vector));

    // Most recent entries below the strip, so the view is useful without
    // requiring a click.
    const recent = entries.slice(-6).reverse();
    content.append(subheading('Recent'));
    for (const entry of recent) {
      content.append(this.buildEntryRow(entry));
    }
  }

  private renderBucketDetail(bucket: Bucket, allEntriesList: DiaryEntry[]): void {
    const content = this.host.content;
    content.replaceChildren();

    const back = document.createElement('button');
    back.className = 'ghost';
    back.textContent = '← Back';
    back.style.marginBottom = '16px';
    back.addEventListener('click', () => this.renderHistory(allEntriesList));
    content.append(back);

    content.append(heading(bucket.label, describeBucket(bucket)));

    const bigDot = document.createElement('div');
    bigDot.className = 'dot';
    bigDot.style.cssText = `width:84px;height:84px;margin:0 0 14px;background:${conicGradient(
      bucket.vector
    )}`;
    content.append(bigDot, buildLegend(bucket.vector));

    if (bucket.topKeywords.length > 0) {
      const words = document.createElement('div');
      words.className = 'entry-keywords';
      words.style.marginBottom = '18px';
      for (const keyword of bucket.topKeywords) {
        words.append(keywordChip(keyword.text, keyword.emotion as Emotion));
      }
      content.append(words);
    }

    for (const entry of bucket.entries) {
      content.append(this.buildEntryRow(entry));
    }
  }

  // -- core memories ---------------------------------------------------

  async showCoreMemories(): Promise<void> {
    const entries = await allEntries();
    const cores = entries.filter((e) => e.isCoreMemory);
    this.open();

    const content = this.host.content;
    content.replaceChildren();
    content.append(
      heading(
        'Core memories',
        'The moments you decided were formative. They sit at the centre of your world.'
      )
    );

    if (cores.length === 0) {
      content.append(
        emptyState(
          'None yet. Open any entry and press "Make core memory" to lift it into the ring at the centre of the world.'
        )
      );
      // Offer the strongest candidates rather than leaving a dead end.
      const candidates = entries
        .filter((e) => e.peak)
        .sort((a, b) => (b.peak?.score ?? 0) - (a.peak?.score ?? 0))
        .slice(0, 3);
      if (candidates.length > 0) {
        content.append(subheading('Strongest moments so far'));
        for (const entry of candidates) content.append(this.buildEntryRow(entry));
      }
      return;
    }

    for (const entry of cores) content.append(this.buildEntryRow(entry));
  }

  // -- one entry -------------------------------------------------------

  async showEntry(entryId: string): Promise<void> {
    const entries = await allEntries();
    const entry = entries.find((e) => e.id === entryId);
    if (!entry) return;

    this.open();
    const content = this.host.content;
    content.replaceChildren();

    const when = new Date(entry.createdAt);
    content.append(
      heading(
        when.toLocaleDateString(undefined, {
          weekday: 'long',
          month: 'long',
          day: 'numeric',
        }),
        `${when.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })} · ` +
          `${Math.round(entry.durationSeconds)}s · ${PALETTE[entry.dominant].label}`
      )
    );

    const bigDot = document.createElement('div');
    bigDot.className = 'dot';
    bigDot.style.cssText = `width:72px;height:72px;margin:0 0 18px;background:${conicGradient(
      entry.vector
    )}`;
    content.append(bigDot);

    if (entry.blend) content.append(buildBlendRow(entry.blend));

    if (entry.peak) {
      const quote = document.createElement('blockquote');
      quote.style.cssText = `margin:0 0 18px;padding-left:14px;border-left:3px solid ${
        PALETTE[entry.peak.emotion].base
      };font-size:15px;line-height:1.55;`;
      quote.textContent = `“${entry.peak.text}”`;
      content.append(quote);
    }

    if (entry.transcript) {
      content.append(subheading('What you said'));
      content.append(paragraph(entry.transcript, 'entry-quote'));
    }

    if (entry.keywords.length > 0) {
      content.append(subheading('Keywords'));
      const words = document.createElement('div');
      words.className = 'entry-keywords';
      for (const keyword of entry.keywords) {
        words.append(keywordChip(keyword.text, keyword.emotion as Emotion));
      }
      content.append(words);
    }

    content.append(subheading('How each channel read it'));
    content.append(buildModalityGrid(entry));

    content.append(this.buildEntryActions(entry, () => void this.showEntry(entryId)));
  }

  // -- diagnostics -----------------------------------------------------

  showDiagnostics(health: HealthResponse | null, faceStatus: Record<string, unknown>): void {
    this.open();
    const content = this.host.content;
    content.replaceChildren();

    content.append(
      heading(
        'Signals',
        'What each channel is running, and how much it is contributing.'
      )
    );

    const grid = document.createElement('div');
    grid.className = 'diag-grid';

    grid.append(
      diagCard('Face (in browser)', {
        Model: 'face-api tiny + expression',
        Ready: String(faceStatus.ready ?? false),
        'Face visible': String(faceStatus.faceVisible ?? false),
        Inference: `${Math.round(Number(faceStatus.lastInferenceMs ?? 0))} ms`,
      })
    );

    if (health) {
      const models = health.models as Record<string, any>;
      grid.append(
        diagCard('Text (MELD)', {
          Source: models?.text?.source ?? 'not loaded',
          Loaded: String(models?.text?.loaded ?? false),
          Device: health.device,
        }),
        diagCard('Speech', {
          Model: models?.asr?.model ?? '—',
          Loaded: String(models?.asr?.loaded ?? false),
        }),
        diagCard('Fusion weights', {
          Text: String(health.fusion_weights.text),
          Face: String(health.fusion_weights.face),
          Voice: String(health.fusion_weights.voice),
        })
      );
    } else {
      grid.append(
        diagCard('Sidecar', {
          Status: 'unreachable',
          Effect: 'face channel only',
        })
      );
    }

    content.append(grid);

    if (!health) {
      content.append(
        paragraph(
          'Start it with: uvicorn app.main:app --reload --port 8000 (from the backend directory).',
          'subtitle'
        )
      );
    }

    content.append(this.buildResetRow());
  }

  /**
   * Wipe the whole diary and start from an empty world.
   *
   * It lives in Signals rather than in the history view on purpose: this is a
   * testing tool, not a feature of the diary. Everything is local and there is
   * no undo, so it names the count before it asks and points at the JSON export
   * for anyone who wanted to keep the entries.
   */
  private buildResetRow(): HTMLElement {
    const row = document.createElement('div');
    row.className = 'diag-reset';

    row.append(subheading('Start over'));
    row.append(
      paragraph(
        'Erases every entry and the accumulated world — orbs, landscape and ' +
          'lifetime mood — leaving the app as it was on first launch. Permanent, ' +
          'and not backed up anywhere: export the raw JSON from History first if ' +
          'you want it back.',
        'subtitle'
      )
    );

    const button = document.createElement('button');
    button.className = 'ghost is-danger';
    button.textContent = 'Erase everything';
    button.addEventListener('click', async () => {
      const entries = await allEntries();
      const count = entries.length;
      if (count === 0) {
        this.callbacks.onToast('Nothing to erase — the world is already empty');
        return;
      }
      const noun = count === 1 ? 'entry' : 'entries';
      if (
        !window.confirm(
          `Erase all ${count} ${noun} and reset the world?\n\n` +
            'Every orb disappears and this cannot be undone.'
        )
      ) {
        return;
      }
      await wipe();
      await this.callbacks.onDiaryChanged();
      this.callbacks.onToast(`Erased ${count} ${noun} — the world is empty again`);
      this.close();
    });

    row.append(button);
    return row;
  }

  // -- shared pieces ---------------------------------------------------

  private buildEntryRow(entry: DiaryEntry): HTMLElement {
    const row = document.createElement('article');
    row.className = 'entry';

    const swatch = document.createElement('span');
    swatch.className = 'entry-swatch';
    swatch.style.background = PALETTE[entry.dominant].base;
    swatch.style.color = PALETTE[entry.dominant].base;

    const body = document.createElement('div');
    body.className = 'entry-body';

    const title = document.createElement('h3');
    // A blended entry is named by both emotions: calling a bittersweet memory
    // simply "Joy" is the exact information loss this whole feature exists to
    // undo.
    title.textContent = entry.blend?.isBlend
      ? entry.blend.components
          .slice(0, 2)
          .map((c) => PALETTE[c.emotion].label)
          .join(' + ')
      : PALETTE[entry.dominant].label;
    if (entry.isCoreMemory) title.textContent += ' · core memory';

    const when = document.createElement('time');
    const date = new Date(entry.createdAt);
    when.dateTime = entry.createdAt;
    when.textContent = date.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });

    body.append(title, when);

    const excerpt = entry.peak?.text ?? entry.transcript;
    if (excerpt) {
      body.append(
        paragraph(
          excerpt.length > 180 ? `${excerpt.slice(0, 180)}…` : excerpt,
          'entry-quote'
        )
      );
    }

    if (entry.keywords.length > 0) {
      const words = document.createElement('div');
      words.className = 'entry-keywords';
      for (const keyword of entry.keywords.slice(0, 5)) {
        words.append(keywordChip(keyword.text, keyword.emotion as Emotion));
      }
      body.append(words);
    }

    const actions = document.createElement('div');
    actions.className = 'entry-actions';
    const open = document.createElement('button');
    open.textContent = 'Open';
    open.addEventListener('click', () => void this.showEntry(entry.id));
    actions.append(open);

    row.append(swatch, body, actions);
    return row;
  }

  private buildEntryActions(entry: DiaryEntry, refresh: () => void): HTMLElement {
    const actions = document.createElement('div');
    actions.style.cssText = 'display:flex;gap:8px;margin-top:22px;flex-wrap:wrap;';

    const core = document.createElement('button');
    core.className = entry.isCoreMemory ? 'ghost is-core' : 'ghost';
    core.textContent = entry.isCoreMemory ? '★ Core memory' : 'Make core memory';
    core.addEventListener('click', async () => {
      await setCoreMemory(entry.id, !entry.isCoreMemory);
      await this.callbacks.onDiaryChanged();
      this.callbacks.onToast(
        entry.isCoreMemory
          ? 'Returned to the outer world'
          : 'Lifted into the core memory ring'
      );
      refresh();
    });

    const remove = document.createElement('button');
    remove.className = 'ghost';
    remove.textContent = 'Delete';
    remove.addEventListener('click', async () => {
      // Deleting a memory is irreversible and this is someone's diary, so it
      // asks — the one place in the app that interrupts you.
      if (!window.confirm('Delete this entry? Its orb disappears from the world.')) {
        return;
      }
      await deleteEntry(entry.id);
      await this.callbacks.onDiaryChanged();
      this.callbacks.onToast('Entry deleted');
      void this.showHistory();
    });

    actions.append(core, remove);
    return actions;
  }

  private buildExportRow(entries: DiaryEntry[]): HTMLElement {
    const row = document.createElement('div');
    row.style.cssText =
      'display:flex;gap:8px;margin:14px 0 20px;padding-bottom:18px;border-bottom:1px solid rgba(255,255,255,0.07);flex-wrap:wrap;';

    const report = document.createElement('button');
    report.className = 'ghost';
    report.textContent = 'Summary for a doctor (PDF)';
    report.addEventListener('click', () => this.renderReportForm(entries));

    const json = document.createElement('button');
    json.className = 'ghost';
    json.textContent = 'Raw data (JSON)';
    json.title = 'Everything, unprocessed — for backup or moving to another device';
    json.addEventListener('click', async () => {
      const data = await exportAll();
      const url = URL.createObjectURL(new Blob([data], { type: 'application/json' }));
      const link = document.createElement('a');
      link.href = url;
      link.download = `mindscape-${new Date().toISOString().slice(0, 10)}.json`;
      link.click();
      URL.revokeObjectURL(url);
      this.callbacks.onToast('Exported');
    });

    row.append(report, json);
    return row;
  }

  /**
   * Asks for a name and a period before generating the report.
   *
   * The name is asked rather than stored: this document is meant to be handed
   * to someone, and a diary that quietly knew your legal name would be a
   * different and more sensitive thing than one that doesn't. It is used for
   * this render and never written to the database.
   */
  private renderReportForm(entries: DiaryEntry[]): void {
    const content = this.host.content;
    content.replaceChildren();

    const back = document.createElement('button');
    back.className = 'ghost';
    back.textContent = '← Back';
    back.style.marginBottom = '16px';
    back.addEventListener('click', () => this.renderHistory(entries));
    content.append(back);

    content.append(
      heading(
        'Summary for a doctor or psychologist',
        'A readable summary of your entries — patterns over time, recurring themes, ' +
          'and your own words. Generated on this device and never uploaded.'
      )
    );

    if (hasSupportLanguage(entries)) content.append(buildSupportNotice());

    const field = document.createElement('label');
    field.style.cssText = 'display:block;margin:0 0 18px;';
    field.append(
      Object.assign(document.createElement('span'), {
        textContent: 'Name to put on the report',
        style: 'display:block;font-size:12px;color:#a8b0c8;margin-bottom:6px;',
      })
    );

    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'Your name';
    input.autocomplete = 'name';
    input.style.cssText =
      'width:100%;max-width:340px;padding:9px 12px;border-radius:10px;' +
      'background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.14);' +
      'color:#f2f4fb;font-family:inherit;font-size:14px;';
    field.append(input);
    content.append(field);

    const periodLabel = document.createElement('p');
    periodLabel.style.cssText = 'font-size:12px;color:#a8b0c8;margin:0 0 8px;';
    periodLabel.textContent = 'Period to cover';
    content.append(periodLabel);

    let period: Period = 'all';
    const switcher = document.createElement('div');
    switcher.className = 'granularity';
    switcher.style.marginBottom = '22px';
    const options: Array<[Period, string]> = [
      ['all', 'Everything'],
      ['90d', 'Last 90 days'],
      ['30d', 'Last 30 days'],
    ];
    for (const [value, label] of options) {
      const button = document.createElement('button');
      button.textContent = label;
      button.setAttribute('aria-pressed', String(value === period));
      button.addEventListener('click', () => {
        period = value;
        for (const sibling of switcher.children) {
          sibling.setAttribute('aria-pressed', String(sibling === button));
        }
      });
      switcher.append(button);
    }
    content.append(switcher);

    const generate = document.createElement('button');
    generate.className = 'ghost';
    generate.style.cssText =
      'background:rgba(255,255,255,0.16);color:#f2f4fb;padding:10px 18px;font-size:13px;';
    generate.textContent = 'Create PDF';
    const run = () => {
      // The modal must be closed first: it sits above the report in the DOM and
      // the print stylesheet hides siblings, not ancestors' overlays.
      this.close();
      generateReport(entries, input.value, period);
    };
    generate.addEventListener('click', run);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') run();
    });

    content.append(
      generate,
      paragraph(
        'Your browser will open its print dialog — choose "Save as PDF" as the ' +
          'destination. The file will be named after you, like ' +
          '"Ada_Lovelace_Diary.pdf". Nothing is sent anywhere.',
        'subtitle'
      )
    );

    input.focus();
  }
}

/**
 * Shown on the export screen when the diary contains language about self-harm
 * or hopelessness.
 *
 * Two jobs, and the first is the one that is easy to forget. This document is
 * about to be handed to another person, and it will quote those passages back —
 * so the person exporting it should know that before they print it, not after
 * someone else has read it. The print preview lets them see exactly what it
 * says and stop if they want to.
 *
 * The second job is the resources. They are offered rather than insisted on,
 * without a diagnosis attached and without implying the app has concluded
 * anything: it matched some words, which is all it can do, and someone who is
 * fine loses two seconds reading this.
 */
function buildSupportNotice(): HTMLElement {
  const box = document.createElement('section');
  box.style.cssText =
    'margin:0 0 20px;padding:14px 16px;border-radius:12px;' +
    'background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.14);';

  const title = document.createElement('p');
  title.style.cssText =
    'margin:0 0 8px;font-size:13px;font-weight:600;color:#f2f4fb;';
  title.textContent = 'Before you share this';
  box.append(title);

  for (const text of [
    'Some of your entries use words about self-harm or hopelessness. The report ' +
      'includes those passages, quoted as you said them, so that whoever reads ' +
      'it sees your words rather than a score. Nothing is rated or scored.',
    'The print preview shows the whole document before you save it — worth a ' +
      'look if you want to know exactly what it says first.',
    'If any of it is true right now, these people are there for it:',
  ]) {
    const line = document.createElement('p');
    line.style.cssText =
      'margin:0 0 8px;font-size:12.5px;line-height:1.5;color:#c6cce0;';
    line.textContent = text;
    box.append(line);
  }

  const list = document.createElement('ul');
  list.style.cssText =
    'margin:0;padding-left:18px;font-size:12.5px;line-height:1.6;color:#c6cce0;';
  for (const text of [
    'Finland — Kriisipuhelin (MIELI), 09 2525 0111',
    'Emergency, anywhere in the EU — 112',
    'Anywhere else — findahelpline.com lists services by country',
  ]) {
    const item = document.createElement('li');
    item.textContent = text;
    list.append(item);
  }
  box.append(list);

  return box;
}

// ---------------------------------------------------------------------------
// Small DOM helpers
// ---------------------------------------------------------------------------

function heading(title: string, subtitle?: string): DocumentFragment {
  const fragment = document.createDocumentFragment();
  const h = document.createElement('h2');
  h.textContent = title;
  fragment.append(h);
  if (subtitle) fragment.append(paragraph(subtitle, 'subtitle'));
  return fragment;
}

function subheading(text: string): HTMLElement {
  const h = document.createElement('h3');
  h.textContent = text;
  h.style.cssText =
    'margin:22px 0 8px;font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:#6f7793;';
  return h;
}

function paragraph(text: string, className?: string): HTMLElement {
  const p = document.createElement('p');
  p.textContent = text;
  if (className) p.className = className;
  return p;
}

function emptyState(text: string): HTMLElement {
  return paragraph(text, 'empty-state');
}

function keywordChip(text: string, emotion: Emotion): HTMLElement {
  const chip = document.createElement('span');
  chip.className = 'kw';
  chip.textContent = text;
  const palette = PALETTE[emotion] ?? PALETTE.neutral;
  chip.style.color = palette.glow;
  // Tinted background at low alpha so the chip reads as its emotion without
  // fighting the text for contrast.
  chip.style.background = `${palette.base}26`;
  return chip;
}

function diagCard(title: string, rows: Record<string, string>): HTMLElement {
  const card = document.createElement('div');
  card.className = 'diag-card';

  const h = document.createElement('h4');
  h.textContent = title;
  card.append(h);

  const dl = document.createElement('dl');
  for (const [key, value] of Object.entries(rows)) {
    const dt = document.createElement('dt');
    dt.textContent = key;
    const dd = document.createElement('dd');
    dd.textContent = value;
    dl.append(dt, dd);
  }
  card.append(dl);
  return card;
}

/**
 * Renders the blend as a proportional bar plus a plain-language line.
 *
 * The distinction that matters here is *believed* vs *unclear*: two emotions the
 * channels corroborated is a real thing that happened, while two emotions they
 * disagreed about is the app not knowing. Showing both as "mixed" would collapse
 * exactly the difference the fusion layer works to preserve.
 */
function buildBlendRow(blend: NonNullable<DiaryEntry['blend']>): HTMLElement {
  const wrap = document.createElement('div');
  wrap.style.cssText = 'margin:0 0 20px;';

  const bar = document.createElement('div');
  bar.style.cssText =
    'display:flex;height:7px;border-radius:4px;overflow:hidden;margin-bottom:8px;';
  for (const part of blend.components) {
    const seg = document.createElement('span');
    seg.style.cssText = `width:${(part.share * 100).toFixed(1)}%;background:${
      PALETTE[part.emotion].base
    };`;
    seg.title = `${PALETTE[part.emotion].label} ${Math.round(part.share * 100)}%`;
    bar.append(seg);
  }
  wrap.append(bar);

  const caption = document.createElement('p');
  caption.style.cssText = 'margin:0;font-size:12px;color:#a8b0c8;';
  if (blend.isBlend) {
    const names = blend.components
      .slice(0, 2)
      .map((c) => PALETTE[c.emotion].label)
      .join(' and ');
    caption.textContent =
      `${names} together — all channels agreed, so this was felt as a blend, ` +
      `not a toss-up.`;
  } else if (blend.components.length > 1) {
    caption.textContent =
      'Mixed signals — the channels disagreed, so this reading is uncertain ' +
      'rather than genuinely blended.';
  } else {
    caption.textContent = `${PALETTE[blend.components[0].emotion].label}, on its own.`;
  }
  wrap.append(caption);
  return wrap;
}

function buildModalityGrid(entry: DiaryEntry): HTMLElement {
  const grid = document.createElement('div');
  grid.className = 'diag-grid';

  for (const modality of entry.modalities ?? []) {
    const rows: Record<string, string> = {
      Reading: modality.dominant ?? '—',
      Certainty: `${Math.round(modality.certainty * 100)}%`,
      Weight: `${Math.round(modality.weight * 100)}%`,
    };
    if (modality.source) rows.Source = modality.source;
    grid.append(diagCard(modality.name, rows));
  }
  return grid;
}

/**
 * Legend for the dot strip, ordered by how much of the period each emotion
 * actually accounts for.
 *
 * Sorted largest-first and with absent emotions dropped, so it doubles as the
 * summary of the period rather than being a static key. A fixed list in array
 * order gives every emotion equal billing and still names Anger, Fear and
 * Disgust on a week that contained none of them — which reads as though they
 * were there.
 */
function buildLegend(vector: EmotionVector, floor = 0.02): HTMLElement {
  const legend = document.createElement('div');
  legend.style.cssText =
    'display:flex;flex-wrap:wrap;gap:12px;margin:4px 0 8px;font-size:11px;color:#a8b0c8;';

  const present = EMOTIONS.map((emotion) => ({ emotion, share: vector[emotion] }))
    .filter((entry) => entry.share >= floor)
    .sort((a, b) => b.share - a.share);

  // Everything below the floor, rolled into one trailing note rather than
  // listed — it keeps the total honest without seven near-zero rows.
  const remainder = 1 - present.reduce((sum, e) => sum + e.share, 0);

  for (const { emotion, share } of present) {
    const item = document.createElement('span');
    item.style.cssText = 'display:flex;align-items:center;gap:5px;';

    const swatch = document.createElement('span');
    swatch.style.cssText = `width:9px;height:9px;border-radius:50%;background:${PALETTE[emotion].base};`;

    const label = document.createElement('span');
    label.textContent = PALETTE[emotion].label;

    const value = document.createElement('span');
    value.style.cssText = 'color:#6f7793;font-variant-numeric:tabular-nums;';
    value.textContent = `${Math.round(share * 100)}%`;

    item.append(swatch, label, value);
    legend.append(item);
  }

  if (remainder >= 0.01) {
    const rest = document.createElement('span');
    rest.style.color = '#6f7793';
    rest.textContent = `+${Math.round(remainder * 100)}% other`;
    legend.append(rest);
  }
  return legend;
}

function describeBucket(bucket: Bucket): string {
  const parts = [
    `${bucket.entryCount} ${bucket.entryCount === 1 ? 'entry' : 'entries'}`,
    `mostly ${PALETTE[bucket.dominant].label}`,
    `${Math.round(bucket.totalSeconds / 60)} min`,
  ];
  if (bucket.coreMemoryCount > 0) parts.push(`${bucket.coreMemoryCount} core`);
  return parts.join(' · ');
}

