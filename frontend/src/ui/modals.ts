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

    content.append(this.buildExportRow());
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

  private buildExportRow(): HTMLElement {
    const row = document.createElement('div');
    row.style.cssText =
      'display:flex;gap:8px;margin-top:26px;padding-top:18px;border-top:1px solid rgba(255,255,255,0.07);';

    const button = document.createElement('button');
    button.className = 'ghost';
    button.textContent = 'Export everything (JSON)';
    button.addEventListener('click', async () => {
      const json = await exportAll();
      const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
      const link = document.createElement('a');
      link.href = url;
      link.download = `mindscape-${new Date().toISOString().slice(0, 10)}.json`;
      link.click();
      URL.revokeObjectURL(url);
      this.callbacks.onToast('Exported');
    });

    row.append(button);
    return row;
  }
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

