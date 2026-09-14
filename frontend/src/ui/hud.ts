/**
 * The always-on overlay: live emotion bars, status pills, transcript, record
 * button.
 *
 * The emotion bars are driven from the render loop rather than from mood-bus
 * events. The bus fires on every face reading (8 Hz) and every text update, and
 * pushing DOM writes at that rate from a callback causes layout thrash that's
 * visible as jitter in the 3D scene. Reading once per frame and writing only
 * what changed is both smoother and cheaper.
 */

import { EMOTIONS, PALETTE, type Emotion, type EmotionVector } from '../emotions';
import { mood } from '../capture/mood';

export interface HudElements {
  emotionBars: HTMLElement;
  faceIndicator: HTMLElement;
  backendIndicator: HTMLElement;
  transcriptPanel: HTMLElement;
  transcriptText: HTMLElement;
  recordButton: HTMLButtonElement;
  recordLabel: HTMLElement;
  recordHint: HTMLElement;
  brandDot: HTMLElement;
  lifetimeSummary: HTMLElement;
  toast: HTMLElement;
}

interface BarRow {
  row: HTMLLIElement;
  fill: HTMLElement;
  value: HTMLElement;
  lastWidth: number;
}

export class Hud {
  private rows = new Map<Emotion, BarRow>();
  private lastDominant: Emotion | null = null;
  private toastTimer = 0;
  private transcript = '';
  private recording = false;

  constructor(private el: HudElements) {
    this.buildBars();
  }

  private buildBars(): void {
    // Fixed order, never resorted — see styles.css. The list stays in the
    // canonical MELD order so the eye can find a given emotion by position.
    for (const emotion of EMOTIONS) {
      const row = document.createElement('li');
      row.className = 'emotion-row';

      const name = document.createElement('span');
      name.className = 'emotion-name';
      name.textContent = PALETTE[emotion].label;

      const track = document.createElement('span');
      track.className = 'emotion-track';

      const fill = document.createElement('span');
      fill.className = 'emotion-fill';
      fill.style.background = PALETTE[emotion].base;
      fill.style.color = PALETTE[emotion].base; // drives the box-shadow glow
      fill.style.width = '0%';
      track.appendChild(fill);

      const value = document.createElement('span');
      value.className = 'emotion-value';
      value.textContent = '0%';

      row.append(name, track, value);
      this.el.emotionBars.appendChild(row);
      this.rows.set(emotion, { row, fill, value, lastWidth: -1 });
    }
  }

  /** Last mood tint written to CSS, so we only touch the DOM when it changes. */
  private lastTint = '';

  /** Called once per animation frame from main.ts. */
  tick(): void {
    const snapshot = mood.current();
    this.tintGlass(snapshot.color, snapshot.clarity);

    for (const emotion of EMOTIONS) {
      const bar = this.rows.get(emotion)!;
      const percent = Math.round(snapshot.vector[emotion] * 100);
      // Skip the write when the rounded value hasn't moved — most frames.
      if (percent !== bar.lastWidth) {
        bar.fill.style.width = `${percent}%`;
        bar.value.textContent = `${percent}%`;
        bar.lastWidth = percent;
      }
    }

    if (snapshot.dominant !== this.lastDominant) {
      if (this.lastDominant) {
        this.rows.get(this.lastDominant)!.row.classList.remove('is-dominant');
      }
      this.rows.get(snapshot.dominant)!.row.classList.add('is-dominant');
      this.lastDominant = snapshot.dominant;
    }

    this.setPill(this.el.faceIndicator, snapshot.faceVisible, 'face', 'no face');
  }

  /**
   * Bleeds a little of the current mood into the glass panels.
   *
   * Kept very subtle — the panels are chrome, not content, and a strongly
   * tinted UI competes with the world for attention. It's just enough that the
   * interface feels part of the same room as the sky rather than pasted on top.
   *
   * Written as a CSS custom property rather than per-element styles so one
   * assignment repaints every panel, and only when the rounded value actually
   * moves. At 60fps an unconditional write here would be 60 style
   * recalculations a second for a change nobody can see.
   */
  private tintGlass(color: string, clarity: number): void {
    const n = parseInt(color.slice(1), 16);
    const r = (n >> 16) & 255;
    const g = (n >> 8) & 255;
    const b = n & 255;
    // Alpha rides on clarity: a confident mood tints the UI a little more
    // strongly, an uncertain one leaves it neutral grey.
    const alpha = (0.05 + 0.07 * clarity).toFixed(3);
    const tint = `rgba(${r}, ${g}, ${b}, ${alpha})`;

    if (tint === this.lastTint) return;
    this.lastTint = tint;
    document.documentElement.style.setProperty('--mood-tint', tint);
  }

  setBackendStatus(online: boolean, detail?: string): void {
    this.setPill(
      this.el.backendIndicator,
      online,
      detail ?? 'listening',
      'offline'
    );
    this.el.backendIndicator.title = online
      ? 'Sidecar reachable — transcript and keywords enabled'
      : 'Sidecar unreachable — running on the face channel only';
  }

  private setPill(
    element: HTMLElement,
    on: boolean,
    onText: string,
    offText: string
  ): void {
    const next = on ? onText : offText;
    if (element.textContent !== next) element.textContent = next;
    element.classList.toggle('pill-on', on);
    element.classList.toggle('pill-off', !on);
  }

  setTranscript(text: string): void {
    this.transcript = text;
    this.renderTranscript();
  }

  /**
   * Shows a placeholder while recording but before the first words land.
   *
   * There is an unavoidable few-second gap at the start: speech has to be
   * recorded, sent, and transcribed before anything can appear. An empty panel
   * during that window reads as "this isn't working" rather than as "this is
   * listening", which is exactly the wrong impression to give someone who has
   * just started talking about their day.
   */
  private renderTranscript(): void {
    const waiting = this.recording && !this.transcript;
    this.el.transcriptPanel.hidden = !this.recording && !this.transcript;
    this.el.transcriptText.textContent = waiting ? 'Listening…' : this.transcript;
    this.el.transcriptText.classList.toggle('is-waiting', waiting);
    // Keep the newest words in view under the fade mask.
    this.el.transcriptText.scrollTop = this.el.transcriptText.scrollHeight;
  }

  setRecording(active: boolean): void {
    this.recording = active;
    this.renderTranscript();
    this.el.recordButton.classList.toggle('is-recording', active);
    this.el.recordLabel.textContent = active ? 'Finish entry' : 'Talk about your day';
    this.el.recordHint.textContent = active
      ? 'Take your time. Everything is processed on this machine.'
      : 'Your camera and voice stay on this machine.';
  }

  setBusy(busy: boolean, label = 'Making sense of it…'): void {
    this.el.recordButton.disabled = busy;
    if (busy) this.el.recordLabel.textContent = label;
  }

  /** Lifetime colour on the brand dot, plus the one-line summary. */
  setLifetime(color: string, summary: string): void {
    this.el.brandDot.style.background = color;
    this.el.brandDot.style.color = color;
    this.el.lifetimeSummary.textContent = summary;
  }

  toast(message: string, ms = 3200): void {
    this.el.toast.textContent = message;
    this.el.toast.hidden = false;
    window.clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => {
      this.el.toast.hidden = true;
    }, ms);
  }
}

/** Human-readable one-liner for the header, e.g. "mostly Joy · 12 entries". */
export function describeLifetime(
  vector: EmotionVector,
  entryCount: number,
  streakDays: number
): string {
  if (entryCount === 0) return 'a world built from how you felt';

  const ranked = EMOTIONS.map((e) => ({ e, p: vector[e] })).sort((a, b) => b.p - a.p);
  const top = ranked[0];
  const second = ranked[1];

  // When the top two are close, saying "mostly X" overstates it.
  const lead =
    top.p - second.p < 0.08
      ? `${PALETTE[top.e].label} and ${PALETTE[second.e].label}`
      : `mostly ${PALETTE[top.e].label}`;

  const parts = [lead, `${entryCount} ${entryCount === 1 ? 'entry' : 'entries'}`];
  if (streakDays > 1) parts.push(`${streakDays}-day streak`);
  return parts.join(' · ');
}
