/**
 * Application entry point.
 *
 * Wiring order matters here: the world starts rendering before the camera or
 * the backend are ready, so the first thing you see is the world rather than a
 * loading screen. Capture attaches to it once permissions are granted.
 */

import './ui/styles.css';
import './ui/report.css';

import { checkHealth, warmup, type HealthResponse } from './api';
import { FaceCapture } from './capture/face';
import { mood } from './capture/mood';
import { DiarySession } from './capture/session';
import { type Emotion, mixedColor } from './emotions';
import { allEntries, loadWorld } from './state/db';
import { summarize } from './state/history';
import { Hud, describeLifetime } from './ui/hud';
import { Modals } from './ui/modals';
import { describePlace } from './world/biomes';
import { preloadMotifModels } from './world/models';
import { MindscapeWorld, type WorldView } from './world/scene';

/** How often to re-check whether the sidecar came up. */
const HEALTH_POLL_MS = 15_000;

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing element #${id}`);
  return found as T;
}

async function main(): Promise<void> {
  // -- world -----------------------------------------------------------

  // Resolves immediately unless someone has registered .glb models in
  // world/models.ts, so this costs nothing by default — but awaiting it here
  // means the island is built once, with the models, rather than built from
  // primitives and then visibly rebuilt a moment later.
  await preloadMotifModels();

  const viewButton = element<HTMLButtonElement>('btn-view');
  const placeCaption = element('place-caption');

  const world = new MindscapeWorld(element('world'), {
    onOrbPicked: (entryId) => void modals.showEntry(entryId),
    // The world owns the view state — it can start a flight of its own — so the
    // button and the caption follow it rather than each keeping their own copy.
    onViewChanged: (view: WorldView) => {
      viewButton.dataset.view = view;
      showPlaceCaption(view === 'mind');
    },
  });
  world.start();

  /**
   * Write the caption for whatever the mind floor is currently showing.
   *
   * Built from text nodes rather than innerHTML. The strings are ours — biome
   * blurbs and motif labels out of fixed tables — but the caption sits next to
   * a transcript of whatever somebody said out loud, and the habit of never
   * assembling markup from content is worth keeping even where today's content
   * happens to be safe.
   */
  function renderPlaceCaption(): void {
    const { headline, detail } = describePlace(world.getPlace(), world.getMotifs());
    placeCaption.replaceChildren();

    const lead = document.createElement('strong');
    lead.textContent = headline;
    placeCaption.append(lead);

    if (detail) placeCaption.append(document.createTextNode(` ${detail}`));
  }

  function showPlaceCaption(visible: boolean): void {
    placeCaption.hidden = !visible;
  }

  viewButton.addEventListener('click', () => world.toggleView());

  // -- hud -------------------------------------------------------------

  const recordButton = element<HTMLButtonElement>('btn-record');
  const hud = new Hud({
    emotionBars: element('emotion-bars'),
    faceIndicator: element('face-indicator'),
    backendIndicator: element('backend-indicator'),
    transcriptPanel: element('transcript-panel'),
    transcriptText: element('transcript-text'),
    recordButton,
    recordHint: element('record-hint'),
    brandDot: document.querySelector('.brand-dot') as HTMLElement,
    lifetimeSummary: element('lifetime-summary'),
    toast: element('toast'),
  });

  // The HUD reads the mood bus once per frame rather than subscribing, for the
  // reasons in hud.ts. This is the only place that drives it.
  const hudLoop = (): void => {
    hud.tick();
    requestAnimationFrame(hudLoop);
  };
  requestAnimationFrame(hudLoop);

  // -- modals ----------------------------------------------------------

  const modals = new Modals(
    {
      root: element('modal-root'),
      content: element('modal-content'),
      closeButton: document.querySelector('.modal-close') as HTMLElement,
      backdrop: document.querySelector('.modal-backdrop') as HTMLElement,
    },
    {
      onDiaryChanged: () => refreshWorld(),
      onToast: (message) => hud.toast(message),
    }
  );

  /** Rebuild the galaxy and the lifetime tint from what's in the database. */
  async function refreshWorld(): Promise<void> {
    const [entries, worldState] = await Promise.all([allEntries(), loadWorld()]);
    world.setEntries(entries);
    world.setLifetimeMood(worldState.lifetimeTotals);

    const summary = summarize(entries);

    // With no face, voice or words arriving, the world settles to this rather
    // than to neutral grey — so opening the app shows the world you have
    // actually built, and it is already coloured before the camera has produced
    // a single reading.
    if (entries.length > 0) mood.setResting(summary.vector);

    hud.setLifetime(
      entries.length > 0 ? mixedColor(summary.vector) : '#6f7793',
      describeLifetime(summary.vector, summary.totalEntries, summary.streakDays)
    );

    // The world rebuilds the landscape on the next frame (see scene.setEntries),
    // so the caption is written after it, on the frame the new ground appears.
    requestAnimationFrame(() => {
      renderPlaceCaption();
      showPlaceCaption(world.getView() === 'mind');
    });
  }

  await refreshWorld();

  // -- capture ---------------------------------------------------------

  const face = new FaceCapture(element<HTMLVideoElement>('camera'));

  const session = new DiarySession(face, {
    onPartialTranscript: (_chunk, full) => hud.setTranscript(full),
    onKeyword: (text, emotion, weight) => {
      world.spawnKeyword(text, emotion as Emotion, weight);
    },
    onStateChange: (state) => {
      const recording = state === 'recording';
      hud.setRecording(recording);
      world.setRecording(recording);
      hud.setBusy(state === 'processing');
    },
    onEntrySaved: async (entry) => {
      await refreshWorld();
      hud.toast(
        entry.peak
          ? `Saved — ${entry.peak.emotion}. "${truncate(entry.peak.text, 60)}"`
          : 'Entry saved to your world'
      );
      // Let the last words drift out on their own rather than cutting them.
      window.setTimeout(() => world.clearKeywords(), 6000);
    },
    onError: (message) => hud.toast(message, 5000),
  });

  recordButton.addEventListener('click', async () => {
    const state = session.getState();

    if (state === 'recording') {
      const entry = await session.stop();
      hud.setBusy(false);
      hud.setTranscript('');
      if (!entry) hud.toast('Nothing was captured — was the mic muted?', 4500);
      return;
    }

    if (state === 'processing' || state === 'starting') return;

    try {
      await session.start();
    } catch {
      hud.toast('Microphone permission is needed to record an entry', 5000);
    }
  });

  /** Which keys drive the camera, and which way. */
  const SPIN_KEYS: Record<string, number> = { KeyA: -1, KeyD: 1 };
  const ZOOM_KEYS: Record<string, number> = { KeyW: 1, KeyS: -1 };
  const held = new Set<string>();
  const applyKeys = (): void => {
    let spin = 0;
    let zoom = 0;
    for (const code of held) {
      spin += SPIN_KEYS[code] ?? 0;
      zoom += ZOOM_KEYS[code] ?? 0;
    }
    world.setSpin(spin);
    world.setZoom(zoom);
  };
  const isCameraKey = (code: string): boolean =>
    SPIN_KEYS[code] !== undefined || ZOOM_KEYS[code] !== undefined;

  // Space bar as a shortcut, since the button is the only control that matters.
  // C flips floors, so you can go down and look at a memory without breaking
  // off mid-sentence to find the mouse.
  document.addEventListener('keydown', (event) => {
    const target = event.target as HTMLElement | null;
    const typing =
      target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA';
    if (typing || !element('modal-root').hidden) return;

    if (event.code === 'Space') {
      event.preventDefault();
      recordButton.click();
    } else if (event.code === 'KeyC') {
      event.preventDefault();
      world.toggleView();
    } else if (isCameraKey(event.code)) {
      held.add(event.code);
      applyKeys();
    }
  });

  // A and D turn the world, W and S pull it closer and push it away, for as
  // long as they are held.
  //
  // Which keys are *down* rather than which was pressed last, because the two
  // come apart in ordinary use: hold A, press D without letting go, and a
  // last-press scheme has the world turning right with the left key still down.
  // Holding both cancels, which is the only sensible reading of both.
  //
  // Key-up is not guarded the way key-down is. A guard there would strand the
  // world spinning if you released the key after opening a modal or clicking
  // into a text field — the release has to be heard wherever it happens.
  document.addEventListener('keyup', (event) => {
    if (!held.delete(event.code)) return;
    applyKeys();
  });

  // Alt-tabbing away never delivers the key-up, so the world would still be
  // turning — or still zooming — when you came back.
  window.addEventListener('blur', () => {
    held.clear();
    applyKeys();
  });

  // -- start the camera ------------------------------------------------

  // Face capture is started eagerly so the world reacts to you before you press
  // anything. If permission is refused the app still works; it just loses the
  // live channel and falls back to the backend's reading at commit time.
  try {
    await face.start();
  } catch (error) {
    console.warn('Camera unavailable:', error);
    hud.toast('No camera — the world will respond to your voice and words only', 5500);
  }

  // -- backend health --------------------------------------------------

  let health: HealthResponse | null = null;

  let warmupRequested = false;

  async function poll(): Promise<void> {
    health = await checkHealth();

    // Kick the models awake the moment we first see the sidecar, so they're
    // ready before the user presses record rather than during their first
    // sentence.
    if (health && !warmupRequested) {
      warmupRequested = true;
      void warmup();
    }

    hud.setBackendStatus(
      health !== null,
      health
        ? health.warm
          ? health.device === 'cuda'
            ? 'gpu'
            : 'ready'
          : 'waking'
        : undefined
    );
  }

  await poll();
  window.setInterval(() => void poll(), HEALTH_POLL_MS);

  if (!health) {
    hud.toast(
      'Sidecar offline — running on the face channel only. Start the backend for transcript and keywords.',
      6000
    );
  }

  // -- nav -------------------------------------------------------------

  element('btn-history').addEventListener('click', () => void modals.showHistory());
  element('btn-memories').addEventListener('click', () => void modals.showCoreMemories());
  element('btn-diagnostics').addEventListener('click', () =>
    modals.showDiagnostics(health, face.getStatus() as unknown as Record<string, unknown>)
  );

  // Stopping the camera on unload avoids leaving the webcam light on if the tab
  // is closed mid-session.
  window.addEventListener('beforeunload', () => {
    face.dispose();
    world.stop();
  });

  // Expose a little for console debugging during development.
  if (import.meta.env.DEV) {
    const THREE = await import('three');
    Object.assign(window, {
      mindscape: { world, session, mood, face, refreshWorld, THREE },
    });
  }
}

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

main().catch((error) => {
  console.error(error);
  document.body.innerHTML = `
    <div style="display:grid;place-items:center;height:100vh;font-family:system-ui;color:#a8b0c8;text-align:center;padding:24px;">
      <div>
        <h1 style="color:#f2f4fb;font-size:18px;">Mindscape failed to start</h1>
        <p style="font-size:13px;max-width:44ch;line-height:1.6;">${String(error)}</p>
        <p style="font-size:12px;color:#6f7793;">Check the browser console for the full trace.</p>
      </div>
    </div>`;
});
