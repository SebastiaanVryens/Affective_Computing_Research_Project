/**
 * Client for the Python sidecar.
 *
 * Every call degrades rather than throws: if the backend is down, the app keeps
 * running on the face channel alone. That's a real mode, not a courtesy — the
 * live face loop is entirely in-browser, so the world still responds to you
 * with no server at all. You just don't get transcript or keywords.
 */

import { type EmotionVector, fromArray, toArray } from './emotions';

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000';

export interface Keyword {
  text: string;
  score: number;
  count: number;
  emotion: string;
  emotionVector: number[];
}

export interface ModalityReading {
  name: 'text' | 'face' | 'voice';
  vector: number[];
  dominant: string | null;
  certainty: number;
  weight: number;
  available: boolean;
  source: string | null;
}

export interface BlendProfile {
  /** Non-neutral mass: how much was happening, independent of how many. */
  charge: number;
  /** Effective number of emotions in play, 1 = single, 2 = two at once. */
  complexity: number;
  components: Array<{ emotion: string; share: number }>;
  /** True only when 2+ emotions AND the channels corroborated the spread. */
  isBlend: boolean;
  supported: boolean;
  label: string;
}

export interface Reading {
  vector: number[];
  emotions: Record<string, number>;
  dominant: string;
  sentiment: string;
  certainty: number;
  blend: BlendProfile | null;
  modalities: ModalityReading[];
}

export interface AnalyzeResponse {
  transcript: {
    text: string;
    segments: Array<{ start: number; end: number; text: string }>;
    language: string;
    confidence: number;
  };
  reading: Reading;
  keywords: Keyword[];
  keywordsByEmotion: Record<string, string[]>;
  sentences: Array<{ text: string; vector: number[] }>;
  peak: { text: string; vector: number[]; emotion: string; score: number } | null;
  audio: { durationSeconds: number; decoded: boolean };
  timing: { totalMs: number };
}

export interface HealthResponse {
  status: string;
  device: string;
  emotions: string[];
  models: Record<string, unknown>;
  /** True once the models are loaded and the first chunk will be fast. */
  warm: boolean;
  fusion_weights: Record<string, number>;
}

let online = false;

export function isOnline(): boolean {
  return online;
}

export async function checkHealth(): Promise<HealthResponse | null> {
  try {
    const response = await fetch(`${BASE_URL}/api/health`, {
      signal: AbortSignal.timeout(2500),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    online = true;
    return (await response.json()) as HealthResponse;
  } catch {
    online = false;
    return null;
  }
}

/**
 * Ask the sidecar to load its models now rather than on the first chunk.
 *
 * Fire-and-forget: the backend returns immediately and loads on a background
 * thread. Without this the first ~19s of a recording is swallowed by the cold
 * start and live transcription appears broken.
 */
export async function warmup(): Promise<void> {
  try {
    await fetch(`${BASE_URL}/api/warmup`, {
      method: 'POST',
      signal: AbortSignal.timeout(3000),
    });
  } catch {
    // Warmup is an optimisation; models still load lazily if this never lands.
  }
}

/**
 * Send a clip for full analysis.
 *
 * `timeoutMs` is generous for the final commit (Whisper on CPU over a two-minute
 * entry is not quick) but short for streaming chunks, where a late answer is
 * worse than no answer — we'd be colouring the world with a stale utterance.
 */
export async function analyze(
  clip: Blob,
  faceVector: EmotionVector | null,
  options: {
    timeoutMs?: number;
    /** Timestamped face readings, so the backend can align them to speech. */
    faceTimeline?: Array<{ t: number; v: number[] }>;
  } = {}
): Promise<AnalyzeResponse | null> {
  const form = new FormData();
  form.append('audio', clip, 'entry.webm');
  if (faceVector) {
    form.append('face_vector', JSON.stringify(toArray(faceVector)));
  }
  if (options.faceTimeline?.length) {
    form.append('face_timeline', JSON.stringify(options.faceTimeline));
  }

  try {
    const response = await fetch(`${BASE_URL}/api/analyze`, {
      method: 'POST',
      body: form,
      signal: AbortSignal.timeout(options.timeoutMs ?? 120_000),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    online = true;
    return (await response.json()) as AnalyzeResponse;
  } catch (error) {
    if ((error as Error)?.name !== 'TimeoutError') online = false;
    console.warn('analyze failed:', error);
    return null;
  }
}

export async function analyzeText(
  text: string,
  faceVector: EmotionVector | null
): Promise<Pick<AnalyzeResponse, 'reading' | 'keywords' | 'keywordsByEmotion' | 'peak'> | null> {
  try {
    const response = await fetch(`${BASE_URL}/api/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        faceVector: faceVector ? toArray(faceVector) : null,
      }),
      signal: AbortSignal.timeout(30_000),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    online = true;
    return await response.json();
  } catch (error) {
    console.warn('analyzeText failed:', error);
    return null;
  }
}

/** Convenience: pull the text channel out of a response as a keyed vector. */
export function textVectorOf(response: AnalyzeResponse): EmotionVector | null {
  const text = response.reading.modalities.find((m) => m.name === 'text');
  return text?.available ? fromArray(text.vector) : null;
}
