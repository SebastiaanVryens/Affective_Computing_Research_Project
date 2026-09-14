/**
 * A recording session, from "Talk about your day" to a saved entry.
 *
 * Coordinates the two timescales:
 *
 *   live   — face at 8 Hz and audio level at 60 Hz go straight into the mood
 *            bus, which the world reads every frame. No network involved.
 *   streamed — ~4s audio chunks go to the sidecar; transcript, text emotion and
 *            keywords come back a beat later and are folded into the same bus.
 *   commit — on stop, the full recording is analysed once more (better context
 *            for Whisper, and the voice SER model sees the whole utterance) and
 *            the result becomes a permanent DiaryEntry.
 *
 * The streamed pass is best-effort. If it's slow or the backend is down, the
 * session still works — you just get the final reading instead of a live one.
 */

import {
  type AnalyzeResponse,
  analyze,
  isOnline,
  textVectorOf,
} from '../api';
import {
  type Emotion,
  type EmotionVector,
  dominant,
  fromArray,
  intensity,
  normalize,
} from '../emotions';
import {
  type DiaryEntry,
  accumulate,
  dayKey,
  loadWorld,
  saveEntry,
  saveWorld,
} from '../state/db';
import { spiralPlacement } from '../world/placement';
import { FaceCapture } from './face';
import { MicCapture } from './mic';
import { mood } from './mood';

/**
 * How much audio each streamed chunk covers.
 *
 * This is the dominant term in how long it takes words to appear: a chunk can't
 * be transcribed until it has been recorded, so the floor is chunk length plus
 * processing. Measured warm, a chunk processes at ~0.17x realtime, so 3s of
 * audio costs ~0.5s to transcribe and words land roughly 3.5s after they're
 * spoken.
 *
 * Shorter would feel snappier still, but Whisper was trained on 30s windows and
 * degrades noticeably on very short ones — and every restart of the live
 * recorder drops a few milliseconds at the seam. 3s keeps the preview readable.
 * None of this affects the saved entry, which is re-transcribed from the
 * gapless continuous recording.
 */
const CHUNK_MS = 3000;

/**
 * Streamed chunks are dropped rather than queued if one is still in flight.
 * On a slow CPU, queueing would build an ever-growing backlog of stale audio
 * and the "live" keywords would fall further behind with every chunk.
 */
const CHUNK_TIMEOUT_MS = 12_000;

export interface SessionEvents {
  onPartialTranscript?: (text: string, full: string) => void;
  onKeyword?: (text: string, emotion: Emotion, weight: number) => void;
  onReading?: (vector: EmotionVector, source: 'chunk' | 'final') => void;
  onStateChange?: (state: SessionState) => void;
  onEntrySaved?: (entry: DiaryEntry) => void;
  onError?: (message: string) => void;
}

export type SessionState =
  | 'idle'
  | 'starting'
  | 'recording'
  | 'processing'
  | 'saved'
  | 'error';

export class DiarySession {
  private mic: MicCapture | null = null;
  private state: SessionState = 'idle';
  private startedAt: Date | null = null;

  private chunkInFlight = false;
  private transcriptParts: string[] = [];
  private lastFinal: AnalyzeResponse | null = null;

  constructor(
    private face: FaceCapture,
    private events: SessionEvents = {}
  ) {}

  getState(): SessionState {
    return this.state;
  }

  getLastResult(): AnalyzeResponse | null {
    return this.lastFinal;
  }

  /** Live transcript assembled from the streamed chunks. */
  getTranscript(): string {
    return this.transcriptParts.join(' ').trim();
  }

  async start(): Promise<void> {
    if (this.state === 'recording' || this.state === 'starting') return;

    this.setState('starting');
    this.transcriptParts = [];
    this.lastFinal = null;
    this.face.resetSession();
    mood.resetUtterance();

    try {
      this.mic = new MicCapture({
        chunkMs: CHUNK_MS,
        onLevel: ({ level, speaking }) => mood.pushArousal(level, speaking),
        onChunk: (clip) => void this.handleChunk(clip),
      });
      await this.mic.start();

      // Start the face clock with the mic, so face timestamps and audio
      // timestamps share an origin and Whisper's segment times line up.
      this.face.markSessionStart();
      this.startedAt = new Date();
      this.setState('recording');
    } catch (error) {
      this.setState('error');
      this.events.onError?.(
        error instanceof Error ? error.message : 'Could not start the microphone'
      );
      throw error;
    }
  }

  /**
   * Handle one streamed chunk.
   *
   * Deliberately does *not* update the diary or the world's permanent state —
   * chunks only steer the live mood and spawn keywords. The record is written
   * once, at commit.
   */
  private async handleChunk(clip: Blob): Promise<void> {
    if (this.chunkInFlight || !isOnline()) return;
    this.chunkInFlight = true;

    try {
      const result = await analyze(clip, this.face.sessionVector(), {
        timeoutMs: CHUNK_TIMEOUT_MS,
      });
      if (!result || this.state !== 'recording') return;

      const text = result.transcript.text.trim();
      if (text) {
        this.transcriptParts.push(text);
        this.events.onPartialTranscript?.(text, this.getTranscript());
      }

      const textVector = textVectorOf(result);
      if (textVector) {
        mood.pushText(textVector);
        this.events.onReading?.(textVector, 'chunk');
      }

      const voice = result.reading.modalities.find((m) => m.name === 'voice');
      if (voice?.available) mood.pushVoice(fromArray(voice.vector));

      for (const keyword of result.keywords.slice(0, 4)) {
        this.events.onKeyword?.(
          keyword.text,
          keyword.emotion as Emotion,
          Math.min(1, keyword.score)
        );
      }
    } catch (error) {
      console.warn('chunk analysis failed', error);
    } finally {
      this.chunkInFlight = false;
    }
  }

  /**
   * Finish the session and write the entry.
   *
   * Returns the saved entry, or null if there was nothing worth saving (no
   * audio captured, or the backend produced no transcript and no face was seen).
   */
  async stop(): Promise<DiaryEntry | null> {
    if (this.state !== 'recording' || !this.mic || !this.startedAt) return null;

    this.setState('processing');
    const recording = await this.mic.stop();
    const durationSeconds = this.mic.elapsedMs() / 1000;
    this.mic = null;

    const faceVector = this.face.sessionVector();

    // The final pass re-analyses the whole recording rather than stitching the
    // chunk results: Whisper is markedly better with full context, and the
    // voice model sees the entire utterance instead of 4-second slices.
    const result = await analyze(recording, faceVector, {
      timeoutMs: 180_000,
      faceTimeline: this.face.sessionTimeline(),
    });
    this.lastFinal = result;

    const entry = await this.buildEntry(result, faceVector, durationSeconds);
    if (!entry) {
      this.setState('idle');
      return null;
    }

    await saveEntry(entry);

    const world = accumulate(await loadWorld(), entry);
    await saveWorld(world);

    this.setState('saved');
    this.events.onEntrySaved?.(entry);
    return entry;
  }

  /** Abandon a session without writing anything. */
  async cancel(): Promise<void> {
    if (this.mic) {
      await this.mic.stop();
      this.mic = null;
    }
    this.transcriptParts = [];
    mood.resetUtterance();
    this.setState('idle');
  }

  private async buildEntry(
    result: AnalyzeResponse | null,
    faceVector: EmotionVector | null,
    durationSeconds: number
  ): Promise<DiaryEntry | null> {
    const createdAt = this.startedAt ?? new Date();

    // Backend-down path: the face channel alone still makes a valid, honest
    // entry. It just has no words in it.
    if (!result) {
      if (!faceVector) return null;
      const world = await loadWorld();
      return {
        id: makeId(),
        createdAt: createdAt.toISOString(),
        day: dayKey(createdAt),
        durationSeconds,
        transcript: this.getTranscript(),
        vector: faceVector,
        dominant: dominant(faceVector),
        sentiment: 'unknown',
        certainty: intensity(faceVector),
        keywords: [],
        modalities: [
          {
            name: 'face',
            vector: Object.values(faceVector),
            dominant: dominant(faceVector),
            certainty: intensity(faceVector),
            weight: 1,
            available: true,
            source: 'face-api-browser',
          },
        ],
        peak: null,
        isCoreMemory: false,
        worldPosition: spiralPlacement(world.entryCount),
      };
    }

    const vector = normalize(fromArray(result.reading.vector));
    const transcript = result.transcript.text.trim() || this.getTranscript();

    // Nothing said, nobody seen — don't litter the world with an empty orb.
    if (!transcript && !faceVector) return null;

    const world = await loadWorld();
    this.events.onReading?.(vector, 'final');

    return {
      id: makeId(),
      createdAt: createdAt.toISOString(),
      day: dayKey(createdAt),
      durationSeconds: result.audio.durationSeconds || durationSeconds,
      transcript,
      vector,
      dominant: result.reading.dominant as Emotion,
      sentiment: result.reading.sentiment,
      certainty: result.reading.certainty,
      keywords: result.keywords,
      modalities: result.reading.modalities,
      peak: result.peak
        ? {
            text: result.peak.text,
            emotion: result.peak.emotion as Emotion,
            score: result.peak.score,
          }
        : null,
      blend: result.reading.blend
        ? {
            charge: result.reading.blend.charge,
            complexity: result.reading.blend.complexity,
            components: result.reading.blend.components.map((c) => ({
              emotion: c.emotion as Emotion,
              share: c.share,
            })),
            isBlend: result.reading.blend.isBlend,
            label: result.reading.blend.label,
          }
        : undefined,
      isCoreMemory: false,
      worldPosition: spiralPlacement(world.entryCount),
    };
  }

  private setState(state: SessionState): void {
    this.state = state;
    this.events.onStateChange?.(state);
  }
}

function makeId(): string {
  return `entry_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}
