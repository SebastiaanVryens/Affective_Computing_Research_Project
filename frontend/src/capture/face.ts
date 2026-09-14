/**
 * Live facial expression, entirely in-browser.
 *
 * face-api's expression head emits exactly MELD's seven classes (under slightly
 * different names), which is the reason this project can fuse face and text
 * without inventing a mapping. The video frames never leave the page.
 *
 * Runs on its own clock rather than in the render loop: inference costs ~15-30ms
 * on CPU and we don't want that on the critical path for a 60fps scene.
 */

import * as faceapi from '@vladmandic/face-api';
import {
  type Emotion,
  type EmotionVector,
  normalize,
  toArray,
  zeroVector,
} from '../emotions';
import { mood } from './mood';

/** face-api's expression label -> ours. Everything maps; nothing is dropped. */
const LABEL_MAP: Record<string, Emotion> = {
  neutral: 'neutral',
  happy: 'joy',
  sad: 'sadness',
  angry: 'anger',
  fearful: 'fear',
  disgusted: 'disgust',
  surprised: 'surprise',
};

const MODEL_URL = '/models';

/**
 * Target inference rate. 8 Hz is comfortably above the rate at which a face
 * actually changes, and leaves the main thread free for the 3D scene.
 */
const TARGET_HZ = 8;

/**
 * face-api reports a detection score per face; below this it's usually a
 * hand, a lamp, or a poster on the wall behind you.
 */
const MIN_DETECTION_SCORE = 0.35;

export interface FaceStatus {
  ready: boolean;
  running: boolean;
  faceVisible: boolean;
  /** Measured inference time, surfaced in the diagnostics panel. */
  lastInferenceMs: number;
  error?: string;
}

export class FaceCapture {
  private video: HTMLVideoElement;
  private stream: MediaStream | null = null;
  private options: faceapi.TinyFaceDetectorOptions | null = null;
  private timer: number | null = null;
  private busy = false;
  private disposed = false;

  private status: FaceStatus = {
    ready: false,
    running: false,
    faceVisible: false,
    lastInferenceMs: 0,
  };

  /** Accumulates the session's face readings, for the entry that gets saved. */
  private sessionTotal = zeroVector();
  private sessionFrames = 0;

  /**
   * Timestamped readings for the whole session, so the backend can line the
   * face up with what was being said at that moment rather than averaging a
   * two-minute entry into one mood. Timestamps are seconds since the session
   * clock started, matching the audio Whisper receives.
   */
  private timeline: Array<{ t: number; v: EmotionVector }> = [];
  private clockStart = 0;

  constructor(video: HTMLVideoElement) {
    this.video = video;
  }

  async load(): Promise<void> {
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL),
      faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
    ]);
    this.options = new faceapi.TinyFaceDetectorOptions({
      inputSize: 224, // multiple of 32; smallest that still finds a face reliably
      scoreThreshold: MIN_DETECTION_SCORE,
    });
    this.status.ready = true;
  }

  async start(): Promise<void> {
    if (!this.status.ready) await this.load();

    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false, // the mic is opened separately, with its own constraints
    });
    this.video.srcObject = this.stream;
    await this.video.play();

    this.status.running = true;
    this.scheduleNext();
  }

  stop(): void {
    this.status.running = false;
    if (this.timer !== null) {
      window.clearTimeout(this.timer);
      this.timer = null;
    }
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  dispose(): void {
    this.disposed = true;
    this.stop();
  }

  private scheduleNext(): void {
    if (!this.status.running || this.disposed) return;
    this.timer = window.setTimeout(() => void this.tick(), 1000 / TARGET_HZ);
  }

  private async tick(): Promise<void> {
    // If the previous inference is still running (slow machine, busy tab),
    // skip this slot rather than queueing up work we can never catch up on.
    if (this.busy || !this.options || this.video.readyState < 2) {
      this.scheduleNext();
      return;
    }

    this.busy = true;
    const started = performance.now();
    try {
      const detection = await faceapi
        .detectSingleFace(this.video, this.options)
        .withFaceLandmarks(true) // `true` selects the tiny landmark net
        .withFaceExpressions();

      this.status.lastInferenceMs = performance.now() - started;

      if (detection && detection.detection.score >= MIN_DETECTION_SCORE) {
        const vector = toEmotionVector(detection.expressions);
        this.status.faceVisible = true;
        mood.pushFace(vector);
        this.accumulate(vector);
      } else {
        this.status.faceVisible = false;
        mood.faceLost();
      }
    } catch (error) {
      this.status.error = error instanceof Error ? error.message : String(error);
    } finally {
      this.busy = false;
      this.scheduleNext();
    }
  }

  private accumulate(vector: EmotionVector): void {
    for (const key of Object.keys(this.sessionTotal) as Emotion[]) {
      this.sessionTotal[key] += vector[key];
    }
    this.sessionFrames++;

    if (this.clockStart > 0) {
      this.timeline.push({
        t: (performance.now() - this.clockStart) / 1000,
        v: { ...vector },
      });
    }
  }

  /** Mean face reading across the session, or null if we never saw a face. */
  sessionVector(): EmotionVector | null {
    if (this.sessionFrames === 0) return null;
    return normalize(this.sessionTotal);
  }

  /**
   * Timestamped readings, downsampled for transport.
   *
   * At 8 Hz a long entry would be thousands of samples; the backend only aligns
   * these to Whisper segments a few seconds wide, so sub-second resolution is
   * wasted bytes. Bucketed to `bucketSeconds` and averaged within each bucket.
   */
  sessionTimeline(bucketSeconds = 1): Array<{ t: number; v: number[] }> {
    if (this.timeline.length === 0) return [];

    const buckets = new Map<number, { total: EmotionVector; n: number }>();
    for (const sample of this.timeline) {
      const key = Math.floor(sample.t / bucketSeconds);
      let bucket = buckets.get(key);
      if (!bucket) {
        bucket = { total: zeroVector(), n: 0 };
        buckets.set(key, bucket);
      }
      for (const emotion of Object.keys(bucket.total) as Emotion[]) {
        bucket.total[emotion] += sample.v[emotion];
      }
      bucket.n++;
    }

    return [...buckets.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([key, bucket]) => ({
        // Bucket midpoint, so a reading isn't biased toward its window's start.
        t: (key + 0.5) * bucketSeconds,
        v: toArray(normalize(bucket.total)),
      }));
  }

  /** Begin the timeline clock. Called when recording actually starts. */
  markSessionStart(): void {
    this.clockStart = performance.now();
    this.timeline = [];
  }

  resetSession(): void {
    this.sessionTotal = zeroVector();
    this.sessionFrames = 0;
    this.timeline = [];
    this.clockStart = 0;
  }

  getStatus(): FaceStatus {
    return { ...this.status };
  }
}

function toEmotionVector(expressions: faceapi.FaceExpressions): EmotionVector {
  const vector = zeroVector();
  for (const [label, target] of Object.entries(LABEL_MAP)) {
    const value = (expressions as unknown as Record<string, number>)[label];
    if (typeof value === 'number') vector[target] += value;
  }
  return normalize(vector);
}
