/**
 * Microphone capture with two jobs running off one stream.
 *
 * 1. A per-frame analyser that produces the live arousal signal. This is what
 *    makes the world react to your voice with no perceptible delay — it's just
 *    RMS and spectral centroid off a Web Audio AnalyserNode, no model involved.
 * 2. A *second* MediaRecorder that produces short standalone clips for the
 *    backend, so transcript and keywords stream in while you're still talking.
 *
 * The two-recorder split is the interesting part, and it exists because
 * MediaRecorder timeslices cannot be transcribed independently. Only the first
 * blob of a timesliced recording carries the WebM header; the rest are raw
 * continuation fragments that no decoder will open on their own.
 *
 * The tempting fix — prepend the first blob to each later one — is wrong, and
 * was the original bug here. That first blob contains the header *and its four
 * seconds of audio*, so every chunk decoded as "the opening of the entry, then
 * this chunk", and the live transcript stuttered the same opening words back
 * over and over.
 *
 * Extracting just the header bytes would work but means parsing EBML to find the
 * first Cluster. Running a second recorder in stop/start cycles is simpler and
 * cannot drift: each cycle yields a complete, self-contained clip. It drops a
 * few milliseconds at each restart, which is fine — these clips only drive the
 * live preview. The continuous recorder above is gapless and is what the saved
 * entry is transcribed from.
 */

export interface AudioLevel {
  /** Smoothed RMS mapped to [0,1]. */
  level: number;
  /** Brightness proxy in [0,1] — rises with tense/raised voice. */
  brightness: number;
  /** Above the adaptive noise floor. */
  speaking: boolean;
}

export interface MicOptions {
  /** How often to hand a chunk to the backend, in ms. */
  chunkMs?: number;
  onLevel?: (level: AudioLevel) => void;
  onChunk?: (clip: Blob, elapsedMs: number) => void;
}

const FFT_SIZE = 1024;

/**
 * Noise floor adapts to the room. Starting pessimistically high and letting it
 * fall avoids a burst of false "speaking" in the first second before we've
 * heard what silence sounds like here.
 */
const INITIAL_NOISE_FLOOR = 0.02;
const FLOOR_ADAPT_RATE = 0.002;
const SPEAKING_MARGIN = 2.2;

export class MicCapture {
  private stream: MediaStream | null = null;
  private context: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private recorder: MediaRecorder | null = null;
  private rafId: number | null = null;

  // Explicit ArrayBuffer type argument: the Web Audio getters reject the
  // default Float32Array<ArrayBufferLike>, which could be SharedArrayBuffer.
  private timeBuffer: Float32Array<ArrayBuffer> = new Float32Array(FFT_SIZE);
  private freqBuffer: Uint8Array<ArrayBuffer> = new Uint8Array(FFT_SIZE / 2);

  private smoothedLevel = 0;
  private noiseFloor = INITIAL_NOISE_FLOOR;

  /**
   * How loudly this session was actually spoken, accumulated per frame.
   *
   * This is the one thing the audio can report that cannot be wrong. The SER
   * model *guesses* an emotion from tone and, measured, guesses badly; loudness
   * is simply measured. It also carries something the words cannot: "I'm fine"
   * said quietly and "I'm fine" said loudly are the same transcript.
   *
   * Only frames above the noise floor count, so a long thinking pause does not
   * quietly average the entry down towards silence.
   */
  private vocalSum = 0;
  private vocalPeak = 0;
  private brightnessSum = 0;
  private vocalFrames = 0;

  /** Data from the continuous recorder, assembled on stop. */
  private chunks: Blob[] = [];
  private startedAt = 0;

  /** The short-lived recorder producing standalone clips for live preview. */
  private liveRecorder: MediaRecorder | null = null;
  private liveTimer: number | null = null;
  private liveRunning = false;
  private mimeType = '';

  constructor(private options: MicOptions = {}) {}

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        // Browser DSP helps ASR but flattens exactly the dynamics the prosody
        // features read. Noise suppression is the worst offender, so it goes;
        // echo cancellation stays because speaker bleed is worse.
        echoCancellation: true,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    this.context = new AudioContext();
    const source = this.context.createMediaStreamSource(this.stream);
    this.analyser = this.context.createAnalyser();
    this.analyser.fftSize = FFT_SIZE;
    this.analyser.smoothingTimeConstant = 0.6;
    source.connect(this.analyser);
    // Note: analyser is intentionally not connected to destination — routing it
    // there would play the user's own voice back at them through the speakers.

    this.timeBuffer = new Float32Array(this.analyser.fftSize);
    this.freqBuffer = new Uint8Array(this.analyser.frequencyBinCount);

    this.mimeType = pickMimeType();
    this.chunks = [];
    this.startedAt = performance.now();
    // Per-session, so a second entry never inherits the first one's loudness.
    this.vocalSum = 0;
    this.vocalPeak = 0;
    this.brightnessSum = 0;
    this.vocalFrames = 0;

    // Continuous recorder — the authoritative recording. No timeslice, so it
    // hands everything over in one seamless piece when stopped.
    this.recorder = new MediaRecorder(
      this.stream,
      this.mimeType ? { mimeType: this.mimeType } : undefined
    );
    this.recorder.ondataavailable = (event) => {
      if (event.data.size > 0) this.chunks.push(event.data);
    };
    this.recorder.start();

    this.liveRunning = true;
    this.startLiveCycle();
    this.loop();
  }

  /**
   * One cycle of the live recorder: record for chunkMs, stop, hand over a
   * complete clip, start again.
   *
   * Restarting from `onstop` rather than on a repeating interval means the next
   * cycle begins only once the previous has actually finished, so cycles can
   * never overlap or pile up on a slow machine.
   */
  private startLiveCycle(): void {
    if (!this.liveRunning || !this.stream) return;

    const recorder = new MediaRecorder(
      this.stream,
      this.mimeType ? { mimeType: this.mimeType } : undefined
    );
    const parts: Blob[] = [];

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) parts.push(event.data);
    };
    recorder.onstop = () => {
      if (parts.length > 0) {
        // A complete little recording, header and all — decodable on its own,
        // containing only its own few seconds of audio.
        this.options.onChunk?.(
          new Blob(parts, { type: this.mimeType || 'audio/webm' }),
          this.elapsedMs()
        );
      }
      this.startLiveCycle();
    };

    recorder.start();
    this.liveRecorder = recorder;
    this.liveTimer = window.setTimeout(() => {
      if (recorder.state !== 'inactive') recorder.stop();
    }, this.options.chunkMs ?? 4000);
  }

  private stopLiveCycle(): void {
    this.liveRunning = false;
    if (this.liveTimer !== null) {
      window.clearTimeout(this.liveTimer);
      this.liveTimer = null;
    }
    if (this.liveRecorder && this.liveRecorder.state !== 'inactive') {
      this.liveRecorder.stop();
    }
    this.liveRecorder = null;
  }

  /** The whole session as one blob, for the final commit pass. */
  fullRecording(): Blob {
    return new Blob(this.chunks, { type: this.mimeType || 'audio/webm' });
  }

  /**
   * Vocal intensity for the session, or null if nobody spoke.
   *
   * Absolute levels are meaningless across machines — a headset mic and a laptop
   * array microphone disagree by more than a whisper differs from a shout — so
   * these are stored raw and normalised against the diary's own history at
   * render time. "Loud" only means anything relative to how *you* usually sound.
   */
  sessionVocals(): {
    mean: number;
    peak: number;
    brightness: number;
    speakingSeconds: number;
  } | null {
    if (this.vocalFrames === 0) return null;
    return {
      mean: this.vocalSum / this.vocalFrames,
      peak: this.vocalPeak,
      brightness: this.brightnessSum / this.vocalFrames,
      // Frames are requestAnimationFrame ticks, so this is approximate and
      // only ever used to discard entries too short to characterise.
      speakingSeconds: this.vocalFrames / 60,
    };
  }

  elapsedMs(): number {
    return this.startedAt === 0 ? 0 : performance.now() - this.startedAt;
  }

  async stop(): Promise<Blob> {
    this.stopLevelLoop();
    // Before the main recorder: the live cycle restarts itself from its own
    // onstop handler, so leaving it running would spawn a new recorder against
    // a stream we are about to tear down.
    this.stopLiveCycle();

    const recorder = this.recorder;
    if (recorder && recorder.state !== 'inactive') {
      // Wait for the final dataavailable so the last few seconds aren't lost.
      await new Promise<void>((resolve) => {
        recorder.addEventListener('stop', () => resolve(), { once: true });
        recorder.stop();
      });
    }

    this.stream?.getTracks().forEach((track) => track.stop());
    await this.context?.close().catch(() => undefined);

    this.stream = null;
    this.context = null;
    this.analyser = null;
    this.recorder = null;

    return this.fullRecording();
  }

  private stopLevelLoop(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  private loop = (): void => {
    if (!this.analyser) return;

    this.analyser.getFloatTimeDomainData(this.timeBuffer);
    this.analyser.getByteFrequencyData(this.freqBuffer);

    let sumSquares = 0;
    for (let i = 0; i < this.timeBuffer.length; i++) {
      sumSquares += this.timeBuffer[i] * this.timeBuffer[i];
    }
    const rms = Math.sqrt(sumSquares / this.timeBuffer.length);

    // Asymmetric smoothing: rise fast so a sudden laugh registers immediately,
    // fall slowly so the world doesn't strobe between syllables.
    const alpha = rms > this.smoothedLevel ? 0.45 : 0.08;
    this.smoothedLevel += (rms - this.smoothedLevel) * alpha;

    // The floor only ratchets downward, tracking the quietest recent moment.
    if (rms < this.noiseFloor) {
      this.noiseFloor += (rms - this.noiseFloor) * 0.1;
    } else {
      this.noiseFloor += FLOOR_ADAPT_RATE * (rms - this.noiseFloor) * 0.01;
    }

    const speaking = this.smoothedLevel > this.noiseFloor * SPEAKING_MARGIN;

    let weighted = 0;
    let total = 0;
    for (let i = 0; i < this.freqBuffer.length; i++) {
      weighted += i * this.freqBuffer[i];
      total += this.freqBuffer[i];
    }
    const centroid = total > 0 ? weighted / total / this.freqBuffer.length : 0;

    // Compressed with a root curve: speech RMS lives in a narrow low band and
    // a linear map would leave the world barely moving at normal volume.
    const level = Math.min(1, Math.pow(this.smoothedLevel / 0.15, 0.6));
    const brightness = Math.min(1, centroid * 3.2);

    if (speaking) {
      this.vocalSum += level;
      this.brightnessSum += brightness;
      this.vocalPeak = Math.max(this.vocalPeak, level);
      this.vocalFrames++;
    }

    this.options.onLevel?.({ level, brightness, speaking });

    this.rafId = requestAnimationFrame(this.loop);
  };
}

function pickMimeType(): string {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4', // Safari
  ];
  for (const type of candidates) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return '';
}
