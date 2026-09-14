/**
 * The render loop and camera.
 *
 * You don't walk around. The camera sits at a fixed distance, drifts slowly, and
 * parallaxes a little with the pointer — enough that the world feels inhabited
 * without asking the user to learn controls while they're trying to talk about
 * their day.
 *
 * The loop reads the mood bus directly every frame rather than subscribing to
 * events. Rendering should never be gated on a network reply, a model, or a
 * callback ordering — if everything upstream stalls, the world keeps turning
 * with the last thing it knew.
 */

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

import { type Emotion, type EmotionVector } from '../emotions';
import { mood } from '../capture/mood';
import type { DiaryEntry } from '../state/db';
import { Atmosphere } from './atmosphere';
import { KeywordField } from './keywords';
import { LiveOrb, MemoryOrbs } from './orbs';

const CAMERA_DISTANCE = 22;
const CAMERA_HEIGHT = 4.2;

export interface SceneCallbacks {
  onOrbPicked?: (entryId: string) => void;
}

export class MindscapeWorld {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private composer: EffectComposer;
  private bloom: UnrealBloomPass;

  private atmosphere = new Atmosphere();
  private orbs = new MemoryOrbs();
  private liveOrb = new LiveOrb();
  private keywords = new KeywordField();

  private clock = new THREE.Clock();
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2();
  /** Where the camera is being nudged by the pointer, in normalised units. */
  private parallax = new THREE.Vector2();
  private parallaxTarget = new THREE.Vector2();

  private frameHandle = 0;
  private running = false;
  private resizeObserver: ResizeObserver | null = null;

  constructor(
    private container: HTMLElement,
    private callbacks: SceneCallbacks = {}
  ) {
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: 'high-performance',
    });
    // Capped at 2: on a 3x phone screen the bloom pass alone would halve the
    // frame rate for detail nobody can see.
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.95;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(
      52,
      container.clientWidth / Math.max(1, container.clientHeight),
      0.1,
      500
    );
    this.camera.position.set(0, CAMERA_HEIGHT, CAMERA_DISTANCE);
    this.camera.lookAt(0, 1, 0);

    this.scene.add(
      this.atmosphere.group,
      this.orbs.group,
      this.liveOrb.mesh,
      this.keywords.group
    );

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(
      new THREE.Vector2(container.clientWidth, container.clientHeight),
      0.5, // strength — enough to make emissive orbs glow, short of blowing out
      0.7, // radius
      // Threshold. Bloom runs before OutputPass, so this is compared against
      // *linear* scene values, not the tone-mapped image: at 0.32 an ordinary
      // sky pixel cleared it and the entire backdrop glowed. The orbs carry
      // emissiveIntensity 1.0–2.4, so sitting just under 1 keeps the glow on
      // the things that are meant to be emitting and off everything else.
      0.9
    );
    this.composer.addPass(this.bloom);
    // Handles the tone-mapping/colour-space conversion at the end of the chain,
    // which RenderPass would otherwise have done on its own.
    this.composer.addPass(new OutputPass());

    this.setSize();
    this.bindEvents();
  }

  // -- lifecycle -------------------------------------------------------

  start(): void {
    if (this.running) return;
    this.running = true;
    this.clock.start();
    this.loop();
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.frameHandle);
  }

  dispose(): void {
    this.stop();
    this.resizeObserver?.disconnect();
    window.removeEventListener('pointermove', this.onPointerMove);
    this.renderer.domElement.removeEventListener('click', this.onClick);
    document.removeEventListener('visibilitychange', this.onVisibilityChange);

    this.atmosphere.dispose();
    this.orbs.dispose();
    this.liveOrb.dispose();
    this.keywords.dispose();
    this.composer.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  // -- world content ---------------------------------------------------

  /** Rebuild the permanent galaxy. Call after saving, deleting, or on load. */
  setEntries(entries: DiaryEntry[]): void {
    this.orbs.rebuild(entries);
  }

  setLifetimeMood(totals: EmotionVector): void {
    this.atmosphere.setLifetimeMood(totals);
  }

  /**
   * Note this does not clear keywords on stop: words from the finished session
   * are left to drift out on their own, because cutting them at the moment you
   * stop talking makes the end of a session feel like a page refresh.
   */
  setRecording(active: boolean): void {
    this.liveOrb.setActive(active);
  }

  spawnKeyword(text: string, emotion: Emotion, weight = 0.5): void {
    this.keywords.spawn(text, emotion, weight);
  }

  clearKeywords(): void {
    this.keywords.clear();
  }

  // -- loop ------------------------------------------------------------

  private loop = (): void => {
    if (!this.running) return;
    this.frameHandle = requestAnimationFrame(this.loop);

    const delta = Math.min(this.clock.getDelta(), 0.1); // clamp: tab-switch jumps
    const elapsed = this.clock.elapsedTime;
    const snapshot = mood.current();

    this.atmosphere.update(delta, elapsed, snapshot);
    this.orbs.update(delta, elapsed, snapshot.arousal);
    this.liveOrb.update(delta, elapsed, snapshot);
    this.keywords.update(delta, elapsed);

    this.updateCamera(delta, elapsed, snapshot.arousal);

    // Bloom swells with vocal energy. Subtle, but it's most of why a loud
    // moment reads as a loud moment.
    this.bloom.strength += (0.5 + snapshot.arousal * 0.35 - this.bloom.strength) *
      Math.min(1, delta * 3);

    this.composer.render();
  };

  private updateCamera(delta: number, elapsed: number, arousal: number): void {
    this.parallax.lerp(this.parallaxTarget, Math.min(1, delta * 2.2));

    // Slow idle orbit plus pointer parallax. The orbit is deliberately slower
    // than the galaxy's own rotation so the two don't beat against each other.
    const orbit = elapsed * 0.035;
    const sway = Math.sin(elapsed * 0.21) * 0.6;

    this.camera.position.x =
      Math.sin(orbit) * CAMERA_DISTANCE + this.parallax.x * 2.4;
    this.camera.position.z = Math.cos(orbit) * CAMERA_DISTANCE;
    this.camera.position.y =
      CAMERA_HEIGHT + sway + this.parallax.y * 1.6 + arousal * 0.4;

    this.camera.lookAt(0, 1 + this.parallax.y * 0.3, 0);
  }

  // -- input -----------------------------------------------------------

  private bindEvents(): void {
    this.resizeObserver = new ResizeObserver(() => this.setSize());
    this.resizeObserver.observe(this.container);
    window.addEventListener('pointermove', this.onPointerMove);
    this.renderer.domElement.addEventListener('click', this.onClick);
    document.addEventListener('visibilitychange', this.onVisibilityChange);
  }

  private onPointerMove = (event: PointerEvent): void => {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.parallaxTarget.set(this.pointer.x, this.pointer.y * 0.6);
  };

  private onClick = (): void => {
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = this.orbs.pick(this.raycaster);
    if (hit) this.callbacks.onOrbPicked?.(hit.entryId);
  };

  /**
   * Pause rendering in a hidden tab.
   *
   * requestAnimationFrame already throttles when backgrounded, but the clock
   * keeps running — so without this, returning to the tab replays the whole
   * elapsed time as one enormous delta and the camera lurches.
   */
  private onVisibilityChange = (): void => {
    if (document.hidden) {
      this.stop();
    } else if (!this.running) {
      this.clock.getDelta(); // discard the accumulated gap
      this.start();
    }
  };

  private setSize(): void {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.composer.setSize(width, height);
    this.bloom.setSize(width, height);
  }
}
