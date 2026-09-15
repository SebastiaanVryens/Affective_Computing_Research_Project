/**
 * The render loop and camera.
 *
 * The world has two floors and one camera that flies between them:
 *
 *   mind — a small piece of ground built from what you keep talking about, at
 *          y = 0. This is what opens, because the first thing the app should
 *          say is "here is what you have become", not "here are four hundred
 *          data points".
 *   core — the memory orbs, one per entry, seventy-eight units below. The
 *          evidence the ground above was summarised from.
 *
 * Both floors are framed at about the same distance, on purpose. They are two
 * rooms, not a room and a map: the orbs surround you at arm's length and so does
 * the ground, so switching changes what is around you rather than how far back
 * you are standing.
 *
 * One scene rather than two, and that is the decision the rest of this file
 * follows from. You can talk in either floor and switch while you are talking,
 * so the live orb, the keyword field, the mood bus and the bloom pass all have
 * to survive the move — which they do trivially if the move is a camera
 * animation and not a teardown. It also makes the descent literal: the memory
 * threads you follow down are real geometry in real space, anchored at both
 * ends to the two things they connect.
 *
 * You still don't walk around. The camera orbits, drifts, and parallaxes with
 * the pointer; the wheel dollies in and out. That is the whole control scheme,
 * because someone trying to talk about their day should not also be learning
 * flight controls.
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

import { type Emotion, type EmotionVector, zeroVector } from '../emotions';
import { mood } from '../capture/mood';
import type { DiaryEntry } from '../state/db';
import { Atmosphere } from './atmosphere';
import { KeywordField } from './keywords';
import { CORE_LEVEL, MIND_LEVEL, Mindscape } from './mindscape';
import { LiveOrb, MemoryOrbs } from './orbs';
import { MemoryThreads } from './threads';
import type { BiomeChoice } from './biomes';
import type { MotifPresence } from './motifs';

export type WorldView = 'mind' | 'core';

/**
 * Where the camera stands on each floor.
 *
 * `stageHeight` is where the live orb and the drifting keywords sit above that
 * floor — on the core floor it reproduces exactly where they have always been,
 * and on the mind floor it puts them hovering over the mouth of the well, so a
 * session recorded from the top has the words rising out of the shaft.
 */
interface Rig {
  radius: number;
  height: number;
  lookHeight: number;
  orbitSpeed: number;
  sway: number;
  stageHeight: number;
  parallax: number;
}

const RIGS: Record<WorldView, Rig> = {
  mind: {
    // Deliberately close to the core rig below. Both floors are meant to be
    // rooms you are standing in — switching view should change what surrounds
    // you, not pull back to survey it from altitude.
    radius: 24,
    height: 9,
    lookHeight: 1.4,
    // A little slower than the core's orbit: the ground is a continuous surface
    // rather than scattered points, so the same angular rate reads as faster.
    orbitSpeed: 0.026,
    sway: 0.7,
    stageHeight: 3.2,
    parallax: 2.6,
  },
  core: {
    radius: 22,
    height: 4.2,
    lookHeight: 1,
    orbitSpeed: 0.035,
    sway: 0.6,
    stageHeight: 1.2,
    parallax: 2.4,
  },
};

/**
 * How long the flight between floors takes at full distance.
 *
 * Long enough to see the shaft go by, short enough that someone who only wanted
 * to check a memory isn't waiting on a cutscene. An interrupted flight scales
 * this by the distance still to cover, so tapping the button twice doesn't buy a
 * fresh two and a half seconds to travel a tenth of the way.
 */
const TRAVEL_SECONDS = 2.6;

const BASE_FOV = 52;

/**
 * Where the haze starts and where it becomes total.
 *
 * These two numbers are what make the mind floor a single picture rather than a
 * set of objects at different distances. `NEAR` sits past the far edge of the
 * island, so nothing you are actually looking at is touched; `FAR` sits inside
 * the sea's own radius (water.ts), so the water reaches full haze *before* it
 * runs out — which is the entire trick to having a horizon instead of a rim.
 *
 * Everything that must not be fogged opts out at its own material: the memory
 * orbs, the drifting keywords, and every additive glow. Fog on an additive
 * surface brightens it with distance, which is worse than no fog.
 */
const FOG_NEAR = 35;
const FOG_FAR = 200;

/**
 * How fast the world turns when you hold a key, in radians per second.
 *
 * About a full turn in five seconds — two orders of magnitude above the ambient
 * drift, because this is a deliberate act rather than weather. Slower than this
 * and holding the key feels like nothing is happening; faster and a tap
 * overshoots whatever you were trying to look at.
 */
const SPIN_SPEED = 1.2;

/** How sharply the spin picks up and lets go. Higher is more immediate. */
const SPIN_EASE = 7;

/**
 * How close and how far the camera may be dollied, as a multiple of the rig's
 * own distance.
 *
 * Named, and shared by the wheel and the keys, because they are two ways of
 * driving one control and a second copy of these numbers is a second thing to
 * forget. In at 0.45 you can read the spine of a book on the island; out at 1.6
 * the far field is still inside the fog, which is what stops the world ending
 * in a visible rim.
 */
const ZOOM_MIN = 0.45;
const ZOOM_MAX = 1.6;

/**
 * How fast the keys dolly, in e-folds per second.
 *
 * Multiplicative rather than linear, the same as the wheel: a fixed number of
 * units per second crawls when you are close and lurches when you are far,
 * because what the eye reads is the *ratio* the distance changed by. At this
 * rate a held key crosses the whole range in about two seconds.
 */
const ZOOM_SPEED = 0.7;

/**
 * The orbit rate below which a world is, to the eye, not turning at all.
 *
 * Roughly one revolution in twenty minutes. Above it you can see the world
 * move against the frame within a few seconds of watching; below it you cannot,
 * and the core needs its own motion or it reads as a still photograph.
 *
 * Only the room gets anywhere near this — its `orbitScale` is a tenth — which
 * is the point. It is a threshold rather than a comparison against the core's
 * speed because every world is slower than the core's floor, and comparing the
 * two would treat every world as still.
 */
const STILL_ENOUGH = 0.006;

export interface SceneCallbacks {
  onOrbPicked?: (entryId: string) => void;
  onViewChanged?: (view: WorldView) => void;
}

export class MindscapeWorld {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private composer: EffectComposer;
  private bloom: UnrealBloomPass;

  private atmosphere = new Atmosphere();
  private mindscape = new Mindscape();
  private orbs = new MemoryOrbs();
  private liveOrb = new LiveOrb();
  private keywords = new KeywordField();
  /**
   * Lives here rather than in the mindscape because it is the only object that
   * spans both floors, and this is the only class that can see both.
   */
  private threads = new MemoryThreads();

  private clock = new THREE.Clock();
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2();
  /** Where the camera is being nudged by the pointer, in normalised units. */
  private parallax = new THREE.Vector2();
  private parallaxTarget = new THREE.Vector2();

  /**
   * Where the camera stands on its circle.
   *
   * Advances *by itself* at the mind floor's rate and nothing else — see
   * `updateCamera`, which explains why the core's rotation is given to the
   * galaxy instead of to this. What the viewer does with A and D is added on
   * top and simply stays: turn the world round twice and it is two turns round,
   * on both floors, until you turn it back.
   *
   * Accumulated rather than derived from elapsed time, because the rate itself
   * changes: saving an entry can change the diary's restlessness, and an angle
   * computed as `elapsed * speed` would jump the moment it did. Never wrapped,
   * so there is no limit on how far round you can go in either direction — sin
   * and cos do not care, and wrapping would put a seam in a control whose whole
   * point is that it has none.
   */
  private orbitAngle = 0;

  /** -1 while turning left, +1 while turning right, 0 when no key is held. */
  private spinInput = 0;
  /** Eased toward `spinInput * SPIN_SPEED`, so the world starts and stops. */
  private spinVelocity = 0;
  /** +1 while pulling in, -1 while pushing out, 0 when no key is held. */
  private zoomInput = 0;

  private view: WorldView = 'mind';
  /** 0 = fully on the island, 1 = fully at the orbs. */
  private blend = 0;
  private travel = 1;
  private travelFrom = 0;
  private travelTo = 0;
  private travelSeconds = TRAVEL_SECONDS;

  private zoom = 1;
  private zoomTarget = 1;
  private lastFov = BASE_FOV;

  /** The haze colour, recomputed each frame from the sky. Reused, never new'd. */
  private haze = new THREE.Color('#6f7793');
  private fog = new THREE.Fog(0x6f7793, FOG_NEAR, FOG_FAR);

  /** The island is rebuilt from these, once per frame at most. See setEntries. */
  private pendingEntries: DiaryEntry[] = [];
  private pendingLifetime: EmotionVector = zeroVector();
  private landscapeDirty = true;

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
      BASE_FOV,
      container.clientWidth / Math.max(1, container.clientHeight),
      0.1,
      // Far enough to keep the island in view from the core floor, plus the
      // sky sphere behind it.
      600
    );

    // The galaxy is moved as a whole rather than by rewriting every entry's
    // stored position: worldPosition is written once when an entry is saved and
    // must never change afterwards, or old memories would drift every time the
    // layout was touched. See world/placement.ts.
    this.orbs.group.position.y = CORE_LEVEL;
    this.scene.fog = this.fog;

    this.scene.add(
      this.atmosphere.group,
      this.mindscape.group,
      this.threads.group,
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
      // The island's beacons and shaft rings clear it deliberately, by carrying
      // colours multiplied past 1 — see mindscape.ts.
      0.9
    );
    this.composer.addPass(this.bloom);
    // Handles the tone-mapping/colour-space conversion at the end of the chain,
    // which RenderPass would otherwise have done on its own.
    this.composer.addPass(new OutputPass());

    this.setSize();
    this.bindEvents();
    // Placed before the first frame so there is never a frame of empty space
    // while the camera settles.
    this.updateCamera(0, 0, 0);
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
    this.renderer.domElement.removeEventListener('wheel', this.onWheel);
    document.removeEventListener('visibilitychange', this.onVisibilityChange);

    this.atmosphere.dispose();
    this.mindscape.dispose();
    this.threads.dispose();
    this.orbs.dispose();
    this.liveOrb.dispose();
    this.keywords.dispose();
    this.composer.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  // -- world content ---------------------------------------------------

  /**
   * Rebuild from what's in the database. Call after saving, deleting, or on load.
   *
   * The orbs rebuild immediately — they're a direct picture of the entry list.
   * The island is deferred to the next frame and coalesced, because it depends
   * on both the entries *and* the lifetime totals, which arrive as two separate
   * calls from main.ts. Marking dirty and rebuilding once makes the result
   * independent of which order they land in, and costs at most one frame.
   */
  setEntries(entries: DiaryEntry[]): void {
    this.orbs.rebuild(entries);
    this.pendingEntries = entries;
    this.landscapeDirty = true;
  }

  setLifetimeMood(totals: EmotionVector): void {
    this.atmosphere.setLifetimeMood(totals);
    this.pendingLifetime = totals;
    this.landscapeDirty = true;
  }

  /** What the ground currently shows, strongest theme first. */
  getMotifs(): MotifPresence[] {
    return this.mindscape.getMotifs();
  }

  /** Which kind of place the diary has turned out to be. */
  getPlace(): BiomeChoice {
    return this.mindscape.getPlace();
  }

  // -- view ------------------------------------------------------------

  getView(): WorldView {
    return this.view;
  }

  /**
   * Fly to the other floor.
   *
   * Safe to call mid-flight: the move restarts from wherever the camera
   * currently is, over a duration scaled to the distance left, so reversing
   * halfway doesn't stall and doesn't snap.
   */
  setView(view: WorldView): void {
    if (view === this.view) return;
    this.view = view;
    this.travelFrom = this.blend;
    this.travelTo = view === 'core' ? 1 : 0;
    this.travel = 0;
    this.travelSeconds =
      TRAVEL_SECONDS * Math.max(0.35, Math.abs(this.travelTo - this.travelFrom));
    this.callbacks.onViewChanged?.(view);
  }

  toggleView(): void {
    this.setView(this.view === 'mind' ? 'core' : 'mind');
  }

  /**
   * Turn the world by hand.
   *
   * @param direction -1 to turn left, +1 to turn right, 0 to stop. Held rather
   *                  than pulsed: the caller reports what is currently down, and
   *                  the world keeps turning until told otherwise. There is no
   *                  limit and no end stop — round and round, either way, on
   *                  whichever floor you are on.
   *
   * Deliberately not a "rotate by N degrees" call. Stepping would mean choosing
   * a step size, and any step large enough to feel responsive is large enough to
   * skip past the thing you were turning toward.
   */
  setSpin(direction: number): void {
    this.spinInput = Math.sign(direction);
  }

  /**
   * Dolly in and out by hand.
   *
   * @param direction +1 to pull in, -1 to push out, 0 to stop. Held, like
   *                  `setSpin`.
   *
   * Drives exactly what the wheel drives and stops where the wheel stops, so
   * the two are one control with two inputs rather than two controls that
   * happen to agree. Reaching the end of the range under a held key is silent —
   * it simply stops, the way it does under the wheel.
   */
  setZoom(direction: number): void {
    this.zoomInput = Math.sign(direction);
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

    if (this.landscapeDirty) {
      this.landscapeDirty = false;
      this.mindscape.rebuild(this.pendingEntries, this.pendingLifetime);
      this.threads.rebuild(this.mindscape.getThreadAnchors());
    }

    this.advanceTravel(delta);

    this.atmosphere.update(delta, elapsed, snapshot);

    // One haze colour, read off the sky and handed to everything that has to
    // agree with it. The fog and the sea's reflection are the same value, which
    // is why the far water has no edge.
    this.atmosphere.horizonColor(this.haze);
    this.fog.color.copy(this.haze);

    this.mindscape.update(delta, elapsed, snapshot, this.haze);
    this.orbs.update(delta, elapsed, snapshot.arousal);
    this.liveOrb.update(delta, elapsed, snapshot);
    this.keywords.update(delta, elapsed);
    this.threads.update(
      (entryId, out) => this.orbs.coreAnchor(entryId, out),
      elapsed
    );

    this.updateCamera(delta, elapsed, snapshot.arousal);

    // Bloom swells with vocal energy. Subtle, but it's most of why a loud
    // moment reads as a loud moment.
    this.bloom.strength += (0.5 + snapshot.arousal * 0.35 - this.bloom.strength) *
      Math.min(1, delta * 3);

    this.composer.render();
  };

  /**
   * Advance the flight between floors.
   *
   * Smootherstep rather than a plain ease: the descent starts and ends with zero
   * acceleration as well as zero velocity, which is the difference between
   * falling and being dropped.
   */
  private advanceTravel(delta: number): void {
    if (this.travel >= 1) return;
    this.travel = Math.min(1, this.travel + delta / this.travelSeconds);
    const t = this.travel;
    const eased = t * t * t * (t * (t * 6 - 15) + 10);
    this.blend = this.travelFrom + (this.travelTo - this.travelFrom) * eased;
  }

  private updateCamera(delta: number, elapsed: number, arousal: number): void {
    this.parallax.lerp(this.parallaxTarget, Math.min(1, delta * 2.2));

    // Held keys move the same target the wheel moves, so they inherit its
    // smoothing and its end stops for free. Exponential, because zoom is a
    // ratio: halving the distance should take the same time whether you started
    // near or far.
    if (this.zoomInput !== 0) {
      this.zoomTarget = clamp(
        this.zoomTarget * Math.exp(-this.zoomInput * ZOOM_SPEED * delta),
        ZOOM_MIN,
        ZOOM_MAX
      );
    }
    this.zoom += (this.zoomTarget - this.zoom) * Math.min(1, delta * 6);

    const blend = this.blend;
    const mind = RIGS.mind;
    const core = RIGS.core;

    /** Non-zero only while actually in flight; peaks halfway down. */
    const rush = this.travel < 1 ? Math.sin(Math.PI * this.travel) : 0;

    const focusY = MIND_LEVEL + (CORE_LEVEL - MIND_LEVEL) * blend;
    const lookHeight = lerp(mind.lookHeight, core.lookHeight, blend);

    // How much this particular world turns.
    //
    // Not a constant any more. A world that circles relentlessly is tiring to
    // sit in front of while you are trying to talk about your day, and it is
    // also a claim — that this place is busy — which most diaries do not
    // support. The mind floor takes the diary's own restlessness almost
    // directly, so a settled one barely moves; the core keeps a floor under it,
    // because a galaxy of memories that is completely static reads as a
    // photograph rather than a place.
    const restlessness = this.mindscape.getRestlessness();
    const mindSpeed = mind.orbitSpeed * (0.06 + 0.94 * restlessness);
    const coreSpeed = core.orbitSpeed * (0.45 + 0.55 * restlessness);

    // The camera's angle only ever advances at the *mind* floor's rate.
    //
    // It used to advance at whichever floor you were on, which is the obvious
    // reading and is wrong in one specific case. The core keeps a floor under
    // its speed on purpose — a galaxy of memories that is completely static
    // reads as a photograph — so on a world whose own rate is near zero the two
    // disagree badly. Go down to the core from a room, wait, come back, and the
    // room has turned thirty degrees while you were away: the still world was
    // being rotated by the moving one through a shared accumulator.
    //
    // Now the mind floor's orientation is a pure function of how long the app
    // has been open and how restless the diary is. Leave a still room and it is
    // exactly where you left it, however long you spend below. A world that
    // does turn still turns, and lands where it would have had you never gone
    // down — which is the same rule, not an exception to it.
    this.orbitAngle += delta * mindSpeed;

    // Then whatever the viewer is asking for, on top.
    //
    // Eased rather than applied raw: a step change in angular rate reads as the
    // world being yanked, and the ease is what makes a tap a nudge and a hold a
    // sweep without needing two separate controls. Added to the same angle the
    // drift uses, so the two compose instead of fighting — hold a key on a world
    // that is already turning and you are speeding it up, not overriding it.
    this.spinVelocity +=
      (this.spinInput * SPIN_SPEED - this.spinVelocity) * Math.min(1, delta * SPIN_EASE);
    this.orbitAngle += delta * this.spinVelocity;

    // A still world gets its core turned for it. A turning one does not.
    //
    // The core's speed floor exists so that a galaxy of memories is never a
    // photograph. On a world that already turns, the camera orbiting at the
    // mind floor's rate satisfies that on its own, and adding anything here
    // would make the core turn faster than the floor above it — reintroducing
    // the very mismatch this is here to remove, just in the other direction.
    //
    // So it is gated on *stillness*, not on the gap between the two speeds. The
    // difference matters: gating on the gap is the same expression as adding
    // `coreSpeed - mindSpeed`, which is what the first attempt did and which
    // spins the galaxy for every biome. Only a floor that genuinely does not
    // move needs rescuing.
    //
    // Rotating the orbs one way and orbiting the camera the other are the same
    // picture — relative motion is all the eye has to go on down there — but
    // only one of them is remembered by the floor above. The descent's extra
    // rate goes here too, so `orbitAngle` has no term a round trip could leave
    // behind; the threads still sweep past rather than approaching head-on,
    // because their lower ends are what is moving.
    const stillness = 1 - Math.min(1, mindSpeed / STILL_ENOUGH);
    const coreSpin = coreSpeed * stillness * blend + rush * 0.035;
    this.orbs.group.rotation.y -= delta * coreSpin;

    // A true dolly toward the look point, so zooming in on the island doesn't
    // also tip the camera further overhead.
    const distance = lerp(mind.radius, core.radius, blend) * this.zoom;
    const height =
      lookHeight + (lerp(mind.height, core.height, blend) - lookHeight) * this.zoom;
    const sway = Math.sin(elapsed * 0.21) * lerp(mind.sway, core.sway, blend);
    const parallaxScale = lerp(mind.parallax, core.parallax, blend);

    this.camera.position.x = Math.sin(this.orbitAngle) * distance + this.parallax.x * parallaxScale;
    this.camera.position.z = Math.cos(this.orbitAngle) * distance;
    this.camera.position.y =
      focusY + height + sway + this.parallax.y * 1.6 + arousal * 0.4;

    this.camera.lookAt(0, focusY + lookHeight + this.parallax.y * 0.3, 0);

    // A couple of degrees of extra field of view while falling. Widening the
    // lens as you accelerate is the oldest trick there is for making a move feel
    // like speed rather than like a lerp, and it costs one matrix update.
    //
    // Kept small. A hard zoom during a move the viewer did not initiate with
    // their own head is one of the reliable ways to make someone queasy, and
    // the threads now carry the sense of speed on their own.
    const fov = BASE_FOV + rush * 4;
    if (Math.abs(fov - this.lastFov) > 0.05) {
      this.camera.fov = fov;
      this.camera.updateProjectionMatrix();
      this.lastFov = fov;
    }

    // Everything that is "where the viewer is" rides along: sky, lights, the
    // live orb, and the words drifting off it.
    this.atmosphere.setFocusHeight(focusY);
    this.mindscape.setDescent(blend);

    const stage = focusY + lerp(mind.stageHeight, core.stageHeight, blend);
    this.liveOrb.mesh.position.y = stage;
    // The keyword field spawns its words relative to its own origin, which on
    // the core floor has always sat 1.2 below the live orb.
    this.keywords.group.position.y = stage - 1.2;
  }

  // -- input -----------------------------------------------------------

  private bindEvents(): void {
    this.resizeObserver = new ResizeObserver(() => this.setSize());
    this.resizeObserver.observe(this.container);
    window.addEventListener('pointermove', this.onPointerMove);
    this.renderer.domElement.addEventListener('click', this.onClick);
    // Not passive: the page must not scroll behind the canvas while zooming.
    this.renderer.domElement.addEventListener('wheel', this.onWheel, { passive: false });
    document.addEventListener('visibilitychange', this.onVisibilityChange);
  }

  private onPointerMove = (event: PointerEvent): void => {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.parallaxTarget.set(this.pointer.x, this.pointer.y * 0.6);
  };

  /**
   * Wheel dollies in and out.
   *
   * The room is framed to be taken in whole, which means a stack of books in it
   * is small. Being able to lean in is what makes "there are books there because
   * you talk about books" something you can verify rather than take on trust.
   */
  private onWheel = (event: WheelEvent): void => {
    event.preventDefault();
    const step = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY;
    this.zoomTarget = clamp(this.zoomTarget * (1 + step * 0.0011), ZOOM_MIN, ZOOM_MAX);
  };

  private onClick = (): void => {
    this.raycaster.setFromCamera(this.pointer, this.camera);

    // Whichever floor the camera is nearer owns the click. Mid-flight this
    // flips at the halfway point, which is also where the island stops being
    // the thing under the cursor.
    if (this.blend > 0.5) {
      const hit = this.orbs.pick(this.raycaster);
      if (hit) this.callbacks.onOrbPicked?.(hit.entryId);
      return;
    }

    const entryId = this.mindscape.pick(this.raycaster);
    if (entryId) this.callbacks.onOrbPicked?.(entryId);
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

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}
