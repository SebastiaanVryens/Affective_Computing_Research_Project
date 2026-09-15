/**
 * Memory orbs — the world's permanent record.
 *
 * One orb per diary entry, coloured by that entry's dominant emotion, placed
 * once and never moved. The world only ever grows: nothing here is cleared
 * between sessions, which is the point. Watching the galaxy fill in over weeks
 * *is* the diary.
 *
 * Placement comes from placement.ts, which both this module and the save path
 * share — see the note there on why there's exactly one definition.
 *
 * Core memories are separate: larger, brighter, lifted onto a ring near the
 * centre where they're always in view.
 */

import * as THREE from 'three';
import { PALETTE, type Emotion } from '../emotions';
import type { DiaryEntry } from '../state/db';
import { LAYER } from './layers';
import { spiralPlacement } from './placement';

const CORE_RING_RADIUS = 6.2;
const CORE_RING_HEIGHT = 1.4;

/**
 * How hard a core memory glows, as a base plus the size of its slow pulse.
 *
 * Kept here because `update()` rewrites emissiveIntensity every frame, so the
 * value passed to the material constructor survives exactly one frame and
 * setting it there alone does nothing. One constant, used in both places.
 *
 * The `glow` palette entries are pale on purpose, so intensity above roughly 1
 * clips several channels at once and the orb turns into a featureless white
 * disc — losing both its emotion colour and any sense of shape. Sitting just
 * under the bloom threshold in scene.ts means only the orb's own dominant
 * channel spills, so a core memory glows in *its* colour instead of in white.
 */
const CORE_EMISSIVE_BASE = 0.8;
const CORE_EMISSIVE_PULSE = 0.18;

export interface OrbPick {
  entryId: string;
  isCore: boolean;
}

/** Vector form of the shared placement function, for scene-side use. */
export function spiralPosition(index: number): THREE.Vector3 {
  const { x, y, z } = spiralPlacement(index);
  return new THREE.Vector3(x, y, z);
}

export class MemoryOrbs {
  readonly group = new THREE.Group();

  private mesh: THREE.InstancedMesh | null = null;
  private coreGroup = new THREE.Group();
  private entries: DiaryEntry[] = [];
  private coreMeshes: THREE.Mesh[] = [];

  /** Per-orb phase offsets so they don't all pulse in lockstep. */
  private phases: Float32Array = new Float32Array(0);
  private baseScales: Float32Array = new Float32Array(0);

  private dummy = new THREE.Object3D();
  private color = new THREE.Color();
  private secondary = new THREE.Color();

  private geometry = new THREE.IcosahedronGeometry(0.5, 3);
  private material = makeOrbMaterial();

  constructor() {
    this.group.add(this.coreGroup);
  }

  /**
   * Rebuild from the full entry list.
   *
   * InstancedMesh can't grow, so adding one entry means reallocating. That's
   * fine at diary scale (hundreds of entries, rebuilt only on save) and far
   * simpler than over-allocating and tracking a high-water mark.
   */
  rebuild(entries: DiaryEntry[]): void {
    this.entries = entries;
    this.disposeMesh();

    const regular = entries.filter((e) => !e.isCoreMemory);
    if (regular.length > 0) {
      // Clone rather than share: the instanced attributes below are sized to
      // this particular instance count, and writing them onto the shared
      // geometry would leave stale buffers behind on the next rebuild.
      this.mesh = new THREE.InstancedMesh(
        this.geometry.clone(),
        this.material,
        regular.length
      );
      this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.phases = new Float32Array(regular.length);
      this.baseScales = new Float32Array(regular.length);

      // Per-instance second emotion, consumed by the patched shader.
      const secondaryColors = new Float32Array(regular.length * 3);
      const blendAmounts = new Float32Array(regular.length);

      regular.forEach((entry, i) => {
        const position = entry.worldPosition
          ? new THREE.Vector3(
              entry.worldPosition.x,
              entry.worldPosition.y,
              entry.worldPosition.z
            )
          : spiralPosition(i);

        // A confident, longer entry earns a bigger orb — the world's visual
        // weight tracks how much you actually said, not just how often.
        const scale =
          0.55 +
          0.5 * entry.certainty +
          0.35 * Math.min(1, entry.durationSeconds / 120);

        this.baseScales[i] = scale;
        this.phases[i] = (i * 0.618) % 1;

        this.dummy.position.copy(position);
        this.dummy.scale.setScalar(scale);
        this.dummy.updateMatrix();
        this.mesh!.setMatrixAt(i, this.dummy.matrix);
        this.mesh!.setColorAt(i, this.color.set(PALETTE[entry.dominant].base));

        // An entry the backend judged a genuine blend gets its second emotion
        // painted across the other side of the orb. Entries with one emotion —
        // or with a spread the channels didn't corroborate — stay single-toned,
        // so a two-toned orb always means something.
        const second = entry.blend?.isBlend ? entry.blend.components[1] : undefined;
        if (second) {
          this.secondary.set(PALETTE[second.emotion].base);
          // The secondary's share drives how much of the orb it claims, capped
          // below 1 so the primary emotion always stays legible.
          blendAmounts[i] = Math.min(0.85, second.share * 1.6);
        } else {
          this.secondary.set(PALETTE[entry.dominant].base);
          blendAmounts[i] = 0;
        }
        secondaryColors[i * 3] = this.secondary.r;
        secondaryColors[i * 3 + 1] = this.secondary.g;
        secondaryColors[i * 3 + 2] = this.secondary.b;
      });

      this.mesh.geometry.setAttribute(
        'aSecondary',
        new THREE.InstancedBufferAttribute(secondaryColors, 3)
      );
      this.mesh.geometry.setAttribute(
        'aBlend',
        new THREE.InstancedBufferAttribute(blendAmounts, 1)
      );

      this.mesh.instanceMatrix.needsUpdate = true;
      if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
      this.group.add(this.mesh);
    }

    this.rebuildCoreMemories(entries.filter((e) => e.isCoreMemory));
  }

  private rebuildCoreMemories(cores: DiaryEntry[]): void {
    for (const mesh of this.coreMeshes) {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
      this.coreGroup.remove(mesh);
    }
    this.coreMeshes = [];

    cores.forEach((entry, i) => {
      const palette = PALETTE[entry.dominant];
      const second = entry.blend?.isBlend ? entry.blend.components[1] : undefined;

      const material = new THREE.MeshStandardMaterial({
        color: palette.base,
        // A blended core memory glows with its second emotion rather than its
        // own highlight — the two-toned orb the film hangs its whole argument
        // on. These are the memories most likely to be blends, so it's where
        // the effect matters most.
        emissive: new THREE.Color(second ? PALETTE[second.emotion].glow : palette.glow),
        // Overwritten on the first frame of update(); see CORE_EMISSIVE_BASE.
        emissiveIntensity: CORE_EMISSIVE_BASE,
        roughness: 0.15,
        metalness: 0.1,
        // The scene fog is tuned for the island's distances; the galaxy lives on
        // the other floor and must look exactly as it always has.
        fog: false,
      });
      const mesh = new THREE.Mesh(new THREE.IcosahedronGeometry(0.78, 4), material);

      const angle = (i / Math.max(1, cores.length)) * Math.PI * 2;
      mesh.position.set(
        Math.cos(angle) * CORE_RING_RADIUS,
        CORE_RING_HEIGHT + Math.sin(angle * 2) * 0.5,
        Math.sin(angle) * CORE_RING_RADIUS
      );
      mesh.userData.entryId = entry.id;
      mesh.userData.isCore = true;
      mesh.userData.phase = i * 0.37;

      this.coreMeshes.push(mesh);
      this.coreGroup.add(mesh);
    });
  }

  update(delta: number, elapsed: number, arousal: number): void {
    // The whole galaxy turns slowly. Speed is fixed rather than arousal-linked:
    // tying rotation to your voice made the scene feel like it was reacting to
    // volume rather than to meaning.
    this.group.rotation.y += delta * 0.014;

    for (let i = 0; i < this.coreMeshes.length; i++) {
      const mesh = this.coreMeshes[i];
      const phase = mesh.userData.phase as number;
      const pulse = 1 + Math.sin(elapsed * 1.1 + phase * Math.PI * 2) * 0.045;
      mesh.scale.setScalar(pulse);
      mesh.rotation.y += delta * 0.22;
      mesh.rotation.x += delta * 0.08;

      const material = mesh.material as THREE.MeshStandardMaterial;
      // Arousal contributes far less than it used to: at full weight a loud
      // moment on its own was enough to push these back into white-out.
      material.emissiveIntensity =
        CORE_EMISSIVE_BASE +
        CORE_EMISSIVE_PULSE * Math.sin(elapsed * 0.9 + phase) +
        arousal * 0.25;
    }

    this.coreGroup.rotation.y -= delta * 0.05; // counter-rotates against the galaxy
  }

  /**
   * Spawns a newly-saved entry with a brief flare, so a fresh memory visibly
   * arrives rather than silently appearing on the next rebuild.
   */
  positionForNewEntry(entryCount: number): THREE.Vector3 {
    return spiralPosition(entryCount);
  }

  /**
   * World position of a core memory's orb, if it has one.
   *
   * Queried per frame by the threads in threads.ts. It has to be live rather
   * than baked because `coreGroup` counter-rotates against the galaxy — a thread
   * anchored to where the orb was would slide off it within a minute.
   */
  coreAnchor(entryId: string, out: THREE.Vector3): boolean {
    for (const mesh of this.coreMeshes) {
      if (mesh.userData.entryId !== entryId) continue;
      mesh.getWorldPosition(out);
      return true;
    }
    return false;
  }

  /** Ray-pick an orb. Used for clicking a memory to read it back. */
  pick(raycaster: THREE.Raycaster): OrbPick | null {
    const coreHits = raycaster.intersectObjects(this.coreMeshes, false);
    if (coreHits.length > 0) {
      return { entryId: coreHits[0].object.userData.entryId as string, isCore: true };
    }

    if (this.mesh) {
      const hits = raycaster.intersectObject(this.mesh, false);
      if (hits.length > 0 && hits[0].instanceId !== undefined) {
        const regular = this.entries.filter((e) => !e.isCoreMemory);
        const entry = regular[hits[0].instanceId];
        if (entry) return { entryId: entry.id, isCore: false };
      }
    }
    return null;
  }

  private disposeMesh(): void {
    if (this.mesh) {
      this.group.remove(this.mesh);
      // The cloned geometry is ours, so it has to go too — dispose() on the
      // InstancedMesh only releases its instance buffers.
      this.mesh.geometry.dispose();
      this.mesh.dispose();
      this.mesh = null;
    }
  }

  dispose(): void {
    this.disposeMesh();
    this.rebuildCoreMemories([]);
    this.geometry.dispose();
    this.material.dispose();
  }
}

/**
 * The live orb: a single sphere at the centre representing the session in
 * progress, before it's been committed to the world. It's the thing that
 * visibly reacts while you talk, and it's what "lands" into the spiral when you
 * finish.
 */
export class LiveOrb {
  readonly mesh: THREE.Mesh;
  private material: THREE.MeshStandardMaterial;
  private targetColor = new THREE.Color('#8d93a8');
  private active = false;

  constructor() {
    this.material = new THREE.MeshStandardMaterial({
      color: '#8d93a8',
      emissive: new THREE.Color('#8d93a8'),
      emissiveIntensity: 1.6,
      roughness: 0.1,
      metalness: 0.0,
      transparent: true,
      opacity: 0,
      fog: false,
    });
    this.mesh = new THREE.Mesh(new THREE.IcosahedronGeometry(1.35, 5), this.material);
    this.mesh.position.set(0, 1.2, 0);
    this.mesh.visible = false;
    // Transparent, and it hangs over the well on the mind floor with the sea
    // behind it — so it has to be ordered in front of the water explicitly, the
    // same as everything else made of light. See ./layers.ts.
    this.mesh.renderOrder = LAYER.glow;
  }

  setActive(active: boolean): void {
    this.active = active;
    this.mesh.visible = true; // stays visible through the fade-out
  }

  update(
    delta: number,
    elapsed: number,
    mood: { color: string; arousal: number; clarity: number; speaking: boolean }
  ): void {
    const targetOpacity = this.active ? 0.92 : 0;
    this.material.opacity += (targetOpacity - this.material.opacity) * Math.min(1, delta * 3);
    if (this.material.opacity < 0.01 && !this.active) {
      this.mesh.visible = false;
      return;
    }

    this.targetColor.set(mood.color);
    this.material.color.lerp(this.targetColor, Math.min(1, delta * 2.5));
    this.material.emissive.lerp(this.targetColor, Math.min(1, delta * 2.5));
    this.material.emissiveIntensity = 1.2 + 1.6 * mood.clarity + mood.arousal;

    // Breathing at rest, jitter while speaking — the orb should look like it's
    // listening when you're quiet and like it's receiving when you're not.
    const breathe = Math.sin(elapsed * 1.4) * 0.03;
    const speech = mood.speaking ? Math.sin(elapsed * 22) * 0.02 * mood.arousal : 0;
    this.mesh.scale.setScalar(1 + breathe + speech + mood.arousal * 0.12);
    this.mesh.rotation.y += delta * (0.12 + mood.arousal * 0.5);
    this.mesh.rotation.z += delta * 0.04;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.dispose();
  }
}

export function emotionColor(emotion: Emotion): string {
  return PALETTE[emotion].base;
}

/**
 * Orb material that can carry two emotions at once.
 *
 * A memory of feeling two things is not a memory of feeling their average, and
 * picking the argmax throws away the more interesting half. So each orb gets a
 * second instanced colour and blends between them across its surface — the
 * two-toned core memory the film is built around.
 *
 * Implemented by patching MeshStandardMaterial rather than writing a material
 * from scratch, so the orbs keep real lighting and still work with the bloom
 * pass. The extra per-instance data rides along as instanced attributes, which
 * means the whole galaxy stays a single draw call however many entries exist.
 */
function makeOrbMaterial(): THREE.MeshStandardMaterial {
  const material = new THREE.MeshStandardMaterial({
    roughness: 0.28,
    metalness: 0.05,
    emissiveIntensity: 1.0,
    // See the note on the core-memory material: the fog belongs to the mind
    // floor, and the galaxy is not on it.
    fog: false,
  });

  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
         attribute vec3 aSecondary;
         attribute float aBlend;
         varying vec3 vSecondary;
         varying float vBlend;
         varying vec3 vLocal;`
      )
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
         vSecondary = aSecondary;
         vBlend = aBlend;
         vLocal = normalize(position);`
      );

    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
         varying vec3 vSecondary;
         varying float vBlend;
         varying vec3 vLocal;`
      )
      .replace(
        '#include <color_fragment>',
        `#include <color_fragment>
         // Soft diagonal sweep rather than a hard equator: a hard split reads
         // as two half-spheres glued together, a gradient reads as one thing
         // that is genuinely both. The swirl term keeps it from looking like a
         // simple vertical fade as the galaxy rotates.
         float swirl = vLocal.y * 0.75 + sin(vLocal.x * 3.0 + vLocal.z * 2.0) * 0.18;
         float t = smoothstep(-0.45, 0.45, swirl) * vBlend;
         diffuseColor.rgb = mix(diffuseColor.rgb, vSecondary, t);`
      );
  };

  return material;
}
