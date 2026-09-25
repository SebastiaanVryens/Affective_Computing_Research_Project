/**
 * The threads between the two floors.
 *
 * One line per core memory, running from where it stands on the ground down to
 * the orb it actually is among the memories, in that memory's own colour.
 *
 * This replaced a shaft of concentric rings, and the reason is worth keeping:
 * the rings were decoration. They said "there is a way down" and nothing else —
 * a lift shaft, drawn in light. The threads say the true thing instead, which is
 * that the two floors are the same diary seen twice, and that the things you
 * marked as mattering are the places where the summary above and the record
 * below are demonstrably the same entry. You can follow one down with your eye.
 *
 * They also do a job the rings did badly. Falling seventy-eight units with the
 * sky travelling alongside you gives the eye nothing to measure the drop
 * against, and that mismatch is what made the descent uncomfortable. A thread is
 * fixed in the world at both ends, so it slides past at exactly the rate you are
 * moving — which is the cue that was missing.
 *
 * The lower end is queried every frame rather than baked, because the ring of
 * core memories in orbs.ts counter-rotates. A thread anchored to where the orb
 * *was* would drift off it within a minute.
 */

import * as THREE from 'three';
import { LAYER } from './layers';

/** Where a thread starts: the top of a beacon on the mind floor. */
export interface ThreadAnchor {
  entryId: string;
  top: THREE.Vector3;
  color: THREE.Color;
}

interface Thread {
  entryId: string;
  mesh: THREE.Mesh;
  material: THREE.MeshBasicMaterial;
  top: THREE.Vector3;
  phase: number;
}

/** Radius at the top and bottom of a thread. Barely more than a line. */
const TOP_RADIUS = 0.05;
const BOTTOM_RADIUS = 0.13;

export class MemoryThreads {
  readonly group = new THREE.Group();

  private threads: Thread[] = [];
  private geometry = new THREE.CylinderGeometry(TOP_RADIUS, BOTTOM_RADIUS, 1, 6, 1, true);

  private from = new THREE.Vector3();
  private to = new THREE.Vector3();
  private mid = new THREE.Vector3();
  private axis = new THREE.Vector3();
  private up = new THREE.Vector3(0, 1, 0);
  private quaternion = new THREE.Quaternion();

  /**
   * Rebuild from the core memories currently on the ground.
   *
   * A diary with nothing marked gets no threads at all, and the well is simply
   * an opening. That is the honest picture: the threads mean "these are the
   * memories you singled out", and drawing one when there are none would be
   * inventing a connection to make the view look furnished.
   */
  rebuild(anchors: ThreadAnchor[]): void {
    this.clear();

    anchors.forEach((anchor, i) => {
      const material = new THREE.MeshBasicMaterial({
        // Pushed past 1 so the bloom pass sees it; the palette's glow colours are
        // pale by design and sit just under the threshold at face value.
        color: anchor.color.clone().multiplyScalar(1.6),
        transparent: true,
        opacity: 0.4,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        // Additive: fog would mix the haze in and then add it, so a distant
        // thread would get brighter rather than fainter.
        fog: false,
      });

      const mesh = new THREE.Mesh(this.geometry, material);
      // A thread runs from the island down past the sea, so on a coast most of
      // its length is seen against water. Additive and depth-writeless like
      // everything else made of light, which means without this the sea is drawn
      // over it — see ./layers.ts.
      mesh.renderOrder = LAYER.glow;
      this.threads.push({
        entryId: anchor.entryId,
        mesh,
        material,
        top: anchor.top.clone(),
        phase: i * 0.8,
      });
      this.group.add(mesh);
    });
  }

  /**
   * Stretch each thread between its two ends.
   *
   * @param bottomOf  Resolves an entry id to where its orb currently is, or
   *                  null if it has none. Queried per frame; see the note at the
   *                  top of the file.
   * @param elapsed   Drives a slow travelling shimmer down each thread.
   */
  update(
    bottomOf: (entryId: string, out: THREE.Vector3) => boolean,
    elapsed: number
  ): void {
    for (const thread of this.threads) {
      if (!bottomOf(thread.entryId, this.to)) {
        thread.mesh.visible = false;
        continue;
      }
      thread.mesh.visible = true;

      this.from.copy(thread.top);
      this.axis.subVectors(this.to, this.from);
      const length = this.axis.length();
      if (length < 1e-3) {
        thread.mesh.visible = false;
        continue;
      }

      this.mid.addVectors(this.from, this.to).multiplyScalar(0.5);
      thread.mesh.position.copy(this.mid);

      // The cylinder is built along +Y, so one rotation takes it onto the axis
      // between the two ends. Scaling only Y keeps the thread the same thickness
      // however long it turns out to be.
      this.quaternion.setFromUnitVectors(this.up, this.axis.divideScalar(length));
      thread.mesh.quaternion.copy(this.quaternion);
      thread.mesh.scale.set(1, length, 1);

      thread.material.opacity = 0.32 + 0.16 * Math.sin(elapsed * 0.7 - thread.phase);
    }
  }

  clear(): void {
    for (const thread of this.threads) {
      this.group.remove(thread.mesh);
      thread.material.dispose();
    }
    this.threads = [];
  }

  dispose(): void {
    this.clear();
    this.geometry.dispose();
  }
}
