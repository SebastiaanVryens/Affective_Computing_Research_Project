/**
 * What you can see out of the window.
 *
 * The room is chosen when somebody's diary is mostly indoor things — books,
 * work, home. But almost nobody's diary is *only* that, and the second thing
 * they keep talking about is exactly what should be outside the glass: the coast
 * if they talk about the sea, a skyline if they talk about the office, hills if
 * they climb. The room says where they spend their time; the view says what the
 * rest of it is about.
 *
 * ── Why this is geometry and not a painting ────────────────────────────────
 *
 * It used to be a canvas texture on a single plane behind the glass, on the
 * argument that a backdrop seen through a window from across a room never needs
 * to be anything else. That argument has one hole in it, and the hole is the
 * camera: this world's camera orbits. It is slow indoors — the room's
 * orbitScale is a tenth — but it never stops, so given a few minutes you look at
 * the window from an angle, and a painted backdrop pinned to the wall behind the
 * glass does not move when you do. Everything else in the scene has parallax;
 * the one thing claiming to be a hundred metres away has none, which reads as a
 * poster rather than as a distance.
 *
 * So it is built now: towers, headlands, treelines and ranges standing in real
 * space beyond the wall. The room keeps its own fog and the scene's sky sphere
 * closes the back, which is what the painting was mostly doing anyway.
 *
 * ── The one thing the painting had that this has to earn back ──────────────
 *
 * Atmospheric perspective. A painter gets it for free by choosing paler colours
 * for the far layer; geometry has to be told. Every instance here is coloured by
 * its own distance, mixed toward the scene's haze — which is the same colour the
 * fog uses, so near and far agree — and that single lerp is most of the reason
 * the result reads as depth rather than as a pile of boxes at one remove.
 *
 * ── Elevation ──────────────────────────────────────────────────────────────
 *
 * The outside sits well below the floor. A room whose window looks out level
 * with the ground is a bungalow; a room looking *down* on rooftops is the flat
 * or the office this biome is chosen for, and it costs one constant.
 *
 * Deterministic from the seed, like every other generated thing in this world —
 * the view from your window is the same view tomorrow.
 */

import * as THREE from 'three';

export type ViewKind = 'city' | 'coast' | 'forest' | 'peaks' | 'meadow';

/**
 * Nothing stands closer than this.
 *
 * Not set from the room's size — the room is only 26 across — but from the
 * *camera's orbit*. The mind rig sits at radius 24 and the wheel can dolly it
 * out to about 38, so anything nearer than that ends up between the viewer and
 * the room: the first build put towers at 19 and they stood in front of the
 * doll's house like pillars in a doorway. The view has to begin outside the
 * furthest the camera can ever get.
 */
const INNER = 44;
const OUTER = 120;

/**
 * How far below the floor the outside world sits.
 *
 * The number that makes this a room with a view rather than a shed with a
 * window. Deep enough that you are plainly above what you are looking at,
 * shallow enough that the nearer rooftops still rise past the sill.
 */
const DROP = 20;

export class WindowView {
  readonly group = new THREE.Group();

  private owned: Array<THREE.BufferGeometry | THREE.Material> = [];
  private meshes: THREE.Object3D[] = [];

  /**
   * @param sky    The scene's haze colour. Everything fades toward it with
   *               distance, so the view and the fog are the same weather.
   * @param accent The room's emotional colour, mixed in very lightly — enough
   *               that the outside belongs to the same world, not enough to look
   *               tinted.
   */
  rebuild(kind: ViewKind, sky: THREE.Color, accent: THREE.Color, seed: number): void {
    this.clear();
    const rng = makeRng(seed);

    switch (kind) {
      case 'city':
        this.buildGround(sky, new THREE.Color('#171b22'), 0.22);
        this.buildCity(rng, sky, accent);
        break;
      case 'coast':
        this.buildGround(sky, new THREE.Color('#1c4257').lerp(accent, 0.1), 0.22);
        this.buildRidges(rng, sky, new THREE.Color('#5d6b78'), 3, 14, 0.5);
        break;
      case 'forest':
        this.buildGround(sky, new THREE.Color('#18220f'), 0.2);
        this.buildTrees(rng, sky, new THREE.Color('#1f3326'));
        break;
      case 'peaks':
        this.buildGround(sky, new THREE.Color('#252b38'), 0.25);
        this.buildRidges(rng, sky, new THREE.Color('#3b4459'), 5, 38, 0.24);
        break;
      case 'meadow':
        this.buildGround(sky, new THREE.Color('#3d4826').lerp(accent, 0.12), 0.24);
        this.buildRidges(rng, sky, new THREE.Color('#5c6b3f'), 4, 12, 0.55);
        this.buildTrees(rng, sky, new THREE.Color('#3c5130'), 0.35);
        break;
    }
  }

  /**
   * The floor of the outside world.
   *
   * Kept dark. It is the one surface out here with no distance falloff of its
   * own — a single flat plate running from just outside the camera's orbit to
   * past the fog — so hazing it the way everything else is hazed turns it into a
   * bright ring under the room, brighter than the towers standing on it. From a
   * high window you barely see the ground anyway.
   *
   * An annulus rather than a disc: the well drops through the middle of the room
   * to the memory orbs, and a solid plate out here would be a lid over it seen
   * from below during the descent.
   */
  private buildGround(sky: THREE.Color, base: THREE.Color, haze: number): void {
    const geometry = this.own(new THREE.RingGeometry(INNER - 6, OUTER + 30, 64, 1));
    geometry.rotateX(-Math.PI / 2);
    const material = this.own(
      new THREE.MeshStandardMaterial({
        color: base.clone().lerp(sky, haze),
        roughness: 1,
        metalness: 0,
        fog: true,
      })
    ) as THREE.MeshStandardMaterial;

    const plate = new THREE.Mesh(geometry, material);
    plate.position.y = -DROP;
    this.add(plate);
  }

  /**
   * A skyline, in one draw call.
   *
   * Two instanced meshes: the blocks, and a thin bright strip on the face of
   * each one standing in for lit windows. A grid of individual window quads
   * would be thirty times the geometry to produce something that, at this
   * distance, is a smear of light either way — which is exactly the trade
   * props.ts makes for the city prop on the island.
   */
  private buildCity(rng: Rng, sky: THREE.Color, accent: THREE.Color): void {
    const COUNT = 300;
    const block = this.own(new THREE.BoxGeometry(1, 1, 1));
    const concrete = this.own(
      new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0.1, fog: true })
    ) as THREE.MeshStandardMaterial;
    const glow = this.own(
      new THREE.MeshBasicMaterial({
        color: new THREE.Color('#ffe6b4').lerp(accent, 0.3),
        transparent: true,
        opacity: 0.5,
        fog: true,
      })
    ) as THREE.MeshBasicMaterial;

    const towers = new THREE.InstancedMesh(block, concrete, COUNT);
    const lights = new THREE.InstancedMesh(block, glow, COUNT);
    const dummy = new THREE.Object3D();
    const tone = new THREE.Color();
    const body = new THREE.Color('#4a5468');

    for (let i = 0; i < COUNT; i++) {
      const angle = rng() * Math.PI * 2;
      const r = INNER + Math.sqrt(rng()) * (OUTER - INNER);
      const x = Math.cos(angle) * r;
      const z = Math.sin(angle) * r;

      const w = 5 + rng() * 7;
      const d = 5 + rng() * 7;
      // Taller nearer the middle, so the skyline has a centre rather than being
      // an even field of blocks — every city does this and it is most of what
      // makes a skyline read as one.
      const h = (14 + rng() * 34) * (1.25 - ((r - INNER) / (OUTER - INNER)) * 0.5);

      dummy.position.set(x, -DROP + h / 2, z);
      dummy.scale.set(w, h, d);
      dummy.rotation.y = rng() * Math.PI * 0.5;
      dummy.updateMatrix();
      towers.setMatrixAt(i, dummy.matrix);
      towers.setColorAt(i, fade(tone, body, sky, r));

      // The lit strip, on the face that looks back toward the room.
      const facing = Math.atan2(-x, -z);
      dummy.position.set(x, -DROP + h * 0.52, z);
      dummy.scale.set(w * 0.34, h * 0.72, d * 0.34);
      dummy.rotation.y = facing;
      dummy.updateMatrix();
      lights.setMatrixAt(i, dummy.matrix);
    }
    if (towers.instanceColor) towers.instanceColor.needsUpdate = true;
    this.add(towers, lights);
  }

  /** A treeline of cones, thinning with distance. */
  private buildTrees(rng: Rng, sky: THREE.Color, base: THREE.Color, density = 1): void {
    const COUNT = Math.round(900 * density);
    const cone = this.own(new THREE.ConeGeometry(0.5, 1, 6));
    const material = this.own(
      new THREE.MeshStandardMaterial({ roughness: 1, metalness: 0, flatShading: true, fog: true })
    ) as THREE.MeshStandardMaterial;

    const trees = new THREE.InstancedMesh(cone, material, COUNT);
    const dummy = new THREE.Object3D();
    const tone = new THREE.Color();

    for (let i = 0; i < COUNT; i++) {
      const angle = rng() * Math.PI * 2;
      const r = INNER + Math.sqrt(rng()) * (OUTER - INNER);
      const x = Math.cos(angle) * r;
      const z = Math.sin(angle) * r;
      const h = 13 + rng() * 15;
      const w = h * (0.3 + rng() * 0.16);

      dummy.position.set(x, -DROP + h / 2, z);
      dummy.scale.set(w, h, w);
      dummy.rotation.y = rng() * Math.PI;
      dummy.updateMatrix();
      trees.setMatrixAt(i, dummy.matrix);
      trees.setColorAt(i, fade(tone, base, sky, r));
    }
    if (trees.instanceColor) trees.instanceColor.needsUpdate = true;
    this.add(trees);
  }

  /**
   * Ranges: rings of broad cones at increasing distance.
   *
   * Each ring is further away and hazier than the one in front of it, which is
   * the same trick the painting used and the only one that reads as *scale*
   * rather than as size.
   */
  private buildRidges(
    rng: Rng,
    sky: THREE.Color,
    base: THREE.Color,
    rings: number,
    height: number,
    haze: number
  ): void {
    const cone = this.own(new THREE.ConeGeometry(0.5, 1, 5));
    const material = this.own(
      new THREE.MeshStandardMaterial({ roughness: 1, metalness: 0, flatShading: true, fog: true })
    ) as THREE.MeshStandardMaterial;

    const perRing = 22;
    const mesh = new THREE.InstancedMesh(cone, material, rings * perRing);
    const dummy = new THREE.Object3D();
    const tone = new THREE.Color();

    let i = 0;
    for (let ring = 0; ring < rings; ring++) {
        const r0 = INNER + 10 + (ring / rings) * (OUTER - INNER - 10);
      for (let k = 0; k < perRing; k++) {
        const angle = (k / perRing) * Math.PI * 2 + rng() * 0.3;
        const r = r0 + rng() * 10;
        const h = height * (0.55 + rng() * 0.9) * (1 + ring * 0.25);
        const w = h * (2.2 + rng() * 1.6);

        dummy.position.set(Math.cos(angle) * r, -DROP + h / 2 - 1, Math.sin(angle) * r);
        dummy.scale.set(w, h, w);
        dummy.rotation.y = rng() * Math.PI;
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
        // Hazier with every ring back, on top of the distance fade.
        mesh.setColorAt(i, fade(tone, base, sky, r, haze + (ring / rings) * 0.3));
        i++;
      }
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    this.add(mesh);
  }

  // -- plumbing ---------------------------------------------------------

  private add(...objects: THREE.Object3D[]): void {
    for (const object of objects) {
      // The view is a ring seventy units across whose bounding sphere is centred
      // on the room; leaving it to the frustum test costs nothing and gets it
      // wrong at the edges of a wide window.
      object.frustumCulled = false;
      this.group.add(object);
      this.meshes.push(object);
    }
  }

  private own<T extends THREE.BufferGeometry | THREE.Material>(thing: T): T {
    this.owned.push(thing);
    return thing;
  }

  clear(): void {
    for (const mesh of this.meshes) {
      this.group.remove(mesh);
      (mesh as THREE.InstancedMesh).dispose?.();
    }
    this.meshes = [];
    for (const thing of this.owned) thing.dispose();
    this.owned = [];
  }

  dispose(): void {
    this.clear();
  }
}

/**
 * Atmospheric perspective, as one lerp.
 *
 * The far things are not smaller versions of the near things — they are paler,
 * and pallor is what the eye actually reads distance from. Without this the view
 * is a field of identically-coloured shapes and no amount of correct perspective
 * makes it look deep.
 */
function fade(
  out: THREE.Color,
  base: THREE.Color,
  sky: THREE.Color,
  radius: number,
  extra = 0
): THREE.Color {
  const t = Math.min(1, Math.max(0, (radius - INNER) / (OUTER - INNER)));
  // The floor of this is what stops the far field reading as a row of black
  // cut-outs: even the nearest thing out here is a long way off, and nothing a
  // long way off is ever its own colour.
  return out.copy(base).lerp(sky, Math.min(0.93, 0.34 + t * 0.5 + extra));
}

type Rng = () => number;

function makeRng(seed: number): Rng {
  let a = (seed | 0) >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
