/**
 * The surface in the mouth of the well.
 *
 * The hole in the middle of the island had no geometry at all. The ground is an
 * annulus, so its inner edge is a free edge with nothing below it, and looking
 * into the well meant looking at whatever the renderer had cleared the frame to
 * — a lightless ellipse with a hard rim, which reads as something covering the
 * hole rather than as the hole itself.
 *
 * The obvious answer is to show the memory orbs through it. That cannot work,
 * and the arithmetic is worth keeping so nobody tries again: the mind camera
 * sits at radius 24 and height 9, the well is 2.8 across, and the orbs are 78
 * units down. The cone of sight through that hole lands between 188 and 244
 * units off-axis at the orbs' depth. The galaxy reaches 40 units at four hundred
 * entries. The orbs are not hidden down there — they are nowhere near the line
 * of sight, and nothing about the sea or the ground changes that.
 *
 * So the opening gets a surface: a still, flat pool sitting in the mouth,
 * darkest at the rim and carrying the colour the diary's own emotions make
 * toward the middle. That colour is the one the orbs themselves are tinted with,
 * so what the pool holds is the light of the memories underneath rather than a
 * decorative wash.
 *
 * It was briefly a shaft — a tapering wall falling away into the dark — and the
 * reason that is gone is worth a line, because the shaft was not wrong so much
 * as it was a different thing. A pit in the middle of the island is a hazard the
 * eye keeps returning to; a pool is somewhere to look into. The way down is
 * unchanged either way: the camera flies through this, it does not land on it.
 *
 * Deliberately *not* the ring shaft this codebase used to have. See threads.ts:
 * those rings were a lift shaft drawn in light, they said "there is a way down"
 * and nothing more, and the memory threads replaced them because the threads say
 * something true instead.
 */

import * as THREE from 'three';
import { type EmotionVector, mixedColor } from '../emotions';
import { LAYER } from './layers';
import { WELL_RADIUS } from './terrain';

/**
 * How far below the rim the surface sits.
 *
 * Small, and it has to be: the ground is a single-sided annulus with no
 * underside, so any sight line that clears the far rim of the hole and passes
 * *beneath* the surface carries on through the terrain and out into the sky.
 * A surface set well down in the opening therefore opens a crescent of
 * background along its far edge, which is precisely the hole-with-nothing-in-it
 * this file exists to close.
 */
const DROP = 0.12;

/**
 * How far the surface reaches under the ground's inner edge.
 *
 * The other half of the same problem. For the far rim to be sealed, the surface
 * has to extend past the hole by at least the drop divided by the tangent of the
 * camera's elevation — and the mind camera sits at radius 24 and height 9, which
 * is about twenty-one degrees, so a 0.12 drop needs a third of a unit. Half a
 * unit leaves margin for the parallax and the wheel.
 *
 * Costless to overshoot: the terrain's rim rises steeply away from the well, so
 * the overhang is buried under the lip, and indoors it is inside the floorboards
 * (see room.ts). Undershooting is what shows.
 */
const SURFACE_RADIUS = WELL_RADIUS + 0.5;

/** Rings across the surface. Only the colour gradient needs them. */
const RINGS = 10;

export class Well {
  readonly group = new THREE.Group();

  private surface: THREE.Mesh | null = null;
  private surfaceMaterial: THREE.MeshBasicMaterial | null = null;
  private sheen: THREE.Mesh | null = null;
  private sheenMaterial: THREE.MeshBasicMaterial | null = null;

  /**
   * @param topY   Ground height at the well's rim. The surface hangs from it, so
   *               it follows the landform rather than assuming a flat island.
   * @param rock   What the ground is made of, for the dark outside of the pool —
   *               this is a cut through this island, not a bowl set into it.
   * @param mouth  The shape of the opening above. The island's is the round hole
   *               in an annulus; the room's is a square cut in floorboards (see
   *               room.ts), and a round surface under it leaks daylight at the
   *               four corners.
   */
  rebuild(
    topY: number,
    rock: string,
    lifetimeTotals: EmotionVector,
    mouth: 'round' | 'square' = 'round'
  ): void {
    this.clear();

    const light = new THREE.Color(mixedColor(lifetimeTotals));
    const square = mouth === 'square';

    // A four-segment ring is a square, so both mouths come off one code path.
    // The radius a ring takes is the circumradius, which for a square is its
    // half-width times root two — and it then needs an eighth-turn to put the
    // flats on the axes rather than the corners.
    const radius = square ? SURFACE_RADIUS * Math.SQRT2 : SURFACE_RADIUS;

    // -- the pool --------------------------------------------------------
    //
    // A ring from zero rather than a circle, because the gradient below wants
    // vertices between the middle and the edge and CircleGeometry has none.
    const geometry = new THREE.RingGeometry(0, radius, square ? 4 : 64, RINGS);
    geometry.rotateX(-Math.PI / 2);

    const position = geometry.attributes.position as THREE.BufferAttribute;
    const colors = new Float32Array(position.count * 3);
    // Near-black, but the island's own stone rather than a neutral: a pool takes
    // its darkness from what it is cut into.
    const dark = new THREE.Color(rock).multiplyScalar(0.11);

    for (let i = 0; i < position.count; i++) {
      // Measured against the *opening*, not against the mesh. The overhang above
      // is never seen, so letting it stretch the gradient would push the light
      // into a smaller and smaller spot for no visible reason.
      const u = Math.min(1, Math.hypot(position.getX(i), position.getZ(i)) / WELL_RADIUS);
      // Squared, so the light gathers in the middle instead of spreading evenly
      // across the surface — an even wash reads as a painted disc, which is the
      // one thing this must not be.
      const lit = (1 - u) * (1 - u) * 0.15;

      colors[i * 3] = dark.r + light.r * lit;
      colors[i * 3 + 1] = dark.g + light.g * lit;
      colors[i * 3 + 2] = dark.b + light.b * lit;
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Unlit and opaque. Nothing in this scene lights the inside of a well, so a
    // lit material would render it black whatever colour it was given; and being
    // opaque keeps it out of the transparent pass entirely, which is where the
    // sea's ordering problems live (see ./layers.ts).
    //
    // Front-facing only, which is also what makes it safe during the descent:
    // the camera drops past this on its way to the orbs, and a surface with no
    // back is simply gone the moment you are under it — the same way the ground
    // above it is.
    this.surfaceMaterial = new THREE.MeshBasicMaterial({
      vertexColors: true,
      side: THREE.FrontSide,
      fog: true,
    });

    this.surface = new THREE.Mesh(geometry, this.surfaceMaterial);
    this.surface.position.y = topY - DROP;
    if (square) this.surface.rotation.y = Math.PI / 4;
    this.group.add(this.surface);

    // -- the breath on it ------------------------------------------------
    //
    // A slow swell of light over the middle, so the pool is alive rather than
    // painted. Additive and faint, and kept under the bloom threshold in
    // scene.ts on purpose: a well mouth that blooms would put a halo over the
    // middle of the island, which is exactly where everything else is.
    this.sheenMaterial = new THREE.MeshBasicMaterial({
      color: light.clone().multiplyScalar(0.5),
      transparent: true,
      opacity: 0.12,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.FrontSide,
      // Additive surfaces must never be fogged: the haze would be mixed in and
      // then added, so the glow would brighten with distance.
      fog: false,
    });
    this.sheenMaterial.userData.baseOpacity = 0.12;

    this.sheen = new THREE.Mesh(
      new THREE.CircleGeometry(WELL_RADIUS * 0.6, 40),
      this.sheenMaterial
    );
    this.sheen.rotation.x = -Math.PI / 2;
    // A whisker above the pool, or the two z-fight across the whole middle.
    this.sheen.position.y = topY - DROP + 0.01;
    this.sheen.renderOrder = LAYER.glow;
    this.group.add(this.sheen);
  }

  /** A slow swell on the light, on the order of a breath. */
  update(elapsed: number): void {
    if (!this.sheenMaterial) return;
    const base = this.sheenMaterial.userData.baseOpacity as number;
    this.sheenMaterial.opacity = base * (0.72 + 0.28 * Math.sin(elapsed * 0.45));
  }

  clear(): void {
    if (this.surface) {
      this.group.remove(this.surface);
      this.surface.geometry.dispose();
      this.surface = null;
    }
    if (this.sheen) {
      this.group.remove(this.sheen);
      this.sheen.geometry.dispose();
      this.sheen = null;
    }
    this.surfaceMaterial?.dispose();
    this.sheenMaterial?.dispose();
    this.surfaceMaterial = null;
    this.sheenMaterial = null;
  }

  dispose(): void {
    this.clear();
  }
}
