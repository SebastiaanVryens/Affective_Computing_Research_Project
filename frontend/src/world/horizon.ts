/**
 * Everything past the ground you are standing on.
 *
 * One ring of geometry running from the island's rim out to the haze, plus —
 * for woodland — a field of trees standing on it. What it *does* depends
 * entirely on the biome:
 *
 *   peaks    — ranges rising out of the distance, in two layers so the
 *              silhouette has depth. Coast and high ground.
 *   treeline — ground that carries on, closed off by forest.
 *   rolling  — open country, low hills, nothing asserted.
 *   none     — not built at all.
 *
 * It exists because the island alone is not a picture. A piece of ground with
 * nothing behind it has nothing to be *in front of*: no sense of distance,
 * nothing for the haze to act on, and a horizon that is simply where the mesh
 * stops. Putting land out there gives the eye something to measure the scene
 * against, and gives the fog a job.
 *
 * The inner third of the ring is cross-faded into `elevation()` itself, so for
 * an inland biome the ground genuinely carries on past the island's edge rather
 * than meeting a second mesh at a seam. For a coast it sinks into the sea floor,
 * which is the same function saying something different.
 *
 * It still answers to the diary, though faintly: the range rises and takes snow
 * when the person talks about high ground, and the rock carries the same
 * lifetime colour the island and the lower sky do. A backdrop that ignored the
 * data would be the one part of this world that wasn't theirs.
 */

import * as THREE from 'three';
import type { EmotionVector } from '../emotions';
import {
  ISLAND_RADIUS,
  alongGrain,
  bandWeights,
  elevation,
  gradientAt,
  gradientPalette,
  groundColorAt,
  type TerrainShape,
} from './terrain';

/** Starts inside the island's outer edge, so the two meshes overlap. */
const INNER_RADIUS = ISLAND_RADIUS * 1.03;
/** Kept inside the sky sphere's radius, or the sky paints over the peaks. */
const OUTER_RADIUS = 152;

/** Where each range crests, as a radius. */
const NEAR_RANGE = 64;
const FAR_RANGE = 112;

/** Over this band the far field stops being the island and starts being scenery. */
const BLEND_FROM = 16;
const BLEND_TO = 36;

const ROCK_DARK = new THREE.Color('#333b4d');
const SNOW = new THREE.Color('#d7dfee');
const TREE_DARK = new THREE.Color('#253a2b');

/** Woodland: how many trees, and the band they stand in. */
const TREE_COUNT = 620;
const TREE_INNER = 16;
const TREE_OUTER = 62;

export class Horizon {
  readonly group = new THREE.Group();

  private mesh: THREE.Mesh | null = null;
  private trees: THREE.InstancedMesh | null = null;

  /**
   * @param prominence How much high ground the diary talks about, in [0, 1].
   *                   Raises the ranges and pulls the snow line down.
   */
  rebuild(shape: TerrainShape, prominence: number, lifetimeTotals: EmotionVector): void {
    this.dispose();
    if (shape.biome.backdrop === 'none') return;

    const geometry = new THREE.RingGeometry(INNER_RADIUS, OUTER_RADIUS, 200, 34);
    geometry.rotateX(-Math.PI / 2);

    const position = geometry.attributes.position as THREE.BufferAttribute;
    const colors = new Float32Array(position.count * 3);
    const bands = bandWeights(lifetimeTotals);
    const palette = gradientPalette();
    const scratch = new THREE.Color();
    const tinted = new THREE.Color();

    const peak = 26 + prominence * 30;
    // Measured the same way the island measures it — absolute height, not
    // height above water. The two used different datums, so on a biome with a
    // low snow line the far field snowed everything above y = 0.4 and met the
    // island's own snow at a hard edge.
    const peakSnow = peak * (prominence > 0.4 ? 0.5 : 0.8);
    const snowLine = Math.min(shape.biome.snowAbove, peakSnow);
    const ground = new THREE.Color(shape.biome.ground.base);
    const rockHue = new THREE.Color(shape.biome.ground.rock);

    for (let i = 0; i < position.count; i++) {
      const x = position.getX(i);
      const z = position.getZ(i);
      const y = farHeight(x, z, shape, peak);
      position.setY(i, y);

      const above = y - (shape.waterLevel ?? 0);
      // Low ground keeps the biome's own colour so the far field reads as the
      // same country; only what actually rises becomes rock and snow.
      const rise = clamp01((above - 2) / (peak * 0.5));
      scratch.copy(ground).lerp(ROCK_DARK, rise * 0.75).lerp(rockHue, rise * 0.35);
      if (shape.biome.backdrop === 'treeline') scratch.lerp(TREE_DARK, 0.3);
      if (y > snowLine) {
        scratch.lerp(SNOW, clamp01((y - snowLine) / (peak * 0.22)) * 0.9);
      }

      // A breath of the diary's own colour, so the far land belongs to the same
      // world as the island rather than being grey scenery behind it.
      gradientAt(colorField(x, z, shape.seed), bands, palette, tinted);
      scratch.lerp(tinted, 0.16);

      // Over the inner band this mesh overlaps the island, so it borrows the
      // island's exact colour there and only becomes scenery further out. The
      // height already cross-fades the same way; without the colour doing it too
      // there is a hard ring at the overlap, which is the tell that gives away
      // that the ground and the distance are two different objects.
      const r = Math.hypot(x, z);
      if (r < BLEND_TO) {
        groundColorAt(x, z, y, shape, bands, palette, tinted);
        scratch.lerp(tinted, 1 - smoothstep(BLEND_FROM, BLEND_TO, r));
      }

      colors[i * 3] = scratch.r;
      colors[i * 3 + 1] = scratch.g;
      colors[i * 3 + 2] = scratch.b;
    }

    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    this.mesh = new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 1,
        metalness: 0,
        // Flat here, unlike the island. At this range the facets are smaller
        // than the haze's own grain and read as crags rather than as polygons,
        // and smooth shading would turn the whole range into a soft mound.
        flatShading: true,
        fog: true,
      })
    );
    this.group.add(this.mesh);

    if (shape.biome.backdrop === 'treeline') this.buildTrees(shape, peak);
  }

  /**
   * The woodland closing the distance.
   *
   * One InstancedMesh of cones — six hundred trees in a single draw call, which
   * is the only reason a treeline is affordable at all. They are silhouettes and
   * nothing more: no trunks, no variation in shape, because at this distance the
   * only thing that survives is the outline of the canopy, and detail nobody can
   * resolve is detail that only costs frames.
   */
  private buildTrees(shape: TerrainShape, peak: number): void {
    const geometry = new THREE.ConeGeometry(0.5, 1, 6);
    const material = new THREE.MeshStandardMaterial({
      color: TREE_DARK,
      roughness: 1,
      metalness: 0,
      flatShading: true,
      fog: true,
    });
    const mesh = new THREE.InstancedMesh(geometry, material, TREE_COUNT);
    const dummy = new THREE.Object3D();

    // Deterministic, like every other placement in this world: the wood has to
    // be the same wood next time you open the app.
    let state = (shape.seed ^ 0x9e3779b9) >>> 0;
    const rand = (): number => {
      state = (Math.imul(state ^ (state >>> 15), 1 | state) + 0x6d2b79f5) >>> 0;
      return (state >>> 8) / 16777216;
    };

    let placed = 0;
    for (let attempt = 0; attempt < TREE_COUNT * 3 && placed < TREE_COUNT; attempt++) {
      const angle = rand() * Math.PI * 2;
      // sqrt keeps the density even in area rather than crowding the inner edge.
      const radius = TREE_INNER + Math.sqrt(rand()) * (TREE_OUTER - TREE_INNER);
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const y = farHeight(x, z, shape, peak);

      // Nothing grows underwater, and nothing grows on the peaks.
      if (shape.waterLevel !== null && y < shape.waterLevel + 0.2) continue;
      if (y > peak * 0.5) continue;

      const height = 2.4 + rand() * 3.4;
      const width = height * (0.3 + rand() * 0.12);
      dummy.position.set(x, y + height * 0.5 - 0.15, z);
      dummy.scale.set(width, height, width);
      dummy.rotation.y = rand() * Math.PI;
      dummy.updateMatrix();
      mesh.setMatrixAt(placed, dummy.matrix);
      placed++;
    }

    // Unused instances would otherwise render at the origin as a pile of cones
    // standing in the well.
    mesh.count = placed;
    this.trees = mesh;
    this.group.add(mesh);
  }

  dispose(): void {
    if (this.mesh) {
      this.group.remove(this.mesh);
      this.mesh.geometry.dispose();
      (this.mesh.material as THREE.Material).dispose();
      this.mesh = null;
    }
    if (this.trees) {
      this.group.remove(this.trees);
      this.trees.geometry.dispose();
      (this.trees.material as THREE.Material).dispose();
      this.trees.dispose();
      this.trees = null;
    }
  }
}

/**
 * Height of the far field at a point.
 *
 * Near the island this *is* the island's own height function, so the two meshes
 * share a surface and there is no seam to find. Past the blend band it becomes
 * whatever the biome's backdrop is.
 */
function farHeight(x: number, z: number, shape: TerrainShape, peak: number): number {
  const r = Math.hypot(x, z);
  const near = elevation(x, z, shape);
  const t = smoothstep(BLEND_FROM, BLEND_TO, r);
  if (t <= 0) return near;
  return near * (1 - t) + backdrop(x, z, r, shape, peak) * t;
}

function backdrop(
  x: number,
  z: number,
  r: number,
  shape: TerrainShape,
  peak: number
): number {
  // Nothing rises until well past the camera.
  //
  // The camera orbits at radius 24, which for an inland biome is *inside* this
  // mesh. Terrain allowed to roll freely from the island's edge outward
  // therefore rises between the viewer and the thing they are looking at, and
  // the mind view becomes a hillside two metres from the lens. Holding the far
  // field flat until it is safely outside the orbit is what keeps the ground
  // something you look at rather than something you are buried in.
  const clearance = smoothstep(30, 58, r);

  // Which side of the grain we are on: 1 fully uphill, 0 fully downhill. For a
  // coast this is literally "how far inland", and it is what turns the far field
  // from a ring of sea around an island into a landmass on one side of a
  // shoreline — with the ranges only rising over the land.
  const landward = 1 - smoothstep(-0.5, 1.1, alongGrain(x, z, shape));

  // Where the far field sits before anything rises out of it: under the sea on
  // the seaward side of a coast, level with the ground anywhere inland.
  const inland = 1 + (fbm(x * 0.02, z * 0.02, shape.seed + 5) * 2 - 1) * 3.2 * clearance;
  const plain =
    shape.biome.edge === 'shore'
      ? ((shape.waterLevel ?? 0) - 9) * (1 - landward) + (inland + 1.2) * landward
      : inland - shape.biome.tilt * clamp(alongGrain(x, z, shape), -2.6, 2.6) * 0.6;

  if (shape.biome.backdrop !== 'peaks') {
    const rise = shape.biome.backdrop === 'rolling' ? 5.5 : 3.2;
    return (
      plain +
      (fbm(x * 0.035, z * 0.035, shape.seed + 17) * 2 - 1) * rise * clearance
    );
  }

  const near = band(r, NEAR_RANGE, 22) * ridged(x * 0.03, z * 0.03, shape.seed + 7);
  const far = band(r, FAR_RANGE, 44) * ridged(x * 0.017, z * 0.017, shape.seed + 91);
  // Ranges only on land. A mountain rising out of the open sea on the seaward
  // side would undo the composition the grain just bought.
  const overLand = shape.biome.edge === 'shore' ? landward : 1;
  // Max rather than sum, so the near range occludes the far one instead of
  // averaging with it into a single lumpy mass.
  return plain + Math.max(near * peak * 0.46, far * peak) * overLand;
}

/** A smooth 0..1 hump centred on `centre`, `width` wide at the base. */
function band(r: number, centre: number, width: number): number {
  const t = clamp01(1 - Math.abs(r - centre) / width);
  return t * t * (3 - 2 * t);
}

/**
 * Ridged noise — the absolute value of signed noise, inverted.
 *
 * Plain fBm gives rounded hills, and the thing that reads as "mountain" at
 * silhouette distance is a sharp crest line.
 */
function ridged(x: number, y: number, seed: number): number {
  const n = fbm(x, y, seed) * 2 - 1;
  return Math.pow(clamp01(1 - Math.abs(n)), 2.1);
}

/** Broad field deciding which of the diary's colours tints which slope. */
function colorField(x: number, z: number, seed: number): number {
  return clamp01((fbm(x * 0.012 + 3.1, z * 0.012 - 7.7, seed + 23) - 0.3) / 0.4);
}

// ---------------------------------------------------------------------------
// Noise — the same shape terrain.ts uses, at this module's own frequencies.
// ---------------------------------------------------------------------------

function hash2(x: number, y: number, seed: number): number {
  const h = Math.sin(x * 127.1 + y * 311.7 + seed * 0.4871) * 43758.5453;
  return h - Math.floor(h);
}

function noise2(x: number, y: number, seed: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);

  const a = hash2(ix, iy, seed);
  const b = hash2(ix + 1, iy, seed);
  const c = hash2(ix, iy + 1, seed);
  const d = hash2(ix + 1, iy + 1, seed);

  return (a + (b - a) * ux) * (1 - uy) + (c + (d - c) * ux) * uy;
}

function fbm(x: number, y: number, seed: number): number {
  return (
    noise2(x, y, seed) * 0.52 +
    noise2(x * 2.07 + 5.2, y * 2.07 - 1.3, seed) * 0.28 +
    noise2(x * 4.13 - 2.7, y * 4.13 + 8.1, seed) * 0.13 +
    noise2(x * 8.3 + 1.9, y * 8.3 - 4.4, seed) * 0.07
  );
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function clamp(x: number, min: number, max: number): number {
  return x < min ? min : x > max ? max : x;
}
