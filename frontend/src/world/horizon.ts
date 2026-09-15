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
import { radialGrid } from './grid';
import {
  ISLAND_RADIUS,
  alongGrain,
  bandWeights,
  earthen,
  elevation,
  gradientAt,
  gradientPalette,
  groundColorAt,
  makeGroundMaterial,
  type TerrainShape,
} from './terrain';

/** Starts inside the island's outer edge, so the two meshes overlap. */
const INNER_RADIUS = ISLAND_RADIUS * 1.03;
/** Kept inside the sky sphere's radius, or the sky paints over the peaks. */
const OUTER_RADIUS = 152;

/**
 * Tessellation of the far field.
 *
 * Coarser along the line of sight than across it, which `radialGrid` expresses
 * as an aspect above 1: the backdrop is seen almost edge-on from a camera nine
 * units off the floor, so a quad that is a third longer radially than it is wide
 * still projects to something square-ish on screen — and it buys back most of
 * the triangles that squaring it up would have cost.
 */
const FAR_SPOKES = 192;
const FAR_ASPECT = 1.3;

/** Where each range crests, as a radius. */
const NEAR_RANGE = 64;
const FAR_RANGE = 112;

/** Over this band the far field stops being the island and starts being scenery. */
const BLEND_FROM = 16;
const BLEND_TO = 36;

const ROCK_DARK = new THREE.Color('#333b4d');
const SNOW = new THREE.Color('#d7dfee');
const TREE_DARK = new THREE.Color('#253a2b');
/** The light end of the canopy. See the per-instance tone in `buildTrees`. */
const TREE_PALE = new THREE.Color('#5c7345');

/** Woodland: how many trees at full strength, and the band they stand in. */
const TREE_COUNT = 900;
const TREE_INNER = 17;
const TREE_OUTER = 74;

export class Horizon {
  readonly group = new THREE.Group();

  private mesh: THREE.Mesh | null = null;
  private material = makeGroundMaterial({
    // The far field runs to 152 units, so its detail has to survive a good deal
    // further out than the island's before fading.
    detailNear: 60,
    detailFar: 165,
    // This mesh deliberately overlaps the island's rim, and the two agree about
    // the height there to within their own tessellation — which is exactly the
    // condition for z-fighting. Biasing the backdrop back a fraction settles it
    // in the island's favour, everywhere, for free.
    depthBias: 1.4,
  });
  private trees: THREE.InstancedMesh | null = null;

  /** Kept so ground cover can ask where the ground is past the island's edge. */
  private shape: TerrainShape | null = null;
  private peak = 0;
  /** How wooded the far field is, in [0, 1]. See `buildTrees`. */
  private woodland = 0;

  /**
   * @param prominence How much high ground the diary talks about, in [0, 1].
   *                   Raises the ranges and pulls the snow line down.
   * @param woodland   How much woodland the diary talks about, in [0, 1]. Sets
   *                   how thickly the far field is forested; a treeline biome
   *                   ignores it and is fully wooded by definition.
   */
  rebuild(
    shape: TerrainShape,
    prominence: number,
    lifetimeTotals: EmotionVector,
    woodland = 0
  ): void {
    this.clear();
    this.shape = shape;
    this.woodland = clamp01(woodland);
    if (shape.biome.backdrop === 'none') return;

    const geometry = radialGrid({
      inner: INNER_RADIUS,
      outer: OUTER_RADIUS,
      spokes: FAR_SPOKES,
      aspect: FAR_ASPECT,
      jitter: 0.32,
      seed: shape.seed + 71,
    });

    const position = geometry.attributes.position as THREE.BufferAttribute;
    const colors = new Float32Array(position.count * 3);
    const bands = bandWeights(lifetimeTotals);
    const palette = gradientPalette();
    const scratch = new THREE.Color();
    const tinted = new THREE.Color();

    const peak = 26 + prominence * 30;
    this.peak = peak;
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
      // world as the island rather than being grey scenery behind it. Tempered
      // against the ground the same way the island tempers it, or the distance
      // is the one place in the world where the palette shows its real strength.
      gradientAt(colorField(x, z, shape.seed), bands, palette, tinted);
      earthen(tinted, ground);
      scratch.lerp(tinted, 0.2);

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
    // Smooth, and sharing the island's material.
    //
    // This mesh used to be flat-shaded on the theory that facets read as crags
    // at range. They did not: with evenly-spaced rings the facets were radial
    // splinters four units long, so what they actually read as was a pinwheel
    // of rays fanning off the island — and being a different material from the
    // island as well, the ground visibly changed substance at the seam. Square
    // quads plus the island's own surface detail gives the crags back without
    // either.
    geometry.computeVertexNormals();

    this.mesh = new THREE.Mesh(geometry, this.material);
    this.group.add(this.mesh);

    this.buildTrees(shape, peak);
  }

  /**
   * Ground height out here, for anything that has to stand on it.
   *
   * Inside the blend band this is the island's own `elevation`, so callers can
   * use it at any radius without having to know where one mesh stops.
   */
  heightAt(x: number, z: number): number {
    if (!this.shape) return 0;
    return farHeight(x, z, this.shape, this.peak);
  }

  /**
   * The woodland closing the distance.
   *
   * One InstancedMesh of cones — up to nine hundred trees in a single draw
   * call, which is the only reason this is affordable at all. They are
   * silhouettes and little more: at this distance what survives is the outline
   * of a canopy, and detail nobody can resolve is detail that only costs frames.
   * The one thing worth spending on is that the cones are not all the same
   * cone — a uniform height and taper reads as a texture, and a scatter of
   * proportions reads as a wood.
   *
   * Every biome can have some, not just the treeline. Somebody who walks in the
   * woods every weekend but talks about the sea slightly more should still get
   * trees on the far shore; a bare hillside behind them would be the landscape
   * quietly discarding half of what they said.
   */
  private buildTrees(shape: TerrainShape, peak: number): void {
    const woodland = shape.biome.backdrop === 'treeline' ? 1 : this.woodland;
    const wanted = Math.round(TREE_COUNT * woodland);
    if (wanted < 12) return;

    const geometry = new THREE.ConeGeometry(0.5, 1, 6);
    const material = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 1,
      metalness: 0,
      flatShading: true,
      fog: true,
    });
    const mesh = new THREE.InstancedMesh(geometry, material, wanted);
    const dummy = new THREE.Object3D();
    // Per-instance colour, which is free on an InstancedMesh and is the only
    // thing standing between "a wood" and "a picket fence". Nine hundred cones
    // of one colour read as a single cut-out shape however much their heights
    // vary, because the eye separates them by tone long before it separates
    // them by outline.
    const tone = new THREE.Color();

    // Deterministic, like every other placement in this world: the wood has to
    // be the same wood next time you open the app.
    let state = (shape.seed ^ 0x9e3779b9) >>> 0;
    const rand = (): number => {
      state = (Math.imul(state ^ (state >>> 15), 1 | state) + 0x6d2b79f5) >>> 0;
      return (state >>> 8) / 16777216;
    };

    // Trees grow in stands, not in an even sprinkle. Accepting a candidate more
    // readily where a slow noise field is high is the cheapest way to get
    // clearings and thickets instead of the airport-lawn look a uniform scatter
    // gives, and it costs one fBm sample per attempt.
    const stand = (x: number, z: number): number =>
      clamp01((fbm(x * 0.055 + 13.3, z * 0.055 - 5.9, shape.seed + 61) - 0.36) / 0.24);

    let placed = 0;
    for (let attempt = 0; attempt < wanted * 6 && placed < wanted; attempt++) {
      const angle = rand() * Math.PI * 2;
      // sqrt keeps the density even in area rather than crowding the inner edge.
      const radius = TREE_INNER + Math.sqrt(rand()) * (TREE_OUTER - TREE_INNER);
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const y = farHeight(x, z, shape, peak);

      // Nothing grows underwater, and nothing grows on the peaks.
      if (shape.waterLevel !== null && y < shape.waterLevel + 0.2) continue;
      if (y > peak * 0.5) continue;
      // The wood thins toward the island rather than beginning at a line. A
      // hard inner radius puts a wall of trunks at a fixed distance all the way
      // round the compass, which is the one arrangement no wood has ever had.
      if (rand() > smoothstep(TREE_INNER, TREE_INNER + 12, radius)) continue;
      if (rand() > 0.12 + 0.88 * stand(x, z)) continue;

      // Two silhouettes rather than one: a narrow spire and a squatter, rounder
      // crown. Still one geometry — the difference is entirely in the taper,
      // which is all a cone has to give and all that reads at this range.
      const spire = rand() < 0.62;
      const height = spire ? 2.6 + rand() * 3.2 : 2.0 + rand() * 2.2;
      const width = height * (spire ? 0.26 + rand() * 0.1 : 0.46 + rand() * 0.2);
      dummy.position.set(x, y + height * 0.5 - 0.15, z);
      dummy.scale.set(width, height, width);
      dummy.rotation.y = rand() * Math.PI;
      dummy.updateMatrix();
      mesh.setMatrixAt(placed, dummy.matrix);

      // Lighter and yellower toward the top of the range, which is roughly what
      // a mixed wood does and, more usefully, is what stops the canopy reading
      // as one flat shape.
      tone
        .copy(TREE_DARK)
        .lerp(TREE_PALE, rand() * rand())
        .multiplyScalar(0.82 + rand() * 0.4);
      mesh.setColorAt(placed, tone);
      placed++;
    }

    // Unused instances would otherwise render at the origin as a pile of cones
    // standing in the well.
    mesh.count = placed;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    this.trees = mesh;
    this.group.add(mesh);
  }

  /** Torn down and rebuilt each time; the material outlives both. */
  clear(): void {
    if (this.mesh) {
      this.group.remove(this.mesh);
      this.mesh.geometry.dispose();
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

  dispose(): void {
    this.clear();
    this.material.dispose();
    this.shape = null;
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

  // Two scales of undulation rather than one.
  //
  // A single low octave gives the far field one wavelength, and one wavelength
  // over a hundred units is a dome — which is what the distance used to be,
  // however much relief it was given. The finer octave is what the middle
  // distance is actually made of: it is too small to change the silhouette and
  // too large to be mistaken for texture, which is exactly the band the eye uses
  // to judge how far away something is.
  const broad = fbm(x * 0.035, z * 0.035, shape.seed + 17) * 2 - 1;
  const fine = fbm(x * 0.105, z * 0.105, shape.seed + 53) * 2 - 1;

  if (shape.biome.backdrop !== 'peaks') {
    const rise = shape.biome.backdrop === 'rolling' ? 5.5 : 3.2;
    return plain + (broad * rise + fine * rise * 0.32) * clearance;
  }

  const near = band(r, NEAR_RANGE, 22) * ridged(x * 0.03, z * 0.03, shape.seed + 7);
  const far = band(r, FAR_RANGE, 44) * ridged(x * 0.017, z * 0.017, shape.seed + 91);
  // Foothills. A range that rises straight out of a flat plain is a cut-out
  // standing on a table; the ground has to start climbing before it.
  const foot = band(r, 36, 30) * ridged(x * 0.058, z * 0.058, shape.seed + 29) * peak * 0.16;
  // Ranges only on land. A mountain rising out of the open sea on the seaward
  // side would undo the composition the grain just bought.
  const overLand = shape.biome.edge === 'shore' ? landward : 1;
  // Max rather than sum, so the near range occludes the far one instead of
  // averaging with it into a single lumpy mass.
  return (
    plain +
    fine * 2.6 * clearance +
    (Math.max(near * peak * 0.46, far * peak) + foot * clearance) * overLand
  );
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
