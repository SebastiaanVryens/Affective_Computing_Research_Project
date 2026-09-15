/**
 * Radial ground meshes that don't fall apart away from the middle.
 *
 * Every piece of ground in this world is a disc or an annulus seen from a camera
 * parked at its centre, so the obvious thing to reach for is THREE.RingGeometry.
 * That was the original mistake, and it is worth writing down because the
 * symptom looks like a shading problem rather than a topology one.
 *
 * RingGeometry spaces its rings *evenly* in radius. The spacing between spokes,
 * though, is an arc — it grows with the radius. So a ring mesh has exactly one
 * radius at which its quads are square, and everywhere else they are stretched,
 * in one direction inside that radius and the other direction outside it. On the
 * island (r 2.8 → 14, 72 rings, 200 spokes) that gave quads three times wider
 * than they were deep at the rim. On the far field (r 13 → 152, 34 rings, 200
 * spokes) it gave the opposite and far worse: at the island's shoulder — which
 * is the part of the distance the camera actually looks across — each quad was
 * four units long and half a unit wide, an 8:1 splinter. Per-vertex colour
 * smeared along those splinters, and the horizon came out as a pinwheel of
 * coloured rays fanning off the island.
 *
 * The fix is to space the rings *geometrically* instead: r(i+1) = r(i) · k. Then
 * the gap between rings grows at the same rate the gap between spokes does, and
 * the quads have the same proportions at every radius — near-square at the well,
 * near-square at the horizon, and near-square everywhere in between. It also
 * puts the vertices where the detail is, which is the near ground, without
 * anybody having to nominate a "detail radius" and tune it.
 *
 * Two more things happen here, both aimed at the same thing: hiding the fact
 * that this is a polar grid at all.
 *
 *   Alternate rings are rotated half a step, so the vertices form a triangular
 *   lattice rather than lining up into radial spokes.
 *
 *   Every interior vertex is nudged by a fraction of its own local spacing.
 *   Noise on a regular grid still reads as a grid, because the eye finds the
 *   lattice and not the noise sitting on it; moving the lattice itself is what
 *   actually kills it.
 *
 * Both fade to nothing at the inner and outer rings, which have to stay exactly
 * circular: the inner one is the mouth of the well and the outer one has to meet
 * whatever mesh continues past it.
 */

import * as THREE from 'three';

export interface RadialGridOptions {
  /** Inner radius. Must be greater than zero — the spacing law is a ratio. */
  inner: number;
  outer: number;
  /** Vertices around the compass. Sets the tangential resolution outright. */
  spokes: number;
  /**
   * Ring spacing as a multiple of spoke spacing. 1 is square.
   *
   * Above 1 trades radial detail for triangle count, which is the right trade
   * for a backdrop: the far field is seen at a grazing angle, so it can afford
   * to be coarser along the line of sight than across it.
   */
  aspect?: number;
  /** How far a vertex may wander, as a fraction of its local spacing. */
  jitter?: number;
  seed?: number;
}

/**
 * A flat annulus in the XZ plane, ready to have its Y written per vertex.
 *
 * Returned flat rather than displaced because every caller computes height from
 * its own function and would immediately overwrite anything set here.
 */
export function radialGrid(options: RadialGridOptions): THREE.BufferGeometry {
  const { inner, outer, spokes } = options;
  const aspect = options.aspect ?? 1;
  const jitter = options.jitter ?? 0;
  const seed = options.seed ?? 1;

  const step = (Math.PI * 2) / spokes;

  // How many rings it takes to walk from inner to outer at the target aspect,
  // then the growth factor recomputed so the last ring lands exactly on `outer`
  // instead of just past it.
  const rings = Math.max(2, Math.ceil(Math.log(outer / inner) / Math.log(1 + aspect * step)) + 1);
  const growth = Math.pow(outer / inner, 1 / (rings - 1));

  const columns = spokes + 1; // the last column repeats the first, closing the seam
  const vertexCount = rings * columns;
  const positions = new Float32Array(vertexCount * 3);

  for (let ring = 0; ring < rings; ring++) {
    const radius = inner * Math.pow(growth, ring);
    // Local spacing, and how much of it a vertex is allowed to wander. Held at
    // zero on the two boundary rings so the mesh still meets its neighbours.
    const spacing = radius * step;
    const edge = ring === 0 || ring === rings - 1 ? 0 : 1;
    const wander = jitter * spacing * edge;
    // Half-step on odd rings: a triangular lattice instead of aligned spokes.
    const offset = (ring % 2) * step * 0.5;

    for (let spoke = 0; spoke < columns; spoke++) {
      // The seam column is the first column, not a vertex near it — it has to
      // be bit-identical or the jitter opens a crack at north.
      const source = spoke === spokes ? 0 : spoke;
      const angle = (source / spokes) * Math.PI * 2 + offset;

      const along = (hash(ring, source, seed) - 0.5) * 2 * wander;
      const across = (hash(ring, source, seed + 977) - 0.5) * 2 * wander;

      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      // Radial and tangential, so the nudge is the same size in both directions
      // regardless of where on the compass the vertex sits.
      const x = cos * (radius + along) - sin * across;
      const z = sin * (radius + along) + cos * across;

      const index = (ring * columns + spoke) * 3;
      positions[index] = x;
      positions[index + 1] = 0;
      positions[index + 2] = z;
    }
  }

  // Wound so the faces point up.
  //
  // Worth stating, because the failure is silent and total: with the other
  // winding `computeVertexNormals` hands back downward normals, the ground is
  // back-face culled from every angle anyone will ever see it from, and what
  // you get is a world where the sky reaches all the way to your feet.
  const indices: number[] = [];
  for (let ring = 0; ring < rings - 1; ring++) {
    for (let spoke = 0; spoke < spokes; spoke++) {
      const a = ring * columns + spoke;
      const b = a + 1;
      const c = a + columns;
      const d = c + 1;
      indices.push(a, b, c, b, d, c);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  return geometry;
}

/** Deterministic 0..1 from a vertex's grid address. */
function hash(ring: number, spoke: number, seed: number): number {
  let h = Math.imul(ring + 1, 0x27d4eb2d) ^ Math.imul(spoke + 1, 0x165667b1) ^ Math.imul(seed, 0x9e3779b9);
  h ^= h >>> 15;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  return (h >>> 8) / 16777216;
}
