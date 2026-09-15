/**
 * The ground of the mindscape.
 *
 * A small islet — a room, not a country. It is deliberately framed at the same
 * distance the memory orbs are, so switching floors changes *what* you are
 * looking at and not how far away you are standing: the orbs surround you at
 * arm's length, and so does this. An island the size of a map would be a
 * different kind of thing entirely — something to survey rather than somewhere
 * to be, and the app has no business asking anyone to survey their own week.
 *
 * So: a low mound of ground a dozen units across, with a well through the middle
 * that drops to the orbs below, standing in a sea that runs to a hazed-out
 * horizon (water.ts) with hills a long way behind it (horizon.ts). Its *shape*
 * comes from how the diary reads — how much feeling is in it — and its *colour*
 * from the same lifetime emotion weights that paint the lower sky, so the
 * ground, the far hills and the horizon behind them are three views of one
 * number.
 *
 * Height is a pure function, `elevation(x, z, shape)`, rather than a baked
 * heightmap. That is the load-bearing decision in this file: the terrain mesh,
 * the prop placement in mindscape.ts and the sea's baked depth in water.ts all
 * have to agree exactly on where the ground is, and a shared function cannot
 * drift out of sync the way a mesh and three samplers can. Props sit *on* the
 * terrain and surf breaks *at* the shore because all of them asked the same
 * question.
 *
 * Surface detail is per-pixel, not per-vertex — see `makeGroundMaterial`. A
 * hillside's identity is mostly in its texture, and at the resolution a mesh
 * this size can afford, vertex colours alone give you a faceted plastic shape.
 */

import * as THREE from 'three';
import { GRADIENT_ORDER, PALETTE, type EmotionVector, focus } from '../emotions';
import { BIOMES, type Biome } from './biomes';
import { NOISE_2D } from './glsl';
import { radialGrid } from './grid';
import { Water } from './water';

/**
 * Outer edge of the islet, where the land has fully sunk.
 *
 * Sized against the camera rig in scene.ts, not against any idea of how big a
 * landscape ought to be: the mind view sits roughly where the core view sits, so
 * the ground has to fit the same frame the orb galaxy fills.
 */
export const ISLAND_RADIUS = 13;
/** The hole in the middle. Everything inside this is the descent to the core. */
export const WELL_RADIUS = 2.8;

/** Where the land stops rolling and starts being rock. */
const ROCK_LINE = 1.8;

/**
 * Tessellation of the island.
 *
 * Spokes set the resolution outright; the rings follow from them, because
 * `radialGrid` spaces rings to keep every quad square rather than taking a
 * count. See ./grid.ts for why an evenly-spaced ring mesh cannot do that.
 */
const ISLAND_SPOKES = 224;

/**
 * The one colour not owned by the biome.
 *
 * Every other ground colour comes from ./biomes.ts, because "what the ground is
 * made of" is the biome's whole job. Deep water is shared: it is what you see
 * *through* the sea, not what the place is made of, and it should look the same
 * off a beach as it does off a tarn.
 */
const DEEP = new THREE.Color('#1d2e42');
const SNOW = new THREE.Color('#dde4f0');

/** Only reached when every band weight is negligible, which nothing produces. */
const FALLBACK = new THREE.Color('#5a6650');

export interface TerrainShape {
  /** Stable per-diary, so the island is the same island every time you open it. */
  seed: number;
  /** Hilliness in [0, 1]. Driven by how much feeling the diary actually carries. */
  relief: number;
  /**
   * Sea level, or null where the biome has no water.
   *
   * Copied off the biome rather than read through it because `elevation` and the
   * sea's depth attribute both want it on the hot path.
   */
  waterLevel: number | null;
  /** The kind of place. Decides the edge, the palette, and whether there is a sea. */
  biome: Biome;

  /**
   * The compass bearing the landform runs along, in radians.
   *
   * Stable per diary. Everything about the ground used to be a function of
   * radius alone, which is why every world came out as a dome on a disc — the
   * one shape that has no direction in it. A grain gives the place an uphill and
   * a downhill, and therefore a composition.
   */
  grain: number;
}

export const DEFAULT_SHAPE: TerrainShape = {
  seed: 1,
  relief: 0.42,
  waterLevel: null,
  biome: BIOMES.bare,
  grain: 0,
};

/**
 * How far along the grain a point lies, in island radii.
 *
 * Negative is uphill, positive is downhill — so for a coast, positive is out to
 * sea. Exported because horizon.ts has to agree about which way the land falls,
 * or the far field runs uphill into the water.
 */
export function alongGrain(x: number, z: number, shape: TerrainShape): number {
  return (x * Math.cos(shape.grain) + z * Math.sin(shape.grain)) / ISLAND_RADIUS;
}

// ---------------------------------------------------------------------------
// The landform
// ---------------------------------------------------------------------------

/**
 * Ground height at a point.
 *
 * Four terms, each doing one job:
 *
 *   `lip`   holds the land down to nothing at the mouth of the well, so the
 *           hole has an edge rather than a cliff of arbitrary height.
 *   `rim`   raises a ridge around the well. Without it the centre is the
 *           flattest part of the ground, which puts the one thing the eye needs
 *           to find — the way down — in the least legible place on screen.
 *   `hills` is the local landscape, and the term the diary's own feeling moves.
 *   `lean`  is the grain: which way this place runs downhill. It is what stops
 *           every world being radially symmetric, and for a coast it is the
 *           thing that decides where the sea is.
 *
 * Valid at any radius, not just inside the mesh — horizon.ts and water.ts both
 * evaluate it far outside the island to find where the ground and the sea floor
 * are, so nothing here may assume it is being asked about visible geometry.
 */
export function elevation(x: number, z: number, shape: TerrainShape): number {
  // A room is flat. Every term below describes a landform, and a floor is not
  // one — see ./room.ts, which draws it instead of this file.
  if (shape.biome.id === 'room') return 0;

  const r = Math.hypot(x, z);

  const lip = smoothstep(WELL_RADIUS, WELL_RADIUS + 2.4, r);

  const fromWell = (r - (WELL_RADIUS + 1.3)) / 1.3;
  const rim = Math.exp(-fromWell * fromWell) * 0.85;

  // Frequency is set against ISLAND_RADIUS, not chosen for its own sake: at
  // this size the ground has room for two or three undulations before they stop
  // reading as landform and start reading as texture.
  const hills = (fbm(x * 0.11, z * 0.11, shape.seed) * 2 - 1) * shape.relief * 2.6;

  // Inland, the ground simply carries on and horizon.ts takes over past the
  // mesh's edge. The lean is clamped so the far field can call this at any
  // radius without the slope running off to infinity.
  if (shape.biome.edge === 'continuous') {
    const lean = -shape.biome.tilt * clamp(alongGrain(x, z, shape), -2.6, 2.6);

    // Close the bowl on the downhill side.
    //
    // Inland water is a pond, and a pond has to be *enclosed* or its edge is
    // wherever the mesh happens to stop — which showed up as a straight blue cut
    // across the hillside where the water disc ended. Lifting the ground back up
    // only where the lean has pushed it down guarantees the low side rises above
    // the waterline again before the disc runs out, so the pond has a far bank.
    // Applied to the downhill side alone, so it does not also raise the uphill
    // side into the camera's line of sight.
    const basin = Math.max(0, -lean) * smoothstep(8, 20, r) * 1.12;

    return (1 + hills + lean + basin) * lip + rim;
  }

  // A coast is not an island.
  //
  // The radial falloff that used to live here sank the land evenly in every
  // direction, which can only ever produce one thing: a disc with water round
  // it. Every world came out as the same island seen from a different angle.
  //
  // Letting the *lean* decide where the water goes instead gives a coast:
  // whichever side is downhill floods, the waterline crosses the view rather
  // than enclosing it, and the land carries on over the other shoulder into the
  // far field. The two sides are deliberately asymmetric — seaward keeps falling
  // so the water gets properly deep, landward rises gently and levels off, since
  // a beach backed by a wall is not a beach.
  const along = alongGrain(x, z, shape);
  const tilt = shape.biome.tilt;
  const shoreLean =
    along > 0
      ? -tilt * Math.min(along, 4) * 1.4
      : -tilt * Math.max(along, -1.2) * 0.55;

  // A wobble on the waterline, or the shore is a ruled line across the view.
  const wobble = (fbm(x * 0.085 + 4.4, z * 0.085 - 2.1, shape.seed + 41) * 2 - 1) * 0.85;

  return (1 + hills + wobble + shoreLean) * lip + rim + seaIsland(x, z, shape);
}

/**
 * Land out in the water.
 *
 * A sea with nothing in it is a colour, not a place: there is no scale, nothing
 * for the eye to travel to, and no reason to look that way at all. One small
 * island a long way out fixes all three, and it is the cheapest possible piece
 * of Attention Restoration Theory's *extent* — the sense that the world carries
 * on past the frame.
 *
 * Exported and added in two places, which is the only fiddly thing about it.
 * The far field draws its own surface from horizon.ts's `backdrop` and never
 * consults this function past the blend band, while water.ts bakes its depth
 * from `elevation`. Put the island in only one of them and you get either an
 * island the sea does not know about — drowned in deep-water colour with no
 * beach — or a shoal of shallow water with no land in it. Both were tried.
 *
 * Placed along the grain, on the downhill side, because that is where the water
 * is. Everything else about it is derived from the seed, so it is the same
 * island every time the diary is opened.
 */
export function seaIsland(x: number, z: number, shape: TerrainShape): number {
  if (shape.biome.edge !== 'shore') return 0;

  // Two, at different sizes and bearings: one to look at and one to stop the
  // first from reading as a deliberately placed object.
  return (
    island(x, z, shape, 0.42, 58, 10.5, 13) +
    island(x, z, shape, -0.63, 46, 5.5, 8.5)
  );
}

/**
 * One landform in the water.
 *
 * @param swing   Bearing away from straight downhill, in radians.
 * @param out     How far from the middle of the world it sits.
 * @param width   Radius at which it has fallen to a third of its height.
 * @param rise    Peak height above the surrounding sea floor.
 */
function island(
  x: number,
  z: number,
  shape: TerrainShape,
  swing: number,
  out: number,
  width: number,
  rise: number
): number {
  const bearing = shape.grain + swing;
  const cx = Math.cos(bearing) * out;
  const cz = Math.sin(bearing) * out;

  const d = Math.hypot(x - cx, z - cz);
  // Cut off well before the falloff would matter, so the whole thing costs one
  // distance check almost everywhere in the world.
  if (d > width * 2.2) return 0;

  // A rough edge, or it is a cone. Sampled on position rather than on the
  // distance so the coastline wanders rather than pulsing in and out.
  const ragged = 1 + (fbm(x * 0.09 + 31.7, z * 0.09 - 12.3, shape.seed + 205) * 2 - 1) * 0.45;
  const t = d / (width * ragged);
  return Math.exp(-t * t) * rise;
}

/**
 * How much of the island's radius the shore takes, by bearing.
 *
 * A single number here gives a perfectly circular coast — every direction the
 * same, which is the one thing no coastline is. Varying it around the compass
 * with a smooth angular noise buys the whole silhouette: where it is wide the
 * land shelves gently and the sea leaves a broad beach, where it is narrow the
 * ground drops away and you get a headland. Sampled on (cos, sin) rather than on
 * the angle itself so it wraps seamlessly at north instead of showing a seam.
 */
function shoreWidth(x: number, z: number, seed: number): number {
  const r = Math.max(1e-4, Math.hypot(x, z));
  const bearing = fbm((x / r) * 1.6 + 21.7, (z / r) * 1.6 - 9.4, seed + 41);
  return 0.12 + 0.34 * clamp01((bearing - 0.28) / 0.44);
}

/** Surface steepness in [0, 1], sampled by finite difference. */
function slopeAt(x: number, z: number, shape: TerrainShape): number {
  const d = 0.5;
  const dx = elevation(x + d, z, shape) - elevation(x - d, z, shape);
  const dz = elevation(x, z + d, shape) - elevation(x, z - d, shape);
  return Math.min(1, Math.hypot(dx, dz) / (2 * d) / 1.4);
}

// ---------------------------------------------------------------------------
// The mesh
// ---------------------------------------------------------------------------

export class Terrain {
  readonly group = new THREE.Group();

  private ground: THREE.Mesh | null = null;
  private material = makeGroundMaterial();
  private water = new Water();
  private shape: TerrainShape = { ...DEFAULT_SHAPE };

  constructor() {
    this.group.add(this.water.mesh);
  }

  getShape(): TerrainShape {
    return this.shape;
  }

  heightAt(x: number, z: number): number {
    return elevation(x, z, this.shape);
  }

  slopeAt(x: number, z: number): number {
    return slopeAt(x, z, this.shape);
  }

  /**
   * Rebuild the ground.
   *
   * Called when the diary changes, which at diary scale is a handful of times
   * per session — so this allocates a fresh geometry rather than trying to
   * update one in place. The grid is a couple of dozen thousand triangles,
   * which is nothing next to the bloom pass that is already running, and fine
   * enough that the fourth noise octave in `elevation` actually survives into
   * the mesh instead of being averaged away between vertices.
   */
  rebuild(shape: TerrainShape, lifetimeTotals: EmotionVector): void {
    this.shape = shape;
    this.disposeGround();

    // Indoors there is no landform to build. The shape is still recorded above,
    // because prop placement asks this class how high the ground is and needs a
    // straight answer (zero) rather than a special case at every call site.
    if (shape.biome.id === 'room') {
      this.water.mesh.visible = false;
      return;
    }

    const geometry = radialGrid({
      inner: WELL_RADIUS,
      outer: ISLAND_RADIUS * 1.08,
      spokes: ISLAND_SPOKES,
      aspect: 1,
      jitter: 0.34,
      seed: shape.seed + 3,
    });

    const position = geometry.attributes.position as THREE.BufferAttribute;
    const colors = new Float32Array(position.count * 3);
    const bands = bandWeights(lifetimeTotals);
    const palette = gradientPalette();
    const scratch = new THREE.Color();

    for (let i = 0; i < position.count; i++) {
      const x = position.getX(i);
      const z = position.getZ(i);
      const y = elevation(x, z, shape);
      position.setY(i, y);

      groundColorAt(x, z, y, shape, bands, palette, scratch);
      colors[i * 3] = scratch.r;
      colors[i * 3 + 1] = scratch.g;
      colors[i * 3 + 2] = scratch.b;
    }

    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    // Smooth, not flat. The surface texture now comes from the material, and
    // faceting on top of it reads as two conflicting scales of detail — the
    // polygons win, and the ground goes back to looking like folded paper.
    geometry.computeVertexNormals();

    this.ground = new THREE.Mesh(geometry, this.material);
    this.group.add(this.ground);

    // A dry biome has no sea mesh at all, rather than one pushed out of sight.
    // An invisible two-hundred-unit disc is still a disc somebody will
    // eventually see the edge of.
    if (shape.waterLevel === null) {
      this.water.mesh.visible = false;
    } else {
      this.water.mesh.visible = true;
      this.water.rebuild(shape, shape.biome.waterRadius);
    }
  }

  /**
   * Per-frame drift.
   *
   * Only the sea moves. The land is the slow layer — it changes when the diary
   * changes and not otherwise, the same contract the memory orbs keep.
   */
  update(
    delta: number,
    elapsed: number,
    mood: { color: string; arousal: number },
    sky: THREE.Color
  ): void {
    this.water.update(delta, elapsed, mood, sky);
  }

  private disposeGround(): void {
    if (!this.ground) return;
    this.group.remove(this.ground);
    this.ground.geometry.dispose();
    this.ground = null;
  }

  dispose(): void {
    this.disposeGround();
    this.material.dispose();
    this.water.dispose();
  }
}

// ---------------------------------------------------------------------------
// Surface
// ---------------------------------------------------------------------------

export interface GroundMaterialOptions {
  /**
   * Where the surface detail starts fading out and where it is gone.
   *
   * The detail below is authored at a fixed world-space frequency, which is
   * right for ground you are standing near and wrong for a mountain range four
   * screen-pixels tall: past a certain distance the noise is finer than a pixel
   * and all it can do is shimmer as the camera turns. Fading it out leaves the
   * distance carrying its silhouette and its colour, which is all that survives
   * out there anyway.
   */
  detailNear?: number;
  detailFar?: number;
  /** Pushes this surface back in the depth buffer. See the far field's overlap. */
  depthBias?: number;
}

/**
 * The ground's material: MeshStandardMaterial with procedural surface detail.
 *
 * Patched rather than written from scratch so the island keeps real lighting and
 * stays compatible with the bloom pass, exactly as the orb material does.
 *
 * Two additions, and the second matters more than the first:
 *
 *   colour — two octaves of noise breaking up the vertex colour. Stops the big
 *            smooth regions the per-vertex palette produces from reading as
 *            painted plastic.
 *   normal — the same noise, differenced into a gradient and used to tilt the
 *            shading normal. This is what actually makes it look like ground:
 *            a surface whose *lighting* varies at a scale finer than its
 *            geometry is the entire difference between terrain and a tinted
 *            shape, and it costs four noise samples.
 *
 * The perturbation is built in world space and then rotated into view space,
 * because `normal` at this point in three's shader is a view-space vector and
 * adding a world-space offset to it would make the lighting swing as the camera
 * orbits.
 *
 * Exported because the far field uses it too, and that is not an optimisation —
 * it is the difference between one landscape and two. The distance used to be a
 * plain vertex-coloured material, so the moment the ground crossed from the
 * island onto the backdrop it stopped being made of anything and became painted
 * polygons. Sharing the recipe means the near ground and the far ground are the
 * same substance seen at different distances, which is what they are.
 */
export function makeGroundMaterial(
  options: GroundMaterialOptions = {}
): THREE.MeshStandardMaterial {
  const detailNear = options.detailNear ?? 55;
  const detailFar = options.detailFar ?? 150;

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.95,
    metalness: 0.0,
    fog: true,
  });

  if (options.depthBias) {
    material.polygonOffset = true;
    material.polygonOffsetFactor = options.depthBias;
    material.polygonOffsetUnits = options.depthBias;
  }

  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
         varying vec3 vGroundPos;
         varying vec3 vGroundNormal;`
      )
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
         vGroundPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
         vGroundNormal = normalize(mat3(modelMatrix) * objectNormal);`
      );

    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
         varying vec3 vGroundPos;
         varying vec3 vGroundNormal;
         ${NOISE_2D}

         // How much fine detail this fragment still deserves.
         float groundDetail(vec3 worldPos) {
           return 1.0 - smoothstep(
             ${detailNear.toFixed(1)}, ${detailFar.toFixed(1)},
             distance(cameraPosition, worldPos)
           );
         }`
      )
      .replace(
        '#include <color_fragment>',
        `#include <color_fragment>
         {
           float detail = groundDetail(vGroundPos);
           float grain = fbm2(vGroundPos.xz * 2.7 + 4.3);
           // Not "patch" — that is a reserved word in GLSL ES 3.0 (tessellation)
           // and the shader will not compile with it, with an error that points
           // at the line after the one you wrote.
           float blotch = fbm2(vGroundPos.xz * 0.6 - 11.1);
           // Multiplicative, so the variation rides on whatever colour this
           // piece of ground already is rather than washing every region toward
           // the same grey.
           diffuseColor.rgb *= mix(1.0, 0.80 + 0.40 * grain, detail);
           // The blotching is broad enough to survive at range, so it keeps
           // most of its strength where the grain has gone.
           diffuseColor.rgb *= mix(
             vec3(0.94, 0.97, 1.02), vec3(1.10, 1.03, 0.90),
             mix(0.5, blotch, 0.35 + 0.65 * detail)
           );
         }`
      )
      .replace(
        '#include <normal_fragment_begin>',
        `#include <normal_fragment_begin>
         {
           float detail = groundDetail(vGroundPos);
           if (detail > 0.01) {
             vec2 gp = vGroundPos.xz * 1.9;
             float e = 0.22;
             float n0 = fbm2(gp);
             float nx = fbm2(gp + vec2(e, 0.0));
             float nz = fbm2(gp + vec2(0.0, e));
             vec3 bumped = normalize(
               vGroundNormal
                 + vec3(-(nx - n0), 0.0, -(nz - n0)) * (0.55 / e) * detail
             );
             normal = normalize((viewMatrix * vec4(bumped, 0.0)).xyz);
           }
         }`
      )
      .replace(
        '#include <roughnessmap_fragment>',
        `#include <roughnessmap_fragment>
         // Wet sand and bare rock are not as matte as turf. Cheap, and it keeps
         // the shoreline from looking like the same material as the hilltop.
         roughnessFactor *= 0.78 + 0.22 * fbm2(vGroundPos.xz * 0.9 + 21.0);`
      );
  };

  return material;
}

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

/**
 * The colour of one patch of ground.
 *
 * Exported, and used by horizon.ts as well as by the island, because the far
 * field's inner edge overlaps this mesh. Two independent recipes for "what
 * colour is the ground here" produced a visible ring where one ended and the
 * other began — the single most obvious way to tell a player they are looking
 * at two objects rather than one place.
 *
 * The emotional layer comes first and everything else is painted over it, in
 * the order a landscape would: the diary decides the hue of the earth, then
 * water decides where there is sand, then slope and height decide where there
 * is rock, then — only where the biome says so — the tops get snow.
 */
export function groundColorAt(
  x: number,
  z: number,
  y: number,
  shape: TerrainShape,
  bands: Float32Array,
  palette: THREE.Color[],
  out: THREE.Color
): THREE.Color {
  const biome = shape.biome;

  // Two earths rather than one, mixed by a second field. A single base colour
  // makes every part of the ground the same kind of ground, which is what a
  // model looks like and not what a place looks like.
  const dryness = clamp01(fbm(x * 0.14 - 6.2, z * 0.14 + 3.3, shape.seed + 13) * 1.6 - 0.35);
  const earth = SCRATCH_EARTH.set(biome.ground.base).lerp(
    SCRATCH_DRY.set(biome.ground.dry),
    dryness
  );

  // A broad, slow field so the emotional colouring arrives as regions of ground
  // rather than as per-vertex speckle.
  const t = clamp01(
    smoothstep(0.3, 0.7, fbm(x * 0.085 + 11.3, z * 0.085 - 4.1, shape.seed + 97))
  );
  gradientAt(t, bands, palette, out);
  earthen(out, earth);
  out.lerp(earth, 1 - biome.tint);

  const sea = shape.waterLevel;
  if (sea !== null) {
    const above = y - sea;
    // Sand reaches further up where the shore shelves gently, which is the point
    // of varying the coast width: a wide shore is a beach and should look like
    // one, a narrow one is a headland and should stay green to the edge. Inland
    // water gets a thin margin instead — a tarn does not have a beach.
    const margin =
      biome.edge === 'shore' ? 0.2 + shoreWidth(x, z, shape.seed) * 2.6 : 0.35;
    if (above < margin) {
      out.lerp(SCRATCH_SAND.set(biome.ground.sand), clamp01((margin - above) / (margin + 0.4)) * 0.82);
    }
    // Underwater goes dark fast. The water is clear in the shallows, so without
    // this the submerged ground shows through as bright land.
    if (above < 0) out.lerp(DEEP, clamp01(-above / 1.1) * 0.92);
  }

  const steep = slopeAt(x, z, shape);
  if (y > ROCK_LINE || steep > 0.5) {
    const rockiness = Math.max(
      clamp01((y - ROCK_LINE) / 1.4),
      clamp01((steep - 0.5) / 0.4) * 0.8
    );
    out.lerp(SCRATCH_ROCK.set(biome.ground.rock), rockiness * 0.8);
  }
  if (y > biome.snowAbove) {
    out.lerp(SNOW, clamp01((y - biome.snowAbove) / 1.4) * 0.88);
  }

  // Hollows darken. Cheap ambient occlusion, and most of what gives a smooth-
  // shaded ground any sense of its own shape under a single directional light.
  return out.multiplyScalar(0.76 + 0.24 * clamp01((y - (sea ?? -1.5)) / 3));
}

/**
 * Pull an emotion colour down to something ground could plausibly be.
 *
 * The palette in emotions.ts is a *UI* palette — #ffd23f for joy, #9b5de5 for
 * fear — chosen so a dot on a chart is unmistakable at eight pixels across.
 * Painting a hillside with it at even a third strength gives exactly what it
 * sounds like: slicks of purple and orange lying on the grass, which read as a
 * texturing bug rather than as a mood. The land looked like something had been
 * spilled on it.
 *
 * The hue is the part that carries the meaning, so the hue is what survives
 * untouched. Saturation is capped near the earth's own, and the lightness is
 * replaced by the earth's, nudged by whether this emotion is a light or a dark
 * one. What comes out is the same ground in a different cast — which is what
 * "your week has a colour" should look like on a landscape.
 *
 * Exported because the far field tints itself from the same palette and would
 * otherwise disagree with the island about how loud that palette is allowed to
 * be.
 */
export function earthen(color: THREE.Color, earth: THREE.Color): THREE.Color {
  color.getHSL(HSL_MOOD);
  earth.getHSL(HSL_EARTH);
  return color.setHSL(
    HSL_MOOD.h,
    Math.min(HSL_MOOD.s, 0.26 + HSL_EARTH.s * 0.55),
    HSL_EARTH.l * (0.8 + 0.4 * HSL_MOOD.l)
  );
}

/** Reused per call. This runs once per vertex over tens of thousands of them. */
const SCRATCH_EARTH = new THREE.Color();
const SCRATCH_DRY = new THREE.Color();
const SCRATCH_SAND = new THREE.Color();
const SCRATCH_ROCK = new THREE.Color();
const HSL_MOOD = { h: 0, s: 0, l: 0 };
const HSL_EARTH = { h: 0, s: 0, l: 0 };

/** The palette in GRADIENT_ORDER. Allocated per rebuild, never per frame. */
export function gradientPalette(): THREE.Color[] {
  return GRADIENT_ORDER.map((e) => new THREE.Color(PALETTE[e].base));
}

/**
 * Lifetime emotion weights in GRADIENT_ORDER.
 *
 * Focused with the same constants atmosphere.ts uses for its lifetime layer, so
 * the ground and the lower sky are literally the same distribution — the island
 * cannot disagree with the horizon behind it about what the diary has been.
 */
export function bandWeights(totals: EmotionVector): Float32Array {
  const focused = focus(totals, 1.6, 0.9);
  const weights = new Float32Array(GRADIENT_ORDER.length);
  for (let i = 0; i < GRADIENT_ORDER.length; i++) weights[i] = focused[GRADIENT_ORDER[i]];
  return weights;
}

/**
 * JS mirror of the sky shader's `gradientAt`.
 *
 * Each emotion claims a slice of 0..1 proportional to its weight and every
 * sample is a soft weighted average of the slices near it — so two emotions held
 * equally give two regions of ground rather than one averaged mud colour, for
 * exactly the reason the sky does it this way.
 */
export function gradientAt(
  t: number,
  weights: Float32Array,
  colors: THREE.Color[],
  out: THREE.Color
): THREE.Color {
  let r = 0;
  let g = 0;
  let b = 0;
  let total = 0;
  let cursor = 0;

  for (let i = 0; i < weights.length; i++) {
    const w = weights[i];
    if (w < 0.012) {
      cursor += w;
      continue;
    }
    const center = cursor + w * 0.5;
    const halfWidth = Math.max(w * 0.5, 0.055);
    const d = (t - center) / halfWidth;
    const influence = Math.exp(-d * d) * w;

    r += colors[i].r * influence;
    g += colors[i].g * influence;
    b += colors[i].b * influence;
    total += influence;
    cursor += w;
  }

  if (total < 1e-4) return out.copy(FALLBACK);
  return out.setRGB(r / total, g / total, b / total);
}

// ---------------------------------------------------------------------------
// Noise
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

/** Four octaves. The fourth is what the denser mesh exists to carry. */
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
