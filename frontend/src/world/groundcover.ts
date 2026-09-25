/**
 * What grows on the ground, as opposed to what stands on it.
 *
 * The props in ./props.ts are *statements*: there are four of them, they are
 * placed one at a time in a named sector, and each one is there because the
 * diary said something. This is the opposite — thousands of blades, petals,
 * fronds and pebbles, none of which means anything on its own, all of which
 * together are the difference between a landscape and a model of one. Ground
 * with nothing growing on it reads as a surface no matter how well it is shaded,
 * because the thing the eye uses to judge that ground is ground is the litter on
 * top of it.
 *
 * It still answers to the person, and that is the point of putting it here
 * rather than scattering a fixed lawn:
 *
 *   grass    a base rate from the biome — a meadow is thick with it, high
 *            ground is nearly bare — lifted by how much the diary is about
 *            things that grow at all.
 *   flowers  from `garden`, and coloured from the *feeling* in the entries that
 *            mentioned it, exactly as that motif's props are.
 *   ferns    from `forest`, with saplings scattered through them, so woodland
 *            has an understorey rather than trunks standing on a lawn.
 *   reeds    from `sea`, and only within a stride of the waterline.
 *   stones   from `mountains`, and on anything steep or high enough to be scree.
 *
 * Everything is instanced: one draw call per species however many there are, so
 * the whole layer is five or six draws and a few tens of thousands of triangles.
 * That budget is what buys the density, and density is the entire effect — a
 * hundred tufts of grass look like a hundred tufts of grass, and four thousand
 * look like a field.
 *
 * Geometry is built by hand into flat arrays rather than merged from primitives.
 * A blade of grass is one triangle; going through BoxGeometry to get one would
 * cost twelve, and at these counts that is the difference between affordable and
 * not.
 */

import * as THREE from 'three';
import { PALETTE, type Emotion } from '../emotions';
import type { MotifPresence } from './motifs';
import { makeRng, type Rng } from './props';
import { ISLAND_RADIUS, WELL_RADIUS, type TerrainShape } from './terrain';

/** How far out cover is scattered. Past this the fog has it anyway. */
const COVER_RADIUS = 32;
/** Nothing grows on the lip of the well. */
const COVER_INNER = WELL_RADIUS + 0.5;

/** Stand-in sea level where the biome has none, matching mindscape.ts. */
const DRY_DATUM = -0.6;

/** A hard ceiling per species, so no diary can bring the frame rate down. */
const MAX_INSTANCES = 9000;

/** Where the ground is when nothing has told us otherwise. */
export type HeightSampler = (x: number, z: number) => number;

interface Layer {
  mesh: THREE.InstancedMesh;
  material: THREE.MeshStandardMaterial;
}

export class GroundCover {
  readonly group = new THREE.Group();

  private layers: Layer[] = [];
  /** Shared across every species, so one write per frame drives all the wind. */
  private wind = { time: { value: 0 }, strength: { value: 1 } };

  /**
   * Scatter cover for one landform.
   *
   * @param heightAt Where the ground is. Handed in rather than read off the
   *                 terrain because cover runs past the island's own mesh and
   *                 out onto the far field, and only the caller knows which of
   *                 the two owns a given point.
   */
  rebuild(shape: TerrainShape, motifs: MotifPresence[], heightAt: HeightSampler): void {
    this.clear();
    if (shape.biome.id === 'room') return;

    const share = (id: string): number =>
      motifs.find((m) => m.motif.id === id)?.share ?? 0;
    const feeling = (id: string): Emotion =>
      motifs.find((m) => m.motif.id === id)?.emotion ?? 'neutral';

    const biome = shape.biome;
    const sea = shape.waterLevel ?? DRY_DATUM;
    const rng = makeRng((shape.seed ^ 0x7f4a7c15) >>> 0);

    /** Ground colour, so what grows agrees with what it grows out of. */
    const earth = new THREE.Color(biome.ground.base);

    // How much of this diary is about living, growing things. Lifts the turf
    // everywhere — someone whose weeks are gardens and woods gets a thicker
    // world than someone whose weeks are deadlines, which is true.
    const green = clamp01(share('garden') * 1.6 + share('forest') * 1.3 + share('pets') * 0.5);

    const turf = TURF_BASE[biome.id] ?? 0.5;

    // -- turf ------------------------------------------------------------
    //
    // The one species that is allowed to be everywhere, and the only one whose
    // job is coverage rather than incident. Loosely clumped: grass thins out
    // rather than stopping, so its floor is high.
    this.scatter({
      geometry: grassTuft(earth, rng),
      count: 9000 * turf * (0.55 + 0.45 * green),
      scale: [0.65, 1.3],
      sway: 1,
      clump: { floor: 0.55, scale: 0.16 },
      rng,
      shape,
      heightAt,
      accept: (y, slope) => y - sea > 0.28 && slope < 0.62 && y < biome.snowAbove - 0.3,
    });

    // -- flowers ---------------------------------------------------------
    //
    // The one species whose colour is not a colour of the ground. Taken from
    // how the person felt in the entries that mentioned growing things, the
    // same source that tints that motif's props — so a bed of flowers is a
    // legible piece of the same sentence the props are making, not decoration
    // that happens to be pink.
    const growing = share('garden');
    if (growing > 0.015) {
      this.scatter({
        geometry: flowers(new THREE.Color(PALETTE[feeling('garden')].base), earth, rng),
        count: 1100 * clamp01(growing * 2.6),
        // Tightly clumped, on a finer field than the grass: flowers come in
        // beds. Spread evenly they are speckle, and a hillside evenly speckled
        // with colour looks like a rash rather than like flowers.
        clump: { floor: 0.03, scale: 0.34, power: 2 },
        scale: [0.7, 1.25],
        sway: 0.8,
        rng,
        shape,
        heightAt,
        accept: (y, slope) => y - sea > 0.4 && slope < 0.5 && y < biome.snowAbove - 0.5,
      });
    }

    // -- understorey -----------------------------------------------------
    const woods = share('forest');
    if (woods > 0.015 || biome.id === 'forest') {
      const strength = clamp01(Math.max(woods * 2.2, biome.id === 'forest' ? 0.55 : 0));
      this.scatter({
        geometry: fern(earth, rng),
        count: 1400 * strength,
        clump: { floor: 0.08, scale: 0.26, power: 1.6 },
        scale: [0.8, 1.5],
        sway: 0.55,
        rng,
        shape,
        heightAt,
        accept: (y, slope) => y - sea > 0.5 && slope < 0.66 && y < biome.snowAbove - 0.5,
      });
      // Saplings: the same silhouette as the far woodland, close enough to
      // resolve. They are what connects the treeline to the ground you are
      // standing on instead of leaving it as a wall at the back.
      //
      // Kept firmly smaller than the motif's own trees. At anything near their
      // size they stop being undergrowth and become a second, competing forest
      // — and since there are ten times as many of them, it is the props that
      // lose, which inverts what the two layers are for.
      this.scatter({
        geometry: sapling(earth, rng),
        count: 260 * strength,
        clump: { floor: 0.1, scale: 0.09, power: 1.5 },
        scale: [0.6, 1.15],
        sway: 0.3,
        rng,
        shape,
        heightAt,
        // Kept off the middle of the island: the props stand there, and a
        // sapling in front of a house is just something in the way.
        accept: (y, slope, r) => r > 9 && y - sea > 0.6 && slope < 0.6 && y < biome.snowAbove - 0.8,
      });
    }

    // -- waterside -------------------------------------------------------
    if (shape.waterLevel !== null) {
      this.scatter({
        geometry: reeds(earth, rng),
        count: 1100 * clamp01(0.3 + share('sea') * 2.2),
        clump: { floor: 0.06, scale: 0.4, power: 1.8 },
        scale: [0.75, 1.5],
        sway: 1.5,
        rng,
        shape,
        heightAt,
        // A narrow band straddling the waterline, which is where reeds are and
        // is also the one place on a coast that otherwise has nothing on it.
        accept: (y, slope) => y - sea > -0.25 && y - sea < 0.75 && slope < 0.55,
      });
    }

    // -- scree -----------------------------------------------------------
    const stony = Math.max(share('mountains') * 2, biome.id === 'alpine' ? 0.7 : 0.06);
    this.scatter({
      geometry: pebbles(new THREE.Color(biome.ground.rock), rng),
      count: 1600 * clamp01(stony),
      clump: { floor: 0.2, scale: 0.22 },
      scale: [0.6, 1.5],
      sway: 0,
      rng,
      shape,
      heightAt,
      // Stone lies where turf will not: steep ground, and high ground.
      accept: (y, slope) => y - sea > 0.2 && (slope > 0.34 || y > 1.5) && y < biome.snowAbove,
    });
  }

  /** One species: build the instances and add the draw call. */
  private scatter(options: {
    geometry: THREE.BufferGeometry;
    count: number;
    scale: [number, number];
    /** How hard the wind moves this. Zero for anything made of rock. */
    sway: number;
    /**
     * How this species drifts.
     *
     * `floor` is the chance of appearing on ground the noise field says nothing
     * grows on — high for turf, which thins rather than stops, and near zero for
     * flowers, which come in beds or not at all. `scale` is the noise frequency,
     * so a bed of flowers is a different size from a stand of ferns. `power`
     * sharpens the edges of a drift.
     */
    clump?: { floor: number; scale: number; power?: number };
    rng: Rng;
    shape: TerrainShape;
    heightAt: HeightSampler;
    /** @param y is absolute height; the caller compares it against sea level. */
    accept: (y: number, slope: number, radius: number) => boolean;
  }): void {
    const wanted = Math.min(MAX_INSTANCES, Math.round(options.count));
    if (wanted < 8) {
      options.geometry.dispose();
      return;
    }

    const { rng, shape, heightAt } = options;
    const clump = options.clump ?? { floor: 0.25, scale: 0.19 };
    const power = clump.power ?? 1;
    const material = coverMaterial(options.sway, this.wind);
    const mesh = new THREE.InstancedMesh(options.geometry, material, wanted);
    const dummy = new THREE.Object3D();

    let placed = 0;
    // Six attempts a head. The filters reject most of the island on a steep or
    // half-drowned landform, and giving up early is better than spinning: a
    // thinner field on difficult ground is the honest result.
    for (let attempt = 0; attempt < wanted * 6 && placed < wanted; attempt++) {
      const angle = rng() * Math.PI * 2;
      // sqrt for even areal density, then a radial falloff on top: cover this
      // fine is invisible past the middle distance, and spending half the
      // instances out there would cost the near ground its density.
      const r = Math.sqrt(
        COVER_INNER * COVER_INNER +
          rng() * (COVER_RADIUS * COVER_RADIUS - COVER_INNER * COVER_INNER)
      );
      if (rng() > falloff(r)) continue;

      const x = Math.cos(angle) * r;
      const z = Math.sin(angle) * r;
      const y = heightAt(x, z);
      if (!options.accept(y, slopeOf(heightAt, x, z), r)) continue;
      // Patchiness. Real ground cover comes in drifts with bare earth between,
      // and an evenly random scatter is as obviously artificial as a grid.
      const drift = Math.pow(patch(x, z, clump.scale, shape.seed), power);
      if (rng() > clump.floor + (1 - clump.floor) * drift) continue;

      const scale = options.scale[0] + rng() * (options.scale[1] - options.scale[0]);
      dummy.position.set(x, y - 0.04, z);
      dummy.scale.set(scale, scale * (0.8 + rng() * 0.45), scale);
      dummy.rotation.set((rng() - 0.5) * 0.18, rng() * Math.PI * 2, (rng() - 0.5) * 0.18);
      dummy.updateMatrix();
      mesh.setMatrixAt(placed, dummy.matrix);
      placed++;
    }

    if (placed === 0) {
      options.geometry.dispose();
      material.dispose();
      mesh.dispose();
      return;
    }

    mesh.count = placed;
    // The wind displaces vertices in the shader, so the bounds three computes
    // from the instance matrices are a little tight and tufts at the edge of
    // frame pop out. Cheaper to skip the test than to widen it every frame.
    mesh.frustumCulled = false;
    this.layers.push({ mesh, material });
    this.group.add(mesh);
  }

  /**
   * @param arousal How loudly the person is talking. Puts a gust through the
   *                grass, the same way it puts a chop on the water — the only
   *                two things in this world that answer to the microphone in
   *                real time, and both of them read as weather rather than as a
   *                meter.
   */
  update(elapsed: number, arousal: number): void {
    this.wind.time.value = elapsed;
    this.wind.strength.value = 0.75 + arousal * 0.55;
  }

  clear(): void {
    for (const layer of this.layers) {
      this.group.remove(layer.mesh);
      layer.mesh.geometry.dispose();
      layer.material.dispose();
      layer.mesh.dispose();
    }
    this.layers = [];
  }

  dispose(): void {
    this.clear();
  }
}

/** How thickly each kind of place is covered before the diary has its say. */
const TURF_BASE: Record<string, number> = {
  meadow: 1,
  forest: 0.9,
  coast: 0.62,
  bare: 0.42,
  alpine: 0.26,
  room: 0,
};

/** Density by radius: full near the camera, thinning into the haze. */
function falloff(r: number): number {
  return 1 - 0.8 * smoothstep(ISLAND_RADIUS * 0.85, COVER_RADIUS, r);
}

/** Steepness at a point, from the same sampler everything else uses. */
function slopeOf(heightAt: HeightSampler, x: number, z: number): number {
  const d = 0.45;
  const dx = heightAt(x + d, z) - heightAt(x - d, z);
  const dz = heightAt(x, z + d) - heightAt(x, z - d);
  return Math.min(1, Math.hypot(dx, dz) / (2 * d) / 1.4);
}

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

/**
 * Cover material: vertex-coloured, double-sided, and bending in the wind.
 *
 * Double-sided because every blade and petal here is a single triangle with no
 * back to it — face culling would make half the field vanish depending on which
 * way the camera happened to be standing.
 *
 * The wind is a vertex-shader bend rather than an animated transform, which is
 * the only way it is affordable: one uniform moves every instance of every
 * species, so four thousand tufts of grass sway for the cost of one number per
 * frame. The bend is scaled by the square of the vertex's own height, so the
 * base of a blade stays planted and the tip travels — a linear falloff makes the
 * whole tuft slide sideways, which reads as an object sliding, not as wind.
 */
function coverMaterial(
  sway: number,
  wind: { time: { value: number }; strength: { value: number } }
): THREE.MeshStandardMaterial {
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.92,
    metalness: 0,
    side: THREE.DoubleSide,
    fog: true,
  });

  if (sway <= 0) return material;

  material.onBeforeCompile = (shader) => {
    shader.uniforms.uWindTime = wind.time;
    shader.uniforms.uWindStrength = wind.strength;

    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
         uniform float uWindTime;
         uniform float uWindStrength;`
      )
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
         #ifdef USE_INSTANCING
         {
           // The instance's own position, so neighbouring tufts are out of
           // phase with each other and the field ripples instead of pulsing.
           vec3 root = instanceMatrix[3].xyz;
           float phase = root.x * 0.42 + root.z * 0.31 + uWindTime * 1.25;
           float gust = sin(phase) + 0.34 * sin(phase * 2.7 + 1.3);
           float lever = transformed.y * transformed.y;
           float bend = gust * lever * ${(sway * 0.16).toFixed(4)} * uWindStrength;
           transformed.x += bend;
           transformed.z += bend * 0.55;
         }
         #endif`
      );
  };

  return material;
}

// ---------------------------------------------------------------------------
// Species
//
// Every builder returns a geometry with its base at y = 0, roughly a unit tall,
// carrying its own vertex colours. Triangle counts are in the comments because
// they are the budget: these are multiplied by thousands.
// ---------------------------------------------------------------------------

/** Accumulates loose triangles into the flat arrays a BufferGeometry wants. */
class Mesher {
  private positions: number[] = [];
  private normals: number[] = [];
  private colors: number[] = [];

  /**
   * One triangle.
   *
   * The normal is bent three-quarters of the way toward straight up rather than
   * being the true face normal. Foliage lit by its own facets goes black
   * wherever a blade happens to face away from the key light, which at this size
   * reads as dirt; lighting it roughly as if it were the ground it grows out of
   * keeps a field of grass looking like a field.
   */
  triangle(
    a: [number, number, number],
    b: [number, number, number],
    c: [number, number, number],
    color: THREE.Color
  ): void {
    const ux = b[0] - a[0];
    const uy = b[1] - a[1];
    const uz = b[2] - a[2];
    const vx = c[0] - a[0];
    const vy = c[1] - a[1];
    const vz = c[2] - a[2];
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    const length = Math.hypot(nx, ny, nz) || 1;
    nx = (nx / length) * 0.25;
    ny = (ny / length) * 0.25 + 0.75;
    nz = (nz / length) * 0.25;
    const n = Math.hypot(nx, ny, nz) || 1;

    for (const vertex of [a, b, c]) {
      this.positions.push(vertex[0], vertex[1], vertex[2]);
      this.normals.push(nx / n, ny / n, nz / n);
      this.colors.push(color.r, color.g, color.b);
    }
  }

  /**
   * A blade: a tapered triangle leaning away from the root.
   *
   * One triangle, because that is what a blade of grass is worth. The taper is
   * free — it is just where the apex sits relative to the base.
   */
  blade(
    x: number,
    z: number,
    yaw: number,
    height: number,
    width: number,
    lean: number,
    color: THREE.Color
  ): void {
    const cos = Math.cos(yaw);
    const sin = Math.sin(yaw);
    // Across the blade, so it has a width at the bottom and a point at the top.
    const ax = -sin * width * 0.5;
    const az = cos * width * 0.5;
    this.triangle(
      [x + ax, 0, z + az],
      [x - ax, 0, z - az],
      [x + cos * lean, height, z + sin * lean],
      color
    );
  }

  build(): THREE.BufferGeometry {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(this.positions, 3));
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(this.normals, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(this.colors, 3));
    return geometry;
  }
}

/** Greens mixed off the biome's own earth, so turf belongs to its ground. */
function foliage(earth: THREE.Color, rng: Rng, shift: number): THREE.Color {
  const base = earth.clone().lerp(new THREE.Color('#5f8f4a'), 0.55 + shift);
  const value = 0.78 + rng() * 0.38;
  return base.multiplyScalar(value);
}

/**
 * A tuft: five blades from one root. Five triangles.
 *
 * Sized against the props rather than against reality: a human figure in this
 * world stands about one unit tall (see PROP_SCALE in mindscape.ts), so turf has
 * to come in at a quarter of that to read as turf. Grass authored at a plausible
 * real-world height turned the island into a wheat field with the props wading
 * through it, which is the failure mode this whole layer is one step away from.
 */
function grassTuft(earth: THREE.Color, rng: Rng): THREE.BufferGeometry {
  const mesher = new Mesher();
  for (let i = 0; i < 5; i++) {
    const yaw = rng() * Math.PI * 2;
    const height = 0.15 + rng() * 0.16;
    mesher.blade(
      (rng() - 0.5) * 0.12,
      (rng() - 0.5) * 0.12,
      yaw,
      height,
      0.028 + rng() * 0.022,
      height * (0.25 + rng() * 0.4),
      // A spread of greens inside a single tuft, or a field of them is one flat
      // colour however many blades it has.
      foliage(earth, rng, (rng() - 0.5) * 0.16)
    );
  }
  return mesher.build();
}

/** Three stems, each with a four-petal head. Fifteen triangles. */
function flowers(petal: THREE.Color, earth: THREE.Color, rng: Rng): THREE.BufferGeometry {
  const mesher = new Mesher();
  const heart = petal.clone().lerp(new THREE.Color('#fff2c2'), 0.55);

  for (let i = 0; i < 3; i++) {
    const x = (rng() - 0.5) * 0.22;
    const z = (rng() - 0.5) * 0.22;
    const height = 0.2 + rng() * 0.16;
    const stem = foliage(earth, rng, -0.06);
    mesher.blade(x, z, rng() * Math.PI * 2, height, 0.022, 0.035, stem);

    // The head, as petals radiating from the top of the stem. Splayed slightly
    // upward so the bloom is visible from an overhead camera, which is the only
    // angle this world is ever seen from.
    const petals = 4;
    const spread = 0.05 + rng() * 0.03;
    const shade = rng() < 0.3 ? petal.clone().lerp(heart, 0.4) : petal;
    for (let p = 0; p < petals; p++) {
      const yaw = (p / petals) * Math.PI * 2 + rng() * 0.3;
      const cos = Math.cos(yaw);
      const sin = Math.sin(yaw);
      mesher.triangle(
        [x, height, z],
        [x + cos * spread - sin * spread * 0.6, height + 0.02, z + sin * spread + cos * spread * 0.6],
        [x + cos * spread + sin * spread * 0.6, height + 0.02, z + sin * spread - cos * spread * 0.6],
        shade
      );
    }
    mesher.triangle(
      [x - 0.014, height + 0.025, z - 0.014],
      [x + 0.019, height + 0.025, z - 0.006],
      [x, height + 0.025, z + 0.019],
      heart
    );
  }
  return mesher.build();
}

/** A low frond cluster. Seven broad, nearly flat blades. */
function fern(earth: THREE.Color, rng: Rng): THREE.BufferGeometry {
  const mesher = new Mesher();
  const dark = earth.clone().lerp(new THREE.Color('#2f5a33'), 0.7);
  for (let i = 0; i < 7; i++) {
    const yaw = (i / 7) * Math.PI * 2 + rng() * 0.5;
    const height = 0.14 + rng() * 0.12;
    mesher.blade(
      (rng() - 0.5) * 0.07,
      (rng() - 0.5) * 0.07,
      yaw,
      height,
      0.08 + rng() * 0.05,
      // Long lean, short rise: a frond arches outward rather than standing up,
      // which is the whole difference between an understorey and more grass.
      0.22 + rng() * 0.14,
      dark.clone().multiplyScalar(0.82 + rng() * 0.36)
    );
  }
  return mesher.build();
}

/** A knee-high tree: a stem and a four-sided crown. Eight triangles. */
function sapling(earth: THREE.Color, rng: Rng): THREE.BufferGeometry {
  const mesher = new Mesher();
  const bark = new THREE.Color('#4a3b30').lerp(earth, 0.3);
  const leaf = earth.clone().lerp(new THREE.Color('#42703c'), 0.72).multiplyScalar(0.85 + rng() * 0.3);

  const trunk = 0.26 + rng() * 0.16;
  mesher.blade(0, 0, 0, trunk, 0.045, 0.015, bark);
  mesher.blade(0, 0, Math.PI / 2, trunk, 0.045, 0.015, bark);

  // The crown, as a ring of leaning triangles meeting at a point. Cheaper than
  // a cone and it keeps the ragged outline a young tree has.
  const top = trunk + 0.4 + rng() * 0.26;
  const spread = 0.15 + rng() * 0.08;
  for (let i = 0; i < 6; i++) {
    const yaw = (i / 6) * Math.PI * 2 + rng() * 0.4;
    const cos = Math.cos(yaw);
    const sin = Math.sin(yaw);
    mesher.triangle(
      [0, top, 0],
      [cos * spread, trunk * 0.75, sin * spread],
      [Math.cos(yaw + 1.1) * spread, trunk * 0.75, Math.sin(yaw + 1.1) * spread],
      leaf.clone().multiplyScalar(0.8 + rng() * 0.4)
    );
  }
  return mesher.build();
}

/** Tall, thin, and nearly upright. Six triangles. */
function reeds(earth: THREE.Color, rng: Rng): THREE.BufferGeometry {
  const mesher = new Mesher();
  const straw = earth.clone().lerp(new THREE.Color('#8f9a56'), 0.62);
  for (let i = 0; i < 6; i++) {
    const height = 0.34 + rng() * 0.32;
    mesher.blade(
      (rng() - 0.5) * 0.16,
      (rng() - 0.5) * 0.16,
      rng() * Math.PI * 2,
      height,
      0.024 + rng() * 0.015,
      height * (0.1 + rng() * 0.16),
      straw.clone().multiplyScalar(0.8 + rng() * 0.42)
    );
  }
  return mesher.build();
}

/** Loose stones. Four flat tetrahedra — twelve triangles, no curvature wasted. */
function pebbles(rock: THREE.Color, rng: Rng): THREE.BufferGeometry {
  const mesher = new Mesher();
  for (let i = 0; i < 4; i++) {
    const x = (rng() - 0.5) * 0.5;
    const z = (rng() - 0.5) * 0.5;
    const size = 0.07 + rng() * 0.11;
    const top = size * (0.5 + rng() * 0.5);
    // Kept light. Stone darker than the ground it lies on reads as a hole in
    // the turf rather than as a stone, and at this size a drift of them looks
    // like a stain on the hillside.
    const shade = rock.clone().multiplyScalar(0.95 + rng() * 0.45);

    const corners: Array<[number, number, number]> = [];
    for (let c = 0; c < 3; c++) {
      const yaw = (c / 3) * Math.PI * 2 + rng() * 0.6;
      corners.push([
        x + Math.cos(yaw) * size * (0.7 + rng() * 0.6),
        0,
        z + Math.sin(yaw) * size * (0.7 + rng() * 0.6),
      ]);
    }
    const apex: [number, number, number] = [x + (rng() - 0.5) * size * 0.4, top, z + (rng() - 0.5) * size * 0.4];
    mesher.triangle(corners[0], corners[1], apex, shade);
    mesher.triangle(corners[1], corners[2], apex, shade);
    mesher.triangle(corners[2], corners[0], apex, shade);
  }
  return mesher.build();
}

// ---------------------------------------------------------------------------
// Noise — value fBm, the same shape terrain.ts uses.
// ---------------------------------------------------------------------------

/**
 * Where cover clumps and where the ground shows through.
 *
 * The frequency also feeds the seed, so two species asking at two scales get two
 * unrelated fields rather than the same drifts at two sizes — otherwise every
 * bed of flowers sits in the middle of every stand of ferns.
 */
function patch(x: number, z: number, scale: number, seed: number): number {
  const field = seed + 131 + Math.round(scale * 1000);
  return clamp01((fbm(x * scale + 7.7, z * scale - 3.1, field) - 0.3) / 0.38);
}

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
    noise2(x, y, seed) * 0.6 +
    noise2(x * 2.07 + 5.2, y * 2.07 - 1.3, seed) * 0.28 +
    noise2(x * 4.13 - 2.7, y * 4.13 + 8.1, seed) * 0.12
  );
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
