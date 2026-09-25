/**
 * The things standing on the landscape.
 *
 * Every prop is built out of boxes, cones and spheres at runtime. No assets, no
 * loader, no download — so the world is fully itself on the first frame of a
 * cold start, and stays that way offline. models.ts is the seam for replacing
 * any of this with real geometry later; nothing else needs to know which kind a
 * given prop turned out to be.
 *
 * Three rules every builder follows, because the placement code in mindscape.ts
 * relies on all three and has no way to check them:
 *
 *   - the prop's base sits at y = 0, so it can be dropped onto terrain height
 *   - it is roughly centred on x/z, so it lands where it was asked to
 *   - it is built from `rng`, never Math.random, so the same diary always
 *     produces the same island
 *
 * That last one matters more than it looks. A world that reshuffled its own
 * furniture every time you saved an entry would not be a place you could come
 * back to, and the whole premise is that this is *yours* and it accumulates.
 *
 * Geometry is shared across every prop that uses a given primitive and scaled
 * per-mesh, so a forest of twenty trees is twenty meshes over four geometries
 * rather than eighty. Materials are pooled by colour, which keeps a landscape
 * with a hundred props down to a couple of dozen materials.
 */

import * as THREE from 'three';
import { PALETTE, type Emotion } from '../emotions';
import { motifModel, pickModel, specFor } from './models';

/**
 * Deterministic PRNG (mulberry32).
 *
 * Seeded per motif, so adding a diary entry that grows the woods does not move
 * the houses. Small, fast, and with none of the low-bit structure that
 * `sin(x) * 43758` has — which matters here because these values drive angles
 * and positions, and visible structure in a "random" scatter reads instantly as
 * a grid.
 */
export type Rng = () => number;

export function makeRng(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Stable 32-bit hash of a string, for seeding an RNG from a motif id. */
export function hashString(text: string): number {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

interface PieceOptions {
  pos?: [number, number, number];
  rot?: [number, number, number];
  scale?: [number, number, number] | number;
  rough?: number;
  metal?: number;
  emissive?: THREE.ColorRepresentation;
  emissiveIntensity?: number;
  opacity?: number;
}

/**
 * Natural colours, before the emotion tint.
 *
 * Kept muted. These sit under a sky that is already carrying the emotional
 * colour of the whole diary, and a saturated green forest under a violet sky
 * reads as two pictures rather than one place.
 */
const HUES = {
  bark: '#4a3b30',
  leaf: '#4c7a4a',
  leafWarm: '#6d854a',
  stone: '#6b6f7a',
  stoneDark: '#4c515c',
  snow: '#dfe6f2',
  sand: '#c9b189',
  wood: '#8a6b4a',
  paper: '#d8d2c4',
  clay: '#9a6a58',
  roof: '#54424a',
  glass: '#8fb4d9',
  lamp: '#ffd9a0',
  cloth: '#6f7bab',
  petal: '#d97a9b',
  fur: '#a08260',
} as const;

/**
 * How far a prop's natural colour is pulled toward the motif's emotion.
 *
 * A third, not more. Past roughly 0.45 a tree stops being a tree that feels
 * sad and becomes a blue cone — the emotional reading wins and the subject is
 * lost, which is the wrong trade for the one layer of the app whose whole job
 * is to show you *what* you talked about rather than how it felt.
 */
const TINT = 0.32;
const TINT_STRONG = 0.5;

export class PropFactory {
  /** Unit primitives, scaled per mesh. Owned here, disposed with the factory. */
  private geo = {
    box: new THREE.BoxGeometry(1, 1, 1),
    cylinder: new THREE.CylinderGeometry(0.5, 0.5, 1, 10),
    cone: new THREE.ConeGeometry(0.5, 1, 9),
    pyramid: new THREE.ConeGeometry(0.5, 1, 4),
    wedge: new THREE.ConeGeometry(0.5, 1, 3),
    rock: new THREE.IcosahedronGeometry(0.5, 0),
    blob: new THREE.IcosahedronGeometry(0.5, 1),
    ball: new THREE.IcosahedronGeometry(0.5, 2),
    capsule: new THREE.CapsuleGeometry(0.5, 1, 4, 8),
    ring: new THREE.TorusGeometry(0.5, 0.14, 8, 18),
  };

  private materials = new Map<string, THREE.MeshStandardMaterial>();

  /**
   * Build one prop for a motif.
   *
   * A registered .glb wins over the procedural builder — that is the entire
   * contract models.ts advertises, and it lives here so no caller has to ask.
   *
   * @param modelKey Overrides which model to look up, leaving the procedural
   *                 fallback on the motif. The room uses it to turn the woods
   *                 into a houseplant; see `indoor` in motifs.ts.
   */
  build(motifId: string, rng: Rng, emotion: Emotion, modelKey = motifId): THREE.Object3D {
    // A key that has variants resolves to one of them here, from the caller's
    // own seeded generator — so two trees in the same wood are two trees.
    const key = pickModel(modelKey, rng);
    const spec = specFor(key);
    if (spec) {
      const model = motifModel(
        key,
        spec.tint ? { color: new THREE.Color(PALETTE[emotion].base), amount: spec.tint } : null
      );
      if (model) {
        // Turned to a random bearing, exactly as a procedural prop is. Without
        // it a row of downloaded trees all face the same way and the eye reads
        // the repetition instantly.
        model.rotation.y += rng() * Math.PI * 2;
        return model;
      }
    }

    const mood = new THREE.Color(PALETTE[emotion].base);
    const glow = new THREE.Color(PALETTE[emotion].glow);
    const builder = BUILDERS[motifId] ?? BUILDERS.stones;
    const prop = builder(this, rng, mood, glow);
    prop.rotation.y += rng() * Math.PI * 2;
    return prop;
  }

  /** A mesh of one shared primitive, in a pooled material. */
  piece(
    geometry: keyof PropFactory['geo'],
    color: THREE.Color,
    options: PieceOptions = {}
  ): THREE.Mesh {
    const mesh = new THREE.Mesh(this.geo[geometry], this.material(color, options));
    if (options.pos) mesh.position.set(...options.pos);
    if (options.rot) mesh.rotation.set(...options.rot);
    if (options.scale !== undefined) {
      if (typeof options.scale === 'number') mesh.scale.setScalar(options.scale);
      else mesh.scale.set(...options.scale);
    }
    return mesh;
  }

  /**
   * Pooled material.
   *
   * Keyed on everything that can differ, so two props asking for the same look
   * share one material and two asking for different looks never collide. The
   * emissive key includes its intensity because a window glowing at 0.4 and a
   * beacon glowing at 2.0 are otherwise indistinguishable to the pool.
   */
  private material(color: THREE.Color, options: PieceOptions): THREE.MeshStandardMaterial {
    const rough = options.rough ?? 0.85;
    const metal = options.metal ?? 0.05;
    const emissive = options.emissive ?? '#000000';
    const intensity = options.emissiveIntensity ?? 1;
    const opacity = options.opacity ?? 1;
    const key = `${color.getHexString()}|${rough}|${metal}|${new THREE.Color(
      emissive
    ).getHexString()}|${intensity}|${opacity}`;

    let material = this.materials.get(key);
    if (!material) {
      material = new THREE.MeshStandardMaterial({
        color: color.clone(),
        roughness: rough,
        metalness: metal,
        emissive: new THREE.Color(emissive),
        emissiveIntensity: intensity,
        transparent: opacity < 1,
        opacity,
      });
      this.materials.set(key, material);
    }
    return material;
  }

  /** A natural colour pulled toward the motif's emotion. */
  tint(hue: string, mood: THREE.Color, amount = TINT): THREE.Color {
    return new THREE.Color(hue).lerp(mood, amount);
  }

  dispose(): void {
    for (const geometry of Object.values(this.geo)) geometry.dispose();
    for (const material of this.materials.values()) material.dispose();
    this.materials.clear();
  }
}

type Builder = (
  f: PropFactory,
  rng: Rng,
  mood: THREE.Color,
  glow: THREE.Color
) => THREE.Object3D;

/**
 * Marks a prop as one that drifts.
 *
 * Read by Mindscape.update(). Only a handful of things float — notes, boats, a
 * cloud — because a landscape where everything bobs looks like it is underwater
 * rather than alive.
 */
function floats(object: THREE.Object3D, rng: Rng, amp: number, speed: number): void {
  object.userData.float = { amp, speed, phase: rng() * Math.PI * 2, baseY: 0 };
}

const BUILDERS: Record<string, Builder> = {
  /**
   * Books: stacks, plus the occasional one left open.
   *
   * Built as stacks rather than shelves because a shelf needs a wall, and a wall
   * on open ground reads as a ruin. A pile of books on grass reads as someone
   * working outdoors, which is nearer the truth of a thesis.
   */
  study(f, rng, mood) {
    const group = new THREE.Group();
    const count = 3 + Math.floor(rng() * 4);
    let y = 0;

    for (let i = 0; i < count; i++) {
      const height = 0.16 + rng() * 0.1;
      const width = 1.0 + rng() * 0.5;
      const depth = 0.75 + rng() * 0.3;
      // Each cover takes a different amount of the emotion colour, so a stack
      // has the variety a row of real spines does without needing a palette.
      const cover = f.tint(HUES.paper, mood, 0.25 + rng() * 0.45);
      group.add(
        f.piece('box', cover, {
          pos: [(rng() - 0.5) * 0.14, y + height / 2, (rng() - 0.5) * 0.14],
          rot: [0, (rng() - 0.5) * 0.5, 0],
          scale: [width, height, depth],
          rough: 0.7,
        })
      );
      y += height;
    }

    if (rng() < 0.45) {
      const page = f.tint(HUES.paper, mood, 0.12);
      for (const sign of [-1, 1]) {
        group.add(
          f.piece('box', page, {
            pos: [sign * 0.42, y + 0.16, 0],
            rot: [0, 0, sign * 0.26],
            scale: [0.86, 0.05, 0.78],
            rough: 0.9,
          })
        );
      }
    }
    group.scale.setScalar(1.3 + rng() * 0.4);
    return group;
  },

  /** Conifers and broadleaves, mixed so a wood doesn't look planted. */
  forest(f, rng, mood) {
    const group = new THREE.Group();
    const conifer = rng() < 0.55;
    const height = 3.4 + rng() * 2.6;
    const bark = f.tint(HUES.bark, mood, TINT * 0.6);
    const foliage = f.tint(rng() < 0.5 ? HUES.leaf : HUES.leafWarm, mood, TINT);

    const trunkHeight = height * (conifer ? 0.3 : 0.45);
    group.add(
      f.piece('cylinder', bark, {
        pos: [0, trunkHeight / 2, 0],
        scale: [0.26 + rng() * 0.1, trunkHeight, 0.26 + rng() * 0.1],
        rough: 0.95,
      })
    );

    if (conifer) {
      const tiers = 3;
      for (let i = 0; i < tiers; i++) {
        const t = i / tiers;
        group.add(
          f.piece('cone', foliage, {
            pos: [0, trunkHeight + height * 0.62 * t, 0],
            scale: [2.3 * (1 - t * 0.42), height * 0.5 * (1 - t * 0.22), 2.3 * (1 - t * 0.42)],
            rough: 0.9,
          })
        );
      }
    } else {
      const blobs = 2 + Math.floor(rng() * 2);
      for (let i = 0; i < blobs; i++) {
        const size = (1.9 + rng() * 0.9) * (1 - i * 0.16);
        group.add(
          f.piece('blob', foliage, {
            pos: [
              (rng() - 0.5) * 0.9,
              trunkHeight + height * 0.22 + i * height * 0.16,
              (rng() - 0.5) * 0.9,
            ],
            scale: [size, size * 0.82, size],
            rough: 0.9,
          })
        );
      }
    }
    return group;
  },

  /**
   * The shoreline: driftwood, pebbles, and now and then a boat.
   *
   * The water itself is terrain, not a prop — a sea made of props would end up
   * as a ring of blue discs. These are the things that tell you the water is
   * water, placed where the land meets it.
   */
  sea(f, rng, mood, glow) {
    const group = new THREE.Group();

    if (rng() < 0.3) {
      const hull = f.tint(HUES.wood, mood, TINT);
      const sail = f.tint('#e8eaf2', mood, TINT_STRONG);
      group.add(
        f.piece('box', hull, { pos: [0, 0.22, 0], scale: [1.9, 0.42, 0.72], rough: 0.7 })
      );
      group.add(
        f.piece('cylinder', hull, { pos: [0, 1.3, 0], scale: [0.09, 2.0, 0.09], rough: 0.7 })
      );
      group.add(
        f.piece('wedge', sail, {
          pos: [0.34, 1.35, 0],
          rot: [0, 0, -0.1],
          scale: [1.5, 1.9, 0.08],
          rough: 0.6,
          emissive: glow,
          emissiveIntensity: 0.12,
        })
      );
      floats(group, rng, 0.14, 0.7);
      return group;
    }

    const pebbles = 3 + Math.floor(rng() * 4);
    for (let i = 0; i < pebbles; i++) {
      const size = 0.35 + rng() * 0.7;
      group.add(
        f.piece('rock', f.tint(rng() < 0.4 ? HUES.sand : HUES.stone, mood, TINT * 0.7), {
          pos: [(rng() - 0.5) * 2.6, size * 0.28, (rng() - 0.5) * 2.6],
          rot: [rng(), rng() * 3, rng()],
          scale: [size, size * 0.55, size * 0.85],
          rough: 0.95,
        })
      );
    }
    if (rng() < 0.5) {
      group.add(
        f.piece('cylinder', f.tint(HUES.wood, mood, TINT * 0.5), {
          pos: [0, 0.2, 0],
          rot: [0, rng() * 3, Math.PI / 2 + (rng() - 0.5) * 0.3],
          scale: [0.2, 2.2, 0.2],
          rough: 1,
        })
      );
    }
    return group;
  },

  /**
   * A rocky outcrop, with snow above the line. Few and large *relative to the
   * ground* — the tallest thing in the room, not a mountain range. At the scale
   * this world is framed at, an actual mountain would be the only thing on
   * screen and everything else would be pebbles at its foot.
   */
  mountains(f, rng, mood) {
    const group = new THREE.Group();
    const height = 3.4 + rng() * 2;
    const radius = 1.5 + rng() * 0.8;
    const rock = f.tint(HUES.stoneDark, mood, TINT * 0.8);

    group.add(
      f.piece('cone', rock, {
        pos: [0, height / 2, 0],
        rot: [0, rng() * 3, 0],
        scale: [radius * 2, height, radius * 1.7],
        rough: 1,
      })
    );
    group.add(
      f.piece('cone', f.tint(HUES.snow, mood, TINT * 0.5), {
        pos: [0, height * 0.79, 0],
        rot: [0, rng() * 3, 0],
        scale: [radius * 0.82, height * 0.42, radius * 0.7],
        rough: 0.75,
      })
    );
    // A shoulder, so the silhouette isn't a perfect cone from every angle.
    if (rng() < 0.7) {
      const side = height * 0.45;
      group.add(
        f.piece('cone', rock, {
          pos: [radius * 0.95, side / 2, radius * 0.4],
          rot: [0, rng() * 3, 0],
          scale: [radius * 1.2, side, radius],
          rough: 1,
        })
      );
    }
    return group;
  },

  /** A small house with a lit window — the only prop that reads as "someone's in". */
  home(f, rng, mood, glow) {
    const group = new THREE.Group();
    const width = 2.1 + rng() * 0.8;
    const bodyHeight = 1.7 + rng() * 0.6;
    const depth = 1.8 + rng() * 0.6;
    const wall = f.tint(HUES.clay, mood, TINT);

    group.add(
      f.piece('box', wall, {
        pos: [0, bodyHeight / 2, 0],
        scale: [width, bodyHeight, depth],
        rough: 0.9,
      })
    );
    group.add(
      f.piece('pyramid', f.tint(HUES.roof, mood, TINT * 0.8), {
        pos: [0, bodyHeight + width * 0.34, 0],
        rot: [0, Math.PI / 4, 0],
        scale: [width * 1.44, width * 0.68, depth * 1.5],
        rough: 0.95,
      })
    );
    // Emissive rather than lit: it has to glow from across the island, and a
    // real light here would mean one point light per house.
    group.add(
      f.piece('box', f.tint(HUES.lamp, mood, 0.2), {
        pos: [0, bodyHeight * 0.55, depth / 2 + 0.02],
        scale: [width * 0.28, bodyHeight * 0.3, 0.06],
        rough: 0.4,
        emissive: glow,
        emissiveIntensity: 0.55,
      })
    );
    if (rng() < 0.4) {
      group.add(
        f.piece('cylinder', f.tint(HUES.roof, mood, TINT), {
          pos: [width * 0.28, bodyHeight + width * 0.5, 0],
          scale: [0.24, 0.9, 0.24],
          rough: 1,
        })
      );
    }
    return group;
  },

  /**
   * A cluster of towers, speckled with windows.
   *
   * Kept short and thin. The words that feed this motif — work, deadline, boss —
   * are the ones a hard week is full of, so it is the motif most likely to come
   * out strongest, and the least deserving of being the only thing you can see.
   * Nothing here should be able to tower over the woods.
   */
  city(f, rng, mood, glow) {
    const group = new THREE.Group();
    const blocks = 1 + Math.floor(rng() * 2);
    const concrete = f.tint(HUES.stone, mood, TINT * 0.55);

    for (let i = 0; i < blocks; i++) {
      const height = 2 + rng() * 2.2;
      const width = 0.65 + rng() * 0.4;
      const x = (rng() - 0.5) * 1.5;
      const z = (rng() - 0.5) * 1.5;
      group.add(
        f.piece('box', concrete, {
          pos: [x, height / 2, z],
          scale: [width, height, width],
          rough: 0.8,
          metal: 0.2,
        })
      );
      // Windows as a vertical stripe rather than a grid of quads: one extra
      // mesh per tower instead of thirty, and at this distance it reads the same.
      group.add(
        f.piece('box', f.tint(HUES.glass, mood, 0.3), {
          pos: [x, height * 0.52, z + width / 2 + 0.01],
          scale: [width * 0.34, height * 0.74, 0.04],
          rough: 0.25,
          metal: 0.4,
          emissive: glow,
          emissiveIntensity: 0.35,
        })
      );
    }
    return group;
  },

  /**
   * Small figures, always in twos and threes, turned toward each other.
   *
   * A single figure standing alone on a hill would be a statement this data
   * cannot support. A group reads as company, which is what the words that
   * trigger this motif are mostly about.
   */
  people(f, rng, mood) {
    const group = new THREE.Group();
    const count = 2 + Math.floor(rng() * 3);
    const radius = 0.7 + rng() * 0.6;

    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2 + rng() * 0.4;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const height = 1.5 + rng() * 0.45;
      const cloth = f.tint(HUES.cloth, mood, TINT_STRONG + rng() * 0.2);
      const skin = f.tint('#c8a68c', mood, TINT * 0.6);

      // Proportions matter more than detail at two meshes per figure. A wide
      // capsule with a big sphere resting on its shoulder is a chess pawn, not
      // a person — the body has to be narrow relative to its height, and the
      // head has to be small enough to leave a visible neck.
      group.add(
        f.piece('capsule', cloth, {
          // CapsuleGeometry(0.5, 1) is two units tall, so the Y scale is half
          // the height it produces. This body spans 0 → 0.78h.
          pos: [x, height * 0.39, z],
          scale: [height * 0.2, height * 0.39, height * 0.2],
          rough: 0.9,
        })
      );
      group.add(
        f.piece('ball', skin, {
          pos: [x, height * 0.9, z],
          scale: height * 0.23,
          rough: 0.85,
        })
      );
    }
    return group;
  },

  /** A plinth with notes drifting off it. The notes are what move. */
  music(f, rng, mood, glow) {
    const group = new THREE.Group();
    const stone = f.tint(HUES.stone, mood, TINT);

    group.add(
      f.piece('cylinder', stone, { pos: [0, 0.3, 0], scale: [1.5, 0.6, 1.5], rough: 0.9 })
    );
    group.add(
      f.piece('ring', f.tint('#e6d9a8', mood, TINT_STRONG), {
        pos: [0, 0.75, 0],
        rot: [Math.PI / 2, 0, 0],
        scale: 1.7,
        rough: 0.35,
        metal: 0.6,
        emissive: glow,
        emissiveIntensity: 0.3,
      })
    );

    const notes = 2 + Math.floor(rng() * 3);
    for (let i = 0; i < notes; i++) {
      const note = new THREE.Group();
      const color = f.tint('#f0e6d0', mood, TINT_STRONG);
      note.add(
        f.piece('ball', color, {
          scale: [0.34, 0.26, 0.26],
          rot: [0, 0, -0.35],
          rough: 0.4,
          emissive: glow,
          emissiveIntensity: 0.5,
        })
      );
      note.add(
        f.piece('box', color, {
          pos: [0.14, 0.42, 0],
          scale: [0.06, 0.8, 0.06],
          rough: 0.4,
          emissive: glow,
          emissiveIntensity: 0.5,
        })
      );
      note.position.set((rng() - 0.5) * 2.2, 1.5 + rng() * 1.8, (rng() - 0.5) * 2.2);
      floats(note, rng, 0.3, 0.9 + rng() * 0.6);
      group.add(note);
    }
    return group;
  },

  /** A ball and a goal. Reads from a distance; nothing else needs to. */
  sport(f, rng, mood, glow) {
    const group = new THREE.Group();
    const frame = f.tint('#e4e8f0', mood, TINT * 0.6);
    const width = 2.2;
    const height = 1.4;

    for (const sign of [-1, 1]) {
      group.add(
        f.piece('cylinder', frame, {
          pos: [sign * width / 2, height / 2, 0],
          scale: [0.16, height, 0.16],
          rough: 0.5,
          metal: 0.35,
        })
      );
    }
    group.add(
      f.piece('cylinder', frame, {
        pos: [0, height, 0],
        rot: [0, 0, Math.PI / 2],
        scale: [0.16, width, 0.16],
        rough: 0.5,
        metal: 0.35,
      })
    );
    group.add(
      f.piece('ball', f.tint('#f2f4fb', mood, TINT), {
        pos: [(rng() - 0.5) * 1.8, 0.3, 1.0 + rng() * 1.1],
        scale: 0.6,
        rough: 0.55,
        emissive: glow,
        emissiveIntensity: 0.15,
      })
    );
    return group;
  },

  /** Flowers. Small, many, scattered thickly — this is ground cover, not furniture. */
  garden(f, rng, mood, glow) {
    const group = new THREE.Group();
    const stems = 2 + Math.floor(rng() * 3);
    const stem = f.tint(HUES.leaf, mood, TINT * 0.6);

    for (let i = 0; i < stems; i++) {
      const height = 0.7 + rng() * 0.6;
      const x = (rng() - 0.5) * 0.7;
      const z = (rng() - 0.5) * 0.7;
      group.add(
        f.piece('cylinder', stem, {
          pos: [x, height / 2, z],
          rot: [(rng() - 0.5) * 0.3, 0, (rng() - 0.5) * 0.3],
          scale: [0.06, height, 0.06],
          rough: 0.95,
        })
      );
      group.add(
        f.piece('ring', f.tint(HUES.petal, mood, TINT_STRONG), {
          pos: [x, height + 0.06, z],
          rot: [Math.PI / 2 + (rng() - 0.5) * 0.4, 0, 0],
          scale: 0.46,
          rough: 0.7,
          emissive: glow,
          emissiveIntensity: 0.2,
        })
      );
      group.add(
        f.piece('ball', f.tint('#f5e3a0', mood, TINT), {
          pos: [x, height + 0.06, z],
          scale: 0.18,
          rough: 0.6,
        })
      );
    }
    return group;
  },

  /** An animal: body, head, ears, tail. Four meshes and unmistakably a creature. */
  pets(f, rng, mood) {
    const group = new THREE.Group();
    const fur = f.tint(HUES.fur, mood, TINT);
    const scale = 0.85 + rng() * 0.5;

    group.add(
      f.piece('blob', fur, { pos: [0, 0.62, 0], scale: [1.5, 0.95, 0.95], rough: 0.95 })
    );
    group.add(
      f.piece('ball', fur, { pos: [0.85, 1.0, 0], scale: 0.72, rough: 0.95 })
    );
    for (const sign of [-1, 1]) {
      group.add(
        f.piece('wedge', fur, {
          pos: [0.86, 1.32, sign * 0.2],
          scale: [0.3, 0.36, 0.3],
          rough: 0.95,
        })
      );
      group.add(
        f.piece('cylinder', fur, {
          pos: [sign * 0.35, 0.22, 0.3],
          scale: [0.17, 0.46, 0.17],
          rough: 0.95,
        })
      );
      group.add(
        f.piece('cylinder', fur, {
          pos: [sign * 0.35, 0.22, -0.3],
          scale: [0.17, 0.46, 0.17],
          rough: 0.95,
        })
      );
    }
    group.add(
      f.piece('cylinder', fur, {
        pos: [-0.8, 0.9, 0],
        rot: [0, 0, 0.9],
        scale: [0.13, 0.85, 0.13],
        rough: 0.95,
      })
    );
    group.scale.setScalar(scale);
    return group;
  },

  /**
   * Scattered stones — the fallback, and what an empty diary's island is made of.
   *
   * Worth having rather than showing bare ground: a landscape with nothing on it
   * looks broken, and a few rocks look like a place that hasn't been built yet.
   */
  stones(f, rng, mood) {
    const group = new THREE.Group();
    const count = 1 + Math.floor(rng() * 3);
    for (let i = 0; i < count; i++) {
      const size = 0.6 + rng() * 1.5;
      group.add(
        f.piece('rock', f.tint(rng() < 0.5 ? HUES.stone : HUES.stoneDark, mood, TINT * 0.5), {
          pos: [(rng() - 0.5) * 2.2, size * 0.32, (rng() - 0.5) * 2.2],
          rot: [rng() * 3, rng() * 3, rng() * 3],
          scale: [size, size * 0.72, size * 0.9],
          rough: 1,
        })
      );
    }
    return group;
  },
};
