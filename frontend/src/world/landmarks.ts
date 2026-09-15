/**
 * The wider world: what stands outside the ring of props.
 *
 * ./props.ts furnishes a dozen units of ground around the well, and past that
 * the landscape was empty — correct terrain with nothing on it, which reads as a
 * backdrop rather than as somewhere. This puts objects out into the middle
 * distance and the far field: a watchtower on a ridge, a boat on the water, a
 * ruin in a field, a small island off the coast.
 *
 * ── Why these objects and not others ────────────────────────────────────────
 *
 * They are not chosen for looking good at distance. Each one is a *landscape
 * feature with a documented psychological reading*, and the readings come from
 * four bodies of work that between them cover why people like the places they
 * like:
 *
 *   Prospect–refuge (Appleton, The Experience of Landscape, 1975). People
 *   prefer places that offer both an unimpeded view — prospect — and somewhere
 *   to be unseen — refuge. "Seeing without being seen." A watchtower is the
 *   purest prospect object there is; a cave or a tent is pure refuge.
 *
 *   The preference matrix (Kaplan & Kaplan). Four predictors of landscape
 *   preference: coherence and legibility, which help you make sense of a place,
 *   and complexity and mystery, which make you want to explore it. Mystery is
 *   specifically information promised but withheld — a path bending out of
 *   sight. A gate, a signpost and a bridge are legibility objects; a path over a
 *   rise is a mystery object.
 *
 *   Attention Restoration Theory (Kaplan). Restorative places have being away,
 *   extent, soft fascination and compatibility. Soft fascination is attention
 *   held effortlessly by something undemanding — moving water, a turning sail.
 *   That is what the windmill, the watermill and the waterfall are for.
 *
 *   Favourite places (Korpela et al.). When people name the places they go to
 *   put themselves back together, waterside environments and managed woodland
 *   come top, and the reported experience is being away from everyday life and
 *   reflecting on personal matters. The jetty, the bench and the small island
 *   are that finding, drawn.
 *
 * ── The line this file does not cross ───────────────────────────────────────
 *
 * The brief that started this was "a watchtower, if you like to stand and watch,
 * not really interact". That is a tempting rule and it is the wrong one, for
 * exactly the reason ./motifs.ts refuses to let mood pick the biome: it would be
 * telling somebody a conclusion about their personality, in a language they
 * cannot argue with, inferred from a diary by a program.
 *
 * So every landmark here fires on *words the person actually said*, through the
 * same motif lexicon everything else uses. The watchtower appears because you
 * talked about watching, noticing, looking out over things — which is auditable,
 * and which you can disagree with by reading your own transcript. It does not
 * appear because something decided you are an observer rather than a
 * participant. The meaning column below is why the object was chosen to stand
 * for that theme; it is not a claim being made about the person.
 *
 * ── Placement ───────────────────────────────────────────────────────────────
 *
 * Two bands past the props. `mid` is the island's shoulder and the near far
 * field, close enough to resolve detail; `far` is silhouette country. Height
 * comes from the same sampler the ground cover uses, so a landmark on the far
 * hills sits on the far hills. Water-footed landmarks float at sea level over
 * genuinely deep water, which is what keeps boats off the beach.
 */

import * as THREE from 'three';
import { PALETTE, type Emotion } from '../emotions';
import type { BiomeId } from './biomes';
import { motifModel } from './models';
import type { MotifPresence } from './motifs';
import { makeRng, type Rng } from './props';
import { ISLAND_RADIUS, alongGrain, type TerrainShape } from './terrain';

/** Where a landmark is allowed to stand. */
type Footing =
  /** Dry ground, any height. */
  | 'land'
  /** High ground — ridges and tops. Prospect objects want this. */
  | 'high'
  /** Within a stride of the waterline, either side. */
  | 'shore'
  /** Floating on water deep enough not to be a beach. */
  | 'water'
  /**
   * Dry land on the seaward side — which, on a coast, can only be the island
   * offshore. `high` is not enough: the inland hills are high too, and palms
   * were turning up along the mountain ridges behind the beach.
   */
  | 'offshore';

type Band = 'mid' | 'far';

interface Landmark {
  id: string;
  label: string;
  /**
   * What it stands for, and whose finding that is.
   *
   * Kept in the code rather than in a document because it is the justification
   * for the object existing, and a justification that lives somewhere else gets
   * out of step with the list it justifies.
   */
  meaning: string;
  /** Key into MOTIF_MODELS. */
  model: string;
  /** Height in world units after normalisation. */
  height: number;
  /** Which places it belongs in. Empty means anywhere outdoors. */
  biomes: BiomeId[];
  band: Band;
  footing: Footing;
  /**
   * The motif whose share decides how many appear, or null for scenery that
   * belongs to the biome rather than to anything the person said.
   */
  motif: string | null;
  /** Most that can ever appear, at full share. */
  max: number;
  /** Minimum gap from any other landmark, in world units. */
  spacing: number;
}

/**
 * The register.
 *
 * Ordered by the reading each one carries rather than by biome, because the
 * reading is the thing being claimed and it should be possible to audit the
 * whole set of claims by reading down the page.
 */
export const LANDMARKS: Landmark[] = [
  // -- prospect: somewhere to stand and look out from ---------------------
  {
    id: 'watchtower',
    label: 'Watchtower',
    meaning:
      'Prospect, in Appleton\'s sense: a place to see from without being seen. ' +
      'The strongest single prospect object in a landscape vocabulary.',
    model: 'land:watchtower',
    height: 6.0,
    biomes: ['forest', 'meadow', 'alpine', 'bare'],
    band: 'mid',
    footing: 'high',
    motif: 'forest',
    max: 2,
    spacing: 22,
  },
  {
    id: 'crag',
    label: 'Crag',
    meaning:
      'Natural prospect. Higher than everything around it and unbuilt, so it ' +
      'reads as a viewpoint that was always there rather than one somebody made.',
    model: 'land:crag',
    height: 4.0,
    biomes: ['alpine', 'coast', 'bare'],
    band: 'mid',
    footing: 'high',
    motif: 'mountains',
    max: 5,
    spacing: 12,
  },
  {
    id: 'bench',
    label: 'Bench',
    meaning:
      'Prospect at rest, and the plainest marker of a favourite place: somewhere ' +
      'a person stops to sit and look. Korpela\'s accounts of restorative places ' +
      'are full of these.',
    model: 'land:bench',
    height: 0.7,
    biomes: ['meadow', 'coast', 'forest'],
    band: 'mid',
    footing: 'high',
    motif: 'people',
    max: 3,
    spacing: 16,
  },

  // -- refuge: somewhere to be unseen -------------------------------------
  {
    id: 'tent',
    label: 'Camp',
    meaning:
      'Refuge — shelter you can see out of. Appleton\'s pair is only satisfied ' +
      'when both halves are present, so the landscape needs these as much as it ' +
      'needs the tower.',
    model: 'land:tent',
    height: 1.6,
    biomes: ['forest', 'meadow', 'alpine'],
    band: 'mid',
    footing: 'land',
    motif: 'forest',
    max: 3,
    spacing: 14,
  },
  {
    id: 'campfire',
    label: 'Campfire',
    meaning:
      'The hearth of a refuge, and soft fascination in its own right — fire is ' +
      'the textbook example of attention held without being demanded.',
    model: 'land:campfire',
    height: 0.7,
    biomes: ['forest', 'meadow', 'alpine'],
    band: 'mid',
    footing: 'land',
    motif: 'forest',
    max: 2,
    spacing: 14,
  },
  {
    id: 'cave',
    label: 'Cave mouth',
    meaning:
      'Refuge at its most literal, and Appleton\'s own recurring example. Dark, ' +
      'enclosed, and a thing the eye returns to.',
    model: 'land:cave',
    height: 3.0,
    biomes: ['alpine', 'forest'],
    band: 'mid',
    footing: 'high',
    motif: 'mountains',
    max: 2,
    spacing: 20,
  },

  // -- legibility and mystery: how a place is read ------------------------
  {
    id: 'bridge',
    label: 'Bridge',
    meaning:
      'Legibility — it says the landscape can be crossed here and not elsewhere. ' +
      'Also the oldest transition image there is, which is why it is placed at ' +
      'the waterline rather than anywhere scenic.',
    model: 'land:bridge',
    height: 1.2,
    biomes: ['forest', 'alpine', 'meadow', 'coast'],
    band: 'mid',
    footing: 'shore',
    motif: null,
    max: 2,
    spacing: 18,
  },
  {
    id: 'gate',
    label: 'Gate',
    meaning:
      'A threshold. Kaplan\'s mystery in its cheapest form: a way through, with ' +
      'what is on the other side withheld.',
    model: 'land:gate',
    height: 1.2,
    biomes: ['meadow', 'forest', 'bare'],
    band: 'mid',
    footing: 'land',
    motif: 'garden',
    max: 3,
    spacing: 15,
  },
  {
    id: 'signpost',
    label: 'Signpost',
    meaning:
      'Pure legibility — somebody has been here and left the way marked. In an ' +
      'empty landscape it is the difference between country and wilderness.',
    model: 'land:signpost',
    height: 1.4,
    biomes: ['forest', 'meadow', 'alpine', 'bare'],
    band: 'mid',
    footing: 'land',
    motif: 'sport',
    max: 3,
    spacing: 16,
  },
  {
    id: 'cairn',
    label: 'Cairn',
    meaning:
      'A waymark on high ground: legibility where there is no path to follow. ' +
      'Also the mark of other people having passed, on ground that otherwise ' +
      'says nobody comes here.',
    model: 'land:cairn',
    height: 1.8,
    biomes: ['alpine', 'meadow', 'bare'],
    band: 'mid',
    footing: 'high',
    motif: 'mountains',
    max: 4,
    spacing: 13,
  },

  // -- soft fascination: things that move by themselves --------------------
  {
    id: 'windmill',
    label: 'Windmill',
    meaning:
      'Soft fascination. Slow, repetitive, undemanding motion is the mechanism ' +
      'Attention Restoration Theory puts at the centre of recovery.',
    model: 'land:windmill',
    height: 9.0,
    biomes: ['meadow', 'coast', 'bare'],
    band: 'far',
    footing: 'high',
    motif: null,
    max: 2,
    spacing: 40,
  },
  {
    id: 'watermill',
    label: 'Watermill',
    meaning:
      'The same, with water: the other half of the pairing Korpela found people ' +
      'return to when they choose somewhere to feel better in.',
    model: 'land:watermill',
    height: 3.6,
    biomes: ['forest', 'meadow'],
    band: 'mid',
    footing: 'shore',
    motif: 'sea',
    max: 1,
    spacing: 30,
  },
  {
    id: 'waterfall',
    label: 'Waterfall',
    meaning:
      'Soft fascination, and the one landscape feature that carries a sound with ' +
      'it even in silence.',
    model: 'land:waterfall',
    height: 6.0,
    biomes: ['alpine', 'forest'],
    band: 'far',
    footing: 'high',
    motif: 'mountains',
    max: 3,
    spacing: 26,
  },

  // -- waterside: the most-named favourite place ---------------------------
  {
    id: 'jetty',
    label: 'Jetty',
    meaning:
      'A built edge to the water — somewhere to stand at the end of. Waterside ' +
      'environments are the favourite-place category people reselect most, and ' +
      'this is what they look like when somebody uses one.',
    model: 'land:jetty',
    height: 0.6,
    biomes: ['coast', 'forest', 'alpine'],
    band: 'mid',
    footing: 'shore',
    motif: 'sea',
    max: 2,
    spacing: 18,
  },
  {
    id: 'boat',
    label: 'Rowing boat',
    meaning:
      'A journey at rest. Small, unmanned and near the shore: it says a crossing ' +
      'is possible, which is not the same as saying one is happening.',
    model: 'land:boat',
    height: 0.6,
    biomes: ['coast', 'forest', 'alpine'],
    band: 'mid',
    footing: 'water',
    motif: 'sea',
    max: 3,
    spacing: 12,
  },
  {
    id: 'ship',
    label: 'Ship',
    meaning:
      'Extent, in the Attention Restoration sense — something far enough out to ' +
      'imply a world continuing past the frame.',
    model: 'land:ship',
    height: 4.5,
    biomes: ['coast'],
    band: 'far',
    footing: 'water',
    motif: 'sea',
    max: 2,
    spacing: 45,
  },
  {
    id: 'sea-rocks',
    label: 'Sea rocks',
    meaning:
      'Endurance, and the thing that turns open water into a place. Bare sea has ' +
      'no scale; a rock in it does.',
    model: 'land:sea-rocks',
    height: 1.2,
    biomes: ['coast'],
    band: 'mid',
    footing: 'water',
    motif: null,
    max: 6,
    spacing: 14,
  },
  {
    id: 'palm',
    label: 'Shore trees',
    meaning:
      'Dressing for the small island offshore, so it reads as somewhere rather ' +
      'than as a bump in the water.',
    model: 'land:palm',
    height: 3.2,
    biomes: ['coast'],
    band: 'far',
    footing: 'offshore',
    motif: 'sea',
    max: 8,
    spacing: 6,
  },

  // -- time: things that were here before ----------------------------------
  {
    id: 'ruin',
    label: 'Ruin',
    meaning:
      'The past persisting in the present. The one object here that is about ' +
      'time rather than about space, and the reason it is placed inland and ' +
      'alone is that a ruin in company is a village.',
    model: 'land:ruin',
    height: 2.6,
    biomes: ['meadow', 'forest', 'alpine', 'bare'],
    band: 'mid',
    footing: 'land',
    motif: 'study',
    max: 3,
    spacing: 20,
  },
  {
    id: 'wreck',
    label: 'Wreck',
    meaning:
      'The same reading, at sea, and much louder — which is why it needs a real ' +
      'share of the diary behind it before it is allowed to appear.',
    model: 'land:wreck',
    height: 4.0,
    biomes: ['coast'],
    band: 'far',
    footing: 'water',
    motif: 'sea',
    max: 1,
    spacing: 50,
  },
  {
    id: 'broken-fence',
    label: 'Old fence',
    meaning:
      'Legibility gone over: a boundary somebody used to keep. Quieter than a ' +
      'ruin and it does the same job on open ground.',
    model: 'land:broken-fence',
    height: 0.9,
    biomes: ['meadow', 'bare', 'forest'],
    band: 'mid',
    footing: 'land',
    motif: null,
    max: 5,
    spacing: 11,
  },

  // -- woodland, at a size the far field can carry -------------------------
  {
    id: 'tree-pine',
    label: 'Pines',
    meaning:
      'Managed woodland is the other favourite-place category people reselect ' +
      'most. These are the individual trees near enough to resolve, in front of ' +
      'the instanced canopy further out.',
    model: 'land:tree-pine',
    height: 4.0,
    biomes: ['forest', 'alpine'],
    band: 'mid',
    footing: 'land',
    motif: 'forest',
    max: 14,
    spacing: 7,
  },
  {
    id: 'tree-oak',
    label: 'Broadleaves',
    meaning: 'The same, with a rounder crown, so a wood is not one repeated shape.',
    model: 'land:tree-oak',
    height: 3.6,
    biomes: ['forest', 'meadow'],
    band: 'mid',
    footing: 'land',
    motif: 'forest',
    max: 12,
    spacing: 8,
  },
];

/** Where each band reaches, in world units from the middle. */
const BANDS: Record<Band, [number, number]> = {
  mid: [ISLAND_RADIUS + 2.5, 40],
  far: [42, 88],
};

/** Stand-in sea level where the biome has none, matching mindscape.ts. */
const DRY_DATUM = -0.6;

export type HeightSampler = (x: number, z: number) => number;

interface Placed {
  x: number;
  z: number;
  gap: number;
}

export class Landmarks {
  readonly group = new THREE.Group();

  private placed: THREE.Object3D[] = [];
  /** Things that turn, bob or flicker. See `update`. */
  private movers: Array<{ object: THREE.Object3D; spin: number; bob: number; phase: number; baseY: number }> = [];

  /**
   * Scatter the wider world.
   *
   * @param heightAt Where the ground is at any radius — the caller owns the
   *                 question of which mesh answers it, exactly as ground cover
   *                 does.
   */
  rebuild(
    shape: TerrainShape,
    motifs: MotifPresence[],
    heightAt: HeightSampler,
    lifetimeEmotion: Emotion
  ): void {
    this.clear();
    if (shape.biome.id === 'room') return;

    const share = (id: string): number =>
      motifs.find((m) => m.motif.id === id)?.share ?? 0;
    const feeling = (id: string): Emotion =>
      motifs.find((m) => m.motif.id === id)?.emotion ?? lifetimeEmotion;

    const rng = makeRng((shape.seed ^ 0x4c414e44) >>> 0);
    const sea = shape.waterLevel ?? DRY_DATUM;
    const placed: Placed[] = [];

    for (const landmark of LANDMARKS) {
      if (landmark.biomes.length && !landmark.biomes.includes(shape.biome.id)) continue;

      // How many. A landmark tied to a motif needs that motif to be a real part
      // of the diary before more than one of it appears; scenery with no motif
      // gets a flat allowance from the biome alone.
      const strength = landmark.motif === null ? 0.55 : Math.min(1, share(landmark.motif) * 2.6);
      if (strength < 0.06) continue;
      const wanted = Math.max(1, Math.round(landmark.max * strength));

      const emotion = landmark.motif ? feeling(landmark.motif) : lifetimeEmotion;
      const [inner, outer] = BANDS[landmark.band];

      for (let i = 0; i < wanted; i++) {
        const spot = this.findSpot(rng, landmark, shape, inner, outer, sea, heightAt, placed);
        if (!spot) continue;

        const model = motifModel(
          landmark.model,
          { color: new THREE.Color(PALETTE[emotion].base), amount: 0.22 }
        );
        if (!model) break; // the .glb is missing; no point trying the rest

        // Darken.
        //
        // These kits are authored as bright, near-white low-poly pieces meant to
        // be read on a plain background. Dropped into this world at their own
        // value they are the brightest thing in frame — a white tower against a
        // dark wood pulls the eye harder than the island does, which inverts
        // what the near ground and the far ground are for. Multiplying the base
        // colour down sits them back in the picture without touching their
        // shape, and it is the same move the ground makes on its own emotion
        // tint (see `earthen` in terrain.ts).
        shade(model, SHADE);

        model.position.set(spot.x, spot.y, spot.z);
        model.rotation.y = rng() * Math.PI * 2;
        // A little lean on anything that grew or was left, none on anything
        // that was built. A tilted watchtower reads as a mistake.
        if (landmark.footing !== 'water' && landmark.id !== 'watchtower') {
          model.rotation.x = (rng() - 0.5) * 0.05;
          model.rotation.z = (rng() - 0.5) * 0.05;
        }
        model.scale.multiplyScalar(0.85 + rng() * 0.35);

        this.register(landmark, model, rng, spot.y);
        this.group.add(model);
        this.placed.push(model);
        placed.push({ x: spot.x, z: spot.z, gap: landmark.spacing });
      }
    }
  }

  /**
   * Rejection-sample a spot in one landmark's band.
   *
   * Generous with attempts because the filters are strict and the bands are
   * large: a landmark that wants high ground on a flat diary may genuinely have
   * nowhere to go, and giving up quietly is the right answer when it does.
   */
  private findSpot(
    rng: Rng,
    landmark: Landmark,
    shape: TerrainShape,
    inner: number,
    outer: number,
    sea: number,
    heightAt: HeightSampler,
    placed: Placed[]
  ): { x: number; y: number; z: number } | null {
    for (let attempt = 0; attempt < 60; attempt++) {
      const angle = rng() * Math.PI * 2;
      // sqrt keeps the density even in area rather than crowding the inner edge.
      const r = Math.sqrt(inner * inner + rng() * (outer * outer - inner * inner));
      const x = Math.cos(angle) * r;
      const z = Math.sin(angle) * r;
      const ground = heightAt(x, z);
      const above = ground - sea;

      let y = ground;
      switch (landmark.footing) {
        case 'water':
          // Deep enough not to be standing on the beach, shallow enough to be
          // somewhere rather than out in the empty middle of the sea.
          if (above > -0.6 || above < -14) continue;
          y = sea;
          break;
        case 'shore':
          if (above < -0.4 || above > 1.4) continue;
          break;
        case 'high':
          if (above < 1.2) continue;
          break;
        case 'offshore':
          // Dry, and on the far side of the grain from the land. See terrain.ts
          // for which way "seaward" runs.
          if (above < 0.8) continue;
          if (alongGrain(x, z, shape) < 0.45) continue;
          break;
        default:
          if (above < 0.4) continue;
      }

      // Nothing built stands on a cliff face.
      if (landmark.footing !== 'water' && slopeOf(heightAt, x, z) > 0.7) continue;

      let clear = true;
      for (const other of placed) {
        const gap = Math.max(landmark.spacing, other.gap);
        if ((other.x - x) ** 2 + (other.z - z) ** 2 < gap * gap) {
          clear = false;
          break;
        }
      }
      if (!clear) continue;

      return { x, y, z };
    }
    return null;
  }

  /** Give the few landmarks that should move something to move with. */
  private register(landmark: Landmark, model: THREE.Object3D, rng: Rng, baseY: number): void {
    const spin =
      landmark.id === 'windmill' ? 0.35 : landmark.id === 'watermill' ? 0.5 : 0;
    const bob = landmark.footing === 'water' && landmark.id !== 'wreck' ? 0.12 : 0;
    if (spin === 0 && bob === 0) return;

    this.movers.push({ object: model, spin, bob, phase: rng() * Math.PI * 2, baseY });
  }

  /**
   * Per frame.
   *
   * Only the soft-fascination objects and anything afloat. The whole argument
   * for the windmill is that it turns; a still one is a tower with sails on it.
   */
  update(delta: number, elapsed: number): void {
    for (const mover of this.movers) {
      if (mover.spin) mover.object.rotation.y += delta * mover.spin;
      if (mover.bob) {
        mover.object.position.y = mover.baseY + Math.sin(elapsed * 0.7 + mover.phase) * mover.bob;
        mover.object.rotation.z = Math.sin(elapsed * 0.5 + mover.phase) * 0.04;
      }
    }
  }

  clear(): void {
    for (const object of this.placed) this.group.remove(object);
    this.placed = [];
    this.movers = [];
  }

  dispose(): void {
    this.clear();
  }
}

/**
 * How far a landmark's own colours are pulled down.
 *
 * Tuned against the island's props rather than against the models: a house on
 * the island reads at roughly this value once the sky and the fog have had their
 * say, and a landmark twice as far away should not be brighter than it.
 */
const SHADE = 0.45;

/**
 * Multiply every base colour in a model.
 *
 * Safe only because `motifModel` clones the materials when a tint is asked for,
 * which every landmark does. Without that this would darken the shared template
 * and every later copy would come out darker than the last.
 */
function shade(model: THREE.Object3D, amount: number): void {
  model.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh) return;
    const material = mesh.material as THREE.MeshStandardMaterial;
    if (material?.color) material.color.multiplyScalar(amount);
  });
}

/** Steepness at a point, from the same sampler everything else uses. */
function slopeOf(heightAt: HeightSampler, x: number, z: number): number {
  const d = 0.8;
  const dx = heightAt(x + d, z) - heightAt(x - d, z);
  const dz = heightAt(x, z + d) - heightAt(x, z - d);
  return Math.min(1, Math.hypot(dx, dz) / (2 * d) / 1.4);
}
