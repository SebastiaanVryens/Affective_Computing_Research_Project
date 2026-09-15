/**
 * The mind: the island you see when the app opens, and the shaft down to the
 * memories underneath it.
 *
 * The world has two floors. This is the upper one — the place the diary has
 * *made*, built out of what you keep talking about. The memory orbs are the
 * lower one: the raw record, one sphere per entry, unsummarised. You arrive at
 * the top, because "here is what you have become" is a better first sentence
 * than "here are four hundred data points", and you can drop through the middle
 * to the evidence whenever you want to check the summary against it.
 *
 * Everything on the island is derived and disposable. Nothing here is stored,
 * nothing here is a source of truth, and deleting an entry takes its tree away
 * with it — the landscape is a *view* of the diary in exactly the way the sky
 * already is. That is also why placement is seeded rather than random: a derived
 * view has to come back the same way twice, or it stops being a place.
 */

import * as THREE from 'three';
import { PALETTE, type EmotionVector, charge } from '../emotions';
import type { DiaryEntry } from '../state/db';
import { chooseBiome, type BiomeChoice } from './biomes';
import { GroundCover } from './groundcover';
import { Horizon } from './horizon';
import { LAYER } from './layers';
import type { ThreadAnchor } from './threads';
import { MAX_MOTIFS, detectMotifs, type MotifPresence } from './motifs';
import { PropFactory, hashString, makeRng, type Rng } from './props';
import { Room } from './room';
import type { ViewKind } from './view';
import { Well } from './well';
import {
  ISLAND_RADIUS,
  Terrain,
  WELL_RADIUS,
  type TerrainShape,
} from './terrain';

/** The island's floor. The camera's mind-view rig is measured from here. */
export const MIND_LEVEL = 0;

/**
 * How far below the island the memory orbs sit.
 *
 * Far enough that the descent is a journey rather than a hop, near enough that
 * the light from the shaft still reaches the orbs and the two floors read as one
 * world. Tuned against the transition length in scene.ts: much deeper and the
 * fall has to speed up past the point where you can see what you passed.
 */
export const CORE_LEVEL = -78;

/** Golden angle — the same trick placement.ts uses, for the same spread. */
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

const BEACON_RING = ISLAND_RADIUS * 0.58;
/** Taller than anything that grows here, short enough not to be a lamp post. */
const BEACON_HEIGHT = 3.6;

/** Inner and outer bounds props are allowed to stand between. */
const PROP_MIN_RADIUS = WELL_RADIUS + 1.6;
const PROP_MAX_RADIUS = ISLAND_RADIUS * 0.8;

/**
 * Global size of everything standing on the ground.
 *
 * The builders in props.ts are authored at natural, human-legible sizes — a tree
 * is a few metres, a book is a hand — and this is the one place that decides how
 * big a metre is here. It exists as a single number because the alternative was
 * fifty tuned constants spread across eleven builders, and the thing that
 * actually needed adjusting turned out to be the scale of the whole room rather
 * than the proportions of anything in it.
 */
const PROP_SCALE = 0.7;

/** Stand-in sea level for biomes that have no water. See findSpot. */
const DRY_DATUM = -0.6;

interface Placed {
  x: number;
  z: number;
  gap: number;
}

interface Floater {
  object: THREE.Object3D;
  amp: number;
  speed: number;
  phase: number;
  baseY: number;
}

export class Mindscape {
  readonly group = new THREE.Group();

  /** Island, props and beacons. Hidden once the camera is down at the core. */
  private island = new THREE.Group();

  private terrain = new Terrain();
  private horizon = new Horizon();
  private cover = new GroundCover();
  private well = new Well();
  private room = new Room();
  private props = new PropFactory();
  private propGroup = new THREE.Group();
  private beaconGroup = new THREE.Group();

  private floaters: Floater[] = [];
  private beacons: Array<{ mesh: THREE.Mesh; material: THREE.MeshBasicMaterial; phase: number }> = [];
  private pickProxies: THREE.Mesh[] = [];
  private anchors: ThreadAnchor[] = [];

  private motifs: MotifPresence[] = [];
  private place: BiomeChoice = chooseBiome([]);
  private restlessness = 0.3;
  /** Last haze colour seen, so a rebuild can paint the window to match the sky. */
  private haze = new THREE.Color('#6f7793');
  private ownedMaterials: THREE.Material[] = [];
  private ownedGeometries: THREE.BufferGeometry[] = [];

  constructor() {
    this.island.add(
      this.terrain.group,
      this.horizon.group,
      this.cover.group,
      this.well.group,
      this.room.group,
      this.propGroup,
      this.beaconGroup
    );
    this.group.add(this.island);
  }

  getMotifs(): MotifPresence[] {
    return this.motifs;
  }

  /** Which kind of place the diary has turned out to be. */
  getPlace(): BiomeChoice {
    return this.place;
  }

  /**
   * How much this world should move, in [0, 1].
   *
   * Read by the camera. A settled diary gets a world that essentially holds
   * still; an unsettled one gets a world that turns. See `restlessnessOf`.
   */
  getRestlessness(): number {
    return this.restlessness * this.place.biome.orbitScale;
  }

  // -- build -----------------------------------------------------------

  /**
   * Rebuild the whole island from the diary.
   *
   * Called on load and after every save. Cheap enough to do wholesale — a
   * hundred props over a dozen shared geometries — and wholesale is what keeps
   * it honest, because an incremental version would eventually disagree with the
   * data it was built from and nobody would be able to tell.
   */
  rebuild(entries: DiaryEntry[], lifetimeTotals: EmotionVector): void {
    this.motifs = detectMotifs(entries);
    this.place = chooseBiome(this.motifs);
    this.restlessness = restlessnessOf(entries);
    const shape = shapeFor(entries, this.motifs, this.place);
    const shareOf = (id: string): number =>
      this.motifs.find((m) => m.motif.id === id)?.share ?? 0;
    const highGround = shareOf('mountains');

    this.terrain.rebuild(shape, lifetimeTotals);

    if (shape.biome.id === 'room') {
      // Indoors replaces the landscape rather than sitting on it: no ground, no
      // sea, no distance. The well stays, because a hole in the floor of the
      // room you live in is a better image than a hole in a hillside.
      this.horizon.clear();
      this.cover.clear();
      this.room.rebuild(lifetimeTotals, viewKindFor(this.motifs), this.haze, shape.seed);
    } else {
      this.room.clear();
      // The far range answers to the same theme the island's outcrops do, so
      // talking about climbing raises the whole world rather than adding rocks
      // to one lawn. The same for the woods: the trees on the far hills are the
      // same fact as the trees on the island, seen from further away.
      this.horizon.rebuild(
        shape,
        Math.min(1, highGround * 2.2),
        lifetimeTotals,
        Math.min(1, shareOf('forest') * 2.4)
      );
      this.cover.rebuild(shape, this.motifs, (x, z) =>
        Math.hypot(x, z) < ISLAND_RADIUS ? this.terrain.heightAt(x, z) : this.horizon.heightAt(x, z)
      );
    }

    // The well belongs to every biome, indoors included — a hole in the floor of
    // the room you live in is as much a way down as a hole in a hillside. Its
    // rim is sampled just outside the ground's inner edge so the shaft hangs
    // from whatever height the landform actually put there.
    this.well.rebuild(
      this.terrain.heightAt(WELL_RADIUS + 0.06, 0),
      shape.biome.ground.rock,
      lifetimeTotals,
      shape.biome.id === 'room' ? 'square' : 'round'
    );

    this.buildProps(shape);
    this.buildBeacons(entries.filter((e) => e.isCoreMemory), shape);
  }

  /**
   * Scatter each motif's props across its own arc of the island.
   *
   * Sectors rather than a uniform sprinkle, because the point is legibility: the
   * woods should be somewhere, so that "the woods have grown" is a thing you can
   * see happen. A motif's arc is proportional to its share, so the thing you
   * talk about most is also the thing the island is mostly made of.
   */
  private buildProps(shape: TerrainShape): void {
    this.clearProps();

    const placed: Placed[] = [];
    // Only the strongest few actually get floor space. Everything weaker still
    // exists and still counts elsewhere — toward the biome, and toward what is
    // outside the window — it just has nowhere to stand.
    const standing = this.motifs.slice(0, MAX_MOTIFS);
    // Sector widths are floored before being renormalised, so a motif with a 3%
    // share still gets a strip wide enough to put three trees in rather than a
    // sliver they'd all have to stack inside.
    const widths = standing.map((m) => Math.max(0.08, m.share));
    const totalWidth = widths.reduce((sum, w) => sum + w, 0) || 1;

    let angle = (shape.seed % 360) * (Math.PI / 180);
    const indoors = shape.biome.id === 'room';

    standing.forEach((presence, index) => {
      const arc = (widths[index] / totalWidth) * Math.PI * 2;
      const rng = makeRng(hashString(presence.motif.id) ^ shape.seed);

      // Indoors, a motif can decline to appear or appear as something else.
      // The arc is still consumed either way, so declining leaves a gap rather
      // than shuffling every other motif around the floor — which would mean
      // the room rearranged itself whenever an outdoor theme came and went.
      const indoor = presence.motif.indoor;
      if (indoors && indoor === false) {
        angle += arc;
        return;
      }
      const modelKey = indoors && typeof indoor === 'string' ? indoor : presence.motif.id;

      for (let i = 0; i < presence.propCount; i++) {
        const spot = this.findSpot(rng, angle, arc, presence.motif.prefer, presence.motif.spacing, shape, placed);
        if (!spot) continue;

        const prop = this.props.build(presence.motif.id, rng, presence.emotion, modelKey);
        prop.position.set(spot.x, spot.y - 0.06, spot.z);
        prop.scale.multiplyScalar(PROP_SCALE * (0.85 + rng() * 0.35));
        // A few degrees of lean, so nothing looks like it was placed by a
        // machine that only knows about vertical.
        prop.rotation.x = (rng() - 0.5) * 0.07;
        prop.rotation.z = (rng() - 0.5) * 0.07;

        this.collectFloaters(prop);
        this.propGroup.add(prop);
        placed.push({ x: spot.x, z: spot.z, gap: presence.motif.spacing });
      }

      angle += arc;
    });

    // An island with nothing on it looks broken rather than new. Stones say
    // "not built yet", which is the true thing to say on day one.
    if (placed.length === 0) {
      const rng = makeRng(shape.seed ^ 0x5eed);
      for (let i = 0; i < 9; i++) {
        const spot = this.findSpot(rng, i * GOLDEN_ANGLE, 0.9, 'any', 2.4, shape, placed);
        if (!spot) continue;
        const prop = this.props.build('stones', rng, 'neutral');
        prop.position.set(spot.x, spot.y - 0.06, spot.z);
        prop.scale.multiplyScalar(PROP_SCALE);
        this.propGroup.add(prop);
        placed.push({ x: spot.x, z: spot.z, gap: 2.4 });
      }
    }
  }

  /**
   * Rejection-sample a standing spot inside one motif's arc.
   *
   * The first 18 tries demand the motif's preferred ground; the rest will settle
   * for anywhere dry. A strict-only version silently drops whole motifs on
   * islands that happen not to have the landform they wanted — a flat diary has
   * no high ground, and "you talked about mountains, so there are no mountains"
   * is the worst possible outcome.
   */
  private findSpot(
    rng: Rng,
    fromAngle: number,
    arc: number,
    prefer: string,
    spacing: number,
    shape: TerrainShape,
    placed: Placed[]
  ): { x: number; y: number; z: number } | null {
    for (let attempt = 0; attempt < 30; attempt++) {
      const strict = attempt < 18;
      const a = fromAngle + rng() * arc;
      // sqrt keeps areal density even; without it everything crowds the middle.
      const r = PROP_MIN_RADIUS + Math.sqrt(rng()) * (PROP_MAX_RADIUS - PROP_MIN_RADIUS);
      const x = Math.cos(a) * r;
      const z = Math.sin(a) * r;
      const y = this.terrain.heightAt(x, z);
      // Dry biomes have no sea to measure against, so they borrow a coast's
      // datum. The ground preferences below are about *relative* height — low
      // ground, high ground — and they need some fixed line to be relative to.
      const above = y - (shape.waterLevel ?? DRY_DATUM);

      if (strict ? !suitsGround(prefer, above) : above < 0.35) continue;
      // Nothing stands on a cliff face.
      if (this.terrain.slopeAt(x, z) > 0.78) continue;

      let clear = true;
      for (const other of placed) {
        const gap = Math.max(spacing, other.gap);
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

  /**
   * A light for every core memory, standing on the island.
   *
   * The one place the two floors touch: these are the same entries as the
   * brightest orbs below, so the shape of what mattered is visible from the top
   * without having to go down for it. Clicking one opens the entry, exactly as
   * clicking its orb does.
   */
  private buildBeacons(cores: DiaryEntry[], shape: TerrainShape): void {
    this.clearBeacons();

    cores.forEach((entry, i) => {
      const angle = i * GOLDEN_ANGLE;
      const radius = BEACON_RING + ((i % 3) - 1) * 1.1;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const y = Math.max(this.terrain.heightAt(x, z), (shape.waterLevel ?? -9e9) + 0.2);

      const glow = new THREE.Color(PALETTE[entry.dominant].glow);
      // Pushed past 1 so the bloom pass in scene.ts sees it. The palette's glow
      // colours are pale by design, and at their face value they sit just under
      // the threshold — which would leave a "beacon" that doesn't glow.
      const column = this.basicMaterial(glow.clone().multiplyScalar(1.7), 0.34);
      const cap = this.basicMaterial(glow.clone().multiplyScalar(2.1), 0.8);

      // Strongly tapered — wide where it leaves the ground, almost nothing at
      // the top. A parallel-sided column with a ball on it is a street lamp; a
      // cone that thins as it rises is light leaving the ground, which is the
      // thing a core memory is supposed to look like from up here.
      const shaftGeo = this.own(new THREE.CylinderGeometry(0.04, 0.34, BEACON_HEIGHT, 10, 1, true));
      const beam = new THREE.Mesh(shaftGeo, column);
      beam.position.set(x, y + BEACON_HEIGHT / 2, z);

      const orbGeo = this.own(new THREE.IcosahedronGeometry(0.2, 2));
      const orb = new THREE.Mesh(orbGeo, cap);
      orb.position.set(x, y + BEACON_HEIGHT, z);

      const ringGeo = this.own(new THREE.TorusGeometry(0.62, 0.05, 6, 28));
      const ring = new THREE.Mesh(ringGeo, column);
      ring.position.set(x, y + 0.1, z);
      ring.rotation.x = -Math.PI / 2;

      // A fat invisible cylinder to click. The beam itself is a couple of pixels
      // wide at the distance the mind view is framed at, which is not a target
      // anyone can hit.
      const proxyGeo = this.own(new THREE.CylinderGeometry(0.85, 0.85, BEACON_HEIGHT + 0.6, 6));
      const proxy = new THREE.Mesh(proxyGeo, this.basicMaterial(glow, 0));
      proxy.position.set(x, y + BEACON_HEIGHT / 2, z);
      proxy.userData.entryId = entry.id;

      this.anchors.push({
        entryId: entry.id,
        top: new THREE.Vector3(x, y + BEACON_HEIGHT, z),
        color: glow.clone(),
      });

      // In front of the sea, always. These are additive and write no depth, so
      // without an explicit order the water is drawn over them and a core memory
      // standing against the sea is simply erased — see ./layers.ts.
      for (const part of [beam, orb, ring, proxy]) part.renderOrder = LAYER.glow;

      this.beacons.push({ mesh: orb, material: cap, phase: i * 0.7 });
      this.beacons.push({ mesh: ring, material: column, phase: i * 0.7 + 1.1 });
      this.pickProxies.push(proxy);
      this.beaconGroup.add(beam, orb, ring, proxy);
    });
  }

  // -- per frame -------------------------------------------------------

  /**
   * @param sky The scene's haze colour. Handed down so the sea reflects exactly
   *            what the fog is made of, which is what lets the far water vanish
   *            into the horizon instead of ending at one.
   */
  update(
    delta: number,
    elapsed: number,
    mood: { color: string; arousal: number; clarity: number },
    sky: THREE.Color
  ): void {
    this.haze.copy(sky);
    this.terrain.update(delta, elapsed, mood, sky);
    this.cover.update(elapsed, mood.arousal);
    this.well.update(elapsed);
    this.room.update(elapsed);

    for (const f of this.floaters) {
      f.object.position.y = f.baseY + Math.sin(elapsed * f.speed + f.phase) * f.amp;
      f.object.rotation.y += delta * 0.25;
    }

    for (const beacon of this.beacons) {
      const pulse = 0.78 + 0.22 * Math.sin(elapsed * 1.05 + beacon.phase);
      beacon.material.opacity = beacon.material.userData.baseOpacity * pulse;
      beacon.mesh.scale.setScalar(1 + (pulse - 0.9) * 0.18);
    }
  }

  /**
   * Hide the island once the camera is fully at the core.
   *
   * Not an optimisation — the sea is hundreds of units across and sits directly
   * overhead down there, so leaving it on would paint a translucent ceiling over
   * the entire sky. The memory threads are not in this group, because seeing the
   * lines you came down is what keeps the two floors one world.
   */
  setDescent(blend: number): void {
    this.island.visible = blend < 0.9;
  }

  /**
   * The top of each core memory's beacon, for the threads to hang from.
   *
   * Recomputed on rebuild rather than derived on demand: the beacons sit on
   * terrain, and the only place that knows where the terrain was when they were
   * placed is the pass that placed them.
   */
  getThreadAnchors(): ThreadAnchor[] {
    return this.anchors;
  }

  /** Ray-pick a core-memory beacon. Returns the entry id, as the orbs do. */
  pick(raycaster: THREE.Raycaster): string | null {
    if (!this.island.visible) return null;
    const hits = raycaster.intersectObjects(this.pickProxies, false);
    return hits.length > 0 ? (hits[0].object.userData.entryId as string) : null;
  }

  // -- plumbing --------------------------------------------------------

  private collectFloaters(prop: THREE.Object3D): void {
    prop.traverse((node) => {
      const spec = node.userData.float as
        | { amp: number; speed: number; phase: number }
        | undefined;
      if (!spec) return;
      this.floaters.push({
        object: node,
        amp: spec.amp,
        speed: spec.speed,
        phase: spec.phase,
        baseY: node.position.y,
      });
    });
  }

  /**
   * An unlit material for the glowing things.
   *
   * Basic rather than standard on purpose: beacons and shaft rings are light
   * sources in the fiction, and a lit material would let the island's key light
   * decide how bright a core memory looks depending on which way it happened to
   * be facing. `baseOpacity` is stashed on the material because update() pulses
   * opacity and needs something stable to pulse around.
   */
  private basicMaterial(
    color: THREE.Color,
    opacity: number,
    side: THREE.Side = THREE.FrontSide
  ): THREE.MeshBasicMaterial {
    const material = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side,
      // Never fogged. These are additive, so "fog" would mix the haze colour in
      // and then *add* it — distant shaft rings would get brighter as they
      // receded, which is the opposite of the thing fog is for.
      fog: false,
    });
    material.userData.baseOpacity = opacity;
    this.ownedMaterials.push(material);
    return material;
  }

  private own<T extends THREE.BufferGeometry>(geometry: T): T {
    this.ownedGeometries.push(geometry);
    return geometry;
  }

  private clearProps(): void {
    for (const child of [...this.propGroup.children]) this.propGroup.remove(child);
    this.floaters = [];
  }

  private clearBeacons(): void {
    for (const child of [...this.beaconGroup.children]) this.beaconGroup.remove(child);
    this.beacons = [];
    this.pickProxies = [];
    this.anchors = [];
  }

  dispose(): void {
    this.clearProps();
    this.clearBeacons();
    this.terrain.dispose();
    this.horizon.dispose();
    this.cover.dispose();
    this.well.dispose();
    this.room.dispose();
    this.props.dispose();
    for (const material of this.ownedMaterials) material.dispose();
    for (const geometry of this.ownedGeometries) geometry.dispose();
    this.ownedMaterials = [];
    this.ownedGeometries = [];
  }
}

/**
 * How much the world should move, from how much the diary does.
 *
 * Two things, both about motion rather than about valence: how much feeling the
 * entries carry at all, and how much that swings from one to the next. A run of
 * even, low-charge days produces a world that is almost still — which is the
 * right image for it, and also a kindness, because a landscape that circles
 * relentlessly is tiring to sit in front of while you are trying to talk.
 *
 * Deliberately *not* about whether the feeling is good or bad. A steady stretch
 * of sadness should be as still as a steady stretch of contentment; turning
 * someone's low week into a spinning world would be editorialising with the
 * camera.
 */
function restlessnessOf(entries: DiaryEntry[]): number {
  if (entries.length === 0) return 0.2;

  const meanCharge =
    entries.reduce((sum, e) => sum + charge(e.vector), 0) / entries.length;

  // Swing measured on *valence*, not on charge.
  //
  // Charge is "how much is happening", and it barely moves between a furious
  // entry and an elated one — both are far from neutral, so a diary that lurches
  // from a row with your boss to a promotion scores almost flat on it. Valence
  // is what actually lurches. Using it is the difference between this reading
  // "an eventful few weeks" and reading "some weeks".
  let swing = 0;
  for (let i = 1; i < entries.length; i++) {
    swing += Math.abs(valenceOf(entries[i]) - valenceOf(entries[i - 1]));
  }
  swing /= Math.max(1, entries.length - 1);

  // Squared, so the quiet end of the range is genuinely quiet. A linear response
  // leaves an unremarkable diary turning a full circle every few minutes, which
  // is exactly the constant motion this is here to stop.
  const raw = 0.45 * meanCharge + 0.75 * swing;
  return Math.max(0, Math.min(1, raw * raw + 0.06));
}

/** Positive mass minus negative mass, in [-1, 1]. Mirrors state/report.ts. */
function valenceOf(entry: DiaryEntry): number {
  const v = entry.vector;
  return v.joy - (v.sadness + v.anger + v.fear + v.disgust);
}

/**
 * What to put outside the window.
 *
 * Nature decides it. `city` only ever appears as a fallback, and that asymmetry
 * is the whole point of this function rather than an accident of it.
 *
 * The naive version ranked all five outdoor motifs together and took the
 * largest. That is wrong, because `city` is not really an outdoor motif — it is
 * "the working world", the bundle of office/commute/deadline words that is half
 * the reason the room biome was chosen in the first place. So in any diary that
 * picked a room, city is close to the top by construction, and it beat the
 * genuine outdoor theme every time: somebody who wrote about the sea every
 * weekend for months got a skyline. (Measured: city 0.233 against sea 0.178. The
 * one case that appeared to work, forest, was a floating-point tie resolved by
 * iteration order — luck, not logic.)
 *
 * The other four are unambiguous. Nobody says "beach" about their desk. So if
 * any of them is genuinely present, it is what is out there, and the only thing
 * they compete with is each other.
 */
function viewKindFor(motifs: MotifPresence[]): ViewKind {
  const nature: Array<[string, ViewKind]> = [
    ['sea', 'coast'],
    ['mountains', 'peaks'],
    ['forest', 'forest'],
    ['garden', 'meadow'],
  ];

  let best: ViewKind | null = null;
  let bestShare = MIN_VIEW_SHARE;
  for (const [motifId, kind] of nature) {
    const share = motifs.find((m) => m.motif.id === motifId)?.share ?? 0;
    if (share > bestShare) {
      bestShare = share;
      best = kind;
    }
  }
  if (best) return best;

  // A skyline is the honest default: somebody whose diary is entirely books and
  // deadlines, with no outdoors in it at all, is statistically looking at a town.
  return 'city';
}

/**
 * How much of the diary a nature theme needs before it takes the window.
 *
 * Low, but not zero. A motif already has to recur across entries to exist at
 * all, so this only guards against one stray weekend outranking a life — it is
 * not meant to be a high bar, and a real habit clears it easily.
 */
const MIN_VIEW_SHARE = 0.08;

/** Whether a height above water suits a motif's preferred ground. */
function suitsGround(prefer: string, above: number): boolean {
  switch (prefer) {
    case 'low':
      return above > 0.25 && above < 1.5;
    case 'high':
      return above > 1.2;
    case 'shore':
      return above > -0.1 && above < 0.8;
    default:
      return above > 0.25;
  }
}

/**
 * Read the landform out of the diary.
 *
 * The biome has already decided *what kind of place* this is; this decides how
 * that place is shaped, from two things the data actually says:
 *
 *   relief — mean emotional charge, scaled by the biome. A diary full of strong
 *            feeling gets a landscape with relief; a long flat stretch gets flat
 *            ground. This is the clearest way the ground answers to the orbs
 *            below it, and the only place feeling touches the landform at all.
 *   water  — how much the person talks about it, within whatever range the
 *            biome allows. A coast's sea rises and floods its lowlands; a wood's
 *            pond gets a little deeper. Somewhere dry stays dry.
 *
 * The seed is hashed from the first entry's id so it is stable for the life of
 * the diary: your ground keeps its shape, and adding entries changes what grows
 * on it rather than rearranging everything underneath.
 */
function shapeFor(
  entries: DiaryEntry[],
  motifs: MotifPresence[],
  place: BiomeChoice
): TerrainShape {
  const seed = hashString(entries[0]?.id ?? 'empty-world') % 100000;

  const meanCharge = entries.length
    ? entries.reduce((sum, e) => sum + charge(e.vector), 0) / entries.length
    : 0.3;

  const shareOf = (id: string): number =>
    motifs.find((m) => m.motif.id === id)?.share ?? 0;

  const base = place.biome.waterLevel;

  return {
    seed,
    // Stable per diary, like everything else about the landform. Derived from
    // the seed rather than chosen, so two people's worlds are oriented
    // differently and neither of them is oriented on purpose.
    grain: (seed % 6283) / 1000,
    relief: (0.2 + 0.6 * Math.min(1, meanCharge * 1.3)) * place.biome.reliefScale,
    // The biome sets the level; the sea motif can lift it within a narrow band,
    // so a coast that is genuinely all about the water floods further inland.
    waterLevel: base === null ? null : base + shareOf('sea') * 1.1,
    biome: place.biome,
  };
}
