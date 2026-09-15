/**
 * Optional .glb models for the landscape props.
 *
 * Everything in props.ts is built out of boxes and cones at runtime, which means
 * the mindscape works with no assets, no download, and no loading screen. This
 * file is the seam for replacing any of it with a real model later, one motif at
 * a time, without touching the rest of the world.
 *
 * ── To add a model ─────────────────────────────────────────────────────────
 *
 *   1. Drop the file in `frontend/public/props/`, e.g. `tree.glb`.
 *   2. Add a line to MOTIF_MODELS below:  forest: '/props/tree.glb',
 *   3. Reload. Every forest prop is now that model.
 *
 * Not `public/models/` — that path is gitignored, because it holds the face-api
 * weights that `npm run fetch-models` copies out of node_modules. These are not
 * regenerable that way and belong in the repository.
 *
 * That is the whole procedure. Nothing else needs to change: scale, origin and
 * orientation are normalised on load (see `normalise`), so a model exported from
 * anywhere lands the right way up, standing on the ground, at the size the motif
 * expects — rather than as a two-hundred-metre tree buried to its branches,
 * which is what raw glTF units usually give you.
 *
 * ── What a model keeps and what it loses ───────────────────────────────────
 *
 * A registered model keeps its own materials and therefore its own colours.
 * Setting `tint` on its entry mixes the motif's emotion into every mesh's base
 * colour, the same way a procedural prop is tinted — which works on a model with
 * plain coloured materials and does nothing useful on a textured one, since
 * there is no safe way to recolour somebody's texture without wrecking it.
 * Everything shipped in public/props is untextured, so all of it tints.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export interface ModelSpec {
  url: string;
  /** Height in world units after normalisation. Defaults to the motif's own. */
  height?: number;
  /**
   * How much of the motif's emotion colour to mix in, 0..1.
   *
   * Only meaningful on a model with plain coloured materials — which is most
   * low-poly CC0 work, and all of what ships here. A textured model keeps its
   * own colours whatever this says, because there is no safe way to recolour
   * somebody's texture without wrecking it.
   */
  tint?: number;
  /** Extra Y rotation in radians, if the model was authored facing sideways. */
  yaw?: number;
  /**
   * How much of the model's own emission to keep, 0..1. Defaults to none.
   *
   * See `tame`. Almost nothing should want this: what glows in this world is
   * decided by this world.
   */
  glow?: number;
  /**
   * Multiplies every base colour, for a pack whose palette is louder than this
   * world's. 1 leaves it alone.
   *
   * props.ts keeps its own hues deliberately muted — "a saturated green forest
   * under a violet sky reads as two pictures rather than one place" — and a
   * model that arrives at full saturation breaks that on its own. This is the
   * knob for putting it back.
   */
  shade?: number;
}

/**
 * Motif id → model, plus a few named pieces the room asks for by hand.
 *
 * Motif ids come from MOTIFS in ./motifs.ts: study, forest, sea, mountains,
 * home, city, people, music, sport, garden, pets. Anything not listed here
 * keeps the procedural builder in props.ts, which is why the app still works
 * with this map emptied out.
 *
 * Everything here is CC0 and untextured — see public/props/CREDITS.md. Heights
 * are in the same units props.ts authors at, so the global PROP_SCALE in
 * mindscape.ts applies on top and a model sits at the same size its
 * hand-built equivalent did.
 */
export const MOTIF_MODELS: Partial<Record<string, string | ModelSpec>> = {
  forest: { url: '/props/tree.glb', height: 4.6, tint: 0.3 },

  // Three tree shapes in four foliage colours each, from the Amir pack. See
  // VARIANT_SETS below for how one is chosen, and CREDITS.md for the trade:
  // these carry their colour in a texture rather than in a material, so the
  // tint has to stay light or it muddies the very thing they were brought in
  // for.
  'forest:a1': { url: '/props/amir/tree-a1.glb', shade: 0.62, height: 3.4, tint: 0.12 },
  'forest:a2': { url: '/props/amir/tree-a2.glb', shade: 0.62, height: 3.4, tint: 0.12 },
  'forest:a3': { url: '/props/amir/tree-a3.glb', shade: 0.62, height: 3.4, tint: 0.12 },
  'forest:a4': { url: '/props/amir/tree-a4.glb', shade: 0.62, height: 3.4, tint: 0.12 },
  'forest:b1': { url: '/props/amir/tree-b1.glb', shade: 0.62, height: 2.9, tint: 0.12 },
  'forest:b2': { url: '/props/amir/tree-b2.glb', shade: 0.62, height: 2.9, tint: 0.12 },
  'forest:b3': { url: '/props/amir/tree-b3.glb', shade: 0.62, height: 2.9, tint: 0.12 },
  'forest:b4': { url: '/props/amir/tree-b4.glb', shade: 0.62, height: 2.9, tint: 0.12 },
  'forest:c1': { url: '/props/amir/tree-c1.glb', shade: 0.62, height: 2.6, tint: 0.12 },
  'forest:c2': { url: '/props/amir/tree-c2.glb', shade: 0.62, height: 2.6, tint: 0.12 },
  'forest:c3': { url: '/props/amir/tree-c3.glb', shade: 0.62, height: 2.6, tint: 0.12 },
  'forest:c4': { url: '/props/amir/tree-c4.glb', shade: 0.62, height: 2.6, tint: 0.12 },
  study: { url: '/props/books.glb', height: 1.0, tint: 0.25 },
  city: { url: '/props/building.glb', height: 5.4, tint: 0.28 },
  home: { url: '/props/house.glb', height: 3.2, tint: 0.28 },
  music: { url: '/props/speaker.glb', height: 2.0, tint: 0.3 },

  // The wider world, fetched by landmarks.ts. Namespaced like the room's
  // furniture, and for the same reason: these are asked for by name, not by
  // motif, so they must never be able to collide with a motif id.
  //
  // Heights are the *authored* size in world units; landmarks.ts carries the
  // same number in its register so placement can reason about how tall a thing
  // is without loading it. Keep the two in step.
  //
  // The tint is deliberately lighter than the props'. A landmark is scenery the
  // diary has earned rather than a statement it is making, and at this distance
  // a strong emotional tint reads as coloured fog rather than as a mood.
  'land:watchtower': { url: '/props/pirate/watchtower.glb', height: 6.0, tint: 0.22 },
  'land:crag': { url: '/props/crag.glb', height: 4.0, tint: 0.22 },
  'land:bench': { url: '/props/graveyard/bench.glb', height: 0.7, tint: 0.22 },
  'land:tent': { url: '/props/survival/tent.glb', height: 1.6, tint: 0.22 },
  'land:campfire': { url: '/props/survival/campfire.glb', height: 0.7, tint: 0.22 },
  'land:cave': { url: '/props/cave.glb', height: 3.0, tint: 0.22 },
  'land:bridge': { url: '/props/bridge.glb', height: 1.2, tint: 0.22 },
  'land:gate': { url: '/props/gate.glb', height: 1.2, tint: 0.22 },
  'land:signpost': { url: '/props/survival/signpost.glb', height: 1.4, tint: 0.22 },
  'land:cairn': { url: '/props/cairn.glb', height: 1.8, tint: 0.22 },
  'land:windmill': { url: '/props/fantasy/windmill.glb', height: 9.0, tint: 0.22 },
  'land:watermill': { url: '/props/fantasy/watermill.glb', height: 3.6, tint: 0.22 },
  'land:waterfall': { url: '/props/waterfall.glb', height: 6.0, tint: 0.22 },
  'land:jetty': { url: '/props/pirate/jetty.glb', height: 0.6, tint: 0.22 },
  'land:boat': { url: '/props/pirate/boat.glb', height: 0.6, tint: 0.22 },
  'land:ship': { url: '/props/pirate/ship.glb', height: 4.5, tint: 0.22 },
  'land:sea-rocks': { url: '/props/pirate/sea-rocks.glb', height: 1.2, tint: 0.22 },
  'land:palm': { url: '/props/palm.glb', height: 3.2, tint: 0.22 },
  'land:ruin': { url: '/props/ruin.glb', height: 2.6, tint: 0.22 },
  'land:wreck': { url: '/props/pirate/wreck.glb', height: 4.0, tint: 0.22 },
  'land:broken-fence': { url: '/props/fantasy/broken-fence.glb', height: 0.9, tint: 0.22 },
  'land:tree-pine': { url: '/props/tree-pine.glb', height: 4.0, tint: 0.22 },
  'land:tree-oak': { url: '/props/tree-oak.glb', height: 3.6, tint: 0.22 },

  // Fixed furniture, fetched by room.ts rather than by a motif. Namespaced so
  // it can never collide with a motif id.
  'room:bookcase': { url: '/props/bookcase.glb', height: 2.3, tint: 0.2 },
  'room:desk': { url: '/props/desk.glb', height: 1.5, tint: 0.2 },
  'room:sofa': { url: '/props/sofa.glb', height: 1.2, tint: 0.3 },
  'room:lamp': { url: '/props/lamp.glb', height: 2.5, tint: 0.2 },
  'room:plant': { url: '/props/plant.glb', height: 1.2, tint: 0.2 },
};

/**
 * Keys that come in interchangeable variants, and what they may be drawn from.
 *
 * A wood built from one model is one tree repeated, and the eye finds the
 * repetition long before it finds anything else — rotating each copy to a
 * random bearing buys a little, but only a little, because the silhouette and
 * the colour are still identical. Three shapes in four foliage colours is
 * twelve visibly different trees, which is past the point where a stand of them
 * reads as a stand rather than as a pattern.
 *
 * Interchangeable is the load-bearing word. Every entry in a set has to be able
 * to stand anywhere any other one could, because the choice is made per prop
 * from the placement RNG and nothing downstream knows which it got. That is
 * also why the variants are *only* variety here and carry no meaning: a colour
 * that meant something would need to be chosen by the diary rather than by a
 * dice roll, and this picks with a dice roll.
 */
const VARIANT_SETS: Record<string, string[]> = {
  forest: [
    'forest:a1', 'forest:a2', 'forest:a3', 'forest:a4',
    'forest:b1', 'forest:b2', 'forest:b3', 'forest:b4',
    'forest:c1', 'forest:c2', 'forest:c3', 'forest:c4',
  ],
  'land:tree-oak': [
    'forest:a1', 'forest:a2', 'forest:a3', 'forest:a4',
    'forest:b1', 'forest:b2', 'forest:b3', 'forest:b4',
    'forest:c1', 'forest:c2', 'forest:c3', 'forest:c4',
  ],
};

/**
 * Which model a caller should actually ask for.
 *
 * Returns the key unchanged when it has no variants, so every call site can go
 * through this and none of them has to know which keys are sets and which are
 * single models — the registry stays the only place that knows.
 *
 * @param rng The caller's own seeded generator, so the wood is the same wood
 *            next time the diary is opened. Never Math.random: a forest that
 *            reshuffled its own species on every save would stop being a place.
 */
export function pickModel(key: string, rng: () => number): string {
  const set = VARIANT_SETS[key];
  if (!set || set.length === 0) return key;
  return set[Math.min(set.length - 1, Math.floor(rng() * set.length))];
}

/** Loaded, normalised templates. Cloned per prop; never added to a scene. */
const templates = new Map<string, THREE.Object3D>();
let loaded: Promise<void> | null = null;

/**
 * Load every registered model.
 *
 * Called once at startup. Resolves immediately when MOTIF_MODELS is empty, which
 * is the default, so the app's cold start is unaffected by this file existing.
 *
 * A model that fails to load is logged and skipped rather than thrown: a missing
 * asset should cost you a nicer tree, not the whole world.
 */
export function preloadMotifModels(): Promise<void> {
  if (loaded) return loaded;

  const entries = Object.entries(MOTIF_MODELS).filter(
    (entry): entry is [string, string | ModelSpec] => entry[1] !== undefined
  );
  if (entries.length === 0) {
    loaded = Promise.resolve();
    return loaded;
  }

  const loader = new GLTFLoader();
  loaded = Promise.all(
    entries.map(async ([id, value]) => {
      const spec = typeof value === 'string' ? { url: value } : value;
      try {
        const gltf = await loader.loadAsync(spec.url);
        tame(gltf.scene, spec);
        templates.set(id, normalise(gltf.scene, spec));
      } catch (error) {
        console.warn(`Model for motif "${id}" failed to load:`, error);
      }
    })
  ).then(() => undefined);

  return loaded;
}

/** The spec for a motif, so callers can read `tint` without re-parsing. */
export function specFor(motifId: string): ModelSpec | null {
  const value = MOTIF_MODELS[motifId];
  if (!value) return null;
  return typeof value === 'string' ? { url: value } : value;
}

/**
 * A fresh copy of a motif's model, or null if it has none.
 *
 * Cloned rather than shared because each prop gets its own position, scale and
 * (when tinting) its own materials. `clone(true)` shares geometry between copies
 * — which is what we want — and shares materials too, so tinting clones the
 * material explicitly.
 */
export function motifModel(
  motifId: string,
  tint: { color: THREE.Color; amount: number } | null
): THREE.Object3D | null {
  const template = templates.get(motifId);
  if (!template) return null;

  const copy = template.clone(true);
  if (tint) {
    copy.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (!mesh.isMesh) return;
      const source = mesh.material as THREE.Material | THREE.Material[];
      const one = Array.isArray(source) ? source[0] : source;
      const cloned = one.clone() as THREE.MeshStandardMaterial;
      if (cloned.color) cloned.color.lerp(tint.color, tint.amount);
      mesh.material = cloned;
    });
  }
  return copy;
}

/** True if at least one model actually loaded, for the disposal path. */
export function hasModels(): boolean {
  return templates.size > 0;
}

/**
 * Take a downloaded model's word for its shape, but not for its lighting.
 *
 * Asset packs are authored against whatever renderer their maker had open, and
 * emission is where that leaks most. The Amir trees ship with an emission map,
 * a white emissive factor and `KHR_materials_emissive_strength: 20` — a glow
 * twenty times full brightness, which is presumably lovely in the scene it was
 * built for and here renders the tree as a solid magenta blob with its base
 * colour nowhere to be seen. That is what the first import of them looked like.
 *
 * It would also have been worse than it looked. This scene's bloom threshold is
 * tuned so that only things that are *meant* to be light clear it — the memory
 * beacons and the orbs, which raise their colours past 1 deliberately (see
 * scene.ts). A tree at emissive strength 20 does not just glow, it blooms, and
 * a wood of them would wash out the island.
 *
 * So the default is to discard it entirely and let this world decide what
 * glows. `glow` in the spec is the opt-out, for the day something arrives that
 * genuinely should emit.
 */
function tame(scene: THREE.Object3D, spec: ModelSpec): void {
  const keep = spec.glow ?? 0;
  const shade = spec.shade ?? 1;

  scene.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh) return;

    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const material of materials as THREE.MeshStandardMaterial[]) {
      if (!material || material.emissiveIntensity === undefined) continue;
      material.emissiveIntensity = keep;
      if (keep === 0) {
        // Intensity alone is not enough to be sure: it multiplies the emissive
        // colour, and a loader or a later clone that resets it to 1 would bring
        // the glow straight back. Removing the map and blacking the colour
        // means there is nothing left to turn up.
        material.emissiveMap = null;
        material.emissive?.setScalar(0);
      }
      // Metalness, for the same reason and with a nastier failure.
      //
      // glTF defaults `metallicFactor` to *one* when it is left out, and plenty
      // of exporters leave it out. A metal has no diffuse response — all of its
      // colour comes from what it reflects — so a fully metallic material in a
      // scene with no environment map reflects nothing and renders as good as
      // black, with its base colour texture invisible behind it. This scene has
      // no environment map.
      //
      // Nothing in this world is meant to look like metal, so the safe answer
      // is that nothing is. The roughness map stays; only the metalness channel
      // of it goes.
      material.metalness = 0;
      material.metalnessMap = null;

      // And the palette, where the pack is louder than the world it landed in.
      if (shade !== 1 && material.color) material.color.multiplyScalar(shade);

      material.needsUpdate = true;
    }
  });
}

/**
 * Put an arbitrary glTF scene into the world's terms.
 *
 * Three transforms, in this order, and the order matters: measure, then scale,
 * then recentre. Recentring before scaling would put the model's base at the
 * origin of a bounding box that is about to change size.
 *
 * The result is a model that is `height` units tall, centred on X/Z, and sitting
 * on y = 0 — which is the contract every procedural builder in props.ts also
 * follows, so the placement code never has to know which kind it got.
 */
function normalise(scene: THREE.Object3D, spec: ModelSpec): THREE.Object3D {
  const wrapper = new THREE.Group();

  const box = new THREE.Box3().setFromObject(scene);
  const size = box.getSize(new THREE.Vector3());
  const target = spec.height ?? 2;

  // Normalised by height, except when the model is plainly not a standing
  // thing. A rug or a tabletop is a few centimetres tall and metres wide, so
  // scaling it to a two-unit *height* makes it the size of a building. Falling
  // back to the widest horizontal dimension keeps those merely wrong rather
  // than catastrophic — and a zero-height export (a single plane, an empty
  // scene) would otherwise divide to infinity.
  const widest = Math.max(size.x, size.z);
  const flat = size.y < widest * 0.25;
  const measure = flat ? widest : size.y;
  const scale = measure > 1e-4 ? target / measure : 1;

  scene.scale.setScalar(scale);
  scene.position.set(
    -((box.min.x + box.max.x) / 2) * scale,
    -box.min.y * scale,
    -((box.min.z + box.max.z) / 2) * scale
  );

  if (spec.yaw) wrapper.rotation.y = spec.yaw;
  wrapper.add(scene);
  return wrapper;
}
