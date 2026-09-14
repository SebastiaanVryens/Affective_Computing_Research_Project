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
  study: { url: '/props/books.glb', height: 1.0, tint: 0.25 },
  city: { url: '/props/building.glb', height: 5.4, tint: 0.28 },
  home: { url: '/props/house.glb', height: 3.2, tint: 0.28 },
  music: { url: '/props/speaker.glb', height: 2.0, tint: 0.3 },

  // Fixed furniture, fetched by room.ts rather than by a motif. Namespaced so
  // it can never collide with a motif id.
  'room:bookcase': { url: '/props/bookcase.glb', height: 2.3, tint: 0.2 },
  'room:desk': { url: '/props/desk.glb', height: 1.5, tint: 0.2 },
  'room:sofa': { url: '/props/sofa.glb', height: 1.2, tint: 0.3 },
  'room:lamp': { url: '/props/lamp.glb', height: 2.5, tint: 0.2 },
  'room:plant': { url: '/props/plant.glb', height: 1.2, tint: 0.2 },
};

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
