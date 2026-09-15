/**
 * Draw order inside the transparent pass.
 *
 * Opaque geometry sorts itself out with the depth buffer and needs nothing from
 * this file. Transparent geometry cannot: it is drawn after everything opaque,
 * in an order three decides, and each surface blends with whatever is already in
 * the framebuffer. So for two transparent things the *later* one wins, whichever
 * is actually nearer the camera.
 *
 * Left alone, three sorts that pass back-to-front by each object's centre. That
 * is wrong here in a specific and unfixable way: the sea is a single mesh two
 * hundred units across whose centre is the origin, so it sorts as though it were
 * standing in the middle of the island — at the same depth as a beacon a few
 * units from it. Which of the two came out on top was then decided by the camera
 * angle.
 *
 * What went wrong, because it is the reason this file exists: every glowing
 * thing in this world is additive with `depthWrite: false`, which is correct —
 * light does not occlude. But it also means a beacon leaves no depth behind it,
 * so when the sea is drawn afterwards it passes the depth test against the
 * *terrain* behind the beacon and paints straight over it. Core memories
 * standing anywhere in front of the water were being erased by water that was
 * behind them, and a beacon half in front of a headland and half in front of the
 * sea was cut off exactly at the waterline.
 *
 * The order below is the one water.ts always meant: ground, then sea, then the
 * things that are made of light. Being explicit costs one number per mesh and
 * takes the answer away from the sort.
 */

export const LAYER = {
  /**
   * The sea. After the opaque ground, before anything glowing.
   *
   * Above zero rather than at it so that a transparent surface which has not
   * opted into this scheme still sorts below the water rather than above it —
   * the sea is scenery, and scenery should never be in front of the memories.
   */
  water: 1,

  /**
   * Everything made of light: beacons, memory threads, the live orb, the words
   * drifting off it.
   *
   * These are last because they are emissive. Nothing in this world is supposed
   * to be in front of a core memory — it is a light source, not an object, and a
   * light source that the sea can cover is not one.
   */
  glow: 2,
} as const;
