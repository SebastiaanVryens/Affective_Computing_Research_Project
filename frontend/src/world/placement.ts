/**
 * Where a memory orb goes in the world.
 *
 * This lives on its own, with no three.js dependency, because two very
 * different layers need the identical answer: `session.ts` computes and stores
 * a position when an entry is saved, and `orbs.ts` uses it when drawing. If
 * those two ever disagreed, existing memories would jump to new positions the
 * next time the layout was touched — so there is exactly one definition.
 *
 * A Fermat (golden-angle) spiral is used rather than random scatter for two
 * reasons: spacing stays even at any count, and position is a pure function of
 * the entry's index, so nothing extra has to be stored or recomputed.
 */

/** ~137.5°, the angle that makes successive points maximally spread out. */
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/** Orbs never come closer than this, or they crowd the camera. */
const INNER_RADIUS = 9;

/** sqrt() growth keeps areal density constant as the galaxy fills in. */
const SPIRAL_SPACING = 1.55;

/** Vertical spread, giving the disc some thickness. */
const HEIGHT_SPREAD = 5.5;

/** How much the outer edge droops — reads as a galactic disc rather than a ring. */
const DISC_SAG = 0.08;

export interface Placement {
  x: number;
  y: number;
  z: number;
}

/**
 * Deterministic position for the entry at `index`.
 *
 * The height uses a hash of the index rather than Math.random() so the same
 * index always lands in the same place, across reloads and across machines.
 */
export function spiralPlacement(index: number): Placement {
  const angle = index * GOLDEN_ANGLE;
  const radius = INNER_RADIUS + SPIRAL_SPACING * Math.sqrt(index);

  const hashed = Math.sin(index * 12.9898) * 43758.5453;
  const unit = hashed - Math.floor(hashed); // fract() -> [0, 1)

  return {
    x: Math.cos(angle) * radius,
    y: (unit - 0.5) * HEIGHT_SPREAD - radius * DISC_SAG,
    z: Math.sin(angle) * radius,
  };
}
