/**
 * GLSL fragments shared by the world's custom materials.
 *
 * The sky in atmosphere.ts carries its own copy of this noise, inlined, and that
 * is deliberate — it is a self-contained shader with its own tuning, and pulling
 * it in here would couple the one shader nobody should have to think about to a
 * file that changes whenever the ground does. Everything written *since* shares
 * from here, because the ground, the sea and the far hills all need to look like
 * they were made by the same hand, and three independently-tuned noise functions
 * is exactly how a scene stops looking like one place.
 */

/**
 * 2D value noise plus fBm.
 *
 * The same cheap hash the sky uses. It has visible axis alignment if you look
 * for it, which is why every caller samples it at an angle or with a domain
 * offset rather than straight along world X/Z.
 */
export const NOISE_2D = /* glsl */ `
  float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
  }

  float noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
      mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
      mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x),
      u.y
    );
  }

  float fbm2(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    // Rotated each octave: without it the lattice of the value noise lines up
    // with itself and the result shows a grid at grazing angles, which on a
    // surface as large as the sea is the first thing you see.
    mat2 turn = mat2(0.86, 0.5, -0.5, 0.86);
    for (int i = 0; i < 4; i++) {
      value += amplitude * noise2(p);
      p = turn * p * 2.03;
      amplitude *= 0.5;
    }
    return value;
  }
`;
