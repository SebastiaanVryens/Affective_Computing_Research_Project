/**
 * The sea.
 *
 * Everything here exists because the previous version — a flat translucent disc
 * with a specular highlight — read as a sheet of dark plastic the island had
 * been placed on. Water is recognised almost entirely by things a flat disc
 * cannot do: it moves, it goes pale where it shallows, it turns to sky at
 * grazing angles, and it breaks white where it meets land. So it gets a real
 * shader.
 *
 * Four pieces, in the order they matter:
 *
 *   swell   — four directional waves summed in the vertex shader, with the
 *             surface normal taken from their analytic derivative rather than
 *             from the mesh. That is what makes the light move across the water
 *             instead of sitting on it.
 *   depth   — baked per-vertex at build time by asking terrain.ts how deep the
 *             sea floor is at that point. Drives colour, opacity and surf, and
 *             it is the reason the shore reads as a shore rather than as a line
 *             where two objects happen to overlap.
 *   fresnel — water is nearly a mirror at a glancing angle and nearly clear
 *             looking straight down. Without this the sea is the same colour to
 *             the horizon, which is the single biggest tell.
 *   surf    — foam where the water is shallow, with the line advancing and
 *             retreating as the swell passes over it.
 *
 * The horizon is not drawn. It is the scene fog eating the far water, which is
 * also what the far hills fade into — one haze for the whole scene, so there is
 * no seam anywhere to find.
 */

import * as THREE from 'three';
import { NOISE_2D } from './glsl';
import { WELL_RADIUS, elevation, type TerrainShape } from './terrain';

/**
 * How far an open sea runs.
 *
 * Paired with the fog in scene.ts, not chosen on its own: the fog must reach
 * total before the water's outer edge does, or the sea ends in a visible rim.
 * Inland biomes pass their own, much smaller radius — a pond's edge is meant to
 * be seen.
 */
export const SEA_RADIUS = 200;

/** Rings, and how much of them is spent near the island. */
const RINGS = 96;
const SPOKES = 128;
/** Radius the dense inner band covers, and the share of rings it gets. */
const DETAIL_RADIUS = 28;
const DETAIL_SHARE = 0.45;

const VERTEX = /* glsl */ `
  #include <common>
  #include <fog_pars_vertex>

  uniform float uTime;
  uniform float uSwell;
  // 0 for an enclosed pond, 1 for open sea. Set from the water's radius.
  uniform float uOpenness;

  attribute float aDepth;

  varying float vDepth;
  varying float vWave;
  varying vec3 vWorld;
  varying vec3 vWaveNormal;

  /**
   * One directional wave, accumulated into height and horizontal gradient.
   *
   * Taking the gradient here rather than computing normals from the displaced
   * mesh is the whole trick: the mesh is far too coarse at distance to carry a
   * two-metre ripple, but the derivative is exact everywhere and costs a cosine.
   */
  void addWave(
    vec2 p, vec2 dir, float amp, float len, float speed, float t,
    inout float h, inout vec2 grad
  ) {
    float k = 6.2831853 / len;
    float phase = dot(dir, p) * k + t * speed;
    h += sin(phase) * amp;
    grad += dir * (cos(phase) * amp * k);
  }

  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vec2 p = world.xz;

    // Waves flatten as the water shallows, so nothing pokes up through a beach.
    float shore = smoothstep(0.0, 1.8, aDepth);
    // And the short ones fade out with distance, where they are below a pixel
    // and would only shimmer.
    float near = 1.0 - smoothstep(55.0, 150.0, length(p));

    float h = 0.0;
    vec2 grad = vec2(0.0);
    // A tarn does not have a swell. Fetch — the distance wind has to work over —
    // is what builds waves, so the size of the water body is the right thing to
    // scale this by, and it happens to be the number the biome already knows.
    float amp = uSwell * shore * (0.16 + 0.84 * uOpenness);

    addWave(p, normalize(vec2( 1.00,  0.16)), 0.22 * amp,        13.0, 1.05, uTime, h, grad);
    addWave(p, normalize(vec2(-0.50,  0.87)), 0.13 * amp,         8.0, 1.40, uTime, h, grad);
    addWave(p, normalize(vec2( 0.32, -0.95)), 0.07 * amp * near,  4.4, 1.95, uTime, h, grad);
    addWave(p, normalize(vec2(-0.86, -0.50)), 0.04 * amp * near,  2.7, 2.50, uTime, h, grad);

    vec3 displaced = position;
    displaced.y += h;

    vWorld = vec3(world.x, world.y + h, world.z);
    vDepth = aDepth;
    // Normalised against this point's own amplitude, so "at the top of a wave"
    // means the same thing in the shallows as it does out at sea.
    vWave = h / max(amp, 1e-3);
    vWaveNormal = normalize(vec3(-grad.x, 1.0, -grad.y));

    vec4 mvPosition = modelViewMatrix * vec4(displaced, 1.0);
    gl_Position = projectionMatrix * mvPosition;

    #include <fog_vertex>
  }
`;

const FRAGMENT = /* glsl */ `
  #include <common>
  #include <fog_pars_fragment>

  uniform float uTime;
  uniform vec3 uDeep;
  uniform vec3 uShallow;
  uniform vec3 uSky;
  uniform vec3 uSunDir;
  uniform float uOpenness;

  varying float vDepth;
  varying float vWave;
  varying vec3 vWorld;
  varying vec3 vWaveNormal;

  ${NOISE_2D}

  void main() {
    // Water under the land does not exist.
    //
    // The sea is one disc spanning the whole world, so a good deal of it lies
    // *beneath* the island. That should never matter — the ground is in front of
    // it — except that the ground is a single-sided ring, so the near inside
    // wall of the well is backface-culled and writes no depth. Looking down the
    // well therefore looked straight through the terrain at buried sea, at zero
    // depth, which the surf below renders as solid foam: a white disc plugging
    // the hole, hiding the shaft and the orbs underneath it.
    //
    // Discarding is the honest fix rather than the cheap one. Making the ground
    // double-sided would hide the symptom and leave a sheet of water sitting
    // inside the hill.
    if (vDepth < 0.0) discard;

    vec3 view = normalize(cameraPosition - vWorld);

    // Ripple finer than the mesh could ever carry, added as normal only. Faded
    // out quickly with distance: past twenty-odd units it is sub-pixel, and
    // sub-pixel normal detail does not read as ripple — it reads as television
    // static, because every pixel samples a different part of it.
    float detail = 1.0 - smoothstep(16.0, 58.0, length(vWorld.xz));
    vec2 rp = vWorld.xz * 1.5 + vec2(uTime * 0.33, uTime * -0.21);
    float e = 0.25;
    float n0 = fbm2(rp);
    float nx = fbm2(rp + vec2(e, 0.0));
    float nz = fbm2(rp + vec2(0.0, e));
    vec3 normal = normalize(
      vWaveNormal + vec3(-(nx - n0), 0.0, -(nz - n0)) * (0.55 * detail / e)
    );

    // Schlick. Deliberately well short of the physical answer — real water is
    // near-total mirror at a grazing angle, but there is no environment map
    // here, so "mirror" means "the flat haze colour" and a physically honest
    // curve turns the whole middle distance into one grey sheet.
    float fresnel = 0.02 + 0.5 * pow(1.0 - clamp(dot(normal, view), 0.0, 1.0), 5.0);

    vec3 body = mix(uShallow, uDeep, smoothstep(0.2, 7.0, vDepth));
    vec3 color = mix(body, uSky, fresnel);

    // Every value from here on is kept under the bloom threshold in scene.ts on
    // purpose. The sea covers most of the frame, and anything on it that clears
    // that threshold does not glint — it blooms, and then the entire surface is
    // a sheet of white.
    vec3 halfway = normalize(uSunDir + view);
    color += vec3(1.0, 0.98, 0.92) * pow(max(dot(normal, halfway), 0.0), 220.0) * 0.4;

    // Surf. The waterline is the depth *under the passing swell*, so the foam
    // runs up the sand and drains back rather than sitting on the shore as a
    // painted ring.
    float underSwell = vDepth - vWave * 0.4;
    float band = 1.0 - smoothstep(0.0, 0.55, underSwell);
    float speckle = fbm2(vWorld.xz * 2.1 + vec2(uTime * 0.24, uTime * 0.17));
    float shoreFoam = band * smoothstep(0.46, 0.86, speckle + band * 0.4);
    // Whitecaps, but only where it is deep enough for the sea to be doing it on
    // its own rather than because of the beach.
    float crest = smoothstep(0.30, 0.46, vWave) * smoothstep(1.4, 6.0, vDepth) * 0.22;
    float foam = clamp(shoreFoam + crest, 0.0, 1.0);
    color = mix(color, vec3(0.70, 0.73, 0.78), foam);

    // Clear in the shallows so the sand reads through it, opaque once there is
    // any depth. Foam is always solid — it is on the surface, not in it.
    float alpha = max(mix(0.62, 1.0, smoothstep(0.0, 2.4, vDepth)), foam);

    gl_FragColor = vec4(color, alpha);

    #include <fog_fragment>
  }
`;

export class Water {
  readonly mesh: THREE.Mesh;

  private material: THREE.ShaderMaterial;
  private geometry: THREE.BufferGeometry | null = null;
  private tint = new THREE.Color('#2b6a96');
  private target = new THREE.Color('#2b6a96');

  constructor() {
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      uniforms: THREE.UniformsUtils.merge([
        THREE.UniformsLib.fog,
        {
          uTime: { value: 0 },
          uSwell: { value: 1 },
          uOpenness: { value: 1 },
          uDeep: { value: new THREE.Color('#123b5e') },
          uShallow: { value: new THREE.Color('#4e9fc2') },
          uSky: { value: new THREE.Color('#7e88a6') },
          // Matches the key light in atmosphere.ts. Constant, because that light
          // is: if it ever moves, this has to move with it or the sea will have
          // its highlight in a different place from everything else.
          uSunDir: { value: new THREE.Vector3(6, 12, 8).normalize() },
        },
      ]),
      transparent: true,
      // Written, not skipped. The sea is one mesh whose triangles are indexed
      // from the middle outwards, so without depth writes the far rings paint
      // over the near ones and the swell folds inside out.
      depthWrite: true,
      side: THREE.DoubleSide,
      fog: true,
    });

    this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), this.material);
    // Drawn after the ground, before the glowing things. Left to itself, three
    // sorts transparent objects by their centre, and the sea's centre is the
    // origin — which puts a two-hundred-unit disc in the same sort position as
    // a beacon standing a few units from it.
    this.mesh.renderOrder = 1;
  }

  /**
   * Rebuild for a given landform.
   *
   * The only thing that actually changes is the baked depth attribute, but it
   * changes for every vertex, so there is nothing to be saved by keeping the
   * old geometry around.
   */
  rebuild(shape: TerrainShape, radius = SEA_RADIUS): void {
    this.disposeGeometry();
    const waterLevel = shape.waterLevel ?? 0;

    // How much like an ocean this water is, from how far it reaches. Drives the
    // swell and the surf, both of which are wrong on a pond.
    this.material.uniforms.uOpenness.value = Math.max(
      0,
      Math.min(1, (radius - 26) / 110)
    );

    const vertexCount = RINGS * (SPOKES + 1);
    const positions = new Float32Array(vertexCount * 3);
    const depths = new Float32Array(vertexCount);

    for (let ring = 0; ring < RINGS; ring++) {
      const ringRadius = radiusAt(ring, radius);
      for (let spoke = 0; spoke <= SPOKES; spoke++) {
        const angle = (spoke / SPOKES) * Math.PI * 2;
        const index = ring * (SPOKES + 1) + spoke;
        const x = Math.cos(angle) * ringRadius;
        const z = Math.sin(angle) * ringRadius;

        positions[index * 3] = x;
        positions[index * 3 + 1] = waterLevel;
        positions[index * 3 + 2] = z;
        depths[index] = waterLevel - elevation(x, z, shape);
      }
    }

    const indices: number[] = [];
    for (let ring = 0; ring < RINGS - 1; ring++) {
      for (let spoke = 0; spoke < SPOKES; spoke++) {
        const a = ring * (SPOKES + 1) + spoke;
        const b = a + 1;
        const c = a + (SPOKES + 1);
        const d = c + 1;
        indices.push(a, c, b, b, c, d);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('aDepth', new THREE.BufferAttribute(depths, 1));
    geometry.setIndex(indices);
    // Set by hand: the vertices are flat here and displaced in the shader, so a
    // computed bounding sphere would be a disc and the mesh would be culled the
    // moment the camera looked along it.
    geometry.boundingSphere = new THREE.Sphere(
      new THREE.Vector3(0, waterLevel, 0),
      radius + 2
    );

    this.geometry = geometry;
    this.mesh.geometry = geometry;
  }

  /**
   * @param sky  The colour the sea reflects — the same one the fog uses, which
   *             is what makes the far water and the haze meet without a seam.
   */
  update(
    delta: number,
    elapsed: number,
    mood: { color: string; arousal: number },
    sky: THREE.Color
  ): void {
    this.material.uniforms.uTime.value = elapsed;
    // A raised voice puts a chop on the water. Small — this is the only thing
    // in the mind view that answers to the microphone in real time, and it
    // should read as weather rather than as a meter.
    this.material.uniforms.uSwell.value +=
      (1 + mood.arousal * 0.55 - this.material.uniforms.uSwell.value) *
      Math.min(1, delta * 1.5);

    this.target.set(mood.color);
    this.tint.lerp(this.target, Math.min(1, delta * 0.4));

    // The sea takes a wash of the day's colour, never more. Water that turned
    // yellow on a good day would stop being water.
    (this.material.uniforms.uDeep.value as THREE.Color)
      .set('#0e3454')
      .lerp(this.tint, 0.22);
    (this.material.uniforms.uShallow.value as THREE.Color)
      .set('#57a8c9')
      .lerp(this.tint, 0.3);
    (this.material.uniforms.uSky.value as THREE.Color).copy(sky);
  }

  private disposeGeometry(): void {
    this.geometry?.dispose();
    this.geometry = null;
  }

  dispose(): void {
    this.disposeGeometry();
    this.material.dispose();
  }
}

/**
 * Radius of one ring.
 *
 * Even spacing would put almost every vertex where nothing happens. The swell
 * only needs resolving near the island — that is where it is compared against a
 * shoreline and where the camera is — so the first 45% of the rings cover the
 * first 28 units and the rest stretch out quadratically to the horizon.
 */
function radiusAt(ring: number, outerRadius: number): number {
  const t = ring / (RINGS - 1);
  // A pond is smaller than the detail band itself, so the band has to shrink
  // with it or every ring lands outside the water.
  const detail = Math.min(DETAIL_RADIUS, outerRadius * 0.6);
  if (t <= DETAIL_SHARE) {
    return WELL_RADIUS + (t / DETAIL_SHARE) * (detail - WELL_RADIUS);
  }
  const out = (t - DETAIL_SHARE) / (1 - DETAIL_SHARE);
  return detail + out * out * (outerRadius - detail);
}
