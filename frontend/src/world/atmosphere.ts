/**
 * The part of the world that reacts instantly: sky and light.
 *
 * There used to be a field of drifting motes in here as well. They are gone —
 * they read as floating debris rather than as atmosphere, and the sense of
 * motion they were carrying during the descent is now carried by the memory
 * threads in threads.ts, which are real objects connecting two real places
 * rather than particles whose only job was to have something go past.
 *
 * Everything here is driven straight from the live mood bus each frame, so a
 * change of expression shows up in roughly one frame plus the smoothing
 * constant. The persistent memory orbs are the slow layer; this is the fast one.
 *
 * Two colours are in play at all times and they mean different things:
 *   - `mood`     : how you feel *right now*, spread across the sky as a field.
 *   - `lifetime` : who you've been across every entry, grounding the lower sky.
 * Today's feeling sitting on top of everything that came before is the visual
 * thesis of the whole app.
 *
 * The sky deliberately changes *slowly* — several seconds to settle — while the
 * light tracks faster. A mood shift should be something you notice having
 * happened, not something you watch happen.
 */

import * as THREE from 'three';
import {
  EMOTIONS,
  GRADIENT_ORDER,
  PALETTE,
  type EmotionVector,
  focus,
  hexToRgb,
  mixedColor,
} from '../emotions';

/**
 * How hard to concentrate the lifetime layer.
 *
 * Gentler than the live sky's focusing. A diary spanning months genuinely does
 * contain several emotions, and flattening that to one or two would misreport
 * it — whereas a single moment usually is one or two things. A wider nucleus
 * keeps the record honest while still avoiding a seven-colour smear.
 */
const LIFETIME_SHARPENING = 1.6;
const LIFETIME_NUCLEUS = 0.9;

/**
 * Relative luminance every sky band is pulled toward, and how hard.
 *
 * The palette is picked for identity, not for even brightness: joy's yellow
 * carries about three times the luminance of sadness's blue. Painted onto the
 * sky as-is, whichever warm emotion is present becomes a bright soft-edged blob
 * floating in a darker field — which reads as a *sun*. A light source, in a
 * world that is supposed to have none, and the only thing on screen out-shining
 * the orbs.
 *
 * Equalising luminance keeps every emotion's hue, which is the entire point of
 * the banded sky, and takes away only its ability to out-shine its neighbours.
 * Hue stays information; brightness stops being accidental information.
 *
 * Not flattened all the way — a little residual variation keeps the sky from
 * looking like flat vinyl — and deliberately dim, which is what keeps the sky
 * clear of the bloom threshold in scene.ts without a separate exposure knob.
 */
const SKY_BAND_LUMA = 0.22;
const SKY_LUMA_FLATTEN = 0.85;

/**
 * A palette colour rescaled to sit at roughly SKY_BAND_LUMA.
 *
 * THREE.Color has already converted the hex out of sRGB, so these components
 * are linear and a plain Rec.709 luminance is the right measure.
 */
function skyBandColor(hex: string): THREE.Color {
  const color = new THREE.Color(hex);
  const luma = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
  if (luma < 1e-4) return color;

  const scale = 1 + SKY_LUMA_FLATTEN * (SKY_BAND_LUMA / luma - 1);
  return color.multiplyScalar(scale);
}

const SKY_VERTEX = /* glsl */ `
  varying vec3 vWorldPosition;
  void main() {
    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPosition.xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const SKY_FRAGMENT = /* glsl */ `
  uniform vec3 uMoodColor;
  uniform vec3 uLifetimeColor;
  uniform float uTime;
  uniform float uArousal;
  uniform float uClarity;

  // One fixed slot per emotion, in canonical order. Colours never change; only
  // the weights do, which is what stops regions of sky from swapping identity
  // whenever the ranking shifts.
  uniform vec3 uBandColors[7];
  uniform float uBandWeights[7];
  // The same seven slots, weighted by every entry in the diary rather than by
  // the current moment. This is what makes an idle world look like *your*
  // world instead of a blank one.
  uniform float uLifetimeWeights[7];

  varying vec3 vWorldPosition;

  /**
   * Maps a position along the emotion gradient to a colour.
   *
   * Each emotion claims a slice of the 0..1 range proportional to its weight,
   * and every sample takes a soft weighted average of the slices near it. Two
   * emotions at 50/50 therefore show as two distinct colours across the sky
   * rather than one averaged one — mixing joy's yellow with sadness's blue
   * numerically gives grey-green mud, which is exactly the information loss the
   * blend work exists to prevent. The sky should show you *both* feelings.
   *
   * Where each t lands on screen is decided by flowField() below, so these
   * slices become organic regions rather than horizontal stripes.
   *
   * Kernel width is tied to each slice's own size, so a dominant emotion spreads
   * broadly and a faint one stays a thin seam instead of smearing everywhere.
   */
  vec3 gradientAt(float t, int layer) {
    vec3 accumulated = vec3(0.0);
    float totalInfluence = 0.0;
    float cursor = 0.0;

    for (int i = 0; i < 7; i++) {
      // layer 0 = how you feel now, layer 1 = every entry you have ever saved.
      float w = (layer == 0) ? uBandWeights[i] : uLifetimeWeights[i];
      if (w < 0.012) { cursor += w; continue; }   // too faint to draw

      float center = cursor + w * 0.5;
      // Floor the width so a narrow band still blends softly at its edges
      // instead of banding hard against its neighbour.
      float halfWidth = max(w * 0.5, 0.055);
      float d = (t - center) / halfWidth;
      float influence = exp(-d * d) * w;

      accumulated += uBandColors[i] * influence;
      totalInfluence += influence;
      cursor += w;
    }

    // Away from every band (possible when weights are tiny) fall back to the
    // averaged tint rather than to black.
    if (totalInfluence < 0.0001) return uMoodColor;
    return accumulated / totalInfluence;
  }

  // Cheap value noise. Good enough for slow, soft cloud banding; a proper
  // simplex implementation would cost more than the look is worth here.
  float hash(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
  }

  float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
      mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x),
          mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
      mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
          mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y),
      f.z);
  }

  /**
   * The shape of the sky: a soft, essentially static field.
   *
   * SKY_SCALE is the number that decides whether you are looking at a mixture
   * or at a shape. The field is what maps screen position to a point on the
   * emotion gradient, so how many times it cycles across the visible sky is
   * exactly how many times each emotion gets to appear. At the original 1.35 it
   * completed barely one cycle inside a 52° field of view, which handed every
   * emotion one enormous contiguous region — and a single large region of one
   * hue, soft-edged against its neighbour, stops reading as sky and starts
   * reading as an *object* in the sky. No amount of recolouring fixes that;
   * it is the frequency that is wrong, not the palette.
   *
   * Cycling several times over instead interleaves all seven bands across the
   * whole sky, which is the point: a mood is a mixture, so the sky should be a
   * mixture everywhere rather than a map of territories.
   *
   * Domain warping — sampling noise at coordinates displaced by other noise —
   * is what turns round blobs into the long curling forms that read as flow
   * rather than as clouds. The warp stays at a lower frequency than the field
   * it displaces, which is what keeps the forms stretched and organic instead
   * of merely granular.
   *
   * The time term is deliberately almost nothing — measured at roughly 3% of the
   * field's range per 30 seconds, which is invisible moment to moment but keeps
   * the sky from looking like a frozen texture over several minutes.
   *
   * This matters because the field decides *where* each emotion sits on screen.
   * If it slides around, a change of feeling arrives as motion, and motion
   * competes with colour for attention. Holding the shape still means a shift in
   * mood reads purely as the sky changing hue in place.
   */
  const float SKY_SCALE = 4.6;

  float flowField(vec3 dir, float time) {
    vec3 p = dir * SKY_SCALE + vec3(0.0, time * 0.001, 0.0);

    vec3 warp = vec3(
      noise(p * 0.5),
      noise(p * 0.5 + vec3(4.7, 2.3, 9.1)),
      noise(p * 0.5 + vec3(8.3, 5.9, 1.7))
    );

    // Three octaves rather than two. At this frequency the extra detail is no
    // longer smoothed away by the colour gradient — it is what stops the
    // mixture looking like regular blobs on a grid.
    float field = noise(p + (warp - 0.5) * 2.4);
    field = field * 0.54
          + noise(p * 2.1 + (warp - 0.5) * 1.1) * 0.30
          + noise(p * 4.3 + (warp - 0.5) * 0.6) * 0.16;

    // Slight upward bias so the composition still has a sky-like top and
    // bottom. Weaker than it was: with the field cycling several times over,
    // a strong bias would pin the same band along the whole horizon.
    field += (dir.y * 0.5 + 0.5) * 0.12 - 0.06;

    // Expand the usable middle of the range: raw value noise clusters hard
    // around 0.5, which would leave most of the screen showing one emotion.
    // Averaging three octaves instead of two tightens that cluster further, so
    // the window has to narrow to match or the outer bands never get drawn.
    return clamp(smoothstep(0.31, 0.69, field), 0.0, 1.0);
  }

  void main() {
    vec3 dir = normalize(vWorldPosition);

    // One organic field across the entire sky decides where each emotion sits.
    // Because it barely moves, a change of mood arrives as colour morphing in
    // place rather than as shapes sliding past — the sky *becomes* different
    // instead of drifting somewhere different.
    float t = flowField(dir, uTime);

    // Two gradients through the same field: today above, everything you have
    // ever recorded below. Both are real gradients — the lower layer used to be
    // a single averaged colour, which turned a whole diary of distinct feelings
    // into one flat tint, exactly the mixing-is-loss problem the rest of the
    // app is built to avoid.
    vec3 todayColour = gradientAt(t, 0);
    vec3 lifetimeColour = gradientAt(t, 1) * 0.42;  // darkened: it is the ground

    // Soft and wide, so it reads as depth rather than as a horizon line.
    float grounding = smoothstep(-0.75, 0.35, dir.y);
    vec3 color = mix(lifetimeColour, todayColour, 0.25 + 0.75 * grounding);

    // An uncertain reading desaturates toward the mean rather than picking a
    // colour it can't justify. Ambiguity should look ambiguous.
    //
    // This is also the backstop for the one case the focusing can't reduce: a
    // perfectly flat distribution has no ranking to cut, so every emotion keeps
    // equal area and the sky would go rainbow. But flat is precisely where
    // clarity is lowest, so pulling hard toward grey here catches exactly that
    // case — the sky is most colourful when it is most sure.
    //
    // Clarity is 1 - normalised entropy over seven emotions, and that does not
    // spread over [0,1] in practice: a realistic mixed reading scores around
    // 0.15, and only a near-pure single emotion passes 0.4. Reading it as a
    // straight 0..1 fraction therefore treated the *normal* case as near-total
    // ambiguity and left the sky grey almost all the time. Remapping the range
    // it actually occupies is what gives an ordinary day a coloured sky while
    // still collapsing a genuinely flat one to grey.
    float vividness = 0.25 + 0.40 * smoothstep(0.0, 0.3, uClarity);
    float luma = dot(color, vec3(0.299, 0.587, 0.114));
    // The grey being mixed toward is held close to the colour's own luminance.
    // Darkening it further would couple two things that should stay separate:
    // every step down in saturation would also be a step down in brightness,
    // which is how a muted sky turned into a murky one last time.
    color = mix(vec3(luma) * 0.94, color, vividness);

    // A very faint second field adds depth without introducing movement. Its
    // strength rides on vocal energy, so a raised voice thickens the air
    // slightly — the only part of the sky that responds in real time to sound.
    float depth = noise(dir * 3.1 + vec3(0.0, uTime * 0.002, 0.0));
    color += (depth - 0.5) * 0.05 * (0.6 + uArousal);

    // Vignette toward the horizon keeps the orbs readable against the sky.
    color *= 0.55 + 0.45 * smoothstep(-0.4, 0.8, dir.y);

    gl_FragColor = vec4(color, 1.0);
  }
`;

export class Atmosphere {
  readonly group = new THREE.Group();

  /** Sky and lights. Rides with the viewer between floors. */
  private frame = new THREE.Group();

  private sky: THREE.Mesh;
  private skyUniforms: Record<string, THREE.IUniform>;

  private keyLight: THREE.DirectionalLight;
  private ambient: THREE.AmbientLight;
  private moodLight: THREE.PointLight;

  private moodColor = new THREE.Color('#8d93a8');
  // Reused each frame; `new THREE.Color()` in the render loop is 60 throwaway
  // allocations a second.
  private moodTarget = new THREE.Color('#8d93a8');
  private lifetimeColor = new THREE.Color('#2a2f45');

  /**
   * Live band weights, smoothed toward the mood bus each frame.
   *
   * The bus is already EMA-smoothed, but it only updates when a reading arrives
   * (8 Hz for face, seconds apart for text). Interpolating again here at frame
   * rate is what turns those steps into a continuous drift — an emotion's band
   * widens and narrows rather than popping between sizes.
   */
  private bandWeights = new Float32Array(EMOTIONS.length).fill(1 / EMOTIONS.length);

  /**
   * The same slots weighted by the whole diary. Set once per save rather than
   * per frame — it only changes when an entry is written.
   */
  private lifetimeWeights = new Float32Array(EMOTIONS.length).fill(
    1 / EMOTIONS.length
  );

  constructor() {
    this.group.add(this.frame);

    this.skyUniforms = {
      uMoodColor: { value: new THREE.Color('#8d93a8') },
      uLifetimeColor: { value: new THREE.Color('#1c2033') },
      uTime: { value: 0 },
      uArousal: { value: 0 },
      uClarity: { value: 0.3 },
      // Constant for the life of the scene. GRADIENT_ORDER, not the canonical
      // MELD order: these slots must line up with the weights the mood bus
      // emits, and that ordering is what puts related emotions side by side.
      uBandColors: {
        value: GRADIENT_ORDER.map((e) => skyBandColor(PALETTE[e].base)),
      },
      uBandWeights: { value: this.bandWeights },
      uLifetimeWeights: { value: this.lifetimeWeights },
    };

    this.sky = new THREE.Mesh(
      new THREE.SphereGeometry(220, 48, 32),
      new THREE.ShaderMaterial({
        uniforms: this.skyUniforms,
        vertexShader: SKY_VERTEX,
        fragmentShader: SKY_FRAGMENT,
        side: THREE.BackSide,
        depthWrite: false,
      })
    );
    this.frame.add(this.sky);

    this.ambient = new THREE.AmbientLight(0xffffff, 0.55);
    this.keyLight = new THREE.DirectionalLight(0xffffff, 0.7);
    this.keyLight.position.set(6, 12, 8);
    // The target has to be a child of this group, not the scene's default at the
    // world origin. The group travels with the camera between floors (see
    // setFocusHeight), and a target left behind at y = 0 would swing the key
    // light through ninety degrees on the way down — the island would be lit
    // from above and the orbs from directly below.
    this.keyLight.target.position.set(0, 0, 0);
    // Sits just behind the camera and takes the live mood colour, so the orbs
    // nearest the viewer pick up "now" while the rest stay in lifetime colour.
    this.moodLight = new THREE.PointLight(0xffffff, 40, 90, 2);
    this.moodLight.position.set(0, 3, 16);

    this.frame.add(this.ambient, this.keyLight, this.keyLight.target, this.moodLight);
  }

  /**
   * Move the sky and the lights to whichever floor is being looked at.
   *
   * The sky is a 220-unit sphere and the lights are positioned relative to a
   * viewer at roughly y = 0. Both assumptions are what they always were — this
   * just keeps them true when the camera drops seventy-odd units to the memory
   * orbs, rather than leaving the viewer down near the floor of a sphere whose
   * gradient was composed for its middle.
   *
   * It also means the descent has no visible sky motion of its own, which is
   * correct: the sense of falling should come from the island receding and the
   * shaft rings passing, not from the horizon sliding, which would read as the
   * *world* moving rather than you.
   */
  setFocusHeight(y: number): void {
    this.frame.position.y = y;
  }

  /**
   * The colour the sky is at eye level, for the scene fog and the sea.
   *
   * There is no cheap way to ask the sky shader what it painted at the horizon —
   * it is a flow field evaluated per pixel — so this reconstructs the same
   * recipe the shader uses down there: mostly the lifetime layer, a little of
   * today, and then the vignette that darkens the sky toward its lower half.
   *
   * It matters that this stays in step with SKY_FRAGMENT. The whole point of one
   * haze colour is that the far water, the far hills and the sky behind both of
   * them end up indistinguishable; get this wrong and the horizon becomes a
   * visible line, which is the exact failure the fog exists to prevent.
   */
  horizonColor(out: THREE.Color): THREE.Color {
    out.copy(this.lifetimeColor).lerp(this.moodColor, 0.45);
    // `grounding` is ~0.3 at eye level in the shader, and the lifetime layer is
    // darkened to 0.42 there; this is that pair of numbers, folded together.
    return out.multiplyScalar(0.62).addScalar(0.045);
  }

  /** Call once when the lifetime totals change (i.e. after an entry is saved). */
  setLifetimeMood(totals: EmotionVector): void {
    // Focused the same way the live mood is, so a long diary doesn't smear into
    // all seven colours at once as entries accumulate.
    const focused = focus(totals, LIFETIME_SHARPENING, LIFETIME_NUCLEUS);
    for (let i = 0; i < GRADIENT_ORDER.length; i++) {
      this.lifetimeWeights[i] = focused[GRADIENT_ORDER[i]];
    }
    this.skyUniforms.uLifetimeWeights.value = this.lifetimeWeights;

    // Still kept as a single colour for the ambient fallback inside the shader
    // and for anything that needs one hex rather than a distribution.
    const [r, g, b] = hexToRgb(mixedColor(totals));
    this.lifetimeColor.setRGB(r * 0.34, g * 0.34, b * 0.38);
  }

  /** Called every frame with the current live mood. */
  update(
    delta: number,
    elapsed: number,
    mood: { color: string; arousal: number; clarity: number; bands?: Float32Array }
  ): void {
    if (mood.bands) {
      // The sky's whole job is to change slowly. These weights decide how much
      // of the screen each emotion claims, so chasing the mood bus closely makes
      // the colours churn while you talk. A ~4s time constant means a shift in
      // feeling arrives as something you notice having happened rather than as
      // something you watch happen.
      const k = Math.min(1, delta * 0.28);
      for (let i = 0; i < this.bandWeights.length; i++) {
        this.bandWeights[i] += (mood.bands[i] - this.bandWeights[i]) * k;
      }
      this.skyUniforms.uBandWeights.value = this.bandWeights;
    }

    this.skyUniforms.uTime.value = elapsed;

    // Lerped rather than assigned: the mood bus is already smoothed, but this
    // second stage keeps colour motion continuous even if a reading lands late.
    // Matched to the band weights above so the ambient light and the sky drift
    // together instead of the room changing colour before the sky does.
    this.moodTarget.set(mood.color);
    this.moodColor.lerp(this.moodTarget, Math.min(1, delta * 0.5));

    (this.skyUniforms.uMoodColor.value as THREE.Color).copy(this.moodColor);
    (this.skyUniforms.uLifetimeColor.value as THREE.Color).copy(this.lifetimeColor);
    this.skyUniforms.uArousal.value +=
      (mood.arousal - this.skyUniforms.uArousal.value) * Math.min(1, delta * 6);
    this.skyUniforms.uClarity.value +=
      (mood.clarity - this.skyUniforms.uClarity.value) * Math.min(1, delta * 0.6);

    this.moodLight.color.copy(this.moodColor);
    this.moodLight.intensity = 30 + 55 * mood.arousal;
  }

  dispose(): void {
    this.sky.geometry.dispose();
    (this.sky.material as THREE.Material).dispose();
  }
}
