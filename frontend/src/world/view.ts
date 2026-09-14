/**
 * What you can see out of the window.
 *
 * The room is chosen when somebody's diary is mostly indoor things — books,
 * work, home. But almost nobody's diary is *only* that, and the second thing
 * they keep talking about is exactly what should be outside the glass: the coast
 * if they talk about the sea, a skyline if they talk about the office, hills if
 * they climb. The room says where they spend their time; the view says what the
 * rest of it is about.
 *
 * Painted onto a canvas rather than built as geometry, which is the right trade
 * for something seen through a window from across a room: it is a backdrop, it
 * never moves, and nothing is ever close enough to it for the flatness to show.
 * A modelled skyline would be several hundred meshes to produce an image a
 * hundred lines of 2D drawing gives exactly.
 *
 * Deterministic from the seed, like every other generated thing in this world —
 * the view from your window is the same view tomorrow.
 */

import * as THREE from 'three';

export type ViewKind = 'city' | 'coast' | 'forest' | 'peaks' | 'meadow';

const WIDTH = 1024;
const HEIGHT = 512;
/** Where the horizon sits, as a fraction of the canvas height. */
const HORIZON = 0.62;

/**
 * Paint the view.
 *
 * @param sky    The scene's haze colour, so the window agrees with the sky the
 *               rest of the app is drawing rather than inventing its own weather.
 * @param accent The room's emotional colour, mixed in very lightly — enough that
 *               the outside belongs to the same world, not enough to look tinted.
 */
export function makeViewTexture(
  kind: ViewKind,
  sky: THREE.Color,
  accent: THREE.Color,
  seed: number
): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  const ctx = canvas.getContext('2d')!;

  const rng = makeRng(seed);
  const horizonY = HEIGHT * HORIZON;

  // Sky: brighter at the horizon than overhead, which is true of every sky and
  // is most of what makes a flat gradient read as distance.
  const high = mix(sky, accent, 0.12).multiplyScalar(1.15);
  const low = mix(sky, new THREE.Color('#ffffff'), 0.4);
  const gradient = ctx.createLinearGradient(0, 0, 0, horizonY);
  gradient.addColorStop(0, css(high));
  gradient.addColorStop(1, css(low));
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, WIDTH, horizonY);

  // Ground below the horizon, before anything is drawn on it.
  const ground = mix(sky, new THREE.Color('#2b3038'), 0.55);
  ctx.fillStyle = css(ground);
  ctx.fillRect(0, horizonY, WIDTH, HEIGHT - horizonY);

  switch (kind) {
    case 'city':
      paintCity(ctx, rng, horizonY, low, accent);
      break;
    case 'coast':
      paintCoast(ctx, rng, horizonY, low, accent);
      break;
    case 'forest':
      paintForest(ctx, rng, horizonY, low);
      break;
    case 'peaks':
      paintPeaks(ctx, rng, horizonY, low);
      break;
    case 'meadow':
      paintMeadow(ctx, rng, horizonY, low, accent);
      break;
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

/**
 * Two depths of skyline, the far one hazed toward the sky.
 *
 * Atmospheric perspective done with one lerp is the whole reason this reads as
 * distance rather than as two rows of rectangles.
 */
function paintCity(
  ctx: CanvasRenderingContext2D,
  rng: Rng,
  horizonY: number,
  haze: THREE.Color,
  accent: THREE.Color
): void {
  const layers: Array<{ fade: number; maxHeight: number; width: [number, number] }> = [
    { fade: 0.68, maxHeight: 0.42, width: [26, 60] },
    { fade: 0.34, maxHeight: 0.62, width: [34, 86] },
  ];

  for (const layer of layers) {
    const body = mix(new THREE.Color('#2c3440'), haze, layer.fade);
    const lit = mix(new THREE.Color('#ffe6b4'), accent, 0.3);

    let x = -40;
    while (x < WIDTH + 40) {
      const w = layer.width[0] + rng() * (layer.width[1] - layer.width[0]);
      const h = horizonY * (0.12 + rng() * layer.maxHeight);
      const top = horizonY - h;

      ctx.fillStyle = css(body);
      ctx.fillRect(x, top, w, h + 8);

      // Windows, only on the near layer — on the far one they would be a single
      // pixel each and read as noise.
      if (layer.fade < 0.5) {
        ctx.fillStyle = css(lit);
        for (let wy = top + 10; wy < horizonY - 8; wy += 13) {
          for (let wx = x + 5; wx < x + w - 5; wx += 11) {
            if (rng() < 0.42) ctx.globalAlpha = 0.25 + rng() * 0.5;
            else continue;
            ctx.fillRect(wx, wy, 4, 6);
          }
        }
        ctx.globalAlpha = 1;
      }
      x += w + 3 + rng() * 10;
    }
  }
}

/** A sea to the horizon, a headland, and a strip of sand. */
function paintCoast(
  ctx: CanvasRenderingContext2D,
  rng: Rng,
  horizonY: number,
  haze: THREE.Color,
  accent: THREE.Color
): void {
  const far = mix(new THREE.Color('#5d6b78'), haze, 0.55);
  ctx.fillStyle = css(far);
  ctx.beginPath();
  ctx.moveTo(0, horizonY);
  ctx.lineTo(0, horizonY - 26);
  for (let x = 0; x <= WIDTH; x += 64) {
    ctx.lineTo(x, horizonY - 14 - rng() * 30);
  }
  ctx.lineTo(WIDTH, horizonY);
  ctx.closePath();
  ctx.fill();

  const sea = ctx.createLinearGradient(0, horizonY, 0, HEIGHT);
  sea.addColorStop(0, css(mix(new THREE.Color('#6fa8bf'), haze, 0.45)));
  sea.addColorStop(1, css(mix(new THREE.Color('#2f6f8d'), accent, 0.12)));
  ctx.fillStyle = sea;
  ctx.fillRect(0, horizonY, WIDTH, HEIGHT - horizonY);

  // A few bands of glint, spaced further apart nearer the horizon.
  ctx.fillStyle = 'rgba(255,255,255,0.18)';
  for (let y = horizonY + 8; y < HEIGHT; y += 9) {
    const t = (y - horizonY) / (HEIGHT - horizonY);
    for (let i = 0; i < 26; i++) {
      if (rng() > 0.35) continue;
      const w = 10 + rng() * 60 * t;
      ctx.fillRect(rng() * WIDTH, y, w, 1.6);
    }
  }
}

/** A treeline, in two depths. */
function paintForest(
  ctx: CanvasRenderingContext2D,
  rng: Rng,
  horizonY: number,
  haze: THREE.Color
): void {
  for (const [fade, scale] of [[0.6, 0.5], [0.22, 0.85]] as const) {
    ctx.fillStyle = css(mix(new THREE.Color('#1f3326'), haze, fade));
    let x = -20;
    while (x < WIDTH + 20) {
      const w = 14 + rng() * 22;
      const h = horizonY * (0.12 + rng() * 0.22) * scale;
      ctx.beginPath();
      ctx.moveTo(x, horizonY + 6);
      ctx.lineTo(x + w / 2, horizonY - h);
      ctx.lineTo(x + w, horizonY + 6);
      ctx.closePath();
      ctx.fill();
      x += w * 0.62;
    }
  }
}

/** Ranges, each one hazier than the one in front. */
function paintPeaks(
  ctx: CanvasRenderingContext2D,
  rng: Rng,
  horizonY: number,
  haze: THREE.Color
): void {
  const ranges = [
    { fade: 0.74, height: 0.5, step: 150 },
    { fade: 0.5, height: 0.38, step: 190 },
    { fade: 0.2, height: 0.26, step: 240 },
  ];

  for (const range of ranges) {
    const rock = mix(new THREE.Color('#3b4459'), haze, range.fade);
    ctx.fillStyle = css(rock);
    ctx.beginPath();
    ctx.moveTo(0, horizonY + 10);
    let x = -60;
    while (x < WIDTH + 120) {
      const w = range.step * (0.6 + rng() * 0.8);
      const h = horizonY * range.height * (0.5 + rng() * 0.7);
      ctx.lineTo(x + w / 2, horizonY - h);
      ctx.lineTo(x + w, horizonY - h * 0.25);
      x += w;
    }
    ctx.lineTo(WIDTH, horizonY + 10);
    ctx.closePath();
    ctx.fill();
  }
}

/** Low hills and a field. */
function paintMeadow(
  ctx: CanvasRenderingContext2D,
  rng: Rng,
  horizonY: number,
  haze: THREE.Color,
  accent: THREE.Color
): void {
  for (const [fade, lift] of [[0.6, 0.16], [0.28, 0.09]] as const) {
    ctx.fillStyle = css(mix(new THREE.Color('#5c6b3f'), haze, fade));
    ctx.beginPath();
    ctx.moveTo(0, horizonY + 10);
    for (let x = 0; x <= WIDTH; x += 80) {
      ctx.lineTo(x, horizonY - horizonY * lift * (0.4 + rng()));
    }
    ctx.lineTo(WIDTH, horizonY + 10);
    ctx.closePath();
    ctx.fill();
  }

  const field = ctx.createLinearGradient(0, horizonY, 0, HEIGHT);
  field.addColorStop(0, css(mix(new THREE.Color('#7c8a4c'), haze, 0.3)));
  field.addColorStop(1, css(mix(new THREE.Color('#5d6a34'), accent, 0.15)));
  ctx.fillStyle = field;
  ctx.fillRect(0, horizonY, WIDTH, HEIGHT - horizonY);
}

// ---------------------------------------------------------------------------

type Rng = () => number;

function makeRng(seed: number): Rng {
  let a = (seed | 0) >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function mix(a: THREE.Color, b: THREE.Color, t: number): THREE.Color {
  return a.clone().lerp(b, t);
}

function css(color: THREE.Color): string {
  return `#${color.getHexString()}`;
}
