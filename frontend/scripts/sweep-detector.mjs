/**
 * How much of the face channel is lost in the detector, not the classifier?
 *
 *   node scripts/sweep-detector.mjs --manifest ../data/meld_faces/dev_frames.csv \
 *     --root ../data/meld_faces --out ../data/meld_faces/detector_sweep.json
 *
 * The baseline run found face-api detecting a face in only ~53% of MELD frames,
 * while the YuNet pass used for training found a speaker in ~94% of the same
 * clips. If that gap is real, no amount of retraining the *expression head*
 * recovers it: on a frame with no detection the channel emits nothing at all,
 * and fusion.py treats that as a distinct state from "the face looked neutral".
 *
 * face.ts hardcodes two numbers that decide this — `inputSize: 224` and
 * `scoreThreshold: 0.35` — and both are pure trade-offs against latency and
 * false positives rather than tuned values. This sweeps them on a subsample and
 * reports detection rate against cost, so the choice can be made from a curve.
 *
 * Deliberately measures detection ONLY. It never runs the landmark or expression
 * nets, because the question here is how many frames the channel can see at all,
 * and mixing in classifier accuracy would blur two separable problems.
 */

import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const HERE = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = path.resolve(HERE, '..');

// face.ts's current settings, marked so the report can point at them.
const SHIPPED = { inputSize: 224, scoreThreshold: 0.35 };

// inputSize must be a multiple of 32 for the tiny detector.
const INPUT_SIZES = [128, 160, 224, 320, 416, 512];
const SCORE_THRESHOLDS = [0.2, 0.35, 0.5];

function parseArgs(argv) {
  const args = { manifest: null, root: null, out: null, sample: 1200 };
  for (let i = 2; i < argv.length; i += 2) {
    const key = argv[i].replace(/^--/, '');
    const value = argv[i + 1];
    if (key === 'manifest') args.manifest = value;
    else if (key === 'root') args.root = value;
    else if (key === 'out') args.out = value;
    else if (key === 'sample') args.sample = Number(value);
    else throw new Error(`unknown argument --${key}`);
  }
  for (const required of ['manifest', 'root', 'out']) {
    if (!args[required]) throw new Error(`--${required} is required`);
  }
  return args;
}

function readCsv(file) {
  const text = fs.readFileSync(file, 'utf8').replace(/\r\n/g, '\n').trim();
  const rows = [];
  let field = '', row = [], quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else quoted = false; }
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
    else field += c;
  }
  row.push(field);
  rows.push(row);
  const header = rows.shift();
  return rows.filter((r) => r.length === header.length)
    .map((r) => Object.fromEntries(header.map((h, i) => [h, r[i]])));
}

async function main() {
  const args = parseArgs(process.argv);

  const wasm = require('@tensorflow/tfjs-backend-wasm');
  wasm.setWasmPaths(path.join(FRONTEND, 'node_modules/@tensorflow/tfjs-backend-wasm/dist/'));
  const faceapi = require('@vladmandic/face-api/dist/face-api.node-wasm.js');
  const jpeg = require('jpeg-js');
  const tf = faceapi.tf;

  await tf.setBackend('wasm');
  await tf.ready();
  await faceapi.nets.tinyFaceDetector.loadFromDisk(path.join(FRONTEND, 'public/models'));

  let rows = readCsv(args.manifest);
  // Evenly spaced subsample rather than the first N, so the sample spans the
  // whole manifest (which is ordered by dialogue) instead of a handful of scenes.
  if (args.sample > 0 && rows.length > args.sample) {
    const step = rows.length / args.sample;
    rows = Array.from({ length: args.sample }, (_, i) => rows[Math.floor(i * step)]);
  }
  console.log(`sweeping ${rows.length} frames from ${args.manifest}`);

  // Decode once and reuse across every setting: decoding dominates otherwise,
  // and it would be charged to whichever setting happened to run first.
  const images = [];
  for (const row of rows) {
    try {
      const raw = jpeg.decode(fs.readFileSync(path.join(args.root, row.path)), { useTArray: true });
      const rgb = new Uint8Array(raw.width * raw.height * 3);
      for (let p = 0, q = 0; p < raw.data.length; p += 4, q += 3) {
        rgb[q] = raw.data[p]; rgb[q + 1] = raw.data[p + 1]; rgb[q + 2] = raw.data[p + 2];
      }
      images.push({ rgb, w: raw.width, h: raw.height, emotion: row.emotion });
    } catch { /* skip unreadable frame */ }
  }
  console.log(`decoded ${images.length}`);

  const results = [];
  for (const inputSize of INPUT_SIZES) {
    for (const scoreThreshold of SCORE_THRESHOLDS) {
      const options = new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold });
      let found = 0;
      const perClass = {};
      const started = Date.now();

      for (const image of images) {
        const tensor = tf.tensor3d(image.rgb, [image.h, image.w, 3], 'int32');
        try {
          const detection = await faceapi.detectSingleFace(tensor, options);
          const hit = detection && detection.score >= scoreThreshold;
          if (hit) found++;
          perClass[image.emotion] = perClass[image.emotion] || { n: 0, found: 0 };
          perClass[image.emotion].n++;
          if (hit) perClass[image.emotion].found++;
        } finally {
          tensor.dispose();
        }
      }

      const ms = (Date.now() - started) / images.length;
      const rate = found / images.length;
      const shipped = inputSize === SHIPPED.inputSize && scoreThreshold === SHIPPED.scoreThreshold;
      results.push({ inputSize, scoreThreshold, rate, ms, perClass, shipped });
      console.log(
        `  inputSize=${String(inputSize).padStart(3)} score=${scoreThreshold.toFixed(2)}  ` +
          `detected ${(100 * rate).toFixed(1)}%  ${ms.toFixed(1)} ms/frame` +
          (shipped ? '   <- what face.ts ships' : '')
      );
    }
  }

  fs.mkdirSync(path.dirname(args.out), { recursive: true });
  fs.writeFileSync(args.out, JSON.stringify({
    shipped: SHIPPED,
    frames: images.length,
    manifest: args.manifest,
    results,
  }, null, 2));

  const base = results.find((r) => r.shipped);
  const best = results.reduce((a, b) => (b.rate > a.rate ? b : a));
  if (base && best !== base) {
    console.log(
      `\nBest detection ${(100 * best.rate).toFixed(1)}% at inputSize=${best.inputSize} ` +
        `score=${best.scoreThreshold} (${best.ms.toFixed(1)} ms/frame) vs shipped ` +
        `${(100 * base.rate).toFixed(1)}% (${base.ms.toFixed(1)} ms/frame).`
    );
    console.log(
      `That is ${((best.rate - base.rate) * 100).toFixed(1)} points of frames the channel ` +
        `currently discards, for ${(best.ms - base.ms).toFixed(1)} ms more per frame.`
    );
    console.log('face.ts runs at 8 Hz, so the per-frame budget is 125 ms.');
  }
  console.log(`\n-> ${args.out}`);
}

main().catch((error) => { console.error(error); process.exit(1); });
