/**
 * Score face-api's stock expression head on a manifest of frames.
 *
 *   node scripts/eval-faceapi.mjs \
 *     --manifest ../data/meld_faces/test_frames.csv \
 *     --root ../data/meld_faces \
 *     --out ../data/meld_faces/faceapi_test.json
 *
 * This is the baseline the trained head has to beat, and until it runs the
 * README's claim that face-api "reads a resting face as slightly sad" is an
 * observation rather than a result — the face channel is the only one in the
 * project with no number attached to it.
 *
 * Fidelity is the whole point, so this deliberately reproduces
 * src/capture/face.ts rather than doing anything smarter: the same three nets
 * loaded from the same `public/models` weights the browser fetches, the same
 * TinyFaceDetectorOptions, the same 0.35 score floor applied twice (once inside
 * the detector, once as a post-filter), and the same label mapping. A baseline
 * that ran a better pipeline than the app would flatter the app.
 *
 * It runs on the WASM backend because the alternative, @tensorflow/tfjs-node,
 * is a native addon that needs a toolchain on Windows. WASM is slower and
 * numerically identical.
 *
 * Frames are read whole and face-api does its own detection, which is why
 * prepare_meld_video.py writes `{split}_frames.csv` separately from the crop
 * manifest: scoring face-api only on frames *our* detector already approved
 * would hand it a pre-filtered test set and call the comparison fair.
 */

import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const HERE = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = path.resolve(HERE, '..');

/**
 * face-api's expression label -> ours, and the canonical order.
 * Must stay identical to src/capture/face.ts's LABEL_MAP and to
 * backend/app/emotions.py's EMOTIONS — the vector crosses to Python as a bare
 * array and is indexed positionally on both sides.
 */
const EMOTIONS = ['neutral', 'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise'];
const LABEL_MAP = {
  neutral: 'neutral',
  happy: 'joy',
  sad: 'sadness',
  angry: 'anger',
  fearful: 'fear',
  disgusted: 'disgust',
  surprised: 'surprise',
};

function parseArgs(argv) {
  const args = {
    manifest: null,
    root: null,
    out: null,
    limit: 0,
    inputSize: 224,      // face.ts: multiple of 32, smallest that finds a face reliably
    scoreThreshold: 0.35, // face.ts: MIN_DETECTION_SCORE
    shard: 0,
    shards: 1,
  };
  for (let i = 2; i < argv.length; i += 2) {
    const key = argv[i].replace(/^--/, '');
    const value = argv[i + 1];
    if (key === 'manifest') args.manifest = value;
    else if (key === 'root') args.root = value;
    else if (key === 'out') args.out = value;
    else if (key === 'limit') args.limit = Number(value);
    else if (key === 'input-size') args.inputSize = Number(value);
    else if (key === 'score-threshold') args.scoreThreshold = Number(value);
    else if (key === 'shard') args.shard = Number(value);
    else if (key === 'shards') args.shards = Number(value);
    else throw new Error(`unknown argument --${key}`);
  }
  for (const required of ['manifest', 'root', 'out']) {
    if (!args[required]) throw new Error(`--${required} is required`);
  }
  return args;
}

/** Minimal CSV reader. Handles quoted fields because pandas quotes on demand. */
function readCsv(file) {
  const text = fs.readFileSync(file, 'utf8').replace(/\r\n/g, '\n').trim();
  const rows = [];
  let field = '';
  let row = [];
  let quoted = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else quoted = false;
      } else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
    else field += c;
  }
  row.push(field);
  rows.push(row);

  const header = rows.shift();
  return rows
    .filter((r) => r.length === header.length)
    .map((r) => Object.fromEntries(header.map((h, i) => [h, r[i]])));
}

/** JPEG -> a tf int32 tensor of shape [h, w, 3]. */
function decodeToTensor(tf, file) {
  const jpeg = require('jpeg-js');
  const raw = jpeg.decode(fs.readFileSync(file), { useTArray: true });
  // jpeg-js hands back RGBA; face-api wants three channels.
  const rgb = new Uint8Array(raw.width * raw.height * 3);
  for (let p = 0, q = 0; p < raw.data.length; p += 4, q += 3) {
    rgb[q] = raw.data[p];
    rgb[q + 1] = raw.data[p + 1];
    rgb[q + 2] = raw.data[p + 2];
  }
  return tf.tensor3d(rgb, [raw.height, raw.width, 3], 'int32');
}

/** face-api's expression object -> a normalised vector in EMOTIONS order. */
function toVector(expressions) {
  const vector = new Array(EMOTIONS.length).fill(0);
  for (const [theirs, ours] of Object.entries(LABEL_MAP)) {
    const value = expressions[theirs];
    if (typeof value === 'number') vector[EMOTIONS.indexOf(ours)] += value;
  }
  const total = vector.reduce((a, b) => a + b, 0);
  return total > 1e-9 ? vector.map((v) => v / total) : vector.map(() => 1 / EMOTIONS.length);
}

async function main() {
  const args = parseArgs(process.argv);

  const wasm = require('@tensorflow/tfjs-backend-wasm');
  wasm.setWasmPaths(path.join(FRONTEND, 'node_modules/@tensorflow/tfjs-backend-wasm/dist/'));
  const faceapi = require('@vladmandic/face-api/dist/face-api.node-wasm.js');
  const tf = faceapi.tf;

  await tf.setBackend('wasm');
  await tf.ready();

  const weights = path.join(FRONTEND, 'public/models');
  if (!fs.existsSync(path.join(weights, 'face_expression_model.bin'))) {
    throw new Error(`face-api weights not found in ${weights} — run: npm run fetch-models`);
  }
  await faceapi.nets.tinyFaceDetector.loadFromDisk(weights);
  await faceapi.nets.faceLandmark68TinyNet.loadFromDisk(weights);
  await faceapi.nets.faceExpressionNet.loadFromDisk(weights);
  console.log(`backend=${tf.getBackend()} weights=${weights}`);

  let rows = readCsv(args.manifest);
  if (args.limit > 0) rows = rows.slice(0, args.limit);

  // Sharding is round-robin rather than contiguous so every shard sees a
  // similar mix of clips and emotions. That keeps each shard's progress rate
  // representative, and means a shard that dies tells you roughly what fraction
  // of the whole was lost rather than which emotions went missing.
  const total = rows.length;
  if (args.shards > 1) {
    rows = rows.filter((_, i) => i % args.shards === args.shard);
    console.log(
      `shard ${args.shard + 1}/${args.shards}: ${rows.length} of ${total} frames`
    );
  } else {
    console.log(`${rows.length} frames from ${args.manifest}`);
  }

  const options = new faceapi.TinyFaceDetectorOptions({
    inputSize: args.inputSize,
    scoreThreshold: args.scoreThreshold,
  });

  const results = [];
  let found = 0;
  let totalMs = 0;
  const started = Date.now();

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const file = path.join(args.root, row.path);
    let record = {
      utterance_key: row.utterance_key,
      frame_index: Number(row.frame_index),
      emotion: row.emotion,
      found: false,
      det_score: 0,
      probs: null,
    };

    let tensor = null;
    try {
      tensor = decodeToTensor(tf, file);
      const t0 = Date.now();
      const detection = await faceapi
        .detectSingleFace(tensor, options)
        .withFaceLandmarks(true) // `true` selects the tiny landmark net, as face.ts does
        .withFaceExpressions();
      totalMs += Date.now() - t0;

      // face.ts applies the floor a second time after detection; so do we.
      if (detection && detection.detection.score >= args.scoreThreshold) {
        record.found = true;
        record.det_score = detection.detection.score;
        record.probs = toVector(detection.expressions);
        found++;
      }
    } catch (error) {
      record.error = String(error.message ?? error).slice(0, 200);
    } finally {
      if (tensor) tensor.dispose();
    }

    results.push(record);

    if ((i + 1) % 500 === 0) {
      const rate = (i + 1) / ((Date.now() - started) / 1000);
      console.log(
        `  ${i + 1}/${rows.length}  found=${found} (${((100 * found) / (i + 1)).toFixed(1)}%)  ` +
          `${rate.toFixed(1)} frames/s`
      );
    }
  }

  const payload = {
    source: 'face-api stock faceExpressionNet (the head src/capture/face.ts ships)',
    detector: 'tinyFaceDetector',
    backend: tf.getBackend(),
    weights_dir: weights,
    input_size: args.inputSize,
    score_threshold: args.scoreThreshold,
    emotions: EMOTIONS,
    frames: rows.length,
    frames_with_face: found,
    mean_inference_ms: results.length ? totalMs / results.length : 0,
    results,
  };
  fs.mkdirSync(path.dirname(args.out), { recursive: true });
  fs.writeFileSync(args.out, JSON.stringify(payload));

  console.log(
    `\n${found}/${rows.length} frames had a detectable face ` +
      `(${((100 * found) / rows.length).toFixed(1)}%), ` +
      `mean ${(totalMs / Math.max(1, results.length)).toFixed(1)} ms/frame`
  );
  console.log(`-> ${args.out}`);
  console.log('Next: python -m training.eval_face_baseline');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
