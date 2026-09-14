/**
 * Copies the face-api weight files we actually use out of node_modules and into
 * public/models, so the app never has to hit a CDN at runtime.
 *
 * We only need three of the seven bundled models:
 *   - tiny_face_detector   : finds the face (fast enough for a live loop)
 *   - face_landmark_68_tiny: landmarks, required input for the expression net
 *   - face_expression      : the 7-class expression head
 */
import { cp, mkdir, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, '..', 'node_modules', '@vladmandic', 'face-api', 'model');
const dest = join(here, '..', 'public', 'models');

const WANTED = ['tiny_face_detector', 'face_landmark_68_tiny', 'face_expression'];

if (!existsSync(src)) {
  console.error(`Could not find face-api models at ${src}\nRun "npm install" first.`);
  process.exit(1);
}

await mkdir(dest, { recursive: true });

const files = await readdir(src);
let copied = 0;
for (const file of files) {
  if (!WANTED.some((w) => file.startsWith(w))) continue;
  await cp(join(src, file), join(dest, file));
  copied++;
}

console.log(`Copied ${copied} weight files into public/models`);
