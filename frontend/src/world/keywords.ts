/**
 * Keywords as floating text in the world.
 *
 * When the backend finishes a chunk, the words it found drift up around the
 * live orb, each tinted by the emotion of the sentence it came from. This is
 * the moment the app stops feeling like a mood ring and starts feeling like
 * something that listened to you.
 *
 * Rendered as canvas-texture sprites rather than via a text geometry library:
 * no extra dependency, no font loading, and sprites always face the camera,
 * which is what you want for floating words anyway.
 */

import * as THREE from 'three';
import { PALETTE, type Emotion } from '../emotions';

/** Device-pixel scale for the label canvases. 2 is crisp without being wasteful. */
const TEXTURE_SCALE = 2;
const FONT_PX = 44;

const LIFETIME_MS = 14_000;
const FADE_IN_MS = 700;
const FADE_OUT_MS = 3_000;

interface FloatingWord {
  sprite: THREE.Sprite;
  material: THREE.SpriteMaterial;
  texture: THREE.CanvasTexture;
  bornAt: number;
  velocity: THREE.Vector3;
  /** Horizontal sway, so words don't rise in straight parallel lines. */
  swayPhase: number;
  swayAmount: number;
  baseScale: number;
}

export class KeywordField {
  readonly group = new THREE.Group();

  private words: FloatingWord[] = [];
  /** Guards against the same keyword spawning on every streamed chunk. */
  private recentlySpawned = new Map<string, number>();

  /**
   * Spawn a word into the world.
   *
   * `weight` in [0,1] scales the text size — a keyword the extractor was sure
   * about arrives larger.
   */
  spawn(text: string, emotion: Emotion, weight = 0.5): void {
    const key = text.toLowerCase();
    const now = performance.now();

    // The streaming chunks overlap in content, so the same word arrives
    // repeatedly. Re-spawning it each time produces a stuttering pile-up.
    const last = this.recentlySpawned.get(key);
    if (last !== undefined && now - last < LIFETIME_MS * 0.6) return;
    this.recentlySpawned.set(key, now);

    const palette = PALETTE[emotion];
    const { texture, aspect } = makeLabelTexture(text, palette.glow);

    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      // Additive keeps words luminous against the sky and stops them reading as
      // flat UI stickers pasted over the scene.
      blending: THREE.AdditiveBlending,
    });

    const sprite = new THREE.Sprite(material);
    const baseScale = 1.1 + weight * 1.3;
    sprite.scale.set(baseScale * aspect, baseScale, 1);

    // Ring around the live orb, at a random angle and a slightly random radius.
    const angle = Math.random() * Math.PI * 2;
    const radius = 2.4 + Math.random() * 2.6;
    sprite.position.set(
      Math.cos(angle) * radius,
      0.4 + Math.random() * 1.6,
      Math.sin(angle) * radius
    );

    this.words.push({
      sprite,
      material,
      texture,
      bornAt: now,
      velocity: new THREE.Vector3(
        Math.cos(angle) * 0.12,
        0.28 + Math.random() * 0.22,
        Math.sin(angle) * 0.12
      ),
      swayPhase: Math.random() * Math.PI * 2,
      swayAmount: 0.15 + Math.random() * 0.2,
      baseScale,
    });
    this.group.add(sprite);
  }

  update(delta: number, elapsed: number): void {
    const now = performance.now();

    for (let i = this.words.length - 1; i >= 0; i--) {
      const word = this.words[i];
      const age = now - word.bornAt;

      if (age > LIFETIME_MS) {
        this.remove(i);
        continue;
      }

      word.sprite.position.addScaledVector(word.velocity, delta);
      word.sprite.position.x +=
        Math.sin(elapsed * 0.7 + word.swayPhase) * word.swayAmount * delta;

      // Rise slows as the word fades, so it settles rather than shooting off.
      word.velocity.multiplyScalar(1 - delta * 0.35);

      const fadeIn = Math.min(1, age / FADE_IN_MS);
      const remaining = LIFETIME_MS - age;
      const fadeOut = Math.min(1, remaining / FADE_OUT_MS);
      word.material.opacity = fadeIn * fadeOut * 0.95;

      // A slight scale-up on entry gives the word a sense of arriving.
      const pop = 0.85 + 0.15 * fadeIn;
      word.sprite.scale.set(
        word.baseScale * pop * (word.sprite.scale.x / word.sprite.scale.y),
        word.baseScale * pop,
        1
      );
    }

    // Keep the dedupe map from growing without bound over a long session.
    if (this.recentlySpawned.size > 200) {
      for (const [key, time] of this.recentlySpawned) {
        if (now - time > LIFETIME_MS) this.recentlySpawned.delete(key);
      }
    }
  }

  private remove(index: number): void {
    const word = this.words[index];
    this.group.remove(word.sprite);
    word.material.dispose();
    word.texture.dispose();
    this.words.splice(index, 1);
  }

  clear(): void {
    for (let i = this.words.length - 1; i >= 0; i--) this.remove(i);
    this.recentlySpawned.clear();
  }

  dispose(): void {
    this.clear();
  }
}

function makeLabelTexture(
  text: string,
  color: string
): { texture: THREE.CanvasTexture; aspect: number } {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;

  const font = `600 ${FONT_PX}px "Inter", system-ui, sans-serif`;
  ctx.font = font;
  const metrics = ctx.measureText(text);

  // Padding has to cover the glow blur, or the shadow gets clipped at the edges.
  const padding = FONT_PX * 0.6;
  const width = Math.ceil(metrics.width + padding * 2);
  const height = Math.ceil(FONT_PX * 1.6 + padding);

  canvas.width = width * TEXTURE_SCALE;
  canvas.height = height * TEXTURE_SCALE;
  ctx.scale(TEXTURE_SCALE, TEXTURE_SCALE);

  ctx.font = font;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  // Two passes: a wide soft glow, then the crisp glyph on top.
  ctx.shadowColor = color;
  ctx.shadowBlur = FONT_PX * 0.5;
  ctx.fillStyle = color;
  ctx.fillText(text, width / 2, height / 2);
  ctx.shadowBlur = 0;
  ctx.fillStyle = '#ffffff';
  ctx.globalAlpha = 0.92;
  ctx.fillText(text, width / 2, height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  // Sprites are viewed at an angle as the galaxy turns; anisotropy is cheap
  // here and stops the text smearing.
  texture.anisotropy = 4;
  texture.needsUpdate = true;

  return { texture, aspect: width / height };
}
