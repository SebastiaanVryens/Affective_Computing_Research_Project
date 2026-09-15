/**
 * The indoor layout.
 *
 * For the diary that is mostly books, work, home and the kettle — which is a
 * great many diaries, and the one case where putting the person on a hillside
 * would be a lie. Their mind is not a landscape. It is a room they spend most of
 * their time in, with their things in it.
 *
 * Drawn as a doll's house: four walls, all of them facing inward, relying on
 * nothing but backface culling. A wall between the camera and the room is
 * back-facing and simply is not drawn, so the near side always opens up and the
 * far side is always solid — from every angle, with no per-frame work and no
 * decision about which wall to hide. The ceiling behaves the same way looking
 * down. It is the oldest trick in architectural rendering and still the right
 * one.
 *
 * The well stays. It is a square hole in the floorboards with the shaft of light
 * coming up through it, which is a better image than the one outdoors — a hole
 * in the floor of the room you live in, with everything you have ever recorded
 * underneath it.
 */

import * as THREE from 'three';
import { PALETTE, type EmotionVector, dominant, intensity } from '../emotions';
import { NOISE_2D } from './glsl';
import { WELL_RADIUS } from './terrain';
import { motifModel } from './models';
import { WindowView, type ViewKind } from './view';

/** Half the floor's width. Matched to ISLAND_RADIUS so the camera rig fits it. */
const HALF = 13;
const WALL_HEIGHT = 8.5;
const FLOOR_THICKNESS = 0.6;

/** Half-width of the square opening in the floor. */
const WELL_HALF = WELL_RADIUS;

export class Room {
  readonly group = new THREE.Group();

  private owned: Array<THREE.BufferGeometry | THREE.Material> = [];
  /** Everything beyond the glass. Owns its own geometry; cleared with the room. */
  private view = new WindowView();
  private lampMaterial: THREE.MeshBasicMaterial | null = null;
  /** The glazing's base colour, so `update` has something stable to modulate. */
  private daylight = new THREE.Color('#cfe4f5');

  /**
   * Build the room.
   *
   * `lifetimeTotals` tints the walls, faintly. It is the same move the ground
   * makes outdoors — the place is coloured by the diary — but the wash is much
   * weaker here, because a wall is a wall and a room that changed colour with
   * the mood would read as a nightclub rather than as somewhere someone lives.
   */
  rebuild(lifetimeTotals: EmotionVector, view: ViewKind, sky: THREE.Color, seed: number): void {
    this.clear();

    // The *dominant* emotion's own colour, not the average of all seven.
    //
    // Averaging is right for the sky, where the whole distribution is on show at
    // once. It is wrong here, because an average of seven hues is grey and a
    // grey room says nothing. Inside Out's rooms are each one colour for exactly
    // this reason — you know whose room you are in from the doorway — and a
    // single emotional wash over a room full of somebody's things is the most
    // this layout can say in one glance.
    //
    // How *strongly* it washes rides on clarity: a diary with one clear
    // prevailing feeling gets a room that commits to it, and an ambivalent one
    // gets something close to plain plaster, because it has not earned the claim.
    const lead = dominant(lifetimeTotals);
    const clarity = intensity(lifetimeTotals);
    const mood = new THREE.Color(PALETTE[lead].base);
    const wash = 0.3 + 0.45 * Math.min(1, clarity * 2.4);

    const wall = this.standard(new THREE.Color('#b9b2a8').lerp(mood, wash), 0.95);
    const trim = this.standard(
      new THREE.Color('#8e867c').lerp(mood, wash * 0.75).multiplyScalar(0.7),
      0.7
    );
    const floorMaterial = this.floorMaterial(mood, wash);

    this.buildFloor(floorMaterial, trim);
    this.buildWalls(wall, trim);
    this.buildGlazing(mood, view, sky, seed);
    this.buildRug(mood, wash);
    this.buildFurniture(mood, wash);
  }

  /**
   * Four boards around a square hole.
   *
   * A single slab would need a hole cut in it, which means either CSG or a
   * bespoke geometry. Four boxes give the same thing for four lines, and the
   * seams between them land under the skirting and the rug.
   */
  private buildFloor(surface: THREE.Material, trim: THREE.Material): void {
    const span = HALF - WELL_HALF;
    const mid = WELL_HALF + span / 2;

    for (const [sx, sz] of [[0, -1], [0, 1], [-1, 0], [1, 0]] as const) {
      const width = sx === 0 ? HALF * 2 : span;
      const depth = sz === 0 ? WELL_HALF * 2 : span;
      const geometry = this.own(new THREE.BoxGeometry(width, FLOOR_THICKNESS, depth));
      const board = new THREE.Mesh(geometry, surface);
      board.position.set(sx * mid, -FLOOR_THICKNESS / 2, sz * mid);
      this.group.add(board);
    }

    // A lip around the opening, so the hole reads as cut rather than as missing.
    const lip = this.own(new THREE.BoxGeometry(WELL_HALF * 2 + 0.5, 0.18, WELL_HALF * 2 + 0.5));
    const frame = new THREE.Mesh(lip, trim);
    frame.position.y = 0.04;
    this.group.add(frame);
  }

  /**
   * Walls, facing inward.
   *
   * PlaneGeometry's normal is +Z, so each plane is rotated to look at the middle
   * of the room. With the default front-side material that makes every wall
   * between the viewer and the room back-facing, and therefore invisible —
   * which is the entire doll's-house effect, for free.
   */
  private buildWalls(surface: THREE.Material, trim: THREE.Material): void {
    const geometry = this.own(new THREE.PlaneGeometry(HALF * 2, WALL_HEIGHT));
    const skirtGeo = this.own(new THREE.BoxGeometry(HALF * 2, 0.45, 0.22));

    const sides: Array<[number, number, number]> = [
      [0, -HALF, 0],
      [0, HALF, Math.PI],
      [-HALF, 0, Math.PI / 2],
      [HALF, 0, -Math.PI / 2],
    ];

    for (const [x, z, rotation] of sides) {
      // The far side is glazing, not plaster — buildGlazing fills it. A solid
      // plane here would sit directly in front of it and the room would go dark.
      if (z === -HALF) continue;

      const plane = new THREE.Mesh(geometry, surface);
      plane.position.set(x, WALL_HEIGHT / 2, z);
      plane.rotation.y = rotation;
      this.group.add(plane);

      const skirt = new THREE.Mesh(skirtGeo, trim);
      skirt.position.set(x, 0.22, z);
      skirt.rotation.y = rotation;
      skirt.translateZ(0.12);
      this.group.add(skirt);
    }

    for (const [x, z, rotation] of sides) {
      if (z !== -HALF) continue;
      const skirt = new THREE.Mesh(skirtGeo, trim);
      skirt.position.set(x, 0.22, z);
      skirt.rotation.y = rotation;
      skirt.translateZ(0.12);
      this.group.add(skirt);
    }

    // The ceiling, facing down — culled whenever the camera is above it, which
    // is always. It exists so the room is closed from any angle the camera
    // could reach rather than being a box with a missing face.
    const ceiling = this.own(new THREE.PlaneGeometry(HALF * 2, HALF * 2));
    const lid = new THREE.Mesh(ceiling, surface);
    lid.position.y = WALL_HEIGHT;
    lid.rotation.x = Math.PI / 2;
    this.group.add(lid);
  }

  /**
   * A glazed wall where the far side would be.
   *
   * The one thing that makes an interior read as inhabited rather than as a box:
   * somewhere the light is coming from. Emissive rather than an actual light,
   * because a real one here would be the only point light in the scene and would
   * have to be balanced against a rig built for open ground.
   */
  private buildGlazing(
    mood: THREE.Color,
    view: ViewKind,
    sky: THREE.Color,
    seed: number
  ): void {
    // What is actually outside, built from whatever the person talks about when
    // they are not indoors. Real geometry standing in real space beyond the
    // wall, not a picture of it — see view.ts for why that distinction turned
    // out to matter.
    this.view.rebuild(view, sky, mood, seed);
    this.group.add(this.view.group);

    // Glass, now that there is something behind it to see.
    //
    // Barely there: a faint cool wash with a little of the sky in it, so the
    // pane catches the light at a glancing angle and disappears head-on. It
    // still carries the room's daylight pulse in `update`, which is what keeps
    // the interior feeling lit from one side.
    this.daylight.copy(sky).lerp(new THREE.Color('#cfe4f5'), 0.5);
    const glass = this.material(
      new THREE.MeshBasicMaterial({
        color: this.daylight.clone(),
        transparent: true,
        opacity: 0.12,
        // Never written, or the glass would occlude the view behind it in the
        // transparent pass and the window would go opaque.
        depthWrite: false,
        side: THREE.DoubleSide,
        fog: false,
      })
    );
    const mullion = this.standard(new THREE.Color('#2d3138'), 0.55);

    // One wall replaced by glass, floor to ceiling. This is the single change
    // that stops the room reading as a box: a room with a small bright rectangle
    // in it is a cell with a window, whereas a room with one glazed side is
    // somewhere with an outside. It also gives the interior a direction — a
    // light side and a dark side — which is what every photograph of a room has
    // and what a uniformly-lit box never does.
    const pane = this.own(new THREE.PlaneGeometry(HALF * 2 - 1.2, WALL_HEIGHT - 0.9));
    const sheet = new THREE.Mesh(pane, glass);
    sheet.position.set(0, WALL_HEIGHT / 2, -HALF + 0.08);
    this.group.add(sheet);
    this.lampMaterial = glass as THREE.MeshBasicMaterial;

    // Mullions over the top of it. Thin, dark and regular — the frame is what
    // makes the bright plane read as glazing rather than as a hole in the wall.
    const post = this.own(new THREE.BoxGeometry(0.16, WALL_HEIGHT, 0.16));
    const rail = this.own(new THREE.BoxGeometry(HALF * 2, 0.16, 0.16));

    for (let i = -3; i <= 3; i++) {
      const bar = new THREE.Mesh(post, mullion);
      bar.position.set(i * (HALF / 3.2), WALL_HEIGHT / 2, -HALF + 0.16);
      this.group.add(bar);
    }
    for (const y of [WALL_HEIGHT * 0.34, WALL_HEIGHT * 0.7, WALL_HEIGHT - 0.08]) {
      const bar = new THREE.Mesh(rail, mullion);
      bar.position.set(0, y, -HALF + 0.16);
      this.group.add(bar);
    }
  }

  /**
   * A rug, so the floor has something on it besides the props.
   *
   * An annulus, not a disc: a disc large enough to be a rug is also large enough
   * to lie straight over the opening in the floor and seal it, which hides the
   * way down — the one thing in this room that has to stay findable.
   */
  private buildRug(mood: THREE.Color, wash: number): void {
    const geometry = this.own(
      new THREE.RingGeometry(WELL_HALF + 1.1, HALF * 0.66, 48, 1)
    );
    geometry.rotateX(-Math.PI / 2);
    const material = this.standard(new THREE.Color('#8d8378').lerp(mood, wash * 0.9), 1);
    const rug = new THREE.Mesh(geometry, material);
    rug.position.y = 0.05;
    this.group.add(rug);
  }

  /**
   * The furniture.
   *
   * Real models where public/props has one, boxes where it does not — the same
   * fallback every motif prop uses, so the room still furnishes itself if the
   * assets are missing or a download failed.
   *
   * Placed outside PROP_MAX_RADIUS in mindscape.ts, so they never fight the
   * diary's own props for floor space, and turned to face the glazing, which is
   * both where the light is and where the view is.
   */
  private buildFurniture(mood: THREE.Color, wash: number): void {
    const wood = this.standard(new THREE.Color('#5d4632').lerp(mood, wash * 0.4), 0.7);
    const fabric = this.standard(new THREE.Color('#8b8378').lerp(mood, wash), 0.95);
    const tint = { color: mood, amount: wash * 0.5 };

    /** A model if we have it, otherwise the caller's boxes. */
    const place = (
      key: string,
      x: number,
      z: number,
      yaw: number,
      fallback: () => void
    ): void => {
      const model = motifModel(key, tint);
      if (!model) {
        fallback();
        return;
      }
      model.position.set(x, 0, z);
      model.rotation.y = yaw;
      this.group.add(model);
    };

    const box = (
      material: THREE.Material,
      w: number, h: number, d: number,
      x: number, y: number, z: number
    ): void => {
      const mesh = new THREE.Mesh(this.own(new THREE.BoxGeometry(w, h, d)), material);
      mesh.position.set(x, y, z);
      this.group.add(mesh);
    };

    // A desk against the left wall, turned toward the window.
    place('room:desk', -HALF + 2.2, -3.0, Math.PI / 2, () => {
      box(wood, 0.9, 0.12, 4.6, -HALF + 1.6, 1.5, -3.0);
      box(wood, 0.7, 1.5, 0.16, -HALF + 1.6, 0.75, -5.0);
      box(wood, 0.7, 1.5, 0.16, -HALF + 1.6, 0.75, -1.0);
    });

    // Shelves along the back wall.
    place('room:bookcase', 1.0, HALF - 1.3, Math.PI, () => {
      for (const y of [0.1, 1.1, 2.1]) box(wood, 3.4, 0.16, 1.1, 1.0, y, HALF - 1.3);
    });
    place('room:bookcase', 5.2, HALF - 1.3, Math.PI, () => {
      for (const y of [0.1, 1.1, 2.1]) box(wood, 3.4, 0.16, 1.1, 5.2, y, HALF - 1.3);
    });

    // A couch on the right, facing the glazing.
    place('room:sofa', HALF - 2.6, 1.6, -Math.PI / 2, () => {
      box(fabric, 1.4, 0.7, 4.2, HALF - 2.6, 0.35, 1.6);
      box(fabric, 0.45, 1.1, 4.2, HALF - 1.8, 0.9, 1.6);
    });

    // Lamps and a plant, in the corners the big pieces leave empty.
    place('room:lamp', -HALF + 2.4, HALF - 2.6, 0, () => {});
    place('room:lamp', HALF - 2.4, -HALF + 3.4, 0, () => {});
    place('room:plant', -HALF + 2.8, 4.8, 0, () => {});
    place('room:plant', HALF - 3.2, HALF - 2.4, 0, () => {});
  }

  /**
   * Floorboards.
   *
   * Same patching approach the ground uses, for the same reason: a flat colour
   * over twenty-six units of floor reads as a sheet of card. Boards are a
   * one-dimensional stripe plus noise along their length, which is most of what
   * wood is at this distance.
   */
  private floorMaterial(mood: THREE.Color, wash: number): THREE.MeshStandardMaterial {
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#7a5c3e').lerp(mood, wash * 0.3),
      roughness: 0.72,
      metalness: 0.02,
      fog: true,
    });

    material.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', `#include <common>\n varying vec3 vRoomPos;`)
        .replace(
          '#include <begin_vertex>',
          `#include <begin_vertex>\n vRoomPos = (modelMatrix * vec4(transformed, 1.0)).xyz;`
        );

      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', `#include <common>\n varying vec3 vRoomPos;\n ${NOISE_2D}`)
        .replace(
          '#include <color_fragment>',
          `#include <color_fragment>
           {
             // Board seams every 0.9 units, and each board a slightly different
             // timber — which is the thing that stops it reading as wallpaper.
             float board = floor(vRoomPos.x / 0.9);
             float seam = smoothstep(0.0, 0.06, abs(fract(vRoomPos.x / 0.9) - 0.5) - 0.44);
             float shade = 0.86 + 0.28 * hash21(vec2(board, 3.0));
             float grain = fbm2(vec2(vRoomPos.z * 3.2, board * 7.0));
             diffuseColor.rgb *= shade * (0.88 + 0.24 * grain);
             diffuseColor.rgb *= 1.0 - seam * 0.45;
           }`
        );
    };

    this.owned.push(material);
    return material;
  }

  /**
   * The daylight shifts, very slightly. Nothing in here moves.
   *
   * Opacity rather than scale: the glazing is a whole wall now, and a wall that
   * changed size would be the most obvious thing on screen. A slow drift in
   * brightness reads as the light outside changing, which is what a window does
   * when you are not watching it.
   */
  update(elapsed: number): void {
    if (!this.lampMaterial) return;
    const drift = 0.94 + 0.06 * Math.sin(elapsed * 0.16);
    this.lampMaterial.color.setScalar(drift).multiply(this.daylight);
  }

  clear(): void {
    // The view owns geometry of its own — a few hundred instanced towers or
    // trees — so detaching its group is not enough to let go of it.
    this.view.clear();
    for (const child of [...this.group.children]) this.group.remove(child);
    for (const thing of this.owned) thing.dispose();
    this.owned = [];
    this.lampMaterial = null;
  }

  dispose(): void {
    this.clear();
  }

  // -- plumbing --------------------------------------------------------

  private own<T extends THREE.BufferGeometry>(geometry: T): T {
    this.owned.push(geometry);
    return geometry;
  }

  private material<T extends THREE.Material>(material: T): T {
    this.owned.push(material);
    return material;
  }

  private standard(color: THREE.Color, roughness: number): THREE.MeshStandardMaterial {
    return this.material(
      new THREE.MeshStandardMaterial({ color, roughness, metalness: 0.02, fog: true })
    );
  }
}
