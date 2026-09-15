# Third-party models

Every file in this folder is **CC0 1.0 (public domain)**. No attribution is
legally required; it is recorded here anyway, because a research project should
be able to say where each of its assets came from.

Downloaded from [Poly Pizza](https://poly.pizza), which hosts these packs with
the authors' permission.

| File           | Model          | Author     | Pack               | Licence  |
| -------------- | -------------- | ---------- | ------------------ | -------- |
| `tree.glb`     | Tree Large     | Kenney     | Nature Kit         | CC0 1.0  |
| `books.glb`    | Books          | Kenney     | Furniture Kit      | CC0 1.0  |
| `building.glb` | Large Building | Kenney     | City Kit           | CC0 1.0  |
| `house.glb`    | House          | Kenney     | Suburban Houses    | CC0 1.0  |
| `speaker.glb`  | Speaker        | Kenney     | Concert Pack       | CC0 1.0  |
| `bookcase.glb` | Bookcase Open  | Kenney     | Furniture Kit      | CC0 1.0  |
| `desk.glb`     | Desk           | Kenney     | Furniture Kit      | CC0 1.0  |
| `sofa.glb`     | Lounge Sofa    | Kenney     | Furniture Kit      | CC0 1.0  |
| `lamp.glb`     | Lamp Round Floor | Kenney   | Furniture Kit      | CC0 1.0  |
| `plant.glb`    | Potted Plant   | Kenney     | Furniture Kit      | CC0 1.0  |

### Landmarks

The wider world — everything standing outside the central ring of props. What
each one *means*, and what in the diary has to be said before it appears, is
documented in `src/world/landmarks.ts`; this table only records where the
geometry came from.

| File               | Model                   | Author | Pack              | Licence |
| ------------------ | ----------------------- | ------ | ----------------- | ------- |
| `pirate/watchtower.glb`   | Tower Complete Small    | Kenney | Pirate Kit        | CC0 1.0 |
| `pirate/boat.glb`         | Boat Row Small          | Kenney | Pirate Kit        | CC0 1.0 |
| `pirate/ship.glb`         | Ship Small              | Kenney | Pirate Kit        | CC0 1.0 |
| `pirate/wreck.glb`        | Ship Wreck              | Kenney | Pirate Kit        | CC0 1.0 |
| `pirate/jetty.glb`        | Structure Platform Dock | Kenney | Pirate Kit        | CC0 1.0 |
| `pirate/sea-rocks.glb`    | Rocks A                 | Kenney | Pirate Kit        | CC0 1.0 |
| `crag.glb`         | Rock Tall D             | Kenney | Nature Kit        | CC0 1.0 |
| `cairn.glb`        | Statue Obelisk          | Kenney | Nature Kit        | CC0 1.0 |
| `ruin.glb`         | Statue Column Damaged   | Kenney | Nature Kit        | CC0 1.0 |
| `cave.glb`         | Cliff Cave Rock         | Kenney | Nature Kit        | CC0 1.0 |
| `waterfall.glb`    | Cliff Waterfall Rock    | Kenney | Nature Kit        | CC0 1.0 |
| `bridge.glb`       | Bridge Wood             | Kenney | Nature Kit        | CC0 1.0 |
| `gate.glb`         | Fence Gate              | Kenney | Nature Kit        | CC0 1.0 |
| `palm.glb`         | Tree Palm Short         | Kenney | Nature Kit        | CC0 1.0 |
| `tree-pine.glb`    | Tree Cone Dark          | Kenney | Nature Kit        | CC0 1.0 |
| `tree-oak.glb`     | Tree Oak                | Kenney | Nature Kit        | CC0 1.0 |
| `survival/tent.glb`         | Tent                    | Kenney | Survival Kit      | CC0 1.0 |
| `survival/campfire.glb`     | Campfire Stand          | Kenney | Survival Kit      | CC0 1.0 |
| `survival/signpost.glb`     | Signpost                | Kenney | Survival Kit      | CC0 1.0 |
| `fantasy/windmill.glb`     | Windmill                | Kenney | Fantasy Town Kit  | CC0 1.0 |
| `fantasy/watermill.glb`    | Watermill               | Kenney | Fantasy Town Kit  | CC0 1.0 |
| `fantasy/broken-fence.glb` | Fence Broken            | Kenney | Fantasy Town Kit  | CC0 1.0 |
| `graveyard/bench.glb`        | Bench                   | Kenney | Graveyard Kit     | CC0 1.0 |

### A note on the subfolders

Four of Kenney's kits — pirate, survival, fantasy town and graveyard — do not
carry their colours in the model. Every mesh samples a shared palette atlas, and
the `.glb` refers to it by the *relative* path `Textures/colormap.png`. Each kit
has its own atlas under that same name, so they cannot all sit in one directory:
the second one to be copied in would silently repaint the first.

Hence one folder per kit, each with its own `Textures/colormap.png` beside the
models that need it. Move a `.glb` out of its folder and it loses its texture
and renders plain white — which is the failure this layout exists to prevent,
and which is easy to miss because nothing errors.

The Nature Kit and the furniture models above are the other sort: no texture,
just per-material colours. Those sit flat in this directory and tint.

## Adding more

`src/world/models.ts` is the registry. Drop a `.glb` in here and add a line to
`MOTIF_MODELS`; scale, origin and orientation are normalised on load.

Two things to check before you do:

- **Untextured models tint, textured ones do not.** The flat files here use
  plain coloured materials and take the diary's emotional colour. The four
  subfolders are atlas-textured and keep their own colours — still fine, just
  inert, and `landmarks.ts` darkens them instead so they sit in the scene.
- **Nothing flat.** Models are normalised by height, so a rug or a tabletop
  scales enormously. `normalise()` falls back to width when a model is much
  wider than it is tall, but a genuinely flat object is better drawn than
  loaded.
