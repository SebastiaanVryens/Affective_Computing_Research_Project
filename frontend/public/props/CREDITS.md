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

## Adding more

`src/world/models.ts` is the registry. Drop a `.glb` in here and add a line to
`MOTIF_MODELS`; scale, origin and orientation are normalised on load.

Two things to check before you do:

- **Untextured models tint, textured ones do not.** Everything above uses plain
  coloured materials, so it takes the diary's emotional colour. A model with a
  baked texture keeps its own colours — still fine, just inert.
- **Nothing flat.** Models are normalised by height, so a rug or a tabletop
  scales enormously. `normalise()` falls back to width when a model is much
  wider than it is tall, but a genuinely flat object is better drawn than
  loaded.
