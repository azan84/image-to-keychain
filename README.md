# image_to_keychain

Automated pipeline that converts a 2D stylized image (anime/chibi PNG) into
a 3D-printable multi-color keychain. Produces a single multi-object **3MF**
(primary, for Bambu Studio + AMS) and per-part **STL** files (fallback).

All parts share the same XY origin, so the slicer stacks them with zero
drift. Each part is a separate object so you can assign AMS filaments
independently per part.

## What it does

```
input.png
   │
   ▼
┌──────────────────┐
│ k-means quantize │   (merges tones/shades so soft shading ≠ a line)
└──────────────────┘
   │
   ├─► silhouette  ─► base plate              (z = 0 .. base_thickness)
   │
   ├─► color boundaries ─► line polygon ─────► lines layer   (z = base .. base + line_thickness)
   │
   └─► per-cluster polygons ─► one color part per cluster    (z = base .. base + color_thickness)
           │
           ▼
      keychain hole subtracted from every layer
           │
           ▼
      STL parts + multi-object 3MF
```

**Lines = color boundaries**, not dark pixels. A pixel is a "line" when its
cluster label differs from a neighbor's. Hair/dark regions become their own
color cluster instead of getting flattened into the line layer.

## Install (WSL Ubuntu)

```bash
# System deps — none required that need sudo.
# Python 3.10+ required.
pip3 install --user --break-system-packages -r requirements.txt
```

Optional (for trimesh niceties only, not required by the pipeline):

```bash
sudo apt-get install -y libspatialindex-dev   # enables rtree, used by trimesh
                                              # Scene.contains() etc.
```

The pipeline itself does **not** require `rtree` — it uses `manifold3d` for
boolean operations and writes polygons via `svgpathtools`.

## Quick start

```bash
# 1. Put your input PNG somewhere.
# 2. Edit config.yaml or pass overrides via CLI flags.
python3 image_to_keychain.py --config config.yaml
```

or

```bash
python3 image_to_keychain.py \
    --input path/to/chibi.png \
    --output-basename my_keychain \
    --target-size-mm 60 \
    --line-thickness 2.0 \
    --color-thickness 1.0 \
    --hole-type round --hole-diameter 4.0 --hole-position top-center
```

### Outputs

```
out/
├── my_keychain.3mf                     # multi-object file for Bambu Studio
└── my_keychain_parts/
    ├── base.stl
    ├── lines.stl
    ├── color_01_050303.stl             # named {idx}_{hex_rgb}
    ├── color_02_c82b27.stl
    └── ...
```

### Debug artifacts

With `save_intermediate: true` (default), the pipeline writes a bunch of
inspectable files under `intermediate/`:

| File | What it is |
|---|---|
| `01_preprocessed_for_vtracer.png` | input repainted with quantized palette |
| `02_colors.svg` | vtracer color trace (debug) |
| `03_line_mask.png` | binary mask fed to the line vectorizer |
| `04_lines.svg` | extracted line polygons |
| `05_lines_preview.png` | red overlay of extracted lines on the original |

Use `--stop-after lines` to stop right after line extraction and inspect
the preview — the simplest way to tune `line_dilate_px`, `max_colors`, etc.

## The two thicknesses

There are **two independently configurable top-layer thicknesses**:

- **`line_thickness`** (default `1.5` mm) — height of the line layer above
  the base.
- **`color_thickness`** (default `1.0` mm) — height of every color part
  above the base.

Both layers are anchored at the top of the base. Because
`line_thickness > color_thickness` by default, the lines **protrude above
the colors by `(line_thickness - color_thickness) = 0.5 mm`** — that 0.5 mm
ridge is what makes the outlines visually "pop" on the finished keychain.

If you want flat lines (same height as colors), set them equal. If you want
more dramatic outline emphasis, widen the gap.

## Key config parameters

| Parameter | Default | What it does |
|---|---|---|
| `target_size_mm` | 60 | Longest XY dimension of the subject (silhouette, not the full image). |
| `base_thickness` | 2.0 | Thickness of the solid backing plate. |
| `line_thickness` | 1.5 | Thickness of the line layer above the base. |
| `color_thickness` | 1.0 | Thickness of each color part above the base. |
| `max_colors` | 6 | k-means cluster count. AMS has 4 slots per unit. |
| `line_dilate_px` | 2 | Grow boundary lines by N pixels before extruding. Bump to 3–4 if lines are too thin to print. |
| `hole_type` | `round` | `round` / `double` / `slot` / `none`. |
| `hole_diameter` | 4.0 | For `round` / `double`. |
| `hole_position` | `top-center` | 8 named positions + `custom` (with `hole_custom_offset`). |
| `hole_edge_margin` | 3.0 | mm from silhouette edge to hole center. |

## In Bambu Studio

1. Open the 3MF. You'll see **N separate objects** (base, lines, and one
   per color cluster) — **not** one merged mesh.
2. Select each object and assign a filament (AMS slot).
3. Slice. The physical stack is: base → colors sitting on top of base →
   lines sitting on top of the colors (slightly taller than them).

## Troubleshooting

**Mesh is non-manifold / not watertight.**

The extrusion is done by `manifold3d.CrossSection.extrude`, which guarantees
manifold output. If a part still reports non-watertight, check that
`process=False` in `pipeline/extrude.py::_manifold_to_trimesh` — setting
`process=True` causes trimesh to merge near-coincident vertices between
disjoint speckle components, which breaks topology.

**Lines don't come out cleanly / are too thin.**

- Bump `line_dilate_px` from 2 to 3 or 4.
- Raise `max_colors` if distinct colors are being merged into one cluster
  (creates false negatives for lines at their boundary).
- Lower `max_colors` if subtle shading is producing spurious lines.

**Background is being treated as foreground.**

If your PNG has a baked-in opaque background (e.g., a grey/checkerboard
from an AI generator), corner-based detection handles this automatically.
If that fails, set `background_cluster_ids: [0, 3]` manually in the
config with the indices listed in the preprocess log.

**Hole isn't cutting through all layers.**

Check that `hole_edge_margin` isn't larger than the silhouette's half-width
— the hole center would fall outside the subject. The pipeline doesn't
validate this; it just lets shapely's `difference` no-op.

**Color-shade bleed** (e.g., a sliver of dark red separated from the main
red shirt).

Expected — it's a legitimate k-means cluster. Either:
- Raise `max_colors` to give it a dedicated filament.
- Lower `max_colors` to merge it into the main color.
- Or just let it be — it becomes its own AMS slot.

## Project layout

```
image_to_keychain/
├── README.md
├── requirements.txt
├── config.yaml
├── image_to_keychain.py          # CLI entry
├── pipeline/
│   ├── preprocess.py             # shared: alpha + k-means + corner bg detect
│   ├── svg_polygons.py           # mask -> vtracer SVG -> shapely MultiPolygon
│   ├── vectorize.py              # Step 1 — debug color SVG
│   ├── extract_lines.py          # Step 2 — color-boundary lines
│   ├── extract_colors.py         # Step 3 — per-cluster color parts
│   ├── silhouette.py             # Step 4 — outer boundary
│   ├── extrude.py                # Steps 5+6 — px->mm + manifold3d extrusion
│   ├── keyhole.py                # Step 7 — keychain hole polygon
│   ├── export_stl.py             # per-part STL export
│   └── export_3mf.py             # multi-object 3MF export
└── tests/
    ├── test_pipeline.py          # watertight + thickness + round-trip
    └── test_alignment.py         # XY origin, target size, Z stacking, hole cut-through
```

## Running tests

```bash
python3 -m pytest tests/ -v
```

All 12 tests run on the sample image at `../Keychain/Gemini_Generated_Image_k3zkqmk3zkqmk3zk.png`.
If that file isn't present, the tests skip.
