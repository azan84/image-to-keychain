---
title: Image to Keychain
emoji: 🔑
colorFrom: red
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Turn a PNG into a multi-color 3D-printable keychain
---

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

## Web UI (recommended)

Drop any PNG into a browser-based UI, tweak sliders with live preview,
download the 3MF. No folder dance, no editing YAML.

```bash
python3 app.py
```

Then open <http://localhost:7860>.

**Shortcuts** — double-clickable launchers that handle setup for you:

- **Windows** — `launch_ui.bat` (runs the server in WSL, opens your browser).
- **macOS** — `launch_ui.command` (creates a `.venv`, installs deps on
  first run, opens your browser). If Gatekeeper blocks it on the first
  launch:
  - **macOS 14 and older**: right-click → **Open** → confirm once.
  - **macOS 15 Sequoia+**: the right-click trick was removed. Go to
    **System Settings → Privacy & Security**, scroll down, click
    **Open Anyway** next to the blocked-file notice.
  - **Or, one line in Terminal**: `xattr -d com.apple.quarantine launch_ui.command`
  
  After any of these once, every future double-click just works.
- **Linux** — `python3 app.py` from a terminal. No launcher needed.

- Drag or paste your PNG into the upload zone (anywhere on disk).
- The right panel shows a top-down preview that updates whenever you
  release a slider.
- Click **Generate 3MF + STL** to run the full pipeline and download the
  output files (3MF for Bambu Studio, zip of per-part STLs for any
  slicer).

The first preview/export on a new image takes a few seconds (k-means +
vectorization). Subsequent slider tweaks on the same image are instant
because image-dependent state is cached.

## Quick start (CLI)

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

## The keychain tab

By default the pipeline builds a **tab** — a rounded-rectangle lug with
the keyring hole cut through it. The tab is exported as its **own
separate object** in the 3MF, aligned in XY/Z with the base but not
merged to it. The subject's own silhouette stays 100% intact; no hole
is cut into it.

    ┌──────────┐
    │   ⭕tab  │   ← separate object
    │
    │  ┌─────┐
    │  │chibi│     ← base + lines + colors, no hole anywhere
    │  │subj │
    │  └─────┘

In Bambu Studio you can drag the tab to overlap the subject wherever
you want, then right-click → **Assemble** (or just keep them as two
parts printed together). Both tab and base are at the same Z range so
they share the build plate at the same level.

Set `tab_enabled: false` to fall back to the legacy "cut the hole
directly into the silhouette" behavior — you'd typically only want
that if you deliberately want to pierce the subject (e.g. a background
plate where a hole in a corner is fine).

## Wording plate (name tag)

Add a line of text on a rounded **name plate** below the image — like the
parametric name keychains on MakerWorld. It comes out as **two extra
objects**, `text_plate` (the strip) and `text` (the raised letters), both
at the base Z range so they weld onto the keychain in the slicer:

```
   ┌──────────────┐
   │   subject    │
   └──────────────┘
   ┌──────────────┐
   │   LUQMAN     │   ← text_plate + text (separate bodies)
   └──────────────┘
```

CLI:

```bash
python image_to_keychain.py --input cat.png --text "LUQMAN"
python image_to_keychain.py --input cat.png --text "LINE 1
LINE 2" --text-height-mm 10 --text-recessed
```

In the **web UI**, open the *Wording plate (below image)* panel, tick
`text_enabled`, type the wording, and the live preview shows the plate.

**Font:** the default is a bold sans — **Arial Bold** on Windows, and
**DejaVu Sans Bold** on Linux / the hosted web app (installed via the
`fonts-dejavu-core` system package in `packages.txt`, an open/redistributable
font). Any other `.ttf`/`.otf` works too: pass its path, or upload it in the
UI's font field.

**Disney look (Waltograph):** Waltograph is **personal-use only**, so it is
**not** bundled in this (public) repo. To use it locally, drop `waltographUI.ttf`
into `./fonts` and set `text_font: disney` (or pass its path). The `fonts/`
`.gitignore` keeps it out of commits, so it won't be redistributed.

Assign a contrasting filament to the `text` body in Bambu Studio, drag the
plate up so it touches the keychain bottom, then multi-select both + the
keychain and **Combine / weld** them into one printed piece.

## Key config parameters

| Parameter | Default | What it does |
|---|---|---|
| `target_size_mm` | 60 | Longest XY dimension of the subject (silhouette, not the tab). |
| `base_thickness` | 2.0 | Thickness of the solid backing plate (including the tab). |
| `line_thickness` | 1.5 | Thickness of the line layer above the base. |
| `color_thickness` | 1.0 | Thickness of each color part above the base. |
| `max_colors` | 6 | k-means cluster count. AMS has 4 slots per unit. |
| `line_dilate_px` | 2 | Grow boundary lines by N pixels before extruding. Bump to 3–4 if lines are too thin to print. |
| **`tab_enabled`** | `true` | Build a tab that carries the hole. `false` → cut hole into silhouette. |
| **`tab_side`** | `top` | `top` / `bottom` / `left` / `right` — which silhouette edge the tab attaches to. Plane-Y top, not Z. |
| **`tab_position`** | 0.5 | 0..1 along the chosen side. 0.5 = centered, 0.0 = start, 1.0 = end. |
| **`tab_width_mm`** | 12.0 | Tab span along the silhouette edge. |
| **`tab_depth_mm`** | 8.0 | How far the tab extends outward. |
| **`tab_corner_radius_mm`** | 2.5 | Rounded corners on the tab. |
| **`tab_overlap_mm`** | 2.0 | How far the tab reaches into the silhouette so the union is one solid piece. Bump this if the silhouette edge is wispy (hair spikes etc.) and the tab looks disconnected. |
| `hole_type` | `round` | `round` / `double` / `slot` / `none`. |
| `hole_diameter` | 4.0 | For `round` / `double`. |
| `hole_edge_margin` | 3.0 | Min distance from hole edge to tab outer edge. |
| `hole_position` | `top-center` | Legacy, only used when `tab_enabled=false`. |
| **`text_enabled`** | `false` | Add a wording plate below the image. Or just pass `--text "NAME"`. |
| **`text_string`** | `NAME` | The wording. Use `\n` (or Enter in the UI) for multiple lines. |
| **`text_font`** | `""` | `.ttf` path or family name; blank = DejaVu Sans Bold (Arial Bold on Windows). `disney` works if you add `waltographUI.ttf` to `./fonts` locally. |
| **`text_height_mm`** | 8.0 | Letter height. |
| **`text_thickness_mm`** | 1.0 | How far raised letters rise above the plate. |
| `text_plate_thickness_mm` | base | Plate thickness; blank = `base_thickness` so it welds to the keychain. |
| `text_margin_mm` | 1.5 | Gap between the image bottom and the plate top (negative = overlap). |
| `text_recessed` | `false` | `true` = engrave letters into the plate instead of raising them. |
| `text_color` / `text_plate_color` | dark / light | Filament-color hints for the two bodies. |

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
