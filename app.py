"""Gradio UI for image_to_keychain.

Run:
    python3 app.py

Then open http://localhost:7860 in a browser (or use the network URL it
prints if you want to reach it from Windows while running in WSL).

Flow:
    1. Drag your PNG into the upload zone (anywhere on disk — no folder
       dance).
    2. Tweak sliders; the 2D preview updates live on slider release.
    3. Click "Generate 3MF + STL" to run the full pipeline and download
       the output files.
"""
from __future__ import annotations

import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

from pipeline.export_3mf import export_3mf
from pipeline.export_stl import export_stl_parts
from pipeline.extract_colors import extract_colors
from pipeline.extract_lines import extract_lines
from pipeline.extrude import apply_transform, build_parts, compute_px_to_mm
from pipeline.keyhole import build_tab_and_hole
from pipeline.preprocess import preprocess
from pipeline.render_preview import render_topdown
from pipeline.silhouette import build_silhouette
from pipeline.text_label import build_text_label


HERE = Path(__file__).parent.resolve()
INTER_ROOT = HERE / "intermediate_ui"
OUT_ROOT = HERE / "out_ui"
INTER_ROOT.mkdir(exist_ok=True)
OUT_ROOT.mkdir(exist_ok=True)


# ---- image-keyed cache so slider drags stay snappy -------------------------

_state_cache: dict[str, dict] = {}


def _image_cache_key(path: Path, max_colors: int, line_dilate: int) -> str:
    h = hashlib.md5(path.read_bytes()).hexdigest()
    return f"{h}|{max_colors}|{line_dilate}"


def _image_dependent_state(image_path: Path, max_colors: int, line_dilate: int) -> dict:
    """Runs the expensive image-dependent pipeline stages and caches the
    result. These depend on image content + max_colors + line_dilate only —
    not on tab/hole params."""
    key = _image_cache_key(image_path, max_colors, line_dilate)
    if key in _state_cache:
        return _state_cache[key]

    inter_dir = INTER_ROOT / key[:12]
    inter_dir.mkdir(exist_ok=True)
    cfg_basic = {
        "max_colors": max_colors,
        "line_dilate_px": line_dilate,
        "alpha_threshold": 128,
        "corner_patch_px": 24,
        "corner_bg_ratio": 0.10,
        "kmeans_seed": 0,
        "kmeans_sample": 20000,
        "kmeans_iters": 15,
        "line_include_silhouette_edge": True,
        "verbose": False,
    }
    pre = preprocess(image_path, cfg_basic)
    lines = extract_lines(image_path, inter_dir, cfg_basic)
    sil = build_silhouette(pre, inter_dir, verbose=False)
    colors = extract_colors(pre, inter_dir, verbose=False)
    state = {"pre": pre, "lines": lines, "sil": sil, "colors": colors,
             "inter_dir": inter_dir, "cfg_basic": cfg_basic}
    _state_cache[key] = state
    return state


# ---- preview + generate ----------------------------------------------------

def _collect_cfg(target_size_mm, base_t, line_t, color_t, max_colors, line_dilate,
                 tab_enabled, tab_side, tab_position, tab_width, tab_depth,
                 tab_corner_r, tab_overlap,
                 hole_type, hole_diameter, hole_edge_margin,
                 hole_spacing, hole_slot_length, hole_slot_width,
                 text_enabled, text_string, text_font_file, text_font_name,
                 text_height, text_thickness, text_recessed, text_margin,
                 text_letter_spacing):
    return {
        "target_size_mm": float(target_size_mm),
        "base_thickness": float(base_t),
        "line_thickness": float(line_t),
        "color_thickness": float(color_t),
        "max_colors": int(max_colors),
        "line_dilate_px": int(line_dilate),
        "alpha_threshold": 128,
        "corner_patch_px": 24,
        "corner_bg_ratio": 0.10,
        "kmeans_seed": 0,
        "kmeans_sample": 20000,
        "kmeans_iters": 15,
        "line_include_silhouette_edge": True,
        "tab_enabled": bool(tab_enabled),
        "tab_side": str(tab_side),
        "tab_position": float(tab_position),
        "tab_width_mm": float(tab_width),
        "tab_depth_mm": float(tab_depth),
        "tab_corner_radius_mm": float(tab_corner_r),
        "tab_overlap_mm": float(tab_overlap),
        "hole_type": str(hole_type),
        "hole_diameter": float(hole_diameter),
        "hole_edge_margin": float(hole_edge_margin),
        "hole_spacing": float(hole_spacing),
        "hole_slot_length": float(hole_slot_length),
        "hole_slot_width": float(hole_slot_width),
        "hole_position": "top-center",
        "hole_custom_offset": [0, 0],
        # Wording plate. An uploaded .ttf (text_font_file) wins over a typed
        # font name; empty falls back to Arial Bold inside the pipeline.
        "text_enabled": bool(text_enabled),
        "text_string": str(text_string or ""),
        "text_font": (text_font_file or text_font_name or ""),
        "text_height_mm": float(text_height),
        "text_thickness_mm": float(text_thickness),
        "text_recessed": bool(text_recessed),
        "text_margin_mm": float(text_margin),
        "text_letter_spacing": float(text_letter_spacing),
        "text_padding_mm": 3.0,
        "text_corner_radius_mm": 2.0,
        "verbose": False,
    }


def _build_text_label(cfg, xform, th, inter_dir):
    """Build the wording plate, tolerating font/render errors so a bad font
    name never kills the preview or the export. Returns (label, error_str)."""
    if not cfg.get("text_enabled"):
        return None, None
    bottom_y = xform["bounds_mm"][0][1]
    if not th.tab.is_empty:
        bottom_y = min(bottom_y, th.tab.bounds[1])
    try:
        label = build_text_label(xform["bounds_mm"], cfg, inter_dir,
                                 bottom_y=bottom_y, verbose=False)
        return label, None
    except Exception as e:
        return None, f"text failed: {e}"


def preview_fn(image_path, *args):
    if image_path is None:
        return None, "Drop an image to start."
    img_path = Path(image_path)

    cfg = _collect_cfg(*args)
    try:
        state = _image_dependent_state(img_path, cfg["max_colors"], cfg["line_dilate_px"])
    except Exception as e:
        return None, f"Preprocess failed: {e}"

    sil = state["sil"]
    lines = state["lines"]
    colors = state["colors"]

    xform = compute_px_to_mm(sil.image_shape, cfg["target_size_mm"], sil.polygon)
    th = build_tab_and_hole(xform["bounds_mm"], cfg, verbose=False)

    sil_mm = apply_transform(sil.polygon, xform)
    lines_mm = apply_transform(lines.polygon, xform) if not lines.polygon.is_empty else MultiPolygon()
    color_parts = [(apply_transform(cp.polygon, xform), cp.rgb) for cp in colors]

    text_label, text_err = _build_text_label(cfg, xform, th, state["inter_dir"])

    # Canvas bounds = silhouette + tab + wording plate combined
    pieces = [sil_mm]
    if not th.tab.is_empty:
        pieces.append(th.tab)
    if text_label is not None and not text_label.is_empty:
        pieces.append(text_label.plate)
    bb = unary_union(pieces).bounds
    bounds_mm = ((bb[0], bb[1]), (bb[2], bb[3]))

    img = render_topdown(sil_mm, lines_mm, color_parts, th.tab, th.hole,
                         bounds_mm, ppmm=10, text_label=text_label)
    status = (
        f"Keychain subject: {xform['bounds_mm'][1][0]:.1f} \u00d7 "
        f"{xform['bounds_mm'][1][1]:.1f} mm"
        f"  \u2192  {len(colors)} color parts + lines + base"
        + (f" + tab" if not th.tab.is_empty else "")
        + (f" + text" if (text_label is not None and not text_label.is_empty) else "")
        + (f"  \u26a0 {text_err}" if text_err else "")
    )
    return img, status


def generate_fn(image_path, *args, progress=gr.Progress()):
    if image_path is None:
        return None, None, "Drop an image first."
    img_path = Path(image_path)
    cfg = _collect_cfg(*args)

    progress(0.1, desc="Preprocessing...")
    state = _image_dependent_state(img_path, cfg["max_colors"], cfg["line_dilate_px"])

    progress(0.4, desc="Building tab + hole...")
    sil = state["sil"]
    xform = compute_px_to_mm(sil.image_shape, cfg["target_size_mm"], sil.polygon)
    th = build_tab_and_hole(xform["bounds_mm"], cfg, verbose=False)
    text_label, text_err = _build_text_label(cfg, xform, th, state["inter_dir"])

    progress(0.5, desc="Extruding...")
    parts = build_parts(
        silhouette_mp=sil.polygon,
        lines_mp=state["lines"].polygon,
        colors=state["colors"],
        hole_mp=th.hole,
        tab_mp=th.tab,
        text_label=text_label,
        image_shape=sil.image_shape,
        cfg=cfg,
        verbose=False,
    )

    progress(0.8, desc="Exporting...")
    stem = img_path.stem or "keychain"
    # Per-run output folder to avoid collisions between concurrent clicks
    run_dir = OUT_ROOT / f"{stem}_{hashlib.md5(str(img_path).encode()).hexdigest()[:6]}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    mf_path = export_3mf(parts, run_dir, stem, verbose=False)
    parts_dir = export_stl_parts(parts, run_dir, stem, verbose=False)

    # Zip the STL parts for a single-file download
    zip_path = run_dir / f"{stem}_parts.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for stl in sorted(parts_dir.glob("*.stl")):
            z.write(stl, arcname=stl.name)

    watertight = sum(1 for p in parts if p.mesh and p.mesh.is_watertight)
    status = (
        f"\u2713 Generated {len(parts)} parts, {watertight}/{len(parts)} watertight. "
        f"3MF ready to drop into Bambu Studio."
        + (f"  \u26a0 {text_err}" if text_err else "")
    )
    return str(mf_path), str(zip_path), status


# ---- UI layout -------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="image_to_keychain") as demo:
        gr.Markdown("# image_to_keychain  \nDrag a chibi PNG, tweak, download a Bambu-ready 3MF.")

        with gr.Row():
            with gr.Column(scale=1, min_width=340):
                image_input = gr.Image(label="Input image", type="filepath",
                                        sources=["upload", "clipboard"])

                with gr.Accordion("Size & thicknesses", open=True):
                    target_size_mm = gr.Slider(20, 120, value=60, step=1,
                                               label="target_size_mm (longest subject dim)")
                    base_t = gr.Slider(0.8, 5.0, value=2.0, step=0.1, label="base_thickness (mm)")
                    line_t = gr.Slider(0.4, 4.0, value=1.5, step=0.1, label="line_thickness (mm)")
                    color_t = gr.Slider(0.4, 3.0, value=1.0, step=0.1,
                                         label="color_thickness (mm)")

                with gr.Accordion("Colors & lines", open=True):
                    max_colors = gr.Slider(3, 10, value=6, step=1, label="max_colors")
                    line_dilate = gr.Slider(0, 8, value=2, step=1, label="line_dilate_px")

                with gr.Accordion("Tab (keyring holder)", open=True):
                    tab_enabled = gr.Checkbox(value=True, label="tab_enabled")
                    tab_side = gr.Dropdown(["top", "bottom", "left", "right"],
                                            value="top", label="tab_side")
                    tab_position = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                              label="tab_position (0..1 along side)")
                    tab_width = gr.Slider(4.0, 30.0, value=12.0, step=0.5,
                                           label="tab_width_mm")
                    tab_depth = gr.Slider(4.0, 20.0, value=8.0, step=0.5,
                                           label="tab_depth_mm")
                    tab_corner_r = gr.Slider(0.0, 6.0, value=2.5, step=0.5,
                                              label="tab_corner_radius_mm")
                    tab_overlap = gr.Slider(0.0, 8.0, value=2.0, step=0.5,
                                             label="tab_overlap_mm")

                with gr.Accordion("Hole", open=True):
                    hole_type = gr.Dropdown(["round", "double", "slot", "none"],
                                             value="round", label="hole_type")
                    hole_diameter = gr.Slider(2.0, 10.0, value=4.0, step=0.1,
                                               label="hole_diameter (mm)")
                    hole_edge_margin = gr.Slider(1.0, 8.0, value=3.0, step=0.1,
                                                   label="hole_edge_margin (mm)")
                    hole_spacing = gr.Slider(2.0, 15.0, value=6.0, step=0.5,
                                              label="hole_spacing (double only)")
                    hole_slot_length = gr.Slider(4.0, 20.0, value=8.0, step=0.5,
                                                   label="hole_slot_length (slot)")
                    hole_slot_width = gr.Slider(2.0, 10.0, value=4.0, step=0.5,
                                                  label="hole_slot_width (slot)")

                with gr.Accordion("Wording plate (below image)", open=False):
                    text_enabled = gr.Checkbox(value=False, label="text_enabled")
                    text_string = gr.Textbox(value="NAME", label="text (use Enter for 2nd line)",
                                              lines=1)
                    text_font_name = gr.Textbox(
                        value="", label="font (.ttf path or family; blank = bold sans default)",
                        placeholder="e.g. arialbd.ttf  — or leave blank")
                    text_font_file = gr.File(
                        label="...or upload your own .ttf/.otf (e.g. a Disney-style font)",
                        file_types=[".ttf", ".otf"], type="filepath")
                    text_height = gr.Slider(3.0, 20.0, value=8.0, step=0.5,
                                            label="text_height_mm (letter height)")
                    text_thickness = gr.Slider(0.4, 3.0, value=1.0, step=0.1,
                                               label="text_thickness_mm (raise above plate)")
                    text_recessed = gr.Checkbox(value=False,
                                                label="text_recessed (engrave instead of raise)")
                    text_margin = gr.Slider(-2.0, 8.0, value=1.5, step=0.5,
                                            label="text_margin_mm (gap below image)")
                    text_letter_spacing = gr.Slider(0.0, 0.5, value=0.0, step=0.02,
                                                    label="text_letter_spacing (tracking)")

                generate_btn = gr.Button("\u2605  Generate 3MF + STL", variant="primary", size="lg")

            with gr.Column(scale=1, min_width=360):
                preview = gr.Image(label="Preview (top-down, live)", type="pil",
                                    interactive=False)
                status = gr.Markdown("Upload an image to begin.")
                file_3mf = gr.File(label="Download 3MF (Bambu Studio)")
                file_zip = gr.File(label="Download STL parts (.zip)")

        controls = [
            target_size_mm, base_t, line_t, color_t, max_colors, line_dilate,
            tab_enabled, tab_side, tab_position, tab_width, tab_depth,
            tab_corner_r, tab_overlap,
            hole_type, hole_diameter, hole_edge_margin,
            hole_spacing, hole_slot_length, hole_slot_width,
            text_enabled, text_string, text_font_file, text_font_name,
            text_height, text_thickness, text_recessed, text_margin,
            text_letter_spacing,
        ]

        # Any slider release OR image change triggers preview
        triggers = [c.release if isinstance(c, gr.Slider) else c.change for c in controls]
        triggers.append(image_input.change)
        for trig in triggers:
            trig(preview_fn, inputs=[image_input] + controls, outputs=[preview, status])

        generate_btn.click(generate_fn, inputs=[image_input] + controls,
                           outputs=[file_3mf, file_zip, status])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=False, share=False,
                theme=gr.themes.Soft())
