"""image_to_keychain — CLI entry point.

Usage:
    python image_to_keychain.py --config config.yaml
    python image_to_keychain.py --input cat.png --output-basename cat --line-thickness 2.0

Pipeline stages:
    1. vectorize          — raster -> color SVG (debug)
    2. extract_lines      — color-boundary line polygon
    3. extract_colors     — per-cluster color polygons
    4. silhouette         — outer boundary polygon
    5 + 6. extrude        — px -> mm transform, layered extrusion
    7. keyhole            — 2D hole polygon subtracted from all layers
    export_stl / export_3mf
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import yaml

from pipeline.extract_colors import extract_colors
from pipeline.extract_lines import extract_lines
from pipeline.export_3mf import export_3mf
from pipeline.export_stl import export_stl_parts
from pipeline.extrude import build_parts, compute_px_to_mm
from pipeline.keyhole import build_keyhole
from pipeline.preprocess import preprocess
from pipeline.silhouette import build_silhouette
from pipeline.util import describe_config, ensure_dir, get_logger
from pipeline.vectorize import vectorize


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _merge_cli_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(cfg)
    for k, v in overrides.items():
        if v is not None:
            merged[k] = v
    return merged


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
              default=None, help="YAML config file. CLI flags override its values.")
@click.option("--input", "input_image", type=click.Path(exists=True, path_type=Path),
              default=None, help="Input PNG/JPG.")
@click.option("--output-basename", type=str, default=None)
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--target-size-mm", type=float, default=None)
@click.option("--base-thickness", type=float, default=None)
@click.option("--line-thickness", type=float, default=None)
@click.option("--color-thickness", type=float, default=None)
@click.option("--line-dilate-px", type=int, default=None)
@click.option("--max-colors", type=int, default=None)
@click.option("--hole-type", type=click.Choice(["round", "double", "slot", "none"]), default=None)
@click.option("--hole-diameter", type=float, default=None)
@click.option("--hole-position", type=str, default=None)
@click.option("--stop-after", type=click.Choice(["vectorize", "lines", "colors", "silhouette",
                                                 "extrude", "hole", "export"]),
              default=None, help="Stop pipeline after this stage (for debugging).")
@click.option("-v", "--verbose", is_flag=True, default=None)
def main(config_path: Path | None, input_image: Path | None, output_basename: str | None,
         output_dir: Path | None, target_size_mm: float | None, base_thickness: float | None,
         line_thickness: float | None, color_thickness: float | None,
         line_dilate_px: int | None, max_colors: int | None,
         hole_type: str | None, hole_diameter: float | None, hole_position: str | None,
         stop_after: str | None, verbose: bool | None) -> None:

    script_dir = Path(__file__).parent.resolve()
    cfg_path = config_path or (script_dir / "config.yaml")
    if not cfg_path.exists():
        raise click.UsageError(f"Config file not found: {cfg_path}")
    cfg = _load_yaml(cfg_path)

    # Resolve relative input_image path relative to the config file
    cfg_dir = cfg_path.parent.resolve()
    if isinstance(cfg.get("input_image"), str):
        p = Path(cfg["input_image"])
        if not p.is_absolute():
            p = (cfg_dir / p).resolve()
        cfg["input_image"] = p

    overrides = {
        "input_image": input_image, "output_basename": output_basename,
        "output_dir": output_dir, "target_size_mm": target_size_mm,
        "base_thickness": base_thickness, "line_thickness": line_thickness,
        "color_thickness": color_thickness,
        "line_dilate_px": line_dilate_px, "max_colors": max_colors,
        "hole_type": hole_type, "hole_diameter": hole_diameter,
        "hole_position": hole_position, "verbose": verbose,
    }
    cfg = _merge_cli_overrides(cfg, overrides)

    log = get_logger(verbose=bool(cfg.get("verbose", True)))
    log.info("image_to_keychain pipeline starting")

    input_path = Path(cfg["input_image"])
    if not input_path.exists():
        raise click.UsageError(f"Input image not found: {input_path}")

    # Resolve output_dir relative to the script dir if not absolute
    od = Path(cfg.get("output_dir", "out"))
    if not od.is_absolute():
        od = (script_dir / od).resolve()
    out_dir = ensure_dir(od)
    intermediate_dir = ensure_dir(script_dir / "intermediate")
    basename = str(cfg.get("output_basename", "keychain"))
    formats = [f.lower() for f in cfg.get("output_formats", ["3mf", "stl"])]

    # --- Step 1: vectorize (debug SVG) -------------------------------------
    vec = vectorize(input_path, intermediate_dir, cfg)
    log.info("Step 1 done: %s (%dx%d px, %d clusters)",
             vec.color_svg_path.name, vec.image_shape[1], vec.image_shape[0], vec.color_count)
    if stop_after == "vectorize":
        return

    # --- Step 2: lines ------------------------------------------------------
    lines = extract_lines(input_path, intermediate_dir, cfg)
    log.info("Step 2 done: %d line px, %d subpolygons",
             lines.line_pixel_count, len(lines.polygon.geoms))
    if stop_after == "lines":
        return

    # Shared preprocess (already cached from vectorize/lines calls)
    pre = preprocess(input_path, cfg)

    # --- Step 3: colors -----------------------------------------------------
    colors = extract_colors(pre, intermediate_dir, verbose=bool(cfg.get("verbose", True)))
    log.info("Step 3 done: %d color parts", len(colors))
    if stop_after == "colors":
        return

    # --- Step 4: silhouette -------------------------------------------------
    sil = build_silhouette(pre, intermediate_dir, verbose=bool(cfg.get("verbose", True)))
    log.info("Step 4 done: silhouette %d subpolygons", len(sil.polygon.geoms))
    if stop_after == "silhouette":
        return

    # --- Step 7 prep: hole polygon in mm -----------------------------------
    xform = compute_px_to_mm(sil.image_shape, float(cfg["target_size_mm"]), sil.polygon)
    hole_mp = build_keyhole(xform["bounds_mm"], cfg, verbose=bool(cfg.get("verbose", True)))
    if stop_after == "hole":
        return

    # --- Steps 5 + 6: extrude ----------------------------------------------
    parts = build_parts(
        silhouette_mp=sil.polygon,
        lines_mp=lines.polygon,
        colors=colors,
        hole_mp=hole_mp,
        image_shape=sil.image_shape,
        cfg=cfg,
        verbose=bool(cfg.get("verbose", True)),
    )
    log.info("Steps 5-6 done: %d parts extruded", len(parts))
    if stop_after == "extrude":
        return

    # --- Export -------------------------------------------------------------
    if "stl" in formats:
        export_stl_parts(parts, out_dir, basename,
                         verbose=bool(cfg.get("verbose", True)))
    if "3mf" in formats:
        export_3mf(parts, out_dir, basename,
                   verbose=bool(cfg.get("verbose", True)))

    log.info("\u2713 pipeline complete")
    log.info("  output dir: %s", out_dir)
    log.info("  basename:   %s", basename)


if __name__ == "__main__":
    main()
