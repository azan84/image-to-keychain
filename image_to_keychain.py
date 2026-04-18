"""image_to_keychain — CLI entry point.

Usage:
    python image_to_keychain.py --config config.yaml
    python image_to_keychain.py --input cat.png --output-basename cat --line-thickness 2.0

Currently implements Steps 1 (vectorize) and 2 (extract lines). Later steps
(color extraction, silhouette, extrusion, keychain hole, 3MF/STL export)
will be wired in as they are built.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import yaml

from pipeline.extract_lines import extract_lines
from pipeline.util import describe_config, ensure_dir, get_logger
from pipeline.vectorize import vectorize


# ---- config merge ----------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _merge_cli_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(cfg)
    for k, v in overrides.items():
        if v is not None:
            merged[k] = v
    return merged


# ---- CLI -------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
              default=None, help="YAML config file. CLI flags override its values.")
@click.option("--input", "input_image", type=click.Path(exists=True, path_type=Path),
              default=None, help="Input PNG/JPG.")
@click.option("--output-basename", type=str, default=None, help="Output filename stem.")
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
              default=None, help="Stop the pipeline after this stage (for debugging).")
@click.option("-v", "--verbose", is_flag=True, default=None)
def main(config_path: Path | None, input_image: Path | None, output_basename: str | None,
         output_dir: Path | None, target_size_mm: float | None, base_thickness: float | None,
         line_thickness: float | None, color_thickness: float | None,
         line_dilate_px: int | None,
         max_colors: int | None, hole_type: str | None, hole_diameter: float | None,
         hole_position: str | None, stop_after: str | None, verbose: bool | None) -> None:

    script_dir = Path(__file__).parent.resolve()
    cfg_path = config_path or (script_dir / "config.yaml")
    if not cfg_path.exists():
        raise click.UsageError(f"Config file not found: {cfg_path}")
    cfg = _load_yaml(cfg_path)

    # Resolve relative paths in the config file relative to its directory
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
    log.debug("resolved config:\n%s", describe_config({
        k: v for k, v in cfg.items() if k != "input_image"
    } | {"input_image": str(cfg["input_image"])}))

    input_path = Path(cfg["input_image"])
    if not input_path.exists():
        raise click.UsageError(f"Input image not found: {input_path}")

    out_dir = ensure_dir(Path(cfg.get("output_dir", script_dir / "out")))
    intermediate_dir = ensure_dir(script_dir / "intermediate")

    # --- Step 1: vectorize --------------------------------------------------
    vec = vectorize(input_path, intermediate_dir, cfg)
    log.info("Step 1 done: %s (image %dx%d, %d colors)",
             vec.color_svg_path.name, vec.image_shape[1], vec.image_shape[0], vec.color_count)
    if stop_after == "vectorize":
        return

    # --- Step 2: extract lines ---------------------------------------------
    lines = extract_lines(input_path, intermediate_dir, cfg)
    log.info("Step 2 done: %s (%d line pixels)",
             lines.line_svg_path.name, lines.line_pixel_count)
    if stop_after in (None, "lines"):
        log.info("\u2713 Stopped after Step 2 as requested. Inspect:")
        log.info("    %s", lines.line_mask_png_path)
        log.info("    %s", lines.line_svg_path)
        log.info("    %s", lines.preview_png_path)
        return

    # Later steps will be added here.
    log.warning("Stages beyond Step 2 are not yet implemented.")


if __name__ == "__main__":
    main()
