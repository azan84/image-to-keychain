"""Step 2 — extract the black line-art as closed filled shapes.

This is the most fragile step of the pipeline. The goal is to produce an SVG
whose paths are *filled closed regions* (not zero-width strokes), so they can
be extruded as solid 3D parts in later steps.

Approach:
    1. Build a binary line mask from the RGBA image:
         line_pixel := brightness < line_color_threshold AND alpha >= alpha_threshold
       The brightness threshold handles anti-aliased edges — we don't match
       pure black, we match "dark enough."
    2. Optionally dilate the mask by `line_dilate_px` pixels. Anime line art
       often vectorizes too thin; a small dilation restores visual weight
       without needing to fiddle with 3D offsets later.
    3. Save the mask as a bi-level PNG.
    4. Vectorize it with vtracer in binary mode — vtracer emits filled closed
       paths from binary input, which is exactly what we need for extrusion.
    5. Emit a preview PNG overlaying the extracted lines on the original
       (red = extracted, grey = background) so the user can verify quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import vtracer
from PIL import Image, ImageFilter

from .util import brightness, get_logger, load_image_rgba, save_png


@dataclass
class LineExtractionResult:
    line_svg_path: Path
    line_mask_png_path: Path
    preview_png_path: Path
    line_pixel_count: int
    image_shape: tuple[int, int]  # (H, W)


def _build_line_mask(rgba: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Return an (H, W) bool array where True = line-art pixel."""
    alpha_cut = int(cfg.get("alpha_threshold", 128))
    line_thresh = int(cfg.get("line_color_threshold", 70))
    alpha = rgba[..., 3]
    rgb = rgba[..., :3]
    bright = brightness(rgb)
    mask = (bright < line_thresh) & (alpha >= alpha_cut)
    return mask


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Morphological dilation by `px` pixels via PIL MaxFilter (size = 2*px+1)."""
    if px <= 0:
        return mask
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    # MaxFilter size must be odd
    size = 2 * px + 1
    img = img.filter(ImageFilter.MaxFilter(size=size))
    return np.array(img) > 127


def _build_preview(rgba: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay extracted lines in red on a desaturated copy of the original."""
    rgb = rgba[..., :3].astype(np.float32)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    base = np.stack([gray, gray, gray], axis=-1) * 0.6 + 0.4 * 255
    alpha = (rgba[..., 3:4] / 255.0)
    base = base * alpha + 240 * (1 - alpha)  # light grey behind transparent
    base[mask] = np.array([230, 50, 50], dtype=np.float32)
    return base.clip(0, 255).astype(np.uint8)


def extract_lines(input_image: Path, intermediate_dir: Path,
                  cfg: dict[str, Any]) -> LineExtractionResult:
    """Run Step 2. Produces:
        intermediate/03_line_mask.png     — bi-level mask
        intermediate/04_lines.svg         — vtracer output (closed filled paths)
        intermediate/05_lines_preview.png — overlay for human review
    """
    log = get_logger(verbose=cfg.get("verbose", True))
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    log.info("Step 2: extracting line art from %s", input_image.name)
    rgba = load_image_rgba(input_image)
    H, W, _ = rgba.shape

    line_thresh = int(cfg.get("line_color_threshold", 70))
    dilate_px = int(cfg.get("line_dilate_px", 0))
    log.info("  brightness threshold: <%d  alpha \u2265 %d  dilate: %dpx",
             line_thresh, cfg.get("alpha_threshold", 128), dilate_px)

    mask = _build_line_mask(rgba, cfg)
    raw_count = int(mask.sum())
    if dilate_px > 0:
        mask = _dilate(mask, dilate_px)
    final_count = int(mask.sum())
    log.info("  line pixels: %d raw \u2192 %d after dilation  (%.2f%% of image)",
             raw_count, final_count, 100.0 * final_count / (H * W))

    if final_count == 0:
        raise RuntimeError(
            "No line-art pixels found. Raise line_color_threshold or check the "
            "input image — it may not have dark outlines."
        )

    # Emit a bi-level PNG: white = line, black = background.
    # vtracer in binary mode traces the foreground (non-black) pixels, so we
    # write lines as white on a black background.
    mask_png = intermediate_dir / "03_line_mask.png"
    mask_img = np.zeros((H, W), dtype=np.uint8)
    mask_img[mask] = 255
    save_png(mask_img, mask_png)
    log.debug("  wrote %s", mask_png)

    lines_svg = intermediate_dir / "04_lines.svg"
    log.info("  vectorizing line mask (binary mode) \u2192 %s", lines_svg.name)
    vtracer.convert_image_to_svg_py(
        str(mask_png),
        str(lines_svg),
        colormode="binary",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=2,         # keep fine detail — line art often has thin strokes
        color_precision=1,
        layer_difference=0,
        corner_threshold=60,
        length_threshold=3.0,
        max_iterations=10,
        splice_threshold=45,
        path_precision=3,
    )

    preview_png = intermediate_dir / "05_lines_preview.png"
    save_png(_build_preview(rgba, mask), preview_png)
    log.info("  wrote preview overlay \u2192 %s", preview_png.name)

    return LineExtractionResult(
        line_svg_path=lines_svg,
        line_mask_png_path=mask_png,
        preview_png_path=preview_png,
        line_pixel_count=final_count,
        image_shape=(H, W),
    )
