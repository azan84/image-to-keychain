"""Step 2 — extract line art from *color boundaries* (not dark pixels).

Approach:
    1. Use the shared preprocess step to get a (H, W) int label map where
       similar tones are clustered into the same label. Soft shading is
       already merged away by k-means, so it does NOT generate a line.
    2. A pixel is a line pixel iff any of its 4-neighbors has a different
       label (including the transition from opaque → transparent background,
       which gives the outer silhouette line).
    3. Optionally dilate the line mask by `line_dilate_px` — at print scale
       raw 1–2 px boundaries are too thin to print reliably.
    4. Emit bi-level mask + vtracer binary SVG + red-overlay preview.

The line color isn't relevant to extraction — lines are boundaries, not
dark pixels. For rendering the 3D part later we'll use a fixed "line color"
(default black) that the user can assign to any AMS filament.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import vtracer
from PIL import Image, ImageFilter
from shapely.geometry import MultiPolygon

from .preprocess import preprocess
from .svg_polygons import svg_to_multipolygon
from .util import get_logger, save_png


@dataclass
class LineExtractionResult:
    line_svg_path: Path
    line_mask_png_path: Path
    preview_png_path: Path
    line_pixel_count: int
    image_shape: tuple[int, int]
    polygon: MultiPolygon          # in image-pixel coords, Y-flipped


def _boundary_mask(labels: np.ndarray) -> np.ndarray:
    """True on both sides of every label transition (4-connectivity)."""
    H, W = labels.shape
    m = np.zeros((H, W), dtype=bool)
    diff_v = labels[:-1, :] != labels[1:, :]
    m[:-1, :] |= diff_v
    m[1:, :] |= diff_v
    diff_h = labels[:, :-1] != labels[:, 1:]
    m[:, :-1] |= diff_h
    m[:, 1:] |= diff_h
    return m


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.filter(ImageFilter.MaxFilter(size=2 * px + 1))
    return np.array(img) > 127


def _build_preview(rgba: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float32)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    base = np.stack([gray, gray, gray], axis=-1) * 0.6 + 0.4 * 255
    alpha = rgba[..., 3:4] / 255.0
    base = base * alpha + 240 * (1 - alpha)
    base[mask] = np.array([230, 50, 50], dtype=np.float32)
    return base.clip(0, 255).astype(np.uint8)


def extract_lines(input_image: Path, intermediate_dir: Path,
                  cfg: dict[str, Any]) -> LineExtractionResult:
    """Step 2 — color-boundary line extraction."""
    log = get_logger(verbose=cfg.get("verbose", True))
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    log.info("Step 2: extracting line art from color boundaries")
    pre = preprocess(input_image, cfg)
    H, W = pre.shape

    include_silhouette = bool(cfg.get("line_include_silhouette_edge", True))
    dilate_px = int(cfg.get("line_dilate_px", 2))
    log.info("  include silhouette edge: %s  dilate: %dpx", include_silhouette, dilate_px)

    mask = _boundary_mask(pre.labels)

    if not include_silhouette:
        # Drop transitions that touch the background.
        not_bg_pair = np.ones_like(mask)
        lab = pre.labels
        not_bg_pair[:-1, :] &= ~((lab[:-1, :] == -1) | (lab[1:, :] == -1))
        not_bg_pair[1:, :]  &= ~((lab[:-1, :] == -1) | (lab[1:, :] == -1))
        not_bg_pair[:, :-1] &= ~((lab[:, :-1] == -1) | (lab[:, 1:] == -1))
        not_bg_pair[:, 1:]  &= ~((lab[:, :-1] == -1) | (lab[:, 1:] == -1))
        mask &= not_bg_pair

    # Lines live on the *opaque* side only — no line extends into transparent space.
    mask &= ~pre.bg_mask

    raw_count = int(mask.sum())
    if dilate_px > 0:
        mask = _dilate(mask, dilate_px)
        mask &= ~pre.bg_mask  # dilation may spill into bg — re-clip
    final_count = int(mask.sum())
    log.info("  line pixels: %d raw \u2192 %d after dilation (%.2f%% of image)",
             raw_count, final_count, 100.0 * final_count / (H * W))

    if final_count == 0:
        raise RuntimeError(
            "No color-boundary line pixels found. Image may be a single flat "
            "color; lower max_colors or check the input."
        )

    # vtracer binary mode traces dark pixels, so lines must be BLACK on WHITE.
    mask_png = intermediate_dir / "03_line_mask.png"
    mask_img = np.full((H, W), 255, dtype=np.uint8)
    mask_img[mask] = 0
    save_png(mask_img, mask_png)

    lines_svg = intermediate_dir / "04_lines.svg"
    log.info("  vectorizing line mask (binary, polygon mode) \u2192 %s", lines_svg.name)
    vtracer.convert_image_to_svg_py(
        str(mask_png),
        str(lines_svg),
        colormode="binary",
        hierarchical="stacked",
        mode="polygon",
        filter_speckle=2,
        color_precision=1,
        layer_difference=0,
        corner_threshold=60,
        length_threshold=3.0,
        max_iterations=10,
        splice_threshold=45,
        path_precision=3,
    )

    line_poly = svg_to_multipolygon(lines_svg, flip_y_height=float(H), min_area=4.0)
    log.info("  line polygon: %d subpolygon(s), bounds %s",
             len(line_poly.geoms), line_poly.bounds)

    preview_png = intermediate_dir / "05_lines_preview.png"
    save_png(_build_preview(pre.rgba, mask), preview_png)
    log.info("  wrote preview overlay \u2192 %s", preview_png.name)

    return LineExtractionResult(
        line_svg_path=lines_svg,
        line_mask_png_path=mask_png,
        preview_png_path=preview_png,
        line_pixel_count=final_count,
        image_shape=(H, W),
        polygon=line_poly,
    )
