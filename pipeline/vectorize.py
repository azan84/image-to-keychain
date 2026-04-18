"""Step 1 — vectorize the input raster into colored SVG paths.

Approach (raster-first):
    1. Load the image as RGBA.
    2. Alpha-composite against white so vtracer sees a clean opaque image.
    3. Optionally k-means quantize to `max_colors` to prevent near-duplicate
       regions (AMS has 4 slots; 20+ colors is unusable).
    4. Mask out the line-art pixels (brightness < threshold) by replacing
       them with a sentinel background color, so colors come out clean.
    5. Run vtracer to produce a color SVG of the interior regions only.

The line art itself is handled separately in extract_lines.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import vtracer
from PIL import Image

from .util import brightness, get_logger, load_image_rgba, save_png


SENTINEL_BG = (255, 255, 255)  # replaces background + line pixels prior to vectorization


@dataclass
class VectorizeResult:
    color_svg_path: Path
    preprocessed_png_path: Path
    image_shape: tuple[int, int]  # (H, W)
    color_count: int              # colors actually kept after quantization


def _kmeans_quantize(rgb: np.ndarray, mask: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Cluster pixels in `rgb` where mask==True to at most k colors. Returns
    a new rgb array with those pixels replaced by their cluster centroid."""
    pixels = rgb[mask].astype(np.float32)
    if len(pixels) == 0 or k <= 1:
        return rgb
    rng = np.random.default_rng(seed)
    # Lightweight Lloyd's algorithm — dependency-free, sufficient for <=8 clusters
    idx = rng.choice(len(pixels), size=min(k, len(pixels)), replace=False)
    centers = pixels[idx].copy()
    for _ in range(12):
        # Assign
        d = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
        labels = d.argmin(axis=1)
        new_centers = np.stack([
            pixels[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
            for j in range(len(centers))
        ])
        if np.allclose(new_centers, centers, atol=0.5):
            centers = new_centers
            break
        centers = new_centers
    out = rgb.copy()
    quantized = centers[labels].astype(np.uint8)
    out[mask] = quantized
    return out


def _preprocess(rgba: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Produce the preprocessed RGB array that gets fed to vtracer, along
    with the number of distinct colors after quantization."""
    H, W, _ = rgba.shape
    rgb = rgba[..., :3].copy()
    alpha = rgba[..., 3]

    # Background mask: fully/mostly transparent OR near-white
    alpha_cut = int(cfg.get("alpha_threshold", 128))
    bg_mask = alpha < alpha_cut

    # Line mask (brightness below threshold AND opaque)
    line_thresh = int(cfg.get("line_color_threshold", 70))
    bright = brightness(rgb)
    line_mask = (bright < line_thresh) & ~bg_mask

    # Interior color mask
    color_mask = ~bg_mask & ~line_mask

    # Replace bg + line pixels with the sentinel so they become one single
    # "background" region in vtracer's output (easy to drop later).
    rgb[bg_mask] = SENTINEL_BG
    rgb[line_mask] = SENTINEL_BG

    # Quantize only the interior color pixels
    max_colors = int(cfg.get("max_colors", 6))
    rgb = _kmeans_quantize(rgb, color_mask, max_colors)

    # Count distinct colors (excluding sentinel)
    unique = {tuple(c) for c in rgb[color_mask].reshape(-1, 3).tolist()}
    return rgb, len(unique)


def vectorize(input_image: Path, intermediate_dir: Path, cfg: dict[str, Any]) -> VectorizeResult:
    """Run Step 1 on the input image. Writes intermediate artifacts to
    `intermediate_dir` and returns paths + metadata."""
    log = get_logger(verbose=cfg.get("verbose", True))
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    log.info("Step 1: loading image %s", input_image)
    rgba = load_image_rgba(input_image)
    H, W, _ = rgba.shape
    log.info("  image size: %dx%d", W, H)

    log.info("  preprocessing (alpha cutoff, line masking, k-means to %d colors)",
             cfg.get("max_colors", 6))
    preprocessed, n_colors = _preprocess(rgba, cfg)
    log.info("  distinct interior colors after quantization: %d", n_colors)

    pre_png = intermediate_dir / "01_preprocessed_for_vtracer.png"
    save_png(preprocessed, pre_png)
    log.debug("  wrote %s", pre_png)

    out_svg = intermediate_dir / "02_colors.svg"
    log.info("  running vtracer \u2192 %s", out_svg.name)

    # vtracer accepts a PNG path. Its convert_image_to_svg_py signature:
    #   (input_path, output_path, colormode, hierarchical, mode,
    #    filter_speckle, color_precision, layer_difference, corner_threshold,
    #    length_threshold, max_iterations, splice_threshold, path_precision)
    vtracer.convert_image_to_svg_py(
        str(pre_png),
        str(out_svg),
        colormode="color",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=6,
        color_precision=max(4, min(8, int(cfg.get("max_colors", 6)))),
        layer_difference=16,
        corner_threshold=60,
        length_threshold=4.0,
        max_iterations=10,
        splice_threshold=45,
        path_precision=3,
    )

    return VectorizeResult(
        color_svg_path=out_svg,
        preprocessed_png_path=pre_png,
        image_shape=(H, W),
        color_count=n_colors,
    )
