"""Step 1 — vectorize the input raster into colored SVG paths.

Works off the shared preprocess labels. We repaint the image with each
cluster's centroid color (and a white sentinel for the background), then
let vtracer trace a stacked color SVG. The resulting SVG is primarily
useful as a human-inspectable debug artifact; downstream stages operate on
the label map directly.

Dark regions (e.g. black hair) are NOT excluded here. They are their own
color cluster and will be handled as a normal color part in Step 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import vtracer

from .preprocess import preprocess
from .util import get_logger, save_png


SENTINEL_BG = np.array([255, 255, 255], dtype=np.uint8)


@dataclass
class VectorizeResult:
    color_svg_path: Path
    preprocessed_png_path: Path
    image_shape: tuple[int, int]
    color_count: int


def vectorize(input_image: Path, intermediate_dir: Path, cfg: dict[str, Any]) -> VectorizeResult:
    log = get_logger(verbose=cfg.get("verbose", True))
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    log.info("Step 1: vectorizing %s", input_image.name)
    pre = preprocess(input_image, cfg)
    H, W = pre.shape

    # Repaint with centroid color; bg pixels get the sentinel (white).
    repainted = np.where(
        pre.bg_mask[..., None],
        SENTINEL_BG[None, None, :],
        pre.palette[np.where(pre.labels >= 0, pre.labels, 0)],
    ).astype(np.uint8)

    pre_png = intermediate_dir / "01_preprocessed_for_vtracer.png"
    save_png(repainted, pre_png)
    log.debug("  wrote %s", pre_png.name)

    out_svg = intermediate_dir / "02_colors.svg"
    log.info("  vtracer \u2192 %s", out_svg.name)
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
        color_count=int(pre.palette.shape[0]),
    )
