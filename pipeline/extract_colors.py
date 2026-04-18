"""Step 3 — extract per-cluster color polygons.

Each non-background cluster in the preprocess label map becomes its own
shapely MultiPolygon. These are the color parts the slicer will assign
filament to via AMS.

No line subtraction happens here — per the project spec, lines overlap
the color regions (they sit on top in Z, so the visual stack works out).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPolygon

from .preprocess import PreprocessResult
from .svg_polygons import mask_to_multipolygon
from .util import get_logger


@dataclass
class ColorPart:
    cluster_id: int
    rgb: tuple[int, int, int]      # (R, G, B) 0-255
    hex_color: str                 # "ff5733"
    polygon: MultiPolygon          # in image-pixel coords, Y-flipped
    pixel_count: int

    @property
    def safe_name(self) -> str:
        """Slugged filename-safe tag, e.g. 'color_02_ff5733'."""
        return f"color_{self.cluster_id:02d}_{self.hex_color}"


def extract_colors(pre: PreprocessResult, intermediate_dir: Path,
                   verbose: bool = True, min_pixel_fraction: float = 0.001) -> list[ColorPart]:
    """Return one ColorPart per non-bg cluster (sorted by pixel count desc).

    `min_pixel_fraction`: drop clusters smaller than this fraction of the
    silhouette — avoids noise clusters that print as invisible specks.
    """
    log = get_logger(verbose=verbose)
    log.info("Step 3: extracting per-cluster color polygons")

    H, W = pre.shape
    sil_px = int((pre.labels >= 0).sum())
    min_px = max(1, int(sil_px * min_pixel_fraction))

    unique = np.unique(pre.labels[pre.labels >= 0])
    log.info("  non-bg clusters to extract: %s", unique.tolist())

    parts: list[ColorPart] = []
    for cid in unique.tolist():
        mask = pre.labels == cid
        px = int(mask.sum())
        if px < min_px:
            log.info("    cluster %d skipped (%d px < %d min)", cid, px, min_px)
            continue
        rgb = tuple(int(v) for v in pre.palette[cid])
        hex_color = "{:02x}{:02x}{:02x}".format(*rgb)
        tag = f"color_{cid:02d}_{hex_color}"
        mp = mask_to_multipolygon(mask, intermediate_dir, tag, flip_y=True, min_area=20.0)
        if mp.is_empty:
            log.warning("    cluster %d vectorized to empty polygon — skipping", cid)
            continue
        part = ColorPart(cluster_id=cid, rgb=rgb, hex_color=hex_color,
                         polygon=mp, pixel_count=px)
        log.info("    %s: %d px, %d subpolygon(s), bbox %s",
                 part.safe_name, px, len(mp.geoms), mp.bounds)
        parts.append(part)

    parts.sort(key=lambda p: p.pixel_count, reverse=True)
    return parts
