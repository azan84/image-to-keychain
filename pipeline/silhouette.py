"""Step 4 — generate the silhouette polygon from the label map.

The silhouette is the outer 2D boundary of the entire subject (everything
that isn't background). It's used as the base plate and for bounding-box
calculations (keychain hole placement, target-size scaling).

Holes inside the silhouette (e.g. a gap between head and raised arm) are
preserved — they become genuine holes in the base plate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import MultiPolygon

from .preprocess import PreprocessResult
from .svg_polygons import mask_to_multipolygon
from .util import get_logger


@dataclass
class SilhouetteResult:
    polygon: MultiPolygon          # in image-pixel coords, Y flipped (math convention)
    image_shape: tuple[int, int]   # (H, W) in pixels


def build_silhouette(pre: PreprocessResult, intermediate_dir: Path,
                     verbose: bool = True) -> SilhouetteResult:
    log = get_logger(verbose=verbose)
    log.info("Step 4: building silhouette polygon")
    sil_mask = pre.labels >= 0
    H, W = pre.shape
    log.info("  silhouette pixels: %d (%.1f%% of image)",
             int(sil_mask.sum()), 100.0 * sil_mask.sum() / (H * W))

    mp = mask_to_multipolygon(sil_mask, intermediate_dir, "silhouette",
                              flip_y=True, min_area=50.0)
    if mp.is_empty:
        raise RuntimeError("Silhouette extraction produced no polygons.")

    polys = list(mp.geoms)
    total_area = sum(p.area for p in polys)
    log.info("  silhouette: %d polygon(s), total area %.0f px\u00b2, bbox %s",
             len(polys), total_area, mp.bounds)
    return SilhouetteResult(polygon=mp, image_shape=(H, W))
