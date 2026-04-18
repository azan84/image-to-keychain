"""Fast 2D top-down preview renderer.

Given the pipeline's shapely polygons in keychain-mm space, paints a
raster that shows roughly what the printed keychain will look like.
This is orders of magnitude faster than extruding + rendering the mesh,
so it's suitable for a live-updating UI.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon

from .extrude import apply_transform


_BG = (250, 250, 250)
_BASE_FILL = (210, 210, 210)
_TAB_FILL = (185, 185, 185)


def render_topdown(sil_mm: MultiPolygon,
                   lines_mm: MultiPolygon,
                   color_parts: list,               # list[ColorPart], mm-transformed polygons expected via cb
                   tab_mm: MultiPolygon,
                   hole_mm: MultiPolygon,
                   bounds_mm: tuple[tuple[float, float], tuple[float, float]],
                   ppmm: int = 10,
                   margin_px: int = 20) -> Image.Image:
    """Render a PIL Image showing base + tab + colors + lines + hole position.

    `color_parts` is a sequence of tuples (mm_polygon, rgb) — already in
    keychain-mm space.
    `bounds_mm` is the (min, max) bounds to use for the canvas. Should
    include both silhouette and tab so nothing is clipped.
    """
    (mn_x, mn_y), (mx_x, mx_y) = bounds_mm
    W = int((mx_x - mn_x) * ppmm) + 2 * margin_px
    H = int((mx_y - mn_y) * ppmm) + 2 * margin_px
    img = Image.new("RGB", (W, H), _BG)
    draw = ImageDraw.Draw(img, "RGBA")

    def to_px(x: float, y: float) -> tuple[int, int]:
        return (int((x - mn_x) * ppmm) + margin_px,
                H - int((y - mn_y) * ppmm) - margin_px)

    def draw_mp(mp: MultiPolygon, fill: tuple) -> None:
        if mp is None or mp.is_empty:
            return
        geoms = mp.geoms if mp.geom_type == "MultiPolygon" else [mp]
        for p in geoms:
            if p.is_empty:
                continue
            ext = [to_px(x, y) for x, y in p.exterior.coords]
            draw.polygon(ext, fill=fill)
            for interior in p.interiors:
                draw.polygon([to_px(x, y) for x, y in interior.coords], fill=_BG)

    # Paint order: tab (below), base (silhouette), colors, lines, hole outline
    draw_mp(tab_mm, _TAB_FILL)
    draw_mp(sil_mm, _BASE_FILL)
    # Colors — paint largest-first so smaller ones stay visible on top
    for poly_mm, rgb in sorted(color_parts, key=lambda t: t[0].area, reverse=True):
        draw_mp(poly_mm, (rgb[0], rgb[1], rgb[2], 255))
    draw_mp(lines_mm, (0, 0, 0, 255))
    # Hole — outline only, so you see where it'll be cut
    if hole_mm is not None and not hole_mm.is_empty:
        for p in hole_mm.geoms:
            pts = [to_px(x, y) for x, y in p.exterior.coords]
            draw.polygon(pts, fill=_BG, outline=(230, 60, 60), width=3)

    return img
