"""Step 7 — compute the keychain hole polygon in keychain-mm space.

Produces a shapely MultiPolygon positioned according to `hole_position` and
`hole_edge_margin`. The actual boolean subtraction happens in extrude.py
(applied to every layer so the hole cuts cleanly through all).

Shapes:
    round  — single circle, diameter = hole_diameter
    double — two circles, center-to-center = hole_spacing along X
    slot   — rounded slot, length along X, width along Y
    none   — returns empty MultiPolygon
"""
from __future__ import annotations

from math import cos, pi, sin
from typing import Any

from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from .util import get_logger


def _circle(cx: float, cy: float, radius: float, segments: int = 64) -> Polygon:
    return Point(cx, cy).buffer(radius, quad_segs=segments // 4)


def _slot(cx: float, cy: float, length: float, width: float, segments: int = 64) -> Polygon:
    # Rounded slot = convex hull of two circles offset along X by (length - width)/2
    offset = (length - width) / 2
    if offset <= 0:
        # degenerate => just a circle
        return _circle(cx, cy, width / 2, segments)
    c1 = _circle(cx - offset, cy, width / 2, segments)
    c2 = _circle(cx + offset, cy, width / 2, segments)
    return unary_union([c1, c2]).convex_hull


def _resolve_center(bounds_mm: tuple[tuple[float, float], tuple[float, float]],
                    position: str, edge_margin: float,
                    custom_offset: tuple[float, float]) -> tuple[float, float]:
    """Map a named position + margin to an (x, y) center in mm."""
    (x0, y0), (x1, y1) = bounds_mm
    top_y = y1 - edge_margin          # Y-up convention
    bot_y = y0 + edge_margin
    left_x = x0 + edge_margin
    right_x = x1 - edge_margin
    mid_x = 0.5 * (x0 + x1)
    mid_y = 0.5 * (y0 + y1)

    mapping = {
        "top-left":     (left_x,  top_y),
        "top-center":   (mid_x,   top_y),
        "top-right":    (right_x, top_y),
        "left-center":  (left_x,  mid_y),
        "right-center": (right_x, mid_y),
        "bottom-left":  (left_x,  bot_y),
        "bottom-center":(mid_x,   bot_y),
        "bottom-right": (right_x, bot_y),
    }
    if position == "custom":
        ox, oy = custom_offset
        # custom offset is mm from the bounding-box top-left (like screen coords)
        return (x0 + ox, y1 - oy)
    if position not in mapping:
        raise ValueError(f"Unknown hole_position: {position!r}")
    return mapping[position]


def build_keyhole(bounds_mm: tuple[tuple[float, float], tuple[float, float]],
                  cfg: dict[str, Any], verbose: bool = True) -> MultiPolygon:
    """Return a MultiPolygon (in keychain-mm coords) of the cutter shape.

    `bounds_mm` is ((min_x, min_y), (max_x, max_y)) of the silhouette AFTER
    px->mm transform. Empty MultiPolygon if hole_type is 'none'.
    """
    log = get_logger(verbose=verbose)
    hole_type = str(cfg.get("hole_type", "round")).lower()
    if hole_type == "none":
        log.info("Step 7: hole_type=none, no keychain hole will be cut")
        return MultiPolygon()

    position = str(cfg.get("hole_position", "top-center"))
    margin = float(cfg.get("hole_edge_margin", 3.0))
    custom_offset = tuple(cfg.get("hole_custom_offset", [0, 0]))
    cx, cy = _resolve_center(bounds_mm, position, margin, custom_offset)

    if hole_type == "round":
        d = float(cfg.get("hole_diameter", 4.0))
        geom = _circle(cx, cy, d / 2)
        log.info("Step 7: round hole d=%.2f at (%.2f, %.2f) mm", d, cx, cy)
    elif hole_type == "double":
        d = float(cfg.get("hole_diameter", 4.0))
        spacing = float(cfg.get("hole_spacing", 6.0))
        c1 = _circle(cx - spacing / 2, cy, d / 2)
        c2 = _circle(cx + spacing / 2, cy, d / 2)
        geom = unary_union([c1, c2])
        log.info("Step 7: double hole d=%.2f spacing=%.2f at (%.2f, %.2f) mm",
                 d, spacing, cx, cy)
    elif hole_type == "slot":
        length = float(cfg.get("hole_slot_length", 8.0))
        width = float(cfg.get("hole_slot_width", 4.0))
        geom = _slot(cx, cy, length, width)
        log.info("Step 7: slot %.2f x %.2f at (%.2f, %.2f) mm", length, width, cx, cy)
    else:
        raise ValueError(f"Unknown hole_type: {hole_type!r}")

    if geom.geom_type == "Polygon":
        return MultiPolygon([geom])
    if geom.geom_type == "MultiPolygon":
        return geom
    # Shouldn't happen, but fall back gracefully
    return MultiPolygon([geom.convex_hull])
