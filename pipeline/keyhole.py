"""Step 7 — keychain tab + hole.

The tab is a rounded-rectangle lug attached to one side of the silhouette
that carries the keyring hole, so the hole never cuts into the subject
itself. If `tab_enabled` is false, we fall back to cutting the hole
directly into the silhouette (legacy behavior).

Returns both polygons in keychain-mm space:
    tab_mp    — union with the base so the base becomes silhouette + tab
    hole_mp   — subtracted from every layer; with a tab it lives inside
                the tab, not the subject

Tab placement:
    tab_side       = top | bottom | left | right
    tab_position   = 0..1 along the silhouette edge (0.5 = centered)
    tab_width_mm   = span along the edge
    tab_depth_mm   = how far the tab extends outward from the edge
    tab_corner_radius_mm
    tab_overlap_mm = how far the tab reaches into the silhouette before
                     extending outward; ensures the union is one solid piece

Hole inside the tab:
    Centered along the attached-edge direction, shifted toward the outer
    end by `hole_edge_margin + hole_effective_radius` so the keyring has
    clearance.

Hole in the silhouette (tab_enabled=false):
    Uses `hole_position` (8 named positions + custom) as before.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shapely.affinity import rotate, translate
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from .util import get_logger


@dataclass
class TabAndHole:
    tab: MultiPolygon       # empty if tab_enabled=False
    hole: MultiPolygon      # empty if hole_type=none
    tab_center: tuple[float, float] | None  # mm — for logging
    hole_center: tuple[float, float] | None


# ---- shapes ----------------------------------------------------------------

def _circle(cx: float, cy: float, radius: float, segments: int = 64) -> Polygon:
    return Point(cx, cy).buffer(radius, quad_segs=max(8, segments // 4))


def _slot(cx: float, cy: float, length: float, width: float, orient_deg: float = 0.0) -> Polygon:
    offset = (length - width) / 2
    if offset <= 0:
        return _circle(cx, cy, width / 2)
    c1 = _circle(cx - offset, cy, width / 2)
    c2 = _circle(cx + offset, cy, width / 2)
    slot = unary_union([c1, c2]).convex_hull
    if orient_deg:
        slot = rotate(slot, orient_deg, origin=(cx, cy))
    return slot


def _rounded_rect(x0: float, y0: float, x1: float, y1: float, r: float) -> Polygon:
    """Axis-aligned rectangle with rounded corners."""
    r = max(0.0, min(r, 0.5 * min(x1 - x0, y1 - y0) - 1e-6))
    if r <= 1e-6:
        return box(x0, y0, x1, y1)
    # Shapely idiom: inset then outset
    return box(x0, y0, x1, y1).buffer(-r, quad_segs=16).buffer(r, quad_segs=16)


# ---- hole-in-silhouette (legacy, tab_enabled=false) ------------------------

def _resolve_silhouette_center(bounds_mm: tuple[tuple[float, float], tuple[float, float]],
                               position: str, margin: float,
                               custom_offset: tuple[float, float]) -> tuple[float, float]:
    (x0, y0), (x1, y1) = bounds_mm
    top_y = y1 - margin
    bot_y = y0 + margin
    left_x = x0 + margin
    right_x = x1 - margin
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
        return (x0 + ox, y1 - oy)
    if position not in mapping:
        raise ValueError(f"Unknown hole_position: {position!r}")
    return mapping[position]


# ---- tab geometry ----------------------------------------------------------

def _build_tab(sil_bounds_mm: tuple[tuple[float, float], tuple[float, float]],
               cfg: dict[str, Any]) -> tuple[Polygon, tuple[float, float], str]:
    """Return (tab_polygon, hole_center, side). Hole center is the suggested
    hole location inside the tab — outer half of the tab along the
    extension direction, centered perpendicular to it."""
    (x0, y0), (x1, y1) = sil_bounds_mm
    side = str(cfg.get("tab_side", "top")).lower()
    pos = float(cfg.get("tab_position", 0.5))
    pos = max(0.0, min(1.0, pos))
    width = float(cfg.get("tab_width_mm", 10.0))
    depth = float(cfg.get("tab_depth_mm", 8.0))
    radius = float(cfg.get("tab_corner_radius_mm", 2.0))
    overlap = float(cfg.get("tab_overlap_mm", 2.0))

    hole_diameter = float(cfg.get("hole_diameter", 4.0))
    hole_slot_length = float(cfg.get("hole_slot_length", 8.0))
    hole_type = str(cfg.get("hole_type", "round")).lower()
    hole_margin = float(cfg.get("hole_edge_margin", 3.0))

    # Effective radius to clear in the "outward" direction so the hole sits
    # near the outer end of the tab without going through the tip
    effective_r = hole_diameter / 2
    if hole_type == "slot":
        effective_r = max(effective_r, hole_slot_length / 2)
    offset_from_outer = hole_margin + effective_r

    if side == "top":
        cx = x0 + (x1 - x0) * pos
        tab = _rounded_rect(cx - width / 2, y1 - overlap,
                            cx + width / 2, y1 + depth, radius)
        hole_cx, hole_cy = cx, (y1 + depth) - offset_from_outer
    elif side == "bottom":
        cx = x0 + (x1 - x0) * pos
        tab = _rounded_rect(cx - width / 2, y0 - depth,
                            cx + width / 2, y0 + overlap, radius)
        hole_cx, hole_cy = cx, (y0 - depth) + offset_from_outer
    elif side == "left":
        cy = y0 + (y1 - y0) * pos
        tab = _rounded_rect(x0 - depth, cy - width / 2,
                            x0 + overlap, cy + width / 2, radius)
        hole_cx, hole_cy = (x0 - depth) + offset_from_outer, cy
    elif side == "right":
        cy = y0 + (y1 - y0) * pos
        tab = _rounded_rect(x1 - overlap, cy - width / 2,
                            x1 + depth, cy + width / 2, radius)
        hole_cx, hole_cy = (x1 + depth) - offset_from_outer, cy
    else:
        raise ValueError(f"Unknown tab_side: {side!r} (expected top/bottom/left/right)")

    return tab, (hole_cx, hole_cy), side


def _build_hole_at(center: tuple[float, float], side: str,
                   cfg: dict[str, Any]) -> Polygon:
    """Build the cutter shape at a given center. For slots, orient the slot
    along the side so it's visually sensible on the tab."""
    hole_type = str(cfg.get("hole_type", "round")).lower()
    cx, cy = center
    if hole_type == "round":
        return _circle(cx, cy, float(cfg.get("hole_diameter", 4.0)) / 2)
    if hole_type == "double":
        d = float(cfg.get("hole_diameter", 4.0))
        spacing = float(cfg.get("hole_spacing", 6.0))
        # Place the pair along the side's long axis (perpendicular to outward)
        if side in ("top", "bottom"):
            c1 = _circle(cx - spacing / 2, cy, d / 2)
            c2 = _circle(cx + spacing / 2, cy, d / 2)
        else:
            c1 = _circle(cx, cy - spacing / 2, d / 2)
            c2 = _circle(cx, cy + spacing / 2, d / 2)
        return unary_union([c1, c2])
    if hole_type == "slot":
        length = float(cfg.get("hole_slot_length", 8.0))
        width = float(cfg.get("hole_slot_width", 4.0))
        # Slot runs along the side direction
        orient = 0.0 if side in ("top", "bottom") else 90.0
        return _slot(cx, cy, length, width, orient_deg=orient)
    raise ValueError(f"Unknown hole_type: {hole_type!r}")


def _as_mp(g) -> MultiPolygon:
    if g is None or g.is_empty:
        return MultiPolygon()
    if g.geom_type == "Polygon":
        return MultiPolygon([g])
    if g.geom_type == "MultiPolygon":
        return g
    # GeometryCollection or similar — pick polygons only
    polys = [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon" and not p.is_empty]
    return MultiPolygon(polys)


# ---- public API ------------------------------------------------------------

def build_tab_and_hole(bounds_mm: tuple[tuple[float, float], tuple[float, float]],
                       cfg: dict[str, Any], verbose: bool = True) -> TabAndHole:
    """Return (tab, hole) polygons in keychain-mm coords.

    `bounds_mm` is the silhouette's bounding box after px->mm transform,
    as ((min_x, min_y), (max_x, max_y)).
    """
    log = get_logger(verbose=verbose)
    hole_type = str(cfg.get("hole_type", "round")).lower()
    tab_enabled = bool(cfg.get("tab_enabled", True))

    if hole_type == "none":
        log.info("Step 7: hole_type=none, no keychain hole will be cut")
        return TabAndHole(tab=MultiPolygon(), hole=MultiPolygon(),
                          tab_center=None, hole_center=None)

    if tab_enabled:
        tab_poly, hole_center, side = _build_tab(bounds_mm, cfg)
        hole = _build_hole_at(hole_center, side, cfg)
        log.info("Step 7: tab on side=%s pos=%.2f  size %.1fx%.1f mm  hole %s at (%.2f, %.2f)",
                 side, float(cfg.get("tab_position", 0.5)),
                 float(cfg.get("tab_width_mm", 10.0)),
                 float(cfg.get("tab_depth_mm", 8.0)),
                 hole_type, *hole_center)
        return TabAndHole(tab=_as_mp(tab_poly), hole=_as_mp(hole),
                          tab_center=hole_center, hole_center=hole_center)

    # Legacy: hole cut directly into silhouette
    position = str(cfg.get("hole_position", "top-center"))
    margin = float(cfg.get("hole_edge_margin", 3.0))
    custom_offset = tuple(cfg.get("hole_custom_offset", [0, 0]))
    center = _resolve_silhouette_center(bounds_mm, position, margin, custom_offset)
    hole = _build_hole_at(center, "top", cfg)   # orientation doesn't matter for circle/double
    log.info("Step 7: hole (no tab) %s at (%.2f, %.2f) mm, position=%s",
             hole_type, *center, position)
    return TabAndHole(tab=MultiPolygon(), hole=_as_mp(hole),
                      tab_center=None, hole_center=center)


# ---- back-compat wrapper ---------------------------------------------------

def build_keyhole(bounds_mm, cfg: dict[str, Any], verbose: bool = True) -> MultiPolygon:
    """Back-compat shim: returns just the hole polygon. New code should call
    build_tab_and_hole() which also gives the tab."""
    return build_tab_and_hole(bounds_mm, cfg, verbose=verbose).hole
