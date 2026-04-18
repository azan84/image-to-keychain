"""Steps 5 & 6 — 2D to 3D extrusion, plus the px -> mm transform.

All parts share a single deterministic transform from image-pixel space
to keychain-mm space. The silhouette's bounding box is placed with its
min corner at the origin so the keychain sits in the +X/+Y octant, which
makes Bambu Studio happy.

Z stacking:
    base       : 0                            .. base_thickness
    color/line : base_thickness               .. base_thickness + (that layer's thickness)

Line thickness is typically > color thickness, so lines protrude above
colors — the visual emphasis effect the user wants.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import manifold3d
import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import orient, unary_union
from shapely.validation import make_valid

from .util import get_logger


@dataclass
class KeychainPart:
    name: str                      # e.g. "base", "lines", "color_02_c82b27"
    role: str                      # "base" | "lines" | "color"
    polygon: MultiPolygon          # mm coords, Y-up
    z_min: float                   # mm
    z_max: float                   # mm
    rgb: tuple[int, int, int] | None   # for colors + lines; None for base
    mesh: trimesh.Trimesh | None = None  # populated after extrude()

    @property
    def thickness(self) -> float:
        return self.z_max - self.z_min


# ---- transform -------------------------------------------------------------

def compute_px_to_mm(image_shape: tuple[int, int], target_size_mm: float,
                     silhouette_poly_px: MultiPolygon) -> dict:
    """Return a transform descriptor: scale (mm/px), translate (mm) so that
    the silhouette's bbox lower-left sits at (0, 0). Y is already flipped
    upstream (silhouette is in math-convention px-space with Y up).

    target_size_mm is applied to the silhouette's longest dimension, not
    the raw image dimension — so a subject that doesn't fill the canvas
    still comes out at the requested keychain size.
    """
    minx, miny, maxx, maxy = silhouette_poly_px.bounds
    subj_w = max(1.0, maxx - minx)
    subj_h = max(1.0, maxy - miny)
    scale = target_size_mm / max(subj_w, subj_h)
    tx = -minx * scale
    ty = -miny * scale
    return {
        "scale": scale,
        "translate": (tx, ty),
        "bounds_mm": ((0.0, 0.0), (subj_w * scale, subj_h * scale)),
    }


def apply_transform(poly: MultiPolygon | Polygon, xform: dict) -> MultiPolygon:
    s = xform["scale"]
    tx, ty = xform["translate"]
    scaled = affinity.scale(poly, xfact=s, yfact=s, origin=(0, 0))
    translated = affinity.translate(scaled, xoff=tx, yoff=ty)
    if translated.geom_type == "Polygon":
        return MultiPolygon([translated])
    return translated


# ---- extrusion -------------------------------------------------------------

def _polygon_to_contours(poly: Polygon) -> list[list[tuple[float, float]]]:
    """Outer ring as CCW, holes as CW (manifold3d Positive fill rule)."""
    p = orient(poly, sign=1.0)  # exterior CCW, interiors CW
    out: list[list[tuple[float, float]]] = []
    ext = list(p.exterior.coords)
    if ext and ext[0] == ext[-1]:
        ext = ext[:-1]
    out.append([(float(x), float(y)) for x, y in ext])
    for interior in p.interiors:
        ring = list(interior.coords)
        if ring and ring[0] == ring[-1]:
            ring = ring[:-1]
        out.append([(float(x), float(y)) for x, y in ring])
    return out


def _multipolygon_to_contours(mp: MultiPolygon) -> list[list[tuple[float, float]]]:
    all_contours: list[list[tuple[float, float]]] = []
    for p in mp.geoms:
        if p.is_empty or p.area <= 1e-9:
            continue
        all_contours.extend(_polygon_to_contours(p))
    return all_contours


def _manifold_to_trimesh(m: manifold3d.Manifold, z_offset: float) -> trimesh.Trimesh:
    mesh = m.to_mesh()
    verts = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64).copy()
    faces = np.asarray(mesh.tri_verts).astype(np.int64)
    if z_offset != 0.0:
        verts[:, 2] += z_offset
    # process=False preserves manifold3d's exact topology; process=True will
    # merge near-coincident vertices that can break watertightness when disjoint
    # speckles happen to share a sub-pixel coordinate.
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _clean_multipolygon(mp: MultiPolygon) -> MultiPolygon:
    """Resolve self-intersections, merge overlaps, drop degenerate pieces.
    Guarantees the result is a MultiPolygon of non-overlapping valid Polygons."""
    if mp.is_empty:
        return MultiPolygon()
    # unary_union resolves overlaps between sub-polygons
    merged = unary_union(list(mp.geoms))
    if merged.is_empty:
        return MultiPolygon()
    # buffer(0) is a classic shapely idiom to heal tiny self-intersections
    merged = merged.buffer(0)
    if not merged.is_valid:
        merged = make_valid(merged)
    if merged.geom_type == "Polygon":
        return MultiPolygon([merged]) if not merged.is_empty and merged.area > 1e-9 else MultiPolygon()
    if merged.geom_type == "MultiPolygon":
        out = [p for p in merged.geoms if not p.is_empty and p.area > 1e-9]
        return MultiPolygon(out) if out else MultiPolygon()
    # GeometryCollection or LineString etc — keep only polygonal parts
    polys: list[Polygon] = []
    for g in getattr(merged, "geoms", []):
        if g.geom_type == "Polygon" and not g.is_empty and g.area > 1e-9:
            polys.append(g)
    return MultiPolygon(polys) if polys else MultiPolygon()


def extrude_multipolygon(mp: MultiPolygon, z_min: float, z_max: float) -> trimesh.Trimesh:
    """Extrude a MultiPolygon into a single watertight mesh via manifold3d.

    manifold3d.CrossSection performs an internal boolean union on the input
    contours (default FillRule.Positive), so overlapping sub-polygons, tiny
    self-intersections, and pathological topologies are all resolved before
    extrusion. Result is guaranteed manifold and watertight.
    """
    height = z_max - z_min
    if height <= 0:
        raise ValueError(f"non-positive extrusion height {height}")

    clean = _clean_multipolygon(mp)
    if clean.is_empty:
        raise ValueError("No extrudable polygons in multipolygon (empty after cleaning)")

    contours = _multipolygon_to_contours(clean)
    if not contours:
        raise ValueError("No extrudable contours")

    cs = manifold3d.CrossSection(contours, manifold3d.FillRule.Positive)
    if cs.is_empty():
        # Fallback: try NonZero, which is more permissive about winding
        cs = manifold3d.CrossSection(contours, manifold3d.FillRule.NonZero)
    if cs.is_empty():
        raise ValueError("CrossSection empty after construction")

    m = cs.extrude(height)
    return _manifold_to_trimesh(m, z_offset=z_min)


# ---- top-level -------------------------------------------------------------

def build_parts(silhouette_mp: MultiPolygon,
                lines_mp: MultiPolygon,
                colors: list,                # list[ColorPart]
                hole_mp: MultiPolygon | None,
                image_shape: tuple[int, int],
                cfg: dict,
                tab_mp: MultiPolygon | None = None,
                verbose: bool = True) -> list[KeychainPart]:
    """Build all keychain parts: base (+tab) + lines + one per color cluster.

    Base polygon = silhouette \u222a tab  MINUS hole.
    Lines and colors are clipped to the original silhouette so they never
    extend onto the tab (tab should be flat base-material only).
    """
    log = get_logger(verbose=verbose)
    target_mm = float(cfg.get("target_size_mm", 60.0))
    base_t = float(cfg.get("base_thickness", 2.0))
    line_t = float(cfg.get("line_thickness", 1.5))
    color_t = float(cfg.get("color_thickness", 1.0))

    xform = compute_px_to_mm(image_shape, target_mm, silhouette_mp)
    log.info("Steps 5-6: extruding parts (scale=%.5f mm/px, translate=(%.3f, %.3f))",
             xform["scale"], *xform["translate"])
    (minx, miny), (maxx, maxy) = xform["bounds_mm"]
    log.info("  keychain XY bounds (subject only): (%.2f, %.2f) -> (%.2f, %.2f) mm",
             minx, miny, maxx, maxy)

    # Apply transform. Tab already arrives in keychain-mm space (built by
    # keyhole.build_tab_and_hole from the same bounds), so no transform.
    sil_mm = apply_transform(silhouette_mp, xform)
    lines_mm = apply_transform(lines_mp, xform) if not lines_mp.is_empty else MultiPolygon()
    color_polys_mm = [(cp, apply_transform(cp.polygon, xform)) for cp in colors]

    has_tab = tab_mp is not None and not tab_mp.is_empty

    def sub_hole(p: MultiPolygon) -> MultiPolygon:
        if hole_mp is None or hole_mp.is_empty:
            return p
        result = p.difference(hole_mp)
        if result.geom_type == "Polygon":
            return MultiPolygon([result]) if not result.is_empty else MultiPolygon()
        if result.geom_type == "MultiPolygon":
            return result
        return MultiPolygon()

    def clip_to_silhouette(p: MultiPolygon) -> MultiPolygon:
        c = p.intersection(sil_mm)
        if c.geom_type == "Polygon":
            return MultiPolygon([c]) if not c.is_empty else MultiPolygon()
        if c.geom_type == "MultiPolygon":
            return c
        return MultiPolygon()

    parts: list[KeychainPart] = []

    # Base = silhouette only. With a tab, the base does NOT get the hole
    # subtracted — the tab is a separate object and carries the hole.
    # Without a tab, subtract the hole from the base (legacy behavior).
    if has_tab:
        base_poly = sil_mm
    else:
        base_poly = sub_hole(sil_mm)
    parts.append(KeychainPart(name="base", role="base", polygon=base_poly,
                              z_min=0.0, z_max=base_t, rgb=None))

    # Tab — separate part with its own hole. Same thickness as the base
    # so the user can weld them together in the slicer.
    if has_tab:
        tab_poly = sub_hole(tab_mp)
        if not tab_poly.is_empty:
            parts.append(KeychainPart(
                name="tab", role="tab", polygon=tab_poly,
                z_min=0.0, z_max=base_t, rgb=None,
            ))
            bb = tab_poly.bounds
            log.info("  tab (separate part) bounds: (%.2f, %.2f) -> (%.2f, %.2f) mm",
                     bb[0], bb[1], bb[2], bb[3])

    # Lines — clipped to silhouette so they never land on the tab. When the
    # tab carries the hole, no sub_hole call is needed here.
    if not lines_mm.is_empty:
        lp = clip_to_silhouette(lines_mm if has_tab else sub_hole(lines_mm))
        if not lp.is_empty:
            parts.append(KeychainPart(name="lines", role="lines", polygon=lp,
                                      z_min=base_t, z_max=base_t + line_t,
                                      rgb=(0, 0, 0)))

    # Colors
    for cp, poly_mm in color_polys_mm:
        poly_to_use = poly_mm if has_tab else sub_hole(poly_mm)
        clipped = clip_to_silhouette(poly_to_use)
        if clipped.is_empty:
            log.warning("  color %s clipped to empty — skipping", cp.safe_name)
            continue
        parts.append(KeychainPart(
            name=cp.safe_name, role="color", polygon=clipped,
            z_min=base_t, z_max=base_t + color_t, rgb=cp.rgb,
        ))

    # Extrude
    for part in parts:
        log.info("  extruding %s (z %.2f..%.2f mm, %d polys)",
                 part.name, part.z_min, part.z_max, len(part.polygon.geoms))
        part.mesh = extrude_multipolygon(part.polygon, part.z_min, part.z_max)
        # Sanity check
        m = part.mesh
        log.info("    -> %d verts, %d faces, watertight=%s, volume=%.2f mm\u00b3",
                 len(m.vertices), len(m.faces), m.is_watertight, m.volume)

    return parts
