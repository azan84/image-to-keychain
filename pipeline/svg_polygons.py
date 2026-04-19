"""Mask -> shapely MultiPolygon via vtracer.

We rasterize a bool mask, ask vtracer to trace it in binary / polygon mode,
then parse the resulting SVG with svgpathtools and build shapely Polygons
(with holes) using even-odd containment analysis.

Shapely + networkx are already in the dep list; rtree is not (libspatialindex
requires sudo), so we cannot use trimesh.load_path on the SVG directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import svgpathtools as spt
import vtracer
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

from .util import save_png


# ---- vtracer wrapper -------------------------------------------------------

def mask_to_svg(mask: np.ndarray, out_svg: Path, tmp_png: Path,
                filter_speckle: int = 2, simplify_tolerance: float = 0.5) -> Path:
    """Write `mask` (True = foreground) as black-on-white PNG and trace it to SVG."""
    if mask.dtype != bool:
        mask = mask.astype(bool)
    img = np.full(mask.shape, 255, dtype=np.uint8)
    img[mask] = 0
    save_png(img, tmp_png)
    # polygon mode yields pure line segments — easier to parse than splines.
    vtracer.convert_image_to_svg_py(
        str(tmp_png),
        str(out_svg),
        colormode="binary",
        hierarchical="stacked",
        mode="polygon",
        filter_speckle=filter_speckle,
        color_precision=1,
        layer_difference=0,
        corner_threshold=60,
        length_threshold=max(2.0, simplify_tolerance * 4),
        max_iterations=10,
        splice_threshold=45,
        path_precision=3,
    )
    return out_svg


# ---- SVG -> shapely --------------------------------------------------------

_TRANSLATE_RE = __import__("re").compile(r"translate\(\s*(-?[\d.eE+-]+)\s*[, ]\s*(-?[\d.eE+-]+)\s*\)")


def _parse_translate(transform: str | None) -> tuple[float, float]:
    """Extract (dx, dy) from an SVG transform attribute. vtracer only ever
    emits `translate(x,y)`, so we parse just that (nothing fancier)."""
    if not transform:
        return (0.0, 0.0)
    m = _TRANSLATE_RE.search(transform)
    if not m:
        return (0.0, 0.0)
    return (float(m.group(1)), float(m.group(2)))


def _sample_subpath(subpath: spt.Path, max_seg_len: float = 1.5) -> list[tuple[float, float]]:
    """Sample a continuous svgpathtools subpath into a list of (x, y) points.
    vtracer polygon mode outputs only Line segments, so this is really just
    a coordinate dump; we still handle curves in case we switch back to splines.
    """
    pts: list[tuple[float, float]] = []
    for seg in subpath:
        if isinstance(seg, spt.Line):
            if not pts:
                pts.append((seg.start.real, seg.start.imag))
            pts.append((seg.end.real, seg.end.imag))
        else:
            # Curve — sample adaptively by segment length
            length = max(seg.length(), 1.0)
            n = max(2, int(length / max_seg_len))
            for i in range(n + 1):
                p = seg.point(i / n)
                if pts and abs(p.real - pts[-1][0]) < 1e-9 and abs(p.imag - pts[-1][1]) < 1e-9:
                    continue
                pts.append((p.real, p.imag))
    # Ensure closed ring
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def _ring_from_subpath(subpath: spt.Path, translate: tuple[float, float] = (0.0, 0.0)) -> Polygon | None:
    pts = _sample_subpath(subpath)
    if len(pts) < 4:
        return None
    if translate != (0.0, 0.0):
        dx, dy = translate
        pts = [(x + dx, y + dy) for (x, y) in pts]
    poly = Polygon(pts)
    if not poly.is_valid:
        fixed = make_valid(poly)
        if fixed.is_empty:
            return None
        if fixed.geom_type == "Polygon":
            poly = fixed
        elif fixed.geom_type == "MultiPolygon":
            poly = max(fixed.geoms, key=lambda g: g.area)
        else:
            return None
    if poly.area <= 0:
        return None
    return poly


def svg_to_multipolygon(svg_path: Path, flip_y_height: float | None = None,
                       min_area: float = 1.0) -> MultiPolygon:
    """Parse an SVG produced by vtracer into a shapely MultiPolygon.

    `flip_y_height`: if given, mirrors Y about this value (use image height
    in pixels so the output is in math-convention XY, Y up).
    `min_area`: drop rings smaller than this (pixels²) — kills speckle.

    Holes vs outers are resolved by even-odd containment: a ring contained
    in an odd number of other rings is a hole, even (including 0) is outer.
    """
    paths, attrs = spt.svg2paths(str(svg_path))

    # Collect every closed subpath as a standalone polygon (ignore holes for now).
    rings: list[Polygon] = []
    for path, attr in zip(paths, attrs):
        translate = _parse_translate(attr.get("transform"))
        for sp in path.continuous_subpaths():
            ring = _ring_from_subpath(sp, translate=translate)
            if ring is None or ring.area < min_area:
                continue
            if flip_y_height is not None:
                # Mirror Y about flip_y_height
                coords = [(x, flip_y_height - y) for (x, y) in ring.exterior.coords]
                ring = Polygon(coords)
                if not ring.is_valid:
                    ring = make_valid(ring)
                    if ring.is_empty or ring.geom_type != "Polygon":
                        continue
            rings.append(ring)

    if not rings:
        return MultiPolygon()

    # Containment (nesting) analysis — sort by area descending so outer rings
    # come first, then each smaller ring is tested against already-placed ones.
    rings.sort(key=lambda p: p.area, reverse=True)
    depth = [0] * len(rings)
    for i in range(len(rings)):
        for j in range(i):
            if rings[j].contains(rings[i].representative_point()):
                depth[i] = depth[j] + 1
                break  # take the first (innermost) enclosing ring

    # Group each outer (even depth) with its immediate holes (depth+1 child)
    outers: list[tuple[int, Polygon]] = [(i, r) for i, r in enumerate(rings) if depth[i] % 2 == 0]
    holes_by_outer: dict[int, list[Polygon]] = {i: [] for i, _ in outers}
    for i, r in enumerate(rings):
        if depth[i] % 2 == 1:
            # Parent is the smallest outer that contains this hole
            rp = r.representative_point()
            best = None
            best_area = float("inf")
            for oi, o in outers:
                if o.contains(rp) and o.area < best_area:
                    best = oi
                    best_area = o.area
            if best is not None:
                holes_by_outer[best].append(r)

    out_polys: list[Polygon] = []
    for oi, outer in outers:
        holes = [list(h.exterior.coords) for h in holes_by_outer.get(oi, [])]
        p = Polygon(list(outer.exterior.coords), holes=holes)
        if not p.is_valid:
            p = make_valid(p)
            if p.is_empty:
                continue
            if p.geom_type == "Polygon":
                out_polys.append(p)
            elif p.geom_type == "MultiPolygon":
                out_polys.extend(list(p.geoms))
        else:
            out_polys.append(p)

    if not out_polys:
        return MultiPolygon()
    # Final validity pass — some vtracer outputs contain near-touching edges
    # that pass Polygon.is_valid but still trip GEOS in downstream booleans.
    # buffer(0) is the standard heal; make_valid catches anything left.
    mp = MultiPolygon(out_polys)
    if not mp.is_valid:
        mp = make_valid(mp)
    clean = mp.buffer(0)
    if clean.is_empty:
        return MultiPolygon()
    if clean.geom_type == "Polygon":
        return MultiPolygon([clean])
    if clean.geom_type == "MultiPolygon":
        return clean
    # GeometryCollection — keep only the Polygon parts
    keep = [g for g in getattr(clean, "geoms", []) if g.geom_type == "Polygon" and not g.is_empty]
    return MultiPolygon(keep) if keep else MultiPolygon()


def mask_to_multipolygon(mask: np.ndarray, tmp_dir: Path, tag: str,
                         flip_y: bool = True, min_area: float = 1.0) -> MultiPolygon:
    """Convenience: bool mask -> shapely MultiPolygon. Writes temp files to tmp_dir."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    png = tmp_dir / f"_tmp_{tag}_mask.png"
    svg = tmp_dir / f"_tmp_{tag}.svg"
    mask_to_svg(mask, svg, png)
    H = mask.shape[0]
    return svg_to_multipolygon(svg, flip_y_height=(H if flip_y else None), min_area=min_area)
