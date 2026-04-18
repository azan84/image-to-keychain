"""3MF export — single file, multiple objects.

For Bambu Studio + AMS, each keychain part (base, lines, each color) is
its own object in the 3MF. The user assigns filaments per-object in the
slicer. Because every mesh shares the same XY origin in keychain-mm
space, the slicer stacks them with zero drift.

trimesh's Scene.export handles 3MF natively. We set a per-geometry color
on the mesh's visual metadata as a hint — some slicers (incl. Bambu
Studio recent versions) pick it up for filament assignment defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh

from .extrude import KeychainPart
from .util import ensure_dir, get_logger


def _color_vertex(mesh: trimesh.Trimesh, rgb: tuple[int, int, int] | None) -> None:
    """Set a flat vertex color on the mesh so slicers can pick up a default
    filament hint. No-op if rgb is None."""
    if rgb is None:
        return
    rgba = np.array([rgb[0], rgb[1], rgb[2], 255], dtype=np.uint8)
    try:
        mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
    except Exception:
        # If the mesh already has a non-color visual, fall back to face colors
        try:
            mesh.visual.face_colors = np.tile(rgba, (len(mesh.faces), 1))
        except Exception:
            pass


def export_3mf(parts: Iterable[KeychainPart], output_dir: Path,
               output_basename: str, verbose: bool = True) -> Path:
    log = get_logger(verbose=verbose)
    ensure_dir(output_dir)
    out = output_dir / f"{output_basename}.3mf"

    scene = trimesh.Scene()
    n = 0
    for part in parts:
        if part.mesh is None:
            continue
        # Copy so we don't mutate the caller's mesh
        mesh = part.mesh.copy()
        # Line color defaults to black; base gets a neutral grey hint
        rgb = part.rgb if part.role != "base" else (200, 200, 200)
        _color_vertex(mesh, rgb)
        scene.add_geometry(mesh, node_name=part.name, geom_name=part.name)
        n += 1

    if n == 0:
        raise RuntimeError("No parts to export to 3MF")

    scene.export(str(out))
    log.info("exported 3MF (%d objects) -> %s (%.2f KB)",
             n, out, out.stat().st_size / 1024)
    return out
