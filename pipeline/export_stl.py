"""STL export — one file per keychain part.

Output layout:
    {output_dir}/{output_basename}_parts/base.stl
    {output_dir}/{output_basename}_parts/lines.stl
    {output_dir}/{output_basename}_parts/color_01_d1a275.stl
    ...

Every file carries the same XY origin and scale, so any slicer that loads
them all will stack them correctly (Bambu Studio, PrusaSlicer, Cura).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .extrude import KeychainPart
from .util import ensure_dir, get_logger


def export_stl_parts(parts: Iterable[KeychainPart], output_dir: Path,
                     output_basename: str, verbose: bool = True) -> Path:
    log = get_logger(verbose=verbose)
    parts_dir = ensure_dir(output_dir / f"{output_basename}_parts")
    log.info("exporting STL parts -> %s", parts_dir)

    for part in parts:
        if part.mesh is None:
            log.warning("  %s has no mesh (skipped)", part.name)
            continue
        out = parts_dir / f"{part.name}.stl"
        part.mesh.export(str(out))
        log.info("  wrote %s (%.2f KB)", out.name, out.stat().st_size / 1024)

    return parts_dir
