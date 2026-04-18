"""Alignment tests — every part must share the same XY origin and scale.

Drift between layers is what makes multi-material prints fail, so we
verify the invariants that guarantee the slicer will stack them perfectly:

    1. Every part's XY bounding box lives inside the base's XY bounding box.
    2. The base's XY bounding box has its lower-left corner at (0, 0)
       (or within 0.001 mm of it — transform rounding).
    3. The keychain's longest XY dimension equals target_size_mm (within
       sub-pixel rounding).
"""
from __future__ import annotations

import pytest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

# Reuse the pipeline_output fixture from test_pipeline
from tests.test_pipeline import pipeline_output  # noqa: F401


EPS_MM = 0.01  # tolerance for float rounding after px -> mm scaling


class TestAlignment:

    def test_base_origin_at_zero(self, pipeline_output):
        base = next(p for p in pipeline_output["parts"] if p.role == "base")
        bb = base.mesh.bounds
        assert abs(bb[0, 0]) < EPS_MM, f"base min_x = {bb[0, 0]}"
        assert abs(bb[0, 1]) < EPS_MM, f"base min_y = {bb[0, 1]}"
        assert abs(bb[0, 2]) < EPS_MM, f"base min_z = {bb[0, 2]}"

    def test_target_size_achieved(self, pipeline_output):
        base = next(p for p in pipeline_output["parts"] if p.role == "base")
        bb = base.mesh.bounds
        longest = max(bb[1, 0] - bb[0, 0], bb[1, 1] - bb[0, 1])
        # Allow a pixel's worth of rounding
        assert abs(longest - pipeline_output["target_size_mm"]) < 0.1, \
            f"longest XY = {longest}, target = {pipeline_output['target_size_mm']}"

    def test_every_part_xy_within_base(self, pipeline_output):
        parts = pipeline_output["parts"]
        base = next(p for p in parts if p.role == "base")
        bb_base = base.mesh.bounds
        for part in parts:
            if part.role == "base":
                continue
            bb = part.mesh.bounds
            assert bb[0, 0] >= bb_base[0, 0] - EPS_MM, f"{part.name} extends left of base"
            assert bb[0, 1] >= bb_base[0, 1] - EPS_MM, f"{part.name} extends below base"
            assert bb[1, 0] <= bb_base[1, 0] + EPS_MM, f"{part.name} extends right of base"
            assert bb[1, 1] <= bb_base[1, 1] + EPS_MM, f"{part.name} extends above base"

    def test_layer_z_stacking(self, pipeline_output):
        """Lines and colors sit on the top surface of the base, lines extend higher."""
        parts = pipeline_output["parts"]
        base = next(p for p in parts if p.role == "base")
        top_of_base = base.mesh.bounds[1, 2]

        for part in parts:
            if part.role == "base":
                continue
            bb = part.mesh.bounds
            assert abs(bb[0, 2] - top_of_base) < EPS_MM, \
                f"{part.name} does not start at top of base ({bb[0, 2]} vs {top_of_base})"

        # Lines must protrude above colors
        lines = next((p for p in parts if p.role == "lines"), None)
        if lines is not None:
            line_top = lines.mesh.bounds[1, 2]
            for c in (p for p in parts if p.role == "color"):
                c_top = c.mesh.bounds[1, 2]
                assert line_top >= c_top - EPS_MM, \
                    f"lines top ({line_top}) should be >= color top ({c_top})"

    def test_keychain_hole_cut_through(self, pipeline_output):
        """The keychain hole must be cut cleanly through every layer.

        contains() needs rtree (which needs libspatialindex), so we check
        via vertex distances instead: no mesh vertex should sit strictly
        inside the hole's interior (i.e. closer to the center than the
        radius by more than a small tolerance). Vertices exactly on the
        circular boundary are allowed.
        """
        import numpy as np
        import yaml

        cfg = yaml.safe_load((PROJECT / "config.yaml").read_text())
        if cfg.get("hole_type") == "none":
            pytest.skip("hole_type=none; nothing to test")

        radius = float(cfg["hole_diameter"]) / 2
        margin = float(cfg["hole_edge_margin"])

        base = next(p for p in pipeline_output["parts"] if p.role == "base")
        bb = base.mesh.bounds
        cx = 0.5 * (bb[0, 0] + bb[1, 0])
        cy = bb[1, 1] - margin

        tol = 0.15  # mm; polygon-approx circle will have inset by ~sin(pi/64)*r
        for part in pipeline_output["parts"]:
            verts_xy = part.mesh.vertices[:, :2]
            dists = np.linalg.norm(verts_xy - np.array([cx, cy]), axis=1)
            min_dist = dists.min()
            assert min_dist >= radius - tol, \
                f"{part.name} has a vertex {min_dist:.3f}mm from hole center (< {radius}mm)"
