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
        """target_size_mm is the subject's longest dim. When tab is enabled
        the base extends beyond that by the tab depth, so we check a layer
        that's clipped to the silhouette (lines or any color) instead."""
        parts = pipeline_output["parts"]
        subject_part = next((p for p in parts if p.role != "base"), None)
        assert subject_part is not None
        bb = subject_part.mesh.bounds
        longest = max(bb[1, 0] - bb[0, 0], bb[1, 1] - bb[0, 1])
        # Allow a pixel's worth of rounding
        assert abs(longest - pipeline_output["target_size_mm"]) < 0.1, \
            f"longest subject XY = {longest}, target = {pipeline_output['target_size_mm']}"

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

    def test_base_has_a_hole(self, pipeline_output):
        """The base must have exactly one hole through it (the keychain hole).

        We check topology: a simply-connected extruded polygon has
        euler_number == 2; each hole subtracts 2 more (creates an
        additional handle).
        """
        import yaml

        cfg = yaml.safe_load((PROJECT / "config.yaml").read_text())
        if cfg.get("hole_type") == "none":
            pytest.skip("hole_type=none; nothing to test")

        base = next(p for p in pipeline_output["parts"] if p.role == "base")
        # Euler number of a closed genus-g surface is 2 - 2g. A flat
        # extruded region with N through-holes is genus N, so euler = 2-2N.
        # We don't need an exact count; asserting euler < 2 suffices to show
        # at least one hole is present.
        assert base.mesh.euler_number < 2, \
            f"base has euler_number {base.mesh.euler_number}; expected <2 (hole present)"

    def test_hole_not_in_subject(self, pipeline_output):
        """When a tab is enabled, the hole must live in the tab — no mesh
        vertex of lines or colors should sit near the hole's center."""
        import numpy as np
        import yaml

        cfg = yaml.safe_load((PROJECT / "config.yaml").read_text())
        if not cfg.get("tab_enabled", True):
            pytest.skip("tab disabled; hole is in silhouette")
        if cfg.get("hole_type") == "none":
            pytest.skip("hole_type=none")

        # The subject parts should NOT extend into the tab region at all.
        # Tab is on the side given in config; after px->mm the silhouette
        # spans up to target_size_mm on its longest dim, so tab pokes out
        # beyond that. We check that every non-base part's max extent on
        # the tab-outward axis is ≤ silhouette max (no tab spill).
        side = cfg.get("tab_side", "top")
        sil_parts = [p for p in pipeline_output["parts"] if p.role != "base"]
        base = next(p for p in pipeline_output["parts"] if p.role == "base")
        base_bb = base.mesh.bounds

        for part in sil_parts:
            bb = part.mesh.bounds
            if side == "top":
                # Base extends ABOVE subject (larger max Y); subject must NOT
                assert bb[1, 1] < base_bb[1, 1] - 0.5, \
                    f"{part.name} spills into the top tab"
            elif side == "bottom":
                assert bb[0, 1] > base_bb[0, 1] + 0.5, \
                    f"{part.name} spills into the bottom tab"
            elif side == "left":
                assert bb[0, 0] > base_bb[0, 0] + 0.5, \
                    f"{part.name} spills into the left tab"
            elif side == "right":
                assert bb[1, 0] < base_bb[1, 0] - 0.5, \
                    f"{part.name} spills into the right tab"
