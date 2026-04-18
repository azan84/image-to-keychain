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
        """target_size_mm is the subject's longest dim. Check a subject
        layer (not base, not tab) since tab is small and base equals the
        silhouette anyway."""
        parts = pipeline_output["parts"]
        subject_part = next((p for p in parts if p.role not in ("base", "tab")), None)
        assert subject_part is not None
        bb = subject_part.mesh.bounds
        longest = max(bb[1, 0] - bb[0, 0], bb[1, 1] - bb[0, 1])
        assert abs(longest - pipeline_output["target_size_mm"]) < 0.1, \
            f"longest subject XY = {longest}, target = {pipeline_output['target_size_mm']}"

    def test_subject_layers_within_base(self, pipeline_output):
        """Lines and colors must sit inside the base (silhouette) footprint.
        The tab is a separate object and is allowed to extend beyond."""
        parts = pipeline_output["parts"]
        base = next(p for p in parts if p.role == "base")
        bb_base = base.mesh.bounds
        for part in parts:
            if part.role in ("base", "tab"):
                continue
            bb = part.mesh.bounds
            assert bb[0, 0] >= bb_base[0, 0] - EPS_MM, f"{part.name} extends left of base"
            assert bb[0, 1] >= bb_base[0, 1] - EPS_MM, f"{part.name} extends below base"
            assert bb[1, 0] <= bb_base[1, 0] + EPS_MM, f"{part.name} extends right of base"
            assert bb[1, 1] <= bb_base[1, 1] + EPS_MM, f"{part.name} extends above base"

    def test_layer_z_stacking(self, pipeline_output):
        """Lines and colors sit on the top surface of the base, lines extend higher.
        Tab is a separate object at the same Z as the base, so skip it here."""
        parts = pipeline_output["parts"]
        base = next(p for p in parts if p.role == "base")
        top_of_base = base.mesh.bounds[1, 2]

        for part in parts:
            if part.role in ("base", "tab"):
                continue
            bb = part.mesh.bounds
            assert abs(bb[0, 2] - top_of_base) < EPS_MM, \
                f"{part.name} does not start at top of base ({bb[0, 2]} vs {top_of_base})"

    def test_tab_shares_base_z_range(self, pipeline_output):
        """The tab part must span the same Z as the base so the user can
        weld them in the slicer with zero Z drift."""
        parts = pipeline_output["parts"]
        tab = next((p for p in parts if p.role == "tab"), None)
        if tab is None:
            pytest.skip("no tab")
        base = next(p for p in parts if p.role == "base")
        tbb = tab.mesh.bounds
        bbb = base.mesh.bounds
        assert abs(tbb[0, 2] - bbb[0, 2]) < EPS_MM, "tab and base z_min differ"
        assert abs(tbb[1, 2] - bbb[1, 2]) < EPS_MM, "tab and base z_max differ"

        # Lines must protrude above colors
        lines = next((p for p in parts if p.role == "lines"), None)
        if lines is not None:
            line_top = lines.mesh.bounds[1, 2]
            for c in (p for p in parts if p.role == "color"):
                c_top = c.mesh.bounds[1, 2]
                assert line_top >= c_top - EPS_MM, \
                    f"lines top ({line_top}) should be >= color top ({c_top})"

    def test_tab_exists_and_has_hole(self, pipeline_output):
        """With tab_enabled=true the tab is a separate part that carries
        the hole. Verify: (1) a 'tab' part exists, (2) it has a hole
        (euler < 2)."""
        import yaml

        cfg = yaml.safe_load((PROJECT / "config.yaml").read_text())
        if not cfg.get("tab_enabled", True) or cfg.get("hole_type") == "none":
            pytest.skip("tab disabled or no hole")

        tab = next((p for p in pipeline_output["parts"] if p.role == "tab"), None)
        assert tab is not None, "tab part missing"
        assert tab.mesh.euler_number < 2, \
            f"tab has euler_number {tab.mesh.euler_number}; expected <2 (hole present)"

    def test_base_intact_when_tab_separate(self, pipeline_output):
        """With a separate tab, the base (silhouette) must stay intact —
        no hole pierces it. Lines and colors legitimately contain rings
        (an outline thickened in 2D is an annulus), so we don't check those.
        """
        import yaml

        cfg = yaml.safe_load((PROJECT / "config.yaml").read_text())
        if not cfg.get("tab_enabled", True):
            pytest.skip("tab disabled")

        base = next(p for p in pipeline_output["parts"] if p.role == "base")
        # Simply-connected extruded region has euler_number == 2.
        assert base.mesh.euler_number == 2, \
            f"base was pierced? euler_number = {base.mesh.euler_number}"

    def test_tab_on_configured_side(self, pipeline_output):
        """The tab must extend beyond the silhouette on the configured side."""
        import yaml

        cfg = yaml.safe_load((PROJECT / "config.yaml").read_text())
        if not cfg.get("tab_enabled", True):
            pytest.skip("tab disabled")

        tab = next((p for p in pipeline_output["parts"] if p.role == "tab"), None)
        base = next(p for p in pipeline_output["parts"] if p.role == "base")
        assert tab is not None
        tab_bb = tab.mesh.bounds
        base_bb = base.mesh.bounds
        side = cfg.get("tab_side", "top")
        if side == "top":
            assert tab_bb[1, 1] > base_bb[1, 1] + 1.0, "tab does not extend above base"
        elif side == "bottom":
            assert tab_bb[0, 1] < base_bb[0, 1] - 1.0, "tab does not extend below base"
        elif side == "left":
            assert tab_bb[0, 0] < base_bb[0, 0] - 1.0, "tab does not extend left of base"
        elif side == "right":
            assert tab_bb[1, 0] > base_bb[1, 0] + 1.0, "tab does not extend right of base"
