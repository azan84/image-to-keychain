"""End-to-end pipeline tests on the sample chibi image.

These are slow-ish (few seconds) because they run the full pipeline, so
they're marked `@pytest.mark.slow` for opt-in skipping.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import trimesh
import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SAMPLE_PNG = (PROJECT / ".." / "Keychain" / "Gemini_Generated_Image_k3zkqmk3zkqmk3zk.png").resolve()


def _load_cfg() -> dict:
    cfg = yaml.safe_load((PROJECT / "config.yaml").read_text())
    cfg["input_image"] = str(SAMPLE_PNG)
    cfg["verbose"] = False
    return cfg


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory) -> dict:
    """Run the full pipeline once and share the result across tests."""
    import sys
    sys.path.insert(0, str(PROJECT))
    from pipeline.extract_colors import extract_colors
    from pipeline.extract_lines import extract_lines
    from pipeline.export_3mf import export_3mf
    from pipeline.export_stl import export_stl_parts
    from pipeline.extrude import build_parts, compute_px_to_mm
    from pipeline.keyhole import build_tab_and_hole
    from pipeline.preprocess import preprocess
    from pipeline.silhouette import build_silhouette

    if not SAMPLE_PNG.exists():
        pytest.skip(f"sample image not found: {SAMPLE_PNG}")

    cfg = _load_cfg()
    out_dir = tmp_path_factory.mktemp("out")
    inter_dir = tmp_path_factory.mktemp("intermediate")

    pre = preprocess(SAMPLE_PNG, cfg)
    lines = extract_lines(SAMPLE_PNG, inter_dir, cfg)
    colors = extract_colors(pre, inter_dir, verbose=False)
    sil = build_silhouette(pre, inter_dir, verbose=False)
    xform = compute_px_to_mm(sil.image_shape, float(cfg["target_size_mm"]), sil.polygon)
    th = build_tab_and_hole(xform["bounds_mm"], cfg, verbose=False)
    parts = build_parts(sil.polygon, lines.polygon, colors, th.hole,
                        sil.image_shape, cfg, tab_mp=th.tab, verbose=False)

    export_stl_parts(parts, out_dir, "test_keychain", verbose=False)
    export_3mf(parts, out_dir, "test_keychain", verbose=False)

    return {
        "parts": parts,
        "out_dir": out_dir,
        "target_size_mm": float(cfg["target_size_mm"]),
        "base_thickness": float(cfg["base_thickness"]),
        "line_thickness": float(cfg["line_thickness"]),
        "color_thickness": float(cfg["color_thickness"]),
    }


class TestPipeline:

    def test_parts_created(self, pipeline_output):
        parts = pipeline_output["parts"]
        roles = [p.role for p in parts]
        assert "base" in roles
        assert "lines" in roles
        assert roles.count("color") >= 1

    def test_every_part_watertight(self, pipeline_output):
        for part in pipeline_output["parts"]:
            assert part.mesh is not None
            assert part.mesh.is_watertight, f"{part.name} not watertight"
            assert part.mesh.is_winding_consistent, f"{part.name} winding inconsistent"
            assert part.mesh.volume > 0

    def test_base_thickness(self, pipeline_output):
        base = next(p for p in pipeline_output["parts"] if p.role == "base")
        assert abs(base.thickness - pipeline_output["base_thickness"]) < 1e-6
        assert abs(base.z_min - 0.0) < 1e-6

    def test_line_thickness_and_z_offset(self, pipeline_output):
        lines = next(p for p in pipeline_output["parts"] if p.role == "lines")
        assert abs(lines.thickness - pipeline_output["line_thickness"]) < 1e-6
        assert abs(lines.z_min - pipeline_output["base_thickness"]) < 1e-6

    def test_color_thickness_and_z_offset(self, pipeline_output):
        colors = [p for p in pipeline_output["parts"] if p.role == "color"]
        assert len(colors) >= 1
        for c in colors:
            assert abs(c.thickness - pipeline_output["color_thickness"]) < 1e-6
            assert abs(c.z_min - pipeline_output["base_thickness"]) < 1e-6

    def test_stl_files_written(self, pipeline_output):
        out = pipeline_output["out_dir"]
        parts_dir = out / "test_keychain_parts"
        assert parts_dir.exists()
        stl_files = list(parts_dir.glob("*.stl"))
        assert len(stl_files) == len(pipeline_output["parts"])
        # Every STL must load back as a valid mesh
        for f in stl_files:
            m = trimesh.load(str(f))
            assert isinstance(m, trimesh.Trimesh)
            assert len(m.vertices) > 0
            assert len(m.faces) > 0

    def test_3mf_file_written_with_all_parts(self, pipeline_output):
        out = pipeline_output["out_dir"]
        mf = out / "test_keychain.3mf"
        assert mf.exists(), "3MF not written"
        assert mf.stat().st_size > 1024
        scene = trimesh.load(str(mf))
        # 3MF loads as a Scene with separate geometries — critical for AMS
        assert hasattr(scene, "geometry"), "3MF did not load as multi-object Scene"
        assert len(scene.geometry) == len(pipeline_output["parts"])
