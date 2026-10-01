"""Unit and integration checks for CAD mass and collision preparation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from dashas_drop_sim.geometry import GeometryError, prepare_part


def _l_prism_mm() -> trimesh.Trimesh:
    # A watertight, consistently wound concave six-sided extrusion.
    polygon = np.array(
        [[0, 0], [40, 0], [40, 10], [10, 10], [10, 30], [0, 30]], dtype=float
    )
    bottom = np.column_stack([polygon, np.zeros(len(polygon))])
    top = bottom.copy()
    top[:, 2] = 10.0
    cap = [[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]]
    faces = [[c, b, a] for a, b, c in cap]
    faces += [[a + 6, b + 6, c + 6] for a, b, c in cap]
    for i in range(6):
        j = (i + 1) % 6
        faces.extend([[i, j, j + 6], [i, j + 6, i + 6]])
    mesh = trimesh.Trimesh(vertices=np.vstack([bottom, top]), faces=faces, process=True)
    assert mesh.is_volume and not mesh.is_convex
    return mesh


def test_stl_box_mass_inertia_centering_and_cache(tmp_path: Path) -> None:
    source = tmp_path / "shifted_box.stl"
    box = trimesh.creation.box(extents=[20, 30, 40])
    box.apply_translation([51, -17, 23])
    box.export(source)

    part = prepare_part(source, tmp_path / "cache", 1.15)
    assert part.catalog_mesh_path == source
    assert part.mass_kg == pytest.approx(0.02 * 0.03 * 0.04 * 1150.0)
    np.testing.assert_allclose(part.center_mass_source_mm, [51, -17, 23], atol=1e-6)
    np.testing.assert_allclose(part.bounds_m, [[-.01, -.015, -.02], [.01, .015, .02]])
    expected_inertia = part.mass_kg / 12 * np.array(
        [(.03**2 + .04**2), (.02**2 + .04**2), (.02**2 + .03**2)]
    )
    np.testing.assert_allclose(np.diag(part.inertia_com_kg_m2), expected_inertia, rtol=1e-6)
    np.testing.assert_allclose(
        trimesh.load(part.visual_mesh_path, force="mesh").center_mass, [0, 0, 0], atol=1e-8
    )
    assert len(part.collision_mesh_paths) == 1
    assert part.collision_quality["method"] == "near_convex_hull"

    manifest = part.visual_mesh_path.parent / "manifest.json"
    before = manifest.stat().st_mtime_ns
    again = prepare_part(source, tmp_path / "cache", 1.15)
    assert again.visual_mesh_path == part.visual_mesh_path
    assert manifest.stat().st_mtime_ns == before
    assert json.loads(manifest.read_text())["mesh_sha256"] == part.mesh_sha256
    denser = prepare_part(source, tmp_path / "cache", 2.30)
    assert denser.visual_mesh_path != part.visual_mesh_path
    assert denser.mass_kg == pytest.approx(2 * part.mass_kg)


def test_reject_open_stl_and_invalid_density(tmp_path: Path) -> None:
    source = tmp_path / "open.stl"
    box = trimesh.creation.box(extents=[10, 20, 30])
    box.update_faces(np.arange(len(box.faces) - 1))
    box.export(source)
    with pytest.raises(GeometryError, match="watertight"):
        prepare_part(source, tmp_path / "cache", 1.15)
    with pytest.raises(GeometryError, match="Density"):
        prepare_part(source, tmp_path / "cache", 0)


def test_concave_uses_metric_coacd_and_exposes_gap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "l_prism.stl"
    _l_prism_mm().export(source)
    calls: list[dict] = []

    class Mesh:
        def __init__(self, vertices: np.ndarray, faces: np.ndarray):
            self.vertices = vertices
            self.faces = faces

    def run_coacd(mesh: Mesh, **kwargs: object) -> list[tuple[np.ndarray, np.ndarray]]:
        calls.append(kwargs)
        # The unit test checks that separate parts survive asset export.  A
        # separate integration test below exercises the actual CoACD binary.
        shape = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        hull = shape.convex_hull
        return [(hull.vertices, hull.faces), (hull.vertices, hull.faces)]

    monkeypatch.setitem(sys.modules, "coacd", SimpleNamespace(Mesh=Mesh, run_coacd=run_coacd))
    part = prepare_part(source, tmp_path / "cache", 1.15)
    assert calls and calls[0]["real_metric"] is True
    assert calls[0]["threshold"] == pytest.approx(.0005)
    assert calls[0]["seed"] == 0
    assert part.collision_quality["method"] == "coacd"
    assert part.collision_quality["source_convexity_volume_gap_fraction"] > .1
    assert len(part.collision_mesh_paths) == 2
    assert all(path.is_file() for path in part.collision_mesh_paths)


def test_concave_decomposition_failure_is_not_replaced_by_hull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "l_prism.stl"
    _l_prism_mm().export(source)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("decomposition unavailable")

    monkeypatch.setitem(
        sys.modules,
        "coacd",
        SimpleNamespace(Mesh=lambda vertices, faces: None, run_coacd=fail),
    )
    with pytest.raises(GeometryError, match="CoACD failed"):
        prepare_part(source, tmp_path / "cache", 1.15)


def test_real_coacd_decomposes_l_prism(tmp_path: Path) -> None:
    pytest.importorskip("coacd")
    source = tmp_path / "l_prism.stl"
    _l_prism_mm().export(source)
    part = prepare_part(source, tmp_path / "cache", 1.15)
    assert part.collision_quality["method"] == "coacd"
    assert len(part.collision_mesh_paths) >= 2
    assert all(trimesh.load(path, force="mesh").is_volume for path in part.collision_mesh_paths)


def test_step_tessellation_keeps_original_frame(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.gp import gp_Pnt

    source = tmp_path / "box.step"
    shape = BRepPrimAPI_MakeBox(gp_Pnt(10, 20, 30), 20, 30, 40).Shape()
    writer = STEPControl_Writer()
    assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
    assert writer.Write(str(source)) == IFSelect_RetDone

    part = prepare_part(source, tmp_path / "cache", 1.15)
    assert part.catalog_mesh_path != source and part.catalog_mesh_path.is_file()
    catalog = trimesh.load(part.catalog_mesh_path, force="mesh")
    assert catalog.is_watertight
    np.testing.assert_allclose(catalog.center_mass, [20, 35, 50], atol=1e-6)
    np.testing.assert_allclose(part.center_mass_source_mm, [20, 35, 50], atol=1e-6)
    assert part.mass_kg == pytest.approx(0.0276, rel=1e-6)
