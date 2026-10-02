"""Roadmap mapping must preserve unknown modes and catalogue equivalences."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from dashas_drop_sim.pose_matching import PoseResolver


def _fake_chute_pose(monkeypatch, *, continuous_axis=None):
    module = ModuleType("chute_pose")
    quats = {
        101: Rotation.identity().as_quat(),
        102: Rotation.from_euler("x", 30, degrees=True).as_quat(),
        103: Rotation.from_euler("x", 10, degrees=True).as_quat(),
        104: Rotation.from_euler("x", 90, degrees=True).as_quat(),
    }
    poses = tuple(
        SimpleNamespace(pose_id=pose_id, quaternion_xyzw=quat)
        for pose_id, quat in quats.items()
    )
    module.build_pose_catalog = lambda _path: SimpleNamespace(poses=poses)
    z_half_turn = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    module.detect_rotational_symmetry = lambda _path, **_kwargs: SimpleNamespace(
        symbol="Cinf" if continuous_axis is not None else "C2",
        continuous_axis_part=continuous_axis,
        elements=tuple(
            SimpleNamespace(rotation_part_from_part=matrix)
            for matrix in (np.eye(3), z_half_turn)
        ),
    )
    monkeypatch.setitem(sys.modules, "chute_pose", module)


def _roadmap_file(tmp_path, mesh, suffix, nodes):
    path = tmp_path / f"roadmap{suffix}"
    if suffix == ".json":
        data = {
            "source": str(mesh),
            "nodes": [
                {"node_id": node_id, "pose_ids": pose_ids}
                for node_id, pose_ids in nodes
            ],
        }
        path.write_text(json.dumps(data), encoding="utf-8")
    else:
        data = {
            "part": {"mesh_source": str(mesh)},
            "poses": [
                {"id": node_id, "equivalent_catalog_pose_ids": pose_ids}
                for node_id, pose_ids in nodes
            ],
        }
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_uses_every_catalogue_orientation_and_part_symmetry(monkeypatch, tmp_path, suffix):
    _fake_chute_pose(monkeypatch)
    mesh = tmp_path / "part.stl"
    mesh.write_bytes(b"valid STL content is supplied by the geometry stage")
    roadmap = _roadmap_file(tmp_path, mesh, suffix, [(5, [101, 102]), (9, [104])])
    resolver = PoseResolver(mesh, roadmap)

    second_catalogue_orientation = Rotation.from_euler("x", 30, degrees=True).as_quat()
    assert resolver.resolve(second_catalogue_orientation).pose_id == 5
    symmetric_orientation = Rotation.from_euler("z", 180, degrees=True).as_quat()
    assert resolver.resolve(symmetric_orientation).pose_id == 5
    assert resolver.reference_quaternion(5) is not None
    unknown = resolver.resolve(Rotation.from_euler("x", 50, degrees=True).as_quat())
    assert unknown.status == "unmatched"
    assert unknown.pose_id is None
    assert unknown.distance_deg == pytest.approx(20)


def test_ambiguous_orientation_is_not_forced_to_one_node(monkeypatch, tmp_path):
    _fake_chute_pose(monkeypatch)
    mesh = tmp_path / "part.stl"
    mesh.write_bytes(b"mesh")
    roadmap = _roadmap_file(tmp_path, mesh, ".yaml", [(5, [101]), (6, [103])])
    resolver = PoseResolver(mesh, roadmap)

    result = resolver.resolve(Rotation.from_euler("x", 5, degrees=True).as_quat())
    assert result.status == "ambiguous"
    assert result.pose_id is None
    assert result.distance_deg == pytest.approx(5)


def test_existing_roadmap_mesh_source_must_match_selected_stl(monkeypatch, tmp_path):
    _fake_chute_pose(monkeypatch)
    original = tmp_path / "original.stl"
    original.write_bytes(b"original")
    selected = tmp_path / "selected.stl"
    selected.write_bytes(b"different")
    roadmap = _roadmap_file(tmp_path, original, ".json", [(5, [101])])

    with pytest.raises(ValueError, match="SHA-256"):
        PoseResolver(selected, roadmap)


def test_step_original_cannot_reuse_stl_catalogue_ids(monkeypatch, tmp_path):
    _fake_chute_pose(monkeypatch)
    generated = tmp_path / "generated.stl"
    generated.write_bytes(b"triangulated")
    step = tmp_path / "part.step"
    step.write_bytes(b"STEP source")
    roadmap = _roadmap_file(tmp_path, generated, ".yaml", [(5, [101])])

    with pytest.raises(ValueError, match="original STL"):
        PoseResolver(generated, roadmap, original_mesh_path=step)


def test_unknown_pose_clusters_are_complete_link_and_order_stable(monkeypatch, tmp_path):
    _fake_chute_pose(monkeypatch)
    mesh = tmp_path / "part.stl"
    mesh.write_bytes(b"mesh")
    resolver = PoseResolver(mesh, None, cluster_tolerance_deg=5)
    angles = [0, 4, 8, 180]
    quats = Rotation.from_euler("z", [[angle] for angle in angles], degrees=True).as_quat()

    assert resolver.resolve(quats[0]).status == "unmatched"
    labels = resolver.cluster_unmatched(quats)
    assert labels[0] == labels[3]
    assert labels[2] != labels[0]
    assert set(labels) == {0, 1}
    permutation = [3, 2, 0, 1]
    permuted = resolver.cluster_unmatched(quats[permutation])
    assert permuted == [labels[index] for index in permutation]


def test_continuous_symmetry_ignores_spin_around_part_axis(monkeypatch, tmp_path):
    _fake_chute_pose(monkeypatch, continuous_axis=(0, 0, 1))
    mesh = tmp_path / "part.stl"
    mesh.write_bytes(b"mesh")
    resolver = PoseResolver(mesh, None)
    quats = Rotation.from_euler("z", [[0], [70], [140]], degrees=True).as_quat()
    tilted = Rotation.from_euler("x", 20, degrees=True).as_quat()

    labels = resolver.cluster_unmatched(np.vstack((quats, tilted)))
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_invalid_quaternion_is_rejected(monkeypatch, tmp_path):
    _fake_chute_pose(monkeypatch)
    mesh = tmp_path / "part.stl"
    mesh.write_bytes(b"mesh")
    resolver = PoseResolver(mesh, None)

    with pytest.raises(ValueError, match="zero quaternion"):
        resolver.resolve([0, 0, 0, 0])


def test_cached_catalogue_preserves_matching_without_rebuilding(monkeypatch, tmp_path):
    _fake_chute_pose(monkeypatch)
    mesh = tmp_path / "part.stl"
    mesh.write_bytes(b"same mesh")
    roadmap = _roadmap_file(tmp_path, mesh, ".yaml", [(5, [101, 102])])
    cache = tmp_path / "pose_cache"
    first = PoseResolver(mesh, roadmap, cache_dir=cache)
    assert first.cache_hit is False

    def forbidden(_path):
        raise AssertionError("The catalogue should have been read from the cache")

    monkeypatch.setattr(sys.modules["chute_pose"], "build_pose_catalog", forbidden)
    second = PoseResolver(mesh, roadmap, cache_dir=cache)
    assert second.cache_hit is True
    assert second.resolve(Rotation.from_euler("x", 30, degrees=True).as_quat()).pose_id == 5
