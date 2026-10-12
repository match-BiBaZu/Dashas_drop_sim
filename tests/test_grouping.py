"""Published examples and failure modes of directed operational pose basins."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dashas_drop_sim.catalog import catalog_models
from dashas_drop_sim.grouping import analyze_groups, basin_memberships, choose_representative, relative_path
from dashas_drop_sim.observed import ObservedPoseResolver
from dashas_drop_sim.grouped_run import retained_group
from dashas_drop_sim.config import RunConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("part,expected", [
    ("Qk1a", [["0"], ["1"], ["2"], ["3"], ["4", "8", "9"], ["5", "6", "7"]]),
    ("Dl1a", [["0", "3"], ["1", "2", "5"], ["4"], ["6"]]),
    ("Dk1i", [["0"], ["1", "5"], ["2", "4"], ["3"], ["6"], ["7"]]),
])
def test_published_pose_groups(part, expected):
    fixture = json.loads((ROOT / "experiments/grouped_pose_study/baseline" / f"{part}_roadmap_fixture.json").read_text())
    result = analyze_groups(catalog_models()[part], fixture["members"], fixture["recognition"])
    assert result["groups"] == expected
    if part == "Dl1a":
        path = next(p for p in result["paths"] if (p["source"], p["target"]) == ("1", "5"))
        assert path["forward_barrier_mm"] > 1.8
        assert path["reverse_barrier_mm"] < .02


def path(a, b, angle=20, forward=.1, reverse=.1):
    return dict(source=a, target=b, angle_deg=angle, forward_barrier_mm=forward, reverse_barrier_mm=reverse)


def test_competing_destinations_do_not_merge_stable_basins():
    groups, review = basin_memberships(["a", "b", "t"], [path("t", "a", reverse=2), path("t", "b", reverse=2), path("a", "b", forward=2, reverse=2)], barrier_mm=.5, max_angle_deg=60)
    assert groups == [["a"], ["b"], ["t"]]
    assert review


def test_chain_diameter_does_not_collapse_distinct_orientations():
    groups, review = basin_memberships(["a", "b", "c"], [path("a", "b", angle=40), path("b", "c", angle=40), path("a", "c", angle=80)], barrier_mm=.5, max_angle_deg=60)
    assert groups == [["a"], ["b"], ["c"]]
    assert review[0]["reason"] == "component_diameter_exceeds_guard"


def test_threshold_is_inclusive_and_zero_can_merge_zero_barriers():
    assert basin_memberships(["a", "b"], [path("a", "b", forward=.5, reverse=.5)], barrier_mm=.5, max_angle_deg=60)[0] == [["a", "b"]]
    assert basin_memberships(["a", "b"], [path("a", "b", forward=0, reverse=0)], barrier_mm=0, max_angle_deg=60)[0] == [["a", "b"]]


def test_symmetry_and_continuous_axis_paths():
    identity = [0, 0, 0, 1]
    half_turn = Rotation.from_euler("z", 180, degrees=True).as_quat()
    assert np.linalg.norm(relative_path(identity, half_turn, {"symmetry_quaternions_xyzw": [identity, half_turn]})) < 1e-10
    assert np.linalg.norm(relative_path(identity, half_turn, {"continuous_axis_part": [0, 0, 1]})) < 1e-10


def test_stability_selects_representative_not_popularity_and_requires_complete_scores():
    assert choose_representative(["a", "b"], {"a": .5, "b": .8}, {"a": 100, "b": 1}, {"a": .5, "b": .5}) == "b"
    assert choose_representative(["a", "b"], {"a": .5, "b": None}, {}, {}) is None
    assert choose_representative(["b", "a"], {"a": .8, "b": .8}, {"a": 1, "b": 1}, {"a": .5, "b": .5}) == "a"


def test_direct_matching_preserves_source_ids_and_group_internal_ambiguity(tmp_path):
    mesh = tmp_path / "mesh.stl"
    mesh.write_bytes(b"mesh")
    qs = Rotation.from_euler("x", [[0], [6], [40]], degrees=True).as_quat()
    data = {"poses": [{"id": name, "quaternion_xyzw": list(q)} for name, q in zip(["part_0042", "part_0008", "part_0001"], qs)]}
    resolver = ObservedPoseResolver(mesh, data=data)
    assert resolver.resolve(qs[0]).pose_id == "part_0042"
    assert resolver.resolve(Rotation.from_euler("x", 3, degrees=True).as_quat()).status == "ambiguous"
    grouped = resolver.grouped({"part_0042": "g", "part_0008": "g", "part_0001": "other"})
    assert grouped.resolve(Rotation.from_euler("x", 3, degrees=True).as_quat()).pose_id == "g"
    outcome = SimpleNamespace(status="settled", settled_trace_quat_xyzw=[qs[1]], final_quat_xyzw=qs[0])
    assert retained_group("g", outcome, grouped)
    outcome.settled_trace_quat_xyzw = [qs[2]]
    assert not retained_group("g", outcome, grouped)
    outcome.status = "unsettled"
    assert not retained_group("g", outcome, grouped)


def test_grouping_config_requires_assessment_and_disables_publication():
    config = RunConfig(mesh_path=Path("unused.stl"), grouping_enabled=True, compute_stability=False)
    with pytest.raises(ValueError, match="compute_stability"):
        config.validate_parameters()
    config.compute_stability = True
    config.validate_parameters()
    assert .5 in config.disturbance_levels_mm
    config.catalog_repo = Path("repo")
    with pytest.raises(ValueError, match="publication"):
        config.validate_parameters()


def test_grouped_cube_exports_and_resumes_without_repeating_trials(tmp_path):
    import trimesh
    from dashas_drop_sim.runner import run_experiment
    mesh = tmp_path / "cube.stl"
    trimesh.creation.box(extents=[20, 20, 20]).export(mesh)
    config = RunConfig(mesh_path=mesh, output_dir=tmp_path / "results", trials=1, workers=1,
                       drop_height_mm=0, length_mm=500, grouping_enabled=True,
                       grouping_barrier_mm=.005, disturbance_levels_mm=(0, .005))
    result = run_experiment(config)
    assert result.summary["group_assessment_complete"]
    assert result.summary["group_disturbance_trials"] == 13
    assert sum(g["count"] for g in result.summary["pose_groups"]) == result.summary["settled_trials"] == 1
    assert len(json.loads((result.run_dir / "representatives/poses.json").read_text())["poses"]) == 1
    before = (result.run_dir / "group_disturbances.jsonl").read_bytes()
    resumed = run_experiment(config, resume_run_dir=result.run_dir)
    assert resumed.summary["completed_trials"] == 1
    assert resumed.summary["pose_groups"] == result.summary["pose_groups"]
    assert (result.run_dir / "group_disturbances.jsonl").read_bytes() == before


def test_observed_roadmap_cannot_be_misread_as_geometry_indices(tmp_path):
    from dashas_drop_sim.pose_matching import PoseResolver
    mesh = tmp_path / "part.stl"
    mesh.write_bytes(b"mesh")
    roadmap = tmp_path / "roadmap.json"
    roadmap.write_text(json.dumps({"catalogue_id_namespace": "input_pose_list_index", "nodes": [{"node_id": 0, "pose_ids": [0]}]}))
    with pytest.raises(ValueError, match="observed_catalog_path"):
        PoseResolver(mesh, roadmap)
