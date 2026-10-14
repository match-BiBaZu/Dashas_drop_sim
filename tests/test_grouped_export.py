"""Grouped handovers preserve source identities, counts and recognition coverage."""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from dashas_drop_sim.catalog_export import recognize_pose, sha256
from dashas_drop_sim.grouped_export import (archive_completed_run, audit_run, export_grouped_poses, read,
                                           verify_inventory)
from dashas_drop_sim.runner import _write_json, _write_jsonl, _write_csv


@pytest.fixture
def grouped_run(tmp_path):
    cad, run, repo = (tmp_path / p for p in ("cad", "run", "repo"))
    for p in (cad, run, repo):
        p.mkdir()
    mesh = cad / "Cube.stl"
    trimesh.creation.box(extents=[20, 20, 20]).export(mesh)
    (cad / "Cube.step").write_text("source CAD fixture")
    levels = [0, .05, .1, .2, .4, .5, .8]
    keys = ["Cube_0001", "Cube_0002", "unknown_001"]
    quats = [Rotation.from_euler("x", angle, degrees=True).as_quat().tolist() for angle in (0, 2, 70)]
    members = [{"id": key, "count": count, "quaternion_xyzw": q, "reference_quaternions_xyzw": [q],
                "position_com_chute_mm": [10, 10, 10], "source_trial": i, "group_id": f"group_{i // 2:03d}"}
               for i, (key, count, q) in enumerate(zip(keys, (1, 1, 2), quats))]
    groups = [{"pose_id": "group_000", "members": keys[:2], "representative_source_pose_id": keys[0],
               "count": 2, "probability": .5, "ci_low": .15, "ci_high": .85, "retention_auc": 1.,
               "retained_by_frequency": True, "complete": True, "rank": 1},
              {"pose_id": "group_001", "members": keys[2:], "representative_source_pose_id": keys[2],
               "count": 2, "probability": .5, "ci_low": .15, "ci_high": .85, "retention_auc": 1.,
               "retained_by_frequency": True, "complete": True, "rank": 2}]
    children = np.random.SeedSequence(42).spawn(4)
    trials = [{"trial": i, "seed": int(children[i].generate_state(1, dtype=np.uint32)[0]), "status": "settled",
               "pose_key": keys[min(i, 2)]} for i in range(4)]
    curves = [{"pose_id": key, "energy_lift_mm": level, "retention": 1., "complete": True}
              for key in keys for level in levels]
    disturbances = [{"source_pose_id": key, "energy_lift_mm": level, "direction_index": i}
                    for key in keys for level in levels for i in range(1 if level == 0 else 12)]
    for i, row in enumerate(disturbances):
        row["disturbance_index"] = i
    descriptor = {"tolerance_deg": 5, "ambiguity_margin_deg": 1, "continuous_axis_part": None,
                  "symmetry_quaternions_xyzw": [[0, 0, 0, 1]], "symmetry_available": True}
    baseline = {"workpiece": "Cube", "geometry": {"stl_sha256": sha256(mesh)}, "poses": members[:2]}
    registry = {"schema_version": 1, "mesh_sha256": sha256(mesh), "next_id": 3,
                "entries": [{"id": m["id"], "anchor_quaternion_xyzw": m["quaternion_xyzw"]} for m in members[:2]]}
    _write_json(run / "summary.json", {"cancelled": False, "completed_trials": 4, "settled_trials": 4,
                "status_counts": {"settled": 4}, "group_assessment_complete": True, "group_stability_curves": curves})
    _write_json(run / "config.json", {"seed": 42, "mesh_path": str(mesh), "length_mm": 500,
                "disturbance_levels_mm": levels, "grouping_barrier_mm": .5, "grouping_max_angle_deg": 60,
                "grouping_minimum_frequency_percent": 1})
    _write_json(run / "manifest.json", {"mesh_source_sha256": sha256(mesh), "center_mass_source_mm": [0, 0, 0],
                                        "recognition": descriptor})
    _write_json(run / "group_members.json", members)
    _write_json(run / "groups.json", {"groups": groups, "assessment_complete": True})
    _write_jsonl(run / "trials.jsonl", trials)
    _write_jsonl(run / "group_disturbances.jsonl", disturbances)
    _write_json(run / "grouping_geometry.json", {"groups": [g["members"] for g in groups], "paths": [], "review": []})
    for name in ("group_stability.csv", "group_stability_summary.csv", "group_frequencies.csv", "frequencies.csv", "stability.csv"):
        _write_csv(run / name, ["pose_id"], [{"pose_id": k} for k in keys])
    return run, repo, baseline, registry, cad


def test_grouped_export_keeps_counts_member_images_recognition_and_registry(grouped_run):
    run, repo, baseline, registry, cad = grouped_run
    data, audit = export_grouped_poses(run, repo, baseline, registry, cad_dir=cad, expected_trials=4)
    folder = repo / "Cube"
    assert audit["disturbance_tests"] == 219
    assert sum(p["count"] for p in data["poses"]) == 4
    assert sum(p["frequency_percent"] for p in data["poses"]) == 100
    assert {p["id"] for p in data["poses"]} == {"Cube_0001", "Cube_0003"}
    first = next(p for p in data["poses"] if p["id"] == "Cube_0001")
    assert first["individual_count"] == 1 and first["group_members"] == ["Cube_0001", "Cube_0002"]
    assert first["trial_ids"] == [0, 1] and first["individual_trial_ids"] == [0]
    assert first["representative_trial"] == 0
    originals = read(folder / "original_observations.json")["poses"]
    assert sum(len(p["trial_ids"]) for p in originals) == 4
    assert recognize_pose(baseline["poses"][1]["quaternion_xyzw"], data)["pose_id"] == "Cube_0001"
    assert {e["id"] for e in read(folder / "pose_registry.json")["entries"]} == {"Cube_0001", "Cube_0003"}
    assert read(folder / "original_pose_registry.json") == registry
    assert len(read(folder / "original_observations.json")["poses"]) == 3
    assert len(list((folder / "images").glob("*.png"))) == 3
    assert read(folder / "poses.json") == yaml.safe_load((folder / "poses.yaml").read_text())
    verify_inventory(folder)


def test_incomplete_assessment_and_corrupt_checkpoint_cannot_replace_export(grouped_run):
    run, repo, baseline, registry, cad = grouped_run
    export_grouped_poses(run, repo, baseline, registry, cad_dir=cad, expected_trials=4)
    before = (repo / "Cube/poses.json").read_bytes()
    summary = read(run / "summary.json")
    summary["group_assessment_complete"] = False
    _write_json(run / "summary.json", summary)
    with pytest.raises(ValueError, match="Incomplete"):
        export_grouped_poses(run, repo, baseline, registry, cad_dir=cad, expected_trials=4)
    assert (repo / "Cube/poses.json").read_bytes() == before
    summary["group_assessment_complete"] = True
    _write_json(run / "summary.json", summary)
    with (run / "group_disturbances.jsonl").open("a") as stream:
        stream.write(json.dumps({"source_pose_id": "Cube_0001", "energy_lift_mm": 0, "direction_index": 0}) + "\n")
    with pytest.raises(ValueError, match="duplicated"):
        audit_run(run, 4)


def test_no_eligible_groups_removes_stale_current_poses(grouped_run):
    run, repo, baseline, registry, cad = grouped_run
    export_grouped_poses(run, repo, baseline, registry, cad_dir=cad, expected_trials=4)
    groups = read(run / "groups.json")
    for group in groups["groups"]:
        group["retained_by_frequency"] = False
    _write_json(run / "groups.json", groups)
    data, _ = export_grouped_poses(run, repo, baseline, registry, cad_dir=cad, expected_trials=4)
    assert data["poses"] == []
    assert read(repo / "Cube/pose_registry.json")["entries"] == []
    assert len(read(repo / "Cube/original_observations.json")["poses"]) == 3
    assert "no groups" in (repo / "Cube/README.md").read_text()
    verify_inventory(repo / "Cube")


def test_completed_raw_data_archive_is_lossless_and_remains_auditable(grouped_run):
    run, *_ = grouped_run
    before = audit_run(run, 4)
    archive_completed_run(run, run.parent)
    assert not (run / "trials.jsonl").exists()
    assert (run / "trials.jsonl.gz").is_file()
    assert audit_run(run, 4) == before
    assert archive_completed_run(run, run.parent) == read(run / "archive.json")
    with pytest.raises(ValueError, match="inside"):
        archive_completed_run(run, run / "other")
