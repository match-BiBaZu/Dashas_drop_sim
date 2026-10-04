"""Catalogue probabilities, stable identities, complete replacement and Git recovery."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from dashas_drop_sim.catalog_export import (
    assign_ids, export_run, git, recognize_pose, sha256,
)
from dashas_drop_sim.catalog_batch import run_with_catalog
from dashas_drop_sim.config import RunConfig
from dashas_drop_sim.runner import run_experiment


@pytest.fixture
def catalogue(tmp_path):
    cad = tmp_path / "cad"
    cad.mkdir()
    mesh = cad / "Cube.stl"
    trimesh.creation.box(extents=[20, 20, 20]).export(mesh)
    (cad / "Cube.step").write_text("STEP fixture: original CAD bytes are copied, not triangulated by the exporter")
    repo = tmp_path / "catalogue"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Export test")
    (repo / "LICENSE").write_text("Test fixture")
    git(repo, "add", "LICENSE")
    git(repo, "commit", "-m", "Initial catalogue")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    git(repo, "remote", "add", "origin", str(remote))
    config = RunConfig(mesh, output_dir=tmp_path / "results", trials=2, workers=1,
                       drop_height_mm=0, length_mm=300, compute_stability=False)
    result = run_experiment(config)
    return config, result, repo, cad


def test_export_is_self_contained_and_published_with_observed_poses_only(catalogue):
    config, run, repo, cad = catalogue
    outcome = export_run(run.run_dir, repo, cad_dir=cad)
    assert outcome["push_status"] == "published"
    assert git(repo, "status", "--porcelain") == ""
    assert git(repo, "rev-parse", "origin/main") == outcome["commit"]
    folder = repo / "Cube"
    data = json.loads((folder / "poses.json").read_text())
    assert data == yaml.safe_load((folder / "poses.yaml").read_text())
    assert data["geometry"]["stl_sha256"] == sha256(config.mesh_path)
    assert data["completed_trials"] == 2
    assert len(data["poses"]) > 0
    assert sum(p["count"] for p in data["poses"]) == data["settled_trials"]
    assert sum(p["frequency_percent"] for p in data["poses"]) == 100 * data["settled_trials"] / 2
    for pose in data["poses"]:
        assert pose["count"] > 0
        assert (folder / pose["image"]).read_bytes().startswith(b"\x89PNG")
        transform = np.asarray(pose["transform_chute_from_source_mm"])
        center = np.r_[data["geometry"]["center_mass_source_mm"], 1]
        assert (transform @ center)[:3] == pytest.approx(pose["position_com_chute_mm"])
        assert recognize_pose(pose["quaternion_xyzw"], data)["pose_id"] == pose["id"]
        assert recognize_pose(-np.asarray(pose["quaternion_xyzw"]), data)["pose_id"] == pose["id"]
    assert not (folder / "roughness_events.jsonl").exists()
    assert run.summary["disturbance_trials"] == 0
    assert export_run(run.run_dir, repo, cad_dir=cad)["commit"] == outcome["commit"]


def test_cancelled_run_and_modified_geometry_preserve_previous_catalogue(catalogue):
    config, run, repo, cad = catalogue
    export_run(run.run_dir, repo, cad_dir=cad)
    original = (repo / "Cube" / "poses.json").read_bytes()
    summary_path = run.run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["cancelled"] = True
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="vollständig"):
        export_run(run.run_dir, repo, cad_dir=cad)
    assert (repo / "Cube" / "poses.json").read_bytes() == original
    summary["cancelled"] = False
    summary_path.write_text(json.dumps(summary))
    config.mesh_path.write_bytes(config.mesh_path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="Eingabegeometrie"):
        export_run(run.run_dir, repo, cad_dir=cad)
    assert (repo / "Cube" / "poses.json").read_bytes() == original


def test_retry_after_push_failure_does_not_simulate_or_renumber(catalogue):
    _config, run, repo, cad = catalogue
    good_remote = git(repo, "remote", "get-url", "origin")
    git(repo, "remote", "set-url", "origin", str(repo.parent / "missing.git"))
    pending = export_run(run.run_dir, repo, cad_dir=cad)
    assert pending["push_status"] == "pending"
    before = (repo / "Cube" / "poses.json").read_bytes()
    git(repo, "remote", "set-url", "origin", good_remote)
    published = export_run(run.run_dir, repo, cad_dir=cad)
    assert published["push_status"] == "published"
    assert published["commit"] == pending["commit"]
    assert (repo / "Cube" / "poses.json").read_bytes() == before


def test_resume_reuses_finished_physics_but_retries_export(catalogue, monkeypatch):
    config, _run, repo, cad = catalogue
    configured = replace(config, catalog_repo=repo, catalog_cad_dir=cad)
    first = run_with_catalog(configured)
    monkeypatch.setattr("dashas_drop_sim.catalog_batch.run_experiment",
                        lambda *a, **kw: pytest.fail("Completed simulation was rerun"))
    again = run_with_catalog(configured)
    assert again.run_dir == first.run_dir


def test_uncommitted_user_changes_are_not_overwritten(catalogue):
    _config, run, repo, cad = catalogue
    export_run(run.run_dir, repo, cad_dir=cad)
    path = repo / "Cube" / "poses.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="verändert"):
        export_run(run.run_dir, repo, cad_dir=cad)


def test_identity_registry_is_independent_of_frequency_and_marks_ambiguity():
    descriptor = {"continuous_axis_part": None, "symmetry_quaternions_xyzw": [[0, 0, 0, 1]],
                  "tolerance_deg": 5, "ambiguity_margin_deg": 1}
    def group(angle):
        return [{"final_quat_xyzw": Rotation.from_euler("x", angle, degrees=True).as_quat().tolist()}]
    registry = {"entries": [], "next_id": 1}
    original = assign_ids([group(0), group(90)], registry, descriptor, "Part")
    repeated = assign_ids([group(90) * 20, group(0)], registry, descriptor, "Part")
    assert [p["id"] for p in repeated] == [original[1]["id"], original[0]["id"]]
    registry["entries"].append({"id": "Part_0100", "anchor_quaternion_xyzw": group(2)[0]["final_quat_xyzw"]})
    ambiguous = assign_ids([group(1)], registry, descriptor, "Part")[0]
    assert ambiguous["id_assignment"] == "ambiguous_new"
    assert len(ambiguous["previous_candidates"]) == 2


def test_recognition_handles_discrete_and_continuous_symmetry():
    pose = {"id": "a", "reference_quaternions_xyzw": [[0, 0, 0, 1]]}
    descriptor = {"continuous_axis_part": None, "symmetry_quaternions_xyzw": [
        [0, 0, 0, 1], Rotation.from_euler("z", 180, degrees=True).as_quat().tolist()],
        "tolerance_deg": 5, "ambiguity_margin_deg": 1}
    data = {"recognition": descriptor, "poses": [pose]}
    assert recognize_pose(Rotation.from_euler("z", 180, degrees=True).as_quat(), data)["pose_id"] == "a"
    descriptor["continuous_axis_part"] = [0, 0, 1]
    assert recognize_pose(Rotation.from_euler("z", 37, degrees=True).as_quat(), data)["pose_id"] == "a"
    assert recognize_pose(Rotation.from_euler("x", 90, degrees=True).as_quat(), data)["status"] == "unmatched"
