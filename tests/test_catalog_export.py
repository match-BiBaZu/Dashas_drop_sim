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
    assign_ids, export_run, git, recognize_pose, sha256, _write_stop_pose_catalogue,
)
from dashas_drop_sim.pose_matching import PoseResolver
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


def test_export_preserves_separate_stop_observation(catalogue):
    config, _run, repo, cad = catalogue
    config.belt_stop_enabled = True
    run = run_experiment(config)
    export_run(run.run_dir, repo, cad_dir=cad, push=False)
    data = json.loads((repo / 'Cube/poses.json').read_text())
    assert data['belt_stop_observation'] == run.summary['belt_stop_observation']
    import csv
    with (repo / 'Cube/trials.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert json.loads(rows[0]['stop_observation'])['final_belt_speed_mm_s'] == 0
    assert 'Ruheprüfung nach Bandstopp' in (repo / 'Cube/README.md').read_text(encoding='utf-8')
    stop = json.loads((repo / 'Cube/poses_after_stop.json').read_text())
    assert stop == yaml.safe_load((repo / 'Cube/poses_after_stop.yaml').read_text(encoding='utf-8'))
    assert stop['evaluation_phase'] == 'after_belt_stop'
    assert stop['completed_trials'] == stop['observed_trials'] == stop['settled_trials'] == 2
    assert sum(p['count'] for p in stop['poses']) == 2
    for pose in stop['poses']:
        assert pose['id'].startswith('Cube_stop_')
        assert (repo / 'Cube' / pose['image']).is_file()
        assert recognize_pose(pose['quaternion_xyzw'], stop)['pose_id'] == pose['id']
    index = json.loads((repo / 'index.json').read_text())
    entry = next(row for row in index['workpieces'] if row['workpiece'] == 'Cube')
    assert entry['stop_settled'] == 2
    assert entry['stop_poses'] == len(stop['poses'])


def test_stop_classes_use_symmetry_and_exclude_unsettled_or_missing_observations(catalogue):
    config, run, repo, cad = catalogue
    export_run(run.run_dir, repo, cad_dir=cad, push=False)
    transport = json.loads((repo / 'Cube/poses.json').read_text())
    transport['recognition']['continuous_axis_part'] = [0, 0, 1]
    resolver = PoseResolver(config.mesh_path, None)
    resolver._continuous_axis = np.array([0, 0, 1])
    rows = []
    quats = list(Rotation.from_euler('z', [[0], [70], [140]], degrees=True).as_quat())
    quats += [Rotation.from_euler('x', 90, degrees=True).as_quat()]
    for index, quat in enumerate(quats):
        rows.append({'trial': index, 'seed': index, 'status': 'unsettled', 'stop_observation': {
            'status': 'settled', 'final_quat_xyzw': quat.tolist(), 'final_pos_chute_mm': [100, 10, 10]}})
    rows += [{'trial': 4, 'seed': 4, 'status': 'unsettled', 'stop_observation': {
        'status': 'unsettled', 'final_quat_xyzw': quats[0].tolist(), 'final_pos_chute_mm': [100, 10, 10]}},
        {'trial': 5, 'seed': 5, 'status': 'stalled', 'stop_observation': None}]
    transport['completed_trials'] = len(rows)
    folder = repo.parent / 'stop-only'
    folder.mkdir()
    result = _write_stop_pose_catalogue(folder, folder, rows, transport, resolver, config.mesh_path)
    assert sorted(p['count'] for p in result['poses']) == [1, 3]
    assert result['observed_trials'] == 5 and result['not_observed_trials'] == 1
    assert result['settled_trials'] == 4
    assert sum(p['frequency_percent'] for p in result['poses']) == pytest.approx(100 * 4 / 6)
    assert sum(p['frequency_settled_percent'] for p in result['poses']) == pytest.approx(100)
    import csv
    with (folder / 'trials_after_stop.csv').open(newline='') as stream:
        assignments = list(csv.DictReader(stream))
    assert not assignments[4]['pose_id'] and not assignments[5]['pose_id']
    assert assignments[5]['status'] == 'not_observed'


def test_retry_upgrades_legacy_stop_export_and_preserves_raw_archive(catalogue):
    config, _run, repo, cad = catalogue
    config.belt_stop_enabled = True
    run = run_experiment(config)
    export_run(run.run_dir, repo, cad_dir=cad, push=False)
    folder = repo / 'Cube'
    data = json.loads((folder / 'poses.json').read_text())
    data.pop('after_stop_catalogue')
    (folder / 'poses.json').write_text(json.dumps(data))
    (folder / 'poses.yaml').write_text(yaml.safe_dump(data))
    for name in ('poses_after_stop.json', 'poses_after_stop.yaml', 'pose_registry_after_stop.json',
                 'frequencies_after_stop.csv', 'trials_after_stop.csv'):
        (folder / name).unlink()
    for path in (folder / 'images').glob('Cube_stop_*.png'):
        path.unlink()
    archive = folder / 'simulation_results.zip'
    archive.write_bytes(b'original complete run archive fixture')
    before_trials = sha256(folder / 'trials.csv')
    checks = {p.relative_to(folder).as_posix(): sha256(p) for p in folder.rglob('*')
              if p.is_file() and p.name != 'files.sha256.json'}
    (folder / 'files.sha256.json').write_text(json.dumps(checks))
    git(repo, 'add', '--', 'Cube')
    git(repo, 'commit', '-m', 'Legacy stationary observations without pose classes')
    outcome = export_run(run.run_dir, repo, cad_dir=cad, push=False)
    assert archive.read_bytes() == b'original complete run archive fixture'
    assert sha256(folder / 'trials.csv') == before_trials
    assert json.loads((folder / 'poses_after_stop.json').read_text())['settled_trials'] == 2
    assert export_run(run.run_dir, repo, cad_dir=cad, push=False)['commit'] == outcome['commit']


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
