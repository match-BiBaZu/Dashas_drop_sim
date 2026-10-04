"""End-to-end persistence, status accounting, and cancellation checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import trimesh
import mujoco.viewer
import numpy as np

from dashas_drop_sim.cli import _config
from dashas_drop_sim.config import RunConfig
from dashas_drop_sim.pose_matching import PoseMatch
from dashas_drop_sim.runner import (
    _assign_unknown_clusters,
    _frequency_rows,
    _same_pose,
    run_experiment,
)


@pytest.fixture
def cube(tmp_path: Path) -> Path:
    source = tmp_path / "cube.stl"
    trimesh.creation.box(extents=[20, 20, 20]).export(source)
    return source


def _short_config(source: Path, *, output_dir: Path | None = None) -> RunConfig:
    return RunConfig(
        mesh_path=source,
        output_dir=output_dir or source.parent / "results",
        trials=1,
        seed=42,
        belt_speed_mm_s=100,
        drop_height_mm=0,
        length_mm=500,
        disturbance_levels_mm=(0,),
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_short_real_cube_run_persists_unknown_pose_and_outputs(cube: Path) -> None:
    result = run_experiment(_short_config(cube))
    run_dir = result.run_dir
    for filename in (
        "config.json", "manifest.json", "trials.jsonl", "trials.csv",
        "frequencies.csv", "disturbances.csv", "stability.csv",
        "stability_summary.csv", "summary.json",
        "roughness_events.csv", "roughness_events.jsonl",
    ):
        assert (run_dir / filename).is_file(), filename

    trial_csv = _csv_rows(run_dir / "trials.csv")
    trial_jsonl = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(trial_csv) == len(trial_jsonl) == 1
    assert trial_csv[0]["status"] == trial_jsonl[0]["status"] == "settled"
    assert trial_csv[0]["match_status"] == trial_jsonl[0]["match_status"] == "unmatched"
    assert trial_csv[0]["pose_key"] == trial_jsonl[0]["pose_key"] == "unknown_001"
    assert trial_jsonl[0]["roadmap_pose_id"] is None

    frequencies = _csv_rows(run_dir / "frequencies.csv")
    assert len(frequencies) == 1
    assert frequencies[0]["pose_id"] == "unknown_001"
    assert int(frequencies[0]["count"]) == 1
    assert float(frequencies[0]["probability"]) == 1
    assert result.summary["completed_trials"] == result.summary["settled_trials"] == 1
    assert result.summary["pose_frequencies"][0]["pose_id"] == "unknown_001"
    assert "0.5 m" in result.summary["interpretation"]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mesh_source_sha256"]
    assert manifest["mesh_triangles_sha256"]
    assert manifest["roadmap_sha256"] is None
    assert result.summary["disturbance_trials"] == 1


def test_same_seed_repeats_release_and_end_state(cube: Path) -> None:
    config = _short_config(cube)
    first = run_experiment(config)
    second = run_experiment(config)

    first_trial = json.loads((first.run_dir / "trials.jsonl").read_text(encoding="utf-8"))
    second_trial = json.loads((second.run_dir / "trials.jsonl").read_text(encoding="utf-8"))
    for field in ("seed", "status", "initial_quat_xyzw", "final_quat_xyzw", "final_qpos", "pose_key"):
        assert first_trial[field] == second_trial[field]
    assert first.summary["pose_frequencies"] == second.summary["pose_frequencies"]


@pytest.mark.parametrize("roughness_enabled", [False, True])
def test_parallel_drops_match_serial_seeded_results(cube: Path, roughness_enabled: bool) -> None:
    serial_config = _short_config(cube, output_dir=cube.parent / "serial")
    serial_config.trials = 2
    serial_config.workers = 1
    serial_config.roughness_enabled = roughness_enabled
    parallel_config = _short_config(cube, output_dir=cube.parent / "parallel")
    parallel_config.trials = 2
    parallel_config.workers = 2
    parallel_config.roughness_enabled = roughness_enabled

    serial = run_experiment(serial_config)
    parallel = run_experiment(parallel_config)
    serial_rows = [json.loads(line) for line in (serial.run_dir / "trials.jsonl").read_text().splitlines()]
    parallel_rows = [json.loads(line) for line in (parallel.run_dir / "trials.jsonl").read_text().splitlines()]
    assert [row["trial"] for row in parallel_rows] == [0, 1]
    for left, right in zip(serial_rows, parallel_rows):
        assert left["seed"] == right["seed"]
        assert left["status"] == right["status"]
        assert left["pose_key"] == right["pose_key"]
        assert left["final_qpos"] == pytest.approx(right["final_qpos"], abs=1e-10)
        assert left["roughness_events"] == right["roughness_events"]
    assert serial.summary["pose_frequencies"] == parallel.summary["pose_frequencies"]
    assert (serial.run_dir / "roughness_events.csv").read_bytes() == (parallel.run_dir / "roughness_events.csv").read_bytes()
    if roughness_enabled:
        assert serial.summary["roughness_impulse_count"] > 0
        assert sum(serial.summary["roughness_surface_counts"].values()) == serial.summary["roughness_impulse_count"]


def test_visible_series_reuses_viewer_and_matches_headless(cube: Path, monkeypatch) -> None:
    class FakeViewer:
        def __init__(self):
            self.cam = SimpleNamespace(distance=0.0, lookat=np.zeros(3))
            self.sync_count = 0
            self.running = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def sync(self):
            self.sync_count += 1

        def is_running(self):
            return self.running

    viewers = []

    def launch(_model, _data):
        viewer = FakeViewer()
        viewers.append(viewer)
        return viewer

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr("dashas_drop_sim.physics.time.sleep", lambda _seconds: None)
    config = _short_config(cube, output_dir=cube.parent / "watched")
    config.trials = 2
    config.workers = 2
    watched = run_experiment(config, visible=True)
    config.output_dir = cube.parent / "unwatched"
    config.workers = 1
    headless = run_experiment(config)
    assert len(viewers) == 1
    assert viewers[0].sync_count > 2
    assert watched.summary["completed_trials"] == 2
    watched_rows = [json.loads(line) for line in (watched.run_dir / "trials.jsonl").read_text().splitlines()]
    headless_rows = [json.loads(line) for line in (headless.run_dir / "trials.jsonl").read_text().splitlines()]
    assert watched_rows == headless_rows


def test_closing_visible_viewer_keeps_completed_trials(cube: Path, monkeypatch) -> None:
    class FakeViewer:
        def __init__(self):
            self.cam = SimpleNamespace(distance=0.0, lookat=np.zeros(3))
            self.running = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def sync(self):
            pass

        def is_running(self):
            return self.running

    viewer = FakeViewer()
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda _model, _data: viewer)
    monkeypatch.setattr("dashas_drop_sim.physics.time.sleep", lambda _seconds: None)
    config = _short_config(cube)
    config.trials = 3

    def emit(event: dict) -> None:
        if event.get("event") == "progress" and event.get("completed") == 1:
            viewer.running = False

    result = run_experiment(config, emit=emit, visible=True)
    assert result.summary["cancelled"] is True
    assert result.summary["completed_trials"] == 1
    assert result.summary["disturbance_trials"] == 0
    assert len((result.run_dir / "trials.jsonl").read_text().splitlines()) == 1


def test_parallel_cancellation_stops_new_drop_jobs(cube: Path) -> None:
    config = _short_config(cube)
    config.trials = 4
    config.workers = 2
    stopped = False

    def emit(event: dict) -> None:
        nonlocal stopped
        if event.get("event") == "progress":
            stopped = True

    result = run_experiment(config, emit=emit, cancel=lambda: stopped)
    assert result.summary["cancelled"] is True
    assert 1 <= result.summary["completed_trials"] < 4
    assert result.summary["disturbance_trials"] == 0


def test_cancel_before_first_trial_writes_valid_partial_result(cube: Path) -> None:
    result = run_experiment(_short_config(cube), cancel=lambda: True)

    assert result.summary["cancelled"] is True
    assert result.summary["completed_trials"] == 0
    assert result.summary["pose_frequencies"] == []
    assert (result.run_dir / "trials.jsonl").read_text(encoding="utf-8") == ""
    assert _csv_rows(result.run_dir / "trials.csv") == []
    assert json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8")) == result.summary


def test_cancel_after_one_trial_preserves_completed_trial(cube: Path) -> None:
    config = _short_config(cube)
    config.trials = 2
    config.workers = 1
    stopped = False

    def emit(event: dict) -> None:
        nonlocal stopped
        if event.get("event") == "progress" and event.get("completed") == 1:
            stopped = True

    result = run_experiment(config, emit=emit, cancel=lambda: stopped)
    records = [json.loads(line) for line in (result.run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]

    assert result.summary["cancelled"] is True
    assert result.summary["completed_trials"] == len(records) == 1
    assert result.summary["disturbance_trials"] == 0
    assert records[0]["pose_key"] == "unknown_001"


def test_unsettled_trial_is_counted_without_pose_assignment() -> None:
    row = {
        "status": "unsettled", "match_status": "not_evaluated",
        "roadmap_pose_id": None, "pose_key": None,
        "final_quat_xyzw": [0, 0, 0, 1],
    }

    class NoClustering:
        def cluster_unmatched(self, _quaternions):
            assert _quaternions == []
            return []

    _assign_unknown_clusters([row], NoClustering())
    frequencies = _frequency_rows([row], total=1)
    assert row["pose_key"] is None
    assert frequencies == [{
        "pose_id": "unsettled", "category": "unassigned", "count": 1,
        "probability": 1.0, "ci_low": pytest.approx(0.20654931437723745),
        "ci_high": 1.0,
    }]


def test_unknown_disturbance_reaching_known_roadmap_pose_is_not_retained() -> None:
    source = {
        "match_status": "unmatched", "roadmap_pose_id": None,
        "final_quat_xyzw": [0, 0, 0, 1],
    }
    outcome = SimpleNamespace(status="settled", final_quat_xyzw=(0.02, 0, 0, 0.9998))

    class CrossingResolver:
        def resolve(self, _quat):
            return PoseMatch("matched", 7, 2.0)

        def cluster_unmatched(self, _quaternions):
            return [0, 0]

    assert _same_pose(source, outcome, CrossingResolver()) is False


def test_stable_intermediate_pose_change_counts_as_lost_even_if_end_returns() -> None:
    source = {"match_status": "matched", "roadmap_pose_id": 3,
              "final_quat_xyzw": [0, 0, 0, 1]}
    outcome = SimpleNamespace(
        status="settled", final_quat_xyzw=(0, 0, 0, 1),
        settled_trace_quat_xyzw=((0, 0, 0, 1), (1, 0, 0, 0), (0, 0, 0, 1)),
    )

    class TwoPoses:
        def resolve(self, quat):
            return PoseMatch("matched", 7 if quat[0] else 3, 0.0)

    assert _same_pose(source, outcome, TwoPoses()) is False


def test_known_roadmap_pose_without_hits_has_zero_count_and_interval() -> None:
    rows = [{"status": "unsettled", "match_status": "not_evaluated",
             "roadmap_pose_id": None, "pose_key": "unsettled"}]
    frequencies = _frequency_rows(rows, total=1, known_pose_ids=(3,))
    zero = next(row for row in frequencies if row["pose_id"] == "3")
    assert zero["category"] == "pose"
    assert zero["count"] == 0
    assert zero["ci_low"] == 0
    assert zero["ci_high"] > 0


def test_cli_config_resolves_paths_relative_to_config_file(cube: Path) -> None:
    config_path = cube.parent / "config.json"
    config_path.write_text(json.dumps({
        "mesh_path": cube.name,
        "output_dir": "results",
        "trials": 1,
    }), encoding="utf-8")

    config = _config(config_path)
    assert config.mesh_path == cube
    assert config.output_dir == cube.parent / "results"
