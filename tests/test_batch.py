"""Workpiece batches isolate settings/results and stop the queue on cancellation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import QApplication, QFileDialog, QLineEdit, QMessageBox

from dashas_drop_sim.batch import Workpiece, batch_configs, find_roadmap
from dashas_drop_sim.config import RunConfig
from dashas_drop_sim import gui


def _models(tmp_path, names):
    paths = []
    for name in names:
        path = tmp_path / f"{name}.stl"
        path.write_bytes(b"mesh validated by the geometry stage")
        paths.append(path)
    return paths


def test_configs_keep_per_workpiece_roadmaps_and_unique_outputs(tmp_path):
    first, second = _models(tmp_path, ["first", "second"])
    roadmap = tmp_path / "first_roadmap.yaml"
    roadmap.write_text("poses: []", encoding="utf-8")
    base = RunConfig(first, trials=12, workers=3, seed=77, roadmap_path=roadmap)
    configs = batch_configs(base, [Workpiece(first, roadmap), Workpiece(second)], tmp_path / "batch")

    assert [config.roadmap_path for config in configs] == [roadmap, None]
    assert len({config.output_dir for config in configs}) == 2
    assert all((config.trials, config.workers, config.seed) == (12, 3, 77) for config in configs)
    assert base.output_dir == Path("results")
    assert find_roadmap(first) == roadmap


def test_entire_queue_is_validated_before_run(tmp_path):
    first = _models(tmp_path, ["first"])[0]
    with pytest.raises(ValueError, match="Werkstück 2"):
        batch_configs(RunConfig(first), [Workpiece(first), Workpiece(tmp_path / "missing.stl")], tmp_path / "batch")
    assert not (tmp_path / "batch").exists()


_HELPER = '''
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
name = Path(config["mesh_path"]).stem
if name == "broken":
    print("Bad geometry", file=sys.stderr, flush=True)
    sys.exit(2)
run_dir = Path(config["output_dir"]) / "run_test"
run_dir.mkdir()
print(json.dumps({"event": "started", "run_dir": str(run_dir)}), flush=True)
if name == "slow":
    sys.stdin.readline()
    sys.exit(130)
summary = {"pose_frequencies": [{"pose_id": name, "count": config["trials"], "probability": 1.0}],
           "stability_ranking": [], "stability_curves": []}
(run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
print(json.dumps({"event": "done", "run_dir": str(run_dir)}), flush=True)
'''


@pytest.fixture
def window(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    helper = tmp_path / "runner.py"
    helper.write_text(_HELPER, encoding="utf-8")

    class TestProcess(QProcess):
        def setArguments(self, args):
            super().setArguments([str(helper), args[-1]])

    monkeypatch.setattr(gui, "QProcess", TestProcess)
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: (_ for _ in ()).throw(AssertionError("Unexpected warning")))
    view = gui.DropSimulationWindow()
    view.output_edit.setText(str(tmp_path / "results"))
    view.trials_spin.setValue(3)
    view.workers_spin.setValue(2)
    yield view, app
    if view._process is not None:
        view._process.kill()
        view._process.waitForFinished(2000)
    view._batch_active = False
    view.close()


def _wait(app, predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert predicate(), "Batch did not reach the expected state"


def test_multiple_file_dialog_populates_queue_and_avoids_duplicates(window, tmp_path, monkeypatch):
    view, _ = window
    paths = _models(tmp_path, ["first", "second"])
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *_args: ([str(path) for path in paths], ""))
    view.batch_add_files_button.click()
    view.batch_add_files_button.click()
    assert view.batch_table.rowCount() == 2
    view.batch_table.cellWidget(0, 1).findChild(QLineEdit).setText(str(tmp_path / "custom.yaml"))
    assert view._queued_workpieces()[0].roadmap_path == tmp_path / "custom.yaml"
    assert view.batch_start_button.isEnabled()


def test_queue_continues_after_failure_and_keeps_results_per_model(window, tmp_path):
    view, app = window
    paths = _models(tmp_path, ["first", "broken", "second"])
    view._add_workpieces([Workpiece(path) for path in paths])
    view._start_batch()
    assert not view.batch_add_files_button.isEnabled()
    _wait(app, lambda: not view._batch_active)

    assert [record["status"] for record in view._batch_records] == ["completed", "failed", "completed"]
    assert view.batch_start_button.isEnabled()
    saved = json.loads((view._batch_dir / "batch.json").read_text(encoding="utf-8"))
    assert saved["status"] == "finished"
    assert len({record["config"]["output_dir"] for record in saved["workpieces"]}) == 3
    view.batch_table.cellWidget(0, 3).click()
    assert view.result_table.item(0, 0).text() == "first"
    assert view.mesh_edit.text() == str(paths[0])
    view.batch_table.cellWidget(2, 3).click()
    assert view.result_table.item(0, 0).text() == "second"


def test_cancel_stops_current_workpiece_and_leaves_rest_unstarted(window, tmp_path):
    view, app = window
    paths = _models(tmp_path, ["slow", "second"])
    view._add_workpieces([Workpiece(path) for path in paths])
    view._start_batch()
    _wait(app, lambda: view._run_dir is not None)
    view._cancel()
    _wait(app, lambda: not view._batch_active)

    assert [record["status"] for record in view._batch_records] == ["cancelled", "pending"]
    assert not view._batch_configs[1].output_dir.exists()
    assert view.batch_table.item(1, 2).text() == "Nicht gestartet"
    saved = json.loads((view._batch_dir / "batch.json").read_text(encoding="utf-8"))
    assert saved["status"] == "cancelled"


def test_failed_process_start_advances_queue(window, tmp_path, monkeypatch):
    view, app = window

    class MissingProcess(QProcess):
        def setProgram(self, _program):
            super().setProgram(str(tmp_path / "missing-python.exe"))

    monkeypatch.setattr(gui, "QProcess", MissingProcess)
    view._add_workpieces([Workpiece(path) for path in _models(tmp_path, ["first", "second"])])
    view._start_batch()
    _wait(app, lambda: not view._batch_active)
    assert [record["status"] for record in view._batch_records] == ["failed", "failed"]
    assert view._process is None
    assert view.batch_start_button.isEnabled()


def test_two_real_workpieces_write_independent_simulation_results(window, tmp_path, monkeypatch):
    import trimesh

    view, app = window
    monkeypatch.setattr(gui, "QProcess", QProcess)
    paths = []
    for name, extents in [("cube", [20, 20, 20]), ("block", [20, 30, 20])]:
        path = tmp_path / f"{name}.stl"
        trimesh.creation.box(extents=extents).export(path)
        paths.append(path)
    view.trials_spin.setValue(1)
    view.workers_spin.setValue(1)
    view.height_spin.setValue(0)
    view.length_spin.setValue(500)
    view.levels_edit.setText("0")
    view._add_workpieces([Workpiece(path) for path in paths])
    view._start_batch()
    _wait(app, lambda: not view._batch_active, timeout=45)

    assert [record["status"] for record in view._batch_records] == ["completed", "completed"], view.log.toPlainText()
    for source, record in zip(paths, view._batch_records):
        run_dir = Path(record["run_dir"])
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        assert summary["completed_trials"] == 1
        assert config["mesh_path"] == str(source)
        assert (run_dir / "trials.csv").is_file()
        assert (run_dir / "frequencies.csv").is_file()
