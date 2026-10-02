"""Frequency renumbering must preserve all references across paired roadmaps."""

from __future__ import annotations

import hashlib
import json
import os

import pytest
import yaml

from dashas_drop_sim.roadmap_reindex import reindex_roadmaps

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _pair(tmp_path):
    yaml_path = tmp_path / "part_roadmap.yaml"
    json_path = tmp_path / "part_roadmap.json"
    yaml_path.write_text(yaml.safe_dump({
        "format": "bibazu_pose_roadmap_handover", "schema_version": 1,
        "classification": {"robust_pose_ids": [0, 2], "metastable_pose_ids": [5],
                           "unresolved_metastable_pose_ids": [5]},
        "poses": [
            {"id": 0, "equivalent_catalog_pose_ids": [100], "stability": "robust"},
            {"id": 2, "equivalent_catalog_pose_ids": [200], "stability": "robust"},
            {"id": 5, "equivalent_catalog_pose_ids": [300], "stability": "metastable"},
        ],
        "transitions": [{"id": "a0:5->2:free_y", "from_pose": 5, "to_pose": 2}],
    }, sort_keys=False), encoding="utf-8")
    json_path.write_text(json.dumps({
        "schema_version": 1,
        "nodes": [{"node_id": node_id, "pose_ids": [catalogue]}
                  for node_id, catalogue in [(0, 100), (2, 200), (5, 300)]],
        "edges": [{"edge_id": "a0:5->2:free_y", "source": 5, "target": 2,
                   "settling_pose_ids": [100]}],
        "unresolved_metastable_node_ids": [5],
    }), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "roadmap_sha256": hashlib.sha256(yaml_path.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    summary = run_dir / "summary.json"
    summary.write_text(json.dumps({"pose_frequencies": [
        {"pose_id": "0", "category": "pose", "count": 2},
        {"pose_id": "2", "category": "pose", "count": 5},
        {"pose_id": "5", "category": "pose", "count": 0},
        {"pose_id": "unknown_001", "category": "pose", "count": 20},
    ]}), encoding="utf-8")
    return yaml_path, json_path, summary


def test_reindexes_yaml_and_json_together(tmp_path):
    yaml_source, json_source, summary = _pair(tmp_path)
    original_yaml = yaml_source.read_bytes()
    original_json = json_source.read_bytes()
    destination = tmp_path / "ordered.yaml"

    mapping, backups, outputs = reindex_roadmaps(yaml_source, summary, destination)

    assert mapping == {2: 0, 0: 1, 5: 2}
    assert backups == []
    assert outputs == (destination, destination.with_suffix(".json"))
    sheet = yaml.safe_load(outputs[0].read_text(encoding="utf-8"))
    graph = json.loads(outputs[1].read_text(encoding="utf-8"))
    assert [pose["id"] for pose in sheet["poses"]] == [0, 1, 2]
    assert [node["node_id"] for node in graph["nodes"]] == [0, 1, 2]
    assert sheet["poses"][0]["equivalent_catalog_pose_ids"] == [200]
    assert graph["nodes"][0]["pose_ids"] == [200]
    assert sheet["classification"]["robust_pose_ids"] == [0, 1]
    assert sheet["classification"]["metastable_pose_ids"] == [2]
    assert graph["unresolved_metastable_node_ids"] == [2]
    assert sheet["transitions"][0]["id"] == "a0:2->0:free_y"
    assert (sheet["transitions"][0]["from_pose"], sheet["transitions"][0]["to_pose"]) == (2, 0)
    assert graph["edges"][0]["edge_id"] == "a0:2->0:free_y"
    assert graph["edges"][0]["settling_pose_ids"] == [100]
    assert yaml_source.read_bytes() == original_yaml
    assert json_source.read_bytes() == original_json


def test_overwrite_backs_up_both_files(tmp_path):
    yaml_source, json_source, summary = _pair(tmp_path)
    old_yaml = yaml_source.read_bytes()
    old_json = json_source.read_bytes()

    mapping, backups, outputs = reindex_roadmaps(json_source, summary, json_source)

    assert mapping[2] == 0
    assert outputs == (yaml_source, json_source)
    assert len(backups) == 2
    assert [backup.read_bytes() for backup in backups] == [old_yaml, old_json]


def test_rejects_results_from_different_roadmap(tmp_path):
    yaml_source, _, summary = _pair(tmp_path)
    yaml_source.write_text(yaml_source.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="different version"):
        reindex_roadmaps(yaml_source, summary, tmp_path / "ordered.yaml")


def test_equal_counts_keep_previous_id_order(tmp_path):
    yaml_source, _, summary = _pair(tmp_path)
    data = json.loads(summary.read_text(encoding="utf-8"))
    data["pose_frequencies"][0]["count"] = 5
    summary.write_text(json.dumps(data), encoding="utf-8")

    mapping, _, _ = reindex_roadmaps(yaml_source, summary, tmp_path / "ordered.yaml")

    assert mapping == {0: 0, 2: 1, 5: 2}


def test_gui_action_saves_both_roadmaps(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
    from dashas_drop_sim.gui import DropSimulationWindow

    yaml_source, _, summary = _pair(tmp_path)
    destination = tmp_path / "ordered.yaml"
    app = QApplication.instance() or QApplication([])
    window = DropSimulationWindow()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: (str(summary), ""))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(destination), ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *_args: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: (_ for _ in ()).throw(AssertionError("Unexpected warning")))
    try:
        window.roadmap_edit.setText(str(yaml_source))
        window.reindex_button.click()
        assert destination.is_file()
        assert destination.with_suffix(".json").is_file()
        assert window.roadmap_edit.text() == str(destination)
    finally:
        window.close()
        del app
