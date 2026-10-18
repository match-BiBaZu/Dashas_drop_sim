"""The bundled catalogue and pose view work without a network connection."""

from __future__ import annotations

import os
import json
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox
import trimesh

from dashas_drop_sim.catalog import catalog_models
from dashas_drop_sim.gui import DropSimulationWindow
from dashas_drop_sim.pose_image import render_pose_image


def test_catalogue_dropdown_keeps_browse_path_and_1300_mm_default(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = DropSimulationWindow(settings_path=tmp_path / "gui_defaults.json")
    try:
        models = catalog_models()
        assert len(models) == 39
        assert window.catalog_combo.count() == len(models) + 1
        index = window.catalog_combo.findText("Qk1a")
        assert index > 0
        window.catalog_combo.setCurrentIndex(index)
        assert window.mesh_edit.text() == str(models["Qk1a"])
        assert window.length_spin.value() == 1300
        assert window.workers_spin.value() >= 1
        assert window.preview_button.text() == "Falltests sichtbar starten"
    finally:
        window.close()
        del app


def test_pose_sketch_contains_part_and_both_chute_surfaces() -> None:
    app = QApplication.instance() or QApplication([])
    path = catalog_models()["Qk1a"]
    mesh = trimesh.load_mesh(path, force="mesh")
    image = render_pose_image(path, mesh.center_mass, [0, 0, 0, 1])
    assert (image.width(), image.height()) == (520, 380)
    colors = {image.pixel(x, y) for x in range(20, 500, 20) for y in range(20, 360, 20)}
    assert len(colors) > 8
    del app


def test_result_row_opens_pose_image_dialog(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    mesh_path = tmp_path / "cube.stl"
    trimesh.creation.box(extents=[20, 20, 20]).export(mesh_path)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "catalog_mesh_path": str(mesh_path),
        "center_mass_source_mm": [0, 0, 0],
    }), encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"mesh_path": str(mesh_path)}), encoding="utf-8")
    titles: list[str] = []
    monkeypatch.setattr(QDialog, "exec", lambda dialog: titles.append(dialog.windowTitle()) or 0)
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: (_ for _ in ()).throw(AssertionError("Unexpected GUI error")))
    window = DropSimulationWindow(settings_path=tmp_path / "gui_defaults.json")
    try:
        window._run_dir = tmp_path
        window._show_pose({
            "pose_id": "unknown_001",
            "representative_quat_xyzw": "[0, 0, 0, 1]",
            "representative_pos_chute_mm": "[100, 10, 10]",
            "representative_source": "observed",
        })
        window._show_pose({
            "pose_id": "8",
            "representative_quat_xyzw": "[0, 0, 0, 1]",
            "representative_pos_chute_mm": "",
            "representative_source": "catalogue",
        })
        assert titles == ["Pose unknown_001", "Pose 8"]
    finally:
        window.close()
        del app


def test_visible_series_event_updates_last_trial(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = DropSimulationWindow(settings_path=tmp_path / "gui_defaults.json")
    try:
        window._mode = "watch"
        assert window._handle_event({"event": "started", "run_dir": str(tmp_path)})
        assert window._run_dir == tmp_path
        assert window._handle_event({
            "event": "progress", "completed": 2, "total": 3,
            "record": {"trial": 1, "status": "settled", "seed": 123,
                       "final_quat_xyzw": [0, 0, 0, 1]},
        })
        assert window.progress.value() == 2
        assert window._preview_record["trial"] == 1
        assert window.preview_image_button.isEnabled()
        assert window.preview_table.item(0, 1).text() == "2"
    finally:
        window.close()
        del app


def test_belt_stop_gui_settings_and_results(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = DropSimulationWindow(settings_path=tmp_path / 'gui_defaults.json')
    try:
        assert window.belt_stop_check.isChecked()
        window.belt_stop_ramp_spin.setValue(0.3)
        window.belt_stop_wait_spin.setValue(3)
        values = window._parameter_values()
        assert values['belt_stop_enabled'] is True
        assert values['belt_stop_ramp_s'] == 0.3
        assert values['belt_stop_observation_s'] == 3
        window._save_defaults()
        reloaded = DropSimulationWindow(settings_path=tmp_path / 'gui_defaults.json')
        try:
            assert reloaded.belt_stop_check.isChecked()
            assert reloaded.belt_stop_ramp_spin.value() == 0.3
            assert reloaded.belt_stop_wait_spin.value() == 3
        finally:
            reloaded.close()
        window.belt_stop_check.setChecked(False)
        assert not window.belt_stop_wait_spin.isEnabled()
        record = {'trial': 0, 'transport_status': 'unsettled', 'status': 'settled',
                  'orientation_change_deg': 30, 'coast_distance_mm': 12,
                  'final_quat_xyzw': '[0,0,0,1]', 'final_pos_chute_mm': '[100,10,10]'}
        (tmp_path / 'stop_observations.csv').write_text(
            'trial,transport_status,status,orientation_change_deg,coast_distance_mm\n0,unsettled,settled,30,12\n')
        window._run_started_at = 0
        window._load_stop_results(tmp_path)
        assert '1 / 1' in window.stop_summary_label.text()
        captured = []
        monkeypatch.setattr(window, '_show_pose', captured.append)
        window._set_stop_rows([record])
        window.stop_table.cellWidget(0, 6).click()
        assert captured[0]['representative_quat_xyzw'] == record['final_quat_xyzw']
        assert window.stop_table.item(0, 1).text() == 'unsettled'
        assert window.stop_table.item(0, 2).text() == 'settled'
    finally:
        window.close()
        del app


def test_gui_roughness_settings_reach_simulation_config(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = DropSimulationWindow(settings_path=tmp_path / "gui_defaults.json")
    try:
        assert window.roughness_check.isChecked()
        assert window.roughness_model_combo.currentData() == "weight_impulse"
        assert window.impulse_delta_v_spin.value() == 49.05
        assert window.force_interval_spin.value() == 0.2
        assert window.roughness_duration_spin.value() == 1.0
        assert window.roughness_spins["roughness_wall_height_mm"].value() == 0.1
        assert window.roughness_spins["roughness_wall_spacing_mm"].value() == 10
        assert window.roughness_spins["roughness_belt_height_mm"].value() == 0
        window.mesh_edit.setText(str(catalog_models()["Qk1a"]))
        window.output_edit.setText(str(tmp_path))
        window.roughness_check.setChecked(True)
        window.roughness_spins["roughness_belt_height_mm"].setValue(0.05)
        config = window._make_config()
        assert config.roughness_enabled
        assert config.roughness_wall_ramp_mm == 3
        assert config.roughness_belt_height_mm == 0.05
    finally:
        window.close()
        del app
