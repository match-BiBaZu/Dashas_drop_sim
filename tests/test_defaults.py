"""Saved defaults survive a restart and invalid input keeps the last good file."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMessageBox

from dashas_drop_sim.catalog import catalog_models
from dashas_drop_sim.gui import DropSimulationWindow


def test_save_parameters_without_model_and_reload_on_gui_restart(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "preferences" / "gui_defaults.json"
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: (_ for _ in ()).throw(AssertionError("Unexpected warning")))
    window = DropSimulationWindow(settings_path=path)
    try:
        window.trials_spin.setValue(23)
        window.workers_spin.setValue(2)
        window.seed_spin.setValue(77)
        window.belt_spin.setValue(150)
        window.height_spin.setValue(75)
        window.height_spread_spin.setValue(15)
        window.lateral_spin.setValue(-10)
        window.lateral_spread_spin.setValue(5)
        window.alpha_spin.setValue(42)
        window.beta_spin.setValue(1)
        window.length_spin.setValue(1400)
        window.timestep_spin.setValue(0.002)
        window.density_spin.setValue(1.17)
        window.belt_mu_spin.setValue(0.3)
        window.wall_mu_spin.setValue(0.25)
        window.levels_edit.setText("0, 0.1, 0.4")
        window.roughness_check.setChecked(True)
        window.roughness_spins["roughness_wall_height_mm"].setValue(0.08)
        window.roughness_spins["roughness_wall_ramp_mm"].setValue(4)
        window.roughness_spins["roughness_wall_spacing_mm"].setValue(12)
        window.roughness_spins["roughness_belt_height_mm"].setValue(0.02)
        window.output_edit.setText(str(tmp_path / "results"))
        expected = window._parameter_values()
        window.save_defaults_button.click()
        assert path.is_file()
        assert json.loads(path.read_text())["inputs"]["mesh_path"] == ""
        window.trials_spin.setValue(100)  # An unsaved edit must not become the default.
    finally:
        window.close()
    reopened = DropSimulationWindow(settings_path=path)
    try:
        assert reopened._parameter_values() == expected
        assert reopened.output_edit.text() == str(tmp_path / "results")
        assert reopened.mesh_edit.text() == ""
    finally:
        reopened.close()
        del app


def test_invalid_settings_do_not_overwrite_saved_defaults(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "gui_defaults.json"
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: warnings.append(_args[-1]))
    window = DropSimulationWindow(settings_path=path)
    try:
        window.save_defaults_button.click()
        original = path.read_bytes()
        window.levels_edit.setText("0, -1")
        window.save_defaults_button.click()
        assert warnings
        assert path.read_bytes() == original
    finally:
        window.close()
        del app


def test_invalid_saved_range_keeps_gui_usable(tmp_path):
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "gui_defaults.json"
    path.write_text(json.dumps({"schema_version": 1, "parameters": {
        "drop_height_mm": 190, "drop_height_spread_mm": 20}, "inputs": {}}))
    window = DropSimulationWindow(settings_path=path)
    try:
        assert window.height_spin.value() == 100
        assert window.height_spread_spin.value() == 0
        assert "nicht geladen" in window.log.toPlainText()
    finally:
        window.close()
        del app


def test_saved_catalogue_model_keeps_explicit_roadmap(tmp_path):
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "gui_defaults.json"
    roadmap = tmp_path / "manual_roadmap.yaml"
    window = DropSimulationWindow(settings_path=path)
    try:
        window.catalog_combo.setCurrentIndex(window.catalog_combo.findText("Qk1a"))
        window.roadmap_edit.setText(str(roadmap))
        window.save_defaults_button.click()
    finally:
        window.close()
    reopened = DropSimulationWindow(settings_path=path)
    try:
        assert reopened.mesh_edit.text() == str(catalog_models()["Qk1a"])
        assert reopened.catalog_combo.currentText() == "Qk1a"
        assert reopened.roadmap_edit.text() == str(roadmap)
    finally:
        reopened.close()
        del app
