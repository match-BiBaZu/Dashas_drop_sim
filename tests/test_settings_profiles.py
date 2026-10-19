"""Current settings survive a fresh GUI start and reproduce the production runs."""
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from dashas_drop_sim.cli import _config
from dashas_drop_sim.gui import DropSimulationWindow
from dashas_drop_sim.settings import (
    GUI_INITIAL_PARAMETERS, PACKAGED_SETTINGS_PATH, PARAMETER_KEYS, load_defaults, save_defaults,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_packaged_settings_cover_every_parameter_and_have_no_machine_paths():
    settings = load_defaults(PACKAGED_SETTINGS_PATH)
    assert set(settings["parameters"]) == PARAMETER_KEYS
    assert settings["inputs"] == {}


def test_fresh_gui_uses_the_complete_saved_parameter_snapshot(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = DropSimulationWindow(settings_path=tmp_path / "absent.json")
    try:
        assert json.loads(json.dumps(window._parameter_values())) == GUI_INITIAL_PARAMETERS
    finally:
        window.close()
        del app


def test_partial_saved_settings_override_packaged_values_without_resetting_others(tmp_path):
    settings = tmp_path / "defaults.json"
    save_defaults(settings, {"trials": 13, "contact_damping_ratio": 0.7, "catalog_push": True}, {})
    app = QApplication.instance() or QApplication([])
    window = DropSimulationWindow(settings_path=settings)
    try:
        assert window.trials_spin.value() == 13
        assert window.contact_damping_spin.value() == 0.7
        assert window.catalog_push_check.isChecked()
        assert not window.compute_stability_check.isChecked()
        assert window.length_spin.value() == 500
        assert window.belt_stop_check.isChecked()
    finally:
        window.close()
        del app


@pytest.mark.parametrize("name,part,trials,workers,push", [
    ("gui-current.json", "Dk1i", 100, 4, False),
    ("dk1i-1000.json", "Dk1i", 1000, 16, False),
    ("kf1i-10000.json", "Kf1i", 10000, 16, True),
])
def test_portable_profiles_preserve_all_physics_and_run_sizes(name, part, trials, workers, push):
    config = _config(CONFIGS / name)
    assert config.mesh_path.stem == part
    assert config.trials == trials and config.workers == workers
    assert config.catalog_push is push
    current = _config(CONFIGS / "gui-current.json").to_dict()
    for key in PARAMETER_KEYS - {"trials", "workers", "catalog_push"}:
        assert config.to_dict()[key] == current[key], key
    assert config.contact_damping_ratio == 0.19
    assert config.roughness_active_duration_s == 1
    assert config.belt_stop_enabled and config.belt_stop_ramp_s == 0.2
    assert config.belt_stop_observation_s == 2
