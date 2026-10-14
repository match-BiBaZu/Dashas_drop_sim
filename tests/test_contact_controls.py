"""Contact settings and force-pulse controls preserve units and saved values."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from dashas_drop_sim.config import RunConfig
from dashas_drop_sim.gui import DropSimulationWindow


@pytest.mark.parametrize("changes", [
    {"contact_damping_ratio": 0},
    {"contact_time_constant_s": 0.001, "timestep_s": 0.001},
    {"contact_rolling_friction_mm": -0.1},
    {"roughness_force_fraction": 1.1},
    {"roughness_force_duration_s": 0},
    {"roughness_force_interval_s": float("nan")},
])
def test_invalid_contact_and_force_settings_are_rejected(changes):
    config = RunConfig(Path("unused.stl"), **changes)
    with pytest.raises(ValueError):
        config.validate_parameters()


def test_weight_force_does_not_use_surface_height_to_validate_pulses():
    config = RunConfig(Path("unused.stl"), roughness_enabled=True,
                       roughness_model="weight_force", roughness_wall_height_mm=2)
    config.validate_parameters()


def test_weight_controls_roundtrip_percent_and_disable_unused_geometry(tmp_path):
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "defaults.json"
    window = DropSimulationWindow(settings_path=path)
    try:
        window.roughness_check.setChecked(True)
        window.roughness_model_combo.setCurrentIndex(window.roughness_model_combo.findData("weight_force"))
        window.force_fraction_spin.setValue(10)
        window.force_duration_spin.setValue(0.05)
        window.force_interval_spin.setValue(0.2)
        window.contact_damping_spin.setValue(0.5)
        assert window._parameter_values()["roughness_force_fraction"] == pytest.approx(0.1)
        assert not window.roughness_spins["roughness_wall_height_mm"].isEnabled()
        assert window.force_fraction_spin.isEnabled()
        window._save_defaults()
    finally:
        window.close()
    reopened = DropSimulationWindow(settings_path=path)
    try:
        assert reopened.roughness_model_combo.currentData() == "weight_force"
        assert reopened.force_fraction_spin.value() == 10
        assert reopened.force_duration_spin.value() == 0.05
        assert reopened.force_interval_spin.value() == 0.2
        assert reopened.contact_damping_spin.value() == 0.5
        assert not reopened.roughness_spins["roughness_wall_height_mm"].isEnabled()
        reopened.roughness_check.setChecked(False)
        assert not reopened.force_fraction_spin.isEnabled()
    finally:
        reopened.close()
        del app
