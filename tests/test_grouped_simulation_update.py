"""Main's contact physics and stop observations coexist with feature grouping."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import trimesh
from PyQt6.QtWidgets import QApplication

from dashas_drop_sim.config import RunConfig
from dashas_drop_sim.gui import DropSimulationWindow
from dashas_drop_sim.runner import run_experiment
from dashas_drop_sim.settings import GUI_INITIAL_PARAMETERS


def test_grouping_and_contact_controls_survive_gui_restart(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings = tmp_path / "defaults.json"
    window = DropSimulationWindow(settings_path=settings)
    try:
        window.grouping_check.setChecked(True)
        window.grouping_barrier_spin.setValue(.25)
        window.grouping_angle_spin.setValue(45)
        window.grouping_frequency_spin.setValue(2)
        window.roughness_model_combo.setCurrentIndex(window.roughness_model_combo.findData("weight_force"))
        window.force_fraction_spin.setValue(15)
        window.contact_damping_spin.setValue(.8)
        expected = window._parameter_values()
        window._save_defaults()
        assert settings.is_file()
    finally:
        window.close()
    reopened = DropSimulationWindow(settings_path=settings)
    try:
        mesh = tmp_path / "cube.stl"
        trimesh.creation.box(extents=[20, 20, 20]).export(mesh)
        reopened.mesh_edit.setText(str(mesh))
        reopened.output_edit.setText(str(tmp_path / "results"))
        config = reopened._make_config()
        assert reopened._parameter_values() == expected
        assert config.grouping_enabled and config.compute_stability
        assert config.grouping_barrier_mm == .25
        assert config.grouping_max_angle_deg == 45
        assert config.grouping_minimum_frequency_percent == 2
        assert config.roughness_force_fraction == .15
        assert config.contact_damping_ratio == .8
        assert config.belt_stop_enabled
        assert not hasattr(reopened, "roadmap_edit")
    finally:
        reopened.close()
        del app


def test_grouped_run_with_impulses_and_belt_stop_resumes(tmp_path):
    mesh = tmp_path / "cube.stl"
    trimesh.creation.box(extents=[20, 20, 20]).export(mesh)
    config = RunConfig(
        mesh_path=mesh, output_dir=tmp_path / "results", trials=1, workers=1,
        drop_height_mm=0, length_mm=500, grouping_enabled=True,
        grouping_barrier_mm=.005, disturbance_levels_mm=(0, .005),
        **GUI_INITIAL_PARAMETERS,
    )
    result = run_experiment(config)
    assert result.summary["group_assessment_complete"]
    assert result.summary["settled_trials"] == 1
    assert sum(group["count"] for group in result.summary["pose_groups"]) == 1
    assert result.summary["belt_stop_observation"]["observed_trials"] == 1
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["surface_roughness"]["instantaneous_impulse"]["delta_v_free_mm_s"] == 49.05
    trials = [json.loads(line) for line in (result.run_dir / "grouped_trials.jsonl").read_text().splitlines()]
    assert trials[0]["roughness_impulse_count"] > 0
    assert trials[0]["stop_observation"] is not None
    assert trials[0]["group_id"] == result.summary["pose_groups"][0]["pose_id"]
    checkpoints = {name: (result.run_dir / name).read_bytes()
                   for name in ("trials.jsonl", "group_disturbances.jsonl")}
    resumed = run_experiment(config, resume_run_dir=result.run_dir)
    assert resumed.summary["pose_groups"] == result.summary["pose_groups"]
    assert resumed.summary["belt_stop_observation"] == result.summary["belt_stop_observation"]
    for name, content in checkpoints.items():
        assert (result.run_dir / name).read_bytes() == content
