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


def test_catalogue_dropdown_keeps_browse_path_and_1300_mm_default() -> None:
    app = QApplication.instance() or QApplication([])
    window = DropSimulationWindow()
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
    window = DropSimulationWindow()
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
