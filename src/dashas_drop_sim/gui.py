"""PyQt6 front end for repeatable chute drop tests.

The GUI deliberately runs the CLI in a separate process.  A slow physics run
therefore cannot block the window, and the CLI remains the single execution
path for interactive and unattended runs.
"""

from __future__ import annotations

import csv
from dataclasses import MISSING, fields
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any
import yaml

from PyQt6.QtCore import QProcess, QTimer, Qt
from PyQt6.QtGui import QCloseEvent, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import RunConfig
from .batch import Workpiece, batch_configs, find_roadmap
from .catalog import SOURCE_COMMIT, catalog_models
from .pose_image import render_pose_image
from .roadmap_reindex import reindex_roadmaps
from .settings import default_settings_path, load_defaults, save_defaults


def _default(name: str, fallback: Any) -> Any:
    """Read dataclass defaults without requiring a valid input model."""
    try:
        field = next(item for item in fields(RunConfig) if item.name == name)
    except (TypeError, StopIteration):
        return fallback
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:
        return field.default_factory()
    return fallback


def _spin(value: float, low: float, high: float, decimals: int = 1, step: float = 1) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(float(value))
    return box


class DropSimulationWindow(QMainWindow):
    """Configuration, launch control, and result view for drop tests."""

    def __init__(self, *, settings_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("BiBaZu Falltest-Simulation")
        self.resize(1020, 850)
        self._process: QProcess | None = None
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._cancel_requested = False
        self._close_when_finished = False
        self._run_started_at = 0.0
        self._output_dir: Path | None = None
        self._run_dir: Path | None = None
        self._mode = ""
        self._catalog_auto_roadmap: str | None = None
        self._batch_active = False
        self._batch_stop_requested = False
        self._batch_configs: list[RunConfig] = []
        self._batch_records: list[dict[str, Any]] = []
        self._batch_index = -1
        self._batch_dir: Path | None = None
        self._settings_path = Path(settings_path) if settings_path is not None else default_settings_path()
        self._build_ui()
        self._update_batch_count()
        self._load_saved_defaults()

    def _build_ui(self) -> None:
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        page = QWidget()
        layout = QVBoxLayout(page)

        files = QGroupBox("Eingaben und Ausgabe")
        file_form = QFormLayout(files)
        self.catalog_combo = QComboBox()
        self.catalog_combo.addItem("Eigenes Modell wählen …", None)
        for name, path in catalog_models().items():
            self.catalog_combo.addItem(name, str(path))
        self.catalog_combo.setToolTip(f"BiBaZu-STL-Katalog, Stand {SOURCE_COMMIT[:8]}")
        self.catalog_combo.currentIndexChanged.connect(self._select_catalog_model)
        file_form.addRow("Werkstückkatalog", self.catalog_combo)
        self.mesh_edit = QLineEdit()
        file_form.addRow("Bauteil (STL oder STEP)", self._path_row(self.mesh_edit, self._choose_mesh))
        self.roadmap_edit = QLineEdit()
        self.roadmap_edit.setPlaceholderText("Optional: Roadmap zum Abgleich der Pose-IDs")
        file_form.addRow("Roadmap", self._path_row(self.roadmap_edit, self._choose_roadmap))
        self.reindex_button = QPushButton("YAML und JSON nach Häufigkeit neu nummerieren …")
        self.reindex_button.setToolTip("Eine summary.json wählen und beide Roadmap-Dateien mit neuen Pose-IDs speichern")
        self.reindex_button.clicked.connect(self._reindex_roadmaps)
        file_form.addRow("Pose-Nummern", self.reindex_button)
        self.output_edit = QLineEdit(str(_default("output_dir", Path.cwd() / "results")))
        file_form.addRow("Ergebnisordner", self._path_row(self.output_edit, self._choose_output))
        layout.addWidget(files)

        batch_group = QGroupBox("Mehrere Werkstücke")
        batch_layout = QVBoxLayout(batch_group)
        batch_note = QLabel("Werkstücke nacheinander mit den unten eingestellten Parametern simulieren. Jeder Eintrag erhält eigene Ergebnisse.")
        batch_note.setWordWrap(True)
        batch_layout.addWidget(batch_note)
        batch_actions = QHBoxLayout()
        self.batch_add_files_button = QPushButton("Mehrere Dateien laden …")
        self.batch_add_files_button.clicked.connect(self._choose_batch_meshes)
        self.batch_catalog_button = QPushButton("Aus Katalog laden …")
        self.batch_catalog_button.clicked.connect(self._choose_batch_catalog)
        self.batch_add_current_button = QPushButton("Aktuelles Werkstück hinzufügen")
        self.batch_add_current_button.clicked.connect(self._add_current_workpiece)
        self.batch_remove_button = QPushButton("Auswahl entfernen")
        self.batch_remove_button.clicked.connect(self._remove_batch_selection)
        self.batch_clear_button = QPushButton("Liste leeren")
        self.batch_clear_button.clicked.connect(lambda: self.batch_table.setRowCount(0))
        self._batch_edit_buttons = [self.batch_add_files_button, self.batch_catalog_button,
                                    self.batch_add_current_button, self.batch_remove_button, self.batch_clear_button]
        for button in self._batch_edit_buttons:
            batch_actions.addWidget(button)
        batch_layout.addLayout(batch_actions)
        self.batch_table = QTableWidget(0, 4)
        self.batch_table.setHorizontalHeaderLabels(["Werkstück", "Roadmap (optional)", "Status", "Ergebnisse"])
        self.batch_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.batch_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        self.batch_table.setColumnWidth(1, 380)
        self.batch_table.setMinimumHeight(150)
        batch_layout.addWidget(self.batch_table)
        self.batch_status = QLabel("Keine Werkstücke in der Warteschlange")
        batch_layout.addWidget(self.batch_status)
        self.batch_table.model().rowsInserted.connect(self._update_batch_count)
        self.batch_table.model().rowsRemoved.connect(self._update_batch_count)
        layout.addWidget(batch_group)

        parameters = QGroupBox("Fallversuche")
        form = QFormLayout(parameters)
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(1, 1_000_000)
        self.trials_spin.setValue(int(_default("trials", 100)))
        form.addRow("Anzahl Versuche", self.trials_spin)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.workers_spin.setValue(int(_default("workers", 1)))
        self.workers_spin.setToolTip("Unabhängige MuJoCo-Prozesse; 1 führt die Versuche nacheinander aus")
        form.addRow("Parallele Prozesse", self.workers_spin)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(int(_default("seed", 0)))
        form.addRow("Zufalls-Seed", self.seed_spin)
        self.belt_spin = _spin(_default("belt_speed_mm_s", 100), 0, 200)
        self.belt_spin.setSuffix(" mm/s")
        form.addRow("Bandgeschwindigkeit", self.belt_spin)
        self.belt_variation_spin = _spin(_default("belt_speed_variation_mm_s", 0), 0, 20, 2, 0.5)
        self.belt_variation_spin.setPrefix("± ")
        self.belt_variation_spin.setSuffix(" mm/s")
        self.belt_variation_spin.setMaximum(min(20, self.belt_spin.value(), 200 - self.belt_spin.value()))
        self.belt_spin.valueChanged.connect(
            lambda value: self.belt_variation_spin.setMaximum(min(20, value, 200 - value)))
        self.belt_variation_spin.setToolTip(
            "Zeitliche Schwankung um die Bandgeschwindigkeit; ±0 schaltet sie aus. "
            "Für das reale Band geschätzt: ±3 mm/s.")
        form.addRow("Bandgeschwindigkeitsschwankung", self.belt_variation_spin)
        self.belt_interval_spin = _spin(_default("belt_variation_interval_s", 0.2), 0.02, 10, 3, 0.05)
        self.belt_interval_spin.setSuffix(" s")
        self.belt_interval_spin.setToolTip(
            "Abstand zwischen zufälligen Geschwindigkeitswerten; Übergänge sind glatt. "
            "0,2 s ist eine ungemessene Startannahme.")
        form.addRow("Änderungsintervall Band", self.belt_interval_spin)
        self.height_spin = _spin(_default("drop_height_mm", 100), 0, 200)
        self.height_spin.setSuffix(" mm")
        self.height_spread_spin = _spin(_default("drop_height_spread_mm", 0), 0, 100)
        self.height_spread_spin.setPrefix("± ")
        self.height_spread_spin.setSuffix(" mm")
        self.height_spread_spin.setMaximum(min(self.height_spin.value(), 200 - self.height_spin.value()))
        self.height_spin.valueChanged.connect(
            lambda value: self.height_spread_spin.setMaximum(min(value, 200 - value)))
        form.addRow("Abwurfhöhe und Streubereich", self._spread_row(self.height_spin, self.height_spread_spin))
        self.lateral_spin = _spin(_default("lateral_mm", 0), -100, 100)
        self.lateral_spin.setSuffix(" mm")
        self.lateral_spin.setToolTip("Seitlicher Versatz von der Winkelhalbierenden zwischen Band und Wand")
        self.lateral_spread_spin = _spin(_default("lateral_spread_mm", 0), 0, 100)
        self.lateral_spread_spin.setPrefix("± ")
        self.lateral_spread_spin.setSuffix(" mm")
        self.lateral_spread_spin.setMaximum(100 - abs(self.lateral_spin.value()))
        self.lateral_spin.valueChanged.connect(
            lambda value: self.lateral_spread_spin.setMaximum(100 - abs(value)))
        form.addRow("Querposition und Streubereich", self._spread_row(self.lateral_spin, self.lateral_spread_spin))
        spread_note = QLabel("Streuung gleichverteilt um den Sollwert; ±0 mm bedeutet einen festen Abwurfwert.")
        spread_note.setWordWrap(True)
        form.addRow("", spread_note)
        self.density_spin = _spin(_default("density_g_cm3", 1.15), 0.01, 30, 3, 0.01)
        self.density_spin.setSuffix(" g/cm³")
        form.addRow("Homogene Dichte", self.density_spin)
        self.belt_mu_spin = _spin(_default("mu_belt", 0.5), 0, 5, 3, 0.01)
        form.addRow("Reibwert PE-Band", self.belt_mu_spin)
        self.wall_mu_spin = _spin(_default("mu_wall", 0.2), 0, 5, 3, 0.01)
        form.addRow("Reibwert PTFE-Wand", self.wall_mu_spin)
        note = QLabel("Die Reibwerte sind vorläufige, einstellbare Simulationsannahmen.")
        note.setWordWrap(True)
        form.addRow("", note)
        layout.addWidget(parameters)

        roughness_group = QGroupBox("Kratzer und Dellen")
        roughness_layout = QVBoxLayout(roughness_group)
        self.roughness_check = QCheckBox("Zufällige Unebenheitsimpulse aktivieren")
        self.roughness_check.setChecked(bool(_default("roughness_enabled", False)))
        roughness_layout.addWidget(self.roughness_check)
        self.roughness_model_combo = QComboBox()
        self.roughness_model_combo.addItem("Kontaktimpulse entlang der Rutsche", "longitudinal_traction")
        self.roughness_model_combo.addItem("Flache Unebenheitsstöße mit Reibung", "microfacet")
        self.roughness_model_combo.setCurrentIndex(max(0, self.roughness_model_combo.findData(
            _default("roughness_model", "longitudinal_traction"))))
        self.roughness_model_combo.setEnabled(self.roughness_check.isChecked())
        self.roughness_check.toggled.connect(self.roughness_model_combo.setEnabled)
        roughness_layout.addWidget(self.roughness_model_combo)
        roughness_grid = QGridLayout()
        roughness_grid.addWidget(QLabel("PTFE-Wand"), 0, 1)
        roughness_grid.addWidget(QLabel("PE-Band"), 0, 2)
        self.roughness_spins: dict[str, QDoubleSpinBox] = {}
        for row, (parameter, label, low, high, default) in enumerate((
            ("height", "Max. wirksame Kantenhöhe", 0, 2, 0.1),
            ("ramp", "Wirksame Kantenlänge", 0.1, 100, 3.0),
            ("spacing", "Mittlerer Abstand", 1, 10000, 10.0),
        ), start=1):
            roughness_grid.addWidget(QLabel(label), row, 0)
            for column, surface in enumerate(("wall", "belt"), start=1):
                name = f"roughness_{surface}_{parameter}_mm"
                spin = _spin(_default(name, default), low, high, 3, 0.01 if parameter == "height" else 1)
                spin.setSuffix(" mm")
                spin.setEnabled(self.roughness_check.isChecked())
                self.roughness_spins[name] = spin
                roughness_grid.addWidget(spin, row, column)
        self.roughness_check.toggled.connect(
            lambda checked: [spin.setEnabled(checked) for spin in self.roughness_spins.values()])
        roughness_layout.addLayout(roughness_grid)
        roughness_note = QLabel(
            "Statistisches Ersatzmodell: Kontaktimpulse entlang der Rutsche oder flache Unebenheitsstöße. "
            "Normalkraft, Relativgeschwindigkeit und Höhe/Länge bestimmen die Impulsstärke, "
            "der Abstand die Häufigkeit entlang des relativen Gleitwegs. "
            "Wandwerte sind Startschätzungen; 0 mm Höhe schaltet eine Fläche aus."
        )
        roughness_note.setWordWrap(True)
        roughness_layout.addWidget(roughness_note)
        layout.addWidget(roughness_group)

        advanced = QGroupBox("Erweiterte Simulationsparameter")
        advanced_form = QFormLayout(advanced)
        self.alpha_spin = _spin(_default("alpha_deg", 45), 0, 90, 1)
        self.alpha_spin.setSuffix(" °")
        advanced_form.addRow("Querneigung", self.alpha_spin)
        self.beta_spin = _spin(_default("beta_deg", 0), -30, 30, 1)
        self.beta_spin.setSuffix(" °")
        advanced_form.addRow("Längsneigung", self.beta_spin)
        self.length_spin = _spin(_default("length_mm", 1300), 100, 100_000, 0, 100)
        self.length_spin.setSuffix(" mm")
        advanced_form.addRow("Beobachtungsstrecke", self.length_spin)
        self.timestep_spin = _spin(_default("timestep_s", 0.002), 0.0001, 0.01, 5, 0.0001)
        self.timestep_spin.setSuffix(" s")
        advanced_form.addRow("Zeitschritt", self.timestep_spin)
        levels = _default("disturbance_levels_mm", (0.5, 1.0, 2.0))
        self.levels_edit = QLineEdit(", ".join(str(v) for v in levels))
        self.levels_edit.setToolTip("Kommagetrennte Stufen für die reproduzierbaren Störversuche")
        advanced_form.addRow("Störstufen (mm)", self.levels_edit)
        layout.addWidget(advanced)

        controls = QHBoxLayout()
        self.start_button = QPushButton("Simulation starten")
        self.start_button.clicked.connect(lambda: self._start("run"))
        self.preview_button = QPushButton("Falltests sichtbar starten")
        self.preview_button.setToolTip("Alle Abwürfe nacheinander im selben MuJoCo-Fenster ansehen")
        self.preview_button.clicked.connect(lambda: self._start("watch"))
        self.cancel_button = QPushButton("Abbrechen")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        self.batch_start_button = QPushButton("Werkstück-Batch starten")
        self.batch_start_button.clicked.connect(self._start_batch)
        self.save_defaults_button = QPushButton("Als Standard speichern")
        self.save_defaults_button.setToolTip("Einstellungen für den nächsten GUI-Start speichern")
        self.save_defaults_button.clicked.connect(self._save_defaults)
        controls.addWidget(self.start_button)
        controls.addWidget(self.preview_button)
        controls.addWidget(self.batch_start_button)
        controls.addWidget(self.cancel_button)
        controls.addStretch()
        layout.addLayout(controls)
        defaults_row = QHBoxLayout()
        defaults_row.addWidget(self.save_defaults_button)
        defaults_row.addWidget(QLabel("Gespeicherte Einstellungen werden beim nächsten Start geladen."))
        defaults_row.addStretch()
        layout.addLayout(defaults_row)

        self.status_label = QLabel("Bereit")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)

        self.tabs = QTabWidget()
        frequency_tab = QWidget()
        frequency_layout = QVBoxLayout(frequency_tab)
        self.result_table = QTableWidget(0, 6)
        self.result_table.setHorizontalHeaderLabels(["Pose / Status", "Anzahl", "Anteil", "KI unten", "KI oben", "Ansicht"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.setMinimumHeight(180)
        frequency_layout.addWidget(self.result_table)
        self.tabs.addTab(frequency_tab, "Häufigkeit")

        stability_tab = QWidget()
        stability_layout = QVBoxLayout(stability_tab)
        stability_layout.addWidget(QLabel("Rangfolge nach Fläche unter der Störkurve (0–1)"))
        self.stability_rank_table = QTableWidget(0, 4)
        self.stability_rank_table.setHorizontalHeaderLabels(["Rang", "Pose", "AUC", "Vollständig"])
        self.stability_rank_table.horizontalHeader().setStretchLastSection(True)
        self.stability_rank_table.setMinimumHeight(120)
        stability_layout.addWidget(self.stability_rank_table)
        stability_layout.addWidget(QLabel("Pose-Erhalt je Störstufe; die Störstufe entspricht einer Hubenergie."))
        self.stability_curve_table = QTableWidget(0, 7)
        self.stability_curve_table.setHorizontalHeaderLabels(
            ["Pose", "Störstufe (mm)", "Energie (J)", "Erhalten", "Richtungen", "Anteil", "Vollständig"]
        )
        self.stability_curve_table.horizontalHeader().setStretchLastSection(True)
        self.stability_curve_table.setMinimumHeight(180)
        stability_layout.addWidget(self.stability_curve_table)
        self.tabs.addTab(stability_tab, "Stabilität")

        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        self.preview_table = QTableWidget(0, 2)
        self.preview_table.setHorizontalHeaderLabels(["Merkmal", "Wert"])
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        self.preview_table.setMinimumHeight(180)
        preview_layout.addWidget(self.preview_table)
        self.preview_image_button = QPushButton("Pose als Bild anzeigen")
        self.preview_image_button.setEnabled(False)
        self.preview_image_button.clicked.connect(self._show_preview_image)
        preview_layout.addWidget(self.preview_image_button)
        self._preview_record: dict[str, Any] | None = None
        self.tabs.addTab(preview_tab, "Letzter Versuch")
        layout.addWidget(self.tabs)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(170)
        layout.addWidget(self.log)
        outer.setWidget(page)
        self.setCentralWidget(outer)

    @staticmethod
    def _spread_row(center: QDoubleSpinBox, spread: QDoubleSpinBox) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(center)
        row.addWidget(spread)
        spread.setToolTip("Halbe Breite des gleichverteilten Streubereichs; begrenzt durch den Sollwert")
        return container

    @staticmethod
    def _path_row(edit: QLineEdit, handler: Any) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Durchsuchen …")
        browse.clicked.connect(handler)
        row.addWidget(edit, 1)
        row.addWidget(browse)
        return container

    def _choose_mesh(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Bauteil wählen", self.mesh_edit.text() or str(Path.cwd()),
            "CAD-Modelle (*.stl *.STL *.step *.STEP *.stp *.STP);;Alle Dateien (*)",
        )
        if path:
            if self._catalog_auto_roadmap and self.roadmap_edit.text() == self._catalog_auto_roadmap:
                self.roadmap_edit.clear()
            self._catalog_auto_roadmap = None
            self.catalog_combo.setCurrentIndex(0)
            self.mesh_edit.setText(path)

    def _choose_batch_meshes(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Mehrere Werkstücke wählen", self.mesh_edit.text() or str(Path.cwd()),
            "CAD-Modelle (*.stl *.STL *.step *.STEP *.stp *.STP);;Alle Dateien (*)",
        )
        self._add_workpieces([Workpiece(Path(path), find_roadmap(Path(path))) for path in paths])

    def _choose_batch_catalog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Werkstücke für den Batch auswählen")
        dialog.resize(400, 500)
        layout = QVBoxLayout(dialog)
        models = QListWidget()
        models.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for name, path in catalog_models().items():
            models.addItem(name)
            models.item(models.count() - 1).setData(Qt.ItemDataRole.UserRole, str(path))
        layout.addWidget(models)
        actions = QHBoxLayout()
        select_all = QPushButton("Alle auswählen")
        select_all.clicked.connect(models.selectAll)
        accept = QPushButton("Hinzufügen")
        accept.clicked.connect(dialog.accept)
        cancel = QPushButton("Abbrechen")
        cancel.clicked.connect(dialog.reject)
        for button in (select_all, accept, cancel):
            actions.addWidget(button)
        layout.addLayout(actions)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._add_workpieces([
                Workpiece(Path(item.data(Qt.ItemDataRole.UserRole)),
                          find_roadmap(Path(item.data(Qt.ItemDataRole.UserRole))))
                for item in models.selectedItems()
            ])

    def _add_current_workpiece(self) -> None:
        mesh = self.mesh_edit.text().strip()
        if not mesh:
            QMessageBox.warning(self, "Werkstück-Batch", "Bitte ein Werkstück wählen.")
            return
        roadmap = self.roadmap_edit.text().strip()
        self._add_workpieces([Workpiece(Path(mesh), Path(roadmap) if roadmap else None)])

    def _add_workpieces(self, workpieces: list[Workpiece]) -> None:
        if self._process is not None or self._batch_active:
            return
        existing = {Path(self.batch_table.item(row, 0).data(Qt.ItemDataRole.UserRole))
                    for row in range(self.batch_table.rowCount())}
        for workpiece in workpieces:
            mesh = workpiece.mesh_path.expanduser().resolve()
            if mesh in existing:
                continue
            row = self.batch_table.rowCount()
            self.batch_table.insertRow(row)
            item = QTableWidgetItem(mesh.stem)
            item.setData(Qt.ItemDataRole.UserRole, str(mesh))
            item.setToolTip(str(mesh))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.batch_table.setItem(row, 0, item)
            edit = QLineEdit(str(workpiece.roadmap_path.expanduser().resolve()) if workpiece.roadmap_path else "")
            edit.setPlaceholderText("Optional: Roadmap für dieses Werkstück")
            self.batch_table.setCellWidget(row, 1, self._path_row(edit, lambda _checked=False, field=edit: self._choose_batch_roadmap(field)))
            self._set_batch_status(row, "Bereit")
            result_button = QPushButton("Ergebnisse")
            result_button.setEnabled(False)
            self.batch_table.setCellWidget(row, 3, result_button)
            existing.add(mesh)
        self._update_batch_count()

    def _choose_batch_roadmap(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Roadmap für Werkstück wählen", edit.text() or str(Path.cwd()),
            "Roadmaps (*.yaml *.yml *.json);;Alle Dateien (*)",
        )
        if path:
            edit.setText(path)

    def _remove_batch_selection(self) -> None:
        for row in sorted({index.row() for index in self.batch_table.selectedIndexes()}, reverse=True):
            self.batch_table.removeRow(row)

    def _update_batch_count(self, *_args: Any) -> None:
        count = self.batch_table.rowCount()
        if not self._batch_active:
            self.batch_status.setText(f"{count} Werkstücke in der Warteschlange")
        self.batch_start_button.setEnabled(count > 0 and self._process is None and not self._batch_active)

    def _set_batch_status(self, row: int, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.batch_table.setItem(row, 2, item)

    def _queued_workpieces(self) -> list[Workpiece]:
        workpieces = []
        for row in range(self.batch_table.rowCount()):
            path = self.batch_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            edit = self.batch_table.cellWidget(row, 1).findChild(QLineEdit)
            roadmap = edit.text().strip()
            workpieces.append(Workpiece(Path(path), Path(roadmap) if roadmap else None))
        return workpieces

    def _set_running_controls(self, busy: bool) -> None:
        self.start_button.setEnabled(not busy)
        self.preview_button.setEnabled(not busy)
        self.reindex_button.setEnabled(not busy)
        self.save_defaults_button.setEnabled(not busy)
        self.batch_start_button.setEnabled(not busy and self.batch_table.rowCount() > 0)
        self.cancel_button.setEnabled(busy)
        self.batch_table.setEnabled(not busy)
        for button in self._batch_edit_buttons:
            button.setEnabled(not busy)

    def _select_catalog_model(self, index: int) -> None:
        path = self.catalog_combo.itemData(index)
        if path is None:
            return
        self.mesh_edit.setText(str(path))
        if self._catalog_auto_roadmap and self.roadmap_edit.text() == self._catalog_auto_roadmap:
            self.roadmap_edit.clear()
        self._catalog_auto_roadmap = None
        roadmap = find_roadmap(Path(path))
        if roadmap is not None:
            self._catalog_auto_roadmap = str(roadmap)
            self.roadmap_edit.setText(self._catalog_auto_roadmap)

    def _choose_roadmap(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Roadmap wählen", self.roadmap_edit.text() or str(Path.cwd()),
            "Roadmaps (*.yaml *.yml *.json);;Alle Dateien (*)",
        )
        if path:
            self.roadmap_edit.setText(path)

    def _reindex_roadmaps(self) -> None:
        if self._process is not None:
            return
        source_text = self.roadmap_edit.text().strip()
        if not source_text:
            QMessageBox.warning(self, "Pose-Nummern", "Bitte zuerst eine YAML- oder JSON-Roadmap wählen.")
            return
        source = Path(source_text).expanduser().resolve()
        summary_start = self._run_dir / "summary.json" if self._run_dir else Path(self.output_edit.text().strip())
        summary_path, _ = QFileDialog.getOpenFileName(
            self, "Simulationsergebnis wählen", str(summary_start), "Zusammenfassung (summary.json);;JSON (*.json)",
        )
        if not summary_path:
            return
        destination_start = source.with_name(f"{source.stem}_frequency_ordered{source.suffix}")
        destination, _ = QFileDialog.getSaveFileName(
            self, "Neu nummerierte Roadmaps speichern", str(destination_start),
            "Roadmaps (*.yaml *.yml *.json)",
        )
        if not destination:
            return
        try:
            mapping, backups, outputs = reindex_roadmaps(source, Path(summary_path), Path(destination))
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            QMessageBox.warning(self, "Pose-Nummern", str(exc))
            return
        selected = Path(destination).expanduser().resolve()
        self.roadmap_edit.setText(str(selected))
        for row, workpiece in enumerate(self._queued_workpieces()):
            if (workpiece.mesh_path == Path(self.mesh_edit.text()).expanduser().resolve()
                    and workpiece.roadmap_path is not None
                    and workpiece.roadmap_path.expanduser().resolve() == source):
                self.batch_table.cellWidget(row, 1).findChild(QLineEdit).setText(str(selected))
        self._catalog_auto_roadmap = None
        self.status_label.setText("Roadmaps neu nummeriert; Ergebnisse zeigen bisherige IDs")
        self._log_line(f"{len(mapping)} Posen neu nummeriert: {outputs[0]} und {outputs[1]}")
        self._log_line("Pose-IDs (alt → neu): " + ", ".join(
            f"{old_id} → {new_id}" for old_id, new_id in sorted(mapping.items(), key=lambda item: item[1])
        ))
        for backup in backups:
            self._log_line(f"Sicherung: {backup}")
        QMessageBox.information(
            self, "Pose-Nummern",
            f"{len(mapping)} Posen nach Häufigkeit neu nummeriert. YAML und JSON wurden gespeichert.\n"
            "Die angezeigten Ergebnisse verwenden noch die bisherigen Pose-Nummern.",
        )

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Ergebnisordner wählen", self.output_edit.text() or str(Path.cwd())
        )
        if path:
            self.output_edit.setText(path)

    def _parameter_widgets(self) -> dict[str, QSpinBox | QDoubleSpinBox]:
        return {
            "trials": self.trials_spin, "workers": self.workers_spin, "seed": self.seed_spin,
            "belt_speed_mm_s": self.belt_spin,
            "belt_speed_variation_mm_s": self.belt_variation_spin,
            "belt_variation_interval_s": self.belt_interval_spin,
            "drop_height_mm": self.height_spin, "drop_height_spread_mm": self.height_spread_spin,
            "lateral_mm": self.lateral_spin, "lateral_spread_mm": self.lateral_spread_spin,
            "density_g_cm3": self.density_spin, "mu_belt": self.belt_mu_spin, "mu_wall": self.wall_mu_spin,
            "alpha_deg": self.alpha_spin, "beta_deg": self.beta_spin,
            "length_mm": self.length_spin, "timestep_s": self.timestep_spin,
            **self.roughness_spins,
        }

    def _parameter_values(self) -> dict[str, Any]:
        level_text = self.levels_edit.text().strip()
        try:
            levels = tuple(float(part.strip()) for part in level_text.split(",") if part.strip())
        except ValueError as exc:
            raise ValueError("Störstufen bitte als kommagetrennte Zahlen eingeben.") from exc
        if not levels:
            raise ValueError("Mindestens eine Störstufe ist erforderlich.")
        return {**{name: widget.value() for name, widget in self._parameter_widgets().items()},
                "roughness_enabled": self.roughness_check.isChecked(),
                "roughness_model": self.roughness_model_combo.currentData(), "disturbance_levels_mm": levels}

    def _save_defaults(self) -> None:
        try:
            parameters = self._parameter_values()
            inputs = {name: str(Path(edit.text().strip()).expanduser().resolve()) if edit.text().strip() else ""
                      for name, edit in (("mesh_path", self.mesh_edit), ("roadmap_path", self.roadmap_edit),
                                         ("output_dir", self.output_edit))}
            save_defaults(self._settings_path, parameters, inputs)
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Starteinstellungen", str(exc))
            return
        self.status_label.setText("Einstellungen als Standard gespeichert")
        self._log_line(f"Starteinstellungen: {self._settings_path}")

    def _load_saved_defaults(self) -> None:
        try:
            data = load_defaults(self._settings_path)
        except (OSError, TypeError, ValueError) as exc:
            self._log_line(f"Gespeicherte Starteinstellungen konnten nicht geladen werden: {exc}")
            return
        if data is None:
            return
        parameters = data["parameters"]
        for name, widget in self._parameter_widgets().items():
            if name in parameters:
                widget.setValue(parameters[name])
        if "roughness_enabled" in parameters:
            self.roughness_check.setChecked(parameters["roughness_enabled"])
        if "roughness_model" in parameters:
            self.roughness_model_combo.setCurrentIndex(self.roughness_model_combo.findData(parameters["roughness_model"]))
        if "disturbance_levels_mm" in parameters:
            self.levels_edit.setText(", ".join(str(value) for value in parameters["disturbance_levels_mm"]))
        for name, edit in (("mesh_path", self.mesh_edit), ("roadmap_path", self.roadmap_edit),
                           ("output_dir", self.output_edit)):
            if name in data["inputs"]:
                edit.setText(data["inputs"][name])
        previous = self.catalog_combo.blockSignals(True)
        self.catalog_combo.setCurrentIndex(max(0, self.catalog_combo.findData(self.mesh_edit.text())))
        self.catalog_combo.blockSignals(previous)
        self._log_line(f"Starteinstellungen geladen: {self._settings_path}")

    def _make_config(self, workpiece: Workpiece | None = None) -> RunConfig:
        mesh = str(workpiece.mesh_path) if workpiece else self.mesh_edit.text().strip()
        output = self.output_edit.text().strip()
        if not mesh:
            raise ValueError("Bitte ein STL- oder STEP-Bauteil wählen.")
        if not output:
            raise ValueError("Bitte einen Ergebnisordner wählen.")
        config = RunConfig(
            mesh_path=Path(mesh).expanduser().resolve(),
            roadmap_path=workpiece.roadmap_path if workpiece else (
                Path(self.roadmap_edit.text().strip()).expanduser().resolve()
                if self.roadmap_edit.text().strip() else None
            ),
            output_dir=Path(output).expanduser().resolve(),
            **self._parameter_values(),
        )
        config.validate()
        return config

    def _start(self, mode: str) -> None:
        if self._process is not None or self._batch_active:
            return
        try:
            config = self._make_config()
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Eingabe prüfen", str(exc))
            return
        self._launch_config(config, mode)

    def _start_batch(self) -> None:
        if self._process is not None or self._batch_active:
            return
        try:
            workpieces = self._queued_workpieces()
            if not workpieces:
                raise ValueError("Bitte Werkstücke zur Warteschlange hinzufügen.")
            base = self._make_config(workpieces[0])
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            batch_dir = base.output_dir / f"batch_{stamp}"
            configs = batch_configs(base, workpieces, batch_dir)
            batch_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Werkstück-Batch", str(exc))
            return
        self._batch_dir = batch_dir
        self._batch_configs = configs
        self._batch_records = [{"config": config.to_dict(), "status": "pending", "run_dir": None}
                               for config in configs]
        self._batch_index = -1
        self._batch_stop_requested = False
        self._batch_active = True
        self.log.clear()
        self._log_line(f"Werkstück-Batch: {batch_dir}")
        for row in range(self.batch_table.rowCount()):
            self._set_batch_status(row, "Wartet")
            self.batch_table.cellWidget(row, 3).setEnabled(False)
        self._set_running_controls(True)
        self._persist_batch()
        self._advance_batch()

    def _persist_batch(self) -> None:
        if self._batch_dir is None:
            return
        data = {"status": "running" if self._batch_active else (
            "cancelled" if self._batch_stop_requested else "finished"
        ), "workpieces": self._batch_records}
        try:
            (self._batch_dir / "batch.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
        except OSError as exc:
            self._log_line(f"Batch-Übersicht konnte nicht gespeichert werden: {exc}")

    def _advance_batch(self) -> None:
        if not self._batch_active or self._process is not None:
            return
        if self._batch_stop_requested or self._batch_index + 1 >= len(self._batch_configs):
            self._finish_batch()
            return
        self._batch_index += 1
        config = self._batch_configs[self._batch_index]
        self.batch_status.setText(f"Werkstück {self._batch_index + 1} / {len(self._batch_configs)}: {config.mesh_path.stem}")
        self._set_batch_status(self._batch_index, "Läuft")
        self._batch_records[self._batch_index]["status"] = "running"
        self._persist_batch()
        self.mesh_edit.setText(str(config.mesh_path))
        self.roadmap_edit.setText(str(config.roadmap_path or ""))
        self._catalog_auto_roadmap = None
        self._log_line(f"Werkstück {self._batch_index + 1}: {config.mesh_path}")
        if not self._launch_config(config, "run"):
            self._record_batch_result("failed", 2)
            QTimer.singleShot(0, self._advance_batch)

    def _record_batch_result(self, outcome: str, code: int) -> None:
        record = self._batch_records[self._batch_index]
        record.update(status=outcome, exit_code=code, run_dir=str(self._run_dir) if self._run_dir else None)
        labels = {"completed": "Abgeschlossen", "failed": "Fehlgeschlagen", "cancelled": "Abgebrochen"}
        self._set_batch_status(self._batch_index, labels[outcome])
        if self._run_dir is not None and any((self._run_dir / name).is_file() for name in ("summary.json", "frequencies.csv")):
            button = QPushButton("Ergebnisse")
            config = self._batch_configs[self._batch_index]
            run_dir = self._run_dir
            button.clicked.connect(lambda _checked=False, settings=config, path=run_dir: self._show_batch_result(settings, path))
            self.batch_table.setCellWidget(self._batch_index, 3, button)
        self._persist_batch()

    def _finish_batch(self) -> None:
        self._batch_active = False
        completed = sum(record["status"] == "completed" for record in self._batch_records)
        failed = sum(record["status"] == "failed" for record in self._batch_records)
        for row, record in enumerate(self._batch_records):
            if record["status"] == "pending":
                self._set_batch_status(row, "Nicht gestartet")
        label = "Batch abgebrochen" if self._batch_stop_requested else "Batch abgeschlossen"
        self.batch_status.setText(f"{label}: {completed} abgeschlossen, {failed} fehlgeschlagen")
        self.status_label.setText(label)
        self._persist_batch()
        self._set_running_controls(False)
        if self._close_when_finished:
            self.close()

    def _show_batch_result(self, config: RunConfig, run_dir: Path) -> None:
        if self._process is not None or self._batch_active:
            return
        self.mesh_edit.setText(str(config.mesh_path))
        self.roadmap_edit.setText(str(config.roadmap_path or ""))
        self._catalog_auto_roadmap = None
        self._run_dir = run_dir
        self._output_dir = config.output_dir
        self._run_started_at = 0
        self._mode = "run"
        self.result_table.setRowCount(0)
        self.stability_rank_table.setRowCount(0)
        self.stability_curve_table.setRowCount(0)
        self.preview_table.setRowCount(0)
        self.preview_image_button.setEnabled(False)
        self._preview_record = None
        self._load_result_file()
        self.tabs.setCurrentIndex(0)
        self.status_label.setText(f"Ergebnisse: {config.mesh_path.stem}")

    def _launch_config(self, config: RunConfig, mode: str) -> bool:
        self._run_dir = None
        try:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            config_path = config.output_dir / f"gui_config_{stamp}.json"
            config_path.write_text(
                json.dumps(config.to_dict(), ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            if self._batch_active:
                self._log_line(f"Werkstück konnte nicht gestartet werden: {exc}")
            else:
                QMessageBox.warning(self, "Eingabe prüfen", str(exc))
            return False

        self._mode = mode
        self._run_started_at = time.time()
        self._output_dir = config.output_dir
        self._run_dir = None
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._cancel_requested = False
        self.result_table.setRowCount(0)
        self.stability_rank_table.setRowCount(0)
        self.stability_curve_table.setRowCount(0)
        self.preview_table.setRowCount(0)
        self.preview_image_button.setEnabled(False)
        self._preview_record = None
        self.tabs.setCurrentIndex(2 if mode == "watch" else 0)
        self.progress.setRange(0, config.trials)
        self.progress.setValue(0)
        self.status_label.setText("Sichtbare Falltests laufen …" if mode == "watch" else "Falltests laufen …")
        if not self._batch_active:
            self.log.clear()
        self._log_line(f"Konfiguration: {config_path}")
        self._set_running_controls(True)

        process = QProcess(self)
        process.setProgram(sys.executable)
        args = ["-m", "dashas_drop_sim", mode, "--config", str(config_path)]
        process.setArguments(args)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self._process = process
        process.start()
        return True

    def _cancel(self) -> None:
        process = self._process
        if self._batch_active:
            self._batch_stop_requested = True
        if process is None:
            if self._batch_active:
                self._finish_batch()
            return
        if self._cancel_requested:
            return
        self._cancel_requested = True
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Abbruch angefordert …")
        self._log_line("Abbruch angefordert; bereits fertige Versuche bleiben gespeichert.")
        process.write(b"stop\n")
        QTimer.singleShot(15_000, self._force_stop_if_needed)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._process is not None:
            self._close_when_finished = True
            self._cancel()
            event.ignore()
            return
        if self._batch_active:
            self._batch_stop_requested = True
            self._finish_batch()
        event.accept()

    def _force_stop_if_needed(self) -> None:
        process = self._process
        if process is not None and self._cancel_requested and process.state() != QProcess.ProcessState.NotRunning:
            self._log_line("Prozess reagiert nicht auf den Abbruch; er wird beendet.")
            process.kill()

    def _read_stdout(self) -> None:
        process = self._process
        if process is not None:
            self._stdout_buffer = self._consume_lines(
                self._stdout_buffer + bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace"),
                is_error=False,
            )

    def _read_stderr(self) -> None:
        process = self._process
        if process is not None:
            self._stderr_buffer = self._consume_lines(
                self._stderr_buffer + bytes(process.readAllStandardError()).decode("utf-8", errors="replace"),
                is_error=True,
            )

    def _consume_lines(self, data: str, *, is_error: bool) -> str:
        lines = data.split("\n")
        for line in lines[:-1]:
            line = line.rstrip("\r")
            if line:
                self._handle_line(line, is_error=is_error)
        return lines[-1]

    def _handle_line(self, line: str, *, is_error: bool) -> None:
        if not is_error:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = None
            if isinstance(event, dict):
                if self._handle_event(event):
                    return
        self._log_line(("Fehler: " if is_error else "") + line)

    def _handle_event(self, event: dict[str, Any]) -> bool:
        kind = str(event.get("event", event.get("type", ""))).lower()
        if kind == "started":
            if event.get("run_dir"):
                self._run_dir = Path(str(event["run_dir"]))
                self._log_line(f"Ergebnisordner: {self._run_dir}")
            return True
        if kind in {"progress", "trial", "trial_complete"}:
            trial = event.get("completed", event.get("trial"))
            total = event.get("total", event.get("trials"))
            try:
                if total is not None:
                    self.progress.setRange(0, max(1, int(total)))
                if trial is not None:
                    self.progress.setValue(max(0, int(trial)))
                    self.status_label.setText(f"Versuch {int(trial)} / {self.progress.maximum()}")
            except (TypeError, ValueError):
                pass
            if self._mode == "watch" and isinstance(event.get("record"), dict):
                self._show_trial_record(event["record"])
            return True
        if kind == "stability_progress":
            try:
                completed = int(event["completed"])
                total = int(event["total"])
                self.progress.setRange(0, max(1, total))
                self.progress.setValue(completed)
                self.status_label.setText(
                    f"Störversuche für Pose {event.get('pose_id', '?')}: {completed} / {total}"
                )
            except (KeyError, TypeError, ValueError):
                self._log_line(str(event))
            return True
        if kind in {"result", "pose_frequency"}:
            self._set_result_rows([event], append=True)
            return True
        if kind in {"summary", "done"}:
            if event.get("run_dir"):
                self._run_dir = Path(str(event["run_dir"]))
                self._log_line(f"Ergebnisordner: {self._run_dir}")
            rows = self._extract_rows(event)
            if rows:
                self._set_result_rows(rows)
            if event.get("summary") and isinstance(event["summary"], str):
                self._log_line(f"Zusammenfassung: {event['summary']}")
            if event.get("message"):
                self._log_line(str(event["message"]))
            return True
        return False

    def _log_line(self, line: str) -> None:
        self.log.append(line)
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if self._process is not None:
            self._log_line(f"Prozessfehler: {self._process.errorString()} ({error.name})")
            if error == QProcess.ProcessError.FailedToStart:
                self._finished(2, QProcess.ExitStatus.CrashExit)

    def _finished(self, code: int, status: QProcess.ExitStatus) -> None:
        self._read_stdout()
        self._read_stderr()
        if self._stdout_buffer.strip():
            self._handle_line(self._stdout_buffer.strip(), is_error=False)
        if self._stderr_buffer.strip():
            self._handle_line(self._stderr_buffer.strip(), is_error=True)
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._load_result_file()
        if self._cancel_requested or code == 130:
            outcome = "cancelled"
            self.status_label.setText("Abgebrochen")
        elif status == QProcess.ExitStatus.NormalExit and code == 0:
            outcome = "completed"
            self.status_label.setText("Simulation abgeschlossen")
            if self._mode in {"run", "watch"}:
                self.progress.setValue(self.progress.maximum())
        else:
            outcome = "failed"
            self.status_label.setText(f"Simulation fehlgeschlagen (Exitcode {code})")
        if self._process is not None:
            self._process.deleteLater()
            self._process = None
        if self._batch_active:
            if outcome == "cancelled":
                self._batch_stop_requested = True
            self._record_batch_result(outcome, code)
            QTimer.singleShot(0, self._advance_batch)
            return
        self._set_running_controls(False)
        if self._close_when_finished:
            self.close()

    def _load_result_file(self) -> None:
        """Show the newest summary from this run, even when CLI emits plain text."""
        base = self._run_dir or self._output_dir
        if base is None:
            return
        if not base.is_dir():
            return
        if self._mode == "preview":
            self._load_preview(base)
            return
        candidates = list(base.glob("summary.json")) + list(base.glob("*/summary.json"))
        candidates += list(base.glob("frequencies.csv")) + list(base.glob("*/frequencies.csv"))
        candidates = [path for path in candidates if path.stat().st_mtime >= self._run_started_at - 2]
        for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                if path.suffix == ".json":
                    data = json.loads(path.read_text(encoding="utf-8"))
                    rows = self._extract_rows(data)
                else:
                    with path.open("r", encoding="utf-8-sig", newline="") as stream:
                        rows = list(csv.DictReader(stream))
                if rows:
                    self._set_result_rows(rows)
                    self._log_line(f"Ergebnisse: {path}")
                    break
            except (OSError, ValueError, TypeError) as exc:
                self._log_line(f"Ergebnistabelle konnte nicht gelesen werden: {exc}")
        self._load_stability(base)

    def _load_preview(self, base: Path) -> None:
        candidates = list(base.glob("preview.json")) + list(base.glob("*/preview.json"))
        candidates = [path for path in candidates if path.stat().st_mtime >= self._run_started_at - 2]
        for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                self._show_trial_record(data)
                self.tabs.setCurrentIndex(2)
                self._log_line(f"Einzelfall: {path}")
                return
            except (OSError, ValueError, TypeError) as exc:
                self._log_line(f"Einzelfall konnte nicht gelesen werden: {exc}")

    def _show_trial_record(self, data: dict[str, Any]) -> None:
        fields = (
            ("Versuch", data["trial"] + 1 if isinstance(data.get("trial"), int) else None),
            ("Status", data.get("status")),
            ("Roadmap-Pose", data.get("roadmap_pose_id")),
            ("Zuordnung", data.get("match_status")),
            ("Abstand zur Roadmap-Pose (°)", data.get("match_distance_deg")),
            ("Seed", data.get("seed")),
            ("Gezogene Abwurfhöhe (mm)", data.get("actual_drop_height_mm")),
            ("Gezogene Querposition (mm)", data.get("actual_lateral_mm")),
            ("Abwurfposition in Rutschenachsen (mm)", data.get("initial_pos_chute_mm")),
            ("Simulationszeit (s)", data.get("sim_time_s")),
            ("Weg (mm)", data.get("travel_mm")),
            ("Erstes Einpendeln (s)", data.get("first_settled_s")),
            ("Endposition in Rutschenachsen (mm)", data.get("final_pos_chute_mm")),
            ("Endorientierung xyzw", data.get("final_quat_xyzw")),
            ("Bandkontakt am Endpunkt", data.get("final_floor_contact")),
            ("Wandkontakt am Endpunkt", data.get("final_wall_contact")),
            ("Unebenheitsimpulse", data.get("roughness_impulse_count", 0)),
        )
        self.preview_table.setRowCount(0)
        for label, value in fields:
            index = self.preview_table.rowCount()
            self.preview_table.insertRow(index)
            if isinstance(value, float):
                display = f"{value:.4f}"
            elif isinstance(value, list):
                display = ", ".join(f"{item:.4f}" if isinstance(item, (int, float)) else str(item)
                                    for item in value)
            else:
                display = "–" if value is None else str(value)
            self.preview_table.setItem(index, 0, QTableWidgetItem(label))
            self.preview_table.setItem(index, 1, QTableWidgetItem(display))
        self.preview_table.resizeColumnToContents(0)
        self._preview_record = data
        self.preview_image_button.setEnabled(bool(data.get("final_quat_xyzw")))

    def _load_stability(self, base: Path) -> None:
        def csv_rows(filename: str) -> list[dict[str, Any]] | None:
            candidates = list(base.glob(filename)) + list(base.glob(f"*/{filename}"))
            candidates = [path for path in candidates if path.stat().st_mtime >= self._run_started_at - 2]
            for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
                try:
                    with path.open("r", encoding="utf-8-sig", newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    self._log_line(f"Stabilitätsdaten: {path}")
                    return rows
                except OSError as exc:
                    self._log_line(f"Stabilitätsdaten konnten nicht gelesen werden: {exc}")
            return None

        ranking = csv_rows("stability_summary.csv")
        curves = csv_rows("stability.csv")
        if ranking is None or curves is None:
            candidates = list(base.glob("summary.json")) + list(base.glob("*/summary.json"))
            for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
                try:
                    summary = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(summary, dict):
                        ranking = ranking if ranking is not None else summary.get("stability_ranking", [])
                        curves = curves if curves is not None else summary.get("stability_curves", [])
                        break
                except (OSError, ValueError):
                    continue
        self._fill_stability_table(
            self.stability_rank_table,
            ranking or [],
            ("rank", "pose_id", "retention_auc", "complete"),
        )
        self._fill_stability_table(
            self.stability_curve_table,
            curves or [],
            ("pose_id", "energy_lift_mm", "energy_j", "retained_count",
             "direction_count", "retention", "complete"),
        )

    @staticmethod
    def _fill_stability_table(table: QTableWidget, rows: list[dict[str, Any]], keys: tuple[str, ...]) -> None:
        table.setRowCount(0)
        for row in rows:
            index = table.rowCount()
            table.insertRow(index)
            for column, key in enumerate(keys):
                value = row.get(key, "")
                if value is None:
                    value = ""
                elif isinstance(value, float):
                    value = f"{value:.4f}"
                table.setItem(index, column, QTableWidgetItem(str(value)))

    @staticmethod
    def _extract_rows(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if not isinstance(data, dict):
            return []
        for key in ("pose_frequencies", "frequencies", "poses", "results", "counts"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                return [dict(pose_id=pose_id, **(details if isinstance(details, dict) else {"count": details}))
                        for pose_id, details in value.items()]
        return []

    def _set_result_rows(self, rows: list[dict[str, Any]], *, append: bool = False) -> None:
        if not append:
            self.result_table.setRowCount(0)
        for row in rows:
            index = self.result_table.rowCount()
            self.result_table.insertRow(index)
            values = (
                row.get("pose_id", row.get("pose", row.get("status", "?"))),
                row.get("count", row.get("n", "")),
                row.get("probability", row.get("frequency", row.get("fraction", ""))),
                row.get("ci_low", row.get("ci95_low", "")),
                row.get("ci_high", row.get("ci95_high", "")),
            )
            for column, value in enumerate(values):
                if isinstance(value, float) and column >= 2:
                    value = f"{value:.3f}"
                item = QTableWidgetItem(str(value))
                if column:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.result_table.setItem(index, column, item)
            button = QPushButton("Bild")
            button.setEnabled(bool(row.get("representative_quat_xyzw")) and self._run_dir is not None)
            button.clicked.connect(lambda _checked=False, selected=row: self._show_pose(selected))
            self.result_table.setCellWidget(index, 5, button)

    def _show_preview_image(self) -> None:
        if self._preview_record is None:
            return
        record = self._preview_record
        self._show_pose({
            "pose_id": record.get("roadmap_pose_id") or "Versuch",
            "representative_quat_xyzw": record.get("final_quat_xyzw"),
            "representative_pos_chute_mm": record.get("final_pos_chute_mm"),
            "representative_source": "observed",
        })

    def _show_pose(self, row: dict[str, Any]) -> None:
        try:
            if self._run_dir is None:
                raise ValueError("Die Bildansicht ist nach Abschluss des Laufs verfügbar.")
            manifest = json.loads((self._run_dir / "manifest.json").read_text(encoding="utf-8"))
            config = json.loads((self._run_dir / "config.json").read_text(encoding="utf-8"))
            mesh_path = Path(manifest.get("catalog_mesh_path") or config["mesh_path"])
            quaternion = row.get("representative_quat_xyzw")
            position = row.get("representative_pos_chute_mm")
            if isinstance(quaternion, str):
                quaternion = json.loads(quaternion)
            if isinstance(position, str):
                position = json.loads(position) if position.strip() else None
            if not mesh_path.is_file() or quaternion is None:
                raise ValueError("Mesh oder Orientierung für diese Pose ist nicht verfügbar.")
            image = render_pose_image(mesh_path, manifest["center_mass_source_mm"], quaternion, position)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Posenbild", str(exc))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Pose {row.get('pose_id', '?')}")
        layout = QVBoxLayout(dialog)
        image_label = QLabel()
        image_label.setPixmap(QPixmap.fromImage(image))
        layout.addWidget(image_label)
        note = ("Gemessene Endorientierung eines Fallversuchs" if row.get("representative_source") == "observed"
                else "Katalogorientierung; diese Pose wurde im Lauf nicht beobachtet")
        layout.addWidget(QLabel(note))
        layout.addWidget(QLabel("Schematischer Ausschnitt: blaues PE-Band, graue PTFE-Wand; +X ist die Bandrichtung."))
        dialog.exec()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    icon_path = Path(__file__).resolve().parents[2] / "WindowsLaunchers" / "icons" / "drop-sim.ico"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))
    window = DropSimulationWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
