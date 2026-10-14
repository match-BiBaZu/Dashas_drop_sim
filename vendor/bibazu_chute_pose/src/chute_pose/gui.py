"""Offline PyQt6 workpiece generation GUI; no conveyor or PLC imports."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from PyQt6.QtCore import QProcess, QProcessEnvironment, QSettings, Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QFont, QFontDatabase
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView, QScrollArea)

from .generate import GenerationConfig, OUTPUTS
from .metrics import METHOD_LABELS

ROOT = Path(__file__).resolve().parents[2]


class WorkpieceTable(QTableWidget):
    """Workpiece list where Ctrl-click adds or removes a file from generation."""

    def __init__(self):
        super().__init__(0, 3)
        self.on_control_click = None

    def mousePressEvent(self, event):
        if (event.button() == Qt.MouseButton.LeftButton
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            item = self.itemAt(event.position().toPoint())
            if item is not None and self.on_control_click is not None:
                self.on_control_click(item.row())
                event.accept()
                return
        super().mousePressEvent(event)


class RoadmapGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        if sys.platform == "win32" and not QFontDatabase.families():
            font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                QApplication.instance().setFont(QFont(families[0], 9))
        self.setWindowTitle("BiBaZu Pose Roadmap Generator")
        self.resize(1250, 900)
        self.settings = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                                  "BiBaZu", "PoseRoadmapGenerator")
        self.process = None
        self.pending = b""
        self.cancelled = False
        self.meshes = []
        self.active_parts = []
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.controls = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.controls)
        layout.addWidget(scroll, 1)
        top = QVBoxLayout(self.controls)
        self.input_dir = QLineEdit(str(ROOT / "Werkstücke_STL_grob"))
        self.output_dir = QLineEdit(str(ROOT / "Poses_Found_Robust" / "gui_static_020"))
        for label, field in (("STL folder", self.input_dir), ("Output folder", self.output_dir)):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(field, 1)
            button = QPushButton("Browse…")
            button.clicked.connect(lambda checked=False, f=field: self.browse(f))
            row.addWidget(button)
            top.addLayout(row)
        split = QSplitter()
        top.addWidget(split)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter workpieces…")
        self.search.textChanged.connect(self.filter_rows)
        left_layout.addWidget(self.search)
        row = QHBoxLayout()
        for label, action in (("Refresh", self.scan), ("Select all", lambda: self.select_rows(True)),
                              ("Select none", lambda: self.select_rows(False))):
            button = QPushButton(label)
            button.clicked.connect(action)
            row.addWidget(button)
        left_layout.addLayout(row)
        self.table = WorkpieceTable()
        self.table.on_control_click = self.toggle_workpiece
        self.table.setHorizontalHeaderLabels(["Generate", "Workpiece", "Status / existing output"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumWidth(470)
        left_layout.addWidget(self.table)
        selection_tip = QLabel("Hold Ctrl and click workpieces to add or remove them from the selection.")
        selection_tip.setWordWrap(True)
        left_layout.addWidget(selection_tip)
        split.addWidget(left)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        formats_box = QGroupBox("Files to generate — only checked formats are written")
        formats_grid = QGridLayout(formats_box)
        self.outputs = {}
        for i, (key, label) in enumerate(OUTPUTS.items()):
            check = QCheckBox(label)
            check.setChecked(key in GenerationConfig("").outputs)
            self.outputs[key] = check
            formats_grid.addWidget(check, i // 3, i % 3)
        right_layout.addWidget(formats_box)
        methods_box = QGroupBox("Calculate and display")
        methods_layout = QVBoxLayout(methods_box)
        self.methods = {}
        for key, label in METHOD_LABELS.items():
            check = QCheckBox(label + (" — experimental" if key in {"standard_csa", "crsa"} else ""))
            check.setChecked(key == "rocking")
            self.methods[key] = check
            methods_layout.addWidget(check)
        form = QFormLayout()
        self.ranking, self.classifier = QComboBox(), QComboBox()
        for key, label in METHOD_LABELS.items():
            self.ranking.addItem(label, key)
            self.classifier.addItem(label, key)
        self.ranking.currentIndexChanged.connect(self.enable_required_method)
        self.classifier.currentIndexChanged.connect(self.enable_required_method)
        form.addRow("Order poses by", self.ranking)
        form.addRow("Robust/metastable selection", self.classifier)
        methods_layout.addLayout(form)
        note = QLabel("CSA/CRSA: two-plane adaptations, not calibrated landing probabilities.\n"
                      "Edge/rolling or coupled-contact cases may be N/A. Legacy 'csa' means CWSA.")
        note.setWordWrap(True)
        methods_layout.addWidget(note)
        right_layout.addWidget(methods_box)
        physics = QGroupBox("Static setup / output options")
        physics_form = QFormLayout(physics)
        self.alpha = self.number(45, 0.01, 89.99)
        self.beta = self.number(0, -88.99, 88.99)
        self.barrier = self.number(0.20, 0, 100, 4)
        self.cwsa_cutoff = self.number(0.65, 0, 1, 4)
        self.classical_cutoff = QLineEdit()
        self.classical_cutoff.setPlaceholderText("Required only for CSA/CRSA classification; sr/mm")
        self.classical_cutoff.setToolTip("User-calibrated raw-score cutoff. There is no published universal threshold.")
        for label, control in (("X tilt (degrees)", self.alpha), ("Y tilt (degrees)", self.beta),
                               ("Rocking cutoff (mm)", self.barrier), ("CWSA cutoff", self.cwsa_cutoff),
                               ("Experimental CSA/CRSA cutoff", self.classical_cutoff)):
            physics_form.addRow(label, control)
        self.robust_only = QCheckBox("Robust poses and robust-only transitions in ALL outputs")
        self.robust_only.setChecked(True)
        self.pose_sheets_include_metastable = QCheckBox("Include metastable poses in pose sheets")
        self.pose_sheets_include_metastable.setChecked(False)
        self.verified = QCheckBox("CAD geometry has been verified")
        physics_form.addRow(self.robust_only)
        physics_form.addRow(self.pose_sheets_include_metastable)
        physics_form.addRow(self.verified)
        self.columns = QSpinBox(); self.columns.setRange(1, 12); self.columns.setValue(6)
        self.per_sheet = QSpinBox(); self.per_sheet.setRange(1, 120); self.per_sheet.setValue(24)
        physics_form.addRow("Pose-sheet columns", self.columns)
        physics_form.addRow("Poses per sheet", self.per_sheet)
        self.existing = QComboBox()
        self.existing.addItem("Skip workpieces with existing output", "skip")
        self.existing.addItem("Overwrite selected formats only", "overwrite")
        physics_form.addRow("Existing files", self.existing)
        physics_form.addRow(QLabel("No prescribed sliding friction or braking loads.\nAt nonzero Y, this does not model longitudinal restraint."))
        right_layout.addWidget(physics)
        split.addWidget(right)
        split.setStretchFactor(0, 1); split.setStretchFactor(1, 1)
        actions = QHBoxLayout()
        self.run_button = QPushButton("Generate selected workpieces")
        self.run_button.clicked.connect(self.start)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        open_button = QPushButton("Open output folder")
        open_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_dir.text())))
        for button in (self.run_button, self.cancel_button, open_button): actions.addWidget(button)
        layout.addLayout(actions)
        self.progress = QProgressBar(); layout.addWidget(self.progress)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(170)
        layout.addWidget(self.log)
        self.input_dir.editingFinished.connect(self.scan)
        self.output_dir.editingFinished.connect(self.refresh_existing)
        self.restore()
        self.scan()

    @staticmethod
    def number(value, low, high, decimals=2):
        control = QDoubleSpinBox(); control.setDecimals(decimals); control.setRange(low, high); control.setValue(value)
        return control

    def browse(self, field):
        path = QFileDialog.getExistingDirectory(self, "Choose folder", field.text())
        if path:
            field.setText(path)
            self.scan() if field is self.input_dir else self.refresh_existing()

    def scan(self):
        folder = Path(self.input_dir.text())
        old = {self.table.item(row, 1).text(): self.table.item(row, 0).checkState() == Qt.CheckState.Checked
               for row in range(self.table.rowCount())}
        if not old:
            old = json.loads(self.settings.value("selected", "{}"))
        self.meshes = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".stl"),
                             key=lambda p: p.name.casefold()) if folder.is_dir() else []
        self.populate_mesh_table(old)

    def toggle_workpiece(self, row):
        item = self.table.item(row, 0)
        selected = item.checkState() == Qt.CheckState.Checked
        item.setCheckState(Qt.CheckState.Unchecked if selected else Qt.CheckState.Checked)

    def populate_mesh_table(self, old):
        self.table.setRowCount(len(self.meshes))
        for row, mesh in enumerate(self.meshes):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            selected = old.get(mesh.name, True)
            check.setCheckState(Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked)
            self.table.setItem(row, 0, check)
            for col, text in ((1, mesh.name), (2, "Pending")):
                item = QTableWidgetItem(text); item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                if col == 1:
                    item.setToolTip(str(mesh.resolve()))
                self.table.setItem(row, col, item)
        self.refresh_existing(); self.filter_rows()

    def filter_rows(self):
        for row, mesh in enumerate(self.meshes):
            self.table.setRowHidden(row, self.search.text().casefold() not in mesh.name.casefold())

    def select_rows(self, selected):
        for row, mesh in enumerate(self.meshes):
            if not self.table.isRowHidden(row):
                self.table.item(row, 0).setCheckState(Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked)

    def refresh_existing(self):
        for row, mesh in enumerate(self.meshes):
            folder = Path(self.output_dir.text()) / mesh.stem
            count = sum(1 for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else 0
            self.table.item(row, 2).setText(f"Existing: {count} files" if count else "Pending")

    def enable_required_method(self):
        for control in (self.ranking, self.classifier):
            self.methods[control.currentData()].setChecked(True)

    def config(self):
        return GenerationConfig(output_dir=self.output_dir.text(),
            outputs=tuple(key for key, check in self.outputs.items() if check.isChecked()),
            methods=tuple(key for key, check in self.methods.items() if check.isChecked()),
            ranking=self.ranking.currentData(), classifier=self.classifier.currentData(),
            alpha=self.alpha.value(), beta=self.beta.value(), rocking_threshold=self.barrier.value(),
            cwsa_threshold=self.cwsa_cutoff.value(),
            classical_threshold=float(self.classical_cutoff.text()) if self.classical_cutoff.text().strip() else None,
            robust_only=self.robust_only.isChecked(),
            pose_sheets_include_metastable=self.pose_sheets_include_metastable.isChecked(),
            geometry_status="verified" if self.verified.isChecked() else "provisional",
            existing=self.existing.currentData(), columns=self.columns.value(), poses_per_sheet=self.per_sheet.value())

    def persist(self):
        self.settings.setValue("input", self.input_dir.text())
        self.settings.setValue("selected", json.dumps({p.name: self.table.item(row, 0).checkState() == Qt.CheckState.Checked
                                                      for row, p in enumerate(self.meshes)}))
        try: self.settings.setValue("config", json.dumps(asdict(self.config())))
        except ValueError: pass
        self.settings.sync()
        if self.settings.status() != QSettings.Status.NoError:
            self.log.appendPlainText("Settings could not be saved: " + self.settings.fileName())

    def restore(self):
        try:
            data = json.loads(self.settings.value("config", "{}"))
            self.input_dir.setText(self.settings.value("input", self.input_dir.text()))
            self.output_dir.setText(data.get("output_dir", self.output_dir.text()))
            for mapping, key in ((self.outputs, "outputs"), (self.methods, "methods")):
                if key in data:
                    for name, check in mapping.items(): check.setChecked(name in data[key])
            for widget, key in ((self.alpha, "alpha"), (self.beta, "beta"), (self.barrier, "rocking_threshold"),
                                (self.cwsa_cutoff, "cwsa_threshold"), (self.columns, "columns"), (self.per_sheet, "poses_per_sheet")):
                if key in data: widget.setValue(data[key])
            for widget, key in ((self.ranking, "ranking"), (self.classifier, "classifier"), (self.existing, "existing")):
                index = widget.findData(data.get(key))
                if index >= 0: widget.setCurrentIndex(index)
            self.robust_only.setChecked(data.get("robust_only", True))
            self.pose_sheets_include_metastable.setChecked(data.get("pose_sheets_include_metastable", False))
            self.verified.setChecked(data.get("geometry_status") == "verified")
            if data.get("classical_threshold") is not None: self.classical_cutoff.setText(str(data["classical_threshold"]))
        except (ValueError, TypeError):
            self.log.appendPlainText("Saved settings could not be read; using defaults.")

    def start(self):
        try:
            config = self.config(); config.validate()
            selected = [str(p.resolve()) for row, p in enumerate(self.meshes)
                        if self.table.item(row, 0).checkState() == Qt.CheckState.Checked]
            if not selected: raise ValueError("Select at least one workpiece.")
            if len({Path(p).stem.casefold() for p in selected}) != len(selected):
                raise ValueError("Duplicate workpiece names would share an output folder.")
        except ValueError as error:
            QMessageBox.warning(self, "Check settings", str(error)); return
        if config.existing == "overwrite" and QMessageBox.question(self, "Overwrite selected formats?",
            "Existing files with matching names will be replaced. Unselected formats are left unchanged.") != QMessageBox.StandardButton.Yes:
            return
        self.persist()
        self.active_parts = [Path(p).stem for p in selected]
        for row, mesh in enumerate(self.meshes):
            if mesh.stem in self.active_parts:
                self.table.item(row, 2).setText("queued")
        self.controls.setEnabled(False); self.run_button.setEnabled(False); self.cancel_button.setEnabled(True)
        self.progress.setRange(0, len(selected)); self.progress.setValue(0)
        self.pending = b""; self.cancelled = False
        process = QProcess(self); self.process = process
        process.setWorkingDirectory(str(ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONPATH", str(ROOT / "src") + os.pathsep + environment.value("PYTHONPATH", ""))
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self.read_output)
        process.finished.connect(self.finished)
        process.errorOccurred.connect(self.process_error)
        request = json.dumps({"meshes": selected, "settings": asdict(config)}).encode("utf-8")
        process.started.connect(lambda: (process.write(request), process.closeWriteChannel()))
        python = str(Path(sys.executable).with_name("python.exe")) if sys.platform == "win32" else sys.executable
        process.start(python, ["-X", "utf8", "-u", "-m", "chute_pose.generate"])

    def read_output(self):
        self.pending += bytes(self.process.readAllStandardOutput())
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace").strip()
            try: event = json.loads(text)
            except ValueError:
                if text: self.log.appendPlainText(text)
                continue
            self.log.appendPlainText(f"{event.get('part', '')}: {event.get('status', '')} {event.get('message', '')}")
            if "detail" in event: self.log.appendPlainText(event["detail"])
            for row, mesh in enumerate(self.meshes):
                if mesh.stem == event.get("part") and "status" in event:
                    self.table.item(row, 2).setText(event["status"])
            if "index" in event: self.progress.setValue(event["index"])

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.log.appendPlainText("Could not start generation: " + self.process.errorString())
            self.finished(1)

    def finished(self, exit_code, *args):
        self.read_output()
        self.controls.setEnabled(True); self.run_button.setEnabled(True); self.cancel_button.setEnabled(False)
        self.log.appendPlainText("Cancelled. Completed files are retained; interrupted staging files may remain."
                                 if self.cancelled else f"Batch finished (exit code {exit_code}).")
        if self.cancelled:
            for row, mesh in enumerate(self.meshes):
                if mesh.stem in self.active_parts and self.table.item(row, 2).text() not in {"completed", "skipped", "failed"}:
                    self.table.item(row, 2).setText("cancelled")
        self.process.deleteLater(); self.process = None

    def cancel(self):
        if self.process:
            self.cancelled = True; self.process.kill()

    def closeEvent(self, event):
        if self.process:
            QMessageBox.information(self, "Generation running", "Cancel the current batch before closing.")
            event.ignore(); return
        self.persist(); event.accept()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = RoadmapGenerator(); window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
