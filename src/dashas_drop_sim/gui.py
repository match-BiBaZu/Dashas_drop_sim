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

from PyQt6.QtCore import QProcess, QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import RunConfig


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

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BiBaZu Falltest-Simulation")
        self.resize(1020, 850)
        self._process: QProcess | None = None
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._cancel_requested = False
        self._run_started_at = 0.0
        self._mode = ""
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        page = QWidget()
        layout = QVBoxLayout(page)

        files = QGroupBox("Eingaben und Ausgabe")
        file_form = QFormLayout(files)
        self.mesh_edit = QLineEdit()
        file_form.addRow("Bauteil (STL oder STEP)", self._path_row(self.mesh_edit, self._choose_mesh))
        self.roadmap_edit = QLineEdit()
        self.roadmap_edit.setPlaceholderText("Optional: Roadmap zum Abgleich der Pose-IDs")
        file_form.addRow("Roadmap", self._path_row(self.roadmap_edit, self._choose_roadmap))
        self.output_edit = QLineEdit(str(_default("output_dir", Path.cwd() / "results")))
        file_form.addRow("Ergebnisordner", self._path_row(self.output_edit, self._choose_output))
        layout.addWidget(files)

        parameters = QGroupBox("Fallversuche")
        form = QFormLayout(parameters)
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(1, 1_000_000)
        self.trials_spin.setValue(int(_default("trials", 100)))
        form.addRow("Anzahl Versuche", self.trials_spin)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(int(_default("seed", 0)))
        form.addRow("Zufalls-Seed", self.seed_spin)
        self.belt_spin = _spin(_default("belt_speed_mm_s", 100), 0, 200)
        self.belt_spin.setSuffix(" mm/s")
        form.addRow("Bandgeschwindigkeit", self.belt_spin)
        self.height_spin = _spin(_default("drop_height_mm", 100), 0, 200)
        self.height_spin.setSuffix(" mm")
        form.addRow("Abwurfhöhe", self.height_spin)
        self.lateral_spin = _spin(_default("lateral_mm", 0), -100, 100)
        self.lateral_spin.setSuffix(" mm")
        self.lateral_spin.setToolTip("Seitlicher Versatz von der Winkelhalbierenden zwischen Band und Wand")
        form.addRow("Querposition", self.lateral_spin)
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

        advanced = QGroupBox("Erweiterte Simulationsparameter")
        advanced_form = QFormLayout(advanced)
        self.alpha_spin = _spin(_default("alpha_deg", 45), 0, 90, 1)
        self.alpha_spin.setSuffix(" °")
        advanced_form.addRow("Querneigung", self.alpha_spin)
        self.beta_spin = _spin(_default("beta_deg", 0), -30, 30, 1)
        self.beta_spin.setSuffix(" °")
        advanced_form.addRow("Längsneigung", self.beta_spin)
        self.length_spin = _spin(_default("length_mm", 3000), 100, 100_000, 0, 100)
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
        self.preview_button = QPushButton("Einzelfall anzeigen")
        self.preview_button.clicked.connect(lambda: self._start("preview"))
        self.cancel_button = QPushButton("Abbrechen")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        controls.addWidget(self.start_button)
        controls.addWidget(self.preview_button)
        controls.addWidget(self.cancel_button)
        controls.addStretch()
        layout.addLayout(controls)

        self.status_label = QLabel("Bereit")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)

        self.result_table = QTableWidget(0, 5)
        self.result_table.setHorizontalHeaderLabels(["Pose / Status", "Anzahl", "Anteil", "KI unten", "KI oben"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.setMinimumHeight(180)
        layout.addWidget(self.result_table)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(170)
        layout.addWidget(self.log)
        outer.setWidget(page)
        self.setCentralWidget(outer)

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
            self.mesh_edit.setText(path)

    def _choose_roadmap(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Roadmap wählen", self.roadmap_edit.text() or str(Path.cwd()),
            "Roadmaps (*.yaml *.yml *.json);;Alle Dateien (*)",
        )
        if path:
            self.roadmap_edit.setText(path)

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Ergebnisordner wählen", self.output_edit.text() or str(Path.cwd())
        )
        if path:
            self.output_edit.setText(path)

    def _make_config(self) -> RunConfig:
        mesh = self.mesh_edit.text().strip()
        output = self.output_edit.text().strip()
        if not mesh:
            raise ValueError("Bitte ein STL- oder STEP-Bauteil wählen.")
        if not output:
            raise ValueError("Bitte einen Ergebnisordner wählen.")
        level_text = self.levels_edit.text().strip()
        try:
            levels = tuple(float(part.strip()) for part in level_text.split(",") if part.strip())
        except ValueError as exc:
            raise ValueError("Störstufen bitte als kommagetrennte Zahlen eingeben.") from exc
        if not levels:
            raise ValueError("Mindestens eine Störstufe ist erforderlich.")
        config = RunConfig(
            mesh_path=Path(mesh).expanduser().resolve(),
            roadmap_path=Path(self.roadmap_edit.text().strip()).expanduser().resolve()
            if self.roadmap_edit.text().strip() else None,
            output_dir=Path(output).expanduser().resolve(),
            trials=self.trials_spin.value(),
            seed=self.seed_spin.value(),
            belt_speed_mm_s=self.belt_spin.value(),
            drop_height_mm=self.height_spin.value(),
            lateral_mm=self.lateral_spin.value(),
            density_g_cm3=self.density_spin.value(),
            mu_belt=self.belt_mu_spin.value(),
            mu_wall=self.wall_mu_spin.value(),
            alpha_deg=self.alpha_spin.value(),
            beta_deg=self.beta_spin.value(),
            length_mm=self.length_spin.value(),
            timestep_s=self.timestep_spin.value(),
            disturbance_levels_mm=levels,
        )
        config.validate()
        return config

    def _start(self, mode: str) -> None:
        if self._process is not None:
            return
        try:
            config = self._make_config()
            config.output_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            config_path = config.output_dir / f"gui_config_{stamp}.json"
            config_path.write_text(
                json.dumps(config.to_dict(), ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Eingabe prüfen", str(exc))
            return

        self._mode = mode
        self._run_started_at = time.time()
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._cancel_requested = False
        self.result_table.setRowCount(0)
        self.progress.setRange(0, 0 if mode == "preview" else config.trials)
        self.progress.setValue(0)
        self.status_label.setText("Einzelfall läuft …" if mode == "preview" else "Falltests laufen …")
        self.log.clear()
        self._log_line(f"Konfiguration: {config_path}")
        self.start_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        process = QProcess(self)
        process.setProgram(sys.executable)
        args = ["-m", "dashas_drop_sim", mode, "--config", str(config_path)]
        if mode == "preview":
            args.extend(["--trial", "0"])
        process.setArguments(args)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self._process = process
        process.start()

    def _cancel(self) -> None:
        process = self._process
        if process is None or self._cancel_requested:
            return
        self._cancel_requested = True
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Abbruch angefordert …")
        self._log_line("Abbruch angefordert; laufender Versuch darf abschließen.")
        process.write(b"stop\n")
        QTimer.singleShot(15_000, self._force_stop_if_needed)

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
            return True
        if kind in {"result", "pose_frequency"}:
            self._set_result_rows([event], append=True)
            return True
        if kind in {"summary", "done"}:
            rows = self._extract_rows(event)
            if rows:
                self._set_result_rows(rows)
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
                self.status_label.setText("Simulation konnte nicht gestartet werden")
                self.start_button.setEnabled(True)
                self.preview_button.setEnabled(True)
                self.cancel_button.setEnabled(False)
                self._process.deleteLater()
                self._process = None

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
        if self._cancel_requested:
            self.status_label.setText("Abgebrochen")
        elif status == QProcess.ExitStatus.NormalExit and code == 0:
            self.status_label.setText("Einzelfall abgeschlossen" if self._mode == "preview" else "Simulation abgeschlossen")
            if self._mode == "run":
                self.progress.setValue(self.progress.maximum())
        else:
            self.status_label.setText(f"Simulation fehlgeschlagen (Exitcode {code})")
        self.start_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if self._process is not None:
            self._process.deleteLater()
            self._process = None

    def _load_result_file(self) -> None:
        """Show the newest summary from this run, even when CLI emits plain text."""
        base = Path(self.output_edit.text().strip()).expanduser()
        if not base.is_dir():
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
                    return
            except (OSError, ValueError, TypeError) as exc:
                self._log_line(f"Ergebnistabelle konnte nicht gelesen werden: {exc}")

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


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = DropSimulationWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
