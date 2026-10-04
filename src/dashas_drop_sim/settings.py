"""Explicitly saved GUI defaults, kept in the user's application settings."""

from dataclasses import fields
import json
import os
from pathlib import Path

from .config import RunConfig


INPUT_KEYS = ("mesh_path", "roadmap_path", "output_dir", "catalog_repo", "catalog_cad_dir")
PARAMETER_KEYS = frozenset(field.name for field in fields(RunConfig)) - frozenset(INPUT_KEYS)


def default_settings_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return root / "BiBaZu" / "DashasDropSim" / "gui_defaults.json"
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "dashas-drop-sim" / "gui_defaults.json"


def _validated(parameters: dict, inputs: dict) -> dict:
    if not isinstance(parameters, dict) or not isinstance(inputs, dict):
        raise ValueError("Die gespeicherten GUI-Einstellungen sind ungültig.")
    parameters = {name: value for name, value in parameters.items() if name in PARAMETER_KEYS}
    config = RunConfig(mesh_path=Path("unused.stl"), **parameters)
    config.validate_parameters()
    if any(not isinstance(value, str) for name, value in inputs.items() if name in INPUT_KEYS):
        raise ValueError("Gespeicherte Datei- und Ordnerpfade müssen Text sein.")
    return {"schema_version": 1, "parameters": parameters,
            "inputs": {name: value for name, value in inputs.items() if name in INPUT_KEYS}}


def load_defaults(path: Path) -> dict | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unbekanntes Format der gespeicherten GUI-Einstellungen.")
    return _validated(data.get("parameters", {}), data.get("inputs", {}))


def save_defaults(path: Path, parameters: dict, inputs: dict) -> None:
    data = _validated(parameters, inputs)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
