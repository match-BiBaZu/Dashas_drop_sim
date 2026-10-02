"""Prepare independent workpiece runs with shared simulation settings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from collections.abc import Sequence

from .config import RunConfig


@dataclass(frozen=True, slots=True)
class Workpiece:
    mesh_path: Path
    roadmap_path: Path | None = None


def find_roadmap(mesh_path: Path) -> Path | None:
    """Use a local roadmap or one unambiguous sibling-repository handover."""
    mesh_path = Path(mesh_path).expanduser().resolve()
    for suffix in (".yaml", ".yml", ".json"):
        candidate = mesh_path.with_name(f"{mesh_path.stem}_roadmap{suffix}")
        if candidate.is_file():
            return candidate
    root = Path(__file__).resolve().parents[3] / "bibazu_geometry_to_pose" / "Poses_Found_Robust"
    matches = sorted(root.glob(f"{mesh_path.stem}_*/{mesh_path.stem}_roadmap.yaml"))
    return matches[0].resolve() if len(matches) == 1 else None


def batch_configs(base: RunConfig, workpieces: Sequence[Workpiece], output_dir: Path) -> list[RunConfig]:
    """Validate the entire queue before launching; keep result folders distinct."""
    if not workpieces:
        raise ValueError("Bitte mindestens ein Werkstück zur Warteschlange hinzufügen.")
    configs = []
    for index, workpiece in enumerate(workpieces, start=1):
        name = re.sub(r"[^\w.-]+", "_", Path(workpiece.mesh_path).stem).strip(". ") or "workpiece"
        config = replace(
            base, mesh_path=workpiece.mesh_path, roadmap_path=workpiece.roadmap_path,
            output_dir=Path(output_dir) / f"{index:03d}_{name}",
        )
        try:
            config.validate()
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"Werkstück {index} ({name}): {exc}") from exc
        configs.append(config)
    return configs
