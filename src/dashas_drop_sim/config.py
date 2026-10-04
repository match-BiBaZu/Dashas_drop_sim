"""Validated, serializable inputs shared by the GUI and batch runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RunConfig:
    mesh_path: Path
    output_dir: Path = Path("results")
    roadmap_path: Path | None = None
    catalog_repo: Path | None = None
    catalog_cad_dir: Path | None = None
    catalog_push: bool = True
    compute_stability: bool = True
    trials: int = 100
    seed: int = 42
    belt_speed_mm_s: float = 100.0
    belt_speed_variation_mm_s: float = 0.0
    belt_variation_interval_s: float = 0.2
    drop_height_mm: float = 100.0
    drop_height_spread_mm: float = 0.0
    lateral_mm: float = 0.0
    lateral_spread_mm: float = 0.0
    density_g_cm3: float = 1.15
    mu_belt: float = 0.40
    mu_wall: float = 0.20
    alpha_deg: float = 45.0
    beta_deg: float = 0.0
    length_mm: float = 1300.0
    workers: int = max(1, min(4, os.cpu_count() or 1))
    timestep_s: float = 0.001
    roughness_enabled: bool = False
    roughness_model: str = "longitudinal_traction"
    roughness_wall_height_mm: float = 0.1
    roughness_wall_ramp_mm: float = 3.0
    roughness_wall_spacing_mm: float = 10.0
    roughness_belt_height_mm: float = 0.0
    roughness_belt_ramp_mm: float = 3.0
    roughness_belt_spacing_mm: float = 10.0
    disturbance_levels_mm: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)

    def validate(self) -> None:
        for name in ("catalog_repo", "catalog_cad_dir"):
            value = getattr(self, name)
            setattr(self, name, None if value in (None, "") else Path(value).expanduser().resolve())
        self.mesh_path = Path(self.mesh_path).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.roadmap_path = (
            None if self.roadmap_path in (None, "") else Path(self.roadmap_path).expanduser().resolve()
        )
        if not self.mesh_path.is_file() or self.mesh_path.suffix.lower() not in {".stl", ".step", ".stp"}:
            raise ValueError("Select an existing STL or STEP part file.")
        if self.roadmap_path is not None and (
            not self.roadmap_path.is_file() or self.roadmap_path.suffix.lower() not in {".json", ".yaml", ".yml"}
        ):
            raise ValueError("The optional roadmap must be an existing JSON or YAML file.")
        self.validate_parameters()

    def validate_parameters(self) -> None:
        """Validate simulation settings even before a workpiece is selected."""
        if not isinstance(self.compute_stability, bool) or not isinstance(self.catalog_push, bool):
            raise ValueError("compute_stability and catalog_push must be true or false")
        if not isinstance(self.trials, int) or self.trials < 1:
            raise ValueError("trials must be a positive integer")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.workers, int) or not 1 <= self.workers <= 32:
            raise ValueError("workers must be an integer between 1 and 32")
        if not isinstance(self.roughness_enabled, bool):
            raise ValueError("roughness_enabled must be true or false")
        if self.roughness_model not in {"longitudinal_traction", "microfacet"}:
            raise ValueError("roughness_model must be longitudinal_traction or microfacet")
        bounds = (
            ("belt_speed_mm_s", self.belt_speed_mm_s, 0, 200),
            ("belt_speed_variation_mm_s", self.belt_speed_variation_mm_s, 0, 20),
            ("belt_variation_interval_s", self.belt_variation_interval_s, 0.02, 10),
            ("drop_height_mm", self.drop_height_mm, 0, 200),
            ("drop_height_spread_mm", self.drop_height_spread_mm, 0, 100),
            ("lateral_mm", self.lateral_mm, -100, 100),
            ("lateral_spread_mm", self.lateral_spread_mm, 0, 100),
            ("density_g_cm3", self.density_g_cm3, 0.01, 30),
            ("mu_belt", self.mu_belt, 0, 5),
            ("mu_wall", self.mu_wall, 0, 5),
            ("length_mm", self.length_mm, 100, 100000),
            ("timestep_s", self.timestep_s, 0.0001, 0.01),
            *[(f"roughness_{surface}_{parameter}_mm", getattr(self, f"roughness_{surface}_{parameter}_mm"), low, high)
              for surface in ("wall", "belt")
              for parameter, low, high in (("height", 0, 2), ("ramp", 0.1, 100), ("spacing", 1, 10000))],
        )
        for name, value, lower, upper in bounds:
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
        for center, spread, low, high, label in (
            (self.belt_speed_mm_s, self.belt_speed_variation_mm_s, 0, 200, "Bandgeschwindigkeit"),
            (self.drop_height_mm, self.drop_height_spread_mm, 0, 200, "Abwurfhöhe"),
            (self.lateral_mm, self.lateral_spread_mm, -100, 100, "Querposition"),
        ):
            if center - spread < low - 1e-9 or center + spread > high + 1e-9:
                raise ValueError(f"{label}: Sollwert ± Streubreite muss zwischen {low} und {high} mm liegen.")
        for surface in ("wall", "belt"):
            height = getattr(self, f"roughness_{surface}_height_mm")
            ramp = getattr(self, f"roughness_{surface}_ramp_mm")
            if self.roughness_enabled and height / ramp > 0.25:
                raise ValueError(f"roughness_{surface}: height/ramp must be at most 0.25 for small imperfections")
        if not all(math.isfinite(x) for x in (self.alpha_deg, self.beta_deg)):
            raise ValueError("Chute angles must be finite")
        if not 0 <= self.alpha_deg <= 90 or not -30 <= self.beta_deg <= 30:
            raise ValueError("Chute angles outside supported range")
        levels = tuple(float(x) for x in self.disturbance_levels_mm)
        if not levels or levels[0] != 0 or any(not math.isfinite(x) or x < 0 for x in levels):
            raise ValueError("Disturbance levels must start at zero and be non-negative")
        if levels != tuple(sorted(set(levels))):
            raise ValueError("Disturbance levels must be unique and increasing")
        self.disturbance_levels_mm = levels

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["mesh_path"] = str(Path(self.mesh_path).expanduser().resolve())
        result["output_dir"] = str(Path(self.output_dir).expanduser().resolve())
        result["roadmap_path"] = None if self.roadmap_path is None else str(Path(self.roadmap_path).expanduser().resolve())
        for name in ("catalog_repo", "catalog_cad_dir"):
            value = getattr(self, name)
            result[name] = None if value is None else str(Path(value).expanduser().resolve())
        return result

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        allowed = cls.__dataclass_fields__.keys()
        extras = set(raw) - set(allowed)
        if extras:
            raise ValueError(f"Unknown configuration keys: {sorted(extras)}")
        payload = dict(raw)
        for key in ("mesh_path", "output_dir", "roadmap_path", "catalog_repo", "catalog_cad_dir"):
            if payload.get(key) is not None:
                payload[key] = Path(payload[key])
        if "disturbance_levels_mm" in payload:
            payload["disturbance_levels_mm"] = tuple(payload["disturbance_levels_mm"])
        config = cls(**payload)
        config.validate()
        return config
