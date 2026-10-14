"""Resistance of nominal sliding poses to braking forces and upset torques."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linprog

from .contacts import ContactPose, PoseCatalog, build_pose_catalog
from .frame import ChuteFrame
from .geometry import load_solid_mesh
from .stability import _contact_boundary_indices, estimate_equal_contact_friction


@dataclass(frozen=True, slots=True)
class DisturbanceCapacity:
    """One pose's worst capacities across the sampled friction range."""

    pose_id: int
    critical_floor_braking_m_s2: float
    critical_wall_braking_m_s2: float
    critical_braking_m_s2: float
    critical_braking_g: float
    braking_surface: str
    braking_mu: float
    braking_point_chute_mm: tuple[float, float, float]
    critical_torque_per_mass_mm_m_s2: float
    critical_torque_normalized: float
    torque_principal_axis: int
    torque_sign: int
    torque_mu: float
    composite_reserve: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DisturbanceAnalysis:
    source: str
    alpha_deg: float
    beta_deg: float
    mu_values: tuple[float, ...]
    length_scale_mm: float
    capacities: tuple[DisturbanceCapacity, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "alpha_deg": self.alpha_deg,
            "beta_deg": self.beta_deg,
            "mu_values": self.mu_values,
            "length_scale_mm": self.length_scale_mm,
            "capacities": [capacity.to_dict() for capacity in self.capacities],
        }


@dataclass(frozen=True, slots=True)
class DisturbanceFilterResult:
    minimum_braking_g: float
    minimum_torque_normalized: float
    accepted_pose_ids: tuple[int, ...]
    rejected_pose_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def filter_disturbance_robustness(
    analysis: DisturbanceAnalysis,
    *,
    minimum_braking_g: float = 0.10,
    minimum_torque_normalized: float = 0.02,
) -> DisturbanceFilterResult:
    """Apply explicit, independently calibrated force and torque thresholds."""

    if not math.isfinite(minimum_braking_g) or minimum_braking_g < 0.0:
        raise ValueError("minimum_braking_g must be finite and non-negative.")
    if (
        not math.isfinite(minimum_torque_normalized)
        or minimum_torque_normalized < 0.0
    ):
        raise ValueError(
            "minimum_torque_normalized must be finite and non-negative."
        )
    accepted = tuple(
        capacity.pose_id
        for capacity in analysis.capacities
        if capacity.critical_braking_g >= minimum_braking_g
        and capacity.critical_torque_normalized >= minimum_torque_normalized
    )
    accepted_set = set(accepted)
    rejected = tuple(
        capacity.pose_id
        for capacity in analysis.capacities
        if capacity.pose_id not in accepted_set
    )
    return DisturbanceFilterResult(
        minimum_braking_g=minimum_braking_g,
        minimum_torque_normalized=minimum_torque_normalized,
        accepted_pose_ids=accepted,
        rejected_pose_ids=rejected,
    )


def _normal_force_system(
    pose: ContactPose,
    points: np.ndarray,
    gravity: np.ndarray,
    mu: float,
    contact_tolerance_mm: float,
    length_scale_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    floor_indices = _contact_boundary_indices(
        points, pose.floor_contact_vertex_indices, (0, 1), contact_tolerance_mm
    )
    wall_indices = _contact_boundary_indices(
        points, pose.wall_contact_vertex_indices, (0, 2), contact_tolerance_mm
    )
    floor_count = len(floor_indices)
    wall_count = len(wall_indices)
    floor_normal = -float(gravity[2])
    wall_normal = -float(gravity[1])
    if floor_normal <= 0.0 or wall_normal <= 0.0:
        raise ValueError("Gravity must press the part into both chute surfaces.")

    floor_force = np.array([-mu, 0.0, 1.0])
    wall_force = np.array([-mu, 1.0, 0.0])
    floor_torques = np.cross(points[floor_indices], floor_force)
    wall_torques = np.cross(points[wall_indices], wall_force)
    equalities = np.zeros((5, floor_count + wall_count), dtype=float)
    equalities[0, :floor_count] = 1.0
    equalities[1, floor_count:] = 1.0
    equalities[2:5, :floor_count] = floor_torques.T / length_scale_mm
    equalities[2:5, floor_count:] = wall_torques.T / length_scale_mm
    targets = np.array([floor_normal, wall_normal, 0.0, 0.0, 0.0])
    return equalities, targets, floor_indices, wall_indices


def _maximum_disturbance(
    equalities: np.ndarray,
    targets: np.ndarray,
    disturbance_torque_per_unit: np.ndarray,
    length_scale_mm: float,
) -> float:
    variable_count = equalities.shape[1] + 1
    augmented = np.zeros((5, variable_count), dtype=float)
    augmented[:, :-1] = equalities
    augmented[2:5, -1] = disturbance_torque_per_unit / length_scale_mm
    objective = np.zeros(variable_count)
    objective[-1] = -1.0
    solution = linprog(
        objective,
        A_eq=augmented,
        b_eq=targets,
        bounds=[(0.0, None)] * variable_count,
        method="highs",
    )
    if solution.status == 3:
        return math.inf
    if not solution.success:
        return 0.0
    return max(0.0, float(solution.x[-1]))


def _pose_capacity(
    pose: ContactPose,
    vertices_centered: np.ndarray,
    principal_axes_part: np.ndarray,
    gravity: np.ndarray,
    gravity_magnitude: float,
    mu_values: np.ndarray,
    contact_tolerance_mm: float,
    length_scale_mm: float,
) -> DisturbanceCapacity:
    rotation = np.asarray(pose.rotation_chute_from_part, dtype=float)
    points = (rotation @ vertices_centered.T).T
    floor_records: list[tuple[float, float, np.ndarray]] = []
    wall_records: list[tuple[float, float, np.ndarray]] = []
    torque_records: list[tuple[float, float, int, int]] = []

    for mu in mu_values:
        equalities, targets, floor_indices, wall_indices = _normal_force_system(
            pose,
            points,
            gravity,
            float(mu),
            contact_tolerance_mm,
            length_scale_mm,
        )
        for surface, indices, records in (
            ("floor", floor_indices, floor_records),
            ("wall", wall_indices, wall_records),
        ):
            for point_index in indices:
                point = points[point_index]
                braking_torque = np.cross(point, np.array([-1.0, 0.0, 0.0]))
                capacity = _maximum_disturbance(
                    equalities, targets, braking_torque, length_scale_mm
                )
                records.append((capacity, float(mu), point.copy()))

        for axis_index, axis_part in enumerate(principal_axes_part.T):
            axis_chute = rotation @ axis_part
            for sign in (-1, 1):
                capacity = _maximum_disturbance(
                    equalities,
                    targets,
                    float(sign) * axis_chute,
                    length_scale_mm,
                )
                torque_records.append((capacity, float(mu), axis_index, sign))

    floor_capacity, floor_mu, floor_point = min(floor_records, key=lambda item: item[0])
    wall_capacity, wall_mu, wall_point = min(wall_records, key=lambda item: item[0])
    if floor_capacity <= wall_capacity:
        braking_capacity = floor_capacity
        braking_mu = floor_mu
        braking_point = floor_point
        braking_surface = "floor"
    else:
        braking_capacity = wall_capacity
        braking_mu = wall_mu
        braking_point = wall_point
        braking_surface = "wall"
    torque_capacity, torque_mu, torque_axis, torque_sign = min(
        torque_records, key=lambda item: item[0]
    )
    braking_normalized = braking_capacity / gravity_magnitude
    torque_normalized = torque_capacity / (gravity_magnitude * length_scale_mm)
    return DisturbanceCapacity(
        pose_id=pose.pose_id,
        critical_floor_braking_m_s2=floor_capacity,
        critical_wall_braking_m_s2=wall_capacity,
        critical_braking_m_s2=braking_capacity,
        critical_braking_g=braking_normalized,
        braking_surface=braking_surface,
        braking_mu=braking_mu,
        braking_point_chute_mm=tuple(float(value) for value in braking_point),
        critical_torque_per_mass_mm_m_s2=torque_capacity,
        critical_torque_normalized=torque_normalized,
        torque_principal_axis=torque_axis,
        torque_sign=torque_sign,
        torque_mu=torque_mu,
        composite_reserve=min(braking_normalized, torque_normalized),
    )


def analyze_disturbance_robustness(
    mesh_path: str | Path,
    *,
    pose_ids: Iterable[int] | None = None,
    alpha_deg: float = 45.0,
    beta_deg: float = 20.0,
    onset_alpha_deg: float = 45.0,
    onset_beta_deg: float = 15.0,
    mu_samples: int = 11,
    catalog: PoseCatalog | None = None,
) -> DisturbanceAnalysis:
    """Find critical -X braking and principal-axis upset moments per pose.

    Capacities are minimised across contact boundary points, both torque
    signs, all three mass-principal axes, and the sampled friction range.
    """

    if mu_samples < 2:
        raise ValueError("mu_samples must be at least 2.")
    frame = ChuteFrame(alpha_deg=alpha_deg, beta_deg=beta_deg)
    gravity = frame.gravity_chute()
    gravity_magnitude = float(np.linalg.norm(gravity))
    friction = estimate_equal_contact_friction(
        onset_alpha_deg=onset_alpha_deg, onset_beta_deg=onset_beta_deg
    )
    mu_values = np.linspace(0.0, friction.mu_static_estimate, mu_samples)
    pose_catalog = catalog or build_pose_catalog(mesh_path)
    requested = set(pose_ids) if pose_ids is not None else None
    selected = [
        pose
        for pose in pose_catalog.poses
        if requested is None or pose.pose_id in requested
    ]
    if requested is not None:
        missing = requested - {pose.pose_id for pose in selected}
        if missing:
            raise ValueError(f"Unknown pose ids: {sorted(missing)}")

    mesh = load_solid_mesh(mesh_path)
    hull = mesh.convex_hull
    vertices_centered = np.asarray(hull.vertices, dtype=float) - np.asarray(
        mesh.center_mass, dtype=float
    )
    length_scale = max(float(np.max(hull.extents)), 1e-9)
    _, principal_axes = np.linalg.eigh(np.asarray(mesh.moment_inertia, dtype=float))
    capacities = tuple(
        _pose_capacity(
            pose,
            vertices_centered,
            principal_axes,
            gravity,
            gravity_magnitude,
            mu_values,
            pose_catalog.contact_tolerance_mm,
            length_scale,
        )
        for pose in selected
    )
    return DisturbanceAnalysis(
        source=str(Path(mesh_path).expanduser().resolve()),
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        mu_values=tuple(float(value) for value in mu_values),
        length_scale_mm=length_scale,
        capacities=capacities,
    )
