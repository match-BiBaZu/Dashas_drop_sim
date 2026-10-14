"""Quasi-static stability filtering for parts sliding along the chute corner."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from .contacts import ContactPose, PoseCatalog, build_pose_catalog
from .frame import ChuteFrame
from .geometry import load_solid_mesh

STABILITY_ALGORITHM_LABEL = "quasi-static force/moment equilibrium"
STABILITY_VALUE_LABEL = (
    "minimum contact load balance index across sampled friction range"
)
FrictionPolicy = Literal["range", "zero"]


@dataclass(frozen=True, slots=True)
class FrictionEstimate:
    """Equal floor/wall friction inferred from the observed slide onset."""

    onset_alpha_deg: float
    onset_beta_deg: float
    mu_static_estimate: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StabilitySample:
    """Result for one pose at one assumed sliding-friction coefficient."""

    mu: float
    equilibrium_feasible: bool
    pressure_margin: float
    acceleration_x_m_s2: float
    stable: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PoseStability:
    """Friction-range result for one theoretical contact pose."""

    pose_id: int
    floor_contact_type: str
    wall_contact_type: str
    samples: tuple[StabilitySample, ...]
    stable_across_range: bool
    stable_at_any_sample: bool
    minimum_pressure_margin: float

    @property
    def minimum_contact_load_balance_index(self) -> float:
        """Return the dimensionless normalized contact-load margin on 0..1."""

        return min(1.0, max(0.0, self.minimum_pressure_margin))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["samples"] = [sample.to_dict() for sample in self.samples]
        result["minimum_contact_load_balance_index"] = (
            self.minimum_contact_load_balance_index
        )
        return result


@dataclass(frozen=True, slots=True)
class StabilityAnalysis:
    """Complete Step-3 result for a mesh, chute orientation and mu range."""

    source: str
    alpha_deg: float
    beta_deg: float
    gravity_chute_m_s2: tuple[float, float, float]
    friction_estimate: FrictionEstimate
    mu_values: tuple[float, ...]
    poses: tuple[PoseStability, ...]
    input_pose_count: int | None = None
    quasistatic_only: bool = False
    friction_policy: FrictionPolicy = "zero"

    @property
    def stable_pose_ids(self) -> tuple[int, ...]:
        return tuple(pose.pose_id for pose in self.poses if pose.stable_across_range)

    @property
    def friction_dependent_pose_ids(self) -> tuple[int, ...]:
        return tuple(
            pose.pose_id
            for pose in self.poses
            if pose.stable_at_any_sample and not pose.stable_across_range
        )

    @property
    def rejected_pose_ids(self) -> tuple[int, ...]:
        return tuple(
            pose.pose_id for pose in self.poses if not pose.stable_at_any_sample
        )

    def to_dict(self) -> dict[str, Any]:
        diagnostics_omitted = self.quasistatic_only or self.friction_policy == "zero"
        friction_dependent_pose_ids: tuple[int, ...] | None = (
            None if diagnostics_omitted else self.friction_dependent_pose_ids
        )
        rejected_pose_ids: tuple[int, ...] | None = (
            None if diagnostics_omitted else self.rejected_pose_ids
        )
        return {
            "source": self.source,
            "alpha_deg": self.alpha_deg,
            "beta_deg": self.beta_deg,
            "gravity_chute_m_s2": self.gravity_chute_m_s2,
            "friction_estimate": self.friction_estimate.to_dict(),
            "mu_values": self.mu_values,
            "friction_policy": self.friction_policy,
            "evaluation_mode": (
                "zero_friction_nominal"
                if self.friction_policy == "zero"
                else (
                    "quasistatic_accepted_only"
                    if self.quasistatic_only
                    else "exhaustive_friction_diagnostics"
                )
            ),
            "input_pose_count": (
                len(self.poses)
                if self.input_pose_count is None
                else self.input_pose_count
            ),
            "stable_pose_ids": self.stable_pose_ids,
            "friction_dependent_pose_ids": friction_dependent_pose_ids,
            "rejected_pose_ids": rejected_pose_ids,
            "poses": [pose.to_dict() for pose in self.poses],
        }


def estimate_equal_contact_friction(
    *, onset_alpha_deg: float, onset_beta_deg: float
) -> FrictionEstimate:
    """Estimate static mu from onset while both PTFE surfaces are contacted.

    With equal coefficients at floor and wall, impending +X sliding gives

    ``mu = tan(beta) / (sin(alpha) + cos(alpha))``.
    """

    if not math.isfinite(onset_alpha_deg) or not math.isfinite(onset_beta_deg):
        raise ValueError("Onset angles must be finite numbers.")
    alpha = math.radians(onset_alpha_deg)
    beta = math.radians(onset_beta_deg)
    denominator = math.sin(alpha) + math.cos(alpha)
    if denominator <= 0.0 or beta < 0.0 or beta >= math.pi / 2.0:
        raise ValueError("Onset angles do not define a positive finite friction estimate.")
    mu = math.tan(beta) / denominator
    if not math.isfinite(mu) or mu < 0.0:
        raise ValueError("Onset angles do not define a positive finite friction estimate.")
    return FrictionEstimate(onset_alpha_deg, onset_beta_deg, mu)


def _unique_projected_indices(
    points: NDArray[np.float64],
    indices: Iterable[int],
    projection_axes: tuple[int, int],
    tolerance: float,
) -> NDArray[np.int64]:
    candidates = np.asarray(tuple(indices), dtype=np.int64)
    projected = points[candidates][:, projection_axes]
    scale = max(tolerance, np.finfo(float).eps)
    keys = np.round(projected / scale).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return candidates[np.sort(first)]


def _contact_boundary_indices(
    points: NDArray[np.float64],
    indices: Iterable[int],
    projection_axes: tuple[int, int],
    tolerance: float,
) -> NDArray[np.int64]:
    """Return endpoints or polygon corners spanning a contact region."""

    unique_indices = _unique_projected_indices(
        points, indices, projection_axes, tolerance
    )
    projected = points[unique_indices][:, projection_axes]
    if len(projected) <= 2:
        return unique_indices

    centered = projected - np.mean(projected, axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    dimension = int(np.count_nonzero(singular_values > tolerance))
    if dimension >= 2:
        return unique_indices[ConvexHull(projected).vertices]

    # Collinear contacts need only their two extremal endpoints.
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    coordinate = centered @ right_vectors[0]
    endpoint_positions = np.array([np.argmin(coordinate), np.argmax(coordinate)])
    return unique_indices[endpoint_positions]


def _solve_pose_sample(
    pose: ContactPose,
    vertices_centered: NDArray[np.float64],
    gravity: NDArray[np.float64],
    mu: float,
    contact_tolerance_mm: float,
    length_scale_mm: float,
    margin_tolerance: float,
) -> StabilitySample:
    rotation = np.asarray(pose.rotation_chute_from_part, dtype=float)
    points = (rotation @ vertices_centered.T).T
    floor_indices = _contact_boundary_indices(
        points,
        pose.floor_contact_vertex_indices,
        (0, 1),
        contact_tolerance_mm,
    )
    wall_indices = _contact_boundary_indices(
        points,
        pose.wall_contact_vertex_indices,
        (0, 2),
        contact_tolerance_mm,
    )

    floor_normal = -float(gravity[2])
    wall_normal = -float(gravity[1])
    acceleration_x = float(gravity[0] - mu * (floor_normal + wall_normal))
    if floor_normal <= 0.0 or wall_normal <= 0.0:
        return StabilitySample(
            mu, False, 0.0, acceleration_x, False, "gravity_loses_floor_or_wall_contact"
        )

    floor_force = np.array([-mu, 0.0, 1.0])
    wall_force = np.array([-mu, 1.0, 0.0])
    floor_torques = np.cross(points[floor_indices], floor_force)
    wall_torques = np.cross(points[wall_indices], wall_force)
    floor_count = len(floor_indices)
    wall_count = len(wall_indices)
    variable_count = floor_count + wall_count + 1
    margin_index = variable_count - 1

    equalities = np.zeros((5, variable_count), dtype=float)
    equalities[0, :floor_count] = 1.0
    equalities[1, floor_count : floor_count + wall_count] = 1.0
    equalities[2:5, :floor_count] = floor_torques.T / length_scale_mm
    equalities[2:5, floor_count : floor_count + wall_count] = (
        wall_torques.T / length_scale_mm
    )
    targets = np.array([floor_normal, wall_normal, 0.0, 0.0, 0.0])

    # Maximise the smallest normalised vertex load. A positive optimum means
    # the required wrench is in the interior of the available contact-wrench
    # set, rather than exactly on a tipping boundary.
    inequalities = np.zeros((floor_count + wall_count, variable_count))
    upper_bounds = np.zeros(floor_count + wall_count)
    for index in range(floor_count):
        inequalities[index, index] = -1.0
        inequalities[index, margin_index] = floor_normal / floor_count
    for local_index in range(wall_count):
        row = floor_count + local_index
        inequalities[row, floor_count + local_index] = -1.0
        inequalities[row, margin_index] = wall_normal / wall_count

    objective = np.zeros(variable_count)
    objective[margin_index] = -1.0
    solution = linprog(
        objective,
        A_ub=inequalities,
        b_ub=upper_bounds,
        A_eq=equalities,
        b_eq=targets,
        bounds=[(0.0, None)] * (variable_count - 1) + [(0.0, 1.0)],
        method="highs",
    )
    if not solution.success:
        return StabilitySample(
            mu, False, 0.0, acceleration_x, False, "no_force_moment_equilibrium"
        )

    margin = float(solution.x[margin_index])
    if acceleration_x < -1e-9:
        return StabilitySample(
            mu, True, margin, acceleration_x, False, "would_not_move_in_positive_x"
        )
    if margin <= margin_tolerance:
        return StabilitySample(
            mu, True, margin, acceleration_x, False, "marginal_tipping_boundary"
        )
    return StabilitySample(mu, True, margin, acceleration_x, True, "stable")


def analyze_pose_stability(
    mesh_path: str | Path,
    *,
    alpha_deg: float = 45.0,
    beta_deg: float = 20.0,
    onset_alpha_deg: float = 45.0,
    onset_beta_deg: float = 15.0,
    mu_samples: int = 11,
    margin_tolerance: float = 1e-6,
    catalog: PoseCatalog | None = None,
    quasistatic_only: bool = False,
    friction_policy: FrictionPolicy = "zero",
) -> StabilityAnalysis:
    """Filter poses using nominal ``mu = 0`` or an optional friction range.

    When ``quasistatic_only`` is true, the returned pose records contain only
    poses that are stable at every sampled coefficient.  The zero-friction
    sample is evaluated first and failures are discarded immediately because
    such a pose cannot pass the complete range.  This preserves the accepted
    quasistatic set exactly while intentionally omitting the much more
    expensive friction-dependent and rejected-pose diagnostics.

    The default ``friction_policy="zero"`` evaluates one nominal
    zero-friction sample and does not classify friction-dependent behaviour.
    Select ``"range"`` explicitly for the inferred static-friction sweep.
    """

    if friction_policy not in {"range", "zero"}:
        raise ValueError("friction_policy must be 'range' or 'zero'.")
    if friction_policy == "range" and mu_samples < 2:
        raise ValueError("mu_samples must be at least 2 for range policy.")
    if not math.isfinite(margin_tolerance) or margin_tolerance < 0.0:
        raise ValueError("margin_tolerance must be a finite non-negative number.")

    frame = ChuteFrame(alpha_deg=alpha_deg, beta_deg=beta_deg)
    gravity = frame.gravity_chute()
    estimate = estimate_equal_contact_friction(
        onset_alpha_deg=onset_alpha_deg, onset_beta_deg=onset_beta_deg
    )
    mu_values = (
        np.array([0.0], dtype=float)
        if friction_policy == "zero"
        else np.linspace(0.0, estimate.mu_static_estimate, mu_samples)
    )
    pose_catalog = catalog or build_pose_catalog(mesh_path)

    mesh = load_solid_mesh(mesh_path)
    hull = mesh.convex_hull
    center_mass = np.asarray(mesh.center_mass, dtype=float)
    vertices_centered = np.asarray(hull.vertices, dtype=float) - center_mass
    length_scale = max(float(np.max(hull.extents)), 1e-9)

    results: list[PoseStability] = []
    for pose in pose_catalog.poses:
        if quasistatic_only:
            first_sample = _solve_pose_sample(
                pose,
                vertices_centered,
                gravity,
                float(mu_values[0]),
                pose_catalog.contact_tolerance_mm,
                length_scale,
                margin_tolerance,
            )
            if not first_sample.stable:
                continue
            accepted_samples = [first_sample]
            for mu in mu_values[1:]:
                sample = _solve_pose_sample(
                    pose,
                    vertices_centered,
                    gravity,
                    float(mu),
                    pose_catalog.contact_tolerance_mm,
                    length_scale,
                    margin_tolerance,
                )
                if not sample.stable:
                    break
                accepted_samples.append(sample)
            if len(accepted_samples) != len(mu_values):
                continue
            samples = tuple(accepted_samples)
        else:
            samples = tuple(
                _solve_pose_sample(
                    pose,
                    vertices_centered,
                    gravity,
                    float(mu),
                    pose_catalog.contact_tolerance_mm,
                    length_scale,
                    margin_tolerance,
                )
                for mu in mu_values
            )
        results.append(
            PoseStability(
                pose_id=pose.pose_id,
                floor_contact_type=pose.floor_contact_type,
                wall_contact_type=pose.wall_contact_type,
                samples=samples,
                stable_across_range=all(sample.stable for sample in samples),
                stable_at_any_sample=any(sample.stable for sample in samples),
                minimum_pressure_margin=min(sample.pressure_margin for sample in samples),
            )
        )

    return StabilityAnalysis(
        source=str(Path(mesh_path).expanduser().resolve()),
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        gravity_chute_m_s2=tuple(float(value) for value in gravity),
        friction_estimate=estimate,
        mu_values=tuple(float(value) for value in mu_values),
        poses=tuple(results),
        input_pose_count=len(pose_catalog.poses),
        quasistatic_only=quasistatic_only,
        friction_policy=friction_policy,
    )
