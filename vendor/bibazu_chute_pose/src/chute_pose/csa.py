"""Contact-wrench solid-angle ranking for two-surface chute poses.

This module deliberately does not reproduce the legacy implementation which
projected perpendicular floor and wall contacts into one fictitious polygon.
Instead it integrates the existing force/moment contact-load margin over an
equal-area spherical cap of nearby gravity directions.  The resulting CWSA
index is dimensionless, independent of the number of candidate poses, and
keeps floor and wall forces mechanically separate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError

from .contacts import ContactPose, PoseCatalog, build_pose_catalog
from .frame import ChuteFrame
from .geometry import load_solid_mesh
from .stability import (
    FrictionPolicy,
    StabilitySample,
    _contact_boundary_indices,
    _solve_pose_sample,
    estimate_equal_contact_friction,
)


CSA_ALGORITHM_LABEL = "contact-wrench solid-angle (CWSA)"
DEFAULT_CSA_CAP_HALF_ANGLE_DEG = 5.0
DEFAULT_CSA_DIRECTION_SAMPLES = 72
DEFAULT_CSA_ROBUST_THRESHOLD = 0.65


@dataclass(frozen=True, slots=True)
class ContactWrenchSolidAnglePose:
    """CWSA result for one fixed-contact pose."""

    pose_id: int
    csa_stability_index: float
    feasible_fraction: float
    feasible_solid_angle_sr: float
    cap_solid_angle_sr: float
    nominal_contact_load_balance_index: float
    mean_feasible_contact_load_balance_index: float
    angular_clearance_deg: float
    floor_patch_solid_angle_sr: float
    wall_patch_solid_angle_sr: float
    applicable: bool
    reason: str

    @property
    def score(self) -> float:
        """Compatibility shorthand for the exported CWSA stability index."""

        return self.csa_stability_index

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContactWrenchSolidAngleAnalysis:
    """CWSA results for one part and one chute operating point."""

    source: str
    alpha_deg: float
    beta_deg: float
    gravity_chute_m_s2: tuple[float, float, float]
    mu_values: tuple[float, ...]
    cap_half_angle_deg: float
    cap_solid_angle_sr: float
    direction_samples: int
    poses: tuple[ContactWrenchSolidAnglePose, ...]

    def pose(self, pose_id: int) -> ContactWrenchSolidAnglePose:
        for value in self.poses:
            if value.pose_id == pose_id:
                return value
        raise KeyError(pose_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": CSA_ALGORITHM_LABEL,
            "requires_positive_x_acceleration": False,
            "load_model": (
                "frictionless_support" if all(mu == 0.0 for mu in self.mu_values)
                else "prescribed_sliding_friction"
            ),
            "models_longitudinal_restraint": False,
            "source": self.source,
            "alpha_deg": self.alpha_deg,
            "beta_deg": self.beta_deg,
            "gravity_chute_m_s2": self.gravity_chute_m_s2,
            "mu_values": self.mu_values,
            "cap_half_angle_deg": self.cap_half_angle_deg,
            "cap_solid_angle_sr": self.cap_solid_angle_sr,
            "direction_samples": self.direction_samples,
            "poses": [value.to_dict() for value in self.poses],
        }


@dataclass(frozen=True, slots=True)
class ContactWrenchSolidAngleFilter:
    """Threshold classification of fixed-contact CWSA results."""

    minimum_score: float
    accepted_pose_ids: tuple[int, ...]
    rejected_pose_ids: tuple[int, ...]
    not_applicable_pose_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _equal_area_cap_directions(
    axis: NDArray[np.float64],
    half_angle_rad: float,
    sample_count: int,
) -> tuple[NDArray[np.float64], ...]:
    """Return deterministic equal-area Fibonacci samples on a spherical cap."""

    unit_axis = np.asarray(axis, dtype=float)
    unit_axis /= np.linalg.norm(unit_axis)
    reference = np.eye(3)[int(np.argmin(np.abs(unit_axis)))]
    first = reference - unit_axis * float(np.dot(reference, unit_axis))
    first /= np.linalg.norm(first)
    second = np.cross(unit_axis, first)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    cap_depth = 1.0 - math.cos(half_angle_rad)
    result: list[NDArray[np.float64]] = []
    for index in range(sample_count):
        fraction = (index + 0.5) / sample_count
        cosine = 1.0 - cap_depth * fraction
        sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
        azimuth = index * golden_angle
        direction = (
            cosine * unit_axis
            + sine
            * (
                math.cos(azimuth) * first
                + math.sin(azimuth) * second
            )
        )
        direction /= np.linalg.norm(direction)
        result.append(direction)
    return tuple(result)


def _triangle_solid_angle(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    third: NDArray[np.float64],
) -> float:
    lengths = tuple(float(np.linalg.norm(value)) for value in (first, second, third))
    if min(lengths) <= np.finfo(float).eps:
        return 0.0
    numerator = abs(float(np.linalg.det(np.stack((first, second, third)))))
    denominator = (
        lengths[0] * lengths[1] * lengths[2]
        + float(np.dot(first, second)) * lengths[2]
        + float(np.dot(second, third)) * lengths[0]
        + float(np.dot(third, first)) * lengths[1]
    )
    return 2.0 * math.atan2(numerator, denominator)


def _planar_patch_solid_angle(
    points: NDArray[np.float64],
    projection_axes: tuple[int, int],
    tolerance_mm: float,
) -> float:
    """Return one real planar contact patch's solid angle at the centre of mass."""

    if len(points) < 3:
        return 0.0
    projected = points[:, projection_axes]
    scale = max(tolerance_mm, np.finfo(float).eps)
    keys = np.round(projected / scale).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    unique_points = points[np.sort(first)]
    unique_projected = unique_points[:, projection_axes]
    if len(unique_points) < 3:
        return 0.0
    try:
        boundary = ConvexHull(unique_projected).vertices
    except QhullError:
        return 0.0
    polygon = unique_points[boundary]
    return float(
        sum(
            _triangle_solid_angle(polygon[0], polygon[index], polygon[index + 1])
            for index in range(1, len(polygon) - 1)
        )
    )


def _patch_diagnostics(
    pose: ContactPose,
    vertices_centered: NDArray[np.float64],
    contact_tolerance_mm: float,
) -> tuple[float, float]:
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
    return (
        _planar_patch_solid_angle(
            points[floor_indices], (0, 1), contact_tolerance_mm
        ),
        _planar_patch_solid_angle(
            points[wall_indices], (0, 2), contact_tolerance_mm
        ),
    )


def _has_contact_reserve(sample: StabilitySample, margin_tolerance: float) -> bool:
    """Check support against tipping independently of longitudinal motion."""

    return sample.equilibrium_feasible and sample.pressure_margin > margin_tolerance


def analyze_contact_wrench_solid_angle(
    mesh_path: str | Path,
    *,
    pose_ids: Iterable[int] | None = None,
    alpha_deg: float = 45.0,
    beta_deg: float = 20.0,
    onset_alpha_deg: float = 45.0,
    onset_beta_deg: float = 15.0,
    mu_samples: int = 11,
    friction_policy: FrictionPolicy = "zero",
    cap_half_angle_deg: float = DEFAULT_CSA_CAP_HALF_ANGLE_DEG,
    direction_samples: int = DEFAULT_CSA_DIRECTION_SAMPLES,
    catalog: PoseCatalog | None = None,
) -> ContactWrenchSolidAngleAnalysis:
    """Integrate contact-load reserve over nearby gravity directions.

    The equal-area cap samples represent solid-angle quadrature points.  At
    each point the score contribution is the minimum contact-load balance
    margin over the requested friction samples, or zero when any friction sample
    loses contact equilibrium or has no positive load margin. The forward
    sliding/acceleration flag is deliberately excluded: deceleration or lack
    of forward motion does not by itself mean tipping. The underlying samples
    default to frictionless support (mu=0), with no sliding load. Explicit
    friction_policy="range" restores the prescribed sliding-friction sweep.
    Neither mode models static friction holding the part along X: longitudinal
    translation is unconstrained, so this is a tipping-support assessment,
    not proof of complete stationary equilibrium on a sloping chute.
    The average is normalized by the cap solid angle
    and lies in ``0..1``.
    """

    if not math.isfinite(cap_half_angle_deg) or not 0.0 < cap_half_angle_deg < 90.0:
        raise ValueError("cap_half_angle_deg must be between 0 and 90 degrees.")
    if direction_samples < 8:
        raise ValueError("direction_samples must be at least 8.")
    if friction_policy not in {"zero", "range"}:
        raise ValueError("friction_policy must be 'zero' or 'range'.")
    if friction_policy == "range" and mu_samples < 2:
        raise ValueError("mu_samples must be at least 2.")

    pose_catalog = catalog or build_pose_catalog(mesh_path)
    poses_by_id = {value.pose_id: value for value in pose_catalog.poses}
    requested_ids = (
        tuple(poses_by_id)
        if pose_ids is None
        else tuple(dict.fromkeys(int(value) for value in pose_ids))
    )
    missing = tuple(value for value in requested_ids if value not in poses_by_id)
    if missing:
        raise KeyError(f"Unknown pose ids: {missing}")

    frame = ChuteFrame(alpha_deg=alpha_deg, beta_deg=beta_deg)
    nominal_gravity = frame.gravity_chute()
    gravity_magnitude = float(np.linalg.norm(nominal_gravity))
    nominal_direction = nominal_gravity / gravity_magnitude
    half_angle_rad = math.radians(cap_half_angle_deg)
    cap_solid_angle = 2.0 * math.pi * (1.0 - math.cos(half_angle_rad))
    directions = _equal_area_cap_directions(
        nominal_direction, half_angle_rad, direction_samples
    )
    mu_values = np.asarray([0.0])
    if friction_policy == "range":
        estimate = estimate_equal_contact_friction(
            onset_alpha_deg=onset_alpha_deg,
            onset_beta_deg=onset_beta_deg,
        )
        mu_values = np.linspace(0.0, estimate.mu_static_estimate, mu_samples)

    mesh = load_solid_mesh(mesh_path)
    hull = mesh.convex_hull
    vertices_centered = np.asarray(hull.vertices, dtype=float) - np.asarray(
        mesh.center_mass, dtype=float
    )
    length_scale = max(float(np.max(hull.extents)), 1e-9)
    margin_tolerance = 1e-6
    results: list[ContactWrenchSolidAnglePose] = []
    for pose_id in requested_ids:
        pose = poses_by_id[pose_id]
        nominal_samples = tuple(
            _solve_pose_sample(
                pose,
                vertices_centered,
                nominal_gravity,
                float(mu),
                pose_catalog.contact_tolerance_mm,
                length_scale,
                margin_tolerance,
            )
            for mu in mu_values
        )
        nominal_margin = (
            min(value.pressure_margin for value in nominal_samples)
            if all(_has_contact_reserve(value, margin_tolerance) for value in nominal_samples)
            else 0.0
        )
        margins: list[float] = []
        feasible: list[bool] = []
        failure_angles: list[float] = []
        for direction in directions:
            gravity = gravity_magnitude * direction
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
            accepted = all(_has_contact_reserve(value, margin_tolerance) for value in samples)
            feasible.append(accepted)
            margins.append(
                min(value.pressure_margin for value in samples) if accepted else 0.0
            )
            if not accepted:
                separation = math.degrees(
                    math.acos(
                        float(np.clip(np.dot(direction, nominal_direction), -1.0, 1.0))
                    )
                )
                failure_angles.append(separation)

        feasible_count = sum(feasible)
        feasible_fraction = feasible_count / direction_samples
        feasible_margins = [
            margin for margin, accepted in zip(margins, feasible) if accepted
        ]
        floor_patch, wall_patch = _patch_diagnostics(
            pose, vertices_centered, pose_catalog.contact_tolerance_mm
        )
        rolling_contact = (
            pose_catalog.continuous_symmetry_axis_part is not None
            and pose.floor_contact_dimension == 1
            and pose.wall_contact_dimension == 1
        )
        results.append(
            ContactWrenchSolidAnglePose(
                pose_id=pose_id,
                csa_stability_index=float(np.clip(np.mean(margins), 0.0, 1.0)),
                feasible_fraction=float(feasible_fraction),
                feasible_solid_angle_sr=float(cap_solid_angle * feasible_fraction),
                cap_solid_angle_sr=float(cap_solid_angle),
                nominal_contact_load_balance_index=float(nominal_margin),
                mean_feasible_contact_load_balance_index=(
                    float(np.mean(feasible_margins)) if feasible_margins else 0.0
                ),
                angular_clearance_deg=(
                    min(failure_angles) if failure_angles else cap_half_angle_deg
                ),
                floor_patch_solid_angle_sr=float(floor_patch),
                wall_patch_solid_angle_sr=float(wall_patch),
                applicable=not rolling_contact,
                reason=(
                    "continuous_rolling_contact_switching_not_modelled"
                    if rolling_contact
                    else "ok"
                ),
            )
        )

    return ContactWrenchSolidAngleAnalysis(
        source=str(Path(mesh_path).expanduser().resolve()),
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        gravity_chute_m_s2=tuple(float(value) for value in nominal_gravity),
        mu_values=tuple(float(value) for value in mu_values),
        cap_half_angle_deg=float(cap_half_angle_deg),
        cap_solid_angle_sr=float(cap_solid_angle),
        direction_samples=direction_samples,
        poses=tuple(results),
    )


def filter_contact_wrench_solid_angle(
    analysis: ContactWrenchSolidAngleAnalysis,
    *,
    minimum_score: float = DEFAULT_CSA_ROBUST_THRESHOLD,
) -> ContactWrenchSolidAngleFilter:
    """Classify applicable fixed-contact poses at a configurable CWSA cutoff."""

    if not math.isfinite(minimum_score) or not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must lie between 0 and 1.")
    accepted = tuple(
        value.pose_id
        for value in analysis.poses
        if value.applicable and value.score >= minimum_score
    )
    not_applicable = tuple(
        value.pose_id for value in analysis.poses if not value.applicable
    )
    rejected = tuple(
        value.pose_id
        for value in analysis.poses
        if value.applicable and value.score < minimum_score
    )
    return ContactWrenchSolidAngleFilter(
        minimum_score=float(minimum_score),
        accepted_pose_ids=accepted,
        rejected_pose_ids=rejected,
        not_applicable_pose_ids=not_applicable,
    )
