"""Finite rocking-energy barriers for seated chute poses.

The contact-force LP in :mod:`chute_pose.disturbance` stops when the first
contact point unloads.  That is deliberately conservative, but unloading is
not the same as overturning for a pose which rocks into another contact.  This
module measures the finite centre-of-mass lift needed to leave such a pose.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contacts import ContactPose, PoseCatalog, build_pose_catalog
from .disturbance import DisturbanceAnalysis
from .frame import ChuteFrame
from .geometry import load_solid_mesh


@dataclass(frozen=True, slots=True)
class RockingBarrier:
    """Weakest sampled straight rocking path for one contact pose."""

    pose_id: int
    excursion_deg: float
    barrier_height_mm: float
    barrier_normalized: float
    critical_axis_chute: tuple[float, float, float]
    peak_angle_deg: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RockingAnalysis:
    source: str
    alpha_deg: float
    beta_deg: float
    excursion_deg: float
    angle_steps: int
    axis_samples: int
    length_scale_mm: float
    barriers: tuple[RockingBarrier, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "alpha_deg": self.alpha_deg,
            "beta_deg": self.beta_deg,
            "excursion_deg": self.excursion_deg,
            "angle_steps": self.angle_steps,
            "axis_samples": self.axis_samples,
            "length_scale_mm": self.length_scale_mm,
            "barriers": [barrier.to_dict() for barrier in self.barriers],
        }


@dataclass(frozen=True, slots=True)
class FiniteDisturbanceFilterResult:
    """Combined finite-rocking and braking-reserve decision."""

    minimum_barrier_height_mm: float
    minimum_braking_g: float
    accepted_pose_ids: tuple[int, ...]
    rejected_pose_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fibonacci_axes(count: int) -> np.ndarray:
    """Return deterministic, approximately uniform signed axes on the sphere."""

    indices = np.arange(count, dtype=float)
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    azimuth = math.pi * (3.0 - math.sqrt(5.0)) * indices
    axes = np.column_stack(
        (radius * np.cos(azimuth), radius * np.sin(azimuth), z)
    )
    # Exact cardinal axes make small test grids deterministic and ensure that
    # the most common longitudinal/transverse rocking directions are present.
    return np.vstack((axes, np.eye(3), -np.eye(3)))


def _axes_tangent_to_symmetry_direction(
    symmetry_axis_chute: np.ndarray, count: int
) -> np.ndarray:
    """Sample quotient-space rocking axes for an SO(2)-symmetric part."""

    axis = np.asarray(symmetry_axis_chute, dtype=float)
    axis /= np.linalg.norm(axis)
    first = np.eye(3)[int(np.argmin(np.abs(axis)))]
    first -= axis * float(np.dot(axis, first))
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    return np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second


def _seated_potential_per_mass(
    points_by_axis: np.ndarray, gravity: np.ndarray
) -> np.ndarray:
    """Potential per mass after translating each orientation into y=z=0."""

    minimum_y = np.min(points_by_axis[..., 1], axis=1)
    minimum_z = np.min(points_by_axis[..., 2], axis=1)
    return gravity[1] * minimum_y + gravity[2] * minimum_z


def _rotate_about_axes(
    points: np.ndarray, axes: np.ndarray, angle_rad: float
) -> np.ndarray:
    """Vectorised Rodrigues rotation of points about all axes through the COM."""

    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    axis_dot_point = axes @ points.T
    cross = np.cross(axes[:, None, :], points[None, :, :])
    return (
        cosine * points[None, :, :]
        + sine * cross
        + (1.0 - cosine) * axes[:, None, :] * axis_dot_point[:, :, None]
    )


def _pose_rocking_barrier(
    pose: ContactPose,
    vertices_centered: np.ndarray,
    gravity: np.ndarray,
    gravity_magnitude: float,
    axes: np.ndarray,
    excursion_deg: float,
    angle_steps: int,
    length_scale_mm: float,
    continuous_symmetry_axis_part: np.ndarray | None = None,
) -> RockingBarrier:
    rotation = np.asarray(pose.rotation_chute_from_part, dtype=float)
    if continuous_symmetry_axis_part is not None:
        axes = _axes_tangent_to_symmetry_direction(
            rotation @ continuous_symmetry_axis_part,
            len(axes),
        )
    points = (rotation @ vertices_centered.T).T
    baseline = float(_seated_potential_per_mass(points[None, :, :], gravity)[0])
    path_peak = np.full(len(axes), baseline, dtype=float)
    peak_step = np.zeros(len(axes), dtype=int)

    for step in range(1, angle_steps + 1):
        angle_rad = math.radians(excursion_deg * step / angle_steps)
        rotated = _rotate_about_axes(points, axes, angle_rad)
        potential = _seated_potential_per_mass(rotated, gravity)
        higher = potential > path_peak
        path_peak[higher] = potential[higher]
        peak_step[higher] = step

    critical_index = int(np.argmin(path_peak))
    barrier_per_mass = max(0.0, float(path_peak[critical_index] - baseline))
    # U/m = g*h, with coordinates in millimetres, so h remains in mm.
    barrier_height = barrier_per_mass / gravity_magnitude
    return RockingBarrier(
        pose_id=pose.pose_id,
        excursion_deg=excursion_deg,
        barrier_height_mm=barrier_height,
        barrier_normalized=barrier_height / length_scale_mm,
        critical_axis_chute=tuple(float(value) for value in axes[critical_index]),
        peak_angle_deg=(
            excursion_deg * int(peak_step[critical_index]) / angle_steps
        ),
    )


def analyze_rocking_barriers(
    mesh_path: str | Path,
    *,
    pose_ids: Iterable[int] | None = None,
    alpha_deg: float = 45.0,
    beta_deg: float = 20.0,
    excursion_deg: float = 5.0,
    angle_steps: int = 20,
    axis_samples: int = 2048,
    catalog: PoseCatalog | None = None,
) -> RockingAnalysis:
    """Measure the weakest finite rocking barrier of selected seated poses.

    Each candidate is rotated about uniformly sampled signed chute-frame axes.
    At every intermediate angle it is translated back into the floor/wall
    corner.  The result is the smallest peak centre-of-mass lift among these
    straight rotational paths; it is a deterministic geometric robustness
    measure, not a fall simulation.
    """

    if not math.isfinite(excursion_deg) or excursion_deg <= 0.0:
        raise ValueError("excursion_deg must be finite and positive.")
    if angle_steps < 1:
        raise ValueError("angle_steps must be at least 1.")
    if axis_samples < 8:
        raise ValueError("axis_samples must be at least 8.")

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
    continuous_symmetry = pose_catalog.continuous_symmetry_axis_part is not None
    effective_excursion_deg = 90.0 if continuous_symmetry else excursion_deg
    effective_angle_steps = max(angle_steps, 180) if continuous_symmetry else angle_steps
    effective_axis_samples = min(axis_samples, 720) if continuous_symmetry else axis_samples
    gravity = ChuteFrame(alpha_deg=alpha_deg, beta_deg=beta_deg).gravity_chute()
    gravity_magnitude = float(np.linalg.norm(gravity))
    axes = _fibonacci_axes(effective_axis_samples)
    barriers = tuple(
        _pose_rocking_barrier(
            pose,
            vertices_centered,
            gravity,
            gravity_magnitude,
            axes,
            effective_excursion_deg,
            effective_angle_steps,
            length_scale,
            (
                None
                if pose_catalog.continuous_symmetry_axis_part is None
                else np.asarray(
                    pose_catalog.continuous_symmetry_axis_part, dtype=float
                )
            ),
        )
        for pose in selected
    )
    return RockingAnalysis(
        source=str(Path(mesh_path).expanduser().resolve()),
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        excursion_deg=effective_excursion_deg,
        angle_steps=effective_angle_steps,
        axis_samples=effective_axis_samples,
        length_scale_mm=length_scale,
        barriers=barriers,
    )


def filter_finite_disturbance_robustness(
    rocking: RockingAnalysis,
    disturbance: DisturbanceAnalysis,
    catalog: PoseCatalog,
    *,
    minimum_barrier_height_mm: float = 0.73,
    minimum_braking_g: float = 0.0,
) -> FiniteDisturbanceFilterResult:
    """Require a rocking barrier, with an optional braking-reserve gate.

    By default only the gravitational barrier is used (no braking load).
    When a positive braking threshold is requested, an edge contact
    can still unload immediately under braking. Applying the same braking limit
    to face-face and edge-containing poses avoids classifying a shallow edge
    landing as robust merely because its geometric rocking path rises briefly.
    """

    for name, value in (
        ("minimum_barrier_height_mm", minimum_barrier_height_mm),
        ("minimum_braking_g", minimum_braking_g),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")

    barriers = {value.pose_id: value for value in rocking.barriers}
    capacities = {value.pose_id: value for value in disturbance.capacities}
    poses = {value.pose_id: value for value in catalog.poses}
    pose_ids = tuple(barriers)
    if set(pose_ids) != set(capacities):
        raise ValueError("Rocking and disturbance analyses must cover the same poses.")
    missing = set(pose_ids) - set(poses)
    if missing:
        raise ValueError(f"Catalog is missing analyzed pose ids: {sorted(missing)}")

    accepted: list[int] = []
    for pose_id in pose_ids:
        has_barrier = (
            barriers[pose_id].barrier_height_mm >= minimum_barrier_height_mm
        )
        capacity = capacities[pose_id]
        has_braking_reserve = (
            minimum_braking_g == 0.0
            or capacity.critical_braking_g >= minimum_braking_g
        )
        if has_barrier and has_braking_reserve:
            accepted.append(pose_id)
    accepted_set = set(accepted)
    return FiniteDisturbanceFilterResult(
        minimum_barrier_height_mm=minimum_barrier_height_mm,
        minimum_braking_g=minimum_braking_g,
        accepted_pose_ids=tuple(accepted),
        rejected_pose_ids=tuple(
            pose_id for pose_id in pose_ids if pose_id not in accepted_set
        ),
    )
