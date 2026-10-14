"""Practical clustering of nearly identical contact-pose orientations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.spatial.transform import Rotation

from .contacts import PoseCatalog
from .symmetry import RotationalSymmetryGroup


@dataclass(frozen=True, slots=True)
class PracticalPoseClass:
    class_id: int
    representative_pose_id: int
    pose_ids: tuple[int, ...]
    floor_contact_type: str
    wall_contact_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PracticalPoseClustering:
    angular_tolerance_deg: float
    surface_displacement_tolerance_mm: float
    classes: tuple[PracticalPoseClass, ...]

    def class_for_pose(self, pose_id: int) -> PracticalPoseClass:
        for pose_class in self.classes:
            if pose_id in pose_class.pose_ids:
                return pose_class
        raise KeyError(pose_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "angular_tolerance_deg": self.angular_tolerance_deg,
            "surface_displacement_tolerance_mm": (
                self.surface_displacement_tolerance_mm
            ),
            "classes": [pose_class.to_dict() for pose_class in self.classes],
        }


def _pair_distance(
    first: np.ndarray,
    second: np.ndarray,
    vertices_centered: np.ndarray,
    symmetry_rotations: tuple[np.ndarray, ...],
    angular_tolerance_rad: float,
    displacement_tolerance_mm: float,
    continuous_axis_part: np.ndarray | None = None,
) -> float:
    if continuous_axis_part is not None:
        first_axis = first @ continuous_axis_part
        second_axis = second @ continuous_axis_part
        angular = math.acos(
            float(np.clip(np.dot(first_axis, second_axis), -1.0, 1.0))
        )
        radius = float(np.max(np.linalg.norm(vertices_centered, axis=1)))
        displacement = 2.0 * radius * math.sin(0.5 * angular)
        return max(
            angular / angular_tolerance_rad,
            displacement / displacement_tolerance_mm,
        )
    best = math.inf
    for symmetry in symmetry_rotations:
        equivalent_first = first @ symmetry
        angular = float(
            Rotation.from_matrix(equivalent_first.T @ second).magnitude()
        )
        first_vertices = (equivalent_first @ vertices_centered.T).T
        second_vertices = (second @ vertices_centered.T).T
        displacement = float(
            np.max(np.linalg.norm(first_vertices - second_vertices, axis=1))
        )
        distance = max(
            angular / angular_tolerance_rad,
            displacement / displacement_tolerance_mm,
        )
        best = min(best, distance)
    return best


def cluster_practical_contact_poses(
    catalog: PoseCatalog,
    vertices_centered: np.ndarray,
    pose_ids: Iterable[int],
    *,
    symmetry: RotationalSymmetryGroup | None = None,
    angular_tolerance_deg: float = 0.25,
    surface_displacement_tolerance_mm: float = 0.5,
) -> PracticalPoseClustering:
    """Complete-link cluster near-identical occupied contact orientations.

    This is not part symmetry. It merges numerical/faceted variants only when
    every pair in a class stays within both an angular and a maximum occupied-
    surface displacement tolerance.
    """

    if not math.isfinite(angular_tolerance_deg) or angular_tolerance_deg <= 0.0:
        raise ValueError("angular_tolerance_deg must be positive and finite.")
    if (
        not math.isfinite(surface_displacement_tolerance_mm)
        or surface_displacement_tolerance_mm <= 0.0
    ):
        raise ValueError(
            "surface_displacement_tolerance_mm must be positive and finite."
        )
    requested = sorted({int(pose_id) for pose_id in pose_ids})
    poses_by_id = {pose.pose_id: pose for pose in catalog.poses}
    missing = set(requested) - poses_by_id.keys()
    if missing:
        raise ValueError(f"Unknown pose ids: {sorted(missing)}")
    vertices = np.asarray(vertices_centered, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices_centered must have shape (n, 3).")
    symmetry_rotations = (
        tuple(
            np.asarray(element.rotation_part_from_part, dtype=float)
            for element in symmetry.elements
        )
        if symmetry is not None
        else (np.eye(3),)
    )
    continuous_axis = (
        np.asarray(symmetry.continuous_axis_part, dtype=float)
        if symmetry is not None and symmetry.continuous_axis_part is not None
        else None
    )
    angular_tolerance = math.radians(angular_tolerance_deg)
    grouped_by_contact: dict[tuple[int, int], list[int]] = {}
    for pose_id in requested:
        pose = poses_by_id[pose_id]
        # Fine facets near the floor-wall bisector can describe the same
        # occupied orientation with the contact dimensions exchanged between
        # the two planes.  Keep the dimensional signature, but make it
        # independent of which plane received which member.  The subsequent
        # angular and whole-surface displacement checks still have to pass for
        # every pair, so genuinely different orientations remain separate.
        grouped_by_contact.setdefault(
            tuple(
                sorted(
                    (pose.floor_contact_dimension, pose.wall_contact_dimension)
                )
            ),
            [],
        ).append(pose_id)

    raw_classes: list[tuple[int, ...]] = []
    for group_ids in grouped_by_contact.values():
        if len(group_ids) == 1:
            raw_classes.append((group_ids[0],))
            continue
        rotations = [
            np.asarray(poses_by_id[pose_id].rotation_chute_from_part, dtype=float)
            for pose_id in group_ids
        ]
        distances = np.zeros((len(group_ids), len(group_ids)), dtype=float)
        for first in range(len(group_ids)):
            for second in range(first + 1, len(group_ids)):
                distance = _pair_distance(
                    rotations[first],
                    rotations[second],
                    vertices,
                    symmetry_rotations,
                    angular_tolerance,
                    surface_displacement_tolerance_mm,
                    continuous_axis,
                )
                distances[first, second] = distance
                distances[second, first] = distance
        hierarchy = linkage(squareform(distances), method="complete")
        labels = fcluster(hierarchy, t=1.0, criterion="distance")
        local_classes: dict[int, list[int]] = {}
        for pose_id, label in zip(group_ids, labels):
            local_classes.setdefault(int(label), []).append(pose_id)
        raw_classes.extend(tuple(sorted(values)) for values in local_classes.values())

    raw_classes.sort()
    classes = tuple(
        PracticalPoseClass(
            class_id=index,
            representative_pose_id=pose_ids_in_class[0],
            pose_ids=pose_ids_in_class,
            floor_contact_type=poses_by_id[
                pose_ids_in_class[0]
            ].floor_contact_type,
            wall_contact_type=poses_by_id[pose_ids_in_class[0]].wall_contact_type,
        )
        for index, pose_ids_in_class in enumerate(raw_classes)
    )
    return PracticalPoseClustering(
        angular_tolerance_deg=angular_tolerance_deg,
        surface_displacement_tolerance_mm=surface_displacement_tolerance_mm,
        classes=classes,
    )
