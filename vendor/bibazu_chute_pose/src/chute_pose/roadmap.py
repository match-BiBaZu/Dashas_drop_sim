"""Pose-class roadmap, actuator adjacency and open-loop route planning."""

from __future__ import annotations

import base64
import heapq
import json
import math
from dataclasses import asdict, dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.text import Annotation
from scipy.spatial.transform import Rotation

from .contacts import ContactPose, build_pose_catalog
from .classical import METHODS as CLASSICAL_METHODS, MODEL_VERSION, analyze_chute_classical
from .csa import (
    DEFAULT_CSA_CAP_HALF_ANGLE_DEG,
    DEFAULT_CSA_DIRECTION_SAMPLES,
    DEFAULT_CSA_ROBUST_THRESHOLD,
    analyze_contact_wrench_solid_angle,
    filter_contact_wrench_solid_angle,
)
from .disturbance import analyze_disturbance_robustness
from .equivalence import PracticalPoseClass, cluster_practical_contact_poses
from .frame import ChuteFrame
from .geometry import load_solid_mesh
from .rocking import (
    analyze_rocking_barriers,
    filter_finite_disturbance_robustness,
)
from .stability import FrictionPolicy, analyze_pose_stability
from .symmetry import detect_rotational_symmetry
from .visualization import create_pose_thumbnails
from .metrics import metric_lines

NodeKind = Literal["robust", "metastable"]
PoseRankingMethod = Literal["rocking", "csa", "cwsa", "standard_csa", "crsa"]
RobustnessMethod = PoseRankingMethod
TransitionKind = Literal["actuated", "passive_tip"]
ActuationKind = Literal[
    "floor_main_neg_x",
    "floor_main_pos_x",
    "wall_main_neg_x",
    "wall_main_pos_x",
    "free_y",
    "free_z",
    "passive",
]

DEFAULT_ROADMAP_SYMMETRY_TOLERANCE_MM = 0.05


@dataclass(frozen=True, slots=True)
class RoadmapNode:
    node_id: int
    pose_ids: tuple[int, ...]
    kind: NodeKind
    cad_status: str
    representative_quaternion_xyzw: tuple[float, float, float, float]
    floor_contact_topology: str
    wall_contact_topology: str
    rocking_barrier_mm: float
    main_face_on_floor: bool
    main_face_on_wall: bool
    csa_stability_index: float | None = None
    csa_feasible_fraction: float | None = None
    csa_feasible_solid_angle_sr: float | None = None
    csa_angular_clearance_deg: float | None = None
    csa_applicable: bool | None = None
    classical_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def original_catalog_pose_id(self) -> int:
        """Return the original representative ID before roadmap relabelling."""

        return self.pose_ids[0]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["original_catalog_pose_id"] = self.original_catalog_pose_id
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RoadmapNode:
        return cls(
            node_id=int(value["node_id"]),
            classical_metrics=value.get("classical_metrics", {}),
            pose_ids=tuple(int(item) for item in value["pose_ids"]),
            kind=value["kind"],
            cad_status=str(value.get("cad_status", "provisional")),
            representative_quaternion_xyzw=tuple(
                float(item) for item in value["representative_quaternion_xyzw"]
            ),  # type: ignore[arg-type]
            floor_contact_topology=str(value["floor_contact_topology"]),
            wall_contact_topology=str(value["wall_contact_topology"]),
            rocking_barrier_mm=float(value["rocking_barrier_mm"]),
            main_face_on_floor=bool(value["main_face_on_floor"]),
            main_face_on_wall=bool(value["main_face_on_wall"]),
            csa_stability_index=(
                float(value["csa_stability_index"])
                if value.get("csa_stability_index") is not None
                else None
            ),
            csa_feasible_fraction=(
                float(value["csa_feasible_fraction"])
                if value.get("csa_feasible_fraction") is not None
                else None
            ),
            csa_feasible_solid_angle_sr=(
                float(value["csa_feasible_solid_angle_sr"])
                if value.get("csa_feasible_solid_angle_sr") is not None
                else None
            ),
            csa_angular_clearance_deg=(
                float(value["csa_angular_clearance_deg"])
                if value.get("csa_angular_clearance_deg") is not None
                else None
            ),
            csa_applicable=(
                bool(value["csa_applicable"])
                if value.get("csa_applicable") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class RoadmapEdge:
    edge_id: str
    source: int
    target: int
    transition_kind: TransitionKind
    actuation: ActuationKind
    axis_chute: tuple[float, float, float]
    signed_angle_deg: float
    capture_interval_deg: tuple[float, float] | None
    capture_width_deg: float
    capture_fraction: float
    target_barrier_score: float
    geometric_score: float
    escape_barrier_mm: float | None = None
    saddle_angle_deg: float | None = None
    axis_error_deg: float = 0.0
    settling_pose_ids: tuple[int, ...] = ()

    @property
    def actuation_count(self) -> int:
        return 1 if self.transition_kind == "actuated" else 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["actuation_count"] = self.actuation_count
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RoadmapEdge:
        interval = value.get("capture_interval_deg")
        return cls(
            edge_id=str(value["edge_id"]),
            source=int(value["source"]),
            target=int(value["target"]),
            transition_kind=value["transition_kind"],
            actuation=value["actuation"],
            axis_chute=tuple(float(item) for item in value["axis_chute"]),  # type: ignore[arg-type]
            signed_angle_deg=float(value["signed_angle_deg"]),
            capture_interval_deg=(
                tuple(float(item) for item in interval) if interval is not None else None
            ),  # type: ignore[arg-type]
            capture_width_deg=float(value["capture_width_deg"]),
            capture_fraction=float(value["capture_fraction"]),
            target_barrier_score=float(value["target_barrier_score"]),
            geometric_score=float(value["geometric_score"]),
            escape_barrier_mm=(
                float(value["escape_barrier_mm"])
                if value.get("escape_barrier_mm") is not None
                else None
            ),
            saddle_angle_deg=(
                float(value["saddle_angle_deg"])
                if value.get("saddle_angle_deg") is not None
                else None
            ),
            axis_error_deg=float(value.get("axis_error_deg", 0.0)),
            settling_pose_ids=tuple(
                int(item) for item in value.get("settling_pose_ids", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class PoseRoadmap:
    schema_version: int
    source: str
    geometry_status: str
    alpha_deg: float
    beta_deg: float
    symmetry_symbol: str
    symmetry_tolerance_mm: float
    main_face_id: int
    main_face_ids: tuple[int, ...]
    main_face_area_mm2: float
    main_face_min_span_mm: float
    opposite_x_min_height_mm: float
    robust_barrier_threshold_mm: float
    axis_tolerance_deg: float
    nodes: tuple[RoadmapNode, ...]
    edges: tuple[RoadmapEdge, ...]
    unresolved_metastable_node_ids: tuple[int, ...]
    pose_ranking_method: PoseRankingMethod = "rocking"
    robustness_method: RobustnessMethod = "rocking"
    minimum_csa_score: float = DEFAULT_CSA_ROBUST_THRESHOLD
    csa_cap_half_angle_deg: float = DEFAULT_CSA_CAP_HALF_ANGLE_DEG
    csa_direction_samples: int = DEFAULT_CSA_DIRECTION_SAMPLES
    csa_rocking_fallback_pose_ids: tuple[int, ...] = ()
    friction_policy: FrictionPolicy = "zero"
    minimum_braking_g: float | None = None
    csa_load_model: str | None = None
    classical_metadata: dict[str, Any] = field(default_factory=dict)

    def node(self, node_or_pose_id: int) -> RoadmapNode:
        # Roadmap display IDs take precedence because the compact 0..N-1
        # namespace can overlap an original catalog member ID.
        for node in self.nodes:
            if node.node_id == node_or_pose_id:
                return node
        for node in self.nodes:
            if node_or_pose_id in node.pose_ids:
                return node
        raise KeyError(node_or_pose_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "classical_metadata": self.classical_metadata,
            "source": self.source,
            "geometry_status": self.geometry_status,
            "alpha_deg": self.alpha_deg,
            "beta_deg": self.beta_deg,
            "symmetry_symbol": self.symmetry_symbol,
            "symmetry_tolerance_mm": self.symmetry_tolerance_mm,
            "main_face_id": self.main_face_id,
            "main_face_ids": self.main_face_ids,
            "main_face_area_mm2": self.main_face_area_mm2,
            "main_face_min_span_mm": self.main_face_min_span_mm,
            "opposite_x_min_height_mm": self.opposite_x_min_height_mm,
            "robust_barrier_threshold_mm": self.robust_barrier_threshold_mm,
            "friction_policy": self.friction_policy,
            "minimum_braking_g": self.minimum_braking_g,
            "csa_load_model": self.csa_load_model,
            "pose_ranking_method": self.pose_ranking_method,
            "robustness_method": self.robustness_method,
            "minimum_csa_score": self.minimum_csa_score,
            "csa_cap_half_angle_deg": self.csa_cap_half_angle_deg,
            "csa_direction_samples": self.csa_direction_samples,
            "csa_rocking_fallback_pose_ids": self.csa_rocking_fallback_pose_ids,
            "axis_tolerance_deg": self.axis_tolerance_deg,
            "node_counts": {
                "total": len(self.nodes),
                "robust": sum(node.kind == "robust" for node in self.nodes),
                "metastable": sum(node.kind == "metastable" for node in self.nodes),
            },
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "unresolved_metastable_node_ids": self.unresolved_metastable_node_ids,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PoseRoadmap:
        if int(value.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported roadmap schema version.")
        return cls(
            schema_version=1,
            classical_metadata=value.get("classical_metadata", {}),
            source=str(value["source"]),
            geometry_status=str(value["geometry_status"]),
            alpha_deg=float(value["alpha_deg"]),
            beta_deg=float(value["beta_deg"]),
            symmetry_symbol=str(value["symmetry_symbol"]),
            symmetry_tolerance_mm=float(value["symmetry_tolerance_mm"]),
            main_face_id=int(value["main_face_id"]),
            main_face_ids=tuple(
                int(item)
                for item in value.get("main_face_ids", (value["main_face_id"],))
            ),
            main_face_area_mm2=float(value["main_face_area_mm2"]),
            main_face_min_span_mm=float(value.get("main_face_min_span_mm", 0.0)),
            opposite_x_min_height_mm=float(
                value.get("opposite_x_min_height_mm", 25.0)
            ),
            robust_barrier_threshold_mm=float(value["robust_barrier_threshold_mm"]),
            axis_tolerance_deg=float(value["axis_tolerance_deg"]),
            nodes=tuple(RoadmapNode.from_dict(item) for item in value["nodes"]),
            edges=tuple(RoadmapEdge.from_dict(item) for item in value["edges"]),
            unresolved_metastable_node_ids=tuple(
                int(item) for item in value.get("unresolved_metastable_node_ids", ())
            ),
            pose_ranking_method=value.get("pose_ranking_method", "rocking"),
            robustness_method=value.get("robustness_method", "rocking"),
            minimum_csa_score=float(
                value.get("minimum_csa_score", DEFAULT_CSA_ROBUST_THRESHOLD)
            ),
            csa_cap_half_angle_deg=float(
                value.get(
                    "csa_cap_half_angle_deg", DEFAULT_CSA_CAP_HALF_ANGLE_DEG
                )
            ),
            csa_direction_samples=int(
                value.get("csa_direction_samples", DEFAULT_CSA_DIRECTION_SAMPLES)
            ),
            csa_rocking_fallback_pose_ids=tuple(
                int(item)
                for item in value.get("csa_rocking_fallback_pose_ids", ())
            ),
            friction_policy=value.get("friction_policy", "range"),
            minimum_braking_g=value.get("minimum_braking_g"),
            csa_load_model=value.get("csa_load_model"),
        )


@dataclass(frozen=True, slots=True)
class RoutePlan:
    start_node: int
    target_node: int
    node_path: tuple[int, ...]
    edge_ids: tuple[str, ...]
    actuation_count: int
    total_abs_angle_deg: float
    geometric_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ACTION_SPECS: dict[str, tuple[np.ndarray, tuple[float, float]]] = {
    "floor_main_neg_x": (np.array([1.0, 0.0, 0.0]), (-180.0, 0.0)),
    "floor_main_pos_x": (np.array([1.0, 0.0, 0.0]), (0.0, 180.0)),
    "wall_main_neg_x": (np.array([1.0, 0.0, 0.0]), (-180.0, 0.0)),
    "wall_main_pos_x": (np.array([1.0, 0.0, 0.0]), (0.0, 180.0)),
    "free_y": (np.array([0.0, 1.0, 0.0]), (-180.0, 180.0)),
    "free_z": (np.array([0.0, 0.0, 1.0]), (-180.0, 180.0)),
}


def geometric_reliability_score(
    capture_width_deg: float,
    available_angle_span_deg: float,
    target_barrier_mm: float,
    robust_barrier_threshold_mm: float = 0.20,
) -> tuple[float, float, float]:
    """Return capture fraction, barrier score and their transparent product."""

    if available_angle_span_deg <= 0.0 or robust_barrier_threshold_mm <= 0.0:
        raise ValueError("Reliability reference spans must be positive.")
    capture = float(np.clip(capture_width_deg / available_angle_span_deg, 0.0, 1.0))
    barrier = float(np.clip(target_barrier_mm / robust_barrier_threshold_mm, 0.0, 1.0))
    return capture, barrier, capture * barrier


def _seated_height_mm(
    rotation: np.ndarray, vertices_centered: np.ndarray, gravity: np.ndarray
) -> float:
    points = (rotation @ vertices_centered.T).T
    potential_per_mass = (
        float(gravity[1]) * float(np.min(points[:, 1]))
        + float(gravity[2]) * float(np.min(points[:, 2]))
    )
    return potential_per_mass / float(np.linalg.norm(gravity))


def _rotation_about(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(axis, dtype=float) * math.radians(angle_deg)).as_matrix()


def _basin_half_width_deg(
    target_rotation: np.ndarray,
    axis: np.ndarray,
    direction: float,
    vertices_centered: np.ndarray,
    gravity: np.ndarray,
    *,
    maximum_deg: float = 90.0,
    coarse_step_deg: float = 1.0,
    refine_step_deg: float = 0.1,
) -> float:
    """Locate the first 1-D seated-energy crest away from a target pose."""

    baseline = _seated_height_mm(target_rotation, vertices_centered, gravity)
    previous = baseline
    peak = baseline
    coarse_boundary = maximum_deg
    found_descent = False
    angle = coarse_step_deg
    while angle <= maximum_deg + 1e-9:
        candidate = _rotation_about(axis, direction * angle) @ target_rotation
        height = _seated_height_mm(candidate, vertices_centered, gravity)
        peak = max(peak, height)
        if peak > baseline + 1e-7 and height < previous - 1e-6:
            coarse_boundary = max(0.0, angle - coarse_step_deg)
            found_descent = True
            break
        previous = height
        angle += coarse_step_deg
    if not found_descent:
        return maximum_deg

    low = max(0.0, coarse_boundary - coarse_step_deg)
    high = min(maximum_deg, coarse_boundary + coarse_step_deg)
    samples = np.arange(low, high + 0.5 * refine_step_deg, refine_step_deg)
    heights = [
        _seated_height_mm(
            _rotation_about(axis, direction * float(value)) @ target_rotation,
            vertices_centered,
            gravity,
        )
        for value in samples
    ]
    return float(samples[int(np.argmax(heights))])


def _class_main_face_placement(
    pose_class: PracticalPoseClass,
    poses: dict[int, ContactPose],
    main_face_ids: tuple[int, ...],
) -> tuple[bool, bool]:
    face_ids = set(main_face_ids)
    return (
        all(
            face_ids.intersection(poses[pose_id].floor_face_ids)
            for pose_id in pose_class.pose_ids
        ),
        all(
            face_ids.intersection(poses[pose_id].wall_face_ids)
            for pose_id in pose_class.pose_ids
        ),
    )


def _planar_face_min_span_mm(points: np.ndarray) -> float:
    """Return the smaller intrinsic PCA span of one planar support face."""

    centered = np.asarray(points, dtype=float) - np.mean(points, axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ axes[:2].T
    spans = np.ptp(coordinates, axis=0)
    positive = spans[spans > 1e-8]
    return float(np.min(positive)) if len(positive) else 0.0


def _best_actuated_relation(
    source_class: PracticalPoseClass,
    target_class: PracticalPoseClass,
    poses: dict[int, ContactPose],
    action: str,
    main_face_ids: tuple[int, ...],
    axis_tolerance_deg: float,
    continuous_axis_part: np.ndarray | None = None,
) -> tuple[float, float, int, int] | None:
    axis, domain = _ACTION_SPECS[action]
    face_ids = set(main_face_ids)
    best: tuple[float, float, int, int] | None = None
    for source_pose_id in source_class.pose_ids:
        source_pose = poses[source_pose_id]
        if action.startswith("floor_main_") and not face_ids.intersection(
            source_pose.floor_face_ids
        ):
            continue
        if action.startswith("wall_main_") and not face_ids.intersection(
            source_pose.wall_face_ids
        ):
            continue
        source_rotation = np.asarray(source_pose.rotation_chute_from_part, dtype=float)
        for target_pose_id in target_class.pose_ids:
            target_rotation = np.asarray(
                poses[target_pose_id].rotation_chute_from_part, dtype=float
            )
            if continuous_axis_part is None:
                rotvec = Rotation.from_matrix(
                    target_rotation @ source_rotation.T
                ).as_rotvec()
                angle_rad = float(np.linalg.norm(rotvec))
                if angle_rad <= 1e-9:
                    continue
                unit = rotvec / angle_rad
                alignment = float(np.dot(unit, axis))
                axis_error = math.degrees(
                    math.acos(float(np.clip(abs(alignment), -1.0, 1.0)))
                )
                signed_angle = math.degrees(angle_rad) * (
                    1.0 if alignment >= 0.0 else -1.0
                )
            else:
                source_axis = source_rotation @ continuous_axis_part
                target_axis = target_rotation @ continuous_axis_part
                source_perpendicular = source_axis - np.dot(source_axis, axis) * axis
                target_perpendicular = target_axis - np.dot(target_axis, axis) * axis
                source_norm = float(np.linalg.norm(source_perpendicular))
                target_norm = float(np.linalg.norm(target_perpendicular))
                if source_norm <= 1e-10 or target_norm <= 1e-10:
                    signed_angle = 0.0
                else:
                    source_perpendicular /= source_norm
                    target_perpendicular /= target_norm
                    signed_angle = math.degrees(
                        math.atan2(
                            float(
                                np.dot(
                                    axis,
                                    np.cross(source_perpendicular, target_perpendicular),
                                )
                            ),
                            float(np.dot(source_perpendicular, target_perpendicular)),
                        )
                    )
                rotated_axis = (
                    Rotation.from_rotvec(math.radians(signed_angle) * axis).as_matrix()
                    @ source_axis
                )
                axis_error = math.degrees(
                    math.acos(
                        float(np.clip(np.dot(rotated_axis, target_axis), -1.0, 1.0))
                    )
                )
                if abs(signed_angle) <= 1e-9 and axis_error <= axis_tolerance_deg:
                    continue
            if axis_error > axis_tolerance_deg:
                continue
            if abs(abs(signed_angle) - 180.0) <= 1e-7:
                if action in {"floor_main_neg_x", "wall_main_neg_x"}:
                    signed_angle = -180.0
                elif action in {"floor_main_pos_x", "wall_main_pos_x"}:
                    signed_angle = 180.0
            if signed_angle < domain[0] - 1e-7 or signed_angle > domain[1] + 1e-7:
                continue
            candidate = (axis_error, signed_angle, source_pose_id, target_pose_id)
            if best is None or (candidate[0], abs(candidate[1])) < (
                best[0],
                abs(best[1]),
            ):
                best = candidate
    return best


def _actuated_edges(
    classes: tuple[PracticalPoseClass, ...],
    nodes_by_id: dict[int, RoadmapNode],
    poses: dict[int, ContactPose],
    main_face_ids: tuple[int, ...],
    main_face_min_span_mm: float,
    opposite_x_min_height_mm: float,
    vertices_centered: np.ndarray,
    gravity: np.ndarray,
    robust_barrier_threshold_mm: float,
    axis_tolerance_deg: float,
    continuous_axis_part: np.ndarray | None = None,
) -> list[RoadmapEdge]:
    edges: list[RoadmapEdge] = []
    class_node_ids = {
        pose_class.class_id: pose_class.representative_pose_id for pose_class in classes
    }
    for source_class in classes:
        source_id = class_node_ids[source_class.class_id]
        for target_class in classes:
            if source_class.class_id == target_class.class_id:
                continue
            target_id = class_node_ids[target_class.class_id]
            for action, (axis, domain) in _ACTION_SPECS.items():
                if (
                    action.startswith("floor_main_")
                    and not nodes_by_id[source_id].main_face_on_floor
                ):
                    continue
                if (
                    action.startswith("wall_main_")
                    and not nodes_by_id[source_id].main_face_on_wall
                ):
                    continue
                if (
                    action in {"floor_main_pos_x", "wall_main_neg_x"}
                    and main_face_min_span_mm <= opposite_x_min_height_mm
                ):
                    continue
                relation = _best_actuated_relation(
                    source_class,
                    target_class,
                    poses,
                    action,
                    main_face_ids,
                    axis_tolerance_deg,
                    continuous_axis_part,
                )
                if relation is None:
                    continue
                axis_error, signed_angle, _, target_pose_id = relation
                target_rotation = np.asarray(
                    poses[target_pose_id].rotation_chute_from_part, dtype=float
                )
                left_width = _basin_half_width_deg(
                    target_rotation, axis, -1.0, vertices_centered, gravity
                )
                right_width = _basin_half_width_deg(
                    target_rotation, axis, 1.0, vertices_centered, gravity
                )
                interval = (
                    max(domain[0], signed_angle - left_width),
                    min(domain[1], signed_angle + right_width),
                )
                capture_width = max(0.0, interval[1] - interval[0])
                target_node = nodes_by_id[target_id]
                capture, barrier_score, score = geometric_reliability_score(
                    capture_width,
                    domain[1] - domain[0],
                    target_node.rocking_barrier_mm,
                    robust_barrier_threshold_mm,
                )
                edges.append(
                    RoadmapEdge(
                        edge_id=f"a{len(edges)}:{source_id}->{target_id}:{action}",
                        source=source_id,
                        target=target_id,
                        transition_kind="actuated",
                        actuation=action,  # type: ignore[arg-type]
                        axis_chute=tuple(float(value) for value in axis),
                        signed_angle_deg=float(signed_angle),
                        capture_interval_deg=tuple(float(value) for value in interval),
                        capture_width_deg=capture_width,
                        capture_fraction=capture,
                        target_barrier_score=barrier_score,
                        geometric_score=max(score, 1e-12),
                        axis_error_deg=float(axis_error),
                    )
                )
    return edges


def _geodesic_escape(
    source_rotation: np.ndarray,
    target_rotation: np.ndarray,
    vertices_centered: np.ndarray,
    gravity: np.ndarray,
    *,
    step_deg: float = 1.0,
) -> tuple[float, float, float, tuple[float, float, float]] | None:
    relative = target_rotation @ source_rotation.T
    rotvec = Rotation.from_matrix(relative).as_rotvec()
    angle_rad = float(np.linalg.norm(rotvec))
    if angle_rad <= 1e-9:
        return None
    axis = rotvec / angle_rad
    total_angle_deg = math.degrees(angle_rad)
    sample_count = max(2, math.ceil(total_angle_deg / step_deg) + 1)
    sample_angles = np.linspace(0.0, total_angle_deg, sample_count)
    heights = np.asarray(
        [
            _seated_height_mm(
                _rotation_about(axis, float(angle)) @ source_rotation,
                vertices_centered,
                gravity,
            )
            for angle in sample_angles
        ]
    )
    peak_index = int(np.argmax(heights))
    barrier = max(0.0, float(heights[peak_index] - heights[0]))
    endpoint_change = float(heights[-1] - heights[0])
    return (
        barrier,
        endpoint_change,
        float(sample_angles[peak_index]),
        tuple(float(value) for value in axis),
    )


def _actuated_relaxation_edges(
    classes: tuple[PracticalPoseClass, ...],
    nodes_by_id: dict[int, RoadmapNode],
    poses: dict[int, ContactPose],
    main_face_ids: tuple[int, ...],
    main_face_min_span_mm: float,
    opposite_x_min_height_mm: float,
    vertices_centered: np.ndarray,
    gravity: np.ndarray,
    robust_barrier_threshold_mm: float,
    axis_tolerance_deg: float,
    continuous_axis_part: np.ndarray | None,
    edge_index_start: int,
) -> list[RoadmapEdge]:
    """Connect an actuation to a downhill passive settling target.

    Rejected catalog poses remain transition metadata rather than roadmap
    nodes.  This first implementation is enabled for Cinf parts, where pose
    equivalence and the single physical orientation axis are unambiguous.
    """

    if continuous_axis_part is None:
        return []
    node_pose_ids = {
        pose_id for pose_class in classes for pose_id in pose_class.pose_ids
    }
    transition_pose_ids = tuple(sorted(set(poses).difference(node_pose_ids)))
    if not transition_pose_ids:
        return []

    settling_targets: dict[
        int, list[tuple[int, float, float, tuple[float, float, float]]]
    ] = {}
    for transition_pose_id in transition_pose_ids:
        source_rotation = np.asarray(
            poses[transition_pose_id].rotation_chute_from_part, dtype=float
        )
        candidates: list[tuple[int, float, float, tuple[float, float, float]]] = []
        for target_class in classes:
            target_id = target_class.representative_pose_id
            best: tuple[float, float, float, tuple[float, float, float]] | None = None
            for target_pose_id in target_class.pose_ids:
                target_rotation = np.asarray(
                    poses[target_pose_id].rotation_chute_from_part, dtype=float
                )
                escape = _geodesic_escape(
                    source_rotation,
                    target_rotation,
                    vertices_centered,
                    gravity,
                )
                if escape is None:
                    continue
                if best is None or (escape[0], escape[1]) < (best[0], best[1]):
                    best = escape
            if (
                best is not None
                and best[0] <= robust_barrier_threshold_mm
                and best[1] < -1e-4
            ):
                candidates.append((target_id, best[0], best[2], best[3]))
        if candidates:
            minimum_barrier = min(value[1] for value in candidates)
            settling_targets[transition_pose_id] = [
                value for value in candidates if value[1] <= minimum_barrier + 0.02
            ]

    edges: list[RoadmapEdge] = []
    face_ids = set(main_face_ids)
    for source_class in classes:
        source_id = source_class.representative_pose_id
        source_node = nodes_by_id[source_id]
        for transition_pose_id, targets in settling_targets.items():
            transition_pose = poses[transition_pose_id]
            transition_class = PracticalPoseClass(
                class_id=-1,
                representative_pose_id=transition_pose_id,
                pose_ids=(transition_pose_id,),
                floor_contact_type=transition_pose.floor_contact_type,
                wall_contact_type=transition_pose.wall_contact_type,
            )
            for action, (axis, domain) in _ACTION_SPECS.items():
                if action.startswith("floor_main_") and not source_node.main_face_on_floor:
                    continue
                if action.startswith("wall_main_") and not source_node.main_face_on_wall:
                    continue
                if (
                    action in {"floor_main_pos_x", "wall_main_neg_x"}
                    and main_face_min_span_mm <= opposite_x_min_height_mm
                ):
                    continue
                relation = _best_actuated_relation(
                    source_class,
                    transition_class,
                    poses,
                    action,
                    tuple(face_ids),
                    axis_tolerance_deg,
                    continuous_axis_part,
                )
                if relation is None:
                    continue
                axis_error, signed_angle, _, _ = relation
                half_width = max(0.1, axis_tolerance_deg)
                interval = (
                    max(domain[0], signed_angle - half_width),
                    min(domain[1], signed_angle + half_width),
                )
                capture_width = max(0.0, interval[1] - interval[0])
                for target_id, escape_barrier, saddle_angle, _ in targets:
                    if target_id == source_id:
                        continue
                    target_node = nodes_by_id[target_id]
                    capture, barrier_score, base_score = geometric_reliability_score(
                        capture_width,
                        domain[1] - domain[0],
                        target_node.rocking_barrier_mm,
                        robust_barrier_threshold_mm,
                    )
                    settling_score = float(
                        np.clip(
                            1.0 - escape_barrier / robust_barrier_threshold_mm,
                            0.0,
                            1.0,
                        )
                    )
                    edge_index = edge_index_start + len(edges)
                    edges.append(
                        RoadmapEdge(
                            edge_id=(
                                f"r{edge_index}:{source_id}->{target_id}:"
                                f"{action}:via{transition_pose_id}"
                            ),
                            source=source_id,
                            target=target_id,
                            transition_kind="actuated",
                            actuation=action,  # type: ignore[arg-type]
                            axis_chute=tuple(float(value) for value in axis),
                            signed_angle_deg=float(signed_angle),
                            capture_interval_deg=tuple(
                                float(value) for value in interval
                            ),
                            capture_width_deg=capture_width,
                            capture_fraction=capture,
                            target_barrier_score=barrier_score,
                            geometric_score=max(
                                base_score * settling_score, 1e-12
                            ),
                            escape_barrier_mm=float(escape_barrier),
                            saddle_angle_deg=float(saddle_angle),
                            axis_error_deg=float(axis_error),
                            settling_pose_ids=(transition_pose_id,),
                        )
                    )
    return edges


def _passive_edges(
    classes: tuple[PracticalPoseClass, ...],
    nodes_by_id: dict[int, RoadmapNode],
    poses: dict[int, ContactPose],
    vertices_centered: np.ndarray,
    gravity: np.ndarray,
    robust_barrier_threshold_mm: float,
) -> tuple[list[RoadmapEdge], tuple[int, ...]]:
    edges: list[RoadmapEdge] = []
    unresolved: list[int] = []
    for source_class in classes:
        source_id = source_class.representative_pose_id
        if nodes_by_id[source_id].kind != "metastable":
            continue
        candidates: list[
            tuple[float, float, float, tuple[float, float, float], int]
        ] = []
        for target_class in classes:
            target_id = target_class.representative_pose_id
            if target_id == source_id:
                continue
            best = None
            for source_pose_id in source_class.pose_ids:
                source_rotation = np.asarray(
                    poses[source_pose_id].rotation_chute_from_part, dtype=float
                )
                for target_pose_id in target_class.pose_ids:
                    target_rotation = np.asarray(
                        poses[target_pose_id].rotation_chute_from_part, dtype=float
                    )
                    escape = _geodesic_escape(
                        source_rotation,
                        target_rotation,
                        vertices_centered,
                        gravity,
                    )
                    if escape is None:
                        continue
                    if best is None or (escape[0], abs(escape[1])) < (
                        best[0],
                        abs(best[1]),
                    ):
                        best = escape
            if best is not None and best[0] < robust_barrier_threshold_mm and best[1] <= 1e-4:
                candidates.append((*best, target_id))
        if not candidates:
            unresolved.append(source_id)
            continue
        minimum_barrier = min(value[0] for value in candidates)
        selected = [
            value for value in candidates if value[0] <= minimum_barrier + 0.02
        ]
        for barrier, _, saddle_angle, axis, target_id in selected:
            target_node = nodes_by_id[target_id]
            escape_score = float(
                np.clip(1.0 - barrier / robust_barrier_threshold_mm, 0.0, 1.0)
            )
            barrier_score = float(
                np.clip(
                    target_node.rocking_barrier_mm / robust_barrier_threshold_mm,
                    0.0,
                    1.0,
                )
            )
            score = max(1e-12, escape_score * barrier_score)
            edges.append(
                RoadmapEdge(
                    edge_id=f"p{len(edges)}:{source_id}->{target_id}",
                    source=source_id,
                    target=target_id,
                    transition_kind="passive_tip",
                    actuation="passive",
                    axis_chute=axis,
                    signed_angle_deg=0.0,
                    capture_interval_deg=None,
                    capture_width_deg=0.0,
                    capture_fraction=escape_score,
                    target_barrier_score=barrier_score,
                    geometric_score=score,
                    escape_barrier_mm=float(barrier),
                    saddle_angle_deg=float(saddle_angle),
                )
            )
    return edges, tuple(unresolved)


def _relabel_roadmap(
    nodes: list[RoadmapNode],
    edges: list[RoadmapEdge],
    unresolved_metastable_node_ids: tuple[int, ...],
    *,
    pose_ranking_method: PoseRankingMethod = "rocking",
) -> tuple[tuple[RoadmapNode, ...], tuple[RoadmapEdge, ...], tuple[int, ...]]:
    """Assign compact stability-ranked roadmap IDs."""

    if pose_ranking_method in {"rocking", "cwsa", "standard_csa", "crsa"}:
        ordered_nodes = sorted(
            nodes,
            key=lambda node: (
                -node.rocking_barrier_mm,
                node.original_catalog_pose_id,
            ),
        )
    elif pose_ranking_method == "csa":
        ordered_nodes = sorted(
            nodes,
            key=lambda node: (
                node.kind != "robust",
                node.csa_stability_index is None,
                -(
                    round(node.csa_stability_index, 6)
                    if node.csa_stability_index is not None
                    else 0.0
                ),
                -node.rocking_barrier_mm,
                node.original_catalog_pose_id,
            ),
        )
    else:
        raise ValueError("pose_ranking_method must be 'rocking' or 'csa'.")
    new_id_by_old_id = {
        node.node_id: new_id for new_id, node in enumerate(ordered_nodes)
    }
    relabelled_nodes = tuple(
        replace(node, node_id=new_id_by_old_id[node.node_id])
        for node in ordered_nodes
    )
    # New methods change presentation order, not identity or route references.
    # The historical 'csa' option retains its legacy CWSA numbering.
    if pose_ranking_method in {"cwsa", "standard_csa", "crsa"}:
        def order_key(node):
            score = (node.csa_stability_index if pose_ranking_method == "cwsa"
                     else node.classical_metrics.get(pose_ranking_method, {}).get("raw_score"))
            return (score is None, -(score or 0.0), node.node_id)
        relabelled_nodes = tuple(sorted(relabelled_nodes, key=order_key))

    relabelled_edges: list[RoadmapEdge] = []
    for edge in edges:
        source = new_id_by_old_id[edge.source]
        target = new_id_by_old_id[edge.target]
        prefix = edge.edge_id.split(":", 1)[0]
        edge_id = (
            f"{prefix}:{source}->{target}"
            if edge.transition_kind == "passive_tip"
            else f"{prefix}:{source}->{target}:{edge.actuation}"
        )
        relabelled_edges.append(
            replace(edge, edge_id=edge_id, source=source, target=target)
        )

    return (
        relabelled_nodes,
        tuple(relabelled_edges),
        tuple(
            new_id_by_old_id[node_id]
            for node_id in unresolved_metastable_node_ids
        ),
    )


def build_pose_roadmap(
    mesh_path: str | Path,
    *,
    alpha_deg: float = 45.0,
    beta_deg: float = 20.0,
    onset_alpha_deg: float = 45.0,
    onset_beta_deg: float = 15.0,
    symmetry_tolerance_mm: float | None = None,
    angular_tolerance_deg: float = 1.0,
    surface_displacement_tolerance_mm: float = 0.5,
    robust_barrier_threshold_mm: float = 0.73,
    minimum_braking_g: float = 0.0,
    opposite_x_min_height_mm: float = 25.0,
    geometry_status: str = "provisional",
    pose_ranking_method: PoseRankingMethod = "rocking",
    robustness_method: RobustnessMethod = "rocking",
    minimum_csa_score: float = DEFAULT_CSA_ROBUST_THRESHOLD,
    csa_cap_half_angle_deg: float = DEFAULT_CSA_CAP_HALF_ANGLE_DEG,
    csa_direction_samples: int = DEFAULT_CSA_DIRECTION_SAMPLES,
    friction_policy: FrictionPolicy = "zero",
    include_csa: bool = False,
    classical_methods: tuple[str, ...] = (),
    minimum_classical_score: float | None = None,
) -> PoseRoadmap:
    """Build a roadmap, optionally measuring CWSA without changing classification."""

    if geometry_status not in {"provisional", "verified"}:
        raise ValueError("geometry_status must be 'provisional' or 'verified'.")
    if opposite_x_min_height_mm < 0.0:
        raise ValueError("opposite_x_min_height_mm must be non-negative.")
    if pose_ranking_method not in {"rocking", "csa", "cwsa", *CLASSICAL_METHODS}:
        raise ValueError("pose_ranking_method must be 'rocking' or 'csa'.")
    if robustness_method not in {"rocking", "csa", "cwsa", *CLASSICAL_METHODS}:
        raise ValueError("robustness_method must be 'rocking' or 'csa'.")
    if friction_policy not in {"range", "zero"}:
        raise ValueError("friction_policy must be 'range' or 'zero'.")
    if not math.isfinite(minimum_csa_score) or not 0.0 <= minimum_csa_score <= 1.0:
        raise ValueError("minimum_csa_score must lie between 0 and 1.")
    classical_methods = tuple(dict.fromkeys((*classical_methods, *(
        method for method in (pose_ranking_method, robustness_method) if method in CLASSICAL_METHODS
    ))))
    if any(method not in CLASSICAL_METHODS for method in classical_methods):
        raise ValueError("Unknown classical method.")
    if robustness_method in CLASSICAL_METHODS and (
        minimum_classical_score is None or not math.isfinite(minimum_classical_score)
        or minimum_classical_score <= 0
    ):
        raise ValueError("CSA/CRSA classification requires an explicit positive raw-score threshold (sr/mm).")
    catalog = build_pose_catalog(mesh_path)
    nominal = analyze_pose_stability(
        mesh_path,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        onset_alpha_deg=onset_alpha_deg,
        onset_beta_deg=onset_beta_deg,
        catalog=catalog,
        quasistatic_only=catalog.continuous_symmetry_axis_part is None,
        friction_policy=friction_policy,
    )
    nominal_ids = nominal.stable_pose_ids
    conditional_ids = (
        nominal.friction_dependent_pose_ids
        if not nominal.quasistatic_only
        else ()
    )
    roadmap_pose_ids = (
        tuple(sorted(set(nominal_ids + conditional_ids)))
        if catalog.continuous_symmetry_axis_part is not None
        else nominal_ids
    )
    disturbance = analyze_disturbance_robustness(
        mesh_path,
        pose_ids=roadmap_pose_ids,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        onset_alpha_deg=onset_alpha_deg,
        onset_beta_deg=onset_beta_deg,
        catalog=catalog,
    )
    rocking = analyze_rocking_barriers(
        mesh_path,
        pose_ids=roadmap_pose_ids,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        catalog=catalog,
    )
    robust_filter = filter_finite_disturbance_robustness(
        rocking,
        disturbance,
        catalog,
        minimum_barrier_height_mm=robust_barrier_threshold_mm,
        minimum_braking_g=minimum_braking_g,
    )
    rocking_robust_pose_ids = set(robust_filter.accepted_pose_ids).intersection(
        nominal_ids
    )
    csa = None
    csa_filter = None
    csa_rocking_fallback_pose_ids: set[int] = set()
    if include_csa or pose_ranking_method in {"csa", "cwsa"} or robustness_method in {"csa", "cwsa"}:
        csa = analyze_contact_wrench_solid_angle(
            mesh_path,
            pose_ids=roadmap_pose_ids,
            alpha_deg=alpha_deg,
            beta_deg=beta_deg,
            onset_alpha_deg=onset_alpha_deg,
            onset_beta_deg=onset_beta_deg,
            cap_half_angle_deg=csa_cap_half_angle_deg,
            direction_samples=csa_direction_samples,
            friction_policy=friction_policy,
            catalog=catalog,
        )
        csa_filter = filter_contact_wrench_solid_angle(
            csa, minimum_score=minimum_csa_score
        )

    classical = analyze_chute_classical(
        mesh_path, catalog, roadmap_pose_ids, alpha_deg=alpha_deg, beta_deg=beta_deg,
        methods=classical_methods,
    ) if classical_methods else {}
    if robustness_method in CLASSICAL_METHODS:
        robust_pose_ids = {
            pid for pid, result in classical.items()
            if getattr(result, robustness_method) is not None
            and getattr(result, robustness_method) >= minimum_classical_score
        }.intersection(nominal_ids)
        if minimum_braking_g > 0:
            robust_pose_ids.intersection_update(
                value.pose_id for value in disturbance.capacities
                if value.critical_braking_g >= minimum_braking_g
            )
    elif robustness_method == "rocking":
        robust_pose_ids = set(rocking_robust_pose_ids)
    else:
        assert csa is not None and csa_filter is not None
        disturbance_by_pose_id = {
            value.pose_id: value for value in disturbance.capacities
        }
        robust_pose_ids = set()
        for pose_id in csa_filter.accepted_pose_ids:
            if (
                minimum_braking_g == 0.0
                or disturbance_by_pose_id[pose_id].critical_braking_g
                >= minimum_braking_g
            ):
                robust_pose_ids.add(pose_id)
        # A fixed-contact solid-angle model cannot represent a circular part
        # reseating continuously as its line contacts roll.  Preserve the
        # established rocking classification for those explicitly reported
        # not-applicable states instead of assigning a misleading zero score.
        csa_rocking_fallback_pose_ids = set(
            csa_filter.not_applicable_pose_ids
        ).intersection(rocking_robust_pose_ids)
        robust_pose_ids.update(csa_rocking_fallback_pose_ids)
        robust_pose_ids.intersection_update(nominal_ids)
    poses = {pose.pose_id: pose for pose in catalog.poses}
    if catalog.continuous_symmetry_axis_part is not None and minimum_braking_g > 0.0:
        # Only apply this dynamic safeguard when braking is explicitly requested.
        # End-face/mantle states of an axially asymmetric Cinf body can be
        # statically admissible yet tip at the first braking impulse.  Keep
        # them as metastable roadmap states; only lateral-lateral support is
        # treated as dynamically robust without experimental calibration.
        robust_pose_ids = {
            pose_id
            for pose_id in robust_pose_ids
            if poses[pose_id].floor_contact_dimension == 1
            and poses[pose_id].wall_contact_dimension == 1
        }
    mesh = load_solid_mesh(mesh_path)
    vertices_centered_full = np.asarray(mesh.vertices, dtype=float) - np.asarray(
        mesh.center_mass, dtype=float
    )
    hull_vertices_centered = np.asarray(mesh.convex_hull.vertices, dtype=float) - np.asarray(
        mesh.center_mass, dtype=float
    )
    symmetry = detect_rotational_symmetry(
        mesh_path, tolerance_mm=symmetry_tolerance_mm
    )
    clustering = cluster_practical_contact_poses(
        catalog,
        vertices_centered_full,
        roadmap_pose_ids,
        symmetry=symmetry,
        angular_tolerance_deg=angular_tolerance_deg,
        surface_displacement_tolerance_mm=max(
            surface_displacement_tolerance_mm, symmetry.tolerance_mm
        ),
    )
    main_face = max(catalog.support_faces, key=lambda value: value.area_mm2)
    main_faces = tuple(
        face
        for face in catalog.support_faces
        if math.isclose(face.area_mm2, main_face.area_mm2, rel_tol=1e-6, abs_tol=1e-6)
    )
    main_face_ids = tuple(face.face_id for face in main_faces)
    main_face_min_span_mm = min(
        _planar_face_min_span_mm(
            hull_vertices_centered[np.asarray(face.vertex_indices, dtype=int)]
        )
        for face in main_faces
    )
    barriers = {value.pose_id: value for value in rocking.barriers}
    csa_by_pose_id = (
        {value.pose_id: value for value in csa.poses} if csa is not None else {}
    )
    nodes: list[RoadmapNode] = []
    for pose_class in clustering.classes:
        representative = poses[pose_class.representative_pose_id]
        main_floor, main_wall = _class_main_face_placement(
            pose_class, poses, main_face_ids
        )
        class_csa_values = tuple(
            csa_by_pose_id[pose_id]
            for pose_id in pose_class.pose_ids
            if pose_id in csa_by_pose_id
        )
        class_csa_applicable = (
            None
            if csa is None
            else bool(class_csa_values)
            and all(value.applicable for value in class_csa_values)
        )
        nodes.append(
            RoadmapNode(
                node_id=pose_class.representative_pose_id,
                pose_ids=pose_class.pose_ids,
                kind=(
                    "robust"
                    if all(pose_id in robust_pose_ids for pose_id in pose_class.pose_ids)
                    else "metastable"
                ),
                cad_status=geometry_status,
                representative_quaternion_xyzw=representative.quaternion_xyzw,
                floor_contact_topology=representative.floor_contact_topology,
                wall_contact_topology=representative.wall_contact_topology,
                rocking_barrier_mm=min(
                    barriers[pose_id].barrier_height_mm for pose_id in pose_class.pose_ids
                ),
                main_face_on_floor=main_floor,
                main_face_on_wall=main_wall,
                csa_stability_index=(
                    min(value.score for value in class_csa_values)
                    if class_csa_applicable
                    else None
                ),
                csa_feasible_fraction=(
                    min(value.feasible_fraction for value in class_csa_values)
                    if class_csa_applicable
                    else None
                ),
                csa_feasible_solid_angle_sr=(
                    min(
                        value.feasible_solid_angle_sr
                        for value in class_csa_values
                    )
                    if class_csa_applicable
                    else None
                ),
                csa_angular_clearance_deg=(
                    min(value.angular_clearance_deg for value in class_csa_values)
                    if class_csa_applicable
                    else None
                ),
                csa_applicable=class_csa_applicable,
                classical_metrics={
                    method: {
                        "raw_score": (
                            min(getattr(classical[pid], method) for pid in pose_class.pose_ids)
                            if all(getattr(classical[pid], method) is not None for pid in pose_class.pose_ids)
                            else None
                        ),
                        "units": "sr/mm",
                        "reason": "; ".join(sorted({
                            getattr(classical[pid], "csa_reason" if method == "standard_csa" else "crsa_reason")
                            for pid in pose_class.pose_ids
                        })),
                    }
                    for method in classical_methods
                },
            )
        )
    # Normalize before any output filtering. Catalogue sampling multiplicity is
    # not physical aspect multiplicity; each practical class contributes once.
    for method in classical_methods:
        total = sum(node.classical_metrics[method]["raw_score"] or 0 for node in nodes)
        for node in nodes:
            metric = node.classical_metrics[method]
            metric["normalized_share"] = (
                metric["raw_score"] / total if metric["raw_score"] is not None and total > 0 else None
            )
    nodes_by_id = {node.node_id: node for node in nodes}
    gravity = ChuteFrame(alpha_deg=alpha_deg, beta_deg=beta_deg).gravity_chute()
    actuated = _actuated_edges(
        clustering.classes,
        nodes_by_id,
        poses,
        main_face_ids,
        main_face_min_span_mm,
        opposite_x_min_height_mm,
        hull_vertices_centered,
        gravity,
        robust_barrier_threshold_mm,
        angular_tolerance_deg,
        (
            None
            if catalog.continuous_symmetry_axis_part is None
            else np.asarray(catalog.continuous_symmetry_axis_part, dtype=float)
        ),
    )
    relaxed = _actuated_relaxation_edges(
        clustering.classes,
        nodes_by_id,
        poses,
        main_face_ids,
        main_face_min_span_mm,
        opposite_x_min_height_mm,
        hull_vertices_centered,
        gravity,
        robust_barrier_threshold_mm,
        angular_tolerance_deg,
        (
            None
            if catalog.continuous_symmetry_axis_part is None
            else np.asarray(catalog.continuous_symmetry_axis_part, dtype=float)
        ),
        len(actuated),
    )
    direct_keys = {(edge.source, edge.target, edge.actuation) for edge in actuated}
    best_relaxed: dict[tuple[int, int, ActuationKind], RoadmapEdge] = {}
    for edge in relaxed:
        key = (edge.source, edge.target, edge.actuation)
        if key in direct_keys:
            continue
        known = best_relaxed.get(key)
        if known is None or (
            edge.geometric_score,
            -(edge.escape_barrier_mm or 0.0),
            -abs(edge.signed_angle_deg),
        ) > (
            known.geometric_score,
            -(known.escape_barrier_mm or 0.0),
            -abs(known.signed_angle_deg),
        ):
            best_relaxed[key] = edge
    relaxed = list(best_relaxed.values())
    passive, unresolved = _passive_edges(
        clustering.classes,
        nodes_by_id,
        poses,
        hull_vertices_centered,
        gravity,
        robust_barrier_threshold_mm,
    )
    relabelled_nodes, relabelled_edges, relabelled_unresolved = _relabel_roadmap(
        nodes,
        actuated + relaxed + passive,
        unresolved,
        pose_ranking_method=pose_ranking_method,
    )
    return PoseRoadmap(
        schema_version=1,
        source=str(Path(mesh_path).expanduser().resolve()),
        geometry_status=geometry_status,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        symmetry_symbol=symmetry.symbol,
        symmetry_tolerance_mm=symmetry.tolerance_mm,
        main_face_id=main_face.face_id,
        main_face_ids=main_face_ids,
        main_face_area_mm2=main_face.area_mm2,
        main_face_min_span_mm=main_face_min_span_mm,
        opposite_x_min_height_mm=opposite_x_min_height_mm,
        robust_barrier_threshold_mm=robust_barrier_threshold_mm,
        axis_tolerance_deg=angular_tolerance_deg,
        nodes=relabelled_nodes,
        edges=relabelled_edges,
        unresolved_metastable_node_ids=relabelled_unresolved,
        pose_ranking_method=pose_ranking_method,
        robustness_method=robustness_method,
        minimum_csa_score=minimum_csa_score,
        csa_cap_half_angle_deg=csa_cap_half_angle_deg,
        csa_direction_samples=csa_direction_samples,
        csa_rocking_fallback_pose_ids=tuple(
            sorted(csa_rocking_fallback_pose_ids)
        ),
        friction_policy=friction_policy,
        minimum_braking_g=minimum_braking_g,
        csa_load_model=csa.to_dict()["load_model"] if csa is not None else None,
        classical_metadata={
            "model": MODEL_VERSION, "experimental": True,
            "methods": classical_methods, "minimum_raw_score": minimum_classical_score,
            "normalization": "applicable practical pose classes before robust filtering; not a calibrated probability",
            "class_aggregation": "minimum across equivalent catalogue representations",
            "longitudinal_restraint_modelled": False,
            "references": ["10.1080/00207549508904822", "10.1007/BF01178966"],
            "diagnostics": {str(pid): result.to_dict() for pid, result in classical.items()},
        } if classical_methods else {},
    )


def find_best_route(
    roadmap: PoseRoadmap,
    start_pose_id: int,
    target_pose_id: int,
    *,
    max_actions: int = 4,
) -> RoutePlan:
    """Maximise geometric route score with at most four open-loop impulses."""

    if max_actions < 0 or max_actions > 4:
        raise ValueError("max_actions must be between 0 and 4.")
    start = roadmap.node(start_pose_id).node_id
    target = roadmap.node(target_pose_id).node_id
    if start == target:
        return RoutePlan(start, target, (start,), (), 0, 0.0, 1.0)
    outgoing: dict[int, list[RoadmapEdge]] = {}
    for edge in roadmap.edges:
        outgoing.setdefault(edge.source, []).append(edge)

    queue: list[tuple[float, int, float, int, tuple[int, ...], tuple[str, ...]]] = [
        (0.0, 0, 0.0, start, (start,), ())
    ]
    best: dict[tuple[int, int], tuple[float, float]] = {(start, 0): (0.0, 0.0)}
    while queue:
        cost, action_count, angle_sum, node_id, nodes, edge_ids = heapq.heappop(queue)
        if node_id == target:
            return RoutePlan(
                start_node=start,
                target_node=target,
                node_path=nodes,
                edge_ids=edge_ids,
                actuation_count=action_count,
                total_abs_angle_deg=angle_sum,
                geometric_score=math.exp(-cost),
            )
        for edge in outgoing.get(node_id, ()):
            new_actions = action_count + edge.actuation_count
            if new_actions > max_actions:
                continue
            if edge.transition_kind == "passive_tip" and edge.target in nodes:
                continue
            edge_score = float(np.clip(edge.geometric_score, 1e-12, 1.0))
            new_cost = cost - math.log(edge_score)
            new_angle = angle_sum + (
                abs(edge.signed_angle_deg) if edge.transition_kind == "actuated" else 0.0
            )
            key = (edge.target, new_actions)
            previous = best.get(key)
            if previous is not None and (new_cost, new_angle) >= previous:
                continue
            best[key] = (new_cost, new_angle)
            heapq.heappush(
                queue,
                (
                    new_cost,
                    new_actions,
                    new_angle,
                    edge.target,
                    nodes + (edge.target,),
                    edge_ids + (edge.edge_id,),
                ),
            )
    raise ValueError(
        f"No route from pose {start_pose_id} to {target_pose_id} within {max_actions} actuations."
    )


def save_roadmap_json(roadmap: PoseRoadmap, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = roadmap.to_dict()
    try:
        thumbnails = create_pose_thumbnails(
            roadmap.source,
            (node.original_catalog_pose_id for node in roadmap.nodes),
            width_px=240,
            height_px=180,
            dpi=120,
        )
        for node, node_payload in zip(roadmap.nodes, payload["nodes"], strict=True):
            buffer = BytesIO()
            plt.imsave(
                buffer,
                thumbnails[node.original_catalog_pose_id],
                format="png",
            )
            node_payload["thumbnail_png_base64"] = base64.b64encode(
                buffer.getvalue()
            ).decode("ascii")
    except (OSError, RuntimeError, ValueError):
        # JSON stays usable for route planning even when a source mesh is no
        # longer available at its recorded path.
        pass
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def load_roadmap_json(path: str | Path) -> PoseRoadmap:
    source = Path(path).expanduser().resolve()
    return PoseRoadmap.from_dict(json.loads(source.read_text(encoding="utf-8")))


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return json.dumps(value, allow_nan=False)
    return json.dumps(str(value), ensure_ascii=False)


def _yaml_lines(value: Any, indent: int = 0) -> list[str]:
    """Serialize the handover schema without adding a runtime YAML dependency."""

    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, child in value.items():
            if isinstance(child, (dict, list, tuple)) and child:
                if isinstance(child, (list, tuple)) and all(
                    not isinstance(item, (dict, list, tuple)) for item in child
                ):
                    inline = ", ".join(_yaml_scalar(item) for item in child)
                    lines.append(f"{prefix}{key}: [{inline}]")
                else:
                    lines.append(f"{prefix}{key}:")
                    lines.extend(_yaml_lines(child, indent + 2))
            elif isinstance(child, (dict, list, tuple)):
                lines.append(f"{prefix}{key}: {'{}' if isinstance(child, dict) else '[]'}")
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(child)}")
        return lines
    if isinstance(value, (list, tuple)):
        lines = []
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(value)}"]


def _transition_action_handover(edge: RoadmapEdge) -> dict[str, Any]:
    conditions = {
        "floor_main_neg_x": ("x", "negative", "main_face_on_floor", None),
        "floor_main_pos_x": (
            "x",
            "positive",
            "main_face_on_floor",
            "main_face_min_span_above_threshold",
        ),
        "wall_main_neg_x": (
            "x",
            "negative",
            "main_face_on_wall",
            "main_face_min_span_above_threshold",
        ),
        "wall_main_pos_x": ("x", "positive", "main_face_on_wall", None),
        "free_y": ("y", "positive" if edge.signed_angle_deg > 0 else "negative", None, None),
        "free_z": ("z", "positive" if edge.signed_angle_deg > 0 else "negative", None, None),
        "passive": ("variable", "passive", None, None),
    }
    axis, direction, surface, additional = conditions[edge.actuation]
    return {
        "name": edge.actuation,
        "axis": axis,
        "axis_vector_chute": edge.axis_chute,
        "direction": direction,
        "commanded_angle_deg": (
            edge.signed_angle_deg if edge.transition_kind == "actuated" else None
        ),
        "impulse_cost": edge.actuation_count,
        "surface_requirement": surface,
        "additional_requirement": additional,
    }


def roadmap_handover_dict(roadmap: PoseRoadmap) -> dict[str, Any]:
    """Return the compact, experiment-editable YAML handover representation."""

    return {
        "format": "bibazu_pose_roadmap_handover",
        "schema_version": 1,
        "part": {
            "name": Path(roadmap.source).stem,
            "mesh_source": roadmap.source,
            "cad_status": roadmap.geometry_status,
        },
        "chute": {
            "coordinate_system": "right_handed",
            "x": "downhill",
            "y": "away_from_wall",
            "z": "away_from_floor",
            "alpha_deg": roadmap.alpha_deg,
            "beta_deg": roadmap.beta_deg,
        },
        "classification": {
            "classical_metadata": roadmap.classical_metadata,
            "symmetry": roadmap.symmetry_symbol,
            "friction_policy": roadmap.friction_policy,
            "minimum_braking_g": roadmap.minimum_braking_g,
            "csa_load_model": roadmap.csa_load_model,
            "pose_ranking_method": roadmap.pose_ranking_method,
            "robustness_method": roadmap.robustness_method,
            "robust_pose_ids": tuple(
                node.node_id for node in roadmap.nodes if node.kind == "robust"
            ),
            "metastable_pose_ids": tuple(
                node.node_id for node in roadmap.nodes if node.kind == "metastable"
            ),
            "main_face_ids": roadmap.main_face_ids,
            "main_face_min_span_mm": roadmap.main_face_min_span_mm,
            "opposite_x_min_height_mm": roadmap.opposite_x_min_height_mm,
            "robust_barrier_threshold_mm": roadmap.robust_barrier_threshold_mm,
            "minimum_csa_score": roadmap.minimum_csa_score,
            "csa_cap_half_angle_deg": roadmap.csa_cap_half_angle_deg,
            "csa_direction_samples": roadmap.csa_direction_samples,
            "csa_rocking_fallback_catalog_pose_ids": (
                roadmap.csa_rocking_fallback_pose_ids
            ),
            "unresolved_metastable_pose_ids": roadmap.unresolved_metastable_node_ids,
        },
        "poses": [
            {
                "id": node.node_id,
                "original_catalog_pose_id": node.original_catalog_pose_id,
                "equivalent_catalog_pose_ids": node.pose_ids,
                "stability": node.kind,
                "planner_role": (
                    "stable_target" if node.kind == "robust" else "transient_intermediate"
                ),
                "orientation_quaternion_xyzw": node.representative_quaternion_xyzw,
                "contacts": {
                    "floor": node.floor_contact_topology,
                    "wall": node.wall_contact_topology,
                    "main_face_on_floor": node.main_face_on_floor,
                    "main_face_on_wall": node.main_face_on_wall,
                },
                "rocking_barrier_mm": node.rocking_barrier_mm,
                "csa_stability_index": node.csa_stability_index,
                "csa_feasible_fraction": node.csa_feasible_fraction,
                "csa_feasible_solid_angle_sr": node.csa_feasible_solid_angle_sr,
                "csa_angular_clearance_deg": node.csa_angular_clearance_deg,
                "csa_applicable": node.csa_applicable,
                "classical_metrics": node.classical_metrics,
                "cad_status": node.cad_status,
            }
            for node in roadmap.nodes
        ],
        "transitions": [
            {
                "id": edge.edge_id,
                "from_pose": edge.source,
                "to_pose": edge.target,
                "directed": True,
                "type": edge.transition_kind,
                "action": _transition_action_handover(edge),
                "capture": {
                    "interval_deg": edge.capture_interval_deg,
                    "width_deg": edge.capture_width_deg,
                    "fraction": edge.capture_fraction,
                },
                "geometry": {
                    "target_barrier_score": edge.target_barrier_score,
                    "geometric_score": edge.geometric_score,
                    "axis_error_deg": edge.axis_error_deg,
                    "passive_escape_barrier_mm": edge.escape_barrier_mm,
                    "passive_saddle_angle_deg": edge.saddle_angle_deg,
                    "passive_settling_via_catalog_pose_ids": (
                        edge.settling_pose_ids
                    ),
                },
                "experimental": {
                    "status": "untested",
                    "trials": None,
                    "successes": None,
                    "empirical_success_rate": None,
                    "difficulty_rating": None,
                    "notes": "",
                },
            }
            for edge in roadmap.edges
        ],
    }


def save_roadmap_yaml(roadmap: PoseRoadmap, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = "---\n" + "\n".join(
        _yaml_lines(roadmap_handover_dict(roadmap))
    ) + "\n"
    destination.write_text(document, encoding="utf-8")
    return destination


_YAML_README_DE = """# BiBaZu-Posenroadmap: YAML-Übergabe

Die YAML-Datei enthält die physisch zusammengeführten Posen und alle aktuell
bekannten **gerichteten** direkten Übergänge. Eine Kante von `from_pose` nach
`to_pose` gilt nicht automatisch in Gegenrichtung.

## Posen

- `id` ist die Roadmap-ID und wird in `from_pose`/`to_pose` referenziert.
- Roadmap-IDs beginnen bei 0 und folgen der unter
  `classification.pose_ranking_method` gewählten Stabilitätsmetrik in
  absteigender Reihenfolge. `original_catalog_pose_id`
  und `equivalent_catalog_pose_ids` bewahren die ursprünglichen Katalog-IDs.
- `classification.robust_pose_ids` listet alle stabilen Zielposen direkt auf;
  `metastable_pose_ids` enthält die möglichen Zwischenlagen.
- `equivalent_catalog_pose_ids` sind durch praktische Rotationssymmetrie
  ununterscheidbare Darstellungen derselben physischen Pose.
- `stability: robust` kennzeichnet eine beobachtungsgeeignete stabile Zielpose.
- `stability: metastable` ist eine mögliche Zwischenlage; sie sollte nicht ohne
  experimentelle Bestätigung als dauerhafte Zielpose verwendet werden.
- Das Quaternion verwendet die Reihenfolge `[x, y, z, w]` und beschreibt die
  Orientierung vom Bauteil- ins feste Rutschenkoordinatensystem.

## Übergänge

- `type: actuated` kostet einen Luftimpuls; `passive_tip` kostet keinen Impuls.
- `axis`, `axis_vector_chute` und `direction` verwenden die feste Rutsche und die
  Rechte-Hand-Regel. Der Sollwert steht in `commanded_angle_deg`.
- `capture.interval_deg` und `capture.width_deg` beschreiben den geometrisch
  berechneten Einfangbereich. `geometry.geometric_score` ist **keine
  Erfolgswahrscheinlichkeit**.
- `geometry.passive_settling_via_catalog_pose_ids` nennt nicht als Knoten
  dargestellte instabile Kataloglagen, über die das Bauteil nach dem Impuls
  passiv in `to_pose` fällt. Ein leerer Wert kennzeichnet einen direkten
  Übergang.
- `classification.unresolved_metastable_pose_ids` bedeutet nur, dass aus der
  jeweiligen Pose **ohne Aktuierung** kein eindeutiges passives Kippen gefunden
  wurde. Aktuierte ausgehende Kanten können trotzdem vorhanden sein.
- Bei `surface_requirement` und `additional_requirement` müssen die genannten
  Aktuatorbedingungen erfüllt sein.

## Experimentelle Ergänzung

Der Block `experimental` jeder Kante ist zum manuellen oder automatischen
Ausfüllen vorgesehen:

- `trials`: Anzahl der Versuche,
- `successes`: erfolgreiche Übergänge zur angegebenen Zielpose,
- `empirical_success_rate`: `successes / trials` zwischen 0 und 1,
- `difficulty_rating`: frei wählbare gemeinsame Skala,
- `notes`: Beobachtungen, Fehlermodi oder Versuchsbedingungen.

Für die spätere Routenplanung sollte nach ausreichenden Versuchen bevorzugt
`empirical_success_rate` verwendet werden. Bis dahin kann der geometrische
Score als klar gekennzeichneter vorläufiger Ersatzwert dienen.
"""


def save_roadmap_yaml_readme(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_YAML_README_DE, encoding="utf-8")
    return destination


def save_roadmap_graphml(roadmap: PoseRoadmap, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    graph = nx.MultiDiGraph()
    graph.graph["classical_metadata"] = json.dumps(roadmap.classical_metadata)
    for node in roadmap.nodes:
        attributes: dict[str, Any] = {
            "original_catalog_pose_id": node.original_catalog_pose_id,
            "pose_ids": "/".join(str(value) for value in node.pose_ids),
            "kind": node.kind,
            "cad_status": node.cad_status,
            "main_face_min_span_mm": roadmap.main_face_min_span_mm,
            "rocking_barrier_mm": node.rocking_barrier_mm,
            "floor_contact": node.floor_contact_topology,
            "wall_contact": node.wall_contact_topology,
            "main_face_on_floor": node.main_face_on_floor,
            "main_face_on_wall": node.main_face_on_wall,
            "pose_ranking_method": roadmap.pose_ranking_method,
            "robustness_method": roadmap.robustness_method,
        }
        if node.csa_applicable is not None:
            attributes["csa_applicable"] = node.csa_applicable
        attributes["classical_metrics"] = json.dumps(node.classical_metrics)
        for key, value in (
            ("csa_stability_index", node.csa_stability_index),
            ("csa_feasible_fraction", node.csa_feasible_fraction),
            ("csa_feasible_solid_angle_sr", node.csa_feasible_solid_angle_sr),
            ("csa_angular_clearance_deg", node.csa_angular_clearance_deg),
        ):
            if value is not None:
                attributes[key] = value
        graph.add_node(node.node_id, **attributes)
    for edge in roadmap.edges:
        graph.add_edge(
            edge.source,
            edge.target,
            key=edge.edge_id,
            edge_id=edge.edge_id,
            transition_kind=edge.transition_kind,
            actuation=edge.actuation,
            signed_angle_deg=edge.signed_angle_deg,
            capture_width_deg=edge.capture_width_deg,
            geometric_score=edge.geometric_score,
            escape_barrier_mm=(
                edge.escape_barrier_mm if edge.escape_barrier_mm is not None else -1.0
            ),
        )
    nx.write_graphml(graph, destination)
    return destination


def _format_roadmap_node_plot_label(
    node: RoadmapNode, *, show_csa: bool
) -> str:
    """Format the stability metrics shown beneath a roadmap node."""

    label = f"Rocking barrier {node.rocking_barrier_mm:.3f} mm"
    if show_csa:
        label += (
            f"\nCWSA {node.csa_stability_index:.3f}"
            if node.csa_stability_index is not None
            else "\nCWSA n/a"
        )
    return label


_ROADMAP_AXIS_COLORS = {
    "x": "#d62728",
    "y": "#2ca02c",
    "z": "#1f77b4",
}


def _roadmap_transition_axis(edge: RoadmapEdge) -> Literal["x", "y", "z"]:
    """Return the chute axis that best describes a plotted transition."""

    if edge.actuation in {
        "floor_main_neg_x",
        "floor_main_pos_x",
        "wall_main_neg_x",
        "wall_main_pos_x",
    }:
        return "x"
    if edge.actuation == "free_y":
        return "y"
    if edge.actuation == "free_z":
        return "z"

    # Passive rocking axes need not be perfectly aligned with a principal
    # chute axis. Colour them by their dominant component while retaining the
    # dashed passive-transition line style.
    dominant = int(np.argmax(np.abs(np.asarray(edge.axis_chute, dtype=float))))
    return ("x", "y", "z")[dominant]


def _roadmap_layout_order(roadmap: PoseRoadmap) -> list[int]:
    """Find a deterministic low-bandwidth order before placing node rows."""

    interaction = nx.Graph()
    interaction.add_nodes_from(sorted(node.node_id for node in roadmap.nodes))
    for edge in roadmap.edges:
        if interaction.has_edge(edge.source, edge.target):
            interaction[edge.source][edge.target]["weight"] += 1
        else:
            interaction.add_edge(edge.source, edge.target, weight=1)
    if interaction.number_of_nodes() <= 1:
        return list(interaction.nodes)
    return list(nx.utils.reverse_cuthill_mckee_ordering(interaction))


def _roadmap_plot_positions(
    roadmap: PoseRoadmap,
    *,
    target_aspect_ratio: float = 1.5,
    minimize_path_length: bool = True,
) -> tuple[dict[int, np.ndarray], int, int]:
    """Arrange poses in a compact grid matching the common canvas ratio.

    The reverse Cuthill-McKee order keeps highly connected poses near one
    another. Serpentine rows preserve that locality at row boundaries. Robust
    and metastable poses deliberately share rows: brightness carries the
    stability distinction without adding an otherwise unnecessary band gap.
    A deterministic node-swap pass then shortens the transitions while keeping
    every pose on the same orderly grid.
    """

    if target_aspect_ratio <= 0.0:
        raise ValueError("target_aspect_ratio must be positive.")
    layout_order = _roadmap_layout_order(roadmap)
    node_count = max(len(layout_order), 1)
    candidates: list[tuple[float, int, int]] = []
    for candidate_columns in range(1, node_count + 1):
        candidate_rows = math.ceil(node_count / candidate_columns)
        occupied_aspect = candidate_columns / candidate_rows
        aspect_error = abs(math.log(occupied_aspect / target_aspect_ratio))
        unused_fraction = (
            candidate_columns * candidate_rows - node_count
        ) / node_count
        candidates.append(
            (aspect_error + 0.12 * unused_fraction, candidate_columns, candidate_rows)
        )
    _, columns, row_count = min(candidates)
    horizontal_spacing = 2.8
    vertical_spacing = 3.0
    positions: dict[int, np.ndarray] = {}
    for start in range(0, len(layout_order), columns):
        row_index = start // columns
        row_nodes = layout_order[start : start + columns]
        x_values = (
            np.arange(len(row_nodes), dtype=float)
            - (len(row_nodes) - 1.0) / 2.0
        ) * horizontal_spacing
        if row_index % 2:
            x_values = x_values[::-1]
        for node_id, x_value in zip(row_nodes, x_values, strict=True):
            positions[node_id] = np.array(
                [float(x_value), -row_index * vertical_spacing], dtype=float
            )
    if minimize_path_length:
        positions = _minimise_roadmap_path_length(roadmap, positions)
    return positions, columns, row_count


def _roadmap_total_path_length(
    roadmap: PoseRoadmap, positions: dict[int, np.ndarray]
) -> float:
    """Return the summed straight-line length of every transition."""

    return float(
        sum(
            np.linalg.norm(
                np.asarray(positions[edge.target], dtype=float)
                - np.asarray(positions[edge.source], dtype=float)
            )
            for edge in roadmap.edges
        )
    )


def _minimise_roadmap_path_length(
    roadmap: PoseRoadmap, positions: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """Shorten transitions by swapping nodes between fixed orderly slots."""

    result = {
        node_id: np.asarray(position, dtype=float).copy()
        for node_id, position in positions.items()
    }
    node_ids = sorted(result)
    edge_weights: dict[tuple[int, int], int] = {}
    incident_pairs: dict[int, set[tuple[int, int]]] = {
        node_id: set() for node_id in node_ids
    }
    for edge in roadmap.edges:
        if edge.source == edge.target:
            continue
        pair = tuple(sorted((edge.source, edge.target)))
        edge_weights[pair] = edge_weights.get(pair, 0) + 1
        incident_pairs[edge.source].add(pair)
        incident_pairs[edge.target].add(pair)

    def affected_cost(pairs: set[tuple[int, int]]) -> float:
        return float(
            sum(
                edge_weights[pair]
                * np.linalg.norm(result[pair[1]] - result[pair[0]])
                for pair in pairs
            )
        )

    # Best-improvement pair swaps are deterministic and inexpensive for the
    # practical roadmap sizes (currently at most a few dozen nodes).
    for _ in range(max(1, 2 * len(node_ids))):
        best_improvement = 1e-9
        best_pair: tuple[int, int] | None = None
        for left_index, left in enumerate(node_ids):
            for right in node_ids[left_index + 1 :]:
                affected = incident_pairs[left] | incident_pairs[right]
                before = affected_cost(affected)
                result[left], result[right] = result[right], result[left]
                after = affected_cost(affected)
                result[left], result[right] = result[right], result[left]
                improvement = before - after
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_pair = (left, right)
        if best_pair is None:
            break
        left, right = best_pair
        result[left], result[right] = result[right], result[left]
    return result


def _roadmap_edge_curvatures(
    roadmap: PoseRoadmap,
    positions: dict[int, np.ndarray] | None = None,
) -> dict[str, float]:
    """Route long and parallel transitions through visibly separate lanes."""

    grouped: dict[tuple[int, int], list[RoadmapEdge]] = {}
    for edge in roadmap.edges:
        grouped.setdefault((edge.source, edge.target), []).append(edge)
    curvatures: dict[str, float] = {}
    for (source, target), edges in grouped.items():
        if positions is None:
            base_curvature = 0.12
        else:
            grid_displacement = (
                np.asarray(positions[target], dtype=float)
                - np.asarray(positions[source], dtype=float)
            ) / np.array((2.8, 3.0), dtype=float)
            crossed_cells = float(np.linalg.norm(grid_displacement))
            base_magnitude = min(0.48, 0.11 + 0.035 * crossed_cells)
            # Use one sign for both directions of an unordered pair. Reversing
            # the arrow then naturally sends the return path to the other side.
            sign = (
                1.0
                if (min(source, target) + max(source, target)) % 2 == 0
                else -1.0
            )
            base_curvature = sign * base_magnitude
        middle = (len(edges) - 1.0) / 2.0
        for index, edge in enumerate(edges):
            curvature = base_curvature + (index - middle) * 0.085
            curvatures[edge.edge_id] = float(np.clip(curvature, -0.68, 0.68))
    return curvatures


def _format_roadmap_edge_plot_label(edge: RoadmapEdge) -> str:
    """Format one compact signed-angle label for a robust transition."""

    angle = 0.0 if abs(edge.signed_angle_deg) < 0.05 else edge.signed_angle_deg
    direction = "+" if angle >= 0.0 else "−"
    return f"{direction}{abs(angle):.1f}°"


def _roadmap_edge_label_position(
    edge: RoadmapEdge,
    positions: dict[int, np.ndarray],
    curvature: float,
) -> np.ndarray:
    """Return the midpoint of the quadratic curve used to draw an edge."""

    source = np.asarray(positions[edge.source], dtype=float)
    target = np.asarray(positions[edge.target], dtype=float)
    delta = target - source
    control = (source + target) / 2.0 + curvature * np.array(
        (delta[1], -delta[0]), dtype=float
    )
    return 0.25 * source + 0.5 * control + 0.25 * target


def _separate_roadmap_angle_labels(
    figure: matplotlib.figure.Figure,
    annotations: list[Annotation],
) -> None:
    """Move overlapping angle labels by small deterministic point offsets."""

    if not annotations:
        return
    for annotation in annotations:
        annotation.set_position((0.0, 0.0))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    pixels_per_point = figure.dpi / 72.0
    occupied: list[matplotlib.transforms.Bbox] = []
    candidate_offsets = [(0.0, 0.0)]
    for radius in range(8, 129, 8):
        candidate_offsets.extend(
            (
                radius * math.cos(index * math.pi / 4.0),
                radius * math.sin(index * math.pi / 4.0),
            )
            for index in range(8)
        )

    for annotation in annotations:
        bbox_patch = annotation.get_bbox_patch()
        if bbox_patch is None:
            base_bbox = annotation.get_window_extent(renderer)
        else:
            base_bbox = bbox_patch.get_window_extent(renderer)
        for offset in candidate_offsets:
            bbox = base_bbox.translated(
                offset[0] * pixels_per_point,
                offset[1] * pixels_per_point,
            )
            padded_bbox = bbox.expanded(1.12, 1.25)
            if not any(padded_bbox.overlaps(existing) for existing in occupied):
                annotation.set_position(offset)
                occupied.append(padded_bbox)
                break
        else:  # pragma: no cover - more labels than current roadmaps contain
            occupied.append(padded_bbox)
    figure.canvas.draw()


def _roadmap_metric_ranks(
    roadmap: PoseRoadmap,
    metric: Literal["rocking_barrier_mm", "csa_stability_index"],
    *,
    absolute_tolerance: float = 1e-6,
) -> dict[int, int]:
    """Descending dense display ranks with ties appropriate to each metric.

    CWSA ties use the same three-decimal formatting as the plotted values.
    Rocking ties compare each value to its group's highest value, avoiding
    rounding-bin boundaries and chains merging different endpoints.
    Node identifiers and full-precision measurements remain untouched.
    """

    values = sorted(
        ((float(getattr(node, metric)), node.node_id) for node in roadmap.nodes
         if getattr(node, metric) is not None),
        key=lambda item: (-item[0], item[1]),
    )
    if metric == "csa_stability_index":
        displayed_values = {node_id: float(f"{value:.3f}") for value, node_id in values}
        ranks = {
            value: rank
            for rank, value in enumerate(sorted(set(displayed_values.values()), reverse=True))
        }
        return {node_id: ranks[value] for node_id, value in displayed_values.items()}

    result: dict[int, int] = {}
    reference: float | None = None
    rank = -1
    for value, node_id in values:
        if reference is None or reference - value > absolute_tolerance:
            rank += 1
            reference = value
        result[node_id] = rank
    return result


def render_pose_roadmap(
    roadmap: PoseRoadmap,
    output_stem: str | Path,
    *,
    stable_only: bool = False,
    show_csa_ranks: bool = False,
    display_methods: tuple[str, ...] | None = None,
    formats: tuple[str, ...] = ("svg", "png"),
    rank_reference: PoseRoadmap | None = None,
) -> tuple[Path, ...]:
    """Render a full roadmap or only its robust nodes and transitions."""

    if not formats or any(fmt not in {"svg", "png"} for fmt in formats):
        raise ValueError("Roadmap formats must be svg and/or png.")
    if show_csa_ranks and display_methods is None:
        display_methods = ("rocking", "cwsa")
    if display_methods is None and (roadmap.classical_metadata.get("methods") or roadmap.pose_ranking_method == "cwsa"):
        display_methods = tuple(dict.fromkeys(("rocking", *(
            ("cwsa",) if any(n.csa_stability_index is not None for n in roadmap.nodes) else ()
        ), *roadmap.classical_metadata.get("methods", ()))))
    elif show_csa_ranks:
        display_methods = tuple(dict.fromkeys((*display_methods, *roadmap.classical_metadata.get("methods", ()))))
    colored_lines = metric_lines(rank_reference or roadmap, display_methods) if display_methods else {}
    # Rank the complete set before filtering the stable-only view. Comparison
    # labels show shared rocking ranks (CWSA ranks), not unique roadmap IDs.
    rocking_ranks = _roadmap_metric_ranks(roadmap, "rocking_barrier_mm") if show_csa_ranks else {}
    csa_ranks = _roadmap_metric_ranks(roadmap, "csa_stability_index") if show_csa_ranks else {}

    if stable_only:
        stable_node_ids = {
            node.node_id for node in roadmap.nodes if node.kind == "robust"
        }
        roadmap = replace(
            roadmap,
            nodes=tuple(
                node for node in roadmap.nodes if node.node_id in stable_node_ids
            ),
            edges=tuple(
                edge
                for edge in roadmap.edges
                if edge.source in stable_node_ids and edge.target in stable_node_ids
            ),
            unresolved_metastable_node_ids=(),
        )

    stem = Path(output_stem).expanduser().resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    graph = nx.DiGraph()
    graph.add_nodes_from(node.node_id for node in roadmap.nodes)
    graph.add_edges_from((edge.source, edge.target) for edge in roadmap.edges)
    positions, _, _ = _roadmap_plot_positions(roadmap)
    figure, axis = plt.subplots(figsize=(18.0, 12.0 + 1.2 * len(display_methods or ())), facecolor="white")
    axis.set_axis_off()
    x_values = [float(position[0]) for position in positions.values()] or [0.0]
    y_values = [float(position[1]) for position in positions.values()] or [0.0]
    axis.set_xlim(min(x_values) - 2.4, max(x_values) + 2.4)
    axis.set_ylim(min(y_values) - 3.0, max(y_values) + 2.4)
    robust = [node.node_id for node in roadmap.nodes if node.kind == "robust"]
    metastable = [node.node_id for node in roadmap.nodes if node.kind == "metastable"]
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=robust,
        node_size=7200,
        node_color="#2b8cbe",
        edgecolors="#084081",
        linewidths=2.2,
        ax=axis,
    )
    meta_collection = nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=metastable,
        node_size=4600,
        node_color="#e5e7eb",
        edgecolors="#6b7280",
        linewidths=1.5,
        ax=axis,
    )
    meta_collection.set_linestyle("--")
    nodes_by_id = {node.node_id: node for node in roadmap.nodes}
    curvatures = _roadmap_edge_curvatures(roadmap, positions)
    # Draw faint metastable-linked transitions first so robust-only paths stay
    # visible on top even in roadmaps with hundreds of edges.
    edges_for_drawing = sorted(
        roadmap.edges,
        key=lambda edge: (
            nodes_by_id[edge.source].kind == "robust"
            and nodes_by_id[edge.target].kind == "robust"
        ),
    )
    for edge in edges_for_drawing:
        robust_only = (
            nodes_by_id[edge.source].kind == "robust"
            and nodes_by_id[edge.target].kind == "robust"
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=[(edge.source, edge.target)],
            edge_color=_ROADMAP_AXIS_COLORS[_roadmap_transition_axis(edge)],
            style="solid" if robust_only else "dashed",
            width=2.45 if robust_only else 0.8,
            alpha=0.88 if robust_only else 0.22,
            arrows=True,
            arrowsize=15 if robust_only else 9,
            connectionstyle=f"arc3,rad={curvatures[edge.edge_id]:.4f}",
            min_source_margin=21,
            min_target_margin=21,
            ax=axis,
        )
    # Keep angle labels only on the bright robust network. Metastable-linked
    # paths remain available as quiet routing context without text congestion.
    angle_annotations: list[Annotation] = []
    for edge in roadmap.edges:
        robust_only = (
            nodes_by_id[edge.source].kind == "robust"
            and nodes_by_id[edge.target].kind == "robust"
        )
        if not robust_only:
            continue
        label_color = _ROADMAP_AXIS_COLORS[_roadmap_transition_axis(edge)]
        angle_annotations.append(
            axis.annotate(
                _format_roadmap_edge_plot_label(edge),
                _roadmap_edge_label_position(
                    edge, positions, curvatures[edge.edge_id]
                ),
                xytext=(0.0, 0.0),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=6.2,
                weight="bold",
                color=label_color,
                alpha=0.96,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 0.25,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": label_color,
                    "alpha": 0.42,
                    "linewidth": 0.5,
                    "shrinkA": 2.0,
                    "shrinkB": 1.0,
                },
                zorder=7,
            )
        )
    thumbnails = create_pose_thumbnails(
        roadmap.source,
        (node.original_catalog_pose_id for node in roadmap.nodes),
    )
    for node in roadmap.nodes:
        robust_node = node.kind == "robust"
        image = OffsetImage(
            thumbnails[node.original_catalog_pose_id],
            zoom=0.48 if robust_node else 0.37,
        )
        image.image.axes = axis
        box = AnnotationBbox(
            image,
            positions[node.node_id],
            frameon=True,
            pad=0.06,
            bboxprops={
                "facecolor": "white",
                "edgecolor": "#084081" if robust_node else "#6b7280",
                "linewidth": 2.4 if robust_node else 1.5,
                "linestyle": "solid" if robust_node else "dashed",
            },
            zorder=8,
        )
        axis.add_artist(box)
        show_csa = (
            roadmap.pose_ranking_method == "csa"
            or roadmap.robustness_method == "csa"
            or show_csa_ranks
        )
        axis.annotate(
            (
                f"{rocking_ranks[node.node_id]} ({csa_ranks.get(node.node_id, 'n/a')})"
                if show_csa_ranks and not display_methods else f"Pose {node.node_id}"
            ),
            positions[node.node_id],
            xytext=(0, 56 if robust_node else 44),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
            color="#111827",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.8,
            },
            zorder=9,
        )
        if display_methods:
            for line_index, (text, color) in enumerate(colored_lines[node.node_id]):
                axis.annotate(
                    text, positions[node.node_id], xytext=(0, -56 - 11 * line_index),
                    textcoords="offset points", ha="center", va="top", fontsize=6.7,
                    color=color, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 0.7},
                    zorder=9,
                )
            continue
        metric_label = _format_roadmap_node_plot_label(
            node,
            show_csa=show_csa,
        )
        axis.annotate(
            metric_label,
            positions[node.node_id],
            xytext=(0, -56 if robust_node else -44),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
            zorder=9,
        )
    _separate_roadmap_angle_labels(figure, angle_annotations)
    if stable_only:
        title = (
            f"{Path(roadmap.source).stem}: stable pose roadmap — "
            f"{len(robust)} stable poses, {len(roadmap.edges)} stable transitions"
        )
    else:
        title = (
            f"{Path(roadmap.source).stem}: pose roadmap — "
            f"{sum(node.kind == 'robust' for node in roadmap.nodes)} robust, "
            f"{sum(node.kind == 'metastable' for node in roadmap.nodes)} metastable"
        )
    axis.set_title(title, fontsize=16, pad=18)
    if display_methods:
        figure.text(
            0.5, 0.95,
            "Coloured metric ranks: 0 = highest; ties at displayed precision. Pose IDs remain unique.\n"
            f"Selection: {roadmap.robustness_method}; X={roadmap.alpha_deg:g}°, Y={roadmap.beta_deg:g}°. "
            "CSA/CRSA are experimental chute adaptations; N/A = unsupported.",
            ha="center", fontsize=9,
        )
    elif show_csa_ranks:
        figure.text(
            0.5,
            0.95,
            "Labels: rocking rank (CWSA rank); 0 = highest. "
            "Ties: rocking within 1e-6 mm, CWSA at 3 decimals.\n"
            f"colours: {roadmap.robustness_method} classification. "
            f"Chute X={roadmap.alpha_deg:g}°, Y={roadmap.beta_deg:g}°",
            ha="center",
            fontsize=10,
        )
    legend_items = [
        Line2D([0], [0], color=_ROADMAP_AXIS_COLORS["x"], lw=2.4, label="X rotation"),
        Line2D([0], [0], color=_ROADMAP_AXIS_COLORS["y"], lw=2.4, label="Y rotation"),
        Line2D([0], [0], color=_ROADMAP_AXIS_COLORS["z"], lw=2.4, label="Z rotation"),
        Line2D(
            [0],
            [0],
            color="#111827",
            lw=2.4,
            ls="solid",
            label="stable" if stable_only else "robust",
        ),
    ]
    if not stable_only:
        legend_items.append(
            Line2D(
                [0],
                [0],
                color="#6b7280",
                lw=0.9,
                ls="dashed",
                alpha=0.45,
                label="metastable",
            )
        )
    legend = axis.legend(
        handles=legend_items,
        loc="upper left",
        fontsize=7,
        ncol=2,
        frameon=True,
        framealpha=0.96,
    )
    legend.set_zorder(20)
    if roadmap.geometry_status == "provisional":
        provisional_note = (
            "PROVISIONAL — known incorrect CAD; recalculate physical transitions"
            if Path(roadmap.source).stem.casefold() == "df1a"
            else "PROVISIONAL — validate CAD, symmetry and poses experimentally"
        )
        figure.text(
            0.5,
            0.02,
            provisional_note,
            ha="center",
            color="#b91c1c",
            fontsize=11,
            weight="bold",
        )
    elif roadmap.unresolved_metastable_node_ids:
        figure.text(
            0.99,
            0.02,
            ("Unresolved passive successor edge for roadmap IDs: "
             if show_csa_ranks else "Unresolved passive successor edge for: ")
            + "/".join(
                str(value) for value in roadmap.unresolved_metastable_node_ids
            ),
            ha="right",
            color="#92400e",
            fontsize=9,
            weight="bold",
        )
    figure.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.88)
    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    paths = tuple(stem.with_suffix('.' + fmt) for fmt in formats)
    for path in paths:
        figure.savefig(path, dpi=220)
    plt.close(figure)
    return paths


def export_pose_roadmap(
    roadmap: PoseRoadmap, output_dir: str | Path, *, comparison_plots: bool = False
) -> tuple[Path, ...]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = Path(roadmap.source).stem
    json_path = save_roadmap_json(roadmap, destination / f"{stem}_roadmap.json")
    yaml_path = save_roadmap_yaml(roadmap, destination / f"{stem}_roadmap.yaml")
    yaml_readme_path = save_roadmap_yaml_readme(
        destination / "ROADMAP_YAML_README.md"
    )
    graphml_path = save_roadmap_graphml(roadmap, destination / f"{stem}_roadmap.graphml")
    svg_path, png_path = render_pose_roadmap(
        roadmap, destination / f"{stem}_roadmap"
    )
    stable_svg_path, stable_png_path = render_pose_roadmap(
        roadmap,
        destination / f"{stem}_roadmap_stable",
        stable_only=True,
    )
    paths = (
        json_path,
        yaml_path,
        yaml_readme_path,
        graphml_path,
        svg_path,
        png_path,
        stable_svg_path,
        stable_png_path,
    )
    if comparison_plots:
        for stable_only in (False, True):
            suffix = "_stable" if stable_only else ""
            paths += render_pose_roadmap(
                roadmap,
                destination / f"{stem}_roadmap_comparison{suffix}",
                stable_only=stable_only,
                show_csa_ranks=True,
            )
    return paths
