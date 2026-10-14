"""Paper-reference CSA/CRSA and an explicitly experimental two-plane extension.

CSA: Ngoi et al. 1995, DOI 10.1080/00207549508904822, section 3.
CRSA: Ngoi et al. 1996, DOI 10.1007/BF01178966, equations 4--8.
See docs/CSA_CRSA.md for the extension, limitations and normalization domain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.optimize import minimize_scalar

from .contacts import PoseCatalog
from .frame import ChuteFrame
from .geometry import load_solid_mesh

MODEL_VERSION = "reseated-patch-normal-load-v1"
METHODS = ("standard_csa", "crsa")


def solid_angle(polygon: np.ndarray, apex: np.ndarray) -> float:
    """Steradians of an ordered convex planar polygon (triangle fan)."""
    vectors = np.asarray(polygon, dtype=float) - apex
    if len(vectors) < 3:
        return 0.0
    if np.any(np.linalg.norm(vectors, axis=1) < 1e-12):
        raise ValueError("Solid-angle apex touches the polygon.")
    total = 0.0
    for i in range(1, len(vectors) - 1):
        a, b, c = vectors[[0, i, i + 1]]
        la, lb, lc = np.linalg.norm([a, b, c], axis=1)
        denominator = la * lb * lc + a @ b * lc + b @ c * la + c @ a * lb
        total += 2.0 * math.atan2(abs(float(a @ np.cross(b, c))), float(denominator))
    return total


def planar_reference(polygon: np.ndarray, com: np.ndarray, normal: np.ndarray) -> dict:
    """One horizontal resting aspect; no chute or dynamic modifications.

    Returns raw weights, not probabilities. Normalize over all physical aspects;
    sum equivalent-aspect weights only once per actual occurrence, not per mesh
    triangle or duplicated catalogue representation.
    """
    polygon, com, normal = map(lambda a: np.asarray(a, dtype=float), (polygon, com, normal))
    normal = normal / np.linalg.norm(normal)
    h = float((com - polygon[0]) @ normal)
    if h <= 0 or len(polygon) < 3:
        raise ValueError("Reference aspect requires a polygon and positive COM height.")
    if not np.allclose((polygon - polygon[0]) @ normal, 0, atol=1e-8):
        raise ValueError("Reference aspect must be planar.")
    omega = solid_angle(polygon, com)
    critical = []
    for p, q in zip(polygon, np.roll(polygon, -1, axis=0)):
        axis = q - p
        length = np.linalg.norm(axis)
        if length < 1e-12:
            raise ValueError("Repeated boundary vertex.")
        axis /= length
        r = com - p
        radius = float(np.linalg.norm(r - (r @ axis) * axis))
        critical.append(solid_angle(polygon, com + (radius - h) * normal))
    return {"omega_sr": omega, "height_mm": h, "critical_angles_sr": critical,
            "standard_csa": omega / h,
            "crsa": (omega - float(np.mean(critical))) / h}


def normalize_weights(weights: list[float], multiplicities: list[int] | None = None) -> list[float]:
    values = np.asarray(weights, dtype=float)
    if multiplicities is not None:
        counts = np.asarray(multiplicities)
        if counts.shape != values.shape or np.any(counts < 1) or np.any(counts != np.floor(counts)):
            raise ValueError("Multiplicities must be positive integers, one per weight.")
        values = values * counts
    if np.any(~np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("Cannot normalize nonfinite, negative or all-zero weights.")
    return (values / values.sum()).tolist()


def _polygon(points: np.ndarray, dimension: int, tolerance: float) -> np.ndarray:
    axes = [i for i in range(3) if i != dimension]
    _, indices = np.unique(np.round(points[:, axes] / tolerance), axis=0, return_index=True)
    points = points[np.sort(indices)]
    if len(points) < 3:
        return np.empty((0, 3))
    try:
        return points[ConvexHull(points[:, axes]).vertices]
    except QhullError:
        return np.empty((0, 3))


def _arc_minimum(a: np.ndarray, b: np.ndarray, c: np.ndarray, end: float) -> np.ndarray:
    """Exact minimum of a*cos(t)+b*sin(t)+c on [0,end], elementwise."""
    result = np.minimum(a + c, a * math.cos(end) + b * math.sin(end) + c)
    stationary = np.mod(np.arctan2(b, a) + math.pi, 2 * math.pi)
    inside = stationary <= end
    return np.where(inside, np.minimum(result, c - np.hypot(a, b)), result)


def _edge_lift(vertices: np.ndarray, com: np.ndarray, p: np.ndarray, q: np.ndarray,
               up: np.ndarray, tolerance: float) -> tuple[float | None, str]:
    """Fixed-pivot escape saddle, enforcing y,z>=0 over the entire rotation.

    Both signs are considered. A contact switch before the saddle is deliberately
    not continued by a guessed path: report unsupported rather than a zero score.
    """
    axis = q - p
    axis /= np.linalg.norm(axis)
    r = com - p
    perpendicular = r - (r @ axis) * axis
    a_com = float(up @ perpendicular)
    b_com = float(up @ np.cross(axis, r))
    v = vertices - p
    parallel = (v @ axis)[:, None] * axis
    cosine = v - parallel
    sine = np.cross(axis, v)
    constant = p + parallel
    saw_switch = False
    for sign in (1, -1):
        # An admissible onset must move away from, not through, either plane.
        onset = _arc_minimum(cosine[:, 1:], sign * sine[:, 1:], constant[:, 1:], 1e-4)
        if np.min(onset) < -min(tolerance, 1e-7):
            continue
        b = sign * b_com
        if b < -1e-9:
            return 0.0, "downhill_escape"
        peak = math.atan2(b, a_com)
        if peak <= 1e-10:
            return 0.0, "marginal_escape"
        minimum = _arc_minimum(cosine[:, 1:], sign * sine[:, 1:], constant[:, 1:], peak)
        if np.min(minimum) < -tolerance:
            saw_switch = True
            continue
        return max(0.0, math.hypot(a_com, b) - a_com), "ok"
    return None, "contact_switch_before_saddle" if saw_switch else "blocked_by_other_plane"


def reseated_edge_lift(vertices: np.ndarray, com: np.ndarray, p: np.ndarray, q: np.ndarray,
                       normal: np.ndarray, interior: np.ndarray, up: np.ndarray,
                       step_deg: float = 0.5) -> tuple[float | None, str]:
    """First seated gravitational saddle in the aspect's outward edge direction.

    Rotations about COM are translated to touch both support planes at every
    angle (same convention as rocking.py), permitting frictionless contact
    changes. Longitudinal COM position is held fixed; no longitudinal holding
    force is included in the stability model. Scan a full revolution, then
    refine the first sampled peak; this is a sampled path, not a global saddle
    proof. Narrow unsampled extrema are a documented numerical limitation.
    """
    axis = q - p
    axis /= np.linalg.norm(axis)
    if float(np.cross(axis, interior - p) @ normal) < 0:
        axis = -axis
    points = vertices - com
    parallel = (points @ axis)[:, None] * axis
    perpendicular = points - parallel
    cross = np.cross(axis, points)
    angles = np.linspace(0, 2 * math.pi, int(math.ceil(360 / step_deg)) + 1)

    def height(angle):
        rotated = parallel + math.cos(angle) * perpendicular + math.sin(angle) * cross
        return float(-up[1] * rotated[:, 1].min() - up[2] * rotated[:, 2].min())

    rotated = (parallel[None] + np.cos(angles)[:, None, None] * perpendicular[None]
               + np.sin(angles)[:, None, None] * cross[None])
    heights = -up[1] * rotated[:, :, 1].min(axis=1) - up[2] * rotated[:, :, 2].min(axis=1)
    tolerance = max(1e-10, float(np.ptp(vertices, axis=0).max()) * 1e-10)
    descending = np.flatnonzero(np.diff(heights) < -tolerance)
    if not len(descending):
        return (0.0, "neutral_path") if np.ptp(heights) <= tolerance else (None, "no_resolved_saddle")
    index = int(descending[0])
    if index == 0:
        return 0.0, "downhill_or_subsample_escape"
    optimum = minimize_scalar(lambda angle: -height(angle),
                              bounds=(angles[index - 1], angles[index + 1]), method="bounded",
                              options={"xatol": 1e-10})
    return max(0.0, max(heights[index], -optimum.fun) - heights[0]), "reseated_saddle"


@dataclass(frozen=True)
class ChuteScores:
    pose_id: int
    standard_csa: float | None
    crsa: float | None
    csa_reason: str
    crsa_reason: str
    patches: tuple[dict, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_chute_classical(mesh_path: str | Path, catalog: PoseCatalog, pose_ids,
                            *, alpha_deg=45.0, beta_deg=0.0,
                            methods=METHODS) -> dict[int, ChuteScores]:
    """Separate patch model; gravity-normal-load weighted Omega/h.

    CRSA uses gravitational saddle lift to raise the virtual apex along the
    patch normal. This exactly recovers the reference construction for a lone
    horizontal plane, but is an experimental chute score, NOT a validated drop
    probability. Loaded edge-only patches and continuous rolling are N/A.
    """
    mesh = load_solid_mesh(mesh_path)
    center = np.asarray(mesh.center_mass)
    centered = np.asarray(mesh.convex_hull.vertices) - center
    gravity = ChuteFrame(alpha_deg, beta_deg).gravity_chute()
    up = -gravity / np.linalg.norm(gravity)
    loads = np.maximum(up[[2, 1]], 0)
    results = {}
    wanted = set(pose_ids)
    for pose in catalog.poses:
        if pose.pose_id not in wanted:
            continue
        rotation = np.asarray(pose.rotation_chute_from_part)
        points = (rotation @ centered.T).T
        shift = np.array([0, -points[:, 1].min(), -points[:, 2].min()])
        points += shift
        com = shift
        patches = []
        csa_terms, crsa_terms = [], []
        csa_reason = crsa_reason = "ok"
        for dimension, indices, load in zip(
            (2, 1), (pose.floor_contact_vertex_indices, pose.wall_contact_vertex_indices), loads
        ):
            if load <= 1e-12:
                continue
            polygon = _polygon(points[list(indices)], dimension, catalog.contact_tolerance_mm)
            normal = np.eye(3)[dimension]
            h = float(com[dimension])
            info = {"plane": "floor" if dimension == 2 else "wall", "height_mm": h,
                    "normal_load_fraction": float(load / max(loads.sum(), 1e-12))}
            if len(polygon) < 3 or h <= 1e-10:
                csa_reason = crsa_reason = "loaded_edge_or_degenerate_patch"
                info["reason"] = csa_reason
                patches.append(info)
                continue
            omega = solid_angle(polygon, com)
            fraction = info["normal_load_fraction"]
            # Weight by the gravity-normal component again to convert normal
            # height into distance along gravity: h_g = h / (up . normal).
            height_gravity = h / load
            csa_terms.append(fraction * omega / height_gravity)
            info.update(omega_sr=omega, height_along_gravity_mm=float(height_gravity))
            deltas, edges = [], []
            if "crsa" in methods:
                for p, q in zip(polygon, np.roll(polygon, -1, axis=0)):
                    lift, reason = reseated_edge_lift(points, com, p, q, normal, polygon.mean(axis=0), up)
                    edge = {"reason": reason, "lift_mm": lift}
                    if lift is not None:
                        apex = com + normal * (lift / load)
                        critical = solid_angle(polygon, apex)
                        deltas.append(max(0.0, omega - critical))
                        edge["critical_solid_angle_sr"] = critical
                    else:
                        crsa_reason = reason
                    edges.append(edge)
                if deltas:
                    crsa_terms.append(fraction * float(np.mean(deltas)) / height_gravity)
                else:
                    crsa_reason = "no_modelled_escape_edges"
                info["edges"] = edges
            patches.append(info)
        if not patches:
            csa_reason = crsa_reason = "gravity_does_not_load_support"
        if catalog.continuous_symmetry_axis_part is not None:
            csa_reason = crsa_reason = "continuous_rolling_not_modelled"
        results[pose.pose_id] = ChuteScores(
            pose.pose_id,
            float(sum(csa_terms)) if csa_reason == "ok" else None,
            float(sum(crsa_terms)) if crsa_reason == "ok" and "crsa" in methods else None,
            csa_reason, crsa_reason if "crsa" in methods else "not_requested", tuple(patches),
        )
    return results
