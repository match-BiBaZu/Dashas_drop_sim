"""Directed seated-energy basins for observed orientations, in equivalent mm."""
from __future__ import annotations

import math
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from .physics import _chute_rotation

MODEL_VERSION = "observed-seated-geodesic-basins-v1"


def relative_path(first, second, descriptor: dict) -> np.ndarray:
    a, b = Rotation.from_quat(first).as_matrix(), Rotation.from_quat(second).as_matrix()
    axis = descriptor.get("continuous_axis_part")
    if axis is not None:
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        u, v = a @ axis, b @ axis
        angle = math.acos(float(np.clip(u @ v, -1, 1)))
        cross = np.cross(u, v)
        if np.linalg.norm(cross) < 1e-10:
            if angle < 1e-8:
                return np.zeros(3)
            cross = np.cross(u, np.eye(3)[np.argmin(np.abs(u))])
        return cross / np.linalg.norm(cross) * angle
    symmetries = Rotation.from_quat(descriptor.get("symmetry_quaternions_xyzw", [[0, 0, 0, 1]])).as_matrix()
    candidates = [Rotation.from_matrix((b @ s) @ a.T).as_rotvec() for s in symmetries]
    return min(candidates, key=lambda rv: (float(np.linalg.norm(rv)), tuple(rv)))


def basin_memberships(keys, paths, *, barrier_mm: float, max_angle_deg: float):
    """SCC basins plus unique absorption, with a complete-group diameter guard."""
    keys = sorted(keys)
    adjacency = {key: set() for key in keys}
    angles = {}
    for path in paths:
        a, b = path["source"], path["target"]
        angles[a, b] = angles[b, a] = path["angle_deg"]
        if path["angle_deg"] <= max_angle_deg + 1e-9:
            if path["forward_barrier_mm"] <= barrier_mm + 1e-9:
                adjacency[a].add(b)
            if path["reverse_barrier_mm"] <= barrier_mm + 1e-9:
                adjacency[b].add(a)
    def reachable(start):
        found, pending = {start}, [start]
        while pending:
            for target in adjacency[pending.pop()] - found:
                found.add(target)
                pending.append(target)
        return found
    reach = {key: reachable(key) for key in keys}
    components, consumed = [], set()
    for key in keys:
        if key not in consumed:
            component = {other for other in reach[key] if key in reach[other]}
            components.append(component)
            consumed.update(component)
    owner = {key: i for i, component in enumerate(components) for key in component}
    exits = {i: {owner[b] for a in component for b in adjacency[a] if owner[b] != i}
             for i, component in enumerate(components)}
    def diameter(members):
        return max((angles.get((a, b), math.inf) for a in members for b in members if a != b), default=0)
    oversized = {i for i, members in enumerate(components) if diameter(members) > max_angle_deg + 1e-9}
    sinks = {i for i in exits if not exits[i] and i not in oversized}
    groups = {i: set(components[i]) for i in sinks}
    separate, review = [], []
    for i, members in enumerate(components):
        if i in oversized:
            separate.extend({key} for key in sorted(members))
            review.append({"members": sorted(members), "reason": "component_diameter_exceeds_guard"})
        elif i not in sinks:
            reachable_components = {owner[key] for a in members for key in reach[a]}
            destinations = reachable_components & sinks
            if len(destinations) == 1 and not (reachable_components & oversized):
                destination = next(iter(destinations))
                combined = groups[destination] | members
                if diameter(combined) <= max_angle_deg + 1e-9:
                    groups[destination] = combined
                    continue
            separate.append(members)
            review.append({"members": sorted(members), "reason": "ambiguous_destination_or_diameter"})
    memberships = sorted((sorted(m) for m in [*groups.values(), *separate]), key=lambda m: tuple(m))
    return memberships, review


def analyze_groups(mesh_path, members: list[dict], descriptor: dict, *, alpha_deg=45,
                   beta_deg=0, barrier_mm=.5, max_angle_deg=60, step_deg=.1):
    if not all(math.isfinite(x) for x in (barrier_mm, max_angle_deg, step_deg)) or barrier_mm < 0 or not 0 < max_angle_deg <= 180 or step_deg <= 0:
        raise ValueError("Invalid grouping thresholds or angular sampling step.")
    mesh = trimesh.load_mesh(mesh_path, process=True)
    if not mesh.is_watertight or not mesh.is_volume:
        raise ValueError("Grouping requires a closed solid mesh in mm.")
    vertices = np.asarray(mesh.convex_hull.vertices) - np.asarray(mesh.center_mass)
    gravity = _chute_rotation(alpha_deg, beta_deg).inv().apply([0, 0, -9.81])
    paths = []
    ordered = sorted(members, key=lambda m: m["id"])
    for i, first in enumerate(ordered):
        rotation = Rotation.from_quat(first["quaternion_xyzw"]).as_matrix()
        for second in ordered[i + 1:]:
            rv = relative_path(first["quaternion_xyzw"], second["quaternion_xyzw"], descriptor)
            angle = float(np.degrees(np.linalg.norm(rv)))
            path = {"source": first["id"], "target": second["id"], "angle_deg": angle,
                    "forward_barrier_mm": math.inf, "reverse_barrier_mm": math.inf}
            if angle <= max_angle_deg + 1e-9:
                fractions = np.linspace(0, 1, max(2, math.ceil(angle / step_deg) + 1))
                rotations = Rotation.from_rotvec(fractions[:, None] * rv).as_matrix() @ rotation
                points = np.einsum("sij,vj->svi", rotations, vertices)
                heights = (gravity[1] * points[:, :, 1].min(axis=1)
                           + gravity[2] * points[:, :, 2].min(axis=1)) / np.linalg.norm(gravity)
                path.update(forward_barrier_mm=max(0., float(heights.max() - heights[0])),
                            reverse_barrier_mm=max(0., float(heights.max() - heights[-1])))
            paths.append(path)
    groups, review = basin_memberships([m["id"] for m in ordered], paths,
                                      barrier_mm=barrier_mm, max_angle_deg=max_angle_deg)
    for path in paths:
        for name in ("forward_barrier_mm", "reverse_barrier_mm"):
            if not math.isfinite(path[name]):
                path[name] = None
    return {"model": MODEL_VERSION, "barrier_mm": barrier_mm, "max_angle_deg": max_angle_deg,
            "sampling_step_deg": step_deg, "paths": paths, "groups": groups, "review": review,
            "interpretation": "Sampled seated shortest-rotation paths; barriers are geometric estimates, not minimum-action proofs."}


def choose_representative(members, scores: dict, frequencies: dict, threshold_retention: dict):
    if any(key not in scores or scores[key] is None for key in members):
        return None
    return min(members, key=lambda key: (-scores[key], -threshold_retention[key], -frequencies[key], key))
