"""Direct observation matching; source IDs are never geometric catalog indices."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from .pose_matching import PoseResolver, _quaternions, _sha256


class ObservedPoseResolver(PoseResolver):
    def __init__(self, mesh_path: Path, catalog_path: Path | None = None, *, data: dict | None = None):
        self.data = data if data is not None else json.loads(Path(catalog_path).read_text(encoding="utf-8-sig"))
        if self.data.get("schema_version", 1) != 1 or not isinstance(self.data.get("poses"), list):
            raise ValueError("Expected a version 1 observed poses catalog.")
        expected = self.data.get("geometry", {}).get("stl_sha256")
        if expected and expected != _sha256(mesh_path):
            raise ValueError("Observed catalog STL SHA-256 differs from the selected mesh.")
        self.mesh_path = self.original_mesh_path = Path(mesh_path)
        self.roadmap_path = None
        self.cache_hit = False
        descriptor = self.data.get("recognition", {})
        if descriptor.get("quaternion_order", "xyzw") != "xyzw" or descriptor.get("rotation", "part_to_chute") != "part_to_chute":
            raise ValueError("Observed catalog requires xyzw part-to-chute rotations.")
        self.match_tolerance_deg = self.cluster_tolerance_deg = float(descriptor.get("tolerance_deg", 5))
        self.ambiguity_margin_deg = float(descriptor.get("ambiguity_margin_deg", 1))
        if not 0 < self.match_tolerance_deg <= 180 or not 0 <= self.ambiguity_margin_deg <= 180:
            raise ValueError("Invalid observation recognition tolerances.")
        self._symmetry_quats = _quaternions(descriptor.get("symmetry_quaternions_xyzw", [[0, 0, 0, 1]]))
        if not len(self._symmetry_quats):
            raise ValueError("Observed symmetry requires at least identity.")
        axis = descriptor.get("continuous_axis_part")
        self._continuous_axis = None if axis is None else np.asarray(axis, dtype=float)
        if axis is not None:
            if self._continuous_axis.shape != (3,) or not np.all(np.isfinite(self._continuous_axis)) or np.linalg.norm(self._continuous_axis) < 1e-12:
                raise ValueError("Invalid continuous symmetry axis.")
            self._continuous_axis = self._continuous_axis / np.linalg.norm(self._continuous_axis)
        self.symmetry_symbol = descriptor.get("symmetry_symbol", "C1")
        self.symmetry_available = bool(descriptor.get("symmetry_available", True))
        ids, quats, seen = [], [], set()
        for pose in self.data["poses"]:
            key = str(pose["id"])
            if not key or key in seen:
                raise ValueError("Observed pose IDs must be unique and nonempty.")
            seen.add(key)
            references = _quaternions(pose.get("reference_quaternions_xyzw", [pose["quaternion_xyzw"]]))
            if not len(references):
                raise ValueError("An observed pose needs a reference orientation.")
            if self._continuous_axis is None:
                # Match all measured references, including their finite symmetry variants.
                references = (Rotation.from_quat(references).as_matrix()[:, None]
                              @ Rotation.from_quat(self._symmetry_quats).as_matrix()[None]).reshape(-1, 3, 3)
                references = Rotation.from_matrix(references).as_quat()
            quats.extend(references)
            ids.extend([key] * len(references))
        self.known_pose_ids = tuple(sorted(seen))
        self._catalogue_quats = np.asarray(quats, dtype=float).reshape(-1, 4)
        self._catalogue_node_ids = np.asarray(ids, dtype=str)
        self._catalogue_axes = (np.empty((0, 3)) if self._continuous_axis is None else
                                Rotation.from_quat(self._catalogue_quats).apply(self._continuous_axis))

    def grouped(self, mapping: dict[str, str]) -> "ObservedPoseResolver":
        import copy
        result = copy.copy(self)
        result._catalogue_node_ids = np.asarray([mapping[str(key)] for key in self._catalogue_node_ids])
        result.known_pose_ids = tuple(sorted(set(mapping.values())))
        return result
