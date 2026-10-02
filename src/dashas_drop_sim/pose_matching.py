"""Compare simulated part orientations with the independent chute pose roadmap.

All quaternions are ``(x, y, z, w)`` rotations from part coordinates into the
right-handed chute frame (x downhill, y away from the wall, z away from the
floor). A catalogue orientation is a geometric candidate, not evidence that a
drop has settled. The runner must call :meth:`resolve` only for settled trials.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from importlib import metadata
import json
import math
from pathlib import Path
from typing import Literal, Sequence
from uuid import uuid4

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.spatial.transform import Rotation
import yaml


MatchStatus = Literal["matched", "unmatched", "ambiguous"]


@dataclass(frozen=True, slots=True)
class PoseMatch:
    status: MatchStatus
    pose_id: int | None
    distance_deg: float | None


def _quaternions(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    quats = np.asarray(values, dtype=float)
    if quats.size == 0:
        return np.empty((0, 4), dtype=float)
    if quats.ndim != 2 or quats.shape[1] != 4 or not np.all(np.isfinite(quats)):
        raise ValueError("Expected finite xyzw quaternions with shape (n, 4).")
    norms = np.linalg.norm(quats, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("A zero quaternion cannot describe an orientation.")
    return quats / norms[:, None]


def _quat(value: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (4,):
        raise ValueError("Expected one xyzw quaternion with shape (4,).")
    quats = _quaternions(array[None, :])
    return quats[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _roadmap(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as source:
        data = json.load(source) if path.suffix.lower() == ".json" else yaml.safe_load(source)
    if not isinstance(data, dict):
        raise ValueError("Roadmap must contain an object at its root.")
    if "poses" in data and isinstance(data["poses"], list):
        return data
    if "nodes" in data and isinstance(data["nodes"], list):
        return data
    raise ValueError("Expected a chute pose YAML handover or JSON roadmap.")


def _roadmap_source(data: dict, path: Path) -> Path | None:
    raw = data.get("part", {}).get("mesh_source") if isinstance(data.get("part"), dict) else data.get("source")
    if not raw:
        return None
    source = Path(str(raw)).expanduser()
    return source if source.is_absolute() else path.parent / source


def _node_catalogue_ids(data: dict) -> dict[int, tuple[int, ...]]:
    mapping: dict[int, tuple[int, ...]] = {}
    owned: dict[int, int] = {}
    for node in data.get("poses", data.get("nodes", [])):
        if not isinstance(node, dict):
            raise ValueError("Every roadmap pose must be an object.")
        node_id = int(node.get("id", node.get("node_id")))
        raw_ids = node.get("equivalent_catalog_pose_ids", node.get("pose_ids"))
        if raw_ids is None:
            raw_ids = [node.get("original_catalog_pose_id")]
        if not isinstance(raw_ids, (list, tuple)) or not raw_ids:
            raise ValueError(f"Roadmap pose {node_id} has no catalogue IDs.")
        ids = tuple(sorted({int(value) for value in raw_ids}))
        if node_id in mapping:
            raise ValueError(f"Duplicate roadmap pose ID {node_id}.")
        for catalogue_id in ids:
            if catalogue_id in owned and owned[catalogue_id] != node_id:
                raise ValueError(f"Catalogue pose {catalogue_id} belongs to two roadmap nodes.")
            owned[catalogue_id] = node_id
        mapping[node_id] = ids
    if not mapping:
        raise ValueError("The roadmap contains no poses.")
    return mapping


def _canonical_quat(quat: np.ndarray) -> tuple[float, ...]:
    signed = quat.copy()
    for value in signed[::-1]:
        if abs(float(value)) > 1e-12:
            if value < 0:
                signed *= -1
            break
    return tuple(float(x) for x in np.round(signed, 12))


class PoseResolver:
    """Map settled orientations to roadmap IDs and group unknown orientations.

    A mapped pose needs a nearest roadmap node within ``match_tolerance_deg``
    (default 5°). If two nodes are within ``ambiguity_margin_deg`` (default 1°)
    of one another, the result is ambiguous. These angular limits are explicit
    classification assumptions; they are not stability thresholds.

    ``mesh_path`` is a validated STL in the same part frame as the simulation.
    It may be the generated STL for a STEP input. ``original_mesh_path`` is the
    user-selected file, used to verify an STL roadmap against its source bytes.
    """

    def __init__(
        self,
        mesh_path: Path,
        roadmap_path: Path | None,
        *,
        original_mesh_path: Path | None = None,
        cache_dir: Path | None = None,
        match_tolerance_deg: float = 5.0,
        ambiguity_margin_deg: float = 1.0,
        cluster_tolerance_deg: float = 5.0,
    ) -> None:
        for name, value, allow_zero in (
            ("match_tolerance_deg", match_tolerance_deg, False),
            ("ambiguity_margin_deg", ambiguity_margin_deg, True),
            ("cluster_tolerance_deg", cluster_tolerance_deg, False),
        ):
            if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero) or value > 180:
                raise ValueError(f"{name} must be finite and between 0 and 180 degrees.")

        self.mesh_path = Path(mesh_path).expanduser().resolve()
        self.original_mesh_path = Path(original_mesh_path or mesh_path).expanduser().resolve()
        self.roadmap_path = None if roadmap_path is None else Path(roadmap_path).expanduser().resolve()
        self.match_tolerance_deg = float(match_tolerance_deg)
        self.ambiguity_margin_deg = float(ambiguity_margin_deg)
        self.cluster_tolerance_deg = float(cluster_tolerance_deg)
        self.symmetry_available = False
        self.symmetry_symbol = "C1"
        self._symmetry_quats = np.array([[0.0, 0.0, 0.0, 1.0]])
        self._continuous_axis: np.ndarray | None = None
        self._catalogue_quats = np.empty((0, 4), dtype=float)
        self._catalogue_axes = np.empty((0, 3), dtype=float)
        self._catalogue_node_ids = np.empty(0, dtype=int)
        self.known_pose_ids: tuple[int, ...] = ()
        self.cache_hit = False
        data = None if self.roadmap_path is None else _roadmap(self.roadmap_path)
        if data is not None:
            if self.original_mesh_path.suffix.lower() != ".stl":
                raise ValueError("Roadmap IDs require the original STL used to build the roadmap; STEP triangulation changes catalogue IDs.")
            source = _roadmap_source(data, self.roadmap_path)
            if source is not None and source.is_file() and _sha256(source) != _sha256(self.original_mesh_path):
                raise ValueError("Selected STL SHA-256 differs from the roadmap mesh source.")

        try:
            chute_pose = importlib.import_module("chute_pose")
        except ModuleNotFoundError as exc:
            if exc.name != "chute_pose" or data is not None:
                raise
            # The roadmap extra is optional. Standalone runs still record
            # unknown orientations, although symmetry cannot be inferred.
            return

        cache_file: Path | None = None
        if cache_dir is not None:
            try:
                package_version = metadata.version("bibazu-chute-pose")
            except metadata.PackageNotFoundError:
                package_version = "unknown"
            cache_key = hashlib.sha256(json.dumps({
                "cache_version": 1,
                "mesh_sha256": _sha256(self.mesh_path),
                "roadmap_sha256": _sha256(self.roadmap_path) if self.roadmap_path else None,
                "package_version": package_version,
                "symmetry_tolerance_mm": data.get("symmetry_tolerance_mm") if data else None,
            }, sort_keys=True).encode("utf-8")).hexdigest()
            cache_file = Path(cache_dir).expanduser().resolve() / f"{cache_key}.json"
            if cache_file.is_file() and self._load_cache(cache_file):
                self.cache_hit = True
                return

        symmetry_kwargs: dict[str, float] = {}
        if data is not None and data.get("symmetry_tolerance_mm") is not None:
            symmetry_kwargs["tolerance_mm"] = float(data["symmetry_tolerance_mm"])
        symmetry = chute_pose.detect_rotational_symmetry(self.mesh_path, **symmetry_kwargs)
        self.symmetry_available = True
        self.symmetry_symbol = symmetry.symbol
        if symmetry.continuous_axis_part is not None:
            axis = np.asarray(symmetry.continuous_axis_part, dtype=float)
            self._continuous_axis = axis / np.linalg.norm(axis)
        else:
            self._symmetry_quats = Rotation.from_matrix(
                np.asarray([element.rotation_part_from_part for element in symmetry.elements], dtype=float)
            ).as_quat()

        if data is None:
            if cache_file is not None:
                self._save_cache(cache_file)
            return
        catalogue = chute_pose.build_pose_catalog(self.mesh_path)
        catalogue_by_id = {pose.pose_id: pose for pose in catalogue.poses}
        mapping = _node_catalogue_ids(data)
        self.known_pose_ids = tuple(sorted(mapping))
        node_ids: list[int] = []
        quats: list[np.ndarray] = []
        for node_id, pose_ids in sorted(mapping.items()):
            for catalogue_id in pose_ids:
                if catalogue_id not in catalogue_by_id:
                    raise ValueError(
                        f"Roadmap catalogue ID {catalogue_id} is absent from the selected STL catalogue."
                    )
                base = _quat(catalogue_by_id[catalogue_id].quaternion_xyzw)
                if self._continuous_axis is None:
                    variants = (Rotation.from_quat(base) * Rotation.from_quat(self._symmetry_quats)).as_quat()
                    quats.extend(variants)
                    node_ids.extend([node_id] * len(variants))
                else:
                    quats.append(base)
                    node_ids.append(node_id)
        self._catalogue_quats = np.asarray(quats, dtype=float)
        self._catalogue_node_ids = np.asarray(node_ids, dtype=int)
        if self._continuous_axis is not None:
            self._catalogue_axes = Rotation.from_quat(self._catalogue_quats).apply(self._continuous_axis)
        if cache_file is not None:
            self._save_cache(cache_file)

    def _load_cache(self, path: Path) -> bool:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["version"] != 1:
                return False
            symmetry_quats = np.asarray(data["symmetry_quats"], dtype=float).reshape(-1, 4)
            axis = data["continuous_axis"]
            continuous_axis = None if axis is None else np.asarray(axis, dtype=float).reshape(3)
            catalogue_quats = np.asarray(data["catalogue_quats"], dtype=float).reshape(-1, 4)
            catalogue_node_ids = np.asarray(data["catalogue_node_ids"], dtype=int)
            catalogue_axes = np.asarray(data["catalogue_axes"], dtype=float).reshape(-1, 3)
            known_pose_ids = tuple(int(value) for value in data["known_pose_ids"])
            if len(catalogue_quats) != len(catalogue_node_ids):
                return False
            if continuous_axis is not None and len(catalogue_axes) != len(catalogue_quats):
                return False
            self.symmetry_available = bool(data["symmetry_available"])
            self.symmetry_symbol = str(data["symmetry_symbol"])
            self._symmetry_quats = symmetry_quats
            self._continuous_axis = continuous_axis
            self._catalogue_quats = catalogue_quats
            self._catalogue_node_ids = catalogue_node_ids
            self._catalogue_axes = catalogue_axes
            self.known_pose_ids = known_pose_ids
            return True
        except (OSError, KeyError, TypeError, ValueError):
            return False

    def _save_cache(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = {
            "version": 1,
            "symmetry_available": self.symmetry_available,
            "symmetry_symbol": self.symmetry_symbol,
            "symmetry_quats": self._symmetry_quats.tolist(),
            "continuous_axis": None if self._continuous_axis is None else self._continuous_axis.tolist(),
            "catalogue_quats": self._catalogue_quats.tolist(),
            "catalogue_node_ids": self._catalogue_node_ids.tolist(),
            "catalogue_axes": self._catalogue_axes.tolist(),
            "known_pose_ids": list(self.known_pose_ids),
        }
        temporary = path.with_name(f"{path.stem}.{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(content, separators=(",", ":")), encoding="utf-8")
        temporary.replace(path)

    def reference_quaternion(self, pose_id: int) -> tuple[float, float, float, float] | None:
        """One catalogue orientation for displaying a known roadmap pose."""
        indices = np.flatnonzero(self._catalogue_node_ids == pose_id)
        if len(indices) == 0:
            return None
        return tuple(float(value) for value in self._catalogue_quats[indices[0]])

    def resolve(self, quat_xyzw: Sequence[float] | np.ndarray) -> PoseMatch:
        """Return one unambiguous roadmap node, or retain the unknown result."""

        query = _quat(quat_xyzw)
        if len(self._catalogue_quats) == 0:
            return PoseMatch("unmatched", None, None)
        if self._continuous_axis is None:
            dots = np.clip(np.abs(self._catalogue_quats @ query), 0.0, 1.0)
            distances = np.degrees(2.0 * np.arccos(dots))
        else:
            axis = Rotation.from_quat(query).apply(self._continuous_axis)
            dots = np.clip(self._catalogue_axes @ axis, -1.0, 1.0)
            distances = np.degrees(np.arccos(dots))
        by_node = [
            (float(np.min(distances[self._catalogue_node_ids == node_id])), int(node_id))
            for node_id in np.unique(self._catalogue_node_ids)
        ]
        by_node.sort()
        best_distance, best_node = by_node[0]
        if best_distance > self.match_tolerance_deg:
            return PoseMatch("unmatched", None, best_distance)
        if len(by_node) > 1 and by_node[1][0] - best_distance <= self.ambiguity_margin_deg:
            return PoseMatch("ambiguous", None, best_distance)
        return PoseMatch("matched", best_node, best_distance)

    def cluster_unmatched(self, quaternions: Sequence[Sequence[float]] | np.ndarray) -> list[int]:
        """Complete-link classes under part symmetry, numbered deterministically.

        Every pair within a class is at most ``cluster_tolerance_deg`` apart.
        This keeps two distinct modes from merging through a chain of samples.
        Labels are ordered by a canonical orientation so input permutations do
        not renumber existing classes.
        """

        quats = _quaternions(quaternions)
        count = len(quats)
        if count == 0:
            return []
        if count == 1:
            return [0]
        if self._continuous_axis is None:
            symmetry = Rotation.from_quat(self._symmetry_quats)
            variants = [
                (Rotation.from_quat(quat) * symmetry).as_quat()
                for quat in quats
            ]
            keys = [min(_canonical_quat(variant) for variant in item) for item in variants]
        else:
            axes = Rotation.from_quat(quats).apply(self._continuous_axis)
            keys = [tuple(float(x) for x in np.round(axis, 12)) for axis in axes]

        order = sorted(range(count), key=lambda index: keys[index])
        sorted_quats = quats[order]
        distances = np.zeros((count, count), dtype=float)
        if self._continuous_axis is None:
            for row, original_index in enumerate(order):
                dots = np.abs(variants[original_index] @ sorted_quats.T)
                distances[row, :] = np.degrees(2.0 * np.arccos(np.clip(np.max(dots, axis=0), 0.0, 1.0)))
        else:
            sorted_axes = axes[order]
            distances = np.degrees(np.arccos(np.clip(sorted_axes @ sorted_axes.T, -1.0, 1.0)))
        # Floating point differences between equivalent symmetry variants can
        # make the two directed values differ slightly. Be conservative.
        distances = np.maximum(distances, distances.T)
        np.fill_diagonal(distances, 0.0)
        tree = linkage(squareform(distances, checks=False), method="complete")
        raw_labels = fcluster(tree, t=self.cluster_tolerance_deg, criterion="distance")
        first_key = {
            raw: min(keys[order[index]] for index in np.flatnonzero(raw_labels == raw))
            for raw in np.unique(raw_labels)
        }
        renumber = {raw: label for label, raw in enumerate(sorted(first_key, key=first_key.get))}
        labels = [0] * count
        for sorted_index, original_index in enumerate(order):
            labels[original_index] = renumber[int(raw_labels[sorted_index])]
        return labels
