"""Prepare a closed, uniformly dense CAD solid for MuJoCo.

All input coordinates are millimetres.  Exported OBJ meshes are metres in the
original source axes, translated so that the original solid's centre of mass
is at the body origin.  Mass and inertia always come from the original solid,
never from its collision approximation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


_CACHE_VERSION = 2
_STEP_LINEAR_DEFLECTION_MM = 0.1
_STEP_ANGULAR_DEFLECTION_RAD = 0.25
_NEAR_CONVEX_MAX_VOLUME_GAP = 0.002
_COACD_THRESHOLD_M = 0.0005
_COACD_SEED = 0


class GeometryError(ValueError):
    """An input solid or its collision decomposition is unusable."""


@dataclass(frozen=True, slots=True)
class PreparedPart:
    source_path: Path
    source_sha256: str
    mesh_sha256: str
    catalog_mesh_path: Path
    visual_mesh_path: Path
    collision_mesh_paths: tuple[Path, ...]
    mass_kg: float
    inertia_com_kg_m2: np.ndarray
    center_mass_source_mm: np.ndarray
    bounds_m: np.ndarray
    radius_m: float
    collision_quality: dict[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mesh_sha256(mesh: trimesh.Trimesh) -> str:
    digest = hashlib.sha256()
    vertices = np.ascontiguousarray(mesh.vertices, dtype="<f8")
    faces = np.ascontiguousarray(mesh.faces, dtype="<i8")
    digest.update(np.asarray(vertices.shape, dtype="<i8").tobytes())
    digest.update(vertices.tobytes())
    digest.update(np.asarray(faces.shape, dtype="<i8").tobytes())
    digest.update(faces.tobytes())
    return digest.hexdigest()


def _validate_solid(mesh: trimesh.Trimesh, source: Path) -> trimesh.Trimesh:
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise GeometryError(f"Expected one closed triangle solid: {source}")
    if not np.isfinite(mesh.vertices).all():
        raise GeometryError(f"Mesh has non-finite coordinates: {source}")
    if not mesh.is_watertight:
        raise GeometryError(f"Mesh is not watertight: {source}")
    if not mesh.is_winding_consistent:
        raise GeometryError(f"Mesh has inconsistent face winding: {source}")
    if not mesh.is_volume or not math.isfinite(float(mesh.volume)) or mesh.volume <= 0:
        raise GeometryError(f"Mesh has no positive enclosed volume: {source}")
    if not np.isfinite(mesh.center_mass).all() or not np.isfinite(mesh.moment_inertia).all():
        raise GeometryError(f"Mesh mass properties are non-finite: {source}")
    return mesh


def _load_step(source: Path) -> trimesh.Trimesh:
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
        from OCP.TopLoc import TopLoc_Location
    except ImportError as exc:
        raise GeometryError("STEP import requires the 'step' extra (cadquery-ocp).") from exc

    reader = STEPControl_Reader()
    if reader.ReadFile(str(source)) != IFSelect_RetDone or reader.TransferRoots() <= 0:
        raise GeometryError(f"OpenCascade could not read STEP solid: {source}")
    shape = reader.OneShape()
    if not BRepCheck_Analyzer(shape).IsValid():
        raise GeometryError(f"STEP shape is topologically invalid: {source}")
    BRepMesh_IncrementalMesh(
        shape, _STEP_LINEAR_DEFLECTION_MM, False, _STEP_ANGULAR_DEFLECTION_RAD, False
    ).Perform()

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None or triangulation.NbTriangles() == 0:
            raise GeometryError(f"OpenCascade could not tessellate a STEP face: {source}")
        transform = location.Transformation()
        offset = len(vertices)
        for index in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(index).Transformed(transform)
            vertices.append([point.X(), point.Y(), point.Z()])
        reversed_face = face.Orientation() == TopAbs_REVERSED
        for index in range(1, triangulation.NbTriangles() + 1):
            triangle = list(triangulation.Triangle(index).Get())
            if reversed_face:
                triangle[1], triangle[2] = triangle[2], triangle[1]
            faces.append([offset + vertex - 1 for vertex in triangle])
        explorer.Next()
    if not faces:
        raise GeometryError(f"STEP shape contains no tessellated faces: {source}")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    # OCCT stores each face's triangulation separately.  Weld coincident edge
    # nodes without moving their coordinates or silently closing real holes.
    mesh.merge_vertices(digits_vertex=8)
    return _validate_solid(mesh, source)


def _load_mesh(source: Path) -> trimesh.Trimesh:
    suffix = source.suffix.lower()
    if suffix in {".step", ".stp"}:
        return _load_step(source)
    if suffix != ".stl":
        raise GeometryError("Part file must be STL, STEP or STP (in millimetres).")
    try:
        mesh = trimesh.load_mesh(source, file_type="stl", force="mesh", process=True)
    except Exception as exc:
        raise GeometryError(f"Could not read STL mesh: {source}") from exc
    return _validate_solid(mesh, source)


def _convex_parts(centered_m: trimesh.Trimesh, volume_gap: float) -> tuple[list[trimesh.Trimesh], str]:
    if volume_gap <= _NEAR_CONVEX_MAX_VOLUME_GAP:
        return [centered_m.convex_hull], "near_convex_hull"
    try:
        import coacd
    except ImportError as exc:
        raise GeometryError("Concave solids require CoACD for collision decomposition.") from exc
    try:
        raw_parts = coacd.run_coacd(
            coacd.Mesh(np.asarray(centered_m.vertices), np.asarray(centered_m.faces)),
            threshold=_COACD_THRESHOLD_M,
            real_metric=True,
            preprocess_mode="off",
            seed=_COACD_SEED,
        )
    except Exception as exc:
        raise GeometryError(f"CoACD failed to decompose the concave solid: {exc}") from exc
    parts: list[trimesh.Trimesh] = []
    for vertices, faces in raw_parts:
        candidate = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        _validate_solid(candidate, Path("CoACD output"))
        parts.append(candidate.convex_hull)
    if not parts:
        raise GeometryError("CoACD returned no convex collision parts.")
    return parts, "coacd"


def _quality(
    original_mm: trimesh.Trimesh,
    centered_m: trimesh.Trimesh,
    parts: list[trimesh.Trimesh],
    method: str,
) -> dict[str, Any]:
    source_volume_mm3 = float(original_mm.volume)
    hull_volume_mm3 = float(original_mm.convex_hull.volume)
    summed_volume_mm3 = float(sum(part.volume for part in parts) * 1e9)
    collision_bounds = np.vstack(
        [
            np.min([part.bounds[0] for part in parts], axis=0),
            np.max([part.bounds[1] for part in parts], axis=0),
        ]
    )
    return {
        "method": method,
        "source_volume_mm3": source_volume_mm3,
        "source_hull_volume_mm3": hull_volume_mm3,
        "source_convexity_volume_gap_fraction": max(
            0.0, (hull_volume_mm3 - source_volume_mm3) / hull_volume_mm3
        ),
        "collision_part_count": len(parts),
        "collision_summed_part_volume_mm3": summed_volume_mm3,
        "collision_summed_volume_relative_error": (
            summed_volume_mm3 - source_volume_mm3
        ) / source_volume_mm3,
        "collision_bounds_max_abs_error_mm": float(
            np.abs(collision_bounds - centered_m.bounds).max() * 1000.0
        ),
        "coacd_threshold_mm": _COACD_THRESHOLD_M * 1000.0 if method == "coacd" else None,
        "note": "Part volumes are summed; overlapping convex parts are not unioned.",
    }


def _from_manifest(source: Path, source_sha: str, folder: Path, data: dict[str, Any]) -> PreparedPart:
    return PreparedPart(
        source_path=source,
        source_sha256=source_sha,
        mesh_sha256=str(data["mesh_sha256"]),
        catalog_mesh_path=source if source.suffix.lower() == ".stl" else folder / "catalog.stl",
        visual_mesh_path=folder / "visual.obj",
        collision_mesh_paths=tuple(folder / name for name in data["collision_meshes"]),
        mass_kg=float(data["mass_kg"]),
        inertia_com_kg_m2=np.asarray(data["inertia_com_kg_m2"], dtype=float),
        center_mass_source_mm=np.asarray(data["center_mass_source_mm"], dtype=float),
        bounds_m=np.asarray(data["bounds_m"], dtype=float),
        radius_m=float(data["radius_m"]),
        collision_quality=dict(data["collision_quality"]),
    )


def prepare_part(source: Path, cache_dir: Path, density_g_cm3: float) -> PreparedPart:
    """Return centred metre-scale assets and original-solid mass properties.

    Geometry files and a manifest are cached under the source checksum, density
    and fixed tessellation/decomposition settings.  The source is never edited.
    """
    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    if not source.is_file():
        raise GeometryError(f"Part file does not exist: {source}")
    if not math.isfinite(density_g_cm3) or density_g_cm3 <= 0:
        raise GeometryError("Density must be positive and finite, in g/cm³.")
    if source.suffix.lower() not in {".stl", ".step", ".stp"}:
        raise GeometryError("Part file must be STL, STEP or STP (in millimetres).")

    source_sha = _file_sha256(source)
    settings = {
        "version": _CACHE_VERSION,
        "source_sha256": source_sha,
        "source_format": source.suffix.lower(),
        "density_g_cm3": float(density_g_cm3),
        "step_linear_deflection_mm": _STEP_LINEAR_DEFLECTION_MM,
        "step_angular_deflection_rad": _STEP_ANGULAR_DEFLECTION_RAD,
        "near_convex_max_volume_gap": _NEAR_CONVEX_MAX_VOLUME_GAP,
        "coacd_threshold_m": _COACD_THRESHOLD_M,
        "coacd_seed": _COACD_SEED,
    }
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()
    folder = cache_dir / key
    manifest = folder / "manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if data["settings"] == settings and all(
                (folder / name).is_file()
                for name in [
                    "visual.obj",
                    *data["collision_meshes"],
                    *(["catalog.stl"] if source.suffix.lower() != ".stl" else []),
                ]
            ):
                return _from_manifest(source, source_sha, folder, data)
        except (KeyError, TypeError, ValueError, OSError):
            pass

    original_mm = _load_mesh(source)
    mesh_sha = _mesh_sha256(original_mm)
    center_mm = np.asarray(original_mm.center_mass, dtype=float).copy()
    centered_m = original_mm.copy()
    centered_m.apply_translation(-center_mm)
    centered_m.apply_scale(0.001)
    hull_volume_mm3 = float(original_mm.convex_hull.volume)
    volume_gap = max(0.0, (hull_volume_mm3 - original_mm.volume) / hull_volume_mm3)
    parts, method = _convex_parts(centered_m, volume_gap)
    quality = _quality(original_mm, centered_m, parts, method)

    # trimesh uses unit density for mesh.moment_inertia.  g/cm³ equals
    # 1e-6 kg/mm³; converting mm² to m² gives the combined 1e-12 factor.
    mass_kg = float(original_mm.volume * density_g_cm3 * 1e-6)
    inertia = np.asarray(original_mm.moment_inertia, dtype=float) * density_g_cm3 * 1e-12
    inertia = 0.5 * (inertia + inertia.T)
    if not np.all(np.linalg.eigvalsh(inertia) > 0):
        raise GeometryError(f"Solid has non-positive inertia: {source}")
    bounds_m = np.asarray(centered_m.bounds, dtype=float)
    radius_m = float(np.linalg.norm(centered_m.vertices, axis=1).max())

    folder.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() != ".stl":
        original_mm.export(folder / "catalog.stl", file_type="stl")
    centered_m.export(folder / "visual.obj", file_type="obj")
    collision_names: list[str] = []
    for index, part in enumerate(parts):
        name = f"collision_{index:03d}.obj"
        part.export(folder / name, file_type="obj")
        collision_names.append(name)
    data = {
        "settings": settings,
        "mesh_sha256": mesh_sha,
        "mass_kg": mass_kg,
        "inertia_com_kg_m2": inertia.tolist(),
        "center_mass_source_mm": center_mm.tolist(),
        "bounds_m": bounds_m.tolist(),
        "radius_m": radius_m,
        "collision_meshes": collision_names,
        "collision_quality": quality,
    }
    temporary = folder / "manifest.json.tmp"
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(manifest)
    return _from_manifest(source, source_sha, folder, data)
