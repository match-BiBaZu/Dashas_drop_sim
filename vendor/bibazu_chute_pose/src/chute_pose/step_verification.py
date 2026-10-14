"""Optional OpenCascade verification of STL symmetry candidates against STEP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .symmetry import RotationalSymmetryGroup


class StepSupportUnavailable(RuntimeError):
    """Raised when the optional OpenCascade dependency is not installed."""


@dataclass(frozen=True, slots=True)
class StepSymmetryElementCheck:
    element_id: int
    angle_deg: float
    relative_symmetric_difference: float
    topology_vertex_error_mm: float
    exact: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StepSymmetryVerification:
    step_source: str
    candidate_symbol: str
    status: str
    exact_volume_tolerance: float
    step_volume_mm3: float
    step_center_mass_mm: tuple[float, float, float]
    checks: tuple[StepSymmetryElementCheck, ...]

    @property
    def exact_confirmed(self) -> bool:
        return self.status == "exact_confirmed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_source": self.step_source,
            "candidate_symbol": self.candidate_symbol,
            "status": self.status,
            "exact_volume_tolerance": self.exact_volume_tolerance,
            "step_volume_mm3": self.step_volume_mm3,
            "step_center_mass_mm": self.step_center_mass_mm,
            "checks": [check.to_dict() for check in self.checks],
        }


def _load_ocp() -> dict[str, Any]:
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_VERTEX
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
        from OCP.gp import gp_Ax1, gp_Dir, gp_Trsf
    except ImportError as exc:
        raise StepSupportUnavailable(
            "STEP verification requires the optional 'step' dependency: "
            "uv run --extra step chute-pose ..."
        ) from exc
    return locals()


def _volume(shape: Any, api: dict[str, Any]) -> tuple[float, Any]:
    properties = api["GProp_GProps"]()
    api["BRepGProp"].VolumeProperties_s(shape, properties)
    return float(properties.Mass()), properties.CentreOfMass()


def _topology_vertices(shape: Any, api: dict[str, Any]) -> np.ndarray:
    points: list[list[float]] = []
    explorer = api["TopExp_Explorer"](shape, api["TopAbs_VERTEX"])
    while explorer.More():
        vertex = api["TopoDS"].Vertex(explorer.Current())
        point = api["BRep_Tool"].Pnt_s(vertex)
        points.append([point.X(), point.Y(), point.Z()])
        explorer.Next()
    if not points:
        raise ValueError("STEP solid has no topological vertices.")
    return np.unique(np.round(np.asarray(points, dtype=float), decimals=12), axis=0)


def verify_step_symmetry(
    step_path: str | Path,
    candidate: RotationalSymmetryGroup,
    *,
    exact_volume_tolerance: float = 1e-7,
) -> StepSymmetryVerification:
    """Verify an STL-derived finite symmetry on the STEP boundary representation.

    Exact confirmation uses the volume of the Boolean symmetric difference.
    Topological vertex error is diagnostic only because analytic circles can
    have seam vertices that do not share the solid's rotational symmetry.
    """

    if (
        not math.isfinite(exact_volume_tolerance)
        or exact_volume_tolerance <= 0.0
    ):
        raise ValueError("exact_volume_tolerance must be positive and finite.")
    source = Path(step_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    api = _load_ocp()
    reader = api["STEPControl_Reader"]()
    status = reader.ReadFile(str(source))
    if status != api["IFSelect_RetDone"]:
        raise ValueError(f"OpenCascade could not read STEP file: {source}")
    if reader.TransferRoots() <= 0:
        raise ValueError(f"STEP file contains no transferable solid: {source}")
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError(f"STEP import produced a null shape: {source}")

    step_volume, center = _volume(shape, api)
    if step_volume <= 0.0:
        raise ValueError(f"STEP shape has non-positive volume: {source}")
    center_array = np.array([center.X(), center.Y(), center.Z()])
    vertices = _topology_vertices(shape, api)
    centered_vertices = vertices - center_array
    vertex_tree = cKDTree(centered_vertices)
    checks: list[StepSymmetryElementCheck] = []

    for element in candidate.elements:
        if element.angle_deg < 1e-9:
            continue
        rotation = np.asarray(element.rotation_part_from_part, dtype=float)
        rotation_vector = Rotation.from_matrix(rotation).as_rotvec()
        angle = float(np.linalg.norm(rotation_vector))
        axis = rotation_vector / angle
        # Principal-axis symmetry detection carries harmless floating-point
        # noise (typically around 1e-8) into otherwise exact coordinate axes.
        # OpenCascade Boolean operations are extremely sensitive to that tiny
        # tilt when nominally coincident faces are compared, so remove only
        # numerically negligible axis components before verification.
        axis[np.abs(axis) < 1e-7] = 0.0
        axis /= np.linalg.norm(axis)
        verification_rotation = Rotation.from_rotvec(axis * angle).as_matrix()
        transform = api["gp_Trsf"]()
        transform.SetRotation(
            api["gp_Ax1"](center, api["gp_Dir"](*axis.tolist())), angle
        )
        rotated_shape = api["BRepBuilderAPI_Transform"](
            shape, transform, True
        ).Shape()
        first_difference = api["BRepAlgoAPI_Cut"](shape, rotated_shape).Shape()
        second_difference = api["BRepAlgoAPI_Cut"](rotated_shape, shape).Shape()
        first_volume, _ = _volume(first_difference, api)
        second_volume, _ = _volume(second_difference, api)
        relative_difference = max(
            0.0, (abs(first_volume) + abs(second_volume)) / step_volume
        )

        transformed_vertices = (verification_rotation @ centered_vertices.T).T
        forward_error = float(np.max(vertex_tree.query(transformed_vertices)[0]))
        backward_error = float(
            np.max(cKDTree(transformed_vertices).query(centered_vertices)[0])
        )
        checks.append(
            StepSymmetryElementCheck(
                element_id=element.element_id,
                angle_deg=element.angle_deg,
                relative_symmetric_difference=relative_difference,
                topology_vertex_error_mm=max(forward_error, backward_error),
                exact=relative_difference <= exact_volume_tolerance,
            )
        )

    if not checks:
        result_status = "no_nontrivial_candidate"
    elif all(check.exact for check in checks):
        result_status = "exact_confirmed"
    elif checks and all(
        check.topology_vertex_error_mm <= candidate.tolerance_mm for check in checks
    ):
        result_status = "practical_only_step_geometry_is_not_exact"
    else:
        result_status = "not_confirmed_by_step"
    return StepSymmetryVerification(
        step_source=str(source),
        candidate_symbol=candidate.symbol,
        status=result_status,
        exact_volume_tolerance=exact_volume_tolerance,
        step_volume_mm3=step_volume,
        step_center_mass_mm=tuple(float(value) for value in center_array),
        checks=tuple(checks),
    )
