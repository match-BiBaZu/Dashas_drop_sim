"""Coordinate conventions and chute orientation transformations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


Vector3 = NDArray[np.float64]
Matrix3 = NDArray[np.float64]


def _rotation_x(angle_rad: float) -> Matrix3:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def _rotation_y(angle_rad: float) -> Matrix3:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


@dataclass(frozen=True, slots=True)
class ChuteFrame:
    """Pose of the chute relative to its neutral position.

    The local chute frame is right-handed:

    * ``+X`` points downhill along the chute.
    * ``+Y`` points away from the side wall into the chute.
    * ``+Z`` points away from the chute floor.
    * The floor is ``z = 0`` and its material interior is ``z >= 0``.
    * The wall is ``y = 0`` and its material interior is ``y >= 0``.
    * Their common seam is the line ``(x, 0, 0)``.

    Starting from the neutral pose, the complete chute is first rotated by
    ``beta`` about the original Y axis. It is then rotated by ``alpha`` about
    the already moved, chute-fixed X axis. For active column-vector rotations
    this gives ``R_world_from_chute = R_y(beta) @ R_x(alpha)``.
    """

    alpha_deg: float
    beta_deg: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha_deg) or not math.isfinite(self.beta_deg):
            raise ValueError("Chute angles must be finite numbers.")

    @property
    def rotation_world_from_chute(self) -> Matrix3:
        alpha = math.radians(self.alpha_deg)
        beta = math.radians(self.beta_deg)
        return _rotation_y(beta) @ _rotation_x(alpha)

    @property
    def rotation_chute_from_world(self) -> Matrix3:
        return self.rotation_world_from_chute.T

    def vector_to_world(self, vector_chute: Vector3) -> Vector3:
        vector = np.asarray(vector_chute, dtype=float)
        if vector.shape != (3,):
            raise ValueError("Expected a three-dimensional vector.")
        return self.rotation_world_from_chute @ vector

    def vector_to_chute(self, vector_world: Vector3) -> Vector3:
        vector = np.asarray(vector_world, dtype=float)
        if vector.shape != (3,):
            raise ValueError("Expected a three-dimensional vector.")
        return self.rotation_chute_from_world @ vector

    def gravity_chute(self, gravity_magnitude: float = 9.81) -> Vector3:
        """Return world gravity expressed in the chute frame, in m/s^2."""

        if not math.isfinite(gravity_magnitude) or gravity_magnitude <= 0.0:
            raise ValueError("Gravity magnitude must be a positive finite number.")
        gravity_world = np.array([0.0, 0.0, -gravity_magnitude], dtype=float)
        return self.vector_to_chute(gravity_world)

    @property
    def floor_inward_normal(self) -> Vector3:
        return np.array([0.0, 0.0, 1.0], dtype=float)

    @property
    def wall_inward_normal(self) -> Vector3:
        return np.array([0.0, 1.0, 0.0], dtype=float)

    @property
    def seam_direction(self) -> Vector3:
        return np.array([1.0, 0.0, 0.0], dtype=float)

