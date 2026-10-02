"""Small, deterministic 3D pose sketch rendered with Qt's software painter."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from scipy.spatial.transform import Rotation
import trimesh


def render_pose_image(
    mesh_path: Path,
    center_mass_source_mm: Sequence[float],
    quat_xyzw: Sequence[float],
    pos_chute_mm: Sequence[float] | None = None,
    *,
    width: int = 520,
    height: int = 380,
) -> QImage:
    """Show the part in chute coordinates with short PE/PTFE surface sections.

    The x position is deliberately omitted: this is a close view of the end
    orientation, not a screenshot of the 1.3 m trajectory.
    """
    mesh = trimesh.load_mesh(Path(mesh_path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("Das Bauteilmesh enthält keine Dreiecke für die Posenansicht.")
    vertices = (np.asarray(mesh.vertices, dtype=float)
                - np.asarray(center_mass_source_mm, dtype=float)) / 1000
    vertices = Rotation.from_quat(quat_xyzw).apply(vertices)
    if pos_chute_mm is None:
        position_yz = np.array([
            max(0.001, 0.001 - float(vertices[:, 1].min())),
            max(0.001, 0.001 - float(vertices[:, 2].min())),
        ])
    else:
        position_yz = np.asarray(pos_chute_mm, dtype=float)[1:3] / 1000
    vertices[:, 1:3] += position_yz

    size = float(np.max(np.ptp(vertices, axis=0)))
    extent = max(0.055, size * 2.2)
    xhalf = extent * 0.65
    side = extent * 0.7
    floor = np.array([[-xhalf, 0, 0], [xhalf, 0, 0], [xhalf, side, 0], [-xhalf, side, 0]])
    wall = np.array([[-xhalf, 0, 0], [xhalf, 0, 0], [xhalf, 0, side], [-xhalf, 0, side]])

    camera = np.array([-0.85, 1.3, 1.15])
    camera /= np.linalg.norm(camera)
    right = np.cross(camera, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, camera)
    def project(points: np.ndarray) -> np.ndarray:
        return np.column_stack((points @ right, points @ up))

    all_projected = project(np.vstack((vertices, floor, wall)))
    lower, upper = all_projected.min(axis=0), all_projected.max(axis=0)
    scale = min((width - 80) / max(upper[0] - lower[0], 1e-9),
                (height - 110) / max(upper[1] - lower[1], 1e-9))
    middle = (lower + upper) / 2
    def screen(points: np.ndarray) -> QPolygonF:
        coords = project(points)
        return QPolygonF([QPointF(width / 2 + (p[0] - middle[0]) * scale,
                                  (height - 15) / 2 - (p[1] - middle[1]) * scale)
                          for p in coords])

    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(QColor("#f7f9fb"))
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#45626b"), 1.4))
        painter.setBrush(QColor("#b8dce6"))
        painter.drawPolygon(screen(floor))
        painter.setPen(QPen(QColor("#7b8587"), 1.4))
        painter.setBrush(QColor("#d9dfdf"))
        painter.drawPolygon(screen(wall))
        painter.setPen(QPen(QColor("#40505a"), 2.5))
        seam = screen(np.array([[-xhalf, 0, 0], [xhalf, 0, 0]]))
        painter.drawLine(seam[0], seam[1])

        faces = np.asarray(mesh.faces, dtype=int)
        triangles = vertices[faces]
        depth = triangles.mean(axis=1) @ camera
        light = np.array([-0.25, 0.55, 0.8])
        light /= np.linalg.norm(light)
        for index in np.argsort(depth):
            triangle = triangles[index]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            magnitude = np.linalg.norm(normal)
            if magnitude <= 1e-12:
                continue
            shade = 0.55 + 0.45 * abs(float((normal / magnitude) @ light))
            color = QColor(int(238 * shade), int(133 * shade), int(75 * shade))
            painter.setPen(QPen(QColor("#744834"), 0.45))
            painter.setBrush(color)
            painter.drawPolygon(screen(triangle))
    finally:
        painter.end()
    return image
