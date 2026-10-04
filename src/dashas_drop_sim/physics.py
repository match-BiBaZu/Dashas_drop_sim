"""MuJoCo scene and repeatable one-body chute experiments.

All mesh vertices are relative to the solid's centre of mass and in metres.
The chute frame agrees with ``chute_pose.frame.ChuteFrame``: +X along the belt,
+Y away from the PTFE wall and +Z away from the PE belt.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from typing import Callable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .config import RunConfig
from .geometry import PreparedPart
from .roughness import ContactRoughness


def _numbers(values: np.ndarray | tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{float(x):.12g}" for x in np.asarray(values).ravel())


def _wxyz(rotation: Rotation) -> np.ndarray:
    x, y, z, w = rotation.as_quat()
    return np.array([w, x, y, z], dtype=float)


def _rotation_from_wxyz(values: np.ndarray) -> Rotation:
    return Rotation.from_quat([values[1], values[2], values[3], values[0]])


def _chute_rotation(alpha_deg: float, beta_deg: float) -> Rotation:
    return Rotation.from_euler("y", beta_deg, degrees=True) * Rotation.from_euler(
        "x", alpha_deg, degrees=True
    )


@dataclass(slots=True)
class TrialOutcome:
    trial: int
    seed: int
    status: str
    initial_quat_xyzw: tuple[float, float, float, float]
    final_quat_xyzw: tuple[float, float, float, float]
    final_pos_chute_mm: tuple[float, float, float]
    sim_time_s: float
    travel_mm: float
    first_settled_s: float | None
    final_floor_contact: bool
    final_wall_contact: bool
    final_qpos: tuple[float, ...]
    settled_trace_quat_xyzw: tuple[tuple[float, float, float, float], ...]
    roughness_events: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return {**asdict(self), "roughness_impulse_count": len(self.roughness_events)}


class ChuteSimulator:
    """One compiled model; each trial uses fresh MuJoCo data via mj_resetData."""

    def __init__(self, config: RunConfig, part: PreparedPart):
        self.config = config
        self.part = part
        self.chute_rotation = _chute_rotation(config.alpha_deg, config.beta_deg)
        self.R_world_chute = self.chute_rotation.as_matrix()
        self.R_chute_world = self.R_world_chute.T
        self.start_x_m = max(0.05, part.radius_m + 0.005)
        self.exit_x_m = config.length_mm / 1000 - part.radius_m
        if self.exit_x_m <= self.start_x_m:
            raise ValueError("The part is longer than the observation section")
        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "belt_floor")
        self.wall_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ptfe_wall")

    def _build_model(self) -> mujoco.MjModel:
        cfg, part = self.config, self.part
        root = ET.Element("mujoco", model="bibazu_drop")
        ET.SubElement(root, "compiler", angle="radian", autolimits="true")
        ET.SubElement(
            root,
            "option",
            timestep=f"{cfg.timestep_s:.9g}",
            gravity="0 0 -9.81",
            integrator="implicitfast",
            solver="Newton",
            iterations="100",
            tolerance="1e-9",
        )
        assets = ET.SubElement(root, "asset")
        files = [part.visual_mesh_path, *part.collision_mesh_paths]
        asset_bytes: dict[str, bytes] = {}
        names: list[str] = []
        for index, path in enumerate(files):
            name = f"asset_{index}.obj"
            names.append(name)
            asset_bytes[name] = Path(path).read_bytes()
            ET.SubElement(assets, "mesh", name=f"mesh_{index}", file=name)

        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1", diffuse="0.8 0.8 0.8")
        chute = ET.SubElement(world, "body", name="chute", quat=_numbers(_wxyz(self.chute_rotation)))
        speed = cfg.belt_speed_mm_s / 1000
        ET.SubElement(
            chute,
            "geom",
            name="belt_floor",
            type="plane",
            size="1 1 0.01",
            surfacevel=_numbers([speed, 0, 0, 0, 0, 0]),
            rgba="0 0 0 0",
        )
        # Rotating a plane -90 deg about chute X makes its inward normal +Y.
        ET.SubElement(
            chute,
            "geom",
            name="ptfe_wall",
            type="plane",
            size="1 1 0.01",
            quat=_numbers(_wxyz(Rotation.from_euler("x", -90, degrees=True))),
            rgba="0 0 0 0",
        )
        length = cfg.length_mm / 1000
        ET.SubElement(
            chute,
            "geom",
            name="belt_visual",
            type="box",
            pos=_numbers([length / 2, 0.075, -0.003]),
            size=_numbers([length / 2, 0.075, 0.003]),
            contype="0",
            conaffinity="0",
            rgba="0.12 0.27 0.32 1",
        )
        ET.SubElement(
            chute,
            "geom",
            name="wall_visual",
            type="box",
            pos=_numbers([length / 2, -0.003, 0.075]),
            size=_numbers([length / 2, 0.003, 0.075]),
            contype="0",
            conaffinity="0",
            rgba="0.75 0.78 0.77 1",
        )

        body = ET.SubElement(world, "body", name="part")
        ET.SubElement(body, "freejoint", name="part_free")
        I = np.asarray(part.inertia_com_kg_m2, dtype=float)
        ET.SubElement(
            body,
            "inertial",
            pos="0 0 0",
            mass=f"{part.mass_kg:.12g}",
            fullinertia=_numbers([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]),
        )
        ET.SubElement(
            body,
            "geom",
            name="part_visual",
            type="mesh",
            mesh="mesh_0",
            contype="0",
            conaffinity="0",
            mass="0",
            rgba="0.92 0.42 0.15 1",
        )
        for index in range(len(part.collision_mesh_paths)):
            ET.SubElement(
                body,
                "geom",
                name=f"part_collision_{index}",
                type="mesh",
                mesh=f"mesh_{index + 1}",
                mass="0",
                rgba="0 0 0 0",
            )

        pairs = ET.SubElement(root, "contact")
        for index in range(len(part.collision_mesh_paths)):
            geom = f"part_collision_{index}"
            for surface, mu in (("belt_floor", cfg.mu_belt), ("ptfe_wall", cfg.mu_wall)):
                attributes = {
                    "geom1": geom, "geom2": surface,
                    "condim": "1" if mu == 0 else "6",
                    "solref": "0.005 1.2",
                }
                if mu > 0:
                    attributes["friction"] = _numbers([mu, mu, 0.005, 0.0001, 0.0001])
                ET.SubElement(pairs, "pair", **attributes)
        xml = ET.tostring(root, encoding="unicode")
        try:
            return mujoco.MjModel.from_xml_string(xml, assets=asset_bytes)
        except ValueError as exc:
            raise ValueError(f"Could not compile the MuJoCo model: {exc}") from exc

    def _local_quat(self, data: mujoco.MjData) -> np.ndarray:
        world_part = _rotation_from_wxyz(np.asarray(data.qpos[3:7]))
        return (self.chute_rotation.inv() * world_part).as_quat()

    def _local_position(self, data: mujoco.MjData) -> np.ndarray:
        return self.R_chute_world @ np.asarray(data.qpos[:3])

    def _contact_flags(self, data: mujoco.MjData) -> tuple[bool, bool]:
        floor = wall = False
        for contact in data.contact:
            if contact.efc_address < 0:
                continue
            floor |= contact.geom1 == self.floor_id or contact.geom2 == self.floor_id
            wall |= contact.geom1 == self.wall_id or contact.geom2 == self.wall_id
        return floor, wall

    def _spawn(self, local_quat_xyzw: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        local_part = Rotation.from_quat(local_quat_xyzw)
        vertices = self._centered_vertices_m
        rotated = local_part.apply(vertices)
        lateral = self.config.lateral_mm / 1000
        margin = 0.002
        bisector_height = max(
            math.sqrt(2) * (margin - float(np.min(rotated[:, 1]))) - lateral,
            math.sqrt(2) * (margin - float(np.min(rotated[:, 2]))) + lateral,
        ) + self.config.drop_height_mm / 1000
        local_pos = np.array(
            [self.start_x_m, (bisector_height + lateral) / math.sqrt(2),
             (bisector_height - lateral) / math.sqrt(2)],
            dtype=float,
        )
        self.data.qpos[:3] = self.R_world_chute @ local_pos
        self.data.qpos[3:7] = _wxyz(self.chute_rotation * local_part)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    @property
    def _centered_vertices_m(self) -> np.ndarray:
        # Cached OBJ may have engine-specific mesh centering; the source STL
        # supplies the exact release clearance in the original part frame.
        if not hasattr(self, "_release_vertices"):
            import trimesh

            mesh = trimesh.load_mesh(self.part.catalog_mesh_path, force="mesh", process=False)
            self._release_vertices = (
                np.asarray(mesh.vertices, dtype=float)
                - np.asarray(self.part.center_mass_source_mm, dtype=float)
            ) / 1000
        return self._release_vertices

    def _window_is_settled(self, history: deque[tuple]) -> bool:
        if not history or history[-1][0] - history[0][0] < 0.5 - 2 * self.config.timestep_s:
            return False
        # Contact impulses make instantaneous qvel and contact flags chatter on
        # a moving belt, even for an orientation fixed to <0.1 degree. Use the
        # measured orientation change over the window as a filtered angular
        # velocity, and require repeated (rather than uninterrupted) support.
        samples = list(history)
        if sum(sample[3] for sample in samples) < 0.25 * len(samples):
            return False
        if sum(sample[4] for sample in samples) < 0.25 * len(samples):
            return False
        quats = np.asarray([sample[1] for sample in samples], dtype=float)
        dots = np.abs(quats @ quats[-1])
        if np.any(dots < math.cos(math.radians(0.5))):
            return False
        # Sign-align unit quaternions before averaging the first/last 100 ms.
        width = max(2, round(0.1 / 0.01))
        def average_quat(segment: np.ndarray) -> np.ndarray:
            aligned = segment * np.where(segment @ segment[0] < 0, -1.0, 1.0)[:, None]
            mean = np.mean(aligned, axis=0)
            return mean / np.linalg.norm(mean)
        start = average_quat(quats[:width])
        end = average_quat(quats[-width:])
        duration = samples[-1][0] - samples[0][0]
        filtered_omega = 2 * math.acos(min(1.0, abs(float(start @ end)))) / duration
        if filtered_omega > 0.02:
            return False
        cross_positions = np.asarray([sample[2] for sample in samples])
        if np.max(np.ptp(cross_positions, axis=0)) > 0.001:
            return False
        cross_speed = np.linalg.norm(np.mean(cross_positions[-width:], axis=0)
                                     - np.mean(cross_positions[:width], axis=0)) / duration
        return bool(cross_speed <= 0.001)

    def _run_until_end(
        self,
        *,
        trial: int,
        seed: int,
        initial_quat: np.ndarray,
        cancel: Callable[[], bool] | None = None,
        viewer=None,
        trace_stable: bool = False,
    ) -> TrialOutcome:
        cfg, data = self.config, self.data
        start_x = float(self._local_position(data)[0])
        belt_m_s = cfg.belt_speed_mm_s / 1000
        max_time = 10.0 if belt_m_s == 0 else max(10.0, 5.0 + 1.5 * (self.exit_x_m - start_x) / belt_m_s)
        history: deque[tuple] = deque()
        first_settled: float | None = None
        settled_trace: list[tuple[float, float, float, float]] = []
        last_trace_time = -math.inf
        last_progress_time = data.time
        last_progress_x = start_x
        sample_stride = max(1, round(0.01 / cfg.timestep_s))
        settle_stride = max(1, round(0.05 / (sample_stride * cfg.timestep_s)))
        viewer_stride = max(1, round(0.02 / cfg.timestep_s))
        wall_start = time.perf_counter() if viewer is not None else 0.0
        roughness = (ContactRoughness(cfg, self.model, self.R_world_chute,
                                     self.floor_id, self.wall_id, seed)
                     if cfg.roughness_enabled and (cfg.roughness_wall_height_mm > 0
                                                   or cfg.roughness_belt_height_mm > 0) else None)
        roughness_stride = max(1, round(0.005 / cfg.timestep_s))
        status = "timeout"
        step = 0
        while data.time < max_time:
            if cancel is not None and cancel():
                status = "cancelled"
                break
            mujoco.mj_step(self.model, data)
            step += 1
            if roughness is not None and step % roughness_stride == 0:
                roughness.advance(data, roughness_stride * cfg.timestep_s)
            if viewer is not None and step % viewer_stride == 0 and viewer.is_running():
                # Keep the same physics steps as a headless run, but show them
                # at roughly real time without blocking on every single step.
                viewer.cam.lookat[:] = self.R_world_chute @ np.array(
                    [self._local_position(data)[0], 0.075, 0.075]
                )
                viewer.sync()
                remaining = data.time - (time.perf_counter() - wall_start)
                if remaining > 0:
                    time.sleep(remaining)
            if step % sample_stride:
                continue
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                status = "invalid"
                break
            pos = self._local_position(data)
            quat = self._local_quat(data)
            floor, wall = self._contact_flags(data)
            history.append((data.time, quat.copy(), pos[1:3].copy(), floor, wall))
            while history and data.time - history[0][0] > 0.51:
                history.popleft()
            reached_exit = belt_m_s > 0 and pos[0] >= self.exit_x_m
            stable = (self._window_is_settled(history)
                      if step // sample_stride % settle_stride == 0 or reached_exit else False)
            if stable and first_settled is None:
                first_settled = float(data.time)
            if stable and trace_stable and data.time - last_trace_time >= 0.5:
                settled_trace.append(tuple(float(x) for x in quat))
                last_trace_time = float(data.time)
            if np.min(pos[1:3]) < -0.05:
                status = "lost"
                break
            if reached_exit:
                status = "settled" if stable else "unsettled"
                break
            if belt_m_s > 0 and data.time - last_progress_time >= 3:
                if pos[0] - last_progress_x < 0.05 * belt_m_s * 3 and (first_settled is not None or data.time >= 3):
                    status = "stalled"
                    break
                last_progress_time, last_progress_x = data.time, pos[0]
        else:
            status = ("settled_stationary" if self._window_is_settled(history) else "unsettled_stationary") if belt_m_s == 0 else "timeout"
        final_pos = self._local_position(data)
        if viewer is not None and viewer.is_running():
            viewer.sync()
        floor, wall = self._contact_flags(data)
        if trace_stable and self._window_is_settled(history) and data.time > last_trace_time + 0.01:
            settled_trace.append(tuple(float(x) for x in self._local_quat(data)))
        return TrialOutcome(
            trial=trial,
            seed=seed,
            status=status,
            initial_quat_xyzw=tuple(float(x) for x in initial_quat),
            final_quat_xyzw=tuple(float(x) for x in self._local_quat(data)),
            final_pos_chute_mm=tuple(float(x * 1000) for x in final_pos),
            sim_time_s=float(data.time),
            travel_mm=float((final_pos[0] - start_x) * 1000),
            first_settled_s=first_settled,
            final_floor_contact=floor,
            final_wall_contact=wall,
            final_qpos=tuple(float(x) for x in data.qpos[:7]),
            settled_trace_quat_xyzw=tuple(settled_trace),
            roughness_events=tuple(roughness.events) if roughness is not None else (),
        )

    def drop(
        self,
        index: int,
        seed: int,
        *,
        cancel: Callable[[], bool] | None = None,
        preview: bool = False,
        viewer=None,
    ) -> TrialOutcome:
        rng = np.random.default_rng(seed)
        local_quat = Rotation.random(random_state=rng).as_quat()
        self._spawn(local_quat)
        if preview and viewer is not None:
            raise ValueError("preview and viewer cannot be combined")
        if preview:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.distance = max(0.4, min(self.config.length_mm / 1000 * 0.6, 2.0))
                viewer.cam.lookat[:] = self.data.qpos[:3]
                return self._run_until_end(trial=index, seed=seed, initial_quat=local_quat,
                                           cancel=cancel, viewer=viewer)
        return self._run_until_end(trial=index, seed=seed, initial_quat=local_quat,
                                   cancel=cancel, viewer=viewer)

    def kick(
        self,
        source_qpos: tuple[float, ...],
        energy_lift_mm: float,
        direction_chute: np.ndarray,
        *,
        index: int = -1,
        cancel: Callable[[], bool] | None = None,
    ) -> TrialOutcome:
        mujoco.mj_resetData(self.model, self.data)
        source = np.asarray(source_qpos, dtype=float)
        local_pos = self.R_chute_world @ source[:3]
        local_pos[0] = self.start_x_m
        self.data.qpos[:3] = self.R_world_chute @ local_pos
        self.data.qpos[3:7] = source[3:7]
        self.data.qvel[:3] = self.R_world_chute @ np.array([self.config.belt_speed_mm_s / 1000, 0, 0])
        self.data.qvel[3:6] = 0
        mujoco.mj_forward(self.model, self.data)
        # Let the reset state re-seat before the angular kick.
        for _ in range(round(0.5 / self.config.timestep_s)):
            if cancel is not None and cancel():
                break
            mujoco.mj_step(self.model, self.data)
        local_pos = self._local_position(self.data)
        local_pos[0] = self.start_x_m
        self.data.qpos[:3] = self.R_world_chute @ local_pos
        part_world = _rotation_from_wxyz(np.asarray(self.data.qpos[3:7]))
        direction = np.asarray(direction_chute, dtype=float)
        direction /= np.linalg.norm(direction)
        body_direction = part_world.inv().apply(self.R_world_chute @ direction)
        inertia = np.asarray(self.part.inertia_com_kg_m2)
        effective_inertia = float(body_direction @ inertia @ body_direction)
        energy_j = self.part.mass_kg * 9.81 * energy_lift_mm / 1000
        speed = math.sqrt(2 * energy_j / effective_inertia) if energy_j > 0 else 0.0
        self.data.qvel[3:6] = speed * body_direction  # free-joint angular qvel is body-local
        mujoco.mj_forward(self.model, self.data)
        return self._run_until_end(
            trial=index,
            seed=(int(np.random.SeedSequence([self.config.seed, index + 1, 0x51AB1E])
                      .generate_state(1, dtype=np.uint32)[0]) if self.config.roughness_enabled else 0),
            initial_quat=self._local_quat(self.data),
            cancel=cancel,
            trace_stable=True,
        )
