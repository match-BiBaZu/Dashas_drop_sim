"""Seeded contact traction changes or micro-impacts during sliding.

These are encounters per relative sliding distance, not a measured terrain map.
An inelastic frictional collision with a shallow virtual facet redirects kinetic energy
at a loaded MuJoCo contact point. The contact Jacobian accounts for the body's
mass, inertia, orientation and the lever arm of that point.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import RunConfig


MODEL_VERSIONS = {"longitudinal_traction": "longitudinal_contact_traction_v1",
                  "microfacet": "sliding_frictional_microfacet_v2"}
MIN_SLIP_M_S = 0.0001  # Suppress contact solver chatter below 0.1 mm/s.


def facet_impulse(inverse_response: np.ndarray,
                  relative_velocity: np.ndarray, normal: np.ndarray,
                  height_m: float, ramp_m: float) -> tuple[float, np.ndarray, float]:
    """Return impulse, facet normal and inverse effective mass for e=0.

    inverse_response is the point mobility J M^-1 J^T. Limit the additional
    impulse to the closing speed caused by the shallow facet, so an ordinary
    normal impact is not counted a second time as surface roughness.
    """
    tangent = relative_velocity - np.dot(relative_velocity, normal) * normal
    speed = float(np.linalg.norm(tangent))
    if speed < MIN_SLIP_M_S or height_m <= 0:
        return 0.0, normal.copy(), 0.0
    slope = height_m / ramp_m
    facet = normal - slope * tangent / speed
    facet /= np.linalg.norm(facet)
    inverse_mass = float(facet @ inverse_response @ facet)
    closing = min(max(0.0, -float(relative_velocity @ facet)),
                  slope * speed / np.sqrt(1 + slope * slope))
    impulse = closing / inverse_mass if inverse_mass > 0 else 0.0
    return float(impulse), facet, inverse_mass


def frictional_facet_impulse(inverse_response: np.ndarray,
                             relative_velocity: np.ndarray, normal: np.ndarray,
                             height_m: float, ramp_m: float,
                             friction: float) -> tuple[np.ndarray, float, float, float]:
    """An inelastic normal impulse followed by limited sliding friction.

    Friction acts in the virtual facet's tangent plane, opposes the residual
    slip, is limited to mu times the *additional* normal impulse, and cannot
    reverse slip in its direction. Both stages dissipate surface-frame energy.
    MuJoCo already handles friction from the ordinary support load.
    """
    normal_impulse, facet, inverse_mass = facet_impulse(
        inverse_response, relative_velocity, normal, height_m, ramp_m)
    impulse = normal_impulse * facet
    after = relative_velocity + inverse_response @ impulse
    tangent = after - np.dot(after, facet) * facet
    speed = float(np.linalg.norm(tangent))
    friction_impulse = 0.0
    if normal_impulse > 0 and speed > MIN_SLIP_M_S and friction > 0:
        direction = -tangent / speed
        mobility = float(direction @ inverse_response @ direction)
        if mobility > 0:
            friction_impulse = min(friction * normal_impulse, speed / mobility)
            impulse += friction_impulse * direction
    return impulse, normal_impulse, friction_impulse, inverse_mass


def longitudinal_contact_impulse(inverse_response: np.ndarray,
                                 relative_velocity: np.ndarray, direction: np.ndarray,
                                 height_m: float, ramp_m: float, load_n: float,
                                 friction: float) -> tuple[np.ndarray, float, float, float]:
    """A bounded local traction increase under the existing support load.

    The uncalibrated equivalent friction increment is mu * height/ramp, acting
    over ramp/|longitudinal slip| seconds. Cap its integrated impulse at the
    impulse cancelling this slip. No normal impulse or free energy is added.
    The ordinary support friction remains in MuJoCo's contact solver.
    """
    slip = float(relative_velocity @ direction)
    inverse_mass = float(direction @ inverse_response @ direction)
    if abs(slip) < MIN_SLIP_M_S or height_m <= 0 or load_n <= 0 or friction <= 0 or inverse_mass <= 0:
        return np.zeros(3), inverse_mass, 0.0, 0.0
    duration = ramp_m / abs(slip)
    limit = friction * height_m / ramp_m * load_n * duration
    magnitude = min(limit, abs(slip) / inverse_mass)
    return -np.sign(slip) * magnitude * direction, inverse_mass, limit, duration


@dataclass(slots=True)
class _Surface:
    name: str
    geom_id: int
    normal: np.ndarray
    friction: float
    height_m: float
    ramp_m: float
    spacing_m: float
    rng: np.random.Generator
    remaining_m: float


class ContactRoughness:
    def __init__(self, config: RunConfig, model: mujoco.MjModel,
                 rotation_world_chute: np.ndarray, floor_id: int, wall_id: int,
                 seed: int):
        self.model = model
        self.mode = config.roughness_model
        self.rotation = rotation_world_chute
        self.longitudinal = self.rotation[:, 0]
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "part")
        self.floor_id = floor_id
        self.load_threshold = float(model.body_mass[self.body_id]) * 9.81 * 1e-6
        self.events: list[dict] = []
        self.surfaces: list[_Surface] = []
        for index, (name, geom_id, local_normal, friction) in enumerate((
            ("wall", wall_id, [0, 1, 0], config.mu_wall),
            ("belt", floor_id, [0, 0, 1], config.mu_belt),
        )):
            height = getattr(config, f"roughness_{name}_height_mm") / 1000
            if height <= 0:
                continue
            spacing = getattr(config, f"roughness_{name}_spacing_mm") / 1000
            rng = np.random.default_rng(np.random.SeedSequence([seed, 0xB1BA2, index]))
            self.surfaces.append(_Surface(
                name, geom_id, self.rotation @ local_normal, friction,
                height, getattr(config, f"roughness_{name}_ramp_mm") / 1000,
                spacing, rng, float(rng.exponential(spacing)),
            ))

    def _contacts(self, data: mujoco.MjData, surface: _Surface) -> list[tuple]:
        candidates = []
        # Read the current surface speed: the conveyor can accelerate/decelerate.
        velocity = (data.geom_xmat[surface.geom_id].reshape(3, 3)
                    @ self.model.geom_surfacevel[surface.geom_id, :3])
        for index, contact in enumerate(data.contact):
            if contact.efc_address < 0 or surface.geom_id not in (contact.geom1, contact.geom2):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, index, force)
            if force[0] <= self.load_threshold:
                continue
            point = np.array(contact.pos)
            jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, data, jacobian, None, point, self.body_id)
            relative = jacobian @ data.qvel - velocity
            tangent = relative - np.dot(relative, surface.normal) * surface.normal
            candidates.append((point, jacobian, relative, float(np.linalg.norm(tangent)), float(force[0])))
        return candidates

    def advance(self, data: mujoco.MjData, elapsed_s: float) -> None:
        # mj_step leaves derived quantities at the pre-integration position.
        # Refresh before reading loaded contact points or the mass matrix.
        mujoco.mj_forward(self.model, data)
        for surface in self.surfaces:
            contacts = self._contacts(data, surface)
            if not contacts:
                continue
            loads = np.array([contact[4] for contact in contacts])
            weights = loads / loads.sum()
            speeds = np.array([contact[3] for contact in contacts])
            speed = float(weights @ np.where(speeds >= MIN_SLIP_M_S, speeds, 0.0))
            if speed <= 0:
                continue
            surface.remaining_m -= speed * elapsed_s
            while surface.remaining_m <= 0:
                surface.remaining_m += float(surface.rng.exponential(surface.spacing_m))
                contacts = self._contacts(data, surface)
                if not contacts:
                    break
                loads = np.array([contact[4] for contact in contacts])
                speeds = np.array([contact[3] for contact in contacts])
                encounter_weights = loads * np.where(speeds >= MIN_SLIP_M_S, speeds, 0.0)
                if encounter_weights.sum() <= 0:
                    break
                point, jacobian, relative, speed, load = contacts[
                    int(surface.rng.choice(len(contacts), p=encounter_weights / encounter_weights.sum()))]
                height = float(surface.rng.uniform(0, surface.height_m))
                response = np.zeros((3, self.model.nv))
                mujoco.mj_solveM(self.model, data, response, np.ascontiguousarray(jacobian))
                mobility = jacobian @ response.T
                friction_limit = duration = 0.0
                if self.mode == "longitudinal_traction":
                    impulse, inverse_mass, friction_limit, duration = longitudinal_contact_impulse(
                        mobility, relative, self.longitudinal, height, surface.ramp_m, load, surface.friction)
                    normal_impulse = 0.0
                    friction_impulse = float(np.linalg.norm(impulse))
                else:
                    impulse, normal_impulse, friction_impulse, inverse_mass = frictional_facet_impulse(
                        mobility, relative, surface.normal, height, surface.ramp_m, surface.friction)
                if np.linalg.norm(impulse) <= 0:
                    continue
                velocity_change = response.T @ impulse
                angular_impulse = np.cross(point - data.xipos[self.body_id], impulse)
                body_energy_change = float(impulse @ (jacobian @ data.qvel)
                                           + 0.5 * impulse @ mobility @ impulse)
                relative_energy_change = float(impulse @ relative
                                               + 0.5 * impulse @ mobility @ impulse)
                data.qvel[:] += velocity_change
                self.events.append({
                    "time_s": float(data.time), "surface": surface.name, "model": self.mode,
                    "contact_pos_chute_mm": (self.rotation.T @ point * 1000).tolist(),
                    "relative_sliding_speed_mm_s": speed * 1000,
                    "sampled_height_mm": height * 1000,
                    "impulse_ns": float(np.linalg.norm(impulse)),
                    "normal_impulse_ns": normal_impulse,
                    "friction_impulse_ns": friction_impulse,
                    "friction_coefficient": surface.friction,
                    "contact_normal_force_n": load,
                    "effective_pulse_duration_s": duration,
                    "friction_impulse_limit_ns": friction_limit,
                    "impulse_chute_ns": (self.rotation.T @ impulse).tolist(),
                    "angular_impulse_chute_nms": (self.rotation.T @ angular_impulse).tolist(),
                    "belt_speed_mm_s": float(self.model.geom_surfacevel[self.floor_id, 0] * 1000),
                    "effective_mass_kg": 1 / inverse_mass,
                    "body_energy_change_j": body_energy_change,
                    "relative_energy_change_j": relative_energy_change,
                })
                mujoco.mj_forward(self.model, data)
