"""Seeded micro-impact approximation for small surface defects during sliding.

These are encounters per relative sliding distance, not a measured terrain map.
An inelastic collision with a shallow virtual facet redirects kinetic energy
at a loaded MuJoCo contact point. The contact Jacobian accounts for the body's
mass, inertia, orientation and the lever arm of that point.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import RunConfig


MODEL_VERSION = "sliding_microfacet_v1"
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


@dataclass(slots=True)
class _Surface:
    name: str
    geom_id: int
    normal: np.ndarray
    velocity: np.ndarray
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
        self.rotation = rotation_world_chute
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "part")
        self.load_threshold = float(model.body_mass[self.body_id]) * 9.81 * 1e-6
        self.events: list[dict] = []
        self.surfaces: list[_Surface] = []
        for index, (name, geom_id, local_normal, local_velocity) in enumerate((
            ("wall", wall_id, [0, 1, 0], [0, 0, 0]),
            ("belt", floor_id, [0, 0, 1], [config.belt_speed_mm_s / 1000, 0, 0]),
        )):
            height = getattr(config, f"roughness_{name}_height_mm") / 1000
            if height <= 0:
                continue
            spacing = getattr(config, f"roughness_{name}_spacing_mm") / 1000
            rng = np.random.default_rng(np.random.SeedSequence([seed, 0xB1BA2, index]))
            self.surfaces.append(_Surface(
                name, geom_id, self.rotation @ local_normal, self.rotation @ local_velocity,
                height, getattr(config, f"roughness_{name}_ramp_mm") / 1000,
                spacing, rng, float(rng.exponential(spacing)),
            ))

    def _contacts(self, data: mujoco.MjData, surface: _Surface) -> list[tuple]:
        candidates = []
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
            relative = jacobian @ data.qvel - surface.velocity
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
                point, jacobian, relative, speed, _load = contacts[
                    int(surface.rng.choice(len(contacts), p=encounter_weights / encounter_weights.sum()))]
                height = float(surface.rng.uniform(0, surface.height_m))
                response = np.zeros((3, self.model.nv))
                mujoco.mj_solveM(self.model, data, response, np.ascontiguousarray(jacobian))
                mobility = jacobian @ response.T
                magnitude, facet, inverse_mass = facet_impulse(
                    mobility, relative, surface.normal, height, surface.ramp_m)
                if magnitude <= 0:
                    continue
                impulse = magnitude * facet
                velocity_change = response.T @ impulse
                body_energy_change = float(impulse @ (jacobian @ data.qvel)
                                           + 0.5 * impulse @ mobility @ impulse)
                relative_energy_change = float(impulse @ relative
                                               + 0.5 * impulse @ mobility @ impulse)
                data.qvel[:] += velocity_change
                self.events.append({
                    "time_s": float(data.time), "surface": surface.name,
                    "contact_pos_chute_mm": (self.rotation.T @ point * 1000).tolist(),
                    "relative_sliding_speed_mm_s": speed * 1000,
                    "sampled_height_mm": height * 1000,
                    "impulse_ns": magnitude,
                    "impulse_chute_ns": (self.rotation.T @ impulse).tolist(),
                    "effective_mass_kg": 1 / inverse_mass,
                    "body_energy_change_j": body_energy_change,
                    "relative_energy_change_j": relative_energy_change,
                })
                mujoco.mj_forward(self.model, data)
