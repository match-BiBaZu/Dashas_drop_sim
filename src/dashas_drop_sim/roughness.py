"""Seeded contact disturbances, with explicitly different energy assumptions.

These are encounters per relative sliding distance, not a measured terrain map.
An inelastic frictional collision with a shallow virtual facet redirects kinetic energy
at a loaded MuJoCo contact point. The contact Jacobian accounts for the body's
mass, inertia, orientation and the lever arm of that point.

The optional ``weight_force`` model instead supplies an external force along
chute +X, F = fraction * mass * 9.81, for a finite duration. It is an active,
uncalibrated disturbance, not passive surface roughness. Exponential waiting
times and uniformly sampled loaded belt/wall contacts are independent of slip.

``weight_impulse`` delivers the whole mass-scaled impulse once as a velocity
jump at a loaded contact. Its scheduling is quantized to physics-step starts;
later separation cannot shorten an impulse already delivered.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import RunConfig


MODEL_VERSIONS = {"longitudinal_traction": "longitudinal_contact_traction_v1",
                  "microfacet": "sliding_frictional_microfacet_v2",
                  "weight_force": "weight_scaled_contact_force_v1",
                  "weight_impulse": "mass_scaled_contact_impulse_v1"}
MIN_SLIP_M_S = 0.0001  # Suppress contact solver chatter below 0.1 mm/s.
CONTACT_DISTANCE_TOLERANCE_M = 1e-7  # 0.1 micrometre; not a flight-phase grace distance.


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


@dataclass(slots=True)
class _ForceContact:
    surface: str
    geom_id: int
    point: np.ndarray
    jacobian: np.ndarray
    relative_velocity: np.ndarray
    normal: np.ndarray
    load_n: float
    friction: float
    surface_velocity: np.ndarray


@dataclass(slots=True)
class _WeightPulse:
    end_s: float
    body_point: np.ndarray
    geom_id: int
    event: dict


@dataclass(slots=True)
class _ForceSegment:
    event: dict
    contact: _ForceContact
    duration_s: float
    end_s: float
    angular_impulse: np.ndarray
    completes_pulse: bool


class _LoadedChuteContactSampler:
    """Physical belt/wall contacts shared by external force and impulse modes."""
    def __init__(self, config: RunConfig, model: mujoco.MjModel,
                 rotation_world_chute: np.ndarray, floor_id: int, wall_id: int):
        self.model = model
        self.rotation = np.asarray(rotation_world_chute)
        self.longitudinal = self.rotation[:, 0]
        self.floor_id = floor_id
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "part")
        self.mass_kg = float(model.body_mass[self.body_id])
        self.load_threshold = self.mass_kg * 9.81 * 1e-6
        self.surfaces = (
            ("wall", wall_id, self.rotation @ np.array([0., 1., 0.]), config.mu_wall),
            ("belt", floor_id, self.rotation @ np.array([0., 0., 1.]), config.mu_belt),
        )

    def _contacts(self, data: mujoco.MjData, geom_id: int | None = None,
                  *, loaded_only: bool = True) -> list[_ForceContact]:
        candidates = []
        for name, surface_id, normal, friction in self.surfaces:
            if geom_id is not None and geom_id != surface_id:
                continue
            velocity = (data.geom_xmat[surface_id].reshape(3, 3)
                        @ self.model.geom_surfacevel[surface_id, :3])
            for index, contact in enumerate(data.contact):
                if surface_id not in (contact.geom1, contact.geom2):
                    continue
                if loaded_only and contact.efc_address < 0:
                    continue
                if contact.dist > CONTACT_DISTANCE_TOLERANCE_M:
                    continue
                other = contact.geom2 if contact.geom1 == surface_id else contact.geom1
                if self.model.geom_bodyid[other] != self.body_id:
                    continue
                force = np.zeros(6)
                if contact.efc_address >= 0:
                    mujoco.mj_contactForce(self.model, data, index, force)
                if loaded_only and force[0] <= self.load_threshold:
                    continue
                point = np.array(contact.pos)
                jacobian = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, data, jacobian, None, point, self.body_id)
                candidates.append(_ForceContact(
                    name, surface_id, point, jacobian, jacobian @ data.qvel - velocity,
                    normal, float(force[0]), friction, velocity))
        return candidates


class WeightForceRoughness(_LoadedChuteContactSampler):
    """Start a finite force at a loaded contact and continue while touching.

    ``prepare_step`` adds only this model's generalized force, and
    ``finish_step`` subtracts that same contribution without resetting other
    applied forces. First/last partial steps use their time-averaged force so
    the integrated impulse is F * actual_duration, even for nonintegral dt.
    Instantaneous zero normal load does not end a geometrically touching
    contact. If the selected surface separates, the point can move to a
    touching contact on the other chute surface, recording that change. Missing
    chute contacts or gaps above 0.1 micrometre do end the pulse.
    Gating is evaluated at step start; loss within a physics step is detected
    at the next step. The force work is a trapezoidal estimate of
    generalized power, not a claimed total kinetic-energy change.
    """

    def __init__(self, config: RunConfig, model: mujoco.MjModel,
                 rotation_world_chute: np.ndarray, floor_id: int, wall_id: int,
                 seed: int):
        super().__init__(config, model, rotation_world_chute, floor_id, wall_id)
        self.force_fraction = config.roughness_force_fraction
        self.force_n = self.force_fraction * self.mass_kg * 9.81
        self.duration_s = config.roughness_force_duration_s
        self.interval_s = config.roughness_force_interval_s
        streams = np.random.SeedSequence([seed, 0xB1BA2, 0xF0]).spawn(2)
        self.time_rng = np.random.default_rng(streams[0])
        self.contact_rng = np.random.default_rng(streams[1])
        self.events: list[dict] = []
        self.next_start_s: float | None = None
        self.active: _WeightPulse | None = None
        self._added_force = np.zeros(model.nv)
        self._segments: list[_ForceSegment] = []
        self._step_velocity: np.ndarray | None = None

    def _wait(self) -> float:
        return float(self.time_rng.exponential(self.interval_s))

    def _start_pulse(self, data: mujoco.MjData, contact: _ForceContact, start_s: float) -> None:
        response = np.zeros((3, self.model.nv))
        mujoco.mj_solveM(self.model, data, response, np.ascontiguousarray(contact.jacobian))
        mobility = contact.jacobian @ response.T
        inverse_mass = float(self.longitudinal @ mobility @ self.longitudinal)
        relative = contact.relative_velocity
        point_chute_mm = (self.rotation.T @ contact.point * 1000).tolist()
        event = {
            "time_s": start_s, "surface": contact.surface, "model": "weight_force",
            "current_surface": contact.surface, "contact_surface_changes": [],
            "force_fraction": self.force_fraction, "force_n": self.force_n,
            "force_chute_n": (self.rotation.T @ (self.force_n * self.longitudinal)).tolist(),
            "configured_duration_s": self.duration_s,
            "configured_impulse_ns": self.force_n * self.duration_s,
            "actual_duration_s": 0.0, "effective_pulse_duration_s": 0.0,
            "end_time_s": start_s, "pulse_status": "active",
            "contact_pos_chute_mm": point_chute_mm,
            "last_contact_pos_chute_mm": point_chute_mm.copy(),
            "relative_sliding_speed_mm_s": float(np.linalg.norm(
                relative - np.dot(relative, contact.normal) * contact.normal) * 1000),
            "relative_longitudinal_speed_mm_s": float(relative @ self.longitudinal * 1000),
            "sampled_height_mm": 0.0, "impulse_ns": 0.0,
            "normal_impulse_ns": 0.0, "friction_impulse_ns": 0.0,
            "friction_coefficient": contact.friction,
            "contact_normal_force_n": contact.load_n,
            "last_contact_normal_force_n": contact.load_n,
            "zero_load_duration_s": 0.0,
            "geometric_contact_tolerance_mm": CONTACT_DISTANCE_TOLERANCE_M * 1000,
            "friction_impulse_limit_ns": 0.0,
            "impulse_chute_ns": [0.0, 0.0, 0.0],
            "angular_impulse_chute_nms": [0.0, 0.0, 0.0],
            "belt_speed_mm_s": float(self.model.geom_surfacevel[self.floor_id, 0] * 1000),
            "effective_mass_kg": 1 / inverse_mass if inverse_mass > 0 else 0.0,
            "body_energy_change_j": 0.0, "relative_energy_change_j": 0.0,
            "energy_semantics": "applied_force_work_trapezoidal",
        }
        self.events.append(event)
        body_rotation = data.xmat[self.body_id].reshape(3, 3)
        body_point = body_rotation.T @ (contact.point - data.xpos[self.body_id])
        self.active = _WeightPulse(start_s + self.duration_s, body_point, contact.geom_id, event)

    def _continuing_contact(self, data: mujoco.MjData) -> _ForceContact | None:
        assert self.active is not None
        contacts = self._contacts(data, self.active.geom_id, loaded_only=False)
        if not contacts:
            contacts = self._contacts(data, loaded_only=False)
            if not contacts:
                return None
        # Keep the force on an actual touching point as rolling changes the
        # manifold. Instantaneous zero solver load is allowed while geometry
        # still touches; missing contacts never receive a force.
        point = (data.xpos[self.body_id]
                 + data.xmat[self.body_id].reshape(3, 3) @ self.active.body_point)
        contact = min(contacts, key=lambda candidate: float(np.linalg.norm(candidate.point - point)))
        if contact.geom_id != self.active.geom_id:
            self.active.geom_id = contact.geom_id
            self.active.event["current_surface"] = contact.surface
            self.active.event["contact_surface_changes"].append({
                "time_s": float(data.time), "surface": contact.surface})
        return contact

    def prepare_step(self, data: mujoco.MjData, elapsed_s: float) -> None:
        if self._step_velocity is not None:
            raise RuntimeError("finish_step must follow each prepared weight-force step")
        if elapsed_s <= 0 or not np.isfinite(elapsed_s):
            raise ValueError("The force step duration must be finite and positive")
        self._step_velocity = data.qvel.copy()
        if self.force_n <= 0:
            return
        # mj_step's contact manifold is from the previous pre-integration
        # position. Refresh it before deciding whether this step is supported.
        mujoco.mj_forward(self.model, data)
        now, step_end = float(data.time), float(data.time + elapsed_s)
        if self.next_start_s is None:
            self.next_start_s = now + self._wait()
        cursor = now
        while cursor < step_end - 1e-14:
            if self.active is None:
                if self.next_start_s >= step_end - 1e-14:
                    break
                start = max(cursor, self.next_start_s)
                contacts = self._contacts(data)
                if not contacts:
                    # A due event in flight is skipped, not postponed until
                    # impact. Re-sample the waiting time without a force event.
                    self.next_start_s = start + self._wait()
                    cursor = start
                    continue
                contact = contacts[int(self.contact_rng.integers(len(contacts)))]
                self._start_pulse(data, contact, start)
                cursor = start
            else:
                contact = self._continuing_contact(data)
                if contact is None:
                    self.active.event["pulse_status"] = "contact_lost"
                    self.active.event["end_time_s"] = cursor
                    self.active = None
                    self.next_start_s = cursor + self._wait()
                    continue
            assert self.active is not None
            end = min(step_end, self.active.end_s)
            duration = max(0.0, end - cursor)
            impulse = self.force_n * duration * self.longitudinal
            completes = end >= self.active.end_s - 1e-14
            if duration > 0:
                self._added_force += contact.jacobian.T @ (impulse / elapsed_s)
                self._segments.append(_ForceSegment(
                    self.active.event, contact, duration, end,
                    np.cross(contact.point - data.xipos[self.body_id], impulse), completes))
            cursor = end
            if completes:
                self.next_start_s = self.active.end_s + self._wait()
                self.active = None
        data.qfrc_applied[:] += self._added_force

    def finish_step(self, data: mujoco.MjData, elapsed_s: float) -> None:
        if self._step_velocity is None:
            return
        data.qfrc_applied[:] -= self._added_force
        self._added_force[:] = 0
        average_velocity = 0.5 * (self._step_velocity + data.qvel)
        for segment in self._segments:
            event, contact = segment.event, segment.contact
            impulse = self.force_n * segment.duration_s * self.longitudinal
            work = float(impulse @ (contact.jacobian @ average_velocity))
            event["actual_duration_s"] += segment.duration_s
            event["effective_pulse_duration_s"] = event["actual_duration_s"]
            event["impulse_ns"] = self.force_n * event["actual_duration_s"]
            event["impulse_chute_ns"] = (
                self.rotation.T @ (event["impulse_ns"] * self.longitudinal)).tolist()
            event["angular_impulse_chute_nms"] = (
                np.asarray(event["angular_impulse_chute_nms"])
                + self.rotation.T @ segment.angular_impulse).tolist()
            event["body_energy_change_j"] += work
            event["relative_energy_change_j"] += work - float(impulse @ contact.surface_velocity)
            event["last_contact_pos_chute_mm"] = (self.rotation.T @ contact.point * 1000).tolist()
            event["last_contact_normal_force_n"] = contact.load_n
            if contact.load_n <= self.load_threshold:
                event["zero_load_duration_s"] += segment.duration_s
            event["end_time_s"] = segment.end_s
            if segment.completes_pulse:
                event["pulse_status"] = "completed"
        self._segments.clear()
        self._step_velocity = None

    def finalize(self, data: mujoco.MjData, *, reason: str = "trial_end") -> None:
        """Label a finite pulse cut short by a trial exit or cancellation."""
        if self._step_velocity is not None:
            raise RuntimeError("finish_step is required before finalizing a force trial")
        if self.active is not None:
            self.active.event["pulse_status"] = reason
            self.active.event["end_time_s"] = float(data.time)
            self.active = None


class WeightImpulseRoughness(_LoadedChuteContactSampler):
    """Apply J = mass * configured speed increment exactly once at contact.

    The free-body COM speed increment specifies impulse strength, not the
    contact-constrained velocity after the next MuJoCo step. The Jacobian
    supplies the contact lever arm and inertia-dependent angular response.
    Events occur at the first physics-step boundary on/after their requested
    time, at most once per step. Each event or contactless skip starts a fresh
    exponential wait from that actual boundary, so overdue events do not pile
    up on landing. Force arrays are untouched.
    """

    def __init__(self, config: RunConfig, model: mujoco.MjModel,
                 rotation_world_chute: np.ndarray, floor_id: int, wall_id: int,
                 seed: int):
        super().__init__(config, model, rotation_world_chute, floor_id, wall_id)
        self.delta_v_mm_s = config.roughness_impulse_delta_v_mm_s
        self.impulse_ns = self.mass_kg * self.delta_v_mm_s / 1000
        self.interval_s = config.roughness_force_interval_s
        streams = np.random.SeedSequence([seed, 0xB1BA2, 0xF1]).spawn(2)
        self.time_rng = np.random.default_rng(streams[0])
        self.contact_rng = np.random.default_rng(streams[1])
        self.next_start_s: float | None = None
        self.events: list[dict] = []

    def _wait(self) -> float:
        return float(self.time_rng.exponential(self.interval_s))

    def prepare_step(self, data: mujoco.MjData, elapsed_s: float) -> None:
        if elapsed_s <= 0 or not np.isfinite(elapsed_s):
            raise ValueError("The impulse step duration must be finite and positive")
        if self.impulse_ns <= 0:
            return
        now = float(data.time)
        if self.next_start_s is None:
            self.next_start_s = now + self._wait()
        if self.next_start_s > now:
            return
        requested_time_s = self.next_start_s
        self.next_start_s = now + self._wait()
        # mj_step leaves contacts and mass factorization at the pre-integration
        # position. Refresh the actual boundary before sampling a loaded point.
        mujoco.mj_forward(self.model, data)
        contacts = self._contacts(data)
        if not contacts:
            return
        contact = contacts[int(self.contact_rng.integers(len(contacts)))]
        response = np.zeros((3, self.model.nv))
        mujoco.mj_solveM(self.model, data, response, np.ascontiguousarray(contact.jacobian))
        mobility = contact.jacobian @ response.T
        inverse_mass = float(self.longitudinal @ mobility @ self.longitudinal)
        impulse = self.impulse_ns * self.longitudinal
        velocity_change = response.T @ impulse
        body_energy_change = float(impulse @ (contact.jacobian @ data.qvel)
                                   + 0.5 * impulse @ mobility @ impulse)
        relative_energy_change = float(impulse @ contact.relative_velocity
                                       + 0.5 * impulse @ mobility @ impulse)
        angular_impulse = np.cross(contact.point - data.xipos[self.body_id], impulse)
        data.qvel[:] += velocity_change
        self.events.append({
            "time_s": now, "requested_time_s": requested_time_s,
            "scheduling_delay_s": now - requested_time_s,
            "surface": contact.surface, "model": "weight_impulse",
            "contact_pos_chute_mm": (self.rotation.T @ contact.point * 1000).tolist(),
            "relative_sliding_speed_mm_s": float(np.linalg.norm(
                contact.relative_velocity - np.dot(contact.relative_velocity, contact.normal)
                * contact.normal) * 1000),
            "relative_longitudinal_speed_mm_s": float(contact.relative_velocity @ self.longitudinal * 1000),
            "impulse_delta_v_mm_s": self.delta_v_mm_s,
            "impulse_ns": self.impulse_ns, "configured_impulse_ns": self.impulse_ns,
            "impulse_chute_ns": (self.rotation.T @ impulse).tolist(),
            "angular_impulse_chute_nms": (self.rotation.T @ angular_impulse).tolist(),
            "contact_normal_force_n": contact.load_n,
            "friction_coefficient": contact.friction,
            "effective_mass_kg": 1 / inverse_mass if inverse_mass > 0 else 0.0,
            "body_energy_change_j": body_energy_change,
            "relative_energy_change_j": relative_energy_change,
            "energy_semantics": "instantaneous_velocity_jump",
            "normal_impulse_ns": 0.0, "friction_impulse_ns": 0.0,
            "friction_impulse_limit_ns": 0.0, "sampled_height_mm": 0.0,
            "configured_duration_s": 0.0, "actual_duration_s": 0.0,
            "effective_pulse_duration_s": 0.0, "end_time_s": now,
            "pulse_status": "completed", "force_n": None,
            "belt_speed_mm_s": float(self.model.geom_surfacevel[self.floor_id, 0] * 1000),
        })

    def finish_step(self, data: mujoco.MjData, elapsed_s: float) -> None:
        pass

    def finalize(self, data: mujoco.MjData) -> None:
        pass


class ContactRoughness:
    def __init__(self, config: RunConfig, model: mujoco.MjModel,
                 rotation_world_chute: np.ndarray, floor_id: int, wall_id: int,
                 seed: int):
        self.model = model
        self.mode = config.roughness_model
        self.active_duration_s = config.roughness_active_duration_s
        self.observation_start_s: float | None = None
        self.window_closed = False
        self.rotation = rotation_world_chute
        self.longitudinal = self.rotation[:, 0]
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "part")
        self.floor_id = floor_id
        self.load_threshold = float(model.body_mass[self.body_id]) * 9.81 * 1e-6
        self.events: list[dict] = []
        self.surfaces: list[_Surface] = []
        self.weight_force = None
        self.weight_impulse = None
        if self.mode == "weight_force":
            self.weight_force = WeightForceRoughness(
                config, model, rotation_world_chute, floor_id, wall_id, seed)
            self.events = self.weight_force.events
            return
        if self.mode == "weight_impulse":
            self.weight_impulse = WeightImpulseRoughness(
                config, model, rotation_world_chute, floor_id, wall_id, seed)
            self.events = self.weight_impulse.events
            return
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
            speed = (abs(float(relative @ self.longitudinal)) if self.mode == "longitudinal_traction"
                     else float(np.linalg.norm(tangent)))
            candidates.append((point, jacobian, relative, speed, float(force[0])))
        return candidates

    def prepare_step(self, data: mujoco.MjData, elapsed_s: float) -> None:
        if self.observation_start_s is None:
            self.observation_start_s = float(data.time)
        if self.window_closed:
            return
        # Only full integration steps inside the window may receive a force.
        # This also bounds finite pulses, without lying about their integral
        # by applying a partial-step force over a full MuJoCo timestep.
        if (self.active_duration_s > 0 and
                data.time + elapsed_s - self.observation_start_s > self.active_duration_s + 1e-12):
            self.finalize(data, reason="window_end")
            self.window_closed = True
            return
        if self.weight_force is not None:
            self.weight_force.prepare_step(data, elapsed_s)
        elif self.weight_impulse is not None:
            self.weight_impulse.prepare_step(data, elapsed_s)

    def finish_step(self, data: mujoco.MjData, elapsed_s: float) -> None:
        if self.weight_force is not None:
            self.weight_force.finish_step(data, elapsed_s)

    def finalize(self, data: mujoco.MjData, *, reason: str = "trial_end") -> None:
        if self.weight_force is not None:
            self.weight_force.finalize(data, reason=reason)

    def advance(self, data: mujoco.MjData, elapsed_s: float) -> None:
        if self.window_closed:
            return
        if (self.active_duration_s > 0 and self.observation_start_s is not None and
                data.time - self.observation_start_s >= self.active_duration_s - 1e-12):
            return
        if self.weight_force is not None or self.weight_impulse is not None:
            return
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
                    friction_limit = surface.friction * normal_impulse
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
                    "relative_sliding_speed_mm_s": float(np.linalg.norm(
                        relative - np.dot(relative, surface.normal) * surface.normal) * 1000),
                    "relative_longitudinal_speed_mm_s": float(relative @ self.longitudinal * 1000),
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
