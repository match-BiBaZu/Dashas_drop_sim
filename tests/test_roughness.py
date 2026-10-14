"""Physical scaling, contact gating and reproducibility of micro-impacts."""

import mujoco
import numpy as np
import pytest
import trimesh

from dashas_drop_sim.config import RunConfig
from dashas_drop_sim.geometry import prepare_part
from dashas_drop_sim.physics import ChuteSimulator
from dashas_drop_sim.roughness import (
    ContactRoughness, facet_impulse, frictional_facet_impulse, longitudinal_contact_impulse,
)


def test_facet_impulse_scales_with_mass_speed_and_slope_without_adding_energy():
    normal = np.array([0., 1., 0.])
    impulse, facet, _ = facet_impulse(np.eye(3), np.array([0.1, 0., 0.]), normal, 0.0001, 0.003)
    heavier, _, _ = facet_impulse(np.eye(3) / 2, np.array([0.1, 0., 0.]), normal, 0.0001, 0.003)
    faster, _, _ = facet_impulse(np.eye(3), np.array([0.2, 0., 0.]), normal, 0.0001, 0.003)
    assert impulse > 0
    assert heavier == pytest.approx(2 * impulse)
    assert faster == pytest.approx(2 * impulse)
    assert facet[1] > 0 and facet[0] < 0
    velocity = np.array([0.1, 0., 0.])
    assert np.dot(velocity + impulse * facet, velocity + impulse * facet) < np.dot(velocity, velocity)
    assert facet_impulse(np.eye(3), np.zeros(3), normal, 0.0001, 0.003)[0] == 0
    assert facet_impulse(np.eye(3), velocity, normal, 0, 0.003)[0] == 0


@pytest.fixture(params=("longitudinal_traction", "microfacet"))
def rough_cube(tmp_path, request):
    path = tmp_path / "cube.stl"
    trimesh.creation.box(extents=[20, 20, 20]).export(path)
    part = prepare_part(path, tmp_path / "cache", 1.15)
    config = RunConfig(mesh_path=path, output_dir=tmp_path / "results", length_mm=300,
                       drop_height_mm=0, workers=1, roughness_enabled=True, roughness_model=request.param)
    config.validate()
    return config, part


def test_seeded_rough_drops_replay_after_reset_and_use_loaded_wall_points(rough_cube):
    config, part = rough_cube
    simulator = ChuteSimulator(config, part)
    first = simulator.drop(0, 123)
    simulator.drop(1, 124)
    repeated = simulator.drop(2, 123)
    assert first.roughness_events
    assert first.roughness_events == repeated.roughness_events
    assert first.final_qpos == repeated.final_qpos
    assert {event["surface"] for event in first.roughness_events} == {"wall"}
    for event in first.roughness_events:
        assert event["relative_energy_change_j"] <= 1e-12
        assert event["impulse_ns"] > 0
        assert 0 < event["sampled_height_mm"] <= config.roughness_wall_height_mm
        assert abs(event["contact_pos_chute_mm"][1]) < 1
        assert event["effective_mass_kg"] <= part.mass_kg * (1 + 1e-8)


def test_no_impulses_without_contact_or_relative_motion(rough_cube):
    config, part = rough_cube
    simulator = ChuteSimulator(config, part)
    simulator._spawn(np.array([0., 0., 0., 1.]))
    simulator.data.qpos[:3] = simulator.R_world_chute @ np.array([0.05, 0.1, 0.1])
    simulator.data.qvel[:3] = [0.1, 0, 0]
    roughness = ContactRoughness(config, simulator.model, simulator.R_world_chute,
                                simulator.floor_id, simulator.wall_id, 123)
    roughness.advance(simulator.data, 100)
    assert roughness.events == []
    simulator.data.qpos[:3] = simulator.R_world_chute @ np.array([0.05, 0.00999, 0.00999])
    simulator.data.qvel[:] = 0
    roughness.advance(simulator.data, 100)
    assert roughness.events == []
    config.belt_speed_mm_s = 0
    config.roughness_belt_height_mm = 0.1
    simulator._spawn(np.array([0., 0., 0., 1.]))  # Reset the model's surface velocity too.
    roughness = ContactRoughness(config, simulator.model, simulator.R_world_chute,
                                simulator.floor_id, simulator.wall_id, 123)
    roughness.advance(simulator.data, 100)
    assert roughness.events == []


def test_zero_amplitude_keeps_original_physics(rough_cube):
    config, part = rough_cube
    config.roughness_enabled = False
    smooth = ChuteSimulator(config, part).drop(0, 123)
    config.roughness_enabled = True
    config.roughness_wall_height_mm = 0
    config.roughness_belt_height_mm = 0
    zero = ChuteSimulator(config, part).drop(0, 123)
    assert zero.to_dict() == smooth.to_dict()


def test_belt_impulses_use_belt_relative_velocity_and_produce_rotation(rough_cube):
    config, part = rough_cube
    config.roughness_wall_height_mm = 0
    config.roughness_belt_height_mm = 0.1
    config.roughness_belt_spacing_mm = 1
    outcome = ChuteSimulator(config, part).drop(0, 123)
    assert outcome.roughness_events
    assert {event["surface"] for event in outcome.roughness_events} == {"belt"}
    assert all(event["relative_energy_change_j"] <= 1e-12 for event in outcome.roughness_events)
    # A moving surface can supply body energy while the collision dissipates
    # energy in the surface frame. Off-centre contacts also impart rotation.
    assert any(event["body_energy_change_j"] > 0 for event in outcome.roughness_events)
    model = ChuteSimulator(config, part)
    model._spawn(np.array([0., 0., 0., 1.]))
    body_id = mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_BODY, "part")
    point = model.data.xipos[body_id] + model.R_world_chute @ np.array([0.01, 0.0, -0.01])
    jacobian = np.zeros((3, model.model.nv))
    mujoco.mj_jac(model.model, model.data, jacobian, None, point, body_id)
    response = np.zeros_like(jacobian)
    mujoco.mj_solveM(model.model, model.data, response, jacobian)
    delta = response.T @ (model.R_world_chute @ np.array([0., 0., 1e-5]))
    assert np.linalg.norm(delta[3:]) > 0


def test_half_timestep_preserves_small_roughness_transport(rough_cube):
    config, part = rough_cube
    config.belt_speed_variation_mm_s = 3
    results = []
    for timestep in (0.001, 0.0005):
        config.timestep_s = timestep
        simulator = ChuteSimulator(config, part)
        simulator._spawn(np.array([0., 0., 0., 1.]))
        simulator.data.qpos[:3] = simulator.R_world_chute @ np.array([0.05, 0.011, 0.011])
        mujoco.mj_forward(simulator.model, simulator.data)
        results.append(simulator._run_until_end(trial=0, seed=123, initial_quat=np.array([0., 0., 0., 1.])))
    assert results[0].status == results[1].status == "settled"
    assert abs(len(results[0].roughness_events) - len(results[1].roughness_events)) <= 2
    assert results[0].final_pos_chute_mm == pytest.approx(results[1].final_pos_chute_mm, abs=1)


def test_roughness_configuration_roundtrip_and_small_slope_validation(rough_cube):
    config, _part = rough_cube
    assert RunConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
    config.roughness_wall_ramp_mm = 0.1
    with pytest.raises(ValueError, match="height/ramp"):
        config.validate()


def test_longitudinal_impulse_is_loaded_friction_limited_and_passive():
    axis = np.array([1., 0., 0.])
    velocity = np.array([0.1, 0., 0.])
    impulse, inverse_mass, limit, duration = longitudinal_contact_impulse(
        np.eye(3), velocity, axis, 0.0001, 0.003, 9.81, 0.2)
    assert impulse[0] < 0 and impulse[1] == impulse[2] == 0
    assert np.linalg.norm(impulse) <= limit
    assert duration == pytest.approx(0.003 / 0.1)
    assert limit == pytest.approx(0.2 * (0.0001 / 0.003) * 9.81 * duration)
    assert inverse_mass == 1
    assert 0 <= (velocity + impulse)[0] < velocity[0]
    assert impulse @ velocity + 0.5 * impulse @ impulse <= 0
    heavy, _, _, _ = longitudinal_contact_impulse(
        np.eye(3) / 2, velocity, axis, 0.0001, 0.003, 2 * 9.81, 0.2)
    assert heavy == pytest.approx(2 * impulse)
    for friction, load in ((0, 9.81), (0.2, 0)):
        assert not np.any(longitudinal_contact_impulse(
            np.eye(3), velocity, axis, 0.0001, 0.003, load, friction)[0])
    # Large budgets cannot reverse the slip, and either direction dissipates.
    backwards = -velocity
    stopped, *_ = longitudinal_contact_impulse(
        np.eye(3), backwards, axis, 0.0001, 0.003, 1e6, 0.2)
    assert stopped == pytest.approx(-backwards)


def test_frictional_facet_has_stronger_longitudinal_component_and_is_passive():
    velocity = np.array([0.1, 0., 0.])
    normal = np.array([0., 1., 0.])
    magnitude, facet, _ = facet_impulse(np.eye(3), velocity, normal, 0.0001, 0.003)
    impulse, normal_impulse, friction_impulse, _ = frictional_facet_impulse(
        np.eye(3), velocity, normal, 0.0001, 0.003, 0.2)
    assert normal_impulse == magnitude
    assert 0 < friction_impulse <= 0.2 * normal_impulse
    assert abs(impulse[0]) > 5 * abs(magnitude * facet[0])
    assert impulse @ velocity + 0.5 * impulse @ impulse <= 0


def test_longitudinal_contact_events_create_a_tipping_moment(rough_cube):
    config, part = rough_cube
    config.roughness_model = "longitudinal_traction"
    simulator = ChuteSimulator(config, part)
    simulator._spawn(np.array([0., 0., 0., 1.]))
    simulator.data.qpos[:3] = simulator.R_world_chute @ np.array([0.05, 0.00999, 0.00999])
    simulator.data.qvel[:3] = simulator.R_world_chute @ np.array([0.1, 0., 0.])
    roughness = ContactRoughness(config, simulator.model, simulator.R_world_chute,
                                simulator.floor_id, simulator.wall_id, 123)
    roughness.advance(simulator.data, 0.5)
    assert roughness.events
    assert np.linalg.norm(simulator.data.qvel[3:]) > 0
    for event in roughness.events:
        assert event["impulse_chute_ns"][1:] == pytest.approx([0, 0], abs=1e-15)
        assert event["normal_impulse_ns"] == 0
        assert event["contact_normal_force_n"] > 0
        assert event["friction_impulse_ns"] <= event["friction_impulse_limit_ns"]
        assert np.linalg.norm(event["angular_impulse_chute_nms"][1:]) > 0
        assert event["relative_energy_change_j"] <= 1e-12


@pytest.fixture
def weight_cube(tmp_path):
    path = tmp_path / "weight_cube.stl"
    trimesh.creation.box(extents=[20, 20, 20]).export(path)
    part = prepare_part(path, tmp_path / "cache", 1.15)
    config = RunConfig(
        mesh_path=path, output_dir=tmp_path / "results", length_mm=300, workers=1,
        belt_speed_mm_s=0, roughness_enabled=True, roughness_model="weight_force",
        roughness_wall_height_mm=0, roughness_belt_height_mm=0,
        roughness_force_duration_s=0.0053, roughness_force_interval_s=10)
    config.validate()
    return config, part


def _loaded_weight_cube(config, part, seed=123):
    simulator = ChuteSimulator(config, part)
    simulator._spawn(np.array([0., 0., 0., 1.]))
    simulator.data.qpos[:3] = simulator.R_world_chute @ np.array([0.05, 0.00999, 0.00999])
    simulator.data.qvel[:] = 0
    mujoco.mj_forward(simulator.model, simulator.data)
    roughness = ContactRoughness(config, simulator.model, simulator.R_world_chute,
                                simulator.floor_id, simulator.wall_id, seed)
    return simulator, roughness


def _force_step(simulator, roughness):
    dt = simulator.config.timestep_s
    roughness.prepare_step(simulator.data, dt)
    applied = simulator.data.qfrc_applied.copy()
    mujoco.mj_step(simulator.model, simulator.data)
    roughness.finish_step(simulator.data, dt)
    return applied


def test_weight_force_is_a_finite_external_force_with_lever_arm_and_preserves_other_forces(weight_cube):
    config, part = weight_cube
    simulator, roughness = _loaded_weight_cube(config, part)
    roughness.weight_force.next_start_s = 0.0
    baseline = np.arange(simulator.model.nv) * 1e-8
    simulator.data.qfrc_applied[:] = baseline
    simulator.data.xfrc_applied[:] = 1e-8
    external_wrench = simulator.data.xfrc_applied.copy()
    velocity_before = simulator.data.qvel.copy()
    roughness.prepare_step(simulator.data, config.timestep_s)
    assert simulator.data.qvel == pytest.approx(velocity_before, abs=0)
    added = simulator.data.qfrc_applied - baseline
    assert simulator.R_chute_world @ added[:3] == pytest.approx(
        [0.1 * part.mass_kg * 9.81, 0, 0], abs=1e-14)
    assert np.linalg.norm(added[3:]) > 0
    assert roughness.events[0]["relative_sliding_speed_mm_s"] == 0
    mujoco.mj_step(simulator.model, simulator.data)
    # An independent force user can add a contribution while the pulse is
    # prepared. Removing this model's force must retain that new contribution.
    extra = np.full(simulator.model.nv, 1e-9)
    simulator.data.qfrc_applied[:] += extra
    roughness.finish_step(simulator.data, config.timestep_s)
    assert simulator.data.qfrc_applied == pytest.approx(baseline + extra, abs=1e-17)
    assert np.array_equal(simulator.data.xfrc_applied, external_wrench)
    event = roughness.events[0]
    assert event["actual_duration_s"] == pytest.approx(config.timestep_s)
    assert event["impulse_ns"] == pytest.approx(event["force_n"] * config.timestep_s)
    assert np.linalg.norm(event["angular_impulse_chute_nms"]) > 0
    assert event["energy_semantics"] == "applied_force_work_trapezoidal"


def test_weight_force_scales_with_mass_and_fraction(weight_cube, tmp_path):
    config, part = weight_cube
    forces = []
    for density, fraction in ((1.15, 0.1), (2.3, 0.1), (1.15, 0.2)):
        config.roughness_force_fraction = fraction
        selected_part = prepare_part(config.mesh_path, tmp_path / f"cache_{density}", density)
        simulator, roughness = _loaded_weight_cube(config, selected_part)
        roughness.weight_force.next_start_s = 0.0
        _force_step(simulator, roughness)
        event = roughness.events[0]
        forces.append(event["force_n"])
        assert event["force_n"] == pytest.approx(fraction * selected_part.mass_kg * 9.81)
        assert event["configured_impulse_ns"] == pytest.approx(
            event["force_n"] * config.roughness_force_duration_s)
    assert forces[1:] == pytest.approx([2 * forces[0], 2 * forces[0]])


def test_weight_force_no_flight_force_and_contact_loss_aborts_pulse(weight_cube):
    config, part = weight_cube
    config.roughness_force_duration_s = 0.05
    simulator, roughness = _loaded_weight_cube(config, part)
    roughness.weight_force.next_start_s = 0.0
    _force_step(simulator, roughness)
    assert roughness.events[0]["actual_duration_s"] == pytest.approx(0.001)
    simulator.data.qpos[:3] = simulator.R_world_chute @ np.array([0.05, 0.1, 0.1])
    simulator.data.qvel[:] = 0
    applied = _force_step(simulator, roughness)
    assert not np.any(applied)
    assert len(roughness.events) == 1
    event = roughness.events[0]
    assert event["pulse_status"] == "contact_lost"
    assert event["actual_duration_s"] == pytest.approx(0.001)
    assert event["impulse_ns"] < event["configured_impulse_ns"]


def test_weight_force_continues_geometric_contact_through_zero_solver_load(weight_cube, monkeypatch):
    config, part = weight_cube
    simulator, roughness = _loaded_weight_cube(config, part)
    roughness.weight_force.next_start_s = 0.0
    _force_step(simulator, roughness)
    assert roughness.events[0]["contact_normal_force_n"] > 0
    # Model transient zero solver load without altering the real touching
    # geometry. Start remains load-gated; continuation is geometry-gated.
    monkeypatch.setattr(mujoco, "mj_contactForce", lambda model, data, index, force: force.fill(0))
    assert roughness.weight_force._contacts(simulator.data) == []
    applied = _force_step(simulator, roughness)
    assert np.linalg.norm(applied) > 0
    event = roughness.events[0]
    assert event["pulse_status"] == "active"
    assert event["actual_duration_s"] == pytest.approx(0.002)
    assert event["zero_load_duration_s"] == pytest.approx(0.001)
    assert event["last_contact_normal_force_n"] == 0
    simulator, roughness = _loaded_weight_cube(config, part)
    roughness.weight_force.next_start_s = 0.0
    assert not np.any(_force_step(simulator, roughness))
    assert roughness.events == []


def test_weight_force_stops_at_geometric_gap_even_if_margin_keeps_contact_rows(weight_cube):
    config, part = weight_cube
    simulator, roughness = _loaded_weight_cube(config, part)
    roughness.weight_force.next_start_s = 0.0
    _force_step(simulator, roughness)
    # A solver margin can retain contact rows for separated objects. It must
    # not turn them into physical support for a continuing force pulse.
    simulator.model.pair_margin[:] = 0.0001
    simulator.data.qpos[:3] = simulator.R_world_chute @ np.array([0.05, 0.010005, 0.010005])
    simulator.data.qvel[:] = 0
    mujoco.mj_forward(simulator.model, simulator.data)
    assert simulator.data.ncon > 0
    assert all(contact.dist > 1e-7 for contact in simulator.data.contact)
    assert not np.any(_force_step(simulator, roughness))
    assert roughness.events[0]["pulse_status"] == "contact_lost"
    # The same separated margin contacts cannot start a new pulse either,
    # even when the contact solver assigns them a positive normal force.
    roughness = ContactRoughness(config, simulator.model, simulator.R_world_chute,
                                simulator.floor_id, simulator.wall_id, 123)
    roughness.weight_force.next_start_s = simulator.data.time
    assert not np.any(_force_step(simulator, roughness))
    assert roughness.events == []


def test_weight_force_tracks_chute_contact_from_wall_to_belt_and_logs_change(weight_cube, monkeypatch):
    config, part = weight_cube
    simulator, roughness = _loaded_weight_cube(config, part)
    roughness.weight_force.next_start_s = 0.0
    original_contacts = roughness.weight_force._contacts

    def wall_starts(data, geom_id=None, *, loaded_only=True):
        contacts = original_contacts(data, geom_id, loaded_only=loaded_only)
        if loaded_only and geom_id is None:
            return [contact for contact in contacts if contact.surface == "wall"]
        return contacts

    monkeypatch.setattr(roughness.weight_force, "_contacts", wall_starts)
    _force_step(simulator, roughness)
    event = roughness.events[0]
    assert event["surface"] == event["current_surface"] == "wall"
    assert event["contact_surface_changes"] == []
    # Remove wall support while retaining an actual belt contact. The finite
    # chute disturbance transfers its lever arm to that real contact point.
    simulator.data.qpos[:3] = simulator.R_world_chute @ np.array([0.05, 0.03, 0.00999])
    simulator.data.qvel[:] = 0
    assert np.linalg.norm(_force_step(simulator, roughness)) > 0
    assert event["surface"] == "wall"
    assert event["current_surface"] == "belt"
    assert event["contact_surface_changes"] == [{"time_s": 0.001, "surface": "belt"}]
    assert event["pulse_status"] == "active"
    assert event["actual_duration_s"] == pytest.approx(0.002)
    assert abs(event["last_contact_pos_chute_mm"][2]) < 0.01
    assert event["last_contact_pos_chute_mm"][1] > 10


def test_due_force_events_in_flight_are_skipped_without_triggering_on_landing(weight_cube, monkeypatch):
    config, part = weight_cube
    simulator, roughness = _loaded_weight_cube(config, part)
    roughness.weight_force.next_start_s = 0.0
    monkeypatch.setattr(roughness.weight_force, "_wait", lambda: 0.02)
    simulator.data.qpos[:3] = simulator.R_world_chute @ np.array([0.05, 0.1, 0.1])
    assert not np.any(_force_step(simulator, roughness))
    assert roughness.events == []
    assert roughness.weight_force.next_start_s == 0.02
    simulator.data.qpos[:3] = simulator.R_world_chute @ np.array([0.05, 0.00999, 0.00999])
    simulator.data.qvel[:] = 0
    assert not np.any(_force_step(simulator, roughness))
    assert roughness.events == []


def test_half_timestep_preserves_weight_force_duration_and_actual_impulse(weight_cube):
    config, part = weight_cube
    impulses = []
    for dt in (0.001, 0.0005):
        config.timestep_s = dt
        simulator, roughness = _loaded_weight_cube(config, part)
        # Both start and end are between integration boundaries at either dt.
        roughness.weight_force.next_start_s = 0.0003
        integrated_force = np.zeros(3)
        for _ in range(round(0.008 / dt)):
            applied = _force_step(simulator, roughness)
            integrated_force += simulator.R_chute_world @ applied[:3] * dt
            assert simulator.data.qfrc_applied == pytest.approx(np.zeros(simulator.model.nv), abs=1e-17)
        assert len(roughness.events) == 1
        event = roughness.events[0]
        assert event["pulse_status"] == "completed"
        assert event["actual_duration_s"] == pytest.approx(config.roughness_force_duration_s)
        assert event["end_time_s"] == pytest.approx(0.0003 + config.roughness_force_duration_s)
        assert event["impulse_ns"] == pytest.approx(event["configured_impulse_ns"])
        assert event["impulse_chute_ns"] == pytest.approx(integrated_force, abs=1e-14)
        impulses.append(event["impulse_ns"])
    assert impulses[0] == pytest.approx(impulses[1], abs=1e-14)


def test_weight_force_seeded_replay_and_stationary_contact_events(weight_cube):
    config, part = weight_cube
    config.roughness_force_interval_s = 0.01
    results = []
    for seed in (123, 456, 123):
        simulator, roughness = _loaded_weight_cube(config, part, seed)
        for _ in range(200):
            _force_step(simulator, roughness)
        roughness.finalize(simulator.data)
        results.append((roughness.events, simulator.data.qpos.copy()))
    assert len(results[0][0]) > 2
    assert results[0][0] == results[2][0]
    assert np.array_equal(results[0][1], results[2][1])
    assert results[0][0] != results[1][0]
    assert {event["surface"] for event in results[0][0]} == {"belt", "wall"}
    for previous, current in zip(results[0][0], results[0][0][1:]):
        assert current["time_s"] > previous["end_time_s"]
    for event in results[0][0]:
        assert event["contact_normal_force_n"] > 0
        assert event["impulse_chute_ns"][1:] == pytest.approx([0, 0], abs=1e-14)
        assert event["actual_duration_s"] <= config.roughness_force_duration_s + 1e-14


def test_weight_force_finalize_records_partial_trial_and_zero_fraction_is_inactive(weight_cube):
    config, part = weight_cube
    simulator, roughness = _loaded_weight_cube(config, part)
    roughness.weight_force.next_start_s = 0.0
    _force_step(simulator, roughness)
    roughness.finalize(simulator.data)
    assert roughness.events[0]["pulse_status"] == "trial_end"
    assert roughness.events[0]["actual_duration_s"] == pytest.approx(0.001)
    config.roughness_force_fraction = 0
    simulator, roughness = _loaded_weight_cube(config, part)
    roughness.weight_force.next_start_s = 0.0
    assert not np.any(_force_step(simulator, roughness))
    assert roughness.events == []
