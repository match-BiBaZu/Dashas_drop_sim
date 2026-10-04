"""Physical scaling, contact gating and reproducibility of micro-impacts."""

import mujoco
import numpy as np
import pytest
import trimesh

from dashas_drop_sim.config import RunConfig
from dashas_drop_sim.geometry import prepare_part
from dashas_drop_sim.physics import ChuteSimulator
from dashas_drop_sim.roughness import ContactRoughness, facet_impulse


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


@pytest.fixture
def rough_cube(tmp_path):
    path = tmp_path / "cube.stl"
    trimesh.creation.box(extents=[20, 20, 20]).export(path)
    part = prepare_part(path, tmp_path / "cache", 1.15)
    config = RunConfig(mesh_path=path, output_dir=tmp_path / "results", length_mm=300,
                       drop_height_mm=0, workers=1, roughness_enabled=True)
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
