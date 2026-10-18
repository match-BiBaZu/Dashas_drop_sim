"""A physical belt stop must preserve transport results and detect real rest."""

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh

from dashas_drop_sim.config import RunConfig
from dashas_drop_sim.geometry import prepare_part
from dashas_drop_sim.physics import ChuteSimulator


@pytest.fixture
def setup(tmp_path):
    mesh = tmp_path / "cube.stl"
    trimesh.creation.box(extents=[20, 20, 20]).export(mesh)
    part = prepare_part(mesh, tmp_path / "cache", 1.15)
    cfg = RunConfig(mesh, length_mm=300, drop_height_mm=0, workers=1,
                    belt_stop_enabled=True, belt_stop_ramp_s=0.2, belt_stop_observation_s=1,
                    belt_speed_variation_mm_s=3, roughness_enabled=True,
                    roughness_model="weight_impulse", roughness_impulse_delta_v_mm_s=10)
    cfg.validate()
    return cfg, part


def test_stop_adds_real_observation_preserves_transport_and_replays(setup):
    cfg, part = setup
    control = ChuteSimulator(replace(cfg, belt_stop_enabled=False), part).drop(0, 123)
    sim = ChuteSimulator(cfg, part)
    outcome = sim.drop(0, 123)
    observation = outcome.stop_observation
    assert observation is not None
    before, after = control.to_dict(), outcome.to_dict()
    after['stop_observation'] = None
    assert before == after  # All poses, impulse events and transport timestamps unchanged.
    assert observation['start_time_s'] == outcome.sim_time_s
    assert observation['stopped_at_s'] == pytest.approx(outcome.sim_time_s + 0.2)
    assert observation['end_time_s'] == pytest.approx(outcome.sim_time_s + 1.2)
    assert observation['final_belt_speed_mm_s'] == 0
    assert observation['status'] == 'settled'
    assert observation['coast_distance_mm'] > 0
    assert np.all(sim.model.geom_surfacevel[sim.wall_id] == 0)
    sim.drop(1, 124)
    assert sim.drop(0, 123).to_dict() == outcome.to_dict()
    sim._spawn(np.array([0., 0., 0., 1.]))
    assert sim.model.geom_surfacevel[sim.floor_id, 0] == 0.1


def test_frictionless_sliding_is_not_rest_and_velocity_is_not_reset(setup):
    cfg, part = setup
    sim = ChuteSimulator(replace(cfg, mu_belt=0, mu_wall=0), part)
    sim._spawn(np.array([0., 0., 0., 1.]))
    sim.data.qpos[:3] = sim.R_world_chute @ np.array([0.05, 0.01, 0.01])
    sim.data.qvel[:3] = sim.R_world_chute[:, 0] * 0.1
    mujoco.mj_forward(sim.model, sim.data)
    observation = sim._observe_belt_stop()
    assert observation['status'] == 'unsettled'
    assert observation['orientation_change_deg'] < 0.01
    assert observation['coast_distance_mm'] == pytest.approx(120, abs=0.1)
    assert (sim.R_chute_world @ sim.data.qvel[:3])[0] == pytest.approx(0.1)


def test_stop_cancellation_is_reported_and_does_not_wait_full_duration(setup):
    cfg, part = setup
    sim = ChuteSimulator(cfg, part)
    sim._spawn(np.array([0., 0., 0., 1.]))
    observation = sim._observe_belt_stop(cancel=lambda: sim.data.time >= 0.05)
    assert observation['status'] == 'cancelled'
    assert observation['stopped_at_s'] is None
    assert observation['end_time_s'] < 0.1


def test_stability_kicks_do_not_include_stop_observation(setup, monkeypatch):
    cfg, part = setup
    sim = ChuteSimulator(cfg, part)
    drop = sim.drop(0, 123)
    monkeypatch.setattr(sim, '_observe_belt_stop', lambda **kwargs: pytest.fail('stop during kick'))
    outcome = sim.kick(drop.final_qpos, 0, np.array([1., 0., 0.]))
    assert outcome.stop_observation is None


@pytest.mark.parametrize('kwargs', [
    {'belt_stop_enabled': 1}, {'belt_stop_ramp_s': 0},
    {'belt_stop_observation_s': 0.1}, {'belt_stop_observation_s': float('nan')},
])
def test_invalid_stop_config_rejected(kwargs):
    with pytest.raises(ValueError):
        RunConfig(Path('unused.stl'), **kwargs).validate_parameters()
