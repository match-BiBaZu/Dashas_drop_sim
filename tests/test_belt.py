"""Bounded, time-correlated conveyor motion and replay after resetting a trial."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from dashas_drop_sim.belt import BeltSpeedProfile
from dashas_drop_sim.config import RunConfig
from dashas_drop_sim.geometry import prepare_part
from dashas_drop_sim.physics import ChuteSimulator


def test_profile_is_smooth_bounded_and_independent_of_sample_order():
    config = RunConfig(Path("unused.stl"), belt_speed_variation_mm_s=3,
                       belt_variation_interval_s=0.2)
    profile = BeltSpeedProfile(config, 123)
    times = np.arange(0, 2, 0.001)
    speeds = [profile.speed_mm_s(t) for t in times]
    assert min(speeds) >= 97 and max(speeds) <= 103
    assert speeds[0] == 100
    assert max(speeds) - min(speeds) > 1
    coarse = BeltSpeedProfile(config, 123)
    coarse.speed_mm_s(10)  # Visiting later times first cannot change earlier samples.
    for t in times[::10]:
        assert coarse.speed_mm_s(t) == profile.speed_mm_s(t)
    for point in profile.control_points()[1:-1]:
        t = point["elapsed_s"]
        assert profile.speed_mm_s(t) == pytest.approx(point["speed_mm_s"])
        assert abs(profile.speed_mm_s(t - 1e-6) - profile.speed_mm_s(t + 1e-6)) < 1e-8
    config.belt_speed_variation_mm_s = 0
    disabled = BeltSpeedProfile(config, 123)
    assert disabled.speed_mm_s(100) == 100 and disabled.control_points() == ()


def test_speed_variation_replays_resets_and_applies_only_to_belt(tmp_path):
    path = tmp_path / "cube.stl"
    trimesh.creation.box(extents=[20, 20, 20]).export(path)
    part = prepare_part(path, tmp_path / "cache", 1.15)
    config = RunConfig(path, length_mm=300, drop_height_mm=0, workers=1,
                       belt_speed_variation_mm_s=3, roughness_enabled=True,
                       roughness_belt_height_mm=0.1)
    config.validate()
    simulator = ChuteSimulator(config, part)
    first = simulator.drop(0, 123)
    simulator.drop(1, 124)
    repeated = simulator.drop(0, 123)
    assert first.to_dict() == repeated.to_dict()
    assert first.belt_speed_control_points
    assert all(97 <= point["speed_mm_s"] <= 103 for point in first.belt_speed_control_points)
    assert np.all(simulator.model.geom_surfacevel[simulator.wall_id] == 0)
    assert 0.097 <= simulator.model.geom_surfacevel[simulator.floor_id, 0] <= 0.103
    assert any(event["belt_speed_mm_s"] != 100 for event in first.roughness_events)
    assert all(event["relative_energy_change_j"] <= 1e-12 for event in first.roughness_events)
    simulator._spawn(np.array([0., 0., 0., 1.]))
    assert simulator.model.geom_surfacevel[simulator.floor_id, 0] == 0.1


@pytest.mark.parametrize("values", [
    {"belt_speed_mm_s": 0, "belt_speed_variation_mm_s": 3},
    {"belt_speed_mm_s": 199, "belt_speed_variation_mm_s": 3},
    {"belt_variation_interval_s": 0.001},
])
def test_invalid_speed_profile_is_rejected(values):
    with pytest.raises(ValueError):
        RunConfig(Path("unused.stl"), **values).validate_parameters()
