"""Contact, reset, and convergence checks for the MuJoCo chute."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from dashas_drop_sim.config import RunConfig
from dashas_drop_sim.geometry import PreparedPart, prepare_part
from dashas_drop_sim.physics import ChuteSimulator


@pytest.fixture
def cube(tmp_path: Path) -> tuple[Path, PreparedPart]:
    source = tmp_path / "cube.stl"
    trimesh.creation.box(extents=[20, 20, 20]).export(source)
    return source, prepare_part(source, tmp_path / "mesh-cache", 1.15)


def _simulator(
    cube: tuple[Path, PreparedPart],
    *,
    belt_speed_mm_s: float = 100,
    mu_wall: float = 0.2,
    length_mm: float = 300,
    drop_height_mm: float = 0,
    timestep_s: float = 0.001,
    alpha_deg: float = 45,
    mu_belt: float = 0.4,
    contact_time_constant_s: float = 0.005,
    contact_damping_ratio: float = 1.2,
    contact_torsional_friction_mm: float = 5.0,
    contact_rolling_friction_mm: float = 0.1,
) -> ChuteSimulator:
    source, part = cube
    config = RunConfig(
        mesh_path=source,
        output_dir=source.parent,
        belt_speed_mm_s=belt_speed_mm_s,
        mu_belt=mu_belt,
        mu_wall=mu_wall,
        length_mm=length_mm,
        drop_height_mm=drop_height_mm,
        timestep_s=timestep_s,
        alpha_deg=alpha_deg,
        contact_time_constant_s=contact_time_constant_s,
        contact_damping_ratio=contact_damping_ratio,
        contact_torsional_friction_mm=contact_torsional_friction_mm,
        contact_rolling_friction_mm=contact_rolling_friction_mm,
    )
    config.validate()
    return ChuteSimulator(config, part)


def _place_cube_in_corner(simulator: ChuteSimulator) -> None:
    simulator._spawn(np.array([0.0, 0.0, 0.0, 1.0]))
    # Leave a one-millimetre clearance to avoid starting with penetration.
    simulator.data.qpos[:3] = simulator.R_world_chute @ np.array([0.05, 0.011, 0.011])
    mujoco.mj_forward(simulator.model, simulator.data)


def _advance(simulator: ChuteSimulator, seconds: float) -> None:
    for _ in range(round(seconds / simulator.config.timestep_s)):
        mujoco.mj_step(simulator.model, simulator.data)


def _pair_sliding_mu(simulator: ChuteSimulator, surface_id: int) -> float:
    matching = [
        index
        for index in range(simulator.model.npair)
        if surface_id in (simulator.model.pair_geom1[index], simulator.model.pair_geom2[index])
    ]
    assert len(matching) == 1  # The convex cube has one collision mesh.
    pair_index = matching[0]
    # MuJoCo's pair default for a literal zero friction is 1; a frictionless
    # contact therefore has condim=1 and no tangential friction directions.
    if simulator.model.pair_dim[pair_index] == 1:
        return 0.0
    return float(simulator.model.pair_friction[pair_index, 0])


def test_belt_surface_velocity_transports_cube_at_100_but_not_zero_mm_s(
    cube: tuple[Path, PreparedPart],
) -> None:
    stationary = _simulator(cube, belt_speed_mm_s=0)
    moving = _simulator(cube, belt_speed_mm_s=100)
    np.testing.assert_allclose(stationary.model.geom_surfacevel[stationary.floor_id], 0)
    assert moving.model.geom_surfacevel[moving.floor_id, 0] == pytest.approx(0.1)
    np.testing.assert_allclose(moving.model.geom_surfacevel[moving.wall_id], 0)

    for simulator in (stationary, moving):
        _place_cube_in_corner(simulator)
        _advance(simulator, 0.5)
        assert simulator._contact_flags(simulator.data)[0]

    still_x = stationary._local_position(stationary.data)[0]
    moving_x = moving._local_position(moving.data)[0]
    assert abs(still_x - 0.05) < 0.0001
    assert moving_x - still_x > 0.02


def test_zero_belt_speed_uses_time_window_instead_of_exit(cube: tuple[Path, PreparedPart]) -> None:
    stationary = _simulator(cube, belt_speed_mm_s=0, length_mm=300)
    _place_cube_in_corner(stationary)
    outcome = stationary._run_until_end(
        trial=0, seed=0, initial_quat=np.array([0.0, 0.0, 0.0, 1.0])
    )
    assert outcome.status == "settled_stationary"
    assert outcome.sim_time_s == pytest.approx(10.0, abs=0.01)
    assert abs(outcome.travel_mm) < 1


def test_wall_friction_is_independent_and_changes_transport(
    cube: tuple[Path, PreparedPart],
) -> None:
    low_wall = _simulator(cube, mu_wall=0)
    high_wall = _simulator(cube, mu_wall=1.0)
    for simulator, expected_wall in ((low_wall, 0), (high_wall, 1.0)):
        assert _pair_sliding_mu(simulator, simulator.floor_id) == pytest.approx(0.4)
        assert _pair_sliding_mu(simulator, simulator.wall_id) == pytest.approx(expected_wall)
        _place_cube_in_corner(simulator)
        _advance(simulator, 0.5)

    assert low_wall._local_position(low_wall.data)[0] > high_wall._local_position(high_wall.data)[0] + 0.02


def test_seeded_drops_are_repeatable_after_full_reset(
    cube: tuple[Path, PreparedPart],
) -> None:
    simulator = _simulator(cube, length_mm=150, drop_height_mm=10)
    first = simulator.drop(0, 123)
    other_seed = simulator.drop(1, 124)
    repeated = simulator.drop(2, 123)

    assert first.initial_quat_xyzw != other_seed.initial_quat_xyzw
    assert first.status == repeated.status
    assert first.sim_time_s == pytest.approx(repeated.sim_time_s, abs=1e-12)
    np.testing.assert_allclose(first.initial_quat_xyzw, repeated.initial_quat_xyzw, atol=1e-12)
    np.testing.assert_allclose(first.final_qpos, repeated.final_qpos, atol=1e-12)
    np.testing.assert_allclose(first.final_pos_chute_mm, repeated.final_pos_chute_mm, atol=1e-9)

    simulator._spawn(np.array([0.0, 0.0, 0.0, 1.0]))
    assert simulator.data.time == 0
    np.testing.assert_allclose(simulator.data.qvel, 0)


def test_sampled_release_is_applied_and_leaves_orientation_stream_unchanged(cube):
    simulator = _simulator(cube, length_mm=300, drop_height_mm=100)
    fixed = simulator.drop(0, 123, cancel=lambda: True)
    simulator.config.drop_height_spread_mm = 20
    simulator.config.lateral_spread_mm = 10
    sampled = simulator.drop(0, 123, cancel=lambda: True)
    repeated = simulator.drop(0, 123, cancel=lambda: True)
    assert sampled.to_dict() == repeated.to_dict()
    assert sampled.initial_quat_xyzw == fixed.initial_quat_xyzw
    assert 80 <= sampled.actual_drop_height_mm <= 120
    assert -10 <= sampled.actual_lateral_mm <= 10
    y, z = sampled.initial_pos_chute_mm[1:]
    assert (y - z) / np.sqrt(2) == pytest.approx(sampled.actual_lateral_mm)
    simulator._spawn(np.array(sampled.initial_quat_xyzw), drop_height_mm=0,
                     lateral_mm=sampled.actual_lateral_mm)
    baseline = simulator._local_position(simulator.data) * 1000
    assert np.array(sampled.initial_pos_chute_mm) - baseline == pytest.approx(
        [0, sampled.actual_drop_height_mm / np.sqrt(2), sampled.actual_drop_height_mm / np.sqrt(2)])


def test_quiet_transport_settles_but_a_short_spinning_drop_does_not(
    cube: tuple[Path, PreparedPart],
) -> None:
    quiet = _simulator(cube, length_mm=300)
    _place_cube_in_corner(quiet)
    quiet_result = quiet._run_until_end(
        trial=0, seed=0, initial_quat=np.array([0.0, 0.0, 0.0, 1.0])
    )
    assert quiet_result.status == "settled"
    assert quiet_result.first_settled_s is not None
    assert quiet_result.travel_mm > 200

    spinning = _simulator(cube, length_mm=100, drop_height_mm=200)
    spinning._spawn(np.array([0.0, 0.0, 0.0, 1.0]))
    spinning.data.qvel[3] = 20.0
    spinning_result = spinning._run_until_end(
        trial=1, seed=0, initial_quat=np.array([0.0, 0.0, 0.0, 1.0])
    )
    assert spinning_result.status == "unsettled"
    assert spinning_result.first_settled_s is None


def test_half_timestep_preserves_cube_trajectory(
    cube: tuple[Path, PreparedPart],
) -> None:
    outcomes: list[tuple[np.ndarray, Rotation]] = []
    for timestep_s in (0.001, 0.0005):
        simulator = _simulator(cube, timestep_s=timestep_s)
        _place_cube_in_corner(simulator)
        _advance(simulator, 0.5)
        outcomes.append(
            (
                simulator._local_position(simulator.data),
                Rotation.from_quat(simulator._local_quat(simulator.data)),
            )
        )

    (position_normal, angle_normal), (position_half, angle_half) = outcomes
    assert abs(position_normal[0] - position_half[0]) < 0.002
    assert np.linalg.norm(position_normal[1:] - position_half[1:]) < 0.001
    assert (angle_normal.inv() * angle_half).magnitude() < np.deg2rad(1)


@pytest.mark.parametrize("timestep_s", [0.001, 0.0005])
def test_lower_contact_damping_makes_a_real_drop_rebound(
    cube: tuple[Path, PreparedPart], timestep_s: float,
) -> None:
    """Measure the rebound, without treating solref as exact restitution."""
    rebound_heights: list[float] = []
    for damping_ratio in (1.2, 0.5):
        simulator = _simulator(
            cube, alpha_deg=0, belt_speed_mm_s=0, timestep_s=timestep_s,
            contact_damping_ratio=damping_ratio,
        )
        simulator._spawn(np.array([0.0, 0.0, 0.0, 1.0]))
        # Flat cube, 100 mm floor clearance, well away from the wall.
        simulator.data.qpos[:3] = [0.05, 0.05, 0.11]
        mujoco.mj_forward(simulator.model, simulator.data)
        has_hit = False
        max_rebound = 0.0
        for _ in range(round(0.7 / timestep_s)):
            mujoco.mj_step(simulator.model, simulator.data)
            floor, wall = simulator._contact_flags(simulator.data)
            assert not wall
            has_hit |= floor
            if has_hit:
                # The cube's half-height is 10 mm; symmetry preserves its pose.
                max_rebound = max(max_rebound, float(simulator.data.qpos[2]) - 0.01)
        assert has_hit
        assert np.all(np.isfinite(simulator.data.qpos))
        rebound_heights.append(max_rebound)

    legacy_rebound, lower_damping_rebound = rebound_heights
    assert legacy_rebound < 0.001
    assert lower_damping_rebound > 0.005
    assert lower_damping_rebound > legacy_rebound + 0.005


def test_torsional_friction_opposes_spin_at_a_floor_contact(
    cube: tuple[Path, PreparedPart],
) -> None:
    spin_speeds: list[float] = []
    for torsional_mm in (0.0, 5.0):
        simulator = _simulator(
            cube, alpha_deg=0, belt_speed_mm_s=0, mu_belt=0.01,
            timestep_s=0.0005, contact_torsional_friction_mm=torsional_mm,
            contact_rolling_friction_mm=0,
        )
        simulator._spawn(np.array([0.0, 0.0, 0.0, 1.0]))
        simulator.data.qpos[:3] = [0.05, 0.05, 0.0105]
        mujoco.mj_forward(simulator.model, simulator.data)
        _advance(simulator, 0.3)
        assert simulator._contact_flags(simulator.data) == (True, False)
        simulator.data.qvel[5] = 5.0
        mujoco.mj_forward(simulator.model, simulator.data)
        _advance(simulator, 0.05)
        spin_speeds.append(abs(float(simulator.data.qvel[5])))

    # Low sliding friction isolates the extra rotational resistance.
    assert spin_speeds[0] > 3.0
    assert spin_speeds[1] < 0.01


@pytest.mark.parametrize(
    ("torsional_mm", "rolling_mm", "expected_dim"),
    [(0.0, 0.0, 3), (2.0, 0.0, 4), (2.0, 0.3, 6)],
)
def test_rotational_contact_friction_uses_millimetres_and_can_be_disabled(
    cube: tuple[Path, PreparedPart], torsional_mm: float,
    rolling_mm: float, expected_dim: int,
) -> None:
    simulator = _simulator(
        cube, contact_torsional_friction_mm=torsional_mm,
        contact_rolling_friction_mm=rolling_mm,
        contact_time_constant_s=0.008, contact_damping_ratio=0.7,
    )
    assert np.all(simulator.model.pair_dim == expected_dim)
    np.testing.assert_allclose(simulator.model.pair_solref, [[0.008, 0.7]] * 2)
    if expected_dim >= 4:
        np.testing.assert_allclose(simulator.model.pair_friction[:, 2], torsional_mm / 1000)
    if expected_dim == 6:
        np.testing.assert_allclose(simulator.model.pair_friction[:, 3:], rolling_mm / 1000)
