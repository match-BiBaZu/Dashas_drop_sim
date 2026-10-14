"""An original tall catalogue part must actually tip under a contact impulse."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dashas_drop_sim import catalog
from dashas_drop_sim.config import RunConfig
from dashas_drop_sim.geometry import PreparedPart, prepare_part
from dashas_drop_sim.physics import ChuteSimulator
from dashas_drop_sim.roughness import ContactRoughness


# Original Dk1i, standing on source Z=0's triangular end face after a 0.5 s
# seating run. This is world XYZ + MuJoCo WXYZ, with alpha=45°, beta=0°.
# The small tilt is deliberate: neither the control nor the impulse case gets
# an idealized box, synthesized collision shape, or different initial state.
START_QPOS_WORLD = np.array([
    0.08492878602400349, -0.00498781731548984, 0.013159229468163045,
    0.9238122877499402, 0.3828451920826222,
    -0.0005790339082709761, 0.0002839386802101388,
])


def offline_dk1i_path() -> Path:
    """The bundled catalogue fallback; independent of the live GUI catalogue."""
    return Path(catalog.__file__).resolve().parent / "assets" / "catalog" / "Dk1i.STL"


@pytest.fixture(scope="module")
def triangular_part(tmp_path_factory) -> tuple[Path, PreparedPart]:
    source = offline_dk1i_path()
    part = prepare_part(source, tmp_path_factory.mktemp("dk1i-collision-cache"), 1.15)
    return source, part


@dataclass
class TippingRun:
    rows: list[dict]
    events: list[dict]
    reached_exit: bool
    config: RunConfig
    part: PreparedPart


def run_controlled_tipping(
    source: Path, part: PreparedPart, *, impulse_enabled: bool,
) -> TippingRun:
    """Exercise the production prepare/step/finish API with one scheduled hit."""
    config = RunConfig(
        mesh_path=source, output_dir=source.parent,
        catalog_repo=None, catalog_cad_dir=None, catalog_push=False,
        compute_stability=False, workers=1, trials=1,
        alpha_deg=45, beta_deg=0, length_mm=1300,
        belt_speed_mm_s=100, belt_speed_variation_mm_s=0, drop_height_mm=0,
        mu_belt=0.4, mu_wall=0.2,
        timestep_s=0.001, contact_time_constant_s=0.005,
        contact_damping_ratio=1.2,
        contact_torsional_friction_mm=5.0, contact_rolling_friction_mm=0.1,
        roughness_enabled=impulse_enabled, roughness_model="weight_impulse",
        roughness_impulse_delta_v_mm_s=490.5,
    )
    config.validate()
    simulator = ChuteSimulator(config, part)
    simulator.data.qpos[:7] = START_QPOS_WORLD
    simulator.data.qpos[3:7] /= np.linalg.norm(simulator.data.qpos[3:7])
    simulator.data.qvel[:3] = simulator.R_world_chute[:, 0] * 0.1
    simulator.data.qvel[3:] = 0
    mujoco.mj_forward(simulator.model, simulator.data)
    initial_axis = Rotation.from_quat(simulator._local_quat(simulator.data)).apply([0, 0, 1])
    roughness = None
    if impulse_enabled:
        roughness = ContactRoughness(
            config, simulator.model, simulator.R_world_chute,
            simulator.floor_id, simulator.wall_id, 123,
        )
        roughness.weight_impulse.next_start_s = 1.0
        roughness.weight_impulse._wait = lambda: 1000.0
    rows = []
    for step in range(round(20 / config.timestep_s)):
        if roughness is not None:
            roughness.prepare_step(simulator.data, config.timestep_s)
        mujoco.mj_step(simulator.model, simulator.data)
        if roughness is not None:
            roughness.finish_step(simulator.data, config.timestep_s)
        if step % 10 == 0:
            quat = simulator._local_quat(simulator.data)
            axis = Rotation.from_quat(quat).apply([0, 0, 1])
            angle = float(np.degrees(np.arccos(np.clip(initial_axis @ axis, -1, 1))))
            rows.append({
                "time_s": float(simulator.data.time),
                "x_mm": float(simulator._local_position(simulator.data)[0] * 1000),
                "long_axis_change_deg": angle,
                "quat_chute_xyzw": quat.tolist(),
                "qpos_world": simulator.data.qpos.tolist(),
                "qvel": simulator.data.qvel.tolist(),
                "contacts": int(simulator.data.ncon),
            })
        if simulator._local_position(simulator.data)[0] >= simulator.exit_x_m:
            break
    reached_exit = bool(simulator._local_position(simulator.data)[0] >= simulator.exit_x_m)
    if roughness is not None:
        roughness.finalize(simulator.data)
    return TippingRun(rows, roughness.events if roughness else [], reached_exit, config, part)


def test_production_contact_impulse_tips_dk1i_and_preserves_control_pose(triangular_part):
    source, part = triangular_part
    quiet = run_controlled_tipping(source, part, impulse_enabled=False)
    hit = run_controlled_tipping(source, part, impulse_enabled=True)

    assert quiet.reached_exit and hit.reached_exit
    assert not quiet.events
    assert len(hit.events) == 1
    event = hit.events[0]
    assert event["model"] == "weight_impulse"
    assert event["time_s"] == pytest.approx(1.0, abs=0.002)
    assert event["impulse_ns"] == pytest.approx(part.mass_kg * 0.4905, rel=1e-10)
    assert event["contact_normal_force_n"] > 0
    assert event["body_energy_change_j"] > 0

    # A large angular change after the kick, and retained through the exit,
    # proves a physical pose change rather than an event counter or marker.
    assert max(row["long_axis_change_deg"] for row in quiet.rows) < 1
    assert max(row["long_axis_change_deg"] for row in hit.rows if row["time_s"] < 1) < 1
    assert max(row["long_axis_change_deg"] for row in hit.rows if row["time_s"] > 1.1) > 75
    assert hit.rows[-1]["long_axis_change_deg"] > 75
    assert quiet.rows[-1]["long_axis_change_deg"] < 1
