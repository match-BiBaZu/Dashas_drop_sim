"""Local paired Kk1a sensitivity study; never exports or publishes a catalogue.

Run from the repository: python -m experiments.kk1a_study --trials 128
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dashas_drop_sim.cli import _config
from dashas_drop_sim.catalog_batch import engine_fingerprint
from dashas_drop_sim.geometry import prepare_part
from dashas_drop_sim.physics import ChuteSimulator
from dashas_drop_sim.roughness import ContactRoughness
from dashas_drop_sim.runner import wilson_interval


class ObservedSimulator(ChuteSimulator):
    def __init__(self, config, part, overrides):
        self.overrides = overrides
        super().__init__(config, part)

    def _build_model(self):
        if not self.overrides.get("analytic_cylinders"):
            return super()._build_model()
        if self.part.source_sha256 != "b8417afd1da3e03481f282abec1b30b43f3d1e7e8f4d93d07a58f8c92ef2669b":
            raise ValueError("Analytic cylinders require the verified original Kk1a STL.")
        # Capture the original, unrounded scene. The shared builder stays unchanged.
        original = ET.tostring
        captured = []
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            captured.append(result)
            return result
        with patch("dashas_drop_sim.physics.ET.tostring", side_effect=capture):
            super()._build_model()
        root = ET.fromstring(captured[-1])
        body = root.find("./worldbody/body[@name='part']")
        for index, (radius, half_length, z) in enumerate(((0.010, 0.0125, 12.5), (0.012, 0.0025, 27.5))):
            geom = body.find(f"geom[@name='part_collision_{index}']")
            geom.attrib.pop("mesh")
            geom.set("type", "cylinder")
            geom.set("size", f"{radius} {half_length}")
            geom.set("pos", " ".join(str(x) for x in (np.array([12, 12, z]) - self.part.center_mass_source_mm) / 1000))
        for geom in list(body.findall("geom")):
            if geom.get("name", "").startswith("part_collision_") and int(geom.get("name").rsplit("_", 1)[-1]) >= 2:
                body.remove(geom)
        pairs = root.find("contact")
        for pair in list(pairs):
            if pair.get("geom1", "").startswith("part_collision_") and int(pair.get("geom1").rsplit("_", 1)[-1]) >= 2:
                pairs.remove(pair)
        files = [self.part.visual_mesh_path, *self.part.collision_mesh_paths]
        assets = {f"asset_{index}.obj": path.read_bytes() for index, path in enumerate(files)}
        return mujoco.MjModel.from_xml_string(original(root, encoding="unicode"), assets=assets)

    def _window_is_settled(self, history):
        if history and (not self.observations or history[-1][0] - self.observations[-1][0] >= 0.04):
            sample = history[-1]
            self.observations.append((sample[0], sample[1].tolist(), sample[2].tolist(),
                                      self.data.qvel[3:6].tolist()))
        return super()._window_is_settled(history)

    def drop(self, *args, **kwargs):
        self.observations = []
        return super().drop(*args, **kwargs)


SIM = None


class DiagnosticPulses(ContactRoughness):
    """Artificial contact impulses to measure a threshold, not a roughness model.

    J = body mass * equivalent delta-v, opposed to wall longitudinal slip.
    The surface-frame energy check identifies unphysical impulses explicitly.
    Encounters use motion of the footprint, also covering rolling without slip.
    """
    amplitude_mm_s = 0.0
    cutoff_s = float("inf")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pulse_rng = np.random.default_rng(np.random.SeedSequence([args[-1], 0xD1A6]))
        self.next_distance = self.pulse_rng.exponential(0.01)
        self.last_position = None

    def advance(self, data, elapsed_s):
        super().advance(data, elapsed_s)
        if data.time > self.cutoff_s:
            return
        position = self.rotation.T @ data.xipos[self.body_id]
        if self.last_position is None:
            self.last_position = position.copy()
            return
        self.next_distance -= abs(position[0] - self.last_position[0])
        self.last_position = position.copy()
        if self.next_distance > 0:
            return
        self.next_distance += self.pulse_rng.exponential(0.01)
        surface = next((surface for surface in self.surfaces if surface.name == "wall"), None)
        if surface is None:
            return
        contacts = self._contacts(data, surface)
        if not contacts:
            return
        weights = np.array([row[4] for row in contacts])
        point, jacobian, relative, speed, load = contacts[self.pulse_rng.choice(len(contacts), p=weights / weights.sum())]
        slip = float(relative @ self.longitudinal)
        if abs(slip) < 1e-6:
            return
        magnitude = self.model.body_mass[self.body_id] * self.amplitude_mm_s / 1000
        impulse = -np.sign(slip) * magnitude * self.longitudinal
        response = np.zeros_like(jacobian)
        mujoco.mj_solveM(self.model, data, response, np.ascontiguousarray(jacobian))
        mobility = jacobian @ response.T
        energy = float(impulse @ relative + 0.5 * impulse @ mobility @ impulse)
        data.qvel[:] += response.T @ impulse
        self.events.append({"surface": "diagnostic_wall", "impulse_ns": magnitude,
                            "relative_longitudinal_speed_mm_s": slip * 1000,
                            "relative_energy_change_j": energy})
        mujoco.mj_forward(self.model, data)


class FrictionPatches(ContactRoughness):
    """Seeded tangential friction variation advected by loaded contact footprints.

    Independent convex-part contacts sample smooth, bounded coefficient changes.
    Only MuJoCo's friction coefficients change; no prescribed forces are added.
    The amplitudes are hypotheses and have not been inferred from scratch depth.
    """
    wall_amplitude = 0.0
    belt_amplitude = 0.0
    spacing_mm = 10.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config, model, _, _, _, seed = args
        self.wall_id = args[4]
        self.states = {}
        self.belt_distance = 0.0
        self.means = {self.wall_id: config.mu_wall, self.floor_id: config.mu_belt}
        for index in range(model.npair):
            one, two = model.pair_geom1[index], model.pair_geom2[index]
            surface = one if one in self.means else two
            mean = self.means[surface]
            model.pair_friction[index, :2] = mean
            rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5F1, index]))
            amplitude = self.wall_amplitude if surface == self.wall_id else self.belt_amplitude
            self.states[index] = {"rng": rng, "mean": mean, "amplitude": amplitude, "surface": surface,
                                  "previous": None, "distance": 0, "a": mean,
                                  "b": float(rng.uniform(max(0, mean - amplitude), mean + amplitude))}

    def advance(self, data, elapsed_s):
        super().advance(data, elapsed_s)
        self.belt_distance += self.model.geom_surfacevel[self.floor_id, 0] * elapsed_s
        for index, state in self.states.items():
            if state["amplitude"] == 0:
                continue
            one, two = self.model.pair_geom1[index], self.model.pair_geom2[index]
            contacts = []
            for ci, contact in enumerate(data.contact):
                if contact.efc_address < 0 or {contact.geom1, contact.geom2} != {one, two}:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, data, ci, force)
                if force[0] > self.load_threshold:
                    contacts.append((float((self.rotation.T @ contact.pos)[0]), float(force[0])))
            if not contacts:
                state["previous"] = None
                continue
            coordinate = sum(x * f for x, f in contacts) / sum(f for _, f in contacts)
            if state["surface"] == self.floor_id:
                coordinate -= self.belt_distance
            if state["previous"] is not None:
                state["distance"] += abs(coordinate - state["previous"])
            state["previous"] = coordinate
            spacing = self.spacing_mm / 1000
            while state["distance"] >= spacing:
                state["distance"] -= spacing
                state["a"] = state["b"]
                state["b"] = float(state["rng"].uniform(max(0, state["mean"] - state["amplitude"]), state["mean"] + state["amplitude"]))
            blend = (1 - np.cos(np.pi * state["distance"] / spacing)) / 2
            self.model.pair_friction[index, :2] = (1 - blend) * state["a"] + blend * state["b"]
        mujoco.mj_forward(self.model, data)


def initialize(config, part, overrides):
    global SIM
    SIM = ObservedSimulator(config, part, overrides)
    if "torsion_mm" in overrides:
        SIM.model.pair_friction[:, 2] = overrides["torsion_mm"] / 1000
    if "rolling_mm" in overrides:
        SIM.model.pair_friction[:, 3:5] = overrides["rolling_mm"] / 1000
    if overrides.get("elliptic"):
        SIM.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    if "condim" in overrides:
        SIM.model.pair_dim[:] = overrides["condim"]
    if "noslip_iterations" in overrides:
        SIM.model.opt.noslip_iterations = overrides["noslip_iterations"]
    if "impratio" in overrides:
        SIM.model.opt.impratio = overrides["impratio"]
    if "solref_s" in overrides:
        SIM.model.pair_solref[:, 0] = overrides["solref_s"]
    if "diagnostic_dv_mm_s" in overrides:
        import dashas_drop_sim.physics as physics
        DiagnosticPulses.amplitude_mm_s = overrides["diagnostic_dv_mm_s"]
        DiagnosticPulses.cutoff_s = overrides.get("diagnostic_cutoff_s", float("inf"))
        physics.ContactRoughness = DiagnosticPulses
    if "patch_wall_amplitude" in overrides:
        import dashas_drop_sim.physics as physics
        FrictionPatches.wall_amplitude = overrides["patch_wall_amplitude"]
        FrictionPatches.belt_amplitude = overrides.get("patch_belt_amplitude", 0)
        FrictionPatches.spacing_mm = overrides.get("patch_spacing_mm", 10)
        physics.ContactRoughness = FrictionPatches


def drop(job):
    index, seed = job
    outcome = SIM.drop(index, seed)
    quat = np.array(outcome.final_quat_xyzw)
    axis = Rotation.from_quat(quat).apply([0, 0, 1])
    group = "longitudinal" if abs(axis[0]) >= 0.9 else "transverse" if abs(axis[0]) <= 0.3 else "oblique"
    observations = SIM.observations
    recent = [row for row in observations if row[0] >= outcome.sim_time_s - 0.5]
    axes = Rotation.from_quat([row[1] for row in recent]).apply([0, 0, 1]) if recent else np.array([axis])
    axis_motion = np.degrees(np.arccos(np.clip(axes @ axes[-1], -1, 1))).max()
    spin = np.mean([abs(row[3][2]) for row in recent]) if recent else abs(SIM.data.qvel[5])
    tilt = np.mean([np.linalg.norm(row[3][:2]) for row in recent]) if recent else np.linalg.norm(SIM.data.qvel[3:5])
    impulse_by_surface = {surface: [event for event in outcome.roughness_events if event["surface"] == surface]
                          for surface in ("wall", "belt")}
    row = {"trial": index, "seed": seed, "status": outcome.status, "axis_group": group,
           "final_axis_chute": axis.tolist(), "final_quat_xyzw": quat.tolist(),
           "final_pos_chute_mm": outcome.final_pos_chute_mm,
           "initial_quat_xyzw": outcome.initial_quat_xyzw, "sim_time_s": outcome.sim_time_s,
           "first_settled_s": outcome.first_settled_s,
           "last_window_axis_motion_deg": float(axis_motion), "mean_spin_rad_s": float(spin),
           "mean_tilt_rad_s": float(tilt),
           "diagnostic_pulses": sum(event["surface"] == "diagnostic_wall" for event in outcome.roughness_events),
           "diagnostic_positive_energy_j": sum(max(0, event["relative_energy_change_j"])
                                                 for event in outcome.roughness_events if event["surface"] == "diagnostic_wall"),
           "roughness": {surface: {"events": len(events),
                         "total_impulse_ns": sum(e["impulse_ns"] for e in events),
                         "max_impulse_ns": max((e["impulse_ns"] for e in events), default=0),
                         "mean_slip_mm_s": float(np.mean([e["relative_longitudinal_speed_mm_s"] for e in events])) if events else 0}
                         for surface, events in impulse_by_surface.items()}}
    if index < 8:
        row["trace"] = observations
    return row


CASES = [
    ("baseline", {}),
    ("smooth", {"roughness_enabled": False, "belt_speed_variation_mm_s": 0}),
    ("length1300", {"length_mm": 1300}),
    ("wall_h020", {"roughness_wall_height_mm": 0.2}),
    ("wall_h050", {"roughness_wall_height_mm": 0.5}),
    ("wall_h075", {"roughness_wall_height_mm": 0.75}),
    ("wall_spacing3", {"roughness_wall_spacing_mm": 3}),
    ("belt_h010", {"roughness_belt_height_mm": 0.1}),
    ("belt_h050", {"roughness_belt_height_mm": 0.5}),
    ("speed_fast020", {"belt_variation_interval_s": 0.02}),
    ("speed_fast050", {"belt_variation_interval_s": 0.05}),
    ("speed_amp10", {"belt_speed_variation_mm_s": 10}),
    ("microfacet", {"roughness_model": "microfacet"}),
    ("torsion010", {"torsion_mm": 0.1}),
    ("torsion050", {"torsion_mm": 0.5}),
    ("torsion000", {"torsion_mm": 0}),
    ("rolling050", {"rolling_mm": 0.5}),
    ("elliptic", {"elliptic": True}),
    ("mu_wall005", {"mu_wall": 0.05}),
    ("mu_wall010", {"mu_wall": 0.1}),
    ("mu_wall040", {"mu_wall": 0.4}),
    ("mu_wall060", {"mu_wall": 0.6}),
    ("mu_belt020", {"mu_belt": 0.2}),
    ("mu_belt060", {"mu_belt": 0.6}),
    ("drop020", {"drop_height_mm": 20}),
    ("drop200", {"drop_height_mm": 200}),
    ("diagnostic003", {"diagnostic_dv_mm_s": 3}),
    ("diagnostic010", {"diagnostic_dv_mm_s": 10}),
    ("diagnostic030", {"diagnostic_dv_mm_s": 30}),
    ("diagnostic100", {"diagnostic_dv_mm_s": 100}),
    ("diagnostic300", {"diagnostic_dv_mm_s": 300}),
    ("early030", {"diagnostic_dv_mm_s": 30, "diagnostic_cutoff_s": 3}),
    ("early100", {"diagnostic_dv_mm_s": 100, "diagnostic_cutoff_s": 3}),
    ("early150", {"diagnostic_dv_mm_s": 150, "diagnostic_cutoff_s": 3}),
    ("early200", {"diagnostic_dv_mm_s": 200, "diagnostic_cutoff_s": 3}),
    ("early300", {"diagnostic_dv_mm_s": 300, "diagnostic_cutoff_s": 3}),
    ("pitch025", {"beta_deg": 0.25}),
    ("pitch050", {"beta_deg": 0.5}),
    ("pitch100", {"beta_deg": 1}),
    ("pitch_minus050", {"beta_deg": -0.5}),
    ("roll44", {"alpha_deg": 44}),
    ("roll46", {"alpha_deg": 46}),
    ("analytic", {"analytic_cylinders": True}),
    ("analytic_smooth", {"analytic_cylinders": True, "roughness_enabled": False, "belt_speed_variation_mm_s": 0}),
    ("analytic_wall020", {"analytic_cylinders": True, "roughness_wall_height_mm": 0.2}),
    ("analytic_wall050", {"analytic_cylinders": True, "roughness_wall_height_mm": 0.5}),
    ("analytic_belt010", {"analytic_cylinders": True, "roughness_belt_height_mm": 0.1}),
    ("analytic_torsion050", {"analytic_cylinders": True, "torsion_mm": 0.5}),
    ("analytic_elliptic", {"analytic_cylinders": True, "elliptic": True}),
    ("point_contacts", {"condim": 3}),
    ("point_strongwall", {"condim": 3, "roughness_wall_height_mm": 0.5}),
    ("point_fastbelt", {"condim": 3, "belt_variation_interval_s": 0.02}),
    ("point_belt010", {"condim": 3, "roughness_belt_height_mm": 0.1}),
    ("small_angular_friction", {"torsion_mm": 0.1, "rolling_mm": 0.01}),
    ("small_angular_strongwall", {"torsion_mm": 0.1, "rolling_mm": 0.01, "roughness_wall_height_mm": 0.5}),
    ("analytic_point", {"analytic_cylinders": True, "condim": 3}),
    ("analytic_point_strongwall", {"analytic_cylinders": True, "condim": 3, "roughness_wall_height_mm": 0.5}),
    ("half_timestep", {"timestep_s": 0.0005}),
    ("analytic_half_timestep", {"analytic_cylinders": True, "timestep_s": 0.0005}),
    ("quarter_timestep", {"timestep_s": 0.00025}),
    ("noslip", {"noslip_iterations": 20}),
    ("elliptic_impratio10", {"elliptic": True, "impratio": 10}),
    ("contact_tau002", {"solref_s": 0.002}),
    ("patch_wall010", {"patch_wall_amplitude": 0.1}),
    ("patch_zero", {"patch_wall_amplitude": 0.0}),
    ("patch_wall020", {"patch_wall_amplitude": 0.2}),
    ("patch_wall020_spacing3", {"patch_wall_amplitude": 0.2, "patch_spacing_mm": 3}),
    ("patch_both", {"patch_wall_amplitude": 0.2, "patch_belt_amplitude": 0.2}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", nargs="+", choices=[case[0] for case in CASES])
    parser.add_argument("--output", type=Path, default=Path("results/kk1a_study/screening"))
    args = parser.parse_args()
    base = replace(_config(Path("configs/catalog39.json")), trials=args.trials, seed=args.seed,
                   catalog_repo=None, catalog_push=False, compute_stability=False)
    part = prepare_part(base.mesh_path, Path("results/catalog39/Kk1a/.mesh_cache"), base.density_g_cm3)
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(index, int(child.generate_state(1, dtype=np.uint32)[0]))
            for index, child in enumerate(np.random.SeedSequence(base.seed).spawn(base.trials))]
    provenance = {"engine_fingerprint": engine_fingerprint(), "source_sha256": part.source_sha256,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "created_utc": datetime.now(timezone.utc).isoformat(), "mujoco_version": mujoco.__version__}
    summaries = []
    baseline_rows = None
    context = multiprocessing.get_context("spawn")
    for name, raw in CASES:
        if args.cases and name not in args.cases:
            continue
        path = args.output / (name + ".json")
        overrides = {key: value for key, value in raw.items() if key not in base.__dataclass_fields__}
        config = replace(base, **{key: value for key, value in raw.items() if key in base.__dataclass_fields__})
        config.validate()
        signature = hashlib.sha256(json.dumps({"config": config.to_dict(), "overrides": overrides,
                                               "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, sort_keys=True).encode()).hexdigest()
        print(json.dumps({"event": "case_started", "case": name}), flush=True)
        started = time.perf_counter()
        if path.is_file() and json.loads(path.read_text())["signature"] == signature:
            rows = json.loads(path.read_text())["trials"]
        else:
            with ProcessPoolExecutor(max_workers=8, mp_context=context, initializer=initialize,
                                     initargs=(config, part, overrides)) as executor:
                rows = [future.result() for future in as_completed([executor.submit(drop, job) for job in jobs])]
            rows.sort(key=lambda row: row["trial"])
            assert len(rows) == base.trials and all(row["status"] != "invalid" for row in rows)
            path.write_text(json.dumps({"signature": signature, "config": config.to_dict(), "overrides": overrides,
                                       "provenance": provenance, "trials": rows}, indent=2), encoding="utf-8")
        count = sum(row["status"] in {"settled", "settled_stationary"} for row in rows)
        longitudinal = sum(row["axis_group"] == "longitudinal" for row in rows)
        transverse = sum(row["axis_group"] == "transverse" for row in rows)
        summary = {"case": name, "seed": base.seed, "trials": len(rows), "settled": count,
                   "longitudinal_percent": 100 * longitudinal / len(rows),
                   "transverse_percent": 100 * transverse / len(rows),
                   "transverse_ci95_percent": [100 * x for x in wilson_interval(transverse, len(rows))],
                   "settled_percent": count / len(rows) * 100,
                   "settled_ci95_percent": [100 * x for x in wilson_interval(count, len(rows))],
                   "status_counts": dict(Counter(row["status"] for row in rows)),
                   "axis_status_counts": dict(Counter(row["axis_group"] + ":" + row["status"] for row in rows)),
                   "wall_events_mean": float(np.mean([row["roughness"]["wall"]["events"] for row in rows])),
                   "diagnostic_positive_energy_j": sum(row["diagnostic_positive_energy_j"] for row in rows),
                   "elapsed_s": time.perf_counter() - started, "parameters": raw}
        if name == "baseline":
            baseline_rows = rows
        elif baseline_rows:
            assert all(row["initial_quat_xyzw"] == ref["initial_quat_xyzw"] and row["seed"] == ref["seed"]
                       for row, ref in zip(rows, baseline_rows))
            summary["paired_gained_settled"] = sum(row["status"] == "settled" and ref["status"] != "settled" for row, ref in zip(rows, baseline_rows))
            summary["paired_lost_settled"] = sum(row["status"] != "settled" and ref["status"] == "settled" for row, ref in zip(rows, baseline_rows))
            summary["paired_gained_longitudinal"] = sum(row["axis_group"] == "longitudinal" and ref["axis_group"] != "longitudinal" for row, ref in zip(rows, baseline_rows))
            summary["paired_lost_longitudinal"] = sum(row["axis_group"] != "longitudinal" and ref["axis_group"] == "longitudinal" for row, ref in zip(rows, baseline_rows))
        summaries.append(summary)
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
