"""Run and persist drop and disturbance experiments without changing a roadmap."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from . import __version__
from .config import RunConfig
from .geometry import PreparedPart, prepare_part
from .physics import ChuteSimulator, TrialOutcome
from .pose_matching import PoseResolver
from .parallel import SimulationPool, drop_job, kick_job


EventCallback = Callable[[dict[str, Any]], None]
CancelCallback = Callable[[], bool]
SETTLED_STATUSES = frozenset({"settled", "settled_stationary"})


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, fields: list[str], records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value) if isinstance(value, (tuple, list, dict)) else value
                             for key, value in record.items()})
    temporary.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided 95% binomial interval for an observed pose fraction."""
    if trials <= 0:
        return 0.0, 1.0
    p = successes / trials
    z2 = z * z
    center = (p + z2 / (2 * trials)) / (1 + z2 / trials)
    half = z * np.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / (1 + z2 / trials)
    return float(max(0, center - half)), float(min(1, center + half))


def disturbance_directions() -> list[tuple[float, float, float]]:
    """The twelve vertices of a regular icosahedron in chute coordinates."""
    phi = (1 + np.sqrt(5)) / 2
    raw = [(0, s, t * phi) for s in (-1, 1) for t in (-1, 1)]
    raw += [(s, t * phi, 0) for s in (-1, 1) for t in (-1, 1)]
    raw += [(t * phi, 0, s) for s in (-1, 1) for t in (-1, 1)]
    return [tuple(float(x) for x in np.asarray(v) / np.linalg.norm(v)) for v in raw]


def _new_run_dir(output_dir: Path, seed: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    for suffix in range(1000):
        candidate = output_dir / (f"run_{stamp}_{seed}" + ("" if suffix == 0 else f"_{suffix}"))
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("Could not allocate a unique run directory")


@dataclass(slots=True)
class RunResult:
    run_dir: Path
    summary: dict[str, Any]


def _pose_key(row: dict[str, Any]) -> str:
    if row["status"] not in SETTLED_STATUSES:
        return str(row["status"])
    if row["match_status"] == "matched":
        return str(row["roadmap_pose_id"])
    if row["match_status"] == "ambiguous":
        return "ambiguous"
    return str(row["pose_key"])


def _assign_unknown_clusters(rows: list[dict[str, Any]], resolver: PoseResolver) -> None:
    unknown = [row for row in rows if row["status"] in SETTLED_STATUSES and row["match_status"] == "unmatched"]
    labels = resolver.cluster_unmatched([row["final_quat_xyzw"] for row in unknown])
    for row, label in zip(unknown, labels):
        row["pose_key"] = f"unknown_{label + 1:03d}"


def _frequency_rows(rows: list[dict[str, Any]], total: int,
                    known_pose_ids: tuple[int, ...] = ()) -> list[dict[str, Any]]:
    counts = Counter(_pose_key(row) for row in rows)
    known_keys = {str(pose_id) for pose_id in known_pose_ids}
    for key in known_keys:
        counts.setdefault(key, 0)
    first: dict[str, dict[str, Any]] = {}
    for row in rows:
        first.setdefault(_pose_key(row), row)
    result: list[dict[str, Any]] = []
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        low, high = wilson_interval(count, total)
        source = first.get(key)
        result.append({
            "pose_id": key,
            "category": "pose" if key in known_keys
            or (source is not None and source["status"] in SETTLED_STATUSES and key != "ambiguous")
            else "unassigned",
            "count": count,
            "probability": count / total,
            "ci_low": low,
            "ci_high": high,
        })
    return result


def _orientation_matches_source(source: dict[str, Any], quat: tuple[float, ...],
                                resolver: PoseResolver) -> bool:
    if source["match_status"] == "matched":
        target = resolver.resolve(quat)
        return target.status == "matched" and target.pose_id == source["roadmap_pose_id"]
    if resolver.resolve(quat).status != "unmatched":
        return False
    labels = resolver.cluster_unmatched([source["final_quat_xyzw"], quat])
    return labels[0] == labels[1]


def _same_pose(source: dict[str, Any], outcome: TrialOutcome, resolver: PoseResolver) -> bool:
    if outcome.status not in SETTLED_STATUSES:
        return False
    # A settled detour into another orientation is a loss even if the part
    # happens to return to its source pose by the exit.
    trace = getattr(outcome, "settled_trace_quat_xyzw", ())
    return all(_orientation_matches_source(source, quat, resolver) for quat in (*trace, outcome.final_quat_xyzw))


def _stability_rows(
    rows: list[dict[str, Any]], config: RunConfig, simulator: ChuteSimulator,
    resolver: PoseResolver, cancel: CancelCallback, emit: EventCallback,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    representatives: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _pose_key(row)
        if row["status"] in SETTLED_STATUSES and key != "ambiguous":
            representatives.setdefault(key, row)
    directions = disturbance_directions()
    individual: list[dict[str, Any]] = []
    jobs: list[tuple[int, str, int, tuple[float, ...], float, int, tuple[float, float, float]]] = []
    for key, source in representatives.items():
        for level in config.disturbance_levels_mm:
            # A zero kick has one baseline trajectory; twelve copies would be
            # perfectly correlated and waste eleven full simulations.
            sample_directions = directions[:1] if level == 0 else directions
            for direction_index, direction in enumerate(sample_directions):
                jobs.append((len(jobs), key, source["trial"], tuple(source["final_qpos"]),
                             level, direction_index, direction))
    planned = len(jobs)

    def record(job, outcome: TrialOutcome) -> None:
        ordinal, key, source_trial, _source_qpos, level, direction_index, direction = job
        source = representatives[key]
        match = resolver.resolve(outcome.final_quat_xyzw) if outcome.status in SETTLED_STATUSES else None
        individual.append({
            "disturbance_index": ordinal,
            "source_pose_id": key,
            "source_trial": source_trial,
            "energy_lift_mm": level,
            "energy_j": simulator.part.mass_kg * 9.81 * level / 1000,
            "direction_index": direction_index,
            "direction_chute": direction,
            "retained": _same_pose(source, outcome, resolver),
            "result_status": outcome.status,
            "result_roadmap_pose_id": match.pose_id if match is not None and match.status == "matched" else None,
            "result_match_status": match.status if match is not None else None,
            "result_quat_xyzw": outcome.final_quat_xyzw,
            "settled_trace_quat_xyzw": outcome.settled_trace_quat_xyzw,
            "travel_mm": outcome.travel_mm,
        })
        emit({"event": "stability_progress", "pose_id": key,
              "completed": len(individual), "total": planned})

    if config.workers > 1 and planned > 1 and not cancel():
        with SimulationPool(config, simulator.part) as pool:
            for job, outcome in pool.run(jobs, kick_job, cancel):
                record(job, outcome)
    else:
        for job in jobs:
            if cancel():
                break
            ordinal, key, source_trial, source_qpos, level, direction_index, direction = job
            outcome = simulator.kick(source_qpos, level, np.asarray(direction),
                                     index=source_trial, cancel=cancel)
            if outcome.status == "cancelled":
                break
            record(job, outcome)
    individual.sort(key=lambda row: row["disturbance_index"])
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in individual:
        grouped.setdefault((row["source_pose_id"], row["energy_lift_mm"]), []).append(row)
    curves: list[dict[str, Any]] = []
    for (key, level), group in sorted(grouped.items()):
        count = sum(bool(item["retained"]) for item in group)
        curves.append({
            "pose_id": key, "energy_lift_mm": level,
            "energy_j": group[0]["energy_j"],
            "retained_count": count, "direction_count": len(group),
            "retention": count / len(group),
            "complete": len(group) == (1 if level == 0 else 12),
        })
    return individual, curves


def _ranking(curves: list[dict[str, Any]], levels: tuple[float, ...]) -> list[dict[str, Any]]:
    """Area under the sampled retention curve, with no binary cut-off."""
    by_pose: dict[str, dict[float, dict[str, Any]]] = {}
    for row in curves:
        by_pose.setdefault(row["pose_id"], {})[row["energy_lift_mm"]] = row
    result = []
    for key, found in by_pose.items():
        complete = all(level in found and found[level]["complete"] for level in levels)
        xs = np.asarray([level for level in levels if level in found], dtype=float)
        ys = np.asarray([found[level]["retention"] for level in levels if level in found], dtype=float)
        if complete and len(xs) >= 2 and xs[-1] > 0:
            auc = float(np.sum((ys[1:] + ys[:-1]) * np.diff(xs) / 2) / xs[-1])
        else:
            auc = None
        result.append({"pose_id": key, "retention_auc": auc, "complete": complete})
    result.sort(key=lambda row: (row["retention_auc"] is None,
                                 -(row["retention_auc"] or 0), row["pose_id"]))
    for rank, row in enumerate((row for row in result if row["retention_auc"] is not None), 1):
        row["rank"] = rank
    return result


def build_manifest(config: RunConfig, part: PreparedPart, resolver: PoseResolver) -> dict[str, Any]:
    """Shared provenance for a batch or a visible single trial."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mesh_source_sha256": part.source_sha256,
        "mesh_triangles_sha256": part.mesh_sha256,
        "roadmap_sha256": _sha256(config.roadmap_path),
        "mass_kg": part.mass_kg,
        "inertia_com_kg_m2": part.inertia_com_kg_m2.tolist(),
        "center_mass_source_mm": part.center_mass_source_mm.tolist(),
        "catalog_mesh_path": str(part.catalog_mesh_path),
        "collision_quality": part.collision_quality,
        "catalog_symmetry": resolver.symmetry_symbol,
        "catalogue_cache_hit": resolver.cache_hit,
        "symmetry_available": resolver.symmetry_available,
        "classification_tolerance_deg": resolver.match_tolerance_deg,
        "classification_ambiguity_margin_deg": resolver.ambiguity_margin_deg,
        "unknown_cluster_tolerance_deg": resolver.cluster_tolerance_deg,
        "software": {
            "dashas-drop-sim": __version__, "mujoco": mujoco.__version__,
            "coacd": _version("coacd"), "trimesh": _version("trimesh"),
            "numpy": np.__version__, "scipy": _version("scipy"),
            "bibazu-chute-pose": _version("bibazu-chute-pose"),
            "cadquery-ocp": _version("cadquery-ocp"),
        },
    }


def run_experiment(
    config: RunConfig, *, emit: EventCallback | None = None,
    cancel: CancelCallback | None = None,
) -> RunResult:
    config.validate()
    emit = emit or (lambda _event: None)
    cancel = cancel or (lambda: False)
    run_dir = _new_run_dir(config.output_dir, config.seed)
    _write_json(run_dir / "config.json", config.to_dict())
    part = prepare_part(config.mesh_path, config.output_dir / ".mesh_cache", config.density_g_cm3)
    resolver = PoseResolver(part.catalog_mesh_path, config.roadmap_path,
                            original_mesh_path=config.mesh_path,
                            cache_dir=config.output_dir / ".pose_cache")
    simulator = ChuteSimulator(config, part)
    _write_json(run_dir / "manifest.json", build_manifest(config, part, resolver))

    child_seeds = np.random.SeedSequence(config.seed).spawn(config.trials)
    jobs = [(index, int(child.generate_state(1, dtype=np.uint32)[0]))
            for index, child in enumerate(child_seeds)]
    rows: list[dict[str, Any]] = []
    with (run_dir / "trials.jsonl").open("w", encoding="utf-8") as stream:
        def record_drop(outcome: TrialOutcome) -> None:
            match = resolver.resolve(outcome.final_quat_xyzw) if outcome.status in SETTLED_STATUSES else None
            row = outcome.to_dict()
            row.update({
                "match_status": match.status if match else "not_evaluated",
                "roadmap_pose_id": match.pose_id if match else None,
                "match_distance_deg": match.distance_deg if match else None,
                "pose_key": None,
            })
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            emit({"event": "progress", "completed": len(rows), "total": config.trials,
                  "status": outcome.status})
        if config.workers > 1 and len(jobs) > 1 and not cancel():
            with SimulationPool(config, part) as pool:
                for _job, outcome in pool.run(jobs, drop_job, cancel):
                    record_drop(outcome)
        else:
            for index, trial_seed in jobs:
                if cancel():
                    break
                outcome = simulator.drop(index, trial_seed, cancel=cancel)
                if outcome.status == "cancelled":
                    break
                record_drop(outcome)

    rows.sort(key=lambda row: row["trial"])
    _assign_unknown_clusters(rows, resolver)
    for row in rows:
        row["pose_key"] = _pose_key(row)
    _write_jsonl(run_dir / "trials.jsonl", rows)
    _write_csv(run_dir / "trials.csv", [
        "trial", "seed", "status", "pose_key", "match_status", "roadmap_pose_id",
        "match_distance_deg", "initial_quat_xyzw", "final_quat_xyzw", "final_pos_chute_mm",
        "sim_time_s", "travel_mm", "first_settled_s", "final_floor_contact",
        "final_wall_contact", "final_qpos",
    ], rows)
    frequencies = _frequency_rows(rows, len(rows), resolver.known_pose_ids) if rows else []
    example_by_key = {row["pose_key"]: row for row in rows
                      if row["status"] in SETTLED_STATUSES and row["pose_key"] != "ambiguous"}
    for frequency in frequencies:
        key = frequency["pose_id"]
        example = example_by_key.get(key)
        if example is not None:
            frequency["representative_quat_xyzw"] = example["final_quat_xyzw"]
            frequency["representative_pos_chute_mm"] = example["final_pos_chute_mm"]
            frequency["representative_source"] = "observed"
        elif key in {str(pose_id) for pose_id in resolver.known_pose_ids}:
            frequency["representative_quat_xyzw"] = resolver.reference_quaternion(int(key))
            frequency["representative_pos_chute_mm"] = None
            frequency["representative_source"] = "catalogue"
    _write_csv(run_dir / "frequencies.csv", [
        "pose_id", "category", "count", "probability", "ci_low", "ci_high",
        "representative_quat_xyzw", "representative_pos_chute_mm", "representative_source",
    ], frequencies)
    for frequency in frequencies:
        emit({"event": "result", **frequency})

    disturbances, curves = _stability_rows(rows, config, simulator, resolver, cancel, emit)
    ranking = _ranking(curves, config.disturbance_levels_mm)
    _write_csv(run_dir / "disturbances.csv", [
        "disturbance_index", "source_pose_id", "source_trial", "energy_lift_mm", "energy_j", "direction_index",
        "direction_chute", "retained", "result_status", "result_roadmap_pose_id",
        "result_match_status", "result_quat_xyzw", "settled_trace_quat_xyzw", "travel_mm",
    ], disturbances)
    _write_csv(run_dir / "stability.csv", [
        "pose_id", "energy_lift_mm", "energy_j", "retained_count", "direction_count",
        "retention", "complete",
    ], curves)
    _write_csv(run_dir / "stability_summary.csv", ["pose_id", "retention_auc", "complete", "rank"], ranking)
    summary = {
        "requested_trials": config.trials,
        "completed_trials": len(rows),
        "cancelled": bool(cancel()),
        "pose_frequencies": frequencies,
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "settled_trials": sum(row["status"] in SETTLED_STATUSES for row in rows),
        "stability_curves": curves,
        "stability_ranking": ranking,
        "disturbance_trials": len(disturbances),
        "interpretation": f"Fractions are conditional on the configured release and virtual {config.length_mm / 1000:g} m chute; friction values are uncalibrated assumptions.",
    }
    _write_json(run_dir / "summary.json", summary)
    emit({"event": "done", "run_dir": str(run_dir), "summary": str(run_dir / "summary.json"),
          "pose_frequencies": frequencies})
    return RunResult(run_dir, summary)
