"""Assess each basin member, preserve microstates, export stable representatives."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from .grouping import analyze_groups, choose_representative
from .observed import ObservedPoseResolver
from .parallel import SimulationPool, kick_job
from .runner import (SETTLED_STATUSES, _write_json, _write_jsonl, _write_csv,
                     disturbance_directions, _ranking, wilson_interval)


def retained_group(source_group, outcome, resolver):
    if outcome.status not in SETTLED_STATUSES:
        return False
    for quat in (*outcome.settled_trace_quat_xyzw, outcome.final_quat_xyzw):
        match = resolver.resolve(quat)
        if match.status != "matched" or match.pose_id != source_group:
            return False
    return True


def assess_and_export(config, simulator, resolver, rows, run_dir, cancel, emit):
    counts = Counter(row["pose_key"] for row in rows if row["status"] in SETTLED_STATUSES)
    first = {}
    for row in sorted(rows, key=lambda r: r["trial"]):
        if row["status"] in SETTLED_STATUSES and row["pose_key"] != "ambiguous":
            first.setdefault(row["pose_key"], row)
    baseline = getattr(resolver, "data", {"poses": [], "recognition": resolver.recognition_descriptor()})
    originals = {str(p["id"]): p for p in baseline["poses"]}
    members = []
    for key in sorted(set(originals) | set(first)):
        original = originals.get(key)
        observed = first.get(key)
        quat = original["quaternion_xyzw"] if original else observed["final_quat_xyzw"]
        position = original.get("position_com_chute_mm") if original else observed["final_pos_chute_mm"]
        if original and position is not None:
            world = Rotation.from_matrix(simulator.R_world_chute @ Rotation.from_quat(quat).as_matrix()).as_quat()
            qpos = [*(simulator.R_world_chute @ (np.asarray(position) / 1000)), world[3], *world[:3]]
        elif observed:
            qpos = observed["final_qpos"]
            position = observed["final_pos_chute_mm"]
        else:
            raise ValueError(f"Pose {key} has neither a recorded position nor a new settled state.")
        members.append({"id": key, "quaternion_xyzw": list(quat),
                        "reference_quaternions_xyzw": original.get("reference_quaternions_xyzw", [quat]) if original else [quat],
                        "position_com_chute_mm": list(position), "qpos": list(qpos),
                        "count": counts[key], "source_trial": observed["trial"] if observed else None})
    descriptor = resolver.recognition_descriptor()
    analysis = analyze_groups(simulator.part.catalog_mesh_path, members, descriptor,
                              alpha_deg=config.alpha_deg, beta_deg=config.beta_deg,
                              barrier_mm=config.grouping_barrier_mm,
                              max_angle_deg=config.grouping_max_angle_deg)
    mapping = {key: f"group_{i:03d}" for i, group in enumerate(analysis["groups"]) for key in group}
    direct = ObservedPoseResolver(simulator.part.catalog_mesh_path, data={"poses": members, "recognition": descriptor})
    grouped = direct.grouped(mapping)
    _write_json(run_dir / "grouping_geometry.json", analysis)
    levels = config.disturbance_levels_mm
    jobs, member_by_id = [], {m["id"]: m for m in members}
    for member in members:
        # Source-ID-based seeds keep assessment order and process scheduling irrelevant.
        seed_index = int.from_bytes(hashlib.sha256(member["id"].encode()).digest()[:4], "little") & 0x7fffffff
        for level in levels:
            directions = disturbance_directions()[:1] if level == 0 else disturbance_directions()
            for direction_index, direction in enumerate(directions):
                jobs.append((len(jobs), member["id"], seed_index, tuple(member["qpos"]), level, direction_index, direction))
    progress_path = run_dir / "group_disturbances.jsonl"
    completed = []
    if progress_path.is_file():
        lines = progress_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            try:
                completed.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                break
        _write_jsonl(progress_path, completed)
    done_ids = {r["disturbance_index"] for r in completed}
    if len(done_ids) != len(completed) or any(i < 0 or i >= len(jobs) for i in done_ids):
        raise ValueError("Invalid disturbance checkpoint.")
    remaining = [job for job in jobs if job[0] not in done_ids]
    with progress_path.open("a", encoding="utf-8") as stream:
        def record(job, outcome):
            ordinal, key, _, _, level, direction_index, direction = job
            result = grouped.resolve(outcome.final_quat_xyzw) if outcome.status in SETTLED_STATUSES else None
            item = {"disturbance_index": ordinal, "source_pose_id": key, "group_id": mapping[key],
                    "energy_lift_mm": level, "energy_j": simulator.part.mass_kg * 9.81 * level / 1000,
                    "direction_index": direction_index, "direction_chute": direction,
                    "retained": retained_group(mapping[key], outcome, grouped),
                    "individual_retained": retained_group(key, outcome, direct),
                    "result_status": outcome.status, "result_group_id": result.pose_id if result and result.status == "matched" else None,
                    "result_quat_xyzw": outcome.final_quat_xyzw,
                    "settled_trace_quat_xyzw": outcome.settled_trace_quat_xyzw,
                    "roughness_seed": outcome.seed, "source_trial": member_by_id[key]["source_trial"]}
            completed.append(item)
            stream.write(json.dumps(item) + "\n")
            stream.flush()
            emit({"event": "stability_progress", "pose_id": key, "completed": len(completed), "total": len(jobs)})
        if config.workers > 1 and jobs and not cancel():
            with SimulationPool(config, simulator.part) as pool:
                for job, outcome in pool.run(remaining, kick_job, cancel):
                    record(job, outcome)
        else:
            for job in remaining:
                if cancel():
                    break
                outcome = simulator.kick(job[3], job[4], np.asarray(job[6]), index=job[2], cancel=cancel)
                if outcome.status != "cancelled":
                    record(job, outcome)
    completed.sort(key=lambda r: r["disturbance_index"])
    _write_jsonl(progress_path, completed)
    curves = []
    for key in member_by_id:
        for level in levels:
            items = [r for r in completed if r["source_pose_id"] == key and r["energy_lift_mm"] == level]
            expected = 1 if level == 0 else 12
            curves.append({"pose_id": key, "group_id": mapping[key], "energy_lift_mm": level,
                           "energy_j": simulator.part.mass_kg * 9.81 * level / 1000,
                           "retained_count": sum(r["retained"] for r in items), "direction_count": len(items),
                           "retention": sum(r["retained"] for r in items) / len(items) if items else 0.,
                           "individual_retention": sum(r["individual_retained"] for r in items) / len(items) if items else 0.,
                           "complete": len(items) == expected})
    ranking = _ranking(curves, levels)
    scores = {r["pose_id"]: r["retention_auc"] for r in ranking}
    threshold = {r["pose_id"]: r["retention"] for r in curves if r["energy_lift_mm"] == config.grouping_barrier_mm}
    groups = []
    for i, keys in enumerate(analysis["groups"]):
        representative = choose_representative(keys, scores, counts, threshold)
        count = sum(counts[key] for key in keys)
        low, high = wilson_interval(count, len(rows))
        group = {"pose_id": f"group_{i:03d}", "members": keys, "representative_source_pose_id": representative,
                 "count": count, "probability": count / len(rows) if rows else 0., "ci_low": low, "ci_high": high,
                 "retention_auc": scores[representative] if representative else None,
                 "complete": representative is not None, "category": "pose",
                 "retained_by_frequency": count * 100 / len(rows) >= config.grouping_minimum_frequency_percent if rows else False}
        if representative:
            group["representative_quat_xyzw"] = member_by_id[representative]["quaternion_xyzw"]
            group["representative_pos_chute_mm"] = member_by_id[representative]["position_com_chute_mm"]
        groups.append(group)
    groups.sort(key=lambda g: (g["retention_auc"] is None, -(g["retention_auc"] or 0), g["pose_id"]))
    for rank, group in enumerate((g for g in groups if g["complete"]), 1):
        group["rank"] = rank
    for member in members:
        member.update(group_id=mapping[member["id"]], retention_auc=scores[member["id"]])
        member.pop("qpos")
    _write_json(run_dir / "group_members.json", members)
    _write_json(run_dir / "groups.json", {"groups": groups, "assessment_complete": all(g["complete"] for g in groups),
                "disturbance_trials": len(completed), "planned_disturbance_trials": len(jobs), "settings": analysis})
    _write_csv(run_dir / "group_stability.csv", list(curves[0]) if curves else [], curves)
    _write_csv(run_dir / "group_stability_summary.csv", ["pose_id", "members", "representative_source_pose_id", "retention_auc", "complete", "rank"], groups)
    _write_csv(run_dir / "group_frequencies.csv", ["pose_id", "members", "representative_source_pose_id", "count", "probability", "ci_low", "ci_high", "retained_by_frequency"], groups)
    for row in rows:
        match = grouped.resolve(row["final_quat_xyzw"]) if row["status"] in SETTLED_STATUSES else None
        row["group_id"] = match.pose_id if match and match.status == "matched" else None
    _write_jsonl(run_dir / "grouped_trials.jsonl", rows)
    # An existing observed-roadmap builder can consume this representative-only catalog.
    export = run_dir / "representatives"
    export.mkdir(exist_ok=True)
    shutil.copyfile(simulator.part.catalog_mesh_path, export / f"{config.mesh_path.stem}.stl")
    poses = []
    for group in groups:
        if group["complete"] and group["retained_by_frequency"]:
            member = member_by_id[group["representative_source_pose_id"]]
            poses.append({**member, "count": group["count"], "frequency_percent": group["probability"] * 100,
                          "group_members": group["members"], "representative_retention_auc": group["retention_auc"]})
    _write_json(export / "poses.json", {"schema_version": 1, "workpiece": config.mesh_path.stem,
                "completed_trials": len(rows), "frequency_denominator": "all completed drop trials, including unsettled outcomes",
                "geometry": {"stl": f"{config.mesh_path.stem}.stl", "stl_sha256": hashlib.sha256((export / f"{config.mesh_path.stem}.stl").read_bytes()).hexdigest()},
                "recognition": descriptor, "poses": poses, "grouping": analysis,
                "assessment_complete": all(g["complete"] for g in groups)})
    _write_json(export / "config.json", config.to_dict())
    return {"pose_groups": groups, "group_stability_curves": curves,
            "group_disturbance_trials": len(completed), "group_assessment_complete": all(g["complete"] for g in groups)}
