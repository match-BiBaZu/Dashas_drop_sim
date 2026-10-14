"""Portable grouped handover exports; never runs physics or publishes Git changes."""
from __future__ import annotations

from collections import Counter
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .catalog_export import sha256, model_pair
from .grouping import choose_representative
from .runner import SETTLED_STATUSES, _write_json, _write_csv, wilson_interval


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jsonl(path):
    path = Path(path)
    compressed = not path.is_file()
    opener = gzip.open if compressed else open
    with opener(path.with_suffix(path.suffix + ".gz") if compressed else path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def payload_sha256(path):
    path = Path(path)
    opener = open if path.is_file() else gzip.open
    selected = path if path.is_file() else path.with_suffix(path.suffix + ".gz")
    digest = hashlib.sha256()
    with opener(selected, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_completed_run(run, output_root):
    """Losslessly compress bulky records after export, preserving full trial data."""
    run, output_root = Path(run).resolve(), Path(output_root).resolve()
    if not run.is_relative_to(output_root):
        raise ValueError("Only completed runs inside this study may be archived.")
    receipt = read(run / "archive.json") if (run / "archive.json").is_file() else {}
    for name in ("trials.jsonl", "trials.csv", "grouped_trials.jsonl", "roughness_events.jsonl", "roughness_events.csv"):
        source = run / name
        if not source.is_file():
            continue
        digest = sha256(source)
        dest = source.with_suffix(source.suffix + ".gz")
        temp = dest.with_suffix(dest.suffix + ".tmp")
        with source.open("rb") as src, gzip.open(temp, "wb", compresslevel=3) as sink:
            shutil.copyfileobj(src, sink, 1024 * 1024)
        recovered = hashlib.sha256()
        with gzip.open(temp, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                recovered.update(block)
        if recovered.hexdigest() != digest:
            raise ValueError("Lossless raw-data compression verification failed.")
        temp.replace(dest)
        receipt[name] = {"payload_sha256": digest, "original_bytes": source.stat().st_size,
                         "gzip_sha256": sha256(dest), "gzip_bytes": dest.stat().st_size}
        _write_json(run / "archive.json", receipt)
        source.unlink()
    return receipt


def inventory(folder):
    _write_json(folder / "files.sha256.json", {
        p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "files.sha256.json"})


def verify_inventory(folder):
    expected = read(folder / "files.sha256.json")
    names = {p.relative_to(folder).as_posix() for p in folder.rglob("*")
             if p.is_file() and p.name != "files.sha256.json"}
    if names != set(expected) or any(sha256(folder / p) != h for p, h in expected.items()):
        raise ValueError(f"Export inventory mismatch: {folder}")


def verify_grouped_catalog(folder):
    data = read(folder / "poses.json")
    original = read(folder / "original_observations.json")
    members = {p["id"]: p for p in original["poses"]}
    registry = read(folder / "pose_registry.json")
    if {p["id"] for p in data["poses"]} != {p["id"] for p in registry["entries"]}:
        raise ValueError("Current registry contains non-representative poses.")
    all_ids = [i for p in members.values() for i in p["trial_ids"]]
    if len(all_ids) != len(set(all_ids)) or any(len(p["trial_ids"]) != p["count"] for p in members.values()):
        raise ValueError("Original trial identities are missing or duplicated.")
    if len(all_ids) + original["unassigned_settled_trials"] != data["settled_trials"]:
        raise ValueError("Original trial identity count is not conserved.")
    for pose in data["poses"]:
        expected = sorted(i for key in pose["group_members"] for i in members[key]["trial_ids"])
        if pose["trial_ids"] != expected or len(expected) != pose["count"]:
            raise ValueError("Combined trial identities do not match group membership.")
        if pose["individual_trial_ids"] != members[pose["id"]]["trial_ids"]:
            raise ValueError("Representative individual trials differ from original observations.")


def audit_run(run, expected_trials=10000):
    """Validate complete drop and member-assessment checkpoints before exporting."""
    run = Path(run)
    summary, config = read(run / "summary.json"), read(run / "config.json")
    trials = jsonl(run / "trials.jsonl")
    if summary.get("cancelled") or len(trials) != expected_trials or summary["completed_trials"] != expected_trials:
        raise ValueError("Expected a complete drop run.")
    if {r["trial"] for r in trials} != set(range(expected_trials)):
        raise ValueError("Drop trial IDs are missing or duplicated.")
    seeds = np.random.SeedSequence(config["seed"]).spawn(expected_trials)
    if any(r["seed"] != int(seeds[r["trial"]].generate_state(1, dtype=np.uint32)[0]) for r in trials):
        raise ValueError("Drop seeds do not match the configured sequence.")
    groups = read(run / "groups.json")
    members = read(run / "group_members.json")
    keys = [m["id"] for m in members]
    assigned = [key for g in groups["groups"] for key in g["members"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(keys):
        raise ValueError("Group memberships do not form a partition.")
    if not summary["group_assessment_complete"] or not groups["assessment_complete"]:
        raise ValueError("Incomplete member stability assessment.")
    counts = Counter(r["pose_key"] for r in trials if r["status"] in SETTLED_STATUSES)
    if any(m["count"] != counts[m["id"]] for m in members):
        raise ValueError("Original observation counts differ from drop trials.")
    if any(g["count"] != sum(counts[k] for k in g["members"]) for g in groups["groups"]):
        raise ValueError("Combined group counts are not conserved.")
    if sum(g["count"] for g in groups["groups"]) + counts["ambiguous"] != summary["settled_trials"]:
        raise ValueError("Settled count conservation failed.")
    levels = config["disturbance_levels_mm"]
    expected = {(key, level, direction) for key in keys for level in levels
                for direction in range(1 if level == 0 else 12)}
    disturbances = jsonl(run / "group_disturbances.jsonl")
    actual = [(r["source_pose_id"], r["energy_lift_mm"], r["direction_index"]) for r in disturbances]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Disturbance tests are incomplete or duplicated.")
    if {r["disturbance_index"] for r in disturbances} != set(range(len(expected))):
        raise ValueError("Disturbance checkpoint IDs are inconsistent.")
    curves = summary["group_stability_curves"]
    scores, threshold = {}, {}
    for key in keys:
        rows = sorted((r for r in curves if r["pose_id"] == key), key=lambda r: r["energy_lift_mm"])
        if [r["energy_lift_mm"] for r in rows] != levels or not all(r["complete"] for r in rows):
            raise ValueError("Incomplete stability curve.")
        scores[key] = sum((a["retention"] + b["retention"]) * (b["energy_lift_mm"] - a["energy_lift_mm"]) / 2
                          for a, b in zip(rows, rows[1:])) / levels[-1]
        threshold[key] = next(r["retention"] for r in rows if r["energy_lift_mm"] == config["grouping_barrier_mm"])
    for g in groups["groups"]:
        if choose_representative(g["members"], scores, counts, threshold) != g["representative_source_pose_id"]:
            raise ValueError("Representative selection differs from complete stability curves.")
        if g["probability"] != g["count"] / expected_trials:
            raise ValueError("Group frequency denominator changed.")
    if dict(Counter(r["status"] for r in trials)) != summary["status_counts"]:
        raise ValueError("Status counts differ from the trial records.")
    return {"completed_trials": len(trials), "settled_trials": summary["settled_trials"],
            "unassigned_settled_trials": counts["ambiguous"], "assessed_members": len(keys),
            "disturbance_tests": len(disturbances), "assessment_complete": True,
            "group_count": len(groups["groups"]),
            "eligible_groups": sum(g["retained_by_frequency"] for g in groups["groups"]),
            "trials_sha256": payload_sha256(run / "trials.jsonl"),
            "disturbances_sha256": payload_sha256(run / "group_disturbances.jsonl")}


def _replace(staged, target):
    """Commit a staged component without exposing a half-written export."""
    backup = staged.parent / "previous"
    if target.exists():
        target.rename(backup)
    try:
        staged.rename(target)
    except BaseException:
        if backup.exists():
            backup.rename(target)
        raise


def export_grouped_poses(run, repository, baseline, registry, *, cad_dir=None, provenance=None,
                         expected_trials=10000):
    run, repository = Path(run), Path(repository)
    audit = audit_run(run, expected_trials)
    summary, config, manifest = (read(run / name) for name in ("summary.json", "config.json", "manifest.json"))
    part = baseline["workpiece"]
    stl, step = model_pair(Path(config["mesh_path"]), cad_dir)
    if sha256(stl) != baseline["geometry"]["stl_sha256"] or sha256(stl) != manifest["mesh_source_sha256"]:
        raise ValueError("Grouped export geometry differs from the frozen baseline or run.")
    members = copy.deepcopy(read(run / "group_members.json"))
    groups = copy.deepcopy(read(run / "groups.json")["groups"])
    # Reserve every historical source identity. New clusters get non-recycled IDs.
    originals_registry = copy.deepcopy(registry)
    next_id = int(registry.get("next_id", 1))
    historic = {str(p["id"]) for p in registry["entries"]} | {str(p["id"]) for p in baseline["poses"]}
    identities = {}
    for member in members:
        key = member["id"]
        if key in historic:
            identities[key] = key
        else:
            while f"{part}_{next_id:04d}" in historic:
                next_id += 1
            identities[key] = f"{part}_{next_id:04d}"
            historic.add(identities[key])
            next_id += 1
    for member in members:
        member["simulation_pose_key"] = member["id"]
        member["id"] = identities[member["id"]]
    for group in groups:
        group["members"] = [identities[k] for k in group["members"]]
        group["representative_source_pose_id"] = identities[group["representative_source_pose_id"]]
    by_id = {m["id"]: m for m in members}
    member_groups = {key: g for g in groups for key in g["members"]}
    original_by_id = {str(p["id"]): p for p in baseline["poses"]}
    trial_ids = {key: [] for key in identities}
    raw_path = run / "trials.jsonl"
    opener = open if raw_path.is_file() else gzip.open
    with opener(raw_path if raw_path.is_file() else raw_path.with_suffix(".jsonl.gz"), "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["status"] in SETTLED_STATUSES and row["pose_key"] in trial_ids:
                trial_ids[row["pose_key"]].append(row["trial"])
    for member in members:
        details = copy.deepcopy(original_by_id.get(member["id"], {}))
        for stale in ("trial_ids", "representative_trial", "roadmap_pose_ids", "roadmap_match_statuses",
                      "id_assignment", "previous_candidates"):
            details.pop(stale, None)
        details.update(member)
        details["trial_ids"] = sorted(trial_ids[member["simulation_pose_key"]])
        details["representative_trial"] = member["source_trial"]
        details["anchor_origin"] = "published_baseline" if member["id"] in original_by_id else "current_drop_observation"
        details["image"] = f"images/{member['id']}.png"
        details["frequency_percent"] = 100 * member["count"] / expected_trials
        details["frequency_settled_percent"] = 100 * member["count"] / summary["settled_trials"] if summary["settled_trials"] else 0
        details["ci95_percent"] = [100 * x for x in wilson_interval(member["count"], expected_trials)]
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(member["quaternion_xyzw"]).as_matrix()
        transform[:3, 3] = np.asarray(member["position_com_chute_mm"]) - transform[:3, :3] @ manifest["center_mass_source_mm"]
        details["transform_chute_from_source_mm"] = transform.tolist()
        by_id[member["id"]] = details
    catalogue = {"schema_version": 1, "workpiece": part, "run_id": run.name,
                 "completed_trials": expected_trials, "settled_trials": summary["settled_trials"],
                 "status_counts": summary["status_counts"], "observation_length_mm": config["length_mm"],
                 "frequency_denominator": "all completed drop trials, including unsettled outcomes",
                 "geometry": {"stl": f"{part}.stl", "step": f"{part}.step", "stl_sha256": sha256(stl),
                              "step_sha256": sha256(step), "center_mass_source_mm": manifest["center_mass_source_mm"]},
                 "recognition": manifest["recognition"], "poses": [], "assessment_complete": True,
                 "grouping": {"enabled": True, "barrier_mm": config["grouping_barrier_mm"],
                              "max_angle_deg": config["grouping_max_angle_deg"],
                              "minimum_frequency_percent": config["grouping_minimum_frequency_percent"]},
                 "provenance": provenance or {}}
    for group in groups:
        if not group["retained_by_frequency"]:
            continue
        representative = copy.deepcopy(by_id[group["representative_source_pose_id"]])
        representative.update(count=group["count"], individual_count=representative["count"],
                              frequency_percent=100 * group["probability"], group_members=group["members"],
                              frequency_settled_percent=100 * group["count"] / summary["settled_trials"],
                              representative_retention_auc=group["retention_auc"], group_stability_rank=group["rank"],
                              ci95_percent=[100 * group["ci_low"], 100 * group["ci_high"]])
        representative["reference_quaternions_xyzw"] = [q for key in group["members"]
                                                        for q in by_id[key]["reference_quaternions_xyzw"]]
        representative["individual_trial_ids"] = representative["trial_ids"]
        representative["trial_ids"] = sorted(i for key in group["members"] for i in by_id[key]["trial_ids"])
        if len(representative["trial_ids"]) != representative["count"]:
            raise ValueError("Grouped trial identities differ from the combined count.")
        catalogue["poses"].append(representative)
    catalogue["poses"].sort(key=lambda p: (-p["count"], p["id"]))
    original_observations = {**catalogue, "poses": list(by_id.values()), "grouping": {"enabled": False},
                             "unassigned_settled_trials": audit["unassigned_settled_trials"]}
    repository.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".grouped-stage-", dir=repository) as temp:
        staged = Path(temp) / part
        staged.mkdir()
        (staged / "images").mkdir()
        shutil.copyfile(stl, staged / f"{part}.stl")
        shutil.copyfile(step, staged / f"{part}.step")
        from .pose_image import render_pose_image
        for member in by_id.values():
            image = render_pose_image(stl, manifest["center_mass_source_mm"], member["quaternion_xyzw"], member["position_com_chute_mm"])
            if not image.save(str(staged / member["image"]), "PNG"):
                raise OSError("Could not render a member pose image.")
        _write_json(staged / "poses.json", catalogue)
        (staged / "poses.yaml").write_text(yaml.safe_dump(catalogue, sort_keys=False, allow_unicode=True), encoding="utf-8")
        _write_json(staged / "original_observations.json", original_observations)
        _write_json(staged / "original_pose_registry.json", originals_registry)
        _write_json(staged / "pose_registry.json", {"schema_version": 1, "mesh_sha256": sha256(stl),
                    "next_id": next_id, "entries": [{"id": p["id"], "anchor_quaternion_xyzw": p["quaternion_xyzw"]}
                                                      for p in catalogue["poses"]]})
        _write_json(staged / "groups.json", {"groups": groups, "assessment_complete": True})
        _write_json(staged / "manifest.json", {**manifest, "grouped_export": True, "provenance": provenance or {}, "audit": audit})
        _write_json(staged / "validation.json", audit)
        portable = {**config, "mesh_path": f"{part}.stl", "output_dir": "./local_results",
                    "observed_catalog_path": "original_observations.json",
                    "catalog_repo": None, "catalog_cad_dir": None, "catalog_push": False}
        _write_json(staged / "config.json", portable)
        for name in ("grouping_geometry.json", "group_stability.csv", "group_stability_summary.csv",
                     "group_frequencies.csv", "frequencies.csv", "stability.csv"):
            if name.endswith(".json"):
                evidence = read(run / name)
                evidence["groups"] = [[identities[k] for k in g] for g in evidence["groups"]]
                for row in evidence["paths"]:
                    row["source"], row["target"] = identities[row["source"]], identities[row["target"]]
                for row in evidence["review"]:
                    row["members"] = [identities[k] for k in row["members"]]
                _write_json(staged / name, evidence)
            else:
                with (run / name).open(encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    fields, rows = reader.fieldnames, list(reader)
                for row in rows:
                    for key in ("pose_id", "representative_source_pose_id"):
                        if row.get(key) in identities:
                            row[key] = identities[row[key]]
                    if row.get("members"):
                        # The simulator's CSV writer uses JSON for list columns.
                        row["members"] = json.dumps([identities[k] for k in json.loads(row["members"])])
                _write_csv(staged / name, fields or [], rows)
        membership = [{"group_id": g["pose_id"], "source_pose_id": key,
                       "representative_source_pose_id": g["representative_source_pose_id"],
                       "individual_count": by_id[key]["count"], "combined_count": g["count"],
                       "retained_by_frequency": g["retained_by_frequency"]} for g in groups for key in g["members"]]
        _write_csv(staged / "membership.csv", ["group_id", "source_pose_id", "representative_source_pose_id",
                   "individual_count", "combined_count", "retained_by_frequency"], membership)
        lines = [f"# {part}: grouped observed poses", "", f"{expected_trials:,} completed drops; {summary['settled_trials']:,} settled. "
                 f"{len(members)} assessed source poses; {len(catalogue['poses'])} groups meet the 1% cutoff.", "",
                 "Grouping uses 0.5 mm equivalent lift and a 60° maximum diameter. All members receive the same disturbance protocol.", "",
                 "| Representative | Image | Members | Combined count | Frequency | Retention AUC |", "| --- | --- | --- | ---: | ---: | ---: |"]
        for pose in catalogue["poses"]:
            lines.append(f"| {pose['id']} | ![pose]({pose['image']}) | {', '.join(pose['group_members'])} | {pose['count']} | {pose['frequency_percent']:.2f}% | {pose['representative_retention_auc']:.4f} |")
        if not catalogue["poses"]:
            lines += ["", "Completed simulation: no groups meet the frequency cutoff. There are no eligible representative poses."]
        lines += ["", "[All original observations and member images](original_observations.json), [memberships](membership.csv), "
                  "[barrier evidence](grouping_geometry.json), [member stability curves](group_stability.csv), "
                  "[individual disturbance curves](stability.csv), [validation](validation.json).", "",
                  "Frequencies retain the original completed-drop denominator. Stability scores describe the configured virtual chute."]
        (staged / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        inventory(staged)
        verify_inventory(staged)
        verify_grouped_catalog(staged)
        if read(staged / "poses.json") != yaml.safe_load((staged / "poses.yaml").read_text()):
            raise ValueError("Grouped JSON/YAML mismatch.")
        _replace(staged, repository / part)
    return catalogue, audit
