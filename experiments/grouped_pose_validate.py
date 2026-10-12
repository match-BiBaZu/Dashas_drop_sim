"""Audit the completed production records and compact handover artifacts."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dashas_drop_sim.grouping import choose_representative
from dashas_drop_sim.runner import SETTLED_STATUSES, _write_json
from .grouped_pose_study import ROOT, SPEC, PARTS


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    study = load(ROOT / "results/grouped_pose_study/study.json")
    audited = {}
    for part in PARTS:
        receipt = study["workpieces"][part]
        assert receipt["status"] == "complete", part
        run = Path(receipt["run_dir"])
        summary, config = load(run / "summary.json"), load(run / "config.json")
        trials = records(run / "trials.jsonl")
        assert len(trials) == summary["completed_trials"] == config["trials"] == 10000
        assert {row["trial"] for row in trials} == set(range(10000))
        seeds = np.random.SeedSequence(42).spawn(10000)
        assert all(row["seed"] == int(seeds[row["trial"]].generate_state(1, dtype=np.uint32)[0]) for row in trials)
        assert dict(Counter(row["status"] for row in trials)) == summary["status_counts"]
        assert config["catalog_repo"] is None and not config["catalog_push"]
        assert config["grouping_enabled"] and config["compute_stability"]
        assert config["grouping_barrier_mm"] == .5 and config["grouping_max_angle_deg"] == 60
        groups = load(run / "groups.json")["groups"]
        members = {row["id"]: row for row in load(run / "group_members.json")}
        all_keys = [key for group in groups for key in group["members"]]
        assert len(all_keys) == len(set(all_keys)) == len(members)
        assert set(all_keys) == set(members)
        counts = Counter(row["pose_key"] for row in trials if row["status"] in SETTLED_STATUSES)
        assert all(member["count"] == counts[key] for key, member in members.items())
        assert all(group["count"] == sum(counts[key] for key in group["members"]) for group in groups)
        assert sum(group["count"] for group in groups) + counts["ambiguous"] == summary["settled_trials"]
        assert all(group["probability"] == group["count"] / 10000 for group in groups)
        assert summary["group_assessment_complete"] and all(group["complete"] for group in groups)
        disturbances = records(run / "group_disturbances.jsonl")
        levels = config["disturbance_levels_mm"]
        planned = len(members) * sum(1 if level == 0 else 12 for level in levels)
        assert len(disturbances) == summary["group_disturbance_trials"] == planned
        assert {row["disturbance_index"] for row in disturbances} == set(range(planned))
        combinations = Counter((row["source_pose_id"], row["energy_lift_mm"], row["direction_index"]) for row in disturbances)
        assert set(combinations) == {(key, level, i) for key in members for level in levels for i in range(1 if level == 0 else 12)}
        assert set(combinations.values()) == {1}
        curves = summary["group_stability_curves"]
        assert all(row["complete"] for row in curves)
        scores, threshold = {}, {}
        for key in members:
            rows = sorted((row for row in curves if row["pose_id"] == key), key=lambda row: row["energy_lift_mm"])
            assert [row["energy_lift_mm"] for row in rows] == levels
            ys = np.asarray([row["retention"] for row in rows])
            scores[key] = float(np.sum((ys[1:] + ys[:-1]) * np.diff(levels) / 2) / levels[-1])
            threshold[key] = next(row["retention"] for row in rows if row["energy_lift_mm"] == .5)
        for group in groups:
            assert choose_representative(group["members"], scores, counts, threshold) == group["representative_source_pose_id"]
        catalog = load(run / "representatives/poses.json")
        retained = {g["representative_source_pose_id"]: g for g in groups if g["retained_by_frequency"]}
        assert {pose["id"] for pose in catalog["poses"]} == set(retained)
        roadmap = load(run / "roadmaps" / part / f"{part}_roadmap.json")
        assert len(roadmap["nodes"]) == len(retained)
        for node in roadmap["nodes"]:
            assert len(node["source_pose_ids"]) == 1
            key = node["source_pose_ids"][0]
            assert key in retained and node["group_members"] == retained[key]["members"]
            first = Rotation.from_quat(node["representative_quaternion_xyzw"])
            second = Rotation.from_quat(members[key]["quaternion_xyzw"])
            assert (first.inv() * second).magnitude() < 1e-8
            assert node["observed_count"] == retained[key]["count"]
        report = SPEC / "report" / part
        compact = load(report / "representatives.json")
        assert (report / compact["geometry"]["stl"]).is_file()
        with (report / "roadmap_membership.csv").open(encoding="utf-8") as stream:
            membership_rows = list(csv.DictReader(stream))
        assert len(membership_rows) == sum(len(group["members"]) for group in retained.values())
        compact_roadmap = load(report / f"{part}_roadmap.json")
        assert compact_roadmap["generation"]["source_sha256"] == sha256(report / "representatives.json")
        audited[part] = {"completed_unique_drops": len(trials), "settled_trials": summary["settled_trials"],
                         "unassigned_settled_trials": counts["ambiguous"], "count_conserved": True,
                         "assessed_members": len(members), "disturbance_tests": len(disturbances),
                         "assessment_complete": True, "representative_roadmap_nodes": len(retained),
                         "trials_sha256": sha256(run / "trials.jsonl"),
                         "disturbances_sha256": sha256(run / "group_disturbances.jsonl"),
                         "run_directory": run.relative_to(ROOT).as_posix()}
    _write_json(SPEC / "report/validation.json", {"baseline": study["baseline"],
                "engine_fingerprint": study["engine_fingerprint"], "workpieces": audited})
    files = {path.relative_to(SPEC / "report").as_posix(): sha256(path)
             for path in (SPEC / "report").rglob("*") if path.is_file() and path.name != "files.sha256.json"}
    _write_json(SPEC / "report/files.sha256.json", files)
    print(json.dumps(audited, indent=2))


if __name__ == "__main__":
    main()
