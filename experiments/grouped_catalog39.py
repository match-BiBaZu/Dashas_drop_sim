"""Resumable 39-workpiece grouped study and explicit grouped-branch publication.

Prepare: python -m experiments.grouped_catalog39 --prepare-only
Execute/publish: python -m experiments.grouped_catalog39 --publish
Audit all completed artifacts: python -m experiments.grouped_catalog39 --audit-only
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from dashas_drop_sim.catalog import catalog_models
from dashas_drop_sim.catalog_batch import engine_fingerprint, signature
from dashas_drop_sim.catalog_export import sha256
from dashas_drop_sim.cli import _config
from dashas_drop_sim.grouped_export import (archive_completed_run, audit_run, export_grouped_poses, export_grouped_roadmap,
                                          inventory, read, verify_inventory)
from dashas_drop_sim.runner import run_experiment, _write_json, _write_csv

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "experiments/grouped_catalog39"
OUTPUT = ROOT / "results/grouped_catalog39"
POSE_REVISION = "9e00d03389d3b81013700def03fb1b19504144be"
ROADMAP_REVISION = "0b3b5c5fe42e787bd47f381b1f1ac57359d5aa55"
REPAIRED_REVISION = "3523350d8e12a1b7e57467044b9cfe25ff49b0bd"
REUSE = ("Qk1a", "Dl1a", "Dk1i")
POSE_REPO = ROOT.parent / "BiBaZu_StableComponentenPoses_grouped"
ROADMAP_REPO = ROOT.parent / "BiBaZu_StableComponentenRoadmaps_grouped"


def git(repo, *args, binary=False):
    repo = Path(repo).resolve()
    result = subprocess.run(["git", "-c", f"safe.directory={repo.as_posix()}", "-c",
                             f"safe.directory={(repo / '.git').as_posix()}", "-C", str(repo), *args],
                            capture_output=True, timeout=180,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip())
    return result.stdout if binary else result.stdout.decode("utf-8", "replace").strip()


def prepare(pose_repo=POSE_REPO, roadmap_repo=ROADMAP_REPO):
    models = catalog_models()
    if len(models) != 39:
        raise ValueError("The frozen study requires exactly 39 bundled workpieces.")
    (SPEC / "baseline").mkdir(parents=True, exist_ok=True)
    (SPEC / "configs").mkdir(exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    provenance_path = SPEC / "baseline/provenance.json"
    if provenance_path.is_file():
        provenance = read(provenance_path)
        if set(provenance["workpieces"]) != set(models):
            raise ValueError("Frozen workpiece list differs from the bundled catalog.")
        for part, mesh in models.items():
            data = provenance["workpieces"][part]
            raw = gzip.decompress((SPEC / "baseline" / f"{part}.json.gz").read_bytes())
            if hashlib.sha256(raw).hexdigest() != data["catalog_sha256"] or sha256(mesh) != data["mesh_sha256"]:
                raise ValueError(f"Frozen baseline checksum mismatch: {part}")
    else:
        provenance = {"pose_revision": POSE_REVISION, "roadmap_revision": ROADMAP_REVISION,
                      "repaired_roadmap_revision": REPAIRED_REVISION,
                      "engine_fingerprint": engine_fingerprint(), "workpieces": {}}
        for part, mesh in models.items():
            raw = git(pose_repo, "show", f"{POSE_REVISION}:{part}/poses.json", binary=True)
            catalogue = json.loads(raw)
            if catalogue["geometry"]["stl_sha256"] != sha256(mesh):
                raise ValueError(f"Published baseline mesh differs: {part}")
            (SPEC / "baseline" / f"{part}.json.gz").write_bytes(gzip.compress(raw, mtime=0))
            registry = git(pose_repo, "show", f"{POSE_REVISION}:{part}/pose_registry.json", binary=True)
            (SPEC / "baseline" / f"{part}_registry.json.gz").write_bytes(gzip.compress(registry, mtime=0))
            entry = {"catalog_sha256": hashlib.sha256(raw).hexdigest(), "mesh_sha256": sha256(mesh),
                     "registry_sha256": hashlib.sha256(registry).hexdigest(), "baseline_roadmap_ids": {}}
            try:
                roadmap = json.loads(git(roadmap_repo, "show", f"{ROADMAP_REVISION}:{part}/{part}_roadmap.json", binary=True))
                entry["baseline_roadmap_ids"] = {key: n["node_id"] for n in roadmap["nodes"] for key in n["source_pose_ids"]}
            except RuntimeError:
                entry["baseline_roadmap_status"] = "no_published_roadmap"
            provenance["workpieces"][part] = entry
        _write_json(provenance_path, provenance)
    if engine_fingerprint() != provenance["engine_fingerprint"]:
        raise ValueError("Simulation engine changed since this study was frozen.")
    template = read(ROOT / "experiments/grouped_pose_study/configs/Qk1a.json")
    template["disturbance_levels_mm"] = [0, .05, .1, .2, .4, .5, .8]
    for part in models:
        folder = OUTPUT / "baseline" / part
        folder.mkdir(parents=True, exist_ok=True)
        raw = gzip.decompress((SPEC / "baseline" / f"{part}.json.gz").read_bytes())
        (folder / "poses.json").write_bytes(raw)
        config = {**template, "mesh_path": f"../../src/dashas_drop_sim/assets/catalog/{part}.STL",
                  "output_dir": f"../../results/grouped_catalog39/{part}",
                  "observed_catalog_path": f"../../results/grouped_catalog39/baseline/{part}/poses.json"}
        # Configs live one additional directory below SPEC.
        for key in ("mesh_path", "output_dir", "observed_catalog_path"):
            config[key] = "../" + config[key]
        path = SPEC / "configs" / f"{part}.json"
        if path.is_file() and read(path) != config:
            raise ValueError("Frozen study configuration changed.")
        _write_json(path, config)
    return provenance


def protocol(config):
    return json.loads(json.dumps({k: v for k, v in config.items() if k not in {
        "mesh_path", "output_dir", "observed_catalog_path", "roadmap_path", "catalog_repo", "catalog_cad_dir", "workers"}}))


def adopt_completed(study, provenance):
    previous = read(ROOT / "results/grouped_pose_study/study.json")
    if previous["engine_fingerprint"] != provenance["engine_fingerprint"]:
        raise ValueError("Completed runs used a different simulation engine.")
    for part in REUSE:
        if part in study["workpieces"]:
            continue
        old = previous["workpieces"][part]
        if old["status"] != "complete":
            raise ValueError(f"Requested reused run is incomplete: {part}")
        run = Path(old["run_dir"])
        cfg = _config(SPEC / "configs" / f"{part}.json")
        cfg.validate()
        if protocol(cfg.to_dict()) != protocol(read(run / "config.json")):
            raise ValueError(f"Completed run uses a different testing protocol: {part}")
        baseline = read(OUTPUT / "baseline" / part / "poses.json")
        manifest = read(run / "manifest.json")
        if baseline["geometry"]["stl_sha256"] != manifest["mesh_source_sha256"]:
            raise ValueError("Reused run geometry differs from the catalog.")
        audit = audit_run(run)
        study["workpieces"][part] = {"status": "simulation_complete", "reused": True,
                                       "run_dir": str(run), "audit": audit, "signature": signature(cfg)}


def refresh_reports(study, provenance, pose_repo, roadmap_repo):
    completed = [p for p, r in study["workpieces"].items() if r["status"] == "complete"]
    status_doc = {"study": "grouped_catalog39", "completed_workpieces": len(completed), "total_workpieces": 39,
                  "target_completed_drops": 390000, "grouping_barrier_mm": .5, "grouping_max_angle_deg": 60,
                  "provenance": provenance, "workpieces": {}}
    lines = ["# Grouped drop simulation: all 39 workpieces", "",
             f"Completed and validated: **{len(completed)}/39** workpieces. Each completed workpiece has 10,000 drops; member disturbance tests are additional.", "",
             "All workpieces use identical simulation settings, a 0.5 mm equivalent-lift grouping barrier, a 60° diameter guard, "
             "and a 1% cutoff after aggregation. Representatives are selected using complete member group-retention curves.", "",
             "Qk1a, Dl1a, and Dk1i reuse the previously audited grouped runs. Full raw trial data remains in the simulator's ignored results directories.", "",
             "| Workpiece | Status | Settled | Source poses | Eligible groups | Disturbance tests | Roadmap connections |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    membership_summary = []
    for part in provenance["workpieces"]:
        record = study["workpieces"].get(part, {})
        row = {"status": record.get("status", "pending")}
        if part in completed:
            audit = record["audit"]
            manifest = read(roadmap_repo / part / "manifest.json")
            groups = read(pose_repo / part / "groups.json")["groups"]
            baseline_ids = provenance["workpieces"][part]["baseline_roadmap_ids"]
            for group in groups:
                group["baseline_roadmap_members"] = sorted(baseline_ids[k] for k in group["members"] if k in baseline_ids)
                group["baseline_representative_pose"] = baseline_ids.get(group["representative_source_pose_id"])
                membership_summary.append({"workpiece": part, "members": group["members"],
                    "baseline_roadmap_members": group["baseline_roadmap_members"],
                    "representative_source_pose_id": group["representative_source_pose_id"],
                    "baseline_representative_pose": group["baseline_representative_pose"],
                    "combined_count": group["count"], "frequency_percent": 100 * group["probability"],
                    "retention_auc": group["retention_auc"], "retained_by_frequency": group["retained_by_frequency"]})
            row.update(**audit, reused=record.get("reused", False), transition_count=manifest["transition_count"],
                       groups=groups, roadmap_status=manifest["status"])
            lines.append(f"| [{part}]({part}/README.md) | Complete | {audit['settled_trials']} | {audit['assessed_members']} | {audit['eligible_groups']} | {audit['disturbance_tests']} | {manifest['transition_count']} |")
        else:
            lines.append(f"| {part} | {row['status']} | — | — | — | — | — |")
        status_doc["workpieces"][part] = row
    lines += ["", "Each component page links representative images, full membership and barrier evidence, and individual/member stability curves. "
              "Percentages use all completed drops. Roadmap connections are computed from representative orientations.", "",
              f"Published comparison baselines: poses `{POSE_REVISION}`, roadmaps `{ROADMAP_REVISION}`. Merge repair: `{REPAIRED_REVISION}`."]
    for repo in (pose_repo, roadmap_repo):
        _write_json(repo / "grouped_study.json", status_doc)
        _write_json(repo / "index.json", {"schema_version": 1, "workpieces": [
            {"workpiece": p, **{k: v for k, v in row.items() if k != "groups"}}
            for p, row in status_doc["workpieces"].items()]})
        (repo / "GROUPED_STUDY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        _write_csv(repo / "grouped_memberships.csv", ["workpiece", "members", "baseline_roadmap_members",
                   "representative_source_pose_id", "baseline_representative_pose", "combined_count",
                   "frequency_percent", "retention_auc", "retained_by_frequency"], membership_summary)
        readme = repo / "README.md"
        text = readme.read_text(encoding="utf-8")
        marker = "<!-- GROUPED STUDY -->"
        if marker in text:
            text = text[:text.index(marker)].rstrip()
        if "<!-- GENERATED ROADMAP INDEX START -->" in text:
            start, end = text.index("<!-- GENERATED ROADMAP INDEX START -->"), text.index("<!-- GENERATED ROADMAP INDEX END -->") + len("<!-- GENERATED ROADMAP INDEX END -->")
            text = text[:start] + text[end:]
        if repo == pose_repo:
            text = "# BiBaZu grouped observed component poses\n\nRepresentative-only catalogs; original observations and all member images are preserved in companion files."
        readme.write_text(text.rstrip() + "\n\n" + marker + "\n" + "\n".join(lines[2:]) + "\n", encoding="utf-8")
    report = SPEC / "report"
    report.mkdir(exist_ok=True)
    _write_json(report / "comparison.json", status_doc)
    _write_csv(report / "memberships.csv", ["workpiece", "members", "baseline_roadmap_members",
               "representative_source_pose_id", "baseline_representative_pose", "combined_count",
               "frequency_percent", "retention_auc", "retained_by_frequency"], membership_summary)
    report_text = "\n".join(lines)
    for part in provenance["workpieces"]:
        report_text = report_text.replace(f"]({part}/README.md)",
            f"](https://github.com/match-BiBaZu/BiBaZu_StableComponentenPoses/blob/grouped/{part}/README.md)")
    (report / "README.md").write_text(report_text + "\n", encoding="utf-8")
    inventory(report)


def publish(repo, message, push):
    if git(repo, "branch", "--show-current") != "grouped":
        raise ValueError("Experiment publication requires branch grouped.")
    git(repo, "add", "--all")
    if git(repo, "diff", "--cached", "--name-only"):
        git(repo, "commit", "-m", message)
    if push:
        git(repo, "push", "-u", "origin", "HEAD:refs/heads/grouped")
        remote = git(repo, "ls-remote", "origin", "refs/heads/grouped").split()[0]
        if remote != git(repo, "rev-parse", "HEAD"):
            raise ValueError("Published grouped branch commit does not match local HEAD.")


def simulate_only(part):
    """Independent process with its own restart journal; never writes catalog branches."""
    provenance = read(SPEC / "baseline/provenance.json")
    if part not in provenance["workpieces"] or engine_fingerprint() != provenance["engine_fingerprint"]:
        raise ValueError("Unknown workpiece or changed frozen simulation engine.")
    cfg = _config(SPEC / "configs" / f"{part}.json")
    cfg.validate()
    journal = OUTPUT / part / "job.json"
    journal.parent.mkdir(parents=True, exist_ok=True)
    master = read(OUTPUT / "study.json")["workpieces"].get(part, {})
    record = read(journal) if journal.is_file() else master
    expected = signature(cfg)
    if record and record["signature"] != expected:
        raise ValueError("Partial workpiece configuration changed.")
    if record.get("status") in {"complete", "simulation_complete"}:
        audit_run(Path(record["run_dir"]))
        _write_json(journal, record)
        return
    def emit(event):
        if event["event"] == "started":
            _write_json(journal, {"status": "running", "run_dir": event["run_dir"], "signature": expected})
        n = event.get("completed", 0)
        if event["event"] not in {"trial", "progress", "stability_progress"} or n % 500 == 0:
            print(json.dumps({"workpiece": part, **{k: event[k] for k in ("event", "completed", "total", "run_dir") if k in event}}), flush=True)
    result = run_experiment(cfg, emit=emit, resume_run_dir=Path(record["run_dir"]) if record.get("run_dir") else None)
    _write_json(journal, {"status": "simulation_complete", "run_dir": str(result.run_dir),
                         "signature": expected, "audit": audit_run(result.run_dir), "reused": False})


def simulation_process(part):
    logs = OUTPUT / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / f"{part}.log").open("ab") as stream:
        result = subprocess.run([sys.executable, "-m", "experiments.grouped_catalog39", "--simulate-only", part],
                                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"Simulation failed for {part}; see {logs / (part + '.log')}")
    return read(OUTPUT / part / "job.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--simulate-only", help=argparse.SUPPRESS)
    parser.add_argument("--concurrent-workpieces", type=int, choices=(1, 2, 3), default=1,
                        help="Independent workpieces in flight; each retains eight simulation workers.")
    parser.add_argument("--parts", nargs="+")
    parser.add_argument("--pose-repo", type=Path, default=POSE_REPO)
    parser.add_argument("--roadmap-repo", type=Path, default=ROADMAP_REPO)
    args = parser.parse_args()
    if args.simulate_only:
        simulate_only(args.simulate_only)
        return
    provenance = prepare(args.pose_repo, args.roadmap_repo)
    study_path = OUTPUT / "study.json"
    study = read(study_path) if study_path.is_file() else {"workpieces": {}, "provenance": provenance}
    adopt_completed(study, provenance)
    _write_json(study_path, study)
    sys.path.insert(0, str(args.roadmap_repo / "src"))
    for repo in (args.pose_repo, args.roadmap_repo):
        if git(repo, "branch", "--show-current") != "grouped":
            raise ValueError("Use clean grouped checkouts for export.")
        # Preserve exact exported bytes, including pose sheets and new companions.
        attributes = repo / ".gitattributes"
        text = attributes.read_text() if attributes.is_file() else ""
        if "# Grouped exports" not in text:
            text += "\n# Grouped exports: checksum inventories cover exact bytes.\n[DKQR][fkl][1-4][ai]/** -text -whitespace\ngrouped_study.json -text\n"
            attributes.write_text(text, encoding="utf-8")
    refresh_reports(study, provenance, args.pose_repo, args.roadmap_repo)
    if args.prepare_only:
        return
    parts = args.parts or [*REUSE, *(p for p in provenance["workpieces"] if p not in REUSE)]
    if set(parts) - set(provenance["workpieces"]):
        raise ValueError("Unknown workpiece requested.")
    if args.concurrent_workpieces > 1 and not args.audit_only:
        pending = [p for p in parts if study["workpieces"].get(p, {}).get("status") != "complete"]
        with ThreadPoolExecutor(max_workers=args.concurrent_workpieces) as executor:
            futures = {executor.submit(simulation_process, part): part for part in pending}
            for future in as_completed(futures):
                part = futures[future]
                study = read(study_path)
                study["workpieces"][part] = future.result()
                _write_json(study_path, study)
                # Serialize exports and Git publication in the coordinator. Workers
                # only mutate their own physical run and independent journal.
                original_args = sys.argv[:]
                sys.argv = [original_args[0], "--parts", part, "--pose-repo", str(args.pose_repo),
                            "--roadmap-repo", str(args.roadmap_repo)] + (["--publish"] if args.publish else [])
                try:
                    main()
                finally:
                    sys.argv = original_args
        study = read(study_path)
        refresh_reports(study, provenance, args.pose_repo, args.roadmap_repo)
        if args.publish:
            for repo in (args.pose_repo, args.roadmap_repo):
                publish(repo, "Update grouped study completion report", True)
        return
    for part in parts:
        record = study["workpieces"].get(part, {})
        if args.audit_only:
            if record.get("status") != "complete":
                raise ValueError(f"Study is incomplete: {part}")
            audit_run(Path(record["run_dir"]))
            for repo in (args.pose_repo, args.roadmap_repo):
                verify_inventory(repo / part)
            continue
        cfg = _config(SPEC / "configs" / f"{part}.json")
        cfg.validate()
        expected_signature = signature(cfg)
        if record and record["signature"] != expected_signature:
            raise ValueError(f"Run configuration/engine changed: {part}")
        if record.get("status") == "complete":
            if not record.get("reused", False):
                archive_completed_run(Path(record["run_dir"]), OUTPUT)
            # Retry a failed Git push independently of completed simulation/export.
            if args.publish:
                for repo in (args.pose_repo, args.roadmap_repo):
                    publish(repo, f"Publish grouped {part}: 10000 audited drops", True)
            print(json.dumps({"event": "skip_completed", "workpiece": part}), flush=True)
            continue
        print(json.dumps({"event": "workpiece_started", "workpiece": part, "reused": record.get("reused", False)}), flush=True)
        if record.get("status") != "simulation_complete":
            def emit(event):
                if event["event"] == "started":
                    study["workpieces"][part] = {"status": "running", "run_dir": event["run_dir"], "signature": expected_signature}
                    _write_json(study_path, study)
                n = event.get("completed", 0)
                if event["event"] not in {"trial", "progress", "stability_progress"} or n % 500 == 0:
                    print(json.dumps({"workpiece": part, **{k: event[k] for k in ("event", "completed", "total", "run_dir") if k in event}}), flush=True)
            result = run_experiment(cfg, emit=emit, resume_run_dir=Path(record["run_dir"]) if record.get("run_dir") else None)
            record = {"status": "simulation_complete", "run_dir": str(result.run_dir), "signature": expected_signature,
                      "audit": audit_run(result.run_dir), "reused": False}
            study["workpieces"][part] = record
            _write_json(study_path, study)
        if engine_fingerprint() != provenance["engine_fingerprint"]:
            raise ValueError("Simulation engine changed during the study.")
        baseline = read(OUTPUT / "baseline" / part / "poses.json")
        raw = gzip.decompress((SPEC / "baseline" / f"{part}_registry.json.gz").read_bytes())
        if hashlib.sha256(raw).hexdigest() != provenance["workpieces"][part]["registry_sha256"]:
            raise ValueError("Frozen source registry checksum mismatch.")
        part_provenance = {"pose_revision": POSE_REVISION, "roadmap_revision": ROADMAP_REVISION,
                           "engine_fingerprint": provenance["engine_fingerprint"], "reused": record.get("reused", False),
                           "baseline_roadmap_ids": provenance["workpieces"][part]["baseline_roadmap_ids"]}
        _, audit = export_grouped_poses(Path(record["run_dir"]), args.pose_repo, baseline, json.loads(raw), provenance=part_provenance)
        export_grouped_roadmap(args.pose_repo / part / "poses.json", args.roadmap_repo, provenance=part_provenance,
                              progress=lambda s: print(json.dumps({"workpiece": part, "event": "roadmap", "message": s}), flush=True))
        record.update(status="complete", audit=audit)
        study["workpieces"][part] = record
        _write_json(study_path, study)
        refresh_reports(study, provenance, args.pose_repo, args.roadmap_repo)
        for repo in (args.pose_repo, args.roadmap_repo):
            publish(repo, f"Publish grouped {part}: 10000 audited drops", args.publish)
        if not record.get("reused", False):
            archive_completed_run(Path(record["run_dir"]), OUTPUT)
        print(json.dumps({"event": "workpiece_complete", "workpiece": part, **audit}), flush=True)
    refresh_reports(study, provenance, args.pose_repo, args.roadmap_repo)
    if args.publish:
        for repo in (args.pose_repo, args.roadmap_repo):
            publish(repo, "Update grouped study completion report", True)


if __name__ == "__main__":
    main()
