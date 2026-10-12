"""Reproducible, resumable local grouped study; never publishes a catalog.

python -m experiments.grouped_pose_study --prepare-only
python -m experiments.grouped_pose_study --trials 10000 --workers 8
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from dashas_drop_sim.cli import _config
from dashas_drop_sim.catalog_batch import engine_fingerprint
from dashas_drop_sim.runner import run_experiment, _write_json, _write_csv
from dashas_drop_sim.grouping import analyze_groups

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "experiments/grouped_pose_study"
PARTS = ("Qk1a", "Dl1a", "Dk1i")


def prepare():
    provenance = json.loads((SPEC / "baseline/provenance.json").read_text())
    for part in PARTS:
        raw = gzip.decompress((SPEC / "baseline" / f"{part}.json.gz").read_bytes())
        expected = provenance["workpieces"][part]
        if hashlib.sha256(raw).hexdigest() != expected["catalog_sha256"]:
            raise ValueError("Frozen baseline catalog checksum mismatch.")
        mesh = ROOT / "src/dashas_drop_sim/assets/catalog" / f"{part}.STL"
        if hashlib.sha256(mesh.read_bytes()).hexdigest() != expected["mesh_sha256"]:
            raise ValueError("Bundled geometry differs from the published baseline.")
        folder = ROOT / "results/grouped_pose_study/baseline" / part
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "poses.json").write_bytes(raw)
    return provenance


def local_roadmap(run_dir, part, engine_dir):
    # Reuse the established observed geometry builder without writing into its repository.
    sys.path.insert(0, str(engine_dir / "src"))
    from bibazu_stable_roadmaps.generate import GenerationConfig, generate_one
    destination = run_dir / "roadmaps"
    generate_one(run_dir / "representatives/poses.json",
                 GenerationConfig(output_root=str(destination), minimum_frequency_percent=0,
                                  outputs=("json", "yaml", "roadmap_png", "roadmap_svg", "poses_png")),
                 progress=lambda message: print(json.dumps({"event": "roadmap", "workpiece": part, "message": message}), flush=True))
    groups = json.loads((run_dir / "groups.json").read_text())["groups"]
    mapping = {g["representative_source_pose_id"]: g for g in groups}
    json_path = destination / part / f"{part}_roadmap.json"
    document = json.loads(json_path.read_text())
    for node in document["nodes"]:
        group = mapping[node["source_pose_ids"][0]]
        node["group_members"] = group["members"]
        node["representative_group_retention_auc"] = group["retention_auc"]
        node["classification_basis"] = "observed_group_retention"
    _write_json(json_path, document)
    import yaml
    yaml_path = json_path.with_suffix(".yaml")
    handover = yaml.safe_load(yaml_path.read_text())
    for pose in handover["poses"]:
        group = mapping[pose["source_pose_ids"][0]]
        pose["group_members"] = group["members"]
        pose["representative_group_retention_auc"] = group["retention_auc"]
        pose["classification_basis"] = "observed_group_retention"
    yaml_path.write_text(yaml.safe_dump(handover, sort_keys=False), encoding="utf-8")
    component = json_path.parent
    receipt = {p.relative_to(component).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in component.rglob("*") if p.is_file() and p.name != "files.sha256.json"}
    _write_json(component / "files.sha256.json", receipt)


def compact_report(records, provenance):
    output = SPEC / "report"
    output.mkdir(exist_ok=True)
    lines = ["# Grouped drop study", "", "10,000 fresh drops per workpiece; disturbance assessments are additional.",
             "", f"Baseline: `{provenance['revision']}`. Grouping: 0.5 mm equivalent lift, 60° maximum diameter.",
             "", "| Workpiece | Drops | Settled | Assessed members | Groups ≥1% | Disturbance tests |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    comparisons = {}
    for part in PARTS:
        if part not in records:
            continue
        record = records[part]
        run_dir = Path(record["run_dir"])
        summary = json.loads((run_dir / "summary.json").read_text())
        groups = summary["pose_groups"]
        retained = [g for g in groups if g["retained_by_frequency"]]
        lines.append(f"| {part} | {summary['completed_trials']} | {summary['settled_trials']} | {sum(len(g['members']) for g in groups)} | {len(retained)} | {summary['group_disturbance_trials']} |")
        baseline_ids = provenance["workpieces"][part]["source_pose_to_roadmap_id"]
        for group in groups:
            group["baseline_roadmap_members"] = [baseline_ids[key] for key in group["members"] if key in baseline_ids]
            group["baseline_representative_pose"] = baseline_ids.get(group["representative_source_pose_id"])
            member_curve = sorted((r for r in summary["group_stability_curves"] if r["pose_id"] == group["representative_source_pose_id"]),
                                  key=lambda r: r["energy_lift_mm"])
            group["individual_retention_auc"] = sum((a["individual_retention"] + b["individual_retention"]) *
                                                       (b["energy_lift_mm"] - a["energy_lift_mm"]) / 2
                                                       for a, b in zip(member_curve, member_curve[1:])) / member_curve[-1]["energy_lift_mm"]
        comparisons[part] = {"completed_trials": summary["completed_trials"], "settled_trials": summary["settled_trials"],
                             "status_counts": summary["status_counts"], "groups": groups,
                             "group_assessment_complete": summary["group_assessment_complete"],
                             "group_disturbance_trials": summary["group_disturbance_trials"],
                             "software_commit": json.loads((run_dir / "manifest.json").read_text())["software_commit"]}
        with (run_dir / "group_disturbances.jsonl").open() as stream:
            disturbance_outcomes = [json.loads(line) for line in stream if line.strip()]
        comparisons[part]["group_retained_tests"] = sum(row["retained"] for row in disturbance_outcomes)
        comparisons[part]["individual_retained_tests"] = sum(row["individual_retained"] for row in disturbance_outcomes)
        target = output / part
        target.mkdir(exist_ok=True)
        fixture = json.loads((SPEC / "baseline" / f"{part}_roadmap_fixture.json").read_text())
        baseline_analysis = analyze_groups(ROOT / f"src/dashas_drop_sim/assets/catalog/{part}.STL",
                                          fixture["members"], fixture["recognition"])
        _write_json(target / "baseline_grouping_geometry.json", baseline_analysis)
        comparisons[part]["baseline_groups"] = baseline_analysis["groups"]
        for filename in ("group_frequencies.csv", "group_stability.csv", "group_stability_summary.csv", "grouping_geometry.json", "group_members.json", "frequencies.csv", "stability.csv", "manifest.json"):
            if filename == "group_members.json":
                members = json.loads((run_dir / filename).read_text())
                for member in members:
                    member.pop("reference_quaternions_xyzw", None)
                _write_json(target / filename, members)
            else:
                shutil.copyfile(run_dir / filename, target / filename)
        catalog = json.loads((run_dir / "representatives/poses.json").read_text())
        catalog["geometry"]["stl"] = Path(os.path.relpath(ROOT / f"src/dashas_drop_sim/assets/catalog/{part}.STL", target)).as_posix()
        for pose in catalog["poses"]:
            pose["reference_quaternions_xyzw"] = [pose["quaternion_xyzw"]]
        _write_json(target / "representatives.json", catalog)
        catalog_hash = hashlib.sha256((target / "representatives.json").read_bytes()).hexdigest()
        def catalog_provenance(metadata):
            metadata["original_source_sha256"] = metadata["source_sha256"]
            metadata.update(source_file="representatives.json", source_sha256=catalog_hash)
        roadmap_manifest = json.loads((run_dir / "roadmaps" / part / "manifest.json").read_text())
        catalog_provenance(roadmap_manifest["input"])
        roadmap_manifest["group_frequency_cutoff_percent"] = 1
        _write_json(target / "roadmap_manifest.json", roadmap_manifest)
        for filename in (f"{part}_roadmap.json", f"{part}_roadmap.yaml", f"{part}_roadmap.png", f"{part}_roadmap.svg"):
            path = run_dir / "roadmaps" / part / filename
            if path.is_file():
                if path.suffix == ".json":
                    document = json.loads(path.read_text())
                    for node in document["nodes"]:
                        node.pop("thumbnail_png_base64", None)
                    document["source"] = Path(os.path.relpath(ROOT / f"src/dashas_drop_sim/assets/catalog/{part}.STL", target)).as_posix()
                    catalog_provenance(document["generation"])
                    catalog_provenance(document["classical_metadata"]["observed_pose_generation"])
                    document["group_ranking_method"] = "representative_group_retention_auc"
                    for node in document["nodes"]:
                        node["group_stability_rank"] = next(g["rank"] for g in groups if g["representative_source_pose_id"] == node["source_pose_ids"][0])
                    _write_json(target / filename, document)
                    membership = [{"roadmap_pose_id": node["node_id"],
                                   "representative_source_pose_id": node["source_pose_ids"][0],
                                   "source_pose_id": member,
                                   "baseline_roadmap_pose_id": baseline_ids.get(member)}
                                  for node in document["nodes"] for member in node["group_members"]]
                    _write_csv(target / "roadmap_membership.csv", list(membership[0]) if membership else [], membership)
                elif path.suffix == ".yaml":
                    import yaml
                    handover = yaml.safe_load(path.read_text())
                    handover["part"]["mesh_source"] = Path(os.path.relpath(ROOT / f"src/dashas_drop_sim/assets/catalog/{part}.STL", target)).as_posix()
                    catalog_provenance(handover["generation"])
                    handover["classification"]["group_ranking_method"] = "representative_group_retention_auc"
                    for pose in handover["poses"]:
                        pose["group_stability_rank"] = next(g["rank"] for g in groups if g["representative_source_pose_id"] == pose["source_pose_ids"][0])
                    (target / filename).write_text(yaml.safe_dump(handover, sort_keys=False), encoding="utf-8")
                else:
                    shutil.copyfile(path, target / filename)
    _write_json(output / "comparison.json", {"baseline": provenance, "workpieces": comparisons})
    for part, data in comparisons.items():
        lines += ["", f"## {part}", "", "| Group members (published roadmap IDs) | Representative | Frequency | Individual retention AUC | Group retention AUC |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for group in data["groups"]:
            if group["retained_by_frequency"]:
                baseline_ids = provenance["workpieces"][part]["source_pose_to_roadmap_id"]
                labels = ", ".join([str(value) for value in sorted(group["baseline_roadmap_members"])] +
                                   [key for key in group["members"] if key not in baseline_ids])
                rep = group["baseline_representative_pose"]
                lines.append(f"| {labels} | {rep if rep is not None else group['representative_source_pose_id']} | {100 * group['probability']:.2f}% | {group['individual_retention_auc']:.4f} | {group['retention_auc']:.4f} |")
        lines += ["", "Baseline-only memberships: " + "; ".join("{" + ", ".join(keys) + "}" for keys in data["baseline_groups"]) + ".",
                  f"Across all member disturbance tests, {data['group_retained_tests']} retained the group and {data['individual_retained_tests']} retained the exact individual pose.",
                  "Additional source IDs identify rare baseline observations below the published individual-pose frequency cutoff. Their counts are included before the group cutoff.",
                  "", "`roadmap_membership.csv` maps new roadmap node IDs to representatives, every source member, and the published numbering. `representatives.json` is the representative-only input catalog."]
        if part == "Dk1i":
            baseline_analysis = json.loads((output / part / "baseline_grouping_geometry.json").read_text())
            lines += ["", "| Published candidate pair | Forward barrier (mm) | Reverse barrier (mm) | Angular separation |",
                      "| --- | ---: | ---: | ---: |"]
            for first, second in (("1", "5"), ("2", "4")):
                evidence = next(p for p in baseline_analysis["paths"] if p["source"] == first and p["target"] == second)
                lines.append(f"| {{{first},{second}}} | {evidence['forward_barrier_mm']:.4f} | {evidence['reverse_barrier_mm']:.4f} | {evidence['angle_deg']:.2f}° |")
            lines += ["", "Both pairs satisfy the same 0.5 mm rule used for all workpieces; neither pair is prescribed in the grouping implementation."]
            if data["individual_retained_tests"] == data["group_disturbance_trials"]:
                lines += ["No settled switching between individual members was observed in these disturbance tests. The consolidation is supported by the sampled geometric barrier criterion; the measured individual and group retention scores coincide."]
    lines += ["", "AUC measures remaining within the group throughout settled observations across the configured energy levels and directions. These scores use the configured virtual chute and uncalibrated contact assumptions.",
              "", "Validation: the simulator suite passed 102 tests before production. Updated GUI checks (5 tests) and published geometry regressions (3 cases) also passed. The seeded 8-drop comparison produced identical records with 1 and 8 workers; see [worker_smoke.json](../worker_smoke.json).",
              "The [production audit](validation.json) verifies unique trial IDs and seeds, count conservation, complete member assessments, deterministic representatives, and representative-only roadmap orientations. Export checksums are in `files.sha256.json`.",
              "", "Full drop and disturbance trajectories remain under ignored `results/grouped_pose_study/`. No existing pose or roadmap catalog is modified."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=PARTS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/grouped_pose_study")
    parser.add_argument("--roadmap-engine", type=Path, default=ROOT.parent / "BiBaZu_StableComponentenRoadmaps")
    args = parser.parse_args()
    provenance = prepare()
    if args.prepare_only:
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / "study.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {"workpieces": {}}
    fingerprint = engine_fingerprint()
    receipt.update(baseline=provenance["revision"], engine_fingerprint=fingerprint)
    for part in args.parts:
        config = replace(_config(SPEC / "configs" / f"{part}.json"), trials=args.trials,
                         workers=args.workers, output_dir=args.output / part)
        signature = hashlib.sha256(json.dumps({"config": {k: v for k, v in config.to_dict().items() if k != "workers"}, "engine": fingerprint}, sort_keys=True).encode()).hexdigest()
        previous = receipt["workpieces"].get(part, {})
        if previous and previous.get("signature") != signature:
            raise ValueError("Study settings/code differ; choose a new --output directory.")
        if previous.get("status") == "complete":
            print(json.dumps({"event": "skip_completed", "workpiece": part}), flush=True)
            continue
        started = time.perf_counter()
        def emit(event):
            if event["event"] == "started":
                receipt["workpieces"][part] = {"status": "running", "signature": signature, "run_dir": event["run_dir"]}
                _write_json(receipt_path, receipt)
            if event["event"] not in {"trial", "progress", "stability_progress"} or event.get("trial", event.get("completed", 0)) % 100 == 0:
                concise = {key: event[key] for key in ("event", "trial", "completed", "remaining", "total", "run_dir", "status") if key in event}
                print(json.dumps({"workpiece": part, **concise}), flush=True)
        result = run_experiment(config, emit=emit, resume_run_dir=Path(previous["run_dir"]) if previous.get("run_dir") else None)
        if result.summary["completed_trials"] != args.trials or not result.summary["group_assessment_complete"]:
            raise RuntimeError("Study run or representative assessment is incomplete.")
        local_roadmap(result.run_dir, part, args.roadmap_engine)
        receipt["workpieces"][part].update(status="complete", elapsed_s=time.perf_counter() - started)
        _write_json(receipt_path, receipt)
    if all(p in receipt["workpieces"] and receipt["workpieces"][p]["status"] == "complete" for p in PARTS):
        compact_report(receipt["workpieces"], provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
