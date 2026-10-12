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
from dashas_drop_sim.runner import run_experiment, _write_json

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
    for part, record in records.items():
        run_dir = Path(record["run_dir"])
        summary = json.loads((run_dir / "summary.json").read_text())
        groups = summary["pose_groups"]
        retained = [g for g in groups if g["retained_by_frequency"]]
        lines.append(f"| {part} | {summary['completed_trials']} | {summary['settled_trials']} | {sum(len(g['members']) for g in groups)} | {len(retained)} | {summary['group_disturbance_trials']} |")
        baseline_ids = provenance["workpieces"][part]["source_pose_to_roadmap_id"]
        for group in groups:
            group["baseline_roadmap_members"] = [baseline_ids[key] for key in group["members"] if key in baseline_ids]
            group["baseline_representative_pose"] = baseline_ids.get(group["representative_source_pose_id"])
        comparisons[part] = {"completed_trials": summary["completed_trials"], "settled_trials": summary["settled_trials"],
                             "status_counts": summary["status_counts"], "groups": groups,
                             "group_assessment_complete": summary["group_assessment_complete"],
                             "group_disturbance_trials": summary["group_disturbance_trials"],
                             "software_commit": json.loads((run_dir / "manifest.json").read_text())["software_commit"]}
        target = output / part
        target.mkdir(exist_ok=True)
        for filename in ("group_frequencies.csv", "group_stability.csv", "group_stability_summary.csv", "grouping_geometry.json", "group_members.json"):
            if filename == "group_members.json":
                members = json.loads((run_dir / filename).read_text())
                for member in members:
                    member.pop("reference_quaternions_xyzw", None)
                _write_json(target / filename, members)
            else:
                shutil.copyfile(run_dir / filename, target / filename)
        for filename in (f"{part}_roadmap.json", f"{part}_roadmap.yaml", f"{part}_roadmap.png", f"{part}_roadmap.svg"):
            path = run_dir / "roadmaps" / part / filename
            if path.is_file():
                if path.suffix == ".json":
                    document = json.loads(path.read_text())
                    for node in document["nodes"]:
                        node.pop("thumbnail_png_base64", None)
                    document["source"] = Path(os.path.relpath(ROOT / f"src/dashas_drop_sim/assets/catalog/{part}.STL", target)).as_posix()
                    _write_json(target / filename, document)
                elif path.suffix == ".yaml":
                    import yaml
                    handover = yaml.safe_load(path.read_text())
                    handover["part"]["mesh_source"] = Path(os.path.relpath(ROOT / f"src/dashas_drop_sim/assets/catalog/{part}.STL", target)).as_posix()
                    (target / filename).write_text(yaml.safe_dump(handover, sort_keys=False), encoding="utf-8")
                else:
                    shutil.copyfile(path, target / filename)
    _write_json(output / "comparison.json", {"baseline": provenance, "workpieces": comparisons})
    for part, data in comparisons.items():
        lines += ["", f"## {part}", "", "| Group members (published roadmap IDs) | Representative | Frequency | Group retention AUC |",
                  "| --- | --- | ---: | ---: |"]
        for group in data["groups"]:
            if group["retained_by_frequency"]:
                baseline_ids = provenance["workpieces"][part]["source_pose_to_roadmap_id"]
                labels = ", ".join(str(baseline_ids.get(key, key)) for key in group["members"])
                rep = group["baseline_representative_pose"]
                lines.append(f"| {labels} | {rep if rep is not None else group['representative_source_pose_id']} | {100 * group['probability']:.2f}% | {group['retention_auc']:.4f} |")
    lines += ["", "AUC measures remaining within the group throughout settled observations across the configured energy levels and directions. These scores use the configured virtual chute and uncalibrated contact assumptions.",
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
