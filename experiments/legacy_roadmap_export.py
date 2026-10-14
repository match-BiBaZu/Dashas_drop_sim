"""Historical grouped-study adapter; excluded from the installed simulator.

Normal simulation only exports observed poses. Use the separate roadmap app
for current roadmap generation and numbering.
"""
from pathlib import Path
import shutil
import tempfile

from scipy.spatial.transform import Rotation
import yaml

from dashas_drop_sim.grouped_export import read, inventory, verify_inventory, _replace
from dashas_drop_sim.runner import _write_json, _write_csv


def export_grouped_roadmap(source, repository, *, provenance=None, progress=lambda s: None):
    from bibazu_stable_roadmaps.generate import GenerationConfig, generate_one, _write_graphml
    source, repository = Path(source), Path(repository)
    catalogue = read(source)
    part, target = catalogue["workpiece"], repository / catalogue["workpiece"]
    groups = {g["representative_source_pose_id"]: g for g in read(source.parent / "groups.json")["groups"]}
    with tempfile.TemporaryDirectory(prefix=".grouped-roadmap-", dir=repository) as temp:
        staging_root = Path(temp)
        staged = staging_root / part
        if catalogue["poses"]:
            status, _ = generate_one(source, GenerationConfig(output_root=str(staging_root), minimum_frequency_percent=0,
                        outputs=("json", "yaml", "graphml", "roadmap_png", "roadmap_svg", "poses_png", "poses_svg")), progress=progress)
            if status != "completed":
                raise ValueError("Eligible representatives did not generate a roadmap.")
            path = staged / f"{part}_roadmap.json"
            doc = read(path)
            for node in doc["nodes"]:
                g = groups[node["source_pose_ids"][0]]
                node.update(group_members=g["members"], representative_group_retention_auc=g["retention_auc"],
                            group_stability_rank=g["rank"], classification_basis="observed_group_retention")
                representative = next(p for p in catalogue["poses"] if p["id"] == node["source_pose_ids"][0])
                if (Rotation.from_quat(node["representative_quaternion_xyzw"]).inv() * Rotation.from_quat(representative["quaternion_xyzw"])).magnitude() > 1e-8:
                    raise ValueError("Roadmap orientation differs from the chosen representative.")
            doc["group_ranking_method"] = "representative_group_retention_auc"
            _write_json(path, doc)
            handover = yaml.safe_load(path.with_suffix(".yaml").read_text())
            for pose in handover["poses"]:
                g = groups[pose["source_pose_ids"][0]]
                pose.update(group_members=g["members"], representative_group_retention_auc=g["retention_auc"],
                            group_stability_rank=g["rank"], classification_basis="observed_group_retention")
            handover["classification"]["group_ranking_method"] = "representative_group_retention_auc"
            path.with_suffix(".yaml").write_text(yaml.safe_dump(handover, sort_keys=False), encoding="utf-8")
            _write_graphml(doc, path.with_suffix(".graphml"))
            rows = [{"roadmap_pose_id": n["node_id"], "representative_source_pose_id": n["source_pose_ids"][0],
                     "source_pose_id": key} for n in doc["nodes"] for key in n["group_members"]]
            result_manifest = read(staged / "manifest.json")
            result_manifest.update(grouped_export=True, grouping_minimum_frequency_percent=1,
                                   provenance=provenance or {}, status="complete")
        else:
            staged.mkdir()
            (staged / "source").mkdir()
            for name in ("poses.json", "pose_registry.json", "config.json", f"{part}.stl"):
                shutil.copyfile(source.parent / name, staged / "source" / name)
            rows = []
            result_manifest = {"format": "bibazu_observed_roadmap_generation", "schema_version": 1,
                               "workpiece": part, "pose_count": 0, "transition_count": 0,
                               "minimum_frequency_percent": 0, "grouping_minimum_frequency_percent": 1,
                               "renumber_by_frequency": True, "generated_utc": read(source.parent / "manifest.json").get("created_utc", ""),
                               "status": "complete_no_eligible_groups", "grouped_export": True, "provenance": provenance or {}}
            (staged / "README.md").write_text(f"# {part}: no eligible groups\n\n{catalogue['completed_trials']:,} drops completed. No groups meet the 1% cutoff; there is no current roadmap.\n", encoding="utf-8")
        _write_csv(staged / "membership.csv", ["roadmap_pose_id", "representative_source_pose_id", "source_pose_id"], rows)
        for name in ("groups.json", "membership.csv", "group_stability.csv", "grouping_geometry.json", "validation.json",
                     "original_observations.json", "original_pose_registry.json"):
            shutil.copyfile(source.parent / name, staged / "source" / name)
        _write_json(staged / "manifest.json", result_manifest)
        readme = staged / "README.md"
        readme.write_text(readme.read_text() + "\n[Group membership](membership.csv), [all groups](source/groups.json), "
                          "[member stability curves](source/group_stability.csv), [barrier evidence](source/grouping_geometry.json).\n"
                          "\nNodes use the chosen representative orientations. Direct connections remain geometric action candidates.\n", encoding="utf-8")
        inventory(staged)
        verify_inventory(staged)
        _replace(staged, target)
    return result_manifest

