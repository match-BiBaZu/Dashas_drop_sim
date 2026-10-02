"""Renumber a chute pose handover from one simulation's observed frequencies."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

import yaml


_TRANSITION_ID = re.compile(r"^(?P<prefix>[^:]+):(?P<source>\d+)->(?P<target>\d+)(?P<suffix>:.*)?$")
_CLASSIFICATION_IDS = (
    "robust_pose_ids", "metastable_pose_ids", "unresolved_metastable_pose_ids",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pose_id(value: object) -> int:
    if isinstance(value, bool) or not (isinstance(value, int) or isinstance(value, str) and value.isdecimal()):
        raise ValueError(f"Invalid roadmap pose ID: {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"Invalid roadmap pose ID: {value!r}")
    return result


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _remap_ids(values: object, mapping: dict[int, int], field: str) -> list[int]:
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list of pose IDs.")
    try:
        return sorted(mapping[_pose_id(value)] for value in values)
    except KeyError as exc:
        raise ValueError(f"{field} refers to a pose absent from the roadmap: {exc.args[0]}") from exc


def _backup_if_exists(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = path.with_name(f"{path.name}.bak.{stamp}")
    shutil.copy2(path, backup)
    return backup


def reindex_roadmaps(
    roadmap_path: Path, summary_path: Path, destination: Path,
) -> tuple[dict[int, int], list[Path], tuple[Path, Path]]:
    """Save frequency-ordered YAML and JSON roadmaps with matching pose IDs.

    The source summary must have been generated with the exact roadmap bytes.
    Ties use the old ID, and unobserved poses follow observed poses.
    Existing destinations are backed up before replacement.
    """
    roadmap_path = Path(roadmap_path).expanduser().resolve()
    summary_path = Path(summary_path).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if roadmap_path.suffix.lower() not in {".yaml", ".yml", ".json"} or destination.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise ValueError("Select a YAML or JSON pose sheet and destination.")
    if not roadmap_path.is_file():
        raise ValueError("Select an existing pose sheet.")
    yaml_suffix = (
        roadmap_path.suffix.lower() if roadmap_path.suffix.lower() in {".yaml", ".yml"}
        else ".yaml" if roadmap_path.with_suffix(".yaml").is_file() else ".yml"
    )
    yaml_source = roadmap_path.with_suffix(yaml_suffix)
    json_source = roadmap_path.with_suffix(".json")
    if not yaml_source.is_file() or not json_source.is_file():
        raise ValueError("Both the YAML and JSON roadmap files must exist beside one another.")
    yaml_destination = destination.with_suffix(yaml_suffix)
    json_destination = destination.with_suffix(".json")
    if yaml_destination == json_destination or len({yaml_source, json_source}) != 2:
        raise ValueError("Invalid paired roadmap paths.")
    if summary_path.name != "summary.json":
        raise ValueError("Select a simulation summary.json file.")
    manifest = _read_json(summary_path.parent / "manifest.json")
    if manifest.get("roadmap_sha256") not in {_sha256(yaml_source), _sha256(json_source)}:
        raise ValueError("The simulation was run with a different version of these roadmaps.")
    summary = _read_json(summary_path)
    frequencies = summary.get("pose_frequencies")
    if not isinstance(frequencies, list):
        raise ValueError("Simulation summary has no pose frequencies.")
    sheet = yaml.safe_load(yaml_source.read_text(encoding="utf-8-sig"))
    if not isinstance(sheet, dict) or sheet.get("format") != "bibazu_pose_roadmap_handover" or sheet.get("schema_version") != 1:
        raise ValueError("Expected a version 1 BiBaZu YAML pose handover.")
    poses = sheet.get("poses")
    if not isinstance(poses, list) or not poses:
        raise ValueError("The pose sheet contains no poses.")
    old_ids: list[int] = []
    for pose in poses:
        if not isinstance(pose, dict):
            raise ValueError("Every pose must be a mapping.")
        old_ids.append(_pose_id(pose.get("id")))
    if len(old_ids) != len(set(old_ids)):
        raise ValueError("The pose sheet contains duplicate IDs.")
    catalogue_by_id = {
        _pose_id(pose["id"]): set(pose.get("equivalent_catalog_pose_ids", [])) for pose in poses
    }
    graph = _read_json(json_source)
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if graph.get("schema_version") != 1 or not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Expected a version 1 BiBaZu JSON roadmap.")
    node_ids = [_pose_id(node.get("node_id")) for node in nodes if isinstance(node, dict)]
    if len(node_ids) != len(nodes) or set(node_ids) != set(old_ids) or len(node_ids) != len(set(node_ids)):
        raise ValueError("YAML and JSON pose IDs do not match.")
    for node in nodes:
        if set(node.get("pose_ids", [])) != catalogue_by_id[_pose_id(node["node_id"])]:
            raise ValueError("YAML and JSON catalogue pose groups do not match.")

    counts = {pose_id: 0 for pose_id in old_ids}
    seen: set[int] = set()
    for row in frequencies:
        if not isinstance(row, dict):
            raise ValueError("Every frequency row must be a mapping.")
        if row.get("category") != "pose":
            continue
        raw_id = row.get("pose_id")
        if isinstance(raw_id, str) and not raw_id.isdecimal():
            continue  # Unknown orientation groups have labels such as unknown_001.
        pose_id = _pose_id(raw_id)
        if pose_id not in counts or pose_id in seen:
            raise ValueError(f"Unexpected or duplicate pose ID in simulation frequencies: {pose_id}")
        count = row.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Invalid simulation count for pose {pose_id}.")
        counts[pose_id] = count
        seen.add(pose_id)
    if not any(counts.values()):
        raise ValueError("The simulation contains no matched poses to rank.")

    order = sorted(old_ids, key=lambda pose_id: (-counts[pose_id], pose_id))
    mapping = {old_id: new_id for new_id, old_id in enumerate(order)}
    for pose in poses:
        pose["id"] = mapping[_pose_id(pose["id"])]
    poses.sort(key=lambda pose: pose["id"])

    classification = sheet.get("classification")
    if not isinstance(classification, dict):
        raise ValueError("The pose sheet has no classification section.")
    for field in _CLASSIFICATION_IDS:
        if field in classification:
            classification[field] = _remap_ids(classification[field], mapping, field)

    transitions = sheet.get("transitions", [])
    if not isinstance(transitions, list):
        raise ValueError("The pose sheet transitions must be a list.")
    if {item.get("id") for item in transitions if isinstance(item, dict)} != {
        item.get("edge_id") for item in edges if isinstance(item, dict)
    } or len(transitions) != len(edges):
        raise ValueError("YAML and JSON transitions do not match.")
    for transition in transitions:
        if not isinstance(transition, dict):
            raise ValueError("Every transition must be a mapping.")
        source = _pose_id(transition.get("from_pose"))
        target = _pose_id(transition.get("to_pose"))
        if source not in mapping or target not in mapping:
            raise ValueError("A transition refers to a pose absent from the roadmap.")
        label = transition.get("id")
        parsed = _TRANSITION_ID.fullmatch(label) if isinstance(label, str) else None
        if parsed is None or int(parsed["source"]) != source or int(parsed["target"]) != target:
            raise ValueError(f"Transition label does not match its pose endpoints: {label!r}")
        transition["from_pose"] = mapping[source]
        transition["to_pose"] = mapping[target]
        transition["id"] = f"{parsed['prefix']}:{mapping[source]}->{mapping[target]}{parsed['suffix'] or ''}"

    for node in nodes:
        node["node_id"] = mapping[_pose_id(node["node_id"])]
    nodes.sort(key=lambda node: node["node_id"])
    if "unresolved_metastable_node_ids" in graph:
        graph["unresolved_metastable_node_ids"] = _remap_ids(
            graph["unresolved_metastable_node_ids"], mapping, "unresolved_metastable_node_ids"
        )
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("Every JSON edge must be a mapping.")
        source = _pose_id(edge.get("source"))
        target = _pose_id(edge.get("target"))
        if source not in mapping or target not in mapping:
            raise ValueError("A JSON edge refers to a pose absent from the roadmap.")
        label = edge.get("edge_id")
        parsed = _TRANSITION_ID.fullmatch(label) if isinstance(label, str) else None
        if parsed is None or int(parsed["source"]) != source or int(parsed["target"]) != target:
            raise ValueError(f"JSON edge label does not match its pose endpoints: {label!r}")
        edge["source"] = mapping[source]
        edge["target"] = mapping[target]
        edge["edge_id"] = f"{parsed['prefix']}:{mapping[source]}->{mapping[target]}{parsed['suffix'] or ''}"

    yaml_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        yaml_destination: yaml.safe_dump(sheet, allow_unicode=True, sort_keys=False, width=120),
        json_destination: json.dumps(graph, ensure_ascii=False, indent=2) + "\n",
    }
    temporary_files: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    replaced: list[Path] = []
    try:
        for path, content in payloads.items():
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary_files[path] = temporary
        for path in payloads:
            backup = _backup_if_exists(path)
            if backup is not None:
                backups[path] = backup
        for path, temporary in temporary_files.items():
            os.replace(temporary, path)
            replaced.append(path)
    except Exception:
        for path in reversed(replaced):
            if path in backups:
                shutil.copy2(backups[path], path)
            else:
                path.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporary_files.values():
            temporary.unlink(missing_ok=True)
    return mapping, list(backups.values()), (yaml_destination, json_destination)
