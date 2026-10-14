"""Selective generation shared by the GUI and a JSON-config batch CLI.

Run: python -m chute_pose.generate --config run.json
The GUI sends the same document on stdin; progress is emitted as JSON lines.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import tempfile
import traceback

from .metrics import METHOD_LABELS, metric_lines
from .roadmap import (build_pose_roadmap, render_pose_roadmap, save_roadmap_graphml,
                      save_roadmap_json, save_roadmap_yaml)
from .visualization import render_pose_sheets

OUTPUTS = {"yaml": "YAML", "json": "JSON", "graphml": "GraphML",
           "roadmap_svg": "Roadmap SVG", "roadmap_png": "Roadmap PNG",
           "comparison_svg": "Metric comparison pose sheets SVG",
           "comparison_png": "Metric comparison pose sheets PNG",
           "poses_svg": "Pose sheets SVG", "poses_png": "Pose sheets PNG"}


@dataclass(frozen=True)
class GenerationConfig:
    output_dir: str
    outputs: tuple[str, ...] = ("yaml", "roadmap_svg", "roadmap_png")
    methods: tuple[str, ...] = ("rocking",)
    ranking: str = "rocking"
    classifier: str = "rocking"
    alpha: float = 45.0
    beta: float = 0.0
    rocking_threshold: float = 0.20
    cwsa_threshold: float = 0.65
    classical_threshold: float | None = None
    robust_only: bool = True
    pose_sheets_include_metastable: bool = False
    geometry_status: str = "provisional"
    existing: str = "skip"
    columns: int = 6
    poses_per_sheet: int = 24

    def validate(self):
        if not self.outputs or set(self.outputs) - OUTPUTS.keys():
            raise ValueError("Select at least one valid output format.")
        if not self.methods or set(self.methods) - METHOD_LABELS.keys():
            raise ValueError("Select at least one valid method.")
        if self.ranking not in METHOD_LABELS or self.classifier not in METHOD_LABELS:
            raise ValueError("Invalid ordering or classification method.")
        if self.ranking not in self.methods or self.classifier not in self.methods:
            raise ValueError("The ordering and classification methods must also be enabled.")
        if not all(math.isfinite(v) for v in (self.alpha, self.beta, self.rocking_threshold, self.cwsa_threshold)):
            raise ValueError("Angles and thresholds must be finite.")
        if not 0 < self.alpha < 90 or not -89 < self.beta < 89:
            raise ValueError("Two loaded planes require 0 < X < 90 and -89 < Y < 89 degrees.")
        if self.rocking_threshold < 0 or not 0 <= self.cwsa_threshold <= 1:
            raise ValueError("Invalid rocking or CWSA threshold.")
        if self.classifier in {"standard_csa", "crsa"} and (
            self.classical_threshold is None or not math.isfinite(self.classical_threshold)
            or self.classical_threshold <= 0
        ):
            raise ValueError("Enter a positive experimental CSA/CRSA raw-score cutoff in sr/mm; no paper-standard cutoff exists.")
        if self.existing not in {"skip", "overwrite"} or self.geometry_status not in {"provisional", "verified"}:
            raise ValueError("Invalid output policy or CAD status.")
        if self.columns < 1 or self.poses_per_sheet < 1 or not self.output_dir.strip():
            raise ValueError("Choose an output folder and positive sheet dimensions.")


def robust_subset(roadmap):
    nodes = tuple(n for n in roadmap.nodes if n.kind == "robust")
    ids = {n.node_id for n in nodes}
    catalogue_ids = {pid for n in nodes for pid in n.pose_ids}
    return replace(roadmap, nodes=nodes,
        edges=tuple(e for e in roadmap.edges if e.source in ids and e.target in ids
                    and all(pid in catalogue_ids for pid in e.settling_pose_ids)),
        unresolved_metastable_node_ids=(),
        csa_rocking_fallback_pose_ids=tuple(p for p in roadmap.csa_rocking_fallback_pose_ids if p in catalogue_ids),
        classical_metadata={**roadmap.classical_metadata, "diagnostics": {
            key: value for key, value in roadmap.classical_metadata.get("diagnostics", {}).items()
            if int(key) in catalogue_ids
        }})


def generate_one(mesh: Path, config: GenerationConfig, progress=lambda message: None):
    config.validate()
    if not mesh.is_file() or mesh.suffix.lower() != ".stl":
        raise ValueError(f"Not an STL file: {mesh}")
    root = Path(config.output_dir).expanduser().resolve()
    destination = root / mesh.stem
    if destination.exists() and any(destination.iterdir()) and config.existing == "skip":
        return "skipped", []
    progress("Calculating poses, stability and transitions")
    full = build_pose_roadmap(
        mesh, alpha_deg=config.alpha, beta_deg=config.beta, friction_policy="zero",
        minimum_braking_g=0.0, robust_barrier_threshold_mm=config.rocking_threshold,
        minimum_csa_score=config.cwsa_threshold, minimum_classical_score=config.classical_threshold,
        geometry_status=config.geometry_status, pose_ranking_method=config.ranking,
        robustness_method=config.classifier, include_csa="cwsa" in config.methods,
        classical_methods=tuple(m for m in config.methods if m in {"standard_csa", "crsa"}),
    )
    # Store the run settings inside selected data exports; no unwanted manifest file.
    full = replace(full, classical_metadata={**full.classical_metadata, "generation_settings": asdict(config)})
    roadmap = robust_subset(full) if config.robust_only else full
    root.mkdir(parents=True, exist_ok=True)
    progress(f"Rendering/exporting {len(roadmap.nodes)} poses")
    files = []
    # Stage the complete workpiece before publishing. Individual files are replaced
    # atomically; unrelated previous outputs are never removed.
    with tempfile.TemporaryDirectory(prefix=".pose-stage-", dir=root) as temp:
        staging = Path(temp)
        stem = staging / f"{mesh.stem}_roadmap"
        for fmt, writer in (("yaml", save_roadmap_yaml), ("json", save_roadmap_json), ("graphml", save_roadmap_graphml)):
            if fmt in config.outputs:
                writer(roadmap, stem.with_suffix('.' + fmt))
        roadmap_formats = tuple(fmt for fmt in ("svg", "png") if f"roadmap_{fmt}" in config.outputs)
        if roadmap_formats:
            render_pose_roadmap(roadmap, staging / f"{mesh.stem}_roadmap", formats=roadmap_formats,
                                display_methods=config.methods, rank_reference=full, stable_only=config.robust_only)
        comparison_formats = tuple(fmt for fmt in ("svg", "png") if f"comparison_{fmt}" in config.outputs)
        pose_formats = tuple(fmt for fmt in ("svg", "png") if f"poses_{fmt}" in config.outputs)
        sheet_roadmap = full if config.pose_sheets_include_metastable else robust_subset(full)
        sheet_products = (
            ("comparison_pose_sheets", "comparison", "Metric comparison", comparison_formats),
            ("pose_sheets", "poses", "Pose sheets", pose_formats),
        )
        if sheet_roadmap.nodes:
            lines = metric_lines(full, config.methods)
            labels = {
                n.original_catalog_pose_id: (
                    f"Pose {n.node_id} ({n.kind})" if config.pose_sheets_include_metastable
                    else f"Pose {n.node_id}"
                )
                for n in sheet_roadmap.nodes
            }
            sheet_mode = "robust_and_metastable" if config.pose_sheets_include_metastable else "robust_only"
            class_title = "robust and metastable" if config.pose_sheets_include_metastable else "robust only"
            for folder_name, filename_suffix, product_title, formats in sheet_products:
                if formats:
                    render_pose_sheets(mesh, staging / folder_name / sheet_mode, pose_ids=labels, pose_labels=labels,
                        metric_labels={n.original_catalog_pose_id: lines[n.node_id] for n in sheet_roadmap.nodes},
                        formats=formats, columns=config.columns, poses_per_sheet=config.poses_per_sheet,
                        filename_prefix=f"{mesh.stem}_{filename_suffix}",
                        sheet_title=f"{mesh.stem}: {product_title}; {class_title}; X={config.alpha:g}, Y={config.beta:g}")
        for source in sorted(staging.rglob("*")):
            if source.is_file():
                target = destination / source.relative_to(staging)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
                files.append(str(target))
    return "completed", files


def emit(**event):
    print(json.dumps(event), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    import sys
    try:
        request = json.loads(args.config.read_text(encoding="utf-8") if args.config else sys.stdin.read())
        config = GenerationConfig(**request["settings"])
        config.validate()
        meshes = [Path(p) for p in request["meshes"]]
        if not meshes:
            raise ValueError("Select at least one workpiece.")
    except Exception as error:
        emit(status="error", message=str(error))
        return 1
    failures = 0
    for index, mesh in enumerate(meshes):
        emit(part=mesh.stem, status="running", index=index, total=len(meshes))
        try:
            status, files = generate_one(mesh, config, lambda text: emit(part=mesh.stem, message=text))
            emit(part=mesh.stem, status=status, files=files, index=index + 1, total=len(meshes))
        except Exception as error:
            failures += 1
            emit(part=mesh.stem, status="failed", message=str(error), detail=traceback.format_exc(), index=index + 1, total=len(meshes))
    emit(status="finished", failures=failures)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
