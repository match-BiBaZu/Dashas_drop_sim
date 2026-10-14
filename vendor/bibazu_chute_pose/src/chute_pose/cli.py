"""Command-line entry point for the deterministic chute-pose pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from .contacts import build_pose_catalog
from .csa import (
    CSA_ALGORITHM_LABEL,
    DEFAULT_CSA_CAP_HALF_ANGLE_DEG,
    DEFAULT_CSA_DIRECTION_SAMPLES,
    DEFAULT_CSA_ROBUST_THRESHOLD,
    analyze_contact_wrench_solid_angle,
)
from .disturbance import (
    analyze_disturbance_robustness,
    filter_disturbance_robustness,
)
from .equivalence import PracticalPoseClass, cluster_practical_contact_poses
from .frame import ChuteFrame
from .geometry import GeometryValidationError, inspect_mesh, load_solid_mesh
from .roadmap import (
    DEFAULT_ROADMAP_SYMMETRY_TOLERANCE_MM,
    build_pose_roadmap,
    export_pose_roadmap,
    find_best_route,
    load_roadmap_json,
)
from .rocking import (
    analyze_rocking_barriers,
    filter_finite_disturbance_robustness,
)
from .stability import STABILITY_ALGORITHM_LABEL, analyze_pose_stability
from .step_verification import StepSupportUnavailable, verify_step_symmetry
from .symmetry import (
    PoseEquivalenceClass,
    detect_rotational_symmetry,
    reduce_catalog_by_symmetry,
)
from .visualization import render_pose_sheets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chute-pose")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Validate one part mesh and print the Step-1 report."
    )
    inspect_parser.add_argument("mesh", type=Path)
    inspect_parser.add_argument("--alpha", type=float, default=45.0)
    inspect_parser.add_argument("--beta", type=float, default=20.0)
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")

    catalog_parser = subparsers.add_parser(
        "catalog", help="Enumerate theoretical simultaneous floor-wall poses."
    )
    catalog_parser.add_argument("mesh", type=Path)
    catalog_parser.add_argument("--json", action="store_true", dest="as_json")

    render_parser = subparsers.add_parser(
        "render", help="Render theoretical poses as grouped PNG contact sheets."
    )
    render_parser.add_argument("mesh", type=Path)
    render_parser.add_argument("--output-dir", type=Path, required=True)
    render_parser.add_argument("--poses-per-sheet", type=int, default=24)
    render_parser.add_argument("--columns", type=int, default=6)
    render_parser.add_argument("--dpi", type=int, default=180)

    stability_parser = subparsers.add_parser(
        "stability", help="Filter theoretical poses by sliding force/moment stability."
    )
    stability_parser.add_argument("mesh", type=Path)
    stability_parser.add_argument("--alpha", type=float, default=45.0)
    stability_parser.add_argument("--beta", type=float, default=20.0)
    stability_parser.add_argument("--onset-alpha", type=float, default=45.0)
    stability_parser.add_argument("--onset-beta", type=float, default=15.0)
    stability_parser.add_argument("--mu-samples", type=int, default=11)
    stability_parser.add_argument(
        "--friction-policy",
        choices=("range", "zero"),
        default="zero",
        help=(
            "'zero' (default) evaluates nominal mu=0 equilibrium only; "
            "'range' additionally requires stability through the inferred "
            "static-friction range."
        ),
    )
    stability_parser.add_argument(
        "--exhaustive-friction-diagnostics",
        action="store_true",
        help=(
            "Evaluate every pose at every friction sample, including rejected "
            "and friction-dependent diagnostics. Large finely tessellated "
            "catalogs otherwise use the exact quasistatic-only fast path."
        ),
    )
    stability_parser.add_argument("--symmetry-tolerance-mm", type=float)
    stability_parser.add_argument("--json", action="store_true", dest="as_json")
    stability_parser.add_argument(
        "--render-output-dir",
        type=Path,
        help="Optionally render only poses stable at every sampled mu value.",
    )
    stability_parser.add_argument(
        "--pose-ranking", choices=("rocking", "csa"), default="rocking"
    )
    stability_parser.add_argument(
        "--csa-cap-half-angle-deg",
        type=float,
        default=DEFAULT_CSA_CAP_HALF_ANGLE_DEG,
    )
    stability_parser.add_argument(
        "--csa-direction-samples",
        type=int,
        default=DEFAULT_CSA_DIRECTION_SAMPLES,
    )

    symmetry_parser = subparsers.add_parser(
        "symmetry", help="Detect STL symmetry, verify it with STEP, and group poses."
    )
    symmetry_parser.add_argument("mesh", type=Path)
    symmetry_parser.add_argument("--step", type=Path)
    symmetry_parser.add_argument("--tolerance-mm", type=float)
    symmetry_parser.add_argument("--angular-tolerance-deg", type=float, default=0.25)
    symmetry_parser.add_argument("--json", action="store_true", dest="as_json")

    disturbance_parser = subparsers.add_parser(
        "disturbance",
        help="Filter nominal poses by critical braking-force and upset-torque reserve.",
    )
    disturbance_parser.add_argument("mesh", type=Path)
    disturbance_parser.add_argument(
        "--pose-ids", help="Comma-separated pose ids; default is the nominal stable set."
    )
    disturbance_parser.add_argument("--alpha", type=float, default=45.0)
    disturbance_parser.add_argument("--beta", type=float, default=20.0)
    disturbance_parser.add_argument("--onset-alpha", type=float, default=45.0)
    disturbance_parser.add_argument("--onset-beta", type=float, default=15.0)
    disturbance_parser.add_argument("--mu-samples", type=int, default=11)
    disturbance_parser.add_argument(
        "--friction-policy",
        choices=("range", "zero"),
        default="zero",
        help=(
            "Select nominal input poses using zero-friction equilibrium "
            "(default) or the inferred static-friction range."
        ),
    )
    disturbance_parser.add_argument("--minimum-braking-g", type=float, default=0.10)
    disturbance_parser.add_argument(
        "--minimum-torque-normalized", type=float, default=0.02
    )
    disturbance_parser.add_argument(
        "--rocking-excursion-deg", type=float, default=5.0
    )
    disturbance_parser.add_argument("--rocking-angle-steps", type=int, default=20)
    disturbance_parser.add_argument("--rocking-axis-samples", type=int, default=2048)
    disturbance_parser.add_argument(
        "--minimum-rocking-barrier-mm", type=float, default=0.20
    )
    disturbance_parser.add_argument("--symmetry-tolerance-mm", type=float)
    disturbance_parser.add_argument(
        "--contact-angle-tolerance-deg", type=float, default=1.0
    )
    disturbance_parser.add_argument(
        "--contact-displacement-tolerance-mm", type=float, default=0.5
    )
    disturbance_parser.add_argument("--render-output-dir", type=Path)
    disturbance_parser.add_argument(
        "--pose-ranking", choices=("rocking", "csa"), default="rocking"
    )
    disturbance_parser.add_argument(
        "--csa-cap-half-angle-deg",
        type=float,
        default=DEFAULT_CSA_CAP_HALF_ANGLE_DEG,
    )
    disturbance_parser.add_argument(
        "--csa-direction-samples",
        type=int,
        default=DEFAULT_CSA_DIRECTION_SAMPLES,
    )
    disturbance_parser.add_argument("--json", action="store_true", dest="as_json")

    roadmap_parser = subparsers.add_parser(
        "roadmap",
        help="Build the pose roadmap and export JSON/YAML/GraphML/images.",
    )
    roadmap_parser.add_argument("mesh", type=Path)
    roadmap_parser.add_argument("--output-dir", type=Path, required=True)
    roadmap_parser.add_argument(
        "--comparison-plots",
        action="store_true",
        help=(
            "Also compute CWSA and export full/stable-only comparison PNG/SVG "
            "plots with shared rocking ranks (CWSA ranks)."
        ),
    )
    roadmap_parser.add_argument("--alpha", type=float, default=45.0)
    roadmap_parser.add_argument("--beta", type=float, default=20.0)
    roadmap_parser.add_argument("--onset-alpha", type=float, default=45.0)
    roadmap_parser.add_argument("--onset-beta", type=float, default=15.0)
    roadmap_parser.add_argument(
        "--friction-policy",
        choices=("range", "zero"),
        default="zero",
        help=(
            "Select roadmap input poses and CWSA using mu=0 (default, no sliding loads) "
            "or the inferred sliding-friction range."
        ),
    )
    roadmap_parser.add_argument(
        "--symmetry-tolerance-mm",
        type=float,
        default=DEFAULT_ROADMAP_SYMMETRY_TOLERANCE_MM,
        help=(
            "Maximum STL mapping error for rotational symmetry; "
            f"default: {DEFAULT_ROADMAP_SYMMETRY_TOLERANCE_MM:g} mm."
        ),
    )
    roadmap_parser.add_argument(
        "--expected-symmetry",
        help="Fail instead of exporting when the detected symbol differs, e.g. C3.",
    )
    roadmap_parser.add_argument("--axis-tolerance-deg", type=float, default=1.0)
    roadmap_parser.add_argument(
        "--surface-displacement-tolerance-mm", type=float, default=0.5
    )
    roadmap_parser.add_argument(
        "--minimum-rocking-barrier-mm", type=float, default=0.73
    )
    roadmap_parser.add_argument(
        "--minimum-braking-g",
        "--minimum-face-face-braking-g",
        dest="minimum_braking_g",
        type=float,
        default=0.0,
        help="Optional braking reserve for every roadmap pose; default 0 disables braking checks.",
    )
    roadmap_parser.add_argument(
        "--pose-ranking", choices=("rocking", "csa", "cwsa", "standard_csa", "crsa"), default="rocking"
    )
    roadmap_parser.add_argument(
        "--robustness-method", choices=("rocking", "csa", "cwsa", "standard_csa", "crsa"), default="rocking"
    )
    roadmap_parser.add_argument("--classical-methods", nargs="+", choices=("standard_csa", "crsa"), default=())
    roadmap_parser.add_argument("--minimum-classical-score", type=float,
                                help="Explicit experimental CSA/CRSA raw cutoff in sr/mm; required for classification.")
    roadmap_parser.add_argument(
        "--minimum-csa-score",
        type=float,
        default=DEFAULT_CSA_ROBUST_THRESHOLD,
        help=(
            "Provisional CWSA robust cutoff in 0..1; used only with "
            "--robustness-method csa."
        ),
    )
    roadmap_parser.add_argument(
        "--csa-cap-half-angle-deg",
        type=float,
        default=DEFAULT_CSA_CAP_HALF_ANGLE_DEG,
    )
    roadmap_parser.add_argument(
        "--csa-direction-samples",
        type=int,
        default=DEFAULT_CSA_DIRECTION_SAMPLES,
    )
    roadmap_parser.add_argument(
        "--opposite-x-min-height-mm",
        type=float,
        default=25.0,
        help=(
            "Minimum available broad main-face support span for +X on floor "
            "or -X on wall."
        ),
    )
    roadmap_parser.add_argument(
        "--geometry-status",
        choices=("provisional", "verified"),
        default="provisional",
    )
    roadmap_parser.add_argument("--json", action="store_true", dest="as_json")

    route_parser = subparsers.add_parser(
        "route", help="Find the highest-scoring open-loop route in a roadmap JSON."
    )
    route_parser.add_argument("roadmap", type=Path)
    route_parser.add_argument("--start-pose", type=int, required=True)
    route_parser.add_argument("--target-pose", type=int, required=True)
    route_parser.add_argument("--max-actions", type=int, default=4)
    route_parser.add_argument("--output", type=Path)
    return parser


def _format_pose_plot_metrics(
    *,
    rocking_barrier_mm: float,
    csa_stability_index: float | None,
    show_csa: bool,
) -> str:
    """Format the stability metrics shown beneath a pose number."""

    lines: list[str] = []
    if show_csa:
        lines.append(
            f"CWSA stability index: {csa_stability_index:.3f}"
            if csa_stability_index is not None
            else "CWSA stability index: n/a (rolling contact)"
        )
    lines.append(f"Rocking barrier: {rocking_barrier_mm:.3f} mm")
    return "".join(f"\n{line}" for line in lines)


def _inspect(args: argparse.Namespace) -> int:
    frame = ChuteFrame(alpha_deg=args.alpha, beta_deg=args.beta)
    report = inspect_mesh(args.mesh)
    gravity = frame.gravity_chute()

    result = {
        "coordinate_system": {
            "handedness": "right-handed",
            "x": "downhill along chute",
            "y": "away from wall; admissible interior y >= 0",
            "z": "away from floor; admissible interior z >= 0",
            "floor": "z = 0",
            "wall": "y = 0",
            "floor_wall_seam": "(x, 0, 0)",
            "contact_modes": ["floor_wall"],
        },
        "orientation": {
            "alpha_deg_moved_x": frame.alpha_deg,
            "beta_deg_original_y": frame.beta_deg,
            "rotation_order": "R_y(beta) @ R_x(alpha)",
            "gravity_chute_m_s2": [float(value) for value in gravity],
        },
        "geometry": report.to_dict(),
    }

    if args.as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Mesh: {report.source}")
        print(f"SHA-256: {report.sha256}")
        print(f"Units: {report.units}")
        print(f"Extents: {report.extents_mm} mm")
        print(f"Volume: {report.volume_mm3:.6f} mm^3")
        print(f"Center of mass: {report.center_mass_mm} mm")
        print(
            "Convex hull: "
            f"{report.hull_vertex_count} vertices, "
            f"{report.hull_face_count} triangles, "
            f"{report.hull_plane_count} oriented planes"
        )
        print(
            "Chute angles: "
            f"alpha={frame.alpha_deg:g} deg about moved X, "
            f"beta={frame.beta_deg:g} deg about original Y"
        )
        print(
            "Gravity in chute frame: "
            f"({gravity[0]:.6f}, {gravity[1]:.6f}, {gravity[2]:.6f}) m/s^2"
        )
        print("Required contact mode: floor_wall")
    return 0


def _catalog(args: argparse.Namespace) -> int:
    catalog = build_pose_catalog(args.mesh)
    if args.as_json:
        print(json.dumps(catalog.to_dict(), indent=2, ensure_ascii=False))
        return 0

    type_counts: dict[str, int] = {}
    for pose in catalog.poses:
        key = f"{pose.floor_contact_type}-{pose.wall_contact_type}"
        type_counts[key] = type_counts.get(key, 0) + 1

    print(f"Mesh: {catalog.source}")
    print(f"Convex support faces: {len(catalog.support_faces)}")
    print(f"Theoretical floor-wall poses: {len(catalog.poses)}")
    for contact_type, count in sorted(type_counts.items()):
        print(f"  {contact_type}: {count}")
    print("Point contacts excluded: yes")
    print("Edge-edge contacts excluded as non-isolated: yes")
    print("Run the 'stability' command for angle-dependent filtering (Step 3).")
    return 0


def _render(args: argparse.Namespace) -> int:
    sheets = render_pose_sheets(
        args.mesh,
        args.output_dir,
        poses_per_sheet=args.poses_per_sheet,
        columns=args.columns,
        dpi=args.dpi,
    )
    print(f"Rendered {len(sheets)} contact sheets:")
    for sheet in sheets:
        first_pose = sheet.pose_ids[0]
        last_pose = sheet.pose_ids[-1]
        print(f"  {sheet.path} (poses {first_pose}-{last_pose})")
    return 0


def _stability(args: argparse.Namespace) -> int:
    catalog = build_pose_catalog(args.mesh)
    quasistatic_only = (
        len(catalog.poses) >= 1000
        and not args.exhaustive_friction_diagnostics
    )
    if quasistatic_only and not args.as_json:
        print(
            f"Large catalog ({len(catalog.poses)} poses): using the "
            "zero-friction quasistatic prefilter."
        )
        print(
            "Use --exhaustive-friction-diagnostics only when rejected and "
            "friction-dependent pose classifications are required."
        )
    analysis = analyze_pose_stability(
        args.mesh,
        alpha_deg=args.alpha,
        beta_deg=args.beta,
        onset_alpha_deg=args.onset_alpha,
        onset_beta_deg=args.onset_beta,
        mu_samples=args.mu_samples,
        catalog=catalog,
        quasistatic_only=quasistatic_only,
        friction_policy=args.friction_policy,
    )
    detected_symmetry = detect_rotational_symmetry(
        args.mesh, tolerance_mm=args.symmetry_tolerance_mm
    )
    verification = None
    if detected_symmetry.is_continuous:
        symmetry = detected_symmetry
        symmetry_policy = "continuous_convex_support_pretest"
    elif args.symmetry_tolerance_mm is not None:
        symmetry = detected_symmetry
        symmetry_policy = "explicit_practical_tolerance"
    elif detected_symmetry.order == 1:
        symmetry = detected_symmetry
        symmetry_policy = "no_nontrivial_symmetry"
    else:
        step_path = _matching_step_path(args.mesh)
        try:
            verification = (
                verify_step_symmetry(step_path, detected_symmetry)
                if step_path is not None
                else None
            )
        except StepSupportUnavailable:
            verification = None
        if verification is not None and verification.exact_confirmed:
            symmetry = detected_symmetry
            symmetry_policy = "exact_step_confirmation"
        else:
            symmetry = replace(
                detected_symmetry,
                symbol="C1",
                elements=(detected_symmetry.elements[0],),
            )
            symmetry_policy = "not_merged_without_exact_step_confirmation"
    reduced = reduce_catalog_by_symmetry(catalog, symmetry)
    stable_ids = set(analysis.stable_pose_ids)
    rocking = analyze_rocking_barriers(
        args.mesh,
        pose_ids=analysis.stable_pose_ids,
        alpha_deg=args.alpha,
        beta_deg=args.beta,
        catalog=catalog,
    )
    rocking_by_pose_id = {value.pose_id: value for value in rocking.barriers}
    csa = (
        analyze_contact_wrench_solid_angle(
            args.mesh,
            pose_ids=analysis.stable_pose_ids,
            alpha_deg=args.alpha,
            beta_deg=args.beta,
            onset_alpha_deg=args.onset_alpha,
            onset_beta_deg=args.onset_beta,
            mu_samples=args.mu_samples,
            cap_half_angle_deg=args.csa_cap_half_angle_deg,
            direction_samples=args.csa_direction_samples,
            friction_policy=args.friction_policy,
            catalog=catalog,
        )
        if args.pose_ranking == "csa"
        else None
    )
    csa_by_pose_id = (
        {value.pose_id: value for value in csa.poses} if csa is not None else {}
    )

    def class_ranking_key(
        value: PoseEquivalenceClass,
    ) -> tuple[float | bool | int, ...]:
        pose_ids = stable_ids.intersection(value.pose_ids)
        if args.pose_ranking == "csa":
            class_values = tuple(csa_by_pose_id[pose_id] for pose_id in pose_ids)
            applicable = all(item.applicable for item in class_values)
            score = min(item.score for item in class_values) if applicable else -1.0
            return (
                not applicable,
                -round(score, 6),
                -min(
                    rocking_by_pose_id[pose_id].barrier_height_mm
                    for pose_id in pose_ids
                ),
                value.representative_pose_id,
            )
        return (
            -min(
                rocking_by_pose_id[pose_id].barrier_height_mm
                for pose_id in pose_ids
            ),
            value.representative_pose_id,
        )

    ranked_stable_classes = tuple(
        sorted(
            (
                value
                for value in reduced.classes
                if stable_ids.intersection(value.pose_ids)
            ),
            key=class_ranking_key,
        )
    )
    stable_class_representatives = tuple(
        min(stable_ids.intersection(value.pose_ids))
        for value in ranked_stable_classes
    )
    stability_by_pose_id = {value.pose_id: value for value in analysis.poses}
    stable_pose_labels = {}
    ranked_class_payloads = []
    for pose_number, value in enumerate(ranked_stable_classes):
        stable_class_ids = stable_ids.intersection(value.pose_ids)
        representative = min(stable_class_ids)
        minimum_margin = min(
            stability_by_pose_id[pose_id].minimum_pressure_margin
            for pose_id in stable_class_ids
        )
        minimum_barrier = min(
            rocking_by_pose_id[pose_id].barrier_height_mm
            for pose_id in stable_class_ids
        )
        class_csa_values = tuple(
            csa_by_pose_id[pose_id]
            for pose_id in stable_class_ids
            if pose_id in csa_by_pose_id
        )
        minimum_csa = (
            min(item.score for item in class_csa_values)
            if class_csa_values and all(item.applicable for item in class_csa_values)
            else None
        )
        ranking_label = _format_pose_plot_metrics(
            rocking_barrier_mm=minimum_barrier,
            csa_stability_index=minimum_csa,
            show_csa=args.pose_ranking == "csa",
        )
        stable_pose_labels[representative] = (
            f"Pose {pose_number}"
            f"{ranking_label}"
            f"\nContact load balance index: {minimum_margin:.3f}"
        )
        ranked_class_payloads.append(
            {
                "pose_number": pose_number,
                "original_catalog_pose_ids": value.pose_ids,
                "rocking_barrier_mm": minimum_barrier,
                "csa_stability_index": minimum_csa,
                "contact_load_balance_index": minimum_margin,
            }
        )
    if args.render_output_dir is not None:
        part_name = args.mesh.stem
        render_pose_sheets(
            args.mesh,
            args.render_output_dir,
            pose_ids=stable_class_representatives,
            sheet_title=(
                f"{part_name}: quasi-statically admissible sliding poses "
                f"at alpha={args.alpha:g} deg, "
                f"beta={args.beta:g} deg\n"
                f"Algorithm: {STABILITY_ALGORITHM_LABEL}; "
                + (
                    "friction criterion: mu=0 only; "
                    if analysis.friction_policy == "zero"
                    else "friction criterion: full sampled range; "
                )
                + (
                    f"ranking: {CSA_ALGORITHM_LABEL}; "
                    if args.pose_ranking == "csa"
                    else "ranking: finite rocking barrier; "
                )
                + f"pose number = descending {args.pose_ranking} stability metric"
            ),
            filename_prefix=f"{part_name}_quasistatic",
            pose_labels=stable_pose_labels,
            catalog=catalog,
        )

    if args.as_json:
        result = analysis.to_dict()
        result["pose_ranking_method"] = args.pose_ranking
        result["csa"] = csa.to_dict() if csa is not None else None
        result["ranked_physical_pose_classes"] = ranked_class_payloads
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    type_counts: dict[str, list[int]] = {}
    for pose in catalog.poses:
        key = f"{pose.floor_contact_type}-{pose.wall_contact_type}"
        type_counts.setdefault(key, [0, 0])[1] += 1
    for result in analysis.poses:
        if result.stable_across_range:
            key = f"{result.floor_contact_type}-{result.wall_contact_type}"
            type_counts.setdefault(key, [0, 0])[0] += 1

    estimate = analysis.friction_estimate
    print(f"Mesh: {analysis.source}")
    print(
        "Chute angles: "
        f"alpha={analysis.alpha_deg:g} deg, beta={analysis.beta_deg:g} deg"
    )
    if analysis.friction_policy == "zero":
        print("Friction policy: nominal zero-friction equilibrium only (mu=0)")
        print(
            "Quasi-statically admissible at mu=0: "
            f"{len(analysis.stable_pose_ids)}/{analysis.input_pose_count}"
        )
    else:
        print(
            "Friction estimate from slide onset: "
            f"mu_s={estimate.mu_static_estimate:.6f} "
            f"(alpha={estimate.onset_alpha_deg:g} deg, "
            f"beta={estimate.onset_beta_deg:g} deg)"
        )
        print(
            f"Robust interval: 0 <= mu <= {estimate.mu_static_estimate:.6f} "
            f"at {len(analysis.mu_values)} samples"
        )
        print(
            f"Quasi-statically admissible at every sampled coefficient: "
            f"{len(analysis.stable_pose_ids)}/{analysis.input_pose_count}"
        )
    for contact_type, (stable_count, total_count) in sorted(type_counts.items()):
        print(f"  {contact_type}: {stable_count}/{total_count}")
    print(
        "Quasi-static pose IDs: "
        + ", ".join(str(value) for value in analysis.stable_pose_ids)
    )
    if analysis.friction_policy == "zero":
        print("Friction-range classifications: not evaluated")
    elif analysis.quasistatic_only:
        print(
            "Friction-dependent/rejected diagnostics: skipped by the "
            "quasistatic-only fast path"
        )
    else:
        print(
            "Friction-dependent candidates: "
            f"{len(analysis.friction_dependent_pose_ids)}; "
            f"rejected at every sample: {len(analysis.rejected_pose_ids)}"
        )
    print(
        f"Applied rotation symmetry: {symmetry.symbol} "
        f"(order {symmetry.order}, tolerance {symmetry.tolerance_mm:.6g} mm)"
    )
    print(f"Symmetry merge policy: {symmetry_policy}")
    if detected_symmetry.order > symmetry.order:
        print(
            f"Unmerged STL candidate: {detected_symmetry.symbol} "
            "(requires exact STEP confirmation or explicit tolerance)"
        )
    print(
        "Quasi-static pose classes after symmetry grouping: "
        f"{len(ranked_stable_classes)}"
    )
    print(f"Pose ranking method: {args.pose_ranking}")
    if csa is not None:
        print(
            f"CWSA cap: {csa.cap_half_angle_deg:g} deg; "
            f"equal-area directions: {csa.direction_samples}"
        )
    if args.render_output_dir is not None:
        print(f"Rendered quasi-static pose classes to: {args.render_output_dir.resolve()}")
    return 0


def _matching_step_path(mesh_path: Path) -> Path | None:
    for candidate in mesh_path.parent.glob(f"{mesh_path.stem}.*"):
        if candidate.suffix.lower() in {".step", ".stp"}:
            return candidate
    return None


def _symmetry(args: argparse.Namespace) -> int:
    catalog = build_pose_catalog(args.mesh)
    symmetry = detect_rotational_symmetry(
        args.mesh, tolerance_mm=args.tolerance_mm
    )
    reduced = reduce_catalog_by_symmetry(
        catalog,
        symmetry,
        angular_tolerance_deg=args.angular_tolerance_deg,
    )
    step_path = args.step or _matching_step_path(args.mesh)
    verification = (
        verify_step_symmetry(step_path, symmetry)
        if step_path and not symmetry.is_continuous
        else None
    )

    if args.as_json:
        result = reduced.to_dict()
        result["step_verification"] = (
            verification.to_dict() if verification is not None else None
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    nontrivial_classes = [value for value in reduced.classes if len(value.pose_ids) > 1]
    print(f"Mesh: {catalog.source}")
    print(
        f"STL symmetry candidate: {symmetry.symbol}, "
        f"{'continuous' if symmetry.is_continuous else f'order {symmetry.order}'}, "
        f"practical tolerance {symmetry.tolerance_mm:.6g} mm"
    )
    if symmetry.is_continuous and step_path is not None:
        print(
            "STEP verification: finite Boolean rotation check skipped for Cinf; "
            "convex-support result remains provisional"
        )
    elif verification is None:
        print("STEP verification: no matching STEP file found")
    else:
        print(f"STEP verification: {verification.status}")
        if verification.checks:
            maximum_difference = max(
                check.relative_symmetric_difference for check in verification.checks
            )
            print(
                "Maximum STEP symmetric-volume difference: "
                f"{100.0 * maximum_difference:.6f}%"
            )
    print(
        f"Pose rotations: {len(catalog.poses)} -> "
        f"{len(reduced.classes)} physical pose classes"
    )
    print(f"Non-singleton equivalence classes: {len(nontrivial_classes)}")
    for value in nontrivial_classes:
        print(
            f"  class {value.class_id}, representative {value.representative_pose_id}: "
            + ", ".join(str(pose_id) for pose_id in value.pose_ids)
        )
    return 0


def _parse_pose_ids(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        pose_ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--pose-ids must be a comma-separated list of integers.") from exc
    if not pose_ids:
        raise ValueError("--pose-ids must contain at least one integer.")
    return pose_ids


def _disturbance(args: argparse.Namespace) -> int:
    catalog = build_pose_catalog(args.mesh)
    requested_pose_ids = _parse_pose_ids(args.pose_ids)
    if requested_pose_ids is None:
        nominal = analyze_pose_stability(
            args.mesh,
            alpha_deg=args.alpha,
            beta_deg=args.beta,
            onset_alpha_deg=args.onset_alpha,
            onset_beta_deg=args.onset_beta,
            mu_samples=args.mu_samples,
            catalog=catalog,
            quasistatic_only=True,
            friction_policy=args.friction_policy,
        )
        requested_pose_ids = nominal.stable_pose_ids
    analysis = analyze_disturbance_robustness(
        args.mesh,
        pose_ids=requested_pose_ids,
        alpha_deg=args.alpha,
        beta_deg=args.beta,
        onset_alpha_deg=args.onset_alpha,
        onset_beta_deg=args.onset_beta,
        mu_samples=args.mu_samples,
        catalog=catalog,
    )
    filtered = filter_disturbance_robustness(
        analysis,
        minimum_braking_g=args.minimum_braking_g,
        minimum_torque_normalized=args.minimum_torque_normalized,
    )
    rocking = analyze_rocking_barriers(
        args.mesh,
        pose_ids=requested_pose_ids,
        alpha_deg=args.alpha,
        beta_deg=args.beta,
        excursion_deg=args.rocking_excursion_deg,
        angle_steps=args.rocking_angle_steps,
        axis_samples=args.rocking_axis_samples,
        catalog=catalog,
    )
    finite_filtered = filter_finite_disturbance_robustness(
        rocking,
        analysis,
        catalog,
        minimum_barrier_height_mm=args.minimum_rocking_barrier_mm,
        minimum_braking_g=args.minimum_braking_g,
    )
    csa = (
        analyze_contact_wrench_solid_angle(
            args.mesh,
            pose_ids=requested_pose_ids,
            alpha_deg=args.alpha,
            beta_deg=args.beta,
            onset_alpha_deg=args.onset_alpha,
            onset_beta_deg=args.onset_beta,
            mu_samples=args.mu_samples,
            cap_half_angle_deg=args.csa_cap_half_angle_deg,
            direction_samples=args.csa_direction_samples,
            friction_policy=args.friction_policy,
            catalog=catalog,
        )
        if args.pose_ranking == "csa"
        else None
    )
    mesh = load_solid_mesh(args.mesh)
    vertices_centered = np.asarray(mesh.vertices, dtype=float) - np.asarray(
        mesh.center_mass, dtype=float
    )
    symmetry = (
        detect_rotational_symmetry(
            args.mesh, tolerance_mm=args.symmetry_tolerance_mm
        )
        if args.symmetry_tolerance_mm is not None
        else None
    )
    clustering = cluster_practical_contact_poses(
        catalog,
        vertices_centered,
        requested_pose_ids,
        symmetry=symmetry,
        angular_tolerance_deg=args.contact_angle_tolerance_deg,
        surface_displacement_tolerance_mm=max(
            args.contact_displacement_tolerance_mm,
            symmetry.tolerance_mm if symmetry is not None else 0.0,
        ),
    )
    capacities = {value.pose_id: value for value in analysis.capacities}
    barriers = {value.pose_id: value for value in rocking.barriers}
    csa_by_pose_id = (
        {value.pose_id: value for value in csa.poses} if csa is not None else {}
    )
    accepted_ids = set(finite_filtered.accepted_pose_ids)

    def class_ranking_key(
        pose_class: PracticalPoseClass,
    ) -> tuple[float | bool | int, ...]:
        if args.pose_ranking == "csa":
            class_values = tuple(
                csa_by_pose_id[pose_id] for pose_id in pose_class.pose_ids
            )
            applicable = all(value.applicable for value in class_values)
            score = min(value.score for value in class_values) if applicable else -1.0
            return (
                not all(pose_id in accepted_ids for pose_id in pose_class.pose_ids),
                not applicable,
                -round(score, 6),
                -min(
                    barriers[pose_id].barrier_height_mm
                    for pose_id in pose_class.pose_ids
                ),
                pose_class.representative_pose_id,
            )
        return (
            -min(
                barriers[pose_id].barrier_height_mm
                for pose_id in pose_class.pose_ids
            ),
            pose_class.representative_pose_id,
        )

    ranked_classes = tuple(
        sorted(
            clustering.classes,
            key=class_ranking_key,
        )
    )
    pose_number_by_class_id = {
        pose_class.class_id: pose_number
        for pose_number, pose_class in enumerate(ranked_classes)
    }
    robust_classes = tuple(
        pose_class
        for pose_class in ranked_classes
        if all(pose_id in accepted_ids for pose_id in pose_class.pose_ids)
    )
    robust_representation_count = sum(
        len(pose_class.pose_ids) for pose_class in robust_classes
    )

    if args.render_output_dir is not None:
        labels = {}
        for pose_class in robust_classes:
            label = f"Pose {pose_number_by_class_id[pose_class.class_id]}"
            minimum_barrier = min(
                barriers[value].barrier_height_mm
                for value in pose_class.pose_ids
            )
            minimum_csa = None
            if args.pose_ranking == "csa":
                class_values = tuple(
                    csa_by_pose_id[pose_id] for pose_id in pose_class.pose_ids
                )
                minimum_csa = (
                    min(value.score for value in class_values)
                    if all(value.applicable for value in class_values)
                    else None
                )
            label += _format_pose_plot_metrics(
                rocking_barrier_mm=minimum_barrier,
                csa_stability_index=minimum_csa,
                show_csa=args.pose_ranking == "csa",
            )
            labels[pose_class.representative_pose_id] = label
        render_pose_sheets(
            args.mesh,
            args.render_output_dir,
            pose_ids=labels,
            sheet_title=(
                f"{args.mesh.stem}: disturbance-robust sliding poses "
                f"at alpha={args.alpha:g} deg, beta={args.beta:g} deg\n"
                f"pose number = descending {args.pose_ranking} stability metric"
            ),
            filename_prefix=f"{args.mesh.stem}_disturbance_robust",
            pose_labels=labels,
            catalog=catalog,
        )

    if args.as_json:
        result = analysis.to_dict()
        result["filter"] = filtered.to_dict()
        result["rocking"] = rocking.to_dict()
        result["finite_disturbance_filter"] = finite_filtered.to_dict()
        result["pose_ranking_method"] = args.pose_ranking
        result["csa"] = csa.to_dict() if csa is not None else None
        result["practical_clustering"] = clustering.to_dict()
        result["robust_practical_classes"] = [
            {
                "pose_number": pose_number_by_class_id[pose_class.class_id],
                **pose_class.to_dict(),
            }
            for pose_class in robust_classes
        ]
        result["ranked_physical_pose_classes"] = [
            {
                "pose_number": pose_number_by_class_id[pose_class.class_id],
                "original_catalog_pose_ids": pose_class.pose_ids,
                "rocking_barrier_mm": min(
                    barriers[value].barrier_height_mm
                    for value in pose_class.pose_ids
                ),
                "csa_stability_index": (
                    min(
                        csa_by_pose_id[value].score
                        for value in pose_class.pose_ids
                    )
                    if csa is not None
                    and all(
                        csa_by_pose_id[value].applicable
                        for value in pose_class.pose_ids
                    )
                    else None
                ),
                "robust": all(
                    pose_id in accepted_ids for pose_id in pose_class.pose_ids
                ),
            }
            for pose_class in ranked_classes
        ]
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print(f"Mesh: {analysis.source}")
    print(f"Nominal input poses: {len(requested_pose_ids)}")
    print(f"Pose ranking method: {args.pose_ranking}")
    print(
        "Finite disturbance thresholds: "
        f"rocking barrier >= {finite_filtered.minimum_barrier_height_mm:.6g} mm; "
        f"braking >= {finite_filtered.minimum_braking_g:.6g} g"
    )
    print(
        f"Disturbance-robust representations in complete classes: "
        f"{robust_representation_count}; practical pose classes: "
        f"{len(robust_classes)}"
    )
    for pose_class in robust_classes:
        member_capacities = [capacities[value] for value in pose_class.pose_ids]
        print(
            f"  Pose {pose_number_by_class_id[pose_class.class_id]}"
            + f": braking={min(value.critical_braking_g for value in member_capacities):.6f} g"
            + ", torque="
            + f"{min(value.critical_torque_normalized for value in member_capacities):.6f}"
            + ", rocking="
            + f"{min(barriers[value].barrier_height_mm for value in pose_class.pose_ids):.6f} mm"
        )
    if args.render_output_dir is not None:
        print(f"Rendered robust classes to: {args.render_output_dir.resolve()}")
    return 0


def _roadmap(args: argparse.Namespace) -> int:
    result = build_pose_roadmap(
        args.mesh,
        alpha_deg=args.alpha,
        beta_deg=args.beta,
        onset_alpha_deg=args.onset_alpha,
        onset_beta_deg=args.onset_beta,
        symmetry_tolerance_mm=args.symmetry_tolerance_mm,
        angular_tolerance_deg=args.axis_tolerance_deg,
        surface_displacement_tolerance_mm=(
            args.surface_displacement_tolerance_mm
        ),
        robust_barrier_threshold_mm=args.minimum_rocking_barrier_mm,
        minimum_braking_g=args.minimum_braking_g,
        opposite_x_min_height_mm=args.opposite_x_min_height_mm,
        geometry_status=args.geometry_status,
        pose_ranking_method=args.pose_ranking,
        robustness_method=args.robustness_method,
        minimum_csa_score=args.minimum_csa_score,
        csa_cap_half_angle_deg=args.csa_cap_half_angle_deg,
        csa_direction_samples=args.csa_direction_samples,
        friction_policy=args.friction_policy,
        include_csa=args.comparison_plots,
        classical_methods=tuple(args.classical_methods),
        minimum_classical_score=args.minimum_classical_score,
    )
    if (
        args.expected_symmetry is not None
        and result.symmetry_symbol.casefold() != args.expected_symmetry.casefold()
    ):
        raise ValueError(
            "Expected rotational symmetry "
            f"{args.expected_symmetry}, detected {result.symmetry_symbol} "
            f"at {result.symmetry_tolerance_mm:g} mm tolerance."
        )
    paths = export_pose_roadmap(
        result, args.output_dir, comparison_plots=args.comparison_plots
    )
    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0
    robust_count = sum(node.kind == "robust" for node in result.nodes)
    metastable_count = sum(node.kind == "metastable" for node in result.nodes)
    actuated_count = sum(edge.transition_kind == "actuated" for edge in result.edges)
    passive_count = sum(edge.transition_kind == "passive_tip" for edge in result.edges)
    print(f"Mesh: {result.source}")
    print(
        "Rotational symmetry: "
        f"{result.symmetry_symbol} "
        f"(STL tolerance {result.symmetry_tolerance_mm:g} mm)"
    )
    print(
        f"Roadmap nodes: {len(result.nodes)} "
        f"({robust_count} robust, {metastable_count} metastable)"
    )
    print(
        f"Pose ranking: {result.pose_ranking_method}; "
        f"robustness classifier: {result.robustness_method}"
    )
    print(f"Quasistatic friction policy: {result.friction_policy}")
    if result.robustness_method == "csa":
        print(
            f"CWSA cutoff: {result.minimum_csa_score:g} "
            f"over a {result.csa_cap_half_angle_deg:g} deg cap "
            f"({result.csa_direction_samples} equal-area directions)"
        )
        if result.csa_rocking_fallback_pose_ids:
            print(
                "Rocking fallback for continuous rolling catalog poses: "
                + ", ".join(
                    str(value) for value in result.csa_rocking_fallback_pose_ids
                )
            )
    print(
        f"Directed transitions: {actuated_count} actuated, "
        f"{passive_count} passive"
    )
    print(
        "Main-face family: "
        + "/".join(str(value) for value in result.main_face_ids)
        + f"; intrinsic minimum span: {result.main_face_min_span_mm:.3f} mm"
    )
    print(
        "Opposite X directions: "
        + (
            "available"
            if {
                "floor_main_pos_x",
                "wall_main_neg_x",
            }.intersection(edge.actuation for edge in result.edges)
            else "not available for this roadmap"
        )
        + f" (support threshold > {result.opposite_x_min_height_mm:.3f} mm)"
    )
    if result.unresolved_metastable_node_ids:
        print(
            "Unresolved metastable nodes: "
            + ", ".join(str(value) for value in result.unresolved_metastable_node_ids)
        )
    print(f"Geometry status: {result.geometry_status}")
    for path in paths:
        print(f"  {path}")
    return 0


def _route(args: argparse.Namespace) -> int:
    roadmap = load_roadmap_json(args.roadmap)
    route = find_best_route(
        roadmap,
        args.start_pose,
        args.target_pose,
        max_actions=args.max_actions,
    )
    payload = route.to_dict()
    output = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output, encoding="utf-8")
        print(destination)
    else:
        print(output, end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect(args)
        if args.command == "catalog":
            return _catalog(args)
        if args.command == "render":
            return _render(args)
        if args.command == "stability":
            return _stability(args)
        if args.command == "symmetry":
            return _symmetry(args)
        if args.command == "disturbance":
            return _disturbance(args)
        if args.command == "roadmap":
            return _roadmap(args)
        if args.command == "route":
            return _route(args)
    except (GeometryValidationError, StepSupportUnavailable, ValueError) as exc:
        parser.error(str(exc))
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
