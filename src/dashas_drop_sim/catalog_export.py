"""Portable observed-pose catalogue, transactional replacement and Git publication."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .catalog import catalog_models
from .pose_matching import PoseResolver
from .runner import SETTLED_STATUSES, _write_csv, _write_json, wilson_interval


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=120,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Git failed")
    return result.stdout.rstrip("\r\n")


def default_cad_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "bibazu_geometry_to_pose" / "Werkstücke_STL_grob"


def model_pair(mesh: Path, cad_dir: Path | None = None) -> tuple[Path, Path]:
    """Require the original, byte-identical STL beside the matching source CAD."""
    root = Path(cad_dir or default_cad_dir()).resolve()
    files = {p.name.casefold(): p for p in root.iterdir() if p.is_file()}
    stem = mesh.stem
    stl = files.get(f"{stem}.stl".casefold())
    step = files.get(f"{stem}.step".casefold()) or files.get(f"{stem}.stp".casefold())
    if stl is None or step is None:
        raise ValueError(f"STL-/STEP-Paar für {stem} fehlt in {root}")
    expected = stl if mesh.suffix.lower() == ".stl" else step
    if sha256(mesh) != sha256(expected):
        raise ValueError(f"Die ausgewählte Geometrie für {stem} stimmt nicht mit dem CAD-Paar überein.")
    return stl, step


def orientation_distance(a, b, descriptor: dict) -> float:
    left, right = Rotation.from_quat(a), Rotation.from_quat(b)
    axis = descriptor["continuous_axis_part"]
    if axis is not None:
        return float(np.degrees(np.arccos(np.clip(left.apply(axis) @ right.apply(axis), -1, 1))))
    variants = (left * Rotation.from_quat(descriptor["symmetry_quaternions_xyzw"])).as_quat()
    return float(np.degrees(2 * np.arccos(np.clip(np.max(np.abs(variants @ right.as_quat())), 0, 1))))


def recognize_pose(quaternion_xyzw, catalogue: dict) -> dict:
    """Resolve an orientation without the original simulation or geometry repo."""
    descriptor = catalogue["recognition"]
    candidates = sorted((min(orientation_distance(quaternion_xyzw, q, descriptor)
                             for q in pose["reference_quaternions_xyzw"]), pose["id"])
                        for pose in catalogue["poses"])
    if not candidates or candidates[0][0] > descriptor["tolerance_deg"]:
        return {"status": "unmatched", "pose_id": None}
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= descriptor["ambiguity_margin_deg"]:
        return {"status": "ambiguous", "pose_id": None, "candidates": [p for d, p in candidates
                if d <= candidates[0][0] + descriptor["ambiguity_margin_deg"]]}
    return {"status": "matched", "pose_id": candidates[0][1], "distance_deg": candidates[0][0]}


def assign_ids(groups: list[list[dict]], registry: dict, descriptor: dict, name: str) -> list[dict]:
    """Keep fixed historical anchors; never recycle IDs or force ambiguous matches."""
    entries = registry.setdefault("entries", [])
    next_id = int(registry.get("next_id", 1))
    choices = []
    for group in groups:
        quat = group[0]["final_quat_xyzw"]
        distances = sorted((orientation_distance(quat, entry["anchor_quaternion_xyzw"], descriptor),
                            entry["id"]) for entry in entries)
        nearby = [(d, p) for d, p in distances if d <= descriptor["tolerance_deg"]]
        ambiguous = len(nearby) > 1 and nearby[1][0] - nearby[0][0] <= descriptor["ambiguity_margin_deg"]
        choices.append((None if not nearby or ambiguous else nearby[0][1], [p for _, p in nearby]))
    claims = Counter(choice for choice, _ in choices if choice is not None)
    output = []
    for group, (choice, candidates) in zip(groups, choices):
        reused = choice is not None and claims[choice] == 1
        if not reused:
            choice = f"{name}_{next_id:04d}"
            next_id += 1
            entries.append({"id": choice, "anchor_quaternion_xyzw": group[0]["final_quat_xyzw"]})
        output.append({"id": choice, "id_assignment": "reused" if reused else (
            "ambiguous_new" if candidates else "new"), "previous_candidates": candidates})
    registry["next_id"] = next_id
    return output


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _index(repo: Path) -> None:
    names = set(catalog_models())
    names.update(p.parent.name for p in repo.glob("*/poses.json"))
    records = []
    lines = ["# BiBaZu – beobachtete stabile Bauteilposen", "",
             "Simulationsergebnisse unter den je Werkstück dokumentierten Parametern. "
             "Häufigkeiten beziehen sich auf alle abgeschlossenen Fallversuche. "
             "Die reale Stabilität ist damit noch nicht experimentell bestätigt.", "",
             "| Werkstück | Status | Versuche | Eingependelt | Posen | Strecke |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name in sorted(names, key=str.casefold):
        path = repo / name / "poses.json"
        row = {"workpiece": name, "status": "pending"}
        if path.is_file():
            item = _read(path)
            row.update(status="complete", trials=item["completed_trials"],
                       settled=item["settled_trials"], poses=len(item["poses"]),
                       length_mm=item["observation_length_mm"], run_id=item["run_id"])
            lines.append(f"| [{name}]({name}/README.md) | Vollständig | {row['trials']} | "
                         f"{row['settled']} | {row['poses']} | {row['length_mm']:g} mm |")
        else:
            lines.append(f"| {name} | Ausstehend | – | – | – | – |")
        records.append(row)
    _write_json(repo / "index.json", {"schema_version": 1, "workpieces": records})
    (repo / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def publish(repo: Path) -> dict:
    try:
        git(repo, "push", "origin", "HEAD")
        return {"push_status": "published"}
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {"push_status": "pending", "push_error": str(exc)}


def _verify_files(folder: Path) -> None:
    expected = _read(folder / "files.sha256.json")
    actual_names = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()
                    and p.name != "files.sha256.json"}
    if actual_names != set(expected) or any(sha256(folder / p) != digest for p, digest in expected.items()):
        raise ValueError(f"Katalogdateien wurden nach dem Export verändert: {folder}")


@contextmanager
def _repository_lock(repo: Path):
    """An OS lock serializes GUI/CLI exporters and is released after crashes."""
    with (repo / ".git" / "catalog-export.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + 30
        while True:
            try:
                handle.seek(0)
                if sys.platform == "win32":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Ein anderer Katalogexport läuft; diesen Export später wiederholen.")
                time.sleep(0.1)
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def export_run(run_dir: Path, repo: Path, *, cad_dir: Path | None = None, push: bool = True) -> dict:
    with _repository_lock(Path(repo).resolve()):
        return _export_run(run_dir, repo, cad_dir=cad_dir, push=push)


def _export_run(run_dir: Path, repo: Path, *, cad_dir: Path | None = None, push: bool = True) -> dict:
    run_dir, repo = Path(run_dir).resolve(), Path(repo).resolve()
    if Path(git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError("Bitte den Stammordner des Ergebnis-Repos wählen.")
    config, summary, manifest = (_read(run_dir / name) for name in ("config.json", "summary.json", "manifest.json"))
    if summary["cancelled"] or summary["completed_trials"] != summary["requested_trials"]:
        raise ValueError("Nur vollständig abgeschlossene Fallserien werden in den Katalog übernommen.")
    rows = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != summary["completed_trials"] or len({r["trial"] for r in rows}) != len(rows):
        raise ValueError("Die Einzelversuche sind unvollständig oder doppelt.")
    mesh = Path(config["mesh_path"])
    name = mesh.stem
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Werkstücknamen dürfen nur Buchstaben, Ziffern, _ und - enthalten.")
    stl, step = model_pair(mesh, cad_dir)
    if sha256(mesh) != manifest["mesh_source_sha256"]:
        raise ValueError("Die Eingabegeometrie wurde seit dem Fallversuch verändert.")
    run_id = hashlib.sha256("".join(sha256(run_dir / f) for f in (
        "config.json", "manifest.json", "trials.jsonl", "summary.json")).encode()).hexdigest()
    target = repo / name
    same = (target / "poses.json").is_file() and _read(target / "poses.json").get("run_id") == run_id
    dirty = git(repo, "status", "--porcelain", "--untracked-files=all")
    if same:
        _verify_files(target)
    if dirty:
        allowed = {"README.md", "index.json"}
        changed = [line[3:].replace('"', '') for line in dirty.splitlines()]
        if not same or any(p not in allowed and not p.startswith(name + "/") for p in changed):
            raise ValueError("Das Ergebnis-Repo enthält ungesicherte Änderungen. Export bleibt lokal erhalten.")
    if not same:
        if target.exists() and not (target / "pose_registry.json").is_file():
            raise ValueError(f"Der vorhandene Ordner {name} gehört nicht zum Posenkatalog.")
        _build_export(run_dir, repo, target, stl, step, rows, config, summary, manifest, run_id)
    _index(repo)
    git(repo, "add", "--", name, "README.md", "index.json")
    if git(repo, "diff", "--cached", "--name-only"):
        git(repo, "commit", "-m", f"Update {name}: {len(rows)} drop trials ({run_id[:12]})")
    result = {"workpiece": name, "run_dir": str(run_dir), "catalog_repo": str(repo),
              "run_id": run_id, "commit": git(repo, "rev-parse", "HEAD"), "push_status": "local"}
    if push:
        result.update(publish(repo))
    _write_json(run_dir / "catalog_export.json", result)
    return result


def _build_export(run_dir, repo, target, stl, step, rows, config, summary, manifest, run_id):
    name = target.name
    registry = (_read(target / "pose_registry.json") if target.exists()
                else {"schema_version": 1, "mesh_sha256": sha256(stl), "entries": [], "next_id": 1})
    if registry["mesh_sha256"] != sha256(stl):
        raise ValueError("Geändertes STL benötigt einen eigenen Werkstücknamen, damit Pose-IDs eindeutig bleiben.")
    resolver = PoseResolver(stl, None, cache_dir=run_dir.parent / ".pose_cache")
    descriptor = manifest.get("recognition", resolver.recognition_descriptor())
    if not descriptor["symmetry_available"]:
        raise ValueError("Der Katalogexport benötigt die installierte Roadmap-/Symmetrie-Abhängigkeit.")
    # Reproduce the recorded symmetry, even if the local dependency was updated.
    resolver._symmetry_quats = np.asarray(descriptor["symmetry_quaternions_xyzw"])
    axis = descriptor["continuous_axis_part"]
    resolver._continuous_axis = None if axis is None else np.asarray(axis)
    resolver.cluster_tolerance_deg = descriptor["tolerance_deg"]
    settled = sorted((r for r in rows if r["status"] in SETTLED_STATUSES), key=lambda r: r["trial"])
    labels = resolver.cluster_unmatched([r["final_quat_xyzw"] for r in settled])
    groups = [[row for row, label in zip(settled, labels) if label == index] for index in sorted(set(labels))]
    identities = assign_ids(groups, registry, descriptor, name)
    scratch = (repo / ".git" / "catalog-staging" / uuid4().hex).resolve()
    if not scratch.is_relative_to((repo / ".git" / "catalog-staging").resolve()):
        raise ValueError("Invalid catalogue staging directory")
    staged = scratch / name
    staged.mkdir(parents=True)
    (staged / "images").mkdir()
    shutil.copyfile(stl, staged / f"{name}.stl")
    shutil.copyfile(step, staged / f"{name}.step")
    poses = []
    trial_ids = {}
    from .pose_image import render_pose_image
    for group, identity in zip(groups, identities):
        first = group[0]
        count = len(group)
        low, high = wilson_interval(count, len(rows))
        pose_id = identity["id"]
        refs = sorted({tuple(r["final_quat_xyzw"]) for r in group})
        roadmap_ids = sorted({r["roadmap_pose_id"] for r in group if r.get("roadmap_pose_id") is not None})
        quaternion = first["final_quat_xyzw"]
        position = first["final_pos_chute_mm"]
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        transform[:3, 3] = np.asarray(position) - transform[:3, :3] @ manifest["center_mass_source_mm"]
        item = {**identity, "count": count, "frequency_percent": 100 * count / len(rows),
                "frequency_settled_percent": 100 * count / len(settled),
                "ci95_percent": [100 * low, 100 * high], "quaternion_xyzw": quaternion,
                "reference_quaternions_xyzw": [list(q) for q in refs],
                "position_com_chute_mm": position, "transform_chute_from_source_mm": transform.tolist(),
                "representative_trial": first["trial"], "trial_ids": [r["trial"] for r in group],
                "roadmap_pose_ids": roadmap_ids,
                "roadmap_match_statuses": sorted({r.get("match_status", "unmatched") for r in group}),
                "image": f"images/{pose_id}.png"}
        image = render_pose_image(stl, manifest["center_mass_source_mm"], quaternion, position)
        if not image.save(str(staged / item["image"]), "PNG"):
            raise OSError("Posenbild konnte nicht gespeichert werden.")
        poses.append(item)
        trial_ids.update({r["trial"]: pose_id for r in group})
    poses.sort(key=lambda p: (-p["count"], p["id"]))
    catalogue = {"schema_version": 1, "workpiece": name, "run_id": run_id,
                 "completed_trials": len(rows), "settled_trials": len(settled),
                 "status_counts": dict(Counter(r["status"] for r in rows)),
                 "observation_length_mm": config["length_mm"],
                 "frequency_denominator": "all completed drop trials, including unsettled outcomes",
                 "frame": {"handedness": "right", "X": "along belt", "Y": "away from wall",
                           "Z": "away from belt", "length_unit": "mm"},
                 "geometry": {"stl": f"{name}.stl", "step": f"{name}.step",
                              "stl_sha256": sha256(stl), "step_sha256": sha256(step),
                              "center_mass_source_mm": manifest["center_mass_source_mm"]},
                 "recognition": descriptor, "poses": poses}
    _write_json(staged / "poses.json", catalogue)
    (staged / "poses.yaml").write_text(yaml.safe_dump(catalogue, allow_unicode=True, sort_keys=False), encoding="utf-8")
    _write_json(staged / "pose_registry.json", registry)
    portable = {k: v for k, v in config.items() if k not in {"catalog_repo", "catalog_cad_dir", "catalog_push"}}
    portable.update(mesh_path=f"{name}.stl", output_dir="./local_results", roadmap_path=None)
    if config.get("roadmap_path"):
        roadmap = Path(config["roadmap_path"])
        if sha256(roadmap) != manifest["roadmap_sha256"]:
            raise ValueError("Roadmap wurde seit der Simulation verändert.")
        shutil.copyfile(roadmap, staged / ("roadmap" + roadmap.suffix))
        portable["roadmap_path"] = "roadmap" + roadmap.suffix
    _write_json(staged / "config.json", portable)
    portable_manifest = {k: v for k, v in manifest.items() if k != "catalog_mesh_path"}
    _write_json(staged / "manifest.json", portable_manifest)
    fields = ["trial", "seed", "status", "pose_id", "match_status", "roadmap_pose_id",
              "initial_quat_xyzw", "final_quat_xyzw", "final_pos_chute_mm", "sim_time_s",
              "actual_drop_height_mm", "actual_lateral_mm", "initial_pos_chute_mm", "roughness_impulse_count"]
    _write_csv(staged / "trials.csv", fields, [{**r, "pose_id": trial_ids.get(r["trial"])} for r in rows])
    _write_csv(staged / "frequencies.csv", ["id", "count", "frequency_percent", "frequency_settled_percent", "ci95_percent"], poses)
    lines = [f"# {name}", "", f"{len(rows)} Fallversuche auf {config['length_mm']:g} mm Strecke; "
             f"{len(settled)} eingependelte Endlagen. Prozentangaben beziehen sich auf alle Fallversuche.", "",
             "Dies sind beobachtete Simulationslagen unter den dokumentierten Parametern; "
             "Reibung und Unebenheitsmodell sind noch nicht an realen Häufigkeiten kalibriert.", "",
             "| Pose | Bild | Anzahl | Anteil | 95-%-Intervall | Anteil eingependelt |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for p in poses:
        lines.append(f"| {p['id']} | ![{p['id']}]({p['image']}) | {p['count']} | "
                     f"{p['frequency_percent']:.2f} % | {p['ci95_percent'][0]:.2f}–{p['ci95_percent'][1]:.2f} % | "
                     f"{p['frequency_settled_percent']:.2f} % |")
    lines += ["", "Ergebnisstatus: " + ", ".join(f"{k}: {v}" for k, v in catalogue["status_counts"].items()), "",
              "Details und Erkennung: [JSON](poses.json), [YAML](poses.yaml). Quaternionen: xyzw, "
              "Bauteil → Rutsche. Symmetrieoperationen werden rechts an die Bauteilrotation multipliziert. "
              "Die kontinuierliche Symmetrieachse wird gerichtet verglichen. "
              "Orientierungsabstand ≤5°; bei konkurrierenden Treffern innerhalb 1° bleibt die Zuordnung mehrdeutig.", "",
              "Die Pose-IDs bleiben über Läufe erhalten. Der Registereintrag einer früher beobachteten, "
              "aktuell nicht aufgetretenen Pose bleibt zur Wiedererkennung reserviert."]
    (staged / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(staged / "files.sha256.json", {p.relative_to(staged).as_posix(): sha256(p)
                                              for p in sorted(staged.rglob("*")) if p.is_file()})
    _verify_files(staged)
    if _read(staged / "poses.json") != yaml.safe_load((staged / "poses.yaml").read_text(encoding="utf-8")):
        raise ValueError("JSON-/YAML-Katalog stimmt nicht überein.")
    backup = scratch / "previous"
    if target.exists():
        target.rename(backup)
    try:
        staged.rename(target)
    except BaseException:
        if backup.exists():
            backup.rename(target)
        raise
    # Only our verified, unique staging directory is removed.
    shutil.rmtree(scratch)
