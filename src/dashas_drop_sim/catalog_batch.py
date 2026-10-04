"""The same resumable catalogue execution path for GUI, CLI and all 39 models."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from importlib import metadata
import json
from pathlib import Path

from .batch import find_roadmap
from .catalog import catalog_models
from .catalog_export import export_run, git, model_pair, sha256
from .config import RunConfig
from .runner import RunResult, _write_json, run_experiment


def engine_fingerprint() -> str:
    root = Path(__file__).parent
    hashes = {name: sha256(root / (name + ".py")) for name in (
        "config", "physics", "roughness", "belt", "release", "geometry", "pose_matching", "runner", "parallel")}
    hashes["versions"] = {name: metadata.version(name) for name in (
        "mujoco", "numpy", "scipy", "trimesh", "coacd", "bibazu-chute-pose")}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def signature(config: RunConfig) -> str:
    settings = config.to_dict()
    for key in ("output_dir", "catalog_repo", "catalog_cad_dir", "catalog_push", "workers"):
        settings.pop(key)
    settings["mesh_path"] = sha256(config.mesh_path)
    settings["roadmap_path"] = sha256(config.roadmap_path) if config.roadmap_path else None
    return hashlib.sha256(json.dumps({"settings": settings, "engine": engine_fingerprint()}, sort_keys=True).encode()).hexdigest()


def run_with_catalog(config: RunConfig, *, emit=None, cancel=None, visible=False) -> RunResult:
    emit = emit or (lambda event: None)
    cancel = cancel or (lambda: False)
    if config.catalog_repo is None:
        return run_experiment(config, emit=emit, cancel=cancel, visible=visible)
    config.validate()
    repo = config.catalog_repo
    git(repo, "rev-parse", "--show-toplevel")
    model_pair(config.mesh_path, config.catalog_cad_dir)
    key = signature(config)
    state_path = repo / ".git" / "catalog-progress" / f"{config.mesh_path.stem}_{key}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    result = None
    if state_path.is_file() and not visible:
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("run_dir"):
            run_dir = Path(previous["run_dir"])
            if (run_dir / "summary.json").is_file():
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                if not summary["cancelled"] and summary["completed_trials"] == config.trials:
                    result = RunResult(run_dir, summary)
                    emit({"event": "resumed", "run_dir": str(run_dir), "workpiece": config.mesh_path.stem})
    if result is None:
        _write_json(state_path, {"status": "running", "signature": key})
        result = run_experiment(config, emit=emit, cancel=cancel, visible=visible)
    state = {"status": "simulated", "signature": key, "run_dir": str(result.run_dir)}
    _write_json(state_path, state)
    if result.summary["cancelled"]:
        state["status"] = "cancelled"
    else:
        try:
            publication = export_run(result.run_dir, repo, cad_dir=config.catalog_cad_dir, push=config.catalog_push)
            state.update(status="complete", publication=publication)
            emit({"event": "catalog_export", **publication})
        except (OSError, RuntimeError, ValueError) as exc:
            state.update(status="export_pending", export_error=str(exc))
            emit({"event": "catalog_export_error", "message": str(exc), "run_dir": str(result.run_dir)})
    _write_json(state_path, state)
    # Local receipt makes retry/status available even outside the batch GUI.
    _write_json(result.run_dir / "catalog_status.json", state)
    return result


def run_catalog_batch(base: RunConfig, *, emit=None, cancel=None) -> dict:
    emit = emit or (lambda event: None)
    cancel = cancel or (lambda: False)
    base.validate()
    if base.catalog_repo is None:
        raise ValueError("catalog_repo is required for catalog-batch")
    models = catalog_models()
    for mesh in models.values():
        model_pair(mesh, base.catalog_cad_dir)
    engine = engine_fingerprint()
    records = []
    state_file = base.output_dir / "catalog_batch.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    for index, (name, mesh) in enumerate(models.items(), start=1):
        if cancel():
            break
        if engine_fingerprint() != engine:
            raise RuntimeError("Simulationscode wurde während des Batches verändert; vor dem Fortsetzen neu starten.")
        config = replace(base, mesh_path=mesh, roadmap_path=find_roadmap(mesh), output_dir=base.output_dir / name)
        emit({"event": "workpiece", "workpiece": name, "index": index, "total": len(models)})
        try:
            result = run_with_catalog(config, emit=emit, cancel=cancel)
            state = json.loads((result.run_dir / "catalog_status.json").read_text(encoding="utf-8"))
            records.append({"workpiece": name, **state})
        except (OSError, ValueError, RuntimeError) as exc:
            records.append({"workpiece": name, "status": "failed", "error": str(exc)})
            emit({"event": "workpiece_error", "workpiece": name, "message": str(exc)})
        _write_json(state_file, {"engine_fingerprint": engine, "config": base.to_dict(),
                                 "cancelled": bool(cancel()), "workpieces": records})
    return {"cancelled": bool(cancel()), "workpieces": records, "state_file": str(state_file)}
