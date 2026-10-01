"""Command line entry point shared by GUI batch runs and manual experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import threading

import numpy as np

from .config import RunConfig
from .geometry import prepare_part
from .physics import ChuteSimulator
from .pose_matching import PoseResolver
from .runner import _new_run_dir, _write_json, build_manifest, run_experiment


def _emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _config(path: Path) -> RunConfig:
    path = path.expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration JSON must contain an object.")
    for key in ("mesh_path", "roadmap_path", "output_dir"):
        if raw.get(key) is not None and not Path(raw[key]).expanduser().is_absolute():
            raw[key] = str((path.parent / Path(raw[key]).expanduser()).resolve())
    return RunConfig.from_dict(raw)


def _cancel_event() -> threading.Event:
    stopped = threading.Event()

    def read_stdin() -> None:
        for line in sys.stdin:
            if line.strip().lower() in {"stop", "cancel", "quit"}:
                stopped.set()
                return

    threading.Thread(target=read_stdin, daemon=True, name="drop-sim-cancel").start()
    try:
        signal.signal(signal.SIGINT, lambda _number, _frame: stopped.set())
    except ValueError:
        pass
    return stopped


def _preview(config: RunConfig, trial_index: int, *, visible: bool, stopped: threading.Event) -> int:
    if trial_index < 0:
        raise ValueError("--trial must be non-negative")
    config.validate()
    run_dir = _new_run_dir(config.output_dir, config.seed)
    _write_json(run_dir / "config.json", config.to_dict())
    part = prepare_part(config.mesh_path, config.output_dir / ".mesh_cache", config.density_g_cm3)
    resolver = PoseResolver(part.catalog_mesh_path, config.roadmap_path, original_mesh_path=config.mesh_path)
    _write_json(run_dir / "manifest.json", build_manifest(config, part, resolver))
    simulator = ChuteSimulator(config, part)
    child = np.random.SeedSequence(config.seed).spawn(trial_index + 1)[-1]
    trial_seed = int(child.generate_state(1, dtype=np.uint32)[0])
    outcome = simulator.drop(trial_index, trial_seed, preview=visible, cancel=stopped.is_set)
    match = resolver.resolve(outcome.final_quat_xyzw) if outcome.status in {"settled", "settled_stationary"} else None
    record = outcome.to_dict()
    record["match_status"] = match.status if match else "not_evaluated"
    record["roadmap_pose_id"] = match.pose_id if match else None
    record["match_distance_deg"] = match.distance_deg if match else None
    _write_json(run_dir / "preview.json", record)
    _emit({"event": "done", "run_dir": str(run_dir),
           "message": f"Einzelfall: {outcome.status}, Roadmap-Pose: {match.pose_id if match and match.pose_id is not None else 'keine'}"})
    return 130 if stopped.is_set() else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BiBaZu MuJoCo fall test simulation")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("gui", help="Start the PyQt6 user interface")
    run_parser = commands.add_parser("run", help="Run a headless batch and save CSV/JSON")
    run_parser.add_argument("--config", type=Path, required=True)
    preview_parser = commands.add_parser("preview", help="Show one seeded trial in a MuJoCo viewer")
    preview_parser.add_argument("--config", type=Path, required=True)
    preview_parser.add_argument("--trial", type=int, default=0)
    preview_parser.add_argument("--headless", action="store_true", help="Run one trial without a viewer")
    args = parser.parse_args(argv)
    try:
        if args.command == "gui":
            from .gui import main as gui_main
            return gui_main()
        config = _config(args.config)
        stopped = _cancel_event()
        if args.command == "preview":
            return _preview(config, args.trial, visible=not args.headless, stopped=stopped)
        result = run_experiment(config, emit=_emit, cancel=stopped.is_set)
        return 130 if result.summary["cancelled"] else 0
    except (OSError, TypeError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
