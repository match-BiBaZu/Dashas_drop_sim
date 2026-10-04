"""Compare worker counts using identical physical trials and exact outcome digests."""

from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from .catalog import catalog_models
from .cli import _config
from .geometry import prepare_part
from .parallel import SimulationPool, drop_job
from .runner import _write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workpiece", default="Dk2a")
    parser.add_argument("--trials", type=int, default=96)
    parser.add_argument("--workers", nargs="+", type=int, default=[8, 12, 16, 24])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/worker_benchmark.json"))
    args = parser.parse_args(argv)
    if args.trials < 1 or args.repeats < 1 or not args.workers:
        parser.error("Positive trials/repeats and worker counts are required")
    config = replace(_config(args.config), mesh_path=catalog_models()[args.workpiece],
                     trials=args.trials, catalog_repo=None, compute_stability=False)
    for workers in args.workers:
        replace(config, workers=workers).validate()
    part = prepare_part(config.mesh_path, config.output_dir / args.workpiece / ".mesh_cache", config.density_g_cm3)
    jobs = [(index, int(seed.generate_state(1, dtype=np.uint32)[0]))
            for index, seed in enumerate(np.random.SeedSequence(config.seed).spawn(args.trials))]
    measurements = []
    reference = None
    failed_workers = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        order = args.workers if repeat % 2 == 0 else list(reversed(args.workers))
        for workers in order:
            if workers in failed_workers:
                continue
            config.workers = workers
            started = time.perf_counter()
            try:
                with SimulationPool(config, part) as pool:
                    outcomes = [outcome for _job, outcome in pool.run(jobs, drop_job, lambda: False)]
                elapsed = time.perf_counter() - started
                outcomes.sort(key=lambda row: row.trial)
                digest = hashlib.sha256(json.dumps([row.to_dict() for row in outcomes],
                                                   sort_keys=True).encode()).hexdigest()
                reference = reference or digest
                valid = (len(outcomes) == len(jobs) and all(row.status != "invalid"
                         and np.all(np.isfinite(row.final_qpos)) for row in outcomes))
                row = {"workers": workers, "repeat": repeat, "elapsed_s": elapsed,
                       "trials_per_second": len(outcomes) / elapsed,
                       "identical_to_reference": digest == reference, "valid": bool(valid), "sha256": digest}
            except (RuntimeError, OSError) as exc:
                failed_workers.add(workers)
                row = {"workers": workers, "repeat": repeat, "elapsed_s": time.perf_counter() - started,
                       "valid": False, "identical_to_reference": False, "error": str(exc)}
            measurements.append(row)
            print(json.dumps(row), flush=True)
            _write_json(args.output, {"workpiece": args.workpiece, "trials": args.trials,
                                     "config": config.to_dict(), "measurements": measurements})
    medians = {workers: statistics.median(r["elapsed_s"] for r in measurements if r["workers"] == workers)
               for workers in args.workers if all(r["valid"] and r["identical_to_reference"]
                  for r in measurements if r["workers"] == workers)}
    best = min(medians, key=medians.get) if medians else None
    # Prefer fewer workers if the measured difference is within five percent.
    recommended = min(w for w, seconds in medians.items() if seconds <= medians[best] * 1.05) if best else None
    result = {"workpiece": args.workpiece, "trials": args.trials, "config": config.to_dict(),
              "measurements": measurements, "median_seconds": medians, "recommended_workers": recommended,
              "timing_scope": "physics trials including process startup/shutdown; cached geometry preparation excluded"}
    _write_json(args.output, result)
    print(json.dumps({"median_seconds": medians, "recommended_workers": recommended}), flush=True)
    return 0 if all(row["valid"] and row["identical_to_reference"] for row in measurements) else 2


if __name__ == "__main__":
    raise SystemExit(main())
