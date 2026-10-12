"""Verify identical seeded physical drops with one and eight worker processes."""
from dataclasses import replace
import hashlib
import json
from .grouped_pose_study import ROOT, SPEC, prepare
from dashas_drop_sim.cli import _config
from dashas_drop_sim.runner import run_experiment, _write_json


def main():
    prepare()
    digests = []
    runs = []
    for workers in (1, 8):
        config = replace(_config(SPEC / "configs/Qk1a.json"), trials=8, workers=workers,
                         grouping_enabled=False, compute_stability=False,
                         output_dir=ROOT / f"results/grouped_pose_smoke/workers_{workers}")
        result = run_experiment(config, emit=lambda event: print(json.dumps({"workers": workers, "event": event["event"], "trial": event.get("trial")}), flush=True))
        digest = hashlib.sha256((result.run_dir / "trials.jsonl").read_bytes()).hexdigest()
        digests.append(digest)
        runs.append({"workers": workers, "trials": result.summary["completed_trials"], "sha256": digest})
    record = {"identical": len(set(digests)) == 1, "runs": runs}
    _write_json(SPEC / "worker_smoke.json", record)
    print(json.dumps(record), flush=True)
    if not record["identical"]:
        raise RuntimeError("Worker counts changed the seeded trial records.")


if __name__ == "__main__":
    main()
