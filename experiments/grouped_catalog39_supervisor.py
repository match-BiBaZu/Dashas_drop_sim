"""Persistent Windows study supervisor: resume, audit and publish the final report."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .grouped_catalog39 import ROOT, OUTPUT, SPEC, POSE_REPO, ROADMAP_REPO, git
from dashas_drop_sim.grouped_export import inventory, read, verify_inventory
from dashas_drop_sim.runner import _write_json


def heartbeat(status, **extra):
    study = read(OUTPUT / "study.json") if (OUTPUT / "study.json").is_file() else {"workpieces": {}}
    complete = {p: r for p, r in study["workpieces"].items() if r["status"] == "complete"}
    _write_json(OUTPUT / "supervisor.json", {"status": status, "supervisor_pid": os.getpid(),
                "updated_utc": datetime.now(timezone.utc).isoformat(), "completed_workpieces": len(complete),
                "completed_published_drops": 10000 * len(complete), "total_workpieces": 39, **extra})


def wait_existing(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return 1
    try:
        while True:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            if code.value != 259:
                return code.value
            heartbeat("running", attached_coordinator_pid=pid)
            time.sleep(30)
    finally:
        kernel.CloseHandle(handle)


def run_stage(arguments, logfile):
    with logfile.open("ab") as log:
        process = subprocess.Popen([sys.executable, "-m", "experiments.grouped_catalog39", *arguments],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"},
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        while process.poll() is None:
            heartbeat("running", coordinator_pid=process.pid, stage_arguments=arguments)
            time.sleep(30)
        return process.returncode


def finalize():
    study = read(OUTPUT / "study.json")
    if len(study["workpieces"]) != 39 or any(r["status"] != "complete" for r in study["workpieces"].values()):
        raise ValueError("Cannot finalize an incomplete 39-workpiece study.")
    for repo in (POSE_REPO, ROADMAP_REPO):
        for part in study["workpieces"]:
            verify_inventory(repo / part)
        if git(repo, "branch", "--show-current") != "grouped":
            raise ValueError("Output checkout left the grouped branch.")
        git(repo, "add", "--all")
        if git(repo, "diff", "--cached", "--name-only"):
            git(repo, "commit", "-m", "Audit completed grouped study: 390000 drops across 39 workpieces")
        git(repo, "push", "origin", "HEAD:refs/heads/grouped")
        if git(repo, "rev-parse", "HEAD") != git(repo, "ls-remote", "origin", "refs/heads/grouped").split()[0]:
            raise ValueError("Remote output branch verification failed.")
    validation = {"completed_workpieces": 39, "completed_drop_trials": 390000,
                  "fresh_additional_drops": 360000, "reused_drop_trials": 30000,
                  "disturbance_trials": sum(r["audit"]["disturbance_tests"] for r in study["workpieces"].values()),
                  "engine_fingerprint": study["provenance"]["engine_fingerprint"],
                  "pose_branch_commit": git(POSE_REPO, "rev-parse", "HEAD"),
                  "roadmap_branch_commit": git(ROADMAP_REPO, "rev-parse", "HEAD"),
                  "workpieces": {p: r["audit"] for p, r in study["workpieces"].items()}}
    _write_json(SPEC / "report/validation.json", validation)
    inventory(SPEC / "report")
    if git(ROOT, "branch", "--show-current") != "feature":
        raise ValueError("Simulator checkout left feature.")
    git(ROOT, "add", "--", "experiments/grouped_catalog39/report")
    if git(ROOT, "diff", "--cached", "--name-only"):
        git(ROOT, "commit", "-m", "Record audited grouped results for all 39 workpieces")
    git(ROOT, "push", "origin", "HEAD:refs/heads/feature")
    if git(ROOT, "rev-parse", "HEAD") != git(ROOT, "ls-remote", "origin", "refs/heads/feature").split()[0]:
        raise ValueError("Remote simulator feature verification failed.")
    heartbeat("complete", **{k: v for k, v in validation.items() if k != "workpieces"},
              simulator_commit=git(ROOT, "rev-parse", "HEAD"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach-pid", type=int)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.attach_pid:
        code = wait_existing(args.attach_pid)
    else:
        code = 1
    for attempt in range(1, 4):
        if code != 0:
            heartbeat("resuming", attempt=attempt)
            code = run_stage(["--publish"], OUTPUT / "supervised_production.log")
        if code == 0:
            code = run_stage(["--audit-only"], OUTPUT / "supervised_audit.log")
        if code == 0:
            try:
                finalize()
                return
            except Exception as error:
                heartbeat("publication_retry", attempt=attempt, error=str(error))
                code = 1
        time.sleep(5)
    heartbeat("failed", exit_code=code,
              error="Three supervisor attempts failed; checkpoints are preserved. Inspect supervised logs and resume the same production command.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
