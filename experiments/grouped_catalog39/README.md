# Grouped catalog study: 39 workpieces

The simulator `feature` branch drives the `grouped` branches of the observed
pose and roadmap repositories. Each workpiece has 10,000 completed drops, plus
73 disturbance tests for every member pose. Qk1a, Dl1a and Dk1i reuse the original
audited grouped runs. Remaining workpieces receive fresh drops.

The frozen configurations apply identical settings and a 0.5 mm equivalent-lift
barrier with a 60-degree group diameter guard. The 1% frequency cutoff follows
group aggregation and uses all completed drops as its denominator. No member
stability exclusion cutoff is applied.

Expected workspace siblings are `BiBaZu_StableComponentenPoses_grouped` and
`BiBaZu_StableComponentenRoadmaps_grouped`, both checked out on `grouped`.
Use `--pose-repo` and `--roadmap-repo` to supply alternative clean checkouts.

```powershell
.venv/Scripts/python.exe -m experiments.grouped_catalog39 --prepare-only
.venv/Scripts/python.exe -m pytest -q --basetemp .pytest_tmp_grouped39
.venv/Scripts/python.exe -m experiments.grouped_pose_study_smoke
.venv/Scripts/python.exe -m experiments.grouped_catalog39 --publish
.venv/Scripts/python.exe -m experiments.grouped_catalog39 --audit-only
```

The production command resumes drop and disturbance checkpoints in
`results/grouped_catalog39/study.json`. Completed workpieces skip physics;
failed Git pushes can be retried independently. Configurations and the engine
fingerprint must remain unchanged. `--parts` supports a selected subset.

`baseline/` stores compressed, byte-exact published pose catalogs and source
registries, hashes and published roadmap numbering. `configs/` stores portable
per-workpiece settings. Full drop and disturbance records remain ignored under
`results/`. [The compact report](report/README.md) records completion,
memberships, representatives, stability and roadmap counts. Each output branch
also contains `GROUPED_STUDY.md` and a machine-readable `grouped_study.json`.

Completed new runs store bulky trial/roughness records as lossless `.gz` files.
`archive.json` records and verifies the original payload hashes. The audit reads
both archived and uncompressed records. Active drop and disturbance checkpoints
stay uncompressed so interruptions can resume directly.

The roadmap merge was resolved using complete published component sets,
retaining both parent histories. Its backup is outside Git in the workspace's
`.task_artifacts/roadmap_merge_backup/` directory. The repaired main commit is
`3523350d8e12a1b7e57467044b9cfe25ff49b0bd`; the comparison baseline remains
`0b3b5c5fe42e787bd47f381b1f1ac57359d5aa55`.
