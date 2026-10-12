# Grouped observed-pose study

Run the frozen Qk1a, Dl1a and Dk1i experiment from the simulator repository:

```powershell
.venv/Scripts/python.exe -m experiments.grouped_pose_study --prepare-only
.venv/Scripts/python.exe -m experiments.grouped_pose_study_smoke
.venv/Scripts/python.exe -m experiments.grouped_pose_study --trials 10000 --workers 8
.venv/Scripts/python.exe -m experiments.grouped_pose_validate
```

The baseline archives are exact `source/poses.json` snapshots from published
StableComponentenRoadmaps commit `0b3b5c5fe42e787bd47f381b1f1ac57359d5aa55`.
Their SHA-256 values and published display-ID mapping are in `baseline/provenance.json`.
Every bundled STL is checked against its snapshot before execution.

Production settings are in `configs/`. Runs write to ignored
`results/grouped_pose_study/`; completed drops and per-member disturbance tests
are checkpointed. Running the same command resumes that study and skips completed
workpieces. A settings or simulation-code change requires a new `--output` directory.
The study does not export, commit or push into the existing pose or roadmap repositories.

The existing observed roadmap builder is used read-only from sibling
`BiBaZu_StableComponentenRoadmaps`; use `--roadmap-engine PATH` for another checkout.
Its outputs are written inside each local run directory. Connections are generated
only for selected representative orientations and remain untested action candidates.
Full member references, raw drop records and disturbance records remain local;
the committed `report/` contains compact results and representative roadmaps.
It also contains a representative-only catalog, a companion roadmap membership
table, the original member frequencies and retention curves, explicit Dk1i pair
barrier evidence, and a validation receipt with full-record checksums.

Grouping compares directed sampled seated-rotation barriers in equivalent lift
millimetres, including finite or continuous CAD symmetry. Mutually reachable states
form basins; transient components enter a group only if they reach a unique sink.
Every final group must satisfy the whole-group angular guard. Oversized or competing
basins are retained separately and listed for review.

The representative maximizes measured group-retention AUC over all configured
disturbance energies and directions. Ties use retention at the grouping threshold,
then observed count, then stable source ID. All members must have complete assessments.
A trajectory which settles outside the group is a loss even if it later returns.
Individual-pose retention is exported alongside group retention. The frequency cutoff
is applied to combined group counts; stability ranks rather than excludes groups.
