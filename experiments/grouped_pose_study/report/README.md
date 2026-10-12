# Grouped drop study

10,000 fresh drops per workpiece; disturbance assessments are additional.

Baseline: `0b3b5c5fe42e787bd47f381b1f1ac57359d5aa55`. Grouping: 0.5 mm equivalent lift, 60° maximum diameter.

| Workpiece | Drops | Settled | Assessed members | Groups ≥1% | Disturbance tests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qk1a | 10000 | 9956 | 13 | 6 | 949 |
| Dl1a | 10000 | 9983 | 16 | 5 | 1168 |
| Dk1i | 10000 | 9998 | 8 | 6 | 584 |

## Qk1a

| Group members (published roadmap IDs) | Representative | Frequency | Individual retention AUC | Group retention AUC |
| --- | --- | ---: | ---: | ---: |
| 5, 6, 7, Qk1a_0002 | 5 | 13.70% | 1.0000 | 1.0000 |
| 0, Qk1a_0005 | 0 | 23.17% | 1.0000 | 1.0000 |
| 3 | 3 | 12.80% | 1.0000 | 1.0000 |
| 1 | 1 | 21.87% | 1.0000 | 1.0000 |
| 2 | 2 | 13.72% | 1.0000 | 1.0000 |
| 4, 8, 9, Qk1a_0013 | 4 | 14.30% | 1.0000 | 1.0000 |

Baseline-only memberships: {0}; {1}; {2}; {3}; {4, 8, 9}; {5, 6, 7}.
Across all member disturbance tests, 930 retained the group and 758 retained the exact individual pose.
Additional source IDs identify rare baseline observations below the published individual-pose frequency cutoff. Their counts are included before the group cutoff.

`roadmap_membership.csv` maps new roadmap node IDs to representatives, every source member, and the published numbering. `representatives.json` is the representative-only input catalog.

## Dl1a

| Group members (published roadmap IDs) | Representative | Frequency | Individual retention AUC | Group retention AUC |
| --- | --- | ---: | ---: | ---: |
| 6, Dl1a_0003 | Dl1a_0003 | 1.55% | 1.0000 | 1.0000 |
| 1, 2, 5, Dl1a_0015 | 1 | 46.28% | 1.0000 | 1.0000 |
| 0, 3, Dl1a_0010, Dl1a_0016 | 0 | 47.24% | 1.0000 | 1.0000 |
| 4, Dl1a_0013 | Dl1a_0013 | 3.32% | 1.0000 | 1.0000 |
| Dl1a_0001, Dl1a_0002 | Dl1a_0002 | 1.13% | 0.9844 | 0.9844 |

Baseline-only memberships: {0, 3}; {1, 2, 5}; {4}; {6}.
Across all member disturbance tests, 1081 retained the group and 1036 retained the exact individual pose.
Additional source IDs identify rare baseline observations below the published individual-pose frequency cutoff. Their counts are included before the group cutoff.

`roadmap_membership.csv` maps new roadmap node IDs to representatives, every source member, and the published numbering. `representatives.json` is the representative-only input catalog.

## Dk1i

| Group members (published roadmap IDs) | Representative | Frequency | Individual retention AUC | Group retention AUC |
| --- | --- | ---: | ---: | ---: |
| 6 | 6 | 7.46% | 1.0000 | 1.0000 |
| 1, 5 | 1 | 26.91% | 1.0000 | 1.0000 |
| 2, 4 | 2 | 27.68% | 1.0000 | 1.0000 |
| 3 | 3 | 14.39% | 1.0000 | 1.0000 |
| 0 | 0 | 16.44% | 1.0000 | 1.0000 |
| 7 | 7 | 7.10% | 1.0000 | 1.0000 |

Baseline-only memberships: {0}; {1, 5}; {2, 4}; {3}; {6}; {7}.
Across all member disturbance tests, 584 retained the group and 584 retained the exact individual pose.
Additional source IDs identify rare baseline observations below the published individual-pose frequency cutoff. Their counts are included before the group cutoff.

`roadmap_membership.csv` maps new roadmap node IDs to representatives, every source member, and the published numbering. `representatives.json` is the representative-only input catalog.

| Published candidate pair | Forward barrier (mm) | Reverse barrier (mm) | Angular separation |
| --- | ---: | ---: | ---: |
| {1,5} | 0.3873 | 0.3888 | 30.04° |
| {2,4} | 0.3954 | 0.3873 | 29.94° |

Both pairs satisfy the same 0.5 mm rule used for all workpieces; neither pair is prescribed in the grouping implementation.
No settled switching between individual members was observed in these disturbance tests. The consolidation is supported by the sampled geometric barrier criterion; the measured individual and group retention scores coincide.

AUC measures remaining within the group throughout settled observations across the configured energy levels and directions. These scores use the configured virtual chute and uncalibrated contact assumptions.

Validation: the simulator suite passed 102 tests before production. Updated GUI checks (5 tests) and published geometry regressions (3 cases) also passed. The seeded 8-drop comparison produced identical records with 1 and 8 workers; see [worker_smoke.json](../worker_smoke.json).
The [production audit](validation.json) verifies unique trial IDs and seeds, count conservation, complete member assessments, deterministic representatives, and representative-only roadmap orientations. Export checksums are in `files.sha256.json`.

Full drop and disturbance trajectories remain under ignored `results/grouped_pose_study/`. No existing pose or roadmap catalog is modified.
