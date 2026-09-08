# Fight boundary specs

Versioned output of `scripts/run_fight_boundary_pipeline.py` on the full canonical
corpus: every cached match of patches 15.14-15.16 with at least one inter-kill
interval (208,141 matches, 10.4 M intervals, 5.46 M consecutive pairs inside the
temporal window), seed 7, 200 bootstrap replicates, 2026-09-08, 82 min on 16 CPUs.

| file | what |
|---|---|
| `spec_pooled.json` | the definition in force for the 15.14-15.16 corpus: G, D, R, lead, min per team, scale cuts, with CIs and provenance |
| `spec_<patch>.json` | the same estimated inside one patch |
| `drift.md` / `drift.json` | per-patch vs pooled comparison and the verdict (`pooled` = one definition serves every patch) |
| `details_<scope>.json` | everything behind a spec: KDE modes and valley, bandwidth sweep, mixture crossing, bootstrap, ARI curve, sharing curve, region crossovers with CIs, lane anisotropy, region transitions |
| `fight_boundary.png` | per-patch G and D against the pooled plateau and the +-10% band |
| `sample_3000/` | the 3,000-matches-per-patch pilot (same seed) kept for comparison |

Definition in force (`spec_pooled.json`): G = 13.7 s (ARI >= 0.9 plateau 10-18 s),
D = 4,264 u (95% CI 4,256-4,274), R = 1,800 u (89.1% Data Dragon range coverage),
B = 10 s, M = 2, pick <= 1 / skirmish <= 3 / teamfight >= 4 on the smaller side.
Verdict **pooled**: per-patch G = 14.0 / 13.5 / 13.7 s, D = 4,285 / 4,265 / 4,241 u.

Pilot vs full (`python scripts/compare_boundary_specs.py config/fight_boundary/sample_3000
config/fight_boundary --label-a pilot --label-b full`): the pooled values moved by 0.1 s
and 1 u; the per-patch spread shrank from 12.6-13.9 s / 4,146-4,309 u in the pilot to
13.5-14.0 s / 4,241-4,285 u, so the pilot's patch 15.15 deviation was sampling noise.
The D interval narrows from 78 u to 18 u with the corpus; the G interval does not,
because the temporal bootstrap subsamples each replicate to 100,000 intervals
(`cluster_bootstrap(max_pairs=100_000)`) and is therefore a conservative bound.
The full corpus also resolves the along-lane sharing crossover the pilot could not:
5,212 / 5,917 / 5,516 u along top / mid / bot against 3,456 / 3,612 / 3,726 u across,
a 1.5x anisotropy that we report but do not encode (see `docs/DEFINITION_EVIDENCE.md`
sections 11 and 14).

`BoundarySpec.from_json(...).detector_overrides()` yields the detector config
keys (`TF2_KILL_CLUSTER_GAP_MS`, `CLUSTER_MAX_DIAMETER`, `TF2_VALIDITY_RADIUS`,
`TF2_ENGAGE_PRE_KILL_MS`, `TF2_MIN_PER_TEAM`); `scale_class(blue, red)` applies
pick <= 1 / skirmish <= 3 / teamfight >= 4 on the smaller side's participation.
Re-run the pipeline when a new patch enters the cache (`--n-matches-per-patch 0`
for every match); a `per_patch` verdict means the new patch needs its own spec.
Method and evidence: `docs/DEFINITION_EVIDENCE.md` (sections 10 and 14).
