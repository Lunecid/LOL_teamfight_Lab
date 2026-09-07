# Fight boundary specs

Versioned output of `scripts/run_fight_boundary_pipeline.py` (patch-sliced,
3,000 matches per patch, seed 7, 200 bootstrap replicates, 2026-09-08).

| file | what |
|---|---|
| `spec_pooled.json` | the definition in force for the 15.14-15.16 corpus: G, D, R, lead, min per team, scale cuts, with CIs and provenance |
| `spec_<patch>.json` | the same estimated inside one patch |
| `drift.md` / `drift.json` | per-patch vs pooled comparison and the verdict (`pooled` = one definition serves every patch) |

`BoundarySpec.from_json(...).detector_overrides()` yields the detector config
keys (`TF2_KILL_CLUSTER_GAP_MS`, `CLUSTER_MAX_DIAMETER`, `TF2_VALIDITY_RADIUS`,
`TF2_ENGAGE_PRE_KILL_MS`, `TF2_MIN_PER_TEAM`); `scale_class(blue, red)` applies
pick <= 1 / skirmish <= 3 / teamfight >= 4 on the smaller side's participation.
Re-run the pipeline when a new patch enters the cache; a `per_patch` verdict
means the new patch needs its own spec.  Method and evidence:
`docs/DEFINITION_EVIDENCE.md`.
