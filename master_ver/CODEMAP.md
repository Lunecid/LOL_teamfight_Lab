# Code map (2026-10-04)

This workspace is the current research lineage (since 2026-10-02). The older
lineage lives in `C:\Users\todtj\PycharmProjects\LOL_teamfight` and is
superseded (`docs/LINEAGE_SUPERSEDED_20261004.md` there).

Single entry point for the whole pipeline (stages, governing sources with
precedence, q specification, open decisions): `reports/architecture_20261005/ARCHITECTURE_KO.md`
with source hashes in `reports/architecture_20261005/SPEC_INDEX.json`.

Folders are dated: `studies/<topic>_<date>/` holds importable code and
`reports/<topic>_<date>/` holds contracts, decisions, reports and some entry
points. A dated folder is a frozen record once a later folder supersedes it.
Hash-bound records (RELEASE, ACCEPTANCE, MANIFEST, CONTRACT files) reference
these exact paths, so files are not moved or renamed.

## Live production path (V search/refit/OOF/calibration)

| Role | Path |
|---|---|
| Supervisor and workers | `reports/v_production_20261004/pipeline.py` |
| Cache builder (input caches) | `reports/v_production_20261004/prepare.py` |
| Candidate runner, gates, selection, calibration | `studies/v_production_20261004/train.py` |
| Split source over caches | `studies/v_production_20261004/source.py` |
| Cache format reader/writer | `studies/v_production_20261004/cache.py` |
| Model families and cards | `studies/v_model_fit_input_20261003/models.py` |
| Fitted preprocessing and table layout | `studies/v_model_fit_input_20261003/{common,preprocess}.py` |
| Acceptance guard | `studies/v_training_gate_20261003/guard.py` |
| Preprocessing bundles and bank surface | `studies/v_full_preprocess_20261003/{artifacts,surface}.py` |
| Partition and OOF membership | `studies/v_preparation_20261002/partition.py` |
| Release issuer, cache audit | `reports/v_production_20261004/{make_release,verify_caches}.py` |
| oof cache launcher, RAM guard | `reports/dual_worker_schedule_20261004/{cache_prep_launcher,ram_guard}.py` |
| Operations guide, amendments | `reports/v_production_20261004/{OPERATIONS_KO,AMENDMENT_20261004_KO}.md` |

Governing contracts: `reports/research_execution_one_seed_20261003/CONTRACT.json`
(one-seed phase), `reports/research_execution_20261003/{CONTRACT.json,PROTOCOL_KO.md}`,
`reports/v_preparation_20261002/CONTRACT.json`, `reports/post_state_label_contract_20261002/CONTRACT.json`,
`reports/population_adoption_20261002/CONTRACT.json`, `reports/current_design_gate_20261004/DECISION.json`.

Live data inputs (do not move): `outputs/v_full_preprocess_20261003/full_1`,
`outputs/shared_history_eight_cpu_20261003/full_1`, `outputs/v_input_projection_20261002/full_raw_20261002_1`,
`outputs/v_model_fit_input_20261003/scope_census_1`, `outputs/item_node_observed_20261002/full_bank_v3_20261002_1`.
Needed by the label stage: `outputs/observable_state_20261002/full_static_preserved_20261002_1` and the
population_adoption locked run.

Run outputs: `D:\LOL_Training_20261004\run_3` (see OPERATIONS_KO.md).

## Order of the remaining work

1. V: six V_LOCK files (final, oof0-oof4) from the live path above.
2. Labels: same-version SVI validation and lock (`reports/label_lock_gate_20261004/STATUS.json`).
3. q: comparison training after the label lock.

## Historical (read-only records)

`scripts/` (dated Codex-line scripts up to 2026-09-16), `studies/*` and `reports/*`
folders not listed above, `worktrees/engagement-state-value` (v4-exact lineage,
a worktree of the old repository), `deliverables/`, `docs/`.
