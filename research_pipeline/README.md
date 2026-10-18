# Research pipeline code map

This source bundle extends the `common_training_r1` smoke with the implementation lineage needed to inspect and continue the established research workflow. It preserves the source subdirectory names so imports such as `studies.observable_state_20261002` resolve when `research_pipeline/` is on `PYTHONPATH`.

## Included code

- FIT and holdout raw-source audits, split/support checks, outcome-ledger producers and validators.
- Generic PostgreSQL catalog schema and verification SQL. Database contents, exact namespace, local registration records, and credentials remain local.
- Terrain and champion-knowledge tools; multi-patch and window contracts are external local inputs. Generated contracts, extracted semantic payloads and map assets are external inputs. Run-specific contract audits, cohort comparison fixtures, distribution reports and private measured findings are omitted.
- Existing observed-state, label, and Rift model packages, with their anchor-frame, input-specification, runtime, streaming, and item dependencies. Superseded V-model workflows are not part of this public snapshot.
- The bounded shared-input tree/hierarchical smoke is in the sibling `common_training_r1/` directory.

These stages are versioned components, not a one-command production workflow. Current objective models have not been trained at confirmatory scale; historical model code is included as source lineage, and no prior metrics are claimed here. Local cohort locks, expected counts and raw fixture identifiers must be supplied through the documented local inputs/environment; they are not embedded in this public snapshot. FIT/holdout raw-source readers, audit rules, disposition and ledger producers are included as source code; generated outputs, private rows, IDs, and hashes are not included. A one-off process handoff helper tied to a specific local run was omitted because its hard-coded process identifiers and termination actions are not portable research code. Report renderers, result packagers, local database registration/publishing helpers, integrated-design/multipatch receipt builders, fixed-cohort cache-source and objective-value audit packages, data-bound design audit scripts, superseded V-model workflows with study-specific split salts, and run drivers pinned to private manifests or fixed data fingerprints are omitted where they expose private measured findings, cohort fingerprints, or machine-specific paths.

## Local configuration

FIT and holdout assignment is read from `LOL_COHORT_POLICY_PATH`, a local JSON file that is intentionally not included. It defines a salt, canonical region prefix, hash bucket parameters, and patch-to-role bucket rules. The code contains only the policy schema and generic engine; supply your approved cohort configuration locally before running the audits or producers.

Configure these environment variables only for stages that need them. Never commit credentials or generated data/products.

| Variable | Purpose |
|---|---|
| `LOL_RESEARCH_ROOT` | This repository's `research_pipeline/` directory and study imports |
| `LOL_ARTIFACT_ROOT` | Local versioned contract/ledger workspace |
| `LOL_RAW_MATCH_ROOT` | Raw match and timeline files |
| `LOL_TRAINING_ROOT` | Existing training inputs and outputs |
| `LOL_EXTERNAL_ROOT`, `LOL_LEGACY_CACHE_ROOT` | External banks and legacy caches |
| `LOL_DDRAGON_RAW_DIR`, `LOL_STATIC_MAPS_ROOT` | Patch static data and verified terrain inputs |
| `LOL_PSQL_BIN`, `LOL_PGHOST`, `LOL_PGPORT`, `LOL_PGUSER`, `LOL_PGDATABASE` | PostgreSQL client/connection configuration |
| `PGPASSFILE` | Local libpq password file outside this repository |
| `LOL_LITERATURE_KERNEL` | Optional local literature-review helper |
| `LOL_COHORT_POLICY_PATH` | Local-only cohort hash and patch-role rules used by raw audit and ledger producers |
| `LOL_EXCLUDED_MATCH_KEY_SHA256`, `LOL_EXPECTED_ACTIVE_ROLES_JSON`, `LOL_REVIEW_MATCH_ID`, `LOL_SPLIT_ASSIGNMENTS_PATH` | Local cohort lock, split-manifest path, and raw-fixture selection for audit/replay scripts |

Some scripts also require frozen contracts, split manifests, label locks, and patch assets from local research storage. Check each stage's source/configuration and its applicable protocol before execution. No raw/cache match data, database rows, split or ledger rows, match ID lists, audit receipts, semantic payloads, map binaries, checkpoints, predictions, or run outputs are included.

Tests that inspect one raw local match require LOL_REVIEW_MATCH_ID to be set by the local operator; no real match identifier is embedded in this repository.






