# Master report code layout

Revision: `master-reporting-layout-20261024`; derived from baseline
`d9d7ac8dbd5c389c8bf70cbd4c0ed1ee8be99e4a` on 2026-10-24.
This is an authorized derived code revision in the isolated grouping worktree.
The external original Master workspace is unchanged and was not consulted.

Canonical implementations are under `scripts/reporting/`. The old files remain
small CLI/import entry points. Paths below are relative to `master_ver/`.

| Compatibility entry point | Canonical implementation |
| --- | --- |
| `scripts/report_b_three_horizons.py` | `scripts/reporting/report_b_three_horizons.py` |
| `scripts/report_causal_position.py` | `scripts/reporting/report_causal_position.py` |
| `scripts/report_event_boundary_cif.py` | `scripts/reporting/report_event_boundary_cif.py` |
| `scripts/report_fixed120_events.py` | `scripts/reporting/report_fixed120_events.py` |
| `scripts/report_position_ablation.py` | `scripts/reporting/report_position_ablation.py` |
| `scripts/report_postkill_full.py` | `scripts/reporting/report_postkill_full.py` |
| `scripts/report_validation_suite.py` | `scripts/reporting/report_validation_suite.py` |
| `scripts/report_window_validation.py` | `scripts/reporting/report_window_validation.py` |

`module-layout.json` is the machine-readable grouping marker for the lead's
sync-overwrite guard. It records this derived layout; this change does not edit
`tools/sync_from_sources.py` or authorize overwriting the layout from the original.

Direct execution uses `runpy.run_path(..., run_name="__main__")`. Arguments after
`argv[0]` and the working directory are preserved. During execution, `runpy` sets
`argv[0]` to the canonical file and restores the caller's `argv[0]` on return.
Imports alias the canonical module in `sys.modules`, so globals and monkeypatches
are shared within each standard import pair (`master_ver.scripts.<name>` and
`master_ver.scripts.reporting.<name>`, `scripts.<name>` and
`scripts.reporting.<name>`, or `<name>` and `reporting.<name>` when `scripts/` is
the import root). No search-path modification or import hook is installed.

All eight canonical `ROOT` expressions use `parents[2]` instead of `parents[1]`,
resolving to the same `master_ver/` directory as before. Algorithm ASTs otherwise
match the fixed baseline, with one intentional metadata exception:
`report_validation_suite.py` now explicitly adds all eight canonical source files
to its provenance inventory. Its previous nonrecursive validation glob, legacy
source references, model/input references, and result handling are retained.
This is a new source-provenance revision; historical records are not rewritten.
No outputs or payloads were read and no report/main was executed for validation.
Historical CLI links and research/result paths remain at their existing locations.

Organization checks use only the standard library, tracked fixed-baseline AST hash fixture,
synthetic reports, and dependency stubs (one CPU thread, CUDA disabled):

```powershell
python -B -m unittest discover -s master_ver/organization_tests -v
```

`organization_tests/baseline_ast_sha256.json` records the eight baseline source
and AST hashes. Baseline AST equality was also checked directly during this
implementation. Distributed checks compare the complete canonical AST after
normalizing only the declared exceptions; negative controls reject algorithm,
ROOT, or inventory mutations. They require neither private Git ancestry nor
`work/` scratch files and run from a fresh public source checkout.

The implementation checks are separate from the independent review performed
later by another agent. These checks do not validate scientific numeric results.
