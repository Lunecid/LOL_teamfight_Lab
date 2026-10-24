# Code map

[Working guidelines](docs/WORKING_GUIDELINES_KO.md)

[Repository overview](README.md) · [EPT](c_ept/MODULES.md) · [CoG](COG/MODULES.md) · [Master](master_ver/MODULES.md)

The implementation folders below are the starting points for new code. Historical
import names and supported module/script commands retain compatibility entry points. Each moved
implementation has one source file; the entry points do not contain a second copy
of its calculations.

## EPT

| Implementation folder | Responsibility |
|---|---|
| [`c_ept/data/`](c_ept/data/) | Read, cast and store event, frame and head rows. |
| [`c_ept/preprocessing/`](c_ept/preprocessing/) | Build feature, terrain, token and trait representations. |
| [`c_ept/domain/`](c_ept/domain/) | Engagement units, targets, state and symmetry transformations. |
| [`c_ept/valuation/`](c_ept/valuation/) | Prices, reward rules, tower tiers and value composition. |
| [`c_ept/models/`](c_ept/models/) | Baselines, prediction heads and negative log-likelihood calculations. |
| [`c_ept/ept/`](c_ept/ept/) | EPT encoder, heads, loss, training step, checkpoints and training orchestration. |
| [`c_ept/runtime/`](c_ept/runtime/) | Resource observations and execution support. |
| [`c_ept/checks/`](c_ept/checks/) | Contract checks used by the research workflow. |
| [`c_ept/workflows/`](c_ept/workflows/) | Stage runners, the manual cost operator and its lightweight startup policy. |
| [`c_ept/reporting/`](c_ept/reporting/) | Existing S1, S2 and S3 report builders. |
| [`c_ept/_compat/`](c_ept/_compat/) | Compatibility entry files for historical module imports and commands. |

For example, new code can use `from c_ept.data import reader` instead of relying
on the old flat `c_ept.reader` path. The two paths refer to the same module object,
so existing imports and monkeypatches continue to use the same state.

The package root keeps configuration, pinned legacy-resource access, protected
cohort definitions and the run-log location anchor. Removed EPT root file paths
are no longer direct-file entry points; use `python -m c_ept.run_s4a` or the
PowerShell wrapper for that command. Legacy imports and module
commands resolve through `_compat/`; workflow implementations live in `workflows/`.
The root now has five Python files (`__init__`, `cohort`, `config`, `legacy`,
`runlog`), compared with 39 before this cleanup. Use the grouped folders for new
edits. See [next execution steps](docs/S4A_NEXT_STEPS_KO.md)
for the distinction between a development checkout and a frozen measurement checkout.

## CoG

[`COG/analysis/fight_boundaries/`](COG/analysis/fight_boundaries/) brings together
ability ranges, map regions, kill pairs, spatial and temporal boundary estimation,
and the boundary pipeline. Old `analysis.<module>` imports remain compatible.

The existing [`COG/acquisition/`](COG/acquisition/), [`COG/data/`](COG/data/),
[`COG/gameplay/`](COG/gameplay/), [`COG/core/`](COG/core/), [`COG/train/`](COG/train/)
and [`COG/app/`](COG/app/) folders retain their respective responsibilities.
Date-specific scripts stay associated with the research tasks that produced them.

## Master / ToG

[`master_ver/scripts/reporting/`](master_ver/scripts/reporting/) holds the existing
report generators for horizons, position analyses, event boundaries and validation.
The old script files forward to these implementations. Their output locations
remain relative to the same Master project root.

[`master_ver/studies/`](master_ver/studies/) retains the dated study packages;
these represent distinct experiments, not versions of one interchangeable model.
[`master_ver/module-layout.json`](master_ver/module-layout.json) records the report
relocations. The source-sync tool refuses an `--apply` that would overwrite this
derived layout. Refresh original sources in a separate mirror and review their
changes before integrating them.

## Shared support

- [`tools/project_runtime/`](tools/project_runtime/) is the small CoG/Master launcher library.
- [`research_pipeline/`](research_pipeline/) contains separately named pipeline components and studies.
- [`common_training_r1/`](common_training_r1/) is one bounded common-input workflow. Its input, model, training and reporting files remain together because the standalone workflow uses those sibling paths.
- [`tools/check_project_boundaries.py`](tools/check_project_boundaries.py) checks the declared packaging and selected direct imports.

This layout groups software responsibilities. It does not merge research
definitions or revise historical results. A relocated implementation is a new
code revision; existing run hashes are not automatically valid for this tree.

Short package descriptions were drafted by Gemini through OpenCodex and reviewed
against the actual file mappings. See the project guides for the exact moves and
compatibility details.
