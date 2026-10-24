# EPT package root cleanup

The author requested physical component grouping after showing the flat c_ept file list. This extends the accepted 2026-10-24 module layout; baseline source is 611a4a4ef092890fca52e00250608e70bc407e81.

## Scope and acceptance

- Move legacy compatibility files out of the visible package root into _compat. Use normal Python package-path lookup for old c_ept.<name> imports, with no global sys.path or meta_path hooks and no eager numerical imports.
- Group workflow, operator and reporting implementations where file-relative contracts can be preserved. Keep protected cohort, configuration and fixed-resource anchors in place.
- Preserve module object identity, monkeypatch state, legacy class loading, CLI dispatch, repository anchors, package distribution and fail-closed execution guards. Old filesystem script paths are not general public CLIs; audit supported module and wrapper commands explicitly.
- Add focused synthetic migration checks and run relevant existing regressions without research data, training, inference or report regeneration. Implementation and independent review remain separate.
- Update current navigation; historical audit hashes and equation inventories retain their historical meaning.

## Boundaries

Do not read protected cohort/holdout/split bodies or checkpoint payloads. Do not edit the failed cost-run checkout, its run records, input data, or original Master workspace. Strata stays off. Do not launch S4b/S6/S7 or a replacement cost run as part of this layout change.

The initial S4a run failed at the repository-cleanliness guard after S CPU completed. File organization in a new source revision neither repairs that run identity nor completes G-S4a. Subsequent measured runs must use a separately frozen checkout and a reviewed execution plan.

## Roles

Sol implements the bounded package cleanup. Codex integrates documentation and packaging evidence; a separate reviewer checks the final diff and tests before publication. Publishing follows the existing authorized code-only public snapshot procedure, excluding private Git ancestry, research data and frozen membership files.
