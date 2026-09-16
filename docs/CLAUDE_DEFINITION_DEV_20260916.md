# Development-only engagement definition: re-detection with TRAIN-patch G/D, population census, relabel and refit (h90)

2026-09-16. Design, implementation, execution and audit: Claude (continuing the Codex line). Answers the readiness audit §5
("G/D were estimated on 15.14, 15.15 and 15.16 together; the whole procedure including the definition is not independent of
15.16 — a supplementary experiment re-estimating on development patches only is needed") and item 3 of the critique response §7
("development-only G/D re-detection changes the population; report the full-population metrics after the change and the
composition of common / changed cases; never call an AUC difference the pure effect of the definition").

## Definition under test

The frozen definition (corpus v3.3, preset `v3.3`) uses the pooled estimates rounded as the manuscript quotes them:
G = 13.7 s (`TF2_KILL_CLUSTER_GAP_MS` 13700; pooled estimate 13.7246 s) and D = 4,264 u (`CLUSTER_MAX_DIAMETER` 4264.0;
pooled 4,263.87 u). The development-only definition takes the estimates of the TRAIN patch alone
(`config/fight_boundary/spec_15.14.json`, 73,972 matches, 3.67 M intervals): G = 13.9637 s, D = 4,285.26 u, rounded with the
same convention: **G = 14.0 s (14000 ms), D = 4,285 u**. R = 1,600 u, B = 15 s, M = 2 and every other detector switch are
unchanged. (15.15 alone gives 13.49 s / 4,265 u; both patches lie inside the pooled ARI plateau 10-18 s and the D CI, which is
why the pooled verdict was "one definition"; this stage asks what actually changes when the strictest development-only choice
is used.)

## Procedure

1. **Re-detection** (unsealed: reads cache timelines only): `gameplay.fights.detect_fights` + `data.index_split._fight_to_ref_row`
   + `measure_postkill_objective_delay.analyse_match` — the exact parent code path (`fc20260915_external_prepare.detect_exposures`)
   — on every MAIN match (210,000) and every external cache, with the dev constants. The same path with the frozen constants is
   first run on 2,000 TRAIN matches and must reproduce the parent `exposures.csv` rows exactly (the parent verified 300 on 9/15).
   Per exposure row the detector's raw cluster participation counts are kept for the cohort rule.
2. **Census**: per set and role, matches whose exposure set (match, patch, s, L, next_start, end, same_match_overlap,
   end_observed) is identical vs changed; rows common / removed / added; for parent rows, how many h90-valid T / N rows are
   affected (join with the parent labels and cohorts by (match, s)). A row whose 8-tuple is unchanged has identical states,
   labels and cohort under both definitions (the extraction is a deterministic function of the tuple and the cache), so only
   changed matches are rebuilt.
3. **Rebuild** of changed matches: the parent extraction function (`fc20260915_extract.extract_chunk`) with the dev exposures,
   then labels with the frozen V (own held-out-fold OOF adapter for TRAIN matches, final adapter otherwise; Y_h = 1[p_post − p_pre > 0]),
   the pre-only q inputs (StateV2 minus `snapshot_age_s`, plus p_pre_V; same 362 names) and the cohort (T: min side cluster
   participation >= 4, from the detector's raw counts). The dev population of a set = parent rows of unchanged matches (copied
   exactly) + rebuilt rows of changed matches. A fixture of 100 unchanged TRAIN matches rebuilt with the parent exposures must
   reproduce the parent rows bitwise (contract test). TEST / external sets are rebuilt only after this run's freeze.
4. **Refit** (base arm = 352 ridge inputs, iq h90 configurations fixed; logit and LightGBM; calibration selected on dev Q_SELECT)
   on the dev TRAIN population, T and N; freeze.
5. **Sealed evaluation** on the dev MAIN TEST and external populations: the refit winners and the frozen parent iq winners
   (parent bundles, not refit) on the same rows; cells all / B40 / B45 / time; paired match bootstrap (1,000, seed 20260915)
   with contrasts refit − frozen per family; on the common rows the frozen winner's predictions must equal the parent's
   (identity check); population metrics under the two definitions are reported side by side (descriptive, not paired) with
   the row census.

## What is reported

Census (matches / rows changed by set, role, cohort); the dev population sizes; frozen-winner metrics on the old population
(parent results), on the dev population, on common rows and on changed rows; refit − frozen contrasts on the dev population;
selected calibrations and stop records. PRIMARY quantity (pre-registered): cohort T, MAIN TEST, LightGBM: Brier of the frozen
parent winner on the dev population vs the parent's reported Brier on the old population, together with the fraction of
changed T rows; the refit − frozen LightGBM Brier contrast on the dev population (all cell) is the pre-registered contrast.

## Not done

Re-estimation of G/D on 15.14 + 15.15 pooled (only per-patch specs exist); any change to V; V refit on a dev-only definition
(V is a state model independent of engagements); human review; causal reading.

## Layout

`outputs/definition_dev_20260916/`: `protocol.json`, `redetect/` (per set csv + json), `census.json`, `changed/`, `extract/<set>/`,
`labels/`, `cohorts/`, `models/`, `selection/`, `predictions/`, `frozen_manifest.json`, `eval/`, `validation.json`, `REPORT.md`.
Scripts `scripts/dd20260916_*.py`, tests `tests/test_dd20260916_contracts.py`.
