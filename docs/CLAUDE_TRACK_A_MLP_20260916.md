# Track A completion: plain MLP and residual MLP on the identical 352 inputs (h90, T/N)

2026-09-16. Design and audit: Claude (Fable 5.1), continuing the Codex line after the user's decision to carry on Codex's
work in this session; implementation and execution: the same session. GPU use for this stage is explicitly authorised
by the user (2026-09-16); the GPU deep search of the old market_event target (`tog_revision` queue, d2/d3) runs
concurrently on the same device. Exploratory follow-up after prior TEST exposure. Do not claim untouched confirmation.

## Why this stage

`DELTA_Q_FAIR_BALANCED_PROTOCOL_20260915.md` (RQ1) compares linear, tree and neural learners on the same
pre-engagement information; `COG_MODEL_COVERAGE_DELTA_Q_20260915.md` section 2 adds the plain matched-input MLP to
Track A next to the residual MLP. `incremental_q_training_20260915` (iq) fitted PT, full logistic and full LightGBM and
deferred both MLP forms. This bounded stage fits exactly those two families under the iq protocol and compares them
with the frozen iq winners on the same rows. It does not touch Track B (CoG representation adapters), h60/h120,
SHAP, revised V, or the definition re-estimation.

## Scope and immutable parents

New output root `outputs/track_a_mlp_20260916`; new scripts prefix `ta20260916_`; tests `tests/test_ta20260916_contracts.py`.
Python `C:/Users/todtj/anaconda3/python.exe` (torch 2.10.0+cu128, sklearn 1.6.1, numpy 2.1.3). Device `cuda` for
training; CPU threads <= 4; no installs, deletes, commits or pushes. Read-only parents: `full_corpus_training_20260915`
(features, labels, schema, V manifest), `cohort_role_training_20260915` (cohorts, A_* predictions),
`incremental_q_training_20260915` (frozen winners and their evaluation predictions), label_validity_full,
objective_channel_ablation, collaborator_critique_audit, source implementations, raw data, the original repository.
Pure helpers of `iq20260915_common` / `fc20260915_common` / `cr20260915_common` may be imported; helpers that write
to a parent root (`Status`, `log_command`, `log_failure`, `log_access`, `load_parent_set`, `load_trainval` of the iq
module, every `cr20260915_data` loader) are never called. Imports run with `-B` / `PYTHONDONTWRITEBYTECODE=1`.
Failures are preserved; stages resume safely without changing selections after evaluation.

## Fixed scientific contract (unchanged from iq)

1. MAIN corpus 210,000 matches: TRAIN 15.14, VALIDATION 15.15 (Q_CAL / Q_SELECT roles), TEST 15.16; external
   EXT_KR_16.13, EXT_KR_16.14_pilot, EXT_KR_16.15, EXT_NA1_16.13 evaluated separately, prior exposure acknowledged.
2. T = known min(cluster_blue, cluster_red) >= 4, N = known min < 4; parent cohort arrays with exact keys; expected
   h90 counts TRAIN T 39,605 / N 159,753; Q_CAL 10,390 / 40,565; Q_SELECT 10,195 / 40,180; TEST 32,981 / 130,595;
   externals as in the iq protocol. Any mismatch blocks fitting.
3. Parent Y = 1[delta_h90 > 0], masks, p_pre and feature values unchanged; TRAIN p_pre/Y from own-match-excluded OOF V
   (adapter provenance verified, V not refit). No post variables, no final winner W, no champion IDs, no snapshot age.
4. Inputs: exactly `predictor_sets['ridge']` of the parent `q_pre_only_schema.json` (352 numeric columns including
   p_pre_V), asserted finite. Median imputer (TRAIN-fitted; identity on finite data) then StandardScaler
   (TRAIN-fitted) for both MLP families. No draft / role / position additions.
5. Equal total training weight per match, row mean 1 (`fc20260915_common.weights`), recomputed in every fit subset.

## Two newly fitted families

Grid for both: width {128, 256, 512} x dropout {0.1, 0.3} = six configurations. Optimiser AdamW (lr 1e-3,
weight_decay 1e-4 on all parameters), batch 512, uniform shuffled minibatches, at most 100 epochs, float32, no AMP,
no gradient clipping, PyTorch default initialisation. Seeds 7, 42, 123; candidate raw probability = mean of the three
seed probabilities (never the best seed); seed probabilities stored.

- `mlp` (plain, CoG-style matched-input MLP): Linear(352, w) -> ReLU -> Dropout(d) -> Linear(w, w) -> ReLU -> Dropout(d)
  -> Linear(w, 1).
- `resmlp` (residual, protocol section 3): Linear(352, w) input projection; two blocks, each
  LayerNorm(w) -> Linear(w, 2w) -> ReLU -> Dropout(d) -> Linear(2w, w) -> Dropout(d) -> residual add; then LayerNorm(w)
  -> ReLU -> Linear(w, 1).

Loss: minibatch mean(w_i * BCEWithLogits_i) over uniform shuffled minibatches (no per-batch renormalisation, no
weighted sampler). Population loss sum(w_i * BCE_i) / sum(w_i); a numeric fixture checks that the full-batch minibatch
loss equals the population loss when the weights have mean 1.

## Stopping, refit, calibration, selection (iq protocol)

1. Internal stop set: the iq allocation `sha256('iq20260915_stop:' + match_id)[:8] mod 10 == 0` (same matches as the
   LightGBM arm, same allocation for T and N). Stop phase: preprocessing fit on fit90 only; after every epoch the
   match-weighted Brier on stop10 (weights recomputed inside stop10, GPU float32 inference, eval mode) is recorded;
   patience 10 epochs; ties keep the earlier epoch (first minimum); cap 100 epochs recorded as `cap_reached`.
2. Refit from scratch on ALL cohort TRAIN rows (preprocessing refit there) for exactly the chosen number of epochs,
   same seed and configuration. Neither a smoke model nor a stop-phase model is a final model.
3. Prediction definition: the saved float32 weights evaluated on CPU in float64 with a fixed chunk size (8192 rows) and
   `torch.set_num_threads(4)`; this is device-independent and is what is stored, frozen, reloaded and compared. The
   stop curve is a GPU float32 diagnostic; its chosen epoch index is recorded.
4. Calibration on cohort Q_CAL only (`cr20260915_common.fit_calibrators`: raw identity, sigmoid = LogisticRegression on
   clipped logit without a positive-slope constraint, isotonic with clipping) on the three-seed raw mean, never per seed.
5. Six configurations x three calibrations = 18 joint candidates per family; names `<config>__<calibration>` frozen in
   lexical order before fitting; ranked by cohort Q_SELECT match-weighted Brier, then log loss, then name. Q_CAL and
   Q_SELECT weights recomputed from their own match ids. No selection inside balanced / time / external subsets.
6. Freeze (`frozen_manifest.json`) of both families' bundles, selections and hashes for BOTH cohorts, plus the hashes of
   the iq frozen manifest and iq evaluation prediction files, BEFORE any TEST/external array or any iq/parent TEST
   prediction is opened. Sealed access is gated on THIS run's frozen manifest, not on the iq one. Reload identity of
   every saved bundle is verified before the freeze.
7. A five-family overall ranking (iq PT, logistic, LightGBM winners and the two MLP winners) by the same rule on the
   same cohort Q_SELECT rows is recorded as information; the iq winners are not refit or reselected.

## Evaluation

Sets: MAIN_TEST, the four external sets; T primary, N secondary; all h90-valid rows. Cells from frozen p_pre and pre
time only: all, B40 = [0.40, 0.60], B45 = [0.45, 0.55], pre-time bins [0,10), [10,20), [20,30), [30, inf), B40 x time.
Metrics per cell as in iq (rows, matches, positives, match-weighted Brier / log loss / AUC, descriptive calibration
intercept/slope, fixed 10-bin reliability and ECE, exact 0/1 counts; weights recomputed inside each cell; AUC NA for
single-class cells; < 30 matches marked sparse without intervals).

Comparators joined by exact (match, s_ms) keys from the iq evaluation predictions (sealed until the freeze):
`lgbm_winner`, `logit_winner`, `pt_winner`, `old_A_specialist`, `old_p_pre_spline`, `old_pooled`, `old_constant`.
New: `mlp_winner`, `resmlp_winner` (and every joint candidate, descriptive).

Predeclared contrasts (a - b; negative Brier / log loss = a better), paired match bootstrap 1,000 replicates, seed
20260915, same draw for every model in a cell, on all / B40 / B45 cells with >= 30 matches and both label classes:

- PRIMARY: `resmlp_winner - lgbm_winner` (protocol section 5, contrast (c))
- planned: `mlp_winner - logit_winner`; `resmlp_winner - mlp_winner`; `mlp_winner - lgbm_winner`;
  `resmlp_winner - pt_winner`; `mlp_winner - pt_winner`.

Time cells carry point estimates and reliability only (CI explicitly not computed). Fixed-model evaluation
uncertainty only; no multiple-comparison adjustment; no promotion of a better-looking pair to primary.

## Required gates and deliverables

Tests: exact 352 schema and forbidden fields; both architectures (layer shapes, parameter counts); weighted minibatch
loss fixture; fit90/stop10 disjointness and reuse of the iq allocation; preprocessing fit on fit90 in the stop phase and
refit on all TRAIN; first-minimum tie rule and cap; deterministic training for a fixed seed (CPU fixture); raw seed
ensemble before calibration; Q_CAL/Q_SELECT weight separation; deterministic selection incl. ties; balanced boundaries
and time bins; key-join completeness; own-match OOF provenance on the TRAIN fixture; bundle reload identity (CPU float64);
sealed gate keyed on this run's freeze even though the iq freeze exists; scripts write only under this root; parent
preservation. Order: integrity snapshot -> contract tests -> protocol.json -> TRAIN-only smoke (fit, freeze,
evaluation on pseudo Q_SELECT rows, stand-in comparators) -> contract tests again on the final code -> full fits ->
freeze -> sealed evaluation -> snapshot after -> post-run checks -> Korean REPORT.md and DEFINITION_AND_EVIDENCE.md.

Save protocol.json, source hashes, commands/environment/failure logs, status.json, candidate registry, stop records
(curves, chosen epochs), fit membership hashes, selection files, frozen manifest, bundles, raw/seed/calibrated
predictions for TRAIN/Q_CAL/Q_SELECT and every evaluation set, metrics and paired intervals, validation.json with real
passed/failed checks. Do not declare success if a required check or model is missing. Keep an honest negative or mixed
result if LightGBM or the strengthened baseline wins.

References and scope: Gorishniy et al. 2021 (ResNet-style tabular baseline, motivation only); Ke et al. 2017 (LightGBM
comparator); Loshchilov & Hutter 2019 (AdamW); PyTorch and scikit-learn documentation for the implementation. Widths,
dropout rates, learning rate, batch size, epoch cap, patience, seeds and reporting cut-offs are operational choices,
not externally established optima; six configurations per family is a bounded search convention, not equal computation.
