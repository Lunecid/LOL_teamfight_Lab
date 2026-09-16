# Explaining the composite system F(S) = q(h(S), V(S)) and the model-class range of group reliance (h90, cohort T)

2026-09-16. Design, implementation, execution and audit: Claude (continuing the Codex line). No fitting. Exploratory follow-up
after prior TEST exposure. Motivated by the literature pass of 2026-09-16 (Chen, Lundberg & Lee 2022 on explaining a series of
models; Aas, Jullum & Løland 2021 on dependent-feature masking; Fisher, Rudin & Dominici 2019 on model class reliance) and by
the limits recorded for the balanced-SHAP stage: masked `p_pre_V` was not recomputed from the masked state, derived `×time`
interactions were not recomputed, and two learners with overlapping Brier intervals distribute attribution differently.

## Part A — composite explanation

The frozen system is F(S) = q(h(S), V(S)): the frozen value model V (raw logistic, 361 state features + champion one-hot)
produces p_pre_V, which enters q together with the 351 other state columns (352 ridge inputs = 351 + p_pre_V). Among the 361
state columns, 95 are deterministic functions of others: `<x>_x_time = <x> · time_minutes / 30` (94 columns) and
`time_minutes_sq = time_minutes²`. The independent ("base") state therefore has 266 columns (361 − 95), which include
`time_minutes` and the 10 champion ids. `snapshot_age_s` is stored but used by neither V nor q.

Groups on the base columns (fixed order): time (1: `time_minutes`), economy_and_experience, combat_and_survival, objectives,
structures, health_mana_other (the `fc20260915_common.shap_group` assignment restricted to base columns) and
champion_identity (10). The prior-probability group of the input-space explanation does not exist here: p_pre is recomputed.

Value function: for an explained row x and a coalition S of groups, the masked base vector takes x on S and a background
row on the complement; the 95 derived columns are recomputed from the masked base; V is applied to the reconstructed 362-name
state (snapshot_age_s = 0, unused) to obtain p_pre; q's 352 inputs are assembled and the frozen bundle's calibrated
prediction is taken. Exact 7-group Shapley (128 coalitions, `fc20260915_shap.group_exact_shapley`) with the mean over the
same 128 TRAIN background rows as the balanced-SHAP stage; explained rows are the same 256 hash-selected MAIN TEST rows per
cell (all, B40) of cohort T, for the same two models (final q = iq LightGBM T winner; plain MLP = Track A winner).
This is an interventional (marginal-background) explanation of F, as in the balanced-SHAP stage; what changes is the
object explained — the original state through both paths — and the on-manifold consistency of the derived columns and
p_pre. It is not a conditional (observational) Shapley value and not a causal decomposition.

Checks: the reconstruction of the explained row's own base vector reproduces the stored 361 inputs and p_pre_V (tolerance
recorded; derived columns are recomputed in floating point) and F(x) equals the parent's stored MAIN TEST prediction for that
row (< 1e-9); efficiency: Σφ + base = F(x) (< 1e-9) and v(full) = F(x); explained and background rows identical to the
balanced-SHAP stage (row keys equal); rerun of the first 3 rows bitwise equal.

## Part B — model-class range of group reliance (Rashomon set), unsealed rows only

Candidates: every saved same-input (352 ridge) candidate for cohort T and N — incremental-q logistic (6 configurations) and
LightGBM (6), Track A plain MLP (6) and residual MLP (6), and the champion-class stage's base arms (1 + 1), each with its
three calibrations (raw, sigmoid, isotonic) → 78 candidates per cohort. Rows: MAIN_VALIDATION Q_SELECT rows of the cohort
(unsealed; the rows on which the parents selected). Loss: match-weighted Brier. The ε-Rashomon set is
{candidate : Brier_QSELECT ≤ min over all candidates + ε} for ε = 0.0005 and 0.001 (both reported; 0.0005 primary — about
the half-width of the paired intervals seen on TEST).

Group reliance of a candidate for group g (7 groups of the q input space, prior_win_probability included): the increase in
match-weighted Q_SELECT Brier when the g columns are jointly permuted across rows (5 fixed permutation seeds, mean; the
"switch" estimator of Fisher et al. in its permutation form), with predictions from the candidate's own bundle and
calibrator. Reported: per candidate the 7 reliances; per group the range [min, max] over the ε-set (the empirical model
class reliance), the reliance ordering of each candidate, and how often each group is the most relied-on group in the set.
This is an empirical Rashomon set of fitted candidates, not the full model-class optimisation of Fisher et al.; the
"switch" loss is the marginal-permutation loss, so dependent groups are pushed off-manifold exactly as in interventional
SHAP (stated limitation; the composite explanation in Part A addresses the p_pre/derived-column part of that dependence).

## Procedure and layout

`outputs/composite_shap_20260916/`: snapshot → contract tests → protocol.json (groups, row tags, ε, models, checks) →
Part B (unsealed) → freeze (plan freeze; sealed TEST rows opened only after) → Part A smoke on TRAIN pseudo rows → Part A
full → snapshot after → post-run checks → Korean REPORT.md. Scripts `scripts/cs20260916_*.py`, tests
`tests/test_cs20260916_contracts.py`. Parents (full corpus, incremental q, Track A, champion class, balanced SHAP) read only.

## Not done

Conditional/observational Shapley (Gaussian, copula or empirical conditional estimators); path-specific attribution (V path
vs direct path); N cohort for Part A; the true model-class optimisation of MCR; any change to models, labels or selections.
