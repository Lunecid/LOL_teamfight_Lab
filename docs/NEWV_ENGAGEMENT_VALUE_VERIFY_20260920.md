# New-V engagement value-change verification

Generated: 2026-09-20T01:42:05+09:00
**Bundle SHA256:** `AC459CC4397630A953D672730CAA687B50FC1EDCD5A029928FAD236FC5B204E3`
**Fit scope:** `train_fit85_match_holdout_frac0.15_seed7_NO_full_train_refit`
**RQ:** Journal **J-RQ1** part ② / Master **M-RQ2** (not q / J-RQ2).

V freeze close: [V_EVALUATOR_FREEZE_CLOSE_20260920.md](V_EVALUATOR_FREEZE_CLOSE_20260920.md)

## 0. Design locks

- Engagement definition unchanged (T, onset/cutoff, h90 primary).
- Only V replaced by fit85 MLP Expanded.
- B40 recomputed from **new** p_pre; old SVI/B40 not reused.
- Exact ΔV=0 → Y_SVI=0; missing scores flagged, never coerced to 0/red.

## 1. Primary table (h90)

- n=32,981 / matches=24,020
- P(SVI=1)=0.498
- mean ΔV=-0.0014; mean |ΔV|=0.1123; median |ΔV|=0.0752
- exact 0 share=0.0000 (n=0)
- **B40 (new p_pre):** n=5,423 / matches=4,945; P(SVI=1)=0.504; mean |ΔV|=0.1840

### By match time

| Band | n | P(SVI=1) | E[ΔV] | mean |ΔV| | median |ΔV| |
|---|---:|---:|---:|---:|---:|
| t_0_10 | 2265 | 0.519 | 0.0050 | 0.1272 | 0.1068 |
| t_10_20 | 11858 | 0.495 | -0.0043 | 0.1114 | 0.0777 |
| t_20_30 | 15446 | 0.500 | 0.0002 | 0.1096 | 0.0615 |
| t_30_inf | 3412 | 0.473 | -0.0053 | 0.1378 | 0.1019 |

### By new p_pre

| p_pre bin | n | P(SVI=1) | E[ΔV] | mean |ΔV| |
|---|---:|---:|---:|---:|
| 0-20 | 8404 | 0.378 | 0.0149 | 0.0647 |
| 20-35 | 3906 | 0.448 | 0.0183 | 0.1594 |
| 35-40 | 1313 | 0.450 | 0.0005 | 0.1706 |
| 40-45 | 1333 | 0.479 | 0.0009 | 0.1798 |
| 45-55 | 2756 | 0.505 | 0.0039 | 0.1842 |
| 55-60 | 1334 | 0.521 | -0.0032 | 0.1824 |
| 60-65 | 1278 | 0.535 | -0.0077 | 0.1773 |
| 65-80 | 4044 | 0.541 | -0.0207 | 0.1597 |
| 80-100 | 8613 | 0.609 | -0.0200 | 0.0682 |

## 2. Material correspondence (new Y_SVI)

NEW Y_SVI from fit85 MLP — not old-label rates. Not independent fight-winner accuracy.

| Axis | n_decided | agree (match-wtd) | tie share | B40 agree |
|---|---:|---:|---:|---:|
| epic_net | 12507 | 0.818 | 0.621 | 0.842 |
| structure_net | 16106 | 0.774 | 0.512 | 0.812 |
| objective_net | 21779 | 0.810 | 0.340 | 0.860 |
| kill_diff | 28711 | 0.904 | 0.129 | 0.928 |
| alive_diff_post | 14584 | 0.794 | 0.558 | 0.788 |

### Kill-axis disagreements

- decided=28711; disagree share=0.097
- among disagree: obj↔SVI=691, obj↔kill=1079, obj tie=1010
- SVI blue & kill red=1273; SVI red & kill blue=1507

## 3. Stability

### Horizons (common valid n=32,981; h90-only n=32,981)

Horizon comparisons use intersection of valid cases; separate from sample-composition change.

| Pair | SVI flip | mean |ΔΔV| | nonzero sign agree |
|---|---:|---:|---:|
| h60_vs_h90 | 0.015 | 0.0059 | 0.985 |
| h90_vs_h120 | 0.007 | 0.0029 | 0.993 |
| h60_vs_h120 | 0.019 | 0.0079 | 0.981 |

### Peer evaluators (h90)

- MLP_vs_A0_wave2: 0.915
- MLP_vs_A_LR_expanded_fit85: 0.914
- Stability diagnostic only; do not retune V to maximize peer agreement.

### Observation refresh (h90)

- same_frame share=0.067 (n_same=2224, n_refreshed=30757)
- same_frame mean |ΔV|=0.0936; P(SVI=1)=0.529
- refreshed mean |ΔV|=0.1140; P(SVI=1)=0.496

## 4. Quiet (no-kill) reference

**Done (RR3):** [REVIEW_RESPONSE_RR3_QUIET_20260920.md](REVIEW_RESPONSE_RR3_QUIET_20260920.md) — same-match 1:1 match under frozen fit85; actual \(L_i\); proposal tolerances \(|\Delta p|\le0.025\), \(|\Delta t|\le120\)s.

TEST matched coverage ≈27.7% (9,140 / 32,981). Matched fights E[|ΔV|]≈0.080 vs quiet ≈0.017; paired excess ≈0.063 (95% CI excludes 0). Background-drift reference only — not a causal control. Old_V quiet ratios not reused.

## 5. RQ1 completion bar

Measurement table + three analysis axes are here. RQ1 closes when the manuscript can state which V, which window, what change, and how it corresponds to material outcomes / definition changes — not when SVI is proven correct.

q (J-RQ2) only after this measurement story is written. TRAIN SVI needs OOF V — separate from this freeze bundle.

