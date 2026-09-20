# Review response RR1+RR2 ??flexible baselines vs frozen q

Generated: 2026-09-21T00:23:18+09:00
**Design:** [REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md](REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md)
**Epistemic:** EXPLORATORY_SCALE_SPLIT_PRIOR_TEST_EXPOSURE (post-TEST review response; not confirmatory preregistration)
**Cohort tag:** `S`

## Locks

- V / SVI / engagement S / frozen `logit_state` weights: **unchanged**
- New: `b_spline`, `PT_flex`, optional `g_q` (PosSlopeSigmoid) selected on Q_SELECT all-S
- Selected calibrators: `{"constant": "identity", "b_linear": "sigmoid", "b_spline": "sigmoid", "PT_linear": "sigmoid", "PT_flex": "sigmoid", "q_base": "sigmoid"}`
- PT_flex config: `{"n_knots_p": 4, "n_knots_t": 4, "C": 0.01, "degree": 3}`
- b_spline config: `{"n_knots": 4, "C": 0.01}`

## TEST 15.16 S ??all

| Model | n | matches | Brier | logloss | AUC |
|---|---:|---:|---:|---:|---:|
| q_RR (sigmoid) | 101205 | 49730 | 0.2456 | 0.6842 | 0.5757 |
| PT_flex | 101205 | 49730 | 0.2489 | 0.6910 | 0.5360 |
| PT_linear | 101205 | 49730 | 0.2492 | 0.6915 | 0.5301 |
| b_spline | 101205 | 49730 | 0.2491 | 0.6914 | 0.5301 |
| b_linear | 101205 | 49730 | 0.2492 | 0.6915 | 0.5301 |
| constant | 101205 | 49730 | 0.2500 | 0.6932 | 0.5000 |
| q_base_raw | 101205 | 49730 | 0.2460 | 0.6854 | 0.5757 |

## TEST 15.16 S ??B40

| Model | n | matches | Brier | logloss | AUC |
|---|---:|---:|---:|---:|---:|
| q_RR (sigmoid) | 31675 | 25039 | 0.2478 | 0.6888 | 0.5507 |
| PT_flex | 31675 | 25039 | 0.2498 | 0.6928 | 0.4999 |
| PT_linear | 31675 | 25039 | 0.2498 | 0.6927 | 0.5038 |
| b_spline | 31675 | 25039 | 0.2498 | 0.6927 | 0.5042 |
| b_linear | 31675 | 25039 | 0.2498 | 0.6927 | 0.5042 |
| constant | 31675 | 25039 | 0.2501 | 0.6933 | 0.5000 |
| q_base_raw | 31675 | 25039 | 0.2481 | 0.6894 | 0.5507 |

### Primary RR contrast: ?Brier(q_RR ??PT_flex)

- **All S:** estimate=-0.00334  CI95=[-0.00368, -0.00300]  bootstrap_fraction_positive=0.0000
- **B40:** estimate=-0.00200  CI95=[-0.00242, -0.00154]  bootstrap_fraction_positive=0.0000
- All S vs PT_linear (continuity): estimate=-0.00363  CI95=[-0.00398, -0.00327]

### Heterogeneity H = D_B40 ??D_outside (D = Brier(q)?’Brier(PT_flex))

- D_B40=-0.00200  D_outside=-0.00393  H=0.00193
- 95% CI=[0.00130, 0.00257]  bootstrap_fraction_positive(H>0)=1.0000

Note: bootstrap_fraction_positive (JSON p_gt0) is match-bootstrap draw share with Delta>0 ??not a classical p-value.
Execution honesty: docs/REVIEW_RESPONSE_RR1_EXECUTION_ADDENDUM_20260920.md
RR0: docs/REVIEW_RESPONSE_RR0_MANIFEST_20260920.json

## Balance neighborhood (0.05-width)

| Bin | n | matches | q Brier | PT_flex Brier | ?Brier | CI95 | low-support |
|---|---:|---:|---:|---:|---:|---|---|
| [0.40,0.45) | 7612 | 7167 | 0.2479 | 0.2498 | -0.00188 | [-0.00277, -0.00101] | False |
| [0.45,0.50) | 8693 | 8133 | 0.2484 | 0.2498 | -0.00140 | [-0.00221, -0.00059] | False |
| [0.50,0.55) | 8329 | 7800 | 0.2479 | 0.2498 | -0.00193 | [-0.00275, -0.00113] | False |
| [0.55,0.60) | 7041 | 6601 | 0.2481 | 0.2501 | -0.00195 | [-0.00289, -0.00096] | False |

## Narrow p_pre bins (diagnostic; do not cherry-merge)

| Bin | n | matches | P(Y=1) | E[?V] | ?Brier(q?’PT_flex) | low-support |
|---|---:|---:|---:|---:|---:|---|
| [0.00,0.05) | 4942 | 4232 | 0.4345 | 0.0065 | -0.00390 | False |
| [0.05,0.10) | 3751 | 3475 | 0.4476 | 0.0150 | -0.00311 | False |
| [0.10,0.15) | 3461 | 3233 | 0.4674 | 0.0201 | -0.00078 | False |
| [0.15,0.20) | 3498 | 3290 | 0.4810 | 0.0196 | -0.00220 | False |
| [0.20,0.25) | 3774 | 3570 | 0.4938 | 0.0209 | -0.00247 | False |
| [0.25,0.30) | 4317 | 4103 | 0.5041 | 0.0213 | -0.00316 | False |
| [0.30,0.35) | 4917 | 4657 | 0.5027 | 0.0156 | -0.00222 | False |
| [0.35,0.40) | 5934 | 5595 | 0.5111 | 0.0135 | -0.00145 | False |
| [0.40,0.45) | 7612 | 7167 | 0.5175 | 0.0086 | -0.00188 | False |
| [0.45,0.50) | 8693 | 8133 | 0.5251 | 0.0079 | -0.00140 | False |
| [0.50,0.55) | 8329 | 7800 | 0.5189 | 0.0021 | -0.00193 | False |
| [0.55,0.60) | 7041 | 6601 | 0.5124 | -0.0021 | -0.00195 | False |
| [0.60,0.65) | 5587 | 5289 | 0.5165 | -0.0056 | -0.00171 | False |
| [0.65,0.70) | 4688 | 4459 | 0.5071 | -0.0099 | -0.00267 | False |
| [0.70,0.75) | 4139 | 3910 | 0.5259 | -0.0123 | -0.00412 | False |
| [0.75,0.80) | 3653 | 3468 | 0.5309 | -0.0154 | -0.00474 | False |
| [0.80,0.85) | 3549 | 3364 | 0.5336 | -0.0166 | -0.00763 | False |
| [0.85,0.90) | 3467 | 3276 | 0.5521 | -0.0159 | -0.00518 | False |
| [0.90,0.95) | 3943 | 3655 | 0.5392 | -0.0187 | -0.00807 | False |
| [0.95,1.00] | 5910 | 5073 | 0.5305 | -0.0101 | -0.00777 | False |

## Interpretation guardrails

- Negative ?Brier ??q better than that baseline on this cell.
- If B40 CI includes 0: **additional lift not clearly confirmed** here ??not 'no information' / study failure.
- Surviving PT_flex ??'beyond the *tested* p/time summaries'; disappearing ??function form of initial edge explained part of the prior linear gap.
- Original primary table vs PT_linear remains the historical result; this pack is review-response.

Artifacts: `C:/Users/todtj/PycharmProjects/LOL_teamfight/outputs/review_response_rr12_20260920_S_qTS/`

