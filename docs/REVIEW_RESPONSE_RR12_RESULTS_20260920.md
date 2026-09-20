# Review response RR1+RR2 — flexible baselines vs frozen q

Generated: 2026-09-20T10:51:39+09:00
**Design:** [REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md](REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md)
**Epistemic:** EXPLORATORY_REVIEW_RESPONSE_PRIOR_TEST_EXPOSURE (post-TEST review response; not confirmatory preregistration)

## Locks

- V / SVI / engagement T / frozen `logit_state` weights: **unchanged**
- New: `b_spline`, `PT_flex`, optional `g_q` (PosSlopeSigmoid) selected on Q_SELECT all-T
- Selected calibrators: `{"constant": "sigmoid", "b_linear": "identity", "b_spline": "identity", "PT_linear": "identity", "PT_flex": "identity", "q_base": "identity"}`
- PT_flex config: `{"n_knots_p": 4, "n_knots_t": 4, "C": 0.01, "degree": 3}`
- b_spline config: `{"n_knots": 6, "C": 0.01}`

## TEST 15.16 T — all

| Model | n | matches | Brier | logloss | AUC |
|---|---:|---:|---:|---:|---:|
| q_RR (identity) | 32981 | 24020 | 0.2355 | 0.6637 | 0.6403 |
| PT_flex | 32981 | 24020 | 0.2392 | 0.6713 | 0.6200 |
| PT_linear | 32981 | 24020 | 0.2397 | 0.6723 | 0.6164 |
| b_spline | 32981 | 24020 | 0.2397 | 0.6723 | 0.6167 |
| b_linear | 32981 | 24020 | 0.2397 | 0.6723 | 0.6162 |
| constant | 32981 | 24020 | 0.2500 | 0.6932 | 0.5000 |
| q_base_raw | 32981 | 24020 | 0.2355 | 0.6637 | 0.6403 |

## TEST 15.16 T — B40

| Model | n | matches | Brier | logloss | AUC |
|---|---:|---:|---:|---:|---:|
| q_RR (identity) | 5423 | 4945 | 0.2474 | 0.6885 | 0.5669 |
| PT_flex | 5423 | 4945 | 0.2495 | 0.6922 | 0.5248 |
| PT_linear | 5423 | 4945 | 0.2497 | 0.6925 | 0.5209 |
| b_spline | 5423 | 4945 | 0.2497 | 0.6926 | 0.5203 |
| b_linear | 5423 | 4945 | 0.2497 | 0.6925 | 0.5203 |
| constant | 5423 | 4945 | 0.2500 | 0.6931 | 0.5000 |
| q_base_raw | 5423 | 4945 | 0.2474 | 0.6885 | 0.5669 |

### Primary RR contrast: ΔBrier(q_RR − PT_flex)

- **All T:** estimate=-0.00373  CI95=[-0.00461, -0.00279]  bootstrap_fraction_positive=0.0000
- **B40:** estimate=-0.00214  CI95=[-0.00422, -0.00003]  bootstrap_fraction_positive=0.0240
- All T vs PT_linear (continuity): estimate=-0.00422  CI95=[-0.00515, -0.00323]

**Wording:** all-T lift vs tested `PT_flex` is the firm panel. B40 is a **small exploratory** improvement (CI upper bound ≈ 0; interval crosses \(\tau=0.001\)). Do not call B40 a “clear ≥0.001 gain.”  
`bootstrap_fraction_positive` = share of match-bootstrap draws with ΔBrier>0 (JSON field `p_gt0`); **not** a classical \(p\)-value.

### Heterogeneity H = D_B40 − D_outside (D = Brier(q)−Brier(PT_flex))

- D_B40=-0.00214  D_outside=-0.00395  H=0.00181
- 95% CI=[-0.00060, 0.00422]  bootstrap_fraction_positive(H>0)=0.9395

Do **not** conclude that balanced states are significantly harder; \(H\) CI covers 0.

### Execution honesty (design ≠ code gaps closed in docs)

See [REVIEW_RESPONSE_RR1_EXECUTION_ADDENDUM_20260920.md](REVIEW_RESPONSE_RR1_EXECUTION_ADDENDUM_20260920.md): uniform knots (not weighted quantiles); fit weights not mean-1; two-stage hyperparam then calibrator selection. RR0 key/hash lock: [REVIEW_RESPONSE_RR0_MANIFEST_20260920.json](REVIEW_RESPONSE_RR0_MANIFEST_20260920.json).

## Balance neighborhood (0.05-width)

| Bin | n | matches | q Brier | PT_flex Brier | ΔBrier | CI95 | low-support |
|---|---:|---:|---:|---:|---:|---|---|
| [0.40,0.45) | 1333 | 1311 | 0.2472 | 0.2492 | -0.00200 | [-0.00620, 0.00223] | False |
| [0.45,0.50) | 1346 | 1318 | 0.2475 | 0.2501 | -0.00263 | [-0.00654, 0.00143] | False |
| [0.50,0.55) | 1410 | 1384 | 0.2453 | 0.2501 | -0.00480 | [-0.00867, -0.00070] | False |
| [0.55,0.60) | 1334 | 1300 | 0.2499 | 0.2493 | 0.00056 | [-0.00350, 0.00499] | False |

## Narrow p_pre bins (diagnostic; do not cherry-merge)

| Bin | n | matches | P(Y=1) | E[ΔV] | ΔBrier(q−PT_flex) | low-support |
|---|---:|---:|---:|---:|---:|---|
| [0.00,0.05) | 3344 | 3049 | 0.3684 | 0.0067 | -0.00619 | False |
| [0.05,0.10) | 2056 | 1958 | 0.3725 | 0.0187 | -0.00299 | False |
| [0.10,0.15) | 1612 | 1545 | 0.3753 | 0.0167 | -0.00170 | False |
| [0.15,0.20) | 1392 | 1364 | 0.4080 | 0.0210 | -0.00118 | False |
| [0.20,0.25) | 1361 | 1325 | 0.4391 | 0.0245 | 0.00056 | False |
| [0.25,0.30) | 1260 | 1229 | 0.4485 | 0.0170 | 0.00343 | False |
| [0.30,0.35) | 1285 | 1257 | 0.4588 | 0.0144 | 0.00033 | False |
| [0.35,0.40) | 1313 | 1284 | 0.4502 | 0.0005 | -0.00075 | False |
| [0.40,0.45) | 1333 | 1311 | 0.4786 | 0.0009 | -0.00200 | False |
| [0.45,0.50) | 1346 | 1318 | 0.5024 | 0.0085 | -0.00263 | False |
| [0.50,0.55) | 1410 | 1384 | 0.5070 | -0.0004 | -0.00480 | False |
| [0.55,0.60) | 1334 | 1300 | 0.5214 | -0.0032 | 0.00056 | False |
| [0.60,0.65) | 1278 | 1249 | 0.5347 | -0.0077 | -0.00615 | False |
| [0.65,0.70) | 1311 | 1272 | 0.5216 | -0.0200 | -0.00575 | False |
| [0.70,0.75) | 1356 | 1327 | 0.5279 | -0.0249 | -0.00461 | False |
| [0.75,0.80) | 1377 | 1354 | 0.5694 | -0.0175 | -0.00254 | False |
| [0.80,0.85) | 1333 | 1296 | 0.5494 | -0.0341 | -0.00532 | False |
| [0.85,0.90) | 1585 | 1532 | 0.5973 | -0.0221 | -0.00650 | False |
| [0.90,0.95) | 1985 | 1891 | 0.6305 | -0.0205 | -0.00761 | False |
| [0.95,1.00] | 3710 | 3376 | 0.6223 | -0.0123 | -0.00709 | False |

## Interpretation guardrails

- Negative ΔBrier ⇒ q better than that baseline on this cell.
- If B40 CI includes 0: **additional lift not clearly confirmed** here — not 'no information' / study failure.
- Surviving PT_flex ⇒ 'beyond the *tested* p/time summaries'; disappearing ⇒ function form of initial edge explained part of the prior linear gap.
- Original primary table vs PT_linear remains the historical result; this pack is review-response.

Artifacts: `C:/Users/todtj/PycharmProjects/LOL_teamfight/outputs/review_response_rr12_20260920/`

