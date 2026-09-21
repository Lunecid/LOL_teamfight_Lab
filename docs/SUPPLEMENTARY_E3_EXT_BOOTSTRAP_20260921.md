# Supplementary E3 — External paired match-cluster bootstrap

**generated:** 2026-09-21T07:31:43.353519+00:00  
**n_boot / seed:** 10000 / 7  
**epistemic:** SUPPLEMENTARY_SCORE_ONLY_NO_REFIT  

ci95 = 95% percentile; ci9875 = Bonferroni 0.05/4 family. Conditional on frozen models; not model-fit uncertainty. frac_boot_gt0 is not a classical p-value.

## Primary family (KR/NA1 16.13 × T/S)

| Cohort | Set | n_rows | matches | ΔBrier (match-mean) | CI95 | CI98.75 | frac>0 | Δ row-wtd |
|---|---|---:|---:|---:|---|---|---:|---:|
| KR_16.13|T | — | — | — | FAIL | None | — | — | — |
| KR_16.13|S | — | — | — | FAIL | None | — | — | — |
| NA1_16.13|T | — | — | — | FAIL | None | — | — | — |
| NA1_16.13|S | — | — | — | FAIL | None | — | — | — |

## Common-match H = mean(d_T − d_S) (optional; not a mechanism)

- **KR_16.13:** n_common=3181; H=+0.00364; CI95=[-0.0006309552549597759, 0.007899648704650248]; Not a mechanism for T/S difference (X-31 withdrawn).
- **NA1_16.13:** n_common=3290; H=+0.00525; CI95=[0.0009369116448284207, 0.009596677776073443]; Not a mechanism for T/S difference (X-31 withdrawn).

## Forbidden

- T/S mechanism
- recalibration will fix transfer
- replace frozen RRX headlines
