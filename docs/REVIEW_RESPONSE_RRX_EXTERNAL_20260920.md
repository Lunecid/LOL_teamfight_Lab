# RRX — external dual-stage (V→W and q→SVI)

Generated: 2026-09-20T17:28:51+09:00
**TRAIN constant prior:** 0.5049
**MAIN feature-order sha16:** `103f0b92094847bc`

Score-only. Fail-closed without cohort keys / feature-order mismatch. V→W and q→SVI use **identical common-valid rows** (sealed W + finite V/q/PT).

## Summary

| Cohort | n | align | V_pre Brier | V_post Brier | ΔBrier(q−PT) | ΔMCB | ΔDSC | q MCB | q DSC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| KR 16.13 | 5202 | True | 0.1511 | 0.1272 | 0.0026 | 0.0045 | 0.0019 | 0.0065 | 0.0082 |
| NA1 16.13 | 5312 | True | 0.1514 | 0.1275 | 0.0040 | 0.0063 | 0.0023 | 0.0100 | 0.0064 |
| KR 16.15 | 507 | True | 0.1533 | 0.1318 | 0.0006 | 0.0021 | 0.0015 | 0.0086 | 0.0102 |
| KR 16.14 pilot | 101 | True | 0.1560 | 0.1069 | -0.0085 | 0.0150 | 0.0235 | 0.0263 | 0.0337 |

### Alignment / fail-closed

- `KR 16.13`: n_loaded=5202, n_common=5202, n_common==n_loaded=True, feature_ok=True
- `NA1 16.13`: n_loaded=5312, n_common=5312, n_common==n_loaded=True, feature_ok=True
- `KR 16.15`: n_loaded=507, n_common=507, n_common==n_loaded=True, feature_ok=True
- `KR 16.14 pilot`: n_loaded=101, n_common=101, n_common==n_loaded=True, feature_ok=True

## Reading

- KR/NA1 16.13: q DSC still > PT_flex, but larger ΔMCB → net ΔBrier > 0 (signal not fully gone; calibration component dominates the loss gap).
- V_pre Brier ~0.15 does not by itself validate EXT ΔV labels.
- EXT CORP is diagnostic — not a fitted EXT recalibrator.
- Detail: `outputs/review_response_rrx_external_20260920/rrx_external_results.json`
