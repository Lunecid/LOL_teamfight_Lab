# RRX — external dual-stage (V→W and q→SVI) — cohort S

Generated: 2026-09-21T00:24:43+09:00
**cohort-tag:** S
**TRAIN constant prior:** 0.4975
**MAIN feature-order sha16:** `103f0b92094847bc`

Score-only. Fail-closed without cohort keys / feature-order mismatch. V→W and q→SVI use **identical common-valid rows** (sealed W + finite V/q/PT). **No CI.**

## Summary

| Cohort | n | align | V_pre Brier | V_post Brier | ΔBrier(q−PT) | ΔMCB | ΔDSC | q MCB | q DSC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| KR 16.13 | 15641 | True | 0.1872 | 0.1776 | -0.0018 | 0.0011 | 0.0029 | 0.0028 | 0.0034 |
| NA1 16.13 | 16100 | True | 0.1871 | 0.1776 | -0.0020 | 0.0013 | 0.0033 | 0.0038 | 0.0035 |
| KR 16.15 | 1307 | True | 0.1829 | 0.1732 | -0.0033 | 0.0014 | 0.0046 | 0.0059 | 0.0050 |
| KR 16.14 pilot | 285 | True | 0.1787 | 0.1631 | 0.0035 | 0.0091 | 0.0056 | 0.0171 | 0.0104 |

### Alignment / fail-closed

- `KR 16.13`: n_loaded=15641, n_common=15641, n_common==n_loaded=True, feature_ok=True
- `NA1 16.13`: n_loaded=16100, n_common=16100, n_common==n_loaded=True, feature_ok=True
- `KR 16.15`: n_loaded=1307, n_common=1307, n_common==n_loaded=True, feature_ok=True
- `KR 16.14 pilot`: n_loaded=285, n_common=285, n_common==n_loaded=True, feature_ok=True

## Note

Hardcoded T Reading section omitted for S. Interpret in `docs/SCALE_SPLIT_TvsS_RESULTS_20260920.md` from JSON fields only.
- Detail: `outputs/review_response_rrx_external_20260920_S/rrx_external_results.json`
