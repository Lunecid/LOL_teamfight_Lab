# RRX — external dual-stage (V→W and q→SVI)

Generated: 2026-09-20T17:04:32+09:00
**TRAIN constant prior (for EXT constant baseline):** 0.5049

Score-only. Fail-closed without cohort keys. **Do not** infer external failure mode from 15.16 MAIN CORP alone.

## Summary

| Cohort | pilot | n | V_pre Brier | V_pre AUC | ΔBrier(q−PT_flex) | q MCB | q DSC | PT MCB | PT DSC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KR 16.13 | False | 5202 | 0.1511 | 0.8636 | 0.0026 | 0.0065 | 0.0082 | 0.0020 | 0.0063 |
| NA1 16.13 | False | 5312 | 0.1514 | 0.8641 | 0.0040 | 0.0100 | 0.0064 | 0.0037 | 0.0041 |
| KR 16.15 | False | 507 | 0.1533 | 0.8599 | 0.0006 | 0.0086 | 0.0102 | 0.0066 | 0.0087 |
| KR 16.14 pilot | True | 101 | 0.1560 | 0.8550 | -0.0085 | 0.0263 | 0.0337 | 0.0113 | 0.0102 |

## Reading

- Main external claim cohorts remain KR/NA1 16.13 (q lift lost vs PT).
- Compare V→W CORP vs q→SVI CORP **on the same external rows** before attributing cause.
- Artifacts: `outputs/review_response_rrx_external_20260920/`
