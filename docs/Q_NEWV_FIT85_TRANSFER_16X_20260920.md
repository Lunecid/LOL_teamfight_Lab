# Transfer — frozen new-V q on 16.x (score-only)

Generated: 2026-09-20T09:37:30+09:00
**Frozen q:** `logit_state` · **Frozen V:** fit85 MLP Expanded

Per [PAPER_COHORT_CONTRACT_20260919.md](PAPER_COHORT_CONTRACT_20260919.md): **210k KR 15.14–15.16** for fit/select/eval; **16.x KR/NA1 = transfer only (no refit)**.

## Teamfight T

| Cohort | n | P(SVI=1) | q Brier | PT Brier | ΔBrier (q−PT) | q AUC | Pilot? |
|---|---:|---:|---:|---:|---:|---:|---|
| KR 16.13 | 5202 | 0.491 | 0.2483 | 0.2456 | 0.00274 | 0.5956 | False |
| NA1 16.13 | 5312 | 0.485 | 0.2534 | 0.2500 | 0.00342 | 0.5850 | False |
| KR 16.15 | 507 | 0.529 | 0.2477 | 0.2507 | -0.00301 | 0.5754 | False |
| KR 16.14 pilot | 101 | 0.485 | 0.2426 | 0.2458 | -0.00320 | 0.6085 | True |

## B40 (new p_pre from frozen V)

| Cohort | B40 n | q Brier | PT Brier | q AUC |
|---|---:|---:|---:|---:|
| KR 16.13 | 928 | 0.2566 | 0.2491 | 0.5448 |
| NA1 16.13 | 885 | 0.2582 | 0.2500 | 0.5646 |
| KR 16.15 | 97 | 0.2617 | 0.2502 | 0.4861 |
