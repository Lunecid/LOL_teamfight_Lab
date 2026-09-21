# RR3 close-out — match scope, timing parity, sparse \(s_Q\)

Generated: 2026-09-20T15:06:39+09:00
Parent results: [REVIEW_RESPONSE_RR3_QUIET_20260920.md](REVIEW_RESPONSE_RR3_QUIET_20260920.md)

## Scope lock (do not over-claim)

> On the **matchable subset** under same-match / |Δp|≤0.025 / |Δt|≤120s / equal \(L_i\), fight |ΔV| exceeds quiet |ΔV|. Not: all fights; not causal fight-only contribution; not q predicting pure fight effect.

### Why coverage ≈28%

- Logical: if quiet must not overlap \([s,s+L]\) and \(|t-s|\le 120\)s with equal length, then **\(L>120\)s is unmatchable**.
- TEST share with \(L>120\)s: **9.2%** (n=3031).
- Empirically, unmatched fights are longer on average (see traits).

## Matched vs unmatched traits (TEST)

| Group | n | mean p_pre | mean t (min) | mean L (s) | mean |ΔV| |
|---|---:|---:|---:|---:|---:|
| Matched | 9140 | 0.5005 | 23.0105 | 52.3765 | 0.0805 |
| Unmatched | 23841 | 0.5063 | 20.5939 | 87.7517 | 0.1287 |

## Coverage by length / time / B40 (TEST)

| Bin | n | matched | coverage |
|---|---:|---:|---:|
| L_le_60s | 12753 | 7020 | 0.550 |
| L_60_120s | 17197 | 2120 | 0.123 |
| L_gt_120s | 3031 | 0 | 0.000 |
| t_2_10 | 2265 | 412 | 0.182 |
| t_10_20 | 11858 | 2559 | 0.216 |
| t_20_30 | 15446 | 4826 | 0.312 |
| t_30_inf | 3412 | 1343 | 0.394 |
| B40 | 5423 | 922 | 0.170 |
| outside_B40 | 27558 | 8218 | 0.298 |

## Timing / state-path parity (TEST sample)

- Sample ok=200 fail=0 (seed=7).
- Fight rebuild |Δp_pre| vs stored label: mean=0.00000 p95=0.00000
- Fight rebuild |Δ(ΔV)| vs stored label: mean=0.00000 p95=0.00000
- Quiet rebuild |Δ(ΔV)| (shipped \(t_0+L-1\)): mean=0.00000 p95=0.00000
- Measured spans: fight mean=54838.4 ms; quiet shipped mean=54838.4 ms; stored L mean=54839.4 ms.

**Convention note:** shipped quiet scored \([t_0,\,t_0+L-1]\) while overlap/kill used \([t_0,\,t_0+L]\). Report measurement length as post−pre query times. The ~1 ms span gap is recorded; it is not treated as the explanation of the ~6%p |ΔV| contrast.

## Statistics labeling

In RR3 tables, means / mean|ΔV| / P(ΔV>0) are match-weighted; median and quantiles are row-level (unweighted). Do not label the whole table as match-weighted.

## \(s_Q\) sparse-cell rule (for RR4)

- Fallback order: `['p_t_L', 'p_t', 'p', 'global']` with min_matches=100.
- Fallback counts: `{"to_p_t": 156, "to_p": 140}`.
- Artifact: `outputs/review_response_rr3_quiet_20260920/quiet_mu_s_Q_CAL_fallback.json`
- Do not impute unsupported long-\(L\) regions with ε defaults for ‘small-ΔV’ filters.

## Primary RR3 contrast (unchanged)

- TEST E[|ΔV|_fight − |ΔV|_quiet]=0.06333 CI95=[0.06118756221474504, 0.06542419549310519]

