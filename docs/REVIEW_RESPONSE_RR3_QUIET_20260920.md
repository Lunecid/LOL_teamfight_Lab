# RR3 — new-V matched quiet reference (fit85 MLP)

Generated: 2026-09-20T11:47:45+09:00
**Design:** [REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md](REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md) §7
**V bundle sha16:** `ac459cc4397630a9` · **CACHE:** `D:\LOL_Project\cache\match_cache_fresh_v3_engage_status13`
**Matching:** |Δp|≤0.025, |Δt|≤120s, step=15s, actual L_i (not constant 90s)

## Claim scope

> Under frozen fit85, matched no-kill windows of the same observed length show how large engagement ΔV is versus background drift. Not a causal fight effect; not old_V quiet reuse.

## Q_CAL (DEV matching / μ_Q, s_Q)

- fights=10390 matched=2987 coverage=0.287
- loaded matches=7550 cache_miss=0 builder_fail=0
- unmatched reasons: `{"no_candidate": 7403}`

| Arm | n | matches | E[ΔV] | E[\|ΔV\|] | median \|ΔV\| | P(ΔV>0) |
|---|---:|---:|---:|---:|---:|---:|
| Fight (matched) | 2987 | 2721 | -0.0006 | 0.0753 | 0.0341 | 0.5081 |
| Quiet (matched) | 2987 | 2721 | -0.0012 | 0.0164 | 0.0081 | 0.4920 |

### Paired contrast (match-bootstrap)

- **E[|ΔV|_fight − |ΔV|_quiet]:** 0.05890  CI95=[0.05551, 0.06252]  bootstrap_fraction_positive=1.0000
- **E[ΔV_fight − ΔV_quiet]:** 0.00066  CI95=[-0.00405, 0.00513]  bootstrap_fraction_positive=0.5985
- **P(ΔV>0)_fight − P(ΔV>0)_quiet:** 0.01608  CI95=[-0.01127, 0.04312]  bootstrap_fraction_positive=0.8865

Balance: mean|Δp|=0.0091 mean|Δt|=72.7688s mean|ΔL|=0.0000ms

## TEST 15.16 (primary quiet contrast)

- fights=32981 matched=9140 coverage=0.277
- loaded matches=24020 cache_miss=0 builder_fail=0
- unmatched reasons: `{"no_candidate": 23841}`

| Arm | n | matches | E[ΔV] | E[\|ΔV\|] | median \|ΔV\| | P(ΔV>0) |
|---|---:|---:|---:|---:|---:|---:|
| Fight (matched) | 9140 | 8438 | 0.0006 | 0.0800 | 0.0387 | 0.5029 |
| Quiet (matched) | 9140 | 8438 | -0.0022 | 0.0166 | 0.0081 | 0.4836 |

### Paired contrast (match-bootstrap)

- **E[|ΔV|_fight − |ΔV|_quiet]:** 0.06333  CI95=[0.06119, 0.06542]  bootstrap_fraction_positive=1.0000
- **E[ΔV_fight − ΔV_quiet]:** 0.00280  CI95=[0.00002, 0.00565]  bootstrap_fraction_positive=0.9750
- **P(ΔV>0)_fight − P(ΔV>0)_quiet:** 0.01922  CI95=[0.00456, 0.03387]  bootstrap_fraction_positive=0.9920

Balance: mean|Δp|=0.0088 mean|Δt|=72.2017s mean|ΔL|=0.0000ms

Close-out (traits, \(L>120\)s constraint, timing parity, \(s_Q\) fallback): [REVIEW_RESPONSE_RR3_CLOSEOUT_20260920.md](REVIEW_RESPONSE_RR3_CLOSEOUT_20260920.md)

## Guardrails

- Do not call quiet a causal control or independent fight-winner accuracy.
- Do not feed L_i or quiet labels into q.
- MAIN result uses new fit85 only; old_V quiet ratios forbidden.

Artifacts: `C:/Users/todtj/PycharmProjects/LOL_teamfight/outputs/review_response_rr3_quiet_20260920/`

