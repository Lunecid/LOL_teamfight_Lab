# Supplementary E2 — Frame-refresh strata (§5.4) + S_hold status (§5.2)

**generated:** 2026-09-21T07:43:55.029958+00:00  
**task:** .ai/tasks/T019.md  

## §5.2 Mechanical S_hold probe

**status:** `INCOMPLETE`  

S_hold requires rebuilding StateV2 at endpoint_h90 from events/frames with timestamp ≤ q_pre (state_value_v2.StateBuilder.at), then scoring frozen V. A verified batch rebuild path over TEST was not executed in T019; mutating only time_minutes/time_minutes_sq on X_pre is forbidden by the contract. Leave probe incomplete rather than a partial clock hack.

Next: T020+: scripted StateBuilder hold rebuild on a fixed match subsample, assert d_clock+d_update≈ΔV, then full TEST.

## §5.4 Observation-refresh strata (TEST, frozen scores)

Cells: same-frame / new-frame × pre_frame_age <30s / ≥30s. ΔBrier = q − PT_flex (match-weighted). Diagnostic only.

### Cohort T
joined 32981/32981 (rate 1.000); same-frame 0.067; pre_age<30s 0.370; median pre_age 37.0s; median followup 74.0s

| Cell | n | matches | P(SVI+) | E|ΔV| | ΔBrier(q−PT) | mean p_pre | mean t_min |
|---|---:|---:|---:|---:|---:|---:|---:|
| same_young | 2204 | 2164 | 0.530 | 0.0942 | -0.00090 | 0.497 | 22.94 |
| same_old | 20 | 20 | 0.450 | 0.0566 | +0.00427 | 0.498 | 16.63 |
| new_young | 10003 | 8979 | 0.486 | 0.1186 | -0.00347 | 0.504 | 21.18 |
| new_old | 20754 | 16923 | 0.497 | 0.1161 | -0.00398 | 0.506 | 21.13 |
| all | 32981 | 24020 | 0.496 | 0.1154 | -0.00373 | 0.505 | 21.26 |

Cell differences are diagnostic of composition under frozen scores; not causal claims that frame refresh caused performance.

### Cohort S
joined 101205/101205 (rate 1.000); same-frame 0.139; pre_age<30s 0.437; median pre_age 33.3s; median followup 60.1s

| Cell | n | matches | P(SVI+) | E|ΔV| | ΔBrier(q−PT) | mean p_pre | mean t_min |
|---|---:|---:|---:|---:|---:|---:|---:|
| same_young | 12912 | 11712 | 0.569 | 0.0783 | -0.00141 | 0.498 | 10.72 |
| same_old | 1114 | 1101 | 0.590 | 0.0772 | +0.00007 | 0.504 | 9.24 |
| new_young | 31314 | 24695 | 0.493 | 0.0906 | -0.00396 | 0.501 | 11.09 |
| new_old | 55865 | 36561 | 0.503 | 0.0912 | -0.00310 | 0.506 | 12.53 |
| all | 101205 | 49730 | 0.509 | 0.0892 | -0.00335 | 0.504 | 11.82 |

Cell differences are diagnostic of composition under frozen scores; not causal claims that frame refresh caused performance.

## Forbidden

- causal frame-refresh effect
- put endpoint/frame-refresh into q inputs
- time-only X_pre mutation as S_hold
- replace journal headlines
