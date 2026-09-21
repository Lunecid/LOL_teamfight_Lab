# RR5 + RR6b — material correspondence, next objective, horizon stability

Generated: 2026-09-20T17:31:09+09:00
**Bundle sha16:** `ac459cc4397630a9`

## RR5a — material correspondence (not independent accuracy)

Correspondence with same-window material nets — not independent fight-winner accuracy. Material features overlap V inputs.

- alive: alive_diff_post = sum(alive)_blue − sum(alive)_red at post snapshot (last frame / state builder). same_frame share=0.067 (pre_snapshot==post_snapshot_h90).

| Axis | scope | n_decided | tie share | agree (match-wtd) | disagree (UW decided) |
|---|---|---:|---:|---:|---:|
| kill_diff | all_T | 28711 | 0.129 | 0.904 | 0.097 |
| kill_diff | B40 | 4653 | 0.142 | 0.928 | 0.074 |
| epic_net | all_T | 12507 | 0.621 | 0.818 | 0.182 |
| epic_net | B40 | 2221 | 0.590 | 0.842 | 0.158 |
| structure_net | all_T | 16106 | 0.512 | 0.774 | 0.228 |
| structure_net | B40 | 2014 | 0.629 | 0.812 | 0.190 |
| objective_net | all_T | 21779 | 0.340 | 0.810 | 0.191 |
| objective_net | B40 | 3099 | 0.429 | 0.860 | 0.141 |
| alive_diff_post | all_T | 14584 | 0.558 | 0.794 | 0.208 |
| alive_diff_post | B40 | 2142 | 0.605 | 0.788 | 0.212 |

## RR5b — first elite objective in (endpoint, endpoint+180s]

Eligible: `('BARON_NASHOR', 'DRAGON', 'RIFTHERALD', 'HORDE', 'ATAKHAN')`. Labels ignore V/q. none ≠ Red.

Denominator notes: Blue/Red rates use *decided* cases only. Incomplete 180s follow-up is **not** all missing data — split `game_ended_before_objective` / `game_ended_in_window_no_objective` / `observation_censored` (cache miss or unknown end). Their sum = cases without an established full 180s window.

| Scope | none | Blue | Red | tie | ended_before | ended_in_win | censored | N | decided |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_T | 4649 | 10076 | 10135 | 9 | 0 | 8112 | 0 | 32981 | 20211 |
| B40 | 1061 | 1894 | 1869 | 2 | 0 | 597 | 0 | 5423 | 3763 |
| SVI_pos | 2370 | 6004 | 3930 | 3 | 0 | 4051 | 0 | 16358 | 9934 |
| SVI_neg | 2279 | 4072 | 6205 | 6 | 0 | 4061 | 0 | 16623 | 10277 |

- Among decided (Blue|Red): SVI+ Blue rate=0.604; SVI− Blue rate=0.396. **Not** q accuracy — measured SVI vs post-endpoint objective.

Convergent correspondence only — not causal fight effect; shares common causes with state. Do not convert none→Red. Labels ignore V/q. game_ended_* vs observation_censored: incomplete 180s follow-up is not all missing data.

## RR6b — horizon stability + endpoint identity

High sign agree can reflect shared endpoints (short remaining games). Report same-endpoint share alongside flip rates. Not a retune target.

- common valid n=32,981; inclusion `{"valid_h60": 32981, "valid_h90": 32981, "valid_h120": 32981}`

| Pair | SVI flip | sign agree | corr(ΔV) | same endpoint share | flip\|same-ep | flip\|diff-ep |
|---|---:|---:|---:|---:|---:|---:|
| h60_vs_h90 | 0.015 | 0.985 | 0.9918 | 0.594 | 0.000 | 0.037 |
| h90_vs_h120 | 0.007 | 0.993 | 0.9959 | 0.810 | 0.000 | 0.034 |
| h60_vs_h120 | 0.019 | 0.981 | 0.9877 | 0.594 | 0.000 | 0.048 |

### Peer sign agree (reused from newv verify)

- MLP_vs_A0_wave2: 0.915
- MLP_vs_A_LR_expanded_fit85: 0.914

Artifacts: `outputs/review_response_rr5_rr6b_20260920/`
