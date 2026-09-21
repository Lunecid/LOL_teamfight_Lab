# Supplementary E4 — Horizon strata + peer label-transfer

**generated:** 2026-09-21T07:51:32.323141+00:00  
**task:** .ai/tasks/T020.md  
**bundle_sha16:** `ac459cc4397630a9`  

Frozen V/q/PT. Diagnostics only — not new journal headlines.

## Incomplete (§7.3 e_fixed / §7.4 definition OAT)

- **e_fixed:** `INCOMPLETE`
- **definition OAT:** `INCOMPLETE`

Match packs required for e_fixed StateBuilder rebuild and engagement-definition OAT are unavailable (CACHE_DIR files=0). Leave INCOMPLETE rather than invent endpoints.

## Horizon strata — cohort T (§7.1 / §7.3)

n_common=32981 / 24020 matches; Y_main vs recomputed Y_h90 agree=1.000000; |ΔV| cut=0.05.

| Scope | n | flip 60–90 | flip 90–120 | same-ep 60–90 | flip|same-ep 60–90 | flip|diff-ep 60–90 |
|---|---:|---:|---:|---:|---:|---:|
| all | 32981 | 0.0151 | 0.0065 | 0.594 | 0.0000 | 0.0373 |
| B40 | 5423 | 0.0138 | 0.0057 | 0.553 | 0.0000 | 0.0310 |
| small_abs_dV | 13151 | 0.0314 | 0.0129 | 0.652 | 0.0000 | 0.0904 |
| same_frame | 2224 | 0.0000 | 0.0000 | 1.000 | 0.0000 | nan |
| new_frame | 30757 | 0.0162 | 0.0070 | 0.565 | 0.0000 | 0.0373 |
| t_2_10 | 2265 | 0.0146 | 0.0079 | 0.611 | 0.0000 | 0.0375 |
| t_10_20 | 11858 | 0.0195 | 0.0076 | 0.557 | 0.0000 | 0.0439 |
| t_20_30 | 15446 | 0.0127 | 0.0056 | 0.588 | 0.0000 | 0.0308 |
| t_30_inf | 3412 | 0.0114 | 0.0059 | 0.744 | 0.0000 | 0.0446 |

### Frozen q/PT label-transfer vs Y_h60 (compact)

| Scope | n | flip vs Y_main | ΔBrier q−PT on main | ΔBrier q−PT on alt |
|---|---:|---:|---:|---:|
| all (alt=Y_h60) | 32981 | 0.0151 | -0.00373 | -0.00368 |
| B40 (alt=Y_h60) | 5423 | 0.0138 | -0.00214 | -0.00219 |
| small_abs_dV (alt=Y_h60) | 13151 | 0.0314 | -0.00136 | -0.00107 |
| same_frame (alt=Y_h60) | 2224 | 0.0000 | -0.00084 | -0.00084 |
| new_frame (alt=Y_h60) | 30757 | 0.0162 | -0.00390 | -0.00384 |
| t_2_10 (alt=Y_h60) | 2265 | 0.0146 | -0.00285 | -0.00219 |
| t_10_20 (alt=Y_h60) | 11858 | 0.0195 | -0.00525 | -0.00545 |
| t_20_30 (alt=Y_h60) | 15446 | 0.0127 | -0.00222 | -0.00215 |
| t_30_inf (alt=Y_h60) | 3412 | 0.0114 | -0.00388 | -0.00360 |

Horizon flips and frozen-score label transfer are diagnostics; not claims that changing the horizon redesigns the method. Separate same-endpoint vs diff-endpoint flip rates (§7.3).

## Horizon strata — cohort S (§7.1 / §7.3)

n_common=101205 / 49730 matches; Y_main vs recomputed Y_h90 agree=1.000000; |ΔV| cut=0.05.

| Scope | n | flip 60–90 | flip 90–120 | same-ep 60–90 | flip|same-ep 60–90 | flip|diff-ep 60–90 |
|---|---:|---:|---:|---:|---:|---:|
| all | 101205 | 0.0180 | 0.0063 | 0.684 | 0.0000 | 0.0570 |
| B40 | 31675 | 0.0180 | 0.0058 | 0.672 | 0.0000 | 0.0548 |
| small_abs_dV | 40468 | 0.0367 | 0.0129 | 0.698 | 0.0000 | 0.1217 |
| same_frame | 14026 | 0.0000 | 0.0000 | 1.000 | 0.0000 | nan |
| new_frame | 87179 | 0.0209 | 0.0073 | 0.633 | 0.0000 | 0.0570 |
| t_2_10 | 50444 | 0.0196 | 0.0073 | 0.678 | 0.0000 | 0.0611 |
| t_10_20 | 33415 | 0.0171 | 0.0056 | 0.696 | 0.0000 | 0.0562 |
| t_20_30 | 14872 | 0.0157 | 0.0048 | 0.667 | 0.0000 | 0.0471 |
| t_30_inf | 2474 | 0.0125 | 0.0061 | 0.722 | 0.0000 | 0.0450 |

### Frozen q/PT label-transfer vs Y_h60 (compact)

| Scope | n | flip vs Y_main | ΔBrier q−PT on main | ΔBrier q−PT on alt |
|---|---:|---:|---:|---:|
| all (alt=Y_h60) | 101205 | 0.0180 | -0.00335 | -0.00326 |
| B40 (alt=Y_h60) | 31675 | 0.0180 | -0.00210 | -0.00185 |
| small_abs_dV (alt=Y_h60) | 40468 | 0.0367 | -0.00028 | -0.00021 |
| same_frame (alt=Y_h60) | 14026 | 0.0000 | -0.00120 | -0.00120 |
| new_frame (alt=Y_h60) | 87179 | 0.0209 | -0.00351 | -0.00339 |
| t_2_10 (alt=Y_h60) | 50444 | 0.0196 | -0.00306 | -0.00288 |
| t_10_20 (alt=Y_h60) | 33415 | 0.0171 | -0.00423 | -0.00422 |
| t_20_30 (alt=Y_h60) | 14872 | 0.0157 | -0.00106 | -0.00102 |
| t_30_inf (alt=Y_h60) | 2474 | 0.0125 | 0.00082 | 0.00084 |

Horizon flips and frozen-score label transfer are diagnostics; not claims that changing the horizon redesigns the method. Separate same-endpoint vs diff-endpoint flip rates (§7.3).

## Peer LR label-transfer — T (§7.2)

peer=`A_LR_expanded_fit85`; n=32981; sign_agree_nonzero=0.9135813699627036; label_flip_all=0.0865.

Frozen q/PT predictions scored against Y_peer; p_pre in q inputs was NOT replaced by peer. This is label-transfer diagnostics, not a peer-trained pipeline.

| Scope | n | flip vs Y_main | ΔBrier q−PT on main | ΔBrier q−PT on alt |
|---|---:|---:|---:|---:|
| all | 32981 | 0.0865 | -0.00373 | -0.00125 |
| B40 | 5423 | 0.0725 | -0.00214 | -0.00067 |
| small_abs_dV | 13151 | 0.1865 | -0.00136 | 0.00355 |
| same_frame | 2224 | 0.0603 | -0.00084 | -0.00107 |
| new_frame | 30757 | 0.0884 | -0.00390 | -0.00118 |
| t_2_10 | 2265 | 0.1055 | -0.00285 | -0.00287 |
| t_10_20 | 11858 | 0.0931 | -0.00525 | -0.00412 |
| t_20_30 | 15446 | 0.0805 | -0.00222 | 0.00129 |
| t_30_inf | 3412 | 0.0777 | -0.00388 | 0.00186 |

## Forbidden

- refit q/PT on peer labels in this task
- replace q input p_pre with peer V
- retune V for peer agreement
- fabricate e_fixed/OAT without match packs
- new journal headlines

