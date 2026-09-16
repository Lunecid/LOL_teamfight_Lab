# Publication tables — objective_channel_ablation_20260915

Exploratory after prior test exposure; named objective-channel ablation, not causal. Values from results/*.json.

## Table 1. W prediction, raw A vs raw B_noobj (equal-match weighted)

| Set | Rows | AUC A | AUC B | Brier A | Brier B | Log loss A | Log loss B |
|---|---|---|---|---|---|---|---|
| MAIN_TEST | 347234 | 0.8586 | 0.8529 | 0.1530 | 0.1561 | 0.4567 | 0.4656 |
| EXT_KR_16.13 | 56908 | 0.8532 | 0.8485 | 0.1570 | 0.1598 | 0.4701 | 0.4787 |
| EXT_KR_16.14_pilot | 1125 | 0.8374 | 0.8374 | 0.1653 | 0.1648 | 0.4892 | 0.4870 |
| EXT_KR_16.15 | 5192 | 0.8567 | 0.8525 | 0.1555 | 0.1581 | 0.4669 | 0.4753 |
| EXT_NA1_16.13 | 59724 | 0.8442 | 0.8385 | 0.1620 | 0.1652 | 0.4842 | 0.4943 |
| MAIN_V_SELECT | 108558 | 0.8583 | 0.8527 | 0.1529 | 0.1560 | 0.4560 | 0.4648 |
| MAIN_V_CAL | 106580 | 0.8577 | 0.8515 | 0.1535 | 0.1568 | 0.4574 | 0.4669 |
| MAIN_Q_CAL | 107348 | 0.8605 | 0.8547 | 0.1519 | 0.1551 | 0.4532 | 0.4624 |
| MAIN_Q_SELECT | 106486 | 0.8613 | 0.8557 | 0.1514 | 0.1545 | 0.4523 | 0.4607 |
| MAIN_TRAIN_heldout_fold | 424160 | 0.8589 | 0.8533 | 0.1527 | 0.1558 | 0.4559 | 0.4647 |

## Table 2. MAIN_TEST bootstrap B minus A (1000 match resamples, fixed models)

| Weighting | Metric | Estimate | 95% interval |
|---|---|---|---|
| match_weighted | brier | 0.00308 | [0.00284, 0.00331] |
| match_weighted | logloss | 0.00887 | [0.00822, 0.00950] |
| row_weighted | brier | 0.00347 | [0.00317, 0.00374] |
| row_weighted | logloss | 0.00986 | [0.00908, 0.01063] |

## Table 3. h90 sign disagreement A vs B_noobj (match-weighted, 95% match bootstrap)

| Set | Cohort | Rows | Matches | Disagreement |
|---|---|---|---|---|
| MAIN_TRAIN | E | 199358 | 69124 | 4.81% [4.71%, 4.92%] |
| MAIN_TRAIN | T | 39605 | 28989 | 5.32% [5.09%, 5.58%] |
| MAIN_TRAIN | N | 159753 | 66545 | 4.79% [4.66%, 4.91%] |
| MAIN_VALIDATION | E | 203170 | 69298 | 4.88% [4.79%, 4.99%] |
| MAIN_VALIDATION | T | 41315 | 30099 | 5.48% [5.25%, 5.73%] |
| MAIN_VALIDATION | N | 161855 | 66695 | 4.82% [4.70%, 4.94%] |
| MAIN_TEST | E | 163576 | 56254 | 4.92% [4.80%, 5.04%] |
| MAIN_TEST | T | 32981 | 24020 | 5.24% [4.98%, 5.48%] |
| MAIN_TEST | N | 130595 | 54182 | 4.88% [4.74%, 5.00%] |
| EXT_KR_16.13 | E | 25430 | 9051 | 5.24% [4.93%, 5.56%] |
| EXT_KR_16.13 | T | 5202 | 3859 | 5.73% [5.09%, 6.41%] |
| EXT_KR_16.13 | N | 20228 | 8642 | 5.23% [4.89%, 5.56%] |
| EXT_KR_16.14_pilot | E | 475 | 177 | 5.49% [3.43%, 7.78%] |
| EXT_KR_16.14_pilot | T | 101 | 80 | 5.62% [1.25%, 10.00%] |
| EXT_KR_16.14_pilot | N | 374 | 165 | 5.24% [3.18%, 7.81%] |
| EXT_KR_16.15 | E | 2219 | 813 | 5.19% [4.22%, 6.36%] |
| EXT_KR_16.15 | T | 507 | 377 | 6.10% [4.11%, 8.31%] |
| EXT_KR_16.15 | N | 1712 | 764 | 5.02% [3.93%, 6.23%] |
| EXT_NA1_16.13 | E | 26538 | 9090 | 5.72% [5.40%, 6.03%] |
| EXT_NA1_16.13 | T | 5312 | 3955 | 6.64% [5.92%, 7.39%] |
| EXT_NA1_16.13 | N | 21226 | 8753 | 5.61% [5.24%, 5.95%] |

## Table 4. MAIN_TEST h90 T single-team acquisitions: acquiring-team-oriented delta (match-weighted; descriptive)

| Window | Objective | Rows | Disagreement | Oriented dA | Oriented dB | A-B |
|---|---|---|---|---|---|---|
| after_(L,e] | baron | 1478 | 5.14% | 0.2345 | 0.1783 | 0.0562 |
| after_(L,e] | elder | 64 | 6.25% | 0.2462 | 0.1121 | 0.1341 |
| after_(L,e] | atakhan | 1771 | 5.25% | 0.1959 | 0.1465 | 0.0495 |
| after_(L,e] | dragon | 5174 | 7.46% | 0.1295 | 0.1012 | 0.0283 |
| after_(L,e] | soul_owned | 473 | 14.80% | 0.1303 | 0.0888 | 0.0416 |
| after_(L,e] | herald | 922 | 4.01% | 0.0884 | 0.0834 | 0.0050 |
| after_(L,e] | horde | 372 | 9.41% | 0.0614 | 0.0411 | 0.0203 |
| full_(q_pre,e] | baron | 2195 | 6.29% | 0.1938 | 0.1452 | 0.0485 |
| full_(q_pre,e] | elder | 106 | 5.66% | 0.2274 | 0.1247 | 0.1026 |
| full_(q_pre,e] | atakhan | 2722 | 8.30% | 0.1686 | 0.1162 | 0.0524 |
| full_(q_pre,e] | dragon | 7662 | 8.08% | 0.1071 | 0.0754 | 0.0317 |
| full_(q_pre,e] | soul_owned | 677 | 14.33% | 0.1164 | 0.0753 | 0.0411 |
| full_(q_pre,e] | herald | 1523 | 4.86% | 0.0626 | 0.0520 | 0.0106 |
| full_(q_pre,e] | horde | 461 | 9.11% | 0.0537 | 0.0340 | 0.0196 |

## Table 5. Frozen specialist q scored against A and B_noobj labels (MAIN_TEST h90)

| Cohort | q | AUC Y_A | AUC Y_B | Brier Y_A | Brier Y_B |
|---|---|---|---|---|---|
| T | spec_ridge_raw | 0.6702 | 0.6791 | 0.2282 | 0.2259 |
| N | spec_ridge_isotonic | 0.5984 | 0.5993 | 0.2427 | 0.2425 |
