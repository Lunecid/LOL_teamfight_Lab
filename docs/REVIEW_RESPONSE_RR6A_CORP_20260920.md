# RR6a — CORP calibration diagnostics (V→W and q→SVI)

Generated: 2026-09-20T15:55:44+09:00
**Bundle:** `outputs/v_redesign_wave4_corrected_20260919/evaluators/A_MLP_expanded_evaluator.joblib` sha16=`ac459cc4397630a9`

Diagnostic only (Dimitriadis–Gneiting–Jordan CORP + Gneiting–Raftery scoring). **Do not** treat TEST-fit isotonic as a frozen-model upgrade.

## Summary table

| Stage | Cell | n | Brier | MCB | DSC | UNC | ECE | AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V_to_W_timeline | overall | 347234 | 0.1552 | 0.0002 | 0.0948 | 0.2499 | 0.0057 | 0.8542 |
| V_to_W_timeline | t_2_10 | 120186 | 0.2283 | 0.0008 | 0.0223 | 0.2499 | 0.0198 | 0.6642 |
| V_to_W_timeline | t_10_20 | 119488 | 0.1544 | 0.0009 | 0.0964 | 0.2499 | 0.0214 | 0.8580 |
| V_to_W_timeline | t_20_30 | 85342 | 0.0891 | 0.0003 | 0.1611 | 0.2500 | 0.0054 | 0.9496 |
| V_to_W_timeline | t_30_inf | 22218 | 0.0856 | 0.0021 | 0.1664 | 0.2499 | 0.0358 | 0.9544 |
| V_to_W_timeline | overall_t_ge_2 | 347234 | 0.1552 | 0.0002 | 0.0948 | 0.2499 | 0.0057 | 0.8542 |
| V_to_W_eng_pre | overall | 32981 | 0.1442 | 0.0007 | 0.1065 | 0.2500 | 0.0101 | 0.8750 |
| V_to_W_eng_post | overall | 32981 | 0.1181 | 0.0005 | 0.1324 | 0.2500 | 0.0044 | 0.9149 |
| V_to_W_eng_pre | t_2_10 | 2265 | 0.2159 | 0.0038 | 0.0378 | 0.2499 | 0.0176 | 0.7114 |
| V_to_W_eng_pre | t_10_20 | 11858 | 0.1668 | 0.0018 | 0.0849 | 0.2498 | 0.0261 | 0.8345 |
| V_to_W_eng_pre | t_20_30 | 15446 | 0.1270 | 0.0012 | 0.1242 | 0.2500 | 0.0135 | 0.9024 |
| V_to_W_eng_pre | t_30_inf | 3412 | 0.1544 | 0.0030 | 0.0981 | 0.2496 | 0.0189 | 0.8569 |
| V_to_W_eng_pre | overall_t_ge_2 | 32981 | 0.1442 | 0.0007 | 0.1065 | 0.2500 | 0.0101 | 0.8750 |
| V_to_W_eng_post | t_2_10 | 1633 | 0.2057 | 0.0049 | 0.0492 | 0.2500 | 0.0275 | 0.7423 |
| V_to_W_eng_post | t_10_20 | 9876 | 0.1521 | 0.0015 | 0.0991 | 0.2497 | 0.0185 | 0.8612 |
| V_to_W_eng_post | t_20_30 | 16961 | 0.1046 | 0.0008 | 0.1462 | 0.2500 | 0.0076 | 0.9323 |
| V_to_W_eng_post | t_30_inf | 4511 | 0.1057 | 0.0026 | 0.1466 | 0.2497 | 0.0276 | 0.9324 |
| V_to_W_eng_post | overall_t_ge_2 | 32981 | 0.1181 | 0.0005 | 0.1324 | 0.2500 | 0.0044 | 0.9149 |
| q_to_SVI | q_all_T | 32981 | 0.2355 | 0.0009 | 0.0154 | 0.2500 | 0.0133 | 0.6403 |
| q_to_SVI | PT_flex_all_T | 32981 | 0.2392 | 0.0005 | 0.0113 | 0.2500 | 0.0063 | 0.6200 |
| q_to_SVI | q_B40 | 5423 | 0.2474 | 0.0018 | 0.0044 | 0.2500 | 0.0167 | 0.5669 |
| q_to_SVI | PT_flex_B40 | 5423 | 0.2495 | 0.0006 | 0.0010 | 0.2500 | 0.0028 | 0.5248 |
| q_to_SVI | q_outside_B40 | 27558 | 0.2342 | 0.0010 | 0.0168 | 0.2500 | 0.0165 | 0.6465 |
| q_to_SVI | PT_flex_outside_B40 | 27558 | 0.2382 | 0.0006 | 0.0124 | 0.2500 | 0.0080 | 0.6255 |
| q_to_SVI | q_t_2_10 | 2265 | 0.2467 | 0.0023 | 0.0053 | 0.2497 | 0.0182 | 0.5589 |
| q_to_SVI | q_t_10_20 | 11858 | 0.2418 | 0.0009 | 0.0091 | 0.2500 | 0.0105 | 0.6045 |
| q_to_SVI | q_t_20_30 | 15446 | 0.2327 | 0.0015 | 0.0188 | 0.2500 | 0.0230 | 0.6540 |
| q_to_SVI | q_t_30_inf | 3412 | 0.2345 | 0.0047 | 0.0195 | 0.2492 | 0.0411 | 0.6528 |
| q_to_SVI | q_overall_t_ge_2 | 32981 | 0.2355 | 0.0009 | 0.0154 | 0.2500 | 0.0133 | 0.6403 |
| q_to_SVI | PT_flex_t_2_10 | 2265 | 0.2495 | 0.0018 | 0.0019 | 0.2497 | 0.0039 | 0.5108 |
| q_to_SVI | PT_flex_t_10_20 | 11858 | 0.2470 | 0.0009 | 0.0038 | 0.2500 | 0.0109 | 0.5645 |
| q_to_SVI | PT_flex_t_20_30 | 15446 | 0.2349 | 0.0008 | 0.0159 | 0.2500 | 0.0062 | 0.6398 |
| q_to_SVI | PT_flex_t_30_inf | 3412 | 0.2384 | 0.0042 | 0.0150 | 0.2492 | 0.0339 | 0.6277 |
| q_to_SVI | PT_flex_overall_t_ge_2 | 32981 | 0.2392 | 0.0005 | 0.0113 | 0.2500 | 0.0063 | 0.6200 |

## CORP definition (corrected)

MCB \(=\) BS − BS\(_\mathrm{iso}\), DSC \(=\) UNC − BS\(_\mathrm{iso}\). Exact identity BS = MCB − DSC + UNC (asserted in code). Do **not** use \(\langle(p-p^*)^2\rangle\) as MCB.

## Reading

- Report **relative** MCB vs DSC contributions; do not claim “calibration is fine” from small absolute MCB alone (ΔBrier vs PT is \(O(10^{-3})\), so MCB differences of that order matter).
- 15.16 MAIN diagnostics do **not** explain 16.x external transfer; RRX must score V→W and q→SVI externally.
- V→W and q→SVI are **separate** stages — never pooled into one reliability diagram.
- Artifacts: `outputs/review_response_rr6a_corp_20260920/`
