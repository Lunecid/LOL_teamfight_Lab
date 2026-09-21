# Scale-split results — teamfight T vs skirmish S under frozen fit85 V

**role_tag:** `EXPLORATORY_SCALE_SPLIT_PRIOR_TEST_EXPOSURE`  
**contract:** [SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md](SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md)  
**source_commit:** `78695a6d5924e6e8755599ee89873bafd993ed36`  
**generated:** 2026-09-21T01:11:39+09:00  

Main reporting uses **identity calibrator** (contract §4 literal). T009 RR12 two-stage selection values are sensitivity only.
Main-contrast sign identical across variants: **True** (identity -0.00335; sensitivity -0.00346).

T headline numbers are **cited from frozen** RR12/RRX artifacts.

## Table 1 — S TEST point metrics (identity; main)

| Model | n | matches | Brier | logloss | AUC |
|---|---:|---:|---:|---:|---:|
| q_S | 101205 | 49730 | 0.2458 | 0.6849 | 0.5767 |
| q_T→S | 101205 | 49730 | 0.2478 | 0.6890 | 0.5625 |
| q_TS | 101205 | 49730 | 0.2460 | 0.6854 | 0.5757 |
| PT_flex_S | 101205 | 49730 | 0.2492 | 0.6915 | 0.5360 |
| PT_linear_S | 101205 | 49730 | 0.2495 | 0.6921 | 0.5301 |
| b(p)_spline_S | 101205 | 49730 | 0.2494 | 0.6919 | 0.5301 |
| b(p)_linear_S | 101205 | 49730 | 0.2495 | 0.6921 | 0.5301 |
| lgbm_state_S (diagnostic) | 101205 | 49730 | 0.2475 | 0.6880 | 0.5560 |

Source: `paired_contrasts_id.json` → `table1_S_TEST`; lgbm diagnostic from S fit `TEST.lgbm_state` (n=101205).

## Table 1b — S TEST (RR12 2-stage selection; sensitivity)

| Model | n | matches | Brier | logloss | AUC |
|---|---:|---:|---:|---:|---:|
| q_S | 101205 | 49730 | 0.2454 | 0.6840 | 0.5767 |
| q_T→S | 101205 | 49730 | 0.2468 | 0.6868 | 0.5625 |
| q_TS | 101205 | 49730 | 0.2456 | 0.6842 | 0.5757 |
| PT_flex_S | 101205 | 49730 | 0.2489 | 0.6910 | 0.5360 |
| PT_linear_S | 101205 | 49730 | 0.2492 | 0.6915 | 0.5301 |
| b(p)_spline_S | 101205 | 49730 | 0.2491 | 0.6914 | 0.5301 |
| b(p)_linear_S | 101205 | 49730 | 0.2492 | 0.6915 | 0.5301 |

Source: `paired_contrasts.json` → `table1_S_TEST`.

## Table 2a — Contrasts, contract-literal identity

| Contrast | role | estimate | CI95 | p_gt0 | n | matches | seed |
|---|---|---:|---|---:|---:|---:|---:|
| q_S − PT_flex_S (S TEST) | **primary** | -0.00335 | [-0.00379, -0.00288] | 0.0000 | 101205 | 49730 | 7 |
| q_S − q_T→S (S TEST) | secondary | -0.00194 | [-0.00236, -0.00153] | 0.0000 | 101205 | 49730 | 7 |
| q_S − q_TS (S TEST) | secondary | -0.00024 | [-0.00039, -0.00009] | 0.0005 | 101205 | 49730 | 7 |
| q_TS − q_T (T TEST) | secondary | 0.00071 | [0.00018, 0.00126] | 0.9940 | 32981 | 24020 | 7 |
| q_S − PT_flex_S (S∩B40) | secondary | -0.00210 | [-0.00266, -0.00149] | 0.0000 | 31675 | 25039 | 7 |

Source: `paired_contrasts_id.json` → `contrasts` (2000 draws, seed 7, w=1/n_m).

## Table 2b — Contrasts, RR12 2-stage selection (sensitivity)

| Contrast | role | estimate | CI95 | p_gt0 | n | matches | seed |
|---|---|---:|---|---:|---:|---:|---:|
| q_S − PT_flex_S (S TEST) | sensitivity | -0.00346 | [-0.00382, -0.00307] | 0.0000 | 101205 | 49730 | 7 |
| q_S − q_T→S (S TEST) | secondary | -0.00138 | [-0.00168, -0.00108] | 0.0000 | 101205 | 49730 | 7 |
| q_S − q_TS (S TEST) | secondary | -0.00012 | [-0.00024, 0.00001] | 0.0345 | 101205 | 49730 | 7 |
| q_TS − q_T (T TEST) | secondary | 0.00071 | [0.00018, 0.00126] | 0.9940 | 32981 | 24020 | 7 |
| q_S − PT_flex_S (S∩B40) | secondary | -0.00210 | [-0.00256, -0.00160] | 0.0000 | 31675 | 25039 | 7 |

Source: `paired_contrasts.json` → `contrasts`.

B40 cell is a **within-S** secondary contrast only (S TEST B40 n=31675 / 101205 ≈ 31.3%; not placed beside T B40).

## Table 3 — CORP point values (S TEST; no intervals)

| Variant | Model | n | MCB | DSC | UNC |
|---|---|---:|---:|---:|---:|
| identity | q_S | 101205 | 0.0008 | 0.0048 | 0.2499 |
| identity | PT_flex_S | 101205 | 0.0007 | 0.0014 | 0.2499 |
| sensitivity | q_S | 101205 | 0.0004 | 0.0048 | 0.2499 |
| sensitivity | PT_flex_S | 101205 | 0.0004 | 0.0014 | 0.2499 |

Source: `paired_contrasts_id.json` / `paired_contrasts.json` → `CORP`.

## Table 4 — EXT S (score-only; no CI; small cohorts reported, not pooled)

| Cohort | n | V_pre Brier | ΔBrier(q_S − PT_flex_S) | ΔMCB | ΔDSC |
|---|---:|---:|---:|---:|---:|
| KR 16.13 | 15641 | 0.1872 | -0.0018 | 0.0011 | 0.0029 |
| NA1 16.13 | 16100 | 0.1871 | -0.0020 | 0.0013 | 0.0033 |
| KR 16.15 | 1307 | 0.1829 | -0.0033 | 0.0014 | 0.0046 |
| KR 16.14 pilot | 285 | 0.1787 | 0.0035 | 0.0091 | 0.0056 |

Source: `rrx_external_results.json` → `cohorts` (unchanged from T009; raw).
Census vs contract §2: KR 15641 / NA1 16100 / KR16.15 1307 / pilot 285.

## Allowed / Forbidden readings

Copied from contract §6 (Forbidden readings):

- Comparing S and T absolute Brier/AUC as an "improvement", a "scale gradient", or "which engagements are more predictable".
- Any mechanism story for a T/S difference (claim ledger X-31 stays withdrawn).
- Promoting a secondary contrast, a λ filter, a bin or a learner because of its TEST value.
- Substituting all-N or pick results for S, or pooling the small external cohorts into a success claim.
- Touching the frozen T artifacts, the journal manuscript, or any lock document before the results report is reviewed.

### Forbidden (quoted phrases kept only in this section)

> scale gradient; more predictable; routing; deploy

## Suggestions (not executed)

- Whether a shared PT_flex object should be frozen once on S Q_SELECT and reused for all S arms.
- Whether EXT S ΔBrier sign vs T EXT should be discussed only after a predeclared transfer protocol.

