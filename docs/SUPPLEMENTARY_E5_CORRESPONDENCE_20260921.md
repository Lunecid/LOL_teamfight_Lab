# Supplementary E5 — Same-case correspondence (§8.1) + nextobj association

**generated:** 2026-09-21T07:56:40.969965+00:00  
**task:** .ai/tasks/T021.md  

Diagnostics only. Not a claim that SVI is a better value metric than CoG labels.

## Incomplete

- **CoG_market_event:** `INCOMPLETE` — market_event needs event packs (CACHE_DIR files=0).
- **R0_R1_outcome_model:** `INCOMPLETE` — Multinomial R0/R1 requires TRAIN nextobj labels with OOF ΔV; only TEST nextobj npz exists. Association tables use TEST labels only.
- **event_prefix_trace:** `INCOMPLETE` — Match packs unavailable for query→event-prefix dumps in hash samples.

## Cohort T

n=32981 / 24020 matches; same_frame=0.067.

### Label functions

- `SVI`: 1[delta_V>0] from frozen fit85 A_MLP_expanded; delta_V==0 → Y=0; missing excluded upstream
- `kill_diff`: sum(kills)_blue−sum(kills)_red at post_h90 minus same at pre (state slots)
- `alive_diff_post`: sum(alive)_blue−sum(alive)_red at post_h90
- `objective_net`: during+after_h90 elite+structure count nets (blue−red)
- `CoG_market_event`: INCOMPLETE — requires match event packs; not approximated

### Material crosstabs (match-weighted agree on decided)

| Axis | scope | n_decided | tie_share | agree_w | disagree_uw |
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

### Kill–SVI disagreement × objective

n_disagree=2780 / kill_decided=28711; obj agrees SVI=691, obj agrees kill=1079, obj tie=1010.

Same-engagement correspondence only. Material features overlap V inputs. Do not call this independent fight-winner accuracy or causal effect.

### Next elite association (TEST labels, selected Blue|Red)

Association of frozen SVI/kill-net with first elite in (endpoint,+180s]. Not q accuracy; not causal ATT. Reuses RR5b labels — no new cache scan.

| Scope | n | n_BlueRed | SVI_agree | kill_agree (kill-decided) |
|---|---:|---:|---:|---:|
| all_T | 32981 | 20211 | 0.604 | 0.670 |
| B40 | 5423 | 3763 | 0.610 | 0.638 |
| SVI_pos | 16358 | 9934 | 0.604 | 0.664 |
| SVI_neg | 16623 | 10277 | 0.604 | 0.676 |

## Cohort S

n=101205 / 49730 matches; same_frame=0.139.

### Label functions

- `SVI`: 1[delta_V>0] from frozen fit85 A_MLP_expanded; delta_V==0 → Y=0; missing excluded upstream
- `kill_diff`: sum(kills)_blue−sum(kills)_red at post_h90 minus same at pre (state slots)
- `alive_diff_post`: sum(alive)_blue−sum(alive)_red at post_h90
- `objective_net`: during+after_h90 elite+structure count nets (blue−red)
- `CoG_market_event`: INCOMPLETE — requires match event packs; not approximated

### Material crosstabs (match-weighted agree on decided)

| Axis | scope | n_decided | tie_share | agree_w | disagree_uw |
|---|---|---:|---:|---:|---:|
| kill_diff | all_T | 83667 | 0.173 | 0.894 | 0.108 |
| kill_diff | B40 | 25574 | 0.193 | 0.923 | 0.080 |
| epic_net | all_T | 23599 | 0.767 | 0.715 | 0.284 |
| epic_net | B40 | 4761 | 0.850 | 0.714 | 0.285 |
| structure_net | all_T | 20502 | 0.797 | 0.684 | 0.317 |
| structure_net | B40 | 2508 | 0.921 | 0.719 | 0.281 |
| objective_net | all_T | 35907 | 0.645 | 0.709 | 0.292 |
| objective_net | B40 | 6060 | 0.809 | 0.726 | 0.273 |
| alive_diff_post | all_T | 34992 | 0.654 | 0.759 | 0.243 |
| alive_diff_post | B40 | 8345 | 0.737 | 0.782 | 0.219 |

### Kill–SVI disagreement × objective

n_disagree=9027 / kill_decided=83667; obj agrees SVI=2200, obj agrees kill=1912, obj tie=4915.

Same-engagement correspondence only. Material features overlap V inputs. Do not call this independent fight-winner accuracy or causal effect.

## Hash samples

Pre-registered cells; sha256(match|s) order; ≤8/cell. See `SUPPLEMENTARY_E5_DISAGREE_SAMPLES_20260921.csv`.

## Forbidden

- SVI is a better value metric than CoG
- independent fight-winner accuracy
- causal ATT from nextobj association
- fabricate market_event without packs
- train R0/R1 on TEST-only as confirmatory
- new journal headlines

