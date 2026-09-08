# Data manifest — what the paper cites, where it lives, how to regenerate

Status: canonical = cited by the ToG draft; superseded = kept for audit only;
regenerable = safe to delete, one command rebuilds it.

## Corpora (irreplaceable unless noted)

| corpus | location | content | status |
|---|---|---|---|
| Main (15.14-15.16) | `D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13` | 205,884 matches, SOLE COPY (raw no longer exists) | canonical - BACK UP |
| Replication (16.13+16.15) | `D:/LOL_Project/fusion_2615/cache/match_cache_fresh_v3_engage_status13` | 10,612+ matches cached from `data/raw/2026_current/kr` (raw retained) | canonical |
| Replays 16.15 | `D:/LOL_Project/data/replays/2026_current/kr/rofl` | 1,053 ROFL (vision axis, playable only on 16.15) | frozen asset |
| Vision windows | `D:/LOL_Project/fusion_2615/vision_windows` | 1,333 pre-fight capture windows | frozen asset |

## Canonical results (all under `D:/LOL_Project/fusion_2615/features/`)

| file | claim it backs | producer |
|---|---|---|
| `scale_decomposition_mlex.json` (+preds) | headline: 948,369 eng, overall .7458, tf .7842, pick-tf -.0677 | `run_scale_decomposition.py` on `corpus_shards_mlex` |
| `scale_decomposition_2026_mlex.json` | replication: 47,759 eng/10,612 matches, pick-tf -.0652 | same on `corpus_shards_2026_mlex` |
| `deep_mlex_fast.json` / `deep_mlex_tokens.json` | fair DL table (LGBM .7311 > FT .7193 > SAINT .7153 > MLP .7016 > TabNet .6683) | `run_deep_tabular_baselines.py` |
| `shap_mlex.json` | attribution (tower distance top) | `run_shap_attribution.py` |
| `label_ablation_pilot.json` (+preds), `label_weight_sensitivity.json` | R2 label defense | `run_label_ablation.py`, `run_label_weight_sensitivity.py` |
| `thresholds/*.json` | R1 threshold sweep (AUC spread .008) | `run_threshold_sensitivity.py` |
| `killless/*.json` | meta-review bound (4.9% at teamfight scale) | `run_killless_encounters.py` |
| `shop_event_sensitivity.json` | 1.6% contamination bound | `run_shop_event_sensitivity.py` |
| `corpus_control_original/2026.json` | matched-size control (.595 vs .571) | `run_corpus_control.py` |
| `vision_ablation_causal.json`, `spatial_control_causal.json`, `causal_stratified_w{10,20,30}.json`, `time_confound_check.json` | vision-as-instrument section | respective scripts |

Definitions: `docs/ENGAGEMENT_SCALE_DEFINITION.md`, `docs/ENGAGEMENT_WINNER_DEFINITION.md`,
`docs/CONSTANTS_JUSTIFICATION.md`. Plan: `docs/TOG_EXTENSION_PLAN.md`.

## Superseded (audit trail only - do NOT cite)

Eq.3-label runs: `scale_decomposition.json`, `scale_decomposition_1615_mlex.json`
(534-match axis, replaced by 2026 axis), `deep_tabular_baselines.json` +
`deep_tabular_tabnet_saint.json` (Eq.3 label), `stratified_results_v2.json`
(double-transpose bug - INVALID), v1 524-engagement results, pre-causal
stratified files.

## Regenerable bulk (delete freely when disk is needed)

- `corpus_shards*/merged_matrix.npy` (~19 GB x2 + smaller) - decomposition rebuilds
- `corpus_shards/` (Eq.3 shards, 2.3 GB) - superseded label; rebuild via
  `build_corpus_shard.py` without label override if ever needed
- fight-index caches, `profile_shards`

Regeneration: shards `build_corpus_shard.py --label-type market_lex --tie-policy drop`
(env selects corpus), then `run_scale_decomposition.py`. Seeds fixed (7); all
sampling deterministic.

## Corpus v3 (2026-09-09, definition of `docs/DEFINITION_EVIDENCE.md` section 17)

| file | what | how |
|---|---|---|
| `corpus_shards_v3_mlex/shard_*.npz` (+`v3_definition.json`, `build.log`) | 541,767 engagements x 7,105 features; labels `y` (market_lex, draws dropped), `y_market_event`, `y_attention_value_win` | `scripts/build_corpus_v3.py` (32 shards, `LOL_CFG_OVERRIDES` = G 13.7 s, D 4,264 u, R 1,600 u, B 15 s, H 35 s) |
| `features/scale_decomposition_v3_mlex.json` (+preds, matrix) | ToG protocol on v3, market_lex: overall .7364, pick .709 / skirmish .719 / teamfight(>=4) .805 | `run_scale_decomposition.py --teamfight-min 4` |
| `features/scale_decomposition_v3_market_event.json` (+preds, matrix) | same, event-priced label: .7024, .673 / .686 / .770 | `--y-key y_market_event` |
| `features/scale_decomposition_v3_attention_value_win.json` (+preds, matrix) | same, Eq.3 label: .6815, .657 / .666 / .744 | `--y-key y_attention_value_win` |
| `runs_corpus_v3/v3_G13.7_D4264_R1600_B15/` | CoG protocol on v3 (patch holdout, 100k/split, Eq.3): 574,312 engagements, test AUC .6590 [.6556, .6624] | `runner.py --mode train --models lgbm` with `LOL_CFG_OVERRIDES` |
| `runs_presence_gate/` | presence-gate two-point comparison (1,800 u/10 s vs 1,600 u/15 s at G 18/D 4,000): .6687 vs .6598 | `scripts/run_presence_gate_points.py` |
| `features/prediction_situation_pilot.json` | 553-match pilot: context 15/30/60 s, horizon 35/45/60 s, labels, frame age | `scripts/run_prediction_situation_pilot.py` |

Regenerable: the three `*.matrix.npy` memmaps (~10.6 GB each) are rebuilt by the decomposition from the shards.
