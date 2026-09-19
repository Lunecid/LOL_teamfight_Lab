# LOL Teamfight Lab

> **Current research (2026-09):** Choice A `shared_lgbm` provisional (\(V=g\\circ f\)); pack [`docs/WINPROB_V_DESIGN_PACK_20260919.md`](docs/WINPROB_V_DESIGN_PACK_20260919.md) + band ledger [`docs/BAND_LEDGER_SHARED_LGBM_20260919.md`](docs/BAND_LEDGER_SHARED_LGBM_20260919.md). V-4 = partial warning only.  
> Authority: [`docs/V_REDESIGN_CONTRACT_20260919.md`](docs/V_REDESIGN_CONTRACT_20260919.md), [`docs/COG_SUCCESSION_LOCK_20260919.md`](docs/COG_SUCCESSION_LOCK_20260919.md), [`docs/COMMON_RESEARCH_SPINE_20260919.md`](docs/COMMON_RESEARCH_SPINE_20260919.md).  
> Legacy sealed \(q\)/SVI (**old V** only): [`docs/SVI_EVIDENCE_CITE_SHEET_20260919.md`](docs/SVI_EVIDENCE_CITE_SHEET_20260919.md). Scripts: `scripts/rr20260919_*.py`.

---

## Legacy lineage below (v3.3 `market_event`)

The following sections describe the **prior** engagement-definition / `market_event` experiment line (≈7k features, 35 s window). Treat them as historical pipeline documentation, **not** as the current SVI / journal–master contract.

---

Research code for predicting which team wins a League of Legends *engagement*, a cluster of
champion kills, from the state of the game 15 s before its first kill. The input is Riot Games
Match-V5 match and timeline data. The repository holds the engagement detector, the pipeline that
estimates the detector's time and distance thresholds from data, the outcome label, feature
extraction, the learners, and the evaluation and audit scripts.

**Current state: engagement definition v3, corpus v3.3 (built 2026-09-09).** This state underlies a
journal extension in preparation. The code state of the IEEE CoG 2026 submission is kept under a tag
(see [History](#history)). Every number on this page is copied from the result file named beside
it; those files are not in the repository (see [Where results live](#where-results-live)). AUC is
the area under the receiver operating characteristic curve: 0.5 is chance, 1.0 is a perfect ranking.

---

## Retraction: "large fights are more predictable"

Drafts written after the CoG submission, on corpora v2 and v3, reported that predictability
decomposes by engagement scale, with the outcomes of large fights more predictable than those of
small ones. The figure quoted was measured on corpus v2 (948,369 engagements in 205,302 matches) with
the `market_lex` label and the teamfight class cut at **at least three** participants on the smaller
side: teamfight AUC 0.7842 against pick AUC 0.7165, a pick − teamfight gap of −0.0677
(`scale_decomposition_mlex.json`, `by_participation_scale` and `bootstrap.pick_minus_teamfight`). The
file records no cut; it is identified by the script default `--teamfight-min 3` of
`scripts/run_scale_decomposition.py`, by a teamfight share of 42.1 %, and by the separate
`teamfight_min4` entry (AUC 0.8084). **That claim is withdrawn.**

- **Cause.** The `time_norm` feature divided the elapsed time by the length of the whole match, so it
  told the model how close the match was to its end. A second, smaller leak placed spatial anchors
  at every tower and objective taken during the match, including those taken after the cutoff
  (`docs/INPUT_FEATURE_AUDIT_V3.md`; `docs/DEFINITION_EVIDENCE.md` section 22).
- **Fix.** Corpus v3.3 divides the elapsed time by 45 min (`TIME_NORM_ABSOLUTE`) and uses only the
  anchors known at the cutoff (`ANCHORS_CAUSAL`; `gameplay/anchors.py`).
- **Size of the leak.** On 5,000 matches (12,603 engagements; same rows and `market_event` label,
  v3.3 detector constants, teamfight class at least four; only the two flags toggled), the
  `time_norm` leak alone raises overall AUC from 0.620 to 0.639, teamfight AUC from 0.642 to 0.691,
  and widens pick − teamfight from −0.024 to −0.070. The anchor leak alone gives overall 0.624 and
  teamfight 0.654 (`leak_ablation_v33.json`, `configs.clean`, `configs.time_leak`,
  `configs.anchor_leak`).
- **What the clean corpus shows.** On corpus v3.3 the gap depends on where the teamfight class is
  cut. Re-scoring the stored out-of-fold predictions of the headline model with 400 match-level
  bootstrap replicates (the 42 engagements whose participation count is stored as −1 are counted as
  picks here):

  | Teamfight class (participants on the smaller side) | pick − teamfight (95 % CI) |
  |---|---|
  | at least 3 | +0.0108 (+0.0070 to +0.0147) |
  | at least 4 (adopted) | −0.0019 (−0.0065 to +0.0028) |
  | at least 5 | −0.0105 (−0.0170 to −0.0041) |

  Source: `tog_revision/A6-definition-sensitivity/scale_cut_sensitivity_v33.json`
  (`participation.cuts.{3,4,5}.gaps.pick_minus_teamfight`), which is being re-verified; the
  adopted-cut value is also in the headline file (below). At the cut of the retracted result (at
  least three) the clean gap is +0.0108: picks are the *more* predictable class, the opposite sign.
  Across the three cuts the gap changes sign. The clean data therefore do not support the retracted
  effect. They do not show that scale has no effect either, because the size and sign of the gap
  follow an arbitrary cut. As a rough sense of scale only, the largest clean gap in absolute value
  (0.0108) is about a sixth of the retracted 0.0677 (0.0677 / 6 = 0.0113). This is not a like-for-like
  test: corpus, label and feature path all differ between the two figures.

---

## What the pipeline does

1. **Detect engagements** (`gameplay/fights.py`). Consecutive champion kills at most *G* apart form a
   chain. A chain is split into groups in which no two kills lie more than *D* apart (complete
   linkage, `gameplay/fight_clustering.py`). A candidate is kept only if, at the cutoff
   τ = first kill − *B*, at least *M* champions of each team are alive within *R* of the first kill.
   The clean-up rules of the CoG detector are unchanged in form; with the v3.3 constants they are:
   - τ must lie at least 2 min after the first timeline frame (`START_OFFSET_MIN`), and τ + 35 s
     (the label horizon) must not pass the last frame; the label is also left empty when the end of
     its window passes the last frame, so no engagement is labelled from a window the match does not
     cover (`gameplay/fights.py`, `detect_fights_teamfight_v2`, the start-offset and horizon guards;
     `gameplay/labels.py`, `_resolve_label_window`);
   - a candidate whose last kill comes more than 60 s after τ is dropped
     (`MAX_MERGED_FIGHT_DURATION_MS`);
   - two kept candidates are merged when the later τ falls within 15 s of the end of the earlier
     label window and their first kills lie within 2,000 u of each other, unless the merged span would
     exceed 60 s (`_merge_adjacent_candidates`);
   - the label window ends at the first team wipe (ace) inside it (`_truncate_fights_at_ace`).
2. **Observe** the window [τ − 30 s, τ] in six 5 s bins: per-player, team-level and event-count
   channels, summarised as 1,015 base features × 7 statistics plus the age of the last timeline
   frame, 7,106 columns in all (`gameplay/pipeline.py`, `gameplay/features.py`).
3. **Label** the window [τ, max(last kill + 1 ms, τ + 35 s)], cut short at a team wipe, with
   `market_event` (`gameplay/labels.py`; see [Label](#label-market_event)).
4. **Predict** with LightGBM, and compare other learners on identical rows.

## Definition constants (corpus v3.3)

*G* and *D* are estimated from 208,141 matches (10,417,458 inter-kill intervals) of patches
15.14–15.16 by `scripts/run_fight_boundary_pipeline.py`. The detector was run with the rounded
values.

| Symbol | Meaning | Estimate (95 % CI) | Value given to the detector | Source |
|---|---|---|---|---|
| *G* | largest gap between consecutive kills of one engagement | 13.7246 s (12.160–15.668) | 13,700 ms | antimode of the kernel density of log inter-kill intervals, 200 bootstrap replicates; clustering agreement (ARI ≥ 0.9) holds from 10 to 18 s |
| *D* | largest distance between kills of one engagement | 4,263.87 u (4,256.05–4,273.97) | 4,264.0 u | distance at which the share of consecutive kill pairs involving a common champion crosses 50 % |
| *R* | presence radius at the cutoff | — | 1,600 u | game rule: a champion's death shares experience with enemies within 1,600 u |
| *B* | lead of the cutoff before the first kill | — | 15 s | game rule: assist credit window on Summoner's Rift |
| *M* | champions per team alive within *R* at the cutoff | — | 2 | unchanged from the CoG detector |
| — | label horizon after the cutoff | — | 35 s | keeps at least 20 s after the first kill, as in the CoG corpus (10 s lead, 30 s horizon) |
| — | gold dead zone of the label | — | 300 g | `market_event` |
| — | scale classes (participants on the smaller side) | — | pick ≤ 1, skirmish 2–3, teamfight ≥ 4 | participation distribution; see the retraction above for other cuts |

Sources: estimates and CIs from `config/fight_boundary/spec_pooled.json`; detector values, horizon,
dead zone and class cut from `corpus_shards_v33/manifest.json` and the `v3.3` preset in
`core/presets.py`. Estimated per patch, *G* is 14.0 / 13.5 / 13.7 s and *D* 4,285 / 4,265 / 4,241 u
(`config/fight_boundary/spec_15.1{4,5,6}.json`), so one pooled definition serves all three patches.
The CI of *G* is conservative: each bootstrap replicate subsamples 100,000 intervals
(`config/fight_boundary/README.md`). ARI is the adjusted Rand index.

*R* and *B* are read from the League of Legends Wiki pages *Experience (champion)* (revision of
2026-08-22) and *Assist* (revision of 2026-05-12). Those pages describe the live game. Riot's notes
for patches 25.14–25.16 and the wiki pages of those patches list no change to either value, so the
values are "no change listed" for the corpus patches, not values checked patch by patch
(`docs/references/rule_constants_evidence.md`, sections 1–2; `docs/tog_manuscript/references_audit.md`,
section 2c).

**Patch numbering.** The API's game versions 15.14, 15.15 and 15.16 are the patches Riot published
as 25.14, 25.15 and 25.16 (notes of 15 July, 29 July and 12 August 2025). The correspondence rests
on Riot data: Data Dragon first lists the champion Yunara at version 15.14.1, Riot's 25.14 notes
introduce her, and she appears in 889 of 4,000 sampled corpus matches of version 15.14; the later two
follow by succession (`docs/tog_manuscript/references_audit.md`, note N2).

## Corpus

| Quantity | Value | Source |
|---|---|---|
| Collection | Korean (KR) server; the collector queried the Master, Grandmaster and Challenger ladders. Tier is a collection constraint, not a measured variable: the cache stores neither tier nor queue | `acquisition/config.py` (`tiers`); `analysis/analysis.py` (`sampling_frame`); keys of the cached metadata |
| Patches | 15.14, 15.15, 15.16 (Riot's 25.14–25.16) | see above |
| Matches in the per-match cache | 210,000 (15.14: 74,673; 15.15: 74,748; 15.16: 60,579) | file count of `match_cache_fresh_v3_engage_status13/`; its patch index |
| Matches with at least one inter-kill interval | 208,141 | `config/fight_boundary/spec_pooled.json` |
| Engagements in corpus v3.3 | 566,452 (32 shards, 7,106 columns) | `model_comparison_input_audit.json`, `corpus_shards_v33/manifest.json` |
| Engagements with a non-draw `market_event` label | 532,547 engagements (15.14: 187,547; 15.15: 191,184; 15.16: 153,816 engagements) in 191,940 matches | `scale_decomposition_v33_market_event.json` (`n`, `patch_distribution`, `n_matches`) |
| Share won by blue | 0.508 | same (`positive_rate`) |
| Scale classes (participants on the smaller side) | pick (≤ 1) 101,798, 19.1 %; skirmish (2–3) 320,878, 60.3 %; teamfight (≥ 4) 109,829, 20.6 %; the other 42 engagements have a participation count stored as −1 and fall in no class | same (`by_participation_scale`) |

The cache holds no player identifiers and no rank. The raw Match-V5 responses are not in the
repository and are no longer kept; what is released instead is set out in
[`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).

## Label: `market_event`

The label names the team that gained more gold from the engagement, counting only what belongs to
it. Over the window [τ, max(last kill + 1 ms, τ + 35 s)], only events that have a map position within
*D* (4,264 u) of the first kill are attributed to the engagement. For each kill the team is credited
with the gold the event reports (bounty plus shutdown bounty) plus a fitted price per kill and per
assist; turret plates, turrets, inhibitors and epic monsters are credited at fitted prices. Ward
kills carry no position and so fall outside the attribution. The prices are regression estimates of
the average team gold per event, not the rule payouts; elemental drakes are priced at 0 g
(`config/game_rules/event_prices.json`, estimated by `scripts/estimate_event_prices.py`). If the gold
difference exceeds 300 g, the team with more gold wins. Otherwise the tie is broken by the kills of
the cluster, then by the champions alive at the last kill, then by structure and monster events.
Engagements that are still tied are dropped (566,452 − 532,547 = 33,905). `y = 1` means that blue
won. Four label variants are stored beside it (`market_event@window`, `market_lex`,
`market_lex@window`, and `attention_value_win`, the CoG Eq. 3 label), each with −1 marking a draw;
the full label family is registered in `gameplay/labels.py`.

The shards hold one more label array, `y`. It is the row-building label: `market_event` with the
tie policy `random` (`corpus_shards_v33/manifest.json`, `label.row_tie_policy`), so the 33,905 draws
carry 0 or 1 from a deterministic coin (`_seeded_tie_coin` in `gameplay/labels.py`: a hash of seed 7,
the engagement key and the window's events). On the 532,547 non-draw rows it equals
`y_market_event` (checked row by row over the 32 shards on 2026-09-14). No number on this page uses
`y`: the headline, scale-cut and learner files record `y_market_event` as their label key, and the
leak ablation builds its own rows with the tie policy `drop` (`scripts/run_leak_ablation.py`).

## Headline result

All 532,547 labelled engagements, match-grouped 5-fold cross-validation (`GroupKFold`), one LightGBM
configuration (400 trees, learning rate 0.05, 31 leaves) on the 6,164 columns that are not constant
across the corpus. Class AUCs are computed on the frozen out-of-fold predictions, without refitting
per class. Intervals come from 1,000 match-level bootstrap replicates
(`scale_decomposition_v33_market_event.json`).

| Engagements | AUC (95 % CI) |
|---|---|
| all | **0.6699** |
| pick (smaller side ≤ 1) | 0.6789 (0.6757–0.6821) |
| skirmish (2–3) | 0.6633 (0.6614–0.6653) |
| teamfight (≥ 4) | 0.6809 (0.6776–0.6840) |
| pick − teamfight | −0.0020 (−0.0065 to +0.0026) |

The class AUCs are point estimates (`by_participation_scale`); the intervals are the 2.5 and 97.5
percentiles of the replicates (`bootstrap`). The gap row is the difference of the two point
estimates, 0.6789 − 0.6809 (unrounded 0.678869 − 0.680888 = −0.0020); the replicate mean of the gap
is −0.0019. The 42 engagements with a participation count stored as −1 are in no class here, while
the re-scoring in the retraction above counts them as picks, which gives −0.0019 at the same cut.

The 11,477 engagements that already have at least four champions per side present at the cutoff
reach 0.6995 (same file, `by_presence_scale`).

## Learner comparison (patch holdout)

Train on 15.14 (187,547 engagements), select on 15.15 (191,184) and test on 15.16 (153,816 engagements
in 55,449 matches). Holding out a whole patch keeps every learner from being tested on the patch it
was fitted on. Every learner gets the same rows, splits and label; an 84-check input audit confirms
it (`tog_revision/A8-input-audit-and-cis/model_comparison_input_audit_extended.json`). Tree and
linear models use the 6,164 non-constant columns; the deep learners use all 7,106. The 942 extra
columns are constant across the corpus, and LightGBM scores 0.6665 on the first column set and 0.6661
on the second (`model_comparison_input_audit.json`).

Test AUCs and their 95 % intervals are from 2,000 match-clustered bootstrap replicates
(`tog_revision/A8-input-audit-and-cis/paired_learner_cis_v33.json`); validation AUCs are from the
result files named below the table.

| Learner | Columns | Validation AUC | Test AUC (95 % CI) |
|---|---:|---:|---|
| LightGBM, larger capacity (up to 3,000 trees, learning rate 0.02, 127 leaves; stopped at 544) | 6,164 | 0.6698 | **0.6692** (0.6666–0.6719) |
| LightGBM, headline configuration | 6,164 | 0.6671 | 0.6665 (0.6638–0.6692) |
| LightGBM inside the deep-baseline script | 7,106 | 0.6666 | 0.6661 (0.6633–0.6688) |
| FT-Transformer | 7,106 | 0.6577 | 0.6561 (0.6533–0.6587) |
| Multilayer perceptron (MLP) | 7,106 | 0.6570 | 0.6552 (0.6525–0.6579) |
| SAINT-style row and column attention, supervised only | 7,106 | 0.6565 | 0.6546 (0.6519–0.6572) |
| Logistic regression (one linear layer, weight decay) | 6,164 | 0.6444 | 0.6408 (0.6380–0.6436) |
| LightGBM on the 8 lead columns only | 8 | 0.6274 | 0.6261 (0.6233–0.6289) |
| TabNet (do not cite; see caveats) | 7,106 | 0.6194 | 0.6188 (0.6160–0.6217) |

Result files: `model_comparison_v33_patch.json` (larger-capacity and headline LightGBM, logistic
regression, lead columns), `deep_tabular_v33_patch_full.json` (LightGBM inside the deep script, MLP,
TabNet), `deep_tabular_v33_patch_ft.json`, `deep_tabular_v33_patch_saint.json`.

Paired differences, same file and replicates: larger-capacity LightGBM − headline LightGBM
+0.0027 (+0.0020 to +0.0034); headline LightGBM − logistic regression +0.0257 (+0.0238 to +0.0274);
larger-capacity LightGBM − lead columns +0.0431 (+0.0412 to +0.0449). Headline LightGBM exceeds
FT-Transformer by 0.0104 (0.0090 to 0.0117) and the MLP by 0.0113 (0.0097 to 0.0128). Among the deep
learners, FT-Transformer − MLP (+0.0008, −0.0005 to +0.0023) and SAINT − MLP (−0.0006, −0.0021 to
+0.0008) are ties, and FT-Transformer − SAINT is +0.0015 (+0.0005 to +0.0024).

The lead columns are the gold, experience, level, kill, creep score and alive-count differences, the
elapsed time and the frame age. By class they score 0.6017 (pick), 0.6156 (skirmish) and 0.6718
(teamfight); the larger LightGBM scores 0.6799, 0.6619 and 0.6808 (`model_comparison_v33_patch.json`).

**Caveats: read before citing the deep rows.** Conclusions about deep learners are provisional until
the corrected runs finish.

- The deep learners were run with one fixed configuration each (FT-Transformer and SAINT: 32-dimensional
  tokens, 3 layers, 8 heads, learning rate 1e-4, batch 64). No hyperparameter search was done, and
  capacity was not matched: about 4.17 M parameters for the MLP, 0.29 M for FT-Transformer and
  0.46 M for SAINT (`paired_learner_cis_v33.json`, caveats).
- TabNet: the published run (script at commit 60945ed) subtracted the mask-entropy term from the loss,
  which rewards dense feature masks; the TabNet paper adds it. The current script adds it, and
  `--tabnet-legacy-sparsity-sign` reproduces the published run.
- SAINT: the published run had no contrastive pre-training, and its evaluation batches, taken in load
  order, let row attention reach other engagements of the same match. The current script uses
  match-disjoint batches; `--saint-batches load_order` reproduces the published run.
- The logistic regression, both LightGBM rows of `model_comparison_v33_patch.json` and the lead-column
  row come from `scripts/run_model_comparison_v33.py` on branch `codex/engagement-state-value`
  (commit 3fb00c3); the audit and the intervals come from `scripts/audit_model_comparison_inputs.py`
  and `scripts/paired_learner_cis_v33.py` on the same branch. The two result files record commit
  a0f5bee with both scripts modified; the script SHA-1s they record are those of the versions
  committed in 37bc991. None of the three scripts is on this branch yet.

---

## Reproducing

The repository ships no match data. Numbers are reproduced either from the derived per-engagement
tables, which are released separately with the extension (not yet published), or from your own
Match-V5 data ([`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md), section 2.3).

### Setup

```bash
pip install -e ".[all]"          # or: pip install -r requirements.txt
pytest                           # 588 tests collected on 2026-09-14
```

Configuration lives in `core/config.py`. Its defaults are the CoG 2026 constants. The presets in
`core/presets.py` (`cog2026`, `v3.3`) are selected with `LOL_CFG_PRESET`, and single fields can be
overridden with `LOL_CFG_OVERRIDES` (a JSON object). The commands below use POSIX shell syntax; in
PowerShell, set each variable with `$env:NAME = "value"` first.

| Variable | Meaning | Default |
|---|---|---|
| `LOL_DETAIL_DIR` | Match-V5 match JSON files | `data/raw/matches/kr/detail` |
| `LOL_TIMELINE_DIR` | Match-V5 timeline JSON files | `data/raw/matches/kr/timeline` |
| `LOL_OUTPUT_ROOT` | cache and run outputs; the cache is `<root>/cache/match_cache_fresh_v3_engage_status13` | `outputs` |

### 1. Data and cache

Fetch match and timeline records with your own Riot API key. Collection tooling is in
`acquisition/` and `scripts/collect_current_season.py` (see `docs/ACQUISITION_2026.md`). Then build
the per-match cache:

```bash
LOL_DETAIL_DIR=<detail> LOL_TIMELINE_DIR=<timeline> LOL_OUTPUT_ROOT=<root> python main.py --mode build_cache
```

### 2. Definition pipeline (*G*, *D*)

```bash
LOL_OUTPUT_ROOT=<root> python scripts/run_fight_boundary_pipeline.py \
    --n-matches-per-patch 0 --n-boot 200 --seed 7 --validity-radius-u 1600 --lead-s 15 \
    --out-dir <boundary>
python scripts/compare_boundary_specs.py config/fight_boundary <boundary> --label-a released --label-b rerun
```

### 3. Event price table (optional; the corpus uses the released table)

```bash
LOL_OUTPUT_ROOT=<root> python scripts/estimate_event_prices.py --n-matches-per-patch 8000 --seed 7 \
    --n-boot 200 --output <prices.json> --table <price_table.json>
```

Write the table outside `config/` and compare it with `config/game_rules/event_prices.json`. The
corpus manifest records the SHA-1 of the table the corpus was built with: the file at commit
5b5f9c8. Commit 006d5e4 changed only the description strings of that file, not its prices, so the
current file has a different hash and the same prices.

### 4. Corpus v3.3 and the headline

```bash
LOL_OUTPUT_ROOT=<root> python scripts/build_corpus_v3.py --out-dir <shards> --output <headline.json>
```

The script sets `LOL_CFG_PRESET=v3.3` itself, builds 32 shards carrying every stored label, and then
runs `scripts/run_scale_decomposition.py --y-key y_market_event --teamfight-min 4`, which checks the
shard manifest first. The decomposition writes a float32 memory-mapped matrix (`*.matrix.npy`,
13,966,440,640 bytes for v3.3).

### 5. Learner comparison

Published deep rows (the two legacy flags reproduce the defects listed in the caveats):

```bash
python scripts/run_deep_tabular_baselines.py --shards <shards> --n-matches 0 --split patch \
    --y-key y_market_event --models lightgbm,mlp,tabnet --tabnet-legacy-sparsity-sign --output <deep_full.json>
python scripts/run_deep_tabular_baselines.py --shards <shards> --n-matches 0 --split patch \
    --y-key y_market_event --models ft_transformer --output <deep_ft.json>
python scripts/run_deep_tabular_baselines.py --shards <shards> --n-matches 0 --split patch \
    --y-key y_market_event --models saint --saint-batches load_order --output <deep_saint.json>
```

`--n-matches 0` uses every match; the default (40,000) draws a subsample. The tree, linear and
lead-column rows, the input audit and the paired intervals need the three scripts on branch
`codex/engagement-state-value` named in the caveats. Pass `--shards`, `--matrix`, `--out` and
`--preds` (or `--tree-results`, `--tree-preds`, `--deep-results`, `--out`) explicitly, because their
defaults are absolute paths on the machine that produced the results.

### 6. Leak ablation

```bash
LOL_OUTPUT_ROOT=<root> python scripts/run_leak_ablation.py --n-matches 5000 --seed 7 --output <leak.json>
```

The script sets the v3.3 detector constants itself and toggles only the two leak flags.

---

## Where results live

Result files are kept outside the repository, on the machine that produced them, under
`D:/LOL_Project/fusion_2615/`.

| Path | Content |
|---|---|
| `corpus_shards_v33/` | 32 shards (`shard_*.npz`, 2,350,323,256 bytes), `manifest.json`, `feature_names.json`, `build.log` |
| `features/scale_decomposition_v33_market_event.json` (+ `.preds.npz`) | headline and its out-of-fold predictions |
| `features/scale_decomposition_v33_{market_event_cat,market_event_window,market_lex,market_lex_window,attention_value_win}.json` | label and encoding variants |
| `features/model_comparison_v33_patch.json` (+ `.preds.npz`), `features/deep_tabular_v33_patch_{full,ft,saint}.json` (+ `.preds.npz`), `features/model_comparison_input_audit.json` | learner comparison and its first input audit |
| `features/tog_revision/A8-input-audit-and-cis/` | 84-check input audit, match-clustered learner intervals |
| `features/tog_revision/A6-definition-sensitivity/` | scale-cut sensitivity (under re-verification) |
| `features/tog_revision/killless_grid/summary.json` | kill-less proximity encounters, 9 gates × 20,000 matches |
| `features/leak_ablation_v33.json` | leak ablation |
| `features/event_prices.json` | price regression with bootstrap intervals |
| `features/fight_boundary_full/` | definition pipeline output, copied to `config/fight_boundary/` |

The kill-less grid counts proximity encounters, not engagements, so its shares must not be divided
into engagement counts. The scanner (`scripts/run_killless_encounters.py`, `encounters_for_match`,
unchanged since commit 5600f8b, which the grid records) marks a frame of the 5 s position grid as
active when some alive champion has at least `--min-per-team` alive champions of each side within
`--radius` of itself; every alive champion is tried as that anchor. An encounter is a run of
consecutive active frames at least `round(--min-duration / 5 s)` frames long (three frames for
13.7 s, four for 20 s). It is kill-less if no champion dies inside it or within `--grace-ms` before
its first frame or after its last. (The module docstring still describes a joint-centroid anchor and
a trailing grace period only; both are out of date, the centroid anchor having been removed in commit
ebcca4a.) With four champions per side within 1,600 u and a 15 s grace, 3.09 % of the encounters of
at least three frames and 2.79 % of those of at least four frames are kill-less: 0.030 and 0.018
kill-less encounters per match, against 0.572 teamfight-class engagements per match in the corpus.
With two per side (three frames), 7.95 % are kill-less, 0.838 per match (`killless_grid/summary.json`,
rows `r1600_t4_dG_g15`, `r1600_t4_d20_g15`, `r1600_t2_dG_g15` and `corpus_reference`; 20,000 matches
each). The other revision runs (label variants, observation
windows and sequence learners, the G × D sweep, the presence gate, SHAP forensics, the corrected deep
learners, the evidence-state holdout, the kill-less characterisation) have not finished; cite none
of their files.

The v3.3 section of [`docs/DATA_MANIFEST.md`](docs/DATA_MANIFEST.md) maps each file to its producer;
the older sections of that manifest describe superseded corpora.

## Documentation

| Document | Content |
|---|---|
| [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md) | what is released, what is not and why; the Riot API terms; review anonymity; the anonymised review copy |
| [`docs/DEFINITION_EVIDENCE.md`](docs/DEFINITION_EVIDENCE.md) (Korean) | evidence for every constant, label decision and audit |
| [`docs/ENGAGEMENT_DEFINITION_V3.md`](docs/ENGAGEMENT_DEFINITION_V3.md) (Korean) | compact statement of the definition, inputs, labels and protocols |
| [`docs/INPUT_FEATURE_AUDIT_V3.md`](docs/INPUT_FEATURE_AUDIT_V3.md) (Korean) | the input audit that found the two leaks |
| [`docs/REPRESENTATION_AUDIT_V3.md`](docs/REPRESENTATION_AUDIT_V3.md) (Korean) | audit from JSON field to model input |
| [`docs/RELEASE_REPORT_V3.md`](docs/RELEASE_REPORT_V3.md) (Korean) | release notes for definition v3 and corpus v3.3 |
| [`docs/DEFINITION_LITERATURE.md`](docs/DEFINITION_LITERATURE.md) | how fights and encounters are defined in the literature |
| [`config/fight_boundary/README.md`](config/fight_boundary/README.md) | the released boundary specifications |
| `docs/tog_manuscript/` | manuscript sources of the extension, with `references_audit.md` for the bibliography |
| `docs/PIPELINE.md`, `docs/FEATURES.md`, `docs/MODELS.md`, `docs/EXPERIMENT.md`, `docs/CoG2026_Paper.md`, `docs/AUDIT.md` | CoG-era documents; constants and numbers there may predate v3.3 |

## Project layout

```
acquisition/   Riot API collection agent
analysis/      definition pipeline (kill pairs, temporal and spatial boundary, boundary spec), reports
app/           orchestration behind main.py and runner.py
config/        fight_boundary/ (definition specs), game_rules/ (event prices, map anchors)
core/          config.py (central configuration), presets.py (cog2026, v3.3), feature contracts
data/          cache I/O, fight index, splits, datasets
gameplay/      fights.py (detector), labels.py, anchors.py, pipeline.py and features.py (features)
scripts/       corpus build, decomposition, baselines, audits, ablations, anonymised export
train/         LightGBM baseline, deep models, fusion
tests/         pytest suite
docs/          documentation; docs/tog_manuscript/ holds the extension's manuscript sources
```

`scripts/prepare_anonymous_release.py` writes an anonymised copy of one commit into a local directory
outside the repository for double-anonymous review. It runs only read-only git commands and never
pushes or uploads ([`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md), section 7).

## History

- **`v1.0-cog2026`** (annotated tag of 2026-07-28 on commit `b3d330c` of 2026-07-03): the code state of the IEEE CoG 2026
  submission. It used hand-set constants (*G* 18 s, *D* 4,000 u, *R* 1,800 u, *B* 10 s, 30 s horizon),
  the Eq. 3 attention-value label and the leaking feature path. `LOL_CFG_PRESET=cog2026` restores those
  settings on the current code. `docs/AUDIT.md` records the code-to-paper audit of that state.
- **Definition v3 and corpus v3.3** (branch `feature/fight-boundary-pipeline`; definition pipeline
  from 2026-09-08, corpus v3.3 built 2026-09-09): data-derived *G* and *D*, rule-anchored *R* and *B*,
  the `market_event` label, the input audit, and the retraction above.

A citation entry will be added when the extension is published.

## Legal

The code is released under the [MIT License](LICENSE).

*LOL Teamfight Lab* is not endorsed by Riot Games and does not reflect the views or opinions of Riot Games or anyone officially involved in producing or managing League of Legends. League of Legends and Riot Games are trademarks or registered trademarks of Riot Games, Inc.
