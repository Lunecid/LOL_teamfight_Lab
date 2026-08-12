# ToG Extension Plan — Decomposed Engagement Outcome Prediction

Decision (2026-08-11, after the CoG 2026 result): the CoG submission scored
+1/−1/+1 with an explicit meta-reviewer *accept* recommendation and was cut on
capacity (89/239). We therefore sharpen rather than rebuild: extend the same
study to an **IEEE Transactions on Games** submission whose headline is
**decomposition and verification**, not a new modality and not a paradigm race.

## Thesis

Engagement outcome predictability under public telemetry is not one number.
It decomposes by engagement scale — picks are predicted by *state* (who is
where, at what health), teamfights barely at all before onset — and every
metric in the paper carries its own verification: label sanity against raw
fight outcomes, threshold sensitivity, patch replication (15.14–16 corpus →
16.15 corpus), and an independent measurement channel (replay HUD readings)
quantifying what minute-resolution telemetry cannot see.

Evidence already in hand (2026-08-11 session, see feature files under
`D:/LOL_Project/fusion_2615/features/`):

- Scale decomposition (n=1,328 joined, causal): telemetry AUC 0.483 on picks
  (below chance) vs 0.578 on teamfights; vision−telemetry +0.100
  CI[+0.009,+0.188] on picks, −0.026 on teamfights; scale interaction +0.127
  CI[+0.016,+0.237] p=0.013. Corpus reality: median 2 kills, 65% decided by
  ≤1 kill, only 34% true teamfights.
- Label sanity: Eq.3 label agrees with the sign of the fight's actual kill
  diff 95.6% (AUC 0.954) — first answer to R2's "learning the labelling
  heuristic" concern; the full ablation below completes it.
- Matched-size corpus control: paper 0.675 (full corpus) → 0.595 (553
  original-corpus matches) → 0.571 (16.15 corpus) → 0.546 (early-fight-skewed
  joined rows). Sample size dominates; the 16.15 shift is real but small.
- Window sensitivity w10/20/30 ≈ equal; trajectory features carry ~nothing
  (0.53) vs static state (0.56–0.58) — measured answer to R2's "temporal
  signal is insufficient": it is a property of the game state, not of the
  models.
- Cross-modal verification: replay HUD HP vs timeline minute boundaries MAE
  0.0065 (96.4% within 2 %p, 1,788 slots); telemetry is a median 37 s stale
  at the prediction cutoff and rank-correlates only 0.126 with the true
  at-cutoff team HP difference.

Vision's role in this paper: **verification instrument, not headline
modality.** The fusion deltas are small and fragile (the causal re-run moved
the gold-stratum interaction to p=0.057); the scale story and the staleness
measurements are the defensible material.

## Reviewer point → planned response

| # | Review point | Response | Status |
|---|---|---|---|
| R2-1 | Eq.3 hand-set weights: "model may learn the labelling heuristic"; wants alternative labels + weight sensitivity | `run_label_ablation.py` + `run_label_weight_sensitivity.py` | **done** — Eq.3 (0.595) is the *hardest* target of the four schemes (micro_win 0.652, kill_survival 0.662, weighted 0.675); schemes agree 87–98%; coefficient perturbations flip ≤4% of labels and move AUC monotonically along an interpretable attention↔material axis |
| R2-2 | DL baselines underpowered (FT-Transformer/TabNet/SAINT absent, no shared feature engineering) | `run_deep_tabular_baselines.py` — LightGBM / MLP / FT-Transformer on identical rows, split, features and early-stopping budget | **done** — LightGBM 0.6767, FT-Transformer 0.6578, MLP 0.6335 (n=192,727 / 40k matches). LightGBM−FT +0.019 CI[+0.014,+0.024]; FT−MLP +0.024 CI[+0.019,+0.030]. The gap narrows from the CoG submission's ~0.10 to 0.019 once the deep model gets the same features and a modern architecture, but does not close |
| R2-3 | 6 timesteps × 60 s resolution starves sequential models | Measured: window-length sweep (w10/20/30 within 0.02) + trajectory-vs-snapshot ablation (0.53 vs 0.58) show static state carries the signal | **done** — write-up pending |
| R2-4 | No anonymised code/data link | Anonymised repo + derived-data release (Riot ToS: derived features, not raw dumps) | planned |
| Meta | "What about teamfights with no kills?" | `run_killless_encounters.py`: proximity encounters from the 5 s grid under the detector's own validity condition minus the kill requirement | script ready, run pending |
| R1-1 | Contribution statement restates RQs | Rewrite around community impact (esports narrative/broadcast tooling; R1's six references) | writing |
| R1-2 | Arbitrary thresholds (18 s kill-cluster window, 4,000-unit spatial split) | `run_threshold_sensitivity.py` | **done** — 12/18/24/30 s × 3k/4k/5k units: engagement counts move ±12%, AUC spread 0.008, positive rate 50.5–50.9% |
| R1-3 | LoL terminology opaque | `docs/ENGAGEMENT_SCALE_DEFINITION.md` formalizes engagement, both scale definitions, the class cutoff with its joint distribution, and the asymmetry covariate | **done** — paper prose pending |
| R1-4/R3 | Limitations/future work thin; methods not self-contained (diagnostic MLP, Layered+Logit, TreeSHAP unmotivated) | Dedicated limitations section; motivate or drop each diagnostic; SHAP re-run required anyway after the name-order fix (5ed2585) | writing |

### Headline experiment

`build_corpus_shard.py` (32 shards, ~12.5 matches/s each) + `run_scale_decomposition.py`
over the full paper corpus: 210,000 matches → **994,365 engagements**, 4,888
non-constant features of 7,105, matrix held as a 19.4 GB disk memmap so only
the per-fold training slice is resident. One fit under match-grouped folds,
frozen out-of-fold predictions scored inside each scale class.

## Definitions to formalize (with citations)

- *Engagement* and encounter detection lineage: Schubert, Drachen & Mahlmann
  (MIT Sloan 2016) — kill-event clustering is our operationalization; state
  the 18 s / 4,000-unit parameters with the sensitivity analysis.
- *Scale classes*: observed event participants per team (code:
  `classify_fight_scale`) — min(blue,red) ≥ 3 teamfight, ≥ 2 skirmish, else
  pick. Post-hoc property: subgroup reporting only, never an operational
  stratifier (it is unknown at the prediction cutoff).
- *Prediction cutoff*: engage_ts = first_kill − 10 s via deterministic
  backtrack; causal contract (no feature may read past the cutoff — the
  distance-to-fight leak found 2026-08-11 is the cautionary example, commit
  782836e).
- Audience/narrative impact framing: Block 2018; Kokkinakis 2020 (DAX);
  Charleer 2018; Chitayat 2024 ×2. Minimap perception lineage: Kim et al.,
  IEEE ToG 2024 (10.1109/TG.2024.3515140).

## Corpus plan

1. **Old corpus (paper corpus, patches 15.14–16, ~210k matches cached)** —
   headline numbers, full-scale decomposition, chronological patch holdout as
   in the CoG submission.
2. **16.15 corpus (553 matches, 2,486 engagements, replays archived)** —
   replication axis + the cross-modal verification channel (1,328 captured
   windows; capture of later engagements pending, `--engagement-selection`).
3. Pilots run on the seeded 553-match old-corpus sample (seed 7) so every
   pipeline change is cheap to validate before full-corpus runs.

## Execution order

1. Label ablation pilot (this session) → full old corpus.
2. Threshold sensitivity (18 s, 4,000 u) on the same sample.
3. Full old-corpus scale decomposition under patch holdout (the big run).
4. Kill-less encounter quantification.
5. FT-Transformer due-diligence baseline.
6. SHAP re-run (post name-fix) for the attribution section.
7. Writing: background, contributions, limitations; anonymised release prep.
