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
| R2-1 | Eq.3 hand-set weights: "model may learn the labelling heuristic"; wants alternative labels + weight sensitivity | `scripts/run_label_ablation.py`: same X, same folds, y under {attention_value_win (Eq.3), micro_win (raw kill advantage), kill_survival, weighted}; pairwise agreement; AUC stability; then weight perturbation on Eq.3 coefficients | pilot running |
| R2-2 | DL baselines underpowered (FT-Transformer/TabNet/SAINT absent, no shared feature engineering) | Reframe: paradigm comparison is no longer the claim. Add one modern tabular DL (FT-Transformer) **on the same engineered tabular features** as due diligence | planned |
| R2-3 | 6 timesteps × 60 s resolution starves sequential models | Measured: window-length sweep + trajectory-vs-snapshot ablation show static state carries the signal; discuss as data property | done (write-up) |
| R2-4 | No anonymised code/data link | Prepare anonymised repo + derived-data release (respect Riot ToS: derived features, not raw dumps) | planned |
| Meta | "What about teamfights with no kills?" | Acknowledge kill-anchored detection bound; quantify prevalence of kill-less proximity encounters from the 5 s position grid on a corpus sample; discuss as scope limit | planned |
| R1-1 | Contribution statement restates RQs | Rewrite around community impact (esports narrative/broadcast tooling; R1's six references) | writing |
| R1-2 | Arbitrary thresholds (18 s kill-cluster window, 4,000-unit spatial split) | Sensitivity sweep over both thresholds on a corpus sample: engagement counts, label stability, AUC | planned |
| R1-3 | LoL terminology opaque | Background section with formal definitions (engagement, pick/skirmish/teamfight via observed participants, engage backtrack, Eq.3) | writing |
| R1-4/R3 | Limitations/future work thin; methods not self-contained (diagnostic MLP, Layered+Logit, TreeSHAP unmotivated) | Dedicated limitations section; motivate or drop each diagnostic; SHAP re-run required anyway after the name-order fix (5ed2585) | writing |

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
