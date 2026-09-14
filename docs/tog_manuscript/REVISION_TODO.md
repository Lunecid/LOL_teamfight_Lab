# ToG revision — what is left (state at 2026-09-14 21:15 KST)

The authors decided to accept every CoG 2026 reviewer recommendation. All experiments that do not depend on the
deep-model search are finished and written into the manuscript; `main.tex` builds with 0 errors, 0 undefined
references and 0 undefined citations (41 pages). This file lists what remains, in order.

## 1. Compute still running

Runner (detached): `scripts/tog_revision_queue.py` on `D:/LOL_Project/fusion_2615/features/tog_revision/queue_wave5.json`,
state in `.../tog_revision/queue_state/` (a job is finished only when `<job>.ok` exists).

```bash
.venv/Scripts/python.exe scripts/tog_revision_queue.py --queue D:/LOL_Project/fusion_2615/features/tog_revision/queue_wave5.json --state-dir D:/LOL_Project/fusion_2615/features/tog_revision/queue_state --status
```

| job | what | estimate |
|---|---|---|
| `b2_saint_d128_published_budget` | SAINT, d_token 128, match-disjoint batches | running |
| `c_saint_d32_pretrain5_published_config` | SAINT with 5 contrastive pre-training epochs | ~2.5 h |
| `d1_hparam_search_lightgbm` | declared LightGBM grid, validation-only selection | ~2.5 h |
| `d2_hparam_search_deep` | declared grids for MLP, TabNet, FT-Transformer, SAINT (±pre-training) | ~22 h |
| `d3_hparam_search_summary` | one test evaluation per family | minutes |

Do not restart the runner while a job runs: a restarted runner adopts surviving jobs only when their process
exits, and one adoption was observed to lag by ~50 minutes.

## 2. Writing that waits for §1

1. `sec_learners.tex`: fill the SAINT d128, SAINT pre-training and search rows. Re-run
   `scripts/learner_rerun_paired_cis_v33.py` (it is built to add the new prediction files) and use its JSON for every
   interval. Then write the answer paragraph. A model-class conclusion is allowed only after every learner row is filled.
2. Extend the 84-check input audit (worktree `scripts/audit_model_comparison_inputs.py`) to the re-run and searched
   prediction files; fill `\pending{audit}`.
3. Restore a model-class claim to the title only if §2.1 supports it (the current working title makes none).

## 3. Writing that can be done now (small)

1. `sec_limitations.tex`: fill `\pending{labels}` (A2 `label_family_v33.json`), `\pending{evidence_state}`
   (A4 `evidence_block_patch_holdout_v33_v2.json`) and `\pending{window}` (A3 `window_sweep_paired_cis_v33.json`, with the
   ten-seed qualification); remove the stale sentences "No other radius has been evaluated yet" (D/2 and no-disc results
   exist) and "Until it is reported, the headline AUC of 0.670 holds for this price table".
2. `sec_intro.tex`: "the time leak raised ... that of picks by 0.003" needs its interval, which includes zero
   ([-0.012, +0.017], `C3-leak-ablation/`).
3. Comparator naming: `sec_learners.tex` pairs re-runs against `lgbm_paper` (0.6665) on the A8 draws, `sec_limitations.tex`
   against the 7,106-column pipeline LightGBM (0.6661). Both are correct; say which one each time, or use one.
4. `sec_conclusion.tex` (missing) and the abstract in `main.tex` (last; R1 asked for one connective sentence between the
   definition result and the prediction result, and one between the prediction result and the observation result).

## 4. Decisions only the authors can make

- Author block and anonymisation — depends on the ToG review model (see `sec_availability.tex`).
- `\pending{anon-url}`, `\pending{release-doi}`, `\pending{data-licence}`: release form, archive, and whether to ask Riot
  for written confirmation that per-engagement tables may be shared.
- Optional experiments reviewers did not strictly require:
  sequence learners at 15 s and at 2.5 s / 10 s bins (`\pending{sequence-arms}`); a convolutional learner (R3 mentioned
  CNN probes; the text explains why the MLP is a baseline, not a probe); a 24,000-match price-regression refit
  (`\pending{price_diag}`); own-model scoring of the per-member drake price variant (`\pending{dragon_per_member}`).
- `\pending{rulefact}`: who receives the 25 g elemental-drake kill gold in 15.14–15.16 (no dated source found).
- Before submission a person should open the 25.14–25.16 patch notes (`docs/references/rule_constants_evidence.md` §2
  records an automated check only).

## 5. Final pass (after §2–§4)

1. Full build; every `\pending` resolved or deliberately kept; `grep -n "\\pending" docs/tog_manuscript/*.tex`.
2. Whole-paper adversarial review against the four CoG reviews and the retraction list in
   `docs/CLAUDE_TOG_PAPER_PLAN.md` §4; trace every printed number to its artefact comment.
3. Update `reviewer_response_matrix.md` statuses and `docs/CLAUDE_TOG_PAPER_PLAN.md` (it still carries two errors fixed in
   the manuscript: the stacked predictor in §7.2–7.3 was the |ΔV|-weighted sign model, and the window result needs the
   ten-seed qualification).

## Results already in the manuscript (for orientation)

| question | result | artefact |
|---|---|---|
| R2 labels | 12 variants on 484,474 common rows: AUC 0.674–0.689, verdict agreement 96–99.99 %; raw kill advantage 0.6878 vs market_event 0.6822; the market_event model transfers to every variant | `A2-label-family/label_family_v33.json` |
| R1 label explained | gold swing decides 85.99 %, kills 9.28 %, survivors 3.51 %, structures 1.22 %; draws 5.99 % of detected | `C4-label-tiers/label_tier_shares_v33.json` |
| R1 thresholds | presence-gate operating point reproduces 0.6665; G × D: 2 of 19 shared-row intervals below zero (both G = 10 s) | `A6-definition-sensitivity/` |
| R4 kill-less | four-per-side gate 3.09 % (kill-adjusted) / 4.54 % (frame-only) of proximity encounters, 0.030 per match vs 0.572 teamfight engagements | `killless_grid/`, `A5-killless/` |
| R2 temporal | 120 s window and 2.5 s bins: no gain robust to both evaluation sampling and training seed (2.5 s seed-mean +0.0016 [−0.0007, +0.0039]); BiGRU 0.6163 vs LightGBM 0.6380 on the same rows | `A3-temporal-windows/` |
| R2 deep so far | TabNet sign-fixed 0.6114; FT-Transformer d192 0.6578; SAINT match-disjoint 0.6550 (published SAINT row leak-exposed) | `A1-deep-baselines/`, `C5-learner-cis/` |
| observation bound | evidence-state reconstruction +0.0068 [+0.0057, +0.0078] on the full corpus, patch holdout | `A4-evidence-state-leak/` |
| stacking | market_event predictor adds nothing to match prediction (B−A [−0.00099, +0.00129]); the realised outcome adds +0.020 | `A9-stacking-market/` |
| leak audit | time leak +0.019 overall, +0.049 on teamfights; re-run bit-identical to the published ablation | `C3-leak-ablation/` |
