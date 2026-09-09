# Objective-aware engagement value experiment

This opt-in experiment adds a small match-outcome value model to the v3.3
engagement pipeline. It does not replace the released labels or detector defaults.
The baseline is Claude's completed audit release, commit `d607e3a`.
The completed 50,000-match experiment and its limitations are recorded in
`STATE_VALUE_50K_RESULTS_20260909.md`; it does not demonstrate added match-win
prediction benefit from the engagement probability.

## Research questions and precise output

1. Can pre-cutoff information predict an increase in a frozen model's evaluation
   of Blue's match-winning probability over an engagement and its outcome window?
2. Does the engagement prediction improve final match-outcome prediction using
   the same pre-cutoff information and the same learner?

`delta_value = V(state_post) - V(state_pre)`. Positive is Blue (1), negative is
Red (0); exact ties and invalid states are retained as -1 with a reason. This is
a model-defined outcome, not a causal effect or a universally correct definition
of engagement victory. The event value is not reduced to gold.

For a perfectly specified win probability conditional on the available history,
expected future probability changes are zero at fixed horizons. A learned model
is approximate, and the sign may still be predictable. Interpret performance as
predicting model-value updates; the `value_pre`-only diagnostic helps identify
dependence on the initial value. Retrospective kill-conditioned sampling further
limits interpretation as a live forecasting system.

## Inputs and data limitations

`gameplay/state_value.py` builds states from the latest snapshot at or before the
query plus events at or before that query. It never uses the legacy global vector,
match length, future anchors, or cached buff/soul columns. The terminal
`GAME_END.winningTeam` is read by a separate target function only.

- Ten role-ordered participants: normalized total/current gold, experience,
  level, health/mana, snapshot alive flag, CS, champion ID, kill/death history.
- Snapshot timestamps and age. A snapshot alive flag is not relabelled as exact
  current life status; later death indicators are separate. Cached role assignments
  are retained as a domain convention, not independently established online roles.
- Team objective counts: elemental dragons by type, Baron, Elder, Herald, grubs,
  Atakhan, explicitly recorded soul events by type, destroyed towers by type,
  inhibitor destruction events, and plates. Inhibitor counts are historical
  destructions, not claims of currently absent inhibitors after respawn.
- Time since objective acquisition and whether Baron/Elder was acquired in the
  previous 60/120/180/300 seconds. These overlapping history intervals are fixed
  feature choices, **not buff duration rules or strategic reward weights**.
- Per-role deaths since the team's last Baron/Elder acquisition. These do not
  prove acquisition eligibility or current buff ownership.
- A small set of objective-history × elapsed-time interactions.

The cached status builder grants buffs to all team members, does not clear them
on death, and processes frame events after writing node states. Its status flags
are therefore deliberately not used in the value model. Exact buff ownership and
expiry remain unavailable in this baseline. The new model represents their effects
through acquisition and subsequent-death history, and does not claim perfect
reconstruction. Explicit soul events are used; no soul is silently inferred from
a missing event. Unknown objective owners are counted diagnostically.

Engagement input defaults to the existing full 7,106-column pre-cutoff tabular
pipeline. `--input compact` uses current and 30-second-prior state vectors for a
quick smoke test only. Valuation and engagement inputs have separate schemas.
The causal status limitations of existing full inputs remain documented in the
v3.3 audit; this addition does not claim to repair every inherited feature.

## Windows, samples and splits

The existing v3.3 `FightRef` sample definition, cutoff and label window are kept.
The post-state is queried at `label_end_ts - 1`, consistently with the existing
exclusive end boundary. No unapproved change to a "last kill + 20 seconds" window
is made. Other simultaneous events may contribute to the whole-state change;
the result is a window-level evaluation, not attribution of all changes to the fight.

Match selection uses a fixed seeded SHA-256 ranking independent of outcomes.
A different hash fixes disjoint match roles: value train 20%, value validation
10%, engagement/stacker train 50%, held-out prediction test 20%. All snapshots and
engagements from a match share its role. The original definition was previously
explored on this corpus; the new prediction split is not an untouched confirmation
of the definition itself.

Value training uses nonterminal minute-grid states from minute 2 plus the pre/post
states of engagements in its own matches. Logistic regression with scaled numeric
features and one-hot champion categories is fitted only on value-training
matches. L2 strength C is selected from {0.0001, 0.001, 0.01, 0.1, 1} by
three-fold match-grouped cross-validation **inside value training only**, minimizing
match-weighted log loss. Preprocessing is fitted separately in each fold. The
value-validation and engagement partitions do not select C. This limited tuning
was added after the initial fixed-C pilot revealed poor probability quality;
the initial result is retained, and subsequent use of the same prediction test
is exploratory rather than a fresh confirmatory test. A versioned model artifact is frozen before generating
any engagement targets. Validation measures discrimination, Brier score, log loss,
calibration and objective coverage; it is not used to optimize downstream AUC.

The objective ablation uses the same selected C, so it is a fixed-regularization
diagnostic rather than a fully retuned best-model comparison. Objective coverage
reports both state counts and distinct matches; repeated states are not independent
Elder examples.

Engagement training uses match-grouped OOF predictions for the stacker training
rows and a model fitted on all engagement-training matches for the held-out test.
Each learner has a fixed configuration. A and B both use the same pre-cutoff X:

- A: X → final match win.
- B: X + OOF engagement probability → final match win.
- C: X + realized engagement label → final match win; retrospective reference only.

All reported probabilities and metrics weight matches equally (windows within
each match share its weight). A paired match bootstrap compares B and A. A positive
Brier-improvement interval means A's loss minus B's loss is positive; the AUC
interval is B minus A. No improvement is also a valid result. Because B receives a
function of X, its benefit is a representation/learning effect, not new information.
Association with final win does not independently establish the constructed
engagement label's semantic validity.
The bootstrap conditions on the fitted models; it does not include retraining or
value-model estimation uncertainty. Fractions with absolute value change below
0.001/0.01/0.05 are diagnostics, not silently imposed label thresholds.

## Reproduction

Use Python with the repository dependencies installed. Run from the repository:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python scripts/build_state_value_dataset.py --cache-dir D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13 --out-dir outputs/state_value_reproduce --n-matches 5000 --input full
python scripts/run_state_value_experiment.py --dataset outputs/state_value_reproduce --out-dir outputs/state_value_reproduce_eval
python -m pytest tests/test_state_value.py -q
```

The builder redirects runtime outputs into the requested output directory and
reads the source cache in place. It never downloads or writes source data. Atomic
per-match checkpoints allow resuming the same settings. A new settings/schema
requires a new output directory. An experiment refuses an incomplete manifest or
an existing completed result. For all matches use `--n-matches 0`; verify resource
capacity before materializing full matrices.

Artifacts include the selected match list, split lists, schema, source-data
exclusions, per-match states/timestamps/IDs, frozen value model and hash, old-label
comparisons, new labels with masks/reasons, OOF fold membership, held-out
predictions, models and `results.json`. Numeric results must be copied from an
actually completed run; launching a build does not validate the method.
