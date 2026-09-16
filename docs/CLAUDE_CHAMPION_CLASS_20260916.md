# Champion-class features for q (h90): do Riot's champion classes, class-conditioned team state and position-pair class combinations improve the frozen 352-input predictors?

2026-09-16. Design, implementation, execution and audit: Claude (continuing the Codex line). Exploratory follow-up after
prior TEST exposure (iq, Track A, horizon sensitivity, balanced SHAP, V mechanism). Proposed by the user on 2026-09-16:
"embed champions with Riot's champion classification and learn structures such as 'a fed assassin matters more than a fed
enemy marksman', including jungle-mid, bottom-duo and 5-vs-5 composition, in a way that transfers across patches".
This stage is the tabular first step of that proposal; a graph representation (Track B) is deferred until this stage
shows whether class-level composition carries signal at all.

## Why this is not already learned (facts checked on 2026-09-16)

- The frozen V uses the ten champion ids only as an additive one-hot block (LogisticRegression, no interactions). On
  VALIDATION-region rows the champion block accounts for 2.9 % of the pre-fight logit variance and cancels exactly in the
  label (V mechanism stage). q's 352 ridge inputs exclude `champion_id` by construction (`Q_FORBIDDEN_TOKENS`), so no q
  model has ever seen which champion or class occupies a slot; the only champion information reaching q is the additive
  prior inside `p_pre_V`.
- Composition at champion level does not repeat: 68,712 distinct blue five-champion teams in 69,124 TRAIN matches;
  jungle-mid champion pairs have median count 3 (88 % seen fewer than 30 times). Composition at class level is dense:
  36 jungle-mid class pairs (top pair 28,208 matches), 209 distinct five-class multisets.
- The parent slot order is positional: slot 3 / 8 is a Marksman in 92 % of TRAIN matches, slot 0 / 5 a Fighter in 69 %,
  slot 2 / 7 a Mage in 61 %, slot 1 / 6 Fighter 63 % / Assassin 23 %, slot 4 / 9 Support 41 % / Tank 40 %. Positions are
  therefore TOP, JG, MID, BOT, SUP for slots 0-4 (blue) and 5-9 (red); class is not determined by position.

## Class source (approved download)

Data Dragon `champion.json` for versions 15.14.1 ... 16.15.1 (26 files, public static CDN, no credentials), fetched
2026-09-16 with the user's approval into `outputs/champion_class_20260916/ddragon/<version>/champion.json`;
`fetch_manifest.json` records URL, bytes and sha256 per file; `tag_table.json` is the per-version table
{champion key -> id, name, primary = tags[0], secondary = tags[1] or null} restricted to the six Riot tags
(Assassin, Fighter, Mage, Marksman, Support, Tank). Entries with key >= 60000 (`Jade_*` special-mode champions in 16.15.1)
are excluded. Across the 26 versions the only tag change is Shyvana (secondary Mage -> Tank from 16.6.1); champions added
after 15.14.1 are Zaahen (904, 15.23.1) and Locke (805, 16.13.1). The 171 TRAIN champion ids are all present in 15.14.1.
Each set uses its own patch table: MAIN_TRAIN 15.14.1, MAIN_VALIDATION 15.15.1, MAIN_TEST 15.16.1, EXT_KR_16.13 and
EXT_NA1_16.13 16.13.1, EXT_KR_16.14_pilot 16.14.1, EXT_KR_16.15 16.15.1. The sha256 of `tag_table.json` is pinned in
`protocol.json` and the frozen manifest.

## Feature blocks (all from the pre-fight inputs of the parent `X_input`; no new raw data)

Let prim(s), sec(s) in {0..5, -1} be the primary / secondary tag index of the champion in slot s (-1 when the id is
missing, <= 0 or absent from the patch table). Per-slot state columns used: `totalGold_norm`, `level_norm`, `kills`,
`deaths`, `alive`, `hp_pct` (all already among the 352 ridge inputs).

| block | width | definition |
|---|---:|---|
| `tags` | 120 | per slot s and tag t: 1[prim(s) = t] and 1[sec(s) = t] (`cc_slot<s>_primary_<tag>`, `cc_slot<s>_secondary_<tag>`) |
| `class_count` | 18 | per team and tag: number of slots with prim = t (12), blue minus red (6) |
| `class_agg` | 108 | per team and tag: sum over slots with prim = t of gold, level, kills, deaths, alive; mean hp_pct (0 when the class is absent) (72); blue minus red (36) |
| `pairs` | 396 | ordered class-pair one-hots (36 each): same-team position pairs TOP-JG, JG-MID, BOT-SUP for blue and red (216); same-position cross-team matchups TOP, JG, MID, BOT, SUP (180) |
| `identity` | 1,710 | per slot: one-hot of the champion id over the TRAIN vocabulary (171 ids, pinned in protocol.json); unseen ids are all-zero |

Arms (inputs of one fitted model):

| arm | inputs | width |
|---|---|---:|
| `base` | 352 ridge (refit control; same configuration as the parent winner) | 352 |
| `tags` | ridge + tags | 472 |
| `class_state` | ridge + tags + class_count + class_agg | 598 |
| `class_pairs` | ridge + tags + class_count + class_agg + pairs | 994 |
| `identity` | ridge + identity | 2,062 |
| `class_pairs_identity` | ridge + tags + class_count + class_agg + pairs + identity | 2,704 |
| `draft_class` | tags + class_count + pairs only (no state, no time, no p_pre) | 534 |
| `draft_identity` | identity only | 1,710 |

The `draft_*` arms measure how much composition alone carries; `identity` is the contrast that separates "class" from
"which champion". Nothing is added to V, the labels or the cohorts.

## Families, configurations, selection, freeze

Families `logit` and `lgbm` with the h90 configurations of the iq family winners fixed per cohort (T: logit_C0.001,
lgbm_L15_M100; N: logit_C0.01, lgbm_L15_M100), read from the iq frozen manifest into `protocol.json`. No configuration
search: the extra blocks might prefer another regularisation, which is a stated limitation. LightGBM keeps the three seeds,
the stop10 internal early-stopping allocation and the from-scratch refit; the logistic keeps the imputer -> scaler ->
weighted lbfgs policy. Per cohort x family x arm, calibrators (raw / sigmoid / isotonic) are fit on cohort Q_CAL and the
calibration is selected on cohort Q_SELECT by match-weighted Brier, then log loss, then name. MLPs are deferred (the GPU is
occupied by the old-target search; the plain MLP was indistinguishable from LightGBM on the 352 inputs).

One `frozen_manifest.json` freezes all 2 x 2 x 8 = 32 selections before any TEST / external array or parent TEST
prediction is opened. TRAIN-only smoke (1/8 of TRAIN matches, pseudo roles) exercises every arm first.

## Evaluation (sealed; after the freeze)

MAIN TEST 15.16 and each external set separately, cohorts T and N (h90-valid rows, counts as in the iq specification),
cells all / B40 / B45 / time bins / B40 x time on the frozen p_pre and pre time. Named models: the 16 arm winners
`<family>_<arm>` plus the parent iq winners (`parent_lgbm_winner`, `parent_logit_winner`, joined by exact (match, s_ms);
metrics only, and the maximum absolute difference between the parent winner and this run's `base` arm is reported as a
reproduction check). Paired match bootstrap (1,000 replicates, seed 20260915) on all / B40 / B45 with >= 30 matches and
both classes, over the 16 arm winners, with the planned contrasts per family: every arm minus `base` (7),
`class_pairs` minus `identity`, `class_pairs_identity` minus `identity`, `class_pairs_identity` minus `class_pairs`.

PRIMARY (pre-registered here): cohort T, LightGBM, `class_pairs` minus `base`, cell all, Brier difference on MAIN TEST.
Everything else is secondary / descriptive; no multiplicity adjustment; intervals cover evaluation-sample uncertainty of
fixed models only.

## Checks

Parents unchanged (integrity snapshot before / after); tag table and identity vocabulary pinned; feature blocks recomputed
in the post-run stage equal the matrices hashed at fit time (determinism); every fit uses the cohort TRAIN rows only, own
held-out fold OOF V for `p_pre_V`, equal-per-match weights, Q_CAL calibrators and Q_SELECT selection reproduced from saved
predictions; reload identity of every bundle; sealed access only after the freeze; evaluation metrics, cells and
bootstrap point estimates recomputed from saved arrays; fresh-process reload of the winners on MAIN TEST.

## Not done here

Subclass taxonomy (13 wiki classes), MLP / GPU arms, the graph adapter (Track B), any change to V or the labels, any
causal reading of a class effect, human review. The question answered is only: does class information, at the tabular
level, lower the sealed Brier of the fixed-configuration learners, and does it do so beyond champion identity.

## Layout

`outputs/champion_class_20260916/`: `ddragon/` (approved download + tag table), `protocol.json`, `contract_tests/`,
`integrity/`, `smoke_train_only/`, `models/<cohort>/<family>/<arm>/<config>.joblib`, `selection/<family>_<arm>_<cohort>.json`,
`predictions/<family>_<arm>_<cohort>_trainval.npz`, `internal_stop/`, `frozen_manifest.json`, `eval/results.json`,
`eval/predictions/<set>_h90_<cohort>.npz`, `validation.json`, `REPORT.md`, `DEFINITION_AND_EVIDENCE.md`, `status.json`,
`logs/`, `commands.txt`, `failures.jsonl`, `access_log.jsonl`. Scripts `scripts/cc20260916_*.py`, tests
`tests/test_cc20260916_contracts.py`.
