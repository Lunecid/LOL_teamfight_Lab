# Claim–evidence ledger — Delta V journal working draft, 16 September 2026

Every numeric claim in [manuscript.md](manuscript.md), [reviewer_response_matrix.md](reviewer_response_matrix.md)
and [collaborator_specification.md](collaborator_specification.md) resolves to a row here, and every row names a
source path, the key or table location inside it, the scope of the value, and the claim that value is allowed to
support. §3 is the correction ledger: it lists interpretive statements in the 16 September findings documents that
the present draft deliberately narrows, together with the corrected wording and the reason.

Paths are relative to the repository root `C:/Users/todtj/문서/LOL_Teamfight/` unless an absolute path is given.
Source content hashes for the five evaluation JSON files are recorded in
[numeric_evidence.json](../../outputs/manuscript_integration_20260916/numeric_evidence.json) under
`source_hashes`, and a wider inventory is in
[source_snapshot.json](../../outputs/manuscript_integration_20260916/source_snapshot.json).

---

## 1. Evidence classes

Every assumption, constant and result in the draft carries one of the following classes. A citation is required
only for class **P**. Classes **E**, **O** and **D** are justified by rationale plus sensitivity analysis, never
by a citation that does not actually establish the value.

| Class | Meaning | Example |
|---|---|---|
| **P — published precedent** | An external peer-reviewed or archival source supports the *method or framing*, within the boundary recorded in §4 | Differencing a frozen win-probability estimate across an event window [Maymin2021] |
| **E — empirical estimate from our data** | A number estimated from this project's corpus, with its own uncertainty and its own estimator | `G = 13.7246 s`, `D = 4263.8688 u` |
| **O — operational choice** | A design decision fixed by the researchers; defensible, not discovered | `h = 90 s` primary cap, `T` cut at 4 per side, B40 range `[0.40, 0.60]` |
| **D — game rule or implementation fact** | A property of the game or of the code path, verifiable by reading the rule or the source | Participant order is `(team_id, participant_id)`; champion experience range 1,600 u |

Mixed items are labelled with their dominant class and explained in the row. `R = 1,600 u` is **D-motivated, O-fixed**;
the 15 s onset lead is **O**; neither is a learned constant.

---

## 2. Numeric claim ledger

### 2.1 Corpus, split and eligibility

| # | Claim in draft | Value | Source | Key / location | Scope | Allowable claim |
|---|---|---|---|---|---|---|
| C-01 | Patch split and row counts | TRAIN 74,673 / VAL 74,748 / TEST 60,579 = 210,000 raw matches; valid ΔV rows 566,104; T 113,901; N 452,203 | `docs/TOG_END_TO_END_READINESS_AUDIT_20260915.md` §4; `outputs/full_corpus_training_20260915/eval/counts_and_exclusions.json` | audit §4 table; stored counts file | Main corpus, all three caps share the mask | Designated patch roles with reported eligibility. **Not** "210,000 matches were trained on" |
| C-02 | Per-split cohort rows | TRAIN T 39,605 / N 159,753; VAL 41,315 / 161,855; TEST 32,981 / 130,595 | `outputs/incremental_q_training_20260915/REPORT.md` §3 data-contract table | role/row table | Fitting and evaluation row counts | Exact row counts per role |
| C-03 | Overlap exclusions | 348 rows excluded where next onset ≤ current last kill | audit §4; spec §4.1 | text | Main corpus | A definitional exclusion, not data loss from failure |
| C-04 | V eligibility, reported separately | 208,603 eligible matches (1,396 no eligible query time, 1 winner conflict); final V fitted on 424,160 queries from 74,168 TRAIN matches; main TEST 347,234 queries from 60,190 matches | audit §4; spec §4.1, §10 | text | Value model only | V eligibility is a different filter from engagement eligibility |
| C-05 | Validation role partition | V_CAL 18,579 / V_SELECT 18,890 / Q_CAL 18,710 / Q_SELECT 18,569 raw matches | spec §4.2 | table | Patch 15.15 by match hash | V and q calibration/selection use disjoint matches |
| C-06 | External cohorts | 21,190 matches, 54,662 valid rows: KR 16.13 10,064; KR 16.14 pilot 200; KR 16.15 926; NA1 16.13 10,000; no EUW1 | audit §4; spec §4.3 | tables | External evaluation only | Transfer evidence with prior-exposure history. **Not** untouched confirmation |

### 2.2 Definition constants

| # | Claim | Value | Class | Source | Allowable claim |
|---|---|---|---|---|---|
| C-07 | Temporal gap `G` | 13.7246 s (op. 13.7); bootstrap CI 12.1601–15.6675 s; stability plateau ≈10–18 s | **E** | `C:/Users/todtj/PycharmProjects/LOL_teamfight/config/fight_boundary/spec_pooled.json`; audit §5 D02–D03 | A pooled KDE-valley estimate with its own interval. Not an externally validated boundary and not Halfaker's estimator |
| C-08 | Spatial diameter `D` | 4263.8688 u (op. 4264); CI 4256.0475–4273.9677 u | **E** | same; audit §5 D04–D05 | Champion-sharing crossing at 0.5, interpolated between 750 u bins; a narrow CI is not fine physical resolution |
| C-09 | Presence radius `R` | 1,600 u | **D-motivated, O-fixed** | audit §5 D06 | Motivated by champion-death experience range. Not proof of combat participation range and **not a learned constant** |
| C-10 | Onset lead `B` | 15 s (`s = K − 15 s`) | **O** | audit §5 D07 | Operational offset informed by assist-time scale and implementation precedent. **Not a learned constant** |
| C-11 | Scale cut | `T` = recorded min cluster participation ≥ 4 per side; `N = E \ T`, disjoint | **O** | audit §5 D09; spec §5 | v3.3 convention; 4 is not an estimated optimum; old `cut = 3` is not used |
| C-12 | Pooled estimation record | 208,141 matches, 10,417,458 gaps, 5,460,008 close kill pairs; per-patch G 13.964/13.490/13.725 s, D 4285.26/4265.27/4241.47 u | **E** | audit §5; spec §5 | Estimation scale and per-patch drift. Some steps use seeded samples; not "every bootstrap refits every gap" |

### 2.3 Value model and label

| # | Claim | Value | Source | Key / location | Allowable claim |
|---|---|---|---|---|---|
| C-13 | Primary V and its metrics vs `W` | Logistic on expanded StateV2, `C = 0.01`, raw calibration; main TEST AUC 0.858621, Brier 0.153045, log loss 0.456742, calibration slope ≈0.976 | audit §7; spec §10 | tables | Final-match-victory prediction quality. **Not** comparable with q metrics (different target) |
| C-14 | V early-window AUC | ≈0.6741 in minutes 2–10 | audit §7 | table | Overall V AUC is not uniform across match time |
| C-15 | V under transfer | External AUC ≈0.84–0.86, calibration slope ≈0.81–0.87 | audit §7; spec §10 | tables | Calibration degrades under patch/region shift |
| C-16 | Endpoint rule | `e(h) = min{L+h, J−1 ms, S_next−1 ms, T_end−1 ms}`; `Y = 1[ΔV > 0]`; h90 primary | audit §8; spec §8 | formulas | Objectives do **not** stop the interval; next global kill does; `S_next` is retrospective, so this is not a live stopping rule |
| C-17 | Endpoint behaviour, main TEST h90 | mean 45.493 s / median 39.440 s follow-up; 14.286% reach cap; 129,036/163,576 stop at next kill; 35.145% no new frame after L (**all 163,576 valid rows**, `outputs/tog_readiness_audit_20260915/calculations.json` → `labels/MAIN_TEST/horizons/90/no_new_frame_after_last_kill_fraction` = 0.35145131315107353); \|Δ\|≤0.005/0.01/0.02 = 7.504/12.696/20.905% | audit §8 recalculation table; calculations.json | table / JSON pointer above | Descriptive properties of the stored labels; the small-Δ band is a diagnostic stratum, not an error bar or exclusion rule. The 35.145% is an **all-valid-row** figure and must not be quoted as a `T` figure (see C-17b) |
| C-17b | Two distinct frame-staleness diagnostics | (a) no new frame after last kill `L`: 35.145% of all valid main-TEST h90 rows, 35.27% of `T`; (b) pre and post states reading the **same** frame: 2,224 / 32,981 `T` rows = 6.74% | (a) audit §8; `docs/LABEL_VALIDITY_FINDINGS_20260915.md` §for T; (b) `outputs/v_mechanism_20260916/REPORT.md` MAIN_TEST T 동일 프레임 row and `results.json` strata | table rows | (b) is strictly narrower than (a) and is the only one that removes frame-based column changes between the two query times. Neither removes elapsed-time/`×time` changes. **Never equate the two, and never say event columns alone determine the label** |
| C-18 | Cap sign disagreement | h60↔h90 1.545%; h90↔h120 0.608%; h60↔h120 1.909% | audit §8; spec §8.1 | table | Endpoint sensitivity of the label; a different experiment from changing V |
| C-19 | Basis for 90 s | CIF before an additional kill 11.93 / 13.21 / 13.75% at 60/90/120 s (+0.538 pp from 90 to 120); 55,734 of 149,002 (37.4%) first acquisitions after 60 s | audit §8.1; spec §8.1 | text | Rationale for an operational cap (**O**). Competing-risk methodology does not fix 90 s, and 37.4% is a conditional denominator |
| C-20 | Alternative-V label sensitivity | `B_reg` (C = 0.1) changes 1.09% of T labels; `B_econ` changes 20.09%; objective-column removal changes 5.358% of T labels (equal-match 5.239%, CI 4.982–5.480%) | spec §13.1, §14.2 (`docs/LABEL_VALIDITY_FINDINGS_20260915.md`, `docs/OBJECTIVE_CHANNEL_ABLATION_FINDINGS_20260915.md`) | tables | Dependence of the label on the valuation model. Each alternative is still a learned model inducing its own label |

### 2.4 Incremental prediction (§8.1 of the manuscript)

All rows: `outputs/incremental_q_training_20260915/eval/results.json`
(sha256 `fdd5da60cdd62c174b08dd2c28a19019fb3237411a50faa33b44e97d24d3dbd3`), main TEST 15.16, h90, equal match
weight. Independent verification receipt: `outputs/claude_dispatch_incremental_q/final_acceptance.json`
(117 checks, 0 failed, 2,961 independent metric comparisons, `unchanged_results_sha256` identical).

| # | Claim | Value | JSON key | Scope | Allowable claim |
|---|---|---|---|---|---|
| C-21 | T PT baseline | Brier 0.22951826, AUC 0.66479497, 32,981 rows | `results/MAIN_TEST/T/metrics_named/all/pt_winner` | T all | Reinforced baseline level. PT includes `p_pre` **and** time splines/tensor interactions — it is not `p_pre` alone |
| C-22 | T full logistic | Brier 0.22840488, AUC 0.66941522 | `.../all/logit_winner` | T all | Matched-input linear level |
| C-23 | T full LightGBM | Brier 0.22839121, AUC 0.66912530 | `.../all/lgbm_winner` | T all | Matched-input tree level; also the overall Q_SELECT winner |
| C-24 | T prior specialist | Brier 0.22817097, AUC 0.67018840 | `.../all/old_A_specialist` | T all, reference column | A different budget and selection history. **Not** the winner of the matched Track A selection |
| C-25 | T primary contrast | LightGBM − PT ΔBrier −0.00112706 [−0.00158846, −0.00064729] | `results/MAIN_TEST/T/bootstrap/all/pairs/0` | T all | Primary contrast **predeclared in this follow-up's protocol/frozen manifest**; interval excludes 0. No external preregistration record exists, and prior TEST exposure preceded the run, so write "predeclared", never "pre-registered" |
| C-26 | T logistic contrast | logistic − PT −0.00111339 [−0.00172423, −0.00050946] | `.../bootstrap/all/pairs/1` | T all | Planned contrast; interval excludes 0 |
| C-27 | T tree vs linear | LightGBM − logistic −0.00001367 [−0.00049137, +0.00044983] | `.../bootstrap/all/pairs/2` | T all | Inconclusive, **not** equivalence |
| C-28 | T B40 cell size | 4,949 rows / 4,570 matches (≈15.0% of T rows) | `results/MAIN_TEST/T/metrics_named/B40/*` `rows`; REPORT §6.1 | T, `p_pre ∈ [0.40, 0.60]` | Cell size must be reported with every B40 claim |
| C-29 | T B40 primary contrast | LightGBM − PT −0.00043498 [−0.00144241, +0.00060074] | `.../bootstrap/B40/pairs/0` | T B40 | Inconclusive **for this learner in this cell**. Do not generalise to all full-input models |
| C-30 | T B40 logistic contrast | logistic − PT −0.00156852 [−0.00305588, −0.00012677] | `.../bootstrap/B40/pairs/1` | T B40 | Favours logistic in an exploratory secondary contrast; no multiple-comparison adjustment |
| C-31 | T B40 vs prior specialist | new LightGBM − old specialist +0.00172689 [+0.00035513, +0.00314621] | `.../bootstrap/B40/pairs/4` | T B40 | Prior specialist better in this cell only; different budgets |
| C-32 | N levels | PT 0.24503980 / logistic 0.24268966 / LightGBM 0.23868813; AUC 0.57786478 / 0.59826983 / 0.62113955; 130,595 rows | `results/MAIN_TEST/N/metrics_named/all/{pt,logit,lgbm}_winner` | N all | Auxiliary cohort; larger increment than T |
| C-33 | N contrasts | LightGBM − PT −0.00635167 [−0.00681807, −0.00591919]; logistic − PT −0.00235014; LightGBM − logistic −0.00400153 | `.../N/bootstrap/all/pairs/{0,1,2}` | N all | Intervals exclude 0 |
| C-34 | N B40 | 37,675 rows; LightGBM − PT −0.00629938 [−0.00702698, −0.00555361] | `.../N/metrics_named/B40/*`, `.../N/bootstrap/B40/pairs/0` | N B40 | Interval excludes 0 |

### 2.5 Matched-input learner families (§8.2)

Source: `outputs/track_a_mlp_20260916/eval/results.json`
(sha256 `d4d4e4f3d7324f4b28fff594dc95e20527cca62afaf1072c658a0e3f9fdc2d45`); narrative
`docs/TRACK_A_MLP_FINDINGS_20260916.md`; report `outputs/track_a_mlp_20260916/REPORT.md`.

| # | Claim | Value | JSON key | Allowable claim |
|---|---|---|---|---|
| C-35 | T plain MLP | Brier 0.22846091, AUC 0.66941480 | `results/MAIN_TEST/T/metrics_named/all/mlp_winner` | Matched-input network level |
| C-36 | T residual MLP | Brier 0.22899915, AUC 0.66676157 | `.../all/resmlp_winner` | Matched-input residual network level |
| C-37 | T primary contrast | residual MLP − LightGBM +0.00060794 [+0.00011657, +0.00109890] | `results/MAIN_TEST/T/bootstrap/all/pairs/0` | Residual MLP worse in this setup |
| C-38 | T plain vs tree | plain MLP − LightGBM +0.00006971 [−0.00045972, +0.00056998] | `.../pairs/3` | Inconclusive, not equivalence |
| C-39 | T plain vs linear | plain MLP − logistic +0.00005603 [−0.00031588, +0.00043189] | `.../pairs/1` | Inconclusive |
| C-40 | T residual vs plain | +0.00053823 [+0.00017134, +0.00088601] | `.../pairs/2` | Plain better than residual |
| C-41 | T B40 plain vs tree | −0.00099580 [−0.00240908, +0.00026496] (ΔAUC +0.016 [+0.002, +0.033]) | `.../bootstrap/B40/pairs/3`; findings §2 | Inconclusive on Brier |
| C-42 | N contrasts | residual − LightGBM +0.00110298 [+0.00079374, +0.00144766]; plain − LightGBM +0.00109090 [+0.00075453, +0.00142742]; plain − logistic −0.00291063 [−0.00328111, −0.00249496] | `results/MAIN_TEST/N/bootstrap/all/pairs/{0,3,1}` | Tree better than both networks on N; both networks better than the linear model |
| C-43 | N B40 | residual − LightGBM +0.00184780 [+0.00122861, +0.00241392] | `.../N/bootstrap/B40/pairs/0` | Tree better in the balanced N cell |
| C-44 | Selected configurations | width 128 in all four selections; residual MLP best epoch 1/1/1 on T; seed spread raw Brier 0.2295–0.2300 (residual) vs 0.2287–0.2295 (plain); 0 runs hit the 100-epoch cap | findings §1–§4; `outputs/track_a_mlp_20260916/protocol.json` | Consistent with rapid overfitting; **not proof** of it under a fixed lr/batch/patience |
| C-44b | Per-family budget | Six candidate configurations per family (six `C` values for the logistic; 3 widths × 2 dropout rates per MLP; 3 leaf counts × 2 minimum-child sizes for LightGBM); three-seed averaging (7/42/123) for the **stochastic** LightGBM and MLP families; 18 joint candidates per MLP family (6 configurations × 3 calibrations) | `outputs/track_a_mlp_20260916/protocol.json` (`seeds`, `prediction`, `grid`, `n_candidates`, selection `rule`) | Write "six candidate configurations per family" and restrict three-seed averaging to the stochastic families. The regularised logistic is **not** a three-seed ensemble |
| C-45 | External, T | residual MLP − LightGBM contains 0 on KR 16.13/16.14/16.15; NA1 16.13 T favours LightGBM (+0.00285) | findings §3 | No confirmed external teamfight gain for networks |

### 2.6 Horizon sensitivity (§8.3)

Source: `outputs/horizon_sensitivity_20260916/eval/results.json`
(sha256 `3daa885346c420fcf4bf9bee8d768e251fdc81247b484ae41ff293bcd39e0e07`); narrative
`docs/HORIZON_SENSITIVITY_FINDINGS_20260916.md`.

| # | Claim | Value | JSON key | Allowable claim |
|---|---|---|---|---|
| C-46 | T Brier levels by cap | LightGBM 0.22781/0.22839/0.22825; logistic 0.22789/0.22840/0.22831; plain MLP 0.22775/0.22846/0.22845; residual 0.22847/0.22900/0.22891; PT 0.22903/0.22952/0.22938 (h60/h90/h120) | `results/h60/MAIN_TEST/T/metrics_named/all/*` and `results/h120/MAIN_TEST/T/metrics_named/all/*`, plus parent h90 columns | Same rows, different labels, each with its own prevalence. **Absolute levels are not comparable across caps**, a lower h60 Brier does **not** establish an easier task, and no cross-horizon superiority is claimed |
| C-47 | h60 primary contrast | LightGBM − PT −0.00122011 [−0.00166578, −0.00075746] | `results/h60/MAIN_TEST/T/bootstrap/all/pairs/0` | Pattern preserved at 60 s |
| C-48 | h120 primary contrast | LightGBM − PT −0.00113193 [−0.00162562, −0.00063166] | `results/h120/.../pairs/0` | Pattern preserved at 120 s |
| C-49 | Residual MLP at both caps | +0.00066272 [+0.00018590, +0.00113266] (h60); +0.00066511 [+0.00017134, +0.00120004] (h120) | `results/h{60,120}/MAIN_TEST/T/bootstrap/all/pairs/4` | Residual MLP worse at every cap |
| C-50 | Point-estimate ranking moves | At h60 the plain MLP has the lowest T Brier (0.22775), while at h90 it is behind LightGBM | `results/h60/MAIN_TEST/T/metrics_named/all/mlp_winner` | The **particular comparisons** of C-47–C-49 persist across caps; learner ordering is **not** invariant, so never write "the ordering is unchanged across h60/h90/h120" |
| C-51 | Configuration constraint | Hyperparameters copied from h90 winners; only calibration re-selected; iteration/epoch counts re-determined by the stop rule | findings §1 | Not a per-cap best-configuration comparison; no horizon chosen on TEST |

### 2.7 Definition sensitivity (§8.4)

Source: `outputs/definition_dev_20260916/eval/results.json`
(sha256 `ad80fcd3f0c08f25f78b7358e0581d20092ad51e780318031392034d7416104c`),
`outputs/definition_dev_20260916/census.json`; narrative `docs/DEFINITION_DEV_FINDINGS_20260916.md`.

| # | Claim | Value | Key | Allowable claim |
|---|---|---|---|---|
| C-52 | Dev-only constants | G 14000 ms, D 4285.0 u (from 15.14-only estimates 13.96 s / 4285 u) | `census.json: definition` | A TRAIN-patch-only alternative. `D = 4285` lies **outside** the pooled CI [4256.0475, 4273.9677]; `G = 14.0 s` lies inside the plateau and the G CI |
| C-53 | Changed matches | TRAIN 3.30% (2,467); VAL 3.47% (2,593); TEST 3.54% (2,143); KR 16.13 4.05%; NA1 16.13 4.01%; KR 16.14 pilot 1.5% | `census.json: sets.*.matches_changed(_frac)` | Report as "3.3–4.4% of matches changed", never as an unqualified "2% changed" |
| C-54 | Exposure rows removed/added | TRAIN 3,690 (1.85%) / 2,303 (1.15%); VAL 3,892 (1.91%) / 2,419 (1.19%); TEST 3,188 (1.95%) / 2,013 (1.23%) | `census.json: sets.*.rows_removed, rows_added, rows_parent` | Removal and addition denominators are reported separately |
| C-55 | Valid T rows | TEST 32,981 → 32,968, of which 1,751 rebuilt | `census.json`; results `rows` field | Population change is small but real |
| C-56 | Frozen metrics on dev population | Main TEST T: LightGBM 0.228391 → 0.22837098 (AUC 0.66913629); logistic 0.22833845; PT 0.22950308. External frozen cells move further: KR 16.15 N 0.24062 → 0.23996 (≈0.0007), KR 16.13 T 0.23510 → 0.23476 | `results/MAIN_TEST/T/metrics_named/all/frozen_*`; REPORT §4 external table | Main-TEST frozen Brier changes in recorded all/B40/B45 cells are below 0.00007; AUC changes reach about 0.00075 (root check: `outputs/manuscript_integration_20260916/definition_metric_scope_check.json`). External Brier changes can be larger. These are descriptive comparisons across different row populations, not a paired effect or an all-metric bound |
| C-57 | Primary contrast | T refit − frozen LightGBM −0.00006839 [−0.00022643, +0.00009740] | `results/MAIN_TEST/T/bootstrap/all/pairs/0` | **Inconclusive**, not equivalence; only one refit per arm, so refit-to-refit variability is unestimated |
| C-58 | N contrast | refit − frozen +0.00022501 [+0.00011128, +0.00033941] | `results/MAIN_TEST/N/bootstrap/all/pairs/0` | Refit slightly worse on N; not dismissed as noise |
| C-59 | External | KR 16.13 T refit − frozen +0.00052 [+0.00010, +0.00099]; other cells contain 0 | findings §4 | One external cell worsens |
| C-60 | Rebuilt-row difficulty | T Brier 0.232 vs 0.228 common rows; AUC 0.655 vs 0.669; ≈5% of T | findings §3 | Merged/split engagements are harder; too few to move the population metric |
| C-61 | Mechanical checks | Parent detector with frozen constants reproduces `exposures.csv` exactly on 2,000 TRAIN matches (5,247 rows, 0 mismatches); 100-match / 250-row rebuild bit-identical to parent inputs, p_pre, p_post, labels | findings §1 | Re-detection machinery is faithful; independent of the interpretive claim |

### 2.8 Champion class and slot pairs (§8.5)

Source: `outputs/champion_class_20260916/eval/results.json`
(sha256 `9b4568ef2dc526f6a815e05d029cd736a74b4ee581e5311cd5c50895373d839d`); narrative
`docs/CHAMPION_CLASS_FINDINGS_20260916.md`; positional audit
`outputs/manuscript_integration_20260916/position_audit.md`.

| # | Claim | Value | JSON key | Allowable claim |
|---|---|---|---|---|
| C-62 | Predeclared primary (champion class) | T LightGBM class_pairs − base +0.00012876 [−0.00008707, +0.00035203] | `results/MAIN_TEST/T/bootstrap/all/pairs/12` | **No detected benefit in the tested setup.** Not equivalence, not absence of learnable information |
| C-63 | N primary | +0.00026502 [+0.00004035, +0.00046602] | `results/MAIN_TEST/N/bootstrap/all/pairs/12` | Small worsening on N |
| C-64 | Identity arm | T +0.00002189 [−0.00014302, +0.00018379]; N +0.00014929 [−0.00004288, +0.00033804] | `.../pairs/13` (T and N) | Champion identity adds nothing detectable for the tree |
| C-65 | Class-state arm | T +0.00032670 [+0.00004087, +0.00063875] | `results/MAIN_TEST/T/bootstrap/all/pairs/11` | Slight worsening on T |
| C-66 | Logistic degradation | T +0.00047 (tags) → +0.00259 (class pairs) → +0.00675 (identity) → +0.00760 (both) | `results/MAIN_TEST/T/bootstrap/all/pairs/{0,2,3,4}` | Confounded by a **fixed** `C` from the 352-column winner as width grows; a design limit, not an independent finding |
| C-67 | Draft-only arms | T class 0.24998 AUC 0.50462828; T identity 0.24997675 AUC 0.50855685; N 0.51027747 / 0.52321592 | `results/MAIN_TEST/{T,N}/metrics_named/all/lgbm_draft_*`, `logit_draft_*` | Near-chance **under this representation and budget**. Not proof of zero information |
| C-68 | External | KR 16.13 N +0.00125 [+0.00072, +0.00178]; NA1 16.13 N +0.00063 [+0.00003, +0.00120]; all T cells contain 0 | findings §2 | No external improvement; two N cells worse |
| C-69 | Positional proxy | StateV2 sets `participant_order = sorted(team_map, key=(team_id, participant_id))` and replaces incoming `meta.role_slots`; the class arm reads only the ten slot columns | `worktrees/engagement-state-value/gameplay/state_value_v2.py:L38-L45`; `scripts/cc20260916_common.py:L53-L64,L151-L165,L183-L249`; `outputs/full_corpus_training_20260915/protocol.json:L363-L367` | Features are **slot-index pairs under a positional proxy assumption**. `cc_matchup_TOP` is not a verified same-role matchup |
| C-70 | Order-disagreement rate | 3,548 of 210,000 matches (1.69%): TRAIN 1,255 (1.68%), VAL 1,251 (1.67%), TEST 1,042 (1.72%); external KR 16.13 3, others 0 | `outputs/full_corpus_training_20260915/eval/counts_and_exclusions.json` rows listed in position_audit.md | A match-level reorder indicator. Agreement in the other 98.3% does **not** establish per-slot role correctness |
| C-71 | Weak role supervision | Cached `meta.role_slots` derived from `teamPosition`/`individualPosition` with participant-ID fills; 149,346 teams recorded, 149,162 used, 184 detectable fill teams; accuracy not validated | `outputs/cohort_role_training_20260915/role_supervision_provenance.json:L1-L36, L252-L256` | Weak proxy with documented fill and coverage limits |
| C-72 | Numerical reproduction | LightGBM base reproduces the parent exactly (max diff 0); logistic raw differs ≈6e-7 (T) to 3.4e-5 (N); N isotonic max difference 0.0118 at step boundaries | findings §5; `outputs/champion_class_20260916/audit_followup/POSTRUN_CHECK_CORRECTION_RECEIPT.json` | Must be preserved whenever logistic arms are compared. Same-run comparisons are not automatically immune to numerical uncertainty |

### 2.9 Label mechanism (§8.6)

Source: `outputs/v_mechanism_20260916/results.json`, `outputs/v_mechanism_20260916/REPORT.md`; narrative
`docs/V_MECHANISM_FINDINGS_20260916.md`.

| # | Claim | Value | Allowable claim |
|---|---|---|---|
| C-73 | Identity verified | `Δlogit = βᵀ(z(S_e) − z(S_pre))`; measured max block-sum deviation 3.042e-14 (`results.json: checks/max_abs_sum_minus_decision` = 3.042011087472929e-14); measured max logit-difference deviation 6.231e-12 (`max_abs_logit_diff_minus_decision` = 6.231459792616079e-12); `1[Δlogit > 0] = Y` on 366,746/366,746 rows (TEST 163,576 + VAL 203,170) | Exact arithmetic identity for the frozen raw-calibrated logistic V. Quote the measured values or the bounds 4e-14 / 7e-12; "≤3e-14" and "≤6.2e-12" are false. Use `1[Δlogit > 0] = Y`, not `sign(Δlogit) = Y`, because `Y` is 0/1 |
| C-74 | Champion cancellation | The frozen `V` preprocessing yields `transformed_columns = 1733`, of which `group_sizes/champion_identity = 1382` (351 numeric + 1382 = 1733); that champion block contributes exactly 0 (`checks/max_abs_champion` = 0.0). Source: `outputs/v_mechanism_20260916/results.json`, `protocol.json`, `frozen_manifest.json`; grouping rule `scripts/vd20260916_common.py:column_group` (`champions__` prefix) | Static additive champion terms cancel from `Δlogit` and its sign. They can still affect `p_pre` and hence \|ΔV\| through the sigmoid, and dynamic champion effects through the realised state remain possible. Cite the recorded 1,382-column champion block with this source, or omit the count; **1,372 is not a verified figure** |
| C-75 | T block shares | combat/survival 35.5%, economy/xp 28.2%, objectives 18.9%, structures 9.6%, health/other 7.2%, time-only 0.7%; time-related columns incl. `×time` 19.9%; mean \|Δlogit\| 0.961, mean \|ΔV\| 0.113 | Shares of absolute contribution in a fitted linear score, **not** causal shares and not exchange prices. Block shares and the time-related figure overlap and must not be summed |
| C-76 | Sign structure | Largest block agrees with `Δlogit` sign in 95.9%; a single block exceeds \|Δlogit\| in 27.7%; N combat/survival 45.2% | Descriptive structure of the decomposition |
| C-77 | Strata | **Identical pre/post frame** rows, 2,224 of 32,981 T (6.7%): combat 72.2%, objectives 13.6%, structures 13.7%, frame blocks 0, time-only 0.5%, time-related 25.0%. Small \|ΔV\| ≤0.5 pp (3,251 rows): structures 19.3%, against 9.6% over all T (C-75) and 7.9% in the \|ΔV\| > 2 pp stratum. After 30 min: objectives 21.9%, time-related 27.5% | Where the label comes from in specific strata. The stratum condition is **identical pre and post frames**, not "no new frame after the last kill" (C-17b), and even there event columns do not decide the label alone — elapsed-time and `×time` terms still change. The 19.3% comparator must be named: 9.6% overall or 7.9% in the >2 pp stratum |
| C-78 | Contributions are not coefficients | Top columns by **mean \|c_j\|**: `numeric__blue_kills_x_time` 0.1095 and `numeric__red_kills_x_time` 0.1020; their fitted `β` are **−1.0564** and **+0.9901**; per-participant `totalGold_norm`, `level_norm`, `laneCS_norm` 0.04–0.11 mean \|c_j\| | Report 0.110/0.102 as mean absolute column contributions, **never as coefficients**. Blue and red kill-interaction coefficients have opposite signs, and main kill terms also differ by side, so do not generalise "positive main, negative interaction". Collinear ridge coefficients under `C = 0.01` are not read individually; prefer omitting detailed signs |
| C-79 | Not done | `V(S(L))` was not stored, so the pre→L→endpoint decomposition was **not** computed | Only the pre/post decomposition is verified |

### 2.10 Explanation (§8.7)

Source: `outputs/balanced_shap_20260916/shap/shap_summary.json`,
`outputs/balanced_shap_20260916/REPORT.md`; narrative `docs/BALANCED_SHAP_FINDINGS_20260916.md`.

| # | Claim | Value | Key | Allowable claim |
|---|---|---|---|---|
| C-80 | Protocol | 7 groups (sizes 1/2/60/68/176/24/21), 128 coalitions, background 128 cohort TRAIN rows, 256 hash-selected rows per cell | `groups`, `group_sizes`, `coalitions`, `models.final_q.T.cells.*.explained_rows` | Exact group Shapley **conditional on the sampled rows and background**, not population inference |
| C-81 | Additivity | max abs additivity error ≈3.33e-16; `Σφ + base` reproduces the parent sealed prediction with error <4e-16 | `models.final_q.T.cells.all.checks` | Implementation correctness of the attribution |
| C-82 | T shares, all vs B40 | `p_pre` 0.0971 [0.090, 0.105] = 61.7% → 0.0216 [0.020, 0.023] = 27.7%; economy 0.0189 → 0.0190 (12.0 → 24.4%); objectives 0.0151 → 0.0135 (9.6 → 17.3%); combat 0.0127 → 0.0108 (8.1 → 13.9%); structures 4.8 → 8.4%; health/other 3.9 → 8.2%; time 0.0% | `models.final_q.T.cells.{all,B40}.global_mean_abs` | Relative shares rise in B40 **mostly because the absolute `p_pre` contribution falls**, partly by construction. Group mean \|φ\| are not shares of total absolute ΔV and not exchange prices. The bracketed intervals here are a **descriptive row bootstrap over the 256 explained rows**, not the match bootstrap of §7.3, and the two kinds must never be presented as one class of "95% interval" |
| C-83 | Model dependence | Plain MLP (its T Brier difference from the tree was **not resolved** by the match bootstrap: +0.00007 [−0.00046, +0.00057], C-38 — not "equal Brier") all-T: objectives 21.0%, combat 20.7%, structures 20.0%, economy 17.2%, `p_pre` 14.8%; B40 `p_pre` 4.1% | findings §1 | Different learners distribute attribution differently; shares are not properties of the game |
| C-84 | N direction | `p_pre` 27.7 → 6.5%; economy 21.8 → 31.4%; combat 19.4 → 22.3%; health/other 14.9 → 20.3% | findings §1 | Same direction in the auxiliary cohort |
| C-85 | Cell overlap and means | 2 overlapping rows; mean `p_pre` 0.518 (all) vs 0.503 (B40) | `cells.*.mean_p_pre` | Full vs B40 is a descriptive comparison of different rows with no paired interval |
| C-86 | Masking limits | Masked `p_pre` is not recomputed from the masked state; derived `×time` relations are not restored | findings §2; protocol §6 | Correlated features can produce off-manifold states; row bootstrap is descriptive, not match-clustered population uncertainty |

### 2.11 Verification counts

| # | Claim | Value | Source |
|---|---|---|---|
| C-87 | Post-run check counts | incremental-q 117/117; Track A MLP 115 (run 2; run 1 had 4 check-code failures, receipt preserved); horizon 91; champion class 91 (pass on the third attempt; failures were parent-reproduction tolerances); definition dev 53 (third attempt; failures were a census invariant); balanced SHAP 36; V mechanism 24 | each run's `validation.json` and `audit_followup/` |
| C-88 | Independent acceptance of the parent run | accepted, 117 validation checks, 0 failed, 2,961 independent metric comparisons, results sha256 unchanged | `outputs/claude_dispatch_incremental_q/final_acceptance.json` |
| C-89 | Readiness audit recalculation | 7 sets × 3 caps = 1,863,414 stored row-by-cap endpoint recomputations; 0 endpoint-formula, sign, future-frame or in-interval-extra-kill errors | `docs/TOG_END_TO_END_READINESS_AUDIT_20260915.md` §8; `outputs/tog_readiness_audit_20260915/calculations.json` |

---

## 3. Correction ledger

Each row records an interpretive statement that appears in a 16 September findings document or an earlier
document, why it is too strong, and the wording the present draft uses instead. **The numerical results of those
documents are not changed by any row below**; only the interpretation is narrowed. Contradictory historical text
is preserved in place as history.

| # | Overclaim as written | Where | Why it is too strong | Corrected wording in this draft |
|---|---|---|---|---|
| X-01 | "챔피언 클래스·조합 특징은 q를 개선하지 않는다" as a general statement (title) | `docs/CHAMPION_CLASS_FINDINGS_20260916.md` title | A primary interval containing 0 is a non-detection in one setup, not a general negative | "No detected benefit for LightGBM in this setup"; manuscript §8.5 |
| X-02 | "이 라벨에서는 … 학습할 것도 없다" ("there is nothing to learn") | same, §5 judgement and §4.1 | Absence of evidence under one representation, learner set and budget is not evidence of absence | "Not detected here; learnable composition information is not ruled out"; §8.5 reading |
| X-03 | "같은 정보를 그래프로 표현한 Track B 어댑터가 찾아낼 신호는 없다 — Track B는 조합 목적으로는 만들지 않는 것이 맞다" | same, §4.1 | Track B graph representations were never executed; a tabular null cannot pre-empt an untested representation | "Graph representations, finer classes and other learners were not tested and are not ruled out"; §8.5, §10.4 |
| X-04 | "슬롯 = 포지션이므로 LightGBM은 '미드 슬롯의 레벨·골드'를 이미 분기 조건으로 쓴다" | same, §3 | Slots are `(team_id, participant_id)` order; no per-match role join exists in the arm | "Slot-index pairs under a positional proxy assumption"; §6.1, §8.5, C-69–C-71 |
| X-05 | "상태가 이미 흡수" ("state already absorbs all class effects") | same, §3 | One of several explanations; overfitting, budget and representation were not separated | "Consistent with, but not established by, these results"; §9 |
| X-06 | "5인 조합이 경기마다 고유해 챔피언 단위 시너지는 학습 불가능하다" | same, §3 | Uniqueness of full 5-champion sets does not make synergy unlearnable in general | Removed; only the measured identity-arm null is reported (C-64) |
| X-07 | "세부 클래스(13종)로 내려가도 결과가 바뀔 근거는 없다" | same, §4.3 | Finer taxonomies were not tested | "Not tested and not ruled out"; §10.4 |
| X-08 | Draft-only AUC ≈0.50 read as "라벨 정보가 없다" | same, §1 and §3 | Near-chance AUC under a fixed logistic `C` and limited tree budgets is not proof of zero information | "Near chance under this representation and budget"; C-67 |
| X-09 | "정의 산출에 TEST 패치가 들어갔다는 우려는 실측으로 닫힌다" ("the concern is closed by measurement") | `docs/DEFINITION_DEV_FINDINGS_20260916.md` §5 | One alternative constant pair bounds the consequence; it does not remove the original TEST dependence, and `D = 4285` lies outside the pooled CI | "Bounds but does not remove"; §8.4 reading, §10.5 |
| X-10 | "교전의 2 %가 바뀐다" as the headline change figure | same, §5 | Conflates removed and added exposure rows, and understates the match-level change | "3.3–4.4% of matches changed; removed 1.85–2.4% and added 1.15–1.5% of exposure rows, reported separately"; C-53–C-54 |
| X-11 | Refit − frozen `−0.00007 [−0.00023, +0.00010]` read as "구별되지 않는다" (indistinguishable) | same, judgement paragraph | Inconclusive is not equivalence; the N cell and one external cell worsen with intervals excluding 0; only one refit per arm exists | "Inconclusive; N and one external cell worsen; refit variability unestimated"; C-57–C-59, §10.5 |
| X-12 | "residual MLP는 즉시 과적합한다" as a mechanism claim | `docs/TRACK_A_MLP_FINDINGS_20260916.md` §4 | An early best epoch under one fixed optimisation budget is evidence consistent with overfitting, not a demonstration | "Consistent with rapid overfitting but not proof"; §8.2, C-44 |
| X-13 | "동일한 352입력에서 신경망은 LightGBM을 넘지 못했다" as a statement about neural networks | same, judgement | Two tabular network forms, six configurations each, one optimiser setting | "Within this bounded comparison of five tabular families"; §8.2, §10.4 |
| X-14 | Balanced-SHAP relative-share increase read as more objective/economic influence in balanced states | `docs/BALANCED_SHAP_FINDINGS_20260916.md` judgement | Absolute non-`p_pre` contributions are essentially unchanged; the share moves because `p_pre` falls, partly by construction of the cell | "Mostly reflects a diminished `p_pre` contribution"; §8.7, C-82 |
| X-15 | Group mean \|φ\| read as shares of the total change or as exchange values | §8.6/§8.7 reading generally | Attribution of `q`'s output is not a decomposition of \|ΔV\| and not a price | Explicitly separated in §8.6 (label decomposition) and §8.7 (predictor attribution) |
| X-16 | B40 LightGBM − PT inconclusive read as "full-input models do not help in balanced states" | cross-document risk | The logistic contrast in the same cell excludes 0 in the opposite direction | "Do not generalise the LightGBM B40 result to all full-input models"; §8.1, C-29–C-30 |
| X-17 | Prior cohort specialist's better T point estimate read as a winner | cross-document risk | Different candidate pool, budget and selection history | "Not the winner of the matched Track A selection"; C-24 |
| X-18 | Treating any 2026-09-16 follow-up as confirmatory because it was frozen | cross-document risk | Main TEST and external cohorts were already exposed before these runs were specified | "Exploratory follow-up after prior TEST exposure"; §7.5 |
| X-19 | Calling label semantic validity an open blocker requiring human review | planning documents | The user declined manual review; it is an open validity gap, not a new mandatory prerequisite | "Open validity gap, outside current scope by decision"; §10.1 |
| X-20 | Citing Kim et al. as evidence of calibration work performed here | risk from earlier drafts | The uncertainty-aware loss was never implemented | "Motivation only; loss not implemented"; §2.1, references.bib `annote` |

### 3.1 Corrections applied to the **first draft of this bundle** (revision pass, 16 September 2026)

Rows X-21 – X-34 record errors found by the independent scientific review of the first writer pass
(`outputs/manuscript_integration_20260916/independent_scientific_review.md`) and the corrections required by
`revision_request.md`. They correct *this bundle's own earlier text*, not the parent findings. The first-pass
receipt (`writer_receipt.json`) is preserved unchanged as a historical record; resolution is recorded in
`revision_receipt.json`.

| # | First-draft statement | Where it stood | Why it was wrong | Corrected wording now in this bundle |
|---|---|---|---|---|
| X-21 | "A tie among three families", "statistically indistinguishable", "learner family choice is largely immaterial" | manuscript abstract, §1.3, §8.2, §8.7, §9, §11.1; matrix R2-2, CE-3; spec §11.2, §12.2 | Intervals containing zero do not show equality; no equivalence test was run, and the residual MLP is measurably worse | "The specified contrast's interval includes zero; equivalence was not tested"; named pairwise contrasts only |
| X-22 | "The ordering is unchanged / stable across h60/h90/h120" | manuscript abstract, §1.3, §8.3, §9, §11.1; matrix R2-2; spec §11.3 | Exact rankings move (plain MLP has the lowest T point estimate at h60; dev frozen logistic vs tree changes) | "The particular comparisons named are retained; exact point-estimate rankings move" |
| X-23 | "The lower T values at h60 mean the 60 s label is easier to predict" | manuscript §8.3; spec §11.3 | Different caps induce different targets and prevalences; no difficulty analysis on a common target exists | Inference removed; only non-comparability of absolute levels is stated |
| X-24 | "Sealed metrics stay within 0.0002" as a general statement | manuscript abstract, §8.4, §10.5, §11.1; spec §16.3 | Not an all-metric bound even on main TEST (AUC changes reach ≈0.00075); external frozen Brier changes reach ≈0.0007, and N/KR 16.13 T refit contrasts are worse with intervals excluding zero | Cell-by-cell figures with the worsened cells named; TEST-dependence retained as a limitation (C-56–C-59) |
| X-25 | "In the third of cases where no new frame arrives after the last kill the event columns alone decide the sign" | manuscript §8.6, §9, §10.3; matrix R2-3; spec §12.3 | Conflates 35.145%/35.27% (no new frame after L) with 2,224/32,981 = 6.74% identical pre/post frames; even there, elapsed-time and `×time` terms change | Two diagnostics separated everywhere; "event columns together with elapsed-time terms" (C-17b, C-77) |
| X-26 | "Unlike a hand-weighted exchange score, this label can be decomposed exactly" | manuscript §9 | A hand-weighted sum is also exactly decomposable | The distinction is learned coefficients plus an explicit state/window definition |
| X-27 | "Top individual coefficients `blue/red_kills_x_time` at 0.110 / 0.102 … positive main effect and negative `×time` interaction" | manuscript §8.6; ledger C-78 | 0.110/0.102 are mean absolute contributions; the coefficients are −1.0564 / +0.9901 and differ in sign by side | Quantity renamed; side-sign generalisation removed (C-78) |
| X-28 | "Structures rise to 19.3% from 7.9% overall" | manuscript §8.6; ledger C-77 | 7.9% is the \|ΔV\| > 2 pp stratum; overall T is 9.6% | Comparator named explicitly |
| X-29 | "`sign(Δlogit) = Y`"; "block sums to ≤3e-14 / ≤6.2e-12"; "1,372 champion one-hot columns" | manuscript §8.6; ledger C-73–C-74; spec §12.3 | `Y` is 0/1 so the indicator form is correct; the measured maxima are 3.042e-14 and 6.231e-12, so the stated bounds are false; the recorded champion block is 1,382 | `1[Δlogit > 0] = Y`; measured values or bounds 4e-14 / 7e-12; 1,382-column block with source |
| X-30 | "The remaining state signal is thin — too thin to resolve in 4,949 rows" | manuscript §9 | Restricts to LightGBM only; the logistic − PT interval in the same cell excludes zero | Inconclusiveness restricted to LightGBM-PT; the logistic result preserved as exploratory with a multiplicity caveat |
| X-31 | "The `N` gain … smaller engagements are less determined by overall match position" | manuscript §9 | This mechanism was never tested; the cohorts differ in several respects at once | Explanation withdrawn; only the measured difference is reported |
| X-32 | "Pre-registered primary" | manuscript §8.1/§8.5; ledger C-25, C-62; matrix CE-2; spec §11.1, §16.4 | No external preregistration record was verified, and prior TEST exposure preceded the follow-ups | "Predeclared in this exploratory follow-up protocol", with the exposure fact restated |
| X-33 | CoG 308's published exchange score identified with `market_event`; CE-5/CE-6 attributed to the collaborator; "all six reviewer references unverified"; "six configurations and three seeds per family" | manuscript §1.4; matrix R1-2, R2-2, §6, CE-5, CE-6; README §2 | Three distinct lineages; CE-5/CE-6 come from our outgoing reply draft; Schubert2016 was verified and is cited; the logistic is not a three-seed ensemble | Lineages separated; per-row provenance labels; one verified plus five unverified references; six candidates per family with three-seed averaging for the stochastic families only |
| X-34 | Reporting all intervals as one kind; ledger C-46 inline JSON pointer containing an unescaped pipe character | manuscript §7.3, §10.6; ledger C-46 | Match-bootstrap metric intervals, descriptive row bootstraps over 256 SHAP rows and boundary-estimator bootstraps are different objects; the pipe broke the Markdown table row | Three interval classes distinguished; C-46 pointer rewritten as two paths without a pipe |

---

## 4. Citation support limits

Derived from [citation_audit.md](../../outputs/manuscript_integration_20260916/citation_audit.md). Full entries
and the same limits in machine-readable form are in [references.bib](references.bib).

| Key | Supports (narrow) | Does not support |
|---|---|---|
| [Kim2020] | Motivation to assess probability calibration and input-dependent uncertainty in LoL win prediction | Any claim that their architecture, uncertainty-aware loss or Monte Carlo calibration is implemented here; their numbers as ours |
| [Maymin2021] | The time/state logistic win-probability lineage for LoL, and valuing events by change in a frozen estimate | Our schema, population, engagement label, endpoint, `V`, `q`, calibration or causal identification; their instrumentation or corpus |
| [Hodge2021] | Live MOBA win prediction as an established setting with data-availability and temporal limits | Our LoL model, features, calibration, `q`, `ΔV`, live availability of our inputs; their accuracies |
| [Halfaker2015] | Deriving an activity boundary by inspecting inter-activity-time distributions and separating temporal components | `G = 13.7 s`; the KDE-valley estimator; late-game behaviour; any causal claim |
| [Lundberg2017] | Additive model-output attribution principles and the role of background/mapping choices | Causal explanation; explanation of `V` or `ΔV`; that a feature caused a fight outcome |
| [Schubert2016] | Spatio-temporal encounter segmentation and outcome modelling as precedent; the scale of the kill-free encounter population (81.1% in their data) | Our `G`, `D`, `B`, `R`; our kill-conditioned population; that an engagement causes a positive `ΔV` |

Local integration contracts for these boundaries: `docs/TEMPORAL_WINPROB_LITERATURE_ALIGNMENT_20260915.md`
(§§1–3) and `docs/DEFINITION_EVIDENCE_REGISTER_20260914.md` (§§2–5). The original candidate metadata audit is
`C:/Users/todtj/PycharmProjects/LOL_teamfight/docs/tog_manuscript/references_audit.md`.

---

## 5. Statements this draft must not contain

1. That `q` predicts teamfight victory, match-win probability, the magnitude of `ΔV`, a causal effect, or the
   occurrence of a fight.
2. That any interval containing zero demonstrates equivalence or absence of an effect.
3. That champion composition, graph representations, finer class taxonomies or untested learners have been ruled
   out on this target.
4. That slot indices are verified lane roles, or that a slot-pair feature is a role matchup.
5. That the definition constants are universal, learned optima, or independent of the test patch.
6. That any 2026-09-16 result is confirmatory, replicated on untouched data, or a live deployment result.
7. That a group attribution value is a share of `ΔV`, an exchange price, or a causal contribution.
8. Any p-value, significance statement, completed test, human-review result, PDF build or repository release that
   does not exist in the recorded artefacts.
9. CoG 308 or 118 metrics presented as, or compared directly with, `ΔV`-target results; or the published CoG 308
   exchange score identified with the intermediate `market_event` implementation without implementation evidence.
10. That learner families "tie", are "statistically indistinguishable", or that learner choice is immaterial; or
    that learner ordering is unchanged across h60/h90/h120 or across TRAIN-only definition boundaries.
11. That a lower Brier at a different endpoint cap shows an easier task, or that h90 is justified by any TEST
    ranking.
12. That definition re-estimation leaves metrics "within 0.0002" as a general statement, or that the retained
    TEST-dependence of the definition is removed.
13. That "no new frame after the last kill" is the same population as identical pre/post frames, or that event
    columns alone determine the label in any stratum.
14. That exact decomposability distinguishes this label from a hand-weighted score, or that mean absolute column
    contributions are coefficients.
15. "Pre-registered" without qualification; use "predeclared in this exploratory follow-up protocol" and preserve
    the fact that prior TEST exposure preceded these follow-ups.
16. Content from our outgoing collaboration reply draft attributed to the collaborator as an incoming request or
    an externally supplied number.
