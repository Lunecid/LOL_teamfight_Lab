# Reviewer-response matrix — prior CoG feedback as input to the journal revision

**Status: working draft, 16 September 2026. Not a response letter and not a submission document.**

## 0. Which review this answers, and which it does not

Three distinct records are involved and are kept apart throughout this file.

| Record | What it is | How it is used here |
|---|---|---|
| **CoG 2026 submission 118** — *"How Predictable Are League of Legends Teamfights Before They Begin? Outcome Prediction Under Minute-Resolution Public Telemetry"* | **Rejected.** 239 regular submissions, 89 accepted. Four review texts (R1 weak accept, R2 weak reject, R3 weak accept, R4 meta-review recommending accept) were supplied with the decision e-mail | **The source of every numbered row in §2–§5.** These are the concerns this journal revision answers |
| **CoG 308** | The separate authoritative **accepted and published** CoG paper, with the manually weighted exchange label and the eight-model comparison | Lineage only. Its label design appears in §6.2 because a collaborator asked about it. Its metrics are never compared with Delta V results |
| **Journal (ToG) review** | **Has not happened.** No journal submission exists | No row below is a response to a journal reviewer. Statuses describe the draft's readiness, not any editor's judgement |

The reviewers of 118 did not see the current target. Between 118 and this draft the outcome label, the engagement
constants, the corpus split, the input contract and the model set all changed, so most rows below are answered by
**redesign** rather than by patching the criticised component. Where a concern is answered by redesign, the row
says so, and says what is still missing on the new target.

Original reviewer text: `C:/Users/todtj/.codex/attachments/68cfc3ee-c632-4cff-a469-80304ef81c41/pasted-text.txt`
(read-only).

## 1. How to read this file

- **Original concern** paraphrases the review in under ~20 words; the source text is at the path above.
- **Implemented change** states what was actually built or written, not what is planned.
- **Evidence** points at a run directory, a JSON key, or a findings document; every number resolves through
  [claim_evidence_ledger.md](claim_evidence_ledger.md).
- **Draft section** points into [manuscript.md](manuscript.md).
- **Remaining condition** states the concrete thing that must exist before the row could be called closed.
- **Status** is one of **completed** (nothing further planned for this row), **partial** (substantive work done,
  a named piece missing), **open** (not addressed on the current target).

No row claims a new experiment, a completed test, a human review, a compiled PDF or a released repository.

### 1.1 The issue IDs below are **new consolidated IDs**

The IDs in §2–§5 are assigned in this file and **do not** carry the meanings of the identically named IDs in the
older matrix at `C:/Users/todtj/PycharmProjects/LOL_teamfight/docs/tog_manuscript/reviewer_response_matrix.md`
(the ID inventory also used by `outputs/manuscript_integration_20260916/manuscript_map.md` §3). For example,
historical R2-1 was the missing anonymised artefact link, whereas new R2-1 is the hand-set label weights and new
R2-4 is the artefact link. Cross-artefact references must state which scheme they use. Crosswalk:

| New ID | Historical ID(s) | Concern |
|---|---|---|
| R1-1 | R1-1 | Abstract reads disjointed |
| R1-2 | R1-2, R1-3 | Contribution statement restates the questions; cite [1]–[6] on community impact |
| R1-3 | R1-4, R1-5 | 18 s / 4,000 u look arbitrary; results should not hinge on them |
| R1-4 | R1-6, R3-6 | Background section defining League of Legends terms; self-contained for non-experts |
| R1-5 | R1-7 | Explain the labelling for less-informed readers |
| R1-6 | R1-8 | Teamfight outcome is hard to label; treat as a limitation |
| R1-7 | R1-9 | Limitations and future work missing |
| R1-8 | — (no historical row) | High-skill single-region focus is reasonable; recorded, no change requested |
| R2-1 | R2-2, R2-3, R2-4 | Hand-crafted label weights; raw-kill-advantage and learned-weight ablations; weight sensitivity |
| R2-2 | R2-5a, R2-5b, R2-5c, R2-6, R2-7, R2-10 | FT-Transformer / TabNet / SAINT absent; competitive setting; equal feature engineering; short-sequence Transformer |
| R2-3 | R2-8, R2-9 | Temporal signal insufficient; alternative resolutions or extended windows |
| R2-4 | R2-1 | No anonymised link to code or data |
| R3-1 | R3-1, R3-6 | Technical choices unmotivated; hard for non-ML readers |
| R3-2 | R3-2, R3-3 | "Diagnostic MLP" unreferenced; attention or CNN probes |
| R3-3 | R3-4 | "Layered+Logit" notation unclear |
| R3-4 | R3-5 | TreeSHAP use unmotivated |
| R4-1 | R4-1, R4-2 | Teamfights without kills |
| R4-2 | R4-3 | Labelling method limited by the API |
| R4-3 | R2-5a–c, R2-6 (meta-review echo) | Deep-learning models limited, may not reach their potential |
| CE-1 – CE-6 | — (not reviewer issues) | Collaborator correspondence and author-proposed extensions, §6 |

The historical audit rows X-1 – X-4 (legacy leaking feature path, leak-exposed SAINT row, stacked-predictor
mismatch, position-error calibration) have **no** row here. They are handled as lineage and limitation text in
[manuscript.md](manuscript.md) §1.4, §6.1 and §8.5 and in [claim_evidence_ledger.md](claim_evidence_ledger.md)
C-69–C-72, not as answers to a 118 reviewer.

---

## 2. Reviewer 1 (score +1, weak accept)

| ID | Original concern | Implemented change | Evidence | Draft section | Remaining condition | Status |
|---|---|---|---|---|---|---|
| R1-1 | Abstract is informative but reads disjointed; add connective sentences | The draft abstract is written as one narrative: problem → why a model-defined outcome → definition → increment result → matched-learner result → sensitivity → explanation → what it does not establish | No experiment | Abstract | Final numbers and wording fixed only after the remaining open items of §11.2; length not yet fitted to any journal template | partial |
| R1-2 | Contribution statement merely restates the research questions; discuss impact on the community, citing audience-experience work [1]–[6] | Contributions were rewritten as seven reusable outputs and audited non-detections (definition with an exact mechanism account, reinforced baseline, matched-input comparison, sensitivity bounds, no detected composition benefit in the tested setup, bounded explanation layer, explicit open list) | §1.3 of the draft; the underlying artefacts are the frozen run directories listed in the ledger | §1.3 | Of the six references the reviewer supplied, **[1] Schubert, Drachen and Mahlmann (2016) was re-verified** in the 2026-09-16 citation audit and is cited as `Schubert2016`; the remaining **five** audience/narrative references ([2] Block et al. 2018, [3] Kokkinakis et al. 2020, [4] Pedrassoli Chitayat et al. 2024, [5] Charleer et al. 2018, [6] Pedrassoli Chitayat et al. 2024 CHI PLAY) were not re-verified and are not in `references.bib`. Schubert2016 supports encounter segmentation and the kill-free encounter share, **not** spectator utility; no spectator- or broadcaster-facing utility was evaluated, so no audience-benefit claim is made | partial |
| R1-3 | Why kills clustered within 18 s? Why 4,000 game units? The thresholds look arbitrary | Both constants are now estimated from kill-event data with bootstrap intervals, stability analysis and per-patch drift: `G = 13.7246 s` (CI 12.1601–15.6675, plateau ≈10–18 s) and `D = 4263.8688 u` (CI 4256.0475–4273.9677). A full re-detection with training-patch-only constants (14.0 s / 4,285 u) quantifies the residual dependence | `config/fight_boundary/spec_pooled.json`; `outputs/definition_dev_20260916/` (census.json, eval/results.json); ledger C-07, C-08, C-52–C-61 | §4.1, §8.4 | Pooled estimation still included the test patch (§4.3); `D = 4,285` falls outside the pooled interval; a 15.14+15.15 estimate and other detector switches (merging, duration, presence) were not varied; untouched confirmation data do not exist | partial |
| R1-4 | Add a background section defining League of Legends terms | Game terms are defined at first use inside the definition, value-model and input sections, and the participant-ordering convention is stated explicitly | §4, §5, §6 of the draft | §4–§6 | No dedicated Background section or term table yet; a non-specialist glossary is still needed for a journal draft | partial |
| R1-5 | Commend the labelling description; add it for less-informed readers | The label is now defined by an explicit formula with the endpoint rule, both exclusions, the exact-zero convention, endpoint statistics, and a worked hypothetical example in the collaboration document | §5.2–§5.4; worked example in [collaborator_specification.md](collaborator_specification.md) §8.2; `docs/TOG_END_TO_END_READINESS_AUDIT_20260915.md` §8 | §5 | A figure of the interval and a real anonymised timeline example are not produced | partial |
| R1-6 | Teamfight outcome is hard to label even for an expert; address this as a limitation | The draft states outright that `Y` is defined by `V`, that no independent outcome criterion has been applied, and that a legible label is not necessarily a valid one. It also reports how far the label moves under alternative valuation models | §10.1; ledger C-20; `docs/LABEL_VALIDITY_FINDINGS_20260915.md` | §10.1, §9 | No independent semantic criterion. A 120-case anonymised review packet exists but collected zero judgements; human review is outside the agreed scope and is **not** treated as a blocking prerequisite | partial |
| R1-7 | The discussion barely covers limitations and future work | A seven-part Limitations section (validity, attribution, retrospective population, unexecuted comparisons, generalisation, statistical reading, population scope) and an explicit open-work table | §10, §11.2 | §10, §11 | Nothing further planned for this row itself | completed |
| R1-8 | The high-skill, single-region focus is reasonable (no change requested) | Recorded rather than changed: the corpus scope and the absence of player-level and tier-level claims are stated | §3.1, §3.4, §10.7 | §3, §10.7 | — | completed |

---

## 3. Reviewer 2 (score −1, weak reject)

| ID | Original concern | Implemented change | Evidence | Draft section | Remaining condition | Status |
|---|---|---|---|---|---|---|
| R2-1 | Hand-set event coefficients and softmax weighting; the model may learn the labelling heuristic; add alternative-label ablations and weight sensitivity | The hand-weighted exchange label was **replaced**, not tuned. The outcome is now the sign of the change in a learned match-win estimate, with out-of-fold value models for training labels. An exact decomposition shows what the learned label actually weights, and label sensitivity is reported for two alternative value models and for removing the explicit objective columns | `outputs/v_mechanism_20260916/` (identity verified on 366,746 rows); ledger C-20, C-73–C-79 | §5, §8.6, §10.1 | The reviewer's specific request — raw kill advantage and learned-weight label variants scored on identical rows — has **not** been re-run on the new target. Existing label-family comparisons belong to the old `market_event` target and are not transferable | partial |
| R2-2 | Deep-learning baselines are underpowered; FT-Transformer, TabNet and SAINT are absent; deep models do not get the same feature engineering | A matched-information track was built: regularised logistic, LightGBM, plain MLP and residual MLP read the **identical** 352 columns under one split, weighting, calibration and selection protocol. Each family had **six candidate configurations** (six `C` values for the logistic; three widths × two dropout rates for each MLP; three leaf counts × two minimum-child sizes for LightGBM), and the **stochastic** families — LightGBM and the two MLPs — additionally averaged three seeds (7/42/123); the regularised logistic is a single deterministic fit per candidate, not a three-seed ensemble. Result: on teamfights the logistic/LightGBM/plain-MLP contrasts have intervals containing zero (equivalence not tested), the residual MLP is measurably worse, and LightGBM is measurably better than both networks on other engagements; those particular comparisons are retained at 60/90/120 s caps, while the exact point-estimate ranking moves (plain MLP leads on `T` at h60) | `outputs/track_a_mlp_20260916/` (protocol.json seed/prediction entries), `outputs/horizon_sensitivity_20260916/`; ledger C-35–C-51 | §6.2, §8.2, §8.3 | FT-Transformer, TabNet, SAINT and the sequence/graph/cross-attention/layered-fusion adapters are **not executed on this target**. Accordingly the draft makes no general deep-learning inferiority claim and no claim that family choice is immaterial; the comparison is bounded by six candidate configurations per family and one optimiser setting | partial |
| R2-3 | Temporal signal is insufficient: 6 timesteps over 30 s with ~60 s snapshot resolution; try other resolutions or longer windows | The 6×30 s window was abandoned. The pre-engagement state is the latest frame at or before the cutoff plus the full causal event history to that instant, and the observation-age problem is now measured rather than assumed: mean snapshot age 33.85 s at teamfight pre-state; 35.145% of all valid main-TEST h90 rows (35.27% of teamfight rows) have no new frame after the last kill, and the strictly narrower condition of pre and post states reading the **same** frame holds for 2,224 of 32,981 teamfight rows (6.74%). The endpoint cap was varied (60/90/120 s), and the label-mechanism analysis shows that in that narrower same-frame stratum the label is carried by event columns **together with** elapsed-time terms, not by events alone | `docs/COLLABORATOR_CRITIQUE_RESPONSE_20260915.md` §3; ledger C-17, C-46–C-51, C-77 | §5.4, §6.1, §8.3, §8.6, §10.3 | Horizon variation changes the **label**, not the input window; a matched information-content comparison over input history length has not been run on this target, and frame resolution remains a property of the public API | partial |
| R2-4 | No anonymised GitHub link or alternative data source | Not addressed. Code, manifests, models, predictions and hashes are retained locally, and every result in the draft names its run directory and JSON key | Run directories and `frozen_manifest.json` per run; ledger §2 | §11.2 | Anonymised repository, licence, reproduction instructions and a data-sharing policy compatible with the API terms. None prepared | open |

---

## 4. Reviewer 3 (score +1, weak accept)

| ID | Original concern | Implemented change | Evidence | Draft section | Remaining condition | Status |
|---|---|---|---|---|---|---|
| R3-1 | Many technical choices are presented without motivation; the paper is hard for non-ML readers | Each methodological section now opens with the question it answers, and each constant carries an evidence class (published precedent / empirical estimate / operational choice / rule-or-implementation fact) so a reader can see what is asserted versus estimated versus chosen | §1 of [claim_evidence_ledger.md](claim_evidence_ledger.md); §4.1 constant table | §4–§7 | Non-specialist framing is still uneven; the background/glossary gap of R1-4 applies here too | partial |
| R3-2 | "Diagnostic MLP" appears without a reference; alternatives such as attention or CNN probes exist | The MLP is no longer a diagnostic probe. Plain and residual MLPs are full comparison learners on the identical input contract, with declared architecture grids, seeds, stopping rule and selection protocol | `outputs/track_a_mlp_20260916/protocol.json`; ledger C-35–C-44 | §6.2, §8.2 | Attention-based and convolutional learners were not run on this target; the draft therefore claims nothing about them | partial |
| R3-3 | The "Layered+Logit" notation is unclear | Answered by removal: no fusion or stacked-logit model exists in the current target's model set. The name appears only in lineage discussion, where it is identified as an earlier 118-era diagnostic | §1.4; [collaborator_specification.md](collaborator_specification.md) §16.1 | §1.4 | If a stacking arm is ever added it needs its own out-of-fold base predictions; value-label cross-fitting alone would not make stacking safe | completed |
| R3-4 | Post-hoc TreeSHAP is introduced without motivation for a non-expert reader | The explanation layer states its purpose, its exact protocol (seven groups, 128 coalitions, 128 background rows, 256 explained rows per cell), its additivity check against the frozen predictions, and five explicit boundaries including that masked `p_pre` is not recomputed | `outputs/balanced_shap_20260916/shap/shap_summary.json`; ledger C-80–C-86 | §2.3, §8.7 | Group definitions depend on a naming-order classification (four inhibitor columns sit in the combat group); a corrected grouping would require regenerating, not relabelling, the explanations | partial |

---

## 5. Meta-review (R4, recommended accept)

| ID | Original concern | Implemented change | Evidence | Draft section | Remaining condition | Status |
|---|---|---|---|---|---|---|
| R4-1 | What about teamfights with no kills? A fight can be won by draining cooldowns or taking map control | Acknowledged explicitly as a scope limit rather than covered. The population is kill-conditioned by construction, and the draft cites external evidence for how large the kill-free class can be: 18,744 of 23,110 encounters (81.1%) in Schubert et al.'s Dota data contain no kill | [Schubert2016] via `outputs/manuscript_integration_20260916/citation_audit.md`; §4.2 | §4.2, §10.7 | Our own kill-free share is **not** re-measured for the current pipeline in this integration. A prior project measurement under comparable gates records 591 of 19,155 proximity encounters (3.09%) with no kill on 20,000 matches, but its denominator is proximity encounters, not engagements, and it was not re-verified today (`C:/Users/todtj/PycharmProjects/LOL_teamfight/docs/tog_manuscript/stale_claims_inventory.md` §8). A kill-free detector and label remain future work | partial |
| R4-2 | The labelling method is limited, though the Riot API may not allow much better | The redesign accepts the constraint and makes the measurement explicit instead of claiming ground truth; §10.1 states the validity gap plainly | ledger C-20, C-73–C-79 | §5.3, §10.1 | Same as R1-6: no independent outcome criterion | partial |
| R4-3 | The deep-learning models are limited and may not reach their potential | Addressed within a bounded matched-input comparison (R2-2), and the draft removes any general deep-learning ranking claim | ledger C-35–C-45 | §8.2, §10.4 | Same as R2-2: modern tabular transformers and the representation track are unexecuted | partial |

---

## 6. Collaborator correspondence items (separate from the 118 review)

These items come from correspondence with a prospective collaborator, not from a reviewer, and are tracked
separately because they concern the CoG 308 label design and a proposed evaluation subset, not the 118 decision.

**Provenance warning.** Two directions of correspondence are mixed in this section and are labelled per row.
The two genuine **incoming** concerns are (i) emphasis on balanced/even game states and (ii) how the exchange-value
coefficients were chosen; the incoming message also recommended reading Kim et al. CE-5 and CE-6, by contrast,
originate in **our own outgoing reply draft**, `docs/COG_COLLABORATION_EMAIL_DRAFT_20260915.md`, which proposes
comparing Kim et al.'s method with simpler calibration and reports our preliminary objective-removal figure. That
file is not evidence of a collaborator request or of a collaborator-supplied number, and no such request is
recorded in the incoming message.

### 6.1 Balanced game states

| ID | Item | Implemented change | Evidence | Draft section | Remaining condition | Status |
|---|---|---|---|---|---|---|
| CE-1 | *(incoming)* Emphasise more even game states | A balanced subset was predeclared in this follow-up's protocol as `p_pre ∈ [0.40, 0.60]` (**B40**, primary) with `[0.45, 0.55]` (**B45**) as a nested sensitivity. Both are defined from the pre-engagement value estimate only — never from realised `ΔV` or the final outcome — and are operational choices, not values prescribed by any cited paper | `outputs/incremental_q_training_20260915/REPORT.md` §6.1–§6.2 | §8.1 | Thresholds are ours; no external basis is claimed | completed (definition) |
| CE-2 | *(our operationalization of the incoming balanced-state concern)* Does a model beat the prior estimate in balanced states? | Executed. In teamfight B40 (4,949 rows / 4,570 matches) the predeclared LightGBM − PT contrast is −0.00043 [−0.00144, +0.00060] (inconclusive); the full logistic − PT contrast in the same cell is −0.00157 [−0.00306, −0.00013] and excludes zero; in `N` B40 (37,675 rows) LightGBM − PT is −0.00630 [−0.00703, −0.00555] | ledger C-28–C-31, C-34 | §8.1 | The teamfight answer is unresolved for LightGBM, not negative, and the logistic contrast in the same cell did resolve a difference, so remaining state signal in balanced fights is not shown to be unresolvable. The result must not be summarised as "full-input models do not help in balanced states"; the logistic contrast is an exploratory secondary result with no multiple-comparison adjustment | completed (executed), interpretation constrained |
| CE-3 | *(our operationalization of the incoming balanced-state concern)* Explain balanced states, not only score them | Executed. Exact seven-group Shapley on the frozen selected `q`: `p_pre` falls from 61.7% to 27.7% of total absolute attribution between full T and B40, while the absolute contributions of economy, objectives and combat are essentially unchanged. These are row-bootstrap descriptive intervals over 256 selected rows, not match-bootstrap metric intervals | ledger C-80–C-86 | §8.7 | The share shift mostly reflects the falling `p_pre` contribution, partly by construction of the cell; a second learner whose Brier difference from the tree was **not resolved** by this protocol (plain MLP − LightGBM +0.00007 [−0.00046, +0.00057]) attributes very differently | completed (executed), interpretation constrained |
| CE-4 | *(our record)* A balanced-state analysis was "planned, not executed" in the September 15 specification | The corresponding section of the collaboration specification is updated from *planned* to *completed* for the matched-input comparison, the balanced-cell evaluation and the balanced-state explanation | [collaborator_specification.md](collaborator_specification.md) §16 | — | The representation track of the same specification remains planned | completed for CE-1–CE-3 |

### 6.2 Exchange-value coefficients (CoG 308 design, retained for reference only)

*(Incoming concern.)* The collaborator asked how the exchange coefficients were chosen. They were **fixed,
domain-informed design choices**, not learned exchange rates. The table below documents the **published CoG 308**
label and is reproduced here because the question was asked; **none of these coefficients is used anywhere in the
Delta V target, and no result in [manuscript.md](manuscript.md) depends on them.** The published score, the later
intermediate `market_event` experiments in the earlier codebase, and the current `ΔV` target are three separate
lineages; we do not identify the published score with the `market_event` implementation here, because that
implementation was not re-inspected in this integration.

Label: `Y_CoG = 1[ Σ_u α_u σ_u v_u > 0 ]`, with `α_u = exp(2 p_u) / Σ_v exp(2 p_v)`, where `σ_u` is the sign of
the benefiting team, `v_u` the event value and `p_u` the importance prior.

| Descriptor in CoG Table I | Event value coefficient | Importance prior |
|---|---:|---:|
| Kill indicator | 1.00 | 0.25 |
| Normalised shutdown | 1.60 | 0.30 |
| Normalised streak | 0.35 | 0.15 |
| Normalised assists | 0.20 | 0.10 |
| Normalised bounty | 0.30 | 0.20 |
| Objective tier | 1.10 | 0.35 |
| Lane priority | 0.25 | 0.15 |
| Special event bonus | `s(u)` | `s(u)` |

Accompanying rules in that original implementation: descriptors clipped to [0, 1]; shutdown and bounty
log-scaled; streak divided by 10 and assists by 4; objective tiers including plate 0.35, dragon 0.75,
inhibitor/Atakhan 0.85 and Baron/Elder/soul/Nexus 1.0; special bonuses including first blood 0.20 and multi-kill
0.25, with ace handled separately; near-zero ties below 1e-8 resolved by a deterministic coin flip. The current
target instead codes exact zero as non-improvement (`Y = 0`), and no exact zeros occur in the retained labels.

**How the response differs from tuning.** The concern was answered by changing the valuation formulation — a
learned state-based win estimate whose coefficients come from fitting match outcome — rather than by adjusting
the coefficients above for a better score. What the learned label actually weights is now reported exactly
(§8.6): for teamfights, combat and survival 35.5%, economy and experience 28.2%, objectives 18.9%, structures
9.6%, health/mana/other 7.2%, with static champion terms cancelling to exactly zero.

### 6.3 Calibration paper recommended by the collaborator, and our own reply content

| ID | Item and its origin | Position taken | Draft section | Status |
|---|---|---|---|---|
| CE-5 | **Author-proposed extension**, not an incoming request: the collaborator recommended reading Kim et al.; the proposal to *compare* their uncertainty-aware calibration with simpler methods, especially in balanced states, appears in **our outgoing reply draft** | Cited for motivation only. Their loss and architecture are **not implemented**, and no such comparison has been run. Our calibration work is limited to comparing raw, sigmoid and isotonic transforms in dedicated validation roles, with `V` calibration evaluated against `W` and `q` calibration against `Y` — two references that are not interchangeable | §2.1, §5.1, §7.1 | open (proposed extension, not requested or executed) |
| CE-6 | **Our own preliminary result**, included in our outgoing reply draft ("approximately 5.4% of teamfight labels"); it was not supplied or quoted by the collaborator | Retained and made precise: 5.358% of main-TEST teamfight rows change label, 5.239% under equal match weighting, interval 4.982–5.480% | §10.1; ledger C-20 | completed |

---

## 7. Status tally

25 rows under the new consolidated IDs of §1.1: 8 reviewer-1 rows, 4 reviewer-2 rows, 4 reviewer-3 rows,
3 meta-review rows and 6 correspondence items (four from the incoming message, CE-4 from our own status record,
CE-5 and CE-6 from our outgoing reply draft).

| Status | Count | Rows |
|---|---:|---|
| **completed** | 4 | R1-7, R1-8, R3-3, CE-6 |
| **completed as executed, with constrained interpretation** | 4 | CE-1, CE-2, CE-3, CE-4 |
| **partial** | 15 | R1-1, R1-2, R1-3, R1-4, R1-5, R1-6, R2-1, R2-2, R2-3, R3-1, R3-2, R3-4, R4-1, R4-2, R4-3 |
| **open** | 2 | R2-4, CE-5 |

The two structurally most important gaps are the same ones the 118 reviewers identified: an independent criterion
for what the label measures (R1-6, R2-1, R4-2) and a genuinely equitable comparison that includes modern tabular
and representation-based deep models (R2-2, R3-2, R4-3). Neither is closed by this draft. A third gap — no
anonymised artefact release (R2-4) — is an execution task rather than a scientific one, and remains untouched.
