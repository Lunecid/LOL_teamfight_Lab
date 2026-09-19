# CoG succession lock — what we inherit and what we improve

**Status:** locked 2026-09-19 (collaborator framing; before Intro RQ finalization)  
**Order rule:** fix **CoG purpose → improvement agenda** first; place completed experiments under that agenda; **do not** invent Intro RQs from every sealed ΔBrier / B40 / SHAP cell.

Companions: [CoG2026_Paper.md](CoG2026_Paper.md), [V_REDESIGN_CONTRACT_20260919.md](V_REDESIGN_CONTRACT_20260919.md) (**current priority: redesign \(\widehat{V}\) before \(q\)**), [V_DYNAMIC_FRAME_CONTRACT_20260919.md](V_DYNAMIC_FRAME_CONTRACT_20260919.md), [EXPERIMENT_DESIGN_MAP_20260919.md](EXPERIMENT_DESIGN_MAP_20260919.md), [COMMON_RESEARCH_SPINE_20260919.md](COMMON_RESEARCH_SPINE_20260919.md) (저널·석사 병렬), [STRATEGIC_VALUE_LABEL_REDESIGN_20260919.md](STRATEGIC_VALUE_LABEL_REDESIGN_20260919.md), [EXPERIMENT_INVENTORY_COHORT_20260919.md](EXPERIMENT_INVENTORY_COHORT_20260919.md), [PAPER_COHORT_CONTRACT_20260919.md](PAPER_COHORT_CONTRACT_20260919.md).

---

## 0. Core purpose (unchanged from CoG)

> **공개 텔레메트리로 관측되는 교전 전 상태를 이용하여, 해당 교전과 그 후속 구간에서 어느 팀이 전략적 이득을 얻을지 예측하고, 그 예측을 가능하게 하는 정보와 한계를 이해한다.**

Short form: *How well can pre-engagement public information forecast post-engagement strategic advantage — and what does that tell us about the information and its limits?*

**Center:** predict and understand.  
**Not center:** proving a particular algorithm is best; proving contested states are inherently unpredictable; winning B40 vs PT as the research motive.

B40, PT, SVI, SHAP, external transfer are **methods / checks** for asking that question more carefully — not the original research goals.

---

## 1. What CoG already did (do not mis-describe)

| Piece | CoG |
|---|---|
| **Unit** | Kill events clustered in time/space into *localized engagements* (retrospective) |
| **Outcome** | Sign of a **fixed exchange-value score** (kills, shutdowns, streaks, assists, bounties, objectives, special events) — operational exchange result, **not** unique fight-win ground truth |
| **Inputs** | Public info in a **30 s window before** retrospectively fixed onset |
| **Comparison** | Tabular / temporal / graph / event / fusion **representations × learners** (not “same-input algorithm race only”) |
| **Reading** | Public data shows macro strategic edge better than fine fight execution |

**Forbidden strawmen about CoG:**

- “CoG only counted kills; we add objectives.” (CoG already used a multi-component exchange score.)
- “CoG claimed the exchange label was the true fight win; we finally admit model-defined labels.” (CoG already treated the label as operational.)
- “We are the first to do fair same-input model comparison.” (CoG already compared representation–learner couples and included MLP on tabular inputs.)
- “We are the first to do patch holdout.” (CoG already used 15.14 train / 15.15 val / 15.16 test.)
- “Detector thresholds were arbitrary with no rationale.” (CoG gave domain reasons for 18 s / 4000 / 1800 etc.; what remains is **empirical support and sensitivity**, not “no rationale.”)

---

## 2. Four improvement directions (design agenda, not experiment taxonomy)

### I1 — Stronger grounding of the **engagement unit**

- Keep: reproducible detector rules with stated domain reasons.  
- Add: how well those rules match observed distributions; **sensitivity** of sample and results to time/space/duration choices; whether narrowing to large teamfights is a **scope choice** (what we study), not cherry-picking for AUC.  
- Core: *persuade what counts as one engagement case* — not “make the detector more complex.”

### I2 — Define engagement **outcome value** via match-win linkage *(main substantive upgrade)*

- Keep: multi-outcome thinking beyond raw kill diff; honesty that the label is operational.  
- Change: from researcher-fixed exchange coefficients to  
  \(\widehat{V}(S_t)=\widehat{P}(\text{Blue wins}\mid S_t)\),  
  \(\Delta\widehat{V}=\widehat{V}(S_{\mathrm{end}})-\widehat{V}(S_{\mathrm{before}})\),  
  primary occurrence label \(\mathrm{SVI}=\mathrm{sign}(\Delta\widehat{V})\).  
- **Dynamic WP (frame-aligned):** parameters frozen; \(S_t\) updates on the public timeline grid so \(\widehat{V}\) changes when the state changes (including at engagement end). Report \(\widehat{V}\to W\) **by the same time bands** as later \(q\) tables — see [V_DYNAMIC_FRAME_CONTRACT_20260919.md](V_DYNAMIC_FRAME_CONTRACT_20260919.md).  
- **Redesign priority (2026-09-19):** rebuild \(\widehat{V}\) as a trustworthy match-WP evaluator **before** further \(q\) search — shared time-conditional default; per-band models as comparator; proper scores for selection — [V_REDESIGN_CONTRACT_20260919.md](V_REDESIGN_CONTRACT_20260919.md).  
- Also fix **evaluation horizon** (how far past the fight to include conversion of advantage) as a definition of the estimand, not a search for best AUC.  
- Core: *clearer warrant and scope for what “advantage” means* — not “rename the label SVI.”  
- Still not: independent fight truth; causal effect of “taking the fight.”

### I3 — Separate **observable information**, **representation**, and **learner**

- Keep: representation × learner framing.  
- Add: what is truly observed vs interpolated/proxy; leakage / availability at prediction time; same-info model swaps vs info/representation expansions; feature-group interpretation without equating SHAP to causal fight drivers.  
- Core: *where predictive signal comes from* — not “run more architectures as separate paper goals.”

### I4 — Stricter reading of **what performance means** and **where it applies**

- Keep: CoG pattern that late / large gold-gap states are easier; close states harder — as a **starting observation**, not a proven unique cause.  
- Add: prior-state baselines (PT / \(b(p)\)); contested strata (e.g. B40); proper scoring; transfer beyond adjacent KR patches (farther patch / other region).  
- Placement of PT & B40:  
  > Purpose = predict post-engagement advantage from pre-engagement info.  
  > Check = how much apparent skill is just reading pre-fight lead.  
- Core: from “report a good score” to “what that score means and how far it travels.”

---

## 3. Locked succession sentence

> **본 연구는 공개 텔레메트리를 이용한 교전 이후 이득 예측이라는 CoG 연구의 목적을 유지한다. 이를 위해 고정 규칙에 기반한 교전 구성의 근거를 보강하고, 수동 교환 점수를 경기 승패와의 관계를 학습한 상태 가치 변화로 확장한다. 그 위에서 교전 전 정보의 표현과 학습기 효과를 구분해 예측을 평가하며, 초기 우세에 대한 의존성, 결과 정의의 민감성, 다른 경기 환경으로의 적용 범위를 검증한다.**

Shorter:

> **공개 데이터 기반 교전 이득 예측을, 더 근거 있는 교전·결과 정의와 더 명확한 예측 평가를 갖춘 연구로 발전시킨다.**

Two intents together: (content) better definition of engagement and advantage; (prediction) stricter evaluation of forecasting that advantage from prior public info.

**Not** a label-audit-only paper. **Not** a model horse-race paper. Center remains engagement-advantage prediction, with definition and evaluation strengthened together.

**Improvement ≠ higher AUC than CoG 0.675.** Target and engagement definition changed; success is first stronger warrants, clearer comparisons, and scoped validation; predictive numbers are results under that redesigned estimand.

---

## 4. Place inventory experiments under I1–I4 (roles, not Intro RQs)

From [EXPERIMENT_INVENTORY_COHORT_20260919.md](EXPERIMENT_INVENTORY_COHORT_20260919.md):

| Improvement | Experiments / artifacts (examples) | Role |
|---|---|---|
| **I1 Unit** | Detector / definition docs (`DEFINITION_EVIDENCE`, engagement v3); size scope (T vs N); horizon/sensitivity when run | Justify *which cases* we study |
| **I2 Outcome** | \(\widehat{V}\) training/calibration; SVI construction; concordance kill/obj/alive; quiet matched \(\|\Delta V\|\); continuous \(\Delta V\) appendix; material cross-target | Explain *what* we measure as advantage |
| **I3 Info / learner** | Primary table same-row slate; lean TabM / Tier B; reselections; B40 info-set ablation; (SHAP only if serving interpretation of used info) | Separate information vs learner contribution |
| **I4 Meaning / scope** | PT / \(b(p)\) contrasts; B40 / \(H\) / lift localization; flex PT; transfer 16.x \(q\to\mathrm{SVI}\) and \(\widehat{V}\to W\); calibration / Brier | Dependence on prior lead; generalization |

**Do not** auto-promote: “SHAP RQ”, “why is B40 hard RQ”, “must beat B40”, or one RQ per output folder.

---

## 5. Deferred: Intro research questions → now branched

Intro RQs are written **from this succession agenda**, not from every sealed ΔBrier cell.

**Authoritative branched wording:** [COMMON_RESEARCH_SPINE_20260919.md](COMMON_RESEARCH_SPINE_20260919.md)

| Branch | RQs | Doc |
|---|---|---|
| **Journal** | J-RQ1 (measure I1·I2) · J-RQ2 (predict I3) · J-RQ3 (meaning/scope I4) | [JOURNAL_RESEARCH_PLAN_20260919.md](JOURNAL_RESEARCH_PLAN_20260919.md) |
| **Master thesis** | M-RQ1 (unit) · M-RQ2 (value) · M-RQ3 (info/learner) · M-RQ4 (validation) | [MASTER_THESIS_RESEARCH_PLAN_20260919.md](MASTER_THESIS_RESEARCH_PLAN_20260919.md) |

Shared evidence; different depth. Master-only M01 (frame/event info blocks) does **not** auto-replace journal primary \(q\).

---

## 6. Non-goals this extension need not solve

High-resolution private data; online/live prediction; kill-less pressure situations as primary sample — listed as CoG limits does **not** require solving all of them here. Staying in **public data + kill-conditioned retrospective engagements** while improving definition, measurement, and evaluation inside that scope is a coherent extension.
