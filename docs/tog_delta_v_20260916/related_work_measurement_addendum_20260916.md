# Addendum: related work in three lineages, learner-comparison standards, attribution, and a measurement-validity table

Drafted 2026-09-16 (evening) by Claude as a **proposed replacement for manuscript §2 and a proposed table for §10.1** of
`manuscript.md`. Written as a separate file so that the accepted integration deliverables and their hash records are
untouched; merge is a copy of the sections below plus `references_addendum.bib` into `references.bib`. Sources were read
in the 2026-09-16 literature pass; the verification level of every source is stated in §5 and in the `annote` of each
BibTeX entry. Abstract-only sources are cited for their problem framing only — **no number, architecture detail or
threshold from an abstract-only source appears in the proposed text.**

> 한국어 요약. 관련연구를 "승패 예측 논문 모음"이 아니라 세 계보(상태 조건부 가치평가 → 교전 식별·교전 기반 예측 → 교전 전
> 방향 예측)로 나누고, 표 형식 학습기 비교의 문헌 표준(시드·탐색 예산·앙상블 회계)과 귀속의 모형 의존성을 관련연구에 넣는다.
> §10.1은 Jacobs & Wallach의 측정 어휘로 "무엇을 재는가"를 표로 정리한다. 원고 본문은 수정하지 않았고, 이 파일과 bib 부록만 추가했다.

---

## 1. Proposed §2 — Background and related work

Claims in this section are kept to what each cited work supports; per-source boundaries are recorded in §5 of this
addendum (verification level) and in the `annote` field of every reference.

### 2.1 Valuing events by the change of an estimated win probability

Three strands of prior work value **observed** actions by differencing a probability model across the action. Maymin
[Maymin2021] fits an in-game logistic win-probability model for League of Legends on elapsed time and both teams'
cumulative kills, towers and large monsters, samples one random minute per game to reduce within-game dependence, and
classifies kills and deaths by whether the estimated win probability rose around them. Xenopoulos, Doraiswamy and Silva
[Xenopoulos2020] define, for Counter-Strike, the value of a damage event as the difference of the predicted round-win
probability between consecutive states and accumulate it into "win probability added"; they evaluate the underlying
probability model on a temporal holdout with log loss, Brier score, AUC and calibration plots, and state that the value
estimates inherit their reliability from that model. Decroos et al. [Decroos2019] (VAEP) value on-ball soccer actions by
the change in the probability of scoring minus the change in the probability of conceding within a fixed number of
subsequent actions, truncate the game state to the last three actions, select the probability model with the Brier
score, and require calibrated probabilities because action values are untransformed probability differences.

Our label follows the same device — a frozen state-conditioned probability estimate differenced across a window — and
inherits the same dependence on the probability model, which is why V is evaluated with proper scoring rules and
calibration diagnostics before any label is derived (§5.1, §7). The lineage ends there: all three works assign value
**after** the action, whereas q predicts, from information available before a detected engagement window starts,
whether the window's change will be positive. None of the three validates our feature schema, engagement definition,
endpoint rule or patch population, and none supports a causal reading of a probability change (§5.3).

### 2.2 Segmenting combat, and predicting from segments

Halfaker et al. [Halfaker2015] read a temporal boundary out of an inter-activity-time distribution; we follow the idea,
not the estimator or the value (our G is a KDE valley on kill gaps). Schubert, Drachen and Mahlmann [Schubert2016]
define Dota encounters from spatio-temporal links with team/opponent relations and report that most encounters contain
no kill (18,744 of 23,110), which is the direct warning that our kill-conditioned population is a strict subset of
combat. Two further works use combat segments as **inputs to match-outcome prediction**: Ke et al. [Ke2022] define
Dota 2 team fights procedurally from kills and proximity and feed a sequence of fights to a recurrent predictor of the
match result, and Yang, Harrison and Roberts [Yang2014] represent professional Dota 2 combat as graphs whose patterns
predict the match winner. Tot et al. [Tot2021] predict whether a team fight is **occurring** from player positions and
camera signals. These four settle what our unit of analysis is not: we neither predict the match result from fights nor
detect fights live; we predict, before a retrospectively identified fight window, the sign of the value change over that
window. (Ke, Yang and Tot were available to us only as abstracts or institutional summaries; §5.)

### 2.3 Comparing learners on identical tabular inputs

The tabular deep-learning literature converges on a comparison protocol: matched hyper-parameter search budgets per
model family, many seeds, and explicit accounting for ensembling [Gorishniy2021; Gorishniy2025; Holzmuller2024].
Gorishniy et al. [Gorishniy2021] find no universally superior family between gradient-boosted trees and neural
networks; Rubachev et al. [Rubachev2025] show that under **time-based splits with many correlated engineered features**
the tree advantage shrinks and simple MLPs with numerical-feature embeddings are competitive while attention- and
retrieval-based models are not; Gorishniy et al. [Gorishniy2022] attribute much of the MLP gain to numerical
embeddings, and TabM [Gorishniy2025] adds a parameter-efficient ensemble inside one MLP. None of these papers reports
Brier score or log loss, and RealMLP [Holzmuller2024] notes that its label-smoothed objective is not a proper scoring
rule. Our comparison is narrower than that standard — five families at fixed configurations with three seeds, selected
by a proper score on a patch-disjoint validation role — and we present it as such (§6.2, §10.4); TabReD's result is the
prior expectation for our finding that trees, a linear model and a plain MLP end within overlapping intervals on a
patch split of engineered inputs, and it is the reason the unexecuted comparisons of §10.4 are named (numerical
embeddings, TabM, FT-Transformer) rather than "deep learning" in general.

### 2.4 Attribution and its model dependence

Lundberg and Lee [Lundberg2017] define SHAP as additive feature attribution relative to an explanation mapping;
attributions depend on the background data and on how dependent features are handled. Aas, Jullum and Løland
[Aas2021] show that independence-based (marginal) masking of dependent features produces unrealistic feature
combinations and can misattribute, and propose conditional estimators. Chen, Lundberg and Lee [Chen2022] explain a
**series** of models — an upstream model whose output feeds a downstream model — by propagating attributions through
both stages with the same baseline, so that the upstream output is recomputed under masking rather than treated as a
fixed input. Fisher, Rudin and Dominici [Fisher2019] show that models with near-identical loss can rely on different
variables and propose reporting the **range** of a variable's reliance over all near-optimal models (model class
reliance). Our q takes as input a state and the value model's own output on that state; §8.7 explains q in its input
space with marginal masking (p_pre held fixed), and the composite-system explanation and the model-class range of
§8.8 (`composite_shap_20260916`) are the responses to [Chen2022] and [Fisher2019]. Neither explanation is causal.

### 2.5 Probability quality

Proper scoring rules [Gneiting2007] justify Brier score and log loss as the selection and reporting criteria for both V
and q. Dimitriadis, Gneiting and Jordan [Dimitriadis2021] replace binned reliability diagrams with an isotonic
(pool-adjacent-violators) diagram and decompose a proper score into miscalibration, discrimination and uncertainty; Kim,
Lee and Chung [Kim2020] motivate treating calibration as a first-class concern for League of Legends winner
prediction (their loss is not implemented here). A property that follows from our label definition, not from any
citation: applying one strictly increasing transform to both ends of the window leaves the sign of ΔV unchanged, so a
global monotone recalibration of V cannot change Y, although it changes |ΔV|, p_pre and the membership of the balanced
cells (§8.6).

### 2.6 What no cited work supplies

No source establishes that G = 13.7 s or D = 4,264 u are correct boundaries; that a 90 s cap is optimal; that the sign
of ΔV is fight victory; that champion composition should or should not be learnable on this target; that a set, graph
or sequence representation would help on it; or that our reported differences generalise across patches and regions.
Those remain our operational choices and estimates, supported in this draft by rationale and sensitivity analysis.

---

## 2. Proposed §10.1 — Measurement validity (Jacobs & Wallach vocabulary)

The construct is "the engagement went well for Blue"; the operationalization is Y = 1[V(S_e) − V(S_pre) > 0] over the
retrospective kill-conditioned window with the frozen V; the measurement is the stored label array. The table separates
what the runs establish from what remains open. Evidence anchors are the run roots; numbers must be read from the cited
`results.json`/findings files, not from this table alone.

| Validity notion [Jacobs2021] | Question for Y | Evidence in this draft | Status |
|---|---|---|---|
| Reliability | Same procedure, same result? | Bitwise reproduction of labels, states and predictions in every follow-up run (fixture and identity checks; `definition_dev`, `composite_shap`, `v_mechanism`) | established |
| Content validity | Does the operationalization cover the construct? | Kill-conditioned windows only; kill-free combat, zoning and vision are excluded by construction; objectives do not stop the window | partial, stated |
| Convergent validity | Do alternative operationalizations agree? | Sign agreement with alternative value models and horizons (label-validity audit 2026-09-15; h60/h120 refits); definition constants from the training patch alone (`definition_dev`) | partial: alternatives agree on most rows, disagree on small-|ΔV| rows |
| Discriminant validity | Does Y measure something other than prior advantage and time? | q beats the p_pre × time spline (PT) on TEST; in balanced states the increment is small and learner-dependent (§8.1) | partial |
| Predictive validity | Does the label relate to an outcome it did not use? | V is trained on the final result W; ΔV of engagements is a martingale-like increment (mean near zero) whose sign q predicts above PT | partial |
| Face validity | Do domain readers accept the label as "fight went well"? | No independent human judgement collected (user decision; outside scope) | open gap |
| Consequential validity | What follows from using the measurement? | No deployment; exploratory manuscript | not applicable |

The open gap is face validity; the draft therefore describes Y as a model-defined direction, never as fight victory.

---

## 3. Sentences the draft can now state (each anchored above)

1. "Prior work values observed actions by the change of an estimated win probability; we predict, before a defined
   engagement window, whether that change will be positive." [Maymin2021; Xenopoulos2020; Decroos2019]
2. "Fight segments have been used as inputs to match-outcome prediction; our unit of analysis is the segment itself."
   [Ke2022; Yang2014; Schubert2016]
3. "On a time-based split of engineered inputs, trees, linear models and plain MLPs are expected to end close;
   attention-based tabular models are not expected to win." [Rubachev2025; Gorishniy2021]
4. "Attributions of q in its input space and of the composite system are different objects; both depend on the
   background and on the learner." [Chen2022; Aas2021; Fisher2019; Lundberg2017]
5. "A global monotone recalibration of V cannot change Y." (definition; §8.6)

---

## 4. What this addendum does not do

It does not edit `manuscript.md`, `references.bib` or the ledger; it does not add numbers from abstract-only sources; it
does not claim the composite-system or Rashomon results before `composite_shap_20260916` passes its post-run checks.

## 5. Verification level of the new sources (2026-09-16 pass)

| Key | Read | Use allowed |
|---|---|---|
| Xenopoulos2020, Decroos2019, Maymin2021 | full text | method, formula, evaluation protocol, positioning |
| Rubachev2025, Gorishniy2021, Gorishniy2025, Holzmuller2024, Gorishniy2022 | full text | protocol standards, qualitative findings; code licences not verified |
| Chen2022, Aas2021, Fisher2019, Jacobs2021, Dimitriadis2021 | full text | method and definitions; G-DeepSHAP code link and MCR code not verified |
| Che2018, Lee2019, Zaheer2017 | full text | mechanism only (not cited in §2 above; reserved for a representation track) |
| Gu2021 (NeuralAC) | partial | framing only; headline numbers to be re-verified before use |
| Ke2022, Yang2014, Tot2021, Ringer2023 | abstract / institutional page only | problem framing only; no numbers, architectures or thresholds |
| Gneiting2007, Kim2020 | not re-read in this pass (Kim2020 verified by the 2026-09-16 citation audit) | standard citation; Kim as motivation only |
