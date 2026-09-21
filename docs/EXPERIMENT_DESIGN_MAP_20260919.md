# Experiment design map — CoG succession (2026-09-19)

**Authority:** [COG_SUCCESSION_LOCK_20260919.md](COG_SUCCESSION_LOCK_20260919.md)  
**Facts inventory:** [EXPERIMENT_INVENTORY_COHORT_20260919.md](EXPERIMENT_INVENTORY_COHORT_20260919.md)  
**Sample rules:** [PAPER_COHORT_CONTRACT_20260919.md](PAPER_COHORT_CONTRACT_20260919.md)

Intro RQ wording is **deferred**. This map places **pipeline stages** and **I1–I4 checks**; it does not promote every output folder to a research question.

---

## 1. Purpose → pipeline → checks

```mermaid
flowchart TB
  purpose["CoG purpose retained<br/>pre-fight public info → post-fight strategic advantage<br/>predict + understand"]

  purpose --> I1 & I2
  I1["I1 Engagement unit<br/>what is one case?"]
  I2["I2 Outcome value<br/>frame-aligned V S_t + SVI + horizon"]

  I1 --> cases["Cases: kill-conditioned engagements<br/>scope choice: teamfight T headline"]
  I2 --> label["Label: SVI = sign ΔV̂<br/>not fight-win truth"]
  I2 --> Vdyn["V̂ performance by time band<br/>same clock grid as frames"]

  cases --> X
  cases --> Y
  label --> Y
  Vdyn --> I4
  X["X: pre-onset public features<br/>352 ridge / subsets"]
  Y["Y: SVI on sealed rows"]

  X --> I3
  Y --> I3
  I3["I3 Info vs representation vs learner<br/>same-row q slate + ablations"]

  I3 --> q["q: predict P SVI=1 | X_pre<br/>selection on Q_SELECT; freeze before TEST"]

  q --> I4
  I1 -.-> I4
  I2 -.-> I4
  I4["I4 Meaning + scope<br/>vs PT / strata / transfer<br/>also checks unit + label sensitivity"]

  I4 --> read["Read results under redesigned estimand<br/>not vs CoG AUC 0.675"]
```

*Execution note (not a design node):* current frozen primary \(q\) = LightGBM from Q_SELECT / sealed incremental_q winner.
---

## 2. Data and evaluation layers (contract)

```mermaid
flowchart LR
  corpus["Study corpus<br/>210k KR<br/>15.14+15.15+15.16"]

  corpus --> train["15.14 TRAIN<br/>fit V queries / q"]
  corpus --> val["15.15 Q_CAL / Q_SELECT<br/>calibrate / select"]
  corpus --> test["15.16 T sealed<br/>n=32,981 — all headlines"]
  corpus --> meas["Pooled T<br/>n=113,901 — concordance / quiet"]
  corpus -.-> ext["EXT 16.x KR/NA1<br/>score-only transfer"]

  test --> primary["Primary contrast<br/>q − PT Brier"]
  meas --> valid["Validation axis<br/>kill/obj/alive · quiet"]
  ext --> xfer["q→SVI · V→W"]
```

---

## 3. Estimand construction (I2 center)

```mermaid
flowchart TB
  S["State S_t from public telemetry"]
  V["V̂ S_t = P̂ Blue wins | S_t<br/>trained on match outcome W"]
  Spre["S_before at engagement onset"]
  Send["S_end after label horizon H"]
  dV["ΔV̂ = V̂ end − V̂ before"]
  SVI["SVI = 1 ΔV̂ > 0<br/>primary prediction target"]

  S --> V
  Spre --> V
  Send --> V
  V --> dV
  dV --> SVI

  mat["Material concordance<br/>kill / obj / alive"]
  quiet["Quiet matched windows<br/>|ΔV| ratio"]
  SVI -.-> mat
  SVI -.-> quiet
```

---

## 4. Prediction and meaning checks (I3 + I4)

```mermaid
flowchart TB
  subgraph inputs [Information]
    ppre["p_pre / time"]
    ridge["352 ridge"]
    sets["Info sets: S0 → econ → combat → full"]
  end

  subgraph learners [Learners same rows]
    pt["PT prior×time baseline"]
    bp["b p prior-only"]
    slate["logistic · LGBM · MLP · TabM · TabNet"]
  end

  inputs --> learners
  learners --> table["Primary table 15.16 T<br/>identical rows + match weights"]

  table --> overall["Overall q − PT"]
  table --> strata["Strata: p_pre · time · B40×time"]
  table --> H["H = E d|B40 − E d|B40c"]
  table --> flex["Flexible PT freeze"]
  table --> abl["B40 info-set ablation"]
  table --> xfer2["Transfer freeze-score"]
```

**Role of PT / B40:** not the project motive — checks of *how much forecast skill is reading pre-fight lead*.

---

## 5. Experiment placement under I1–I4

| Stage | Role | Artifacts (inventory) | Status |
|---|---|---|---|
| **I1 Unit** | Justify cases / scope T | Detector docs; T vs N; definition sensitivity (if/when run) | Partly prior docs; sensitivity not full paper lock |
| **I2 Outcome** | Warrant for SVI | **V redesign first** (shared time-conditional); then concordance / quiet / ΔV | **In progress** — [V_REDESIGN_CONTRACT](V_REDESIGN_CONTRACT_20260919.md); old band ledger kept |
| **I3 Info–learner** | Separate info vs model | Primary table; reselection; lean TabM; Tier B; B40 ablation | Done |
| **I4 Meaning–scope** | Prior-lead dependence + transfer | PT/\(b(p)\); lift localization; \(H\); flex PT; EXT transfer | Done |
| **Contract** | Same corpus ≠ same eval sample | Cohort contract; phase1 sealed audit | Done |

---

## 6. What this design is *not*

```mermaid
flowchart LR
  bad1["Model horse-race as purpose"] -.-> x["out"]
  bad2["B40 must-beat-PT as purpose"] -.-> x
  bad3["One RQ per output folder"] -.-> x
  bad4["Beat CoG AUC 0.675 as success"] -.-> x
  bad5["Label audit only / no prediction"] -.-> x
```

Center remains: **forecast post-engagement advantage from pre-engagement public info**, with stronger unit/outcome warrants and stricter evaluation.

---

## 7. Next after this map

1. Confirm which **I1** sensitivity / scope claims enter the paper (vs appendix / prior detector papers).  
2. Write Intro RQs **per manuscript branch** from the succession sentence — see [COMMON_RESEARCH_SPINE_20260919.md](COMMON_RESEARCH_SPINE_20260919.md):  
   - Journal: J-RQ1–3 ([JOURNAL_RESEARCH_PLAN_20260919.md](JOURNAL_RESEARCH_PLAN_20260919.md))  
   - Master: M-RQ1–4 ([MASTER_THESIS_RESEARCH_PLAN_20260919.md](MASTER_THESIS_RESEARCH_PLAN_20260919.md))  
3. Shared runs once; placement in [SHARED_EXPERIMENT_MATRIX_20260919.md](SHARED_EXPERIMENT_MATRIX_20260919.md).  
4. Paper spine: Methods follow I1→I2→I3→I4; Results report prediction under that estimand, then meaning/scope checks.
