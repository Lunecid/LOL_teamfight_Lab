# Redesign: strategic value improvement as primary label

**Status:** design lock (2026-09-19); evaluation-contract revision; **Intro RQ wording deferred** until succession agenda is fixed  
**Prior lock (read first):** [COG_SUCCESSION_LOCK_20260919.md](COG_SUCCESSION_LOCK_20260919.md) — CoG purpose inherited; four improvement directions (unit / outcome / info–learner / meaning–scope); experiments are placed under those directions, not each promoted to an Intro RQ.  
**Decision:** keep \(\mathrm{sign}(\widehat{\Delta V})\) as the **primary prediction target**; never call it “teamfight win”; put kill / objective / quiet analyses on a **validation axis** beside it.

Authority for sample rules: [PAPER_COHORT_CONTRACT_20260919.md](PAPER_COHORT_CONTRACT_20260919.md).  
Model slate / transfer: [SVI_MODEL_RESELECTION_TRANSFER_20260919.md](SVI_MODEL_RESELECTION_TRANSFER_20260919.md).  
Forecast-value literature use: [RELATED_FORECAST_VALUE_LIT_20260919.md](RELATED_FORECAST_VALUE_LIT_20260919.md).  
Experiment inventory (facts only): [EXPERIMENT_INVENTORY_COHORT_20260919.md](EXPERIMENT_INVENTORY_COHORT_20260919.md).  
\(\widehat{V}\) frame dynamics / time-band tables: [V_DYNAMIC_FRAME_CONTRACT_20260919.md](V_DYNAMIC_FRAME_CONTRACT_20260919.md).

---

## 0. Succession context (do not invert)

Paper structure follows:

1. **CoG core purpose** — pre-engagement public info → post-engagement strategic advantage (predict + understand).  
2. **Four improvements** — engagement unit warrant; match-linked outcome (\(\widehat{V}\)/SVI + horizon); info vs representation vs learner; meaning of scores + transfer scope.  
3. **Then** Intro RQs and Results sections that ask what changed under the new estimand — not “how do we narrate the small ΔBrier / B40 CI.”

PT, B40, SHAP, transfer, model slate = **checks / tools** under (2), not the original motive for the project.

Forbidden CoG strawmen and the locked succession sentence: see succession lock §1–§3.

## 1. Problem the redesign solves

| Old pressure | Redesign response |
|---|---|
| “한타 승리” has no unique definition | Do **not** claim fight win. Predict **whether estimated strategic value improves**. |
| Hand-weighted exchange scores | Keep learned \(\widehat{V}\) → \(\widehat{\Delta V}\). |
| \(q\) may only reconstruct prior value | Primary contrast = **\(q\) vs PT**; contested increment (B40) as **Axis B** |
| \(\Delta V\) only as good as \(V\) | Validation = material **concordance** + quiet **matched-state** drift — not “external truth”; plus **frame-aligned** \(\widehat{V}\to W\) by time band (C03) |
| “Dynamic WP” ambiguous | Frozen params; \(S_t\) updates on public frame grid; \(\Delta\widehat{V}\) at engagement end — not live re-fit |

---

## 2. Terminology (mandatory)

| Term | Meaning | Use in paper |
|---|---|---|
| **Engagement** | Kill-conditioned detector event | Sample definition |
| **Estimated strategic value** | \(\widehat{V}(S_t)\): frozen model’s **estimated** \(P(\text{Blue wins}\mid S_t)\) | Ideal \(V\) vs estimate \(\widehat{V}\) distinguished |
| **SVI** | \(Y=1[\widehat{V}(S_{\mathrm{end}})>\widehat{V}(S_{\mathrm{pre}})]\) | Primary label: **occurrence** of upward revision, not magnitude |
| **B40** | \(0.4\le \widehat{V}(S_{\mathrm{pre}})\le 0.6\) | Balanced **prior-win summary**, not “empty information” or “high leverage” |
| **Material outcome** | Kill / objective / alive differentials in the window | Concordance checks |
| **Quiet reference** | No-kill window with **matched** \(p_{\mathrm{pre}}\), time, duration | Drift reference, not causal fight effect |
| **Teamfight win** | Folk concept | Forbidden as claim/label |

**Interpretation limit (must state early):** high \(q\) does **not** imply large expected \(\widehat{\Delta V}\) or that taking the fight is advantageous (sign vs mean can disagree).

Allowed shorthand: “SVI”, “\(\mathrm{sign}(\widehat{\Delta V})\)”.  
Forbidden: “predict who wins the teamfight”, “fight victory label”, “B40 = empty prior”, “B40 = high leverage”.

---

## 3. Research questions — provisional mapping (Intro wording deferred)

**Authoritative order:** [COG_SUCCESSION_LOCK_20260919.md](COG_SUCCESSION_LOCK_20260919.md).  
Section below retains earlier measurement templates (ΔBrier, \(H\), ablation, …) as **I3/I4 procedures**, not as the paper’s starting RQs. Final Intro RQs will be written from the succession sentence, then constrained by the inventory — not the reverse.

~~Earlier draft title~~ *Incremental predictive performance and state-dependent forecast value* remains a useful **evaluation slogan under I4**, not the CoG-inherited purpose.

---

### Axis A — Predictability (primary)

**RQ-A:** 동일 15.16 T (\(n=32{,}981\))에서 \(\Delta\mathrm{Brier}=\mathrm{Brier}(q)-\mathrm{Brier}(\mathrm{PT})\) (음수 ⇒ \(q\) 우세). Match-clustered bootstrap CI.  
\(b(p)\) 대비는 보조. 단독 ΔBrier ≠ forecast encompassing.

**Supporting procedures (not separate Intro RQs):**

| ID | Role | What |
|---|---|---|
| A1 | primary estimate | sealed \(q\) (LightGBM) vs PT on identical rows |
| A2 | model robustness | TabM / MLP / logistic / TabNet ≈ same small lift; FT collapsed |
| A3 | stronger baseline | 유연 PT(Q_SELECT 선택·동결) 대비에도 전체 lift 유지? |
| A4 | transfer | freeze‑score on KR/NA1 16.x (selection never on EXT) |

**Evidence lock (cites):**

| Claim | Value | Source |
|---|---|---|
| Primary \(q\) | LightGBM | `svi_primary_table_20260919`, `svi_reselection_20260919` |
| LGBM − PT | **−0.00113** [−0.00156, −0.00069]; \(P(\Delta<0)=1\) | primary table |
| LGBM − \(b(p)\) | −0.00192 [−0.00245, −0.00141] | primary table |
| TabM − PT | −0.00105 [−0.00163, −0.00047]; does not beat LGBM | primary + lean TabM |
| TabNet − PT | −0.00108 (≈ LGBM) | overnight Tier B |
| Flex PT | `flex_df7_ix`; overall \(q-\mathrm{flex}\approx -0.00136\) | `svi_state_dependent_20260919` |
| Transfer (headline) | KR/NA1 16.13 LGBM−PT ≈ −0.0015 / −0.0020 | `svi_transfer_2026_20260919` |

**Locked reading (Axis A):** 교전 전 특징은 PT 대비 **작은 그러나 재현 가능한** 추가 예측력을 갖는다. 효과 크기는 \(\tau=0.001\) 근처의 작은 증분이다.

---

### Axis B — Contested difficulty (primary addition)

**RQ-B:** 축 A의 추가 이득이 B40에서 유지되는가? 바깥(\(B40^{c}\))과 다른가?

손실차 \(d_i=(Y_i-q_i)^2-(Y_i-\mathrm{PT}_i)^2\) 에 대해

\[
H = E[d_i\mid B40] - E[d_i\mid B40^{c}]
\]

를 **같은 경기 부트스트랩**에서 직접 추정.  
금지: “전체 유의 + B40 비유의 ⇒ 효과가 다르다” (Gelman–Stern).  
B40 표현: “추가 이득을 **명확히 확인하지 못했다**” (CI∋0만으로 “추가 정보 없음” 금지).  
실질 중요도: \(\tau=0.001\) — B40 CI가 \(-\tau\)를 배제하는지 보고.

**Supporting procedures:**

| ID | Role | What |
|---|---|---|
| B1 | primary strata | B40 / \(B40^{c}\) / B45 ΔBrier + \(H\) CI |
| B2 | localization | \(p_{\mathrm{pre}}\) bands, time bands, B40×time (descriptive) |
| B3 | info-set ablation | 동일 B40 eval 행: S0 → econ → +combat → ridge (순서 의존; 고유 기여 과장 금지) |
| B4 | size vs sign | 연속 \(\|\widehat{\Delta V}\|\) all vs B40 (appendix; 주 라벨은 이진 SVI) |

**Evidence lock (cites):**

| Claim | Value | Source |
|---|---|---|
| B40 LGBM − PT | −0.00044 [−0.00146, +0.00066] — **CI∋0** | lift localization / primary |
| \(B40^{c}\) point | ≈ −0.00131 | state-dependent |
| \(H\) | **+0.00087** [−0.00042, +0.00195] — CI∋0; \(P(H<0)=0.095\) | `svi_state_dependent_20260919` |
| Lift not uniform | late \(t{\ge}30\) ≈ −0.00316; high-skew B80 ≈ −0.00243 | lift localization |
| B40 ablation | S0…S3 Brier ≥ sealed PT on B40; no clear beat of PT | state-dependent RQ4 |
| mean\(\|\Delta V\|\) | all 0.110 vs B40 **0.189** | state-dependent RQ5 |
| Flex PT on B40 | \(q-\mathrm{flex}\approx -0.00012\) (near zero) | state-dependent RQ3 |

**Locked reading (Axis B):** 경합(B40)에서는 PT 대비 추가 이득을 **명확히 확인하지 못했다**. \(H\)도 0을 포함해 B40 vs 바깥 차이를 강하게 주장하지 않는다. 점추정·층화는 “이득이 편향·후반·비스큐 쪽에 더 실림”을 **서술**할 뿐이다. 큰 \(\|\Delta V\|\)는 0-근처 부호잡음 **단순 설명만** 약화한다 — 본질적 비예측성 주장 금지.

---

### Validation axis (not an RQ — label integrity)

| Check | Locked cite | Role |
|---|---|---|
| Kill–SVI concordance | agree ≈ 0.937 / disagree ≈ 0.063 (pooled T) | SVI ≠ kill exchange |
| Among disagree, obj⇄SVI | ≈ 0.36 | correspondence hook |
| Quiet matched \(\|\Delta V\|\) ratio | type-B ≈ 2.3–2.7× | engagements move \(V\) more than matched quiet |
| Cross-target | kill→SVI substitute worse than native SVI | SVI not reducible to kills |

Sources: `svi_overnight_20260919`, `svi_cohort_aligned_20260919`, `svi_validation_suite_20260919`, `svi_material_quiet_20260919`.

---

### Optional auxiliary — combination / encompassing

\(p_{\lambda}=(1-\lambda)\mathrm{PT}+\lambda q\), \(\lambda\)는 TRAIN OOF 또는 Q_CAL에서 선택 후 동결.  
Clements–Harvey **보조**. ΔBrier 우위를 encompassing으로 이름 붙이지 않음.

---

### Adopted results paragraph (locked template)

> On the sealed 15.16 evaluation sample, pre-engagement features yield a **small, reproducible** reduction in Brier loss relative to a prior×time baseline (Axis A; LightGBM − PT ≈ −0.00113). Within that predictive regime, **contested** pre-engagement states (\(p_{\mathrm{pre}}\in[0.4,0.6]\)) do **not** show a clearly confirmed incremental gain (Axis B; B40 CI includes 0; \(H\) CI includes 0). Lift localization suggests much of the overall signal concentrates in skewed and late-game strata, not in B40. Larger mean \(\|\widehat{\Delta V}\|\) on B40 indicates that magnitude and sign-predictability must be separated. These findings concern the present information set and learners; they do **not** establish inherent unpredictability of contested states, empty priors, or absence of label noise.

### \(|\widehat{\Delta V}|\) on B40 (careful)

큰 \(|\widehat{\Delta V}|\)는 “모든 사례가 0 근처라 부호가 애매하다”는 **단순 설명만** 약화시킨다.  
시그모이드 \(V=\sigma(z)\)에서 \(\Delta V\approx V(1-V)\Delta z\) 이므로 중간대에서 확률 스케일 변화가 커 보일 수 있으며, \(\mathrm{sign}(\Delta V)=\mathrm{sign}(\Delta z)\).  
권장: “0 근처 부호 불안정만으로는 설명하기 어렵다. 다만 \(\widehat{V}\)/종료시점 민감성까지 배제하지는 않는다.”

### Non-claims

- SVI ≠ fight win; \(\widehat{\Delta V}\) ≠ causal fight effect.  
- Concordance ≠ kill identity; 15.14/15.15 ≠ independent holdout.  
- Continuous \(\widehat{\Delta V}\) = Axis B appendix only, not primary label.  
- Literature / B40 results do **not** prove inherent unpredictability.  
- Do not rename B40 as “empty prior” or “high leverage.”

---

## 4. Paper spine

```
1 Intro     SVI definition; Axis A (predictability) + Axis B (contested)
2 Related   WPA; scoring rules (Gneiting–Raftery); forecast comparison
            (DM; Clements–Harvey; Giacomini–White as framing — not identical tests)
3 Data      210k corpus; 15.16 T eval sample
4 Value ˆV  Train on W; calibration
5 Label     SVI; sign≠magnitude; |ΔV| caveats
6 Predictor q  352 inputs; lean slate; LightGBM winner
7 Evaluation
    7.1 Axis A: same-row table — q vs PT (+ b(p)); model slate; flex PT; transfer
    7.2 Axis B: B40, B40^c, H; lift localization; info-set ablation
    7.3 Validation: concordance + quiet + cross-target
    7.4 Appendix: continuous ΔV (size vs sign)
8 Results   Exploratory — locked readings in §3
9 Discussion  Predictability vs contested difficulty; no empty-prior / high-LI
10 Limits
```

Abstract template:

> We predict **strategic value improvement**—whether a frozen estimated match-win probability revises upward over a kill-conditioned engagement. We ask (A) how much pre-engagement features improve forecasts beyond a prior×time baseline, and (B) whether that incremental value persists in **contested** pre-engagement states.

---

## 5. Evaluation redesign

### 5.0 Corpus vs evaluation sample

| Layer | Population | Use |
|---|---|---|
| **Study corpus** | 210k matches (15.14–15.16) | All rows from here |
| **Prediction eval** | **15.16 T, \(n=32{,}981\)** | Every model in main table; RQ1–2 |
| **Measurement** | Pooled T | Concordance, label mix |
| **Quiet** | Matched state/time/duration | Drift reference |
| **Transfer** | KR/NA1 16.x | Dedicated section |

### 5.1 Primary table + state-dependent block

Same 15.16 rows; match weights; B40 / \(B40^{c}\) / time bands; report \(H\) with CI.

### 5.2–5.3 Material / Quiet

Unchanged from prior lock (decided-set denoms; matched quiet; signed \(\Delta V\)).

---

## 6. Epistemic status

> Exploratory follow-up after prior TEST/external exposure; candidates and rules for *this* comparison frozen before sealed reopen. Not confirmatory. Bootstrap = eval-sample variability under frozen artifacts unless re-fit is stated.

---

## 7. Locked choices

1. Keep binary SVI; continuous \(\widehat{\Delta V}\) = Axis B appendix only.  
2. **Two Intro RQs only:** Axis A (predictability) + Axis B (contested difficulty). B40 win not required.  
3. ΔBrier primary; combination/encompassing optional auxiliary.  
4. Same 15.16 T rows for all prediction claims.  
5. Lean slate; LightGBM primary \(q\); deep expansion secondary.  
6. Transfer: freeze \(\widehat{V}\), \(q\), PT, \(b(p)\), preproc, calibrators.  
7. Evidence cites: primary table + lift localization + state-dependent REPORT; validation from overnight/aligned suites.
