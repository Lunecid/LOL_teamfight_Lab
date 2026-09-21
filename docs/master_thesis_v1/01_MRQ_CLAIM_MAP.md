# M-RQ ↔ 장 ↔ Claim 대응표 (Phase 2)

작성: 2026-09-21  
근거: [`../MASTER_THESIS_RESEARCH_PLAN_20260919.md`](../MASTER_THESIS_RESEARCH_PLAN_20260919.md) §4·§7, [`../journal_manuscript_v1/01_OUTLINE_AND_CLAIM_EVIDENCE.md`](../journal_manuscript_v1/01_OUTLINE_AND_CLAIM_EVIDENCE.md), [`00_INVENTORY_AND_STATUS.md`](00_INVENTORY_AND_STATUS.md)

**번호 규칙:** 저널 **J-RQ1**(측정) ≈ 석사 **M-RQ2**. 양쪽에서 “RQ1”로 혼용하지 않는다.

---

## 1. 연구질문 ↔ 석사 장

| M-RQ | 질문(계획서 고정 문구) | 석사 장 | 주 claim | 공백·미실행 |
|---|---|---|---|---|
| **M-RQ1** | 공개 상태·사건 기록에서 킬 조건부 교전을 어떤 근거로 구성하며, 경계·규모 선택이 사례에 어떤 변화를 만드는가 | 3장 데이터, 4장 사례 정의 | 정의·상수표·T/S 규칙 (Methods §1) | v3.3 전면 재민감도 없음(F5); 파일럿/계보 재사용 |
| **M-RQ2** | 승패 연결 가치모형으로 교전 전후 전략적 가치 개선을 어떻게 정의하며, 관측 결과와 얼마나 대응하고 모형·구간에 얼마나 의존하는가 | 5장 | C1–C8 | — |
| **M-RQ3** | SVI는 교전 전 공개정보로 어느 정도 예측되며, 정보 구성·학습기 선택이 성능·확률 품질에 어떻게 반영되는가 | 6장 | C9–C14, C18–C21 | M-F0–F3 미실행 |
| **M-RQ4** | 예측은 초기 우세에 얼마나 의존하며, 균형·시간·관측·패치·지역이 달라질 때 범위는? | 7장 | C11–C12, C15–C17, C22 | F1 어댑터, F4 T–S EXT 대비 deferred |

서론(1)·관련연구(2)·통합논의(8)는 전 M-RQ를 관통한다.

---

## 2. Claim별 상세

### M-RQ1 — 사례 구성 (3–4장)

| 항목 | 내용 | 출처 | 상태 |
|---|---|---|---|
| 교전 검출 v3.3 | G=13.7s, D=4264u, R=1600, B=15s, M=2 | Methods §1; `tog_manuscript/sec_definition.tex` (RO) | 완료(서술) |
| T 정의 | `cohort==1` ⟺ min(cluster_blue, red) ≥ 4; fine==2 | lineage / PAPER_COHORT | 완료 |
| S 정의 | `cohort==0 & fine==1` (2≤n_min≤3); pick 제외 | SCALE_SPLIT 계약 | 완료 |
| h90 endpoint | L+1000h 상한 등 | LABEL_ENDPOINT / Methods | 완료 |
| 정의 민감도 재실행 | G×D 등 v3.3 코퍼스 | THESIS_V2 F5 | **미실행** — 계보·파일럿만 인용 |

해석 제한: 독립 경계 주석 없으면 ‘실제 한타 탐지 정확도’ 주장 금지. 규모 변경으로 생긴 코호트 간 절대 성능 차이를 순수 정의 효과라 부르지 않음.

### M-RQ2 — 결과 가치 (5장)

| ID | 국문 진술 | 수치 | 아티팩트 | 상태 |
|---|---|---|---|---|
| C1 | 동결 fit85 MLP \(\widehat V\)가 15.16 시간상태에서 경기 승패를 순위화한다 | Brier 0.1552; AUC 0.8542 | RR6a | supported |
| C2 | 초반 밴드가 후반보다 약하다 | [2,10) Brier 0.2283; AUC 0.6642 | RR6a | supported |
| C3 | 교전 시점 V_pre/V_post가 초반 시간상태보다 낫다 | 0.1442 / 0.1181; AUC 0.8750 / 0.9149 | RR6a | supported |
| C4 | ΔV의 방향·부호평균·규모는 별개 객체다 | triad (RR4) | RR4 | supported |
| C5 | 매칭 quiet에서 교전 \|ΔV\| ≫ quiet | 0.0633 [0.0612, 0.0654] | RR3 | exploratory scope |
| C6 | 결정된 킬축 부호와 SVI 일치 ≈0.90 | 0.904 all-T | RR5a | correspondence |
| C7 | 180s 내 결정 엘리트 오브젝트에서 SVI± Blue 비율 차이 | 0.604 vs 0.396 | RR5b | correspondence |
| C8 | 구간 간 SVI flip 낮음; 상당 부분은 동일 endpoint | flip 0.007–0.019 | RR6b | supported |

해석 제한: V→W 품질 ≠ SVI가 독립 한타 정답. ΔV̂ ≠ 인과 기여. 물질 대응 ≠ q 정확도(F5).

### M-RQ3 — 사전 예측 (6장)

| ID | 국문 진술 | 수치 | 아티팩트 | 상태 |
|---|---|---|---|---|
| C9 | 15.16 all-T에서 동결 q가 PT_flex보다 Brier 우세 | −0.00373 [−0.00461, −0.00279]; AUC 0.6403 | RR12 | exploratory |
| C10 | PT_linear 연속성 | −0.00422 [−0.00515, −0.00323] | RR12 | exploratory |
| C11 | B40에서도 소폭 이득 | −0.00214 [−0.00422, −0.00003] | RR12 | exploratory |
| C12 | H=D_B40−D_outside가 명확히 0이 아님을 주장하지 않음 | 0.00181 [−0.00060, 0.00422] | RR12 | exploratory |
| C13 | Brier 이득이 DSC↑·MCB↑와 동행 | ΔMCB +0.00033; ΔDSC +0.00406 | RR6a | decomposition |
| C14 | λ·s_Q로 소 \|ΔV\| 제외해도 all-T 이득 유지 | ≈−0.004, CI가 0 제외 | RR4 | post-hoc |
| C18 | all-S에서 q_S가 PT_flex_S 우세 (identity) | −0.00335 [−0.00379, −0.00288] | SS | exploratory |
| C19 | T→S 신호 일부 + q_S 추가 | −0.00141; −0.00194 | SS | secondary |
| C20 | T∪S 합동은 어느 쪽에도 이득 없음 | −0.00024; +0.00071 | SS | secondary |
| C21 | S∩B40 이득 nonzero | −0.00210; H_S +0.00178 | SS | exploratory |

공백: **M-F0–F3** 정보군 단계 비교 미실행 → 6장에 “향후 심화”로만.

### M-RQ4 — 적용 범위 (7장)

| ID | 국문 진술 | 수치 | 아티팩트 | 상태 |
|---|---|---|---|---|
| C11–C12 | 초기 균형(B40) 의존 (T 내) | 위와 동일 | RR12 | exploratory |
| C15 | 주 EXT KR/NA1에서 q Brier가 PT_flex보다 **나쁨** | +0.0026 / +0.0040 | RRX | supported ordering |
| C16 | EXT에서 DSC는 q 우세·MCB가 손실 지배 | MCB/DSC 표 | RRX | diagnostic |
| C17 | EXT V_pre Brier ≈0.15 | 0.1511 / 0.1514 | RRX | supported |
| C21 | S∩B40 (코호트 내) | 위 | SS | exploratory |
| C22 | EXT S: q_S가 PT보다 나음(점추정) | −0.0018 / −0.0020 | SS rrx | ordering; pending 문장 |

공백: F1 어댑터, F4 T–S EXT 대비 검정 — deferred. 재보정으로 전이 복구 주장 금지(F3).

---

## 3. Forbidden → 장별 주의

| F# | 금지 | 특히 주의할 장 |
|---|---|---|
| F1 | AUC 천장 | 6, 8 |
| F2 | B40=전투숙련 결여 | 7, 8 |
| F3 | 재보정=전이 복구 | 7, 8 |
| F4 | 최적 아키텍처 | 6, 8 |
| F5 | 물질대응=q/ATT | 5, 8 |
| F6 | 규모 기울기·더 예측가능 | 4, 6, 7 |
| F7 | T/S 기전 | 6, 7 |
| F8 | 합동/pick/소EXT 대체 | 6, 7 |

---

## 4. 장별 파일 연결

| 장 | 파일 | M-RQ | Claims |
|---|---|---|---|
| 1 서론 | `ch01_서론.md` | 전체 | 논문 thesis 문장 |
| 2 관련연구 | `ch02_관련연구.md` | 전체 | — |
| 3 데이터 | `ch03_데이터와사전정보.md` | M1 | 정의·특징 계보 |
| 4 사례 | `ch04_교전사례정의.md` | M1 | T/S 규칙 |
| 5 가치 | `ch05_결과가치.md` | M2 | C1–C8 |
| 6 예측 | `ch06_사전예측.md` | M3 | C9–C14, C18–C21 |
| 7 검증 | `ch07_검증과적용범위.md` | M4 | C11–17, C22 |
| 8 논의·결론 | `ch08_논의결론.md` | 전체 | Forbidden 재확인 |
| 부록 | `APPENDIX_재현.md` | — | 스크립트·JSON 링크 |
