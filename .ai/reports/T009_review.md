# T009 검토 (Claude) — 커밋 `aa2249f` (`62f4455` + `source_commit` 갱신)

reviewed_at: 2026-09-21 · reviewer: Claude 세션 · scope: `git diff 06e856a..aa2249f`(스크립트 6개, 결과 문서, rr12 S 문서 4쌍, rrx S 문서, primary_fit `_S`/`_TS`, manifest, 보고서·로그 11개) + rr12 S JSON 4개 + 로그. `outputs/`는 열지 않음.

## 판정
**조건부 통과.** 수치 파이프라인(합집합·페어링·부트스트랩·가드·manifest)은 건전하다. 그러나 S 쪽 arms가 계약 §4의 "identity calibrator"를 벗어난 채(rr12 2단계가 sigmoid를 선택) 보고됐고 보고서가 이 이탈을 표기하지 않았다. **저널 포함 결정(계약 §8)은 T010 완료 후.** T010은 재적합 없이 rr12 S 재실행 3회 + 문서 수정으로 끝난다.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| R1 가드 (rr12) | ✓ | 기본값 외 `--q-model` + 접미사 없음 → `SystemExit`(보고서에 출력). `_qTS_on_T`가 동결 T 문서 3개를 건드리지 않음(`git diff` 0) |
| R2 가드 (rrx) | ✓ | `--cohort-tag S` 접미사 없음 → `SystemExit`; S 마스크 `(cohort==0)&(fine==1)`; Reading 절은 T 전용, S는 Note로 대체(`q DSC still` 0건) |
| G1 manifest 생성기 | ✓ | `ss20260920_scale_split_manifest.py --check-integrity --with-t009`. T008 `integrity`·`row_counts`·`census`·digests 27개가 T008 manifest와 동일; `generator` 필드; `digests_T009` 12개; `EXT_S_census` 차이 0 |
| 합집합 라벨 | ✓ | TRAIN 162 654 (= 39 605 + 123 049) / Q_CAL 41 692 / Q_SELECT 41 254 / TEST 134 186; 키 중복 0; 경기 가중은 적합 시 재계산 |
| 학습기 고정 | ✓ | `_S`·`_TS` 모두 `fixed-learner=logit_state`, lgbm은 진단. PT_flex 선정 knots 4/4, C 0.01 — S 3회 실행 모두 동일(결정적) |
| `_qTS_on_T` (T 모드) | ✓ | 재선정 결과가 동결 T와 동일(PT_flex 4/4/0.01, b_spline knots 6, 보정기 전부 identity) → q_TS − q_T는 identity 쌍으로 계약 부합 |
| 페어링 | ✓ | `align_keys`가 (match, s) 순서 일치를 단언; T 행은 동결 `review_response_rr12_20260920/prediction_table.npz`의 `p_q_base` |
| 부트스트랩 | ✓ | 2 000회, seed 7, 경기 클러스터, `bootstrap_delta_brier` 재사용 |
| primary 1개 · 금지 문구 | ✓ | md에 `primary` 1행; 금지 문구는 Forbidden 절(64·72행)에만 |
| EXT S | ✓ | census 15 641 / 16 100 / 1 307 / 285, 차이 0; rrx는 raw `predict_proba`(보정기 미적용) |
| 수치 전사 | ✓ | `ss20260920_write_results.py`가 JSON 필드에서만 조립 |
| 동결 T 문서·원고·잠금 | ✓ | diff 0 |

## R3 — 필수: 계약 §4 "identity calibrator" 이탈 (미보고)
- 계약 §4 머리말: "no learner re-selection; **identity calibrator**"; arm (2) q_T→S: "frozen `logit_state.joblib` … applied to S rows **as-is**".
- 실제: rr12의 2단계(Q_CAL에 PosSlopeSigmoid 적합 → Q_SELECT에서 identity vs sigmoid 선택)가 S 3회 실행 모두에서 `q_base`·`PT_flex`·`PT_linear`·`b_spline`·`b_linear`에 **sigmoid**를 선택했다(`calibrator_choice` 필드, 로그 "choose sigmoid"). T에서는 같은 절차가 identity를 골라 계약과 우연히 일치했다.
- 영향 (S TEST Brier, 보고값 sigmoid → raw identity):

| Arm | 보고(sigmoid) | raw (`q_base_raw`) | 차이 |
|---|---:|---:|---:|
| q_S | 0.24544 | 0.24581 | 0.00037 |
| q_T→S | 0.24683 | **0.24775** | 0.00092 |
| q_TS | 0.24556 | 0.24605 | 0.00049 |
| PT_flex_S | 0.24890 | TEST raw 미저장 (Q_SELECT에서 sigmoid가 0.0003 개선) | — |

- 주 대비 q_S − PT_flex_S = −0.00346 [−0.00382, −0.00307]는 부호·τ=0.001 초과 모두 강건할 것으로 예상(raw 근사 ≈ −0.0034). **보조 q_S − q_T→S는 −0.00138(sigmoid 쌍) vs ≈ −0.0019(raw 쌍 근사)로 CI 폭보다 크게 달라진다.** q_T→S에 S Q_CAL로 적합한 sigmoid를 씌우면 "그대로 적용"이 아니다.
- 계약 arm (4)의 "same grid and two-stage rule as RR1/RR12"는 baseline에 한해 보정기 단계를 포함하는 것으로 읽힐 수 있다. 계약 문구의 내부 긴장은 인정한다(작성자 = 이 세션). 그러나 q arms에 대한 "identity"·"as-is"는 명확하고, 어느 쪽이든 AGENTS.md상 이탈은 보고 대상이었다.
- **처리(T010):** 계약 문언(전 모델 identity)으로 재계산해 **주 결과**로 두고, 현재 값은 "RR12 2단계 선택 변형(민감도)"으로 병기한다. TEST를 본 뒤 변형을 고르는 것은 §6 금지 3항의 취지에 반하므로 사전 선언 쪽이 주다. 저자가 (b) "T와 동일한 2단계 절차"로 계약을 개정하길 원하면 §4에 날짜 표기 개정 이력을 추가한 뒤 그것을 주로 두되, raw q_T→S 행은 반드시 함께 보고한다.

## R4 — 필수 (작음): 표 1 lgbm 행의 분모
결과 md 표 1 "S TEST"의 `lgbm_state_S (diagnostic)` 행은 n 31 059 / 15 282 = **Q_SELECT** 값(0.2474 / 0.6879 / 0.5563)이다. `find_model(prim, "lgbm_state")`가 첫 블록(Q_SELECT)을 반환한다. TEST 블록(0.2475 / 0.6880 / 0.5560, n 101 205)로 교체하거나 행 라벨을 Q_SELECT로 바꾼다. 협업자 리뷰 ⑤와 같은 분모 오기 유형.

## R5 — 필수 (작음): rr12 S 문서의 하드코딩 T 서술
`rr20260920_review_response_rr12.py` L825 "B40: small exploratory support (CI near 0); do not claim clear tau=0.001 gain."이 무조건 출력된다. S 문서 3개에서는 거짓(S B40 CI [−0.00256, −0.00160]). B40 CI가 0을 포함할 때만 출력하도록 조건화하고 S 문서를 재생성한다. R2와 같은 계열.

## G2 — 필수: RR0 동결 digest 대조 (수용 기준 미완)
manifest `T_freeze_untouched`는 T008 스냅샷 19파일(라벨·oof)만 대조한다. `_qTS_on_T`가 T 모드로 실행됐으므로 RR0 manifest의 동결 sha16을 생성기에서 대조해 `T_frozen_rr0_digests_match`를 기록한다: `logit_state d6e8fc31310a5a58`, `primary_table a80340fede7b6e6a`, `selection_freeze 294a3e4261f9a40e`, `rr12_paired_ci 9484c09c4e097f25`, `rr12_prediction_table b16b1ce417d61f53`, `rr12_PT_flex 15acc630abe5930d`, `rr12_PT_linear 2215150ca4bc0f89`, `rr12_b_spline b34f46c62d72637b`, `rr12_baseline_selection 4b798eed3795e789`.

## 경미
- **M3:** 결과 md `source_commit` = 06e856a, json = 62f4455 (`aa2249f`가 json만 갱신). 재생성으로 해소.
- **M4:** `Q_NEWV_FIT85_PRIMARY_20260920_TS.md`의 "Primary contrast … −0.00422"는 T∪S 합집합 TEST(134 186행) 수치로 계약 셀이 아니다. `_S.md`·`_TS.md` 상단에 배너 1줄 추가(추가만): "T009 파이프라인 부산물; 여기의 'primary'는 T 파이프라인의 PT_linear 대비이며 규모 분리 주 대비가 아님; 합집합 TEST는 계약 셀 아님".

## 결과 읽기 (계약 §6 안에서; R3 재계산 전 잠정)
- **주:** S TEST에서 q_S가 PT_flex_S를 이김. −0.00346 [−0.00382, −0.00307], p_gt0 = 0 (sigmoid 변형). raw 근사도 같은 부호·크기.
- **전이:** 저널 T 모델을 S에 그대로 써도 PT_flex_S를 이김(−0.00207 [−0.00237, −0.00175]; sigmoid 변형) → T 모델의 신호가 S로 일부 전이. S 전문가가 추가 이득(−0.00138 sigmoid; raw ≈ −0.0019).
- **합집합:** S에서 q_TS ≈ q_S (−0.00012 [−0.00024, +0.00001]); T에서 q_TS가 q_T보다 나쁨(+0.00071 [+0.00018, +0.00126]). 합집합은 S에 이득 없고 T에 소폭 비용 → 저널의 T-only 유지와 정합.
- **S∩B40:** −0.00210 [−0.00256, −0.00160]; H = D_B40 − D_outside = +0.00196 [+0.00128, +0.00262]. S 내부에서 B40 이득은 바깥보다 작지만 0과 명확히 구분된다. T의 B40(CI가 0에 닿음)과 나란히 놓지 않는다.
- **EXT S (CI 없음):** KR/NA1 16.13에서 q_S − PT_flex_S = −0.0018 / −0.0020 (raw). T EXT는 +0.0026 / +0.0040였다. 코호트 간 EXT 비교는 사전 선언된 대비가 아니므로 결과 문서 "Suggestions"대로 후속 사전 선언 후보로만 둔다.
- AUC 절대값(q_S 0.577 vs T 0.636) 비교 서술은 §6에 따라 금지.

## 미검증 항목
- `outputs/` 파일 자체는 열지 않았다. 판정은 보고서·결과 JSON·rr12 JSON·로그·스크립트 diff에 근거한다.
- PT_flex_S의 raw TEST Brier는 저장돼 있지 않아 identity 쌍 주 대비는 T010에서 계산한다.
