# 정의와 근거 (incremental q training 2026-09-15)

상태 어휘: **실행·검증** = 이번 실행에서 수행하고 validation.json 검사로 확인; **상속** = 부모 산출물 그대로 사용(재계산 없음, hash 확인); **연기** = 이번 범위에서 실행하지 않음.

## 진행 전/후

- 전: 기존 T/N specialist 후보군(상수, p_pre 단일 logistic/spline, ridge C=.01, 경제 입력 LightGBM 250 trees)과 pooled q가 h90 참조였다. p_pre와 시간의 상호작용 기준선, C 탐색 logistic, 352입력 LightGBM 조기종료 family는 없었다.
- 후: T/N 각각 PT·전체 logistic·전체 LightGBM 3 family × 18 후보를 적합·선정·동결(2026-09-15 21:01:36)하고 MAIN TEST와 외부 4세트에서 평가했다. validation.json 실패 0개.

| 용어 | 정의 | 근거 파일 | 상태 |
|---|---|---|---|
| Y (h90) | 1[V(endpoint_h90) − V(s−1ms) > 0], 같은 V adapter 양 끝; 정확히 0이면 0 | full_corpus_training_20260915/labels/*_labels.npz | 상속 |
| p_pre | V(S_pre) = P(최종 Blue 승리 W=1 \| S_pre) 추정, TRAIN은 자기 경기 제외 OOF fold adapter, 그 외 final V | labels npz adapter_id/sha256; selection *.oof_provenance | 상속·검증 |
| q | P(ΔV>0 \| S_pre) = Y의 확률 추정. p_pre 균형(B40/B45)은 Y 균형을 함의하지 않음; Y 기준선은 측정 양성률·상수 Brier로만 기술 | eval/results.json | 실행·검증 |
| T / N | T: 알려진 min(cluster_blue, cluster_red) ≥ 4; N: 알려진 min < 4 | cohort_role_training_20260915/cohorts | 상속 |
| 352 입력 | q_pre_only_schema.json predictor_sets["ridge"] 순서 그대로 | protocol.json inputs | 실행·검증 |
| PT 설계 | SplineTransformer(n_knots=5, degree=3, quantile, include_bias=False, extrapolation=constant) p_pre_V·time_minutes 각 6열 + 36 곱(p major) = 48열 → 표준화 → 가중 logistic | selection/pt_*.json design | 실행·검증 |
| 경기 가중치 | 각 적합/평가 부분집합 안에서 경기 총합 동일, 행 평균 1 | selection weights, eval 셀 | 실행·검증 |
| stop10 | int(sha256('iq20260915_stop:'+match)[:8],16) mod 10 == 0 인 TRAIN 경기 | internal_stop/lgbm_*.json | 실행·검증 |
| 조기종료 | stop10 경기가중 Brier, patience 50, 동점은 이른 반복; 선택 반복 수로 전체 TRAIN 재적합 | internal_stop | 실행·검증 |
| 보정 | raw / sigmoid(logit(clip 1e-8) LogisticRegression C=1e6) / isotonic(clip), cohort Q_CAL | selection calibrators | 실행·검증 |
| 선정 규칙 | cohort Q_SELECT 경기가중 Brier → log loss → 후보 이름 | selection/*.json, frozen_manifest.json | 실행·검증 |
| B40 / B45 | .40 ≤ p_pre ≤ .60 / .45 ≤ p_pre ≤ .55 (경계 포함) | eval/predictions cell__* | 실행·검증 |
| 시간 구간 | time_minutes [0,10), [10,20), [20,30), [30,∞) | eval/predictions cell__time_* | 실행·검증 |
| 부트스트랩 | 경기 단위 1000회, seed 20260915, 셀 내 모든 모델 동일 draw, 퍼센타일 95% | eval/results.json bootstrap | 실행·검증 |
| 기존 참조 | A_<set>_h90_<cohort>.npz의 spec_<동결 선택>, pooled, spec_p_pre_spline/logistic/constant | eval checks.legacy | 상속·검증 |
| h60/h120 새 적합, MLP, CoG, SHAP, V 재적합 | — | — | 연기 |

## 주요 수치 근거

- MAIN TEST T 전체 Brier: PT 기준선(신규) 0.22952, 전체 logistic(신규) 0.22840, 전체 LightGBM(신규) 0.22839, 기존 A specialist 0.22817, 기존 pooled q 0.22983, 기존 p_pre spline 0.22979, 기존 p_pre logistic 0.23030, 기존 상수 0.25000 (eval/results.json, 예측 `eval/predictions/MAIN_TEST_h90_T.npz`).
- MAIN TEST N 전체 Brier: PT 기준선(신규) 0.24504, 전체 logistic(신규) 0.24269, 전체 LightGBM(신규) 0.23869, 기존 A specialist 0.24268, 기존 pooled q 0.24310, 기존 p_pre spline 0.24634, 기존 p_pre logistic 0.24736, 기존 상수 0.25002 (eval/results.json, 예측 `eval/predictions/MAIN_TEST_h90_N.npz`).

## 무결성

- 부모 snapshot diff all_equal = True; 봉인 세트 접근은 동결 이후만(validation.json `sealed_accesses_only_after_freeze` = True).
- 명세 sha256 `c98fe463930e6ee7ff82d03f8d25cf3d28a2f869d29fd0988611b64be9ec7b2d`, protocol sha256 `2e47dc86a9e3535d21c91d84c668f952dd52079c5230c30dc1d3a1bedb33dae8`, frozen_manifest sha256 `60527473bbca5b10f808c05d11e4fc6fa45c94e89e134d51c59faa9f4a5784d5`.

