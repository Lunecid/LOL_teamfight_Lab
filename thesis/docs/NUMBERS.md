# 수치 레지스트리 (NUMBERS)

`latex/config/numbers.tex` 의 각 매크로에 대한 값·의미·출처·상태를 기록한다.
값을 바꿀 때는 이 표의 출처(실행 ID, 로그, 커밋)를 함께 갱신한다.

상태: **P** = CoG 2026 논문 보고치, **C** = 감사 후 공개 코드 산출치, **R** = 재실행 예정, **V** = 검증 진행 중

| 매크로 | 값 | 의미 | 출처 | 상태 |
|---|---|---|---|---|
| `\numMatches` | 206,442 | 총 경기 수 | 논문 §III; `docs/AUDIT.md` (국소화 수정과 무관) | P=C |
| `\numMatchesPatchA/B/C` | 73,331 / 73,484 / 59,627 | 패치 15.14 / 15.15 / 15.16 경기 수 | 동일 | P=C |
| `\numEngagementsPaper` | 1,115,123 | 논문 실험 시점 교전 수 (수정 전 국소화) | `docs/AUDIT.md` | P |
| `\numEngagementsCorrected` | 994,365 | 감사 후 코드 교전 수 | `docs/AUDIT.md`, `README.md` 재현성 노트 | C |
| `\engPerMatchPaper` / `\engPerMatchCorrected` | 5.4 / 4.8 | 경기당 교전 수 | 동일 | P / C |
| `\corpusReductionPct` | 11 | 수정에 따른 말뭉치 감소율 (%) | `docs/AUDIT.md` | C |
| `\numSubsample` | 100,000 | 분할·시드당 균등 부표본 | `core/config.py::GLOBAL_SUBSAMPLE_PER_SPLIT` | P=C |
| `\numTests` | 376 | 단위 시험 수 | `pytest` (`docs/AUDIT.md`) | C |
| `\overlapPct` | 3.67 | 시간 겹침 교전 비율 (%) | `docs/EXPERIMENT.md` §9.2 (11,037 건) | P |
| `\aucLGBM` | .675 | LightGBM 시험 AUC | 논문 Table; 3 시드 평균 | P |
| `\aucLGBMCorrected` | .669 | 감사 후 코드 LightGBM 시험 AUC (시드 7) | `README.md` 재현성 노트 | C |
| `\aucMLP` | .626 | 정합 입력 MLP | 논문 (`analysis/mlp_ablation.py`) | P=C |
| `\aucBiGRU`, `\aucLayered` | .581 / .581 | BiGRU, 계층적 융합 | 논문 | P |
| `\aucTransformer`, `\aucXAttn` | .576 / .571 | Transformer, Cross-Attn | 논문 | P |
| `\aucSTGNN`, `\aucGraphSAGE` | .569 / .569 | ST-GNN, GraphSAGE | 논문 | P |
| `\aucNeuralLow/High` | .569 / .581 | 비테이블 신경망 표현의 범위 | 위 값에서 유도 | P |
| `\gapRepresentation` | 0.045 | MLP − 최선 비테이블 신경망 | .626 − .581 | P |
| `\gapLearner` | 0.049 | LightGBM − MLP (정합 입력) | .675 − .626 | P |
| `\aucDropTotal/Loc/Label` | 0.006 / −0.0017 / −0.0038 | 감사 수정에 따른 LightGBM AUC 변화 | `docs/AUDIT.md` | C |
| `\aucEarly/Mid/Late` | .616 / .712 / .807 | 국면별 LightGBM AUC | 논문 §III-E | P |
| `\aucClose/Moderate/OneSided` | .621 / .680 / .796 | 골드 격차별 LightGBM AUC | 논문 §III-E | P |
| `\shapSupportPaper/Rerun` | 17.6 / 17.0 | 서포터 역할 SHAP 기여 비율 (%) | 논문 Table VI / `analysis/shap_role_rollup.py` 시드 7 | P / C |
| `\numReplays`, `\numReplaysExcluded` | 500 / 12 | 2026 리플레이 수, 10 분 미만 제외 수 | `docs/REPLAY_VALIDATION_2026.md` | V |
| `\numGoldMatches`, `\numGoldBins`, `\goldSeed` | 100 / 10 / 20260715 | 골드셋 표본 | 동일 | V |
| `\goldPatchPublic`, `\goldPatchApi` | 26.13 / 16.13 | 골드셋 패치 | 동일 | V |
| `\numGoldCandidates`, `\goldCandidatesPerMatch` | 480 / 4.8 | 골드셋 검출기 후보 구간 | 동일 (기술적 통과; 정확도 아님) | V |

## 아직 매크로가 없는 수치 (채워질 때 추가)

- 표 9.3 절제: 처치별 $\Delta_i$, 95 % CI, DeLong $p$, Holm 유의 → `\ablDelta<T>`, `\ablCiLo<T>`, … (R)
- 표 9.4 SHAP: 역할 × 신호 비율 → `\shap<Role><Signal>` (R)
- 표 9.5 검출기: 정밀도·재현율·$F_1$·IoU·온셋 오차·$\kappa$ → `\detPrec`, … (V)
- 재실행 시 표 9.1–9.2 전체 (R)

## 설계 상수 (매크로 없음)

알고리즘·관측 창 상수(18 s, 4,000 단위, 10 s, 1,800 단위, 15 s / 2,000 단위, 60 s, 30 s, 5 s, 6 구간, 76/26/44, K=64, D_e=12, β=2.0)는
`core/config.py` 가 진실 공급원이며 `tables/tab_app_detection.tex`(부록 C) 와 `tables/tab_notation.tex`(표 3.1) 에 나열한다.
값을 바꾸면 그 두 표와 본문 언급을 `grep` 으로 찾아 함께 고친다.
