# T007 검토 (Claude) — 커밋 `7cee54e`

reviewed_at: 2026-09-20 밤 · reviewer: Claude 세션 · scope: 4개 스크립트 diff(`09ecfe6..7cee54e`) + `.ai/reports/T007.md`

## 판정
**통과. T008 진행 가능.** T009의 3~4단계(rr12를 여러 q 모델로 반복 실행)는 아래 **R1**을 먼저 적용해야 한다.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| (a) 기본값(T) 동작 불변 | ✓ | 4개 스크립트 모두 기본 경로·마스크·ROLE·문구 동일. 보고서: TEST/Q_CAL/Q_SELECT 재생성 npz 바이트 동일, RR0 manifest `TEST_h90` sha16 `0a2e977b231acf85` 일치 |
| (b) S 마스크 | ✓ | `(cohort==0) & (fine==1) & (valid_h90==1)`; `fine` 부재 시 `SystemExit`; 메타에 `cohort`·`cohort_mask_rule` 기록 |
| (c) `--reuse-evaluators` | ✓ | fold joblib + `bundle_oof_fold{k}` 둘 다 로드, 하나라도 없으면 `SystemExit`(자동 refit 없음); 평가기는 항상 T `oof_evaluators/`; `PosSlopeSigmoid`를 `ok`·`coef_`·`intercept_`만으로 재구성 — 클래스가 그 셋만 씀(확인) |
| (d) S census | ✓ | `EXPECTED_ENG_S` 5개 값, 불일치는 경고만 |
| (e) 보고서의 검증 출력 | ✓ | py_compile, data_root·`fine` 존재, oof 파일 10개 존재, sha 대조 |
| (f) 모델링 로직 | ✓ | 후보·격자·선정 규칙·부트스트랩 불변 |
| reuse 경로 런타임 안전성 | ✓ (정적) | 점수 블록이 참조하는 변수가 reuse 모드에서 모두 정의됨(AST 검사). 아직 실행되지 않았으므로 T008이 첫 실행 |

## R1 — 필수 (T009 3~4단계 전; T008에는 영향 없음)
`scripts/rr20260920_review_response_rr12.py`: `--q-model`로 다른 q를 지정해도 출력이 코호트 태그만으로 정해진다. T009는 S 행에서 q_S / q_T→S / q_TS 세 번, T 행에서 q_TS 한 번을 돌리므로 **서로 덮어쓰고**, 특히 `--cohort-tag T --q-model <q_TS>`는 **동결된 T 산출물** `outputs/review_response_rr12_20260920/`과 `docs/REVIEW_RESPONSE_RR12_RESULTS_20260920.{md,json}`을 덮어쓴다.
- 추가: `--out-suffix ""`(기본 빈 문자열). 비어 있지 않으면 `OUT` 디렉터리명과 두 docs 파일명 모두에 접미사(예: `_qTS_on_T`)를 붙인다.
- 가드: `--q-model`이 주어지고 그 경로가 기본 `<QDIR>/models/logit_state.joblib`와 다른데 `--out-suffix`가 비어 있으면 `SystemExit("non-default --q-model requires --out-suffix (would overwrite frozen outputs)")`.
- 검증: 기본값 실행 경로가 바뀌지 않았음을 `py_compile` + 경로 문자열 확인으로 보고.

## M2 — 권장 (T008 실행 전, 한 줄)
reuse 모드의 `fold_matches[k]`는 교전 fold 역할에서 만든 집합이라 이후 stray 검사가 자기참조다. fold joblib에 저장된 `n_held_matches`와 크기 일치를 확인하면 "V 학습에서 제외된 경기 집합"과 같은 집합임을 검증할 수 있다:
`assert len(fold_matches[k]) == int(fold_obj["n_held_matches"]), (k, len(fold_matches[k]), fold_obj["n_held_matches"])` — `fold_obj` 로드 직후에 추가. 참고값(`docs/Q_NEWV_FIT85_OOF_META_SLIM_20260920.json`): fold별 14,628–15,017.

## M1 — 경미 (refit 경로만; 지금은 불필요)
refit 분기에서 `evaluator_sha`를 첫 dump 직후 계산하고, 점수 후 `n_eng_labeled`를 넣어 다시 dump하므로 기록된 sha16이 최종 파일과 다르다. 원래처럼 점수 후 한 번만 dump하고 그때 sha를 계산하면 된다. T 재적합은 계획에 없으므로 T008/T009와 무관.

## 참고
- `selection_freeze.json`·`baseline_selection.json`·라벨 메타에 키가 추가됨(`cohort_tag`, `fixed_learner`, `cohort_mask_rule` 등). 스키마 추가일 뿐이며 동결 T 파일은 재실행하지 않는 한 불변.
- 다음: T008 (`--cohort S` 라벨 + `--reuse-evaluators`), 보고서·manifest push → 검토.
