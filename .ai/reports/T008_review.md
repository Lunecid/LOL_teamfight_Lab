# T008 검토 (Claude) — 커밋 `325ab76` (`b205cbe` + manifest `source_commit` 갱신)

reviewed_at: 2026-09-20 밤 · reviewer: Claude 세션 · scope: `git diff 1229b1d..325ab76` (스크립트 1개 +17/−3, manifest 신규, 보고서, 로그 4개) + `.ai/reports/T008.md` + `docs/SCALE_SPLIT_RR0_MANIFEST_20260920.json`. `outputs/`는 열지 않음.

## 판정
**통과. T009 진행 가능.** T009 0단계는 R1(T007 검토)에 더해 아래 **R2**·**G1**을 포함한다. 작업서를 벗어난 스크립트 수정(reuse 모드 held 집합)은 **승인**한다 — 사유는 아래.

## 저자 고지 사항(reuse 모드 수정)에 대한 판단
- T007 시점의 reuse 모드는 `fold_matches[k]`를 교전 fold 역할(`load_engagements(fold k)`)의 경기 집합으로 만들었다. 이 집합은 V^(−k)가 실제로 제외한 경기 집합(V-bucket held)의 **진부분집합**이다: `T008_fold_set_compare.txt` 기준 fold0–4에서 `only_eng = 0`, `only_v = 1 003 / 1 054 / 981 / 987 / 1 019`(V 행은 있으나 그 역할에 교전이 없는 경기). 따라서 M2 단언(`len == n_held_matches`)은 13 625 vs 14 628로 즉시 실패했고, 그 이전 stray 검사는 자기참조였다.
- 수정 후 reuse 모드는 refit 경로(L187)와 **동일한 구성식** `D.load_v_rows(L,"MAIN",all_roles,bucket_only=True)` → `sub_role == fold{k}`로 집합을 만든다. 그러므로 `fold_matches[k]` = V^(−k) 적합에서 제외된 경기 집합이며, stray(L217)·bad(L358) 검사가 "교전 경기 ⊂ V-held"를 실제로 검증한다. leave-match 보장은 구조적으로 성립(경기 단위 해시가 V 행과 교전 행에 공통).
- 모델링 로직·평가기·라벨 정의 변경 없음. 작업서의 "스크립트 변경 없음"은 M2 한 줄을 전제로 한 문구였고, 이 수정은 M2의 취지(V 학습 제외 집합과의 일치 확인)를 그대로 실현한 것이다. → 승인, 작업서 이탈로 기록만 남김.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| M2 단언 | ✓ | fold0–4 `len(fold_matches)` = `n_held_matches` = 14 628 / 15 017 / 14 796 / 14 769 / 14 958. T `Q_NEWV_FIT85_OOF_META_SLIM_20260920.json`의 fold별 `n_held_matches`와 동일 → 같은 V-held 분할 |
| S census | ✓ | TEST 101 205 / Q_CAL 31 302 / Q_SELECT 31 059 / TRAIN fold 24 030 · 25 005 · 24 520 · 24 690 · 24 804 (Σ 123 049); `n_missing` 0; 계약 §2 census와 차이 0 |
| 키 분리 S∩T | ✓ | `n_S_keys` 286 615 (= 4 역할 합), `n_T_keys` 93 171 (= 32 981 + 10 390 + 10 195 + 39 605), 교집합 0 |
| fold 분리 | ✓ | fold별 S 경기 11 962 / 12 352 / 12 138 / 12 138 / 12 230 ⊂ V-held; T 경기 5 732 / 5 858 / 5 725 / 5 845 / 5 829 ⊂ V-held. T 값은 계보 `docs/lineage_20260915/cohort_manifest.json`의 TRAIN fold `T_matches`와 정확히 일치(교차 계보 정합) |
| 값 일관성 | ✓ | `Y == (ΔV > 0)`, ΔV ≈ p_post − p_pre(최대 잔차 3e−8), B40 재계산 일치, `n_exact_zero` TEST 0 / Q_CAL 0 / Q_SELECT 1 / TRAIN 1 |
| `pre_ok` | ✓ | 라벨 npz에 열이 없는 것이 맞음 — builder L112·OOF L355에서 `pre_ok==1 & valid_h90==1` 필터 후 저장. manifest `pre_ok_note`로 기록 |
| T 동결 무변경 | ✓ | 19파일 pre/post 스냅샷 동일(`T008_T_freeze_pre.json`). RR0 manifest와 교차: `TEST_h90` `0a2e977b231acf85`, `TRAIN_oof_STATUS` `f924a5e2be9cf217`, `TRAIN_oof_h90_meta` `14870768b7fde836`, fit85 번들 `ac459cc4397630a9` 일치 |
| role_tag / contract / mask | ✓ | `EXPLORATORY_SCALE_SPLIT_PRIOR_TEST_EXPOSURE`; `cohort==0 & fine==1 & valid_h90==1`; `evaluator_policy` 기록 |
| 변경 파일 범위 | ✓ | `git diff --name-status 1229b1d..325ab76`: 스크립트는 OOF 1개만; 원고·잠금 문서·`docs/tog_manuscript/` 변경 없음 |

## G1 — 필수 (T009 1단계 전)
manifest의 `integrity` 블록(키 분리, fold 분리의 `V_sha16`/`S_match_sha16`/`T_match_sha16`, 값 검사, T freeze 스냅샷)을 만든 **생성 스크립트가 커밋되지 않았다**(`scripts/` 변경은 OOF 스크립트뿐). T009가 같은 manifest를 갱신하므로:
- `scripts/ss20260920_scale_split_manifest.py`로 커밋하고, manifest에 `generator` 필드(스크립트 경로·인자)를 추가한다.
- 재실행 시 `integrity` 블록의 모든 값(sha16 포함)이 현재 manifest와 동일해야 한다. 다르면 BLOCKED.

## R2 — 필수 (T009 6단계 전)
`scripts/rr20260920_review_response_rrx_external.py`(EXT dual-stage)는 CLI 인자가 없고 다음을 하드코딩한다:
- 경로 `QOUT`/`RR12`/`LAB`/`OUT`(L31–36)과 **동결 T 문서** `docs/REVIEW_RESPONSE_RRX_EXTERNAL_20260920.md`(L457) — S 실행이 T 문서를 덮어쓴다.
- 코호트 마스크 `coh["cohort"] == 1`(L117) — S에는 `(cohort==0) & (fine==1)`이 필요.
- "Reading" 절(L440–455)이 T 서술("KR/NA1 16.13: q DSC still > PT_flex, but larger ΔMCB → net ΔBrier > 0 …")을 **상수 문자열**로 출력 — S 결과에는 참이 아닐 수 있는 문장이 자동 기입된다.

필요 조치(기본값 T 동작 불변): `--cohort-tag {T,S}`(S 마스크 + S 경로 `_S`), `--out-suffix`, 기본값 외 실행에서 접미사가 비면 `SystemExit`("would overwrite frozen outputs"), S 실행에서는 하드코딩 Reading 절을 출력하지 않음(S 해석은 T009 보고서에서 JSON 필드로만 작성). EXT S 행 수를 계약 §2 census와 대조: KR 16.13 15 641 / NA1 16.13 16 100 / KR 16.15 1 307 / KR 16.14 pilot 285.

## 설계 메모 (T009 보고서 작성 시 참고; 금지 해석 아님)
- S TEST의 B40 셀은 31 675 / 101 205 ≈ 31.3 %(T는 5 423 / 32 981 ≈ 16.4 %). 셀 크기·비율이 코호트 간에 크게 다르므로 B40 결과는 **코호트 내부 대비**로만 서술하고 T와 나란히 놓지 않는다.
- P(SVI=1) 경기가중(S): TRAIN 0.4982 / Q_CAL 0.5062 / Q_SELECT 0.5139 / TEST 0.5107.
- 합집합(q_TS) TRAIN = 39 605 + 123 049 = 162 654행; 경기 가중은 합집합 안에서 재계산(계약 §4).
- M1(T007 검토, refit 경로의 sha 계산 시점)은 미해결이나 reuse 모드만 쓰는 현재 계획에는 영향 없음.

## 미검증 항목
- `outputs/` 파일 자체는 열지 않았다(작업 원칙). 판정은 보고서·manifest·로그·스크립트 diff에 근거한다.
- manifest `generated` 시각(23:51:35 KST)은 `b205cbe` 생성 시각이며 `325ab76`은 `source_commit` 필드만 바꿨다(재생성 아님). G1 재실행 시 `generated`·`source_commit`은 갱신되어도 됨.
