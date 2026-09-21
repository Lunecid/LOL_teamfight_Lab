# 증거 상태 원장 (STATUS)

갱신: 2026-09-21 · 기준 커밋: `688ec0b` (`feature/fight-boundary-pipeline` 동기).
규칙: 본문의 완료·미실행 서술과 부록 E의 상태 표(`latex/tables/gen/tab_status.tex`, `tools/gen_status_table.py`가 이 파일에서 생성)는 이 원장에서 파생된다. 상태를 바꾸면 근거 열을 함께 바꾼다.
상태 어휘: `FROZEN` 동결 주 결과 · `DONE` 실행 완료 · `PASS`/`PARTIAL`/`INCOMPLETE` 검사 결과 · `NOT_RUN` 설계만 있고 미실행 · `BLOCKED` 외부 요인으로 중단 · `DEFERRED` 이연.
"실행하였다"는 DONE/PASS에만, "부분적으로"는 PARTIAL에만, "미실행/이연/차단"은 나머지에만 쓴다. 상태 어휘 탐색(`tools/check_style.py`)은 검토 위치를 알려줄 뿐이며 문장의 의미는 사람이 확인한다.

## 1. 주 결과 (동결; 헤드라인 교체 금지)

| ID | 내용 | 상태 | 근거 |
|---|---|---|---|
| MAIN_T_HEADLINE | T: q 대 PT_flex, 15.16 TEST, ΔBrier −0.00373 [−0.00461, −0.00279] | FROZEN | `docs/REVIEW_RESPONSE_RR12_RESULTS_20260920.json`; G0 재집계 −0.00373234 PASS |
| MAIN_S_HEADLINE | S: q_S 대 PT_flex_S, identity, ΔBrier −0.00335 [−0.00379, −0.00288] | FROZEN | `docs/SCALE_SPLIT_TvsS_RESULTS_20260920.json`; G0 재집계 −0.00335262 PASS |

## 2. 보완 실험 (15.16 노출 이후의 진단; `docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md`)

| ID | 하위 분석 | 상태 | 근거 | 본문 |
|---|---|---|---|---|
| G0_PROVENANCE_RECORD | 실행 출처 기록 (commit, 스크립트 해시, 패키지) | PARTIAL | `docs/G0_INTEGRITY_STATUS_20260921.json` `checks.G0.1_provenance_record` (worktree 모듈 미해시) | 부록 A |
| G0_EVALUATOR_HASH | OOF 평가기 저장 후 해시 기록 | PASS | 동일 `G0.2_evaluator_hash_after_finalize` | 부록 A |
| G0_S_REUSE_GUARD | S 코호트 평가기 재사용·쓰기 보호 | PASS | 동일 `G0.3_s_reuse_and_write_guard` | 부록 A |
| G0_JOIN_KEY | 라벨–교전 결합 키 무결성 | PASS | 동일 `G0.4_join_key_integrity` | 부록 A |
| G0_FOLD_HOLDOUT_SET | fold 홀드아웃 경기 집합 해시 검증 | INCOMPLETE | 동일 `G0.5_fold_holdout_match_set` (outputs 접근 필요) | 부록 A, 8.4 |
| G0_BUNDLE_RELOAD | 평가기 번들 재적재 동일성 | PARTIAL | 동일 `G0.6_bundle_reload_parity` (키·행수 확인, 점수 동일성 미실행) | 부록 A |
| G0_HEADLINE_REAGGREGATE | 동결 예측표에서 헤드라인 재집계 | PASS | 동일 `phase_b.headlines` (T −0.00373234, S −0.00335262) | 부록 A, 6.4 |
| G0_MISSING_DATA_POLICY | 자료 없는 검사는 INCOMPLETE로 표기 | PASS | 동일 `G0.8_missing_data_policy` | 부록 A |
| G0_FUTURE_INFO_INVARIANCE | 사후 사건 제거 후 특징 재구성 불변성 | INCOMPLETE | 동일 `G0.F_future_info_invariance` | 부록 A, 8.4 |
| G0_DIGESTS_RR0 | 동결 아티팩트 12개 digest 대조 | PASS | 동일 `phase_b.digests` (12 MATCH) | 부록 A |
| E1_INFOGROUP_LGBM | 프레임×사건 정보군 LGBM 4 arm (T, S) | DONE | `docs/SUPPLEMENTARY_E1_INFOGROUPS_20260921.json`; T018 | 6.8 |
| E1_MATCHED_LEARNER | 동일 352열 logistic 대 LGBM 정합 비교 | NOT_RUN | 설계 §4.5 (선택 항목); T018 미포함 | 6.8, 부록 E |
| E2_S_HOLD | S_hold 산술 분해 (ΔV = d_clock + d_update) | DONE | `docs/SUPPLEMENTARY_E2_S_HOLD_20260921.json` `full`; T022 | 5.9 |
| E2_FRAME_STRATA | 관측 갱신 2×2 층화 (T, S) | DONE | `docs/SUPPLEMENTARY_E2_FRAME_STRATA_20260921.json` `section_5_4`; T019 | 7.3 |
| E2_QUIET_REMATCH | quiet 대조의 프레임 나이 강화 매칭 | NOT_RUN | 설계 §5.5 (보조) | 부록 E |
| E3_EXT_BOOTSTRAP | 외부 KR/NA1 × T/S 짝 경기 군집 부트스트랩 | DONE | `docs/SUPPLEMENTARY_E3_EXT_BOOTSTRAP_20260921.json` `primary` (md 표의 FAIL은 작성기 렌더 오류); T017 | 7.5 |
| E3_COMMON_MATCH_H | 공통 경기 H = mean(d_T − d_S) | DONE | 동일 `common_match_H` | 7.5 |
| E4_HORIZON_STRATA | 지평 60/90/120 층화·같은 종점 분리 (T, S) | DONE | `docs/SUPPLEMENTARY_E4_SENSITIVITY_20260921.json` `horizon`; T020 | 5.8 |
| E4_PEER_TRANSFER | 동료 평가기(LR) 라벨 이식, 고정 q/PT | DONE | 동일 `peer_label_transfer`; T020 | 5.8, 8.3 |
| E4_PEER_REFIT | 동료 평가기 OOF 라벨로 q/PT 재학습 | NOT_RUN | 설계 §7.2 두 번째 실험 | 부록 E |
| E4_E_FIXED | 대안 정지 규칙 e_fixed = min(L+30 s, 경기 종료−1 ms) | DONE | `docs/SUPPLEMENTARY_E4_E_FIXED_20260921.json`; T023 | 4.6 |
| E4_OAT_CASE_COMPOSITION | 정의 상수 일변량 변경의 사례 구성 census (5,000 경기; 설계 20,000) | DONE | `docs/SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json`; T023 | 4.8 |
| E4_OAT_SCORE_SENSITIVITY | 변경 정의에서 고정 q/PT 점수 민감도 | NOT_RUN | 동일 JSON에 점수 필드 없음 | 4.8, 부록 E |
| E4_OAT_REFIT_MODE | 변경 정의에서 q·PT 재학습 (Refit mode) | NOT_RUN | 설계 §7.5 (선택) | 부록 E |
| E5_MATERIAL_AXES | 동일 사례 물질 축 교차표 (kill, epic, structure, objective, alive) | DONE | `docs/SUPPLEMENTARY_E5_CORRESPONDENCE_20260921.json` `cohorts.*.material_axes`; T021 | 5.7 |
| E5_NEXTOBJ_ASSOC | 다음 엘리트 오브젝트와의 연관 (T; RR5b 라벨 재사용) | DONE | 동일 `cohorts.T.nextobj` | 5.7 |
| E5_CASE_SAMPLE_EXPORT | 사전 셀 hash 표본 CSV | DONE | `docs/SUPPLEMENTARY_E5_DISAGREE_SAMPLES_20260921.csv` | 5.7 |
| E5_MARKET_EVENT_SAME_WINDOW | 같은 창의 교환가치 라벨(v3.3 market_event) 대 SVI | DONE | `docs/SUPPLEMENTARY_E5_MARKET_R0R1_20260921.json` `market_event`; T023 | 5.7 |
| E5_R0R1 | 후속 결과 다항 로지스틱 R0/R1 | DONE | 동일 `r0_r1`; T023 | 5.7 |
| E5_EVENT_PREFIX_TRACE | 질의→사건 접두→V→SVI→q 사례 추적 덤프 | INCOMPLETE | `SUPPLEMENTARY_E5_CORRESPONDENCE` `incomplete.event_prefix_trace` (T023에서도 미실행) | 부록 E |
| E6_RULES_LOCK | 미노출 평가 규칙 사전 잠금 (sha256) | DONE | `docs/SUPPLEMENTARY_E6_LOCK_20260921.json`; T024 | 8.4 |
| E6_COLLECTION | api_patch ≥ 16.16 KR 수집 (목표 500 경기) | BLOCKED | T024: Riot API HTTP 401 | 8.4, 부록 E |
| E6_SCORE_ONLY | 미노출 표본 점수 전용 평가 | NOT_RUN | E6_COLLECTION 차단 | 부록 E |

## 3. 기존 이연 항목 (`docs/journal_manuscript_v1/THESIS_V2_FOLLOWUPS.md`)

| ID | 내용 | 상태 | 비고 |
|---|---|---|---|
| F1_EXT_ADAPTER | 외부 확률 어댑터 | DEFERRED | — |
| F2_NMIN_CELLS | 참여 인원 셀(2/3/4/5+) 재점수 | DEFERRED | — |
| F3_PTFLEXS_REUSE | PT_flex_S 단일 객체 재사용 (위생) | DEFERRED | — |
| F4_TS_EXT_CONTRAST | T–S 외부 대비 | DONE | E3_COMMON_MATCH_H로 실행(공통 경기 대비; 기전 아님) |
| F5_DEFINITION_SENSITIVITY | 정의 상수 민감도 재실행 | PARTIAL | E4_OAT_CASE_COMPOSITION만 실행; 점수 민감도 NOT_RUN |
| MF_INFO_GROUPS | M-F0–M-F3 정보군 모듈 | DONE | E1_INFOGROUP_LGBM |

## 4. 저자 결정 (`\pendingauthor{}` 항목별 추적)

| ID | 항목 | 위치 | 상태 |
|---|---|---|---|
| A1 | pick 제외 사유 문장 확정 | 4.5 | 대기 |
| A2 | 외부 T/S 병기 문장 유지 여부 (E3 구간 추가됨) | 7.4–7.5 | 대기 |
| A3 | 릴리스 태그·DOI·파생 자료 라이선스 | 부록 A | 대기 |
| A4 | 학위명·학위수여년월·심사일·심사위원 | `config/meta.tex` | 대기 |
| A5 | 초록의 핵심 수치 유지 여부 | 초록 | 대기 (현재 유지) |
