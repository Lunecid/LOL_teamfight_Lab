# 부록 — 재현 자료 링크

작성: 2026-09-21  
목적: 본문에 코드를 붙이지 않고, 핵심 결정·증거 파일로 연결한다. `outputs/`·원본 NPZ는 열지 않는다 (`AGENTS.md`).

---

## A. 잠금 · 계약

| 문서 | 역할 |
|---|---|
| [`../JOURNAL_FINISH_LOCK_20260920.md`](../JOURNAL_FINISH_LOCK_20260920.md) | 저널 동결·Forbidden |
| [`../RESPONSE_EVIDENCE_MATRIX_20260920.md`](../RESPONSE_EVIDENCE_MATRIX_20260920.md) | RR0–RRX 매트릭스 |
| [`../SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md`](../SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md) | T/S 계약 |
| [`../Q_PREDICTION_DESIGN_CONTRACT_20260920.md`](../Q_PREDICTION_DESIGN_CONTRACT_20260920.md) | q 설계 |
| [`../V_Q_TARGET_LOCK_20260920.md`](../V_Q_TARGET_LOCK_20260920.md) | V/q 목표 |
| [`../PAPER_COHORT_CONTRACT_20260919.md`](../PAPER_COHORT_CONTRACT_20260919.md) | 코호트 |
| [`../COMMON_RESEARCH_SPINE_20260919.md`](../COMMON_RESEARCH_SPINE_20260919.md) | 공통 정의 |

## B. 결과 JSON / MD (수치 원장)

| 파일 | 내용 |
|---|---|
| `../REVIEW_RESPONSE_RR12_RESULTS_20260920.{md,json}` | T 주 대비·bootstrap |
| `../REVIEW_RESPONSE_RR3_*` | quiet |
| `../REVIEW_RESPONSE_RR4_MARGIN_20260920.md` | λ·triad |
| `../REVIEW_RESPONSE_RR5_RR6B_20260920.md` | 물질·horizon |
| `../REVIEW_RESPONSE_RR6A_CORP_20260920.md` | CORP |
| `../REVIEW_RESPONSE_RRX_EXTERNAL_20260920.md` | EXT T |
| `../SCALE_SPLIT_TvsS_RESULTS_20260920.{md,json}` | S 주결과(identity) |
| `../SCALE_SPLIT_RR0_MANIFEST_20260920.json` | S 무결성 |
| `../TEST_BAND_LEDGER_MLP_FIT85_20260920.md` | V 시간상태 |
| `../Q_NEWV_FIT85_PRIMARY_20260920.md` | q 1차 |
| `.ai/CONTEXT.md` | 수치 요약 원장 |

## C. 스크립트 (경로만; 재실행은 작업서 없이 하지 않음)

| 스크립트 | 역할 |
|---|---|
| `scripts/rr20260920_review_response_rr12.py` | RR12 / 코호트 매개변수화 |
| `scripts/rr20260920_q_build_newv_labels.py` | SVI 라벨 |
| `scripts/rr20260920_q_train_oof_mlp_folds.py` | OOF V / q 경로 |
| `scripts/rr20260920_q_newv_primary_fit.py` | q 적합 |
| `scripts/rrx_external.py` | 외부 |
| `scripts/ss20260920_*.py` | 규모분리 유틸 |
| `scripts/rr20260919_v_redesign_fit_wave4_corrected.py` | V 계보(동결 전) |

## D. 저널 원고 · 석사 계획

| 경로 | 역할 |
|---|---|
| `../journal_manuscript_v1/` | 영문 스파인·claim map |
| `../MASTER_THESIS_RESEARCH_PLAN_20260919.md` | 석사 M-RQ·목차 |
| `../journal_manuscript_v1/THESIS_V2_FOLLOWUPS.md` | 미실행 후속 |
| `../tog_manuscript/` | CoG 계보 **읽기 전용** |
| `../lineage_20260915/` | 정의·코호트 바이트 사본 |

## E. 작업 보고서 (실행 정직성)

`.ai/reports/T007.md` … `T013.md`, `.ai/tasks/T00x.md` — 규모분리·집필 패키지 이력.

## F. 부록에 넣지 말 것

- 새 실험 산출물, 체크포인트, 원시 매치 JSON, 비공개 API 키.
- Forbidden 주장을 뒷받침하는 ‘추가 분석’을 부록으로 우회 제시.
