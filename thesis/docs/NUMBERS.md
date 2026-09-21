# 수치 레지스트리 (NUMBERS)

`latex/config/numbers.tex` 의 매크로 값과 출처. 규칙(`AGENTS.md`): 새 실험·재집계·재학습으로 수치를 만들지 않는다. 모든 값은 아래 출처 문서의 필드·행에서 옮긴 것이며, 값을 바꿀 때는 출처를 함께 갱신한다.
계보 태그: **F** = fit85/SVI 주 결과(동결), **L** = 정의 계보 (`docs/tog_manuscript/`, `market_event` 라벨, 인용 전용), **O** = 2026-09-15 구 로지스틱 V (인용 전용), **S** = 보완 실험(15.16 노출 이후의 진단; 헤드라인 대체 금지). 네 계보의 수치는 한 표에 함께 놓지 않는다. 실행 상태는 [`STATUS.md`](STATUS.md).

| 매크로 군 | 출처 파일 → 필드/행 | 계보 |
|---|---|---|
| `\numMatches*`, `\numEng*`, `\numT{Train,Val,Test,Pooled}`, `\numN*` | `docs/lineage_20260915/README.md` L82–86 (`e0ec3d0:docs/tog_delta_v_20260916/manuscript.md` L200–208) | F (코퍼스) |
| `\numTQcal…\numTVselMatches`, `\numTTestMatches`, `\numTBforty*`, `\numTExt*` | `docs/PAPER_COHORT_CONTRACT_20260919.md` §2, §4; `docs/lineage_20260915/cohort_manifest.json` `sets.<SET>.by_sub_role.<ROLE>.h90.T_matches`; `REVIEW_RESPONSE_RR12_RESULTS_20260920.json` `census`; `REVIEW_RESPONSE_RRX_EXTERNAL_20260920.md` | F |
| `\numS*`, `\numUnionTrain*`, `\foldHeld*` | `docs/SCALE_SPLIT_RR0_MANIFEST_20260920.json` `row_counts`, `integrity`; `cohort_manifest.json` `sets.MAIN_TEST.skirmish`; `.ai/reports/logs/T009_union.txt`; `.ai/reports/T008.md` | F |
| `\numV*` | `docs/A_MLP_expanded_evaluator_meta_20260919.json` `meta.census`; `TEST_BAND_LEDGER_MLP_FIT85_20260920.md` | F |
| `\constG*`, `\ari*`, `\numMatchesDef`, `\numIntervalsDef` | `docs/tog_manuscript/sec_definition.tex` §Kill episodes (`config/fight_boundary/spec_pooled.json`, `details_pooled.json`) | L |
| `\constD*`, `\numPairsDef`, `\share{Linked,Separated}Pairs` | 동일 §The spatial boundary D | L |
| `\constR`, `\constB`, `\constM`, `\gateShare*`, `\numEngGateCog`, `\numEngMthree` | 동일 §Presence gate; `presence_gate_summary.json` | L |
| `\constGpatch*`, `\constDpatch*` | 동일 §Re-estimation per patch (`config/fight_boundary/drift.json`) | L |
| `\shareNmin*`, `\num{Pick,Skirmish,Teamfight}Lineage`, `\share{Pick,Skirmish,Teamfight}`, `\cutDiff*`, `\numEngLineage`, `\numMatchesLineage`, `\aucLineageOOF` | 동일 §Scale classes, §Validity summary | L |
| `\sharePickTest` (≈18 %) | `docs/journal_manuscript_v1/04_DISCUSSION.md` §6 | F |
| `\killless*`, `\killlessSchubert` | 동일 §What a kill-anchored definition cannot see | L |
| `\followup*`, `\shareCapReached`, `\shareNextKillStop`, `\shareNoNewFrame`, `\shareSmallDv*`, `\signDis*` | `docs/lineage_20260915/README.md` (manuscript.md L351–358; 구 V 아래 계산, 결과 시점은 V 무관) | O/F |
| `\dim*`, `\hidden*`, `\lrV`, `\wdV`, `\batchV`, `\maxEpochsV`, `\patienceV`, `\holdoutV`, `\calib*`, `\bundleSha`, `\Ltime*`, `\gruGap` | `docs/A_MLP_expanded_evaluator_meta_20260919.json` (`preproc_dims`, `calibration`, `fit_scope`); `BAND_LEDGER_WAVE4_CORRECTED_20260919.md` L44–54; `V_EVALUATOR_FREEZE_CLOSE_20260920.md` | F |
| `\vw*`, `\vpre*`, `\vpost*`, `\numVEarly` | `TEST_BAND_LEDGER_MLP_FIT85_20260920.md` L5–12; `REVIEW_RESPONSE_RR6A_CORP_20260920.md` rows `V_to_W_*` | F |
| `\triadHighP` | `REVIEW_RESPONSE_RR4_MARGIN_20260920.md` triad | F |
| `\quiet*`, `\numQuiet*`, `\shareLongEng` | `REVIEW_RESPONSE_RR3_CLOSEOUT_20260920.md`, `..._RR3_QUIET_20260920.json` | F |
| `\killAgree*`, `\objBlue*`, `\objGameEnded` | `REVIEW_RESPONSE_RR5_RR6B_20260920.md` RR5a 교차표, L35–38 | F |
| `\flip*`, `\sameEp*`, `\peerAgree*` | 동일 horizon 표; `CONTINUITY_LEDGER_MLP_FIT85_20260920.md` L15–27 | F |
| `\dimInputQ`, `\dimInputLgbm`, `\knots*`, `\ridgeC`, `\splineDeg`, `\numBootDraws*`, `\bootSeed`, `\tauMeaningful` | `journal_manuscript_v1/02_METHODS_CORE.md` §4, §6; `REVIEW_RESPONSE_RR1_EXECUTION_ADDENDUM_20260920.md`; RR12 json `PT_flex`, `b_spline` | F |
| `\q*`, `\ptflex*`, `\ptlin*`, `\dBrierT*`, `\het T`, `\qMCB`, `\qDSC`, `\ptflexMCB`, `\ptflexDSC`, `\dMCBT`, `\dDSCT`, `\posRateT`, `\shareTBforty` | `REVIEW_RESPONSE_RR12_RESULTS_20260920.json` `TEST_all.*`, `TEST_B40.*`, `bootstrap.*`, `TEST_all.constant.p_pos`; `RR6A_CORP` rows `q_to_SVI` (5 dp) | F |
| `\qS*`, `\ptflexS*`, `\ptlinS*`, `\dBrierS*`, `\dBrierTtoS*`, `\dBrierSvs*`, `\dBrierTSvsT*`, `\hetS*`, `\qSMCB`, `\qSDSC`, `\ptflexSMCB`, `\ptflexSDSC`, `\dMCBS`, `\dDSCS`, `\posRateS`, `\shareSBforty` | `docs/SCALE_SPLIT_TvsS_RESULTS_20260920.json` `table1_S_TEST_identity`, `table2a_identity`, `table2b_sensitivity`, `table3_CORP_identity`; `REVIEW_RESPONSE_RR12_RESULTS_20260920_S_{qS,qT,qTS}_id.json` | F |
| `\lambdaDeltaApprox`, `\numRowsLambdaOne`, `\sharePonlyFallback` | `REVIEW_RESPONSE_RR4_MARGIN_20260920.md` λ 표 | F |
| `\extT*` | `REVIEW_RESPONSE_RRX_EXTERNAL_20260920.md` 요약표; `Q_NEWV_FIT85_TRANSFER_16X_20260920.md` L19–30 (PT_linear 계열) | F |
| `\extS*` | `SCALE_SPLIT_TvsS_RESULTS_20260920.json` `table4_EXT_S`; `REVIEW_RESPONSE_RRX_EXTERNAL_20260920_S.md` | F |
| `\old*` | `e0ec3d0:outputs/cohort_role_training_20260915/REPORT.md` (SCALE_SPLIT 계약 §1에 인용) | O |
| `\meSkirmAUC`, `\meTeamfightAUC` | `docs/tog_manuscript/sec_prediction.tex` (SCALE_SPLIT 계약 §1에 인용) | L |
| `\cog*`, `\aucCog` (Baek & Kwon 2026 정의 상수; 2장 인용 문장 전용) | `docs/CoG2026_Paper.md` §3.1–3.2·상수표; `docs/tog_manuscript/sec_label.tex` L56–60 (fourteen weights); `sec_intro.tex` L111–120 (AUC 0.675) | L |
| 보완 실험 E1–E5·G0 (`config/numbers_supp.tex`, `tables/gen/*.tex`; 생성물) | `docs/SUPPLEMENTARY_E*_20260921.json`, `docs/G0_INTEGRITY_STATUS_20260921.json` → 매크로별 통계량·분자·분모·필드는 [`NUMBERS_SUPP.md`](NUMBERS_SUPP.md) (`tools/gen_supp.py`) | **S** (보완, 진단) |

## 알려진 문서 간 불일치와 채택 값 (`.ai/CONTEXT.md` 기준)

- B40 CI 상단 세 가지 인쇄 → **RR12 값** (−0.00003) 채택, RR4 재추출(−0.0001)은 본문에 병기.
- 외부 ΔBrier 두 계열 → 표는 **q−PT_flex (RRX)**; PT_linear 계열은 문장으로만.
- PT_linear 연속성 CI 두 가지 (800회 vs 2,000회) → **RR12 (2,000회)** 채택.
- 첫 시간대 라벨 `t_0_10` vs `t_2_10` → **[2,10)** (t<2분 제외).
- CORP q/PT_flex MCB 4 dp (0.0009/0.0005) vs 5 dp (0.00086/0.00053) → **5 dp**로 한 번만 인쇄.
- `PAPER_COHORT_CONTRACT` §5–§6 수치는 구 V 계보이므로 헤드라인으로 쓰지 않음.

## 이 논문에 없는 수치

- 정보군 모듈 M-F0–M-F3, 외부 어댑터 F1, 참여 인원 셀 F2, 정의 상수 재민감도 F5 — 미실행 (`latex/chapters/appendix/E_deferred.tex`).
- 표본 흐름의 중간 단계 수(검출 → 계급 → 유효 → 유한 점수) — 동결 문서에 기록 없음, 추정하지 않음.
