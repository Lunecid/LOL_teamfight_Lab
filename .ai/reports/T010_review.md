# T010 검토 (Claude) — 커밋 `69130a4` (`e753fb4` + `source_commit` 갱신)

reviewed_at: 2026-09-21 · reviewer: Claude 세션 · scope: `git diff 78695a6..69130a4`(스크립트 4개, 결과 문서, rr12 `_id` 문서 3쌍, manifest, 손편집 문서 5개, 보고서·로그 6개) + rr12 `_id` JSON 3개. `outputs/` 미열람.

## 판정
**통과.** R3·R4·R5·G2·M3·M4 모두 반영. 계약 문언(identity) 주 대비 **−0.00335 [−0.00379, −0.00288]**, 민감도(sigmoid) −0.00346 [−0.00382, −0.00307], 부호 동일 → BLOCKED 아님. 회귀 1건(**E1** 인코딩 손상)은 이 세션이 지정 브랜치에서 직접 복구했다(아래).

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| R3 identity 재실행 | ✓ | rr12 `--calibrator {select,identity}` (기본 `select` = 현 동작, T 불변). `_qS_id/_qT_id/_qTS_id` JSON `calibrator_policy=identity`, `calibrator_choice` 전 모델 identity. 표 1: q_S 0.24581 / q_T→S 0.24775 / q_TS 0.24605 / PT_flex_S 0.24916 / PT_linear_S 0.24949 |
| 보조 대비 (identity) | ✓ | q_S − q_T→S −0.00194 [−0.00236, −0.00153]; q_S − q_TS −0.00024 [−0.00039, −0.00009]; q_TS − q_T (T) +0.00071 [+0.00018, +0.00126]; S∩B40 −0.00210 [−0.00266, −0.00149]; H_S +0.00178 [+0.00096, +0.00259]; q_T→S − PT_flex_S −0.00141 [−0.00186, −0.00094]; q_S − PT_linear_S −0.00368 [−0.00412, −0.00321] |
| 표 2a/2b 병기 | ✓ | `primary` 1행(2a만); 2b는 sensitivity; `primary_sign_same_across_variants: true` |
| R4 lgbm 행 | ✓ | `prim["TEST"]["lgbm_state"]` 명시 선택; n 101 205 / 49 730, Brier 0.2475 |
| R5 | ✓ | rr12 L825 조건화(B40 CI가 0 포함 시만); `grep 'CI near 0'` S 문서 6개 모두 0 |
| G2 | ✓ | `T_frozen_rr0_digests_match: true` — 9개 sha16 일치, 불일치 시 exit 2 |
| G1 재확인 | ✓ | T008 `integrity`·`digests` 27개 불변; `digests_T009`에 `_id` 3개·`paired_contrasts_id` 추가 |
| M3 / M4 | ✓ | md·json `source_commit` 동일 실행; `_S.md`·`_TS.md` 배너 |
| 금지 문구 · 동결 T 문서 3개 · 원고 · 잠금 | ✓ | Forbidden 절 안에만; diff 0 |
| **E1 (회귀)** | ✗→복구 | 손편집한 5개 문서에서 비ASCII(Δ · − · — · ← · →)가 `??`/`?`로 **손실 치환**: `Q_NEWV_FIT85_PRIMARY_20260920_{S,TS}.md`(5줄씩), `REVIEW_RESPONSE_RR12_RESULTS_20260920_S_{qS,qT,qTS}.md`(9줄씩). 스크립트 생성본(`_id.md` 3개, 결과 md, rrx S md)은 정상. **복구:** `aa2249f` 판본을 되살린 뒤 의도한 편집만 UTF-8로 재적용(배너 prepend / "CI near 0" 1줄 삭제). `git diff aa2249f -- <file>`이 그 줄들만 남김을 확인. Windows 편집기의 cp949 저장으로 추정 — 이후 손편집은 UTF-8 고정 |
| 경미 | — | manifest `task` 필드가 `--with-t009` 실행에서 "T009"로 고정(T010에도). 표기만 |

## 계약 §8 승격 — 저자 결정 (2026-09-21)
**C. 저널 본문 2-코호트.** 이 세션이 잠금 문장 개정(추가만)과 원고 통합을 T011·T012로 실행한다. 계약 §6 금지 해석(규모 기울기·"더 예측 가능"·기전·합동/pick 대체)은 원고에서도 유지한다.

## 미검증 항목
- `outputs/` 파일 자체는 열지 않았다. 판정은 보고서·JSON·로그·스크립트 diff에 근거한다.
