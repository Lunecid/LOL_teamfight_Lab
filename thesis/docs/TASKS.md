# 작업 목록 (TASKS)

학위논문 완성까지의 작업. **여기(Claude)** = 계획·설계·원고, **Cursor** = 코드·실행(작업서 `.ai/tasks/T0xx.md`, 보고서 `.ai/reports/`), **저자** = 결정·주석·행정.
집필 규칙: 새 실험·재집계·새 수치 금지 (`AGENTS.md`); 이연 항목(F1–F6, M-F0–M-F3)은 실행하더라도 별도 예비 선언 계약과 작업서로만 진행하며 동결 산출물을 덮어쓰지 않는다.

## A. 저자 결정 (원고 차단 항목)

| ID | 결정 | 영향 위치 | 상태 |
|---|---|---|---|
| A1 | pick 제외 사유 문장 (저널 `\pending{pick-exclusion-wording}`과 동일 문안) | 4.5 | pending |
| A2 | EXT T/S 병기 문장 유지 여부 → 초록 Variant A/B | 7.4, 국·영문 초록 | pending |
| A3 | 학위명 (공학/이학/데이터사이언스학 석사), 학위수여년월, 최종심사일, 심사위원 | `config/meta.tex` | pending |
| A4 | 릴리스 태그·DOI·파생 자료 라이선스 | 부록 A | pending |
| A5 | 이연 항목 중 학위논문에 실제로 실행할 것 (F5 정의 민감도, F6 정보군 모듈, 사례 추적 우선 후보) | 8.6, 부록 E | pending |

## B. 원고 (여기)

| ID | 작업 | 우선 | 의존 |
|---|---|---|---|
| B1 | 개념 그림 추가: 코호트·역할 표본 흐름 (저널 fig3), ΔBrier by p_pre bin 점도표 (저널 fig2; RR12·S 20-bin 표 값만 사용) | P1 | — |
| B2 | 사례 추적 절 초안 (연구계획 §6.1; 새 수치 없이 기존 예측표에서 사례 선택 규칙 명시) | P2 | A5 |
| B3 | 문체·용어 통일 검토 (`STYLE.md`), 초록 2쪽 이내 확인 | P2 | A1–A2 |
| B4 | 참고문헌: 저장소 풀의 `[verify]` 표시 항목 재확인, 학위논문 추가 4건 DOI 확인 | P1 | — |
| B5 | 부산대 제출 체크리스트 (`CHECKLIST.md`) | P0 | A3 |

## C. 실행 (Cursor; 작업서 필요; 모두 이연 계약 아래)

통합 설계: [`docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md`](../../docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md) (G0·E1–E6). 헤드라인 교체 금지.

| ID | 작업 | 계약 | 산출물 → 반영 |
|---|---|---|---|
| C0 | G0 무결성 전수 (해시 재계산, 미래정보 불변성, reload parity, 헤드라인 재집계) | 보완설계 §3; 일부는 `tests/test_q_provenance_fixes.py` | `integrity.json` |
| C1 | E4/F5 정의·endpoint·peer 민감도 (Frozen-score 우선; Refit은 선택) | 보완설계 §7; `THESIS_V2` F5 | 4.8 표 |
| C4 | E3 외부 paired bootstrap (KR/NA1 × T/S; 재학습 없음) | 보완설계 §6; T017 **DONE** | `docs/SUPPLEMENTARY_E3_*` |
| C2 | E1 / M-F0–M-F3 정보군 (LGBM 4 arm + Q0; identity) | 보완설계 §4; T018 **DONE** | `docs/SUPPLEMENTARY_E1_*` |
| C3 | E2 프레임 갱신 층화·S_hold (V 재선정 없음) | 보완설계 §5; T019 **PARTIAL** (§5.4 DONE / S_hold INCOMPLETE) | `docs/SUPPLEMENTARY_E2_*` |
| C5 | F2 참여 인원 셀 재점수 (재적합 없음) | F2 | 7장 보조 표 |
| C6 | F1 외부 확률 어댑터 | F1 | 7.4 보조 |
| C7 | E5 최소 사례 추적·동일 사례 지표 비교 (강한 주장 시) | 보완설계 §8 | 5–6장 / B2 |
| C8 | 그림 데이터 내보내기: CORP 다이어그램 q/PT_flex, T/S (저널 T014와 공유) | T014 | 7.3 그림 |

## D. 검증·행정

| ID | 작업 | 우선 |
|---|---|---|
| D1 | 수치 대조: `numbers.tex` ↔ `NUMBERS.md` 출처 필드 전수 확인 (스크립트) | P0 |
| D2 | 금지 문구 grep (F1–F8, "더 예측 가능", "규모 기울기", "재보정으로 복구") | P0 |
| D3 | 모의 심사 → `CLAIMS.md` 예상 질문 갱신 | P1 |
| D4 | 데이터사이언스전문대학원 심사 일정·서류 (2심 이상, 심사요구서, 연구윤리 서약서) | P0 |

## 순서 제안

```
지금       : A1–A5 결정 요청, B1 그림, B4 문헌 확인, D1–D2 스크립트
           : 보완설계 등록됨 → C0(G0 전수) 작업서부터 (헤드라인 교체 금지)
1차 (원고) : B2 사례 추적(결정 후), B3 문체, D3 모의 심사
2차 (실행) : A5에서 고른 항목만 (석사 최소: C0→C2→C3→C4→C1→C7) 작업서 → Cursor → 절 갱신
마무리     : A3 반영, B5 체크리스트, D4
```
