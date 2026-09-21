# 작업 목록 (TASKS)

학위논문 완성까지의 작업. **여기(Claude)** = 계획·설계·원고, **Cursor** = 코드·실행(작업서 `.ai/tasks/T0xx.md`, 보고서 `.ai/reports/`), **저자** = 결정·주석·행정.
집필 규칙: 새 실험·재집계·새 수치 금지 (`AGENTS.md`); 보완 실험은 `docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md`의 계약과 작업서로만 진행하며 동결 헤드라인을 대체하지 않는다. 실행 상태의 원장은 [`STATUS.md`](STATUS.md).

## A. 저자 결정 (원고 차단 항목; `\pendingauthor{}` 추적은 `STATUS.md` §4)

| ID | 결정 | 영향 위치 | 상태 |
|---|---|---|---|
| A1 | pick 제외 사유 문장 확정 | 4.5 | pending |
| A2 | 외부 T/S 병기 문장 유지 여부 (E3 구간이 추가된 상태에서 결정) | 7.5, 영문 초록 | pending |
| A3 | 릴리스 태그·DOI·파생 자료 라이선스 | 부록 A | pending |
| A4 | 학위명 (공학/이학/데이터사이언스학 석사), 학위수여년월, 최종심사일, 심사위원 | `config/meta.tex` | pending |
| A5 | 초록의 핵심 수치 유지 여부 (현재 유지) | 국·영문 초록 | pending |

## B. 원고 (여기)

| ID | 작업 | 우선 | 의존 | 상태 |
|---|---|---|---|---|
| B0 | 서술 재구성: 일반 논문 서술(Widom), CoG 2026 3인칭·2장 배치, 결과별 해석 조건, 캡션 이름화, 보완 실험 E1–E5·G0 통합 | P0 | — | **done** (2026-09-21) |
| B1 | 개념 그림 추가: 코호트·역할 표본 흐름, ΔBrier by p_pre bin 점도표 (기존 원장의 bin 표 값만 사용), 신뢰도 다이어그램(T014) | P1 | C8 | pending |
| B2 | 사례 추적 절 본문화 (5.8의 hash 표본; event-prefix 덤프가 실행되면 질의→사건 접두→V̂→SVI→q 서사) | P2 | C7 | pending |
| B3 | 문체 재검토: `tools/check_style.py --verbose`의 N/V/S/ST 위치를 사람이 읽고 불확실성 보존 확인 | P1 | — | 1차 완료; 최종 검토 전 재실행 |
| B4 | 참고문헌: 저장소 풀의 `[verify]` 표시 항목 재확인, 학위논문 추가 4건 DOI 확인 | P1 | — | pending |
| B5 | 부산대 제출 체크리스트 (`CHECKLIST.md`) | P0 | A3, A4 | pending |

## C. 실행 (Cursor; 작업서 필요; 보완설계 계약 아래)

| ID | 작업 | 계약 | 상태 (STATUS.md) | 반영 위치 |
|---|---|---|---|---|
| C0 | G0 무결성: 잔여 항목 (G0.5 fold 홀드아웃 해시, G0.F 미래정보 불변성, G0.1 worktree 모듈 해시, G0.6 점수 동일성) | 보완설계 §3 | PARTIAL / INCOMPLETE | 부록 A, 8.4 |
| C1 | E4 정의 OAT 점수 민감도 (frozen-score) 및 Refit mode | 보완설계 §7.4–7.5 | NOT_RUN | 4.8 |
| C2 | E1 동일 352열 logistic 대 LGBM 정합 비교 | 보완설계 §4.5 | NOT_RUN | 6.8, 8.1 |
| C3 | E2 quiet 대조의 프레임 나이 강화 매칭 (보조) | 보완설계 §5.5 | NOT_RUN | 5.6 |
| C4 | E4 동료 평가기 OOF 라벨로 q/PT 재학습 (두 번째 실험) | 보완설계 §7.2 | NOT_RUN | 5.9 |
| C5 | F2 참여 인원 셀 재점수 (재적합 없음) | THESIS_V2 F2 | DEFERRED | 7장 보조 표 |
| C6 | F1 외부 확률 어댑터 | THESIS_V2 F1 | DEFERRED | 7.5 보조 |
| C7 | E5 event-prefix 사례 추적 덤프 | 보완설계 §8.1 | INCOMPLETE | 5.8 / B2 |
| C8 | 그림 데이터 내보내기: CORP 다이어그램 q/PT_flex, T/S (T014와 공유) | T014 | pending | 7.4 그림 |
| C9 | E6 미노출 표본 수집 (API 키 갱신 후) → 점수 전용 평가 | 보완설계 §9; T024 | BLOCKED | 7.5, 8.4 |

완료(반영됨): E1 정보군 LGBM(T018), E2 S_hold·층화(T019/T022), E3 외부 부트스트랩(T017), E4 지평·동료 평가기·e_fixed·OAT 사례 구성(T020/T023), E5 물질 축·nextobj·market_event·R0/R1·표본 CSV(T021/T023), E6 규칙 잠금(T024), G0 Phase A+B(T015/T016).

## D. 검증·행정

| ID | 작업 | 우선 | 상태 |
|---|---|---|---|
| D1 | 차단 검사 `python thesis/tools/check_thesis.py` (생성 수치·상태 표·참조·라벨·헤드라인·M-RQ 해시) | P0 | 통과 (2026-09-21) |
| D2 | 편집 보고 `python thesis/tools/check_style.py --baseline thesis/tools/style_baseline.json --verbose` | P1 | 실행; 위치 검토 완료 |
| D3 | 역개요·자기 심사 (`REVERSE_OUTLINE.md`, `SELF_REVIEW.md`; paper-self-review 7단계) | P1 | 작성 |
| D4 | G0 잔여 항목이 주 결과의 타당성에 직접 영향을 주는지 최종 제출 전 판단 (C0 결과에 따라) | P0 | pending |
| D5 | 모의 심사 → `CLAIMS.md` 예상 질문 갱신 | P2 | pending |
| D6 | 데이터사이언스전문대학원 심사 일정·서류 (2심 이상, 심사요구서, 연구윤리 서약서) | P0 | pending |

## 순서 제안

```
지금       : A1–A5 결정 요청; B4 문헌 확인; C8 그림 데이터
1차 (원고) : B1 그림, B2 사례 추적(C7 후), B3 문체 최종 검토
2차 (실행) : C0 → C2 → C1 → C7 → C9(키 갱신 시) 작업서 → Cursor → 절 갱신 → gen_supp/STATUS 갱신
마무리     : A4 반영, D4 판단, B5 체크리스트, D6
```
