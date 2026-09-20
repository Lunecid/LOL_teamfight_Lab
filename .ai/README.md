# .ai/ — Claude(설계) ↔ Cursor Agent(실행) 인계 폴더

```
Claude: 논의·설계 → 저자: 결정 승인 → 작업서 저장(Git) → Cursor Agent: 구현·검증 → .ai/reports/ 보고서 → Claude: 필요한 부분만 검토
```

| 경로 | 내용 |
|---|---|
| `CONTEXT.md` | 현재 목표, 기준 커밋, 문서 지도, 고정 계약, 검증된 수치 원장 |
| `RQ_DECISION_BRIEF.md` | 저자 결정 대기 안건(연구질문 구조) |
| `tasks/T00x.md` | 실행 작업서 (approval / base_commit / 범위 / 편집 목록 / 검증 / 중단 조건) |
| `reports/T00x.md` | Cursor가 남기는 결과 보고서 (양식: `reports/README.md`) |

## Cursor 실행용 프롬프트

```text
@.ai/tasks/T001.md 를 실행하라.

이번 역할은 승인된 작업의 구현과 검증이다. 전체 프로젝트의 계획을 다시 세우지 말라.
@AGENTS.md 와 @.ai/CONTEXT.md 를 먼저 읽고, 작업서가 지정한 파일과 필요한 의존 문서만 확인한 뒤,
허용 범위 안에서 편집하고 작업서의 검증 명령을 실제 실행하라.

기준 커밋 이후 대상 파일에 변경이 있다면 작업서가 여전히 유효한지 확인하라.
"찾을 문자열"이 0회 또는 2회 이상 매치되거나, 근거 문서의 값이 작업서와 다르면 그 항목을 BLOCKED로 보고하고 나머지를 진행하라.

결과는 .ai/reports/T001.md 에 양식대로 기록하라.
최종 응답에는 완료 여부, 변경 파일, 검증 결과, BLOCKED 항목만 적어라. push·merge는 하지 말라.
```

## 실행 순서와 의존
- T001 (기술 교정) → 보고서 검토 후 → T002 (Methods 전사), T003 (Discussion 분리) 병렬 가능
- T004 (동반 문서 배너)는 독립. 단 P4의 WORKING_NOTES 링크는 T003 이후가 자연스럽다.

## Claude 검토 시 돌려줄 것
작업서 + `.ai/reports/T00x.md` + 해당 파일 `git diff`. 원고 전체나 로그 전문은 보내지 않는다.
