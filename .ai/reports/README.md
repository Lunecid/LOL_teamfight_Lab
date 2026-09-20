# 보고서 양식 (.ai/reports/T00x.md)

```markdown
# T00x 보고서 — <작업 제목>

status: DONE | PARTIAL | BLOCKED
base_commit_checked: <실행 시점 HEAD sha>
executed_at: <YYYY-MM-DD HH:MM KST>

## 변경 파일
- path (+N / −M lines)

## 편집 항목별 결과
| 항목 | 결과 (applied / skipped / BLOCKED) | 비고 (매치 횟수, 근거 값 확인 결과) |

## 실행한 검증 명령과 출력 (핵심만; 전체 로그는 .ai/reports/logs/T00x_*.txt)
```bash
<명령>
<출력 요약>
```

## 계획과 달라진 점
- (없으면 "없음")

## 미실행 검증
- (없으면 "없음")

## BLOCKED 사유와 필요한 결정
- (없으면 "없음")

## 다음 작업에 넘길 메모
- (예: 줄 번호 변화, 새로 발견한 불일치)
```

규칙: 실패·미실행은 그대로 적는다. 수치 출처를 확인하지 못한 항목은 "확인 못 함"이라고 쓴다. 보고서는 Claude가 diff와 함께 읽는 유일한 산출물이다.
