# EPT 폴더 정리 결과

2026-10-24, 기준 소스 `611a4a4ef092890fca52e00250608e70bc407e81` 이후 추가 정리다. 루트 Python 파일을 **39개에서 5개**로 줄였다. 파일을 IDE에서 숨기는 설정 대신 실제 구현과 호환 진입 파일을 이동했다.

```text
c_ept/
  data/            읽기·형변환·저장소·학습 행
  preprocessing/   특징·지형·토큰·특성 전처리
  domain/          교전·정답·상태·대칭 정의
  models/          B0/B1·머리 인터페이스·NLL
  ept/             Transformer·학습·체크포인트 구현
  valuation/       가격·보상·가치 합성
  workflows/       S1/S2/S3/S4a 실행·수동 operator·초기화 정책
  reporting/       S1/S2/S3 보고서 생성
  checks/          S3 판정·진단
  runtime/         공용 자원 계측
  _compat/         이전 import/module 명령의 호환 파일 34개
  tests/           소프트웨어 검사
  __init__.py
  cohort.py        보호된 분할 정의, 이동·본문 열람 없음
  config.py
  legacy.py
  runlog.py
```

기존 호환 파일 24개를 `_compat/`로 옮겼고, 실행·보고·판정·초기화 구현 10개를 역할별 패키지로 옮겼다. 새 구현 사본을 두 벌 유지하지 않는다. 정확한 옛 경로·실제 구현·호환 파일 위치는 [경로 등록부](../../module-layout-20261024.json)의 `ept_visible_root_cleanup`에 있다.

## 호환과 검증 범위

`c_ept.__path__`에 `_compat` 검색 경로를 추가해 정상 Python import 규칙을 사용한다. 전역 sys.path나 meta_path hook을 바꾸지 않는다. 기존 import와 canonical import는 같은 모듈 객체와 globals를 사용한다. `python -m c_ept.run_s4a`, 수동 PowerShell wrapper, console 진입 이름을 보존했다. 삭제된 루트 `.py` 파일을 경로로 직접 실행하는 외부 호출은 module 명령으로 바꿔야 한다.

구현자는 357개 주요 회귀 항목의 누적 통과 근거와 추가 집중 검사를 남겼다. 최초 실행의 실패는 contextmanager 검사 unwrap, Windows 출력 인코딩, 기존 공용 runtime import를 빠뜨린 정적 검사 목록으로 나눠 기록했다. 첫 로그를 삭제하거나 최종 단일 실행 통과라고 바꾸지 않았다. 이후 독립 검토자가 동결된 최종 네 파일 묶음을 새 프로세스에서 한 번 실행해 **357 passed, 실패·오류·skip 0개**를 확인했다(62.09초). 해당 결과는 별도 검토자의 `ept-root-review-final-357.xml`과 로그에 기록됐다. 재검증·최종 판정과 한계는 [독립 검토](2026-10-24-ept-root-cleanup-review.md)를 따른다.

소스 대조에서는 새로 이동한 구현 10개 중 7개가 전체 AST 동등, 나머지 3개는 저장소 기준 경로와 고정 추출 provenance 목록의 명시적 조정 이후 동등했다. 이전에 묶은 구현 24개와 그 호환 파일은 줄바꿈 정규화 후 내용이 유지됐다. 실제 새 저장소 경로에 대한 추출 코드 검증 목록도 포함하므로 호환 파일만 검사하고 구현 변경을 놓치지 않는다. S4a의 저장소 청결·입력 해시·E6·실행 identity·resume 거부 조건은 완화하지 않았다.

격리 wheel 검사는 보호된 cohort 본문을 읽거나 복사하지 않고 실패 sentinel로 대체한 소스에서 수행했다. 호환 파일의 패키징 포함과 설치 위치의 import/help 명령을 검증한다. 이 wheel은 실제 연구 실행용 배포물이 아니다. 기존 외부 입력·고정 legacy 자산·환경 파일이 필요한 source-backed 실행 계약은 그대로다. 보호된 소스를 읽는 일부 전체 reader gate 및 실제 연구 데이터 통합 검사는 실행하지 않았다.

## 역할과 다음 실행

Sol이 구현하고 별도 검토자가 소스 차이·독립 probe·회귀를 검토한다. Gemini는 OpenCodex의 기존 연결로 파일명·역할만 받아 짧은 폴더 설명을 제안했다. 응답이 중간에 끊겨 완성된 영어 세 문장을 검토·수정해 사용했고, 전체 주석을 Gemini가 다시 작성했다고 주장하지 않는다. Strata는 계속 껐다.

수정한 개발 폴더는 `codex/modularize-three-projects` 브랜치다. 사진처럼 수정 표시가 남아 있는 이전 비용 실행 체크아웃에는 이 배치를 덮어쓰지 않는다. IDE에서 정리된 개발 폴더를 열어야 새 구조가 보인다. 실제 학습·추론·보고 재생성이나 새 비용 측정은 실행하지 않았다. 기존 실패 run과 S CPU 결과는 보존한다. [다음 실행 준비](../../S4A_NEXT_STEPS_KO.md)에 따라 전용 실행 커밋·장치·측정 범위를 정한 뒤 진행한다. 이번 정리만으로 G-S4a 완료나 S4b/S6/S7 실행 승인이 생기지 않는다.

저자의 추가 요청으로 현재 작업 원칙 전체를 [프로젝트 작업 지침](../../WORKING_GUIDELINES_KO.md)에 통합했고, 루트 AGENTS.md의 진입 규칙과 영어·한국어 README에서 연결했다. 기존 CoG 지침 본문은 줄바꿈 정규화 후 그대로 보존됐으며 전역 메모리·에이전트 설정은 변경하지 않았다.
