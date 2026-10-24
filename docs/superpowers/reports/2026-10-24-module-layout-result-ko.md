# 모듈별 실제 코드 배치 결과

## 구현 범위

2026-10-24 추가 요청에 따라 39개 실제 구현 파일을 역할별 패키지에 옮겼다. 변경 전 파일에는 작은 호환 import 또는 실행 진입점만 둔다. 알고리즘 사본을 두 벌 유지하지 않는다. 모듈 지도는 루트 MODULES.md, 정확한 경로는 docs/module-layout-20261024.json을 따른다.

- EPT 24개: data, preprocessing, domain, valuation, models. 기존 ept/runtime/checks와 위치 의존 설정·고정 리소스·보호된 cohort·실행 명령은 유지한다.
- CoG 7개: analysis/fight_boundaries. 공간·시간 경계 추정, 지도, 사거리, 킬 쌍과 파이프라인을 한곳에 둔다.
- Master 8개: scripts/reporting. 이전 스크립트 경로에서 직접 실행과 import가 가능하다.
- 연구별 날짜 폴더와 common_training_r1의 독립 실행 형제 파일은 기존 연구·실행 경로를 보존한다. 모든 과거 스크립트를 한 번에 재설계한 것은 아니다.

## 호환과 의미

이전 import와 새 import는 같은 모듈 객체/함수 globals를 사용한다. EPT는 이전 pickle 이름을 읽을 수 있고 새 직렬화는 canonical 경로를 사용한다. 새 pickle을 옛 체크아웃이 읽는다고 보장하지 않는다. CoG는 과거 pickle 루트를 보존한다. Master는 argv[0]을 제외한 인자와 cwd를 유지하고, 실행 중 argv[0]은 canonical 파일을 가리키는 runpy 동작을 따른다.

EPT의 24개 이동 본문은 바이트와 전체 AST가 이동 전과 같다. CoG는 내부 import 경로와 호환 metadata 등록을 조정한다. Master는 같은 프로젝트 루트를 유지하도록 ROOT의 부모 깊이를 조정하고, validation provenance에 canonical 구현 8개를 추가한다. 실제 기존 보고서나 결과를 다시 생성하지 않는다.

재동기화 도구는 Master module-layout.json marker가 있는 파생 트리의 --apply를 원본 열람/복사 전에 거부한다. marker 기반 보호이며 marker가 없는 다른 트리까지 안전하다고 주장하지 않는다.

## 검증

수행한 검증과 범위는 docs/module-layout-validation-20261024.json에 기록한다. 구현자의 자체 시험과 별도 검토자의 검토를 구분한다.

- EPT: 서로 다른 234개 제한 시험 통과, 추가로 ancestry/work 없는 소스 복사본에서 이동 시험 52개 반복 통과.
- CoG: 조직화 시험 4개 및 작은 기존 합성 시험 7개 통과; 공개 소스 배치의 독립 실행도 확인.
- Master: 조직화 시험 8개가 작업트리와 ancestry 없는 공개 배치에서 각각 통과.
- 통합: 패키지 선언 4개/선택 AST 19개 검사 통과, 경계 시험 11개와 동기화 보호 시험 3개 통과.

초기 테스트 하네스 실패와 줄바꿈 비교 차이는 로컬 기록에 보존한다. 알고리즘을 바꾸거나 기존 검사를 무조건 통과시키는 방식으로 없애지 않았다. EPT의 source-scanner 시험은 실제 canonical 파일을 검사하도록 경로를 맞추고 기존 제한 조건을 유지한다.

실제 학습·추론·보고 생성·원본 동기화·새 실험은 실행하지 않았다. 기존 비용 측정과 겹치는 제한 CPU 검증을 배타적 성능 측정으로 해석하지 않는다. 기존 run hash와 연구 정의는 새 소스 배치의 자동 승인 근거가 아니다.

## 설명과 탐색

Gemini는 기존 OpenCodex 경로로 폴더명과 역할 요약만 받아 짧은 설명을 작성했다. 리드가 실제 구현 목록과 대조하고 잘못 해석할 수 있는 표현을 수정했다. Strata는 사용하지 않았다. 영어 메인 README와 한국어 README에서 모듈 지도로 이동할 수 있다.

## 독립 검토 및 게시

독립 검토 결과는 **ACCEPT(검토 범위 내 수용)**이다. CoG의 P2(namespace 하드코딩에 따른 `COG.analysis.*` import 실패 및 mixed-root alias 덮어쓰기)는 실제 namespace에서 이전 이름과 부모를 도출하는 수정으로 해결됐고, 검토자가 조직화 시험 4개(새 프로세스 8개 문맥)와 별도 작성한 qualified/mixed/pickle probe 8개를 재검증했다. 39개 canonical 본문의 기존 해시는 유지됐으며, 완료된 상태 문서를 포함한 최종 검토 파일 해시는 [독립 검토 기록](2026-10-24-module-layout-independent-review.md)에 고정한다(검토 기록 자체의 해시는 제외). 이 수용은 모듈 배치·호환과 명시한 제한 검증 범위에 한정하며 전체 프로젝트 무오류나 과학 결과 동등성을 뜻하지 않는다. 해시 목록은 검토한 작업트리의 RAW 바이트 기준이므로 Git core.autocrlf의 줄바꿈 정규화가 적용된 새 clone/원격 파일과 RAW SHA-256이 같다고 보장하지 않는다. 게시 담당자는 공개 트리의 committed Git blob 정체성을 확인하고 알고리즘 AST 검사를 유지한다; 검토자는 원격 전송 검증이나 커밋·push를 수행하지 않았다.
