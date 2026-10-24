# 2026-10-24 공개 소스 스냅샷

## 후속 모듈 배치 변경

아래 설명과 `publication-manifest-20261024.json`의 파일 해시는 최초 공개 스냅샷 `972f90b`의 기록이다. 이후 README 개편과 모듈 배치 변경으로 현재 파일의 해시는 달라질 수 있다. 현재 배치는 [모듈별 코드 지도](../MODULES.md), [39개 이동 목록](module-layout-20261024.json), [검증 요약](module-layout-validation-20261024.json)을 따른다. 기존 보고서·실험 산출물을 다시 생성하지 않았으며, 이전 실행 해시를 새 코드의 승인으로 취급하지 않는다.

## 출처와 이력

구현 및 독립 검토 대상은 로컬 `codex/modularize-three-projects`의
`d9d7ac8dbd5c389c8bf70cbd4c0ed1ee8be99e4a`다. 공개판은 기존 공개 master
`42e88b1c78801738cd2a232e1032d4f71e2b6f18`를 부모로 하는 별도 스냅샷이다.
구현 커밋은 출처 표기일 뿐 공개 커밋의 부모가 아니다. 과거 비공개 커밋 이력을 병합하지 않았다.

`publication-manifest-20261024.json`은 옮긴 소스별 Git blob을 기록한다.
공개 안내용 README, 이 문서, manifest, 검증 기록과 `.gitignore` 보완은 게시 단계에서 작성했다.
과학 소스는 검토된 blob을 그대로 옮긴다. 기존 공개 파일은 유지한다.

사용자는 저장소가 PUBLIC이고 기존 로컬 훅이 Master·tools·검토 문서를 막는다는 설명을
받은 뒤 이 저장소로의 게시를 다시 명시했다. 공개 master의 fast-forward 스냅샷으로
게시하며, 로컬 훅 파일 자체는 바꾸지 않는다.

## 포함·제외 범위

- EPT, CoG, Master 소스와 테스트, 공용 launcher runtime 및 이번 모듈화 문서를 포함한다.
- 원시 데이터·가중치·체크포인트 payload·자격 증명·원고·에이전트 대화와 작업 로그를 추가하지 않는다.
- `master_ver/protocol_full.json`은 실제 경기 membership을 포함하므로 제외한다. 필요한 경우 사용자가 자신의 로컬 입력으로 제공한다.
- 과거 고정 경기 ID를 쓰는 `master_ver/scripts/full_corpus_preflight_20260915_schema.py`도 제외한다.
- `legacy_oldlineage`, 비공개 과거 검토 문서, vendor 도구는 새로 옮기지 않는다.
- 테스트의 합성·고정 예제 문자열은 보존한다. acquisition 테스트의 키 모양 문자열은 기존 공개 CoG blob과 같은 placeholder다.

패턴 검사는 선정된 텍스트 소스에 한정했다. 보호된 cohort/holdout/split 모듈 11개는
본문을 검사하지 않고 Git 객체로만 복사했다. 모든 파일을 보안 감사했다는 의미가 아니다.
기존 공개 이력은 재작성하지 않았다.

## 실행에 필요한 별도 입력

패키지는 source-backed launcher이며 완전한 연구 실행 환경을 배포하는 wheel이 아니다.
각 프로젝트에는 해당 데이터, 과학 라이브러리와 로컬 입력이 필요하다.

특히 EPT의 `C_EPT_LEGACY_ROOT`는 사용자가 별도로 준비한 **절대 경로**여야 한다.
`c_ept/legacy_pins.json`의 23개 파일과 SHA-256 계약에 맞는 리소스를 그 루트에 준비해야 한다.
기본값은 원래 로컬 워크스페이스 경로이며 다른 컴퓨터에서 유효하다고 가정할 수 없다.
이 공개 저장소의 `COG/`를 지정하는 것만으로는 부족하다. 해당 디렉터리에는 고정 리소스 중
21개가 없고, 존재하는 두 파일도 이름 일치만으로 해시 일치를 보장하지 않는다.
검증을 우회하거나 pins를 새 파일의 해시로 덮어쓰지 않는다.

Master launcher는 공개판의 `CODEMAP.md`, train.py, worker.py를 표식으로 검사할 수 있다.
참조 전용 `reports/v_production_20261004/pipeline.py`와 연구 결과·문서는 공개판에 없을 수 있다.
`list`·`check` 성공은 입력 데이터 또는 연구 실행의 성공을 뜻하지 않는다.

## 검증과 해석 한계

기존 검토 문서는 원래 소스 작업트리에서 수행한 검증 기록이다. 기록에 있는 로컬 경로는
공개 복사본의 존재 보증이 아니다. 이번 공개판의 별도 구조·단위 검사 결과는
`publication-validation-20261024.json`에 기록한다.

다음 테스트의 baseline-equivalence fixture는 비공개 기준 커밋
`282dcc36f2e175b33782fc0b4bcb31581e663612`를 `git show`로 읽는다.

- `COG/tests/test_fusion_spec.py`
- `master_ver/studies/v_production_20261004/tests/test_metrics.py`

따라서 깨끗한 공개 clone에서 이 파일들의 **전체 테스트**는 그대로 재현할 수 없다.
해당 기준 커밋을 공개 이력에 넣지 않았으며, 게시를 위해 동등성 시험을 삭제하거나
무조건 통과하도록 바꾸지도 않았다. 원래 비교 결과는 로컬 검증 기록으로만 해석한다.
공개판에서 실행한 launcher·패키지 경계·합성 자원/체크포인트 검사는 이 기준 커밋을 사용하지 않는다.

원래 검토된 변경 범위는 순수 함수 추출, 재사용 가능한 실행 경계와 제한된 EPT 최적화다.
모든 모놀리스 제거, 전체 테스트 통과, 모든 설계·코드 오류의 부재 또는 연구 결과 동등성을 주장하지 않는다.
진행 중인 Task 8 실행의 코드·입력·프로세스·원본 환경은 이번 게시 작업에서 변경하지 않는다.
