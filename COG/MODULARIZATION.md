# COG modernization branch overlay

이 문서는 `codex/modularize-three-projects`의 코드 경계 개선을 설명한다.
launcher overlay는 원본 작업공간, sync manifest, 과학 코드, 해시 결합 기록을 변경하지 않는다.
이 overlay는 과거 구현의 동기화 사본과 별도로 유지한다.

## 설치 범위와 충돌 해결

배포 이름은 `lol-teamfight-cog`, 유일한 Python 패키지는 `lol_teamfight_cog`다.
wheel에는 `src/lol_teamfight_cog` launcher만 포함한다. `acquisition`, `analysis`,
`core`, `studies`, `reports` 등 과거 최상위 패키지는 설치하지 않는다.
두 launcher의 동시 설치는 가능하지만 과거 generic 패키지의 동시 설치는 지원하지 않는다.
이미 설치된 `lol-teamfight-lab`은 이 변경으로 자동 제거되지 않는다.

이것은 독립 실행 연구 wheel이 아니다. 명시적인 외부 소스 루트와 기존 연구 환경이 필요하다.
기본 설치는 launcher와 공용 `lol-project-runtime==0.1.0`만 요구한다. 이전 연구 의존성은 선택
`legacy` extra로 옮겼고 `all`이 이를 포함한다. 연구 환경 선택/설치는 통합 담당 범위다.
기존 의존성 범위는 환경 재현 lock이 아니다.

## 실행 계약

`python -m lol_teamfight_cog` 또는 `lol-teamfight-cog`만 실행하면 도움말로 끝난다.
`--help`, `list`, `check`는 과거 코드를 import/실행하지 않는다.
설치 절차는 `../tools/project_runtime/README.md`를 따른다. 세 로컬 패키지 경로를
명시하고 원격 인덱스는 사용하지 않는다. 설치 전 개발 호출에는 공용 runtime의
`src`와 이 프로젝트의 `src` 두 경로가 필요하다.

```powershell
lol-teamfight-cog --help
lol-teamfight-cog list --source-root 'X:\research\COG'
lol-teamfight-cog check --source-root 'X:\research\COG' --entrypoint main
```

`run --source-root <절대 경로> --entrypoint <허용 이름> -- <명시적 연구 인자>`
형태만 자식을 실행한다. 인자 생략, 상대 소스 루트, 누락 표식, 허용 목록 밖 진입점,
루트 밖으로 향하는 표식/스크립트 링크는 거부한다. cwd/환경에서 루트를 추측하지 않는다.
파일 표식 검사는 트리 종류와 가용성 검사이며 소스 진위/해시 검증은 아니다.

자식은 현재 `sys.executable`, 절대 스크립트 경로, 검증한 소스 루트의 cwd로 실행한다.
`--` 이후의 인자와 자식 종료 코드를 보존한다. shell을 쓰지 않는다.
상대 입력/출력 인자는 **호출한 터미널이 아니라 명시한 소스 루트** 기준이다.
출력 위치를 재작성하거나 생성하지 않고 기존 환경과 출력 설정을 자식에 그대로 전달한다.
명시적인 run은 실제 연구 작업을 시작할 수 있으므로 데이터/환경 승인은 별도 실행 작업에서 확인한다.
과거 코드의 help도 import 부작용이 있을 수 있다. 안전한 도움말은 launcher의 `--help`다.

## 구현 경계

- `cli.PROJECT`: 프로젝트별 표식과 고정 진입점 설정.
- `lol_project_runtime.launcher.validate_source_root`: 절대 루트/표식/경로 포함 관계 검증.
- `lol_project_runtime.launcher.source_manifest`: 가용한 사본과 참조 전용 진입점 상태/실패 이유.
- `lol_project_runtime.launcher.build_launch_plan`: 허용 이름, 명시적 인자, 실행 경로와 cwd를 검증/구성.
- `lol_project_runtime.launcher.execute_plan`: 검증된 명령의 subprocess 실행과 종료 코드 전달.
- `lol_project_runtime.cli.main`: 두 프로젝트가 공유하는 help/list/check/run 본문.

두 프로젝트는 서로 의존하지 않고 작은 공용 runtime 하나에만 의존한다.
각 패키지는 설정과 얇은 어댑터만 유지하며 검증/실행 구현과 합성 계약은 중복하지 않는다.
로컬 패키지 하나의 명시적 설치 부담을 받아들이는 대신 영구 복제본의 수정 불일치를 피한다.

허용 진입점은 `main -> main.py` 하나다. 기존 `runner.py::_run_isolated_per_model`은
`sys.argv[0]`을 재사용하므로 console script를 직접 그 함수에 연결하면 재실행 경로가
잘못될 수 있다. 절대 `main.py`로 시작하는 명령 구성과 고정 cwd가 그 경계 오류를 방지한다.
launcher 담당은 `runner.py`, `experiment_runner.py`, `app/experiment_runtime.py`,
과학 모듈과 `scripts/`를 수정하지 않는다. Gibbs의 별도 fusion 추출 범위는 아래에 기록한다.
launcher overlay만으로 과거 모놀리스를 분해했다고 주장하지 않는다.

## 합성 검증

저장소 루트에서 `python -B -m unittest discover -s COG/launcher_tests -p "test_*.py"`.
기존 과학 test suite/conftest는 수집하지 않는다. 새 launcher만 import하며 임시 합성
트리와 mocked subprocess를 사용한다. 실제 학습, 데이터 접근, 과거 연구 import는 없다.
환경이 symlink 생성을 금지하면 해당 보안 회귀 시험은 skip으로 보고한다.

2026-10-24 검증: 19개 중 16개 통과, 3개 skip (Windows symlink 권한 1개, Master 전용 참조 시험 2개).
실제 symlink 생성과 별도로 resolve 결과의 루트 이탈 거부 시험은 통과했다.
첫 sandbox 실행은 임시 디렉터리 쓰기/정리 권한 오류로 실패했고, 동일한 합성 테스트를
권한 검토 후 재실행해 위 결과를 얻었다. setuptools 파일 기반 패키지 검색에서도
고유 launcher 패키지만 발견했다. 현재 worktree의 list 사전 점검만 실행했으며
이 초기 단계에서는 설치/빌드를 실행하지 않았고, 이후 scratch editable 설치 검증은 아래에 기록한다.
연구 진입점, 학습, 기존 과학 테스트는 실행하지 않았다.
커밋/푸시는 통합 담당에게 남긴다.

공용 runtime 통합 뒤 재검증: runtime 자체 4개 통과와 두 어댑터의 위 계약 시험을
모두 통과했다 (전체 42개 중 38개 통과, 4개 skip).
각 어댑터가 같은 runtime 함수/구성 타입을 사용하는지, 고유 namespace 세 개만
발견되는지, 하위 명령 help가 자식을 실행하지 않는지도 확인했다.
공용 runtime README의 추가 smoke에서 scratch venv의 offline editable 설치,
설치된 launcher 진입점과 전체 합성 계약 재실행까지 통과했다.
일반 배포 wheel과 과거 연구 entrypoint import 성공은 검증하지 않았다.

## 별도 과학 코드 추출과 리비전 경계

2026-10-24 저자 지시에 따라 Gibbs가 사본 코드의 순수 함수 추출을 별도 담당한다.
범위는 `runner.py`의 fusion 표현식을 `app/fusion_spec.py`로 분리하는 것이다.
이 문서는 담당 범위를 기록하며 완료나 과학 동등성 검증을 보증하지 않는다.
launcher 담당은 이 파일들을 수정하지 않는다. 완료 상태는 통합 담당이 확인한다.
추출된 코드는 새 리비전이며 기존 scientific code hash와 같다고 취급하지 않는다.
옛 원본 작업공간과 결과/정의는 동결 보존하며, 변경된 코드로 과거 실행을 자동
resume하거나 release/acceptance/manifest 해시를 재작성하지 않는다.
실행 재개는 새 브랜치 리비전과 해당 검증 절차를 명시하는 별도 작업이다.
