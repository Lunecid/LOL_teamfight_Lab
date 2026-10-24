# Shared project runtime

현대화 브랜치의 공용 실행 경계다. 배포 이름은 `lol-project-runtime==0.1.0`,
Python namespace는 `lol_project_runtime`이다. 연구 구현, 데이터, 모델은 포함하지 않는다.

COG와 Master가 검증/명령 실행/CLI 본문을 각각 복제하면 오류 수정이 갈라질 수 있다.
이 패키지에 구현을 한 번만 두고 각 프로젝트에는 `Project` 설정과 얇은 어댑터만 남긴다.
추가 로컬 패키지 하나를 설치하는 비용 대신 단일 구현과 단일 합성 계약을 얻는다.
상대 file URL 의존성이나 저장소 위치를 추측하는 fallback은 사용하지 않는다.
두 어댑터는 `lol-project-runtime==0.1.0`을 명시적으로 요구한다.

## 명시적 로컬 설치

통합 담당이 고른 별도 Python 환경에 setuptools>=64와 wheel이 이미 있어야 한다.
저장소 루트에서 다음과 같이 세 로컬 경로를 명시한다. 아래 Python 경로는 실제
선택한 환경의 절대 경로로 바꾼다. 기존 연구 환경에는 설치하지 않았다.
아래 검증 기록의 설치는 scratch venv에 복사한 launcher 소스만 사용했다.

```powershell
& 'X:\chosen-environment\python.exe' -m pip install --no-index --no-build-isolation --no-deps -e ./tools/project_runtime -e ./COG -e ./master_ver
& 'X:\chosen-environment\python.exe' -m pip check
```

`--no-index`는 같은 이름의 원격 배포를 잘못 받지 않게 한다. `--no-deps`를 쓰므로
위 세 경로를 모두 포함해야 한다. 연구 의존성은 기존 승인된 환경에서 별도 관리한다.
직접 설치하지 않고 소스를 호출하는 개발 시에는 공용 `src`와 해당 프로젝트 `src`
두 경로를 명시해야 한다. launcher가 이 경로를 자동 탐색하거나 PYTHONPATH를 바꾸지는 않는다.
wheel도 launcher/runtime만 담으며 외부 연구 소스 루트가 필요하다.

## 재사용 API

- `launcher.Project / SourceEntry`: 프로젝트 표식과 고정 진입점 구성.
- `validate_source_root`: 절대 루트/표식/경로 포함 관계 검사.
- `source_manifest`: 각 진입점의 파일 가용성과 참조 전용 여부 보고.
- `build_launch_plan`: 허용 목록과 전달 인자로 절대 실행 경로/cwd 구성.
- `execute_plan`: shell 없이 실행하고 자식 종료 코드 전달.
- `cli.main(project, argv)`: 공통 help/list/check/run 동작.

manifest의 available은 파일 가용성만 뜻한다. dependency import, 과학 계약,
데이터, 실행 가능성까지 검증했다는 뜻은 아니다. source root는 신뢰하는 트리를
명시해야 한다. 표식 검사는 해시 검증이 아니다.

## 합성 검증

저장소 루트에서 다음 테스트를 각각 실행한다. 새 launcher/runtime만 import한다.

```powershell
python -B -m unittest discover -s tools/project_runtime/tests -p 'test_*.py'
python -B -m unittest discover -s COG/launcher_tests -p 'test_*.py'
python -B -m unittest discover -s master_ver/launcher_tests -p 'test_*.py'
```

`tests/launcher_contract.py`는 프로젝트 설정별로 재사용하는 단일 합성 계약이다.
실제 연구 subprocess는 실행하지 않으며 별도 환경/의존성 검증을 대체하지 않는다.

2026-10-24 통합 검증: runtime 4개 통과, COG 16개 통과/3개 skip,
Master 18개 통과/1개 skip (총 38개 통과/4개 skip).
Windows symlink 생성 권한 skip과 별개로 resolve 결과의 루트 이탈 거부는 검증했다.
원격 설치와 실제 연구 import/help 실행은 하지 않았다.

추가 packaging smoke: scratch의 `--system-site-packages --without-pip` venv에
세 패키지의 pyproject/src 스냅샷만 복사하여 offline editable 설치를 성공했다.
`--no-index --no-deps --no-build-isolation`로 설치했으며 기존 Anaconda에는 설치하지 않았다.
설치된 console script/module 도움말, 무인자 호출, 실제 worktree의 파일 preflight,
상대 루트/무인자 run 거부, 누락 pipeline 처리와 합성 테스트까지 22개 명령을 통과했다.
metadata/namespace/venv 설치 위치를 확인했고, 검사 종료 시 원본 launcher 스냅샷 해시도 같았다.
pip의 editable wheel 빌드는 통과했지만 일반 배포 wheel과 실제 연구 import는 검증하지 않았다.
증거는 별도 scratch artifact `launcher-packaging-smoke/run-01/evidence.json` 및
22개 로그에 보관한다. 스크립트/보고서/로그 해시는 같은 artifact의 manifest를 따른다.

위 22개는 runtime/COG/Master **세 배포**만의 기록이다. 별도 EPT 추가 검사는
같은 scratch venv에 승인 worktree root를 offline editable 설치하고 metadata/source mapping,
console/module `--help`만 확인했다 (4/4 명령 통과). EPT의 기본 Check/GPU는 실행하지 않았다.
console import trace에는 과학/native 모듈이 나타나지 않았다.
압축 증거와 네 배포의 정확한 버전/범위는
`../../docs/superpowers/reports/2026-10-24-packaging-validation.md`에 기록한다.
