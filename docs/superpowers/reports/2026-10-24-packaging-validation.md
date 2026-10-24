# Packaging validation — 2026-10-24

결과: **최초 세 배포 smoke 22/22 명령 통과**, 합성 시험 **38 passed / 4 skipped**.
EPT는 이후 별도 **4/4 명령 통과**로 확인했다. 최초 22개에 EPT를 포함했다고 해석하지 않는다.
수정 대상은 이 검증 문서와 package 문서뿐이며 구현 코드는 변경하지 않았다.

## 검증한 배포

| 배포 | 설치 확인 버전 | 고유 namespace | 검증 단계 |
|---|---|---|---|
| lol-project-runtime | 0.1.0 | lol_project_runtime | 최초 3배포 |
| lol-teamfight-cog | 0.1.0 | lol_teamfight_cog | 최초 3배포 |
| lol-teamfight-master | 0.1.0 | lol_teamfight_master | 최초 3배포 |
| lol-teamfight-ept | 0.1.0 | c_ept | 별도 EPT 추가 |

COG/Master는 `lol-project-runtime==0.1.0`을 요구한다. EPT root metadata의 기본 의존성은 없다.
각 배포의 설치 metadata 위치가 scratch venv의 site-packages 내부임을 확인했다.
최초 세 배포는 launcher pyproject/src 파일만 scratch로 복사해 editable 설치했다.
EPT는 승인된 격리 worktree root를 같은 scratch venv에 별도 editable 설치했다.
EPT의 namespace는 원래 worktree의 `c_ept/__init__.py`로, console entrypoint는
`c_ept.s4a_operator:main`으로 매핑되는 것을 metadata와 import 없는 package 위치 조회로 확인했다.

## 환경 격리와 범위

- 승인 worktree: `C:\Users\todtj\.codex\worktrees\modularize-three-projects\LOL_teamfight`.
- scratch: `C:\Users\todtj\.codex\visualizations\2026\10\22\01a16549-9684-7a63-9d92-63a80a0a2cb2\launcher-packaging-smoke`.
- venv: 위 scratch의 `run-01/venv`; `--system-site-packages --without-pip`로 생성.
  기존 환경의 pip/build 도구는 상속하지만 설치 대상 prefix는 별도 venv다.
  완전히 의존성을 차단한 hermetic 환경이라는 뜻은 아니다.
- 설치: `pip install --no-index --no-deps --no-build-isolation --disable-pip-version-check -e <명시적 로컬 경로>`.
  네트워크 인덱스/의존성 다운로드 없이 설치했다. Anaconda에는 설치하지 않았다.
- smoke 자식의 PYTHONPATH를 제거하고 unrelated scratch cwd에서 실행했다.
  module/metadata 검사는 `python -I -B`로 수행했다.
- 학습, science runner, 데이터/모델/동결 artifact 내용, 보호된 split/cohort 본문은 읽거나 실행하지 않았다.
  EPT는 `--help`만 실행했으며 무인자 기본 Check, Check/Start/Resume/Status/GPU는 실행하지 않았다.
- editable wheel 빌드/설치는 통과했다. 일반 배포 wheel, 과학 실행 가능성, 연구 결과 동등성의 검증은 아니다.

## 최초 세 배포: 22개 명령 결과

전체 명령 argv/cwd/log 경로는 scratch `run-01/evidence.json`에 보존한다.
아래 기대 exit와 실제 exit가 모두 일치했다. exit 2는 의도한 거부의 성공이다.

| # | evidence command name | 기대/실제 exit | 확인 |
|---|---|---|---|
| 1 | create-venv | 0 / 0 | scratch venv 생성 |
| 2 | verify-isolation | 0 / 0 | prefix=venv, base_prefix와 다름 |
| 3 | offline-editable-install | 0 / 0 | 로컬 세 배포 설치 |
| 4 | installed-metadata | 0 / 0 | 이름/버전/namespace/설치 위치/entrypoint |
| 5 | COG-console-help | 0 / 0 | 설치된 console --help |
| 6 | COG-console-empty | 0 / 0 | 기본 호출은 도움말 |
| 7 | COG-module-help | 0 / 0 | 설치된 module --help |
| 8 | COG-source-manifest | 0 / 0 | 명시한 worktree 파일 가용성 |
| 9 | COG-source-preflight | 0 / 0 | main 진입점 파일 check |
| 10 | COG-relative-root-rejected | 2 / 2 | 상대 루트 거부 |
| 11 | COG-unconfigured-run-rejected | 2 / 2 | 전달 인자 없는 run 거부 |
| 12 | master_ver-console-help | 0 / 0 | 설치된 console --help |
| 13 | master_ver-console-empty | 0 / 0 | 기본 호출은 도움말 |
| 14 | master_ver-module-help | 0 / 0 | 설치된 module --help |
| 15 | master_ver-source-manifest | 0 / 0 | 사본/참조 전용 파일 상태 구분 |
| 16 | master_ver-source-preflight | 0 / 0 | v-worker 진입점 파일 check |
| 17 | master_ver-relative-root-rejected | 2 / 2 | 상대 루트 거부 |
| 18 | master_ver-unconfigured-run-rejected | 2 / 2 | 전달 인자 없는 run 거부 |
| 19 | master-reference-missing | 2 / 2 | 누락 v-pipeline 참조 명확히 거부 |
| 20 | runtime-unit-tests | 0 / 0 | 4 passed |
| 21 | COG-unit-tests | 0 / 0 | 16 passed / 3 skipped |
| 22 | master_ver-unit-tests | 0 / 0 | 18 passed / 1 skipped |

합성 시험 총 42개 중 38개 통과, 4개 skip:
COG/Master에서 실제 symlink 생성 권한이 없어 각각 1개씩 skip;
COG에 참조 전용 진입점이 없어 Master 전용 참조 시험 2개 skip.
별도의 mocked resolve 루트 이탈 거부 시험은 양쪽 모두 통과했다.
연구 subprocess 실행은 mocked이며 합성 파일은 import/실행하면 실패하도록 구성했다.
`check`는 파일 가용성/경로 preflight일 뿐 과학 모듈 import 성공의 증거가 아니다.

## 별도 EPT: 4개 명령 결과

| evidence command name | 기대/실제 exit | 확인 |
|---|---|---|
| ept-offline-editable-install | 0 / 0 | 승인 worktree root를 scratch venv에 설치 |
| ept-metadata-source-mapping | 0 / 0 | 0.1.0 / c_ept / source mapping / metadata probe EPT import 없음 |
| ept-console-help | 0 / 0 | 설치된 console --help 및 import trace |
| ept-module-help | 0 / 0 | 설치된 c_ept.s4a_operator --help |

console help import trace의 EPT namespace는 `c_ept`, `c_ept.s4a_runtime`,
`c_ept.s4a_operator`뿐이었다. numpy/torch/scipy/sklearn/lightgbm 또는 다른
EPT 과학 모듈 import는 관측되지 않았다 (`forbidden_help_imports=[]`).
도움말은 argparse에서 종료되어 기본 Check나 native runtime 준비로 진행하지 않았다.
최초 metadata probe의 직접 operator import 제안은 자동 승인 검토에서 거절되어 실행하지 않았다.
승인된 수정 probe는 EPT를 import하지 않고 metadata/최상위 find_spec만 사용한다.

## 증거와 해석 한계

scratch의 다음 파일들이 상세 원천이다. 저장소 보고서는 이를 압축한 결과다.

- `run-01/evidence.json`: 최초 22개 명령, 세 배포 metadata, 소스 snapshot SHA-256.
- `run-01/*.log`, `run-01/REPORT.md`, `run-01/artifact-manifest.json`: 로그와 artifact 해시.
- `smoke.py`: 최초 smoke 재현 스크립트.
- `ept-run-01/evidence.json`, `ept-run-01/*.log`, `ept-run-01/artifact-manifest.json`: 별도 EPT 증거/해시.
- `ept_smoke.py`: EPT help-only 재현 스크립트.

최초 smoke 종료 시 snapshot pyproject/src 해시는 유지되었다. 이후 Gemini docstring
편집은 별도 리비전이며, executable AST가 같다는 lead 보고와 snapshot 원문 해시 동일성을
혼동하지 않는다. 이 보고서 작업은 해당 구현을 재편집하지 않았다.
EPT 추가 검사 동안 root pyproject 해시는 유지되었다.
과거 scientific hash/결과, frozen source workspace 또는 live S4 실행을 바꾸지 않았다.
커밋/푸시는 통합 담당 범위다.
