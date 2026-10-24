# LOL Teamfight Lab

EPT, IEEE CoG, Master/ToG 연구 소스와 프로젝트별 실행 어댑터를 모은 저장소입니다.
2026-10-24 공개판은 검토된 구현 커밋 `d9d7ac8dbd5c389c8bf70cbd4c0ed1ee8be99e4a`의
선별 소스를 기존 공개 이력 위에 추가한 스냅샷입니다. 이 구현 커밋의 비공개 과거 이력은 포함하지 않습니다.

| 폴더 | 역할 | 실행 패키지 |
|---|---|---|
| `c_ept/` | EPT 모델, 학습·평가·비용 측정, 체크포인트와 실행 제어 | 루트의 `lol-teamfight-ept` |
| `COG/` | CoG 계보 코드, fusion 파서와 실행 어댑터 | `lol-teamfight-cog` |
| `master_ver/` | Master/ToG 계보 소스, 순수 지표 함수와 실행 어댑터 | `lol-teamfight-master` |
| `tools/project_runtime/` | CoG/Master 공용 경로 검사와 자식 프로세스 실행 | `lol-project-runtime` |
| `research_pipeline/`, `common_training_r1/` | 데이터 처리·연구 파이프라인 소스 | 소스에서 사용 |
| `docs/` | 모듈화 안내, 검토·검증 기록과 공개판 범위 | 문서 |

## 시작하기

각 연구 환경을 따로 만들고, 현재 학습 중인 Python 환경에는 설치하지 마세요.
아래 명령은 **진입점 설치와 도움말 확인만** 합니다. 과학 라이브러리 설치나 학습은 수행하지 않습니다.
설치 환경에는 호환되는 `pip`, `setuptools`, `wheel`이 먼저 준비되어 있어야 합니다.

```powershell
# 저장소 루트에서, 이미 준비한 별도 Python 환경 사용
python -m pip install --no-index --no-deps --no-build-isolation -e .
lol-teamfight-ept --help

# CoG/Master launcher 전용 환경
python -m pip install --no-index --no-deps --no-build-isolation -e tools/project_runtime -e COG -e master_ver
lol-teamfight-cog --help
lol-teamfight-master --help
```

연구 실행에는 로컬 데이터, 설정, 기존 연구 소스와 의존 환경이 별도로 필요합니다.
EPT 환경 계약은 `environment/s4a-runtime.json`, 설정 예시는 `configs/s4a.example.json`에 있습니다.
`C_EPT_LEGACY_ROOT`에는 `c_ept/legacy_pins.json`에 고정된 23개 리소스를 별도로 준비해야 합니다.
공개 `COG/` 폴더만으로는 이 조건을 충족하지 못합니다.
기존 실험의 입력·코드 해시는 이 공개 스냅샷의 해시와 같다고 취급할 수 없습니다.
공개 복사본에서 기존 run을 자동 재개하거나 검증 해시를 덮어쓰지 마세요.

CoG/Master의 `list`·`check`는 명시적인 절대 `--source-root`를 받으며 연구 코드를 실행하지 않습니다.
`run`은 별도의 명시적 명령입니다. 도움말과 설치가 성공해도 데이터 준비 또는 연구 실행 성공을 뜻하지 않습니다.

- [설치·모듈 경계·검증 한계](docs/MODULARIZATION_KO.md)
- [공용 실행 어댑터](tools/project_runtime/README.md)
- [CoG 프로젝트 안내](COG/MODULARIZATION.md)
- [Master 프로젝트 안내](master_ver/MODULARIZATION.md)
- [구현 및 독립 검토 결과](docs/superpowers/reports/2026-10-24-modularization-result-ko.md)
- [공개 스냅샷의 범위와 재현 한계](docs/PUBLICATION_20261024.md)

## 가벼운 구조 검사

```powershell
$env:PYTHONUTF8 = "1"  # Windows subprocess 출력도 UTF-8로 통일
python -B tools/check_project_boundaries.py
python -B -m unittest discover -s tests -p test_project_boundaries.py
python -B -m unittest discover -s COG/launcher_tests -p "test_*.py"
python -B -m unittest discover -s master_ver/launcher_tests -p "test_*.py"
```

전체 데이터, 학습 가중치, 경기 목록, 원고 및 로컬 작업 이력은 이번 게시 대상이 아닙니다.
검토 문서에 나오는 원본 워크스페이스의 일부 문서·결과 경로는 공개판에 없습니다.
순수 함수 추출·실행 경계 개선은 전체 프로젝트의 완전한 재설계나 모든 오류의 부재를 의미하지 않습니다.
