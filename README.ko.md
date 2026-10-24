# LOL Teamfight Lab

[English](README.md) | 한국어

공개 매치·타임라인 데이터로 리그 오브 레전드의 한타와 교전을 살펴보는 연구 코드를 모았습니다. 교전의 결과, 교전 전후의 추정 승률 변화, 교전이 진행되는 과정을 연구합니다.

이 저장소에는 서로 연결된 세 프로젝트가 있습니다. 각 프로젝트는 고유한 연구 질문과 실험 설정을 가지며, 데이터 준비부터 특징 추출, 모델 학습, 예측 평가까지의 코드를 담고 있습니다. 관심 있는 연구 주제에 맞는 폴더부터 살펴보면 됩니다.

## 세 가지 프로젝트

| 프로젝트 | 연구 내용 |
|---|---|
| [EPT — `c_ept/`](c_ept/) | 교전 과정을 모델링합니다. 입력 특징 구성, 모델 학습·평가, 계산 비용 측정을 다루며 현재 개발 중인 프로젝트입니다. |
| [CoG — `COG/`](COG/) | 킬이 발생한 교전의 결과를 1분 단위 공개 타임라인 데이터로 예측합니다. IEEE Conference on Games 연구에서 사용한 코드를 담고 있습니다. |
| [Master / ToG — `master_ver/`](master_ver/) | 교전 전후의 추정 승률 변화와, 교전 전에 알 수 있는 정보로 무엇을 예측할 수 있는지 연구합니다. |

세 프로젝트의 주제는 연결되어 있지만 교전의 정의와 평가 조건은 다릅니다. 모델과 결과도 각 연구의 맥락에 맞춰 해석해야 합니다.

데이터 처리와 학습을 돕는 코드는 [`research_pipeline/`](research_pipeline/)과 [`common_training_r1/`](common_training_r1/)에 있습니다. [`tools/project_runtime/`](tools/project_runtime/)은 CoG와 Master에서 소스 경로를 확인하고 프로그램을 실행할 때 함께 사용하는 작은 라이브러리입니다.

[모듈별 코드 지도](MODULES.md)에서 실제 구현 폴더를 찾아볼 수 있습니다.

## 시작하기

이 공개판에는 소스코드가 들어 있으며, 매치 데이터와 학습된 가중치는 포함하지 않습니다. 실험을 실행하려면 로컬 입력과 해당 연구에 맞는 환경을 따로 준비해야 합니다. 그 전에 명령줄 도구를 설치해 도움말부터 살펴볼 수 있습니다.

저장소 루트에서 아래 명령을 실행합니다. 환경은 따로 준비하고, `pip`, `setuptools`, `wheel`을 미리 설치해 두세요. 아래 명령은 실행 도구만 설치하며 과학 계산 라이브러리를 내려받지 않습니다.

EPT용 환경에서:

```powershell
python -m pip install --no-index --no-deps --no-build-isolation -e .
lol-teamfight-ept --help
```

별도로 준비한 CoG·Master 실행 도구용 환경에서:

```powershell
python -m pip install --no-index --no-deps --no-build-isolation -e tools/project_runtime -e COG -e master_ver
lol-teamfight-cog --help
lol-teamfight-master --help
```

EPT 실험 환경은 [기록된 실행 환경](environment/s4a-runtime.json)과 [설정 예시](configs/s4a.example.json)를 기준으로 준비합니다. 기존 연구 리소스도 별도로 필요합니다. `C_EPT_LEGACY_ROOT`에 그 폴더의 절대 경로를 지정하고, [`legacy_pins.json`](c_ept/legacy_pins.json)에 기록된 파일과 해시를 확인하세요. 공개 `COG/` 폴더만으로는 필요한 리소스가 모두 갖춰지지 않습니다.

실험을 시작하지 않고 프로젝트 구조만 가볍게 확인할 수도 있습니다.

```powershell
python -B tools/check_project_boundaries.py
```

## 더 살펴보기

- [프로젝트 구성과 설치 안내](docs/MODULARIZATION_KO.md) — 패키지 배치와 모듈 간 역할을 설명합니다.
- [CoG 연구 소개](COG/README.md) — 영어로 작성된 기존 연구 개요입니다. 코드 구성은 한국어 [CoG 모듈 안내](COG/MODULARIZATION.md)에서 볼 수 있습니다.
- [Master 모듈 안내](master_ver/MODULARIZATION.md) — 소스 구성과 분리한 지표 함수를 설명합니다.
- [공통 실행 라이브러리](tools/project_runtime/README.md) — 설치 방법과 실행 도구의 동작을 안내합니다.
- [공개 범위와 재현 안내](docs/PUBLICATION_20261024.md) — 포함된 파일, 별도로 필요한 자료, 일부 비교 테스트에 원래 개발 이력이 필요한 이유를 정리했습니다.
