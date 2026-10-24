# EPT·CoG·Master 모듈화 결과

기준은 `282dcc36f2e175b33782fc0b4bcb31581e663612`, 변경 브랜치는
`codex/modularize-three-projects`다. 기존 Task 8 실행 checkout과 원본 연구 작업공간은
수정하지 않고 별도 worktree에서 구현했다. 구현은 Sol 작업자들, 통합은 Codex,
독립 정적 검토는 별도 Astra 작업자가 담당했다. Gemini 주석 초안은 OpenCodex 경유다.

## 변경 결과

| 대상 | 분리한 부품과 동작 |
|---|---|
| EPT | `c_ept.runtime.resources`에 자원 계측, `ept.checkpoints`에 검증·저장, `ept.training_step`에 배치 계산을 분리했다. S3의 기존 이름과 테스트 주입 경계, v5 체크포인트 계약은 유지했다. |
| EPT 평가 | `m2_rate`로 평가 때 사용하지 않는 표지·구역 계산을 생략한다. 학습에서는 전체 M2를 호출하여 난수 소비 순서를 유지한다. STOP 평가 래퍼에도 명시적으로 `eval()`을 적용했다. |
| CoG | fusion 명세 파싱·별칭·조합 순서를 `app.fusion_spec`으로 분리하고 기존 runner 이름을 재노출한다. 모델별 subprocess 실패를 요약만 하고 정상 종료하던 경로는 비정상 종료를 전달하도록 수정했다. |
| Master | sigmoid와 행 단위 logloss를 독립 `metrics` 모듈로 분리했다. float64 계산·검증·클리핑 순서를 유지하고 보정기 클래스의 원래 pickle 정의 경로는 옮기지 않았다. |
| 설치·실행 | EPT·CoG·Master의 배포 이름과 namespace를 구분했다. CoG/Master의 중복 실행 도구는 `lol-project-runtime` 하나로 모았다. 명시적인 소스 루트와 `run`을 요구한다. |
| 구조 검사 | pyproject 4개와 명시한 소스 19개의 의존 경계를 검사한다. 공유 부품에서 실행기로 역의존하는 변경과 generic package 재포함 등을 거부한다. |

전후 구조도와 환경별 설치 명령은 [모듈화 안내](../../MODULARIZATION_KO.md)에 있다.
CoG/Master의 기존 연구 코드는 소스 루트에서 실행하며 launcher wheel 안에 모두 넣지 않는다.
EPT는 설정·환경 명세·Git checkout을 사용하는 editable 설치가 지원 경로다.
패키지 설치 성공만으로 연구 의존성과 데이터 준비가 완료되지는 않는다.

## 검증 근거

- [EPT 구현·회귀](2026-10-24-ept-modularization-plan.md),
  [명령과 결과 JSON](2026-10-24-ept-test-evidence.json): 자원/저장/배치의 작은 시험,
  평가 모드, 중단 후 재개 비트 일치, 실제 AdamW 상태/계산의 합성 회귀를 실행했다.
  최초 실패와 수정 후 결과를 구별해서 남겼다.
- [순수 부품 검증](2026-10-24-pure-components-validation.md): CoG 파싱,
  Master 계산과 오류, 기존 소스와의 실행 AST 비교, M2 출력·기울기·학습 RNG·파라미터 등록을 점검했다.
  CoG 종료 코드 수정은 동일성 주장과 별도로 시험한다.
- [설치 검증](2026-10-24-packaging-validation.md): 공통/CoG/Master 22개 명령과
  EPT 4개 명령의 예상 종료 코드가 맞았다. 합성 실행 계약 시험은 38개 통과, 4개 skip이다.
  skip은 Windows symlink 권한 및 CoG에 해당하지 않는 참조 경로 사례이며,
  경로 이탈 거부는 별도 mock 시험으로 확인했다. 설치는 scratch 환경에서 했고 Anaconda를 변경하지 않았다.
- 경계 검사와 11개 단위 시험은 통과했다. 위 결과들의 중복 시험을 더해 전체 고유 시험 수로 표시하지 않는다.
- [독립 검토](2026-10-24-independent-modularization-review.md)는 구현자 자체 시험과 별개다.
  검토자는 코드·AST·해시를 확인했고 시험을 직접 실행한 것으로 표시하지 않았다.

## 주석과 오류 정정

OpenCodex의 `google-antigravity/gemini-3.8-flash`에 필요한 코드만 전달했다.
길이 제한으로 불완전하게 끝난 첫 응답은 적용하지 않았다. 완결된 네 묶음의 37개
docstring 초안을 Codex가 코드와 대조해 수정했다. [주석 증거 JSON](2026-10-24-gemini-comment-evidence.json)에
실제 응답 모델, 공급자가 반환한 사용량, 적용 전후 해시와 실행 AST 동일성 결과를 남겼다.

Gemini 초안에 있던 원자적 계수·무조건적인 예외 정리·보안 샌드박스·무난수 실행 등의 과장,
타깃 사전을 가중치라고 부른 설명, NaN/Inf 처리와 모드 복원 범위를 바로잡았다.
과거 설계 문서가 checkout에 미동봉된 사실을 표시했으며 원래 과학 수식은 보존했다.
이는 변경 부품과 26개 파일의 제한된 낡은 주석 점검 결과이며 저장소의 모든 주석 교체를 뜻하지 않는다.

## 남은 한계와 기존 실행

전체 저장소의 무오류, 실제 학습 수렴, GPU 경로, 전체 crash-loop matrix,
실제 데이터 수치 재현은 이번 검증의 결론이 아니다. 학습 조정기와 역사적 연구 실행기에는
여전히 여러 책임이 남아 있다. 향후 분리는 관찰 가능한 계약과 시험을 먼저 확보해서 진행해야 한다.
원시 자료·checkpoint payload·보호된 분할 모듈 본문은 열지 않았다.

평가에서 불필요한 분기를 생략하는 것은 확인했지만 시간/메모리 개선율은 측정하지 않았다.
작은 CPU 단일 스레드 시험과 패키징 작업은 기존 `train_S_cpu` 측정과 같은 호스트에서
겹쳤다. 따라서 Task 8 결과를 배타적 호스트의 무간섭 비용으로 해석하면 안 된다.
측정된 실행 자체는 중단·재개·교체하지 않았고 Strata는 켜지 않았다.

이 브랜치는 새 소프트웨어 버전이다. 원래 입력·분할·라벨·평가 정의와 원고의 동결 수치는
바꾸지 않았으나, 달라진 소스 해시를 과거 실행의 동일한 코드로 취급할 수 없다.
현재 run을 이 코드로 Resume하지 않는다. 기존 S4a 측정과 G-S4a 정지 경계는 유지한다.
새 branch의 커밋·일반 push는 승인된 코드와 문서만 대상으로 하며 기존 실행 branch에 병합하지 않는다.
