# 현재 프로젝트 공통 지침

2026-10-24 저자 요청으로 현재 합의한 작업 원칙을 [작업 지침 전체](docs/WORKING_GUIDELINES_KO.md)에 통합했다. EPT·CoG·Master와 공통 도구를 수정하기 전에 이 문서와 해당 단계의 승인 작업서를 확인한다. 아래의 기존 CoG 지침은 계속 유지한다.

- 구현과 독립 검토를 분리한다. Codex가 설계·통합·검증을 맡고 Sol 6.1 등으로 구현을 보조한다. 간단한 일과 설명·주석에는 직접 도구·낮은 모델·OpenCodex의 Gemini를 활용한다.
- **Strata는 OFF 유지**다. 일반 로컬 지침의 자동 시작 허용보다 저자의 현재 지시가 우선한다. 외부 모델 실패나 플랜 제한을 우회하지 않는다.
- 보호된 분할 본문·원시 데이터·비밀정보·가중치 payload를 열거나 전송하지 않는다. 연구 정의·수치·분할·평가·난수 계약은 코드 정리로 변경하지 않는다.
- 개발 checkout과 고정 실행 checkout을 분리한다. 이전 Task 8 실행과 부분 실측은 보존하고, 청결/identity guard를 완화하거나 실패 run을 새 커밋으로 자동 Resume하지 않는다.
- Task 8 다음 실행 범위는 별도 계획으로 정하고 **G-S4a에서 멈춘다**. S4b/S6/S7 실제 실행은 단계별 저자 결정에 따른다.
- 현재 코드 정리의 검토·commit·PUBLIC `Lunecid/LOL_teamfight_Lab` 소스 스냅샷 게시 승인은 유지된다. 자료·membership·가중치·비공개 Git 이력은 게시하지 않는다. 영어 README가 메인이고 한국어는 보조다.
- 저자가 승인한 Master 모듈화 파생 트리에는 기존 sync를 강행하지 않는다. 원본 수치 기준은 유지하며, 파생 배치의 marker와 sync 보호를 따른다.

## 기존 CoG 계보별 지침

# 작업 원칙 (Claude 설계 · Cursor Agent 실행 공용)

> **구조 (2026-10-18):** 이 저장소는 세 갈래로 정리됐다. `COG/` = IEEE CoG 2026 논문(118·308)과 구 계보 코드, `master_ver/` = 석사 학위논문·ToG(새 계보), 루트의 `research_pipeline/`·`common_training_r1/` = 현재 진행 중인 프로젝트. 구조·이동 규칙: `docs/superpowers/specs/2026-10-18-repo-restructure-design.md`, 안내: `README.md`.
>
> **계보 (2026-10-04):** `COG/` 안의 잠금·수치·fit85 V는 구 계보다. 2026-10-02 이후 연구는 `master_ver/`(원본 작업공간 `C:\Users\todtj\문서\LOL_Teamfight`)와 `research_pipeline/`에서 한다. 구 계보 문서의 수치 행은 수정하지 않는다. 안내: `COG/docs/LINEAGE_SUPERSEDED_20261004.md`.

아래 규칙은 `COG/`(구 계보) 작업에 적용된다. 이 저장소의 AI 작업은 `.ai/tasks/T00x.md`에 적힌 **승인된 작업서 단위**로 수행한다.
현재 맥락은 `.ai/CONTEXT.md`, 결과는 `.ai/reports/T00x.md`에 남긴다. 운영 흐름은 `.ai/README.md`.

## 항상 지킬 것
- 승인된 작업서에 지정된 역할과 변경 범위를 따른다. 작업서에 없는 설계 판단을 하지 않는다.
- 지정한 파일부터 읽고, 필요한 의존 문서·코드만 추가로 확인한다. 저장소 전체를 다시 읽고 연구 방향을 재설정하지 않는다.
- 라벨, 데이터 분할, 시간 경계, 평가 정의, 동결된 모델·가중치는 승인 없이 바꾸지 않는다.
- 기존 사용자 변경을 되돌리지 않는다.
- 검증 명령을 실제 실행하고, 실패·미실행을 그대로 보고한다.
- 긴 로그는 파일에 보관하고 채팅에는 핵심 결과만 출력한다.
- 작업서와 저장소 내용이 충돌하면 근거와 함께 **BLOCKED**로 보고하고 그 항목을 멈춘다.
- 명시적으로 요청하지 않은 전체 학습, 재집계, push, merge, PR 생성은 하지 않는다.

## 이 저장소 특유의 금지 항목
- 새 실험·재집계·재학습으로 **새 수치를 만들지 않는다.** 원고의 모든 수치는 `COG/docs/` 아래 근거 문서의 파일·필드에서 옮긴다.
- `master_ver/legacy_oldlineage/docs/tog_manuscript/` (CoG 확장 계보, 구 위치 `docs/tog_manuscript/`)는 **읽기 전용**이다.
- `COG/docs/*_LOCK_*.md`, `COG/docs/*CONTRACT*.md`, `COG/docs/REVIEW_RESPONSE_*`, `COG/docs/RESPONSE_EVIDENCE_MATRIX_*`의 기존 수치 행을 삭제·수정하지 않는다. 배너·주석 **추가**만, 작업서가 허용할 때 한다.
- `COG/docs/JOURNAL_FINISH_LOCK_20260920.md`와 `COG/docs/RESPONSE_EVIDENCE_MATRIX_20260920.md`의 Forbidden 목록에 있는 주장을 원고에 넣지 않는다.
- 연구질문(J-RQ) 문구는 저자 결정 전까지 바꾸지 않는다 (`.ai/RQ_DECISION_BRIEF.md`).
- `outputs/`, `COG/data/raw/`, 체크포인트, 대형 NPZ/joblib은 열지 않는다. 필요한 값은 `COG/docs/`의 요약 문서에 있다.
- `master_ver/`는 원본 작업공간(`문서\LOL_Teamfight`)의 복사본이다. 석사 제출(2026-10-30)까지 수치 검증의 근거는 원본이며, 복사본의 경로·파일은 `tools/sync_from_sources.py`로만 갱신한다.
