# 석사학위논문 초안 패키지 v1 (`master_thesis_v1`)

작성: 2026-09-21  
상태: **현황 파악 + 부산대 제본 순서의 국문 장 골격**. 완결 원고·HWP/LaTeX 제출본 아님.

## 목적

1. 저널·규모분리·잠금에 쌓인 완료 증거를 한곳에 정리한다.
2. [`MASTER_THESIS_RESEARCH_PLAN_20260919.md`](../MASTER_THESIS_RESEARCH_PLAN_20260919.md)의 M-RQ1–4·목차안을 부산대 석사 제본 순서에 맞춰 재배치한다.
3. 장마다 **무엇을 쓸지 / 어느 문서에서 수치를 옮길지 / 무엇이 미실행·금지인지**를 고정한다.

## 부산대 양식 안내

- 공식 서식(HWP): **학생지원시스템(Onestop) → 졸업 → 학위청구논문 → 논문작성지침** (로그인 필요).
- 공개 지침의 제본 순서만 이 패키지에 반영:

  표지 → 면지 → 속표지·인준지 → 목차 → **국문 초록**(2쪽 이내) → 본문 → 참고문헌 → 부록 → **영문 초록**(2쪽 이내)

- 글꼴·줄간격 등은 지도교수 재량. 본 패키지는 Markdown 골격이다.

## 파일 목록

| 파일 | 역할 |
|---|---|
| [00_INVENTORY_AND_STATUS.md](00_INVENTORY_AND_STATUS.md) | Phase 1: 완료/미실행/금지 현황 |
| [01_MRQ_CLAIM_MAP.md](01_MRQ_CLAIM_MAP.md) | Phase 2: M-RQ ↔ 장 ↔ C1–C22 |
| [00_FRONT_MATTER.md](00_FRONT_MATTER.md) | 표지·목차·국/영 초록 자리 |
| [ch01_서론.md](ch01_서론.md) … [ch08_논의결론.md](ch08_논의결론.md) | 본문 장 골격 |
| [APPENDIX_재현.md](APPENDIX_재현.md) | 재현·스크립트·결과 링크 |

## 저널과의 관계

| | 저널 `journal_manuscript_v1` | 석사 `master_thesis_v1` |
|---|---|---|
| 언어 | 영문(투고) | **국문** |
| 깊이 | 검증 스파인 + T/S 병행 | 설계·대안·한계 장 확장 |
| 수치 | 동일 원장 | **동일 원장만 전사** |
| 심화 | 이연 | M-F* / F1–F5는 “미실행”만 |

`docs/tog_manuscript/`는 CoG 계보 **읽기 전용**. LOCK/CONTRACT의 기존 수치 행을 수정하지 않는다.

## 집필 규칙

- 새 실험·재집계·재학습·새 수치 금지 (`AGENTS.md`).
- Forbidden 목록: `JOURNAL_FINISH_LOCK` + claim map F1–F8.
- J-RQ / M-RQ 번호 혼동 금지 (J-RQ1 ≈ M-RQ2).
- T와 S는 **코호트별 블록**; 규모 기울기·기전·합동 대체 금지.

## 다음 단계 (이 패키지 밖)

1. 저자: DOSSIER pending·RQ 결정 (`thesis/docs/TASKS.md` A1–A5).
2. 장별 국문 산문 확장: Markdown 골격 → [`thesis/latex/`](../../thesis/latex/) 절 초고.
3. 보완 실험: [`../SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md`](../SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md) — 석사는 G0→E1… 순서; **작업서 없이 실행하지 않음**.
4. Onestop HWP / Overleaf (`thesis/docs/OVERLEAF.md`).

## LaTeX 구현 (2026-09-21 추가)

부산대 규격 LaTeX 프로젝트는 [`thesis/`](../../thesis/README.md) 에 있다 (8장·부록 A–E, 수치 매크로 `config/numbers.tex`, 출처 `thesis/docs/NUMBERS.md`, 주장 지도 `thesis/docs/CLAIMS.md`). 이 패키지의 규칙(새 수치 금지, Forbidden, 코호트별 보고)을 그대로 따른다.

## 서술 규칙 변경 (2026-09-21 추가; 저자 지시)

LaTeX 학위논문(`thesis/`)은 이 패키지의 장 골격을 유지하되 서술은 **일반 논문 구성**(문제 정의 → 어려움 → 정의 → 해법 → 근거)을 따른다. `ch01_서론.md` §1.3과 `ch02_관련연구.md` §2.3의 "선행 CoG 연구의 목적을 유지한다"는 문구는 학위논문 본문에서 3인칭 관련 연구 서술(제 2.2·2.5 절, "Baek과 Kwon (2026)")로 대체되었으며, 한계는 출판 논문 / 파이프라인 코드 / 학회 이후 분석으로 귀속한다. 보완 실험 E1–E5·G0의 결과는 `thesis/docs/STATUS.md`(하위 분석 상태)와 `thesis/docs/CLAIMS.md`(C23–C32)로 관리한다. 이 패키지의 기존 행은 수정하지 않는다.
