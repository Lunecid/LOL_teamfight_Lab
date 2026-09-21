# 석사 학위논문 (부산대학교 데이터사이언스전문대학원 양식)

**비동기 공개 텔레메트리 기반 교전 구성과 전략적 가치 개선 예측: League of Legends를 중심으로**

`docs/master_thesis_v1/` 의 8장 골격(M-RQ1–4)과 저널 원고 `docs/journal_manuscript_v1/` 의 동결 증거(C1–C22)를 **부산대학교 대학원 학위논문 규격**의
LaTeX 프로젝트(Overleaf 호환)로 구현한 것이다. 집필 규칙은 저장소의 `AGENTS.md`(새 수치 금지), `docs/JOURNAL_FINISH_LOCK_20260920.md`(금지 주장 F1–F8),
`docs/SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md` §6(T/S 코호트별 보고)을 따른다. 실행 작업은 `.ai/tasks/` 작업서로 Cursor에 인계한다.

```
thesis/
├── README.md              # 이 문서
├── docs/                  # (md) 설계·계획·출처 계층 — 원고와 분리
│   ├── OUTLINE.md         #   장·절·객체 지도, RQ 대응, 상태
│   ├── CLAIMS.md          #   주장 → 근거(표·그림·수치) → 출처
│   ├── STYLE.md           #   용어집(국·영), 표기 규칙, 문체 규칙
│   ├── NUMBERS.md         #   수치 매크로 레지스트리와 출처
│   ├── OVERLEAF.md        #   Overleaf 설정과 git 동기화
│   ├── CHECKLIST.md       #   부산대 제출 체크리스트
│   ├── TASKS.md           #   저자 결정·원고·실행(이연 계약)·검증 작업 목록
├── forms/                 # 부산대 공식 서식 원본 (HWP 가이드, Word 서식, 여백 그림)
└── latex/                 # ★ Overleaf 프로젝트 루트
    ├── main.tex           #   조립 루트: 무엇을 어떤 순서로 붙이는가
    ├── pnuthesis.cls      #   기반 클래스: 규격(여백·표지·인준지·차례·초록)만 담당
    ├── latexmkrc          #   XeLaTeX + bibtex
    ├── config/
    │   ├── meta.tex       #   제목·저자·소속·일자 (설정 객체)
    │   ├── numbers.tex    #   실험 산출 수치 매크로 — 단일 진실 공급원
    │   └── notation.tex   #   수학 기호·연산자 매크로
    ├── frontmatter/       #   abstract_ko.tex, abstract_en.tex
    ├── chapters/
    │   ├── 01_introduction/   # 서론 (I1–I4, M-RQ1–4, 기여·비주장, 증거 지위)
    │   ├── 02_related_work/   # 승률 모형·교전 검출·사건 가치·적정 점수·측정 타당성·선행 연구 위치
    │   ├── 03_data/           # 비동기 텔레메트리, 코퍼스, 데이터 역할, 교전 전 입력, 시간 계약 (M-RQ1)
    │   ├── 04_engagement/     # v3.3 정의: G, D, R/B/M, 상수표, 규모 계급·코호트, 결과 시점, 킬 없는 전투 (M-RQ1)
    │   ├── 05_value/          # 동결 평가기 V̂, SVI, 삼중 분해, quiet 대조, 대응, 지평 (M-RQ2)
    │   ├── 06_prediction/     # q, PT_flex, 선정·동결, 평가 규칙, T/S 주 결과, 전이·합동 (M-RQ3)
    │   ├── 07_validation/     # B40, λ·s_Q, CORP, 외부 점수 전용 (M-RQ4)
    │   ├── 08_discussion/     # 연결, 기여, 평가기 의존, 금지 주장, 한계, 향후, 결론
    │   └── appendix/          # A 재현, B 상수표 전체, C census, D 증거 추적, E 이연 계획
    ├── tables/            #   표 객체: 파일당 하나 (tab_<label>.tex)
    ├── figures/           #   그림 객체 (fig_<label>.tex) + src/ 이미지
    ├── algorithms/        #   알고리즘 객체 (alg_<label>.tex)
    ├── bib/references.bib #   IEEEtranN 참고문헌 (확인된 문헌만)
    ├── fonts/             #   나눔명조·나눔고딕 (SIL OFL) — 경로로 직접 로드
    └── build.sh
```

## 설계 원칙 (객체지향 비유)

| 개념 | 대응 | 규칙 |
|---|---|---|
| 기반 클래스 | `pnuthesis.cls` | 규격(레이아웃)만. 내용·수치·기호를 넣지 않는다 |
| 설정 객체 | `config/meta.tex` | 표지·인준지·초록에 들어가는 메타 정보만 |
| 상수 객체 | `config/numbers.tex` | 실험·데이터 수치는 본문·표·초록 어디서든 매크로(`\aucLGBM{}`)로만 쓴다. 재실행 시 이 파일만 갱신 |
| 공용 인터페이스 | `config/notation.tex` | 기호는 매크로로 정의하고 본문은 매크로를 호출 |
| 조립 루트 | `main.tex` | 순서만 정한다. 내용 편집 금지 |
| 모듈 | `chapters/<장>/chapter.tex` | `\chapter` 와 절 `\input` 목록만 |
| 단위 | `chapters/<장>/<NN>_<slug>.tex` | 절 하나 = 파일 하나. 헤더의 정의 라벨/참조 라벨/포함 객체가 그 파일의 인터페이스 |
| 재사용 객체 | `tables/`, `figures/`, `algorithms/` | 캡션·라벨·내용을 한 파일에 캡슐화. 여러 절에서 `\input` 가능 |
| 문서화 계층 | `docs/*.md` | 원고에 넣지 않는 설계·근거·상태·작업 명세 |

절을 옮기거나 이름을 바꿀 때는 (1) 파일 이동, (2) `chapter.tex` 의 `\input` 수정, (3) `docs/OUTLINE.md` 갱신의 세 단계만 필요하다.

## 공식 규격 출처

| 출처 | 내용 |
|---|---|
| [학생지원시스템 › 졸업 › 학위청구논문 › 논문 작성 지침](https://onestop.pusan.ac.kr/page?menuCD=000000000000302) | 제본 순서, 용지, 표지 색상, 언어, 초록 규정, 서식 파일 |
| `forms/학위논문작성가이드(국문석사).hwp` | 공식 HWP 가이드 (여백·글자 크기·표지·인준지·초록 서식) |
| `forms/학위논문작성가이드(영문석사).hwp` | 영문 논문용 가이드 |
| `forms/학위논문_국문석사서식_.docx`, `forms/학위논문_영문석사서식_.docx` | 공식 Word 서식 (필요시 활용) |
| `forms/margins_diagram.png` | 가이드의 [그림 1] 본문 여백 |

가이드에서 추출한 핵심 규격:

- **용지** A4 (210 × 297 mm), 80 g 이상 백색 모조지. 표지는 회색 레자크지, 글씨는 흑색.
- **여백** 위 35 / 아래 25 / 왼쪽 25(+제본 5) / 오른쪽 25 mm, 꼬리말 15 mm (HWP 가이드 `PAGE_DEF` 값).
- **제본 순서** 표지 → 면지 → 속표지 및 인준지 → 목차 → 초록(본문 언어) → 본문 → 참고문헌 → 부록 → 초록(다른 언어, 2쪽 이내).
- **표지** "석 사 학 위 논 문"(16 pt) / 제목(22 pt) / 성명(16) / 부산대학교 대학원(16) / OOO과(16) / 학위수여년월(16).
  전문대학원은 "부산대학교 대학원" 자리에 전문대학원 명칭을 쓴다 (`\pnugradschool{데이터사이언스전문대학원}`).
- **속표지·인준지** 제목(22) / "이 논문을 OO석사 학위논문으로 제출함"(16) / 성명·대학원·학과·지도교수(14) /
  "OOO의 OO석사 학위논문을 인준함"(16) / **최종 심사 연월일**(14, 반드시 기재) / 위원장·위원·위원(14).
- **초록** 제목(14) / 성명(12) / 부산대학교 대학원 OOO과(11) / 요약(12) / 본문(11). 국문·영문 각 2쪽 이내.
- **본문·참고문헌·부록**의 글꼴·크기·줄간격은 지도교수 지도에 따라 자유 (기본값: 11 pt, 줄간격 160 %).
- 규정은 바뀔 수 있으므로 최종 제출 전 **100 % 인쇄 후 자로 확인**하고 학과 사무실에 문의할 것.

## 빌드

XeLaTeX 계열 엔진이 필요하다 (한글: `kotex`/`xetexko`). Overleaf 설정은 `docs/OVERLEAF.md` 참고.

```bash
cd thesis/latex
./build.sh              # tectonic 이 있으면 tectonic, 없으면 latexmk -xelatex
# 또는
latexmk -xelatex main.tex
```

- [Tectonic](https://tectonic-typesetting.github.io/) 단일 바이너리로 빌드 가능 (필요 패키지를 자동 내려받음).
  Tectonic 의 XeTeX 는 ICU 한국어 줄바꿈 자원이 없어 클래스에서 `\XeTeXlinebreaklocale ""` 로 비활성화하고
  xetexko 의 자체 줄바꿈을 쓴다 (TeX Live 에서도 동일하게 동작).
- 글꼴은 `fonts/` 의 나눔 글꼴을 경로로 직접 읽으므로 시스템 설치가 필요 없다. 라틴 글꼴은 TeX Gyre (배포판 포함).

## 작업 흐름

1. **계획·설계·원고** (이 폴더): 절·표·수치를 고친다. 수치는 `latex/config/numbers.tex` 만, 출처는 `docs/NUMBERS.md`.
2. **실행** (Cursor): 이연 항목(F1–F6, M-F0–M-F3)은 `.ai/tasks/T0xx.md` 작업서와 예비 선언 계약 아래에서만 실행하고 `.ai/reports/` 에 보고한다.
3. **반영**: 보고서의 값을 `numbers.tex` 에 옮기고 `NUMBERS.md` 에 출처 필드를 기록한 뒤 해당 절·표만 갱신한다.

## 작성 현황

- [x] 클래스·앞부속(표지, 인준지, 차례, 국·영문 초록), 소속 = 데이터사이언스전문대학원 데이터사이언스학과
- [x] 8장 + 부록 A–E 초안: 모든 수치는 원장 매크로, 참고문헌은 저장소 검증 풀
- [ ] 저자 결정 5건 (`docs/TASKS.md` A1–A5): pick 제외 문장, EXT T/S 병기, 학위명·일자·심사위원, 릴리스 DOI, 실행할 이연 항목
- [ ] 그림: 코호트·역할 표본 흐름, ΔBrier by p_pre 점도표 (저널 SVG의 값), 신뢰도 다이어그램(T014)
- [ ] 사례 추적 절, 문체 통일, 금지 문구 grep, 제출 체크리스트
