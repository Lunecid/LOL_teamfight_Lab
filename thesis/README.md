# 석사 학위논문 (부산대학교 데이터사이언스전문대학원 양식)

**비동기 공개 텔레메트리 기반 교전 구성과 전략적 가치 개선 예측: League of Legends를 중심으로**

`docs/master_thesis_v1/` 의 8장 골격(M-RQ1–4), 동결 주 결과(C1–C22)와 보완 실험(E1–E5, G0; C23–C32)을 **부산대학교 학위논문 규격**의 LaTeX 프로젝트(Overleaf 호환)로 구현한 것이다.
집필 규칙은 저장소의 `AGENTS.md`(새 수치 금지)와 `docs/STYLE.md`(서술 계약: 일반 논문 서술, 결과별 해석 조건, 캡션은 이름만, 상태 어휘)를 따른다. 실행 작업은 `.ai/tasks/` 작업서로 Cursor에 인계한다.

```
thesis/
├── README.md              # 이 문서
├── docs/                  # (md) 설계·상태·출처 계층 — 원고와 분리
│   ├── STYLE.md           #   서술·표기 계약 (Widom 순서, CoG 2026 3인칭, 조건의 위치, 캡션·표주, 검사 도구 역할)
│   ├── STATUS.md          #   증거 상태 원장: 하위 분석 ID별 DONE/NOT_RUN/PARTIAL/BLOCKED, 저자 결정 A1–A5
│   ├── CLAIMS.md          #   주장 C1–C32 → 근거 객체 → 해석 조건; M-RQ 대응; 해석 범위 내부 ID
│   ├── NUMBERS.md         #   주 결과 매크로의 출처 (config/numbers.tex)
│   ├── NUMBERS_SUPP.md    #   보완 실험 매크로·생성 표의 통계량·분자·분모·출처 (생성물)
│   ├── FEATURE_CENSUS.md  #   입력 특징 전수조사: 두 특징 경로, 폭 대조, 블록별 열 이름 (생성물)
│   ├── OUTLINE.md         #   장·절·라벨·파일·객체 지도 (생성물)
│   ├── REVERSE_OUTLINE.md #   절별 문단 첫 문장 (생성물; 역개요 점검용)
│   ├── SELF_REVIEW.md     #   paper-self-review 7단계 결과와 주장 감사 (내부 점검)
│   ├── TASKS.md           #   저자 결정·원고·실행·검증 작업 목록
│   ├── CHECKLIST.md       #   부산대 제출 체크리스트
│   └── OVERLEAF.md        #   Overleaf 설정과 git 동기화
├── tools/                 # 생성기와 검사기 (Python 3, 표준 라이브러리)
│   ├── gen_supp.py        #   docs/SUPPLEMENTARY_*.json → config/numbers_supp.tex, tables/gen/*.tex, docs/NUMBERS_SUPP.md
│   ├── gen_status_table.py#   docs/STATUS.md → tables/gen/tab_status.tex
│   ├── gen_feature_census.py # docs/ 특징 매니페스트 → docs/FEATURE_CENSUS.md, tables/gen/tab_input_blocks.tex
│   ├── gen_outline.py     #   main.tex 도달 파일 → docs/OUTLINE.md, docs/REVERSE_OUTLINE.md
│   ├── check_numbers_supp.py  # 생성물이 JSON 원장과 일치하는지 (차단)
│   ├── check_thesis.py    #   차단 검사: 생성물·미정의 참조·라벨·헤드라인 불변·M-RQ 문구 해시
│   └── check_style.py     #   편집 보고(차단 아님): 부정·대조 표현, 계승·저널 어휘, 상태 어휘, 캡션, 숫자
├── forms/                 # 부산대 공식 서식 원본 (HWP 가이드, Word 서식, 여백 그림)
└── latex/                 # ★ Overleaf 프로젝트 루트
    ├── main.tex           #   조립 루트
    ├── pnuthesis.cls      #   기반 클래스: 규격(여백·표지·인준지·차례·초록)만 담당
    ├── config/
    │   ├── meta.tex       #   제목·저자·소속·일자
    │   ├── numbers.tex    #   주 결과 수치 매크로 (+ Baek & Kwon 2026 cite-only 상수)
    │   ├── numbers_supp.tex   # 보완 실험 수치 매크로 (생성물)
    │   └── notation.tex   #   기호·연산자·\tabnote
    ├── frontmatter/       #   abstract_ko.tex, abstract_en.tex
    ├── chapters/
    │   ├── 01_introduction/   # 게임과 한타 / 문제 / 기존 접근과 어려움 / 접근과 결과 / M-RQ / 기여와 구성
    │   ├── 02_related_work/   # 승률 모형 / 교전 검출·교전 단위 예측(Baek & Kwon 2026, 버전별 한계) / 사건 가치 / 적정 점수 / 남은 문제와 위치
    │   ├── 03_data/           # 텔레메트리, 코퍼스, 데이터 역할, 입력, 시간 계약 (M-RQ1)
    │   ├── 04_engagement/     # G, D, R/B/M, 상수표, 코호트, 결과 시점(+e_fixed), 분석 대상, 정의 민감도(+OAT) (M-RQ1)
    │   ├── 05_value/          # V̂, SVI, 삼중 분해, quiet, 대응(+물질 축·nextobj), 같은 사례 라벨, 지평·동료 평가기, S_hold 분해 (M-RQ2)
    │   ├── 06_prediction/     # q, PT_flex, 선정·동결, 평가 규칙(증거의 지위), T/S 주 결과, 전이·합동, 정보군 비교 (M-RQ3)
    │   ├── 07_validation/     # B40, λ·s_Q, 관측 갱신 층화, CORP, 외부 점수 전용(+부트스트랩) (M-RQ4)
    │   ├── 08_discussion/     # 결과의 연결, 해석의 범위(표), 평가기 의존성, 한계, 향후 과제, 결론
    │   └── appendix/          # A 재현·무결성·설계–실행 차이, B 상수표, C census, D 증거 추적, E 실행 상태
    ├── tables/            #   손으로 쓴 표 (tab_<label>.tex; 캡션은 이름만, 조건은 \tabnote)
    │   └── gen/           #   생성 표 (gen_supp.py, gen_status_table.py, gen_feature_census.py) — 손으로 고치지 않음
    ├── figures/           #   그림 객체
    ├── bib/               #   definition_refs.bib, econometrics_for_lol.bib, thesis_extra.bib (검증된 문헌만)
    ├── fonts/             #   나눔명조·나눔고딕 (SIL OFL) — 경로로 직접 로드
    └── build.sh
```

## 작업 흐름

1. **원고** (여기): 절·표를 고친다. 주 결과 수치는 `latex/config/numbers.tex`만, 출처는 `docs/NUMBERS.md`.
2. **실행** (Cursor): 보완 실험은 `docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md`의 계약과 `.ai/tasks/T0xx.md` 작업서 아래에서만 실행하고 `.ai/reports/`에 보고한다.
3. **반영**: 새 `docs/SUPPLEMENTARY_*.json`이 생기면 `python thesis/tools/gen_supp.py`로 매크로·표를 재생성하고, `docs/STATUS.md`의 상태를 고친 뒤 `python thesis/tools/gen_status_table.py`를 실행하며, 해당 절만 갱신한다.
4. **검사**: `python thesis/tools/check_thesis.py`(차단; `gen_supp`·`gen_status_table`·`gen_feature_census`의 생성물 일치를 포함), `python thesis/tools/check_style.py --baseline thesis/tools/style_baseline.json --verbose`(보고), `python thesis/tools/gen_outline.py`(문서 갱신).

## 빌드

XeLaTeX 계열 엔진이 필요하다 (한글: `kotex`/`xetexko`). Overleaf 설정은 `docs/OVERLEAF.md` 참고.

```bash
cd thesis/latex
./build.sh              # tectonic 이 있으면 tectonic, 없으면 latexmk -xelatex
```

- Tectonic 의 XeTeX 는 ICU 한국어 줄바꿈 자원이 없어 클래스에서 `\XeTeXlinebreaklocale ""` 로 비활성화하고 xetexko 의 자체 줄바꿈을 쓴다.
- 글꼴은 `fonts/` 의 나눔 글꼴을 경로로 직접 읽으므로 시스템 설치가 필요 없다. 라틴 글꼴은 TeX Gyre.

## 공식 규격 출처

| 출처 | 내용 |
|---|---|
| [학생지원시스템 › 졸업 › 학위청구논문 › 논문 작성 지침](https://onestop.pusan.ac.kr/page?menuCD=000000000000302) | 제본 순서, 용지, 표지 색상, 언어, 초록 규정, 서식 파일 |
| `forms/학위논문작성가이드(국문석사).hwp` | 공식 HWP 가이드 (여백·글자 크기·표지·인준지·초록 서식) |
| `forms/학위논문_국문석사서식_.docx`, `forms/학위논문_영문석사서식_.docx` | 공식 Word 서식 |
| `forms/margins_diagram.png` | 가이드의 [그림 1] 본문 여백 |

핵심 규격: A4, 80 g 이상 백색 모조지, 표지 회색 레자크지; 여백 위 35 / 아래 25 / 왼쪽 25(+제본 5) / 오른쪽 25 mm; 제본 순서 표지 → 면지 → 속표지·인준지 → 목차 → 국문 초록 → 본문 → 참고문헌 → 부록 → 영문 초록(각 초록 2쪽 이내); 표지·인준지·초록 글자 크기(16/22/14/12/11 pt); 전문대학원은 "부산대학교 대학원" 자리에 전문대학원 명칭(`\pnugradschool{데이터사이언스전문대학원}`). 최종 제출 전 100 % 인쇄 후 자로 확인.

## 작성 현황 (2026-09-21)

- [x] 클래스·앞부속(표지, 인준지, 차례, 국·영문 초록), 소속 = 데이터사이언스전문대학원 데이터사이언스학과
- [x] 8장 + 부록 A–E; 일반 논문 서술로 재구성(Widom), Baek & Kwon 2026은 2장에서 3인칭, 결과별 해석 조건, 캡션 이름화
- [x] 보완 실험 E1–E5·G0 통합(생성 매크로·표), 실행 상태 원장(`STATUS.md`)과 부록 E 상태 표
- [x] 입력 특징 전수조사(`docs/FEATURE_CENSUS.md`): 351/352/361/362/257/7,105 대조, 3.4절 사실 오류 수정
- [x] 차단 검사 통과(`check_thesis.py`), 자기 심사(`SELF_REVIEW.md`)
- [ ] 저자 결정 A1–A5 (`docs/TASKS.md`)
- [ ] 그림 추가(B1), 사례 추적 본문화(B2; event-prefix 덤프 후), 문헌 재확인(B4), 제출 체크리스트(B5)
