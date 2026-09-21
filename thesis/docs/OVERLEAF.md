# Overleaf 설정과 동기화 (OVERLEAF)

## 프로젝트 만들기

1. `thesis/latex/` 폴더를 zip 으로 묶어 Overleaf 에 **New Project → Upload Project** 로 올린다 (`fonts/` 포함, 약 16 MB).
2. 메뉴(Menu) 에서 **Compiler = XeLaTeX**, **Main document = main.tex** 로 설정한다.
   `latexmkrc` 가 bibtex 실행과 synctex 옵션을 지정한다.
3. 첫 컴파일 후 차례·참고문헌이 비어 있으면 한 번 더 컴파일한다.

## 편집 단위

- 절 하나 = 파일 하나 (`chapters/<장>/<NN>_<slug>.tex`). Overleaf 의 파일 트리에서 절을 직접 연다.
- 표·그림·알고리즘은 `tables/`, `figures/`, `algorithms/` 의 객체 파일을 연다. 본문에는 `\input{...}` 만 있다.
- 수치는 `config/numbers.tex` 만 고친다. 본문의 매크로가 자동으로 바뀐다.
- 새 절: 파일을 만들고 `chapters/<장>/chapter.tex` 의 `\input` 목록에 한 줄 추가한다.

## 저장소와 동기화

Overleaf 의 GitHub Sync 는 저장소 전체를 프로젝트로 가져오므로(코드·데이터 포함) 쓰지 않는다.
대신 Overleaf 프로젝트의 git 주소(Menu → Git)를 두 번째 원격으로 두고 `thesis/latex` 서브트리만 주고받는다.

```bash
# 최초 1회
git remote add overleaf https://git.overleaf.com/<project-id>

# 저장소 → Overleaf
git subtree push --prefix thesis/latex overleaf master

# Overleaf → 저장소 (Overleaf 에서 고친 내용 가져오기)
git subtree pull --prefix thesis/latex overleaf master --squash
```

Overleaf 쪽 브랜치 이름은 `master` 로 고정되어 있다. 충돌이 나면 `docs/OUTLINE.md` 의 파일 단위로 나눠 해결한다.

## 주의

- 컴파일 시간 제한(무료 플랜 1 분)에 걸리면 `\includeonly` 대신 `main.tex` 의 `\input` 줄을 잠시 주석 처리한다.
- `fonts/` 의 나눔 글꼴은 경로로 읽으므로 Overleaf 의 시스템 글꼴 유무와 무관하다.
- `.gitignore` 의 빌드 산출물(`*.aux`, `main.pdf` 등)은 Overleaf 에 올리지 않는다.
