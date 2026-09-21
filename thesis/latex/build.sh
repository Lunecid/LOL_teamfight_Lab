#!/usr/bin/env bash
# 부산대 학위논문 빌드 스크립트
#   1) tectonic 이 있으면 tectonic 으로 (bibtex 포함 자동 재실행)
#   2) 없으면 latexmk -xelatex
set -euo pipefail
cd "$(dirname "$0")"
if command -v tectonic >/dev/null 2>&1; then
  tectonic -X compile --keep-logs main.tex
elif command -v latexmk >/dev/null 2>&1; then
  latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
else
  echo "tectonic 또는 latexmk(XeLaTeX) 가 필요합니다." >&2
  echo "  tectonic: https://tectonic-typesetting.github.io/  |  TeX Live: https://tug.org/texlive/" >&2
  exit 1
fi
echo "==> main.pdf 생성 완료"
