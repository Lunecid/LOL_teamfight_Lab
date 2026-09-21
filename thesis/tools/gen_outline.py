#!/usr/bin/env python3
"""Generate thesis/docs/OUTLINE.md (section map: chapter / section / file / floats) and
thesis/docs/REVERSE_OUTLINE.md (first sentence of every paragraph per section, for the
reverse-outline check of the research-paper-writing skill) from the LaTeX sources reachable
from main.tex. Read-only over the sources; rewrites the two docs."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LATEX = ROOT / "thesis" / "latex"
OUT_OUTLINE = ROOT / "thesis" / "docs" / "OUTLINE.md"
OUT_REVERSE = ROOT / "thesis" / "docs" / "REVERSE_OUTLINE.md"


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def inputs_of(path):
    s = strip_comments(path.read_text(encoding="utf-8"))
    return [m.group(1) for m in re.finditer(r"\\input\{([^}]+)\}", s)]


def detex(s):
    s = re.sub(r"\\(?:cite[tp]?|ref|eqref|label|path|texttt|emph|textbf)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", "", s)
    s = re.sub(r"[{}$]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def first_sentence(par):
    par = detex(par)
    m = re.search(r"^(.*?[다요음임가까]\.)(\s|$)", par)
    return (m.group(1) if m else par)[:160]


def main():
    chapters = []  # (title, label, [(sec_title, sec_label, file, floats, paragraphs)])
    main_tex = LATEX / "main.tex"
    for chap in inputs_of(main_tex):
        cp = LATEX / (chap + ".tex")
        if not cp.exists() or "chapters/" not in chap:
            continue
        cs = strip_comments(cp.read_text(encoding="utf-8"))
        ctitle = re.search(r"\\chapter\{([^}]*)\}", cs)
        clabel = re.search(r"\\label\{([^}]*)\}", cs)
        secs = []
        if "appendix/chapter" in chap:
            files = [LATEX / (f + ".tex") for f in inputs_of(cp)]
        else:
            files = [LATEX / (f + ".tex") for f in inputs_of(cp)]
        for f in files:
            if not f.exists():
                continue
            fs = strip_comments(f.read_text(encoding="utf-8"))
            if "appendix/chapter" in chap:
                t = re.search(r"\\chapter\{([^}]*)\}", fs)
                l = re.search(r"\\label\{(app:[^}]*)\}", fs)
            else:
                t = re.search(r"\\section\{([^}]*)\}", fs)
                l = re.search(r"\\label\{(sec:[^}]*)\}", fs)
            floats = re.findall(r"\\input\{(tables|figures)/([^}]+)\}", fs)
            body = re.sub(r"\\(?:section|chapter)\{[^}]*\}\s*\\label\{[^}]*\}", "", fs)
            body = re.sub(r"\\begin\{(table|figure|longtable|equation|description|itemize|enumerate|definition)\}.*?\\end\{\1\}", "", body, flags=re.S)
            paras = [p for p in re.split(r"\n\s*\n", body) if p.strip() and not p.strip().startswith("\\input") and not p.strip().startswith("\\begin")]
            secs.append((detex(t.group(1)) if t else "(도입)", l.group(1) if l else "", f.relative_to(LATEX), [f"{d}/{n}" for d, n in floats], [first_sentence(p) for p in paras]))
        chapters.append((detex(ctitle.group(1)) if ctitle else chap, clabel.group(1) if clabel else "", secs))
    # OUTLINE.md
    o = ["# 논문 구조 (OUTLINE)", "", "`tools/gen_outline.py`가 `latex/main.tex`에서 도달 가능한 장·절 파일로부터 생성한다 (손으로 고치지 않음). 조립 순서: 표지 → 면지 → 속표지·인준지 → 차례 → 국문 초록 → 본문 1–8장 → 참고 문헌 → 부록 A–E → 영문 초록.", "",
         "| 장 | 절 | 라벨 | 파일 | 포함 객체 |", "|---|---|---|---|---|"]
    for ct, cl, secs in chapters:
        o.append(f"| **{ct}** (`{cl}`) | | | | |")
        for st, sl, fp, fl, _ in secs:
            o.append(f"| | {st} | `{sl}` | `latex/{fp}` | {', '.join('`'+x+'`' for x in fl)} |")
    o += ["", "## M-RQ ↔ 장", "", "| M-RQ | 장 | 주 결과 | 보완 실험 |", "|---|---|---|---|",
          "| M-RQ1 교전 사례 구성 | 3, 4 | 정의·상수표·T/S 규칙 | e_fixed (4.6), OAT 사례 구성 (4.8) |",
          "| M-RQ2 결과 가치 | 5 | C1–C8 | 물질 축·후속 결과 (5.7–5.8), 지평·동료 평가기 (5.9), S_hold (5.10) |",
          "| M-RQ3 사전 예측 | 6 | C9–C10, C18–C20 | 정보군 비교 (6.8) |",
          "| M-RQ4 적용 범위 | 7 | C11–C17, C21–C22 | 관측 갱신 층화 (7.3), 외부 부트스트랩 (7.5) |", ""]
    OUT_OUTLINE.write_text("\n".join(o), encoding="utf-8")
    # REVERSE_OUTLINE.md
    r = ["# 역개요 (REVERSE OUTLINE)", "", "`tools/gen_outline.py`가 각 절의 문단 첫 문장을 뽑은 것이다 (research-paper-writing 스킬의 reverse-outline 점검용). 검토 규칙: 각 첫 문장이 그 절의 메시지에 연결되는가, 문단이 하나의 메시지를 갖는가. 검토 결과는 `SELF_REVIEW.md` §역개요.", ""]
    for ct, cl, secs in chapters:
        r.append(f"## {ct}")
        for st, sl, fp, fl, paras in secs:
            r.append(f"\n### {st} (`{sl}`)")
            for i, p in enumerate(paras, 1):
                r.append(f"{i}. {p}")
        r.append("")
    OUT_REVERSE.write_text("\n".join(r), encoding="utf-8")
    print(f"wrote {OUT_OUTLINE.relative_to(ROOT)} and {OUT_REVERSE.relative_to(ROOT)} ({sum(len(s) for _,_,s in chapters)} sections)")


if __name__ == "__main__":
    main()
