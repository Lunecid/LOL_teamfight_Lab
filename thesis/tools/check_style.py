#!/usr/bin/env python3
"""Editorial report (never blocks). Locates places a human should read:

  N1..N6  negative-parallel / hedge patterns          (지만, 아니다, 않는다, 도…도 아니, 주장하지 않, 군더더기)
  V       narrative-voice phrases to re-voice          (선행 연구, 계승, 출발점, 동반 저널, J-RQ, 마감 잠금, I1–I4 …)
  S       strawman vocabulary about prior work         (임의, 근거 없, 최초, 처음으로, 킬 수만)
  ST      status vocabulary near a completion verb     (정합 비교, 점수 민감도, E6, 미노출, S_hold, OAT, 사건 접두, 무결성 …)
  C       captions longer than a name / with a clause ending
  D       literal digits in chapter 1 prose (outside \\cite/\\ref/labels/macros)

Prints a per-file table and totals; with --verbose lists file:line excerpts; --baseline FILE compares
counts to a saved run; --write-baseline FILE saves the current counts.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LATEX = ROOT / "thesis" / "latex"
DIRS = ["chapters", "frontmatter", "tables", "figures"]

PATTERNS = {
    "N1": r"지만(?=[\s,.)])",
    "N2": r"아니(다|라|며|고|었|지)|아닌",
    "N3": r"않(는다|으며|고|았|는|을|은|음|지)",
    "N4": r"도\s[^.]{0,40}?도\s아니",
    "N5": r"(주장|뜻|말|추론|부르|읽|쓰|제시|합산|증명|보증|의미)하?지\s*않",
    "N6": r"동시에|두 진술이|그대로 밝힌다|역사를 고쳐|이를 그대로",
    "V": r"선행 연구|앞선 연구|계승|출발점|확장한다|동반 저널|저널 원고|J-RQ|마감 잠금|규모분리 계약|\bI[1-4]\b|저자의 선행|저자가 참여|계보",
    "S": r"킬 수만|임의로|근거 없|처음으로|처음 도입|최초",
}
STATUS_KW = r"정합 비교|점수 민감도|E6|미노출|S\\?_hold|S_hold|OAT|사건 접두|event-prefix|무결성|G0|정보군|Refit|재학습"
DONE_VERB = r"실행하였|실행했|보고한다|보였다|확인하였|검증하였|수행하였|완료"
NOTDONE_VERB = r"미실행|이연|차단|미완료|남아 있|실행하지 않"

PROTECT = [r"\\cite[tp]?\*?(\[[^\]]*\])?\{[^}]*\}", r"\\(?:ref|eqref|autoref|pageref|cref|label|path|texttt|url|input|include)\{[^}]*\}",
           r"\$[^$]*\$", r"\\pendingauthor\{[^}]*\}", r"\\todo\{[^}]*\}"]


def strip(s: str) -> str:
    s = re.sub(r"(?<!\\)%.*", "", s)
    for p in PROTECT:
        s = re.sub(p, " ⟨M⟩ ", s)
    return s


def captions(raw: str):
    out = []
    for m in re.finditer(r"\\caption(\[[^\]]*\])?\{", raw):
        i = m.end()
        depth = 1
        j = i
        while j < len(raw) and depth:
            if raw[j] == "{":
                depth += 1
            elif raw[j] == "}":
                depth -= 1
            j += 1
        cap = raw[i:j - 1]
        line = raw[: m.start()].count("\n") + 1
        out.append((line, cap, bool(m.group(1))))
    return out


def caption_problem(cap: str):
    plain = re.sub(r"\$[^$]*\$", "X", cap)
    plain = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", "", plain)
    plain = re.sub(r"[{}]", "", plain).strip()
    if re.search(r"[다음임]\.?\s*$", plain):
        return "sentence ending"
    if ";" in plain or "구간 없음" in plain or "가중" in plain or "보정기" in plain or "굵게" in plain or "읽는다" in plain or "비교하" in plain or "대비하" in plain or "출처" in plain or "전체 표" in plain or re.search(r"RR\d", plain) or "=" in plain:
        return "explanatory clause"
    if len(plain) > 45:
        return f"long ({len(plain)} chars)"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--baseline")
    ap.add_argument("--write-baseline")
    a = ap.parse_args()
    files = []
    for d in DIRS:
        files += sorted((LATEX / d).rglob("*.tex"))
    cols = ["N1", "N2", "N3", "N4", "N5", "N6", "V", "S", "ST", "C", "D"]
    counts = {}
    hits = []
    for f in files:
        raw = f.read_text(encoding="utf-8")
        rel = str(f.relative_to(LATEX))
        s = strip(raw)
        c = {k: 0 for k in cols}
        for i, line in enumerate(s.splitlines(), 1):
            for k in ("N1", "N2", "N3", "N4", "N5", "N6", "V", "S"):
                for m in re.finditer(PATTERNS[k], line):
                    c[k] += 1
                    hits.append((k, rel, i, line.strip()[:110]))
            if re.search(STATUS_KW, line) and re.search(DONE_VERB + "|" + NOTDONE_VERB, line):
                c["ST"] += 1
                hits.append(("ST", rel, i, line.strip()[:110]))
            if rel.startswith("chapters/01_introduction") and not rel.endswith("chapter.tex"):
                for m in re.finditer(r"(?<![A-Za-z\\{])\d[\d,{}.]*", line):
                    tok = m.group(0)
                    if re.fullmatch(r"20\d\d", tok) or re.fullmatch(r"1[56]\.\d+", tok):
                        continue
                    c["D"] += 1
                    hits.append(("D", rel, i, line.strip()[:110]))
        for line, cap, short in captions(raw):
            prob = caption_problem(cap)
            if short:
                prob = (prob + "; " if prob else "") + "has [short] form"
            if prob:
                c["C"] += 1
                hits.append(("C", rel, line, f"[{prob}] {cap[:90]}"))
        if any(c.values()):
            counts[rel] = c
    tot = {k: sum(v[k] for v in counts.values()) for k in cols}
    w = max(len(k) for k in counts) if counts else 10
    print("file".ljust(w) + " " + " ".join(k.rjust(4) for k in cols))
    for rel, c in counts.items():
        print(rel.ljust(w) + " " + " ".join(str(c[k]).rjust(4) for k in cols))
    print("TOTAL".ljust(w) + " " + " ".join(str(tot[k]).rjust(4) for k in cols))
    if a.baseline and Path(a.baseline).exists():
        base = json.loads(Path(a.baseline).read_text(encoding="utf-8"))["total"]
        print("baseline delta: " + " ".join(f"{k}:{tot[k] - base.get(k, 0):+d}" for k in cols))
    if a.write_baseline:
        Path(a.write_baseline).write_text(json.dumps({"total": tot, "files": counts}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"baseline written to {a.write_baseline}")
    if a.verbose:
        print()
        for k in cols:
            hk = [h for h in hits if h[0] == k]
            if not hk:
                continue
            print(f"== {k} ({len(hk)})")
            for _, rel, i, ex in hk:
                print(f"  {rel}:{i}: {ex}")
    print("\ncheck_style: report only (no gate). Read the ST/V/S hits in context before editing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
