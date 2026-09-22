#!/usr/bin/env python3
"""Blocking checks for the thesis (exit 1 on any failure). Only things a machine can verify:

  1. numbers_supp.tex / tables/gen / NUMBERS_SUPP.md identical to a fresh gen_supp.py run
  2. tables/gen/tab_status.tex identical to a fresh render of thesis/docs/STATUS.md
  3. main.log: no undefined references or citations (build first)
  4. labels: no duplicate labels, no dangling \\ref, every tab:/fig: label referenced
  5. headline macros unchanged (STATUS.md section 1)
  6. the four M-RQ question bodies unchanged (sha256 vs thesis/tools/mrq_hashes.json)

Reported, not gated: \\pendingauthor items, \\todo items, generated tables not \\input anywhere.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LATEX = ROOT / "thesis" / "latex"
TOOLS = ROOT / "thesis" / "tools"
SRC_DIRS = ["chapters", "tables", "figures", "frontmatter"]

HEADLINES = {  # STATUS.md section 1 (FROZEN); values as printed in config/numbers.tex
    "dBrierT": "-0.00373",
    "dBrierS": "-0.00335",
}


def tex_files():
    for d in SRC_DIRS:
        yield from sorted((LATEX / d).rglob("*.tex"))
    yield LATEX / "main.tex"


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def check_generated():
    bad = 0
    rc, out = run([sys.executable, str(TOOLS / "check_numbers_supp.py")])
    print(out)
    bad += rc != 0
    rc, out = run([sys.executable, str(TOOLS / "gen_status_table.py"), "--check"])
    print(out)
    bad += rc != 0
    rc, out = run([sys.executable, str(TOOLS / "gen_feature_census.py"), "--check"])
    print(out)
    bad += rc != 0
    return bad


def check_log():
    log = LATEX / "main.log"
    if not log.exists():
        print("main.log: MISSING (build with tectonic first)")
        return 1
    txt = log.read_text(encoding="utf-8", errors="replace")
    undef_ref = re.findall(r"Reference `([^']+)' on page \d+ undefined", txt)
    undef_cite = re.findall(r"Citation `([^']+)' on page \d+ undefined", txt)
    multi = re.findall(r"Label `([^']+)' multiply defined", txt)
    bad = 0
    for name, lst in (("undefined references", undef_ref), ("undefined citations", undef_cite), ("multiply defined labels", multi)):
        u = sorted(set(lst))
        if u:
            print(f"main.log: {len(u)} {name}: {', '.join(u[:12])}{' ...' if len(u) > 12 else ''}")
            bad += 1
    if not bad:
        print("main.log: OK (no undefined refs/citations)")
    return bad


def reachable_files():
    """Files reachable from main.tex through \\input (the document as compiled)."""
    seen = []
    stack = [LATEX / "main.tex"]
    while stack:
        f = stack.pop()
        if f in seen or not f.exists():
            continue
        seen.append(f)
        s = strip_comments(f.read_text(encoding="utf-8"))
        for m in re.finditer(r"\\input\{([^}]+)\}", s):
            q = LATEX / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
            stack.append(q)
    return seen


def check_labels():
    labels = {}
    refs = set()
    inputs = set()
    for f in reachable_files():
        s = strip_comments(f.read_text(encoding="utf-8"))
        for m in re.finditer(r"\\label\{([^}]+)\}", s):
            labels.setdefault(m.group(1), []).append(f.relative_to(LATEX))
        for m in re.finditer(r"\\(?:ref|eqref|autoref|pageref|cref|Cref)\{([^}]+)\}", s):
            for r in m.group(1).split(","):
                refs.add(r.strip())
        for m in re.finditer(r"\\input\{([^}]+)\}", s):
            inputs.add(m.group(1))
    bad = 0
    dup = {k: v for k, v in labels.items() if len(v) > 1}
    if dup:
        print(f"labels: {len(dup)} duplicate: " + ", ".join(f"{k} ({', '.join(map(str, v))})" for k, v in list(dup.items())[:8]))
        bad += 1
    dangling = sorted(r for r in refs if r not in labels)
    if dangling:
        print(f"labels: {len(dangling)} dangling refs: {', '.join(dangling[:12])}")
        bad += 1
    unref = sorted(k for k in labels if (k.startswith("tab:") or k.startswith("fig:")) and k not in refs)
    if unref:
        print(f"labels: {len(unref)} unreferenced float labels: {', '.join(unref[:12])}")
        bad += 1
    if not bad:
        print(f"labels: OK ({len(labels)} labels, {len(refs)} distinct refs, {len(reachable_files())} files reachable from main.tex)")
    # generated tables not input anywhere (report only)
    gen = sorted((LATEX / "tables" / "gen").glob("*.tex"))
    not_used = [g.name for g in gen if f"tables/gen/{g.stem}" not in inputs]
    if not_used:
        print(f"note: generated tables not \\input anywhere: {', '.join(not_used)}")
    return bad


def check_headlines():
    s = (LATEX / "config" / "numbers.tex").read_text(encoding="utf-8")
    bad = 0
    for mac, val in HEADLINES.items():
        m = re.search(r"\\newcommand\{\\" + mac + r"\}\{([^}]*)\}", s)
        got = m.group(1) if m else None
        if got != val:
            print(f"headline: \\{mac} = {got!r}, expected {val!r} (STATUS.md section 1)")
            bad += 1
    if not bad:
        print("headlines: OK")
    return bad


def mrq_bodies():
    bodies = {}
    for f in sorted((LATEX / "chapters" / "01_introduction").glob("*.tex")):
        s = strip_comments(f.read_text(encoding="utf-8"))
        for m in re.finditer(r"\\item\[(M-RQ\d)[^\]]*\]\s*(.*?)(?=\\item\[|\\end\{description\})", s, re.S):
            body = re.sub(r"\s+", " ", m.group(2)).strip()
            bodies[m.group(1)] = body
    return bodies


def check_mrq(write=False):
    hfile = TOOLS / "mrq_hashes.json"
    bodies = mrq_bodies()
    hashes = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in bodies.items()}
    if write:
        hfile.write_text(json.dumps({"hashes": hashes, "bodies": bodies}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"mrq: wrote {hfile.relative_to(ROOT)} ({len(hashes)} questions)")
        return 0
    if not hfile.exists():
        print("mrq: hash file missing (run with --write-mrq once)")
        return 1
    ref = json.loads(hfile.read_text(encoding="utf-8"))["hashes"]
    bad = 0
    for k in ("M-RQ1", "M-RQ2", "M-RQ3", "M-RQ4"):
        if k not in hashes:
            print(f"mrq: {k} not found in chapters/01_introduction")
            bad += 1
        elif hashes[k] != ref.get(k):
            print(f"mrq: {k} body changed (locked wording)")
            bad += 1
    if not bad:
        print("mrq: OK (four locked questions unchanged)")
    return bad


def report_markers():
    pend, todo = [], []
    for f in tex_files():
        s = strip_comments(f.read_text(encoding="utf-8"))
        for i, line in enumerate(s.splitlines(), 1):
            if "\\pendingauthor{" in line:
                pend.append(f"{f.relative_to(LATEX)}:{i}")
            if "\\todo{" in line:
                todo.append(f"{f.relative_to(LATEX)}:{i}")
    print(f"note: \\pendingauthor items = {len(pend)}: {', '.join(pend)}")
    print(f"note: \\todo items = {len(todo)}: {', '.join(todo)}")


def main() -> int:
    if "--write-mrq" in sys.argv:
        return check_mrq(write=True)
    bad = 0
    bad += check_generated()
    bad += check_log()
    bad += check_labels()
    bad += check_headlines()
    bad += check_mrq()
    report_markers()
    print("check_thesis: " + ("PASS" if bad == 0 else f"FAIL ({bad} gate(s))"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
