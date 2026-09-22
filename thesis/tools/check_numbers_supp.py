#!/usr/bin/env python3
"""Blocking check: the committed numbers_supp.tex, tables/gen/*.tex and NUMBERS_SUPP.md must be
byte-identical to a fresh run of gen_supp.py against the JSON ledgers in docs/. Exit 1 on drift."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "thesis" / "tools"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(TOOLS / "gen_supp.py"), "--out", tmp], check=True, capture_output=True)
        tmp = Path(tmp)
        bad = 0
        pairs = [
            (tmp / "thesis/latex/config/numbers_supp.tex", ROOT / "thesis/latex/config/numbers_supp.tex"),
            (tmp / "thesis/docs/NUMBERS_SUPP.md", ROOT / "thesis/docs/NUMBERS_SUPP.md"),
        ]
        for p in sorted((tmp / "thesis/latex/tables/gen").glob("*.tex")):
            pairs.append((p, ROOT / "thesis/latex/tables/gen" / p.name))
        for fresh, committed in pairs:
            if not committed.exists():
                print(f"MISSING  {committed.relative_to(ROOT)}")
                bad += 1
                continue
            if fresh.read_bytes() != committed.read_bytes():
                print(f"DRIFT    {committed.relative_to(ROOT)}  (re-run thesis/tools/gen_supp.py)")
                bad += 1
        # stale generated files that the generator no longer produces
        for p in sorted((ROOT / "thesis/latex/tables/gen").glob("*.tex")):
            # tables produced by the other generators (gen_status_table.py, gen_feature_census.py)
            other = {"tab_status.tex", "tab_input_blocks.tex"}
            if not (tmp / "thesis/latex/tables/gen" / p.name).exists() and p.name not in other:
                print(f"STALE    {p.relative_to(ROOT)}  (not produced by gen_supp.py)")
                bad += 1
    print("numbers_supp: " + ("OK" if bad == 0 else f"{bad} problem(s)"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
