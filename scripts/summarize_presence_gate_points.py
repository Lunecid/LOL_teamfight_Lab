"""Summarise the presence-gate re-runs produced by ``scripts/run_presence_gate_points.py``.

For every point in ``<OUTPUT_ROOT>/runs_presence_gate/manifest.json`` the newest ``run_*``
directory is read: fight-index counts per patch, split sizes, LightGBM train/val/test AUC
with the bootstrap CI parsed from the baseline log, AP, Brier, AUC by phase, and the
positive rate.  When two points are given (``--a``/``--b``, default: the first two), the
test predictions are joined on the engagement identity (match id + first-kill time, i.e.
``t_start_ts + B``) so each model's AUC can be reported on the engagements both gates
accept and on the ones only that gate accepts.  Labels may still differ on the joined rows
because the label window is anchored on the cutoff, which moves with B.

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/summarize_presence_gate_points.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    # average ranks for ties
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _latest_run(point_dir: Path) -> Optional[Path]:
    runs = sorted([p for p in point_dir.glob("run_*") if p.is_dir()])
    return runs[-1] if runs else None


def _load(p: Path):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def _bootstrap_ci(log_path: Path) -> Dict[str, tuple]:
    out = {}
    if not log_path.exists():
        return out
    for line in open(log_path, encoding="utf-8", errors="replace"):
        m = re.search(r"\[LGBM\]\[BOOTSTRAP\] (val|test) AUC=([0-9.]+) 95% CI=\[([0-9.]+), ([0-9.]+)\]", line)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


def _read_preds(p: Path, b_ms: int):
    rows = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = f"{r['match_id']}|{int(r['t_start_ts']) + b_ms}"
            rows[key] = (int(r["y"]), float(r["p"]))
    return rows


def summarise_point(name: str, rec: dict) -> Optional[dict]:
    run = _latest_run(Path(rec["run_dir"]))
    if run is None:
        return None
    base = run / "models" / "lgbm" / "baseline"
    fi = _load(run / "meta" / "fight_index.json") or {}
    sp = _load(run / "meta" / "split.json") or {}
    rep = _load(base / "report.json") or {}
    mt = _load(base / "minute_report_test.json") or {}
    st = _load(base / "situation_report_test.json") or {}
    ci = _bootstrap_ci(base / "run.log")
    metrics = rep.get("metrics", {})
    return {"name": name, "R_u": rec["R_u"], "B_ms": rec["B_ms"], "run": str(run), "elapsed_s": rec.get("elapsed_s"),
            "n_fights": int(fi.get("n_fights", 0) or 0), "patch_counts": fi.get("patch_counts_fights"),
            "split": {k: sp.get(k) for k in ("n_train", "n_val", "n_test")},
            "auc": {k: (metrics.get(k) or {}).get("auc") for k in ("train", "val", "test")},
            "ci": ci, "test_overall": (mt.get("overall") or {}), "by_phase": (st.get("by_phase") or {}),
            "pred_test": base / "pred_test.csv"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default=None)
    ap.add_argument("--b", default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)
    out_root = Path(os.environ.get("LOL_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs")))
    manifest = _load(out_root / "runs_presence_gate" / "manifest.json")
    if not manifest:
        print("no manifest at", out_root / "runs_presence_gate")
        return 1
    points = {}
    for name, rec in manifest["points"].items():
        s = summarise_point(name, rec)
        if s:
            points[name] = s

    print("| point | R (u) | B (s) | engagements | per patch 15.14/15.15/15.16 | pos rate (test) | train AUC | val AUC [95% CI] | test AUC [95% CI] | test AP | Brier | elapsed |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in points.values():
        pc = s["patch_counts"] or {}
        if isinstance(pc, str):
            pc = {}
        pcs = " / ".join(f"{int(pc.get(p, 0)):,}" for p in ("15.14", "15.15", "15.16")) if pc else "-"
        ci_v, ci_t = s["ci"].get("val"), s["ci"].get("test")
        f = lambda c, a: (f"{a:.4f} [{c[1]:.4f}, {c[2]:.4f}]" if c else (f"{a:.4f}" if a is not None else "-"))
        to = s["test_overall"]
        print(f"| {s['name']} | {s['R_u']:.0f} | {s['B_ms'] / 1000:g} | {s['n_fights']:,} | {pcs} | "
              f"{to.get('pos_rate', float('nan')):.3f} | {s['auc']['train'] if s['auc']['train'] is None else round(s['auc']['train'], 4)} | "
              f"{f(ci_v, s['auc']['val'])} | {f(ci_t, s['auc']['test'])} | {to.get('ap', float('nan')):.4f} | "
              f"{to.get('brier', float('nan')):.4f} | {(s['elapsed_s'] or 0) / 60:.0f} min |")

    print("\n| point | phase | n | AUC | AP |")
    print("|---|---|---|---|---|")
    for s in points.values():
        for ph, v in (s["by_phase"] or {}).items():
            n = int((v.get("tp", 0) or 0) + (v.get("tn", 0) or 0) + (v.get("fp", 0) or 0) + (v.get("fn", 0) or 0))
            print(f"| {s['name']} | {ph} | {n:,} | {v.get('auc', float('nan')):.4f} | {v.get('ap', float('nan')):.4f} |")

    names = list(points)
    a = args.a or (names[0] if names else None)
    b = args.b or (names[1] if len(names) > 1 else None)
    joined = None
    if a and b and a in points and b in points and points[a]["pred_test"].exists() and points[b]["pred_test"].exists():
        pa = _read_preds(points[a]["pred_test"], int(points[a]["B_ms"]))
        pb = _read_preds(points[b]["pred_test"], int(points[b]["B_ms"]))
        common = sorted(set(pa) & set(pb))
        only_a = sorted(set(pa) - set(pb))
        only_b = sorted(set(pb) - set(pa))
        ya = np.array([pa[k][0] for k in common]); pa_c = np.array([pa[k][1] for k in common])
        yb = np.array([pb[k][0] for k in common]); pb_c = np.array([pb[k][1] for k in common])
        joined = {"a": a, "b": b, "n_test_a": len(pa), "n_test_b": len(pb), "n_common": len(common),
                  "n_only_a": len(only_a), "n_only_b": len(only_b),
                  "label_agreement_common": float(np.mean(ya == yb)) if common else None,
                  "auc_a_common": _auc(ya, pa_c) if common else None, "auc_b_common": _auc(yb, pb_c) if common else None,
                  "auc_a_only_a": _auc(np.array([pa[k][0] for k in only_a]), np.array([pa[k][1] for k in only_a])) if only_a else None,
                  "auc_b_only_b": _auc(np.array([pb[k][0] for k in only_b]), np.array([pb[k][1] for k in only_b])) if only_b else None}
        print(f"\nTest-set join on match id + first-kill time ({a} vs {b}):")
        print(f"- test rows: {a} {len(pa):,}, {b} {len(pb):,}; common {len(common):,}; only {a} {len(only_a):,}; only {b} {len(only_b):,}")
        if common:
            print(f"- label agreement on common rows: {joined['label_agreement_common']:.3f}")
            print(f"- AUC on common rows: {a} {joined['auc_a_common']:.4f}, {b} {joined['auc_b_common']:.4f}")
        if only_a:
            print(f"- AUC of {a} on rows only it accepts: {joined['auc_a_only_a']:.4f}")
        if only_b:
            print(f"- AUC of {b} on rows only it accepts: {joined['auc_b_only_b']:.4f}")

    if args.json_out:
        ser = {k: {kk: (str(vv) if isinstance(vv, Path) else vv) for kk, vv in v.items()} for k, v in points.items()}
        json.dump({"points": ser, "joined": joined}, open(args.json_out, "w", encoding="utf-8"), indent=1)
        print("\nwrote", args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
