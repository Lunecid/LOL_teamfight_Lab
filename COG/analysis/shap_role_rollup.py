"""shap_role_rollup.py — RQ4 / Table VI: role-level and within-role signal-type
TreeSHAP attribution.

Reads the per-feature mean|SHAP| CSVs produced in-pipeline
(``models/lgbm/baseline/shap_mean_abs.csv``; computed on 2,000 stratified
validation instances per seed), maps each engineered tabular feature to its
player role (slot prefix) and signal type, and reports the three-seed-mean
role percentages and within-role signal composition (paper Table VI).

Mapping
-------
Role   : slot prefix bTOP/rTOP -> Top, bJNG/rJNG -> Jungle, ... (blue+red pooled).
Signal : attribute after the slot prefix and before the temporal __suffix:
           cs_*  -> Cbt (champion combat statistics)
           ds_*  -> Dmg (damage profile)
           curGold/totalGold/gps -> Gld   xp/level -> XP
           laneCS/jgCS -> CS              x/y -> Pos
           everything else (identity, runes, spells, hp/mp/alive/cc, buffs,
           soul, ult) -> Misc
Non-slotted features (global / spatial / event-count) are "team-level" and are
reported separately (excluded from the role percentages).

Usage
-----
    python -m analysis.shap_role_rollup --run_root <runs_dir> --seeds 7,42,123
    python -m analysis.shap_role_rollup --shap a.csv b.csv c.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

SLOT_TO_ROLE: Dict[str, str] = {
    "bTOP": "Top", "rTOP": "Top",
    "bJNG": "Jungle", "rJNG": "Jungle",
    "bMID": "Mid", "rMID": "Mid",
    "bBOT": "Bot", "rBOT": "Bot",
    "bSUP": "Support", "rSUP": "Support",
}
ROLES: List[str] = ["Top", "Jungle", "Mid", "Bot", "Support"]
SIGNALS: List[str] = ["Cbt", "Dmg", "Gld", "XP", "CS", "Pos", "Misc"]


def signal_of(attr: str) -> str:
    if attr.startswith("cs_"):
        return "Cbt"
    if attr.startswith("ds_"):
        return "Dmg"
    if attr in ("curGold_norm", "totalGold_norm", "gps_norm"):
        return "Gld"
    if attr in ("xp_norm", "level_norm"):
        return "XP"
    if attr in ("laneCS_norm", "jgCS_norm"):
        return "CS"
    if attr in ("x_norm", "y_norm"):
        return "Pos"
    return "Misc"


def parse_feature(name: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (role, signal) for a slotted feature, else (None, None)."""
    i = name.rfind("__")
    base = name[:i] if i >= 0 else name
    for slot, role in SLOT_TO_ROLE.items():
        if base.startswith(slot + "_"):
            return role, signal_of(base[len(slot) + 1:])
    return None, None


def rollup_one(path: str):
    role_shap: Dict[str, float] = defaultdict(float)
    role_sig: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    team_level = 0.0
    total = 0.0
    with open(path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                v = abs(float(row.get("mean_abs_shap", "")))
            except (TypeError, ValueError):
                continue
            total += v
            role, sig = parse_feature(str(row.get("feature", "")))
            if role is None:
                team_level += v
            else:
                role_shap[role] += v
                role_sig[role][sig] += v
    return role_shap, role_sig, sum(role_shap.values()), team_level, total


def _find_shap_per_seed(run_root: str, seeds: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for seed in seeds:
        cands = sorted(glob.glob(
            os.path.join(run_root, f"**/*seed={seed}/models/lgbm/baseline/shap_mean_abs.csv"),
            recursive=True,
        ))
        if cands:
            out[seed] = cands[-1]  # latest run for this seed
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Role-level / signal-type SHAP rollup (Table VI)")
    ap.add_argument("--run_root", type=str, default=None,
                    help="runs dir to search for one shap_mean_abs.csv per seed")
    ap.add_argument("--seeds", type=str, default="7,42,123")
    ap.add_argument("--shap", type=str, nargs="*", default=None,
                    help="explicit shap_mean_abs.csv paths (one per seed)")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args(argv)

    if args.shap:
        paths = list(args.shap)
    elif args.run_root:
        found = _find_shap_per_seed(args.run_root, [s.strip() for s in args.seeds.split(",")])
        paths = [found[s] for s in sorted(found)]
        for s, p in found.items():
            print(f"[seed {s}] {p}")
    else:
        ap.error("provide --run_root or --shap")
        return

    if not paths:
        print("[ERROR] no SHAP files found")
        return

    role_pct: Dict[str, List[float]] = defaultdict(list)
    sig_pct: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    team_pct: List[float] = []
    for p in paths:
        role_shap, role_sig, role_total, team_level, total = rollup_one(p)
        for role in ROLES:
            role_pct[role].append(100.0 * role_shap[role] / role_total if role_total else 0.0)
            rt = role_shap[role]
            for sig in SIGNALS:
                sig_pct[role][sig].append(100.0 * role_sig[role][sig] / rt if rt else 0.0)
        team_pct.append(100.0 * team_level / total if total else 0.0)

    def mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\nThree-seed mean over {len(paths)} SHAP file(s). Role % sums to 100 "
          f"(team-level excluded = {mean(team_pct):.1f}%).\n")
    print(f"{'Role':8s} {'%':>5}  " + " ".join(f"{s:>5}" for s in SIGNALS))
    order = sorted(ROLES, key=lambda r: -mean(role_pct[r]))
    table = {}
    for role in order:
        rp = mean(role_pct[role])
        sigs = {s: round(mean(sig_pct[role][s]), 1) for s in SIGNALS}
        print(f"{role:8s} {rp:5.1f}  " + " ".join(f"{sigs[s]:5.1f}" for s in SIGNALS))
        table[role] = {"pct": round(rp, 1), **sigs}

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"roles": table, "team_level_pct": round(mean(team_pct), 1)}, f, indent=2)
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
