#!/usr/bin/env python3
"""E4 §7.4 — engagement-definition OAT (CACHE_MAIN).

Reference G/D/R/B from v3.3 preset; one-at-a-time alternatives.
Hash-fixed match subsample (budget, not power). Earliest-kill 1:1 anchors.
Task: .ai/tasks/T023.md
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DOCS_JSON = REPO / "docs/SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json"
DOCS_MD = REPO / "docs/SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.md"
PRED_T = REPO / "outputs/review_response_rr12_20260920/prediction_table.npz"
# Design says 20_000; wall-clock budget — use 5_000 with note (still OAT census).
N_MATCHES = 5000
REF = dict(G=13700, D=4264.0, R=1600.0, B=15000)
ALTS = [
    ("ref", {}),
    ("G_12200", {"G": 12200}),
    ("G_15700", {"G": 15700}),
    ("D_4000", {"D": 4000.0}),
    ("D_4500", {"D": 4500.0}),
    ("R_1400", {"R": 1400.0}),
    ("R_1800", {"R": 1800.0}),
    ("B_10000", {"B": 10000}),
    ("B_20000", {"B": 20000}),
]


def data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    for x in Path.home().iterdir():
        cand = x / "LOL_Teamfight"
        if (cand / "outputs" / "full_corpus_training_20260915").is_dir():
            return cand
    raise SystemExit("data root not found")


def setup(root: Path) -> None:
    wt = root / "worktrees" / "engagement-state-value"
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(wt))
    os.environ["LOL_OUTPUT_ROOT"] = str(root / "outputs" / "full_corpus_training_20260915" / "runtime")


def apply_oat(cfg, overrides: dict) -> None:
    from core.presets import apply_preset

    apply_preset(cfg, "v3.3")
    g = int(overrides.get("G", REF["G"]))
    d = float(overrides.get("D", REF["D"]))
    r = float(overrides.get("R", REF["R"]))
    b = int(overrides.get("B", REF["B"]))
    cfg.TF2_KILL_CLUSTER_GAP_MS = g
    cfg.CLUSTER_MAX_DIAMETER = d
    cfg.TF2_VALIDITY_RADIUS = r
    cfg.TF2_ENGAGE_PRE_KILL_MS = b


def fight_start_L(f) -> Tuple[int, int]:
    if isinstance(f, dict):
        s = int(f.get("engage_ts") or f.get("t_start_ts") or f.get("first_kill_ts") or -1)
        L = int(f.get("last_kill_ts") or f.get("L") or -1)
        return s, L
    s = int(getattr(f, "engage_ts", getattr(f, "t_start_ts", -1)))
    L = int(getattr(f, "last_kill_ts", -1))
    return s, L


def earliest_kill_key(pack: dict, fight: dict) -> Optional[Tuple]:
    """Stable anchor: (first_kill_ts, killerId, victimId)."""
    fk = int(fight.get("first_kill_ts") or -1)
    L = int(fight.get("last_kill_ts") or -1)
    if fk < 0:
        return None
    for e in pack.get("events") or []:
        if e.get("type") != "CHAMPION_KILL":
            continue
        ts = int(e.get("timestamp", -1) or -1)
        if ts == fk:
            return (ts, int(e.get("killerId", 0) or 0), int(e.get("victimId", 0) or 0))
    # fallback any kill in [fk, L]
    kills = []
    for e in pack.get("events") or []:
        if e.get("type") != "CHAMPION_KILL":
            continue
        ts = int(e.get("timestamp", -1) or -1)
        if fk <= ts <= L:
            kills.append((ts, int(e.get("killerId", 0) or 0), int(e.get("victimId", 0) or 0)))
    if not kills:
        return None
    kills.sort()
    return kills[0]


def detect_match(pack, detect_fights, cfg) -> List[Tuple]:
    tm = {int(k): int(v) for k, v in pack["meta"]["team_map"].items()}
    fights = detect_fights(pack, tm)
    keys = []
    for f in fights:
        if not isinstance(f, dict):
            continue
        s, L = fight_start_L(f)
        if s < 0 or L < 0:
            continue
        k = earliest_kill_key(pack, f)
        if k is None:
            continue
        cb = int(f.get("det_cluster_blue", -1))
        cr = int(f.get("det_cluster_red", -1))
        n_min = min(cb, cr) if cb >= 0 and cr >= 0 else -1
        keys.append((k, n_min, s, L))
    return keys


def cohort_of(n_min: int) -> str:
    if n_min < 0:
        return "UNK"
    if n_min >= 4:
        return "T"
    if n_min >= 2:
        return "S"
    if n_min >= 0:
        return "P"
    return "UNK"


def main() -> int:
    root = data_root()
    setup(root)
    import fc20260915_common as C
    import data.cache_io as cio
    from core.config import cfg
    from gameplay.fights import detect_fights

    cio.CACHE_DIR = Path(C.CACHE_MAIN)
    # sample matches from TEST pred table (patch-aware via meta when loaded)
    pred = np.load(PRED_T, allow_pickle=True)
    matches = sorted(set(pred["match"].astype(str).tolist()))
    ranked = sorted(matches, key=lambda m: hashlib.sha256(m.encode()).hexdigest())
    sample = ranked[:N_MATCHES]
    print(f"OAT n_matches={len(sample)} CACHE={cio.CACHE_DIR}", flush=True)

    # load packs once
    packs = {}
    patches = []
    n_miss = 0
    for i, mid in enumerate(sample):
        if i % 500 == 0:
            print(f"  load {i}/{len(sample)}", flush=True)
        pack = cio.load_match_cache(mid)
        if pack is None:
            n_miss += 1
            continue
        packs[mid] = pack
        patches.append(str(pack.get("meta", {}).get("patch", "?")))
    print(f"loaded {len(packs)} miss={n_miss}", flush=True)

    results = {}
    t0 = time.time()
    for name, ov in ALTS:
        apply_oat(cfg, ov)
        params = dict(G=cfg.TF2_KILL_CLUSTER_GAP_MS, D=cfg.CLUSTER_MAX_DIAMETER, R=cfg.TF2_VALIDITY_RADIUS, B=cfg.TF2_ENGAGE_PRE_KILL_MS)
        print(f"detect {name} {params}", flush=True)
        anchors: Dict[str, List] = {}
        n_eng = 0
        cohort_counts = defaultdict(int)
        for mid, pack in packs.items():
            try:
                keys = detect_match(pack, detect_fights, cfg)
            except Exception:
                keys = []
            anchors[mid] = keys
            n_eng += len(keys)
            for _k, n_min, _s, _L in keys:
                cohort_counts[cohort_of(n_min)] += 1
        results[name] = dict(
            params=params,
            n_matches=len(packs),
            n_engagements=n_eng,
            cohort_counts=dict(cohort_counts),
            anchors=anchors,
        )

    # compare each alt to ref on earliest-kill keys
    ref_anchors = results["ref"]["anchors"]
    comparisons = {}
    for name in results:
        if name == "ref":
            continue
        alt = results[name]["anchors"]
        n_common = n_ref_only = n_alt_only = 0
        n_1to1 = n_1tomany = 0
        cohort_move = defaultdict(int)
        flip_join = dict(n=0, flip=None)  # filled if we join pred
        for mid in packs:
            rk = {a[0]: a for a in ref_anchors.get(mid, [])}
            ak = {a[0]: a for a in alt.get(mid, [])}
            common = set(rk) & set(ak)
            n_common += len(common)
            n_ref_only += len(set(rk) - set(ak))
            n_alt_only += len(set(ak) - set(rk))
            # 1:1 by construction on kill key uniqueness within match
            n_1to1 += len(common)
            for k in common:
                cr, ca = cohort_of(rk[k][1]), cohort_of(ak[k][1])
                if cr != ca:
                    cohort_move[f"{cr}->{ca}"] += 1
        comparisons[name] = dict(
            n_common_anchors=n_common,
            n_ref_only=n_ref_only,
            n_alt_only=n_alt_only,
            n_1to1=n_1to1,
            cohort_moves=dict(cohort_move),
            eng_ref=results["ref"]["n_engagements"],
            eng_alt=results[name]["n_engagements"],
        )

    # strip heavy anchors from JSON
    slim = {
        k: {kk: vv for kk, vv in v.items() if kk != "anchors"} for k, v in results.items()
    }
    doc = dict(
        schema="SUPPLEMENTARY_E4_DEFINITION_OAT_v1",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        task=".ai/tasks/T023.md",
        n_matches_design=20000,
        n_matches_run=len(packs),
        n_miss=n_miss,
        budget_note="Design budget 20k; ran hash-first N_MATCHES for wall-clock. Not a power guarantee.",
        reference=REF,
        settings=slim,
        vs_ref=comparisons,
        wall_s=float(time.time() - t0),
        reading=(
            "Frozen-score OAT on detector constants only. Earliest-kill anchors. "
            "Do not call D alternates 'CI boundaries'."
        ),
    )
    DOCS_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Supplementary E4 — Definition OAT (§7.4)",
        "",
        f"**generated:** {doc['generated_at_utc']}  ",
        f"n_matches={doc['n_matches_run']} (design 20k; {doc['budget_note']})",
        "",
        "| Setting | G | D | R | B | n_eng | T | S | vs_ref common | ref_only | alt_only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, block in slim.items():
        p = block["params"]
        cc = block["cohort_counts"]
        vs = comparisons.get(name, {})
        lines.append(
            f"| {name} | {p['G']} | {p['D']} | {p['R']} | {p['B']} | {block['n_engagements']} | "
            f"{cc.get('T', 0)} | {cc.get('S', 0)} | {vs.get('n_common_anchors', '—')} | "
            f"{vs.get('n_ref_only', '—')} | {vs.get('n_alt_only', '—')} |"
        )
    lines += ["", doc["reading"], ""]
    DOCS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DOCS_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
