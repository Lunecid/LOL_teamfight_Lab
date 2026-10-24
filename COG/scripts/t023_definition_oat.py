#!/usr/bin/env python3
"""E4 §7.4 — engagement-definition OAT (CACHE_MAIN).

Reference constants from the v3.3 preset; one-at-a-time alternatives.
Hash-fixed match subsample (budget, not power). Earliest-kill 1:1 anchors.

Covers the four boundary/gate constants (G, D, R, B) and the assembly
constants the manuscript records as unmeasured:
  I      TF2_INTERACTION_RADIUS          participation count -> scale class
  MR     CONTINUOUS_FIGHT_MERGE_RADIUS   adjacent-candidate merge radius
  MD     MAX_MERGED_FIGHT_DURATION_MS    duration cap (rejects late candidates)
  SHOPEX TF2_EXCLUDE_SHOP_INTERACTIONS   shop events as interaction participants

I decides the teamfight/skirmish split, so every arm also reports the
per-class counts and the class moves of the 1:1 anchors against the reference.

Also supports --census-only: reference arm alone over the full CACHE_MAIN
corpus (N_MATCHES=0) or a hash subsample, writing a separate class-census
JSON/MD (never overwrites the 5k OAT file).

Task: .ai/tasks/T023.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DOCS_JSON = REPO / "docs/SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.json"
DOCS_MD = REPO / "docs/SUPPLEMENTARY_E4_DEFINITION_OAT_20260921.md"
CENSUS_JSON = REPO / "docs/SUPPLEMENTARY_E4_CLASS_CENSUS_20260922.json"
CENSUS_MD = REPO / "docs/SUPPLEMENTARY_E4_CLASS_CENSUS_20260922.md"
PRED_T = REPO / "outputs/review_response_rr12_20260920/prediction_table.npz"
# Design says 20_000; wall-clock budget — use 5_000 with note (still OAT census).
# 0 = every match in CACHE_MAIN that has a *.meta.json (full-corpus census).
DEFAULT_N_MATCHES = 5000
REF = dict(G=13700, D=4264.0, R=1600.0, B=15000,
           I=3000.0, MR=2000.0, MD=60000, SHOPEX=False)

# The only cfg attributes apply_oat() writes. Names verified against core/config.py.
CFG_ATTR = {
    "G": "TF2_KILL_CLUSTER_GAP_MS",
    "D": "CLUSTER_MAX_DIAMETER",
    "R": "TF2_VALIDITY_RADIUS",
    "B": "TF2_ENGAGE_PRE_KILL_MS",
    "I": "TF2_INTERACTION_RADIUS",
    "MR": "CONTINUOUS_FIGHT_MERGE_RADIUS",
    "MD": "MAX_MERGED_FIGHT_DURATION_MS",
    "SHOPEX": "TF2_EXCLUDE_SHOP_INTERACTIONS",
}
CAST = {"G": int, "D": float, "R": float, "B": int,
        "I": float, "MR": float, "MD": int, "SHOPEX": bool}

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
    # interaction radius: sets participation count, hence the class split
    ("I_2000", {"I": 2000.0}),
    ("I_2500", {"I": 2500.0}),
    ("I_3500", {"I": 3500.0}),
    ("I_4264", {"I": 4264.0}),      # = D, the cluster's own diameter
    ("SHOPEX_on", {"SHOPEX": True}),
    # assembly constants that merge or reject candidates
    ("MR_1000", {"MR": 1000.0}),
    ("MR_3000", {"MR": 3000.0}),
    ("MD_45000", {"MD": 45000}),
    ("MD_90000", {"MD": 90000}),
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
    unknown = set(overrides) - set(CFG_ATTR)
    if unknown:
        raise SystemExit(f"unknown OAT knob(s): {sorted(unknown)}")
    for knob, attr in CFG_ATTR.items():
        if not hasattr(cfg, attr):
            raise SystemExit(f"cfg has no attribute {attr} (knob {knob})")
        setattr(cfg, attr, CAST[knob](overrides.get(knob, REF[knob])))


def current_params(cfg) -> dict:
    """Every knob as the detector will read it, after apply_oat()."""
    return {k: getattr(cfg, a) for k, a in CFG_ATTR.items()}


def changed_label(overrides: dict) -> str:
    return ", ".join(f"{k}={overrides[k]}" for k in sorted(overrides)) or "—"


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
        # (kill_key, n_min, start, last, cluster_blue, cluster_red)
        keys.append((k, n_min, s, L, cb, cr))
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


def list_cache_match_ids(cache_dir: Path) -> List[str]:
    """Every match_id with a *.meta.json under CACHE_MAIN."""
    ids: List[str] = []
    # scandir avoids building a huge Path list for ~210k matches × 3 files.
    with os.scandir(cache_dir) as it:
        for entry in it:
            name = entry.name
            if name.endswith(".meta.json") and entry.is_file():
                mid = name[: -len(".meta.json")]
                if mid:
                    ids.append(mid)
    ids.sort()
    return ids


def hash_first_pred_sample(n_matches: int) -> List[str]:
    pred = np.load(PRED_T, allow_pickle=True)
    matches = sorted(set(pred["match"].astype(str).tolist()))
    ranked = sorted(matches, key=lambda m: hashlib.sha256(m.encode()).hexdigest())
    return ranked[:n_matches]


def _share(n: int, total: int) -> Optional[float]:
    return (float(n) / float(total)) if total else None


def _n_min_bucket(n_min: int) -> str:
    if 1 <= n_min <= 5:
        return str(n_min)
    return "other"


def run_census(n_matches: int) -> int:
    """Reference-arm class census. Streams packs (no full-corpus pack dict)."""
    root = data_root()
    setup(root)
    import fc20260915_common as C
    import data.cache_io as cio
    from core.config import cfg
    from gameplay.fights import detect_fights

    cio.CACHE_DIR = Path(C.CACHE_MAIN)
    cache_dir = Path(cio.CACHE_DIR)

    if n_matches == 0:
        sample = list_cache_match_ids(cache_dir)
        sample_mode = "full_cache_meta"
    else:
        sample = hash_first_pred_sample(n_matches)
        sample_mode = "hash_first_pred_table"

    print(
        f"CENSUS-ONLY n_matches_request={n_matches} sample={len(sample)} "
        f"mode={sample_mode} CACHE={cache_dir}",
        flush=True,
    )

    # Verify CFG attrs / reference params before the long run.
    apply_oat(cfg, {})
    params = current_params(cfg)
    print(f"ref params verified: {params}", flush=True)
    expected = {k: CAST[k](v) for k, v in REF.items()}
    for k, v in expected.items():
        if params[k] != v:
            raise SystemExit(f"ref param mismatch {k}: got {params[k]!r} expected {v!r}")

    t0 = time.time()
    n_miss = 0
    n_loaded = 0
    n_detect_err = 0
    n_eng = 0
    cohort_counts: Counter = Counter()
    n_min_hist: Counter = Counter()
    joint_5x5: Counter = Counter()  # keys "b,r" for b,r in 1..5
    diagonal = Counter()  # 2v2 / 4v4 / 5v5 among engagements

    for i, mid in enumerate(sample):
        if i % 1000 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{i}/{len(sample)}] loaded={n_loaded} miss={n_miss} "
                f"eng={n_eng} elapsed_s={elapsed:.0f}",
                flush=True,
            )
        pack = cio.load_match_cache(mid)
        if pack is None:
            n_miss += 1
            continue
        n_loaded += 1
        try:
            keys = detect_match(pack, detect_fights, cfg)
        except Exception:
            n_detect_err += 1
            keys = []
        n_eng += len(keys)
        for _k, n_min, _s, _L, cb, cr in keys:
            cohort_counts[cohort_of(n_min)] += 1
            n_min_hist[_n_min_bucket(n_min)] += 1
            if 1 <= cb <= 5 and 1 <= cr <= 5:
                joint_5x5[f"{cb},{cr}"] += 1
            if cb == cr and cb in (2, 4, 5):
                diagonal[f"{cb}v{cr}"] += 1

    wall_s = float(time.time() - t0)
    cohort_order = ["T", "S", "P", "UNK"]
    cohort_block = {
        c: {"count": int(cohort_counts.get(c, 0)), "share": _share(int(cohort_counts.get(c, 0)), n_eng)}
        for c in cohort_order
        if c != "UNK" or cohort_counts.get(c, 0)
    }
    # Always include UNK key if present; keep T/S/P always.
    for c in ("T", "S", "P"):
        cohort_block.setdefault(c, {"count": 0, "share": _share(0, n_eng)})

    n_min_block = {}
    for b in ["1", "2", "3", "4", "5", "other"]:
        cnt = int(n_min_hist.get(b, 0))
        if b == "other" and cnt == 0:
            continue
        n_min_block[b] = {"count": cnt, "share": _share(cnt, n_eng)}

    joint_block = {
        k: {"count": int(v), "share": _share(int(v), n_eng)}
        for k, v in sorted(joint_5x5.items(), key=lambda kv: (int(kv[0].split(",")[0]), int(kv[0].split(",")[1])))
    }
    diag_block = {
        k: {"count": int(diagonal.get(k, 0)), "share": _share(int(diagonal.get(k, 0)), n_eng)}
        for k in ("2v2", "4v4", "5v5")
    }

    note = (
        "Census of detector scale class only (reference arm / empty overrides on v3.3). "
        "Not the exchange-value labeled subset. AUC/cut sensitivity not recomputed. "
        "T=min(cluster_blue,cluster_red)>=4; S=2..3; P=0..1 (via n_min). "
        "May differ slightly from stored corpus row counts (566,104 valid; T 113,901)."
    )
    doc = dict(
        schema="SUPPLEMENTARY_E4_CLASS_CENSUS_v1",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        task=".ai/tasks/T023.md",
        mode="census_only_reference_arm",
        sample_mode=sample_mode,
        n_matches_request=int(n_matches),
        n_matches_listed=len(sample),
        n_matches_loaded=n_loaded,
        n_miss=n_miss,
        n_detect_err=n_detect_err,
        n_engagements=n_eng,
        cohort_counts={c: cohort_block[c]["count"] for c in cohort_block},
        cohort_shares={c: cohort_block[c]["share"] for c in cohort_block},
        cohort=cohort_block,
        n_min_histogram=n_min_block,
        joint_cluster_blue_red_1to5=joint_block,
        diagonal_cited=diag_block,
        params=params,
        reference=REF,
        cache_path=str(cache_dir),
        wall_s=wall_s,
        note=note,
    )
    CENSUS_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    def pct(share: Optional[float]) -> str:
        return "—" if share is None else f"{100.0 * share:.2f}%"

    lines = [
        "# Supplementary E4 — Reference-arm class census",
        "",
        f"**generated:** {doc['generated_at_utc']}  ",
        f"**cache:** `{cache_dir}`  ",
        f"**sample_mode:** {sample_mode} (n_matches_request={n_matches})  ",
        f"**wall_s:** {wall_s:.1f}  ",
        "",
        note,
        "",
        "## Scale",
        "",
        f"| metric | value |",
        f"|---|---:|",
        f"| n_matches_listed | {len(sample)} |",
        f"| n_matches_loaded | {n_loaded} |",
        f"| n_miss | {n_miss} |",
        f"| n_detect_err | {n_detect_err} |",
        f"| n_engagements | {n_eng} |",
        "",
        "## Cohort (scale class)",
        "",
        "| class | count | share |",
        "|---|---:|---:|",
    ]
    for c in cohort_order:
        if c not in cohort_block:
            continue
        lines.append(
            f"| {c} | {cohort_block[c]['count']} | {pct(cohort_block[c]['share'])} |"
        )
    lines += [
        "",
        "## n_min histogram",
        "",
        "| n_min | count | share |",
        "|---|---:|---:|",
    ]
    for b, block in n_min_block.items():
        lines.append(f"| {b} | {block['count']} | {pct(block['share'])} |")
    lines += [
        "",
        "## Diagonal (blue==red among 1..5 joints recorded)",
        "",
        "| cell | count | share of engagements |",
        "|---|---:|---:|",
    ]
    for k in ("2v2", "4v4", "5v5"):
        lines.append(f"| {k} | {diag_block[k]['count']} | {pct(diag_block[k]['share'])} |")
    lines += [
        "",
        "## Params (reference arm)",
        "",
        "```",
        json.dumps(params, indent=2),
        "```",
        "",
    ]
    CENSUS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {CENSUS_JSON}", flush=True)
    print(f"wrote {CENSUS_MD}", flush=True)
    print(
        f"DONE eng={n_eng} T={cohort_counts.get('T', 0)} "
        f"S={cohort_counts.get('S', 0)} P={cohort_counts.get('P', 0)} "
        f"wall_s={wall_s:.1f}",
        flush=True,
    )
    return 0


def run_oat(n_matches: int) -> int:
    root = data_root()
    setup(root)
    import fc20260915_common as C
    import data.cache_io as cio
    from core.config import cfg
    from gameplay.fights import detect_fights

    cio.CACHE_DIR = Path(C.CACHE_MAIN)
    if n_matches == 0:
        raise SystemExit(
            "OAT mode refuses N_MATCHES=0 (would overwrite 5k OAT with a full-corpus "
            "multi-arm run). Use --census-only --n-matches 0 for the reference census."
        )
    sample = hash_first_pred_sample(n_matches)
    print(f"OAT n_matches={len(sample)} CACHE={cio.CACHE_DIR}", flush=True)

    # load packs once
    packs = {}
    n_miss = 0
    for i, mid in enumerate(sample):
        if i % 500 == 0:
            print(f"  load {i}/{len(sample)}", flush=True)
        pack = cio.load_match_cache(mid)
        if pack is None:
            n_miss += 1
            continue
        packs[mid] = pack
    print(f"loaded {len(packs)} miss={n_miss}", flush=True)

    results = {}
    t0 = time.time()
    for name, ov in ALTS:
        apply_oat(cfg, ov)
        params = current_params(cfg)
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
            for _k, n_min, _s, _L, _cb, _cr in keys:
                cohort_counts[cohort_of(n_min)] += 1
        results[name] = dict(
            params=params,
            changed=changed_label(ov),
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
        n_1to1 = 0
        cohort_move = defaultdict(int)
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
        "| Setting | changed | n_eng | T | S | pick | common | ref_only | alt_only | class moves |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, block in slim.items():
        cc = block["cohort_counts"]
        vs = comparisons.get(name, {})
        moves = vs.get("cohort_moves") or {}
        moves_s = ", ".join(f"{k} {v}" for k, v in sorted(moves.items())) or "—"
        lines.append(
            f"| {name} | {block['changed']} | {block['n_engagements']} | "
            f"{cc.get('T', 0)} | {cc.get('S', 0)} | {cc.get('P', 0)} | "
            f"{vs.get('n_common_anchors', '—')} | {vs.get('n_ref_only', '—')} | "
            f"{vs.get('n_alt_only', '—')} | {moves_s} |"
        )
    lines += ["", doc["reading"], ""]
    DOCS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DOCS_MD}", flush=True)
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E4 definition OAT / reference class census")
    p.add_argument(
        "--census-only",
        action="store_true",
        help="Run only the reference arm and write CLASS_CENSUS docs (not the 5k OAT file).",
    )
    p.add_argument(
        "--n-matches",
        type=int,
        default=DEFAULT_N_MATCHES,
        help=(
            "Match budget. Default 5000 (hash-first from pred table). "
            "0 = every *.meta.json under CACHE_MAIN (census-only)."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.census_only:
        return run_census(int(args.n_matches))
    return run_oat(int(args.n_matches))


if __name__ == "__main__":
    raise SystemExit(main())
