"""Representation audit: do the fields in the JSON reach the model input, and correctly?

Four levels, one report (JSON + markdown):

  A. cache        every node / global feature over a sample of matches: share of zeros,
                  share of matches where it never changes, min / max  -> features that are
                  never populated or never vary
  B. tabular      every column of the corpus shards: constant share, zero share, and the
                  columns that are categorical identifiers stored as numbers
  C. model input  what the two learners actually see: the ToG decomposition keeps every
                  non-constant column; the CoG LightGBM run additionally drops
                  quasi-constant / within-fight-constant / correlated columns (its lists)
  D. alignment    end-to-end spot checks on real engagements: a frame-held feature in the
                  last bin equals the cache frame at or before the cutoff; event bins equal
                  raw event counts in the window; time_norm and frame_age_s are what they
                  claim; the label recomputed from the cache equals the shard label

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/audit_representation.py \\
        --shards D:/LOL_Project/fusion_2615/corpus_shards_v31_mevent \\
        --lgbm-run "D:/LOL_Project/runs_corpus_v3/v3_G13.7_D4264_R1600_B15/run_*" \\
        --output D:/LOL_Project/fusion_2615/features/representation_audit_v31.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

V3 = {"TF2_KILL_CLUSTER_GAP_MS": 13700, "CLUSTER_MAX_DIAMETER": 4264.0, "TF2_VALIDITY_RADIUS": 1600.0,
      "TF2_ENGAGE_PRE_KILL_MS": 15000, "FIGHT_HORIZON_SEC": 35, "FIGHT_CONTEXT_SEC": 30,
      "LABEL_TYPE": "market_event", "LABEL_TIE_POLICY": "drop",
      "FIGHT_INDEX_NUM_WORKERS": 1, "FIGHT_INDEX_CACHE_ENABLED": False, "DUMP_FIGHTS": False}
SUFFIXES = ("__last", "__mean", "__std", "__min", "__max", "__delta", "__slope")
ROLE_RE = re.compile(r"^(b|r)(TOP|JNG|MID|BOT|SUP)_(.+)$")
CATEGORICAL_BASES = {"champion_id", "champion_name_id", "summoner_spell_1_id", "summoner_spell_2_id",
                     "primary_style_id", "sub_style_id", "primary_rune_1", "primary_rune_2", "primary_rune_3",
                     "primary_rune_4", "sub_rune_1", "sub_rune_2", "stat_perk_offense", "stat_perk_flex",
                     "stat_perk_defense"} | {f"itemhash{i}" for i in range(16)} | {f"{t}_ban_{i}" for t in ("blue", "red") for i in range(5)}


def base_of(name: str) -> str:
    for s in SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return name


def family_of(base: str) -> str:
    m = ROLE_RE.match(base)
    if m:
        f = m.group(3)
        if f.startswith("itemhash"):
            return "player.item_hash"
        if f.startswith("cs_"):
            return "player.champion_stats"
        if f.startswith("ds_"):
            return "player.damage_stats"
        if f in ("champion_id", "champion_name_id", "summoner_spell_1_id", "summoner_spell_2_id"):
            return "player.identity"
        if f.startswith(("primary_", "sub_", "stat_perk")):
            return "player.runes"
        if f in ("has_baron", "has_elder", "baron_remain_norm", "elder_remain_norm", "ult_level_norm") or f.startswith("soul_"):
            return "player.status"
        return "player.snapshot"
    if base == "time_norm":
        return "global.time"
    if base.endswith("_ban_0") or "_ban_" in base:
        return "global.bans"
    if base.endswith("_t100") or base.endswith("_t200"):
        return "event.bins"
    if base.startswith(("pos_", "dist_", "near_", "zone_", "in_tower", "standoff", "mean_min", "d_standoff", "d_mean")):
        return "spatial"
    if base == "frame_age_s":
        return "freshness"
    return "global.team_diff"


# ------------------------------------------------------------------ A. cache
def audit_cache(mids, n_matches: int):
    from data.cache_io import load_match_cache
    from core.contract import NODE_IDX
    from core.config import GLOBAL_IDX
    node_names = sorted(NODE_IDX, key=NODE_IDX.get)
    glob_names = sorted(GLOBAL_IDX, key=GLOBAL_IDX.get)
    zeros = np.zeros(len(node_names)); total = 0
    const_matches = np.zeros(len(node_names)); n_m = 0
    vmin = np.full(len(node_names), np.inf); vmax = np.full(len(node_names), -np.inf)
    gz = np.zeros(len(glob_names)); gconst = np.zeros(len(glob_names)); gmin = np.full(len(glob_names), np.inf); gmax = np.full(len(glob_names), -np.inf); gtotal = 0
    for mid in mids[:n_matches]:
        pack = load_match_cache(mid)
        if not pack:
            continue
        nm = np.asarray(pack["node_minute"], dtype=np.float64)   # (T, 10, F)
        gm = np.asarray(pack["global_minute"], dtype=np.float64)
        n_m += 1
        flat = nm.reshape(-1, nm.shape[2])
        zeros += (flat == 0).sum(axis=0); total += flat.shape[0]
        const_matches += (nm.max(axis=(0, 1)) == nm.min(axis=(0, 1)))
        vmin = np.minimum(vmin, np.nanmin(flat, axis=0)); vmax = np.maximum(vmax, np.nanmax(flat, axis=0))
        gz += (gm == 0).sum(axis=0); gtotal += gm.shape[0]
        gconst += (gm.max(axis=0) == gm.min(axis=0))
        gmin = np.minimum(gmin, gm.min(axis=0)); gmax = np.maximum(gmax, gm.max(axis=0))
    node = {n: {"zero_share": float(zeros[i] / max(total, 1)), "const_match_share": float(const_matches[i] / max(n_m, 1)),
                "min": float(vmin[i]), "max": float(vmax[i])} for i, n in enumerate(node_names)}
    glob_ = {n: {"zero_share": float(gz[i] / max(gtotal, 1)), "const_match_share": float(gconst[i] / max(n_m, 1)),
                 "min": float(gmin[i]), "max": float(gmax[i])} for i, n in enumerate(glob_names)}
    return {"n_matches": n_m, "node": node, "global": glob_}


# ------------------------------------------------------------------ B. tabular
def audit_tabular(shard_dir: Path, max_shards: int = 4):
    names = json.load(open(shard_dir / "feature_names.json", encoding="utf-8"))["names"]
    paths = sorted(shard_dir.glob("shard_*.npz"))[:max_shards]
    X = np.concatenate([np.load(p, allow_pickle=True)["X"] for p in paths], axis=0)
    col_min, col_max = X.min(axis=0), X.max(axis=0)
    const = col_max == col_min
    zero = (X == 0).mean(axis=0)
    nan = np.isnan(X).mean(axis=0)
    fam = collections.defaultdict(lambda: {"cols": 0, "const": 0, "zero_ge_0.99": 0, "nan_any": 0})
    for j, n in enumerate(names):
        f = family_of(base_of(n))
        fam[f]["cols"] += 1; fam[f]["const"] += int(const[j]); fam[f]["zero_ge_0.99"] += int(zero[j] >= 0.99); fam[f]["nan_any"] += int(nan[j] > 0)
    const_bases = collections.Counter(base_of(n) for j, n in enumerate(names) if const[j])
    fully_const_bases = sorted(b for b, c in const_bases.items() if c == 7)
    cat_cols = [n for n in names if (ROLE_RE.sub(r"\3", base_of(n)) in CATEGORICAL_BASES) or (base_of(n) in CATEGORICAL_BASES)]
    return {"n_rows": int(X.shape[0]), "n_cols": int(X.shape[1]), "n_const_cols": int(const.sum()),
            "families": dict(fam), "fully_constant_bases": fully_const_bases, "n_fully_constant_bases": len(fully_const_bases),
            "categorical_as_numeric_cols": len(cat_cols), "categorical_bases": sorted({base_of(n) for n in cat_cols})[:12]}


# ------------------------------------------------------------------ C. model input
def audit_model_input(lgbm_run_glob: str | None):
    out = {}
    if not lgbm_run_glob:
        return out
    runs = sorted(glob.glob(lgbm_run_glob))
    if not runs:
        return {"error": f"no run matching {lgbm_run_glob}"}
    base = Path(runs[-1]) / "models" / "lgbm" / "baseline"
    for key in ("features_used", "features_dropped", "features_dropped_constant", "features_dropped_quasi_constant"):
        p = base / f"{key}.txt"
        if p.exists():
            lines = [l.strip() for l in open(p, encoding="utf-8", errors="replace") if l.strip()]
            fam = collections.Counter(family_of(base_of(l)) for l in lines)
            out[key] = {"n": len(lines), "by_family": dict(fam)}
    out["run"] = runs[-1]
    return out


# ------------------------------------------------------------------ D. alignment
def audit_alignment(shard_dir: Path, n_matches: int = 40):
    from core.config import cfg
    for k, v in V3.items():
        setattr(cfg, k, v)
    from core.contract import NODE_IDX
    from core.config import GLOBAL_IDX, EVENT_FEATURE_NAMES
    from data.cache_io import load_match_cache
    from data.index_split import build_fight_index
    from gameplay.pipeline import build_ms_sequence
    from gameplay.features import build_sequence_features
    from train.baseline import _ref_engage_ts, _ref_label_end_ts, _ref_first_kill_ts, _ref_last_kill_ts
    names = json.load(open(shard_dir / "feature_names.json", encoding="utf-8"))["names"]
    bases = [base_of(n) for n in names[: len(names) // 7 + 1] if n.endswith("__last")]
    bases = [base_of(n) for n in names if n.endswith("__last")]
    col = {b: i for i, b in enumerate(bases)}
    # shard labels keyed by (match, engage_ts)
    shard_y = {}
    for p in sorted(shard_dir.glob("shard_*.npz")):
        with np.load(p, allow_pickle=True) as b:
            for g, t, y in zip(b["groups"], b["engage_ts"], b["y"]):
                shard_y[(str(g), int(t))] = int(y)
    mids = sorted({k[0] for k in shard_y})[:n_matches]
    refs = build_fight_index(cache_match_ids=mids)
    hold_feats = ["level_norm", "totalGold_norm", "hp_pct", "cs_armor", "ds_totalDamageDoneToChampions", "alive"]
    slots = ["bTOP", "bJNG", "bMID", "bBOT", "bSUP", "rTOP", "rJNG", "rMID", "rBOT", "rSUP"]
    res = collections.Counter(); worst = {}
    ev_kill_idx = EVENT_FEATURE_NAMES.index("kills_t100") if "kills_t100" in EVENT_FEATURE_NAMES else None
    n_checked = 0
    for r in refs:
        pack = load_match_cache(r.match_id)
        if not pack:
            continue
        tm = pack["meta"]["team_map"]
        raw = build_ms_sequence(pack, tm, -1, engage_ts=_ref_engage_ts(r), label_end_ts=_ref_label_end_ts(r),
                                first_kill_ts=_ref_first_kill_ts(r), last_kill_ts=_ref_last_kill_ts(r))
        if not raw:
            continue
        feats = build_sequence_features(raw, tm, pack["meta"].get("role_slots", None), "full")
        xs = np.asarray(feats["x_seq"], dtype=np.float64)   # (L, 1015)
        tau = int(r.t_start_ts)
        ts = np.asarray(pack["minute_ts"], dtype=np.int64)
        i_last = int(np.searchsorted(ts, tau - 1, side="right") - 1)
        role_slots = {int(k): int(v) for k, v in (pack["meta"].get("role_slots") or {}).items()}
        slot_to_pid = {s: p for p, s in role_slots.items()}
        n_checked += 1
        # (i) frame-held features in the last bin == cache frame at or before the cutoff
        for s_i, s in enumerate(slots):
            pid = slot_to_pid.get(s_i)
            if pid is None:
                continue
            for f in hold_feats:
                b = f"{s}_{f}"
                if b not in col:
                    continue
                got = xs[-1, col[b]]
                exp = float(pack["node_minute"][i_last, pid - 1, NODE_IDX[f]])
                ok = abs(got - exp) <= 1e-4 * max(1.0, abs(exp))
                res["hold_ok" if ok else "hold_mismatch"] += 1
                if not ok and "hold" not in worst:
                    worst["hold"] = {"match": r.match_id, "feature": b, "got": got, "expected": exp, "tau": tau, "frame_idx": i_last}
        # (ii) event bins: blue kills in [tau-30s, tau) == sum of kills_t100 over bins
        if ev_kill_idx is not None and "kills_t100" in col:
            got = float(xs[:, col["kills_t100"]].sum())
            exp = sum(1 for e in pack["events"] if e.get("type") == "CHAMPION_KILL" and tm.get(str(e.get("killerId")), tm.get(e.get("killerId"), 0)) == 100
                      and (tau - 30000) <= int(e["timestamp"]) < tau)
            ok = abs(got - exp) < 1e-6
            res["event_ok" if ok else "event_mismatch"] += 1
            if not ok and "event" not in worst:
                worst["event"] = {"match": r.match_id, "got": got, "expected": exp, "tau": tau}
        # (iii) time_norm == (bin centre)/45 min ; frame_age == tau - last frame
        if "time_norm" in col:
            q = (tau - 30000) + 5 * 5000 + 2500
            exp = min(q / (45 * 60000.0), 1.0)
            ok = abs(xs[-1, col["time_norm"]] - exp) < 1e-5
            res["time_ok" if ok else "time_mismatch"] += 1
        age_exp = (tau - int(ts[i_last])) / 1000.0
        age_got = (tau - int(raw.get("global_snap_last_ts", -1))) / 1000.0
        res["age_ok" if abs(age_exp - age_got) < 1.0 else "age_mismatch"] += 1
        # (iv) label from this path == shard label
        key = (r.match_id, tau)
        if key in shard_y:
            y_here = int(feats["y"]) if feats.get("y") is not None else -1
            res["label_ok" if y_here == shard_y[key] else "label_mismatch"] += 1
            if y_here != shard_y[key] and "label" not in worst:
                worst["label"] = {"match": r.match_id, "tau": tau, "here": y_here, "shard": shard_y[key]}
        else:
            res["label_row_not_in_shard"] += 1
    return {"n_matches": len(mids), "n_engagements_checked": n_checked, "counts": dict(res), "first_mismatch": worst}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--lgbm-run", default=None)
    ap.add_argument("--n-cache-matches", type=int, default=400)
    ap.add_argument("--n-align-matches", type=int, default=40)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    from analysis.kill_pairs import list_cache_match_ids, sample_ids
    mids = sample_ids(list_cache_match_ids(), args.n_cache_matches, 7)
    out = {"A_cache": audit_cache(mids, args.n_cache_matches)}
    print("A done", f"{time.time() - t0:.0f}s", flush=True)
    out["B_tabular"] = audit_tabular(args.shards)
    print("B done", f"{time.time() - t0:.0f}s", flush=True)
    out["C_model_input"] = audit_model_input(args.lgbm_run)
    print("C done", flush=True)
    out["D_alignment"] = audit_alignment(args.shards, args.n_align_matches)
    print("D done", f"{time.time() - t0:.0f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    # ---- console summary
    A = out["A_cache"]
    dead = [n for n, v in A["node"].items() if v["zero_share"] > 0.999]
    never_vary = [n for n, v in A["node"].items() if v["const_match_share"] > 0.999 and n not in dead]
    print("\n[A] node features never populated (zero >99.9%):", dead)
    print("[A] node features that never vary within a match (not dead):", never_vary)
    print("[A] global:", {n: (round(v["zero_share"], 3), round(v["const_match_share"], 3)) for n, v in A["global"].items() if v["zero_share"] > 0.99 or v["const_match_share"] > 0.99})
    B = out["B_tabular"]
    print(f"\n[B] rows {B['n_rows']:,} cols {B['n_cols']} constant cols {B['n_const_cols']} fully-constant bases {B['n_fully_constant_bases']} categorical-as-numeric cols {B['categorical_as_numeric_cols']}")
    for f, v in sorted(B["families"].items()):
        print(f"    {f:24s} cols {v['cols']:5d} const {v['const']:5d} zero>=99% {v['zero_ge_0.99']:5d} nan {v['nan_any']}")
    print("    fully constant bases (first 40):", B["fully_constant_bases"][:40])
    C = out["C_model_input"]
    for k, v in C.items():
        if isinstance(v, dict) and "n" in v:
            print(f"[C] {k}: {v['n']} ", {kk: vv for kk, vv in sorted(v['by_family'].items())})
    D = out["D_alignment"]
    print("\n[D] alignment:", D["counts"], "| first mismatch:", D["first_mismatch"])
    print("wrote", args.output, f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
