#!/usr/bin/env python3
"""RR5a material crosstabs + RR5b next-objective (180s) + RR6b horizon/peer stability polish.

Design: docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md §9–10
Locks: frozen fit85 V; correspondence not independent accuracy; 180s proposal window;
       eligible elite enums frozen below; no V/q retrain.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

BUNDLE = (
    REPO
    / "outputs"
    / "v_redesign_wave4_corrected_20260919"
    / "evaluators"
    / "A_MLP_expanded_evaluator.joblib"
)
LAB = REPO / "outputs" / "q_newv_fit85_20260920" / "labels"
OUT = REPO / "outputs" / "review_response_rr5_rr6b_20260920"
ROLE = "EXPLORATORY_RR5_RR6B_MEASUREMENT_PRIOR_TEST_EXPOSURE"

# DEV-frozen eligible elite types for RR5b (no gold weights)
ELIGIBLE_MONSTERS = (
    "BARON_NASHOR",
    "DRAGON",  # includes elemental + ELDER via monsterSubType
    "RIFTHERALD",
    "HORDE",
    "ATAKHAN",
)
NEXT_OBJ_WINDOW_MS = 180_000
HS = (60, 90, 120)


def _data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


def _setup(data_root: Path) -> None:
    wt = data_root / "worktrees" / "engagement-state-value"
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(data_root / "scripts"))
    sys.path.insert(0, str(wt))
    os.environ.setdefault(
        "LOL_OUTPUT_ROOT",
        str(data_root / "outputs" / "full_corpus_training_20260915" / "runtime"),
    )


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(np.asarray(g).astype(str), return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{float(x):.{nd}f}"


def scrub(o):
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items()}
    if isinstance(o, list):
        return [scrub(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def slot_diff(X, names, field):
    ix = {n: i for i, n in enumerate(names)}
    blue = [ix[f"participant_slot{s}_{field}"] for s in range(0, 5)]
    red = [ix[f"participant_slot{s}_{field}"] for s in range(5, 10)]
    return X[:, blue].sum(axis=1) - X[:, red].sum(axis=1)


def count_net(during, after, keys, blue, red):
    ki = {str(k): i for i, k in enumerate(keys)}
    C = during + after
    return C[:, ki[blue]].astype(float) - C[:, ki[red]].astype(float)


def crosstab_2x2(y_svi: np.ndarray, material: np.ndarray, g: np.ndarray) -> Dict[str, Any]:
    """Exact counts: SVI {0,1} × material sign {−1,0,+1}; weighted agree on decided only."""
    y = np.asarray(y_svi, int)
    s = np.sign(np.asarray(material, float)).astype(int)
    g = np.asarray(g).astype(str)
    # raw counts
    cells = {}
    for ys in (0, 1):
        for ms in (-1, 0, 1):
            cells[f"SVI{ys}_mat{ms}"] = int(((y == ys) & (s == ms)).sum())
    decided = s != 0
    tie_n = int((~decided).sum())
    n = int(len(y))
    # unweighted disagreement among decided
    y_pm = np.where(y == 1, 1, -1)
    if decided.any():
        disagree_uw = float(np.mean(y_pm[decided] != s[decided]))
        w = match_weights(g[decided])
        agree_w = float(np.average(y_pm[decided] == s[decided], weights=w))
    else:
        disagree_uw = float("nan")
        agree_w = float("nan")
    return dict(
        n=n,
        n_decided=int(decided.sum()),
        n_tie=tie_n,
        tie_share=float(tie_n / n) if n else float("nan"),
        cells=cells,
        agreement_rate_match_weighted_decided=agree_w,
        disagreement_rate_unweighted_decided=disagree_uw,
        note=(
            "Weighted agreement only on decided (material≠0). "
            "Unweighted disagreement reported separately — not complements of each other across ties."
        ),
    )


def material_axes(E, idx_e, names):
    keys = [str(k) for k in E["count_keys"]]
    during = E["during"][idx_e]
    after = E["after_h90"][idx_e]

    def net(b, r):
        return count_net(during, after, keys, b, r)

    epic = (
        net("baron_blue", "baron_red")
        + net("dragon_blue", "dragon_red")
        + net("elder_blue", "elder_red")
        + net("herald_blue", "herald_red")
        + net("horde_blue", "horde_red")
        + net("atakhan_blue", "atakhan_red")
        + net("soul_owned_blue", "soul_owned_red")
    )
    struct = net("tower_blue", "tower_red") + net("inhibitor_blue", "inhibitor_red")
    Xpre, Xpost = E["X_pre"][idx_e], E["X_post_h90"][idx_e]
    kill_diff = slot_diff(Xpost, names, "kills") - slot_diff(Xpre, names, "kills")
    alive_post = slot_diff(Xpost, names, "alive")
    # freshness: same snapshot?
    same_frame = E["post_snapshot_h90"][idx_e] == E["pre_snapshot"][idx_e]
    return dict(
        epic_net=epic,
        structure_net=struct,
        objective_net=epic + struct,
        kill_diff=kill_diff,
        alive_diff_post=alive_post,
        same_frame=same_frame.astype(bool),
        alive_note=(
            "alive_diff_post = sum(alive)_blue − sum(alive)_red at post snapshot (last frame / state builder). "
            f"same_frame share={float(same_frame.mean()):.3f} (pre_snapshot==post_snapshot_h90)."
        ),
    )


def classify_elite(e: dict) -> Optional[Tuple[str, str]]:
    """Return (family, detail) if eligible; else None."""
    if str(e.get("type", "")) != "ELITE_MONSTER_KILL":
        return None
    mt = str(e.get("monsterType", "") or "")
    if mt not in ELIGIBLE_MONSTERS:
        return None
    sub = str(e.get("monsterSubType", "") or "").upper()
    if mt == "DRAGON":
        if sub == "ELDER_DRAGON":
            return ("elder", sub)
        kind = sub[: -len("_DRAGON")] if sub.endswith("_DRAGON") else (sub or "OTHER")
        return ("dragon", kind)
    fam = {
        "BARON_NASHOR": "baron",
        "RIFTHERALD": "herald",
        "HORDE": "horde",
        "ATAKHAN": "atakhan",
    }[mt]
    return (fam, mt)


def team_of_elite(e: dict, team_map: Dict[int, int]) -> int:
    team = int(e.get("killerTeamId", 0) or 0)
    if team in (100, 200):
        return team
    kid = int(e.get("killerId", 0) or 0)
    return int(team_map.get(kid, 0))


def build_team_map(pack: dict) -> Dict[int, int]:
    """participantId -> 100/200 from pack if available."""
    tm = {}
    info = pack.get("info") or pack.get("meta") or {}
    # common layouts
    for p in pack.get("participants") or info.get("participants") or []:
        pid = int(p.get("participantId", 0) or 0)
        team = int(p.get("teamId", 0) or 0)
        if pid and team in (100, 200):
            tm[pid] = team
    return tm


def game_end_ms(pack: dict) -> Optional[int]:
    info = pack.get("info") or {}
    for key in ("gameDuration", "gameEndTimestamp"):
        if key in info and info[key] is not None:
            # gameDuration is seconds in some schemas
            v = int(info[key])
            if key == "gameDuration" and v < 10_000:
                return v * 1000
            if key == "gameEndTimestamp":
                # absolute — prefer duration if present
                continue
            return v
    # last event timestamp as soft end
    evs = pack.get("events") or []
    if not evs:
        return None
    return int(max(int(e.get("timestamp", 0) or 0) for e in evs))


def next_objective_label(
    events: List[dict],
    team_map: Dict[int, int],
    endpoint: int,
    game_end: Optional[int],
    window_ms: int = NEXT_OBJ_WINDOW_MS,
) -> Dict[str, Any]:
    """First eligible elite in (endpoint, endpoint+window]; no V/q reads."""
    t1 = endpoint + window_ms
    if game_end is not None and game_end <= endpoint:
        return dict(label="game_ended_before_objective", team=None, family=None, detail=None, ts=None)
    # censored if we cannot observe full window
    censored = game_end is not None and game_end < t1

    cands = []
    for e in events:
        ts = int(e.get("timestamp", -1) or -1)
        if not (endpoint < ts <= t1):
            continue
        fam = classify_elite(e)
        if fam is None:
            continue
        team = team_of_elite(e, team_map)
        cands.append((ts, team, fam[0], fam[1]))
    if not cands:
        if censored:
            return dict(label="observation_censored", team=None, family=None, detail=None, ts=None)
        if game_end is not None and game_end <= t1:
            # game ended in window with no elite
            return dict(label="none", team=None, family=None, detail=None, ts=None, game_end_in_window=True)
        return dict(label="none", team=None, family=None, detail=None, ts=None)

    cands.sort(key=lambda x: x[0])
    t0 = cands[0][0]
    same_t = [c for c in cands if c[0] == t0]
    teams = {c[1] for c in same_t}
    if 0 in teams or len(teams) > 1 or (100 in teams and 200 in teams):
        return dict(
            label="tie_ambiguous",
            team=None,
            family=same_t[0][2],
            detail=same_t[0][3],
            ts=t0,
            n_same_ts=len(same_t),
        )
    team = next(iter(teams))
    if team not in (100, 200):
        return dict(label="tie_ambiguous", team=None, family=same_t[0][2], detail=same_t[0][3], ts=t0)
    side = "Blue" if team == 100 else "Red"
    return dict(label=side, team=team, family=same_t[0][2], detail=same_t[0][3], ts=t0)


def horizon_block(E, idx_map, lab_keys, ev):
    """RR6b: common-support flips + endpoint identity."""
    from v_redesign_evaluator_bundle import predict_calibrated

    em = E["match"].astype(str)
    es = E["s"].astype(np.int64)
    pre_ok = E["pre_ok"] == 1
    base = np.array([(a, int(b)) in lab_keys for a, b in zip(em.tolist(), es.tolist())], dtype=bool)
    base &= pre_ok
    common = base.copy()
    for h in HS:
        common &= E[f"valid_h{h}"] == 1
    idx = np.where(common)[0]
    Xpre = E["X_pre"][idx]
    p0 = predict_calibrated(ev, Xpre)
    g = em[idx]
    endpoints = {h: E[f"endpoint_h{h}"][idx].astype(np.int64) for h in HS}
    dVs, Ys = {}, {}
    for h in HS:
        p1 = predict_calibrated(ev, E[f"X_post_h{h}"][idx])
        dV = p1 - p0
        dVs[h] = dV
        Ys[h] = (dV > 0).astype(np.int8)

    # endpoint identity
    ep_same = {}
    for a, b in ((60, 90), (90, 120), (60, 120)):
        same = endpoints[a] == endpoints[b]
        ep_same[f"h{a}_vs_h{b}"] = dict(
            share_same_endpoint=float(same.mean()),
            n_same=int(same.sum()),
            n=int(len(same)),
            mean_abs_L_diff_ms=float(np.mean(np.abs(endpoints[a] - endpoints[b]))),
            among_same_endpoint_svi_flip=float(np.mean(Ys[a][same] != Ys[b][same])) if same.any() else None,
            among_diff_endpoint_svi_flip=float(np.mean(Ys[a][~same] != Ys[b][~same])) if (~same).any() else None,
        )

    flips = {}
    for a, b in ((60, 90), (90, 120), (60, 120)):
        m = (np.sign(dVs[a]) != 0) & (np.sign(dVs[b]) != 0)
        corr = float(np.corrcoef(dVs[a], dVs[b])[0, 1]) if len(dVs[a]) > 2 else float("nan")
        flips[f"h{a}_vs_h{b}"] = dict(
            svi_flip_rate=float(np.mean(Ys[a] != Ys[b])),
            mean_abs_delta_diff=float(np.mean(np.abs(dVs[a] - dVs[b]))),
            sign_agree_nonzero=float(np.mean(np.sign(dVs[a])[m] == np.sign(dVs[b])[m])) if m.any() else None,
            corr_deltaV=corr,
        )

    # inclusion counts
    n_any = {}
    for h in HS:
        n_any[f"valid_h{h}"] = int((base & (E[f"valid_h{h}"] == 1)).sum())
    return dict(
        design="common_valid_h60_h90_h120_intersection",
        n_common=int(common.sum()),
        n_matches_common=int(len(np.unique(g))),
        inclusion=n_any,
        pairwise_svi=flips,
        endpoint_identity=ep_same,
        reading=(
            "High sign agree can reflect shared endpoints (short remaining games). "
            "Report same-endpoint share alongside flip rates. Not a retune target."
        ),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-matches-nextobj", type=int, default=None, help="debug cap for RR5b cache scan")
    ap.add_argument("--skip-nextobj", action="store_true")
    args = ap.parse_args(argv)

    if not BUNDLE.is_file():
        raise SystemExit(f"missing {BUNDLE}")
    if not (LAB / "TEST_h90.npz").is_file():
        raise SystemExit(f"missing {LAB / 'TEST_h90.npz'}")

    data_root = _data_root()
    _setup(data_root)
    import fc20260915_common as C
    import fc20260915_data as D
    import data.cache_io as cio
    from v_redesign_evaluator_bundle import load_evaluator, predict_calibrated

    cio.CACHE_DIR = Path(C.CACHE_MAIN)
    OUT.mkdir(parents=True, exist_ok=True)

    print("load TEST labels + engagements…", flush=True)
    lab = np.load(LAB / "TEST_h90.npz", allow_pickle=False)
    L = D.Layout(False)
    E = D.load_engagements(L, "MAIN", ["TEST"], states=True, counts=True)
    names = list(E["names"])
    e_key = {
        (a, int(b)): i
        for i, (a, b) in enumerate(zip(E["match"].astype(str).tolist(), E["s"].astype(np.int64).tolist()))
    }
    idx_e, idx_l = [], []
    for i, (a, b) in enumerate(zip(lab["match"].astype(str).tolist(), lab["s"].astype(np.int64).tolist())):
        j = e_key.get((a, int(b)))
        if j is not None:
            idx_l.append(i)
            idx_e.append(j)
    idx_l = np.asarray(idx_l, int)
    idx_e = np.asarray(idx_e, int)
    ok = lab["Y_SVI"][idx_l] >= 0
    if "missing_score" in lab.files:
        ok &= lab["missing_score"][idx_l] == 0
    ok &= E["pre_ok"][idx_e] == 1
    ok &= E["valid_h90"][idx_e] == 1
    idx_l, idx_e = idx_l[ok], idx_e[ok]

    y = lab["Y_SVI"][idx_l].astype(int)
    g = lab["match"][idx_l].astype(str)
    p_pre = lab["p_pre"][idx_l].astype(float)
    dV = lab["delta_V"][idx_l].astype(float)
    b40 = lab["B40"][idx_l].astype(bool) if "B40" in lab.files else (p_pre >= 0.4) & (p_pre <= 0.6)
    tmin = lab["s"][idx_l].astype(float) / 60000.0
    endpoint = E["endpoint_h90"][idx_e].astype(np.int64)
    s_arr = E["s"][idx_e].astype(np.int64)

    print("RR5a material crosstabs…", flush=True)
    axes = material_axes(E, idx_e, names)
    material_report = dict(
        n=int(len(y)),
        alive_note=axes.pop("alive_note"),
        same_frame_share=float(axes.pop("same_frame").mean()),
        axes={},
        reading=(
            "Correspondence with same-window material nets — not independent fight-winner accuracy. "
            "Material features overlap V inputs."
        ),
    )
    crosstab_rows = []
    for name in ("kill_diff", "epic_net", "structure_net", "objective_net", "alive_diff_post"):
        ct = crosstab_2x2(y, axes[name], g)
        ct_b40 = crosstab_2x2(y[b40], axes[name][b40], g[b40]) if b40.any() else dict(n=0)
        material_report["axes"][name] = dict(all_T=ct, B40=ct_b40)
        crosstab_rows.append(
            dict(
                axis=name,
                scope="all_T",
                n=ct["n"],
                n_decided=ct["n_decided"],
                tie_share=ct["tie_share"],
                agree_w=ct["agreement_rate_match_weighted_decided"],
                disagree_uw=ct["disagreement_rate_unweighted_decided"],
                **ct["cells"],
            )
        )
        if ct_b40.get("n"):
            crosstab_rows.append(
                dict(
                    axis=name,
                    scope="B40",
                    n=ct_b40["n"],
                    n_decided=ct_b40["n_decided"],
                    tie_share=ct_b40["tie_share"],
                    agree_w=ct_b40["agreement_rate_match_weighted_decided"],
                    disagree_uw=ct_b40["disagreement_rate_unweighted_decided"],
                    **ct_b40["cells"],
                )
            )

    # --- RR5b next objective ---
    nextobj = dict(skipped=True)
    if not args.skip_nextobj:
        print("RR5b next elite objective (endpoint, +180s]…", flush=True)
        labels_out = []
        by_match = defaultdict(list)
        for i in range(len(g)):
            by_match[g[i]].append(i)
        match_ids = sorted(by_match.keys())
        if args.max_matches_nextobj:
            match_ids = match_ids[: args.max_matches_nextobj]
        n_miss = n_fail = 0
        t0 = time.time()
        for mi, mid in enumerate(match_ids):
            if mi % 500 == 0:
                print(f"  nextobj matches {mi}/{len(match_ids)}", flush=True)
            try:
                pack = cio.load_match_cache(mid)
            except Exception:
                n_miss += 1
                for i in by_match[mid]:
                    labels_out.append(
                        dict(i=i, label="observation_censored", family=None, detail=None, ts=None, team=None)
                    )
                continue
            if not pack or "events" not in pack:
                n_fail += 1
                for i in by_match[mid]:
                    labels_out.append(
                        dict(i=i, label="observation_censored", family=None, detail=None, ts=None, team=None)
                    )
                continue
            tm = build_team_map(pack)
            gend = game_end_ms(pack)
            events = pack["events"]
            for i in by_match[mid]:
                lab_i = next_objective_label(events, tm, int(endpoint[i]), gend)
                labels_out.append(dict(i=i, **lab_i))
        # align
        labels_out.sort(key=lambda d: d["i"])
        lab_arr = [d["label"] for d in labels_out]
        fam_arr = [d.get("family") for d in labels_out]
        # distributions by SVI / B40
        def dist_table(mask, scope):
            labs = np.array(lab_arr, dtype=object)[mask]
            counts = {k: int((labs == k).sum()) for k in sorted(set(lab_arr))}
            n = int(mask.sum())
            return dict(scope=scope, n=n, counts=counts, shares={k: (v / n if n else float("nan")) for k, v in counts.items()})

        nextobj = dict(
            window_ms=NEXT_OBJ_WINDOW_MS,
            eligible_monsters=list(ELIGIBLE_MONSTERS),
            n_matches_scanned=len(match_ids),
            n_cache_miss=n_miss,
            n_pack_fail=n_fail,
            wall_s=time.time() - t0,
            all_T=dist_table(np.ones(len(y), bool), "all_T"),
            B40=dist_table(b40, "B40"),
            SVI_pos=dist_table(y == 1, "SVI_pos"),
            SVI_neg=dist_table(y == 0, "SVI_neg"),
            by_time={},
            reading=(
                "Convergent correspondence only — not causal fight effect; shares common causes with state. "
                "Do not convert none→Red. Labels ignore V/q."
            ),
        )
        for lo, hi, name in ((2.0, 10.0, "t_2_10"), (10.0, 20.0, "t_10_20"), (20.0, 30.0, "t_20_30"), (30.0, 1e9, "t_30_inf")):
            m = (tmin >= lo) & (tmin < hi)
            if m.any():
                nextobj["by_time"][name] = dist_table(m, name)
        # persist per-row labels
        np.savez_compressed(
            OUT / "next_objective_labels.npz",
            match=g,
            s=s_arr,
            endpoint=endpoint,
            Y_SVI=y,
            B40=b40.astype(np.int8),
            label=np.array(lab_arr, dtype=object),
            family=np.array(fam_arr, dtype=object),
        )
        print(f"  nextobj done in {nextobj['wall_s']:.1f}s miss={n_miss}", flush=True)

    # --- RR6b ---
    print("RR6b horizon / endpoint identity…", flush=True)
    ev = load_evaluator(BUNDLE)
    lab_keys = set(zip(g.tolist(), s_arr.tolist()))
    # use full engagement keys for horizon common support
    lab_full = np.load(LAB / "TEST_h90.npz", allow_pickle=False)
    lab_keys_full = set(
        zip(lab_full["match"].astype(str).tolist(), lab_full["s"].astype(np.int64).tolist())
    )
    rr6b = horizon_block(E, idx_e, lab_keys_full, ev)

    # peer from existing verify if present
    verify_path = REPO / "outputs" / "newv_engagement_value_verify_20260920" / "results.json"
    peers = None
    if verify_path.is_file():
        peers = json.loads(verify_path.read_text(encoding="utf-8")).get("peer_sign_agree")

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        epistemic=ROLE,
        design="docs/REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md §9–10",
        bundle_sha16=sha16(BUNDLE),
        RR5a_material=material_report,
        RR5b_next_objective=nextobj,
        RR6b_horizon=rr6b,
        peer_sign_agree_reused=peers,
    )
    (OUT / "rr5_rr6b_results.json").write_text(json.dumps(scrub(payload), indent=2) + "\n", encoding="utf-8")

    # CSVs
    with (OUT / "material_crosstab.csv").open("w", newline="", encoding="utf-8") as f:
        if crosstab_rows:
            w = csv.DictWriter(f, fieldnames=list(crosstab_rows[0].keys()))
            w.writeheader()
            w.writerows(crosstab_rows)

    next_rows = []
    if not nextobj.get("skipped"):
        for scope_key in ("all_T", "B40", "SVI_pos", "SVI_neg"):
            block = nextobj[scope_key]
            for lab_name, cnt in block["counts"].items():
                next_rows.append(
                    dict(scope=scope_key, label=lab_name, n=cnt, share=block["shares"][lab_name], N=block["n"])
                )
        with (OUT / "next_objective.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["scope", "label", "n", "share", "N"])
            w.writeheader()
            w.writerows(next_rows)

    stab_rows = []
    for k, v in rr6b["pairwise_svi"].items():
        row = dict(pair=k, **v, **{f"ep_{kk}": vv for kk, vv in rr6b["endpoint_identity"][k].items()})
        stab_rows.append(row)
    with (OUT / "stability_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stab_rows[0].keys()))
        w.writeheader()
        w.writerows(stab_rows)

    # markdown
    md = [
        "# RR5 + RR6b — material correspondence, next objective, horizon stability",
        "",
        f"Generated: {payload['generated']}",
        f"**Bundle sha16:** `{payload['bundle_sha16']}`",
        "",
        "## RR5a — material correspondence (not independent accuracy)",
        "",
        material_report["reading"],
        "",
        f"- alive: {material_report['alive_note']}",
        "",
        "| Axis | scope | n_decided | tie share | agree (match-wtd) | disagree (UW decided) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in crosstab_rows:
        md.append(
            f"| {r['axis']} | {r['scope']} | {r['n_decided']} | {fmt(r['tie_share'],3)} | "
            f"{fmt(r['agree_w'],3)} | {fmt(r['disagree_uw'],3)} |"
        )
    if not nextobj.get("skipped"):
        md += [
            "",
            "## RR5b — first elite objective in (endpoint, endpoint+180s]",
            "",
            f"Eligible: `{ELIGIBLE_MONSTERS}`. Labels ignore V/q. none ≠ Red.",
            "",
            "| Scope | none | Blue | Red | tie_ambiguous | censored / ended | N |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for sk in ("all_T", "B40", "SVI_pos", "SVI_neg"):
            c = nextobj[sk]["counts"]
            N = nextobj[sk]["n"]
            md.append(
                f"| {sk} | {c.get('none',0)} | {c.get('Blue',0)} | {c.get('Red',0)} | "
                f"{c.get('tie_ambiguous',0)} | {c.get('observation_censored',0)+c.get('game_ended_before_objective',0)} | {N} |"
            )
        md.append("")
        md.append(nextobj["reading"])
    md += [
        "",
        "## RR6b — horizon stability + endpoint identity",
        "",
        rr6b["reading"],
        "",
        f"- common valid n={rr6b['n_common']:,}; inclusion `{json.dumps(rr6b['inclusion'])}`",
        "",
        "| Pair | SVI flip | sign agree | corr(ΔV) | same endpoint share | flip\\|same-ep | flip\\|diff-ep |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k, v in rr6b["pairwise_svi"].items():
        ep = rr6b["endpoint_identity"][k]
        md.append(
            f"| {k} | {fmt(v['svi_flip_rate'],3)} | {fmt(v.get('sign_agree_nonzero'),3)} | {fmt(v.get('corr_deltaV'))} | "
            f"{fmt(ep['share_same_endpoint'],3)} | {fmt(ep.get('among_same_endpoint_svi_flip'),3)} | "
            f"{fmt(ep.get('among_diff_endpoint_svi_flip'),3)} |"
        )
    if peers:
        md += ["", "### Peer sign agree (reused from newv verify)", ""]
        for k, v in peers.items():
            if k.startswith("MLP_vs_") and not k.endswith("_n_compared") and not k.endswith("_error"):
                md.append(f"- {k}: {fmt(v, 3) if isinstance(v, float) else v}")
    md += ["", f"Artifacts: `{OUT.relative_to(REPO).as_posix()}/`", ""]
    doc = REPO / "docs" / "REVIEW_RESPONSE_RR5_RR6B_20260920.md"
    doc.write_text("\n".join(md), encoding="utf-8")
    (OUT / "REVIEW_RESPONSE_RR5_RR6B_20260920.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT / "rr5_rr6b_results.json", doc, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
