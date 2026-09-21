#!/usr/bin/env python3
"""Phase-2: material-axis agreement + kill-less quiet-window ΔV reference (smoke).

Material axis (full MAIN_TEST teamfights T)
------------------------------------------
Compare sign(ΔV) / ΔV to externally readable window outcomes that are NOT the
V model output: objective/structure nets from stored event counts, and
kill / alive differentials from pre/post StateV2 matrices in extract chunks.

Quiet reference (smoke)
-----------------------
Type-B windows: fixed horizon H=90 s with no CHAMPION_KILL in [t, t+H], on a
hash-sampled subset of MAIN_TEST matches. Score frozen V at t and t+H-1.
Compare ΔV distributions to teamfight rows in the same (time-band × p_pre-bin)
strata. Not a causal control — a reference for background drift.

Exploratory; prior TEST exposure. Writes under outputs/reviewer_response_phase2_*.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _default_data_root() -> Path:
    for p in (Path.home() / "Documents" / "LOL_Teamfight", Path.home() / "문서" / "LOL_Teamfight"):
        if (p / "outputs" / "full_corpus_training_20260915").is_dir():
            return p
    return Path.home() / "문서" / "LOL_Teamfight"


DEFAULT_DATA = _default_data_root()
DEFAULT_OUT = REPO / "outputs" / "reviewer_response_phase2_20260919"
H_MS = 90_000
TIME_BANDS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 1e9))
P_EDGES = np.array([0.0, 0.2, 0.35, 0.45, 0.55, 0.65, 0.8, 1.01])


def _setup_paths(data_root: Path) -> None:
    wt = data_root / "worktrees" / "engagement-state-value"
    scripts = data_root / "scripts"
    # worktree gameplay (state_value_v2) must precede the workspace package
    for p in (str(scripts), str(wt)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(g, return_inverse=True, return_counts=True)
    return (1.0 / c[inv]).astype(np.float64)


def agreement(y: np.ndarray, s: np.ndarray, g: np.ndarray) -> Dict[str, Any]:
    """y in {0,1}, s in {-1,0,1}; agreement on non-zero s (and optionally all)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(int)
    g = np.asarray(g)
    out: Dict[str, Any] = {}
    decided = s != 0
    if decided.any():
        w = match_weights(g[decided])
        agree = ((y[decided] == 1) & (s[decided] == 1)) | ((y[decided] == 0) & (s[decided] == -1))
        out["n_decided"] = int(decided.sum())
        out["matches_decided"] = int(len(np.unique(g[decided])))
        out["agreement_rate"] = float(np.average(agree, weights=w))
        out["sign_pos_rate"] = float(np.average(s[decided] == 1, weights=w))
        out["y_pos_rate_decided"] = float(np.average(y[decided], weights=w))
    else:
        out.update(n_decided=0, matches_decided=0, agreement_rate=None,
                   sign_pos_rate=None, y_pos_rate_decided=None)
    out["n_tie_or_zero"] = int((~decided).sum())
    out["tie_share"] = float((~decided).mean())
    return out


def spearman_safe(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    from scipy.stats import spearmanr
    r, _ = spearmanr(a, b)
    return None if not np.isfinite(r) else float(r)


# ------------------------------------------------------------------ material
def material_from_labels(data_root: Path) -> Dict[str, Any]:
    lab = np.load(data_root / "outputs" / "full_corpus_training_20260915" / "labels" / "MAIN_TEST_labels.npz",
                  allow_pickle=False)
    coh = np.load(data_root / "outputs" / "cohort_role_training_20260915" / "cohorts" / "MAIN_TEST_cohort.npz",
                  allow_pickle=False)
    m = (coh["cohort"] == 1) & (lab["valid_h90"] == 1)
    keys = [str(k) for k in lab["count_keys"]]
    ki = {k: i for i, k in enumerate(keys)}
    C = lab["during_counts"][m] + lab["after_counts_h90"][m]
    y = lab["Y_h90"][m].astype(int)
    delta = lab["delta_h90"][m].astype(float)
    p_pre = lab["p_pre"][m].astype(float)
    g = lab["match"][m]
    t_min = lab["s"][m].astype(float) / 60000.0

    def net(blue: str, red: str) -> np.ndarray:
        return C[:, ki[blue]].astype(float) - C[:, ki[red]].astype(float)

    epic = (net("baron_blue", "baron_red") + net("dragon_blue", "dragon_red")
            + net("elder_blue", "elder_red") + net("herald_blue", "herald_red")
            + net("horde_blue", "horde_red") + net("atakhan_blue", "atakhan_red")
            + net("soul_owned_blue", "soul_owned_red"))
    struct = net("tower_blue", "tower_red") + net("inhibitor_blue", "inhibitor_red")
    obj = epic + struct
    # plates have no team split in COUNT_KEYS

    def sgn(x: np.ndarray) -> np.ndarray:
        return np.sign(x).astype(int)

    axes = {
        "epic_net": epic,
        "structure_net": struct,
        "objective_net": obj,
    }
    report = {
        "n": int(m.sum()),
        "matches": int(len(np.unique(g))),
        "axes": {},
        "note": "Event counts are window totals (during engagement + after to h90 endpoint). "
                "champion_kill has no team split in COUNT_KEYS; combat axis uses extract states.",
    }
    for name, val in axes.items():
        s = sgn(val)
        block = agreement(y, s, g)
        block["spearman_vs_delta"] = spearman_safe(val, delta)
        block["mean_abs_net"] = float(np.mean(np.abs(val)))
        block["any_nonzero_share"] = float(np.mean(val != 0))
        # stratified by |delta|
        for tag, mm in (
            ("abs_dV_le_0.5pp", np.abs(delta) <= 0.005),
            ("abs_dV_gt_2pp", np.abs(delta) > 0.02),
        ):
            block[tag] = agreement(y[mm], s[mm], g[mm]) if mm.any() else {"n_decided": 0}
        report["axes"][name] = block

    # B40 slice
    b40 = (p_pre >= 0.40) & (p_pre <= 0.60)
    report["B40"] = {
        name: agreement(y[b40], sgn(val[b40]), g[b40]) for name, val in axes.items()
    }
    report["arrays_meta"] = dict(y_pos_rate=float(y.mean()), mean_delta=float(delta.mean()))
    # stash for quiet matching
    report["_stash"] = dict(y=y, delta=delta, p_pre=p_pre, g=g, t_min=t_min, obj=obj)
    return report


def material_combat_from_extract(data_root: Path, fight_keys: set) -> Dict[str, Any]:
    """Kill and alive differentials from e_X_pre / e_X_post_h90 for MAIN_TEST T rows."""
    _setup_paths(data_root)
    import fc20260915_common as C  # noqa: E402

    states_dir = C.OUT / "extract" / "MAIN" / "states"
    # discover name indices from first chunk
    z0 = np.load(next(states_dir.glob("chunk_*.npz")), allow_pickle=False)
    names = [str(n) for n in z0["names"]]
    ix = {n: i for i, n in enumerate(names)}

    def slot_sum(X: np.ndarray, field: str, slots: range) -> np.ndarray:
        cols = [ix[f"participant_slot{s}_{field}"] for s in slots]
        return X[:, cols].sum(axis=1)

    y_list, d_list, g_list = [], [], []
    kill_pre, kill_post, alive_post = [], [], []
    obj_post = []  # not needed
    n_hit = 0
    for path in sorted(states_dir.glob("chunk_*.npz")):
        z = np.load(path, allow_pickle=False)
        em = z["e_match"].astype(str)
        es = z["e_s"].astype(np.int64)
        keys = np.char.add(em, np.char.add("|", es.astype(str)))
        hit = np.array([k in fight_keys for k in keys], dtype=bool)
        if not hit.any():
            continue
        # need labels for these — load Y/delta by key from caller stash via fight_keys map
        Xp = z["e_X_pre"][hit]
        Xq = z["e_X_post_h90"][hit]
        valid = z["e_valid_h90"][hit].astype(bool)
        if not valid.any():
            continue
        Xp, Xq = Xp[valid], Xq[valid]
        kh = keys[hit][valid]
        # blue slots 0-4, red 5-9 in participant order used by StateV2
        kb_pre = slot_sum(Xp, "kills", range(0, 5)) - slot_sum(Xp, "kills", range(5, 10))
        kb_post = slot_sum(Xq, "kills", range(0, 5)) - slot_sum(Xq, "kills", range(5, 10))
        alive = slot_sum(Xq, "alive", range(0, 5)) - slot_sum(Xq, "alive", range(5, 10))
        kill_pre.append(kb_pre)
        kill_post.append(kb_post)
        alive_post.append(alive)
        g_list.append(em[hit][valid])
        # store keys to join Y later
        y_list.append(kh)
        n_hit += int(valid.sum())

    if n_hit == 0:
        return dict(n=0, error="no extract rows matched fight keys")

    keys_all = np.concatenate(y_list)
    g_all = np.concatenate(g_list)
    d_kill = np.concatenate(kill_post) - np.concatenate(kill_pre)
    alive = np.concatenate(alive_post)
    return dict(
        n=int(n_hit),
        keys=keys_all,
        match=g_all,
        kill_diff_window=d_kill,
        alive_diff_post=alive,
        note="Blue−Red. Kill diff = (post cumulative kills − pre) difference across teams; "
             "alive_diff at post-state snapshot.",
    )


def join_combat(material: Dict[str, Any], combat: Dict[str, Any]) -> Dict[str, Any]:
    stash = material["_stash"]
    key_te = np.char.add(stash["g"].astype(str),
                         np.char.add("|", (stash["t_min"] * 60000).astype(np.int64).astype(str)))
    # t_min was s/60000 — better rebuild from labels in caller
    return combat  # joined in main with proper keys


# ------------------------------------------------------------------ quiet smoke
def _kill_times(pack: dict) -> np.ndarray:
    return np.asarray([
        int(e["timestamp"]) for e in pack["events"]
        if e.get("type") == "CHAMPION_KILL"
    ], dtype=np.int64)


def _no_kill(kill_ts: np.ndarray, t0: int, t1: int) -> bool:
    if kill_ts.size == 0:
        return True
    return not bool(np.any((kill_ts >= t0) & (kill_ts <= t1)))


def quiet_smoke(data_root: Path, n_matches: int, seed: int,
                fight_delta: np.ndarray, fight_p: np.ndarray, fight_t: np.ndarray,
                fight_g: np.ndarray) -> Dict[str, Any]:
    _setup_paths(data_root)
    import fc20260915_common as C  # noqa: E402
    from fc20260915_common import load_v_adapter  # noqa: E402
    from data.cache_io import load_match_cache  # noqa: E402
    from core.config import NODE_FEATURE_NAMES  # noqa: E402
    from gameplay.state_value_v2 import StateBuilder, state_matrix  # noqa: E402
    from gameplay.state_value import final_outcome  # noqa: E402

    adapter = load_v_adapter(C.OUT / "models" / "v" / "v_final_raw.joblib")
    names = list(adapter.state_names)

    # MAIN_TEST match ids that appear in T fights
    matches = np.unique(fight_g)
    rng = np.random.default_rng(seed)
    # deterministic hash sample
    scored = sorted(matches, key=lambda m: hash((seed, str(m))) % (2**63))
    pick = scored[:n_matches]

    quiet_delta, quiet_p, quiet_t, quiet_g = [], [], [], []
    quiet_snap_age_pre, quiet_snap_age_post, quiet_same_frame = [], [], []
    loaded_ids: List[str] = []
    n_loaded = n_skip = n_windows = 0
    step_ms = 30_000
    start_min_ms = 120_000

    for mid in pick:
        pack = load_match_cache(str(mid))
        if pack is None:
            n_skip += 1
            continue
        try:
            winner, terminal = final_outcome(pack["events"])
            builder = StateBuilder(pack, list(NODE_FEATURE_NAMES))
        except Exception:
            n_skip += 1
            continue
        n_loaded += 1
        loaded_ids.append(str(mid))
        kills = _kill_times(pack)
        ts = np.asarray(pack["minute_ts"], dtype=np.int64)
        if len(ts) < 3:
            continue
        last = int(min(terminal if terminal > 0 else ts[-1], ts[-1]))
        # candidate starts
        t0s = list(range(start_min_ms, max(start_min_ms, last - H_MS), step_ms))
        if not t0s:
            continue
        # at most 3 quiet windows per match, spaced
        chosen = []
        for t0 in t0s:
            t1 = t0 + H_MS
            if t1 > last:
                break
            if not _no_kill(kills, t0, t1):
                continue
            chosen.append(t0)
            if len(chosen) >= 3:
                break
        for t0 in chosen:
            try:
                s0 = builder.at(t0)
                s1 = builder.at(t0 + H_MS - 1)
                X = state_matrix([s0, s1], names)
                p = adapter.predict_matrix(X, names, adapter.state_version)
            except Exception:
                continue
            quiet_delta.append(float(p[1] - p[0]))
            quiet_p.append(float(p[0]))
            quiet_t.append(t0 / 60000.0)
            quiet_g.append(str(mid))
            quiet_snap_age_pre.append(float(t0 - s0.snapshot_ms) / 1000.0)
            quiet_snap_age_post.append(float((t0 + H_MS - 1) - s1.snapshot_ms) / 1000.0)
            quiet_same_frame.append(int(s0.snapshot_ms == s1.snapshot_ms))
            n_windows += 1

    qd = np.asarray(quiet_delta, dtype=float)
    qp = np.asarray(quiet_p, dtype=float)
    qt = np.asarray(quiet_t, dtype=float)
    qg = np.asarray(quiet_g)

    # Fight comparator = T rows in the *same loaded matches* as the quiet smoke
    # (not the full MAIN_TEST T cohort — that mismatch inflated |ΔV| ratios).
    loaded_set = set(loaded_ids)
    fm = np.array([str(x) in loaded_set for x in fight_g], dtype=bool) if loaded_set else np.zeros(len(fight_g), bool)
    fd, fp, ft, fg = fight_delta[fm], fight_p[fm], fight_t[fm], fight_g[fm]

    def strata_summary(delta, p, t, g):
        rows = []
        for (tlo, thi) in TIME_BANDS:
            for i in range(len(P_EDGES) - 1):
                plo, phi = float(P_EDGES[i]), float(P_EDGES[i + 1])
                m = (t >= tlo) & (t < thi) & (p >= plo) & (p < phi)
                if m.sum() < 20:
                    continue
                w = match_weights(g[m])
                rows.append(dict(
                    time_lo=tlo, time_hi=thi, p_lo=plo, p_hi=phi,
                    n=int(m.sum()), matches=int(len(np.unique(g[m]))),
                    mean_delta=float(np.average(delta[m], weights=w)),
                    mean_abs_delta=float(np.average(np.abs(delta[m]), weights=w)),
                    P_pos=float(np.average(delta[m] > 0, weights=w)),
                ))
        return rows

    fight_rows = strata_summary(fd, fp, ft, fg) if fm.any() else []
    quiet_rows = strata_summary(qd, qp, qt, qg) if len(qd) else []

    # join strata present in both
    def key(r):
        return (r["time_lo"], r["time_hi"], r["p_lo"], r["p_hi"])

    fmap = {key(r): r for r in fight_rows}
    qmap = {key(r): r for r in quiet_rows}
    paired = []
    for k in sorted(set(fmap) & set(qmap)):
        f, q = fmap[k], qmap[k]
        paired.append(dict(
            stratum=k,
            fight_n=f["n"], quiet_n=q["n"],
            fight_mean_delta=f["mean_delta"], quiet_mean_delta=q["mean_delta"],
            fight_P_pos=f["P_pos"], quiet_P_pos=q["P_pos"],
            excess_mean_delta=f["mean_delta"] - q["mean_delta"],
            excess_P_pos=f["P_pos"] - q["P_pos"],
        ))

    return dict(
        design="type_B_no_kill_fixed_H90_matched_matches",
        n_matches_requested=n_matches,
        n_matches_loaded=n_loaded,
        n_matches_skipped=n_skip,
        n_quiet_windows=int(len(qd)),
        n_fight_rows_matched=int(fm.sum()),
        n_fight_matches_matched=int(len(loaded_set)),
        overall_quiet=dict(
            mean_delta=float(qd.mean()) if len(qd) else None,
            mean_abs_delta=float(np.abs(qd).mean()) if len(qd) else None,
            P_pos=float((qd > 0).mean()) if len(qd) else None,
            mean_snap_age_pre_s=float(np.mean(quiet_snap_age_pre)) if quiet_snap_age_pre else None,
            mean_snap_age_post_s=float(np.mean(quiet_snap_age_post)) if quiet_snap_age_post else None,
            same_frame_share=float(np.mean(quiet_same_frame)) if quiet_same_frame else None,
        ),
        overall_fight=dict(
            mean_delta=float(np.average(fd, weights=match_weights(fg))) if fm.any() else None,
            mean_abs_delta=float(np.average(np.abs(fd), weights=match_weights(fg))) if fm.any() else None,
            P_pos=float(np.average(fd > 0, weights=match_weights(fg))) if fm.any() else None,
            n=int(fm.sum()),
            note="T rows restricted to matches successfully loaded for the quiet smoke",
        ),
        paired_strata=paired,
        interpretation="excess_* = fight minus quiet in the same time×p_pre stratum on the "
                       "matched match set; reference drift, not causal fight effect.",
    )


# ------------------------------------------------------------------ report
def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def write_report(out_dir: Path, payload: Dict[str, Any]) -> None:
    lines = []
    w = lines.append
    w("# Phase-2: material axis + quiet-window ΔV (smoke)")
    w("")
    w(f"Generated: {payload['generated']}")
    w(f"Data root: `{payload['data_root']}`")
    w("")
    w("Exploratory. Labels remain model-defined sign(ΔV). Material axes are checks, not a new primary label.")
    w("")
    w("## 1. Material axis vs sign(ΔV) — MAIN_TEST teamfights T")
    w("")
    mat = payload["material"]
    w(f"N = {mat['n']} rows / {mat['matches']} matches.")
    w("")
    w("| Axis | Decided n | Tie share | Agreement with Y | Spearman(net, ΔV) |")
    w("|---|---:|---:|---:|---:|")
    for name, ax in mat["axes"].items():
        w(f"| {name} | {ax['n_decided']} | {_fmt(ax['tie_share'], 3)} | "
          f"{_fmt(ax['agreement_rate'], 3)} | {_fmt(ax['spearman_vs_delta'], 3)} |")
    w("")
    w("### Small vs large |ΔV|")
    w("")
    for name, ax in mat["axes"].items():
        a = ax.get("abs_dV_le_0.5pp", {})
        b = ax.get("abs_dV_gt_2pp", {})
        w(f"- **{name}**: |ΔV|≤0.5pp agreement {_fmt(a.get('agreement_rate'), 3)} "
          f"(n_decided {a.get('n_decided')}); |ΔV|>2pp {_fmt(b.get('agreement_rate'), 3)} "
          f"(n_decided {b.get('n_decided')}).")
    w("")
    if "combat" in payload and payload["combat"].get("n"):
        c = payload["combat"]
        w("## 2. Combat axis (extract pre/post states)")
        w("")
        w(f"Joined rows: {c['n']}.")
        w("")
        w("| Axis | Decided n | Tie share | Agreement with Y | Spearman vs ΔV |")
        w("|---|---:|---:|---:|---:|")
        for name, ax in c["axes"].items():
            w(f"| {name} | {ax['n_decided']} | {_fmt(ax['tie_share'], 3)} | "
              f"{_fmt(ax['agreement_rate'], 3)} | {_fmt(ax.get('spearman_vs_delta'), 3)} |")
        w("")
        w(c.get("note", ""))
        w("")
    q = payload["quiet"]
    w("## 3. Quiet (no-kill) reference windows — smoke")
    w("")
    w(f"Design: {q['design']}. Loaded {q['n_matches_loaded']}/{q['n_matches_requested']} matches; "
      f"{q['n_quiet_windows']} quiet windows.")
    w("")
    oq, of_ = q["overall_quiet"], q["overall_fight"]
    w("| Population | n | mean ΔV | mean |ΔV| | P(ΔV>0) |")
    w("|---|---:|---:|---:|---:|")
    w(f"| Teamfights T (full) | {of_['n']} | {_fmt(of_['mean_delta'])} | {_fmt(of_['mean_abs_delta'])} | {_fmt(of_['P_pos'], 3)} |")
    w(f"| Quiet no-kill H=90s | {q['n_quiet_windows']} | {_fmt(oq['mean_delta'])} | {_fmt(oq['mean_abs_delta'])} | {_fmt(oq['P_pos'], 3)} |")
    w("")
    w(f"Quiet same-frame share {_fmt(oq.get('same_frame_share'), 3)}; "
      f"mean snapshot age pre/post {_fmt(oq.get('mean_snap_age_pre_s'), 1)} / {_fmt(oq.get('mean_snap_age_post_s'), 1)} s.")
    w("")
    w("### Stratum excess (fight − quiet)")
    w("")
    if not q["paired_strata"]:
        w("No overlapping strata with ≥20 rows each (increase --n-matches).")
    else:
        w("| Time | p_pre | fight n | quiet n | excess mean ΔV | excess P(ΔV>0) |")
        w("|---|---|---:|---:|---:|---:|")
        for r in q["paired_strata"]:
            tlo, thi, plo, phi = r["stratum"]
            w(f"| {tlo:.0f}-{thi if thi < 100 else 30}+ | [{plo:.2f},{phi:.2f}) | "
              f"{r['fight_n']} | {r['quiet_n']} | {_fmt(r['excess_mean_delta'])} | {_fmt(r['excess_P_pos'], 3)} |")
    w("")
    w("## 4. Discussion prompts")
    w("")
    for line in payload["takeaways"]:
        w(f"- {line}")
    w("")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): convert(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, list):
        return [convert(v) for v in o]
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return convert(o.tolist())
    return o


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-matches", type=int, default=800, help="quiet-smoke match budget")
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--skip-quiet", action="store_true")
    ap.add_argument("--skip-combat-extract", action="store_true")
    args = ap.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root

    print("[1/3] material from labels…", flush=True)
    material = material_from_labels(data_root)
    stash = material["_stash"]

    # proper keys: reload s from labels
    lab = np.load(data_root / "outputs" / "full_corpus_training_20260915" / "labels" / "MAIN_TEST_labels.npz",
                  allow_pickle=False)
    coh = np.load(data_root / "outputs" / "cohort_role_training_20260915" / "cohorts" / "MAIN_TEST_cohort.npz",
                  allow_pickle=False)
    m = (coh["cohort"] == 1) & (lab["valid_h90"] == 1)
    fight_keys_arr = np.char.add(lab["match"][m].astype(str),
                                 np.char.add("|", lab["s"][m].astype(str)))
    fight_keys = set(fight_keys_arr.tolist())
    y = lab["Y_h90"][m].astype(int)
    delta = lab["delta_h90"][m].astype(float)
    key_to_y = dict(zip(fight_keys_arr.tolist(), y.tolist()))
    key_to_delta = dict(zip(fight_keys_arr.tolist(), delta.tolist()))
    key_to_g = dict(zip(fight_keys_arr.tolist(), lab["match"][m].astype(str).tolist()))

    combat_block: Dict[str, Any] = dict(n=0)
    if not args.skip_combat_extract:
        print("[2/3] combat from extract states…", flush=True)
        raw = material_combat_from_extract(data_root, fight_keys)
        if raw.get("n", 0) > 0:
            keys = raw["keys"]
            y_c = np.array([key_to_y[k] for k in keys], dtype=int)
            d_c = np.array([key_to_delta[k] for k in keys], dtype=float)
            g_c = np.array([key_to_g[k] for k in keys])
            kd = raw["kill_diff_window"]
            al = raw["alive_diff_post"]
            combat_block = dict(
                n=int(raw["n"]),
                note=raw["note"],
                axes={
                    "kill_diff_window": {
                        **agreement(y_c, np.sign(kd).astype(int), g_c),
                        "spearman_vs_delta": spearman_safe(kd, d_c),
                    },
                    "alive_diff_post": {
                        **agreement(y_c, np.sign(al).astype(int), g_c),
                        "spearman_vs_delta": spearman_safe(al, d_c),
                    },
                },
            )
            # disagreement cases: Y vs kill when both decided
            sk = np.sign(kd).astype(int)
            decided = sk != 0
            disagree = decided & (((y_c == 1) & (sk == -1)) | ((y_c == 0) & (sk == 1)))
            combat_block["kill_disagree_share_among_decided"] = (
                float(disagree.sum() / max(1, decided.sum())) if decided.any() else None
            )
            # when kill says red but Y blue (or vice versa): objective often?
            combat_block["n_kill_disagree"] = int(disagree.sum())
        else:
            combat_block = raw

    if args.skip_quiet:
        quiet = dict(design="skipped", n_matches_requested=0, n_matches_loaded=0,
                     n_matches_skipped=0, n_quiet_windows=0, overall_quiet={},
                     overall_fight={}, paired_strata=[], interpretation="skipped")
    else:
        print(f"[3/3] quiet smoke on {args.n_matches} matches…", flush=True)
        quiet = quiet_smoke(
            data_root, args.n_matches, args.seed,
            fight_delta=stash["delta"], fight_p=stash["p_pre"],
            fight_t=stash["t_min"], fight_g=stash["g"],
        )

    takeaways = []
    ax = material["axes"]["objective_net"]
    takeaways.append(
        f"Objective/structure net agrees with sign(ΔV) on {_fmt(ax['agreement_rate'], 3)} of decided rows "
        f"(tie share {_fmt(ax['tie_share'], 3)}) — partial overlap, not identity."
    )
    if combat_block.get("n"):
        ka = combat_block["axes"]["kill_diff_window"]
        takeaways.append(
            f"Kill differential agrees with Y on {_fmt(ka['agreement_rate'], 3)} of decided rows; "
            f"disagreement share {_fmt(combat_block.get('kill_disagree_share_among_decided'), 3)} "
            f"— where ΔV and kill exchange part ways (objects/tempo candidates)."
        )
    if quiet.get("n_quiet_windows"):
        takeaways.append(
            f"Quiet no-kill windows: mean ΔV {_fmt(quiet['overall_quiet']['mean_delta'])} vs fights "
            f"{_fmt(quiet['overall_fight']['mean_delta'])}; compare excess_* in strata, not raw means alone."
        )
    takeaways.append(
        "Next discussion: keep ΔV as strategic-value label, use material axes as validity checks, "
        "and treat quiet excess as background-drift reference (type-A proximity still pending)."
    )

    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        data_root=str(data_root),
        material=material,
        combat=combat_block,
        quiet=quiet,
        takeaways=takeaways,
    )
    (out_dir / "results.json").write_text(
        json.dumps(convert(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out_dir, payload)
    print(json.dumps({
        "out_dir": str(out_dir),
        "obj_agreement": ax["agreement_rate"],
        "combat_n": combat_block.get("n"),
        "quiet_n": quiet.get("n_quiet_windows"),
        "quiet_mean_dV": quiet.get("overall_quiet", {}).get("mean_delta"),
        "fight_mean_dV": quiet.get("overall_fight", {}).get("mean_delta"),
        "n_paired_strata": len(quiet.get("paired_strata") or []),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
