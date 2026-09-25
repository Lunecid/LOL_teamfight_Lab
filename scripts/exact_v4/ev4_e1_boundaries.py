#!/usr/bin/env python3
"""E1 (v4-exact plan, section 1): joint re-estimation of the engagement boundaries (G, D).

The detector's two-step rule is kept unchanged (gameplay/fight_clustering.py):
  (1) temporal bundles: consecutive kills of a match with gap <= G join (cluster_kills_temporal);
  (2) inside each bundle, greedy first-fit complete-linkage grouping with max diameter D
      (split_kill_cluster_spatial; kills processed in time order, a kill joins the first group
      whose every member lies within D).

Pre-specified method (not tuned to results):
  * Pair universe U, fixed and independent of (G, D): every pair of time-consecutive kills
    (i, i+1) of the same match with both positions present; dt = 0 is included.
  * Signal s_i = 1 if the two kills share >= 1 participant (killer, victim or assister).
  * Rule decision r_i(G, D) = 1 if kills i and i+1 end in the same engagement (same bundle AND
    same greedy group).
  * Disagreement E(G, D) = sum_i [r_i != s_i], equal pair weights; FP = r=1,s=0 (joined without a
    shared participant), FN = r=0,s=1 (split although they share one).
  * Grid G = 6.0..30.0 s step 0.5; D = 2,000..8,000 step 100 (full grid, no coarse-to-fine needed
    when the 1,000-match timing allows it). Argmin on 15.14 (ties: smallest G, then smallest D).
  * Near-optimal region {E <= 1.01 * min}; surface profiles.
  * Match-level bootstrap (multinomial match weights) of the argmin over the full grid; percentile
    95% intervals.
  * Cross-check for G: 2-component Gaussian mixture on log10(dt) (posterior 0.5) and the KDE
    valley (analysis.temporal_boundary, as in the thesis), pooled and by phase.
  * Phase sensitivity (<15, 15-25, >=25 min by the first kill of the pair), 15.15 stability.
  * Reproduction checks first: (a) greedy == WT split_kill_cluster_spatial on >= 500 bundles,
    plus an end-to-end check of the surface kernel against the WT two-step rule;
    (b) p2d fair scan (G = 13.7, D = 3,500..5,000, 15.14, 0 < dt <= G) -> best D 4,900 / 134,307;
    (c) p2b pooled crossing 4,262.26 only when its scope is 15.14.

Patches 15.14 and 15.15 only: the kill table is filtered by patch before anything else.
Writes only to outputs/reest_exact_v4_20260925/e1.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
WT = HERE.parents[2]
DIAG = Path("C:/Users/todtj/문서/LOL_Teamfight/outputs/diag_survival_dbscan_20260925")
OUT = Path("C:/Users/todtj/문서/LOL_Teamfight/outputs/reest_exact_v4_20260925/e1")
KILLS = DIAG / "p1" / "kills_full.npz"
MATCHES = DIAG / "p1" / "matches_full.json"
P2B_JSON = DIAG / "p2b" / "distance_boundary_full.json"
P2D_JSON = DIAG / "p2b" / "review_fixes_full.json"

# worktree modules first, so that the diagnostic scripts' repo path cannot shadow them
sys.path.insert(0, str(WT))
import analysis.temporal_boundary as TB  # noqa: E402
import gameplay.fight_clustering as FC  # noqa: E402
from analysis.spatial_boundary import crossover  # noqa: E402

sys.path.insert(1, str(DIAG / "scripts"))
import p2b_distance_boundary as P2B  # noqa: E402

PATCHES = ("15.14", "15.15")
G_GRID_S = np.round(np.arange(6.0, 30.0 + 1e-9, 0.5), 1)
G_GRID_MS = np.round(G_GRID_S * 1000).astype(np.int64)
D_GRID = np.arange(2000.0, 8000.0 + 1e-9, 100.0)
PHASE_EDGES_MS = (15 * 60_000, 25 * 60_000)
PHASES = ("lt15", "15to25", "ge25")
NCAT = 6  # cat = phase * 2 + s
G_CUR_MS, D_CUR = 13_700, 4264.0
NEAR = 1.01
B_BOOT = 1000
SEED = 20260925
CHUNK = 1000
WORKERS = 4


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------- data
def load_patches():
    """Kill columns for 15.14/15.15 only; the patch filter is applied before any other use."""
    M = json.loads(MATCHES.read_text(encoding="utf-8"))
    patch_of = np.array([m["patch"] for m in M])
    z = np.load(KILLS)
    mi_all = z["match_idx"]
    out = {}
    for p in PATCHES:
        keep_m = patch_of == p
        sel = keep_m[mi_all]
        K = {c: z[c][sel] for c in ("t_ms", "killer", "victim", "assist_mask", "x", "y", "match_idx")}
        del sel
        out[p] = K
    del mi_all
    return out


def prepare(K):
    """Pair arrays (pair i = kills (i, i+1)), match ranges, participant sharing, phase."""
    mask = P2B.participant_mask(K)
    cp = P2B.consecutive_pairs(K, mask)  # dt, sh, ok (same match and both positions finite)
    mi = K["match_idx"]
    starts = np.flatnonzero(np.r_[True, mi[1:] != mi[:-1]])
    ends = np.r_[starts[1:], len(mi)]
    t = K["t_ms"].astype(np.int64)
    phase = np.digitize(t[:-1], PHASE_EDGES_MS)  # phase of the first kill of the pair
    U = cp["ok"]  # same match, both positions present, dt >= 0 (time-sorted)
    s = cp["sh"].astype(np.int64)
    cat = np.where(U, phase * 2 + s, -1).astype(np.int8)
    return {"t": t, "x": K["x"].astype(np.float64), "y": K["y"].astype(np.float64),
            "cat": np.r_[cat, np.int8(-1)], "U": U, "s": cp["sh"], "dt": cp["dt"], "phase": phase,
            "starts": starts.astype(np.int64), "ends": ends.astype(np.int64), "same": np.r_[mi[1:] == mi[:-1]]}


# ----------------------------------------------------------------------------------- kernels
_NB = None


def _numba():
    global _NB
    if _NB is not None:
        return _NB
    from numba import njit

    @njit(cache=False)
    def greedy_labels(px, py, r2, lab):
        """First-fit diameter grouping in the given (time) order; returns the number of groups."""
        n = px.shape[0]
        ng = 0
        for q in range(n):
            placed = False
            for g in range(ng):
                ok = True
                for p in range(q):
                    if lab[p] == g:
                        dx = px[q] - px[p]
                        dy = py[q] - py[p]
                        if dx * dx + dy * dy > r2:
                            ok = False
                            break
                if ok:
                    lab[q] = g
                    placed = True
                    break
            if not placed:
                lab[q] = ng
                ng += 1
        return ng

    @njit(cache=False)
    def surface(t, x, y, cat, mstarts, mends, gms, d2grid, ncat, out):
        """out[m, cat, g, d] += 1 for every universe pair (i, i+1) the two-step rule joins."""
        nm = mstarts.shape[0]
        nG = gms.shape[0]
        nD = d2grid.shape[0]
        maxn = 1
        for mm in range(nm):
            if mends[mm] - mstarts[mm] > maxn:
                maxn = mends[mm] - mstarts[mm]
        lab = np.empty(maxn, np.int64)
        loc = np.empty(maxn, np.int64)
        px = np.empty(maxn, np.float64)
        py = np.empty(maxn, np.float64)
        for mm in range(nm):
            s0 = mstarts[mm]
            e0 = mends[mm]
            for gi in range(nG):
                G = gms[gi]
                b = s0
                while b < e0:
                    e = b + 1
                    while e < e0 and t[e] - t[e - 1] <= G:
                        e += 1
                    if e - b >= 2:
                        has = False
                        for a in range(b, e - 1):
                            if cat[a] >= 0:
                                has = True
                                break
                        if has:
                            npos = 0
                            for k in range(b, e):
                                if np.isfinite(x[k]) and np.isfinite(y[k]):
                                    loc[k - b] = npos
                                    px[npos] = x[k]
                                    py[npos] = y[k]
                                    npos += 1
                                else:
                                    loc[k - b] = -1
                            maxd2 = 0.0
                            for i in range(npos):
                                for j in range(i + 1, npos):
                                    dx = px[i] - px[j]
                                    dy = py[i] - py[j]
                                    dd = dx * dx + dy * dy
                                    if dd > maxd2:
                                        maxd2 = dd
                            for di in range(nD):
                                r2 = d2grid[di]
                                if maxd2 <= r2:
                                    # the whole bundle fits: every universe pair is joined for this and
                                    # every larger D on the (ascending) grid
                                    for a in range(b, e - 1):
                                        c = cat[a]
                                        if c >= 0:
                                            for dj in range(di, nD):
                                                out[mm, c, gi, dj] += 1
                                    break
                                greedy_labels(px[:npos], py[:npos], r2, lab)
                                for a in range(b, e - 1):
                                    c = cat[a]
                                    if c >= 0:
                                        la = loc[a - b]
                                        lb = loc[a + 1 - b]
                                        if la >= 0 and lb >= 0 and lab[la] == lab[lb]:
                                            out[mm, c, gi, di] += 1
                    b = e
        return out

    _NB = (greedy_labels, surface)
    return _NB


def _chunk_job(args):
    """Worker: joined counts for a block of matches -> per-match disagreement and category totals."""
    t, x, y, cat, ms, me, gms, d2, s_is1 = args
    _, surface = _numba()
    nm = len(ms)
    out = np.zeros((nm, NCAT, len(gms), len(d2)), dtype=np.int32)
    t0 = time.perf_counter()
    surface(t, x, y, cat, ms, me, gms, d2, NCAT, out)
    el = time.perf_counter() - t0
    # per-match number of universe pairs by category
    ncnt = np.zeros((nm, NCAT), dtype=np.int64)
    for k in range(nm):
        c = cat[ms[k]:me[k]]
        c = c[c >= 0]
        ncnt[k] = np.bincount(c, minlength=NCAT)
    s1 = np.array(s_is1, dtype=bool)
    joined0 = out[:, ~s1].sum(1)                     # FP per match
    split1 = ncnt[:, s1].sum(1)[:, None, None] - out[:, s1].sum(1)  # FN per match
    E = (joined0 + split1).reshape(nm, -1)
    assert E.max() < 32767 and E.min() >= 0
    return {"E": E.astype(np.int16), "joined": out.sum(0).astype(np.int64), "n": ncnt.sum(0), "elapsed": el}


def run_surface(P, cat, gms, d2, match_sel=None, workers=WORKERS, chunk=CHUNK, s_is1=None):
    """Full surface over the grid for the matches in match_sel (indices into P['starts'])."""
    if s_is1 is None:
        s_is1 = [c % 2 == 1 for c in range(NCAT)]
    ms_all, me_all = P["starts"], P["ends"]
    idx = np.arange(len(ms_all)) if match_sel is None else np.asarray(match_sel)
    jobs = []
    for j in range(0, len(idx), chunk):
        blk = idx[j:j + chunk]
        # contiguous copy of the kills of these matches
        segs = [np.arange(ms_all[k], me_all[k]) for k in blk]
        kk = np.concatenate(segs)
        lens = np.array([len(s) for s in segs], dtype=np.int64)
        ms = np.r_[0, np.cumsum(lens)[:-1]].astype(np.int64)
        me = (ms + lens).astype(np.int64)
        c = cat[kk].copy()
        # the last kill of each match has no pair inside the match
        c[me - 1] = -1
        jobs.append((P["t"][kk], P["x"][kk], P["y"][kk], c, ms, me,
                     np.asarray(gms, np.int64), np.asarray(d2, np.float64), s_is1))
    t0 = time.perf_counter()
    if workers <= 1:
        res = [_chunk_job(j) for j in jobs]
    else:
        with get_context("spawn").Pool(workers) as pool:
            res = pool.map(_chunk_job, jobs, chunksize=1)
    wall = time.perf_counter() - t0
    E = np.concatenate([r["E"] for r in res], axis=0)
    joined = np.sum([r["joined"] for r in res], axis=0)
    n = np.sum([r["n"] for r in res], axis=0)
    return {"E_match": E, "joined": joined, "n": n, "wall_s": wall,
            "cpu_s": float(sum(r["elapsed"] for r in res))}


# ------------------------------------------------------------------------------- summaries
def surface_from(joined, n, cats_s0, cats_s1):
    fp = joined[list(cats_s0)].sum(0)
    fn = n[list(cats_s1)].sum() - joined[list(cats_s1)].sum(0)
    N = int(n[list(cats_s0)].sum() + n[list(cats_s1)].sum())
    return fp.astype(np.int64), fn.astype(np.int64), N


def argmin_first(E):
    k = int(np.argmin(E))  # row-major: smallest G first, then smallest D
    return np.unravel_index(k, E.shape)


def describe(fp, fn, N, n_s1):
    E = fp + fn
    gi, di = argmin_first(E)
    emin = int(E[gi, di])
    near = E <= NEAR * emin
    ng, nd = np.nonzero(near)
    per_g = {}
    for g in sorted(set(ng.tolist())):
        ds = D_GRID[nd[ng == g]]
        per_g[f"{G_GRID_S[g]:.1f}"] = [float(ds.min()), float(ds.max()), int(len(ds))]
    prof_g = [{"G": float(G_GRID_S[g]), "min_E": int(E[g].min()), "argmin_D": float(D_GRID[int(np.argmin(E[g]))]),
               "rel": float(E[g].min() / emin)} for g in range(len(G_GRID_S))]
    prof_d = [{"D": float(D_GRID[d]), "min_E": int(E[:, d].min()), "argmin_G": float(G_GRID_S[int(np.argmin(E[:, d]))]),
               "rel": float(E[:, d].min() / emin)} for d in range(len(D_GRID))]
    border = np.concatenate([E[0], E[-1], E[:, 0], E[:, -1]])
    ag = np.array([p["argmin_D"] for p in prof_g])
    return {
        "G_star_s": float(G_GRID_S[gi]), "D_star": float(D_GRID[di]),
        "E_min": emin, "rate": emin / N, "N_pairs_U": int(N), "n_sharing": int(n_s1),
        "FP_joined_without_sharing": int(fp[gi, di]), "FN_split_while_sharing": int(fn[gi, di]),
        "FP_rate_among_nonsharing": float(fp[gi, di] / max(N - n_s1, 1)),
        "FN_rate_among_sharing": float(fn[gi, di] / max(n_s1, 1)),
        "n_cells_tied_at_min": int((E == emin).sum()),
        "argmin_on_grid_edge": {"G": bool(gi in (0, len(G_GRID_S) - 1)), "D": bool(di in (0, len(D_GRID) - 1))},
        "near_optimal": {"threshold": f"E <= {NEAR} * min", "E_max": float(NEAR * emin), "n_cells": int(near.sum()),
                         "G_range_s": [float(G_GRID_S[ng.min()]), float(G_GRID_S[ng.max()])],
                         "D_range": [float(D_GRID[nd.min()]), float(D_GRID[nd.max()])],
                         "D_range_by_G": per_g},
        "surface_shape": {
            "border_min_rel": float(border.min() / emin),
            "corners_rel": {"G6_D2000": float(E[0, 0] / emin), "G6_D8000": float(E[0, -1] / emin),
                            "G30_D2000": float(E[-1, 0] / emin), "G30_D8000": float(E[-1, -1] / emin)},
            "profile_over_G": prof_g, "profile_over_D": prof_d,
            "argmin_D_vs_G_slope_per_s": float(np.polyfit(G_GRID_S, ag, 1)[0]),
        },
    }


def boot_argmin(E_match, B, seed, grid_shape):
    """Multinomial match bootstrap of the full-grid argmin (weights applied to per-match E)."""
    nm = E_match.shape[0]
    rng = np.random.default_rng(seed)
    W = np.empty((B, nm), dtype=np.int16)
    for j in range(0, B, 100):
        b = min(100, B - j)
        W[j:j + b] = rng.multinomial(nm, np.full(nm, 1.0 / nm), size=b).astype(np.int16)
    acc = np.zeros((B, E_match.shape[1]), dtype=np.float64)
    step = 4000
    for j in range(0, nm, step):
        acc += W[:, j:j + step].astype(np.float64) @ E_match[j:j + step].astype(np.float64)
    ks = np.argmin(acc, axis=1)
    gi, di = np.unravel_index(ks, grid_shape)
    g = G_GRID_S[gi]
    d = D_GRID[di]
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    ug, cg = np.unique(g, return_counts=True)
    ud, cd = np.unique(d, return_counts=True)
    return {"B": int(B), "seed": int(seed), "weights": "multinomial(n_matches, 1/n_matches)",
            "G_ci95_s": q(g), "D_ci95": q(d), "G_median_s": float(np.median(g)), "D_median": float(np.median(d)),
            "G_dist": {f"{a:.1f}": int(b) for a, b in zip(ug, cg)}, "D_dist": {f"{a:.0f}": int(b) for a, b in zip(ud, cd)},
            "share_on_edge_G": float(np.isin(gi, [0, len(G_GRID_S) - 1]).mean()),
            "share_on_edge_D": float(np.isin(di, [0, len(D_GRID) - 1]).mean())}


def gap_crosscheck(P):
    """Mixture posterior-0.5 crossing and KDE valley on log10 dt (dt > 0), pooled and by phase."""
    same = P["same"]
    dt = P["dt"][same] / 1000.0
    ph = P["phase"][same]
    res = {"n_pairs_same_match": int(same.sum()), "n_dt0_excluded": int((dt <= 0).sum())}
    for name, sel in [("pooled", np.ones(dt.size, bool))] + [(PHASES[k], ph == k) for k in range(3)]:
        v = dt[sel]
        lg = TB.to_log(v)
        mix = TB.mixture_crossing(lg)
        val = TB.kde_valley(lg).as_dict()
        res[name] = {"n": int(v.size), "n_positive": int(lg.size), "mixture": mix, "kde_valley": val}
    return res


def _gap_job(args):
    name, dt, ph = args
    same = np.ones(dt.size, bool)
    return name, gap_crosscheck({"same": same, "dt": dt, "phase": ph})


# ------------------------------------------------------------------------------ checks
def canon(lab):
    m, out = {}, []
    for v in lab:
        m.setdefault(int(v), len(m))
        out.append(m[int(v)])
    return tuple(out)


def check_a(P, rng, n_bundles=600):
    """Greedy grouping: p2b.greedy_groups and the numba kernel vs WT split_kill_cluster_spatial."""
    greedy_labels, _ = _numba()
    t, x, y = P["t"], P["x"], P["y"]
    same = P["same"]
    results = {"n_tested": 0, "n_equal_p2b": 0, "n_equal_numba": 0, "n_split_cases": 0, "sizes": {}, "mismatch_examples": []}
    tried = 0
    while results["n_tested"] < n_bundles and tried < 200_000:
        tried += 1
        G = int(rng.choice(G_GRID_MS))
        D = float(rng.choice(D_GRID))
        i = int(rng.integers(0, len(t) - 1))
        # grow the bundle containing kill i at gap G
        b = i
        while b > 0 and same[b - 1] and t[b] - t[b - 1] <= G:
            b -= 1
        e = i + 1
        while e < len(t) and same[e - 1] and t[e] - t[e - 1] <= G:
            e += 1
        n = e - b
        # oversample splitting-prone bundles: keep n>=3 always, n==2 with prob 0.1
        if n < 2 or (n == 2 and rng.random() > 0.1):
            continue
        kills = [{"timestamp": int(t[k]), "position": (float(x[k]), float(y[k])), "killer_id": 0,
                  "victim_id": 0, "assisting_ids": [], "_k": k - b} for k in range(b, e)]
        cl = FC.cluster_kills_temporal(kills, G)
        assert len(cl) == 1, "bundle construction differs from cluster_kills_temporal"
        parts = FC.split_kill_cluster_spatial(cl[0], D)
        wt = np.full(n, -1)
        for ci, c in enumerate(parts):
            for kd in c["kills"]:
                wt[kd["_k"]] = ci
        Pm = np.stack([x[b:e], y[b:e]], 1)
        p2b = P2B.greedy_groups(Pm, t[b:e], D)
        lab = np.empty(n, np.int64)
        greedy_labels(x[b:e].copy(), y[b:e].copy(), D * D, lab)
        cw, cp_, cn = canon(wt), canon(p2b), canon(lab)
        results["n_tested"] += 1
        results["n_equal_p2b"] += int(cw == cp_)
        results["n_equal_numba"] += int(cw == cn)
        results["n_split_cases"] += int(len(parts) > 1)
        results["sizes"][str(min(n, 10))] = results["sizes"].get(str(min(n, 10)), 0) + 1
        if (cw != cp_ or cw != cn) and len(results["mismatch_examples"]) < 5:
            results["mismatch_examples"].append({"start": int(b), "n": int(n), "G_ms": G, "D": D})
    results["pass"] = results["n_equal_p2b"] == results["n_tested"] == results["n_equal_numba"]
    return results


def check_a2(P, rng, n_matches=300):
    """End-to-end: surface kernel vs the WT two-step rule on random matches for random grid cells."""
    ms_all, me_all = P["starts"], P["ends"]
    msel = np.sort(rng.choice(len(ms_all), n_matches, replace=False))
    cells = [(int(rng.integers(0, len(G_GRID_MS))), int(rng.integers(0, len(D_GRID)))) for _ in range(6)]
    cells += [(int(np.where(G_GRID_S == 13.5)[0][0]), int(np.where(D_GRID == 4300)[0][0])),
              (0, 0), (len(G_GRID_MS) - 1, len(D_GRID) - 1)]
    res = run_surface(P, P["cat"], G_GRID_MS, D_GRID ** 2, match_sel=msel, workers=1, chunk=n_matches)
    out = []
    for gi, di in cells:
        G, D = int(G_GRID_MS[gi]), float(D_GRID[di])
        joined = np.zeros(NCAT, np.int64)
        for m in msel:
            s, e = ms_all[m], me_all[m]
            kills = [{"timestamp": int(P["t"][k]), "position": (float(P["x"][k]), float(P["y"][k])),
                      "killer_id": 0, "victim_id": 0, "assisting_ids": [], "_k": int(k)} for k in range(s, e)]
            eng = {}
            eid = 0
            for c in FC.cluster_kills_temporal(kills, G):
                for part in FC.split_kill_cluster_spatial(c, D):
                    for kd in part["kills"]:
                        eng[kd["_k"]] = eid
                    eid += 1
            for k in range(s, e - 1):
                c = P["cat"][k]
                if c >= 0 and eng[k] == eng[k + 1]:
                    joined[c] += 1
        kern = res["joined"][:, gi, di]
        out.append({"G_s": float(G_GRID_S[gi]), "D": D, "wt_joined_by_cat": joined.tolist(),
                    "kernel_joined_by_cat": kern.tolist(), "equal": bool(np.array_equal(joined, kern))})
    return {"n_matches": int(n_matches), "cells": out, "pass": all(c["equal"] for c in out)}


def check_b(P):
    """p2d fair scan: 15.14, G = 13.7 s, D = 3,500..5,000, pairs with 0 < dt <= G and positions."""
    ref = json.loads(P2D_JSON.read_text(encoding="utf-8"))["fair_comparison_15.14"]
    dgrid = np.arange(3500.0, 5001.0, 100.0)
    win = P["U"] & (P["dt"] > 0) & (P["dt"] <= G_CUR_MS)
    cat = np.r_[np.where(win, P["s"].astype(np.int64), -1).astype(np.int8), np.int8(-1)]
    r = run_surface(P, cat, np.array([G_CUR_MS]), dgrid ** 2, workers=WORKERS, s_is1=[c % 2 == 1 for c in range(NCAT)])
    fp = r["joined"][0, 0]
    fn = r["n"][1] - r["joined"][1, 0]
    err = {f"{d:.1f}": int(fp[k] + fn[k]) for k, d in enumerate(dgrid)}
    best = min(err, key=err.get)
    same_all = all(err[k] == int(ref["greedy_by_D"][k]) for k in ref["greedy_by_D"])
    return {"n_pairs": int(win.sum()), "ref_n_pairs": int(ref["n_pairs"]), "greedy_by_D": err,
            "best_D": float(best), "best": err[best], "ref_best_D": ref["greedy_best_D"], "ref_best": ref["greedy_best"],
            "all_16_values_equal": bool(same_all),
            "pass": bool(same_all and int(win.sum()) == int(ref["n_pairs"]) and float(best) == ref["greedy_best_D"])}


def check_c(P):
    """Scope of the p2b pooled crossing; reproduced only when it is 15.14."""
    ref = json.loads(P2B_JSON.read_text(encoding="utf-8"))["reproduce"]
    pooled_n = int(ref["pooled"]["n_pairs"])
    win = P["U"] & (P["dt"] > 0) & (P["dt"] <= G_CUR_MS)
    n1514 = int(win.sum())
    out = {"pooled_crossing": ref["pooled"]["crossover"], "pooled_n_pairs": pooled_n, "n_pairs_15.14_same_rule": n1514,
           "scope_is_15.14": pooled_n == n1514,
           "note": "p2b docstring: 'Estimation populations: pooled 15.14-15.16'. The pooled crossing covers more "
                   "patches than 15.14 (n_pairs differs), so it is not reproduced (hard rule: no 15.16)."}
    if pooled_n == n1514:
        out["status"] = "in scope"
    else:
        out["status"] = "skipped (scope is not 15.14)"
    # the same file also holds a 15.14-only crossing, which is in scope; reproduce it as an extra check
    if "15.14" in ref:
        dx = P["x"][1:] - P["x"][:-1]
        dy = P["y"][1:] - P["y"][:-1]
        dd = np.hypot(dx.astype(np.float32), dy.astype(np.float32)).astype(np.float64)
        c, _ = crossover(dd[win], P["s"][win].astype(float))
        out["extra_15.14_crossing"] = {"ref": ref["15.14"]["crossover"], "ref_n": int(ref["15.14"]["n_pairs"]),
                                       "reproduced": c, "n": n1514,
                                       "equal": bool(c is not None and abs(c - ref["15.14"]["crossover"]) < 1e-6)}
    return out


# ------------------------------------------------------------------------------ main
def estimate_patch(P, tag, workers):
    t0 = time.perf_counter()
    r = run_surface(P, P["cat"], G_GRID_MS, D_GRID ** 2, workers=workers)
    joined, n = r["joined"], r["n"]
    fp, fn, N = surface_from(joined, n, [0, 2, 4], [1, 3, 5])
    n_s1 = int(n[[1, 3, 5]].sum())
    est = describe(fp, fn, N, n_s1)
    phase = {}
    for k, name in enumerate(PHASES):
        pfp, pfn, pN = surface_from(joined, n, [2 * k], [2 * k + 1])
        d = describe(pfp, pfn, pN, int(n[2 * k + 1]))
        d["surface_shape"] = {kk: v for kk, v in d["surface_shape"].items() if kk not in ("profile_over_G", "profile_over_D")}
        phase[name] = d
    tb = time.perf_counter()
    boot = boot_argmin(r["E_match"], B_BOOT, SEED + (0 if tag == "15.14" else 1), fp.shape)
    boot["elapsed_s"] = time.perf_counter() - tb
    # current rule (G = 13.7, D = 4,264) on the same universe, for scale
    rc = run_surface(P, P["cat"], np.array([G_CUR_MS]), np.array([D_CUR ** 2]), workers=workers)
    cfp, cfn, cN = surface_from(rc["joined"], rc["n"], [0, 2, 4], [1, 3, 5])
    cur = {"G_s": 13.7, "D": D_CUR, "E": int(cfp[0, 0] + cfn[0, 0]), "rate": float((cfp[0, 0] + cfn[0, 0]) / cN),
           "FP": int(cfp[0, 0]), "FN": int(cfn[0, 0])}
    np.savez_compressed(OUT / f"surface_{tag.replace('.', '')}.npz", G_s=G_GRID_S, D=D_GRID, FP=fp, FN=fn, E=fp + fn,
                        joined_by_cat=joined, n_by_cat=n, cat_legend=np.array([f"{PHASES[c // 2]}_s{c % 2}" for c in range(NCAT)]),
                        boot_note=np.array(f"B={B_BOOT} seed={boot['seed']}"))
    return {"n_matches": int(len(P["starts"])), "n_kills": int(len(P["t"])),
            "n_dt0_pairs_in_U": int((P["U"] & (P["dt"] == 0)).sum()),
            "n_pairs_U_by_cat": {f"{PHASES[c // 2]}_s{c % 2}": int(n[c]) for c in range(NCAT)},
            "estimate": est, "bootstrap": boot, "phase": phase, "current_rule": cur,
            "surface_wall_s": r["wall_s"], "surface_cpu_s": r["cpu_s"], "total_s": time.perf_counter() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("time", "checks", "full"), default="full")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    workers = min(a.workers, 4)
    OUT.mkdir(parents=True, exist_ok=True)
    T0 = time.perf_counter()
    data = load_patches()
    PP = {p: prepare(data[p]) for p in PATCHES}
    del data
    rng = np.random.default_rng(SEED)
    res = {}
    if a.stage == "time":
        P = PP["15.14"]
        sel = np.sort(rng.choice(len(P["starts"]), 1000, replace=False))
        r = run_surface(P, P["cat"], G_GRID_MS, D_GRID ** 2, match_sel=sel, workers=1, chunk=1000)
        print(json.dumps({"matches": 1000, "grid": [len(G_GRID_MS), len(D_GRID)], "cpu_s": r["cpu_s"], "wall_s": r["wall_s"],
                          "projected_full_1514_cpu_s": r["cpu_s"] * len(P["starts"]) / 1000}, indent=1))
        return 0

    res["timing_1000_matches_15.14"] = {}
    P = PP["15.14"]
    sel = np.sort(rng.choice(len(P["starts"]), 1000, replace=False))
    r = run_surface(P, P["cat"], G_GRID_MS, D_GRID ** 2, match_sel=sel, workers=1, chunk=1000)
    res["timing_1000_matches_15.14"] = {"cpu_s_incl_compile": r["cpu_s"], "wall_s": r["wall_s"],
                                        "projected_full_grid_cpu_s": r["cpu_s"] * len(P["starts"]) / 1000,
                                        "decision": "full grid (no coarse-to-fine needed)"}
    print("timing", res["timing_1000_matches_15.14"], flush=True)

    tc = time.perf_counter()
    checks = {"a_greedy_vs_WT": check_a(P, rng), "a2_kernel_vs_WT_two_step": check_a2(P, rng),
              "b_p2d_fair_scan": check_b(P), "c_p2b_pooled_crossing": check_c(P)}
    checks["elapsed_s"] = time.perf_counter() - tc
    res["reproduction_checks"] = checks
    print(json.dumps({k: v.get("pass", v.get("status")) if isinstance(v, dict) else v for k, v in checks.items()}), flush=True)
    if a.stage == "checks":
        print(json.dumps(checks, indent=1, default=float))
        return 0
    if not (checks["a_greedy_vs_WT"]["pass"] and checks["a2_kernel_vs_WT_two_step"]["pass"] and checks["b_p2d_fair_scan"]["pass"]):
        (OUT / "e1_results_FAILED_CHECKS.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        print("reproduction checks failed; stopping", flush=True)
        return 2

    for p in PATCHES:
        print("estimating", p, flush=True)
        res[p] = estimate_patch(PP[p], p, workers)
        print(p, {k: res[p]["estimate"][k] for k in ("G_star_s", "D_star", "E_min", "rate")},
              res[p]["bootstrap"]["G_ci95_s"], res[p]["bootstrap"]["D_ci95"], flush=True)

    tg = time.perf_counter()
    jobs = []
    for p in PATCHES:
        same = PP[p]["same"]
        jobs.append((p, PP[p]["dt"][same], PP[p]["phase"][same]))
    with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        for name, g in pool.map(_gap_job, jobs):
            res[name]["gap_crosscheck"] = g
    res["gap_crosscheck_elapsed_s"] = time.perf_counter() - tg

    b14, b15 = res["15.14"]["bootstrap"], res["15.15"]["bootstrap"]
    ov = lambda a_, b_: bool(a_[0] <= b_[1] and b_[0] <= a_[1])
    res["stability"] = {"G_ci_overlap": ov(b14["G_ci95_s"], b15["G_ci95_s"]),
                        "D_ci_overlap": ov(b14["D_ci95"], b15["D_ci95"]),
                        "G_star": [res["15.14"]["estimate"]["G_star_s"], res["15.15"]["estimate"]["G_star_s"]],
                        "D_star": [res["15.14"]["estimate"]["D_star"], res["15.15"]["estimate"]["D_star"]],
                        "E_at_1514_argmin_on_1515_rel": None}
    s15 = np.load(OUT / "surface_1515.npz")
    gi = int(np.where(G_GRID_S == res["15.14"]["estimate"]["G_star_s"])[0][0])
    di = int(np.where(D_GRID == res["15.14"]["estimate"]["D_star"])[0][0])
    res["stability"]["E_at_1514_argmin_on_1515_rel"] = float(s15["E"][gi, di] / s15["E"].min())

    inputs = {str(p): sha256(p) for p in (KILLS, MATCHES, P2B_JSON, P2D_JSON)}
    mods = {m.__name__: {"file": m.__file__, "sha256": sha256(Path(m.__file__))} for m in (FC, TB, P2B)}
    mods["analysis.spatial_boundary"] = {"file": sys.modules["analysis.spatial_boundary"].__file__,
                                         "sha256": sha256(Path(sys.modules["analysis.spatial_boundary"].__file__))}
    import numba
    import sklearn
    import scipy
    res["meta"] = {"script": str(HERE), "script_sha256": sha256(HERE), "inputs_sha256": inputs, "modules": mods,
                   "generated_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(),
                   "numpy": np.__version__, "numba": numba.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__,
                   "workers": workers, "seed": SEED, "patches": list(PATCHES),
                   "grid": {"G_s": [float(G_GRID_S[0]), float(G_GRID_S[-1]), 0.5, len(G_GRID_S)],
                            "D": [float(D_GRID[0]), float(D_GRID[-1]), 100.0, len(D_GRID)]},
                   "universe": "consecutive kills (i, i+1), same match, both positions present, dt >= 0",
                   "signal": "share >= 1 participant among killer, victim, assisters (ids 1..10)",
                   "rule": "WT cluster_kills_temporal (gap <= G) then split_kill_cluster_spatial (greedy first-fit, diameter <= D)",
                   "tie_break": "argmin in row-major order: smallest G, then smallest D",
                   "near_optimal": f"E <= {NEAR} * min", "phases_ms": list(PHASE_EDGES_MS),
                   "bootstrap": f"{B_BOOT} multinomial match-weight replicates, full grid, percentile 95%",
                   "total_wall_s": time.perf_counter() - T0}
    (OUT / "e1_results.json").write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print("done", time.perf_counter() - T0, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
