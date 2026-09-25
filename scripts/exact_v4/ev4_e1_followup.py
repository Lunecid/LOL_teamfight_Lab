#!/usr/bin/env python3
"""E1 follow-up (fallback plan, pending author approval): G from the kill-gap distribution, D conditional on G.

Background: the pre-specified joint sharing criterion does not identify G (argmin at the 30 s grid
edge; E1 report). The fallback takes G from the thesis valley method (analysis.temporal_boundary.
kde_valley, unchanged: KDE on log10 dt, bandwidth factor 0.08, 1M-point subsample seed 7, search
split by a 2-component mixture) and then D = argmin over the D grid of E_w = w*FP + FN at that G, on
the same consecutive-pair universe U as E1 (FP = joined without sharing, FN = split while sharing).

Items
  1. G valley, pooled + match-level bootstrap (multinomial match weights == resampling matches,
     B = 1,000, the same kde_valley estimator applied to every replicate) and by phase.
  2. Two-component Gaussian mixture on log10 dt (dt > 0), all points, >= 10 starts
     (kmeans init n_init=10 + random_from_data init n_init=10, best log-likelihood kept), pooled and by
     phase; crossing = root of w1 N1 = w2 N2 between the means (posterior 0.5).
  3. Conditional D at G in {13.7, 14.0}; w in {1, 2, 4}; 1% band; match bootstrap (B = 1,000) of D*
     using per-match FP/FN arrays (weights applied after resampling).
  4. Conditional D by phase at G = 14.0 (also 13.7), w = 1, 2 (4 as extra).
  5. Within-bundle sharing rate of joined and of split pairs at (14.0, 4,300) and (13.7, 4,264).

Reuses ev4_e1_boundaries.py by import (load_patches, prepare, the numba surface kernel); that file
is not modified. Patches 15.14 and 15.15 only; the kill table is filtered by patch first.
Writes only to outputs/reest_exact_v4_20260925/e1/followup.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
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
sys.path.insert(0, str(HERE.parent))
import ev4_e1_boundaries as E1  # noqa: E402  (also puts the worktree first on sys.path)

TB = E1.TB
OUT = E1.OUT / "followup"
PRESPEC = E1.OUT.parent / "records" / "e1_prespec_20260925T084446Z.json"
E1_JSON = E1.OUT / "e1_results.json"
SURF = {"15.14": E1.OUT / "surface_1514.npz", "15.15": E1.OUT / "surface_1515.npz"}

PATCHES = E1.PATCHES
PHASES = E1.PHASES
D_GRID = E1.D_GRID
G_EVAL_MS = np.array([13_700, 14_000], dtype=np.int64)
G_EVAL_S = [13.7, 14.0]
WEIGHTS = (1.0, 2.0, 4.0)
NEAR = 1.01
B_G = 1000
B_D = 1000
SEED = 20260926
WORKERS = 4
CHUNK = 1000
N_INIT = 10
REF_POINTS = [(14.0, 4300.0), (13.7, 4264.0)]


def sha256(p: Path) -> str:
    return E1.sha256(Path(p))


# ------------------------------------------------------------------ D surfaces (per match)
def _chunk_job(args):
    """Worker: per-match joined counts by category on the (G, D) sub-grid, and per-match n by category."""
    t, x, y, cat, ms, me, gms, d2 = args
    _, surface = E1._numba()
    nm = len(ms)
    out = np.zeros((nm, E1.NCAT, len(gms), len(d2)), dtype=np.int32)
    surface(t, x, y, cat, ms, me, gms, d2, E1.NCAT, out)
    ncnt = np.zeros((nm, E1.NCAT), dtype=np.int32)
    for k in range(nm):
        c = cat[ms[k]:me[k]]
        c = c[c >= 0]
        ncnt[k] = np.bincount(c, minlength=E1.NCAT)
    assert out.max() < 32767
    return out.astype(np.int16), ncnt


def run_match_surface(P, gms, dgrid, workers):
    ms_all, me_all = P["starts"], P["ends"]
    idx = np.arange(len(ms_all))
    jobs = []
    for j in range(0, len(idx), CHUNK):
        blk = idx[j:j + CHUNK]
        segs = [np.arange(ms_all[k], me_all[k]) for k in blk]
        kk = np.concatenate(segs)
        lens = np.array([len(s) for s in segs], dtype=np.int64)
        ms = np.r_[0, np.cumsum(lens)[:-1]].astype(np.int64)
        me = (ms + lens).astype(np.int64)
        c = P["cat"][kk].copy()
        c[me - 1] = -1
        jobs.append((P["t"][kk], P["x"][kk], P["y"][kk], c, ms, me,
                     np.asarray(gms, np.int64), np.asarray(dgrid, np.float64) ** 2))
    with get_context("spawn").Pool(workers) as pool:
        res = pool.map(_chunk_job, jobs, chunksize=1)
    J = np.concatenate([r[0] for r in res], axis=0)   # (nm, 6, nG, nD)
    N = np.concatenate([r[1] for r in res], axis=0)   # (nm, 6)
    return J, N


def fp_fn_match(J, N, phases=(0, 1, 2)):
    """Per-match FP and FN over the chosen phases: (nm, nG, nD) int32 each."""
    c0 = [2 * k for k in phases]
    c1 = [2 * k + 1 for k in phases]
    fp = J[:, c0].astype(np.int32).sum(1)
    fn = N[:, c1].astype(np.int32).sum(1)[:, None, None] - J[:, c1].astype(np.int32).sum(1)
    return fp, fn


def band(E, emin):
    near = np.flatnonzero(E <= NEAR * emin)
    contiguous = bool(near.size and (near[-1] - near[0] + 1 == near.size))
    return {"D_range": [float(D_GRID[near[0]]), float(D_GRID[near[-1]])], "n_D": int(near.size), "contiguous": contiguous}


def cond_D(fp, fn, n0, n1):
    """fp, fn: totals over the D grid (1-D). Returns argmin/band per weight."""
    out = {}
    for w in WEIGHTS:
        E = w * fp + fn
        k = int(np.argmin(E))
        emin = float(E[k])
        out[f"w{w:g}"] = {"D_star": float(D_GRID[k]), "E_w_min": emin, "FP": int(fp[k]), "FN": int(fn[k]),
                          "E_w_rate_per_pair": emin / (n0 + n1),
                          "argmin_on_grid_edge": bool(k in (0, len(D_GRID) - 1)),
                          "band_1pct": band(E, emin),
                          "E_w_rel_at_D": {f"{d:.0f}": float(E[int(np.where(D_GRID == d)[0][0])] / emin)
                                           for d in (3300., 3600., 4000., 4300., 4500., 4900., 5100., 5500.)}}
    return out


def boot_D(fpm, fnm, seed):
    """fpm, fnm: (nm, nG, nD) per match. Multinomial match weights, same W for all G and w."""
    nm = fpm.shape[0]
    nG, nD = fpm.shape[1], fpm.shape[2]
    rng = np.random.default_rng(seed)
    W = np.empty((B_D, nm), dtype=np.float32)
    for j in range(0, B_D, 100):
        b = min(100, B_D - j)
        W[j:j + b] = rng.multinomial(nm, np.full(nm, 1.0 / nm), size=b)
    A = fpm.reshape(nm, -1).astype(np.float64)
    C = fnm.reshape(nm, -1).astype(np.float64)
    FP = np.zeros((B_D, nG * nD))
    FN = np.zeros((B_D, nG * nD))
    for j in range(0, nm, 5000):
        Wj = W[:, j:j + 5000].astype(np.float64)
        FP += Wj @ A[j:j + 5000]
        FN += Wj @ C[j:j + 5000]
    FP = FP.reshape(B_D, nG, nD)
    FN = FN.reshape(B_D, nG, nD)
    out = {"B": B_D, "seed": int(seed), "weights": "multinomial(n_matches, 1/n_matches); same replicates for all G and w"}
    for gi, g in enumerate(G_EVAL_S):
        for w in WEIGHTS:
            d = D_GRID[np.argmin(w * FP[:, gi] + FN[:, gi], axis=1)]
            u, c = np.unique(d, return_counts=True)
            out[f"G{g:.1f}_w{w:g}"] = {"D_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                       "D_median": float(np.median(d)),
                                       "D_dist": {f"{a:.0f}": int(b) for a, b in zip(u, c)},
                                       "share_on_grid_edge": float(np.isin(d, [D_GRID[0], D_GRID[-1]]).mean())}
    return out


def within_bundle(P, J, N, gi, di, gms):
    """Sharing rate of joined pairs vs split pairs inside temporal bundles (dt <= G), plus across bundles."""
    U, s, dt = P["U"], P["s"].astype(bool), P["dt"]
    within = U & (dt <= gms)
    js1 = int(J[:, [1, 3, 5], gi, di].astype(np.int64).sum())
    js0 = int(J[:, [0, 2, 4], gi, di].astype(np.int64).sum())
    nw1 = int((within & s).sum())
    nw0 = int((within & ~s).sum())
    na1 = int((U & ~(dt <= gms) & s).sum())
    na0 = int((U & ~(dt <= gms) & ~s).sum())
    sp1, sp0 = nw1 - js1, nw0 - js0
    r = lambda a, b: a / (a + b) if a + b else float("nan")
    by_phase = {}
    for k, name in enumerate(PHASES):
        pk = P["phase"] == k
        j1 = int(J[:, 2 * k + 1, gi, di].astype(np.int64).sum())
        j0 = int(J[:, 2 * k, gi, di].astype(np.int64).sum())
        w1 = int((within & s & pk).sum())
        w0 = int((within & ~s & pk).sum())
        by_phase[name] = {"joined_sharing_rate": r(j1, j0), "split_within_bundle_sharing_rate": r(w1 - j1, w0 - j0),
                          "n_joined": j1 + j0, "n_split_within_bundle": (w1 - j1) + (w0 - j0)}
    return {"n_within_bundle_pairs": nw1 + nw0, "n_joined": js1 + js0, "n_split_within_bundle": sp1 + sp0,
            "n_across_bundles": na1 + na0,
            "joined_sharing_rate": r(js1, js0), "split_within_bundle_sharing_rate": r(sp1, sp0),
            "within_bundle_all_sharing_rate": r(nw1, nw0), "across_bundles_sharing_rate": r(na1, na0),
            "FP": js0, "FN_within_bundle": sp1, "FN_across_bundles": na1, "FN_total": sp1 + na1,
            "by_phase": by_phase}


# ---------------------------------------------------------------------- gap distribution
def gm_all_points(args):
    """Two-component GMM on all log10 dt points with >= 10 starts; best log-likelihood kept."""
    from scipy.optimize import brentq
    from scipy.stats import norm
    from sklearn.mixture import GaussianMixture

    key, x, ref = args
    X = x.reshape(-1, 1)
    fits = []
    t0 = time.perf_counter()
    for init, rs in (("kmeans", 0), ("random_from_data", 1)):
        gm = GaussianMixture(2, n_init=N_INIT, init_params=init, random_state=rs, tol=1e-5, max_iter=1000).fit(X)
        o = np.argsort(gm.means_.ravel())
        mu = gm.means_.ravel()[o]
        sd = np.sqrt(gm.covariances_.ravel()[o])
        w = gm.weights_[o]
        f = lambda v: w[0] * norm.pdf(v, mu[0], sd[0]) - w[1] * norm.pdf(v, mu[1], sd[1])
        cr = brentq(f, mu[0], mu[1]) if f(mu[0]) > 0 > f(mu[1]) else float("nan")
        ll = float(gm.score(X))
        fits.append({"init": init, "n_init": N_INIT, "random_state": rs, "converged": bool(gm.converged_),
                     "n_iter": int(gm.n_iter_), "loglik_mean": ll, "loglik_total": ll * x.size,
                     "means_s": [float(10 ** m) for m in mu], "means_log10": [float(m) for m in mu],
                     "sigmas_log10": [float(s) for s in sd], "weights": [float(v) for v in w],
                     "crossing_s": float(10 ** cr)})
    best = max(fits, key=lambda d: d["loglik_mean"])
    out = {"n_points": int(x.size), "best": best, "fits": fits,
           "fit_s": time.perf_counter() - t0}
    if ref is not None:
        mu_r = np.log10(ref["means_s"])
        s_r = np.array(ref["sigmas_log10"])
        w_r = np.array(ref["weights"])
        ll = np.log(w_r[0] * norm.pdf(x, mu_r[0], s_r[0]) + w_r[1] * norm.pdf(x, mu_r[1], s_r[1]))
        out["E1_reported_fit"] = {"crossing_s": ref["crossing_s"], "means_s": ref["means_s"], "n_fit": ref.get("n_fit"),
                                  "loglik_mean_on_all_points": float(ll.mean()),
                                  "loglik_total_on_all_points": float(ll.sum()),
                                  "delta_total_best_minus_E1": float(best["loglik_mean"] * x.size - ll.sum())}
    return key, out


_BX = {}


def _boot_init(lg, pm, nm):
    _BX["lg"], _BX["pm"], _BX["nm"] = lg, pm, nm


def _valley_boot_job(args):
    patch_id, reps = args
    lg, pm, nm = _BX["lg"], _BX["pm"], _BX["nm"]
    out = []
    for b in reps:
        rng = np.random.default_rng([SEED, patch_id, b])
        cnt = rng.multinomial(nm, np.full(nm, 1.0 / nm))
        x = np.repeat(lg, cnt[pm])
        v = TB.kde_valley(x)
        out.append((b, v.valley_s, bool(v.ok), v.mode1_s, v.mode2_s, v.depth, v.search_lo_s, v.search_hi_s))
    return out


def valley_boot(lg, pm, nm, patch_id, workers):
    reps = np.arange(B_G)
    jobs = [(patch_id, reps[j:j + 10].tolist()) for j in range(0, B_G, 10)]
    t0 = time.perf_counter()
    with get_context("spawn").Pool(workers, initializer=_boot_init, initargs=(lg, pm, nm)) as pool:
        res = [r for chunk in pool.imap_unordered(_valley_boot_job, jobs) for r in chunk]
    res.sort()
    v = np.array([r[1] for r in res], dtype=np.float64)
    ok = np.array([r[2] for r in res])
    vo = v[ok]
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    ci = q(vo) if vo.size else [float("nan")] * 2
    u, c = np.unique(np.round(vo, 3), return_counts=True)
    return {"B": B_G, "seed": [SEED, patch_id, "rep"], "weights": "multinomial(n_matches, 1/n_matches) per replicate",
            "estimator": "TB.kde_valley defaults (bandwidth 0.08, grid -0.5..3.0 log10 x 1401, 1M subsample seed 7, mixture n_init=3 on the subsample)",
            "valley_ok_share": float(ok.mean()), "valley_ci95_s": ci, "valley_median_s": float(np.median(vo)) if vo.size else None,
            "contains_13.7": bool(ci[0] <= 13.7 <= ci[1]), "contains_14.0": bool(ci[0] <= 14.0 <= ci[1]),
            "valley_dist_s": {f"{a:.3f}": int(b) for a, b in zip(u, c)},
            "mode1_ci95_s": q(np.array([r[3] for r in res])[ok]) if vo.size else None,
            "mode2_ci95_s": q(np.array([r[4] for r in res])[ok]) if vo.size else None,
            "depth_ci95": q(np.array([r[5] for r in res])[ok]) if vo.size else None,
            "elapsed_s": time.perf_counter() - t0}


# ------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("time", "full"), default="full")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    workers = min(a.workers, 4)
    OUT.mkdir(parents=True, exist_ok=True)
    T0 = time.perf_counter()
    data = E1.load_patches()
    PP = {p: E1.prepare(data[p]) for p in PATCHES}
    del data
    e1 = json.loads(E1_JSON.read_text(encoding="utf-8"))

    # log10 dt (dt > 0) per patch, with the pair's match (local index) and phase
    GAP = {}
    for p in PATCHES:
        P = PP[p]
        same = P["same"]
        mloc = np.searchsorted(P["starts"], np.arange(len(P["t"]) - 1), side="right") - 1
        dt = P["dt"][same] / 1000.0
        pos = dt > 0
        GAP[p] = {"lg": np.log10(dt[pos]), "pm": mloc[same][pos].astype(np.int64), "ph": P["phase"][same][pos],
                  "nm": int(len(P["starts"])), "n_dt0": int((~pos).sum())}

    if a.stage == "time":
        g = GAP["15.14"]
        t0 = time.perf_counter()
        v = TB.kde_valley(g["lg"])
        print("kde_valley pooled 15.14", v.as_dict(), time.perf_counter() - t0, flush=True)
        t0 = time.perf_counter()
        _boot_init(g["lg"], g["pm"], g["nm"])
        print(_valley_boot_job((0, [0])), time.perf_counter() - t0, flush=True)
        return 0

    res = {"plan": "fallback (pending author approval): G from the kill-gap valley; D = argmin_D (w*FP + FN) at that G",
           "patches": list(PATCHES)}

    # ---- item 3/4/5: per-match FP/FN on G in {13.7, 14.0} x D grid, plus the current rule
    for p in PATCHES:
        P = PP[p]
        t0 = time.perf_counter()
        J, N = run_match_surface(P, G_EVAL_MS, D_GRID, workers)
        Jc, Nc = run_match_surface(P, np.array([13_700]), np.array([4264.0]), workers)
        n_s0 = int(N[:, [0, 2, 4]].sum())
        n_s1 = int(N[:, [1, 3, 5]].sum())
        fpm, fnm = fp_fn_match(J, N)
        FP, FN = fpm.sum(0).astype(np.int64), fnm.sum(0).astype(np.int64)
        # consistency with the saved E1 surface and current rule
        S = np.load(SURF[p])
        chk = {}
        for gi, g in enumerate(G_EVAL_S):
            row = int(np.where(np.isclose(S["G_s"], g))[0][0]) if np.isclose(S["G_s"], g).any() else None
            if row is not None:
                chk[f"G{g:.1f}_equals_E1_surface_FP_FN"] = bool(np.array_equal(FP[gi], S["FP"][row]) and np.array_equal(FN[gi], S["FN"][row]))
        cfp, cfn = fp_fn_match(Jc, Nc)
        cur = e1[p]["current_rule"]
        chk["current_rule_equals_E1"] = bool(int(cfp.sum()) == cur["FP"] and int(cfn.sum()) == cur["FN"])
        chk["n_pairs_U_equals_E1"] = bool(n_s0 + n_s1 == e1[p]["estimate"]["N_pairs_U"])
        cd = {"N_pairs_U": n_s0 + n_s1, "n_sharing": n_s1, "checks": chk}
        for gi, g in enumerate(G_EVAL_S):
            cd[f"G{g:.1f}"] = cond_D(FP[gi], FN[gi], n_s0, n_s1)
        cd["bootstrap"] = boot_D(fpm, fnm, SEED + (0 if p == "15.14" else 1))
        # by phase
        ph = {}
        for k, name in enumerate(PHASES):
            pf, pn = fp_fn_match(J, N, phases=(k,))
            n0, n1 = int(N[:, 2 * k].sum()), int(N[:, 2 * k + 1].sum())
            ph[name] = {"n_pairs": n0 + n1, "sharing_rate": n1 / (n0 + n1)}
            for gi, g in enumerate(G_EVAL_S):
                ph[name][f"G{g:.1f}"] = cond_D(pf.sum(0)[gi].astype(np.int64), pn.sum(0)[gi].astype(np.int64), n0, n1)
        cd["by_phase"] = ph
        # within-bundle sharing
        di43 = int(np.where(D_GRID == 4300.0)[0][0])
        cd["within_bundle_sharing"] = {
            "G14.0_D4300": within_bundle(P, J, N, 1, di43, 14_000),
            "G13.7_D4264": within_bundle(P, Jc, Nc, 0, 0, 13_700),
            "definition": "within bundle = pair in U with dt <= G (consecutive kills, so same temporal bundle); "
                          "joined = same greedy group; split = same bundle, different group; sharing = >= 1 common participant"}
        cd["elapsed_s"] = time.perf_counter() - t0
        np.savez_compressed(OUT / f"match_fp_fn_{p.replace('.', '')}.npz", G_s=np.array(G_EVAL_S), D=D_GRID,
                            FP=fpm.astype(np.int16), FN=fnm.astype(np.int16), n_by_cat=N.astype(np.int16),
                            joined_by_cat=J)
        res.setdefault(p, {})["conditional_D"] = cd
        print(p, "D done", {g: cd[f"G{g:.1f}"]["w2"]["D_star"] for g in G_EVAL_S}, chk, round(cd["elapsed_s"], 1), flush=True)
        del J, N, Jc, Nc, fpm, fnm

    # ---- item 2: mixtures, all points, >= 10 starts
    jobs = []
    for p in PATCHES:
        g = GAP[p]
        for k, name in enumerate(["pooled"] + list(PHASES)):
            sel = np.ones(g["lg"].size, bool) if name == "pooled" else g["ph"] == (k - 1)
            ref = e1[p]["gap_crosscheck"][name]["mixture"]
            jobs.append(((p, name), g["lg"][sel], ref))
    t0 = time.perf_counter()
    with get_context("spawn").Pool(workers) as pool:
        mix = dict(pool.map(gm_all_points, jobs, chunksize=1))
    print("mixtures", round(time.perf_counter() - t0, 1), flush=True)

    # ---- item 1: valley (thesis method), pooled + by phase; sensitivity with the 10-start all-point bounds
    for pid, p in enumerate(PATCHES):
        g = GAP[p]
        gv = {"n_dt0_excluded": g["n_dt0"]}
        for k, name in enumerate(["pooled"] + list(PHASES)):
            sel = np.ones(g["lg"].size, bool) if name == "pooled" else g["ph"] == (k - 1)
            v = TB.kde_valley(g["lg"][sel]).as_dict()
            e1v = e1[p]["gap_crosscheck"][name]["kde_valley"]
            v["reproduces_E1"] = bool((np.isnan(v["valley_s"]) and np.isnan(e1v["valley_s"])) or abs(v["valley_s"] - e1v["valley_s"]) < 1e-9)
            v["valley_exists"] = v["ok"]
            mb = mix[(p, name)]["best"]["means_log10"]
            sens = TB.kde_valley(g["lg"][sel], bounds_log10=(mb[0], mb[1]), max_points=None).as_dict()
            v["sensitivity_all_points_bounds_from_10start_mixture"] = sens
            gv[name] = v
        print(p, "valley point", {n: (gv[n]["valley_s"], gv[n]["ok"]) for n in ["pooled"] + list(PHASES)}, flush=True)
        gv["pooled"]["bootstrap"] = valley_boot(g["lg"], g["pm"], g["nm"], pid, workers)
        print(p, "valley boot", gv["pooled"]["bootstrap"]["valley_ci95_s"], gv["pooled"]["bootstrap"]["valley_ok_share"],
              round(gv["pooled"]["bootstrap"]["elapsed_s"], 1), flush=True)
        res[p]["G_valley"] = gv
        res[p]["mixture_all_points"] = {name: mix[(p, name)] for name in ["pooled"] + list(PHASES)}

    import numba
    import scipy
    import sklearn
    inputs = {str(q): sha256(q) for q in (E1.KILLS, E1.MATCHES, PRESPEC, E1_JSON, SURF["15.14"], SURF["15.15"])}
    mods = {m.__name__: {"file": m.__file__, "sha256": sha256(Path(m.__file__))} for m in (E1, TB, E1.FC, E1.P2B)}
    res["meta"] = {"script": str(HERE), "script_sha256": sha256(HERE), "inputs_sha256": inputs, "modules": mods,
                   "generated_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                   "platform": platform.platform(), "numpy": np.__version__, "numba": numba.__version__,
                   "sklearn": sklearn.__version__, "scipy": scipy.__version__, "workers": workers, "seed": SEED,
                   "D_grid": [float(D_GRID[0]), float(D_GRID[-1]), 100.0], "G_eval_s": G_EVAL_S, "weights_FP": list(WEIGHTS),
                   "band": f"E_w <= {NEAR} * min", "phases": "first kill of the pair: <15, 15-25, >=25 min",
                   "universe": "E1 U: consecutive kills of a match, both positions present, dt >= 0",
                   "valley_bootstrap": f"B={B_G}, multinomial match weights, kde_valley on every replicate",
                   "D_bootstrap": f"B={B_D}, multinomial match weights over per-match FP/FN arrays",
                   "total_wall_s": time.perf_counter() - T0}
    (OUT / "e1_followup.json").write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print("done", round(time.perf_counter() - T0, 1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
