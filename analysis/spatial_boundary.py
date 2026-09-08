"""Data-driven spatial boundary for kill-episode segmentation.

Ground-truth proxy.  Two consecutive kills that involve the same champion
(as killer, victim or assister) belong to the same fight by construction, and
a champion cannot be in two places at once, so within a short time window the
*sharing rate* as a function of distance traces where "same fight" ends.  The
boundary D is the distance at which the sharing rate crosses 0.5 -- the Bayes
decision point that misclassifies the fewest pairs.  Uncertainty comes from a
match-level bootstrap; the same estimate is repeated per map region and, in
lanes, along versus across the lane direction.

All distances in game units; time gaps in seconds.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from analysis.map_regions import REGIONS, along_across

DIST_EDGES = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 8000, 12000, 1e9])


def sharing_curve(dd: np.ndarray, sh: np.ndarray, edges: np.ndarray = DIST_EDGES, min_n: int = 40):
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dd > lo) & (dd <= hi)
        if m.sum() >= min_n:
            xs.append(float((lo + min(hi, 14000)) / 2)); ys.append(float(sh[m].mean())); ns.append(int(m.sum()))
    return list(zip(xs, ys, ns))


def crossover(dd: np.ndarray, sh: np.ndarray, level: float = 0.5, min_n: int = 40) -> Tuple[Optional[float], list]:
    """Distance where the binned sharing rate first drops below ``level``.

    Linear interpolation between bin centres.  None when the rate never drops
    below the level inside the populated bins (reported as "> last bin").
    """
    curve = sharing_curve(dd, sh, min_n=min_n)
    if len(curve) < 2 or curve[0][1] < level:
        return None, curve
    for (x0, y0, _), (x1, y1, _) in zip(curve[:-1], curve[1:]):
        if y0 >= level > y1:
            return float(x0 + (y0 - level) / max(y0 - y1, 1e-9) * (x1 - x0)), curve
    return None, curve


def bootstrap_crossover(dd: np.ndarray, sh: np.ndarray, match_idx: np.ndarray, n_boot: int = 200, seed: int = 7) -> Optional[dict]:
    """Match-level bootstrap of the crossover.

    Pairs are grouped by match through one argsort; each replicate resamples matches
    with replacement and gathers their pair index ranges (no per-match searches).
    """
    rng = np.random.default_rng(seed)
    if match_idx.size == 0:
        return None
    order = np.argsort(match_idx, kind="stable")
    mi_sorted = match_idx[order]
    uniq, starts = np.unique(mi_sorted, return_index=True)
    ends = np.append(starts[1:], mi_sorted.size)
    dd_s, sh_s = dd[order], sh[order]
    m = uniq.size
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, m, m)
        lengths = ends[pick] - starts[pick]
        idx = np.repeat(starts[pick] - np.cumsum(np.concatenate([[0], lengths[:-1]])), lengths) + np.arange(lengths.sum())
        c, _ = crossover(dd_s[idx], sh_s[idx])
        if c is not None:
            vals.append(c)
    if not vals:
        return None
    v = np.asarray(vals)
    return {"p2.5": float(np.percentile(v, 2.5)), "p50": float(np.percentile(v, 50)),
            "p97.5": float(np.percentile(v, 97.5)), "found": len(vals) / n_boot, "n_boot": int(n_boot)}


def mass_near_boundary(dd: np.ndarray, D: float, tol: float = 0.10) -> float:
    """Share of in-window pairs within +-tol of D: how much a small change in D moves."""
    if dd.size == 0 or not np.isfinite(D):
        return float("nan")
    return float(np.mean((dd >= (1 - tol) * D) & (dd <= (1 + tol) * D)))


def by_region(dd, sh, region1, match_idx, n_boot: int = 200, seed: int = 7, min_n: int = 200) -> Dict[str, dict]:
    out = {}
    for reg in REGIONS:
        m = region1 == reg
        if m.sum() < min_n:
            continue
        cx, curve = crossover(dd[m], sh[m])
        out[reg] = {"n": int(m.sum()), "crossover_u": cx,
                    "bootstrap": bootstrap_crossover(dd[m], sh[m], match_idx[m], n_boot=n_boot, seed=seed),
                    "share_le_4000": float(np.mean(dd[m] <= 4000)),
                    "shared_rate_le_4000": float(sh[m & (dd <= 4000)].mean()) if (m & (dd <= 4000)).any() else None,
                    "shared_rate_gt_4000": float(sh[m & (dd > 4000)].mean()) if (m & (dd > 4000)).any() else None,
                    "curve": curve}
    return out


def lane_anisotropy(disp, tangent, sh, region1, across_cap: float = 500.0, along_cap: float = 1000.0,
                    min_n: int = 200) -> Dict[str, dict]:
    """Conditional crossovers: along the lane (holding across <= cap) and across (holding along <= cap)."""
    out = {}
    for reg in ("top", "mid", "bot"):
        m = (region1 == reg) & (np.abs(tangent).sum(axis=1) > 0)
        if m.sum() < min_n:
            continue
        along, across = along_across(disp[m], tangent[m])
        s = sh[m]
        rec = {"n": int(m.sum())}
        for name, comp, other, cap in (("along", along, across, across_cap), ("across", across, along, along_cap)):
            keep = other <= cap
            cx, curve = crossover(comp[keep], s[keep])
            rec[name] = {"crossover_u": cx, "n": int(keep.sum()), "conditioned_on_other_le_u": cap, "curve": curve,
                         "last_bin": {"u": curve[-1][0], "shared": curve[-1][1]} if curve else None}
        out[reg] = rec
    return out


def region_transitions(dd, sh, region1, region2) -> dict:
    same = region1 == region2
    return {"same_region_share": float(np.mean(same)) if same.size else float("nan"),
            "shared_rate_same_region": float(sh[same].mean()) if same.any() else None,
            "shared_rate_diff_region": float(sh[~same].mean()) if (~same).any() else None,
            "shared_rate_diff_region_le_4000": float(sh[~same & (dd <= 4000)].mean()) if (~same & (dd <= 4000)).any() else None}


def estimate_spatial(pairs: Dict[str, np.ndarray], gap_s: float, n_boot: int = 200, seed: int = 7) -> dict:
    """Everything the spatial side reports, for pairs inside the temporal window."""
    dt, dd, sh = pairs["dt"], pairs["dd"], pairs["shared"].astype(bool)
    win = (dt <= gap_s) & np.isfinite(dd)
    if win.sum() < 200:
        return {"ok": False, "n_pairs_in_window": int(win.sum())}
    dd_w, sh_w, mi_w = dd[win], sh[win], pairs["match_idx"][win]
    r1, r2 = pairs["region1"].astype(str)[win], pairs["region2"].astype(str)[win]
    tan = pairs["tangent1"][win]
    disp = np.stack([pairs["x2"][win] - pairs["x1"][win], pairs["y2"][win] - pairs["y1"][win]], axis=1)
    cx, curve = crossover(dd_w, sh_w)
    boot = bootstrap_crossover(dd_w, sh_w, mi_w, n_boot=n_boot, seed=seed)
    return {"ok": cx is not None, "gap_s": float(gap_s), "n_pairs_in_window": int(win.sum()),
            "crossover_u": cx, "bootstrap": boot, "curve": curve,
            "mass_near_boundary": mass_near_boundary(dd_w, cx) if cx else None,
            "share_le_2000": float(np.mean(dd_w <= 2000)), "share_le_4000": float(np.mean(dd_w <= 4000)),
            "by_region": by_region(dd_w, sh_w, r1, mi_w, n_boot=max(20, n_boot // 4), seed=seed),
            "lane_anisotropy": lane_anisotropy(disp, tan, sh_w, r1),
            "region_transition": region_transitions(dd_w, sh_w, r1, r2)}
