"""Data-driven temporal boundary for kill-episode segmentation.

The detector groups champion kills into episodes when consecutive kills fall
within a gap G.  This module estimates G from the corpus itself instead of
fixing it by hand, and reports how well-defined and how stable that estimate is.

Basis.  On a log scale the distribution of consecutive inter-kill intervals is
bimodal: kills inside one fight follow each other within seconds, while kills of
the next fight are separated by a respawn-and-return cycle of tens of seconds.
The antimode (valley) between the two modes is the natural boundary.  Three
independent readings are provided so the claim does not rest on one estimator:

* ``kde_valley``        -- antimode of a kernel density estimate;
* ``mixture_crossing``  -- posterior-0.5 crossing of a two-component Gaussian
                           mixture on log-intervals;
* ``plateau_ari``       -- how much the resulting clustering changes across gap
                           values (Adjusted Rand Index), i.e. the width of the
                           plateau around the valley.

Uncertainty comes from a cluster bootstrap over matches (matches are the
sampling unit; intervals within a match are not independent).  Stability is
assessed by re-estimating inside strata (patch, game phase, match kill count).

All functions take intervals in seconds; internally everything is log10.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# intervals
# --------------------------------------------------------------------------

def consecutive_intervals(kill_ts_s: Sequence[float]) -> np.ndarray:
    """Positive gaps between time-sorted kills of one match, in seconds."""
    ts = np.sort(np.asarray(kill_ts_s, dtype=np.float64))
    if ts.size < 2:
        return np.empty(0, dtype=np.float64)
    d = np.diff(ts)
    return d[d > 0]


def to_log(dt_s: np.ndarray) -> np.ndarray:
    dt_s = np.asarray(dt_s, dtype=np.float64)
    return np.log10(dt_s[dt_s > 0])


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

MIN_MODE_SEPARATION_LOG10 = 0.30  # mixture components must lie at least a factor of 2 apart in seconds
MAX_VALLEY_DEPTH = 0.95           # density(valley)/min(mode densities) must dip below this


@dataclass
class ValleyEstimate:
    mode1_s: float
    mode2_s: float
    valley_s: float
    depth: float          # density(valley) / min(density(mode1), density(mode2)); < 1 means a real dip
    n: int
    bandwidth: float
    ok: bool              # valley is interior to the two mixture components and depth < MAX_VALLEY_DEPTH
    search_lo_s: float = float("nan")   # mixture component means bounding the search
    search_hi_s: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)


def kde_valley(log_dt: np.ndarray, bandwidth: float = 0.08,
               grid: tuple = (-0.5, 3.0, 1401), bounds_log10: Optional[tuple] = None,
               seed: int = 7) -> ValleyEstimate:
    """Non-parametric antimode of log10-intervals between the two mixture components.

    Two steps.  A two-component Gaussian mixture on log10-intervals locates the
    short-interval (within-fight) and long-interval (between-fight) components;
    then the valley is the minimum of a Gaussian KDE strictly between the two
    component means.  Modes are the KDE maxima on either side of the valley.
    Using the mixture only to *bound* the search keeps the estimate free of
    the mixture's parametric shape while avoiding the fragility of picking
    peaks by height on a noisy density.

    ``bandwidth`` follows scipy's convention: a factor multiplied by the sample
    standard deviation of log10-intervals (0.08 x ~0.6 ~ 0.05 log10 units).
    ``bounds_log10`` overrides the mixture-derived search interval.

    ``ok`` is False when the components are not separated by at least
    ``MIN_MODE_SEPARATION_LOG10``, when the minimum sits at an end of the search
    interval (monotone density, no dip), or when the dip is shallower than
    ``MAX_VALLEY_DEPTH``.
    """
    from scipy.stats import gaussian_kde

    x = np.asarray(log_dt, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    nan = float("nan")
    if n < 50:
        return ValleyEstimate(nan, nan, nan, nan, n, bandwidth, False)

    if bounds_log10 is None:
        mix = mixture_crossing(x, seed=seed)
        if "means_s" not in mix:
            return ValleyEstimate(nan, nan, nan, nan, n, bandwidth, False)
        lo_b, hi_b = np.log10(mix["means_s"][0]), np.log10(mix["means_s"][1])
    else:
        lo_b, hi_b = float(bounds_log10[0]), float(bounds_log10[1])
    if hi_b - lo_b < MIN_MODE_SEPARATION_LOG10:
        return ValleyEstimate(nan, nan, nan, nan, n, bandwidth, False, float(10 ** lo_b), float(10 ** hi_b))

    kde = gaussian_kde(x, bw_method=bandwidth)
    g = np.linspace(*grid)
    f = kde(g)
    # The mixture only decides which side of the distribution each mode lives
    # on: split at the midpoint of the two component means, take the KDE peak
    # on each side, and look for the valley between those peaks.  (Searching
    # only between the component *means* fails when the long-interval
    # component is skewed and its mean sits well below its KDE peak.)
    mid = 0.5 * (lo_b + hi_b)
    left = np.where(g <= mid)[0]
    right = np.where(g > mid)[0]
    if left.size < 3 or right.size < 3:
        return ValleyEstimate(nan, nan, nan, nan, n, bandwidth, False, float(10 ** lo_b), float(10 ** hi_b))
    m1 = left[int(np.argmax(f[left]))]
    m2 = right[int(np.argmax(f[right]))]
    between = np.arange(m1, m2 + 1)
    vi = between[int(np.argmin(f[between]))]
    interior = m1 < vi < m2
    depth = float(f[vi] / max(min(f[m1], f[m2]), 1e-12))
    ok = bool(interior and depth < MAX_VALLEY_DEPTH)
    return ValleyEstimate(float(10 ** g[m1]), float(10 ** g[m2]), float(10 ** g[vi]) if interior else nan,
                          depth, n, bandwidth, ok, float(10 ** lo_b), float(10 ** hi_b))


def mixture_crossing(log_dt: np.ndarray, seed: int = 7) -> dict:
    """Two-component Gaussian mixture on log10-intervals.

    Returns the component means (seconds), weights, and the point between the
    means where the posterior responsibility of the short-interval component
    drops below 0.5 -- the boundary a generative model would draw.
    """
    from sklearn.mixture import GaussianMixture

    x = np.asarray(log_dt, dtype=np.float64)
    x = x[np.isfinite(x)].reshape(-1, 1)
    if x.shape[0] < 50:
        return {"ok": False, "n": int(x.shape[0])}
    gm = GaussianMixture(2, random_state=seed, n_init=3).fit(x)
    order = np.argsort(gm.means_.ravel())
    means = gm.means_.ravel()[order]
    weights = gm.weights_[order]
    sigmas = np.sqrt(gm.covariances_.ravel()[order])
    grid = np.linspace(means[0], means[1], 2000).reshape(-1, 1)
    post = gm.predict_proba(grid)[:, order[0]]
    below = np.where(post < 0.5)[0]
    crossing = float(10 ** grid[below[0], 0]) if below.size else float("nan")
    return {"ok": bool(below.size > 0), "n": int(x.shape[0]),
            "means_s": [float(10 ** m) for m in means], "sigmas_log10": [float(s) for s in sigmas],
            "weights": [float(w) for w in weights], "crossing_s": crossing}


# --------------------------------------------------------------------------
# clustering agreement across gap values
# --------------------------------------------------------------------------

def temporal_clusters(kill_ts_s: Sequence[float], gap_s: float) -> np.ndarray:
    """Single-linkage temporal clustering: label per kill (time order)."""
    ts = np.asarray(kill_ts_s, dtype=np.float64)
    if ts.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(ts, kind="stable")
    s = ts[order]
    new = np.concatenate([[True], np.diff(s) > gap_s])
    lab_sorted = np.cumsum(new) - 1
    labels = np.empty_like(lab_sorted)
    labels[order] = lab_sorted
    return labels


def plateau_ari(per_match_ts: Iterable[Sequence[float]], gaps_s: Sequence[float],
                ref_gap_s: float) -> Dict[str, float]:
    """Adjusted Rand Index between the clustering at each gap and at ``ref_gap_s``.

    Matches are concatenated with disjoint label offsets so cross-match pairs
    never count as co-clustered.
    """
    from sklearn.metrics import adjusted_rand_score

    per_match_ts = [np.asarray(t, dtype=np.float64) for t in per_match_ts]
    ref_all, offset = [], 0
    for t in per_match_ts:
        lab = temporal_clusters(t, ref_gap_s)
        ref_all.append(lab + offset)
        offset += int(lab.max()) + 1 if lab.size else 0
    ref_all = np.concatenate(ref_all) if ref_all else np.empty(0, dtype=np.int64)
    out: Dict[str, float] = {}
    for g in gaps_s:
        cand, offset = [], 0
        for t in per_match_ts:
            lab = temporal_clusters(t, g)
            cand.append(lab + offset)
            offset += int(lab.max()) + 1 if lab.size else 0
        cand = np.concatenate(cand) if cand else np.empty(0, dtype=np.int64)
        out[f"{g:g}"] = float(adjusted_rand_score(ref_all, cand)) if ref_all.size else float("nan")
    return out


def episode_counts(per_match_ts: Iterable[Sequence[float]], gaps_s: Sequence[float]) -> Dict[str, float]:
    """Mean number of temporal episodes per match at each gap."""
    per_match_ts = [np.asarray(t, dtype=np.float64) for t in per_match_ts]
    out = {}
    for g in gaps_s:
        counts = [int(temporal_clusters(t, g).max()) + 1 if t.size else 0 for t in per_match_ts]
        out[f"{g:g}"] = float(np.mean(counts)) if counts else float("nan")
    return out


# --------------------------------------------------------------------------
# uncertainty and stability
# --------------------------------------------------------------------------

def cluster_bootstrap(per_match_logdt: Sequence[np.ndarray], n_boot: int = 200, seed: int = 7,
                      bandwidth: float = 0.08, max_pairs: Optional[int] = 100_000) -> dict:
    """Bootstrap the KDE valley by resampling matches with replacement.

    Returns percentile intervals for the valley, both modes and the depth, plus
    the share of replicates in which a valley was found at all.
    """
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(a, dtype=np.float64) for a in per_match_logdt if len(a) > 0]
    m = len(arrs)
    if m == 0:
        return {"ok": False}
    valleys, m1, m2, depths, found = [], [], [], [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, m, m)
        x = np.concatenate([arrs[i] for i in idx])
        if max_pairs is not None and x.size > max_pairs:
            x = rng.choice(x, max_pairs, replace=False)
        est = kde_valley(x, bandwidth=bandwidth)
        if est.ok:
            found += 1
            valleys.append(est.valley_s); m1.append(est.mode1_s); m2.append(est.mode2_s); depths.append(est.depth)

    def pct(v):
        v = np.asarray(v, dtype=np.float64)
        if v.size == 0:
            return None
        return {"p2.5": float(np.percentile(v, 2.5)), "p50": float(np.percentile(v, 50)),
                "p97.5": float(np.percentile(v, 97.5)), "mean": float(v.mean())}

    return {"ok": True, "n_boot": int(n_boot), "n_matches": int(m), "valley_found_share": found / n_boot,
            "valley_s": pct(valleys), "mode1_s": pct(m1), "mode2_s": pct(m2), "depth": pct(depths)}


def stratified_valleys(records: Sequence[dict], key: Callable[[dict], str],
                       bandwidth: float = 0.08, min_pairs: int = 2000) -> Dict[str, dict]:
    """Re-estimate the valley inside strata.

    ``records`` are dicts with at least ``log_dt`` (array).  ``key`` maps a
    record to its stratum label.  Strata with fewer than ``min_pairs``
    intervals are reported but flagged.
    """
    groups: Dict[str, List[np.ndarray]] = {}
    for r in records:
        groups.setdefault(str(key(r)), []).append(np.asarray(r["log_dt"], dtype=np.float64))
    out = {}
    for k in sorted(groups):
        x = np.concatenate(groups[k]) if groups[k] else np.empty(0)
        est = kde_valley(x, bandwidth=bandwidth)
        d = est.as_dict()
        d["enough"] = bool(x.size >= min_pairs)
        d["n_matches"] = len(groups[k])
        out[k] = d
    return out


def bandwidth_sweep(log_dt: np.ndarray, bandwidths: Sequence[float] = (0.05, 0.08, 0.12, 0.16)) -> Dict[str, dict]:
    """The valley must not be an artefact of one smoothing choice."""
    return {f"{bw:g}": kde_valley(log_dt, bandwidth=bw).as_dict() for bw in bandwidths}
