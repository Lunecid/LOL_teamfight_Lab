"""v4-exact V (base win-probability) models: candidates, team swap, calibration, metrics, frozen bundle + predict API.

Used by scripts/exact_v4/ev4_03_fit_v.py (fit / selection, R5) and by the later stages (ev4_04 labels etc.), which
first call ``assert_v_usable(frozen_manifest)`` (refuses a run whose stop rule says stop, a pilot, a smoke run unless
allowed, or a run without a frozen V), then load the frozen V with ``load_v(bundle_dir, expected_sha256)`` and call
``V.predict(X)`` on a StateV3 matrix (996 columns, gameplay.state_value_v3.STATE_V3_COLUMNS order; snapshot_age_s
is a V input) -> P(blue wins).

Everything here is patch-agnostic (a matrix in, probabilities out).  The split guard lives in the calling scripts.

Candidates (FIXED grids, plan section 4; tuned only by 15.14 match-grouped 5-fold CV in ev4_03):
  logistic  standardised inputs (pooled over blue/red swap pairs), L2, C in {0.001, 0.01, 0.1}
            (sklearn convention: minimise C * sum(logloss) + ||w||^2 / 2, intercept unpenalised), L-BFGS on the
            float32 matrix without a standardised copy (centering / scaling folded into the weights).
  lgbm      LightGBM binary, num_leaves in {31, 63}; learning_rate 0.05, min_data_in_leaf 200, feature_fraction 0.8,
            max_bin 255, max 3,000 rounds, early stopping (100 rounds) on the inner fold, deterministic, 4 threads.
  mlp       shared-player MLP: one encoder shared by the 10 player blocks (72 fields -> 64 -> 64, GELU), outputs
            concatenated in role-slot order (blue TOP..UTILITY, red TOP..UTILITY) with the 276 team / global
            columns -> 256 -> 128 -> 1 (GELU).  AdamW lr 1e-3, batch 1,024, weight decay in {1e-4, 1e-3}, max 20
            epochs, early stopping (patience 3 epochs) on the inner fold, 3 seeds with logits averaged.
            Inputs standardised with statistics pooled over the 10 slots per player field and over blue/red pairs
            for the team columns (so the standardisation is swap-equivariant).
Team symmetry: logistic and LightGBM train on the stored rows with a seeded random 50 % of them team-swapped
(blue <-> red columns, target flipped) instead of adding copies; the MLP swaps a fresh random 50 % of each minibatch.
Calibration: PositiveSlopeSigmoid (copied unchanged from C:/Users/todtj/문서/LOL_Teamfight/scripts/fc20260915_common.py:
p_cal = expit(a + b logit(clip(p, 1e-12))), b >= 1e-6, L-BFGS-B), fitted on V_CAL.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + MKL OpenMP clash on Windows/conda (repo convention)

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

MAX_THREADS = 4
SEED = 20260925
SIMPLICITY_ORDER: Tuple[str, ...] = ("logistic", "lgbm", "mlp")
SELECT_TOL = 0.0005
V_CLIP = 1e-12                                   # fc20260915_common.V_CLIP
MINUTE_BINS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("lt15", 0, 900_000), ("15to25", 900_000, 1_500_000), ("ge25", 1_500_000, None))
BUNDLE_FORMAT = "ev4_v_bundle_1"

GRIDS: Dict[str, Dict[str, Any]] = {
    "logistic": {"C": [0.001, 0.01, 0.1], "fixed": {"penalty": "l2", "standardise": "pooled swap pairs",
                                                   "solver": "scipy L-BFGS-B", "maxiter": 1000, "gtol": 1e-6,
                                                   "ftol": 1e-12}},
    "lgbm": {"num_leaves": [31, 63],
             "fixed": {"objective": "binary", "learning_rate": 0.05, "min_data_in_leaf": 200, "feature_fraction": 0.8,
                       "max_bin": 255, "max_rounds": 3000, "early_stopping_rounds": 100, "num_threads": 4,
                       "deterministic": True}},
    "mlp": {"weight_decay": [1e-4, 1e-3],
            "fixed": {"encoder": [72, 64, 64], "head": [256, 128, 1], "activation": "GELU", "optimizer": "AdamW",
                      "lr": 1e-3, "batch": 1024, "max_epochs": 20, "patience": 3, "seeds_final": [0, 1, 2],
                      "seeds_cv": [0], "threads": 4}},
}
LGBM_FIXED: Dict[str, Any] = dict(objective="binary", learning_rate=0.05, min_data_in_leaf=200, feature_fraction=0.8,
                                  max_bin=255, deterministic=True, force_col_wise=True, verbosity=-1, seed=SEED,
                                  feature_fraction_seed=SEED, bagging_seed=SEED, data_random_seed=SEED,
                                  metric="binary_logloss", num_threads=MAX_THREADS)
LGBM_MAX_ROUNDS = 3000
LGBM_PATIENCE = 100
MLP_MAX_EPOCHS = 20
MLP_PATIENCE = 3
MLP_BATCH = 1024
MLP_LR = 1e-3
MLP_SEEDS_FINAL = (0, 1, 2)
MLP_SEEDS_CV = (0,)
ENC_DIM = 64
HEAD = (256, 128)


# ============================================================================ hashing / splits
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_int(key: str) -> int:
    """Deterministic 64-bit integer from sha256(key) (same rule as ev4_02_extract.hash_int)."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)


def cv_fold(mid: str, k: int = 5) -> int:
    return hash_int(f"{mid}:ev4_03:cv{k}") % k


def inner_holdout(mid: str) -> bool:
    """10 % of matches (by sha256) are the early-stopping 'inner fold' of any training set."""
    return hash_int(f"{mid}:ev4_03:inner") % 10 == 0


def v_split(mid: str) -> str:
    """15.15 V rows: 50 % of matches V_SELECT, 50 % V_CAL (by sha256)."""
    return "select" if hash_int(f"{mid}:ev4_03:vsplit") % 2 == 0 else "cal"


def oos_half(mid: str) -> int:
    return hash_int(f"{mid}:ev4_03:mart_oos") % 2


def per_match(mids: Sequence[str], fn: Callable[[str], Any]) -> np.ndarray:
    cache: Dict[str, Any] = {}
    out = []
    for m in mids:
        if m not in cache:
            cache[m] = fn(m)
        out.append(cache[m])
    return np.asarray(out)


def savez_deterministic(path: Path, **arrays: np.ndarray) -> None:
    """np.savez equivalent (stored, uncompressed) with fixed zip timestamps, so equal arrays give equal bytes."""
    import io
    import zipfile
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arrays[name]), allow_pickle=False)
            zi = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.external_attr = 0o600 << 16
            zf.writestr(zi, buf.getvalue())


# ============================================================================ team swap
_SWAP_PREFIXES = (("obj_blue_", "obj_red_"), ("obj_red_", "obj_blue_"), ("blue_", "red_"), ("red_", "blue_"))


def swap_name(n: str) -> str:
    for a, b in _SWAP_PREFIXES:
        if n.startswith(a):
            return b + n[len(a):]
    return n


def swap_permutation(names: Sequence[str]) -> np.ndarray:
    """perm with X_swapped = X[:, perm]: every blue column takes its red partner and vice versa.

    Raises if a partner is missing, if a column carries a 'blue' / 'red' token without being mapped, or if the
    map is not an involution."""
    idx = {n: i for i, n in enumerate(names)}
    if len(idx) != len(names):
        raise ValueError("duplicate column names")
    perm = np.empty(len(names), dtype=np.int64)
    for j, n in enumerate(names):
        m = swap_name(n)
        if m not in idx:
            raise ValueError(f"swap partner {m!r} of {n!r} missing")
        toks = n.split("_")
        if m == n and ("blue" in toks or "red" in toks):
            raise ValueError(f"team token in an unmapped column {n!r}")
        perm[j] = idx[m]
    if not np.array_equal(perm[perm], np.arange(len(names))):
        raise ValueError("swap map is not an involution")
    return perm


def swap_pairs(perm: np.ndarray) -> List[np.ndarray]:
    return [np.array([j, int(perm[j])]) for j in range(len(perm)) if perm[j] > j]


def swap_mask(n: int, seed: int = SEED) -> np.ndarray:
    """Seeded random 50 % of rows (Bernoulli 0.5 per row, numpy default_rng(seed))."""
    return np.random.default_rng(seed).random(n) < 0.5


def apply_swap_inplace(X: np.ndarray, y: np.ndarray, mask: np.ndarray, perm: np.ndarray, chunk: int = 65536) -> None:
    """Swap the team blocks of rows[mask] and flip their target, in place (an involution: apply twice to undo)."""
    rows = np.flatnonzero(mask)
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        X[r] = X[r][:, perm]
    y[rows] = 1 - y[rows]


# ============================================================================ preprocessing
def nan_fill_values(X: np.ndarray, chunk: int = 50_000) -> Tuple[np.ndarray, int]:
    """Column nan-means (0 for all-NaN columns) and the NaN count."""
    p = X.shape[1]
    s = np.zeros(p)
    c = np.zeros(p)
    nn = 0
    for a in range(0, X.shape[0], chunk):
        xb = X[a:a + chunk].astype(np.float64)
        m = np.isnan(xb)
        nn += int(m.sum())
        s += np.where(m, 0.0, xb).sum(0)
        c += (~m).sum(0)
    fill = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    return fill, nn


def fill_nan_inplace(X: np.ndarray, fill: np.ndarray, chunk: int = 50_000) -> int:
    n = 0
    for a in range(0, X.shape[0], chunk):
        xb = X[a:a + chunk]
        m = np.isnan(xb)
        if m.any():
            n += int(m.sum())
            xb[m] = np.broadcast_to(fill.astype(X.dtype), xb.shape)[m]
    return n


def pooled_standardizer(X: np.ndarray, rows: Optional[np.ndarray], groups: Sequence[np.ndarray],
                        chunk: int = 50_000, sd_floor: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """(mu, sd) over `rows` (all rows if None), with first and second moments pooled within each group of columns.

    sd < sd_floor -> 1 (constant column: standardised value 0)."""
    p = X.shape[1]
    idx = np.arange(X.shape[0]) if rows is None else np.asarray(rows)
    s1 = np.zeros(p)
    s2 = np.zeros(p)
    for a in range(0, len(idx), chunk):
        r = idx[a:a + chunk]
        xb = X[r].astype(np.float64) if rows is not None else X[a:a + chunk].astype(np.float64)
        s1 += xb.sum(0)
        s2 += (xb * xb).sum(0)
    n = max(len(idx), 1)
    m1, m2 = s1 / n, s2 / n
    mu, ex2 = m1.copy(), m2.copy()
    for g in groups:
        g = np.asarray(g)
        mu[g] = m1[g].mean()
        ex2[g] = m2[g].mean()
    sd = np.sqrt(np.maximum(ex2 - mu ** 2, 0.0))
    sd[sd < sd_floor] = 1.0
    return mu, sd


def player_layout(names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(slot_idx (10, n_fields) column indices in role-slot order, other_idx, field names)."""
    from gameplay.state_value_v3 import PLAYER_FIELDS, SLOT_PREFIXES
    idx = {n: i for i, n in enumerate(names)}
    slot_idx = np.array([[idx[pre + f] for f in PLAYER_FIELDS] for pre in SLOT_PREFIXES], dtype=np.int64)
    used = set(slot_idx.ravel().tolist())
    other = np.array([i for i in range(len(names)) if i not in used], dtype=np.int64)
    return slot_idx, other, list(PLAYER_FIELDS)


def mlp_groups(names: Sequence[str], perm: np.ndarray) -> List[np.ndarray]:
    slot_idx, other, _ = player_layout(names)
    groups = [slot_idx[:, f] for f in range(slot_idx.shape[1])]
    oset = set(other.tolist())
    groups += [g for g in swap_pairs(perm) if int(g[0]) in oset]
    return groups


# ============================================================================ calibration (copied)
def clipped_logit(p, eps=V_CLIP):
    """Numeric transform only: clip to [eps, 1-eps] then log(p) - log1p(-p)."""
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(q) - np.log1p(-q)


def _expit(z):
    from scipy.special import expit
    return expit(z)


class PositiveSlopeSigmoid:
    """p_cal = expit(a + b * logit(clip(p, 1e-12))) with b constrained > 0 (L-BFGS-B bound b >= SLOPE_MIN).

    Copied unchanged from C:/Users/todtj/문서/LOL_Teamfight/scripts/fc20260915_common.py (class
    PositiveSlopeSigmoid) plus to_dict / from_dict.
    """
    kind = 'sigmoid_pos'
    SLOPE_MIN = 1e-6

    def __init__(self):
        self.a = None
        self.b = None
        self.fit_info = None

    def fit(self, p, y, w):
        from scipy.optimize import minimize
        z = clipped_logit(p)
        y = np.asarray(y, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        sw = w.sum()

        def nll(theta):
            a, b = theta
            eta = a + b * z
            val = np.sum(w * (np.logaddexp(0.0, eta) - y * eta)) / sw
            mu = _expit(eta)
            r = w * (mu - y) / sw
            return val, np.array([r.sum(), np.dot(r, z)])

        res = minimize(nll, x0=np.array([0.0, 1.0]), jac=True, method='L-BFGS-B',
                       bounds=[(None, None), (self.SLOPE_MIN, None)],
                       options=dict(maxiter=1000, ftol=1e-15, gtol=1e-10))
        self.a, self.b = float(res.x[0]), float(res.x[1])
        _, grad = nll(res.x)
        self.fit_info = dict(optimizer='scipy.optimize.minimize L-BFGS-B', success=bool(res.success),
                             message=str(res.message), nit=int(res.nit), objective=float(res.fun),
                             intercept_a=self.a, slope_b=self.b, slope_lower_bound=self.SLOPE_MIN,
                             slope_bound_active=bool(self.b <= self.SLOPE_MIN * (1 + 1e-9)),
                             final_gradient=grad.tolist(), n=int(len(y)), weight_sum=float(sw),
                             logit_clip=V_CLIP)
        if not np.isfinite(res.x).all():
            raise ValueError('non-finite sigmoid calibration parameters')
        return self

    def predict(self, p):
        return _expit(self.a + self.b * clipped_logit(p))

    def describe(self):
        return dict(kind='sigmoid_pos', intercept_a=self.a, slope_b=self.b, fit=self.fit_info)

    def to_dict(self) -> Dict[str, Any]:
        return self.describe()

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PositiveSlopeSigmoid":
        c = cls()
        c.a, c.b, c.fit_info = float(d["intercept_a"]), float(d["slope_b"]), d.get("fit")
        return c


# ============================================================================ metrics
def log_loss(y, p) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), V_CLIP, 1 - V_CLIP)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def row_log_loss(y, p) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), V_CLIP, 1 - V_CLIP)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def _logit_fit(y: np.ndarray, z: np.ndarray, offset_only: bool = False) -> Tuple[float, float]:
    """ML fit of y ~ a + b z (or y ~ a + 1 * z when offset_only) by damped Newton (step halving until the
    log-likelihood does not decrease).  Returns (nan, nan) when the slope is not identified (z nearly constant),
    the fit does not converge in 100 steps, or |b| > 100 (quasi-separation)."""
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    nan = (float("nan"), float("nan"))
    if not offset_only and np.std(z) < 1e-8:
        return nan

    def ll(a, b):
        eta = a + b * z
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    a, b = 0.0, 1.0
    cur = ll(a, b)
    for _ in range(100):
        eta = a + b * z
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        wv = np.maximum(mu * (1 - mu), 1e-12)
        r = y - mu
        if offset_only:
            step = np.array([r.sum() / wv.sum(), 0.0])
        else:
            H = np.array([[wv.sum(), (wv * z).sum()], [(wv * z).sum(), (wv * z * z).sum()]])
            g = np.array([r.sum(), (r * z).sum()])
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                return nan
        t = 1.0
        while t > 1e-8:
            new = ll(a + t * step[0], b + t * step[1])
            if new >= cur - 1e-12:
                break
            t *= 0.5
        a, b, cur = a + t * step[0], b + t * step[1], new
        if np.max(np.abs(t * step)) < 1e-10:
            return (float(a), float(b)) if abs(b) <= 100 else nan
    return nan


def binary_metrics(y, p) -> Dict[str, Any]:
    """log loss, Brier, AUC, calibration slope / intercept (y ~ a + b logit p) and calibration-in-the-large
    (y ~ a + offset logit p)."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    out: Dict[str, Any] = {"n": int(len(y))}
    if not len(y):
        return out
    out["log_loss"] = log_loss(y, p)
    out["brier"] = float(np.mean((p - y) ** 2))
    out["mean_p"] = float(p.mean())
    out["mean_y"] = float(y.mean())
    if 0 < y.sum() < len(y):
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(y, p))
        z = clipped_logit(p)
        a, b = _logit_fit(y, z)
        out["cal_intercept"], out["cal_slope"] = a, b
        out["cal_in_the_large"] = _logit_fit(y, z, offset_only=True)[0]
    else:
        out["auc"] = out["cal_intercept"] = out["cal_slope"] = out["cal_in_the_large"] = None
    return out


def metrics_by_bins(y, p, t_ms) -> Dict[str, Any]:
    t_ms = np.asarray(t_ms)
    out = {"all": binary_metrics(y, p)}
    for name, lo, hi in MINUTE_BINS:
        m = (t_ms >= lo) & ((t_ms < hi) if hi is not None else True)
        out[name] = binary_metrics(np.asarray(y)[m], np.asarray(p)[m])
    return out


def cluster_mean_se(v: np.ndarray, clusters: Sequence[str]) -> Tuple[float, float, int]:
    """Mean of v with a cluster-robust (CR1) standard error."""
    v = np.asarray(v, dtype=np.float64)
    codes, inv = np.unique(np.asarray(clusters), return_inverse=True)
    G = len(codes)
    n = len(v)
    m = float(v.mean())
    sums = np.bincount(inv, weights=v - m, minlength=G)
    se = float(math.sqrt(G / max(G - 1, 1) * np.sum(sums ** 2)) / n) if G > 1 else float("nan")
    return m, se, G


def paired_logloss_delta(y, p_a, p_b, clusters) -> Dict[str, Any]:
    """mean(logloss_a - logloss_b) with a match-cluster SE."""
    d = row_log_loss(y, p_a) - row_log_loss(y, p_b)
    m, se, G = cluster_mean_se(d, clusters)
    return {"delta": m, "se_cluster": se, "n_clusters": G, "ci95": [m - 1.96 * se, m + 1.96 * se]}


# ============================================================================ selection rule
def select_candidate(losses: Mapping[str, float], tol: float = SELECT_TOL,
                     order: Sequence[str] = SIMPLICITY_ORDER) -> Dict[str, Any]:
    """Lowest V_SELECT log loss; a simpler candidate (order logistic < lgbm < mlp) whose loss is within `tol` of the
    best is taken instead.  Returns the choice and the reasoning."""
    known = [c for c in order if c in losses]
    extra = sorted(set(losses) - set(order))
    if extra:
        raise ValueError(f"unknown candidates {extra}")
    if not known:
        raise ValueError("no candidates")
    vals = {c: float(losses[c]) for c in known}
    if not all(math.isfinite(v) for v in vals.values()):
        raise ValueError(f"non-finite loss: {vals}")
    best_c = min(known, key=lambda c: (vals[c], order.index(c)))
    best = vals[best_c]
    chosen = next(c for c in known if vals[c] <= best + tol + 1e-12)
    return {"chosen": chosen, "best": best_c, "best_loss": best, "tol": tol, "losses": vals,
            "within_tol": [c for c in known if vals[c] <= best + tol + 1e-12], "order": list(order),
            "simpler_taken": chosen != best_c}


# ============================================================================ logistic
class LogisticModel:
    kind = "logistic"

    def __init__(self, C: float):
        self.C = float(C)
        self.w: Optional[np.ndarray] = None   # coefficients on standardised inputs
        self.b: float = 0.0
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.fit_info: Dict[str, Any] = {}

    def _v_off(self) -> Tuple[np.ndarray, float]:
        v = (self.w / self.sd).astype(np.float32)
        off = float(self.b - np.dot(self.mu / self.sd, self.w))
        return v, off

    def fit(self, X: np.ndarray, y: np.ndarray, rows: np.ndarray, mu: np.ndarray, sd: np.ndarray,
            maxiter: int = 1000, gtol: float = 1e-6, ftol: float = 1e-12) -> "LogisticModel":
        """Fit on X[rows] without copying X: the loss runs over all rows with 0/1 weights."""
        from scipy.optimize import minimize
        from scipy.special import expit
        n, p = X.shape
        wt = np.zeros(n)
        wt[np.asarray(rows)] = 1.0
        N = wt.sum()
        if N <= 0:
            raise ValueError("no training rows")
        yf = np.asarray(y, dtype=np.float64)
        self.mu, self.sd = np.asarray(mu, np.float64), np.asarray(sd, np.float64)
        CN = self.C * N
        musd = self.mu / self.sd
        ncall = [0]

        def f(theta):
            ncall[0] += 1
            w, b = theta[:p], theta[p]
            v = (w / self.sd).astype(np.float32)
            eta = (X @ v).astype(np.float64) + (b - float(np.dot(musd, w)))
            loss = float(np.dot(wt, np.logaddexp(0.0, eta) - yf * eta)) / N + float(np.dot(w, w)) / (2 * CN)
            r = wt * (expit(eta) - yf) / N
            rs = float(r.sum())
            gx = (X.T @ r.astype(np.float32)).astype(np.float64)
            gw = (gx - self.mu * rs) / self.sd + w / CN
            return loss, np.concatenate([gw, [rs]])

        t0 = time.time()
        res = minimize(f, np.zeros(p + 1), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, gtol=gtol, ftol=ftol, maxcor=20))
        self.w, self.b = res.x[:p].copy(), float(res.x[p])
        self.fit_info = {"success": bool(res.success), "message": str(res.message), "nit": int(res.nit),
                         "nfev": ncall[0], "objective": float(res.fun), "n_train": int(N), "C": self.C,
                         "grad_max": float(np.max(np.abs(res.jac))), "seconds": round(time.time() - t0, 2)}
        return self

    def predict_logit(self, X: np.ndarray, chunk: int = 200_000) -> np.ndarray:
        v, off = self._v_off()
        return np.concatenate([(X[a:a + chunk] @ v).astype(np.float64) + off
                               for a in range(0, X.shape[0], chunk)]) if X.shape[0] else np.zeros(0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _expit(self.predict_logit(X))

    def save(self, d: Path) -> Dict[str, Any]:
        d.mkdir(parents=True, exist_ok=True)
        savez_deterministic(d / "logistic.npz", w=self.w, b=np.array([self.b]), mu=self.mu, sd=self.sd)
        return {"kind": self.kind, "C": self.C, "files": ["logistic.npz"], "fit_info": self.fit_info}

    @classmethod
    def load(cls, d: Path, spec: Mapping[str, Any]) -> "LogisticModel":
        m = cls(spec["C"])
        z = np.load(d / "logistic.npz")
        m.w, m.b, m.mu, m.sd = z["w"], float(z["b"][0]), z["mu"], z["sd"]
        m.fit_info = dict(spec.get("fit_info") or {})
        return m


# ============================================================================ LightGBM
class LGBMModel:
    kind = "lgbm"

    def __init__(self, num_leaves: int, max_rounds: int = LGBM_MAX_ROUNDS, patience: int = LGBM_PATIENCE,
                 threads: int = MAX_THREADS):
        self.num_leaves = int(num_leaves)
        self.max_rounds = int(max_rounds)
        self.patience = int(patience)
        self.threads = int(min(threads, MAX_THREADS))
        self.booster = None
        self.best_iteration: int = 0
        self.fit_info: Dict[str, Any] = {}

    def params(self) -> Dict[str, Any]:
        p = dict(LGBM_FIXED)
        p["num_leaves"] = self.num_leaves
        p["num_threads"] = self.threads
        return p

    @staticmethod
    def dataset(X: np.ndarray, y: np.ndarray):
        import lightgbm as lgb
        ds = lgb.Dataset(X, label=np.asarray(y, dtype=np.float64), free_raw_data=False,
                         params={"max_bin": LGBM_FIXED["max_bin"], "verbosity": -1,
                                 "data_random_seed": SEED, "num_threads": MAX_THREADS})
        ds.construct()
        return ds

    def fit(self, full_ds, train_rows: np.ndarray, stop_rows: np.ndarray) -> "LGBMModel":
        import lightgbm as lgb
        t0 = time.time()
        tr = full_ds.subset(sorted(np.asarray(train_rows).tolist()))
        va = full_ds.subset(sorted(np.asarray(stop_rows).tolist()))
        evals: Dict[str, Any] = {}
        self.booster = lgb.train(self.params(), tr, num_boost_round=self.max_rounds, valid_sets=[va],
                                 valid_names=["inner"],
                                 callbacks=[lgb.early_stopping(self.patience, verbose=False),
                                            lgb.record_evaluation(evals)])
        self.best_iteration = int(self.booster.best_iteration or self.booster.current_iteration())
        curve = evals.get("inner", {}).get("binary_logloss", [])
        self.fit_info = {"best_iteration": self.best_iteration, "rounds_run": len(curve),
                         "hit_max_rounds": len(curve) >= self.max_rounds,
                         "inner_logloss_best": float(min(curve)) if curve else None,
                         "n_train": int(len(train_rows)), "n_stop": int(len(stop_rows)),
                         "seconds": round(time.time() - t0, 2), "num_leaves": self.num_leaves,
                         "max_rounds": self.max_rounds, "patience": self.patience}
        return self

    def predict_proba(self, X: np.ndarray, chunk: int = 200_000) -> np.ndarray:
        if not X.shape[0]:
            return np.zeros(0)
        return np.concatenate([self.booster.predict(X[a:a + chunk], num_iteration=self.best_iteration,
                                                    num_threads=self.threads)
                               for a in range(0, X.shape[0], chunk)]).astype(np.float64)

    def save(self, d: Path) -> Dict[str, Any]:
        d.mkdir(parents=True, exist_ok=True)
        txt = self.booster.model_to_string(num_iteration=self.best_iteration)
        (d / "lgbm_model.txt").write_text(txt, encoding="utf-8")
        return {"kind": self.kind, "num_leaves": self.num_leaves, "best_iteration": self.best_iteration,
                "files": ["lgbm_model.txt"], "fit_info": self.fit_info}

    @classmethod
    def load(cls, d: Path, spec: Mapping[str, Any]) -> "LGBMModel":
        import lightgbm as lgb
        m = cls(spec["num_leaves"])
        m.booster = lgb.Booster(model_str=(d / "lgbm_model.txt").read_text(encoding="utf-8"))
        m.best_iteration = int(spec["best_iteration"])
        m.fit_info = dict(spec.get("fit_info") or {})
        return m


# ============================================================================ MLP
_NET_CLS = None


def _net_class():
    global _NET_CLS
    if _NET_CLS is None:
        import torch
        import torch.nn as nn

        class SharedPlayerMLP(nn.Module):
            def __init__(self, n_fields: int, n_slots: int, n_other: int, enc_dim: int = ENC_DIM,
                         head: Sequence[int] = HEAD):
                super().__init__()
                self.n_fields, self.n_slots = n_fields, n_slots
                self.enc = nn.Sequential(nn.Linear(n_fields, enc_dim), nn.GELU(), nn.Linear(enc_dim, enc_dim), nn.GELU())
                d = n_slots * enc_dim + n_other
                layers: List[nn.Module] = []
                for h in head:
                    layers += [nn.Linear(d, h), nn.GELU()]
                    d = h
                layers += [nn.Linear(d, 1)]
                self.head = nn.Sequential(*layers)

            def forward(self, players, other):
                e = self.enc(players).reshape(players.shape[0], -1)
                return self.head(torch.cat([e, other], dim=1)).squeeze(-1)

        _NET_CLS = SharedPlayerMLP
    return _NET_CLS


class MLPModel:
    kind = "mlp"

    def __init__(self, weight_decay: float, names: Sequence[str], seeds: Sequence[int] = MLP_SEEDS_FINAL,
                 max_epochs: int = MLP_MAX_EPOCHS, patience: int = MLP_PATIENCE, batch: int = MLP_BATCH,
                 lr: float = MLP_LR, threads: int = MAX_THREADS):
        self.weight_decay = float(weight_decay)
        self.names = list(names)
        self.seeds = [int(s) for s in seeds]
        self.max_epochs, self.patience, self.batch, self.lr = int(max_epochs), int(patience), int(batch), float(lr)
        self.threads = int(min(threads, MAX_THREADS))
        self.slot_idx, self.other_idx, _ = player_layout(self.names)
        self.perm = swap_permutation(self.names)
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.states: List[Dict[str, np.ndarray]] = []
        self.fit_info: Dict[str, Any] = {}

    def _new_net(self):
        cls = _net_class()
        return cls(self.slot_idx.shape[1], self.slot_idx.shape[0], len(self.other_idx))

    def _forward(self, net, xb_raw: np.ndarray):
        import torch
        xt = (torch.from_numpy(np.ascontiguousarray(xb_raw, dtype=np.float32)) - self._mu_t) / self._sd_t
        players = xt[:, self._slot_t].reshape(xt.shape[0], self.slot_idx.shape[0], self.slot_idx.shape[1])
        return net(players, xt[:, self._other_t])

    def _prep_tensors(self):
        import torch
        torch.set_num_threads(self.threads)
        self._mu_t = torch.from_numpy(self.mu.astype(np.float32))
        self._sd_t = torch.from_numpy(self.sd.astype(np.float32))
        self._slot_t = torch.from_numpy(self.slot_idx.ravel())
        self._other_t = torch.from_numpy(self.other_idx)

    def _logits_net(self, net, X: np.ndarray, rows: Optional[np.ndarray] = None, chunk: int = 32768) -> np.ndarray:
        import torch
        net.eval()
        n = X.shape[0] if rows is None else len(rows)
        out = np.empty(n, dtype=np.float64)
        with torch.no_grad():
            for a in range(0, n, chunk):
                xb = X[a:a + chunk] if rows is None else X[rows[a:a + chunk]]
                out[a:a + chunk] = self._forward(net, xb).numpy()
        return out

    def fit(self, X: np.ndarray, y: np.ndarray, train_rows: np.ndarray, stop_rows: np.ndarray,
            mu: np.ndarray, sd: np.ndarray, log: Optional[Callable[[str], None]] = None) -> "MLPModel":
        """X / y in the natural (unswapped) orientation; a fresh random 50 % of every minibatch is team-swapped."""
        import torch
        import torch.nn.functional as F
        self.mu, self.sd = np.asarray(mu, np.float64), np.asarray(sd, np.float64)
        self._prep_tensors()
        train_rows = np.asarray(train_rows)
        stop_rows = np.asarray(stop_rows)
        y_stop = np.asarray(y, np.float64)[stop_rows]
        self.states, runs = [], []
        t_all = time.time()
        for seed in self.seeds:
            t0 = time.time()
            torch.manual_seed(SEED + 1000 * seed)
            rng = np.random.default_rng([SEED, seed])
            net = self._new_net()
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            best, best_state, bad, hist = float("inf"), None, 0, []
            for ep in range(self.max_epochs):
                net.train()
                order = rng.permutation(train_rows)
                tl, nb = 0.0, 0
                for a in range(0, len(order), self.batch):
                    r = np.sort(order[a:a + self.batch])
                    xb = X[r].astype(np.float32, copy=True)
                    yb = np.asarray(y[r], dtype=np.float32).copy()
                    sw = rng.random(len(r)) < 0.5
                    if sw.any():
                        xb[sw] = xb[sw][:, self.perm]
                        yb[sw] = 1.0 - yb[sw]
                    logit = self._forward(net, xb)
                    loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(yb))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    tl += float(loss.detach()) * len(r)
                    nb += len(r)
                val = log_loss(y_stop, _expit(self._logits_net(net, X, stop_rows)))
                hist.append({"epoch": ep + 1, "train_loss": tl / max(nb, 1), "inner_logloss": val})
                if log:
                    log(f"      mlp wd={self.weight_decay:g} seed={seed} epoch {ep + 1}: train {tl / max(nb, 1):.5f} "
                        f"inner {val:.5f} ({time.time() - t0:.0f} s)")
                if val < best - 1e-7:
                    best, bad = val, 0
                    best_state = {k: v.detach().clone().numpy() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            self.states.append(best_state)
            runs.append({"seed": seed, "best_inner_logloss": best, "epochs_run": len(hist),
                         "best_epoch": int(np.argmin([h["inner_logloss"] for h in hist]) + 1),
                         "hit_max_epochs": len(hist) >= self.max_epochs and bad < self.patience,
                         "history": hist, "seconds": round(time.time() - t0, 2)})
        self.fit_info = {"weight_decay": self.weight_decay, "seeds": self.seeds, "runs": runs,
                         "n_train": int(len(train_rows)), "n_stop": int(len(stop_rows)),
                         "max_epochs": self.max_epochs, "patience": self.patience, "batch": self.batch, "lr": self.lr,
                         "seconds": round(time.time() - t_all, 2)}
        return self

    def _nets(self):
        import torch
        if getattr(self, "_net_cache", None) is None:
            nets = []
            for st in self.states:
                net = self._new_net()
                net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in st.items()})
                net.eval()
                nets.append(net)
            self._net_cache = nets
        return self._net_cache

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        self._prep_tensors()
        if not X.shape[0]:
            return np.zeros(0)
        return np.mean([self._logits_net(net, X) for net in self._nets()], axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _expit(self.predict_logit(X))

    def save(self, d: Path) -> Dict[str, Any]:
        d.mkdir(parents=True, exist_ok=True)
        files = []
        for s, st in zip(self.seeds, self.states):
            fn = f"mlp_seed{s}.npz"
            savez_deterministic(d / fn, **{k: np.asarray(v) for k, v in st.items()})
            files.append(fn)
        savez_deterministic(d / "mlp_standardizer.npz", mu=self.mu, sd=self.sd)
        files.append("mlp_standardizer.npz")
        return {"kind": self.kind, "weight_decay": self.weight_decay, "seeds": self.seeds, "files": files,
                "architecture": {"encoder": [int(self.slot_idx.shape[1]), ENC_DIM, ENC_DIM], "head": list(HEAD) + [1],
                                 "n_slots": int(self.slot_idx.shape[0]), "n_other": int(len(self.other_idx)),
                                 "activation": "GELU"},
                "fit_info": self.fit_info}

    @classmethod
    def load(cls, d: Path, spec: Mapping[str, Any], names: Sequence[str]) -> "MLPModel":
        m = cls(spec["weight_decay"], names, seeds=spec["seeds"])
        m.states = []
        for s in m.seeds:
            z = np.load(d / f"mlp_seed{s}.npz")
            m.states.append({k: z[k] for k in z.files})
        z = np.load(d / "mlp_standardizer.npz")
        m.mu, m.sd = z["mu"], z["sd"]
        m.fit_info = dict(spec.get("fit_info") or {})
        return m


class GoldDiffLogistic:
    """Baseline: logistic regression of the target on gold difference and gold difference x time.

    gold_diff = sum of blue totalGold_norm - sum of red totalGold_norm (StateV3 player blocks); features
    [gold_diff, gold_diff * time_minutes], standardised, unpenalised (sklearn penalty=None), intercept."""
    kind = "gold_diff_logistic"

    def __init__(self, names: Sequence[str]):
        idx = {n: i for i, n in enumerate(names)}
        from gameplay.state_value_v3 import SLOT_PREFIXES
        self.blue = [idx[p + "totalGold_norm"] for p in SLOT_PREFIXES if p.startswith("blue_")]
        self.red = [idx[p + "totalGold_norm"] for p in SLOT_PREFIXES if p.startswith("red_")]
        self.time = idx["time_minutes"]
        self.model = None

    def features(self, X: np.ndarray) -> np.ndarray:
        gd = X[:, self.blue].astype(np.float64).sum(1) - X[:, self.red].astype(np.float64).sum(1)
        return np.column_stack([gd, gd * X[:, self.time].astype(np.float64)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GoldDiffLogistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.model = make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=1000))
        self.model.fit(self.features(X), np.asarray(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.features(X))[:, 1]

    def describe(self) -> Dict[str, Any]:
        sc, lr = self.model[0], self.model[1]
        return {"kind": self.kind, "features": ["gold_diff", "gold_diff_x_time_minutes"],
                "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
                "coef": lr.coef_.ravel().tolist(), "intercept": float(lr.intercept_[0])}


# ============================================================================ frozen bundle + predict API
class FrozenV:
    """V(state matrix) -> P(blue wins) = calibrator(model(X)), with the NaN fill of the training data."""

    def __init__(self, model, calibrator: PositiveSlopeSigmoid, names: Sequence[str], fill: np.ndarray,
                 spec: Mapping[str, Any], bundle_sha256: Optional[str] = None):
        self.model, self.calibrator = model, calibrator
        self.names = list(names)
        self.fill = np.asarray(fill, dtype=np.float64)
        self.spec = dict(spec)
        self.bundle_sha256 = bundle_sha256
        self.kind = model.kind

    def _prepare(self, X: np.ndarray, columns: Optional[Sequence[str]] = None) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if columns is not None and list(columns) != self.names:
            idx = {n: i for i, n in enumerate(columns)}
            missing = [n for n in self.names if n not in idx]
            if missing:
                raise ValueError(f"X lacks V columns {missing[:5]}")
            X = X[:, [idx[n] for n in self.names]]
        if X.shape[1] != len(self.names):
            raise ValueError(f"X has {X.shape[1]} columns, V needs {len(self.names)} (StateV3 order)")
        X = np.array(X, dtype=np.float32, copy=True)
        fill_nan_inplace(X, self.fill)
        return X

    def predict_raw(self, X: np.ndarray, columns: Optional[Sequence[str]] = None) -> np.ndarray:
        return self.model.predict_proba(self._prepare(X, columns))

    def predict(self, X: np.ndarray, columns: Optional[Sequence[str]] = None) -> np.ndarray:
        """Calibrated P(blue wins) for each row of a StateV3 matrix."""
        return self.calibrator.predict(self.predict_raw(X, columns))

    __call__ = predict

    def predict_states(self, states) -> np.ndarray:
        from gameplay.state_value_v3 import state_matrix_v3
        return self.predict(state_matrix_v3(states, self.names))


def _name_hash(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _strip_timing(o: Any) -> Any:
    """Drop wall-clock fields ('seconds') so that a refit with equal results writes an identical bundle.json."""
    if isinstance(o, dict):
        return {k: _strip_timing(v) for k, v in o.items() if k != "seconds"}
    if isinstance(o, list):
        return [_strip_timing(v) for v in o]
    return o


def save_bundle(d: Path, model, calibrator: PositiveSlopeSigmoid, names: Sequence[str], fill: np.ndarray,
                extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Write model files + prep.npz + bundle.json (lists every file's sha256).  Returns {'bundle_sha256', ...}:
    the bundle identity is the sha256 of bundle.json."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    spec = _strip_timing(model.save(d))
    savez_deterministic(d / "prep.npz", fill=np.asarray(fill, dtype=np.float64))
    files = list(spec["files"]) + ["prep.npz"]
    bundle = {"format": BUNDLE_FORMAT, "kind": model.kind, "model": spec, "calibrator": calibrator.to_dict(),
              "columns": list(names), "columns_name_hash": _name_hash(names), "n_columns": len(names),
              "file_sha256": {f: sha256_file(d / f) for f in files}, "output": "P(blue team wins)",
              "predict": "ev4_v_models.load_v(dir).predict(X_stateV3)", "extra": dict(extra or {})}
    blob = json.dumps(bundle, indent=2, sort_keys=True, default=_json_default).encode("utf-8")
    (d / "bundle.json").write_bytes(blob)
    return {"dir": str(d), "bundle_sha256": sha256_bytes(blob), "file_sha256": bundle["file_sha256"]}


def load_v(d: Path, expected_sha256: Optional[str] = None,
           expected_columns_hash: Optional[str] = None) -> FrozenV:
    """Load a saved V bundle; verifies every file's sha256, the bundle sha256 (if given) and the column hash."""
    d = Path(d)
    blob = (d / "bundle.json").read_bytes()
    digest = sha256_bytes(blob)
    if expected_sha256 is not None and digest != str(expected_sha256).strip().lower():
        raise RuntimeError(f"V bundle sha256 {digest} != expected {expected_sha256}")
    b = json.loads(blob.decode("utf-8"))
    if b.get("format") != BUNDLE_FORMAT:
        raise RuntimeError(f"unknown V bundle format {b.get('format')}")
    for f, h in b["file_sha256"].items():
        if sha256_file(d / f) != h:
            raise RuntimeError(f"V bundle file {f} does not match its sha256")
    names = b["columns"]
    if _name_hash(names) != b["columns_name_hash"]:
        raise RuntimeError("V bundle column list does not match its hash")
    if expected_columns_hash is not None and b["columns_name_hash"] != expected_columns_hash:
        raise RuntimeError("V bundle columns differ from the expected StateV3 columns")
    kind = b["kind"]
    if kind == "logistic":
        model = LogisticModel.load(d, b["model"])
    elif kind == "lgbm":
        model = LGBMModel.load(d, b["model"])
    elif kind == "mlp":
        model = MLPModel.load(d, b["model"], names)
    else:
        raise RuntimeError(f"unknown V kind {kind}")
    fill = np.load(d / "prep.npz")["fill"]
    return FrozenV(model, PositiveSlopeSigmoid.from_dict(b["calibrator"]), names, fill, b, bundle_sha256=digest)


class VNotUsable(RuntimeError):
    """The ev4_03 run must not be used downstream (stop rule, pilot, smoke, or no frozen V)."""


def assert_v_usable(frozen_manifest: Any, allow_smoke: bool = False) -> Dict[str, Any]:
    """Gate for every later stage (ev4_04 labels, ev4_05 ...): call it BEFORE load_v.

    frozen_manifest: the ev4_03 frozen_manifest.json as a dict, its path, or the ev4_03 output directory.
    Raises VNotUsable when
      * the manifest has no stop_rule, or stop_rule.stop_before_record1 is not False (F12: the chosen V did not beat
        the gold-difference baseline at full size; record 1 must not be written, V must not be used);
      * the run is a pilot (--pilot-matches: diagnostic only, never frozen);
      * the run is a smoke run and allow_smoke is False;
      * the manifest carries no frozen V (frozen false / V_frozen missing or without bundle_sha256).
    Returns the manifest's V_frozen block ({'dir', 'bundle_sha256', 'kind', ...}); load it with
    load_v(dir, expected_sha256=bundle_sha256)."""
    m = frozen_manifest
    if isinstance(m, (str, Path)):
        p = Path(m)
        if p.is_dir():
            p = p / "frozen_manifest.json"
        if not p.is_file():
            raise VNotUsable(f"frozen manifest missing: {p}")
        m = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(m, Mapping):
        raise VNotUsable("frozen manifest is not a mapping")
    stop = m.get("stop_rule")
    if not isinstance(stop, Mapping) or "stop_before_record1" not in stop:
        raise VNotUsable("frozen manifest carries no stop_rule; refusing to use its V")
    if stop["stop_before_record1"] is not False:
        raise VNotUsable(f"stop rule: the chosen V ({stop.get('chosen')}) does not beat the gold-difference baseline "
                         f"on raw V_SELECT log loss ({stop.get('chosen_raw_select_logloss')} vs "
                         f"{stop.get('gold_raw_select_logloss')}); do not write record 1 and do not use this V")
    if m.get("pilot"):
        raise VNotUsable("pilot run (--pilot-matches): diagnostic only, its V is never frozen")
    if m.get("smoke") and not allow_smoke:
        raise VNotUsable("smoke run: its V may be used only by smoke stages (allow_smoke=True)")
    vf = m.get("V_frozen")
    if m.get("frozen") is False or not isinstance(vf, Mapping) or not vf.get("bundle_sha256"):
        raise VNotUsable(f"no frozen V in the manifest (reason: {m.get('not_frozen_reason')})")
    return dict(vf)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ============================================================================ martingale checks
BONF_LEVEL = 0.90


def bonferroni_z(k: int, level: float = BONF_LEVEL) -> float:
    from scipy.stats import norm
    return float(norm.ppf(1 - (1 - level) / (2 * k)))


def martingale_means(dv: np.ndarray, clusters: Sequence[str], strata: Mapping[str, np.ndarray],
                     bands: Mapping[str, float], level: float = BONF_LEVEL) -> Dict[str, Any]:
    """Mean dV per stratum with Bonferroni simultaneous `level` intervals (match-cluster SE); pass when the whole
    interval lies inside [-band, band]."""
    k = len(strata)
    z = bonferroni_z(k, level)
    out: Dict[str, Any] = {"level": level, "k": k, "z": z, "strata": {}}
    cl = np.asarray(clusters)
    for name, m in strata.items():
        m = np.asarray(m, dtype=bool)
        band = float(bands[name])
        if m.sum() < 2:
            out["strata"][name] = {"n": int(m.sum()), "band": band, "pass": None}
            continue
        mean, se, G = cluster_mean_se(dv[m], cl[m])
        lo, hi = mean - z * se, mean + z * se
        out["strata"][name] = {"n": int(m.sum()), "n_clusters": G, "mean": mean, "se_cluster": se,
                               "sd": float(np.std(dv[m], ddof=1)), "ci": [lo, hi], "band": band,
                               "mean_in_band": bool(abs(mean) <= band),
                               "pass": bool(lo >= -band and hi <= band)}
    out["all_pass"] = all(s.get("pass") is True for s in out["strata"].values())
    return out


def _whiten_basis(X: np.ndarray, rel_tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(mu, sd, U, lam): standardise columns (constant -> dropped via sd=inf) and eigen-decompose the correlation."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    sd = X.std(0)
    keep = sd > 1e-10
    Xs = (X[:, keep] - mu[keep]) / sd[keep]
    C = Xs.T @ Xs / len(Xs)
    lam, U = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order]
    ok = lam > rel_tol * max(lam[0], 1e-300)
    return mu, np.where(keep, sd, np.inf), U[:, ok], lam[ok]


def _components(X: np.ndarray, mu, sd, U, lam, k: int) -> np.ndarray:
    keep = np.isfinite(sd)
    Xs = (np.asarray(X, dtype=np.float64)[:, keep] - mu[keep]) / sd[keep]
    return Xs @ U[:, :k] / np.sqrt(lam[:k])


def _cluster_matrix(inv: np.ndarray, G: int):
    from scipy.sparse import csr_matrix
    n = len(inv)
    return csr_matrix((np.ones(n), (inv, np.arange(n))), shape=(G, n))


def _wald_cr1(A: np.ndarray, y: np.ndarray, bread: np.ndarray, P, G: int) -> Tuple[float, np.ndarray]:
    """(Wald statistic of all slopes, beta) with the CR1 covariance; P = cluster x row indicator (sparse)."""
    n, K = A.shape
    beta = bread @ (A.T @ y)
    resid = y - A @ beta
    S = P @ (A * resid[:, None])
    corr = G / max(G - 1, 1) * (n - 1) / max(n - K, 1)
    V = corr * bread @ (S.T @ S) @ bread
    b = beta[1:]
    Vs = V[1:, 1:]
    try:
        W = float(b @ np.linalg.solve(Vs, b))
    except np.linalg.LinAlgError:
        W = float(b @ np.linalg.pinv(Vs) @ b)
    return W, beta


def efficiency_regression(dv: np.ndarray, X: np.ndarray, clusters: Sequence[str], oos_split: np.ndarray,
                          min_clusters_per_param: int = 10, n_boot: int = 0, seed: int = SEED) -> Dict[str, Any]:
    """OLS of dV on X_t (in-sample, all rows) with a match-cluster-robust (CR1) joint Wald F test of all slopes.

    X is reduced to its identifiable principal directions (eigenvalues > 1e-8 x the largest, after dropping
    constant columns); the joint test of all identifiable slopes equals the test on those whitened components.
    If there are fewer than 10 clusters per component, only the leading floor(G / 10) components are used and
    the result is flagged 'reduced' (small samples only).  Out-of-sample R^2: fit on one half of the matches
    (basis and OLS from that half), evaluate on the other half, both directions; R^2 relative to the fit-half mean
    and relative to 0 (the martingale prediction).

    PRIMARY p-value (author pre-decision 2026-09-25, records/stage2_predecisions_20260925T151437Z.json key
    'martingale_b_primary_p'): the wild cluster bootstrap, run when n_boot > 0 (null imposed: dV = constant +
    residual; Rademacher weight per match; the studentised CR1 Wald statistic is recomputed in every replicate):
    p = (1 + #{W* >= W}) / (n_boot + 1) -> 'p_value_primary'.  The asymptotic CR1 F test is SECONDARY
    ('p_value_F'): it over-rejects with heavy-tailed dV (smoke: F p = 7e-17 vs bootstrap p = 0.715; simulation:
    about 10 % at nominal 5 % with t(4) noise, n / k = 80).  With n_boot = 0 'p_value_primary' is None."""
    from scipy.stats import f as fdist
    dv = np.asarray(dv, dtype=np.float64)
    cl = np.asarray(clusters)
    n = len(dv)
    codes, inv = np.unique(cl, return_inverse=True)
    G = len(codes)
    mu, sd, U, lam = _whiten_basis(X)
    rank = U.shape[1]
    kmax = max(1, G // min_clusters_per_param)
    k = min(rank, kmax)
    Z = _components(X, mu, sd, U, lam, k)
    A = np.column_stack([np.ones(n), Z])
    K = A.shape[1]
    bread = np.linalg.pinv(A.T @ A)
    P = _cluster_matrix(inv, G)
    W, beta = _wald_cr1(A, dv, bread, P, G)
    resid = dv - A @ beta
    F = W / k
    p = float(fdist.sf(F, k, max(G - 1, 1)))
    sst = float(np.sum((dv - dv.mean()) ** 2))
    r2 = 1 - float(np.sum(resid ** 2)) / sst if sst > 0 else float("nan")
    out: Dict[str, Any] = {"n": n, "n_clusters": G, "n_columns": int(X.shape[1]),
                           "n_constant_dropped": int(np.sum(~np.isfinite(sd))), "rank": rank, "k_used": k,
                           "reduced": bool(k < rank), "F": F, "df": [k, max(G - 1, 1)], "p_value_F": p,
                           "primary_test": "wild cluster (match) bootstrap, Rademacher, null imposed",
                           "secondary_test": "CR1 Wald F (asymptotic, oversized in small samples)",
                           "p_value_primary": None,
                           "wald": W, "r2_in_sample": r2,
                           "r2_adj_in_sample": 1 - (1 - r2) * (n - 1) / max(n - K, 1) if math.isfinite(r2) else None}
    if n_boot > 0:
        t0 = time.time()
        rng = np.random.default_rng(seed)
        u0 = dv - dv.mean()
        ws = []
        for _ in range(int(n_boot)):
            v = rng.choice(np.array([-1.0, 1.0]), size=G)
            ws.append(_wald_cr1(A, dv.mean() + u0 * v[inv], bread, P, G)[0])
        ws = np.asarray(ws)
        out["wild_bootstrap"] = {"n_boot": int(n_boot), "p_value": float((1 + np.sum(ws >= W)) / (n_boot + 1)),
                                 "weights": "Rademacher per match, null imposed", "seed": seed,
                                 "W_star_quantiles": {q: float(np.quantile(ws, q)) for q in (0.5, 0.9, 0.95, 0.99)},
                                 "seconds": round(time.time() - t0, 2)}
        out["p_value_primary"] = out["wild_bootstrap"]["p_value"]
    oos = []
    for fit_half in (0, 1):
        fm = np.asarray(oos_split) == fit_half
        em = ~fm
        if fm.sum() < 3 or em.sum() < 1:
            continue
        Gf = len(np.unique(cl[fm]))
        mu2, sd2, U2, lam2 = _whiten_basis(X[fm])
        k2 = min(U2.shape[1], max(1, Gf // min_clusters_per_param))
        Af = np.column_stack([np.ones(int(fm.sum())), _components(X[fm], mu2, sd2, U2, lam2, k2)])
        bf, *_ = np.linalg.lstsq(Af, dv[fm], rcond=None)
        Ae = np.column_stack([np.ones(int(em.sum())), _components(X[em], mu2, sd2, U2, lam2, k2)])
        pred = Ae @ bf
        sse = float(np.sum((dv[em] - pred) ** 2))
        ybar = float(dv[fm].mean())
        oos.append({"fit_half": fit_half, "n_fit": int(fm.sum()), "n_eval": int(em.sum()), "k_used": k2,
                    "r2_oos_vs_fit_mean": 1 - sse / float(np.sum((dv[em] - ybar) ** 2)),
                    "r2_oos_vs_zero": 1 - sse / float(np.sum(dv[em] ** 2)),
                    "null_expectation_approx": -k2 / max(int(fm.sum()) - k2 - 1, 1)})
    out["oos"] = oos
    if oos:
        out["r2_oos_mean"] = float(np.mean([o["r2_oos_vs_fit_mean"] for o in oos]))
        out["r2_oos_vs_zero_mean"] = float(np.mean([o["r2_oos_vs_zero"] for o in oos]))
    return out
