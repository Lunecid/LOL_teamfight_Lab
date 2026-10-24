"""T3.2 E3 variance functions: match-level, union-indicator and shared-count two-way variances for matches whose
players repeat.

Binding specification: reports/tog_experiments_20261012/PRESPEC_E3_KO.md section 3 (variance definitions; SDD Ruling
18).  PLAN_E1_E5_KO.md Task 8 gives match_var, cgm_twoway_var and twoway_interval and their four tests; union_var,
residual_variances, residual_interval and stratum_difference are added for the PRESPEC.  Pure numerical code: no file
is read or written.  Task 9 (the E3 run) calls these functions on the per-match losses D_m of the external KR
15.18-15.22 matches, in ascending match_index, with the player matrix of e3_players.py (int64 (M, 10), -1 =
unidentified).

Notation: M matches with residuals e_m.
  e_m     D_m - theta_hat for a mean (theta_hat = (1/M) sum_m D_m); influence values for a stratum difference (below).
  k_m     identified players of match m: the entries >= 0 of its player_idx row.  -1 = unidentified and links nothing.
          A player may appear at most once per match (a repeat is refused).
  c_mm'   identified players that matches m and m' share: entry (m, m') of A A^T, where A is the sparse M x players
          0/1 incidence matrix (c_mm = k_m).  A linked pair is a distinct pair of matches m < m' with c_mm' >= 1.

Variances (PRESPEC_E3 section 3)
  V1 = (1/M^2) sum_m e_m^2                                            match level (one way)
  VU = (1/M^2) sum_{m,m'} 1[m = m' or c_mm' > 0] e_m e_m'
     = V1 + cross_U,   cross_U = (2/M^2) sum_{linked m < m'} e_m e_m'  union indicator (multiple membership), the
                                                                       primary two-way variance; can be negative
  V2 = (1/M^2) [sum_m e_m^2 + sum_p (sum_{m contains p} e_m)^2 - sum_m k_m e_m^2]
     = V1 + (2/M^2) sum_{m < m'} c_mm' e_m e_m'                         the plan's shared-count form, reported as a
                                                                       conservative variant (it counts a linked pair
                                                                       c_mm' times); can be negative
  V = max(V1, VU), fallback = VU < V1, interval theta_hat -/+ z sqrt(V), design_effect = sqrt(V / V1) (NaN when
  V1 = 0, i.e. all residuals are 0).  z: at level 0.95 the literal Z_PRESPEC_95 = 1.96 of PRESPEC_E3 section 3
  ("theta_hat -/+ 1.96 sqrt(V)"; not Phi^-1(0.975) = 1.959964, Ruling 38); at any other level Phi^-1((1 + level) / 2).
  Also reported: cross_U and its sign, the number of linked pairs (c >= 1), the number with c >= 2 and their share among
  the linked pairs (NaN when no pair is linked).
  VU and V2 are both summed over the linked pairs (the strict upper triangle of A A^T), so with no linked pair both
  equal V1 exactly.  The player-sum form of V2 is an identity the tests check.

Stratum difference (PRESPEC_E3 section 3)
  Matches m in S_s (M_s of them) carry D_m^s.  theta_s = (1/M_s) sum_{m in S_s} D_m^s, Delta = theta_hi - theta_lo,
  and over all M matches the influence residuals are
    e_m = (M/M_hi) 1[m in S_hi] (D_m^hi - theta_hi) - (M/M_lo) 1[m in S_lo] (D_m^lo - theta_lo),
  so that V1 = Var(theta_hi) + Var(theta_lo) - 2 Cov(theta_hi, theta_lo) for independent matches.  The same variances
  apply: residual_variances(e, player_idx), residual_interval(Delta, e, player_idx).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import sparse
from scipy.special import ndtri

Z_PRESPEC_95 = 1.96          # PRESPEC_E3 section 3: interval theta_hat -/+ 1.96 sqrt(V) at level 0.95 (literal; Ruling 38)


def match_var(values: Any) -> float:
    """V1 of the mean of ``values`` (one per match): (1/M^2) sum_m (D_m - theta_hat)^2."""

    _, e = _centered(values)
    return _v1(e)


def cgm_twoway_var(values: Any, player_idx: Any) -> float:
    """V2, the plan's shared-count two-way variance of the mean of ``values``; can be negative."""

    _, e = _centered(values)
    return residual_variances(e, player_idx)["V2"]


def union_var(values: Any, player_idx: Any) -> float:
    """VU, the union-indicator (multiple-membership) two-way variance of the mean of ``values``; can be negative."""

    _, e = _centered(values)
    return residual_variances(e, player_idx)["VU"]


def residual_variances(e: Any, player_idx: Any) -> dict[str, Any]:
    """V1, V2, VU, cross_U and the linked-pair counts of the residuals ``e`` (one per match, used as given).

    Keys: V1, V2, VU, cross_U (= VU - V1), share_c_ge_2, cross_U_sign (-1, 0 or 1), pairs_c_ge_1, pairs_c_ge_2 and
    matches (M).
    """

    e = _finite_vector(e, "e")
    idx = _player_matrix(player_idx, e.size)
    m2 = float(e.size) ** 2
    first, second, shared = _linked_pairs(idx)
    products = e[first] * e[second]
    v1 = _v1(e)
    cross_u = 2.0 * float(products.sum()) / m2
    cross_2 = 2.0 * float((shared * products).sum()) / m2
    linked = int(first.size)
    linked_2 = int(np.count_nonzero(shared >= 2))
    return {"V1": v1, "V2": v1 + cross_2, "VU": v1 + cross_u, "cross_U": cross_u,
            "share_c_ge_2": linked_2 / linked if linked else math.nan,
            "cross_U_sign": int(np.sign(cross_u)), "pairs_c_ge_1": linked, "pairs_c_ge_2": linked_2,
            "matches": int(e.size)}


def residual_interval(point: Any, e: Any, player_idx: Any, level: float = 0.95) -> dict[str, Any]:
    """The interval point -/+ z sqrt(V), V = max(V1, VU), of an estimate whose residuals are ``e``.

    Keys: point, se_match (sqrt V1), se_twoway (sqrt V), design_effect (sqrt(V / V1)), interval (lower, upper),
    fallback (VU < V1), then the keys of residual_variances, then V, level and z.
    """

    point = _finite_scalar(point, "point")
    z = _z(level)
    variances = residual_variances(e, player_idx)
    v1, vu = variances["V1"], variances["VU"]
    v = max(v1, vu)
    se_twoway = math.sqrt(v)
    half = z * se_twoway
    out: dict[str, Any] = {"point": point, "se_match": math.sqrt(v1), "se_twoway": se_twoway,
                           "design_effect": math.sqrt(v / v1) if v1 > 0.0 else math.nan,
                           "interval": (point - half, point + half), "fallback": bool(vu < v1)}
    out.update(variances)
    out.update({"V": v, "level": float(level), "z": z})
    return out


def twoway_interval(values: Any, player_idx: Any, level: float = 0.95) -> dict[str, Any]:
    """theta_hat = mean of ``values`` (one per match) with its interval theta_hat -/+ z sqrt(max(V1, VU)).

    Keys as residual_interval: point, se_match, se_twoway, design_effect, interval, fallback, V1, V2, VU, cross_U,
    share_c_ge_2, cross_U_sign, pairs_c_ge_1, pairs_c_ge_2, matches, V, level, z.
    """

    point, e = _centered(values)
    return residual_interval(point, e, player_idx, level)


def stratum_difference(D_hi: Any, in_hi: Any, D_lo: Any, in_lo: Any) -> tuple[float, np.ndarray]:
    """(Delta = theta_hi - theta_lo, influence residuals e over all M matches) of a difference of two stratum means.

    ``D_hi`` and ``D_lo`` hold one value per match (M of them) and the boolean masks ``in_hi`` and ``in_lo`` mark the
    matches of each stratum; values outside a stratum are ignored (NaN allowed there).  The strata may overlap.
    """

    hi, in_hi = _stratum(D_hi, in_hi, "hi")
    lo, in_lo = _stratum(D_lo, in_lo, "lo")
    if hi.size != lo.size:
        raise ValueError(f"both strata must cover the same M matches, got {hi.size} and {lo.size}")
    m = hi.size
    theta_hi, theta_lo = float(np.mean(hi[in_hi])), float(np.mean(lo[in_lo]))
    e = np.zeros(m, dtype=np.float64)
    e[in_hi] += (m / int(in_hi.sum())) * (hi[in_hi] - theta_hi)
    e[in_lo] -= (m / int(in_lo.sum())) * (lo[in_lo] - theta_lo)
    return theta_hi - theta_lo, e


def _linked_pairs(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(m, m', c_mm') of every distinct pair of matches m < m' that shares at least one identified player."""

    m = idx.shape[0]
    known = idx >= 0
    match_of_slot = np.nonzero(known)[0]                 # row-major, the order of idx[known]
    if match_of_slot.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    _, column = np.unique(idx[known], return_inverse=True)
    column = column.reshape(-1)
    incidence = sparse.csr_array((np.ones(match_of_slot.size, dtype=np.int32), (match_of_slot, column)),
                                 shape=(m, int(column.max()) + 1))
    shared = (incidence @ incidence.T).tocsr()            # c_mm' (symmetric; diagonal k_m)
    shared.eliminate_zeros()
    first = np.repeat(np.arange(m, dtype=shared.indices.dtype), np.diff(shared.indptr))
    upper = shared.indices > first
    return first[upper], shared.indices[upper], shared.data[upper]


def _player_matrix(player_idx: Any, m: int) -> np.ndarray:
    idx = np.asarray(player_idx)
    if idx.ndim != 2 or idx.shape[0] != m:
        raise ValueError(f"player_idx must have shape (M, k) with M = {m} matches, got {idx.shape}")
    if not np.issubdtype(idx.dtype, np.integer):
        raise TypeError(f"player_idx must hold integers (-1 = unidentified), got dtype {idx.dtype}")
    if idx.size and idx.min() < -1:
        raise ValueError("player_idx entries must be player numbers >= 0 or -1 (unidentified)")
    ordered = np.sort(idx, axis=1)
    if np.any((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)):
        raise ValueError("a player appears twice in one match")
    return idx


def _stratum(values: Any, mask: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask)
    if mask.dtype != np.bool_:
        raise TypeError(f"in_{name} must be a boolean mask over the matches, got dtype {mask.dtype}")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or mask.shape != values.shape:
        raise ValueError(f"D_{name} and in_{name} must be 1-D of the same length, got {values.shape} and {mask.shape}")
    if not mask.any():
        raise ValueError(f"stratum {name} has no match")
    if not np.isfinite(values[mask]).all():
        raise ValueError(f"D_{name} must be finite inside its stratum")
    return values, mask


def _finite_vector(values: Any, name: str) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array (one entry per match), got shape {v.shape}")
    if not np.isfinite(v).all():
        raise ValueError(f"{name} must be finite")
    return v


def _finite_scalar(value: Any, name: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x}")
    return x


def _centered(values: Any) -> tuple[float, np.ndarray]:
    v = _finite_vector(values, "values")
    point = float(np.mean(v))
    return point, v - point


def _v1(e: np.ndarray) -> float:
    return float(np.sum(e * e)) / float(e.size) ** 2


def _z(level: Any) -> float:
    """Z_PRESPEC_95 (1.96) at level 0.95, the PRESPEC's literal value; Phi^-1((1 + level) / 2) at any other level."""

    level = float(level)
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie strictly between 0 and 1, got {level}")
    if level == 0.95:
        return Z_PRESPEC_95
    return float(ndtri(0.5 + level / 2.0))
