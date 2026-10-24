"""T3.3 E3 run: player-match two-way intervals on the external KR 15.18-15.22 matches (lock, then one opening).

Binding specification: reports/tog_experiments_20261012/PRESPEC_E3_KO.md (sections 2-9) and EXPOSURE_KO.md
(PLAN_E1_E5_KO.md Task 9; SDD Rulings 18, 38, 39).  Two stages, each run once; every output is written once under
D:/LOL_Training_20261004/tog_e3_1 and holds aggregates only (no player number and no match key leaves the matrix).

  lock  (Ruling 39 part A) Loads the locked player matrix of Task 7 (PLAYERS_kr_15_18_22.npz; not rebuilt) and five
        columns of the external row table Q_ROWS_h90.npz (engagement_key, match_index, patch, canonical_match_key,
        eval_n_kills; its sealed y column is not read).  Checks that the matrix rows are np.unique(Q_ROWS.match_index)
        with the same canonical keys and one patch per match (the patch of a match comes from the row table), reads the
        contrast weights and the report values from reports/external_validation_20261009/EXTERNAL_REPORT.json, checks
        the six score files the contrasts need (keys a permutation of the row keys, raw p finite in [0, 1]) and
        cross-checks every hash with the external score lock.  The sealed label file is hashed as file bytes only; no
        loss is computed.  Writes E3_LOCK.json: PRESPEC_E3's SHA-256 and the commit holding it, the code (and the
        mirrored e3_evaluate.py), the inputs, the six cells, row, match and stratum counts, the player distribution,
        the rules, the interpreter and numpy/scipy versions and the UTC time.
  run   (part B, only after the review of the lock) Verifies the lock (every hash, the rules, the paths and the
        versions), writes the opening to tog_e3_1/E3_OPENING.json and appends it to
        reports/tog_experiments_20261012/OPENINGS.json (EXPOSURE_KO.md rule 1; the external folder is not touched),
        then reads the sealed labels once: the second opening of that file (the first: EXTERNAL_LABELS_OPENED.json
        openings[0], 2026-10-09).  In PRESPEC_E3 section 5 order: point checks, the meanD bootstrap reproduction, the
        variances, the report E3_REPORT.json.  A failure after the opening writes E3_FAILED_run_<n>.json.

Definitions (PRESPEC_E3 section 3).  D_m = sum_c w_c * (match-mean raw-probability Brier of cell c) with the weights of
EXTERNAL_REPORT.json contrasts.P1 and contrasts.S_effect_LGBM_meanD (primary position share) and the plan's D1 pair
(secondary); theta = (1/M) sum_m D_m; negative means the first cell (LightGBM full input, position on) is better.
Kill strata 1, 2-3, 4+ from eval_n_kills: per stratum the rows of the stratum are averaged within each match that has
rows there (scale_headcount.py:177-205), P1 over those M_s matches; Delta = theta_4+ - theta_1 with the influence
values of boot.stratum_difference.  Variances from boot.py (Task 8): V1, VU, V2, V = max(V1, VU), z = 1.96 at 0.95.

Mirror of the external evaluation (the meanD bootstrap must reproduce the report interval, BLOCKED otherwise):
  scores    ordered_scores        = e3_evaluate.py:175-181 (raw p reordered to the row-table order by engagement key)
  labels    sealed_labels         = e3_evaluate.py:164-169 (h90__engagement_key, h90__valid all true, h90__y as float)
  losses    cell_losses           = e3_evaluate.py:106-107 (qeval.per_match_loss, qeval.align_cells; matches in
                                    ascending match_index, e3_evaluate.py:172 and qeval.py:445)
  bootstrap pooled_bootstrap      = e3_evaluate.py:108 (qeval.paired_match_bootstrap, 2,000 draws, PCG64 seed
                                    20261011, percentile with numpy.quantile linear); cells in sorted key order
                                    (e3_evaluate.py:174) and the meanD coefficients in the report's order.  A cell's
                                    resampled mean does not depend on the other cells passed (test at the real size),
                                    so the six cells here give the draws of the external 23-cell call bit for bit.

Usage: python -B -m studies.tog_experiments_20261012.e3_run lock
       python -B -m studies.tog_experiments_20261012.e3_run run --confirm-second-opening --lock-sha256 <reviewed lock>
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
from dataclasses import dataclass  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Iterable, Mapping  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.stats import chi2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.external_validation_20261009 import e2a_common  # noqa: E402
from studies.q_production_20261005 import qeval  # noqa: E402
from studies.q_production_20261005.qtest import read_json  # noqa: E402
from studies.tog_experiments_20261012.boot import (  # noqa: E402
    Z_PRESPEC_95, residual_interval, stratum_difference, twoway_interval)
from studies.tog_experiments_20261012.common import (  # noqa: E402
    document_record, git_head, peak_memory_bytes, run_record, sha256_file, utc, write_once)
from studies.tog_experiments_20261012.e3_players import DATASET, MAX_FAILED_SHARE, PLAYERS, SOURCES, TOG_E3  # noqa: E402

VERSION = "tog_e3_run_v1_20261012"
PATCHES = e2a_common.EXPECTED_PATCHES[DATASET]                  # 15.18-15.22
PRIMARY = "S_effect_LGBM_meanD"
REPORTED = ("P1", PRIMARY)                                      # the contrasts whose points the external report holds
CONTRASTS = ("P1", PRIMARY, "D1")
PRESPEC_WEIGHTS = {                                             # PRESPEC_E3 section 3, by cell short name
    "P1": {"LGBM_S1_A1_M1_E1_C1_R1_D1": 1.0, "PT_FLEX": -1.0},
    PRIMARY: {"LGBM_S1_A0_M0_E1_C1_R1_D0": 0.5, "LGBM_S0_A0_M0_E1_C1_R1_D0": -0.5,
              "LGBM_S1_A0_M0_E1_C1_R1_D1": 0.5, "LGBM_S0_A0_M0_E1_C1_R1_D1": -0.5},
    "D1": {"LGBM_S1_A0_M0_E1_C1_R1_D1": 1.0, "LGBM_S0_A0_M0_E1_C1_R1_D1": -1.0},
}
SEED, RESAMPLES = qeval.BOOTSTRAP_SEED, qeval.BOOTSTRAP_RESAMPLES  # 20261011, 2000
LEVEL = 0.95                                                    # z = Z_PRESPEC_95 = 1.96 in boot.py (Ruling 38)
POINT_TOL = 1e-9                                                # PRESPEC_E3 section 3 point check
REPRO_DECIMALS = 6                                              # PRESPEC_E3 section 6: "소수 여섯째 자리까지"
KILLS_FIELD = "eval_n_kills"
STRATA = qeval.KILL_COUNT_ORDER                                 # ("1", "2-3", "4+")
DELTA = ("4+", "1")                                             # Delta = theta_4+ - theta_1
PRESPEC_COMMIT = "c8e78b4c563ed27110072f0d11301acd977250b6"     # the author's approval of PRESPEC_E3 (as of d78912e)
ENV_KEYS = ("python", "numpy", "scipy")
LOCK_NAME, OPENING_NAME, REPORT_NAME = "E3_LOCK.json", "E3_OPENING.json", "E3_REPORT.json"
CODE_FILES = ("studies/tog_experiments_20261012/e3_run.py", "studies/tog_experiments_20261012/boot.py",
              "studies/tog_experiments_20261012/common.py", "studies/tog_experiments_20261012/e3_players.py",
              "studies/tog_experiments_20261012/__init__.py", "studies/q_production_20261005/qeval.py",
              "studies/q_production_20261005/qtest.py", "studies/q_production_20261005/__init__.py",
              "studies/external_validation_20261009/e2a_common.py", "studies/external_validation_20261009/__init__.py")
MIRROR = "studies/external_validation_20261009/e3_evaluate.py"   # mirrored, not imported
MIRROR_LINES = {"scores": "175-181", "labels": "164-169", "match_order": "172", "cell_order": "174",
                "losses": "106-107", "bootstrap": "108", "levels": "41", "contrasts": "216, 230"}
DECISIONS = {  # PRESPEC_E3 section 4, on the pooled V interval of the primary position share (meanD)
    "lowered": ("외부 합동 자료에서 위치 몫(meanD)의 V 구간이 0을 포함한다: 주장 2를 '가장 큰 단일 출처'에서 "
                "'외부 패치와 신경망 쌍에서 일관된 개선'으로 낮춘다."),
    "kept": "외부 합동 자료에서 위치 몫(meanD)의 V 구간이 0을 포함하지 않는다: 주장 2를 그대로 쓴다.",
}
SIGN = ("D_m and every contrast are Brier(first cell) - Brier(second cell); negative means the first cell (LightGBM "
        "full input, position on) is better; the opposite sign of E1's G (PRESPEC_E3 line 5)")
PER_MATCH = ("D_m = sum_c w_c * per-match mean raw-probability Brier of cell c (qeval.per_match_loss, align_cells, "
             "contrast_per_match); theta = (1/M) sum_m D_m")
ORDER = ("matches in ascending match_index (np.unique): e3_evaluate.py:172, qeval.per_match_loss (qeval.py:445) and "
         "the player matrix (np.array_equal(PLAYERS.match_index, np.unique(Q_ROWS.match_index)))")
KILL_RULE = ("eval_n_kills of the row table: 1, 2-3, 4+ (qeval.kill_count_strata); per stratum s the rows of s are "
             "averaged within each match with rows there (S_s, M_s matches; scale_headcount.py:177-205) and P1 uses "
             "those M_s matches (e_m = D_m^s - theta_s, denominator M_s^2); Delta = theta_4+ - theta_1 with "
             "e_m = (M/M_4+) 1[m in S_4+] (D_m^4+ - theta_4+) - (M/M_1) 1[m in S_1] (D_m^1 - theta_1) over all M "
             "matches (boot.stratum_difference, 4+ first)")
OPENINGS_RULE = ("EXPOSURE_KO.md rule 1: every ToG experiment that opens sealed labels appends experiment, UTC time and "
                 "lock SHA-256")
OPENING_NOTE = ("second opening of the external KR 15.18-15.22 sealed labels (the first: "
                "reports/external_validation_20261009/EXTERNAL_LABELS_OPENED.json openings[0], 2026-10-09); recorded in "
                "tog_e3_1 and OPENINGS.json only, the external folder is not modified (PRESPEC_E3 section 5)")
DISPERSION_NOTE = ("patches share players, so the five patch estimates are not independent; descriptive only, not used "
                   "for the decision (PRESPEC_E3 section 3)")
BLOCKED_RULES = ("PRESPEC_E3 section 6: a point differs from EXTERNAL_REPORT.json by more than 1e-9; the meanD 95 % "
                 "match bootstrap does not reproduce the report interval to six decimals (matches in ascending "
                 "match_index); more than 1 % of the matches without a key-file mapping; no eval_n_kills column (that "
                 "part only); no PRESPEC hash in the lock")


@dataclass(frozen=True)
class Paths:
    """Where the stages read and write; the defaults are the real run (the tests point them at tmp_path)."""

    out: Path = TOG_E3
    players: Path = TOG_E3 / f"PLAYERS_{DATASET}.npz"
    rows: Path = SOURCES[DATASET]["rows_path"]
    scores: Path = e2a_common.DATA_ROOT / "scores" / DATASET
    sealed: Path = e2a_common.DATA_ROOT / "labels" / DATASET / f"EXTERNAL_LABELS_SEALED_{DATASET}.npz"
    external: Path = ROOT / "reports/external_validation_20261009"
    prespec: Path = ROOT / "reports/tog_experiments_20261012/PRESPEC_E3_KO.md"
    exposure: Path = ROOT / "reports/tog_experiments_20261012/EXPOSURE_KO.md"
    openings: Path = ROOT / "reports/tog_experiments_20261012/OPENINGS.json"

    @property
    def players_json(self) -> Path:
        return self.players.with_suffix(".json")

    @property
    def rows_json(self) -> Path:
        return self.rows.with_suffix(".json")

    @property
    def report(self) -> Path:
        return self.external / "EXTERNAL_REPORT.json"

    @property
    def score_lock(self) -> Path:
        return self.external / "EXTERNAL_SCORE_LOCK.json"

    @property
    def labels_opened(self) -> Path:
        return self.external / "EXTERNAL_LABELS_OPENED.json"

    def score(self, key: str, suffix: str) -> Path:
        return self.scores / f"{key}.{suffix}"


@dataclass(frozen=True, eq=False)
class Inputs:
    """Everything the run needs except the labels, aligned and checked (no player number leaves this object)."""

    match_index: np.ndarray            # (M,) ascending, the player-matrix order
    player_idx: np.ndarray             # (M, 10) int64, -1 = unidentified
    failed_matches: int                # map_status != 0
    patch_of_match: np.ndarray         # (M,) str, from the row table
    keys: np.ndarray                   # row engagement keys, row-table order
    match: np.ndarray                  # row match_index
    row_patch: np.ndarray              # row patch (str)
    n_kills: np.ndarray | None         # row eval_n_kills, None when the column is missing
    preds: dict[str, np.ndarray]       # cell -> raw p in row-table order, cells in sorted key order
    score_order_identical: dict[str, bool]
    alignment: dict[str, bool]


# ---------------------------------------------------------------- contrasts and report values

def short_name(key: str) -> str:
    """The cell name inside an external score key (final__module__<short>__<family>__<selection>)."""

    parts = str(key).split("__")
    return parts[2] if len(parts) >= 3 else str(key)


def contrast_weights(report: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """P1 and S_effect_LGBM_meanD as EXTERNAL_REPORT.json stores them (cells and order kept) and the D1 pair.

    The weights must equal PRESPEC_E3 section 3 cell by cell; otherwise ValueError."""

    contrasts = report.get("contrasts") or {}
    out: dict[str, dict[str, float]] = {}
    for name in REPORTED:
        if name not in contrasts:
            raise ValueError(f"EXTERNAL_REPORT.json has no contrast {name}")
        weights = {str(key): float(w) for key, w in contrasts[name].items()}
        shorts = {short_name(key): w for key, w in weights.items()}
        if len(shorts) != len(weights) or shorts != PRESPEC_WEIGHTS[name]:
            raise ValueError(f"EXTERNAL_REPORT.json contrasts.{name} differs from PRESPEC_E3 section 3")
        out[name] = weights
    by_short = {short_name(key): key for key in out[PRIMARY]}
    out["D1"] = {by_short[short]: w for short, w in PRESPEC_WEIGHTS["D1"].items()}
    return out


def selected_cells(weights: Mapping[str, Mapping[str, float]]) -> dict[str, list[str]]:
    cells = {name: list(weights[name]) for name in CONTRASTS}
    cells["all"] = sorted({key for name in CONTRASTS for key in weights[name]})
    return cells


def report_values(report: Mapping[str, Any], weights: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """The external report's numbers that E3 checks: rows, matches, the P1 and meanD points (full precision), the meanD
    95 % interval, the D1 difference of the two cells' match-weighted raw Brier and the per-patch points."""

    q = report["datasets"][DATASET]["q_h90"]
    by_patch = report["datasets"][DATASET]["q_h90_by_patch"]
    if float(report["levels"][PRIMARY]) != LEVEL or float(q["contrasts"][PRIMARY]["level"]) != LEVEL:
        raise ValueError("the external meanD interval is not at level 0.95")
    s1, s0 = list(weights["D1"])
    return {"rows": int(q["rows"]), "matches": int(q["matches"]),
            "points": {name: float(q["contrasts"][name]["point"]) for name in REPORTED},
            "intervals": {name: [float(x) for x in q["contrasts"][name]["interval"]] for name in REPORTED},
            "levels": {name: float(q["contrasts"][name]["level"]) for name in REPORTED},
            "d1_metrics_difference": float(q["metrics"][s1]["raw"]["brier"]) - float(q["metrics"][s0]["raw"]["brier"]),
            "points_by_patch": {patch: {name: float(by_patch[patch]["contrasts"][name]["point"]) for name in REPORTED}
                                for patch in PATCHES}}


# ---------------------------------------------------------------- the e3_evaluate.py mirror

def ordered_scores(path: Path, keys: np.ndarray) -> tuple[np.ndarray, bool]:
    """(raw p in the order of ``keys``, whether the file stores that order) as e3_evaluate.py:175-181."""

    with np.load(path, allow_pickle=False) as data:
        stored, p = data["engagement_key"], data["p"]
    if stored.shape != keys.shape or p.shape != keys.shape:
        raise RuntimeError(f"{Path(path).name}: {stored.size} scored rows for {keys.size} q rows")
    identical = bool(np.array_equal(stored, keys))
    if identical:
        order = np.arange(len(keys))
    else:
        position = {k: i for i, k in enumerate(stored.tolist())}
        missing = sum(k not in position for k in keys.tolist())
        if missing or len(position) != stored.size:
            raise RuntimeError(f"{Path(path).name}: the scored keys are not the q row keys ({missing} missing)")
        order = np.asarray([position[k] for k in keys.tolist()], dtype=np.int64)
    return np.asarray(p, dtype=np.float64)[order], identical


def sealed_labels(path: Path, keys: np.ndarray) -> np.ndarray:
    """h90 labels of the q rows as e3_evaluate.py:164-169; the only function that reads the sealed file."""

    with np.load(path, allow_pickle=False) as sealed:
        sealed_keys, valid, y_all = sealed["h90__engagement_key"], sealed["h90__valid"], sealed["h90__y"]
    index = {k: i for i, k in enumerate(sealed_keys.tolist())}
    missing = sum(k not in index for k in keys.tolist())
    if missing:
        raise RuntimeError(f"{missing} q rows have no sealed h90 label")
    pick = np.asarray([index[k] for k in keys.tolist()], dtype=np.int64)
    if not np.asarray(valid, dtype=bool)[pick].all():
        raise RuntimeError("a q row has an invalid sealed h90 label")
    y = np.asarray(y_all, dtype=np.float64)[pick]
    if not np.isin(y, (0.0, 1.0)).all():
        raise RuntimeError("the sealed h90 labels are not binary")
    return y


def cell_losses(y: np.ndarray, preds: Mapping[str, np.ndarray], match: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(ascending match ids, per-match Brier of every cell) as e3_evaluate.py:106-107."""

    losses = {key: qeval.per_match_loss(y, p, match) for key, p in preds.items()}
    return qeval.align_cells(losses)


def per_match(aligned: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    """D_m = sum_c w_c * L_c[m] (qeval.contrast_per_match, the weights' order)."""

    return qeval.contrast_per_match(aligned, weights)


def pooled_bootstrap(shared: np.ndarray, aligned: Mapping[str, np.ndarray], weights: Mapping[str, Mapping[str, float]],
                     *, level: float = LEVEL, return_draws: bool = False) -> dict[str, Any]:
    """P1, meanD and D1 on shared match draws: the call of e3_evaluate.py:108 (qeval.paired_match_bootstrap)."""

    contrasts = {name: weights[name] for name in CONTRASTS}
    return qeval.paired_match_bootstrap(shared, aligned, contrasts, levels={name: level for name in contrasts},
                                        resamples=RESAMPLES, seed=SEED, return_draws=return_draws)


# ---------------------------------------------------------------- checks, decision and summaries

def point_check(point: float, expected: float, tol: float = POINT_TOL) -> dict[str, Any]:
    """|theta - report| <= tol (PRESPEC_E3 section 3)."""

    diff = abs(float(point) - float(expected))
    return {"point": float(point), "report": float(expected), "abs_diff": diff, "tolerance": tol, "passed": bool(diff <= tol)}


def reproduction(interval: Iterable[float], expected: Iterable[float], decimals: int = REPRO_DECIMALS) -> dict[str, Any]:
    """Both ends equal the report's after rounding to ``decimals`` places (PRESPEC_E3 section 6)."""

    ours, theirs = [float(x) for x in interval], [float(x) for x in expected]
    rounded, report_rounded = [round(x, decimals) for x in ours], [round(x, decimals) for x in theirs]
    return {"interval": ours, "report_interval": theirs, "decimals": decimals, "rounded": rounded,
            "report_rounded": report_rounded, "abs_diff": [abs(a - b) for a, b in zip(ours, theirs)],
            "bit_identical": ours == theirs, "reproduced": len(ours) == len(theirs) == 2 and rounded == report_rounded}


def decide(interval: Iterable[float]) -> str:
    """PRESPEC_E3 section 4: 'lowered' when the interval contains 0 (an end at 0 counts), else 'kept'."""

    lower, upper = (float(x) for x in interval)
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower > upper:
        raise ValueError(f"not an interval: {lower}, {upper}")
    return "lowered" if lower <= 0.0 <= upper else "kept"


def interval_entry(out: Mapping[str, Any]) -> dict[str, Any]:
    """A boot.py interval with theta / sqrt(V), theta / sqrt(V1) and the conservative V2 variant beside it."""

    entry = dict(out)
    entry["interval"] = list(out["interval"])
    point, se, se1, v1 = out["point"], out["se_twoway"], out["se_match"], out["V1"]
    entry["theta_over_se_twoway"] = point / se if se > 0.0 else math.nan
    entry["theta_over_se_match"] = point / se1 if se1 > 0.0 else math.nan
    v2 = max(v1, out["V2"])
    half = out["z"] * math.sqrt(v2)
    entry["conservative_V2"] = {"V": v2, "se": math.sqrt(v2), "interval": [point - half, point + half],
                                "design_effect": math.sqrt(v2 / v1) if v1 > 0.0 else math.nan,
                                "rule": "max(V1, V2) (the plan's shared-count form; PRESPEC_E3 section 3 conservative "
                                        "variant, not used for the decision)"}
    return entry


def _boot_entry(contrast: Mapping[str, Any], boot: Mapping[str, Any]) -> dict[str, Any]:
    return {"point": contrast["point"], "interval": list(contrast["interval"]), "level": contrast["level"],
            "quantiles": list(contrast["quantiles"]), "resamples": boot["resamples"], "seed": boot["seed"],
            "matches": boot["matches"], "method": "qeval.paired_match_bootstrap (percentile, numpy.quantile linear)"}


def influence_bootstrap(point: float, e: Any, *, resamples: int = RESAMPLES, seed: int = SEED, level: float = LEVEL,
                        return_draws: bool = False) -> dict[str, Any]:
    """Match bootstrap on the influence values: draw b = point + mean(e at the positions of qeval.bootstrap_indices).

    For a mean (e = D - theta) this is the ordinary match bootstrap of D; for Delta it uses the influence values of
    PRESPEC_E3 section 3 ("V1, VU, V2 and the match bootstrap use the same definition")."""

    e = np.asarray(e, dtype=np.float64)
    if e.ndim != 1 or e.size == 0 or not np.isfinite(e).all():
        raise ValueError("influence values must be a finite non-empty 1-D array")
    draws = np.empty(int(resamples), dtype=np.float64)
    for b, positions in enumerate(qeval.bootstrap_indices(e.size, resamples, seed)):
        draws[b] = float(point) + float(np.mean(e[positions]))
    quantiles = qeval.bootstrap_quantiles(level)
    low, high = np.quantile(draws, quantiles)
    out: dict[str, Any] = {"point": float(point), "interval": [float(low), float(high)], "level": float(level),
                           "quantiles": list(quantiles), "resamples": int(resamples), "seed": int(seed),
                           "matches": int(e.size), "method": "point + mean of the influence values at "
                                                             "qeval.bootstrap_indices(M) (percentile, numpy.quantile linear)"}
    if return_draws:
        out["draws"] = draws
    return out


def _positions(match_index: np.ndarray, ids: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(match_index, ids)
    if pos.size and (pos.max() >= match_index.size or not np.array_equal(match_index[pos], ids)):
        raise RuntimeError("a stratum match is not a row of the player matrix")
    return pos


def kill_labels(n_kills: np.ndarray | None) -> tuple[np.ndarray | None, str | None]:
    """(stratum label per row, None) or (None, why the kill-strata part is BLOCKED; PRESPEC_E3 section 6)."""

    if n_kills is None:
        return None, f"the row table has no {KILLS_FIELD} column (PRESPEC_E3 section 6: that part only)"
    try:
        labels = qeval.kill_count_strata(np.asarray(n_kills))
    except ValueError as error:
        return None, f"{KILLS_FIELD} is not a positive integer column ({error}); PRESPEC_E3 section 6: that part only"
    empty = [name for name in STRATA if not (labels == name).any()]
    if empty:
        return None, f"kill strata without rows: {empty} (PRESPEC_E3 section 6: that part only)"
    return labels, None


def kill_strata(y: np.ndarray, preds: Mapping[str, np.ndarray], match: np.ndarray, n_kills: np.ndarray,
                weights: Mapping[str, float], match_index: np.ndarray, player_idx: np.ndarray, *,
                level: float = LEVEL) -> dict[str, Any]:
    """Per-stratum P1 within S_s and Delta = theta_4+ - theta_1 (PRESPEC_E3 section 3), with their intervals."""

    labels, reason = kill_labels(n_kills)
    if reason:
        raise ValueError(reason)
    m = int(match_index.size)
    cells = sorted(weights)
    strata: dict[str, Any] = {}
    values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in STRATA:
        rows = labels == name
        shared, aligned = cell_losses(y[rows], {key: preds[key][rows] for key in cells}, match[rows])
        D = per_match(aligned, weights)
        pos = _positions(match_index, shared)
        boot = qeval.paired_match_bootstrap(shared, aligned, {"P1": dict(weights)}, levels={"P1": level},
                                            resamples=RESAMPLES, seed=SEED)
        strata[name] = {"rows": int(rows.sum()), "matches": int(shared.size),
                        "twoway": interval_entry(twoway_interval(D, player_idx[pos], level)),
                        "bootstrap": _boot_entry(boot["contrasts"]["P1"], boot)}
        full = np.full(m, np.nan)
        full[pos] = D
        mask = np.zeros(m, dtype=bool)
        mask[pos] = True
        values[name] = (full, mask)
    hi, lo = DELTA
    delta, e = stratum_difference(values[hi][0], values[hi][1], values[lo][0], values[lo][1])
    expected = strata[hi]["twoway"]["point"] - strata[lo]["twoway"]["point"]
    difference = {"strata": [hi, lo], "point": delta,
                  "twoway": interval_entry(residual_interval(delta, e, player_idx, level)),
                  "bootstrap": influence_bootstrap(delta, e, level=level),
                  "equals_stratum_points": bool(delta == expected), "influence_sum": float(np.sum(e)),
                  "rule": "boot.stratum_difference(D_4+, in_4+, D_1, in_1) then boot.residual_interval over all M matches"}
    return {"field": KILLS_FIELD, "rule": KILL_RULE, "strata": strata, "difference": difference}


def dispersion(points: Mapping[str, float], variances: Mapping[str, float], pooled: float) -> dict[str, Any]:
    """sum_j (theta_j - theta)^2 / V_j against a chi-square with (patches - 1) degrees of freedom (descriptive)."""

    terms = {str(k): (float(points[k]) - float(pooled)) ** 2 / float(variances[k]) for k in points}
    statistic = float(sum(terms.values()))
    df = len(terms) - 1
    return {"statistic": statistic, "df": df, "p_value": float(chi2.sf(statistic, df)),
            "chi2_upper_95": float(chi2.ppf(0.95, df)), "terms": terms, "pooled": float(pooled), "note": DISPERSION_NOTE}


def player_distribution(player_idx: Any, patch_of_match: Any) -> dict[str, Any]:
    """Players, matches per player (summary and histogram as [k, players in exactly k matches]) and patch overlap."""

    idx = np.asarray(player_idx)
    patch = np.asarray(patch_of_match).astype(str)
    known = idx >= 0
    _, player = np.unique(idx[known], return_inverse=True)
    player = player.reshape(-1)
    counts = np.bincount(player)
    names, code = np.unique(patch, return_inverse=True)
    slot_patch = code.reshape(-1)[np.nonzero(known)[0]]
    pairs = np.unique(player.astype(np.int64) * names.size + slot_patch)
    patches_per_player = np.bincount(pairs // names.size, minlength=counts.size)
    histogram = np.bincount(counts)
    q50, q90, q99 = np.quantile(counts, [0.5, 0.9, 0.99]).tolist()
    return {"players": int(counts.size), "identified_slots": int(known.sum()), "unidentified_slots": int((~known).sum()),
            "matches_per_player": {"min": int(counts.min()), "median": q50, "p90": q90, "p99": q99, "max": int(counts.max()),
                                   "mean": float(counts.mean()), "share_in_one_match": float((counts == 1).mean()),
                                   "histogram": [[k, int(n)] for k, n in enumerate(histogram) if n]},
            "players_in_more_than_one_patch": int((patches_per_player > 1).sum()),
            "share_in_more_than_one_patch": float((patches_per_player > 1).mean()),
            "players_by_patch": {str(n): int(np.unique(player[slot_patch == i]).size) for i, n in enumerate(names)}}


def sentence_15_16(design_effect_meand: float, design_effect_p1: float) -> str:
    """PRESPEC_E3 section 2: one descriptive sentence on 15.16, which has no player identities."""

    return (f"15.16(본 자료 TEST)은 Riot 원본 상세가 없어(경기 209,999개 중 0개) 선수를 식별할 수 없으므로 이중 군집 "
            f"구간을 내지 않고, 외부 KR 15.18–15.22의 설계 효과(위치 몫 meanD DE = {design_effect_meand:.3f}, P1 DE = "
            f"{design_effect_p1:.3f})를 15.16 경기 단위 구간이 선수 의존으로 넓어질 수 있는 정도로만 기술한다.")


# ---------------------------------------------------------------- inputs (no label)

def load_inputs(paths: Paths, cells: Iterable[str]) -> Inputs:
    """The locked player matrix, five row-table columns and the raw p of ``cells`` in row order; alignment checks."""

    with np.load(paths.players, allow_pickle=False) as data:
        match_index = np.asarray(data["match_index"])
        player_keys = np.asarray(data["canonical_match_key"])
        player_idx = np.asarray(data["player_idx"])
        map_status = np.asarray(data["map_status"])
    with np.load(paths.rows, allow_pickle=False) as data:
        available = set(data.files)
        keys = np.asarray(data["engagement_key"])
        match = np.asarray(data["match_index"])
        row_patch = np.asarray(data["patch"])
        row_keys = np.asarray(data["canonical_match_key"])
        n_kills = np.asarray(data[KILLS_FIELD]) if KILLS_FIELD in available else None
    unique, first, inverse = np.unique(match, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    patch_of_match = row_patch[first].astype(str)
    alignment = {
        "players_match_index_is_np_unique_of_q_rows": bool(np.array_equal(match_index, unique)),
        "players_keys_equal_q_row_keys": bool(np.array_equal(player_keys, row_keys[first])),
        "one_key_and_one_patch_per_match": bool(np.array_equal(row_keys, row_keys[first][inverse])
                                                and np.array_equal(row_patch, row_patch[first][inverse])),
        "patches_are_the_external_patches": sorted(set(patch_of_match.tolist())) == sorted(PATCHES),
        "player_matrix_int64_m_by_10": bool(player_idx.dtype == np.int64 and player_idx.shape == (unique.size, PLAYERS)),
        "player_entries_at_least_minus_one": bool(player_idx.size == 0 or int(player_idx.min()) >= -1),
        "engagement_keys_unique": bool(np.unique(keys).size == keys.size),
    }
    preds, identical = {}, {}
    for key in sorted(cells):
        preds[key], identical[key] = ordered_scores(paths.score(key, "npz"), keys)
    alignment["scores_finite_in_unit_interval"] = all(bool(np.isfinite(p).all() and p.min() >= 0.0 and p.max() <= 1.0)
                                                      for p in preds.values())
    return Inputs(match_index=match_index, player_idx=player_idx, failed_matches=int((map_status != 0).sum()),
                  patch_of_match=patch_of_match, keys=keys, match=match, row_patch=row_patch.astype(str), n_kills=n_kills,
                  preds=preds, score_order_identical=identical, alignment=alignment)


def label_free_counts(inputs: Inputs) -> dict[str, Any]:
    labels, reason = kill_labels(inputs.n_kills)
    if reason:
        strata: dict[str, Any] = {"blocked": True, "reason": reason}
    else:
        strata = {name: {"rows": int((labels == name).sum()), "matches": int(np.unique(inputs.match[labels == name]).size)}
                  for name in STRATA}
    m = int(inputs.match_index.size)
    return {"rows": int(inputs.keys.size), "matches": m,
            "rows_by_patch": {p: int((inputs.row_patch == p).sum()) for p in PATCHES},
            "matches_by_patch": {p: int((inputs.patch_of_match == p).sum()) for p in PATCHES},
            "kill_strata": strata,
            "mapping": {"failed": inputs.failed_matches, "matches": m, "failed_share": inputs.failed_matches / m,
                        "max_failed_share": MAX_FAILED_SHARE},
            "players": player_distribution(inputs.player_idx, inputs.patch_of_match)}


def bound_files(paths: Paths, cells: Iterable[str]) -> dict[str, Path]:
    """Every file the lock binds by SHA-256 (EXPOSURE_KO.md and OPENINGS.json are appended to, so not bound)."""

    files = {"prespec": paths.prespec}
    files.update({f"code:{name}": ROOT / name for name in CODE_FILES})
    files[f"mirror:{MIRROR}"] = ROOT / MIRROR
    files.update({"input:players": paths.players, "input:players_json": paths.players_json, "input:q_rows": paths.rows,
                  "input:q_rows_json": paths.rows_json, "input:external_report": paths.report,
                  "input:external_score_lock": paths.score_lock, "input:external_labels_opened": paths.labels_opened})
    for key in cells:
        files[f"score:{key}:npz"] = paths.score(key, "npz")
        files[f"score:{key}:json"] = paths.score(key, "json")
    files["sealed"] = paths.sealed
    return files


def cross_checks(paths: Paths, verify: Mapping[str, Mapping[str, str]], cells: Iterable[str]) -> dict[str, bool]:
    """The bound hashes against the Task 7 record, the row manifest, the external score lock and the first opening."""

    score_lock = (read_json(paths.score_lock) or {}).get("datasets", {}).get(DATASET, {})
    opened = (read_json(paths.labels_opened) or {}).get("openings") or [{}]
    players_record = read_json(paths.players_json) or {}
    rows_sha, sealed_sha = verify["input:q_rows"]["sha256"], verify["sealed"]["sha256"]
    lock_files = score_lock.get("files", {})
    checks = {
        "players_npz_sha256_equals_its_record": verify["input:players"]["sha256"] == players_record.get("npz_sha256"),
        "players_record_row_table_sha256_equals_q_rows":
            (players_record.get("run") or {}).get("input_sha256", {}).get(paths.rows.as_posix()) == rows_sha,
        "q_rows_sha256_equals_its_manifest": (read_json(paths.rows_json) or {}).get("rows_sha256") == rows_sha,
        "q_rows_sha256_equals_external_score_lock": lock_files.get("q_inputs:Q_ROWS_h90.npz", {}).get("sha256") == rows_sha,
        "sealed_sha256_equals_external_score_lock": (score_lock.get("sealed", {}).get("sha256") == sealed_sha
                                                     == lock_files.get("labels:SEALED", {}).get("sha256")),
        "sealed_sha256_equals_first_opening": opened[0].get("sealed_sha256", {}).get(DATASET) == sealed_sha,
    }
    for key in cells:
        meta = read_json(paths.score(key, "json")) or {}
        cell = score_lock.get("cells", {}).get(key, {})
        name = short_name(key)
        checks[f"score:{name}:npz_sha256_equals_its_record_and_the_external_lock"] = (
            verify[f"score:{key}:npz"]["sha256"] == meta.get("pred_sha256") == cell.get("pred_sha256"))
        checks[f"score:{name}:json_sha256_equals_the_external_lock"] = verify[f"score:{key}:json"]["sha256"] == cell.get("meta_sha256")
        checks[f"score:{name}:row_table_sha256_equals_q_rows"] = meta.get("rows_table_sha256") == rows_sha
    return checks


def environment() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "implementation": platform.python_implementation(), "executable": sys.executable,
            "platform": platform.platform()}


def rules(weights: Mapping[str, Mapping[str, float]], values: Mapping[str, Any]) -> dict[str, Any]:
    """The rules the run applies; the lock binds them and the run refuses others."""

    return {
        "dataset": DATASET, "patches": list(PATCHES), "sign": SIGN,
        "contrasts": {name: dict(weights[name]) for name in CONTRASTS},
        "roles": {"P1": "interval only", PRIMARY: "primary position share; the PRESPEC_E3 section 4 decision",
                  "D1": "secondary position share (the plan's D1 pair); interval only"},
        "per_match": PER_MATCH, "match_order": ORDER, "kill_strata": KILL_RULE,
        "variance": {"primary": "V = max(V1, VU)", "V1": "(1/M^2) sum_m e_m^2",
                     "VU": "(1/M^2) sum_{m,m'} 1[m = m' or c_mm' > 0] e_m e_m'",
                     "V2": "(1/M^2) [sum e^2 + sum_{m != m'} c_mm' e_m e_m'] (conservative variant)",
                     "z": Z_PRESPEC_95, "level": LEVEL, "interval": "theta -/+ z sqrt(V)", "design_effect": "sqrt(V / V1)",
                     "implementation": "boot.twoway_interval once per contrast and subset; boot.residual_interval for Delta"},
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "level": LEVEL,
                      "quantiles": list(qeval.bootstrap_quantiles(LEVEL)), "unit": "match",
                      "generator": "qeval.bootstrap_indices: numpy.random.Generator(numpy.random.PCG64(seed))"
                                   ".integers(0, M, size=M) per draw",
                      "interval": "percentile; numpy.quantile method='linear'",
                      "pooled": "qeval.paired_match_bootstrap over the six cells, P1, meanD and D1 on shared draws "
                                "(e3_evaluate.py:106-108)",
                      "strata": "qeval.paired_match_bootstrap within S_s (scale_headcount.py:177-205)",
                      "difference": "point + mean of the influence values over qeval.bootstrap_indices(M)"},
        "point_check": {"tolerance": POINT_TOL, "report_points": {name: values["points"][name] for name in REPORTED}},
        "bootstrap_reproduction": {"contrast": PRIMARY, "report_interval": values["intervals"][PRIMARY],
                                   "report_level": values["levels"][PRIMARY], "decimals": REPRO_DECIMALS,
                                   "rule": "both ends of the pooled meanD 95 % match bootstrap equal the report interval "
                                           "after rounding to six decimals; BLOCKED otherwise"},
        "decision": {"contrast": PRIMARY, "interval": "pooled V interval at level 0.95", "outcomes": DECISIONS,
                     "rule": "lower <= 0 <= upper -> lowered; otherwise kept"},
        "dispersion": {"df": len(PATCHES) - 1, "statistic": "sum_j (theta_j - theta)^2 / V_j, V_j = max(V1_j, VU_j) "
                                                            "within patch j", "note": DISPERSION_NOTE},
        "blocked": BLOCKED_RULES,
    }


def _prespec_problems(prespec: Mapping[str, Any]) -> dict[str, Any]:
    if not (prespec.get("sha256") and prespec.get("commit") and prespec.get("unchanged_since_commit")):
        return {"prespec_hash_or_commit_missing": dict(prespec)}
    if prespec["commit"] != PRESPEC_COMMIT:
        return {"prespec_not_the_approved_commit": {"commit": prespec["commit"], "approved": PRESPEC_COMMIT}}
    return {}


# ---------------------------------------------------------------- lock (part A)

def lock_stage(paths: Paths = Paths(), *, require_committed_code: bool = True) -> int:
    """E3_LOCK.json (PRESPEC_E3 section 5).  The sealed labels are hashed as bytes only; no loss is computed."""

    started, tick = utc(), time.perf_counter()
    target = paths.out / LOCK_NAME
    if target.exists():
        raise FileExistsError(f"{target} exists; it is written once")
    prespec = document_record(paths.prespec)
    problems = _prespec_problems(prespec)
    if problems:
        return _blocked(paths.out, "lock", problems, stage=_stage("lock", started, tick))
    code = {name: document_record(ROOT / name) for name in CODE_FILES}
    if require_committed_code and not all(entry["commit"] and entry["unchanged_since_commit"] for entry in code.values()):
        raise RuntimeError("commit the E3 code before writing the lock")
    report = read_json(paths.report)
    if report is None:
        raise RuntimeError(f"{paths.report} is missing")
    weights = contrast_weights(report)
    values = report_values(report, weights)
    cells = selected_cells(weights)
    tick_load = time.perf_counter()
    inputs = load_inputs(paths, cells["all"])
    bad = sorted(name for name, ok in inputs.alignment.items() if not ok)
    if bad:
        raise RuntimeError(f"alignment checks failed: {bad}")
    if (values["rows"], values["matches"]) != (int(inputs.keys.size), int(inputs.match_index.size)):
        raise RuntimeError("the external report counts other rows or matches than the row table")
    counts = label_free_counts(inputs)
    parts = {"load_and_check": _part(tick_load)}
    if counts["mapping"]["failed_share"] > MAX_FAILED_SHARE:
        return _blocked(paths.out, "lock", {"mapping_failed_share_over_max": counts["mapping"]},
                        stage=_stage("lock", started, tick, parts=parts))
    tick_hash = time.perf_counter()
    verify = {label: {"path": str(path), "sha256": sha256_file(path)} for label, path in bound_files(paths, cells["all"]).items()}
    cross = cross_checks(paths, verify, cells["all"])
    if not all(cross.values()):
        raise RuntimeError(f"hash cross-checks failed: {sorted(k for k, ok in cross.items() if not ok)}")
    parts["hash"] = _part(tick_hash)
    first = ((read_json(paths.labels_opened) or {}).get("openings") or [{}])[0]
    payload = {
        "version": VERSION, "experiment": "E3", "dataset": DATASET, "locked_utc": utc(),
        "prespec": {**prespec, "approved_commit": PRESPEC_COMMIT},
        "exposure_ledger": document_record(paths.exposure),
        "code": code, "mirrored_code": {MIRROR: {**document_record(ROOT / MIRROR), "lines": MIRROR_LINES}},
        "git_head": git_head(),
        "selected_cells": cells,
        "label_policy": ("the sealed label file is hashed as file bytes only and nothing in it is opened before the run "
                         "stage; " + OPENING_NOTE),
        "external_first_opening": {"utc": first.get("utc"), "by": first.get("by"),
                                   "sealed_sha256": (first.get("sealed_sha256") or {}).get(DATASET)},
        "openings_before_lock": _openings(paths.openings),
        "score_order_identical_to_row_table": inputs.score_order_identical,
        "alignment": inputs.alignment, "cross_checks": cross, "counts": counts,
        "rules": rules(weights, values), "environment": environment(), "verify": verify,
        "stage": _stage("lock", started, tick, parts=parts),
        "run": run_record(rows=counts["rows"], matches=counts["matches"],
                          inputs=[paths.players, paths.rows, paths.report, paths.sealed]
                          + [paths.score(key, "npz") for key in cells["all"]],
                          code=[ROOT / name for name in CODE_FILES], seed=SEED, started_utc=started),
    }
    paths.out.mkdir(parents=True, exist_ok=True)
    write_once(target, _clean(payload))
    print(f"lock written: {target} ({len(verify)} hashed files; PRESPEC commit {prespec['commit'][:7]}; rows "
          f"{counts['rows']}, matches {counts['matches']}, players {counts['players']['players']})")
    return 0


def verify_lock(lock: Mapping[str, Any], paths: Paths) -> dict[str, Any]:
    """Why the run may not open the sealed labels (empty when the lock holds)."""

    problems: dict[str, Any] = {}
    prespec, verify = lock.get("prespec") or {}, lock.get("verify") or {}
    if not (prespec.get("sha256") and prespec.get("commit")):
        problems["prespec_hash_missing"] = True                       # PRESPEC_E3 section 6
    else:
        if prespec["commit"] != PRESPEC_COMMIT:
            problems["prespec_not_the_approved_commit"] = prespec["commit"]
        if (verify.get("prespec") or {}).get("sha256") != prespec["sha256"]:
            problems["prespec_entry_differs"] = True
    cells = (lock.get("selected_cells") or {}).get("all") or []
    expected = {label: str(path) for label, path in bound_files(paths, cells).items()}
    missing = sorted(set(expected) - set(verify))
    if missing or not cells:
        problems["entries_missing"] = missing or ["selected_cells"]
    moved = {label: {"locked": verify[label]["path"], "now": path} for label, path in expected.items()
             if label in verify and verify[label]["path"] != path}
    if moved:
        problems["paths_differ"] = moved
    changed = {}
    for label, entry in verify.items():
        path = Path(entry["path"])
        now = sha256_file(path) if path.is_file() else None
        if now != entry["sha256"]:
            changed[label] = {"path": entry["path"], "locked": entry["sha256"], "now": now}
    if changed:
        problems["changed_since_lock"] = changed
    else:
        try:
            report = read_json(paths.report)
            weights = contrast_weights(report)
            current = _clean(rules(weights, report_values(report, weights)))
        except (KeyError, TypeError, ValueError) as error:
            problems["rules_unreadable"] = repr(error)
        else:
            if lock.get("rules") != current:
                problems["rules_differ"] = True
            if sorted(cells) != selected_cells(weights)["all"]:
                problems["cells_differ"] = True
    now_env, locked_env = environment(), lock.get("environment") or {}
    differs = {key: {"locked": locked_env.get(key), "now": now_env[key]} for key in ENV_KEYS if locked_env.get(key) != now_env[key]}
    if differs:
        problems["environment_differs"] = differs
    return problems


# ---------------------------------------------------------------- run (part B)

def run_stage(paths: Paths = Paths(), *, expected_lock_sha256: str | None = None) -> int:
    """Verify the lock, record the opening, read the sealed labels once and write E3_REPORT.json once.

    ``expected_lock_sha256`` (the SHA-256 of the reviewed lock) makes any other lock file BLOCKED before the opening."""

    started, tick = utc(), time.perf_counter()
    for name in (REPORT_NAME, OPENING_NAME):
        if (paths.out / name).exists():
            raise FileExistsError(f"{paths.out / name} exists; the sealed labels are opened once under this lock")
    lock_path = paths.out / LOCK_NAME
    lock = read_json(lock_path)
    if lock is None:
        raise RuntimeError("no E3_LOCK.json: run the lock stage first")
    problems = verify_lock(lock, paths)
    if expected_lock_sha256 is not None and sha256_file(lock_path) != expected_lock_sha256:
        problems["lock_is_not_the_reviewed_lock"] = {"expected": expected_lock_sha256, "now": sha256_file(lock_path)}
    if problems:
        return _blocked(paths.out, "run", problems, lock=str(lock_path))
    parts = {"verify_lock": _part(tick)}
    tick_load = time.perf_counter()
    report = read_json(paths.report)
    weights = contrast_weights(report)
    values = report_values(report, weights)
    inputs = load_inputs(paths, selected_cells(weights)["all"])
    bad = sorted(name for name, ok in inputs.alignment.items() if not ok)
    if bad:
        raise RuntimeError(f"alignment checks failed: {bad}")
    _, kills_reason = kill_labels(inputs.n_kills)
    parts["load"] = _part(tick_load)

    # The opening is recorded in tog_e3_1 and OPENINGS.json before the sealed file is read (EXPOSURE_KO.md rule 1).
    before = _openings(paths.openings)
    if any(o.get("experiment") == "E3" for o in before):
        raise RuntimeError("E3 has already opened its sealed labels (OPENINGS.json)")
    entry = {"experiment": "E3", "utc": utc(), "lock": str(lock_path), "lock_sha256": sha256_file(lock_path),
             "prespec_sha256": lock["prespec"]["sha256"], "labels": [str(paths.sealed)],
             "sealed_sha256": lock["verify"]["sealed"]["sha256"], "note": OPENING_NOTE}
    write_once(paths.out / OPENING_NAME, _clean({**entry, "openings_before": before,
                                                 "external_first_opening": lock.get("external_first_opening")}))
    append_opening(paths.openings, entry)
    try:
        return _evaluate(paths, lock, weights, values, inputs, kills_reason, entry, before, parts, started, tick)
    except BaseException as error:
        _failure(paths.out, entry, error, parts)
        raise


def _evaluate(paths: Paths, lock: Mapping[str, Any], weights: Mapping[str, Mapping[str, float]], values: Mapping[str, Any],
              inputs: Inputs, kills_reason: str | None, entry: Mapping[str, Any], before: list, parts: dict[str, Any],
              started: str, tick: float) -> int:
    lock_path = paths.out / LOCK_NAME
    tick_part = time.perf_counter()
    y = sealed_labels(paths.sealed, inputs.keys)                     # the one read of the sealed file
    shared, aligned = cell_losses(y, inputs.preds, inputs.match)
    if not np.array_equal(shared, inputs.match_index):
        raise RuntimeError("the per-match losses are not in the player-matrix order")
    D = {name: per_match(aligned, weights[name]) for name in CONTRASTS}
    points = {name: float(np.mean(D[name])) for name in CONTRASTS}
    point_checks = {name: point_check(points[name], values["points"][name]) for name in REPORTED}
    boot = pooled_bootstrap(shared, aligned, weights)
    repro = reproduction(boot["contrasts"][PRIMARY]["interval"], values["intervals"][PRIMARY])
    repro["bootstrap_point_equals_report_point"] = boot["contrasts"][PRIMARY]["point"] == values["points"][PRIMARY]
    parts["losses_and_checks"] = _part(tick_part)
    failed = {}
    if not all(check["passed"] for check in point_checks.values()):
        failed["point_check_failed"] = point_checks
    if not repro["reproduced"]:
        failed["bootstrap_reproduction_failed"] = repro
    if failed:
        return _blocked(paths.out, "run", failed, opening=entry, lock=str(lock_path), stages=parts)

    tick_part = time.perf_counter()
    pooled = {name: {"twoway": interval_entry(twoway_interval(D[name], inputs.player_idx, LEVEL)),
                     "bootstrap": _boot_entry(boot["contrasts"][name], boot)} for name in CONTRASTS}
    interval = pooled[PRIMARY]["twoway"]["interval"]
    outcome = decide(interval)
    decision = {"contrast": PRIMARY, "point": pooled[PRIMARY]["twoway"]["point"], "interval": interval, "level": LEVEL,
                "z": pooled[PRIMARY]["twoway"]["z"], "variance": "V = max(V1, VU)", "contains_zero": outcome == "lowered",
                "outcome": outcome, "text": DECISIONS[outcome], "rule": "PRESPEC_E3 section 4"}
    parts["pooled"] = _part(tick_part)

    tick_part = time.perf_counter()
    by_patch = {}
    for patch in PATCHES:
        mask = inputs.patch_of_match == patch
        by_patch[patch] = {"rows": int((inputs.row_patch == patch).sum()), "matches": int(mask.sum()),
                           "contrasts": {name: interval_entry(twoway_interval(D[name][mask], inputs.player_idx[mask], LEVEL))
                                         for name in CONTRASTS}}
    spread = {name: dispersion({p: by_patch[p]["contrasts"][name]["point"] for p in PATCHES},
                               {p: by_patch[p]["contrasts"][name]["V"] for p in PATCHES}, pooled[name]["twoway"]["point"])
              for name in CONTRASTS}
    patch_checks = {p: {name: point_check(by_patch[p]["contrasts"][name]["point"], values["points_by_patch"][p][name])
                        for name in REPORTED} for p in PATCHES}
    parts["by_patch"] = _part(tick_part)

    tick_part = time.perf_counter()
    if kills_reason:
        strata: dict[str, Any] = {"blocked": True, "reason": kills_reason}
    else:
        strata = kill_strata(y, inputs.preds, inputs.match, inputs.n_kills, weights["P1"], inputs.match_index,
                             inputs.player_idx)
    parts["kill_strata"] = _part(tick_part)

    d1 = {**point_check(points["D1"], values["d1_metrics_difference"]),
          "note": "descriptive: the report's D1 point is the difference of the two cells' match-weighted raw Brier "
                  "(PRESPEC_E3 section 2); not a section 6 condition"}
    sentence = sentence_15_16(pooled[PRIMARY]["twoway"]["design_effect"], pooled["P1"]["twoway"]["design_effect"])
    counts = {key: value for key, value in lock["counts"].items() if key != "players"}
    inputs_used = [paths.players, paths.rows, paths.report, paths.sealed, lock_path] + [
        paths.score(key, "npz") for key in inputs.preds]
    payload = {
        "version": VERSION, "experiment": "E3", "dataset": DATASET, "utc": utc(),
        "prespec": {"path": str(paths.prespec), "sha256": lock["prespec"]["sha256"], "commit": lock["prespec"]["commit"]},
        "lock": {"path": str(lock_path), "sha256": entry["lock_sha256"], "locked_utc": lock["locked_utc"]},
        "opening": entry, "openings_before": before, "labels_opened": entry["labels"], "rules": lock["rules"],
        "counts": counts,
        "checks": {"points": point_checks, "bootstrap_reproduction": repro, "d1_point_vs_report_metrics": d1,
                   "points_by_patch_vs_report": patch_checks},
        "pooled": pooled, "decision": decision, "by_patch": by_patch, "dispersion": spread, "kill_strata": strata,
        "descriptive": {"players": lock["counts"]["players"], "test_15_16_sentence": sentence,
                        "p1_level_note": "EXTERNAL_REPORT.json levels.P1 is 0.975; E3 reports P1 at 0.95 (PRESPEC_E3 "
                                         "section 3)"},
        "environment": environment(),
        "stages": {"lock": lock["stage"], "run": {"started_utc": started, "finished_utc": utc(),
                                                 "wall_seconds": time.perf_counter() - tick,
                                                 "peak_memory_bytes_main_process": peak_memory_bytes(), "parts": parts,
                                                 "processes": "main process only, no worker, no GPU"}},
        "run": run_record(rows=counts["rows"], matches=counts["matches"], inputs=inputs_used,
                          code=[ROOT / name for name in CODE_FILES], seed=SEED, started_utc=started),
    }
    write_once(paths.out / REPORT_NAME, _clean(payload))
    _print_summary(pooled, decision, strata)
    return 0


# ---------------------------------------------------------------- records and helpers

def append_opening(path: Path, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Append one opening to OPENINGS.json (created when missing; EXPOSURE_KO.md rule 1); returns the earlier ones."""

    path = Path(path)
    payload = read_json(path) if path.exists() else {"rule": OPENINGS_RULE, "openings": []}
    before = list(payload.get("openings", []))
    if any(o.get("experiment") == entry["experiment"] for o in before):
        raise RuntimeError(f"{entry['experiment']} has already opened its sealed labels")
    payload["openings"] = before + [dict(entry)]
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    pending.write_text(json.dumps(_clean(payload), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(pending, path)
    return before


def _openings(path: Path) -> list[dict[str, Any]]:
    return list((read_json(path) or {}).get("openings", [])) if Path(path).exists() else []


def _part(tick: float) -> dict[str, Any]:
    return {"wall_seconds": time.perf_counter() - tick, "peak_memory_bytes_main_process": peak_memory_bytes()}


def _stage(name: str, started: str, tick: float, **extra: Any) -> dict[str, Any]:
    return {"stage": name, "started_utc": started, "finished_utc": utc(), "wall_seconds": time.perf_counter() - tick,
            "peak_memory_bytes_main_process": peak_memory_bytes(), "processes": "main process only, no worker, no GPU",
            **extra}


def _next_free(out: Path, stem: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    k = 1
    while (out / f"{stem}_{k}.json").exists():
        k += 1
    return out / f"{stem}_{k}.json"


def _blocked(out: Path, which: str, reasons: Mapping[str, Any], **extra: Any) -> int:
    """Write E3_BLOCKED_<which>_<n>.json once (n = the first free number, so earlier attempts stay) and return 2."""

    path = _next_free(out, f"E3_BLOCKED_{which}")
    write_once(path, _clean({"version": VERSION, "experiment": "E3", "blocked_stage": which, "blocked": True, "utc": utc(),
                             "reasons": reasons, **extra}))
    print(json.dumps({"BLOCKED": which, "file": str(path), "reasons": sorted(reasons)}))
    return 2


def _failure(out: Path, entry: Mapping[str, Any], error: BaseException, parts: Mapping[str, Any]) -> None:
    """E3_FAILED_run_<n>.json: a failure after the opening leaves a record (no data values, message cut to 500)."""

    path = _next_free(out, "E3_FAILED_run")
    lines = traceback.format_exception(type(error), error, error.__traceback__)
    write_once(path, _clean({"version": VERSION, "experiment": "E3", "failed": True, "utc": utc(), "opening": dict(entry),
                             "error": {"type": type(error).__name__, "message": str(error)[:500],
                                       "traceback_tail": "".join(lines[-6:])[-3000:]},
                             "stages": dict(parts),
                             "rule": "do not re-run under this lock; the opening stays recorded (Ruling 31 practice)"}))
    print(json.dumps({"FAILED": "run", "file": str(path), "error": type(error).__name__}))


def _clean(value: Any) -> Any:
    """JSON-ready copy: numpy scalars and arrays to Python, tuples to lists, paths to text, NaN and inf to null."""

    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _print_summary(pooled: Mapping[str, Any], decision: Mapping[str, Any], strata: Mapping[str, Any]) -> None:
    for name in CONTRASTS:
        t = pooled[name]["twoway"]
        b = pooled[name]["bootstrap"]
        print(f"E3 {name}: theta={t['point']:.6f} V-interval=[{t['interval'][0]:.6f}, {t['interval'][1]:.6f}] "
              f"DE={t['design_effect']:.3f} theta/sqrtV={t['theta_over_se_twoway']:.2f} z={t['z']} "
              f"bootstrap=[{b['interval'][0]:.6f}, {b['interval'][1]:.6f}]")
    if not strata.get("blocked"):
        d = strata["difference"]
        print(f"E3 Delta(4+ - 1): {d['point']:.6f} [{d['twoway']['interval'][0]:.6f}, {d['twoway']['interval'][1]:.6f}]")
    print("decision:", decision["outcome"], decision["interval"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("stage", choices=("lock", "run"))
    parser.add_argument("--confirm-second-opening", action="store_true",
                        help="run: required; the step opens the external sealed labels a second time, once")
    parser.add_argument("--lock-sha256", help="run: required; the SHA-256 of the reviewed E3_LOCK.json")
    args = parser.parse_args(argv)
    if args.stage == "lock":
        return lock_stage()
    if not (args.confirm_second_opening and args.lock_sha256):
        parser.error("the run stage opens the external sealed labels a second time; pass --confirm-second-opening "
                     "and --lock-sha256 <SHA-256 of the reviewed lock>")
    return run_stage(expected_lock_sha256=args.lock_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
