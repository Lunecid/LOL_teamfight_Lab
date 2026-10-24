"""V report helpers: the sorted weighted AUC equals sklearn's, and the match bootstrap is deterministic and sensible."""
import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("v_report", ROOT / "reports" / "v_report_20261005" / "v_report.py")
vr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vr)


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    match = rng.integers(0, 300, n)
    p = np.round(rng.uniform(size=n), 2)                      # rounding creates ties
    y = (rng.uniform(size=n) < p).astype(np.float64)
    return y, p, match


def test_sorted_auc_matches_sklearn_with_ties_and_weights():
    y, p, match = _data()
    w = np.random.default_rng(1).uniform(0.1, 2.0, len(y))
    order = np.argsort(p, kind="stable")
    ps = p[order]
    groups = np.flatnonzero(np.concatenate(([True], ps[1:] != ps[:-1])))
    assert abs(vr._auc_sorted(y[order], w[order], groups) - roc_auc_score(y, p, sample_weight=w)) < 1e-12


def test_report_tables_and_outputs(tmp_path):
    y, p, match = _data(6000)
    rng = np.random.default_rng(2)
    arrays = {"y": y, "p_raw": p, "p_cal": np.clip(p * 0.9 + 0.05, 0, 1), "match_index": match, "query_ms": rng.uniform(60_000, 3_000_000, len(y)),
              "query_index": np.arange(len(y), dtype=np.int64), "logit": np.zeros(len(y)), "game_key": np.asarray([b"k"] * len(y), dtype="S64")}
    scored = {"arrays": arrays, "family": "lightgbm", "card_id": "c", "check": {"rows": len(y)}, "lock_sha256": "s"}
    result = vr.report(scored)
    assert result["rows"] == len(y) and set(result["minute"]["raw"]) == set(vr.qeval.MINUTE_ORDER)
    assert result["minute"]["raw"]["[0,2)"]["suppressed"] or result["minute"]["raw"]["[0,2)"]["rows"] >= 100
    paths = vr.write_outputs(tmp_path, "final", scored, result, 1.0)
    lines = Path(paths["minute"]).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + len(vr.qeval.MINUTE_ORDER)


def test_stratum_bootstrap_is_deterministic_and_brackets_the_point():
    y, p, match = _data()
    labels = np.where(np.arange(len(y)) % 2 == 0, "a", "b")
    a = vr.stratum_bootstrap(y, p, match, labels, ["a", "b"], resamples=200)
    b = vr.stratum_bootstrap(y, p, match, labels, ["a", "b"], resamples=200)
    assert a == b
    rows = labels == "a"
    _, local, m = np.unique(match[rows], return_inverse=True, return_counts=True)
    point = roc_auc_score(y[rows], p[rows], sample_weight=1.0 / m[local])
    lo, hi = a["a"]["auc"]
    assert lo < point < hi and hi - lo < 0.1
