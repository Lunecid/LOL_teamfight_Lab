"""Synthetic CPU checks of ``c_ept.evaluation.proper`` (S4b-prep M10, D11): the proper scores, the PIT histogram, the quantile hits,
the Brier score and the weighted mean in one module. No data, no model, no device.

``report_s3`` (frozen S3 code) is imported by its public names only, as the reference the new module has to equal on fixtures.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from c_ept.evaluation import metrics, proper
from c_ept.reporting import report_s3
from c_ept.valuation import compose

HERE = Path(__file__).resolve().parent


def q19_fixture(seed=3, n=40):
    """Quantile rows (non-decreasing along the row), including rows with ties (point masses) and a constant row; realized values that
    hit a quantile exactly; weights of different size."""
    rng = np.random.default_rng(seed)
    q = np.sort(rng.normal(0.0, 300.0, size=(n, 19)), axis=1)
    q[0] = 120.0                                       # a point mass: all 19 quantiles equal
    q[1, 5:12] = q[1, 5]                               # a tie block inside a row
    y = rng.normal(0.0, 300.0, size=n)
    y[2] = q[2, 7]                                     # realized exactly at a quantile
    y[0] = 120.0                                       # realized at the point mass
    w = rng.uniform(0.2, 3.0, size=n)
    return q, y, w


def test_levels_are_the_composer_levels():
    assert proper.LEVELS is compose.Q_LEVELS
    assert proper.LEVELS.shape == (19,) and proper.LEVELS[0] == 1 / 20 and proper.LEVELS[-1] == 19 / 20


def test_crps19_equals_the_s3_report_values():
    # bit for bit, not within a tolerance: the same arithmetic as the frozen function
    for seed in (3, 4, 5):
        q, y, w = q19_fixture(seed)
        assert proper.crps19(q, y, w) == report_s3.crps19(q, y, w)
        a, b = proper.pit_histogram(q, y, w), report_s3.pit_histogram(q, y, w)
        assert set(a) == set(b)
        for key in a:
            assert np.array_equal(a[key], b[key]), key
        a, b = proper.quantile_hits(q, y, w), report_s3.quantile_hits(q, y, w)
        assert set(a) == set(b)
        for key in a:
            assert np.array_equal(a[key], b[key]), key


def test_crps19_hand_values():
    # all quantiles equal: the pinball proxy is the absolute error (the S3 R81 statement), weighted
    assert proper.crps19(np.zeros((1, 19)), np.array([10.0]), np.ones(1)) == pytest.approx(10.0)
    assert proper.crps19(np.zeros((2, 19)), np.array([10.0, -4.0]), np.array([1.0, 3.0])) == pytest.approx(5.5)


@pytest.mark.parametrize('fn', ['crps19', 'pit_histogram', 'quantile_hits'])
def test_the_quantile_scores_refuse_what_cannot_be_judged(fn):
    f = getattr(proper, fn)
    q, y, w = q19_fixture(n=4)
    with pytest.raises(ValueError):
        f(q[:, :18], y, w)                              # not 19 levels
    with pytest.raises(ValueError):
        f(q[:0], y[:0], w[:0])                          # no row
    with pytest.raises(ValueError):
        f(q, y[:3], w)                                  # shapes
    bad = q.copy()
    bad[1, 3] = np.nan
    with pytest.raises(ValueError):
        f(bad, y, w)
    bad = q.copy()
    bad[1, 4], bad[1, 5] = bad[1, 5], bad[1, 4] - 1.0   # a decreasing row
    with pytest.raises(ValueError):
        f(bad, y, w)
    with pytest.raises(ValueError):
        f(q, y, -w)                                     # negative weights
    with pytest.raises(ValueError):
        f(q, y, np.zeros(4))                            # weights add up to 0


def test_brier_hand_example():
    # (0.5 - 1)^2 = 0.25 with weight 1 and (1 - 0)^2 = 1 with weight 3: (0.25 + 3) / 4
    assert proper.brier(np.array([0.5, 1.0]), np.array([1, 0]), np.array([1.0, 3.0])) == pytest.approx(0.8125, abs=0, rel=1e-15)
    with pytest.raises(ValueError):
        proper.brier(np.array([1.2]), np.array([1]), np.ones(1))      # not a probability
    with pytest.raises(ValueError):
        proper.brier(np.array([0.2]), np.array([2]), np.ones(1))      # not a binary label


def test_weighted_mean_is_the_normalised_dot_product():
    v, w = np.array([1.0, 2.0, 6.0]), np.array([1.0, 1.0, 2.0])
    assert proper.weighted_mean(v, w) == float(np.dot(v, w / w.sum())) == 3.75
    with pytest.raises(ValueError):
        proper.weighted_mean(v, np.zeros(3))
    with pytest.raises(ValueError):
        proper.weighted_mean(v, w[:2])
    with pytest.raises(ValueError):
        proper.weighted_mean(np.array([]), np.array([]))


def endpoint_fixture(seed=11, n_matches=12):
    """Synthetic endpoints and NLL entries for the metrics.py bit-identity pin (no data)."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 6, size=n_matches)
    match = np.repeat(np.arange(n_matches), sizes)
    row = np.concatenate([np.arange(s) for s in sizes])
    n = len(match)
    y = (rng.uniform(size=n) < 0.4).astype(int)
    endpoints = [{'name': 'onset60', 'module': 'M2', 'layer': 'rate', 'horizon_s': 60.0, 'target': 'onset',
                  'match': match, 'row': row, 'y': y,
                  'p_ept': np.clip(0.4 + 0.2 * (y - 0.4) + rng.normal(0, 0.1, n), 0, 1),
                  'p_b1': np.clip(0.4 + 0.1 * (y - 0.4) + rng.normal(0, 0.1, n), 0, 1)}]
    nll = [{'name': 'h1', 'match': match, 'row': row, 'nll_ept': rng.uniform(0.5, 2, n), 'nll_b1': rng.uniform(0.5, 2, n),
            'scale': 1.5, 'common_head': True},
           {'name': 'h2', 'match': match, 'row': row, 'nll_ept': rng.uniform(0.5, 2, n), 'nll_b1': rng.uniform(0.5, 2, n)}]
    return endpoints, nll


# sha256 of the canonical JSON of summarize_endpoints(endpoint_fixture(), bootstrap=200, seed=20261022), taken from the UNMODIFIED
# metrics.py at f116532 before it was made to wrap proper.weighted_mean: every old output stays bit for bit.
METRICS_DIGEST = 'af47b6c6a2074cbb73445643a7c25070205bc3a9a224b79d3c68244b97a584ec'


def test_metrics_summary_is_bit_identical_to_the_pre_m10_output():
    endpoints, nll = endpoint_fixture()
    out = metrics.summarize_endpoints(endpoints, nll, bootstrap=200, seed=20261022)
    text = json.dumps(out, sort_keys=True, allow_nan=False)
    assert hashlib.sha256(text.encode('utf-8')).hexdigest() == METRICS_DIGEST


def test_metrics_wraps_the_proper_weighted_mean():
    import ast
    tree = ast.parse((HERE.parent / 'evaluation' / 'metrics.py').read_text(encoding='utf-8'))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == 'c_ept.evaluation' for a in n.names}
    assert 'proper' in names
