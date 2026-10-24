"""qtest: SCORE_LOCK gating, write-once files, and the independent metric route (no real models)."""
import json

import numpy as np
import pytest

from studies.q_production_20261005 import qeval, qtest


def test_independent_metrics_match_qeval():
    rng = np.random.default_rng(7)
    n = 5000
    match = np.sort(rng.integers(0, 900, size=n))
    y = rng.integers(0, 2, size=n).astype(float)
    p = np.clip(rng.beta(2, 2, size=n), 1e-6, 1 - 1e-6)
    p[::17] = 0.5  # ties
    raw = qeval.evaluate(y, p, match)["match_weighted"]
    mine = qtest._independent(y, p, match)
    for name in ("brier", "logloss", "auc"):
        assert abs(raw[name] - mine[name]) <= qtest.RECOMPUTE_ATOL


def test_key_of_matches_supervisor_naming():
    assert qtest.key_of("final/module/X/lightgbm/selected:lightgbm") == "final__module__X__lightgbm__selected~lightgbm"


def test_write_once_refuses_second_write(tmp_path):
    target = tmp_path / "A.json"
    qtest.write_once(target, {"a": 1})
    with pytest.raises(FileExistsError):
        qtest.write_once(target, {"a": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}


def test_lock_refuses_while_final_jobs_are_open(tmp_path):
    (tmp_path / "COVERAGE.json").write_text(json.dumps({"by_block": {"search/common": {"completed": 40},
                                                                     "final/module": {"completed": 3, "pending": 1}}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="still open"):
        qtest.lock(tmp_path)


def test_lock_refuses_twice(tmp_path):
    (tmp_path / qtest.LOCK_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        qtest.lock(tmp_path)


def test_coverage_open_ignores_search_blocks(tmp_path):
    (tmp_path / "COVERAGE.json").write_text(json.dumps({"by_block": {"search/common": {"pending": 2},
                                                                     "pure_set/pure_set": {"running": 1}}}), encoding="utf-8")
    assert qtest.coverage_open(tmp_path) == {"pure_set/pure_set": {"running": 1}}


def test_evaluate_requires_lock(tmp_path):
    with pytest.raises(RuntimeError, match="SCORE_LOCK"):
        qtest.evaluate(tmp_path)
