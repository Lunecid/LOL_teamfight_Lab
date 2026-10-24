"""q fitting: STOP early stopping, fixed-epoch REFIT, exact resume after a crash, reload, tables."""
import json

import numpy as np
import pytest
import torch
from scipy import sparse

from studies.q_production_20261005 import qtrain


def _data(n=600, d=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    logits = x @ rng.normal(size=d) * 0.8
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(np.float32)
    match = rng.integers(0, n // 3, size=n)
    _, inverse, counts = np.unique(match, return_inverse=True, return_counts=True)
    w_raw = 1.0 / counts[inverse]
    return x, y, w_raw, match


def _sources(x, y, w_raw, batch=64, as_csr=False, fail_at=None):
    n = len(y)
    parts = {"fit": np.arange(0, int(n * 0.6)), "stop": np.arange(int(n * 0.6), int(n * 0.75)), "refit": np.arange(0, int(n * 0.75)),
             "q_dev": np.arange(int(n * 0.75), int(n * 0.9)), "q_cal": np.arange(int(n * 0.9), n)}
    calls = {"n": 0}

    def make(rows):
        w_norm = w_raw[rows] * len(rows) / w_raw[rows].sum()

        def source():
            for start in range(0, len(rows), batch):
                sl = rows[start:start + batch]
                calls["n"] += 1
                if fail_at is not None and calls["n"] == fail_at:
                    raise KeyboardInterrupt("simulated power loss")
                inputs = sparse.csr_matrix(x[sl]) if as_csr else x[sl]
                yield qtrain.Batch(inputs, y[sl], w_norm[start:start + len(sl)], w_raw[sl], sl)
        return source
    return {name: make(rows) for name, rows in parts.items()}


class _Net(torch.nn.Module):
    def __init__(self, d=6):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 16), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _forward(model, inputs, device):
    return model(torch.as_tensor(inputs, device=device))


RULES = qtrain.NeuralRules(batch_size=64, max_epochs=15, patience=3, weight_decay=1e-3)


def test_fit_stops_on_stop_brier_and_refits_for_the_best_epoch_count(tmp_path):
    qtrain.configure_runtime()
    x, y, w, _ = _data()
    result = qtrain.fit_neural(lambda: _Net(), _forward, _sources(x, y, w), RULES, tmp_path / "job", {"job": "a"})
    fit = json.loads((tmp_path / "job" / "FIT_STATE.json").read_text())
    briers = [e["stop_brier"] for e in fit["epochs"]]
    best = int(np.argmin(briers)) + 1
    assert result["best_epoch"] == fit["best_epoch"] and result["refit_epochs"] == result["best_epoch"]
    assert len(fit["epochs"]) == min(RULES.max_epochs, fit["best_epoch"] + RULES.patience)
    assert briers[fit["best_epoch"] - 1] <= min(briers) + RULES.min_delta and best >= 1
    assert result["reload"]["passed"] and set(result["scores"]) == {"q_dev", "q_cal"}


def test_resume_after_a_crash_reproduces_the_uninterrupted_run(tmp_path):
    x, y, w, _ = _data(seed=1)
    qtrain.configure_runtime()
    clean = qtrain.fit_neural(lambda: _Net(), _forward, _sources(x, y, w), RULES, tmp_path / "clean", {"job": "b"})
    qtrain.configure_runtime()
    with pytest.raises(KeyboardInterrupt):
        qtrain.fit_neural(lambda: _Net(), _forward, _sources(x, y, w, fail_at=23), RULES, tmp_path / "crash", {"job": "b"})
    resumed = qtrain.fit_neural(lambda: _Net(), _forward, _sources(x, y, w), RULES, tmp_path / "crash", {"job": "b"})
    assert resumed["best_epoch"] == clean["best_epoch"] and resumed["model_sha256"] == clean["model_sha256"]
    np.testing.assert_array_equal(np.load(tmp_path / "clean" / "pred_q_dev.npz")["logit"], np.load(tmp_path / "crash" / "pred_q_dev.npz")["logit"])


def test_state_of_another_job_is_never_continued(tmp_path):
    x, y, w, _ = _data(seed=2)
    qtrain.configure_runtime()
    qtrain.fit_neural(lambda: _Net(), _forward, _sources(x, y, w), RULES, tmp_path / "job", {"job": "c"})
    with pytest.raises(ValueError, match="another job"):
        qtrain.fit_neural(lambda: _Net(), _forward, _sources(x, y, w), RULES, tmp_path / "job", {"job": "different"})


def test_match_brier_weights_each_match_equally():
    p, yv = np.asarray([0.9, 0.9, 0.2]), np.asarray([1, 0, 0])
    w = np.asarray([0.5, 0.5, 1.0])          # match A has two rows, match B one
    assert np.isclose(qtrain.match_brier(p, yv, w), ((0.01 + 0.81) / 2 + 0.04) / 2)


def test_lightgbm_early_stops_on_weighted_brier_and_logistic_fits(tmp_path):
    x, y, w, _ = _data(n=900, seed=3)
    result = qtrain.fit_lightgbm({"num_leaves": 15, "min_data_in_leaf": 20}, _sources(x, y, w, as_csr=True), tmp_path / "lgbm", {"job": "d"})
    assert result["status"] == "completed" and 1 <= result["best_iteration"] <= 2000 and result["reload"]["max_abs_logit_diff"] == 0.0
    logistic = qtrain.fit_logistic({"C": 0.1}, _sources(x, y, w, as_csr=True), tmp_path / "lr")
    assert logistic["status"] == "completed" and logistic["scores"]["q_dev"]["rows"] == 135
