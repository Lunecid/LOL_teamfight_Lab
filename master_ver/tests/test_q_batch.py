"""q job inputs: superset lanes, condition masks, family inputs, and an end-to-end fit through the sources."""
import hashlib

import numpy as np
import pytest
import torch

from studies.q_production_20261005 import features as qf
from studies.q_production_20261005 import qbatch, qmodels, qtrain, rows as qrows
from tests.test_q_features import _row


def _context(tmp_path, n=48, scope="REFIT_TRAIN"):
    semantic = [_row(seed) for seed in range(n)]
    writer = qf.RawStoreWriter(tmp_path / "store", np.arange(n, dtype=np.int64) * 2)
    for i, row in enumerate(semantic):
        writer.add(i * 2, row)
    writer.close()
    store = qf.RawStore(tmp_path / "store")
    rng = np.random.default_rng(0)
    match = np.arange(n) // 3
    keys = np.asarray([hashlib.sha256(f"m{m}".encode()).hexdigest().encode() for m in match], dtype="S64")
    role = np.where(np.arange(n) < 30, 0, 3).astype(np.int8)
    rows = {"query_pre": np.arange(n, dtype=np.int64) * 2, "role": role, "q_stop": (match % 5 == 0) & (role == 0),
            "canonical_match_key": keys, "y": (rng.random(n) < 0.5).astype(np.int8), "weight_raw": qrows.match_weights(match),
            "match_index": match}
    xy = rng.uniform(500, 14500, size=(n, 10, 2))
    return qbatch.ScopeContext(store, rows, xy, scope), rows


def test_superset_lanes_and_masks(tmp_path):
    context, _ = _context(tmp_path)
    rows = np.arange(10)
    structured, graph = qbatch.assemble(context, rows, qbatch.Condition(G=1), "relational_mpnn")
    assert structured.node_numeric.shape == (10, 10, 133) and structured.team_numeric.shape == (10, 2, 13)
    assert structured.global_numeric.shape == (10, 80) and graph.edge.shape == (10, 10, 9, 3)
    s0, g0 = qbatch.assemble(context, rows, qbatch.Condition(G=0, S=0, A=0, M=0), "team_set_pair_readout")
    assert not s0.node_numeric[..., 102:].any() and not s0.team_numeric[..., 8:].any() and not g0.edge.any()
    no_m_r = qbatch.assemble(context, rows, qbatch.Condition(G=1, S=1, A=0, M=1, E=1, R=0), "relational_mpnn")[0]
    m = no_m_r.node_numeric[..., 109:].reshape(10, 10, 8, 3)
    assert m[:, :, :2].any() and not m[:, :, 2:].any()            # E stats kept, R stats removed with their parent
    slot = qbatch.assemble(context, rows, qbatch.Condition(), "shared_slot_mlp")
    assert slot.global_numeric.shape == (10, 280)
    pure = qbatch.assemble(context, rows, qbatch.Condition(S=1, A=0, M=0), "pure_set")
    assert pure.node_numeric.shape == (10, 10, 104)
    table = qbatch.assemble(context, rows, qbatch.Condition(), "lightgbm")
    base = context.transform(context.positions[rows]).table_csr
    assert table.shape == (10, base.shape[1] + 504)
    b_only = qbatch.assemble(context, rows, qbatch.Condition(S=0, A=0, M=0, E=0, C=0, R=0, D=0, inputs="B"), "lightgbm")
    assert b_only[:, base.shape[1]:].nnz == 0
    with pytest.raises(ValueError, match="invalid condition"):
        qbatch.assemble(context, rows, qbatch.Condition(S=0, A=1), "lightgbm")


def test_scope_statistics_use_only_scope_rows(tmp_path):
    fit, rows = _context(tmp_path / "a", scope="FIT")
    refit, _ = _context(tmp_path / "b", scope="REFIT_TRAIN")
    train = np.asarray(rows["role"]) == 0
    assert set(fit.scope_rows) == set(np.flatnonzero(train & ~rows["q_stop"])) and set(refit.scope_rows) == set(np.flatnonzero(train))
    assert np.isclose(fit.sigma, qbatch.spatial.fit_sigma(fit.xy[fit.scope_rows]))
    assert fit.pre.evidence.scope == "FIT" and refit.pre.evidence.scope == "REFIT_TRAIN"


def test_end_to_end_graph_fit_through_sources(tmp_path):
    qtrain.configure_runtime()
    context, rows = _context(tmp_path)
    train = np.flatnonzero(np.asarray(rows["role"]) == 0)
    fit_rows, stop_rows = train[~rows["q_stop"][train]], train[rows["q_stop"][train]]
    q_dev = np.flatnonzero(np.asarray(rows["role"]) == 3)
    cond, family = qbatch.Condition(G=1), "relational_mpnn"
    w = rows["weight_raw"]
    sources = {"fit": qbatch.source(context, fit_rows, cond, family, 8, qrows.normalized_weights(rows["match_index"][fit_rows])),
               "stop": qbatch.source(context, stop_rows, cond, family, 8, w[stop_rows]),
               "refit": qbatch.source(context, train, cond, family, 8, qrows.normalized_weights(rows["match_index"][train])),
               "q_dev": qbatch.source(context, q_dev[:9], cond, family, 8, w[q_dev[:9]]),
               "q_cal": qbatch.source(context, q_dev[9:], cond, family, 8, w[q_dev[9:]])}
    build = lambda: qmodels.QGraphModel(family, hidden_width=16, encoder_kwargs=context.encoder_kwargs(family))  # noqa: E731
    result = qtrain.fit_neural(build, qbatch.forward_for(family), sources, qtrain.NeuralRules(max_epochs=4, patience=2), tmp_path / "job", {"job": "e2e"})
    assert result["status"] == "completed" and result["reload"]["passed"] and result["scores"]["q_dev"]["rows"] == 9
