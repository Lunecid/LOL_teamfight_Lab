"""q worker: saved scope contexts reload exactly, every family kind runs from a spec, release-bound CLI, calibration."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from studies.q_production_20261005 import features as qf
from studies.q_production_20261005 import q_worker, qbatch, qtrain, release, rows as qrows
from tests.test_q_features import _row

N = 60


def _rows(n=N, seed=0):
    rng = np.random.default_rng(seed)
    match = np.arange(n) // 3
    keys = np.asarray([hashlib.sha256(f"m{m}".encode()).hexdigest().encode() for m in match], dtype="S64")
    role = np.select([match < 12, match < 16], [0, 3], 4).astype(np.int8)
    return {"canonical_match_key": keys, "engagement_key": np.char.add(keys, np.asarray([f":{i}".encode() for i in range(n)])),
            "match_index": match, "role": role, "q_stop": (match % 5 == 0) & (role == 0), "y": (rng.random(n) < 0.5).astype(np.int8),
            "weight_raw": qrows.match_weights(match), "query_pre": np.arange(n, dtype=np.int64) * 2, "pre_ms": rng.integers(120_000, 2_400_000, n),
            "elapsed_min": rng.uniform(2, 40, n), "pt_p_pre": rng.uniform(0.05, 0.95, n), "label_svi_delta": rng.normal(0, 0.05, n)}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("q_run")
    writer = qf.RawStoreWriter(root / "raw" / "store", np.arange(N, dtype=np.int64) * 2)
    for i in range(N):
        writer.add(i * 2, _row(i))
    writer.close()
    (root / "raw" / "store" / "BUILD.json").write_text("{}", encoding="utf-8")
    rows = _rows()
    rows_sha = qrows.assemble.write_npz(root / "Q_ROWS_h90.npz", rows)
    base = root / "inputs" / "main_h90"
    base.mkdir(parents=True)
    xy = np.random.default_rng(1).uniform(500, 14500, size=(N, 10, 2))
    np.save(base / "XY.npy", xy)
    store = qf.RawStore(root / "raw" / "store")
    provenance = {"rows_sha256": rows_sha, "xy_sha256": qrows.sha256_file(base / "XY.npy"), "store_build_sha256": "test"}
    for scope in ("FIT", "REFIT_TRAIN"):
        qbatch.ScopeContext(store, rows, xy, scope).save(base / scope, provenance)
    inputs = {"rows": str(root / "Q_ROWS_h90.npz"), "rows_sha256": rows_sha, "xy": str(base / "XY.npy"), "xy_sha256": provenance["xy_sha256"],
              "store": str(root / "raw" / "store"), "contexts": {s: str(base / s) for s in ("FIT", "REFIT_TRAIN")}}
    return root, rows, xy, inputs


def _spec(run, job_id, family, card, factors, inputs, out, job_inputs="condition"):
    job = {"job_id": job_id, "family": family, "card": card, "factors": factors, "inputs": job_inputs, "identity": hashlib.sha256(job_id.encode()).hexdigest()}
    return {"job": job, "variant": "main_h90", "inputs": inputs, "out_dir": str(out), "device": "cpu", "release": "", "release_digest": "test"}


@pytest.fixture
def quick_rules(monkeypatch):
    monkeypatch.setattr(q_worker.qtrain, "neural_rules", lambda family, wd=None: qtrain.NeuralRules(batch_size=16, max_epochs=3, patience=1, weight_decay=float(wd or 0.0)))


def test_saved_context_reloads_exactly(run_dir, tmp_path):
    root, rows, xy, inputs = run_dir
    store = qf.RawStore(root / "raw" / "store")
    fitted = qbatch.ScopeContext(store, rows, xy, "REFIT_TRAIN")
    loaded = qbatch.ScopeContext(store, rows, xy, "REFIT_TRAIN", saved=inputs["contexts"]["REFIT_TRAIN"])
    assert loaded.sigma == fitted.sigma and np.array_equal(loaded.m_median, fitted.m_median)
    idx = np.arange(N)
    for family in ("relational_mpnn", "lightgbm"):
        a = qbatch.assemble(fitted, idx, qbatch.Condition(G=1), family)
        b = qbatch.assemble(loaded, idx, qbatch.Condition(G=1), family)
        if family == "lightgbm":
            assert (a != b).nnz == 0
        else:
            for lane in ("node_numeric", "team_numeric", "global_numeric"):
                np.testing.assert_array_equal(getattr(a[0], lane), getattr(b[0], lane))
            np.testing.assert_array_equal(a[1].edge, b[1].edge)
    moved = dict(rows, q_stop=~rows["q_stop"] & (rows["role"] == 0))
    with pytest.raises(ValueError, match="other scope rows"):
        qbatch.ScopeContext(store, moved, xy, "FIT", saved=inputs["contexts"]["FIT"])


GRAPH = {"G": 1, "S": 1, "A": 1, "M": 1, "E": 1, "C": 1, "R": 1, "D": 1}


@pytest.mark.parametrize("family,card,factors", [
    ("relational_mpnn", {"hidden_width": 16, "weight_decay": 0.001}, GRAPH),
    ("team_set_pair_readout", {"hidden_width": 16, "weight_decay": 0.001}, dict(GRAPH, G=0, A=0, M=0)),
    ("tied_message", {"hidden_width": 16, "weight_decay": 0.001}, dict(GRAPH, A=0, M=0, S=0)),
    ("graphsage", {"hidden_width": 16, "weight_decay": 0.001}, dict(GRAPH, G=None)),
    ("shared_slot_mlp", {"hidden_width": 16, "weight_decay": 0.001}, dict(GRAPH, G=None)),
    ("pure_set", {"hidden_width": 16, "weight_decay": 0.001}, {"G": None, "S": 1, "A": 0, "M": 0, "E": 1, "C": 1, "R": 1, "D": 1}),
    ("plain_mlp", {"hidden": [16, 8], "weight_decay": 0.001}, dict(GRAPH, G=None)),
    ("lightgbm", {"num_leaves": 15, "min_data_in_leaf": 5}, dict(GRAPH, G=None)),
    ("logistic", {"C": 0.1}, dict(GRAPH, G=None)),
])
def test_each_family_runs_from_a_spec(run_dir, tmp_path, quick_rules, family, card, factors):
    root, rows, _, inputs = run_dir
    out = tmp_path / family
    result = q_worker.run(_spec(root, f"t/{family}", family, card, factors, inputs, out), out)
    assert result["status"] == "completed", result
    q_dev = np.flatnonzero(rows["role"] == 3)
    assert result["scores"]["q_dev"]["rows"] == len(q_dev)
    assert result["q_dev_keys_sha256"] == q_worker.keys_sha256(rows["engagement_key"][q_dev])
    assert np.array_equal(np.load(out / "pred_q_dev.npz")["row"], q_dev)
    if family == "team_set_pair_readout":
        assert result["structure"]["parameter_gap"] <= 0.05 and result["structure"]["g0_width"] % 8 == 0
    if family == "tied_message":
        assert result["structure"]["parameter_gap"] <= 0.05


def test_fit_time_limit_is_a_terminal_failure(run_dir, tmp_path, monkeypatch):
    root, _, _, inputs = run_dir
    monkeypatch.setattr(q_worker.qtrain, "neural_rules", lambda family, wd=None: qtrain.NeuralRules(batch_size=16, max_epochs=3, patience=3, time_limit_s=0.0))
    out = tmp_path / "limit"
    result = q_worker.run(_spec(root, "t/limit", "relational_mpnn", {"hidden_width": 16, "weight_decay": 0.001}, GRAPH, inputs, out), out)
    assert result["status"] == "failed" and result["reason"] == "fit_time_limit" and result["terminal"]
    state = json.loads((out / "FIT_STATE.json").read_text(encoding="utf-8"))
    assert state["time_limit"]["epoch"] == 1 and not state["done"]


def test_b_minus_map_reference_and_ft_forward(run_dir, tmp_path):
    root, rows, xy, inputs = run_dir
    out = tmp_path / "bmm"
    result = q_worker.run(_spec(root, "t/bmm", "lightgbm", {"num_leaves": 15, "min_data_in_leaf": 5}, {}, inputs, out, job_inputs="B_minus_map"), out)
    assert result["status"] == "completed"
    store = qf.RawStore(root / "raw" / "store")
    context = qbatch.ScopeContext(store, rows, xy, "REFIT_TRAIN", saved=inputs["contexts"]["REFIT_TRAIN"])
    cond = qbatch.Condition(G=None)
    build = q_worker.builder("ft_transformer", {"d_token": 8, "layers": 1, "heads": 2}, context, cond, {}, np.arange(N))
    model = build()
    logits = qbatch.forward_for("ft_transformer")(model, qbatch.assemble(context, np.arange(2), cond, "ft_transformer"), "cpu")
    assert logits.shape == (2,) and model.token_count > 1600


@pytest.mark.parametrize("family,card", [("constant", {}), ("time_only", {"C": 0.1}), ("p_only", {"C": 0.1}), ("PT_linear", {"C": 0.1}),
                                         ("PT_flex", {"p_knots": 4, "time_knots": 4, "degree": 3, "C": 0.1})])
def test_references(run_dir, tmp_path, family, card):
    root, rows, _, inputs = run_dir
    out = tmp_path / family
    out.mkdir()
    result = q_worker.run(_spec(root, f"t/{family}", family, card, {}, inputs, out, job_inputs="pt"), out)
    assert result["status"] == "completed", result
    if family == "constant":
        assert np.allclose(np.load(out / "pred_q_dev.npz")["p"], result["rate"])


def test_reference_allowlist(run_dir, tmp_path, monkeypatch):
    """time_only never reads pt_p_pre: corrupting it changes nothing."""

    root, rows, _, inputs = run_dir
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    q_worker.run_reference({"family": "time_only", "card": {"C": 0.1}}, rows, a)
    q_worker.run_reference({"family": "time_only", "card": {"C": 0.1}}, dict(rows, pt_p_pre=np.full(N, np.nan)), b)
    assert np.array_equal(np.load(a / "pred_q_dev.npz")["logit"], np.load(b / "pred_q_dev.npz")["logit"])


def test_cli_with_release_and_calibration(run_dir, tmp_path):
    root, rows, _, inputs = run_dir
    rel = release.issue(tmp_path / "release")
    payload = json.loads(rel.read_text(encoding="utf-8"))
    assert "studies/q_production_20261005/q_worker.py" in payload["evidence"]
    out = tmp_path / "jobs" / "p_only"
    out.mkdir(parents=True)
    spec = dict(_spec(root, "t/p_only", "p_only", {"C": 0.1}, {}, inputs, out, job_inputs="pt"), release=str(rel), release_digest=payload["digest"])
    (out / "SPEC.json").write_text(json.dumps(spec), encoding="utf-8")

    def cli(path):        # a fresh interpreter, as the supervisor runs it (the release check sees only the worker's imports)
        env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONIOENCODING": "utf-8"}
        return subprocess.run([sys.executable, "-B", str(Path(q_worker.__file__)), "--spec", str(path)], capture_output=True, text=True, env=env)

    assert cli(out / "SPEC.json").returncode == 0
    result = json.loads((out / "RESULT.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed" and result["release_digest"] == payload["digest"]
    cal = tmp_path / "jobs" / "calibrate"
    cal.mkdir()
    spec = {"kind": "calibrate", "out_dir": str(cal), "release": str(rel), "release_digest": payload["digest"],
            "targets": [{"job_id": "t/p_only", "result_dir": str(out), "rows": inputs["rows"], "rows_sha256": inputs["rows_sha256"]}]}
    (cal / "SPEC.json").write_text(json.dumps(spec), encoding="utf-8")
    assert cli(cal / "SPEC.json").returncode == 0
    record = json.loads((cal / "calibrators" / "t__p_only.json").read_text(encoding="utf-8"))
    assert record["calibrator"]["status"] in ("fitted", "failed") and record["calibrator"]["rows"] == int((rows["role"] == 4).sum())
    bad = dict(spec, release_digest="0" * 64, out_dir=str(tmp_path / "jobs" / "bad"))
    (cal / "BAD.json").write_text(json.dumps(bad), encoding="utf-8")
    refused = cli(cal / "BAD.json")
    assert refused.returncode != 0 and "another release" in refused.stderr


def test_identical_variant_aliases_the_main_inputs(tmp_path, monkeypatch):
    from studies.q_production_20261005 import q_prepare

    run = tmp_path / "run"
    (run / "inputs" / "main_h90").mkdir(parents=True)
    rows = _rows()
    qrows.assemble.write_npz(run / "Q_ROWS_h90.npz", rows)
    monkeypatch.setattr(q_prepare, "query_xy", lambda q: np.full((len(q), 10, 2), 1000.0))      # no player at (0, 0)
    source = q_prepare.ensure_rows(run, "exclude_zero_coordinate_h90")
    variant = json.loads((source.parent / "VARIANT.json").read_text(encoding="utf-8"))
    assert variant["identical_to_main"] and variant["counts"]["excluded"] == 0
    main = {"release_digest": "d", "rows_file": "Q_ROWS_h90.npz", "rows_sha256": "r", "xy_file": "inputs/main_h90/XY.npy", "contexts": {}}
    (run / "inputs" / "main_h90" / "PREPARED.json").write_text(json.dumps(main), encoding="utf-8")
    record = q_prepare._alias_main(run, "exclude_zero_coordinate_h90", source, {"digest": "d"})
    assert record["identical_to"] == "main_h90" and record["rows_file"] == "Q_ROWS_h90.npz" and record["xy_file"] == "inputs/main_h90/XY.npy"
    monkeypatch.setattr(q_prepare, "query_xy", lambda q: np.where(np.arange(len(q))[:, None, None] == 0, 0.0, np.full((len(q), 10, 2), 1000.0)))
    other = tmp_path / "run2"
    other.mkdir()
    qrows.assemble.write_npz(other / "Q_ROWS_h90.npz", rows)
    source = q_prepare.ensure_rows(other, "exclude_zero_coordinate_h90")
    assert not json.loads((source.parent / "VARIANT.json").read_text(encoding="utf-8"))["identical_to_main"]


def test_pre_q_probe_runs_every_family_on_saved_contexts(run_dir):
    import importlib.util

    root, rows, _, inputs = run_dir
    spec = importlib.util.spec_from_file_location("pre_q_gate", Path(__file__).resolve().parents[1] / "reports" / "q_production_20261005" / "pre_q_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    prepared = {"rows_file": "Q_ROWS_h90.npz", "xy_file": "inputs/main_h90/XY.npy",
                "contexts": {s: {"dir": f"inputs/main_h90/{s}"} for s in ("FIT", "REFIT_TRAIN")}}
    (root / "inputs" / "main_h90" / "PREPARED.json").write_text(json.dumps(prepared), encoding="utf-8")
    out = gate.probe(root, rows_per_family=16)
    assert set(out["families"]) == {"relational_mpnn", "team_set_pair_readout", "tied_message", "graphsage", "shared_slot_mlp", "pure_set",
                                    "plain_mlp", "lightgbm", "logistic"}
    assert out["families"]["lightgbm"]["table_columns"] > 504 and out["families"]["relational_mpnn"]["fit_epoch_minutes_estimate"] > 0


def test_derive_variants():
    rows = _rows()
    rows["role"][-6:] = 5
    rows["label_svi_delta"][-6:] = np.nan
    near, counts = qrows.derive_variant(rows, "exclude_near_zero_h90")
    kept = (np.abs(rows["label_svi_delta"]) >= 0.01) | (rows["role"] == 5)
    assert counts["rows"] == int(kept.sum()) and (near["role"] == 5).sum() == 6 and counts["test_filter"]
    assert np.allclose(near["weight_raw"], qrows.match_weights(near["match_index"]))
    zero = np.zeros(N, dtype=bool)
    zero[[1, 2]] = True
    zc, counts = qrows.derive_variant(rows, "exclude_zero_coordinate_h90", zero)
    assert counts["excluded"] == 2 and len(zc["role"]) == N - 2
