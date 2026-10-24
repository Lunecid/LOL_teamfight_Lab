from __future__ import annotations

import os
import json
import pickle

import numpy as np
import pytest
from scipy import sparse

from studies.v_model_fit_input_20261003.models import build_model
from studies.v_production_20261004 import train


def _table_batches(rows: int, *, role: str, swapped: bool = False):
    x = sparse.csr_matrix(np.arange(rows * 3, dtype=np.float32).reshape(rows, 3) / 10.0)
    y = np.arange(rows, dtype=np.int64) % 2
    return [{
        "inputs": {"table_csr": x},
        "W": y,
        "metadata": {"role": role, "game_key": np.arange(rows), "swapped": swapped, "nominal_index": np.arange(rows)},
    }]


def _dense_batches(rows: int, *, role: str, swapped: bool = False, width: int = 4):
    x = np.arange(rows * width, dtype=np.float32).reshape(rows, width) / 10.0
    y = np.arange(rows, dtype=np.int64) % 2
    return [{
        "inputs": {"dense_features": x},
        "W": y,
        "metadata": {"role": role, "game_key": np.arange(rows), "swapped": swapped, "nominal_index": np.arange(rows)},
    }]


class _FakeTableModel:
    def __init__(self, offset: float = 0.0):
        self.offset = float(offset)

    def fit(self, matrix, target):
        return self

    def predict_logits(self, matrix):
        return np.arange(matrix.shape[0], dtype=np.float64) + self.offset


def _result(family: str, card_id: str, loss: float) -> train.CandidateResult:
    return train.CandidateResult(
        job_id=f"{family}:{card_id}", family=family, card_id=card_id,
        seed=20261001, context="final", status="completed", output_dir=".",
        fit_rows=1, stop_rows=1, refit_rows=1, dev_rows=1,
        chosen_epoch=None, fit_metrics={}, stop_metrics={}, refit_metrics={},
        dev_metrics={"raw_logloss": loss}, reload_metrics={}, checkpoint=None,
    )


def test_target_flip_and_positive_slope_calibrator():
    _, labels, metadata = train._normalise_batch({
        "inputs": {"dense_features": np.zeros((2, 4), dtype=np.float32)},
        "W": np.array([0, 1]),
        "metadata": {"role": "TRAIN", "game_key": ["g0", "g1"], "swapped": [True, False]},
    })
    assert labels.tolist() == [1.0, 1.0]
    assert metadata["target_is_supervised"] is True

    calibrator = train.fit_positive_slope_calibrator(np.array([0.2, 0.8, 0.3, 0.7]), np.array([0, 1, 0, 1]))
    assert calibrator.a > 0
    assert np.isfinite(calibrator.predict([0.25, 0.75])).all()


def test_plain_mlp_job_writes_stop_refit_reload_evidence(tmp_path, monkeypatch):
    def builder(family, card_id, **kwargs):
        assert family == "plain_mlp"
        return build_model(family, card=card_id, input_dim=4)

    monkeypatch.setattr(train, "build_locked_model", builder)
    source = lambda split, **kwargs: _dense_batches(4, role="TRAIN" if split in {"fit", "stop", "refit"} else "V_DEV")
    job = train.CandidateJob(
        family="plain_mlp",
        card_id="plain_128_64_wd0.001",
        seed=20261001,
        acceptance_path=tmp_path / "acceptance.json",
        output_dir=tmp_path / "candidate",
        fit_source=source,
        stop_source=source,
        refit_source=source,
        dev_source=source,
        dimensions=4,
        config=train.RunnerConfig(max_epochs=1),
        expected_fit_rows=4,
        expected_stop_rows=4,
        expected_refit_rows=4,
        expected_dev_rows=4,
    )
    result = train.run_candidate_job(job)
    assert result.status == "completed"
    assert result.chosen_epoch == 1
    assert result.fit_rows == 4
    assert result.stop_rows == 4
    assert result.refit_rows == 4
    assert result.reload_metrics["prediction_equal"] is True
    assert (tmp_path / "candidate" / "checkpoint_best.pt").is_file()
    assert json.loads((tmp_path / "candidate" / "progress.json").read_text())[
        "status"
    ] == "completed"


def test_logistic_job_is_fresh_refit_and_rejects_test_rows(tmp_path, monkeypatch):
    built_dimensions = []

    def builder(family, card_id, **kwargs):
        built_dimensions.append(kwargs.get("dimensions"))
        return build_model(family, card=card_id, max_iter=50)

    monkeypatch.setattr(train, "build_locked_model", builder)
    source = lambda split, **kwargs: _table_batches(6, role="TRAIN" if split == "fit" else "V_DEV")
    job = train.CandidateJob(
        family="logistic",
        card_id="lr_C0.001",
        seed=20261001,
        acceptance_path=tmp_path / "acceptance.json",
        output_dir=tmp_path / "candidate",
        fit_source=None,
        refit_source=lambda split, **kwargs: _table_batches(6, role="TRAIN"),
        dev_source=source,
        refit_dimensions=3,
        expected_refit_rows=6,
        expected_dev_rows=6,
    )
    result = train.run_candidate_job(job)
    assert result.status == "completed"
    assert result.chosen_epoch is None
    assert result.fit_rows == 0
    assert built_dimensions == [3]
    assert (tmp_path / "candidate" / "model_refit.pkl").is_file()

    bad = lambda split, **kwargs: _table_batches(6, role="TEST")
    rejected = train.CandidateJob(
        family="logistic",
        card_id="lr_C0.001",
        seed=20261001,
        acceptance_path=tmp_path / "acceptance.json",
        output_dir=tmp_path / "rejected",
        fit_source=bad,
        refit_source=bad,
        dev_source=source,
    )
    try:
        train.run_candidate_job(rejected)
    except ValueError as exc:
        assert "refit source" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("TEST rows must never enter fitting")


def test_v_dev_selection_uses_exact_family_winner_then_canonical_card_order():
    plain_cards = [card["id"] for card in train.family_cards("plain_mlp")]
    exact = train.select_v_dev([
        _result("plain_mlp", plain_cards[0], 0.4000),
        _result("plain_mlp", plain_cards[1], 0.3999),
    ])
    assert exact.card_id == plain_cards[1]

    canonical_tie = train.select_v_dev([
        _result("plain_mlp", plain_cards[0], 0.4000),
        _result("plain_mlp", plain_cards[1], 0.4000),
    ])
    assert canonical_tie.card_id == plain_cards[0]

    logistic_card = train.family_cards("logistic")[0]["id"]
    family_tie = train.select_v_dev([
        _result("plain_mlp", plain_cards[1], 0.4000),
        _result("logistic", logistic_card, 0.40008),
    ])
    assert family_tie.family == "logistic"

    # The V family tolerance is .0001 (q uses .0005); a .0004 gap is not a tie.
    outside_tie = train.select_v_dev([
        _result("plain_mlp", plain_cards[1], 0.4000),
        _result("logistic", logistic_card, 0.4004),
    ])
    assert outside_tie.family == "plain_mlp"
    assert train.V_FAMILY_TIE_TOLERANCE == 0.0001


def test_runtime_enables_deterministic_algorithms():
    import torch

    train.configure_runtime(20261001, "cpu")
    assert torch.are_deterministic_algorithms_enabled()
    assert train.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_table_refit_rejects_truncated_source_before_solver(tmp_path):
    class Model:
        fit_calls = 0

        def fit(self, matrix, target):
            Model.fit_calls += 1
            return self

    progress = train._Progress(tmp_path / "progress.json", job_id="refit-truncated", total_rows=4)
    with pytest.raises(ValueError, match="refit consumed 3 rows; expected 4"):
        train._fit_refit_table(
            Model(), _table_batches(3, role="TRAIN"), family="logistic", chosen_epoch=None,
            config=train.RunnerConfig(), progress=progress, expected_rows=4,
            excluded_game_keys=(), output_dir=tmp_path,
        )
    assert Model.fit_calls == 0


def test_oof_and_ledger_provenance_are_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="excluded_game_keys"):
        train.CandidateJob(
            family="logistic", card_id="lr_C0.001", seed=20261001,
            acceptance_path=tmp_path / "acceptance.json", output_dir=tmp_path / "oof",
            refit_source=None, context="oof0", fold=0,
        )

    _, labels, metadata = train._normalise_batch({
        "inputs": {"table_csr": sparse.csr_matrix(np.zeros((2, 3), dtype=np.float32))},
        "W": np.array([0, 1]), "metadata": {"role": "TRAIN"},
    })
    ledger = train._Ledger(expected_rows=2, excluded_game_keys=(), split="fit")
    with pytest.raises(ValueError, match="game_keys"):
        ledger.update(metadata, labels)


def test_lightgbm_truncated_stop_fails_before_estimator_fit(tmp_path):
    class Estimator:
        def __init__(self):
            self.fit_calls = 0

        def fit(self, *args, **kwargs):
            self.fit_calls += 1
            self.best_iteration_ = 1

    class Model:
        def __init__(self):
            self.estimator = Estimator()

    model = Model()
    progress = train._Progress(tmp_path / "progress.json", job_id="lgb-stop", total_rows=4)
    with pytest.raises(ValueError, match="stop consumed"):
        train._fit_table(
            model, "lightgbm", _table_batches(4, role="TRAIN"), _table_batches(3, role="TRAIN"),
            config=train.RunnerConfig(), progress=progress, output_dir=tmp_path,
            expected_fit_rows=4, expected_stop_rows=4, excluded_game_keys=(),
        )
    assert model.estimator.fit_calls == 0


def test_reload_mismatch_and_truncated_source_raise(tmp_path):
    job = train.CandidateJob(
        family="logistic", card_id="lr_C0.001", seed=20261001,
        acceptance_path=tmp_path / "acceptance.json", output_dir=tmp_path / "reload",
        refit_source=None, dev_source=None,
    )
    mismatch = tmp_path / "mismatch"
    mismatch.mkdir()
    with (mismatch / "model_refit.pkl").open("wb") as stream:
        pickle.dump(_FakeTableModel(offset=1.0), stream)
    with pytest.raises(RuntimeError, match="reload changed predictions"):
        train._independent_reload(
            job, _FakeTableModel(offset=0.0), source=_table_batches(4, role="V_DEV"),
            output_dir=mismatch, expected_rows=4,
            progress=train._Progress(mismatch / "progress.json", job_id="mismatch", total_rows=4),
        )

    truncated = tmp_path / "truncated"
    truncated.mkdir()
    with (truncated / "model_refit.pkl").open("wb") as stream:
        pickle.dump(_FakeTableModel(offset=0.0), stream)
    with pytest.raises(ValueError, match="reload consumed"):
        train._independent_reload(
            job, _FakeTableModel(offset=0.0), source=_table_batches(3, role="V_DEV"),
            output_dir=truncated, expected_rows=4,
            progress=train._Progress(truncated / "progress.json", job_id="truncated", total_rows=4),
        )


def test_changed_reload_labels_fail_candidate_before_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "build_locked_model", lambda *args, **kwargs: _FakeTableModel())

    def dev_source(split, **kwargs):
        values = np.array([0, 1, 0, 1] if split == "dev" else [1, 1, 0, 1])
        batch = _table_batches(4, role="V_DEV")[0]
        batch = dict(batch)
        batch["W"] = values
        return [batch]

    job = train.CandidateJob(
        family="logistic", card_id="lr_C0.001", seed=20261001,
        acceptance_path=tmp_path / "acceptance.json", output_dir=tmp_path / "changed",
        refit_source=lambda split, **kwargs: _table_batches(4, role="TRAIN"),
        dev_source=dev_source, expected_refit_rows=4, expected_dev_rows=4,
    )
    with pytest.raises(ValueError, match="reload (consumption|logloss) differs"):
        train.run_candidate_job(job)
    record = json.loads((tmp_path / "changed" / "candidate_result.json").read_text())
    assert record["status"] == "failed"


def test_neural_refit_uses_its_own_dimensions(tmp_path, monkeypatch):
    built_dimensions = []

    def builder(family, card_id, **kwargs):
        dimensions = kwargs["dimensions"]
        built_dimensions.append(dimensions)
        return build_model(family, card=card_id, input_dim=int(dimensions))

    monkeypatch.setattr(train, "build_locked_model", builder)

    def source(split, **kwargs):
        role = "V_DEV" if split in {"dev", "reload"} else "TRAIN"
        width = 4 if split in {"refit", "dev", "reload"} else 3
        return _dense_batches(4, role=role, width=width)

    job = train.CandidateJob(
        family="plain_mlp", card_id="plain_128_64_wd0.001", seed=20261001,
        acceptance_path=tmp_path / "acceptance.json", output_dir=tmp_path / "widths",
        fit_source=source, stop_source=source, refit_source=source, dev_source=source,
        dimensions=3, refit_dimensions=4, config=train.RunnerConfig(max_epochs=1),
        expected_fit_rows=4, expected_stop_rows=4, expected_refit_rows=4, expected_dev_rows=4,
    )
    result = train.run_candidate_job(job)
    assert result.status == "completed"
    assert built_dimensions == [3, 4, 4]


def test_gate_replaces_a_nonconverging_logistic_probe_and_records_it(tmp_path, monkeypatch):
    from studies.v_model_fit_input_20261003.models import LogisticAdapter, LogisticNonconvergenceError

    requested = train.family_cards("logistic")[2]
    original = LogisticAdapter.fit

    def fit(self, *args, **kwargs):  # the requested card cannot converge on this batch
        if self.C == float(requested["C"]):
            raise LogisticNonconvergenceError({"success": False, "nit": 4000})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LogisticAdapter, "fit", fit)
    rng = np.random.default_rng(3)
    x = sparse.csr_matrix(rng.normal(size=(1024, 5)).astype(np.float32))
    y = (np.arange(1024) % 2).astype(np.int64)
    batch = {"inputs": {"table_csr": x}, "W": y,
             "metadata": {"role": "TRAIN", "game_key": [f"g{i}" for i in range(1024)], "nominal_index": np.arange(1024)}}
    result = train.run_full_input_consumption_gate(
        lambda split, **kwargs: [batch], output_dir=tmp_path / "gate", dimensions=None, expected_rows=1024,
        families=("logistic",), card_id=2,
        resource_limits={"rss_bytes": 64 << 30, "free_disk_bytes": 0, "available_ram_bytes": 0})
    row = result["families"][0]
    assert result["status"] == "passed" and row["reload_equal"] is True
    assert row["requested_card_id"] == requested["id"] and row["card_id"] != requested["id"]
    assert row["probe_nonconverged"][0]["card_id"] == requested["id"]


def test_full_input_gate_writes_probe_only_progress(tmp_path):
    source = lambda split, **kwargs: _dense_batches(1024, role="TRAIN")
    result = train.run_full_input_consumption_gate(
        source,
        output_dir=tmp_path / "gate",
        dimensions=4,
        expected_rows=1024,
        families=("plain_mlp",),
        resource_limits={"rss_bytes": 64 << 30, "free_disk_bytes": 0, "available_ram_bytes": 0},
    )
    assert result["probe_only"] is True
    assert result["training_ready"] is False
    progress = json.loads((tmp_path / "gate" / "progress.json").read_text())
    assert progress["status"] == "completed"
    assert progress["overall_rows"] == 1024
    assert "consumption:plain_mlp:" in progress["stage"] or progress["stage"] == "done"
