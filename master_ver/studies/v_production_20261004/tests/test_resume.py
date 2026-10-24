"""A candidate interrupted at any point resumes to the uninterrupted result exactly."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import torch
from scipy import sparse

from studies.v_model_fit_input_20261003.models import build_model
from studies.v_production_20261004 import train, worker

ROWS, WIDTH, BATCH = 96, 6, 32
_RNG = np.random.default_rng(7)
_X = {split: _RNG.normal(size=(ROWS, WIDTH)).astype(np.float32) for split in ("TRAIN", "V_DEV")}
_Y = {split: (_X[split][:, 0] + 0.5 * _RNG.normal(size=ROWS) > 0).astype(np.int64) for split in ("TRAIN", "V_DEV")}


class Interrupt(RuntimeError):
    """Stands in for a power loss: the process dies mid-phase."""


def _metadata(role: str, start: int, stop: int) -> dict:
    return {"role": role, "game_key": [f"g{i}" for i in range(start, stop)], "swapped": False, "nominal_index": np.arange(start, stop)}


def _source(fail_at: tuple[str, int] | None = None):
    """Replayable dense source; ``fail_at=(split, k)`` dies inside the k-th pass over ``split``."""

    calls: dict[str, int] = {}

    def source(split, epoch=0, batch_size=1024, **_):
        occurrence = calls.get(split, 0)
        calls[split] = occurrence + 1
        role = "V_DEV" if split in {"dev", "reload"} else "TRAIN"

        def batches():
            for index, start in enumerate(range(0, ROWS, BATCH)):
                if fail_at == (split, occurrence) and index == 1:
                    raise Interrupt(f"interrupted in pass {occurrence} of {split}")
                rows = slice(start, start + BATCH)
                yield {"inputs": {"dense_features": _X[role][rows]}, "W": _Y[role][rows], "metadata": _metadata(role, start, start + BATCH)}
        return batches()
    return source


def _job(root, name: str, source, *, max_epochs: int = 4) -> train.CandidateJob:
    return train.CandidateJob(
        family="plain_mlp", card_id="plain_128_64_wd0.001", seed=20261001,
        acceptance_path=root / "acceptance.json", output_dir=root / name,
        fit_source=source, stop_source=source, refit_source=source, dev_source=source,
        dimensions=WIDTH, refit_dimensions=WIDTH, config=train.RunnerConfig(max_epochs=max_epochs),
        expected_fit_rows=ROWS, expected_stop_rows=ROWS, expected_refit_rows=ROWS, expected_dev_rows=ROWS)


def _builder(family, card_id, **kwargs):
    torch.manual_seed(int(kwargs["seed"]))  # as guard.build_locked_model does
    return build_model(family, card=card_id, input_dim=int(kwargs["dimensions"]))


@pytest.fixture(scope="module", autouse=True)
def _locked_builder():
    patch = pytest.MonkeyPatch()
    patch.setattr(train, "build_locked_model", _builder)
    yield
    patch.undo()


@pytest.fixture(scope="module")
def reference(tmp_path_factory, _locked_builder):
    root = tmp_path_factory.mktemp("reference")
    result = train.run_candidate_job(_job(root, "straight", _source()))
    assert result.status == "completed" and result.chosen_epoch >= 2
    return result, _weights(root / "straight" / "model_refit.pt")


def _after_crash(output):
    """What the worker does before a relaunch: keep the directory, set the failed attempt aside."""

    (output / "attempts").mkdir(exist_ok=True)
    shutil.move(str(output / "candidate_result.json"), str(output / "attempts" / f"candidate_result_{len(list((output / 'attempts').iterdir()))}.json"))


def _scientific(result: train.CandidateResult) -> dict:
    payload = result.to_dict()
    for name in ("output_dir", "checkpoint"):
        payload.pop(name)
    return json.loads(json.dumps(payload, sort_keys=True))


def _weights(path):
    return torch.load(path, map_location="cpu", weights_only=False)["model_state"]


def _assert_same(result, output, reference):
    expected, weights = reference
    assert _scientific(result) == _scientific(expected)
    restored = _weights(output / "model_refit.pt")
    assert restored.keys() == weights.keys()
    assert all(torch.equal(restored[key], weights[key]) for key in weights)


@pytest.mark.parametrize("fail_at", [("fit", 2), ("stop", 1), ("refit", 1), ("dev", 0)])
def test_interrupted_candidate_resumes_bitwise(tmp_path, reference, fail_at):
    with pytest.raises(Interrupt):
        train.run_candidate_job(_job(tmp_path, "cut", _source(fail_at)))
    output = tmp_path / "cut"
    _after_crash(output)
    result = train.run_candidate_job(_job(tmp_path, "cut", _source()))
    _assert_same(result, output, reference)
    events = [json.loads(line) for line in (output / "RESUME_LOG.jsonl").read_text(encoding="utf-8").splitlines()]
    resumed = {(row.get("phase"), row.get("resumed_after_epoch")) for row in events if "resumed_after_epoch" in row}
    skipped = {row.get("skipped") for row in events if "skipped" in row}
    if fail_at[0] == "fit":
        assert ("fit", 2) in resumed
    if fail_at[0] == "stop":
        assert ("fit", 1) in resumed
    if fail_at[0] == "refit":
        assert "FIT_DONE.json" in skipped and ("refit", 1) in resumed
    if fail_at[0] == "dev":  # every REFIT epoch was done: restored from the last one, then scored and reload-checked
        assert "FIT_DONE.json" in skipped and ("refit", reference[0].chosen_epoch) in resumed


def test_twice_interrupted_candidate_still_matches(tmp_path, reference):
    output = tmp_path / "twice"
    for fail_at in (("fit", 1), ("fit", 1)):  # the second run dies in its own second FIT pass (epoch 3)
        with pytest.raises(Interrupt):
            train.run_candidate_job(_job(tmp_path, "twice", _source(fail_at)))
        _after_crash(output)
    result = train.run_candidate_job(_job(tmp_path, "twice", _source()))
    _assert_same(result, output, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the CUDA resume check runs where training runs")
def test_cuda_candidate_interrupted_in_fit_and_refit_resumes_bitwise(tmp_path):
    import dataclasses

    def job(name, source):
        return dataclasses.replace(_job(tmp_path, name, source), config=train.RunnerConfig(max_epochs=4, device="cuda"))

    straight = train.run_candidate_job(job("straight", _source()))
    output = tmp_path / "cut"
    for fail_at in (("fit", 2), ("refit", 1)):  # dies in FIT epoch 3, then again in REFIT epoch 2
        with pytest.raises(Interrupt):
            train.run_candidate_job(job("cut", _source(fail_at)))
        _after_crash(output)
    result = train.run_candidate_job(job("cut", _source()))
    _assert_same(result, output, (straight, _weights(tmp_path / "straight" / "model_refit.pt")))


def test_torn_last_checkpoint_falls_back_one_epoch(tmp_path, reference):
    with pytest.raises(Interrupt):
        train.run_candidate_job(_job(tmp_path, "torn", _source(("fit", 3))))
    output = tmp_path / "torn"
    _after_crash(output)
    (output / "checkpoint_epoch_03.pt").write_bytes(b"\0" * 64)  # what a power loss can leave
    result = train.run_candidate_job(_job(tmp_path, "torn", _source()))
    _assert_same(result, output, reference)
    assert '"fallback_from_epoch": 3' in (output / "RESUME_LOG.jsonl").read_text(encoding="utf-8")


def test_resume_files_of_another_configuration_are_ignored(tmp_path):
    with pytest.raises(Interrupt):
        train.run_candidate_job(_job(tmp_path, "other", _source(("fit", 2)), max_epochs=4))
    output = tmp_path / "other"
    _after_crash(output)
    result = train.run_candidate_job(_job(tmp_path, "other", _source(), max_epochs=3))
    assert result.status == "completed"
    assert len(result.fit_metrics["epochs"]) == 3
    assert "identity or epoch target differs" in (output / "RESUME_LOG.jsonl").read_text(encoding="utf-8")


def _table_job(tmp_path, monkeypatch, name):
    def builder(family, card_id, **kwargs):
        return build_model(family, card=card_id, max_iter=200)

    monkeypatch.setattr(train, "build_locked_model", builder)
    cls = type(builder("logistic", "lr_C0.001"))
    calls = []
    original = cls.fit

    def counting_fit(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, "fit", counting_fit)

    def source(fail_split=None):
        def table(split, **_):
            role = "V_DEV" if split in {"dev", "reload"} else "TRAIN"
            if split == fail_split:
                raise Interrupt(f"interrupted in {split}")
            return [{"inputs": {"table_csr": sparse.csr_matrix(_X[role])}, "W": _Y[role], "metadata": _metadata(role, 0, ROWS)}]
        return table

    def job(src):
        return train.CandidateJob(family="logistic", card_id="lr_C0.001", seed=20261001,
                                  acceptance_path=tmp_path / "acceptance.json", output_dir=tmp_path / name,
                                  refit_source=src, dev_source=src, refit_dimensions=WIDTH,
                                  expected_refit_rows=ROWS, expected_dev_rows=ROWS)
    return job, source, calls


def test_table_refit_is_reused_only_after_scoring_and_reload_passed(tmp_path, monkeypatch):
    job, source, calls = _table_job(tmp_path, monkeypatch, "lr")
    straight = train.run_candidate_job(job(source()))
    (tmp_path / "lr" / "candidate_result.json").unlink()  # power loss just before the result record
    (tmp_path / "lr" / "progress.json").unlink()
    assert (tmp_path / "lr" / "REFIT_SCORED.json").is_file() and len(calls) == 1
    resumed = train.run_candidate_job(job(source()))
    assert len(calls) == 1  # the solver did not run again
    assert _scientific(resumed) == _scientific(straight)


def test_table_refit_interrupted_before_its_reload_check_is_recomputed(tmp_path, monkeypatch):
    job, source, calls = _table_job(tmp_path, monkeypatch, "lr2")
    with pytest.raises(Interrupt):
        train.run_candidate_job(job(source("reload")))
    assert not (tmp_path / "lr2" / "REFIT_SCORED.json").exists()
    _after_crash(tmp_path / "lr2")
    result = train.run_candidate_job(job(source()))
    assert result.status == "completed" and len(calls) == 2  # the reload check needs the in-memory model
    assert result.reload_metrics["prediction_equal"] is True


def _binding(**device):
    return {"scientific_sha256": "s" * 64, "acceptance_sha256": "a" * 64,
            "device": {"host": "main", "gpu": "G", "torch": "2", "cuda": "12", **device}}


def test_worker_keeps_only_a_directory_bound_to_this_release_and_device(tmp_path):
    context = tmp_path / "final"
    output = context / "candidates" / "plain_mlp" / "card"
    output.mkdir(parents=True)
    (output / "checkpoint_epoch_01.pt").write_bytes(b"x")
    (output / "CANDIDATE_BINDING.json").write_text(json.dumps({**_binding(), "worker_pid": 1, "spec": "s", "started_utc": "t"}))
    (output / "candidate_result.json").write_text(json.dumps({"status": "failed", "error": {"type": "OutOfMemoryError", "message": "CUDA out of memory"}}))
    state = worker.prepare_candidate_dir(output, context, _binding())
    assert state["resumed"] is True and state["launches"][0]["worker_pid"] == 1
    assert not (output / "candidate_result.json").exists() and len(list((output / "attempts").iterdir())) == 1

    (output / "candidate_result.json").write_text(json.dumps({"status": "completed"}))
    assert "terminal" in worker.prepare_candidate_dir(output, context, _binding())

    state = worker.prepare_candidate_dir(output, context, _binding(gpu="other"))
    assert state == {"resumed": False, "launches": []} and not output.exists()
    assert "not bound to this release/acceptance/device" in (context / "STALE_LOG.jsonl").read_text(encoding="utf-8")


def test_worker_never_continues_a_scientific_failure(tmp_path):
    context = tmp_path / "final"
    output = context / "candidates" / "plain_mlp" / "card"
    output.mkdir(parents=True)
    (output / "CANDIDATE_BINDING.json").write_text(json.dumps(_binding()))
    (output / "candidate_result.json").write_text(json.dumps({"status": "failed", "error": {
        "type": "RuntimeError", "message": "checkpoint reload changed predictions (max_abs_logit_diff=1e-3)"}}))
    assert worker.prepare_candidate_dir(output, context, _binding()) == {"resumed": False, "launches": []}
    assert not output.exists() and "failed scientifically" in (context / "STALE_LOG.jsonl").read_text(encoding="utf-8")
