import json
import pickle
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from reports.v_production_20261004 import pipeline
from reports.v_production_20261004 import remote
from studies.v_production_20261004 import worker
from studies.v_production_20261004.train import FAMILIES, family_cards
from studies.v_training_gate_20261003.guard import require_training_ready


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _workspace(tmp_path, monkeypatch):
    """A fake workspace root holding the audit, contract and one scientific module."""

    root = tmp_path / "workspace"
    _write(root / worker.AUDIT_REL, json.dumps({"complete": True}))
    _write(root / worker.CONTRACT_REL, "{}")
    _write(root / "studies/runner.py", "verified source")
    evidence = {name: worker.sha256_file(root / name) for name in ("studies/runner.py", worker.CONTRACT_REL)}
    scientific_rel = "reports/v_production_20261004/SCIENTIFIC_test.json"
    _write(root / scientific_rel, json.dumps({"digest": worker.scientific_digest(evidence), "evidence": evidence}))
    monkeypatch.setattr(worker, "ROOT", root)
    monkeypatch.setattr(pipeline, "ROOT", root)
    return root, scientific_rel


def test_release_verifies_scheduler_and_scientific_evidence(tmp_path, monkeypatch):
    root, scientific_rel = _workspace(tmp_path, monkeypatch)
    scheduler = _write(root / "reports/pipeline.py", "scheduler")
    release = _write(tmp_path / "RELEASE.json", json.dumps({
        "scientific_runner_verified": True, "scientific_release": scientific_rel,
        "scientific_sha256": worker.sha256_file(root / scientific_rel),
        "scheduler_evidence": {"reports/pipeline.py": worker.sha256_file(scheduler)}}))
    assert pipeline.verify_release(release)["scientific_runner_verified"]
    scheduler.write_text("changed scheduler", encoding="utf-8")
    with pytest.raises(ValueError, match="stale release evidence"):
        pipeline.verify_release(release)
    scheduler.write_text("scheduler", encoding="utf-8")
    (root / "studies/runner.py").write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="stale scientific evidence"):
        pipeline.verify_release(release)


def test_scientific_digest_is_order_independent_and_content_bound():
    assert worker.scientific_digest({"a": "1", "b": "2"}) == worker.scientific_digest({"b": "2", "a": "1"})
    assert worker.scientific_digest({"a": "1"}) != worker.scientific_digest({"a": "2"})


def test_acceptance_is_portable_across_machines(tmp_path, monkeypatch):
    root, scientific_rel = _workspace(tmp_path, monkeypatch)
    run_root = tmp_path / "run"
    plan = {"fit_cache_name": "final_FIT_fit_eval", "refit_cache_name": "final_REFIT_TRAIN_refit_eval"}
    for name in plan.values():
        _write(run_root / "cache" / name / "MANIFEST.json", f"manifest {name}")
    for scope in worker.SCOPES:
        _write(run_root / "final" / f"CACHE_{scope}_INVENTORY.json", json.dumps({"hash_evidence": {}}))
    _write(run_root / "final/gates/FIT/logistic/CONSUMPTION_GATE.json", json.dumps({"status": "passed"}))
    digest = worker.write_acceptance(run_root, "final", plan, scientific_rel)
    acceptance = run_root / "final" / "TRAINING_ACCEPTANCE.json"
    keys = json.loads(acceptance.read_text(encoding="utf-8"))["hash_evidence"]
    assert all(not key.startswith(("/", "\\")) and ":" not in key for key in keys)
    monkeypatch.chdir(root)
    require_training_ready(acceptance)
    # Same files under a different run root and workspace (the remote layout).
    other = tmp_path / "other_machine"
    shutil.copytree(run_root, other / "run")
    shutil.copytree(root, other / "workspace")
    monkeypatch.chdir(other / "workspace")
    require_training_ready(other / "run" / "final" / "TRAINING_ACCEPTANCE.json")
    assert worker.sha256_file(other / "run" / "final" / "TRAINING_ACCEPTANCE.json") == digest
    assert worker.write_acceptance(run_root, "final", plan, scientific_rel) == digest  # deterministic


def test_model_dimensions_follow_separate_fit_and_refit_artifacts(monkeypatch):
    import studies.v_full_preprocess_20261003.artifacts as artifacts

    def load(path):
        count = 3 if path.name.endswith("_FIT") else 5
        return SimpleNamespace(categorical=(("item", SimpleNamespace(num_embeddings=count)),))

    monkeypatch.setattr(artifacts, "load_bundle", load)
    fit, refit = SimpleNamespace(table_width=12), SimpleNamespace(table_width=18)
    assert worker.dimensions(fit, "final", "FIT", "plain_mlp") == 12
    assert worker.dimensions(refit, "final", "REFIT_TRAIN", "plain_mlp") == 18
    assert worker.dimensions(fit, "final", "FIT", "team_set_mlp")["vocab_sizes"]["item"] == 3
    assert worker.dimensions(refit, "final", "REFIT_TRAIN", "team_set_mlp")["vocab_sizes"]["item"] == 5
    assert worker.dimensions(refit, "final", "REFIT_TRAIN", "logistic") is None


def _record(family, card_id, status="completed", reason=None):
    error = None if reason is None else {"reason": reason, "type": "LogisticNonconvergenceError"}
    return {"family": family, "card_id": card_id, "status": status, "error": error}


def _grid(overrides=None):
    overrides = overrides or {}
    rows = []
    for family in FAMILIES:
        for card in family_cards(family):
            rows.append(overrides.get((family, card["id"]), _record(family, card["id"])))
    return rows


def test_supervisor_lock_archives_dead_holder_and_rejects_live_one(tmp_path, monkeypatch):
    lock = tmp_path / "SUPERVISOR.lock"
    lock.write_text("31184", encoding="utf-8")
    monkeypatch.setattr(pipeline, "supervisor_alive", lambda pid, root, path: False)
    pipeline.acquire_lock(lock)
    assert lock.read_text(encoding="utf-8") == str(pipeline.os.getpid())
    assert len(list(tmp_path.glob("SUPERVISOR.lock.stale-31184-*"))) == 1
    monkeypatch.setattr(pipeline, "supervisor_alive", lambda pid, root, path: True)
    with pytest.raises(RuntimeError, match="held by live PID"):
        pipeline.acquire_lock(lock)


def test_supervisor_identity_rejects_reused_pid(tmp_path):
    lock = tmp_path / "SUPERVISOR.lock"
    lock.write_text(str(pipeline.os.getpid()), encoding="utf-8")
    assert not pipeline.supervisor_alive(pipeline.os.getpid(), tmp_path, lock)
    assert not pipeline.supervisor_alive(0, tmp_path, lock)


def test_stop_skips_a_lock_holder_that_is_not_this_runs_supervisor(tmp_path, monkeypatch, capsys):
    (tmp_path / "SUPERVISOR.lock").write_text(str(pipeline.os.getpid()), encoding="utf-8")
    monkeypatch.setattr(pipeline, "live_workers", lambda root: [])
    monkeypatch.setattr(pipeline, "live_helpers", lambda root: [])
    assert pipeline.stop(tmp_path, include_remote=False) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["stopped"] == [] and report["not_a_supervisor_of_this_run"] == [pipeline.os.getpid()]


def test_cache_state_rechecks_manifest_after_builder_scan(tmp_path, monkeypatch):
    path = pipeline.cache_dir(tmp_path, "oof3", "REFIT_TRAIN")
    path.mkdir(parents=True)

    def finished_during_scan(context, scope):
        (path / "MANIFEST.json").write_text("{}", encoding="utf-8")
        return False

    monkeypatch.setattr(pipeline, "_builder_running", finished_during_scan)
    assert pipeline.cache_state(tmp_path, "oof3", "REFIT_TRAIN") == "ready"
    (path / "MANIFEST.json").unlink()
    monkeypatch.setattr(pipeline, "_builder_running", lambda context, scope: False)
    assert pipeline.cache_state(tmp_path, "oof3", "REFIT_TRAIN") == "orphaned"
    assert pipeline.cache_state(tmp_path, "oof1", "FIT") == "missing"


def test_control_file_bounds_live_settings(tmp_path):
    defaults = dict(pipeline.CONTROL_DEFAULTS)
    assert pipeline.read_control(tmp_path, defaults) == defaults
    _write(tmp_path / "CONTROL.json", json.dumps({"gpu_slots": 6, "remote_slots": 2, "remote_contexts": ["final", "oof3"],
                                                  "cpu_budget": 99, "gate_slots": "x"}))
    control = pipeline.read_control(tmp_path, defaults)
    assert control["gpu_slots"] == 6 and control["remote_slots"] == 2 and control["remote_contexts"] == ["final", "oof3"]
    assert control["cpu_budget"] == defaults["cpu_budget"] and control["gate_slots"] == defaults["gate_slots"]
    # The protocol's 12 h per-fit abort limit can only be raised (author decision), never lowered.
    _write(tmp_path / "CONTROL.json", json.dumps({"fit_hours_limit": 6, "table_slots": 2}))
    control = pipeline.read_control(tmp_path, defaults)
    assert control["fit_hours_limit"] == pipeline.FIT_HOURS_LIMIT and control["table_slots"] == 2
    _write(tmp_path / "CONTROL.json", json.dumps({"fit_hours_limit": 24, "table_slots": 3}))
    control = pipeline.read_control(tmp_path, defaults)
    assert control["fit_hours_limit"] == 24.0 and control["table_slots"] == defaults["table_slots"]


def test_retry_classification_and_fit_phases():
    assert pipeline.is_retryable("RuntimeError: resource limit exceeded at start: available RAM=1")
    assert pipeline.is_retryable("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate")
    assert pipeline.is_retryable("", code=pipeline.EXIT_RSS_LIMIT)
    assert not pipeline.is_retryable("ValueError: refit consumed 3 rows; expected 4")
    assert [pipeline.fit_phase(stage) for stage in (None, "fit", "stop", "fit_solver", "refit", "refit_solver", "dev", "reload")] == \
        ["FIT", "FIT", "FIT", "FIT", "REFIT", "REFIT", "SCORE", "SCORE"]


def test_gates_are_ordered_context_major_and_candidates_longest_first(tmp_path):
    final, oof3 = pipeline.ContextPlan(tmp_path, "final"), pipeline.ContextPlan(tmp_path, "oof3")
    gate = lambda family: {"kind": "gate", "family": family}  # noqa: E731
    cand = lambda family: {"kind": "candidate", "family": family}  # noqa: E731
    rows = [(oof3, gate("team_set_mlp")), (final, gate("logistic")), (final, cand("plain_mlp")),
            (oof3, cand("team_set_mlp")), (final, cand("lightgbm")), (final, {"kind": "select"})]
    rows.sort(key=lambda row: pipeline.task_sort_key(*row))
    assert [(ctx.name, task["kind"], task.get("family")) for ctx, task in rows] == [
        ("final", "select", None), ("final", "gate", "logistic"), ("oof3", "gate", "team_set_mlp"),
        ("oof3", "candidate", "team_set_mlp"), ("final", "candidate", "lightgbm"), ("final", "candidate", "plain_mlp")]


def test_vlock_reuse_requires_binding_and_checkpoint_hash(tmp_path):
    checkpoint = _write(tmp_path / "model_refit.pkl", "model")
    lock = {"complete": True, "scientific_sha256": "s1", "training_acceptance_sha256": "a1",
            "selected": {"checkpoint": str(checkpoint)}, "checkpoint_sha256": worker.sha256_file(checkpoint)}
    _write(tmp_path / "V_LOCK.json", json.dumps(lock))
    assert pipeline.vlock_reusable(tmp_path, "s1", "a1")
    assert not pipeline.vlock_reusable(tmp_path, "s2", "a1")
    checkpoint.write_text("changed", encoding="utf-8")
    assert not pipeline.vlock_reusable(tmp_path, "s1", "a1")


def test_terminal_candidate_reuse_is_bound_to_scientific_release_and_acceptance(tmp_path):
    out = tmp_path / "candidate"
    _write(out / "candidate_result.json", json.dumps(_record("lightgbm", "lgbm_l31_m200")))
    _write(out / "CANDIDATE_BINDING.json", json.dumps({"scientific_sha256": "s1", "acceptance_sha256": "a1"}))
    assert pipeline.terminal_candidate(out, "s1", "a1")["status"] == "completed"
    assert pipeline.terminal_candidate(out, "s2", "a1") is None
    assert pipeline.terminal_candidate(out, "s1", "a2") is None
    _write(out / "candidate_result.json", json.dumps(_record("logistic", "lr_C0.001", "failed", "nonconverged")))
    assert pipeline.terminal_candidate(out, "s1", "a1")["status"] == "failed"
    _write(out / "candidate_result.json", json.dumps(_record("logistic", "lr_C0.001", "failed")))
    assert pipeline.terminal_candidate(out, "s1", "a1") is None


def test_remote_records_point_at_local_files(tmp_path):
    record = {"output_dir": r"D:\LOL_Worker_2070\run_3\final\candidates\team_set_mlp\v_set_h64_wd0.001",
              "checkpoint": r"D:\LOL_Worker_2070\run_3\final\candidates\team_set_mlp\v_set_h64_wd0.001\model_refit.pt"}
    local = worker.local_checkpoint(record, tmp_path / "v_set_h64_wd0.001")
    assert local["output_dir"] == str(tmp_path / "v_set_h64_wd0.001")
    assert local["checkpoint"] == str(tmp_path / "v_set_h64_wd0.001" / "model_refit.pt")


def test_candidate_grid_allows_only_logistic_nonconvergence():
    assert pipeline.check_candidate_grid(_grid()) == {"expected": 20, "completed": 20, "nonconverged": []}
    logistic = _grid({("logistic", "lr_C1"): _record("logistic", "lr_C1", "failed", "nonconverged")})
    assert pipeline.check_candidate_grid(logistic)["nonconverged"] == ["logistic/lr_C1"]
    with pytest.raises(ValueError, match="incomplete"):
        pipeline.check_candidate_grid(_grid()[:-1])
    with pytest.raises(ValueError, match="not terminal"):
        pipeline.check_candidate_grid(_grid({("lightgbm", "lgbm_l31_m200"): _record("lightgbm", "lgbm_l31_m200", "failed", "nonconverged")}))


def test_admission_respects_slots_cpu_ram_and_outstanding_reservations():
    gib = pipeline.GiB
    table = {"resource": "table", "cpu": 1, "ram": 26 * gib}
    neural = {"resource": "gpu", "cpu": 1, "ram": 8 * gib}
    common = {"gpu_slots": 2, "gate_slots": 2, "cpu_budget": 4, "last_start": 0.0, "now": 10_000.0}
    assert pipeline.admission(table, [], available_ram=30 * gib, **common) is None
    resident_table = [{"resource": "table", "cpu": 1, "ram": 26 * gib, "rss": 26 * gib}]
    assert pipeline.admission(table, resident_table, available_ram=60 * gib, **common) == "slot"
    allocating = [{"resource": "table", "cpu": 1, "ram": 26 * gib, "rss": 0}]
    assert pipeline.admission(neural, allocating, available_ram=30 * gib, **common) == "ram"
    assert pipeline.admission(neural, allocating, available_ram=40 * gib, **common) is None
    assert pipeline.admission(neural, resident_table, available_ram=10 * gib, **common) is None
    two_gpu = [{"resource": "gpu", "cpu": 1, "ram": 8 * gib, "rss": 8 * gib}] * 2
    assert pipeline.admission(neural, two_gpu, available_ram=60 * gib, **common) == "slot"
    busy = [{"resource": "gpu", "cpu": 3, "ram": 1, "rss": 1}]
    assert pipeline.admission({"resource": "table", "cpu": 3, "ram": 1}, busy, available_ram=60 * gib, **common) == "cpu"
    assert pipeline.admission({"resource": "table", "cpu": 3, "ram": 1}, [], available_ram=60 * gib, **{**common, "cpu_budget": 3}) is None
    assert pipeline.admission(neural, [], available_ram=60 * gib, **{**common, "last_start": 9_990.0}) == "settle"
    assert pipeline.admission({**neural, "not_before": 10_100.0}, [], available_ram=60 * gib, **common) == "cooldown"


def test_archive_moves_prior_output_under_stale_with_log(tmp_path):
    context = tmp_path / "final"
    target = context / "gates" / "FIT" / "logistic"
    _write(target / "CONSUMPTION_GATE.json", "{}")
    moved = pipeline.archive(target, context, "test")
    assert not target.exists() and (moved / "CONSUMPTION_GATE.json").is_file()
    assert moved.relative_to(context).parts[0] == "stale"
    assert json.loads((context / "STALE_LOG.jsonl").read_text(encoding="utf-8").splitlines()[0])["reason"] == "test"


def test_gate_reuse_requires_matching_binding(tmp_path):
    gate = tmp_path / "gate"
    _write(gate / "CONSUMPTION_GATE.json", json.dumps({"status": "passed", "families": [{"reload_equal": True}]}))
    assert not pipeline.gate_reusable(gate, "s1", "m1")
    digest = worker.sha256_file(gate / "CONSUMPTION_GATE.json")
    _write(gate / "GATE_BINDING.json", json.dumps({"scientific_sha256": "s1", "cache_manifest_sha256": "m1", "gate_sha256": digest}))
    assert pipeline.gate_reusable(gate, "s1", "m1")
    assert not pipeline.gate_reusable(gate, "s2", "m1")
    assert not pipeline.gate_reusable(gate, "s1", "m2")


def test_inventory_keys_are_run_root_relative(tmp_path):
    run_root = tmp_path / "run"
    cache_root = run_root / "cache" / "final_FIT_fit_eval"
    names = ("node.npy", "role.npy", "structured.npz", "csr.npz")
    for name in names:
        _write(cache_root / name, name)
    cache = SimpleNamespace(root=cache_root, rows=2, manifest={
        "numeric_files": {"node": names[0]}, "metadata_files": {"role": names[1]},
        "batches": [{"structured_file": names[2], "csr_file": names[3]}]})
    hashes = worker.cache_inventory(cache, run_root / "final" / "inventory.json", run_root)
    assert set(hashes) == {f"cache/final_FIT_fit_eval/{name}" for name in names}


def test_remote_launch_builds_an_escape_free_command(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(remote, "scp_to", lambda *args, **kwargs: None)

    def fake_json(code, timeout=180):
        captured.append(code)
        compile(code, "<remote>", "exec")
        return {"pid": 4242, "ok": True}

    monkeypatch.setattr(remote, "remote_json", fake_json)
    spec = _write(tmp_path / "spec.json", "{}")
    assert remote.launch(spec, f"{remote.REMOTE_RUN}/final/workers/spec.json", fresh_dir=f"{remote.REMOTE_RUN}/final/candidates/x/y") == 4242
    command_code = captured[-1]
    assert "v_production_20261004\\\\worker.py" in command_code and "\\\\run_3\\\\final" in command_code


class _FakeProcess:
    def __init__(self, code):
        self.code, self.pid = code, pipeline.os.getpid()

    def poll(self):
        return self.code

    def kill(self):
        pass

    def wait(self, timeout=None):
        return self.code


def _fake_supervisor(monkeypatch, tmp_path, *, fail_key=None, retry_key=None):
    launched = []
    monkeypatch.setattr(pipeline, "verify_release", lambda path: {"scientific_release": "reports/S.json", "scientific_sha256": "s1"})
    monkeypatch.setattr(pipeline, "cache_state", lambda root, context, scope: "ready")
    monkeypatch.setattr(pipeline, "_prepare_commands", lambda: [])
    monkeypatch.setattr(pipeline, "START_SETTLE_SECONDS", 0)
    monkeypatch.setattr(pipeline, "RETRY_COOLDOWN_SECONDS", 0)
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: None)

    def setup(ctx, scientific_sha):
        ctx.root.mkdir(parents=True, exist_ok=True)
        ctx.plan = {"context": ctx.name}
        ctx.tasks = {f"gate:FIT:{family}": {"kind": "gate", "state": "ready", "scope": "FIT", "family": family,
                                           "resource": "gate", "cpu": 1, "ram": 0} for family in ("logistic", "plain_mlp")}
        ctx.state = "gates"

    def candidates(ctx, scientific_sha, acceptance_sha, *_):
        for family, resource in (("lightgbm", "table"), ("team_set_mlp", "gpu"), ("plain_mlp", "gpu")):
            ctx.tasks[f"candidate:{family}:c0"] = {"kind": "candidate", "state": "ready", "family": family, "card_id": "c0",
                                                  "resource": resource, "cpu": 1, "ram": 0}
        ctx.state = "candidates"

    def launch(spec):
        key = pipeline.read_json(spec)["task_key"]
        launched.append(key)
        if key == retry_key and launched.count(key) == 1:
            spec.with_suffix(".log").write_text("torch.OutOfMemoryError: CUDA out of memory", encoding="utf-8")
            return _FakeProcess(1)
        return _FakeProcess(3 if key == fail_key else 0)

    monkeypatch.setattr(pipeline, "setup_context", setup)
    monkeypatch.setattr(pipeline, "write_acceptance", lambda *args: "acceptance")
    monkeypatch.setattr(pipeline, "plan_candidates", candidates)
    monkeypatch.setattr(pipeline, "launch_local", launch)
    release = _write(tmp_path / "RELEASE.json", "{}")
    return launched, release


def test_supervisor_runs_gates_candidates_then_select_for_each_context(tmp_path, monkeypatch):
    launched, release = _fake_supervisor(monkeypatch, tmp_path)
    code = pipeline.supervise(tmp_path / "run", release, ["final", "oof0"], gpu_slots=2, gate_slots=2, cpu_budget=8)
    assert code == 0
    per_context = ["gate:FIT:logistic", "gate:FIT:plain_mlp", "candidate:lightgbm:c0",
                   "candidate:team_set_mlp:c0", "candidate:plain_mlp:c0", "select"]
    assert sorted(launched) == sorted(per_context * 2)
    progress = json.loads((tmp_path / "run" / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "finished"
    assert all(item["state"] == "done" for item in progress["contexts"].values())
    assert not (tmp_path / "run" / "SUPERVISOR.lock").exists()


def test_supervisor_retries_a_cuda_memory_failure_once_then_completes(tmp_path, monkeypatch):
    launched, release = _fake_supervisor(monkeypatch, tmp_path, retry_key="candidate:team_set_mlp:c0")
    code = pipeline.supervise(tmp_path / "run", release, ["final"], gpu_slots=2, gate_slots=2, cpu_budget=8)
    assert code == 0
    assert launched.count("candidate:team_set_mlp:c0") == 2
    progress = json.loads((tmp_path / "run" / "progress.json").read_text(encoding="utf-8"))
    assert progress["contexts"]["final"]["retries"] == {"candidate:team_set_mlp:c0": 1}


def test_adopted_process_reads_the_outcome_from_worker_files(tmp_path):
    out = tmp_path / "candidate"
    gone = pipeline.AdoptedProcess(999_999_999, 0.0, "candidate", out, tmp_path)
    assert gone.poll() == 1  # vanished without a result
    _write(out / "candidate_result.json", json.dumps({"status": "completed"}))
    assert gone.poll() == 0
    _write(out / "candidate_result.json", json.dumps({"status": "failed", "error": {"reason": "nonconverged"}}))
    assert gone.poll() == 0
    _write(out / "candidate_result.json", json.dumps({"status": "failed", "error": {"type": "ValueError"}}))
    assert gone.poll() == 3
    import psutil

    me = psutil.Process()
    assert pipeline.AdoptedProcess(me.pid, me.create_time(), "gate", tmp_path, tmp_path).poll() is None
    assert pipeline.AdoptedProcess(me.pid, me.create_time() - 100, "gate", tmp_path, tmp_path).poll() == 1  # reused PID


def test_supervisor_adopts_a_live_worker_instead_of_relaunching_it(tmp_path, monkeypatch):
    launched, release = _fake_supervisor(monkeypatch, tmp_path)
    run_root = tmp_path / "run"
    key = "candidate:team_set_mlp:c0"
    spec = _write(run_root / "final" / "workers" / "candidate__team_set_mlp__c0__old.json",
                  json.dumps({"context": "final", "task_key": key, "kind": "candidate", "scientific_sha256": "s1"}))
    _write(run_root / "final" / "candidates" / "team_set_mlp" / "c0" / "candidate_result.json", json.dumps({"status": "completed"}))
    monkeypatch.setattr(pipeline, "live_worker_specs", lambda root: [(999_999_999, 0.0, spec)])
    monkeypatch.setattr(pipeline, "live_helpers", lambda root: [])

    def candidates(ctx, scientific_sha, acceptance_sha, *_):
        for family, resource in (("lightgbm", "table"), ("team_set_mlp", "gpu")):
            task_key = f"candidate:{family}:c0"
            ctx.tasks[task_key] = {"kind": "candidate", "state": "adopted" if task_key in ctx.adopted else "ready", "family": family,
                                   "card_id": "c0", "resource": resource, "cpu": 1, "ram": 0}
        ctx.state = "candidates"

    monkeypatch.setattr(pipeline, "plan_candidates", candidates)
    code = pipeline.supervise(run_root, release, ["final"], gpu_slots=2, gate_slots=2, cpu_budget=8)
    assert code == 0
    assert key not in launched and "candidate:lightgbm:c0" in launched
    adoption = next(run_root.glob("ADOPTION_*.json"))
    assert json.loads(adoption.read_text(encoding="utf-8"))["records"][0]["action"] == "adopted"


def test_supervisor_ends_workers_bound_to_another_scientific_release(tmp_path, monkeypatch):
    launched, release = _fake_supervisor(monkeypatch, tmp_path)
    run_root = tmp_path / "run"
    spec = _write(run_root / "final" / "workers" / "old.json",
                  json.dumps({"context": "final", "task_key": "candidate:team_set_mlp:c0", "kind": "candidate", "scientific_sha256": "old"}))
    monkeypatch.setattr(pipeline, "live_worker_specs", lambda root: [(999_999_999, 0.0, spec)])
    monkeypatch.setattr(pipeline, "live_helpers", lambda root: [])
    assert pipeline.supervise(run_root, release, ["final"], gpu_slots=2, gate_slots=2, cpu_budget=8) == 0
    assert "candidate:team_set_mlp:c0" in launched  # rerun under the current release
    assert json.loads(next(run_root.glob("ADOPTION_*.json")).read_text(encoding="utf-8"))["records"][0]["action"] == "ended"


def test_remote_transfer_failures_pause_then_resume_then_give_up(tmp_path):
    supervisor = pipeline.Supervisor(tmp_path, tmp_path / "RELEASE.json", ["final"], dict(pipeline.CONTROL_DEFAULTS))
    ctx = supervisor.plans[0]
    for attempt in range(1, pipeline.REMOTE_SYNC_RETRIES + 1):
        supervisor._remote_retry(ctx, "fit_synced", "scp: Connection closed", now=100.0)
        assert ctx.remote["state"] == "retry_wait" and ctx.remote["attempts"] == attempt
        assert ctx.remote["resume_state"] == "fit_synced" and ctx.remote["not_before"] == 100.0 + pipeline.RETRY_COOLDOWN_SECONDS
    supervisor._remote_retry(ctx, "fit_synced", "scp: Connection closed", now=100.0)
    assert ctx.remote["state"] == "failed"


def test_supervisor_fails_one_context_closed_and_still_finishes(tmp_path, monkeypatch):
    launched, release = _fake_supervisor(monkeypatch, tmp_path, fail_key="candidate:lightgbm:c0")
    code = pipeline.supervise(tmp_path / "run", release, ["final"], gpu_slots=2, gate_slots=2, cpu_budget=8)
    assert code == 1
    assert "select" not in launched
    failed = json.loads((tmp_path / "run" / "final" / "CONTEXT_FAILED.json").read_text(encoding="utf-8"))
    assert "candidate:lightgbm:c0" in failed["error"]


class _CalTableModel:
    def predict_logits(self, matrix):
        return np.linspace(-1.0, 1.0, matrix.shape[0])


def test_select_worker_locks_v_from_twenty_terminal_candidates(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    context_root = run_root / "final"
    acceptance = _write(context_root / "TRAINING_ACCEPTANCE.json", "{}")
    acceptance_sha = worker.sha256_file(acceptance)
    losses = {"lightgbm": 0.40, "logistic": None, "plain_mlp": 0.40005, "shared_slot_mlp": 0.41, "team_set_mlp": 0.42}
    for family in FAMILIES:
        for index, card in enumerate(family_cards(family)):
            out = context_root / "candidates" / family / card["id"]
            out.mkdir(parents=True)
            with (out / "model_refit.pkl").open("wb") as stream:
                pickle.dump(_CalTableModel(), stream)
            loss = losses[family]
            record = {"version": "test", "job_id": f"{family}:{card['id']}", "family": family, "card_id": card["id"],
                      "seed": 20261001, "context": "final", "output_dir": f"D:/remote/{family}/{card['id']}", "fit_rows": 1,
                      "stop_rows": 1, "refit_rows": 1, "dev_rows": 1, "chosen_epoch": None, "fit_metrics": {}, "stop_metrics": {},
                      "refit_metrics": {}, "reload_metrics": {}}
            if loss is None:
                record.update(status="failed", dev_metrics={}, checkpoint=None, error={"reason": "nonconverged"})
            else:
                record.update(status="completed", dev_metrics={"raw_logloss": loss + index * 1e-3, "consumed": {"keys_sha256": "k"}},
                              checkpoint=f"D:/remote/{family}/{card['id']}/model_refit.pkl", error=None)
            _write(out / "candidate_result.json", json.dumps(record))
            _write(out / "CANDIDATE_BINDING.json", json.dumps({"scientific_sha256": "s1", "acceptance_sha256": acceptance_sha,
                                                              "device": {"host": "test"}}))
    tracked = _write(run_root / "cache" / "x" / "tracked.bin", "cache shard")
    for scope in worker.SCOPES:
        _write(context_root / f"CACHE_{scope}_INVENTORY.json", json.dumps({"hash_evidence": {"cache/x/tracked.bin": worker.sha256_file(tracked)}}))
    rng = np.random.default_rng(0)
    labels = (rng.random(64) < 1.0 / (1.0 + np.exp(-2.0 * np.linspace(-1.0, 1.0, 64)))).astype(np.float32)

    class Source:
        excluded_game_keys = ()

        def iter_batches(self, split):
            assert split == "cal"
            yield {"inputs": {"table_csr": sparse.csr_matrix(np.zeros((64, 3), dtype=np.float32))}, "target": labels,
                   "metadata": {"roles": ("V_CAL",) * 64, "role": ("V_CAL",) * 64, "game_keys": tuple(f"g{i}" for i in range(64)),
                                "target_is_supervised": True}}

    monkeypatch.setattr(worker, "_open_source", lambda spec, family, scope=None: (Source(), ()))
    plan = {"counts": {"fit": 1, "stop": 1, "refit": 1, "dev": 1, "cal": 64},
            "dimensions": {f"{scope}:{family}": None for scope in worker.SCOPES for family in FAMILIES}}
    spec = {"kind": "select", "context": "final", "run_root": str(run_root), "scientific_sha256": "s1", "plan": plan}
    assert worker.run_select(spec) == 0
    lock = json.loads((context_root / "V_LOCK.json").read_text(encoding="utf-8"))
    # LightGBM .40000 vs plain .40005 is a tie at 1e-4; the family order prefers LightGBM.
    assert lock["selected"]["family"] == "lightgbm" and lock["selected"]["card_id"] == family_cards("lightgbm")[0]["id"]
    assert lock["selected"]["checkpoint"] == str(context_root / "candidates" / "lightgbm" / family_cards("lightgbm")[0]["id"] / "model_refit.pkl")
    assert lock["grid"]["nonconverged"] == sorted(f"logistic/{card['id']}" for card in family_cards("logistic"))
    assert lock["tie_tolerance"] == 0.0001 and lock["complete"] is True and lock["scientific_sha256"] == "s1"
    tracked.write_text("changed shard", encoding="utf-8")
    (context_root / "V_LOCK.json").unlink()
    with pytest.raises(RuntimeError, match="changed during training"):
        worker.run_select(spec)
