"""Scheduler side of power-loss recovery: kept directories, cache rollback, fetch hints, time limit."""
import json
from pathlib import Path

import pytest

from reports.v_production_20261004 import pipeline, resume_run
from studies.v_production_20261004.train import family_cards


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _binding(scientific="s1", acceptance="a1"):
    return json.dumps({"scientific_sha256": scientific, "acceptance_sha256": acceptance, "device": {"host": "main"}})


def test_cache_rollback_drops_the_last_receipts_and_unreceipted_shards(tmp_path):
    cache = tmp_path / "oof1_FIT_fit_eval"
    receipts = []
    for index in range(5):
        for folder in ("structured", "csr"):
            _write(cache / folder / f"batch_{index:08d}.npz", "x")
        receipts.append({"batch_index": index, "rows": 1024})
    _write(cache / "structured" / "batch_00000005.npz", "written without a receipt")
    _write(cache / "BATCHES.json", json.dumps(receipts))
    summary = pipeline.cache_rollback(cache)
    assert summary == {"receipts": 5, "kept": 3, "shards_moved": 5}
    assert [row["batch_index"] for row in json.loads((cache / "BATCHES.json").read_text(encoding="utf-8"))] == [0, 1, 2]
    assert sorted(path.name for path in (cache / "structured").iterdir()) == [f"batch_{i:08d}.npz" for i in range(3)]
    moved = next(cache.glob("rollback_*"))
    assert len(list(moved.rglob("*.npz"))) == 5  # set aside, not deleted


def test_plan_keeps_resumable_dirs_archives_others_and_uses_the_remote_scan(tmp_path):
    ctx = pipeline.ContextPlan(tmp_path, "final")
    cards = {family: [card["id"] for card in family_cards(family)] for family in pipeline.FAMILY_ORDER}
    keep = ctx.root / "candidates" / "team_set_mlp" / cards["team_set_mlp"][0]
    _write(keep / "CANDIDATE_BINDING.json", _binding())
    old = ctx.root / "candidates" / "lightgbm" / cards["lightgbm"][0]
    _write(old / "CANDIDATE_BINDING.json", _binding(scientific="old"))
    rows = {("final", "plain_mlp", cards["plain_mlp"][0]): {"scientific_sha256": "s1", "acceptance_sha256": "a1", "status": "completed"},
            ("final", "plain_mlp", cards["plain_mlp"][1]): {"scientific_sha256": "s1", "acceptance_sha256": "a1", "status": None, "resumable": True},
            ("final", "plain_mlp", cards["plain_mlp"][2]): {"scientific_sha256": "old", "acceptance_sha256": "a1", "status": "completed"}}
    pipeline.plan_candidates(ctx, "s1", "a1", rows, now=100.0)
    tasks = ctx.tasks
    assert keep.exists() and tasks[f"candidate:team_set_mlp:{cards['team_set_mlp'][0]}"]["resume"] is True
    assert not old.exists() and "candidate not terminal" in (ctx.root / "STALE_LOG.jsonl").read_text(encoding="utf-8")
    assert tasks[f"candidate:plain_mlp:{cards['plain_mlp'][0]}"]["state"] == "fetch_ready"
    assert tasks[f"candidate:plain_mlp:{cards['plain_mlp'][1]}"]["remote_until"] == 100.0 + pipeline.REMOTE_AFFINITY_SECONDS
    assert tasks[f"candidate:plain_mlp:{cards['plain_mlp'][2]}"]["state"] == "ready"  # another release: recompute


def test_plan_archives_a_directory_whose_last_attempt_failed_scientifically(tmp_path):
    ctx = pipeline.ContextPlan(tmp_path, "final")
    card = family_cards("team_set_mlp")[0]["id"]
    crashed = ctx.root / "candidates" / "team_set_mlp" / card
    _write(crashed / "CANDIDATE_BINDING.json", _binding())
    _write(crashed / "candidate_result.json", json.dumps({"status": "failed", "error": {
        "type": "RuntimeError", "message": "checkpoint reload changed predictions"}}))
    oom = ctx.root / "candidates" / "team_set_mlp" / family_cards("team_set_mlp")[1]["id"]
    _write(oom / "CANDIDATE_BINDING.json", _binding())
    _write(oom / "candidate_result.json", json.dumps({"status": "failed", "error": {"type": "OutOfMemoryError", "message": "CUDA out of memory"}}))
    pipeline.plan_candidates(ctx, "s1", "a1", {}, now=0.0)
    assert not crashed.exists() and oom.exists()
    assert not ctx.tasks[f"candidate:team_set_mlp:{card}"].get("resume")


def test_a_task_with_local_checkpoints_is_never_sent_to_the_remote(tmp_path):
    supervisor = pipeline.Supervisor(tmp_path, tmp_path / "RELEASE.json", ["final"], dict(pipeline.CONTROL_DEFAULTS, remote_slots=2))
    ctx = supervisor.plans[0]
    ctx.remote.update(state="ready", families={"plain_mlp": "passed"})
    task = {"kind": "candidate", "state": "ready", "family": "plain_mlp", "card_id": "c0"}
    assert supervisor._remote_candidate_ok(ctx, task, now=0.0)
    task["resume"] = True
    assert not supervisor._remote_candidate_ok(ctx, task, now=0.0)


@pytest.mark.parametrize("slots, down_for", [(0, None), (2, pipeline.REMOTE_LOST_SECONDS)])
def test_fetch_hint_falls_back_to_computing_when_the_lane_is_gone(tmp_path, monkeypatch, slots, down_for):
    supervisor = pipeline.Supervisor(tmp_path, tmp_path / "RELEASE.json", ["final"], dict(pipeline.CONTROL_DEFAULTS, remote_slots=slots))
    supervisor.release = {"scientific_sha256": "s1", "scientific_release": "reports/S.json"}
    if down_for is not None:
        supervisor.remote["down_since"] = 0.0
    ctx = supervisor.plans[0]
    ctx.state, ctx.plan = "candidates", {"context": "final"}
    task = {"kind": "candidate", "state": "fetch_ready", "family": "plain_mlp", "card_id": "c0", "resource": "gpu", "cpu": 1, "ram": 0}
    ctx.tasks["candidate:plain_mlp:c0"] = task
    monkeypatch.setattr(pipeline, "_prepare_commands", lambda: [])
    monkeypatch.setattr(pipeline, "launch_local", lambda spec: _Proc())
    monkeypatch.setattr(pipeline, "START_SETTLE_SECONDS", 0)
    supervisor._start_tasks(now=float(down_for or 0) + 1.0, available_ram=1 << 40)
    assert task["state"] == "running"  # computed locally instead of waiting forever


def test_cache_resume_short_of_disk_waits_instead_of_failing(tmp_path):
    supervisor = pipeline.Supervisor(tmp_path, tmp_path / "RELEASE.json", ["oof1"], dict(pipeline.CONTROL_DEFAULTS))
    ctx = supervisor.plans[0]
    log = _write(tmp_path / "b.log", "RuntimeError: {'message': 'insufficient disk headroom', 'estimate': {}}")
    supervisor.builders.append({"key": "oof1:FIT", "ctx": ctx, "scope": "FIT", "process": _Proc(code=1), "slot": (12, 13),
                                "log": log, "resume": True, "t0": 0.0})
    supervisor._reap_builders(now=10.0)
    assert ctx.retries.get("cache:FIT", 0) == 0 and "cache_disk" in supervisor.alarms


def test_remote_snippets_travel_on_stdin_not_the_command_line(monkeypatch):
    from reports.v_production_20261004 import remote

    seen = {}

    def fake_run(command, timeout, stdin=None):
        seen.update(command=command, stdin=stdin)
        return type("R", (), {"stdout": b'{"ok": true}\n'})()

    monkeypatch.setattr(remote, "_run", fake_run)
    long_code = "x = 1\n" * 5000  # far past cmd.exe's 8,191-character command line
    assert remote.remote_json(long_code) == {"ok": True}
    assert len(" ".join(seen["command"])) < 1000 and seen["command"][-1].endswith(" -B -")
    assert seen["stdin"].decode("utf-8").endswith(long_code)


def test_a_waiting_table_fit_keeps_its_ram_from_smaller_tasks(tmp_path, monkeypatch):
    supervisor = pipeline.Supervisor(tmp_path, tmp_path / "RELEASE.json", ["final"], dict(pipeline.CONTROL_DEFAULTS, gpu_slots=8))
    supervisor.release = {"scientific_sha256": "s1", "scientific_release": "reports/S.json"}
    ctx = supervisor.plans[0]
    ctx.state, ctx.plan = "candidates", {"context": "final"}
    GiB = pipeline.GiB
    table = {"kind": "candidate", "state": "ready", "family": "lightgbm", "card_id": "c0", "resource": "table", "cpu": 3, "ram": 20 * GiB}
    neural = {"kind": "candidate", "state": "ready", "family": "team_set_mlp", "card_id": "c0", "resource": "gpu", "cpu": 1, "ram": 5 * GiB}
    ctx.tasks.update({"candidate:lightgbm:c0": table, "candidate:team_set_mlp:c0": neural})
    started = []
    monkeypatch.setattr(pipeline, "_prepare_commands", lambda: [])
    monkeypatch.setattr(pipeline, "launch_local", lambda spec: started.append(pipeline.read_json(spec)["task_key"]) or _Proc())
    monkeypatch.setattr(pipeline, "START_SETTLE_SECONDS", 0)
    supervisor._start_tasks(now=0.0, available_ram=16 * GiB)
    assert started == [] and supervisor.table_wait["needs_bytes"] == 20 * GiB  # 16 GiB: neither starts
    supervisor._start_tasks(now=1.0, available_ram=30 * GiB)
    assert started == ["candidate:lightgbm:c0"] and supervisor.table_wait is None  # the table fit goes first
    supervisor.running[0]["rss"] = 20 * GiB  # the table fit now holds its reservation
    supervisor._start_tasks(now=2.0, available_ram=10 * GiB)
    assert started == ["candidate:lightgbm:c0", "candidate:team_set_mlp:c0"]  # table slot busy: no holdback


def test_control_overrides_ram_reservations_within_bounds(tmp_path):
    _write(tmp_path / "CONTROL.json", json.dumps({"ram_reserve_gib": {"cache": 6, "lightgbm": 18.5, "gate": 100, "bogus": 3}}))
    control = pipeline.read_control(tmp_path, dict(pipeline.CONTROL_DEFAULTS))
    assert control["ram_reserve_gib"] == {"cache": 6.0, "lightgbm": 18.5}  # out of range and unknown names ignored
    assert pipeline.reserve_bytes(control, "lightgbm") == int(18.5 * pipeline.GiB)
    assert pipeline.reserve_bytes(control, "gate") == pipeline.RAM_RESERVE["gate"]
    assert pipeline.reserve_bytes({}, "cache") == pipeline.CACHE_BUILDER_RAM


def test_a_growing_cache_builder_keeps_its_reservation(tmp_path, monkeypatch):
    supervisor = pipeline.Supervisor(tmp_path, tmp_path / "RELEASE.json", ["final"],
                                     dict(pipeline.CONTROL_DEFAULTS, gpu_slots=8, ram_reserve_gib={"cache": 8}))
    supervisor.release = {"scientific_sha256": "s1", "scientific_release": "reports/S.json"}
    ctx = supervisor.plans[0]
    ctx.state, ctx.plan = "candidates", {"context": "final"}
    GiB = pipeline.GiB
    ctx.tasks["candidate:team_set_mlp:c0"] = {"kind": "candidate", "state": "ready", "family": "team_set_mlp", "card_id": "c0",
                                              "resource": "gpu", "cpu": 1, "ram": 5 * GiB}
    supervisor.builders.append({"key": "oof2:FIT", "ctx": ctx, "scope": "FIT", "process": _Proc(), "slot": (12, 13),
                                "log": tmp_path / "b.log", "resume": False, "t0": 0.0, "rss": 2 * GiB})
    started = []
    monkeypatch.setattr(pipeline, "_prepare_commands", lambda: [])
    monkeypatch.setattr(pipeline, "launch_local", lambda spec: started.append(1) or _Proc())
    monkeypatch.setattr(pipeline, "START_SETTLE_SECONDS", 0)
    supervisor._start_tasks(now=0.0, available_ram=10 * GiB)  # 10 - (8 - 2) = 4 GiB left: not enough
    assert started == []
    supervisor.builders[0]["rss"] = 8 * GiB
    supervisor._start_tasks(now=1.0, available_ram=10 * GiB)
    assert started == [1]


def test_table_lane_finishes_one_context_before_the_next():
    final, oof0 = pipeline.ContextPlan(Path("."), "final"), pipeline.ContextPlan(Path("."), "oof0")
    lgbm_oof0 = {"kind": "candidate", "family": "lightgbm", "resource": "table"}
    logistic_final = {"kind": "candidate", "family": "logistic", "resource": "table"}
    assert pipeline.task_sort_key(final, logistic_final) < pipeline.task_sort_key(oof0, lgbm_oof0)
    neural = {"kind": "candidate", "family": "team_set_mlp", "resource": "gpu"}
    assert pipeline.task_sort_key(oof0, neural) < pipeline.task_sort_key(final, {"kind": "candidate", "family": "plain_mlp", "resource": "gpu"})


def test_resume_script_starts_children_in_a_hidden_console():
    assert resume_run.DETACHED & 0x08000000 and not resume_run.DETACHED & 0x00000008  # CREATE_NO_WINDOW, not DETACHED_PROCESS


def test_retry_keeps_a_resumable_candidate_directory(tmp_path):
    supervisor = pipeline.Supervisor(tmp_path, tmp_path / "RELEASE.json", ["final"], dict(pipeline.CONTROL_DEFAULTS))
    supervisor.release = {"scientific_sha256": "s1"}
    ctx = supervisor.plans[0]
    ctx.acceptance_sha, ctx.state = "a1", "candidates"
    out = ctx.root / "candidates" / "team_set_mlp" / "c0"
    _write(out / "CANDIDATE_BINDING.json", _binding())
    _write(out / "checkpoint_epoch_01.pt", "x")
    task = {"kind": "candidate", "state": "running", "family": "team_set_mlp", "card_id": "c0"}
    ctx.tasks["candidate:team_set_mlp:c0"] = task
    supervisor._finish(ctx, "candidate:team_set_mlp:c0", task, 1, 0.0, extra="torch.OutOfMemoryError: CUDA out of memory")
    assert task["state"] == "ready" and task["resume"] is True and (out / "checkpoint_epoch_01.pt").exists()
    events = [json.loads(line)["event"] for line in (tmp_path / "LEDGER.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events == ["task_retry"]


def test_fit_time_counts_checkpointed_segments_once():
    progress = {"phase_resumed_seconds": {"FIT": 20 * 3600.0}}
    assert pipeline.fit_seconds(progress, "FIT", 100.0, 100.0 + 5 * 3600) == 25 * 3600
    assert pipeline.fit_seconds(progress, "REFIT", 100.0, 100.0 + 5 * 3600) == 5 * 3600
    assert pipeline.fit_seconds({}, "FIT", 100.0, 50.0) == 0.0


class _Proc:
    def __init__(self, pid=4242, code=None):
        self.pid, self.code = pid, code

    def poll(self):
        return self.code


def test_orphaned_cache_is_rolled_back_and_resumed_missing_waits_for_the_launcher(tmp_path, monkeypatch):
    supervisor = pipeline.Supervisor(tmp_path, tmp_path / "RELEASE.json", ["oof1"], dict(pipeline.CONTROL_DEFAULTS))
    ctx = supervisor.plans[0]
    launched = []
    monkeypatch.setattr(supervisor, "_launch_builder", lambda c, scope, resume, rollback, now: launched.append((scope, resume, rollback)))
    monkeypatch.setattr(pipeline, "cache_rollback", lambda path: {"kept": 7})
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda path: type("U", (), {"free": 500 << 30})())
    monkeypatch.setattr(pipeline, "launcher_alive", lambda: True)
    supervisor._manage_caches(ctx, {"FIT": "orphaned", "REFIT_TRAIN": "missing"}, now=0.0)
    assert launched == [("FIT", True, {"kept": 7})]  # the live launcher owns the missing one
    monkeypatch.setattr(pipeline, "launcher_alive", lambda: False)
    supervisor._manage_caches(ctx, {"FIT": "building", "REFIT_TRAIN": "missing"}, now=0.0)
    assert launched[-1] == ("REFIT_TRAIN", False, {})


def test_failed_cache_builds_retry_then_fail_the_context(tmp_path, monkeypatch):
    supervisor = pipeline.Supervisor(tmp_path, tmp_path / "RELEASE.json", ["oof1"], dict(pipeline.CONTROL_DEFAULTS))
    ctx = supervisor.plans[0]
    for attempt in range(pipeline.CACHE_BUILD_RETRIES):
        supervisor.builders.append({"key": "oof1:FIT", "ctx": ctx, "scope": "FIT", "process": _Proc(code=1), "slot": (12, 13),
                                    "log": tmp_path / "b.log", "resume": True, "t0": 0.0})
        supervisor._reap_builders(now=10.0)
        assert supervisor.builder_not_before["oof1:FIT"] == 10.0 + pipeline.RETRY_COOLDOWN_SECONDS
    supervisor._manage_caches(ctx, {"FIT": "orphaned", "REFIT_TRAIN": "ready"}, now=10_000.0)
    assert ctx.state == "failed" and "failed 2 builds" in ctx.error


def test_remote_result_found_at_start_is_fetched_and_a_failed_hint_is_recomputed(tmp_path, monkeypatch):
    supervisor = pipeline.Supervisor(tmp_path, tmp_path / "RELEASE.json", ["final"], dict(pipeline.CONTROL_DEFAULTS, remote_slots=2))
    supervisor.release = {"scientific_sha256": "s1"}
    ctx = supervisor.plans[0]
    ctx.state, ctx.acceptance_sha = "candidates", "a1"
    task = {"kind": "candidate", "state": "fetch_ready", "family": "plain_mlp", "card_id": "c0", "resource": "gpu", "cpu": 1, "ram": 0}
    ctx.tasks["candidate:plain_mlp:c0"] = task
    started = []
    monkeypatch.setattr(supervisor, "_start_helper", lambda c, kind, args, meta=None: started.append((kind, meta)))
    monkeypatch.setattr(pipeline, "_prepare_commands", lambda: [])
    supervisor._start_tasks(now=0.0, available_ram=1 << 40)
    assert task["state"] == "fetching" and started == [("fetch", {"key": "candidate:plain_mlp:c0", "hint": True})]
    log = _write(tmp_path / "fetch.log", "scp: Connection closed")
    supervisor.helpers.append({"kind": "fetch", "ctx": ctx, "process": _Proc(code=1), "log": log, "meta": started[0][1], "t0": 0.0})
    supervisor._reap_helpers(now=1.0)
    assert task["state"] == "ready" and ctx.state == "candidates"


def test_remote_checkpoints_keep_a_task_off_the_local_gpu_for_a_while(tmp_path, monkeypatch):
    supervisor = pipeline.Supervisor(tmp_path, tmp_path / "RELEASE.json", ["final"], dict(pipeline.CONTROL_DEFAULTS))
    supervisor.release = {"scientific_sha256": "s1", "scientific_release": "reports/S.json"}
    ctx = supervisor.plans[0]
    ctx.state, ctx.plan = "candidates", {"context": "final"}
    task = {"kind": "candidate", "state": "ready", "family": "plain_mlp", "card_id": "c0", "resource": "gpu", "cpu": 1, "ram": 0,
            "remote_until": 100.0}
    ctx.tasks["candidate:plain_mlp:c0"] = task
    monkeypatch.setattr(pipeline, "_prepare_commands", lambda: [])
    monkeypatch.setattr(pipeline, "launch_local", lambda spec: _Proc())
    monkeypatch.setattr(pipeline, "START_SETTLE_SECONDS", 0)
    supervisor._start_tasks(now=50.0, available_ram=1 << 40)
    assert task["state"] == "ready"  # waits for the lane that holds its checkpoints
    supervisor._start_tasks(now=150.0, available_ram=1 << 40)
    assert task["state"] == "running"


def test_resume_script_starts_guard_and_supervisor_with_the_active_release(tmp_path, monkeypatch):
    run_root = tmp_path / "run_3"
    _write(run_root / "ACTIVE_RELEASE.json", json.dumps({"release": "reports/v_production_20261004/RELEASE_X.json"}))
    started = []
    monkeypatch.setattr(resume_run, "running", lambda script: [])
    monkeypatch.setattr(resume_run, "start", lambda command, log: started.append(command) or 77)
    assert resume_run.main(["--run-root", str(run_root)]) == 0
    assert started[0][-1].endswith("ram_guard.py")
    assert started[1][-4:] == ["--run-root", str(run_root.resolve()), "--release", "reports/v_production_20261004/RELEASE_X.json"]
    assert "resume_requested" in (run_root / "LEDGER.jsonl").read_text(encoding="utf-8")
