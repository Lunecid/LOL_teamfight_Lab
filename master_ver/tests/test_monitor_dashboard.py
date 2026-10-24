"""Read-only run_3 dashboard: file sharing, process table and display summaries."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import psutil
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reports" / "monitor_20261005"))

import dashboard  # noqa: E402


def test_any_open_reader_blocks_replace_so_writers_must_retry(tmp_path):
    """Windows refuses os.replace onto a file a reader holds open, even one opened with FILE_SHARE_DELETE.

    So the dashboard keeps each read short, reads only files whose writers retry
    WinError 5/32 (or append-only logs), reads a finished candidate result once,
    and never opens ram_guard_state.json.
    """

    target, pending = tmp_path / "progress.json", tmp_path / ".progress.tmp"
    target.write_text('{"a": 1}', encoding="utf-8")
    pending.write_text('{"a": 2}', encoding="utf-8")
    with dashboard.open_shared(target) as stream:
        assert json.loads(stream.read()) == {"a": 1}
        with pytest.raises(PermissionError):
            os.replace(pending, target)
    os.replace(pending, target)
    assert dashboard.read_json(target) == {"a": 2}


def test_ram_guard_state_write_retries_while_a_reader_holds_the_file(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("ram_guard_under_test", ROOT / "reports" / "dual_worker_schedule_20261004" / "ram_guard.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    monkeypatch.setattr(guard, "STATE", tmp_path / "ram_guard_state.json")
    guard.STATE.write_text("{}", encoding="utf-8")
    reader = open(guard.STATE, "rb")
    threading.Timer(0.3, reader.close).start()
    guard.save_state({123: 1.5})
    assert json.loads(guard.STATE.read_text(encoding="utf-8")) == {"123": 1.5}


def test_candidate_grid_reads_a_finished_result_once(tmp_path, monkeypatch):
    folder = tmp_path / "final" / "candidates" / "logistic" / "lr_C0.001"
    folder.mkdir(parents=True)
    (folder / "candidate_result.json").write_text('{"status": "completed"}', encoding="utf-8")
    reads = []
    real = dashboard.read_json
    monkeypatch.setattr(dashboard, "read_json", lambda path: reads.append(path) or real(path))
    for _ in range(3):
        grid = dashboard.candidate_grid(tmp_path, "final")
    assert grid["logistic"][0]["state"] == "done" and grid["logistic"][1]["state"] == "waiting"
    assert len(reads) == 1


def test_read_jsonl_tail_drops_partial_first_line(tmp_path):
    path = tmp_path / "LEDGER.jsonl"
    path.write_text("".join(json.dumps({"event": "e", "i": i}) + "\n" for i in range(400)), encoding="utf-8")
    rows = dashboard.read_jsonl_tail(path, 5, tail_bytes=1000)
    assert [row["i"] for row in rows] == [395, 396, 397, 398, 399]
    assert all(set(row) == {"event", "i"} for row in dashboard.read_jsonl_tail(path, 50, tail_bytes=1000))


def test_process_table_matches_psutil_and_sees_suspension():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], creationflags=0x08000000)
    try:
        time.sleep(0.5)
        row = next(r for r in dashboard.process_table() if r["pid"] == child.pid)
        memory = psutil.Process(child.pid).memory_info()
        assert row["name"].lower() == "python.exe" and not row["suspended"]
        assert abs(row["ws"] - memory.rss) < 32 << 20 and abs(row["private"] - memory.vms) < 32 << 20
        psutil.Process(child.pid).suspend()
        time.sleep(0.3)
        assert next(r for r in dashboard.process_table() if r["pid"] == child.pid)["suspended"]
        psutil.Process(child.pid).resume()
    finally:
        child.kill()
        child.wait()


def test_classify_roles():
    worker = ["python.exe", "-B", r"C:\x\LOL_Teamfight\studies\v_production_20261004\worker.py", "--spec",
              r"D:\LOL_Training_20261004\run_3\final\workers\candidate__logistic__lr_C0.001__20261005T004205Z.json"]
    gate = worker[:-1] + [r"D:\LOL_Training_20261004\run_3\oof3\workers\gate__REFIT_TRAIN__team_set_mlp__20261004T165233Z.json"]
    prepare = ["python.exe", r"C:\x\LOL_Teamfight\studies\v_production_20261004\prepare.py", "--context", "oof2", "--scope", "FIT"]
    assert dashboard.classify(worker) == ("worker", "final · logistic/lr_C0.001")
    assert dashboard.classify(gate) == ("worker", "oof3 · 게이트 REFIT_TRAIN/team_set_mlp")
    assert dashboard.classify(prepare) == ("cache", "캐시 생성 oof2 FIT")
    assert dashboard.classify(["python.exe", "-m", "http.server"]) is None


def test_context_summary_by_state():
    gates = dashboard.context_summary("oof3", {"state": "gates", "tasks": {"done": 4, "ready": 6}}, False)
    assert (gates["gates_done"], gates["cands_done"]) == (4, 0)
    cands = dashboard.context_summary("final", {"state": "candidates", "tasks": {"done": 22, "ready": 5, "running": 1, "running_remote": 2}}, False)
    assert (cands["gates_done"], cands["cands_done"], cands["running_remote"]) == (10, 12, 2)
    locked = dashboard.context_summary("oof0", {"state": "candidates", "tasks": {"done": 3}}, True)
    assert (locked["state"], locked["cands_done"]) == ("done", dashboard.CANDIDATES_PER_CONTEXT)


def test_worker_progress_epoch_and_stage():
    counts = {"fit": 830209, "stop": 93276, "refit": 923485}
    epoch = dashboard.worker_progress({"stage": "fit", "epoch": 12, "rows": 415104, "total_rows": 830209 * 30, "eta_seconds": 14000}, counts)
    assert epoch["basis"] == "epoch" and epoch["max_epochs"] == 30 and epoch["eta_seconds"] is None
    assert abs(epoch["fraction"] - 415104 / 830209) < 1e-12
    stage = dashboard.worker_progress({"stage": "stop", "epoch": 4, "rows": 8192, "total_rows": 74446, "eta_seconds": 30}, counts)
    assert stage["basis"] == "stage" and stage["eta_seconds"] == 30
    assert dashboard.worker_progress({"stage": "refit_solver", "rows": 0, "total_rows": None}, counts)["fraction"] is None


def test_overlay_running_keeps_done_cells():
    grids = {"final": {"logistic": [{"card": "lr_C0.001", "state": "waiting"}, {"card": "lr_C0.01", "state": "done"}]}}
    workers = [{"context": "final", "task": "candidate:logistic:lr_C0.001", "host": "main"},
               {"context": "final", "task": "candidate:logistic:lr_C0.01", "host": "rtx2070"}]
    cells = dashboard.overlay_running(grids, workers)["final"]["logistic"]
    assert [cell["state"] for cell in cells] == ["main", "done"]
    assert grids["final"]["logistic"][0]["state"] == "waiting"  # input untouched
