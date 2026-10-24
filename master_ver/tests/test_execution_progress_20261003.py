from __future__ import annotations

import datetime as dt
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from studies.execution_progress_20261003 import progress as ep


@pytest.mark.skipif(os.name != "nt", reason="Windows process API regression")
def test_windows_liveness_is_read_only_for_real_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        assert ep.is_pid_alive(child.pid) is True
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=10)
    assert ep.is_pid_alive(child.pid) is False


class FakeClock:
    def __init__(self) -> None:
        self.monotonic = 0.0
        self.utc = dt.datetime(2026, 10, 3, 0, 0, tzinfo=dt.timezone.utc)

    def tick(self, seconds: float) -> None:
        self.monotonic += float(seconds)
        self.utc += dt.timedelta(seconds=float(seconds))

    def now_mono(self) -> float:
        return self.monotonic

    def now_utc(self) -> dt.datetime:
        return self.utc


def _writer(tmp_path: Path, *, plan=None, total=10, pid=1, clock=None):
    clock = clock or FakeClock()
    writer = ep.ProgressStateWriter(
        tmp_path / "progress.json",
        run_id="test-run",
        stage="load",
        total_units=total,
        pid=pid,
        clock=clock.now_mono,
        utc_clock=clock.now_utc,
        stage_plan=plan,
    )
    return writer, clock


def test_writer_records_monotonic_observations_and_rate_eta(tmp_path):
    writer, clock = _writer(tmp_path, pid=123)
    clock.tick(10)
    writer.update(completed=5)
    clock.tick(10)
    state = writer.update(completed=10)

    assert state["run_id"] == "test-run"
    assert state["stage"] == "load"
    assert state["status"] == "running"
    assert state["completed"] == 10
    assert state["total"] == 10
    assert state["elapsed_seconds"] == 20.0
    assert state["started_at_utc"].endswith("Z")
    metrics = state["stage_metrics"]["load"]
    assert [item["completed"] for item in metrics["observations"]] == [0, 5, 10]

    snapshot = ep.derive_progress(state, now_utc=clock.now_utc, pid_checker=lambda pid: True)
    assert snapshot["rate_units_per_second"] == pytest.approx(0.5)
    assert snapshot["stage_eta_seconds"] == 0.0
    assert snapshot["total_eta_seconds"] is None


def test_insufficient_observations_keep_eta_unknown(tmp_path):
    writer, clock = _writer(tmp_path, pid=123)
    clock.tick(10)
    state = writer.update(completed=1)
    snapshot = ep.derive_progress(state, now_utc=clock.now_utc, pid_checker=lambda pid: True)
    assert snapshot["rate_units_per_second"] is None
    assert snapshot["stage_eta_seconds"] is None
    assert snapshot["total_eta_seconds"] is None
    assert "eta=unknown" in ep.render_progress(snapshot)


def test_total_eta_requires_estimate_for_every_future_stage(tmp_path):
    plan = {"load": None, "fit": 30.0, "save": 10.0}
    writer, clock = _writer(tmp_path, plan=plan, pid=123)
    clock.tick(10)
    writer.update(completed=5)
    clock.tick(10)
    state = writer.update(completed=10)
    snapshot = ep.derive_progress(state, now_utc=clock.now_utc, pid_checker=lambda pid: True)
    assert snapshot["stage_eta_seconds"] == 0.0
    assert snapshot["total_eta_seconds"] == 40.0

    writer2, clock2 = _writer(tmp_path / "unknown", plan={"load": None, "fit": None}, pid=123)
    clock2.tick(10)
    writer2.update(completed=5)
    clock2.tick(10)
    state2 = writer2.update(completed=8)
    snapshot2 = ep.derive_progress(state2, now_utc=clock2.now_utc, pid_checker=lambda pid: True)
    assert snapshot2["total_eta_seconds"] is None


def test_stale_pid_suppresses_eta_without_fabricating_progress(tmp_path):
    writer, clock = _writer(tmp_path, pid=999)
    clock.tick(10)
    writer.update(completed=5)
    clock.tick(10)
    state = writer.update(completed=6)
    snapshot = ep.derive_progress(state, now_utc=clock.now_utc, pid_checker=lambda pid: False)
    assert snapshot["stale"] is True
    assert "pid_not_alive" in snapshot["stale_reasons"]
    assert snapshot["stage_eta_seconds"] is None
    assert "stale=yes" in ep.render_progress(snapshot)


def test_stage_regression_and_count_reset_fail_closed(tmp_path):
    writer, clock = _writer(tmp_path, plan={"load": None, "fit": 5.0}, pid=123)
    clock.tick(1)
    writer.set_stage("fit", total=5)
    with pytest.raises(ep.ProgressStateError, match="stage regression"):
        writer.set_stage("load")
    failed = ep.read_state(tmp_path / "progress.json")
    assert failed["status"] == "failed"
    assert failed["completed"] == 0
    assert failed["last_error"]["type"] == "ProgressStateError"
    assert "stage regression" in failed["last_error"]["traceback"]

    writer2, clock2 = _writer(tmp_path / "reset", pid=123)
    clock2.tick(1)
    writer2.update(completed=3)
    with pytest.raises(ep.ProgressStateError, match="cannot regress"):
        writer2.update(completed=2)
    failed2 = ep.read_state(tmp_path / "reset" / "progress.json")
    assert failed2["status"] == "failed"
    assert failed2["completed"] == 3


def test_inconsistent_counts_are_invalid_and_do_not_get_a_fraction(tmp_path):
    writer, clock = _writer(tmp_path, pid=123)
    state = writer.state
    state["completed"] = 11
    snapshot = ep.derive_progress(state, pid_checker=lambda pid: True)
    assert snapshot["fraction"] is None
    assert snapshot["stage_eta_seconds"] is None
    assert snapshot["validation_errors"]
    assert "completed cannot exceed total" in snapshot["validation_errors"]


def test_atomic_reader_retries_a_transient_partial_document(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    target.write_text('{"ok": true}\n', encoding="utf-8")
    original_open = ep.Path.open
    responses = iter(["{", '{"ok": true}\n'])

    class FakeHandle:
        def __init__(self, text):
            self.text = text

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self.text

    def transient_open(self, *args, **kwargs):
        try:
            text = next(responses)
        except StopIteration:
            return original_open(self, *args, **kwargs)
        return FakeHandle(text)

    monkeypatch.setattr(ep.Path, "open", transient_open)
    assert ep.read_state(target, retries=2, retry_delay=0, sleep=lambda _: None) == {"ok": True}


def test_failed_state_trace_is_rendered_and_watch_returns_error(tmp_path):
    writer, clock = _writer(tmp_path, pid=123)
    error = RuntimeError("worker exploded")
    writer.fail(error, traceback_text="Traceback (most recent call last):\nworker exploded\n")
    output = io.StringIO()
    code = ep.watch_state(tmp_path / "progress.json", once=True, stream=output)
    rendered = output.getvalue()
    assert code == 1
    assert "ERROR RuntimeError: worker exploded" in rendered
    assert "worker exploded" in rendered


def test_terminal_state_cannot_become_silent_success(tmp_path):
    writer, _ = _writer(tmp_path, total=None, pid=123)
    writer.complete()
    with pytest.raises(ep.ProgressStateError, match="terminal state"):
        writer.update(status="running", completed=1)
    state = ep.read_state(tmp_path / "progress.json")
    assert state["status"] == "completed"
    assert state["last_error"] is None


def test_complete_rejects_unfinished_future_stage_and_records_failure(tmp_path):
    writer, clock = _writer(tmp_path, plan={"load": None, "fit": 5.0}, pid=123)
    clock.tick(1)
    writer.update(completed=10)
    with pytest.raises(ep.ProgressStateError, match="final stage"):
        writer.complete()
    state = ep.read_state(tmp_path / "progress.json")
    assert state["status"] == "failed"
    assert state["completed"] == 10
    assert "final stage" in state["last_error"]["message"]
