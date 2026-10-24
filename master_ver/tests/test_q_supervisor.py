"""q supervisor: the full DAG on the real plan with a fake launcher (selection, reuse, retry, resume, calibration)."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("q_supervisor", ROOT / "reports" / "q_production_20261005" / "q_supervisor.py")
sup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sup)

DIGEST = "d" * 64


class Done:
    """A 'process' that already finished: the fake launcher writes its outputs at launch."""

    _next = 10 ** 9

    def __init__(self) -> None:
        Done._next += 1
        self.pid = Done._next

    def poll(self):
        return 0


def _brier(identity: str) -> float:
    return 0.2 + int(identity[:6], 16) / 16 ** 6 * 0.01


class FakeLauncher:
    def __init__(self, fail_once=(), nonconverged=("lr_C0.001",), identical=()) -> None:
        self.fail_once, self.nonconverged, self.launched, self.identical = set(fail_once), set(nonconverged), [], set(identical)

    def __call__(self, command, log, env, below_normal):
        self.launched.append(command)
        if any("q_prepare.py" in str(part) for part in command):
            run, variant = Path(command[command.index("--run") + 1]), command[command.index("--variant") + 1]
            base = run / "inputs" / variant
            base.mkdir(parents=True, exist_ok=True)
            (base / "PREPARED.json").write_text(json.dumps({"release_digest": DIGEST, "rows_file": f"inputs/{variant}/Q_ROWS.npz", "rows_sha256": "r",
                                                            "xy_sha256": "x", "contexts": {s: {"dir": f"inputs/{variant}/{s}"} for s in ("FIT", "REFIT_TRAIN")},
                                                            **({"identical_to": "main_h90"} if variant in self.identical else {})}),
                                              encoding="utf-8")
            return Done()
        spec_path = Path(command[command.index("--spec") + 1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        out = Path(spec["out_dir"])
        if spec.get("kind") == "calibrate":
            cal = Path(spec["calibrators_dir"])
            cal.mkdir(parents=True, exist_ok=True)
            for target in spec["targets"]:
                (cal / f"{sup.key_of(target['job_id'])}.json").write_text(json.dumps({"calibrator": {"status": "fitted"}}), encoding="utf-8")
            result = {"status": "completed"}
        else:
            job = spec["job"]
            if job["family"] == "logistic" and job["card_id"] in self.nonconverged:
                result = {"status": "failed", "reason": "nonconverged", "terminal": True}
            elif job["job_id"] in self.fail_once:
                self.fail_once.discard(job["job_id"])
                result = {"status": "failed", "reason": "RuntimeError", "terminal": False}
            else:
                result = {"status": "completed", "scores": {"q_dev": {"raw_match_brier": _brier(job["identity"])}}, "q_dev_keys_sha256": "K"}
        (out / "RESULT.json").write_text(json.dumps(result), encoding="utf-8")
        return Done()


def _run(tmp_path):
    run = tmp_path / "q_run"
    (run / "raw" / "store").mkdir(parents=True)
    (run / "raw" / "store" / "BUILD.json").write_text("{}", encoding="utf-8")
    for h in (60, 90, 120):
        (run / f"Q_ROWS_h{h}.npz").write_bytes(b"")
    rel = tmp_path / "SCIENTIFIC_Q_test.json"
    rel.write_text(json.dumps({"digest": DIGEST}), encoding="utf-8")
    sup.write_json(run / "CONTROL.json", {**sup.DEFAULT_CONTROL, "gpu_slots": 500, "cpu_slots": 500, "hold_families": [], "poll_s": 0})
    return run, rel


def _drive(supervisor, steps=60):
    for _ in range(steps):
        supervisor.step()
    return json.loads((supervisor.run / "COVERAGE.json").read_text(encoding="utf-8"))


def test_full_dag_with_reuse_retry_and_resume(tmp_path):
    run, rel = _run(tmp_path)
    retry_job = "final/module/N_G1_S1_A0_M0_E1_C1_R1_D1/relational_mpnn/selected:common"
    launcher = FakeLauncher(fail_once={retry_job}, identical={"exclude_zero_coordinate_h90"})
    first = sup.Supervisor(run, release_path=rel, launcher=launcher, probe=lambda: {"ram_free_gib": None, "vram_free_gib": None})
    for _ in range(3):
        first.step()
    second = sup.Supervisor(run, release_path=rel, launcher=launcher, probe=lambda: {"ram_free_gib": None, "vram_free_gib": None})
    coverage = _drive(second)
    selection = json.loads((run / "Q_SEARCH_SELECTION.json").read_text(encoding="utf-8"))
    assert selection["common"]["complete"] and selection["common"]["denominator"] == 10
    assert selection["logistic"]["selected"] != "lr_C0.001" and not selection["logistic"]["complete"]
    reused = sum(v.get("reused", 0) for v in coverage["by_block"].values())
    assert reused == 19 + 3                                             # search identities (FT removed) + D-Q16 identical variant
    final = json.loads((run / "Q_SELECTION.json").read_text(encoding="utf-8"))
    assert final["selection"]["complete"] and final["selection"]["selected"] in sup.qselect.FAMILY_ORDER
    assert (run / "jobs" / sup.key_of(retry_job) / "RESULT_attempt1.json").is_file()
    assert json.loads((run / "jobs" / sup.key_of(retry_job) / "RESULT.json").read_text(encoding="utf-8"))["status"] == "completed"
    launches = [c for c in launcher.launched if any("q_worker.py" in str(p) for p in c)]
    specs = [c[c.index("--spec") + 1] for c in launches]
    assert len(specs) == len(set(specs)) + 1                           # one retry, otherwise every spec launched once
    assert not coverage["failed"] and not coverage["exhausted"] and not coverage["blocked"]
    calibrators = len(list((run / "calibrators").glob("*.json")))
    assert calibrators == 176 + 2                                       # every final result and the two selected pure-set cards
    twin = json.loads((run / "jobs" / sup.key_of("final/sensitivity/N_G1_S1_A1_M1_E1_C1_R1_D1/relational_mpnn/selected:common/exclude_zero_coordinate_h90")
                       / "REUSE.json").read_text(encoding="utf-8"))
    assert "D-Q16" in twin["reason"] and twin["reuse_of_key"].startswith("search__common__N_G1_S1_A1_M1_E1_C1_R1_D1")
    ledger = [json.loads(line) for line in (run / "LEDGER.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["event"] == "interrupted" for e in ledger)            # workers listed by the first supervisor were not alive
    prepared = [c for c in launcher.launched if any("q_prepare.py" in str(p) for p in c)]
    assert len(prepared) == 5


def test_held_and_abandoned_families(tmp_path):
    run, rel = _run(tmp_path)
    sup.write_json(run / "CONTROL.json", {**sup.DEFAULT_CONTROL, "gpu_slots": 500, "cpu_slots": 500, "poll_s": 0, "hold_families": ["graphsage"]})
    supervisor = sup.Supervisor(run, release_path=rel, launcher=FakeLauncher(), probe=lambda: {"ram_free_gib": None, "vram_free_gib": None})
    coverage = _drive(supervisor, 40)
    assert coverage["held"] and all("graphsage" in job for job in coverage["held"])
    assert not (run / "Q_SELECTION.json").exists()                      # P1 waits for the held family
    sup.write_json(run / "CONTROL.json", {**sup.DEFAULT_CONTROL, "gpu_slots": 500, "cpu_slots": 500, "poll_s": 0, "abandon_families": ["graphsage"]})
    _drive(supervisor, 10)
    final = json.loads((run / "Q_SELECTION.json").read_text(encoding="utf-8"))
    assert final["selection"]["complete"] is False and "graphsage" in final["selection"]["missing"]


def test_pause_prepares_inputs_but_trains_nothing(tmp_path):
    run, rel = _run(tmp_path)
    sup.write_json(run / "CONTROL.json", {**sup.DEFAULT_CONTROL, "gpu_slots": 500, "cpu_slots": 500, "poll_s": 0, "pause": True})
    launcher = FakeLauncher()
    supervisor = sup.Supervisor(run, release_path=rel, launcher=launcher, probe=lambda: {"ram_free_gib": None, "vram_free_gib": None})
    _drive(supervisor, 5)
    assert all(any("q_prepare.py" in str(p) for p in c) for c in launcher.launched) and len(launcher.launched) == 5


def test_release_and_plan_binding(tmp_path):
    run, rel = _run(tmp_path)
    sup.Supervisor(run, release_path=rel, launcher=FakeLauncher(), probe=lambda: {})
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"digest": "e" * 64}), encoding="utf-8")
    with pytest.raises(SystemExit, match="bound to release"):
        sup.Supervisor(run, release_path=other, launcher=FakeLauncher(), probe=lambda: {})


def test_admission_respects_ram(tmp_path):
    run, rel = _run(tmp_path)
    sup.write_json(run / "CONTROL.json", {**sup.DEFAULT_CONTROL, "gpu_slots": 500, "cpu_slots": 500, "poll_s": 0, "variants": []})
    launcher = FakeLauncher()
    supervisor = sup.Supervisor(run, release_path=rel, launcher=launcher, probe=lambda: {"ram_free_gib": 20.0, "vram_free_gib": 100.0})
    supervisor.step()                    # prepare main_h90 (8 GiB) fits in 20 - 8 reserve
    supervisor.step()                    # neural jobs at 4 GiB each: (20 - 8) / 4 = 3 in one step
    started = [c for c in launcher.launched if any("q_worker.py" in str(p) for p in c)]
    assert len(started) == 3
    supervisor.step()                    # the three are done (fake), but nothing new is reserved for them: 3 more
    started = [c for c in launcher.launched if any("q_worker.py" in str(p) for p in c)]
    assert len(started) == 6
