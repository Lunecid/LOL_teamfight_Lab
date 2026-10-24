"""Focused runtime contract tests; no model fitting or CUDA probe is run."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from studies.training_runtime_20261002 import preflight as PF
from studies.training_runtime_20261002.profiles import (
    CONTROLLED_ENV_VARS,
    GIB,
    compute_gpu_allocator_cap,
    cpu_allocation,
    read_current_affinity,
    select_cpu_ids,
)


def _four_current_cpus() -> list[int]:
    current = read_current_affinity()
    if len(current) < 4:
        pytest.skip("the test process needs four logical CPUs")
    return current[:4]


def test_cpu_budget_and_phase_allocation_has_no_conflicting_backend_aliases():
    allocation = cpu_allocation([10, 12, 14, 16])
    assert allocation["feasible"] is True
    assert allocation["allowed_logical_cpus"] == [10, 12, 14, 16]
    assert allocation["extraction"]["cpus"] == [10, 12, 14]
    assert allocation["extraction"]["workers"] == 3
    assert allocation["extraction"]["threads_per_worker"] == 1
    assert allocation["cpu_fit"]["cpus"] == [10, 12, 14]
    assert allocation["cpu_fit"]["threads"] == 3
    assert allocation["cpu_fit"]["library_profile"] == {"name": "linear", "n_jobs": 1}
    assert allocation["cpu_fit"]["fit_backend"] == {"name": "lightgbm", "lgbm_num_threads": 3}
    assert "sklearn_n_jobs" not in allocation["cpu_fit"]["fit_backend"]
    assert allocation["gpu_host"]["cpus"] == [16]
    assert allocation["supervisor"]["cpus"] == [16]
    assert allocation["phases_exclusive"] == ["extraction", "cpu_fit"]


def test_explicit_cpu_ids_must_be_unique_and_within_current_affinity():
    assert select_cpu_ids([10, 12, 14, 16], current_cpus=[10, 12, 14, 16, 18]) == [
        10,
        12,
        14,
        16,
    ]
    with pytest.raises(ValueError, match="four unique"):
        select_cpu_ids([10, 12, 14], current_cpus=[10, 12, 14, 16])
    with pytest.raises(ValueError, match="within"):
        select_cpu_ids([10, 12, 14, 99], current_cpus=[10, 12, 14, 16])


def test_gpu_allocator_cap_boundaries():
    total = 10 * GIB
    assert compute_gpu_allocator_cap(total, 9 * GIB) == min(int(total * 0.85), 8 * GIB)
    assert compute_gpu_allocator_cap(total, 2 * GIB) == GIB
    assert compute_gpu_allocator_cap(total, GIB) is None
    assert compute_gpu_allocator_cap(0, 4 * GIB) is None


def test_child_environment_overrides_without_mutating_parent(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "parent")
    before = {name: os.environ.get(name) for name in CONTROLLED_ENV_VARS}
    child = PF._child_environment("gpu_host")
    after = {name: os.environ.get(name) for name in CONTROLLED_ENV_VARS}
    assert after == before
    assert child["OMP_NUM_THREADS"] == "1"
    assert child["OMP_MAX_ACTIVE_LEVELS"] == "1"
    assert child["OMP_NESTED"] == "FALSE"
    assert child["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert all(child[name] == "1" for name in CONTROLLED_ENV_VARS[:6])
    assert child["NVIDIA_TF32_OVERRIDE"] == "0"


def test_fresh_cpu_child_reports_settings_and_does_not_change_parent(monkeypatch):
    cpus = _four_current_cpus()
    monkeypatch.setenv("OMP_NUM_THREADS", "77")
    before = os.environ["OMP_NUM_THREADS"]
    report = PF._run_child_probe("cpu", cpu_ids=cpus[:3], project_root=PF.PROJECT_ROOT)
    assert report["ok"] is True
    assert report["probe_status"] == "passed"
    assert report["actual_affinity"] == cpus[:3]
    assert report["controlled_environment"]["OMP_NUM_THREADS"] == "3"
    assert report["controlled_environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert report["fit_performed"] is False
    assert report["threadpools_ok"] is True
    assert report["observed_thread_counts"]
    assert all(1 <= value <= 3 for value in report["observed_thread_counts"])
    assert os.environ["OMP_NUM_THREADS"] == before


def test_threadpool_introspection_is_fail_closed():
    assert PF._validate_threadpool_info([{"user_api": "blas", "num_threads": 3}]) == (
        True,
        None,
        [3],
    )
    valid, reason, observed = PF._validate_threadpool_info([])
    assert valid is False
    assert "no numerical threadpools" in reason
    assert observed == []
    valid, reason, observed = PF._validate_threadpool_info([{"num_threads": 4}])
    assert valid is False
    assert "invalid observed" in reason
    assert observed == []


def test_nonzero_child_exit_overrides_success_json(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=7,
            stdout=json.dumps({"ok": True, "probe_status": "passed"}),
            stderr="child failed",
        )

    monkeypatch.setattr(PF.subprocess, "run", fake_run)
    report = PF._run_child_probe("cpu", cpu_ids=[0, 2, 4], project_root=PF.PROJECT_ROOT)
    assert report["ok"] is False
    assert report["probe_passed"] is False
    assert report["probe_status"] == "failed"


def test_resource_gate_fails_before_any_probe(monkeypatch):
    cpus = _four_current_cpus()
    output = PF.PROJECT_ROOT / "reports" / "compute_execution_design_20261002" / "test-resource-gate"
    if output.exists():
        pytest.skip("test output target already exists")
    monkeypatch.setattr(
        PF,
        "_host_resources",
        lambda root: {
            "host_memory": {"feasible": True},
            "disk_workspace_free_bytes": 0,
            "resource_infeasible_reasons": ["workspace disk free bytes are non-positive"],
            "resource_feasible": False,
        },
    )

    def unexpected_probe(*args, **kwargs):
        pytest.fail("resource-infeasible preflight must not launch a child probe")

    monkeypatch.setattr(PF, "_run_child_probe", unexpected_probe)
    manifest = PF.run_preflight(
        out_dir=output,
        probe_cuda="off",
        cpu_ids=cpus,
        project_root=PF.PROJECT_ROOT,
    )
    assert manifest["status"] == "failed"
    assert manifest["all_ok"] is False
    assert manifest["probes"] == {}
    assert manifest["resources"]["resource_feasible"] is False
    assert "disk free bytes" in manifest["error"]["message"]
    for path in output.iterdir():
        path.unlink()
    output.rmdir()


def test_cpu_probe_uses_fit_cpus_and_records_supervisor_affinity(monkeypatch):
    cpus = _four_current_cpus()
    output = PF.PROJECT_ROOT / "reports" / "compute_execution_design_20261002" / "test-supervisor-affinity"
    if output.exists():
        pytest.skip("test output target already exists")
    monkeypatch.setattr(
        PF,
        "_host_resources",
        lambda root: {
            "host_memory": {"feasible": True},
            "disk_workspace_free_bytes": 1,
            "resource_infeasible_reasons": [],
            "resource_feasible": True,
        },
    )
    calls = []

    def fake_probe(kind, **kwargs):
        calls.append((kind, list(kwargs["cpu_ids"])))
        return {"ok": True, "probe_status": "passed", "fit_performed": False}

    monkeypatch.setattr(PF, "_run_child_probe", fake_probe)
    monkeypatch.setattr(PF, "_pin_supervisor_affinity", lambda ids: ([1, 2, 3, 4], list(ids)))
    monkeypatch.setattr(PF, "_restore_supervisor_affinity", lambda previous: None)
    manifest = PF.run_preflight(
        out_dir=output,
        probe_cuda="off",
        cpu_ids=cpus,
        project_root=PF.PROJECT_ROOT,
    )
    assert manifest["status"] == "success"
    assert calls == [("cpu", cpus[:3])]
    assert manifest["supervisor"]["actual_affinity"] == [cpus[3]]
    assert manifest["supervisor"]["restored_after_run"] is True
    for path in output.iterdir():
        path.unlink()
    output.rmdir()


def test_output_path_escape_and_existing_output_are_rejected(tmp_path):
    outside = Path(PF.PROJECT_ROOT.anchor) / "training_runtime_outside"
    with pytest.raises(PF.PreflightError, match="inside project workspace"):
        PF.validate_output_directory(outside, project_root=PF.PROJECT_ROOT)
    existing = PF.PROJECT_ROOT / "reports" / "compute_execution_design_20261002" / "test-existing-output"
    existing.mkdir(parents=True, exist_ok=True)
    try:
        with pytest.raises(PF.PreflightError, match="reuse existing"):
            PF.validate_output_directory(existing, project_root=PF.PROJECT_ROOT)
    finally:
        # The directory is empty and belongs to this test; remove only this
        # explicitly named temporary target so the repository stays clean.
        existing.rmdir()


def test_optin_cuda_unavailable_is_recorded_without_claiming_cuda_ready(monkeypatch, tmp_path):
    cpus = _four_current_cpus()

    def fake_probe(kind, **kwargs):
        if kind == "cpu":
            return {"ok": True, "probe_status": "passed", "fit_performed": False}
        return {
            "ok": True,
            "probe_passed": False,
            "probe_status": "unavailable",
            "cuda_available": False,
        }

    monkeypatch.setattr(PF, "_run_child_probe", fake_probe)
    output = PF.PROJECT_ROOT / "reports" / "compute_execution_design_20261002" / "test-optin-unavailable"
    if output.exists():
        pytest.skip("test output target already exists")
    manifest = PF.run_preflight(
        out_dir=output,
        probe_cuda="optin",
        cpu_ids=cpus,
        project_root=PF.PROJECT_ROOT,
    )
    assert manifest["status"] == "success"
    assert manifest["all_ok"] is True
    assert manifest["training_ready"] is False
    assert manifest["cuda_ready"] is False
    assert manifest["cuda_probe_status"] == "skipped_unavailable"
    assert json.loads((output / PF.MANIFEST_NAME).read_text(encoding="utf-8"))["cuda_ready"] is False
    assert (output / PF.STATUS_NAME).exists()
    assert (output / PF.LOG_NAME).exists()
    # Keep this focused test's run evidence separate and removable by the
    # caller's external pytest basetemp cleanup.
    for path in output.iterdir():
        path.unlink()
    output.rmdir()
