"""Read-only hardware preflight and fresh-process numerical probes.

The parent process stays import-light.  NumPy and Torch are imported only in
the child selected by ``--child-probe`` after all numerical environment
overrides and the child affinity have been applied.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .profiles import (
    CONTROLLED_ENV_VARS,
    GIB,
    compute_gpu_allocator_cap,
    compute_host_memory_cap,
    controlled_environment,
    cpu_allocation,
    runtime_profile,
    select_cpu_ids,
)


PACKAGE_NAME = "studies.training_runtime_20261002"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_NAME = "manifest.json"
STATUS_NAME = "status.json"
ERROR_NAME = "error.json"
LOG_NAME = "run.log"


class PreflightError(RuntimeError):
    """A validation or run-contract failure with a machine-readable reason."""


def utc_now() -> str:
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
    )


def _append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{utc_now()} {message}\n")
        handle.flush()


def validate_output_directory(
    out_dir: Path | str,
    *,
    project_root: Path | str = PROJECT_ROOT,
) -> Path:
    """Canonicalize and validate a new output directory inside the workspace."""

    root = Path(project_root).resolve()
    raw_candidate = Path(out_dir).expanduser()
    if raw_candidate.exists() or raw_candidate.is_symlink():
        raise PreflightError(
            f"refusing to reuse existing output directory: {raw_candidate.resolve(strict=False)}"
        )
    candidate = raw_candidate.resolve(strict=False)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise PreflightError(
            f"output directory must be inside project workspace: {candidate}"
        ) from exc
    if not relative.parts:
        raise PreflightError("output directory must be a new child of project workspace")
    # ``resolve`` catches ordinary symlink escapes.  ``is_symlink`` also
    # rejects a broken symlink whose target cannot be created safely.
    if candidate.exists() or candidate.is_symlink():
        raise PreflightError(f"refusing to reuse existing output directory: {candidate}")
    if candidate.parent.exists():
        parent = candidate.parent.resolve()
        try:
            parent.relative_to(root)
        except ValueError as exc:
            raise PreflightError(
                f"output parent must be inside project workspace: {parent}"
            ) from exc
    return candidate


def _child_environment(lease: str) -> Dict[str, str]:
    """Copy and override the child environment without mutating ``os.environ``."""

    environment = os.environ.copy()
    environment.update(controlled_environment(lease=lease))
    return environment


def _parse_cpu_ids(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    if not value.strip():
        raise PreflightError("--cpu-ids cannot be empty")
    try:
        return [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise PreflightError("--cpu-ids must be comma-separated integers") from exc


def _set_child_affinity(cpu_ids: Sequence[int]) -> List[int]:
    requested = list(dict.fromkeys(int(value) for value in cpu_ids))
    if not requested:
        raise RuntimeError("child affinity is empty")
    try:
        import psutil

        process = psutil.Process()
        process.cpu_affinity(requested)
        actual = list(process.cpu_affinity())
    except Exception as psutil_error:
        setter = getattr(os, "sched_setaffinity", None)
        getter = getattr(os, "sched_getaffinity", None)
        if setter is None or getter is None:
            raise RuntimeError(f"cannot set child affinity: {psutil_error}") from psutil_error
        setter(0, set(requested))
        actual = list(getter(0))
    actual = list(dict.fromkeys(int(value) for value in actual))
    if set(actual) != set(requested):
        raise RuntimeError(f"child affinity mismatch: requested={requested} actual={actual}")
    return actual


def _pin_supervisor_affinity(cpu_ids: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Pin this supervisor process and return ``(previous, actual)`` affinity."""

    requested = list(dict.fromkeys(int(value) for value in cpu_ids))
    if len(requested) != 1:
        raise RuntimeError("supervisor affinity must contain exactly one logical CPU")
    try:
        import psutil

        process = psutil.Process()
        previous = list(process.cpu_affinity())
        process.cpu_affinity(requested)
        actual = list(process.cpu_affinity())
    except Exception as psutil_error:
        getter = getattr(os, "sched_getaffinity", None)
        setter = getattr(os, "sched_setaffinity", None)
        if getter is None or setter is None:
            raise RuntimeError(f"cannot pin supervisor affinity: {psutil_error}") from psutil_error
        previous = list(getter(0))
        setter(0, set(requested))
        actual = list(getter(0))
    actual = list(dict.fromkeys(int(value) for value in actual))
    if set(actual) != set(requested):
        try:
            _restore_supervisor_affinity(previous)
        except Exception:
            pass
        raise RuntimeError(f"supervisor affinity mismatch: requested={requested} actual={actual}")
    return previous, actual


def _restore_supervisor_affinity(previous: Sequence[int]) -> None:
    values = list(dict.fromkeys(int(value) for value in previous))
    if not values:
        raise RuntimeError("cannot restore empty supervisor affinity")
    try:
        import psutil

        psutil.Process().cpu_affinity(values)
    except Exception as psutil_error:
        setter = getattr(os, "sched_setaffinity", None)
        if setter is None:
            raise RuntimeError(f"cannot restore supervisor affinity: {psutil_error}") from psutil_error
        setter(0, set(values))


def _validate_threadpool_info(items: Sequence[Mapping[str, Any]]) -> Tuple[bool, Optional[str], List[int]]:
    """Require actual BLAS/OpenMP introspection with one-to-three threads."""

    observed: List[int] = []
    if not items:
        return False, "threadpoolctl returned no numerical threadpools", observed
    for item in items:
        value = item.get("num_threads")
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 3:
            return False, f"invalid observed thread count: {value!r}", observed
        observed.append(int(value))
    return True, None, observed


def _controlled_settings_snapshot() -> Dict[str, str]:
    return {name: os.environ.get(name, "") for name in CONTROLLED_ENV_VARS}


def _cpu_probe() -> Dict[str, Any]:
    """Run a fixed float64 matrix/metric calculation without fitting."""

    import numpy as np

    started = time.perf_counter()
    rng = np.random.default_rng(20261002)
    left = rng.normal(size=(16, 8)).astype(np.float64)
    right = rng.normal(size=(8, 4)).astype(np.float64)
    scores = left @ right
    metric = float(np.mean(np.square(scores, dtype=np.float64)))
    finite = bool(np.isfinite(scores).all() and np.isfinite(metric))
    threadpool_payload: Dict[str, Any]
    try:
        from threadpoolctl import threadpool_info

        observed_info = threadpool_info()
        threadpools_ok, threadpool_error, observed_threads = _validate_threadpool_info(observed_info)
        threadpool_payload = {
            "threadpools": [
                {
                    key: item.get(key)
                    for key in ("user_api", "internal_api", "num_threads", "prefix")
                    if key in item
                }
                for item in observed_info
            ],
            "threadpools_ok": threadpools_ok,
            "observed_thread_counts": observed_threads,
            "threadpool_introspection_error": threadpool_error,
        }
        finite = finite and threadpools_ok
    except Exception as exc:
        threadpool_payload = {
            "threadpools": [],
            "threadpools_ok": False,
            "observed_thread_counts": [],
            "threadpoolctl_error": f"{type(exc).__name__}: {exc}",
        }
        finite = False
    return {
        "probe": "cpu",
        "probe_status": "passed" if finite else "failed",
        "ok": finite,
        "numpy_version": str(np.__version__),
        "dtype": str(scores.dtype),
        "matrix_shape": [16, 8, 4],
        "metric": metric,
        "finite": finite,
        **threadpool_payload,
        "fit_performed": False,
        "optimizer_step": False,
        "training_operation": False,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _configure_torch_for_probe(torch: Any) -> None:
    """Apply strict deterministic IEEE-fp32 settings, failing on API errors."""

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261002)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"


def _cuda_memory_snapshot(torch: Any, device: Any) -> Dict[str, Optional[int]]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "live_free_bytes": int(free),
        "total_bytes": int(total),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _cuda_probe(*, allocator_cap_bytes: Optional[int] = None) -> Dict[str, Any]:
    """Run the tiny CUDA forward/backward probe without an optimizer step."""

    import torch

    started = time.perf_counter()
    report: Dict[str, Any] = {
        "probe": "cuda",
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "training_operation": False,
        "optimizer_step": False,
        "fit_performed": False,
        "amp": False,
        "tf32": False,
        "torch_compile": False,
    }
    if not bool(torch.cuda.is_available()):
        report.update(
            {
                "ok": True,
                "probe_passed": False,
                "probe_status": "unavailable",
                "cuda_available": False,
                "skip_reason": "torch.cuda.is_available() is false",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return report

    _configure_torch_for_probe(torch)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError(
            "Torch thread settings did not apply: "
            f"intra={torch.get_num_threads()} interop={torch.get_num_interop_threads()}"
        )
    if (
        torch.backends.fp32_precision != "ieee"
        or torch.backends.cuda.matmul.fp32_precision != "ieee"
        or torch.backends.cudnn.fp32_precision != "ieee"
    ):
        raise RuntimeError("Torch FP32 precision API did not resolve to ieee")
    if not torch.are_deterministic_algorithms_enabled() or torch.backends.cudnn.benchmark:
        raise RuntimeError("Torch deterministic settings did not apply")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    initial_memory = _cuda_memory_snapshot(torch, device)
    allocator_cap = allocator_cap_bytes
    if allocator_cap is None:
        allocator_cap = compute_gpu_allocator_cap(
            initial_memory["total_bytes"] or 0, initial_memory["live_free_bytes"] or 0
        )
    if allocator_cap is None:
        raise RuntimeError("GPU allocator cap is non-positive")
    total_memory = initial_memory["total_bytes"] or 0
    torch.cuda.set_per_process_memory_fraction(
        float(allocator_cap) / float(total_memory), device=device
    )
    torch.cuda.reset_peak_memory_stats(device)
    before = _cuda_memory_snapshot(torch, device)

    # Each graph has ten nodes and 90 directed non-self edges: four ally and
    # five enemy edges per destination, hence 40 ally and 50 enemy edges per
    # graph. The batched layout keeps every source in its destination graph.
    batches, nodes, features = 2, 10, 8
    total_nodes = batches * nodes
    local_destination = torch.arange(nodes, device=device, dtype=torch.long).repeat_interleave(nodes - 1)
    local_offsets = torch.arange(1, nodes, device=device, dtype=torch.long).repeat(nodes)
    local_source = torch.remainder(local_destination + local_offsets, nodes)
    batch_offsets = torch.arange(batches, device=device, dtype=torch.long) * nodes
    destination = local_destination.repeat(batches) + batch_offsets.repeat_interleave(nodes * (nodes - 1))
    source = local_source.repeat(batches) + batch_offsets.repeat_interleave(nodes * (nodes - 1))
    team = torch.arange(nodes, device=device, dtype=torch.long) // 5
    edge_ally = team[source.remainder(nodes)] == team[destination.remainder(nodes)]
    edge_self = source == destination
    cross_batch = torch.div(source, nodes, rounding_mode="floor") != torch.div(
        destination, nodes, rounding_mode="floor"
    )
    expected_edges = batches * nodes * (nodes - 1)
    ally_edges = int(edge_ally.sum().item())
    enemy_edges = int((~edge_ally).sum().item())
    if (
        int(destination.numel()) != expected_edges
        or ally_edges != batches * 40
        or enemy_edges != batches * 50
        or bool(edge_self.any())
        or bool(cross_batch.any())
    ):
        raise RuntimeError(
            "edge contract failed: "
            f"edges={destination.numel()} ally={ally_edges} enemy={enemy_edges} "
            f"self={bool(edge_self.any())} cross_batch={bool(cross_batch.any())}"
        )
    node_features = torch.randn(
        (total_nodes, features), device=device, dtype=torch.float32, requires_grad=True
    )
    gathered = node_features[source]
    grouped = gathered.reshape(batches, nodes, nodes - 1, features)
    ally_grouped = edge_ally.reshape(batches, nodes, nodes - 1, 1)
    ally_message = (grouped * ally_grouped).sum(dim=2)
    enemy_message = (grouped * (~ally_grouped)).sum(dim=2)
    messages = (ally_message + enemy_message).reshape(total_nodes, features)

    linear = torch.nn.Linear(features, features, device=device, dtype=torch.float32)
    activated = torch.nn.functional.gelu(linear(messages))
    mat_left = torch.randn((8, 8), device=device, dtype=torch.float32, requires_grad=True)
    mat_right = torch.randn((8, 8), device=device, dtype=torch.float32, requires_grad=True)
    matmul = mat_left @ mat_right
    loss = activated.square().mean() + matmul.square().mean()
    loss.backward()
    torch.cuda.synchronize(device)

    gradients = [
        node_features.grad,
        mat_left.grad,
        mat_right.grad,
        linear.weight.grad,
        linear.bias.grad,
    ]
    finite_gradients = bool(
        all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients)
    )
    finite_outputs = bool(torch.isfinite(loss).item() and torch.isfinite(activated).all().item())
    after = _cuda_memory_snapshot(torch, device)
    passed = finite_gradients and finite_outputs
    report.update(
        {
            "ok": passed,
            "probe_passed": passed,
            "probe_status": "passed" if passed else "failed",
            "cuda_available": True,
            "device_index": 0,
            "device_name": str(properties.name),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "torch_intraop_threads": int(torch.get_num_threads()),
            "torch_interop_threads": int(torch.get_num_interop_threads()),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "fp32_precision": {
                "generic": str(torch.backends.fp32_precision),
                "cuda_matmul": str(torch.backends.cuda.matmul.fp32_precision),
                "cudnn": str(torch.backends.cudnn.fp32_precision),
            },
            "batches": batches,
            "nodes": nodes,
            "features": features,
            "edges_per_graph": 90,
            "edge_count": expected_edges,
            "ally_edges_per_graph": 40,
            "enemy_edges_per_graph": 50,
            "self_edges": int(edge_self.sum().item()),
            "cross_batch_edges": int(cross_batch.sum().item()),
            "gather_sum_ally_enemy": True,
            "linear_gelu": True,
            "forward_backward": True,
            "gradient_count": len(gradients),
            "loss": float(loss.detach().cpu().item()),
            "finite_outputs": finite_outputs,
            "finite_gradients": finite_gradients,
            "memory_initial": initial_memory,
            "memory_before": before,
            "memory_after": after,
            "allocator_cap_bytes": allocator_cap,
            "allocator_fraction": float(allocator_cap) / float(total_memory),
            "allocator_cap_applies_to_process_only": True,
            "external_gpu_use_not_limited_by_cap": True,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return report


def _child_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="private fresh-process runtime probe")
    parser.add_argument("--child-probe", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--cpu-ids", required=True)
    parser.add_argument("--allocator-cap-bytes", type=int, default=None)
    args = parser.parse_args(argv)
    cpu_ids = [int(item) for item in args.cpu_ids.split(",") if item]
    started = time.perf_counter()
    try:
        # This is intentionally the first environment-sensitive action in the
        # child.  NumPy/Torch are imported only after it completes.
        actual_affinity = _set_child_affinity(cpu_ids)
        if args.child_probe == "cpu":
            report = _cpu_probe()
        else:
            report = _cuda_probe(allocator_cap_bytes=args.allocator_cap_bytes)
        report.update(
            {
                "actual_affinity": actual_affinity,
                "requested_affinity": cpu_ids,
                "controlled_environment": _controlled_settings_snapshot(),
                "child_elapsed_seconds": time.perf_counter() - started,
            }
        )
        print(json.dumps(report, ensure_ascii=True, sort_keys=True), flush=True)
        return 0 if bool(report.get("ok")) else 1
    except Exception as exc:  # deterministic CUDA failures remain errors
        report = {
            "probe": args.child_probe,
            "ok": False,
            "probe_passed": False,
            "probe_status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-4:],
            "requested_affinity": cpu_ids,
            "controlled_environment": _controlled_settings_snapshot(),
            "child_elapsed_seconds": time.perf_counter() - started,
        }
        print(json.dumps(report, ensure_ascii=True, sort_keys=True), flush=True)
        return 1


def _run_child_probe(
    kind: str,
    *,
    cpu_ids: Sequence[int],
    project_root: Path = PROJECT_ROOT,
    allocator_cap_bytes: Optional[int] = None,
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    """Run one probe in a fresh Python process and return its JSON report."""

    if kind not in {"cpu", "cuda"}:
        raise ValueError(f"unknown probe kind: {kind}")
    lease = "cpu_fit" if kind == "cpu" else "gpu_host"
    command = [
        sys.executable,
        "-m",
        f"{PACKAGE_NAME}.preflight",
        "--child-probe",
        kind,
        "--cpu-ids",
        ",".join(str(int(value)) for value in cpu_ids),
    ]
    if allocator_cap_bytes is not None:
        command.extend(["--allocator-cap-bytes", str(int(allocator_cap_bytes))])
    completed = subprocess.run(
        command,
        cwd=str(project_root),
        env=_child_environment(lease),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    stdout = completed.stdout.strip()
    try:
        report = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        report = {}
    if not isinstance(report, dict):
        report = {}
    if completed.returncode != 0:
        # A non-zero child exit is fail-closed even if stale or malformed
        # stdout claims success.
        report["ok"] = False
        report["probe_passed"] = False
        report["probe_status"] = "failed"
        report.setdefault("error", completed.stderr.strip() or "child probe failed")
    report["child_returncode"] = int(completed.returncode)
    report["child_stderr"] = completed.stderr
    if completed.stderr.strip():
        report["stderr_present"] = True
    return report


def _host_resources(project_root: Path) -> Dict[str, Any]:
    resources: Dict[str, Any] = {
        "host_memory": {
            "start_available_bytes": None,
            "cap_bytes": None,
            "feasible": False,
            "reason": "psutil.virtual_memory unavailable",
        },
        "disk_workspace_free_bytes": None,
        "resource_infeasible_reasons": [],
    }
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
        cap = compute_host_memory_cap(available)
        resources["host_memory"] = {
            "start_available_bytes": available,
            "cap_bytes": cap,
            "feasible": cap is not None,
            "reason": None if cap is not None else "host memory cap is non-positive",
        }
    except Exception as exc:
        resources["host_memory"]["reason"] = f"psutil error: {type(exc).__name__}: {exc}"
    try:
        free = int(shutil.disk_usage(project_root).free)
        resources["disk_workspace_free_bytes"] = free
        if free <= 0:
            resources["resource_infeasible_reasons"].append(
                "workspace disk free bytes are non-positive"
            )
    except Exception as exc:
        resources["resource_infeasible_reasons"].append(
            f"workspace disk usage unavailable: {type(exc).__name__}: {exc}"
        )
    if not resources["host_memory"]["feasible"]:
        resources["resource_infeasible_reasons"].append(resources["host_memory"]["reason"])
    resources["resource_feasible"] = not resources["resource_infeasible_reasons"]
    return resources


def _status_payload(*, status: str, stage: str, started_at: str, **extra: Any) -> Dict[str, Any]:
    return {
        "status": status,
        "stage": stage,
        "started_at_utc": started_at,
        "updated_at_utc": utc_now(),
        **extra,
    }


def run_preflight(
    *,
    out_dir: Path | str,
    probe_cuda: str = "off",
    cpu_ids: Optional[Iterable[int]] = None,
    project_root: Path | str = PROJECT_ROOT,
) -> Dict[str, Any]:
    """Create one immutable preflight run and return its manifest."""

    if probe_cuda not in {"off", "optin", "required"}:
        raise PreflightError("probe_cuda must be one of: off, optin, required")
    root = Path(project_root).resolve()
    output = validate_output_directory(out_dir, project_root=root)
    requested_ids = None if cpu_ids is None else list(cpu_ids)
    selected_ids = select_cpu_ids(requested_ids)
    profile = runtime_profile(allowed_cpus=selected_ids)
    if requested_ids is not None:
        profile["cpu"]["source"] = "explicit_cpu_ids_within_current_affinity"
    if not profile["cpu"]["feasible"]:
        raise PreflightError(profile["cpu"]["reason"])

    output.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    _append_log(output / LOG_NAME, "preflight started")
    _atomic_write_json(
        output / STATUS_NAME,
        _status_payload(status="running", stage="profile", started_at=started_at),
    )

    manifest: Dict[str, Any] = {
        "schema": "training_runtime_preflight_20261002_v1",
        "out_dir": str(output),
        "code": {
            "package": PACKAGE_NAME,
            "profile_version": profile["profile_version"],
            "preflight_schema": "training_runtime_preflight_20261002_v1",
        },
        "environment": {"versions": profile["versions"]},
        "status": "running",
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "profile": profile,
        "requested_cuda_probe": probe_cuda,
        "training_ready": False,
        "hardware_probe_only": True,
        "all_ok": False,
        "requested_probe_passed": False,
        "cuda_ready": False,
        "probes": {},
        "resources": {},
        "supervisor": {
            "requested_affinity": [selected_ids[3]],
            "actual_affinity": None,
            "restored_after_run": False,
        },
    }

    supervisor_previous_affinity: Optional[List[int]] = None
    try:
        resources = _host_resources(root)
        manifest["resources"] = resources
        _atomic_write_json(
            output / STATUS_NAME,
            _status_payload(status="running", stage="resource_gate", started_at=started_at),
        )
        if not bool(resources.get("resource_feasible")):
            reasons = resources.get("resource_infeasible_reasons") or [
                "host resource preflight is infeasible"
            ]
            raise PreflightError("host resource preflight failed: " + "; ".join(map(str, reasons)))
        _atomic_write_json(
            output / STATUS_NAME,
            _status_payload(status="running", stage="cpu_probe", started_at=started_at),
        )
        _append_log(output / LOG_NAME, "running CPU fresh-process probe")
        cpu_fit_cpus = profile["cpu"]["cpu_fit"]["cpus"]
        cpu_report = _run_child_probe("cpu", cpu_ids=cpu_fit_cpus, project_root=root)
        manifest["probes"]["cpu"] = cpu_report
        cpu_passed = bool(cpu_report.get("ok")) and cpu_report.get("probe_status") == "passed"
        if not cpu_passed:
            raise PreflightError(f"CPU probe failed: {cpu_report.get('error', cpu_report)}")

        # Pin the supervisor only after the CPU child has been spawned. A
        # subprocess inherits the supervisor mask, so this ordering lets the
        # CPU child use its first three IDs while the GPU child inherits the
        # final host ID.
        supervisor_previous_affinity, supervisor_actual = _pin_supervisor_affinity(
            profile["cpu"]["gpu_host"]["cpus"]
        )
        manifest["supervisor"]["actual_affinity"] = supervisor_actual

        cuda_report: Optional[Dict[str, Any]] = None
        if probe_cuda != "off":
            _atomic_write_json(
                output / STATUS_NAME,
                _status_payload(status="running", stage="cuda_probe", started_at=started_at),
            )
            _append_log(output / LOG_NAME, f"running CUDA fresh-process probe mode={probe_cuda}")
            cuda_report = _run_child_probe(
                "cuda", cpu_ids=profile["cpu"]["gpu_host"]["cpus"], project_root=root
            )
            manifest["probes"]["cuda"] = cuda_report
            if cuda_report.get("probe_status") == "unavailable":
                if probe_cuda == "required":
                    raise PreflightError("CUDA probe required but CUDA is unavailable")
                manifest["cuda_probe_status"] = "skipped_unavailable"
                manifest["requested_probe_passed"] = True
            elif not bool(cuda_report.get("ok")) or cuda_report.get("probe_status") != "passed":
                raise PreflightError(f"CUDA probe failed: {cuda_report.get('error', cuda_report)}")
            else:
                manifest["requested_probe_passed"] = True
                manifest["cuda_ready"] = True
        else:
            manifest["cuda_probe_status"] = "not_requested"
            manifest["requested_probe_passed"] = True

        if cuda_report and cuda_report.get("probe_status") == "passed":
            before = cuda_report.get("memory_initial") or cuda_report.get("memory_before") or {}
            cap = compute_gpu_allocator_cap(
                before.get("total_bytes", 0), before.get("live_free_bytes", 0)
            )
            manifest["resources"]["gpu"] = {
                "total_bytes": before.get("total_bytes"),
                "live_free_bytes": before.get("live_free_bytes"),
                "allocator_cap_bytes": cap,
                "allocator_cap_applies_to_process_only": True,
                "external_gpu_use_not_limited_by_cap": True,
                "feasible": cap is not None,
                "reason": None if cap is not None else "GPU allocator cap is non-positive",
            }
            if cap is None:
                manifest["resources"]["resource_infeasible_reasons"].append(
                    "GPU allocator cap is non-positive"
                )
                manifest["resources"]["resource_feasible"] = False
        manifest["all_ok"] = bool(
            cpu_passed
            and manifest["requested_probe_passed"]
            and manifest["resources"].get("resource_feasible")
        )
        manifest["status"] = "success" if manifest["all_ok"] else "failed"
    except Exception as exc:
        manifest["status"] = "failed"
        cuda_report = manifest.get("probes", {}).get("cuda")
        if isinstance(cuda_report, dict) and "GPU allocator cap" in str(cuda_report.get("error", "")):
            manifest["resources"]["gpu"] = {
                "feasible": False,
                "allocator_cap_bytes": None,
                "allocator_cap_applies_to_process_only": True,
                "external_gpu_use_not_limited_by_cap": True,
                "reason": str(cuda_report.get("error")),
            }
            manifest["resources"].setdefault("resource_infeasible_reasons", []).append(
                str(cuda_report.get("error"))
            )
            manifest["resources"]["resource_feasible"] = False
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        manifest["all_ok"] = False
        _append_log(output / LOG_NAME, f"preflight failed: {type(exc).__name__}: {exc}")
    finally:
        if supervisor_previous_affinity is not None:
            try:
                _restore_supervisor_affinity(supervisor_previous_affinity)
                manifest["supervisor"]["restored_after_run"] = True
            except Exception as restore_exc:
                manifest["status"] = "failed"
                manifest["all_ok"] = False
                manifest["restore_error"] = {
                    "type": type(restore_exc).__name__,
                    "message": str(restore_exc),
                }

    manifest["finished_at_utc"] = utc_now()
    _atomic_write_json(output / MANIFEST_NAME, manifest)
    _atomic_write_json(
        output / STATUS_NAME,
        _status_payload(
            status=manifest["status"],
            stage="complete" if manifest["status"] == "success" else "failed",
            started_at=started_at,
            all_ok=manifest["all_ok"],
            error=manifest.get("error"),
        ),
    )
    if manifest["status"] == "failed":
        _atomic_write_json(output / ERROR_NAME, manifest.get("error", {"message": "preflight failed"}))
    _append_log(output / LOG_NAME, f"preflight {manifest['status']}")
    return manifest


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="training runtime hardware preflight")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--probe-cuda", choices=("off", "optin", "required"), default="off")
    parser.add_argument("--cpu-ids", default=None, help="four comma-separated logical CPU IDs")
    args = parser.parse_args(argv)
    try:
        manifest = run_preflight(
            out_dir=args.out_dir,
            probe_cuda=args.probe_cuda,
            cpu_ids=_parse_cpu_ids(args.cpu_ids),
        )
    except Exception as exc:
        payload = {
            "ok": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True), flush=True)
        return 1
    payload = {
        "ok": bool(manifest.get("all_ok")),
        "status": manifest.get("status"),
        "out_dir": str(args.out_dir),
        "manifest": str(Path(args.out_dir) / MANIFEST_NAME),
        "training_ready": False,
        "cuda_ready": bool(manifest.get("cuda_ready")),
        "all_ok": bool(manifest.get("all_ok")),
    }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True), flush=True)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess
    if "--child-probe" in sys.argv:
        raise SystemExit(_child_main())
    raise SystemExit(_main())


__all__ = [
    "ERROR_NAME",
    "LOG_NAME",
    "MANIFEST_NAME",
    "PACKAGE_NAME",
    "PROJECT_ROOT",
    "PreflightError",
    "STATUS_NAME",
    "_child_environment",
    "_child_main",
    "_cuda_probe",
    "_main",
    "_run_child_probe",
    "run_preflight",
    "validate_output_directory",
]
