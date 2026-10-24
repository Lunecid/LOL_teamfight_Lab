"""Small, import-light runtime profiles for the 2026-10-02 study.

This module deliberately contains only the runtime contract.  Numerical
libraries are imported by the fresh child processes used by ``preflight``.
"""
from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROFILE_SCHEMA = "training_runtime_profile_20261002_v1"
PROFILE_VERSION = "2026-10-02"
GIB = 1024 ** 3
HOST_MEMORY_LIMIT_BYTES = 24 * GIB
GPU_FREE_RESERVE_BYTES = GIB
GPU_MEMORY_FRACTION = 0.85

NUMERIC_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
CONTROLLED_ENV_VARS = NUMERIC_THREAD_ENV_VARS + (
    "OMP_MAX_ACTIVE_LEVELS",
    "OMP_NESTED",
    "CUBLAS_WORKSPACE_CONFIG",
    "NVIDIA_TF32_OVERRIDE",
)


def _distribution_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def version_metadata() -> Dict[str, Any]:
    """Return version-only metadata safe for the public profile JSON.

    In particular, this does not inspect usernames, account variables,
    hostnames, command lines, or filesystem paths.
    """

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": {
            name: version
            for name in ("numpy", "torch", "psutil", "pytest")
            if (version := _distribution_version(name)) is not None
        },
    }


def read_current_affinity() -> List[int]:
    """Read the current process affinity without changing machine settings."""

    try:
        import psutil

        values = psutil.Process().cpu_affinity()
    except Exception:
        # ``os.sched_getaffinity`` is available on the supported Unix runtimes
        # and is a useful read-only fallback when psutil is unavailable.
        getter = getattr(__import__("os"), "sched_getaffinity", None)
        if getter is None:
            count = __import__("os").cpu_count() or 1
            values = list(range(count))
        else:
            values = list(getter(0))
    return list(dict.fromkeys(int(value) for value in values))


def select_cpu_ids(
    requested_cpus: Optional[Iterable[int]] = None,
    *,
    current_cpus: Optional[Iterable[int]] = None,
) -> List[int]:
    """Select four IDs, validating an explicit list against caller affinity."""

    available = read_current_affinity() if current_cpus is None else list(
        dict.fromkeys(int(value) for value in current_cpus)
    )
    if requested_cpus is None:
        selected = available[:4]
    else:
        selected = list(dict.fromkeys(int(value) for value in requested_cpus))
        if len(selected) != 4:
            raise ValueError("--cpu-ids must contain four unique logical CPU IDs")
        missing = [value for value in selected if value not in available]
        if missing:
            raise ValueError(
                "--cpu-ids must be within the caller's current affinity; "
                f"outside={missing}"
            )
    return selected


def cpu_allocation(allowed_cpus: Optional[Iterable[int]] = None) -> Dict[str, Any]:
    """Describe the four-logical-CPU lease and its non-overlapping phases.

    The first four IDs returned by the operating system are retained in their
    returned order.  This records logical affinity only; it makes no physical
    core or NUMA claim.
    """

    available = read_current_affinity() if allowed_cpus is None else list(
        dict.fromkeys(int(value) for value in allowed_cpus)
    )
    selected = available[:4]
    result: Dict[str, Any] = {
        "allowed_logical_cpus": selected,
        "source": "psutil.currentaffinity_first4",
        "physical_core_claim": False,
        "budget_logical_cpus": 4,
        "feasible": len(selected) == 4,
        "reason": None,
        "extraction": {
            "cpus": selected[:3],
            "workers": 3,
            "threads_per_worker": 1,
            "exclusive_with": ["cpu_fit", "gpu_host"],
        },
        "cpu_fit": {
            "cpus": selected[:3],
            "threads": 3,
            "library_profile": {"name": "linear", "n_jobs": 1},
            "fit_backend": {"name": "lightgbm", "lgbm_num_threads": 3},
        },
        "gpu_host": {
            "cpus": selected[3:4],
            "threads": 1,
            "concurrent_with": ["cpu_fit"],
        },
        "supervisor": {
            "cpus": selected[3:4],
            "threads": 1,
            "shared": True,
            "shared_with": ["gpu_host"],
        },
        "phases_exclusive": ["extraction", "cpu_fit"],
    }
    if len(selected) != 4:
        result["reason"] = (
            "four logical CPUs are required; current affinity exposes "
            f"{len(selected)}"
        )
    return result


def controlled_environment(*, lease: str = "cpu_fit") -> Dict[str, str]:
    """Return child-process overrides without mutating the parent environment."""

    if lease not in {"cpu_extraction", "cpu_fit", "gpu_host"}:
        raise ValueError(f"unknown runtime lease: {lease}")
    # High-level libraries receive their explicit n_jobs/threads setting from
    # the profile. CPU fitting uses one high-level fit and three BLAS threads;
    # extraction and GPU hosting keep every numerical library at one thread.
    thread_count = "3" if lease == "cpu_fit" else "1"
    env = {name: thread_count for name in NUMERIC_THREAD_ENV_VARS}
    env.update(
        {
            "OMP_MAX_ACTIVE_LEVELS": "1",
            "OMP_NESTED": "FALSE",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NVIDIA_TF32_OVERRIDE": "0",
        }
    )
    return env


def compute_host_memory_cap(
    start_available_bytes: int,
    *,
    fraction: float = 0.60,
    limit_bytes: int = HOST_MEMORY_LIMIT_BYTES,
) -> Optional[int]:
    """Compute the host working-set cap, or ``None`` when infeasible."""

    available = int(start_available_bytes)
    limit = int(limit_bytes)
    if available <= 0 or limit <= 0 or not 0 < float(fraction) <= 1:
        return None
    cap = min(limit, int(available * float(fraction)))
    return cap if cap > 0 else None


def compute_gpu_allocator_cap(
    total_bytes: int,
    live_free_bytes: int,
    *,
    fraction: float = GPU_MEMORY_FRACTION,
    reserve_bytes: int = GPU_FREE_RESERVE_BYTES,
) -> Optional[int]:
    """Compute the process allocator cap from total and sampled live memory.

    ``None`` records an infeasible resource boundary.  The cap is an allocator
    policy only; external GPU users can still consume memory outside it.
    """

    total = int(total_bytes)
    live_free = int(live_free_bytes)
    reserve = int(reserve_bytes)
    if total <= 0 or live_free <= 0 or reserve < 0 or not 0 < float(fraction) <= 1:
        return None
    cap = min(int(total * float(fraction)), live_free - reserve)
    return cap if cap > 0 else None


def runtime_profile(*, allowed_cpus: Optional[Iterable[int]] = None) -> Dict[str, Any]:
    """Build the serializable runtime profile without importing NumPy/Torch."""

    return {
        "schema": PROFILE_SCHEMA,
        "profile_version": PROFILE_VERSION,
        "training_ready": False,
        "versions": version_metadata(),
        "cpu": cpu_allocation(allowed_cpus),
        "controlled_environment": controlled_environment(lease="cpu_fit"),
        "controlled_environment_by_lease": {
            lease: controlled_environment(lease=lease)
            for lease in ("cpu_extraction", "cpu_fit", "gpu_host")
        },
        "determinism": {
            "cuda_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "float32_api": "ieee_fp32",
            "amp": False,
            "tf32": False,
            "torch_compile": False,
        },
        "resource_policy": {
            "host_memory_fraction": 0.60,
            "host_memory_limit_bytes": HOST_MEMORY_LIMIT_BYTES,
            "gpu_allocator_fraction": GPU_MEMORY_FRACTION,
            "gpu_free_reserve_bytes": GPU_FREE_RESERVE_BYTES,
            "gpu_allocator_cap_applies_to_process_only": True,
            "external_gpu_use_not_limited_by_cap": True,
        },
    }


def profile_json(*, allowed_cpus: Optional[Iterable[int]] = None) -> str:
    """Return deterministic ASCII JSON for the profiles CLI."""

    import json

    return json.dumps(
        runtime_profile(allowed_cpus=allowed_cpus),
        ensure_ascii=True,
        sort_keys=True,
        indent=2,
    )


__all__ = [
    "CONTROLLED_ENV_VARS",
    "GIB",
    "PROFILE_SCHEMA",
    "PROFILE_VERSION",
    "compute_gpu_allocator_cap",
    "compute_host_memory_cap",
    "controlled_environment",
    "cpu_allocation",
    "profile_json",
    "read_current_affinity",
    "runtime_profile",
    "select_cpu_ids",
    "version_metadata",
]
