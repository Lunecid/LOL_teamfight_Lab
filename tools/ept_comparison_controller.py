"""Launch one dormant comparison worker into an owned Windows Job.

The caller supplies reviewed source/config/input checks and the literal Worker
command. This module neither authorizes a campaign nor imports scientific code.
Only launch handles and the new Job may be terminated; PID-global kills are absent.
"""
from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
import math
import os
from pathlib import Path
import subprocess
import time

from c_ept.comparison.infrastructure import PRIVATE_BYTES_LIMIT, CampaignBudget, CampaignError
from tools.ept_training_job import WindowsJob


def sample_job_private_bytes(job):
    """A point sample of private bytes of current Job members, using stdlib Win32.

    Membership is verified on the same handle used for memory sampling. Process
    exits can race the sample. This is not a simultaneous/continuous OS cap and
    does not establish the lifetime private-memory peak between samples.
    """
    if os.name != "nt":
        raise OSError("Windows Job private-memory sampling requires Windows")
    kernel = job.k
    capacity = max(16, job.active_count() + 8)
    for _ in range(3):
        class ProcessIds(C.Structure):
            _fields_ = [("assigned", W.DWORD), ("listed", W.DWORD),
                        ("pids", C.c_size_t * capacity)]
        info = ProcessIds()
        ok = kernel.QueryInformationJobObject(job._handle(), 3, C.byref(info), C.sizeof(info), None)
        if ok and info.listed <= capacity and info.assigned <= info.listed:
            break
        code = C.get_last_error()
        if not ok and code != 234:  # ERROR_MORE_DATA
            raise OSError(code, "QueryInformationJobObject(process ids)")
        capacity = max(capacity * 2, info.assigned + 8)
    else:
        raise OSError("Job process list remained unstable; memory sample unavailable")

    class Memory(C.Structure):
        _fields_ = [("cb", W.DWORD), ("PageFaultCount", W.DWORD)] + [
            (name, C.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
    psapi = C.WinDLL("psapi", use_last_error=True)
    memory_info = psapi.GetProcessMemoryInfo
    memory_info.argtypes = [W.HANDLE, C.c_void_p, W.DWORD]
    memory_info.restype = W.BOOL
    total = 0
    for pid in info.pids[:info.listed]:
        handle = kernel.OpenProcess(0x410, False, pid)  # query + VM_READ, never terminate rights
        if not handle:
            code = C.get_last_error()
            if code == 87:  # exited before opening
                continue
            raise OSError(code, "OpenProcess(private sample)")
        try:
            contained = W.BOOL()
            if not kernel.IsProcessInJob(handle, job._handle(), C.byref(contained)):
                raise OSError(C.get_last_error(), "IsProcessInJob(private sample)")
            if not contained.value:  # an exited PID was reused outside this Job
                continue
            memory = Memory()
            memory.cb = C.sizeof(memory)
            if not memory_info(handle, C.byref(memory), C.sizeof(memory)):
                raise OSError(C.get_last_error(), "GetProcessMemoryInfo(private sample)")
            total += int(memory.PrivateUsage)
        finally:
            if not kernel.CloseHandle(handle):
                raise OSError(C.get_last_error(), "CloseHandle(private sample)")
    return total


def _worker_command(command, phase, model):
    """Refuse shell commands and ungated/incorrect scientific entry points."""
    if not isinstance(command, (list, tuple)) or not command or any(type(arg) is not str for arg in command):
        raise ValueError("command must be a literal string argument sequence")
    command = list(command)
    prefix = ["C:/Users/todtj/anaconda3/python.exe", "-B", "-m", "c_ept.comparison.operator"]
    if [arg.replace("\\", "/").lower() for arg in command[:4]] != [arg.lower() for arg in prefix]:
        raise ValueError("fixed Python -B -m c_ept.comparison.operator required")
    options = command[4:]
    expected = {"--config", "--action", "--stage", "--await-job-binding"}
    if model is not None:
        expected.add("--model")
    parsed = {}
    while options:
        name = options.pop(0)
        if name not in expected or name in parsed:
            raise ValueError("unknown or duplicate worker argument")
        if name == "--await-job-binding":
            parsed[name] = True
        else:
            if not options:
                raise ValueError("missing worker argument value")
            parsed[name] = options.pop(0)
    if (set(parsed) != expected or parsed.get("--action") != "Worker"
            or parsed.get("--stage") != phase or parsed.get("--model") != model
            or not parsed.get("--config")):
        raise ValueError("gated Worker command must match phase/model and literal config")
    return command


def run_owned(command, *, cwd, budget: CampaignBudget, phase, model=None, check,
              env=None, poll_seconds=.25, cleanup_seconds=10., stdout=None, stderr=None,
              job_factory=WindowsJob, popen_factory=subprocess.Popen,
              memory_sampler=sample_job_private_bytes, sleep=time.sleep,
              before=None,accept=None):
    """Run one reserved phase; return a durable success/failure/cleanup receipt.

    ``check()`` is mandatory and must raise on reviewed source/config/input drift
    or absent future run authorization. It runs before launch, before G, at every
    sample, and after cleanup. Worker-side guards remain mandatory too. Worker
    stdin must wait for exactly G with --await-job-binding before runtime import.
    Dependencies are injectable solely for software tests; no fallback launcher.
    The parent owns the ledger; workers must not mutate it.
    """
    command = _worker_command(command, phase, model)
    if not callable(check):
        raise TypeError("a source/config/input/authorization check callback is required")
    for value, label in ((poll_seconds, "poll_seconds"), (cleanup_seconds, "cleanup_seconds")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be positive and finite")
    source = Path(cwd).resolve(strict=True)
    if not source.is_dir():
        raise ValueError("cwd must be the reviewed source directory")
    budget.begin(phase, model)
    began = budget.clock()
    receipt = {"phase": phase, "model": model, "state": "failed", "error": None,
               "returncode": None, "job_assigned": False, "released": False,
               "tree_gone": False, "job_gone": False, "job_closed": False,
               "cleanup_errors": [], "private_sample_count": 0,
               "peak_sampled_private_bytes": None, "private_limit_bytes": PRIVATE_BYTES_LIMIT,
               "memory_scope": "sampled current child-tree private; not a continuous OS limit"}
    child = job = None
    creation_attempted = False
    interrupt = None
    try:
        check()
        budget.check(reserve_seconds=cleanup_seconds)
        if before is not None:
            before()
            budget.check(reserve_seconds=cleanup_seconds)
        creation_attempted = True
        job = job_factory()
        options = {"cwd": str(source), "env": env, "stdin": subprocess.PIPE,
                   "stdout": stdout, "stderr": stderr, "shell": False, "close_fds": True}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        child = popen_factory(command, **options)
        receipt["launch_pid"] = child.pid  # evidence only, never used to kill/adopt
        job.assign(child)
        receipt["job_assigned"] = True
        check()
        budget.check(reserve_seconds=cleanup_seconds)
        job.release(child)  # existing WindowsJob writes G only after exact-handle assignment
        receipt["released"] = True
        while True:
            check()
            budget.check(reserve_seconds=cleanup_seconds)
            private = memory_sampler(job)
            if type(private) is not int or private < 0:
                raise CampaignError("child-tree private sample unavailable/invalid")
            receipt["private_sample_count"] += 1
            receipt["peak_sampled_private_bytes"] = max(receipt["peak_sampled_private_bytes"] or 0, private)
            if private >= PRIVATE_BYTES_LIMIT:
                raise CampaignError("sampled child-tree private reached 40 GiB")
            status = child.poll()
            if status is not None:
                receipt["returncode"] = status
                if status != 0:
                    raise CampaignError(f"worker exited with status {status}; no retry")
                break
            sleep(min(poll_seconds, max(0., budget.check(reserve_seconds=cleanup_seconds) - cleanup_seconds)))
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            interrupt = exc
    finally:
        cleanup_started = budget.clock()
        deadline = cleanup_started + cleanup_seconds

        def remaining():
            return max(0., deadline - budget.clock())

        def cleanup_error(exc):
            receipt["cleanup_errors"].append(f"{type(exc).__name__}: {exc}")

        # If assignment failed, EOF refuses runtime entry; kill only this Popen
        # handle if necessary. If assignment partially succeeded the Job is also
        # stopped below. No descendants could start before G.
        if child is not None and not receipt["released"]:
            try:
                if child.stdin is not None and not child.stdin.closed:
                    child.stdin.close()
                if child.poll() is None:
                    child.kill()
            except Exception as exc:
                cleanup_error(exc)
        if job is not None:
            try:
                if not job.gone():
                    if receipt["error"] is None:
                        receipt["error"] = "worker exited leaving active Job members"
                    job.stop(timeout=remaining())
            except Exception as exc:
                cleanup_error(exc)
        if child is not None:
            try:
                receipt["returncode"] = child.wait(timeout=remaining())
            except Exception as exc:
                cleanup_error(exc)
        if job is not None:
            try:
                receipt["job_gone"] = job.gone() is True
            except Exception as exc:
                cleanup_error(exc)
            try:
                job.close()
                receipt["job_closed"] = getattr(job, "handle", None) is None
                if not receipt["job_closed"]:
                    raise CampaignError("Job close returned with an owned handle still open")
            except Exception as exc:
                cleanup_error(exc)
        elif not creation_attempted:
            receipt["job_gone"] = receipt["job_closed"] = True  # no Job/child was created
        try:
            root_gone = child is None or child.poll() is not None
        except Exception as exc:
            root_gone = False
            cleanup_error(exc)
        receipt["tree_gone"] = receipt["job_gone"] and root_gone
        receipt["cleanup_seconds"] = budget.clock() - cleanup_started
        try:
            check()
        except BaseException as exc:
            if receipt["error"] is None:
                receipt["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                interrupt = exc
        clean = all(receipt[key] is True for key in ("tree_gone", "job_gone", "job_closed"))
        if not clean:
            receipt["manual_attention"] = "cleanup unconfirmed; preserve evidence; no resume/retry"
        try:
            budget.check()
            if receipt['error'] is None and not receipt['cleanup_errors'] and clean and accept is not None:
                accept()
                budget.check()
        except CampaignError as exc:
            if receipt["error"] is None:
                receipt["error"] = f"{type(exc).__name__}: {exc}"
        except BaseException as exc:
            if receipt['error'] is None:
                receipt['error']=f'{type(exc).__name__}: {exc}'
            if isinstance(exc,(KeyboardInterrupt,SystemExit)):
                interrupt=exc
        receipt["elapsed_seconds"] = budget.clock() - began
        success = receipt["error"] is None and not receipt["cleanup_errors"] and clean
        receipt["state"] = "completed" if success else "failed"
        if not budget.finish(success=success, receipt=receipt):
            receipt["state"] = "failed"
    if interrupt is not None:
        raise interrupt
    return receipt
