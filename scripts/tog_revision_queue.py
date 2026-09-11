"""Resource-aware, resumable job queue for the ToG revision experiments.

The revision needs ~20 long runs on one machine (one shared GPU, 16 threads, ~40 GB free RAM, a
nearly full D: drive), several of which depend on each other and some of which take days.  Running
them by hand invites two failures: oversubscribing the GPU or RAM so a run dies at hour nine, and
losing track of what finished when the session that launched it goes away.  This runner owns both.

Queue file (JSON list), one object per job - the run_specs the implementation agents return:

    {"name": "...", "command": "...", "env": "K=V K=V", "cwd": "...",
     "est_wall_min": 90, "cpu_threads": 8, "gpu_mem_gb": 6, "ram_gb": 20, "disk_gb": 3,
     "outputs": ["..."], "depends_on": ["other job name"]}

A job starts only when every dependency finished with exit code 0 and its resources fit: at most
one GPU job at a time (any job with gpu_mem_gb >= 0.5), the sum of running cpu_threads within
--cpu-budget, the sum of ram_gb within --ram-budget, and free space on the output drive at least
disk_gb plus --disk-margin.  State lives in --state-dir as <name>.log / .running / .done, so
restarting the runner resumes: finished jobs are skipped, and a job whose recorded process is gone
is retried once before it is marked failed.  Dependents of a failed job never start.

    python scripts/tog_revision_queue.py --queue Q.json --state-dir D:/.../queue_state          # run
    python scripts/tog_revision_queue.py --queue Q.json --state-dir D:/.../queue_state --status # table
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

POLL_S = 30


def pid_alive(pid: int) -> bool:
    """True while the process exists and has not exited (stdlib only; psutil is not in the project venv)."""
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def parse_env(text: str) -> dict:
    out = {}
    for tok in str(text or "").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def drive_of(job: dict) -> str:
    for p in list(job.get("outputs") or []) + [job.get("cwd") or "."]:
        drive = os.path.splitdrive(str(p))[0]
        if drive:
            return drive + "\\"
    return os.path.splitdrive(os.getcwd())[0] + "\\"


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def state_of(job: dict, state_dir: Path) -> str:
    done = read_json(state_dir / f"{job['name']}.done")
    if done is not None:
        return "done" if done.get("rc") == 0 else ("failed" if done.get("final") else "retry")
    running = read_json(state_dir / f"{job['name']}.running")
    if running is not None:
        return "running" if pid_alive(int(running.get("pid", -1))) else "orphaned"
    return "pending"


def status_table(jobs, state_dir: Path) -> str:
    rows = [f"{'job':44s} {'state':9s} {'est_min':>7s} {'cpu':>4s} {'gpu':>4s} {'ram':>4s}  deps"]
    states = {j["name"]: state_of(j, state_dir) for j in jobs}
    for j in jobs:
        st = states[j["name"]]
        if st in ("pending", "retry") and any(states.get(d) == "failed" for d in (j.get("depends_on") or [])):
            st = "blocked"
        rows.append(f"{j['name'][:44]:44s} {st:9s} {j.get('est_wall_min', 0):7.0f} "
                    f"{j.get('cpu_threads', 1):4.0f} {j.get('gpu_mem_gb', 0):4.1f} {j.get('ram_gb', 0):4.0f}  "
                    f"{','.join(j.get('depends_on') or [])}")
    return "\n".join(rows)


def launch(job: dict, state_dir: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(parse_env(job.get("env", "")))
    threads = str(int(job.get("cpu_threads", 1) or 1))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(k, threads)
    log = open(state_dir / f"{job['name']}.log", "a", encoding="utf-8")
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start: {job['command']}\n")
    log.flush()
    bash = shutil.which("bash") or "bash"
    proc = subprocess.Popen([bash, "-c", job["command"]], cwd=job.get("cwd") or None, env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    (state_dir / f"{job['name']}.running").write_text(
        json.dumps({"pid": proc.pid, "start": time.time()}), encoding="utf-8")
    return proc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--state-dir", type=Path, required=True)
    ap.add_argument("--cpu-budget", type=int, default=14)
    ap.add_argument("--ram-budget", type=float, default=36.0)
    ap.add_argument("--disk-margin", type=float, default=10.0, help="GB kept free on the output drive")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated job names to consider (default: all)")
    a = ap.parse_args()

    jobs = json.loads(a.queue.read_text(encoding="utf-8"))
    names = [j["name"] for j in jobs]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate job names in queue")
    unknown = {d for j in jobs for d in (j.get("depends_on") or [])} - set(names)
    if unknown:
        raise SystemExit(f"unknown dependencies: {sorted(unknown)}")
    if a.only:
        keep = {s.strip() for s in a.only.split(",") if s.strip()}
        jobs = [j for j in jobs if j["name"] in keep]
    a.state_dir.mkdir(parents=True, exist_ok=True)
    if a.status:
        print(status_table(jobs, a.state_dir))
        return 0

    live: dict[str, subprocess.Popen] = {}
    print(f"[queue] {len(jobs)} jobs, state {a.state_dir}", flush=True)
    while True:
        # reap finished processes
        for name, proc in list(live.items()):
            rc = proc.poll()
            if rc is None:
                continue
            prior = read_json(a.state_dir / f"{name}.done") or {}
            attempts = int(prior.get("attempts", 0)) + 1
            final = rc == 0 or attempts >= 2
            (a.state_dir / f"{name}.done").write_text(json.dumps(
                {"rc": rc, "end": time.time(), "attempts": attempts, "final": final}), encoding="utf-8")
            (a.state_dir / f"{name}.running").unlink(missing_ok=True)
            del live[name]
            print(f"[queue] {time.strftime('%H:%M:%S')} {name} exited rc={rc} attempt {attempts}", flush=True)

        states = {j["name"]: state_of(j, a.state_dir) for j in jobs}
        for j in jobs:  # a job launched by an earlier runner whose process vanished gets one retry
            if states[j["name"]] == "orphaned" and j["name"] not in live:
                (a.state_dir / f"{j['name']}.running").unlink(missing_ok=True)
                prior = read_json(a.state_dir / f"{j['name']}.done") or {}
                attempts = int(prior.get("attempts", 0)) + 1
                (a.state_dir / f"{j['name']}.done").write_text(json.dumps(
                    {"rc": -1, "end": time.time(), "attempts": attempts, "final": attempts >= 2,
                     "note": "process vanished"}), encoding="utf-8")
                states[j["name"]] = "failed" if attempts >= 2 else "retry"

        running = [j for j in jobs if j["name"] in live]
        cpu_used = sum(float(j.get("cpu_threads", 1) or 1) for j in running)
        ram_used = sum(float(j.get("ram_gb", 0) or 0) for j in running)
        gpu_busy = any(float(j.get("gpu_mem_gb", 0) or 0) >= 0.5 for j in running)

        for j in jobs:
            if states[j["name"]] not in ("pending", "retry") or j["name"] in live:
                continue
            deps = j.get("depends_on") or []
            if any(states.get(d) in ("failed",) for d in deps):
                continue
            if not all(states.get(d) == "done" for d in deps):
                continue
            needs_gpu = float(j.get("gpu_mem_gb", 0) or 0) >= 0.5
            cpu = float(j.get("cpu_threads", 1) or 1)
            ram = float(j.get("ram_gb", 0) or 0)
            if needs_gpu and gpu_busy:
                continue
            if running and (cpu_used + cpu > a.cpu_budget or ram_used + ram > a.ram_budget):
                continue
            free_gb = shutil.disk_usage(drive_of(j)).free / 1e9
            if free_gb < float(j.get("disk_gb", 0) or 0) + a.disk_margin:
                print(f"[queue] {j['name']} waits: {free_gb:.0f} GB free on {drive_of(j)}", flush=True)
                continue
            live[j["name"]] = launch(j, a.state_dir)
            running.append(j)
            cpu_used += cpu
            ram_used += ram
            gpu_busy = gpu_busy or needs_gpu
            print(f"[queue] {time.strftime('%H:%M:%S')} started {j['name']} (cpu {cpu:.0f}, ram {ram:.0f} GB, "
                  f"gpu {'yes' if needs_gpu else 'no'})", flush=True)

        states = {j["name"]: state_of(j, a.state_dir) for j in jobs}
        blocked = [j["name"] for j in jobs if states[j["name"]] in ("pending", "retry") and
                   any(states.get(d) == "failed" for d in (j.get("depends_on") or []))]
        open_jobs = [n for n, s in states.items() if s in ("pending", "retry", "running")
                     and n not in blocked]
        if not live and not open_jobs:
            print("[queue] nothing left to run\n" + status_table(jobs, a.state_dir), flush=True)
            return 0 if all(s == "done" for s in states.values()) else 1
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
