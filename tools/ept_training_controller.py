"""One explicit EPT-T002 Start. Default Check never creates a control directory.

The wrapper runs in the foreground of this supervisor, with its GUI hidden.
No retry, resume, folder discovery, service operation or global environment edit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

try:
    from . import ept_training_monitor as monitor
    from .ept_training_job import WindowsJob
except ImportError:
    import ept_training_monitor as monitor
    from ept_training_job import WindowsJob

WALL_SECONDS = 24 * 3600
POLL_SECONDS = 2
PYTHON = Path(r'C:\Users\todtj\anaconda3\python.exe')
POWERSHELL = Path(r'C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe')
CONTROLLER_SCHEMA = 'ept-training-controller-v1'


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def child_environment(original=None):
    env = dict(os.environ if original is None else original)
    for key in list(env):
        if key.upper().startswith(('CODEX_', 'OPENAI_', 'CONDA_', 'GIT_')) or key.upper() in ('PYTHONPATH', 'PYTHONHOME'):
            del env[key]
    env['PATH'] = os.pathsep.join(p for p in env.get('PATH', '').split(os.pathsep)
                                 if 'codex' not in p.casefold() and not p.casefold().startswith(str(PYTHON.parent).casefold()))
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               MKL_THREADING_LAYER='SEQUENTIAL', KMP_DUPLICATE_LIB_OK='FALSE',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='0')
    return env


def hidden_options():
    return {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}


def wrapper_command(source, config, control, action):
    return [str(POWERSHELL), '-NoLogo', '-NoProfile', '-NonInteractive', '-File',
            str(source / 'run_ept_training.ps1'), '-Action', action,
            '-Config', str(config), '-Control', str(control)]


def source_check(source, reviewed, env):
    monitor.hex_field(reviewed, 40)
    for args in (['rev-parse', 'HEAD'], ['status', '--porcelain', '--untracked-files=all']):
        result = subprocess.run(['git', '-C', str(source), *args], env=env, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, encoding='utf-8', timeout=30, **hidden_options())
        monitor.require(result.returncode == 0, 'reviewed source Git query failed')
        if args[0] == 'rev-parse':
            monitor.require(result.stdout.strip() == reviewed, 'reviewed code commit mismatch')
        else:
            monitor.require(not result.stdout.strip(), 'reviewed source whole tree is dirty')


def preflight(source, config_path, control, reviewed, *, check_wrapper=True):
    config = monitor.load_config(config_path)
    monitor.require(Path(__file__).resolve() == source / 'tools/ept_training_controller.py'
                    and Path(monitor.__file__).resolve() == source / 'tools/ept_training_monitor.py'
                    and Path(sys.modules[WindowsJob.__module__].__file__).resolve() == source / 'tools/ept_training_job.py',
                    'controller/monitor/job must run from the explicit reviewed source')
    monitor.require(Path(config['python']) == PYTHON.resolve() == Path(sys.executable).resolve(), 'fixed Anaconda interpreter required')
    runs = monitor.canonical(config['runs_root'])
    monitor.require(control.is_relative_to(runs) and control != runs and not control.is_relative_to(source)
                    and not config_path.is_relative_to(source) and config_path.is_relative_to(control), 'control/config external path binding mismatch')
    for name in ('run_ept_training.ps1', 'tools/ept_training_controller.py', 'tools/ept_training_monitor.py',
                 'tools/ept_training_job.py', 'c_ept/workflows/training_operator.py', 'c_ept/workflows/run_training.py'):
        monitor.require((source / name).is_file() and (source / name).resolve().is_relative_to(source), 'fixed script missing/escaped: ' + name)
    env = child_environment()
    monitor.require(shutil.which('git', path=env['PATH']) is not None, 'git unavailable in sanitized child PATH')
    source_check(source, reviewed, env)
    if check_wrapper:
        result = subprocess.run(wrapper_command(source, config_path, control, 'Check'), cwd=source, env=env,
                                stdin=subprocess.DEVNULL, capture_output=True, timeout=180, **hidden_options())
        monitor.require(result.returncode == 0, 'operator metadata Check refused: ' + result.stderr.decode('utf-8', errors='replace')[-1500:])
    return config, env


def no_previous_attempt(control):
    for name in ('launch.once', 'operator_start.once', 'run.json', 'controller.json'):
        monitor.require(not (control / name).exists(), 'attempt/ref already exists; no automatic retry: ' + name)


def take_launch_token(control, binding):
    control.mkdir(parents=True, exist_ok=True)
    # O_EXCL is the cross-process one-launch gate. Never remove this token.
    fd = os.open(control / 'launch.once', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        import json
        handle.write(json.dumps(binding, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


class ProcessTree:
    """Job owns launch descendants; PID samples are supplemental evidence only.

    The non-job path is defensive diagnostics for old/partial callers, never the
    production launcher. Unowned live/unconfirmed references prevent cleanup claims.
    """
    def __init__(self, pid, job=None):
        import psutil
        self.psutil = psutil
        root = psutil.Process(pid)
        self.root = (pid, root.create_time())
        self.known = {self.root}
        self.referenced = set()
        self.job = job

    def refresh(self):
        if getattr(self, 'job', None) is not None:
            return  # containment survives the parent without descendant sampling
        for pid, created in list(self.known):
            if monitor.generation_state(pid, created) != 'alive':
                continue
            try:
                children = self.psutil.Process(pid).children(recursive=True)
                self.known.update((p.pid, p.create_time()) for p in children)
            except self.psutil.NoSuchProcess:
                continue
            except self.psutil.Error as exc:
                raise monitor.MetadataError('cannot establish owned process tree') from exc

    def bind_owner(self, process):
        pair = (process['pid'], process['create_time'])
        self.referenced.add(pair)  # retain even a refused reference as cleanup evidence
        self.refresh()
        generation = monitor.generation_state(*pair)
        if generation not in ('gone', 'pid_reused'):
            if self.job is not None:
                if self.job.contains(*pair):
                    self.known.add(pair)
                else:
                    # Exit may race the membership query; never adopt an unknown
                    # live/unconfirmed generation just because it came from a ref.
                    monitor.require(monitor.generation_state(*pair) in ('gone', 'pid_reused'),
                                    'RUN Python owner is not in the wrapper process tree/job generation')
            else:
                monitor.require(pair in self.known, 'RUN Python owner is not in the wrapper process tree')
        return pair

    def unowned_references_gone(self):
        return all(monitor.generation_state(*pair) in ('gone', 'pid_reused')
                   for pair in getattr(self, 'referenced', set()) - self.known)

    def stop(self):
        if getattr(self, 'job', None) is not None:
            return self.job.stop(timeout=10) and self.unowned_references_gone()
        self.refresh()
        # Kill only generations observed in this launch tree; PID reuse is excluded.
        for pid, created in sorted(self.known, key=lambda p: p == self.root):
            if monitor.generation_state(pid, created) == 'alive':
                try:
                    self.psutil.Process(pid).kill()
                except self.psutil.NoSuchProcess:
                    pass
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if all(monitor.generation_state(pid, created) in ('gone', 'pid_reused') for pid, created in self.known):
                return self.unowned_references_gone()
            time.sleep(.1)
        return False

    def gone(self):
        if getattr(self, 'job', None) is not None:
            return self.job.gone() and self.unowned_references_gone()
        self.refresh()
        return (all(monitor.generation_state(pid, created) in ('gone', 'pid_reused') for pid, created in self.known)
                and self.unowned_references_gone())


def read_bound_run(control, source, config, reviewed, previous=None):
    ref_path = control / 'run.json'
    if not ref_path.exists():
        monitor.require(previous is None, 'durable run ref disappeared')
        return None, None
    ref = monitor.read_json(ref_path)
    run = monitor.validate_ref(ref, config, source)
    monitor.require(ref['code_commit'] == reviewed and (previous is None or ref == previous), 'exact run ref drift')
    manifest = monitor.read_json(run / 'RUN.json')
    monitor.validate_run(manifest, run, source, config)
    monitor.require(all(ref[k] == manifest[k] for k in monitor.REF_FIELDS - {'run'}), 'RUN/ref binding mismatch')
    path = run / 'PROCESS.json'
    if not path.exists():
        return ref, None  # bootstrapping; cannot infer failure or start another child
    process = monitor.read_json(path)
    monitor.validate_process(process, run, manifest)
    return ref, process


def supervise(source, config_path, control, reviewed):
    began = time.monotonic()
    no_previous_attempt(control)
    config, env = preflight(source, config_path, control, reviewed)
    no_previous_attempt(control)
    monitor.require(time.monotonic() - began < WALL_SECONDS, 'whole-launch budget exhausted in preflight')
    binding = {'schema': CONTROLLER_SCHEMA, 'source': str(source), 'config_path': str(config_path),
               'control': str(control), 'code_commit': reviewed, 'operator_config_sha256': monitor.digest(config),
               'controller_script': str(Path(__file__).resolve()), 'wrapper_script': str(source / 'run_ept_training.ps1')}
    take_launch_token(control, binding)
    import psutil
    record = {**binding, 'state': 'launching', 'attempts': 1, 'wall_limit_seconds': WALL_SECONDS,
              'controller_pid': os.getpid(), 'controller_create_time': psutil.Process().create_time(),
              'wrapper_pid': None, 'wrapper_create_time': None, 'owner_pid': None, 'owner_create_time': None,
              'started_at': stamp(), 'updated_at': stamp(), 'exit_code': None, 'run_ref': None,
              'elapsed_seconds': time.monotonic() - began, 'error': None, 'tree_gone': None,
              'limits': 'whole launch <=24h; trainer sampled host private <40GiB; no retry; Strata OFF',
              'api_local_tokens': None, 'cost': None}
    monitor.atomic_json(control / 'controller.json', record)
    child, tree, job, ref, process = None, None, None, None, None
    exit_code, error = None, None
    try:
        with (control / 'operator.stdout').open('xb') as stdout, (control / 'operator.stderr').open('xb') as stderr:
            # The cooperative gate prevents Python before exact launch-handle assignment.
            monitor.require(monitor.load_config(config_path) == config, 'config changed after Check')
            source_check(source, reviewed, env)
            job = WindowsJob()
            record['job_ownership'] = {'kind': 'windows_job_kill_on_close', 'assigned': False, 'released': False,
                                       'binding': 'launch-owned Popen handle; wrapper awaits G before Python'}
            child = subprocess.Popen(wrapper_command(source, config_path, control, 'Start') + ['-AwaitJobBinding'], cwd=source,
                                     env=env, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, **hidden_options())
            record['wrapper_pid'] = child.pid
            job.assign(child)
            record['job_ownership']['assigned'] = True
            tree = ProcessTree(child.pid, job=job)
            record.update(state='running', wrapper_create_time=tree.root[1])
            monitor.atomic_json(control / 'controller.json', record)
            job.release(child)
            record['job_ownership']['released'] = True
            monitor.atomic_json(control / 'controller.json', record)
            while True:
                tree.refresh()
                ref, process = read_bound_run(control, source, config, reviewed, ref)
                if ref is not None:
                    record['run_ref'] = ref
                if process is not None:
                    record['process_snapshot'] = process
                    owner = tree.bind_owner(process)
                    if record['owner_pid'] is not None:
                        monitor.require(owner == (record['owner_pid'], record['owner_create_time']), 'Python owner PID generation drift')
                    record.update(owner_pid=owner[0], owner_create_time=owner[1], process_snapshot=process)
                record.update(elapsed_seconds=time.monotonic() - began, updated_at=stamp())
                monitor.atomic_json(control / 'controller.json', record)
                if record['elapsed_seconds'] >= WALL_SECONDS:
                    raise TimeoutError('whole-launch 24h wall cap reached; terminating owned tree')
                exit_code = child.poll()
                if exit_code is not None:
                    break
                time.sleep(min(POLL_SECONDS, max(.01, WALL_SECONDS - record['elapsed_seconds'])))
            # Fast-exiting children still receive one final exact-ref check.
            ref, process = read_bound_run(control, source, config, reviewed, ref)
            record.update(run_ref=ref, process_snapshot=process)
            if process is not None:
                owner = tree.bind_owner(process)
                monitor.require(record['owner_pid'] is None or owner == (record['owner_pid'], record['owner_create_time']), 'final Python owner PID generation drift')
                record.update(owner_pid=owner[0], owner_create_time=owner[1])
            monitor.require(exit_code == 0, 'wrapper failed; no retry')
            monitor.require(ref is not None and process is not None, 'wrapper exited without durable RUN/ref/PROCESS')
            monitor.require(process['state'] == 'finished' and process['exit_code'] == 0, 'wrapper success lacks terminal successful Python record')
            monitor.require(monitor.generation_state(process['pid'], process['create_time']) in ('gone', 'pid_reused') and tree.gone(), 'wrapper exited but Python owner/tree remains alive')
            terminal = monitor.inspect_run(Path(ref['run']), source, config)
            monitor.require(terminal['state'] == 'completed', 'final metadata is not a verified completion')
            monitor.require(monitor.load_config(config_path) == config, 'operator config changed during launch')
            source_check(source, reviewed, env)
            monitor.require(time.monotonic() - began <= WALL_SECONDS, 'whole-launch wall cap reached during final verification')
            record['tree_gone'] = tree.gone()
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        if child is not None and getattr(child, 'stdin', None) is not None and not child.stdin.closed:
            # EOF before G exits the gate; it never starts Python.
            child.stdin.close()
        if tree is not None:
            try:
                record['tree_gone'] = tree.stop()
            except Exception as stop_exc:
                record['termination_error'] = f'{type(stop_exc).__name__}: {stop_exc}'
                record['tree_gone'] = False
        elif job is not None:
            try:
                stopped = job.stop(timeout=10)
                if child is not None and child.poll() is None:
                    # Only the retained Popen launch handle, never a PID/ref adoption.
                    child.terminate()
                record['tree_gone'] = stopped and (child is None or child.poll() is not None)
            except Exception as stop_exc:
                record.update(tree_gone=False, termination_error=f'{type(stop_exc).__name__}: {stop_exc}')
        if child is not None:
            try:
                exit_code = child.wait(timeout=0 if job is not None else 10)
            except subprocess.TimeoutExpired:
                pass
    finally:
        if tree is not None:
            record['owned_generations'] = [{'pid': pid, 'create_time': created} for pid, created in sorted(tree.known)]
        if process is not None:
            record['owner_generation_at_end'] = monitor.generation_state(process['pid'], process['create_time'])
            if record['owner_generation_at_end'] not in ('gone', 'pid_reused'):
                record['tree_gone'] = False
                record['manual_attention'] = 'RUN owner still alive/unconfirmed; cleanup not established; no restart'
        if tree is not None:
            record['unowned_run_references'] = [{'pid': pid, 'create_time': created}
                                               for pid, created in sorted(tree.referenced - tree.known)]
        if job is not None:
            try:
                record['job_gone_before_close'] = job.gone()
                if not record['job_gone_before_close']:
                    record['tree_gone'] = False
            except Exception as job_exc:
                record.update(tree_gone=False, job_probe_error=str(job_exc))
            finally:
                try:
                    job.close()
                    record['job_closed'] = True
                except Exception as close_exc:
                    record.update(job_closed=False, tree_gone=False,
                                  cleanup_error=f'{type(close_exc).__name__}: {close_exc}',
                                  manual_attention='Job handle close failed; cleanup unconfirmed; no restart')
        if record['tree_gone'] is False:
            record.setdefault('manual_attention', 'Owned-tree cleanup unconfirmed; preserve evidence and do not restart')
            if error is None:
                error = 'final job/owner cleanup not confirmed'
        record.update(state='completed' if error is None else 'failed', exit_code=exit_code,
                      error=error, elapsed_seconds=time.monotonic()-began, updated_at=stamp(), ended_at=stamp())
        monitor.atomic_json(control / 'controller.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('Check', 'Start', 'Status'), nargs='?', default='Check')
    parser.add_argument('--source', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--reviewed-code', required=True)
    args = parser.parse_args(argv)
    source, config, control = (monitor.canonical(getattr(args, key)) for key in ('source', 'config', 'control'))
    try:
        if args.action == 'Start':
            result = supervise(source, config, control, args.reviewed_code)
        elif args.action == 'Check':
            c, _ = preflight(source, config, control, args.reviewed_code)
            result = {'state': 'READY_WITHIN_SCOPE', 'source': str(source), 'config_path': str(config),
                      'control': str(control), 'code_commit': args.reviewed_code, 'operator_config_sha256': monitor.digest(c),
                      'attempt_available': not any((control / p).exists() for p in ('launch.once', 'operator_start.once', 'run.json', 'controller.json'))}
        else:
            result = monitor.read_json(control / 'controller.json')
            monitor.require(result.get('schema') == CONTROLLER_SCHEMA and result.get('source') == str(source)
                            and result.get('config_path') == str(config) and result.get('control') == str(control)
                            and result.get('code_commit') == args.reviewed_code
                            and result.get('operator_config_sha256') == monitor.digest(monitor.load_config(config)), 'controller status self-binding mismatch')
            result = dict(result)
            result['controller_generation_now'] = monitor.generation_state(result['controller_pid'], result['controller_create_time'])
            if result['state'] in ('launching', 'running') and result['controller_generation_now'] != 'alive':
                result['observation'] = 'controller interrupted/unconfirmed; no restart authorized'
        import json
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 1 if result.get('state') == 'failed' else 0
    except (monitor.MetadataError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
