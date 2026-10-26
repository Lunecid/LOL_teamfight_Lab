"""Stage bodies for the supervisor tests (``c_ept/tests/test_run_supervisor.py``, S4b-prep K5b): short CPU Python functions that a child process of the
supervisor imports by name and calls as ``body(run_dir, *args)``. Every argument is text (it travels on the command line of the child).

Not a test module (no ``test_`` prefix, nothing here is collected). Standard library only: a child must start in a fraction of a second, load no
native library and never see a GPU (the supervisor gives the children ``CUDA_VISIBLE_DEVICES=''``).

해석: 감독자 시험이 자식 프로세스에서 실제로 돌리는 단계 본문들이다. 표준 라이브러리만 쓰고 인자는 모두 글자이다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _put(run_dir, relative, text) -> Path:
    path = Path(run_dir) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def _count(run_dir, name) -> None:
    """Append one line to ``<run_dir>/<name>``: how many times a stage body ran."""
    with (Path(run_dir) / name).open('a', encoding='utf-8') as stream:
        stream.write('x\n')


def make_output(run_dir, relative, text='output') -> None:
    """Write one output file and count the call in ``calls.txt``."""
    _count(run_dir, 'calls.txt')
    _put(run_dir, relative, text)


def make_two_outputs(run_dir, first, second) -> None:
    _count(run_dir, 'calls.txt')
    _put(run_dir, first, 'first')
    _put(run_dir, second, 'second')


def copy_input(run_dir, source, relative) -> None:
    """Read an earlier stage's output and write a file from it."""
    _count(run_dir, 'calls.txt')
    _put(run_dir, relative, 'derived from ' + Path(run_dir, source).read_text(encoding='utf-8'))


def no_output(run_dir) -> None:
    _count(run_dir, 'calls.txt')


def raise_runtime_error(run_dir, key) -> None:
    raise RuntimeError(f'boom while reading {key}')


def raise_key_error(run_dir, key) -> None:
    raise KeyError(key)


def raise_os_error(run_dir, path) -> None:
    raise PermissionError(13, 'Permission denied', path)


def raise_gate_refusal(run_dir) -> None:
    from c_ept.run import guards
    raise guards.GateRefused('spec', 'the stage refuses to start')


def print_text(run_dir, text) -> None:
    """Print ``text`` on stdout and on stderr (the supervisor captures both), then finish without an error."""
    print(text, flush=True)
    print(text, file=sys.stderr, flush=True)


def exit_with(run_dir, code) -> None:
    sys.exit(int(code))


def fail_once_then_work(run_dir, relative) -> None:
    """Fail on the first call (a marker file says so), work on the later ones."""
    marker = Path(run_dir) / 'failed_once.marker'
    if not marker.exists():
        marker.write_text('1', encoding='utf-8')
        raise RuntimeError('first call fails')
    make_output(run_dir, relative, 'second try')


def report_environment(run_dir, relative) -> None:
    """Write what the child sees: the device list, the thread counts, and which parent variables survived."""
    names = ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CODEX_TEST_MARK', 'PYTHONPATH', 'GIT_DIR',
             'KMP_DUPLICATE_LIB_OK', 'MKL_THREADING_LAYER')
    _put(run_dir, relative, json.dumps({name: os.environ.get(name) for name in names} | {
        'numpy_loaded': 'numpy' in sys.modules, 'isolated': sys.flags.isolated, 'utf8_mode': sys.flags.utf8_mode,
        'dont_write_bytecode': sys.dont_write_bytecode, 'write_through': sys.stdout.write_through, 'stdout_encoding': sys.stdout.encoding}))


def print_then_sleep(run_dir, text, ready) -> None:
    """Print ``text`` (no flush call), publish ``<run_dir>/<ready>``, and sleep: the text must reach the supervisor even when the tree is killed."""
    print(text)
    _put(run_dir, ready, '{}')
    time.sleep(120)


def sleep_until_file(run_dir, go, relative) -> None:
    """Wait (at most 60 s) for ``<run_dir>/<go>`` to appear, then write the output. Lets a test hold a run open."""
    _put(run_dir, 'started.marker', str(os.getpid()))
    deadline = time.monotonic() + 60
    while not (Path(run_dir) / go).exists():
        if time.monotonic() > deadline:
            raise TimeoutError('the test never released the stage')
        time.sleep(0.02)
    make_output(run_dir, relative, 'released')


def _spawn_sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(120)'], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def _publish_processes(run_dir, ready, grandchild) -> None:
    tmp = Path(run_dir) / (ready + '.tmp')
    tmp.write_text(json.dumps({'child': os.getpid(), 'grandchild': grandchild.pid}), encoding='utf-8')
    os.replace(tmp, Path(run_dir) / ready)


def spawn_and_sleep(run_dir, ready) -> None:
    """Start a grandchild that sleeps for two minutes, publish both process numbers in ``<run_dir>/<ready>``, and sleep too."""
    grandchild = _spawn_sleeper()
    _publish_processes(run_dir, ready, grandchild)
    time.sleep(120)


def spawn_and_return(run_dir, ready) -> None:
    """Start a grandchild that sleeps for two minutes, publish the process numbers, and finish normally (leaving the grandchild behind)."""
    grandchild = _spawn_sleeper()
    _publish_processes(run_dir, ready, grandchild)


def noisy(run_dir, kilobytes) -> None:
    """Print ``kilobytes`` KiB of text, then a last line that must be in the tail."""
    block = ('abcdefghij' * 100 + '\n')
    for _ in range(int(kilobytes)):
        sys.stdout.write(block)
    print('LAST LINE', flush=True)
