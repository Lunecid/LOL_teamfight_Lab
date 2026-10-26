"""Supervisor of the run kernel: gates, run lock, ledger, one Windows-Job-owned child per stage, exit codes, scrubbed errors (S4b-prep K5b, decision D11,
Q7).

``run(profile, run_dir, *, stages, ...) -> int`` starts a run (or resumes one) from a hash-locked profile and returns the exit code of the process. The
stage bodies are functions that the caller names (K6 supplies the real runners); the supervisor knows nothing of training. D11: "the supervisor and
ledger reuse the components proven in T009": the Windows Job (``run.job``, a copy of the T002 job module), the single-owner ledger (``run.ledger``) and the
strict IO (``run.io``).

Order of one call (nothing is reserved before step 5; a refusal in steps 1-4 leaves no run folder, no lock and no ledger)
-----------------------------------------------------------------------------------------------------------------------
    1. ``guards.native_policy()``, the very first call: the default E6 check imports ``runlog``, which imports numpy, and the native threading policy
       must be fixed before the first native import. Then the plan is checked (stage names, bodies, paths: ``ledger.check_paths``).
    2. ``guards.e6()`` (a model server runs => refusal), the child environment (``guards.child_environment`` with the runner's device list and thread
       count; ``git`` must be on its filtered PATH, because the children's ``identity.capture`` and ``start_gate`` call it), the decision record
       (``guards.spec_check`` on the bytes that are read once; for an existing run ``Ledger.check_spec_file``: constants and growth rule), the budget (the
       wall budget must cover the training cap plus the finalisation margin; one ``Budget.check``).
    3. A new run: the identity is captured and ``identity.start_gate`` is passed, and the header is built (nothing is written).
    4. (an existing run is opened read-only and bound to this profile and plan.)
    5. The run lock ``RUN.lock`` is taken (exclusive create); then ``Ledger.create`` (a new run) or ``Ledger(run_dir)`` + the spec check again + the
       identity captured again (it must equal the saved one) + ``start_gate`` + ``resume()`` (an existing run; K2 review N4).
    6. For every stage not finished: E6, ``identity.verify``, ``begin``, the child process in a Job (kill-on-close), ``finish``.
    7. A last ``resume()`` verifies everything the ledger recorded; then exit code 0.

Exit codes (Q7; the table is ``exit_code_for``)
-----------------------------------------------
    0   success: every stage finished and the final resume check passed.
    2   a refusal (the run did not start, or stopped, because a rule said no; nothing computed was lost; fix the cause and start again):
        ``GateRefused`` (native, e6, environment, spec, budget), ``IORefused`` and its subclasses ``ProfileRefused``, ``LedgerRefused``, ``Tampered``,
        ``SupervisorRefused``, ``RunLocked`` and ``StageRefused`` (a stage whose child exited with 2, e.g. the TEST gate of K6), ``IdentityChanged`` and
        ``IdentityUnconfirmed``, ``Paused`` (Git or a file could not be read now: resumable), a decision record that is unopenable or unreadable (a
        ``GateRefused('spec')``), and a budget that is already insufficient at the start (``GateRefused('budget')``).
        These types are tested BEFORE any generic ``RuntimeError``, because ``GateRefused`` is a ``RuntimeError``.
    3   a computation failure: ``StageFailed`` (a stage whose child exited with another code or raised), ``BudgetStop`` (the budget ran out DURING
        the run: the child tree is killed and the reason is recorded in the ledger), an ``OSError`` (while hashing, for example), a ``JobError`` and
        every other exception (``RuntimeError``, ``KeyError``, ...). ``KeyboardInterrupt`` is not caught.
    Every text that is printed or recorded (error line, traceback, child output, ledger note, stage log) goes through ``run.scrub.scrub_text`` first.

Symbols
-------
    R, P      the run folder and the profile (``P.kind``, ``P.sha256``)
    S_1..S_n  the stages: ``Stage(name, body, args, inputs, outputs)``; ``body`` is ``module:function``, called in the child as ``function(R, *args)``
    W, M      ``budget.wall_s`` and ``budget.memory_bytes``; ``C`` the training cap (``TRAINING_CAP_S`` = 24 h for a train profile, 0 for the others, or
              the caller's value), ``F`` the finalisation margin (``FINALISATION_MARGIN_S``, 1 h)
    k         the match keys the caller gives (``scrub_keys``)
    rc        the exit code of a child

Equations
---------
    start budget     accept  <=>  W >= C + F  and  Budget.check() passes                    (otherwise GateRefused('budget') = 2)
    stage budget     before ``begin`` of a stage with ``wall_cap_s`` = c:  accept  <=>  (W - e) >= c + F,  e the wall time the budget has used so far
                     (otherwise GateRefused('budget') = 2, reason ``refused`` in the ledger notes, before any work of that stage)
    best_stop first  the training cap acts before the supervisor's budget ONLY if the budget that is left holds it: that is what the start rule gives
                     for a plan whose other work fits in F, and what the stage rule gives for every stage that declares ``wall_cap_s``. Time used by
                     earlier stages is not covered by the start rule; declare ``wall_cap_s`` on the training stage (K6 passes the training cap).
    stage result     rc = 0 => finish(S_i, outputs);  rc = 2 => StageRefused (2);  rc other => StageFailed (3)
    mid-run budget   Budget.check() raising during the child's wait or during any hash of the ledger => BudgetStop (3), Job stopped, note kind ``stop``
    exit code        2 if the error is in {GateRefused, IORefused, IdentityChanged, Paused}, else 3

해석
----
감독자는 단계 본문을 모르고 이름으로 받은 함수를 자식 프로세스에서 돌릴 뿐이다. 예약(실행 폴더, 잠금, 원장)보다 먼저 모든 관문을 거치므로, 관문이 거부하면 흔적이 남지
않는다. 자식은 Windows Job이 소유한다. 자식은 stdin에서 한 글자를 받을 때까지 아무 일도 하지 않고, 감독자가 Job에 넣은 뒤에야 풀어 주므로 손자 프로세스도 모두 Job 안에 있다.
예산이 실행 중에 다하거나 감독자가 죽으면 트리 전체가 죽는다. 종료 코드는 둘로 가른다. 2는 "규칙이 거절했다"(고치고 다시 시작하면 되는 것)이고 3은 "계산이 끝나지 않았다"이다.
거부 유형을 일반 ``RuntimeError``보다 먼저 검사하는 이유는 ``GateRefused``가 ``RuntimeError``이기 때문이다. 실행 중 예산 초과는 3이고 이유를 원장에 남긴다. 학습 안의 벽시계
상한(M8 ``on_wall_cap='best_stop'``)이 감독자 예산보다 먼저 작동하려면 남은 예산이 학습 상한과 마무리 여유의 합 이상이어야 한다. 시작 때 한 번 확인하는 것만으로는
앞선 단계가 쓴 시간을 알 수 없으므로, 상한을 선언한 단계(``wall_cap_s``)는 ``begin`` 직전에 남은 예산을 다시 확인하고 모자라면 일을 시작하기 전에 2로 거부한다.
출력과 기록에 나가는 모든 글자는 먼저 경기 키를 가린다. 자식 출력의 꼬리가 잘렸으면 잘린 첫 줄을 버리고 가린다(키의 일부가 잘림 자리에 걸릴 수 있기 때문이다).
자식은 ``-I``(작업 폴더와 사용자 site를 ``sys.path``에 넣지 않음)로 시작하지만 ``-I``는 PYTHON* 환경 변수를 무시하므로, UTF-8 모드, 바이트코드 쓰기 금지, 버퍼 없는 출력은
``-X utf8 -B -u``로 직접 켠다.

Child interpreter flags (K5b review I1). The T002 child was ``python -B -m c_ept.workflows.training_operator ...`` with PYTHONUTF8=1, PYTHONUNBUFFERED=1 and
PYTHONIOENCODING=utf-8 in the environment, started from the source folder. The kernel child is ``python -I -X utf8 -B -u -c BOOTSTRAP``: the same UTF-8 mode, no
bytecode and unbuffered output as flags (the variables stay in the environment but ``-I`` ignores them). Remaining differences that could matter for K8: ``-I``
(no PYTHONPATH, no user site, no current folder on ``sys.path``; the repository is put first explicitly); the entry is ``-c`` and not ``-m`` (``__main__`` is not
the operator module, ``sys.argv`` differs); the kernel child sets no PYTHONIOENCODING effect (``-I`` ignores it; stdio encoding comes from UTF-8 mode alone, and
the stdio error handler of the two children is not compared here); the child is started by ``subprocess.Popen`` with ``CREATE_NO_WINDOW`` and not through the
PowerShell wrapper, which only forwards the same environment. None of these touches numerical code paths; K8 still compares the outputs of both stacks.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from c_ept.run import guards
from c_ept.run import identity as run_identity
from c_ept.run import io as run_io
from c_ept.run import job as run_job
from c_ept.run import ledger as run_ledger
from c_ept.run import scrub as run_scrub

EXIT_OK, EXIT_REFUSED, EXIT_FAILED = 0, 2, 3
TRAINING_CAP_S = 24 * 3600                                          # = ept.train.WALL_CAP_SECONDS (a test pins the equality); K6 passes the constant itself
FINALISATION_MARGIN_S = 3600.0                                      # what the wall budget must hold beyond the training cap (hashing, the final resume)
LOCK_FILE = 'RUN.lock'
LOG_FOLDER = 'logs'
DEFAULT_IDENTITY_MODULES = ('c_ept.run.supervisor',)
OUTPUT_TAIL_BYTES = 64 * 1024                                       # the end of a child's output that is kept (merged stdout and stderr)
DETAIL_CHARS = 2000                                                 # the longest reason text that goes into a ledger note
MEMORY_SAMPLE_INTERVAL_S = 0.25                                     # the memory of the process tree is sampled at most this often by the stop check (the clock every call)
_BODY = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*')

# The child's first lines: read the release byte (the cooperative gate of ``run.job``), then run ``child_main``. ``-I`` (isolated mode) keeps the
# current folder, PYTHONPATH and the user site out of ``sys.path``; the repository is put in front explicitly.
_BOOTSTRAP = ('import sys\n'
              'if sys.stdin.buffer.read(1) != b"G":\n'
              '    sys.exit(125)\n'
              'sys.path.insert(0, sys.argv[1])\n'
              'from c_ept.run import supervisor\n'
              'sys.exit(supervisor.child_main(*sys.argv[2:]))\n')


# ---- exceptions and the exit-code table -----------------------------------------------------------------------------------------------------
class SupervisorRefused(run_io.IORefused):
    """The supervisor refuses a call (a stage plan that is not acceptable, a run folder that changed while the run started, ...). Exit code 2."""


class RunLocked(SupervisorRefused):
    """Another call holds ``RUN.lock`` of the run folder, or a call that was cut left it behind. The message names no path. Exit code 2."""


class StageRefused(SupervisorRefused):
    """A stage's child exited with code 2: the stage refused (for example the TEST gate of K6). ``exit_code`` and ``tail`` (the scrubbed end of the child's
    output) are attributes. Exit code 2."""

    def __init__(self, message, exit_code=None, tail=''):
        super().__init__(message)
        self.exit_code, self.tail = exit_code, tail


class StageFailed(RuntimeError):
    """A stage's child exited with a code other than 0 and 2 (it raised, or was killed). Exit code 3."""

    def __init__(self, message, exit_code=None, tail=''):
        super().__init__(message)
        self.exit_code, self.tail = exit_code, tail


class BudgetStop(RuntimeError):
    """The supervisor's budget (wall clock or memory) ran out during the run. The child tree has been stopped. Exit code 3, reason ``budget`` in the ledger."""


class OwnedProcessError(RuntimeError):
    """The processes of a stage could not be stopped (the Job still holds a live process). Exit code 3; the cleanup is not certified."""


_REFUSALS = (guards.GateRefused, run_io.IORefused, run_identity.IdentityChanged, run_identity.Paused)


def exit_code_for(exc: BaseException) -> int:
    """The exit code of an error: 2 for a refusal (``_REFUSALS``, tested first because ``GateRefused`` is a ``RuntimeError``), else 3 (the table in the
    module text)."""
    return EXIT_REFUSED if isinstance(exc, _REFUSALS) else EXIT_FAILED


def describe(exc: BaseException, keys) -> str:
    """The text for an error, with every match key in ``keys`` removed: one line ``Type: message`` for a refusal and for a stage that failed (its message
    carries the scrubbed end of the child's output), the full traceback (with its chain) for every other error."""
    if isinstance(exc, (StageFailed, StageRefused)):
        text = f'{type(exc).__name__}: {exc}' + (f'\n{exc.tail}' if exc.tail else '')
    elif isinstance(exc, _REFUSALS):
        text = f'{type(exc).__name__}: {exc}'
    else:
        text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return run_scrub.scrub_text(text, keys)


def main_guard(function: Callable, *args, scrub_keys=(), stream=None, **kwargs) -> int:
    """Call ``function(*args, **kwargs)`` and return its exit code: the function's int result (None is 0), or, when it raises an ``Exception``, 2 or 3 by
    ``exit_code_for`` after ``describe`` (keys removed) was written to ``stream`` (default ``sys.stderr``). ``KeyboardInterrupt`` is not caught. The
    command line of K6 wraps its work in it, so that an OS error of ``profile.load`` (it carries the path of a referenced profile) is scrubbed too."""
    keys = list(scrub_keys)
    try:
        result = function(*args, **kwargs)
    except Exception as exc:                                        # noqa: BLE001 - the exit code is the answer
        print(describe(exc, keys), file=stream if stream is not None else sys.stderr, flush=True)
        return exit_code_for(exc)
    return EXIT_OK if result is None else result


# ---- a stage -------------------------------------------------------------------------------------------------------------------------------
def _body_spec(body) -> str:
    """``module:qualified.name`` of a stage body: a text already in that form, or a function that a child can import again (top level, not a lambda, not
    of ``__main__``)."""
    if isinstance(body, str):
        text = body
    elif callable(body) and isinstance(getattr(body, '__module__', None), str) and isinstance(getattr(body, '__qualname__', None), str):
        if '<' in body.__qualname__ or body.__module__ == '__main__':
            raise SupervisorRefused('a stage body given as a function must be importable by name (not nested, not a lambda, not defined in __main__)')
        text = f'{body.__module__}:{body.__qualname__}'
    else:
        raise SupervisorRefused('a stage body is a "module:function" text or a function that can be imported by name')
    if _BODY.fullmatch(text) is None:
        raise SupervisorRefused('a stage body must have the form module.path:function')
    return text


@dataclass(frozen=True)
class Stage:
    """One stage of a plan: ``name`` (the ledger's rule: a lowercase word), ``body`` (``'module:function'``, or such a function; normalised to the text),
    ``args`` (texts; the child calls ``function(run_dir, *args)``), ``inputs`` and ``outputs`` (the paths ``Ledger.begin`` and ``Ledger.finish`` get; their
    form is checked by ``run`` with ``ledger.check_paths`` before anything is reserved), ``wall_cap_s`` (None, or the wall-clock cap in seconds of the work
    inside this stage, for example the training cap: before ``begin`` the budget that is left must hold it plus the finalisation margin, else a budget
    refusal). ``SupervisorRefused`` for what is not acceptable."""

    name: str
    body: object
    args: tuple = ()
    inputs: tuple = ()
    outputs: tuple = ()
    wall_cap_s: object = None

    def __post_init__(self):
        if self.wall_cap_s is not None:
            _number(self.wall_cap_s, 'wall_cap_s')
        if not run_ledger.valid_stage_name(self.name):
            raise SupervisorRefused('a stage name must be a lowercase word of at most 48 letters, digits and underscores that starts with a letter')
        object.__setattr__(self, 'body', _body_spec(self.body))
        for field_name in ('args', 'inputs', 'outputs'):
            value = getattr(self, field_name)
            if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, '__iter__'):
                raise SupervisorRefused(f'stage {field_name} must be a list')
            object.__setattr__(self, field_name, tuple(value))
        if not all(isinstance(arg, str) and '\x00' not in arg for arg in self.args):
            raise SupervisorRefused('stage args must be texts without NUL characters')


# ---- the child process in a Job -------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OwnedResult:
    """``returncode`` of the child, ``output`` (the end of its merged stdout and stderr, at most ``tail_bytes`` bytes) and whether bytes were dropped."""

    returncode: int
    output: bytes
    truncated: bool


class _Tail:
    """Reads a pipe to its end and keeps the last ``limit`` bytes (a training log can be gigabytes)."""

    def __init__(self, limit: int):
        self.limit, self.buffer, self.dropped = limit, bytearray(), 0

    def drain(self, stream) -> None:
        try:
            while True:
                block = stream.read1(65536) if hasattr(stream, 'read1') else stream.read(65536)
                if not block:
                    return
                self.buffer += block
                if len(self.buffer) > 2 * self.limit:
                    cut = len(self.buffer) - self.limit
                    del self.buffer[:cut]
                    self.dropped += cut
        except (OSError, ValueError):                                # the pipe was closed under us (the tree was stopped)
            return

    def result(self) -> tuple[bytes, bool]:
        excess = max(0, len(self.buffer) - self.limit)
        return bytes(self.buffer[excess:]), (self.dropped + excess) > 0


def _terminate(job, child) -> None:
    """Best effort after an error: stop the Job (the child, its descendants), then the child itself; ``OwnedProcessError`` if a process is still alive."""
    stopped = False
    try:
        stopped = job.stop(timeout=10)
    except Exception:                                               # noqa: BLE001 - the job may be closed already; the child is stopped below
        pass
    if child is not None:
        try:
            child.kill()
        except OSError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            stopped = False
    if not stopped:
        raise OwnedProcessError('the processes of the stage could not be confirmed stopped; check the machine before starting again')


# Ref: tools/ept_comparison_controller.py ``run_owned`` (T009, line 111): one child per phase, assigned to a fresh Job before the release byte, polled until it
# exits, the tree stopped on every error. Differences, all deliberate (listed in the docstring): Q8 forbids the import of ``tools``, the signature is the
# kernel's, there is no ledger receipt (the supervisor writes a ledger note), no ``reserve_seconds`` and no memory sample from the Job.
def run_owned(argv, *, env, cwd, check=None, poll_s: float = 0.25, tail_bytes: int = OUTPUT_TAIL_BYTES) -> OwnedResult:
    """Run ``argv`` as a child that a Windows Job owns, and return its ``OwnedResult``.

    Differences from the T009 ``run_owned`` (``tools/ept_comparison_controller.py``): (1) it is not imported (Q8) but written again here, on the copy of the
    T002 Job module; (2) the memory is not sampled from the Job: the stop ``check`` the supervisor passes samples the process tree of the supervisor
    (``Budget``), so a grandchild whose parent exited has left the process tree, stays in the Job, and is NOT counted against the memory budget (a residual;
    the copied Job module can only query process counts, and it must stay identical to the T002 file); (3) there is no ``reserve_seconds`` cleanup
    reserve: the wall rule is the stage rule (``Stage.wall_cap_s`` plus the finalisation margin F), and stopping a Job takes seconds against F = 1 h; (4) no
    durable receipt: the supervisor records a stop in the ledger notes; (5) the output is read through a pipe and its tail kept, and when a stop ends the call
    the tail read so far is attached to the raised exception as ``child_output`` ``(bytes, truncated)``.

    The child is started with a pipe on stdin and must read one byte before it does anything (``run.job`` protocol; the supervisor's own child does, see
    ``child_main``). It is assigned to a fresh Job (kill-on-close) and only then released, so every descendant is in the Job. While it runs, ``check()``
    (if given) is called every ``poll_s`` seconds; what ``check`` raises (a budget, a stop request) ends the call: the Job is stopped, the child and all
    its descendants are dead, and the exception propagates. When the child exits by itself, whatever it left running is stopped too, so no process of a
    finished stage survives it. ``OwnedProcessError`` if the tree cannot be confirmed dead; ``run.job.JobError`` (an ``OSError``) if the Job cannot be
    made or the child cannot be assigned. ``TypeError`` / ``ValueError`` for an ``argv`` that is not a non-empty list of texts.

    The merged output is read by a thread; only its last ``tail_bytes`` bytes are kept.
    """
    if isinstance(argv, (str, bytes)) or not isinstance(argv, (list, tuple)):
        raise TypeError('argv must be a list of texts')
    if not argv:
        raise ValueError('argv must not be empty')
    if not all(isinstance(item, str) for item in argv):
        raise TypeError('argv must be a list of texts')
    creation = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    job = run_job.WindowsJob()
    child = reader = None
    tail = _Tail(tail_bytes)
    try:
        child = subprocess.Popen(list(argv), cwd=str(cwd), env=dict(env), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 creationflags=creation)
        job.assign(child)
        reader = threading.Thread(target=tail.drain, args=(child.stdout,), daemon=True)
        reader.start()
        job.release(child)
        while True:
            try:
                code = child.wait(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                if check is not None:
                    check()
        if not job.gone() and not job.stop(timeout=10):               # a descendant that outlived the child is stopped, never left behind
            raise OwnedProcessError('a process of the finished stage is still alive')
        reader.join(timeout=10)
        output, truncated = tail.result()
        return OwnedResult(returncode=code, output=output, truncated=truncated)
    except BaseException as original:
        try:
            _terminate(job, child)
        except OwnedProcessError as stop_failure:                   # K5b review m2: the original error stays the raised one; the failure is attached
            original.add_note(str(stop_failure))
        if reader is not None:
            reader.join(timeout=10)
        original.child_output = tail.result()                       # what the child wrote before it was stopped (it is unbuffered)
        raise
    finally:
        if child is not None and child.stdin is not None:
            try:
                child.stdin.close()                                   # EOF before the release byte makes a gated child exit without work
            except (OSError, ValueError):
                pass
        if reader is not None:
            reader.join(timeout=10)
        if child is not None and child.stdout is not None and (reader is None or not reader.is_alive()):
            try:
                child.stdout.close()                                  # not while the reader still holds it: close() would wait for the reader's read
            except (OSError, ValueError):
                pass
        job.close()


def _resolve_body(spec: str) -> Callable:
    module_name, _, qualname = spec.partition(':')
    target = importlib.import_module(module_name)
    for part in qualname.split('.'):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError('a stage body must be callable')
    return target


def child_main(body: str, run_dir: str, stage: str, *args: str) -> int:
    """The entry point of a stage's child process (called by the bootstrap after the release byte): fix the native policy, import the body and call
    ``body(run_dir, *args)``. Returns the child's exit code: 0, or ``exit_code_for`` of the error that the body raised (2 for a refusal, 3 for the rest),
    after the traceback was printed to stderr (the supervisor scrubs it)."""
    try:
        guards.native_policy()
        _resolve_body(body)(run_dir, *args)
    except Exception as exc:                                         # noqa: BLE001 - the exit code is the answer, the traceback goes to the supervisor
        print(f'stage {stage!r} stopped', file=sys.stderr)
        traceback.print_exc()
        return exit_code_for(exc)
    return EXIT_OK


# ---- the run lock -------------------------------------------------------------------------------------------------------------------------------
class _RunLock:
    """``<run>/RUN.lock``, created exclusively (the system lets one process win); removed on exit. A lock that a power cut or a killed process left behind
    is deleted by hand (the message says so). Ref: c_ept/workflows/run_s3.py ``_acquire_lock`` / ``_release_lock`` (``S3.lock``), whose rule this keeps;
    the message names no path (a path can hold a key)."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / LOCK_FILE

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise RunLocked('another call holds the run lock (RUN.lock in the run folder). Only one call may work on a run folder at a time. If none is '
                            'running (a power cut or a killed process leaves the lock behind), delete that file yourself and start again; this command '
                            'never deletes it') from None
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(f'pid={os.getpid()}\ntime={time.time():.0f}\n')
        return self

    def __exit__(self, *exc_info):
        try:
            self.path.unlink()
        except OSError as error:
            print(f'WARNING: the run lock could not be removed ({type(error).__name__}); delete it before the next call on this run', file=sys.stderr)


# ---- the gates and bindings of one call ---------------------------------------------------------------------------------------------------------
def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise SupervisorRefused(f'{name} must be a finite number of seconds, zero or more')
    return value


def training_cap(profile, training_cap_s=None) -> float:
    """The seconds of training that a run of this profile may need: ``TRAINING_CAP_S`` for a train profile, 0 for the others, or ``training_cap_s``."""
    if training_cap_s is not None:
        return _number(training_cap_s, 'training_cap_s')
    return float(TRAINING_CAP_S) if profile.kind == 'train' else 0.0


def required_wall_s(profile, training_cap_s=None, margin_s=FINALISATION_MARGIN_S) -> float:
    """``C + F``: the wall-clock budget that a run of ``profile`` needs at least (the training cap and the finalisation margin)."""
    return training_cap(profile, training_cap_s) + _number(margin_s, 'margin_s')


def _validate_plan(stages, run_dir: Path) -> list:
    if not isinstance(stages, (list, tuple)) or not stages or not all(isinstance(stage, Stage) for stage in stages):
        raise SupervisorRefused('stages must be a non-empty list of Stage objects')
    names = [stage.name for stage in stages]
    if len(set(names)) != len(names):
        raise SupervisorRefused('a stage name is used twice in the plan')
    for stage in stages:
        run_ledger.check_paths(run_dir, inputs=stage.inputs, outputs=stage.outputs)
        for item in (*stage.inputs, *stage.outputs):                # K5b review m4: the names the supervisor itself writes
            key = run_ledger.key_of(run_dir, item, external=True)
            if key is not None and (key.casefold() in (LOCK_FILE.casefold(), LOG_FOLDER) or key.casefold().startswith(LOG_FOLDER + '/')):
                raise SupervisorRefused(f'stage {stage.name!r}: a path names a file of the supervisor itself ({LOCK_FILE} or the {LOG_FOLDER} folder)')
    return list(stages)


def _read_spec(path) -> bytes:
    """The decision record, read once (at most ``MAX_SPEC_BYTES``); an unopenable or too large file is ``GateRefused('spec')`` without the path."""
    data = None
    try:
        with Path(path).open('rb') as stream:
            data = stream.read(run_ledger.MAX_SPEC_BYTES + 1)
    except OSError:
        data = None
    if data is None or len(data) > run_ledger.MAX_SPEC_BYTES:
        raise guards.GateRefused('spec', 'unreadable: the decision record cannot be opened or is too large')
    return data


def _exists(run_dir: Path) -> bool:
    return os.path.lexists(run_dir / run_ledger.HEADER_FILE) or os.path.lexists(run_dir / run_ledger.LEDGER_FILE)


def _bind(ledger, profile, plan: list) -> None:
    """An existing run belongs to its profile and its plan: the lock of the profile file is the same, and the stages in the ledger are the start of the
    plan (a finished or begun stage is never a different stage)."""
    if ledger.header.profile_sha256 != profile.sha256:
        raise run_ledger.LedgerRefused('the run was started with another profile; a run folder never mixes two profiles')
    done = [name for name, _ in ledger.stages]
    if [stage.name for stage in plan][:len(done)] != done:
        raise run_ledger.LedgerRefused('the stages in the ledger are not the start of this plan; a run folder never mixes two plans')


def _changed_paths(saved, now) -> list:
    return sorted(path for path in set(saved.files) | set(now.files) if saved.files.get(path) != now.files.get(path))


def scrub_tail(output: bytes, truncated: bool, keys) -> str:
    """The text of the end of a child's output with every key removed (K5b review m1). When bytes were dropped in front (``truncated``), the first line
    is dropped too (everything up to and including the first newline; all of it when there is no newline): the cut can fall inside a key, and a key
    fragment cannot be matched by ``scrub_text``."""
    if truncated:
        cut = output.find(b'\n')
        output = output[cut + 1:] if cut >= 0 else b''
    return run_scrub.scrub_text(output.decode('utf-8', errors='replace'), keys)


def _stop_check(budget: guards.Budget, interval_s: float = MEMORY_SAMPLE_INTERVAL_S) -> Callable[[], None]:
    """The ``check`` of the ledger and of ``run_owned``: the budget, with its refusal turned into ``BudgetStop`` (during the run it is a failure, 3).

    The clock is read at every call (a passed wall budget is found at once), the memory of the process tree only when ``interval_s`` seconds of the budget's
    clock have passed since the last sample (K5b review m3): the ledger calls the check for every MiB it hashes, and a psutil walk of the tree costs more
    than the hash of a MiB. The first call samples."""
    last = []

    def check() -> None:
        elapsed = budget.elapsed_s()
        if last and 0 <= elapsed - last[0] < interval_s and elapsed <= budget.wall_s:
            return
        try:
            budget.check()
            last[:] = [elapsed]
        except guards.GateRefused as refusal:
            if refusal.gate == 'budget':
                raise BudgetStop(str(refusal)) from None
            raise
    return check


def _bounded(text: str) -> str:
    return text if len(text) <= DETAIL_CHARS else '...' + text[-DETAIL_CHARS:]


def _write_log(run_dir: Path, stage: str, text: str) -> None:
    """``<run>/logs/<stage>.log``: the scrubbed end of the child's output (all or nothing). Not a ledger file: the ledger records no log."""
    folder = run_dir / LOG_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'{stage}.log'
    temporary = folder / f'{stage}.log.{os.getpid()}.tmp'
    try:
        temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


class _State:
    """What the error handler needs to know: the ledger (once it is open and owned), the stage in progress."""
    ledger = None
    stage = None


def _record_stop(state: _State, exc: BaseException, keys: list) -> None:
    """Write the stop reason of the stage in progress into the ledger notes (``kind='stop'``: reason ``budget``, ``refused`` or ``failure``, a bounded
    scrubbed detail and the child's exit code). Best effort: a ledger that cannot be written (it changed behind us) keeps its silence."""
    ledger = state.ledger
    if ledger is None:
        return
    reason = 'budget' if isinstance(exc, BudgetStop) else 'refused' if isinstance(exc, _REFUSALS) else 'failure'
    detail = exc.tail if isinstance(exc, (StageFailed, StageRefused)) and exc.tail else f'{type(exc).__name__}: {exc}'
    record = {'reason': reason, 'detail': _bounded(run_scrub.scrub_text(detail, keys))}
    if isinstance(exc, (StageFailed, StageRefused)) and exc.exit_code is not None:
        record['exit_code'] = exc.exit_code
    try:
        ledger.check = None                                         # the stop must be recorded even when the budget is what stopped us
        ledger.note(state.stage, 'stop', record)
    except Exception:                                               # noqa: BLE001 - the original error is what the caller gets
        pass


# ---- the call ------------------------------------------------------------------------------------------------------------------------------
def _default_repo() -> Path:
    return Path(__file__).resolve().parents[2]


def run(profile, run_dir, *, stages, spec_path, budget, cuda_visible_devices, threads, identity_modules=DEFAULT_IDENTITY_MODULES, repo=None, runner=None,
        scrub_keys=(), stream=None, environ=None, training_cap_s=None, margin_s=FINALISATION_MARGIN_S, poll_s=0.25, e6=None, capture=None, verify=None,
        start_gate=None) -> int:
    """Start or resume the run in ``run_dir`` for ``profile`` and return the exit code (0, 2 or 3; the module text has the table). Never raises an
    ``Exception``: an error is printed (keys removed) to ``stream`` (default ``sys.stderr``) and mapped.

    ``stages``: the plan (``Stage`` objects). ``spec_path``: the decision record. ``budget``: a ``guards.Budget`` (wall clock and memory). The runner
    decides the children's ``cuda_visible_devices`` (``''`` for a CPU or test child, ``'0'`` for the T002 training child) and ``threads`` (no default: a
    forgotten setting must not run a training on the CPU). ``identity_modules``: the entry modules of ``identity.capture`` (K6 names the runner modules);
    ``repo``: the repository root (default: the one that holds this package; it is the children's working folder). ``runner``: constants for the header
    (batch size, device, effective M fields; the key ``supervisor`` is the supervisor's own). ``scrub_keys``: the match keys to remove from everything
    printed or recorded. ``environ``: the environment the children's is made from (default ``os.environ``). ``training_cap_s`` / ``margin_s``: see
    ``required_wall_s``. ``poll_s``: how often the budget is checked while a child runs. ``e6``, ``capture``, ``verify``, ``start_gate``: replacements
    of ``guards.e6``, ``identity.capture``, ``identity.verify`` and ``identity.start_gate`` (tests; the default is the real function, looked up at each call).
    """
    keys = list(scrub_keys)
    state = _State()
    try:
        return _execute(state, profile, run_dir, stages, spec_path, budget, cuda_visible_devices, threads, identity_modules, repo, runner, keys, environ,
                        training_cap_s, margin_s, poll_s, e6, capture, verify, start_gate)
    except Exception as exc:                                        # noqa: BLE001 - every error becomes an exit code
        _record_stop(state, exc, keys)
        print(describe(exc, keys), file=stream if stream is not None else sys.stderr, flush=True)
        return exit_code_for(exc)


def _execute(state, profile, run_dir, stages, spec_path, budget, cuda_visible_devices, threads, identity_modules, repo, runner, keys, environ,
             training_cap_s, margin_s, poll_s, e6, capture, verify, start_gate) -> int:
    guards.native_policy()                                          # 1. before anything can import numpy (the default E6 probe does)
    if not isinstance(budget, guards.Budget):
        raise SupervisorRefused('budget must be a guards.Budget')
    run_dir = Path(os.path.abspath(run_dir))
    repo = Path(repo) if repo is not None else _default_repo()
    plan = _validate_plan(stages, run_dir)
    required = required_wall_s(profile, training_cap_s, margin_s)
    e6_check = e6 if e6 is not None else guards.e6
    start_e6 = e6_check()                                           # 2. a model server running refuses the start
    env = guards.child_environment(environ if environ is not None else os.environ, cuda_visible_devices=cuda_visible_devices, threads=threads)
    if shutil.which('git', path=env['PATH']) is None:
        raise guards.GateRefused('environment', 'git is not on the PATH that the children get (their identity check calls it)')
    existing = _exists(run_dir)
    header = probe = None
    if existing:
        probe = run_ledger.Ledger(run_dir)                          # read-only: no reservation
        _bind(probe, profile, plan)
        probe.check_spec_file(spec_path)
    else:
        data = _read_spec(spec_path)
        guards.spec_check(spec_path, expected_sha256=hashlib.sha256(data).hexdigest())
        fingerprint = run_ledger.spec_fingerprint(data)
    if budget.wall_s < required:
        raise guards.GateRefused('budget', f'the wall-clock budget of {budget.wall_s:g} s is smaller than the training cap plus the finalisation '
                                           f'margin ({required:g} s), so the run could be cut off by the budget before the training cap ends it')
    budget.check()
    if not existing:                                                # 3. a new run: the identity and the header, still without a write
        identity = (capture if capture is not None else run_identity.capture)(identity_modules, repo=repo)
        (start_gate if start_gate is not None else run_identity.start_gate)(repo, identity)
        constants = dict(runner) if isinstance(runner, Mapping) else {} if runner is None else None
        if constants is None:
            raise SupervisorRefused('runner must be a mapping')
        if 'supervisor' in constants:
            raise SupervisorRefused("the runner constants may not use the key 'supervisor': it is the supervisor's own")
        constants['supervisor'] = {'cuda_visible_devices': cuda_visible_devices, 'threads': threads, 'wall_budget_s': budget.wall_s,
                                   'memory_bytes': budget.memory_bytes, 'training_cap_s': training_cap(profile, training_cap_s),
                                   'finalisation_margin_s': margin_s, 'stages': [stage.name for stage in plan]}
        header = run_ledger.Header(identity=identity, profile_sha256=profile.sha256, runner=constants, **fingerprint)
        run_io.canonical_sha256(header.to_record())                 # strict JSON, found out before the run folder exists
    stop_check = _stop_check(budget)
    with _RunLock(run_dir):                                         # 5. the reservation: the lock first, then the ledger
        if _exists(run_dir) != existing:
            raise SupervisorRefused('the run folder changed while the run was starting; start again')
        if existing:
            ledger = run_ledger.Ledger(run_dir)
            if ledger.header != probe.header:
                raise SupervisorRefused('the header of the run changed while the run was starting; start again')
            ledger.check = stop_check
            ledger.check_spec_file(spec_path)
            saved = ledger.header.identity
            now = (capture if capture is not None else run_identity.capture)(identity_modules, repo=repo)     # K2 review N4: capture again, equal, ...
            if now != saved:
                raise run_identity.IdentityChanged('the HEAD commit differs from the identity of the run' if now.commit != saved.commit
                                                   else 'imported code differs from the identity of the run', _changed_paths(saved, now))
            (start_gate if start_gate is not None else run_identity.start_gate)(repo, saved)                  # ... then the start gate
            finished = ledger.resume()
            _bind(ledger, profile, plan)
        else:
            ledger = run_ledger.Ledger.create(run_dir, header)
            ledger.check = stop_check
            saved, finished = header.identity, []
        state.ledger = ledger
        ledger.note(None, 'e6', start_e6)
        for stage in plan:                                          # 6.
            if stage.name in finished:
                continue
            state.stage = stage.name
            ledger.note(stage.name, 'e6', e6_check())
            (verify if verify is not None else run_identity.verify)(saved, repo=repo)
            if stage.wall_cap_s is not None:                        # K5b review I2: the budget that is left must hold this stage's cap and the margin
                left = budget.wall_s - budget.elapsed_s()
                if left < stage.wall_cap_s + margin_s:
                    raise guards.GateRefused('budget', f'stage {stage.name!r}: {left:.0f} s of the wall budget are left, less than its cap of '
                                                       f'{stage.wall_cap_s:g} s plus the finalisation margin of {margin_s:g} s')
            ledger.begin(stage.name, inputs=stage.inputs)
            # -I keeps the working folder, PYTHONPATH and the user site out of sys.path but also makes the interpreter ignore every PYTHON* variable of the
            # child settings, so UTF-8 mode, no bytecode and unbuffered output are asked for by flags (K5b review I1).
            argv = [sys.executable, '-I', '-X', 'utf8', '-B', '-u', '-c', _BOOTSTRAP, str(repo), stage.body, str(run_dir), stage.name, *stage.args]
            try:
                result = run_owned(argv, env=env, cwd=repo, check=stop_check, poll_s=poll_s)
            except BaseException as stopped:
                partial = getattr(stopped, 'child_output', None)
                if partial is not None:                             # a stopped child's last output is kept in the log (scrubbed)
                    _write_log(run_dir, stage.name, scrub_tail(*partial, keys))
                raise
            tail = scrub_tail(result.output, result.truncated, keys)
            _write_log(run_dir, stage.name, tail)
            if result.returncode == EXIT_REFUSED:
                raise StageRefused(f'stage {stage.name!r} refused (exit code 2)', EXIT_REFUSED, _bounded(tail))
            if result.returncode != EXIT_OK:
                raise StageFailed(f'stage {stage.name!r} failed (exit code {result.returncode})', result.returncode, _bounded(tail))
            try:
                ledger.finish(stage.name, stage.outputs)
            except run_ledger.Tampered:
                raise
            except run_ledger.LedgerRefused as refused:             # the stage ran but did not leave what it declared: a failure of the computation
                raise StageFailed(f'stage {stage.name!r} finished without its declared outputs ({refused})', 0, '') from None
            state.stage = None
        done = ledger.resume()                                      # 7. everything the ledger recorded, verified once more
        if done != [stage.name for stage in plan]:
            raise run_ledger.LedgerRefused('the ledger does not hold every stage of the plan as finished')
    return EXIT_OK
