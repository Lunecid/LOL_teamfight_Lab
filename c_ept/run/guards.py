"""Gate set of the run kernel: native policy, E6, child environment, decision-record check and budget (S4b-prep K4, decision D11, review A5-F3).

The earlier stacks (S3, S4a, T002, T003, T008/T009; those modules are frozen and unchanged) each implement the gates that a run passes before
it starts, in their own way (A5-F3). The kernel has one function or class for each gate. Where the stacks differ, the T002 behaviour is the
default, because K8 requires the kernel's T002 profile to reproduce the T002 stack bit for bit. The module imports only the standard library;
the heavy parts (``c_ept.runlog`` for the process detection, ``c_ept.config`` for the decision record, ``c_ept.runtime.resources`` for the
memory sampler) are imported inside the gate that needs them, so that ``native_policy`` can run before numpy, torch or scipy are imported.

What each gate keeps from the earlier stacks:

* ``native_policy``: ``workflows.s4a_runtime.prepare_native_runtime`` (S4a, T002, T003 and the DEV evaluation call it before the native imports),
  unchanged in effect and in the record it returns. The comparison (T008/T009) workers never call it: they take the thread counts from
  their child environment and set ``torch.set_num_threads`` from their config.
* ``e6``: ``run_s1.require_no_strata`` (probe: ``runlog.strata_check``, the one detection rule), ``s4a_runtime.validate_e6_probe`` (the strict
  record check) and ``run_s4a._e6`` (any probe error is a refusal). S1 only warns when psutil is missing; T002 refuses because the record is then
  unavailable. Kept: the T002 rule, refuse when E6 cannot be confirmed. The comparison stack had no E6 check at all.
* ``child_environment``: ``tools/ept_training_controller.child_environment`` (T002: removes CODEX_, OPENAI_, CONDA_, GIT_, PYTHONPATH and
  PYTHONHOME, filters PATH, sets the child settings), ``comparison/operator.child_environment`` (T009: removes nothing, filters nothing, sets the
  settings and two more thread counts) and ``tools/ept_dev_evaluate.child_env`` (DEV evaluation: removes PYTHONPATH and PYTHONHOME, sets the
  settings). Kept: the T002 function, which is the stricter one and the one under which the T002 training child started.
* ``spec_check``: ``config.check_spec`` (S1-S3: the decision record must agree with the code constants; its hash is recorded). It reads through
  ``check_spec`` and not through ``run.io``: the one exception to "every kernel module reads through ``run.io``", because ``check_spec`` is the S1-S3
  rule and its lenient parser is kept, not reimplemented.
* ``Budget``: ``evaluation.runtime.Budget`` (wall time and sampled private memory, injected clock and reader) and the ``BudgetExceeded`` checks of
  ``comparison.infrastructure``. Changed: the bound is inclusive (``used == bound`` passes), the old checks stop at the bound.

Every refusal is a ``GateRefused`` with the name of the gate (``native``, ``e6``, ``environment``, ``spec``, ``budget``). K5 maps every
``GateRefused`` to exit code 2 (Q7). The messages name variables, processes and bounds; they never carry the value of an environment variable, a
command line, a match key, a record path or the text of a file. No gate writes a file, starts a process or stops one, so a refusal leaves nothing behind: K5 calls the
gates before it reserves the run folder, the lock or the ledger. How often E6 runs (start only, per stage, per epoch) is decided by K5 and K6;
``e6`` is a plain callable, so ``E6Guard(guards.e6)`` counts its calls.

Symbols
-------
    P, K     the removed name prefixes ``REMOVED_PREFIXES`` and the removed names ``REMOVED_KEYS`` of the child environment
    S        the child settings ``CHILD_SETTINGS``: the 11 variables that the T002 controller sets (threads 8, SEQUENTIAL, CUBLAS, device 0, ...)
    S'       S with the device list ``cuda_visible_devices`` and the thread count ``threads`` of the call (the defaults '0' and 8 give S)
    B, B'    the environment mapping that is given and the dictionary that ``child_environment`` returns; ``up(k)`` the upper-cased name k
    I        the folder of the running interpreter, case-folded: ``casefold(str(Path(sys.executable).resolve().parent))``
    keep(p)  whether the PATH entry p stays: ``'codex' not in casefold(p) and not casefold(p).startswith(I)``
    D        the decision record of ``spec_check``; ``h = SHA-256(bytes of D)``; ``h*`` the expected hash (or None)
    W, M     ``wall_s`` seconds and ``memory_bytes`` bytes of a ``Budget``
    t0, t    the clock reading when the budget was made and at a check; ``e = t - t0`` the elapsed seconds
    m        one memory sample in bytes (the private bytes of the process tree)
    N        the native modules ``NATIVE_MODULES`` (numpy, torch, scipy)

Equations
---------
    native_policy accepts  <=>  strip_casefold(KMP_DUPLICATE_LIB_OK) in OVERRIDE_OFF  and  not (N meets sys.modules  and  MKL_THREADING_LAYER != 'SEQUENTIAL')
    native_policy sets     :    KMP_DUPLICATE_LIB_OK := 'FALSE',  MKL_THREADING_LAYER := 'SEQUENTIAL'                  (a refusal sets nothing)
    child_environment(B)   =    { (k, F(k, v))  :  (k, v) in B,  not (up(k) starts with a prefix of P),  up(k) not in K,  up(k) not a name of S }  union  S'
                                F(k, v) = join(os.pathsep, [ p in split(v)  :  keep(p) ])  if up(k) = 'PATH', else v;  a missing PATH becomes ''
                                refused when two keys of B have one up(k), when the device list is not ([0-9]+(,[0-9]+)*)?, or when threads is not an int in 1..1024
    spec_check accepts     <=>  D is readable  and  its operating point equals the code constants G_ms, L_u, tau_lead_ms, has presence_gate = null
                                and holds the four size-class keys (the S1-S3 rule of ``check_spec``)  and  (h* is None  or  h = h*);    it returns h
    e6 accepts             <=>  the probe record is well formed (the fields of ``runlog.strata_check``, consistent counts)  and  running = False
    Budget.check accepts   <=>  0 <= e <= W  and  m is an int with 0 < m <= M              (inclusive bounds: ``e == W`` and ``m == M`` pass)
    peak                   :    peak_bytes = max of the samples read so far, including the sample that was refused; peak_upper = max of the upper bounds seen

해석
----
관문은 실행이 시작하기 전에 한 번씩 거치는 검사이고, 거부하면 아무것도 만들지 않는다. 옛 스택은 같은 검사를 스택마다 달리 구현했다(A5-F3). 커널은 T002 스택의 동작을
기본값으로 삼아 하나로 합친다. T002와 달라지면 K8의 동등성 시험이 깨지기 때문이다. 스택마다 다른 점은 다음과 같다. 네이티브 정책은 S4a·T002만 부르고 T009는 부르지
않는다. E6는 T009에 없었다. 자식 환경은 세 벌이며 T002 것이 가장 엄격하다(없애는 접두사와 PATH 거르기까지 한다).

``native_policy``는 새 프로세스에서 numpy·torch·scipy를 불러오기 전에 MKL을 순차 계층으로 고정하고, 오류를 숨기는 ``KMP_DUPLICATE_LIB_OK`` 설정을 거부한다. 이미 불러온
뒤라면 순차 계층이 적용된 새 프로세스가 필요하다고 거부한다. ``e6``는 모델 서버(Strata, llama, ollama)가 도는 동안 시작을 거부한다(저자 규칙 E6). 검사 기록을 얻을 수
없을 때(psutil 없음, 탐지 오류, 형식이 틀린 기록)도 거부한다. 확인하지 못한 것을 "없음"으로 읽지 않는다는 T002의 규칙이다. 이것은 S1의 경고 후 계속과 다르다.

``child_environment``는 자식 프로세스에 넘기는 환경을 만든다. 부모의 CODEX_·OPENAI_·CONDA_·GIT_ 변수와 PYTHONPATH·PYTHONHOME을 지워 자식이 부모의 도구 상태나 저장소 위치에
기대지 않게 하고, PATH에서 codex가 든 항목과 인터프리터 폴더로 시작하는 항목을 뺀다. 그다음 T002의 자식 설정(스레드 8, 순차 MKL, CUBLAS 작업 공간, 장치 0 등)을 덮어쓴다.
환경 변수 이름은 Windows에서 대소문자를 구별하지 않으므로 이름은 대문자로 비교하고, 설정 이름의 다른 철자는 지워 한 이름이 두 번 나오지 않게 한다. 입력에 같은 이름의 철자가
둘이면(``Path``와 ``PATH``) 자식이 어느 값을 볼지 모호하므로 거부한다. 입력은 바꾸지 않는다. 장치 목록과 스레드 수는 키워드로 바꿀 수 있고(기본값은 T002의 장치 0, 스레드 8),
CPU 자식이나 시험 자식은 장치 없음과 스레드 1로 만들어 GPU 캠페인이 쓰는 장치를 보지 못하게 한다.

``spec_check``는 결정 기록(SPEC)이 코드 상수와 맞는지 본다(S1-S3과 같은 검사). 맞으면 기록 바이트의 SHA-256을 돌려주어 호출자가 실행 기록에 남기게 한다. 호출자가 기대하는
해시를 주면 그것과도 맞아야 한다. 이 관문은 파일을 읽기만 하므로 실행 폴더, 잠금, 원장을 만들기 전에 불러도 아무 흔적이 없다.

``Budget``은 벽시계 시간과 메모리 한도이다. 한도는 포함이다(정확히 W초, M바이트는 통과). 한도를 넘거나, 시계가 거꾸로 가거나, 메모리를 읽지 못하면 거부한다. 읽지 못한 메모리를
0으로 읽으면 한도 검사가 거짓으로 통과하므로 읽지 못함은 거부이다.
"""
from __future__ import annotations

import math
import os
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

GATES = ('native', 'e6', 'environment', 'spec', 'budget')

# ---- native policy (Ref: c_ept/workflows/s4a_runtime.py prepare_native_runtime, RUNTIME_POLICY_VERSION) ------------------------------------
NATIVE_POLICY_VERSION = 1
OVERRIDE_OFF = ('', '0', 'false', 'no', 'off')                     # values of KMP_DUPLICATE_LIB_OK that do not hide library errors
NATIVE_MODULES = ('numpy', 'torch', 'scipy')                       # importing one of these fixes the MKL threading layer of the process
THREAD_KEYS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')   # recorded by native_policy, set by child_environment

# ---- child environment (Ref: tools/ept_training_controller.py child_environment, the T002 controller) --------------------------------------
REMOVED_PREFIXES = ('CODEX_', 'OPENAI_', 'CONDA_', 'GIT_')
REMOVED_KEYS = ('PYTHONPATH', 'PYTHONHOME')
PATH_BANNED_TEXT = 'codex'                                         # a PATH entry that contains this text (any case) is dropped
DEFAULT_THREADS = 8                                                # the T002 thread count of every thread variable in CHILD_SETTINGS
DEFAULT_CUDA_VISIBLE_DEVICES = '0'                                 # the T002 device list
MAX_THREADS = 1024                                                 # the largest thread count that ``child_environment`` accepts
CHILD_SETTINGS = MappingProxyType({
    'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8',
    'MKL_THREADING_LAYER': 'SEQUENTIAL', 'KMP_DUPLICATE_LIB_OK': 'FALSE',
    'OMP_NUM_THREADS': str(DEFAULT_THREADS), 'MKL_NUM_THREADS': str(DEFAULT_THREADS), 'OPENBLAS_NUM_THREADS': str(DEFAULT_THREADS),
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'CUDA_VISIBLE_DEVICES': DEFAULT_CUDA_VISIBLE_DEVICES,
})
_SETTING_NAMES = frozenset(CHILD_SETTINGS)                         # upper-case already
_SHA256 = re.compile(r'[0-9a-f]{64}')
# fullmatch: the empty text (no device) or numbers joined by single commas. K5b (K4 re-check R-n5): an empty item (',', '0,', ',0', '0,,1') is
# refused, because CUDA reads the list up to the first item it cannot parse, so '0,' and ',0' do not mean what they look like.
_DEVICES = re.compile(r'(?:[0-9]+(?:,[0-9]+)*)?')
_MAX_BYTES = 1 << 63                                               # the largest ``memory_bytes`` of a Budget: no host has more


class GateRefused(RuntimeError):
    """A gate refused to let the run start (or go on). ``gate`` is one of ``GATES`` (``ValueError`` for another name, so that a misspelt gate
    cannot pass silently); ``reason`` is a sentence without any secret.

    K5 maps every ``GateRefused`` to exit code 2 (Q7). ``args`` holds ``(gate, reason)``, so the exception survives pickling.
    """

    def __init__(self, gate: str, reason: str):
        if gate not in GATES:                                      # GATES holds text only, so a gate that is not text is not in it either
            raise ValueError('gate must be one of GATES')
        super().__init__(gate, reason)
        self.gate, self.reason = gate, reason

    def __str__(self) -> str:
        return f'{self.gate} gate: {self.reason}'


# ---- native policy ---------------------------------------------------------------------------------------------------------------------
def native_policy() -> dict:
    """Fix the native-thread policy of this process and return its record; ``GateRefused('native', ...)`` when the policy cannot hold.

    Call it in a new process before numpy, torch or scipy are imported. It refuses an error-hiding ``KMP_DUPLICATE_LIB_OK`` and a process that
    already imported a native module without ``MKL_THREADING_LAYER=SEQUENTIAL``; otherwise it sets ``KMP_DUPLICATE_LIB_OK=FALSE`` and
    ``MKL_THREADING_LAYER=SEQUENTIAL`` (so that sklearn's ``setdefault`` cannot switch the error hiding on). A refusal sets nothing. The record
    equals the one of ``prepare_native_runtime`` (T002), so the run fingerprint is the same; the thread counts are read, not set (the child
    environment sets them).

    해석: 새 프로세스에서 네이티브 모듈을 불러오기 전에 MKL을 순차 계층으로 고정한다. 이미 다른 정책으로 불러온 프로세스는 이 계약 밖이므로 거부한다.
    """
    override = os.environ.get('KMP_DUPLICATE_LIB_OK', '').strip().casefold()
    if override not in OVERRIDE_OFF:
        raise GateRefused('native', 'KMP_DUPLICATE_LIB_OK hides library errors; unset it and start a new process')
    if any(name in sys.modules for name in NATIVE_MODULES) and os.environ.get('MKL_THREADING_LAYER') != CHILD_SETTINGS['MKL_THREADING_LAYER']:
        raise GateRefused('native', 'a native module is already imported without MKL_THREADING_LAYER=SEQUENTIAL; '
                                    'start a new process that applies the policy before any native import')
    os.environ['KMP_DUPLICATE_LIB_OK'] = CHILD_SETTINGS['KMP_DUPLICATE_LIB_OK']
    os.environ['MKL_THREADING_LAYER'] = CHILD_SETTINGS['MKL_THREADING_LAYER']
    return {'policy_version': NATIVE_POLICY_VERSION, 'mkl_threading_layer': 'SEQUENTIAL',
            'numpy_mkl_parallelism': 'sequential', 'kmp_duplicate_lib_ok': 'disabled',
            'python_executable': str(Path(sys.executable).resolve()),
            'thread_environment': {key: os.environ.get(key) for key in THREAD_KEYS}}


# ---- E6 --------------------------------------------------------------------------------------------------------------------------------
_E6_FIELDS = {'running', 'matched', 'bridges_ignored', 'scanned', 'cmdline_unreadable', 'rule'}


def _unconfirmed(why: str) -> GateRefused:
    return GateRefused('e6', f'E6 cannot be confirmed: {why}')


def _check_probe_record(result) -> None:
    """Refuse a record that is not a well-formed, consistent E6 check record, and refuse while it says a model server runs.

    Ref: c_ept/workflows/s4a_runtime.py validate_e6_probe (same fields, same type rules, same consistency rules). Reimplemented because the
    kernel does not import that workflow module; a test runs both on the same records.

    식: running = (matched != []); len(matched) + bridges <= scanned; unreadable + bridges <= scanned; pids distinct and sorted.
    """
    if type(result) is not dict or set(result) != _E6_FIELDS:
        raise _unconfirmed('the check record has the wrong fields (a process listing may be unavailable)')
    if type(result['running']) is not bool or type(result['matched']) is not list:
        raise _unconfirmed('running or matched has the wrong type')
    if not isinstance(result['rule'], str) or not result['rule'].strip():
        raise _unconfirmed('the detection rule is not recorded')
    for key in ('bridges_ignored', 'scanned', 'cmdline_unreadable'):
        if type(result[key]) is not int or result[key] < 0:
            raise _unconfirmed('a count is not a non-negative whole number')
    matched = result['matched']
    for entry in matched:
        if (type(entry) is not dict or set(entry) != {'pid', 'name'} or type(entry['pid']) is not int or entry['pid'] < 0
                or not isinstance(entry['name'], str)):
            raise _unconfirmed('a matched process record is malformed')
    pids = [entry['pid'] for entry in matched]
    if len(set(pids)) != len(pids) or matched != sorted(matched, key=lambda entry: (entry['pid'], entry['name'])) or result['running'] != bool(matched):
        raise _unconfirmed('running and matched are inconsistent')
    # A matched name can have an unreadable command line, so these counts overlap; an ignored bridge had a readable command and is not matched.
    if (len(matched) + result['bridges_ignored'] > result['scanned']
            or result['cmdline_unreadable'] + result['bridges_ignored'] > result['scanned']):
        raise _unconfirmed('the process counts do not add up')
    if result['running']:
        found = ', '.join(f"{entry['name']} (pid {entry['pid']})" for entry in matched)
        raise GateRefused('e6', f'a Strata / llama / ollama model server is running ({found}); nothing was stopped: stop it yourself and start again')


def e6(*, processes=None, probe=None) -> dict:
    """Refuse while a model server (Strata, llama, ollama) runs (author rule E6); return the check record when none does.

    ``processes``: an iterable of ``(pid, name, command line)`` that replaces the machine's process list (tests); ``probe``: a function without
    arguments that gives the check record instead (not both). The default probe is ``runlog.strata_check``, the one detection rule. E6 is
    refused when it cannot be confirmed: no record (psutil missing), a malformed record or any error of the probe (only its type is named).
    Nothing is started or stopped, and no command line is shown. The default probe imports ``c_ept.runlog``, which imports numpy: call
    ``native_policy`` first, so that the policy is applied before the first native import.

    해석: 모델 서버가 도는 동안은 시작을 거부한다. 확인하지 못했을 때도 거부한다(T002 규칙). 프로세스를 멈추지 않는다. 기본 탐지기가 numpy를 불러오므로
    ``native_policy``를 먼저 부른다.
    """
    if processes is not None and probe is not None:
        raise ValueError('pass a process listing or a probe, not both')
    try:
        if probe is None:
            from c_ept import runlog
            record = runlog.strata_check(processes)
        else:
            record = probe()
        _check_probe_record(record)
    except GateRefused:
        raise
    except Exception as error:                                     # noqa: BLE001 - as run_s4a._e6: whatever the probe does wrong is a refusal
        raise GateRefused('e6', f'E6 unavailable or active: {type(error).__name__}') from error
    return record


# ---- child environment -----------------------------------------------------------------------------------------------------------------
def child_environment(base: Mapping[str, str], *, cuda_visible_devices: str = DEFAULT_CUDA_VISIBLE_DEVICES, threads: int = DEFAULT_THREADS) -> dict:
    """The environment for a child process: ``base`` without the parent's tool and repository variables, PATH filtered, the child settings set.

    Removed (name compared in upper case, so ``git_dir`` goes too): every name that starts with ``CODEX_``, ``OPENAI_``, ``CONDA_`` or ``GIT_``,
    and ``PYTHONPATH`` and ``PYTHONHOME``. PATH (any spelling) loses the entries that contain ``codex`` in any case and the entries that start
    with the folder of the running interpreter, ``Path(sys.executable).resolve().parent`` (case-insensitive prefix); a missing PATH becomes the
    empty string. Then ``CHILD_SETTINGS`` are set, replacing any spelling of their names. Everything else is copied as it is. ``base`` is not
    changed and a new dict is returned.

    The device and the thread count are the two settings of a child that a caller may need to change: ``cuda_visible_devices`` (numbers joined by single
    commas, or the empty text for no device; an empty item such as ``'0,'`` is refused; default ``'0'``, the T002 value) becomes ``CUDA_VISIBLE_DEVICES``, and ``threads`` (a whole number from 1 to
    ``MAX_THREADS``; default 8, the T002 value) becomes every thread variable of ``CHILD_SETTINGS`` (``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``,
    ``OPENBLAS_NUM_THREADS``). A CPU child or a test child (the test harness sets no device and one thread, because a GPU campaign shares the
    host) is made with ``cuda_visible_devices=''`` and ``threads=1``. The defaults give exactly the T002 dictionary.

    ``GateRefused('environment', ...)`` for a malformed override, for an unknown interpreter, and for a ``base`` that holds two spellings of one
    name under case folding (Windows reads names without case, so the child's value would be ambiguous; ``os.environ`` cannot hold two there);
    ``TypeError`` for a name or value that is not text. No message carries a value.

    식: B' = { (k, F(k, v)) : (k, v) in B, k not removed, up(k) not in names(S) } union S', with S' = S where the device is ``cuda_visible_devices``
    and each thread variable is ``threads``.

    Ref: tools/ept_training_controller.py child_environment (T002). That function reads the fixed interpreter ``PYTHON``; here it is
    ``sys.executable``, which is the same interpreter because the T002 controller requires it. A test runs both on the same inputs.

    해석: 자식이 부모의 도구 상태에 기대지 않도록 환경을 걸러 새로 만든다. T002 스택이 쓰던 규칙 그대로이며, 대소문자 철자가 다른 같은 이름이 둘 남지 않게 한다.
    CPU 자식이나 시험 자식은 장치 없음(``''``)과 스레드 1로 만들어 공유 GPU를 보지 못하게 한다. 기본값은 T002 자식(장치 0, 스레드 8)이다.
    """
    if not (isinstance(cuda_visible_devices, str) and _DEVICES.fullmatch(cuda_visible_devices)):
        raise GateRefused('environment', 'cuda_visible_devices must be the empty text or numbers joined by single commas')
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= MAX_THREADS:
        raise GateRefused('environment', f'threads must be a whole number from 1 to {MAX_THREADS}')
    if not sys.executable:
        raise GateRefused('environment', 'the running interpreter is unknown (sys.executable is empty), so its folder cannot be taken out of PATH')
    settings = {**CHILD_SETTINGS, 'CUDA_VISIBLE_DEVICES': cuda_visible_devices, **{key: str(threads) for key in THREAD_KEYS}}
    interpreter = str(Path(sys.executable).resolve().parent).casefold()
    result: dict = {}
    names = set()
    for key, value in base.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError('environment names and values must be text')
        name = key.upper()
        if name in names:
            raise GateRefused('environment', f'the environment holds two spellings of the name {name}')
        names.add(name)
        if name.startswith(REMOVED_PREFIXES) or name in REMOVED_KEYS or name in _SETTING_NAMES:
            continue
        if name == 'PATH':
            value = os.pathsep.join(entry for entry in value.split(os.pathsep)
                                    if PATH_BANNED_TEXT not in entry.casefold() and not entry.casefold().startswith(interpreter))
        result[key] = value
    if 'PATH' not in names:
        result['PATH'] = ''
    result.update(settings)
    return result


# ---- decision record -------------------------------------------------------------------------------------------------------------------
def spec_check(path=None, *, expected_sha256=None) -> str:
    """Refuse when the decision record does not agree with the code constants; return its SHA-256 (64 hex digits) when it does.

    ``path``: the record (default ``config.SPEC_PATH``, read when called). ``expected_sha256``: when given, the record's hash must equal it too.
    The check is ``config.check_spec`` (the S1-S3 check), so the record is read once and the hash is that of the bytes that were checked.
    This is the one exception to "every kernel module reads through ``run.io``": ``check_spec`` is the S1-S3 reader and keeps its lenient parser
    (``json.loads`` takes the last of two equal keys, accepts NaN, follows links, has no size limit), and the kernel does not reimplement it.
    The refusal is ``GateRefused('spec', <fixed text>)``, and the text names only the kind of failure: ``constant mismatch`` (the record disagrees
    with the code constants), ``unreadable`` (the file cannot be read or is not an object with an operating point) or ``hash mismatch``. It carries
    neither the record path nor a record value, and no exception is chained to it. ``ValueError`` for a malformed ``expected_sha256``. The gate
    only reads: call it before the run folder, the lock and the ledger are made.

    해석: 결정 기록의 운영점이 코드의 상수와 같아야 시작한다. 파일을 읽기만 하므로 거부해도 흔적이 남지 않는다. 읽기는 run.io가 아니라 S1-S3의 check_spec이 한다(느슨한
    파서 그대로이며 다시 만들지 않는다). 거부 문구에는 경로도 기록 값도 싣지 않는다.
    """
    if expected_sha256 is not None and not (isinstance(expected_sha256, str) and _SHA256.fullmatch(expected_sha256)):
        raise ValueError('expected_sha256 must be 64 lowercase hex digits')
    from c_ept import config
    target = config.SPEC_PATH if path is None else path
    refusal = digest = None
    try:
        digest = config.check_spec(target)
    except config.SpecMismatch:
        refusal = 'constant mismatch: the decision record disagrees with the code constants'
    except (OSError, ValueError, RecursionError):                 # unreadable file; not UTF-8 JSON (JSONDecodeError and UnicodeDecodeError are ValueErrors)
        refusal = 'unreadable: the decision record cannot be read or is malformed'
    if refusal is None and expected_sha256 is not None and digest != expected_sha256:
        refusal = 'hash mismatch: the decision record hash differs from the expected hash'
    if refusal is not None:
        raise GateRefused('spec', refusal)                        # outside the except block: no exception is chained, so no path or value is printed
    return digest


# ---- budget ----------------------------------------------------------------------------------------------------------------------------
def _finite(value) -> bool:
    """True for an int or float (not a bool) that is a finite float-range number; an int beyond the float range (``10 ** 400``) is False, not an ``OverflowError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _tree_private_bytes():
    """The default memory reader: ``(private bytes, upper bound)`` of this process and its children (``runtime.resources.tree_memory``)."""
    from c_ept.runtime import resources
    return resources.tree_memory()


class Budget:
    """A wall-clock and memory budget: ``check()`` raises ``GateRefused('budget', ...)`` once either bound is passed.

    ``wall_s``: seconds (a positive finite number); ``memory_bytes``: bytes (a whole number from 1 to 2^63). Both bounds are inclusive: elapsed
    ``e == wall_s`` and a sample ``m == memory_bytes`` pass. A number that is not finite or is beyond the float range (an int above about
    1.8e308) is refused with ``ValueError``, never ``OverflowError``; so is a clock reading of that kind. ``clock`` (default ``time.monotonic``)
    is read when the budget is made and at each check; ``measure`` (default: the private bytes of the process tree) gives a number of bytes or
    the pair ``(bytes, upper bound)`` of ``runtime.resources.tree_memory``, the bound being None or a non-negative whole number. A sample that
    is not a whole number above 0, or a pair that is malformed, refuses (a tree that cannot be read, or psutil missing, would read as 0 and
    pass). A clock that goes backwards refuses. The wall check comes first and reads no memory. Errors of ``measure`` propagate unchanged.

    ``peak_bytes`` is the largest sample read so far (the refused one included) and ``samples`` their number; ``peak_upper`` the largest upper
    bound seen (None while none is known; an unknown bound erases nothing), as ``evaluation.runtime.Budget`` kept it; ``elapsed_s()`` the seconds
    since the budget was made.

    식: e = t - t0;  accept <=> 0 <= e <= W  and  0 < m <= M.   기호 | 뜻 | 단위 | 범위: W wall_s | 허용 시간 | s | > 0;  M memory_bytes | 허용 메모리 | byte | 1..2^63;
    e | 경과 시간 | s | >= 0;  m | 표본 메모리 | byte | > 0.

    해석: 시간과 메모리 한도를 넘으면 거부한다. 한도 값은 통과이고 넘는 순간부터 거부이다. 메모리를 읽지 못하면 한도를 지킨다고 말할 수 없으므로 거부한다.
    실수 범위를 넘는 큰 정수는 OverflowError가 아니라 ValueError로 거부한다. 수명 상한(위쪽 경계)은 알려진 가장 큰 값을 ``peak_upper``에 남긴다.
    """

    def __init__(self, wall_s, memory_bytes, *, clock=time.monotonic, measure=None):
        if not _finite(wall_s) or wall_s <= 0:
            raise ValueError('wall_s must be a positive finite number of seconds')
        if isinstance(memory_bytes, bool) or not isinstance(memory_bytes, int) or not 0 < memory_bytes <= _MAX_BYTES:
            raise ValueError('memory_bytes must be a whole number of bytes from 1 to 2^63')
        self.wall_s, self.memory_bytes = wall_s, memory_bytes
        self._clock, self._measure = clock, measure
        self._start = self._now()
        self.peak_bytes, self.peak_upper, self.samples = 0, None, 0

    def _now(self):
        now = self._clock()
        if not _finite(now):
            raise ValueError('the clock must give a finite number of seconds')
        return now

    def elapsed_s(self):
        """Seconds since the budget was made (``e = t - t0``)."""
        return self._now() - self._start

    def check(self) -> None:
        elapsed = self.elapsed_s()
        if elapsed < 0:
            raise GateRefused('budget', 'the clock went backwards, so the wall-clock budget cannot be checked')
        if elapsed > self.wall_s:
            raise GateRefused('budget', f'wall-clock budget of {self.wall_s:g} s exceeded ({elapsed:.1f} s used)')
        got = self._measure() if self._measure is not None else _tree_private_bytes()
        private, upper = got if isinstance(got, tuple) and len(got) == 2 else (None, None) if isinstance(got, tuple) else (got, None)
        if type(private) is not int or private <= 0:
            raise GateRefused('budget', 'memory sample unavailable, so the memory budget cannot be checked')
        if upper is not None and (type(upper) is not int or upper < 0):
            raise GateRefused('budget', 'memory sample unavailable: its upper bound is malformed')
        self.samples += 1
        self.peak_bytes = max(self.peak_bytes, private)
        if upper is not None:
            self.peak_upper = upper if self.peak_upper is None else max(self.peak_upper, upper)
        if private > self.memory_bytes:
            raise GateRefused('budget', f'memory budget of {self.memory_bytes} bytes exceeded ({private} bytes sampled)')
