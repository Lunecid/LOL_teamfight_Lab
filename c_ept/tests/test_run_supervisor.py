"""Tests of the supervisor of the run kernel (``c_ept.run.supervisor``, with ``c_ept.run.scrub`` and ``c_ept.run.job``; S4b-prep K5b, decision D11, Q7).

The supervisor starts a run (or resumes one) from a hash-locked profile: the gates first (native policy, E6, environment, decision record, budget),
all before anything is reserved; then the run lock, the ledger (created, or opened and resumed after the identity is captured again and equal); then
the stages one by one, each in a child process that a Windows Job owns; then a last resume check and exit code 0. Exit codes (Q7): 0 success, 2 a
refusal (gate, strict IO, ledger tamper, identity, pause, a stage that refuses), 3 a computation failure (a stage that fails, a budget that runs out
during the run, an OS error). Every text it prints or records is scrubbed of match keys.

These tests use synthetic files in ``tmp_path`` and short CPU Python child processes (``c_ept/tests/run_stage_bodies.py``) that see
``CUDA_VISIBLE_DEVICES=''`` and one thread. No data, no GPU, no model. The identity, E6 and Git gates are injected (the working tree of a test
run is not a clean checkout); the defaults are wired to the real functions and a test says so.

해석: 감독자는 (1) 예약 전에 모든 관문을 거치고, (2) 실행 잠금을 잡은 뒤에야 원장을 만들거나 열고, (3) 단계마다 Windows Job이 소유한 자식 프로세스로 돌리며,
(4) 끝낸 뒤 한 번 더 검증하고 0으로 끝난다. 거부는 2, 계산 실패는 3이고, 출력과 기록에는 경기 키가 남지 않는다.
"""
import ast
import inspect
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from c_ept.run import guards
from c_ept.run import identity as run_identity
from c_ept.run import io as run_io
from c_ept.run import job as run_job
from c_ept.run import ledger as run_ledger
from c_ept.run import profile as run_profile
from c_ept.run import scrub as run_scrub
from c_ept.run import supervisor
from c_ept.run.identity import Identity
from c_ept.run.supervisor import Stage
from c_ept.tests.test_run_ledger import dump_spec, put, real_spec, sha, spec_doc

REPO = Path(__file__).resolve().parents[2]
SUPERVISOR_SOURCE = Path(supervisor.__file__)
BODIES = 'c_ept.tests.run_stage_bodies'
KEY = 'KR_1234567890'                                  # a match key: <platform>_<game id>
GAME_ID = '1234567890'
STDLIB = set(sys.stdlib_module_names)
PROFILE = b'{\n  "kind": "train"\n}\n'
E6_RECORD = {'running': False, 'matched': [], 'bridges_ignored': 0, 'scanned': 5, 'cmdline_unreadable': 0, 'rule': 'test rule'}


# ---- the rig -------------------------------------------------------------------------------------------------------------------------------
class Rig:
    """One test's world: a run folder, a decision record that agrees with the code constants, a train profile, a budget and the injected gates.

    ``log`` records, in order, the gates that were called: ``native``, ``e6``, ``capture``, ``start_gate``, ``verify`` and ``resume``.
    """

    def __init__(self, tmp_path, monkeypatch):
        self.tmp = tmp_path
        self.run_dir = tmp_path / 'run'
        self.spec = real_spec(tmp_path)
        self.profile_path = put(tmp_path / 'profile.json', PROFILE)
        self.profile = run_profile.load(self.profile_path)
        self.log = []
        self.stream = io.StringIO()
        self.identity = Identity(commit='ab' * 20, files={'c_ept/run/supervisor.py': 'cd' * 32}, protected=())
        self.over_memory = threading.Event()
        real_native, real_resume = guards.native_policy, run_ledger.Ledger.resume
        rig = self

        def native():
            rig.log.append('native')
            return real_native()

        def resume(self_):
            rig.log.append('resume')
            return real_resume(self_)
        monkeypatch.setattr(guards, 'native_policy', native)
        monkeypatch.setattr(run_ledger.Ledger, 'resume', resume)

    # -- injected gates
    def e6(self):
        self.log.append('e6')
        return dict(E6_RECORD)

    def capture(self, modules, *, repo=None):
        self.log.append('capture')
        return self.identity

    def verify(self, saved, *, repo=None):
        self.log.append('verify')

    def start_gate(self, repo, identity, **kwargs):
        self.log.append('start_gate')

    def measure(self):
        return 2 ** 31 if self.over_memory.is_set() else 10 ** 6

    def budget(self, wall_s=10 ** 6):
        return guards.Budget(wall_s, 2 ** 30, measure=self.measure)

    # -- stages and runs
    def stage(self, name, function, *args, inputs=(), outputs=(), wall_cap_s=None):
        return Stage(name, f'{BODIES}:{function}', args=tuple(args), inputs=tuple(inputs), outputs=tuple(outputs), wall_cap_s=wall_cap_s)

    def two_stages(self):
        return [self.stage('train', 'make_output', 'train/best.bin', 'weights', inputs=[self.profile_path], outputs=['train/best.bin']),
                self.stage('evaluate', 'copy_input', 'train/best.bin', 'evaluate/scores.bin', inputs=['train/best.bin'], outputs=['evaluate/scores.bin'])]

    def kwargs(self, stages, **over):
        values = dict(stages=stages, spec_path=self.spec, budget=self.budget(), cuda_visible_devices='', threads=1, scrub_keys=[KEY], stream=self.stream,
                      e6=self.e6, capture=self.capture, verify=self.verify, start_gate=self.start_gate, training_cap_s=10.0, margin_s=5.0, poll_s=0.05,
                      repo=REPO)
        values.update(over)
        return values

    def run(self, stages, **over):
        return supervisor.run(self.profile, self.run_dir, **self.kwargs(stages, **over))

    # -- what is on disk
    def ledger(self):
        return run_ledger.Ledger(self.run_dir)

    def nothing_was_reserved(self):
        return not self.run_dir.exists()

    def text(self):
        return self.stream.getvalue()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    return Rig(tmp_path, monkeypatch)


def wait_for(path, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if Path(path).exists():
            try:
                return json.loads(Path(path).read_text(encoding='utf-8'))
            except (ValueError, OSError):                                             # the file is there but not written yet
                pass
        time.sleep(0.02)
    raise AssertionError('the file did not appear: ' + str(path))


def gone(pid, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return True
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.05)
    return False


# ---- the interface, the table of exit codes and the module's own discipline --------------------------------------------------------------------
def test_run_has_the_planned_signature():
    parameters = inspect.signature(supervisor.run).parameters
    assert list(parameters)[:2] == ['profile', 'run_dir']
    assert all(parameters[name].kind is inspect.Parameter.KEYWORD_ONLY for name in list(parameters)[2:])
    required = {name for name, parameter in parameters.items() if parameter.default is inspect.Parameter.empty}
    assert required == {'profile', 'run_dir', 'stages', 'spec_path', 'budget', 'cuda_visible_devices', 'threads'}     # the runner decides device and threads
    assert (supervisor.EXIT_OK, supervisor.EXIT_REFUSED, supervisor.EXIT_FAILED) == (0, 2, 3)
    assert supervisor.TRAINING_CAP_S == 24 * 3600
    assert inspect.signature(supervisor.run).return_annotation in ('int', int)


def test_the_training_cap_is_the_cap_of_the_training_loop():
    """K6 will pass ``train.WALL_CAP_SECONDS``; the supervisor's default must be the same number, or the budget gate would lie."""
    source = (REPO / 'c_ept' / 'ept' / 'train.py').read_text(encoding='utf-8')
    assert 'WALL_CAP_SECONDS = 24 * 3600' in source and supervisor.TRAINING_CAP_S == 24 * 3600


def test_the_exit_code_table_is_in_the_module_text():
    text = supervisor.__doc__
    for name in ('GateRefused', 'IORefused', 'ProfileRefused', 'IdentityChanged', 'IdentityUnconfirmed', 'Paused', 'Tampered', 'LedgerRefused', 'RunLocked',
                 'StageRefused', 'OSError', 'RuntimeError', 'BudgetStop', 'StageFailed', 'unopenable', 'budget'):
        assert name in text, name
    assert 'exit code 0' in text.lower() or '0 success' in text.lower()


def test_the_module_imports_only_the_standard_library_and_public_kernel_modules():
    tree = ast.parse(SUPERVISOR_SOURCE.read_text(encoding='utf-8'))
    allowed = {'io', 'identity', 'guards', 'ledger', 'job', 'scrub'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split('.')[0] in STDLIB for a in node.names), [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, 'a relative import'
            if (node.module or '').split('.')[0] not in STDLIB:
                assert node.module == 'c_ept.run' and {a.name for a in node.names} <= allowed, (node.module, [a.name for a in node.names])


def test_the_module_follows_the_kernel_boundary_rule_q8():
    """Q8/K7 for ``c_ept/run/``: no private (underscore) name or module from another package, and no command module (``tools`` included)."""
    commands = ('c_ept.workflows.run_', 'c_ept.workflows.s4a_operator', 'c_ept.workflows.training_operator', 'c_ept.comparison.operator', 'tools')
    for path in (SUPERVISOR_SOURCE, Path(run_scrub.__file__), Path(run_job.__file__)):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ''
                assert not any(part.startswith('_') and part != '__future__' for part in module.split('.')), module
                assert not any(a.name.startswith('_') for a in node.names if module != '__future__'), module
                assert not module.startswith(commands), module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(part.startswith('_') for part in alias.name.split('.')), alias.name
                    assert not alias.name.startswith(commands), alias.name


def test_importing_the_supervisor_loads_no_native_or_model_code():
    code = ('import sys; import c_ept.run.supervisor; '
            'bad = sorted({"numpy", "torch", "scipy", "psutil", "c_ept.runlog", "c_ept.config", "c_ept.legacy", "c_ept.runtime"} & sys.modules.keys()); print(bad)')
    done = subprocess.run([sys.executable, '-B', '-c', code], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == '[]', done.stdout


@pytest.mark.parametrize('exception, code', [
    (guards.GateRefused('spec', 'x'), 2), (guards.GateRefused('budget', 'x'), 2), (guards.GateRefused('e6', 'x'), 2),     # a RuntimeError, but a refusal first
    (run_io.IORefused('x'), 2), (run_profile.ProfileRefused('x'), 2), (run_ledger.LedgerRefused('x'), 2), (run_ledger.Tampered('x'), 2),
    (run_identity.IdentityChanged('x'), 2), (run_identity.IdentityUnconfirmed('x'), 2), (run_identity.Paused('x'), 2),
    (supervisor.SupervisorRefused('x'), 2), (supervisor.RunLocked('x'), 2), (supervisor.StageRefused('x'), 2),
    (RuntimeError('x'), 3), (KeyError('KR_1'), 3), (ValueError('x'), 3), (OSError('x'), 3), (PermissionError(13, 'denied', 'p'), 3),
    (FileNotFoundError('x'), 3), (supervisor.BudgetStop('x'), 3), (supervisor.StageFailed('x'), 3), (run_job.JobError(5, 'x'), 3),
    (TypeError('x'), 3), (AssertionError('x'), 3), (ZeroDivisionError('x'), 3)])
def test_exit_code_for_maps_refusals_to_2_before_the_generic_runtime_error_and_the_rest_to_3(exception, code):
    assert supervisor.exit_code_for(exception) == code


def test_the_refusal_classes_are_what_the_ledger_and_the_table_say():
    assert issubclass(supervisor.SupervisorRefused, run_io.IORefused) and issubclass(supervisor.RunLocked, supervisor.SupervisorRefused)
    assert issubclass(supervisor.StageRefused, supervisor.SupervisorRefused)
    assert issubclass(supervisor.StageFailed, RuntimeError) and issubclass(supervisor.BudgetStop, RuntimeError)
    assert not issubclass(supervisor.StageFailed, run_io.IORefused) and not issubclass(supervisor.BudgetStop, guards.GateRefused)


# ---- key scrubbing -----------------------------------------------------------------------------------------------------------------------
SCRUB_TEXTS = [
    f'Traceback (most recent call last):\n  File "x.py", line 3, in f\nKeyError: {KEY!r}\n',
    f'opened D:/data/{KEY}/timeline.json then {GAME_ID} again and {KEY}',
    'nothing to hide here', '', f'{GAME_ID}{GAME_ID}', f'x{KEY}y_{GAME_ID}', f'EUW1_5550001 and KR_5550002 and {KEY}',
    f'KR_{GAME_ID[:5]} is not the key but a prefix of it: {KEY}',
]
SCRUB_KEY_SETS = [[KEY], [KEY, 'EUW1_5550001', 'KR_5550002'], [], [''], [GAME_ID], ['12345', KEY], ['EUW1_5550001', KEY], ['NOUNDERSCORE'], [KEY, KEY],
                  ['KR_', 'KR_1'], ['_123']]


@pytest.mark.parametrize('text', SCRUB_TEXTS)
@pytest.mark.parametrize('keys', SCRUB_KEY_SETS)
def test_scrub_text_agrees_with_the_frozen_run_s3_function(text, keys):
    """``run_s3._scrub_text`` is frozen and imports the numpy-based store builder; the kernel keeps the same rule and a test runs both."""
    from c_ept.workflows import run_s3
    assert run_scrub.scrub_text(text, keys) == run_s3._scrub_text(text, keys)


def test_scrub_text_replaces_the_key_and_the_game_id():
    text = f'bad value {KEY} in D:/x/{KEY}.json; game {GAME_ID}'
    assert run_scrub.scrub_text(text, [KEY]) == 'bad value <key> in D:/x/<key>.json; game <id>'
    assert run_scrub.scrub_text(text, []) == text and run_scrub.scrub_text(text, ['']) == text
    assert run_scrub.scrub_text('', [KEY]) == ''
    assert run_scrub.scrub_text(f'{KEY}', iter([KEY])) == '<key>'                 # any iterable of keys, read once


def test_scrub_text_keeps_the_reference_to_the_frozen_rule():
    source = Path(run_scrub.__file__).read_text(encoding='utf-8')
    assert '# Ref:' in source and 'run_s3' in source and 'store_build' in source
    imported = [a.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Import) for a in n.names]
    imported += [n.module or '' for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom)]
    assert all(name.split('.')[0] in STDLIB for name in imported), imported                    # the frozen modules import numpy; this one imports none


# ---- the Windows Job copy ---------------------------------------------------------------------------------------------------------------
def test_the_job_module_is_the_code_of_the_t002_job_module():
    """``tools/ept_training_job.py`` is frozen (the T002 controller checks that ``WindowsJob`` comes from that file), and ``c_ept/run`` may not
    import from ``tools`` (Q8). So the kernel holds a copy; this test pins that the two define exactly the same code."""
    def definitions(path):
        tree = ast.parse(Path(path).read_text(encoding='utf-8'))
        return {node.name: ast.dump(node) for node in tree.body if isinstance(node, ast.ClassDef)}
    old, new = definitions(REPO / 'tools' / 'ept_training_job.py'), definitions(run_job.__file__)
    assert sorted(old) == ['JobError', 'WindowsJob', '_Accounting', '_BasicLimits', '_ExtendedLimits', '_IoCounters']
    assert old == new
    assert '# Ref:' in Path(run_job.__file__).read_text(encoding='utf-8')


# ---- run_owned: the child belongs to a Job ----------------------------------------------------------------------------------------------
GATED = 'import sys\nif sys.stdin.buffer.read(1) != b"G": sys.exit(125)\n'


def owned(code, *args, check=None, tail_bytes=65536, poll_s=0.05):
    env = guards.child_environment(os.environ, cuda_visible_devices='', threads=1)
    return supervisor.run_owned([sys.executable, '-I', '-c', GATED + code, *args], env=env, cwd=REPO, check=check, poll_s=poll_s, tail_bytes=tail_bytes)


def test_run_owned_returns_the_exit_code_and_the_output_of_a_gated_child():
    result = owned('import sys; print("hello"); print("oops", file=sys.stderr); sys.exit(7)')
    assert result.returncode == 7 and b'hello' in result.output and b'oops' in result.output and not result.truncated


def test_the_child_waits_for_the_release_and_runs_only_after_the_job_holds_it(monkeypatch):
    """The child reads one byte before it does anything; the job writes it only after the child is assigned (the T002 gate)."""
    events = []
    real_assign, real_release = run_job.WindowsJob.assign, run_job.WindowsJob.release

    def assign(self, child):
        real_assign(self, child)
        events.append(('assigned', time.time()))

    def release(self, child):
        time.sleep(0.8)                                                               # a child that did not wait would have printed by now
        events.append(('released', time.time()))
        real_release(self, child)
    monkeypatch.setattr(run_job.WindowsJob, 'assign', assign)
    monkeypatch.setattr(run_job.WindowsJob, 'release', release)
    result = owned('import time; print(repr(time.time()))')
    assert result.returncode == 0 and [name for name, _ in events] == ['assigned', 'released']
    assert float(result.output.decode().strip()) >= events[1][1]                         # the child ran after the release, so after the assignment


def test_child_is_killed_with_the_job(tmp_path):
    """A check that raises (a budget, a stop request) ends the call with the whole process tree dead: the child and its grandchild."""
    ready = tmp_path / 'ready.json'
    code = ('import json, subprocess, sys, time\n'
            'g = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL)\n'
            'open(sys.argv[1], "w").write(json.dumps({"child": __import__("os").getpid(), "grandchild": g.pid}))\n'
            'time.sleep(120)\n')

    class Halt(Exception):
        pass

    def check():
        if ready.exists() and ready.read_text(encoding='utf-8').endswith('}'):
            raise Halt
    started = time.monotonic()
    with pytest.raises(Halt):
        owned(code, str(ready), check=check)
    pids = json.loads(ready.read_text(encoding='utf-8'))
    assert gone(pids['child']) and gone(pids['grandchild']) and time.monotonic() - started < 40


def test_a_grandchild_left_behind_by_a_finished_child_is_killed_with_the_job(tmp_path):
    ready = tmp_path / 'ready.json'
    code = ('import json, os, subprocess, sys\n'
            'g = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,\n'
            '                     stderr=subprocess.DEVNULL)\n'
            'open(sys.argv[1], "w").write(json.dumps({"child": os.getpid(), "grandchild": g.pid}))\n')
    alive_before_close = []
    real_close = run_job.WindowsJob.close

    def close(self):                                                                   # the Job's own close would kill the grandchild anyway;
        pids = json.loads(ready.read_text(encoding='utf-8'))                           # so look before it: run_owned must have stopped the tree itself
        alive_before_close.append(psutil.pid_exists(pids['grandchild']) and psutil.Process(pids['grandchild']).status() != psutil.STATUS_ZOMBIE)
        real_close(self)
    run_job.WindowsJob.close = close
    try:
        result = owned(code, str(ready))
    finally:
        run_job.WindowsJob.close = real_close
    assert result.returncode == 0 and alive_before_close == [False]
    pids = json.loads(ready.read_text(encoding='utf-8'))
    assert gone(pids['child']) and gone(pids['grandchild'])


def test_the_output_is_kept_as_a_bounded_tail():
    result = owned('import sys\nfor _ in range(300): sys.stdout.write("0123456789" * 100 + "\\n")\nprint("LAST LINE", flush=True)', tail_bytes=2000)
    assert result.returncode == 0 and result.truncated and len(result.output) <= 2000 and result.output.rstrip().endswith(b'LAST LINE')


def test_run_owned_refuses_a_command_that_is_not_a_list_of_text():
    env = guards.child_environment(os.environ, cuda_visible_devices='', threads=1)
    for argv in ('python', [], [5], [sys.executable, None], None):
        with pytest.raises((TypeError, ValueError)):
            supervisor.run_owned(argv, env=env, cwd=REPO)


# ---- stages ------------------------------------------------------------------------------------------------------------------------------
def top_level_function():
    return None


def test_a_stage_is_a_name_an_importable_body_and_text_arguments():
    stage = Stage('train', 'c_ept.tests.run_stage_bodies:make_output', args=('train/best.bin',), inputs=['in.bin'], outputs=['out.bin'])
    assert (stage.name, stage.body, stage.args, stage.inputs, stage.outputs) == ('train', 'c_ept.tests.run_stage_bodies:make_output', ('train/best.bin',),
                                                                              ('in.bin',), ('out.bin',))
    by_function = Stage('evaluate', top_level_function)
    assert by_function.body == f'{__name__}:top_level_function' and by_function.args == () and by_function.inputs == ()
    with pytest.raises(AttributeError):
        stage.name = 'x'                                                              # frozen


@pytest.mark.parametrize('arguments', [
    ('Train', 'a.b:c'), ('', 'a.b:c'), ('con', 'a.b:c'), ('a b', 'a.b:c'), (5, 'a.b:c'), ('x' * 49, 'a.b:c'),
    ('train', 'no_colon'), ('train', ':f'), ('train', 'a.b:'), ('train', 'a b:c'), ('train', 'a.b:c d'), ('train', '../a:c'), ('train', 5), ('train', None),
    ('train', lambda: None), ('train', 'a.b:c', (5,)), ('train', 'a.b:c', 'text'), ('train', 'a.b:c', ('a\x00b',)),
])
def test_a_stage_that_is_not_acceptable_is_refused(arguments):
    name, body, *rest = arguments
    with pytest.raises(supervisor.SupervisorRefused):
        Stage(name, body, args=rest[0] if rest else ())


def test_a_nested_function_and_a_function_of_main_cannot_be_a_body():
    def nested():
        return None
    with pytest.raises(supervisor.SupervisorRefused):
        Stage('train', nested)
    main_function = type(top_level_function)(top_level_function.__code__, {}, 'f')
    main_function.__module__ = '__main__'
    with pytest.raises(supervisor.SupervisorRefused):
        Stage('train', main_function)


# ---- a whole run ---------------------------------------------------------------------------------------------------------------------------
def test_a_new_run_runs_every_stage_in_order_and_exits_0(rig):
    assert rig.run(rig.two_stages()) == 0
    ledger = rig.ledger()
    assert ledger.stages == (('train', 'finished'), ('evaluate', 'finished'))
    assert sorted(ledger.stage('train').outputs) == ['train/best.bin'] and sorted(ledger.stage('evaluate').outputs) == ['evaluate/scores.bin']
    assert (rig.run_dir / 'evaluate' / 'scores.bin').read_text(encoding='utf-8') == 'derived from weights'
    assert (rig.run_dir / 'calls.txt').read_text(encoding='utf-8').count('x') == 2
    header = ledger.header
    assert header.profile_sha256 == rig.profile.sha256 and header.identity == rig.identity
    assert dict(header.spec_blocks) == run_ledger.spec_fingerprint(rig.spec.read_bytes())['spec_blocks']
    assert dict(header.spec_top) == run_ledger.spec_fingerprint(rig.spec.read_bytes())['spec_top'] and header.spec_sha256 == sha(rig.spec.read_bytes())
    assert not (rig.run_dir / supervisor.LOCK_FILE).exists()                          # the lock is released
    assert rig.text() == ''                                                           # a successful run says nothing


def test_the_header_records_the_runner_constants_of_the_supervisor(rig):
    assert rig.run(rig.two_stages(), runner={'batch_size': 32}) == 0
    runner = rig.ledger().header.runner
    assert runner['batch_size'] == 32
    assert dict(runner['supervisor']) == {'cuda_visible_devices': '', 'threads': 1, 'wall_budget_s': 10 ** 6, 'memory_bytes': 2 ** 30, 'training_cap_s': 10.0,
                                          'finalisation_margin_s': 5.0, 'stages': ('train', 'evaluate')}
    rig.run_dir = rig.tmp / 'second_run'
    assert rig.run(rig.two_stages(), runner={'supervisor': 1}) == 2                    # the key is the supervisor's own
    assert rig.nothing_was_reserved()


def test_e6_records_and_the_stage_order_are_in_the_ledger_notes(rig):
    assert rig.run(rig.two_stages()) == 0
    notes = rig.ledger().notes
    assert [(n.stage, n.kind) for n in notes] == [(None, 'e6'), ('train', 'e6'), ('evaluate', 'e6')]
    assert all(dict(n.record)['running'] is False and dict(n.record)['rule'] == 'test rule' for n in notes)


def test_every_stage_has_a_log_without_keys(rig):
    stages = [rig.stage('train', 'print_text', f'seen {KEY} and {GAME_ID}', outputs=[])]
    assert rig.run(stages) == 0
    log = (rig.run_dir / supervisor.LOG_FOLDER / 'train.log').read_text(encoding='utf-8')
    assert 'seen <key> and <id>' in log and KEY not in log and GAME_ID not in log


def test_running_a_finished_run_again_runs_no_stage_and_exits_0(rig):
    assert rig.run(rig.two_stages()) == 0
    before = {p.relative_to(rig.run_dir).as_posix(): p.read_bytes() for p in rig.run_dir.rglob('*') if p.is_file() and p.name != 'LEDGER.json'}
    assert rig.run(rig.two_stages()) == 0
    after = {p.relative_to(rig.run_dir).as_posix(): p.read_bytes() for p in rig.run_dir.rglob('*') if p.is_file() and p.name != 'LEDGER.json'}
    assert before == after and (rig.run_dir / 'calls.txt').read_text(encoding='utf-8').count('x') == 2          # no stage ran again
    assert rig.ledger().stages == (('train', 'finished'), ('evaluate', 'finished'))


def test_a_failed_stage_is_3_and_a_later_start_continues_from_it(rig):
    stages = [rig.stage('train', 'make_output', 'train/best.bin', 'weights', outputs=['train/best.bin']),
              rig.stage('evaluate', 'fail_once_then_work', 'evaluate/scores.bin', outputs=['evaluate/scores.bin'])]
    assert rig.run(stages) == 3
    ledger = rig.ledger()
    assert ledger.stages == (('train', 'finished'), ('evaluate', 'begun'))
    stop = [n for n in ledger.notes if n.kind == 'stop']
    assert len(stop) == 1 and stop[0].stage == 'evaluate' and dict(stop[0].record)['reason'] == 'failure' and dict(stop[0].record)['exit_code'] == 3
    assert 'first call fails' in dict(stop[0].record)['detail']
    assert 'first call fails' in rig.text()
    rig.stream.truncate(0)
    assert rig.run(stages) == 0
    assert rig.ledger().stages == (('train', 'finished'), ('evaluate', 'finished'))
    assert (rig.run_dir / 'calls.txt').read_text(encoding='utf-8').count('x') == 2         # train once; evaluate once (the failed call wrote nothing)


def test_a_stage_that_refuses_in_its_child_is_2_with_the_reason_in_the_ledger(rig):
    stages = [rig.stage('train', 'raise_gate_refusal')]
    assert rig.run(stages) == 2
    stop = [n for n in rig.ledger().notes if n.kind == 'stop']
    assert len(stop) == 1 and dict(stop[0].record)['reason'] == 'refused' and dict(stop[0].record)['exit_code'] == 2
    assert 'GateRefused' in rig.text() or 'spec gate' in rig.text()


def test_a_stage_that_exits_with_another_code_is_3(rig):
    assert rig.run([rig.stage('train', 'exit_with', '9')]) == 3
    assert dict([n for n in rig.ledger().notes if n.kind == 'stop'][0].record)['exit_code'] == 9


def test_declared_outputs_that_are_missing_make_the_stage_a_failure_not_a_tamper(rig):
    stages = [rig.stage('train', 'no_output', outputs=['train/never_written.bin'])]
    assert rig.run(stages) == 3
    ledger = rig.ledger()
    assert ledger.stages == (('train', 'begun'),)
    assert dict([n for n in ledger.notes if n.kind == 'stop'][0].record)['reason'] == 'failure'


def test_a_plan_that_is_not_the_start_of_the_ledger_is_refused(rig):
    assert rig.run(rig.two_stages()) == 0
    other = [rig.stage('train', 'make_output', 'train/best.bin', 'weights', outputs=['train/best.bin']), rig.stage('score', 'no_output')]
    assert rig.run(other) == 2
    assert rig.run(rig.two_stages()[:1]) == 2                                          # a shorter plan than the finished stages is not the same plan
    assert rig.ledger().stages == (('train', 'finished'), ('evaluate', 'finished'))


def test_a_resume_with_another_profile_is_refused(rig, tmp_path):
    assert rig.run(rig.two_stages()) == 0
    other_path = put(tmp_path / 'other.json', b'{\n  "kind": "train",\n  "epochs": 2\n}\n')
    rig.profile = run_profile.load(other_path)
    assert rig.run(rig.two_stages()) == 2
    assert 'profile' in rig.text()


# ---- the order of the gates and the absence of leftovers ----------------------------------------------------------------------------------
def test_native_policy_is_the_first_call_and_the_gates_follow_in_the_documented_order(rig):
    assert rig.run(rig.two_stages()) == 0
    assert rig.log == ['native', 'e6', 'capture', 'start_gate', 'e6', 'verify', 'e6', 'verify', 'resume']        # the last is the final resume check


def test_the_defaults_are_wired_to_the_real_gates(rig, monkeypatch):
    seen = []
    monkeypatch.setattr(guards, 'e6', lambda: (seen.append('e6'), dict(E6_RECORD))[1])
    monkeypatch.setattr(run_identity, 'capture', lambda modules, *, repo=None: (seen.append(('capture', tuple(modules), repo)), rig.identity)[1])
    monkeypatch.setattr(run_identity, 'verify', lambda saved, *, repo=None: seen.append('verify'))
    monkeypatch.setattr(run_identity, 'start_gate', lambda repo, identity, **kw: seen.append('start_gate'))
    stages = [rig.stage('train', 'make_output', 'train/best.bin', 'w', outputs=['train/best.bin'])]
    assert rig.run(stages, e6=None, capture=None, verify=None, start_gate=None, identity_modules=('c_ept.run.supervisor',)) == 0
    assert seen == ['e6', ('capture', ('c_ept.run.supervisor',), REPO), 'start_gate', 'e6', 'verify']


def test_native_policy_runs_before_numpy_is_imported(tmp_path):
    """In a fresh interpreter: importing the supervisor loads no native module, and ``native_policy`` is called (first) before any is loaded."""
    script = tmp_path / 'order.py'
    script.write_text(
        'import sys\n'
        f'sys.path.insert(0, {str(REPO)!r})\n'
        'from c_ept.run import guards, supervisor\n'
        'seen = []\n'
        'real = guards.native_policy\n'
        'def native():\n'
        '    seen.append(("native", sorted({"numpy", "torch", "scipy"} & sys.modules.keys())))\n'
        '    return real()\n'
        'guards.native_policy = native\n'
        'def e6():\n'
        '    seen.append(("e6", sorted({"numpy", "torch", "scipy"} & sys.modules.keys())))\n'
        '    raise guards.GateRefused("e6", "stop here")\n'
        'class P: kind = "train"; sha256 = "ab" * 32\n'
        'budget = guards.Budget(10 ** 6, 2 ** 30, measure=lambda: 10 ** 6)\n'
        'code = supervisor.run(P(), sys.argv[1], stages=[supervisor.Stage("train", "a.b:c")], spec_path="x", budget=budget, cuda_visible_devices="",\n'
        '                      threads=1, e6=e6, stream=open(sys.argv[2], "w"))\n'
        'print(code, seen)\n', encoding='utf-8')
    done = subprocess.run([sys.executable, '-I', str(script), str(tmp_path / 'run'), str(tmp_path / 'err.txt')], capture_output=True, text=True, timeout=60,
                          env=guards.child_environment(os.environ, cuda_visible_devices='', threads=1))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "2 [('native', []), ('e6', [])]"
    assert not (tmp_path / 'run').exists()


GATE_REFUSALS = {
    'native': lambda rig, mp: mp.setattr(guards, 'native_policy', lambda: (_ for _ in ()).throw(guards.GateRefused('native', 'x'))),
    'e6': lambda rig, mp: mp.setattr(rig, 'e6', lambda: (_ for _ in ()).throw(guards.GateRefused('e6', 'a model server runs'))),
    'spec constants': lambda rig, mp: real_spec(rig.tmp, lambda d: d['operating_point'].update(G_ms=-1)),
    'spec unopenable': lambda rig, mp: rig.spec.unlink(),
    'spec not json': lambda rig, mp: rig.spec.write_bytes(b'not json'),
    'identity': lambda rig, mp: mp.setattr(rig, 'capture', lambda modules, *, repo=None: (_ for _ in ()).throw(run_identity.IdentityChanged('x'))),
    'start gate': lambda rig, mp: mp.setattr(rig, 'start_gate', lambda repo, identity, **kw: (_ for _ in ()).throw(run_identity.IdentityUnconfirmed('x'))),
    'paused': lambda rig, mp: mp.setattr(rig, 'start_gate', lambda repo, identity, **kw: (_ for _ in ()).throw(run_identity.Paused('git is slow'))),
}


@pytest.mark.parametrize('name', sorted(GATE_REFUSALS))
def test_a_refusal_before_the_reservation_is_2_and_leaves_no_folder_lock_or_ledger(rig, monkeypatch, name):
    GATE_REFUSALS[name](rig, monkeypatch)
    assert rig.run(rig.two_stages()) == 2
    assert rig.nothing_was_reserved(), sorted(rig.run_dir.rglob('*'))
    assert rig.text() != '' and KEY not in rig.text()


def test_the_environment_gate_refuses_before_the_reservation(rig):
    assert rig.run(rig.two_stages(), cuda_visible_devices='0,') == 2                  # an empty item in the device list
    assert rig.nothing_was_reserved() and 'environment' in rig.text()
    assert rig.run(rig.two_stages(), threads=0) == 2 and rig.nothing_was_reserved()
    assert rig.run(rig.two_stages(), environ={'PATH': 'a', 'Path': 'b'}) == 2 and rig.nothing_was_reserved()


def test_git_must_be_on_the_filtered_path_before_anything_is_reserved(rig, tmp_path):
    empty = tmp_path / 'no_git_here'
    empty.mkdir()
    environ = dict(os.environ, PATH=str(empty))
    assert rig.run(rig.two_stages(), environ=environ) == 2
    assert rig.nothing_was_reserved() and 'git' in rig.text()
    # a PATH that holds git only inside the interpreter folder is filtered away, so git is missing for the child too
    interpreter = str(Path(sys.executable).resolve().parent)
    assert rig.run(rig.two_stages(), environ=dict(os.environ, PATH=interpreter)) == 2 and rig.nothing_was_reserved()
    assert rig.run(rig.two_stages()) == 0                                              # the real PATH has git


def test_an_invalid_plan_is_refused_before_the_reservation(rig):
    good = rig.two_stages()
    for stages in ([], [good[0], good[0]], [good[0], 'not a stage'], good[0], None, [Stage('a', 'x.y:z', outputs=['../out'])],
                   [Stage('a', 'x.y:z', inputs=[5])]):
        assert rig.run(stages) == 2, stages
        assert rig.nothing_was_reserved()


def test_a_budget_that_cannot_cover_the_training_cap_and_the_margin_is_2_at_the_start(rig):
    assert rig.run(rig.two_stages(), budget=rig.budget(wall_s=14.9)) == 2                # cap 10 s + margin 5 s = 15 s
    assert rig.nothing_was_reserved() and 'budget' in rig.text()
    assert rig.run(rig.two_stages(), budget=rig.budget(wall_s=15.0)) == 0               # the bound is inclusive


def test_the_default_training_cap_is_24_hours_for_a_train_profile_and_zero_for_the_others(rig, tmp_path):
    stages = rig.two_stages()
    assert rig.run(stages, training_cap_s=None, margin_s=3600.0, budget=rig.budget(wall_s=24 * 3600 + 3599)) == 2 and rig.nothing_was_reserved()
    assert rig.run(stages, training_cap_s=None, margin_s=3600.0, budget=rig.budget(wall_s=24 * 3600 + 3600)) == 0
    evaluate = run_profile.load(put(tmp_path / 'evaluate.json', b'{\n  "kind": "evaluate",\n  "role": "STOP",\n  "models": ["b0"]\n}\n'))
    assert supervisor.required_wall_s(evaluate, None, 3600.0) == 3600.0 and supervisor.required_wall_s(rig.profile, None, 3600.0) == 24 * 3600 + 3600.0
    assert supervisor.required_wall_s(rig.profile, 7.0, 1.0) == 8.0


def test_a_memory_sample_over_the_budget_at_the_start_is_2(rig):
    rig.over_memory.set()
    assert rig.run(rig.two_stages()) == 2 and rig.nothing_was_reserved()


def test_a_decision_record_that_changed_a_block_is_refused_on_resume_before_the_lock(rig, monkeypatch):
    assert rig.run(rig.two_stages()) == 0
    notes = len(rig.ledger().notes)
    real_spec(rig.tmp, lambda d: d['decisions']['C9'].update(chosen='other'))
    locks = []
    real_enter = supervisor._RunLock.__enter__
    monkeypatch.setattr(supervisor._RunLock, '__enter__', lambda self: (locks.append(1), real_enter(self))[1])
    rig.log.clear()
    assert rig.run(rig.two_stages()) == 2
    assert 'spec' in rig.text() and len(rig.ledger().notes) == notes and not (rig.run_dir / supervisor.LOCK_FILE).exists()
    assert locks == [] and rig.log == ['native', 'e6']                                  # the lock was never taken, no identity was captured: the refusal came first


def test_a_new_top_level_section_and_a_new_block_in_the_record_are_accepted_on_resume(rig):
    stages = [rig.stage('train', 'make_output', 'train/best.bin', 'w', outputs=['train/best.bin']),
              rig.stage('evaluate', 'make_output', 'evaluate/scores.bin', 's', inputs=['train/best.bin'], outputs=['evaluate/scores.bin'])]
    assert rig.run(stages[:1]) == 0
    real_spec(rig.tmp, lambda d: (d.update(S4b_time_base={'unit': 'ms'}), d['decisions'].update(D21={'rule': 'x'})))
    assert rig.run(stages) == 0                                                         # K5b ruling R1
    real_spec(rig.tmp, lambda d: d['G2_time_base'].update(unit='ms'))
    assert rig.run(stages) == 2


# ---- the lock ------------------------------------------------------------------------------------------------------------------------------
def test_the_run_lock_is_taken_before_the_ledger_is_created(rig, monkeypatch):
    seen = []
    real_create = run_ledger.Ledger.create.__func__

    def create(cls, run_dir, header):
        seen.append((Path(run_dir) / supervisor.LOCK_FILE).exists())
        return real_create(cls, run_dir, header)
    monkeypatch.setattr(run_ledger.Ledger, 'create', classmethod(create))
    assert rig.run(rig.two_stages()) == 0
    assert seen == [True]


def test_a_lock_that_is_held_makes_the_start_a_refusal_and_changes_nothing(rig):
    rig.run_dir.mkdir()
    lock = put(rig.run_dir / supervisor.LOCK_FILE, b'pid=1\n')
    assert rig.run(rig.two_stages()) == 2
    assert 'lock' in rig.text() and 'delete' in rig.text() and str(rig.run_dir) not in rig.text()
    assert sorted(p.name for p in rig.run_dir.iterdir()) == [supervisor.LOCK_FILE] and lock.read_bytes() == b'pid=1\n'


def test_a_run_folder_that_changed_between_the_checks_and_the_lock_is_refused(rig, monkeypatch):
    """The gates ran for a new run; by the time the lock is taken a ledger has appeared (another call finished meanwhile): the gates no longer apply."""
    calls = []

    def exists(run_dir):
        calls.append(1)
        return len(calls) > 1                                                             # first (before the lock): no run; second (under it): a run
    monkeypatch.setattr(supervisor, '_exists', exists)
    assert rig.run(rig.two_stages()) == 2
    assert 'changed while the run was starting' in rig.text() and not (rig.run_dir / supervisor.LOCK_FILE).exists()
    assert not (rig.run_dir / 'HEADER.json').exists()


def test_the_lock_is_released_after_a_failure_and_after_a_refusal(rig):
    assert rig.run([rig.stage('train', 'raise_runtime_error', 'x')]) == 3
    assert not (rig.run_dir / supervisor.LOCK_FILE).exists()
    assert rig.run([rig.stage('train', 'raise_gate_refusal')]) == 2
    assert not (rig.run_dir / supervisor.LOCK_FILE).exists()


def test_two_concurrent_starts_of_one_run_folder_one_wins_and_the_other_gets_2(rig, tmp_path):
    other_stream = io.StringIO()
    stages = [rig.stage('train', 'sleep_until_file', 'go.marker', 'train/best.bin', outputs=['train/best.bin'])]
    results = {}

    def first():
        results['first'] = rig.run(stages)
    thread = threading.Thread(target=first)
    thread.start()
    try:
        wait_for(rig.run_dir / 'started.marker')                                        # the first start is inside its stage: it holds the lock
        assert (rig.run_dir / supervisor.LOCK_FILE).exists()
        second = supervisor.run(rig.profile, rig.run_dir, **rig.kwargs(stages, stream=other_stream))
        assert second == 2 and 'lock' in other_stream.getvalue()
    finally:
        (rig.run_dir / 'go.marker').write_text('go', encoding='utf-8')
        thread.join(60)
    assert results == {'first': 0}
    assert rig.ledger().stages == (('train', 'finished'),)


# ---- the existing run: identity captured again, equal, then the start gate; the final check ----------------------------------------------------
def test_resume_recaptures_identity_requires_equality_then_runs_start_gate(rig):
    assert rig.run(rig.two_stages()[:1]) == 0
    rig.log.clear()
    assert rig.run(rig.two_stages()) == 0
    # native, the start E6; then under the lock: capture again (equal to the saved identity), the start gate, resume; then for the one stage that is
    # left the E6 check and the identity check; at last the final resume
    assert rig.log == ['native', 'e6', 'capture', 'start_gate', 'resume', 'e6', 'verify', 'resume']


def test_an_identity_that_differs_on_resume_is_2_and_neither_the_start_gate_nor_a_stage_runs(rig):
    assert rig.run(rig.two_stages()[:1]) == 0
    rig.log.clear()
    rig.identity = Identity(commit='ab' * 20, files={'c_ept/run/supervisor.py': 'ee' * 32}, protected=())
    assert rig.run(rig.two_stages()) == 2
    assert 'capture' in rig.log and 'start_gate' not in rig.log and 'resume' not in rig.log
    assert 'c_ept/run/supervisor.py' in rig.text()                                       # the differing path is named
    assert rig.ledger().stages == (('train', 'finished'),) and not (rig.run_dir / supervisor.LOCK_FILE).exists()
    rig.identity = Identity(commit='cd' * 20, files={'c_ept/run/supervisor.py': 'cd' * 32}, protected=())
    assert rig.run(rig.two_stages()) == 2                                                # another commit


def test_the_start_gate_refusing_on_resume_stops_before_the_resume(rig, monkeypatch):
    assert rig.run(rig.two_stages()[:1]) == 0
    rig.log.clear()
    monkeypatch.setattr(rig, 'start_gate', lambda repo, identity, **kw: (_ for _ in ()).throw(run_identity.IdentityChanged('dirty', ['c_ept/x.py'])))
    assert rig.run(rig.two_stages()) == 2
    assert 'resume' not in rig.log and rig.ledger().stages == (('train', 'finished'),)


def test_the_identity_is_checked_before_every_stage_and_a_change_stops_the_run(rig, monkeypatch):
    calls = []

    def verify(saved, *, repo=None):
        calls.append(1)
        if len(calls) == 2:
            raise run_identity.IdentityChanged('imported code changed since the identity was bound', ['c_ept/run/ledger.py'])
    monkeypatch.setattr(rig, 'verify', verify)
    assert rig.run(rig.two_stages()) == 2
    assert rig.ledger().stages == (('train', 'finished'),) and 'ledger.py' in rig.text()
    notes = [n for n in rig.ledger().notes if n.kind == 'stop']
    assert len(notes) == 1 and dict(notes[0].record)['reason'] == 'refused' and notes[0].stage == 'evaluate'


def test_a_final_resume_check_runs_after_the_last_stage_and_finds_a_tamper(rig, monkeypatch):
    real_finish = run_ledger.Ledger.finish

    def finish(self, stage, outputs):
        digest = real_finish(self, stage, outputs)
        if stage == 'evaluate':
            (rig.run_dir / 'train' / 'best.bin').write_bytes(b'tampered after the stage')
        return digest
    monkeypatch.setattr(run_ledger.Ledger, 'finish', finish)
    assert rig.run(rig.two_stages()) == 2                                                # Tampered
    assert rig.log[-1] == 'resume' and 'changed' in rig.text()


def test_an_os_error_while_hashing_is_3_and_the_path_is_scrubbed(rig, monkeypatch):
    real = run_io.sha_file

    def sha_file(path, *, check=None):
        if Path(path).name == 'best.bin':
            raise PermissionError(13, 'Permission denied', f'D:/data/{KEY}/best.bin')
        return real(path, check=check)
    monkeypatch.setattr(run_io, 'sha_file', sha_file)
    assert rig.run(rig.two_stages()) == 3
    assert KEY not in rig.text() and GAME_ID not in rig.text() and '<key>' in rig.text()
    stop = [n for n in rig.ledger().notes if n.kind == 'stop']
    assert KEY not in json.dumps([dict(n.record) for n in stop]) and dict(stop[0].record)['reason'] == 'failure'


# ---- a budget that runs out during the run -------------------------------------------------------------------------------------------------
def test_a_budget_stop_during_a_stage_is_3_with_the_reason_in_the_ledger_and_the_tree_is_dead(rig):
    stages = [rig.stage('train', 'spawn_and_sleep', 'ready.json', inputs=[rig.profile_path])]

    def raise_the_memory():
        wait_for(rig.run_dir / 'ready.json')
        rig.over_memory.set()
    watcher = threading.Thread(target=raise_the_memory)
    watcher.start()
    started = time.monotonic()
    code = rig.run(stages)
    watcher.join(30)
    assert code == 3 and time.monotonic() - started < 60
    pids = json.loads((rig.run_dir / 'ready.json').read_text(encoding='utf-8'))
    assert gone(pids['child']) and gone(pids['grandchild'])
    ledger = rig.ledger()
    assert ledger.stages == (('train', 'begun'),)
    stop = [n for n in ledger.notes if n.kind == 'stop']
    assert len(stop) == 1 and stop[0].stage == 'train' and dict(stop[0].record)['reason'] == 'budget' and 'memory' in dict(stop[0].record)['detail']
    assert not (rig.run_dir / supervisor.LOCK_FILE).exists()
    # the run can be started again with a budget that holds
    rig.over_memory.clear()
    assert rig.run([rig.stage('train', 'make_output', 'train/best.bin', 'w', inputs=[rig.profile_path], outputs=['train/best.bin'])]) == 0


def test_a_budget_stop_while_the_ledger_hashes_is_3_too(rig):
    class Clock:
        now = 0.0
    clock = Clock()
    budget = guards.Budget(100.0, 2 ** 30, clock=lambda: clock.now, measure=rig.measure)
    stages = [rig.stage('train', 'make_output', 'train/best.bin', 'w', inputs=[rig.profile_path], outputs=['train/best.bin'])]
    real = run_io.sha_file
    state = {'calls': 0}

    def sha_file(path, *, check=None):
        state['calls'] += 1
        if state['calls'] == 6:
            clock.now = 1000.0                                                            # the clock passes the budget while the ledger hashes a file
        return real(path, check=check)
    run_io.sha_file = sha_file
    try:
        assert rig.run(stages, budget=budget) == 3
    finally:
        run_io.sha_file = real
    stop = [n for n in rig.ledger().notes if n.kind == 'stop']
    assert len(stop) == 1 and dict(stop[0].record)['reason'] == 'budget'
    assert rig.ledger().stages == () and not (rig.run_dir / 'calls.txt').exists()          # the stop came from the hashing inside ``begin``: the stage never began


# ---- the children's environment ---------------------------------------------------------------------------------------------------------------
def test_a_cpu_child_sees_an_empty_device_list_and_the_runners_thread_count(rig):
    environ = dict(os.environ, CODEX_TEST_MARK='1', PYTHONPATH=str(rig.tmp), GIT_DIR=str(rig.tmp), CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='16')
    stages = [rig.stage('train', 'report_environment', 'report.json', outputs=['report.json'])]
    assert rig.run(stages, environ=environ, cuda_visible_devices='', threads=1) == 0
    seen = json.loads((rig.run_dir / 'report.json').read_text(encoding='utf-8'))
    assert seen['CUDA_VISIBLE_DEVICES'] == '' and seen['OMP_NUM_THREADS'] == seen['MKL_NUM_THREADS'] == seen['OPENBLAS_NUM_THREADS'] == '1'
    assert seen['CODEX_TEST_MARK'] is None and seen['PYTHONPATH'] is None and seen['GIT_DIR'] is None
    assert seen['KMP_DUPLICATE_LIB_OK'] == 'FALSE' and seen['MKL_THREADING_LAYER'] == 'SEQUENTIAL' and seen['numpy_loaded'] is False and seen['isolated'] == 1
    # K5b review I1: ``-I`` ignores the PYTHON* variables of the child settings, so the interpreter flags say the same things (UTF-8, no bytecode, unbuffered)
    assert seen['utf8_mode'] == 1 and seen['dont_write_bytecode'] is True and seen['write_through'] is True and seen['stdout_encoding'] == 'utf-8'


def pyc_snapshot():
    return {str(p): p.stat().st_mtime_ns for folder in ('run', 'tests') for p in (REPO / 'c_ept' / folder).rglob('*.pyc')}


def test_a_child_writes_no_bytecode_into_the_repository(rig):
    before = pyc_snapshot()
    stages = [rig.stage('train', 'make_output', 'train/best.bin', 'w', outputs=['train/best.bin'])]
    assert rig.run(stages) == 0
    assert pyc_snapshot() == before


def test_the_output_a_child_wrote_before_it_was_killed_reaches_the_log_because_the_child_is_unbuffered(rig):
    stages = [rig.stage('train', 'print_then_sleep', 'the last words before the kill', 'ready.json')]

    def raise_the_memory():
        wait_for(rig.run_dir / 'ready.json')
        rig.over_memory.set()
    watcher = threading.Thread(target=raise_the_memory)
    watcher.start()
    assert rig.run(stages) == 3
    watcher.join(30)
    assert 'the last words before the kill' in (rig.run_dir / supervisor.LOG_FOLDER / 'train.log').read_text(encoding='utf-8')


def test_a_gpu_child_gets_the_device_list_and_threads_that_the_runner_names(rig):
    stages = [rig.stage('train', 'report_environment', 'report.json', outputs=['report.json'])]
    assert rig.run(stages, cuda_visible_devices='3', threads=4) == 0
    seen = json.loads((rig.run_dir / 'report.json').read_text(encoding='utf-8'))
    assert seen['CUDA_VISIBLE_DEVICES'] == '3' and seen['OMP_NUM_THREADS'] == '4'


# ---- scrubbed tracebacks --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('function, error', [('raise_runtime_error', 'RuntimeError'), ('raise_key_error', 'KeyError')])
def test_exit_codes_and_scrubbed_traceback_for_runtimeerror_and_keyerror(rig, function, error):
    """A stage that raises RuntimeError or KeyError (with a match key in it) ends the run with 3; the key and the game id are in none of the output, the
    stage log, the ledger notes."""
    assert rig.run([rig.stage('train', function, KEY)]) == 3
    assert error in rig.text() and '<key>' in rig.text() and KEY not in rig.text() and GAME_ID not in rig.text()
    log = (rig.run_dir / supervisor.LOG_FOLDER / 'train.log').read_text(encoding='utf-8')
    assert error in log and KEY not in log and GAME_ID not in log
    everything = ''.join(p.read_text(encoding='utf-8', errors='replace') for p in rig.run_dir.rglob('*') if p.is_file())
    assert KEY not in everything and GAME_ID not in everything


@pytest.mark.parametrize('error', [RuntimeError(f'boom while reading {KEY}'), KeyError(KEY), ValueError(f'bad {GAME_ID}'), OSError(5, 'x', f'D:/{KEY}/a.json')])
def test_an_error_in_the_supervisor_itself_is_3_and_scrubbed(rig, error):
    def e6():
        raise error
    assert rig.run(rig.two_stages(), e6=e6) == 3
    assert KEY not in rig.text() and GAME_ID not in rig.text() and type(error).__name__ in rig.text()
    assert rig.nothing_was_reserved()


def test_a_refusal_message_is_scrubbed_too(rig):
    def e6():
        raise guards.GateRefused('e6', f'a server for {KEY} runs')
    assert rig.run(rig.two_stages(), e6=e6) == 2
    assert KEY not in rig.text() and '<key>' in rig.text()


def test_the_error_of_a_referenced_profile_that_cannot_be_read_is_3_and_its_path_is_scrubbed(tmp_path, monkeypatch):
    """K3 re-check n2: a measure profile names a train profile; an OS error while reading it carries the path. The guard around ``profile.load`` scrubs it."""
    folder = tmp_path / 'profiles'
    folder.mkdir()
    train = put(folder / 'train.json', PROFILE)
    measure = put(folder / 'measure.json', ('{\n  "kind": "measure",\n  "steps": 3,\n  "train_profile": "train.json",\n  "train_sha256": "%s"\n}\n'
                                            % sha(PROFILE)).encode())
    assert run_profile.load(measure).kind == 'measure'
    real = run_profile._read_file
    calls = []

    def read_file(path):
        calls.append(path)
        if len(calls) == 2:                                                              # the referenced train profile
            raise PermissionError(13, 'Permission denied', f'D:/profiles/{KEY}/train.json')
        return real(path)
    monkeypatch.setattr(run_profile, '_read_file', read_file)
    stream = io.StringIO()
    code = supervisor.main_guard(run_profile.load, measure, scrub_keys=[KEY], stream=stream)
    assert code == 3 and KEY not in stream.getvalue() and GAME_ID not in stream.getvalue() and 'PermissionError' in stream.getvalue() and '<key>' in stream.getvalue()
    assert train.exists()


def test_main_guard_returns_the_result_of_the_function_or_the_code_of_its_error():
    stream = io.StringIO()
    assert supervisor.main_guard(lambda: None, stream=stream) == 0 and supervisor.main_guard(lambda: 5, stream=stream) == 5
    assert supervisor.main_guard(lambda: (_ for _ in ()).throw(run_profile.ProfileRefused('unknown field')), stream=stream) == 2
    assert 'ProfileRefused' in stream.getvalue() and 'Traceback' not in stream.getvalue()            # a refusal is one line; a failure has its traceback
    assert supervisor.main_guard(lambda: 1 / 0, stream=stream) == 3 and 'Traceback' in stream.getvalue() and 'ZeroDivisionError' in stream.getvalue()
    with pytest.raises(KeyboardInterrupt):
        supervisor.main_guard(lambda: (_ for _ in ()).throw(KeyboardInterrupt()), stream=stream)       # not swallowed


def test_describe_scrubs_the_chain_of_a_traceback():
    try:
        try:
            raise KeyError(KEY)
        except KeyError as inner:
            raise RuntimeError(f'while handling {GAME_ID}') from inner
    except RuntimeError as outer:
        text = supervisor.describe(outer, [KEY])
    assert KEY not in text and GAME_ID not in text and 'Traceback' in text and '<key>' in text and '<id>' in text


# ---- K5b review fixes --------------------------------------------------------------------------------------------------------------------------
class FakeClock:
    now = 0.0


def test_a_stage_may_declare_a_wall_cap_and_nothing_else_is_accepted():
    assert Stage('train', 'a.b:c').wall_cap_s is None and Stage('train', 'a.b:c', wall_cap_s=60).wall_cap_s == 60
    for bad in (-1, True, float('nan'), float('inf'), '60', [1]):
        with pytest.raises(supervisor.SupervisorRefused):
            Stage('train', 'a.b:c', wall_cap_s=bad)


def test_the_wall_cap_of_a_stage_is_checked_against_the_budget_that_is_left_before_its_begin(rig):
    """K5b review I2: earlier stages use wall time, so ``W >= C + F`` at the start does not keep the training cap first. Before ``begin`` of a stage that
    declares a cap, the budget that is left must hold the cap plus the margin; else a budget refusal (2) before any work of that stage."""
    clock = FakeClock()
    budget = guards.Budget(100.0, 2 ** 30, clock=lambda: clock.now, measure=rig.measure)
    calls = []

    def verify(saved, *, repo=None):
        calls.append(1)
        if len(calls) == 2:
            clock.now = 50.0                                                              # the first stage used 50 s of the 100 s
    stages = [rig.stage('train', 'make_output', 'train/best.bin', 'w', outputs=['train/best.bin']),
              rig.stage('evaluate', 'make_output', 'evaluate/scores.bin', 's', wall_cap_s=60.0, outputs=['evaluate/scores.bin'])]    # 60 + 5 > 100 - 50
    assert rig.run(stages, budget=budget, verify=verify) == 2
    ledger = rig.ledger()
    assert ledger.stages == (('train', 'finished'),)                                       # evaluate never began
    stop = [n for n in ledger.notes if n.kind == 'stop']
    assert len(stop) == 1 and stop[0].stage == 'evaluate' and dict(stop[0].record)['reason'] == 'refused' and 'budget' in dict(stop[0].record)['detail']
    assert 'budget' in rig.text() and not (rig.run_dir / 'evaluate').exists()
    # the same plan with the time left (100 - 20 >= 60 + 5) runs; the bound is inclusive
    clock.now = 0.0
    calls.clear()
    rig.run_dir = rig.tmp / 'second_run'

    def verify_fast(saved, *, repo=None):
        if len(calls) == 0:
            calls.append(1)
        else:
            clock.now = 35.0                                                              # exactly 100 - 35 = 65 = 60 + 5
    assert rig.run(stages, budget=guards.Budget(100.0, 2 ** 30, clock=lambda: clock.now, measure=rig.measure), verify=verify_fast) == 0


def test_the_docstring_states_when_the_training_cap_acts_first():
    assert 'always acts first' not in supervisor.__doc__ and 'wall_cap_s' in supervisor.__doc__ and 'only' in supervisor.__doc__


@pytest.mark.parametrize('output, truncated, expected', [
    (f'ey_{GAME_ID[2:]}/timeline.json\nnext line {KEY}\n'.encode(), True, 'next line <key>\n'),          # the cut fell inside a key: its fragment is dropped
    (f'{GAME_ID[4:]}\nnext line {KEY}\n'.encode(), True, 'next line <key>\n'),
    (f'no newline at all {KEY[3:]}'.encode(), True, ''),
    (f'first line {KEY}\nsecond\n'.encode(), False, 'first line <key>\nsecond\n'),                       # nothing was cut: nothing is dropped
    (b'', True, ''), (b'', False, '')])
def test_a_truncated_tail_drops_its_first_line_before_it_is_scrubbed(output, truncated, expected):
    text = supervisor.scrub_tail(output, truncated, [KEY])
    assert text == expected and KEY[3:] not in text and GAME_ID[4:] not in text


def test_run_owned_keeps_the_original_error_and_attaches_the_stop_failure(monkeypatch):
    class Halt(Exception):
        pass

    def check():
        raise Halt
    monkeypatch.setattr(supervisor, '_terminate', lambda job, child: (_ for _ in ()).throw(supervisor.OwnedProcessError('the tree could not be confirmed stopped')))
    with pytest.raises(Halt) as caught:
        owned('import time; time.sleep(30)', check=check)
    assert any('could not be confirmed' in note for note in caught.value.__notes__)


def test_a_failed_stop_does_not_change_the_reason_of_a_budget_stop(rig, monkeypatch):
    real = supervisor._terminate
    monkeypatch.setattr(supervisor, '_terminate', lambda job, child: (real(job, child), (_ for _ in ()).throw(supervisor.OwnedProcessError('unconfirmed')))[1])
    stages = [rig.stage('train', 'spawn_and_sleep', 'ready.json')]

    def raise_the_memory():
        wait_for(rig.run_dir / 'ready.json')
        rig.over_memory.set()
    watcher = threading.Thread(target=raise_the_memory)
    watcher.start()
    assert rig.run(stages) == 3
    watcher.join(30)
    stop = [n for n in rig.ledger().notes if n.kind == 'stop']
    assert dict(stop[0].record)['reason'] == 'budget'


def test_the_memory_is_sampled_at_most_every_quarter_of_a_second_but_the_clock_every_time():
    clock, samples = FakeClock(), []
    budget = guards.Budget(1000.0, 2 ** 30, clock=lambda: clock.now, measure=lambda: (samples.append(clock.now), 10 ** 6)[1])
    check = supervisor._stop_check(budget)
    assert supervisor.MEMORY_SAMPLE_INTERVAL_S == 0.25
    for _ in range(2000):
        check()
        clock.now += 0.001                                                                 # two seconds in 2000 calls
    assert 7 <= len(samples) <= 9 and samples[0] == 0.0                                    # the first call samples; then one per quarter second
    clock.now = 5000.0                                                                     # the wall budget is passed: refused at the very next call
    with pytest.raises(supervisor.BudgetStop, match='wall-clock'):
        check()


def test_a_memory_overrun_is_found_by_the_next_sample_after_the_interval():
    clock, state = FakeClock(), {'big': False}
    budget = guards.Budget(1000.0, 2 ** 30, clock=lambda: clock.now, measure=lambda: 2 ** 31 if state['big'] else 10 ** 6)
    check = supervisor._stop_check(budget)
    check()
    state['big'] = True
    clock.now = 0.1
    check()                                                                               # inside the interval: not sampled yet
    clock.now = 0.26
    with pytest.raises(supervisor.BudgetStop, match='memory'):
        check()


@pytest.mark.parametrize('inputs, outputs', [
    ([], ['logs/train.log']), ([], ['logs/sub/x.txt']), ([], ['LOGS/x.txt']), (['RUN.lock'], []), ([], ['run.LOCK']), (['logs'], []), ([], ['logs']),
])
def test_the_names_the_supervisor_owns_cannot_be_stage_inputs_or_outputs(rig, inputs, outputs):
    assert rig.run([Stage('train', f'{BODIES}:no_output', inputs=inputs, outputs=outputs)]) == 2
    assert rig.nothing_was_reserved()


def test_run_owned_names_the_function_it_follows_and_its_differences():
    source = SUPERVISOR_SOURCE.read_text(encoding='utf-8')
    assert '# Ref: tools/ept_comparison_controller.py' in source
    doc = supervisor.run_owned.__doc__
    assert 'reserve_seconds' in doc and 'left the process tree' in doc and 'Job' in doc


def test_a_key_in_the_output_of_a_child_killed_by_the_budget_reaches_neither_the_log_nor_a_ledger_record(rig):
    """K5b re-check: the stopped child's last output is logged (review I1), and it goes through the scrub like every other text."""
    stages = [rig.stage('train', 'print_then_sleep', f'reading {KEY} then {GAME_ID}', 'ready.json')]

    def raise_the_memory():
        wait_for(rig.run_dir / 'ready.json')
        rig.over_memory.set()
    watcher = threading.Thread(target=raise_the_memory)
    watcher.start()
    assert rig.run(stages) == 3
    watcher.join(30)
    log = (rig.run_dir / supervisor.LOG_FOLDER / 'train.log').read_text(encoding='utf-8')
    assert 'reading <key> then <id>' in log
    everything = ''.join(p.read_text(encoding='utf-8', errors='replace') for p in rig.run_dir.rglob('*') if p.is_file() and p.name != 'ready.json')
    assert KEY not in everything and GAME_ID not in everything and KEY not in rig.text() and GAME_ID not in rig.text()
    assert [n for n in rig.ledger().notes if n.kind == 'stop' and dict(n.record)['reason'] == 'budget']
