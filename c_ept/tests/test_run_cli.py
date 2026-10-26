"""Tests of the command line of the run kernel (``python -m c_ept.run.cli {train,evaluate,measure}``; S4b-prep K6c, decisions D10 P6, D11, D20; K6b review r4).

The command line takes a hash-locked profile and a run folder (``--profile``, ``--run-dir``), the inputs manifest, the decision record, the budget and the runner constants, and the text of the
author's approval (``--approval``, recorded as given in the ledger header), and runs the profile through the supervisor and the gates of the kernel (E6, the native-thread policy, the start gate
of the code identity, the budget). The exit codes are those of the supervisor: 0 finished, 2 a refusal (also a wrong command line), 3 a computation that did not finish. For the evaluation of
the roles DEV and TEST it refuses a stage body other than the default one.

These tests use the toy EPT world, short CPU children that see ``CUDA_VISIBLE_DEVICES=''`` and the seams that ``main`` takes for them (the injected gates of the supervisor, the toy loaders and
bodies, the match keys); the process entry passes none. No data, no GPU, no model server: a test that asks for the device ``cuda`` has the CUDA check and the supervisor replaced, so that no
run can start.

해석: 이 시험은 명령줄이 프로파일 한 장과 실행 폴더로 학습·평가·측정을 감독자 아래에서 돌리고, 승인 문구를 글자 그대로 원장 머리글에 남기며, 종료 코드 0·2·3을 지키고, DEV·TEST 평가에서
기본이 아닌 단계 본문을 거부하는지 장난감 자료와 CPU로 확인한다. GPU 실행은 하지 않는다.
"""
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from c_ept.run import cli, guards
from c_ept.run import evaluate as run_evaluate
from c_ept.run import io as run_io
from c_ept.run import ledger as run_ledger
from c_ept.run import measure, numeric, runners, supervisor
from c_ept.run.identity import Identity
from c_ept.tests.test_run_evaluate import SYNTHETIC, TOY_BODY, TOY_LOADER as TOY_EVAL_LOADER, eval_manifest_document, write_eval_manifest
from c_ept.tests.test_run_ledger import real_spec
from c_ept.tests.test_run_measure import CALL_BODY, STATS_BODY, write_measure
from c_ept.tests.test_run_runners import E6_RECORD, KEY, REPO, S4B, TOY_LOADER, manifest_document, reps_sha, s4b_scales, world, write_manifest   # noqa: F401

TRAIN_BODY = 'c_ept.tests.run_runner_bodies:train_stage'
APPROVAL = 'Approved by the author: run the toy on the CPU, "no GPU".\nSecond line with a backslash \\ and 한글 승인 '
IDENTITY = Identity(commit='ab' * 20, files={'c_ept/run/cli.py': 'cd' * 32}, protected=())
CPU_ARGUMENTS = ['--device', 'cpu', '--threads', '1', '--batch-size', '2', '--wall-hours', '3000']


@pytest.fixture(autouse=True)
def restore_torch_state(monkeypatch):
    """The runners change process-wide settings (threads, the numeric policy, the global random streams); give them back after each test."""
    from c_ept.tests.test_run_numeric import read_flags, write_flags
    threads, flags = torch.get_num_threads(), read_flags()
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8'))
    yield
    torch.set_num_threads(threads)
    write_flags(flags)


def gates(**over) -> dict:
    """The seams of ``main`` that stand for the machine of a test run: the gates of the supervisor (the working tree is not a clean checkout, the identity is made up, E6 is asked of no
    process list), a budget that reads a made-up memory, a short poll and the match key that the output must not show."""
    values = dict(budget=guards.Budget(10 ** 7, 2 ** 40, measure=lambda: 10 ** 6), e6=lambda: dict(E6_RECORD), capture=lambda modules, *, repo=None: IDENTITY,
                  verify=lambda saved, repo=None: None, start_gate=lambda repo, identity, **kwargs: None, poll_s=0.05, repo=REPO, scrub_keys=[KEY])
    values.update(over)
    return values


def run_cli(arguments, **seams):
    stream, out = io.StringIO(), io.StringIO()
    code = cli.main([str(a) for a in arguments], stream=stream, out=out, **seams)
    return code, stream.getvalue(), out.getvalue()


class Setup:
    """The files of one command line: a train profile (S4b settings, one epoch), the inputs manifest of the toy world, the decision record and the loss-scale file."""

    def __init__(self, folder, world, reps_sha, **train_fields):
        self.folder = Path(folder)
        self.train = self.folder / 'train.json'
        self.train.write_bytes((json.dumps({'kind': 'train', **{**S4B, 'epochs': 1}, **train_fields}, indent=2) + '\n').encode('utf-8'))
        self.scales = s4b_scales(self.folder, reps_sha)
        self.manifest = write_manifest(self.folder, manifest_document(world, self.scales))
        self.eval_manifest = write_eval_manifest(self.folder, eval_manifest_document(world, self.scales))
        self.spec = real_spec(self.folder)

    def common(self, run_dir, *, inputs=None, extra=()):
        return ['--run-dir', run_dir, '--inputs', inputs or self.manifest, '--spec', self.spec, *CPU_ARGUMENTS, *extra]

    def train_arguments(self, run_dir, *extra, profile=None):
        return ['train', '--profile', profile or self.train, *self.common(run_dir, extra=['--allowance', '1800,120,300', *extra])]

    def evaluate_profile(self, name, checkpoint, role='STOP', **fields):
        path = self.folder / f'{name}.json'
        path.write_bytes((json.dumps({'kind': 'evaluate', 'role': role, 'models': ['seed1'], 'checkpoints': {'seed1': str(checkpoint)}, **fields}, indent=2) + '\n').encode('utf-8'))
        return path


@pytest.fixture(scope='module')
def setup(tmp_path_factory, world, reps_sha):
    return Setup(tmp_path_factory.mktemp('cli_files'), world, reps_sha)


@pytest.fixture(scope='module')
def trained(setup, tmp_path_factory):
    """The train subcommand once, on the toy world: ``(run folder, exit code, standard error text, standard output text)``. The other tests read this run."""
    run_dir = tmp_path_factory.mktemp('cli_runs') / 'train_run'
    stream, out = io.StringIO(), io.StringIO()
    code = cli.main([str(a) for a in setup.train_arguments(run_dir, '--approval', APPROVAL)], stream=stream, out=out, **gates(loader=TOY_LOADER, bodies={'train': TRAIN_BODY}))
    return run_dir, code, stream.getvalue(), out.getvalue()


# ---- the three subcommands, end to end on the toy ----------------------------------------------------------------------------------------------------
def test_the_train_subcommand_runs_a_profile_through_the_supervisor_and_records_the_approval_as_given(trained, setup):
    run_dir, code, err, out = trained
    assert code == 0, err
    assert err == '' and 'train' in out and 'exit code 0' in out and KEY not in out
    ledger = run_ledger.Ledger(run_dir)
    assert ledger.stages == (('stats', 'finished'), ('train', 'finished'))
    runner = run_io.read_json(run_dir / run_ledger.HEADER_FILE)['runner']
    assert runner['approval'] == {'text': APPROVAL, 'sha256': hashlib.sha256(APPROVAL.encode('utf-8')).hexdigest()}
    assert runner['numeric'] == numeric.requested_record() and runner['device'] == 'cpu' and runner['threads'] == 1 and runner['batch_size'] == 2
    assert runner['training_stage']['allowance'] == {'epoch_s': 1800.0, 'checkpoint_write_s': 120.0, 'finalisation_s': 300.0}
    record = run_io.read_json(run_dir / runners.TRAIN_RECORD_FILE)
    assert record['numeric']['measured']['device'] == 'cpu' and record['profile_sha256'] == ledger.header.profile_sha256


def test_the_evaluate_subcommand_evaluates_the_stop_split_of_the_trained_run_and_may_take_another_stage_body(trained, setup, tmp_path):
    run_dir, code, err, out = trained
    assert code == 0, err
    profile = setup.evaluate_profile('stop', run_dir)
    seams = gates(loader=TOY_EVAL_LOADER, backend=SYNTHETIC, allow_custom=True)
    code, err, out = run_cli(['evaluate', '--profile', profile, *setup.common(tmp_path / 'evaluation', inputs=setup.eval_manifest, extra=['--approval', APPROVAL])], **seams)
    assert code == 0, err
    evaluation = tmp_path / 'evaluation'
    record = run_io.read_json(evaluation / run_evaluate.RECORD_FILE)
    assert record['role'] == 'STOP' and record['sources']['body'] == run_evaluate.body_record(run_evaluate.BODY) and record['numeric']['measured']['device'] == 'cpu'
    header = run_io.read_json(evaluation / run_ledger.HEADER_FILE)['runner']
    assert header['approval']['text'] == APPROVAL and header['numeric'] == numeric.requested_record()
    other = tmp_path / 'evaluation_with_toy_body'
    code, err, out = run_cli(['evaluate', '--profile', profile, *setup.common(other, inputs=setup.eval_manifest)], **{**seams, 'bodies': {run_evaluate.STAGE: TOY_BODY}})
    assert code == 0, err                                                                              # the role STOP takes another stage body (it is recorded, not refused)
    assert run_io.read_json(other / run_evaluate.RECORD_FILE)['sources']['body'] == run_evaluate.body_record(TOY_BODY)


def test_the_measure_subcommand_measures_two_identical_calls_and_prints_the_headline(setup, tmp_path):
    path, profile = write_measure(setup.folder, steps=2, repeat=2, name='cli_measure.json', train_name='cli_measure_train.json', epochs=1)
    run_dir = tmp_path / 'measurement'
    code, err, out = run_cli(['measure', '--profile', path, *setup.common(run_dir, extra=['--approval', APPROVAL])],
                             **gates(loader=TOY_LOADER, bodies={'stats': STATS_BODY, 'call': CALL_BODY}))
    assert code == 0, err
    document = run_io.read_json(run_dir / measure.MEASURE_FILE)
    assert document['identical'] is True and document['repeat'] == 2 and document['runner']['batch_size'] == 2
    assert 'identical' in out and 'measure' in out and 'exit code 0' in out and KEY not in out and err == ''
    assert run_io.read_json(run_dir / run_ledger.HEADER_FILE)['runner']['approval']['text'] == APPROVAL


# ---- the refusals and the exit codes ------------------------------------------------------------------------------------------------------------------
def test_a_wrong_command_line_is_exit_2_with_the_usage_and_no_traceback(setup, tmp_path):
    run_dir = tmp_path / 'never'
    cases = [[], ['train'], ['launch', '--profile', 'p', '--run-dir', run_dir], ['train', '--run-dir', run_dir], ['train', '--profile', setup.train],
             ['train', '--profile', setup.train, '--run-dir', run_dir, '--inputs', setup.manifest],                                                        # no wall budget
             ['train', '--profile', setup.train, '--run-dir', run_dir, '--inputs', setup.manifest, '--wall-hours', '0'],
             ['train', '--profile', setup.train, '--run-dir', run_dir, '--inputs', setup.manifest, '--wall-hours', '-5'],
             ['train', '--profile', setup.train, '--run-dir', run_dir, '--inputs', setup.manifest, '--wall-hours', 'many'],
             ['train', '--profile', setup.train, '--run-dir', run_dir, '--inputs', setup.manifest, '--wall-hours', 'nan'],
             ['train', '--profile', setup.train, '--run-dir', run_dir, '--inputs', setup.manifest, '--wall-hours', '5', '--device', 'tpu'],
             ['train', '--profile', setup.train, '--run-dir', run_dir, '--inputs', setup.manifest, '--wall-hours', '5', '--no-such-option'],
             ['train', '--profile', setup.train, '--run-dir', run_dir, '--inputs', setup.manifest, '--wall-hours', '5', '--allowance', '1,2']]
    for arguments in cases:
        code, err, out = run_cli(arguments, **gates())
        assert code == 2 and 'usage' in err.lower() and 'Traceback' not in err, (arguments, err)
    assert not run_dir.exists()


def test_help_is_printed_to_the_output_and_is_exit_0(tmp_path):
    code, err, out = run_cli(['--help'])
    assert code == 0 and err == '' and all(word in out for word in ('train', 'evaluate', 'measure', '--profile', '--run-dir', '--approval', '--inputs', '--wall-hours'))


def test_a_system_exit_that_carries_a_message_instead_of_a_number_is_a_refusal_and_not_a_success(monkeypatch):
    def stops(argv):
        raise SystemExit('a message, not a number')
    monkeypatch.setattr(guards, 'native_policy', lambda: None)
    monkeypatch.setattr(cli, 'build_parser', lambda stream, out: SimpleNamespace(parse_args=stops))
    assert run_cli(['train'])[0] == 2


def test_the_native_thread_policy_is_the_first_thing_that_main_does(monkeypatch):
    """The runners import numpy and torch; the policy of the native threads must be applied before. ``main`` applies it before it parses the line (and so before any import of the runners)."""
    calls = []
    monkeypatch.setattr(guards, 'native_policy', lambda: calls.append('native_policy'))
    real = cli.build_parser

    def spy(stream, out):
        calls.append('build_parser')
        return real(stream, out)
    monkeypatch.setattr(cli, 'build_parser', spy)
    assert run_cli(['--help'])[0] == 0 and calls == ['native_policy', 'build_parser']
    def refuses():
        raise guards.GateRefused('native', 'a native module was imported without the policy (test)')
    monkeypatch.setattr(guards, 'native_policy', refuses)
    code, err, out = run_cli(['--help'])
    assert code == 2 and 'native' in err                                                                  # a policy that cannot hold is a refusal, before anything is parsed


def test_a_profile_of_another_kind_than_the_subcommand_is_exit_2_before_anything_is_reserved(setup, trained, tmp_path):
    run_dir = tmp_path / 'never'
    path, _ = write_measure(setup.folder, name='kind_measure.json', train_name='kind_train.json')
    code, err, out = run_cli(['train', '--profile', path, *setup.common(run_dir, extra=['--allowance', '1,1,1'])], **gates(loader=TOY_LOADER))
    assert code == 2 and 'train profile' in err and not run_dir.exists()
    code, err, out = run_cli(['measure', '--profile', setup.train, *setup.common(run_dir)], **gates(loader=TOY_LOADER))
    assert code == 2 and 'measure profile' in err and not run_dir.exists()
    code, err, out = run_cli(['evaluate', '--profile', setup.train, *setup.common(run_dir, inputs=setup.eval_manifest)], **gates(loader=TOY_EVAL_LOADER, backend=SYNTHETIC))
    assert code == 2 and 'evaluate profile' in err and not run_dir.exists()


def test_an_option_that_the_command_does_not_read_is_refused_instead_of_ignored(setup, trained, tmp_path):
    run_dir = tmp_path / 'never'
    cases = [(['train', '--profile', setup.train, *setup.common(run_dir, extra=['--allowance', '1,1,1', '--access-dir', tmp_path]), ], 'access-dir'),
             (['train', '--profile', setup.train, *setup.common(run_dir, extra=['--allowance', '1,1,1', '--lock', tmp_path]), ], 'lock'),
             (['train', '--profile', setup.train, *setup.common(run_dir, extra=['--allowance', '1,1,1', '--rehearsal-run', tmp_path]), ], 'rehearsal-run'),
             (['measure', '--profile', setup.train, *setup.common(run_dir, extra=['--allowance', '1,1,1'])], 'allowance'),
             (['evaluate', '--profile', setup.evaluate_profile('options', trained[0]), *setup.common(run_dir, inputs=setup.eval_manifest, extra=['--allowance', '1,1,1'])], 'allowance')]
    for arguments, option in cases:
        code, err, out = run_cli(arguments, **gates(loader=TOY_LOADER))
        assert code == 2 and f'--{option}' in err and 'not by' in err and not run_dir.exists(), (option, err)


def test_a_gpu_run_needs_the_text_of_the_approval_and_a_cpu_run_does_not(setup, tmp_path, monkeypatch):
    """The device ``cuda`` is the default: a GPU run is an approved run. Here the CUDA check and the supervisor are replaced, so that nothing can start."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    def never(*args, **kwargs):
        raise AssertionError('a supervised run was started')
    monkeypatch.setattr(supervisor, 'run', never)
    arguments = ['train', '--profile', setup.train, '--run-dir', tmp_path / 'gpu', '--inputs', setup.manifest, '--spec', setup.spec, '--wall-hours', '3000', '--allowance', '1800,120,300']
    for approval in ([], ['--approval', ''], ['--approval', '   ']):
        code, err, out = run_cli([*arguments, *approval], **gates(loader=TOY_LOADER))
        assert code == 2 and 'approval' in err and 'GPU' in err and not (tmp_path / 'gpu').exists(), (approval, err)
    code, err, out = run_cli([*arguments, '--approval', 'go'], **gates(loader=TOY_LOADER))                                # an approval is not the machine: no CUDA here, so the runner refuses
    assert code == 2 and 'CUDA' in err and not (tmp_path / 'gpu').exists()


def test_the_allowance_is_required_when_the_profile_ends_at_the_wall_cap_with_its_best_stop(setup, tmp_path):
    run_dir = tmp_path / 'never'
    arguments = ['train', '--profile', setup.train, *setup.common(run_dir)]
    code, err, out = run_cli(arguments, **gates(loader=TOY_LOADER))
    assert code == 2 and 'allowance' in err and not run_dir.exists()


@pytest.mark.parametrize('role', ['DEV', 'TEST'])
def test_a_stage_body_other_than_the_default_is_refused_for_the_roles_dev_and_test(setup, trained, tmp_path, role):
    run_dir, code, err, out = trained
    assert code == 0, err
    fields = {'holdout_lock_sha256': 'ab' * 32} if role == 'TEST' else {}
    profile = setup.evaluate_profile(f'body_{role}', run_dir, role=role, **fields)
    access = run_evaluate.AccessLedger.create(tmp_path / 'access')
    target = tmp_path / 'never'
    extra = ['--access-dir', access.folder, *(['--lock', tmp_path / 'lock.json', '--rehearsal-run', tmp_path / 'rehearsal'] if role == 'TEST' else [])]
    seams = gates(loader=TOY_EVAL_LOADER, backend=SYNTHETIC)
    for bodies in ({run_evaluate.STAGE: TOY_BODY}, {run_evaluate.STAGE: lambda run_dir_, *texts: None}):
        code, err, out = run_cli(['evaluate', '--profile', profile, *setup.common(target, inputs=setup.eval_manifest, extra=extra)], **seams, bodies=bodies)
        assert code == 2 and 'stage body' in err and role in err and not target.exists(), (bodies, err)
    assert [e['event'] for e in access.entries()] == ['created']                                              # nothing was logged, nothing was opened


def test_the_default_stage_body_is_accepted_for_dev_whether_it_is_named_or_not(setup, trained, tmp_path):
    run_dir, code, err, out = trained
    assert code == 0, err
    access = run_evaluate.AccessLedger.create(tmp_path / 'access')
    profile = setup.evaluate_profile('dev', run_dir, role='DEV')
    seams = gates(loader=TOY_EVAL_LOADER, backend=SYNTHETIC, allow_custom=True)
    target = tmp_path / 'dev_run'
    code, err, out = run_cli(['evaluate', '--profile', profile, *setup.common(target, inputs=setup.eval_manifest, extra=['--access-dir', access.folder])], **seams)
    assert code == 0, err
    entries = access.entries()
    assert [e['event'] for e in entries] == ['created', 'dev_access'] and entries[1]['record']['sources']['body'] == run_evaluate.body_record(run_evaluate.BODY)
    named = tmp_path / 'dev_named'
    code, err, out = run_cli(['evaluate', '--profile', profile, *setup.common(named, inputs=setup.eval_manifest, extra=['--access-dir', access.folder])], **seams,
                             bodies={run_evaluate.STAGE: run_evaluate.BODY, 'train': 'x.y:z'})                      # the default body named, and a body of another stage that is none of the evaluation's
    assert code == 0, err
    assert [e['event'] for e in access.entries()].count('dev_access') == 2


def test_a_gate_that_refuses_is_exit_2_and_a_stage_that_fails_is_exit_3(setup, tmp_path):
    from c_ept.run.guards import GateRefused

    def server_running():
        raise GateRefused('e6', 'a model server is running (test)')
    run_dir = tmp_path / 'gate'
    code, err, out = run_cli(setup.train_arguments(run_dir), **gates(loader=TOY_LOADER, bodies={'train': TRAIN_BODY}, e6=server_running))
    assert code == 2 and 'model server' in err and not run_dir.exists() and 'exit code 2' in out
    failing = tmp_path / 'failing'
    code, err, out = run_cli(setup.train_arguments(failing), **gates(loader=TOY_LOADER, bodies={'train': 'c_ept.tests.run_runner_bodies:no_such_body'}))
    assert code == 3 and 'exit code 3' in out and run_ledger.Ledger(failing).stages == (('stats', 'finished'), ('train', 'begun'))


def test_the_text_of_an_error_holds_no_match_key_and_a_runner_constant_out_of_range_is_exit_2(setup, tmp_path):
    arguments = setup.train_arguments(tmp_path / 'absent')
    arguments[arguments.index('--inputs') + 1] = str(tmp_path / f'absent_{KEY}.json')                      # a manifest that is not there: the message of the refusal names the file
    code, err, out = run_cli(arguments, **gates(loader=TOY_LOADER))
    assert code == 2 and KEY not in err and KEY not in out and not (tmp_path / 'absent').exists()
    for option, bad in (('--threads', '0'), ('--batch-size', '0'), ('--threads', '5000')):
        arguments = setup.train_arguments(tmp_path / 'constants')
        arguments[arguments.index(option) + 1] = bad
        code, err, out = run_cli(arguments, **gates(loader=TOY_LOADER, bodies={'train': TRAIN_BODY}))
        assert code == 2 and not (tmp_path / 'constants').exists() and 'Traceback' not in err, (option, bad, err)


# ---- the match keys, the budget and the process entry ----------------------------------------------------------------------------------------------
def test_the_match_keys_come_from_the_s1_store_of_the_manifest_and_a_store_that_cannot_give_them_refuses_the_start(setup, monkeypatch):
    from c_ept.data import store_core
    import numpy as np

    class FakeStore:
        def __init__(self, path):
            self.path = path

        def array(self, name):
            assert name == 'matches/key'
            return np.array([b'KR_1', b'', b'EUW1_22'], dtype='S8')
    monkeypatch.setattr(store_core, 'Store', FakeStore)
    assert cli.match_keys(setup.manifest) == ['KR_1', 'EUW1_22']

    class NoKeys(FakeStore):
        def array(self, name):
            raise KeyError(name)
    monkeypatch.setattr(store_core, 'Store', NoKeys)
    with pytest.raises(run_io.IORefused, match='scrub'):
        cli.match_keys(setup.manifest)
    code, err, out = run_cli(setup.train_arguments(setup.folder / 'unscrubbed'), **{k: v for k, v in gates(loader=TOY_LOADER).items() if k != 'scrub_keys'})
    assert code == 2 and 'scrub' in err and not (setup.folder / 'unscrubbed').exists()


def test_the_budget_is_made_from_the_wall_hours_and_the_memory_and_a_budget_too_small_for_the_stage_is_refused(setup, tmp_path):
    budget = cli.make_budget(2.5, 40.0)
    assert (budget.wall_s, budget.memory_bytes) == (9000.0, 40 * 2 ** 30)
    for bad in ((0, 40), (-1, 40), (float('nan'), 40), (float('inf'), 40), (1, 0), (1, -2)):
        with pytest.raises(run_io.IORefused):
            cli.make_budget(*bad)
    run_dir = tmp_path / 'short'
    arguments = setup.train_arguments(run_dir)
    arguments[arguments.index('--wall-hours') + 1] = '1'
    code, err, out = run_cli(arguments, **{k: v for k, v in gates(loader=TOY_LOADER, bodies={'train': TRAIN_BODY}).items() if k != 'budget'})
    assert code == 2 and 'budget' in err and not run_dir.exists()                                          # 1 h cannot hold the 24 h training cap and the margin


def test_the_process_entry_prints_the_usage_without_importing_numpy_or_torch_and_a_bad_line_is_exit_2():
    """``python -m c_ept.run.cli`` applies the native-thread policy before the first native import: the module imports the standard library and the light kernel modules only."""
    code = ('import sys\n'
            'from c_ept.run import cli\n'
            'assert "numpy" not in sys.modules and "torch" not in sys.modules, sorted(m for m in sys.modules if m in ("numpy", "torch"))\n'
            'print("light")\n')
    light = subprocess.run([sys.executable, '-B', '-c', code], cwd=REPO, capture_output=True, text=True, encoding='utf-8', timeout=120)
    assert light.returncode == 0 and light.stdout.strip() == 'light', light.stderr
    helped = subprocess.run([sys.executable, '-B', '-m', 'c_ept.run.cli', '--help'], cwd=REPO, capture_output=True, text=True, encoding='utf-8', timeout=120)
    assert helped.returncode == 0 and 'measure' in helped.stdout and '--approval' in helped.stdout and helped.stderr == ''
    bad = subprocess.run([sys.executable, '-B', '-m', 'c_ept.run.cli', 'train'], cwd=REPO, capture_output=True, text=True, encoding='utf-8', timeout=120)
    assert bad.returncode == 2 and 'usage' in bad.stderr.lower() and 'Traceback' not in bad.stderr
