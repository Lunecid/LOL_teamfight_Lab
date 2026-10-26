"""EPT-T011 A5-F8: an unexpected exception keeps its class name and a key-scrubbed traceback; exit codes stay as they were.

Synthetic only: no real store, GPU or data. The sentinel match key below must never reach stderr.
"""
import gc
import json
from pathlib import Path

import numpy as np
import pytest

from c_ept.workflows.s4a_runtime import RuntimeRefused

KEY = 'TOYKEY_0042'        # a sentinel match key; 0042 is the game id inside it


@pytest.fixture(autouse=True)
def supported_runtime_test_environment(monkeypatch):
    """The same explicit disabled prerequisite that the other S4a tests set; production refusal is not bypassed."""
    monkeypatch.setenv('KMP_DUPLICATE_LIB_OK', 'FALSE')


def probe():
    return dict(running=False, matched=[], bridges_ignored=0, scanned=2, cmdline_unreadable=1, rule='fixture')


def toy_s1(root):
    """A complete store under ``root`` whose only array is ``matches/key`` (the sentinel): what the scrub reads and nothing else."""
    from c_ept.data.store_core import Store
    root = Path(root)
    store = Store(root)
    values = np.asarray([KEY.encode('ascii')], dtype='S24')
    array = store.allocate('matches/key', values.shape, 'S24')
    array[...] = values
    array.flush()
    del array
    store.finalize()
    store.maps.clear()      # Windows does not delete a folder that holds a mapped file
    gc.collect()
    return root


@pytest.fixture
def failing_main(tmp_path, monkeypatch):
    """``run(failure, s1=<path>)``: S4a main with every guard stubbed and the input loader raising ``failure``.

    Returns (return value, or the exception that escaped main; the PROCESS.json record of the run).
    """
    from c_ept.workflows import run_s4a as r, run_s1
    root = tmp_path / 'runs'
    root.mkdir()
    meta = {'paths': {'s1': str(root / 'source1/s1'), 's2': str(root / 'source2/s2'), 's3': str(root / 'source3')}}
    monkeypatch.setattr(r, 'APPROVED_RUNS_ROOT', root)
    monkeypatch.setattr(r, 'repository_state', lambda: dict(code_commit='a' * 40, code_dirty=False, repo_root=str(tmp_path)))
    monkeypatch.setattr(r, 'inspect_inputs', lambda args: meta)
    monkeypatch.setattr(r, 'provenance_record', lambda: {'spec_sha256': None, 'gpu_devices': None})   # no decision record or nvidia-smi here
    monkeypatch.setattr(run_s1, 'require_no_strata', lambda: probe())

    def run(failure, s1=None):
        def fail_loading(_meta):
            raise failure
        monkeypatch.setattr(r, '_load_inputs', fail_loading)
        argv = ['--s1', str(toy_s1(tmp_path / 's1') if s1 is None else s1), '--s2', 'toy2', '--s3', 'toy3',
                '--runs-root', str(root), '--catalog-db', str(tmp_path / 'catalog.sqlite'), '--sizes', 'S', '--devices', 'cpu']
        try:
            returned = r.main(argv)
        except BaseException as escaped:      # noqa: BLE001 - the test reports what escaped
            returned = escaped
        (process,) = root.glob('*/PROCESS.json')
        return returned, json.loads(process.read_text(encoding='utf-8'))
    return run


@pytest.mark.parametrize('failure_type', [ValueError, OSError])
def test_unexpected_error_shows_class_name_and_scrubbed_traceback_and_keeps_exit_code(failing_main, capsys, failure_type):
    returned, process = failing_main(failure_type(f'the match {KEY} is broken, and its id is 0042'))
    err = capsys.readouterr().err
    assert returned == 2                                                    # the refusal classification and exit code are unchanged
    assert err.startswith('S4a refused:')
    assert failure_type.__name__ in err and 'Traceback' in err and 'is broken' in err
    assert 'TOYKEY' not in err and '<key>' in err and 'its id is <id>' in err
    assert process['state'] == 'failed' and process['exit_code'] == returned
    assert process['exception_type'] == failure_type.__name__


def test_guard_refusal_keeps_its_one_line_message_and_gets_no_traceback(failing_main, capsys):
    returned, process = failing_main(RuntimeRefused('synthetic guard refusal'))
    assert returned == 2 and capsys.readouterr().err == 'S4a refused: synthetic guard refusal\n'
    assert process['exception_type'] == 'RuntimeRefused' and process['exit_code'] == 2


def test_error_outside_the_refusal_classes_still_escapes_as_exit_1_and_is_recorded(failing_main):
    escaped, process = failing_main(RuntimeError('synthetic runtime failure'))
    assert isinstance(escaped, RuntimeError)
    assert process['exit_code'] == 1 and process['exception_type'] == 'RuntimeError'


def test_error_outside_the_refusal_classes_prints_python_traceback_that_is_not_scrubbed(failing_main, capsys):
    """Review I1 (wording): the runner scrubs only the exceptions it catches. A RuntimeError (CUDA out of memory, a trainer resource
    bound) or a KeyError ends with exit 1 and Python's own traceback; the operator doc says so, and so this test pins that it is true."""
    from c_ept.workflows import run_s4a as r
    import traceback
    assert not issubclass(RuntimeError, r._REFUSAL_ERRORS) and not issubclass(KeyError, r._REFUSAL_ERRORS)
    escaped, process = failing_main(KeyError(f'x/{KEY}'))
    assert isinstance(escaped, KeyError) and process['exit_code'] == 1 and process['exception_type'] == 'KeyError'
    assert capsys.readouterr().err == ''                      # nothing is printed by the runner: no scrubbed text, no class-name line
    shown = ''.join(traceback.format_exception(type(escaped), escaped, escaped.__traceback__))      # what Python prints for the escaped exception
    assert KEY in str(escaped) and KEY in shown and '<key>' not in shown                              # the key is still in it: nothing scrubbed it


@pytest.mark.parametrize('name,recorded', [
    ('Ünicode_오류', False),            # not ASCII
    ('E' * 129, False),                 # longer than the validators allow (128)
    ('pkg.Error', False),               # a dotted name is not a class name
    ('E' * 128, True),                  # the longest name the validators accept
    ('_Err1', True)])
def test_exception_type_is_written_only_for_a_class_name_the_validators_accept(failing_main, name, recorded):
    """Review M3: a recorded failure must not turn into a metadata refusal because of its class name. The field is left out, and the strict
    status validator then accepts the record exactly as written."""
    from c_ept.workflows import s4a_operator as op
    returned, process = failing_main(type(name, (ValueError,), {})('synthetic failure'))
    assert returned == 2 and process['state'] == 'failed' and process['exit_code'] == 2
    assert ('exception_type' in process) is recorded
    result = op.status(Path(process['run_path']))
    assert result['state'] == 'recorded_failure' and result['recorded_exit_code'] == 2
    assert result.get('recorded_exception_type') == (name if recorded else None)


def test_t002_process_record_leaves_out_an_exception_type_the_validators_would_refuse(tmp_path):
    from c_ept.workflows import run_training
    manifest = {'run_identity_sha256': 'b' * 64, 'code_commit': 'a' * 40, 'identity': {'operator_config_sha256': 'c' * 64}}
    kept = run_training._process_record(tmp_path, manifest, 'failed', 'failed', 2, 'ValueError')
    left_out = run_training._process_record(tmp_path, manifest, 'failed', 'failed', 2, 'Ünicode_오류')
    assert kept['exception_type'] == 'ValueError' and 'exception_type' not in left_out
    assert left_out['state'] == 'failed' and left_out['exit_code'] == 2


def test_scrub_is_best_effort_and_says_so_when_the_s1_store_cannot_be_read(failing_main, capsys, tmp_path):
    returned, _ = failing_main(ValueError('synthetic failure without a key'), s1=tmp_path / 'no-such-s1')
    err = capsys.readouterr().err
    assert returned == 2 and 'ValueError' in err and 'Traceback' in err
    assert 'could not be loaded' in err                                      # an unscrubbed text is never presented as scrubbed


def test_failure_report_scrubs_only_what_is_not_a_guard_refusal(tmp_path):
    from c_ept.workflows.run_s4a import failure_report
    s1 = toy_s1(tmp_path / 's1')
    guard = RuntimeRefused(f'guard says {KEY}')
    assert failure_report('X refused', guard, (RuntimeRefused,), s1=s1) == f'X refused: guard says {KEY}'     # unchanged text
    try:
        raise KeyError(f'x/{KEY}')
    except KeyError as error:
        text = failure_report('X refused', error, (RuntimeRefused,), s1=s1)
    assert 'KeyError' in text and 'Traceback' in text and 'TOYKEY' not in text and 'x/<key>' in text
