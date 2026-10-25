"""Synthetic-only metadata, budget, best-loader and fake Job operator probes.

No production metadata/data, monitor, model payload, or evaluation is executed.
Only the loader test serializes a newly created tiny CPU model into tmp_path.
The supervisor's process, process inspector and WindowsJob are entirely fake.
"""
from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from c_ept.evaluation import guard, runtime


ROOT = Path(__file__).resolve().parents[1]
HASH = 'a' * 64


@pytest.fixture(autouse=True)
def synthetic_runtime(monkeypatch):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        monkeypatch.setenv(key, '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('PYTHONUTF8', '1')
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    original = builtins.__import__

    def forbidden_import(name, *args, **kwargs):
        if (name in ('ept_training_monitor', 'c_ept.cohort', 'research_pipeline.cohort_policy')
                or name.startswith('research_pipeline.holdout_r3')):
            raise AssertionError('forbidden monitor/protected module in synthetic tests: ' + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', forbidden_import)


@pytest.fixture
def operator(monkeypatch):
    # Loading by file avoids importing an unrelated installed `tools` package.
    old_path = list(sys.path)
    spec = importlib.util.spec_from_file_location('_synthetic_ept_dev_operator', ROOT / 'tools/ept_dev_evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path[:] = old_path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    return path


def config(tmp_path):
    c = {**guard.FIXED, **{key: str((tmp_path / key).resolve()) for key in guard.PATHS},
         'reviewed_commit': 'b' * 40, 'definitions_sha256': HASH,
         'final_audit_sha256': HASH, 'final_report_sha256': HASH}
    # Only a syntactic config reference: no files in this root are read/written.
    c['output'] = str(Path('D:/LOL_Data/runs') / ('ept_dev_SYNTHETIC_' + tmp_path.name))
    return c


def completed_parent(losses=(3., 1., 1., 2.)):
    return {'head': {'done': True, 'epoch': len(losses), 'best_epoch': 1,
                     'best': {'folder': 'epoch_0002', 'sha256': HASH, 'bytes': 13}},
            'history': {'rows': [{'stop_loss': value} for value in losses]}}


@pytest.fixture
def metadata_world(tmp_path, monkeypatch):
    """All input manifests and immutable identities are small invented JSONs."""
    c = config(tmp_path)
    c['output'] = str(tmp_path / 'fresh-output')
    source = Path(c['source'])
    monkeypatch.setattr(guard, '__file__', str(source / 'c_ept/evaluation/guard.py'))
    definitions = source / 'docs/EPT_DEV_DEFINITIONS_KO.md'
    definitions.parent.mkdir(parents=True)
    definitions.write_text('synthetic definition only', encoding='utf-8')
    c['definitions_sha256'] = guard.sha_file(definitions)
    reports = Path(c['training_control'])/'reports'
    c['final_report_sha256'] = guard.sha_file(save(reports/'FINAL_REPORT_KO.md', {'synthetic':True}))
    artifacts = {name: guard.sha_file(save(reports/name, {'synthetic':True}))
                 for name in ('curve.png','history.json','history.csv')}
    audit = {'schema':'ept-pilot-final-audit-v1','state':'completed','code_commit':guard.TRAIN_COMMIT,
             'operator_config_sha256':guard.TRAIN_CONFIG,'run_identity_sha256':guard.TRAIN_ID,
             'run':c['training_run'],'source':c['training_source'],
             'manual_visual_review':'verified_against_json_csv','final_report_sha256':c['final_report_sha256'],
             'reviewed_artifacts':artifacts}
    c['final_audit_sha256'] = guard.sha_file(save(reports/'final-audit.json', audit))
    s1, s2, s3, run = (Path(c[k]) for k in ('s1', 's2', 's3', 'training_run'))
    labels = tmp_path / 'labels'
    head = save(s3 / 'fit/b1/open/kill/head.json', {
        'schema': 'b1_head_v1', 'kind': 'open', 'name': 'kill',
        'head': {'files': {'model': {'file': 'NEVER_READ.joblib', 'sha256': HASH}}}})
    b1 = save(s3 / 'fit/b1/b1_heads.json', {
        'schema': 'b1_heads_v2', 'fit_roles': ['FIT'], 'stop_roles': ['STOP'],
        'heads': {'open': {'kill': {'dir': 'open/kill', 'head_sha256': guard.sha_file(head),
                                  'files': {'model': {'file': 'open/kill/NEVER_READ.joblib', 'sha256': HASH}}}}}})
    s3run = save(s3 / 'RUN.json', {'code_commit': guard.B1_COMMIT, 'code_dirty': False, 'smoke': None})
    s3stages = save(s3 / 'STAGES.json', {'stages': {
        'fit': {'complete': True, 'outputs': {'fit/b1/b1_heads.json': guard.sha_file(b1)}},
        'labels': {'complete': True}}})
    s1store = save(s1 / 'STORE.json', {'synthetic': True})
    s2store = save(s2 / 'STORE.json', {'synthetic': True})
    stats = save(run / 'stats/STATS.json', {'synthetic': True})
    scales = save(run / 'B0_SCALES.json', {'synthetic': True})
    save(run / 'STAGES.json', {'stages': {
        'stats': {'outputs': {'stats/STATS.json': guard.sha_file(stats)}},
        'scales': {'outputs': {'B0_SCALES.json': guard.sha_file(scales)}}}})
    prices = save(tmp_path / 'prices.json', {'synthetic': True})
    rules = save(tmp_path / 'rules.json', {'synthetic': True})
    labelmeta = save(labels / 'LABELS.json', {'synthetic': True})
    files = {'NEVER_READ.npz': HASH}
    save(labels / 'STORE.json', {'complete': True, 'files': files})
    meta = {'paths': {**{k: c[k] for k in ('s1', 's2', 's3')},
                      'prices': str(prices), 'rules': str(rules), 'labels': str(labels)},
            'source_json_sha256': {k: guard.sha_file(p) for k, p in (
                ('S3RUN', s3run), ('S3STAGES', s3stages), ('S1STORE', s1store), ('S2STORE', s2store))},
            'prices_sha256': guard.sha_file(prices), 'rules_sha256': guard.sha_file(rules),
            'label_provenance_sha256': guard.sha_file(labelmeta),
            'expected_provenance': {'labels_store_sha256': guard.digest(files)}}
    save(run / 'RUN.json', {'identity': {'inputs': meta}})
    monkeypatch.setattr(guard, 'verify_history_hashes', lambda _: {'synthetic': True, 'verified_count': 16})
    return c, completed_parent()


def block_payload_and_models(monkeypatch):
    original_import, original_open = builtins.__import__, Path.open
    opened = []

    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'numpy', 'lightgbm', 'joblib') or name.startswith(('c_ept.ept', 'c_ept.models', 'c_ept.data')):
            raise AssertionError('metadata Check imported model/data code: ' + name)
        return original_import(name, *args, **kwargs)

    def guarded_open(path, *args, **kwargs):
        assert path.suffix.lower() not in ('.pt', '.joblib', '.npz', '.npy', '.bin'), str(path)
        opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(Path, 'open', guarded_open)
    return opened


def test_live_parent_refused_before_payload_or_other_preflight(metadata_world, monkeypatch):
    c, _ = metadata_world
    events = []
    block_payload_and_models(monkeypatch)

    def live(_):
        events.append('parent')
        raise guard.EvaluationRefused('parent training is live')

    def forbidden(*args, **kwargs):
        pytest.fail('post-parent operation reached while parent is live')

    monkeypatch.setattr(guard, 'sha_file', forbidden)
    monkeypatch.setattr(guard, 'input_metadata', forbidden)
    with pytest.raises(guard.EvaluationRefused, match='live'):
        guard.preflight(c, parent_probe=live, source_probe=lambda *_: events.append('source'))
    assert events == ['source', 'source', 'parent']
    assert not Path(c['output']).exists()


def test_default_cli_check_is_metadata_only(metadata_world, monkeypatch, capsys):
    c, parent = metadata_world
    opened = block_payload_and_models(monkeypatch)
    # Import the CLI while imports of torch/model/data code are prohibited too.
    old_path = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location('_metadata_only_cli', ROOT / 'tools/ept_dev_evaluate.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        preflight = guard.preflight
        checked = []

        def synthetic_check(value):
            result = preflight(value, parent_probe=lambda _: parent, source_probe=lambda *_: None)
            checked.append(result)
            return result

        monkeypatch.setattr(guard, 'load_config', lambda _: c)
        monkeypatch.setattr(guard, 'preflight', synthetic_check)
        monkeypatch.setattr(cli, 'supervise', lambda *_: pytest.fail('Check must never start'))
        assert cli.main(['--config', str(Path(c['source']) / 'synthetic.json')]) == 0
        assert json.loads(capsys.readouterr().out)['state'] == 'READY_WITHIN_SCOPE'
        assert checked[0]['best']['epoch'] == 2
        assert checked[0]['metadata_hashes']
        assert opened and all(p.is_relative_to(Path(c['source']).parent) for p in opened)
        assert not Path(checked[0]['best']['path']).exists()
        assert not Path(c['output']).exists()
    finally:
        sys.path[:] = old_path


def test_strict_config_accepts_only_exact_profile(tmp_path):
    c = config(tmp_path)
    loaded = guard.load_config(save(tmp_path / 'config.json', c))
    assert loaded == {**c, **{k: str(Path(c[k]).resolve()) for k in guard.PATHS}}


@pytest.mark.parametrize('field,value', [
    ('role', 'TEST'), ('role', 'CAL'), ('device', 'cpu'), ('bootstrap', 2000.),
    ('batch_size', True), ('draws', 999), ('native_threads', 1), ('seed', 0),
    ('schema', 'unknown'), ('reviewed_commit', 'x' * 40),
    ('definitions_sha256', 'A' * 64), ('source', 'relative/path'),
    ('extra', 'unexpected'), ('output', 'D:/LOL_Data/runs/ept_dev_outer/nested'),
    ('output', 'D:/LOL_Data/runs/not_dev'),
])
def test_strict_config_rejects_drift(tmp_path, field, value):
    c = config(tmp_path)
    c[field] = value
    with pytest.raises(guard.EvaluationRefused):
        guard.load_config(save(tmp_path / 'bad.json', c))


@pytest.mark.parametrize('case', ['missing', 'same_source', 'overlap'])
def test_config_missing_separation_and_output_overlap(tmp_path, case):
    c = config(tmp_path)
    if case == 'missing':
        del c['seed']
    elif case == 'same_source':
        c['source'] = c['training_source']
    else:
        c['s1'] = c['output']
    with pytest.raises(guard.EvaluationRefused):
        guard.load_config(save(tmp_path / 'bad.json', c))


@pytest.mark.parametrize('text', ['{"a": 1, "a": 2}', '{"x": NaN}', '{"x": Infinity}'])
def test_metadata_json_rejects_duplicates_and_nonfinite(tmp_path, text):
    path = tmp_path / 'bad.json'
    path.write_text(text, encoding='utf-8')
    with pytest.raises(guard.EvaluationRefused):
        guard.read_json(path)


@pytest.mark.parametrize('relative', ['../escape.pt', 'sub/../../escape.pt', 'C:/escape.pt',
                                      'C:escape.pt', '/escape.pt', 'sub/model.pt:stream', r'..\escape.pt'])
def test_artifact_path_escape_refused(tmp_path, relative):
    with pytest.raises(guard.EvaluationRefused):
        guard.inside(tmp_path, relative)


def test_artifact_resolved_parent_escape_refused(tmp_path, monkeypatch):
    original = Path.resolve
    root = tmp_path / 'root'
    root.mkdir()
    escaping = root / 'redirect' / 'best.pt'

    def resolve(path, *args, **kwargs):
        if path == escaping:
            return original(tmp_path / 'outside' / 'best.pt')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'resolve', resolve)
    with pytest.raises(guard.EvaluationRefused, match='escapes'):
        guard.inside(root, 'redirect/best.pt')


def test_best_is_earliest_strict_stop_minimum_without_payload(tmp_path, monkeypatch):
    block_payload_and_models(monkeypatch)
    best = guard.best_reference(completed_parent(), tmp_path)
    assert best == {'path': str(tmp_path / 'epoch_0002/best.pt'), 'sha256': HASH, 'bytes': 13, 'epoch': 2}
    assert not Path(best['path']).exists()
    later = completed_parent()
    later['head'].update(best_epoch=2, best={'folder': 'epoch_0003', 'sha256': HASH, 'bytes': 13})
    with pytest.raises(guard.EvaluationRefused, match='STOP best epoch'):
        guard.best_reference(later, tmp_path)


@pytest.mark.parametrize('change', ['not_done', 'bad_epoch', 'nonfinite', 'bool_loss', 'short_history', 'wrong_folder', 'size_bool', 'bad_hash'])
def test_invalid_best_metadata_refused(tmp_path, change):
    p = completed_parent()
    if change == 'not_done': p['head']['done'] = False
    elif change == 'bad_epoch': p['head']['best_epoch'] = True
    elif change == 'nonfinite': p['history']['rows'][0]['stop_loss'] = float('nan')
    elif change == 'bool_loss': p['history']['rows'][0]['stop_loss'] = True
    elif change == 'short_history': p['history']['rows'].pop()
    elif change == 'wrong_folder': p['head']['best']['folder'] = '../escape'
    elif change == 'size_bool': p['head']['best']['bytes'] = True
    else: p['head']['best']['sha256'] = 'invalid'
    with pytest.raises(guard.EvaluationRefused):
        guard.best_reference(p, tmp_path)


def test_budget_phase_order_and_persisted_binding(tmp_path):
    clock = SimpleNamespace(now=10.)
    budget = runtime.Budget(tmp_path, HASH, clock=lambda: clock.now, measure=lambda: (123, 456))
    with pytest.raises(guard.EvaluationRefused, match='transition'):
        budget.enter('inference')
    for phase in runtime.PHASE_LIMITS:
        clock.now += 1
        budget.enter(phase)
        saved = guard.read_json(tmp_path / 'phase.json')
        assert saved['phase'] == phase and saved['config_sha256'] == HASH
        assert saved['phase_started_monotonic'] == clock.now
    assert budget.samples == 4 and budget.peak_private == 123 and budget.peak_upper == 456
    for invalid in ('report', 'metrics', 'unknown'):
        with pytest.raises(guard.EvaluationRefused, match='transition'):
            budget.enter(invalid)


@pytest.mark.parametrize('phase', list(runtime.PHASE_LIMITS))
def test_budget_phase_exact_deadline_refused(tmp_path, phase):
    clock = SimpleNamespace(now=0.)
    budget = runtime.Budget(tmp_path, HASH, clock=lambda: clock.now)
    for name in runtime.PHASE_LIMITS:
        budget.enter(name)
        if name == phase:
            break
    clock.now = runtime.PHASE_LIMITS[phase] - .001
    budget.check()
    clock.now += .001
    with pytest.raises(TimeoutError, match='budget'):
        budget.check()


def test_budget_total_not_reset_by_phase_changes(tmp_path, monkeypatch):
    # Shrink only the synthetic total limit to isolate it from each phase limit.
    monkeypatch.setattr(runtime, 'TOTAL_SECONDS', 20)
    clock = SimpleNamespace(now=0.)
    budget = runtime.Budget(tmp_path, HASH, clock=lambda: clock.now)
    budget.enter('preflight')
    clock.now = 10.
    budget.enter('inference')
    clock.now = 15.
    budget.enter('metrics')
    clock.now = 19.
    budget.enter('report')
    assert budget.start == 0
    budget.check()
    clock.now += 1
    with pytest.raises(TimeoutError, match='budget'):
        budget.check()


@pytest.mark.parametrize('private', [None, True, 0, -1, 1.0, runtime.MEMORY_BYTES])
def test_budget_invalid_or_at_cap_memory_refused(tmp_path, private):
    budget = runtime.Budget(tmp_path, HASH, clock=lambda: 0., measure=lambda: (private, None))
    with pytest.raises(guard.EvaluationRefused, match='memory'):
        budget.enter('preflight')


def test_budget_tracks_sampled_peak_separately_from_lifetime_upper(tmp_path):
    samples = iter([(runtime.MEMORY_BYTES - 1, runtime.MEMORY_BYTES + 5), (12, None)])
    budget = runtime.Budget(tmp_path, HASH, clock=lambda: 0., measure=lambda: next(samples))
    budget.enter('preflight')
    budget.check()
    assert budget.peak_private == runtime.MEMORY_BYTES - 1
    assert budget.peak_upper == runtime.MEMORY_BYTES + 5
    assert budget.samples == 2


@pytest.fixture
def fake_supervisor(tmp_path, monkeypatch, operator):
    c = config(tmp_path)
    c['source'] = str(operator.SOURCE)
    c['output'] = str(tmp_path / 'supervised')
    events = []
    mode = SimpleNamespace(exit_code=0, gone=True, assign_error=False, release_error=False,
                           close_error=False, result=True, wrong_result=False, elapsed=0.,
                           changed_config=False, phase=None, preflight_error=False, private=1024)
    child = SimpleNamespace(pid=918273, returncode=None, stdin=io.BytesIO())

    def poll():
        return child.returncode

    def wait(timeout):
        events.append('wait')
        if child.returncode is None:
            raise subprocess.TimeoutExpired('fake-child', timeout)
        return child.returncode

    child.poll, child.wait = poll, wait

    class FakeJob:
        def __init__(self): events.append('job')
        def assign(self, process):
            assert process is child
            events.append('assign')
            if mode.assign_error: raise OSError('synthetic assign failure')
        def release(self, process):
            events.append('release')
            record = guard.read_json(Path(c['output']) / 'controller.json')
            assert record['assigned'] is True and record['child_pid'] == child.pid
            if mode.release_error: raise OSError('synthetic release failure')
            out = Path(c['output'])
            if mode.preflight_error:
                save(out/'WORKER_FAILURE.json', {'state':'failed','error':'parent live'})
                child.returncode=2
                return
            checked={'config_sha256':guard.digest(c),'metadata_hashes':{},'best':{'epoch':2}}
            save(out/'CHECK.json', checked)
            receipt={'state':'passed','config_sha256':guard.digest(c),'check_sha256':guard.sha_file(out/'CHECK.json'),
                     'metadata_hashes_sha256':guard.digest({}), 'started_monotonic':0.,'finished_monotonic':0.,
                     **{k:record[k] for k in ('controller_pid','controller_create_time','child_pid','child_create_time')}}
            save(out/'PREFLIGHT_RECEIPT.json',receipt)
            receipt_hash=guard.sha_file(out/'PREFLIGHT_RECEIPT.json')
            bindings={'preflight_receipt_sha256':receipt_hash,'check_sha256':receipt['check_sha256'],
                      'metadata_hashes_sha256':receipt['metadata_hashes_sha256']}
            manifest=guard.read_json(out/'RUN_MANIFEST.json')
            save(out/'RUN_MANIFEST.json',{**manifest,**bindings})
            if mode.result:
                save(Path(c['output']) / 'RESULT.json', {'state': 'completed',
                     'config_sha256': 'wrong' if mode.wrong_result else guard.digest(c),
                     'best':checked['best'],**bindings})
            if mode.phase:
                save(Path(c['output']) / 'phase.json', {'phase': mode.phase, 'pid': child.pid,
                     'config_sha256': guard.digest(c), 'phase_started_monotonic': 0.})
            child.returncode = mode.exit_code
            mode.elapsed = getattr(mode, 'after_release', 0.)
        def gone(self):
            events.append('gone')
            return mode.gone
        def stop(self, timeout):
            assert 0 <= timeout <= 10
            events.append('stop')
            mode.gone = True
            child.returncode = -1 if child.returncode is None else child.returncode
            return True
        def close(self):
            events.append('close')
            if mode.close_error: raise OSError('synthetic close failure')
        def contains(self, pid, created): return pid==child.pid and created==17.

    def popen(*args, **kwargs):
        events.append('popen')
        assert args[0][0] == str(guard.PYTHON)
        assert '--worker' in args[0]
        return child

    monkeypatch.setitem(sys.modules, 'ept_training_job', SimpleNamespace(WindowsJob=FakeJob))
    monkeypatch.setitem(sys.modules, 'psutil', SimpleNamespace(
        Process=lambda *a: SimpleNamespace(create_time=lambda: 17.,memory_info=lambda:SimpleNamespace(private=mode.private)),
        NoSuchProcess=ProcessLookupError))
    monkeypatch.setattr(operator, 'subprocess', SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE,
                        CREATE_NO_WINDOW=getattr(subprocess, 'CREATE_NO_WINDOW', 0), TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(operator, 'time', SimpleNamespace(monotonic=lambda: mode.elapsed,
                        sleep=lambda _: pytest.fail('unexpected real wait in fake supervision')))
    monkeypatch.setattr(guard, 'preflight', lambda _: {'synthetic': True, 'best': {'epoch': 2}})
    monkeypatch.setattr(guard, 'clean_source', lambda *args: events.append('clean'))
    loads = []

    def load(_):
        loads.append(1)
        return {**c, 'seed': 0} if mode.changed_config and len(loads) > 1 else copy.deepcopy(c)

    monkeypatch.setattr(guard, 'load_config', load)
    return operator, c, mode, events, child


def test_fake_job_success_assigns_before_release_and_closes(fake_supervisor):
    cli, c, mode, events, child = fake_supervisor
    result = cli.supervise(c, 'synthetic-config.json')
    assert result['state'] == 'completed' and result['exit_code'] == 0
    assert all(result[k] is True for k in ('assigned', 'released', 'job_closed', 'job_gone_before_close', 'tree_gone'))
    assert events.index('popen') < events.index('assign') < events.index('release') < events.index('close')
    assert 'stop' not in events
    assert guard.read_json(Path(c['output']) / 'final-audit.json')['controller'] == result
    assert guard.read_json(Path(c['output']) / 'RUN_MANIFEST.json')['retry_limit'] == 0


@pytest.mark.parametrize('fault', ['exit', 'descendant', 'assign', 'release', 'missing_result', 'wrong_result',
                                  'changed_config', 'total_timeout', 'preflight_timeout', 'invalid_phase', 'close'])
def test_fake_job_failure_cleans_owned_resources(fake_supervisor, fault):
    cli, c, mode, events, child = fake_supervisor
    if fault == 'exit': mode.exit_code = 5
    elif fault == 'descendant': mode.gone = False
    elif fault == 'assign': mode.assign_error = True
    elif fault == 'release': mode.release_error = True
    elif fault == 'missing_result': mode.result = False
    elif fault == 'wrong_result': mode.wrong_result = True
    elif fault == 'changed_config': mode.changed_config = True
    elif fault == 'total_timeout': mode.after_release = runtime.TOTAL_SECONDS
    elif fault == 'preflight_timeout': mode.after_release = runtime.PHASE_LIMITS['preflight']
    elif fault == 'invalid_phase': mode.phase = 'unknown'
    else: mode.close_error = True
    result = cli.supervise(c, 'synthetic-config.json')
    assert result['state'] == 'failed'
    assert result['error'] or result.get('cleanup_error')
    assert 'close' in events
    if fault == 'close':
        assert result['job_closed'] is False and result['tree_gone'] is False
    elif fault == 'changed_config':
        assert 'popen' not in events and 'stop' in events and 'wait' not in events
        assert result['tree_gone'] is True and result['job_closed'] is True
    else:
        assert 'stop' in events and 'wait' in events
        assert result['tree_gone'] is True and result['job_closed'] is True
    if fault == 'assign': assert 'release' not in events
    assert guard.read_json(Path(c['output']) / 'final-audit.json')['controller']['state'] == 'failed'


def test_supervisor_preflight_failure_preserves_attempt_and_refuses_retry(fake_supervisor):
    cli, c, mode, events, child = fake_supervisor
    mode.preflight_error=True
    result=cli.supervise(c,'synthetic-config.json')
    out=Path(c['output'])
    assert result['state']=='failed' and (out/'launch.once').is_file() and (out/'FAILURE.json').is_file()
    assert guard.read_json(out/'WORKER_FAILURE.json')['error']=='parent live'
    assert not (out/'CHECK.json').exists()
    with pytest.raises(FileExistsError):
        cli.supervise(c, 'synthetic-config.json')


def test_supervisor_refuses_output_reuse_without_launch(fake_supervisor):
    cli, c, mode, events, child = fake_supervisor
    Path(c['output']).mkdir()
    sentinel = Path(c['output']) / 'existing.txt'
    sentinel.write_text('preserve', encoding='utf-8')
    with pytest.raises(FileExistsError):
        cli.supervise(c, 'synthetic-config.json')
    assert sentinel.read_text(encoding='utf-8') == 'preserve'
    assert 'popen' not in events


def test_worker_gate_refuses_before_config_and_runtime(operator, monkeypatch):
    monkeypatch.setattr(operator.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(b'')))
    monkeypatch.setattr(guard, 'load_config', lambda *_: pytest.fail('config read before G gate'))
    monkeypatch.setattr(runtime, 'execute', lambda *_: pytest.fail('actual evaluation forbidden'))
    with pytest.raises(guard.EvaluationRefused, match='gate'):
        operator.worker('unused.json')


@pytest.mark.parametrize('private', [None, 0, 1.5, runtime.MEMORY_BYTES])
def test_supervisor_samples_private_before_release_and_fails_closed(fake_supervisor, private):
    cli,c,mode,events,_=fake_supervisor
    mode.private=private
    result=cli.supervise(c,'synthetic-config.json')
    assert result['state']=='failed' and 'release' not in events
    assert result['tree_gone'] is True and result['job_closed'] is True


def test_independent_sampler_rejects_gap_during_native_work(fake_supervisor):
    cli,c,mode,_,child=fake_supervisor
    job=SimpleNamespace(contains=lambda *_:True)
    record={'child_create_time':17.,'memory_samples':0,'max_memory_sample_gap_seconds':0.,
            'peak_sampled_child_private_bytes':0}
    cli._sample_child(child,job,record,sys.modules['psutil'])
    mode.elapsed=cli.MAX_SAMPLE_GAP+.01
    with pytest.raises(guard.EvaluationRefused,match='sampling gap'):
        cli._sample_child(child,job,record,sys.modules['psutil'])


def _worker_context(fake_supervisor, monkeypatch):
    cli,c,mode,events,child=fake_supervisor
    out=Path(c['output']); out.mkdir()
    ctl={'child_pid':os.getpid(),'child_create_time':17.,'controller_pid':os.getppid(),
         'controller_create_time':17.,'config_sha256':guard.digest(c),'assigned':True,'started_monotonic':0.}
    save(out/'controller.json',ctl)
    save(out/'RUN_MANIFEST.json',{'config_sha256':guard.digest(c),'config':c})
    monkeypatch.setattr(cli.sys,'stdin',SimpleNamespace(buffer=io.BytesIO(b'G')))
    return cli,c,out


def test_worker_preflight_failure_is_preserved_before_heavy_runtime(fake_supervisor, monkeypatch):
    cli,c,out=_worker_context(fake_supervisor,monkeypatch)
    calls=[]
    def refuse(value, **kwargs):
        calls.append(kwargs)
        raise guard.EvaluationRefused('parent live')
    monkeypatch.setattr(guard,'preflight',refuse)
    monkeypatch.setattr(runtime,'execute',lambda *_:pytest.fail('heavy execution before preflight'))
    with pytest.raises(guard.EvaluationRefused,match='parent live'):
        cli.worker('synthetic-config.json')
    assert calls==[{'allow_reserved':True}] and not (out/'CHECK.json').exists()
    assert 'parent live' in guard.read_json(out/'WORKER_FAILURE.json')['error']


@pytest.mark.parametrize('tamper', [False,True])
def test_worker_binds_preflight_receipt_to_result(fake_supervisor, monkeypatch, tamper):
    cli,c,out=_worker_context(fake_supervisor,monkeypatch)
    checked={'config_sha256':guard.digest(c),'metadata_hashes':{},'best':{'epoch':2}}
    monkeypatch.setattr(guard,'preflight',lambda value,**kw:checked)
    def execute(value,check):
        assert check==checked
        save(out/'RESULT.json',{'state':'completed','config_sha256':guard.digest(c),'best':checked['best']})
        if tamper: save(out/'CHECK.json',{**checked,'metadata_hashes':{'changed':'a'*64}})
    monkeypatch.setattr(runtime,'execute',execute)
    if tamper:
        with pytest.raises(guard.EvaluationRefused,match='metadata changed'):
            cli.worker('synthetic-config.json')
        assert (out/'WORKER_FAILURE.json').is_file()
    else:
        cli.worker('synthetic-config.json')
        result=guard.read_json(out/'RESULT.json')
        assert result['preflight_receipt_sha256']==guard.sha_file(out/'PREFLIGHT_RECEIPT.json')
        assert result['check_sha256']==guard.sha_file(out/'CHECK.json')


def test_best_loader_tiny_cpu_model_strict_hash_and_state(tmp_path, monkeypatch):
    import numpy as np
    import torch

    torch.set_num_threads(1)

    class Tiny(torch.nn.Module):
        def __init__(self, config, static):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

    # Stub only architecture construction: exercise real weights_only torch.load,
    # strict state loading, static identity, CPU placement and eval mode.
    package = ModuleType('c_ept.ept')
    package.encoder = SimpleNamespace(EncoderConfig=lambda **kwargs: SimpleNamespace(**kwargs))
    package.heads = SimpleNamespace(EPT=Tiny)
    monkeypatch.setitem(sys.modules, 'c_ept.ept', package)
    static = np.zeros((2, 2), dtype=np.float32)
    model = Tiny(None, static)
    with torch.no_grad():
        model.linear.weight.fill_(.25)
        model.linear.bias.fill_(-.5)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    raw = buffer.getvalue()
    path = tmp_path / 'best.pt'
    path.write_bytes(raw)
    reference = {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    head = {'identity': {'static_sha256': hashlib.sha256(static.tobytes()).hexdigest(), 'encoder': {}}}
    calls = []
    original_load = torch.load

    def load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', load)
    loaded = runtime.load_best(reference, head, static, 'cpu')
    assert not loaded.training
    assert next(loaded.parameters()).device.type == 'cpu'
    assert calls == [{'weights_only': True, 'map_location': 'cpu'}]
    assert torch.equal(loaded.linear(torch.tensor([[2., 2.]])), torch.tensor([[.5]]))
    for change in ({'sha256': '0' * 64}, {'bytes': len(raw) + 1}, {'path': str(tmp_path / 'last.pt')}):
        with pytest.raises(guard.EvaluationRefused):
            runtime.load_best({**reference, **change}, head, static, 'cpu')
    with pytest.raises(guard.EvaluationRefused, match='static'):
        runtime.load_best(reference, head, static + 1, 'cpu')
    assert len(calls) == 1  # Invalid metadata never reaches deserialization.
    invalid = io.BytesIO()
    torch.save({'unexpected': torch.zeros(1)}, invalid)
    path.write_bytes(invalid.getvalue())
    ref = {**reference, 'bytes': len(invalid.getvalue()), 'sha256': hashlib.sha256(invalid.getvalue()).hexdigest()}
    with pytest.raises(RuntimeError, match='state_dict'):
        runtime.load_best(ref, head, static, 'cpu')
