"""EPT-T002 synthetic profile/guard composition only; no real train or GPU work."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from c_ept.workflows import training_operator as op, run_training as runner, run_s4a as primitive


@pytest.fixture
def config(tmp_path, monkeypatch):
    root = tmp_path / 'runs'
    root.mkdir()
    monkeypatch.setattr(op, 'RUNS_ROOT', root)
    c = dict(op.FIXED, python=str(Path(sys.executable).resolve()),
             runs_root=str(root), catalog_db=str(tmp_path / 'catalog.sqlite'))
    for key in ('s1', 's2', 's3'):
        folder = tmp_path / key
        folder.mkdir()
        c[key] = str(folder)
    path = tmp_path / 'operator.json'
    path.write_text(json.dumps(c), encoding='utf-8')
    return c, path, root / 'control'


@pytest.mark.parametrize('key,value', [
    ('schema', 's4a-operator-v2'), ('profile', 'gpu_cost'), ('size', 'M'), ('device', 'cpu'),
    ('max_epochs', 2), ('max_epochs', 30.), ('max_epochs', True), ('share', 1.), ('share', True),
    ('share', float('nan')), ('share', float('inf')), ('patience', 3), ('patience', 2.),
    ('batch_size', 16), ('seed', 1), ('seed', True), ('lr', .002), ('weight_decay', 0.),
    ('warmup_steps', 0), ('dev_sample', 200), ('resume', 'old'), ('python', 'python'),
    ('python', str(Path(r'C:\wrong\python.exe'))), ('s1', './relative'),
    ('s1', r'C:/noncanonical/path'), ('runs_root', str(Path(r'C:\wrong'))),
    ('catalog_db', str(Path(r'C:\outside.sqlite'))),
])
def test_config_strict_selection(config, key, value):
    c, path, _ = config
    c[key] = value
    path.write_text(json.dumps(c), encoding='utf-8')
    with pytest.raises(op.OperatorError):
        op.load_config(path)


def test_config_positive_and_duplicate_missing(config):
    c, path, _ = config
    assert op.load_config(path) == c
    path.write_text(path.read_text()[:-1] + ',"patience":2}', encoding='utf-8')
    with pytest.raises(op.OperatorError):
        op.load_config(path)
    c.pop('patience')
    path.write_text(json.dumps(c))
    with pytest.raises(op.OperatorError):
        op.load_config(path)


def test_check_default_readonly_no_runner_or_control(config, monkeypatch):
    c, path, control = config
    monkeypatch.setattr(op, 'preflight', lambda loaded: 'a' * 40)
    monkeypatch.setattr(op.importlib, 'import_module', lambda name: pytest.fail('Check imported training runner'))
    before = path.read_bytes()
    assert op.main(['--config', str(path), '--control', str(control)]) == 0
    assert not control.exists() and path.read_bytes() == before


def test_check_uses_runtime_source_e6_and_only_bounded_metadata(config, monkeypatch):
    c, path, control = config
    calls = []
    for name in ('runtime_check', 'code_check', 'gpu_metadata', 'e6_check'):
        monkeypatch.setattr(op.common, name, lambda name=name: calls.append(name) or 'a' * 40)
    for key in ('s1', 's2'):
        (Path(c[key]) / 'STORE.json').write_text('{"complete":true,"files":{}}')
    (Path(c['s3']) / 'RUN.json').write_text('{"stage":"s3"}')
    (Path(c['s3']) / 'STAGES.json').write_text('{}')
    original = Path.open
    def guarded(path, *args, **kw):
        if path.name not in ('STORE.json', 'RUN.json', 'STAGES.json', 'operator.json'):
            pytest.fail('unexpected preflight file access: ' + path.name)
        return original(path, *args, **kw)
    monkeypatch.setattr(Path, 'open', guarded)
    assert op.main(['--config', str(path), '--control', str(control)]) == 0
    assert calls == ['runtime_check', 'code_check', 'gpu_metadata', 'e6_check']
    assert not control.exists()


def test_start_requires_control_and_passes_supported_runner_contract(config, monkeypatch):
    c, path, control = config
    captured = []
    monkeypatch.setattr(op.importlib, 'import_module', lambda name: captured.append(name) or SimpleNamespace(main=lambda args: captured.append(args) or 7))
    assert op.main(['--action', 'Start', '--config', str(path)]) == 2
    assert captured == []
    assert op.main(['--action', 'Start', '--config', str(path), '--control', str(control)]) == 7
    assert captured == ['c_ept.workflows.run_training', ['--config', str(path), '--control', str(control),
                        '--operator-config-sha256', op.config_digest(c)]]


def test_no_resume_or_cost_arguments(config):
    _, path, control = config
    with pytest.raises(SystemExit):
        op.main(['--action', 'Resume', '--config', str(path)])
    with pytest.raises(SystemExit):
        runner.main(['--config', str(path), '--control', str(control), '--resume', 'old'])


def test_control_and_config_path_restrictions(config):
    c, path, control = config
    assert op.control_path(control, c) == control
    for candidate in (Path(c['runs_root']), path.parent / 'elsewhere', Path(c['runs_root']) / '20261025_c_ept_s4a_1' / 'ctl'):
        with pytest.raises(op.OperatorError):
            op.control_path(candidate, c)
    with pytest.raises(op.OperatorError):
        op.config_path(op.ROOT / 'configs/ept_training.example.json')


@pytest.mark.parametrize('filename', ['run.json', 'run.json.tmp', 'operator_start.once'])
def test_start_once_refuses_prior_attempt_without_overwrite(config, filename):
    c, _, control = config
    control.mkdir()
    old = control / filename
    old.write_bytes(b'prior attempt, including corrupt metadata')
    with primitive.run_lock(control), pytest.raises(primitive.RuntimeRefused):
        runner._reserve(control, c)
    assert old.read_bytes() == b'prior attempt, including corrupt metadata'


def test_operator_claim_does_not_conflict_with_controller_token(config):
    c, _, control = config
    control.mkdir()
    (control / 'launch.once').write_bytes(b'controller owned')
    with primitive.run_lock(control):
        runner._reserve(control, c)
    assert (control / 'operator_start.once').is_file()
    assert (control / 'launch.once').read_bytes() == b'controller owned'
    with primitive.run_lock(control), pytest.raises(primitive.RuntimeRefused):
        runner._reserve(control, c)


def probe():
    return dict(running=False, matched=[], bridges_ignored=0, scanned=1, cmdline_unreadable=0, rule='synthetic')


@pytest.fixture
def composed(config, tmp_path, monkeypatch):
    """Real stage, OS lock, row/batch/stats and train-stage primitives; trainer stub."""
    import torch
    from c_ept import runlog
    from c_ept.workflows import run_s1
    from c_ept.ept import data, train, scales, loss, checkpoints as cp, epoch_metrics as em
    from c_ept.tests.ept_toys import build_ept_world, toy_static
    from c_ept.tests.test_ept_epoch_metrics import fixture_state
    c, path, control = config
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    world = build_ept_world(tmp_path / 'toy', n_events=3, n_edges=2)
    bundle = SimpleNamespace(s1=world.s1, s2=world.s2, labels=world.labels, static=toy_static())
    source = Path(c['s3']) / scales.SCALES_NAME
    source.write_text('{}')
    meta = {'paths': {'s1': str(world.s1.root), 's2': str(world.s2.root), 's3': c['s3']},
            'expected_provenance': {'synthetic': True}, 'terrain_files_sha256': {'toy': 'b' * 64}}
    code = dict(code_commit='a' * 40, code_dirty=False, repo_root=str(op.ROOT))
    events = []
    monkeypatch.setattr(op, 'preflight', lambda c: events.append('preflight'))
    monkeypatch.setattr(primitive, 'APPROVED_RUNS_ROOT', Path(c['runs_root']))
    monkeypatch.setattr(primitive, 'repository_state', lambda: events.append('source_guard') or code)
    monkeypatch.setattr(primitive, 'inspect_inputs', lambda args: events.append('input_guard') or meta)
    monkeypatch.setattr(primitive, '_load_inputs', lambda metadata: events.append('load_inputs') or bundle)
    monkeypatch.setattr(run_s1, 'require_no_strata', lambda: events.append('E6') or probe())
    monkeypatch.setattr(runlog, 'register', lambda *args: events.append('register'))
    # Test process stays native1, GPUOFF. CUDA availability/telemetry are metadata stubs.
    monkeypatch.setattr(train, 'deterministic_cpu', lambda threads, seed: events.append(('deterministic', threads, seed)))
    monkeypatch.setattr(primitive, '_runtime_fingerprint', lambda policy: {**policy, 'dtype': 'torch.float32', 'threads': 8, 'deterministic_algorithms': True})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 0)
    monkeypatch.setattr(primitive, '_cuda_sync', lambda device: None)
    monkeypatch.setattr(scales, 'load_scales', lambda p, expected_provenance: {key: 1. for key in loss.BLOCKS})
    fit_stats, proxy_matches, stage = data.fit_stats, data.proxy_matches, primitive._stage
    monkeypatch.setattr(data, 'fit_stats', lambda *args: events.append('whole_fit_stats') or fit_stats(*args))
    monkeypatch.setattr(data, 'proxy_matches', lambda *args: events.append(('proxy', args[1], args[2])) or proxy_matches(*args))
    monkeypatch.setattr(primitive, '_stage', lambda *args: events.append(('stage', args[3])) or stage(*args))
    write = primitive._write
    def ordered_write(target, doc):
        events.append(('write', target.name))
        write(target, doc)
    monkeypatch.setattr(primitive, '_write', ordered_write)

    def trainer_stub(cfg, training, folder, *, device, epochs, check, seed):
        assert cfg.patience == c['patience'] == 2
        assert (cfg.encoder, cfg.lr, cfg.weight_decay, cfg.warmup_steps) == (primitive._model_config('S'), .001, .01, 10)
        assert (device, epochs, seed) == ('cuda', 30, 20261022)
        assert training.fit.rows.role == 'FIT' and training.stop.rows.role == 'STOP'
        assert training.fit.batch_size == training.stop.batch_size == 32
        assert (control / 'run.json').exists()
        run, manifest, identity, digest = op.bound_run(control, c)
        process = op.read_json(run / 'PROCESS.json')
        op.validate_process_metadata(process, run, identity, digest)
        assert process['state'] == 'running'
        assert manifest['stages'] == op.STAGES
        check()
        events.append('train_stub')
        head, loop = fixture_state((4., 4., 4.), horizon=epochs)
        head['seed'] = seed
        head['identity'] = {**train._identity(cfg, training, training.static, epochs, seed, 'b' * 64, torch.device('cpu')),
                            'device': 'cuda', 'threads': 8, 'deterministic_algorithms': True}
        cp.save_checkpoint(folder, 3, {'last.pt': b'last', 'best.pt': b'best', **em.encode(loop, head)}, head)
        return SimpleNamespace(epochs_run=3, best_epoch=0, epoch_seconds=[0.] * 3, load_seconds=[0.] * 3,
                               step_seconds=[0.] * 3, stop_seconds=[0.] * 3, timing_components={}, seconds=0., peak_rss_bytes=1)
    monkeypatch.setattr(train, 'train', trainer_stub)
    return c, path, control, events, meta


def test_real_guard_stage_train_composition_one_fit_and_ref_before_training(composed, monkeypatch):
    c, path, control, events, meta = composed
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    assert runner.main(['--config', str(path), '--control', str(control), '--operator-config-sha256', op.config_digest(c)]) == 0
    assert [item for item in events if isinstance(item, tuple) and item[0] == 'stage'] == [('stage', name) for name in op.STAGES]
    assert events.count('train_stub') == events.count('whole_fit_stats') == 1
    assert ('proxy', .25, 20261022) in events
    assert ('deterministic', 8, 20261022) in events
    first = events.index(('write', 'RUN.json'))
    assert first < events.index(('write', 'PROCESS.json')) < events.index(('write', 'run.json')) < events.index('load_inputs') < events.index('train_stub')
    assert events.count('source_guard') > 4 and events.count('input_guard') > 4 and events.count('E6') > 4
    run, manifest, _, _ = op.bound_run(control, c)
    assert set(manifest['results']) == set(op.STAGES)
    assert manifest['results']['train_S_cuda']['data']['termination_reason'] == 'early_stopping'
    assert not (run / 'COST.json').exists()
    assert op.status(control, c)['state'] == 'recorded_completion'
    before = (control / 'run.json').read_bytes()
    assert runner.main(['--config', str(path), '--control', str(control), '--operator-config-sha256', op.config_digest(c)]) == 2
    assert (control / 'run.json').read_bytes() == before and events.count('train_stub') == 1


@pytest.mark.parametrize('guard', ['repository_state', 'inspect_inputs', '_output_paths', '_e6'])
def test_existing_guards_are_not_bypassed(composed, monkeypatch, guard):
    c, path, control, events, meta = composed
    def refused(*args, **kw):
        raise primitive.RuntimeRefused('synthetic ' + guard)
    monkeypatch.setattr(primitive, guard, refused)
    assert runner.main(['--config', str(path), '--control', str(control), '--operator-config-sha256', op.config_digest(c)]) == 2
    assert 'train_stub' not in events and not (control / 'run.json').exists()
    assert (control / 'operator_start.once').exists() == (guard in ('repository_state', '_e6'))


def test_failure_preserves_bound_process_and_ref_no_retry(composed, monkeypatch):
    c, path, control, events, _ = composed
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    def fail(meta):
        raise ValueError('synthetic loader failure after ref')
    monkeypatch.setattr(primitive, '_load_inputs', fail)
    assert runner.main(['--config', str(path), '--control', str(control), '--operator-config-sha256', op.config_digest(c)]) == 2
    run, manifest, identity, digest = op.bound_run(control, c)
    process = op.read_json(run / 'PROCESS.json')
    op.validate_process_metadata(process, run, identity, digest)
    assert process['state'] == 'failed' and process['exit_code'] == 2
    assert 'train_stub' not in events


@pytest.mark.parametrize('damage', ['patience', 'batch_size', 'cuda', 'runtime', 'inputs'])
def test_training_refused_on_guard_drift(composed, monkeypatch, damage):
    c, path, control, events, meta = composed
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    from c_ept.ept import train
    if damage == 'patience':
        monkeypatch.setattr(train, 'TrainConfig', lambda *args: SimpleNamespace(patience=3))
    elif damage == 'cuda':
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    elif damage == 'batch_size':
        c['batch_size'] = 16
        path.write_text(json.dumps(c))
    else:
        count = []
        if damage == 'runtime':
            def drift(policy):
                count.append(1)
                return dict(dtype='torch.float32', threads=8, deterministic_algorithms=True, generation=len(count))
            monkeypatch.setattr(primitive, '_runtime_fingerprint', drift)
        else:
            def drift(args):
                count.append(1)
                return meta if len(count) == 1 else {**meta, 'changed': True}
            monkeypatch.setattr(primitive, 'inspect_inputs', drift)
    assert runner.main(['--config', str(path), '--control', str(control), '--operator-config-sha256', op.config_digest(c)]) == 2
    assert 'train_stub' not in events


def test_native_bootstrap_fresh_process_before_heavy_import(config):
    _, path, control = config
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', CUDA_VISIBLE_DEVICES='')
    code = "from c_ept.workflows import training_operator as op; import sys; assert not any(k in sys.modules for k in ('torch','numpy','scipy')); sys.exit(op.main(sys.argv[1:]))"
    result = subprocess.run([sys.executable, '-B', '-c', code, '--config', str(path), '--control', str(control)], env=env, capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 2 and 'KMP_DUPLICATE_LIB_OK' in result.stderr
    assert not control.exists()


@pytest.mark.parametrize('damage', ['ref', 'process', 'run_digest', 'profile', 'json', 'stage'])
def test_status_rejects_corrupt_binding_without_runtime_or_payload(config, monkeypatch, damage):
    c, _, control = config
    control.mkdir()
    run = Path(c['runs_root']) / '20261025_c_ept_ept_pilot_1'
    run.mkdir()
    identity = {'args': c, 'code': {'code_commit': 'a' * 40, 'code_dirty': False, 'repo_root': str(op.ROOT)},
                'operator_config_sha256': op.config_digest(c), 'runtime': {}, 'inputs': {}}
    manifest = dict(schema=op.RUN_SCHEMA, stage='ept_training_single', operator_api_version=1, identity=identity,
                    run_identity_sha256=op.config_digest(identity), code_commit='a' * 40, code_dirty=False,
                    runtime={}, operator_config_sha256=op.config_digest(c), stages=op.STAGES, results={})
    process = runner._process_record(run, manifest, 'running')
    ref = dict(run=str(run), code_commit='a' * 40, operator_config_sha256=op.config_digest(c),
               run_identity_sha256=manifest['run_identity_sha256'])
    if damage == 'ref': ref['run_identity_sha256'] = 'b' * 64
    if damage == 'process': process['code_commit'] = 'b' * 40
    if damage == 'run_digest': manifest['run_identity_sha256'] = 'b' * 64
    if damage == 'profile': identity['args'] = {**c, 'patience': 3}
    if damage == 'stage': manifest['stages'] = [*op.STAGES, 'infer']
    for target, record in ((control/'run.json', ref), (run/'RUN.json', manifest), (run/'PROCESS.json', process)):
        target.write_text(json.dumps(record), encoding='utf-8')
    if damage == 'json': (run/'RUN.json').write_text('{corrupt')
    monkeypatch.setattr(op, 'preflight', lambda *a: pytest.fail('Status preflight'))
    monkeypatch.setattr(op, 'prepare_native_runtime', lambda: pytest.fail('Status native bootstrap'))
    with pytest.raises(op.OperatorError):
        op.status(control, c)


@pytest.mark.parametrize('damage', ['patience', 'batch_size', 'device', 'csv'])
def test_final_head_identity_and_metrics_checked_without_payload(config, tmp_path, monkeypatch, damage):
    from c_ept.ept import checkpoints as cp, epoch_metrics as em
    from c_ept.tests.test_ept_epoch_metrics import fixture_state
    c, _, _ = config
    head, loop = fixture_state((4., 4., 4.), horizon=30)
    head['seed'] = c['seed']
    head['identity'] = dict(patience=2, epochs=30, seed=c['seed'], lr=.001, weight_decay=.01, warmup_steps=10,
                            encoder=asdict(primitive._model_config('S')), device='cuda', dtype='torch.float32',
                            threads=8, deterministic_algorithms=True,
                            fit=dict(role='FIT', training=True, batch_size=32, seed=c['seed']),
                            stop=dict(role='STOP', training=False, batch_size=32, seed=c['seed']))
    if damage == 'batch_size': head['identity']['fit']['batch_size'] = 16
    if damage == 'device': head['identity']['device'] = 'cpu'
    cp.save_checkpoint(tmp_path, 3, {'last.pt': b'last', 'best.pt': b'best', **em.encode(loop, head)}, head)
    if damage == 'patience':
        c = {**c, 'patience': 3}
    if damage == 'csv':
        (tmp_path/'epoch_0003/epoch_metrics.csv').write_text('corrupt')
    original = Path.open
    def no_payload(path, *args, **kw):
        assert path.suffix != '.pt', 'metadata verifier read toy payload'
        return original(path, *args, **kw)
    monkeypatch.setattr(Path, 'open', no_payload)
    with pytest.raises((RuntimeRefusedError, RuntimeError)):
        runner._verify_final_head(tmp_path, c)


RuntimeRefusedError = primitive.RuntimeRefused


@pytest.mark.parametrize('action', ['Check', 'Start', 'Status'])
def test_wrapper_quoted_paths_and_exact_flags(tmp_path, action):
    mock = tmp_path / 'mock.ps1'
    capture = tmp_path / 'captured.json'
    mock.write_text("$args | ConvertTo-Json -Compress | Set-Content -LiteralPath $env:EPT_TEST_CAPTURE -Encoding utf8\nexit 7\n")
    config = tmp_path / 'external config.json'
    control = tmp_path / 'control directory'
    env = dict(os.environ, EPT_TEST_CAPTURE=str(capture))
    script = op.ROOT / 'run_ept_training.ps1'
    result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script),
                             '-Action', action.lower(), '-Config', str(config), '-Control', str(control), '-Python', str(mock)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
    assert result.returncode == 7, result.stderr
    assert json.loads(capture.read_text(encoding='utf-8-sig')) == ['-B', '-m', 'c_ept.workflows.training_operator',
                 '--action', action, '--config', str(config), '--control', str(control)]
