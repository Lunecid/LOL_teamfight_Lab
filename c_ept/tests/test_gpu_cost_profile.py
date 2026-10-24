"""GPU-only operator policy: synthetic metadata and stub dispatch, no training."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

REPO = Path(__file__).resolve().parents[2]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


op = load_module('gpu_profile_operator', 'c_ept/workflows/s4a_operator.py')


def example(version=2):
    name = 's4a.gpu-cost.example.json' if version == 2 else 's4a.example.json'
    return json.loads((REPO / 'configs' / name).read_text(encoding='utf-8'))


def save_config(tmp_path, config):
    path = tmp_path / 'operator.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    return path


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('version', [1, 2])
def test_valid_profile_preserves_config_and_digest(tmp_path, version):
    config = example(version)
    before = copy.deepcopy(config)
    path = save_config(tmp_path, config)
    assert op.load_config(path, require_inputs=False) == before
    old_digest = hashlib.sha256(json.dumps(config, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert op.config_digest(config) == old_digest
    assert op.config_digest(dict(reversed(list(config.items())))) == old_digest
    assert config == before  # v1 never acquires implicit profile/default fields


INVALID_V2 = [
    ('schema', 's4a-operator-v3'), ('schema', 's4a-operator-v1'),
    ('schema', None), ('schema', {}),
    ('measurement_profile', None), ('measurement_profile', 'full'),
    ('measurement_profile', 'gpu'), ('measurement_profile', ['gpu_cost']),
    ('measurement_profile', 'GPU_COST'), ('extra', True),
    ('devices', ['cpu']), ('devices', ['cpu', 'cuda']),
    ('devices', ['cuda', 'cpu']), ('devices', []),
    ('devices', ['cuda', 'cuda']), ('devices', 'cuda'),
    ('sizes', ['S']), ('sizes', ['M']), ('sizes', []),
    ('sizes', ['M', 'S']), ('sizes', ['S', 'M', 'L']),
    ('epochs', 1), ('epochs', 3), ('epochs', True), ('epochs', 2.0),
    ('share', .5), ('share', True), ('share', '0.25'),
    ('dev_sample', 201), ('dev_sample', True), ('dev_sample', 200.0),
    ('task7_integration_verified', 'yes'),
    ('runs_root', 'D:/not-approved/runs'),
    ('catalog_db', 'C:/outside/catalog.sqlite'), ('python', 'relative.exe'),
    ('s1', 'relative'), ('s2', ''), ('s3', None),
]


@pytest.mark.parametrize('key,value', INVALID_V2)
def test_v2_rejects_scope_drift_before_preflight(tmp_path, monkeypatch, key, value):
    config = example()
    config[key] = value
    path = save_config(tmp_path, config)
    monkeypatch.setattr(op, 'prepare_native_runtime', lambda: {})
    monkeypatch.setattr(op, 'preflight', lambda c: pytest.fail('invalid config reached preflight'))
    monkeypatch.setattr(op.importlib, 'import_module', lambda name: pytest.fail('runner imported'))
    before = snapshot(tmp_path)
    assert op.main(['--action', 'Start', '--config', str(path)]) == 2
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('key', list(example()))
def test_v2_requires_every_config_key(tmp_path, key):
    config = example()
    del config[key]
    with pytest.raises(op.OperatorError):
        op.load_config(save_config(tmp_path, config), require_inputs=False)


@pytest.mark.parametrize('key,value', [
    ('devices', ['cuda']), ('devices', ['cpu']),
    ('sizes', ['S']), ('sizes', ['M']), ('epochs', 3),
    ('share', .5), ('dev_sample', 201),
    ('measurement_profile', 'gpu_cost'), ('schema', 's4a-operator-v2'),
])
def test_old_v1_out_of_scope_values_still_refused(tmp_path, key, value):
    config = example(1)
    config[key] = value
    with pytest.raises(op.OperatorError):
        op.load_config(save_config(tmp_path, config), require_inputs=False)


@pytest.fixture
def synthetic_config(tmp_path, monkeypatch):
    config = example()
    config['python'] = sys.executable
    for key in ('s1', 's2', 's3'):
        folder = tmp_path / key
        folder.mkdir()
        config[key] = str(folder)
    for key in ('s1', 's2'):
        (Path(config[key]) / 'STORE.json').write_text('{}', encoding='utf-8')
    for name in ('RUN.json', 'STAGES.json'):
        (Path(config['s3']) / name).write_text('{}', encoding='utf-8')
    monkeypatch.setattr(op, 'prepare_native_runtime', lambda: {'scope': 'synthetic'})
    monkeypatch.setattr(op, 'runtime_check', lambda: None)
    monkeypatch.setattr(op, 'code_check', lambda: 'a' * 40)
    monkeypatch.setattr(op, 'gpu_metadata', lambda: ['synthetic CUDA metadata'])
    monkeypatch.setattr(op, 'e6_check', lambda: None)
    return config, save_config(tmp_path, config)


def test_gpu_check_is_readonly_and_never_loads_runner(synthetic_config, tmp_path, monkeypatch):
    config, path = synthetic_config
    monkeypatch.setattr(op.importlib, 'import_module', lambda name: pytest.fail('Check loaded runner'))
    before = snapshot(tmp_path)
    assert op.main(['--config', str(path)]) == 0
    assert snapshot(tmp_path) == before


def test_gpu_start_dispatches_cuda_only_to_stub(synthetic_config, tmp_path, monkeypatch):
    config, path = synthetic_config
    captured = []
    def import_stub(name):
        assert name == 'c_ept.run_s4a'
        return types.SimpleNamespace(OPERATOR_API_VERSION=1,
            main=lambda argv: captured.append(argv) or 17)
    monkeypatch.setattr(op.importlib, 'import_module', import_stub)
    before = snapshot(tmp_path)
    assert op.main(['--action', 'Start', '--config', str(path)]) == 17
    assert len(captured) == 1
    argv = captured[0]
    assert argv[argv.index('--sizes') + 1:argv.index('--devices')] == ['S', 'M']
    assert argv[argv.index('--devices') + 1:argv.index('--operator-config-sha256')] == ['cuda']
    assert argv[-2:] == ['--operator-config-sha256', op.config_digest(config)]
    assert '--resume' not in argv
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('guard', ['runtime_check', 'code_check', 'gpu_metadata', 'e6_check', 'task7'])
def test_gpu_profile_keeps_preflight_guards(synthetic_config, tmp_path, monkeypatch, guard):
    config, path = synthetic_config
    def refuse():
        raise op.OperatorError('synthetic guard refusal')
    if guard == 'task7':
        config['task7_integration_verified'] = False
        save_config(tmp_path, config)
    else:
        monkeypatch.setattr(op, guard, refuse)
    monkeypatch.setattr(op.importlib, 'import_module', lambda name: pytest.fail('guard failure dispatched'))
    before = snapshot(tmp_path)
    assert op.main(['--action', 'Start', '--config', str(path)]) == 2
    assert snapshot(tmp_path) == before


def test_profile_is_bound_in_digest():
    config = example()
    assert op.config_digest(config) != op.config_digest(example(1))
    for changed in (dict(config, measurement_profile='full'),
                    {k: v for k, v in config.items() if k != 'measurement_profile'}):
        assert op.config_digest(config) != op.config_digest(changed)


@pytest.mark.parametrize('drift', [None, 'profile_digest', 'v1_config', 'args', 'runtime', 'inputs'])
def test_real_runner_parse_and_resume_identity_remain_strict(monkeypatch, drift):
    # Import-only runner definitions: never main, stores, source verification or tensors.
    runner = load_module('gpu_profile_runner', 'c_ept/workflows/run_s4a.py')
    monkeypatch.setitem(sys.modules, 'c_ept.run_s3', types.SimpleNamespace(
        check_resume_code=lambda manifest, code: None))  # isolate config/identity guard
    config = example()
    digest = op.config_digest(config)
    argv = [*op.runner_args(config), '--operator-config-sha256', digest]
    args = runner._parser().parse_args(argv)
    normal, bound_digest = runner._normal_args(args)
    assert normal['sizes'] == ['S', 'M'] and normal['devices'] == ['cuda']
    assert (normal['epochs'], normal['share'], normal['dev_sample']) == (2, .25, 200)
    assert (normal['batch_size'], normal['lr'], normal['weight_decay'],
            normal['warmup_steps'], normal['seed']) == (32, .001, .01, 10, 20261022)
    assert bound_digest == digest and runner.OPERATOR_API_VERSION == 1
    resumed = runner._parser().parse_args([*op.runner_args(config, Path(config['runs_root']) / '20261024_c_ept_s4a_2'),
        '--operator-config-sha256', digest])
    assert runner._normal_args(resumed) == (normal, digest)
    identity = {'args': normal, 'operator_config_sha256': digest,
        'code': {'code_commit': 'a' * 40, 'code_dirty': False, 'repo_root': str(REPO)},
        'runtime': {'fixture': 'synthetic'}, 'inputs': {'fixture': 'synthetic'}}
    manifest = {'schema': runner.RUN_SCHEMA, 'stage': 's4a', 'operator_api_version': 1,
        'identity': copy.deepcopy(identity), 'run_identity_sha256': runner._digest(identity),
        'operator_config_sha256': digest, 'code_commit': 'a' * 40}
    before = copy.deepcopy(manifest)
    if drift == 'profile_digest':
        identity['operator_config_sha256'] = op.config_digest(dict(config, measurement_profile='full'))
    elif drift == 'v1_config':
        identity['operator_config_sha256'] = op.config_digest(example(1))
    elif drift == 'args':
        identity['args']['devices'] = ['cpu', 'cuda']
    elif drift in ('runtime', 'inputs'):
        identity[drift]['fixture'] = 'drift'
    if drift is None:
        runner.verify_resume(manifest, identity)
    else:
        with pytest.raises(runner.RuntimeRefused, match='identity differs'):
            runner.verify_resume(manifest, identity)
    assert manifest == before
