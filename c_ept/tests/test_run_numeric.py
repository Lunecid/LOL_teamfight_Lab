"""Tests of the numeric policy of the run kernel (``c_ept.run.numeric``; S4b-prep K6c, decisions D10 P6 and D32 item 5, perf audit section 8 item 3).

The policy is the T009 float32 policy: the default dtype float32, the highest float32 matmul precision, TF32 off for cuBLAS and cuDNN, cuDNN deterministic and not
benchmarking, deterministic algorithms with ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` set before CUDA exists, ``fill_uninitialized_memory`` off (audit item 5) and no autocast.
``apply`` sets it, reads the state back (versions, device name, attention-backend flags, threads, as Gate 0 recorded it) and refuses to go on when the state is not the policy.

These tests run on the CPU only and never create a CUDA context (a test asserts it; the device and cuDNN queries of the GPU branch are answered by stand-ins). The
process-wide settings that ``apply`` changes are given back after every test.

해석: 이 시험은 숫자 정책 하나(T009의 float32 정책)가 실제로 켜지고, 켜진 상태를 그대로 적어 돌려주며, 켜지지 않으면 거부하는지를 확인한다. CPU만 쓰고 CUDA 문맥은 만들지 않는다.
"""
import os

import pytest
import torch

from c_ept.run import guards
from c_ept.run import io as run_io
from c_ept.run import numeric, supervisor

SETTINGS = ('default_dtype', 'float32_matmul_precision', 'cuda_matmul_allow_tf32', 'cudnn_allow_tf32', 'cudnn_deterministic', 'cudnn_benchmark', 'deterministic_algorithms',
            'fill_uninitialized_memory', 'autocast_enabled', 'cublas_workspace_config')
OPPOSITE = {'float32_matmul_precision': 'high', 'cuda_matmul_allow_tf32': True, 'cudnn_allow_tf32': True, 'cudnn_deterministic': False, 'cudnn_benchmark': True,
            'deterministic_algorithms': False, 'fill_uninitialized_memory': True}


def read_flags() -> dict:
    """The process-wide settings that the policy sets, as they are now."""
    import torch.utils.deterministic as deterministic
    return {'float32_matmul_precision': torch.get_float32_matmul_precision(), 'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'fill_uninitialized_memory': deterministic.fill_uninitialized_memory}


def write_flags(flags: dict) -> None:
    import torch.utils.deterministic as deterministic
    torch.set_float32_matmul_precision(flags['float32_matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32 = flags['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = flags['cudnn_allow_tf32']
    torch.backends.cudnn.deterministic = flags['cudnn_deterministic']
    torch.backends.cudnn.benchmark = flags['cudnn_benchmark']
    torch.use_deterministic_algorithms(flags['deterministic_algorithms'])
    deterministic.fill_uninitialized_memory = flags['fill_uninitialized_memory']


@pytest.fixture(autouse=True)
def restore_process_state(monkeypatch):
    """``apply`` changes process-wide settings and one environment variable; give them back after each test. The variable starts at a value that is not the policy's, so that
    the tests see ``apply`` set it (monkeypatch puts back what was there, or nothing)."""
    before, threads = read_flags(), torch.get_num_threads()
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    yield
    write_flags(before)
    torch.set_num_threads(threads)


@pytest.fixture
def no_gpu_queries(monkeypatch):
    """The device name and the cuDNN version are what the GPU branch asks; on a machine whose GPU is visible they would create a CUDA context. The stand-ins raise what torch
    raises without a device (an ``AssertionError`` and a ``ValueError``), so ``gate0.numeric_record`` answers None for both."""
    def no_device(index=0):
        raise AssertionError('Invalid device id')

    def no_cudnn():
        raise ValueError('no cuDNN')
    monkeypatch.setattr(torch.cuda, 'get_device_name', no_device)
    monkeypatch.setattr(torch.backends.cudnn, 'version', no_cudnn)


# ---- the requested policy ---------------------------------------------------------------------------------------------------------------------
def test_the_requested_policy_is_the_t009_float32_policy_with_the_two_items_of_the_audit():
    assert tuple(numeric.REQUIRED) == SETTINGS
    assert dict(numeric.REQUIRED) == {'default_dtype': 'torch.float32', 'float32_matmul_precision': 'highest', 'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': False,
                                      'cudnn_deterministic': True, 'cudnn_benchmark': False, 'deterministic_algorithms': True, 'fill_uninitialized_memory': False,
                                      'autocast_enabled': False, 'cublas_workspace_config': ':4096:8'}


def test_the_policy_agrees_with_the_function_that_t009_and_n2_call_for_the_same_settings():
    """``comparison.models.configure_float32`` is what T009 trained and evaluated under. After it and after ``apply`` the shared settings are the same."""
    from c_ept.comparison import models
    write_flags({**OPPOSITE, 'float32_matmul_precision': 'medium'})
    models.configure_float32()
    t009 = read_flags()
    write_flags({**OPPOSITE, 'float32_matmul_precision': 'medium'})
    numeric.apply('cpu')
    ours = read_flags()
    shared = [name for name in t009 if name != 'fill_uninitialized_memory']                 # the one setting that T009 never set (audit item 5)
    assert {name: ours[name] for name in shared} == {name: t009[name] for name in shared} and ours['fill_uninitialized_memory'] is False


def test_the_policy_names_the_same_settings_as_the_gate_0_record_and_adds_two():
    from c_ept.run import gate0
    assert set(gate0.REQUIRED_NUMERIC) <= set(numeric.REQUIRED) and set(numeric.REQUIRED) - set(gate0.REQUIRED_NUMERIC) == {'fill_uninitialized_memory', 'cublas_workspace_config'}
    assert all(numeric.REQUIRED[name] == wanted for name, wanted in gate0.REQUIRED_NUMERIC.items())


def test_the_requested_record_is_strict_json_with_the_hash_of_the_policy():
    record = numeric.requested_record()
    assert record == {'requested': dict(numeric.REQUIRED), 'requested_sha256': run_io.canonical_sha256(dict(numeric.REQUIRED))}
    assert numeric.requested_record() == record and len(record['requested_sha256']) == 64
    changed = {**numeric.REQUIRED, 'cudnn_benchmark': True}
    assert run_io.canonical_sha256(changed) != record['requested_sha256']                    # a changed policy is another hash


# ---- applying and measuring ---------------------------------------------------------------------------------------------------------------------
def test_apply_sets_every_setting_of_the_policy_whatever_it_was_before():
    write_flags(OPPOSITE)
    record = numeric.apply('cpu')
    assert read_flags() == {name: numeric.REQUIRED[name] for name in read_flags()}
    assert all(record['measured'][name] == numeric.REQUIRED[name] for name in SETTINGS)
    assert record['requested'] == dict(numeric.REQUIRED) and record['requested_sha256'] == numeric.requested_record()['requested_sha256']
    assert record['measured_sha256'] == run_io.canonical_sha256(record['measured']) and set(record) == {'requested', 'requested_sha256', 'measured', 'measured_sha256'}


def test_the_measured_state_has_the_fields_of_the_gate_0_record_the_fill_flag_and_the_device(no_gpu_queries):
    from c_ept.run import gate0
    record = numeric.apply('cpu')['measured']
    gate0_record = gate0.numeric_record()
    assert set(record) == set(gate0_record) | {'fill_uninitialized_memory', 'device'}
    assert {name: record[name] for name in gate0_record} == gate0_record                          # every field of the Gate 0 record has the value that Gate 0 would have recorded
    assert record['device'] == 'cpu' and record['device_name'] is None and record['cudnn_version'] is None
    assert set(record['sdp_backends']) <= {'flash', 'mem_efficient', 'math', 'cudnn'} and all(isinstance(flag, bool) for flag in record['sdp_backends'].values())
    assert record['amp_used_by_this_code'] is False and record['compile_used_by_this_code'] is False
    assert record['torch_version'] == torch.__version__ and record['threads'] == torch.get_num_threads() and isinstance(record['cuda_version'], (str, type(None)))
    run_io.canonical_sha256(record)                                                       # strict JSON


def test_a_cpu_run_never_creates_a_cuda_context():
    numeric.apply('cpu')
    assert torch.cuda.is_initialized() is False


def test_the_gpu_branch_reads_the_device_name_and_the_cudnn_version(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index=0: 'Fake GPU 5080')
    monkeypatch.setattr(torch.backends.cudnn, 'version', lambda: 91002)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    measured = numeric.apply('cuda')['measured']
    assert (measured['device'], measured['device_name'], measured['cudnn_version']) == ('cuda', 'Fake GPU 5080', 91002)
    assert all(measured[name] == numeric.REQUIRED[name] for name in SETTINGS)


def test_a_gpu_that_cannot_be_asked_is_recorded_as_none_not_guessed(no_gpu_queries, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    measured = numeric.apply('cuda')['measured']
    assert measured['device'] == 'cuda' and measured['device_name'] is None and measured['cudnn_version'] is None


def test_measure_reads_the_state_back_without_setting_anything():
    write_flags(OPPOSITE)
    before = read_flags()
    measured = numeric.measure('cpu')
    assert read_flags() == before and measured['float32_matmul_precision'] == 'high' and measured['deterministic_algorithms'] is False
    assert any('float32_matmul_precision' in text for text in numeric.differences(measured))


def test_differences_names_every_setting_that_is_not_the_policy_with_both_values():
    measured = {**numeric.apply('cpu')['measured'], 'cudnn_benchmark': True, 'float32_matmul_precision': 'high'}
    texts = numeric.differences(measured)
    assert len(texts) == 2 and any('cudnn_benchmark' in t and 'True' in t and 'False' in t for t in texts) and any('float32_matmul_precision' in t and "'high'" in t for t in texts)
    assert numeric.differences(numeric.apply('cpu')['measured']) == []
    assert len(numeric.differences({})) == len(SETTINGS)                                      # a setting that was not measured is not the policy
    good = numeric.apply('cpu')['measured']
    for name, wrong in (('cudnn_benchmark', 0), ('cudnn_deterministic', 1), ('deterministic_algorithms', 1.0), ('cuda_matmul_allow_tf32', 0)):
        assert len(numeric.differences({**good, name: wrong})) == 1, name                       # 0 == False and 1 == True in Python, but a number is not a flag: another type is a difference


# ---- the refusals -------------------------------------------------------------------------------------------------------------------------------
def test_a_setting_that_cannot_be_held_is_refused_naming_it_and_the_refusal_is_exit_code_2(monkeypatch):
    write_flags(OPPOSITE)
    with monkeypatch.context() as patch:                                                       # undone before the fixture gives the flags back (it uses the same function)
        patch.setattr(torch, 'set_float32_matmul_precision', lambda value: None)               # a library that ignores the call: the state stays what it was ('high')
        with pytest.raises(numeric.NumericRefused, match='float32_matmul_precision') as raised:
            numeric.apply('cpu')
    assert isinstance(raised.value, guards.GateRefused) and raised.value.gate == 'environment' and supervisor.exit_code_for(raised.value) == 2
    assert 'inconsistent' in str(raised.value)                    # torch 2.10 refuses to answer when the legacy flag (False) and the new setting ('high') disagree: recorded, not raised


def test_a_state_that_torch_will_not_read_back_is_a_difference_not_an_error_of_the_measurement(monkeypatch):
    def refuses():
        raise RuntimeError('PyTorch is checking the matmul precision without a specific backend name ...')
    monkeypatch.setattr(torch, 'get_float32_matmul_precision', refuses)
    measured = numeric.measure('cpu')
    assert measured['float32_matmul_precision'] == numeric.MIXED_PRECISION and any('float32_matmul_precision' in text for text in numeric.differences(measured))


def test_another_default_dtype_is_refused_and_nothing_is_set_or_changed():
    before = torch.get_default_dtype()
    write_flags(OPPOSITE)
    torch.set_default_dtype(torch.float64)
    try:
        with pytest.raises(numeric.NumericRefused, match='default_dtype'):
            numeric.apply('cpu')
        assert torch.get_default_dtype() == torch.float64 and read_flags() == OPPOSITE                  # not changed behind the caller's back, and no flag was set
    finally:
        torch.set_default_dtype(before)


def test_an_ambient_autocast_is_refused_and_nothing_is_set():
    write_flags(OPPOSITE)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with pytest.raises(numeric.NumericRefused, match='autocast_enabled'):
            numeric.apply('cpu')
    assert read_flags() == OPPOSITE
    assert numeric.apply('cpu')['measured']['autocast_enabled'] is False


@pytest.mark.parametrize('device', ['tpu', None, 'cuda:0', '', 'CPU', 0])
def test_an_unknown_device_is_refused_before_anything_is_set(device):
    write_flags(OPPOSITE)
    with pytest.raises(numeric.NumericRefused, match='device'):
        numeric.apply(device)
    assert read_flags() == OPPOSITE


def test_the_cublas_workspace_is_set_while_cuda_does_not_exist_yet_and_refused_after(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    assert numeric.apply('cpu')['measured']['cublas_workspace_config'] == ':4096:8' and os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'          # another value: set
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG')
    assert numeric.apply('cpu')['measured']['cublas_workspace_config'] == ':4096:8' and os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'          # no value: set
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)                             # the context exists: cuBLAS has read its workspace setting already
    assert numeric.apply('cpu')['measured']['cublas_workspace_config'] == ':4096:8'              # the right value: nothing to fix
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    write_flags(OPPOSITE)
    with pytest.raises(numeric.NumericRefused, match='CUBLAS_WORKSPACE_CONFIG') as raised:
        numeric.apply('cpu')
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':16:8' and read_flags() == OPPOSITE          # too late to change it: refused, nothing edited
    assert supervisor.exit_code_for(raised.value) == 2
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG')
    with pytest.raises(numeric.NumericRefused, match='CUBLAS_WORKSPACE_CONFIG'):                   # no value and a context: the library has the default workspace, which is not the policy's
        numeric.apply('cpu')


def test_the_refusal_is_a_gate_refusal_that_survives_pickling():
    import pickle
    refusal = numeric.NumericRefused('environment', 'numeric policy: example')
    again = pickle.loads(pickle.dumps(refusal))
    assert isinstance(again, numeric.NumericRefused) and again.gate == 'environment' and again.reason == 'numeric policy: example'


def test_the_policy_is_idempotent_and_gives_the_same_record_twice():
    first, second = numeric.apply('cpu'), numeric.apply('cpu')
    assert first == second
