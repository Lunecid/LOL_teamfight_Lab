"""Numeric policy of the run kernel: the float32 settings of a run, applied, read back and recorded (S4b-prep K6c; decisions D10 P6 and D32 item 5; perf audit section 8 item 3).

D10 P6: "the reporting device is deterministic GPU (float32, deterministic algorithms) with a recorded numeric policy". Until K6c the EPT path never set or recorded the matmul
precision, the cuDNN flags or the attention backend: ``ept.train`` sets the deterministic algorithms and the seeds, and the float32 set-up of the comparison stack
(``comparison.models.configure_float32``) was called for N2 only (the ``model_spec`` branch of ``train._run``). A torch upgrade could switch the default of any of them without a
trace in the run. This module is the one place that says what the policy is. ``apply`` is called by the train, evaluate and measure runners before any model is built; it sets the
policy, reads the state of the process back and refuses to go on when the state is not the policy. The state that it returns is written to the stage records and, as the request, to
the ledger header (the policy and its hash are part of the identity of a run).

Policy (the settings of ``REQUIRED``)
-------------------------------------
    default dtype float32                                    checked, never changed (the reported runs are float32; another default is another run)
    ``torch.set_float32_matmul_precision('highest')``        IEEE float32 matmuls (no TF32 or bf16x3 shortcuts)
    TF32 off for cuBLAS and cuDNN                            ``torch.backends.cuda.matmul.allow_tf32 = cudnn.allow_tf32 = False``
    cuDNN deterministic on, benchmark off                    ``torch.backends.cudnn.deterministic = True``, ``.benchmark = False``
    ``torch.use_deterministic_algorithms(True)``             with ``CUBLAS_WORKSPACE_CONFIG=:4096:8``, set before the CUDA context exists (cuBLAS reads it once)
    ``torch.utils.deterministic.fill_uninitialized_memory = False``   audit item 5: a valid program never reads uninitialised memory, so no value changes
    autocast off                                             checked, never changed (an ambient autocast context is refused)
The attention backend flags (``sdp_backends``), the versions, the device name and the thread count are recorded, not set: they are the state of the library and the machine.

Symbols
-------
    R      the requested state ``REQUIRED``: ten settings, name -> value
    M      the measured state of the process, ``measure(d)``: the settings of R read back, and the recorded-only fields (SDP backends, versions, device, threads)
    d      the device of the run, ``'cpu'`` or ``'cuda'``; only a CUDA run asks the GPU for its name and the cuDNN version (a CPU run never creates a CUDA context)
    h_R    the hash of the policy, ``canonical_sha256(R)``;  h_M the hash of the measured state

Equations
---------
    (NUM-1)  apply(d):  check that the default dtype is float32 and that no autocast is on; hold ``CUBLAS_WORKSPACE_CONFIG = ':4096:8'`` (set it while CUDA does not exist, refuse
             when CUDA exists and the value differs); set the other settings of R; M := measure(d)
    (NUM-2)  accept  <=>  for every name in R:  M[name] = R[name] (same type)      otherwise ``NumericRefused`` (a ``GateRefused`` of the gate 'environment': exit code 2)
    (NUM-3)  record = {requested: R, requested_sha256: h_R, measured: M, measured_sha256: h_M}

해석
----
숫자 정책은 "같은 입력이 같은 비트를 내는 조건"을 한 곳에 적은 것이다. 지금까지 EPT 경로는 결정적 알고리즘과 시드만 켜고 행렬곱 정밀도, cuDNN 플래그, 어텐션 백엔드는 라이브러리 기본값에 맡겼으며
기록도 하지 않았다. torch 버전이 오르면 기본값이 조용히 바뀔 수 있다. ``apply``는 학습·평가·측정 단계가 모형을 만들기 전에 정책을 켜고, 켜진 상태를 다시 읽어 정책과 다르면 거부한다(종료 코드 2).
기본 dtype과 autocast는 켜는 것이 아니라 확인하는 것이다: 다른 값은 다른 실행이므로 몰래 고치지 않고 거부한다. ``CUBLAS_WORKSPACE_CONFIG``는 cuBLAS가 처음 불릴 때 한 번 읽으므로 CUDA 문맥이
생기기 전에만 정할 수 있다. 이미 문맥이 있고 값이 다르면 늦었으므로 고치지 않고 거부한다. CPU 실행은 GPU에 이름도 cuDNN 버전도 묻지 않는다(묻는 것만으로 CUDA 문맥이 생기기 때문이다).
어텐션 백엔드 플래그, 버전, 장치 이름, 스레드 수는 정책이 아니라 기계와 라이브러리의 상태이므로 기록만 한다.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from types import MappingProxyType

from c_ept.run import guards
from c_ept.run import io as run_io

DEVICES = ('cpu', 'cuda')
CUBLAS_VARIABLE = 'CUBLAS_WORKSPACE_CONFIG'
SDP_BACKENDS = ('flash', 'mem_efficient', 'math', 'cudnn')

# Ref: c_ept/comparison/models.py ``configure_float32`` (T009: float32 required, highest matmul precision, TF32 off for cuBLAS and cuDNN, cuDNN deterministic and not benchmarking,
# deterministic algorithms), c_ept/run/gate0.py ``numeric_record`` / ``REQUIRED_NUMERIC`` (the fields as Gate 0 recorded them), c_ept/ept/train.py ``_deterministic_gpu`` (CUBLAS_WORKSPACE_CONFIG
# ':4096:8'). PyTorch notes "Reproducibility": ``torch.use_deterministic_algorithms``, ``CUBLAS_WORKSPACE_CONFIG``, ``torch.utils.deterministic.fill_uninitialized_memory``.
REQUIRED = MappingProxyType({
    'default_dtype': 'torch.float32',
    'float32_matmul_precision': 'highest',
    'cuda_matmul_allow_tf32': False,
    'cudnn_allow_tf32': False,
    'cudnn_deterministic': True,
    'cudnn_benchmark': False,
    'deterministic_algorithms': True,
    'fill_uninitialized_memory': False,
    'autocast_enabled': False,
    'cublas_workspace_config': ':4096:8',
})


class NumericRefused(guards.GateRefused):
    """The numeric policy cannot be held in this process: a setting is not what the policy requires and cannot be set, or the device is not one of ``DEVICES``. A gate refusal of
    the gate 'environment' (exit code 2); it has the signature of ``GateRefused`` so that it survives pickling. The message names the setting, its value and the required value."""


def _refuse(reason: str):
    raise NumericRefused('environment', f'numeric policy: {reason}')


def _check_device(device) -> None:
    if not (isinstance(device, str) and device in DEVICES):
        _refuse(f'the device must be one of {DEVICES}')


def requested_record() -> dict:
    """``{'requested': R, 'requested_sha256': h_R}``: the policy as the ledger header holds it (what a run asks for; the measured state is in the stage records)."""
    requested = dict(REQUIRED)
    return {'requested': requested, 'requested_sha256': run_io.canonical_sha256(requested)}


def _probe(function):
    """``function()``, or None when it cannot answer (torch raises for the device name and the cuDNN version where no GPU is visible): a value that is not there is recorded as None."""
    try:
        return function()
    except Exception:                                                        # noqa: BLE001 - whatever torch does without a device means "not available here"
        return None


MIXED_PRECISION = 'inconsistent: the legacy and the new matmul precision APIs disagree'


def _matmul_precision(torch) -> str:
    """``torch.get_float32_matmul_precision()``, or ``MIXED_PRECISION`` where torch (2.9 and later) refuses to answer because the legacy flags (``allow_tf32``) and the new precision
    settings disagree: a state that is not the policy is a value to record, not an error of the measurement."""
    try:
        return torch.get_float32_matmul_precision()
    except RuntimeError:
        return MIXED_PRECISION


def measure(device: str) -> dict:
    """The state of the process now, as plain JSON, without setting anything: the settings of ``REQUIRED`` and, as Gate 0 recorded them (``gate0.numeric_record``), the attention
    backend flags (``sdp_backends``), ``amp_used_by_this_code`` and ``compile_used_by_this_code`` (properties of the code path: it has no autocast and no ``torch.compile`` call, named as
    such), the torch, CUDA and cuDNN versions, the thread count and the device name, plus ``fill_uninitialized_memory`` and ``device``. ``device_name`` and ``cudnn_version`` are None on a
    CPU run (the GPU is not asked: asking creates a CUDA context) and when the GPU cannot answer. ``NumericRefused`` for a device that is not one of ``DEVICES``.

    해석: 숫자 설정이 지금 어떤지를 있는 그대로 읽는다. CPU 실행은 GPU에 묻지 않는다."""
    _check_device(device)
    import torch
    import torch.utils.deterministic as deterministic
    cuda, sdp = device == 'cuda', {}
    for name in SDP_BACKENDS:
        if hasattr(torch.backends.cuda, f'{name}_sdp_enabled'):
            sdp[name] = bool(getattr(torch.backends.cuda, f'{name}_sdp_enabled')())
    return {'default_dtype': str(torch.get_default_dtype()), 'float32_matmul_precision': _matmul_precision(torch),
            'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32), 'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
            'cudnn_deterministic': bool(torch.backends.cudnn.deterministic), 'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
            'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()), 'fill_uninitialized_memory': getattr(deterministic, 'fill_uninitialized_memory', None),
            'sdp_backends': sdp, 'autocast_enabled': bool(torch.is_autocast_enabled()) or bool(torch.is_autocast_enabled('cpu')),
            'amp_used_by_this_code': False, 'compile_used_by_this_code': False, 'cublas_workspace_config': os.environ.get(CUBLAS_VARIABLE),
            'torch_version': torch.__version__, 'cuda_version': torch.version.cuda, 'cudnn_version': _probe(torch.backends.cudnn.version) if cuda else None,
            'threads': int(torch.get_num_threads()), 'device': device, 'device_name': _probe(lambda: torch.cuda.get_device_name(0)) if cuda else None}


def differences(measured: Mapping) -> list:
    """The settings of ``measured`` that are not the policy, as ``name=value (the policy requires value)`` texts, in the order of ``REQUIRED``; empty when the state is the policy. A
    setting that is missing from ``measured`` is a difference (None is not what the policy requires), and a value of another type is one too (``0`` is not ``False``)."""
    texts = []
    for name, wanted in REQUIRED.items():
        have = measured.get(name)
        if not (type(have) is type(wanted) and have == wanted):
            texts.append(f'{name}={have!r} (the policy requires {wanted!r})')
    return texts


def _hold_cublas_workspace(torch) -> None:
    """``CUBLAS_WORKSPACE_CONFIG`` is the policy's value: set it while no CUDA context exists, refuse when one exists and the value differs (cuBLAS has read it already)."""
    wanted, have = REQUIRED['cublas_workspace_config'], os.environ.get(CUBLAS_VARIABLE)
    if have == wanted:
        return
    if torch.cuda.is_initialized():
        _refuse(f'{CUBLAS_VARIABLE}={have!r} (the policy requires {wanted!r}) and the CUDA context exists already, so cuBLAS has read its workspace setting; '
                'start a new process whose environment has the value')
    os.environ[CUBLAS_VARIABLE] = wanted


def apply(device: str) -> dict:
    """Set the numeric policy, read the state of the process back and return the record ``{'requested', 'requested_sha256', 'measured', 'measured_sha256'}`` (NUM-3). Call it before any
    model is built. ``NumericRefused`` (exit code 2) when the device is not one of ``DEVICES``, when the default dtype is not float32 or an autocast is on (nothing is set then: these are
    checked, not changed), when the CUBLAS workspace setting cannot be held, or when the state after the calls is not the policy (a library that ignores a call).

    수식: (NUM-1) 확인(dtype, autocast) → CUBLAS 작업 공간 → 나머지 설정 켜기 → M = measure(d);  (NUM-2) 모든 이름에서 M = R이어야 통과.
    해석: 모형을 만들기 전에 정책을 켜고, 켜진 뒤의 상태를 다시 읽어 정책과 같을 때만 계속한다. 기본 dtype과 autocast는 고치지 않고 거부한다. 켜는 것은 프로세스 전체 설정이므로, 되돌리는 일은
    부른 쪽(시험)이 한다."""
    _check_device(device)
    import torch
    import torch.utils.deterministic as deterministic
    if str(torch.get_default_dtype()) != REQUIRED['default_dtype']:
        _refuse(f"default_dtype={str(torch.get_default_dtype())!r} (the policy requires {REQUIRED['default_dtype']!r}); the default dtype is checked, not changed, and nothing was set")
    if torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu'):
        _refuse('autocast_enabled=True (the policy requires False); an ambient autocast context is refused, not left, and nothing was set')
    if not hasattr(deterministic, 'fill_uninitialized_memory'):
        _refuse('fill_uninitialized_memory=None (the policy requires False); this torch has no such setting')
    _hold_cublas_workspace(torch)
    torch.set_float32_matmul_precision(REQUIRED['float32_matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32 = REQUIRED['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = REQUIRED['cudnn_allow_tf32']
    torch.backends.cudnn.deterministic = REQUIRED['cudnn_deterministic']
    torch.backends.cudnn.benchmark = REQUIRED['cudnn_benchmark']
    torch.use_deterministic_algorithms(REQUIRED['deterministic_algorithms'])
    deterministic.fill_uninitialized_memory = REQUIRED['fill_uninitialized_memory']
    measured = measure(device)
    wrong = differences(measured)
    if wrong:
        _refuse('the state of the process is not the policy after it was set: ' + '; '.join(wrong))
    return {**requested_record(), 'measured': measured, 'measured_sha256': run_io.canonical_sha256(measured)}
