"""Acceptance test of the run kernel against the T002 stack (S4b-prep K8; decision D11, the acceptance condition).

Placement: this file is ``tests/test_run_kernel_equivalence.py`` (the path of the plan), outside the tree that ``tools/check_project_boundaries.py`` walks (``c_ept/``). It reads the frozen
stack's private ``run_s4a._train_stage`` on purpose; in ``c_ept/tests/`` and in STRICT_FILES that read needs two ``frozen_reference`` entries and so a higher ceiling (3 -> 5), which the
author has not approved.

D11: the kernel that runs the T002 profile reproduces the checkpoint and the losses of the T002 stack bit for bit, on a small synthetic store; if it does not, stop and report (fallback:
D11 option (c)). This file is that test. It uses no data and no GPU: the toy EPT world of ``ept_toys`` (eight matches), one hand-written ``B0_SCALES.json``, short CPU trainings.

What it proves, and what it does not (K8 review m1). Both sides run the CURRENT ``c_ept/ept/train.py`` (D11 keeps the training loop there): the test shows that the kernel wrapper (profile, data
preparation, stage bodies, ledger records) around the shared trainer is equivalent to the T002 wrapper (``run_training`` and ``run_s4a._train_stage``). That today's trainer at the old defaults is the
trainer of the T002 commit rests on the default-identity tests of the M tasks (Global Constraint 1), not on this file. "Reproduces the T002 stack" means the T002 wrapper on today's trainer.

Symbols
-------
    P_0      the T002 profile: ``{"kind": "train"}`` with every field at its old default (S encoder, 25% of the matches, seed 20261022, lr 0.001, weight decay 0.01, warmup 10, patience 2,
             every Track M field at the value of the earlier runs); only ``epochs`` is shortened (3 instead of 30), because the world has eight matches
    D, s     the toy store and the training seed 20261022 (the batch seed, the membership seed and the seed of ``deterministic_cpu``)
    O(D, s)  the T002 stack: ``run_training`` calls ``run_s4a._train_stage`` with ``data.fit_stats``, ``data.proxy_matches`` and ``train.deterministic_cpu(threads=8, seed)`` prepared as in
             ``run_training._execute_reserved`` (lines 137, 198-204); the call is the same function with the device ``'cpu'`` in place of ``'cuda'``
    K(D, s)  the kernel: ``runners.write_stats`` and ``runners.train`` (the bodies of the two supervised stages) for P_0, ``RunnerConstants(device='cpu', threads=8)``
    A_x      an artefact of a run: ``best.pt`` and ``last.pt`` of every epoch, the epoch-metrics files, ``head.json`` and the two call files of the call ledger, ``STATS.json``
    L_x      the loss of every optimizer step (the value that ``training_step.optimize_batch`` returns), recorded by a spy

Equations
---------
    (EQ-1)  bytes(best.pt_K) == bytes(best.pt_O),  bytes(epoch_metrics.json_K) == bytes(epoch_metrics.json_O),  bytes(epoch_metrics.csv_K) == ...,  bytes(STATS_K) == bytes(STATS_O)
            for every epoch (the best weights, the per-epoch losses ``fit_batch_mean_losses`` and ``stop_losses`` and the learning rates, the statistics)
    (EQ-2)  L_K == L_O  (the loss of every step, as float32 values written as Python floats: a list compared with ``==``)
    (EQ-3)  last.pt_K and last.pt_O hold the same tensors bit for bit (model, optimizer, random state, ``loop``), except the wall-clock fields of ``loop`` (``LOOP_CLOCKS``)
    (EQ-4)  head.json_K and head.json_O are equal for every path except the telemetry of ``VOLATILE_HEAD``; the identity (settings, seeds, device, threads, the hash of the
            statistics and of the batches) is part of what is compared, and the hash of each last.pt in the head is the hash of the file that sits next to it
    (EQ-5)  the call files are equal except the clocks, the memory and the hashes of the heads (``VOLATILE_CALL``); ``identity_sha256`` is equal

Why some fields are not compared with ``==``. A run writes how long it took (the seconds of the epoch, the step, the forward and backward pass, the STOP evaluation) and how much memory
it used, and the hash of a ``last.pt`` or a ``head.json`` that holds them. Two runs of the same code do not agree on those numbers; they are not part of what the run computed. The
lists ``LOOP_CLOCKS`` and ``VOLATILE_*`` name each of them, the test checks that each exists on both sides (an exclusion cannot hide a missing key) and that the lists have the same length.

The numeric policy. The kernel applies the D10 P6 policy (``c_ept.run.numeric``: highest float32 matmul precision, TF32 off, cuDNN deterministic, deterministic algorithms,
``fill_uninitialized_memory`` off) before it builds the model; the T002 stack applied none of it (the torch defaults and ``ept.train``'s own deterministic algorithms). On the CPU none of
these settings changes a value (K6c proved it for the settings that matter on the CPU; ``fill_uninitialized_memory`` changes only the value of memory that a valid program never reads).
The test puts the process in the stock torch state before the T002 stack runs, asserts that this state is not the policy, asserts that the kernel recorded the policy as held, and
runs the two stacks in both orders.

해석
----
같은 장난감 자료, 같은 시드, 같은 T002 설정으로 옛 T002 스택과 새 핵심을 CPU에서 돌려, 최선 가중치, 에폭마다의 지표와 학습률, 통계, 단계마다의 손실, 마지막 체크포인트의 모든
텐서가 바이트까지 같음을 ``==``로 확인한다. 같지 않으면 시험을 느슨하게 만들지 않고 원인을 찾아 보고한다(D11의 (c)가 대안). 비교에서 빼는 것은 걸린 시간과 메모리처럼 같은 코드를
두 번 돌려도 다르게 나오는 측정값뿐이고, 빼는 이름은 모두 아래 목록에 있으며 양쪽에 실제로 있는지 확인한다. 핵심은 D10 P6 숫자 정책을 켜지만 옛 스택은 켜지 않았다. CPU에서는 값이
바뀌지 않으므로(K6c 증명) 비트 동일이 기대되고, 그래서 옛 스택은 정책이 아닌 기본 상태에서 돌리고 핵심은 정책을 켠 상태를 기록으로 확인한다. 순서를 바꿔 돌리는 것은 한 프로세스
안에서 먼저 돈 쪽의 전역 상태가 뒤의 쪽에 새어 들어가지 않았음을 확인하기 위해서다.
"""
import dataclasses
import gc
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from c_ept.ept import data, encoder
from c_ept.ept import train as ept_train
from c_ept.run import io as run_io
from c_ept.run import numeric as run_numeric
from c_ept.run import profile as run_profile
from c_ept.run import runners
from c_ept.tests.ept_toys import build_ept_world, toy_static
from c_ept.tests.test_run_numeric import read_flags, write_flags
from c_ept.tests.test_run_runners import SCALES, quiet_probe, write_scales
from c_ept.workflows import run_s4a                       # the frozen T002 stack, compared against on purpose (D11 acceptance condition); its private _train_stage is read as an attribute

SEED = 20261022
EPOCHS = 3
T002_ENCODER = encoder.EncoderConfig(d=128, layers=4, heads=4)   # ``run_s4a._model_config('S')``: the S encoder that T002 trained (its identity is part of what the heads compare)
THREADS = 8                                                # the T002 thread count (``run_training`` calls ``deterministic_cpu(threads=8, ...)``)
TERRAIN_SOURCE = {'terrain_files_sha256': {'TERRAIN.json': '6' * 64, 'terrain.npz': '7' * 64, 'geodesic.npy': '8' * 64}}
STOCK = {'float32_matmul_precision': 'highest', 'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': True, 'cudnn_deterministic': False, 'cudnn_benchmark': False,
         'deterministic_algorithms': False, 'fill_uninitialized_memory': True}                 # the torch defaults: the state in which T002 started (``train`` sets its own deterministic algorithms)
LOOP_CLOCKS = ('epoch_seconds', 'load_seconds', 'step_seconds', 'stop_seconds', *ept_train.COMPONENT_FIELDS)       # last.pt, ``loop``: the seconds of every epoch and of its parts
VOLATILE_HEAD = (('seconds',), ('memory', 'peak_bytes'), ('memory', 'peak_rss_bytes'), ('memory', 'peak_upper_bytes'), ('memory', 'samples'),
                 ('files', 'last.pt', 'sha256'),                                                # the hash of a last.pt that holds the clocks above
                 *(('timing', name) for name in ('load_seconds', 'step_seconds', 'stop_seconds', *ept_train.COMPONENT_FIELDS)))
VOLATILE_CALL = {'calls/call_0001.json': (('seconds',), ('peak_bytes',), ('peak_rss_bytes',), ('peak_upper_bytes',), ('head_sha256_end',), ('pending_sha256',)),
                 'calls/call_0001.pending.json': tuple(('start_accounting', name) for name in ('seconds', 'peak_bytes', 'peak_rss_bytes', 'peak_upper_bytes'))}   # clocks, memory, hashes of the heads
CALL_FILES = tuple(VOLATILE_CALL)


# ---- helpers: the comparison of one artefact -----------------------------------------------------------------------------------------------------
def flatten(document, prefix=()) -> dict:
    """``{path tuple: leaf}`` of a JSON document: a dict is walked, a list or a scalar is a leaf."""
    if isinstance(document, dict):
        flat = {}
        for key, value in document.items():
            flat.update(flatten(value, prefix + (key,)))
        return flat
    return {prefix: document}


def assert_same_json(old: dict, new: dict, volatile, what: str) -> None:
    """The documents ``old`` and ``new`` have the same paths and the same leaf at every path except the ``volatile`` ones, which must exist on both sides (a list of clock readings has
    the same length on both sides). ``==`` on every leaf: a float is equal or the test fails."""
    a, b = flatten(old), flatten(new)
    assert a.keys() == b.keys(), (what, sorted(map(str, a.keys() ^ b.keys())))
    for path in volatile:
        assert path in a and path in b, (what, 'the volatile field is not there', path)
        if isinstance(a[path], list):
            assert len(a[path]) == len(b[path]), (what, path)
    different = sorted(str(path) for path in a if path not in volatile and a[path] != b[path])
    assert not different, (what, 'differ at', different)


def same_value(x, y, path='') -> None:
    """The same tensors, arrays and values, bit for bit (a tensor by dtype, shape and bytes), in the same structure."""
    if isinstance(x, dict):
        assert isinstance(y, dict) and x.keys() == y.keys(), (path, 'keys')
        for key in x:
            same_value(x[key], y[key], f'{path}/{key}')
    elif isinstance(x, (list, tuple)):
        assert type(x) is type(y) and len(x) == len(y), (path, 'sequence')
        for index, (u, v) in enumerate(zip(x, y)):
            same_value(u, v, f'{path}[{index}]')
    elif torch.is_tensor(x):
        assert torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape, (path, 'tensor')
        assert x.detach().cpu().contiguous().numpy().tobytes() == y.detach().cpu().contiguous().numpy().tobytes(), (path, 'tensor bytes')
    elif isinstance(x, np.ndarray):
        assert isinstance(y, np.ndarray) and x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), (path, 'array')
    else:
        assert type(x) is type(y) and x == y, (path, x, y)


def load_checkpoint(path: Path) -> dict:
    return torch.load(io.BytesIO(path.read_bytes()), weights_only=True, map_location='cpu')       # as ``ept.train`` reads its own last.pt


def without(loop: dict, names) -> dict:
    return {key: value for key, value in loop.items() if key not in names}


# ---- the two stacks --------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def world(tmp_path_factory):
    """The EPT toy world (eight matches; 7 events and 5 edges per view)."""
    folder = tmp_path_factory.mktemp('equivalence_world')
    built = build_ept_world(folder, n_events=7, n_edges=5, consistent_first_kill=True)
    yield built
    del built
    gc.collect()                                                                                    # Windows keeps a folder with open memory maps


@pytest.fixture(autouse=True)
def restore_torch_state(monkeypatch):
    """Both stacks change process-wide settings (threads, deterministic algorithms, the global random streams, and the kernel the numeric policy); give them back after each test."""
    threads, flags = torch.get_num_threads(), read_flags()
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    yield
    torch.set_num_threads(threads)
    write_flags(flags)


def t002_profile(tmp_path):
    """The T002 profile: ``{"kind": "train"}`` with the epochs shortened (the file is written with LF line ends; the lock is the hash of its bytes)."""
    path = tmp_path / 'T002.json'
    path.write_bytes((json.dumps({'kind': 'train', 'epochs': EPOCHS}, indent=2) + '\n').encode('utf-8'))
    return run_profile.load(path)


class StepSpy:
    """The loss of every optimizer step, as the trainer saw it: ``training_step.optimize_batch`` returns it, the spy passes it on unchanged and writes ``float(loss)`` into
    ``current``. One spy serves both runs of a test (a second spy would wrap the first); ``begin`` starts a new list."""

    def __init__(self, monkeypatch):
        self.current, real = [], ept_train.training_step.optimize_batch

        def spy(*args, **kwargs):
            total = real(*args, **kwargs)
            self.current.append(float(total.detach()))
            return total
        monkeypatch.setattr(ept_train.training_step, 'optimize_batch', spy)

    def begin(self) -> list:
        self.current = []
        return self.current


def run_t002_stack(world, profile, run_dir: Path) -> Path:
    """The T002 stack on the CPU: the preparation of ``run_training._execute_reserved`` and its call of ``run_s4a._train_stage`` (the device ``'cpu'``). Returns the checkpoint folder.

    The process is put in the stock torch state first (T002 did not apply the numeric policy), and the arguments are those of the T002 operator config (``batch_size`` 32, ``lr`` 0.001,
    ``weight_decay`` 0.01, ``warmup_steps`` 10, ``share`` 0.25, ``seed`` 20261022; ``epochs`` shortened)."""
    write_flags(STOCK)
    assert run_numeric.differences(run_numeric.measure('cpu')), 'the stock state of the process must not be the numeric policy'
    args = SimpleNamespace(batch_size=runners.BATCH_SIZE, seed=profile['seed'], lr=profile['lr'], weight_decay=profile['weight_decay'], warmup_steps=profile['warmup'],
                           epochs=profile['epochs'], share=profile['fraction'])
    bundle = SimpleNamespace(s1=world.s1, s2=world.s2, labels=world.labels, static=toy_static(), meta={'terrain_files_sha256': TERRAIN_SOURCE['terrain_files_sha256']})
    fitted = data.fit_stats(bundle.s1, bundle.s2)                                         # run_training.py:198: the whole FIT
    run_dir.mkdir(parents=True)
    fitted.save(run_dir / 'stats' / 'STATS.json')                                         # run_training.py:199-200
    stats = data.Stats.load(run_dir / 'stats' / 'STATS.json')                             # run_training.py:205: the statistics are read back from the file, as the kernel reads them (K8 review m3)
    proxy = data.proxy_matches(bundle.s1, args.share, args.seed)                         # run_training.py:204
    ept_train.deterministic_cpu(threads=THREADS, seed=args.seed)                          # run_training.py:137
    run_s4a._train_stage(run_dir, args, bundle, stats, dict(SCALES), proxy, 'S', 'cpu', lambda: None)
    return run_dir / 'train' / 'S_cpu'


def run_kernel(world, profile, run_dir: Path, tmp_path: Path) -> tuple:
    """The kernel on the CPU: the statistics stage and the training stage of the T002 profile, as the supervisor's children call them. Returns ``(checkpoint folder, record)``."""
    inputs = runners.Inputs(s1=world.s1, s2=world.s2, labels=world.labels, static=toy_static(), static_source=TERRAIN_SOURCE, scales=dict(SCALES),
                            scales_path=write_scales(tmp_path))
    constants = runners.RunnerConstants(device='cpu', threads=THREADS)
    runners.write_stats(profile, run_dir, inputs=inputs, constants=constants)
    record = runners.train(profile, run_dir, inputs=inputs, constants=constants, probe=quiet_probe)
    return run_dir / runners.CHECKPOINT_DIR, record


# ---- the acceptance test -----------------------------------------------------------------------------------------------------------------------
def test_the_profile_under_test_is_the_t002_profile(tmp_path):
    """The profile of the equivalence test sets nothing but the epochs: every other field has its old default and the kernel builds the ``TrainConfig`` of the T002 stack from it."""
    profile = t002_profile(tmp_path)
    assert profile.kind == 'train' and profile['seed'] == SEED and profile['fraction'] == 0.25 and profile['size'] == 'S'
    assert {name: profile[name] for name in runners.IDENTITY_DEFAULTS} == dict(runners.IDENTITY_DEFAULTS)           # every Track M field at the value of the earlier runs
    assert profile['field_channels'] == 8 and profile.membership_seed() == SEED
    assert runners.train_config(profile) == ept_train.TrainConfig(T002_ENCODER, 0.001, 0.01, 10)                  # what ``run_s4a._train_stage`` builds (patience 2: the dataclass default)
    assert runners.train_config(profile).patience == 2 and (runners.BATCH_SIZE, ept_train.THREADS) == (32, THREADS)


@pytest.mark.parametrize('order', ['t002_first', 'kernel_first'])
def test_kernel_t002_profile_reproduces_the_t002_stack(world, tmp_path, monkeypatch, order):
    profile = t002_profile(tmp_path)
    spy, steps, results = StepSpy(monkeypatch), {}, {}
    runs = {'t002': lambda: run_t002_stack(world, profile, tmp_path / 'old'), 'kernel': lambda: run_kernel(world, profile, tmp_path / 'new', tmp_path)}
    for name in (('t002', 'kernel') if order == 't002_first' else ('kernel', 't002')):
        steps[name] = spy.begin()                                                     # one list of step losses for each run
        results[name] = runs[name]()
    old, (new, record) = results['t002'], results['kernel']

    # the numeric policy: the kernel held it, and says so; the T002 stack ran without it (asserted in ``run_t002_stack``)
    numeric = record['numeric']
    assert numeric['requested'] == dict(run_numeric.REQUIRED) and run_numeric.differences(numeric['measured']) == [] and numeric['measured']['device'] == 'cpu'
    assert numeric['measured']['fill_uninitialized_memory'] is False and numeric['measured']['deterministic_algorithms'] is True
    assert record['effective_threads'] == THREADS and record['runner']['threads'] == THREADS

    # (EQ-2) the loss of every step
    assert len(steps['kernel']) == len(steps['t002']) > 0
    assert steps['kernel'] == steps['t002']

    # the statistics (the same bytes), and the same epochs
    assert (tmp_path / 'new' / runners.STATS_FILE).read_bytes() == (tmp_path / 'old' / 'stats' / 'STATS.json').read_bytes()
    old_epochs = sorted(p.name for p in old.glob('epoch_[0-9][0-9][0-9][0-9]'))
    assert old_epochs and old_epochs == sorted(p.name for p in new.glob('epoch_[0-9][0-9][0-9][0-9]'))
    assert sorted(p.relative_to(old).as_posix() for p in old.rglob('*') if p.is_file()) == sorted(p.relative_to(new).as_posix() for p in new.rglob('*') if p.is_file())

    for epoch in old_epochs:
        # (EQ-1) the best weights, the per-epoch losses and the learning rates: the same bytes
        for name in ('best.pt', 'epoch_metrics.json', 'epoch_metrics.csv'):
            if (old / epoch / name).exists() or (new / epoch / name).exists():
                assert (old / epoch / name).read_bytes() == (new / epoch / name).read_bytes(), (epoch, name)
        # (EQ-3) the last checkpoint: every tensor, the optimizer, the random state and the loop (without the clocks)
        last_old, last_new = load_checkpoint(old / epoch / 'last.pt'), load_checkpoint(new / epoch / 'last.pt')
        for name in LOOP_CLOCKS:                                                      # an excluded clock exists on both sides, one reading for each epoch run
            assert name in last_old['loop'] and name in last_new['loop'] and len(last_old['loop'][name]) == len(last_new['loop'][name]), (epoch, name)
        same_value({**last_old, 'loop': without(last_old['loop'], LOOP_CLOCKS)}, {**last_new, 'loop': without(last_new['loop'], LOOP_CLOCKS)}, epoch)
        # (EQ-4) the head: every field but the telemetry; the hash of each last.pt in the head is the hash of the file next to it
        head_old, head_new = json.loads((old / epoch / 'head.json').read_text()), json.loads((new / epoch / 'head.json').read_text())
        assert_same_json(head_old, head_new, VOLATILE_HEAD, f'{epoch}/head.json')
        for folder, head in ((old, head_old), (new, head_new)):
            assert head['files']['last.pt']['sha256'] == hashlib.sha256((folder / epoch / 'last.pt').read_bytes()).hexdigest()
            assert head['files']['last.pt']['bytes'] == (folder / epoch / 'last.pt').stat().st_size

    # (EQ-5) the call ledger
    for name in CALL_FILES:
        assert_same_json(json.loads((old / name).read_text()), json.loads((new / name).read_text()), VOLATILE_CALL[name], name)
    assert json.loads((old / CALL_FILES[0]).read_text())['identity_sha256'] == json.loads((new / CALL_FILES[0]).read_text())['identity_sha256']

    # the kernel's own record agrees with the artefacts that the T002 stack made
    final = json.loads((old / old_epochs[-1] / 'head.json').read_text())
    assert record['termination']['epochs_run'] == final['epoch'] == len(old_epochs) and record['termination']['best_epoch'] + 1 == int(final['best']['folder'][-4:])
    assert run_io.read_json(tmp_path / 'new' / runners.TRAIN_RECORD_FILE) == record


def test_the_comparison_sees_a_difference_of_one_bit(world, tmp_path):
    """The comparison is not vacuous: a run of the kernel under another seed does not reproduce the T002 stack, and the helpers name the artefact that differs."""
    profile = t002_profile(tmp_path)
    old = run_t002_stack(world, profile, tmp_path / 'old')
    other = run_profile.load(_write_profile(tmp_path, {'kind': 'train', 'epochs': EPOCHS, 'seed': SEED + 1}))
    new, _ = run_kernel(world, other, tmp_path / 'new', tmp_path)
    assert (old / 'epoch_0001' / 'best.pt').read_bytes() != (new / 'epoch_0001' / 'best.pt').read_bytes()
    with pytest.raises(AssertionError):
        same_value(load_checkpoint(old / 'epoch_0001' / 'last.pt')['model'], load_checkpoint(new / 'epoch_0001' / 'last.pt')['model'])
    flipped = {'model': {'w': torch.zeros(3)}}
    other_bit = {'model': {'w': torch.tensor([0.0, 0.0, torch.finfo(torch.float32).tiny])}}
    with pytest.raises(AssertionError):
        same_value(flipped, other_bit)
    assert_same_json({'a': 1, 'seconds': [1.0]}, {'a': 1, 'seconds': [2.0]}, (('seconds',),), 'x')
    with pytest.raises(AssertionError):
        assert_same_json({'a': 1.0, 'seconds': [1.0]}, {'a': 1.0000000000000002, 'seconds': [2.0]}, (('seconds',),), 'x')
    with pytest.raises(AssertionError):
        assert_same_json({'a': 1}, {'a': 1, 'seconds': [2.0]}, (('seconds',),), 'x')                 # a volatile field that is missing on one side is not excluded silently


def test_the_comparison_sees_a_subtle_difference_in_the_step_losses(world, tmp_path, monkeypatch):
    """K8 review m2, a finer control than another seed: the same seed (so the same membership, batches and initial weights) and ``warmup`` 11 in place of 10 changes the learning rate of the
    steps only, and the step-loss lists and the weights must differ. The comparison of the step losses (EQ-2) is therefore sensitive to a change of the optimizer schedule, not only to
    another data draw."""
    profile = t002_profile(tmp_path)
    other = run_profile.load(_write_profile(tmp_path, {'kind': 'train', 'epochs': EPOCHS, 'warmup': 11}))
    assert other['seed'] == profile['seed'] == SEED and other['warmup'] == profile['warmup'] + 1
    spy, steps = StepSpy(monkeypatch), {}
    steps['t002'] = spy.begin()
    old = run_t002_stack(world, profile, tmp_path / 'old')
    steps['kernel'] = spy.begin()
    new, _ = run_kernel(world, other, tmp_path / 'new', tmp_path)
    assert len(steps['kernel']) == len(steps['t002']) > 0
    assert steps['kernel'] != steps['t002']                                         # the lists of the step losses differ (the first loss is computed before any update; from the second step on the warm-up slope differs)
    assert (old / 'epoch_0001' / 'best.pt').read_bytes() != (new / 'epoch_0001' / 'best.pt').read_bytes()
    with pytest.raises(AssertionError):
        same_value(load_checkpoint(old / 'epoch_0001' / 'last.pt')['model'], load_checkpoint(new / 'epoch_0001' / 'last.pt')['model'])


def _write_profile(tmp_path, document, name='other.json') -> Path:
    path = tmp_path / name
    path.write_bytes((json.dumps(document, indent=2) + '\n').encode('utf-8'))
    return path
