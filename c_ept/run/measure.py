"""Measurement runner of the run kernel (P6): short training calls with the same setting, each in a fresh child process, compared bit for bit (S4b-prep K6c; decisions D10 P6, D11, D32).

D10 P6: "the reporting device is deterministic GPU (float32, deterministic algorithms) with a recorded numeric policy, after a short same-setting bit-identity check of two GPU calls".
This module is that check. ``start(profile_path, run_dir, ...)`` reads a hash-locked measure profile (K3: it names a train profile by relative path and SHA-256, a step budget ``steps`` and a
number of calls ``repeat`` >= 2), refuses what is wrong before anything is reserved and hands the supervisor (``run.supervisor``) a plan of stages, each a function
``module:function(run_dir, *texts)`` that a child process calls:

    stats      ``stats_stage``    the statistics of the train profile (``runners.write_stats``: the loss scales are checked against the training condition, the D7 list, the D12 population)
    call_k     ``call_stage``     call k = 1 .. repeat: one epoch of the first ``steps`` FIT batches, the STOP value on its first ``steps`` batches, the same statistics, the same seed, the same
                                  device; a fresh process each (``supervisor.run_owned``: the Job, the filtered environment, no inherited state), so that two calls cannot share a cache, a random
                                  state or a CUDA context. The record ``measure/call_k/RECORD.json`` holds the loss of every step as bits, a hash of every key of the model, the optimiser and the
                                  random state, the metrics, the checkpoint files, the numeric state and the time of every step.
    compare    ``compare_stage``  the comparison of the calls and ``MEASURE.json``: ``identical`` (true or false), the first differing key, both numeric states and the times per step; it first
                                  checks that the checkpoint files of every call are on disk with the hashes that the call recorded (``verify_files``)

The measured configuration is the train profile except for the horizon: the epoch budget is replaced by one epoch and every FIT and STOP epoch is cut to ``steps`` batches, so the trainer's cosine schedule
runs over ``total_steps = 1 x n`` steps, not over the real horizon. The epoch boundary, the resume path and the batch orders of epochs >= 1 are NOT exercised, and neither is the part of the schedule
after n steps. The check is therefore a same-setting identity check of the first n steps; the P6 profile has to choose ``steps`` so that the batches cover the variety of batch shapes (and ``steps`` >=
``warmup_steps`` if the path after the warm-up matters). The runner reads the roles
FIT and STOP only (``check_roles``; DEV and TEST are refused) and uses no GPU unless the runner constants say so. The numeric policy of ``c_ept.run.numeric`` is set and read back in every call
before any model is built.

Symbols
-------
    P, T      the measure profile and the train profile that it names (``P.train``);  n = P.steps the step budget;  r = P.repeat the number of calls (2 <= r <= 10)
    K_i       call i = 1 .. r: ``epochs = 1``; the batches of K_i are the first n of the FIT batches of epoch 0 and the first n of the STOP batches (all of them when there are fewer)
    L_i       the n losses ``float(total_t)`` of the steps t = 1 .. n of K_i, each as the 16 hexadecimal digits of its IEEE-754 binary64 big-endian bytes
    H_i       the flat table of K_i, key -> text: the numeric state (``numeric/<field>``), the loss count and each loss (``loss/<t>``), the STOP value, the learning rates, the epoch-metric files, the
              head without its timing, the model, the AdamW state, the random state and the loop state without its timing (key by key), and the best checkpoint (``best/<epoch>``)
    order     the rank of a key: numeric state, loss count, losses by step, STOP value, learning rates, epoch metrics, head, model, optimiser, random state, loop state, best checkpoint, then by name

Equations
---------
    (MEA-1)  K_i = train(T, epochs = 1, steps = n, seed = T.seed, device = d) from a fresh process, for every i
    (MEA-2)  identical  <=>  for every i >= 2:  keys(H_i) = keys(H_1)  and  H_i[k] = H_1[k] for every key k
    (MEA-3)  first difference  =  the first (call, key) in the order (call index, order(key)) with H_i[k] != H_1[k] (an absent key is None)

해석
----
측정은 "같은 설정으로 두 번 돌리면 같은 비트가 나오는가"를 묻는다. 학습 호출마다 새 프로세스를 쓰는 이유는 한 프로세스 안의 두 번째 호출이 첫 호출의 캐시, 난수 상태, CUDA 문맥에 기댈 수 있기
때문이다. 호출은 학습 프로파일의 설정에서 지평만 바꿔(한 에폭, 코사인 일정은 n걸음 위에서 돈다) 첫 에폭의 처음 n개 FIT 배치와 처음 n개 STOP 배치만 돌고, 에폭 경계·이어 하기·둘째 에폭 이후의 배치 순서는 시험하지 않으며, 걸음마다의 손실을 비트로 적는다(``training_step.optimize_batch``를
감싸는 기록기: 학습 고리 자체는 고치지 않는다). 비교는 체크포인트의 시간이 든 파일(``last.pt``의 시간 열, ``head.json``의 시간·메모리)을 바이트로 견주지 않고, 시간을 뺀 상태를 키마다 해시해서 견준다:
달라진 첫 키가 곧 첫 어긋남이다(원인에 가까운 순서: 숫자 설정, 걸음별 손실, STOP 값, 지표 파일, 머리, 모형, 최적화기, 난수). 측정은 FIT와 STOP만 읽는다. 한 호출이 중간에 끊기면 이어하지
않는다(두 번째 호출이 첫 호출과 같은 출발이어야 하므로 새 실행 폴더를 쓴다). 차이를 찾는 것도 끝난 측정이므로 종료 코드는 0이고, 판정은 ``MEASURE.json``의 ``identical``과 화면의 한 줄이다.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib
import itertools
import json
import math
import os
import re
import struct
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import torch

from c_ept.ept import checkpoints, data
from c_ept.ept import train as ept_train
from c_ept.ept import training_step
from c_ept.run import identity as run_identity
from c_ept.run import io as run_io
from c_ept.run import ledger as run_ledger
from c_ept.run import numeric as run_numeric
from c_ept.run import profile as run_profile
from c_ept.run import runners, supervisor

MEASURE_FILE = 'MEASURE.json'
CALL_FOLDER = 'measure'
STATS_STAGE, COMPARE_STAGE = 'stats', 'compare'
ROLES = ('FIT', 'STOP')                                                       # the roles that a measurement reads; DEV and TEST are refused
SCHEMA_CALL, SCHEMA_MEASURE = 'ept-run-measure-call-v1', 'ept-run-measure-v1'
BODIES = MappingProxyType({'stats': 'c_ept.run.measure:stats_stage', 'call': 'c_ept.run.measure:call_stage', 'compare': 'c_ept.run.measure:compare_stage'})
MEASURE_MODULES = ('c_ept.run.supervisor', 'c_ept.run.measure')               # the entry modules of the code identity: the supervisor and everything the stage bodies import
LOADER = runners.LOADER
LISTED_DIFFERENCES = 20                                                       # the differences that MEASURE.json lists (it counts all of them)
TIMING_KEYS = ('epoch_seconds', 'load_seconds', 'step_seconds', 'stop_seconds', *checkpoints.COMPONENT_FIELDS)        # the keys of the loop state of ``last.pt`` that are times
HEAD_VOLATILE = ('seconds', 'memory', 'timing', 'checkpoint_bytes', 'files')   # the keys of ``head.json`` that are times, memory and the hash of ``last.pt`` (its loop state holds times)
ORDER = ('numeric', 'loss_count', 'loss', 'stop_loss', 'lrs', 'epoch_metrics', 'head', 'model', 'optimizer', 'rng', 'loop', 'best')
_NUMBER_TEXT = re.compile(r'[0-9]{1,9}')
_LOADER_NAME = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*')


class MeasureRefused(run_io.IORefused):
    """The measurement cannot be made as asked: a profile that is not a measure profile, a role that a measurement may not read, a call that was cut, a state that cannot be hashed. A refusal
    (exit code 2); the message names the rule, never a path or a key."""


def call_name(number: int) -> str:
    return f'call_{number}'


def call_paths(number: int) -> dict:
    """The paths of call ``number`` relative to the run folder: ``folder``, ``ckpt`` (the checkpoint folder of the trainer), ``record`` (``RECORD.json``) and ``e6`` (the per-epoch checks)."""
    folder = f'{CALL_FOLDER}/{call_name(number)}'
    return {'folder': folder, 'ckpt': f'{folder}/ckpt', 'record': f'{folder}/RECORD.json', 'e6': f'{folder}/E6.json'}


# ---- the batches of a short call and the roles that it may read -----------------------------------------------------------------------------------
def _close(iterator) -> None:
    close = getattr(iterator, 'close', None)
    if close is not None:
        close()


def _first(iterator, count: int):
    """The first ``count`` items of ``iterator``; the iterator is closed afterwards, whether or not it ran dry (a producer behind it is stopped)."""
    try:
        yield from itertools.islice(iterator, count)
    finally:
        _close(iterator)


class StepLimited:
    """The first ``limit`` batches of every epoch of ``batches`` (FIT or STOP batches of ``ept.data.Batches``, or anything with the same attributes): ``len`` is ``min(limit, len(batches))``
    and ``epoch(e, start)`` yields the batches ``start ..`` of the epoch ``e`` up to that length and builds no batch beyond it. Every other attribute (``rows``, ``stats``, ``training``,
    ``batch_size``, ``seed``, ``swap_mode``) is that of ``batches``. ``MeasureRefused`` for a ``limit`` that is not a whole number of at least 1.

    해석: 짧은 호출은 한 에폭의 처음 몇 배치만 돈다. 배치는 시드·에폭·번호만의 함수이므로 처음 n개는 실제 학습의 첫 에폭의 처음 n개와 같다. 그보다 뒤의 배치는 만들지 않는다."""

    def __init__(self, batches, limit):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise MeasureRefused('the steps of a measurement must be a whole number of at least 1')
        self._batches, self._limit = batches, limit

    def __getattr__(self, name):                                                  # only called when the normal lookup fails
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._batches, name)

    def __len__(self) -> int:
        return min(self._limit, len(self._batches))

    def epoch(self, epoch, start=0):
        count = len(self)
        if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start <= count:
            raise ValueError(f'start must be an int in 0 .. {count}, not {start!r}')
        return _first(self._batches.epoch(epoch, start), count - start)


def limit_steps(train_data, steps: int):
    """``train_data`` (``ept.train.TrainData``) with its FIT and STOP batches cut to the first ``steps`` batches of every epoch (``StepLimited``); the scales and the static channels are as they were."""
    return dataclasses.replace(train_data, fit=StepLimited(train_data.fit, steps), stop=StepLimited(train_data.stop, steps))


def check_roles(train_data) -> None:
    """``MeasureRefused`` unless the FIT batches hold the rows of the role FIT and the STOP batches those of the role STOP: a measurement reads FIT and STOP only, and DEV and TEST are refused.

    해석: 측정은 학습과 같은 FIT 배치와 조기 종료용 STOP 배치만 읽는다. DEV나 TEST 행이 들어 있으면 일을 시작하기 전에 거부한다."""
    for name, role in (('fit', 'FIT'), ('stop', 'STOP')):
        got = getattr(getattr(getattr(train_data, name, None), 'rows', None), 'role', None)
        if got != role:
            raise MeasureRefused(f'a measurement reads the roles FIT and STOP only: data.{name} holds the rows of {got!r} (DEV and TEST are refused)')


# ---- the record of a step ---------------------------------------------------------------------------------------------------------------------------
def bits_of(value: float) -> str:
    """The 16 hexadecimal digits of the IEEE-754 binary64 big-endian bytes of ``value`` (the loss of a step as the trainer reads it, ``float(total)``)."""
    return struct.pack('>d', float(value)).hex()


class StepRecorder:
    """The loss of every optimiser step as bits (``bits``) and the host time of every step in seconds (``seconds``), made by ``recording_steps``. The time of a step is the clock before the step
    to the clock after the loss was read back (the read is the synchronisation that the trainer makes next anyway, ``float(total.detach())``); ``clock`` is read twice per step."""

    def __init__(self, clock=time.perf_counter):
        self.clock, self.bits, self.seconds = clock, [], []

    def wrap(self, step):
        def recording(*args, **kwargs):
            began = self.clock()
            total = step(*args, **kwargs)
            value = float(total.detach())
            self.seconds.append(self.clock() - began)
            self.bits.append(bits_of(value))
            return total                                                          # the value that the trainer reads is the one the step made
        return recording


@contextlib.contextmanager
def recording_steps(recorder: StepRecorder):
    """Within the block ``training_step.optimize_batch`` (what ``ept.train._run`` calls once for every step) records into ``recorder`` and returns what it returned; the function that was there is
    put back at the end, also after an error.

    해석: 학습 고리는 걸음마다의 손실을 따로 남기지 않는다(에폭 평균만). 고리를 고치지 않고 한 걸음을 하는 함수만 감싸서 손실을 비트로, 걸음의 시간을 초로 적는다."""
    real = training_step.optimize_batch
    training_step.optimize_batch = recorder.wrap(real)
    try:
        yield recorder
    finally:
        training_step.optimize_batch = real


# ---- the hash of a state, key by key ------------------------------------------------------------------------------------------------------------
_SCALARS = (bool, int, float, str, type(None))


def _leaf(value) -> str:
    """The SHA-256 of one value of a state: a tensor by its dtype, shape and bytes, an array likewise, a number, text or None by its type and text (a float by its bits)."""
    digest = hashlib.sha256()
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().contiguous().reshape(-1)
        digest.update(f'tensor|{value.dtype}|{tuple(value.shape)}|'.encode())
        digest.update(flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b'')
    elif hasattr(value, 'dtype') and hasattr(value, 'shape') and hasattr(value, 'tobytes'):             # a numpy array or scalar
        digest.update(f'array|{value.dtype}|{tuple(value.shape)}|'.encode() + value.tobytes())
    elif isinstance(value, float):
        digest.update(b'float|' + struct.pack('>d', value))
    elif isinstance(value, _SCALARS):
        digest.update(f'{type(value).__name__}|{value!r}'.encode())
    else:
        raise MeasureRefused(f'state: a value of type {type(value).__name__} cannot be hashed')
    return digest.hexdigest()


def _path(prefix: str, key) -> str:
    return f'{prefix}/{key:05d}' if isinstance(key, int) and not isinstance(key, bool) else f'{prefix}/{key}'


def state_hashes(state, prefix: str) -> dict:
    """``{path: sha256}`` of every leaf of ``state`` (tensors, arrays, numbers, texts, None, and mappings and lists of them), the path being ``prefix`` and the keys joined by ``/`` (an integer key or a
    list position is written with five digits, so that the paths sort as the numbers do). A list or tuple of scalars only is one leaf (the 625 numbers of the Python random state are one key).
    ``MeasureRefused`` for a value that cannot be hashed.

    해석: 상태를 키마다 해시하면 어긋난 첫 키가 곧 어느 매개변수(또는 최적화기·난수 항목)부터 달라졌는지 알려 준다. 한 비트만 달라도 그 키 하나만 달라진다."""
    out: dict = {}

    def walk(node, path):
        if isinstance(node, Mapping):
            for key in sorted(node, key=lambda item: (type(item).__name__, item)):
                walk(node[key], _path(path, key))
        elif isinstance(node, (list, tuple)) and not all(isinstance(item, _SCALARS) for item in node):
            for index, item in enumerate(node):
                walk(item, _path(path, index))
        elif isinstance(node, (list, tuple)):
            out[path] = hashlib.sha256(f'{type(node).__name__}|{list(node)!r}'.encode()).hexdigest()
        else:
            out[path] = _leaf(node)
    walk(state, prefix)
    return out


def _canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()


def _epoch_hashes(out_dir: Path) -> dict:
    """The flat table of hashes of every epoch folder of a checkpoint folder: the two epoch-metric files, the head without its times, the best state, and the model, optimiser, random and loop
    state of ``last.pt`` without the times of the loop, key by key. ``last.pt`` is read with ``weights_only=True``, as the trainer reads it on a resume."""
    table = {}
    for folder in sorted(path for path in out_dir.glob('epoch_[0-9][0-9][0-9][0-9]') if path.is_dir()):
        name = folder.name
        for label, file in (('json', 'epoch_metrics.json'), ('csv', 'epoch_metrics.csv')):
            table[f'epoch_metrics/{name}/{label}'] = run_io.sha_file(folder / file)
        head = json.loads((folder / 'head.json').read_bytes())
        table[f'head/{name}'] = _canonical({key: value for key, value in head.items() if key not in HEAD_VOLATILE})
        if (folder / 'best.pt').is_file():
            table[f'best/{name}'] = run_io.sha_file(folder / 'best.pt')
        last = torch.load(folder / 'last.pt', weights_only=True, map_location='cpu')
        for label in ('model', 'optimizer', 'rng'):
            table.update(state_hashes(last[label], f'{label}/{name}'))
        table.update(state_hashes({key: value for key, value in last['loop'].items() if key not in TIMING_KEYS}, f'loop/{name}'))
    return table


# ---- one call -------------------------------------------------------------------------------------------------------------------------------------------
def _check_number(number) -> int:
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise MeasureRefused('the call number must be a whole number of at least 1')
    return number


def _summary(values) -> dict:
    """``{'steps', 'total', 'median', 'p90', 'min', 'max'}`` of a list of seconds (the 90th percentile by the nearest rank)."""
    ordered = sorted(values)
    if not ordered:
        return {'steps': 0, 'total': 0.0, 'median': 0.0, 'p90': 0.0, 'min': 0.0, 'max': 0.0}
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {'steps': len(ordered), 'total': float(sum(ordered)), 'median': float(median), 'p90': float(ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]), 'min': float(ordered[0]),
            'max': float(ordered[-1])}


def call(profile, run_dir, *, inputs: runners.Inputs, constants: runners.RunnerConstants, number: int, make_data=None, probe=None, verifier=None, clock=time.perf_counter) -> dict:
    """One short training call of the measure profile ``profile``: the train profile that it names, trained for one epoch on the first ``profile['steps']`` FIT batches (and its STOP value on the
    first ``steps`` STOP batches), from the statistics of ``stats/STATS.json`` under ``run_dir``, into ``measure/call_<number>/ckpt``. Writes and returns ``measure/call_<number>/RECORD.json``.

    Order. The kind of the profile and the call number, the CUDA check, the numeric policy (``numeric.apply``: before any row is read or model built), the call folder (it must be empty: a call
    that was cut is not resumed), the statistics (``IORefused`` when the statistics stage has not made them), the data (``make_data(train_profile, inputs, stats, constants)``, default
    ``runners.make_train_data``), the roles (``check_roles``), the cut to ``steps`` batches, the check of the loss scales against the training condition, then ``ept.train.train`` with ``epochs = 1``
    and the same seed as every call, with E6 and the identity check before the epoch (``runners.EpochCheck``; ``probe`` and ``verifier`` as there).

    The record holds what two calls must agree on, flat in ``hashes`` (the STOP value as bits, the learning rates, the epoch-metric files, the head without its times, the best checkpoint, and the
    model, optimiser, random and loop state key by key), the loss of every step as bits (``losses``), what may differ (``timing``: the time of every step and the epoch's parts; ``files``: the hash
    of every checkpoint file, ``last.pt`` and ``head.json`` hold times) and the numeric state that was read back (``numeric``) and the process (``process``).

    수식: (MEA-1) 한 호출 = train(T, epochs=1, 처음 n개의 FIT 배치, 처음 n개의 STOP 배치, seed=T.seed). 기록의 hashes는 (MEA-2)의 H_i.
    해석: 호출은 학습 프로파일의 설정 그대로 한 에폭의 처음 n걸음만 돈다. 같은 시드와 같은 통계에서 새 프로세스로 부르면 두 호출의 hashes가 같아야 한다."""
    if getattr(profile, 'kind', None) != 'measure':
        raise MeasureRefused('the measurement takes a measure profile')
    _check_number(number)
    run_dir = Path(run_dir)
    runners.check_cuda(constants)
    state = run_numeric.apply(constants.device)                                   # D10 P6 (K6c): held before any row is read or any model is built
    paths = call_paths(number)
    folder = run_dir / paths['folder']
    if folder.exists() and any(folder.iterdir()):
        raise MeasureRefused('the folder of this call is not empty: a call that was cut is not resumed (two calls must start alike), so start a new run folder')
    stats_path = run_dir / runners.STATS_FILE
    if not stats_path.is_file():
        raise run_io.IORefused('the statistics file of the statistics stage is missing')
    train_profile, steps = profile.train, profile['steps']
    config = runners.train_config(train_profile)
    stats = data.Stats.load(stats_path)
    train_data = (make_data if make_data is not None else runners.make_train_data)(train_profile, inputs, stats, constants)
    check_roles(train_data)
    train_data = limit_steps(train_data, steps)
    rows = train_data.fit.rows.head_rows
    if inputs.scales_path is None:
        raise runners.ScalesConditionRefused('the inputs name no scales file, so the condition of the loss scales cannot be checked')
    scales = {'file_sha256': run_io.sha_file(inputs.scales_path), 'condition': runners.check_scales_condition(train_profile, inputs.scales_path, rows.k_reps_sha256())}
    watcher = runners.EpochCheck(run_dir / paths['e6'], probe, verifier)
    out_dir = run_dir / paths['ckpt']
    ept_train.deterministic_cpu(threads=constants.threads, seed=train_profile['seed'])
    recorder = StepRecorder(clock)
    with recording_steps(recorder):
        result = ept_train.train(config, train_data, out_dir, device=constants.device, epochs=1, check=watcher, seed=train_profile['seed'])
    latest = checkpoints.latest_metadata(out_dir)
    if latest is None:
        raise RuntimeError('the call returned without a committed checkpoint')
    final, head, _ = latest
    if head.get('done') is not True or head.get('epoch') != 1:
        raise RuntimeError('the checkpoint of the call is not the end of one epoch')
    if len(recorder.bits) != len(train_data.fit):
        raise RuntimeError('the steps that were recorded are not the steps of the epoch')
    hashes = {f'stop_loss/epoch_{index:04d}': bits_of(value) for index, value in enumerate(result.stop_losses, start=1)}
    hashes['lrs'] = _canonical([bits_of(lr) for lr in result.lrs])
    hashes.update(_epoch_hashes(out_dir))
    files = {path.relative_to(run_dir).as_posix(): run_io.sha_file(path) for path in sorted(out_dir.rglob('*')) if path.is_file()}
    components = {key: float(result.timing_components['epochs'][key][0]) for key in checkpoints.COMPONENT_FIELDS}
    reason = 'early_stopping' if head['stopped_early'] else 'budget' if 'budget_end' in head else 'max_epochs'
    record = {'schema': SCHEMA_CALL, 'call': number, 'profile_sha256': profile.sha256, 'train_profile_sha256': train_profile.sha256, 'steps': steps, 'steps_made': len(recorder.bits),
              'epochs_run': int(result.epochs_run), 'roles_read': list(ROLES), 'batches': {'fit': len(train_data.fit), 'stop': len(train_data.stop)}, 'runner': constants.to_record(),
              'process': {'pid': os.getpid(), 'executable': sys.executable}, 'numeric': state, 'scales': scales,
              'effective_threads': int(head['threads']),                  # m5: what the trainer ran with (``numeric.measured.threads`` was read before the trainer set its own count)
              'losses': {'bits': list(recorder.bits), 'sha256': run_io.canonical_sha256(recorder.bits)}, 'stop_losses': [float(v) for v in result.stop_losses], 'hashes': hashes, 'files': files,
              'timing': {'step_seconds': [float(s) for s in recorder.seconds], 'epoch_seconds': float(result.epoch_seconds[0]), 'load_seconds': float(result.load_seconds[0]),
                         'train_step_seconds': float(result.step_seconds[0]), 'stop_seconds': float(result.stop_seconds[0]), 'components': components,
                         'training_seconds': float(result.seconds)},
              'termination': {'reason': reason, 'done': True, 'epochs_run': int(result.epochs_run), 'stopped_early': bool(head['stopped_early'])},
              'checkpoint': {'final': final.relative_to(run_dir).as_posix(), 'head_sha256': files[f'{final.relative_to(run_dir).as_posix()}/head.json']},
              'e6': {'file': paths['e6'], 'checks': len(watcher.checks)}}
    run_io.write_json_atomic(run_dir / paths['record'], record)
    return record


# ---- the comparison ---------------------------------------------------------------------------------------------------------------------------------
def order_key(key: str) -> tuple:
    """The rank of a key for the order of the differences: its group (the part before the first ``/``) in the order of ``ORDER``, an unknown group last, then the key itself."""
    group = key.split('/')[0]
    return (ORDER.index(group) if group in ORDER else len(ORDER), key)


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _check_record(record) -> None:
    ok = (isinstance(record, Mapping) and isinstance(record.get('numeric'), Mapping) and isinstance(record['numeric'].get('measured'), Mapping)
          and isinstance(record.get('losses'), Mapping) and isinstance(record['losses'].get('bits'), (list, tuple)) and isinstance(record.get('hashes'), Mapping))
    if not ok:
        raise MeasureRefused('a call record must hold its numeric state, its losses and its hashes')


def flatten(record) -> dict:
    """The table ``H`` of a call record: ``numeric/<field>`` for every field of the measured numeric state, ``loss_count`` and ``loss/<step>`` (six digits) for the losses, and the keys of
    ``hashes``; every value is a text. ``MeasureRefused`` for a record that is not a call record."""
    _check_record(record)
    flat = {f'numeric/{name}': _text(value) for name, value in record['numeric']['measured'].items()}
    bits = record['losses']['bits']
    flat['loss_count'] = str(len(bits))
    flat.update({f'loss/{index:06d}': value for index, value in enumerate(bits, start=1)})
    flat.update({key: _text(value) for key, value in record['hashes'].items()})
    return flat


def compare(records) -> dict:
    """The comparison of the call records ``records`` (two or more) with the first as the reference: ``{'identical', 'first_difference', 'differences': {'count', 'listed'}, 'keys_compared',
    'calls_compared'}``. A difference is ``{'call', 'key', 'reference', 'other'}`` (an absent key has the value None), ordered by the call and then by ``order_key``; ``listed`` holds the first
    ``LISTED_DIFFERENCES`` of them and ``count`` all of them. ``MeasureRefused`` for fewer than two records or a record that is not a call record.

    수식: (MEA-2) identical ⇔ 모든 i ≥ 2에서 H_i = H_1 (같은 키 집합, 같은 값);  (MEA-3) 첫 어긋남 = (호출 번호, 키 순서)에서 처음으로 H_i[k] ≠ H_1[k]인 곳."""
    if not isinstance(records, (list, tuple)) or len(records) < 2:
        raise MeasureRefused('at least two calls are compared')
    tables = [flatten(record) for record in records]
    reference, differences = tables[0], []
    for number, other in enumerate(tables[1:], start=2):
        for key in sorted(set(reference) | set(other), key=order_key):
            if reference.get(key) != other.get(key):
                differences.append({'call': number, 'key': key, 'reference': reference.get(key), 'other': other.get(key)})
    return {'identical': not differences, 'first_difference': differences[0] if differences else None,
            'differences': {'count': len(differences), 'listed': differences[:LISTED_DIFFERENCES]}, 'keys_compared': len(reference), 'calls_compared': len(records)}


def verify_files(run_dir, record: Mapping) -> int:
    """The number of checkpoint files of the call ``record`` that are on disk with the hash that the call recorded; ``MeasureRefused`` for a file that changed or is missing, and for a recorded
    path that is not a file of the call's own checkpoint folder.

    해석: 호출 기록의 ``hashes``는 호출이 끝날 때 잰 값이다. 체크포인트 파일이 그 뒤에 바뀌었거나 기록이 다른 곳의 파일을 가리키면, 같은 기록끼리 견주어도 같은 파일을 견준 것이 아니다. 비교 전에 확인한다."""
    run_dir, prefix = Path(run_dir), call_paths(record['call'])['ckpt'] + '/'
    files = record.get('files')
    if not isinstance(files, Mapping) or not files:
        raise MeasureRefused(f"call {record['call']}: the record names no checkpoint file")
    bad = 0
    for relative, digest in files.items():
        if not isinstance(relative, str) or not relative.startswith(prefix) or '..' in relative.split('/') or not isinstance(digest, str):
            raise MeasureRefused(f"call {record['call']}: the record names a file that is not in the checkpoint folder of the call")
        try:
            actual = run_io.sha_file(run_dir / relative)
        except OSError:
            actual = None
        bad += actual != digest
    if bad:
        raise MeasureRefused(f"call {record['call']}: {bad} of {len(files)} checkpoint file(s) changed or are missing since the call recorded their hashes")
    return len(files)


def document_of(profile, run_dir, records: list, constants: Mapping) -> dict:
    """``MEASURE.json``: the verdict of ``compare``, the profile, the calls with the hashes of their records (and the number of their checkpoint files that were found with the recorded hash,
    ``verify_files``), both (all) numeric states and whether they are equal, and the time of every step of every call with a summary. ``records`` are the call records as read from ``run_dir``."""
    result = compare(records)
    calls, numeric_states, timings = [], {}, {}
    for record in records:
        paths = call_paths(record['call'])
        calls.append({'call': record['call'], 'record': paths['record'], 'record_sha256': run_io.sha_file(Path(run_dir) / paths['record']), 'losses_sha256': record['losses']['sha256'],
                      'hashes_sha256': run_io.canonical_sha256(record['hashes']), 'numeric_measured_sha256': record['numeric']['measured_sha256'], 'steps_made': record['steps_made'],
                      'files_verified': verify_files(run_dir, record)})
        numeric_states[call_name(record['call'])] = record['numeric']['measured']
        timings[call_name(record['call'])] = {'step_seconds': list(record['timing']['step_seconds']), 'summary': _summary(record['timing']['step_seconds']),
                                              **{key: record['timing'][key] for key in ('epoch_seconds', 'load_seconds', 'train_step_seconds', 'stop_seconds', 'components', 'training_seconds')}}
    return {'schema': SCHEMA_MEASURE, **result, 'repeat': profile['repeat'],
            'profile': {'sha256': profile.sha256, 'train_profile': profile['train_profile'], 'train_sha256': profile['train_sha256'], 'steps': profile['steps'], 'repeat': profile['repeat']},
            'runner': dict(constants), 'roles_read': list(ROLES), 'calls': calls, 'numeric': numeric_states,
            'numeric_equal': len({state['measured_sha256'] for state in (record['numeric'] for record in records)}) == 1, 'timings': timings}


def headline(document: Mapping) -> str:
    """The one line that the command prints: whether the calls are identical and, if not, the call and the key of the first difference. It names no path and no match key."""
    steps = f"{document['profile']['steps']} steps"
    if document['identical']:
        return f"P6 measurement: {document['calls_compared']} calls of {steps} are identical on {document['keys_compared']} keys (numeric policy equal: {document['numeric_equal']})"
    first = document['first_difference']
    return (f"P6 measurement: NOT identical: the first difference is in call {first['call']} at {first['key']} ({document['differences']['count']} of "
            f"{document['keys_compared']} keys differ; {document['calls_compared']} calls of {steps})")


# ---- the stages of the supervisor -------------------------------------------------------------------------------------------------------------------
def _whole_text(name: str, text, low: int) -> int:
    if not isinstance(text, str) or _NUMBER_TEXT.fullmatch(text) is None or int(text) < low:
        raise run_io.IORefused(f'stage argument {name} must be a whole number of at least {low}')
    return int(text)


def _loader(spec):
    if not isinstance(spec, str) or _LOADER_NAME.fullmatch(spec) is None:
        raise run_io.IORefused('the inputs loader must be named module.path:function')
    module_name, _, function = spec.partition(':')
    try:
        return getattr(importlib.import_module(module_name), function)
    except (ImportError, AttributeError):
        raise runners.InputsRefused('the inputs loader cannot be imported') from None


def _load_measure(profile_path, profile_sha256):
    """The measure profile of a stage body, its lock checked; ``IORefused`` for another file, another kind."""
    profile = run_profile.load(profile_path)
    if not isinstance(profile_sha256, str) or profile.sha256 != profile_sha256:
        raise run_io.IORefused('the profile file is not the one the run was started with (its lock, the sha256 of the file, differs)')
    if profile.kind != 'measure':
        raise run_io.IORefused(f'the measurement takes a measure profile, not a {profile.kind} profile')
    return profile


def _open_stage(profile_path, profile_sha256, manifest_path, loader, device, threads, batch_size) -> tuple:
    """The measure profile (its lock checked), the constants and the inputs of a stage body, from the seven text arguments; ``IORefused`` for any of them that is not acceptable, before anything is
    read from the stores or written."""
    profile = _load_measure(profile_path, profile_sha256)
    constants = runners.RunnerConstants(batch_size=_whole_text('batch_size', batch_size, 1), device=device, threads=_whole_text('threads', threads, 1))
    opened = _loader(loader)(manifest_path)
    if not isinstance(opened, runners.Inputs):
        raise runners.InputsRefused('the inputs loader did not return Inputs')
    return profile, constants, opened


def stats_stage(run_dir, profile_path, profile_sha256, manifest_path, loader, device, threads, batch_size) -> None:
    """Stage body ``stats``: ``runners.write_stats`` for the train profile that the measure profile names."""
    profile, constants, opened = _open_stage(profile_path, profile_sha256, manifest_path, loader, device, threads, batch_size)
    runners.write_stats(profile.train, run_dir, inputs=opened, constants=constants)


def call_stage(run_dir, profile_path, profile_sha256, manifest_path, loader, device, threads, batch_size, number) -> None:
    """Stage body ``call_<number>``: ``call`` for the arguments of ``stage_arguments``. A refusal (the numeric gate, E6, the identity check, a role, a folder that is not empty) is exit code 2 of the
    child, every other error 3 (``supervisor.child_main``). The saved identity is read from the ledger header (read-only) and re-verified before the epoch."""
    _whole_text('call number', number, 1)
    profile, constants, opened = _open_stage(profile_path, profile_sha256, manifest_path, loader, device, threads, batch_size)
    if int(number) > profile['repeat']:
        raise MeasureRefused('the call number is above the repeat of the profile')
    saved = run_ledger.Ledger(run_dir).header.identity
    call(profile, run_dir, inputs=opened, constants=constants, number=int(number), verifier=lambda: run_identity.verify(saved))


def compare_stage(run_dir, profile_path, profile_sha256) -> None:
    """Stage body ``compare``: read the records of the ``repeat`` calls, compare them and write ``MEASURE.json``."""
    profile = _load_measure(profile_path, profile_sha256)
    records = []
    for number in range(1, profile['repeat'] + 1):
        record = run_io.read_json(Path(run_dir) / call_paths(number)['record'])
        if record.get('schema') != SCHEMA_CALL or record.get('call') != number or record.get('profile_sha256') != profile.sha256:
            raise MeasureRefused(f'the record of call {number} is not a record of this measurement')
        records.append(record)
    run_io.write_json_atomic(Path(run_dir) / MEASURE_FILE, document_of(profile, run_dir, records, records[0]['runner']))


# ---- the plan and the start ---------------------------------------------------------------------------------------------------------------------------
def _absolute(path) -> Path:
    return Path(os.path.abspath(path))


def _common_arguments(profile, profile_path, manifest_path, loader: str, constants: runners.RunnerConstants) -> tuple:
    return (str(_absolute(profile_path)), profile.sha256, str(_absolute(manifest_path)), loader, constants.device, str(constants.threads), str(constants.batch_size))


def plan(profile, profile_path, manifest_path, *, constants: runners.RunnerConstants, loader: str = LOADER, bodies=None, manifest=None) -> list:
    """The stages of a measurement for the supervisor: ``stats``, ``call_1`` .. ``call_<repeat>`` and ``compare``, each ``module:function(run_dir, *texts)``.

    The inputs of a stage are what its ledger hashes before and after it: the measure profile file, the train profile file that it names, the inputs manifest and the files that the manifest binds
    (``runners.manifest_files``) and, for a call, the two files of the statistics stage; the comparison binds the call records. The outputs are fixed files. ``bodies`` (None or a mapping with
    the keys ``stats``, ``call`` and ``compare``, ``module:function`` texts) replaces a default body (tests wrap the real ones); ``manifest`` is the manifest that the caller has read already."""
    chosen = {**BODIES, **(bodies or {})}
    profile_file, manifest_file = _absolute(profile_path), _absolute(manifest_path)
    train_file = profile_file.parent / profile['train_profile']
    bound = runners.manifest_files(manifest if manifest is not None else runners.read_manifest(manifest_file))
    common = [profile_file.as_posix(), train_file.as_posix(), manifest_file.as_posix(), *bound]
    arguments = _common_arguments(profile, profile_file, manifest_file, loader, constants)
    stages = [supervisor.Stage(STATS_STAGE, chosen['stats'], args=arguments, inputs=common, outputs=(runners.STATS_FILE, runners.STATS_RECORD_FILE))]
    for number in range(1, profile['repeat'] + 1):
        paths = call_paths(number)
        stages.append(supervisor.Stage(call_name(number), chosen['call'], args=(*arguments, str(number)), inputs=[runners.STATS_FILE, runners.STATS_RECORD_FILE, *common],
                                       outputs=(paths['record'], paths['e6'])))
    stages.append(supervisor.Stage(COMPARE_STAGE, chosen['compare'], args=(str(profile_file), profile.sha256), inputs=[call_paths(n)['record'] for n in range(1, profile['repeat'] + 1)],
                                   outputs=(MEASURE_FILE,)))
    return stages


def runner_record(profile, stages: list, manifest: Mapping, manifest_sha256: str, constants: runners.RunnerConstants, scales_condition: dict, approval=None) -> dict:
    """The runner constants and the effective values for ``Header.runner`` (written once, when the ledger is made): the batch size, the device and the threads, the measurement (the profile as it
    is with its lock and the lock of the train profile that it names, the settings of that train profile written out, the steps, the repeat, the roles that are read and the stages), the inputs
    manifest with the files it binds, the loss-scale file with the condition it was made under, the numeric policy that the run asks for and the text of the approval as given."""
    return {**constants.to_record(),
            'measurement': {'profile': profile.as_dict(), 'profile_sha256': profile.sha256, 'train_profile_sha256': profile.train.sha256, 'settings': runners.effective_settings(profile.train),
                            'steps': profile['steps'], 'repeat': profile['repeat'], 'roles_read': list(ROLES), 'stages': [stage.name for stage in stages]},
            'inputs': {'manifest_sha256': manifest_sha256, 'bound': runners.manifest_files(manifest)},
            'scales': {'file_sha256': run_io.sha_file(manifest['scales']), 'condition': scales_condition},
            'numeric': run_numeric.requested_record(), 'approval': runners.approval_record(approval)}


def _start(profile, profile_path, run_dir, *, manifest_path, spec_path, budget, constants, scrub_keys, loader, bodies, stream, out, approval, options) -> int:
    """Everything that can be refused before a run folder exists, cheaply: that ``profile`` is the file at ``profile_path`` and a measure profile, the approval text, CUDA, the manifest (read once),
    the stores (complete manifests), the terrain against the build of the S2 views (when the manifest names one) and the condition of the scales file; then the supervisor, and on exit code 0
    the headline."""
    if getattr(profile, 'kind', None) != 'measure':
        raise run_io.IORefused(f'the measurement takes a measure profile, not a {getattr(profile, "kind", None)} profile')
    if run_profile.load(profile_path).sha256 != profile.sha256:
        raise run_io.IORefused('the loaded profile is not the file at profile_path (its sha256 differs): the children load the file and check its lock')
    runners.approval_record(approval)
    runners.check_cuda(constants)
    manifest, manifest_sha256 = runners.read_manifest_with_sha(manifest_path)
    condition = runners.check_scales_condition(profile.train, manifest['scales'])
    _, s2, _ = runners.open_stores(manifest)
    if manifest['terrain'] is not None:
        runners.check_terrain_build(s2.manifest, runners.terrain_hashes(manifest['terrain']))
    stages = plan(profile, profile_path, manifest_path, constants=constants, loader=loader, bodies=bodies, manifest=manifest)
    code = supervisor.run(profile, run_dir, stages=stages, spec_path=spec_path, budget=budget, cuda_visible_devices=constants.cuda_visible_devices, threads=constants.threads,
                          identity_modules=MEASURE_MODULES, runner=runner_record(profile, stages, manifest, manifest_sha256, constants, condition, approval), scrub_keys=scrub_keys,
                          stream=stream, **options)
    if code == supervisor.EXIT_OK:
        print(headline(run_io.read_json(Path(run_dir) / MEASURE_FILE)), file=out if out is not None else sys.stdout, flush=True)
    return code


def measure(profile, run_dir, *, profile_path, manifest_path, spec_path, budget, constants: runners.RunnerConstants = runners.RunnerConstants(), scrub_keys=(), loader: str = LOADER,
            bodies=None, stream=None, out=None, approval=None, **options) -> int:
    """The measurement runner of the plan (P6; ``runners.measure`` is its entry point under that name): the measure profile ``profile`` (loaded) of the file ``profile_path`` (the children load that
    file and check its lock, so it must be the file that ``profile`` was loaded from) is run in ``run_dir`` and the exit code of the process is returned (0, 2 or 3: the table of ``supervisor``).
    Everything that is wrong with the profile, the manifest or the scales before the supervisor starts is a printed refusal (2) with the match keys of ``scrub_keys`` removed, and leaves no run
    folder. A measurement that finds a difference is a finished measurement: exit code 0, ``identical`` false in ``MEASURE.json`` and ``NOT identical`` in the line that is printed to ``out``
    (default: standard output). ``approval``: the text of the author's approval, recorded as given in the ledger header (``runners.approval_record``). ``options`` go to ``supervisor.run`` (``e6``,
    ``capture``, ``verify``, ``start_gate``, ``poll_s``, ``margin_s``, ``repo``, ``environ``).

    Importing this module imports numpy and torch. A process that has to apply the native-thread policy first (the command line) imports ``c_ept.run.guards``, calls ``native_policy()`` and only
    then imports this module; ``supervisor.run`` calls it again as its first step."""
    return supervisor.main_guard(lambda: _start(profile, profile_path, run_dir, manifest_path=manifest_path, spec_path=spec_path, budget=budget, constants=constants, scrub_keys=scrub_keys,
                                                loader=loader, bodies=bodies, stream=stream, out=out, approval=approval, options=options),
                                 scrub_keys=scrub_keys, stream=stream)


def start(profile_path, run_dir, *, manifest_path, spec_path, budget, constants: runners.RunnerConstants = runners.RunnerConstants(), scrub_keys=(), loader: str = LOADER, bodies=None,
          stream=None, out=None, approval=None, **options) -> int:
    """``measure`` for the measure profile at ``profile_path``, which is loaded here (a profile that is not acceptable is a printed refusal, exit code 2, as every other refusal)."""
    return supervisor.main_guard(lambda: _start(run_profile.load(profile_path), profile_path, run_dir, manifest_path=manifest_path, spec_path=spec_path, budget=budget,
                                                constants=constants, scrub_keys=scrub_keys, loader=loader, bodies=bodies, stream=stream, out=out, approval=approval, options=options),
                                 scrub_keys=scrub_keys, stream=stream)
