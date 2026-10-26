"""Tests of the measurement runner of the run kernel (``c_ept.run.measure``; S4b-prep K6c, decisions D10 P6, D11, D32).

D10 P6: the reporting device is deterministic GPU float32, after "a short same-setting bit-identity check of two GPU calls". The runner reads a measure profile that names a train profile
(path and SHA-256) and a step budget, and under the supervisor runs the statistics stage of that train profile, then ``repeat`` (two or more) short training calls, each in a fresh child
process with the same device and the same seed, then a comparison stage. A call trains one epoch of the first ``steps`` FIT batches, evaluates STOP on its first ``steps`` batches and
records the loss of every step as bits, a hash of every key of the model, optimiser and random state, the checkpoint files, the numeric state and the time of every step. The comparison
finds whether the calls agree on every hash and, if not, the first key at which they differ. It reads the roles FIT and STOP only.

These tests use the toy EPT world, short CPU children that see ``CUDA_VISIBLE_DEVICES=''`` and, for the non-determinism that the measurement must find, a child body that nudges the
parameters by one ulp at a chosen step of a chosen call. No data, no GPU, no model server. The numeric flags that ``train`` sets are given back after each test.

해석: 이 시험은 측정 프로파일 한 장이 통계 단계, 새 프로세스의 짧은 학습 호출 둘 이상, 비교 단계로 이어지고, 같은 설정의 두 호출이 비트까지 같음을(또는 처음 다른 키를) 해시로 판정하는지
장난감 자료와 CPU로 확인한다. 실제 GPU 호출은 하지 않는다.
"""
import dataclasses
import hashlib
import io
import json
import os
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from c_ept.ept import data, training_step
from c_ept.ept import train as ept_train
from c_ept.run import guards
from c_ept.run import io as run_io
from c_ept.run import ledger as run_ledger
from c_ept.run import measure, numeric, runners, supervisor
from c_ept.run import profile as run_profile
from c_ept.run.identity import Identity
from c_ept.tests.test_run_runners import (E6_RECORD, KEY, REPO, S4B, TOY_LOADER, inputs, manifest_document, quiet_probe, reps_sha, s4b_scales, with_scales, world,   # noqa: F401
                                          write_manifest)

CPU2 = runners.RunnerConstants(device='cpu', threads=1, batch_size=2)                       # two engagements a batch: the toy FIT split has three batches, the STOP split four
STATS_BODY, CALL_BODY = 'c_ept.tests.run_measure_bodies:stats_stage', 'c_ept.tests.run_measure_bodies:call_stage'
PERTURBED_BODY = 'c_ept.tests.run_measure_bodies:perturbed_call_stage'
APPROVAL = 'Approved by the author: measure P6 on the CPU toy, "no GPU".\nSecond line, a backslash \\ and 한글 승인 '


@pytest.fixture(autouse=True)
def restore_torch_state(monkeypatch):
    """``call`` changes process-wide settings (threads, the numeric policy, the global random streams); give them back after each test."""
    from c_ept.tests.test_run_numeric import read_flags, write_flags
    threads, flags = torch.get_num_threads(), read_flags()
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8'))
    yield
    torch.set_num_threads(threads)
    write_flags(flags)


def write_measure(folder, *, steps=2, repeat=2, name='measure.json', train_name='train.json', **train_fields):
    """A measure profile and the train profile that it names (by the relative path and the SHA-256 of the file); ``(path of the measure profile, its loaded Profile)``."""
    train = folder / train_name
    train.write_bytes((json.dumps({'kind': 'train', **{**S4B, **train_fields}}, indent=2) + '\n').encode('utf-8'))
    document = {'kind': 'measure', 'train_profile': train_name, 'train_sha256': hashlib.sha256(train.read_bytes()).hexdigest(), 'steps': steps, 'repeat': repeat}
    path = folder / name
    path.write_bytes((json.dumps(document, indent=2) + '\n').encode('utf-8'))
    return path, run_profile.load(path)


# ---- the batches of a short call -----------------------------------------------------------------------------------------------------------------
class FakeBatches:
    """What ``ept.train`` asks of FIT or STOP batches: ``len``, ``epoch``, and the attributes of the rows, the statistics and the order."""
    rows = SimpleNamespace(role='FIT')
    stats, training, batch_size, seed, swap_mode = SimpleNamespace(sha256='s' * 64), True, 2, 1, 'average'

    def __init__(self, count):
        self.count, self.closed, self.generators = count, [], []

    def __len__(self):
        return self.count

    def epoch(self, epoch, start=0):
        def generate():
            try:
                for index in range(start, self.count):
                    yield (epoch, index)
            finally:
                self.closed.append(epoch)
        generator = generate()
        self.generators.append(generator)                                                                 # kept alive: only an explicit close can end it, not the collector
        return generator


def test_the_step_limit_keeps_the_first_batches_of_every_epoch_and_the_attributes_of_the_batches():
    inner = FakeBatches(5)
    limited = measure.StepLimited(inner, 3)
    assert len(limited) == 3 and list(limited.epoch(0)) == [(0, 0), (0, 1), (0, 2)] and list(limited.epoch(4)) == [(4, 0), (4, 1), (4, 2)]
    assert list(limited.epoch(1, 2)) == [(1, 2)] and list(limited.epoch(1, 3)) == []                   # a start inside the limit, and at its end
    assert (limited.rows.role, limited.training, limited.batch_size, limited.seed, limited.swap_mode, limited.stats.sha256) == ('FIT', True, 2, 1, 'average', 's' * 64)
    assert inner.closed == [0, 4, 1]                                                                   # an inner iterator that was started is closed whether or not it ran dry (one that never ran has nothing to close)
    assert len(measure.StepLimited(FakeBatches(2), 9)) == 2                                            # a limit above the length is the length
    for bad_start in (-1, 4, 1.5, True):
        with pytest.raises(ValueError, match='start'):
            limited.epoch(0, bad_start)
    for bad in (0, -1, 1.5, True, '2', None):
        with pytest.raises(measure.MeasureRefused, match='steps'):
            measure.StepLimited(FakeBatches(2), bad)
    with pytest.raises(AttributeError):
        limited.no_such_attribute


def test_a_step_limited_call_reads_no_batch_beyond_its_limit():
    built = []

    class Counting(FakeBatches):
        def epoch(self, epoch, start=0):
            for index in range(start, self.count):
                built.append(index)
                yield (epoch, index)
    list(measure.StepLimited(Counting(50), 2).epoch(0))
    assert built == [0, 1]                                                                             # the generator was cut: batch 2 was never assembled


# ---- the roles that a measurement may read --------------------------------------------------------------------------------------------------------
def test_the_measurement_reads_the_roles_fit_and_stop_only():
    def batches(role):
        return SimpleNamespace(rows=SimpleNamespace(role=role))
    good = SimpleNamespace(fit=batches('FIT'), stop=batches('STOP'))
    measure.check_roles(good)
    for fit, stop in (('FIT', 'DEV'), ('FIT', 'TEST'), ('DEV', 'STOP'), ('TEST', 'STOP'), ('STOP', 'FIT'), ('FIT', 'FIT'), (None, 'STOP')):
        with pytest.raises(measure.MeasureRefused, match='FIT and STOP') as raised:
            measure.check_roles(SimpleNamespace(fit=batches(fit), stop=batches(stop)))
        assert supervisor.exit_code_for(raised.value) == 2
    assert measure.ROLES == ('FIT', 'STOP')


# ---- the record of a step ------------------------------------------------------------------------------------------------------------------------
def test_the_recorder_keeps_every_loss_as_bits_and_every_step_time_and_gives_the_function_back(monkeypatch):
    ticks = iter([10.0, 10.5, 11.0, 11.75])

    def fake_step(*args, **kwargs):
        return torch.tensor(0.1 + len(args))
    monkeypatch.setattr(training_step, 'optimize_batch', fake_step)
    recorder = measure.StepRecorder(clock=lambda: next(ticks))
    with measure.recording_steps(recorder):
        assert training_step.optimize_batch is not fake_step
        first, second = training_step.optimize_batch(1, 2), training_step.optimize_batch(1, 2, 3)
    assert training_step.optimize_batch is fake_step                                                    # restored
    assert float(first) == float(torch.tensor(2.1)) and float(second) == float(torch.tensor(3.1))      # the value that the trainer reads is the one the step made
    assert recorder.seconds == [0.5, 0.75]
    assert recorder.bits == [struct.pack('>d', float(torch.tensor(2.1))).hex(), struct.pack('>d', float(torch.tensor(3.1))).hex()] and all(len(b) == 16 for b in recorder.bits)
    with pytest.raises(RuntimeError, match='boom'):
        def failing(*args, **kwargs):
            raise RuntimeError('boom')
        monkeypatch.setattr(training_step, 'optimize_batch', failing)
        with measure.recording_steps(measure.StepRecorder()):
            training_step.optimize_batch()
    assert training_step.optimize_batch is failing                                                      # restored after an error too


def test_the_digests_of_a_state_are_made_key_by_key_and_see_the_dtype_the_shape_and_one_bit():
    state = {'a': torch.arange(6, dtype=torch.float32).reshape(2, 3), 'b': {'c': torch.tensor(3), 'd': [torch.zeros(2), 2.5, 'text', None, True]}, 'e': np.arange(3),
             'f': [1, 2.5, 'x'], 'g': {0: torch.zeros(1), 12: torch.ones(1)}}
    flat = measure.state_hashes(state, 'model/epoch_0001')
    assert set(flat) == {'model/epoch_0001/a', 'model/epoch_0001/b/c', 'model/epoch_0001/b/d/00000', 'model/epoch_0001/b/d/00001', 'model/epoch_0001/b/d/00002',
                         'model/epoch_0001/b/d/00003', 'model/epoch_0001/b/d/00004', 'model/epoch_0001/e', 'model/epoch_0001/f', 'model/epoch_0001/g/00000', 'model/epoch_0001/g/00012'}
    assert [key for key in flat if key.startswith('model/epoch_0001/g/')] == sorted(key for key in flat if key.startswith('model/epoch_0001/g/'))     # an integer key sorts as a number does
    assert measure.state_hashes(state, 'model/epoch_0001') == flat and all(len(v) == 64 for v in flat.values())
    other = {**state, 'a': state['a'].clone()}
    other['a'].view(-1)[5] = torch.nextafter(other['a'].view(-1)[5], torch.tensor(float('inf')))
    changed = measure.state_hashes(other, 'model/epoch_0001')
    assert [key for key in flat if flat[key] != changed[key]] == ['model/epoch_0001/a']                 # one ulp in one element: one key
    assert measure.state_hashes({**state, 'a': state['a'].to(torch.float64)}, 'x')['x/a'] != measure.state_hashes(state, 'x')['x/a']
    assert measure.state_hashes({**state, 'a': state['a'].reshape(3, 2)}, 'x')['x/a'] != measure.state_hashes(state, 'x')['x/a']
    with pytest.raises(measure.MeasureRefused, match='state'):
        measure.state_hashes({'x': object()}, 'p')


# ---- the comparison ----------------------------------------------------------------------------------------------------------------------------
def fake_record(bits=('1' * 16, '2' * 16, '3' * 16), model=None, numeric_state=None, stop='4' * 16, extra=None):
    hashes = {'stop_loss/epoch_0001': stop, 'lrs/epoch_0001': 'a' * 64, 'epoch_metrics/epoch_0001/json': 'b' * 64, 'head/epoch_0001': 'c' * 64, 'best/epoch_0001': 'd' * 64,
              'optimizer/epoch_0001/state/00000/exp_avg': 'e' * 64, 'rng/epoch_0001/torch': 'f' * 64, 'loop/epoch_0001/best_value': '9' * 64, **(model or {'model/epoch_0001/w': '7' * 64})}
    hashes.update(extra or {})
    return {'numeric': {'measured': numeric_state or {'threads': 1, 'device': 'cpu', 'torch_version': '2.10.0'}}, 'losses': {'bits': list(bits)}, 'hashes': hashes}


def test_two_records_that_agree_on_every_key_are_identical():
    result = measure.compare([fake_record(), fake_record()])
    assert result['identical'] is True and result['first_difference'] is None and result['differences'] == {'count': 0, 'listed': []}
    assert result['keys_compared'] == len(measure.flatten(fake_record())) and result['calls_compared'] == 2


def test_the_first_difference_is_the_earliest_in_time_and_cause_not_the_first_in_the_alphabet():
    """The order: the numeric state (a cause), the loss of each step in order, the STOP loss, the epoch metrics, the head, the model, the optimiser, the random state, the best checkpoint."""
    reference = fake_record()
    cases = [({'numeric_state': {'threads': 8, 'device': 'cpu', 'torch_version': '2.10.0'}}, 'numeric/threads'),
             ({'bits': ('1' * 16, '2' * 16, '9' * 16)}, 'loss/000003'),
             ({'bits': ('1' * 16, '8' * 16, '9' * 16)}, 'loss/000002'),                                  # two steps differ: the earlier one
             ({'stop': '5' * 16}, 'stop_loss/epoch_0001'),
             ({'extra': {'epoch_metrics/epoch_0001/json': '0' * 64}}, 'epoch_metrics/epoch_0001/json'),
             ({'extra': {'head/epoch_0001': '0' * 64}}, 'head/epoch_0001'),
             ({'model': {'model/epoch_0001/w': '0' * 64}}, 'model/epoch_0001/w'),
             ({'extra': {'optimizer/epoch_0001/state/00000/exp_avg': '0' * 64}}, 'optimizer/epoch_0001/state/00000/exp_avg'),
             ({'extra': {'rng/epoch_0001/torch': '0' * 64}}, 'rng/epoch_0001/torch'),
             ({'extra': {'best/epoch_0001': '0' * 64}}, 'best/epoch_0001')]
    for change, key in cases:
        result = measure.compare([reference, fake_record(**change)])
        assert result['identical'] is False and result['first_difference']['key'] == key and result['first_difference']['call'] == 2, (key, result['first_difference'])
    both = measure.compare([reference, fake_record(bits=('1' * 16, '2' * 16, '9' * 16), model={'model/epoch_0001/w': '0' * 64}, stop='5' * 16)])
    assert both['first_difference']['key'] == 'loss/000003' and both['differences']['count'] == 3
    assert [d['key'] for d in both['differences']['listed']] == ['loss/000003', 'stop_loss/epoch_0001', 'model/epoch_0001/w']
    assert both['first_difference'] == {'call': 2, 'key': 'loss/000003', 'reference': '3' * 16, 'other': '9' * 16}


def test_a_key_that_only_one_call_has_is_a_difference_and_the_call_that_differs_is_named():
    third = fake_record(extra={'model/epoch_0001/extra': '1' * 64})
    result = measure.compare([fake_record(), fake_record(), third])
    assert result['identical'] is False and result['first_difference'] == {'call': 3, 'key': 'model/epoch_0001/extra', 'reference': None, 'other': '1' * 64} and result['calls_compared'] == 3
    short = measure.compare([fake_record(), fake_record(bits=('1' * 16, '2' * 16))])
    assert short['first_difference']['key'] in ('loss/000003', 'loss_count') and short['identical'] is False          # a call that made fewer steps is not the same call
    assert measure.compare([fake_record(), fake_record(bits=('1' * 16, '2' * 16))])['differences']['count'] == 2


def test_the_comparison_lists_at_most_a_bounded_number_of_differences_but_counts_them_all():
    many = fake_record(model={f'model/epoch_0001/w{i:03d}': str(i % 10) * 64 for i in range(200)})
    other = fake_record(model={f'model/epoch_0001/w{i:03d}': str((i + 1) % 10) * 64 for i in range(200)})
    result = measure.compare([many, other])
    assert result['differences']['count'] == 200 and len(result['differences']['listed']) == measure.LISTED_DIFFERENCES < 200


@pytest.mark.parametrize('records', [[], [fake_record()], [fake_record(), {'losses': {'bits': []}, 'hashes': {}}], [fake_record(), None]])
def test_fewer_than_two_calls_or_a_record_that_is_not_a_call_is_refused(records):
    with pytest.raises(measure.MeasureRefused):
        measure.compare(records)


# ---- one call, in this process ---------------------------------------------------------------------------------------------------------------------
def ready_world(tmp_path, inputs, reps_sha, **train_fields):
    profile_path, profile = write_measure(tmp_path, **train_fields)
    ready = with_scales(inputs, s4b_scales(tmp_path, reps_sha))
    run_dir = tmp_path / 'run'
    runners.write_stats(profile.train, run_dir, inputs=ready, constants=CPU2)
    return profile_path, profile, ready, run_dir


def test_a_call_trains_the_first_steps_of_one_epoch_and_records_losses_hashes_numeric_state_and_timings(inputs, tmp_path, reps_sha):
    profile_path, profile, ready, run_dir = ready_world(tmp_path, inputs, reps_sha)
    record = measure.call(profile, run_dir, inputs=ready, constants=CPU2, number=1, probe=quiet_probe)
    assert run_io.read_json(run_dir / measure.call_paths(1)['record']) == record and record['schema'] == measure.SCHEMA_CALL
    assert (record['call'], record['steps'], record['steps_made'], record['epochs_run']) == (1, 2, 2, 1)
    assert record['roles_read'] == ['FIT', 'STOP'] and record['batches'] == {'fit': 2, 'stop': 2}              # three FIT batches and four STOP batches were cut to two
    assert record['profile_sha256'] == profile.sha256 and record['train_profile_sha256'] == profile.train.sha256 and record['runner'] == CPU2.to_record()
    assert record['process'] == {'pid': os.getpid(), 'executable': record['process']['executable']}
    head = run_io.read_json(run_dir / record['checkpoint']['final'] / 'head.json')
    assert record['effective_threads'] == head['threads'] and isinstance(record['effective_threads'], int)         # m5: what the trainer ran with, read after it set its own count
    bits = record['losses']['bits']
    assert len(bits) == 2 and all(len(b) == 16 for b in bits) and all(np.isfinite(struct.unpack('>d', bytes.fromhex(b))[0]) for b in bits)
    assert record['losses']['sha256'] == run_io.canonical_sha256(bits)
    assert record['numeric']['measured']['device'] == 'cpu' and all(record['numeric']['measured'][k] == numeric.REQUIRED[k] for k in numeric.REQUIRED)
    hashes = record['hashes']
    assert hashes['stop_loss/epoch_0001'] == struct.pack('>d', record['stop_losses'][0]).hex() and len(record['stop_losses']) == 1
    assert {key.split('/')[0] for key in hashes} == {'stop_loss', 'lrs', 'epoch_metrics', 'head', 'model', 'optimizer', 'rng', 'loop', 'best'}
    best = run_dir / measure.call_paths(1)['ckpt'] / 'epoch_0001' / 'best.pt'
    assert hashes['best/epoch_0001'] == hashlib.sha256(best.read_bytes()).hexdigest()
    state = torch.load(best, weights_only=True, map_location='cpu')
    assert {key for key in hashes if key.startswith('model/epoch_0001/')} == {f'model/epoch_0001/{name}' for name in torch.load(
        run_dir / measure.call_paths(1)['ckpt'] / 'epoch_0001' / 'last.pt', weights_only=True, map_location='cpu')['model']} and state
    files = record['files']
    assert files and all(hashlib.sha256((run_dir / name).read_bytes()).hexdigest() == digest for name, digest in files.items())
    assert f"{measure.call_paths(1)['ckpt']}/epoch_0001/last.pt" in files and f"{measure.call_paths(1)['ckpt']}/epoch_0001/head.json" in files
    timing = record['timing']
    assert len(timing['step_seconds']) == 2 and all(isinstance(s, float) and s > 0 for s in timing['step_seconds']) and timing['epoch_seconds'] > 0
    assert set(timing) >= {'step_seconds', 'epoch_seconds', 'load_seconds', 'train_step_seconds', 'stop_seconds', 'components', 'training_seconds'}
    assert record['termination']['reason'] == 'max_epochs' and torch.cuda.is_initialized() is False


def test_two_calls_of_one_process_are_bit_identical_on_the_cpu_and_the_comparison_says_so(inputs, tmp_path, reps_sha):
    """What the measurement asks of the GPU (D10 P6) is what T002 already had on the CPU: same settings, same seed, same bits."""
    profile_path, profile, ready, run_dir = ready_world(tmp_path, inputs, reps_sha)
    first = measure.call(profile, run_dir, inputs=ready, constants=CPU2, number=1, probe=quiet_probe)
    torch.set_num_threads(1)                                                                              # the CPU trainer leaves its own thread count (8) behind; a fresh process starts at 1
    second = measure.call(profile, run_dir, inputs=ready, constants=CPU2, number=2, probe=quiet_probe)
    assert first['hashes'] == second['hashes'] and first['losses']['bits'] == second['losses']['bits'] and first['files'] != second['files']          # other folders, other files
    result = measure.compare([first, second])
    assert result['identical'] is True and result['first_difference'] is None and result['keys_compared'] > 20, result['differences']


def test_the_measure_module_says_what_the_short_calls_do_not_exercise():
    """m2 of the K6c review: one epoch cut to ``steps`` batches runs the cosine schedule over ``steps`` steps; the text must say that the epoch boundary and the resume are not exercised."""
    assert 'NOT exercised' in measure.__doc__ and 'cosine schedule' in measure.__doc__ and 'warmup_steps' in measure.__doc__


def test_the_comparison_stage_checks_that_the_checkpoint_files_are_the_ones_that_the_calls_recorded(inputs, tmp_path, reps_sha):
    """The hashes of a record were taken when the call ended. A checkpoint file that changed afterwards (or a record that names a file outside the call's folder) means that the records
    are not about the files on disk: the comparison refuses (exit code 2) instead of comparing records alone."""
    profile_path, profile, ready, run_dir = ready_world(tmp_path, inputs, reps_sha)
    measure.call(profile, run_dir, inputs=ready, constants=CPU2, number=1, probe=quiet_probe)
    torch.set_num_threads(1)
    measure.call(profile, run_dir, inputs=ready, constants=CPU2, number=2, probe=quiet_probe)
    measure.compare_stage(run_dir, str(profile_path), profile.sha256)
    document = run_io.read_json(run_dir / measure.MEASURE_FILE)
    assert document['identical'] is True and [c['files_verified'] for c in document['calls']] == [len(run_io.read_json(run_dir / measure.call_paths(k)['record'])['files']) for k in (1, 2)]
    best = run_dir / measure.call_paths(2)['ckpt'] / 'epoch_0001' / 'best.pt'
    original = best.read_bytes()
    best.write_bytes(original + b'\x00')                                                                  # one byte more after the call has ended
    (run_dir / measure.MEASURE_FILE).unlink()
    with pytest.raises(measure.MeasureRefused, match='call 2: 1 of') as raised:
        measure.compare_stage(run_dir, str(profile_path), profile.sha256)
    assert supervisor.exit_code_for(raised.value) == 2 and not (run_dir / measure.MEASURE_FILE).exists()
    best.write_bytes(original)
    record_path = run_dir / measure.call_paths(1)['record']
    record = run_io.read_json(record_path)
    record['files']['measure/call_2/ckpt/epoch_0001/best.pt'] = record['files'][f"{measure.call_paths(1)['ckpt']}/epoch_0001/best.pt"]      # a record that points at the other call's folder
    run_io.write_json_atomic(record_path, record)
    with pytest.raises(measure.MeasureRefused, match='not in the checkpoint folder'):
        measure.compare_stage(run_dir, str(profile_path), profile.sha256)
    best.unlink()
    record['files'].pop('measure/call_2/ckpt/epoch_0001/best.pt')
    run_io.write_json_atomic(record_path, record)
    with pytest.raises(measure.MeasureRefused, match='changed or are missing'):
        measure.compare_stage(run_dir, str(profile_path), profile.sha256)                                  # the file of call 2 is gone


@pytest.mark.parametrize('step, key_group', [(1, 'loss'), (2, 'stop_loss')])
def test_a_nudge_of_the_parameters_in_the_second_call_is_found_and_named(inputs, tmp_path, reps_sha, step, key_group):
    """The injected non-determinism: after the optimiser step ``step`` of the second call every parameter moves by one ulp. At step 1 the loss of step 2 differs; at the last step the
    losses of the steps agree and the STOP value that follows differs. Either way the comparison names the call and the first key."""
    from c_ept.tests.run_measure_bodies import one_ulp_up
    profile_path, profile, ready, run_dir = ready_world(tmp_path, inputs, reps_sha)
    first = measure.call(profile, run_dir, inputs=ready, constants=CPU2, number=1, probe=quiet_probe)
    real, seen = training_step.optimize_batch, []

    def nudging(fa, *args, **kwargs):
        total = real(fa, *args, **kwargs)
        seen.append(1)
        if len(seen) == step:
            for parameter in fa.parameters():
                one_ulp_up(parameter)
        return total
    training_step.optimize_batch = nudging
    torch.set_num_threads(1)                                                                              # the CPU trainer leaves its own thread count (8) behind; a fresh process starts at 1
    try:
        second = measure.call(profile, run_dir, inputs=ready, constants=CPU2, number=2, probe=quiet_probe)
    finally:
        training_step.optimize_batch = real
    result = measure.compare([first, second])
    assert result['identical'] is False and result['first_difference']['call'] == 2 and result['first_difference']['key'].split('/')[0] == key_group, result['first_difference']
    if step == 1:
        assert result['first_difference']['key'] == 'loss/000002' and first['losses']['bits'][0] == second['losses']['bits'][0]                  # step 1 itself was the same
    assert result['differences']['count'] > 1 and any(d['key'].startswith('model/') for d in result['differences']['listed'])


def test_a_call_refuses_the_roles_dev_and_test_before_it_trains(inputs, tmp_path, reps_sha):
    profile_path, profile, ready, run_dir = ready_world(tmp_path, inputs, reps_sha)

    def dev_as_stop(train_profile, opened, stats, constants):
        from c_ept.run import runners as r
        good = r.make_train_data(train_profile, opened, stats, constants)
        stop = SimpleNamespace(rows=SimpleNamespace(role='DEV'))
        return dataclasses.replace(good, stop=stop)
    with pytest.raises(measure.MeasureRefused, match='FIT and STOP') as raised:
        measure.call(profile, run_dir, inputs=ready, constants=CPU2, number=1, probe=quiet_probe, make_data=dev_as_stop)
    assert supervisor.exit_code_for(raised.value) == 2 and not (run_dir / measure.CALL_FOLDER).exists()


def test_numeric_states_that_differ_between_two_calls_are_the_first_difference_and_are_not_equal(inputs, tmp_path, reps_sha):
    """Two calls of one process (the first leaves the thread count of the CPU trainer, 8, behind, so the second reads another state): the numeric state is the first key at which they
    differ and MEASURE.json says that the states are not equal. In fresh processes (the supervised runs) the states are equal."""
    profile_path, profile, ready, run_dir = ready_world(tmp_path, inputs, reps_sha)
    torch.set_num_threads(1)
    measure.call(profile, run_dir, inputs=ready, constants=CPU2, number=1, probe=quiet_probe)
    torch.set_num_threads(3)                                                                               # another count than the first call read (1)
    measure.call(profile, run_dir, inputs=ready, constants=CPU2, number=2, probe=quiet_probe)
    measure.compare_stage(run_dir, str(profile_path), profile.sha256)
    document = run_io.read_json(run_dir / measure.MEASURE_FILE)
    assert document['numeric_equal'] is False and document['identical'] is False
    assert document['first_difference'] == {'call': 2, 'key': 'numeric/threads', 'reference': '1', 'other': '3'}
    assert document['numeric']['call_1']['threads'] == 1 and document['numeric']['call_2']['threads'] == 3


def test_a_call_refuses_loss_scales_that_were_made_for_other_representatives_than_the_rows_of_the_call(inputs, tmp_path, reps_sha):
    from c_ept.run import supervisor as sup
    profile_path, profile, ready, run_dir = ready_world(tmp_path, inputs, reps_sha)
    other = with_scales(inputs, s4b_scales(tmp_path, 'f' * 64, 'other_reps.json'))                          # the right condition, other 31 representatives than the rows hold
    with pytest.raises(runners.ScalesConditionRefused, match='representatives') as raised:
        measure.call(profile, run_dir, inputs=other, constants=CPU2, number=1, probe=quiet_probe)
    assert sup.exit_code_for(raised.value) == 2 and not (run_dir / measure.CALL_FOLDER).exists()
    with pytest.raises(runners.ScalesConditionRefused, match='no scales file'):
        measure.call(profile, run_dir, inputs=with_scales(inputs, None), constants=CPU2, number=1, probe=quiet_probe)


def test_a_call_folder_that_is_not_empty_is_refused_because_a_cut_call_is_not_resumed(inputs, tmp_path, reps_sha):
    profile_path, profile, ready, run_dir = ready_world(tmp_path, inputs, reps_sha)
    folder = run_dir / measure.call_paths(1)['folder']
    folder.mkdir(parents=True)
    (folder / 'leftover.txt').write_text('a call that was cut', encoding='utf-8')
    with pytest.raises(measure.MeasureRefused, match='not empty'):
        measure.call(profile, run_dir, inputs=ready, constants=CPU2, number=1, probe=quiet_probe)
    assert sorted(p.name for p in folder.iterdir()) == ['leftover.txt']                                      # nothing was added or removed


def test_a_call_needs_the_statistics_of_the_statistics_stage_and_a_measure_profile(inputs, tmp_path, reps_sha):
    profile_path, profile = write_measure(tmp_path)
    ready = with_scales(inputs, s4b_scales(tmp_path, reps_sha))
    with pytest.raises(run_io.IORefused, match='statistics'):
        measure.call(profile, tmp_path / 'run', inputs=ready, constants=CPU2, number=1, probe=quiet_probe)
    with pytest.raises(measure.MeasureRefused, match='measure profile'):
        measure.call(profile.train, tmp_path / 'run', inputs=ready, constants=CPU2, number=1, probe=quiet_probe)
    for bad in (0, -1, 1.5, True, '1', None):
        with pytest.raises(measure.MeasureRefused, match='call number'):
            measure.call(profile, tmp_path / 'run', inputs=ready, constants=CPU2, number=bad, probe=quiet_probe)


def test_a_call_holds_the_numeric_policy_before_it_builds_anything(inputs, tmp_path, reps_sha, monkeypatch):
    from c_ept.tests.test_run_numeric import OPPOSITE, read_flags, write_flags
    profile_path, profile, ready, run_dir = ready_world(tmp_path, inputs, reps_sha)
    write_flags(OPPOSITE)
    seen = []
    real = ept_train.train

    def spy(*args, **kwargs):
        seen.append(read_flags())
        return real(*args, **kwargs)
    monkeypatch.setattr(ept_train, 'train', spy)
    measure.call(profile, run_dir, inputs=ready, constants=CPU2, number=1, probe=quiet_probe)
    assert seen == [{name: numeric.REQUIRED[name] for name in OPPOSITE}]
    monkeypatch.setattr(ept_train, 'train', real)
    torch.set_default_dtype(torch.float64)

    def never(*args, **kwargs):
        raise AssertionError('data was built before the numeric policy was held')
    monkeypatch.setattr(data, 'ept_head_rows', never)
    try:
        with pytest.raises(numeric.NumericRefused, match='default_dtype'):
            measure.call(profile, run_dir, inputs=ready, constants=CPU2, number=2, probe=quiet_probe)
    finally:
        torch.set_default_dtype(torch.float32)
    assert not (run_dir / measure.call_paths(2)['folder']).exists()


# ---- the plan, the stage bodies and the supervised run -----------------------------------------------------------------------------------------------
class MeasureRig:
    """One supervised measurement's world: the profiles, a decision record that agrees with the code constants, the injected gates of the supervisor (identity, E6, Git: the working tree
    of a test run is not a clean checkout) and a large budget that reads a made-up memory."""

    def __init__(self, tmp_path, world, reps_sha, *, steps=2, repeat=2, **train_fields):
        from c_ept.tests.test_run_ledger import real_spec
        self.tmp, self.run_dir, self.spec = tmp_path, tmp_path / 'run', real_spec(tmp_path)
        self.profile_path, self.profile = write_measure(tmp_path, steps=steps, repeat=repeat, **train_fields)
        self.scales = s4b_scales(tmp_path, reps_sha)
        self.manifest = write_manifest(tmp_path, manifest_document(world, self.scales))
        self.identity = Identity(commit='ab' * 20, files={'c_ept/run/measure.py': 'cd' * 32}, protected=())
        self.stream, self.out, self.log = io.StringIO(), io.StringIO(), []

    def e6(self):
        self.log.append('e6')
        return dict(E6_RECORD)

    def capture(self, modules, *, repo=None):
        self.log.append(('capture', tuple(modules)))
        return self.identity

    def budget(self, wall_s=10 ** 7):
        return guards.Budget(wall_s, 2 ** 40, measure=lambda: 10 ** 6)

    def start(self, **over):
        values = dict(manifest_path=self.manifest, spec_path=self.spec, budget=self.budget(), constants=CPU2, scrub_keys=[KEY], stream=self.stream, out=self.out, loader=TOY_LOADER,
                      bodies={'stats': STATS_BODY, 'call': CALL_BODY}, e6=self.e6, capture=self.capture, verify=lambda saved, repo=None: None,
                      start_gate=lambda repo, identity, **kwargs: None, poll_s=0.05, repo=REPO)
        values.update(over)
        return measure.start(self.profile_path, self.run_dir, **values)

    def ledger(self):
        return run_ledger.Ledger(self.run_dir)


@pytest.fixture
def rig(tmp_path, world, reps_sha):
    return MeasureRig(tmp_path, world, reps_sha)


def test_the_plan_is_a_statistics_stage_one_stage_for_each_call_and_a_comparison_with_text_arguments_and_bound_inputs(rig):
    manifest = runners.read_manifest(rig.manifest)
    stages = measure.plan(rig.profile, rig.profile_path, rig.manifest, constants=CPU2, manifest=manifest)
    assert [stage.name for stage in stages] == ['stats', 'call_1', 'call_2', 'compare']
    stats, first, second, compare = stages
    assert stats.body == 'c_ept.run.measure:stats_stage' and first.body == second.body == 'c_ept.run.measure:call_stage' and compare.body == 'c_ept.run.measure:compare_stage'
    common = (str(rig.profile_path), rig.profile.sha256, str(rig.manifest), runners.LOADER, 'cpu', '1', '2')
    assert stats.args == common and first.args == (*common, '1') and second.args == (*common, '2') and compare.args == (str(rig.profile_path), rig.profile.sha256)
    bound = runners.manifest_files(manifest)
    base = {rig.profile_path.as_posix(), (rig.tmp / 'train.json').as_posix(), rig.manifest.as_posix(), *bound}                    # the measure profile, the train profile it names, the manifest and its files
    assert set(stats.inputs) == base
    assert set(first.inputs) == {runners.STATS_FILE, runners.STATS_RECORD_FILE, *base} == set(second.inputs)
    assert stats.outputs == (runners.STATS_FILE, runners.STATS_RECORD_FILE)
    assert first.outputs == tuple(measure.call_paths(1)[k] for k in ('record', 'e6')) and second.outputs == tuple(measure.call_paths(2)[k] for k in ('record', 'e6'))
    assert set(compare.inputs) == {measure.call_paths(1)['record'], measure.call_paths(2)['record']} and compare.outputs == (measure.MEASURE_FILE,)
    assert all(stage.wall_cap_s is None for stage in stages)
    swapped = measure.plan(rig.profile, rig.profile_path, rig.manifest, constants=CPU2, manifest=manifest, loader='x.y:z', bodies={'call': 'x.y:w'})
    assert swapped[1].body == swapped[2].body == 'x.y:w' and swapped[0].body == 'c_ept.run.measure:stats_stage' and swapped[1].args[3] == 'x.y:z'


def test_the_repeat_of_the_profile_makes_that_many_call_stages(tmp_path, world, reps_sha):
    rig = MeasureRig(tmp_path, world, reps_sha, repeat=4)
    stages = measure.plan(rig.profile, rig.profile_path, rig.manifest, constants=CPU2, manifest=runners.read_manifest(rig.manifest))
    assert [stage.name for stage in stages] == ['stats', 'call_1', 'call_2', 'call_3', 'call_4', 'compare'] and [stage.args[-1] for stage in stages[1:5]] == ['1', '2', '3', '4']


def test_the_stage_bodies_are_importable_functions_of_a_run_folder_and_texts():
    import importlib
    import inspect
    assert list(inspect.signature(measure.stats_stage).parameters) == ['run_dir', 'profile_path', 'profile_sha256', 'manifest_path', 'loader', 'device', 'threads', 'batch_size']
    assert list(inspect.signature(measure.call_stage).parameters) == ['run_dir', 'profile_path', 'profile_sha256', 'manifest_path', 'loader', 'device', 'threads', 'batch_size', 'number']
    assert list(inspect.signature(measure.compare_stage).parameters) == ['run_dir', 'profile_path', 'profile_sha256']
    for name, function in (('stats', measure.stats_stage), ('call', measure.call_stage), ('compare', measure.compare_stage)):
        module, _, attribute = measure.BODIES[name].partition(':')
        assert getattr(importlib.import_module(module), attribute) is function
    assert measure.MEASURE_MODULES == ('c_ept.run.supervisor', 'c_ept.run.measure')


def test_a_stage_body_checks_the_lock_of_the_profile_and_its_text_arguments_before_it_does_anything(rig, tmp_path):
    good = (str(rig.profile_path), rig.profile.sha256, str(rig.manifest), TOY_LOADER, 'cpu', '1', '2', '1')
    for number, change in enumerate(({1: 'f' * 64}, {1: 'nothex'}, {4: 'tpu'}, {5: 'x'}, {5: '0'}, {6: '-3'}, {6: ''}, {3: 'not a loader'}, {3: 'c_ept.run.runners:missing'}, {7: '0'},
                                     {7: 'one'}, {7: ''}, {7: '3'})):                                             # the last: a call number above the repeat of the profile (2)
        arguments = tuple(change.get(i, text) for i, text in enumerate(good))
        with pytest.raises(run_io.IORefused):
            measure.call_stage(rig.run_dir, *arguments)
    assert not rig.run_dir.exists()
    with pytest.raises(measure.MeasureRefused, match='above the repeat'):                                      # the repeat of the profile is 2: a third call is not part of this measurement
        measure.call_stage(rig.run_dir, *good[:7], '3')
    assert not rig.run_dir.exists()
    with pytest.raises(run_io.IORefused, match='measure profile'):
        measure.call_stage(rig.run_dir, str(rig.tmp / 'train.json'), hashlib.sha256((rig.tmp / 'train.json').read_bytes()).hexdigest(), *good[2:])
    with pytest.raises(run_io.IORefused, match='measure profile'):
        measure.stats_stage(rig.run_dir, str(rig.tmp / 'train.json'), hashlib.sha256((rig.tmp / 'train.json').read_bytes()).hexdigest(), *good[2:7])
    with pytest.raises(run_io.IORefused, match='lock'):
        measure.compare_stage(rig.run_dir, str(rig.profile_path), '0' * 64)
    assert not rig.run_dir.exists()


def test_measure_compares_two_identical_calls(rig):
    """The plan's test (K6): two identical short calls, each in a fresh child process under the supervisor, with the same device and seed; the checkpoint bytes, the loss of every step and
    the epoch metrics are compared by SHA-256 and ``MEASURE.json`` says that they are identical. It holds both numeric states and the time of every step."""
    assert rig.start() == 0, rig.stream.getvalue()
    document = run_io.read_json(rig.run_dir / measure.MEASURE_FILE)
    ledger = rig.ledger()
    assert ledger.stages == (('stats', 'finished'), ('call_1', 'finished'), ('call_2', 'finished'), ('compare', 'finished'))
    assert sorted(ledger.stage('compare').outputs) == [measure.MEASURE_FILE] and ledger.stage('compare').outputs[measure.MEASURE_FILE] == run_io.sha_file(rig.run_dir / measure.MEASURE_FILE)
    assert document['schema'] == measure.SCHEMA_MEASURE and document['identical'] is True and document['first_difference'] is None and document['repeat'] == 2
    assert document['differences'] == {'count': 0, 'listed': []} and document['keys_compared'] > 20 and document['calls_compared'] == 2
    assert document['profile'] == {'sha256': rig.profile.sha256, 'train_profile': 'train.json', 'train_sha256': rig.profile.train.sha256, 'steps': 2, 'repeat': 2}
    assert document['runner'] == CPU2.to_record() and document['roles_read'] == ['FIT', 'STOP']
    records = [run_io.read_json(rig.run_dir / measure.call_paths(k)['record']) for k in (1, 2)]
    pids = {r['process']['pid'] for r in records}
    assert len(pids) == 2 and os.getpid() not in pids                                                           # two fresh processes, neither is this one
    assert document['calls'] == [{'call': k, 'record': measure.call_paths(k)['record'], 'record_sha256': run_io.sha_file(rig.run_dir / measure.call_paths(k)['record']),
                                  'losses_sha256': records[k - 1]['losses']['sha256'], 'hashes_sha256': run_io.canonical_sha256(records[k - 1]['hashes']),
                                  'numeric_measured_sha256': records[k - 1]['numeric']['measured_sha256'], 'steps_made': 2, 'files_verified': len(records[k - 1]['files'])} for k in (1, 2)]
    assert document['numeric'] == {'call_1': records[0]['numeric']['measured'], 'call_2': records[1]['numeric']['measured']} and document['numeric_equal'] is True
    assert all(document['numeric'][name]['device'] == 'cpu' and document['numeric'][name]['float32_matmul_precision'] == 'highest' for name in document['numeric'])
    for number in (1, 2):
        timings = document['timings'][f'call_{number}']
        assert len(timings['step_seconds']) == 2 and timings['step_seconds'] == records[number - 1]['timing']['step_seconds'] and timings['summary']['steps'] == 2
        assert set(timings['summary']) == {'steps', 'total', 'median', 'p90', 'min', 'max'} and timings['summary']['min'] <= timings['summary']['median'] <= timings['summary']['max']
    assert records[0]['hashes'] == records[1]['hashes'] and records[0]['losses']['bits'] == records[1]['losses']['bits']
    header = ledger.header
    written = run_io.read_json(rig.run_dir / run_ledger.HEADER_FILE)['runner']                                  # the plain JSON of the header (the object freezes its lists into tuples)
    assert header.profile_sha256 == rig.profile.sha256 and written['numeric'] == numeric.requested_record() and written['approval'] is None
    assert written['measurement'] == {'profile': rig.profile.as_dict(), 'profile_sha256': rig.profile.sha256, 'train_profile_sha256': rig.profile.train.sha256,
                                      'settings': runners.effective_settings(rig.profile.train), 'steps': 2, 'repeat': 2, 'roles_read': ['FIT', 'STOP'],
                                      'stages': ['stats', 'call_1', 'call_2', 'compare']}
    assert header.runner['batch_size'] == 2 and header.runner['device'] == 'cpu' and ('capture', measure.MEASURE_MODULES) in rig.log
    assert 'identical' in rig.out.getvalue() and 'KR_' not in rig.out.getvalue() and rig.stream.getvalue() == ''


def test_a_perturbation_in_the_third_of_three_calls_is_detected_and_the_run_still_ends_with_exit_0(tmp_path, world, reps_sha, monkeypatch):
    rig = MeasureRig(tmp_path, world, reps_sha, repeat=3)
    monkeypatch.setenv('K6C_PERTURB', '3:1')
    assert rig.start(bodies={'stats': STATS_BODY, 'call': PERTURBED_BODY}) == 0, rig.stream.getvalue()
    document = run_io.read_json(rig.run_dir / measure.MEASURE_FILE)
    assert document['identical'] is False and document['repeat'] == 3 and document['calls_compared'] == 3
    assert document['first_difference']['call'] == 3 and document['first_difference']['key'] == 'loss/000002' and document['differences']['count'] > 1
    assert rig.ledger().stages[-1] == ('compare', 'finished') and len(document['calls']) == 3
    headline = rig.out.getvalue()
    assert 'NOT identical' in headline and 'call 3' in headline and 'loss/000002' in headline


def test_a_measure_profile_that_is_not_one_or_a_train_profile_that_does_not_match_is_exit_2_before_the_reservation(rig, tmp_path):
    train = tmp_path / 'train.json'
    other = tmp_path / 'eval.json'
    other.write_bytes(b'{"kind": "evaluate", "role": "STOP", "models": ["ept"]}\n')
    rig.profile_path = other
    assert rig.start() == 2 and 'measure profile' in rig.stream.getvalue() and not rig.run_dir.exists()
    rig.profile_path = train
    rig.stream.truncate(0)
    assert rig.start() == 2 and 'measure profile' in rig.stream.getvalue() and not rig.run_dir.exists()
    rig.profile_path = tmp_path / 'measure.json'
    train.write_bytes(train.read_bytes() + b' ')                                                              # the train profile was edited after the measure profile locked its hash
    rig.stream.truncate(0)
    assert rig.start() == 2 and 'train_sha256' in rig.stream.getvalue() and not rig.run_dir.exists()


def test_a_cuda_measurement_without_cuda_is_exit_2_before_the_reservation(rig, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert rig.start(constants=runners.RunnerConstants()) == 2 and not rig.run_dir.exists() and 'CUDA' in rig.stream.getvalue()


def test_a_scales_file_of_another_condition_is_exit_2_before_the_reservation(rig, tmp_path):
    from c_ept.tests.test_run_runners import write_scales
    plain = write_scales(tmp_path, 'plain_scales.json')                                                        # the plain law and the raw gold, for an S4b train profile
    rig.manifest = write_manifest(tmp_path, {**run_io.read_json(rig.manifest), 'scales': str(plain)}, 'plain.json')
    assert rig.start() == 2 and 'kill_law' in rig.stream.getvalue() and not rig.run_dir.exists()


def test_the_budget_has_to_hold_the_finalisation_margin_and_nothing_is_reserved_otherwise(rig):
    assert rig.start(budget=rig.budget(wall_s=supervisor.FINALISATION_MARGIN_S - 1)) == 2 and not rig.run_dir.exists() and 'budget' in rig.stream.getvalue()


def test_the_header_holds_the_approval_text_exactly_as_given_and_a_blank_one_is_refused(rig, tmp_path):
    assert rig.start(approval='   ') == 2 and not rig.run_dir.exists() and 'approval' in rig.stream.getvalue()
    assert rig.start(approval=APPROVAL) == 0, rig.stream.getvalue()
    assert rig.ledger().header.runner['approval'] == {'text': APPROVAL, 'sha256': hashlib.sha256(APPROVAL.encode('utf-8')).hexdigest()}


def test_a_call_that_fails_is_exit_3_and_the_comparison_never_runs(rig):
    assert rig.start(bodies={'stats': STATS_BODY, 'call': 'c_ept.tests.run_measure_bodies:no_such_body'}) == 3
    assert rig.ledger().stages == (('stats', 'finished'), ('call_1', 'begun')) and not (rig.run_dir / measure.MEASURE_FILE).exists()


def test_runners_measure_is_the_entry_point_that_the_plan_names_and_delegates_to_this_module():
    import inspect
    parameters = inspect.signature(runners.measure).parameters
    assert list(parameters)[:2] == ['profile', 'run_dir'] and parameters['profile_path'].default is inspect.Parameter.empty
    assert parameters['profile_path'].kind is inspect.Parameter.KEYWORD_ONLY
    assert list(inspect.signature(measure.measure).parameters)[:2] == ['profile', 'run_dir'] and list(inspect.signature(measure.start).parameters)[:2] == ['profile_path', 'run_dir']


def test_runners_measure_refuses_a_loaded_profile_that_is_not_the_file(rig, tmp_path):
    other_path, other = write_measure(tmp_path, steps=3, name='other_measure.json', train_name='other_train.json')
    assert runners.measure(other, rig.run_dir, profile_path=rig.profile_path, manifest_path=rig.manifest, spec_path=rig.spec, budget=rig.budget(), constants=CPU2, stream=rig.stream) == 2
    assert 'profile' in rig.stream.getvalue() and not rig.run_dir.exists()
