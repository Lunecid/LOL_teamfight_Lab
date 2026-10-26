"""Tests of the training runner of the run kernel (``c_ept.run.runners``; S4b-prep K6a, decisions D11, D12, D24).

The runner turns a hash-locked train profile into the objects of the training code: the encoder size, the ``TrainConfig`` (the Track M switches), the
statistics (the D7 list of constant columns, the D12 population), the selected FIT/STOP matches, the rows and the batches (kill law, K condition,
swap mode), and into the two stage bodies of the supervisor (the statistics, the training). It writes the effective values to the ledger header and to
the hashed stage records, refuses a loss-scale file that was made under another kill law or K condition (decision D24, plan Q9), declares the
wall-clock cap of the training stage (24 h plus the epoch bound, two checkpoint writes and the finalisation, the M8 hand-off) and checks E6 before
every epoch.

These tests use the toy EPT world of ``ept_toys`` (eight matches), a hand-written ``B0_SCALES.json``, ``tmp_path`` run folders and short CPU children
that see ``CUDA_VISIBLE_DEVICES=''``. No data, no GPU, no model server. The identity, E6 and Git gates of the supervisor are injected (the working
tree of a test run is not a clean checkout). The equivalence of a whole run with the T002 stack is the task K8, not this file.

해석: 이 시험은 프로파일 한 장이 학습 코드의 실제 인자로 바뀌는 길(설정, 통계, 경기 선택, 행, 배치)과 감독자의 두 단계 본문(통계, 학습), 원장에 남는 명시 기록, 손실 척도
조건의 거부, 학습 단계의 상한 선언, 에폭마다의 E6 확인을 장난감 자료로 확인한다.
"""
import dataclasses
import gc
import hashlib
import importlib
import inspect
import io
import itertools
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from c_ept import runlog
from c_ept.ept import batch_prep, data, encoder, loss
from c_ept.ept import train as ept_train
from c_ept.preprocessing import features
from c_ept.run import io as run_io
from c_ept.run import numeric as run_numeric
from c_ept.run import prefetch
from c_ept.run import profile as run_profile
from c_ept.run import runners
from c_ept.tests.ept_toys import build_ept_world, toy_static
from c_ept.valuation import value

SCALES = {key: 1.0 + 0.25 * i for i, key in enumerate(loss.BLOCKS)}
S4B ={'kind': 'train', 'kill_law': 'shifted', 'k_condition': 'binned', 'decay_groups': 'split', 'swap_mode': 'random_view', 'on_wall_cap': 'best_stop',
       'column_drop': 'store_and_subset', 'stats_population': 'selected_fit', 'selection_seed': 20261022, 'epochs': 2, 'patience': 5}
T002_ENCODER = encoder.EncoderConfig(d=128, layers=4, heads=4)


def profile_of(tmp_path, name='profile.json', **fields):
    """A loaded train profile: the file is written with LF line ends (the lock is the hash of its bytes)."""
    import json
    path = tmp_path / name
    path.write_bytes((json.dumps({'kind': 'train', **fields}, indent=2) + '\n').encode('utf-8'))
    return run_profile.load(path)


# ---- the runner constants ----------------------------------------------------------------------------------------------------------------------
def test_the_runner_constants_are_the_t002_constants_and_choose_the_device_list_of_the_children():
    constants = runners.RunnerConstants()
    assert (constants.batch_size, constants.device, constants.threads) == (32, 'cuda', 8)
    assert constants.cuda_visible_devices == '0'
    cpu = runners.RunnerConstants(device='cpu', threads=1)
    assert cpu.cuda_visible_devices == '' and cpu.to_record() == {'batch_size': 32, 'device': 'cpu', 'threads': 1, 'cuda_visible_devices': ''}
    assert constants.to_record() == {'batch_size': 32, 'device': 'cuda', 'threads': 8, 'cuda_visible_devices': '0'}
    assert runners.BATCH_SIZE == 32 and ept_train.THREADS == constants.threads                          # the T002 values, and the trainer's own thread count


@pytest.mark.parametrize('bad', [{'batch_size': 0}, {'batch_size': True}, {'batch_size': '32'}, {'device': 'tpu'}, {'device': None}, {'threads': 0}, {'threads': False},
                                 {'threads': 1.5}, {'threads': 4096}])
def test_runner_constants_that_are_not_acceptable_are_refused_as_a_refusal_not_a_failure(bad):
    with pytest.raises(run_io.IORefused):
        runners.RunnerConstants(**bad)


# ---- the profile -> TrainConfig mapping ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('size', ['S', 'M'])
@pytest.mark.parametrize('field_channels', [8, 0, 3])
def test_the_encoder_follows_the_size_and_the_field_channels(tmp_path, size, field_channels):
    config = runners.encoder_config(profile_of(tmp_path, size=size, field_channels=field_channels))
    assert config == dataclasses.replace(encoder.SIZES[size], field_channels=field_channels)


def test_the_default_profile_gives_the_train_config_of_the_t002_stack(tmp_path):
    """T002 built ``TrainConfig(EncoderConfig(d=128, layers=4, heads=4), 0.001, 0.01, 10)`` and refused a patience that was not 2."""
    default = profile_of(tmp_path)
    assert runners.encoder_config(default) == T002_ENCODER
    assert runners.train_config(default) == ept_train.TrainConfig(T002_ENCODER, 0.001, 0.01, 10)
    assert runners.train_config(default).patience == 2 and runners.train_config(default).model_spec is None


def test_an_s4b_profile_gives_a_train_config_that_carries_every_track_m_switch(tmp_path):
    profile = profile_of(tmp_path, **S4B, terrain_smooth_weight=0.5, eval_average=False, zone_mark=False, lr=0.002, weight_decay=0.02, warmup=3)
    config = runners.train_config(profile)
    assert config == ept_train.TrainConfig(runners.encoder_config(profile), 0.002, 0.02, 3, patience=5, kill_law='shifted', k_condition='binned', decay_groups='split',
                                           column_drop='store_and_subset', stats_population='selected_fit', terrain_smooth_weight=0.5, swap_mode='random_view',
                                           eval_average=False, on_wall_cap='best_stop', zone_mark=False)


def test_every_train_config_field_comes_from_the_profile_or_is_named_as_runner_only():
    names = {field.name for field in dataclasses.fields(ept_train.TrainConfig)}
    from_profile = set(run_profile.TRAIN_KEYS) | {'encoder'}                                         # the encoder is made from size and field_channels
    assert names - from_profile == set(runners.TRAIN_CONFIG_RUNNER_ONLY) == {'model_spec'}           # N2's model_spec is the S7 profile's business
    assert from_profile <= names


# ---- the routes of the profile fields into the real signatures (introspection) --------------------------------------------------------------------
def resolve(name):
    module, _, attribute = name.partition(':')
    target = importlib.import_module(module)
    for part in attribute.split('.'):
        target = getattr(target, part)
    return target


def test_every_route_names_a_real_parameter_of_a_real_callable_and_a_real_profile_field():
    fields = set(run_profile.FIELDS['train'])
    assert runners.ROUTES                                                                              # a table, not an empty promise
    for callee, routes in runners.ROUTES.items():
        parameters = inspect.signature(resolve(callee)).parameters
        for parameter, field in routes.items():
            assert parameter in parameters, (callee, parameter)
            assert field in fields, (callee, field)


def test_every_field_of_a_train_profile_has_a_consumer():
    routed = {field for routes in runners.ROUTES.values() for field in routes.values()}
    consumed = routed | set(runners.PROFILE_FIELDS_OTHERWISE_USED)
    assert consumed == set(run_profile.FIELDS['train']), sorted(set(run_profile.FIELDS['train']) ^ consumed)
    assert set(runners.PROFILE_FIELDS_OTHERWISE_USED) == {'size', 'selection_seed'}                  # the encoder table key and the membership seed (a derived argument)


def test_the_signatures_that_the_runner_relies_on_still_have_the_shape_it_calls_them_with():
    """The calls of ``runners`` use these shapes; a change of a signature of the training code fails here, not in the middle of a run."""
    heads = inspect.signature(data.ept_head_rows).parameters
    assert [name for name, p in heads.items() if p.kind is inspect.Parameter.KEYWORD_ONLY] == ['matches', 'kill_law', 'k_condition']
    assert list(inspect.signature(data.proxy_matches).parameters) == ['s1', 'share', 'seed']
    assert list(inspect.signature(data.Batches).parameters)[:3] == ['rows', 'batch_size', 'seed'] and 'swap_mode' in inspect.signature(data.Batches).parameters
    assert {'column_drop', 'stats_population', 'store_constant', 'matches', 'check'} <= set(inspect.signature(data.fit_stats).parameters)
    assert list(inspect.signature(ept_train.train).parameters) == ['cfg', 'data', 'out_dir', 'device', 'epochs', 'check', 'seed']
    assert list(inspect.signature(data.store_constant_columns).parameters) == ['s2', 'fit_views']


def test_the_effective_settings_name_every_field_explicitly_including_the_defaults(tmp_path):
    default = profile_of(tmp_path)
    effective = runners.effective_settings(default)
    assert {name: effective['profile'][name] for name in run_profile.FIELDS['train']} == {name: default[name] for name in run_profile.FIELDS['train']}
    for name in ('kill_law', 'k_condition', 'decay_groups', 'terrain_smooth_weight', 'swap_mode', 'eval_average', 'on_wall_cap', 'zone_mark', 'column_drop', 'stats_population'):
        assert name in effective['profile'], name                                                   # explicit, also where the value is the old default
    assert effective['profile_sha256'] == default.sha256
    assert effective['membership_seed'] == default['seed'] == 20261022                                # no selection_seed: the training seed selects (T002)
    fixed = profile_of(tmp_path, 'fixed.json', seed=7, selection_seed=20261022)
    assert runners.effective_settings(fixed)['membership_seed'] == 20261022 and runners.effective_settings(fixed)['profile']['seed'] == 7
    assert 'decay_rule_sha256' not in effective                                                      # the rule table matters only for a split run
    split = runners.effective_settings(profile_of(tmp_path, 'split.json', decay_groups='split'))
    assert split['decay_rule_sha256'] == ept_train.decay_rule_sha256()


# ---- the toy world and the data preparation ------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def world(tmp_path_factory):
    """The EPT toy world whose units all have a kill by their first-kill team (the rows of the shifted kill law), 7 events and 5 edges per view."""
    folder = tmp_path_factory.mktemp('runner_world')
    built = build_ept_world(folder, n_events=7, n_edges=5, consistent_first_kill=True)
    yield built
    del built
    gc.collect()                                                                                      # Windows keeps a folder with open memory maps


@pytest.fixture(scope='module')
def inputs(world):
    return runners.Inputs(s1=world.s1, s2=world.s2, labels=world.labels, static=toy_static(), static_source=None, scales=dict(SCALES), scales_path=None)


CPU = runners.RunnerConstants(device='cpu', threads=1)


def same_rows(a, b):
    """Two ``EngagementRows`` hold the same engagements, frame rows, views and fields of every block."""
    assert a.role == b.role and np.array_equal(a.units, b.units) and np.array_equal(a.frames, b.frames) and a.blocks.keys() == b.blocks.keys()
    for key in a.blocks:
        x, y = a.blocks[key], b.blocks[key]
        assert x.fields.keys() == y.fields.keys() and np.array_equal(x.views, y.views) and np.array_equal(x.group, y.group), key
        for name in x.fields:
            assert np.array_equal(x.fields[name], y.fields[name]), (key, name)


def same_batches(a, b):
    same_rows(a.rows, b.rows)
    assert (a.batch_size, a.seed, a.training, a.swap_mode, len(a)) == (b.batch_size, b.seed, b.training, b.swap_mode, len(b))
    assert a.stats.sha256 == b.stats.sha256


def t002_data(world, share, seed, **rows_kw):
    """The data of the T002 stack, built as ``run_training`` builds it: the statistics of the whole FIT, the rows of the whole role cut to the proxy matches."""
    proxy = data.proxy_matches(world.s1, share, seed)
    stats = data.fit_stats(world.s1, world.s2)
    grouped = {role: data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, role, **rows_kw), proxy[role]) for role in ('FIT', 'STOP')}
    batches = {role: data.Batches(grouped[role], 32, seed, stats=stats, training=role == 'FIT') for role in ('FIT', 'STOP')}
    return stats, batches


def test_the_selected_matches_are_the_proxy_selection_of_the_membership_seed(world, tmp_path):
    default = profile_of(tmp_path)
    chosen = runners.select_matches(default, world.s1)
    expected = data.proxy_matches(world.s1, 0.25, 20261022)
    assert chosen.keys() == expected.keys() and all(np.array_equal(chosen[role], expected[role]) for role in expected)
    other = profile_of(tmp_path, 'other.json', seed=3, fraction=0.5)                                  # no selection seed: the training seed selects (T002)
    assert all(np.array_equal(runners.select_matches(other, world.s1)[role], data.proxy_matches(world.s1, 0.5, 3)[role]) for role in expected)
    fixed = profile_of(tmp_path, 'fixed.json', seed=3, fraction=0.5, selection_seed=5)                # a selection seed: fixed whatever the training seed is (T009)
    assert all(np.array_equal(runners.select_matches(fixed, world.s1)[role], data.proxy_matches(world.s1, 0.5, 5)[role]) for role in expected)


def test_the_default_profile_prepares_the_statistics_and_the_data_of_the_t002_stack(world, inputs, tmp_path):
    profile = profile_of(tmp_path)
    stats = runners.make_stats(profile, inputs)
    train_data = runners.make_train_data(profile, inputs, stats, CPU)
    ref_stats, ref = t002_data(world, 0.25, 20261022)
    assert stats.sha256 == ref_stats.sha256 and stats.dropped == () and stats.dropped_store == () and stats.population == 'whole_fit' and stats.column_drop == 'none'
    same_batches(train_data.fit, ref['FIT'])
    same_batches(train_data.stop, ref['STOP'])
    assert train_data.scales == SCALES and train_data.static is inputs.static and train_data.static_source is None
    assert (train_data.fit.rows.head_rows.kill_law, train_data.fit.rows.head_rows.k_condition) == ('poisson', 'raw')


def test_an_s4b_profile_prepares_the_shifted_binned_selected_fit_data(world, inputs, tmp_path, monkeypatch):
    profile = profile_of(tmp_path, **S4B)
    calls = []
    real = data.store_constant_columns
    monkeypatch.setattr(data, 'store_constant_columns', lambda *args, **kwargs: calls.append(args) or real(*args, **kwargs))
    stats = runners.make_stats(profile, inputs)
    train_data = runners.make_train_data(profile, inputs, stats, CPU)
    assert len(calls) == 1                                                                            # the D7 list is computed once, for the whole FIT store
    fit_views = np.flatnonzero(features.split_views(world.s1, world.s2, 'FIT'))
    d7 = real(world.s2, fit_views)
    proxy = data.proxy_matches(world.s1, 0.25, 20261022)
    direct = data.fit_stats(world.s1, world.s2, matches=proxy['FIT'], column_drop='store_and_subset', stats_population='selected_fit', store_constant=d7)
    assert stats.sha256 == direct.sha256 and stats.dropped_store == d7 and set(d7) <= set(stats.dropped)
    assert stats.population == 'selected_fit' and stats.column_drop == 'store_and_subset'
    selected_views = features.split_views(world.s1, world.s2, 'FIT') & np.isin(np.asarray(world.s2.array('views/match')), proxy['FIT'])
    assert stats.n_views == int(selected_views.sum()) < len(fit_views)                                # the statistics are those of the selected FIT only (D12)
    ref_stats, ref = t002_data(world, 0.25, 20261022, kill_law='shifted', k_condition='binned')       # the rows: whole role cut to the matches ...
    for role, batches in (('FIT', train_data.fit), ('STOP', train_data.stop)):
        same_rows(batches.rows, ref[role].rows)                                                       # ... are the rows of the runner
        assert batches.stats.sha256 == stats.sha256 and batches.batch_size == 32 and batches.seed == 20261022
    assert train_data.fit.swap_mode == 'random_view' and train_data.stop.swap_mode == 'average'       # the STOP batches are never swapped at random
    fit_rows, stop_rows = train_data.fit.rows.head_rows, train_data.stop.rows.head_rows
    assert (fit_rows.kill_law, fit_rows.k_condition, stop_rows.kill_law, stop_rows.k_condition) == ('shifted', 'binned', 'shifted', 'binned')
    expected = value.gold_reps_sha256(value.fit_gold_bin_means(world.s1, world.labels, population='in_scope').means)
    assert fit_rows.k_reps_sha256() == stop_rows.k_reps_sha256() == expected                          # one source of the 31 representatives


# ---- S4b speed-ups SPD item 2 (decision D32): the ordered prefetch of the kernel ----------------------------------------------------------------------
def producer_threads() -> list:
    return [t for t in threading.enumerate() if t.name == prefetch.THREAD_NAME and t.is_alive()]


def no_producer_left(seconds=5.0) -> bool:
    end = time.monotonic() + seconds
    while producer_threads() and time.monotonic() < end:
        time.sleep(0.01)
    return not producer_threads()


def same_batch_tensors(a, b, where=''):
    assert list(a) == list(b), where
    for key in a:
        assert a[key].dtype == b[key].dtype and torch.equal(a[key], b[key]), (where, key)


def test_the_batches_of_the_runner_come_through_the_ordered_prefetch_and_are_those_of_the_plain_stack(inputs, tmp_path):
    """Same order, same contents, bit for bit: the FIT batches arrive framed (the CPU work of the producer, ``batch_prep.prepare_batch``: the coins of 'random_view' applied), the STOP batches as they are."""
    profile = profile_of(tmp_path, **S4B)
    stats = runners.make_stats(profile, inputs)
    train_data = runners.make_train_data(profile, inputs, stats, CPU)
    assert isinstance(train_data.fit, prefetch.Prefetch) and isinstance(train_data.stop, prefetch.Prefetch)
    assert train_data.fit.depth == train_data.stop.depth == prefetch.DEPTH == 2
    fit_plain, stop_plain = train_data.fit.source, train_data.stop.source
    assert isinstance(fit_plain, data.Batches) and isinstance(stop_plain, data.Batches) and fit_plain.swap_mode == 'random_view'
    for epoch in range(3):
        pairs = list(itertools.zip_longest(fit_plain.epoch(epoch), train_data.fit.epoch(epoch)))
        assert len(pairs) == len(fit_plain) == len(train_data.fit) and all(a is not None and b is not None for a, b in pairs)
        for j, (plain, ahead) in enumerate(pairs):
            assert 'swap_view' in plain and 'swap_view' not in ahead
            assert isinstance(ahead, batch_prep.PreparedBatch) and ahead.settings == {'kill_law': profile['kill_law'], 'zone_mark': profile['zone_mark']}
            made = batch_prep.Preparation(kill_law=profile['kill_law'], zone_mark=profile['zone_mark'])(plain)           # the CPU work of item 3 too: the index lists of the plan
            assert any(key.startswith('plan/') for key in ahead)
            same_batch_tensors(made, ahead, f'FIT epoch {epoch} batch {j}')
    stop_pairs = list(itertools.zip_longest(stop_plain.epoch(0), train_data.stop.epoch(0)))
    assert len(stop_pairs) == len(stop_plain) > 0
    for j, (plain, ahead) in enumerate(stop_pairs):
        same_batch_tensors(plain, ahead, f'STOP batch {j}')


def test_the_fit_batches_of_the_runner_are_packed_for_a_device_that_copies_and_the_stop_batches_never(inputs, tmp_path, monkeypatch):
    """Item 4 (S4b speed-ups, D32): the producer of the FIT batches packs them when the preparation says that the device copies (``batch_prep.should_pack``; forced here, the CPU does not pack); nothing else is packed."""
    profile = profile_of(tmp_path, **S4B)
    stats = runners.make_stats(profile, inputs)
    plain = list(runners.make_train_data(profile, inputs, stats, CPU).fit.epoch(0))
    assert plain and all(batch.packed is None for batch in plain)                                       # the CPU: nothing is packed
    monkeypatch.setattr(batch_prep, 'should_pack', lambda device: True)
    train_data = runners.make_train_data(profile, inputs, stats, CPU)
    fit, stop = list(train_data.fit.epoch(0)), list(train_data.stop.epoch(0))
    assert len(fit) == len(plain) and all(isinstance(batch.packed, batch_prep.Packed) for batch in fit)
    for a, b in zip(plain, fit):
        same_batch_tensors(a, b, 'packed FIT batch')                                                    # the same tensors, now views of the buffers
    assert stop and all(not hasattr(batch, 'packed') for batch in stop)


def test_the_cpu_work_of_the_fit_batches_is_done_in_the_producer_thread_and_the_stop_batches_get_none(inputs, tmp_path, monkeypatch):
    profile = profile_of(tmp_path, **S4B)
    stats = runners.make_stats(profile, inputs)
    seen, real = [], batch_prep.prepare_batch
    monkeypatch.setattr(batch_prep, 'prepare_batch', lambda batch: (seen.append(threading.current_thread().name), real(batch))[1])
    train_data = runners.make_train_data(profile, inputs, stats, CPU)
    fit, stop = list(train_data.fit.epoch(0)), list(train_data.stop.epoch(0))
    assert seen == [prefetch.THREAD_NAME] * len(fit) and fit and stop                                    # every FIT batch, in the producer; no STOP batch


def test_the_prefetch_depth_is_a_recorded_run_constant(world, inputs, tmp_path, reps_sha):
    record = runners.prefetch_record()
    assert record['depth'] == prefetch.DEPTH == 2 and record['producers'] == 1 and record['ordered'] is True
    profile = profile_of(tmp_path)
    manifest_path = write_manifest(tmp_path, manifest_document(world, write_scales(tmp_path)))
    manifest, manifest_sha = runners.read_manifest_with_sha(manifest_path)
    header = runners.runner_record(profile, manifest, manifest_sha, runners.RunnerConstants(), ALLOWANCE, {'kill_law': 'poisson', 'k_condition': 'raw', 'k_reps_sha256': None})
    assert header['prefetch'] == record                                                                  # the ledger header says how the batches were made
    ready = with_scales(inputs, write_scales(tmp_path, 'again.json'))
    run_dir = tmp_path / 'run'
    runners.write_stats(profile_of(tmp_path, epochs=1), run_dir, inputs=ready, constants=CPU)
    trained = runners.train(profile_of(tmp_path, epochs=1), run_dir, inputs=ready, constants=CPU, probe=quiet_probe)
    assert trained['prefetch'] == record and run_io.read_json(run_dir / runners.TRAIN_RECORD_FILE)['prefetch'] == record     # and so does the hashed record of the training stage
    assert trained['runner'] == CPU.to_record() and 'prefetch' not in CPU.to_record()                    # the runner constants of the evaluation records are not touched


@pytest.mark.parametrize('error,expected', [(RuntimeError('the trainer failed'), (RuntimeError, 'the trainer failed')), (KeyboardInterrupt(), (KeyboardInterrupt, '')),
                                            (None, (RuntimeError, 'committed checkpoint'))], ids=['error', 'keyboard-interrupt', 'returns-with-an-epoch-left-open'])
def test_the_training_runner_stops_its_producers_when_the_trainer_fails_is_interrupted_or_returns_early(inputs, tmp_path, monkeypatch, error, expected):
    made = []
    real_make = runners.make_train_data
    monkeypatch.setattr(runners, 'make_train_data', lambda *a, **k: made.append(real_make(*a, **k)) or made[-1])

    def trainer(cfg, train_data, out_dir, *, device, epochs, check, seed):
        fit, stop = train_data.fit.epoch(0), train_data.stop.epoch(0)                                     # an epoch of each is running: two producers
        next(fit), next(stop)
        assert len(producer_threads()) == 2
        if error is not None:
            raise error                                                                                    # (the other case returns without a checkpoint and without finishing the epochs)

    monkeypatch.setattr(ept_train, 'train', trainer)
    profile = profile_of(tmp_path, epochs=1)
    ready = with_scales(inputs, write_scales(tmp_path))
    small = runners.RunnerConstants(batch_size=1, device='cpu', threads=1)                                # many batches: both producers are still working (the queue is full) when the trainer fails
    runners.write_stats(profile, tmp_path / 'run', inputs=ready, constants=small)
    with pytest.raises(expected[0], match=expected[1]):
        runners.train(profile, tmp_path / 'run', inputs=ready, constants=small, probe=quiet_probe)
    assert no_producer_left() and len(made) == 1
    assert len(list(made[0].fit.epoch(0))) == len(made[0].fit)                                           # and both wrappers are free again: nothing is left running or locked
    assert no_producer_left()


def test_a_prefetching_training_is_bit_identical_to_the_one_on_plain_batches(inputs, tmp_path, monkeypatch, reps_sha):
    """The proof of item 2 in miniature (the whole matrix is the proof script of the commit): the S4b profile trained through the kernel, once with the prefetching batches and once with the plain
    ``Batches`` underneath (the loop does the CPU work itself then): the checkpoint of every epoch is the same bits (the best weights, the metrics files, and in ``last.pt`` the model, the optimizer
    state and the random-number state, which would differ if the producer thread drew a number from a global stream)."""
    def same_state(a, b, where):
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor) and torch.equal(a, b), where
        elif isinstance(a, dict):
            assert list(a) == list(b), where
            for key in a:
                same_state(a[key], b[key], f'{where}.{key}')
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b), where
            for i, (x, y) in enumerate(zip(a, b)):
                same_state(x, y, f'{where}[{i}]')
        else:
            assert a == b, where

    folders = {}
    real_make = runners.make_train_data
    for name, plain in (('prefetch', False), ('plain', True)):
        if plain:
            monkeypatch.setattr(runners, 'make_train_data', lambda *a, **k: (lambda td: dataclasses.replace(td, fit=td.fit.source, stop=td.stop.source))(real_make(*a, **k)))
        profile = profile_of(tmp_path, f'{name}.json', **{**S4B, 'epochs': 2})
        ready = with_scales(inputs, s4b_scales(tmp_path, reps_sha, f'{name}_scales.json'))
        run_dir = tmp_path / name
        runners.write_stats(profile, run_dir, inputs=ready, constants=CPU)
        runners.train(profile, run_dir, inputs=ready, constants=CPU, probe=quiet_probe)
        folders[name] = sorted((run_dir / 'train' / 'ckpt').glob('epoch_*'))
    assert len(folders['prefetch']) == len(folders['plain']) == 2
    for a, b in zip(folders['prefetch'], folders['plain']):
        for file in ('best.pt', 'epoch_metrics.json', 'epoch_metrics.csv'):
            assert (a / file).exists() == (b / file).exists() and (not (a / file).exists() or (a / file).read_bytes() == (b / file).read_bytes()), (a.name, file)
        last_a, last_b = torch.load(a / 'last.pt', weights_only=True), torch.load(b / 'last.pt', weights_only=True)
        for part in ('model', 'optimizer', 'rng'):
            same_state(last_a[part], last_b[part], f'{a.name}.{part}')


# ---- the loss scales and their condition (D24, plan Q9) ---------------------------------------------------------------------------------------------
def scales_document(**changes) -> dict:
    """A complete, valid ``B0_SCALES.json`` record written by hand (the thirteen blocks, the role FIT, the rule of the module, a full provenance, the units
    of the counts); ``changes`` replace top-level fields (the hash follows ``scales``)."""
    from c_ept.ept import scales
    keys = list(loss.BLOCKS)
    document = {'schema': 'b0_scales_v1', 's3_run': '20261022_c_ept_s3_3', 'role': 'FIT', 'rule': scales.RULE,
                's1': {'run': '20261020_c_ept_s1_1', 'files_sha256': '1' * 64}, 's2': {'run': '20261021_c_ept_s2_3', 'files_sha256': '2' * 64},
                'labels_store_sha256': '3' * 64, 'model_files_sha256': {'fit/b0/b0_heads.json': '4' * 64, 'fit/b0/b0_heads.npz': '5' * 64},
                'scales': dict(SCALES), 'n_rows': {k: 10 for k in keys}, 'n_unscored': {k: 0 for k in keys}, 'by_size': {},
                'n_rows_unit': {k: {'M2Rate': 'frame_row', 'M3Gold': 'team_entry'}.get(k.split('/')[0], 'design_row') for k in keys},
                'n_unscored_unit': {k: {'M2Rate': 'time_bin_cell', 'M3Gold': 'team_entry'}.get(k.split('/')[0], 'design_row') for k in keys},
                'size_normalised': sorted(f'{h}/{k}' for h in loss.SIZE_NORMALISED for k in ('masked', 'open'))}
    document.update(changes)
    document.setdefault('scales_sha256', hashlib.sha256(json.dumps(document['scales'], sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest())
    return document


def write_scales(tmp_path, name='B0_SCALES.json', **changes):
    path = tmp_path / name
    path.write_bytes(json.dumps(scales_document(**changes), indent=1).encode('utf-8'))
    return path


@pytest.fixture(scope='module')
def reps_sha(world):
    return value.gold_reps_sha256(value.fit_gold_bin_means(world.s1, world.labels, population='in_scope').means)


def test_a_scales_file_made_under_the_training_condition_is_accepted(tmp_path, reps_sha):
    plain = write_scales(tmp_path)                                                                    # an old file: no condition field means poisson and raw
    assert runners.check_scales_condition(profile_of(tmp_path), plain) == {'kill_law': 'poisson', 'k_condition': 'raw', 'k_reps_sha256': None}
    binned = write_scales(tmp_path, 'binned.json', kill_law='shifted', k_condition='binned', k_reps_sha256=reps_sha)
    s4b = profile_of(tmp_path, 's4b.json', **S4B)
    condition = {'kill_law': 'shifted', 'k_condition': 'binned', 'k_reps_sha256': reps_sha}
    assert runners.check_scales_condition(s4b, binned, k_reps_sha256=reps_sha) == condition
    assert runners.check_scales_condition(s4b, binned) == condition                                   # the early check does not know the representatives yet: the file must name them


def test_a_scales_file_of_another_condition_is_refused_as_a_refusal(tmp_path, reps_sha):
    from c_ept.run import supervisor
    other = 'b' * 64
    cases = [({}, {'kill_law': 'shifted'}, None, 'kill_law'),
             ({'kill_law': 'shifted'}, {}, None, 'kill_law'),
             ({'k_condition': 'binned', 'k_reps_sha256': reps_sha}, {}, None, 'k_condition'),
             ({}, {'k_condition': 'binned'}, None, 'k_condition'),
             ({'k_condition': 'binned'}, {'k_condition': 'binned'}, None, 'representatives'),                          # a binned file that records no representatives
             ({'k_condition': 'binned', 'k_reps_sha256': other}, {'k_condition': 'binned'}, reps_sha, 'representatives')]       # other representatives than the rows were made of
    for number, (file_fields, training, actual, word) in enumerate(cases):
        path = write_scales(tmp_path, f'case{number}.json', **file_fields)
        with pytest.raises(runners.ScalesConditionRefused, match=word) as raised:
            runners.check_scales_condition(profile_of(tmp_path, f'p{number}.json', **training), path, k_reps_sha256=actual)
        assert supervisor.exit_code_for(raised.value) == 2 and isinstance(raised.value, run_io.IORefused)


def test_a_scales_file_that_is_missing_or_not_valid_is_refused_too(tmp_path):
    profile = profile_of(tmp_path)
    with pytest.raises(runners.ScalesConditionRefused, match='loss-scale file'):
        runners.check_scales_condition(profile, tmp_path / 'absent.json')
    broken = write_scales(tmp_path, 'broken.json', scales_sha256='0' * 64)
    with pytest.raises(runners.ScalesConditionRefused, match='hash'):
        runners.check_scales_condition(profile, broken)


# ---- the inputs manifest -----------------------------------------------------------------------------------------------------------------------
def manifest_document(world, scales_path, **changes) -> dict:
    document = {'schema': 'ept-run-inputs-v1', 's1': str(world.s1.root), 's2': str(world.s2.root), 'labels': str(world.labels.root), 'terrain': None,
                'scales': str(scales_path), 'scales_provenance': None}
    document.update(changes)
    return document


def write_manifest(tmp_path, document, name='INPUTS.json'):
    path = tmp_path / name
    path.write_bytes(json.dumps(document, indent=1).encode('utf-8'))
    return path


def test_the_manifest_names_the_stores_and_the_files_that_the_ledger_binds(world, tmp_path):
    scales_path = write_scales(tmp_path)
    manifest = runners.read_manifest(write_manifest(tmp_path, manifest_document(world, scales_path)))
    bound = runners.manifest_files(manifest)
    assert bound == sorted(bound) and len(bound) == 4
    assert {p.rsplit('/', 1)[-1] for p in bound} == {'STORE.json', 'B0_SCALES.json'} and sum(p.endswith('/STORE.json') for p in bound) == 3
    assert all(p.startswith(('C:/', 'c:/')) or p.startswith('/') for p in bound) and '\\' not in ''.join(bound)       # absolute, forward slashes: the ledger's input form
    with_terrain = runners.manifest_files({**manifest, 'terrain': str(tmp_path / 'terrain')})
    assert len(with_terrain) == 7 and {p.rsplit('/', 1)[-1] for p in with_terrain} >= {'TERRAIN.json', 'terrain.npz', 'geodesic.npy'}


@pytest.mark.parametrize('change, word', [({'schema': 'other'}, 'schema'), ({'extra': 1}, 'fields'), ({'s1': 'relative/path'}, 's1'), ({'s2': 7}, 's2'),
                                          ({'labels': ''}, 'labels'), ({'scales': 'a\x00b'}, 'scales'), ({'terrain': 5}, 'terrain'),
                                          ({'scales_provenance': [1]}, 'scales_provenance')])
def test_a_manifest_that_is_not_acceptable_is_refused(world, tmp_path, change, word):
    scales_path = write_scales(tmp_path)
    with pytest.raises(run_io.IORefused, match=word):
        runners.read_manifest(write_manifest(tmp_path, manifest_document(world, scales_path, **change)))
    document = manifest_document(world, scales_path)
    del document['labels']
    with pytest.raises(run_io.IORefused, match='fields'):
        runners.read_manifest(write_manifest(tmp_path, document, 'missing.json'))


def test_the_terrain_must_be_the_terrain_that_built_the_s2_views():
    """The static channels and the geodesic bias come from the terrain folder as it is now; the S2 views were made with the terrain of the build (A5-F10)."""
    from c_ept import runlog
    hashes = {'TERRAIN.json': 'a' * 64, 'terrain.npz': 'b' * 64, 'geodesic.npy': 'c' * 64}
    built = runlog.sha256_json(hashes)
    runners.check_terrain_build({'stages': {'views': {'terrain_sha256': built}}}, hashes)
    for manifest in ({}, {'stages': {}}, {'stages': {'views': {}}}, {'stages': {'views': {'terrain_sha256': 'nothex'}}}, {'stages': {'views': {'terrain_sha256': 'd' * 64}}},
                     {'stages': {'views': 5}}):
        with pytest.raises(run_io.IORefused, match='terrain'):
            runners.check_terrain_build(manifest, hashes)


def test_the_real_loader_needs_a_terrain_and_checks_it_against_the_store(world, tmp_path):
    scales_path = write_scales(tmp_path)
    with pytest.raises(run_io.IORefused, match='terrain'):
        runners.load_inputs(write_manifest(tmp_path, manifest_document(world, scales_path)))           # the toy manifest has no terrain
    folder = tmp_path / 'terrain'
    folder.mkdir()
    for name in runners.TERRAIN_FILES:
        (folder / name).write_bytes(b'x')
    with pytest.raises(run_io.IORefused, match='terrain'):
        runners.load_inputs(write_manifest(tmp_path, manifest_document(world, scales_path, terrain=str(folder)), 'with_terrain.json'))     # the toy S2 records no terrain hash


def test_the_scales_are_read_with_the_expected_provenance_when_the_manifest_gives_one(world, tmp_path):
    scales_path = write_scales(tmp_path)
    document = scales_document()
    provenance = {name: document[name] for name in ('s3_run', 's1', 's2', 'labels_store_sha256', 'model_files_sha256')}
    manifest = runners.read_manifest(write_manifest(tmp_path, manifest_document(world, scales_path, scales_provenance=provenance)))
    values, path = runners.read_scales(manifest)
    assert values == SCALES and path == scales_path
    wrong = runners.read_manifest(write_manifest(tmp_path, manifest_document(world, scales_path, scales_provenance={**provenance, 's3_run': 'another'}), 'wrong.json'))
    with pytest.raises(runners.ScalesConditionRefused, match='expected'):
        runners.read_scales(wrong)


# ---- the statistics stage and the training (in this process, the way the stage bodies run them) -------------------------------------------------
E6_RECORD = {'running': False, 'matched': [], 'bridges_ignored': 0, 'scanned': 5, 'cmdline_unreadable': 0, 'rule': 'test rule'}


@pytest.fixture(autouse=True)
def restore_torch_state(monkeypatch):
    """``train`` changes process-wide settings (threads, deterministic algorithms, the global random streams, and with K6c the numeric policy: matmul precision, TF32, the cuDNN flags,
    ``fill_uninitialized_memory`` and ``CUBLAS_WORKSPACE_CONFIG``); give them back after each test."""
    from c_ept.tests.test_run_numeric import read_flags, write_flags
    threads, flags = torch.get_num_threads(), read_flags()
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8'))                # monkeypatch puts back what was there, or nothing
    yield
    torch.set_num_threads(threads)
    write_flags(flags)


def quiet_probe():
    return dict(E6_RECORD)


def with_scales(inputs, path):
    return dataclasses.replace(inputs, scales_path=path)


def s4b_scales(tmp_path, reps_sha, name='s4b_scales.json'):
    return write_scales(tmp_path, name, kill_law='shifted', k_condition='binned', k_reps_sha256=reps_sha)


def test_the_statistics_stage_writes_the_statistics_and_an_explicit_record(world, inputs, tmp_path, reps_sha):
    profile = profile_of(tmp_path, **S4B)
    scales_path = s4b_scales(tmp_path, reps_sha)
    run_dir = tmp_path / 'run'
    record = runners.write_stats(profile, run_dir, inputs=with_scales(inputs, scales_path), constants=CPU)
    stats = data.Stats.load(run_dir / runners.STATS_FILE)
    on_disk = run_io.read_json(run_dir / runners.STATS_RECORD_FILE)
    assert on_disk == record and record['schema'] == runners.STATS_RECORD_SCHEMA
    assert record['profile_sha256'] == profile.sha256 and record['settings'] == runners.effective_settings(profile) and record['runner'] == CPU.to_record()
    assert record['stats'] == {'sha256': stats.sha256, 'n_views': stats.n_views, 'population': 'selected_fit', 'column_drop': 'store_and_subset'}
    d7 = list(stats.dropped_store)
    assert record['d7'] == {'columns': d7, 'sha256': runlog.sha256_json(d7)} and d7 == ['t_s', 'rift_element_known', 'rift_element_air', 'flag_stasis']
    assert record['dropped'] == {'columns': list(stats.dropped), 'sha256': runlog.sha256_json(list(stats.dropped))}
    reps = value.fit_gold_bin_means(world.s1, world.labels, population='in_scope')
    assert record['k_reps'] == {'sha256': reps_sha, 'fallback': [int(b) for b in reps.fallback]}
    assert record['scales'] == {'file_sha256': hashlib.sha256(scales_path.read_bytes()).hexdigest(),
                                'condition': {'kill_law': 'shifted', 'k_condition': 'binned', 'k_reps_sha256': reps_sha}}
    selected = runners.select_matches(profile, world.s1)
    assert record['membership'] == {'seed': 20261022, 'fraction': 0.25, 'matches': {'FIT': len(selected['FIT']), 'STOP': len(selected['STOP'])},
                                    'selected_sha256': runlog.sha256_json({role: [int(m) for m in selected[role]] for role in ('FIT', 'STOP')})}


def test_the_statistics_record_of_the_default_profile_names_no_list_and_no_representatives(inputs, tmp_path):
    scales_path = write_scales(tmp_path)
    record = runners.write_stats(profile_of(tmp_path), tmp_path / 'run', inputs=with_scales(inputs, scales_path), constants=CPU)
    assert record['d7'] is None and record['k_reps'] is None and record['dropped'] == {'columns': [], 'sha256': runlog.sha256_json([])}
    assert record['stats']['population'] == 'whole_fit' and record['stats']['column_drop'] == 'none'
    assert record['scales']['condition'] == {'kill_law': 'poisson', 'k_condition': 'raw', 'k_reps_sha256': None}


def test_the_statistics_stage_refuses_scales_of_another_condition_before_it_writes_anything(inputs, tmp_path, reps_sha):
    plain = write_scales(tmp_path)                                                                    # the scales of the plain law and the raw gold
    run_dir = tmp_path / 'run'
    with pytest.raises(runners.ScalesConditionRefused, match='kill_law'):
        runners.write_stats(profile_of(tmp_path, **S4B), run_dir, inputs=with_scales(inputs, plain), constants=CPU)
    other = s4b_scales(tmp_path, 'f' * 64, 'other.json')                                              # the right condition but other representatives than the rows have
    with pytest.raises(runners.ScalesConditionRefused, match='representatives'):
        runners.write_stats(profile_of(tmp_path, 'again.json', **S4B), run_dir, inputs=with_scales(inputs, other), constants=CPU)
    assert not run_dir.exists()
    with pytest.raises(runners.ScalesConditionRefused, match='no scales file'):
        runners.write_stats(profile_of(tmp_path), run_dir, inputs=inputs, constants=CPU)               # no file, no condition to check: a refusal, not a pass
    assert not run_dir.exists()


def test_the_s4b_profile_trains_on_the_toy_world_and_the_record_holds_every_effective_value(world, inputs, tmp_path, reps_sha):
    profile = profile_of(tmp_path, **S4B)
    scales_path = s4b_scales(tmp_path, reps_sha)
    ready = with_scales(inputs, scales_path)
    run_dir = tmp_path / 'run'
    runners.write_stats(profile, run_dir, inputs=ready, constants=CPU)
    verified = []
    record = runners.train(profile, run_dir, inputs=ready, constants=CPU, probe=quiet_probe, verifier=lambda: verified.append(1))
    assert len(verified) == 2 and record['effective_threads'] == ept_train.THREADS != CPU.threads                  # on the CPU the trainer forces its own thread count (m3)
    assert run_io.read_json(run_dir / runners.TRAIN_RECORD_FILE) == record and record['schema'] == runners.TRAIN_RECORD_SCHEMA
    assert record['profile_sha256'] == profile.sha256 and record['runner'] == CPU.to_record() and record['settings'] == runners.effective_settings(profile)
    assert record['identity'] == {'kill_law': 'shifted', 'k_condition': 'binned', 'decay_groups': 'split', 'terrain_smooth_weight': 0.0, 'swap_mode': 'random_view',
                                  'eval_average': True, 'on_wall_cap': 'best_stop', 'zone_mark': True, 'column_drop': 'store_and_subset', 'stats_population': 'selected_fit'}
    stats = data.Stats.load(run_dir / runners.STATS_FILE)
    assert record['stats'] == {'sha256': stats.sha256, 'dropped_store_sha256': runlog.sha256_json(list(stats.dropped_store))}
    assert record['k_reps']['sha256'] == reps_sha and record['scales']['condition']['k_reps_sha256'] == reps_sha
    assert record['decay_groups']['policy'] == 'split' and record['decay_groups']['rule_sha256'] == ept_train.decay_rule_sha256() and record['decay_groups']['head']
    assert record['termination'] == {'reason': 'max_epochs', 'done': True, 'epochs_run': 2, 'best_epoch': record['termination']['best_epoch'], 'stopped_early': False,
                                     'budget_end': False}
    assert record['time']['cap_seconds'] == ept_train.WALL_CAP_SECONDS and 0 < record['time']['training_seconds'] < 3600 and record['time']['over_cap'] is False
    assert record['config']['encoder'] == dataclasses.asdict(runners.encoder_config(profile)) and record['config']['lr'] == 0.001
    files = record['checkpoint']['files']
    assert files and all(hashlib.sha256((run_dir / name).read_bytes()).hexdigest() == digest for name, digest in files.items())
    assert 'train/ckpt/epoch_0002/head.json' in files and record['checkpoint']['final'] == 'train/ckpt/epoch_0002'
    e6 = run_io.read_json(run_dir / runners.E6_FILE)
    assert e6['schema'] == runners.EPOCH_CHECKS_SCHEMA
    assert e6['checks'] == [{'call': 0, 'identity': 'verified', 'e6': E6_RECORD}, {'call': 1, 'identity': 'verified', 'e6': E6_RECORD}]       # one check for each epoch of the call
    assert record['e6'] == {'file': runners.E6_FILE, 'checks': 2}


def test_the_default_profile_trains_and_its_record_names_the_old_defaults_explicitly(inputs, tmp_path):
    profile = profile_of(tmp_path, epochs=1)
    ready = with_scales(inputs, write_scales(tmp_path))
    run_dir = tmp_path / 'run'
    runners.write_stats(profile, run_dir, inputs=ready, constants=CPU)
    record = runners.train(profile, run_dir, inputs=ready, constants=CPU, probe=quiet_probe)
    assert record['identity'] == {'kill_law': 'poisson', 'k_condition': 'raw', 'decay_groups': 'all', 'terrain_smooth_weight': 0.0, 'swap_mode': 'average',
                                  'eval_average': True, 'on_wall_cap': 'fail', 'zone_mark': True, 'column_drop': 'none', 'stats_population': 'whole_fit'}
    assert record['k_reps'] is None and record['decay_groups'] == {'policy': 'all', 'rule_sha256': None, 'head': None}
    assert record['termination']['reason'] == 'max_epochs' and record['stats']['dropped_store_sha256'] == runlog.sha256_json([])


def test_the_training_sets_the_numeric_policy_before_the_trainer_builds_a_model_and_records_the_state(inputs, tmp_path, monkeypatch):
    """D10 P6 (K6c): a process whose settings are not the policy is put under it before ``ept.train.train`` builds the model, and the record of the stage holds the state that was read back."""
    from c_ept.tests.test_run_numeric import OPPOSITE, read_flags, write_flags
    profile = profile_of(tmp_path, epochs=1)
    ready = with_scales(inputs, write_scales(tmp_path))
    run_dir = tmp_path / 'run'
    runners.write_stats(profile, run_dir, inputs=ready, constants=CPU)
    write_flags(OPPOSITE)
    seen = []
    real = ept_train.train

    def spy(*args, **kwargs):
        seen.append(read_flags())
        return real(*args, **kwargs)
    monkeypatch.setattr(ept_train, 'train', spy)
    record = runners.train(profile, run_dir, inputs=ready, constants=CPU, probe=quiet_probe)
    assert seen == [{name: run_numeric.REQUIRED[name] for name in OPPOSITE}]                     # the policy was in force when the trainer was called
    numeric = record['numeric']
    assert numeric['requested'] == dict(run_numeric.REQUIRED) and numeric['requested_sha256'] == run_numeric.requested_record()['requested_sha256']
    assert all(numeric['measured'][name] == run_numeric.REQUIRED[name] for name in run_numeric.REQUIRED) and numeric['measured']['device'] == 'cpu'
    assert numeric['measured_sha256'] == run_io.canonical_sha256(numeric['measured']) and run_io.read_json(run_dir / runners.TRAIN_RECORD_FILE)['numeric'] == numeric
    assert torch.cuda.is_initialized() is False                                                  # a CPU run never creates a CUDA context


def test_a_state_that_is_not_the_policy_refuses_the_training_before_any_row_is_built(inputs, tmp_path, monkeypatch):
    from c_ept.run import supervisor
    profile = profile_of(tmp_path, epochs=1)
    ready = with_scales(inputs, write_scales(tmp_path))
    run_dir = tmp_path / 'run'
    runners.write_stats(profile, run_dir, inputs=ready, constants=CPU)

    def never(*args, **kwargs):
        raise AssertionError('data or a model was built before the numeric policy was held')
    monkeypatch.setattr(data, 'ept_head_rows', never)
    monkeypatch.setattr(ept_train, 'train', never)
    torch.set_default_dtype(torch.float64)
    try:
        with pytest.raises(run_numeric.NumericRefused, match='default_dtype') as raised:
            runners.train(profile, run_dir, inputs=ready, constants=CPU, probe=quiet_probe)
    finally:
        torch.set_default_dtype(torch.float32)
    assert supervisor.exit_code_for(raised.value) == 2 and not (run_dir / 'train').exists()


class Clock:
    """The real clock plus an offset that a test moves: no time is actually spent."""

    def __init__(self, monkeypatch):
        self.bump = 0.0
        monkeypatch.setattr(ept_train, '_clock', lambda: time.perf_counter() + self.bump)


def test_the_wall_cap_with_best_stop_ends_in_budget_and_the_runner_counts_it_as_a_success(inputs, tmp_path, monkeypatch, reps_sha):
    """The clock passes the 24 h cap while the second epoch trains: that epoch is finished, the run ends with ``termination_reason`` 'budget' (the head key
    ``budget_end``), and the runner returns its record like any other end; the record holds the training seconds against the cap."""
    clock = Clock(monkeypatch)
    calls = []

    def probe():
        calls.append(1)
        if len(calls) == 2:
            clock.bump += 90_000.0
        return dict(E6_RECORD)
    profile = profile_of(tmp_path, **{**S4B, 'epochs': 3})
    ready = with_scales(inputs, s4b_scales(tmp_path, reps_sha))
    run_dir = tmp_path / 'run'
    runners.write_stats(profile, run_dir, inputs=ready, constants=CPU)
    record = runners.train(profile, run_dir, inputs=ready, constants=CPU, probe=probe)
    assert record['termination'] == {'reason': 'budget', 'done': True, 'epochs_run': 2, 'best_epoch': record['termination']['best_epoch'], 'stopped_early': False,
                                     'budget_end': True}
    assert record['time']['over_cap'] is True and record['time']['training_seconds'] >= 90_000.0 and record['time']['cap_seconds'] == ept_train.WALL_CAP_SECONDS
    assert record['identity']['on_wall_cap'] == 'best_stop' and record['checkpoint']['final'] == 'train/ckpt/epoch_0002'


def test_a_refusing_e6_probe_stops_the_training_and_is_written_down(inputs, tmp_path, reps_sha):
    from c_ept.run import guards, supervisor
    calls = []

    def probe():
        calls.append(1)
        if len(calls) == 2:
            raise guards.GateRefused('e6', 'a model server is running (test)')
        return dict(E6_RECORD)
    profile = profile_of(tmp_path, **{**S4B, 'epochs': 3})
    ready = with_scales(inputs, s4b_scales(tmp_path, reps_sha))
    run_dir = tmp_path / 'run'
    runners.write_stats(profile, run_dir, inputs=ready, constants=CPU)
    with pytest.raises(guards.GateRefused, match='model server') as raised:
        runners.train(profile, run_dir, inputs=ready, constants=CPU, probe=probe)
    assert supervisor.exit_code_for(raised.value) == 2
    checks = run_io.read_json(run_dir / runners.E6_FILE)['checks']
    assert [sorted(c) for c in checks] == [['call', 'e6', 'identity'], ['call', 'identity', 'refused']] and 'model server' in checks[1]['refused']
    assert [c['identity'] for c in checks] == ['not_checked', 'not_checked']                         # no verifier given: the record says so, it does not claim a check
    assert not (run_dir / runners.TRAIN_RECORD_FILE).exists()                                         # a record is written for a finished training only


def test_the_identity_of_the_imported_files_is_verified_before_e6_at_every_epoch_and_a_change_stops_the_training(inputs, tmp_path, reps_sha):
    """D11: each epoch re-checks the hashes of the imported module files. The verifier runs before E6; a change is written down and raised (exit code 2)."""
    from c_ept.run import identity as run_identity, supervisor
    order, calls = [], []

    def probe():
        order.append('e6')
        return dict(E6_RECORD)

    def verifier():
        order.append('identity')
        calls.append(1)
        if len(calls) == 2:
            raise run_identity.IdentityChanged('imported code changed since the identity was bound', ['c_ept/run/runners.py'])
    profile = profile_of(tmp_path, **{**S4B, 'epochs': 3})
    ready = with_scales(inputs, s4b_scales(tmp_path, reps_sha))
    run_dir = tmp_path / 'run'
    runners.write_stats(profile, run_dir, inputs=ready, constants=CPU)
    with pytest.raises(run_identity.IdentityChanged, match='imported code changed') as raised:
        runners.train(profile, run_dir, inputs=ready, constants=CPU, probe=probe, verifier=verifier)
    assert supervisor.exit_code_for(raised.value) == 2 and order == ['identity', 'e6', 'identity']                # the verifier first, E6 not asked after a change
    checks = run_io.read_json(run_dir / runners.E6_FILE)['checks']
    assert checks[0] == {'call': 0, 'identity': 'verified', 'e6': E6_RECORD} and checks[1]['identity'] == 'refused' and 'imported code changed' in checks[1]['refused']
    assert not (run_dir / runners.TRAIN_RECORD_FILE).exists()
    order.clear()
    paused = profile_of(tmp_path, 'paused.json', **{**S4B, 'epochs': 3})
    run_dir = tmp_path / 'run_paused'
    runners.write_stats(paused, run_dir, inputs=ready, constants=CPU)

    def cannot_read():
        raise run_identity.Paused('imported files cannot be read now')
    with pytest.raises(run_identity.Paused) as raised:
        runners.train(paused, run_dir, inputs=ready, constants=CPU, probe=probe, verifier=cannot_read)
    assert supervisor.exit_code_for(raised.value) == 2 and order == []
    assert run_io.read_json(run_dir / runners.E6_FILE)['checks'][0]['identity'] == 'refused'


def test_a_second_call_on_a_finished_training_continues_the_e6_list_and_writes_the_same_record(inputs, tmp_path, reps_sha):
    """The supervisor runs the training stage again when the child was cut after the trainer's last checkpoint and before the ledger finished the stage: the
    trainer returns the finished run without training, the E6 list goes on (one check for the call) and the record says the same about the training."""
    profile = profile_of(tmp_path, **S4B)
    ready = with_scales(inputs, s4b_scales(tmp_path, reps_sha))
    run_dir = tmp_path / 'run'
    runners.write_stats(profile, run_dir, inputs=ready, constants=CPU)
    first = runners.train(profile, run_dir, inputs=ready, constants=CPU, probe=quiet_probe)
    second = runners.train(profile, run_dir, inputs=ready, constants=CPU, probe=quiet_probe)
    assert len(run_io.read_json(run_dir / runners.E6_FILE)['checks']) == 3 and second['e6']['checks'] == 3 and first['e6']['checks'] == 2
    for name in ('termination', 'time', 'checkpoint', 'identity', 'k_reps', 'stats', 'scales'):
        assert second[name] == first[name], name


def _train_twice_with_second_state(inputs, tmp_path, reps_sha, monkeypatch, change):
    """Train, then call the stage again (a resume) in a process whose measured numeric state is the first one's with ``change`` applied. Return ``(first record, second call: record or exception)``."""
    profile = profile_of(tmp_path, **S4B)
    ready = with_scales(inputs, s4b_scales(tmp_path, reps_sha))
    run_dir = tmp_path / 'run'
    runners.write_stats(profile, run_dir, inputs=ready, constants=CPU)
    first = runners.train(profile, run_dir, inputs=ready, constants=CPU, probe=quiet_probe)
    real = run_numeric.apply

    def changed(device):
        state = real(device)
        measured = {**state['measured'], **change}
        return {**state, 'measured': measured, 'measured_sha256': run_io.canonical_sha256(measured)}
    monkeypatch.setattr(run_numeric, 'apply', changed)
    try:
        second = runners.train(profile, run_dir, inputs=ready, constants=CPU, probe=quiet_probe)
    except Exception as refusal:                                                                     # noqa: BLE001 - the test looks at which refusal it was
        second = refusal
    return first, second, run_dir


def test_a_resumed_training_records_the_numeric_state_of_every_process_and_may_change_the_thread_count_only(inputs, tmp_path, reps_sha, monkeypatch):
    """m1 of the K6c review: the E6 file keeps one measured state per process of the run, the record names them, and a different thread count (the trainer sets its own) is no reason to refuse."""
    first, second, run_dir = _train_twice_with_second_state(inputs, tmp_path, reps_sha, monkeypatch, {'threads': 3})
    assert isinstance(second, dict), second
    assert [a['attempt'] for a in first['numeric_attempts']] == [0] and [a['attempt'] for a in second['numeric_attempts']] == [0, 1]
    assert second['numeric_attempts'][0] == first['numeric_attempts'][0] and first['numeric_attempts'][0]['measured_sha256'] == first['numeric']['measured_sha256']
    assert second['numeric_attempts'][1]['measured_sha256'] == second['numeric']['measured_sha256'] != first['numeric']['measured_sha256']
    kept = run_io.read_json(run_dir / runners.E6_FILE)['numeric_attempts']
    assert [a['measured'] for a in kept] == [first['numeric']['measured'], second['numeric']['measured']] and kept[1]['measured']['threads'] == 3


@pytest.mark.parametrize('change', [{'torch_version': '0.0.0'}, {'device_name': 'another GPU'}, {'cuda_version': '0.0'}, {'sdp_backends': {'flash': False}}, {'cudnn_version': 1}])
def test_a_training_resumed_under_another_torch_device_or_attention_state_is_refused_with_exit_2(inputs, tmp_path, reps_sha, monkeypatch, change):
    from c_ept.run import supervisor
    first, second, run_dir = _train_twice_with_second_state(inputs, tmp_path, reps_sha, monkeypatch, change)
    assert isinstance(second, run_numeric.NumericRefused) and supervisor.exit_code_for(second) == 2
    assert next(iter(change)) in str(second) and 'resumed' in str(second)
    assert len(run_io.read_json(run_dir / runners.E6_FILE)['numeric_attempts']) == 1                  # the refused process left no attempt behind
    assert run_io.read_json(run_dir / runners.TRAIN_RECORD_FILE) == first                           # and the record of the finished run is as it was


def test_the_numeric_attempts_are_not_in_a_check_file_that_no_numeric_state_wrote(tmp_path):
    """A bare ``EpochCheck`` (no numeric state given) writes the old file: no list is invented."""
    watcher = runners.EpochCheck(tmp_path / 'E6.json', quiet_probe)
    watcher()
    assert 'numeric_attempts' not in run_io.read_json(tmp_path / 'E6.json') and watcher.numeric_attempts == []


def test_a_cuda_run_without_cuda_is_refused_before_any_work(inputs, tmp_path, reps_sha, monkeypatch):
    from c_ept.run import guards
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)                                    # whatever the machine has: this test never uses a GPU
    profile = profile_of(tmp_path, **S4B)
    ready = with_scales(inputs, s4b_scales(tmp_path, reps_sha))
    run_dir = tmp_path / 'run'
    runners.write_stats(profile, run_dir, inputs=ready, constants=CPU)

    def no_rows(*args, **kwargs):
        raise AssertionError('rows were built before the CUDA check')
    monkeypatch.setattr(data, 'ept_head_rows', no_rows)                                               # the check comes first: no row, no statistics file, no scales file is read
    with pytest.raises(guards.GateRefused, match='CUDA'):
        runners.train(profile, run_dir, inputs=ready, constants=runners.RunnerConstants(), probe=quiet_probe)
    assert not (run_dir / 'train').exists()


def test_a_cuda_run_without_cuda_is_exit_2_before_the_reservation(rig, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert rig.start(constants=runners.RunnerConstants()) == 2
    assert not rig.run_dir.exists() and 'CUDA' in rig.stream.getvalue()


def test_the_statistics_stage_builds_no_rows_for_the_plain_law_and_the_raw_gold(inputs, tmp_path, monkeypatch):
    """The representatives exist only under the binned condition: the stage does not build the whole FIT rows to find out that there are none."""
    def no_rows(*args, **kwargs):
        raise AssertionError('rows were built')
    monkeypatch.setattr(data, 'ept_head_rows', no_rows)
    record = runners.write_stats(profile_of(tmp_path), tmp_path / 'run', inputs=with_scales(inputs, write_scales(tmp_path)), constants=CPU)
    assert record['k_reps'] is None


def test_the_identity_defaults_are_the_defaults_of_the_profile_fields(tmp_path):
    """The cross-check at the end of a training compares the trainer's identity with the profile; a table that drifted from the profile would show only after
    the training. The names are profile fields and the defaults are those of the profile."""
    default = profile_of(tmp_path)
    assert dict(runners.IDENTITY_DEFAULTS) == {name: default[name] for name in runners.IDENTITY_DEFAULTS}


def test_the_training_needs_the_statistics_of_the_statistics_stage(inputs, tmp_path, reps_sha):
    profile = profile_of(tmp_path, **S4B)
    ready = with_scales(inputs, s4b_scales(tmp_path, reps_sha))
    with pytest.raises(run_io.IORefused, match='statistics'):
        runners.train(profile, tmp_path / 'run', inputs=ready, constants=CPU, probe=quiet_probe)


def test_the_training_refuses_statistics_that_were_made_for_another_setting(inputs, tmp_path, reps_sha):
    """The statistics file of the stage before is hash-bound by the ledger, but a file from a run with another ``column_drop`` must not be trained on either."""
    ready = with_scales(inputs, s4b_scales(tmp_path, reps_sha))
    run_dir = tmp_path / 'run'
    runners.write_stats(profile_of(tmp_path, **S4B), run_dir, inputs=ready, constants=CPU)
    other = profile_of(tmp_path, 'other.json', **{**S4B, 'column_drop': 'store'})
    with pytest.raises(ValueError, match='column_drop'):
        runners.train(other, run_dir, inputs=ready, constants=CPU, probe=quiet_probe)


# ---- the plan of the supervisor ---------------------------------------------------------------------------------------------------------------
ALLOWANCE = runners.CapAllowance(epoch_s=1800.0, checkpoint_write_s=120.0, finalisation_s=300.0)
TOY_LOADER = 'c_ept.tests.run_runner_bodies:toy_inputs'
TRAIN_BODY = 'c_ept.tests.run_runner_bodies:train_stage'
KEY = 'KR_1234567890'                                                 # a match key: <platform>_<game id>
REPO = Path(__file__).resolve().parents[2]


def test_the_cap_allowance_is_declared_and_summed_for_the_m8_hand_off():
    assert ALLOWANCE.total() == 1800.0 + 2 * 120.0 + 300.0 and ALLOWANCE.to_record() == {'epoch_s': 1800.0, 'checkpoint_write_s': 120.0, 'finalisation_s': 300.0}
    for bad in ({'epoch_s': -1.0}, {'epoch_s': float('nan')}, {'checkpoint_write_s': float('inf')}, {'finalisation_s': True}, {'epoch_s': '5'}):
        with pytest.raises(run_io.IORefused):
            runners.CapAllowance(**{'epoch_s': 1.0, 'checkpoint_write_s': 1.0, 'finalisation_s': 1.0, **bad})
    with pytest.raises(TypeError):
        runners.CapAllowance()                                                                         # declared, never defaulted: the numbers come from the measurement P6


def test_the_training_stage_declares_the_cap_of_the_trainer_plus_the_epoch_two_writes_and_the_finalisation(tmp_path):
    best_stop = profile_of(tmp_path, **S4B)
    fail = profile_of(tmp_path, 'fail.json')
    assert runners.training_stage_cap(best_stop, ALLOWANCE) == ept_train.WALL_CAP_SECONDS + 1800.0 + 2 * 120.0 + 300.0
    assert runners.training_stage_cap(fail, ALLOWANCE) == ept_train.WALL_CAP_SECONDS                   # under 'fail' the trainer never goes past the cap: no allowance


def test_the_plan_has_a_statistics_stage_and_a_training_stage_with_text_arguments_and_bound_inputs(world, tmp_path, reps_sha):
    profile_path = tmp_path / 'profile.json'
    profile_path.write_bytes((json.dumps({'kind': 'train', **S4B}, indent=2) + '\n').encode('utf-8'))
    profile = run_profile.load(profile_path)
    scales_path = s4b_scales(tmp_path, reps_sha)
    manifest_path = write_manifest(tmp_path, manifest_document(world, scales_path))
    stages = runners.plan(profile, profile_path, manifest_path, constants=CPU, allowance=ALLOWANCE)
    assert [stage.name for stage in stages] == ['stats', 'train']
    stats, training = stages
    assert stats.body == 'c_ept.run.runners:stats_stage' and training.body == 'c_ept.run.runners:train_stage' and stats.wall_cap_s is None
    assert training.wall_cap_s == runners.training_stage_cap(profile, ALLOWANCE)
    bound = runners.manifest_files(runners.read_manifest(manifest_path))
    expected_args = (str(profile_path), profile.sha256, str(manifest_path), runners.LOADER, 'cpu', '1', '32')
    assert stats.args == training.args == expected_args and all(isinstance(a, str) for a in stats.args)
    assert sorted(stats.inputs) == sorted([profile_path.as_posix(), manifest_path.as_posix(), *bound])
    assert sorted(training.inputs) == sorted([runners.STATS_FILE, runners.STATS_RECORD_FILE, profile_path.as_posix(), manifest_path.as_posix(), *bound])
    assert stats.outputs == (runners.STATS_FILE, runners.STATS_RECORD_FILE) and training.outputs == (runners.TRAIN_RECORD_FILE, runners.E6_FILE)
    assert all(stage.inputs and stage.outputs for stage in stages)                                     # real, non-empty inputs and outputs (K5a hand-off)
    swapped = runners.plan(profile, profile_path, manifest_path, constants=CPU, allowance=ALLOWANCE, loader='x.y:z', bodies={'train': 'x.y:w'})
    assert swapped[0].body == 'c_ept.run.runners:stats_stage' and swapped[1].body == 'x.y:w' and swapped[0].args[3] == 'x.y:z'


def test_the_public_entry_points_take_the_profile_and_the_run_folder_first_and_keep_everything_else_by_keyword():
    keyword_only = inspect.Parameter.KEYWORD_ONLY
    parameters = inspect.signature(runners.train).parameters
    assert list(parameters)[:2] == ['profile', 'run_dir'] and all(p.kind is keyword_only for p in list(parameters.values())[2:])
    assert parameters['inputs'].default is inspect.Parameter.empty and parameters['probe'].default is None                      # the stores cannot come from a profile
    assert parameters['verifier'].default is None
    parameters = inspect.signature(runners.start).parameters
    assert list(parameters)[:2] == ['profile_path', 'run_dir'] and all(p.kind in (keyword_only, inspect.Parameter.VAR_KEYWORD) for p in list(parameters.values())[2:])
    assert all(parameters[name].default is inspect.Parameter.empty for name in ('manifest_path', 'spec_path', 'budget', 'allowance'))


def test_the_header_record_of_the_default_constants_is_the_t002_gpu_run(world, tmp_path, reps_sha):
    profile = profile_of(tmp_path)
    manifest_path = write_manifest(tmp_path, manifest_document(world, write_scales(tmp_path)))
    manifest, manifest_sha = runners.read_manifest_with_sha(manifest_path)
    record = runners.runner_record(profile, manifest, manifest_sha, runners.RunnerConstants(), ALLOWANCE, {'kill_law': 'poisson', 'k_condition': 'raw', 'k_reps_sha256': None})
    assert (record['batch_size'], record['device'], record['threads'], record['cuda_visible_devices']) == (32, 'cuda', 8, '0')
    assert 'kernel' not in record and record['inputs']['manifest_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert record['training_stage']['wall_cap_s'] == ept_train.WALL_CAP_SECONDS == record['training_stage']['trainer_cap_s']             # on_wall_cap 'fail': no allowance


def test_the_stage_bodies_are_importable_functions_of_a_run_folder_and_seven_texts():
    for body in (runners.stats_stage, runners.train_stage):
        parameters = list(inspect.signature(body).parameters)
        assert parameters == ['run_dir', 'profile_path', 'profile_sha256', 'manifest_path', 'loader', 'device', 'threads', 'batch_size']
    assert resolve(runners.BODIES['stats']) is runners.stats_stage and resolve(runners.BODIES['train']) is runners.train_stage
    assert resolve(runners.LOADER) is runners.load_inputs


def test_a_stage_body_checks_the_lock_of_the_profile_and_its_text_arguments_before_it_does_anything(world, tmp_path, reps_sha):
    profile_path = tmp_path / 'profile.json'
    profile_path.write_bytes((json.dumps({'kind': 'train', **S4B}, indent=2) + '\n').encode('utf-8'))
    profile = run_profile.load(profile_path)
    manifest_path = write_manifest(tmp_path, manifest_document(world, s4b_scales(tmp_path, reps_sha)))
    run_dir = tmp_path / 'run'
    good = (str(profile_path), profile.sha256, str(manifest_path), TOY_LOADER, 'cpu', '1', '32')
    for number, change in enumerate(({1: 'f' * 64}, {1: 'nothash'}, {4: 'tpu'}, {5: 'x'}, {5: '0'}, {5: '1.5'}, {6: '-3'}, {6: ''}, {3: 'not a loader'}, {3: 'c_ept.run.runners:missing'})):
        arguments = tuple(change.get(i, text) for i, text in enumerate(good))
        with pytest.raises(run_io.IORefused):
            runners.stats_stage(run_dir, *arguments)
    assert not run_dir.exists()
    other = tmp_path / 'evaluate.json'
    other.write_bytes(b'{"kind": "measure", "train_profile": "profile.json", "train_sha256": "' + profile.sha256.encode() + b'", "steps": 3}\n')
    with pytest.raises(run_io.IORefused, match='train profile'):
        runners.stats_stage(run_dir, str(other), run_profile.load(other).sha256, *good[2:])


class Rig:
    """One supervised run's world: a run folder, a decision record that agrees with the code constants, the injected gates of the supervisor (identity, E6, Git:
    the working tree of a test run is not a clean checkout) and a budget that is large and reads a made-up memory (the child is not the test's business)."""

    def __init__(self, tmp_path, world, reps_sha, profile_fields=None):
        from c_ept.run.identity import Identity
        from c_ept.tests.test_run_ledger import real_spec
        self.tmp, self.run_dir, self.spec = tmp_path, tmp_path / 'run', real_spec(tmp_path)
        self.profile_path = tmp_path / 'profile.json'
        self.profile_path.write_bytes((json.dumps({'kind': 'train', **(S4B if profile_fields is None else profile_fields)}, indent=2) + '\n').encode('utf-8'))
        self.scales = s4b_scales(tmp_path, reps_sha)
        self.manifest = write_manifest(tmp_path, manifest_document(world, self.scales))
        self.identity = Identity(commit='ab' * 20, files={'c_ept/run/runners.py': 'cd' * 32}, protected=())
        self.log, self.stream = [], io.StringIO()

    def e6(self):
        self.log.append('e6')
        return dict(E6_RECORD)

    def capture(self, modules, *, repo=None):
        self.log.append(('capture', tuple(modules)))
        return self.identity

    def verify(self, saved, *, repo=None):
        self.log.append('verify')

    def start_gate(self, repo, identity, **kwargs):
        self.log.append('start_gate')

    def budget(self, wall_s=10 ** 7):
        from c_ept.run import guards
        return guards.Budget(wall_s, 2 ** 40, measure=lambda: 10 ** 6)

    def start(self, **over):
        values = dict(manifest_path=self.manifest, spec_path=self.spec, budget=self.budget(), constants=CPU, allowance=ALLOWANCE, scrub_keys=[KEY], stream=self.stream,
                      loader=TOY_LOADER, bodies={'train': TRAIN_BODY}, e6=self.e6, capture=self.capture, verify=self.verify, start_gate=self.start_gate, poll_s=0.05, repo=REPO)
        values.update(over)
        return runners.start(self.profile_path, self.run_dir, **values)

    def ledger(self):
        from c_ept.run import ledger
        return ledger.Ledger(self.run_dir)


@pytest.fixture
def rig(tmp_path, world, reps_sha):
    return Rig(tmp_path, world, reps_sha)


def test_an_s4b_profile_runs_through_the_supervisor_and_ends_with_exit_0_and_complete_ledger_records(rig, world, reps_sha):
    assert rig.start() == 0, rig.stream.getvalue()
    assert rig.stream.getvalue() == ''
    ledger = rig.ledger()
    assert ledger.stages == (('stats', 'finished'), ('train', 'finished'))
    assert sorted(ledger.stage('stats').outputs) == sorted([runners.STATS_FILE, runners.STATS_RECORD_FILE])
    assert sorted(ledger.stage('train').outputs) == sorted([runners.TRAIN_RECORD_FILE, runners.E6_FILE])
    assert all(ledger.stage('train').inputs[name] == ledger.stage('stats').outputs[name] for name in (runners.STATS_FILE, runners.STATS_RECORD_FILE))       # the chain
    header = ledger.header
    assert header.runner['batch_size'] == 32 and header.runner['device'] == 'cpu' and header.runner['threads'] == 1 and header.runner['cuda_visible_devices'] == ''
    assert header.runner['supervisor']['cuda_visible_devices'] == '' and header.runner['supervisor']['threads'] == 1
    profile = run_profile.load(rig.profile_path)
    assert header.profile_sha256 == profile.sha256 and header.runner['settings'] == runners.effective_settings(profile)
    assert header.runner['training_stage']['wall_cap_s'] == runners.training_stage_cap(profile, ALLOWANCE) == header.runner['supervisor']['training_cap_s']
    assert header.runner['training_stage']['allowance'] == ALLOWANCE.to_record() and header.runner['training_stage']['trainer_cap_s'] == ept_train.WALL_CAP_SECONDS
    assert header.runner['scales'] == {'file_sha256': hashlib.sha256(rig.scales.read_bytes()).hexdigest(),
                                       'condition': {'kill_law': 'shifted', 'k_condition': 'binned', 'k_reps_sha256': reps_sha}}
    assert header.runner['inputs']['manifest_sha256'] == hashlib.sha256(rig.manifest.read_bytes()).hexdigest()
    assert list(header.runner['inputs']['bound']) == runners.manifest_files(runners.read_manifest(rig.manifest))
    assert ('capture', runners.RUNNER_MODULES) in rig.log
    train_record = run_io.read_json(rig.run_dir / runners.TRAIN_RECORD_FILE)
    assert train_record['termination']['reason'] == 'max_epochs' and train_record['identity']['kill_law'] == 'shifted' and train_record['k_reps']['sha256'] == reps_sha
    checks = run_io.read_json(rig.run_dir / runners.E6_FILE)['checks']
    assert checks == [{'call': 0, 'identity': 'verified', 'e6': E6_RECORD}, {'call': 1, 'identity': 'verified', 'e6': E6_RECORD}]       # the identity of the header was verified at every epoch
    assert (rig.run_dir / 'verified_commit.txt').read_text(encoding='utf-8').split() == [rig.identity.commit] * 2                  # ... and it is the identity that the ledger header holds
    assert [note.kind for note in ledger.notes].count('e6') == 3                                      # the start and each stage, by the supervisor; the epochs are in E6.json
    stats_record = run_io.read_json(rig.run_dir / runners.STATS_RECORD_FILE)
    assert stats_record['d7']['columns'] == ['t_s', 'rift_element_known', 'rift_element_air', 'flag_stasis'] and stats_record['scales']['condition']['k_reps_sha256'] == reps_sha


APPROVAL = 'Approved by the author on 2026-10-27: run "P6" on the GPU.\nSecond line, a quote " and a backslash \\ and 한글 승인, a trailing space '


def test_the_header_holds_the_requested_numeric_policy_and_the_approval_text_exactly_as_given(rig):
    """K6c: the header (written once, when the ledger is made) names the policy that the run asked for and the text of the approval, byte for byte, with its hash; the stage record
    names the state that the child read back."""
    assert rig.start(approval=APPROVAL) == 0, rig.stream.getvalue()
    runner = rig.ledger().header.runner
    assert runner['approval'] == {'text': APPROVAL, 'sha256': hashlib.sha256(APPROVAL.encode('utf-8')).hexdigest()}
    assert runner['numeric'] == run_numeric.requested_record()
    record = run_io.read_json(rig.run_dir / runners.TRAIN_RECORD_FILE)
    assert record['numeric']['requested_sha256'] == runner['numeric']['requested_sha256'] and record['numeric']['measured']['device'] == 'cpu'
    assert all(record['numeric']['measured'][name] == run_numeric.REQUIRED[name] for name in run_numeric.REQUIRED)


def test_a_run_started_without_an_approval_text_records_null_and_the_text_is_never_invented(rig):
    assert rig.start() == 0, rig.stream.getvalue()
    assert rig.ledger().header.runner['approval'] is None


@pytest.mark.parametrize('bad', ['', '   ', '\n\t', 5, b'bytes', ['text']])
def test_an_approval_that_is_not_a_text_with_a_visible_character_is_refused_before_the_reservation(rig, bad):
    assert rig.start(approval=bad) == 2 and not rig.run_dir.exists() and 'approval' in rig.stream.getvalue()


def test_the_budget_has_to_hold_the_training_cap_with_the_epoch_allowance_and_the_finalisation_margin(rig):
    profile = run_profile.load(rig.profile_path)
    cap = runners.training_stage_cap(profile, ALLOWANCE)
    assert rig.start(budget=rig.budget(wall_s=cap + 3600.0 - 1.0)) == 2                              # the supervisor's margin is 1 h on top of the stage's cap
    assert not rig.run_dir.exists() and 'budget' in rig.stream.getvalue()


def test_a_scales_file_of_another_condition_is_exit_2_before_anything_is_reserved(rig, tmp_path):
    rig.scales = write_scales(tmp_path, 'plain_scales.json')                                          # the plain law and the raw gold, for an S4b profile
    rig.manifest = write_manifest(tmp_path, {**run_io.read_json(rig.manifest), 'scales': str(rig.scales)}, 'plain.json')
    assert rig.start() == 2
    assert not rig.run_dir.exists() and 'kill_law' in rig.stream.getvalue()


def test_representatives_that_differ_from_the_rows_are_found_in_the_child_and_end_as_exit_2(rig, tmp_path):
    rig.scales = s4b_scales(tmp_path, 'e' * 64, 'wrong_reps.json')                                    # the file looks right, so the early check passes; the rows say otherwise
    rig.manifest = write_manifest(tmp_path, {**run_io.read_json(rig.manifest), 'scales': str(rig.scales)}, 'wrong.json')
    assert rig.start() == 2
    ledger = rig.ledger()
    assert ledger.stages == (('stats', 'begun'),)
    stops = [dict(note.record) for note in ledger.notes if note.kind == 'stop']
    assert len(stops) == 1 and stops[0]['reason'] == 'refused' and stops[0]['exit_code'] == 2 and 'representatives' in stops[0]['detail']
    assert not (rig.run_dir / runners.STATS_FILE).exists() and not (rig.run_dir / runners.STATS_RECORD_FILE).exists()           # nothing was written before the refusal


def test_a_profile_or_manifest_that_is_not_acceptable_is_exit_2_and_its_text_holds_no_key(rig, tmp_path):
    other = tmp_path / 'eval.json'
    other.write_bytes(b'{"kind": "evaluate", "role": "STOP", "models": ["ept"]}\n')
    rig.profile_path = other
    assert rig.start() == 2 and 'train profile' in rig.stream.getvalue()
    rig.profile_path.write_bytes((json.dumps({'kind': 'train', **S4B}) + '\n').encode('utf-8'))
    rig.stream.truncate(0)
    broken = write_manifest(tmp_path, {'schema': 'ept-run-inputs-v1'}, 'broken.json')
    assert rig.start(manifest_path=broken) == 2 and 'fields' in rig.stream.getvalue() and not rig.run_dir.exists()
    rig.stream.truncate(0)
    assert rig.start(manifest_path=tmp_path / f'absent_{KEY}.json') == 2 and KEY not in rig.stream.getvalue() and 'regular JSON file' in rig.stream.getvalue()


def test_a_change_of_the_imported_code_during_the_training_is_exit_2_with_the_refusal_written_down(tmp_path, world, reps_sha):
    """The child's identity check fails at the second epoch (its verify is replaced in ``run_runner_bodies``): the child ends with a refusal, the supervisor
    exits 2 and notes it, the first epoch's check is on record and no training record is written."""
    rig = Rig(tmp_path, world, reps_sha, {**S4B, 'epochs': 3})
    assert rig.start(bodies={'train': 'c_ept.tests.run_runner_bodies:changed_identity_train_stage'}) == 2
    ledger = rig.ledger()
    assert ledger.stages == (('stats', 'finished'), ('train', 'begun'))
    stops = [dict(note.record) for note in ledger.notes if note.kind == 'stop']
    assert len(stops) == 1 and stops[0]['reason'] == 'refused' and stops[0]['exit_code'] == 2 and 'imported code changed' in stops[0]['detail']
    checks = run_io.read_json(rig.run_dir / runners.E6_FILE)['checks']
    assert [c['identity'] for c in checks] == ['verified', 'refused'] and not (rig.run_dir / runners.TRAIN_RECORD_FILE).exists()


def test_a_store_that_is_not_complete_or_a_terrain_that_is_not_the_build_is_exit_2_before_the_reservation(rig, tmp_path):
    original = run_io.read_json(rig.manifest)
    missing = tmp_path / 'no_store'
    rig.manifest = write_manifest(tmp_path, {**original, 's2': str(missing)}, 'no_store.json')
    assert rig.start() == 2 and 's2 store' in rig.stream.getvalue() and not rig.run_dir.exists()
    rig.stream.truncate(0)
    folder = tmp_path / 'terrain'
    folder.mkdir()
    for name in runners.TERRAIN_FILES:
        (folder / name).write_bytes(b'x')
    rig.manifest = write_manifest(tmp_path, {**original, 'terrain': str(folder)}, 'terrain.json')
    assert rig.start() == 2 and 'terrain' in rig.stream.getvalue() and not rig.run_dir.exists()          # the toy S2 records no terrain hash


def test_the_manifest_is_read_once_in_the_parent_and_the_header_and_the_inputs_hash_it_from_the_same_bytes(rig, monkeypatch):
    reads = []
    for name in ('read_json', 'read_json_with_sha'):
        real = getattr(run_io, name)
        monkeypatch.setattr(run_io, name, lambda path, *a, _real=real, **k: (reads.append(Path(path)), _real(path, *a, **k))[1])
    assert rig.start() == 0, rig.stream.getvalue()
    assert [p for p in reads if p == rig.manifest] == [rig.manifest]


def test_a_training_stage_that_ends_at_the_wall_cap_in_budget_is_a_success_of_the_stage(tmp_path, world, reps_sha):
    """The 'budget' end is not an error: the child (whose clock passes the 24 h cap while the second epoch trains) exits 0, the ledger finishes the stage and
    the run ends with exit code 0; the record of the stage says 'budget' and how many seconds the training used against the cap."""
    rig = Rig(tmp_path, world, reps_sha, {**S4B, 'epochs': 3})
    assert rig.start(bodies={'train': 'c_ept.tests.run_runner_bodies:budget_train_stage'}) == 0, rig.stream.getvalue()
    assert rig.ledger().stages == (('stats', 'finished'), ('train', 'finished'))
    record = run_io.read_json(rig.run_dir / runners.TRAIN_RECORD_FILE)
    assert record['termination']['reason'] == 'budget' and record['termination']['epochs_run'] == 2 and record['termination']['done'] is True
    assert record['time']['over_cap'] is True and record['time']['training_seconds'] >= 90_000.0
    assert [note.kind for note in rig.ledger().notes if note.kind == 'stop'] == []                  # nothing is recorded as a stop


def test_a_whole_fit_population_with_the_store_list_uses_every_fit_view_and_none_computes_no_list(world, inputs, tmp_path, monkeypatch):
    calls = []
    real = data.store_constant_columns
    monkeypatch.setattr(data, 'store_constant_columns', lambda *args, **kwargs: calls.append(args) or real(*args, **kwargs))
    plain = runners.make_stats(profile_of(tmp_path), inputs)
    assert calls == [] and plain.dropped == ()                                                        # column_drop 'none' computes no list
    store = runners.make_stats(profile_of(tmp_path, 'store.json', column_drop='store'), inputs)
    fit_views = np.flatnonzero(features.split_views(world.s1, world.s2, 'FIT'))
    assert len(calls) == 1 and store.n_views == len(fit_views) and store.population == 'whole_fit' and store.dropped == store.dropped_store == real(world.s2, fit_views)
    selected = runners.make_stats(profile_of(tmp_path, 'selected.json', stats_population='selected_fit'), inputs)
    assert selected.population == 'selected_fit' and selected.dropped == () and selected.n_views < store.n_views


