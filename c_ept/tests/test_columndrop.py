"""Tests of the training loop under the column drop and the population of the statistics (S4b preparation, task M5, decisions D7 and D12).

What is pinned here. ``TrainConfig.column_drop`` ('none' by default) and ``TrainConfig.stats_population`` ('whole_fit' by default) are the settings of the run for the two
facts that the statistics record (``data.Stats.column_drop`` and ``.population``, with the list of the dropped columns). Each is written into the identity of the run only
when it is not the default (absent means the default), the list of the dropped columns of a run that drops is written next to it as a hash, and the run refuses statistics that
were made under the other setting before it does any work, in both directions. The statistics and the functions that make them are tested in ``test_ept_data.py``.

What the "default" tests prove. A test of this file can only compare the code as it is now: a default run has the identity of the code before this task with no key added, which
the test checks by name; that the default run gives the bits of the base commit de0c3a8 is shown by the comparison script of the T012 session folder (``scratchpad\\s4bprep``,
``m5_proof.py``) that was run at both commits.
"""
import dataclasses
import gc
import json
import math

import pytest

from c_ept import runlog
from c_ept.ept import data, encoder, loss, train
from c_ept.tests.ept_toys import build_ept_world, toy_static  # noqa: F401  (the fixtures are imported by ruling R1)


@pytest.fixture(scope='module')
def run_world(tmp_path_factory):
    """The EPT toy world of the training tests (7 events, 5 edges), the statistics of every earlier run, and those that drop the D7 list of the toy."""
    folder = tmp_path_factory.mktemp('columndrop_train')
    world = build_ept_world(folder, n_events=7, n_edges=5)
    store = data.store_constant_columns(world.s2, world.fit_views)
    plain = data.fit_stats(world.s1, world.s2)
    dropping = data.fit_stats(world.s1, world.s2, column_drop='store', store_constant=store)
    yield world, plain, dropping, store
    del world
    gc.collect()


def run_data(world, stats):
    rows = {role: data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, role)) for role in ('FIT', 'STOP')}
    fit = data.Batches(rows['FIT'], max(1, math.ceil(len(rows['FIT'].units) / 3)), 7, stats=stats, training=True)
    stop = data.Batches(rows['STOP'], max(1, len(rows['STOP'].units)), 7, stats=stats, training=False)
    scales = {key: 1.0 + 0.25 * i for i, key in enumerate(loss.BLOCKS)}
    return train.TrainData(fit=fit, stop=stop, scales=scales, static=toy_static())


def run_cfg(**over):
    return train.TrainConfig(encoder=encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.1, kappas=1, field_channels=2), lr=3e-3, weight_decay=0.01,
                             warmup_steps=2, patience=10, **over)


def run_train(folder, world, stats, cfg, epochs=2):
    return train.train(cfg, run_data(world, stats), folder, device='cpu', epochs=epochs, check=lambda: None, seed=3)


def test_the_train_config_has_the_column_drop_and_the_population_after_the_decay_groups_with_the_defaults_and_refuses_another_name():
    names = [f.name for f in dataclasses.fields(train.TrainConfig)]
    assert names.index('column_drop') == names.index('decay_groups') + 1 and names.index('stats_population') == names.index('column_drop') + 1
    assert names[:6] == ['encoder', 'lr', 'weight_decay', 'warmup_steps', 'patience', 'model_spec']
    small = encoder.EncoderConfig(d=16, layers=1, heads=2)
    cfg = train.TrainConfig(small, 1e-3, 0.0, 1)
    assert cfg.column_drop == 'none' and cfg.stats_population == 'whole_fit'
    assert train.TrainConfig(small, 1e-3, 0.0, 1, 2, None, 'poisson', 'raw', 'all').column_drop == 'none'            # the positional calls of the earlier tasks
    both = train.TrainConfig(small, 1e-3, 0.0, 1, column_drop='store_and_subset', stats_population='selected_fit')
    assert both.column_drop == 'store_and_subset' and both.stats_population == 'selected_fit'
    for bad in ('Store', 'all', '', None, 1):
        with pytest.raises(ValueError, match='column_drop'):
            train.TrainConfig(small, 1e-3, 0.0, 1, column_drop=bad)
        with pytest.raises(ValueError, match='stats_population'):
            train.TrainConfig(small, 1e-3, 0.0, 1, stats_population=bad)


def test_a_dropping_run_records_the_setting_and_the_hash_of_its_list_in_its_identity_and_the_default_run_does_not(run_world, tmp_path):
    world, plain, dropping, store = run_world
    default = run_train(tmp_path / 'default', world, plain, run_cfg())
    dropped = run_train(tmp_path / 'dropped', world, dropping, run_cfg(column_drop='store'))
    assert 't_s' in store and len(store) > 3 and dropping.dropped == store
    for key in ('column_drop', 'column_drop_sha256', 'stats_population'):
        assert key not in default.head['identity']                                       # absent means the default: the identity of every earlier run
    identity = dropped.head['identity']
    assert identity['column_drop'] == 'store' and 'stats_population' not in identity
    assert identity['column_drop_sha256'] == runlog.sha256_json({'dropped': list(dropping.dropped), 'dropped_store': list(dropping.dropped_store)})
    assert identity['fit']['stats_sha256'] == dropping.sha256 != default.head['identity']['fit']['stats_sha256']
    assert all(math.isfinite(v) for v in dropped.stop_losses) and dropped.stop_losses != default.stop_losses       # other inputs, other losses


def test_the_statistics_and_the_settings_must_agree_before_any_work_in_both_directions(run_world, tmp_path):
    world, plain, dropping, _ = run_world
    with pytest.raises(ValueError, match='column_drop'):
        run_train(tmp_path / 'a', world, plain, run_cfg(column_drop='store'))              # the settings ask for a drop that the statistics do not make
    with pytest.raises(ValueError, match='column_drop'):
        run_train(tmp_path / 'b', world, dropping, run_cfg())                              # the statistics drop columns that the settings do not ask for
    with pytest.raises(ValueError, match='column_drop'):
        run_train(tmp_path / 'c', world, dropping, run_cfg(column_drop='store_and_subset'))
    with pytest.raises(ValueError, match='stats_population'):
        run_train(tmp_path / 'd', world, plain, run_cfg(stats_population='selected_fit'))
    selected = data.fit_stats(world.s1, world.s2, matches=tuple(sorted({int(m) for m in world.s2.array('views/match')[world.fit_views]})),
                              stats_population='selected_fit')
    with pytest.raises(ValueError, match='stats_population'):
        run_train(tmp_path / 'e', world, selected, run_cfg())
    for name in 'abcde':
        assert not (tmp_path / name).exists() or not list((tmp_path / name).glob('epoch_*'))
    ok = run_train(tmp_path / 'f', world, selected, run_cfg(stats_population='selected_fit'), epochs=1)
    assert ok.head['identity']['stats_population'] == 'selected_fit' and 'column_drop' not in ok.head['identity']


def test_a_run_cannot_be_resumed_under_the_other_setting_in_either_direction(run_world, tmp_path):
    world, plain, dropping, _ = run_world
    for first, stats_first, second, stats_second in (('plain', plain, 'drop', dropping), ('drop', dropping, 'plain', plain)):
        cfg = {'plain': run_cfg(), 'drop': run_cfg(column_drop='store')}
        run_train(tmp_path / first, world, stats_first, cfg[first], epochs=2)
        with pytest.raises(ValueError, match='differ'):                                     # ``_check_identity`` names the keys that differ
            run_train(tmp_path / first, world, stats_second, cfg[second], epochs=2)
    json.dumps(plain.to_json())                                                              # the statistics of every earlier run are plain JSON as before
