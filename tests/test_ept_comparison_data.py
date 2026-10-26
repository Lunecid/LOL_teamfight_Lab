"""Tiny CPU synthetic guarded stores only; no research fixture or LightGBM fit."""
import importlib.abc
import json
import math
import sys
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest


class DenyProtected(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('c_ept.cohort', 'research_pipeline.cohort_policy') or fullname.startswith('research_pipeline.holdout_r3'):
            raise AssertionError('protected import forbidden')


_denied = DenyProtected()
sys.meta_path.insert(0, _denied)
try:
    from c_ept import config as C
    from c_ept.comparison import data
    from c_ept.data import head_rows
    from c_ept.ept import data as neural, loss
    from c_ept.models import baselines_b1 as b1
    from c_ept.preprocessing import features, tokens
finally:
    sys.meta_path.remove(_denied)


class GuardedArray:
    """Shape/dtype metadata is legal; reading any unselected value fails."""
    def __init__(self, values, allowed):
        self.values = values
        self.allowed = np.asarray(allowed)
        self.shape, self.ndim, self.dtype = values.shape, values.ndim, values.dtype
        self.reads = []

    def __len__(self):
        return len(self.values)

    def __array__(self, dtype=None, copy=None):
        raise AssertionError('whole nonselected value array read')

    def __getitem__(self, key):
        first = key[0] if isinstance(key, tuple) else key
        indices = np.atleast_1d(np.arange(len(self))[first])
        if not np.isin(indices, self.allowed).all():
            raise AssertionError('nonselected values read')
        self.reads.extend(indices.tolist())
        return self.values[key]


class Store:
    def __init__(self, arrays):
        self.arrays = arrays
        self.reads = []
        self.manifest = {'arrays': {k: {'shape': list(v.shape)} for k, v in arrays.items()}}

    def array(self, key):
        self.reads.append(key)
        return self.arrays[key]


def world():
    # All indices are invented; frame/unit/onset indices intentionally differ.
    n = 18
    role = np.array([C.ROLES.index('FIT')] * 9 + [C.ROLES.index('STOP')] * 7 + [C.ROLES.index('DEV')] * 2)
    fm = np.arange(n)[::-1].copy()
    um = np.roll(np.arange(n), 3)
    ou = np.arange(n)[::-1].copy()
    mark, size = um % C.N_MARKS, um % 4
    evunit = np.repeat(np.arange(n), 2)
    evkind = np.tile([1, 2], n)
    evmatch = um[evunit]
    frame_t = 100000 + fm * 100
    evref = 110000 + evmatch * 100 + (evkind - 1) * 10000
    s1 = Store({'matches/role': role, 'frame_rows/match': fm, 'frame_rows/t0': frame_t,
                'frame_rows/exposure_ms': np.full((n, C.N_BINS), 10000),
                'units/match': um, 'units/mark': mark, 'units/size': size,
                'units/in_scope': np.ones(n, dtype=bool), 'units/kill0': np.arange(n),
                'units/K': np.column_stack([um % 3 + 1, um % 2 + 1]),
                'units/O': np.broadcast_to((um % 2)[:, None], (n, features.N_O)).copy(),
                'units/conv': np.broadcast_to((um % 3)[:, None], (n, features.N_O)).copy(),
                'units/conv_exposure_ms': np.full(n, 30000),
                'onsets/row': n - 1 - um[ou], 'onsets/unit': ou, 'onsets/mark': mark[ou], 'onsets/bin': um[ou] % C.N_BINS,
                'event_rows/kind': evkind, 'event_rows/unit': evunit, 'event_rows/ref': evref,
                'kills/x': 2000 + um * 100, 'kills/y': 3000 + um * 100, 'kills/pos_missing': np.zeros(n, dtype=bool)})
    vm = np.r_[fm, evmatch]
    v = len(vm)
    arrays = {'views/src': np.r_[np.zeros(n, dtype=int), evkind], 'views/src_index': np.r_[np.arange(n), np.arange(2*n)],
              'views/match': vm, 'views/ref': np.r_[frame_t, evref]}
    rng = np.random.default_rng(4)
    arrays.update({'x/players_num': rng.uniform(0, 100, (v, 10, len(tokens.PLAYER_NUM))),
                   'x/summary': rng.uniform(0, 100, (v, len(tokens.SUMMARY))),
                   'x/players_zone': np.zeros((v, 10), dtype=np.int8), 'x/landmarks': np.zeros((v, neural.N_LANDMARKS, 2)),
                   'x/geo_pp': np.ones((v, 10, 10)), 'x/geo_pl': np.ones((v, 10, neural.N_LANDMARKS)),
                   'x/ev_type': np.full((v, 2), -1), 'x/ev_dt': np.zeros((v, 2)), 'x/ev_team': np.zeros((v, 2), dtype=int),
                   'x/ev_xy': np.zeros((v, 2, 2)), 'x/ev_actors': np.zeros((v, 2), dtype=int), 'x/ev_gold': np.zeros((v, 2)),
                   'x/ev_n': np.zeros(v, dtype=int), 'x/ed_n': np.zeros(v, dtype=int),
                   'x/ed_i': np.full((v, 2), -1), 'x/ed_j': np.full((v, 2), -1), 'x/ed_kind': np.full((v, 2), -1), 'x/ed_dt': np.zeros((v, 2))})
    labels = Store({'labels/gold_bin': np.repeat((um % 31)[:, None], 2, axis=1),
                    'labels/kill_gold': np.repeat((1000 + um * 50)[:, None], 2, axis=1).astype(float)})
    return SimpleNamespace(s1=s1, s2=Store(arrays), labels=labels, static=np.zeros((1, 1, 1), dtype=np.float32))


def guard_world(w, membership, roles=('FIT', 'STOP')):
    selected = [m for role in roles for m in (membership.fit if role == 'FIT' else membership.stop)]
    vm = w.s2.array('views/match')
    guards = []
    for key, array in list(w.s2.arrays.items()):
        if key.startswith('x/'):
            guarded = GuardedArray(array, np.flatnonzero(np.isin(vm, selected)))
            w.s2.arrays[key] = guarded
            guards.append(guarded)
    units = np.flatnonzero(np.isin(w.s1.array('units/match'), selected))
    for store, keys in ((w.labels, tuple(w.labels.arrays)), (w.s1, ('kills/x', 'kills/y', 'kills/pos_missing'))):
        for key in keys:
            guarded = GuardedArray(store.arrays[key], units)
            store.arrays[key] = guarded
            guards.append(guarded)
    return guards


def scales():
    return {key: 1. for key in loss.BLOCKS}


def test_membership_proxy_ceil_nested_and_roundtrip():
    w = world()
    a = data.select_membership(w.s1)
    b = data.select_membership(w.s1, .5)
    expected = neural.proxy_matches(w.s1)
    assert a.fit == tuple(expected['FIT']) and a.stop == tuple(expected['STOP'])
    assert len(a.fit) == math.ceil(9 * .25) and len(a.stop) == math.ceil(7 * .25)
    assert set(a.fit) <= set(b.fit) and set(a.stop) <= set(b.stop)
    assert set(a.fit).isdisjoint(a.stop)
    record = json.loads(json.dumps(a.to_record()))
    assert data.Membership.from_record(record) == a
    assert a.verify_against(w.s1) is a
    assert set(w.s1.reads) == {'matches/role'}
    with pytest.raises(FrozenInstanceError):
        a.fit = (0,)
    record['fit'][-1] = 999
    with pytest.raises(ValueError, match='SHA256'):
        data.Membership.from_record(record)


def test_membership_resigned_forgery_and_role_drift_refused():
    w = world()
    a = data.select_membership(w.s1)
    candidates = tuple(sorted(set(range(9)) - set(a.fit)))[:len(a.fit)]
    for forged in (replace(a, fit=candidates), replace(a, fit=(16,)), replace(a, selection_seed=a.selection_seed + 1)):
        back = data.Membership.from_record(forged.to_record())
        with pytest.raises(ValueError, match='authoritative'):
            back.verify_against(w.s1)
    w.s1.arrays['matches/role'][0] = C.ROLES.index('DEV')
    with pytest.raises(ValueError, match='authoritative'):
        a.verify_against(w.s1)


@pytest.mark.parametrize('changes', [{'share': 0}, {'share': float('nan')}, {'selection_seed': -1}, {'selection_seed': True}, {'selection_seed': 1.5}])
def test_selection_invalid_arguments(changes):
    with pytest.raises((ValueError, TypeError)):
        data.select_membership(world().s1, **changes)


def test_stats_reads_only_selected_fit_views_and_matches_original_values():
    w = world()
    member = data.select_membership(w.s1)
    views = np.flatnonzero(np.isin(w.s2.array('views/match'), member.fit))
    wanted = w.s2.array('x/players_num')[views].reshape(-1, len(tokens.PLAYER_NUM))
    guards = guard_world(w, member, ('FIT',))
    ticks = []
    stats = data.fit_stats_subset(w.s1, w.s2, member, lambda: ticks.append(1))
    assert stats.n_views == len(views) == len(member.fit) * 3
    j = tokens.PLAYER_NUM.index('totalGold')
    assert stats.arrays['players_num'].centre[j] == pytest.approx(wanted[:, j].mean())
    assert stats.arrays['players_num'].scale[j] == pytest.approx(wanted[:, j].std())
    assert len(ticks) >= 2 and guards[0].reads
    with pytest.raises(AssertionError, match='nonselected'):
        neural.fit_stats(w.s1, w.s2)


def test_check_refusal_precedes_any_store_access():
    w = world()
    member = data.select_membership(w.s1)
    w.s1.reads.clear()
    def refuse():
        raise RuntimeError('guard refuses')
    for call in (lambda: data.fit_stats_subset(w.s1, w.s2, member, refuse),
                 lambda: data.build_neural_data(w, member, scales=scales(), check=refuse),
                 lambda: data.fit_b1_subset(w.s1, w.s2, w.labels, member, check=refuse)):
        with pytest.raises(RuntimeError, match='guard refuses'):
            call()
        assert not w.s1.reads and not w.s2.reads and not w.labels.reads


def test_neural_reuses_loaded_stats_and_only_selected_rows(monkeypatch, tmp_path):
    w = world()
    member = data.select_membership(w.s1)
    stats = data.fit_stats_subset(w.s1, w.s2, member, lambda: None)
    stats.save(tmp_path / 'Stats.json')
    stats = neural.Stats.load(tmp_path / 'Stats.json')
    guards = guard_world(w, member)
    def refuse_refit(*args, **kwargs):
        raise AssertionError('supplied Stats refitted')
    monkeypatch.setattr(data, 'fit_stats_subset', refuse_refit)
    first, reused, receipt = data.build_neural_data(w, member, scales=scales(), stats=stats, batch_size=1, check=lambda: None)
    second, _, other = data.build_neural_data(w, member, scales=scales(), stats=stats, batch_size=1, train_seed=99, check=lambda: None)
    assert reused is stats and first.fit.stats is stats and first.stop.stats is stats
    assert receipt['membership_sha256'] == other['membership_sha256'] == member.sha256
    assert receipt['stats_sha256'] == stats.sha256 and receipt['stats_fitted_here'] is False
    assert first.fit.seed != second.fit.seed
    for batches, selected in ((first.fit, member.fit), (first.stop, member.stop)):
        for block in batches.rows.blocks.values():
            assert set(block.fields['match']) == set(selected)
        assert batches.rows.head_rows.matches.tolist() == list(selected)
        for batch in batches.epoch(0):
            assert np.isin(w.s2.array('views/match')[batch['view'].numpy()], selected).all()
    from c_ept.ept.train import fit_biases
    biases, info = fit_biases(first.fit.rows)
    assert info['rate_net.out.bias']['rows'] == len(member.fit)
    assert all(np.isfinite(v).all() for v in biases.values())
    assert all(g.reads for g in guards)


def test_neural_fit_stats_once_by_default_and_scale_identity(monkeypatch):
    w = world()
    member = data.select_membership(w.s1)
    real = data.fit_stats_subset
    calls = []
    def counted(*a, **kw):
        calls.append(1)
        return real(*a, **kw)
    monkeypatch.setattr(data, 'fit_stats_subset', counted)
    factors = scales()
    training, stats, receipt = data.build_neural_data(w, member, scales=factors, check=lambda: None)
    assert calls == [1] and training.scales is factors
    assert stats.n_views == len(member.fit) * 3 and receipt['stats_fitted_here'] is True


def test_wrong_stats_count_rejected_before_target_reads():
    w = world()
    member = data.select_membership(w.s1)
    stats = neural.fit_stats(w.s1, w.s2)
    with pytest.raises(ValueError, match='view count'):
        data.build_neural_data(w, member, scales=scales(), stats=stats, check=lambda: None)
    assert not w.labels.reads


def test_fresh_b1_calls_existing_fit_head_all_selected_inputs(monkeypatch):
    w = world()
    member = data.select_membership(w.s1)
    col = tokens.SUMMARY.index('dragon_up')
    vm = w.s2.array('views/match')
    w.s2.arrays['x/summary'][:, col] = np.where(np.isin(vm, member.fit), 0., vm + 1.)
    full_dropped = features.dropped_columns(w.s2, np.flatnonzero(vm < 9))
    assert 'dragon_up' not in full_dropped
    expected_drop = features.dropped_columns(w.s2, np.flatnonzero(np.isin(vm, member.fit)))
    expected = {}
    for role, selected in (('FIT', member.fit), ('STOP', member.stop)):
        # Independent oracle: legacy whole-role head rows filtered by global match.
        rows = b1.HeadRows(w.s1, w.s2, w.labels, expected_drop, role)
        for kind, names in (('masked', b1.HEAD_NAMES), ('open', b1.OPEN_HEAD_NAMES)):
            for name in names:
                d = rows.data(name, kind)
                keep = np.isin(d.match, selected)
                expected[(role, kind, name)] = (d.X[keep], d.y[keep], None if d.e is None else d.e[keep],
                                                None if d.team is None else d.team[keep], d.row[keep])
    guards = guard_world(w, member)
    calls = []
    for name in b1.HEAD_NAMES:
        class Spy:
            def __init__(self, kind, params, name=name):
                self.kind, self.params, self.name = kind, params, name
                self.n_zero_exposure_ = 0
            def fit(self, design, y, *extra, layout, valid, check):
                check()
                X = design.take()
                fit = expected[('FIT', self.kind, self.name)]
                stop = expected[('STOP', self.kind, self.name)]
                np.testing.assert_array_equal(X, fit[0])
                np.testing.assert_array_equal(y, fit[1])
                np.testing.assert_array_equal(valid[0], stop[0])
                np.testing.assert_array_equal(valid[1], stop[1])
                field = 3 if self.name == 'M3Gold' else 2
                if fit[field] is not None:
                    np.testing.assert_array_equal(extra[0], fit[field])
                    np.testing.assert_array_equal(valid[2], stop[field])
                assert self.params is None
                calls.append((self.kind, self.name))
                return self
        monkeypatch.setitem(b1.HEAD_CLASSES, name, Spy)
    real_fit_head = b1.fit_head
    def checked_rows(fit, stop, name, kind, **kw):
        for rows, role in ((fit, 'FIT'), (stop, 'STOP')):
            d = rows.data(name, kind)
            np.testing.assert_array_equal(d.row, expected[(role, kind, name)][4])
            assert set(d.match) == set(member.fit if role == 'FIT' else member.stop)
        return real_fit_head(fit, stop, name, kind, **kw)
    monkeypatch.setattr(b1, 'fit_head', checked_rows)
    result = data.fit_b1_subset(w.s1, w.s2, w.labels, member, check=lambda: None)
    assert len(calls) == 13
    assert result.dropped == expected_drop and 'dragon_up' in result.dropped
    assert result.membership_sha256 == member.sha256
    receipt = result.comparison_receipt
    assert receipt['params']['seed'] == 1 and receipt['params']['num_threads'] == 8
    assert receipt['early_stop'] == 50 and receipt['max_rounds'] == 1000
    assert set(result.masked) == set(b1.HEAD_NAMES) and set(result.open) == set(b1.OPEN_HEAD_NAMES)
    assert guards[0].reads and w.labels.arrays['labels/gold_bin'].reads


@pytest.mark.parametrize('role', ['TEST', 'CAL', 'EXTERNAL', 'INVALID'])
def test_bad_role_rejected_before_access_even_with_matches(role):
    class Never:
        def array(self, key):
            raise AssertionError('role validation bypassed')
    with pytest.raises(ValueError):
        head_rows.HeadRows(Never(), Never(), Never(), role, lambda *_: None, matches=(0,))


def test_explicit_selection_refuses_other_role_and_preserves_legacy_default():
    w = world()
    with pytest.raises(ValueError, match='requested role'):
        neural.ept_head_rows(w.s1, w.s2, w.labels, 'FIT', matches=(9,))
    with pytest.raises(ValueError, match='distinct'):
        neural.ept_head_rows(w.s1, w.s2, w.labels, 'FIT', matches=(0, 0))
    old = neural.EngagementRows(neural.ept_head_rows(w.s1, w.s2, w.labels, 'FIT'))
    full = neural.EngagementRows(neural.ept_head_rows(w.s1, w.s2, w.labels, 'FIT', matches=tuple(range(9))))
    for key in old.blocks:
        assert old.blocks[key].fields.keys() == full.blocks[key].fields.keys()
        for field in old.blocks[key].fields:
            np.testing.assert_array_equal(old.blocks[key].fields[field], full.blocks[key].fields[field])
        np.testing.assert_array_equal(old.blocks[key].views, full.blocks[key].views)
    assert neural.fit_stats(w.s1, w.s2).sha256 == neural.fit_stats(w.s1, w.s2, matches=tuple(range(9))).sha256


# ---- the settings of S4b (preparation M3 review m6, M5: decisions D5, D21 B16, D7, D12): defaults unchanged, the others passed through ------------------
def flagged_world():
    """The world with two summary columns that are constant over the whole store (``atakhan_in_patch``) and inside the selected FIT only (``dragon_up``)."""
    w = world()
    member = data.select_membership(w.s1)
    vm = w.s2.array('views/match')
    w.s2.arrays['x/summary'][:, tokens.SUMMARY.index('atakhan_in_patch')] = 0.0
    w.s2.arrays['x/summary'][:, tokens.SUMMARY.index('dragon_up')] = np.where(np.isin(vm, member.fit), 3.0, vm + 1.0)
    w.s2.arrays['x/summary'][:, tokens.SUMMARY.index('old_champion_kill_b')] = 0.0      # one half of a swap pair is constant, the other half varies
    return w, member


class FakeHead:
    """What ``fit_head`` returns in the tests that look at the settings: only the law that the kill head says it was fitted for."""
    def __init__(self, kill_law):
        self.kill_law = kill_law


def spy_fit_head(monkeypatch):
    seen = []

    def fake(fit, stop, name, kind, **kw):
        seen.append((fit, stop, name, kind))
        return FakeHead(fit.kill_law) if name == 'M3Kills' else FakeHead('poisson')
    monkeypatch.setattr(b1, 'fit_head', fake)
    return seen


def test_defaults_of_the_settings_are_the_calls_and_the_receipts_of_the_earlier_runs():
    w = world()
    member = data.select_membership(w.s1)
    training, stats, receipt = data.build_neural_data(w, member, scales=scales(), check=lambda: None)
    assert stats.sha256 == neural.fit_stats(w.s1, w.s2, matches=member.fit).sha256 == data.fit_stats_subset(w.s1, w.s2, member, lambda: None).sha256
    assert stats.column_drop == 'none' and stats.population == 'whole_fit' and stats.dropped == ()
    assert set(receipt) == {'schema', 'membership_sha256', 'membership', 'stats_sha256', 'stats_fitted_here', 'predictor_fit_scope', 'head_init_scope', 'train_seed',
                            'rows', 'loss_scales_scope', 'auxiliary_scope'}                  # nothing is added for the default settings
    for batches in (training.fit, training.stop):
        assert batches.rows.head_rows.kill_law == 'poisson' and batches.rows.head_rows.k_condition == 'raw'
    explicit, same, other = data.build_neural_data(w, member, scales=scales(), check=lambda: None, kill_law='poisson', k_condition='raw', column_drop='none',
                                                   stats_population='whole_fit')
    assert same.sha256 == stats.sha256 and other == receipt
    import inspect
    for fn in (data.build_neural_data, data.fit_stats_subset, data.fit_b1_subset):
        parameters = inspect.signature(fn).parameters
        assert all(parameters[name].default == default for name, default in (('kill_law', 'poisson'), ('k_condition', 'raw'), ('column_drop', 'none'),
                                                                               ('stats_population', 'whole_fit')) if name in parameters), fn
    assert 'kill_law' in inspect.signature(data.fit_b1_subset).parameters and 'k_condition' in inspect.signature(data.fit_b1_subset).parameters
    assert 'column_drop' in inspect.signature(data.fit_b1_subset).parameters
    assert {'column_drop', 'stats_population'} <= set(inspect.signature(data.build_neural_data).parameters)
    assert {'kill_law', 'k_condition'} <= set(inspect.signature(data.build_neural_data).parameters)


def test_the_neural_rows_take_the_kill_law_and_the_k_condition_and_the_receipt_says_so():
    w = world()
    member = data.select_membership(w.s1)
    training, stats, receipt = data.build_neural_data(w, member, scales=scales(), check=lambda: None, kill_law='shifted', k_condition='binned')
    for batches, role in ((training.fit, 'FIT'), (training.stop, 'STOP')):
        rows = batches.rows.head_rows
        assert rows.kill_law == 'shifted' and rows.k_condition == 'binned' and rows.role == role
        assert 'd' in batches.rows.blocks[('M3Kills', 'open')].fields                         # the first-kill team of the shifted law
    reps = training.fit.rows.head_rows.k_reps_sha256()
    assert reps == training.stop.rows.head_rows.k_reps_sha256() and len(reps) == 64
    assert receipt['kill_law'] == 'shifted' and receipt['k_condition'] == 'binned' and receipt['k_reps_sha256'] == reps
    assert 'column_drop' not in receipt and 'stats_population' not in receipt
    plain = data.build_neural_data(w, member, scales=scales(), check=lambda: None)[2]
    assert {k: v for k, v in receipt.items() if k not in ('kill_law', 'k_condition', 'k_reps_sha256', 'rows')} == {k: v for k, v in plain.items() if k != 'rows'}
    for bad in ({'kill_law': 'bogus'}, {'k_condition': 'bogus'}, {'column_drop': 'bogus'}, {'stats_population': 'bogus'}):
        fresh = world()
        with pytest.raises(ValueError):
            data.build_neural_data(fresh, member, scales=scales(), check=lambda: None, **bad)
        assert not fresh.labels.reads                                                         # refused before the labels are read


def test_the_neural_data_pass_the_column_drop_and_the_population_to_the_statistics():
    w, member = flagged_world()
    training, stats, receipt = data.build_neural_data(w, member, scales=scales(), check=lambda: None, column_drop='store_and_subset', stats_population='selected_fit')
    views = np.flatnonzero(np.isin(w.s2.array('views/match'), member.fit))
    assert stats.column_drop == 'store_and_subset' and stats.population == 'selected_fit' and stats.n_views == len(views)
    assert stats.dropped == features.dropped_columns(w.s2, views, pooled=True)              # the list of the B1 of the same run
    fit_views = np.flatnonzero(features.split_views(w.s1, w.s2, 'FIT'))
    assert stats.dropped_store == features.dropped_columns(w.s2, fit_views, pooled=True)    # the D7 list is judged over the whole FIT store
    assert 'atakhan_in_patch' in stats.dropped_store and 'dragon_up' in stats.dropped and 'dragon_up' not in stats.dropped_store
    assert receipt['column_drop'] == 'store_and_subset' and receipt['stats_population'] == 'selected_fit' and receipt['stats_sha256'] == stats.sha256
    assert receipt['dropped_store'] == list(stats.dropped_store) and receipt['dropped'] == list(stats.dropped)
    assert training.fit.stats is stats and training.stop.stats is stats
    # the store-only drop leaves the columns that are constant only inside the selection
    only, store_stats, _ = data.build_neural_data(w, member, scales=scales(), check=lambda: None, column_drop='store', stats_population='selected_fit')
    assert store_stats.dropped == store_stats.dropped_store == stats.dropped_store
    # a supplied Stats must have been made under the same settings: the default ones, or the other ones, are refused
    plain = data.fit_stats_subset(w.s1, w.s2, member, lambda: None)
    for given, kw in ((plain, dict(column_drop='store_and_subset', stats_population='selected_fit')), (stats, {}), (store_stats, dict(column_drop='store_and_subset',
                                                                                                                                         stats_population='selected_fit')),
                      (stats, dict(column_drop='store_and_subset'))):
        with pytest.raises(ValueError, match='column_drop|population'):
            data.build_neural_data(w, member, scales=scales(), check=lambda: None, stats=given, **kw)
    again = data.build_neural_data(w, member, scales=scales(), check=lambda: None, stats=stats, column_drop='store_and_subset', stats_population='selected_fit')
    assert again[1] is stats and again[2]['stats_fitted_here'] is False


def test_fit_stats_subset_reads_the_whole_store_only_for_the_list_and_the_default_does_not():
    w, member = flagged_world()
    guards = guard_world(w, member, ('FIT',))
    data.fit_stats_subset(w.s1, w.s2, member, lambda: None)                                  # the default: only the selected FIT views are read
    assert all(set(g.reads) <= set(np.flatnonzero(np.isin(w.s2.array('views/match'), member.fit))) for g in guards)
    with pytest.raises(AssertionError, match='nonselected'):                                  # the D7 list is judged over the whole FIT store: other FIT matches are read
        data.fit_stats_subset(w.s1, w.s2, member, lambda: None, column_drop='store', stats_population='selected_fit')


def test_b1_subset_takes_the_kill_law_and_the_k_condition_and_records_them(monkeypatch):
    w = world()
    member = data.select_membership(w.s1)
    seen = spy_fit_head(monkeypatch)
    result = data.fit_b1_subset(w.s1, w.s2, w.labels, member, check=lambda: None, kill_law='shifted', k_condition='binned')
    assert len(seen) == 13 and all(f.kill_law == s.kill_law == 'shifted' and f.k_condition == s.k_condition == 'binned' for f, s, _, _ in seen)
    reps = seen[0][0].k_reps_sha256()
    assert result.kill_law == 'shifted' and result.k_condition == 'binned' and result.k_reps_sha256 == reps and len(reps) == 64
    receipt = result.comparison_receipt
    assert receipt['kill_law'] == 'shifted' and receipt['k_condition'] == 'binned' and receipt['k_reps_sha256'] == reps
    seen.clear()
    default = data.fit_b1_subset(w.s1, w.s2, w.labels, member, check=lambda: None)
    assert all(f.kill_law == 'poisson' and f.k_condition == 'raw' for f, _, _, _ in seen)
    assert default.kill_law == 'poisson' and default.k_condition == 'raw' and default.k_reps_sha256 is None
    assert not {'kill_law', 'k_condition', 'k_reps_sha256', 'column_drop', 'dropped_store', 'column_drop_scope'} & set(default.comparison_receipt)      # the receipt of the earlier runs
    for bad in ({'kill_law': 'bogus'}, {'k_condition': 'bogus'}, {'column_drop': 'bogus'}):
        fresh = world()
        with pytest.raises(ValueError):
            data.fit_b1_subset(fresh.s1, fresh.s2, fresh.labels, member, check=lambda: None, **bad)
        assert not fresh.labels.reads and not fresh.s2.reads


def test_b1_subset_uses_the_list_of_the_epts_when_a_column_drop_is_asked_for(monkeypatch):
    w, member = flagged_world()
    seen = spy_fit_head(monkeypatch)
    views = np.flatnonzero(np.isin(w.s2.array('views/match'), member.fit))
    fit_views = np.flatnonzero(features.split_views(w.s1, w.s2, 'FIT'))
    store = features.dropped_columns(w.s2, fit_views, pooled=True)
    both = data.fit_b1_subset(w.s1, w.s2, w.labels, member, check=lambda: None, column_drop='store_and_subset')
    assert both.dropped == features.dropped_columns(w.s2, views, pooled=True) and set(store) <= set(both.dropped) and 'dragon_up' in both.dropped
    assert 'old_champion_kill_b' in features.dropped_columns(w.s2, views) and 'old_champion_kill_b' not in both.dropped        # the pair is judged together
    only = data.fit_b1_subset(w.s1, w.s2, w.labels, member, check=lambda: None, column_drop='store')
    assert only.dropped == store and 'dragon_up' not in only.dropped
    assert only.comparison_receipt['column_drop'] == 'store' and only.comparison_receipt['dropped'] == list(store)
    neural_scope = data.build_neural_data(w, member, scales=scales(), check=lambda: None, column_drop='store', stats_population='selected_fit')[2]['column_drop_scope']
    assert only.comparison_receipt['column_drop_scope'] == neural_scope and 'column_drop_scope' in both.comparison_receipt         # one text for both receipts (review m3)
    assert both.comparison_receipt['column_drop'] == 'store_and_subset' and both.comparison_receipt['dropped_store'] == list(store)
    assert all(tuple(f.dropped) == tuple(s.dropped) == both.dropped for f, s, _, _ in seen[:13])
    neural_stats = data.fit_stats_subset(w.s1, w.s2, member, lambda: None, column_drop='store_and_subset', stats_population='selected_fit')
    assert neural_stats.dropped == both.dropped and neural_stats.dropped_store == store           # one list for the EPT and for B1 (decision D7)
    plain = data.fit_b1_subset(w.s1, w.s2, w.labels, member, check=lambda: None)
    assert plain.dropped == features.dropped_columns(w.s2, views)                                  # the default is the list of the earlier runs, judged one column at a time


def test_the_neural_data_take_the_swap_mode_for_the_fit_batches_only_and_the_default_is_the_earlier_data(tmp_path):
    """S4b preparation M7 review follow-up (D10 P1): ``swap_mode`` goes to the FIT batches only (evaluation batches never swap at random), the default gives the batches and the
    receipt of the earlier runs bit for bit, a mode that is not the default is named in the receipt, and 'random_view' batches pass the check of ``train`` against the settings."""
    import inspect
    import torch
    from c_ept.ept import encoder, train
    w = world()
    member = data.select_membership(w.s1)
    assert inspect.signature(data.build_neural_data).parameters['swap_mode'].default == 'average'
    plain, stats, receipt = data.build_neural_data(w, member, scales=scales(), check=lambda: None)
    explicit, _, same = data.build_neural_data(w, member, scales=scales(), check=lambda: None, stats=stats, swap_mode='average')
    _, _, receipt = data.build_neural_data(w, member, scales=scales(), check=lambda: None, stats=stats)       # the same call with the statistics supplied (the receipt says whether they were fitted here)
    assert same == receipt and 'swap_mode' not in receipt and plain.fit.swap_mode == explicit.fit.swap_mode == 'average'
    for a, b in zip(plain.fit.epoch(0), explicit.fit.epoch(0)):
        assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a) and 'swap_view' not in a
    cfg = train.TrainConfig(encoder.EncoderConfig(d=16, layers=1, heads=2), 1e-3, 0.0, 1, swap_mode='random_view')
    swapped, _, named = data.build_neural_data(w, member, scales=scales(), check=lambda: None, stats=stats, swap_mode='random_view')
    assert swapped.fit.swap_mode == 'random_view' and swapped.stop.swap_mode == 'average' and named['swap_mode'] == 'random_view'
    assert {k: v for k, v in named.items() if k != 'swap_mode'} == receipt
    batches = list(swapped.fit.epoch(0))
    assert batches and all(b['swap_view'].dtype == torch.bool and len(b['swap_view']) == len(b['view']) for b in batches)
    assert all('swap_view' not in b for b in swapped.stop.epoch(0))
    from pathlib import Path
    run = lambda d, c, folder: train.train(c, d, Path(folder), device='cpu', epochs=1, check=lambda: None, seed=0)       # noqa: E731
    with pytest.raises(ValueError, match='swap_mode'):                                          # the batches of the default are not made for the settings
        run(plain, cfg, tmp_path / 'refused')
    with pytest.raises(ValueError, match=r'data\.static'):                                      # the swapped batches pass the swap check; this toy world's static grid is the next (unrelated) refusal
        run(swapped, cfg, tmp_path / 'run')
    with pytest.raises(ValueError, match='swap_mode'):
        data.build_neural_data(w, member, scales=scales(), check=lambda: None, stats=stats, swap_mode='Random')
