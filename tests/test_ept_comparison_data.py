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
