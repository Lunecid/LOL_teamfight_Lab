"""Tests of ``c_ept.value``: the victim level of every kill, the f0 value of every engagement unit, V_start, the winner, the accounting
identity and the label store (stage S3 Task 4).

No run folder, raw file or catalog is read: every store is a toy one under pytest's folders. The S1 stores go through the real S1
extraction (``s3_toys.build_value_s1``: synthetic MatchRecords sent through ``run_s1.build_store``), so the units, the attribution and
conversion of the objectives, the onsets and the frame rows are the ones S1 makes. Truth comes from three places: numbers worked out by
hand for the toy matches (the ``TOY_*`` constants of ``s3_toys``, with the working in its comments), a brute-force second implementation
written here from the text of the rules (``oracle_*``: one loop per kill and per objective, plain Python, the scalar ``assist_pool``, the
level from the raw LEVEL_UP events), which is compared on the toy matches and on random ones, and refusals of inputs that cannot be
trusted (a victim level of -1 in a valued unit, a price table with a missing group, a store that is not consistent).

The seven tests of the task brief come first, word for word. The others pin the price table, the gold bins and their representatives,
the winner and the sure stratum (ruling R12), the accounting identity (ruling R11), V_start against the onsets, the label store, the
victim levels with the process pool and the E6 guard, and the imports of the module.
"""
import ast
import dataclasses
import gc
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from c_ept import config as C
from c_ept.valuation import flowprice
from c_ept.data import frame_rows, reader
from c_ept.valuation import rulepay
from c_ept.workflows import run_s1
from c_ept.domain import targets, units
from c_ept.valuation import value
from c_ept.data.store_core import Store
from c_ept.tests import synth
from c_ept.tests.conftest import PROTECTED_IMPORT_PRELUDE, empty_process_list, import_watch_names, module_level_imports
from c_ept.tests.s3_toys import (   # noqa: F401  (the fixtures are imported by ruling R1)
    KEY_PREFIX, TOY_ROW_OFF, TOY_EXEC_U, TOY_DRAW_U, TOY_F0, TOY_F0_CONV, TOY_F0_UNIT, TOY_GOLD, TOY_GOLD_BIN3_MEAN, TOY_GOLD_BINS,
    TOY_GOLD_OVERFLOW_MEAN, TOY_KILL_COUNTS, TOY_KILL_GOLD_TOTAL, TOY_LEVEL_AFTER_UP, TOY_LEVELS, TOY_N_KILLS, TOY_N_OBJECTIVES,
    TOY_N_ROWS, TOY_N_UNITS, TOY_OBJ_VALUE_TOTAL, TOY_OBJECTIVE_COUNTS, TOY_PRICES, TOY_SURE, TOY_TOWER_U, TOY_U, TOY_UNIT_VALUES,
    TOY_V_START, TOY_VALUED_UNITS, build_value_s1, level_up, random_value_matches, random_value_record, toy_key, toy_prices,
    toy_record_with_level_up_at_kill, toy_rules, toy_uv, toy_value_s1, value_kill, value_obj, value_record_0, value_record_1,
    value_toy_matches, value_toy_records, value_toy_records_that_leak_a_key)

REPO = Path(C.__file__).parents[1]
GROUPS = ('dragon', 'elder', 'grubs', 'herald', 'baron', 'atakhan', 'tower_outer', 'tower_inner', 'tower_base', 'tower_nexus', 'inhibitor',
          'plate')
TIERS = ('tower_outer', 'tower_inner', 'tower_base', 'tower_nexus')
S3_ROLES = ('FIT', 'STOP', 'DEV')
KINDS_OF_GROUP = ((0, -1), (1, -1), (2, -1), (3, -1), (4, -1), (5, -1), (6, 0), (6, 1), (6, 2), (6, 3), (7, -1), (8, -1))   # (kind, tier)


@pytest.fixture(scope='module')
def toy_s1(toy_value_s1):
    """The name the task brief uses: the S1 store of the value toys, made by the real extraction (it has the units and the onsets that
    the ``toy_s1`` of ``s3_toys`` lacks)."""
    return toy_value_s1


def message_of(raised) -> str:
    """The text of a caught error. The toy stores hold sentinel match keys (``KEY_PREFIX``); none may get into a message."""
    assert KEY_PREFIX not in str(raised.value) and KEY_PREFIX not in repr(raised.value)
    return str(raised.value)


def sum_over_onsets(s1, uv, bins):
    """Per frame row: f0(unit) + f0(conversions) summed over the ``onsets`` of the row with bin < ``bins``; NaN for the rows of a match
    that has no labels (a role stage S3 does not value)."""
    row, unit, bin_ = (np.asarray(s1.array(f'onsets/{n}')).astype(np.int64) for n in ('row', 'unit', 'bin'))
    keep = bin_ < bins
    out = np.bincount(row[keep], weights=(uv.f0_unit + uv.f0_conv)[unit[keep]], minlength=len(s1.array('frame_rows/t0')))
    out[~uv.match_valued[np.asarray(s1.array('frame_rows/match'))]] = np.nan
    return out


# ---- the seven tests of the task brief ---------------------------------------------------------------------------------------------------
def test_unit_f0_hand_example(toy_s1, toy_prices, toy_rules):
    uv = value.unit_values(toy_s1, toy_prices, toy_rules, TOY_LEVELS)
    assert uv.f0_unit[TOY_U] == pytest.approx(TOY_F0) and uv.winner[TOY_U] == np.sign(TOY_F0)


def test_execution_and_draw_units(toy_s1, toy_prices, toy_rules):
    uv = value.unit_values(toy_s1, toy_prices, toy_rules, TOY_LEVELS)
    assert uv.kill_gold[TOY_EXEC_U].sum() == 0 and uv.winner[TOY_DRAW_U] == 0


def test_towers_use_their_tier_price(toy_s1, toy_prices, toy_rules):
    uv = value.unit_values(toy_s1, toy_prices, toy_rules, TOY_LEVELS)
    assert uv.obj_value[TOY_TOWER_U].sum() == pytest.approx(toy_prices['tower_inner'])


def test_gold_bins_and_bin_means():
    assert list(value.gold_bin(np.array([0, 99, 100, 2999, 3000, 9999]))) == [0, 0, 1, 29, 30, 30]


def test_v_start_matches_onsets(toy_s1, toy_uv):
    assert np.allclose(value.v_start(toy_s1, toy_uv)[:, 0], sum_over_onsets(toy_s1, toy_uv, bins=6), equal_nan=True)


def test_accounting_identity_exact(toy_s1, toy_uv, toy_prices, toy_rules):
    assert value.accounting(toy_s1, toy_uv, toy_prices, toy_rules, TOY_LEVELS)['max_abs_error'] == 0.0


def test_kill_levels_count_level_up_at_same_ms():                    # a LEVEL_UP on the ms of the kill
    assert value.kill_levels_for_record(toy_record_with_level_up_at_kill())[0] == TOY_LEVEL_AFTER_UP


# ---- the toy world --------------------------------------------------------------------------------------------------------------------
def test_the_toy_store_has_the_shape_the_hand_numbers_assume(toy_s1):
    assert len(toy_s1.array('units/f')) == TOY_N_UNITS and len(toy_s1.array('kills/t')) == TOY_N_KILLS
    assert len(toy_s1.array('frame_rows/t0')) == TOY_N_ROWS and len(toy_s1.array('objectives/t')) == TOY_N_OBJECTIVES
    assert np.asarray(toy_s1.array('matches/row_off')).tolist() == list(TOY_ROW_OFF)
    assert [C.ROLES[i] for i in np.asarray(toy_s1.array('matches/role'))] == ['FIT', 'STOP', 'DEV', 'CAL']
    assert len(TOY_LEVELS) == TOY_N_KILLS and TOY_LEVELS.dtype == np.int16


def test_the_hand_worked_levels_agree_with_the_raw_events_of_the_toy_matches():
    for m, builder, patch, lo in ((0, value_record_0, '15.18', 0), (1, value_record_1, '15.20', 9)):
        rec = builder(toy_key(m), patch)
        kt = units.kill_table(rec)
        want = [level_of(rec, int(v), int(t)) for t, v in zip(kt.t, kt.victim)]
        assert want == TOY_LEVELS[lo:lo + len(want)].tolist() and len(want) == (9 if m == 0 else 4)


# ---- oracles written from the text of the rules ---------------------------------------------------------------------------------------
def group_name(kind: int, sub: int):
    """The price group of an objective by its kind and tower tier, written out here (not read from ``flowprice``); None: no group."""
    if not 0 <= kind < len(C.O_KINDS):
        return None
    name = C.O_KINDS[kind]
    if name == 'tower':
        return TIERS[sub] if 0 <= sub <= 3 else None
    return name


def level_of(rec, pid: int, t: int) -> int:
    """The level of player ``pid`` at ``t``: the highest LEVEL_UP level at or before ``t`` (the ms included), 1 if there is none."""
    return max([e['level'] for e in rec.events if e.get('type') == 'LEVEL_UP' and e['participantId'] == pid and e['timestamp'] <= t],
               default=1)


def oracle_match(rec, prices, rules):
    """``(per unit dicts, unit table)`` of one match, in the order of the S1 units: the kill gold (bounty + shutdown + pool, one kill at a
    time with the scalar ``assist_pool``), the champion kills, the attributed and the converted objective value, each by team. The
    partition of the kills and objectives into units comes from the same ``units`` / ``targets`` functions that S1 uses."""
    kt = units.kill_table(rec)
    u = units.build_units(kt, rec.team, rec.T_ms)
    obj = targets.objective_table(rec, u, kt)
    _, _, conv_unit = targets.conversions(obj, u, rec.T_ms)
    side = {pid: 0 if rec.team[pid - 1] == 100 else 1 for pid in range(1, 11)}
    out = []
    for i, comp in enumerate(u.comp):
        d = dict(kill_gold=[0.0, 0.0], n_kills=[0, 0], obj_value=[0.0, 0.0], conv_value=[0.0, 0.0])
        for r in comp:
            killer = int(kt.killer[r])
            if not 1 <= killer <= 10:
                continue                                                                 # an execution pays nobody
            t, victim = int(kt.t[r]), int(kt.victim[r])
            n_assisters = bin(int(kt.amask[r]) & 0x3FF).count('1')
            pool = rulepay.assist_pool(int(kt.bounty[r]), level_of(rec, victim, t), t, rec.patch, n_assisters, rules)
            d['kill_gold'][side[killer]] += int(kt.bounty[r]) + int(kt.shutdown[r]) + pool
            d['n_kills'][side[killer]] += 1
        for j in range(len(obj.t)):
            taker = int(obj.taker[j])
            if taker not in (0, 1):
                continue
            name = group_name(int(obj.kind[j]), int(obj.sub[j]))
            price = prices[name] if name else 0.0
            if int(obj.unit[j]) == i:
                d['obj_value'][taker] += price
            if int(conv_unit[j]) == i:
                d['conv_value'][taker] += price
        out.append(d)
    return out, u


def oracle_world(matches, records, prices, rules, roles=S3_ROLES, horizons=C.EVAL_H_MS):
    """The labels of the matches [(patch, key, macro)] by the brute-force way: per-unit dicts (None for a match whose role is not in
    ``roles``), the unit tables and the V_start of every frame row (NaN for such a match)."""
    per_match, rows = [], []
    for patch, key, macro in matches:
        rec = records(key, patch)
        fr = frame_rows.frame_rows(rec)
        if macro not in roles:
            per_match.append((None, units.build_units(units.kill_table(rec), rec.team, rec.T_ms)))
            rows.append(np.full((len(fr.t0), len(horizons)), np.nan))
            continue
        dicts, u = oracle_match(rec, prices, rules)
        per_match.append((dicts, u))
        val = [(d['kill_gold'][0] + d['obj_value'][0]) - (d['kill_gold'][1] + d['obj_value'][1]) + d['conv_value'][0] - d['conv_value'][1]
               for d in dicts]
        v = np.zeros((len(fr.t0), len(horizons)))
        for r, (t0, hi) in enumerate(zip(fr.t0.tolist(), fr.hi.tolist())):
            for j, h in enumerate(horizons):
                end = min(t0 + h, hi)
                v[r, j] = sum(val[i] for i in range(len(val)) if u.in_scope[i] and t0 < int(u.f[i]) <= end)
        rows.append(v)
    return per_match, np.concatenate(rows)


def assert_unit_values_equal_oracle(uv, per_match, tol=1e-9):
    i = 0
    for dicts, u in per_match:
        for k in range(len(u.f)):
            if dicts is None:
                assert not uv.valued[i] and np.isnan(uv.kill_gold[i]).all() and np.isnan(uv.f0_unit[i])
                assert uv.winner[i] == 0 and not uv.sure[i] and (uv.gold_bin[i] == -1).all() and (uv.n_kills[i] == 0).all()
            else:
                d = dicts[k]
                assert uv.valued[i]
                for name in ('kill_gold', 'obj_value', 'conv_value'):
                    assert getattr(uv, name)[i].tolist() == pytest.approx(d[name], abs=tol), (i, name)
                f0 = (d['kill_gold'][0] + d['obj_value'][0]) - (d['kill_gold'][1] + d['obj_value'][1])
                assert uv.f0_unit[i] == pytest.approx(f0, abs=tol) and uv.f0_conv[i] == pytest.approx(
                    d['conv_value'][0] - d['conv_value'][1], abs=tol)
                assert uv.n_kills[i].tolist() == d['n_kills']
                bins = [min(int(g // 100), 30) if n else -1 for g, n in zip(d['kill_gold'], d['n_kills'])]
                assert uv.gold_bin[i].tolist() == bins
                rounded = round(f0, 6)
                assert uv.winner[i] == (rounded > 0) - (rounded < 0) and bool(uv.sure[i]) == (abs(rounded) >= 150)
            i += 1
    assert i == len(uv.f0_unit)


# ---- the price table ---------------------------------------------------------------------------------------------------------------
def write_prices(tmp_path, values=None, name='PRICES.json', extra=None, mutate=None):
    """A toy PRICES.json: one entry per group with ``value`` and ``used`` and the other top-level keys of the real file."""
    values = dict(TOY_PRICES if values is None else values)
    doc = {g: {'value': v, 'used': 'immediate', 'price_flow': None, 'immediate': v} for g, v in values.items()}
    doc.update({'kill_gold_coef': {'kill_gold_coef': 1.12}, 'probe': {'found': False}, 'B': 2000, 'seed': 20261021, 'n_matches': 19100,
                'provenance': {'code_commit': 'x'}})
    doc.update(extra or {})
    if mutate is not None:
        mutate(doc)
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding='utf-8')
    return path


def test_load_prices_reads_the_value_of_every_group(tmp_path):
    prices = value.load_prices(write_prices(tmp_path))
    assert list(prices) == list(GROUPS) and prices == TOY_PRICES
    assert all(type(v) is float for v in prices.values())


def test_load_prices_turns_whole_numbers_into_floats(tmp_path):
    prices = value.load_prices(write_prices(tmp_path, {g: 100 + i for i, g in enumerate(GROUPS)}))
    assert prices['plate'] == 111.0 and type(prices['plate']) is float


@pytest.mark.parametrize('mutate', [
    lambda d: d.pop('baron'),
    lambda d: d['baron'].pop('value'),
    lambda d: d['baron'].update(value=None),
    lambda d: d['baron'].update(value=float('nan')),
    lambda d: d['baron'].update(value=float('inf')),
    lambda d: d['baron'].update(value='1830'),
    lambda d: d['baron'].update(value=True),
    lambda d: d['baron'].update(value=[1830.0]),
    lambda d: d.update(baron=1830.0),
    lambda d: d.update(baron=None)])
def test_load_prices_refuses_a_missing_group_or_a_value_that_is_not_a_number(tmp_path, mutate):
    with pytest.raises(ValueError, match='baron'):
        value.load_prices(write_prices(tmp_path, mutate=mutate))


def test_load_prices_says_when_a_price_is_none(tmp_path):
    with pytest.raises(ValueError, match='baron is None'):
        value.load_prices(write_prices(tmp_path, mutate=lambda d: d['baron'].update(value=None)))


def test_load_prices_refuses_a_file_that_is_not_a_price_table(tmp_path):
    for text in ('[1, 2]', '"x"', '{ not json'):
        p = tmp_path / 'bad.json'
        p.write_text(text, encoding='utf-8')
        with pytest.raises(ValueError):
            value.load_prices(p)


def test_load_prices_ignores_the_other_keys_of_the_file(tmp_path):
    assert value.load_prices(write_prices(tmp_path, extra={'unknown_extra': {'value': 1}})) == TOY_PRICES


def test_object_price_maps_every_group_by_kind_and_tier(toy_prices):
    kind, tier = np.array([k for k, _ in KINDS_OF_GROUP]), np.array([s for _, s in KINDS_OF_GROUP])
    got = value.object_price(kind, tier, toy_prices)
    assert got.dtype == np.float64 and got.tolist() == [toy_prices[g] for g in GROUPS]


def test_object_price_gives_zero_where_the_group_is_not_known(toy_prices):
    #              another monster, a tower without a tier, a tower with a tier that does not exist, a kind that does not exist, a negative
    kind, tier = np.array([-1, 6, 6, 9, 100, 0]), np.array([-1, -1, 4, 0, 0, 7])
    assert value.object_price(kind, tier, toy_prices).tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, toy_prices['dragon']]


def test_object_price_reads_the_tier_of_a_tower_only(toy_prices):
    assert value.object_price(np.array([4, 4, 7, 8]), np.array([3, 0, 2, 1]), toy_prices).tolist() == [
        toy_prices['baron'], toy_prices['baron'], toy_prices['inhibitor'], toy_prices['plate']]


def test_object_price_works_on_single_numbers_and_on_narrow_dtypes(toy_prices):
    assert float(value.object_price(6, 1, toy_prices)) == toy_prices['tower_inner']
    assert value.object_price(np.array([6], dtype=np.int8), np.array([2], dtype=np.int8), toy_prices).tolist() == [toy_prices['tower_base']]
    assert value.object_price(np.zeros((2, 3), dtype=int), np.full((2, 3), -1), toy_prices).shape == (2, 3)


def test_object_price_refuses_a_price_table_that_is_not_complete(toy_prices):
    for broken in ({k: v for k, v in toy_prices.items() if k != 'plate'}, {**toy_prices, 'plate': None},
                   {**toy_prices, 'plate': float('nan')},
                   {**toy_prices, 'plate': 'x'}, [1.0] * 12):
        with pytest.raises((ValueError, TypeError)):
            value.object_price(np.array([8]), np.array([-1]), broken)


# ---- gold bins ------------------------------------------------------------------------------------------------------------------------
def test_gold_edges_are_the_lower_edges_of_31_bins():
    assert value.N_GOLD_BINS == 31 and len(value.GOLD_EDGES) == 31
    assert value.GOLD_EDGES.tolist() == [100.0 * b for b in range(31)]
    assert value.gold_bin(value.GOLD_EDGES).tolist() == list(range(31))                  # an edge belongs to the bin that starts there


def test_gold_bin_cuts_at_the_hundreds_and_the_last_bin_is_open():
    gold = np.array([0.0, 0.01, 99.99, 100.0, 199.999999, 200.0, 2899.9, 2900.0, 2999.99, 3000.0, 3000.01, 5000.0, 1e9])
    assert value.gold_bin(gold).tolist() == [0, 0, 0, 1, 1, 2, 28, 29, 29, 30, 30, 30, 30]


def test_gold_bin_is_int8_and_keeps_the_shape():
    out = value.gold_bin(np.array([[0, 100], [3000, 250]]))
    assert out.dtype == np.int8 and out.shape == (2, 2) and out.tolist() == [[0, 1], [30, 2]]
    assert int(value.gold_bin(450.0)) == 4


def test_gold_bin_refuses_gold_that_cannot_be_binned():
    for bad in ([-0.5], [np.nan], [np.inf], [100.0, -1.0]):
        with pytest.raises(ValueError):
            value.gold_bin(np.array(bad))


def test_the_bin_of_each_toy_gold_of_note_03_section_d():
    assert value.gold_bin(np.array(TOY_GOLD)).tolist() == list(TOY_GOLD_BINS)


# ---- the bin means ---------------------------------------------------------------------------------------------------------------------
def fake_uv(kill_gold, n_kills):
    """A ``UnitValues`` made from kill gold (n, 2) and champion kills (n, 2) alone: the other fields are neutral."""
    kill_gold, n_kills = np.asarray(kill_gold, dtype=np.float64).reshape(-1, 2), np.asarray(n_kills, dtype=np.int32).reshape(-1, 2)
    n = len(kill_gold)
    bins = np.where(n_kills >= 1, value.gold_bin(np.where(n_kills >= 1, kill_gold, 0.0)), -1).astype(np.int8)
    zero = np.zeros(n)
    return value.UnitValues(kill_gold=kill_gold, obj_value=np.zeros((n, 2)), conv_value=np.zeros((n, 2)), f0_unit=zero, f0_conv=zero.copy(),
                            winner=np.zeros(n, dtype=np.int8), sure=np.zeros(n, dtype=bool), gold_bin=bins, n_kills=n_kills,
                            valued=np.ones(n, dtype=bool), match_valued=np.ones(1, dtype=bool))


def test_gold_bin_means_returns_the_means_and_the_fallback_list_by_name():
    uv = fake_uv([[300.0, 0.0]], [[1, 0]])
    reps = value.gold_bin_means(uv, np.arange(1))
    assert isinstance(reps, value.GoldReps) and reps.means is reps[0] and reps.fallback is reps[1]
    assert isinstance(reps.fallback, tuple) and 3 not in reps.fallback and len(reps.fallback) == 30


def test_bin_means_of_note_03_section_d():
    uv = fake_uv([[g, 0.0] for g in TOY_GOLD], [[1, 0]] * 6)                       # one team of each unit has the kill gold, the other none
    means, fallback = value.gold_bin_means(uv, np.arange(6))
    assert means.shape == (31,) and means.dtype == np.float64
    assert means[3] == pytest.approx(TOY_GOLD_BIN3_MEAN) and means[4] == 450.0 and means[30] == pytest.approx(TOY_GOLD_OVERFLOW_MEAN)
    assert sorted(set(range(31)) - set(fallback)) == [3, 4, 30]                         # the bins that had no entry are listed


def test_an_empty_bin_takes_its_midpoint_and_the_open_bin_takes_3000():
    uv = fake_uv([[300.0, 0.0], [450.0, 0.0]], [[1, 0], [1, 0]])
    means, fallback = value.gold_bin_means(uv, np.arange(2))
    assert means[0] == 50.0 and means[1] == 150.0 and means[29] == 2950.0 and means[30] == 3000.0
    assert means[3] == 300.0 and means[4] == 450.0
    assert list(fallback) == [b for b in range(31) if b not in (3, 4)]


def test_a_team_without_kills_is_not_an_entry_but_a_team_with_zero_gold_is():
    uv = fake_uv([[0.0, 700.0], [0.0, 0.0]], [[1, 1], [0, 0]])                  # unit 0: blue has a kill worth 0 gold, red one worth 700
    means, fallback = value.gold_bin_means(uv, np.arange(2))
    assert means[0] == 0.0 and means[7] == 700.0 and 0 not in fallback and 7 not in fallback and len(fallback) == 29


def test_bin_means_use_only_the_selected_units_and_reproduce_their_gold_sum():
    gold = np.array([[310.0, 120.0], [390.0, 130.0], [3500.0, 140.0], [3900.0, 150.0], [330.0, 160.0]])
    uv = fake_uv(gold, np.ones((5, 2)))
    select = np.array([True, True, False, True, False])                                  # a boolean mask over the units
    means, fallback = value.gold_bin_means(uv, select)
    assert means[3] == pytest.approx((310.0 + 390.0) / 2) and means[1] == pytest.approx((120.0 + 130.0 + 150.0) / 3) and means[30] == 3900.0
    ordered, same = value.gold_bin_means(uv, np.flatnonzero(select))                    # the same units as indices
    assert np.array_equal(ordered, means) and list(same) == list(fallback)
    counts = np.bincount(uv.gold_bin[select].ravel(), minlength=31)
    assert float(np.sum(counts * means)) == pytest.approx(float(gold[select].sum()))    # the means reproduce the gold of the selected units


def test_bin_means_leave_out_a_unit_that_is_not_valued():
    uv = fake_uv([[300.0, 0.0], [np.nan, np.nan]], [[1, 0], [0, 0]])
    uv = dataclasses.replace(uv, valued=np.array([True, False]), gold_bin=np.array([[3, -1], [-1, -1]], dtype=np.int8))
    means, _ = value.gold_bin_means(uv, np.arange(2))
    assert means[3] == 300.0 and np.isfinite(means).all()


def test_bin_means_refuse_a_selection_that_does_not_fit():
    uv = fake_uv([[300.0, 0.0]] * 3, [[1, 0]] * 3)
    for bad in (np.array([True, False]), np.array([5]), np.array([-4]), np.array([0.5])):
        with pytest.raises(ValueError):
            value.gold_bin_means(uv, bad)


def test_the_bin_means_of_the_toy_fit_units(toy_s1, toy_uv):
    means, fallback = value.gold_bin_means(toy_uv, value.role_units(toy_s1, ('FIT',)))
    # the FIT team entries with a kill: unit 0 blue 403.125 (bin 4), unit 1 blue 1,186.875 (11) and red 450 (4), unit 3 300 and 300 (3),
    # unit 4 blue 300 (3), unit 5 blue 300 (3)
    assert means[3] == 300.0 and means[4] == pytest.approx((403.125 + 450.0) / 2) and means[11] == 1186.875
    assert sorted(set(range(31)) - set(fallback)) == [3, 4, 11]
    other, other_fallback = value.gold_bin_means(toy_uv, value.role_units(toy_s1, ('STOP', 'DEV')))
    # unit 6 red 575 and unit 7 blue 500 (bin 5), unit 8 blue 300 (bin 3) and red 930 (bin 9)
    assert other[5] == pytest.approx(537.5) and other[3] == 300.0 and other[9] == 930.0
    assert sorted(set(range(31)) - set(other_fallback)) == [3, 5, 9]


def test_role_units_selects_the_units_of_the_matches_of_the_asked_roles(toy_s1):
    assert np.flatnonzero(value.role_units(toy_s1, ('FIT',))).tolist() == [0, 1, 2, 3, 4, 5]
    assert np.flatnonzero(value.role_units(toy_s1, ('STOP', 'DEV'))).tolist() == [6, 7, 8]
    for roles in (('TEST',), ('EXTERNAL',), ('CAL',), ('NOPE',), ()):
        with pytest.raises(ValueError):
            value.role_units(toy_s1, roles)


# ---- the population of the bin means (S4b preparation M3, decision D21 B16) -------------------------------------------------------------
# The representatives are the FIT means of the gold bins. ``population='all'`` (the default) is every entry of the selected units, in scope or not, as in every earlier run (the S3
# as-built labels), ``'in_scope'`` the entries of the selected units that are in scope (``units/in_scope`` of S1) only. Hand values (five units, one entry each team):
#   gold (blue, red):  u0 (310, 120)  u1 (390, 130)  u2 (3500, 140)  u3 (3900, 150)  u4 (330, 160);   in scope: u0 u1 u3 (u2 and u4 lie outside the scope)
#   all five units:   bin 1 = (120 + 130 + 140 + 150 + 160) / 5 = 140,  bin 3 = (310 + 390 + 330) / 3 = 1030 / 3,  bin 30 = (3500 + 3900) / 2 = 3700
#   in-scope units:   bin 1 = (120 + 130 + 150) / 3 = 400 / 3,          bin 3 = (310 + 390) / 2 = 350,            bin 30 = 3900
SCOPE_GOLD = np.array([[310.0, 120.0], [390.0, 130.0], [3500.0, 140.0], [3900.0, 150.0], [330.0, 160.0]])
SCOPE_MASK = np.array([True, True, False, True, False])


def plain_bin_means(uv, selected) -> np.ndarray:
    """The bin means of (VAL-7) by a plain loop over the entries of the units ``selected`` (never the function under test): the exact sum of each bin divided by its count,
    the middle of the range (3,000 for the open bin) for a bin without an entry."""
    import math
    sums = {b: [] for b in range(31)}
    for u in selected:
        for t in (0, 1):
            if uv.gold_bin[u, t] >= 0:
                sums[int(uv.gold_bin[u, t])].append(float(uv.kill_gold[u, t]))
    return np.array([math.fsum(sums[b]) / len(sums[b]) if sums[b] else (100.0 * b + 50.0 if b < 30 else 3000.0) for b in range(31)])


def test_representatives_come_from_in_scope_fit_units():
    uv = fake_uv(SCOPE_GOLD, np.ones((5, 2)))
    fit = np.arange(5)                                                     # all five units are FIT units: the role selection
    default = value.gold_bin_means(uv, fit)
    scoped = value.gold_bin_means(uv, fit, population='in_scope', in_scope=SCOPE_MASK)
    assert default.means[1] == 140.0 and default.means[3] == pytest.approx(1030 / 3) and default.means[30] == 3700.0           # the hand values of the section text
    assert scoped.means[1] == pytest.approx(400 / 3) and scoped.means[3] == 350.0 and scoped.means[30] == 3900.0
    # the other bins are the middle of their range in both, and the open bin 30 has entries in both
    assert default.means[0] == 50.0 and scoped.means[0] == 50.0 and sorted(set(range(31)) - set(scoped.fallback)) == [1, 3, 30]
    # an out-of-scope unit changes the default representatives and does not change the in-scope ones, down to the bit and the fallback list
    other = SCOPE_GOLD.copy()
    other[2], other[4] = [5000.0, 995.0], [111.0, 222.0]                   # the gold of the two out-of-scope units: other bins (30, 9 and 1, 2)
    uv2 = fake_uv(other, np.ones((5, 2)))
    again = value.gold_bin_means(uv2, fit, population='in_scope', in_scope=SCOPE_MASK)
    assert np.array_equal(again.means, scoped.means) and again.fallback == scoped.fallback
    assert not np.array_equal(value.gold_bin_means(uv2, fit).means, default.means)
    # a unit that is in scope but not selected (not a FIT unit) is not an entry either: the population is the selected units that are in scope
    fewer = value.gold_bin_means(uv, np.array([0, 1, 2]), population='in_scope', in_scope=SCOPE_MASK)
    assert fewer.means[3] == 350.0 and fewer.means[1] == pytest.approx(125.0) and fewer.means[30] == 3000.0 and 30 in fewer.fallback


def test_the_default_population_is_every_selected_unit_as_in_every_earlier_run():
    """The default is the plain loop of (VAL-7) over the selected units, scope not asked, and the keyword at 'all' is the default. (The comparison with the code of the
    base commit is the before/after proof script; this test compares with a second implementation written from the text of the rule.)"""
    rng = np.random.default_rng(7)
    gold = rng.uniform(0.0, 3600.0, (40, 2))
    kills = rng.integers(0, 3, (40, 2))
    uv = fake_uv(gold, kills)
    for selected in (np.arange(40), np.arange(0, 40, 3), np.array([4, 5, 6])):
        want = plain_bin_means(uv, selected)
        for got in (value.gold_bin_means(uv, selected), value.gold_bin_means(uv, selected, population='all')):
            assert np.array_equal(got.means, want) and got.means.dtype == np.float64
    assert value.GOLD_REP_POPULATIONS == ('all', 'in_scope')
    import inspect
    parameters = inspect.signature(value.gold_bin_means).parameters
    assert [p for p in parameters] == ['uv', 'units', 'population', 'in_scope']
    assert parameters['population'].kind is inspect.Parameter.KEYWORD_ONLY and parameters['population'].default == 'all' and parameters['in_scope'].default is None


def test_the_population_of_the_bin_means_is_checked_before_any_entry_is_read():
    uv = fake_uv(SCOPE_GOLD, np.ones((5, 2)))
    fit = np.arange(5)
    for bad in ('in-scope', 'IN_SCOPE', 'selected', 'All', '', None, 3, ('in_scope',)):             # 'selected' was the name of 'all' in the first commit of this task: no alias
        with pytest.raises(ValueError, match='population'):
            value.gold_bin_means(uv, fit, population=bad)
        with pytest.raises(ValueError, match='population'):
            value.check_gold_rep_population(bad)
    assert value.check_gold_rep_population('in_scope') == 'in_scope' and value.check_gold_rep_population('all') == 'all'
    with pytest.raises(ValueError, match='in_scope'):                                      # the scope of the units must be given: it is not in the labels
        value.gold_bin_means(uv, fit, population='in_scope')
    for bad in (SCOPE_MASK[:3], SCOPE_MASK.astype(np.int64), np.flatnonzero(SCOPE_MASK), SCOPE_MASK.reshape(5, 1), 'yes'):
        with pytest.raises(ValueError, match='in_scope'):
            value.gold_bin_means(uv, fit, population='in_scope', in_scope=bad)
    with pytest.raises(ValueError, match='in_scope'):                                      # a scope mask is read by the in-scope population only: no silent no-op
        value.gold_bin_means(uv, fit, in_scope=SCOPE_MASK)


class ArrayStore:
    """A store of arrays by name (the ``array`` method of ``Store``): the part of S1 or of the labels that the representatives read."""

    def __init__(self, arrays: dict, base=None):
        self.arrays, self.base = arrays, base

    def array(self, name):
        return self.arrays[name] if name in self.arrays else self.base.array(name)


def test_the_in_scope_fit_representatives_of_the_toy_stores_leave_out_the_units_out_of_scope(toy_s1, toy_uv):
    """The FIT units 0 .. 5 of the toy (the gold of the units is in the comment of ``test_the_bin_means_of_the_toy_fit_units``); the scope is set by hand:
    unit 1 (blue 1,186.875 in bin 11, red 450 in bin 4) lies outside it. All FIT units: bin 4 = (403.125 + 450) / 2, bin 11 = 1,186.875. In scope: bin 4 = 403.125 and bin 11 has no entry
    (its middle 1,150). Bin 3 (units 3, 4, 5: 300 each) and the units of the other roles are the same in both."""
    labels = ArrayStore({'labels/gold_bin': toy_uv.gold_bin, 'labels/kill_gold': toy_uv.kill_gold})
    selected = value.fit_gold_bin_means(toy_s1, labels, population='all')
    assert np.array_equal(selected.means, value.gold_bin_means(toy_uv, value.role_units(toy_s1, ('FIT',))).means)               # the labels as they are: labels/gold_reps
    assert selected.means[4] == pytest.approx((403.125 + 450.0) / 2) and selected.means[11] == 1186.875
    scope = np.ones(TOY_N_UNITS, dtype=bool)
    scope[1] = False
    s1 = ArrayStore({'units/in_scope': scope}, toy_s1)
    scoped = value.fit_gold_bin_means(s1, labels, population='in_scope')
    assert scoped.means[4] == 403.125 and scoped.means[3] == 300.0 and scoped.means[11] == 1150.0 and 11 in scoped.fallback and 11 not in selected.fallback
    assert [b for b in range(31) if scoped.means[b] != selected.means[b]] == [4, 11]
    assert np.array_equal(value.in_scope_units(s1), scope) and value.in_scope_units(s1).dtype == np.bool_
    # the role selection is FIT alone: the units of STOP and DEV (6 .. 8) are never an entry of the representatives
    labels_other = ArrayStore({'labels/kill_gold': np.where(np.arange(TOY_N_UNITS)[:, None] >= 6, 99999.0, toy_uv.kill_gold)}, labels)
    assert np.array_equal(value.fit_gold_bin_means(s1, labels_other, population='in_scope').means, scoped.means)
    for bad in ('IN_SCOPE', None):
        with pytest.raises(ValueError, match='population'):
            value.fit_gold_bin_means(s1, labels, population=bad)
    with pytest.raises(ValueError, match='labels'):                                         # the labels of another store: counts only
        value.fit_gold_bin_means(s1, ArrayStore({'labels/gold_bin': toy_uv.gold_bin[:4], 'labels/kill_gold': toy_uv.kill_gold[:4]}), population='all')


def test_an_in_scope_population_without_any_entry_is_refused_with_counts_and_the_all_population_is_not():
    """Review m1: a scope mask that leaves no team entry with a kill in any bin would give all 31 representatives as midpoints; that is a wrong store or a wrong mask, not a
    population, so it is refused (counts only, no value of the data). The population 'all' keeps what every earlier run did: the midpoints and the full fallback list."""
    uv = fake_uv(SCOPE_GOLD, np.ones((5, 2)))
    fit = np.arange(5)
    for mask in (np.zeros(5, dtype=bool), np.array([False, False, False, False, False])):
        with pytest.raises(ValueError, match=r"no team entry with a kill in any gold bin \(0 of 5 units of the population 'all' are in scope\)") as raised:
            value.gold_bin_means(uv, fit, population='in_scope', in_scope=mask)
        assert '3500' not in str(raised.value) and '310' not in str(raised.value)
    # in-scope units that have no kill at all are no entry either
    quiet = fake_uv(SCOPE_GOLD, np.array([[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]]))
    with pytest.raises(ValueError, match=r"no team entry with a kill in any gold bin \(2 of 5 units of the population 'all' are in scope\)"):
        value.gold_bin_means(quiet, fit, population='in_scope', in_scope=np.array([True, True, False, False, False]))
    one = value.gold_bin_means(quiet, fit, population='in_scope', in_scope=np.array([True, True, True, False, False]))              # unit 2 only: bins 30 (3500) and 1 (140)
    assert sorted(set(range(31)) - set(one.fallback)) == [1, 30] and one.means[30] == 3500.0 and one.means[1] == 140.0
    nothing = fake_uv(SCOPE_GOLD, np.zeros((5, 2)))                                          # no kill anywhere: 'all' still gives the midpoints, as before
    everything = value.gold_bin_means(nothing, fit)
    assert everything.fallback == tuple(range(31)) and np.array_equal(everything.means, value.GOLD_FALLBACK)
    with pytest.raises(ValueError, match='no team entry with a kill in any gold bin'):
        value.gold_bin_means(nothing, fit, population='in_scope', in_scope=np.ones(5, dtype=bool))


def test_fit_gold_bin_means_refuses_an_in_scope_fit_population_without_any_entry(toy_s1, toy_uv):
    labels = ArrayStore({'labels/gold_bin': toy_uv.gold_bin, 'labels/kill_gold': toy_uv.kill_gold})
    nothing = ArrayStore({'units/in_scope': np.zeros(TOY_N_UNITS, dtype=bool)}, toy_s1)
    with pytest.raises(ValueError, match=r"no team entry with a kill in any gold bin \(0 of 6 units of the population 'all' are in scope\)"):           # the six FIT units of the toy
        value.fit_gold_bin_means(nothing, labels, population='in_scope')
    assert len(value.fit_gold_bin_means(nothing, labels, population='all').fallback) == 28                 # the same store, every FIT unit: the three bins of the toy have entries


def test_the_sha256_of_the_representatives_is_that_of_the_31_float64_numbers_little_endian_c_order():
    import hashlib
    import struct
    reps = np.arange(31, dtype=np.float64) * 100.0
    want = hashlib.sha256(struct.pack('<31d', *reps.tolist())).hexdigest()                    # 31 times 8 bytes, written out with struct (little-endian doubles)
    assert value.gold_reps_sha256(reps) == want and len(want) == 64
    assert value.gold_reps_sha256(reps.astype('>f8')) == want                                 # the byte order of the array is not the byte order of the hash
    assert value.gold_reps_sha256(np.asfortranarray(reps)) == want
    other = reps.copy()
    other[30] = np.nextafter(other[30], np.inf)                                               # one ulp of one representative changes it
    assert value.gold_reps_sha256(other) != want
    for bad in (reps[:30], reps.astype(np.int64), reps.reshape(31, 1), 'x', None):
        with pytest.raises(ValueError, match='gold_reps'):
            value.gold_reps_sha256(bad)


# ---- the winner and the sure stratum (ruling R12) --------------------------------------------------------------------------------------
def test_winner_is_the_sign_of_f0_rounded_to_a_millionth_of_a_gold():
    f0 = np.array([5.0, -5.0, 0.0, 1e-9, -1e-9, 4e-7, -4e-7, 6e-7, 1e-6, -1e-6, 5.55e-17])
    winner, sure = value.winner_of(f0)
    assert winner.dtype == np.int8 and sure.dtype == np.bool_
    assert winner.tolist() == [1, -1, 0, 0, 0, 0, 0, 1, 1, -1, 0]
    assert not sure.any()


def test_the_sure_stratum_starts_at_150_gold_either_way():
    winner, sure = value.winner_of(np.array([149.9, 149.9999996, 149.9999994, 149.999999, 150.0, 150.4, -149.9999996, -149.9999994, -150.0,
                                             -2000.0, 0.0]))
    # 149.9999996 rounds up to 150.000000 (sure), 149.9999994 rounds down to 149.999999 (not sure)
    assert sure.tolist() == [False, True, False, False, True, True, True, False, True, True, False]
    assert winner.tolist() == [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 0]


def test_a_sum_that_misses_zero_by_rounding_noise_is_a_draw():
    assert 0.1 + 0.2 - 0.3 > 0                                                          # the noise is there: np.sign alone would say blue
    assert value.winner_of(np.array([0.1 + 0.2 - 0.3]))[0].tolist() == [0]


def test_winner_of_refuses_f0_that_is_not_finite():
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            value.winner_of(np.array([1.0, bad]))


# ---- the truth of the winner endpoint under the tie rule (S4b-prep M2, decision D5 rule (i)) -----------------------------------------------
# The models decide the winner on the binned gold (each team's kill gold is the representative of its bin); under the rule 'binned' the gold term of the
# truth side is computed on the same variable, so two teams in one bin tie when their realized objective values are equal (the objective term is the
# realized attributed price of the labels; for a tower that is not the models' expected tier price). The twelve units of ``TIE_CASES`` are worked out
# by hand with the made-up representatives rep_b = 100 b + 40 (40, 140, ..., 3,040: no representative is the middle of its bin, so a mix-up of the bin edge, the bin
# middle and the representative changes a number). A team without a kill has the bin -1 and the gold 0.
TIE_REPS = 100.0 * np.arange(31) + 40.0
# (blue gold, red gold, blue objectives, red objectives, raw f0 worked out by hand, binned f worked out by hand); gold None: no kill
TIE_CASES = (
    (410.0, 450.0, 0.0, 0.0, -40.0, 0.0),                # 0: one bin (4): the 40 gold of difference is no difference of the representatives
    (410.0, 450.0, 30.0, 0.0, -10.0, 30.0),              # 1: the same bin, an objective of blue breaks the tie (the raw sign says red)
    (410.0, 450.0, 0.0, 25.0, -65.0, -25.0),             # 2: the same bin, an objective of red breaks the tie
    (610.0, 340.0, 0.0, 0.0, 270.0, 300.0),              # 3: bins 6 and 3: 640 - 340
    (None, 340.0, 0.0, 0.0, -340.0, -340.0),             # 4: blue has no kill: its gold is 0, not the representative of a bin
    (None, 340.0, 350.0, 0.0, 10.0, 10.0),               # 5: ... and its objectives of 350 are added as they are: 350 - 340
    (None, None, 0.0, 0.0, 0.0, 0.0),                    # 6: no kill on either side: a tie
    (None, None, 0.0, 200.0, -200.0, -200.0),            # 7: no kill, the objectives alone decide
    (300.0, 399.99, 0.0, 0.0, -99.99, 0.0),              # 8: the lowest gold of bin 3 and one just below its top: one bin, a tie
    (299.99, 300.0, 0.0, 0.0, -0.01, -100.0),            # 9: either side of the edge 300 (bins 2 and 3): 240 - 340, the whole step of 100
    (3000.0, 4500.0, 0.0, 0.0, -1500.0, 0.0),            # 10: the open bin 30 holds both: a tie
    (0.0, None, 0.0, 0.0, 0.0, 40.0),                    # 11: a kill worth 0 gold is a kill (bin 0, 40) and a team without a kill has 0
)


def tie_arrays():
    """The columns of ``TIE_CASES``: (f0, bins, obj, binned f by hand). The raw f0 is built the way ``unit_values`` builds it, the bins by ``gold_bin``."""
    gold = np.array([[0.0 if g is None else g for g in case[:2]] for case in TIE_CASES])
    has_kill = np.array([[g is not None for g in case[:2]] for case in TIE_CASES])
    obj = np.array([case[2:4] for case in TIE_CASES])
    bins = np.full(gold.shape, -1, dtype=np.int8)
    bins[has_kill] = value.gold_bin(gold[has_kill])
    f0 = (gold[:, 0] + obj[:, 0]) - (gold[:, 1] + obj[:, 1])
    return f0, bins, obj, np.array([case[5] for case in TIE_CASES])


def test_binned_truth_uses_bin_representatives():
    f0, bins, obj, want = tie_arrays()
    assert bins.tolist() == [[4, 4], [4, 4], [4, 4], [6, 3], [-1, 3], [-1, 3], [-1, -1], [-1, -1], [3, 3], [2, 3], [30, 30], [0, -1]]
    assert f0.tolist() == pytest.approx([case[4] for case in TIE_CASES])                # the hand-worked raw values are what the inputs hold
    binned = value.winner_truth(f0, bins, TIE_REPS, obj, 'binned')
    assert binned.dtype == np.float64 and binned.shape == (len(TIE_CASES),)
    assert binned.tolist() == want.tolist()                                              # exact: every term is a whole number of gold
    assert np.sign(binned).tolist() == [0, 1, -1, 1, -1, 1, 0, -1, 0, -1, 0, 1]          # blue +1, a tie 0, red -1
    gold_rule = value.winner_truth(f0, bins, TIE_REPS, obj, 'gold')
    assert np.sign(gold_rule).tolist() == [-1, -1, -1, 1, -1, 1, 0, -1, -1, -1, -1, 0]   # the raw sign decides the pairs of one bin (0, 1, 8, 10); a kill worth 0 gold against none is a raw tie (11)
    # the units where the two rules disagree are the ones the rule is about: a pair in one bin (0, 8, 10), an objective over a pair in one bin (1)
    # and a kill worth 0 gold against no kill (11)
    assert np.flatnonzero(np.sign(binned) != np.sign(gold_rule)).tolist() == [0, 1, 8, 10, 11]


def test_binned_truth_ties_two_teams_of_one_bin_an_objective_breaks_the_tie_and_the_bin_edge_is_a_whole_step():
    reps = TIE_REPS
    for blue, red in ((300.0, 399.99), (300.0, 300.0), (350.0, 351.0), (3000.0, 9000.0)):          # one bin each, at its edge and inside, and the open bin
        b = value.gold_bin(np.array([blue, red])).astype(np.int8)
        assert b[0] == b[1]
        f = value.winner_truth(np.array([blue - red]), b[None, :], reps, np.zeros((1, 2)), 'binned')
        assert f.tolist() == [0.0]
        for objective, sign in ((1.0, 1.0), (-1.0, -1.0)):                                       # the objectives of the pair decide: blue more, red more
            f = value.winner_truth(np.array([blue - red]), b[None, :], reps, np.array([[max(objective, 0.0), max(-objective, 0.0)]]), 'binned')
            assert np.sign(f).tolist() == [sign]
    edge = value.gold_bin(np.array([299.99, 300.0]))
    assert edge.tolist() == [2, 3]                                                               # the edge belongs to the bin above it
    assert value.winner_truth(np.array([-0.01]), edge[None, :], reps, np.zeros((1, 2)), 'binned').tolist() == [-100.0]


def test_binned_truth_equals_the_raw_f0_when_every_bin_holds_one_gold_value():
    """The two rules differ only through the discretisation: with a representative equal to the gold of its bin the sums are the same floats."""
    gold = np.array([[403.125, 520.0], [1186.875, 650.5], [300.0, 930.5], [3000.0, 1200.0], [0.0, 120.0]])
    obj = np.array([[0.0, 0.0], [410.0, 350.1], [0.1, 0.2], [0.3, 0.0], [0.7, 0.7]])
    bins = value.gold_bin(gold)
    assert bins.tolist() == [[4, 5], [11, 6], [3, 9], [30, 12], [0, 1]]                         # ten different bins: no bin holds two golds
    reps = np.full(31, 77.0)
    reps[bins.ravel()] = gold.ravel()
    f0 = (gold[:, 0] + obj[:, 0]) - (gold[:, 1] + obj[:, 1])
    assert value.winner_truth(f0, bins, reps, obj, 'binned').tobytes() == f0.tobytes()
    reps[bins[2, 0]] += 1.0                                                                       # one representative off by a gold: that unit moves, the others stay
    moved = value.winner_truth(f0, bins, reps, obj, 'binned')
    assert moved[2] == pytest.approx(f0[2] + 1.0, abs=1e-9) and np.array_equal(np.delete(moved, 2), np.delete(f0, 2))


def test_gold_rule_returns_the_f0_unchanged_bit_for_bit_and_reads_no_other_argument():
    f0 = np.array([-0.0, 0.0, 1.5, -2.5, 403.125, 1e-300, -1e-12, 0.1 + 0.2 - 0.3, 123456789.987654321])
    out = value.winner_truth(f0, None, None, None, 'gold')                                       # the other arguments are not read at all
    assert out.dtype == np.float64 and out.shape == f0.shape and out.tobytes() == f0.tobytes()   # the sign of -0.0 and the noise of 0.1 + 0.2 - 0.3 survive
    assert out is not f0 and not np.shares_memory(out, f0)                                       # the label array is never aliased
    out[0] = 9.0
    assert f0[0] == 0.0
    f0, bins, obj, _ = tie_arrays()
    assert value.winner_truth(f0, bins, TIE_REPS, obj, 'gold').tobytes() == f0.tobytes()
    assert value.winner_truth(f0, 'not an array', 'not reps', 'not objectives', 'gold').tobytes() == f0.tobytes()
    as_float32 = f0.astype(np.float32)
    assert value.winner_truth(as_float32, None, None, None, 'gold').tobytes() == as_float32.astype(np.float64).tobytes()


def test_the_winner_truth_rules_are_the_two_names_of_the_profile_field_with_gold_first():
    assert value.WINNER_TRUTH_RULES == ('gold', 'binned')
    assert value.check_winner_truth('gold') == 'gold' and value.check_winner_truth('binned') == 'binned'


@pytest.mark.parametrize('rule', ['Gold', 'raw', '', ' binned', 'binned ', None, 3, 1.0, ('gold',), np.array(['gold']), b'gold'])
def test_an_unknown_winner_truth_rule_is_refused_by_name_and_echoes_no_data(rule):
    f0 = np.array([123456.789])
    with pytest.raises(ValueError, match='winner_truth') as raised:
        value.winner_truth(f0, np.zeros((1, 2), dtype=np.int8), TIE_REPS, np.zeros((1, 2)), rule)
    assert '123456' not in str(raised.value)                                                    # no label value in the message
    with pytest.raises(ValueError, match='winner_truth'):
        value.check_winner_truth(rule)


def test_a_long_unknown_rule_text_is_cut_in_the_message():
    with pytest.raises(ValueError, match='winner_truth') as raised:
        value.check_winner_truth('x' * 40 + 'SECRET' + 'y' * 100)
    assert 'x' * 40 in str(raised.value) and 'SECRET' not in str(raised.value)


@pytest.mark.parametrize('rule', ['gold', 'binned'])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_winner_truth_refuses_an_f0_that_is_not_finite_under_either_rule(rule, bad):
    f0, bins, obj, _ = tie_arrays()
    f0[4] = bad                                                                                  # a unit that is not valued has NaN and no winner
    with pytest.raises(ValueError, match='finite'):
        value.winner_truth(f0, bins, TIE_REPS, obj, rule)


def test_winner_truth_refuses_label_arrays_that_do_not_fit_the_binned_rule_without_echoing_a_value():
    f0, bins, obj, _ = tie_arrays()
    nan_obj = obj.copy(); nan_obj[3, 1] = np.nan
    inf_obj = obj.copy(); inf_obj[3, 0] = np.inf
    bad_reps = TIE_REPS.copy(); bad_reps[7] = np.nan
    negative_reps = TIE_REPS.copy(); negative_reps[7] = -1.0
    too_high, too_low = bins.copy(), bins.copy()
    too_high[2, 0], too_low[2, 1] = 31, -2
    cases = {'f0 of the wrong rank': (f0[:, None], bins, TIE_REPS, obj),
             'no bins': (f0, None, TIE_REPS, obj), 'bins of one team': (f0, bins[:, :1], TIE_REPS, obj), 'bins of another length': (f0, bins[:-1], TIE_REPS, obj),
             'bins that are floats': (f0, bins.astype(np.float64), TIE_REPS, obj), 'bins above 30': (f0, too_high, TIE_REPS, obj),
             'bins below -1': (f0, too_low, TIE_REPS, obj), 'no representatives': (f0, bins, None, obj),
             'a short representative table': (f0, bins, TIE_REPS[:30], obj), 'a representative that is NaN': (f0, bins, bad_reps, obj),
             'a representative below 0': (f0, bins, negative_reps, obj), 'no objectives': (f0, bins, TIE_REPS, None),
             'objectives of one team': (f0, bins, TIE_REPS, obj[:, :1]), 'objectives of another length': (f0, bins, TIE_REPS, obj[:-1]),
             'objectives with a NaN': (f0, bins, TIE_REPS, nan_obj), 'objectives with an infinity': (f0, bins, TIE_REPS, inf_obj),
             'text for the representatives': (f0, bins, 'abcxyz', obj)}
    for what, (a, b, c, d) in cases.items():
        with pytest.raises(ValueError, match='winner_truth') as raised:
            value.winner_truth(a, b, c, d, 'binned')
        assert 'abcxyz' not in str(raised.value), what                                          # the text that was passed is not echoed


def test_winner_truth_on_the_labels_of_unit_values_reads_the_toy_units(toy_s1, toy_uv):
    """The arrays that ``unit_values`` makes (bins as int8 with -1, the obj_value, the f0) and the FIT bin means as the representatives: no hand
    value needs a tie here, so the test pins the numbers of the units that have a pair of bins. Unit 9 is a unit of a role that is not valued."""
    means, _ = value.gold_bin_means(toy_uv, value.role_units(toy_s1, ('FIT',)))
    valued = np.flatnonzero(toy_uv.valued)
    got = value.winner_truth(toy_uv.f0_unit[valued], toy_uv.gold_bin[valued], means, toy_uv.obj_value[valued], 'binned')
    assert got.shape == (len(valued),)
    by_unit = dict(zip(valued.tolist(), got.tolist()))
    assert by_unit[3] == 0.0                                                                     # 300 and 300 in bin 3: a tie under either rule
    assert by_unit[0] == pytest.approx(means[4])                                                 # blue has bin 4, red no kill: (426.5625 + 0) - (0 + 0)
    assert by_unit[1] == pytest.approx((means[11] + 410.0) - (means[4] + 350.0))                 # bins 11 and 4 with the objectives 410 and 350
    assert (np.sign(got) == np.sign(toy_uv.f0_unit[valued])).all()                               # no toy unit changes its winner under the FIT bin means
    with pytest.raises(ValueError, match='finite'):
        value.winner_truth(toy_uv.f0_unit, toy_uv.gold_bin, means, toy_uv.obj_value, 'binned')    # unit 9 has NaN labels
    with pytest.raises(ValueError, match='finite'):
        value.winner_truth(toy_uv.f0_unit, toy_uv.gold_bin, means, toy_uv.obj_value, 'gold')


def ledger_world(tmp_path, records, matches=None):
    """A store of the given synthetic records, all FIT matches of 15.18 unless ``matches`` says otherwise."""
    matches = matches or [('15.18', toy_key(i), 'FIT') for i in range(len(records))]
    table = dict(zip((m[1] for m in matches), records))
    return build_value_s1(tmp_path / 's1', matches, lambda key, patch: table[key])


def near_draw_record(key, patch):
    """One execution at 150,000 ms with two blue assisters (so its people hold players 1, 2 and the red victim 9); blue gets a dragon
    (credit 1) and an elder (credit 2), red gets the grubs (credit 9, the victim): with the prices 0.1, 0.2 and 0.3 the two teams differ
    by 5.5e-17 gold, which is noise."""
    kills = [value_kill(150_000, 5_000, 5_000, 0, 9, (1, 2))]
    objectives = [value_obj(152_000, 'dragon', 0, 1, (), (5_000, 5_100)), value_obj(153_000, 'elder', 0, 2, (), (5_000, 5_200)),
                  value_obj(154_000, 'grubs', 1, 9, (), (5_000, 5_300))]
    return synth.make_record(T_ms=500_000, kills=kills, objectives=objectives, patch=patch, key=key)


def test_an_end_to_end_draw_from_rounding_noise_and_the_sure_threshold(tmp_path, toy_rules):
    s1 = ledger_world(tmp_path, [near_draw_record(toy_key(0), '15.18')])
    assert np.asarray(s1.array('objectives/unit')).tolist() == [0, 0, 0]                       # all three belong to the unit
    prices = {**TOY_PRICES, 'dragon': 0.1, 'elder': 0.2, 'grubs': 0.3}
    uv = value.unit_values(s1, prices, toy_rules, np.array([1], dtype=np.int16))
    assert uv.f0_unit[0] != 0.0 and abs(uv.f0_unit[0]) < 1e-15
    assert uv.winner[0] == 0 and not uv.sure[0]
    for dragon, sure in ((150.0, True), (149.9999996, True), (149.9999994, False), (149.999999, False), (151.0, True)):
        one = value.unit_values(s1, {**TOY_PRICES, 'dragon': dragon, 'elder': 0.0, 'grubs': 0.0}, toy_rules, np.array([1], dtype=np.int16))
        assert one.winner[0] == 1 and bool(one.sure[0]) is sure


def test_the_side_of_a_kill_follows_the_team_array_not_the_slot(tmp_path, toy_rules):
    """matches/team gives blue the odd players: red 4 (by the slot rule, players 1-5, he would be blue) kills blue 3 for 450, blue 1 kills
    red 2 for 300, and the dragon is taken by team 100 (blue)."""
    rec = synth.make_record(T_ms=500_000, kills=[value_kill(150_000, 5_000, 5_000, 4, 3, (), 450), value_kill(152_000, 5_100, 5_000, 1, 2)],
                            objectives=[value_obj(154_000, 'dragon', 0, 3, (), (5_000, 5_100))], patch='15.18', key=toy_key(0))
    rec = dataclasses.replace(rec, team=np.array([100, 200] * 5, dtype=np.int16))
    s1 = ledger_world(tmp_path, [rec])
    uv = value.unit_values(s1, TOY_PRICES, toy_rules, np.array([1, 1], dtype=np.int16))
    assert uv.kill_gold[0].tolist() == [300.0, 450.0] and uv.n_kills[0].tolist() == [1, 1]
    assert uv.obj_value[0].tolist() == [TOY_PRICES['dragon'], 0.0]
    assert uv.f0_unit[0] == pytest.approx(300.0 + TOY_PRICES['dragon'] - 450.0)


# ---- unit_values: the hand-worked toy matches ------------------------------------------------------------------------------------------
@pytest.mark.parametrize('u', sorted(TOY_UNIT_VALUES))
def test_every_valued_toy_unit_has_its_hand_worked_values(toy_uv, u):
    want = TOY_UNIT_VALUES[u]
    for name in ('kill_gold', 'obj_value', 'conv_value'):
        assert getattr(toy_uv, name)[u].tolist() == pytest.approx(list(want[name])), name
    assert toy_uv.n_kills[u].tolist() == list(want['n_kills'])
    assert toy_uv.gold_bin[u].tolist() == list(want['gold_bin'])
    assert toy_uv.f0_unit[u] == pytest.approx(TOY_F0_UNIT[u]) and toy_uv.f0_conv[u] == pytest.approx(TOY_F0_CONV[u])
    assert toy_uv.winner[u] == want['winner'] and bool(toy_uv.sure[u]) is TOY_SURE[u] and bool(toy_uv.valued[u])


def test_f0_is_blue_minus_red_and_the_conversions_are_separate(toy_uv):
    kg, ov, cv = toy_uv.kill_gold, toy_uv.obj_value, toy_uv.conv_value
    assert np.allclose(toy_uv.f0_unit[:9], (kg[:9, 0] + ov[:9, 0]) - (kg[:9, 1] + ov[:9, 1]))
    assert np.allclose(toy_uv.f0_conv[:9], cv[:9, 0] - cv[:9, 1])
    assert toy_uv.f0_unit[0] == pytest.approx(403.125)                       # the conversion (-410) is not in f0 of the unit
    assert toy_uv.f0_conv[0] == pytest.approx(-410.0)


def test_a_unit_of_a_role_that_is_not_valued_has_no_labels(toy_uv):
    assert toy_uv.valued.tolist() == [True] * 9 + [False] and toy_uv.match_valued.tolist() == [True, True, True, False]
    for name in ('kill_gold', 'obj_value', 'conv_value'):
        assert np.isnan(getattr(toy_uv, name)[9]).all()
    assert np.isnan(toy_uv.f0_unit[9]) and np.isnan(toy_uv.f0_conv[9])
    assert toy_uv.winner[9] == 0 and not toy_uv.sure[9] and toy_uv.gold_bin[9].tolist() == [-1, -1] and toy_uv.n_kills[9].tolist() == [0, 0]


def test_the_arrays_of_the_labels_are_read_only(toy_uv):
    for name in ('kill_gold', 'obj_value', 'conv_value', 'f0_unit', 'f0_conv', 'winner', 'sure', 'gold_bin', 'n_kills', 'valued',
                 'match_valued'):
        assert not getattr(toy_uv, name).flags.writeable, name


def test_unit_values_dtypes_and_shapes(toy_uv):
    n = TOY_N_UNITS
    for name, dtype, shape in (('kill_gold', np.float64, (n, 2)), ('obj_value', np.float64, (n, 2)), ('conv_value', np.float64, (n, 2)),
                               ('f0_unit', np.float64, (n,)), ('f0_conv', np.float64, (n,)), ('winner', np.int8, (n,)),
                               ('sure', np.bool_, (n,)), ('gold_bin', np.int8, (n, 2)), ('n_kills', np.int32, (n, 2)),
                               ('valued', np.bool_, (n,)), ('match_valued', np.bool_, (4,))):
        a = getattr(toy_uv, name)
        assert a.dtype == dtype and a.shape == shape, name


def test_the_assist_pool_cap_the_early_multiplier_and_the_level_in_the_hand_kills(toy_s1, toy_rules):
    """One kill at a time: every pool of the toy matches by its own formula, through the S1 kill table (a check of the per-kill gold apart
    from the sums by unit)."""
    t, bounty, shutdown, killer = (np.asarray(toy_s1.array(f'kills/{n}')) for n in ('t', 'bounty', 'shutdown', 'killer'))
    n_ass = [bin(int(a)).count('1') for a in np.asarray(toy_s1.array('kills/amask'))]
    paid = []
    for i in range(13):                                                                   # the kills of the matches 0 (15.18) and 1 (15.20)
        pool = rulepay.assist_pool(int(bounty[i]), int(TOY_LEVELS[i]), int(t[i]), '15.18' if i < 9 else '15.20', n_ass[i], toy_rules)
        paid.append(int(bounty[i]) + int(shutdown[i]) + pool if 1 <= killer[i] <= 10 else 0.0)
    assert paid[:9] == pytest.approx([403.125, 516.25, 450.0, 670.625, 0.0, 300.0, 300.0, 300.0, 300.0])
    assert paid[9:] == pytest.approx([575.0, 500.0, 930.0, 300.0])


def test_an_execution_pays_nobody_even_with_a_bounty_and_assisters(toy_s1, toy_uv):
    kills = np.asarray(toy_s1.array('kills/killer'))
    execution = int(np.flatnonzero(kills == 0)[0])
    assert np.asarray(toy_s1.array('kills/bounty'))[execution] == 300 and np.asarray(toy_s1.array('kills/amask'))[execution] != 0
    assert toy_uv.kill_gold[TOY_EXEC_U].tolist() == [0.0, 0.0] and toy_uv.n_kills[TOY_EXEC_U].tolist() == [0, 0]
    assert toy_uv.winner[TOY_EXEC_U] == 0 and toy_uv.gold_bin[TOY_EXEC_U].tolist() == [-1, -1]
    assert toy_uv.f0_conv[TOY_EXEC_U] == pytest.approx(TOY_PRICES['tower_outer'])           # its conversion is a different label


def test_a_level_up_on_the_ms_of_the_kill_counts_in_the_pool(toy_s1, toy_prices, toy_rules):
    """Unit 1, first kill: the victim's LEVEL_UP to 7 is on the ms of the kill (base 310, cap 155: pool 116.25); at level 6 the cap is 150
    and the pool 112.5."""
    levels = TOY_LEVELS.copy()
    levels[1] = 6
    lower = value.unit_values(toy_s1, toy_prices, toy_rules, levels)
    assert lower.kill_gold[TOY_U, 0] == pytest.approx(1186.875 - 116.25 + 112.5)


def test_the_victim_level_decides_the_cap_of_the_pool(toy_s1, toy_prices, toy_rules):
    levels = TOY_LEVELS.copy()
    levels[3] = 6                                                                     # K3: level 6 instead of 18: cap 150 instead of 210
    uv = value.unit_values(toy_s1, toy_prices, toy_rules, levels)
    assert uv.kill_gold[TOY_U, 0] == pytest.approx(1186.875 - 170.625 + 150 * 0.8125)


def test_the_shutdown_bounty_is_gold_but_not_part_of_the_pool(toy_uv):
    # unit 7: bounty 200 + shutdown 200, four assisters at 210 s, level 5: pool min(100, 150) = 100, not min(200, 150) = 150
    assert toy_uv.kill_gold[7, 0] == pytest.approx(500.0)
    assert toy_uv.kill_gold[1, 1] == 450.0                                            # unit 1 red: 300 + 150 shutdown, nobody assisted


def test_the_rules_argument_is_used_and_none_means_the_p3_rules(toy_s1, toy_prices, toy_rules):
    base = value.unit_values(toy_s1, toy_prices, toy_rules, TOY_LEVELS)
    assert np.array_equal(value.unit_values(toy_s1, toy_prices, None, TOY_LEVELS).kill_gold, base.kill_gold, equal_nan=True)
    seg = rulepay._Segment((15, 18), (15, 22), None)
    low = rulepay.RuleTable({'champion_kill_base_bounty': (dataclasses.replace(seg, value=(300,) * 18),),
                             'assist_bounty': (dataclasses.replace(seg, value=(0.25, 0.25)),),
                             'assist_bounty_early_game_multiplier': (dataclasses.replace(seg, value=(90.0, 210.0, 0.5, 1.0)),),
                             'first_blood_bonus': (dataclasses.replace(seg, value=(0, 0)),)})
    uv = value.unit_values(toy_s1, toy_prices, low, TOY_LEVELS)
    # unit 0: pool = min(0.25 x 300, 0.25 x 300) x 0.6875 = 51.5625
    assert uv.kill_gold[0, 0] == pytest.approx(300 + 51.5625)
    assert uv.kill_gold[2].tolist() == [0.0, 0.0]


def test_the_patch_of_each_kill_is_the_patch_of_its_match(toy_s1, toy_prices):
    """A table whose pool share differs by patch: the kills of the match of patch 15.18 (FIT, match 0) use the 15.18 share, those of 15.20
    (STOP, match 1) the 15.20 share, and a patch without rules refuses."""
    def seg(lo, hi, v):
        return rulepay._Segment(lo, hi, v)
    base = tuple(300 + 10 * max(0, level - 6) for level in range(1, 19))
    multiplier = (90.0, 210.0, 0.5, 1.0)

    def table(served=None):
        return rulepay.RuleTable({
            'champion_kill_base_bounty': (seg((15, 18), (15, 22), base),),
            'assist_bounty': (seg((15, 18), (15, 19), (0.5, 0.5)), seg((15, 20), (15, 22), (0.2, 0.2))),
            'assist_bounty_early_game_multiplier': (seg((15, 18), (15, 22), multiplier),),
            'first_blood_bonus': (seg((15, 18), (15, 22), (0, 0)),)}, served=served)
    uv = value.unit_values(toy_s1, toy_prices, table(), TOY_LEVELS)
    assert uv.kill_gold[0, 0] == pytest.approx(300 + 103.125)                          # 15.18: share 0.5
    assert uv.kill_gold[7, 0] == pytest.approx(200 + 200 + min(0.2 * 200, 0.2 * 300) * 1.0)       # 15.20: share 0.2, pool 40
    with pytest.raises(ValueError):
        value.unit_values(toy_s1, toy_prices, table(served=[(15, 18)]), TOY_LEVELS)     # the 15.20 kills have no rules


def test_the_prices_argument_is_used_per_group(toy_s1, toy_rules, toy_prices):
    twice = {g: 2 * p for g, p in toy_prices.items()}
    one, two = (value.unit_values(toy_s1, p, toy_rules, TOY_LEVELS) for p in (toy_prices, twice))
    assert np.allclose(two.obj_value[:9], 2 * one.obj_value[:9]) and np.allclose(two.conv_value[:9], 2 * one.conv_value[:9])
    assert np.array_equal(two.kill_gold, one.kill_gold, equal_nan=True)                  # the prices are not in the kill gold
    only_dragon = {g: (7.0 if g == 'dragon' else 0.0) for g in toy_prices}
    d = value.unit_values(toy_s1, only_dragon, toy_rules, TOY_LEVELS)
    assert d.obj_value[:9].tolist() == [[7.0, 0.0] if u == 1 else [0.0, 0.0] for u in range(9)]
    assert d.conv_value[:9, 1].tolist() == [7.0] + [0.0] * 8              # the red dragon without a position, converted to unit 0


def test_an_objective_without_a_known_group_or_taker_is_worth_nothing(toy_s1, toy_uv, toy_prices):
    # unit 3 holds the tower without a tier (attributed, taker blue): no price. The dragon without a team and the monster of unknown kind
    # are in no unit and in no conversion
    assert toy_uv.obj_value[TOY_DRAW_U].tolist() == [0.0, 0.0]
    assert np.asarray(toy_s1.array('objectives/unit'))[8] == TOY_DRAW_U
    total = np.nansum(toy_uv.obj_value) + np.nansum(toy_uv.conv_value)
    assert total == pytest.approx(sum(TOY_OBJ_VALUE_TOTAL[k][i] for k in ('attributed', 'converted') for i in (0, 1)))


def test_an_attributed_and_a_converted_objective_go_to_different_labels(toy_s1, toy_uv):
    ou, cu = (np.asarray(toy_s1.array(f'objectives/{n}')) for n in ('unit', 'conv_unit'))
    assert not ((ou >= 0) & (cu >= 0)).any()
    assert ou[0] == 1 and cu[1] == 0 and cu[3] == 1 and cu[4] == 1 and cu[5] == 2     # the attributions and conversions as worked out
    assert toy_uv.obj_value[1].tolist() == [410.0, 350.0] and toy_uv.conv_value[1].tolist() == [630.0, 160.0]


def test_unit_values_with_a_narrower_role_set_values_only_those_units(toy_s1, toy_prices, toy_rules):
    uv = value.unit_values(toy_s1, toy_prices, toy_rules, TOY_LEVELS, roles=('FIT',))
    assert uv.valued.tolist() == [True] * 6 + [False] * 4 and uv.match_valued.tolist() == [True, False, False, False]
    assert np.isnan(uv.f0_unit[6:]).all() and np.isfinite(uv.f0_unit[:6]).all()
    assert uv.f0_unit[:6].tolist() == pytest.approx(list(TOY_F0_UNIT[:6]))
    both = value.unit_values(toy_s1, toy_prices, toy_rules, TOY_LEVELS)
    assert both.f0_unit[:6].tolist() == uv.f0_unit[:6].tolist()
    for roles in (('TEST',), ('EXTERNAL',), ('CAL',), ('AUDIT',), ('UNUSED',), ('NOPE',), ()):
        with pytest.raises(ValueError):
            value.unit_values(toy_s1, toy_prices, toy_rules, TOY_LEVELS, roles=roles)


def test_unit_values_does_not_change_its_inputs(toy_s1, toy_prices, toy_rules):
    levels, prices = TOY_LEVELS.copy(), dict(toy_prices)
    value.unit_values(toy_s1, prices, toy_rules, levels)
    assert np.array_equal(levels, TOY_LEVELS) and prices == toy_prices


# ---- unit_values: refusals ---------------------------------------------------------------------------------------------------------------
def test_a_valued_kill_without_a_victim_level_is_refused_and_counted(toy_s1, toy_prices, toy_rules):
    levels = TOY_LEVELS.copy()
    levels[[2, 9]] = -1                                                                  # a FIT kill and a STOP kill
    with pytest.raises(ValueError) as raised:
        value.unit_values(toy_s1, toy_prices, toy_rules, levels)
    text = message_of(raised)
    assert '2 kills' in text and 'kill_levels' in text
    ok = value.unit_values(toy_s1, toy_prices, toy_rules, levels, roles=('DEV',))          # a role whose kills all have a level: no refusal
    assert not ok.valued[:9].any()


def test_a_kill_with_a_level_of_minus_one_in_a_match_that_is_not_valued_is_no_problem(toy_s1, toy_prices, toy_rules):
    assert TOY_LEVELS[-1] == -1 and np.isnan(value.unit_values(toy_s1, toy_prices, toy_rules, TOY_LEVELS).f0_unit[9])


def test_an_execution_needs_a_level_too(toy_s1, toy_prices, toy_rules):
    levels = TOY_LEVELS.copy()
    levels[4] = -1                                                                       # the execution
    with pytest.raises(ValueError):
        value.unit_values(toy_s1, toy_prices, toy_rules, levels)


@pytest.mark.parametrize('bad', [np.array([19] + [1] * 13, dtype=np.int16), np.array([0] + [1] * 13, dtype=np.int16),
                                 np.arange(5, dtype=np.int16), TOY_LEVELS.astype(np.float64), TOY_LEVELS.reshape(2, 7),
                                 TOY_LEVELS[:-1], np.append(TOY_LEVELS, 3)])
def test_levels_that_cannot_be_victim_levels_are_refused(toy_s1, toy_prices, toy_rules, bad):
    with pytest.raises(ValueError):
        value.unit_values(toy_s1, toy_prices, toy_rules, bad)


def test_unit_values_refuses_bad_prices_and_rules(toy_s1, toy_prices, toy_rules):
    with pytest.raises(ValueError):
        value.unit_values(toy_s1, {k: v for k, v in toy_prices.items() if k != 'dragon'}, toy_rules, TOY_LEVELS)
    with pytest.raises(TypeError):
        value.unit_values(toy_s1, toy_prices, 'P3', TOY_LEVELS)


def test_unit_values_refuses_a_store_whose_unit_indices_are_inconsistent(toy_s1, toy_prices, toy_rules, tmp_path):
    root = tmp_path / 'copy'
    shutil.copytree(toy_s1.root, root)
    ku = np.load(root / 'kills' / 'unit.npy', mmap_mode='r+')
    ku[0] = 3                                                           # the kill of unit 0 now says unit 3 (another unit, same match)
    ku.flush()
    del ku
    gc.collect()
    broken = Store(root)
    uv = value.unit_values(broken, toy_prices, toy_rules, TOY_LEVELS)                    # still in its own match: allowed, the sums change
    assert uv.kill_gold[0, 0] == 0.0 and uv.kill_gold[3, 0] == pytest.approx(300 + 403.125)
    ku = np.load(root / 'kills' / 'unit.npy', mmap_mode='r+')
    ku[0] = 7                                                                            # a unit of another match
    ku.flush()
    del ku
    gc.collect()
    with pytest.raises(ValueError, match='match'):
        value.unit_values(Store(root), toy_prices, toy_rules, TOY_LEVELS)
    ku = np.load(root / 'kills' / 'unit.npy', mmap_mode='r+')
    ku[0] = -1
    ku.flush()
    del ku
    gc.collect()
    with pytest.raises(ValueError):
        value.unit_values(Store(root), toy_prices, toy_rules, TOY_LEVELS)


# ---- the victim level of a record --------------------------------------------------------------------------------------------------
def test_kill_levels_for_record_of_the_toy_matches():
    for m, builder, patch, lo in ((0, value_record_0, '15.18', 0), (1, value_record_1, '15.20', 9)):
        got = value.kill_levels_for_record(builder(toy_key(m), patch))
        assert got.dtype == np.int16 and got.tolist() == TOY_LEVELS[lo:lo + len(got)].tolist()


def test_the_level_is_that_of_the_victim_at_the_time_of_the_kill():
    rec = synth.make_record(T_ms=600_000, kills=[value_kill(t, 5_000, 5_000, 1, 6) for t in (50_000, 100_000, 100_001, 200_000, 300_000)],
                            events=[level_up(6, 2, 100_000), level_up(6, 5, 200_000), level_up(6, 4, 250_000), level_up(7, 9, 90_000),
                                    level_up(1, 12, 10_000)], patch='15.18', key=toy_key(0))
    # before the first LEVEL_UP: 1; on the ms of the LEVEL_UP: counted; a lower level after a higher one does not lower it; the level of
    # another player (7) or of the killer (1) is not the victim's
    assert value.kill_levels_for_record(rec).tolist() == [1, 2, 2, 5, 5]


def test_a_record_without_level_ups_or_without_kills():
    rec = synth.make_record(T_ms=600_000, kills=[value_kill(200_000, 5_000, 5_000, 1, 6)], patch='15.18', key=toy_key(0))
    assert value.kill_levels_for_record(rec).tolist() == [1]
    empty = value.kill_levels_for_record(synth.make_record(T_ms=300_000, patch='15.18', key=toy_key(0)))
    assert empty.dtype == np.int16 and empty.shape == (0,)


def test_the_level_of_an_execution_is_the_victims_too():
    rec = synth.make_record(T_ms=600_000, kills=[value_kill(200_000, 5_000, 5_000, 0, 9)], events=[level_up(9, 7, 150_000)],
                            patch='15.18', key=toy_key(0))
    assert value.kill_levels_for_record(rec).tolist() == [7]


def test_kill_levels_for_record_follows_the_order_of_the_s1_kills(toy_s1):
    rec = value_record_0(toy_key(0), '15.18')
    kt = units.kill_table(rec)
    assert np.array_equal(kt.t, np.asarray(toy_s1.array('kills/t'))[:9])
    assert np.array_equal(kt.victim, np.asarray(toy_s1.array('kills/victim'))[:9])


# ---- the victim levels of a store (raw reads) --------------------------------------------------------------------------------------------
def test_kill_levels_of_the_toy_store_equal_the_hand_levels(toy_s1):
    got = value.kill_levels(toy_s1, S3_ROLES, workers=1, records=value_toy_records)
    assert got.dtype == np.int16 and got.tolist() == TOY_LEVELS.tolist()               # the CAL match is not asked for: -1


def test_kill_levels_with_a_process_pool_gives_the_same_levels(toy_s1):
    report = {}
    got = value.kill_levels(toy_s1, S3_ROLES, workers=2, records=value_toy_records, report=report)
    assert got.tolist() == TOY_LEVELS.tolist()
    assert report == {'matches': 3, 'unreadable': 0, 'kills_read': 13, 'kills_unread': 0}


def test_kill_levels_asks_only_for_the_roles_it_is_given(toy_s1):
    asked = []

    def records(key, patch):
        asked.append(key)
        return value_toy_records(key, patch)
    got = value.kill_levels(toy_s1, ('STOP',), workers=1, records=records)
    assert asked == [toy_key(1)]
    assert got.tolist() == [-1] * 9 + TOY_LEVELS[9:13].tolist() + [-1]
    asked.clear()
    value.kill_levels(toy_s1, 'DEV', workers=1, records=records)                          # a single role name works too
    assert asked == [toy_key(2)]


@pytest.mark.parametrize('roles', [('TEST',), ('EXTERNAL',), ('CAL',), ('AUDIT',), ('UNUSED',), ('NOPE',), (), ('FIT', 'TEST')])
def test_kill_levels_refuses_roles_that_stage_s3_does_not_read(toy_s1, roles):
    asked = []
    with pytest.raises(ValueError):
        value.kill_levels(toy_s1, roles, workers=1, records=lambda key, patch: asked.append(key))
    assert not asked


@pytest.mark.parametrize('workers', [0, -1, 1.5, True, '2', None])
def test_kill_levels_refuses_a_bad_worker_count(toy_s1, workers):
    with pytest.raises(ValueError):
        value.kill_levels(toy_s1, S3_ROLES, workers=workers, records=value_toy_records)


def test_kill_levels_stops_at_e6_before_any_match_is_read(toy_s1, fake_processes):
    asked = []
    fake_processes([synth.STRATA_SERVER, synth.STRATA_BRIDGE])

    def records(key, patch):
        asked.append(key)
        return value_toy_records(key, patch)
    with pytest.raises(run_s1.StrataRunning) as raised:
        value.kill_levels(toy_s1, S3_ROLES, workers=1, records=records)
    assert not asked and 'E6' in str(raised.value) and KEY_PREFIX not in str(raised.value)


def test_kill_levels_counts_an_unreadable_match_and_keeps_minus_one_there(toy_s1):
    def records(key, patch):
        if key == toy_key(1):
            raise reader.MatchReadError('timeline', 'PermissionError', 13, patch)
        return value_toy_records(key, patch)
    report = {}
    with pytest.warns(RuntimeWarning, match='1 of 3 matches') as caught:
        got = value.kill_levels(toy_s1, S3_ROLES, workers=1, records=records, report=report)
    assert KEY_PREFIX not in ' '.join(str(w.message) for w in caught)
    assert got.tolist() == TOY_LEVELS[:9].tolist() + [-1] * 4 + [-1] and report == {
        'matches': 3, 'unreadable': 1, 'kills_read': 9, 'kills_unread': 4}


def test_kill_levels_without_a_report_argument_still_returns_just_the_levels(toy_s1):
    assert isinstance(value.kill_levels(toy_s1, S3_ROLES, workers=1, records=value_toy_records), np.ndarray)


def test_kill_levels_removes_a_match_key_from_an_error_that_carries_it(toy_s1):
    def records(key, patch):
        if key == toy_key(1):
            raise KeyError(key)
        return value_toy_records(key, patch)
    with pytest.raises(Exception) as raised:
        value.kill_levels(toy_s1, S3_ROLES, workers=1, records=records)
    assert KEY_PREFIX not in str(raised.value) and KEY_PREFIX not in repr(raised.value) and '<key>' in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__context__ is None


def test_kill_levels_removes_a_match_key_from_an_error_of_a_worker_process_too(toy_s1):
    with pytest.raises(Exception) as raised:
        value.kill_levels(toy_s1, S3_ROLES, workers=2, records=value_toy_records_that_leak_a_key)
    chain, error = [], raised.value
    while error is not None:                                     # the error, its cause (the traceback text of the worker) and its context
        chain.append(error)
        error = error.__cause__ or error.__context__
    assert chain and all(KEY_PREFIX not in str(e) + repr(e) for e in chain) and '<key>' in str(raised.value)


def records_that_leak_the_key_in_a_cause(key, patch):
    """Module level (the workers import it): the error of the match 1 reads like a file problem, and only the error that caused it holds
    the match key."""
    if key == toy_key(1):
        try:
            raise KeyError(key)
        except KeyError as inner:
            raise RuntimeError('the match file is not as expected') from inner
    return value_toy_records(key, patch)


def records_that_leak_the_game_id(key, patch):
    """Module level: the error of the match 1 quotes a value of the file, the game id without the platform."""
    if key == toy_key(1):
        raise KeyError(f'the timeline {toy_key(1).rsplit("_", 1)[1]} has no frames')
    return value_toy_records(key, patch)


def error_chain(error) -> list:
    """The error, its causes and its contexts."""
    chain, todo, seen = [], [error], set()
    while todo:
        e = todo.pop()
        if e is not None and id(e) not in seen:
            seen.add(id(e))
            chain.append(e)
            todo += [e.__cause__, e.__context__]
    return chain


@pytest.mark.parametrize('workers', [1, 2])
def test_kill_levels_removes_a_match_key_that_only_the_cause_chain_of_an_error_holds(toy_s1, workers):
    """The text of the error is clean and the key is in the error that caused it: the scrub of ``store_build`` looks through the chain."""
    with pytest.raises(Exception) as raised:
        value.kill_levels(toy_s1, S3_ROLES, workers=workers, records=records_that_leak_the_key_in_a_cause)
    chain = error_chain(raised.value)
    assert isinstance(raised.value, ValueError) and 'RuntimeError' in str(raised.value)
    assert all(KEY_PREFIX not in str(e) + repr(e) for e in chain)
    if workers == 1:                                             # a pool adds the traceback text of the worker as the cause (checked above)
        assert raised.value.__cause__ is None and raised.value.__context__ is None


@pytest.mark.parametrize('workers', [1, 2])
def test_kill_levels_removes_the_bare_game_id_of_a_match_key_from_an_error(toy_s1, workers):
    """A message that quotes a value of the file can hold the number of the match without the platform: it becomes <id>."""
    game_id = toy_key(1).rsplit('_', 1)[1]
    with pytest.raises(Exception) as raised:
        value.kill_levels(toy_s1, S3_ROLES, workers=workers, records=records_that_leak_the_game_id)
    chain = error_chain(raised.value)
    assert all(game_id not in str(e) and KEY_PREFIX not in str(e) + repr(e) for e in chain)
    assert '<id>' in str(raised.value) and 'match row 1' in str(raised.value)


def test_kill_levels_does_not_touch_an_error_that_holds_neither_the_key_nor_the_game_id(toy_s1):
    def records(key, patch):
        if key == toy_key(1):
            raise RuntimeError('an error that names no match')
        return value_toy_records(key, patch)
    with pytest.raises(RuntimeError, match='an error that names no match'):
        value.kill_levels(toy_s1, S3_ROLES, workers=1, records=records)


def test_kill_levels_lets_a_format_error_through_without_the_key(toy_s1):
    def records(key, patch):
        raise reader.MatchFormatError(key, patch, 'a LEVEL_UP has no level')
    with pytest.raises(reader.MatchFormatError) as raised:
        value.kill_levels(toy_s1, S3_ROLES, workers=1, records=records)
    assert 'LEVEL_UP' in str(raised.value) and KEY_PREFIX not in str(raised.value)


def test_kill_levels_refuses_raw_files_that_do_not_agree_with_the_s1_kills(toy_s1):
    def records(key, patch):
        return value_toy_records(toy_key(0), '15.18') if key == toy_key(1) else value_toy_records(key, patch)       # another match's file
    with pytest.raises(ValueError) as raised:
        value.kill_levels(toy_s1, S3_ROLES, workers=1, records=records)
    assert 'match row 1' in message_of(raised)

    def shifted(key, patch):                                                                # same number of kills, other times
        rec = value_toy_records(key, patch)
        events = tuple(dict(e, timestamp=e['timestamp'] + 1) if e['type'] == 'CHAMPION_KILL' else e for e in rec.events)
        return dataclasses.replace(rec, events=events)
    with pytest.raises(ValueError, match='match row'):
        value.kill_levels(toy_s1, S3_ROLES, workers=1, records=shifted)


def test_the_module_never_passes_the_holdout_switch_to_the_reader():
    tree = ast.parse(Path(value.__file__).read_text(encoding='utf-8'))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg == 'allow_holdout']


# ---- V_start -----------------------------------------------------------------------------------------------------------------------
def test_v_start_of_the_toy_rows_by_hand(toy_s1, toy_uv):
    got = value.v_start(toy_s1, toy_uv)
    assert got.shape == (TOY_N_ROWS, 3) and got.dtype == np.float64
    assert np.allclose(got, TOY_V_START, equal_nan=True)
    assert np.isnan(got[22:]).all() and np.isfinite(got[:22]).all()                         # the rows of the CAL match have no labels


@pytest.mark.parametrize('j, horizon', list(enumerate(C.EVAL_H_MS)))
def test_v_start_is_the_sum_over_the_onsets_with_bin_below_the_horizon(toy_s1, toy_uv, j, horizon):
    assert np.allclose(value.v_start(toy_s1, toy_uv)[:, j], sum_over_onsets(toy_s1, toy_uv, horizon // C.BIN_MS), equal_nan=True)


def test_v_start_does_not_read_the_onsets(toy_s1, toy_uv):
    class NoOnsets:
        def __init__(self, s):
            self.s = s

        def array(self, name):
            if name.startswith('onsets/'):
                raise AssertionError(f'v_start read {name}')
            return self.s.array(name)
    assert np.allclose(value.v_start(NoOnsets(toy_s1), toy_uv), TOY_V_START, equal_nan=True)


def test_v_start_counts_the_unit_whose_aftermath_runs_past_the_window(toy_s1, toy_uv):
    # row 0 (t0 = 120,300): the first kill of unit 1 is at 150,000, inside H = 60 s; its conversions (190,000 and 200,000) are outside the
    # window and in the value all the same (the window is for the first kill)
    assert value.v_start(toy_s1, toy_uv)[0, 0] == pytest.approx(TOY_F0_UNIT[0] + TOY_F0_CONV[0] + TOY_F0_UNIT[1] + TOY_F0_CONV[1])


def test_v_start_leaves_out_the_unit_in_progress_and_the_unit_after_the_cut(toy_s1, toy_uv):
    v = value.v_start(toy_s1, toy_uv)
    assert v[3, 0] == 0.0 and v[3, 1] == 0.0       # row 3 (t0 = 300,300): unit 2 began at 300,000 and is in progress, so not in it
    assert v[2, 0] == 520.0                                          # row 2 (t0 = 240,300): the same unit is an occurrence of this row
    assert (v[7:10] == 0.0).all() and v[8, 2] == 0.0               # unit 5 (680,000) is after T - 30 s = 670,000: in no window of its match


def test_v_start_with_other_horizons(toy_s1, toy_uv):
    got = value.v_start(toy_s1, toy_uv, horizons_ms=(10_000, 30_000, 60_000, 90_000))
    assert got.shape == (TOY_N_ROWS, 4)
    assert got[0].tolist() == pytest.approx([0.0, 1260.0, 1260.0, 1260.0])        # row 0 reaches unit 0 at 135,000, unit 1 at 150,000
    assert np.allclose(got[:, 2], TOY_V_START[:, 0], equal_nan=True)
    assert value.v_start(toy_s1, toy_uv, horizons_ms=(60_000,)).shape == (TOY_N_ROWS, 1)


def test_v_start_default_horizons_are_the_evaluation_horizons(toy_s1, toy_uv):
    assert np.array_equal(value.v_start(toy_s1, toy_uv), value.v_start(toy_s1, toy_uv, horizons_ms=C.EVAL_H_MS), equal_nan=True)


BOUNDARY_FIRST_KILLS = (180_300, 240_301, 330_000, 420_300, 470_000, 470_001)     # the first kills of the six units of boundary_record
# V_start in gold (each unit 300) for the rows t0 = 120,300 + 60,000 j (j = 0 .. 5, hi = min(t0 + 180,000, 470,000)) and H = 60, 120, 180 s:
#   unit f = 180,300  is the t0 of row 1 (in progress there: out) and t0 + 60 s of row 0 (the end of the window: in)
#   unit f = 240,301  is one ms after the t0 of row 2 (in); 330,000 is in rows 1 - 4 as far as H reaches it
#   unit f = 420,300  is the t0 of row 5 (out), the hi of row 2 (in) and t0 + 120 s of row 3 (in)
#   unit f = 470,000  is T - 30 s: in scope, and the hi of rows 3 - 5 (in); unit f = 470,001 is out of scope and in no window
BOUNDARY_V_START = np.array([[300.0, 300.0, 600.0], [0.0, 300.0, 600.0], [300.0, 600.0, 900.0], [300.0, 600.0, 900.0],
                             [300.0, 600.0, 600.0], [300.0, 300.0, 300.0]])


def boundary_record(key, patch):
    """T = 500,000 ms (so T - 30 s = 470,000 and the frames are 0, 60,300, ..., 420,300: rows t0 = 120,300 ... 420,300); one kill of blue 1
    on red 6 (bounty 300, no assister) for each of ``BOUNDARY_FIRST_KILLS``, far apart in space, so that each is a unit of its own."""
    places = ((1_000, 1_000), (3_000, 1_000), (5_000, 1_000), (7_000, 1_000), (9_000, 1_000), (13_000, 13_000))
    kills = [value_kill(t, x, y, 1, 6) for t, (x, y) in zip(BOUNDARY_FIRST_KILLS, places)]
    return synth.make_record(T_ms=500_000, kills=kills, patch=patch, key=key)


@pytest.fixture(scope='module')
def boundary_world(tmp_path_factory):
    folder = tmp_path_factory.mktemp('boundary')
    s1 = build_value_s1(folder / 's1', [('15.18', toy_key(0), 'FIT')], lambda key, patch: boundary_record(key, patch))
    uv = value.unit_values(s1, TOY_PRICES, rulepay.P3_RULES, np.ones(6, dtype=np.int16))
    yield types.SimpleNamespace(s1=s1, uv=uv)
    del s1
    gc.collect()
    shutil.rmtree(folder, ignore_errors=True)


def test_v_start_at_the_edges_of_the_window_by_hand(boundary_world):
    w = boundary_world
    assert np.asarray(w.s1.array('units/f')).tolist() == list(BOUNDARY_FIRST_KILLS) and w.uv.f0_unit.tolist() == [300.0] * 6
    assert np.asarray(w.s1.array('units/in_scope')).tolist() == [True] * 5 + [False]
    assert np.asarray(w.s1.array('frame_rows/t0')).tolist() == [120_300 + 60_000 * j for j in range(6)]
    assert np.array_equal(value.v_start(w.s1, w.uv), BOUNDARY_V_START)


def test_v_start_at_the_edges_equals_the_onsets(boundary_world):
    w = boundary_world
    for j, h in enumerate(C.EVAL_H_MS):
        assert np.array_equal(value.v_start(w.s1, w.uv)[:, j], sum_over_onsets(w.s1, w.uv, h // C.BIN_MS))


def test_v_start_never_looks_past_the_end_of_the_occurrence_window(boundary_world):
    """A horizon longer than the 180 s of the rows reaches no further than hi: row 0 with H = 240 s ends at hi = 300,300, not at 360,300
    (where the unit at 330,000 would be)."""
    w = boundary_world
    got = value.v_start(w.s1, w.uv, horizons_ms=(180_000, 240_000, 600_000))
    assert np.array_equal(got[:, 0], got[:, 1]) and np.array_equal(got[:, 0], got[:, 2]) and got[0, 1] == 600.0


def test_v_start_leaves_out_a_unit_whose_scope_flag_is_off(toy_s1, toy_uv, tmp_path):
    """In S1 every unit in a window is in scope, so the flag cannot differ from the window; a store with the flag of unit 0 switched off
    shows that V_start asks for it: row 0 then holds unit 1 alone."""
    root = tmp_path / 'copy'
    shutil.copytree(toy_s1.root, root)
    flag = np.load(root / 'units' / 'in_scope.npy', mmap_mode='r+')
    flag[0] = False
    flag.flush()
    del flag
    gc.collect()
    got = value.v_start(Store(root), toy_uv)
    assert got[0, 0] == pytest.approx(TOY_F0_UNIT[1] + TOY_F0_CONV[1]) and got[0, 0] != TOY_V_START[0, 0]
    assert np.allclose(got[1:], TOY_V_START[1:], equal_nan=True)


@pytest.mark.parametrize('bad', [(), (0,), (-60_000,), (60_000.5,), (True,), ('60',), (60_000, 0), None, 60_000])
def test_v_start_refuses_horizons_that_are_not_positive_whole_ms(toy_s1, toy_uv, bad):
    with pytest.raises(ValueError):
        value.v_start(toy_s1, toy_uv, horizons_ms=bad)


def test_v_start_refuses_labels_of_another_store(toy_s1, toy_uv):
    short = dataclasses.replace(toy_uv, f0_unit=toy_uv.f0_unit[:5], f0_conv=toy_uv.f0_conv[:5])
    with pytest.raises(ValueError):
        value.v_start(toy_s1, short)


def test_v_start_of_a_valued_match_without_units_is_zero_not_nan(toy_s1, toy_uv):
    assert (value.v_start(toy_s1, toy_uv)[17:22] == 0.0).all()


# ---- the accounting identity (ruling R11) -----------------------------------------------------------------------------------------------
def test_accounting_of_the_toy_matches(toy_s1, toy_uv, toy_prices, toy_rules):
    acc = value.accounting(toy_s1, toy_uv, toy_prices, toy_rules, TOY_LEVELS)
    assert acc['max_abs_error'] == 0.0 and type(acc['max_abs_error']) is float
    assert acc['max_abs_value_error'] < 1e-9
    assert (acc['n_matches'], acc['n_valued_matches']) == (4, 3)
    total = acc['total']
    obj = total['objectives']
    assert {k: obj[k] for k in TOY_OBJECTIVE_COUNTS} == TOY_OBJECTIVE_COUNTS and obj['both'] == 0
    assert total['kills'] == {'champion': TOY_KILL_COUNTS['champion'], 'in_units': TOY_KILL_COUNTS['champion'],
                              'executions': TOY_KILL_COUNTS['executions'], 'all': TOY_KILL_COUNTS['all']}
    assert total['kill_gold']['kills'] == pytest.approx(list(TOY_KILL_GOLD_TOTAL))
    assert total['kill_gold']['units'] == pytest.approx(list(TOY_KILL_GOLD_TOTAL))
    for name in ('all', 'attributed', 'converted', 'rest'):
        assert total['objective_value'][name] == pytest.approx(list(TOY_OBJ_VALUE_TOTAL[name])), name


def test_accounting_has_per_match_figures_and_the_total_is_json(toy_s1, toy_uv, toy_prices, toy_rules):
    acc = value.accounting(toy_s1, toy_uv, toy_prices, toy_rules, TOY_LEVELS)
    pm = acc['per_match']
    for name in ('objective_count_error', 'kill_count_error', 'kill_gold_error', 'objective_value_error'):
        assert pm[name].shape == (4,), name
    assert (pm['objective_count_error'] == 0).all() and (pm['kill_count_error'] == 0).all()
    assert (np.abs(pm['kill_gold_error']) < 1e-9).all() and (np.abs(pm['objective_value_error']) < 1e-9).all()
    json.dumps({k: v for k, v in acc.items() if k != 'per_match'}, allow_nan=False)


def test_accounting_counts_the_objectives_that_have_no_group_or_no_taker(toy_s1, toy_uv, toy_prices, toy_rules):
    obj = value.accounting(toy_s1, toy_uv, toy_prices, toy_rules, TOY_LEVELS)['total']['objectives']
    assert (obj['unknown_group'], obj['unknown_taker']) == (2, 1)


def test_accounting_sees_an_objective_that_is_both_attributed_and_converted(toy_s1, toy_uv, toy_prices, toy_rules, tmp_path):
    root = tmp_path / 'copy'
    shutil.copytree(toy_s1.root, root)
    cu = np.load(root / 'objectives' / 'conv_unit.npy', mmap_mode='r+')
    cu[0] = 0                                                                            # objective 0 is attributed to unit 1 already
    cu.flush()
    del cu
    gc.collect()
    acc = value.accounting(Store(root), toy_uv, toy_prices, toy_rules, TOY_LEVELS)
    assert acc['max_abs_error'] >= 1.0 and acc['total']['objectives']['both'] == 1
    assert acc['per_match']['objective_count_error'].tolist() == [1, 0, 0, 0]


def test_accounting_sees_a_champion_kill_that_is_in_no_unit(toy_s1, toy_uv, toy_prices, toy_rules, tmp_path):
    root = tmp_path / 'copy'
    shutil.copytree(toy_s1.root, root)
    ku = np.load(root / 'kills' / 'unit.npy', mmap_mode='r+')
    ku[3] = -1
    ku.flush()
    del ku
    gc.collect()
    acc = value.accounting(Store(root), toy_uv, toy_prices, toy_rules, TOY_LEVELS)
    kills = acc['total']['kills']
    assert acc['max_abs_error'] >= 1.0 and kills['champion'] - kills['in_units'] == 1
    assert acc['per_match']['kill_count_error'].tolist()[0] >= 1 and acc['per_match']['kill_count_error'].tolist()[1:] == [0, 0, 0]


def edited_copy(toy_s1, tmp_path, name, index, new):
    """A copy of the toy S1 store in which ``name`` (for example 'units/O') has the entry ``index`` set to ``new``."""
    root = tmp_path / 'copy'
    shutil.copytree(toy_s1.root, root)
    a = np.load(root / (name + '.npy'), mmap_mode='r+')
    a[index] = new
    a.flush()
    del a
    gc.collect()
    return Store(root)


@pytest.mark.parametrize('name, index, new', [('units/O', (1, 0), 3), ('units/O', (3, 12), 2), ('units/conv', (1, 4), 2),
                                              ('units/conv', (2, 0), 1)])
def test_accounting_sees_a_unit_whose_objectives_are_not_those_of_s1(toy_s1, toy_uv, toy_prices, toy_rules, tmp_path, name, index, new):
    acc = value.accounting(edited_copy(toy_s1, tmp_path, name, index, new), toy_uv, toy_prices, toy_rules, TOY_LEVELS)
    assert acc['max_abs_error'] >= 1.0 and acc['per_match']['objective_count_error'].tolist()[0] >= 1


def test_accounting_sees_a_unit_whose_champion_kills_are_not_those_of_s1(toy_s1, toy_uv, toy_prices, toy_rules, tmp_path):
    acc = value.accounting(edited_copy(toy_s1, tmp_path, 'units/K', (1, 0), 5), toy_uv, toy_prices, toy_rules, TOY_LEVELS)
    assert acc['max_abs_error'] >= 3.0 and acc['per_match']['kill_count_error'].tolist()[0] >= 3


def test_accounting_counts_a_kill_in_a_unit_of_another_match_as_not_in_a_unit(toy_s1, toy_uv, toy_prices, toy_rules, tmp_path):
    acc = value.accounting(edited_copy(toy_s1, tmp_path, 'kills/unit', 0, 7), toy_uv, toy_prices, toy_rules, TOY_LEVELS)
    kills = acc['total']['kills']
    assert acc['max_abs_error'] >= 1.0 and kills['champion'] - kills['in_units'] == 1


def test_accounting_sees_labels_that_do_not_belong_to_the_store(toy_s1, toy_uv, toy_prices, toy_rules):
    wrong = dataclasses.replace(toy_uv, n_kills=toy_uv.n_kills + np.array([[1, 0]] + [[0, 0]] * 9, dtype=np.int32))
    assert value.accounting(toy_s1, wrong, toy_prices, toy_rules, TOY_LEVELS)['max_abs_error'] >= 1.0
    with pytest.raises(ValueError) as raised:
        value.accounting(toy_s1, dataclasses.replace(toy_uv, kill_gold=toy_uv.kill_gold[:4]), toy_prices, toy_rules, TOY_LEVELS)
    assert 'unit' in message_of(raised)


def test_accounting_value_error_is_a_gold_diagnostic_that_sees_a_wrong_sum(toy_s1, toy_uv, toy_prices, toy_rules):
    kg = toy_uv.kill_gold.copy()
    kg[1, 0] += 5.0                                                                       # 5 gold missing from the labels
    acc = value.accounting(toy_s1, dataclasses.replace(toy_uv, kill_gold=kg), toy_prices, toy_rules, TOY_LEVELS)
    assert acc['max_abs_value_error'] == pytest.approx(5.0) and acc['max_abs_error'] == 0.0     # the counts still hold: R11
    assert acc['per_match']['kill_gold_error'][0] == pytest.approx(5.0)
    ov = toy_uv.obj_value.copy()
    ov[7, 0] += 2.5
    assert value.accounting(toy_s1, dataclasses.replace(toy_uv, obj_value=ov), toy_prices, toy_rules,
                            TOY_LEVELS)['max_abs_value_error'] == pytest.approx(2.5)


def test_accounting_refuses_levels_it_cannot_use(toy_s1, toy_uv, toy_prices, toy_rules):
    bad = TOY_LEVELS.copy()
    bad[0] = -1
    with pytest.raises(ValueError):
        value.accounting(toy_s1, toy_uv, toy_prices, toy_rules, bad)


def test_accounting_asks_for_the_roles_of_the_labels(toy_s1, toy_uv, toy_prices, toy_rules):
    with pytest.raises(ValueError, match='roles'):
        value.accounting(toy_s1, toy_uv, toy_prices, toy_rules, TOY_LEVELS, roles=('FIT',))
    fit = value.unit_values(toy_s1, toy_prices, toy_rules, TOY_LEVELS, roles=('FIT',))
    acc = value.accounting(toy_s1, fit, toy_prices, toy_rules, TOY_LEVELS, roles=('FIT',))
    assert acc['max_abs_error'] == 0.0 and acc['n_valued_matches'] == 1 and acc['total']['objectives']['all'] == 11
    assert acc['total']['kills']['champion'] == 8


# ---- the label store ------------------------------------------------------------------------------------------------------------------
PROVENANCE = {'prices_sha256': 'a' * 64, 'rules_sha256': 'b' * 64, 's1_files_sha256': 'c' * 64, 'code_commit': 'd' * 40,
              'code_dirty': False}
LABEL_ARRAYS = {'labels/kill_gold': ('f8', (TOY_N_UNITS, 2)), 'labels/obj_value': ('f8', (TOY_N_UNITS, 2)),
                'labels/conv_value': ('f8', (TOY_N_UNITS, 2)), 'labels/f0_unit': ('f8', (TOY_N_UNITS,)),
                'labels/f0_conv': ('f8', (TOY_N_UNITS,)), 'labels/winner': ('i1', (TOY_N_UNITS,)), 'labels/sure': ('?', (TOY_N_UNITS,)),
                'labels/gold_bin': ('i1', (TOY_N_UNITS, 2)), 'labels/v_start': ('f8', (TOY_N_ROWS, 3)), 'labels/gold_reps': ('f8', (31,))}


@pytest.fixture
def written(toy_s1, toy_uv, tmp_path):
    reps = value.gold_bin_means(toy_uv, value.role_units(toy_s1, ('FIT',)))[0]
    out = value.write_labels(tmp_path / 'labels', toy_s1, toy_uv, value.v_start(toy_s1, toy_uv), reps, dict(PROVENANCE))
    yield types.SimpleNamespace(out=out, reps=reps, store=Store(out))
    gc.collect()


def test_write_labels_writes_a_complete_store_with_every_array(written, toy_uv):
    assert written.out == written.store.root and isinstance(written.out, Path)
    assert written.store.manifest['complete'] is True
    written.store.verify()
    assert sorted(written.store.manifest['arrays']) == sorted(LABEL_ARRAYS)
    for name, (dtype, shape) in LABEL_ARRAYS.items():
        a = written.store.array(name)
        assert a.dtype == np.dtype(dtype) and a.shape == shape, name


def test_write_labels_stores_the_values_of_the_labels(written, toy_uv, toy_s1):
    s = written.store
    for name in ('kill_gold', 'obj_value', 'conv_value', 'f0_unit', 'f0_conv', 'winner', 'sure', 'gold_bin'):
        assert np.array_equal(np.asarray(s.array(f'labels/{name}')), getattr(toy_uv, name), equal_nan=True), name
    assert np.array_equal(np.asarray(s.array('labels/v_start')), TOY_V_START, equal_nan=True)
    assert np.array_equal(np.asarray(s.array('labels/gold_reps')), written.reps) and written.reps.shape == (31,)


def test_labels_json_is_the_provenance_and_nothing_more(written):
    text = (written.out / 'LABELS.json').read_text(encoding='utf-8')
    assert json.loads(text) == PROVENANCE
    assert 'LABELS.json' in written.store.manifest['files'] and sorted(written.store.manifest['files']) == sorted(
        [f'{n}.npy' for n in LABEL_ARRAYS] + ['LABELS.json'])


def test_write_labels_leaves_no_match_key_in_any_file(written):
    for p in written.out.rglob('*'):
        if p.is_file():
            assert KEY_PREFIX.encode() not in p.read_bytes(), p.name


def test_write_labels_refuses_to_write_over_a_store(toy_s1, toy_uv, written, tmp_path):
    reps = written.reps
    with pytest.raises(FileExistsError):
        value.write_labels(written.out, toy_s1, toy_uv, value.v_start(toy_s1, toy_uv), reps, dict(PROVENANCE))
    assert Store(written.out).manifest['complete'] is True


def test_write_labels_refuses_arrays_that_do_not_fit_the_store(toy_s1, toy_uv, tmp_path):
    v, reps = value.v_start(toy_s1, toy_uv), value.gold_bin_means(toy_uv, np.arange(9))[0]
    for i, (vs, gr, uv) in enumerate([(v[:-1], reps, toy_uv), (v[:, :2], reps, toy_uv), (v, reps[:-1], toy_uv),
                                      (v, reps, dataclasses.replace(toy_uv, f0_unit=toy_uv.f0_unit[:-1]))]):
        with pytest.raises(ValueError):
            value.write_labels(tmp_path / f'bad{i}', toy_s1, uv, vs, gr, dict(PROVENANCE))
        assert not (tmp_path / f'bad{i}' / 'STORE.json').exists()


def test_write_labels_refuses_provenance_that_is_not_json(toy_s1, toy_uv, tmp_path):
    v, reps = value.v_start(toy_s1, toy_uv), value.gold_bin_means(toy_uv, np.arange(9))[0]
    for i, bad in enumerate([None, [1], {'x': float('nan')}, {'x': object()}, {1: 2}]):
        with pytest.raises((ValueError, TypeError)):
            value.write_labels(tmp_path / f'bad{i}', toy_s1, toy_uv, v, reps, bad)
        assert not (tmp_path / f'bad{i}' / 'STORE.json').exists()


# ---- many random matches against the brute-force oracle ----------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def random_world(tmp_path_factory):
    folder = tmp_path_factory.mktemp('random_value')
    matches = random_value_matches(12)
    s1 = build_value_s1(folder / 's1', matches, random_value_record)
    with empty_process_list():
        levels = value.kill_levels(s1, S3_ROLES, workers=1, records=random_value_record)
    per_match, v_start = oracle_world(matches, random_value_record, TOY_PRICES, rulepay.P3_RULES)
    uv = value.unit_values(s1, TOY_PRICES, rulepay.P3_RULES, levels)
    yield types.SimpleNamespace(s1=s1, matches=matches, levels=levels, per_match=per_match, v_start=v_start, uv=uv)
    del s1
    gc.collect()
    shutil.rmtree(folder, ignore_errors=True)


def test_the_random_matches_reach_the_cases_they_are_there_for(random_world):
    w = random_world
    kills = np.asarray(w.s1.array('kills/killer'))
    assert len(w.uv.f0_unit) > 100 and (kills == 0).sum() > 0 and (w.uv.winner == 0).sum() > 0 and (w.uv.winner == 1).sum() > 10
    assert (w.uv.winner == -1).sum() > 10 and w.uv.sure.sum() > 10 and (~w.uv.sure[w.uv.valued]).sum() > 3
    assert (np.asarray(w.s1.array('objectives/unit')) >= 0).sum() > 20 and (np.asarray(w.s1.array('objectives/conv_unit')) >= 0).sum() > 5
    assert (w.levels[w.levels > 0] > 1).sum() > 100 and not w.uv.valued.all() and w.uv.valued.any()
    assert (w.uv.gold_bin[w.uv.valued] >= 0).any() and (w.uv.gold_bin == -1).any()
    assert np.isfinite(w.v_start).any() and np.isnan(w.v_start).any() and (w.v_start[np.isfinite(w.v_start)] != 0).sum() > 50


def test_random_levels_equal_the_event_oracle(random_world):
    want = []
    for patch, key, macro in random_world.matches:
        rec = random_value_record(key, patch)
        kt = units.kill_table(rec)
        want += [level_of(rec, int(v), int(t)) if macro in S3_ROLES else -1 for t, v in zip(kt.t, kt.victim)]
    assert random_world.levels.tolist() == want


def test_random_unit_values_equal_the_brute_force_oracle(random_world):
    assert_unit_values_equal_oracle(random_world.uv, random_world.per_match)


def test_random_v_start_equals_the_oracle_and_the_onsets(random_world):
    got = value.v_start(random_world.s1, random_world.uv)
    assert np.allclose(got, random_world.v_start, atol=1e-9, equal_nan=True)
    for j, h in enumerate(C.EVAL_H_MS):
        assert np.allclose(got[:, j], sum_over_onsets(random_world.s1, random_world.uv, h // C.BIN_MS), atol=1e-9, equal_nan=True)


def test_random_v_start_with_other_horizons_equals_the_oracle(random_world):
    horizons = (20_000, 45_000, 100_000, 180_000)
    _, want = oracle_world(random_world.matches, random_value_record, TOY_PRICES, rulepay.P3_RULES, horizons=horizons)
    assert np.allclose(value.v_start(random_world.s1, random_world.uv, horizons_ms=horizons), want, atol=1e-9, equal_nan=True)


def test_random_accounting_is_exact(random_world):
    acc = value.accounting(random_world.s1, random_world.uv, TOY_PRICES, rulepay.P3_RULES, random_world.levels)
    assert acc['max_abs_error'] == 0.0 and acc['max_abs_value_error'] < 1e-6
    assert acc['total']['objectives']['all'] == acc['total']['objectives']['attributed'] + acc['total']['objectives']['converted'] + acc[
        'total']['objectives']['rest']


def test_random_bin_means_equal_the_oracle(random_world):
    s1, uv = random_world.s1, random_world.uv
    role = np.asarray(s1.array('matches/role'))[np.asarray(s1.array('units/match'))]
    fit = role == C.ROLES.index('FIT')
    means, fallback = value.gold_bin_means(uv, fit)
    entries = []                                                                    # (bin, gold) of every FIT team entry with a kill
    for (patch, key, macro), (dicts, u) in zip(random_world.matches, random_world.per_match):
        if macro == 'FIT':
            entries += [(min(int(g // 100), 30), g) for d in dicts for g, n in zip(d['kill_gold'], d['n_kills']) if n >= 1]
    for b in range(31):
        golds = [g for bb, g in entries if bb == b]
        if golds:
            assert means[b] == pytest.approx(sum(golds) / len(golds)) and b not in fallback
        else:
            assert means[b] == (3000.0 if b == 30 else 100.0 * b + 50.0) and b in fallback
    assert len(entries) > 20


def test_random_labels_survive_a_round_trip_through_the_store(random_world, tmp_path):
    reps = value.gold_bin_means(random_world.uv, value.role_units(random_world.s1, ('FIT',)))[0]
    out = value.write_labels(tmp_path / 'labels', random_world.s1, random_world.uv, value.v_start(random_world.s1, random_world.uv), reps,
                             dict(PROVENANCE))
    s = Store(out)
    s.verify()
    assert np.array_equal(np.asarray(s.array('labels/f0_unit')), random_world.uv.f0_unit, equal_nan=True)
    del s
    gc.collect()


# ---- how the module is imported ----------------------------------------------------------------------------------------------------------
def test_no_module_level_import_reaches_the_s1_runner():
    imports = module_level_imports(Path(value.__file__))
    chain = import_watch_names(('c_ept.run_s1', 'c_ept.run_s2', 'c_ept.store_build', 'c_ept.cohort', 'c_ept.runlog'))
    assert not [i for i in imports if i in chain or i.startswith(chain)], sorted(imports)


def test_the_e6_guard_is_imported_inside_kill_levels():
    tree = ast.parse(Path(value.__file__).read_text(encoding='utf-8'))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'kill_levels')
    inner = [n for n in ast.walk(fn) if isinstance(n, ast.ImportFrom) and n.module == 'c_ept.workflows' and any(a.name == 'run_s1' for a in n.names)]
    assert inner
    guard = [n for n in ast.walk(fn) if isinstance(n, ast.Attribute) and n.attr == 'require_no_strata']
    assert guard


@pytest.mark.parametrize('module', ['c_ept.value', 'c_ept.valuation.value'])
def test_importing_the_module_loads_no_s1_module(module):
    chain = import_watch_names(('c_ept.run_s1', 'c_ept.cohort', 'c_ept.store_build', 'c_ept.runlog', 'c_ept.run_s2'))
    code = (PROTECTED_IMPORT_PRELUDE + f'import sys, {module}\n'
            f'names = {chain!r}\n'
            'print(sorted(m for m in names if m in sys.modules))')
    done = subprocess.run([sys.executable, '-B', '-c', code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0 and done.stdout.strip() == '[]', done.stderr[-500:]


def test_the_module_names_no_array_the_labels_do_not_need():
    tree = ast.parse(Path(value.__file__).read_text(encoding='utf-8'))
    named = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for prefix in ('event_rows/', 'onsets/', 'frames/', 'objectives/x', 'objectives/y', 'kills/x', 'kills/y', 'matches/champ'):
        assert not [v for v in named if v.startswith(prefix)], prefix


def test_flowprice_groups_are_the_groups_of_the_price_table():
    assert flowprice.GROUPS == GROUPS and tuple(TOY_PRICES) == GROUPS


# ---- the "all flow prices" table and the repricing of the units (S4b preparation M13, decision D19) --------------------------------------------
FIXED_IMMEDIATE_D19 = ('tower_base', 'tower_nexus', 'inhibitor')           # decision D19: the base and nexus towers and the inhibitors keep the immediate value in the sensitivity


def write_price_table(tmp_path, flow=None, mutate=None, name='PRICES.json'):
    """A toy PRICES.json with the fields of ``flowprice.price_table`` that matter here, for every group: ``value`` (the price in use), ``used``, ``fixed_immediate``, ``price_flow`` and ``immediate``.
    The immediate values are the toy prices; unless ``flow`` says otherwise the flow price of the group number i is the immediate value + 10 (i + 1). As in the real file the elder is the one group
    whose flow price is in use, and a fixed-immediate group has a flow price too, which nothing may read."""
    flow = {} if flow is None else flow
    doc = {}
    for i, g in enumerate(GROUPS):
        pf = flow[g] if g in flow else TOY_PRICES[g] + 10.0 * (i + 1)
        used = 'flow' if g == 'elder' else 'immediate'
        doc[g] = {'value': pf if used == 'flow' else TOY_PRICES[g], 'used': used, 'fixed_immediate': g in FIXED_IMMEDIATE_D19, 'price_flow': pf, 'immediate': TOY_PRICES[g]}
    doc.update({'kill_gold_coef': {'kill_gold_coef': 1.12}, 'B': 2000, 'seed': 20261021, 'n_matches': 19100})
    if mutate is not None:
        mutate(doc)
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding='utf-8')
    return path


def test_flow_price_table_uses_the_flow_price_of_every_group_but_the_fixed_immediate_ones(tmp_path):
    """D19: the sensitivity "all flow prices" prices every group by its flow price except the base and nexus towers and the inhibitors, which stay at the immediate value (the ``value`` of the file).
    The fixed groups are the ones of ``flowprice.FIXED_IMMEDIATE``, as D19 words them, and the flow price in their entry is never read. The table is the dict that ``Composer`` and ``unit_values`` take;
    the primary table of ``load_prices`` is not changed by the new reader."""
    path = write_price_table(tmp_path)
    table = value.load_flow_prices(path)
    assert isinstance(table, dict) and list(table) == list(GROUPS) and all(type(v) is float for v in table.values())
    for i, g in enumerate(GROUPS):
        assert table[g] == (TOY_PRICES[g] if g in FIXED_IMMEDIATE_D19 else TOY_PRICES[g] + 10.0 * (i + 1)), g
    assert tuple(table.fixed_immediate) == FIXED_IMMEDIATE_D19 == tuple(flowprice.FIXED_IMMEDIATE) and tuple(table.fallback) == ()
    primary = value.load_prices(path)
    assert primary == {g: (TOY_PRICES[g] + 10.0 * (i + 1) if g == 'elder' else TOY_PRICES[g]) for i, g in enumerate(GROUPS)}      # the file's own value, as before
    assert primary != table and all(primary[g] == table[g] for g in FIXED_IMMEDIATE_D19 + ('elder',))
    assert table == dict(table) and value.object_price(np.array([6]), np.array([1]), table).tolist() == [table['tower_inner']]      # usable wherever a price table goes


def test_flow_price_table_falls_back_when_flow_is_missing(tmp_path):
    """A group whose flow price was not estimable (its ``price_flow`` is None: collinear, too few events) keeps its ``value`` and the table says which groups did: ``fallback``, in the order of the groups.
    A fixed-immediate group with no flow price is not a fallback (it never reads the flow price)."""
    path = write_price_table(tmp_path, flow={'atakhan': None, 'grubs': None, 'tower_nexus': None})
    table = value.load_flow_prices(path)
    assert table['grubs'] == TOY_PRICES['grubs'] and table['atakhan'] == TOY_PRICES['atakhan']
    assert tuple(table.fallback) == ('grubs', 'atakhan')
    assert table['tower_nexus'] == TOY_PRICES['tower_nexus']
    assert table['dragon'] == TOY_PRICES['dragon'] + 10.0 and table['baron'] == TOY_PRICES['baron'] + 50.0       # the other groups read their flow price
    both = write_price_table(tmp_path, flow={'grubs': None}, mutate=lambda d: d['grubs'].update(value=None), name='both.json')
    with pytest.raises(ValueError, match='grubs'):
        value.load_flow_prices(both)


@pytest.mark.parametrize('mutate, group', [
    (lambda d: d.pop('baron'), 'baron'),
    (lambda d: d['baron'].pop('price_flow'), 'baron'),
    (lambda d: d['baron'].update(price_flow=float('nan')), 'baron'),
    (lambda d: d['baron'].update(price_flow='1830'), 'baron'),
    (lambda d: d['baron'].update(price_flow=True), 'baron'),
    (lambda d: d['tower_base'].update(value=None), 'tower_base'),
    (lambda d: d['inhibitor'].update(value=float('inf')), 'inhibitor'),
    (lambda d: d['dragon'].update(fixed_immediate=True), 'dragon'),               # the file disagrees with the code about which groups are fixed
    (lambda d: d['tower_base'].update(fixed_immediate=False), 'tower_base')])
def test_flow_price_table_refuses_what_it_cannot_read_and_names_the_group(tmp_path, mutate, group):
    with pytest.raises(ValueError, match=group):
        value.load_flow_prices(write_price_table(tmp_path, mutate=mutate))


def test_flow_price_table_refuses_a_file_that_is_not_a_price_table(tmp_path):
    for text in ('[1, 2]', '"x"', '{ not json'):
        p = tmp_path / 'bad.json'
        p.write_text(text, encoding='utf-8')
        with pytest.raises(ValueError):
            value.load_flow_prices(p)


FIELDS = tuple(f.name for f in dataclasses.fields(value.UnitValues))


def same_labels(a, b):
    for name in FIELDS:
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y, equal_nan=x.dtype.kind == 'f'), name


def test_repricing_the_units_is_unit_values_made_with_the_other_prices(toy_s1, toy_uv, toy_prices, toy_rules):
    """The truth of the sensitivity is recomputed in memory: the kill gold, the gold bins and the kill counts of the labels stay, the objective values, f0, f0 of the conversions and the winner are those
    of ``unit_values`` made with the other prices, in every bit; with the prices of the labels it is the labels themselves, and V_start of the repriced units is that of the new labels."""
    flow = {g: p + 7.5 * (i + 1) for i, (g, p) in enumerate(toy_prices.items())}
    same_labels(value.reprice_units(toy_s1, toy_uv, toy_prices), toy_uv)
    other = value.reprice_units(toy_s1, toy_uv, flow)
    want = value.unit_values(toy_s1, flow, toy_rules, TOY_LEVELS)
    same_labels(other, want)
    assert np.array_equal(value.v_start(toy_s1, other), value.v_start(toy_s1, want), equal_nan=True)
    assert not np.array_equal(other.f0_unit, toy_uv.f0_unit, equal_nan=True) and np.array_equal(other.kill_gold, toy_uv.kill_gold, equal_nan=True)      # the prices moved f0 (control), not the kill gold
    assert not other.f0_unit.flags.writeable and not other.obj_value.flags.writeable
    with pytest.raises(ValueError, match='roles'):
        value.reprice_units(toy_s1, toy_uv, flow, roles=('FIT',))                            # labels of all three roles, repriced for one
    fit = value.unit_values(toy_s1, toy_prices, toy_rules, TOY_LEVELS, roles=('FIT',))
    same_labels(value.reprice_units(toy_s1, fit, flow, roles=('FIT',)), value.unit_values(toy_s1, flow, toy_rules, TOY_LEVELS, roles=('FIT',)))
    with pytest.raises(ValueError):
        value.reprice_units(toy_s1, toy_uv, {k: v for k, v in flow.items() if k != 'plate'})
