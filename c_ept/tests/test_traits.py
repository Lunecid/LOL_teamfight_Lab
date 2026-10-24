"""Tests of ``c_ept.traits``: level-scaled champion traits (no ids), the level clock and the legacy item vectors of the ten players.

The golden tests (``golden_records``, 20 real matches of the S1 golden lock) run only with ``C_EPT_S1_RUN`` set; the other tests use
synthetic matches and the pinned LEGACY tables. Failure messages carry match numbers (the position in the golden list), never match
keys. Run with ``--tb=short``.
"""
import ast
import dataclasses
import errno
import functools
import inspect
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pytest

from c_ept import casts
from c_ept import config as C
from c_ept import legacy, reader, run_s1, traits
from c_ept.checks import common, explain_mismatch, measure_levels, measure_levels_wide, sweep_s1
from c_ept.reader import MatchFormatError
from c_ept.tests import synth
from c_ept.tests.conftest import UnreadableTestFile, golden_folder, read_json

COL = {name: i for i, name in enumerate(traits.TRAIT_NAMES)}


def rec_with(champs=None, **kw):
    """A synthetic match (patch 15.18 unless told otherwise) with the given ten Data Dragon champion keys."""
    rec = synth.make_record(**kw)
    return rec if champs is None else dataclasses.replace(rec, champ=np.array(champs, dtype=np.int32))


def level_up(t, pid, level):
    return {'type': 'LEVEL_UP', 'timestamp': t, 'participantId': pid, 'level': level}


def buy(t, pid, item):
    return {'type': 'ITEM_PURCHASED', 'timestamp': t, 'participantId': pid, 'itemId': item}


def factor(n):
    """(n - 1)(0.7025 + 0.0175 (n - 1)) written out for the levels the hand tests use (hand arithmetic: 0.72, 7.74, 17)."""
    return {1: 0.0, 2: 0.72, 10: 7.74, 18: 17.0}[n]


# ---- reading test files without showing a match key --------------------------------------------------------------------------------
SENTINEL = 'SENTINEL-MATCH-KEY-4711'          # stands for a match key in the name of a file


def shown(exc: BaseException) -> str:
    """Everything a test report can show of an exception: its text, its repr, its args and the whole formatted traceback with its chain."""
    return ' | '.join([str(exc), repr(exc), repr(exc.args), ''.join(traceback.format_exception(exc))])


@pytest.mark.parametrize('case, said', [('missing', 'missing'), ('directory', 'unreadable'), ('invalid_json', 'not valid'),
                                        ('invalid_utf8', 'not valid')])
def test_read_json_does_not_show_the_path_of_a_file_it_cannot_read(tmp_path, case, said):
    """``Path.read_text`` puts the whole path into FileNotFoundError, and a short traceback prints it; the path of a match file holds the
    match key. The helper names the kind of file and what is wrong, and nothing in the exception or its chain holds the path."""
    path = tmp_path / f'{SENTINEL}.events.json'
    if case == 'directory':
        path.mkdir()
    elif case == 'invalid_json':
        path.write_text('{"a": [1, 2', encoding='utf-8')
    elif case == 'invalid_utf8':
        path.write_bytes(b'{"a": "\xff\xfe"}')
    with pytest.raises(UnreadableTestFile) as err:
        read_json(path, 'cache events.json')
    exc = err.value
    assert str(exc).startswith('cache events.json ') and said in str(exc)
    assert SENTINEL not in shown(exc) and str(tmp_path) not in shown(exc)
    assert exc.__cause__ is None and exc.__context__ is None             # no chained exception, which would still hold the path


def test_read_json_returns_the_parsed_file(tmp_path):
    path = tmp_path / 'x.json'
    path.write_text('{"a": [1, 2.5, null], "b": "å"}', encoding='utf-8')
    assert read_json(path, 'any file') == {'a': [1, 2.5, None], 'b': 'å'}


def test_the_cache_reader_of_these_tests_does_not_show_the_key_of_a_missing_match():
    with pytest.raises(UnreadableTestFile, match='^cache events.json missing$') as err:
        cache_pack(SENTINEL, '15.18')
    assert SENTINEL not in shown(err.value)


# ---- growth formula ------------------------------------------------------------------------------------------------------------------
def test_stat_at_level_endpoints():
    assert traits.stat_at_level(560, 96, 1) == 560 and traits.stat_at_level(560, 96, 18) == pytest.approx(560 + 17 * 96)


def test_attack_speed_level_18():
    assert traits.attack_speed_at_level(0.61, 1.36, 18) == pytest.approx(0.61 * (1 + 17 * 1.36 / 100))


def test_growth_formula_hand_values():
    """Level 2: 1 * (0.7025 + 0.0175) = 0.72; level 10: 9 * (0.7025 + 0.0175 * 9) = 9 * 0.86 = 7.74 (worked by hand)."""
    assert traits.stat_at_level(560, 96, 2) == pytest.approx(560 + 96 * 0.72) == pytest.approx(629.12)
    assert traits.stat_at_level(560, 96, 10) == pytest.approx(560 + 96 * 7.74) == pytest.approx(1303.04)
    assert traits.attack_speed_at_level(0.61, 1.36, 10) == pytest.approx(0.61 * (1 + 1.36 * 7.74 / 100))
    assert traits.attack_speed_at_level(0.61, 1.36, 10) == pytest.approx(0.674211, abs=1e-6)       # 0.61 * 1.105264


def test_growth_formula_increase_per_level_is_the_wiki_second_form():
    """The wiki also states the increase from level n - 1 to n as g (0.65 + 0.035 n) for n >= 2: the same polynomial, differenced."""
    for n in range(2, 19):
        assert traits.stat_at_level(0.0, 1.0, n) - traits.stat_at_level(0.0, 1.0, n - 1) == pytest.approx(0.65 + 0.035 * n)


def test_growth_formula_works_on_arrays_and_refuses_levels_outside_1_to_18():
    n = np.arange(1, 19)
    assert np.allclose(traits.stat_at_level(100.0, 2.0, n), [traits.stat_at_level(100.0, 2.0, int(k)) for k in n])
    for bad in (0, 19, -3, 1.5, float('nan')):
        with pytest.raises(ValueError, match='level'):
            traits.stat_at_level(560, 96, bad)
        with pytest.raises(ValueError, match='level'):
            traits.attack_speed_at_level(0.61, 1.36, bad)


# ---- the cast rule and the weight of the import ------------------------------------------------------------------------------------
def modules_after(statement: str) -> list:
    """The c_ept and gameplay modules that a fresh interpreter has loaded after ``statement`` (run in the repository folder)."""
    code = f"import sys; {statement}; print(sorted(m for m in sys.modules if m.split('.')[0] in ('c_ept', 'gameplay')))"
    r = subprocess.run([sys.executable, '-B', '-c', code], cwd=Path(C.__file__).parents[1], capture_output=True, text=True,
                       encoding='utf-8')
    assert r.returncode == 0, r.stderr
    return ast.literal_eval(r.stdout.strip())


def test_importing_traits_loads_neither_the_legacy_modules_nor_the_s1_runner():
    """The pins are checked and LEGACY is imported on the first call, not at import: every worker of a view builder imports traits.
    The cast rule comes from ``c_ept.casts``, which does not pull in the S1 runner either."""
    loaded = modules_after('from c_ept import traits')
    assert 'c_ept.traits' in loaded and 'c_ept.casts' in loaded
    assert 'c_ept.run_s1' not in loaded and not [m for m in loaded if m.startswith('gameplay')]


def test_the_cast_rule_has_a_module_of_its_own_and_the_runner_re_exports_it():
    """``narrow(name, a, dtype)`` (S1 and S2 casts) lives in c_ept.casts, which loads nothing of c_ept but itself; run_s1 and traits use
    that same function, so the callers and tests that say ``run_s1.narrow`` keep working."""
    assert modules_after('import c_ept.casts') == ['c_ept', 'c_ept.casts']
    assert run_s1.narrow is casts.narrow and traits.narrow is casts.narrow
    assert list(inspect.signature(casts.narrow).parameters) == ['name', 'a', 'dtype']
    tree = ast.parse(Path(casts.__file__).read_text(encoding='utf-8'))
    imported = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module}
    assert imported and imported <= set(sys.stdlib_module_names) | {'numpy'}          # numpy and the standard library only


# ---- names ---------------------------------------------------------------------------------------------------------------------------


def test_trait_names_have_no_ids():
    assert not [n for n in traits.TRAIT_NAMES if n.endswith('_id') or n == 'champion']


def test_trait_names_are_the_19_of_the_brief_in_order():
    assert traits.TRAIT_NAMES == (
        'tag_Assassin', 'tag_Fighter', 'tag_Mage', 'tag_Marksman', 'tag_Support', 'tag_Tank', 'is_ranged', 'attackrange', 'partype_Mana',
        'partype_Energy', 'partype_none', 'partype_other', 'lvl_healthMax', 'lvl_armor', 'lvl_magicResist', 'lvl_attackDamage',
        'lvl_attackSpeed', 'lvl_movementSpeed', 'champ_unknown')
    assert len(traits.TRAIT_NAMES) == len(set(traits.TRAIT_NAMES)) == 19


def test_trait_names_come_from_legacy_fields():
    """Every static trait is a field of the LEGACY champion vector; every lvl_ trait has a base (and, but for speed, a growth) field."""
    fields = set(legacy.load().champion_attributes.CHAMPION_VECTOR_FIELDS)
    assert {n for n in traits.TRAIT_NAMES if not n.startswith('lvl_')} <= fields
    growth_field = {'lvl_healthMax': 'growth_healthMax', 'lvl_armor': 'growth_armor', 'lvl_magicResist': 'growth_magicResist',
                    'lvl_attackDamage': 'growth_attackDamage', 'lvl_attackSpeed': 'growth_attackSpeed_pct', 'lvl_movementSpeed': None}
    assert set(growth_field) == {n for n in traits.TRAIT_NAMES if n.startswith('lvl_')}
    for name, growth in growth_field.items():
        assert 'base_' + name[len('lvl_'):] in fields and (growth is None or growth in fields), name


def test_item_vector_names_are_the_22_legacy_names_in_legacy_order():
    names = traits.item_vector_names()
    assert names == legacy.load().item_state.ITEM_VECTOR_NAMES and len(names) == 22
    assert not [n for n in names if n.endswith('_id') or n.endswith('Id')]


# ---- champion traits -----------------------------------------------------------------------------------------------------------------
ANNIE = 1      # Data Dragon key of Annie; 15.18: health 560 + 96, armor 23 + 4, magic resist 30 + 1.3, attack damage 50 + 2.65,
#                attack speed 0.61 + 1.36 %, move speed 335, range 625, tags Mage and Support, resource Mana
OTHERS = (86, 120, 61, 51, 412, 23, 64, 238, 222)


@pytest.mark.parametrize('level', [1, 2, 10, 18])
def test_annie_traits_by_hand(level):
    rec = rec_with([ANNIE, *OTHERS])
    levels = np.ones(10, dtype=np.int16)
    levels[0] = level
    row = traits.champion_traits(rec, levels)[0]
    f = factor(level)
    want = {'tag_Assassin': 0, 'tag_Fighter': 0, 'tag_Mage': 1, 'tag_Marksman': 0, 'tag_Support': 1, 'tag_Tank': 0, 'is_ranged': 1,
            'attackrange': 625, 'partype_Mana': 1, 'partype_Energy': 0, 'partype_none': 0, 'partype_other': 0,
            'lvl_healthMax': 560 + 96 * f, 'lvl_armor': 23 + 4 * f, 'lvl_magicResist': 30 + 1.3 * f, 'lvl_attackDamage': 50 + 2.65 * f,
            'lvl_attackSpeed': 0.61 * (1 + 1.36 * f / 100), 'lvl_movementSpeed': 335, 'champ_unknown': 0}
    assert set(want) == set(traits.TRAIT_NAMES)
    assert np.allclose(row, [want[n] for n in traits.TRAIT_NAMES], rtol=1e-6, atol=0)


def test_traits_shape_dtype_and_level_only_moves_the_level_columns():
    rec = rec_with()
    low, high = (traits.champion_traits(rec, np.full(10, n, dtype=np.int16)) for n in (1, 18))
    assert low.dtype == high.dtype == np.float32 and low.shape == high.shape == (10, 19) and np.isfinite(low).all()
    moved = [n for n in traits.TRAIT_NAMES if not np.array_equal(low[:, COL[n]], high[:, COL[n]])]
    assert set(moved) <= {'lvl_healthMax', 'lvl_armor', 'lvl_magicResist', 'lvl_attackDamage', 'lvl_attackSpeed'}
    assert {'lvl_healthMax', 'lvl_armor', 'lvl_attackDamage', 'lvl_attackSpeed'} <= set(moved)      # growth is not lost
    assert np.array_equal(low[:, COL['lvl_movementSpeed']], high[:, COL['lvl_movementSpeed']])      # no movement-speed growth


def test_traits_take_frame_levels_and_level_clock_levels_alike():
    """Frame rows pass the float frame value, event-clock views the int16 value of level_at: the same levels give the same traits."""
    rec = synth.make_record()
    k = 3
    frame_levels = rec.frame_num[k, :, 2]
    assert frame_levels.dtype == np.float64
    assert np.array_equal(traits.champion_traits(rec, frame_levels), traits.champion_traits(rec, frame_levels.astype(np.int16)))


def test_unknown_champion_has_champ_unknown_one_and_nan_everywhere_else():
    champs = [ANNIE, 0, 86, 9999, 412, 23, 64, 238, 222, 99_999]       # 0, 9999 (not in the table) and 99,999 (a mode id): unknown
    out = traits.champion_traits(rec_with(champs), np.full(10, 7, dtype=np.int16))
    unknown = [1, 3, 9]
    assert out[:, COL['champ_unknown']].tolist() == [1.0 if i in unknown else 0.0 for i in range(10)]
    for i in range(10):
        others = np.delete(out[i], COL['champ_unknown'])
        assert np.isnan(others).all() if i in unknown else np.isfinite(others).all(), i
    for n in ('lvl_healthMax', 'lvl_armor', 'lvl_magicResist', 'lvl_attackDamage', 'lvl_attackSpeed', 'lvl_movementSpeed'):
        assert np.isnan(out[unknown, COL[n]]).all() and np.isfinite(out[[0, 2, 4], COL[n]]).all()


@pytest.mark.parametrize('bad', [np.zeros(10), np.full(10, 19), np.full(10, np.nan), np.full(10, 1.5), np.ones(9), np.ones((10, 2))])
def test_traits_refuse_levels_that_are_not_ten_whole_levels_from_1_to_18(bad):
    with pytest.raises(ValueError, match='levels'):
        traits.champion_traits(rec_with(), bad)


def test_the_trait_vector_holds_no_ids():
    """A row has exactly the 19 named traits: no column for the champion key, and the only mark of a champion as such is the 0/1
    flag champ_unknown. The champion key is looked up in the table of the patch and never enters a row."""
    out = traits.champion_traits(rec_with(), np.full(10, 5, dtype=np.int16))
    assert out.shape == (10, len(traits.TRAIT_NAMES)) == (10, 19)
    assert not [n for n in traits.TRAIT_NAMES if 'id' in n.lower().split('_') or n.lower().endswith('id')]
    assert set(out[:, COL['champ_unknown']].tolist()) <= {0.0, 1.0}


@pytest.mark.parametrize('patch', C.PATCHES[:4])
def test_traits_equal_the_data_dragon_stats_of_every_champion(patch):
    """Independent of the LEGACY champion vector: the stats block of the pinned Data Dragon table (hp, hpperlevel, ...) gives the
    expected traits of every champion at levels 1, 9 and 18, so a swapped column or a wrong growth field shows up here."""
    path = C.LEGACY_ROOT / 'config' / 'game_rules' / 'datadragon_v2' / f'champions_{patch}.json'
    table = {int(k): v for k, v in json.loads(path.read_text(encoding='utf-8'))['champions'].items() if int(k) < 10_000}
    keys = sorted(table)
    assert len(keys) > 150
    keys += keys[:(-len(keys)) % 10]                                         # pad to a multiple of ten with repeated champions
    for level in (1, 9, 18):
        f = (level - 1) * (0.7025 + 0.0175 * (level - 1))
        for start in range(0, len(keys), 10):
            chunk = keys[start:start + 10]
            out = traits.champion_traits(rec_with(chunk, patch=patch), np.full(10, level, dtype=np.int16))
            for row, key in zip(out, chunk):
                s, tags = table[key]['stats'], table[key]['tags']
                partype = (table[key].get('partype') or '').strip()
                reach = table[key].get('attackrange', s['attackrange'])       # the table's own field first, as the legacy vector reads it
                want = {f'tag_{t}': float(t in tags) for t in ('Assassin', 'Fighter', 'Mage', 'Marksman', 'Support', 'Tank')}
                want.update(is_ranged=float(reach > 300), attackrange=reach, partype_Mana=float(partype == 'Mana'),
                            partype_Energy=float(partype == 'Energy'), partype_none=float(partype in ('None', '')),
                            partype_other=float(partype not in ('Mana', 'Energy', 'None', '')),
                            lvl_healthMax=s['hp'] + s['hpperlevel'] * f, lvl_armor=s['armor'] + s['armorperlevel'] * f,
                            lvl_magicResist=s['spellblock'] + s['spellblockperlevel'] * f,
                            lvl_attackDamage=s['attackdamage'] + s['attackdamageperlevel'] * f,
                            lvl_attackSpeed=s['attackspeed'] * (1 + s['attackspeedperlevel'] * f / 100),
                            lvl_movementSpeed=s['movespeed'], champ_unknown=0.0)
                assert set(want) == set(traits.TRAIT_NAMES)
                assert np.allclose(row, [want[n] for n in traits.TRAIT_NAMES], rtol=1e-6, atol=1e-6), (patch, level, key)


# ---- the level clock -----------------------------------------------------------------------------------------------------------------
def test_level_at_is_the_highest_level_up_at_or_before_the_reference_time():
    """Player 1: level 2 at 100 s, 3 at 150 s; player 2: level 2 at 100 s; player 4: level 3 at 50 s, then a lower level 2 at 60 s
    (an anomaly the rule 'highest level up to ref' ignores). Nobody else levels: they stay at level 1."""
    events = [level_up(100_000, 1, 2), level_up(100_000, 2, 2), level_up(150_000, 1, 3), level_up(50_000, 4, 3), level_up(60_000, 4, 2)]
    rec = synth.make_record(events=events)
    want = {0: [1] * 10, 49_999: [1] * 10, 50_000: [1, 1, 1, 3, 1, 1, 1, 1, 1, 1], 99_999: [1, 1, 1, 3, 1, 1, 1, 1, 1, 1],
            100_000: [2, 2, 1, 3, 1, 1, 1, 1, 1, 1],            # a level up at exactly the reference time counts
            149_999: [2, 2, 1, 3, 1, 1, 1, 1, 1, 1], 150_000: [3, 2, 1, 3, 1, 1, 1, 1, 1, 1], 10 ** 9: [3, 2, 1, 3, 1, 1, 1, 1, 1, 1]}
    index = traits.LevelIndex(rec)
    for t, levels in want.items():
        got = traits.level_at(rec, t)
        assert got.dtype == np.int16 and got.shape == (10,) and got.tolist() == levels, t
        assert np.array_equal(index.at(t), got)                      # the index is what level_at wraps


@pytest.mark.parametrize('field, value, reason', [
    ('level', 19, 'level 19'), ('level', 0, 'level 0'), ('level', 'x', 'type str'), ('level', 2.0, 'type float'),
    ('level', True, 'type bool'), ('level', None, 'type NoneType'), ('participantId', 11, 'participantId 11'),
    ('participantId', 0, 'participantId 0'), ('participantId', '3', 'type str')])
def test_level_up_with_a_bad_field_raises_format_error_without_the_key(field, value, reason):
    bad = level_up(100_000, 1, 2)
    bad[field] = value
    rec = synth.make_record(key='SYN_level_secret', events=[bad])
    with pytest.raises(MatchFormatError) as err:
        traits.level_at(rec, 200_000)
    assert reason in str(err.value) and 'SYN_level_secret' not in str(err.value) and '15.18' in str(err.value)
    for missing in ('level', 'participantId'):
        gone = level_up(100_000, 1, 2)
        del gone[missing]
        with pytest.raises(MatchFormatError, match=f'no {missing}'):
            traits.level_at(synth.make_record(events=[gone]), 200_000)


def test_level_at_refuses_a_reference_time_that_is_not_a_whole_number_of_ms():
    rec = synth.make_record()
    for bad in (1.5, 100_000.0, '100000', True, None):
        with pytest.raises(TypeError, match='ref_ms'):
            traits.level_at(rec, bad)
    assert traits.level_at(rec, np.int64(5)).tolist() == [1] * 10


# ---- the item vectors, synthetic -----------------------------------------------------------------------------------------------------
def item_col(name):
    return traits.item_vector_names().index(name)


def test_item_vectors_count_the_purchases_up_to_the_reference_time():
    """Long Sword (attack damage 10, 350 gold) for player 1 at 60 s, Cloth Armor (armor 15, 300 gold) for player 1 at 90 s and Boots
    (move speed 25, 300 gold, boots tier 1) for player 7 at 70 s: a purchase counts from its own ms on, and only for its buyer."""
    rec = synth.make_record(events=[buy(60_000, 1, 1036), buy(90_000, 1, 1029), buy(70_000, 7, 1001)])
    index = traits.ItemIndex(rec)
    ad, armor, gold, ms, boots = (item_col(n) for n in ('item_attackDamage', 'item_armor', 'item_gold_owned', 'item_movementSpeed',
                                                        'boots_tier'))
    before, at60, at70, at90 = (index.vectors(t) for t in (59_999, 60_000, 70_000, 90_000))
    for v in (before, at60, at70, at90):
        assert v.dtype == np.float32 and v.shape == (10, 22) and np.isfinite(v).all()
    assert not before[:, [ad, armor, gold, ms, boots]].any()
    assert at60[0, [ad, armor, gold]].tolist() == [10, 0, 350] and at60[6, [ms, boots, gold]].tolist() == [0, 0, 0]
    assert at70[6, [ms, boots, gold]].tolist() == [25, 1, 300] and at70[0, [ad, gold]].tolist() == [10, 350]
    assert at90[0, [ad, armor, gold]].tolist() == [10, 15, 650] and at90[6, [ms, boots, gold]].tolist() == [25, 1, 300]
    assert not np.delete(at90[:, [ad, armor, gold, ms, boots]], [0, 6], axis=0).any()        # nobody else owns anything of the four items


@pytest.mark.parametrize('patch', C.PATCHES[:4])
def test_item_replay_gets_the_runes_and_the_patch_of_the_record(patch):
    """The legacy replay needs the runes of the record and its patch (the 15.x rules): without them the grants below do not happen.
    Player 4 runs Biscuit Delivery (a 50-gold biscuit at 2:00, 4:00 and 6:00) and player 3 Magical Footwear (boots, 25 move speed and
    300 gold, at 12:00 when nobody has a takedown); the numbers are the rules of the legacy rule files, worked by hand."""
    rec = synth.make_record(patch=patch, T_ms=1_500_000)
    runes = {pid: dict(r) for pid, r in rec.runes_by_pid.items()}
    runes['4']['primary_rune_1'], runes['3']['primary_rune_1'] = 8345, 8304
    rec = dataclasses.replace(rec, runes_by_pid=runes)
    index = traits.ItemIndex(rec)
    gold, ms, boots = item_col('item_gold_owned'), item_col('item_movementSpeed'), item_col('boots_tier')
    at = {t: index.vectors(t) for t in (119_999, 120_000, 400_000, 719_999, 720_000, 900_000)}
    assert [at[t][3, gold] for t in (119_999, 120_000, 400_000, 900_000)] == [0, 50, 150, 150]          # three biscuits, none after 6:00
    assert [at[t][2, [ms, boots, gold]].tolist() for t in (719_999, 720_000, 900_000)] == [[0, 0, 0], [25, 1, 300], [25, 1, 300]]
    others = np.delete(at[900_000], [2, 3], axis=0)
    assert not others[:, [ms, boots, gold]].any()                       # nobody else is granted anything


def test_item_index_gives_the_replay_the_arguments_of_the_record(monkeypatch):
    """No golden data needed: a spy on LEGACY's ``ItemStateIndex`` (the name ``ItemIndex`` calls) records how it is built for a synthetic
    match. The brief's ``rec.champions`` is the field ``rec.champ`` of the real record: player i + 1 has ``champ[i]``. The keyword
    arguments are pinned exactly, so a new one (summoner spells, say) has to be added here on purpose."""
    item_state = legacy.load().item_state
    real, calls = item_state.ItemStateIndex, []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(item_state, 'ItemStateIndex', spy)
    champs = [key for key, _ in synth.CHAMPION_POOL[:10]]                # ten different keys: a slot shifted by one would show
    rec = rec_with(champs, patch='15.20')
    traits.ItemIndex(rec)
    assert len(calls) == 1
    args, kw = calls[0]
    table = item_state.load_item_table_v2(rec.patch)
    assert len(args) == 2 and list(args[0]) == list(rec.events) and args[1] is table
    assert set(kw) == {'exact', 'effect_flags', 'champion_by_pid', 'runes_by_pid', 'patch'}
    assert kw['exact'] is True
    assert kw['champion_by_pid'] == {str(i + 1): int(c) for i, c in enumerate(rec.champ)}
    assert list(kw['champion_by_pid'].values()) == champs and all(isinstance(c, int) for c in kw['champion_by_pid'].values())
    assert kw['runes_by_pid'] is rec.runes_by_pid
    assert kw['patch'] == rec.patch == '15.20'
    assert kw['effect_flags'] == item_state.load_effect_flags(table, patch=rec.patch)
    assert kw['effect_flags'] == item_state.load_effect_flags(patch=rec.patch)       # the table check changes no value


def test_the_effect_flags_are_checked_against_the_item_table(monkeypatch):
    """LEGACY's own default loads the flags together with the item table, so a flagged id that is not in the table stops the build with a
    KeyError. ``ItemIndex`` keeps that check: here the table of the patch lacks one id of the stasis flag."""
    item_state = legacy.load().item_state
    flagged = min(item_state.load_effect_flags(patch='15.18')['stasis'])
    short = {k: v for k, v in item_state.load_item_table_v2('15.18').items() if k != flagged}
    traits._effect_flags.cache_clear()                                    # the flags of a patch are kept once loaded
    try:
        with monkeypatch.context() as patched:
            patched.setattr(item_state, 'load_item_table_v2', lambda patch, *args, **kwargs: short)
            with pytest.raises(KeyError, match='absent from the item table'):
                traits.ItemIndex(rec_with(patch='15.18'))
    finally:
        traits._effect_flags.cache_clear()


def test_item_vectors_refuse_a_reference_time_that_is_not_a_whole_number_of_ms():
    index = traits.ItemIndex(synth.make_record())
    for bad in (1.5, 100_000.0, '100000', True, None):
        with pytest.raises(TypeError, match='ref_ms'):
            index.vectors(bad)
    assert index.vectors(np.int64(0)).shape == (10, 22)


# ---- the golden matches (C_EPT_S1_RUN) -------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def cache_pack(key, patch):
    """events.json and meta.json of the match in the cache bundle, read with json (not with the legacy cache_io); read only.

    ``read_json`` (conftest) keeps the file name, which is the match key, out of the error of a missing file."""
    base = C.CACHE_ROOT / patch / 'cache' / C.CACHE_SETS[patch]
    events = read_json(base / f'{key}.events.json', 'cache events.json')
    meta = read_json(base / f'{key}.meta.json', 'cache meta.json')
    static = meta['static_meta']
    return events, meta, ast.literal_eval(static) if isinstance(static, str) else static


def legacy_on_cache_pack(rec, t):
    """(10, 22) float64: LEGACY ``item_vectors_at`` called directly on the cache pack (events.json, meta.json) of the same match, with
    the arguments taken from the cache meta and not from the raw record (``rec`` only names the match). Row i is participant i + 1."""
    events, meta, static = cache_pack(rec.key, rec.patch)
    item_state = legacy.load().item_state
    patch = str(meta['patch'])
    table = item_state.load_item_table_v2(patch)
    got = item_state.item_vectors_at(events, table, t, exact=True, effect_flags=item_state.load_effect_flags(table, patch=patch),
                                     champion_by_pid=static['champion_by_pid'], runes_by_pid=static['runes_by_pid'], patch=patch)
    return np.array([[got[pid][n] for n in item_state.ITEM_VECTOR_NAMES] for pid in range(1, 11)], dtype=np.float64)


def columns_that_differ(got, want):
    return [n for j, n in enumerate(traits.item_vector_names()) if not np.array_equal(got[:, j], want[:, j], equal_nan=True)]


def test_item_vectors_match_legacy_on_cache(golden_records):
    for rec in golden_records:
        assert np.allclose(traits.ItemIndex(rec).vectors(600_000), legacy_on_cache_pack(rec, 600_000), equal_nan=True)


def test_item_vectors_equal_legacy_on_cache_at_several_times_bit_for_bit(golden_records):
    """The brief's check at 600 s, at five more times, and stricter: the float32 vectors equal the float32 cast of the legacy values
    (the wrapper changes nothing but the dtype). The match numbers (positions in the golden list) name a failure, never a key."""
    for i, rec in enumerate(golden_records):
        index = traits.ItemIndex(rec)
        for t in (0, 120_000, 600_000, 900_000, 1_200_000, rec.T_ms):
            got, want = index.vectors(t), legacy_on_cache_pack(rec, t)
            assert got.dtype == np.float32 and got.shape == (10, 22), i
            assert columns_that_differ(got, want.astype(np.float32)) == [], (i, t)
            assert np.allclose(got, want, equal_nan=True), (i, t)


def test_item_vectors_at_a_time_do_not_depend_on_later_events_or_call_order(golden_records):
    """A3: the vectors at ref are the same when every event after ref is cut off before the replay, and an index answers the same in
    any order of questions (nothing is remembered between calls)."""
    refs = (120_000, 300_000, 600_000, 900_000)
    for i, rec in enumerate(golden_records):
        index = traits.ItemIndex(rec)
        ascending = {t: index.vectors(t) for t in refs}
        for t in reversed(refs):
            cut = dataclasses.replace(rec, events=tuple(e for e in rec.events if e['timestamp'] <= t))
            assert np.array_equal(traits.ItemIndex(cut).vectors(t), ascending[t]), (i, t)
            assert np.array_equal(index.vectors(t), ascending[t]), (i, t)


def test_golden_champions_are_known_and_their_traits_finite(golden_records):
    for i, rec in enumerate(golden_records):
        out = traits.champion_traits(rec, rec.frame_num[2, :, 2])
        assert out[:, COL['champ_unknown']].tolist() == [0.0] * 10 and np.isfinite(out).all(), i


def test_level_at_matches_frame_level(golden_records):        # 20 matches, every regular frame
    """The level clock against the frame level at every regular frame of the 20 golden matches. Measured on 2026-10-20 at commit 39195c7
    with c_ept/checks/measure_levels.py: 582 regular frames, 5,820 player-frames, no difference, so the equality is exact. Four LEVEL_UP
    events fall on exactly a frame's ms and the frame already shows them: that is what makes the inclusive ``<= ref`` rule part of the
    test. Over all 28,701 S1 matches 150 of 7,573,150 player-frames differ, 0.002% (same date and commit, but measured by the scratch
    predecessor of c_ept/checks/sweep_s1.py, which was moved there and not run again from the package): each is a LEVEL_UP at exactly
    the frame ms that the frame snapshot does not show yet, so the frame lags the clock by one level; none has another cause.
    Frame-clock rows therefore take the frame value, and views on the event clock take ``level_at``.

    If a pinned count changes: measure again, list the differing cells and their cause (c_ept/checks/explain_mismatch.py). A frame that
    lags by one level at a LEVEL_UP on its ms is exempted cell by cell and the new count is pinned here; for any other cause stop and
    report it [S2-8]. The counts are pinned so that the test cannot pass on fewer matches or frames."""
    frames = cells = on_frame_ms = 0
    for i, rec in enumerate(golden_records):
        frame_ms = set(rec.frame_ts.tolist())
        on_frame_ms += sum(1 for e in rec.events if e['type'] == 'LEVEL_UP' and e['timestamp'] in frame_ms)
        for k, ts in enumerate(rec.frame_ts):
            frames, cells = frames + 1, cells + 10
            assert np.array_equal(traits.level_at(rec, ts), rec.frame_num[k, :, 2].astype(int)), (i, k)
    assert (frames, cells, on_frame_ms) == (582, 5_820, 4)


# ---- the measurement scripts of c_ept/checks ----------------------------------------------------------------------------------------
SCRIPTS = (measure_levels, measure_levels_wide, explain_mismatch, sweep_s1)
SCRIPT_NAMES = [m.__name__.rsplit('.', 1)[-1] for m in SCRIPTS]
REPO = Path(C.__file__).parents[1]


def level_clock_record():
    """A synthetic match with 7 regular frames (0, 60,300, ..., 360,300 ms; levels 1, 1, 2, 3, 3, 4, 5 for everybody) whose LEVEL_UPs
    (level 2 at 100 s, 3 at 170 s, 4 at 290 s, 5 at 340 s, for all ten players) agree with the frames, except:
    player 1 gets level 3 at exactly the ms of frame 3 and that frame still shows level 2 (the frame lags by one level);
    player 2's frame 5 shows level 6 where the clock says 4 (another cause);
    player 3 gets level 4 at exactly the ms of frame 5 and that frame shows it (a level up on a frame's ms that agrees)."""
    when = [(100_000, 2), (170_000, 3), (290_000, 4), (340_000, 5)]
    events = []
    for pid in range(1, 11):
        for t, level in when:
            if (pid, level) == (1, 3):
                t = 180_300
            elif (pid, level) == (3, 4):
                t = 300_300
            events.append(level_up(t, pid, level))
    rec = synth.make_record(T_ms=400_000, events=events)
    assert rec.frame_ts.tolist() == [0, 60_300, 120_300, 180_300, 240_300, 300_300, 360_300]
    assert rec.frame_num[:, 0, 2].tolist() == [1, 1, 2, 3, 3, 4, 5]
    num = rec.frame_num.copy()                                       # the arrays of a record are read-only: edit a copy
    num[3, 0, 2], num[5, 1, 2] = 2, 6
    return dataclasses.replace(rec, frame_num=num)


def test_compare_levels_counts_the_cells_and_names_the_cause_of_each_difference():
    cells = []
    got = common.compare_levels(level_clock_record(), cells)
    assert dict(got) == {'matches': 1, 'frames': 7, 'cells': 70, 'level_ups': 40, 'level_ups_on_frame_ms': 2, 'mismatch_cells': 2,
                         'mismatch_lag_by_one_at_frame_ms': 1, 'mismatch_other': 1}
    assert [(c['frame'], c['ms'], c['pid'], c['level_at'], c['frame_level'], c['cause']) for c in cells] == [
        (3, 180_300, 1, 3, 2, 'lag_by_one_at_frame_ms'), (5, 300_300, 2, 4, 6, 'other')]


def test_compare_levels_of_a_match_that_agrees_has_every_count_and_all_zero_differences():
    events = [level_up(t, pid, lv) for pid in range(1, 11) for t, lv in [(100_000, 2), (170_000, 3), (290_000, 4), (340_000, 5)]]
    got = common.compare_levels(synth.make_record(T_ms=400_000, events=events))
    assert set(got) == set(common.LEVEL_KEYS) and got['mismatch_cells'] == got['mismatch_other'] == 0 and got['cells'] == 70


def test_the_json_of_a_check_is_sorted_utf8_with_unix_line_ends(tmp_path):
    common.write_json(tmp_path / 'c.json', {'b': 1, 'a': {'z': 2}})
    assert (tmp_path / 'c.json').read_bytes() == b'{\n  "a": {\n    "z": 2\n  },\n  "b": 1\n}\n'


def write_store(folder, rows):
    """An S1-store-like folder ``folder/s1/matches`` with the arrays key, patch and role for ``rows`` [(key, patch, macro), ...]."""
    d = folder / 's1' / 'matches'
    d.mkdir(parents=True)
    np.save(d / 'key.npy', np.array([r[0].encode('ascii') for r in rows], dtype='S24'))
    np.save(d / 'patch.npy', np.array([C.PATCHES.index(r[1]) for r in rows], dtype=np.int8))
    np.save(d / 'role.npy', np.array([C.ROLES.index(r[2]) for r in rows], dtype=np.int8))


def test_the_match_lists_come_from_the_arrays_of_the_store_and_of_the_golden_lock(tmp_path):
    rows = [('SYN_a', '15.18', 'FIT'), ('SYN_b', '15.18', 'DEV'), ('SYN_c', '15.19', 'FIT'), ('SYN_d', '15.18', 'FIT'),
            ('SYN_e', '15.19', 'CAL')]
    assert common.STORE_DIR == run_s1.STORE_DIR                                 # the folder name the lists are read from
    write_store(tmp_path, rows)
    write_store(tmp_path / 'golden', rows[:2])
    (tmp_path / 's1' / 'STORE.json').write_text('{}', encoding='utf-8')
    assert common.s1_matches(tmp_path) == rows
    assert common.s1_matches(tmp_path / 's1') == rows                      # the store folder names the run as well
    assert common.golden_matches(tmp_path) == rows[:2]
    other = tmp_path / 'elsewhere'
    write_store(other, rows[2:3])
    assert common.golden_matches(tmp_path, other) == rows[2:3]
    with pytest.raises(FileNotFoundError, match='no S1 store'):
        common.s1_matches(tmp_path / 'nothing_here')


def test_the_sample_takes_the_first_fit_and_the_first_of_every_other_role_per_patch(tmp_path):
    rows = [(f'SYN_{p}_{m}_{i}', p, m) for p in ('15.18', '15.19') for m in ('FIT', 'DEV', 'CAL') for i in range(4)]
    write_store(tmp_path, rows)
    got = common.sample_matches(tmp_path, fit=3, other=1)
    want = [r for r in rows if (r[2] == 'FIT' and r[0].endswith(('_0', '_1', '_2'))) or (r[2] != 'FIT' and r[0].endswith('_0'))]
    assert sorted(got) == sorted(want) and len(got) == 2 * (3 + 1 + 1)
    assert [r[1] for r in got] == sorted(r[1] for r in got)                              # in the order of the patches


@pytest.mark.parametrize('name', ['common', *SCRIPT_NAMES])
def test_a_check_module_does_nothing_when_it_is_imported(name):
    code = f'import c_ept.checks.{name} as m; assert callable(getattr(m, "main", print))'
    r = subprocess.run([sys.executable, '-B', '-c', code], cwd=REPO, capture_output=True, text=True, encoding='utf-8')
    assert (r.returncode, r.stdout, r.stderr) == (0, '', '')


@pytest.mark.parametrize('name', SCRIPT_NAMES)
def test_a_check_script_explains_itself_when_run_as_a_module(name):
    r = subprocess.run([sys.executable, '-B', '-m', f'c_ept.checks.{name}', '--help'], cwd=REPO, capture_output=True, text=True,
                       encoding='utf-8')
    text = ' '.join(r.stdout.split())                                  # argparse wraps the lines
    assert r.returncode == 0 and 'S1 run folder' in text and '--out' in text and 'counts' in text, r.stderr


@pytest.mark.parametrize('script', SCRIPTS, ids=SCRIPT_NAMES)
def test_every_script_stops_on_a_running_strata_server_before_it_reads_anything(script, tmp_path, fake_processes, capsys):
    """E6 comes first: with the server running the refusal is the E6 message, not the complaint about the run folder that does not
    exist; without the server the same call gets to the missing store."""
    fake_processes([synth.STRATA_SERVER])
    assert script.main([str(tmp_path / 'no_run_here')]) == 2
    err = capsys.readouterr().err
    assert 'E6' in err and 'does not stop' in err and 'no S1 store' not in err
    fake_processes([])
    assert script.main([str(tmp_path / 'no_run_here')]) == 2
    err = capsys.readouterr().err
    assert 'no S1 store' in err and 'E6' not in err
    assert list(tmp_path.iterdir()) == []                                                  # nothing was written


@pytest.mark.parametrize('script', SCRIPTS, ids=SCRIPT_NAMES)
def test_no_script_writes_into_the_s1_run_folder(script, tmp_path, capsys):
    """``--out`` inside the run folder (or its store) is refused as an argument error, before any work and before anything is written."""
    write_store(tmp_path / 'run', [('SYN_a', '15.18', 'FIT')])
    (tmp_path / 'run' / 's1' / 'STORE.json').write_text('{}', encoding='utf-8')
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*'))
    for run, out in ((tmp_path / 'run', tmp_path / 'run' / 'counts.json'), (tmp_path / 'run' / 's1', tmp_path / 'run' / 'counts.json'),
                     (tmp_path / 'run', tmp_path / 'run' / 'golden' / 'new' / 'counts.json')):
        with pytest.raises(SystemExit) as err:
            script.main([str(run), '--out', str(out)])
        assert err.value.code == 2 and 'S1 run folder' in capsys.readouterr().err
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*')) == before


@pytest.mark.parametrize('script', (measure_levels, measure_levels_wide, explain_mismatch),
                         ids=['measure_levels', 'measure_levels_wide', 'explain_mismatch'])
def test_a_script_that_cannot_read_a_match_file_stops_without_showing_its_key(script, tmp_path, monkeypatch, capsys):
    """These three scripts loop over the records and have no handler: what ``read_match`` raises reaches the traceback. The reader's
    errors carry no key (``MatchReadError`` for a file that cannot be read, ``MatchFormatError``, ``MatchRejected``), so the traceback
    shows none and the scripts need no handler of their own; ``sweep_s1`` and ``sweep_state`` count their errors by class and need none
    either. The match here is named like a match key, its files exist and ``open`` is refused."""
    patch = '15.18'
    write_store(tmp_path / 'run', [(SENTINEL, patch, 'FIT')])
    write_store(tmp_path / 'run' / 'golden', [(SENTINEL, patch, 'FIT')])                  # the lock that measure_levels reads
    rec = synth.make_record(T_ms=400_000, key=SENTINEL, patch=patch)
    synth.write_raw(tmp_path / 'files', patch, SENTINEL, synth.raw_timeline(rec), synth.raw_detail(rec))
    monkeypatch.setattr(reader.read_match, '__defaults__', (tmp_path / 'files',))        # the scripts read with the default root

    def refused(path, *args, **kwargs):
        raise PermissionError(errno.EACCES, 'Permission denied', str(path))
    monkeypatch.setattr(reader, 'open', refused, raising=False)
    with pytest.raises(OSError) as err:
        script.main([str(tmp_path / 'run')])
    captured = capsys.readouterr()
    assert SENTINEL not in shown(err.value) and SENTINEL not in captured.out + captured.err
    assert type(err.value) is reader.MatchReadError


def shows_a_key(text, records):
    """Whether any match key of ``records`` is in ``text`` (a bool: a failure message must not carry the keys it looked for)."""
    return any(rec.key in text for rec in records)


def test_measure_levels_reproduces_the_pinned_counts_of_the_golden_matches(golden_records, tmp_path, capsys):
    run = os.environ['C_EPT_S1_RUN']
    golden = golden_folder(run)                    # the folder the golden_records fixture reads (C_EPT_GOLDEN_DIR or <run>/golden) ...
    same_lock = [(k, p) for k, p, _ in common.golden_matches(run, golden)] == [(r.key, r.patch) for r in golden_records]
    assert same_lock                               # (a boolean: a failure message must not print the keys)
    out = tmp_path / 'levels.json'
    assert measure_levels.main([run, '--golden-dir', str(golden), '--out', str(out)]) == 0      # ... and the script is told the same one
    printed = capsys.readouterr().out
    got = json.loads(out.read_text(encoding='utf-8'))
    pinned = {'matches': 20, 'frames': 582, 'cells': 5_820, 'level_ups_on_frame_ms': 4, 'mismatch_cells': 0}
    assert got['script'] == measure_levels.SCRIPT and {k: got['counts'][k] for k in pinned} == pinned
    assert set(got['counts']) == set(common.LEVEL_KEYS) and got['counts']['level_ups'] > 0
    assert got['run'] == common.run_folder(run).name
    assert not shows_a_key(printed, golden_records) and not shows_a_key(out.read_text(encoding='utf-8'), golden_records)


@pytest.mark.parametrize('script, extra', [(measure_levels_wide, []), (explain_mismatch, []), (sweep_s1, ['--workers', '1'])],
                         ids=['measure_levels_wide', 'explain_mismatch', 'sweep_s1'])
def test_the_other_scripts_run_on_a_few_matches_of_the_s1_store(script, extra, golden_records, tmp_path, capsys):
    run = os.environ['C_EPT_S1_RUN']
    out = tmp_path / 'counts.json'
    assert script.main([run, '--limit', '3', '--out', str(out), *extra]) == 0
    printed = capsys.readouterr().out
    got = json.loads(out.read_text(encoding='utf-8'))
    assert got['script'] == script.SCRIPT and got['counts']['matches'] == 3 and got['counts']['mismatch_cells'] == 0
    assert got.get('errors', {}) == {} and got['run'] == common.run_folder(run).name
    assert not shows_a_key(printed, golden_records) and not shows_a_key(out.read_text(encoding='utf-8'), golden_records)
