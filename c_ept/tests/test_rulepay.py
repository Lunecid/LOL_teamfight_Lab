"""Tests of ``c_ept.rulepay``: the checked rule-payout table, the base bounty by level and the assist gold pool of a kill (stage S3 Task 3).

Every table here is a small file written under pytest's ``tmp_path`` by ``write_rules`` in the layout of ``rule_payouts_v2``; the checked
table of the s3a run and every other file under D:\\ are never read. Truth comes from three places: numbers worked out by hand from decision
P3 (the working is in the comments), a second implementation of the pool written out in this file from the text of P3 (``ref_pool``:
plain Python, one kill at a time, the base bounty from its closed form instead of the list of 18 values), and the refusals the loader
owes the author (an f0 rule that is not verified, a table that disagrees with P3, a value that cannot be a payout).

The three tests of the task brief come first, word for word. The others pin P3 as numbers, the loader (what it accepts and what it
refuses), a table for a patch other than 15.18-15.22 (16.13: other breakpoints of the early-game multiplier), the vectorised form against
the scalar one and against the reference, the refusal of bad inputs, and the module's imports.
"""
import inspect
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from c_ept.valuation import rulepay
from c_ept.tests.conftest import module_level_imports

REPO = Path(__file__).resolve().parents[2]
F0_RULES = ('champion_kill_base_bounty', 'assist_bounty', 'assist_bounty_early_game_multiplier', 'first_blood_bonus')
P3_PATCHES = ('15.18', '15.19', '15.20', '15.21', '15.22')
P3_BASE = tuple(300 + 10 * max(0, level - 6) for level in range(1, 19))          # 300 for levels 1-6, +10 per level from 7, 420 at 18
SEG_16_13 = dict(lo='16.13', hi='16.13', flat=55, full=175, first_blood=(100, 0))      # the breakpoints that the game file holds for 16.13
SEG_OTHER = dict(lo='16.5', hi='16.7', base=tuple(range(200, 218)), pool_share=0.4, cap_share=0.3, flat=30, full=90, m0=0.25, m1=0.75)


# ---- toy tables in the layout of rule_payouts_v2 -----------------------------------------------------------------------------------------
def entry(value, status='verified', lo='15.18', hi='15.22'):
    """One entry of a rule: the fields the loader reads, and the ones it has to leave alone."""
    return {'patch_range': [lo, hi], 'value': value, 'unit': 'gold', 'status': status, 'sources': ['toy_source'], 'note': 'a toy entry',
            'quotes': [{'text': 'toy', 'source': 'toy_source'}], 'game_data': {'finding': 'toy', 'status_if_game_data_counts': 'verified'}}


def f0_values(base=P3_BASE, pool_share=0.5, cap_share=0.5, flat=90, full=210, m0=0.5, m1=1.0, first_blood=(0, 0)):
    """The ``value`` of the four f0 rules (the defaults are P3), with the text fields that the real table has next to the numbers."""
    return {
        'champion_kill_base_bounty': {'level_1_to_6': base[0], 'by_level': list(base)},
        'assist_bounty': {'pool': '0.5 * kill bounty dispensed by the victim', 'formula': 'toy text', 'pool_share': pool_share,
                          'cap_share_of_base': cap_share},
        'assist_bounty_early_game_multiplier': {'flat_until_s': flat, 'full_at_s': full, 'start_multiplier': m0, 'end_multiplier': m1,
                                                'rate_per_s': 0.0041667},
        'first_blood_bonus': {'killer_bonus': first_blood[0], 'assist_pool_bonus': first_blood[1]},
    }


def write_rules(tmp_path, segments=None, *, base_status=None, assist_status=None, mult_status=None, fb_status=None, extra=None, mutate=None,
                name='RULE_PAYOUTS.json'):
    """Write a toy ``rule_payouts_v2`` table under ``tmp_path`` and return its path.

    ``segments``: a list of dicts, one per patch range, each with the keyword arguments of ``f0_values`` plus ``lo``, ``hi`` (default
    15.18 and 15.22) and ``status`` ({rule id: status} for this range); the default is one P3 range, every rule verified. The four
    ``*_status`` arguments set the status of that rule in every range. ``extra``: more rules (not f0 ones). ``mutate``: called with the
    document before it is written; it changes the document in place or returns a replacement.
    """
    override = dict(zip(F0_RULES, (base_status, assist_status, mult_status, fb_status)))
    rules = {rid: [] for rid in F0_RULES}
    for seg in ([{}] if segments is None else segments):
        seg = dict(seg)
        lo, hi, status = seg.pop('lo', '15.18'), seg.pop('hi', '15.22'), seg.pop('status', {})
        for rid, value in f0_values(**seg).items():
            rules[rid].append(entry(value, override[rid] or status.get(rid, 'verified'), lo, hi))
    rules.update(extra or {})
    doc = {'schema': 'rule_payouts_v2', 'created': '2026-10-21', 'rules': rules, 'sources': {'toy_source': 'a toy source'}}
    if mutate is not None:
        out = mutate(doc)
        doc = doc if out is None else out
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding='utf-8')
    return path


def set_value(rid, **fields):
    """A ``mutate`` for ``write_rules``: change fields of the value of the first entry of rule ``rid``."""
    def mutate(doc):
        doc['rules'][rid][0]['value'].update(fields)
    return mutate


def drop_value(rid, field):
    """A ``mutate`` for ``write_rules``: remove a field from the value of the first entry of rule ``rid``."""
    def mutate(doc):
        del doc['rules'][rid][0]['value'][field]
    return mutate


def replace_entry_field(rid, field, new=None, *, drop=False):
    """A ``mutate`` for ``write_rules``: set (or drop) a field of the first entry of rule ``rid`` itself, next to its ``value``."""
    def mutate(doc):
        if drop:
            del doc['rules'][rid][0][field]
        else:
            doc['rules'][rid][0][field] = new
    return mutate


def add_entry(rid, lo, hi, status='verified', **values):
    """A ``mutate`` for ``write_rules``: one more entry of rule ``rid`` for the patches ``lo`` .. ``hi`` (``values``: as ``f0_values``)."""
    def mutate(doc):
        doc['rules'][rid].append(entry(f0_values(**values)[rid], status, lo, hi))
    return mutate


def ref_pool(bounty, level, t_ms, n_assisters, *, share=0.5, cap=0.5, flat=90, full=210, m0=0.5, m1=1.0, base=None):
    """The assist pool of one kill, written out from the text of P3 (not from the module): plain Python, base bounty in closed form."""
    if n_assisters < 1:
        return 0.0
    base = 300 + 10 * max(0, level - 6) if base is None else base
    ramp = min(1.0, max(0.0, (t_ms / 1000 - flat) / (full - flat)))
    return min(share * bounty, cap * base) * (m0 + (m1 - m0) * ramp)


# ---- the three tests of the task brief ---------------------------------------------------------------------------------------------------
def test_base_bounty_by_level():
    assert [rulepay.base_bounty(l, '15.20') for l in (1, 6, 7, 17, 18)] == [300, 300, 310, 410, 420]


def test_assist_pool_cap_and_early_multiplier():
    assert rulepay.assist_pool(300, 6, 300_000, '15.20', 2) == pytest.approx(150.0)
    assert rulepay.assist_pool(700, 6, 300_000, '15.20', 1) == pytest.approx(150.0)    # cap 0.5 x 300
    assert rulepay.assist_pool(300, 6, 90_000, '15.20', 1) == pytest.approx(75.0)      # 1:30, multiplier 0.5
    assert rulepay.assist_pool(300, 6, 150_000, '15.20', 1) == pytest.approx(112.5)    # 2:30, multiplier 0.75
    assert rulepay.assist_pool(300, 6, 300_000, '15.20', 0) == 0.0


def test_rule_table_refuses_unverified_f0_entries(tmp_path):
    with pytest.raises(ValueError):
        rulepay.RuleTable.load(write_rules(tmp_path, assist_status='documented'))


# ---- decision P3 as numbers --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('patch', P3_PATCHES)
def test_p3_rules_hold_the_approved_values(patch):
    pr = rulepay.P3_RULES.for_patch(patch)
    assert pr.base_by_level == P3_BASE
    assert [pr.base_by_level[i] for i in (0, 5, 6, 16, 17)] == [300, 300, 310, 410, 420]
    assert (pr.pool_share, pr.cap_share_of_base) == (0.5, 0.5)
    assert (pr.flat_until_s, pr.full_at_s, pr.start_multiplier, pr.end_multiplier) == (90.0, 210.0, 0.5, 1.0)
    assert (pr.first_blood_killer_bonus, pr.first_blood_assist_pool_bonus) == (0, 0)       # 15.18-15.22 have no First Blood gold


HAND = [    # (bounty, victim level, t_ms, assisters, pool): P3 on 15.x, the working in the comment
    (300, 6, 120_000, 1, 93.75),         # min(150, 150) x (0.5 + 0.5 x 30/120 = 0.625)
    (100, 6, 210_000, 1, 50.0),          # min(50, 150) x 1.0: the bounty is the smaller side
    (400, 7, 300_000, 2, 155.0),         # min(200, 155) x 1.0: the cap of level 7 is 0.5 x 310
    (500, 18, 180_000, 4, 183.75),       # min(250, 210) x (0.5 + 0.5 x 90/120 = 0.875)
    (0, 6, 300_000, 1, 0.0),             # a bounty of 0 gives a pool of 0
    (300, 1, 0, 1, 75.0),                # t = 0: min(150, 150) x 0.5
    (250, 12, 100_000, 1, 1625 / 24),    # min(125, 180) x (0.5 + 0.5 x 10/120 = 13/24)
]


@pytest.mark.parametrize('patch', P3_PATCHES)
@pytest.mark.parametrize('bounty, level, t_ms, n, pool', HAND)
def test_assist_pool_hand_values(patch, bounty, level, t_ms, n, pool):
    assert rulepay.assist_pool(bounty, level, t_ms, patch, n) == pytest.approx(pool, rel=1e-12, abs=1e-12)
    assert rulepay.assist_pools([bounty], [level], [t_ms], patch, [n])[0] == pytest.approx(pool, rel=1e-12, abs=1e-12)


def test_the_number_of_assisters_decides_only_whether_a_pool_exists():
    """The pool is the total over all assisters (each gets an equal share of it): 1 and 5 assisters give the same pool, 0 gives none."""
    pools = [rulepay.assist_pool(300, 6, 300_000, '15.19', n) for n in (0, 1, 2, 5, 100)]
    assert pools == [0.0, 150.0, 150.0, 150.0, 150.0]


def test_the_pool_is_not_rounded_and_is_a_python_float():
    value = rulepay.assist_pool(250, 12, 100_000, '15.19', 1)
    assert type(value) is float and value == pytest.approx(1625 / 24, rel=1e-12) and value != round(value)
    assert type(rulepay.base_bounty(7, '15.19')) is int


def test_the_pool_grows_with_time_and_stops_at_the_cap():
    """Between 90 s and 210 s the pool of a cap-bound kill rises linearly from 75 to 150; outside it is flat."""
    times = np.arange(0, 400_000, 7_500)
    pool = rulepay.assist_pools(np.full(times.shape, 700), 6, times, '15.21', 1)
    assert np.all(np.diff(pool) >= 0)
    assert pool[0] == pytest.approx(75.0) and pool[-1] == pytest.approx(150.0)
    ramp = (times > 90_000) & (times < 210_000)
    assert np.allclose(pool[ramp], 75.0 + 75.0 * (times[ramp] / 1000 - 90) / 120)
    assert np.all(pool[times <= 90_000] == 75.0) and np.all(pool[times >= 210_000] == 150.0)


def test_the_interface_is_pinned_for_the_callers():
    """Task 4 calls these positionally: the order of the arguments is part of the contract."""
    assert list(inspect.signature(rulepay.base_bounty).parameters) == ['level', 'patch', 'rules']
    assert list(inspect.signature(rulepay.assist_pool).parameters) == ['bounty', 'victim_level', 't_ms', 'patch', 'n_assisters', 'rules']
    assert list(inspect.signature(rulepay.assist_pools).parameters) == ['bounty', 'victim_level', 't_ms', 'patch', 'n_assisters', 'rules']
    assert list(inspect.signature(rulepay.RuleTable.load).parameters) == ['path', 'patches']
    assert rulepay.F0_RULES == F0_RULES and rulepay.P3_PATCHES == P3_PATCHES


# ---- the built-in rules serve 15.18-15.22 and nothing else -------------------------------------------------------------------------------
@pytest.mark.parametrize('patch', ['15.17', '15.23', '14.24', '16.13', '16.1', '15.9', '25.18'])
def test_the_default_rules_refuse_every_other_patch(patch):
    with pytest.raises(ValueError):
        rulepay.base_bounty(6, patch)
    with pytest.raises(ValueError):
        rulepay.assist_pool(300, 6, 150_000, patch, 1)
    with pytest.raises(ValueError):
        rulepay.assist_pools([300], [6], [150_000], patch, [1])
    with pytest.raises(ValueError):
        rulepay.assist_pools([300, 300], [6, 6], [150_000, 150_000], ['15.20', patch], [1, 1])


@pytest.mark.parametrize('patch', ['', '15', '15.', '.20', '15.20.1', '15.020', ' 15.20', '15.2x', '25.S1.1', 'KR_7123456789', 15.2, None,
                                   b'15.20'])
def test_a_malformed_patch_is_refused_and_never_echoed(patch):
    for call in (lambda: rulepay.base_bounty(6, patch), lambda: rulepay.assist_pool(300, 6, 150_000, patch, 1),
                 lambda: rulepay.assist_pools([300], [6], [150_000], patch, [1])):
        with pytest.raises(ValueError) as err:
            call()
        message = str(err.value).replace('"15.20"', '')      # the message shows the form of a patch with this one fixed example
        if isinstance(patch, str) and patch:                 # (the empty string is a part of every message)
            assert patch not in message


# ---- the loader: what it accepts ---------------------------------------------------------------------------------------------------------
def test_a_verified_table_loads_and_serves_its_patches(tmp_path):
    table = rulepay.RuleTable.load(write_rules(tmp_path))
    for patch in P3_PATCHES:
        assert table.for_patch(patch) == rulepay.P3_RULES.for_patch(patch)
    for patch in ('15.17', '15.23', '16.13'):
        with pytest.raises(ValueError):
            table.for_patch(patch)


def test_a_loaded_p3_table_prices_like_the_built_in_rules(tmp_path):
    table = rulepay.RuleTable.load(write_rules(tmp_path))
    assert rulepay.base_bounty(12, '15.21', rules=table) == rulepay.base_bounty(12, '15.21') == 360
    for bounty, level, t_ms, n, pool in HAND:
        assert rulepay.assist_pool(bounty, level, t_ms, '15.19', n, rules=table) == rulepay.assist_pool(bounty, level, t_ms, '15.19', n)
    with pytest.raises(ValueError):
        rulepay.assist_pool(300, 6, 150_000, '16.13', 1, rules=table)


def test_a_loaded_table_replaces_the_built_in_rules(tmp_path):
    table = rulepay.RuleTable.load(write_rules(tmp_path, [SEG_16_13]))
    with pytest.raises(ValueError):
        rulepay.assist_pool(300, 6, 150_000, '15.20', 1, rules=table)       # the table is used instead of the built-in rules, not on top
    assert rulepay.assist_pool(300, 6, 300_000, '16.13', 1, rules=table) == pytest.approx(150.0)


def test_entries_of_sensitivity_only_rules_may_be_documented_or_unknown(tmp_path):
    extra = {'turret_outer_gold': [entry({'amount': 125}, 'documented')], 'objective_bounty': [entry('see the note', 'unknown')],
             'first_turret_bonus': [entry(None, 'documented')], 'a_note': 'not even a list'}
    table = rulepay.RuleTable.load(write_rules(tmp_path, extra=extra))
    assert table.for_patch('15.20') == rulepay.P3_RULES.for_patch('15.20')


def test_the_loader_reads_utf8_and_accepts_a_path_or_a_string(tmp_path):
    path = write_rules(tmp_path)
    doc = json.loads(path.read_text(encoding='utf-8'))
    doc['open_questions'] = ['\ud55c\uae00 note \u2014 with non-ASCII text']              # as the real table has: Korean text and dashes
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')
    assert any(b > 127 for b in path.read_bytes())
    assert rulepay.RuleTable.load(path).for_patch('15.18') == rulepay.RuleTable.load(str(path)).for_patch('15.18')


# ---- the loader: what it refuses ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('rid', F0_RULES)
@pytest.mark.parametrize('status', ['documented', 'unknown', 'Verified', 'draft', '', None])
def test_every_f0_rule_must_be_verified(tmp_path, rid, status):
    def mutate(doc):
        doc['rules'][rid][0]['status'] = status
    with pytest.raises(ValueError, match='verified'):
        rulepay.RuleTable.load(write_rules(tmp_path, mutate=mutate))


@pytest.mark.parametrize('rid', F0_RULES)
def test_an_entry_without_a_status_is_refused(tmp_path, rid):
    def mutate(doc):
        del doc['rules'][rid][0]['status']
    with pytest.raises(ValueError, match='verified'):
        rulepay.RuleTable.load(write_rules(tmp_path, mutate=mutate))


@pytest.mark.parametrize('rid', F0_RULES)
@pytest.mark.parametrize('how', ['absent', 'empty', 'not a list'])
def test_a_missing_f0_rule_is_refused(tmp_path, rid, how):
    def mutate(doc):
        if how == 'absent':
            del doc['rules'][rid]
        else:
            doc['rules'][rid] = [] if how == 'empty' else {'0': 1}
    with pytest.raises(ValueError, match=rid):
        rulepay.RuleTable.load(write_rules(tmp_path, mutate=mutate))


def test_one_unverified_range_of_a_rule_refuses_the_whole_table(tmp_path):
    segs = [dict(), dict(SEG_16_13, status={'assist_bounty_early_game_multiplier': 'documented'})]
    with pytest.raises(ValueError, match='verified'):
        rulepay.RuleTable.load(write_rules(tmp_path, segs))


@pytest.mark.parametrize('how', ['wrong schema', 'no schema', 'no rules', 'rules not a mapping', 'document is a list',
                                 'document is a number'])
def test_a_file_that_is_not_a_rule_payouts_v2_table_is_refused(tmp_path, how):
    def mutate(doc):
        if how == 'wrong schema':
            doc['schema'] = 'rule_payouts_v1'
        elif how == 'no schema':
            del doc['schema']
        elif how == 'no rules':
            del doc['rules']
        elif how == 'rules not a mapping':
            doc['rules'] = list(doc['rules'])
        else:
            return [doc] if how == 'document is a list' else 3
    with pytest.raises(ValueError):
        rulepay.RuleTable.load(write_rules(tmp_path, mutate=mutate))


def test_a_file_that_is_not_json_is_refused(tmp_path):
    bad = tmp_path / 'broken.json'
    bad.write_text('{"schema": "rule_payouts_v2", "rules": ', encoding='utf-8')
    with pytest.raises(ValueError):
        rulepay.RuleTable.load(bad)
    bad.write_bytes(b'\xff\xfe\x00\x01 not utf-8')
    with pytest.raises(ValueError):
        rulepay.RuleTable.load(bad)


@pytest.mark.parametrize('patch_range', [['15.18'], ['15.18', '15.22', '15.23'], '15.18-15.22', ['15.22', '15.18'], ['15.18', 'x'],
                                         [15.18, 15.22], ['15.18', None], None])
def test_a_malformed_or_reversed_patch_range_is_refused(tmp_path, patch_range):
    def mutate(doc):
        doc['rules']['assist_bounty'][0]['patch_range'] = patch_range
    with pytest.raises(ValueError):
        rulepay.RuleTable.load(write_rules(tmp_path, mutate=mutate))


def test_two_entries_of_one_rule_that_cover_the_same_patch_are_refused(tmp_path):
    def mutate(doc):
        doc['rules']['assist_bounty'].append(entry({'pool_share': 0.5, 'cap_share_of_base': 0.5}, lo='15.22', hi='15.23'))
    with pytest.raises(ValueError, match='assist_bounty'):
        rulepay.RuleTable.load(write_rules(tmp_path, mutate=mutate))


def test_patch_ranges_compare_patch_numbers_not_text(tmp_path):
    """'15.9' < '15.18' as numbers (but not as text), so the range 15.9 .. 15.22 holds 15.10 and 15.18 and not 15.8 or 15.23."""
    table = rulepay.RuleTable.load(write_rules(tmp_path, [dict(lo='15.9', hi='15.22')]))
    for patch in ('15.9', '15.10', '15.18', '15.22'):
        assert table.for_patch(patch) == rulepay.P3_RULES.for_patch('15.20')
    for patch in ('15.8', '15.23', '16.1'):
        with pytest.raises(ValueError):
            table.for_patch(patch)


def test_a_patch_in_the_next_season_is_not_in_a_range_of_the_season_before(tmp_path):
    table = rulepay.RuleTable.load(write_rules(tmp_path, [SEG_16_13]))
    for patch in ('15.30', '16.12', '16.14', '17.13'):
        with pytest.raises(ValueError):
            table.for_patch(patch)


# the base case of the malformed-value tests loads; each case below changes one thing of it (on a patch other than 15.18-15.22, so that a
# refusal comes from the value and cannot come from the agreement with P3)
BASE_RID, ASSIST_RID, MULT_RID, FB_RID = F0_RULES
MALFORMED = {
    'by_level too short': set_value(BASE_RID, by_level=list(P3_BASE[:17])),
    'by_level too long': set_value(BASE_RID, by_level=list(P3_BASE) + [430]),
    'by_level has text': set_value(BASE_RID, by_level=['300'] + list(P3_BASE[1:])),
    'by_level has a fraction': set_value(BASE_RID, by_level=[300.5] + list(P3_BASE[1:])),
    'by_level has a zero': set_value(BASE_RID, by_level=[0] + list(P3_BASE[1:])),
    'by_level has a negative': set_value(BASE_RID, by_level=[-300] + list(P3_BASE[1:])),
    'by_level has a bool': set_value(BASE_RID, by_level=[True] + list(P3_BASE[1:])),
    'by_level has a null': set_value(BASE_RID, by_level=[None] + list(P3_BASE[1:])),
    'by_level is text': set_value(BASE_RID, by_level='300'),
    'by_level is missing': drop_value(BASE_RID, 'by_level'),
    'pool_share zero': set_value(ASSIST_RID, pool_share=0),
    'pool_share above one': set_value(ASSIST_RID, pool_share=1.5),
    'pool_share negative': set_value(ASSIST_RID, pool_share=-0.5),
    'pool_share as percent': set_value(ASSIST_RID, pool_share=50),
    'pool_share text': set_value(ASSIST_RID, pool_share='0.5'),
    'pool_share NaN': set_value(ASSIST_RID, pool_share=float('nan')),
    'pool_share too large for a float': set_value(ASSIST_RID, pool_share=10 ** 400),
    'flat_until_s too large for a float': set_value(MULT_RID, flat_until_s=10 ** 400),
    'by_level amount too large for a float': set_value(BASE_RID, by_level=[10 ** 400] + list(P3_BASE[1:])),
    'pool_share missing': drop_value(ASSIST_RID, 'pool_share'),
    'cap_share zero': set_value(ASSIST_RID, cap_share_of_base=0),
    'cap_share above one': set_value(ASSIST_RID, cap_share_of_base=1.01),
    'cap_share null': set_value(ASSIST_RID, cap_share_of_base=None),
    'cap_share missing': drop_value(ASSIST_RID, 'cap_share_of_base'),
    'ramp of zero width': set_value(MULT_RID, flat_until_s=55, full_at_s=55),
    'ramp that ends before it starts': set_value(MULT_RID, flat_until_s=55, full_at_s=30),
    'flat_until_s negative': set_value(MULT_RID, flat_until_s=-1),
    'start multiplier above one': set_value(MULT_RID, start_multiplier=1.5),
    'start multiplier negative': set_value(MULT_RID, start_multiplier=-0.1),
    'end multiplier above one': set_value(MULT_RID, end_multiplier=2),
    'end multiplier text': set_value(MULT_RID, end_multiplier='1.0'),
    'end multiplier infinite': set_value(MULT_RID, end_multiplier=float('inf')),
    'full_at_s missing': drop_value(MULT_RID, 'full_at_s'),
    'start_multiplier missing': drop_value(MULT_RID, 'start_multiplier'),
    'killer bonus negative': set_value(FB_RID, killer_bonus=-100),
    'killer bonus text': set_value(FB_RID, killer_bonus='100'),
    'killer bonus fraction': set_value(FB_RID, killer_bonus=99.5),
    'killer bonus missing': drop_value(FB_RID, 'killer_bonus'),
    'assist-pool bonus null': set_value(FB_RID, assist_pool_bonus=None),
    'assist-pool bonus missing': drop_value(FB_RID, 'assist_pool_bonus'),
    'assist-pool bonus text': set_value(FB_RID, assist_pool_bonus='0'),
    'assist-pool bonus bool': set_value(FB_RID, assist_pool_bonus=False),
    'value is not an object': replace_entry_field(ASSIST_RID, 'value', [0.5, 0.5]),
    'value is null': replace_entry_field(ASSIST_RID, 'value', None),
    'value is missing': replace_entry_field(ASSIST_RID, 'value', drop=True),
}


def test_the_base_case_of_the_malformed_value_tests_loads(tmp_path):
    table = rulepay.RuleTable.load(write_rules(tmp_path, [SEG_16_13]))
    assert table.for_patch('16.13').flat_until_s == 55.0 and table.for_patch('16.13').first_blood_killer_bonus == 100


def test_whole_numbers_written_as_floats_and_shares_written_as_integers_are_accepted(tmp_path):
    """Another writer may print 300.0 for 300 and 1 for 1.0; the values are the same numbers."""
    def mutate(doc):
        doc['rules'][BASE_RID][0]['value']['by_level'] = [float(g) for g in P3_BASE]
        doc['rules'][ASSIST_RID][0]['value'].update(pool_share=1, cap_share_of_base=1)
        doc['rules'][MULT_RID][0]['value'].update(flat_until_s=90.0, full_at_s=210.0, start_multiplier=1, end_multiplier=1)
    table = rulepay.RuleTable.load(write_rules(tmp_path, [dict(lo='16.13', hi='16.13')], mutate=mutate))
    rules = table.for_patch('16.13')
    assert rules.base_by_level == P3_BASE and all(type(g) is int for g in rules.base_by_level)
    assert (rules.pool_share, rules.cap_share_of_base, rules.start_multiplier, rules.end_multiplier) == (1.0, 1.0, 1.0, 1.0)
    assert rulepay.assist_pool(300, 6, 0, '16.13', 1, rules=table) == pytest.approx(300.0)       # min(300, 300) x 1.0


def test_the_multiplier_may_start_at_zero_and_end_at_one(tmp_path):
    table = rulepay.RuleTable.load(write_rules(tmp_path, [dict(SEG_16_13, m0=0.0, m1=1.0)]))
    assert rulepay.assist_pool(300, 6, 55_000, '16.13', 1, rules=table) == 0.0               # m = 0 until 55 s: no assist gold at all
    assert rulepay.assist_pool(300, 6, 115_000, '16.13', 1, rules=table) == pytest.approx(75.0)    # 150 x 0.5
    assert rulepay.assist_pool(300, 6, 175_000, '16.13', 1, rules=table) == pytest.approx(150.0)


@pytest.mark.parametrize('case', list(MALFORMED))
def test_a_value_that_cannot_be_a_payout_is_refused(tmp_path, case):
    with pytest.raises(ValueError):
        rulepay.RuleTable.load(write_rules(tmp_path, [SEG_16_13], mutate=MALFORMED[case]))


def test_a_first_blood_share_of_the_assist_pool_is_refused_because_the_pool_has_no_term_for_it(tmp_path):
    """Verified or not, a table with a First Blood bonus for the assist pool cannot be priced by this module: fail closed."""
    with pytest.raises(ValueError, match='First Blood'):
        rulepay.RuleTable.load(write_rules(tmp_path, [dict(SEG_16_13, first_blood=(100, 50))]))
    table = rulepay.RuleTable.load(write_rules(tmp_path, [dict(SEG_16_13, first_blood=(100, 0))]))      # the killer's bonus: not the pool's
    assert table.for_patch('16.13').first_blood_killer_bonus == 100 and table.for_patch('16.13').first_blood_assist_pool_bonus == 0


# ---- the loader: agreement with P3 on 15.18-15.22 ----------------------------------------------------------------------------------------
DISAGREE = {
    'level 7 pays 311': dict(base=P3_BASE[:6] + (311,) + P3_BASE[7:]),
    'a flat 250': dict(base=(250,) * 18),
    'level 18 pays 430': dict(base=P3_BASE[:17] + (430,)),
    'pool share 0.4': dict(pool_share=0.4),
    'pool share 1': dict(pool_share=1.0),
    'cap share 0.6': dict(cap_share=0.6),
    'ramp starts at 60 s': dict(flat=60),
    'ramp ends at 180 s': dict(full=180),
    'ramp of 16.13': dict(flat=55, full=175),
    'start multiplier 0.25': dict(m0=0.25),
    'end multiplier 0.9': dict(m1=0.9),
    'First Blood pays the killer': dict(first_blood=(100, 0)),
}


@pytest.mark.parametrize('case', list(DISAGREE))
def test_a_verified_table_that_disagrees_with_p3_is_refused(tmp_path, case):
    with pytest.raises(ValueError, match='P3'):
        rulepay.RuleTable.load(write_rules(tmp_path, [DISAGREE[case]]))


@pytest.mark.parametrize('deviant', P3_PATCHES)
def test_the_agreement_is_checked_on_every_patch_of_the_range(tmp_path, deviant):
    """One patch of 15.18-15.22 with another value is enough: the table has one range per patch and only ``deviant`` differs."""
    def one_range_per_patch(**deviation):
        return [dict(lo=p, hi=p, **(deviation if p == deviant else {})) for p in P3_PATCHES]
    rulepay.RuleTable.load(write_rules(tmp_path, one_range_per_patch(), name='control.json'))                # all five equal P3: loads
    with pytest.raises(ValueError, match='P3'):
        rulepay.RuleTable.load(write_rules(tmp_path, one_range_per_patch(pool_share=0.4), name='bad.json'))


def test_a_range_that_reaches_beyond_15_18_to_15_22_is_checked_on_the_part_inside(tmp_path):
    with pytest.raises(ValueError, match='P3'):
        rulepay.RuleTable.load(write_rules(tmp_path, [dict(lo='15.9', hi='15.30', pool_share=0.4)]))
    with pytest.raises(ValueError, match='P3'):
        rulepay.RuleTable.load(write_rules(tmp_path, [dict(lo='15.22', hi='15.30', cap_share=0.4)], name='edge.json'))
    outside = rulepay.RuleTable.load(write_rules(tmp_path, [dict(lo='15.23', hi='15.30', pool_share=0.4)], name='outside.json'))
    assert outside.for_patch('15.25').pool_share == 0.4                                      # no P3 patch inside: nothing to agree with


def test_a_table_for_a_part_of_the_range_is_checked_on_that_part(tmp_path):
    table = rulepay.RuleTable.load(write_rules(tmp_path, [dict(lo='15.18', hi='15.20')]))
    assert table.for_patch('15.20') == rulepay.P3_RULES.for_patch('15.20')
    with pytest.raises(ValueError):
        table.for_patch('15.21')
    with pytest.raises(ValueError, match='P3'):
        rulepay.RuleTable.load(write_rules(tmp_path, [dict(lo='15.18', hi='15.20', flat=60)], name='bad.json'))


def test_a_patch_that_is_not_served_by_all_four_f0_rules_is_not_served(tmp_path):
    """The base bounty covers 15.18-15.22 but the multiplier only 15.18-15.20: 15.21 is refused when priced, not guessed."""
    def mutate(doc):
        doc['rules'][MULT_RID][0]['patch_range'] = ['15.18', '15.20']
    table = rulepay.RuleTable.load(write_rules(tmp_path, mutate=mutate))
    assert rulepay.base_bounty(7, '15.20', rules=table) == 310
    with pytest.raises(ValueError):
        rulepay.base_bounty(7, '15.21', rules=table)
    with pytest.raises(ValueError):
        rulepay.assist_pool(300, 6, 150_000, '15.21', 1, rules=table)


def test_other_patches_need_not_equal_p3(tmp_path):
    table = rulepay.RuleTable.load(write_rules(tmp_path, [dict(), SEG_16_13, SEG_OTHER]))
    assert table.for_patch('16.13').flat_until_s == 55.0 and table.for_patch('16.6').pool_share == 0.4
    assert table.for_patch('15.20') == rulepay.P3_RULES.for_patch('15.20')


# ---- the loader: which patches it has to serve -------------------------------------------------------------------------------------------
def real_shape_segments():
    """A 15.18-15.22 range with a documented multiplier (as the real table has today) and a 16.13 range with an unknown First Blood part."""
    return [dict(status={MULT_RID: 'documented'}), dict(SEG_16_13, status={FB_RID: 'unknown'})]


def test_by_default_the_whole_file_must_be_verified(tmp_path):
    path = write_rules(tmp_path, real_shape_segments())
    with pytest.raises(ValueError, match='verified'):
        rulepay.RuleTable.load(path)


def test_asking_for_patches_checks_only_the_entries_that_cover_them(tmp_path):
    segs = [dict(), dict(SEG_16_13, status={FB_RID: 'unknown'})]
    path = write_rules(tmp_path, segs)
    with pytest.raises(ValueError, match='verified'):
        rulepay.RuleTable.load(path)                                       # the unknown part of 16.13 refuses the whole file
    table = rulepay.RuleTable.load(path, patches=P3_PATCHES)               # S3 asks for 15.18-15.22 only
    assert table.for_patch('15.19') == rulepay.P3_RULES.for_patch('15.19')
    with pytest.raises(ValueError):
        table.for_patch('16.13')                                           # the entry that was not asked for is not used
    with pytest.raises(ValueError, match='verified'):
        rulepay.RuleTable.load(path, patches=('15.20', '16.13'))
    with pytest.raises(ValueError, match='verified'):
        rulepay.RuleTable.load(path, patches=['16.13'])


def test_asked_for_patches_may_be_a_single_patch_string_or_any_iterable(tmp_path):
    path = write_rules(tmp_path, [dict(), SEG_16_13])
    assert rulepay.RuleTable.load(path, patches='16.13').for_patch('16.13').flat_until_s == 55.0
    assert rulepay.RuleTable.load(path, patches=iter(['15.20'])).for_patch('15.20') == rulepay.P3_RULES.for_patch('15.20')
    assert rulepay.RuleTable.load(path, patches={'15.20', '16.13'}).for_patch('16.13').full_at_s == 175.0


def test_a_documented_range_that_was_not_asked_for_does_not_matter(tmp_path):
    """The real table today: a 15.x multiplier that is only documented, a verified 16.13. Asking for 16.13 alone loads; for 15.x, not."""
    segs = [dict(status={MULT_RID: 'documented'}), SEG_16_13]
    path = write_rules(tmp_path, segs)
    assert rulepay.RuleTable.load(path, patches=['16.13']).for_patch('16.13').flat_until_s == 55.0
    with pytest.raises(ValueError, match='verified'):
        rulepay.RuleTable.load(path, patches=['15.20'])
    with pytest.raises(ValueError, match='verified'):
        rulepay.RuleTable.load(path)


@pytest.mark.parametrize('patches', [(), [], ('15.17',), ('15.20', '16.13'), ('x',), (None,), ('15.20', 15.2)])
def test_asked_for_patches_that_the_file_does_not_cover_are_refused(tmp_path, patches):
    with pytest.raises(ValueError):
        rulepay.RuleTable.load(write_rules(tmp_path), patches=patches)


def test_the_p3_agreement_is_checked_when_the_asked_for_patches_are_in_the_range(tmp_path):
    path = write_rules(tmp_path, [dict(pool_share=0.4)])
    with pytest.raises(ValueError, match='P3'):
        rulepay.RuleTable.load(path, patches=['15.20'])
    with pytest.raises(ValueError, match='P3'):
        rulepay.RuleTable.load(path)


# ---- the table serves exactly the asked patches, and every overlap counts (fix round 1: probes A, B and D of the task review) ------------
def test_probe_a_an_unverified_entry_that_overlaps_a_read_one_refuses_the_load(tmp_path):
    """Probe A: the multiplier is verified for 15.18-15.22 and also documented for 15.22-15.25 (the entries overlap at 15.22). Asking for
    15.18-15.21 used to load and to serve 15.22, while the default load refused the file. Now both refuse: two entries cover 15.22."""
    path = write_rules(tmp_path, mutate=add_entry(MULT_RID, '15.22', '15.25', 'documented', flat=60))
    with pytest.raises(ValueError):
        rulepay.RuleTable.load(path)
    with pytest.raises(ValueError, match='same patch'):
        rulepay.RuleTable.load(path, patches=('15.18', '15.19', '15.20', '15.21'))


@pytest.mark.parametrize('asked', ['15.20', ['15.18', '15.19'], P3_PATCHES], ids=['one patch', 'two patches', 'all five'])
def test_probe_b_two_verified_entries_for_one_patch_refuse_the_load_whether_or_not_one_is_read(tmp_path, asked):
    """Probe B: two verified assist-pool entries disagree on 15.22 (pool_share 0.5 and 0.4). Asking for 15.20 used to load silently and
    to use 0.5, while the default load refused ("two entries cover the same patch"). Now every call refuses, as the default does."""
    path = write_rules(tmp_path, mutate=add_entry(ASSIST_RID, '15.22', '15.30', pool_share=0.4))
    with pytest.raises(ValueError, match='same patch'):
        rulepay.RuleTable.load(path)
    with pytest.raises(ValueError, match='same patch'):
        rulepay.RuleTable.load(path, patches=asked)


def test_an_overlap_between_entries_that_are_not_asked_for_is_refused_too(tmp_path):
    """The overlap check looks at every entry of the file: here neither overlapping entry covers an asked patch."""
    path = write_rules(tmp_path, [dict(), dict(lo='16.13', hi='16.14')], mutate=add_entry(ASSIST_RID, '16.14', '16.15'))
    with pytest.raises(ValueError, match='same patch'):
        rulepay.RuleTable.load(path)
    with pytest.raises(ValueError, match='same patch'):
        rulepay.RuleTable.load(path, patches=P3_PATCHES)


def test_entries_that_touch_without_overlapping_are_fine(tmp_path):
    """15.18-15.20 and 15.21-15.22 share no patch: not an overlap, with or without asked patches."""
    path = write_rules(tmp_path, [dict(lo='15.18', hi='15.20'), dict(lo='15.21', hi='15.22')])
    assert rulepay.RuleTable.load(path).for_patch('15.21') == rulepay.P3_RULES.for_patch('15.21')
    assert rulepay.RuleTable.load(path, patches='15.21').for_patch('15.21') == rulepay.P3_RULES.for_patch('15.21')


def test_the_order_of_the_entries_in_the_file_does_not_matter_to_the_overlap_check(tmp_path):
    """Later patches first is a valid file; an overlap is found wherever the two entries stand in the list."""
    reversed_ok = write_rules(tmp_path, [SEG_16_13, dict()], name='reversed.json')
    whole = rulepay.RuleTable.load(reversed_ok)
    assert whole.for_patch('16.13').flat_until_s == 55.0 and whole.for_patch('15.20') == rulepay.P3_RULES.for_patch('15.20')
    for asked, served, not_served in ((P3_PATCHES, '15.20', '16.13'), (['16.13'], '16.13', '15.20'),
                                      (['15.20', '16.13'], '16.13', '15.19')):
        table = rulepay.RuleTable.load(reversed_ok, patches=asked)
        assert table.for_patch(served) is not None
        with pytest.raises(ValueError):
            table.for_patch(not_served)
    with pytest.raises(ValueError, match='same patch'):                  # the later patches first, and the overlap in the middle
        rulepay.RuleTable.load(write_rules(tmp_path, [dict(lo='15.25', hi='15.30'), dict(lo='15.18', hi='15.22'),
                                                      dict(lo='15.22', hi='15.24', pool_share=0.4)], name='middle.json'))


def test_the_constructor_keeps_the_invariants_itself():
    """``load`` and ``P3_RULES`` are the ways to a table, but the invariants do not depend on the caller having checked first."""
    def segment(lo, hi):
        return rulepay._Segment(lo, hi, (0, 0))                          # the value is not looked at when the table is built
    one = {rid: (segment((15, 18), (15, 22)),) for rid in F0_RULES}
    assert rulepay.RuleTable(one) is not None
    with pytest.raises(ValueError, match='same patch'):
        rulepay.RuleTable({**one, ASSIST_RID: (segment((15, 18), (15, 22)), segment((15, 22), (15, 30)))})
    with pytest.raises(ValueError):                                      # a rule of the four is missing
        rulepay.RuleTable({rid: segs for rid, segs in one.items() if rid != FB_RID})


def test_probe_d_the_table_serves_exactly_the_asked_patches(tmp_path):
    """Probe D: a verified entry for 15.9-15.30, asked for 15.20 only, used to serve all of 15.9-15.30 (15.29 included)."""
    path = write_rules(tmp_path, [dict(lo='15.9', hi='15.30')])
    table = rulepay.RuleTable.load(path, patches='15.20')
    assert table.for_patch('15.20') == rulepay.P3_RULES.for_patch('15.20')
    for patch in ('15.9', '15.19', '15.21', '15.29', '15.30', '16.1'):
        with pytest.raises(ValueError):
            table.for_patch(patch)
    with pytest.raises(ValueError):
        rulepay.base_bounty(6, '15.29', rules=table)
    with pytest.raises(ValueError):
        rulepay.assist_pool(300, 6, 150_000, '15.29', 1, rules=table)
    with pytest.raises(ValueError):                                      # one patch that was not asked for refuses the whole call
        rulepay.assist_pools([300, 300], [6, 6], [150_000, 150_000], ['15.20', '15.29'], [1, 1], rules=table)
    assert rulepay.assist_pool(300, 6, 150_000, '15.20', 1, rules=table) == pytest.approx(112.5)
    whole = rulepay.RuleTable.load(path)                                 # without patches the table serves every patch of the entries
    assert [whole.for_patch(p) == rulepay.P3_RULES.for_patch('15.20') for p in ('15.9', '15.29', '15.30')] == [True] * 3


def test_a_table_loaded_for_patches_serves_those_and_no_others_however_wide_the_entries_are(tmp_path):
    path = write_rules(tmp_path, [dict(lo='15.9', hi='15.30')])
    table = rulepay.RuleTable.load(path, patches=['15.10', '15.20', '15.30'])
    served = []
    for minor in range(5, 35):
        try:
            table.for_patch(f'15.{minor}')
            served.append(minor)
        except ValueError:
            pass
    assert served == [10, 20, 30]


def test_a_documented_entry_for_other_patches_does_not_widen_what_is_served(tmp_path):
    """The variant of probe A without an overlap: verified entries for 15.18-15.22 and a documented multiplier for 15.23-15.25. Asking
    for 15.18-15.21 loads (the documented entry is not read) and serves those four patches only, although the verified entries
    cover 15.22 too; the default load refuses the documented entry."""
    path = write_rules(tmp_path, mutate=add_entry(MULT_RID, '15.23', '15.25', 'documented', flat=60))
    with pytest.raises(ValueError, match='verified'):
        rulepay.RuleTable.load(path)
    asked = ('15.18', '15.19', '15.20', '15.21')
    table = rulepay.RuleTable.load(path, patches=asked)
    for patch in asked:
        assert table.for_patch(patch) == rulepay.P3_RULES.for_patch(patch)
    for patch in ('15.22', '15.23', '15.25'):
        with pytest.raises(ValueError):
            table.for_patch(patch)


def test_the_p3_agreement_concerns_the_patches_the_table_serves(tmp_path):
    """An entry for 15.18-15.30 with another pool share disagrees with P3 on 15.18-15.22. Asking for 15.20 (in the range) refuses it;
    asking for 15.25 (outside) serves 15.25 only, so there is no patch of P3 to agree with."""
    path = write_rules(tmp_path, [dict(lo='15.18', hi='15.30', pool_share=0.4)])
    with pytest.raises(ValueError, match='P3'):
        rulepay.RuleTable.load(path)
    with pytest.raises(ValueError, match='P3'):
        rulepay.RuleTable.load(path, patches='15.20')
    outside = rulepay.RuleTable.load(path, patches='15.25')
    assert outside.for_patch('15.25').pool_share == 0.4
    with pytest.raises(ValueError):
        outside.for_patch('15.20')


# ---- a loaded table for other patches: 16.13 has other breakpoints of the multiplier -----------------------------------------------------
@pytest.fixture()
def three_ranges(tmp_path):
    return rulepay.RuleTable.load(write_rules(tmp_path, [dict(), SEG_OTHER, SEG_16_13]))


def test_a_16_13_table_moves_the_breakpoints_of_the_multiplier(three_ranges):
    """16.13: 0.5 until 55 s, linear to 1.0 at 175 s (the game file; the 15.x breakpoints are 90 s and 210 s)."""
    pool = lambda t_ms, patch: rulepay.assist_pool(300, 6, t_ms, patch, 1, rules=three_ranges)      # noqa: E731
    assert pool(55_000, '16.13') == pytest.approx(75.0)                    # 150 x 0.5
    assert pool(115_000, '16.13') == pytest.approx(112.5)                  # 150 x (0.5 + 0.5 x 60/120)
    assert pool(145_000, '16.13') == pytest.approx(131.25)                 # 150 x (0.5 + 0.5 x 90/120)
    assert pool(175_000, '16.13') == pytest.approx(150.0)                  # 150 x 1.0
    assert pool(115_000, '15.20') == pytest.approx(90.625)                 # the same time on 15.20: 150 x (0.5 + 0.5 x 25/120)
    assert pool(115_000, '15.20') == pytest.approx(rulepay.assist_pool(300, 6, 115_000, '15.20', 1))
    assert pool(60_000, '16.13') > pool(60_000, '15.20') - 1e-12           # 16.13 ramps earlier: never lower than 15.x at the same time
    assert pool(60_000, '16.13') == pytest.approx(150 * (0.5 + 0.5 * 5 / 120))


def test_every_value_of_the_patch_is_read_from_the_table(three_ranges):
    """Base bounty, both shares and the multiplier of 16.5-16.7 differ from P3 in every number, and the answers follow the table."""
    assert [rulepay.base_bounty(l, '16.6', rules=three_ranges) for l in (1, 6, 7, 18)] == [200, 205, 206, 217]
    assert rulepay.assist_pool(300, 6, 300_000, '16.6', 2, rules=three_ranges) == pytest.approx(46.125)    # min(120, 0.3 x 205) x 0.75
    assert rulepay.assist_pool(100, 6, 60_000, '16.6', 1, rules=three_ranges) == pytest.approx(20.0)  # min(40, 61.5) x (0.25 + 0.5 x 30/60)
    assert rulepay.assist_pool(100, 6, 0, '16.6', 1, rules=three_ranges) == pytest.approx(10.0)            # min(40, 61.5) x 0.25
    assert rulepay.assist_pool(100, 6, 60_000, '16.6', 0, rules=three_ranges) == 0.0
    for patch in ('16.4', '16.8', '16.12', '16.14'):
        with pytest.raises(ValueError):
            rulepay.base_bounty(6, patch, rules=three_ranges)


def test_the_default_rules_are_not_changed_by_loading_a_table(three_ranges):
    assert rulepay.assist_pool(300, 6, 115_000, '15.20', 1) == pytest.approx(90.625)
    with pytest.raises(ValueError):
        rulepay.assist_pool(300, 6, 115_000, '16.13', 1)


def test_rules_must_be_a_rule_table(tmp_path):
    for bad in (str(write_rules(tmp_path)), {'15.20': 1}, 3):
        with pytest.raises(TypeError):
            rulepay.base_bounty(6, '15.20', rules=bad)
        with pytest.raises(TypeError):
            rulepay.assist_pools([300], [6], [150_000], '15.20', [1], rules=bad)


# ---- the vectorised form -----------------------------------------------------------------------------------------------------------------
GRID = list(itertools.product((0, 100, 300, 450, 700), (1, 6, 7, 12, 18),
                              (0, 45_000, 90_000, 120_000, 150_000, 210_000, 600_000), (0, 1, 3), P3_PATCHES))


def grid_arrays():
    bounty, level, t_ms, n, patch = (np.array(col) for col in zip(*GRID))
    return bounty, level, t_ms, patch, n


def test_assist_pools_match_the_scalar_form_and_the_reference_on_a_grid():
    bounty, level, t_ms, patch, n = grid_arrays()
    out = rulepay.assist_pools(bounty, level, t_ms, patch, n)
    assert out.shape == (len(GRID),) and out.dtype == np.float64
    scalar = np.array([rulepay.assist_pool(b, lv, t, p, k) for b, lv, t, k, p in GRID])
    reference = np.array([ref_pool(b, lv, t, k) for b, lv, t, k, _ in GRID])
    assert np.array_equal(out, scalar)
    assert np.allclose(out, reference, rtol=1e-12, atol=1e-12)


def test_assist_pools_match_the_reference_on_random_draws():
    rng = np.random.default_rng(20261021)
    size = 5000
    bounty = rng.uniform(0, 1500, size)                                    # a float bounty is as good as an integer one
    level, t_ms, n = rng.integers(1, 19, size), rng.integers(0, 3_000_000, size), rng.integers(0, 6, size)
    out = rulepay.assist_pools(bounty, level, t_ms, '15.20', n)
    reference = np.array([ref_pool(b, lv, t, k) for b, lv, t, k in zip(bounty, level, t_ms, n)])
    assert np.allclose(out, reference, rtol=1e-12, atol=1e-12)
    assert np.all(out >= 0) and np.all(out[n == 0] == 0) and np.all(out <= 210.0)


@pytest.mark.parametrize('small', [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize('big', [np.int32, np.int64, np.uint32, np.float32, np.float64])
def test_assist_pools_take_the_dtypes_that_the_stores_use(small, big):
    """S1 stores the bounty as int32; a level or a count may be any integer type, and whole numbers may come as floats."""
    level, n = np.array([1, 6, 7, 18], small), np.array([1, 2, 0, 5], small)
    bounty, t_ms = np.array([300, 700, 250, 500], big), np.array([0, 120_000, 100_000, 50_000], big)
    out = rulepay.assist_pools(bounty, level, t_ms, '15.19', n)
    assert out.dtype == np.float64
    assert np.allclose(out, [75.0, 93.75, 0.0, 105.0], rtol=1e-12)      # 150 x 0.5; 150 x 0.625; no assisters; min(250, 210) x 0.5


def test_assist_pools_mix_patches_of_a_loaded_table(three_ranges):
    patch = np.array(['15.20', '16.13', '15.18', '16.6', '16.13', '15.22'])
    ones = np.ones(6, dtype=int)
    out = rulepay.assist_pools(300 * ones, 6 * ones, 115_000 * ones, patch, ones, rules=three_ranges)
    assert out[[0, 2, 5]] == pytest.approx(90.625)                         # 15.x
    assert out[[1, 4]] == pytest.approx(112.5)                            # 16.13
    assert out[3] == pytest.approx(46.125)                                # 16.6: min(0.4 x 300, 0.3 x 205) x 0.75 (the ramp ended at 90 s)
    assert np.array_equal(out, [rulepay.assist_pool(300, 6, 115_000, str(p), 1, rules=three_ranges) for p in patch])


def test_one_unknown_patch_in_an_array_refuses_the_whole_call():
    patch = np.array(['15.20', '15.20', '16.13'])
    with pytest.raises(ValueError):
        rulepay.assist_pools([300, 300, 300], [6, 6, 6], [1000, 1000, 1000], patch, [1, 1, 1])


def test_a_patch_array_may_be_an_object_array_or_a_list_of_strings():
    args = ([300, 450], [6, 7], [150_000, 200_000], None, [1, 2])
    expected = rulepay.assist_pools(*args[:3], np.array(['15.19', '15.21']), args[4])
    assert np.array_equal(rulepay.assist_pools(*args[:3], ['15.19', '15.21'], args[4]), expected)
    assert np.array_equal(rulepay.assist_pools(*args[:3], np.array(['15.19', '15.21'], dtype=object), args[4]), expected)
    with pytest.raises(ValueError):
        rulepay.assist_pools(*args[:3], np.array(['15.19', 15.21], dtype=object), args[4])


def test_assist_pools_keep_the_shape_of_the_input():
    bounty = np.array([[300, 700], [100, 0]])
    out = rulepay.assist_pools(bounty, 6, 300_000, '15.20', 1)             # single values are repeated
    assert out.shape == (2, 2) and out.tolist() == [[150.0, 150.0], [50.0, 0.0]]
    assert rulepay.assist_pools(np.array([], dtype=np.int32), np.array([], dtype=np.int8), np.array([]), np.array([], dtype=str),
                                np.array([], dtype=np.int16)).shape == (0,)
    one = rulepay.assist_pools(300, 6, 150_000, '15.20', 1)
    assert isinstance(one, np.ndarray) and one.shape == () and float(one) == 112.5


def test_the_scalar_form_refuses_arrays():
    with pytest.raises(ValueError):
        rulepay.assist_pool([300, 400], 6, 150_000, '15.20', 1)
    with pytest.raises(ValueError):
        rulepay.assist_pool(300, 6, 150_000, ['15.20'], 1)
    assert rulepay.assist_pool(np.int32(300), np.int8(6), np.int32(150_000), np.str_('15.20'), np.int16(1)) == 112.5


def test_assist_pools_with_different_shapes_are_refused():
    with pytest.raises(ValueError):
        rulepay.assist_pools([300, 300], [6, 6, 6], [1000, 1000], '15.20', [1, 1])
    with pytest.raises(ValueError):
        rulepay.assist_pools([300, 300], [6, 6], [1000, 1000], ['15.20', '15.20', '15.20'], [1, 1])
    with pytest.raises(ValueError):
        rulepay.assist_pools([300, 300], [6, 6], [1000, 1000], '15.20', [[1, 1]])


def test_the_inputs_are_not_changed():
    arrays = [np.array([300, 700]), np.array([6, 18]), np.array([90_000, 400_000]), np.array(['15.20', '15.21']), np.array([1, 0])]
    before = [a.copy() for a in arrays]
    rulepay.assist_pools(*arrays)
    assert all(np.array_equal(a, b) for a, b in zip(arrays, before))


# ---- bad inputs --------------------------------------------------------------------------------------------------------------------------
OK = dict(bounty=300, victim_level=6, t_ms=150_000, patch='15.20', n_assisters=1)
BAD_INPUTS = {
    'negative bounty': dict(bounty=-1),
    'bounty NaN': dict(bounty=float('nan')),
    'bounty infinite': dict(bounty=float('inf')),
    'bounty text': dict(bounty='300'),
    'level 0': dict(victim_level=0),
    'level 19': dict(victim_level=19),
    'level negative': dict(victim_level=-6),
    'level not whole': dict(victim_level=6.5),
    'level NaN': dict(victim_level=float('nan')),
    'level text': dict(victim_level='6'),
    'negative time': dict(t_ms=-1),
    'time NaN': dict(t_ms=float('nan')),
    'time infinite': dict(t_ms=float('inf')),
    'time text': dict(t_ms='150000'),
    'negative assisters': dict(n_assisters=-1),
    'assisters not whole': dict(n_assisters=1.5),
    'assisters NaN': dict(n_assisters=float('nan')),
    'assisters text': dict(n_assisters='1'),
}


@pytest.mark.parametrize('case', list(BAD_INPUTS))
def test_bad_inputs_are_refused_by_both_forms(case):
    bad = {**OK, **BAD_INPUTS[case]}
    with pytest.raises(ValueError):
        rulepay.assist_pool(**bad)
    column = lambda name: np.array([OK[name], bad[name], OK[name]])      # noqa: E731  the bad value sits in the middle row
    with pytest.raises(ValueError):
        rulepay.assist_pools(column('bounty'), column('victim_level'), column('t_ms'), column('patch'), column('n_assisters'))


def test_boolean_inputs_are_refused():
    for name in ('bounty', 'victim_level', 't_ms', 'n_assisters'):
        with pytest.raises(ValueError):
            rulepay.assist_pool(**{**OK, name: True})
        with pytest.raises(ValueError):
            rulepay.assist_pools(**{k: (np.array([True]) if k == name else np.array([v])) for k, v in OK.items()})


@pytest.mark.parametrize('level', [0, 19, -1, 6.5, float('nan'), '6', True, None, [6, 7]])
def test_base_bounty_refuses_a_bad_level(level):
    with pytest.raises(ValueError):
        rulepay.base_bounty(level, '15.20')


def test_base_bounty_takes_whole_numbers_of_any_type():
    for level in (6, np.int8(6), np.uint16(6), np.int64(6), 6.0, np.float32(6.0)):
        assert rulepay.base_bounty(level, '15.20') == 300 and type(rulepay.base_bounty(level, '15.20')) is int
    assert rulepay.base_bounty(np.int16(18), '15.20') == 420


def test_error_messages_hold_no_input_values():
    """No value is echoed: not a patch (it could be a match key), and not the numbers (a message may reach a log)."""
    cases = [dict(patch='KR_7123456789'), dict(bounty=-123456), dict(victim_level=77777), dict(t_ms=-998877), dict(n_assisters=-445566)]
    for case in cases:
        for call in (lambda: rulepay.assist_pool(**{**OK, **case}),
                     lambda: rulepay.assist_pools(**{k: np.array([v]) for k, v in {**OK, **case}.items()})):
            with pytest.raises(ValueError) as err:
                call()
            assert not any(str(v) in str(err.value) for v in case.values()), str(err.value)


def test_a_bad_table_value_does_not_leak_a_match_key_either(tmp_path):
    """A table cannot hold a match key, but the loader's messages name rules and fields only, never the file's path."""
    path = write_rules(tmp_path, name='KR_7123456789.json', mutate=set_value(BASE_RID, by_level=[0] * 18))
    with pytest.raises(ValueError) as err:
        rulepay.RuleTable.load(path)
    assert 'KR_7123456789' not in str(err.value)


# ---- how the module is imported ----------------------------------------------------------------------------------------------------------
def test_rulepay_imports_only_the_standard_library_and_numpy():
    """The rules never read a match, a store or a run folder: nothing of the c_ept runners (E6 guard, S1 readers) is reachable from here."""
    imports = {name.split('.')[0] for name in module_level_imports(Path(rulepay.__file__))}
    assert imports <= {'__future__', 'dataclasses', 'json', 'math', 'numpy', 'pathlib', 're'}, sorted(imports)


def test_importing_the_module_loads_no_other_c_ept_module():
    code = ('import sys\nbefore = set(sys.modules)\nimport c_ept.rulepay\n'
            'print(sorted(m for m in set(sys.modules) - before if m.startswith("c_ept")))')
    done = subprocess.run([sys.executable, '-B', '-c', code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0 and done.stdout.strip() == "['c_ept', 'c_ept.rulepay', 'c_ept.valuation', 'c_ept.valuation.rulepay']", done.stderr[-500:]
