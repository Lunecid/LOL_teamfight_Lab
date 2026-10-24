r"""Golden tests: the finished S1 run against the link-rule prototype, and the golden lock of 20 matches.

These tests read a real S1 run (the one of task 8a, on the cached matches of 15.18-15.21) and the prototype folder
``D:\LOL_Data\runs\20261019_link_rule_fit_1`` (read only). They run only when the environment variable ``C_EPT_S1_RUN`` names the run
folder (or its ``s1`` store); without it every test is skipped. ``C_EPT_GOLDEN_DIR`` overrides the golden folder (default ``<run>\golden``).

    $env:C_EPT_S1_RUN = 'D:\LOL_Data\runs\20261020_c_ept_s1_1'
    & $PY -B -m pytest c_ept/tests/test_golden_s1.py -q --tb=short -p no:cacheprovider

``test_golden_20_unchanged`` is skipped until ``python -m c_ept.run_s1 --golden 20 --out <run>\golden`` has written GOLDEN.json.

The statistics come from ``c_ept.report_s1``, which recomputes the prototype's formulas on the stored tables; each test names the prototype
file it compares with. Floats are compared exactly when the computation mirrors the prototype's (same operations on the same numbers); the
rates of the aftermath probe are rounded to 4 decimals as the probe does, and the onset shares are compared within 0.001 (ruling 25).
Run with ``--tb=short``: a long traceback prints fixture arguments, which can hold match keys.
"""
import json
import os
import shutil

import numpy as np
import pytest

from c_ept import cohort, report_s1, run_s1
from c_ept.store_core import Store
from c_ept.tests.conftest import golden_folder

RUN = os.environ.get('C_EPT_S1_RUN')
pytestmark = pytest.mark.skipif(not RUN, reason='C_EPT_S1_RUN is not set: there is no S1 run to compare with the prototype')

CHECK_UNITS = ('units_per_kill', 'share_single_kill', 'spread_gt_4264_n', 'spread_max')      # the brief's four numbers of link_15s


# ---- fixtures ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='session')
def s1():
    """The S1 store of the run (``C_EPT_S1_RUN`` names the run folder or its ``s1`` store; ``load_s1`` takes either); it verifies the
    SHA-256 of every file."""
    return run_s1.load_s1(RUN)


@pytest.fixture(scope='session')
def golden_dir():
    """The golden folder of the run: ``C_EPT_GOLDEN_DIR`` or ``<run folder>\\golden``. The lookup is ``conftest.golden_folder``, the one
    that the ``golden_records`` fixture and the other tests use, not a copy of its rule."""
    return golden_folder(RUN)


@pytest.fixture(scope='session')
def ref():
    """The prototype's reference values: fit_kills.npz and the JSON files (``report_s1.load_prototype``)."""
    return report_s1.load_prototype()


@pytest.fixture
def link_units(proto):
    """The prototype module. When it cannot be loaded the test fails with a message: a missing prototype never makes a golden test pass."""
    if proto is None:
        pytest.fail('the prototype module link_units.py could not be loaded (see the warning of the loader): the comparison cannot run',
                    pytrace=False)
    return proto


def failing(c):
    """Names of the checks of a comparison that failed (numbers only, no match key)."""
    return [name for name, v in c['checks'].items() if not v['ok']]


# ---- the tests of the brief ---------------------------------------------------------------------------------------------------------
def test_kills_equal_prototype_npz(s1, ref):
    """The kills of the 19,100 FIT matches equal fit_kills.npz (t, x, y, killer, victim, amask, the match of every kill), and so do T,
    the teams and the patch of the matches: the S1 order of the FIT matches is the prototype's (patch, then key). NaN equals NaN."""
    c = report_s1.cmp_kills(s1, ref)
    assert c['ok'], failing(c)
    assert (c['matches'], c['kills']) == (19_100, 991_410)


def test_units_equal_prototype(s1, link_units):
    """For every FIT match the units (lists of kill indices) equal the prototype's link_components(t, x, y, 15000) on the stored kills,
    in the same order. The comparison runs on all 19,100 matches."""
    c = report_s1.cmp_units(s1, link_units)
    assert c['ok'] and c['matches'] == 19_100, c['checks']


def test_fit_checks_equal_link_g15_check(s1, ref):
    """link_g15_check.json link_15s: units per kill, the share of single-kill units, the units wider than 4,264 and the widest spread
    (recomputed in float64 from the stored kills with the prototype's expression), the size cohorts of the units in scope."""
    got, g15 = report_s1.fit_unit_stats(s1), ref['g15']
    for key in CHECK_UNITS:
        assert got[key] == g15[key], key
    assert got['cohort_share_in_scope'] == g15['cohort_share_in_scope']
    assert (got['n_kills'], got['n_units'], got['n_in_scope'], got['spread_gt_4264_n']) == (991_410, 597_564, 574_285, 432)
    # the stored columns agree with their definitions: f, last, tau = f - 15,000, delta = max(last + 1, f + 20,000), n_kills and kill0
    # from the kills of each unit; nb, nr and the size class from the people of the kills; the scope; the float32 spread
    assert got['stored_times_equal_prototype_rule'] and got['stored_nb_nr_equal_people_rule']
    assert got['stored_size_equals_people_rule'] and got['stored_in_scope_equals_prototype_rule']
    assert got['stored_spread_equals_float32_of_float64']
    c = report_s1.cmp_unit_checks(s1, ref)
    assert c['ok'], failing(c)


def test_attribution_equals_probe(s1, ref):
    """objective_participation_probe.json and objective_attribution_probe.json against our stored attribution (``objectives/unit >= 0``),
    for the objectives of the FIT matches with a position at 120 s or later. Two explained deviations are pinned (rulings 16 and 23):

    1. n is the prototype's n + 2: the prototype skips FIT matches without any kill, and two plates fall in such matches. They have no
       active unit, so the count without an active unit grows by the same 2.
    2. The overlap count of the herald and the one of ``all`` are the prototype's - 1: one herald has killerTeamId 300, so its taker is
       unknown and our rule never attributes an objective with an unknown taker (ruling 16); the prototype counts its overlap.

    The expected counts are rebuilt from the probes' shares (share * n, an integer), the deviations are applied and the shares are
    recomputed from the adjusted counts; everything else is the prototype's own filter and compares exactly.
    """
    extra_n, missing_overlap = {'plates': 2}, {'herald': 1}
    assert report_s1.ATTRIBUTION_DEVIATIONS == {'extra_n': extra_n, 'missing_overlap': missing_overlap}
    got, part, attr = report_s1.fit_attribution_stats(s1), ref['part'], ref['attr']
    assert part['all']['n'] == attr['all']['n'] == 582_135 and got['all']['n'] == 582_135 + 2
    for kind in report_s1.PROTO_KINDS:
        assert part[kind]['n'] == attr['by_kind'][kind]['n']
        n = part[kind]['n'] + extra_n.get(kind, 0)
        overlap = round(part[kind]['participant_overlap_with_active_unit'] * part[kind]['n']) - missing_overlap.get(kind, 0)
        no_active = round(attr['by_kind'][kind]['share_no_active_unit'] * attr['by_kind'][kind]['n']) + extra_n.get(kind, 0)
        g = got[kind]
        assert (g['n'], g['overlap'], g['n'] - g['active']) == (n, overlap, no_active), kind
        assert g['participant_overlap_with_active_unit'] == overlap / n and g['share_no_active_unit'] == no_active / n, kind
    n = 582_135 + 2
    overlap = round(part['all']['participant_overlap_with_active_unit'] * 582_135) - 1
    no_active = round(attr['all']['share_no_active_unit'] * 582_135) + 2
    g = got['all']
    assert (g['overlap'], g['n'] - g['active']) == (overlap, no_active)
    assert g['participant_overlap_with_active_unit'] == overlap / n and g['share_no_active_unit'] == no_active / n
    assert got['other']['n'] == 0                                   # no objective of kind -1 (the probes have no 'other_monster' either)
    c = report_s1.cmp_attribution(s1, ref)
    assert c['ok'], failing(c)


def test_aftermath_equals_probe(s1, ref):
    """aftermath_probe.json: the units with a winner and the winner / loser / excess rates of the 24 ten-second bins behind delta, from
    our stored tables with the probe's formula (a 240 s window per unit, the objectives that no unit claims; not the unique conv_unit).
    The rates are rounded to 4 decimals as the probe rounds them (numpy's round)."""
    got, after = report_s1.fit_aftermath_stats(s1), ref['after']
    assert got['units_with_winner'] == after['units_with_winner'] == 562_069
    for ours, theirs in (('winner_rate', 'winner_rate_per_unit_minute'), ('loser_rate', 'loser_rate_per_unit_minute'),
                         ('excess_rate', 'excess_rate_per_unit_minute')):
        assert np.round(got[ours], 4).tolist() == after[theirs], ours
    assert [round(v, 4) for v in got['excess_rate']] == after['excess_rate_per_unit_minute']
    c = report_s1.cmp_aftermath(s1, ref)
    assert c['ok'], failing(c)


def test_rows_close_to_prototype(s1, ref):
    """The rows of the FIT matches with kills (the prototype's set): the minutes 120, 180, ... s counted from our T with the prototype's
    loop give 457,441, its g2_anchor_check row count; the frame-clock rows (t0 = the real frame time) are 457,274. The 167 rows
    between them are those whose frame falls at or after T - 30 s (ruling 24a). All FIT matches: 457,445 and 457,278. In-scope FIT
    units: 574,285."""
    rows = report_s1.row_counts(s1)
    assert rows['fit_with_kills'] == {'matches': 18_957, 'nominal': 457_441, 'actual': 457_274, 'frame_late': 167}
    assert (rows['fit_all']['nominal'], rows['fit_all']['actual']) == (457_445, 457_278)
    assert ref['anchor']['rows'] == rows['fit_with_kills']['nominal'] and ref['anchor']['units_in_scope'] == 574_285
    assert report_s1.fit_in_scope_units(s1) == 574_285
    assert report_s1.fit_frame_rows(s1) == 457_274
    c = report_s1.cmp_rows(s1)
    assert c['ok'], failing(c)


def test_onsets_60s_close_to_prototype(s1, ref):
    """The shares of the FIT rows with 0 / 1 / 2 / 3+ first kills in the first 6 bins (60 s) against g2_anchor_check.json first_kill
    (the first-kill anchor, like ours) within 0.001 (ruling 25). The reference counts nominal-minute rows, ours the real frame times."""
    got, want = report_s1.fit_onset_count_share(s1, n_bins=6), ref['anchor']['first_kill']['share_rows_by_count']
    assert got['rows'] == 457_274
    for k in ('0', '1', '2', '3+'):
        assert abs(got[k] - want[k]) <= 0.001, k
    assert abs(sum(got[k] for k in ('0', '1', '2', '3+')) - 1) < 1e-12
    c = report_s1.cmp_onsets(s1, ref)
    assert c['ok'], failing(c)


def test_accounting_identity_fit(s1):
    """Every objective with a known kind and taker is attributed to a unit, converted to one or in the rest, never two of them: on the FIT
    matches both = 0 and unit + conversion + rest = all (in total and per kind). The same holds on all stored matches, and the cell
    counts of ``units/O`` and ``units/conv`` equal those of the attributed / converted objectives."""
    a = report_s1.fit_accounting(s1)
    assert a['both_unit_and_conversion'] == 0 and a['unit'] + a['conversion'] + a['rest'] == a['all'] > 0
    for name, e in a['by_kind'].items():
        assert e['both_unit_and_conversion'] == 0 and e['unit'] + e['conversion'] + e['rest'] == e['all'], name
    assert sum(e['all'] for e in a['by_kind'].values()) == a['all']
    c = report_s1.cmp_accounting(s1)
    assert c['ok'], failing(c)


def test_golden_20_unchanged(s1, golden_dir, tmp_path):
    """The golden lock: the first 20 FIT matches of 15.18 (sorted key order) are extracted again by one worker in one block and every
    array file has the SHA-256 that GOLDEN.json holds. The lock itself is checked too: it was taken from committed extraction code
    (``extraction_clean`` true, no dirty path, a 40-digit commit), and its arrays equal the first 20 matches of the S1 run, array by
    array (an 8-worker, 58-block run of the same code). Skipped until the lock is written. The re-extracted store and its block files hold
    the 20 real match keys: both folders are removed again, whatever the outcome (pytest keeps the last 3 basetemps)."""
    lock = golden_dir / 'GOLDEN.json'
    if not lock.is_file():
        pytest.skip(r'GOLDEN.json is not written yet: python -m c_ept.run_s1 --golden 20 --out <run>\golden')
    info = json.loads(lock.read_text(encoding='utf-8'))
    assert info['n_matches'] == 20 and len(info['files']) == 72
    assert info['extraction_clean'] is True and info['dirty_paths'] == [], info['dirty_paths']
    assert len(info['code_commit']) == 40 and info['extraction_modules'] == [f'c_ept/{m}.py' for m in run_s1.EXTRACTION_MODULES]
    sel = cohort.match_list('15.18', 'FIT')[:info['n_matches']]
    assert run_s1.plan_sha256(sel) == info['plan_sha256']                 # the same 20 matches (hashes only: no key in a failure message)
    store_dir, block_dir = tmp_path / 's1', tmp_path / 'blocks'
    try:
        run_s1.build_store(sel, store_dir, block_dir, workers=1, block_matches=len(sel))
        again = Store(store_dir).manifest['files']                     # the SHA-256 of every array file, as the store recorded them
    finally:
        for folder in (store_dir, block_dir):
            shutil.rmtree(folder, ignore_errors=True)
    leftovers = [folder.name for folder in (store_dir, block_dir) if folder.exists()]        # names only: the folders held real keys
    assert leftovers == []
    assert again == info['files']
    c = report_s1.cmp_golden(s1, golden_dir)
    assert c['ok'], failing(c)
