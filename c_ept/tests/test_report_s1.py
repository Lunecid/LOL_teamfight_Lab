"""Tests of c_ept.report_s1 on synthetic stores: its helpers, the extras for the S1 gate and the statistics that the golden tests apply to
the real run. Each is checked against a brute-force loop written from the definition (not from the vectorized code) or a hand-worked case.

No raw file and no prototype folder is read. Stores live under pytest's tmp_path; the random matches come from ``synth.records_from_seed``.
"""
import numpy as np
import pytest

from c_ept import config as C
from c_ept import report_s1 as R
from c_ept import run_s1
from c_ept.tests import synth
from c_ept.tests.conftest import empty_process_list

MACROS = ('FIT', 'STOP', 'DEV', 'CAL', 'AUDIT')


# ---- records ------------------------------------------------------------------------------------------------------------------------
def records_cov(key, patch):
    """T = 900,000; five single kills, 3,000 units and 30 s or more apart, so five units: f = 100,000 (out of scope), 200,000, 230,000,
    700,000 and 860,000, delta = f + 20,000. Their conversion windows (delta, delta + 60,000] cut to 120,000 .. 870,000 (T - 30,000):
    (120,000, 180,000], (220,000, 280,000], (250,000, 310,000], (720,000, 780,000] and nothing for the last (it starts at 880,000)."""
    kills = [dict(t=t, x=1000 + 3000 * i, y=1000, killer=1, victim=6) for i, t in enumerate((100_000, 200_000, 230_000, 700_000, 860_000))]
    return synth.make_record(T_ms=900_000, kills=kills, key=key, patch=patch)


def records_blind(key, patch):
    """One unit (kills at 300,000 and 305,000: people 2, 3, 4, 6, 7, 8; tau 285,000, delta 320,000) and four objectives in it; blue 1-5, red
    6-10, the team that got each objective is blue:
        dragon  killer 5, assist 7   overlap {7}: only a player of the other team
        baron   killer 2, assist 7   overlap {2, 7}: includes the other team
        herald  killer 3             overlap {3}
        tower   killer 2, assist 3   overlap {2, 3}: not a monster
    and two plates of killer 2 on the edges of the window [tau, delta): at 285,000 (= tau, attributed) and at 320,000 (= delta, not)."""
    kills = [dict(t=300_000, x=5000, y=5000, killer=2, victim=6, assists=(3,)),
             dict(t=305_000, x=5100, y=5000, killer=7, victim=4, assists=(8,))]
    objectives = [
        dict(t=310_000, type='ELITE_MONSTER_KILL', x=5100, y=5050, killer=5, assists=(7,), team=100, monster='DRAGON', sub='FIRE_DRAGON'),
        dict(t=311_000, type='ELITE_MONSTER_KILL', x=5100, y=5050, killer=2, assists=(7,), team=100, monster='BARON_NASHOR'),
        dict(t=312_000, type='ELITE_MONSTER_KILL', x=5100, y=5050, killer=3, team=100, monster='RIFTHERALD'),
        dict(t=313_000, type='BUILDING_KILL', x=5100, y=5050, killer=2, assists=(3,), team=200, building='TOWER_BUILDING',
             tower='INNER_TURRET'),
        dict(t=285_000, type='TURRET_PLATE_DESTROYED', x=5100, y=5050, killer=2, team=200),
        dict(t=320_000, type='TURRET_PLATE_DESTROYED', x=5100, y=5050, killer=2, team=200)]
    return synth.make_record(T_ms=900_000, kills=kills, objectives=objectives, key=key, patch=patch)


def records_count(key, patch):
    """Nine objectives and no kill: towers of the four tiers (the outer one twice), an inhibitor, a plate, a grub whose killerTeamId is 300
    and a baron without a team (the taker of both is unknown)."""
    where = dict(x=5000, y=5000, killer=1)
    tower = dict(type='BUILDING_KILL', team=200, building='TOWER_BUILDING', **where)
    objectives = [
        dict(t=200_000, tower='OUTER_TURRET', **tower), dict(t=210_000, tower='OUTER_TURRET', **tower),
        dict(t=220_000, tower='INNER_TURRET', **tower), dict(t=230_000, tower='BASE_TURRET', **tower),
        dict(t=240_000, tower='NEXUS_TURRET', **tower),
        dict(t=250_000, type='BUILDING_KILL', team=200, building='INHIBITOR_BUILDING', **where),
        dict(t=260_000, type='TURRET_PLATE_DESTROYED', team=200, **where),
        dict(t=270_000, type='ELITE_MONSTER_KILL', team=300, monster='HORDE', **where),
        dict(t=280_000, type='ELITE_MONSTER_KILL', monster='BARON_NASHOR', **where)]
    return synth.make_record(T_ms=600_000, objectives=objectives, key=key, patch=patch)


def records_hand(key, patch):
    hand = {'HAND_COV': records_cov, 'HAND_BLIND': records_blind, 'HAND_COUNT': records_count}
    return hand[key](key, patch) if key in hand else synth.records_from_seed(key, patch)


def build(tmp_path_factory, name, matches):
    root = tmp_path_factory.mktemp(name)
    with empty_process_list():                    # the module fixtures run before the per-test fake: E6 must not see the real processes
        run_s1.build_store(matches, root / 's1', root / 'blocks', workers=1, block_matches=9, records=records_hand,
                           role_of=synth.roles_of(matches))               # synthetic keys are not in the cohort split
    return run_s1.load_s1(root / 's1')


@pytest.fixture(scope='module')
def rand(tmp_path_factory):
    """40 random synthetic matches: the four patches and the five macros in turn."""
    return build(tmp_path_factory, 'rand', [(C.PATCHES[i % 4], f'SYN_{i:04d}', MACROS[i % 5]) for i in range(40)])


@pytest.fixture(scope='module')
def cov_store(tmp_path_factory):
    return build(tmp_path_factory, 'cov', [('15.18', 'HAND_COV', 'FIT')])


@pytest.fixture(scope='module')
def hand(tmp_path_factory):
    return build(tmp_path_factory, 'hand', [('15.18', 'HAND_BLIND', 'FIT'), ('15.19', 'HAND_COUNT', 'FIT')])


def test_the_store_builder_of_the_fixtures_does_not_see_a_running_model_server(tmp_path_factory, fake_processes):
    """Ruling 50: the module fixtures below build their stores before any per-test fake process list is in place, so the E6 check would see
    the machine's real processes, and with a model server running (the author keeps Strata on while coding) every one of them would error at
    setup. ``build`` gives the check an empty list for the length of the build. Here a server is "running" when it is called."""
    fake_processes([synth.STRATA_SERVER])
    with pytest.raises(run_s1.StrataRunning):
        run_s1.require_no_strata()                                                  # the plain check stops, as it must
    store = build(tmp_path_factory, 'busy', [('15.18', 'HAND_COV', 'FIT')])
    assert store.manifest['complete']
    with pytest.raises(run_s1.StrataRunning):
        run_s1.require_no_strata()                                                  # and the server list is back after the build


def arr(s, name):
    return np.asarray(s.array(name))


def everyone(s):
    return np.ones(len(arr(s, 'matches/T_ms')), dtype=bool)


# ---- helpers --------------------------------------------------------------------------------------------------------------------------
def test_union_length_equals_merging_the_intervals():
    rng = np.random.default_rng(3)
    for _ in range(200):
        n_matches = int(rng.integers(1, 5))
        match = rng.integers(0, n_matches, 30)
        start = rng.integers(120_000, 800_000, 30)
        end = start + rng.integers(-20_000, 60_000, 30)                          # some intervals are empty or reversed
        want = np.zeros(n_matches)
        for m in range(n_matches):
            covered, reach = 0, -1
            for a, b in sorted((int(a), int(b)) for a, b, k in zip(start, end, match) if k == m and b > a):
                covered += max(0, b - max(a, reach))
                reach = max(reach, b)
            want[m] = covered
        assert np.array_equal(R.union_length(match, start, end, n_matches + 1)[:n_matches], want)
    assert R.union_length(np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), 3).tolist() == [0, 0, 0]


@pytest.mark.parametrize('T_ms, rows', [(150_000, 0), (150_001, 1), (179_999, 1), (209_999, 1), (210_000, 1), (210_001, 2), (900_000, 13),
                                        (60_000, 0), (0, 0)])
def test_nominal_row_count_is_the_prototype_loop(T_ms, rows):
    assert R.nominal_row_count(T_ms) == rows                       # minutes 120, 180, ... s while t0 < T - 30 s (strict)


def test_check_same_array_py_ratio_and_dist():
    assert R._check(1.0005, 1.0, tol=0.001)['ok'] and not R._check(1.002, 1.0, tol=0.001)['ok'] and not R._check(1, 2)['ok']
    assert R._check([1, 2], [1, 2])['ok'] and R._check(np.int64(3), 3)['ok']
    nan = np.array([1.0, np.nan])
    assert R._same_array(nan, nan.copy())['ok'] and not R._same_array(nan, np.array([1.0, 2.0]))['ok']
    assert not R._same_array(np.arange(3), np.arange(4))['ok'] and not R._same_array(np.arange(3), np.arange(3.0), dtype=True)['ok']
    assert R._same_array(np.arange(3), np.arange(3.0))['ok']
    assert R._py({'a': (np.int64(1), np.float32(2.5)), 'b': np.arange(2)}) == {'a': [1, 2.5], 'b': [0, 1]}
    assert R._ratio(1, 4) == 0.25 and R._ratio(1, 0) is None and R._dist(np.array([])) == {'n': 0}
    d = R._dist(np.arange(101))
    assert (d['n'], d['mean'], d['p0'], d['p50'], d['p99'], d['p100']) == (101, 50.0, 0.0, 50.0, 99.0, 100.0)
    assert set(R.SIZE_NAMES) == {'T', 'S', 'ASYM', 'P'}
    got = R.size_code(np.array([4, 5, 2, 3, 2, 4, 1, 0]), np.array([4, 4, 3, 3, 4, 3, 5, 0])).tolist()
    assert got == [0, 0, 1, 1, 2, 2, 3, 3]


def test_a_missing_glyph_stops_the_figure_and_leaves_no_file(tmp_path):
    plt = R._plt()
    fig, ax = plt.subplots()
    ax.set_title('한글과 ASCII, t0, delta')
    R._save(plt, fig, tmp_path / 'ok.png')
    assert (tmp_path / 'ok.png').stat().st_size > 1000
    fig, ax = plt.subplots()
    ax.set_title('\U0001d11e')                                       # a musical symbol that Malgun Gothic does not have
    with pytest.raises(RuntimeError, match='missing from font'):
        R._save(plt, fig, tmp_path / 'bad.png')
    assert not (tmp_path / 'bad.png').exists()


def test_assert_no_keys_names_the_number_of_hits_and_never_a_key(hand):
    key = bytes(arr(hand, 'matches/key')[0]).decode('ascii')
    R.assert_no_keys(hand, 'counts 1 2 3 {"a": 4}')
    with pytest.raises(RuntimeError) as err:
        R.assert_no_keys(hand, f'text {key} text')
    assert str(err.value) == '1 match keys occur in the report text' and key not in str(err.value)


# ---- hand-worked extras ---------------------------------------------------------------------------------------------------------------
def test_conversion_coverage_of_a_hand_made_match(cov_store):
    cov = R.coverage_extras(cov_store)
    span = 900_000 - 30_000 - 120_000
    assert span == 750_000
    for group in ('all', 'fit'):
        assert cov['all_units'][group]['pooled'] == (60_000 + 90_000 + 60_000) / span == 0.28
        assert cov['in_scope_units'][group]['pooled'] == (90_000 + 60_000) / span == 0.2
        assert cov['all_units'][group]['per_match']['n'] == 1 and cov['all_units'][group]['per_match']['p50'] == 0.28
    assert cov['converted_share_of_valid']['all']['total'] is None            # no objective at all: no share, not an error


def test_team_blind_overlap_of_a_hand_made_unit(hand):
    sel = R.patch_mask(hand, '15.18')                                          # the match HAND_BLIND
    got = R.team_blind_extras(hand, sel)
    assert got['monsters'] == {'attributed': 3, 'attributed_with_empty_overlap': 0, 'overlap_only_non_taker_team': 1,
                               'share_only_non_taker_team': 1 / 3, 'overlap_includes_non_taker_team': 2,
                               'share_includes_non_taker_team': 2 / 3}
    assert got['by_kind']['dragon']['overlap_only_non_taker_team'] == 1 and got['by_kind']['baron']['overlap_includes_non_taker_team'] == 1
    assert got['by_kind']['baron']['overlap_only_non_taker_team'] == 0 and got['by_kind']['herald']['overlap_includes_non_taker_team'] == 0
    assert got['buildings_and_plates']['attributed'] == 2 and got['buildings_and_plates']['overlap_includes_non_taker_team'] == 0
    assert got['monster_credit'] == {'objectives': 3, 'credited_champions': 5, 'on_non_taker_team': 2, 'share_on_non_taker_team': 0.4}
    assert R.team_blind_extras(hand, R.patch_mask(hand, '15.19'))['monsters']['attributed'] == 0      # no unit in the other match


def test_the_attribution_window_includes_tau_and_excludes_delta(hand):
    got = R.attribution_stats(hand, R.patch_mask(hand, '15.18'))                    # the match HAND_BLIND: six objectives after 120 s
    assert (got['all']['n'], got['all']['overlap'], got['all']['active']) == (6, 5, 5)
    assert (got['plates']['n'], got['plates']['overlap'], got['plates']['active']) == (2, 1, 1)     # the plate at delta has no active unit
    assert (got['dragon']['n'], got['dragon']['overlap']) == (1, 1) and got['dragon']['no_champion'] == 0


def test_tower_tiers_and_unknown_takers_of_a_hand_made_match(hand):
    got = R.count_extras(hand, R.patch_mask(hand, '15.19'))                    # the match HAND_COUNT
    assert got['objectives'] == 9 and got['tower_tiers'] == {'outer': 2, 'inner': 1, 'base': 1, 'nexus': 1}
    assert (got['tower_without_tier'], got['tier_on_other_kinds'], got['kind_unknown']) == (0, 0, 0)
    assert got['taker_unknown']['total'] == 2 and got['taker_unknown']['share_of_objectives'] == 2 / 9
    assert {k: v for k, v in got['taker_unknown']['by_kind'].items() if v} == {'grubs': 1, 'baron': 1}


# ---- statistics against loops over the random store -----------------------------------------------------------------------------------
def cohort(nb, nr):
    lo, hi = min(nb, nr), max(nb, nr)
    return 'P' if lo < 2 else 'T' if lo >= 4 else 'S' if hi <= 3 else 'ASYM'


@pytest.mark.parametrize('which', ['fit', 'all'])
def test_unit_stats_equal_a_loop_over_the_units(rand, which):
    s = rand
    sel = R.fit_mask(s) if which == 'fit' else everyone(s)
    kill_off, unit_off = arr(s, 'matches/kill_off'), arr(s, 'matches/unit_off')
    t, x, y, unit = (arr(s, f'kills/{c}') for c in ('t', 'x', 'y', 'unit'))
    killer, victim, amask = (arr(s, f'kills/{c}') for c in ('killer', 'victim', 'amask'))
    team, T = arr(s, 'matches/team'), arr(s, 'matches/T_ms')
    n_kills = n_units = singles = 0
    spreads, shares = [], {c: 0 for c in R.SIZE_NAMES}
    in_scope = 0
    for m in np.flatnonzero(sel):
        a, b = int(kill_off[m]), int(kill_off[m + 1])
        n_kills += b - a
        for u in range(int(unit_off[m]), int(unit_off[m + 1])):
            rows = a + np.flatnonzero(unit[a:b] == u)
            n_units += 1
            singles += len(rows) == 1
            if len(rows) > 1:
                px, py = x[rows], y[rows]
                spreads.append(float(np.sqrt(((px[:, None] - px[None, :]) ** 2 + (py[:, None] - py[None, :]) ** 2).max())))
            f = int(t[rows[0]])
            if 120 <= f / 1000.0 <= float(T[m]) / 1000.0 - 30:
                people = set()
                for i in rows:
                    people |= {int(q) for q in (killer[i], victim[i]) if 1 <= q <= 10}
                    people |= {q + 1 for q in range(10) if int(amask[i]) >> q & 1}
                nb, nr = sum(team[m][q - 1] == 100 for q in people), sum(team[m][q - 1] == 200 for q in people)
                shares[cohort(nb, nr)] += 1
                in_scope += 1
    got = R.unit_stats(s, sel)
    assert (got['n_kills'], got['n_units'], got['n_in_scope']) == (n_kills, n_units, in_scope)
    assert got['units_per_kill'] == n_units / n_kills and got['share_single_kill'] == singles / n_units
    assert got['multi_kill_units'] == len(spreads) and got['spread_max'] == max(spreads)
    assert got['spread_gt_4264_n'] == sum(v > 4264.0 for v in spreads) and got['spread_p99'] == float(np.percentile(spreads, 99))
    assert got['cohort_share_in_scope'] == {c: v / in_scope for c, v in shares.items()}
    assert all(v for k, v in got.items() if k.startswith('stored_'))          # the stored columns equal their recomputed definitions


def test_attribution_stats_equal_a_loop_over_the_objectives(rand):
    s = rand
    kind_name = {'dragon': 'dragon', 'elder': 'dragon', 'grubs': 'grubs', 'herald': 'herald', 'baron': 'baron', 'atakhan': 'atakhan',
                 'tower': 'towers', 'inhibitor': 'inhibitors', 'plate': 'plates'}
    obj_off, unit_off = arr(s, 'matches/obj_off'), arr(s, 'matches/unit_off')
    ot, kind, has, unit, cred = (arr(s, f'objectives/{c}') for c in ('t', 'kind', 'has_pos', 'unit', 'cred'))
    tau, delta = arr(s, 'units/tau'), arr(s, 'units/delta')
    want = {name: [0, 0, 0, 0] for name in (*R.PROTO_KINDS, 'all')}                  # n, attributed, no champion, active
    for m in range(len(obj_off) - 1):
        for o in range(int(obj_off[m]), int(obj_off[m + 1])):
            if not has[o] or ot[o] < 120_000:
                continue
            name = kind_name[C.O_KINDS[kind[o]]]
            active = any(tau[u] <= ot[o] < delta[u] for u in range(int(unit_off[m]), int(unit_off[m + 1])))
            for key in (name, 'all'):
                for i, v in enumerate((1, unit[o] >= 0, cred[o] == 0, active)):
                    want[key][i] += int(v)
    got = R.attribution_stats(s, everyone(s))
    assert want['all'][0] > 100 and want['all'][1] > 0
    for name, (n, attributed, no_champion, active) in want.items():
        g = got[name]
        assert (g['n'], g['overlap'], g['no_champion'], g['active']) == (n, attributed, no_champion, active), name
        if n:
            assert g['participant_overlap_with_active_unit'] == attributed / n and g['share_no_active_unit'] == (n - active) / n


def test_aftermath_stats_equal_the_per_unit_loop_of_the_probe(rand):
    s = rand
    n = len(arr(s, 'matches/T_ms'))
    uo, oo, T = arr(s, 'matches/unit_off'), arr(s, 'matches/obj_off'), arr(s, 'matches/T_ms')
    in_scope, delta, K = arr(s, 'units/in_scope'), arr(s, 'units/delta'), arr(s, 'units/K').astype(np.int64)
    ot, taker, has, unit = (arr(s, f'objectives/{c}') for c in ('t', 'taker', 'has_pos', 'unit'))
    win, lose, exposure, n_units = np.zeros(24, dtype=np.int64), np.zeros(24, dtype=np.int64), np.zeros(24), 0
    for m in range(n):
        free = [(int(ot[o]), int(taker[o])) for o in range(int(oo[m]), int(oo[m + 1])) if has[o] and unit[o] < 0 and taker[o] >= 0]
        for u in range(int(uo[m]), int(uo[m + 1])):
            gold = int(K[u, 2] + K[u, 4] - K[u, 3] - K[u, 5])                        # blue minus red: bounty + shutdown bounty
            if not in_scope[u] or gold == 0:
                continue
            n_units += 1
            lo, side = int(delta[u]), 0 if gold > 0 else 1
            for k in range(24):
                exposure[k] += max(0, min(lo + 10_000 * (k + 1), int(T[m])) - (lo + 10_000 * k)) / 1000.0
            for to, tk in free:
                if lo < to <= lo + 240_000:
                    k = (to - lo - 1) // 10_000
                    if tk == side:
                        win[k] += 1
                    else:
                        lose[k] += 1
    got = R.aftermath_stats(s, everyone(s), chunk=7)                                   # a small chunk: the chunking must not matter
    assert n_units > 100 and win.sum() > 0 and lose.sum() > 0
    assert got['units_with_winner'] == n_units and got['winner_count'] == win.tolist() and got['loser_count'] == lose.tolist()
    assert np.allclose(got['exposure_s'], exposure, rtol=0, atol=1e-6) and got['bins_s'] == list(range(0, 240, 10))
    rate_w, rate_l = win / np.maximum(exposure, 1) * 60.0, lose / np.maximum(exposure, 1) * 60.0
    assert np.allclose(got['winner_rate'], rate_w) and np.allclose(got['loser_rate'], rate_l)
    assert np.allclose(got['excess_rate'], rate_w - rate_l)
    assert np.allclose(got['cumulative_excess_per_unit'], np.cumsum(win - lose) / n_units)


def test_rows_with_an_engagement_onset_shares_and_rows_per_unit_equal_loops(rand):
    s = rand
    sel = everyone(s)
    rm, t0, hi = (arr(s, f'frame_rows/{c}') for c in ('match', 't0', 'hi'))
    um, f, last, tau, delta, scope = (arr(s, f'units/{c}') for c in ('match', 'f', 'last', 'tau', 'delta', 'in_scope'))
    by_match = {m: np.flatnonzero(um == m) for m in range(len(sel))}
    inside = dict.fromkeys(('ongoing_kills', 'started_no_kill', 'inside_window', 'in_progress', 'in_progress_in_scope'), 0)
    counts = {nb: [0, 0, 0, 0] for nb in (6, 12, 18)}
    per_unit = np.zeros(len(um), dtype=int)
    for r in range(len(rm)):
        us = by_match[int(rm[r])]
        ok = scope[us]
        inside['ongoing_kills'] += bool(np.any(ok & (f[us] <= t0[r]) & (t0[r] <= last[us])))
        inside['started_no_kill'] += bool(np.any(ok & (tau[us] <= t0[r]) & (t0[r] < f[us])))
        inside['inside_window'] += bool(np.any(ok & (tau[us] <= t0[r]) & (t0[r] < delta[us])))
        inside['in_progress'] += bool(np.any((f[us] <= t0[r]) & (t0[r] < delta[us])))
        inside['in_progress_in_scope'] += bool(np.any(ok & (f[us] <= t0[r]) & (t0[r] < delta[us])))
        for nb in counts:
            counts[nb][min(int(np.sum(ok & (f[us] > t0[r]) & (f[us] <= min(t0[r] + 10_000 * nb, hi[r])))), 3)] += 1
        per_unit[us] += ok & (f[us] > t0[r]) & (f[us] <= hi[r])
    got = R.in_progress_stats(s, sel)
    assert len(rm) > 100 and all(got[k]['rows'] == len(rm) and got[k]['inside'] == v for k, v in inside.items())
    for nb, c in counts.items():
        share = R.onset_count_share(s, sel, nb)
        assert [share[k] for k in ('0', '1', '2', '3+')] == [v / len(rm) for v in c] and share['rows'] == len(rm)
    hist = np.bincount(np.minimum(per_unit[scope], 3), minlength=4)
    got = R.rows_per_unit(s, sel)
    assert got['units'] == scope.sum() and [got['count'][k] for k in ('0', '1', '2', '3+')] == hist.tolist()
    assert got['max_rows'] == per_unit[scope].max()


def test_frame_lateness_and_event_row_ages_equal_loops(rand):
    s = rand
    frame_off, ts = arr(s, 'matches/frame_off'), arr(s, 'frames/ts')
    late = R.lateness_extras(s, min_matches=1)
    n_frames = np.diff(frame_off)
    assert late['k'] == list(range(int(n_frames.max()))) and late['matches'] == [int((n_frames > k).sum()) for k in late['k']]
    assert late['median_ms'][0] == 0 and late['median_ms'][1:] == [300.0] * (len(late['k']) - 1)      # the synthetic frames are 300 ms late
    assert late['p99_ms'][1:] == [300.0] * (len(late['k']) - 1) and late['max_ms'][1:] == [300] * (len(late['k']) - 1)
    assert R.lateness_extras(s, min_matches=len(n_frames) + 1)['k'] == []
    # ages: for the first kill f of every unit in scope, f minus the last frame strictly before it; for delta the same while delta < T
    uo, um = arr(s, 'matches/unit_off'), arr(s, 'units/match')
    f, delta, scope, T = arr(s, 'units/f'), arr(s, 'units/delta'), arr(s, 'units/in_scope'), arr(s, 'matches/T_ms')
    want = {'M3': [], 'M4': []}
    for u in np.flatnonzero(scope):
        m = int(um[u])
        frames = ts[int(frame_off[m]):int(frame_off[m + 1])]
        for name, ref in (('M3', int(f[u])), ('M4', int(delta[u]))):
            if name == 'M3' or ref < T[m]:
                want[name].append(ref - int(frames[frames < ref][-1]))
    got = R.age_extras(s, everyone(s))
    for name, ages in want.items():
        ages = np.array(ages)
        assert got[name]['rows'] == len(ages) > 50 and got[name]['age_ms']['p50'] == float(np.median(ages))
        assert got[name]['age_ms']['p100'] == float(ages.max()) and got[name]['over_61s'] == int((ages >= 61_000).sum())
        assert got[name]['hist_1s'] == np.bincount(np.minimum(ages // 1000, 61), minlength=62)[:61].tolist()
    assert uo[-1] == len(um)
