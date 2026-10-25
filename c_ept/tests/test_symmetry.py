"""Tests of ``c_ept.symmetry``: the exact team swap of a view (the 180-degree rotation of the map with the two teams exchanged).

Besides the brief's tests, a mirrored match is built here from a match: the players take the places [5-9, 0-4], every position
(x, y) becomes (14,875 - x, 14,875 - y), the team ids and the lanes (top and bottom change places under a rotation) are exchanged in
the events, and the view of that match must equal the swap of the view of the match, array by array, for everything the swap carries.
The wall fraction and the geodesics are the documented exceptions (the terrain is not mirror-exact). The landmark rows and the building
columns of ``geo_pl`` stay in place under ``swap_view``; they equal the mirrored view once ``symmetry.LANDMARK_SWAP`` is applied (the
landmarks exactly, ``geo_pl`` within three cells). The golden tests run only with ``C_EPT_S1_RUN`` set.
Nothing here edits ``state.LANDMARKS`` (R12-m): mirrored positions are new arrays and a changed landmark list is a list of ``dict(entry)``
copies. Run with ``--tb=short``.
"""
import copy
import dataclasses
import math

import numpy as np
import pytest

from c_ept import config as C
from c_ept.domain import state, symmetry, targets
from c_ept.preprocessing import terrain, tokens
from c_ept.domain import units
from c_ept.tests import synth
from c_ept.tests.view_helpers import (K, NAN, P, REF, differing, frame_at, frames_with, raw_building, raw_elite, raw_item, raw_kill,
                                      raw_level, record, rich_events, views_equal)

W = C.W
PERM = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]                       # the player order of a swapped view
LANE_SWAP = {'TOP_LANE': 'BOT_LANE', 'BOT_LANE': 'TOP_LANE', 'MID_LANE': 'MID_LANE'}


# ---- a mirrored match ---------------------------------------------------------------------------------------------------------------
def mirror_pid(p):
    """The id, in the mirrored match, of the player with id ``p`` (0 and anything else that is no player stays as it is)."""
    return (p - 1 + 5) % 10 + 1 if type(p) is int and 1 <= p <= 10 else p


def mirror_event(e):
    e = copy.deepcopy(e)
    for name in ('killerId', 'victimId', 'participantId', 'creatorId'):
        if name in e:
            e[name] = mirror_pid(e[name])
    if 'assistingParticipantIds' in e:
        e['assistingParticipantIds'] = [mirror_pid(a) for a in e['assistingParticipantIds']]
    for name in ('teamId', 'killerTeamId', 'winningTeam'):
        if e.get(name) in (100, 200):
            e[name] = 300 - e[name]
    if e.get('laneType') in LANE_SWAP:
        e['laneType'] = LANE_SWAP[e['laneType']]
    if isinstance(e.get('position'), dict):
        e['position'] = {'x': W - e['position']['x'], 'y': W - e['position']['y']}
    for name in ('victimDamageReceived', 'victimDamageDealt'):
        for d in e.get(name) or ():
            if isinstance(d, dict) and 'participantId' in d:
                d['participantId'] = mirror_pid(d['participantId'])
    return e


def mirror_record(rec):
    """The match seen from the other side: players [5-9, 0-4], positions rotated, teams and lanes exchanged. The team array keeps its
    layout (the first five players are team 100), so the former red team plays as 100."""
    frames = np.array(rec.frame_num)[:, PERM, :]
    frames[..., P['x']], frames[..., P['y']] = W - frames[..., P['x']], W - frames[..., P['y']]
    return dataclasses.replace(
        rec, frame_num=frames, end_xy=W - np.array(rec.end_xy)[PERM], champ=np.array(rec.champ)[PERM],
        runes_by_pid={str(q + 1): rec.runes_by_pid[str(PERM[q] + 1)] for q in range(10)}, events=tuple(mirror_event(e) for e in rec.events))


# what a swap carries exactly: everything except the terrain-derived arrays (wall fraction, geodesics) and the landmarks, which stay in
# place under swap_view (the landmark rows and the building columns of geo_pl follow symmetry.LANDMARK_SWAP: see below)
CARRIED = ('players_zone', 'ev_type', 'ev_dt', 'ev_team', 'ev_xy', 'ev_actors', 'ev_gold', 'ev_n', 'ev_dropped', 'ed_i', 'ed_j', 'ed_dt',
           'ed_kind', 'ed_n', 'ed_dropped', 'summary')
LM = [dict(e) for e in state.LANDMARKS]               # plain copies of the 34 landmarks: LANDMARKS itself is only read (R12-m)
N_BUILDINGS = 28                                      # the first 28 landmark rows are the buildings, then the six nests
# geo_pl against the mirrored match, in map units. The columns of the buildings only: a nest column is the distance to the pit itself,
# while the rotated pit is another one (the dragon pit lies where the baron pit does). Three cells (200 u each): one for the 125 u by
# which the rotation constant of the swap (14,875) differs from the one that maps the 75 x 75 grid onto itself (15,000), one for the
# partner that lies up to 133 u from the rotated building, one for the snapping of both ends. Measured on the open grid: 483 u at most
# over 400,000 random positions and the 28 columns; with the rows in place (no permutation) the gap is 14,000-16,000 u.
GEO_PL_TOLERANCE = 3 * terrain.CELL


def geo_pl_gap(swapped, mirrored):
    """The largest difference between the building columns of two ``geo_pl`` arrays, in map units; the cells that are NaN must be the
    same cells."""
    a, b = (np.asarray(g)[:, :N_BUILDINGS].astype(float) for g in (swapped, mirrored))
    assert np.array_equal(np.isnan(a), np.isnan(b))
    return float(np.nanmax(np.abs(a - b))) if (~np.isnan(a)).any() else 0.0


def assert_swap_commutes(terr, rec, ref, k):
    """swap_view(view of the match) equals the view of the mirrored match in every array that the swap carries, and in the players block
    except the wall fraction (the 75 x 75 grid spans [0, 15,000], the swap constant is 14,875: the terrain is not mirror-exact). With
    ``LANDMARK_SWAP`` applied to the rows, the landmarks equal the mirrored match's exactly and the building columns of ``geo_pl`` to
    within ``GEO_PL_TOLERANCE``."""
    a = symmetry.swap_view(tokens.build_view(tokens.MatchContext(rec, terr), ref, k))
    b = tokens.build_view(tokens.MatchContext(mirror_record(rec), terr), ref, k)
    columns = [i for i in range(len(tokens.PLAYER_NUM)) if i != P['wall_fraction']]
    assert np.array_equal(a['players_num'][:, columns], b['players_num'][:, columns], equal_nan=True)
    wrong = [key for key in CARRIED if not np.array_equal(a[key], b[key], equal_nan=True)]
    assert not wrong, wrong
    swap = symmetry.LANDMARK_SWAP
    assert np.array_equal(a['landmarks'][swap], b['landmarks'], equal_nan=True)
    gap = geo_pl_gap(a['geo_pl'][:, swap], b['geo_pl'])
    assert gap <= GEO_PL_TOLERANCE, gap


def rotated_distance(i, j):
    """The distance, in map units, from the rotation (W - x, W - y) of landmark ``i`` to landmark ``j``."""
    return math.hypot(LM[j]['x'] - (W - LM[i]['x']), LM[j]['y'] - (W - LM[i]['y']))


def partner_of(i):
    """The building nearest to the rotation of building ``i``: found from the positions alone, without ``LANDMARK_SWAP``."""
    return min((k for k in range(N_BUILDINGS) if k != i), key=lambda k: rotated_distance(i, k))


def siege_events(ref=REF, step=22_000):
    """BUILDING_KILL events for the 14 pairs of buildings that the rotation exchanges (found with ``partner_of``), in a mix that makes the
    states of the two buildings of a pair differ: of every third pair the lower row falls, of the next the higher row, of the third both
    (the second 40 s after the first). The pairs fall from 3 s to 5 min before ``ref``, at the building's own place moved by (+40, -30) u;
    the nexus turrets have a position, so the state module takes the nearest one that stands. An inhibitor killed more than 5 min before
    ``ref`` is back by then; a turret stays down."""
    pairs = sorted({tuple(sorted((i, partner_of(i)))) for i in range(N_BUILDINGS)})
    assert len(pairs) == 14
    out = []
    for q, (lo, hi) in enumerate(pairs):
        t = ref - 3_000 - step * q
        for row, when in {0: [(lo, t)], 1: [(hi, t)], 2: [(lo, t), (hi, t + 40_000)]}[q % 3]:
            e = LM[row]
            building = 'INHIBITOR_BUILDING' if e['kind'] == 'inhibitor' else 'TOWER_BUILDING'
            out.append(raw_building(when, e['team'], e['lane'], building, e['tower'], killer=1, x=e['x'] + 40, y=e['y'] - 30))
    return out


# ---- the views to swap ----------------------------------------------------------------------------------------------------------------
@pytest.fixture(params=['rich_event_clock', 'rich_frame_row', 'missing_position', 'golden'])
def any_view(request, open_terrain):
    if request.param == 'golden':
        rec = request.getfixturevalue('golden_records')[0]               # skips the test without C_EPT_S1_RUN
        k = len(rec.frame_ts) // 2
        return tokens.build_view(tokens.MatchContext(rec, open_terrain), int(rec.frame_ts[k]) + 7_000, k)
    rec = record(rich_events())
    if request.param == 'missing_position':
        rec = frames_with(rec, {(K, 2, 'x'): NAN, (K, 2, 'y'): NAN, (K - 1, 7, 'x'): NAN, (K - 1, 7, 'y'): NAN})
    ref = 600_300 if request.param == 'rich_frame_row' else REF
    return tokens.build_view(tokens.MatchContext(rec, open_terrain), ref, K)


# ---- the brief's tests ------------------------------------------------------------------------------------------------------------
def test_swap_involution(any_view):
    assert views_equal(symmetry.swap_view(symmetry.swap_view(any_view)), any_view)       # NaN equal


def test_swap_rotates_and_reorders(any_view):
    s = symmetry.swap_view(any_view)
    x = tokens.PLAYER_NUM.index('x')
    assert np.allclose(s['players_num'][0, x], 14_875 - any_view['players_num'][5, x])


def test_swap_marks():
    assert symmetry.swap_mark(0) == 25 and symmetry.swap_mark(49) == 24 and symmetry.swap_mark(50) == 51
    m = np.arange(52)
    assert np.array_equal(symmetry.swap_mark(symmetry.swap_mark(m)), m)


def test_swap_targets():
    K = np.arange(8)
    assert list(symmetry.swap_K(K)) == [1, 0, 3, 2, 5, 4, 7, 6]
    O = np.arange(18)                                                                          # noqa: E741 (the brief's names)
    assert np.array_equal(symmetry.swap_O(symmetry.swap_O(O)), O) and symmetry.swap_O(O)[0] == 1


def test_obj_swap_pairs_blue_red():
    n = state.OBJ_STATE_NAMES
    p = symmetry.OBJ_SWAP
    assert n[p[n.index('blue_inhibs_down')]] == 'red_inhibs_down' and n[p[n.index('dragon_up')]] == 'dragon_up'


# ---- the marks and the targets --------------------------------------------------------------------------------------------------------
def test_swap_mark_is_the_first_kill_mark_of_the_mirrored_kill():
    """The mark of a first kill is (killer team, killer slot, victim slot) or an execution mark: the mirrored kill has the other team and
    the same slots, and an execution of a blue victim becomes the execution of a red one."""
    for killer in range(0, 11):
        for victim in range(1, 11):
            swapped = units.first_kill_mark(mirror_pid(killer), mirror_pid(victim))
            assert symmetry.swap_mark(units.first_kill_mark(killer, victim)) == swapped


def test_swap_mark_types_and_range():
    assert type(symmetry.swap_mark(7)) is int and type(symmetry.swap_mark(np.int64(7))) is int
    for dtype in (np.int8, np.int16, np.int64):
        out = symmetry.swap_mark(np.array([0, 24, 25, 49, 50, 51], dtype=dtype))
        assert out.dtype == dtype and list(out) == [25, 49, 0, 24, 51, 50]
    assert symmetry.swap_mark(np.arange(52).reshape(4, 13)).shape == (4, 13)
    for bad in (-1, 52, np.array([3, 52])):
        with pytest.raises(ValueError):
            symmetry.swap_mark(bad)
    for bad in (1.0, True, 'a', None, np.array([1.0])):
        with pytest.raises(TypeError):
            symmetry.swap_mark(bad)


def test_swap_K_is_the_kill_gold_of_the_mirrored_kills():
    """K counts, per team of the killer, kills, bounty, shutdown and assisters: the mirrored kills score for the other team."""
    team = synth.TEAM
    rows = [(1_000, 100, 1, 6, (2, 3), 300, 50), (2_000, 150, 7, 2, (8,), 200, 0), (3_000, 200, 4, 9, (), 150, 25)]
    mirrored = [(t, x, mirror_pid(k), mirror_pid(v), tuple(mirror_pid(a) for a in assists), b, sd) for t, x, k, v, assists, b, sd in rows]
    a, b = synth.kt(rows), synth.kt(mirrored)
    comp = np.arange(3)
    got, want = targets.unit_K(a, comp, team), targets.unit_K(b, comp, team)
    assert got.tolist() != want.tolist() and np.array_equal(symmetry.swap_K(got), want)
    assert [f[:-2] for f in targets.K_FIELDS[0::2]] == [f[:-2] for f in targets.K_FIELDS[1::2]]               # the pairs are (x_b, x_r)
    assert all(f.endswith('_b') for f in targets.K_FIELDS[0::2]) and all(f.endswith('_r') for f in targets.K_FIELDS[1::2])


def test_swap_O_flips_the_taker_of_every_kind():
    assert targets.N_CELLS == 2 * len(C.O_KINDS) == 18
    counts = np.arange(18)
    out = symmetry.swap_O(counts)
    for kind in range(len(C.O_KINDS)):
        assert out[2 * kind] == 2 * kind + 1 and out[2 * kind + 1] == 2 * kind                                  # cell = 2 * kind + taker


def test_swap_K_and_O_take_batches_and_refuse_other_shapes():
    assert symmetry.swap_K(np.arange(24).reshape(3, 8)).shape == (3, 8) and symmetry.swap_O(np.zeros((2, 5, 18))).shape == (2, 5, 18)
    assert symmetry.swap_K([1, 2, 3, 4, 5, 6, 7, 8]).tolist() == [2, 1, 4, 3, 6, 5, 8, 7]
    for fn, n in ((symmetry.swap_K, 8), (symmetry.swap_O, 18)):
        with pytest.raises(ValueError):
            fn(np.zeros(n + 1))
        with pytest.raises(ValueError):
            fn(np.zeros((n, 2)))
    a = np.arange(8)
    out = symmetry.swap_K(a)
    out[0] = 99
    assert a[0] == 0                                                                                            # a new array


# ---- OBJ_SWAP ----------------------------------------------------------------------------------------------------------------------
def test_obj_swap_is_a_permutation_that_is_its_own_inverse():
    p = symmetry.OBJ_SWAP
    assert p.shape == (89,) and p.dtype.kind == 'i' and sorted(p.tolist()) == list(range(89)) and np.array_equal(p[p], np.arange(89))
    assert not p.flags.writeable


def test_obj_swap_exchanges_the_teams_and_crosses_the_lanes():
    n = list(state.OBJ_STATE_NAMES)
    partner = lambda name: n[symmetry.OBJ_SWAP[n.index(name)]]                          # noqa: E731
    assert partner('blue_inhib_top_respawn_s') == 'red_inhib_bot_respawn_s'           # the rotation turns the top lane into the bottom lane
    assert partner('red_inhib_top_respawn_s') == 'blue_inhib_bot_respawn_s'
    assert partner('blue_inhib_mid_respawn_s') == 'red_inhib_mid_respawn_s'
    assert partner('blue_plates_bot') == 'red_plates_top' and partner('red_plates_mid') == 'blue_plates_mid'
    for name in ('towers_outer', 'towers_nexus', 'baron_buff_s', 'elder_buff_n', 'dragons', 'dragon_hextech', 'has_soul', 'voidgrubs',
                 'nexus_turret_respawn_max_s', 'inhibs_down', 'atakhans', 'heralds', 'barons', 'elders'):
        assert partner('blue_' + name) == 'red_' + name and partner('red_' + name) == 'blue_' + name


def test_obj_swap_leaves_what_belongs_to_no_team_in_place():
    n = list(state.OBJ_STATE_NAMES)
    fixed = [name for name in n if not name.startswith(('blue_', 'red_'))]
    assert sum(1 for name in n if name.startswith('blue_')) == sum(1 for name in n if name.startswith('red_')) == 29
    assert len(fixed) == 89 - 2 * 29 == 31 and 't_s' in fixed and 'dragon_next_spawn_s' in fixed and 'rift_element_fire' in fixed
    assert all(n[symmetry.OBJ_SWAP[n.index(name)]] == name for name in fixed)                 # nests and map-level flags stay what they are
    assert [i for i in range(89) if symmetry.OBJ_SWAP[i] == i] == [n.index(name) for name in fixed]


# ---- LANDMARK_SWAP: the rows of the 34 landmarks (final review M8) ----------------------------------------------------------------------
def test_landmark_swap_is_a_read_only_permutation_that_is_its_own_inverse():
    p = symmetry.LANDMARK_SWAP
    assert tokens.N_LANDMARKS == len(LM) == 34
    assert p.shape == (34,) and p.dtype.kind == 'i' and not p.flags.writeable
    assert sorted(p.tolist()) == list(range(34)) and np.array_equal(p[p], np.arange(34))


def test_landmark_swap_exchanges_the_teams_crosses_the_lanes_and_keeps_the_kind_of_building():
    p = symmetry.LANDMARK_SWAP
    for i in range(N_BUILDINGS):
        a, b = LM[i], LM[p[i]]
        assert {a['team'], b['team']} == {100, 200}, i                                       # the other team's building ...
        assert (a['kind'], a['building'], a['tower']) == (b['kind'], b['building'], b['tower']), i       # ... of the same kind and tier
        assert b['lane'] == LANE_SWAP[a['lane']], i                       # ... in the lane that the rotation turns this one into
    assert not any(p[i] == i for i in range(N_BUILDINGS))                                      # no building is its own partner
    assert sum(LM[p[i]]['lane'] != LM[i]['lane'] for i in range(N_BUILDINGS)) == 16   # the 16 buildings of the top and bottom lanes cross


def test_landmark_swap_leaves_the_nests_in_their_rows():
    """A nest belongs to no team, and the dragon pit does not become a baron pit: the rotation puts the dragon pit on the baron pit (31 u)
    and each of the others on a place that is not its own (at least 7,400 u from it), so the nest rows are the one place where
    the rotated position is not the position of the same row. The loader flags that with the side marker (spec 5.2). The two Atakhan
    places do rotate onto each other (235 u), but their two rows are always one state (the mirrored-match test below), so exchanging
    them would not show in ``landmarks``: the nests stay, as the brief says."""
    p = symmetry.LANDMARK_SWAP
    assert [int(p[i]) for i in range(N_BUILDINGS, 34)] == list(range(N_BUILDINGS, 34))
    assert all(LM[i]['team'] is None and rotated_distance(i, i) > 7_000 for i in range(N_BUILDINGS, 34))
    assert rotated_distance(28, 29) < 40                                                       # dragon -> baron
    d = rotated_distance(32, 33)
    assert 230 < d < 240 and d == pytest.approx(rotated_distance(33, 32))                      # Atakhan place 0 -> place 1


def test_landmark_swap_pairs_a_nexus_turret_with_the_one_at_its_rotated_position():
    """The two nexus turrets of a team are told apart by their slot, and the rotation exchanges the slots (the turret that stands nearer
    the mid lane of one team lies where the other team's farther one does): a pairing by slot or by row would be wrong."""
    nexus = [i for i in range(N_BUILDINGS) if LM[i]['tower'] == 'NEXUS_TURRET']
    assert len(nexus) == 4
    p = symmetry.LANDMARK_SWAP
    for i in nexus:
        mates = [k for k in nexus if LM[k]['team'] != LM[i]['team']]
        assert p[i] == min(mates, key=lambda k: rotated_distance(i, k))
        assert LM[p[i]]['slot'] != LM[i]['slot']
        assert rotated_distance(i, p[i]) < 90 < 570 < rotated_distance(i, next(k for k in mates if k != p[i]))


def test_landmark_swap_partners_lie_within_133_units_of_the_rotated_building_and_nothing_else_is_near():
    """The partner is the building nearest to the rotated one, of all 28; the next one is at least 4 times as far (5.1 at the least)."""
    p = symmetry.LANDMARK_SWAP
    worst = 0.0
    for i in range(N_BUILDINGS):
        d = rotated_distance(i, p[i])
        others = [rotated_distance(i, k) for k in range(N_BUILDINGS) if k != p[i]]
        assert d < 134 and min(others) > 4 * d, i
        worst = max(worst, d)
    assert symmetry.LANDMARK_SWAP_OFFSET == pytest.approx(worst) and 133 < worst < 134


def landmarks_with(**edits):
    """Plain copies of the landmarks with ``edits`` ({row: {key: value}}) applied: the input of ``derive_landmark_swap`` for a test."""
    out = [dict(e) for e in state.LANDMARKS]
    for row, change in edits.items():
        out[int(row.lstrip('r'))].update(change)
    return out


def test_the_derivation_gives_the_constant_and_follows_the_positions_not_the_slots():
    perm, offset = symmetry.derive_landmark_swap(state.LANDMARKS)
    assert np.array_equal(perm, symmetry.LANDMARK_SWAP) and offset == symmetry.LANDMARK_SWAP_OFFSET
    assert perm[7] == 22 and perm[8] == 21                                                       # blue slot 0 <-> red slot 1
    a, b = LM[21], LM[22]                                                                      # the red nexus turrets, rows 21 and 22
    turned = landmarks_with(r21=dict(x=b['x'], y=b['y']), r22=dict(x=a['x'], y=a['y']))       # put the two the other way round
    perm, _ = symmetry.derive_landmark_swap(turned)
    assert perm[7] == 21 and perm[8] == 22 and perm[21] == 7 and perm[22] == 8                  # the pairing follows the places


def test_the_derivation_refuses_landmarks_that_do_not_rotate_onto_each_other():
    two_on_one = landmarks_with(r8=dict(x=LM[7]['x'] + 1, y=LM[7]['y']))        # both blue nexus turrets now rotate to the same place
    with pytest.raises(ValueError, match='its own inverse'):
        symmetry.derive_landmark_swap(two_on_one)
    wrong_lane = landmarks_with(r0=dict(lane='TOP_LANE'), r10=dict(lane='BOT_LANE'))       # the labels of two inhibitors exchanged
    with pytest.raises(ValueError, match='lane'):
        symmetry.derive_landmark_swap(wrong_lane)
    far = landmarks_with(r24=dict(x=LM[24]['x'] + 400))                                 # an inhibitor that is 400 u off its rotation
    with pytest.raises(ValueError, match='more than 200 u'):
        symmetry.derive_landmark_swap(far)
    no_partner = [dict(e) for e in LM if not (e['team'] == 200 and e['kind'] == 'inhibitor')]       # blue inhibitors, no red ones
    with pytest.raises(ValueError, match='no partner'):
        symmetry.derive_landmark_swap(no_partner)


def rows_differing(x, y):
    """The rows in which two (34, 2) landmark arrays differ (NaN equals NaN)."""
    same = (x == y) | (np.isnan(x) & np.isnan(y))
    return np.flatnonzero(~same.all(axis=1))


@pytest.mark.parametrize('ref, on_timers', [(REF, True), (800_000, True), (1_100_000, False)])
def test_landmark_swap_gives_the_landmark_state_of_the_mirrored_match(open_terrain, ref, on_timers):
    """Every building is killed once, on both teams, in every lane and tier: the swapped view with its landmark rows (and the building
    columns of geo_pl) permuted by LANDMARK_SWAP is the view of the mirrored match, and with the rows in place it is not."""
    rec = record(rich_events() + siege_events())
    k = frame_at(rec, ref)
    a = symmetry.swap_view(tokens.build_view(tokens.MatchContext(rec, open_terrain), ref, k))
    b = tokens.build_view(tokens.MatchContext(mirror_record(rec), open_terrain), ref, k)
    down = b['landmarks'][:N_BUILDINGS, 0] == 0
    assert down.sum() >= 12 and {LM[i]['team'] for i in np.flatnonzero(down)} == {100, 200}      # a view with many buildings down
    assert (b['landmarks'][:N_BUILDINGS, 1] > 0).any() == on_timers                              # inhibitors on their timers, or all back
    assert np.array_equal(b['landmarks'][32], b['landmarks'][33], equal_nan=True)                # ... but the Atakhan rows hold one state
    assert len(rows_differing(a['landmarks'], b['landmarks'])) >= 8                              # in place, many rows are the wrong ones
    assert geo_pl_gap(a['geo_pl'], b['geo_pl']) > 5_000                                         # and so are the columns of geo_pl
    assert_swap_commutes(open_terrain, rec, ref, k)                                              # permuted: equal, geo_pl within 3 cells


def test_a_nexus_turret_kill_without_a_position_is_the_one_case_that_the_permutation_does_not_cover(open_terrain):
    """``state.building_kills`` takes, without a position, the standing nexus turret with the lowest slot, and a slot is not
    rotation-covariant: the mirrored match loses slot 0 of the other team, which LANDMARK_SWAP pairs with slot 1. So the two nexus rows
    of that team differ and nothing else does. Real BUILDING_KILL events all have a position (0 of 311,622 in the S1 matches, Task 10)."""
    events = rich_events() + [raw_building(REF - 20_000, 100, 'MID_LANE', 'TOWER_BUILDING', 'NEXUS_TURRET', killer=6, x=None, y=None)]
    rec = record(events)
    a = symmetry.swap_view(tokens.build_view(tokens.MatchContext(rec, open_terrain), REF, K))
    b = tokens.build_view(tokens.MatchContext(mirror_record(rec), open_terrain), REF, K)
    assert rows_differing(a['landmarks'][symmetry.LANDMARK_SWAP], b['landmarks']).tolist() == [21, 22]


# ---- swap_view, array by array -----------------------------------------------------------------------------------------------------
def test_swap_view_players_block(any_view):
    v, s = any_view, symmetry.swap_view(any_view)
    expect = v['players_num'][PERM].copy()
    for name in ('x', 'y', 'prev_x', 'prev_y'):
        expect[:, P[name]] = W - expect[:, P[name]]
    assert np.array_equal(s['players_num'], expect, equal_nan=True) and s['players_num'].dtype == v['players_num'].dtype
    assert np.array_equal(s['players_num'][:, P['has_prev']], v['players_num'][PERM, P['has_prev']])


def test_swap_view_zone_follows_the_rotated_position_and_minus_one_stays_minus_one(any_view, open_terrain):
    """R12-n: ZONE_SWAP has ten entries; a missing position (-1) must not be looked up in it."""
    s = symmetry.swap_view(any_view)
    z = any_view['players_zone']
    expect = np.where(z >= 0, terrain.ZONE_SWAP[np.where(z >= 0, z, 0)], -1)[PERM]
    assert np.array_equal(s['players_zone'], expect) and s['players_zone'].dtype == np.int8
    known = np.isfinite(s['players_num'][:, [P['x'], P['y']]]).all(axis=1)
    assert np.array_equal(s['players_zone'][known], open_terrain.zone(s['players_num'][known, P['x']], s['players_num'][known, P['y']]))
    assert np.array_equal(s['players_zone'] == -1, ~known)
    assert terrain.ZONE_SWAP[-1] != -1                                                  # the trap: index -1 is the last zone, not "missing"


def test_a_missing_position_keeps_zone_minus_one_through_the_swap(open_terrain):
    rec = frames_with(record(rich_events()), {(K, 2, 'x'): NAN, (K, 2, 'y'): NAN})
    v = tokens.build_view(tokens.MatchContext(rec, open_terrain), REF, K)
    s = symmetry.swap_view(v)
    assert v['players_zone'][2] == -1 and s['players_zone'][7] == -1 and list(s['players_zone']).count(-1) == 1
    assert np.isnan(s['players_num'][7, [P['x'], P['y']]]).all() and np.isnan(s['geo_pp'][7]).all() and np.isnan(s['geo_pp'][:, 7]).all()


def test_swap_view_events_and_edges(any_view):
    v, s = any_view, symmetry.swap_view(any_view)
    assert np.array_equal(s['ev_team'], np.where(v['ev_team'] == 0, 1, np.where(v['ev_team'] == 1, 0, v['ev_team'])))
    assert np.array_equal(s['ev_xy'], W - v['ev_xy'], equal_nan=True)
    for key in ('ev_type', 'ev_dt', 'ev_gold', 'ev_n', 'ev_dropped', 'ed_dt', 'ed_kind', 'ed_n', 'ed_dropped', 'landmarks'):
        assert np.array_equal(s[key], v[key], equal_nan=True) and s[key].dtype == v[key].dtype, key
    for key in ('ed_i', 'ed_j'):
        assert np.array_equal(s[key], np.where(v[key] >= 0, (v[key] + 5) % 10, -1)) and s[key].dtype == np.int8
    assert np.array_equal(s['ev_actors'], symmetry.swap_actors(v['ev_actors'])) and s['ev_actors'].dtype == np.int16


def test_swap_actors_exchanges_the_low_and_the_high_five_bits_for_every_mask():
    masks = np.arange(1024, dtype=np.int16)
    out = symmetry.swap_actors(masks)
    assert out.dtype == np.int16 and np.array_equal(symmetry.swap_actors(out), masks) and sorted(out.tolist()) == list(range(1024))
    for pid in range(1, 11):
        assert symmetry.swap_actors(np.array([1 << (pid - 1)], dtype=np.int16))[0] == 1 << (mirror_pid(pid) - 1)
    assert symmetry.swap_actors(np.array([0], dtype=np.int16))[0] == 0


def test_swap_participants_adds_five_and_keeps_minus_one():
    p = np.array([-1, 0, 1, 4, 5, 9], dtype=np.int8)
    assert symmetry.swap_participants(p).tolist() == [-1, 5, 6, 9, 0, 4] and symmetry.swap_participants(p).dtype == np.int8


def test_swap_view_summary(any_view):
    v, s = any_view, symmetry.swap_view(any_view)
    assert np.array_equal(s['summary'][:2], v['summary'][:2])
    for i in range(8):                                                                       # old_<type>_b and old_<type>_r change places
        assert s['summary'][2 + 2 * i] == v['summary'][3 + 2 * i] and s['summary'][3 + 2 * i] == v['summary'][2 + 2 * i]
    assert np.array_equal(s['summary'][18:], v['summary'][18:][symmetry.OBJ_SWAP], equal_nan=True) and s['summary'].dtype == np.float32


def test_swap_view_distances(any_view):
    v, s = any_view, symmetry.swap_view(any_view)
    assert np.array_equal(s['geo_pp'], v['geo_pp'][np.ix_(PERM, PERM)], equal_nan=True)
    assert np.array_equal(s['geo_pl'], v['geo_pl'][PERM], equal_nan=True)                    # the landmark columns stay: the landmarks stay
    assert s['geo_pp'].dtype == s['geo_pl'].dtype == np.float16


def test_swap_view_makes_new_arrays_and_leaves_its_argument_alone(any_view):
    before = {k: a.copy() for k, a in any_view.items()}
    s = symmetry.swap_view(any_view)
    assert list(s) == list(any_view) and views_equal(any_view, before)
    for key, arr in s.items():
        assert not np.shares_memory(arr, any_view[key]), key
        arr[...] = 0
    assert views_equal(any_view, before)


def test_swap_view_refuses_a_view_it_cannot_swap_completely(any_view):
    extra = dict(any_view, frame_row=np.array(3))
    with pytest.raises(ValueError, match='frame_row'):
        symmetry.swap_view(extra)                                                             # no silent pass-through of an unknown array
    missing = {k: a for k, a in any_view.items() if k != 'geo_pl'}
    with pytest.raises(ValueError, match='geo_pl'):
        symmetry.swap_view(missing)
    wrong = dict(any_view, geo_pp=any_view['geo_pp'][:5])
    with pytest.raises(ValueError, match='geo_pp'):
        symmetry.swap_view(wrong)


def test_the_docstring_says_what_the_loader_still_has_to_do():
    """R12-i and the final review (M8): the landmark rows and the landmark columns of geo_pl stay in place; the loader permutes them with
    LANDMARK_SWAP, or gives the landmarks no per-row identity and mirrors ownership and positions; the learned terrain field (G6c) is
    sampled at the physical position."""
    doc = ' '.join(symmetry.swap_view.__doc__.split())                                    # one line: a phrase may break across lines
    low = doc.lower()
    assert 'landmark' in low and 'ownership' in low and 'loader' in low and 'position' in low and 'physically consistent' in low
    assert 'LANDMARK_SWAP' in doc and 'no per-row identity' in low and 'geo_pl[:, LANDMARK_SWAP]' in doc
    assert 'G6c' in doc and 'physical position' in low and 'W - x' in doc
    assert 'LANDMARK_SWAP' in symmetry.__doc__                                              # the module text names it next to OBJ_SWAP


# ---- the swap against a mirrored match ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('which', ['rich', 'seed_11', 'seed_12'])
def test_swap_view_equals_the_view_of_the_mirrored_match(open_terrain, which):
    extra = [raw_kill(300_300, 2, 7, assists=[1, 3], x=6100, y=6050, dealers=[1, 3, 9]), raw_level(150_000, 8, 4),
             raw_item(310_000, 9, 3006)]
    rec = record(rich_events() + extra) if which == 'rich' else synth.records_from_seed('SYN_' + which[5:], '15.19')
    ks = [3, 8, 14] if which != 'rich' else [3, K, 19]
    for k in ks:
        t0 = int(rec.frame_ts[k])
        for ref in (t0, t0 + 9_999, t0 + 33_000):
            assert_swap_commutes(open_terrain, rec, ref, k)


def test_swap_view_equals_the_view_of_the_mirrored_match_when_the_edge_cap_cuts_inside_an_event(open_terrain):
    """Eight kills fill 120 edges; the oldest event is an objective whose assisters stand on both teams (10 pairs), so the 128th edge falls
    inside its pair list: the pairs that are kept, and the count that is cut, must be the same in the mirrored match."""
    kills = [raw_kill(REF - 1000 * (i + 1), 1, 6, assists=[2, 3, 4, 5], x=6000, y=6000) for i in range(8)]
    elite = raw_elite(REF - 20_000, 1, 100, assists=[7, 3, 8, 2])                              # killer 1, blue 2 and 3, red 7 and 8
    rec = record(kills + [elite])
    v = tokens.build_view(tokens.MatchContext(rec, open_terrain), REF, K)
    assert (v['ed_n'], v['ed_dropped']) == (128, 2)
    assert_swap_commutes(open_terrain, rec, REF, K)


def test_the_mirror_of_the_mirror_is_the_match():
    rec = record(rich_events())
    twice = mirror_record(mirror_record(rec))
    assert twice == rec                                                                       # the helper itself is an involution


def test_the_mirrored_match_really_differs(open_terrain):
    """Guard against a vacuous commutation test: the swapped view differs from the view itself in the carried arrays."""
    rec = record(rich_events())
    v = tokens.build_view(tokens.MatchContext(rec, open_terrain), REF, K)
    assert {'players_num', 'players_zone', 'ev_team', 'ev_xy', 'ev_actors', 'summary', 'ed_i'} <= set(differing(v, symmetry.swap_view(v)))


def test_golden_swap_view_equals_the_view_of_the_mirrored_match(golden_records, open_terrain):
    n = 0
    for m in (0, 7, 13):
        rec = golden_records[m]
        for k in (3, len(rec.frame_ts) // 2, len(rec.frame_ts) - 2):
            t0 = int(rec.frame_ts[k])
            for ref in (t0, t0 + 21_000):
                try:
                    assert_swap_commutes(open_terrain, rec, ref, k)
                except AssertionError as err:
                    raise AssertionError(f'golden match {m} frame {k} ref {ref}: {str(err)[:200]}') from None
                n += 1
    assert n == 18
