"""P3 rule contract tests. Every timeline below is SYNTHETIC (invented integers), not an observed case."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import engagement_labels_v3_rules as R  # noqa: E402

INF = math.inf


def test_horizon_wins_synthetic():
    kills = [100_000, 115_000, 400_000]
    L = 115_000
    nk = R.next_kill_after(kills, L)
    e, reasons, _ = R.endpoint_rule(L, 90, nk, R.effective_next_start(900_000, 2_000_000), 2_000_000)
    assert nk == 400_000 and e == L + 90_000 and reasons == ('horizon',)


def test_next_kill_truncates_and_is_excluded_synthetic():
    kills = [100_000, 115_000, 160_000]
    e, reasons, _ = R.endpoint_rule(115_000, 90, R.next_kill_after(kills, 115_000), INF, 2_000_000)
    assert e == 159_999 and reasons == ('next_kill',)
    assert not any(115_000 < t <= e for t in kills)


def test_kill_at_last_kill_timestamp_is_not_next_kill_synthetic():
    kills = [100_000, 115_000, 115_000, 300_000]
    assert R.next_kill_after(kills, 115_000) == 300_000


def test_next_engagement_truncates_synthetic():
    e, reasons, _ = R.endpoint_rule(115_000, 120, 250_000, R.effective_next_start(150_000, 2_000_000), 2_000_000)
    assert e == 149_999 and reasons == ('next_engagement_start',)


def test_game_end_truncates_synthetic():
    e, reasons, _ = R.endpoint_rule(115_000, 120, INF, R.effective_next_start(170_000, 170_000), 170_000)
    assert e == 169_999 and reasons == ('game_end',)


def test_sentinel_next_start_equal_to_end_is_absent():
    assert R.effective_next_start(170_000, 170_000) == INF
    assert R.effective_next_start(169_999, 170_000) == 169_999


def test_all_tied_reasons_are_kept_synthetic():
    # next kill at L+60001 and next engagement start at the same ms, horizon 60 s -> three-way tie.
    L = 115_000
    e, reasons, cand = R.endpoint_rule(L, 60, L + 60_001, L + 60_001, 2_000_000)
    assert e == L + 60_000 and reasons == ('horizon', 'next_kill', 'next_engagement_start')
    assert cand['game_end'] == 1_999_999


def test_monotone_in_horizon_synthetic():
    L, nk, ns, end = 115_000, 200_000, INF, 2_000_000
    es = [R.endpoint_rule(L, h, nk, ns, end)[0] for h in R.HORIZONS_S]
    assert es == sorted(es) and es == [175_000, 199_999, 199_999]


def test_validity_flags_synthetic():
    v = R.endpoint_validity(endpoint=200_000, L=115_000, q_pre=99_999, support_start=0, last_frame=300_000)
    assert all(v.values())
    v = R.endpoint_validity(endpoint=400_000, L=115_000, q_pre=99_999, support_start=0, last_frame=300_000)
    assert not v['endpoint_le_last_frame']


def test_label_threshold_zero_is_non_improvement():
    assert R.label_from_delta(0.0) == 0
    assert R.label_from_delta(-0.0) == 0
    assert R.label_from_delta(5e-324) == 1
    assert R.label_from_delta(-1e-12) == 0
    with pytest.raises(ValueError):
        R.label_from_delta(float('nan'))


def test_near_zero_bands_are_descriptions_only():
    f = R.near_zero_flags(0.0075)
    assert f == {'abs_delta_le_0.005': False, 'abs_delta_le_0.01': True, 'abs_delta_le_0.02': True}
    assert R.label_from_delta(0.0075) == 1


def test_non_integer_timestamps_rejected():
    with pytest.raises(TypeError):
        R.endpoint_rule(115_000.5, 90, INF, INF, 2_000_000)


def test_event_categories_and_counts_synthetic():
    ev = [{'type': 'ELITE_MONSTER_KILL', 'timestamp': 120_000, 'monsterType': 'DRAGON', 'monsterSubType': 'FIRE_DRAGON', 'killerTeamId': 100},
          {'type': 'DRAGON_SOUL_GIVEN', 'timestamp': 120_000, 'teamId': 100},
          {'type': 'DRAGON_SOUL_GIVEN', 'timestamp': 121_000, 'teamId': 0},
          {'type': 'ELITE_MONSTER_KILL', 'timestamp': 130_000, 'monsterType': 'DRAGON', 'monsterSubType': 'ELDER_DRAGON', 'killerTeamId': 200},
          {'type': 'BUILDING_KILL', 'timestamp': 140_000, 'teamId': 200, 'buildingType': 'TOWER_BUILDING'}]
    rows = sorted([(e['timestamp'], *R.categorize(e)) for e in ev], key=lambda r: r[0])
    c = R.count_events(rows, 115_000, 130_000)
    assert c['dragon'] == 1 and c['dragon_FIRE'] == 1 and c['dragon_blue'] == 1
    assert c['soul_owned'] == 1 and c['soul_owned_blue'] == 1 and c['soul_teamid0_unassigned'] == 1
    assert c['elder'] == 1 and c['elder_red'] == 1 and 'tower' not in c
    assert 'soul_teamid0_unassigned_blue' not in c and 'soul_teamid0_unassigned_red' not in c
    c2 = R.count_events(rows, 130_000, 140_000)
    assert c2 == {'tower': 1, 'tower_blue': 1}
