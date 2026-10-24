import numpy as np
import pytest
from studies.tog_experiments_20261012.e2_groups import pre_tau_group, stratum

TEAM = np.array([100] * 5 + [200] * 5)
O = 500.0  # a zero coordinate means "unknown position", so test points are offset

def _xy(points):
    return np.array(points, dtype=float)

def test_far_apart_is_separated():
    xy = _xy([(O + i * 3000.0, O) for i in range(10)])
    assert pre_tau_group(xy, TEAM) == (0, 0, 0)
    assert stratum(0, 0) == "separated"

def test_chain_links_within_radius():
    pts = [(0, 0), (1500, 0), (3000, 0), (9000, 9000), (12000, 9000),
           (4500, 0), (6000, 0), (20000, 0), (23000, 0), (26000, 0)]
    assert pre_tau_group(_xy(pts) + O, TEAM) == (2, 3, 0)  # 3 blue + 2 red chained
    assert stratum(2, 3) == "small"

def test_unknown_players_excluded():
    pts = [(1000, 1000)] * 5 + [(1100, 1000)] * 5
    xy = _xy(pts); xy[9] = (np.nan, np.nan); xy[8] = (0.0, 500.0)
    assert pre_tau_group(xy, TEAM) == (3, 5, 2)

def test_strata_edges():
    assert stratum(1, 5) == "pick"
    assert stratum(3, 4) == "asymmetric"
    assert stratum(4, 4) == "teamfight"

def test_equal_size_groups_more_balanced_wins():
    # {blue 0, red 5, 6, 7} and {blue 3, 4, red 8, 9}, far apart; blue 1 and 2 parked alone.
    # Both mixed groups have size 4; n_small 2 beats n_small 1 of the group that holds index 0.
    pts = [(1000, 1000), (10000, 10000), (14000, 10000), (20000, 1000), (20500, 1000),
           (1500, 1000), (1000, 1500), (1500, 1500), (20000, 1500), (20500, 1500)]
    assert pre_tau_group(_xy(pts), TEAM) == (2, 2, 0)

def test_larger_group_beats_first_found():
    # {blue 0, red 5} (holds index 0, found first) and {blue 1, 2, 3, red 6, 7, 8}; blue 4 and red 9 parked alone.
    pts = [(1000, 1000), (10000, 1000), (10500, 1000), (11000, 1000), (1000, 10000),
           (1500, 1000), (10000, 1500), (10500, 1500), (11000, 1500), (20000, 20000)]
    assert pre_tau_group(_xy(pts), TEAM) == (3, 3, 0)

@pytest.mark.parametrize("radius", [0.0, -2000.0, float("nan"), float("inf")])
def test_radius_must_be_finite_and_positive(radius):
    xy = _xy([(O + i * 3000.0, O) for i in range(10)])
    with pytest.raises(ValueError):
        pre_tau_group(xy, TEAM, radius)
