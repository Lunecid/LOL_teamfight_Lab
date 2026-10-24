"""Tests for the q spatial inputs (studies/q_production_20261005/spatial.py).

Synthetic data unless stated.  Expected values come from independent loops, scipy, numpy.linalg,
the recorded design validation (reports/feature_control_study_20261001/GRAPH_DESIGN_VALIDATION.json)
and the validator's own aggregate() (validate_graph_amendment.py, extracted with ast; its validate()
is never run because it rewrites design files).  The last test reads 50 real engagement rows and
skips when the files are absent.
"""
from __future__ import annotations

import ast
import inspect
import itertools
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree

from studies.q_production_20261005 import spatial as sp

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "reports/feature_control_study_20261001"
LABELS = Path(r"D:\LOL_Training_20261004\labels_1")
STANDARD = np.array([100] * 5 + [200] * 5)
SHUFFLED = np.array([200, 100, 200, 100, 200, 100, 200, 100, 200, 100])


def _rows(n: int, seed: int, *, scale: float = 15000.0, shuffled: bool = False):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, scale, size=(n, 10, 2))
    team = np.stack([rng.permutation(STANDARD) for _ in range(n)]) if shuffled else np.tile(STANDARD, (n, 1))
    return xy, team


def _edge_matrix(values: np.ndarray) -> np.ndarray:
    """[n,90] edge values -> [n,10,10] with out[n, receiver, sender]."""

    out = np.zeros((len(values), 10, 10))
    out[:, sp.EDGE_DST, sp.EDGE_SRC] = values
    return out


def _hand_phi_s(xy: np.ndarray, team: np.ndarray, sigma: float) -> np.ndarray:
    """Independent loop implementation of the 64 Phi_S values of one row (HSG section 4 order)."""

    d = [[math.hypot(*(xy[j] - xy[i])) / sigma for j in range(10)] for i in range(10)]
    w = [[math.exp(-d[i][j] ** 2) for j in range(10)] for i in range(10)]
    values = []
    for i in range(10):
        allies = [j for j in range(10) if j != i and team[j] == team[i]]
        enemies = [j for j in range(10) if team[j] != team[i]]
        ally, enemy = sum(w[i][j] for j in allies) / 4, sum(w[i][j] for j in enemies) / 5
        values += [ally, enemy, ally - enemy, min(d[i][j] for j in allies), min(d[i][j] for j in enemies)]
    for code in (100, 200):
        members = [j for j in range(10) if team[j] == code]
        pairs = [d[a][b] for a, b in itertools.combinations(members, 2)]
        tree = minimum_spanning_tree(np.array([[d[a][b] for b in members] for a in members])).toarray()
        weight = np.array([[w[a][b] if a != b else 0.0 for b in members] for a in members])
        lam = np.linalg.eigvalsh(np.diag(weight.sum(axis=1)) - weight)[1]
        values += [sum(pairs) / 10, max(pairs), tree.sum(), tree.max(), lam]
    blue = [j for j in range(10) if team[j] == 100]
    red = [j for j in range(10) if team[j] == 200]
    cross = [d[a][b] for a in blue for b in red]
    values += [sum(cross) / 25, min(cross), sum(w[a][b] for a in blue for b in red) / 25,
               math.hypot(*(xy[red].mean(axis=0) - xy[blue].mean(axis=0))) / sigma]
    return np.array(values)


def _cards() -> dict:
    path = DESIGN / "EXECUTION_CARDS.json"
    if not path.is_file():
        pytest.skip(f"design cards absent: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


# --------------------------------------------------------------------------- dimensions and layout

def test_dimensions_names_and_edge_order():
    xy, team = _rows(7, 1, shuffled=True)
    sigma = sp.fit_sigma(xy)
    s200, s_names = sp.tabular_s200(xy, sigma)
    a, a_names = sp.phi_s_A(xy, team, sigma)
    assert s200.shape == (7, 200) and s200.dtype == np.float64 and len(set(s_names)) == len(s_names) == 200
    assert a.shape == (7, 64) and a.dtype == np.float64 and len(set(a_names)) == len(a_names) == 64
    assert s_names[:3] == ["S.p01.x", "S.p01.y", "S.p02.x"] and s_names[18:20] == ["S.p10.x", "S.p10.y"]
    assert s_names[20:25] == ["S.p01_p02.dx_over_sigma", "S.p01_p02.dy_over_sigma", "S.p01_p02.d_over_sigma",
                              "S.p01_p02.w", "S.p01_p03.dx_over_sigma"]
    assert s_names[-1] == "S.p09_p10.w"
    assert a_names[0] == "A.node.p01.ally_w_sum_div4" and a_names[49] == "A.node.p10.nearest_enemy_d_over_sigma"
    assert a_names[50] == "A.team.blue.within_mean_d_over_sigma" and a_names[59] == "A.team.red.laplacian_lambda2"
    assert a_names[60:] == ["A.cross.cross_mean_d_over_sigma", "A.cross.cross_min_d_over_sigma",
                            "A.cross.cross_mean_w", "A.cross.centroid_d_over_sigma"]
    for names, width in ((sp.m_stat_names(True, False), 60), (sp.m_stat_names(False, True), 180), (sp.m_stat_names(), 240)):
        m, m_names = sp.phi_ps_M(xy, team, sigma, np.ones((7, 10, len(names))), names)
        assert m.shape == (7, width) and len(set(m_names)) == len(m_names) == width
    e = sp.edges(xy, team, sigma)
    assert e.vector.shape == (7, 90, 3) and e.weight.shape == e.log_weight.shape == e.relation.shape == (7, 90)
    assert not np.any(e.src == e.dst)
    assert {(int(j), int(i)) for j, i in zip(e.src, e.dst)} == {(j, i) for j in range(10) for i in range(10) if j != i}
    assert ((e.relation == sp.ALLY).sum(axis=1) == 40).all() and ((e.relation == sp.ENEMY).sum(axis=1) == 50).all()
    assert all(k == 9 * i + j - (j > i) for k, (j, i) in enumerate(zip(e.src, e.dst)))
    names = sp.edge_names()
    assert len(set(names)) == 90 and names[:2] == ["p02->p01", "p03->p01"] and names[-1] == "p09->p10"
    assert sp.slot_participant_order() == tuple(range(1, 11))


def test_layout_matches_execution_cards():
    cards = _cards()
    tab, graph = cards["schema"]["tabular_spatial"], cards["graph"]
    assert tab["xy_fields"] == sp.S200_XY.stop - sp.S200_XY.start == 20
    assert tab["unordered_pairs"] == len(sp.PAIR_I) == 45
    assert tab["spatial_numeric_fields"] == len(sp.tabular_s200_names()) == 200
    card_to_field = {"dx/sigma": "dx_over_sigma", "dy/sigma": "dy_over_sigma", "distance/sigma": "d_over_sigma", "weight": "w"}
    assert [card_to_field[name] for name in tab["pair_fields"]] == list(sp.S200_PAIR_FIELDS)
    assert tab["pair_order"] == "canonical participant order i<j" and tab["reverse"] == "negate dx/dy only"
    assert (graph["nodes"], graph["ally_directed"], graph["enemy_directed"], graph["self_edges"]) == (10, 40, 50, 0)
    assert graph["w"] == "exp(-(distance/sigma)^2)" and graph["edge_direction"] == "j->i uses x_j-x_i,y_j-y_i"
    assert graph["edge_vector"] == 3
    s0 = sp.edges_s0(STANDARD[None])
    assert s0.vector[0, 0].tolist() == graph["S0"]["vector"] and s0.weight[0, 0] == graph["S0"]["weight"]
    assert (graph["A"]["node"], graph["A"]["team"], graph["A"]["cross_global"], graph["A"]["total"]) == (
        10 * len(sp.A_NODE_FIELDS), 2 * len(sp.A_TEAM_FIELDS), len(sp.A_CROSS_FIELDS), len(sp.phi_s_names()))
    assert tuple(graph["M"]["E"]) == sp.M_E_STATS and tuple(graph["M"]["R"]) == sp.M_R_STATS
    assert graph["M"]["per_stat_per_node"] == len(sp.M_SUMMARIES)
    assert (graph["M"]["E_only"], graph["M"]["R_only"], graph["M"]["both"]) == tuple(
        len(sp.phi_ps_names(sp.m_stat_names(e, r))) for e, r in ((True, False), (False, True), (True, True)))


# --------------------------------------------------------------------------- known geometry

def test_known_geometry_by_hand():
    """Blue collinear at {0,1,2,4,7} sigma on odd slots, Red coincident at 12 sigma on even slots."""

    sigma = 1000.0
    blue_slots, red_slots, positions = [1, 3, 5, 7, 9], [0, 2, 4, 6, 8], [0.0, 1.0, 2.0, 4.0, 7.0]
    xy = np.zeros((1, 10, 2))
    for slot, p in zip(blue_slots, positions):
        xy[0, slot] = (p * sigma, 0.0)
    xy[0, red_slots] = (12.0 * sigma, 0.0)
    a, names = sp.phi_s_A(xy, SHUFFLED[None], sigma)
    value = dict(zip(names, a[0]))
    assert value["A.team.blue.within_mean_d_over_sigma"] == pytest.approx(3.4, rel=1e-14)   # 34 / 10
    assert value["A.team.blue.within_max_d_over_sigma"] == 7.0
    assert value["A.team.blue.mst_total_d_over_sigma"] == 7.0                            # gaps 1, 1, 2, 3
    assert value["A.team.blue.mst_max_edge_d_over_sigma"] == 3.0
    weight = np.array([[math.exp(-(p - q) ** 2) if a_ != b_ else 0.0 for b_, q in enumerate(positions)]
                       for a_, p in enumerate(positions)])
    assert value["A.team.blue.laplacian_lambda2"] == pytest.approx(
        np.linalg.eigvalsh(np.diag(weight.sum(axis=1)) - weight)[1], rel=1e-12)
    assert value["A.team.red.laplacian_lambda2"] == pytest.approx(5.0, abs=1e-12)        # K5, unit weights
    for field in ("within_mean_d_over_sigma", "within_max_d_over_sigma", "mst_total_d_over_sigma", "mst_max_edge_d_over_sigma"):
        assert value[f"A.team.red.{field}"] == 0.0
    cross = [12.0 - p for p in positions]
    assert value["A.cross.cross_mean_d_over_sigma"] == pytest.approx(9.2, rel=1e-14)
    assert value["A.cross.cross_min_d_over_sigma"] == 5.0
    assert value["A.cross.cross_mean_w"] == pytest.approx(sum(math.exp(-c * c) for c in cross) / 5, rel=1e-12)
    assert value["A.cross.centroid_d_over_sigma"] == pytest.approx(12.0 - 2.8, rel=1e-14)
    for slot, p in zip(blue_slots, positions):
        node = f"A.node.p{slot + 1:02d}."
        ally = sum(math.exp(-(p - q) ** 2) for q in positions if q != p) / 4
        enemy = math.exp(-(12.0 - p) ** 2)
        assert value[node + "ally_w_sum_div4"] == pytest.approx(ally, rel=1e-12)
        assert value[node + "enemy_w_sum_div5"] == pytest.approx(enemy, rel=1e-12)
        assert value[node + "ally_minus_enemy_w"] == pytest.approx(ally - enemy, rel=1e-12)
        assert value[node + "nearest_ally_d_over_sigma"] == min(abs(p - q) for q in positions if q != p)
        assert value[node + "nearest_enemy_d_over_sigma"] == 12.0 - p
    enemy_red = sum(math.exp(-c * c) for c in cross) / 5
    for slot in red_slots:
        node = f"A.node.p{slot + 1:02d}."
        assert value[node + "ally_w_sum_div4"] == 1.0 and value[node + "nearest_ally_d_over_sigma"] == 0.0
        assert value[node + "enemy_w_sum_div5"] == pytest.approx(enemy_red, rel=1e-12)
        assert value[node + "ally_minus_enemy_w"] == pytest.approx(1.0 - enemy_red, rel=1e-12)
        assert value[node + "nearest_enemy_d_over_sigma"] == 5.0
    np.testing.assert_allclose(a[0], _hand_phi_s(xy[0], SHUFFLED, sigma), rtol=1e-12, atol=1e-15)


def test_s200_hand_values_reverse_pair_and_edge_identity():
    sigma = 1000.0
    xy, team = _rows(1, 3)
    xy[0, 0] = (0.0, 0.0)                      # exact (0,0): an ordinary coordinate (D-I1)
    xy[0, 5] = (3000.0, 4000.0)
    s200, names = sp.tabular_s200(xy, sigma)
    value = dict(zip(names, s200[0]))
    assert (value["S.p01_p06.dx_over_sigma"], value["S.p01_p06.dy_over_sigma"], value["S.p01_p06.d_over_sigma"]) == (3.0, 4.0, 5.0)
    assert value["S.p01_p06.w"] == pytest.approx(math.exp(-25.0), rel=1e-15)
    assert (value["S.p01.x"], value["S.p01.y"], value["S.p06.x"], value["S.p06.y"]) == (0.0, 0.0, 3000.0, 4000.0)
    assert np.array_equal(s200[0, sp.S200_XY], xy[0].reshape(20))
    for k, (i, j) in enumerate(itertools.combinations(range(10), 2)):
        dx, dy = xy[0, j] - xy[0, i]
        d = math.hypot(dx, dy) / sigma
        np.testing.assert_allclose(s200[0, 20 + 4 * k:24 + 4 * k], [dx / sigma, dy / sigma, d, math.exp(-d * d)],
                                   rtol=1e-12, atol=1e-15)
    swapped = xy.copy()
    swapped[0, [0, 5]] = xy[0, [5, 0]]          # exchanging the two players reverses the pair
    reverse, _ = sp.tabular_s200(swapped, sigma)
    k = names.index("S.p01_p06.dx_over_sigma")
    assert reverse[0, k] == -s200[0, k] and reverse[0, k + 1] == -s200[0, k + 1]
    assert reverse[0, k + 2] == s200[0, k + 2] and reverse[0, k + 3] == s200[0, k + 3]
    e = sp.edges(xy, team, sigma)
    edge = {(int(j), int(i)): n for n, (j, i) in enumerate(zip(e.src, e.dst))}
    for k, (i, j) in enumerate(zip(sp.PAIR_I, sp.PAIR_J)):  # pair (i<j) carries edge j -> i exactly
        assert np.array_equal(s200[:, 20 + 4 * k:23 + 4 * k], e.vector[:, edge[(int(j), int(i))]])
        assert np.array_equal(s200[:, 23 + 4 * k], e.weight[:, edge[(int(j), int(i))]])


def test_edges_reverse_sign_relations_and_formula():
    xy, team = _rows(5, 4, shuffled=True)
    sigma = sp.fit_sigma(xy)
    e = sp.edges(xy, team, sigma)
    edge = {(int(j), int(i)): n for n, (j, i) in enumerate(zip(e.src, e.dst))}
    for (j, i), k in edge.items():
        r = edge[(i, j)]
        assert np.array_equal(e.vector[:, k, :2], -e.vector[:, r, :2])
        assert np.array_equal(e.vector[:, k, 2], e.vector[:, r, 2]) and np.array_equal(e.weight[:, k], e.weight[:, r])
        assert np.array_equal(e.relation[:, k], np.where(team[:, j] == team[:, i], sp.ALLY, sp.ENEMY))
        dx, dy = xy[:, j, 0] - xy[:, i, 0], xy[:, j, 1] - xy[:, i, 1]
        d = np.hypot(dx, dy) / sigma
        np.testing.assert_allclose(e.vector[:, k], np.stack([dx / sigma, dy / sigma, d], axis=1), rtol=1e-12)
        np.testing.assert_allclose(e.weight[:, k], np.exp(-d * d), rtol=1e-12)
    assert np.array_equal(e.log_weight, -(e.vector[:, :, 2] * e.vector[:, :, 2]))
    assert np.array_equal(e.weight, np.exp(e.log_weight))


def test_s0_constants_keep_relations_without_coordinates():
    xy, team = _rows(4, 5, shuffled=True)
    s1 = sp.edges(xy, team, sp.fit_sigma(xy))
    s0 = sp.edges_s0(team)
    assert list(inspect.signature(sp.edges_s0).parameters) == ["team_ids"]
    assert np.array_equal(s0.src, s1.src) and np.array_equal(s0.dst, s1.dst) and np.array_equal(s0.relation, s1.relation)
    assert s0.vector.shape == (4, 90, 3) and (s0.vector == 0.0).all()
    assert (s0.weight == 1.0).all() and (s0.log_weight == 0.0).all()


def test_sigma_is_median_of_positive_pair_distances():
    xy, _ = _rows(3, 6)
    xy[0, 4] = xy[0, 2]                         # coincident pair: zero distance is not positive
    xy[1, 7] = (0.0, 0.0)                       # (0,0) counts like any other coordinate
    xy[2, :] = (500.0, 500.0)                   # all coincident: no positive distance
    distances = [math.hypot(*(xy[r, j] - xy[r, i])) for r in range(3) for i, j in itertools.combinations(range(10), 2)]
    positive = [d for d in distances if d > 0]
    assert len(positive) == 44 + 45
    assert sp.fit_sigma(xy) == pytest.approx(float(np.median(positive)), rel=1e-14)
    assert np.allclose(sp.pair_distances(xy).ravel(), distances, rtol=1e-14)
    with pytest.raises(ValueError, match="no positive pair distance"):
        sp.fit_sigma(xy[2:3])


# --------------------------------------------------------------------------- MST, lambda2, invariances

def test_mst_and_lambda2_against_scipy_and_eigvalsh():
    xy, team = _rows(25, 7, shuffled=True)
    sigma = sp.fit_sigma(xy)
    a, names = sp.phi_s_A(xy, team, sigma)
    column = {name: k for k, name in enumerate(names)}
    for r in range(len(xy)):
        for code, label in ((100, "blue"), (200, "red")):
            points = xy[r, team[r] == code]
            distance = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)) / sigma
            tree = minimum_spanning_tree(distance).toarray()
            assert np.count_nonzero(tree) == 4
            assert a[r, column[f"A.team.{label}.mst_total_d_over_sigma"]] == pytest.approx(tree.sum(), rel=1e-12)
            assert a[r, column[f"A.team.{label}.mst_max_edge_d_over_sigma"]] == pytest.approx(tree.max(), rel=1e-12)
            weight = np.exp(-distance ** 2)
            np.fill_diagonal(weight, 0.0)
            eigenvalues = np.linalg.eigvalsh(np.diag(weight.sum(axis=1)) - weight)
            assert abs(eigenvalues[0]) < 1e-12 and eigenvalues[1] >= -1e-12
            assert a[r, column[f"A.team.{label}.laplacian_lambda2"]] == pytest.approx(eigenvalues[1], rel=1e-9, abs=1e-12)
        np.testing.assert_allclose(a[r], _hand_phi_s(xy[r], team[r], sigma), rtol=1e-9, atol=1e-12)


def test_within_team_permutation_moves_node_features_only():
    n = 30
    xy, team = _rows(n, 8, shuffled=True)
    rng = np.random.default_rng(80)
    stats = rng.normal(size=(n, 10, 8))
    sigma = sp.fit_sigma(xy)
    perm = np.tile(np.arange(10), (n, 1))
    for r in range(n):
        for code in (100, 200):
            slots = np.flatnonzero(team[r] == code)
            perm[r, slots] = rng.permutation(slots)
    rows = np.arange(n)[:, None]
    xy_p, team_p, stats_p = xy[rows, perm], team[rows, perm], stats[rows, perm]
    assert np.array_equal(team_p, team)
    node, group, cross = sp.split_phi_s(sp.phi_s_A(xy, team, sigma)[0])
    node_p, group_p, cross_p = sp.split_phi_s(sp.phi_s_A(xy_p, team_p, sigma)[0])
    np.testing.assert_allclose(node_p, node[rows, perm], atol=1e-6)
    np.testing.assert_allclose(group_p, group, atol=1e-6)
    np.testing.assert_allclose(cross_p, cross, atol=1e-6)
    m = sp.phi_ps_M(xy, team, sigma, stats, sp.m_stat_names())[0].reshape(n, 10, -1)
    m_p = sp.phi_ps_M(xy_p, team_p, sigma, stats_p, sp.m_stat_names())[0].reshape(n, 10, -1)
    np.testing.assert_allclose(m_p, m[rows, perm], atol=1e-6)
    weight = _edge_matrix(sp.edges(xy, team, sigma).weight)
    weight_p = _edge_matrix(sp.edges(xy_p, team_p, sigma).weight)
    np.testing.assert_allclose(weight_p, weight[np.arange(n)[:, None, None], perm[:, :, None], perm[:, None, :]], atol=1e-12)


def test_translation_invariance_and_zero_coordinate_is_ordinary():
    xy, team = _rows(10, 9, shuffled=True)
    sigma = sp.fit_sigma(xy)
    shifted = xy - xy[:, :1, :]                 # slot 0 lands exactly on (0,0) in every row
    assert sp.zero_coordinate_players(shifted)[:, 0].all() and not sp.zero_coordinate_players(shifted)[:, 1:].any()
    a = sp.phi_s_A(xy, team, sigma)[0]
    a_shifted = sp.phi_s_A(shifted, team, sigma)[0]
    np.testing.assert_allclose(a_shifted, a, rtol=1e-9, atol=1e-9)
    s, s_shifted = sp.tabular_s200(xy, sigma)[0], sp.tabular_s200(shifted, sigma)[0]
    np.testing.assert_allclose(s_shifted[:, sp.S200_PAIRS], s[:, sp.S200_PAIRS], rtol=1e-9, atol=1e-9)
    assert sp.fit_sigma(shifted) == pytest.approx(sigma, rel=1e-9)
    for r in range(len(xy)):
        np.testing.assert_allclose(a_shifted[r], _hand_phi_s(shifted[r], team[r], sigma), rtol=1e-9, atol=1e-12)


def test_phi_ps_by_hand_column_order_and_active_parents():
    n = 4
    xy, team = _rows(n, 10, shuffled=True)
    sigma = sp.fit_sigma(xy)
    stats = np.random.default_rng(11).uniform(0.0, 5.0, size=(n, 10, 8))
    names = sp.m_stat_names()
    m, columns = sp.phi_ps_M(xy, team, sigma, stats, names)
    for r in range(n):
        for i in range(10):
            w = [math.exp(-(math.hypot(*(xy[r, j] - xy[r, i])) / sigma) ** 2) for j in range(10)]
            for k, z in enumerate(names):
                ally = sum(w[j] * stats[r, j, k] for j in range(10) if j != i and team[r, j] == team[r, i]) / 4
                enemy = sum(w[j] * stats[r, j, k] for j in range(10) if team[r, j] != team[r, i]) / 5
                base = (i * len(names) + k) * 3
                assert columns[base:base + 3] == [f"M.p{i + 1:02d}.{z}.{s}" for s in sp.M_SUMMARIES]
                np.testing.assert_allclose(m[r, base:base + 3], [ally, enemy, ally - enemy], rtol=1e-12, atol=1e-15)
    e_only = sp.phi_ps_M(xy, team, sigma, stats[:, :, :2], sp.m_stat_names(True, False))[0]
    r_only = sp.phi_ps_M(xy, team, sigma, stats[:, :, 2:], sp.m_stat_names(False, True))[0]
    both = m.reshape(n, 10, 8, 3)
    assert np.array_equal(both[:, :, :2], e_only.reshape(n, 10, 2, 3))
    assert np.array_equal(both[:, :, 2:], r_only.reshape(n, 10, 6, 3))


def test_row_chunking_does_not_change_values(monkeypatch):
    xy, team = _rows(11, 14, shuffled=True)
    sigma = sp.fit_sigma(xy)
    stats = np.random.default_rng(15).normal(size=(11, 10, 2))
    a, m = sp.phi_s_A(xy, team, sigma)[0], sp.phi_ps_M(xy, team, sigma, stats, sp.M_E_STATS)[0]
    monkeypatch.setattr(sp, "CHUNK_ROWS", 3)
    assert np.array_equal(sp.phi_s_A(xy, team, sigma)[0], a)
    assert np.array_equal(sp.phi_ps_M(xy, team, sigma, stats, sp.M_E_STATS)[0], m)


def test_inputs_are_validated_not_repaired():
    xy, team = _rows(2, 12)
    sigma = sp.fit_sigma(xy)
    six_blue = team.copy()
    six_blue[0, 5] = 100
    other = team.copy()
    other[1, 0] = 300
    for bad in (six_blue, other, team[:, :9], team.astype(float)):
        with pytest.raises(ValueError):
            sp.edges(xy, bad, sigma)
        with pytest.raises(ValueError):
            sp.phi_s_A(xy, bad, sigma)
    with pytest.raises(ValueError):
        sp.edges_s0(six_blue)
    nan = xy.copy()
    nan[1, 3, 0] = np.nan
    for call in (lambda: sp.fit_sigma(nan), lambda: sp.tabular_s200(nan, sigma), lambda: sp.phi_s_A(nan, team, sigma),
                 lambda: sp.edges(nan, team, sigma), lambda: sp.phi_ps_M(nan, team, sigma, np.ones((2, 10, 2)), sp.M_E_STATS)):
        with pytest.raises(ValueError, match="non-finite"):
            call()
    for bad_sigma in (0.0, -1.0, np.nan, np.inf, [1.0, 2.0]):
        with pytest.raises(ValueError):
            sp.tabular_s200(xy, bad_sigma)
    stats = np.ones((2, 10, 2))
    stats[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sp.phi_ps_M(xy, team, sigma, stats, sp.M_E_STATS)
    with pytest.raises(ValueError):
        sp.phi_ps_M(xy, team, sigma, np.ones((2, 10, 3)), sp.M_E_STATS)
    with pytest.raises(ValueError):
        sp.phi_ps_M(xy, team, sigma, np.ones((2, 10, 2)), ("level", "level"))
    empty_xy, empty_team = np.zeros((0, 10, 2)), np.zeros((0, 10), dtype=np.int64)
    assert sp.phi_s_A(empty_xy, empty_team, 1.0)[0].shape == (0, 64)
    assert sp.tabular_s200(empty_xy, 1.0)[0].shape == (0, 200)
    assert sp.edges(empty_xy, empty_team, 1.0).vector.shape == (0, 90, 3)


# --------------------------------------------------------------------------- recorded design validation

def _validator_aggregate():
    path = DESIGN / "validate_graph_amendment.py"
    if not path.is_file():
        pytest.skip(f"design validator absent: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    found = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "aggregate"]
    assert len(found) == 1
    namespace = {"np": np, "mask": ~np.eye(10, dtype=bool)}  # the closure variable of validate()
    exec(compile(ast.Module(body=found, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["aggregate"]


def _softmax_by_receiver(log_weight: np.ndarray) -> np.ndarray:
    """Stable softmax over the 9 senders of every receiver -> [n,10,10] alpha[n, receiver, sender]."""

    logits = log_weight.reshape(-1, 10, 9)
    alpha = np.exp(logits - logits.max(axis=2, keepdims=True))
    alpha /= alpha.sum(axis=2, keepdims=True)
    return _edge_matrix(alpha.reshape(-1, 90))


def test_cross_check_with_validator_aggregate():
    aggregate = _validator_aggregate()
    rng = np.random.default_rng(20261001)       # the validator's synthetic draws, in its order
    xy, h = rng.normal(size=(10, 2)), rng.normal(size=(10, 4))
    perm = np.r_[rng.permutation(5), 5 + rng.permutation(5)]
    team = STANDARD[None]
    m1, a1 = aggregate(xy, h, True)             # log weight -(d)^2, i.e. sigma = 1
    m0, a0 = aggregate(xy, h, False)
    s1 = sp.edges(xy[None], team, 1.0)
    alpha1 = _softmax_by_receiver(s1.log_weight)[0]
    np.testing.assert_allclose(alpha1, a1, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(alpha1 @ h, m1, rtol=1e-12, atol=1e-12)
    weight = _edge_matrix(s1.weight)[0]
    np.testing.assert_allclose(weight / weight.sum(axis=1, keepdims=True), a1, rtol=1e-12, atol=1e-15)
    alpha0 = _softmax_by_receiver(sp.edges_s0(team).log_weight)[0]
    np.testing.assert_allclose(alpha0, a0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(alpha0 @ h, m0, rtol=1e-12, atol=1e-12)
    _, extreme = aggregate(xy * 1e6, h, True)
    far = sp.edges(xy[None] * 1e6, team, 1.0)
    assert np.isfinite(far.log_weight).all() and np.isfinite(far.weight).all()
    np.testing.assert_allclose(_softmax_by_receiver(far.log_weight)[0], extreme, rtol=1e-9, atol=1e-12)
    permuted = sp.edges(xy[perm][None], team, 1.0)
    np.testing.assert_allclose(_softmax_by_receiver(permuted.log_weight)[0], a1[np.ix_(perm, perm)], rtol=1e-12, atol=1e-15)


def test_graph_design_validation_fixtures():
    path = DESIGN / "GRAPH_DESIGN_VALIDATION.json"
    if not path.is_file():
        pytest.skip(f"graph design validation absent: {path}")
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    assert record["directed_edges"] == {"ally": 40, "enemy": 50, "self": 0}
    dims = record["property_dimensions"]
    assert (dims["node"], dims["team"], dims["cross_team"] + dims["centroid"], dims["total"]) == (
        10 * len(sp.A_NODE_FIELDS), 2 * len(sp.A_TEAM_FIELDS), len(sp.A_CROSS_FIELDS), len(sp.phi_s_names()))
    sigma = float(record["synthetic_sigma"])
    rng = np.random.default_rng(int(record["synthetic_seed"]))
    xy = rng.uniform(0.0, 3000.0, size=(1, 10, 2))   # "NumPy default_rng uniform [0,3000), shape (10,2)"
    team = STANDARD[None]
    blocks = (sp.tabular_s200(xy, sigma)[0], sp.phi_s_A(xy, team, sigma)[0], sp.edges(xy, team, sigma).vector,
              sp.phi_ps_M(xy, team, sigma, np.ones((1, 10, 8)), sp.m_stat_names())[0])
    assert all(np.isfinite(block).all() for block in blocks)
    np.testing.assert_allclose(sp.phi_s_A(xy + 1234.5, team, sigma)[0], blocks[1], rtol=1e-9, atol=1e-12)
    check = next(text for text in record["checks"] if text.startswith("MST at collinear positions"))
    found = re.search(r"\[([\d,\s]+)\]: total (\d+), maximum edge (\d+), four edges", check)
    positions = [float(v) for v in found.group(1).split(",")]
    line = np.zeros((1, 10, 2))
    line[0, :5, 0] = np.asarray(positions) * sigma
    line[0, 5:] = xy[0, 5:]
    a, names = sp.phi_s_A(line, team, sigma)
    value = dict(zip(names, a[0]))
    assert value["A.team.blue.mst_total_d_over_sigma"] == float(found.group(2))
    assert value["A.team.blue.mst_max_edge_d_over_sigma"] == float(found.group(3))
    assert "K5 unit-weight Laplacian: nonzero eigenvalues all 5" in record["checks"]
    k5 = np.zeros((1, 10, 2))
    k5[0, 5:] = (9000.0, 9000.0)
    assert sp.phi_s_A(k5, team, sigma)[0][0, names.index("A.team.blue.laplacian_lambda2")] == pytest.approx(5.0, abs=1e-12)


# --------------------------------------------------------------------------- bank reader

QUERY_DTYPE = np.dtype([("match_index", "<u4"), ("local_query", "<i4"), ("query_ms", "<i8"),
                        ("frame_index", "<i4"), ("event_limit", "<i4"), ("item_query", "<i8")])
MATCH_DTYPE = np.dtype([("key", "S64"), ("shard", "<u2"), ("local_match", "<u2")])


def _fake_bank(root: Path, matches: list[dict]) -> tuple[Path, Path]:
    """Minimal projection + bank with the real field layout (bank.ShardBuilder offsets, QUERY/MATCH_TABLE)."""

    bank, projection = root / "bank", root / "projection"
    bank.mkdir(parents=True)
    projection.mkdir()
    (bank / "FIELD_MANIFEST.json").write_text(json.dumps({
        "participant_order": list(range(1, 11)), "team_order": [100, 200],
        "ordered_fields": [{"id": "S.x", "block": "xy", "index": 0}, {"id": "S.y", "block": "xy", "index": 1}]}),
        encoding="utf-8")
    shards = sorted({m["shard"] for m in matches})
    bindings, local_of = [], {}
    for shard in shards:
        parts = {name: [] for name in sp.SHARD_FIELDS}
        for local, k in enumerate(k for k, m in enumerate(matches) if m["shard"] == shard):
            m = matches[k]
            local_of[k] = local
            clock = np.asarray(m["minute_ts"], dtype=np.int64)
            m.setdefault("frame_index", np.searchsorted(clock, np.asarray(m["queries"]), side="right") - 1)
            parts["xy"].append(np.asarray(m["xy"], dtype=np.float32))
            parts["minute_ts"].append(clock)
            parts["query_ms"].append(np.asarray(m["queries"], dtype=np.int32))
            parts["query_frame_index"].append(np.asarray(m["frame_index"], dtype=np.int32))
            parts["team_ids"].append(np.asarray(m["team"], dtype=np.int16))
        arrays = {}
        for name, values in parts.items():
            arrays[name] = np.concatenate(values)
            arrays["offset_" + name] = np.concatenate([[0], np.cumsum([len(v) for v in values])]).astype(np.int64)
        directory = bank / f"train_{shard:04d}"
        directory.mkdir()
        np.savez_compressed(directory / "BANK.npz", **arrays)
        bindings.append({"directory": directory.name, "archive_sha256": sp.sha256_file(directory / "BANK.npz")})
    query_rows, match_rows = [], []
    for k, m in enumerate(matches):
        match_rows.append((f"{k:064x}".encode("ascii"), shards.index(m["shard"]), local_of[k]))
        for local_query, (query_ms, frame) in enumerate(zip(m["queries"], m["frame_index"])):
            query_rows.append((k, local_query, query_ms, frame, 0, len(query_rows)))
    np.save(projection / "QUERY_TABLE.npy", np.array(query_rows, dtype=QUERY_DTYPE))
    np.save(projection / "MATCH_TABLE.npy", np.array(match_rows, dtype=MATCH_DTYPE))
    (projection / "BANK_METADATA.json").write_text(json.dumps({"source_root": str(bank), "source_shards": bindings}),
                                                    encoding="utf-8")
    return projection, bank


def _three_matches(seed: int = 13) -> tuple[list[dict], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    xy_a = rng.uniform(1.0, 15000.0, size=(3, 10, 2)).astype(np.float32)
    xy_b = rng.uniform(1.0, 15000.0, size=(3, 10, 2)).astype(np.float32)
    xy_b[0, 6] = 0.0                            # exact (0,0) kept as observed
    xy_b[1, 3, 0] = np.nan                      # recorded non-finite coordinate
    xy_c = rng.uniform(1.0, 15000.0, size=(2, 10, 2)).astype(np.float32)
    matches = [
        dict(shard=0, minute_ts=[0, 60000, 120000], xy=xy_a, team=STANDARD, queries=[50, 100000, 125000]),
        dict(shard=0, minute_ts=[0, 60000, 120000], xy=xy_b, team=SHUFFLED, queries=[30000, 70000]),
        dict(shard=1, minute_ts=[5000, 65000], xy=xy_c, team=STANDARD, queries=[1000, 65000]),
    ]
    return matches, (xy_a, xy_b, xy_c)


def test_reader_on_a_synthetic_bank(tmp_path):
    matches, (xy_a, xy_b, xy_c) = _three_matches()
    projection, _ = _fake_bank(tmp_path, matches)
    request = np.array([6, 0, 3, 5, 0, 4, 1, 2])  # unsorted, duplicated, both shards
    with sp.BankXYReader(projection, verify_hashes=True) as reader:
        record = reader.read(request)
        assert sorted(reader.shards_opened) == [0, 1]            # each BANK.npz opened once
    assert record.xy.shape == (8, 10, 2) and record.xy.dtype == np.float64
    assert record.has_frame.tolist() == [True, True, True, False, True, True, True, False]
    assert record.support.tolist() == [True, True, True, False, True, False, True, False]
    assert record.frame_ms.tolist() == [65000, 0, 0, -1, 0, 60000, 60000, -1]
    assert record.frame_index.tolist() == [1, 0, 0, -1, 0, 1, 1, -1]
    assert record.match_index.tolist() == [2, 0, 1, 2, 0, 1, 0, 0]
    assert record.query_ms.tolist() == [65000, 50, 30000, 1000, 50, 70000, 100000, 125000]
    for k, frame in ((0, xy_c[1]), (1, xy_a[0]), (2, xy_b[0]), (4, xy_a[0]), (6, xy_a[1])):
        assert np.array_equal(record.xy[k], frame.astype(np.float64))
    assert np.array_equal(record.xy[5], xy_b[1].astype(np.float64), equal_nan=True)   # kept as read
    assert np.isnan(record.xy[[3, 7]]).all()                                          # no frame: nothing imputed
    assert np.array_equal(record.team_ids[2], SHUFFLED) and np.array_equal(record.team_ids[0], STANDARD)
    xy, support = sp.xy_at_queries(request, projection=projection)
    assert np.array_equal(xy, record.xy, equal_nan=True) and np.array_equal(support, record.support)
    counts = sp.support_counts(record)
    assert counts == {"rows": 8, "supported": 5, "unsupported_no_frame": 2, "unsupported_nonfinite": 1,
                      "zero_coordinate_players": 1, "rows_with_zero_coordinate_player": 1,
                      "zero_coordinate_players_by_slot": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      "slot_participant_ids": list(range(1, 11))}
    assert sp.query_support_counts(request, projection=projection) == counts
    empty = sp.read_query_xy(np.zeros(0, dtype=np.int64), projection=projection)
    assert empty.xy.shape == (0, 10, 2) and empty.support.shape == (0,)
    with pytest.raises(IndexError):
        sp.read_query_xy([7], projection=projection)
    with pytest.raises(TypeError):
        sp.read_query_xy(np.array([True]), projection=projection)


def test_reader_refuses_changed_or_inconsistent_sources(tmp_path):
    matches, _ = _three_matches()
    projection, _ = _fake_bank(tmp_path / "hash", matches)
    metadata = json.loads((projection / "BANK_METADATA.json").read_text(encoding="utf-8"))
    metadata["source_shards"][1]["archive_sha256"] = "0" * 64
    (projection / "BANK_METADATA.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert sp.read_query_xy([6], projection=projection).support.tolist() == [True]
    with pytest.raises(ValueError, match="bound bank shard changed"):
        sp.read_query_xy([6], projection=projection, verify_hashes=True)
    table = np.load(projection / "QUERY_TABLE.npy")
    table["frame_index"][1] = 0                  # projection now disagrees with the bank
    np.save(projection / "QUERY_TABLE.npy", table)
    with pytest.raises(ValueError, match="frame indices differ"):
        sp.read_query_xy([1], projection=projection)
    future, _ = _three_matches()
    future[0]["frame_index"] = np.array([0, 2, 2])  # query 100000 bound to the frame at 120000
    projection, _ = _fake_bank(tmp_path / "future", future)
    with pytest.raises(ValueError, match="future frame"):
        sp.read_query_xy([1], projection=projection)


# --------------------------------------------------------------------------- real data (50 rows)

def _head(npz: Path, name: str, count: int) -> np.ndarray:
    """First rows of one uncompressed .npz member without loading the whole column."""

    with zipfile.ZipFile(npz) as archive, archive.open(name + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read_header(stream)
        assert not fortran and len(shape) == 1
        count = min(count, shape[0])
        return np.frombuffer(stream.read(count * dtype.itemsize), dtype=dtype, count=count).copy()


def test_real_data_smoke_50_engagement_pre_queries():
    engagements = LABELS / "ENGAGEMENTS.npz"
    needed = [engagements, sp.PROJECTION / "QUERY_TABLE.npy", sp.PROJECTION / "MATCH_TABLE.npy",
              sp.PROJECTION / "BANK_METADATA.json", sp.PROJECTION / "TEAM_IDS.npy", sp.BANK / "FIELD_MANIFEST.json"]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        pytest.skip("real-data smoke test needs the label table, projection and bank: missing " + ", ".join(missing))
    query_pre = _head(engagements, "query_pre", 50)
    order = np.argsort(query_pre, kind="stable")
    query = query_pre[order]
    f_pre = _head(engagements, "F_pre_ms", 50)[order]
    pre_ms = _head(engagements, "pre_ms", 50)[order]
    match_index = _head(engagements, "match_index", 50)[order]
    record = sp.read_query_xy(query, verify_hashes=True)
    assert record.xy.shape == (50, 10, 2) and record.xy.dtype == np.float64
    assert record.support.shape == (50,) and record.support.dtype == bool
    assert record.has_frame.all()                         # every engagement row has its supported pre frame
    assert np.isfinite(record.xy[record.support]).all()
    assert np.array_equal(record.frame_ms, f_pre) and np.array_equal(record.match_index, match_index.astype(np.int64))
    assert np.array_equal(record.query_ms, pre_ms)
    age = pre_ms - record.frame_ms
    assert ((age >= 0) & (age <= 10000)).all()
    team_table = np.load(sp.PROJECTION / "TEAM_IDS.npy", mmap_mode="r", allow_pickle=False)
    assert np.array_equal(record.team_ids, np.asarray(team_table[record.match_index]))
    del team_table
    xy, support = sp.xy_at_queries(query)
    assert np.array_equal(xy, record.xy, equal_nan=True) and np.array_equal(support, record.support)
    rows = record.support
    assert rows.any()
    sigma = sp.fit_sigma(xy[rows])
    assert np.isfinite(sigma) and sigma > 0
    assert np.isfinite(sp.tabular_s200(xy[rows], sigma)[0]).all()
    assert np.isfinite(sp.phi_s_A(xy[rows], record.team_ids[rows], sigma)[0]).all()
    assert np.isfinite(sp.edges(xy[rows], record.team_ids[rows], sigma).vector).all()
