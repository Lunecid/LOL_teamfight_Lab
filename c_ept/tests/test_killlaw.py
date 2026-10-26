"""Tests of ``c_ept.models.killlaw``: the shifted kill law of decision D5 (S4b preparation, task M1).

The law. For a champion mark m (0 .. 49) the team that took the first kill is d = m // 25 and its kill count is K_d = 1 + N_d with N_d ~ Poisson(lambda_d); the other team keeps
K ~ Poisson(mu). The execution marks 50 and 51 keep the plain law for both teams. ``law='poisson'`` is the plain law of every earlier run and must give the bits of the code
before this task.

Every number below that is not a property of the law is a hand value: Poisson(1.3) gives P(K = 1) = 1.3 e^-1.3 = 0.3543, the shifted law with lambda_d = 0.3 gives
P(K_d = 1) = e^-0.3 = 0.7408 and P(K_d = 0) = 0.

What the "default" tests prove (review m5). A test of this file can only compare the code as it is now. Where a test says the default equals the explicit ``'poisson'``, that is
all it says (the keyword changes nothing); where it compares with a copy of the old formula (``old_poisson_head``, the ``_poisson_cells`` expression of ``loss``) or with a
value pinned by an older test (the kill bias of ``test_ept_train``, the notes of ``test_compose``), it is a real comparison with the code before the task. The evidence that
``kill_law='poisson'`` gives the bits of the base commit 68abf41 is the set of comparison scripts of the T012 session folder (``scratchpad\\s4bprep``): ``m1_proof_ept.py``
(loss, gradient, biases, rows digest, identity, a 3-epoch run, frame-averaged rates), ``m1_proof_compose.py`` (closed form, draws, ``_poisson_head``, open draws),
``m1_proof_models.py`` (B0 and B1 saved files, scores, the z array of the M0 test) and ``m1_proof_inference.py`` (every field of ``run_inference``), each run at the base
commit and at the final commit with equal sha256 per digest, and the equivalence acceptance test K8 of the kernel track.
"""
import dataclasses
import gc
import inspect
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.special import gammainc

from c_ept.data import head_rows
from c_ept.domain import units
from c_ept.ept import data, encoder, heads, loss, train
from c_ept.models import baselines_b0, baselines_b1, killlaw, nll as nll_scores
from c_ept.preprocessing import features
from c_ept.tests.ept_toys import build_ept_world, consistent_ept_world, toy_static  # noqa: F401  (the fixtures are imported by ruling R1)
from c_ept import config as C
from c_ept.tests.s3_toys import (KEY_PREFIX, TOY_EXPECTED_V60, TOY_ROW_SPECS, TOY_ROWS, HeadsOracle, ToyHeads, consistent_heads_world,  # noqa: F401
                                 heads_world, toy_composer)
from c_ept.evaluation.adapters import sample_open_value
from c_ept.tests.test_compose import BLUE, MIN_ONSETS, RED, cell_values, rich_composer, z_test_dispersion, z_test_sum
from c_ept.valuation import compose

RNG = np.random.default_rng(20261026)


def old_poisson_head(mu, n_poisson=6):
    """The table of ``compose._poisson_head`` of the code before this task, copied: P(k) for k = 0 .. 5 and P(k >= 6)."""
    p = np.empty(mu.shape + (n_poisson,))
    p[..., 0] = np.exp(-mu)
    for k in range(1, n_poisson):
        p[..., k] = p[..., k - 1] * mu / k
    return p, gammainc(float(n_poisson), mu)


# =====================================================================================================================================
# the first-kill team of a mark
# =====================================================================================================================================
def test_the_laws_and_the_refusal_of_another_name():
    assert killlaw.LAWS == ('poisson', 'shifted')
    assert killlaw.check_law('shifted') == 'shifted'
    for bad in ('Poisson', 'nb', '', None, 1):
        with pytest.raises(ValueError, match='kill_law'):
            killlaw.check_law(bad)


def test_the_first_team_is_the_killer_team_of_the_mark_and_an_execution_has_none():
    for killer in range(1, 11):
        for victim in range(1, 11):
            m = units.first_kill_mark(killer, victim)
            assert killlaw.first_team(np.array([m]))[0] == (0 if killer <= units.N_SLOTS else 1), (killer, victim)
    assert killlaw.first_team(np.array([50, 51])).tolist() == [-1, -1]
    assert killlaw.first_team(np.arange(52)).tolist() == [0] * 25 + [1] * 25 + [-1, -1]
    assert killlaw.N_PAIR_MARKS == units.EXEC_MARK0 and killlaw.PAIRS_PER_TEAM * 2 == units.EXEC_MARK0
    for bad in (np.array([-1]), np.array([52])):
        with pytest.raises(ValueError, match='mark'):
            killlaw.first_team(bad)
    with pytest.raises(TypeError, match='mark'):
        killlaw.first_team(np.array([1.0]))


def test_the_team_swap_of_a_mark_swaps_the_first_team():
    from c_ept.ept import data
    d = killlaw.first_team(np.arange(52))
    swapped = killlaw.first_team(data.MARK_PERM)
    assert np.array_equal(swapped, np.where(d >= 0, 1 - d, d))


# =====================================================================================================================================
# the probability mass: hand values
# =====================================================================================================================================
def test_shifted_logpmf_matches_hand_values():
    k = np.array([[1, 1], [1, 1], [0, 1], [2, 1]])
    rate = np.array([[0.3, 1.3], [1.3, 0.3], [0.3, 1.3], [0.3, 1.3]])
    d = np.array([0, 1, 0, 0])
    got = killlaw.logpmf(k, rate, d, 'shifted')
    assert got.shape == (4, 2)
    # row 0: team 0 is d with lambda 0.3: P(K_d = 1) = e^-0.3 = 0.7408; team 1 plain with mu 1.3: P(K = 1) = 1.3 e^-1.3 = 0.3543
    assert math.isclose(math.exp(got[0, 0]), 0.7408, abs_tol=5e-5) and math.isclose(math.exp(got[0, 0]), math.exp(-0.3), rel_tol=1e-12)
    assert math.isclose(math.exp(got[0, 1]), 0.3543, abs_tol=5e-5) and math.isclose(math.exp(got[0, 1]), 1.3 * math.exp(-1.3), rel_tol=1e-12)
    # row 1: the first team is red: its entry is lambda = 0.3 (K_d = 1 has e^-0.3), blue is plain with mu 1.3
    assert math.isclose(math.exp(got[1, 1]), math.exp(-0.3), rel_tol=1e-12) and math.isclose(math.exp(got[1, 0]), 1.3 * math.exp(-1.3), rel_tol=1e-12)
    # row 2: the first team with no kill is impossible: log P(K_d = 0) = -inf (the plain law would give e^-0.3)
    assert got[2, 0] == -np.inf and np.isfinite(got[2, 1])
    # row 3: K_d = 2 is N_d = 1: 0.3 e^-0.3
    assert math.isclose(math.exp(got[3, 0]), 0.3 * math.exp(-0.3), rel_tol=1e-12)


def test_the_plain_law_gives_the_poisson_hand_value_and_a_zero_count_is_possible():
    got = killlaw.logpmf(np.array([[1, 0]]), np.array([[1.3, 0.3]]), np.array([0]), 'poisson')
    assert math.isclose(math.exp(got[0, 0]), 1.3 * math.exp(-1.3), rel_tol=1e-12) and math.isclose(math.exp(got[0, 1]), math.exp(-0.3), rel_tol=1e-12)
    assert math.isclose(math.exp(got[0, 0]), 0.3543, abs_tol=5e-5)


def test_execution_marks_keep_the_plain_law():
    k = RNG.integers(0, 6, (50, 2))
    rate = RNG.uniform(0.2, 3.0, (50, 2))
    d = np.full(50, -1)
    assert np.array_equal(killlaw.logpmf(k, rate, d, 'shifted'), killlaw.logpmf(k, rate, d, 'poisson'))
    assert np.array_equal(killlaw.shift(d, 'shifted'), np.zeros((50, 2), dtype=np.int64))
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    assert np.array_equal(killlaw.sample(a, rate, d, 'shifted'), killlaw.sample(b, rate, d, 'poisson'))
    assert (killlaw.sample(np.random.default_rng(1), np.full((2000, 2), 0.05), np.full(2000, -1), 'shifted') == 0).any()     # 0 kills happen


def test_the_shift_puts_one_on_the_entry_of_the_first_team_only():
    assert killlaw.shift(np.array([0, 1, -1]), 'shifted').tolist() == [[1, 0], [0, 1], [0, 0]]
    assert killlaw.shift(np.array([0, 1, -1]), 'poisson').tolist() == [[0, 0], [0, 0], [0, 0]]
    assert killlaw.excess(np.array([[3, 2], [1, 4], [0, 1]]), np.array([0, 1, -1]), 'shifted').tolist() == [[2, 2], [1, 3], [0, 1]]
    assert killlaw.excess(np.array([[3, 2]]), np.array([0]), 'poisson').tolist() == [[3, 2]]
    with pytest.raises(ValueError, match='1 row'):
        killlaw.excess(np.array([[0, 2], [1, 1]]), np.array([0, 0]), 'shifted')              # a first team without a kill: counted, no match id
    with pytest.raises(ValueError, match='first team'):
        killlaw.shift(np.array([2]), 'shifted')


# =====================================================================================================================================
# the sample
# =====================================================================================================================================
def test_shifted_sample_never_gives_zero_to_the_first_team():
    n = 40_000
    m = RNG.integers(0, 50, n)
    d = killlaw.first_team(m)
    rate = np.column_stack([np.full(n, 0.05), np.full(n, 0.05)])
    k = killlaw.sample(np.random.default_rng(7), rate, d, 'shifted')
    assert k.shape == (n, 2) and k.dtype == np.int64
    assert (k[np.arange(n), d] >= 1).all()                                          # every first team has a kill
    assert (k[np.arange(n), 1 - d] == 0).any()                                      # the other team may have none
    # the excess has the mean lambda = 0.05 (the plain law for the other team too)
    assert abs((k[np.arange(n), d] - 1).mean() - 0.05) < 0.005 and abs(k[np.arange(n), 1 - d].mean() - 0.05) < 0.005


def test_the_sample_uses_one_poisson_call_of_the_same_shape_so_the_random_stream_is_that_of_the_plain_law():
    rate = RNG.uniform(0.1, 2.5, (300, 2))
    d = killlaw.first_team(RNG.integers(0, 52, 300))
    a, b = np.random.default_rng(11), np.random.default_rng(11)
    shifted = killlaw.sample(a, rate, d, 'shifted')
    plain = b.poisson(rate).astype(np.int64)
    assert np.array_equal(shifted, plain + killlaw.shift(d, 'shifted'))
    assert np.array_equal(a.random(25), b.random(25))                              # the next numbers of the stream are the same


# =====================================================================================================================================
# the tail and the table of the closed form
# =====================================================================================================================================
def test_the_tail_of_the_shifted_law_is_the_tail_of_the_excess():
    rate = np.array([[0.7, 1.9], [2.4, 0.2]])
    d = np.array([0, 1])
    got = killlaw.tail_ge(6, rate, d, 'shifted')
    assert got[0, 0] == gammainc(5.0, 0.7) and got[0, 1] == gammainc(6.0, 1.9)    # P(K_d >= 6) = P(N_d >= 5); the plain team P(K >= 6)
    assert got[1, 1] == gammainc(5.0, 0.2) and got[1, 0] == gammainc(6.0, 2.4)
    assert (killlaw.tail_ge(1, rate, d, 'shifted')[[0, 1], [0, 1]] == 1.0).all()   # P(K_d >= 1) = 1
    assert np.allclose(killlaw.tail_ge(2, rate, d, 'shifted')[[0, 1], [0, 1]], 1.0 - np.exp(-rate[[0, 1], [0, 1]]), rtol=1e-12)   # P(N_d >= 1)


def test_the_table_of_the_shifted_law_is_the_plain_table_moved_by_one_for_the_first_team_and_adds_up_to_one():
    mu = RNG.uniform(0.05, 4.0, (200, 2))
    d = RNG.integers(-1, 2, 200)
    p, tail = killlaw.pmf_head(mu, d, 'shifted')
    plain, plain_tail = old_poisson_head(mu)
    assert p.shape == (200, 2, 6) and tail.shape == (200, 2)
    assert np.allclose(p.sum(axis=-1) + tail, 1.0, rtol=0, atol=1e-12)
    for i in range(200):
        for t in (0, 1):
            if d[i] == t:
                assert p[i, t, 0] == 0.0 and np.array_equal(p[i, t, 1:], plain[i, t, :5]) and tail[i, t] == gammainc(5.0, mu[i, t])
            else:
                assert np.array_equal(p[i, t], plain[i, t]) and tail[i, t] == plain_tail[i, t]


# =====================================================================================================================================
# torch and numpy
# =====================================================================================================================================
def test_torch_and_numpy_logpmf_agree():
    n = 500
    k = RNG.integers(0, 7, (n, 2))
    rate = RNG.uniform(0.02, 6.0, (n, 2))
    d = RNG.integers(-1, 2, n)
    for law in killlaw.LAWS:
        got = killlaw.torch_logpmf(torch.as_tensor(k), torch.as_tensor(rate), torch.as_tensor(d), law).numpy()
        want = killlaw.logpmf(k, rate, d, law)
        assert np.array_equal(np.isinf(got), np.isinf(want)) and np.array_equal(got[np.isinf(got)], want[np.isinf(want)])
        assert np.allclose(got[np.isfinite(got)], want[np.isfinite(want)], rtol=1e-12, atol=1e-12), law
    assert np.isinf(killlaw.logpmf(k, rate, d, 'shifted')).any() and not np.isinf(killlaw.logpmf(k, rate, d, 'poisson')).any()


def test_the_torch_score_of_an_impossible_cell_is_minus_infinity_and_the_gradient_of_the_other_cells_stays_finite():
    rate = torch.tensor([[0.3, 1.3], [0.4, 0.9]], dtype=torch.float64, requires_grad=True)
    k = torch.tensor([[0, 1], [2, 3]])
    logp = killlaw.torch_logpmf(k, rate, torch.tensor([0, 1]), 'shifted')
    assert logp[0, 0] == -float('inf') and torch.isfinite(logp[0, 1]) and torch.isfinite(logp[1, 0]) and torch.isfinite(logp[1, 1])
    logp[torch.isfinite(logp)].sum().backward()
    assert torch.isfinite(rate.grad).all() and rate.grad[0, 0] == 0.0               # the impossible cell passes no gradient (and no NaN)


def test_the_torch_first_team_equals_the_numpy_one_and_refuses_what_numpy_refuses():
    m = torch.arange(52)
    assert np.array_equal(killlaw.torch_first_team(m).numpy(), killlaw.first_team(np.arange(52)))
    with pytest.raises(ValueError, match='mark'):
        killlaw.torch_first_team(torch.tensor([52]))
    with pytest.raises(TypeError, match='mark'):
        killlaw.torch_first_team(torch.tensor([1.0]))


# =====================================================================================================================================
# the default law at the level of the functions: against copies of the code before this task (see the module text for the cross-version evidence)
# =====================================================================================================================================
def test_poisson_default_is_bit_identical():
    n = 400
    k = RNG.integers(0, 8, (n, 2))
    rate = RNG.uniform(0.0, 5.0, (n, 2))
    rate[::50] = 0.0
    k[::50] = 0
    d = RNG.integers(-1, 2, n)
    # the score: the minus of nll.poisson with the exposure 1 (ruling P10), whatever d is
    want = -nll_scores.poisson(k.astype(np.float64), rate, np.ones((n, 1)))
    for dd in (d, None):
        assert np.array_equal(killlaw.logpmf(k, rate, dd, 'poisson'), want)
    # the sample: one poisson call of the shape of the rates, int64
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    for dd in (d, None):
        assert np.array_equal(killlaw.sample(a, rate, dd, 'poisson'), b.poisson(rate).astype(np.int64))
    # the tail P(K >= 6) and the table P(K = k), k = 0 .. 5 of the closed form
    p, tail = killlaw.pmf_head(rate, d, 'poisson')
    old_p, old_tail = old_poisson_head(rate)
    assert np.array_equal(p, old_p) and np.array_equal(tail, old_tail)
    assert np.array_equal(killlaw.tail_ge(6, rate, d, 'poisson'), gammainc(6.0, rate))
    assert np.array_equal(killlaw.pmf_head(rate, None, 'poisson')[0], old_p)
    # the excess and the shift of the plain law are the counts themselves
    assert np.array_equal(killlaw.excess(k, d, 'poisson'), k)


# =====================================================================================================================================
# the rows: the first team of the open kill rows and the assertion K_d >= 1 (Review Focus 2)
# =====================================================================================================================================
ALL_HEADS = [(name, 'masked') for name in head_rows.HEAD_NAMES] + [(name, 'open') for name in head_rows.OPEN_HEAD_NAMES]


def rows_of(world, role='FIT', law=None):
    """``head_rows.HeadRows`` of the world with the b0 builder; ``law`` None leaves the keyword out (the default of the code)."""
    kw = {} if law is None else {'kill_law': law}
    return head_rows.HeadRows(world.s1, world.s2, world.labels, role, lambda s2, views: features.b0(s2, views, world.fit_means), **kw)


def n_first_team_without_a_kill(oracle, unit_ids):
    """The units whose champion mark names a team with no kill, counted by a plain loop from the text of the rule (K_d >= 1)."""
    return sum(1 for u in unit_ids if oracle.mark[u] < 50 and oracle.K[u, oracle.mark[u] // 25] == 0)


def test_rows_assert_the_first_kill_team_scored(heads_world):
    """The random heads world has marks that do not follow the kills, so the shifted law finds units with probability 0 and the rows refuse them, with a
    count and never a match key; the plain law and the other heads read nothing of the rule."""
    oracle = HeadsOracle(heads_world)
    n_open = n_first_team_without_a_kill(oracle, oracle.units('FIT'))
    n_masked = n_first_team_without_a_kill(oracle, [oracle.on_unit[i] for i in oracle.onsets('FIT')])
    assert n_open > 0 and n_masked > 0                                       # the control: the world does break the rule
    shifted = rows_of(heads_world, law='shifted')
    with pytest.raises(ValueError, match=rf'{n_masked} onset\(s\)') as raised:
        shifted.parts('M3Kills', 'masked')
    assert KEY_PREFIX not in str(raised.value) and 'first kill' in str(raised.value)
    with pytest.raises(ValueError, match=rf'{n_open} unit\(s\)') as raised:
        shifted.parts('M3Kills', 'open')
    assert KEY_PREFIX not in str(raised.value)
    for head, kind in ALL_HEADS:
        if head != 'M3Kills':
            shifted.parts(head, kind, masked_obj_k=True)                     # no other head is affected
    for law in (None, 'poisson'):
        plain = rows_of(heads_world, law=law)
        plain.parts('M3Kills', 'masked')
        plain.parts('M3Kills', 'open')                                       # the plain law accepts what it always accepted


def test_a_world_whose_marks_follow_the_kills_passes_the_assertion(consistent_heads_world):
    oracle = HeadsOracle(consistent_heads_world)
    assert n_first_team_without_a_kill(oracle, range(len(oracle.mark))) == 0
    for role in ('FIT', 'STOP', 'DEV'):
        shifted = rows_of(consistent_heads_world, role=role, law='shifted')
        shifted.parts('M3Kills', 'masked')
        shifted.parts('M3Kills', 'open')


def test_open_rows_carry_d(consistent_heads_world):
    world = consistent_heads_world
    oracle = HeadsOracle(world)
    unit_ids = oracle.units('FIT')
    shifted = rows_of(world, law='shifted')
    p = shifted.parts('M3Kills', 'open')
    want = [-1 if oracle.mark[u] >= 50 else int(oracle.mark[u]) // 25 for u in unit_ids]
    assert p.codes['d'].tolist() == want and p.codes['d'].dtype == np.int64 and p.first_team.tolist() == want
    assert {-1, 0, 1} <= set(want)                                           # an execution start (-1), a blue and a red first kill all occur
    assert set(p.codes) == {'s', 'd'} and p.y.tolist() == [oracle.K[u, :2].tolist() for u in unit_ids]
    assert all(d < 0 or oracle.K[u, d] >= 1 for u, d in zip(unit_ids, want))
    # the masked rows get their first team from the mark, which is a code of the rows already: no new code
    mp = shifted.parts('M3Kills', 'masked')
    assert set(mp.codes) == {'q', 'm', 'z', 's'} and mp.first_team.dtype == np.int64
    assert np.array_equal(mp.first_team, np.where(mp.codes['m'] < 50, mp.codes['m'] // 25, -1))
    # a head other than M3Kills has no first team
    assert shifted.parts('M3Gold', 'open').first_team is None and shifted.parts('M3Obj', 'masked').first_team is None
    assert 'd' not in shifted.parts('M3Gold', 'open').codes


def test_the_default_rows_equal_the_rows_of_the_explicit_plain_law_and_carry_no_first_team(consistent_heads_world):
    a, b = rows_of(consistent_heads_world), rows_of(consistent_heads_world, law='poisson')
    for head, kind in ALL_HEADS:
        x, y = a.parts(head, kind, masked_obj_k=True), b.parts(head, kind, masked_obj_k=True)
        assert x.first_team is None and y.first_team is None and 'd' not in x.codes and set(x.codes) == set(y.codes), (head, kind)
        for name in head_rows.HeadParts._fields:
            u, v = getattr(x, name), getattr(y, name)
            if isinstance(u, dict):
                assert all(np.array_equal(u[k], v[k]) for k in u), (head, kind, name)
            elif isinstance(u, np.ndarray):
                assert np.array_equal(u, v, equal_nan=True) and u.dtype == v.dtype, (head, kind, name)
            else:
                assert u == v, (head, kind, name)
    assert a.kill_law == 'poisson' and rows_of(consistent_heads_world, law='shifted').kill_law == 'shifted'


def test_a_name_that_is_no_kill_law_is_refused_before_any_array_is_read(consistent_heads_world):
    class NoRead:
        def array(self, name):
            raise AssertionError('an array was read')
    with pytest.raises(ValueError, match='kill_law'):
        head_rows.HeadRows(NoRead(), NoRead(), NoRead(), 'FIT', None, kill_law='negative_binomial')


def test_shifted_engagement_rows_carry_d_in_the_open_kill_block_and_nowhere_else(consistent_ept_world):
    world = consistent_ept_world
    key = ('M3Kills', 'open')
    assert data.ROW_FIELDS[key] == ('pos', 'match', 'row', 'unit', 's', 'y')            # the required fields do not change: the default batches are the old ones
    assert dict(data.OPTIONAL_ROW_FIELDS) == {key: ('d',)}
    plain = data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, 'FIT'))
    shifted = data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, 'FIT', kill_law='shifted'))
    assert shifted.head_rows.kill_law == 'shifted' and plain.head_rows.kill_law == 'poisson'
    for block in plain.blocks:
        fields = set(plain.blocks[block].fields)
        assert set(shifted.blocks[block].fields) == fields | ({'d'} if block == key else set()), block
        assert all(np.array_equal(shifted.blocks[block].fields[f], plain.blocks[block].fields[f]) for f in fields), block
    d = shifted.blocks[key].fields['d']
    oracle = HeadsOracle(world.base)
    assert d.dtype == np.int64 and d.tolist() == [-1 if oracle.mark[u] >= 50 else int(oracle.mark[u]) // 25
                                                  for u in shifted.blocks[key].fields['row']]


def test_the_batch_holds_d_and_the_swap_moves_it_with_the_teams(consistent_ept_world):
    world = consistent_ept_world
    rows = data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, 'FIT', kill_law='shifted'))
    plain_rows = data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, 'FIT'))
    name = 'M3Kills/open/d'
    seen = 0
    for batch, old in zip(data.Batches(rows, 4, 1).epoch(0), data.Batches(plain_rows, 4, 1).epoch(0)):
        assert name in batch and name not in old
        assert set(data.block(batch, 'M3Kills', 'open')) == {'pos', 'match', 'row', 'unit', 's', 'y', 'd'}
        assert set(data.block(old, 'M3Kills', 'open')) == {'pos', 'match', 'row', 'unit', 's', 'y'}
        swapped = data.swap(batch)
        d = batch[name]
        assert torch.equal(swapped[name], torch.where(d >= 0, 1 - d, d)) and torch.equal(data.swap(swapped)[name], d)
        y, sy = batch['M3Kills/open/y'], swapped['M3Kills/open/y']
        assert torch.equal(sy, y.flip(-1))                                                  # the kills of the teams change places with d
        first = d >= 0
        assert (sy[first.nonzero()[:, 0], swapped[name][first]] >= 1).all()               # the first team of the swapped rows still has a kill
        seen += int(first.sum())
    assert seen > 0
    assert data.FIELD_SWAP.keys() == {(h, k, f) for (h, k), fields in data.ROW_FIELDS.items() for f in fields}      # the table of the required fields is unchanged


def test_the_b0_open_kill_head_has_the_first_team_block_only_under_the_shifted_law(consistent_heads_world):
    world = consistent_heads_world
    key = ('M3Kills', 'open')
    plain = baselines_b0.layout_for(*key)
    assert plain is baselines_b0.LAYOUTS[key] and baselines_b0.layout_for(*key, 'poisson') is plain
    shifted = baselines_b0.layout_for(*key, 'shifted')
    assert [b.name for b in shifted.blocks] == ['s', 'd'] and shifted.blocks[1] == baselines_b0.Block('d', -1, 3, True)
    assert shifted.n_numeric == plain.n_numeric and shifted.width == plain.width + 2
    assert baselines_b0.column_names(shifted)[-2:] == ['d=0', 'd=1']
    for other in baselines_b0.LAYOUTS:
        if other != key:
            assert baselines_b0.layout_for(*other, 'shifted') is baselines_b0.LAYOUTS[other], other
    with pytest.raises(ValueError, match='kill_law'):
        baselines_b0.layout_for(*key, 'nb')
    rows = baselines_b0.HeadRows(world.s1, world.s2, world.labels, world.fit_means, 'FIT', kill_law='shifted')
    d = rows.data(*key)
    p = rows.parts(*key)
    assert d.layout == shifted and d.X.shape == (len(p.y), shifted.width) and np.array_equal(d.first_team, p.first_team)
    one_hot = d.X[:, plain.width:plain.width + 2]
    assert np.array_equal(one_hot, np.column_stack([p.first_team == 0, p.first_team == 1]).astype(np.float64))      # d = -1 is the dropped first level
    old = baselines_b0.HeadRows(world.s1, world.s2, world.labels, world.fit_means, 'FIT').data(*key)
    assert old.layout == plain and old.first_team is None and old.X.shape[1] == plain.width


def test_the_b1_open_kill_head_has_the_first_team_slot_only_under_the_shifted_law(consistent_heads_world):
    world = consistent_heads_world
    key = ('M3Kills', 'open')
    cols = features.b1_columns(())
    plain, shifted = baselines_b1.make_layout(*key, cols), baselines_b1.make_layout(*key, cols, 'shifted')
    assert plain.slots == ('feat', 's') and shifted.slots == ('feat', 's', 'd')
    assert shifted.names == plain.names + ('d',) and shifted.categorical == plain.categorical + (len(cols) + 1,) and shifted.d == plain.d
    assert baselines_b1.SLOTS[key] == ('feat', 's')                                           # the table of the default designs is unchanged
    for other in baselines_b1.SLOTS:
        if other != key:
            assert baselines_b1.make_layout(*other, cols, 'shifted') == baselines_b1.make_layout(*other, cols), other
    rows = baselines_b1.HeadRows(world.s1, world.s2, world.labels, (), 'FIT', kill_law='shifted')
    dd = rows.data(*key)
    p = rows.parts(*key)
    assert dd.layout == shifted and dd.X.shape == (len(p.y), shifted.width) and dd.X.dtype == np.float32
    assert np.array_equal(dd.X[:, -1], (p.first_team + 1).astype(np.float32)) and np.array_equal(dd.first_team, p.first_team)
    old = baselines_b1.HeadRows(world.s1, world.s2, world.labels, (), 'FIT').data(*key)
    assert old.layout == plain and old.first_team is None
    assert np.array_equal(old.X, dd.X[:, :-1], equal_nan=True)                                # the rest of the design is the old design


# =====================================================================================================================================
# the EPT: the loss, the start bias of the kill head, the settings and the identity of a run (commit 3)
# =====================================================================================================================================
def kill_block(masked: bool, n: int = 40, seed: int = 1, executions_only: bool = False):
    """A made-up block of M3Kills rows whose first team has a kill: the target (a masked block holds the mark ``m``, an open block the first team ``d``), the marks, the
    first teams, the counts and the sizes."""
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 4, n)
    m = rng.integers(50, 52, n) if executions_only else rng.integers(0, 52, n)
    d = killlaw.first_team(m)
    y = rng.integers(0, 5, (n, 2))
    first = np.flatnonzero(d >= 0)
    y[first, d[first]] += 1                                                         # the first team has at least one kill
    target = {'y': torch.as_tensor(y), 's': torch.as_tensor(s), ('m' if masked else 'd'): torch.as_tensor(m if masked else d)}
    return target, m, d, y, s


def kill_rates(n: int = 40, seed: int = 2, dtype=torch.float32):
    return torch.as_tensor(np.random.default_rng(seed).uniform(0.1, 3.0, (n, 4, 2)), dtype=dtype)


def test_the_shifted_torch_score_of_the_kill_head_equals_the_numpy_score_masked_and_open():
    for masked in (True, False):
        target, m, d, y, s = kill_block(masked)
        out = kill_rates()
        rate = out.double().numpy()[np.arange(len(s)), s]
        got = loss.head_nll('M3Kills', out, target, kill_law='shifted').numpy()
        assert got.dtype == np.float64 and np.isfinite(got).all()
        np.testing.assert_allclose(got, -killlaw.logpmf(y, rate, d, 'shifted').sum(axis=1), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(loss.head_nll('M3Kills', out, target).numpy(), -killlaw.logpmf(y, rate, d, 'poisson').sum(axis=1), rtol=1e-12, atol=1e-12)
        assert (got != loss.head_nll('M3Kills', out, target).numpy()).any()         # the law changes the score of a champion mark


def test_the_shifted_kill_score_of_one_row_is_the_hand_value():
    """Mark 3 (blue took the first kill), kills (1, 1), rates (lambda_blue, mu_red) = (0.3, 1.3): -log P = 0.3 [N = 0] + (1.3 - log 1.3) [K = 1] = 1.3376."""
    out = torch.zeros(1, 4, 2, dtype=torch.float64)
    out[0, 0] = torch.tensor([0.3, 1.3], dtype=torch.float64)
    got = loss.head_nll('M3Kills', out, {'y': torch.tensor([[1, 1]]), 's': torch.tensor([0]), 'm': torch.tensor([3])}, kill_law='shifted')
    assert got.item() == pytest.approx(0.3 + 1.3 - math.log(1.3), rel=1e-12)
    open_ = loss.head_nll('M3Kills', out, {'y': torch.tensor([[1, 1]]), 's': torch.tensor([0]), 'd': torch.tensor([0])}, kill_law='shifted')
    assert open_.item() == got.item()                                               # the open and the masked block of one unit give one score
    red = loss.head_nll('M3Kills', out.flip(-1), {'y': torch.tensor([[1, 1]]), 's': torch.tensor([0]), 'd': torch.tensor([1])}, kill_law='shifted')
    assert red.item() == got.item()                                                 # the team swap: red first with the rates swapped


def test_the_shifted_kill_score_gradient_is_one_minus_the_excess_over_the_rate_for_both_teams():
    rate = torch.tensor([[[0.8, 2.5]]], dtype=torch.float64, requires_grad=True)     # (1 row, size 0 only below)
    out = rate.expand(1, 4, 2) * 1.0
    target = {'y': torch.tensor([[3, 2]]), 's': torch.tensor([0]), 'd': torch.tensor([0])}
    loss.head_nll('M3Kills', out, target, kill_law='shifted').sum().backward()
    # blue is the first team: d(lambda - (k - 1) log lambda) = 1 - (3 - 1) / 0.8 ; red is plain: 1 - 2 / 2.5
    assert rate.grad[0, 0].tolist() == pytest.approx([1 - 2 / 0.8, 1 - 2 / 2.5], rel=1e-12)


def test_a_first_team_without_a_kill_scores_infinity_and_passes_no_nan_to_the_gradient():
    _, m, d, y, s = kill_block(False, n=6)
    d[0], y[0, 0] = 0, 0                                                            # row 0: blue first, no blue kill
    target = {'y': torch.as_tensor(y), 's': torch.as_tensor(s), 'd': torch.as_tensor(d)}
    out = kill_rates(6).requires_grad_()
    nll = loss.head_nll('M3Kills', out, target, kill_law='shifted')
    assert nll[0].item() == float('inf')
    finite = torch.isfinite(nll)
    assert finite.sum() >= 1
    nll[finite].sum().backward()
    assert torch.isfinite(out.grad).all() and (out.grad[0] == 0).all()


def test_the_shifted_kill_score_needs_the_mark_or_the_first_team_and_ignores_the_law_for_other_heads():
    target, m, d, y, s = kill_block(True)
    out = kill_rates()
    with pytest.raises(ValueError, match="'m' .* 'd'"):
        loss.head_nll('M3Kills', out, {'y': target['y'], 's': target['s']}, kill_law='shifted')
    with pytest.raises(ValueError, match='kill_law'):
        loss.head_nll('M3Kills', out, target, kill_law='nb')
    mu = torch.as_tensor(np.random.default_rng(3).uniform(0.2, 2.0, (5, 18)), dtype=torch.float32)
    obj = {'s': torch.zeros(5, dtype=torch.long), 'y': torch.zeros(5, 18, dtype=torch.long)}
    assert torch.equal(loss.head_nll('M3Obj', mu, obj, kill_law='shifted'), loss.head_nll('M3Obj', mu, obj))
    # executions start the unit: the plain law for both teams, whatever the law is
    target, *_ = kill_block(True, executions_only=True)
    assert torch.equal(loss.head_nll('M3Kills', out, target, kill_law='shifted'), loss.head_nll('M3Kills', out, target))


def test_the_default_kill_score_is_the_old_formula_bit_for_bit_and_the_default_argument_is_poisson():
    for masked in (True, False):
        target, m, d, y, s = kill_block(masked)
        out = kill_rates()
        mu, yd = out[torch.arange(len(s)), target['s']].double(), target['y'].double()
        log_mu = torch.log(torch.where(yd > 0, mu, torch.ones_like(mu)))                # the cells of ``_poisson_cells`` as they were before this task
        want = (mu - yd * log_mu + torch.lgamma(yd + 1.0)).sum(-1)
        assert torch.equal(loss.head_nll('M3Kills', out, target), want)
        assert torch.equal(loss.head_nll('M3Kills', out, target, kill_law='poisson'), want)
        assert torch.equal(loss.head_nll('M3Kills', out, {'y': target['y'], 's': target['s']}), want)    # the plain law needs neither m nor d
    assert inspect.signature(loss.head_nll).parameters['kill_law'].default == 'poisson'


def stub_kill_rows(**given):
    from c_ept.tests.test_ept_train import stub_rows
    return stub_rows(**given)


def test_shifted_kill_bias_is_log_mean_of_the_excess():
    """Size 0: masked (mark 3 = blue first) y = (2, 1); masked (mark 30 = red first) y = (0, 3); open (blue first) y = (1, 2). The effective counts (the excess for the first
    team, the count for the other) are (1, 1), (0, 2), (0, 2): means (1/3, 5/3). The first team of every row is the 'd entry' of that row; the bias of a team slot is the
    log mean over all the rows of its effective count, which is the maximum-likelihood constant of the slot under the shifted law."""
    rows = stub_kill_rows(M3Kills_masked=dict(s=[0, 0], m=[3, 30], y=[[2, 1], [0, 3]], match=[1, 1]), M3Kills_open=dict(s=[0], d=[0], y=[[1, 2]], match=[1]))
    got, info = train.fit_biases(rows, kill_law='shifted')
    np.testing.assert_allclose(got['kill_out.bias'], np.log([1 / 3, 5 / 3] * 4), rtol=1e-12)           # sizes 1 .. 3 take the team's mean over all sizes
    assert info['kill_out.bias']['kill_law'] == 'shifted' and 'excess' in info['kill_out.bias']['kill_law_note']
    # every row has blue first: the bias of the blue slot is log mean(y - 1), that of the red slot log mean(y)
    y = np.array([[3, 2], [1, 0], [2, 4]])
    rows = stub_kill_rows(M3Kills_masked=dict(s=[1, 1], m=[0, 24], y=y[:2], match=[1, 1]), M3Kills_open=dict(s=[1], d=[0], y=y[2:], match=[1]))
    got, _ = train.fit_biases(rows, kill_law='shifted')
    np.testing.assert_allclose(got['kill_out.bias'][2:4], [math.log((y[:, 0] - 1).mean()), math.log(y[:, 1].mean())], rtol=1e-12)
    # an execution start is the plain law for both teams, and the floor keeps a slot without an excess finite
    rows = stub_kill_rows(M3Kills_masked=dict(s=[2, 2], m=[50, 51], y=[[1, 0], [0, 2]], match=[1, 1]), M3Kills_open=dict(s=[2], d=[-1], y=[[0, 0]], match=[1]))
    got, _ = train.fit_biases(rows, kill_law='shifted')
    np.testing.assert_allclose(got['kill_out.bias'][4:6], np.log([1 / 3, 2 / 3]), rtol=1e-12)
    rows = stub_kill_rows(M3Kills_masked=dict(s=[3], m=[0], y=[[1, 0]], match=[1]))
    got, _ = train.fit_biases(rows, kill_law='shifted')
    assert got['kill_out.bias'][6] == math.log(train.RATE_FLOOR) and got['kill_out.bias'][7] == math.log(train.RATE_FLOOR) and np.isfinite(got['kill_out.bias']).all()


def test_the_default_kill_bias_is_the_pinned_hand_value_and_equals_the_explicit_plain_law_with_its_record():
    rows = stub_kill_rows(M3Kills_masked=dict(s=[0, 0, 2], y=[[1, 2], [3, 0], [2, 2]], match=[1, 1, 2]), M3Kills_open=dict(s=[2], y=[[1, 0]], match=[1]))
    a, info_a = train.fit_biases(rows)
    b, info_b = train.fit_biases(rows, kill_law='poisson')
    np.testing.assert_array_equal(a['kill_out.bias'], np.log([2.0, 1.0, 1.75, 1.0, 1.5, 1.0, 1.75, 1.0]))      # the hand value of test_ept_train
    assert all(np.array_equal(a[k], b[k]) for k in a) and info_a == info_b
    assert 'kill_law' not in info_a['kill_out.bias'] and 'kill_law_note' not in info_a['kill_out.bias']


def test_the_shifted_kill_bias_needs_the_first_team_of_the_rows_and_refuses_another_law_name():
    with pytest.raises(ValueError, match="'d'"):
        train.fit_biases(stub_kill_rows(M3Kills_masked=dict(s=[0], m=[3], y=[[2, 1]], match=[1]), M3Kills_open=dict(s=[0], y=[[1, 2]], match=[1])), kill_law='shifted')
    with pytest.raises(ValueError, match='kill_law'):
        train.fit_biases(stub_kill_rows(), kill_law='nb')


def test_the_train_config_has_the_kill_law_after_model_spec_with_the_default_law_and_refuses_another_name():
    names = [f.name for f in dataclasses.fields(train.TrainConfig)]
    # kill_law follows model_spec directly; the fields of later tasks (k_condition, S4b preparation M3) come after it, so no positional call can change
    assert names[5:7] == ['model_spec', 'kill_law'] and names[:6] == ['encoder', 'lr', 'weight_decay', 'warmup_steps', 'patience', 'model_spec']
    cfg = train.TrainConfig(encoder.EncoderConfig(d=16, layers=1, heads=2), 1e-3, 0.0, 1)
    assert cfg.kill_law == 'poisson' and train.TrainConfig(encoder.EncoderConfig(d=16, layers=1, heads=2), 1e-3, 0.0, 1, 2, None).kill_law == 'poisson'
    assert train.TrainConfig(encoder.EncoderConfig(d=16, layers=1, heads=2), 1e-3, 0.0, 1, kill_law='shifted').kill_law == 'shifted'
    with pytest.raises(ValueError, match='kill_law'):
        train.TrainConfig(encoder.EncoderConfig(d=16, layers=1, heads=2), 1e-3, 0.0, 1, kill_law='nb')


@pytest.fixture(scope='module')
def run_world(tmp_path_factory):
    """The consistent EPT toy world of the training tests (7 events, 5 edges) with FIT statistics."""
    folder = tmp_path_factory.mktemp('killlaw_train')
    world = build_ept_world(folder, n_events=7, n_edges=5, consistent_first_kill=True)
    yield world, data.fit_stats(world.s1, world.s2)
    del world
    gc.collect()


def run_data(world, stats, law):
    rows = {role: data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, role, kill_law=law)) for role in ('FIT', 'STOP')}
    fit = data.Batches(rows['FIT'], max(1, math.ceil(len(rows['FIT'].units) / 3)), 7, stats=stats, training=True)
    stop = data.Batches(rows['STOP'], max(1, len(rows['STOP'].units)), 7, stats=stats, training=False)
    scales = {key: 1.0 + 0.25 * i for i, key in enumerate(loss.BLOCKS)}
    return train.TrainData(fit=fit, stop=stop, scales=scales, static=toy_static())


def run_cfg(law, lr=3e-3):
    return train.TrainConfig(encoder=encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.1, kappas=1, field_channels=2), lr=lr, weight_decay=0.01,
                             warmup_steps=2, patience=10, kill_law=law)


def run_train(folder, world, stats, cfg, data_law, epochs=2):
    return train.train(cfg, run_data(world, stats, data_law), folder, device='cpu', epochs=epochs, check=lambda: None, seed=3)


def test_a_shifted_run_records_the_law_in_its_identity_starts_at_the_shifted_bias_and_trains(run_world, tmp_path):
    world, stats = run_world
    shifted = run_train(tmp_path / 'shifted', world, stats, run_cfg('shifted'), 'shifted')
    plain = run_train(tmp_path / 'plain', world, stats, run_cfg('poisson'), 'poisson')
    assert shifted.head['identity']['kill_law'] == 'shifted' and 'kill_law' not in plain.head['identity']             # absent means the plain law
    assert shifted.head['bias_init']['kill_out.bias']['kill_law'] == 'shifted' and 'kill_law' not in plain.head['bias_init']['kill_out.bias']
    assert np.isfinite(shifted.stop_losses).all() and np.isfinite(shifted.fit_batch_mean_losses).all() and len(shifted.stop_losses) == 2
    assert shifted.stop_losses != plain.stop_losses                                                                   # the law is not a label
    # with the learning rate 0 the weights after the run are the starting values: the kill bias is the shifted FIT bias (and not the plain one)
    run_train(tmp_path / 'frozen', world, stats, run_cfg('shifted', lr=0.0), 'shifted', epochs=1)
    rows = data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, 'FIT', kill_law='shifted'))
    want, _ = train.fit_biases(rows, kill_law='shifted')
    plain_want, _ = train.fit_biases(rows)
    start = torch.load(tmp_path / 'frozen' / 'epoch_0001' / 'last.pt', weights_only=True)['model']['heads.kill_out.bias'].numpy()
    np.testing.assert_allclose(start, want['kill_out.bias'].astype(np.float32), rtol=1e-6)
    assert not np.allclose(want['kill_out.bias'], plain_want['kill_out.bias'], rtol=1e-3)


def test_the_law_of_the_rows_and_the_law_of_the_settings_must_agree_before_any_work(run_world, tmp_path):
    world, stats = run_world
    with pytest.raises(ValueError, match='kill_law'):
        run_train(tmp_path / 'a', world, stats, run_cfg('shifted'), 'poisson')
    with pytest.raises(ValueError, match='kill_law'):
        run_train(tmp_path / 'b', world, stats, run_cfg('poisson'), 'shifted')
    assert not (tmp_path / 'a').exists() or not list((tmp_path / 'a').glob('epoch_*'))


def test_a_run_cannot_be_resumed_under_the_other_kill_law_in_either_direction(run_world, tmp_path):
    world, stats = run_world
    for first, second in (('shifted', 'poisson'), ('poisson', 'shifted')):
        run_train(tmp_path / first, world, stats, run_cfg(first), first, epochs=2)
        with pytest.raises(ValueError, match='kill_law'):                           # the message of ``_check_identity`` lists the keys that differ
            run_train(tmp_path / first, world, stats, run_cfg(second), second, epochs=2)


def test_the_identity_check_names_the_kill_law_when_it_is_the_one_key_that_differs():
    """The rows digest differs between the laws as well, so the run test above could pass without the identity key. Here the key is the only difference: the key is absent in the
    identity of a plain run and ``_check_identity`` compares the union of the keys, so an identity with the key is refused against one without it, and the other way round."""
    plain, shifted = {'seed': 3, 'epochs': 2}, {'seed': 3, 'epochs': 2, 'kill_law': 'shifted'}
    for saved, now in ((plain, shifted), (shifted, plain)):
        with pytest.raises(ValueError, match=r"\['kill_law'\] differ"):
            train._check_identity(saved, now)
    train._check_identity(shifted, dict(shifted))
    train._check_identity(plain, dict(plain))


class FirstTeamStub(torch.nn.Module):
    """A model whose M3 head gives the kill rate the meaning of the shifted law for the mark it is asked about: the entry of the first team is the rate lambda of the
    excess (0.3 + 0.1 s) and the other entry the rate mu of the count (1.4 + 0.2 s), an execution has mu in both. It reads only the condition m, as the head of the masked
    reading does, so the two frames of the frame average are asked about the mark and the swapped mark."""

    def __init__(self):
        super().__init__()
        self.heads = SimpleNamespace(m3=self.m3)

    def encode(self, batch):
        n = batch['side'].shape[0]
        return encoder.Encoded(torch.zeros(n, 55, 4), batch['side'][:, None].expand(n, 4).clone(), torch.ones(n, 55, dtype=torch.bool))

    def m3(self, enc, q, m, z, pos=None):
        n = len(m)
        d, size = killlaw.torch_first_team(m), torch.arange(4, dtype=torch.float32)
        lam, mu = 0.3 + 0.1 * size, 1.4 + 0.2 * size
        rate = mu[None, :, None].expand(n, 4, 2).clone()
        for team in (0, 1):
            rate[d == team, :, team] = lam
        return heads.M3Out(torch.zeros(n, 4), rate, torch.zeros(n, 4, 2, 6, 31))


def test_the_frame_average_keeps_the_meaning_of_the_first_team_entry_of_the_shifted_law():
    """The swapped frame is asked about the swapped mark, whose first team is the other team, and its answer is moved back by the team table: the entry of the first
    team of the swapped frame comes back to the entry of the first team of the recorded frame, so lambda is averaged with lambda and mu with mu and the average is the
    answer of the recorded frame itself (a model that is consistent under the swap needs no change of the frame average for the shifted law)."""
    from c_ept.tests.ept_toys import encoder_batch
    model = FirstTeamStub()
    fa = heads.FrameAveraged(model)
    m = torch.tensor([0, 7, 24, 25, 49, 30, 50, 51])
    batch = encoder_batch(views=len(m), events=5, edges=4, seed=0)
    batch['side'] = torch.zeros(len(m))
    enc = fa.encode(batch)
    q, z = torch.zeros(len(m), dtype=torch.long), torch.zeros(len(m), dtype=torch.long)
    got = fa.m3(enc, q, m, z).kill_rate
    want = model.m3(None, q, m, z).kill_rate
    assert torch.equal(got, want)
    # control: the stub does have different entries, and the swapped frame alone is not the recorded one
    swapped = model.m3(None, q, data.swap({'M2Zone/masked/m': m})['M2Zone/masked/m'], z).kill_rate
    assert not torch.equal(swapped, want) and torch.equal(swapped.flip(-1)[m < 50], want[m < 50])


# =====================================================================================================================================
# the composer: the closed form and the draws (commit 4)
# =====================================================================================================================================
def note_expected_shifted(t0_s, T_s, n_bins):
    """E[V_H] of the toy of note 03 E under the shifted law, by hand. With the mark 0 the blue kills are 1 + N with the gold bin 6 fixed (600 whatever the count),
    the red kills are Poisson(ln 4/3) with the gold bin 4 (400) when there is a kill: E[g_red] = (1 - 3/4) 400 = 100; the dragon of the first team adds 0.1 x 200 = 20.
    So f0 = 600 - 100 + 20 = 520 (the plain law gave 300 - 100 + 20 = 220: the first team had no kill with probability 1/2), the tower conversion adds 0.8 c and the mark 1
    is the mirror image: E[V_H] = sum over the bins of 0.01 x seconds x 0.2 x (520 + 0.8 c)."""
    total = 0.0
    for q in range(n_bins):
        seconds = min(max(min(t0_s + 180.0, T_s - 30.0) - (t0_s + 10.0 * q), 0.0), 10.0)
        c = max(0.0, min(60.0, T_s - (t0_s + (q + 0.5) * 10.0 + 20.0)))
        total += 0.01 * seconds * 0.2 * (520.0 + 0.8 * c)
    return total


class TrueMarkToyHeads(ToyHeads):
    """The heads of note 03 section E with the real marks: the toy used the marks 0 and 1 for 'blue first' and 'red first', but the real mark 1 is a blue first kill too
    (mark = (d 5 + i) 5 + j), and the shifted law reads the first team from the mark. Here blue takes the first kill with the mark 0 (probability 0.6) and red with the
    mark 25 (probability 0.4); every other law of the toy is that of the note."""

    @staticmethod
    def first(m) -> np.ndarray:
        m = np.asarray(m, dtype=np.int64)
        return np.where(m < 50, m // 25, 0)

    def mark_prob(self, rows, q) -> np.ndarray:
        p = np.zeros((len(self._rows(rows)), C.N_MARKS))
        p[:, 0], p[:, 25] = 0.6, 0.4
        return p


def test_the_shifted_closed_form_of_the_note_03_toy_is_the_hand_value_with_a_first_team_that_always_scores():
    heads_ = TrueMarkToyHeads()
    plain = toy_composer(heads=heads_).expect(TOY_ROWS)
    assert plain[0, 0] == pytest.approx(TOY_EXPECTED_V60)                                # with the real marks the plain law gives the plain hand value of the note
    shifted = toy_composer(heads=TrueMarkToyHeads(), kill_law='shifted')
    got = shifted.expect(TOY_ROWS)
    want = np.array([[note_expected_shifted(t0, T, n) for n in (6, 12, 18)] for t0, T in TOY_ROW_SPECS])
    assert got == pytest.approx(want, rel=1e-12)
    assert shifted.expect([0])[0, 0] == pytest.approx(note_expected_shifted(1000.0, 5000.0, 6), rel=1e-12)   # 68.16 against the plain 32.16
    assert note_expected_shifted(1000.0, 5000.0, 6) == pytest.approx(68.16)
    assert (got > plain).all()                                                           # K_d >= 1 always: more gold for the first team


def oracle_expect_shifted(c, i):
    """``test_compose.oracle_expect`` with the kill table of the shifted law written out with ``math``: for the first team d of the mark m, P(K_d = k) = P(N = k - 1) =
    e^-r r^(k-1) / (k-1)! (k >= 1) and 0 for k = 0; the other team and an execution keep the plain law; the tail is one minus the sum."""
    h, rows = c.heads, c.rows
    t0, T = int(rows.t0_ms[i]), int(rows.T_ms[i])
    reps = np.asarray(c.gold_reps, dtype=np.float64)
    one = np.array([i])
    lam = h.rate(one)[0]
    zz, ss = np.repeat(np.arange(10), 4), np.tile(np.arange(4), 10)
    r10, r40 = np.full(10, i), np.full(40, i)
    total = np.zeros(3)
    for q in range(18):
        a = lam[q] * rows.exposure_ms[i, q] / 1000.0
        if a == 0.0:
            continue
        cexp = max(0.0, min(60.0, (T - (t0 + (q + 0.5) * 10_000 + 20_000)) / 1000.0))
        value = cell_values(c, i, q)
        pi = h.mark_prob(one, np.array([q]))[0]
        e_q = 0.0
        for m in range(52):
            first = m // 25 if m < 50 else -1
            rho = h.zone_prob(one, np.array([q]), np.array([m]))[0]
            size = h.size_prob(r10, np.full(10, q), np.full(10, m), np.arange(10))
            q40, m40 = np.full(40, q), np.full(40, m)
            mu = h.kill_rate(r40, q40, m40, zz, ss)
            G = np.stack([h.gold_prob(r40, q40, m40, zz, ss, np.full((40, 2), layer)) @ reps for layer in range(1, 7)], axis=-1)
            pk = np.zeros((40, 2, 6))
            for team in (0, 1):
                moved = 1 if team == first else 0
                for k in range(6):
                    j = k - moved
                    if j >= 0:
                        pk[:, team, k] = np.exp(-mu[:, team]) * mu[:, team] ** j / math.factorial(j)
            tail = 1.0 - pk.sum(axis=-1)
            gold_mean = (pk[..., 1:] * G[..., :5]).sum(axis=-1) + tail * G[..., 5]
            zeros = np.zeros((40, 4))
            obj = h.obj_rate(r40, q40, m40, zz, ss, zeros)
            m4 = features.m4_masked_inputs(h.row_view(r40), q40, m40, zz, zeros, np.zeros((40, 18)))
            conv = h.conv_rate(m4)
            g = gold_mean[:, 0] - gold_mean[:, 1] + obj @ value + (conv @ value) * cexp
            e_q += pi[m] * float((rho[zz] * size[zz, ss] * g).sum())
        for j, n_bins in enumerate((6, 12, 18)):
            if q < n_bins:
                total[j] += a * e_q
    return total


def test_the_shifted_closed_form_equals_the_brute_force_oracle_on_heads_that_differ_in_every_argument():
    c = rich_composer(kill_law='shifted')
    got = c.expect([0, 1, 2])
    want = np.array([oracle_expect_shifted(c, i) for i in range(3)])
    assert got == pytest.approx(want, rel=1e-10)
    assert (np.abs(got) > 1.0).all()
    plain = rich_composer().expect([0, 1, 2])
    assert not np.allclose(got, plain, rtol=1e-3)                                        # the law is not a label: the values move


def test_the_default_composer_equals_the_explicit_plain_composer_and_the_plain_oracle_and_the_old_table():
    a, b = rich_composer(), rich_composer(kill_law='poisson')
    assert a.kill_law == 'poisson' == b.kill_law
    assert np.array_equal(a.expect([0, 1, 2]), b.expect([0, 1, 2]))
    assert a.expect([0, 1, 2]) == pytest.approx(np.array([oracle_plain_expect(a, i) for i in range(3)]), rel=1e-10)
    da, db = rich_composer(S=200, seed=3).simulate_onsets(0), rich_composer(S=200, seed=3, kill_law='poisson').simulate_onsets(0)
    for name in ('sim', 'q', 'm', 'z', 's', 'k', 'gold_bin', 'gold', 'O', 'conv', 'value'):
        assert np.array_equal(getattr(da, name), getattr(db, name)), name
    assert np.array_equal(compose._poisson_head(np.array([[0.4, 3.0]]))[0], old_poisson_head(np.array([[0.4, 3.0]]))[0])
    assert np.array_equal(compose._poisson_head(np.array([[0.4, 3.0]]))[1], old_poisson_head(np.array([[0.4, 3.0]]))[1])


def test_a_composer_refuses_a_name_that_is_no_kill_law_and_the_keyword_is_keyword_only():
    with pytest.raises(ValueError, match='kill_law'):
        toy_composer(kill_law='negative_binomial')
    assert inspect.signature(compose.Composer).parameters['kill_law'].kind is inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(compose.Composer).parameters['kill_law'].default == 'poisson'


def test_the_shifted_draws_are_the_plain_draws_with_one_more_kill_for_the_first_team_and_the_same_random_stream():
    plain, shifted = rich_composer(S=400, seed=5), rich_composer(S=400, seed=5, kill_law='shifted')
    for i in range(3):
        a, b = plain.simulate_onsets(i), shifted.simulate_onsets(i)
        for name in ('sim', 'q', 'm', 'z', 's'):                                          # everything drawn before the kills
            assert np.array_equal(getattr(a, name), getattr(b, name)), (i, name)
        assert np.array_equal(b.k, a.k + killlaw.shift(killlaw.first_team(b.m), 'shifted')), i
        assert (b.k[np.arange(len(b.m)), np.maximum(killlaw.first_team(b.m), 0)][killlaw.first_team(b.m) >= 0] >= 1).all()
        assert np.array_equal(a.O, b.O) and np.array_equal(a.conv, b.conv)               # the rich heads do not read K: the stream after the kills is the same
        assert not np.array_equal(a.gold_bin, b.gold_bin)                                  # the layer of the gold law follows the kills


def test_the_shifted_kills_are_the_excess_poisson_for_the_first_team_and_poisson_for_the_other():
    c = rich_composer(S=3000, seed=21, kill_law='shifted')
    k_max = 0
    for i in range(3):
        d = c.simulate_onsets(i)
        mu = c.heads.kill_rate(np.full(len(d.q), i), d.q, d.m, d.z, d.s)
        shift = killlaw.shift(killlaw.first_team(d.m), 'shifted')
        assert (d.k[shift == 1] >= 1).all()
        for team in (BLUE, RED):
            excess = d.k[:, team] - shift[:, team]
            assert abs(z_test_sum(excess, mu[:, team])) < 4.5, (i, team)
            assert abs(z_test_dispersion(excess, mu[:, team])) < 4.5, (i, team)
        k_max = max(k_max, int(d.k.max()))
    assert k_max >= 6                                                                      # the layer cap is used


def test_the_shifted_simulation_mean_agrees_with_the_shifted_closed_form():
    """The criterion of D22 (``test_compose``, M0): the z of a mean needs onsets, so a cell with fewer than MIN_ONSETS expected onsets in the S simulations
    (S x E[N(H)], formula COMP-8) is held to four standard errors of the 180-s horizon of its row (the onsets of a shorter horizon are among those of H = 180 s, so that
    standard error bounds its true one), a cell with many expected onsets and sample sd 0 to 0.01 gold, and every other cell to z < 4.5 of its own sd. The 180-s column must
    itself have the onsets (it is the yardstick), so a seed that draws no onset fails here and cannot pass by a zero sd."""
    for composer in (toy_composer(S=20_000, seed=2, kill_law='shifted'), rich_composer(S=20_000, seed=4, kill_law='shifted')):
        idx = list(range(len(composer.rows)))
        r, e = composer.simulate(idx), composer.expect(idx)
        S = composer.S
        onsets = S * composer.expected_onsets(idx)                                         # (rows, 3): expected onsets in the S simulations (COMP-8)
        assert (onsets[:, -1] >= MIN_ONSETS).all() and (r.sd[:, -1] > 0).all(), (onsets, r.sd)
        few = onsets < MIN_ONSETS
        flat = (r.sd == 0.0) & ~few
        se = np.where(few, (r.sd / np.sqrt(S))[:, -1:], r.sd / np.sqrt(S))
        z = (r.mean - e) / np.where(flat, 1.0, se)
        assert np.abs(z[~flat]).max(initial=0.0) < 4.5, (z, onsets)
        assert (np.abs(r.mean - e)[flat] < 0.01).all(), (r.mean, e)
        assert (np.abs(e) > 0).any()


def oracle_plain_expect(c, i):
    from c_ept.tests.test_compose import oracle_expect
    return oracle_expect(c, i)


# =====================================================================================================================================
# the evaluation adapters: the open draws of a unit and the open B1 laws (commit 4)
# =====================================================================================================================================
def open_laws():
    size = np.array([.3, .4, .2, .1])
    kills = np.array([[.9, .6], [1.4, 1.1], [2.1, .7], [3.0, 2.5]])
    gold = np.full((4, 2, 6, 31), 1 / 31)
    return size, kills, gold


def test_the_open_draws_of_the_shifted_law_add_one_kill_to_the_first_team_and_keep_the_random_stream():
    from c_ept.evaluation.adapters import _open_draws
    seen = {}

    def objects_for(name):
        def objects(s, K):
            seen.setdefault(name, []).append((s.copy(), K.copy()))
            return np.full((len(s), 18), .05)
        return objects
    prices, reps = np.linspace(-3, 3, 18), np.arange(31.) * 100
    kw = dict(row=17, match=9, draws=600, seed=20261022, check=lambda: None, batch_size=None)
    plain = _open_draws(open_laws(), objects_for('plain'), prices, reps, **kw)
    for team in (0, 1):
        _open_draws(open_laws(), objects_for(f'shifted{team}'), prices, reps, d=team, kill_law='shifted', **kw)
        (s0, k0), (s1, k1) = seen['plain'][0], seen[f'shifted{team}'][0]
        assert np.array_equal(s0, s1)                                                      # the sizes are drawn before the kills: the same stream
        assert np.array_equal(k1[:, :2], k0[:, :2] + np.eye(2, dtype=np.int64)[team])      # the first team has one more kill
        assert (k1[:, team] >= 1).all() and (k0[:, team] == 0).any()
    _open_draws(open_laws(), objects_for('execution'), prices, reps, d=-1, kill_law='shifted', **kw)
    assert np.array_equal(seen['execution'][0][1], seen['plain'][0][1])                    # an execution start keeps the plain law
    default = _open_draws(open_laws(), objects_for('default'), prices, reps, d=0, **kw)
    assert all(np.array_equal(a, b) for a, b in zip(default, plain))                       # the law is the plain one unless asked: d alone changes nothing


def test_sample_open_value_with_certain_gold_has_the_hand_values_of_the_kill_law():
    """Size 0 only, no kill rate at all and the gold bin 3 (300) for every count: the plain law has no kill (f = 0: p+ = 0, p0 = 1); the shifted law gives the first team its one
    certain kill: blue first f = +300, red first f = -300, an execution start the plain f = 0."""
    size = np.array([1., 0., 0., 0.])
    gold = np.zeros((4, 2, 6, 31))
    gold[..., 3] = 1.0
    laws = (size, np.zeros((4, 2)), gold)
    objects = lambda s, K: np.zeros((len(s), 18))                                          # noqa: E731
    kw = dict(row=1, match=2, draws=50, seed=3, check=lambda: None, batch_size=None)
    reps, prices = np.arange(31.) * 100, np.zeros(18)
    assert sample_open_value(laws, objects, prices, reps, **kw)[:2] == (0.0, 1.0)
    assert sample_open_value(laws, objects, prices, reps, d=0, kill_law='shifted', **kw)[:2] == (1.0, 0.0)
    assert sample_open_value(laws, objects, prices, reps, d=1, kill_law='shifted', **kw)[:2] == (0.0, 0.0)
    assert sample_open_value(laws, objects, prices, reps, d=-1, kill_law='shifted', **kw)[:2] == (0.0, 1.0)


def test_the_shifted_open_draws_need_the_first_team_of_the_unit():
    from c_ept.evaluation.adapters import _open_draws
    objects = lambda s, K: np.zeros((len(s), 18))                                          # noqa: E731
    kw = dict(row=1, match=2, draws=10, seed=3, check=lambda: None, batch_size=None)
    for call in (sample_open_value, _open_draws):
        with pytest.raises(ValueError, match='first-kill team'):
            call(open_laws(), objects, np.ones(18), np.arange(31.), kill_law='shifted', **kw)
        with pytest.raises(ValueError, match='first-kill team'):
            call(open_laws(), objects, np.ones(18), np.arange(31.), d=2, kill_law='shifted', **kw)
        with pytest.raises(ValueError, match='kill_law'):
            call(open_laws(), objects, np.ones(18), np.arange(31.), d=0, kill_law='nb', **kw)


class StubOpenHead:
    """An open B1 head that records the codes of its calls and answers with a table that depends on them."""

    def __init__(self, out, slots=None):
        self.out, self.seen = out, []
        self.layout_ = None if slots is None else SimpleNamespace(slots=slots)

    def predict_blocks(self, numeric, *args, **codes):
        self.seen.append((args, {k: np.array(v) for k, v in codes.items()}))
        return self.out(len(numeric), *args, **codes)


def kill_table(n, s, d=None):
    """The answer of the stub kill head: 1 + s / 2 for both teams, plus 0.25 on the entry of the first team when the first team is given."""
    out = 1.0 + 0.5 * np.asarray(s, dtype=np.float64)[:, None] * np.ones((1, 2))
    if d is not None:
        d = np.asarray(d)
        out[np.flatnonzero(d >= 0), d[d >= 0]] += 0.25
    return out


def stub_b1(kill_slots):
    uniform = lambda n, *a, **k: np.full((n, 31), 1 / 31)                                  # noqa: E731
    kill = StubOpenHead(kill_table, kill_slots)
    heads_ = {'M3Size': StubOpenHead(lambda n, **k: np.full((n, 4), .25)), 'M3Kills': kill, 'M3Gold': StubOpenHead(uniform),
              'M3Obj': StubOpenHead(None), 'M4Conv': StubOpenHead(None)}
    return SimpleNamespace(open=heads_), kill


def test_the_b1_open_heads_give_the_first_team_to_a_kill_head_that_has_the_slot_and_only_then():
    from c_ept.evaluation.adapters import B1OpenHeads
    numeric = np.zeros((3, 5), dtype=np.float32)
    d = np.array([0, 1, -1])
    b1, kill = stub_b1(('feat', 's', 'd'))
    _, rates, _ = B1OpenHeads(b1).laws(numeric, check=lambda: None, d=d)
    assert all('d' in codes and np.array_equal(codes['d'], d) for _, codes in kill.seen) and len(kill.seen) == 4          # once for each size
    assert rates.shape == (3, 4, 2)
    assert np.allclose(rates[:, 0], [[1.25, 1.0], [1.0, 1.25], [1.0, 1.0]])                 # the stub marks the first-team entry
    with pytest.raises(ValueError, match='first-kill team'):
        B1OpenHeads(b1).laws(numeric, check=lambda: None)
    b1, kill = stub_b1(('feat', 's'))
    B1OpenHeads(b1).laws(numeric, check=lambda: None, d=d)
    assert all(set(codes) == {'s'} for _, codes in kill.seen)                               # a head without the slot is called as before
    b1, kill = stub_b1(None)
    B1OpenHeads(b1).laws(numeric, check=lambda: None)
    assert all(set(codes) == {'s'} for _, codes in kill.seen)


# =====================================================================================================================================
# B0 and B1: fit, score, predict and save with the kill law (commit 5)
# =====================================================================================================================================
SMALL = {'num_leaves': 7, 'min_data_in_leaf': 5, 'num_iterations': 12, 'min_data_per_group': 5, 'cat_smooth': 1.0, 'num_threads': 1}
HAND_Y = np.array([[2, 1], [0, 3], [1, 2], [3, 1]])
HAND_D = np.array([0, 1, 0, -1])
HAND_EXCESS = np.array([[1, 1], [0, 2], [0, 2], [3, 1]])                    # cell 0: 1 0 0 3 (mean 1.0), cell 1: 1 2 2 1 (mean 1.5)


@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')         # four rows with an all-constant design: the optimiser has nothing to move
def test_the_b0_kill_head_of_the_shifted_law_fits_the_excess_of_the_first_team_and_scores_the_shifted_mass():
    """Four rows with a constant design, so each cell is a Poisson regression with an intercept only and its maximum-likelihood rate is the mean of its target. The target of
    a cell is the count minus 1 on the rows where its team is the first team: the cell means are 1.0 and 1.5 (the plain law would give 1.5 and 1.75)."""
    X, e = np.zeros((4, 1)), np.ones(4)
    shifted = baselines_b0.M3Kills('open', kill_law='shifted').fit(X, HAND_Y, e, n_numeric=1, first_team=HAND_D)
    plain = baselines_b0.M3Kills('open').fit(X, HAND_Y, e, n_numeric=1)
    np.testing.assert_allclose(shifted.predict(X)[0], [1.0, 1.5], rtol=1e-6)
    np.testing.assert_allclose(plain.predict(X)[0], [1.5, 1.75], rtol=1e-6)
    assert shifted.kill_law == 'shifted' and plain.kill_law == 'poisson' and shifted.n_events_.tolist() == HAND_EXCESS.sum(axis=0).tolist()
    # the score of row 0 (blue first, K = (2, 1), rates (1.0, 1.5)): [1.0 - 1 log 1.0 + log 1!] + [1.5 - 1 log 1.5 + log 1!] = 1 + 1.5 - log 1.5
    rate = shifted.predict(X)
    got = shifted.nll(X, HAND_Y, e, first_team=HAND_D)
    assert got[0] == pytest.approx(1.0 + 1.5 - math.log(1.5), rel=1e-6)
    assert np.allclose(got, -killlaw.logpmf(HAND_Y, rate, HAND_D, 'shifted').sum(axis=1), rtol=1e-12)
    assert np.allclose(shifted.nll(X, HAND_Y, e, cells=True, first_team=HAND_D).sum(axis=1), got)
    with pytest.raises(ValueError, match='first'):
        baselines_b0.M3Kills('open', kill_law='shifted').fit(X, HAND_Y, e, n_numeric=1)                     # the shifted fit needs the first teams
    with pytest.raises(ValueError, match='first team with no kill'):
        baselines_b0.M3Kills('open', kill_law='shifted').fit(X, np.array([[0, 1]] * 4), e, n_numeric=1, first_team=np.zeros(4, dtype=int))
    with pytest.raises(ValueError, match='kill_law'):
        baselines_b0.M3Kills('open', kill_law='nb')


def test_the_b1_kill_head_of_the_shifted_law_fits_the_excess_of_the_first_team_and_scores_the_shifted_mass():
    """The same four rows for the boosted head: a few rows cannot split a leaf (min_data_in_leaf 200), so the booster is its starting rate, the mean of the target of its
    cell, and the label of the shifted law is the excess (the plain Poisson objective then fits lambda_d). The validation rows are shifted the same way."""
    X = np.zeros((4, 2), dtype=np.float32)
    e = np.ones(4)
    shifted = baselines_b1.M3Kills('open', kill_law='shifted').fit(X, HAND_Y, e, first_team=HAND_D, valid=(X, HAND_Y, e), valid_first_team=HAND_D)
    plain = baselines_b1.M3Kills('open').fit(X, HAND_Y, e)
    np.testing.assert_allclose(shifted.predict(X)[0], [1.0, 1.5], rtol=1e-6)
    np.testing.assert_allclose(plain.predict(X)[0], [1.5, 1.75], rtol=1e-6)
    assert shifted.kill_law == 'shifted' and shifted.n_events_.tolist() == HAND_EXCESS.sum(axis=0).tolist()
    got = shifted.nll(X, HAND_Y, e, first_team=HAND_D)
    assert got[0] == pytest.approx(1.0 + 1.5 - math.log(1.5), rel=1e-6)
    assert np.allclose(got, -killlaw.logpmf(HAND_Y, shifted.predict(X), HAND_D, 'shifted').sum(axis=1), rtol=1e-12)
    with pytest.raises(ValueError, match='first'):
        baselines_b1.M3Kills('open', kill_law='shifted').fit(X, HAND_Y, e)
    with pytest.raises(ValueError, match='first'):
        baselines_b1.M3Kills('open', kill_law='shifted').fit(X, HAND_Y, e, first_team=HAND_D, valid=(X, HAND_Y, e))       # a validation set without its first teams
    with pytest.raises(ValueError, match='kill_law'):
        baselines_b1.M3Kills('open', kill_law='nb')


@pytest.fixture(scope='module')
def law_heads(consistent_heads_world):
    """B0 and B1 fitted on the consistent world under both laws (B1 with small trees)."""
    w = consistent_heads_world
    dropped = features.dropped_columns(w.s2, w.fit_views)
    return SimpleNamespace(
        world=w, dropped=dropped,
        b0={law: baselines_b0.B0Heads.fit(w.s1, w.s2, w.labels, w.fit_means, kill_law=law) for law in killlaw.LAWS},
        b1={law: baselines_b1.B1Heads.fit(w.s1, w.s2, w.labels, dropped, params=SMALL, kill_law=law) for law in killlaw.LAWS})


def test_b0_heads_fitted_for_the_shifted_law_change_the_kill_heads_only(law_heads):
    plain, shifted = law_heads.b0['poisson'], law_heads.b0['shifted']
    assert plain.kill_law == 'poisson' and shifted.kill_law == 'shifted'
    for kind in ('masked', 'open'):
        for name, head in getattr(shifted, kind).items():
            arrays_a, arrays_b = head.state()[0], getattr(plain, kind)[name].state()[0]
            same = set(arrays_a) == set(arrays_b) and all(np.array_equal(arrays_a[k], arrays_b[k]) for k in arrays_a)
            assert same == (name != 'M3Kills'), (kind, name)            # every head but the kill head is the head of the plain fit, bit for bit
    for kind in ('masked', 'open'):
        assert getattr(shifted, kind)['M3Kills'].kill_law == 'shifted' and getattr(plain, kind)['M3Kills'].kill_law == 'poisson'
    assert [b.name for b in shifted.open['M3Kills'].blocks_] == ['s', 'd'] and [b.name for b in plain.open['M3Kills'].blocks_] == ['s']
    assert [b.name for b in shifted.masked['M3Kills'].blocks_] == ['q', 'm', 'z', 's']


def test_b0_scores_of_the_shifted_law_are_the_shifted_mass_of_the_rows_and_the_other_heads_score_as_before(law_heads):
    w = law_heads.world
    plain, shifted = law_heads.b0['poisson'], law_heads.b0['shifted']
    for kind in ('masked', 'open'):
        a, b = plain.nll(w.s1, w.s2, w.labels, 'DEV', kind), shifted.nll(w.s1, w.s2, w.labels, 'DEV', kind)
        assert set(a) == set(b)
        rows = baselines_b0.HeadRows(w.s1, w.s2, w.labels, w.fit_means, 'DEV', kill_law='shifted')
        d = rows.data('M3Kills', kind)
        want = -killlaw.logpmf(d.y, getattr(shifted, kind)['M3Kills'].predict(d.X), d.first_team, 'shifted').sum(axis=1)
        assert np.allclose(b['M3Kills'].nll, want, rtol=1e-12) and np.isfinite(b['M3Kills'].nll).all()
        for name in a:
            if name != 'M3Kills':
                assert np.array_equal(a[name].nll, b[name].nll), (kind, name)
    # the scale of the kill block is a scale like the others: positive and finite, by size
    from c_ept.ept import scales
    rows = baselines_b0.HeadRows(w.s1, w.s2, w.labels, w.fit_means, 'FIT', kill_law='shifted')
    scores = shifted.nll(w.s1, w.s2, w.labels, 'FIT', 'open')['M3Kills']
    sizes = scales._size_of_rows(w.s1, 'open', np.asarray(scores.row))
    value, n_rows, by_size = scales.head_scale('M3Kills', scores, sizes)
    assert value > 0 and np.isfinite(value) and n_rows == len(scores.nll) and rows.kill_law == 'shifted'


def test_b0_save_and_load_keep_the_kill_law_and_a_plain_save_is_the_save_of_every_earlier_run(law_heads, tmp_path):
    w = law_heads.world
    plain, shifted = law_heads.b0['poisson'], law_heads.b0['shifted']
    shifted.save(tmp_path / 'shifted')
    plain.save(tmp_path / 'plain')
    doc_s = json.loads((tmp_path / 'shifted' / baselines_b0.JSON_NAME).read_text(encoding='utf-8'))
    doc_p = json.loads((tmp_path / 'plain' / baselines_b0.JSON_NAME).read_text(encoding='utf-8'))
    assert doc_s['kill_law'] == 'shifted' and 'kill_law' not in doc_p and doc_s['schema'] == doc_p['schema'] == baselines_b0.SCHEMA
    assert 'kill_law' not in json.dumps(doc_p)                                            # nothing of the law in a plain save
    back = baselines_b0.B0Heads.load(tmp_path / 'shifted')
    assert back.kill_law == 'shifted' and back.open['M3Kills'].kill_law == 'shifted'
    a, b = shifted.nll(w.s1, w.s2, w.labels, 'DEV', 'open')['M3Kills'], back.nll(w.s1, w.s2, w.labels, 'DEV', 'open')['M3Kills']
    assert np.array_equal(a.nll, b.nll)
    assert baselines_b0.B0Heads.load(tmp_path / 'plain').kill_law == 'poisson'
    # a saved set of the shifted law is refused by a load that reads the layout of the plain law: the design differs
    doc_s_bad = dict(doc_s)
    doc_s_bad.pop('kill_law')
    (tmp_path / 'bad').mkdir()
    for name in (baselines_b0.NPZ_NAME,):
        (tmp_path / 'bad' / name).write_bytes((tmp_path / 'shifted' / name).read_bytes())
    (tmp_path / 'bad' / baselines_b0.JSON_NAME).write_text(json.dumps(doc_s_bad), encoding='utf-8')
    with pytest.raises(ValueError):
        baselines_b0.B0Heads.load(tmp_path / 'bad')


def test_the_composer_of_b0_heads_fitted_for_the_shifted_law_equals_the_brute_force_oracle(law_heads):
    w = law_heads.world
    rows = np.flatnonzero(features.split_rows(w.s1, 'DEV'))[:3]
    match = np.asarray(w.s1.array('frame_rows/match'))[rows].astype(np.int64)
    src, index = np.asarray(w.s2.array('views/src')), np.asarray(w.s2.array('views/src_index'))
    view_of = {int(i): v for v, (s, i) in enumerate(zip(src, index)) if s == 0}
    table = compose.RowTable(frame_row=rows, match=match, t0_ms=np.asarray(w.s1.array('frame_rows/t0'))[rows],
                             exposure_ms=np.asarray(w.s1.array('frame_rows/exposure_ms'))[rows],
                             T_ms=np.asarray(w.s1.array('matches/T_ms'))[match], views=[view_of[int(r)] for r in rows],
                             destroyed=np.zeros((len(rows), 2, 4), dtype=np.int64), roles=('DEV',))
    from c_ept.tests.test_compose import TOY_COMPOSE_PRICES, TOY_GOLD_REPS, hand_tier_model
    got = {}
    for law in killlaw.LAWS:
        c = compose.Composer(law_heads.b0[law].composer_heads('masked'), TOY_COMPOSE_PRICES, hand_tier_model(), TOY_GOLD_REPS, table, w.s2, S=50, seed=3, kill_law=law)
        got[law] = (c, c.expect(range(len(rows))))
    want = np.array([oracle_expect_shifted(got['shifted'][0], i) for i in range(len(rows))])
    assert got['shifted'][1] == pytest.approx(want, rel=1e-9) and np.isfinite(want).all()
    assert not np.allclose(got['shifted'][1], got['poisson'][1], rtol=1e-3)
    assert got['poisson'][1] == pytest.approx(np.array([oracle_plain_expect(got['poisson'][0], i) for i in range(len(rows))]), rel=1e-9)


def test_b1_heads_fitted_for_the_shifted_law_change_the_kill_heads_only_and_record_the_law(law_heads, tmp_path):
    w = law_heads.world
    plain, shifted = law_heads.b1['poisson'], law_heads.b1['shifted']
    assert plain.kill_law == 'poisson' and shifted.kill_law == 'shifted'
    for kind in ('masked', 'open'):
        assert getattr(shifted, kind)['M3Kills'].kill_law == 'shifted'
        assert getattr(shifted, kind)['M3Kills'].n_events_.sum() < getattr(plain, kind)['M3Kills'].n_events_.sum()       # the excess has fewer events than the counts
    assert shifted.open['M3Kills'].layout_.slots == ('feat', 's', 'd') and plain.open['M3Kills'].layout_.slots == ('feat', 's')
    assert shifted.masked['M3Kills'].layout_.slots == plain.masked['M3Kills'].layout_.slots
    # the events of a cell are the excess counts of the FIT rows (a plain loop from the text of the law)
    oracle = HeadsOracle(w)
    units_ = oracle.units('FIT')
    first = [oracle.mark[u] // 25 if oracle.mark[u] < 50 else -1 for u in units_]
    events = np.array([[oracle.K[u, t] - (1 if first[j] == t else 0) for t in (0, 1)] for j, u in enumerate(units_)]).sum(axis=0)
    assert shifted.open['M3Kills'].n_events_.tolist() == events.tolist()
    # a B1 head is the head of the plain fit unless it is the kill head
    for name, head in shifted.masked.items():
        if name != 'M3Kills':
            assert head.summary() == plain.masked[name].summary(), name
    # the scores are the shifted mass of the rows
    rows = baselines_b1.HeadRows(w.s1, w.s2, w.labels, law_heads.dropped, 'DEV', kill_law='shifted')
    for kind in ('masked', 'open'):
        d = rows.data('M3Kills', kind)
        got = shifted.nll(w.s1, w.s2, w.labels, 'DEV', kind)['M3Kills'].nll
        want = -killlaw.logpmf(d.y, getattr(shifted, kind)['M3Kills'].predict(d.X), d.first_team, 'shifted').sum(axis=1)
        assert np.allclose(got, want, rtol=1e-12) and np.isfinite(got).all()
    # save and load
    shifted.save(tmp_path / 'shifted')
    plain.save(tmp_path / 'plain')
    doc_s = json.loads((tmp_path / 'shifted' / baselines_b1.JSON_NAME).read_text(encoding='utf-8'))
    doc_p = json.loads((tmp_path / 'plain' / baselines_b1.JSON_NAME).read_text(encoding='utf-8'))
    assert doc_s['kill_law'] == 'shifted' and 'kill_law' not in doc_p and 'kill_law' not in json.dumps(doc_p)
    back = baselines_b1.B1Heads.load(tmp_path / 'shifted')
    assert back.kill_law == 'shifted' and back.open['M3Kills'].layout_.slots == ('feat', 's', 'd')
    d = rows.data('M3Kills', 'open')
    assert np.array_equal(back.open['M3Kills'].predict(d.X), shifted.open['M3Kills'].predict(d.X))
    assert baselines_b1.B1Heads.load(tmp_path / 'plain').kill_law == 'poisson'


def test_the_b1_chain_gives_the_rates_of_the_shifted_heads_and_the_composer_simulates_with_them(law_heads):
    w = law_heads.world
    shifted = law_heads.b1['shifted']
    rows = np.flatnonzero(features.split_rows(w.s1, 'DEV'))[:2]
    match = np.asarray(w.s1.array('frame_rows/match'))[rows].astype(np.int64)
    src, index = np.asarray(w.s2.array('views/src')), np.asarray(w.s2.array('views/src_index'))
    view_of = {int(i): v for v, (s, i) in enumerate(zip(src, index)) if s == 0}
    table = compose.RowTable(frame_row=rows, match=match, t0_ms=np.asarray(w.s1.array('frame_rows/t0'))[rows],
                             exposure_ms=np.asarray(w.s1.array('frame_rows/exposure_ms'))[rows],
                             T_ms=np.asarray(w.s1.array('matches/T_ms'))[match], views=[view_of[int(r)] for r in rows],
                             destroyed=np.zeros((len(rows), 2, 4), dtype=np.int64), roles=('DEV',))
    from c_ept.tests.test_compose import TOY_COMPOSE_PRICES, TOY_GOLD_REPS, hand_tier_model
    c = compose.Composer(shifted.composer_heads('masked'), TOY_COMPOSE_PRICES, hand_tier_model(), TOY_GOLD_REPS, table, w.s2, S=20, seed=3, kill_law='shifted')
    for i in range(len(rows)):
        d = c.simulate_onsets(i)
        if len(d.q):
            assert (d.k[np.arange(len(d.m))[killlaw.first_team(d.m) >= 0], killlaw.first_team(d.m)[killlaw.first_team(d.m) >= 0]] >= 1).all()


def dev_row_table(w, n):
    """The ``compose.RowTable`` of the first ``n`` DEV frame rows of the heads world (no tower events)."""
    rows = np.flatnonzero(features.split_rows(w.s1, 'DEV'))[:n]
    match = np.asarray(w.s1.array('frame_rows/match'))[rows].astype(np.int64)
    src, index = np.asarray(w.s2.array('views/src')), np.asarray(w.s2.array('views/src_index'))
    view_of = {int(i): v for v, (s, i) in enumerate(zip(src, index)) if s == 0}
    return compose.RowTable(frame_row=rows, match=match, t0_ms=np.asarray(w.s1.array('frame_rows/t0'))[rows],
                            exposure_ms=np.asarray(w.s1.array('frame_rows/exposure_ms'))[rows],
                            T_ms=np.asarray(w.s1.array('matches/T_ms'))[match], views=[view_of[int(r)] for r in rows],
                            destroyed=np.zeros((len(rows), 2, 4), dtype=np.int64), roles=('DEV',))


def test_the_chains_of_b0_and_b1_say_the_kill_law_of_their_kill_head_and_the_composer_refuses_heads_of_the_other_law(law_heads):
    """Review I1: a composer that reads the rates of shifted heads as plain (or the other way round) would drop or add the first team's certain kill without a word, so the
    chains declare the law of their M3Kills head and the composer compares it with its own before it binds the heads (the expensive step)."""
    from c_ept.tests.test_compose import TOY_COMPOSE_PRICES, TOY_GOLD_REPS, hand_tier_model
    w = law_heads.world
    table = dev_row_table(w, 2)
    for name, family in (('b0', law_heads.b0), ('b1', law_heads.b1)):
        for heads_law in killlaw.LAWS:
            chain = family[heads_law].composer_heads('masked')
            assert chain.kill_law == heads_law, (name, heads_law)
            for composer_law in killlaw.LAWS:
                args = (chain, TOY_COMPOSE_PRICES, hand_tier_model(), TOY_GOLD_REPS, table, w.s2)
                if composer_law == heads_law:
                    assert compose.Composer(*args, S=4, seed=1, kill_law=composer_law).kill_law == composer_law
                else:
                    with pytest.raises(ValueError, match='kill_law') as raised:
                        compose.Composer(*args, S=4, seed=1, kill_law=composer_law)
                    text = str(raised.value)
                    assert heads_law in text and composer_law in text and KEY_PREFIX not in text, (name, heads_law, composer_law)      # both laws named, no match key
        # the default composer (no keyword) is the plain composer: it takes plain heads and refuses shifted ones
        compose.Composer(family['poisson'].composer_heads('masked'), TOY_COMPOSE_PRICES, hand_tier_model(), TOY_GOLD_REPS, table, w.s2, S=4, seed=1)
        with pytest.raises(ValueError, match='kill_law'):
            compose.Composer(family['shifted'].composer_heads('masked'), TOY_COMPOSE_PRICES, hand_tier_model(), TOY_GOLD_REPS, table, w.s2, S=4, seed=1)


def test_the_composer_checks_the_law_of_heads_that_declare_one_before_it_binds_them_and_leaves_the_others_alone():
    """Heads that declare no law (the EPT compose heads, the toy heads) are not checked here: the training identity of the EPT carries their law. Heads that declare one are
    compared before ``bind`` (a refused composer has bound nothing)."""
    class Declaring(ToyHeads):
        def __init__(self, law):
            super().__init__()
            self.kill_law = law
    for heads_law, composer_law, ok in (('poisson', 'poisson', True), ('shifted', 'shifted', True), ('shifted', 'poisson', False), ('poisson', 'shifted', False)):
        heads_ = Declaring(heads_law)
        if ok:
            toy_composer(heads=heads_, kill_law=composer_law)
            assert heads_.n_bind == 1
        else:
            with pytest.raises(ValueError, match='kill_law'):
                toy_composer(heads=heads_, kill_law=composer_law)
            assert heads_.n_bind == 0
    for composer_law in killlaw.LAWS:                                                # no declaration: both laws are accepted, as before
        toy_composer(heads=ToyHeads(), kill_law=composer_law)


def test_the_default_fits_equal_the_explicit_plain_law_fits(law_heads, consistent_heads_world, tmp_path):
    """``kill_law='poisson'`` is the default: the default fit equals it bit for bit (saved files, scores)."""
    w = consistent_heads_world
    default0 = baselines_b0.B0Heads.fit(w.s1, w.s2, w.labels, w.fit_means)
    default0.save(tmp_path / 'b0')
    law_heads.b0['poisson'].save(tmp_path / 'b0law')
    for name in (baselines_b0.NPZ_NAME, baselines_b0.JSON_NAME):
        assert (tmp_path / 'b0' / name).read_bytes() == (tmp_path / 'b0law' / name).read_bytes()
    default1 = baselines_b1.B1Heads.fit(w.s1, w.s2, w.labels, law_heads.dropped, params=SMALL)
    a, b = default1.nll(w.s1, w.s2, w.labels, 'DEV', 'open'), law_heads.b1['poisson'].nll(w.s1, w.s2, w.labels, 'DEV', 'open')
    assert all(np.array_equal(a[k].nll, b[k].nll) for k in a)


class Overlay:
    """Small synthetic additions to a linked toy store (as in ``tests/test_ept_dev_adapters``)."""

    def __init__(self, base, arrays):
        self.base, self.arrays = base, arrays

    def array(self, key):
        if key == 'matches/key':
            raise AssertionError('no match keys needed')
        return self.arrays[key] if key in self.arrays else self.base.array(key)


@pytest.fixture(scope='module')
def inference_world(tmp_path_factory):
    """A DEV bundle on the consistent EPT toy world, a tiny EPT, FIT statistics and B1 heads of both laws fitted on the world."""
    from c_ept.tests.s3_toys import toy_tier_model
    from c_ept.valuation import flowprice
    world = build_ept_world(tmp_path_factory.mktemp('killlaw_inference'), n_events=7, n_edges=5, consistent_first_kill=True)
    n_matches = len(np.asarray(world.s1.array('matches/role')))
    s1 = Overlay(world.s1, {'matches/obj_off': np.zeros(n_matches + 1, dtype=np.int64),
                            **{f'objectives/{k}': np.empty(0, dtype=np.int64) for k in ('t', 'kind', 'sub', 'taker')}})
    n_units = len(s1.array('units/f'))
    labels = Overlay(world.labels, {'labels/f0_unit': np.resize(np.array([-1., 0., 1.]), n_units), 'labels/gold_reps': np.arange(31, dtype=float) * 100})
    stats = data.fit_stats(s1, world.s2)
    torch.manual_seed(17)
    model = heads.EPT(encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=.1, kappas=1, field_channels=2), toy_static())
    bundle = SimpleNamespace(s1=s1, s2=world.s2, labels=labels, role='DEV', prices={k: 100. for k in flowprice.GROUPS}, tier=toy_tier_model())
    dropped = features.dropped_columns(world.s2, np.flatnonzero(features.split_views(world.s1, world.s2, 'FIT')))
    b1 = {law: baselines_b1.B1Heads.fit(world.s1, world.s2, world.labels, dropped, params=SMALL, kill_law=law) for law in killlaw.LAWS}
    yield bundle, model, stats, b1
    del world
    gc.collect()


def test_run_inference_under_the_shifted_law_scores_and_draws_with_the_first_team_and_the_plain_law_adds_nothing(inference_world):
    from c_ept.evaluation.adapters import run_inference
    bundle, model, stats, b1 = inference_world
    common = dict(check=lambda: None, device='cpu', draws=6, batch_size=4)
    plain = run_inference(bundle, model, stats, b1['poisson'], **common)
    shifted = run_inference(bundle, model, stats, b1['shifted'], kill_law='shifted', **common)
    assert 'kill_law' not in plain['provenance'] and shifted['provenance']['kill_law'] == 'shifted'
    by_name = lambda out: {r['name']: r for r in out['nll']}                                  # noqa: E731
    a, b = by_name(plain), by_name(shifted)
    assert set(a) == set(b)
    for key in ('M3Kills/masked', 'M3Kills/open'):
        assert np.isfinite(b[key]['nll_ept']).all() and np.isfinite(b[key]['nll_b1']).all() and len(b[key]['row']) > 0
        assert not np.array_equal(a[key]['nll_ept'], b[key]['nll_ept']) and not np.array_equal(a[key]['nll_b1'], b[key]['nll_b1'])     # the law changes the score
    for key in a:                                                                              # every other block is scored exactly as before (the EPT is the same model)
        if key not in ('M3Kills/masked', 'M3Kills/open'):
            assert np.array_equal(a[key]['nll_ept'], b[key]['nll_ept']), key
    positive = [e for e in shifted['endpoints'] if e['name'] == 'M3/open/positive_value'][0]
    assert len(positive['row']) > 0 and np.isfinite(np.asarray(positive['p_ept'], dtype=float)).all()
    again = run_inference(bundle, model, stats, b1['poisson'], kill_law='poisson', **common)       # the keyword at its default is no change at all
    assert all(np.array_equal(x['nll_ept'], y['nll_ept']) and np.array_equal(x['nll_b1'], y['nll_b1']) for x, y in zip(plain['nll'], again['nll']))
    with pytest.raises(ValueError, match='kill_law'):
        run_inference(bundle, model, stats, b1['poisson'], kill_law='shifted', **common)


def test_run_inference_refuses_a_kill_law_that_is_not_the_one_of_the_b1_heads_or_not_a_law():
    from c_ept.evaluation.adapters import run_inference
    bundle = SimpleNamespace(role='DEV')
    b1 = SimpleNamespace(fit_roles=('FIT',), stop_roles=('STOP',))
    with pytest.raises(ValueError, match='kill_law'):
        run_inference(bundle, object(), object(), b1, check=lambda: None, device='cpu', kill_law='nb')
    with pytest.raises(ValueError, match='kill_law'):
        run_inference(bundle, object(), object(), b1, check=lambda: None, device='cpu', kill_law='shifted')             # B1 heads of the plain law
    b1.kill_law = 'shifted'
    with pytest.raises(ValueError, match='kill_law'):
        run_inference(bundle, object(), object(), b1, check=lambda: None, device='cpu')                                  # the plain law asked of shifted heads
    assert inspect.signature(run_inference).parameters['kill_law'].default == 'poisson'


def test_random_view_frames_keep_the_first_kill_team_with_its_kills(run_world):
    """S4b preparation M7 with the shifted law (decisions D5, D10 P1): in a view that is swapped, the first-kill team d (masked rows: the mark; open rows: the field d) and the kill columns
    change places together, so the shifted score of every row is the score of the same engagement seen from the other side. With every coin up, the batch scored by the one-frame wrapper
    is, bit for bit, the swapped batch scored as recorded (targets and scores); the recorded-frame score of the kill blocks is another number (control), and no score is infinite
    (a first-kill team that was swapped without its kills would score the impossible count 0 for it)."""
    world, stats = run_world
    rows = data.EngagementRows(data.ept_head_rows(world.s1, world.s2, world.labels, 'FIT', kill_law='shifted'))
    batch = next(iter(data.Batches(rows, 100, 0, stats=stats, swap_mode='random_view').epoch(0)))
    plain = {k: v for k, v in batch.items() if k != 'swap_view'}
    up = dict(plain, swap_view=torch.ones(len(plain['view']), dtype=torch.bool))
    torch.manual_seed(3)
    fa = heads.FrameAveraged(heads.EPT(encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.0, kappas=1, field_channels=2), toy_static()).eval(), average=False)
    whole = data.swap(plain)
    with torch.no_grad():
        framed = train.score_blocks(fa, fa.encode(up), up, kill_law='shifted')
        seen = train.score_blocks(fa, fa.encode(whole), whole, kill_law='shifted')
        recorded = train.score_blocks(fa, fa.encode(plain), plain, kill_law='shifted')
    assert set(framed) == set(seen) == set(recorded) and 'M3Kills/open' in framed and 'M3Kills/masked' in framed
    for name in framed:
        assert set(framed[name][1]) == set(seen[name][1]) and all(torch.equal(framed[name][1][f], seen[name][1][f]) for f in framed[name][1]) and torch.equal(framed[name][2], seen[name][2]), name
        assert torch.isfinite(framed[name][2]).all(), name
    d, framed_d = plain['M3Kills/open/d'], framed['M3Kills/open'][1]['d']
    assert torch.equal(framed_d, torch.where(d >= 0, 1 - d, d)) and not torch.equal(framed_d, d)           # the open first-kill team moved with the kills
    for name in ('M3Kills/open', 'M3Kills/masked'):
        assert not torch.equal(framed[name][2], recorded[name][2]), name


# =====================================================================================================================================
# run_inference(b0=) under the S4b condition: the shifted kill law with the binned K block (S4b preparation M11 review, m2)
# =====================================================================================================================================
@pytest.fixture(scope='module')
def s4b_condition(inference_world):
    """B1 and B0 fitted on the consistent world under the S4b condition (shifted law, binned K), plus the B0 of the plain law with the binned K block."""
    from c_ept.models import baselines_b0
    bundle, model, stats, b1 = inference_world
    fit = np.flatnonzero(features.split_views(bundle.s1, bundle.s2, 'FIT'))
    dropped = features.dropped_columns(bundle.s2, fit)
    means = features.b0_fit_means(bundle.s2, fit)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        b1s = baselines_b1.B1Heads.fit(bundle.s1, bundle.s2, bundle.labels, dropped, params=SMALL, kill_law='shifted', k_condition='binned')
        b0s = baselines_b0.B0Heads.fit(bundle.s1, bundle.s2, bundle.labels, means, kill_law='shifted', k_condition='binned')
        b0_raw = baselines_b0.B0Heads.fit(bundle.s1, bundle.s2, bundle.labels, means, kill_law='shifted')
    return SimpleNamespace(b1=b1s, b0=b0s, b0_raw=b0_raw)


def test_b0_through_the_evaluation_path_under_the_s4b_condition_scores_one_per_head_and_the_refusals_hold(inference_world, s4b_condition):
    """The condition of the S4b run: kill_law 'shifted', k_condition 'binned', gold_reps 'in_scope_fit'. The shifted score of the kill blocks (the first team's count is 1 + N_d) and the first-team pairing run
    for B0 as for B1; nll_b0 is the HR-5 score of ``B0Heads.nll`` of the same condition on the rows the model is scored on; B0 as the model is 1.0 on every head. A B0 of the other law or condition
    and one of other representatives are refused."""
    from c_ept.evaluation import inference, metrics
    from c_ept.evaluation.adapters import run_inference
    bundle, model, stats, b1 = inference_world
    s = s4b_condition
    kw = dict(kill_law='shifted', k_condition='binned', gold_reps='in_scope_fit', check=lambda: None, device='cpu', draws=6, batch_size=4)
    out = run_inference(bundle, model, stats, s.b1, b0=s.b0, **kw)
    assert out['provenance']['b0'] == {'kill_law': 'shifted', 'k_condition': 'binned', 'k_reps_sha256': s.b0.k_reps_sha256, **s.b0.saved_sha256()}
    by_name = {r['name']: r for r in out['nll']}
    for kind in ('masked', 'open'):
        for name, scores in s.b0.nll(bundle.s1, bundle.s2, bundle.labels, 'DEV', kind).items():
            record = by_name[f'{name}/{kind}']
            values, matches = scores.per_row()
            keep = np.isin(np.unique(scores.row), record['row'])
            assert len(record['row']) > 0 and np.isfinite(record['nll_b0']).all(), (kind, name)
            np.testing.assert_allclose(record['nll_b0'], values[keep], rtol=1e-12, atol=0, err_msg=f'{kind} {name}')
    plain = run_inference(bundle, model, stats, s.b1, **kw)
    assert all(np.array_equal(a['nll_b1'], b['nll_b1']) and np.array_equal(a['nll_ept'], b['nll_ept']) for a, b in zip(plain['nll'], out['nll']))
    ids = np.unique(np.concatenate([r['match'] for r in out['nll'] if len(r['match'])]))
    matrix = inference.resample_matrix(len(ids), B=40)
    one = metrics.relative_structural([dict(r, nll_ept=r['nll_b0']) for r in out['nll']], matrix, ids=ids)
    assert one['status'] == 'available' and one['metrics']['ept']['estimate'] == 1.0
    assert all(h['metrics']['ept']['estimate'] == 1.0 for h in one['secondary']['per_head'].values())
    # the refusals under this condition
    with pytest.raises(ValueError, match='B0.*kill_law'):
        run_inference(bundle, model, stats, s.b1, b0=_with(s.b0, kill_law='poisson'), **kw)
    with pytest.raises(ValueError, match='B0.*k_condition'):
        run_inference(bundle, model, stats, s.b1, b0=s.b0_raw, **kw)
    calls = []
    with pytest.raises(ValueError, match='B0.*other representatives'):
        run_inference(bundle, model, stats, s.b1, b0=_with(s.b0, k_reps_sha256='0' * 64), **{**kw, 'check': lambda: calls.append(1)})
    assert len(calls) == 1
    with pytest.raises(ValueError, match='B0.*no k_reps_sha256'):
        run_inference(bundle, model, stats, s.b1, b0=_with(s.b0, k_reps_sha256=None), **kw)


def _with(b0, **changes):
    import copy
    other = copy.copy(b0)
    for k, v in changes.items():
        setattr(other, k, v)
    return other
