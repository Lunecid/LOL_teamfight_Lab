"""Tests of the heads of the EPT (stage S4a Task 3): ``ept.heads``.

No run folder, raw file or catalog is read, and nothing is put on a GPU. Most tests run a tiny model (``d`` = 16, two layers, two heads) on
``ept_toys.encoder_batch`` (random view tensors, no store); one test reads the tensors of the 7 / 5 toy world through ``data.ViewTensors``.

    laws and rates       every categorical law adds up to 1 and has the floor of ruling R13 (also when it is saturated, and on hand cases);
                         the rates are positive, scale with the exposure and are 0 without it, with no gradient from such a cell
    inputs               NaN landmark seconds reach the heads as a zero and a mask and give finite outputs and gradients; the conditions
                         are checked and the outputs do not depend on their memory layout; every condition reaches the heads that the plan
                         gives it to and the heads take no more than the plan gives; ``pos`` picks the views of the rows
    forms                the equations (EH-1) .. (EH-8) against a plain computation from the weights; the state effect on a rate depends on
                         the bin; the gold law of a team reads that team's kill layer only; the mark tables name the killer and the victim;
                         marks follow the players when a team is renumbered; the open reading; every parameter gets a gradient; sizes S and M
    frame average        exact invariance under the team swap (bit for bit, several batch sizes, both frames, mixed sides, ``pos`` with repeated
                         views, the open reading of every head), the log mean evaluated in the recorded frame, the averaging space and the
                         table of every output axis on a stub whose answers are known, the conditions that the swapped call gets and their
                         check before the swap, the conversion window (one value for each row, linear through the average), two separate
                         model calls of one shape, the convexity claim of the average
    determinism          forward and backward bit-identical for one seed and thread count, also in a second process
"""
import gc
import hashlib
import inspect
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from c_ept import config as C
from c_ept import head_rows, symmetry, terrain, tokens, units
from c_ept.ept import data, encoder, heads
from c_ept.tests.ept_toys import build_ept_world, encoder_batch, toy_static

TINY = dict(d=16, layers=2, heads=2, dropout=0.0)
N_PLAYERS = tokens.N_PLAYERS
MIX = 1e-6                                           # the uniform share of ruling R13
LAWS = (('mark', 52), ('zone', 10), ('size', 4), ('gold', 31))


def tiny_model(seed: int = 0, **changes) -> heads.EPT:
    """A tiny model in evaluation mode, built after ``torch.manual_seed(seed)`` (the same weights for the same seed)."""
    torch.manual_seed(seed)
    return heads.EPT(encoder.EncoderConfig(**{**TINY, **changes}), toy_static()).eval()


def conditions(n: int, seed: int = 0, dtype=torch.float32) -> dict:
    """Made-up conditions of ``n`` rows, one row for each view: the bin q, mark m, zone z, size s, the kills K (kills of both teams, kill gold
    in thousands), the objectives O, an exposure of 18 bins (seconds) and the length of a conversion window. Row 0 has an execution mark
    (50), row 1 the other one (51), row 2 a first kill without a position (z = -1)."""
    g = torch.Generator().manual_seed(seed)
    c = {'q': torch.randint(0, C.N_BINS, (n,), generator=g), 'm': torch.randint(0, C.N_MARKS, (n,), generator=g),
         'z': torch.randint(-1, 10, (n,), generator=g), 's': torch.randint(0, 4, (n,), generator=g),
         'K': (torch.rand(n, 4, generator=g) * 5).to(dtype), 'O': torch.randint(0, 3, (n, 18), generator=g).to(dtype),
         'exposure': (1.0 + torch.rand(n, 18, generator=g) * 9).to(dtype), 'window': (1.0 + torch.rand(n, generator=g) * 59).to(dtype)}
    c['m'][:2] = torch.tensor([50, 51])[:n]
    if n > 2:
        c['z'][2] = -1
    return c


def pooled(h: torch.Tensor, events: int = 3) -> encoder.Encoded:
    """An ``Encoded`` that only has the pooled states ``h`` (n, d): the player tokens and the others are zero. The heads of the rate read ``h`` only."""
    n, d = h.shape
    n_tok = N_PLAYERS + tokens.N_LANDMARKS + events + 1
    return encoder.Encoded(torch.zeros(n, n_tok, d, dtype=h.dtype), h, torch.ones(n, n_tok, dtype=torch.bool))


def fabricated(n: int, d: int = 16, seed: int = 0, dtype=torch.float32, events: int = 3) -> encoder.Encoded:
    """An ``Encoded`` with random states for every token (n views): the heads read the players 0-9 and the pooled state only."""
    g = torch.Generator().manual_seed(seed)
    n_tok = N_PLAYERS + tokens.N_LANDMARKS + events + 1
    h_tok = torch.randn(n, n_tok, d, generator=g, dtype=torch.float64).to(dtype)
    return encoder.Encoded(h_tok, h_tok[:, -1].clone(), torch.ones(n, n_tok, dtype=torch.bool))


@pytest.fixture(scope='module')
def tiny():
    """A tiny model, five random views and their encodings (no gradient)."""
    model = tiny_model()
    batch = encoder_batch(views=5, events=9, edges=7, seed=1)
    with torch.no_grad():
        enc = model.encode(batch)
    return model, batch, enc


# =====================================================================================================================================
# laws
# =====================================================================================================================================
def test_mark_and_zone_laws_sum_to_one(tiny):
    """Law A2 (conservation): the mark law of every bin, the zone law of every bin and mark, the size law and the gold law of every size, team
    and kill layer are probability laws over their classes; every probability has the floor of ruling R13, and nothing is NaN or infinite."""
    model, batch, enc = tiny
    c = conditions(5)
    m2 = model.heads.m2(enc, c['exposure'])
    size_logp, _, gold_logp = model.heads.m3(enc, c['q'], c['m'], c['z'])
    laws = {'mark': (m2.mark_logp, (5, 18, 52)), 'zone': (m2.zone_logp, (5, 18, 52, 10)), 'size': (size_logp, (5, 4)),
            'gold': (gold_logp, (5, 4, 2, 6, 31))}
    for name, (logp, shape) in laws.items():
        p, classes = logp.exp(), shape[-1]
        assert logp.shape == shape and torch.isfinite(logp).all(), name
        assert torch.allclose(p.sum(-1), torch.ones(shape[:-1]), atol=1e-5, rtol=0), name
        assert p.min() >= MIX / classes * (1 - 1e-3), name
        assert p.max() < 0.99, name                                          # the laws of a new model are far from certain: the test can fail
    assert [classes for _, classes in LAWS] == [shape[-1] for _, shape in laws.values()]


@pytest.mark.parametrize('law, part, classes',
                         [('mark', 'mark_net.out', 52), ('zone', 'zone_net.out', 10), ('size', 'size_out', 4), ('gold', 'gold_out', 31)])
def test_a_saturated_law_still_has_the_r13_floor(law, part, classes):
    """Ruling R13: p = (1 - 1e-6) p_model + 1e-6 / C over the full class set. A law whose output layer is scaled up so far that the softmax is
    one-hot has the floor 1e-6 / C for every class but one (not 0, not NaN), and still adds up to 1."""
    model, enc, c = tiny_model(2), fabricated(4), conditions(4, seed=3)
    module = model.heads
    for name in part.split('.'):
        module = getattr(module, name)
    with torch.no_grad():
        module.weight.mul_(1e5)
    if law in ('mark', 'zone'):
        logp = getattr(model.heads.m2(enc, c['exposure']), law + '_logp')
    else:
        logp = getattr(model.heads.m3(enc, c['q'], c['m'], c['z']), law + '_logp')
    p = logp.exp()
    assert torch.isfinite(logp).all() and torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-5, rtol=0)
    assert abs(p.min().item() * classes / MIX - 1.0) < 1e-3                          # the floor is reached: the law is one-hot but for it
    assert p.max() > 0.999


def test_the_r13_mixture_is_the_formula():
    """(EH-4) on hand cases in float64: uniform logits give the uniform law, a spread of logits gives (1 - 1e-6) softmax + 1e-6 / C, and a logit
    that is 100 above the others leaves the others at 1e-6 / C plus the softmax share (about 3.7e-44, below float64 resolution of the sum)."""
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0], [100.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    want = (1 - MIX) * torch.softmax(logits, -1) + MIX / 4
    got = heads.mixed_log_softmax(logits).exp()
    assert torch.allclose(got, want, rtol=1e-12, atol=0)
    assert torch.allclose(got[0], torch.full((4,), 0.25, dtype=torch.float64), rtol=1e-12, atol=0)
    assert torch.allclose(got[2, 1:], torch.full((3,), MIX / 4, dtype=torch.float64), rtol=1e-9, atol=0)
    along = heads.mixed_log_softmax(logits.T, dim=0)                                  # another axis: the same numbers
    assert torch.equal(along, heads.mixed_log_softmax(logits).T)
    huge = heads.mixed_log_softmax(torch.tensor([[1e4, 0.0, -1e4]]))                  # float32: finite where the plain log-softmax is not
    assert torch.isfinite(huge).all() and abs(huge[0, 1].exp().item() * 3 / MIX - 1.0) < 1e-3


# =====================================================================================================================================
# rates and exposure
# =====================================================================================================================================
def test_rates_are_positive_and_scale_with_exposure(tiny):
    """Plan Q7 (the Poisson heads include the exposure) and review focus 4: the rate of a bin is the expected count lambda x exposure, so it is
    positive where there is exposure, twice as large for twice the exposure, and exactly 0 where there is none (the other cells unchanged).
    The conversions of M4 have one exposure for the row. The expected counts of the kills and the objectives (exposure 1) are positive."""
    model, batch, enc = tiny
    H, c = model.heads, conditions(5)
    one = torch.ones(5, 18)
    base = H.m2(enc, one).rate
    assert base.shape == (5, 18) and (base > 0).all() and torch.isfinite(base).all()
    assert torch.allclose(H.m2(enc, 2 * one).rate, 2 * base, rtol=1e-6, atol=0)
    assert torch.allclose(H.m2(enc, 7.5 * one).rate / 7.5, base, rtol=1e-6, atol=0)           # the rate per second does not depend on it
    some = one.clone()
    some[:, [3, 5]], some[2, 17] = 0.0, 0.0
    cut = H.m2(enc, some).rate
    keep = some > 0
    assert (cut[~keep] == 0).all() and torch.equal(cut[keep], base[keep])
    # M4: one exposure for the row (a window in seconds)
    args = (enc, c['q'], c['m'], c['z'], c['K'], c['O'])
    row = torch.ones(5)
    conv = H.m4(*args, row)
    assert conv.shape == (5, 18) and (conv > 0).all()
    assert torch.allclose(H.m4(*args, 2 * row), 2 * conv, rtol=1e-6, atol=0)
    empty = row.clone()
    empty[1] = 0.0
    assert (H.m4(*args, empty)[1] == 0).all() and torch.equal(H.m4(*args, empty)[[0, 2, 3, 4]], conv[[0, 2, 3, 4]])
    assert torch.equal(H.m4(*args, 7.5), H.m4(*args, torch.full((5,), 7.5)))                            # a number is the same window for every row
    # M3: expected counts, exposure 1
    _, kill_rate, _ = H.m3(enc, c['q'], c['m'], c['z'])
    assert kill_rate.shape == (5, 4, 2) and (kill_rate > 0).all()
    obj = H.m3_obj(enc, c['q'], c['m'], c['z'], c['s'], c['K'])
    assert obj.shape == (5, 18) and (obj > 0).all()


def test_zero_exposure_cells_have_no_gradient_in_the_heads():
    """Review focus 4, the head half: a cell without exposure has the output 0 and no gradient reaches the pooled state or any parameter from it,
    while a cell with exposure does get one (so the check can fail)."""
    model, c = tiny_model(4), conditions(3)
    enc = fabricated(3)
    h = enc.h_pool.clone().requires_grad_(True)
    live = encoder.Encoded(enc.h_tok, h, enc.mask)
    exposure = torch.ones(3, 18)
    exposure[1, 7] = 0.0
    rate = model.heads.m2(live, exposure).rate
    dead = torch.autograd.grad(rate[1, 7], h, retain_graph=True)[0]
    alive = torch.autograd.grad(rate[1, 8], h, retain_graph=True)[0]
    assert (dead == 0).all() and alive.abs().sum() > 0
    rate[1, 7].backward(retain_graph=True)
    assert all(p.grad is None or (p.grad == 0).all() for p in model.heads.parameters())
    window = torch.tensor([30.0, 0.0, 30.0])
    conv = model.heads.m4(live, c['q'], c['m'], c['z'], c['K'], c['O'], window)
    assert (torch.autograd.grad(conv[1].sum(), h, retain_graph=True)[0][1] == 0).all()
    assert torch.autograd.grad(conv[0].sum(), h)[0][0].abs().sum() > 0


# =====================================================================================================================================
# missing values
# =====================================================================================================================================
@pytest.fixture(scope='module')
def small_world(tmp_path_factory):
    """The EPT toy world with 7 event tokens and 5 edges per view (the real store has 128 and 128)."""
    folder = tmp_path_factory.mktemp('heads_world')
    world = build_ept_world(folder, n_events=7, n_edges=5)
    yield world
    del world
    gc.collect()                                   # Windows does not delete a folder with open memory maps
    shutil.rmtree(folder, ignore_errors=True)


class NanLandmarkSeconds:
    """An S2 store whose landmark rows that may be NaN (``tokens.NAN_ALLOWED``) have NaN seconds-left in every view: all the rest is the store."""

    def __init__(self, s2):
        self._s2, self.manifest = s2, s2.manifest
        self.rows = sorted({row for row, col in tokens.NAN_ALLOWED['landmarks']['cells'] if col == 1})

    def array(self, name):
        a = np.array(self._s2.array(name))
        if name == 'x/landmarks':
            a[:, self.rows, 1] = np.nan
        return a


def test_all_nan_landmark_seconds_give_finite_outputs(small_world):
    """Review focus 3: the seconds-left of every landmark that may have none are NaN in every view of the store. ``ViewTensors`` turns them into
    a zero and a mask, so every output of every head, in both readings, and every gradient stays finite."""
    stats = data.fit_stats(small_world.s1, small_world.s2)
    s2 = NanLandmarkSeconds(small_world.s2)
    views = np.arange(len(small_world.s2.array('views/src')))
    assert np.isnan(s2.array('x/landmarks')[:, s2.rows, 1]).all() and len(s2.rows) == 4
    batch = data.ViewTensors(s2, views, stats)
    assert (batch['landmarks_nan'][:, s2.rows] == 1).all() and (batch['landmarks'][:, s2.rows, 1] == 0).all()
    assert (batch['landmarks_nan'].sum(1) == 4).all()                                                  # only those four rows are masked
    n, model = len(views), tiny_model(5)
    model.train()
    enc, c = model.encode(batch), conditions(len(views), seed=6)
    outs = [*model.heads.m2(enc, c['exposure']), *model.heads.m3(enc, c['q'], c['m'], c['z']), model.heads.m3_obj(enc, c['q'], c['m'], c['z'], c['s'], c['K']),
            model.heads.m4(enc, c['q'], c['m'], c['z'], c['K'], c['O'], c['window']), *model.heads.m3(enc, None, None, None),
            model.heads.m3_obj(enc, None, None, None, c['s'], c['K']), model.heads.m4(enc, None, None, None, c['K'], c['O'], c['window'])]
    assert len(outs) == 13 and all(torch.isfinite(t).all() for t in outs) and enc.h_pool.shape == (n, 16)       # 3 + 3 + 1 + 1 masked, 3 + 1 + 1 open
    sum(t.sum() for t in outs).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # the test can fail: the zero and the mask are what keeps the network finite, not luck. A tensor that really holds the NaN (as the store does)
    # makes the outputs NaN; and a batch in which every one of the 34 landmarks has no seconds (all masked, all zero) is finite as well
    raw = {**batch, 'landmarks': batch['landmarks'].clone()}
    raw['landmarks'][:, s2.rows, 1] = float('nan')
    assert not torch.isfinite(model.eval().encode(raw).h_pool).all()
    every = encoder_batch(views=3, events=6, edges=4, seed=4)
    every['landmarks'][:, :, 1], every['landmarks_nan'][:] = 0.0, 1.0
    enc = model.encode(every)
    assert torch.isfinite(enc.h_pool).all() and all(torch.isfinite(t).all() for t in model.heads.m2(enc, torch.ones(3, 18)))


# =====================================================================================================================================
# the state effect depends on the bin
# =====================================================================================================================================
def state_effect(H, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """Delta_q = log lambda_q(h2) - log lambda_q(h1) for the 18 bins: what moving the pooled state from h1 to h2 does to the log rate of a bin
    (the other tokens are zero; the head of the rate reads the pooled state and the bin only)."""
    one = torch.ones(1, 18, dtype=h1.dtype)
    low, high = (H.m2(pooled(h[None]), one).rate.log()[0] for h in (h1, h2))
    return high - low


def test_rate_depends_on_lead_bin_interaction():
    """B2-a: the effect of the state on the rate depends on the bin (the incidence slopes were 1.16, 0.88 and 0.95 in the three windows of the
    first 180 s). log lambda_q = MLP([h, Gamma_q]) lets the same move of h change the log rate by a different amount in each bin; the additive form
    h . beta + gamma_q, whose effect is the same number in every bin, cannot. The measure is the spread of Delta_q over the bins, on
    parameters of a trained size (the test is about the form, not about the starting values)."""
    H = tiny_model(1).double().heads
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        H.bin_emb.weight.copy_(torch.randn(H.bin_emb.weight.shape, generator=g, dtype=torch.float64))
        H.rate_net.out.weight.copy_(torch.randn(H.rate_net.out.weight.shape, generator=g, dtype=torch.float64))
    spreads, beta, gamma = [], torch.randn(16, generator=g, dtype=torch.float64), torch.randn(18, generator=g, dtype=torch.float64)
    additive = lambda h: (h @ beta)[None] + gamma                                                       # noqa: E731  (18,)
    for _ in range(5):
        h1, h2 = (torch.randn(16, generator=g, dtype=torch.float64) for _ in range(2))
        delta = state_effect(H, h1, h2)
        assert delta.shape == (18,)
        spreads.append((delta.max() - delta.min()).item())
        control = additive(h2) - additive(h1)                                                          # the same measure on the additive form
        assert (control.max() - control.min()) < 1e-12
    assert min(spreads) > 0.1, spreads


# =====================================================================================================================================
# the frame average
# =====================================================================================================================================
MARK, ZONE, TEAM, CELL = (torch.as_tensor(np.asarray(p, dtype=np.int64)) for p in (data.MARK_PERM, data.ZONE_PERM, data.TEAM_PERM, data.CELL_PERM))


def logmean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The log of the mean of the two probabilities exp(a) and exp(b)."""
    return torch.logaddexp(a, b) - math.log(2.0)


def swapped_conditions(c: dict) -> dict:
    """The conditions of the swapped frame, from the references that ``data`` itself is built on: ``symmetry`` for the marks and O, the zone
    partners of ``terrain.ZONE_SWAP`` (-1 stays) and the exchange of the two teams in K = (kills blue, kills red, kill gold blue, red)."""
    z = c['z'].numpy()
    return {**c, 'm': torch.as_tensor(symmetry.swap_mark(c['m'].numpy())),
            'z': torch.as_tensor(np.where(z >= 0, terrain.ZONE_SWAP[np.maximum(z, 0)].astype(np.int64), z)),
            'K': c['K'][:, [1, 0, 3, 2]], 'O': torch.as_tensor(symmetry.swap_O(c['O'].numpy()))}


def check_swap_invariance(fa, batch: dict, c: dict, pos=None):
    """``FA(swap b)`` is ``FA(b)`` with its outputs moved by the tables of ``data``, bit for bit (``torch.equal``, no tolerance): all four methods, the
    masked and the open reading of M3, of the objectives and of M4. The rows are the views of ``batch`` in order or, with ``pos``, the views it
    lists (some twice); ``c`` holds the conditions of the rows. The swapped call gets the conditions of the swapped frame from the numpy references
    of ``symmetry`` and ``terrain``. Returns the encodings of the batch and of its swap."""
    swapped = data.swap(batch)
    s = swapped_conditions(c)
    assert not torch.equal(s['m'], c['m']) and not torch.equal(s['K'], c['K']) and not torch.equal(s['O'], c['O'])
    with torch.no_grad():
        enc, enc_s = fa.encode(batch), fa.encode(swapped)
        # M2: rate and the three laws; the axes that the swap changes are the marks and the zones
        a, b = fa.m2(enc, c['exposure'], pos), fa.m2(enc_s, c['exposure'], pos)
        assert torch.equal(a.rate, b.rate)
        assert torch.equal(a.mark_logp, b.mark_logp[..., MARK])
        assert torch.equal(a.zone_logp, b.zone_logp[:, :, MARK][:, :, :, ZONE])
        # M3: the size law has no axis to change, the two teams of the kills and of the gold law change places
        a, b = fa.m3(enc, c['q'], c['m'], c['z'], pos), fa.m3(enc_s, c['q'], s['m'], s['z'], pos)
        assert torch.equal(a.size_logp, b.size_logp)
        assert torch.equal(a.kill_rate, b.kill_rate[..., TEAM])
        assert torch.equal(a.gold_logp, b.gold_logp[:, :, TEAM])
        a, b = fa.m3(enc, None, None, None, pos), fa.m3(enc_s, None, None, None, pos)                       # the open reading (no q, m, z)
        assert torch.equal(a.size_logp, b.size_logp) and torch.equal(a.kill_rate, b.kill_rate[..., TEAM]) and torch.equal(a.gold_logp, b.gold_logp[:, :, TEAM])
        # the objectives and the conversions, in both readings: the 18 cells change places pairwise (blue and red taker of every kind)
        for q, m, z, m_swapped, z_swapped in ((c['q'], c['m'], c['z'], s['m'], s['z']), (None, None, None, None, None)):
            a = fa.m3_obj(enc, q, m, z, c['s'], c['K'], pos)
            b = fa.m3_obj(enc_s, q, m_swapped, z_swapped, c['s'], s['K'], pos)
            assert torch.equal(a, b[:, CELL])
            a = fa.m4(enc, q, m, z, c['K'], c['O'], c['window'], pos)
            b = fa.m4(enc_s, q, m_swapped, z_swapped, s['K'], s['O'], c['window'], pos)
            assert torch.equal(a, b[:, CELL])
    return enc, enc_s


def test_frame_average_is_exactly_swap_invariant():
    """Law A1 and review focus 1: the encoder and the heads are not swap invariant themselves (the side marker and the team blocks are inputs), but
    the frame average is, exactly. ``FA(swap b)`` is ``FA(b)`` with its outputs moved by the tables of ``data``: asserted with ``torch.equal``, that is
    bit for bit, no tolerance. The two terms of the average are the same two model calls in either order (the swap is its own inverse, bit for bit, on
    these batches), the tables are index permutations and the sum of two floats does not depend on their order; the model is in evaluation mode and
    deterministic. Cases: batch sizes 5-8 (a vectorised kernel rounds a value by its place in the buffer, which depends on the size); every view
    recorded in the recorded frame, every view in the swapped frame, and views of both kinds in one batch (as drawn and alternating); rows that
    share a view (``pos`` lists every view twice, in two orders); and the open reading of M3, of the objectives and of M4."""
    model = tiny_model(7)
    fa = heads.FrameAveraged(model)
    for views in (5, 6, 7, 8):
        for sides in ('drawn', 'recorded', 'swapped', 'alternating'):
            batch = encoder_batch(views=views, events=9, edges=7, seed=2 + views)
            if sides != 'drawn':
                batch['side'] = {'recorded': torch.zeros(views), 'swapped': torch.ones(views), 'alternating': (torch.arange(views) % 2).float()}[sides]
            swapped = data.swap(batch)
            assert all(torch.equal(data.swap(swapped)[k], batch[k]) for k in batch) and not torch.equal(swapped['side'], batch['side'])
            check_swap_invariance(fa, batch, conditions(views, seed=8 + views))
        mixed = {**batch, 'side': (torch.arange(views) % 2).float()}
        assert mixed['side'].min() == 0 and mixed['side'].max() == 1
        g = torch.Generator().manual_seed(views)
        pos = torch.cat([torch.randperm(views, generator=g), torch.randperm(views, generator=g)])            # every view twice
        assert len(pos) == 2 * views and len(set(pos.tolist())) == views
        check_swap_invariance(fa, mixed, conditions(2 * views, seed=18 + views), pos)
    # the test can fail: one frame alone is not invariant, and the average is not that frame
    batch, c = encoder_batch(views=8, events=9, edges=7, seed=10), conditions(8, seed=16)
    enc, _ = check_swap_invariance(fa, batch, c)
    with torch.no_grad():
        once, other = model.heads.m2(model.encode(batch), c['exposure']), model.heads.m2(model.encode(data.swap(batch)), c['exposure'])
        assert (once.mark_logp - other.mark_logp[..., MARK]).abs().max() > 1e-4 and (once.rate - other.rate).abs().max() > 1e-4
        assert (fa.m2(enc, c['exposure']).mark_logp - once.mark_logp).abs().max() > 1e-5
        k_once = model.heads.m3(model.encode(batch), c['q'], c['m'], c['z']).kill_rate
        assert (k_once - fa.m3(enc, c['q'], c['m'], c['z']).kill_rate).abs().max() > 1e-6


@pytest.mark.parametrize('rows', [3, 5, 6, 7, 9])
def test_the_log_mean_is_the_same_whichever_frame_a_row_came_in(rows):
    """(EH-9), the evaluation in the recorded frame: for log laws of the shapes of the mark, zone and gold laws, the log mean of a batch that came in
    the recorded frame (answer ``a``, answer of its swap ``b``) is, bit for bit, what the swapped batch (answers ``b`` and ``a``, its rows came in the
    swapped frame) gives after the axes are moved. A mean formed in the frame of each batch would compute the same pair of values at different places of
    the buffer, where PyTorch's vectorised exp and log1p round the value differently (the sizes of this test have such places). It is also the log of the
    mean of the two laws."""
    g = torch.Generator().manual_seed(rows)
    cases = (((rows, 18, 52), lambda t: heads._back(t, -1, heads._MARK)),
             ((rows, 18, 52, 10), lambda t: heads._back(heads._back(t, 2, heads._MARK), 3, heads._ZONE)),
             ((rows, 4, 2, 6, 31), lambda t: heads._back(t, 2, heads._TEAM)))
    came_in_recorded, came_in_swapped = torch.zeros(rows, dtype=torch.bool), torch.ones(rows, dtype=torch.bool)
    for shape, back in cases:
        a, b = (torch.log_softmax(torch.randn(shape, generator=g), -1) for _ in range(2))
        first = heads._log_mean(a, b, came_in_recorded, back)
        second = heads._log_mean(b, a, came_in_swapped, back)
        assert torch.equal(back(second), first) and torch.equal(back(first), second), shape
        assert torch.allclose(first.exp(), 0.5 * (a.exp() + back(b).exp()), atol=1e-6, rtol=0), shape
        mixed_sides = torch.arange(rows) % 2 == 1                                          # rows of both kinds in one batch
        both = heads._log_mean(a, b, mixed_sides, back)
        assert torch.allclose(both.exp(), 0.5 * (a.exp() + back(b).exp()), atol=1e-6, rtol=0), shape


def test_the_average_never_scores_worse_than_the_mean_of_its_two_frames():
    """(EH-9), the convexity claim: -log p is convex in p and mu - y log mu in mu, so the NLL of the frame-averaged law or mean is not above the mean of
    the NLLs of the two frames (the swapped frame scored with the swapped target, which is the same cell of the original frame). Checked cell by cell
    for the mark, zone, size and gold laws and for the Poisson means, and strictly below somewhere: the two frames of this model really differ."""
    model = tiny_model(21)
    fa = heads.FrameAveraged(model)
    batch = encoder_batch(views=4, events=9, edges=7, seed=6)
    c = conditions(4, seed=12)
    one = torch.ones(4, 18)
    with torch.no_grad():
        for layer in (model.heads.mark_net.out, model.heads.zone_net.out, model.heads.size_out, model.heads.gold_out, model.heads.kill_out,
                      model.heads.rate_net.out):                                     # outputs of a trained size: the laws of a new model are nearly uniform
            layer.weight.mul_(40.0)
        enc = fa.encode(batch)
        avg, here, there = fa.m2(enc, one), model.heads.m2(enc.original, one), model.heads.m2(enc.swapped, one)
        s_avg, s_here = fa.m3(enc, c['q'], c['m'], c['z']), model.heads.m3(enc.original, c['q'], c['m'], c['z'])
        s_there = model.heads.m3(enc.swapped, c['q'], torch.as_tensor(data.MARK_PERM)[c['m']], torch.where(c['z'] >= 0, ZONE[c['z'].clamp(min=0)], c['z']))
    laws = [(avg.mark_logp, here.mark_logp, there.mark_logp[..., MARK]), (avg.zone_logp, here.zone_logp, there.zone_logp[:, :, MARK][:, :, :, ZONE]),
            (s_avg.size_logp, s_here.size_logp, s_there.size_logp), (s_avg.gold_logp, s_here.gold_logp, s_there.gold_logp[:, :, TEAM])]
    for averaged, first, second in laws:
        gap = 0.5 * (-first - second) - (-averaged)                                  # mean NLL of the frames minus NLL of the average: not negative
        assert gap.min() > -1e-6 and gap.max() > 1e-4
    y = torch.randint(0, 4, (4, 18)).float()                                         # counts: the Poisson NLL of a mean mu is mu - y log mu
    for averaged, first, second in ((avg.rate, here.rate, there.rate), (s_avg.kill_rate, s_here.kill_rate, s_there.kill_rate[..., TEAM])):
        target = y if averaged.shape == y.shape else y[:, :8].reshape(4, 4, 2)
        nll = lambda mu: mu - target * mu.log()                                      # noqa: E731
        gap = 0.5 * (nll(first) + nll(second)) - nll(averaged)
        assert gap.min() > -1e-5 and gap.max() > 1e-6


class StubHeads:
    """Heads with known answers: one fixed table for each frame, ``m2_out[f]`` for the views whose side marker is f (the pooled state of the stub
    encoding is the side marker of each view). Every call is recorded with its conditions. The tables are not normalised: only the averaging is
    under test."""

    def __init__(self, n: int):
        g = torch.Generator().manual_seed(11)
        rnd = lambda *shape: torch.randn(*shape, generator=g)                                     # noqa: E731
        pos = lambda *shape: 0.5 + torch.rand(*shape, generator=g)                                # noqa: E731
        self.m2_out = [heads.M2Out(pos(n, 18), rnd(n, 18, 52), rnd(n, 18, 52, 10)) for _ in range(2)]
        self.m3_out = [heads.M3Out(rnd(n, 4), pos(n, 4, 2), rnd(n, 4, 2, 6, 31)) for _ in range(2)]
        self.obj_out, self.conv_out = [pos(n, 18) for _ in range(2)], [pos(n, 18) for _ in range(2)]
        self.calls = []

    @staticmethod
    def pick(tables, frame: torch.Tensor, pos):
        """The answer of each view in the frame ``frame`` (n,) of that view, for the rows ``pos`` (all views in order if None); ``tables`` is the pair
        of tables (tensors or tuples of tensors) of the frames 0 and 1."""
        rows = torch.arange(len(frame))

        def one(a, b):
            out = torch.stack([a, b])[frame, rows]
            return out if pos is None else out[pos]

        first, second = tables
        return type(first)(*(one(a, b) for a, b in zip(first, second))) if isinstance(first, tuple) else one(first, second)

    def m2(self, enc, exposure, pos=None):
        self.calls.append(('m2', {'exposure': exposure, 'pos': pos}))
        return self.pick(self.m2_out, enc.h_pool[:, 0].long(), pos)

    def m3(self, enc, q, m, z, pos=None):
        self.calls.append(('m3', {'q': q, 'm': m, 'z': z, 'pos': pos}))
        return self.pick(self.m3_out, enc.h_pool[:, 0].long(), pos)

    def m3_obj(self, enc, q, m, z, s, K, pos=None):
        self.calls.append(('m3_obj', {'q': q, 'm': m, 'z': z, 's': s, 'K': K, 'pos': pos}))
        return self.pick(self.obj_out, enc.h_pool[:, 0].long(), pos)

    def m4(self, enc, q, m, z, K, O, exposure, pos=None):                                     # noqa: E741
        self.calls.append(('m4', {'q': q, 'm': m, 'z': z, 'K': K, 'O': O, 'exposure': exposure, 'pos': pos}))
        return self.pick(self.conv_out, enc.h_pool[:, 0].long(), pos)


class StubModel(torch.nn.Module):
    """The model of the stub heads: its encoding of a view is its side marker (so the heads know the frame of each view) and it records the number of
    views and the shapes of every call."""

    def __init__(self, n: int):
        super().__init__()
        self.heads, self.encoded = StubHeads(n), []

    def encode(self, batch):
        n = batch['side'].shape[0]
        self.encoded.append((n, {k: tuple(v.shape) for k, v in batch.items()}, float(batch['side'][0])))
        return encoder.Encoded(torch.zeros(n, 55, 4), batch['side'][:, None].expand(n, 4).clone(), torch.ones(n, 55, dtype=torch.bool))


def stub_batch(n: int = 3) -> dict:
    batch = encoder_batch(views=n, events=5, edges=4, seed=0)
    batch['side'] = torch.zeros(n)
    return batch


def test_frame_average_hand_case_on_a_stub():
    """(EH-9) on tables that are known: the rates and the kill counts are averaged arithmetically, the laws are averaged as probabilities and then logged
    (``logaddexp``), after the swapped answer has been moved back by the table of each of its axes: the marks of ``mark_logp`` and the first two axes of
    the zone law by ``MARK_PERM``, the zones by ``ZONE_PERM``, the teams of ``kill_rate`` and ``gold_logp`` by ``TEAM_PERM``, the cells of the objectives and
    the conversions by ``CELL_PERM``; the rate of M2 and the size law have no axis to move."""
    model, batch = StubModel(3), stub_batch()
    fa, stub = heads.FrameAveraged(model), model.heads
    enc = fa.encode(batch)
    c = conditions(3)
    a, b = stub.m2_out
    out = fa.m2(enc, c['exposure'])
    assert torch.allclose(out.rate, 0.5 * (a.rate + b.rate), atol=1e-6, rtol=0)
    assert torch.allclose(out.mark_logp, logmean(a.mark_logp, b.mark_logp[..., MARK]), atol=1e-6, rtol=0)
    assert torch.allclose(out.zone_logp, logmean(a.zone_logp, b.zone_logp[:, :, MARK][:, :, :, ZONE]), atol=1e-6, rtol=0)
    a, b = stub.m3_out
    out = fa.m3(enc, c['q'], c['m'], c['z'])
    assert torch.allclose(out.size_logp, logmean(a.size_logp, b.size_logp), atol=1e-6, rtol=0)
    assert torch.allclose(out.kill_rate, 0.5 * (a.kill_rate + b.kill_rate[..., TEAM]), atol=1e-6, rtol=0)
    assert torch.allclose(out.gold_logp, logmean(a.gold_logp, b.gold_logp[:, :, TEAM]), atol=1e-6, rtol=0)
    a, b = stub.obj_out
    assert torch.allclose(fa.m3_obj(enc, c['q'], c['m'], c['z'], c['s'], c['K']), 0.5 * (a + b[:, CELL]), atol=1e-6, rtol=0)
    a, b = stub.conv_out
    assert torch.allclose(fa.m4(enc, c['q'], c['m'], c['z'], c['K'], c['O'], c['window']), 0.5 * (a + b[:, CELL]), atol=1e-6, rtol=0)
    # the tables differ from the identity where they should, so that a call that forgot a map would not pass
    assert not torch.equal(MARK, torch.arange(52)) and not torch.equal(ZONE, torch.arange(10)) and not torch.equal(TEAM, torch.arange(2))
    assert not torch.equal(CELL, torch.arange(18))


def test_the_swapped_call_gets_the_conditions_of_the_swapped_frame():
    """The heads of the swapped frame are asked about the swapped marks, zones (-1 stays), kills and objectives; the bin, the size, the exposure and the
    view numbers do not change; an open reading is open in both frames. The swapped conditions come from the references of ``symmetry`` and ``terrain``."""
    model, batch = StubModel(3), stub_batch()
    fa, stub = heads.FrameAveraged(model), model.heads
    enc, c = fa.encode(batch), conditions(3, seed=5)
    s, pos = swapped_conditions(c), torch.tensor([2, 0, 1])
    assert (c['z'] == -1).any() and (c['m'] >= 50).any()
    fa.m2(enc, c['exposure'], pos)
    fa.m3(enc, c['q'], c['m'], c['z'], pos)
    fa.m3_obj(enc, c['q'], c['m'], c['z'], c['s'], c['K'], pos)
    fa.m4(enc, c['q'], c['m'], c['z'], c['K'], c['O'], c['window'], pos)
    fa.m3(enc, None, None, None)
    fa.m3_obj(enc, None, None, None, c['s'], c['K'])
    fa.m4(enc, None, None, None, c['K'], c['O'], c['window'])
    assert [name for name, _ in stub.calls] == ['m2', 'm2', 'm3', 'm3', 'm3_obj', 'm3_obj', 'm4', 'm4', 'm3', 'm3', 'm3_obj', 'm3_obj', 'm4', 'm4']
    expected = [('m2', {'exposure': c['exposure'], 'pos': pos}, {'exposure': c['exposure'], 'pos': pos}),
                ('m3', {k: c[k] for k in 'qmz'} | {'pos': pos}, {'q': c['q'], 'm': s['m'], 'z': s['z'], 'pos': pos}),
                ('m3_obj', {k: c[k] for k in ('q', 'm', 'z', 's', 'K')} | {'pos': pos}, {'q': c['q'], 'm': s['m'], 'z': s['z'], 's': c['s'], 'K': s['K'], 'pos': pos}),
                ('m4', {k: c[k] for k in ('q', 'm', 'z', 'K', 'O')} | {'exposure': c['window'], 'pos': pos},
                 {'q': c['q'], 'm': s['m'], 'z': s['z'], 'K': s['K'], 'O': s['O'], 'exposure': c['window'], 'pos': pos}),
                ('m3', {'q': None, 'm': None, 'z': None, 'pos': None}, {'q': None, 'm': None, 'z': None, 'pos': None}),
                ('m3_obj', {'q': None, 'm': None, 'z': None, 's': c['s'], 'K': c['K'], 'pos': None},
                 {'q': None, 'm': None, 'z': None, 's': c['s'], 'K': s['K'], 'pos': None}),
                ('m4', {'q': None, 'm': None, 'z': None, 'K': c['K'], 'O': c['O'], 'exposure': c['window'], 'pos': None},
                 {'q': None, 'm': None, 'z': None, 'K': s['K'], 'O': s['O'], 'exposure': c['window'], 'pos': None})]
    for i, (name, first, second) in enumerate(expected):
        for seen, want in ((stub.calls[2 * i], first), (stub.calls[2 * i + 1], second)):
            assert seen[0] == name and seen[1].keys() == want.keys(), (i, name)
            for key, value in want.items():
                got = seen[1][key]
                assert (got is None and value is None) or torch.equal(got, value), (i, name, key)


def test_the_model_is_run_twice_in_separate_calls_of_one_shape():
    """The two frames are two calls of the encoder, each on a batch of the same views and the same shapes (never one stacked batch of both: a row
    that sits at another place of a batch is rounded differently, and the exact identity needs the same arithmetic for the same row in either
    order): the second call is the swap of the first (side marker flipped), and the keys are the view keys only."""
    model, batch = StubModel(3), stub_batch()
    fa = heads.FrameAveraged(model)
    batch['M2Mark/masked/m'] = torch.zeros(7, dtype=torch.long)                      # a row block of a ``Batches`` batch: not an input of the encoder
    fa.encode(batch)
    assert len(model.encoded) == 2
    (n0, shapes0, side0), (n1, shapes1, side1) = model.encoded
    assert (n0, n1) == (3, 3) and shapes0 == shapes1 and (side0, side1) == (0.0, 1.0)
    assert set(shapes0) == set(data.VIEW_KEYS)


def test_frame_average_needs_a_model_with_an_encoder_and_heads():
    with pytest.raises(TypeError, match='encode'):
        heads.FrameAveraged(torch.nn.Linear(2, 2))


def test_the_frame_average_shares_the_parameters_and_the_mode_of_its_model():
    """The wrapper adds no weight (the two frames use the one model) and carries the training mode of the model."""
    model = tiny_model(22)
    fa = heads.FrameAveraged(model)
    assert [id(p) for p in fa.parameters()] == [id(p) for p in model.parameters()] and len(list(fa.parameters())) > 0
    fa.train()
    assert model.training and model.heads.training and model.encoder.training
    fa.eval()
    assert not model.training and not model.encoder.blocks[0].training


def test_the_frame_average_is_the_same_for_a_batch_recorded_in_either_frame_and_for_mixed_rows():
    """The log laws are averaged in the recorded frame (side 0) of every row and moved back for a row that came in the swapped frame. On the stub
    this means, row by row, the numbers of the plain formula whichever side the row has: the answer for the frame in which the view came in, and the
    swapped answer moved back. The rows of a mixed batch do not disturb each other, and the side of a row is that of its view (``pos``)."""
    c = conditions(3)
    for sides in ((0, 0, 0), (1, 1, 1), (0, 1, 0)):
        model, batch = StubModel(3), stub_batch()
        batch['side'] = torch.tensor(sides, dtype=torch.float32)
        fa, stub = heads.FrameAveraged(model), model.heads
        enc, side = fa.encode(batch), torch.tensor(sides)
        assert enc.side.tolist() == [s > 0 for s in sides]
        a, b = stub.pick(stub.m2_out, side, None), stub.pick(stub.m2_out, 1 - side, None)             # the answer of each view's frame, and of its swap
        out = fa.m2(enc, c['exposure'])
        assert torch.allclose(out.mark_logp, logmean(a.mark_logp, b.mark_logp[..., MARK]), atol=1e-6, rtol=0), sides
        assert torch.allclose(out.zone_logp, logmean(a.zone_logp, b.zone_logp[:, :, MARK][:, :, :, ZONE]), atol=1e-6, rtol=0), sides
        pos = torch.tensor([2, 0])
        a, b = stub.pick(stub.m3_out, side, pos), stub.pick(stub.m3_out, 1 - side, pos)
        picked = fa.m3(enc, c['q'][:2], c['m'][:2], c['z'][:2], pos=pos)
        assert torch.allclose(picked.size_logp, logmean(a.size_logp, b.size_logp), atol=1e-6, rtol=0), sides
        assert torch.allclose(picked.gold_logp, logmean(a.gold_logp, b.gold_logp[:, :, TEAM]), atol=1e-6, rtol=0), sides
        assert torch.allclose(picked.kill_rate, 0.5 * (a.kill_rate + b.kill_rate[..., TEAM]), atol=1e-6, rtol=0), sides


def test_the_conversion_window_is_a_number_or_one_value_per_row():
    """Ruling R-S4a-11: the exposure of M4 is the length of the conversion window, one value for the whole row (registry ``e``): a number or a tensor of
    shape (n,). A tensor with a value for each cell is refused, by the heads and by the frame average (whatever its heads accept: the stub's accept
    anything). The average moves the cells of the swapped answer to their partner cells, so a window of its own for each cell would be moved with
    them: with 10 s for the blue taker and 1 s for the red taker the relative error of a conversion mean was 4.9."""
    model, enc, c = tiny_model(23), fabricated(4, seed=3), conditions(4, seed=13)
    H, fa = model.heads, heads.FrameAveraged(model)
    args = (c['q'], c['m'], c['z'], c['K'], c['O'])
    for ok in (7.5, torch.tensor(7.5), torch.full((4,), 7.5)):                                             # a number, a 0-d tensor, one value for each row
        assert torch.equal(H.m4(enc, *args, ok), H.m4(enc, *args, 7.5)), ok
    stub = heads.FrameAveraged(StubModel(4))
    stub_enc = stub.encode(stub_batch(4))
    real_enc = fa.encode(encoder_batch(views=4, events=9, edges=7, seed=23))
    for bad in (torch.ones(4, 18), torch.ones(4, 1), torch.ones(1, 4), torch.ones(4, 18, 1), torch.ones(5), torch.ones(3)):
        with pytest.raises(ValueError, match='conversion window'):
            H.m4(enc, *args, bad)
        with pytest.raises(ValueError, match='conversion window'):
            fa.m4(real_enc, *args, bad)
        with pytest.raises(ValueError, match='conversion window'):
            stub.m4(stub_enc, *args, bad)                                                                  # the frame average refuses it itself
    for ok in (7.5, torch.tensor(7.5), torch.full((4,), 7.5)):
        assert fa.m4(real_enc, *args, ok).shape == (4, 18) and stub.m4(stub_enc, *args, ok).shape == (4, 18)


def test_the_frame_averaged_conversions_are_linear_in_the_window():
    """A window is one value for the row, so the frame-averaged conversions are that value times the frame-averaged rate: ``FA.m4(E) = E x FA.m4(1)``
    up to the rounding of one product. Checked in both readings, for a batch of both kinds of side, for windows given as (n,), as a number and with
    ``pos`` (rows that share a view). The conversions of the blue and of the red taker differ, so a window that was moved to the wrong cell would show."""
    model = tiny_model(24)
    fa = heads.FrameAveraged(model)
    batch = encoder_batch(views=6, events=9, edges=7, seed=24)
    batch['side'] = (torch.arange(6) % 2).float()
    c, c7, pos = conditions(6, seed=14), conditions(7, seed=17), torch.tensor([3, 0, 0, 5, 1, 3, 2])
    with torch.no_grad():
        enc = fa.encode(batch)
        for q, m, z, cc, p in ((c['q'], c['m'], c['z'], c, None), (None, None, None, c, None), (c7['q'], c7['m'], c7['z'], c7, pos), (None, None, None, c7, pos)):
            n = len(cc['K'])
            unit = fa.m4(enc, q, m, z, cc['K'], cc['O'], 1.0, p)
            assert (unit[:, 0::2] - unit[:, 1::2]).abs().max() > 1e-3                                      # blue and red taker cells differ
            for window in (cc['window'], torch.full((n,), 7.5), 7.5):
                got = fa.m4(enc, q, m, z, cc['K'], cc['O'], window, p)
                want = unit * (window[:, None] if isinstance(window, torch.Tensor) else window)
                assert torch.allclose(got, want, rtol=1e-6, atol=0), (q is None, p is None)


@pytest.mark.parametrize('method', ['m3', 'm3_obj', 'm4'])
def test_the_frame_average_refuses_bad_conditions_before_it_swaps_them(method):
    """The swap of the frame average indexes tables by the marks, the zones, K and O (``data.swap``). A value out of range, a float, a mask or a uint8
    (which indexes as a mask) must be a ValueError of the documented kind, raised before the swap: not an IndexError in it, and not a silently wrong
    row of a table. The conditions that are fine are cast (an int8 mark gives the answer of the int64 one)."""
    model, c = tiny_model(25), conditions(5, seed=15)
    fa = heads.FrameAveraged(model)
    with torch.no_grad():
        enc = fa.encode(encoder_batch(views=5, events=9, edges=7, seed=25))
        run = {'m3': lambda c: fa.m3(enc, c['q'], c['m'], c['z']), 'm3_obj': lambda c: fa.m3_obj(enc, c['q'], c['m'], c['z'], c['s'], c['K']),
               'm4': lambda c: fa.m4(enc, c['q'], c['m'], c['z'], c['K'], c['O'], c['window'])}[method]
        run(c)                                                                                                     # the good conditions work
        put = lambda t, v: torch.cat([t[:1], torch.tensor([v], dtype=t.dtype), t[2:]])                           # noqa: E731  (a copy with the value of row 1 changed)
        cases = [('z', put(c['z'], 15), 'z must lie'), ('z', put(c['z'], 10), 'z must lie'), ('z', put(c['z'], -2), 'z must lie'), ('m', put(c['m'], 52), 'm must lie'),
                 ('m', put(c['m'], -1), 'm must lie'), ('q', put(c['q'], 18), 'q must lie'), ('q', put(c['q'], -1), 'q must lie'),
                 ('m', c['m'].float(), 'm must be an integer tensor'), ('z', c['z'].double(), 'z must be an integer tensor'),
                 ('m', c['m'].to(torch.uint8), 'm must be an integer tensor'), ('z', (c['z'] + 1).to(torch.uint8), 'z must be an integer tensor'),
                 ('m', c['m'].bool(), 'm must be an integer tensor'), ('m', c['m'][:3], 'm must be an integer tensor'),
                 ('q', c['q'].float(), 'q must be an integer tensor')]
        if method == 'm3_obj':
            cases += [('s', put(c['s'], 4), 's must lie'), ('s', put(c['s'], -1), 's must lie'), ('s', c['s'].float(), 's must be an integer tensor'),
                      ('s', c['s'].to(torch.uint8), 's must be an integer tensor')]
        if method in ('m3_obj', 'm4'):
            cases += [('K', c['K'][:, :3], 'K must be a tensor of shape'), ('K', torch.cat([c['K'], c['K']], 1), 'K must be a tensor of shape'),
                      ('K', -c['K'] - 1.0, 'K must be finite and not negative'), ('K', c['K'] * float('nan'), 'K must be finite and not negative')]
        if method == 'm4':
            cases += [('O', c['O'][:, :17], 'O must be a tensor of shape'), ('O', torch.cat([c['O'], c['O']], 1), 'O must be a tensor of shape'),
                      ('O', -c['O'] - 1.0, 'O must be finite and not negative')]
        for name, value, message in cases:
            with pytest.raises(ValueError, match=message):
                run({**c, name: value})
        for partly in ({'q': None}, {'m': None}, {'z': None}, {'q': None, 'm': None}):                            # a masked reading with a part missing
            with pytest.raises(ValueError, match='given together'):
                run({**c, **partly})
        for narrow in (torch.int8, torch.int16, torch.int32):                                                     # other integer types are cast
            got, want = run({**c, 'm': c['m'].to(narrow), 'z': c['z'].to(narrow), 'q': c['q'].to(narrow)}), run(c)
            assert all(torch.equal(x, y) for x, y in (zip(got, want) if isinstance(got, tuple) else [(got, want)])), narrow


# =====================================================================================================================================
# forms of the heads
# =====================================================================================================================================
def slot(s: int, t: int, layer: int) -> int:
    """The row of ``gold_slot`` of the size s, team t and kill layer index ``layer`` (kills - 1, 0 .. 5): the order of the axes of ``gold_logp``."""
    return (s * 2 + t) * 6 + layer


def test_the_gold_law_reads_only_the_teams_own_kill_layer():
    """Review of the composer and plan P4: the kill-gold law of a team is indexed by the size, the team and that team's own kill layer, and by nothing
    of the other team. ``compose`` asks both teams at the same layer and keeps the law of each, which is right only then. (a) Changing the parameters of one
    place (size 2, blue-red team 1, layer 4 of 1 .. 6) changes exactly that place of ``gold_logp`` and no other, also none of the other team and none of
    the other layers of the team. (b) The gradient of one entry reaches the parameters of its own place only. (c) The axes are size, team, layer, bin."""
    model, enc, c = tiny_model(9), fabricated(3), conditions(3, seed=2)
    H = model.heads
    with torch.no_grad():
        before = H.m3(enc, c['q'], c['m'], c['z']).gold_logp
        assert before.shape == (3, 4, 2, 6, 31)
        saved = H.gold_slot.weight[slot(2, 1, 3)].clone()
        H.gold_slot.weight[slot(2, 1, 3)] += 1.0
        after = H.m3(enc, c['q'], c['m'], c['z']).gold_logp
        H.gold_slot.weight[slot(2, 1, 3)].copy_(saved)
    moved = (after != before).any(-1).any(0)                                      # (4, 2, 6): the places whose law moved, in any row
    want = torch.zeros(4, 2, 6, dtype=torch.bool)
    want[2, 1, 3] = True
    assert torch.equal(moved, want)
    H.zero_grad()
    H.m3(enc, c['q'], c['m'], c['z']).gold_logp[0, 1, 0, 4, 7].backward()
    reached = H.gold_slot.weight.grad.abs().sum(1) > 0
    assert reached.nonzero().flatten().tolist() == [slot(1, 0, 4)]
    assert (before[:, 1, 0, 4] != before[:, 1, 1, 4]).any() and (before[:, 1, 0, 4] != before[:, 1, 0, 3]).any()           # the places differ


def test_the_kill_rate_has_the_axes_size_then_team():
    """``kill_rate`` (n, 4, 2) is size, then team: a change of the output bias of the place (size 3, team 0) moves exactly that mean."""
    model, enc, c = tiny_model(10), fabricated(2), conditions(2, seed=3)
    H = model.heads
    base = H.m3(enc, c['q'], c['m'], c['z']).kill_rate
    with torch.no_grad():
        H.kill_out.bias[3 * 2 + 0] += 0.5
    moved = H.m3(enc, c['q'], c['m'], c['z']).kill_rate
    changed = (moved != base).any(0)
    assert changed.nonzero().tolist() == [[3, 0]] and torch.allclose(moved[:, 3, 0], base[:, 3, 0] * math.exp(0.5), rtol=1e-6)


def test_the_mark_tables_name_the_killer_and_the_victim():
    """The 50 pair marks are (killer team, killer slot, victim slot) of ``units.first_kill_mark``: the player rows of the killer and of the victim are
    the rows of the two participants (row = participant - 1, rows 0-4 blue, 5-9 red), and every mark is reached once. An execution has no pair: its
    embedding reads the mean state of the victim's team only."""
    seen = []
    for killer in range(1, 11):
        for victim in range(1, 11):
            if (killer <= 5) == (victim <= 5):
                continue
            mark = units.first_kill_mark(killer, victim)
            assert heads.MARK_KILLER_ROW[mark] == killer - 1 and heads.MARK_VICTIM_ROW[mark] == victim - 1
            seen.append(mark)
    assert sorted(seen) == list(range(50)) and heads.N_PAIR_MARKS == 50
    assert units.first_kill_mark(0, 3) == 50 and units.first_kill_mark(0, 8) == 51                                       # the executions
    H, enc = tiny_model(11).double().heads, fabricated(2, dtype=torch.float64, seed=4)
    players, team = enc.h_tok[:, :10], enc.h_tok[:, :10].reshape(2, 2, 5, 16).mean(2)
    base = H.mark(players, team, torch.tensor([50, 51]))
    other = enc.h_tok[:, :10].clone()
    other[:, 5:] += 3.0                                                                                                 # the red players change
    new = H.mark(other, other.reshape(2, 2, 5, 16).mean(2), torch.tensor([50, 51]))
    assert torch.equal(new[0], base[0]) and not torch.allclose(new[1], base[1])                                         # blue victim: red is no input


def plain_mark_embedding(H, players: torch.Tensor, m: int) -> torch.Tensor:
    """(EH-2) for one mark of one row by hand from the definition mark = (killer team 5 + killer slot) 5 + victim slot; ``players`` (10, d)."""
    kind = H.mark.kind.weight
    if m < 50:
        team, killer, victim = m // 25, (m % 25) // 5, m % 5
        return H.mark.killer.weight @ players[5 * team + killer] + H.mark.victim.weight @ players[5 * (1 - team) + victim] + kind[0]
    return H.mark.team.weight @ players[5 * (m - 50):5 * (m - 50) + 5].mean(0) + kind[1]


def test_the_mark_embedding_is_the_equation_for_every_mark():
    """(EH-2) by hand for all 52 marks of two rows, against the table of all marks (``every``) and the one for a single mark (``forward``)."""
    H, enc = tiny_model(12).double().heads, fabricated(2, dtype=torch.float64, seed=6)
    players = enc.h_tok[:, :10]
    team = players.reshape(2, 2, 5, 16).mean(2)
    table = H.mark.every(players, team)
    assert table.shape == (2, 52, 16)
    for row in range(2):
        for m in range(52):
            by_hand = plain_mark_embedding(H, players[row], m)
            assert torch.allclose(table[row, m], by_hand, atol=1e-12, rtol=0), (row, m)
            assert torch.allclose(H.mark(players, team, torch.full((2,), m))[row], by_hand, atol=1e-12, rtol=0), (row, m)
    assert torch.allclose(H.mark.absent(2), H.mark.kind.weight[2].expand(2, 16))
    assert (table[:, 0] - table[:, 25]).abs().max() > 1e-4                                                             # the killer's team matters


def test_the_mark_law_follows_the_players_when_a_team_is_renumbered():
    """Design 5.2: the slot of a player is no role. If the rows of a team are renumbered (new row r holds old row perm[r], inside each team), the mark
    law, the zone law and the size and the kills of M3 of every mark follow the players: new mark m' has the numbers of the old mark that names the
    same two players. Float64, so that a wrong cell cannot hide behind rounding (1e-12)."""
    model = tiny_model(3).double()
    H, enc, c = model.heads, fabricated(4, dtype=torch.float64, seed=5), conditions(4, seed=7, dtype=torch.float64)
    g = torch.Generator().manual_seed(1)
    perm = torch.cat([torch.randperm(5, generator=g), 5 + torch.randperm(5, generator=g)])
    assert not torch.equal(perm, torch.arange(10))
    h_tok = enc.h_tok.clone()
    h_tok[:, :10] = enc.h_tok[:, :10][:, perm]
    renumbered = encoder.Encoded(h_tok, enc.h_pool, enc.mask)
    old = torch.arange(52)
    for m in range(50):
        killer, victim = perm[heads.MARK_KILLER_ROW[m]], perm[heads.MARK_VICTIM_ROW[m]]                              # the old rows of the two players
        old[m] = 25 * (killer // 5) + 5 * (killer % 5) + victim % 5
    assert not torch.equal(old, torch.arange(52))
    e = c['exposure'].double()
    new, was = H.m2(renumbered, e), H.m2(enc, e)
    assert torch.allclose(new.rate, was.rate, atol=1e-12, rtol=0)
    assert torch.allclose(new.mark_logp, was.mark_logp[:, :, old], atol=1e-12, rtol=0)
    assert torch.allclose(new.zone_logp, was.zone_logp[:, :, old], atol=1e-12, rtol=0)
    assert (new.mark_logp - was.mark_logp).abs().max() > 1e-4                                                         # the test can fail
    m3_new, m3_was = H.m3(renumbered, c['q'], c['m'], c['z']), H.m3(enc, c['q'], old[c['m']], c['z'])
    for a, b in zip(m3_new, m3_was):
        assert torch.allclose(a, b, atol=1e-12, rtol=0)


def test_each_condition_reaches_the_heads_that_the_plan_gives_it_to():
    """Plan Q7 and design 3.3: M2's mark law reads the players and the bin and not the pooled state, its rate reads the pooled state and the bin, its zone
    law reads all three; M3 reads (q, m, z); M3's objectives also the size and K; M4 (q, m, z) and K and O. Each input moves the outputs it should (a dead
    input, such as a condition added to the wrong sum, would not)."""
    model = tiny_model(4).double()
    H, enc, c = model.heads, fabricated(4, dtype=torch.float64, seed=8), conditions(4, seed=9, dtype=torch.float64)
    moved = lambda a, b: float((a - b).abs().max().detach()) > 1e-9                                              # noqa: E731
    one = torch.ones(4, 18, dtype=torch.float64)
    more = {'q': (c['q'] + 1) % 18, 'm': (c['m'] + 3) % 52, 'z': (c['z'] + 4) % 11 - 1, 's': (c['s'] + 1) % 4, 'K': c['K'] + 1.0, 'O': c['O'] + 1.0}
    # M2: which input reaches which output
    pool = encoder.Encoded(enc.h_tok, enc.h_pool + 0.5, enc.mask)
    base, moved_pool = H.m2(enc, one), H.m2(pool, one)
    assert moved(base.rate, moved_pool.rate) and moved(base.zone_logp, moved_pool.zone_logp) and not moved(base.mark_logp, moved_pool.mark_logp)
    players = enc.h_tok.clone()
    players[:, :10] += 0.5
    moved_players = H.m2(encoder.Encoded(players, enc.h_pool, enc.mask), one)
    assert moved(base.mark_logp, moved_players.mark_logp) and moved(base.zone_logp, moved_players.zone_logp) and not moved(base.rate, moved_players.rate)
    # M3, the objectives and M4: every condition moves the outputs
    base3 = H.m3(enc, c['q'], c['m'], c['z'])
    obj = H.m3_obj(enc, c['q'], c['m'], c['z'], c['s'], c['K'])
    conv = H.m4(enc, c['q'], c['m'], c['z'], c['K'], c['O'], c['window'])
    for name in 'qmz':
        other = {**c, name: more[name]}
        assert all(moved(a, b) for a, b in zip(base3, H.m3(enc, other['q'], other['m'], other['z']))), name
        assert moved(obj, H.m3_obj(enc, other['q'], other['m'], other['z'], c['s'], c['K'])), name
        assert moved(conv, H.m4(enc, other['q'], other['m'], other['z'], c['K'], c['O'], c['window'])), name
    assert moved(obj, H.m3_obj(enc, c['q'], c['m'], c['z'], more['s'], c['K'])) and moved(obj, H.m3_obj(enc, c['q'], c['m'], c['z'], c['s'], more['K']))
    assert moved(conv, H.m4(enc, c['q'], c['m'], c['z'], more['K'], c['O'], c['window'])) and moved(conv, H.m4(enc, c['q'], c['m'], c['z'], c['K'], more['O'], c['window']))
    # the open reading is another reading of the same heads: it differs from the masked one
    assert all(moved(a, b) for a, b in zip(base3, H.m3(enc, None, None, None)))


def mixed(p: torch.Tensor) -> torch.Tensor:
    """(EH-4) by hand: log((1 - 1e-6) p + 1e-6 / C) for a probability row ``p`` (C,)."""
    return torch.log((1 - MIX) * p + MIX / p.shape[-1])


def test_the_rate_the_mark_law_and_the_zone_law_are_the_equations():
    """(EH-1), (EH-3), (EH-5) and (EH-4) against a plain computation from the weights (float64, 1e-12), for several rows, bins and marks, with the mark
    embedding of (EH-2) written out by hand from the definition of a mark."""
    H, enc = tiny_model(13).double().heads, fabricated(3, dtype=torch.float64, seed=3)
    exposure = (1.0 + torch.rand(3, 18, generator=torch.Generator().manual_seed(4), dtype=torch.float64) * 9)
    out = H.m2(enc, exposure)
    assert out.rate.dtype == out.mark_logp.dtype == out.zone_logp.dtype == torch.float64
    for row in (0, 2):
        h, players = enc.h_pool[row], enc.h_tok[row, :10]
        emb_m = torch.stack([plain_mark_embedding(H, players, m) for m in range(52)])                                  # (52, d)
        for q in (0, 9, 17):
            emb_q = H.bin_emb.weight[q]
            net = H.rate_net                                                                                           # (EH-1)
            eta = net.out(F.gelu(net.maps[0](h) + net.maps[1](emb_q) + net.bias))
            assert torch.allclose(out.rate[row, q], torch.exp(eta[0]) * exposure[row, q], rtol=1e-12, atol=0), (row, q)
            net = H.mark_net                                                                                           # (EH-3) with (EH-4)
            score = net.out(F.gelu(net.maps[0](emb_m) + net.maps[1](emb_q) + net.bias))[:, 0]                          # (52,)
            assert torch.allclose(out.mark_logp[row, q], mixed(torch.softmax(score, 0)), atol=1e-12, rtol=0), (row, q)
            net = H.zone_net                                                                                           # (EH-5) with (EH-4)
            for m in (0, 7, 30, 50, 51):
                logits = net.out(F.gelu(net.maps[0](emb_m[m]) + net.maps[1](emb_q) + net.maps[2](h) + net.bias))
                assert torch.allclose(out.zone_logp[row, q, m], mixed(torch.softmax(logits, 0)), atol=1e-12, rtol=0), (row, q, m)


def test_the_m3_laws_are_the_equation():
    """(EH-6) and (EH-4) against a plain computation from the weights (float64, 1e-12): the size law, the kill means and the gold law of every size,
    team and kill layer, in the masked and in the open reading."""
    H, enc, c = tiny_model(14).double().heads, fabricated(2, dtype=torch.float64, seed=2), conditions(2, seed=1, dtype=torch.float64)
    for masked in (True, False):
        q, m, z = (c['q'], c['m'], c['z']) if masked else (None, None, None)
        out = H.m3(enc, q, m, z)
        for row in range(2):
            h, players = enc.h_pool[row], enc.h_tok[row, :10]
            if masked:
                emb_q, emb_z, emb_m = H.bin_emb.weight[int(q[row])], H.zone_emb.weight[int(z[row]) + 1], plain_mark_embedding(H, players, int(m[row]))
            else:
                emb_q, emb_z, emb_m = H.bin_emb.weight[18], H.zone_emb.weight[11], H.mark.kind.weight[2]                # the values of "none"
            net = H.m3_net
            trunk = F.gelu(net.maps[0](h) + net.maps[1](emb_q) + net.maps[2](emb_m) + net.maps[3](emb_z) + net.bias)
            assert torch.allclose(out.size_logp[row], mixed(torch.softmax(H.size_out(trunk), 0)), atol=1e-12, rtol=0), (masked, row)
            assert torch.allclose(out.kill_rate[row], torch.exp(H.kill_out(trunk)).reshape(4, 2), rtol=1e-12, atol=0), (masked, row)
            slots = F.gelu(H.gold_in(trunk)[None] + H.gold_slot.weight)                                                 # (48, d)
            gold = torch.softmax(H.gold_out(slots).reshape(4, 2, 6, 31), -1)
            assert torch.allclose(out.gold_logp[row], torch.log((1 - MIX) * gold + MIX / 31), atol=1e-12, rtol=0), (masked, row)


def test_the_objective_and_conversion_means_are_the_equation():
    """(EH-7) and (EH-8) against a plain computation from the weights (float64, 1e-12), in the masked and in the open reading; the conversions are the
    window of the row times the exponential."""
    H, enc, c = tiny_model(15).double().heads, fabricated(2, dtype=torch.float64, seed=9), conditions(2, seed=4, dtype=torch.float64)
    for masked in (True, False):
        q, m, z = (c['q'], c['m'], c['z']) if masked else (None, None, None)
        obj, conv = H.m3_obj(enc, q, m, z, c['s'], c['K']), H.m4(enc, q, m, z, c['K'], c['O'], c['window'])
        for row in range(2):
            h, players = enc.h_pool[row], enc.h_tok[row, :10]
            if masked:
                emb_q, emb_z, emb_m = H.bin_emb.weight[int(q[row])], H.zone_emb.weight[int(z[row]) + 1], plain_mark_embedding(H, players, int(m[row]))
            else:
                emb_q, emb_z, emb_m = H.bin_emb.weight[18], H.zone_emb.weight[11], H.mark.kind.weight[2]
            emb_s, log_k, log_o = H.size_emb.weight[int(c['s'][row])], torch.log(1 + c['K'][row]), torch.log(1 + c['O'][row])
            net = H.obj_net                                                                                             # (EH-7)
            eta = net.out(F.gelu(net.maps[0](h) + net.maps[1](emb_q) + net.maps[2](emb_m) + net.maps[3](emb_z) + net.maps[4](emb_s) + net.maps[5](log_k)
                                 + net.bias))
            assert torch.allclose(obj[row], torch.exp(eta), rtol=1e-12, atol=0), (masked, row)
            net = H.conv_net                                                                                            # (EH-8)
            eta = net.out(F.gelu(net.maps[0](h) + net.maps[1](emb_q) + net.maps[2](emb_m) + net.maps[3](emb_z) + net.maps[4](log_k) + net.maps[5](log_o)
                                 + net.bias))
            assert torch.allclose(conv[row], torch.exp(eta) * c['window'][row], rtol=1e-12, atol=0), (masked, row)


def test_the_open_reading_uses_its_own_values_of_none_and_the_weights_of_the_masked_one():
    """Design 4: one network for the two readings. The open reading (q, m, z all None) reads the "none" rows of the bin, mark-kind and zone tables;
    changing those changes the open outputs and not the masked ones, and changing a weight that both read changes both."""
    model, enc, c = tiny_model(16), fabricated(3, seed=1), conditions(3, seed=5)
    H = model.heads
    with torch.no_grad():
        masked, opened = H.m3(enc, c['q'], c['m'], c['z']), H.m3(enc, None, None, None)
        assert all(a.shape == b.shape for a, b in zip(masked, opened))
        H.bin_emb.weight[18] += 0.5
        H.zone_emb.weight[11] += 0.5
        H.mark.kind.weight[2] += 0.5
        masked2, opened2 = H.m3(enc, c['q'], c['m'], c['z']), H.m3(enc, None, None, None)
        assert all(torch.equal(a, b) for a, b in zip(masked, masked2)) and all(not torch.equal(a, b) for a, b in zip(opened, opened2))
        H.m3_net.bias += 0.1
        masked3, opened3 = H.m3(enc, c['q'], c['m'], c['z']), H.m3(enc, None, None, None)
        assert all(not torch.equal(a, b) for a, b in zip(masked2, masked3)) and all(not torch.equal(a, b) for a, b in zip(opened2, opened3))


def test_conditions_are_checked():
    """A condition of the wrong shape, type or range, a partial masked reading, an exposure that is negative, NaN or of another shape and a ``pos`` that
    names no view are refused with a ValueError that names them, so that a mistake does not turn into an index error or a silent NaN."""
    model, enc, c = tiny_model(17), fabricated(3), conditions(3, seed=6)
    H, q, m, z = model.heads, c['q'], c['m'], c['z']
    edge = H.m3_obj(enc, torch.tensor([0, 17, 5]), torch.tensor([0, 51, 49]), torch.tensor([-1, 9, 0]), torch.tensor([0, 3, 1]), c['K'])
    assert edge.shape == (3, 18) and torch.isfinite(edge).all()                                                 # the extreme values are accepted
    bad = lambda t, i, v: torch.cat([t[:i], torch.tensor([v], dtype=t.dtype), t[i + 1:]])                      # noqa: E731  (a copy with one value changed)
    for name, call in (('q', lambda v: H.m3(enc, v, m, z)), ('m', lambda v: H.m3(enc, q, v, z)), ('z', lambda v: H.m3(enc, q, m, v)),
                       ('s', lambda v: H.m3_obj(enc, q, m, z, v, c['K']))):
        good = {'q': q, 'm': m, 'z': z, 's': c['s']}[name]
        above, below = {'q': (18, -1), 'm': (52, -1), 'z': (10, -2), 's': (4, -1)}[name]                    # the first value out of range, on each side
        for v in (bad(good, 1, 99), bad(good, 1, above), bad(good, 1, below)):
            with pytest.raises(ValueError, match=f'{name} must lie'):
                call(v)
        for wrong in (good.float(), good.bool(), good.to(torch.uint8), good[:2]):                               # not integers, masks (a uint8 indexes as one), not one per row
            with pytest.raises(ValueError, match=f'{name} must be an integer tensor'):
                call(wrong)
        for narrow in (torch.int8, torch.int16, torch.int32):                                                   # other integer types are accepted and cast
            got, want = call(good.to(narrow)), call(good)
            assert all(torch.equal(x, y) for x, y in (zip(got, want) if isinstance(got, tuple) else [(got, want)])), (name, narrow)
    for bad_q, bad_m, bad_z in ((None, m, z), (q, None, z), (q, m, None), (q, None, None), (None, None, z)):
        with pytest.raises(ValueError, match='given together'):
            H.m3(enc, bad_q, bad_m, bad_z)
    k_nan = c['K'].clone()
    k_nan[0, 1] = float('nan')
    for call, want in ((lambda v: H.m3_obj(enc, q, m, z, c['s'], v), 'K'), (lambda v: H.m4(enc, q, m, z, v, c['O'], c['window']), 'K'),
                       (lambda v: H.m4(enc, q, m, z, c['K'], v, c['window']), 'O')):
        shape = (3, 4) if want == 'K' else (3, 18)
        for v in (-torch.ones(shape), torch.full(shape, float('inf')), torch.ones(3, shape[1] + 1), torch.ones(shape[1])):
            with pytest.raises(ValueError, match=f'{want} must be'):
                call(v)
    with pytest.raises(ValueError, match='K must be finite'):
        H.m3_obj(enc, q, m, z, c['s'], k_nan)
    for e in (-torch.ones(3, 18), torch.full((3, 18), float('nan')), torch.ones(3, 17), torch.ones(2, 18)):
        with pytest.raises(ValueError, match='exposure'):
            H.m2(enc, e)
    with pytest.raises(ValueError, match='exposure'):
        H.m4(enc, q, m, z, c['K'], c['O'], torch.tensor([1.0, -1.0, 1.0]))
    for pos in (torch.tensor([0, 3]), torch.tensor([-1, 0]), torch.tensor([0.0, 1.0]), torch.tensor([True, False]), torch.tensor([0, 1], dtype=torch.uint8),
                torch.zeros(2, 2, dtype=torch.long)):
        with pytest.raises(ValueError, match='pos must'):
            H.m2(enc, torch.ones(2, 18), pos)


def test_pos_selects_the_views_of_the_rows():
    """``pos`` lists the view of every row (a view may serve several rows, as several onsets of one frame row do): the answer is that of the encodings
    taken row by row, bit for bit, for all four methods."""
    model, batch = tiny_model(18), encoder_batch(views=4, events=9, edges=7, seed=3)
    H = model.heads
    pos = torch.tensor([3, 0, 0, 2, 1, 3])
    c = conditions(6, seed=7)
    with torch.no_grad():
        enc = model.encode(batch)
        rows = encoder.Encoded(enc.h_tok[pos], enc.h_pool[pos], enc.mask[pos])
        for a, b in zip(H.m2(enc, c['exposure'], pos), H.m2(rows, c['exposure'])):
            assert torch.equal(a, b)
        for a, b in zip(H.m3(enc, c['q'], c['m'], c['z'], pos), H.m3(rows, c['q'], c['m'], c['z'])):
            assert torch.equal(a, b)
        assert torch.equal(H.m3_obj(enc, c['q'], c['m'], c['z'], c['s'], c['K'], pos), H.m3_obj(rows, c['q'], c['m'], c['z'], c['s'], c['K']))
        assert torch.equal(H.m4(enc, None, None, None, c['K'], c['O'], c['window'], pos), H.m4(rows, None, None, None, c['K'], c['O'], c['window']))


def test_the_heads_do_not_depend_on_the_memory_layout_of_the_conditions():
    """The output depends on the values of K and O and not on their strides (a matrix product and a vectorised log round by layout; the swapped
    conditions of the frame average come from other tensors than the caller's): a transposed copy with the same values gives the same bits."""
    H = tiny_model(19).heads
    for n in range(2, 13):                         # the product of a matrix with a strided operand is rounded differently for some sizes only
        enc, c = fabricated(n, seed=2), conditions(n, seed=8 + n)
        k_strided = c['K'].T.contiguous().T                                                  # the same values, another layout
        o_strided = c['O'].T.contiguous().T
        assert not k_strided.is_contiguous() and not o_strided.is_contiguous() and torch.equal(k_strided, c['K'])
        args = (enc, c['q'], c['m'], c['z'])
        assert torch.equal(H.m4(*args, c['K'], c['O'], c['window']), H.m4(*args, k_strided, o_strided, c['window'])), n
        assert torch.equal(H.m3_obj(*args, c['s'], c['K']), H.m3_obj(*args, c['s'], k_strided)), n
        # the layout that a numpy reference leaves (``symmetry.swap_O`` indexes the last axis): the case in which the frame average first met the problem
        swapped = torch.as_tensor(symmetry.swap_O(c['O'].numpy()))
        assert not swapped.is_contiguous()
        assert torch.equal(H.m4(*args, c['K'], swapped, c['window']), H.m4(*args, c['K'], swapped.contiguous(), c['window'])), n


def test_every_head_parameter_gets_a_finite_gradient():
    """No part of the heads is cut off from the outputs: every parameter has a finite gradient after one backward pass that uses both readings of every
    method (the "none" rows of the tables only count in the open reading), and the tables and layers that carry a condition get a gradient that is not 0."""
    model, batch, c = tiny_model(20), encoder_batch(views=5, events=9, edges=7, seed=4), conditions(5, seed=10)
    model.train()
    H = model.heads
    enc = model.encode(batch)
    outs = [*H.m2(enc, c['exposure']), *H.m3(enc, c['q'], c['m'], c['z']), *H.m3(enc, None, None, None),
            H.m3_obj(enc, c['q'], c['m'], c['z'], c['s'], c['K']), H.m3_obj(enc, None, None, None, c['s'], c['K']),
            H.m4(enc, c['q'], c['m'], c['z'], c['K'], c['O'], c['window']), H.m4(enc, None, None, None, c['K'], c['O'], c['window'])]
    sum(t.sum() for t in outs).backward()
    named = dict(H.named_parameters())
    assert [name for name, p in named.items() if p.grad is None or not torch.isfinite(p.grad).all()] == []
    for name in ('bin_emb.weight', 'zone_emb.weight', 'size_emb.weight', 'mark.kind.weight', 'mark.killer.weight', 'mark.victim.weight', 'mark.team.weight',
                 'gold_slot.weight', 'gold_in.weight', 'rate_net.maps.0.weight', 'rate_net.maps.1.weight', 'mark_net.maps.0.weight', 'zone_net.maps.2.weight',
                 'm3_net.maps.3.weight', 'obj_net.maps.5.weight', 'conv_net.maps.4.weight', 'conv_net.maps.5.weight', 'kill_out.weight', 'size_out.bias'):
        assert named[name].grad.abs().sum() > 0, name


@pytest.mark.parametrize('size', ['S', 'M'])
def test_the_two_default_sizes_run_on_views_of_the_real_shape(size):
    """The sizes of plan Q6 build and run together with the heads on views with the 128 event tokens and 128 edges of the real store (two views, CPU):
    the output shapes, finite values, and the frame average of the same model."""
    cfg = encoder.SIZES[size]
    model = heads.EPT(cfg, toy_static()).eval()
    batch = encoder_batch(views=2, events=128, edges=128, seed=22)
    c = conditions(2, seed=11)
    with torch.no_grad():
        fa = heads.FrameAveraged(model)
        enc = fa.encode(batch)
        out = [*fa.m2(enc, c['exposure']), *fa.m3(enc, c['q'], c['m'], c['z']), fa.m3_obj(enc, c['q'], c['m'], c['z'], c['s'], c['K']),
               fa.m4(enc, c['q'], c['m'], c['z'], c['K'], c['O'], c['window'])]
    shapes = [(2, 18), (2, 18, 52), (2, 18, 52, 10), (2, 4), (2, 4, 2), (2, 4, 2, 6, 31), (2, 18), (2, 18)]
    assert [tuple(t.shape) for t in out] == shapes and all(torch.isfinite(t).all() for t in out)
    n_heads = sum(p.numel() for p in model.heads.parameters())
    assert 5_000 < n_heads < 1_000_000 and sum(p.numel() for p in model.parameters()) > n_heads


def test_the_heads_module_reads_no_key_no_holdout_and_no_cohort_module():
    from c_ept.tests.conftest import all_imports
    source = Path(heads.__file__)
    text = source.read_text(encoding='utf-8')
    assert 'matches/key' not in text and 'allow_holdout' not in text and 'split_module' not in text and 'bucket(' not in text
    assert not all_imports(source) & {'c_ept.cohort', 'c_ept.reader', 'c_ept.run_s1', 'c_ept.run_s2', 'c_ept.run_s3', 'pickle', 'lightgbm'}
    assert 'cuda' not in text.replace('CUDA', '').lower()                                              # nothing here puts a tensor on a GPU


# =====================================================================================================================================
# determinism
# =====================================================================================================================================
def fingerprint(threads: int = 8, seed: int = 20261022) -> str:
    """SHA-256 of the outputs and of the gradients of a small model with the frame average (dropout on in the encoder, so the random draws count),
    built, run and differentiated under ``torch.use_deterministic_algorithms(True)`` with ``threads`` threads from one seed."""
    before = torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)
    try:
        torch.manual_seed(seed)
        model = heads.EPT(encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.1), toy_static())
        model.train()
        fa = heads.FrameAveraged(model)
        c = conditions(4, seed=3)
        enc = fa.encode(encoder_batch(views=4, events=11, edges=9, seed=5))
        outs = [*fa.m2(enc, c['exposure']), *fa.m3(enc, c['q'], c['m'], c['z']), fa.m3_obj(enc, c['q'], c['m'], c['z'], c['s'], c['K']),
                fa.m4(enc, c['q'], c['m'], c['z'], c['K'], c['O'], c['window'])]
        sum(t.sum() for t in outs).backward()
        digest = hashlib.sha256()
        for t in (*outs, *(p.grad for p in model.parameters())):
            digest.update(t.detach().numpy().tobytes() if t is not None else b'none')
        return digest.hexdigest()
    finally:
        torch.use_deterministic_algorithms(before[0])
        torch.set_num_threads(before[1])


def test_cpu_forward_is_deterministic():
    """Review focus 5: one seed and one thread count give the same bits, run after run (forward and backward, through the frame average, dropout on),
    and in a second process too (nothing may depend on the address of an object, the order of a set or the state of the process)."""
    first = fingerprint()
    assert fingerprint() == first
    assert fingerprint(seed=1) != first                                              # the fingerprint can differ
    root = Path(heads.__file__).resolve().parents[2]
    code = 'from c_ept.tests.test_ept_heads import fingerprint; print(fingerprint())'
    run = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True, timeout=300, check=False)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip().splitlines()[-1] == first


def test_the_heads_take_only_what_the_plan_gives_them():
    """At the level of the heads: the masked M3 head gets no O (plan P4: O is the outcome given K and the size), the masked M4 head gets (q, m, z, K, O)
    and no size (design 3.3, leakage test (a)), and M2 gets no condition at all."""
    names = lambda f: list(inspect.signature(f).parameters)                                                      # noqa: E731
    for cls in (heads.EPTHeads, heads.FrameAveraged):
        assert names(cls.m2) == ['self', 'enc', 'exposure', 'pos']
        assert names(cls.m3) == ['self', 'enc', 'q', 'm', 'z', 'pos']
        assert names(cls.m3_obj) == ['self', 'enc', 'q', 'm', 'z', 's', 'K', 'pos']
        assert names(cls.m4) == ['self', 'enc', 'q', 'm', 'z', 'K', 'O', 'exposure', 'pos']


def test_the_output_table_names_every_head_of_the_rows():
    """``HEAD_OUTPUT`` is how a training loop finds the output of each head of ``head_rows``: the method and the field."""
    assert set(heads.HEAD_OUTPUT) == set(head_rows.HEAD_NAMES)
    for head, (method, field) in heads.HEAD_OUTPUT.items():
        assert callable(getattr(heads.EPTHeads, method)), head
        out_type = {'m2': heads.M2Out, 'm3': heads.M3Out}.get(method)
        assert (field is None) == (out_type is None) and (field is None or field in out_type._fields), head
