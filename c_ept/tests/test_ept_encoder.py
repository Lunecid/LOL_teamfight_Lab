"""Tests of the relational encoder of the EPT (stage S4a Task 2): ``ept.encoder``.

No run folder, raw file or catalog is read, and nothing is put on a GPU. Most tests use ``ept_toys.encoder_batch`` (random view tensors of
any size, no store); two use the tensors of the 7 / 5 toy world (7 event tokens and 5 edges per view, not the 128 and 128 of the real store:
the encoder reads the sizes from the batch). The encoder of the tests is tiny (``d`` = 16, two layers, two heads).

    shapes and padding   the output shapes, the mask, a padded event slot that does not matter (any content), is no key (the same states
                         with more or fewer slots) and is zero in the output, a view that is encoded the same alone and in a mixed batch,
                         finite outputs for missing positions
    equivariance         the players of one team change places and the outputs change places with them (float64, 1e-9)
    biases               the geodesic and the distance bias enter the attention scores (a perturbation reaches exactly the scores it
                         should, a zeroed bias gives another output), the bias is exactly swap-covariant (the physical places, the
                         relations and the geodesics follow the players and the landmarks), the type table
    co-participation     the decay scales are positive and learned, the bias of an edge list against a hand case
    terrain field        bilinear sampling (a bilinear function is reproduced exactly, the border repeats, a corner is the mean of four
                         cells), zero at the start, the static channels, no gradient in a wall, the smoothness penalty, the physical place
                         of a token after a swap
    determinism          forward and backward bit-identical for one seed and thread count, also in a second process
"""
import ast
import copy
import gc
import hashlib
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from c_ept import config as C
from c_ept import terrain, tokens
from c_ept.ept import data, encoder
from c_ept.tests.ept_toys import build_ept_world, encoder_batch, toy_static

TINY = dict(d=16, layers=2, heads=2, dropout=0.0)
N_PLAYERS, N_LANDMARKS = tokens.N_PLAYERS, tokens.N_LANDMARKS
FIXED = N_PLAYERS + N_LANDMARKS                      # the tokens before the events; the summary token is the last one


def tiny(seed: int = 0, static=None, **changes) -> encoder.EPTEncoder:
    """A tiny encoder in evaluation mode, built after ``torch.manual_seed(seed)`` (the same weights for the same seed). ``static`` is the
    terrain that the encoder requires: the toy terrain (``ept_toys.toy_static``) unless a test gives its own or ``False`` (no static channels)."""
    torch.manual_seed(seed)
    return encoder.EPTEncoder(encoder.EncoderConfig(**{**TINY, **changes}), toy_static() if static is None else static).eval()


def as_double(batch: dict) -> dict:
    return {k: v.double() if v.is_floating_point() else v for k, v in batch.items()}


def pick(batch: dict, rows) -> dict:
    """The views ``rows`` of a batch (a list of view numbers) as a batch of their own."""
    return {k: v[rows] for k, v in batch.items()}


def permute_players(batch: dict, perm) -> dict:
    """The batch in which the players are renumbered: new row r holds old row ``perm[r]``. Everything that names a player follows: the
    four player tensors, both axes of ``geo_pp`` (and its mask), the rows of ``geo_pl``, the participants of the events and the two ends
    of the edges (old player s becomes the r with ``perm[r] == s``; the padding -1 stays)."""
    perm = torch.as_tensor(perm)
    inverse = torch.argsort(perm)
    out = dict(batch)
    for key in ('players', 'players_nan', 'players_zone', 'players_xy', 'geo_pl', 'geo_pl_nan'):
        out[key] = batch[key][:, perm]
    for key in ('geo_pp', 'geo_pp_nan'):
        out[key] = batch[key][:, perm][:, :, perm]
    out['ev_actors'] = batch['ev_actors'][..., perm]
    for key in ('ed_i', 'ed_j'):
        out[key] = torch.where(batch[key] >= 0, inverse[batch[key].clamp(min=0)], batch[key])
    return out


# =====================================================================================================================================
# the toy world and the layout of the synthetic batch
# =====================================================================================================================================
@pytest.fixture(scope='module')
def small_world(tmp_path_factory):
    """The EPT toy world with 7 event tokens and 5 edges per view (the real store has 128 and 128)."""
    folder = tmp_path_factory.mktemp('encoder_world')
    world = build_ept_world(folder, n_events=7, n_edges=5)
    yield world
    del world
    gc.collect()                                   # Windows does not delete a folder with open memory maps
    shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture(scope='module')
def small_batch(small_world):
    """Every view of the 7 / 5 world as view tensors (213 views: all three clocks, missing positions, views without an event)."""
    stats = data.fit_stats(small_world.s1, small_world.s2)
    return data.ViewTensors(small_world.s2, np.arange(len(small_world.s2.array('views/src'))), stats)


def test_the_synthetic_batch_has_the_layout_of_the_view_tensors(small_batch):
    """The encoder tests run on ``encoder_batch``; it must have the keys, the trailing shapes and the dtypes of the real tensors."""
    fake = encoder_batch(views=3, events=7, edges=5, seed=0)
    assert set(fake) == set(small_batch) == set(data.VIEW_KEYS)
    for key, t in small_batch.items():
        assert fake[key].dtype == t.dtype and fake[key].shape[1:] == t.shape[1:], key
    assert encoder_batch(views=2, events=4, edges=3, dtype=torch.float64)['players'].dtype == torch.float64
    same = encoder_batch(views=3, events=7, edges=5, seed=0)
    assert all(torch.equal(fake[k], same[k]) for k in fake)                                  # a seed gives one batch


# =====================================================================================================================================
# shapes and the padding mask
# =====================================================================================================================================
def test_shapes_and_padding_mask():
    """Views with different numbers of event tokens (0, 3, 9 and a random one) are mixed in one batch: the content of the padded slots is not read
    (it may be anything, also out of range), they are no keys of the attention (9 slots of which at most 4 are real give, at the tokens that exist
    in both, the states of only 4 slots) and their output is exactly zero."""
    batch = encoder_batch(views=5, events=9, edges=7, seed=3)
    batch['ev_valid'][2] = torch.arange(9) < 3                                       # view 2: three event tokens
    assert batch['ev_valid'].sum(1)[:3].tolist() == [0, 9, 3]
    enc = tiny()
    out = enc(batch)
    t = FIXED + 9 + 1
    assert out.h_tok.shape == (5, t, 16) and out.h_pool.shape == (5, 16) and out.mask.shape == (5, t) and out.mask.dtype == torch.bool
    assert out.mask[:, :FIXED].all() and out.mask[:, -1].all() and torch.equal(out.mask[:, FIXED:FIXED + 9], batch['ev_valid'])
    assert torch.equal(out.h_pool, out.h_tok[:, -1])                                  # the pooled state is the summary token's
    assert torch.isfinite(out.h_tok).all() and (out.h_tok[~out.mask] == 0).all()    # padded tokens: exactly zero
    assert (out.h_tok[out.mask].abs() > 0).any()
    # junk in the padded slots (values outside every range) changes nothing at the tokens that count
    junk = {k: v.clone() for k, v in batch.items()}
    pad = ~batch['ev_valid']
    for key, value in (('ev_type', 99), ('ev_team', -7), ('ev_num', 1e6), ('ev_xy', -1e6), ('ev_nan', 5.0), ('ev_actors', 3.0)):
        junk[key][pad] = value
    again = enc(junk)
    assert torch.equal(again.h_tok, out.h_tok) and torch.equal(again.h_pool, out.h_pool)
    # the junk test cannot see a padded slot that is read as a key (its content is the same fill whatever the junk): the amount of padding can.
    # A padded slot that took part in the attention would change the states of the real tokens when slots are added or taken away.
    wide = encoder_batch(views=4, events=9, edges=5, seed=17)
    wide['ev_valid'][:, 4:] = False
    narrow = {k: (v[:, :4] if k.startswith('ev_') else v) for k, v in wide.items()}
    assert wide['ev_valid'].sum() > 0 and wide['ev_valid'].sum() == narrow['ev_valid'].sum()
    a, b = enc(wide), enc(narrow)
    assert torch.allclose(a.h_tok[:, :FIXED + 4], b.h_tok[:, :FIXED + 4], atol=1e-5)
    assert torch.allclose(a.h_tok[:, -1], b.h_tok[:, -1], atol=1e-5) and torch.allclose(a.h_pool, b.h_pool, atol=1e-5)
    # a view alone gives what it gives inside the mixed batch (the views do not talk to each other)
    for v in range(5):
        alone = enc(pick(batch, [v]))
        assert torch.allclose(alone.h_tok[0], out.h_tok[v], atol=1e-5) and torch.allclose(alone.h_pool[0], out.h_pool[v], atol=1e-5), v
    # the size of the event list is read from the batch: no event token at all, and the 128 of the real store
    none = enc(encoder_batch(views=2, events=0, edges=0, seed=1))
    assert none.h_tok.shape == (2, FIXED + 1, 16) and none.mask.all() and torch.isfinite(none.h_tok).all()
    full = enc(encoder_batch(views=2, events=128, edges=128, seed=2))
    assert full.h_tok.shape == (2, FIXED + 128 + 1, 16) and torch.isfinite(full.h_tok).all()


def test_the_encoder_reads_the_view_tensors_of_a_store_with_other_sizes_than_the_real_one(small_batch):
    """Real ``ViewTensors`` (all three clocks, players without a position, views without an event or an edge, 7 events and 5 edges)."""
    out = tiny()(small_batch)
    n = len(small_batch['view'])
    assert out.h_tok.shape == (n, FIXED + 7 + 1, 16) and torch.isfinite(out.h_tok).all()
    assert torch.equal(out.mask[:, FIXED:FIXED + 7], small_batch['ev_valid'])
    assert small_batch['players_nan'][..., data.PLAYER_NAN_COLUMNS.index('x')].any() and (~small_batch['ev_valid']).any()


def test_a_missing_batch_key_is_named():
    batch = encoder_batch(views=2, events=3, edges=2)
    del batch['geo_pl'], batch['ed_kind']
    with pytest.raises(ValueError, match="geo_pl.*ed_kind|ed_kind.*geo_pl"):
        tiny()(batch)


def test_the_token_layout_is_the_order_of_the_tokens_and_row_blocks_are_ignored(small_batch):
    """The heads read the player tokens by number: players, landmarks, events, summary, in that order. The keys of the row blocks that a batch of
    ``data.Batches`` carries next to the view tensors are not read."""
    layout = encoder.token_layout(7)
    assert (layout.players, layout.landmarks, layout.events, layout.summary) == (slice(0, 10), slice(10, 44), slice(44, 51), 51)
    batch = {**pick(small_batch, [1, 5, 9]), 'M2Mark/masked/y': torch.zeros(3, dtype=torch.long), 'M4Conv/open/K': torch.zeros(3, 4)}
    out = tiny()(batch)
    assert out.h_tok.shape[1] == layout.summary + 1 and torch.equal(out.h_pool, out.h_tok[:, layout.summary])
    assert torch.equal(out.mask[:, layout.events], batch['ev_valid']) and out.mask[:, layout.players].all() and out.mask[:, layout.landmarks].all()
    assert torch.equal(out.h_tok, tiny()(pick(small_batch, [1, 5, 9])).h_tok)


def test_missing_positions_and_masked_cells_give_finite_outputs_and_gradients():
    """Review focus 3: no player has a position, every maskable cell is masked, no event has a position: the encoder stays finite."""
    batch = encoder_batch(views=3, events=6, edges=4, seed=4)
    x, y = (data.PLAYER_NAN_COLUMNS.index(c) for c in ('x', 'y'))
    batch['players_nan'][:] = 1.0
    batch['players_xy'][:] = C.W / 2
    batch['players_zone'][:] = -1
    batch['geo_pp_nan'][:] = 1.0
    batch['geo_pl_nan'][:] = 1.0
    batch['geo_pp'][:] = 0.0
    batch['geo_pl'][:] = 0.0
    batch['landmarks_nan'][:] = 1.0
    batch['summary_nan'][:] = 1.0
    batch['ev_nan'][batch['ev_valid']] = 1.0
    batch['ev_xy'][:] = C.W / 2
    assert batch['players_nan'][..., x].all() and batch['players_nan'][..., y].all()
    enc = tiny()
    out = enc(batch)
    assert torch.isfinite(out.h_tok).all()
    out.h_pool.square().sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in enc.parameters())


def test_every_parameter_gets_a_finite_gradient():
    """No part of the encoder is cut off from the output: every parameter has a gradient after one backward pass, and all are finite."""
    enc = tiny(field_channels=3)
    enc.train()
    out = enc(encoder_batch(views=5, events=9, edges=7, seed=21))
    (out.h_tok.square().sum() + out.h_pool.sum()).backward()
    names = [name for name, p in enc.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert names == []
    for name in ('bias.type_bias', 'bias.co.w', 'bias.co.w_back', 'bias.co.log_kappa', 'field.learned', 'field_proj.weight'):
        assert dict(enc.named_parameters())[name].grad.abs().sum() > 0, name
    # the rows of w_back of the undirected edge kinds are never read: they get a gradient of zero, the other rows one that is not zero
    undirected = list(encoder.UNDIRECTED_EDGE_KINDS)
    assert (enc.bias.co.w_back.grad[:, undirected] == 0).all() and (enc.bias.co.w_back.grad[:, [0, 1, 3]].abs().sum(dim=(0, 2)) > 0).all()
    # without static channels the projection of the field has no gradient while the learned field is zero (its input is zero); the field
    # itself gets one, then the projection does too
    bare = tiny(field_channels=3, static=False)
    bare.train()
    (bare(encoder_batch(views=5, events=9, edges=7, seed=21)).h_tok.square().sum()).backward()
    assert bare.field_proj.weight.grad.abs().sum() == 0 and bare.field.learned.grad.abs().sum() > 0


@pytest.mark.parametrize('size', ['S', 'M'])
def test_the_two_default_sizes_encode_a_view_of_the_real_shape(size):
    """The sizes of plan Q6 build and run on views with the 128 event tokens and 128 edges of the real store (two views; CPU)."""
    enc = encoder.EPTEncoder(encoder.SIZES[size], toy_static()).eval()
    out = enc(encoder_batch(views=2, events=128, edges=128, seed=22))
    cfg = encoder.SIZES[size]
    assert out.h_tok.shape == (2, 173, cfg.d) and torch.isfinite(out.h_tok).all() and len(enc.blocks) == cfg.layers
    assert enc.bias.type_bias.shape == (cfg.heads, 6) and enc.field.learned.shape == (75, 75, 8) and enc.bias.co.log_kappa.shape == (3,)


def test_view_context_enters_and_the_view_number_does_not():
    """Plan Q5: the kind of the view, its frame age and the side marker are inputs; the number of the view is no input."""
    batch = encoder_batch(views=2, events=5, edges=3, seed=6)
    enc = tiny()
    base = enc(batch)
    for key, new in (('kind', (batch['kind'] + 1) % 3), ('age', batch['age'] + 1.0), ('side', 1.0 - batch['side'])):
        other = {**batch, key: new}
        assert not torch.allclose(enc(other).h_pool, base.h_pool, atol=1e-6), key
    other = {**batch, 'view': batch['view'] + 1000}
    assert torch.equal(enc(other).h_tok, base.h_tok)


# =====================================================================================================================================
# permutation equivariance of the players of one team
# =====================================================================================================================================
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_players_within_a_team_are_permutation_equivariant(seed):
    """The slot of a player is no role (design 5.2): inside a team the rows may be in any order. New row r holds old row perm[r]; then the
    outputs of the player tokens are the old outputs in that order and everything else (landmarks, events, the summary, the pooled state)
    is unchanged. Float64, so a wrong cell cannot hide behind rounding (1e-9)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.cat([torch.randperm(5, generator=g), 5 + torch.randperm(5, generator=g)])
    assert not torch.equal(perm, torch.arange(10))
    batch = as_double(encoder_batch(views=4, events=11, edges=9, seed=10 + seed))
    enc = tiny(seed).double()
    reordered = permute_players(batch, perm)
    # S2 may present the same undirected pairs in either orientation and any list order.
    undirected = batch['ed_valid'] & ((batch['ed_kind'] == tokens.EDGE_SIDE) | (batch['ed_kind'] == tokens.EDGE_OBJECTIVE))
    assert undirected.any(), 'fixture must exercise an undirected edge'
    for kind in (tokens.EDGE_SIDE, tokens.EDGE_OBJECTIVE):
        assert (undirected & (batch['ed_kind'] == kind)).any(), kind
    reordered['ed_i'], reordered['ed_j'] = (
        torch.where(undirected, reordered['ed_j'], reordered['ed_i']),
        torch.where(undirected, reordered['ed_i'], reordered['ed_j']))
    edge_order = torch.randperm(batch['ed_i'].shape[1], generator=g)
    assert not torch.equal(edge_order, torch.arange(len(edge_order)))
    for key in ('ed_i', 'ed_j', 'ed_kind', 'ed_dt', 'ed_valid'):
        reordered[key] = reordered[key][:, edge_order]
    old, new = enc(batch), enc(reordered)
    assert torch.allclose(new.h_tok[:, :N_PLAYERS], old.h_tok[:, :N_PLAYERS][:, perm], atol=1e-9, rtol=0)
    assert torch.allclose(new.h_tok[:, N_PLAYERS:], old.h_tok[:, N_PLAYERS:], atol=1e-9, rtol=0)
    assert torch.allclose(new.h_pool, old.h_pool, atol=1e-9, rtol=0)
    assert (old.h_tok[:, :N_PLAYERS] - old.h_tok[:, :N_PLAYERS][:, perm]).abs().max() > 1e-3       # the test can fail: the order matters to a row


# =====================================================================================================================================
# the attention bias: distance, geodesic, pair type
# =====================================================================================================================================
def changed(a: torch.Tensor, b: torch.Tensor) -> set:
    """The index tuples (view, token i, token j) at which the biases ``a`` and ``b`` differ in at least one head."""
    return {tuple(i) for i in (a != b).any(dim=1).nonzero().tolist()}


def token_permutation(events: int) -> torch.Tensor:
    """The renumbering of the tokens by ``data.swap``: new token r is old token ``pi[r]`` (players and landmarks change places, the events
    and the summary token keep theirs)."""
    pi = torch.arange(FIXED + events + 1)
    pi[:N_PLAYERS] = torch.from_numpy(np.array(data.PLAYER_PERM))
    pi[N_PLAYERS:FIXED] = N_PLAYERS + torch.from_numpy(np.array(data.LANDMARK_PERM))
    return pi


def test_geodesic_and_distance_biases_enter_attention():
    """The bias that is added to the attention scores depends on the geodesic and on the distance, and on nothing else of those inputs:
    one geodesic cell reaches one score (two for a player-landmark pair, which both tokens read), one moved player reaches the scores of
    its row and column. A copy with the distance or the geodesic parameters at zero, or with the whole bias at zero, encodes differently."""
    batch = encoder_batch(views=3, events=6, edges=5, seed=1)
    enc = tiny()
    base = enc.attention_bias(batch)
    n = FIXED + 6 + 1
    assert base.shape == (3, 2, n, n) and torch.isfinite(base).all()

    def moved(key, index, amount):
        other = {k: v.clone() for k, v in batch.items()}
        other[key][index] += amount
        return enc.attention_bias(other)

    assert changed(base, moved('geo_pp', (1, 2, 7), 0.7)) == {(1, 2, 7)}                          # one player-player geodesic cell
    assert changed(base, moved('geo_pl', (0, 4, 9), 0.7)) == {(0, 4, N_PLAYERS + 9), (0, N_PLAYERS + 9, 4)}      # and a player-landmark one
    row_and_column = {(2, 6, j) for j in range(n)} | {(2, j, 6) for j in range(n)}
    after = changed(base, moved('players_xy', (2, 6, 0), 500.0))                                 # a moved player: its row and column
    assert after and after <= row_and_column
    token = FIXED + int(((batch['ev_valid'][1]) & (batch['ev_nan'][1, :, 0] == 0)).nonzero()[0])         # a moved event with a position
    event_pairs = {(1, i, token) for i in range(N_PLAYERS)} | {(1, token, i) for i in range(N_PLAYERS)}
    assert changed(base, moved('ev_xy', (1, token - FIXED, 1), -400.0)) == event_pairs          # reaches the scores of the players
    # the switched-off copies: the bias is exactly zero, and the whole encoder gives another answer
    reference = enc(batch).h_tok

    def zeroed(select):
        other = copy.deepcopy(enc)
        with torch.no_grad():
            for p in select(other):
                p.zero_()
        return other

    off = zeroed(lambda m: m.bias.parameters())
    assert torch.equal(off.attention_bias(batch), torch.zeros_like(base))
    for select in (lambda m: m.bias.dist.parameters(), lambda m: m.bias.geo.parameters(), lambda m: m.bias.parameters()):
        assert not torch.allclose(zeroed(select)(batch).h_tok, reference, atol=1e-6)
    # each term alone is a term: with only the distance alive the geodesic does not matter, and the other way round
    dist_only = zeroed(lambda m: [*m.bias.geo.parameters(), *m.bias.co.parameters(), m.bias.type_bias])
    geo_only = zeroed(lambda m: [*m.bias.dist.parameters(), *m.bias.co.parameters(), m.bias.type_bias])
    far, geodesic = {**batch, 'players_xy': batch['players_xy'] + 3.0}, {**batch, 'geo_pp': batch['geo_pp'] + 1.0, 'geo_pl': batch['geo_pl'] + 1.0}
    assert changed(dist_only.attention_bias(batch), dist_only.attention_bias(far)) and not changed(
        dist_only.attention_bias(batch), dist_only.attention_bias(geodesic))
    assert changed(geo_only.attention_bias(batch), geo_only.attention_bias(geodesic)) and not changed(
        geo_only.attention_bias(batch), geo_only.attention_bias(far))


def test_the_distance_term_depends_on_the_euclidean_distance_only():
    """A shift of two players together and a turn of their separation by 90 degrees leave the bias between them as it was (the same
    distance, to the bit); a longer separation changes it."""
    enc = tiny(1)
    batch = encoder_batch(views=1, events=2, edges=0, seed=2)
    batch['side'][:] = 0.0
    batch['players_nan'][:] = 0.0
    xy = batch['players_xy']
    xy[0, 0], xy[0, 5] = torch.tensor([1000.0, 2000.0]), torch.tensor([4000.0, 2000.0])         # three thousand units apart, along x

    def bias(a, b):
        other = {**batch, 'players_xy': xy.clone()}
        other['players_xy'][0, 0], other['players_xy'][0, 5] = torch.tensor(a), torch.tensor(b)
        return enc.attention_bias(other)[0, :, 0, 5]
    base = bias([1000.0, 2000.0], [4000.0, 2000.0])
    assert torch.equal(bias([6000.0, 5000.0], [9000.0, 5000.0]), base)                         # shifted together
    assert torch.equal(bias([1000.0, 2000.0], [1000.0, 5000.0]), base)                         # turned by 90 degrees
    assert not torch.equal(bias([1000.0, 2000.0], [4500.0, 2000.0]), base)                     # farther apart


def test_a_missing_position_never_enters_a_distance():
    """A player without a position (its place is filled with W/2 and its mask channel is 1) and an event without one have no distance:
    moving the filled place changes no score."""
    batch = encoder_batch(views=2, events=5, edges=3, seed=8)
    x = data.PLAYER_NAN_COLUMNS.index('x')
    assert batch['players_nan'][0, 3, x] == 1.0 and batch['ev_nan'][1].any()
    enc = tiny()
    base = enc.attention_bias(batch)
    shifted = {k: v.clone() for k, v in batch.items()}
    shifted['players_xy'][0, 3] += 777.0
    no_place = batch['ev_nan'][..., 0].bool()
    shifted['ev_xy'][no_place] += 555.0
    assert torch.equal(enc.attention_bias(shifted), base)
    nan = batch['geo_pp_nan'].bool()
    garbled = {**batch, 'geo_pp': torch.where(nan, torch.full_like(batch['geo_pp'], 9.0), batch['geo_pp'])}
    assert torch.equal(enc.attention_bias(garbled), base)                                       # a masked geodesic cell is not read
    # with only the distance network (or only the geodesic network) alive, a pair with a token without a place has no score at all
    for keep in ('dist', 'geo'):
        only = copy.deepcopy(enc)
        with torch.no_grad():
            for name, p in only.bias.named_parameters():
                if not name.startswith(keep):
                    p.zero_()
        bias = only.attention_bias(batch)
        assert bias.abs().sum() > 0 and (bias[0, :, 3, :] == 0).all() and (bias[0, :, :, 3] == 0).all(), keep       # the player 3 of view 0
        if keep == 'dist':
            event = FIXED + int(no_place[1].nonzero()[0])                                        # an event of view 1 without a position
            assert (bias[1, :, event, :] == 0).all() and (bias[1, :, :, event] == 0).all()


@pytest.mark.parametrize('dtype, atol', [(torch.float64, 1e-12), (torch.float32, 1e-6)])
def test_attention_bias_is_exactly_swap_covariant(dtype, atol):
    """Design 5.2 and law A1: the teams may change places. In the swapped batch the players and the landmarks sit in other rows, the
    positions are turned by 180 degrees and the side marker is flipped; the bias between two tokens must not notice. The physical places
    of the tokens, the type and the co-participation terms are bit for bit the same (a landmark that stays in place, a nest, stays at its
    physical place, which is not the turned frame place); the learned terms are the same up to the rounding of a matrix product that
    sees its rows in another order (1e-12 in float64, 1e-6 in float32)."""
    batch = encoder_batch(views=4, events=7, edges=6, seed=9, dtype=dtype)
    batch['side'] = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=dtype)
    enc = tiny(3).to(dtype)
    base = enc.attention_bias(batch)
    swapped = data.swap(batch)
    assert swapped['side'].tolist() == [1.0, 0.0, 1.0, 0.0]
    pi, located = token_permutation(7), FIXED + 7
    here, there = enc.physical_positions(batch), enc.physical_positions(swapped)
    assert torch.equal(there.xy, here.xy[:, pi[:located]]) and torch.equal(there.known, here.known[:, pi[:located]])
    want = base[:, :, pi][:, :, :, pi]
    assert torch.allclose(enc.attention_bias(swapped), want, atol=atol, rtol=0)
    assert (want - base).abs().max() > 1e-3                                                      # the test can fail: the tokens do move
    for keep in ('type_bias', 'co'):                                                             # the exact terms alone
        only = copy.deepcopy(enc)
        with torch.no_grad():
            for name, p in only.bias.named_parameters():
                if not name.startswith(keep):
                    p.zero_()
        assert torch.equal(only.attention_bias(swapped), only.attention_bias(batch)[:, :, pi][:, :, :, pi]), keep


class Recorder(torch.nn.Module):
    """Stands in for a small network of the bias: keeps the input it gets and answers with zero for every head."""

    def __init__(self, heads):
        super().__init__()
        self.heads, self.seen = heads, None

    def forward(self, x):
        self.seen = x.detach().clone()
        return torch.zeros(*x.shape, self.heads, dtype=x.dtype)


def test_the_networks_of_the_distance_and_the_geodesic_bias_read_the_documented_inputs():
    """(ENC-5): the distance network gets log(1 + distance / L_U) with the distance between the physical places (an unknown place is moved to
    the middle of the map), the geodesic network gets the standardised geodesic cell, zero where the cell is masked."""
    batch = encoder_batch(views=3, events=5, edges=2, seed=14, dtype=torch.float64)
    batch['side'] = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    x = data.PLAYER_NAN_COLUMNS.index('x')
    junk_player, junk_event = batch['players_nan'][..., x] > 0.5, batch['ev_valid'] & (batch['ev_nan'][..., 0] > 0.5)
    batch['players_xy'][junk_player] = 123.0                                                           # junk where a place is unknown
    batch['ev_xy'][junk_event] = 456.0
    batch['ev_xy'][~batch['ev_valid']] = -789.0
    batch['geo_pp'][batch['geo_pp_nan'] > 0.5] = 9.0                                                   # and in the masked geodesic cells
    batch['geo_pl'][batch['geo_pl_nan'] > 0.5] = 9.0
    enc = tiny(1).double()
    enemy_dist, enemy_geo, landmark_geo = Recorder(2), Recorder(2), Recorder(2)
    enc.bias.dist[1], enc.bias.geo[1], enc.bias.geo[2] = enemy_dist, enemy_geo, landmark_geo
    enc.attention_bias(batch)
    here = enc.physical_positions(batch)
    assert (here.xy[~here.known] == C.W / 2.0).all() and (~here.known).any() and junk_player.any() and junk_event.any()   # the middle of the map
    gap = (here.xy[:, :N_PLAYERS, None, :] - here.xy[:, None, :N_PLAYERS, :]).square().sum(-1).sqrt()
    assert enemy_dist.seen.shape == (3, 10, 10) and torch.allclose(enemy_dist.seen, torch.log1p(gap / C.L_U), atol=1e-12, rtol=0)
    assert torch.equal(enemy_geo.seen, torch.where(batch['geo_pp_nan'] < 0.5, batch['geo_pp'], torch.zeros_like(batch['geo_pp'])))
    assert torch.equal(landmark_geo.seen, torch.where(batch['geo_pl_nan'] < 0.5, batch['geo_pl'], torch.zeros_like(batch['geo_pl'])))
    assert batch['geo_pp_nan'].any() and C.L_U == 2000.0


def test_the_attention_layer_is_the_equation():
    """(ENC-4) against a plain computation from the weights of the layer: normalise, project, softmax of the scaled scores plus the bias
    (a padded key at minus infinity), project back, add; then the feed-forward branch after its own normalisation, added."""
    enc = tiny(5, layers=1).double()
    block = enc.blocks[0]
    g = torch.Generator().manual_seed(1)
    v, n, d, heads = 3, 12, 16, 2
    h = torch.randn(v, n, d, generator=g, dtype=torch.float64)
    bias = torch.randn(v, heads, n, n, generator=g, dtype=torch.float64)
    bias[:, :, :, 9:] = float('-inf')
    x = block.norm1(h)
    q, k, val = (t.permute(0, 2, 1, 3) for t in block.qkv(x).view(v, n, 3, heads, d // heads).unbind(2))
    weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // heads) + bias, dim=-1)
    after_attention = h + block.out((weights @ val).transpose(1, 2).reshape(v, n, d))
    want = after_attention + block.ffn(block.norm2(after_attention))
    assert (weights[:, :, :, 9:] == 0).all() and torch.allclose(block(h, bias), want, atol=1e-10, rtol=0)


def test_the_token_input_is_the_equation():
    """(ENC-3) against a plain computation from the weights: the linear image of the standardised row and the mask channels, the embeddings of
    the categories, the view context and, for a token with a known place, the linear image of the field sample."""
    batch = encoder_batch(views=3, events=6, edges=2, seed=15, dtype=torch.float64)
    batch['side'] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    enc = tiny(2, field_channels=3).double()
    with torch.no_grad():
        enc.field.learned.normal_(std=1.0)
    positions = enc.physical_positions(batch)
    got = enc._embed(batch, positions)                                                                 # noqa: SLF001  (the equation under test)
    ctx = enc.kind_emb.weight[batch['kind']] + enc.context_in(torch.stack([batch['age'], batch['side']], -1))
    field = torch.where(positions.known[..., None], enc.field_proj(enc.field.sample(positions.xy)), torch.zeros(()))
    team = torch.tensor([0] * 5 + [1] * 5)
    players = (enc.players_in(torch.cat([batch['players'], batch['players_nan']], -1)) + enc.zone_emb.weight[batch['players_zone'] + 1]
               + enc.group_emb.weight[team])
    landmarks = enc.landmarks_in(torch.cat([batch['landmarks'], batch['landmarks_nan']], -1)) + enc.landmark_emb.weight
    valid = batch['ev_valid']
    events = (enc.events_in(torch.cat([batch['ev_num'], batch['ev_nan']], -1)) + enc.type_emb.weight[batch['ev_type'].clamp(min=0)]
              + enc.team_emb.weight[batch['ev_team']])
    summary = enc.summary_in(torch.cat([batch['summary'], batch['summary_nan']], -1))[:, None]
    want = torch.cat([players + field[:, :10], landmarks + field[:, 10:44], events + field[:, 44:], summary], 1) + ctx[:, None]
    assert torch.allclose(got[:, :44], want[:, :44], atol=1e-10, rtol=0) and torch.allclose(got[:, -1], want[:, -1], atol=1e-10, rtol=0)
    assert torch.allclose(got[:, 44:-1][valid], want[:, 44:-1][valid], atol=1e-10, rtol=0)             # the padded slots are not read
    assert positions.known[:, :10].any() and (field[:, 44:][valid & (batch['ev_nan'][..., 0] == 0)].abs() > 0).any()


def test_the_two_team_blocks_are_told_apart():
    """Two players with the same row, the same place and the same relations but in different team blocks get different states: the team block
    is a token embedding (design 6.1 types attention by the team, and the heads read the player rows of a team)."""
    batch = encoder_batch(views=2, events=0, edges=0, seed=16)
    for key in ('players', 'players_nan', 'players_zone', 'players_xy'):
        batch[key][:] = batch[key][:, :1]                                                              # all ten players alike
    batch['geo_pp'][:] = 0.3
    batch['geo_pp_nan'][:] = 0.0
    batch['geo_pl'][:] = batch['geo_pl'][:, :1]
    batch['geo_pl_nan'][:] = 0.0
    out = tiny()(batch)
    assert not torch.allclose(out.h_tok[:, 0], out.h_tok[:, 5], atol=1e-6)
    assert torch.allclose(out.h_tok[:, 0], out.h_tok[:, 4], atol=1e-6) and torch.allclose(out.h_tok[:, 5], out.h_tok[:, 9], atol=1e-6)


def test_the_type_bias_names_the_pairs():
    """The learned constant of a pair type (design 6.1, plan Q6): the same team, the other team, a player and a landmark, a player and an
    event he took part in, a player and another event, and every other pair. With the other terms at zero the bias is that constant,
    compared with a plain loop over the tokens (two heads, so that a mix-up of the heads shows)."""
    batch = encoder_batch(views=2, events=8, edges=4, seed=7)
    enc = tiny()
    with torch.no_grad():
        for p in enc.bias.parameters():
            p.zero_()
        enc.bias.type_bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) * torch.tensor([[1.0], [10.0]]))
    classes = (encoder.PAIR_ALLY, encoder.PAIR_ENEMY, encoder.PAIR_LANDMARK, encoder.PAIR_EVENT, encoder.PAIR_ACTOR, encoder.PAIR_OTHER)
    assert sorted(classes) == list(range(6))

    def expected(v, i, j):
        kind = lambda t: 'p' if t < N_PLAYERS else 'l' if t < FIXED else 'e' if t < FIXED + 8 else 's'      # noqa: E731
        ki, kj = kind(i), kind(j)
        if ki == kj == 'p':
            return encoder.PAIR_ALLY if i // 5 == j // 5 else encoder.PAIR_ENEMY
        if {ki, kj} == {'p', 'l'}:
            return encoder.PAIR_LANDMARK
        if {ki, kj} == {'p', 'e'}:
            player, event = (i, j - FIXED) if ki == 'p' else (j, i - FIXED)
            return encoder.PAIR_ACTOR if batch['ev_actors'][v, event, player] > 0.5 else encoder.PAIR_EVENT
        return encoder.PAIR_OTHER
    got = enc.attention_bias(batch)
    n = FIXED + 8 + 1
    want = torch.zeros(2, 2, n, n)
    for v in range(2):
        for i in range(n):
            for j in range(n):
                want[v, :, i, j] = enc.bias.type_bias[:, expected(v, i, j)]
    assert torch.equal(got, want)
    used = {expected(v, i, j) for v in range(2) for i in range(n) for j in range(n)}
    assert used == set(range(6))                                                                 # every type occurs in the test


# =====================================================================================================================================
# co-participation: decay scales and edge weights
# =====================================================================================================================================
def test_coparticipation_decay_is_positive_and_learned():
    """The decay scales kappa_r are 3 positive numbers that the optimiser changes: a parameter of its own, positive for any value of it
    (also at the ends of what a float holds), a decay in (0, 1] that falls with the age of an edge, a gradient from a loss of the
    encoder, and a value that is different after one step."""
    enc = tiny(2)
    co = enc.bias.co
    assert isinstance(co.log_kappa, torch.nn.Parameter) and co.log_kappa.requires_grad and co.log_kappa.shape == (3,)
    assert torch.allclose(co.kappa, torch.tensor([5.0, 20.0, 80.0]))                             # a start between 5 s and the 120 s window
    age = torch.tensor([0.0, 1.0, 5.0, 20.0, 120.0])
    decay = co.decay(age)
    assert decay.shape == (5, 3) and ((decay > 0) & (decay <= 1)).all() and (decay[0] == 1).all()
    assert (decay[1:] < decay[:-1]).all()                                                        # older edges count less
    assert torch.allclose(decay[2], torch.exp(-5.0 / torch.tensor([5.0, 20.0, 80.0])))
    with torch.no_grad():
        co.log_kappa.copy_(torch.tensor([-30.0, 0.0, 30.0]))
    assert (co.kappa > 0).all() and torch.isfinite(co.decay(age)).all()                          # positive and finite at any value
    assert (co.decay(age) >= 0).all() and (co.decay(age) <= 1).all()
    with torch.no_grad():
        co.log_kappa.copy_(torch.log(torch.tensor([5.0, 20.0, 80.0])))
    batch = encoder_batch(views=3, events=6, edges=8, seed=4)
    enc.train()
    loss = enc(batch).h_pool.square().sum()
    loss.backward()
    grad = co.log_kappa.grad
    assert grad is not None and torch.isfinite(grad).all() and (grad.abs() > 0).all()           # every scale learns
    before = co.kappa.detach().clone()
    torch.optim.Adam(co.parameters(), lr=1.0).step()                      # a step of the optimiser family of S4a (large: the gradient is tiny)
    assert not torch.equal(co.kappa, before) and (co.kappa > 0).all() and torch.isfinite(co.kappa).all()


def test_coparticipation_bias_matches_the_hand_case():
    """Three edges between the players 2 and 7 (hand case): 2->7 of kind 1 aged 10 s, 2->7 of kind 3 aged 30 s and 7->2 of kind 1 aged
    0 s, and a padded slot with junk. With kappa = (5, 20, 80) s, forward weights (1, 2, 3) for kind 1 and -1 on the first scale for
    kind 3, and backward weights (0.5, 0.25, 0.125) for kind 1: the bias of (2, 7) adds the forward decays of the first two edges and
    the backward decay of the third, the bias of (7, 2) the reverse, and no other pair has any."""
    enc = tiny()
    co = enc.bias.co
    kappa = (5.0, 20.0, 80.0)
    with torch.no_grad():
        co.log_kappa.copy_(torch.log(torch.tensor(kappa)))
        co.w.zero_()
        co.w_back.zero_()
        co.w[:, 1, :] = torch.tensor([1.0, 2.0, 3.0])
        co.w[:, 3, 0] = -1.0
        co.w_back[:, 1, :] = torch.tensor([0.5, 0.25, 0.125])
    batch = encoder_batch(views=1, events=2, edges=4, seed=0)
    batch['ed_valid'] = torch.tensor([[True, True, True, False]])
    batch['ed_i'], batch['ed_j'] = torch.tensor([[2, 2, 7, 5]]), torch.tensor([[7, 7, 2, 1]])
    batch['ed_kind'], batch['ed_dt'] = torch.tensor([[1, 3, 1, 4]]), torch.tensor([[10.0, 30.0, 0.0, 9.0]])

    def fwd(kind_weights, age):
        return sum(w * math.exp(-age / k) for w, k in zip(kind_weights, kappa))
    one, three, back = (1.0, 2.0, 3.0), (-1.0, 0.0, 0.0), (0.5, 0.25, 0.125)
    want = torch.zeros(10, 10)
    want[2, 7] = fwd(one, 10.0) + fwd(three, 30.0) + fwd(back, 0.0)
    want[7, 2] = fwd(back, 10.0) + fwd(one, 0.0)
    got = co(batch)
    assert got.shape == (1, 2, 10, 10)
    for head in range(2):
        assert torch.allclose(got[0, head], want, atol=1e-6)
    # the term is a part of the bias of the player-player scores: without it the bias differs by exactly this and nowhere else
    without = copy.deepcopy(enc)
    with torch.no_grad():
        without.bias.co.w.zero_()
        without.bias.co.w_back.zero_()
    part = enc.attention_bias(batch) - without.attention_bias(batch)
    assert torch.allclose(part[0, :, :N_PLAYERS, :N_PLAYERS], got[0], atol=1e-6)
    part[0, :, :N_PLAYERS, :N_PLAYERS] = 0.0
    assert (part == 0).all()
    # the scales are learned through this formula: d b(2, 7) / d log kappa_r is the derivative of w_r exp(-age / kappa_r) summed over the
    # three edges (a forward edge 2->7 of kind 1 aged 10 s, kind 3 aged 30 s, and the backward reading of the edge 7->2 aged 0 s)
    got[0, 0, 2, 7].backward()
    ages = ((10.0, one), (30.0, three), (0.0, back))
    analytic = [sum(w[r] * math.exp(-age / kappa[r]) * age / kappa[r] for age, w in ages) for r in range(3)]
    assert torch.allclose(co.log_kappa.grad, torch.tensor(analytic), atol=1e-6)


def single_edge_batch(kind: int, first: int, second: int, age: float = 10.0) -> dict:
    """Two views that each hold the one edge (``first``, ``second``) of the kind ``kind`` and the age ``age``."""
    batch = encoder_batch(views=2, events=3, edges=1, seed=30)
    batch['ed_valid'][:] = True
    batch['ed_i'][:], batch['ed_j'][:], batch['ed_kind'][:], batch['ed_dt'][:] = first, second, kind, age
    return batch


def test_the_direction_of_an_edge_matters_only_where_the_order_is_a_role():
    """Ruling R-S4a-9 and design 5.2 ("a slot number is no role"). S2 lists the pairs of the killing side and of an objective as the killer
    first and then the other participants by ascending participant number, so for two helpers the order of an edge is their slot rank;
    the other kinds name a role first (the killer, the assister, the player who dealt damage; the victim second). Reversing an edge of
    the kinds EDGE_SIDE and EDGE_OBJECTIVE therefore changes nothing, and reversing any other kind changes the bias and the encoding."""
    enc = tiny(7)
    undirected = (tokens.EDGE_SIDE, tokens.EDGE_OBJECTIVE)
    assert encoder.UNDIRECTED_EDGE_KINDS == undirected and (enc.bias.co.w_back[:, list(undirected)] == 0).all()    # their rows start at 0
    for kind in range(len(tokens.EDGE_KINDS)):
        forward, reverse = single_edge_batch(kind, 2, 7), single_edge_batch(kind, 7, 2)
        same_bias = torch.equal(enc.attention_bias(forward), enc.attention_bias(reverse))
        same_state = torch.equal(enc(forward).h_tok, enc(reverse).h_tok)
        assert same_bias == same_state == (kind in undirected), tokens.EDGE_KINDS[kind]
    # an undirected edge is one symmetric entry, and the weights of its backward reading are not read at all (any value there changes nothing)
    side = single_edge_batch(tokens.EDGE_SIDE, 2, 7)
    co, base = enc.bias.co(side), enc.attention_bias(side)
    assert torch.equal(co, co.transpose(-1, -2)) and co.abs().max() > 0
    with torch.no_grad():
        enc.bias.co.w_back[:, list(undirected)] = 100.0
    assert torch.equal(enc.attention_bias(side), base)
    # while a directed kind does read its backward weights
    killer = single_edge_batch(tokens.EDGE_KILLER, 2, 7)
    before = enc.attention_bias(killer)
    with torch.no_grad():
        enc.bias.co.w_back[:, tokens.EDGE_KILLER] = 100.0
    assert not torch.equal(enc.attention_bias(killer), before)
    # the learning follows: the forward weight of an undirected kind gets the gradient of both readings, its backward weight none
    fresh = tiny(7)
    fresh.bias.co(side)[0, 0].sum().backward()
    assert fresh.bias.co.w.grad[0, tokens.EDGE_SIDE].abs().sum() > 0 and (fresh.bias.co.w_back.grad[:, list(undirected)] == 0).all()


def test_the_default_sizes_are_the_two_of_plan_q6():
    s, m = encoder.SIZES['S'], encoder.SIZES['M']
    assert (s.d, s.layers, s.heads) == (128, 4, 4) and (m.d, m.layers, m.heads) == (256, 6, 8)
    assert encoder.EncoderConfig() == s and (s.kappas, s.field_channels) == (3, 8) and 0.0 <= s.dropout < 1.0
    for bad in (dict(d=0), dict(layers=0), dict(heads=3), dict(heads=0), dict(dropout=1.0), dict(dropout=-0.1), dict(kappas=0),
                dict(field_channels=-1), dict(d=True), dict(layers=1.5)):
        with pytest.raises(ValueError):
            encoder.EncoderConfig(**{**TINY, **bad})


# =====================================================================================================================================
# the learned terrain field
# =====================================================================================================================================
GRID, CELL = terrain.GRID, terrain.CELL
COEF = torch.tensor([[1.0, 4.0, 7.0, 10.0], [2.0, -5.0, 8.0, -11.0], [3.0, 6.0, -9.0, 12.0]], dtype=torch.float64)      # a, b, c, d of 3 channels


def bilinear_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """a + b x / 1000 + c y / 1000 + d x y / 1e6 for each of the three channels of ``COEF``: a function that bilinear interpolation of its
    values at the cell centres reproduces exactly (it is linear in x and in y separately). Shape (..., 3)."""
    a, b, c, d = (COEF[:, i] for i in range(4))
    x, y = x[..., None], y[..., None]
    return a + b * x / 1000 + c * y / 1000 + d * x * y / 1e6


def cell_centres() -> tuple:
    """(X, Y) float64 (75, 75) indexed [iy, ix]: the map units of the centre of each cell (cell ix covers [200 ix, 200 ix + 200))."""
    centre = CELL * (torch.arange(GRID, dtype=torch.float64) + 0.5)
    return centre[None, :].expand(GRID, GRID), centre[:, None].expand(GRID, GRID)


def test_terrain_field_samples_bilinearly_and_starts_at_zero():
    """Spec 5.3 (G6c): a 75 x 75 grid of learned vectors, sampled bilinearly at the place of a token, zero at the start. The sampling is
    exact for a function that is linear in x and in y separately (any cell-centre interpolation error shows), the border cell repeats
    outside the centres of the border cells, and a point on the corner of four cells gets their mean."""
    enc = tiny(field_channels=3, static=False).double()
    field = enc.field
    assert field.learned.shape == (GRID, GRID, 3) and isinstance(field.learned, torch.nn.Parameter) and not field.learned.any()
    g = torch.Generator().manual_seed(0)
    points = torch.rand(500, 2, generator=g, dtype=torch.float64) * 14800.0 + 100.0                  # x, y in [100, 14900]
    assert torch.equal(field.sample(points), torch.zeros(500, 3, dtype=torch.float64))             # nothing learned yet: exactly zero
    x_cell, y_cell = cell_centres()
    table = bilinear_function(x_cell, y_cell)
    with torch.no_grad():
        field.learned.copy_(table)
    got = field.sample(points)
    assert torch.allclose(got, bilinear_function(points[:, 0], points[:, 1]), atol=1e-9, rtol=0)
    assert torch.allclose(encoder.bilinear(table, points), got, atol=1e-12, rtol=0)                  # the function and the module agree
    # outside the centres of the border cells the border repeats: the value at the nearest point inside, along the other coordinate too
    y = points[:50, 1]
    for out, inside in ((-300.0, 100.0), (50.0, 100.0), (14950.0, 14900.0), (20000.0, 14900.0)):
        along_x = encoder.bilinear(table, torch.stack([torch.full_like(y, out), y], -1))
        assert torch.allclose(along_x, bilinear_function(torch.full_like(y, inside), y), atol=1e-9, rtol=0), out
        along_y = encoder.bilinear(table, torch.stack([y, torch.full_like(y, out)], -1))
        assert torch.allclose(along_y, bilinear_function(y, torch.full_like(y, inside)), atol=1e-9, rtol=0), out
    far = encoder.bilinear(table, torch.tensor([[-1.0e6, -1.0e6], [1.0e6, 1.0e6]], dtype=torch.float64))   # the map corners, far away
    assert torch.allclose(far[0], table[0, 0], atol=1e-12) and torch.allclose(far[1], table[-1, -1], atol=1e-12)
    # a point on the corner shared by the cells (6, 8), (7, 8), (6, 9), (7, 9) is their mean; at a cell centre it is that cell's value
    corner = encoder.bilinear(table, torch.tensor([[200.0 * 7, 200.0 * 9]], dtype=torch.float64))[0]
    assert torch.allclose(corner, table[8:10, 6:8].mean(dim=(0, 1)), atol=1e-9, rtol=0)
    centre = encoder.bilinear(table, torch.tensor([[200.0 * 7 + 100.0, 200.0 * 9 + 100.0]], dtype=torch.float64))[0]
    assert torch.allclose(centre, table[9, 7], atol=1e-12, rtol=0)
    # the learned vector reaches the token: with a projection of zero the field is without effect, otherwise it changes the encoding
    batch = as_double(encoder_batch(views=2, events=4, edges=3, seed=12))
    with torch.no_grad():
        field.learned.copy_(table)
    seen = enc(batch).h_tok
    with torch.no_grad():
        saved, enc.field_proj.weight[:] = enc.field_proj.weight.clone(), 0.0
        off = enc(batch).h_tok
        field.learned.zero_()
        zero = enc(batch).h_tok
        enc.field_proj.weight.copy_(saved)
    assert torch.equal(off, zero) and not torch.allclose(seen, zero, atol=1e-9)


def test_the_field_is_sampled_at_the_physical_place_of_every_token_after_a_swap():
    """The field is attached to the physical map (spec 5.3): the same token, seen after a team swap, reads the same place, so it gets the
    same field vector, to the bit. The tokens are the players, the landmarks and the events that have a position."""
    batch = encoder_batch(views=4, events=6, edges=3, seed=13)
    batch['side'] = torch.tensor([1.0, 0.0, 0.0, 1.0])
    enc = tiny(field_channels=3)
    with torch.no_grad():
        enc.field.learned.normal_(std=1.0)
    here, there = enc.physical_positions(batch), enc.physical_positions(data.swap(batch))
    pi = token_permutation(6)[:FIXED + 6]
    assert torch.equal(there.xy, here.xy[:, pi]) and torch.equal(there.known, here.known[:, pi])
    assert torch.equal(enc.field.sample(there.xy), enc.field.sample(here.xy)[:, pi])
    assert not torch.equal(enc.field.sample(here.xy), enc.field.sample(here.xy)[:, pi])                  # the test can fail


def test_the_learned_field_has_no_gradient_in_a_wall_and_the_penalty_ignores_walls():
    """Spec 5.3: "no gradient goes to a cell that cannot be walked" and the smoothness penalty is the squared difference between
    neighbouring walkable cells. A block of 10 x 10 wall cells in the middle of the grid; tokens stand in and around it."""
    walk = np.ones((GRID, GRID, 1), dtype=np.float32)
    walk[10:20, 10:20, 0] = 0.0
    enc = tiny(static=walk, field_channels=3)
    assert enc.field.learned.shape == (GRID, GRID, 3) and not enc.field.learned.any()
    assert enc.field.penalty().item() == 0.0                                                         # nothing learned: no difference
    with torch.no_grad():
        enc.field.learned.normal_(std=1.0)
    points = torch.tensor([[200.0 * ix, 200.0 * iy] for ix in range(8, 22) for iy in range(8, 22)] + [[3000.0, 3000.0], [2500.0, 3500.0]])
    sample = enc.field.sample(points)
    assert sample.shape == (len(points), 3 + 1)
    sample[:, :3].square().sum().backward()
    grad = enc.field.learned.grad
    assert (grad[10:20, 10:20] == 0).all()                                                           # no gradient in the wall
    assert (grad[:10].abs() > 0).any() and (grad[10:20, 20:22].abs() > 0).any() and (grad[10:20, 8:10].abs() > 0).any()   # but around it
    # the penalty: a hand case, the 2 x 2 walkable block of the left corner, channel 0 = [[0, 1], [3, 7]] (rows iy), the rest 100
    corner = np.zeros((GRID, GRID, 1), dtype=np.float32)
    corner[:2, :2, 0] = 1.0
    small = tiny(static=corner, field_channels=2)
    with torch.no_grad():
        small.field.learned[:] = 100.0
        small.field.learned[:2, :2, 0] = torch.tensor([[0.0, 1.0], [3.0, 7.0]])
        small.field.learned[:2, :2, 1] = 0.0
    # the four neighbour pairs of the block: (0, 1) and (3, 7) in a row, (0, 3) and (1, 7) in a column: 1 + 16 + 9 + 36 = 62, over 4 pairs
    assert math.isclose(small.field.penalty().item(), 62.0 / 4.0)
    small.field.penalty().backward()
    assert (small.field.learned.grad[2:] == 0).all() and (small.field.learned.grad[:, 2:] == 0).all()    # the walls take no part
    assert small.field.learned.grad[:2, :2].abs().sum() > 0
    none = np.zeros((GRID, GRID, 1), dtype=np.float32)
    assert tiny(static=none, field_channels=2).field.penalty().item() == 0.0                          # no walkable pair: zero, not NaN


def test_static_channels_of_a_terrain(open_terrain, tmp_path):
    """The fixed channels of the field: walkable, the ten zones (one-hot) and the wall fraction, at the cell centres."""
    static = encoder.static_channels(open_terrain)
    assert static.shape == (GRID, GRID, 12) and static.dtype == np.float32 and len(encoder.STATIC_CHANNELS) == 12
    assert encoder.STATIC_CHANNELS[0] == 'walkable' and encoder.STATIC_CHANNELS[-1] == 'wall_fraction'
    assert (static[..., 0] == 1).all() and (static[..., 1:11].sum(-1) == 1).all()                    # an open grid; one zone per cell
    x_cell, y_cell = cell_centres()
    for iy, ix in ((0, 0), (74, 74), (10, 60), (37, 37), (60, 5)):
        zone = int(terrain.zone_of(float(x_cell[iy, ix]), float(y_cell[iy, ix])))
        assert static[iy, ix, 1 + zone] == 1
        assert static[iy, ix, 11] == open_terrain.wall_fraction(float(x_cell[iy, ix]), float(y_cell[iy, ix]))
    assert math.isclose(static[0, 0, 11], 16 / 25, rel_tol=1e-6) and static[37, 37, 11] == 0.0       # cells outside the grid count as wall
    walk = np.zeros((GRID, GRID), dtype=bool)
    walk[20:30, 20:40] = True
    out = tmp_path / 'small'
    terrain.write_terrain_files(out, walk, np.zeros((150, 150)), np.zeros((150, 150)), terrain.geodesic_table(walk))
    terr = terrain.Terrain(out)
    walled = encoder.static_channels(terr)
    del terr
    gc.collect()                                                                                     # Windows does not delete a mapped file
    shutil.rmtree(out, ignore_errors=True)
    assert np.array_equal(walled[..., 0], walk.astype(np.float32)) and walled[..., 11].max() > 0.9 and walled[25, 30, 11] == 0.0
    batch = encoder_batch(views=2, events=3, edges=2)
    for channels in (8, 0):                                                                          # with a learned part, and static only
        assert torch.isfinite(tiny(static=walled, field_channels=channels)(batch).h_tok).all()
    for bad in (walled[..., :0], walled[:70], np.where(walk[..., None], 0.5, walled), np.full_like(walled, np.nan)):
        with pytest.raises(ValueError):
            tiny(static=bad)


# =====================================================================================================================================
# the static channels are required; every tensor is made where its inputs are
# =====================================================================================================================================
def test_the_toy_static_channels_have_the_rules_of_the_real_helper(open_terrain):
    """``ept_toys.toy_static`` is the terrain that the tests give the encoder: its open grid equals ``static_channels`` of the open terrain in
    every bit, and with the block of walls the walkable flag is 0 there and the wall fraction is 1 in the middle of it."""
    assert np.array_equal(toy_static(False), encoder.static_channels(open_terrain))
    walled = toy_static()
    assert walled.shape == (GRID, GRID, 12) and walled.dtype == np.float32 and not walled.flags.writeable
    assert (walled[20:35, 30:45, 0] == 0).all() and walled[..., 0].sum() == GRID * GRID - 15 * 15 and (walled[..., 1:11].sum(-1) == 1).all()
    assert walled[27, 37, 11] == 1.0 and walled[0, 0, 11] > 0.6 and walled[60, 10, 11] == 0.0


def test_static_channels_are_required_unless_false_is_given():
    """Review minor 8: an encoder without the static channels trains without the walkable mask, the zones and the wall fraction, so it has to
    be asked for with ``static=False``; leaving the argument out, None and True are refused."""
    cfg = encoder.EncoderConfig(**TINY)
    with pytest.raises(TypeError):
        encoder.EPTEncoder(cfg)
    for refused in (None, True):
        with pytest.raises(ValueError, match='static=False'):
            encoder.EPTEncoder(cfg, refused)
    bare = encoder.EPTEncoder(cfg, False)
    assert bare.field.n_channels == cfg.field_channels and (bare.field.walk == 1).all() and bare.field.static.shape == (GRID, GRID, 0)
    full = encoder.EPTEncoder(cfg, toy_static())
    assert full.field.n_channels == cfg.field_channels + 12 and torch.equal(full.field.walk, torch.from_numpy(toy_static()[..., :1].copy()))
    assert encoder.EPTEncoder(encoder.EncoderConfig(**{**TINY, 'field_channels': 0}), False).field is None      # nothing learned, nothing fixed


TENSOR_FACTORIES = frozenset({'zeros', 'ones', 'full', 'empty', 'arange', 'tensor', 'as_tensor', 'rand', 'randn', 'randint', 'randperm', 'eye',
                              'linspace', 'logspace', 'from_numpy', 'scalar_tensor'})
BUILD_TIME = frozenset({'__init__', '__post_init__', '_landmark_places', 'static_channels', 'token_layout'})      # run before a module is moved


def factories_without_a_device(source: str) -> list:
    """``(function, line)`` of every call ``torch.<factory>(...)`` without a ``device=`` argument in a function that is not run when a module
    is built (the parameters and buffers of a module follow it to a device, a tensor made while encoding does not)."""
    found = set()
    for func in ast.walk(ast.parse(source)):
        if not isinstance(func, ast.FunctionDef) or func.name in BUILD_TIME:
            continue
        for call in ast.walk(func):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == 'torch' and call.func.attr in TENSOR_FACTORIES
                    and not any(keyword.arg == 'device' for keyword in call.keywords)):
                found.add((func.name, call.lineno))
    return sorted(found)


def test_every_tensor_made_while_encoding_gets_its_device_from_its_inputs():
    """Review important 1: ``torch.zeros(shape)`` is on the CPU whatever the batch is, and adding it to a CUDA tensor fails, which would stop
    the GPU half of the cost check. Every tensor that the encoder makes while it runs is made with ``*_like``, ``new_*`` or ``device=``."""
    assert factories_without_a_device(Path(encoder.__file__).read_text(encoding='utf-8')) == []
    assert factories_without_a_device('def f(x):\n    return torch.zeros(3)\n') == [('f', 2)]                 # the scan can fail
    assert factories_without_a_device('def f(x):\n    return torch.zeros(3, device=x.device)\n') == []
    assert factories_without_a_device('class A:\n    def __init__(self):\n        self.a = torch.zeros(3)\n') == []


@pytest.mark.skipif(os.environ.get('C_EPT_CUDA_TESTS') != '1' or not torch.cuda.is_available(),
                    reason='opt-in: set C_EPT_CUDA_TESTS=1 (and have a GPU that nothing else, such as the local model, is using)')
def test_the_encoder_runs_forward_and_backward_on_the_gpu():
    """Review important 1 on a GPU: the encoder and a toy batch on CUDA give the output on CUDA, equal to the CPU output to 2e-3 (the attention
    kernels differ), and every parameter gets a finite gradient on CUDA. Opt-in, because the test makes a CUDA context, which a suite run must
    not do while the GPU belongs to another job; the controller runs it before the cost check."""
    cpu = tiny(3)
    batch = encoder_batch(views=3, events=9, edges=7, seed=41)
    expected = cpu(batch)
    gpu = copy.deepcopy(cpu).to('cuda')
    out = gpu({k: v.to('cuda') for k, v in batch.items()})
    assert out.h_tok.device.type == out.h_pool.device.type == out.mask.device.type == 'cuda'
    assert torch.equal(out.mask.cpu(), expected.mask) and torch.allclose(out.h_tok.cpu(), expected.h_tok, atol=2e-3)
    (out.h_tok.square().sum() + out.h_pool.sum()).backward()
    assert all(p.grad is not None and p.grad.device.type == 'cuda' and bool(torch.isfinite(p.grad).all()) for p in gpu.parameters())


def test_the_encoder_module_reads_no_key_no_holdout_and_no_cohort_module():
    from c_ept.tests.conftest import all_imports
    source = Path(encoder.__file__)
    text = source.read_text(encoding='utf-8')
    assert 'matches/key' not in text and 'allow_holdout' not in text and 'split_module' not in text and 'bucket(' not in text
    assert not all_imports(source) & {'c_ept.cohort', 'c_ept.reader', 'c_ept.run_s1', 'c_ept.run_s2', 'c_ept.run_s3', 'pickle', 'lightgbm'}
    assert 'cuda' not in text.replace('CUDA', '').lower()                                              # nothing here puts a tensor on a GPU


# =====================================================================================================================================
# determinism
# =====================================================================================================================================
def fingerprint(threads: int = 8, seed: int = 20261022) -> str:
    """SHA-256 of the forward outputs and of the gradients of a small encoder (dropout on, so the random draws count) built, run and
    differentiated under ``torch.use_deterministic_algorithms(True)`` with ``threads`` threads from one seed."""
    before = torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)
    try:
        torch.manual_seed(seed)
        enc = encoder.EPTEncoder(encoder.EncoderConfig(d=16, layers=2, heads=2, dropout=0.1), toy_static())
        enc.train()
        out = enc(encoder_batch(views=4, events=11, edges=9, seed=5))
        (out.h_tok.square().sum() + out.h_pool.sum()).backward()
        digest = hashlib.sha256()
        for t in (out.h_tok, out.h_pool, *(p.grad for p in enc.parameters())):
            digest.update(t.detach().numpy().tobytes() if t is not None else b'none')
        return digest.hexdigest()
    finally:
        torch.use_deterministic_algorithms(before[0])
        torch.set_num_threads(before[1])


def test_cpu_forward_is_deterministic():
    """Review focus 5: one seed and one thread count give the same bits, run after run (forward and backward, dropout on), and in a second
    process too (nothing may depend on the address of an object, the order of a set or the state of the process)."""
    first = fingerprint()
    assert fingerprint() == first
    assert fingerprint(seed=1) != first                                              # the fingerprint can differ
    root = Path(encoder.__file__).resolve().parents[2]
    code = 'from c_ept.tests.test_ept_encoder import fingerprint; print(fingerprint())'
    run = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True, timeout=300, check=False)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip().splitlines()[-1] == first
