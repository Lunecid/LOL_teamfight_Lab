"""Small synthetic CPU checks for N2; no stores, fitting, checkpoints or CUDA.

The guards are installed before importing project/torch modules. Run this file
with ``python -B -m pytest -p no:cacheprovider tests/test_ept_team_set.py`` and
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; native thread limits/CUDA hiding are set below.
"""
import importlib.abc
import os
import sys

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True


class _DenyProtected(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('c_ept.cohort', 'research_pipeline.cohort_policy') or fullname.startswith('research_pipeline.holdout_r3'):
            raise AssertionError('protected import denied in N2 synthetic tests')


_guard = _DenyProtected()
_guard_active = True
sys.meta_path.insert(0, _guard)


def _deny_open(event, args):
    if not _guard_active or event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = os.path.abspath(os.fsdecode(args[0])).replace('\\', '/').lower()
    if ('/c_ept/cohort.' in path or '/c_ept/__pycache__/cohort.' in path
            or '/research_pipeline/cohort_policy.' in path or '/research_pipeline/__pycache__/cohort_policy.' in path
            or '/research_pipeline/holdout_r3/' in path):
        raise AssertionError('protected file open denied in N2 synthetic tests')


sys.addaudithook(_deny_open)

import numpy as np
import pytest
import torch

from c_ept.ept import data, encoder, heads
from c_ept.models.neural import TeamSetModel
from c_ept.models.neural.team_set import TeamSetEncoder
from c_ept.preprocessing import tokens
from c_ept.tests.ept_toys import encoder_batch, toy_static

torch.set_num_threads(1)


def teardown_module():
    global _guard_active
    _guard_active = False
    sys.meta_path.remove(_guard)


def tiny(seed=7, **changes):
    torch.manual_seed(seed)
    cfg = encoder.EncoderConfig(**({'d': 16, 'layers': 1, 'heads': 2, 'dropout': 0.0, 'field_channels': 2} | changes))
    return TeamSetModel(cfg, toy_static(), hidden=32).eval()


def batch(seed=0, events=5, edges=4):
    return encoder_batch(views=2, events=events, edges=edges, seed=seed)


def permute_players(b, perm):
    """New row r = old perm[r], with every player reference remapped."""
    perm = torch.as_tensor(perm)
    inverse = torch.argsort(perm)
    out = dict(b)
    for key in ('players', 'players_nan', 'players_zone', 'players_xy', 'geo_pl', 'geo_pl_nan'):
        out[key] = b[key][:, perm]
    for key in ('geo_pp', 'geo_pp_nan'):
        out[key] = b[key][:, perm][:, :, perm]
    out['ev_actors'] = b['ev_actors'][..., perm]
    for key in ('ed_i', 'ed_j'):
        out[key] = torch.where(b[key] >= 0, inverse[b[key].clamp_min(0)], b[key])
    return out


def test_protected_import_and_open_guards_without_accessing_sources():
    for name in ('c_ept.cohort', 'research_pipeline.cohort_policy', 'research_pipeline.holdout_r3.sentinel'):
        with pytest.raises(AssertionError, match='protected import'):
            _guard.find_spec(name)
    for path in ('c_ept/cohort.py', 'research_pipeline/cohort_policy.py', 'research_pipeline/holdout_r3/sentinel.py',
                 'c_ept/__pycache__/cohort.cpython-311.pyc'):
        # Emit the audit event only. No import, file open, or source read occurs.
        with pytest.raises(AssertionError, match='protected file open'):
            sys.audit('open', path, 'r', 0)


def test_explicit_model_api_never_constructs_ept_attention(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('N2 must not initialize EPT/attention')
    for cls in (heads.EPT, encoder.EPTEncoder, encoder.RelationalBias, encoder.Coparticipation, encoder._Block, torch.nn.MultiheadAttention):
        monkeypatch.setattr(cls, '__init__', forbidden)
    model = tiny()
    assert model.architecture_id == 'n2_teamset_v1'
    assert isinstance(model, torch.nn.Module) and not isinstance(model, heads.EPT)
    assert isinstance(model.encoder, TeamSetEncoder) and not isinstance(model.encoder, encoder.EPTEncoder)
    assert model.encoder.cfg.d == 16 and model.encoder.hidden == 32
    assert type(model.heads) is heads.EPTHeads and model.heads.cfg is model.encoder.cfg
    assert not any(name in dict(model.encoder.named_modules()) for name in ('bias', 'blocks', 'attention'))
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('events,edges', [(0, 0), (1, 1), (5, 4)])
def test_token_layout_masks_zero_padding_and_empty_sets(events, edges):
    model, b = tiny(), batch(events=events, edges=edges)
    out = model.encode(b)
    assert type(out) is encoder.Encoded
    assert out.h_tok.shape == (2, 10 + 34 + events + 1, 16)
    assert out.h_pool.shape == (2, 16)
    assert out.mask.dtype is torch.bool
    assert torch.equal(out.mask[:, 44:44 + events], b['ev_valid'])
    assert out.mask[:, :44].all() and out.mask[:, -1].all()
    assert torch.count_nonzero(out.h_tok[~out.mask]) == 0
    assert torch.equal(out.h_pool, out.h_tok[:, -1])
    assert torch.isfinite(out.h_tok).all()


@pytest.mark.parametrize('key', [k for k in data.VIEW_KEYS if k != 'view'])
def test_missing_essential_view_field_fails_closed(key):
    b = batch()
    del b[key]
    with pytest.raises(ValueError, match=key):
        tiny().encode(b)


def test_identifiers_and_row_targets_are_not_predictor_inputs():
    model, b = tiny(), batch()
    expected = model.encode(b)
    out = dict(b)
    del out['view']
    out['future_target_sentinel'] = object()
    actual = model.encode(out)
    assert torch.equal(expected.h_tok, actual.h_tok)
    assert torch.equal(expected.h_pool, actual.h_pool)


@pytest.mark.parametrize('seed', [0, 4, 19])
def test_within_team_permutation_equivariance_with_all_index_remapping(seed):
    model, b = tiny(), batch(seed)
    perm = [3, 1, 4, 0, 2, 8, 5, 9, 6, 7]
    a, z = model.encode(b), model.encode(permute_players(b, perm))
    torch.testing.assert_close(z.h_tok[:, :10], a.h_tok[:, perm], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(z.h_tok[:, 10:], a.h_tok[:, 10:], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(z.h_pool, a.h_pool, atol=2e-6, rtol=2e-6)
    assert torch.equal(z.mask, a.mask)


def test_event_and_edge_set_order_is_irrelevant():
    model, b = tiny(), batch()
    perm, edges = torch.tensor([4, 2, 0, 3, 1]), torch.tensor([3, 0, 2, 1])
    changed = {k: v[:, perm] if k.startswith('ev_') else v[:, edges] if k.startswith('ed_') else v for k, v in b.items()}
    a, z = model.encode(b), model.encode(changed)
    expected = torch.cat([a.h_tok[:, :44], a.h_tok[:, 44:49][:, perm], a.h_tok[:, -1:]], 1)
    torch.testing.assert_close(z.h_tok, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(z.h_pool, a.h_pool, atol=2e-6, rtol=2e-6)


def test_padding_masked_geometry_and_unknown_positions_never_enter_outputs_or_gradients():
    model, b = tiny(), batch()
    a = model.encode(b)
    poisoned = {k: v.clone() for k, v in b.items()}
    invalid = ~b['ev_valid']
    for key in ('ev_num', 'ev_nan', 'ev_xy', 'ev_actors'):
        poisoned[key][invalid] = float('nan')
    for key in ('ev_type', 'ev_team'):
        poisoned[key][invalid] = 9999
    invalid_ed = ~b['ed_valid']
    for key in ('ed_i', 'ed_j', 'ed_kind'):
        poisoned[key][invalid_ed] = 9999
    poisoned['ed_dt'][invalid_ed] = float('nan')
    for key in ('geo_pp', 'geo_pl'):
        poisoned[key][b[key + '_nan'] > 0.5] = float('nan')
    unknown = b['players_nan'][..., data.PLAYER_NAN_COLUMNS.index('x')] > 0.5
    poisoned['players_xy'][unknown] = float('nan')
    z = model.encode(poisoned)
    assert torch.equal(a.h_tok, z.h_tok)
    assert torch.equal(a.h_pool, z.h_pool)
    z.h_pool.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters())


def test_unordered_edge_endpoint_order_is_ignored_but_directed_roles_matter():
    model, b = tiny(), batch(events=2, edges=1)
    b['ed_kind'][1, 0] = tokens.EDGE_SIDE
    a = model.encode(b)
    changed = dict(b, ed_i=b['ed_j'], ed_j=b['ed_i'])
    assert torch.equal(a.h_pool, model.encode(changed).h_pool)
    b['ed_kind'][1, 0] = 0  # killer -> victim, a directed kind
    changed = dict(b, ed_i=b['ed_j'], ed_j=b['ed_i'])
    assert not torch.allclose(model.encode(b).h_pool, model.encode(changed).h_pool, atol=1e-7, rtol=0)


@pytest.mark.parametrize('key', ['ed_i', 'ed_j', 'ed_kind', 'ed_dt'])
def test_invalid_live_edges_fail_closed(key):
    b = batch()
    b[key][1, 0] = -1
    with pytest.raises(ValueError, match='edge'):
        tiny().encode(b)


def test_same_geometry_embedding_and_field_policy_as_existing_ept():
    model, b = tiny(), batch()
    actual = model.encoder.physical_positions(b)
    reference = encoder.EPTEncoder.physical_positions(model.encoder, b)
    assert all(torch.equal(a, z) for a, z in zip(actual, reference))
    assert torch.equal(model.encoder._embed(b, actual), encoder.EPTEncoder._embed(model.encoder, b, reference))
    swapped = data.swap(b)
    a = model.encoder.physical_positions(b).xy
    z = model.encoder.physical_positions(swapped).xy
    perm = list(data.PLAYER_PERM) + [10 + int(i) for i in data.LANDMARK_PERM] + list(range(44, a.shape[1]))
    assert torch.equal(z, a[:, perm])
    field = model.encoder.field
    assert type(field) is encoder.TerrainField
    assert field.learned.shape == (75, 75, 2) and field.n_channels == 14
    assert np.array_equal(field.static.numpy(), toy_static())
    assert torch.count_nonzero(field.learned) == 0
    model.encode(b).h_pool[:, 0].sum().backward()
    assert torch.count_nonzero(field.learned.grad[field.walk[..., 0] == 0]) == 0
    assert torch.isfinite(field.penalty())
    cfg = encoder.EncoderConfig(d=16, heads=2, field_channels=0)
    assert TeamSetModel(cfg, False, hidden=32).encoder.field is None
    assert TeamSetModel(cfg, toy_static(), hidden=32).encoder.field.n_channels == 12
    cfg_learned = encoder.EncoderConfig(d=16, heads=2, field_channels=2)
    assert TeamSetModel(cfg_learned, False, hidden=32).encoder.field.n_channels == 2
    for bad in (None, True):
        with pytest.raises(ValueError, match='static'):
            TeamSetModel(cfg, bad, hidden=32)
    with pytest.raises(TypeError):
        TeamSetModel(cfg)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='hidden'):
            TeamSetModel(cfg, False, hidden=bad)


def test_each_allowed_field_has_a_computational_path():
    """Positive interventions: every ViewTensors field except view is consumed.

    No data are read and no target/performance comparison is made. The raw
    geometry/actor/edge interventions catch a constant relation-summary shortcut.
    """
    model, b = tiny(), batch(seed=11)
    edits = {
        'kind': ((1,), lambda x: (x + 1) % 3), 'side': ((1,), lambda x: 1 - x), 'age': ((1,), lambda x: x + 0.7),
        'players': ((1, 0, 0), lambda x: x + 1), 'players_nan': ((1, 0, 0), lambda x: 1 - x),
        'players_zone': ((1, 0), lambda x: (x + 2) % 10), 'players_xy': ((1, 0, 0), lambda x: x + 311),
        'landmarks': ((1, 0, 0), lambda x: x + 1), 'landmarks_nan': ((1, 0, 0), lambda x: 1 - x),
        'summary': ((1, 0), lambda x: x + 1), 'summary_nan': ((1, 0), lambda x: 1 - x),
        'ev_valid': ((1, 0), lambda x: ~x), 'ev_type': ((1, 0), lambda x: (x + 1) % encoder.N_EVENT_TYPES),
        'ev_team': ((1, 0), lambda x: (x + 1) % 3), 'ev_num': ((1, 0, 0), lambda x: x + 1),
        'ev_xy': ((1, 0, 0), lambda x: x + 311), 'ev_nan': ((1, 0, 0), lambda x: 1 - x),
        'ev_actors': ((1, 0, 0), lambda x: 1 - x), 'ed_valid': ((1, 0), lambda x: ~x),
        'ed_i': ((1, 0), lambda x: (x + 1) % 10), 'ed_j': ((1, 0), lambda x: (x + 1) % 10),
        'ed_kind': ((1, 0), lambda x: (x + 1) % len(tokens.EDGE_KINDS)), 'ed_dt': ((1, 0), lambda x: x + 17),
        'geo_pp': ((1, 0, 1), lambda x: x + 1), 'geo_pp_nan': ((1, 0, 1), lambda x: 1 - x),
        'geo_pl': ((1, 0, 0), lambda x: x + 1), 'geo_pl_nan': ((1, 0, 0), lambda x: 1 - x),
    }
    assert set(edits) == set(data.VIEW_KEYS) - {'view'}
    # Use a located event so ev_xy has a meaningful intervention.
    b['ev_nan'][1, 0, 0] = 0
    a = model.encode(b).h_pool
    for key, (at, change) in edits.items():
        z = dict(b)
        z[key] = b[key].clone()
        z[key][at] = change(z[key][at])
        assert not torch.equal(a, model.encode(z).h_pool), f'{key} was not consumed'


def _conditions(n=2):
    return dict(q=torch.arange(n), m=torch.tensor([3, 50]), z=torch.tensor([2, -1]), s=torch.tensor([0, 2]),
                K=torch.tensor([[1., 2., .3, .7], [3., 1., .8, .2]]), O=torch.arange(n * 18).reshape(n, 18).float() % 3,
                exposure=torch.ones(n, 18), window=torch.tensor([.5, 1.]))


def test_all_shared_heads_forward_backward_every_parameter_is_used():
    model, b = tiny(), batch()
    enc, c = model.encode(b), _conditions()
    outputs = list(model.heads.m2(enc, c['exposure']))
    for q, m, z in ((c['q'], c['m'], c['z']), (None, None, None)):
        outputs += list(model.heads.m3(enc, q, m, z))
        outputs += [model.heads.m3_obj(enc, q, m, z, c['s'], c['K']),
                    model.heads.m4(enc, q, m, z, c['K'], c['O'], c['window'])]
    assert all(torch.isfinite(t).all() for t in outputs)
    sum(t.square().mean() for t in outputs).backward()
    missing, nonfinite, disconnected = [], [], []
    for name, p in model.named_parameters():
        if p.grad is None:
            missing.append(name)
        elif not torch.isfinite(p.grad).all():
            nonfinite.append(name)
        elif p.numel() and torch.count_nonzero(p.grad) == 0:
            disconnected.append(name)
    assert not missing and not nonfinite and not disconnected, (missing, nonfinite, disconnected)
    assert not torch.cuda.is_initialized()


def test_frame_averaged_team_swap_for_all_prediction_laws():
    model, b = tiny(), batch()
    fa, c = heads.FrameAveraged(model).eval(), _conditions()
    a, z = fa.encode(b), fa.encode(data.swap(b))
    swapped = dict(m=torch.as_tensor(data.MARK_PERM)[c['m']],
                   z=torch.where(c['z'] >= 0, torch.as_tensor(data.ZONE_PERM)[c['z'].clamp_min(0)], c['z']),
                   K=c['K'][:, torch.as_tensor(data.K_PERM)], O=c['O'][:, torch.as_tensor(data.CELL_PERM)])
    mark, zone = torch.as_tensor(data.MARK_PERM), torch.as_tensor(data.ZONE_PERM)
    team, cell = torch.as_tensor(data.TEAM_PERM), torch.as_tensor(data.CELL_PERM)
    am2, zm2 = fa.m2(a, c['exposure']), fa.m2(z, c['exposure'])
    assert torch.equal(am2.rate, zm2.rate)
    assert torch.equal(am2.mark_logp, zm2.mark_logp.index_select(-1, mark))
    assert torch.equal(am2.zone_logp, zm2.zone_logp.index_select(2, mark).index_select(3, zone))
    for q, m, zone_input in ((c['q'], c['m'], c['z']), (None, None, None)):
        sm, sz = (swapped['m'], swapped['z']) if q is not None else (None, None)
        am3, zm3 = fa.m3(a, q, m, zone_input), fa.m3(z, q, sm, sz)
        assert torch.equal(am3.size_logp, zm3.size_logp)
        assert torch.equal(am3.kill_rate, zm3.kill_rate.index_select(-1, team))
        assert torch.equal(am3.gold_logp, zm3.gold_logp.index_select(2, team))
        assert torch.equal(fa.m3_obj(a, q, m, zone_input, c['s'], c['K']),
                           fa.m3_obj(z, q, sm, sz, c['s'], swapped['K']).index_select(-1, cell))
        assert torch.equal(fa.m4(a, q, m, zone_input, c['K'], c['O'], c['window']),
                           fa.m4(z, q, sm, sz, swapped['K'], swapped['O'], c['window']).index_select(-1, cell))


def test_in_memory_state_dict_roundtrip_and_seed_reproducibility():
    a, b = tiny(seed=13), tiny(seed=13)
    inputs = batch()
    assert torch.equal(a.encode(inputs).h_tok, b.encode(inputs).h_tok)
    z = tiny(seed=14)
    z.load_state_dict(a.state_dict(), strict=True)
    assert torch.equal(a.encode(inputs).h_tok, z.encode(inputs).h_tok)
    assert 'encoder.field.static' not in a.state_dict()  # same external static identity policy as EPT


def test_default_capacity_audit_no_trainable_padding():
    cfg, static = encoder.EncoderConfig(), toy_static()
    # Count-only independent reference construction; never part of N2 init/encode.
    ept, n2 = heads.EPT(cfg, static), TeamSetModel(cfg, static)
    count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    a, z = count(ept), count(n2)
    assert count(ept.heads) == count(n2.heads)
    assert count(ept.encoder.field) == count(n2.encoder.field) == 45000
    assert 0.8 <= z / a <= 1.2, (a, z, z / a)
    assert count(TeamSetModel(cfg, static, hidden=256)) < z
    print(f'N2_CAPACITY ept_total={a} n2_total={z} ratio={z / a:.9f} heads={count(n2.heads)} '
          f'ept_encoder={count(ept.encoder)} n2_encoder={count(n2.encoder)} learned_field={count(n2.encoder.field)} '
          f'static_field={n2.encoder.field.static.numel()}')


def test_one_frame_wrapper_on_n2_returns_the_heads_answers_and_is_not_swap_invariant_while_the_average_is():
    """EPT-T012 S4b preparation M7 (decisions D10 P1, D17): N2 is trained and evaluated with the same swap settings as the EPT. ``FrameAveraged(N2, average=False)`` is one call of the
    encoder and the heads of N2 as they are (recorded frame, or the swapped frame of a view with its coin up); the average of the same model stays exactly invariant (control)."""
    model, b, c = tiny(), batch(), _conditions()
    one = heads.FrameAveraged(model, average=False).eval()
    with torch.no_grad():
        enc = one.encode(b)
        plain = model.encode(b)
        assert enc.swapped is None and torch.equal(enc.original.h_pool, plain.h_pool)
        assert all(torch.equal(x, y) for x, y in zip(one.m2(enc, c['exposure']), model.heads.m2(plain, c['exposure'])))
        assert all(torch.equal(x, y) for x, y in zip(one.m3(enc, c['q'], c['m'], c['z']), model.heads.m3(plain, c['q'], c['m'], c['z'])))
        assert torch.equal(one.m3_obj(enc, None, None, None, c['s'], c['K']), model.heads.m3_obj(plain, None, None, None, c['s'], c['K']))
        assert torch.equal(one.m4(enc, None, None, None, c['K'], c['O'], c['window']), model.heads.m4(plain, None, None, None, c['K'], c['O'], c['window']))
        up = one.encode(dict(b, swap_view=torch.ones(2, dtype=torch.bool)))                      # every coin up: the encoding of the swapped batch
        assert torch.equal(up.original.h_pool, model.encode(data.swap(b)).h_pool) and torch.equal(up.side, (1 - b['side']) > 0.5)         # the side marker flips with the swap
        first, second = one.m2(enc, c['exposure']), one.m2(up, c['exposure'])
        assert (first.mark_logp - second.mark_logp.index_select(-1, torch.as_tensor(data.MARK_PERM))).abs().max() > 1e-6        # one frame alone is not invariant
        fa = heads.FrameAveraged(model).eval()
        assert torch.equal(fa.m2(fa.encode(b), c['exposure']).mark_logp, fa.m2(fa.encode(data.swap(b)), c['exposure']).mark_logp.index_select(-1, torch.as_tensor(data.MARK_PERM)))
