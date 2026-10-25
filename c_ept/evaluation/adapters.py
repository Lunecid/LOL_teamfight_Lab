"""Inference-only scientific adapters. Call only after the operator's terminal guard.

The caller owns role-scoped stores, best-only verified model loading, frozen FIT
stats/scales, identity and resource guards. This module never opens an artifact.
"""
from contextlib import contextmanager

import numpy as np
import torch

from c_ept import config as C
from c_ept.ept import data, loss
from c_ept.ept.heads import EPT, FrameAveraged
from c_ept.models import baselines_b1 as B1, nll as nll_math
from c_ept.valuation import compose, towertier
from .endpoints import endpoint, m2_endpoints, m4_endpoints

# G9b structural estimand: M2 whole layer plus M3/M4 open views. The five
# masked M3/M4 training blocks remain diagnostics and never enter this mean.
STRUCTURAL_HEADS = ('M2Rate/masked', 'M2Mark/masked', 'M2Zone/masked',
                    'M3Size/open', 'M3Kills/open', 'M3Gold/open', 'M3Obj/open', 'M4Conv/open')


def _numpy(t):
    return t.detach().cpu().numpy()


@contextmanager
def _evaluation(model):
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        for m, mode in modes:
            m.training = mode


def _law(name, a, shape, probability=False):
    return compose._checked(name, a, shape, unit_sum=probability)


class B1OpenHeads:
    """Open event-view heads; there is deliberately no composer fallback."""
    def __init__(self, b1):
        missing = set(B1.OPEN_HEAD_NAMES) - set(b1.open)
        if missing:
            raise ValueError(f'missing B1 open heads: {sorted(missing)}')
        self.heads = b1.open

    def laws(self, numeric, *, check):
        n = len(numeric)
        check()
        size = _law('B1 open size', self.heads['M3Size'].predict_blocks(numeric), (n, 4), True)
        kill = np.empty((n, 4, 2))
        gold = np.empty((n, 4, 2, 6, 31))
        for s in range(4):
            check()
            ss = np.full(n, s, dtype=np.int64)
            kill[:, s] = self.heads['M3Kills'].predict_blocks(numeric, s=ss)
            for t in (0, 1):
                for layer in range(1, 7):
                    check()
                    gold[:, s, t, layer - 1] = self.heads['M3Gold'].predict_blocks(
                        numeric, np.full(n, t, dtype=np.int64), s=ss,
                        layer=np.full(n, layer, dtype=np.int64))
        return size, _law('B1 open kills', kill, (n, 4, 2)), _law('B1 open gold', gold, (n, 4, 2, 6, 31), True)

    def objects(self, numeric, size, generated_K):
        k = np.asarray(generated_K, dtype=np.float64).copy()
        k[:, 2:] /= 1000.0
        x = np.column_stack((numeric, k)).astype(np.float32)
        return _law('B1 open objects', self.heads['M3Obj'].predict_blocks(x, s=size), (len(size), 18))


def sample_open_value(laws, objects, prices, gold_reps, *, row, match, draws, seed, check, batch_size):
    """Sample the same size -> kills -> gold -> objects law as Composer (no conversion).

    objects receives ONLY generated K. RNG is stable per (seed, match, unit),
    independent of endpoint ordering and batching. Paired models share initial
    streams; Poisson rejection sampling need not consume equal random numbers.
    """
    size, kills, gold = laws
    size = _law('size', size, (4,), True)
    kills = _law('kills', kills, (4, 2))
    gold = _law('gold', gold, (4, 2, 6, 31), True)
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(match), int(row)]))
    s = compose._inverse_cdf(np.broadcast_to(size, (draws, 4)), rng.random(draws))
    k = rng.poisson(kills[s]).astype(np.int64)
    layer = np.clip(k, 1, 6) - 1
    u = rng.random((draws, 2))
    bins = np.column_stack([compose._inverse_cdf(gold[s, t, layer[:, t]], u[:, t]) for t in (0, 1)])
    g = np.where(k > 0, gold_reps[bins], 0.0)
    K = np.column_stack((k, g))
    rates = np.empty((draws, 18))
    for lo in range(0, draws, batch_size):
        check()
        hi = min(lo + batch_size, draws)
        rates[lo:hi] = objects(s[lo:hi], K[lo:hi])
    rates = _law('objects', rates, (draws, 18))
    O = rng.poisson(rates)
    f = g[:, 0] - g[:, 1] + (O * prices).sum(axis=1)
    plus, draw = float(np.mean(f > 0)), float(np.mean(f == 0))
    # Unbiased sample-variance estimate of the MC mean's SE; zero at boundaries.
    se = np.sqrt(plus * (1 - plus) / (draws - 1))
    draw_se = np.sqrt(draw * (1 - draw) / (draws - 1))
    return plus, draw, float(se), float(draw_se)


def open_cell_values(bundle, match, unit):
    """Fixed flow prices and the existing tower-tier expectation at M3 time f."""
    prices = compose._prices(bundle.prices)
    when = np.asarray(bundle.s1.array('units/f'))[unit].astype(np.int64)
    destroyed = towertier.destroyed_before(bundle.s1, match, when)
    out = np.empty((len(unit), 18), dtype=np.float64)
    for kind, name in enumerate(C.O_KINDS):
        for team in (0, 1):
            if name == 'tower':
                taker = np.full(len(unit), team, dtype=np.int64)
                victim = towertier.victim_counts(destroyed, taker)
                price = towertier.expected_tower_price(bundle.tier.prob(victim, taker, when / 1000.), prices)
            else:
                price = prices[name]
            out[:, 2 * kind + team] = price if team == 0 else -price
    if not np.isfinite(out).all():
        raise ValueError('nonfinite open value prices')
    return out


def _output(fa, enc, head, b):
    pos = b['pos']
    q, m, z = (b.get(k) for k in ('q', 'm', 'z'))
    if head == 'M2Rate':
        return fa.m2_rate(enc, b['e'], pos=pos)
    if head in ('M2Mark', 'M2Zone'):
        m2 = fa.m2(enc, 1.0, pos=pos)
        return m2.mark_logp if head == 'M2Mark' else m2.zone_logp
    if head in ('M3Size', 'M3Kills', 'M3Gold'):
        m3 = fa.m3(enc, q, m, z, pos=pos)
        return {'M3Size': m3.size_logp, 'M3Kills': m3.kill_rate, 'M3Gold': m3.gold_logp}[head]
    if head == 'M3Obj':
        return fa.m3_obj(enc, q, m, z, b['s'], b['K'], pos=pos)
    return fa.m4(enc, q, m, z, b['K'], b['O'], b['e'], pos=pos)


def _b1_predict(head, kind, fitted, parts, columns, idx):
    """Use the canonical training design including thousand-gold and zone coding."""
    p = parts
    if head == 'M4Conv' and kind == 'masked':
        X = B1._m4_design(p.m4[idx], len(columns))
    else:
        take = idx if p.take is None else p.take[idx]
        X = B1._assemble(B1.SLOTS[(head, kind)], p.feat, take,
                         {k: v[idx] for k, v in p.codes.items()},
                         None if p.K is None else p.K[idx], None if p.O is None else p.O[idx])
    if head == 'M3Gold':
        return fitted.predict(X, p.team[idx])
    return fitted.predict(X)


def _aggregate(match, row, scores):
    """HR-5: sum bins/teams within an evaluation row; never silently inner-join."""
    ids, first, inv = np.unique(row, return_index=True, return_inverse=True)
    if not np.array_equal(match, match[first][inv]):
        raise ValueError('one evaluation row has multiple matches')
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite NLL')
    return match[first], ids, np.bincount(inv, weights=scores, minlength=len(ids))


def _nll_record(head, kind, match, row, ept, b1, counts, scales):
    """Called only after paired target/condition/scored-row equality is proven."""
    key = loss.block_key(head, kind)
    record = dict(name=key, match=match, row=row, nll_ept=ept, nll_b1=b1, counts=counts,
                  common_head=scales is not None and key in STRUCTURAL_HEADS,
                  common_head_basis='G9b M2 whole layer plus M3/M4 open only; paired canonical targets/conditions/views/scored rows, HR-5 sums')
    if scales is not None:
        record.update(scale=float(scales[key]), nll_ept_normalized=ept / scales[key], nll_b1_normalized=b1 / scales[key])
    return record


def _empty_block(head, kind, reason, excluded, scales, nll_rows, endpoints):
    ids, values = np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    nll_rows.append(_nll_record(head, kind, ids, ids.copy(), values, values.copy(),
                               {'eligible': 0, 'unavailable_reason': reason,
                                'excluded_zero_exposure_cells_or_units': excluded}, scales))
    if head == 'M2Rate':
        zero = np.empty((0, 18))
        out = m2_endpoints(ids, ids, zero, zero, zero, zero)
        for ep in out:
            ep['counts'].update(candidate=excluded, zero_exposure=excluded, unavailable_reason=reason)
        endpoints.extend(out)
    elif (head, kind) == ('M4Conv', 'open'):
        zero = np.empty((0, 18))
        out = m4_endpoints(ids, ids, zero, values, zero, zero, excluded_no_window=excluded)
        for ep in out:
            ep['counts']['unavailable_reason'] = reason
        endpoints.extend(out)


def run_inference(bundle, model, stats, b1, *, check, device='cuda', batch_size=32,
                  draws=1000, seed=20261022, scales=None):
    """Return paired endpoint arrays and HR-5 per-row NLLs from DEV only.

    bundle: role-scoped s1/s2/labels stores, prices, tier (already loaded).
    check: mandatory zero-argument terminal/authorization/resource guard.
    scales: existing parent-run block -> positive B0 scale mapping, never fitted.
    NLLs remain raw; scale and normalized arrays are additional explicit fields.
    """
    if not callable(check):
        raise TypeError('check must be callable')
    check()  # before any store access
    for name, val, minimum in (('batch_size', batch_size, 1), ('draws', draws, 2), ('seed', seed, 0)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)) or val < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if getattr(bundle, 'role', 'DEV') != 'DEV':
        raise ValueError('only DEV is permitted')
    if tuple(b1.fit_roles) != ('FIT',) or tuple(b1.stop_roles) != ('STOP',):
        raise ValueError('B1 requires FIT training and STOP selection')
    from c_ept.comparison.models import supported_model
    if not isinstance(model, FrameAveraged) and not supported_model(model):
        raise TypeError('model must be a registered EPT/N2 or FrameAveraged model')
    fa = model if isinstance(model, FrameAveraged) else FrameAveraged(model)
    if not supported_model(fa.model):
        raise TypeError('FrameAveraged must wrap a registered EPT/N2 model')
    wanted_device = torch.device(device)
    if any(p.device.type != wanted_device.type or (wanted_device.index is not None and p.device.index != wanted_device.index)
           or p.dtype != torch.float32 for p in fa.parameters()):
        raise ValueError('caller must load the float32 model on the requested device')
    if scales is not None and (set(scales) != set(loss.BLOCKS) or any(
            isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in scales.values())):
        raise ValueError('scales must be the frozen positive thirteen-block mapping')
    for kind, names in (('masked', B1.HEAD_NAMES), ('open', B1.OPEN_HEAD_NAMES)):
        if set(getattr(b1, kind)) != set(names):
            raise ValueError(f'incomplete B1 {kind} heads')
    open_heads = B1OpenHeads(b1)
    e_rows = data.ept_head_rows(bundle.s1, bundle.s2, bundle.labels, 'DEV')
    b_rows = B1.HeadRows(bundle.s1, bundle.s2, bundle.labels, b1.dropped, 'DEV')
    # Prove legitimate empty populations from canonical masks, never by catching
    # a missing-field/view error. All nonempty blocks keep canonical validation.
    for field in ('_frame_in', '_onset_in', '_unit_in'):
        if not np.array_equal(getattr(e_rows, field), getattr(b_rows, field)):
            raise ValueError('paired DEV role masks disagree')
    has_frame = bool(e_rows._frame_in.any())
    has_onset = bool(e_rows._onset_in.any())
    has_open = bool((e_rows._unit_in & np.asarray(bundle.s1.array('units/in_scope'))).any())
    onset_unit = np.asarray(bundle.s1.array('onsets/unit')).astype(np.int64)
    reps = compose._gold_reps(bundle.labels.array('labels/gold_reps'))
    endpoints, nll_rows, m3_parts = [], [], []
    with _evaluation(fa):
        for head, kind in data.ROW_FIELDS:
            check()
            empty_reason = ('no_DEV_frame_rows' if head == 'M2Rate' and not has_frame else
                            'no_DEV_onsets' if kind == 'masked' and head != 'M2Rate' and not has_onset else
                            'no_DEV_in_scope_units' if kind == 'open' and not has_open else None)
            if empty_reason:
                _empty_block(head, kind, empty_reason, 0, scales, nll_rows, endpoints)
                continue
            ep = e_rows.parts(head, kind, masked_obj_k=True)
            bp = b_rows.parts(head, kind, masked_obj_k=True)
            for field in ('match', 'row', 'y', 'e', 'team', 'K', 'O'):
                a, b = getattr(ep, field), getattr(bp, field)
                if (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
                    raise ValueError(f'{head}/{kind}: paired {field} mismatch')
            if set(ep.codes) != set(bp.codes) or any(not np.array_equal(ep.codes[k], bp.codes[k]) for k in ep.codes):
                raise ValueError(f'{head}/{kind}: paired condition mismatch')
            views = e_rows.view_numbers(head, kind)
            if not np.array_equal(views, b_rows.view_numbers(head, kind)):
                raise ValueError(f'{head}/{kind}: paired view mismatch')
            fields, views, _ = data._block_rows(head, kind, ep, views, onset_unit)
            n = len(fields['row'])
            if not n:
                excluded = len(e_rows.frame().rows) if head == 'M2Rate' else int(ep.n_zero_exposure)
                _empty_block(head, kind, 'no_eligible_rows_under_canonical_head_mask', excluded, scales, nll_rows, endpoints)
                continue
            bpred, bscores = [], []
            fitted = getattr(b1, kind)[head]
            for lo in range(0, len(bp.row), batch_size):
                check()
                idx = np.arange(lo, min(lo + batch_size, len(bp.row)))
                p = _b1_predict(head, kind, fitted, bp, b1.columns, idx)
                if bp.e is not None:
                    e = bp.e[idx] if p.ndim == 1 else bp.e[idx, None]
                    score = nll_math.poisson(bp.y[idx], p, e)
                    if score.ndim > 1:
                        score = score.sum(-1)
                else:
                    score = nll_math.categorical(bp.y[idx], p)
                bpred.append(p)
                bscores.append(score)
            bpred, bscores = np.concatenate(bpred), np.concatenate(bscores)
            if head == 'M2Rate':
                brate = np.zeros((n, C.N_BINS))
                take = np.searchsorted(fields['row'], bp.row)
                brate[take, bp.codes['q']] = bpred
            else:
                brate = bpred
            scores, seen, output = [], [], []
            cell_values = open_cell_values(bundle, fields['match'], fields['row']) if (head, kind) == ('M3Size', 'open') else None
            for lo in range(0, n, batch_size):
                check()
                idx = np.arange(lo, min(lo + batch_size, n))
                unique, inv = np.unique(views[idx], return_inverse=True)
                vt = {k: v.to(wanted_device) for k, v in data.ViewTensors(bundle.s2, unique, stats).items()}
                enc = fa.encode(vt)
                b = {k: torch.as_tensor(v[idx], device=wanted_device) for k, v in fields.items()}
                b['pos'] = torch.as_tensor(inv, device=wanted_device)
                pred = _output(fa, enc, head, b)
                scores.append(_numpy(loss.head_nll(head, pred, b)))
                seen.append(_numpy(loss.scored_rows(head, b)))
                if head in ('M2Rate', 'M4Conv'):
                    output.append(_numpy(pred))  # these are expected counts, not rates
                if (head, kind) == ('M3Size', 'open'):
                    m3 = fa.m3(enc, None, None, None, pos=b['pos'])
                    e_laws = tuple(_numpy(t) for t in (m3.size_logp.exp(), m3.kill_rate, m3.gold_logp.exp()))
                    numeric = bp.feat[idx] if bp.take is None else bp.feat[bp.take[idx]]
                    b_laws = open_heads.laws(numeric, check=check)
                    for j, at in enumerate(idx):
                        def e_obj(s, K):
                            k = K.copy()
                            k[:, 2:] /= 1000.
                            return _numpy(fa.m3_obj(enc, None, None, None,
                                torch.as_tensor(s, device=wanted_device),
                                torch.as_tensor(k, dtype=torch.float32, device=wanted_device),
                                pos=b['pos'][j].expand(len(s))))
                        def b_obj(s, K):
                            return open_heads.objects(np.repeat(numeric[j:j + 1], len(s), axis=0), s, K)
                        common = dict(row=int(fields['row'][at]), match=int(fields['match'][at]), draws=draws,
                                      seed=seed, check=check, batch_size=batch_size)
                        pair = [sample_open_value(tuple(a[j] for a in laws), obj, cell_values[at], reps, **common)
                                for laws, obj in ((e_laws, e_obj), (b_laws, b_obj))]
                        m3_parts.append((common['match'], common['row'], *pair[0], *pair[1]))
            escores, keep = np.concatenate(scores), np.concatenate(seen)
            bkeep = bp.y >= 0 if bp.e is None else bp.e > 0
            em, er, en = _aggregate(fields['match'][keep], fields['row'][keep], escores[keep])
            bm, br, bn = _aggregate(bp.match[bkeep], bp.row[bkeep], bscores[bkeep])
            if not np.array_equal(em, bm) or not np.array_equal(er, br):
                raise ValueError(f'{head}/{kind}: paired scored rows mismatch')
            record = _nll_record(head, kind, em, er, en, bn,
                                 {'eligible': len(er), 'unscored_design_entries': int((~keep).sum()),
                                  'excluded_zero_exposure_cells_or_units': int(ep.n_zero_exposure)}, scales)
            nll_rows.append(record)
            if head == 'M2Rate':
                mu = np.concatenate(output)
                erate = np.divide(mu, fields['e'], out=np.zeros_like(mu), where=fields['e'] > 0)
                occurrence = m2_endpoints(fields['match'], fields['row'], fields['y'], fields['e'], erate, brate)
                excluded = len(e_rows.frame().rows) - n
                for item in occurrence:
                    item['counts']['candidate'] += excluded
                    item['counts']['zero_exposure'] += excluded
                endpoints.extend(occurrence)
            elif (head, kind) == ('M4Conv', 'open'):
                mu = np.concatenate(output)
                erate = np.divide(mu, fields['e'][:, None], out=np.zeros_like(mu), where=fields['e'][:, None] > 0)
                endpoints.extend(m4_endpoints(fields['match'], fields['row'], fields['y'], fields['e'], erate, brate,
                                             excluded_no_window=ep.n_zero_exposure))
    check()
    match = np.asarray([v[0] for v in m3_parts], dtype=np.int64)
    row = np.asarray([v[1] for v in m3_parts], dtype=np.int64)
    a = np.asarray([v[2:] for v in m3_parts], dtype=np.float64).reshape(-1, 8)
    truth = np.asarray(bundle.labels.array('labels/f0_unit'))[row] if len(row) else np.empty(0)
    if truth.shape != row.shape or not np.isfinite(truth).all():
        raise ValueError('missing/nonfinite M3 f0 truth')
    endpoints.insert(3, endpoint('M3/open/positive_value', 'M3', 'open', None, 'f0_gt0', match, row, truth > 0,
        a[:, 0], a[:, 4], draw_truth=truth == 0, p_draw_ept=a[:, 1], p_draw_b1=a[:, 5],
        mc_se_ept=a[:, 2], mc_se_b1=a[:, 6], mc_se_draw_ept=a[:, 3], mc_se_draw_b1=a[:, 7],
        counts={'candidate': len(row), 'eligible': len(row), 'draw': int((truth == 0).sum()),
                'unavailable_reason': None if len(row) else 'no_DEV_in_scope_units'}))
    return {'endpoints': endpoints, 'nll': nll_rows, 'provenance': {
        'adapter': 'EPT-T003/open-v1', 'role': 'DEV', 'frame_averaged': True, 'draws': int(draws), 'seed': int(seed),
        'rng': 'NumPy default_rng SeedSequence([seed,match,unit]); same initial streams, no strict Poisson CRN guarantee',
        'mc_uncertainty': 'Bernoulli sample SE; boundary zero SE does not prove exact probability',
        'm2_window': 'existing observed exposure clipped at occurrence boundary T-30s; no full-H filter',
        'm3_tower_value': 'canonical expected tower-tier price at first-kill time f, destroyed_before(f)',
        'm3_nll': 'conditional joint-law factors scored at observed size/K; never used for p+ prediction',
        'm4_condition': 'actual K/O at delta; existing exposure clipped at T',
        'nll_weighting': 'HR-5 bin/team sum per row, then endpoint match balancing; not training loss',
        'scales': None if scales is None else dict(scales), 'stats_sha256': stats.sha256,
        'structural_heads': list(STRUCTURAL_HEADS),
        'comparison': 'whole-FIT B1 versus FIT/STOP-25%-EPT pilot; no generalization or controlled architecture claim'}}
