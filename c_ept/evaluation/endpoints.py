"""EPT-T003 endpoint definitions; no IO, fitting, or population substitution."""
import numpy as np

from c_ept import config as C

HORIZONS = (60, 120, 180)
CONVERSION_TARGETS = tuple((kind, team, 2 * C.O_KINDS.index(kind) + t)
                           for kind in ('dragon', 'baron', 'tower')
                           for t, team in enumerate(('blue', 'red')))


def occurrence_probability(rate, exposure):
    """Per-cell observed-window probability; caller sums hazards for M2."""
    rate, exposure = np.broadcast_arrays(np.asarray(rate, dtype=float), np.asarray(exposure, dtype=float))
    if not np.isfinite(exposure).all() or (exposure < 0).any():
        raise ValueError('exposure must be finite and nonnegative')
    if not np.isfinite(rate).all() or (rate < 0).any():
        raise ValueError('rate must be finite and nonnegative')
    return -np.expm1(-rate * exposure)


def endpoint(name, module, layer, horizon_s, target, match, row, y, p_ept, p_b1, **extra):
    arrays = {k: np.asarray(v) for k, v in dict(match=match, row=row, y=y, p_ept=p_ept, p_b1=p_b1).items()}
    n = len(arrays['row'])
    if any(v.shape != (n,) for v in arrays.values()):
        raise ValueError('endpoint arrays must be aligned one-dimensional arrays')
    if any(arrays[k].dtype.kind not in 'iu' for k in ('match', 'row')):
        raise ValueError('match and row must be integer identifiers')
    if len(np.unique(arrays['row'])) != n:
        raise ValueError('duplicate endpoint row')
    if not np.isin(arrays['y'], (0, 1)).all():
        raise ValueError('endpoint truth must be binary')
    for k in ('p_ept', 'p_b1'):
        if not np.isfinite(arrays[k]).all() or ((arrays[k] < 0) | (arrays[k] > 1)).any():
            raise ValueError('endpoint probabilities must be finite in [0,1]')
    counts = dict(extra.pop('counts', {}))
    counts.update(positive=int(np.count_nonzero(arrays['y'])), negative=int(np.count_nonzero(arrays['y'] == 0)),
                  matches=len(np.unique(arrays['match'])), match_weight_sum=len(np.unique(arrays['match'])))
    return dict(name=name, module=module, layer=layer, horizon_s=horizon_s, target=target, counts=counts,
                **arrays, **extra)


def m2_endpoints(match, row, y, exposure, rate_ept, rate_b1):
    """Keep exposure-positive rows, including windows censored before H."""
    y, exposure, rate_ept, rate_b1 = map(np.asarray, (y, exposure, rate_ept, rate_b1))
    if any(a.shape != (len(row), C.N_BINS) for a in (y, exposure, rate_ept, rate_b1)):
        raise ValueError('M2 requires aligned 18-bin arrays')
    if (y < 0).any() or not np.isfinite(y).all() or (y != np.floor(y)).any() or ((exposure == 0) & (y > 0)).any():
        raise ValueError('invalid M2 counts or positive count without exposure')
    # Validate before selecting: invalid exposures/rates must not disappear in a mask.
    occurrence_probability(rate_ept, exposure)
    occurrence_probability(rate_b1, exposure)
    out = []
    for h in HORIZONS:
        q = h * 1000 // C.BIN_MS
        e = exposure[:, :q]
        keep = e.sum(1) > 0
        probs = [occurrence_probability((r[:, :q] * e).sum(1), 1)[keep] for r in (rate_ept, rate_b1)]
        out.append(endpoint(f'M2/t0/H{h}/occurrence', 'M2', 't0', h, 'first_kill_ge1',
                            np.asarray(match)[keep], np.asarray(row)[keep], (y[:, :q].sum(1) > 0)[keep], *probs,
                            counts={'candidate': len(row), 'eligible': int(keep.sum()),
                                    'zero_exposure': int((~keep).sum()),
                                    'clipped_window': int(((e.sum(1) < h) & keep).sum())}))
    return out


def m4_endpoints(match, row, y, exposure, rate_ept, rate_b1, *, excluded_no_window=0):
    y, exposure = np.asarray(y), np.asarray(exposure)
    if y.shape != (len(row), 2 * len(C.O_KINDS)) or exposure.shape != (len(row),):
        raise ValueError('invalid M4 row/cell shape')
    if not np.isfinite(y).all() or (y < 0).any() or (y != np.floor(y)).any() or ((exposure[:, None] == 0) & (y > 0)).any():
        raise ValueError('invalid M4 counts or positive count without exposure')
    probs = [occurrence_probability(r, exposure[:, None]) for r in (rate_ept, rate_b1)]
    if any(p.shape != y.shape for p in probs):
        raise ValueError('M4 rates must match count cells')
    keep = exposure > 0
    return [endpoint(f'M4/open/{kind}/{team}', 'M4', 'open', None, f'{kind}_{team}_ge1',
                     np.asarray(match)[keep], np.asarray(row)[keep], (y[:, cell] > 0)[keep],
                     probs[0][keep, cell], probs[1][keep, cell],
                     counts={'candidate': len(row) + excluded_no_window, 'eligible': int(keep.sum()),
                             'zero_exposure': int((~keep).sum()) + excluded_no_window})
            for kind, team, cell in CONVERSION_TARGETS]
