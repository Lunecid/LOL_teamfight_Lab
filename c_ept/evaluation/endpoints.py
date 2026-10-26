"""EPT-T003 endpoint definitions; no IO, fitting, or population substitution (the primary population of a t0 forecast is chosen by the caller: ``primary_rows``, S4b preparation M12)."""
from collections.abc import Mapping

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


def primary_rows(t0_ms, hi_ms, H_s):
    """The primary population of a t0 forecast of horizon ``H_s`` seconds (60, 120 or 180): the rows whose observation window (t0, hi] holds the whole horizon, hi - t0 >= H x 1000.

    Whole milliseconds only: ``t0_ms`` and ``hi_ms`` are integer arrays of one length (TypeError for a float or a bool, ValueError for another shape or a horizon that is none of ``HORIZONS``).
    (S4b preparation M12, decision D14.)

    수식
    ----
    (PRI-1)  primary(H) = [ hi − t₀ ≥ H·1000 ms ],      hi = min(t₀ + 180 s, T − 30 s)   ⇔   t₀ + H ≤ T − 30 s     (H = 60, 120, 180 s)

      기호 | 뜻                                              | 단위 | 범위       | 코드 변수
      t₀   | 분 프레임의 기준 시각                           | ms   | ≥ 0        | t0_ms (frame_rows/t0)
      hi   | 발생 관측 창의 끝, min(t₀ + 180 s, T − 30 s)    | ms   | 정수       | hi_ms (frame_rows/hi)
      H    | 예측 지평                                       | s    | 60/120/180 | H_s

    해석
    ----
    주 모집단은 지평 H가 경기 끝 컷 T − 30 s 안에 온전히 드는 행이다. 이 행들에서는 앞 H/10개 발생 구간의 관측 노출이 모두 10 s이므로(구간 노출이 hi로 잘리지 않는다) 발생의 참값이 T로 잘리지 않고, 공칭 노출(10 s)과
    관측 노출이 같다. T는 이 선별에만 쓰이고 합성이나 모형에는 들어가지 않는다(결정 D14). 정수 ms로 견주는 것은 경계(hi − t₀가 H·1000과 같은 행)가 부동소수점 비교로 흔들리지 않게 하기 위함이다.
    """
    t0, hi = np.asarray(t0_ms), np.asarray(hi_ms)
    for name, a in (('t0_ms', t0), ('hi_ms', hi)):
        if a.dtype.kind not in 'iu':
            raise TypeError(f'{name} must be whole milliseconds (integers), not {a.dtype}')
    if t0.ndim != 1 or t0.shape != hi.shape:
        raise ValueError(f't0_ms and hi_ms must be one-dimensional arrays of one length, not {t0.shape} and {hi.shape}')
    if isinstance(H_s, (bool, np.bool_)) or not isinstance(H_s, (int, np.integer)) or int(H_s) not in HORIZONS:
        raise ValueError(f'H_s must be one of {HORIZONS} seconds, not {H_s!r}')
    return (hi.astype(np.int64) - t0.astype(np.int64)) >= int(H_s) * 1000


def _population(population, n):
    """``{H: bool mask (n,)}`` for the three horizons from ``population`` (None: no selection). ValueError, naming ``population``, for anything but a mapping from exactly the horizons to boolean masks of
    the n rows."""
    if population is None:
        return None
    if not isinstance(population, Mapping) or set(population) != set(HORIZONS):
        raise ValueError(f'population must map each of the horizons {HORIZONS} to a boolean mask of the rows (primary_rows), not {type(population).__name__}')
    out = {}
    for h in HORIZONS:
        mask = np.asarray(population[h])
        if mask.dtype != bool or mask.shape != (n,):
            raise ValueError(f'population[{h}] must be a boolean mask of the {n} rows, not {mask.dtype} {mask.shape}')
        out[h] = mask
    return out


def m2_endpoints(match, row, y, exposure, rate_ept, rate_b1, *, population=None):
    """Keep exposure-positive rows, including windows censored before H.

    ``population`` (S4b preparation M12, decision D14): None (the default) keeps every row, as before. Otherwise a mapping from each horizon (60, 120, 180) to a boolean mask of the rows, for example
    ``{H: primary_rows(t0, hi, H)}`` (the primary population) or its complement (the secondary population, scored on the same observed exposure for both comparators): only the rows of the mask are
    kept at that horizon, and the counts of the endpoint say how many rows were offered (``population_candidate``) and how many the mask selected (``population_selected``). Without a population the counts have
    no such keys.

    Counts: ``candidate`` and ``population_candidate`` are the rows offered to this function; a caller that leaves rows out before the call (the adapter, rows without any exposure) adds them to both, so the two
    keep one denominator (M12 review m2). ``population_selected`` counts the mask over the offered rows, ``eligible`` the selected rows with exposure. ``zero_exposure`` counts the offered rows without exposure in
    the array given as ``exposure``; with the nominal exposure (10 s in every bin, decision D14) that is 0 by construction. Rows without observed exposure are never primary, so the primary result does not
    depend on it (M12 review m3).

    수식 (M2 발생 확률): p = 1 − exp(−Σ_{q<H/10} λ_q·Δ_q); 주 모집단 행에서는 앞 H/10개 구간의 Δ_q = 10 s이다(결정 D14)."""
    y, exposure, rate_ept, rate_b1 = map(np.asarray, (y, exposure, rate_ept, rate_b1))
    if any(a.shape != (len(row), C.N_BINS) for a in (y, exposure, rate_ept, rate_b1)):
        raise ValueError('M2 requires aligned 18-bin arrays')
    if (y < 0).any() or not np.isfinite(y).all() or (y != np.floor(y)).any() or ((exposure == 0) & (y > 0)).any():
        raise ValueError('invalid M2 counts or positive count without exposure')
    selection = _population(population, len(row))
    # Validate before selecting: invalid exposures/rates must not disappear in a mask.
    occurrence_probability(rate_ept, exposure)
    occurrence_probability(rate_b1, exposure)
    out = []
    for h in HORIZONS:
        q = h * 1000 // C.BIN_MS
        e = exposure[:, :q]
        has_exposure = e.sum(1) > 0
        keep = has_exposure if selection is None else has_exposure & selection[h]
        probs = [occurrence_probability((r[:, :q] * e).sum(1), 1)[keep] for r in (rate_ept, rate_b1)]
        counts = {'candidate': len(row), 'eligible': int(keep.sum()),
                  'zero_exposure': int((~has_exposure).sum()),
                  'clipped_window': int(((e.sum(1) < h) & keep).sum())}
        if selection is not None:
            counts.update(population_candidate=len(row), population_selected=int(selection[h].sum()))
        out.append(endpoint(f'M2/t0/H{h}/occurrence', 'M2', 't0', h, 'first_kill_ge1',
                            np.asarray(match)[keep], np.asarray(row)[keep], (y[:, :q].sum(1) > 0)[keep], *probs, counts=counts))
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
