"""Pure, paired DEV diagnostics; no model, dataset, cohort or device imports.

Endpoints contain name/module/layer/horizon_s/target, one-dimensional match,
stable row keys (unique within each match), binary y, p_ept and p_b1. An explicit
horizon_s=None explicitly denotes a non-fixed-horizon target (e.g. M3 value at
engagement completion); target and adapter provenance define its observation
window. A missing horizon_s field is still invalid. Optional
draw_truth/p_draw_ept/p_draw_b1 must be supplied together. Optional mc_se_ept,
mc_se_b1, mc_se_draw_ept, mc_se_draw_b1 are row-level Monte Carlo standard errors.
NLL entries contain name/match/row/nll_ept/nll_b1 and optionally a positive fixed
scale. Only explicitly marked common_head=True entries enter the structural
normalized head mean; this is never an automatic sum of training loss blocks.
All inputs are already eligible paired rows. Selection/exclusion counts can be
supplied in counts; this module cannot infer exclusions from eligible rows.
"""
from __future__ import annotations

from collections.abc import Mapping
import numbers

import numpy as np


def _json(value):
    """Convert finite metadata to JSON types; never hide invalid NaN as null."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        if not np.isfinite(value):
            raise ValueError('nonfinite numeric value')
        return float(value)
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise ValueError('metadata keys must be strings')
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json(v) for v in value]
    raise ValueError(f'unsupported metadata type: {type(value).__name__}')


def _key(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, tuple):
        if not value:
            raise ValueError('empty composite key')
        return tuple(_key(v) for v in value)
    if isinstance(value, bool) or not isinstance(value, (str, numbers.Integral)):
        raise ValueError('keys must be nonempty strings, integers or tuples of these')
    if isinstance(value, str) and not value:
        raise ValueError('empty key')
    # Preserve integer/string identity, rather than silently casting everything.
    return value


def _rows(entry):
    for key in ('name', 'match', 'row'):
        if key not in entry:
            raise ValueError(f'missing {key}')
    if not isinstance(entry['name'], str) or not entry['name'].strip():
        raise ValueError('name must be a nonempty string')
    match, row = entry['match'], entry['row']
    if isinstance(match, (str, bytes)) or np.asarray(match).ndim != 1:
        raise ValueError('match must be one-dimensional')
    if isinstance(row, (str, bytes)) or (isinstance(row, np.ndarray) and row.ndim != 1):
        raise ValueError('row must be one-dimensional stable keys')
    if len(match) != len(row):
        raise ValueError('match/row shape mismatch')
    lookup, index, seen = {}, [], set()
    for m, r in zip(match, row):
        m, r = _key(m), _key(r)
        if (m, r) in seen:
            raise ValueError('duplicate (match, row) key')
        seen.add((m, r))
        if m not in lookup:
            lookup[m] = len(lookup)
        index.append(lookup[m])
    index = np.asarray(index, dtype=np.int64)
    sizes = np.bincount(index, minlength=len(lookup))
    weights = 1.0 / sizes[index] if len(index) else np.empty(0)
    return index, weights, list(lookup)


def _array(entry, key, size, *, binary=False, probability=False, nonnegative=False):
    if key not in entry:
        raise ValueError(f'missing {key}')
    raw = np.asarray(entry[key])
    if raw.ndim != 1 or len(raw) != size or raw.dtype.kind not in 'biuf':
        raise ValueError(f'{key}: expected numeric one-dimensional array of length {size}')
    values = raw.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f'{key}: nonfinite input')
    if binary and not np.isin(values, [0, 1]).all():
        raise ValueError(f'{key}: labels must be binary (bool accepted)')
    if probability and ((values < 0) | (values > 1)).any():
        raise ValueError(f'{key}: invalid probability')
    if nonnegative and (values < 0).any():
        raise ValueError(f'{key}: must be nonnegative')
    return values


class _Ranks:
    """Sort once per model; bootstrap recomputes weighted tie-group masses."""

    def __init__(self, y, score):
        self.order = np.argsort(score, kind='stable')
        ordered = score[self.order]
        self.start = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1] if len(y) else np.array([], dtype=int)
        self.y = y[self.order]
        self.threshold = ordered[self.start]

    def masses(self, weights):
        if not len(self.order):
            return np.empty(0), np.empty(0)
        w = weights[self.order]
        return (np.add.reduceat(w * self.y, self.start),
                np.add.reduceat(w * (1 - self.y), self.start))

    def auc(self, weights):
        positive, negative = self.masses(weights)
        p, n = positive.sum(), negative.sum()
        if p == 0 or n == 0:
            return None
        lower = np.cumsum(negative) - negative
        return float(np.sum(positive * (lower + .5 * negative)) / (p * n))

    def roc(self, weights):
        positive, negative = self.masses(weights)
        if positive.sum() == 0 or negative.sum() == 0:
            return {'status': 'unavailable', 'reason': 'no_rows' if not len(weights) else 'single_class',
                    'fpr': None, 'tpr': None, 'thresholds': None}
        return {'status': 'available', 'reason': None,
                'fpr': [0.0] + (np.cumsum(negative[::-1]) / negative.sum()).tolist(),
                'tpr': [0.0] + (np.cumsum(positive[::-1]) / positive.sum()).tolist(),
                # null is the origin sentinel (score strictly above the maximum).
                'thresholds': [None] + self.threshold[::-1].tolist()}


def _mean(values, weights):
    if not len(weights) or weights.sum() == 0:
        return None
    # Normalize first to avoid overflow in otherwise representable means.
    result = float(np.dot(values, weights / weights.sum()))
    if not np.isfinite(result):
        raise ValueError('nonfinite metric arithmetic')
    return result


def _ci(values):
    return np.quantile(values, [.025, .975]).tolist() if values else None


def _comparison(point, samples, requested, *, reason=None):
    """The three columns always use exactly the same valid paired replicates."""
    valid = len(samples)
    invalid = requested - valid
    result = {'valid_replicates': valid, 'invalid_replicates': invalid,
              'ci_status': ('not_requested' if requested == 0 else
                            'unavailable_invalid_replicates' if invalid else 'available'),
              'reason': reason}
    columns = list(zip(*samples)) if samples else [(), ()]
    delta = None if point[0] is None or point[1] is None else point[0] - point[1]
    for label, value, draws in zip(('ept', 'b1', 'delta'), (*point, delta),
                                   (*columns, [a - b for a, b in samples])):
        result[label] = {'estimate': value, 'ci95': _ci(draws) if invalid == 0 else None,
                         'conditional_ci95': _ci(draws) if invalid else None}
    return result


def _guard(check):
    if check is not None:
        if check() is False:
            raise RuntimeError('evaluation guard refused continuation')


def _bootstrap(index, weights, n_matches, evaluate, bootstrap, rng, check):
    samples = {}
    for b in range(bootstrap):
        if b % 25 == 0:
            _guard(check)
        # Every occurrence of a sampled match retains its original 1/n_m rows.
        multiplicity = np.bincount(rng.integers(n_matches, size=n_matches), minlength=n_matches) if n_matches else np.empty(0)
        values = evaluate(weights * multiplicity[index])
        for name, pair in values.items():
            samples.setdefault(name, [])
            if all(value is not None for value in pair):
                if not np.isfinite(pair).all():
                    raise ValueError('nonfinite bootstrap arithmetic')
                samples[name].append(pair)
    _guard(check)
    return samples


def _calibration(y, p, w):
    # Division agrees with decimal input boundaries such as float(.3); linspace
    # can produce .30000000000000004 and misclassify a prediction of exactly .3.
    edges = np.arange(11, dtype=np.float64) / 10
    bins = np.minimum(np.searchsorted(edges, p, side='right') - 1, 9)
    result = []
    for i in range(10):
        mask = bins == i
        result.append({'left': float(edges[i]), 'right': float(edges[i + 1]),
                       'right_closed': i == 9, 'n_rows': int(mask.sum()),
                       'weight_sum': float(w[mask].sum()),
                       'mean_prediction': _mean(p[mask], w[mask]),
                       'observed_fraction': _mean(y[mask], w[mask])})
    return result


def _counts(entry, index, weights, matches, y=None):
    result = {'n_rows': len(index), 'n_matches': len(matches), 'weight_sum': float(weights.sum()),
              'selection': _json(entry.get('counts'))}
    if y is not None:
        result.update(n_positive=int(y.sum()), n_negative=int(len(y) - y.sum()),
                      positive_weight=float(weights[y == 1].sum()),
                      negative_weight=float(weights[y == 0].sum()))
    return result


def _endpoint(entry, bootstrap, rng, check):
    index, weights, matches = _rows(entry)
    for field in ('module', 'layer', 'target'):
        if not isinstance(entry.get(field), str) or not entry[field].strip():
            raise ValueError(f'missing or empty {field}')
    if 'horizon_s' not in entry:
        raise ValueError('missing horizon_s (use explicit None for non-fixed observation windows)')
    horizon = entry['horizon_s']
    if horizon is not None and (isinstance(horizon, bool) or not isinstance(horizon, numbers.Real)
                                or not np.isfinite(horizon) or horizon <= 0):
        raise ValueError('horizon_s must be None or positive and finite')
    y = _array(entry, 'y', len(index), binary=True)
    probabilities = [_array(entry, f'p_{m}', len(index), probability=True) for m in ('ept', 'b1')]
    ranks = [_Ranks(y, p) for p in probabilities]
    losses = [(p - y) ** 2 for p in probabilities]
    draw_keys = ('draw_truth', 'p_draw_ept', 'p_draw_b1')
    draw = None
    if any(k in entry for k in draw_keys):
        truth = _array(entry, 'draw_truth', len(index), binary=True)
        draw_p = [_array(entry, f'p_draw_{m}', len(index), probability=True) for m in ('ept', 'b1')]
        if ((truth == 1) & (y == 1)).any():
            raise ValueError('draw truth overlaps positive truth')
        if any((p + d > 1 + 1e-12).any() for p, d in zip(probabilities, draw_p)):
            raise ValueError('positive plus draw probability exceeds one')
        draw = {'n_rows': int(truth.sum()), 'weighted_fraction': _mean(truth, weights),
                'mean_probability': {m: _mean(p, weights) for m, p in zip(('ept', 'b1'), draw_p)}}
    mc = {}
    for field in ('mc_se_ept', 'mc_se_b1', 'mc_se_draw_ept', 'mc_se_draw_b1'):
        if field in entry:
            se = _array(entry, field, len(index), nonnegative=True)
            mc[field] = {'weighted_mean': _mean(se, weights),
                         'max': float(se.max()) if len(se) else None,
                         'n_rows': len(se)}

    def evaluate(w):
        result = {'auc': tuple(rank.auc(w) for rank in ranks),
                  'brier': tuple(_mean(loss, w) for loss in losses)}
        if draw is not None:
            result['draw_brier'] = tuple(_mean((p - truth) ** 2, w) for p in draw_p)
        return result

    point = evaluate(weights)
    samples = _bootstrap(index, weights, len(matches), evaluate, bootstrap, rng, check)
    reason = 'no_rows' if not len(index) else 'single_class' if len(np.unique(y)) < 2 else None
    return {**{k: _json(entry[k]) for k in ('name', 'module', 'layer', 'horizon_s', 'target')},
            'counts': _counts(entry, index, weights, matches, y),
            'metrics': {k: _comparison(v, samples.get(k, []), bootstrap,
                                      reason=reason if k == 'auc' else 'no_rows' if not len(index) else None)
                        for k, v in point.items()},
            'roc': {m: rank.roc(weights) for m, rank in zip(('ept', 'b1'), ranks)},
            'reliability': {m: _calibration(y, p, weights) for m, p in zip(('ept', 'b1'), probabilities)},
            'draw': draw, 'mc_uncertainty': mc or None}


def _nll(entry, bootstrap, rng, check):
    index, weights, matches = _rows(entry)
    values = [_array(entry, f'nll_{m}', len(index)) for m in ('ept', 'b1')]
    scale = entry.get('scale')
    if scale is not None and (isinstance(scale, bool) or not isinstance(scale, numbers.Real) or not np.isfinite(scale) or scale <= 0):
        raise ValueError('NLL scale must be positive finite and fixed before evaluation')
    if not isinstance(entry.get('common_head', False), bool):
        raise ValueError('common_head must be boolean')
    if entry.get('common_head') and scale is None:
        raise ValueError('structural common head requires a fixed scale')

    def evaluate(w):
        pair = tuple(_mean(value, w) for value in values)
        result = {'nll': pair}
        if scale is not None:
            result['normalized_nll'] = tuple(None if v is None else v / scale for v in pair)
        return result

    point = evaluate(weights)
    samples = _bootstrap(index, weights, len(matches), evaluate, bootstrap, rng, check)
    return {'name': entry['name'], 'scale': scale, 'common_head': entry.get('common_head', False),
            'counts': _counts(entry, index, weights, matches),
            'metrics': {k: _comparison(v, samples.get(k, []), bootstrap,
                                      reason='no_rows' if not len(index) else None) for k, v in point.items()}}


def _structural(entries, bootstrap, rng, check):
    """Equal mean of explicitly named, fixed-scale common heads, jointly paired."""
    selected = [e for e in entries if e.get('common_head', False)]
    if not selected:
        return {'status': 'unavailable', 'reason': 'no_explicit_common_heads', 'heads': [], 'metrics': None}
    universe = {}
    prepared = []
    for entry in selected:
        index, weights, matches = _rows(entry)
        for m in matches:
            if m not in universe:
                universe[m] = len(universe)
        mapping = np.array([universe[m] for m in matches], dtype=np.int64)
        values = [_array(entry, f'nll_{m}', len(index)) / entry['scale'] for m in ('ept', 'b1')]
        prepared.append((mapping[index], weights, values))

    def evaluate(multiplicity):
        means = [tuple(_mean(v, w * multiplicity[i]) for v in vals) for i, w, vals in prepared]
        return tuple(None if any(p[j] is None for p in means) else float(np.mean([p[j] for p in means])) for j in (0, 1))

    size = len(universe)
    point = evaluate(np.ones(size))
    draws = []
    for b in range(bootstrap):
        if b % 25 == 0:
            _guard(check)
        mult = np.bincount(rng.integers(size, size=size), minlength=size) if size else np.empty(0)
        pair = evaluate(mult)
        if all(v is not None for v in pair):
            draws.append(pair)
    _guard(check)
    return {'status': 'available' if all(p is not None for p in point) else 'unavailable',
            'reason': None if all(p is not None for p in point) else 'empty_common_head',
            'heads': [e['name'] for e in selected], 'n_union_matches': size,
            'definition': 'equal mean of named common-head match-balanced NLL / fixed scale',
            'bootstrap_population': 'union of common-head eligible matches; shared multiplicities',
            'metrics': _comparison(point, draws, bootstrap, reason='empty_common_head' if any(p is None for p in point) else None)}


def summarize_endpoints(endpoints, nll=(), *, bootstrap=2000, seed=20261022, check=None):
    """Return JSON-safe summaries without emitting identifiers or predictions.

    check() is invoked before work, between endpoints, every 25 bootstrap
    replicates and after each loop. Exceptions (or literal False) abort the job.
    bootstrap=0 explicitly disables intervals. Single-class/empty endpoints are
    valid unavailable outcomes, whereas malformed inputs always raise ValueError.
    """
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, numbers.Integral) or bootstrap < 0:
        raise ValueError('bootstrap must be a nonnegative integer')
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    _guard(check)
    endpoints, nll = list(endpoints), list(nll)
    for collection in (endpoints, nll):
        names = [e.get('name') for e in collection]
        if any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
            raise ValueError('missing or duplicate endpoint/head name')
    rng = np.random.default_rng(seed)
    result = {'schema_version': 'ept-dev-metrics-v1',
              'bootstrap': {'replicates': int(bootstrap), 'seed': int(seed), 'confidence': .95,
                            'method': 'paired match percentile', 'delta': 'EPT-B1'},
              'weighting': '1 / eligible rows in each match, separately per endpoint',
              'auc_definition': 'pooled weighted positive-negative pairs; ties count half; not within-match AUC',
              'comparison': 'EPT FIT/STOP 25% pilot versus existing whole-FIT B1; unequal training scope',
              'limitations': ['Single training seed; CI covers match sampling, not training-seed variation.',
                             'M3/M4 are conditional layers, not whole t0 or composed-value performance.',
                             'No aggregate AUC and no all-training-block NLL sum.'],
              'endpoints': [], 'nll': []}
    for entry in endpoints:
        _guard(check)
        result['endpoints'].append(_endpoint(entry, bootstrap, rng, check))
    for entry in nll:
        _guard(check)
        result['nll'].append(_nll(entry, bootstrap, rng, check))
    result['structural_normalized_nll'] = _structural(nll, bootstrap, rng, check)
    _guard(check)
    return _json(result)
