"""Matched predictor inputs for T008; no store is opened by this module.

Public API:
  select_membership(s1, share=.25, selection_seed=20261022) -> Membership
  fit_stats_subset(s1, s2, membership, check) -> ept.data.Stats
  build_neural_data(bundle, membership, *, scales, batch_size=32,
                    train_seed=20261022, check, stats=None)
      -> (train.TrainData, ept.data.Stats, JSON-ready receipt)
  fit_b1_subset(s1, s2, labels, membership, *, check, params=None) -> B1Heads

Prepare and persist Membership/Stats once. Neural workers may pass loaded Stats;
then no statistics are refitted. The caller must verify the saved Stats hash and
its membership binding against the preparation receipt before supplying it.
B1 is always freshly fitted through the existing 13-head fit implementation.
Its ``comparison_receipt`` and ``membership_sha256`` attributes are deliberately
external provenance: legacy B1 serialization remains unchanged. Persist/verify
the receipt beside saved B1 heads. Original whole-FIT B0 scales are supplied by
the caller, without refitting or changing either loss or evaluation scales.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from c_ept import runlog
from c_ept.ept import data as neural
from c_ept.preprocessing import features


@dataclass(frozen=True)
class Membership:
    """Immutable global S1 match indices and a signed selection recipe.

    ``signature`` hashes the authoritative role table, never match keys. The
    ``sha256`` content digest covers both selections, signature, share and seed.
    Neither signature is an authentication mechanism; ``verify_against`` also
    reproduces the deterministic selection against the supplied S1 role table.
    """
    fit: tuple[int, ...]
    stop: tuple[int, ...]
    selection_seed: int
    share: float
    signature: str

    def __post_init__(self):
        if isinstance(self.selection_seed, bool) or not isinstance(self.selection_seed, Integral) or self.selection_seed < 0:
            raise ValueError('selection_seed must be a nonnegative integer')
        if isinstance(self.share, bool) or not isinstance(self.share, Real) or not 0 < self.share <= 1:
            raise ValueError('share must be in (0, 1]')
        for name in ('fit', 'stop'):
            values = tuple(getattr(self, name))
            if not values or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in values):
                raise ValueError(f'{name} must contain nonnegative integer match indices')
            values = tuple(int(v) for v in values)
            if values != tuple(sorted(set(values))):
                raise ValueError(f'{name} must be sorted and distinct')
            object.__setattr__(self, name, values)
        if set(self.fit) & set(self.stop):
            raise ValueError('FIT and STOP selections must be disjoint')
        if not isinstance(self.signature, str) or len(self.signature) != 64 or any(c not in '0123456789abcdef' for c in self.signature):
            raise ValueError('signature must be a lowercase SHA256 role-table digest')
        object.__setattr__(self, 'selection_seed', int(self.selection_seed))
        object.__setattr__(self, 'share', float(self.share))

    @property
    def fit_indices(self):
        return self.fit

    @property
    def stop_indices(self):
        return self.stop

    def _body(self):
        return {'schema': 'matched_membership_v1', 'signature': self.signature,
                'selection_seed': self.selection_seed, 'share': self.share,
                'fit': list(self.fit), 'stop': list(self.stop)}

    @property
    def sha256(self):
        return runlog.sha256_json(self._body())

    def to_record(self):
        return {**self._body(), 'sha256': self.sha256}

    @classmethod
    def from_record(cls, record):
        if not isinstance(record, dict) or set(record) != {'schema', 'signature', 'selection_seed', 'share', 'fit', 'stop', 'sha256'}:
            raise ValueError('invalid membership record fields')
        if record['schema'] != 'matched_membership_v1':
            raise ValueError('unsupported membership schema')
        value = cls(record['fit'], record['stop'], record['selection_seed'], record['share'], record['signature'])
        if value.sha256 != record['sha256']:
            raise ValueError('membership content does not match its SHA256')
        return value

    def verify_against(self, s1):
        """Refuse another role table, wrong roles, or a re-signed forged selection."""
        expected = select_membership(s1, self.share, self.selection_seed)
        if self != expected:
            raise ValueError('membership differs from the authoritative roles or deterministic selection')
        return self


def select_membership(s1, share=.25, selection_seed=20261022):
    """Reuse proxy_matches exactly: sorted per-role permutation prefixes/ceil counts.

    Only matches/role metadata is read. Selection has no training-seed input.
    """
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, Integral) or selection_seed < 0:
        raise ValueError('selection_seed must be a nonnegative integer')
    selected = neural.proxy_matches(s1, share=share, seed=int(selection_seed))
    roles = np.asarray(s1.array('matches/role'))
    if roles.ndim != 1 or roles.dtype.kind not in 'iu':
        raise ValueError('matches/role must be a one-dimensional integer table')
    signature = runlog.sha256_json({'roles': roles.tolist()})
    return Membership(tuple(selected['FIT']), tuple(selected['STOP']), selection_seed, share, signature)


def _tick(check):
    if not callable(check):
        raise TypeError('check must be a callable execution guard')
    check()


def _fit_views(s1, s2, membership):
    # split_views does the authoritative role/range validation first.
    role = features.split_views(s1, s2, 'FIT')
    return np.flatnonzero(role & np.isin(np.asarray(s2.array('views/match')), membership.fit))


def fit_stats_subset(s1, s2, membership, check):
    """Fit Stats on selected FIT views only; no STOP/DEV predictor value is read."""
    _tick(check)
    membership.verify_against(s1)
    return neural.fit_stats(s1, s2, matches=membership.fit, check=check)


def build_neural_data(bundle, membership, *, scales, batch_size=32,
                      train_seed=20261022, check, stats=None):
    """Return (TrainData, Stats, receipt) using selected FIT/STOP global rows.

    ``bundle`` supplies s1/s2/labels/static and optionally static_source or
    meta['terrain_files_sha256']. TrainData.fit.rows is selected FIT, so the
    existing trainer's fit_biases/head priors see only the shared selection.
    A supplied Stats is reused by identity and never fitted again. Its saved
    hash/membership provenance is the preparation caller's verification duty.
    """
    _tick(check)
    membership.verify_against(bundle.s1)
    fitted_here = stats is None
    if fitted_here:
        stats = fit_stats_subset(bundle.s1, bundle.s2, membership, check)
    if not isinstance(stats, neural.Stats) or stats.role != 'FIT':
        raise ValueError('stats must be FIT Stats from the matched preparation')
    n_views = len(_fit_views(bundle.s1, bundle.s2, membership))
    if n_views == 0 or stats.n_views != n_views:
        raise ValueError('Stats view count differs from selected FIT')
    grouped = {}
    for role, selected in (('FIT', membership.fit), ('STOP', membership.stop)):
        _tick(check)
        rows = neural.ept_head_rows(bundle.s1, bundle.s2, bundle.labels, role, matches=selected)
        grouped[role] = neural.EngagementRows(rows)
    batches = {role: neural.Batches(rows, batch_size, train_seed, stats=stats, training=role == 'FIT')
               for role, rows in grouped.items()}
    from c_ept.ept.train import TrainData
    source = getattr(bundle, 'static_source', None)
    if source is None and 'terrain_files_sha256' in getattr(bundle, 'meta', {}):
        source = {'terrain_files_sha256': bundle.meta['terrain_files_sha256']}
    data = TrainData(batches['FIT'], batches['STOP'], scales, bundle.static, static_source=source)
    receipt = {'schema': 'matched_neural_data_v1', 'membership_sha256': membership.sha256,
               'membership': membership.to_record(), 'stats_sha256': stats.sha256,
               'stats_fitted_here': fitted_here, 'predictor_fit_scope': 'selected FIT',
               'head_init_scope': 'selected FIT', 'train_seed': int(train_seed),
               'rows': {role: rows.counts() for role, rows in grouped.items()},
               'loss_scales_scope': 'original shared S3 B0; supplied unchanged',
               'auxiliary_scope': 'frozen targets and input assets may use declared whole FIT'}
    _tick(check)
    return data, stats, receipt


def fit_b1_subset(s1, s2, labels, membership, *, check, params=None):
    """Fresh original B1 heads on selected FIT, stopped on selected STOP.

    params=None retains P9 seed=1/native threads=8/patience=50/max rounds=1000.
    params is the original B1 synthetic-test override, never a cache selector.
    Selection identity is available on the returned object and external receipt;
    callers must persist that receipt alongside legacy B1 model files.
    """
    _tick(check)
    membership.verify_against(s1)
    from c_ept.models import baselines_b1 as b1
    dropped = features.dropped_columns(s2, _fit_views(s1, s2, membership))
    _tick(check)
    fit_rows = b1.HeadRows(s1, s2, labels, dropped, 'FIT', matches=membership.fit)
    stop_rows = b1.HeadRows(s1, s2, labels, dropped, 'STOP', matches=membership.stop)
    fitted = {kind: {} for kind in b1.KINDS}
    for kind, names in (('masked', b1.HEAD_NAMES), ('open', b1.OPEN_HEAD_NAMES)):
        for name in names:
            _tick(check)
            fitted[kind][name] = b1.fit_head(fit_rows, stop_rows, name, kind, params=params, check=check)
    result = b1.B1Heads(fitted['masked'], fitted['open'], dropped)
    result.membership_sha256 = membership.sha256
    result.comparison_receipt = {'schema': 'matched_b1_fit_v1', 'membership_sha256': membership.sha256,
                                 'membership': membership.to_record(), 'predictor_fit_scope': 'selected FIT',
                                 'stop_scope': 'selected STOP', 'dropped': list(dropped),
                                 'params': dict(b1.FIXED_PARAMS), 'max_rounds': b1.MAX_ROUNDS,
                                 'early_stop': b1.EARLY_STOP, 'params_override': None if params is None else dict(params)}
    _tick(check)
    return result
