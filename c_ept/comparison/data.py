"""Matched predictor inputs for T008; no store is opened by this module.

Public API:
  select_membership(s1, share=.25, selection_seed=20261022) -> Membership
  fit_stats_subset(s1, s2, membership, check, *, column_drop='none',
                   stats_population='whole_fit') -> ept.data.Stats
  build_neural_data(bundle, membership, *, scales, batch_size=32,
                    train_seed=20261022, check, stats=None, kill_law='poisson',
                    k_condition='raw', column_drop='none',
                    stats_population='whole_fit')
      -> (train.TrainData, ept.data.Stats, JSON-ready receipt)
  fit_b1_subset(s1, s2, labels, membership, *, check, params=None,
                kill_law='poisson', k_condition='raw', column_drop='none') -> B1Heads

The S4b settings (kill_law, k_condition: S4b preparation M1, M3; column_drop,
stats_population: M5, decisions D7, D12) default to the calls, rows, statistics
and receipts of every earlier run; a receipt names a setting only when it is not
the default. With a column drop the D7 list is judged over the FIT views of the
whole store, the statistics over the selected FIT, and B1 drops the list that the
EPT of the same run drops (swap pairs judged together).

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
from c_ept.data import head_rows
from c_ept.ept import data as neural
from c_ept.models import killlaw
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


def _check_settings(kill_law, k_condition, column_drop, stats_population=None):
    """ValueError for a name that is not a kill law, a K condition, a column drop or a population (no store is read).

    해석: S4b 프로파일의 설정 이름을 자료를 읽기 전에 막는다. 기본값('poisson', 'raw', 'none', 'whole_fit')이 이전 실행의 호출이다.
    """
    killlaw.check_law(kill_law)
    head_rows.check_k_condition(k_condition)
    if column_drop not in neural.COLUMN_DROPS:
        raise ValueError(f'column_drop must be one of {neural.COLUMN_DROPS}, not {column_drop!r}')
    if stats_population is not None and stats_population not in neural.POPULATIONS:
        raise ValueError(f'stats_population must be one of {neural.POPULATIONS}, not {stats_population!r}')


def _fit_store_views(s1, s2):
    """The views of every FIT match of the store: the population of the D7 list (decision D12: the list is a whole-FIT asset of every run)."""
    return np.flatnonzero(features.split_views(s1, s2, 'FIT'))


def fit_stats_subset(s1, s2, membership, check, *, column_drop='none', stats_population='whole_fit'):
    """Fit Stats on selected FIT views only; no STOP/DEV predictor value is read.

    ``column_drop`` and ``stats_population`` (S4b preparation M5, decisions D7 and D12) go to ``ept.data.fit_stats``; the defaults are the call and the statistics of
    every earlier run (the selected matches, no record of the population, no dropped column). For a column drop the D7 list is judged over the views of every FIT match
    of the store (``store_constant_columns``), which are FIT views too, then the statistics are those of the selected FIT.
    """
    _tick(check)
    _check_settings('poisson', 'raw', column_drop, stats_population)
    membership.verify_against(s1)
    if column_drop == 'none' and stats_population == 'whole_fit':
        return neural.fit_stats(s1, s2, matches=membership.fit, check=check)
    store = ()
    if column_drop != 'none':
        _tick(check)
        store = neural.store_constant_columns(s2, _fit_store_views(s1, s2))
    return neural.fit_stats(s1, s2, matches=membership.fit, check=check, column_drop=column_drop, stats_population=stats_population, store_constant=store)


def build_neural_data(bundle, membership, *, scales, batch_size=32,
                      train_seed=20261022, check, stats=None, kill_law='poisson', k_condition='raw',
                      column_drop='none', stats_population='whole_fit', swap_mode='average'):
    """Return (TrainData, Stats, receipt) using selected FIT/STOP global rows.

    ``bundle`` supplies s1/s2/labels/static and optionally static_source or
    meta['terrain_files_sha256']. TrainData.fit.rows is selected FIT, so the
    existing trainer's fit_biases/head priors see only the shared selection.
    A supplied Stats is reused by identity and never fitted again. Its saved
    hash/membership provenance is the preparation caller's verification duty.

    ``kill_law``, ``k_condition`` (S4b preparation M1, M3; M3 review m6) go to
    ``ept_head_rows`` for the FIT and the STOP rows, ``column_drop`` and
    ``stats_population`` (M5, D7, D12) to ``fit_stats_subset``; the defaults are
    the call, the rows and the receipt of every earlier run. A supplied Stats
    made under other column_drop / population settings is refused. The receipt
    names a setting only when it is not the default (absent means the default).

    ``swap_mode`` (S4b preparation M7 review follow-up, decision D10 P1) is the
    frame of the training batches (``ept.data.SWAP_MODES``); it goes to the FIT
    batches only (the STOP batches are never swapped at random), and ``train``
    refuses FIT batches made for another mode than its settings. The receipt
    names it only when it is not 'average'.
    """
    _tick(check)
    _check_settings(kill_law, k_condition, column_drop, stats_population)
    neural.check_swap_mode(swap_mode)
    membership.verify_against(bundle.s1)
    fitted_here = stats is None
    if fitted_here:
        stats = fit_stats_subset(bundle.s1, bundle.s2, membership, check, column_drop=column_drop, stats_population=stats_population)
    if not isinstance(stats, neural.Stats) or stats.role != 'FIT':
        raise ValueError('stats must be FIT Stats from the matched preparation')
    if stats.column_drop != column_drop or stats.population != stats_population:
        raise ValueError(f"stats were made with column_drop {stats.column_drop!r} and population {stats.population!r} but column_drop {column_drop!r} and "
                         f"stats_population {stats_population!r} are asked for")
    n_views = len(_fit_views(bundle.s1, bundle.s2, membership))
    if n_views == 0 or stats.n_views != n_views:
        raise ValueError('Stats view count differs from selected FIT')
    law = {} if kill_law == 'poisson' else {'kill_law': kill_law}
    condition = {} if k_condition == 'raw' else {'k_condition': k_condition}      # the defaults are not passed: the calls are those of every earlier run
    grouped = {}
    for role, selected in (('FIT', membership.fit), ('STOP', membership.stop)):
        _tick(check)
        rows = neural.ept_head_rows(bundle.s1, bundle.s2, bundle.labels, role, matches=selected, **law, **condition)
        grouped[role] = neural.EngagementRows(rows)
    swap = {} if swap_mode == 'average' else {'swap_mode': swap_mode}              # the default is not passed: the call is that of every earlier run
    batches = {role: neural.Batches(rows, batch_size, train_seed, stats=stats, training=role == 'FIT', **(swap if role == 'FIT' else {}))
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
    # a setting is named in the receipt only when it is not the default: the receipt of an earlier run has none of these keys
    if kill_law != 'poisson':
        receipt['kill_law'] = kill_law
    if k_condition != 'raw':
        receipt['k_condition'] = k_condition
        receipt['k_reps_sha256'] = grouped['FIT'].head_rows.k_reps_sha256()
    if column_drop != 'none':
        receipt.update(column_drop=column_drop, dropped_store=list(stats.dropped_store), dropped=list(stats.dropped),
                       column_drop_scope=neural.COLUMN_DROP_SCOPE)
    if stats_population != 'whole_fit':
        receipt['stats_population'] = stats_population
    if swap_mode != 'average':
        receipt['swap_mode'] = swap_mode
    _tick(check)
    return data, stats, receipt


def fit_b1_subset(s1, s2, labels, membership, *, check, params=None, kill_law='poisson', k_condition='raw', column_drop='none'):
    """Fresh original B1 heads on selected FIT, stopped on selected STOP.

    params=None retains P9 seed=1/native threads=8/patience=50/max rounds=1000.
    params is the original B1 synthetic-test override, never a cache selector.
    Selection identity is available on the returned object and external receipt;
    callers must persist that receipt alongside legacy B1 model files.

    ``kill_law`` and ``k_condition`` (S4b preparation M1, M3; M3 review m6) are
    those of ``B1Heads.fit``: the rows, the heads and the set carry them (and the
    sha256 of the representatives of the binned K block); the defaults are the
    fit of every earlier run. ``column_drop`` (M5, decision D7) names the columns
    left out of b1: 'none' (default) the columns constant, one at a time, in the
    selected FIT (every earlier run); 'store' the D7 list over the whole FIT
    store; 'store_and_subset' that list and the columns constant inside the
    selected FIT, a swap pair judged on its pooled values: the list that
    ``ept.data.fit_stats`` gives the EPT of the same run. The receipt names these
    settings only when they are not the defaults.
    """
    _tick(check)
    _check_settings(kill_law, k_condition, column_drop)
    membership.verify_against(s1)
    from c_ept.models import baselines_b1 as b1
    store = ()
    if column_drop == 'none':
        dropped = features.dropped_columns(s2, _fit_views(s1, s2, membership))
    else:
        store = neural.store_constant_columns(s2, _fit_store_views(s1, s2))
        dropped = store if column_drop == 'store' else features.dropped_columns(s2, _fit_views(s1, s2, membership), pooled=True)
        if neural.AGE_COLUMN in dropped or not set(store) <= set(dropped):
            raise ValueError('the list of the selected FIT must hold the D7 list and never frame_age_s: the views of the list are not those of the store')
    _tick(check)
    law = {} if kill_law == 'poisson' else {'kill_law': kill_law}
    condition = {} if k_condition == 'raw' else {'k_condition': k_condition}      # the defaults are not passed: the calls are those of every earlier run
    fit_rows = b1.HeadRows(s1, s2, labels, dropped, 'FIT', matches=membership.fit, **law, **condition)
    stop_rows = b1.HeadRows(s1, s2, labels, dropped, 'STOP', matches=membership.stop, **law, **condition)
    fitted = {kind: {} for kind in b1.KINDS}
    for kind, names in (('masked', b1.HEAD_NAMES), ('open', b1.OPEN_HEAD_NAMES)):
        for name in names:
            _tick(check)
            fitted[kind][name] = b1.fit_head(fit_rows, stop_rows, name, kind, params=params, check=check)
    recorded = {} if k_condition == 'raw' else {'k_reps_sha256': fit_rows.k_reps_sha256()}       # the 31 numbers of the binned K block
    result = b1.B1Heads(fitted['masked'], fitted['open'], dropped, **law, **condition, **recorded)
    result.membership_sha256 = membership.sha256
    result.comparison_receipt = {'schema': 'matched_b1_fit_v1', 'membership_sha256': membership.sha256,
                                 'membership': membership.to_record(), 'predictor_fit_scope': 'selected FIT',
                                 'stop_scope': 'selected STOP', 'dropped': list(dropped),
                                 'params': dict(b1.FIXED_PARAMS), 'max_rounds': b1.MAX_ROUNDS,
                                 'early_stop': b1.EARLY_STOP, 'params_override': None if params is None else dict(params)}
    if kill_law != 'poisson':
        result.comparison_receipt['kill_law'] = kill_law
    if k_condition != 'raw':
        result.comparison_receipt.update(k_condition=k_condition, **recorded)
    if column_drop != 'none':
        result.comparison_receipt.update(column_drop=column_drop, dropped_store=list(store), column_drop_scope=neural.COLUMN_DROP_SCOPE)
    _tick(check)
    return result
