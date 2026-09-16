"""P2 (exploratory) temporal win probability: causal one-minute histories and guarded inference.

p_t approximates P(final Blue win | observations available at t). Nothing here is a causal model.

* Candidate A: the frozen P1 snapshot adapter (independent_wp_v2_participant_order, expanded family),
  applied to the state at the query position only, i.e. f(X_t, t).
* Candidate B: a small SimpleRNN over nine one-minute history positions t-8min, ..., t
  (architecture adapted from Silva, Pappa, Chaimowicz, SBGames 2018; not a reproduction).

Every history position is an objective_history_v2_participant_order state built by the frozen
gameplay.state_value_v2.StateBuilder from frames and events whose timestamps are at or before that
position. Positions before the first observed frame are zero-padded and masked: the recurrence
carries the previous hidden state through them (initial state zeros).

BatchSource (retrospective replay over a stored match pack) and ObservationLog (chronological
appends) feed the same assemble_history(). Their agreement is a replay property; it does not
validate availability or latency of a live public API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from gameplay.state_value import SNAPSHOT_FIELDS
from gameplay.state_value_v2 import STATE_VERSION, StateBuilder, StateV2
from train.independent_winprob_v2 import (CHAMPION_COLUMNS, MODEL_VERSION as P1_MODEL_VERSION,
                                          IndependentWinProbabilityV2, check_v2_names, load_adapter)

MODULE_VERSION = 'temporal_winprob_v3_p2'
HISTORY_VERSION = 'causal_history_v3_step60000ms_x9'
STEP_MS = 60_000
N_POSITIONS = 9
CANDIDATE_A = 'A_frozen_p1_expanded_v2'
CANDIDATE_B = 'B_simplernn_history'
CANDIDATE_B_VERSION = 'simplernn_history_v3_p2'
EXCLUDED_PREDICTORS = ('snapshot_age_s',)
PREDICTOR_EVENT_TYPES = ('CHAMPION_KILL', 'ELITE_MONSTER_KILL', 'DRAGON_SOUL_GIVEN', 'BUILDING_KILL',
                         'TURRET_PLATE_DESTROYED')
CALIBRATIONS = ('raw', 'sigmoid', 'isotonic')
ARCHITECTURE = {
    'cell': 'SimpleRNN (Elman): h_k = tanh(W x_k + U h_{k-1} + b)', 'units': 8, 'input_dropout': 0.25,
    'dropout_mask': 'one Bernoulli mask per sequence over input units, shared across positions (training only)',
    'output': 'dense(1) + sigmoid on the final hidden state', 'positions': N_POSITIONS, 'step_ms': STEP_MS,
    'masking': 'masked (padded) positions carry the previous hidden state; initial state zeros',
    'numeric_inputs': 'all StateV2 features except snapshot_age_s and champion IDs, standardised with FIT-only '
                      'mean/std (std 0 -> scale 1)',
    'champion_inputs': 'participant_slotN_champion_id one-hot per slot over FIT-only category lists; '
                       'equivalent to a per-slot embedding column of the input kernel; unseen IDs contribute zero',
    'initialisation': 'kernel glorot-uniform over one-hot-expanded input width, recurrent orthogonal, '
                      'biases zero, output glorot-uniform'}


class SchemaError(ValueError):
    pass


class ModelVersionError(ValueError):
    pass


class ChronologyError(ValueError):
    pass


class CrossMatchError(ValueError):
    pass


class FutureObservationError(ValueError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def _ms(value):
    """Integer millisecond timestamps only; non-integral values are rejected, never rounded."""
    if isinstance(value, (bool, np.bool_)):
        raise SchemaError('timestamp must be integer milliseconds')
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return int(value)
    raise SchemaError(f'timestamp must be integer milliseconds, got {value!r}')


def history_times(query_ms):
    q = _ms(query_ms)
    return tuple(q - (N_POSITIONS - 1 - k) * STEP_MS for k in range(N_POSITIONS))


def _check_state(state, names, position_ms):
    if not isinstance(state, StateV2) or state.state_version != STATE_VERSION:
        raise SchemaError('history positions must be StateV2 objects of the declared state version')
    if list(state.values) != list(names):
        raise SchemaError('state feature schema mismatch')
    if int(state.query_ms) != int(position_ms):
        raise SchemaError('state query time differs from its history position')
    if int(state.snapshot_ms) > int(position_ms):
        raise FutureObservationError('state snapshot after its history position')


# ----------------------------------------------------------------------------- histories
@dataclass(frozen=True)
class CausalHistory:
    match_id: str
    query_ms: int
    support_start_ms: int
    states: tuple
    state_names: tuple
    source_kind: str
    history_version: str = HISTORY_VERSION

    def __post_init__(self):
        if self.history_version != HISTORY_VERSION:
            raise ModelVersionError('history version mismatch')
        if len(self.states) != N_POSITIONS:
            raise SchemaError('history must have exactly nine positions')
        if int(self.query_ms) < int(self.support_start_ms):
            raise ValueError('query precedes observed source support')
        names = check_v2_names(self.state_names)
        for h, s in zip(history_times(self.query_ms), self.states):
            if h < self.support_start_ms:
                if s is not None:
                    raise SchemaError('state supplied for a position before source support')
            else:
                if s is None:
                    raise SchemaError('missing state at a supported history position')
                _check_state(s, names, h)

    @property
    def times(self):
        return history_times(self.query_ms)

    @property
    def mask(self):
        return tuple(s is not None for s in self.states)

    @property
    def snapshot_ms(self):
        return tuple(int(s.snapshot_ms) if s is not None else -1 for s in self.states)

    def matrix(self):
        X = np.zeros((N_POSITIONS, len(self.state_names)), dtype=float)
        for k, s in enumerate(self.states):
            if s is not None:
                X[k] = np.asarray([s.values[n] for n in self.state_names], dtype=float)
        return X

    @classmethod
    def from_observed_states(cls, match_id, query_ms, support_start_ms, observed, state_names):
        """Build a history from chronological (match_id, StateV2) observations; no interpolation."""
        mid, q, support = str(match_id), _ms(query_ms), _ms(support_start_ms)
        names = check_v2_names(state_names)
        by_time, last = {}, None
        for item in observed:
            if not (isinstance(item, tuple) and len(item) == 2):
                raise SchemaError('observed states must be (match_id, StateV2) pairs')
            omid, state = item
            if str(omid) != mid:
                raise CrossMatchError('observed state belongs to another match')
            if not isinstance(state, StateV2) or state.state_version != STATE_VERSION:
                raise SchemaError('observed states must be StateV2 objects of the declared state version')
            t = _ms(state.query_ms)
            if last is not None and t <= last:
                raise ChronologyError('observed states are not strictly chronological')
            if t > q:
                raise FutureObservationError('observed state after the query time')
            if list(state.values) != names:
                raise SchemaError('state feature schema mismatch')
            by_time[t], last = state, t
        states = []
        for h in history_times(q):
            if h < support:
                states.append(None)
            elif h not in by_time:
                raise ValueError(f'missing observed state at required history position {h} (no interpolation)')
            else:
                states.append(by_time[h])
        return cls(mid, q, support, tuple(states), tuple(names), 'observed_states')


def assemble_history(source, query_ms, state_names):
    """The single causal history assembly shared by training, evaluation and on-demand inference."""
    names = check_v2_names(state_names)
    q = _ms(query_ms)
    source.check_query(q)
    support = int(source.support_start_ms)
    if q < support:
        raise ValueError('query precedes observed source support')
    states = []
    for h in history_times(q):
        if h < support:
            states.append(None)
            continue
        s = source.state_at(h)
        _check_state(s, names, h)
        states.append(s)
    return CausalHistory(str(source.match_id), q, support, tuple(states), tuple(names), source.kind)


class BatchSource:
    """Retrospective replay over a stored pack; the frozen V2 builder filters to each position."""
    kind = 'retrospective_batch'

    def __init__(self, match_id, pack, node_names):
        self.match_id = str(match_id)
        pack_mid = (pack.get('meta') or {}).get('match_id')
        if pack_mid is not None and str(pack_mid) != self.match_id:
            raise CrossMatchError('pack belongs to another match')
        ts = np.asarray(pack['minute_ts'], dtype=np.int64)
        if not len(ts):
            raise SchemaError('no frames')
        self.support_start_ms = max(0, int(ts[0]))
        self.last_frame_ms = int(ts[-1])
        self._builder = StateBuilder(pack, list(node_names))
        self.participant_order = self._builder.participant_order

    def check_query(self, q):
        if q > self.last_frame_ms:
            raise ValueError('query after the last stored frame')

    def state_at(self, position_ms):
        return self._builder.at(position_ms)


class ObservationLog:
    """Chronological per-match observation log (frames and events appended in timestamp order).

    The roster/team map is required at construction. Appends must not go back in time, must carry
    this match ID and the declared node schema; a participant's champion may not change. A query is
    refused while the log holds any observation after the query time, or at/after an observed GAME_END.
    """
    kind = 'incremental_log'

    def __init__(self, match_id, team_map, node_names):
        self.match_id = str(match_id)
        tm = {int(k): int(v) for k, v in dict(team_map).items()}
        if set(tm) != set(range(1, 11)) or any(list(tm.values()).count(t) != 5 for t in (100, 200)):
            raise SchemaError('invalid participant roster/team map')
        self.team_map = tm
        self.node_names = [str(n) for n in node_names]
        if len(set(self.node_names)) != len(self.node_names) or not (set(SNAPSHOT_FIELDS) | {'champion_id'}) <= set(self.node_names):
            raise SchemaError('unsupported node schema')
        self._required = [self.node_names.index(n) for n in (*SNAPSHOT_FIELDS, 'champion_id')]
        self._champion = self.node_names.index('champion_id')
        self._frame_ts, self._frames, self._events = [], [], []
        self._watermark = None
        self._terminal_ms = None
        self._champions = None
        self._dtype = None

    @property
    def observed_until_ms(self):
        return self._watermark

    def _check(self, match_id, ts):
        if str(match_id) != self.match_id:
            raise CrossMatchError('observation belongs to another match')
        if ts < 0:
            raise ChronologyError('negative timestamp')
        if self._watermark is not None and ts < self._watermark:
            raise ChronologyError('observation older than an already appended observation')

    def add_frame(self, match_id, timestamp_ms, node_row, node_names):
        ts = _ms(timestamp_ms)
        self._check(match_id, ts)
        if [str(n) for n in node_names] != self.node_names:
            raise SchemaError('frame node schema differs from the log schema')
        if self._frame_ts and ts <= self._frame_ts[-1]:
            raise ChronologyError('duplicate or out-of-order frame timestamp')
        row = np.array(node_row, copy=True)
        if row.shape != (10, len(self.node_names)):
            raise SchemaError('frame must have shape (10, n_node_features)')
        if self._dtype is not None and row.dtype != self._dtype:
            raise SchemaError('frame dtype changed')
        if not np.isfinite(row[:, self._required].astype(float)).all():
            raise SchemaError('non-finite required node fields')
        champions = row[:, self._champion].copy()
        if self._champions is not None and not np.array_equal(champions, self._champions):
            raise SchemaError('participant champion changed between frames')
        self._champions, self._dtype = champions, row.dtype
        self._frame_ts.append(ts)
        self._frames.append(row)
        self._watermark = ts

    def add_event(self, match_id, event):
        if not isinstance(event, dict) or 'timestamp' not in event:
            raise SchemaError('event must be a dict with a timestamp')
        ts = _ms(event['timestamp'])
        self._check(match_id, ts)
        self._events.append(dict(event))
        self._watermark = ts
        if event.get('type') == 'GAME_END' and self._terminal_ms is None:
            self._terminal_ms = ts

    def source(self, query_ms):
        q = _ms(query_ms)
        if not self._frame_ts:
            raise ValueError('no observed frame yet')
        if self._watermark is not None and q < self._watermark:
            raise FutureObservationError('observation log contains data after the query time')
        if self._terminal_ms is not None and q >= self._terminal_ms:
            raise ValueError('query at or after an observed GAME_END (terminal, outside nonterminal support)')
        return _LogView(self, q)


class _LogView:
    """Frozen view of an ObservationLog for one query.

    The frozen legacy builder rejects any query after its last frame (a retrospective bound). The view
    appends a NaN bound-sentinel frame at query+1 ms. It is never read: a position <= query selects the
    latest real frame at or before it, and reading NaN would fail the builder's finiteness check.
    """
    kind = ObservationLog.kind

    def __init__(self, log, q):
        self.match_id, self.query_ms = log.match_id, q
        self.support_start_ms = max(0, int(log._frame_ts[0]))
        sentinel = np.full_like(log._frames[-1], np.nan)
        pack = {'minute_ts': np.asarray([*log._frame_ts, q + 1], dtype=np.int64),
                'node_minute': np.stack([*log._frames, sentinel]),
                'meta': {'match_id': log.match_id, 'team_map': {str(p): t for p, t in log.team_map.items()}},
                'events': list(log._events)}
        self._builder = StateBuilder(pack, log.node_names)

    def check_query(self, q):
        if q != self.query_ms:
            raise ValueError('log view was opened for a different query time')

    def state_at(self, position_ms):
        if _ms(position_ms) > self.query_ms:
            raise FutureObservationError('history position after the query time')
        return self._builder.at(position_ms)


def check_sequence_membership(table_match, table_time, seq_index, seq_mask, seq_times, query_match, query_time):
    """Violations of: history index belongs to the query's match, exact position time, masks, last position valid.

    Tables are per partition, so an index into the query's partition table also proves partition membership.
    """
    seq_index, seq_mask, seq_times = np.asarray(seq_index), np.asarray(seq_mask, dtype=bool), np.asarray(seq_times)
    expected_times = np.asarray(query_time, dtype=np.int64)[:, None] - (N_POSITIONS - 1 - np.arange(N_POSITIONS)) * STEP_MS
    out = {'rows': int(len(seq_index)),
           'position_times_wrong': int((seq_times != expected_times).any(axis=1).sum()),
           'masked_with_index': int(((~seq_mask) & (seq_index >= 0)).sum()),
           'valid_without_index': int((seq_mask & (seq_index < 0)).sum()),
           'last_position_masked': int((~seq_mask[:, -1]).sum())}
    ok = seq_mask & (seq_index >= 0)
    rows, cols = np.nonzero(ok)
    idx = seq_index[rows, cols]
    out['cross_match_positions'] = int((np.asarray(table_match)[idx] != np.asarray(query_match)[rows]).sum())
    out['time_mismatch_positions'] = int((np.asarray(table_time)[idx] != seq_times[rows, cols]).sum())
    out['future_positions'] = int((seq_times[rows, cols] > np.asarray(query_time)[rows]).sum())
    out['violations'] = int(sum(v for k, v in out.items() if k != 'rows'))
    return out


# ----------------------------------------------------------------------------- predictions and deltas
@dataclass(frozen=True)
class Prediction:
    match_id: str
    query_ms: int
    p: float
    candidate: str
    candidate_version: str
    model_sha256: str
    calibration: str
    history_version: str
    state_version: str
    snapshot_ms: int
    history_times: tuple
    history_mask: tuple
    history_snapshot_ms: tuple
    source_kind: str
    components: dict = field(default_factory=dict)

    @property
    def snapshot_age_ms(self):
        return int(self.query_ms - self.snapshot_ms)

    def model_key(self):
        return (self.candidate, self.candidate_version, self.model_sha256, self.calibration,
                self.history_version, self.state_version)

    def as_dict(self):
        return {'match_id': self.match_id, 'query_ms': self.query_ms, 'p': self.p, 'snapshot_ms': self.snapshot_ms,
                'snapshot_age_ms': self.snapshot_age_ms, 'history_times': list(self.history_times),
                'history_mask': list(self.history_mask), 'history_snapshot_ms': list(self.history_snapshot_ms),
                'source_kind': self.source_kind, 'components': dict(self.components),
                'model': dict(zip(('candidate', 'candidate_version', 'model_sha256', 'calibration',
                                   'history_version', 'state_version'), self.model_key()))}


def endpoint_delta(pre, post):
    """p_post - p_pre from the SAME frozen model/calibration; a model mismatch fails."""
    if not isinstance(pre, Prediction) or not isinstance(post, Prediction):
        raise TypeError('endpoint_delta requires Prediction objects')
    if pre.model_key() != post.model_key():
        raise ModelVersionError('endpoints were scored by different frozen models, versions or calibrations')
    if pre.match_id != post.match_id:
        raise CrossMatchError('endpoints belong to different matches')
    if not pre.query_ms < post.query_ms:
        raise ChronologyError('pre endpoint must precede post endpoint')
    return {'match_id': pre.match_id, 'pre_query_ms': pre.query_ms, 'post_query_ms': post.query_ms,
            'p_pre': pre.p, 'p_post': post.p, 'delta': post.p - pre.p,
            'pre_snapshot_ms': pre.snapshot_ms, 'post_snapshot_ms': post.snapshot_ms,
            'pre_snapshot_age_ms': pre.snapshot_age_ms, 'post_snapshot_age_ms': post.snapshot_age_ms,
            'model': dict(zip(('candidate', 'candidate_version', 'model_sha256', 'calibration',
                               'history_version', 'state_version'), pre.model_key())),
            'semantics': 'difference between two frozen-model probabilities of final Blue win; not a causal effect'}


# ----------------------------------------------------------------------------- candidate A
class CandidateA:
    name = CANDIDATE_A

    def __init__(self, adapter, model_sha256):
        if not isinstance(adapter, IndependentWinProbabilityV2):
            raise TypeError('candidate A must wrap the P1 IndependentWinProbabilityV2 adapter')
        if adapter.model_version != P1_MODEL_VERSION or adapter.state_version != STATE_VERSION:
            raise ModelVersionError('candidate A adapter version mismatch')
        if adapter.family != 'expanded':
            raise ModelVersionError('candidate A must be the expanded family')
        self.adapter = adapter
        self.model_sha256 = str(model_sha256)
        self.candidate_version = P1_MODEL_VERSION
        self.calibration = f'{adapter.family}_{adapter.calibration}'
        self.state_names = tuple(adapter.state_names)

    @classmethod
    def load(cls, path, expected_sha256):
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise ModelVersionError('candidate A model file hash mismatch')
        return cls(load_adapter(path), actual)

    def predict_last_matrix(self, X_last, names, state_version):
        return self.adapter.predict_matrix(X_last, names, state_version)

    def predict_histories(self, histories):
        states = [h.states[-1] for h in histories]
        return self.adapter.predict_states(states), {}


# ----------------------------------------------------------------------------- candidate B
class HistoryPreprocessor:
    def __init__(self, state_names, numeric_mean, numeric_scale, champion_vocab):
        self.state_names = tuple(check_v2_names(state_names))
        names = list(self.state_names)
        self.numeric_names = tuple(n for n in names if n not in EXCLUDED_PREDICTORS and n not in CHAMPION_COLUMNS)
        self.numeric_index = np.asarray([names.index(n) for n in self.numeric_names], dtype=np.int64)
        self.champion_index = np.asarray([names.index(c) for c in CHAMPION_COLUMNS], dtype=np.int64)
        self.numeric_mean = np.asarray(numeric_mean, dtype=np.float64)
        self.numeric_scale = np.asarray(numeric_scale, dtype=np.float64)
        self.champion_vocab = [np.asarray(v, dtype=np.int64) for v in champion_vocab]
        if self.numeric_mean.shape != (len(self.numeric_names),) or self.numeric_scale.shape != self.numeric_mean.shape:
            raise SchemaError('preprocessor numeric shape mismatch')
        if not (np.isfinite(self.numeric_mean).all() and np.isfinite(self.numeric_scale).all() and (self.numeric_scale > 0).all()):
            raise SchemaError('invalid preprocessor scale')
        if len(self.champion_vocab) != len(CHAMPION_COLUMNS) or any(np.any(np.diff(v) <= 0) for v in self.champion_vocab):
            raise SchemaError('champion vocabularies must be strictly increasing, one per slot')
        sizes = [len(v) for v in self.champion_vocab]
        self.champion_offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        self.n_numeric = len(self.numeric_names)
        self.n_champion_columns = int(sum(sizes))
        self.input_width = self.n_numeric + self.n_champion_columns

    @classmethod
    def fit(cls, state_names, X_fit):
        names = check_v2_names(state_names)
        X_fit = np.asarray(X_fit, dtype=np.float64)
        if X_fit.ndim != 2 or X_fit.shape[1] != len(names) or not np.isfinite(X_fit).all() or not len(X_fit):
            raise SchemaError('FIT states must be a finite (n, n_state_names) matrix')
        num = [i for i, n in enumerate(names) if n not in EXCLUDED_PREDICTORS and n not in CHAMPION_COLUMNS]
        mean = X_fit[:, num].mean(axis=0)
        std = X_fit[:, num].std(axis=0)
        scale = np.where(std > 0, std, 1.)
        vocab = []
        for c in CHAMPION_COLUMNS:
            ids = X_fit[:, names.index(c)]
            if not np.array_equal(ids, np.round(ids)):
                raise SchemaError('non-integral champion ID')
            vocab.append(np.unique(ids.astype(np.int64)))
        return cls(names, mean, scale, vocab)

    def encode_states(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.state_names):
            raise SchemaError('state matrix shape mismatch')
        if not np.isfinite(X).all():
            raise SchemaError('non-finite state values')
        Z = (X[:, self.numeric_index] - self.numeric_mean) / self.numeric_scale
        ids = X[:, self.champion_index]
        if not np.array_equal(ids, np.round(ids)):
            raise SchemaError('non-integral champion ID')
        ids = ids.astype(np.int64)
        C = np.full(ids.shape, -1, dtype=np.int64)
        for j, vocab in enumerate(self.champion_vocab):
            pos = np.searchsorted(vocab, ids[:, j])
            found = (pos < len(vocab)) & (vocab[np.minimum(pos, len(vocab) - 1)] == ids[:, j])
            C[found, j] = self.champion_offsets[j] + pos[found]
        return Z, C

    def encode_histories(self, H, mask):
        H = np.asarray(H, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if H.ndim != 3 or H.shape[1:] != (N_POSITIONS, len(self.state_names)) or mask.shape != H.shape[:2]:
            raise SchemaError('history array shape mismatch')
        if (H[~mask] != 0).any():
            raise SchemaError('padded positions must be zero')
        N = len(H)
        Xnum = np.zeros((N, N_POSITIONS, self.n_numeric), dtype=np.float64)
        C = np.full((N, N_POSITIONS, len(CHAMPION_COLUMNS)), -1, dtype=np.int64)
        rows, cols = np.nonzero(mask)
        if len(rows):
            Z, Ci = self.encode_states(H[rows, cols])
            Xnum[rows, cols], C[rows, cols] = Z, Ci
        return Xnum, C, mask

    def arrays(self):
        out = {'state_names': np.asarray(self.state_names), 'numeric_names': np.asarray(self.numeric_names),
               'numeric_mean': self.numeric_mean, 'numeric_scale': self.numeric_scale}
        out.update({f'champion_vocab_{j}': v for j, v in enumerate(self.champion_vocab)})
        return out

    @classmethod
    def from_arrays(cls, z):
        pre = cls(z['state_names'].tolist(), z['numeric_mean'], z['numeric_scale'],
                  [z[f'champion_vocab_{j}'] for j in range(len(CHAMPION_COLUMNS))])
        if list(pre.numeric_names) != z['numeric_names'].tolist():
            raise SchemaError('stored numeric feature order differs')
        return pre


PARAM_NAMES = ('kernel', 'recurrent_kernel', 'bias', 'output_kernel', 'output_bias')


def simplernn_forward(params, Xnum, C, mask):
    """Canonical float64 numpy inference; returns (probability, logit). No dropout."""
    kernel = np.asarray(params['kernel'], dtype=np.float64)
    U = np.asarray(params['recurrent_kernel'], dtype=np.float64)
    b = np.asarray(params['bias'], dtype=np.float64)
    wo = np.asarray(params['output_kernel'], dtype=np.float64)
    bo = float(np.asarray(params['output_bias'], dtype=np.float64))
    N, T, D = Xnum.shape
    units = kernel.shape[0]
    if kernel.shape[1] < D or U.shape != (units, units) or b.shape != (units,) or wo.shape != (units,):
        raise SchemaError('SimpleRNN parameter shapes mismatch')
    Wn, Wc = kernel[:, :D], kernel[:, D:].T
    h = np.zeros((N, units), dtype=np.float64)
    for k in range(T):
        a = Xnum[:, k, :] @ Wn.T
        a += h @ U.T
        a += b
        cc = C[:, k, :]
        valid = cc >= 0
        a += (Wc[np.where(valid, cc, 0)] * valid[..., None]).sum(axis=1)
        h = np.where(mask[:, k, None], np.tanh(a), h)
    logit = h @ wo + bo
    return expit(logit), logit


def _logit(p):
    # Identical clipping to train.independent_winprob_v2._logit (P1 calibrators).
    return np.log(np.clip(p, 1e-10, 1 - 1e-10) / np.clip(1 - p, 1e-10, 1))


def apply_calibration(spec, p):
    p = np.asarray(p, dtype=np.float64)
    method = spec['method']
    if method == 'raw':
        return p
    if method == 'sigmoid':
        return expit(_logit(p) * float(spec['sigmoid_coef']) + float(spec['sigmoid_intercept']))
    if method == 'isotonic':
        x, y = np.asarray(spec['isotonic_x'], dtype=np.float64), np.asarray(spec['isotonic_y'], dtype=np.float64)
        if len(x) == 1:
            return np.repeat(y[0], len(p))
        return np.interp(np.clip(p, x[0], x[-1]), x, y)
    raise ModelVersionError(f'unknown calibration {method!r}')


def fit_calibrations(p_cal, y_cal):
    """raw / sigmoid (Platt on logit, C=1e6) / isotonic (clip), unweighted on CALIBRATE, as P1."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    p_cal, y_cal = np.asarray(p_cal, dtype=np.float64), np.asarray(y_cal).astype(int)
    sig = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000).fit(_logit(p_cal).reshape(-1, 1), y_cal)
    iso = IsotonicRegression(out_of_bounds='clip').fit(p_cal, y_cal)
    specs = {'raw': {'method': 'raw'},
             'sigmoid': {'method': 'sigmoid', 'sigmoid_coef': float(sig.coef_[0, 0]), 'sigmoid_intercept': float(sig.intercept_[0])},
             'isotonic': {'method': 'isotonic', 'isotonic_x': np.asarray(iso.X_thresholds_, dtype=np.float64),
                          'isotonic_y': np.asarray(iso.y_thresholds_, dtype=np.float64)}}
    check = {'sigmoid': float(np.max(np.abs(apply_calibration(specs['sigmoid'], p_cal) - sig.predict_proba(_logit(p_cal).reshape(-1, 1))[:, 1]))),
             'isotonic': float(np.max(np.abs(apply_calibration(specs['isotonic'], p_cal) - iso.predict(p_cal))))}
    return specs, {'sigmoid': sig, 'isotonic': iso}, check


class CandidateB:
    name = CANDIDATE_B

    def __init__(self, preprocessor, members, bundle_sha256=None):
        if not isinstance(preprocessor, HistoryPreprocessor):
            raise TypeError('preprocessor must be a HistoryPreprocessor')
        if not members:
            raise SchemaError('candidate B needs at least one member')
        self.preprocessor = preprocessor
        self.members = []
        for m in members:
            params = {k: np.asarray(m['params'][k], dtype=np.float64) for k in PARAM_NAMES}
            if params['kernel'].shape[1] != preprocessor.input_width:
                raise SchemaError('member kernel width differs from the preprocessor input width')
            if m['calibration']['method'] not in CALIBRATIONS:
                raise ModelVersionError('unknown calibration')
            self.members.append({'seed': int(m['seed']), 'params': params, 'calibration': dict(m['calibration'])})
        self.state_names = preprocessor.state_names
        self.candidate_version = CANDIDATE_B_VERSION
        self.calibration = 'per_seed:' + ','.join(f"{m['seed']}={m['calibration']['method']}" for m in self.members)
        self.model_sha256 = bundle_sha256 or 'unsaved'

    def predict_encoded(self, Xnum, C, mask):
        members, raw = {}, {}
        for m in self.members:
            p_raw, _ = simplernn_forward(m['params'], Xnum, C, mask)
            raw[m['seed']] = p_raw
            members[m['seed']] = apply_calibration(m['calibration'], p_raw)
        p = np.mean(np.stack([members[m['seed']] for m in self.members]), axis=0)
        return p, members, raw

    def predict_history_arrays(self, H, mask, names, state_version, history_version):
        if state_version != STATE_VERSION or history_version != HISTORY_VERSION:
            raise ModelVersionError('state/history version mismatch')
        if tuple(str(n) for n in names) != self.state_names:
            raise SchemaError('state feature schema mismatch')
        return self.predict_encoded(*self.preprocessor.encode_histories(H, mask))

    def predict_histories(self, histories):
        H = np.stack([h.matrix() for h in histories])
        mask = np.asarray([h.mask for h in histories], dtype=bool)
        p, members, _ = self.predict_history_arrays(H, mask, histories[0].state_names, STATE_VERSION,
                                                    histories[0].history_version)
        return p, members

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / 'preprocessing.npz', **self.preprocessor.arrays())
        files = {'preprocessing': {'file': 'preprocessing.npz', 'sha256': sha256_file(directory / 'preprocessing.npz')}}
        members = []
        for m in self.members:
            name = f"member_seed{m['seed']}.npz"
            cal = m['calibration']
            np.savez(directory / name, **m['params'], calibration_method=np.asarray(cal['method']),
                     sigmoid_coef=np.asarray(cal.get('sigmoid_coef', np.nan)),
                     sigmoid_intercept=np.asarray(cal.get('sigmoid_intercept', np.nan)),
                     isotonic_x=np.asarray(cal.get('isotonic_x', []), dtype=np.float64),
                     isotonic_y=np.asarray(cal.get('isotonic_y', []), dtype=np.float64))
            members.append({'seed': m['seed'], 'file': name, 'sha256': sha256_file(directory / name),
                            'calibration': cal['method']})
        core = {'module_version': MODULE_VERSION, 'candidate': CANDIDATE_B, 'candidate_version': CANDIDATE_B_VERSION,
                'history_version': HISTORY_VERSION, 'state_version': STATE_VERSION,
                'state_names_sha256': sha256_json(list(self.state_names)), 'architecture': ARCHITECTURE,
                'ensemble': 'arithmetic mean of the calibrated member probabilities',
                'files': files, 'members': members}
        manifest = dict(core, bundle_sha256=sha256_json(core))
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        self.model_sha256 = manifest['bundle_sha256']
        return manifest

    @classmethod
    def load(cls, directory, expected_bundle_sha256=None):
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        stored = manifest.pop('bundle_sha256', None)
        if sha256_json(manifest) != stored:
            raise ModelVersionError('bundle manifest hash mismatch')
        if expected_bundle_sha256 is not None and stored != expected_bundle_sha256:
            raise ModelVersionError('bundle is not the expected frozen model')
        expected = {'module_version': MODULE_VERSION, 'candidate': CANDIDATE_B, 'candidate_version': CANDIDATE_B_VERSION,
                    'history_version': HISTORY_VERSION, 'state_version': STATE_VERSION}
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ModelVersionError('bundle version mismatch')
        if sha256_file(directory / manifest['files']['preprocessing']['file']) != manifest['files']['preprocessing']['sha256']:
            raise ModelVersionError('preprocessing file hash mismatch')
        with np.load(directory / manifest['files']['preprocessing']['file'], allow_pickle=False) as z:
            pre = HistoryPreprocessor.from_arrays(z)
        if sha256_json(list(pre.state_names)) != manifest['state_names_sha256']:
            raise ModelVersionError('state schema hash mismatch')
        members = []
        for rec in manifest['members']:
            path = directory / rec['file']
            if sha256_file(path) != rec['sha256']:
                raise ModelVersionError(f"member file hash mismatch: {rec['file']}")
            with np.load(path, allow_pickle=False) as z:
                method = str(z['calibration_method'])
                if method != rec['calibration']:
                    raise ModelVersionError('member calibration differs from manifest')
                cal = {'method': method}
                if method == 'sigmoid':
                    cal.update(sigmoid_coef=float(z['sigmoid_coef']), sigmoid_intercept=float(z['sigmoid_intercept']))
                elif method == 'isotonic':
                    cal.update(isotonic_x=z['isotonic_x'].copy(), isotonic_y=z['isotonic_y'].copy())
                members.append({'seed': rec['seed'], 'params': {k: z[k].copy() for k in PARAM_NAMES}, 'calibration': cal})
        return cls(pre, members, bundle_sha256=stored)


# ----------------------------------------------------------------------------- service
class TemporalWinProbabilityService:
    """On-demand p_t for one frozen candidate. Histories must match its schema and versions."""

    def __init__(self, candidate):
        if not isinstance(candidate, (CandidateA, CandidateB)):
            raise TypeError('unsupported candidate')
        if candidate.model_sha256 in (None, '', 'unsaved'):
            raise ModelVersionError('service requires a frozen, hashed model')
        self.candidate = candidate

    def _check(self, h):
        if not isinstance(h, CausalHistory):
            raise TypeError('predict requires CausalHistory objects')
        if h.history_version != HISTORY_VERSION:
            raise ModelVersionError('history version mismatch')
        if tuple(h.state_names) != tuple(self.candidate.state_names):
            raise SchemaError('history state schema differs from the model schema')

    def predict_many(self, histories):
        histories = list(histories)
        if not histories:
            return []
        for h in histories:
            self._check(h)
        p, members = self.candidate.predict_histories(histories)
        c = self.candidate
        out = []
        for i, h in enumerate(histories):
            snaps = h.snapshot_ms
            out.append(Prediction(h.match_id, int(h.query_ms), float(p[i]), c.name, c.candidate_version, c.model_sha256,
                                  c.calibration, h.history_version, STATE_VERSION, int(snaps[-1]), tuple(h.times),
                                  tuple(h.mask), tuple(snaps), h.source_kind,
                                  {str(k): float(v[i]) for k, v in members.items()}))
        return out

    def predict(self, history):
        return self.predict_many([history])[0]

    def predict_source(self, source, query_ms):
        return self.predict(assemble_history(source, query_ms, self.candidate.state_names))


# ----------------------------------------------------------------------------- training (torch, CPU)
def train_simplernn(Z_table, C_table, seq_index, seq_mask, y, input_width, *, seed, units=8, dropout=0.25,
                    lr=1e-3, batch_size=64, epochs=50, rho=0.9, eps=1e-7, threads=4, on_epoch=None):
    """Fixed-budget RMSprop training with binary cross entropy; unweighted, one sample per FIT query row.

    Z_table/C_table are encoded unique FIT states; seq_index (N, 9) indexes them (masked positions ignored).
    Returns float64 parameters, per-epoch log and a float32 torch evaluation function (for a parity check).
    """
    import torch
    torch.set_num_threads(int(threads))
    torch.manual_seed(int(seed))
    gen = torch.Generator().manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    Z_table = np.asarray(Z_table)
    D = Z_table.shape[1]
    if input_width < D:
        raise SchemaError('input width smaller than numeric width')
    return _train_torch(torch, gen, rng, Z_table, np.asarray(C_table), np.asarray(seq_index),
                        np.asarray(seq_mask, dtype=bool), np.asarray(y), D, units, dropout, lr, batch_size, epochs,
                        rho, eps, on_epoch, width=int(input_width))


def torch_forward(torch, P, Zb, Cb, Mb, keep=None, gen=None):
    kernel, recurrent, bias, out_k, out_b = P
    B, T, D = Zb.shape
    Wn, Wc = kernel[:, :D], kernel[:, D:]
    valid = Cb >= 0
    g = Cb.clamp(min=0)
    weight = valid.to(Zb.dtype)
    if keep is not None:
        Zb = Zb * (torch.bernoulli(torch.full((B, D), keep), generator=gen) / keep)[:, None, :]
        mc = torch.bernoulli(torch.full((B, Wc.shape[1]), keep), generator=gen) / keep
        weight = weight * torch.gather(mc, 1, g.reshape(B, -1)).reshape(g.shape)
    champ = (Wc.t()[g] * weight[..., None]).sum(2)
    h = Zb.new_zeros((B, kernel.shape[0]))
    for k in range(T):
        a = Zb[:, k] @ Wn.t() + h @ recurrent.t() + bias + champ[:, k]
        h = torch.where(Mb[:, k, None], torch.tanh(a), h)
    return h @ out_k + out_b


def _train_torch(torch, gen, rng, Z_table, C_table, seq_index, seq_mask, y, D, units, dropout, lr, batch_size,
                 epochs, rho, eps, on_epoch, width):
    import math
    import time
    import torch.nn.functional as F
    if C_table.size and int(C_table.max()) >= width - D:
        raise SchemaError('champion column index outside the declared input width')
    pad = len(Z_table)
    Zt = torch.from_numpy(np.vstack([Z_table, np.zeros((1, D))]).astype(np.float32))
    Ct = torch.from_numpy(np.vstack([C_table, np.full((1, C_table.shape[1]), -1)]).astype(np.int64))
    idx = torch.from_numpy(np.where(seq_mask, seq_index, pad).astype(np.int64))
    Mt = torch.from_numpy(seq_mask)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    limit = math.sqrt(6. / (width + units))
    kernel = (torch.rand((units, width), generator=gen) * 2 - 1) * limit
    q, r = torch.linalg.qr(torch.randn((units, units), generator=gen))
    recurrent = q * torch.sign(torch.diagonal(r))[None, :]
    out_limit = math.sqrt(6. / (units + 1))
    out_k = (torch.rand((units,), generator=gen) * 2 - 1) * out_limit
    P = [kernel, recurrent, torch.zeros(units), out_k, torch.zeros(())]
    for t in P:
        t.requires_grad_(True)
    opt = torch.optim.RMSprop(P, lr=lr, alpha=rho, eps=eps)
    keep = 1. - dropout
    N = len(y)
    history = []
    for epoch in range(epochs):
        started = time.time()
        perm = rng.permutation(N)
        total, seen = 0., 0
        for s in range(0, N, batch_size):
            b = torch.from_numpy(perm[s:s + batch_size])
            ib = idx[b]
            logit = torch_forward(torch, P, Zt[ib], Ct[ib], Mt[b], keep=keep, gen=gen)
            loss = F.binary_cross_entropy_with_logits(logit, yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
            seen += len(b)
        rec = {'epoch': epoch + 1, 'train_bce_with_dropout': total / seen, 'seconds': round(time.time() - started, 2),
               'finite_parameters': bool(all(torch.isfinite(t).all() for t in P))}
        history.append(rec)
        if on_epoch is not None:
            on_epoch(rec)
        if not rec['finite_parameters']:
            raise FloatingPointError('non-finite parameters during training')
    params = {name: t.detach().to(torch.float64).numpy().copy() for name, t in zip(PARAM_NAMES, P)}
    params['output_bias'] = np.asarray(params['output_bias'], dtype=np.float64)

    def eval_torch(Zb, Cb, Mb):
        with torch.no_grad():
            return torch.sigmoid(torch_forward(torch, P, torch.from_numpy(Zb.astype(np.float32)),
                                               torch.from_numpy(Cb.astype(np.int64)), torch.from_numpy(Mb))).numpy()
    return params, history, eval_torch
