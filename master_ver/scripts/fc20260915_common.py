"""Full-corpus retraining 2026-09-15: shared helpers (pure; importing has no side effects).

Design: docs/CLAUDE_FULL_CORPUS_TRAIN_20260915.md (Codex). Implementation/execution: Claude Opus 5.
Everything written by the pipeline lands in outputs/full_corpus_training_20260915.

This module holds the frozen partition/sampling formulas, the positive-slope sigmoid calibrator,
the versioned V adapter class that is serialized with joblib (module name fc20260915_common must be
importable, i.e. scripts/ on sys.path), q candidate helpers restored from the P4 runner, metrics,
and atomic byte-exact JSON / status writers.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / 'worktrees' / 'engagement-state-value'
REPO = Path('C:/Users/todtj/PycharmProjects/LOL_teamfight')
OUT = ROOT / 'outputs' / 'full_corpus_training_20260915'
PREFLIGHT = ROOT / 'outputs' / 'full_corpus_preflight_20260915'
MAIN_MANIFEST = PREFLIGHT / 'main_matches.csv'
EXPOSURES = ROOT / 'outputs' / 'postkill_objective_delay_full' / 'exposures.csv'
EXTERNAL_MANIFEST_DIR = PREFLIGHT / 'external'
CACHE_MAIN = Path('D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13')
SPEC = ROOT / 'docs' / 'CLAUDE_FULL_CORPUS_TRAIN_20260915.md'

VERSION = 'full_corpus_training_20260915'
ROLE_TAG = 'FULL_CORPUS_EXPLORATORY_MODEL_DEFINED_LABELS_NOT_GROUND_TRUTH'

# ------------------------------------------------------------------ frozen partition / sampling design
MAIN_EXPECTED = {'TRAIN': ('15.14', 74673), 'VALIDATION': ('15.15', 74748), 'TEST': ('15.16', 60579)}
MAIN_TOTAL = 210000
EXTERNAL_SETS = {  # set_id -> (expected complete matches, api patch string, prior-use flag from preflight)
    'KR_16.13': (10064, '16.13', 'previously_used'),
    'KR_16.14_pilot': (200, '16.14', 'pilot_provenance'),
    'KR_16.15': (926, '16.15', 'unknown'),
    'NA1_16.13': (10000, '16.13', 'unknown'),
    'EUW1_complete': (0, None, 'unknown'),
}
FOLD_TAG = 'full-v-oof-20260915:'
VAL_TAG = 'full-val-20260915:'
QUERY_TAG = 'full-v-query-20260915'
N_FOLDS = 5
VAL_ROLES = ('V_CAL', 'V_SELECT', 'Q_CAL', 'Q_SELECT')
GRID_START_MS = 120_000
GRID_STEP_MS = 60_000
BUCKET_MS = 300_000
V_C = 0.01
V_MAX_ITER = 1500
V_CLIP = 1e-12
HORIZONS_S = (60, 90, 120)
PRIMARY_H = 90
MODEL_VERSION = 'fullcorpus_wp_v2_participant_order_20260915'
STATE_VERSION = 'objective_history_v2_participant_order'
P_PRE = 'p_pre_V'


def h8(tag, match_id):
    return int(hashlib.sha256((tag + str(match_id)).encode('utf-8')).hexdigest()[:8], 16)


def train_fold(match_id):
    """TRAIN match fold = int(sha256('full-v-oof-20260915:'+match_id).hexdigest()[:8],16) % 5."""
    return h8(FOLD_TAG, match_id) % N_FOLDS


def validation_role(match_id):
    """VALIDATION role = int(sha256('full-val-20260915:'+match_id).hexdigest()[:8],16) % 4."""
    return VAL_ROLES[h8(VAL_TAG, match_id) % 4]


def query_rank(match_id, query_ms):
    """Deterministic per-query hash; exact string '<match_id>:<query_ms>:full-v-query-20260915'."""
    return hashlib.sha256(f'{match_id}:{int(query_ms)}:{QUERY_TAG}'.encode('utf-8')).hexdigest()


def value_grid(terminal_ms, first_frame_ms, last_frame_ms):
    """Causal minute queries: t = 120000 + k*60000, first_frame <= t <= last_frame and t < terminal."""
    hi = min(int(terminal_ms), int(last_frame_ms) + 1)
    return [t for t in range(GRID_START_MS, hi, GRID_STEP_MS) if t >= int(first_frame_ms)]


def bucket_sample(match_id, grid):
    """One query per nonempty fixed 5-minute bucket [0,5),[5,10),...: minimum query_rank in the bucket."""
    buckets = {}
    for t in grid:
        b = int(t) // BUCKET_MS
        r = query_rank(match_id, t)
        if b not in buckets or r < buckets[b][0]:
            buckets[b] = (r, int(t))
    return [buckets[b][1] for b in sorted(buckets)]


# ------------------------------------------------------------------ hashing / io
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in (sorted(o) if isinstance(o, set) else o)]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, Path):
        return str(o)
    return o


def sha256_json(value):
    return sha256_bytes(json.dumps(clean(value), sort_keys=True, ensure_ascii=False).encode('utf-8'))


def json_bytes(obj):
    return json.dumps(clean(obj), indent=1, ensure_ascii=False, allow_nan=False).encode('utf-8')


def write_json(path, obj):
    """Atomic byte write (no newline translation) so the returned hash equals the file hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json_bytes(obj)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_bytes(data)
    for attempt in range(50):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))
    else:
        raise PermissionError(f'could not replace {path}')
    return sha256_bytes(data)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    return sha256_file(path)


class Status:
    """Per-stage status file plus an aggregated status.json (stage, processed/total, PID, elapsed, error, next)."""

    def __init__(self, out, stage_group):
        self.out = Path(out)
        self.group = stage_group
        self.t0 = time.time()
        (self.out / 'status').mkdir(parents=True, exist_ok=True)
        (self.out / 'logs').mkdir(parents=True, exist_ok=True)
        self.log_path = self.out / 'logs' / f'{stage_group}.log'
        self.history = []

    def log(self, msg):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")} +{time.time() - self.t0:8.1f}s] [{self.group}] {msg}'
        print(line, flush=True)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    def update(self, state, stage, processed=None, total=None, error=None, next_step=None, **extra):
        rec = dict(group=self.group, state=state, stage=stage, pid=os.getpid(),
                   updated=time.strftime('%Y-%m-%d %H:%M:%S'), elapsed_s=round(time.time() - self.t0, 1),
                   processed=processed, total=total, error=error, next_step=next_step, **extra)
        if not self.history or self.history[-1]['stage'] != stage or self.history[-1]['state'] != state:
            self.history.append(dict(stage=stage, state=state, at=rec['updated']))
        rec['history'] = self.history[-60:]
        write_json(self.out / 'status' / f'{self.group}.json', rec)
        aggregate_status(self.out)
        self.log(f'STATUS {state} {stage} processed={processed} total={total} error={error} next={next_step} '
                 + json.dumps(clean(extra), ensure_ascii=False)[:400])


def aggregate_status(out):
    out = Path(out)
    groups = {}
    for p in sorted((out / 'status').glob('*.json')):
        try:
            groups[p.stem] = json.loads(p.read_text(encoding='utf-8'))
        except Exception as exc:  # concurrent writer; next update refreshes
            groups[p.stem] = {'read_error': repr(exc)}
    running = [g for g, r in groups.items() if r.get('state') == 'running']
    latest = max(groups.items(), key=lambda kv: kv[1].get('updated', ''), default=(None, {}))
    agg = dict(version=VERSION, role=ROLE_TAG, updated=time.strftime('%Y-%m-%d %H:%M:%S'),
               running_groups=running, latest_group=latest[0],
               latest={k: latest[1].get(k) for k in ('state', 'stage', 'processed', 'total', 'pid', 'elapsed_s',
                                                      'error', 'next_step')} if latest[0] else None,
               groups={g: {k: r.get(k) for k in ('state', 'stage', 'processed', 'total', 'pid', 'elapsed_s', 'error',
                                                  'next_step', 'updated', 'eta_s', 'rate_per_s')}
                       for g, r in groups.items()})
    try:
        write_json(out / 'status.json', agg)
    except PermissionError:
        pass


# ------------------------------------------------------------------ weights and metrics (restored P4 helpers)
def weights(g):
    """Equal total weight per match within the given subset, normalized to mean 1 (P4 weights())."""
    _, ix, c = np.unique(np.asarray(g), return_inverse=True, return_counts=True)
    w = 1 / c[ix]
    return w / w.mean()


def score(y, p, w):
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    return dict(auc=float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) > 1 else None,
                brier=float(brier_score_loss(y, p, sample_weight=w)),
                logloss=float(log_loss(y, p, labels=[0, 1], sample_weight=w)))


def logit8(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibration(y, p, w):
    """Restored P4 calibration(): joint logistic recalibration slope/intercept and 10 reliability bins."""
    from sklearn.linear_model import LogisticRegression
    out = score(y, p, w)
    out.update(n=int(len(y)), observed=float(np.average(y, weights=w)), predicted=float(np.average(p, weights=w)))
    if len(np.unique(y)) > 1 and np.std(p) > 1e-8:
        m = LogisticRegression(C=1e6, max_iter=1000).fit(logit8(p), y, sample_weight=w)
        out.update(intercept=float(m.intercept_[0]), slope=float(m.coef_[0, 0]))
    bins = []
    for lo in np.arange(0, 1, .1):
        a = (p >= lo) & (p < lo + .1 if lo < .85 else p <= 1)
        if a.any():
            bins.append(dict(lo=float(lo), n=int(a.sum()), predicted=float(np.average(p[a], weights=w[a])),
                             observed=float(np.average(y[a], weights=w[a]))))
    out['bins'] = bins
    return out


def citl(y, p, w):
    """Calibration-in-the-large: intercept with logit(p) offset, slope fixed at 1 (weighted Newton)."""
    if len(np.unique(y)) < 2:
        return None
    o = logit8(p).ravel()
    a = 0.0
    for _ in range(200):
        mu = 1 / (1 + np.exp(-(a + o)))
        grad = np.sum(w * (y - mu))
        hess = np.sum(w * mu * (1 - mu))
        if hess <= 0:
            return None
        step = grad / hess
        a += step
        if abs(step) < 1e-13:
            break
    return float(a)


def ece(y, p, w, bins=10):
    edges = np.linspace(0., 1., bins + 1)
    total, acc = w.sum(), 0.
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi + (1e-10 if hi >= 1. else 0.))
        if mask.any():
            ww = w[mask]
            acc += float(ww.sum() / total) * abs(float(np.average(y[mask], weights=ww)) - float(np.average(p[mask], weights=ww)))
    return float(acc)


def evaluate(y, p, g, bins=True, min_matches=30):
    """Match-weighted evaluation cell; explicit NA for empty / single-class / constant prediction."""
    import warnings
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    g = np.asarray(g)
    if len(y) == 0:
        return dict(n=0, matches=0, auc=None, brier=None, logloss=None, auc_na_reason='empty')
    w = weights(g)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = calibration(y, p, w)
    if not bins:
        out.pop('bins')
    out['matches'] = int(len(np.unique(g)))
    out['positives'] = int(y.sum())
    out['auc_na_reason'] = None if out['auc'] is not None else 'single class'
    if 'slope' not in out:
        out['intercept'] = None
        out['slope'] = None
        out['calibration_na_reason'] = 'single class' if len(np.unique(y)) < 2 else 'constant prediction'
    out['citl_intercept_offset'] = citl(y, p, w)
    out['ece_10bin'] = ece(y, p, w)
    out['sparse_lt30_matches'] = out['matches'] < min_matches
    return out


def auc_direct(y, p, w):
    pos = y == 1
    if pos.all() or not pos.any():
        return None
    u, inv = np.unique(p, return_inverse=True)
    wp = np.bincount(inv, weights=w * pos, minlength=len(u))
    wn = np.bincount(inv, weights=w * ~pos, minlength=len(u))
    below = np.cumsum(wn) - wn
    return float(np.sum(wp * (below + 0.5 * wn)) / (wp.sum() * wn.sum()))


def brier_direct(y, p, w):
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def logloss_direct(y, p, w):
    eps = np.finfo(np.float64).eps
    q = np.clip(p, eps, 1 - eps)
    return float(-np.sum(w * (y * np.log(q) + (1 - y) * np.log(1 - q))) / np.sum(w))


# ------------------------------------------------------------------ V calibration and adapter
def clipped_logit(p, eps=V_CLIP):
    """Numeric transform only: clip to [eps, 1-eps] then log(p) - log1p(-p)."""
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(q) - np.log1p(-q)


def _expit(z):
    from scipy.special import expit
    return expit(z)


class IdentityCalibrator:
    kind = 'raw'

    def predict(self, p):
        return np.asarray(p, dtype=np.float64)

    def describe(self):
        return dict(kind='raw', mapping='identity')


class PositiveSlopeSigmoid:
    """p_cal = expit(a + b * logit(clip(p, 1e-12))) with b constrained > 0 (L-BFGS-B bound b >= SLOPE_MIN).

    Fitted by weighted maximum likelihood with scipy.optimize.minimize (installed optimizer). The bound
    keeps the map strictly increasing so that sign(delta) is preserved; whether it is active is recorded.
    """
    kind = 'sigmoid_pos'
    SLOPE_MIN = 1e-6

    def __init__(self):
        self.a = None
        self.b = None
        self.fit_info = None

    def fit(self, p, y, w):
        from scipy.optimize import minimize
        z = clipped_logit(p)
        y = np.asarray(y, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        sw = w.sum()

        def nll(theta):
            a, b = theta
            eta = a + b * z
            # log(1+exp(eta)) - y*eta, stable
            val = np.sum(w * (np.logaddexp(0.0, eta) - y * eta)) / sw
            mu = _expit(eta)
            r = w * (mu - y) / sw
            return val, np.array([r.sum(), np.dot(r, z)])

        res = minimize(nll, x0=np.array([0.0, 1.0]), jac=True, method='L-BFGS-B',
                       bounds=[(None, None), (self.SLOPE_MIN, None)],
                       options=dict(maxiter=1000, ftol=1e-15, gtol=1e-10))
        self.a, self.b = float(res.x[0]), float(res.x[1])
        _, grad = nll(res.x)
        self.fit_info = dict(optimizer='scipy.optimize.minimize L-BFGS-B', success=bool(res.success),
                             message=str(res.message), nit=int(res.nit), objective=float(res.fun),
                             intercept_a=self.a, slope_b=self.b, slope_lower_bound=self.SLOPE_MIN,
                             slope_bound_active=bool(self.b <= self.SLOPE_MIN * (1 + 1e-9)),
                             final_gradient=grad.tolist(), n=int(len(y)), weight_sum=float(sw),
                             logit_clip=V_CLIP)
        if not np.isfinite(res.x).all():
            raise ValueError('non-finite sigmoid calibration parameters')
        return self

    def predict(self, p):
        return _expit(self.a + self.b * clipped_logit(p))

    def describe(self):
        return dict(kind='sigmoid_pos', intercept_a=self.a, slope_b=self.b, fit=self.fit_info)


class FullCorpusWinProbV2:
    """Versioned V adapter: StateV2 matrix (exact names/version) -> P(final Blue win).

    base: sklearn Pipeline from train.state_value_experiment.logistic (fit-only preprocessing inside);
    calibrator: IdentityCalibrator or PositiveSlopeSigmoid. fit_record documents fitting membership.
    """

    def __init__(self, state_names, feature_names, base, calibrator, fit_record):
        self.model_version = MODEL_VERSION
        self.state_version = STATE_VERSION
        self.state_names = [str(n) for n in state_names]
        if len(set(self.state_names)) != len(self.state_names):
            raise ValueError('duplicate state names')
        if any(re.match(r'^slot\d+_', n) for n in self.state_names):
            raise ValueError('legacy slot names rejected')
        self.feature_names = [str(n) for n in feature_names]
        if 'snapshot_age_s' in self.feature_names:
            raise ValueError('snapshot_age_s must not be a V feature')
        self.keep = [self.state_names.index(n) for n in self.feature_names]
        self.base = base
        self.calibrator = calibrator
        self.fit_record = dict(fit_record)

    @property
    def calibration(self):
        return self.calibrator.kind

    def raw_matrix(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.state_names):
            raise ValueError('state matrix shape mismatch')
        if not np.isfinite(X).all():
            raise ValueError('non-finite state values')
        return self.base.predict_proba(X[:, self.keep])[:, 1]

    def predict_matrix(self, X, names, state_version):
        if state_version != self.state_version:
            raise ValueError('state version mismatch')
        if [str(n) for n in names] != self.state_names:
            raise ValueError('state feature schema mismatch')
        return np.asarray(self.calibrator.predict(self.raw_matrix(X)), dtype=np.float64)

    def identity(self):
        return dict(model_version=self.model_version, state_version=self.state_version,
                    calibration=self.calibration, fit_record=self.fit_record)


def load_v_adapter(path, expected_sha256=None):
    import joblib
    path = Path(path)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f'adapter hash mismatch for {path}')
    model = joblib.load(path)
    if not isinstance(model, FullCorpusWinProbV2):
        raise TypeError('not a FullCorpusWinProbV2 adapter')
    if model.model_version != MODEL_VERSION or model.state_version != STATE_VERSION:
        raise ValueError('adapter version mismatch')
    return model


# ------------------------------------------------------------------ q helpers (restored P4 rules, p_pre -> p_pre_V)
ECON_SLOT_SUFFIXES = ['totalGold_norm', 'curGold_norm', 'level_norm', 'xp_norm', 'alive', 'laneCS_norm', 'jgCS_norm',
                      'kills', 'deaths']
Q_CANDIDATES = ('constant', 'p_pre_logistic', 'p_pre_spline', 'ridge_raw', 'ridge_sigmoid', 'ridge_isotonic',
                'economic_raw', 'economic_sigmoid', 'economic_isotonic')
ECON_SEEDS = (7, 42, 123)
LGBM_PARAMS = dict(n_estimators=250, num_leaves=15, learning_rate=.04, min_child_samples=100, reg_lambda=1,
                   n_jobs=4, verbosity=-1, colsample_bytree=.9)
Q_FORBIDDEN_TOKENS = ('snapshot_age', 'p_post', 'delta', 'endpoint', 'duration', 'label', 'winner', 'game_end',
                      'next_kill', 'next_start', 'reason', 'horizon', 'during_', 'after_last', 'position', 'xy_',
                      'champion_id', 'match', 'patch', 'Y_', 'L_ms', 'first_kill', 'terminal', 'fold')


def _team_aggregate_rule(n):
    return ((n.startswith('blue_') or n.startswith('red_')) and not n.endswith('_x_time') and 'age_' not in n
            and 'last_' not in n and not n.endswith('_ever'))


def econ_rule(n):
    return (n == 'time_minutes'
            or (n.startswith('participant_slot') and any(n.endswith(a) for a in ECON_SLOT_SUFFIXES))
            or _team_aggregate_rule(n))


def shap_group(n):
    if n == P_PRE:
        return 'prior_win_probability'
    if n.startswith('time_') or n == 'snapshot_age_s':
        return 'time_and_observation_age'
    if any(k in n for k in ['baron', 'elder', 'dragon', 'soul', 'herald', 'horde', 'atakhan']):
        return 'objectives'
    if any(k in n for k in ['Gold', 'xp_', 'level_', 'CS_']):
        return 'economy_and_experience'
    if any(k in n for k in ['kills', 'deaths', 'death_', 'alive']):
        return 'combat_and_survival'
    if any(k in n for k in ['tower', 'inhibitor', 'plates']):
        return 'structures'
    if 'champion_id' in n:
        return 'champion_identity'
    return 'health_mana_other'


def feature_unit(n):
    base = n[:-len('_x_time')] if n.endswith('_x_time') else n
    suffix = ' multiplied by query_minutes/30 (phase interaction)' if n.endswith('_x_time') else ''
    if n == P_PRE:
        return 'probability of final Blue win at q_pre = s-1 ms (held-out-fold V on TRAIN, final V elsewhere)'
    if n == 'time_minutes':
        return 'minutes (q_pre / 60000)'
    if n == 'time_minutes_sq':
        return 'minutes squared'
    if n == 'unknown_objective_team_count':
        return 'count of objective/structure/soul events without a Blue/Red team up to q_pre'
    m = re.match(r'participant_slot(\d)_(.+)$', n)
    if m:
        f = m.group(2)
        if f.endswith('_norm'):
            return 'normalized cache value at last frame <= q_pre (unitless)'
        if f.endswith('_pct'):
            return 'fraction [0, 1] at last frame <= q_pre'
        if f == 'alive':
            return 'binary at last frame <= q_pre'
        if f in ('kills', 'deaths'):
            return 'count up to q_pre'
        if f == 'death_age_minutes':
            return 'minutes since last death, capped at 10 (10 if none)'
        if f == 'champion_id':
            return 'categorical identifier (excluded)'
        return 'binary indicator up to q_pre'
    if base.endswith('_age_minutes'):
        return 'minutes since last acquisition, capped at 10' + suffix
    if '_acquired_last_' in base or base.endswith('_ever') or '_soul_' in base:
        return 'binary indicator' + suffix
    return 'count up to q_pre' + suffix


def q_feature_sets(input_names):
    names = list(input_names)
    ridge = [n for n in names if not n.endswith('champion_id')]
    econ = [n for n in names if econ_rule(n)]
    return {'p_pre': [P_PRE], 'ridge': ridge, 'economic': econ}


def predict_q_bundle(b, X):
    kind = b['candidate']
    if kind == 'constant':
        return np.full(len(X), b['prior'])
    Xi = X[:, b['input_columns']]
    if kind in ('p_pre_logistic', 'p_pre_spline', 'ridge_raw'):
        return b['pipeline'].predict_proba(Xi)[:, 1]
    if kind.startswith('ridge_'):
        raw = b['base_pipeline'].predict_proba(Xi)[:, 1]
    elif kind.startswith('economic_'):
        raw = np.mean([m.predict_proba(Xi)[:, 1] for m in b['seed_models']], axis=0)
        if kind == 'economic_raw':
            return raw
    else:
        raise ValueError(kind)
    if kind.endswith('_sigmoid'):
        return b['calibrator'].predict_proba(logit8(raw))[:, 1]
    if kind.endswith('_isotonic'):
        return b['calibrator'].predict(raw)
    raise ValueError(kind)


TIME_BANDS_Q = ((0, 2), (2, 10), (10, 20), (20, 30), (30, 1000))
TIME_BANDS_V = ((2, 10), (10, 20), (20, 30), (30, 1000))
SIX_SOULS = ('AIR', 'EARTH', 'FIRE', 'WATER', 'HEXTECH', 'CHEMTECH')


def objective_strata(X, names, unassigned):
    """P1 objective-history strata on StateV2 matrices (either team), with explicit diagnostic soul strata."""
    ix = {n: i for i, n in enumerate(names)}

    def col(n):
        return X[:, ix[n]]

    def either(n):
        return (col('blue_' + n) > 0) | (col('red_' + n) > 0)

    owned = either('soul_event_recorded')
    unassigned = np.asarray(unassigned) > 0
    strata = {
        'no_baron_elder_or_soul_history': ~(either('baron_ever') | either('elder_ever') | owned | unassigned),
        'dragons_any': either('dragons'),
        'dragon_count_diff_ge2': np.abs(col('blue_dragons') - col('red_dragons')) >= 2,
        'herald_ever_any': either('herald_ever'),
        'grubs_horde_ever_any': either('horde_ever'),
        'atakhan_any': either('atakhan_ever'),
        'inhibitor_kill_any': either('inhibitor_kills'),
        'baron_ever_any': either('baron_ever'),
        'baron_acquired_last_180s_any': either('baron_acquired_last_180s'),
        'elder_ever_any': either('elder_ever'),
        'elder_acquired_last_180s_any': either('elder_acquired_last_180s'),
        'owned_soul_any': owned,
        'unassigned_soul_teamId0_any_DIAGNOSTIC': unassigned,
        'unknown_objective_team_count_positive': col('unknown_objective_team_count') > 0,
    }
    for d in SIX_SOULS:
        strata[f'owned_soul_{d}'] = either(f'soul_{d}')
    return strata
