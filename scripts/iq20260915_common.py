"""Incremental q training 2026-09-15: shared helpers (pure; importing has no side effects).

Design: docs/CLAUDE_INCREMENTAL_Q_TRAIN_20260915.md (Codex). Implementation/execution: Claude Opus 5.
Everything written by this run lands in outputs/incremental_q_training_20260915. Parents (full corpus, cohort/role,
label validity, objective-channel ablation, collaborator critique audit) are READ ONLY; parent helpers that write
logs or gates (cr20260915_data.load_eval_set, cr20260915_common.outcome_gate/Status/log_command) are never called.

Reused pure helpers: fc20260915_common.weights/evaluate/ece/logit8/sha256_*/write_json/save_npz/q_feature_sets and
cr20260915_common.fit_calibrators/apply_calibration/_FastAUC. Model classes here are serialized with joblib, so
scripts/ must be on sys.path and this module importable as iq20260915_common when bundles are loaded.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import warnings

import numpy as np

import fc20260915_common as C
import cr20260915_common as K

ROOT = C.ROOT
OUT = ROOT / 'outputs' / 'incremental_q_training_20260915'
SMOKE = OUT / 'smoke_train_only'
FC = C.OUT
CR = K.OUT
LVO = ROOT / 'outputs' / 'label_validity_full_20260915'
OCO = ROOT / 'outputs' / 'objective_channel_ablation_20260915'
AUDIT = ROOT / 'outputs' / 'collaborator_critique_audit_20260915'
SPEC = ROOT / 'docs' / 'CLAUDE_INCREMENTAL_Q_TRAIN_20260915.md'
PROVENANCE_DOCS = (ROOT / 'docs' / 'DELTA_Q_FAIR_BALANCED_PROTOCOL_20260915.md',
                   ROOT / 'docs' / 'COLLABORATOR_CRITIQUE_RESPONSE_20260915.md')
PYTHON = 'C:/Users/todtj/anaconda3/python.exe'
VERSION = 'incremental_q_training_20260915'
ROLE_TAG = 'EXPLORATORY_FOLLOWUP_AFTER_PRIOR_TEST_EXPOSURE_MODEL_DEFINED_LABELS_NOT_CONFIRMATORY'

H = 90
COHORTS = ('T', 'N')
COHORT_CODE = {'T': 1, 'N': 0}
EXT = ('KR_16.13', 'KR_16.14_pilot', 'KR_16.15', 'NA1_16.13')
EVAL_SETS = ('MAIN_TEST',) + tuple(f'EXT_{s}' for s in EXT)
SEALED_SETS = EVAL_SETS
EXPECTED_COUNTS = {  # h90-valid rows per cohort (spec section "Fixed scientific contract" 2; external from parent cohort manifest)
    'TRAIN': {'T': 39605, 'N': 159753}, 'Q_CAL': {'T': 10390, 'N': 40565}, 'Q_SELECT': {'T': 10195, 'N': 40180},
    'MAIN_TEST': {'T': 32981, 'N': 130595}, 'EXT_KR_16.13': {'T': 5202, 'N': 20228},
    'EXT_KR_16.14_pilot': {'T': 101, 'N': 374}, 'EXT_KR_16.15': {'T': 507, 'N': 1712}, 'EXT_NA1_16.13': {'T': 5312, 'N': 21226}}
EXPECTED_RIDGE_COUNT = 352
PT_INPUTS = ('p_pre_V', 'time_minutes')
PT_DIM = 48

# ------------------------------------------------------------------ fixed search spaces (spec "Three newly fitted families")
FAMILIES = ('pt', 'logit', 'lgbm')
C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
LOGIT_TOL = 1e-8
LOGIT_MAX_ITER = 5000
LOGIT_RETRY_MAX_ITER = 20000
LOGIT_RANDOM_STATE = 7
SPLINE = dict(n_knots=5, degree=3, knots='quantile', include_bias=False, extrapolation='constant')
LGBM_GRID = ((15, 50), (15, 100), (31, 50), (31, 100), (63, 50), (63, 100))
LGBM_SEEDS = (7, 42, 123)
LGBM_FIXED = dict(objective='binary', learning_rate=0.04, n_estimators=1000, reg_lambda=1.0, colsample_bytree=0.9,
                  subsample=1.0, n_jobs=4, verbosity=-1, metric='None', deterministic=True, force_col_wise=True)
LGBM_PATIENCE = 50
STOP_TAG = 'iq20260915_stop:'
STOP_MOD = 10
CALS = K.CALS  # ('raw', 'sigmoid', 'isotonic')
BOOT_REPS = 1000
BOOT_SEED = 20260915
SPARSE_MATCHES = 30
TIME_BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, np.inf))
BALANCED = {'B40': (0.40, 0.60), 'B45': (0.45, 0.55)}
LEGACY = {  # evaluation name -> key inside parent eval/predictions/A_<set>_h90_<cohort>.npz
    'old_A_specialist': None,  # spec_<specialist_chosen>
    'old_pooled': 'pooled', 'old_p_pre_spline': 'spec_p_pre_spline', 'old_p_pre_logistic': 'spec_p_pre_logistic',
    'old_constant': 'spec_constant'}
NEW_WINNERS = ('pt_winner', 'logit_winner', 'lgbm_winner')
CONTRASTS = (  # (a, b, label); a - b; negative Brier / log loss = a better
    ('lgbm_winner', 'pt_winner', 'PRIMARY: full LightGBM - PT'),
    ('logit_winner', 'pt_winner', 'planned: full logistic - PT'),
    ('lgbm_winner', 'logit_winner', 'planned: full LightGBM - full logistic'),
    ('pt_winner', 'old_p_pre_spline', 'planned: PT - old p_pre spline'),
    ('lgbm_winner', 'old_A_specialist', 'planned: new LightGBM - old A specialist'))


def config_names(family):
    if family in ('pt', 'logit'):
        return [f'{family}_C{c:g}' for c in C_GRID]
    if family == 'lgbm':
        return [f'lgbm_L{nl}_M{mc}' for nl, mc in LGBM_GRID]
    raise ValueError(family)


def config_params(family, config):
    i = config_names(family).index(config)
    if family == 'lgbm':
        return dict(num_leaves=LGBM_GRID[i][0], min_child_samples=LGBM_GRID[i][1])
    return dict(C=C_GRID[i])


def candidate_names(family):
    """18 joint candidates '<config>__<calibration>' in lexical tie order."""
    return sorted(f'{cfg}__{cal}' for cfg in config_names(family) for cal in CALS)


def split_candidate(name):
    cfg, cal = name.split('__')
    return cfg, cal


def select_rule(metrics):
    """metrics: {candidate: (brier, logloss)} -> (chosen, ranking). Brier, then log loss, then name (lexical)."""
    rows = sorted((float(b), float(l), str(n)) for n, (b, l) in metrics.items())
    return rows[0][2], [r[2] for r in rows]


# ------------------------------------------------------------------ deterministic allocations
def is_stop_match(match_id):
    """Internal early-stopping match: int(sha256('iq20260915_stop:'+match)[:8], 16) % 10 == 0 (same for T and N)."""
    return C.h8(STOP_TAG, match_id) % STOP_MOD == 0


def stop_mask(g):
    g = np.asarray(g).astype(str)
    u, inv = np.unique(g, return_inverse=True)
    return np.asarray([is_stop_match(m) for m in u.tolist()], dtype=bool)[inv]


def smoke_keep(g, modulus=8):
    g = np.asarray(g).astype(str)
    u, inv = np.unique(g, return_inverse=True)
    return np.asarray([C.h8('iq20260915_smoke_subset:', m) % modulus == 0 for m in u.tolist()], dtype=bool)[inv]


def membership_sha(g):
    return C.sha256_json(sorted(set(np.asarray(g).astype(str).tolist())))


def weights(g):
    """Equal total weight per match within this subset, row mean 1 (fc20260915_common.weights); empty -> empty."""
    g = np.asarray(g)
    return C.weights(g) if len(g) else np.zeros(0)


def weight_record(g, w):
    g = np.asarray(g).astype(str)
    _, inv = np.unique(g, return_inverse=True)
    tot = np.bincount(inv, weights=w)
    return dict(rows=int(len(g)), matches=int(len(tot)), weight_sum=float(np.sum(w)), weight_mean=float(np.mean(w)) if len(w) else None,
                per_match_total_min=float(tot.min()) if len(tot) else None, per_match_total_max=float(tot.max()) if len(tot) else None,
                membership_sha256=membership_sha(g))


# ------------------------------------------------------------------ PT design
class PTDesign:
    """p_pre_V x time_minutes tensor-product spline design.

    Each marginal: SplineTransformer(n_knots=5, degree=3, knots='quantile', include_bias=False,
    extrapolation='constant') fit (unweighted, like parent preprocessing) on cohort TRAIN rows only.
    Columns: Bp (6), Bt (6), then Bp_i * Bt_j for i in p-bases (major), j in t-bases (minor): 6 + 6 + 36 = 48.
    """

    def fit(self, p, t):
        from sklearn.preprocessing import SplineTransformer
        p = np.asarray(p, dtype=np.float64).reshape(-1, 1)
        t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
        self.p_spline = SplineTransformer(**SPLINE).fit(p)
        self.t_spline = SplineTransformer(**SPLINE).fit(t)
        self.n_p = int(self.p_spline.n_features_out_)
        self.n_t = int(self.t_spline.n_features_out_)
        self.names = ([f'Bp{i}' for i in range(self.n_p)] + [f'Bt{j}' for j in range(self.n_t)]
                      + [f'Bp{i}*Bt{j}' for i in range(self.n_p) for j in range(self.n_t)])
        self.fit_rows = int(len(p))
        return self

    def transform(self, p, t):
        p = np.asarray(p, dtype=np.float64).reshape(-1, 1)
        t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
        Bp = self.p_spline.transform(p)
        Bt = self.t_spline.transform(t)
        prod = (Bp[:, :, None] * Bt[:, None, :]).reshape(len(p), self.n_p * self.n_t)
        return np.hstack([Bp, Bt, prod])

    def describe(self):
        kp = self.p_spline.bsplines_[0].t
        kt = self.t_spline.bsplines_[0].t
        return dict(inputs=list(PT_INPUTS), spline_params=SPLINE, n_p=self.n_p, n_t=self.n_t, dimension=len(self.names),
                    names=self.names, order='Bp(6), Bt(6), Bp_i*Bt_j with i major and j minor',
                    p_knot_vector=kp.tolist(), t_knot_vector=kt.tolist(),
                    p_interior_knots_unique=int(len(np.unique(kp[SPLINE['degree']:-SPLINE['degree']]))),
                    t_interior_knots_unique=int(len(np.unique(kt[SPLINE['degree']:-SPLINE['degree']]))),
                    fit_rows=self.fit_rows, fit_weighting='unweighted rows of cohort TRAIN (knot quantiles)')


# ------------------------------------------------------------------ logistic families
def _logistic_attempt(Z, y, w, C_value, max_iter):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=C_value, solver='lbfgs', tol=LOGIT_TOL, max_iter=max_iter, random_state=LOGIT_RANDOM_STATE)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        lr.fit(Z, y, sample_weight=w)
    conv = [str(x.message)[:400] for x in wl if issubclass(x.category, ConvergenceWarning)]
    other = sorted(set(f'{x.category.__name__}: {str(x.message)[:200]}' for x in wl if not issubclass(x.category, ConvergenceWarning)))
    n_iter = int(np.max(lr.n_iter_))
    # diagnostic gradient of sklearn's lbfgs objective: sum(w*l)/sum(w) + ||beta||^2 / (2 C sum(w))
    eta = Z @ lr.coef_[0] + lr.intercept_[0]
    mu = 1.0 / (1.0 + np.exp(-eta))
    sw = float(np.sum(w))
    r = w * (mu - y) / sw
    grad = np.concatenate([Z.T @ r + lr.coef_[0] / (C_value * sw), [r.sum()]])
    rec = dict(C=float(C_value), max_iter=int(max_iter), tol=LOGIT_TOL, solver='lbfgs', random_state=LOGIT_RANDOM_STATE,
               n_iter=n_iter, convergence_warnings=conv, other_warnings=other,
               converged=bool(n_iter < max_iter and not conv), seconds=round(time.time() - t0, 2),
               diagnostic_max_abs_gradient=float(np.max(np.abs(grad))), coef_l2=float(np.linalg.norm(lr.coef_)),
               intercept=float(lr.intercept_[0]))
    return lr, rec


def fit_logistic_policy(Z, y, w, C_value):
    """First attempt max_iter 5000; ONLY if not converged one predeclared retry at 20000 (same tol/data/settings).

    converged = n_iter < max_iter and no sklearn ConvergenceWarning. Returns (model, attempts, eligible).
    The first attempt's coefficients are retained in its record when a retry happens.
    """
    lr, rec = _logistic_attempt(Z, y, w, C_value, LOGIT_MAX_ITER)
    attempts = [rec]
    if not rec['converged']:
        rec['retained_first_attempt_coef'] = lr.coef_[0].tolist()
        lr, rec2 = _logistic_attempt(Z, y, w, C_value, LOGIT_RETRY_MAX_ITER)
        rec2['retry_reason'] = 'first attempt not converged'
        attempts.append(rec2)
    return lr, attempts, bool(attempts[-1]['converged'])


class LinearQBase:
    """family 'pt': PTDesign -> StandardScaler -> weighted LogisticRegression on (p_pre_V, time_minutes).
    family 'logit': SimpleImputer(median) -> StandardScaler -> weighted LogisticRegression on the 352 ridge inputs.
    All preprocessing fit (unweighted) on the cohort TRAIN rows passed to fit(); the logistic fit uses match weights."""

    def __init__(self, family, config, input_names):
        if family not in ('pt', 'logit'):
            raise ValueError(family)
        self.family = family
        self.config = config
        self.C_value = config_params(family, config)['C']
        self.input_names = list(input_names)
        if family == 'pt' and tuple(self.input_names) != PT_INPUTS:
            raise ValueError('PT inputs must be exactly (p_pre_V, time_minutes)')

    def design(self, Xin):
        Xin = np.asarray(Xin, dtype=np.float64)
        if Xin.shape[1] != len(self.input_names):
            raise ValueError('input width mismatch')
        if self.family == 'pt':
            return self.scaler.transform(self.pt.transform(Xin[:, 0], Xin[:, 1]))
        return self.scaler.transform(self.imputer.transform(Xin))

    def fit(self, Xin, y, w):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        Xin = np.asarray(Xin, dtype=np.float64)
        if not np.isfinite(Xin).all():
            raise ValueError('non-finite TRAIN inputs (parent schema contract violated)')
        if self.family == 'pt':
            self.pt = PTDesign().fit(Xin[:, 0], Xin[:, 1])
            B = self.pt.transform(Xin[:, 0], Xin[:, 1])
        else:
            self.imputer = SimpleImputer(strategy='median').fit(Xin)
            B = self.imputer.transform(Xin)
        self.scaler = StandardScaler().fit(B)
        Z = self.scaler.transform(B)
        self.model, self.attempts, self.eligible = fit_logistic_policy(Z, np.asarray(y, dtype=np.int64), np.asarray(w, dtype=np.float64), self.C_value)
        std = B.std(axis=0)
        self.design_record = dict(columns=int(B.shape[1]), rank_standardized=int(np.linalg.matrix_rank(Z)),
                                  min_column_std=float(std.min()), zero_variance_columns=int(np.sum(self.scaler.var_ == 0)),
                                  scaler_n_samples_seen=int(self.scaler.n_samples_seen_) if np.ndim(self.scaler.n_samples_seen_) == 0 else int(np.max(self.scaler.n_samples_seen_)),
                                  fit_rows=int(len(y)))
        if self.family == 'pt':
            self.design_record['pt'] = self.pt.describe()
        else:
            self.design_record['imputer_statistics_sha256'] = C.sha256_bytes(self.imputer.statistics_.astype(np.float64).tobytes())
        return self

    def raw(self, Xin):
        return self.model.predict_proba(self.design(Xin))[:, 1]


# ------------------------------------------------------------------ LightGBM family
def brier_evaluator(w_stop):
    """LightGBM sklearn-API eval metric: explicit match-weighted Brier on the internal stop rows (lower is better)."""
    w_stop = np.asarray(w_stop, dtype=np.float64)
    den = float(w_stop.sum())

    def weighted_brier(y_true, y_pred, weight=None):
        if len(y_pred) != len(w_stop):
            raise ValueError('evaluator weight length mismatch')
        if weight is not None and not np.allclose(weight, w_stop, rtol=1e-6, atol=1e-7):
            raise ValueError('dataset weights differ from the explicit stop weights')
        return 'weighted_brier', float(np.sum(w_stop * (np.asarray(y_pred, dtype=np.float64) - y_true) ** 2) / den), False
    return weighted_brier


def first_argmin_iteration(curve):
    """1-based iteration of the first minimum (ties -> earlier)."""
    return int(np.argmin(np.asarray(curve, dtype=np.float64))) + 1


class LgbmQBase:
    """Three-seed LightGBM on the 352 ridge inputs (SimpleImputer(median), no scaler).

    Per seed: internal stop phase (fit90 rows -> imputer + early stopping on stop10 match-weighted Brier, patience 50,
    ties earlier) then a from-scratch refit on ALL cohort TRAIN rows (imputer refit there) with n_estimators = chosen
    iteration. Candidate raw probability = mean of the three seed probabilities."""

    def __init__(self, config, input_names, seeds=LGBM_SEEDS, fixed=None):
        self.family = 'lgbm'
        self.config = config
        self.params = dict(LGBM_FIXED if fixed is None else fixed, **config_params('lgbm', config))
        self.input_names = list(input_names)
        self.seeds = tuple(seeds)

    def fit(self, Xin, y, g, log=None):
        import lightgbm as lgb
        from lightgbm import LGBMClassifier
        from sklearn.impute import SimpleImputer
        Xin = np.asarray(Xin, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        g = np.asarray(g).astype(str)
        if not np.isfinite(Xin).all():
            raise ValueError('non-finite TRAIN inputs (parent schema contract violated)')
        stop = stop_mask(g)
        fit90 = ~stop
        w_fit = weights(g[fit90])
        w_stop = weights(g[stop])
        w_all = weights(g)
        imp90 = SimpleImputer(strategy='median').fit(Xin[fit90])
        X90, Xst = imp90.transform(Xin[fit90]), imp90.transform(Xin[stop])
        self.imputer = SimpleImputer(strategy='median').fit(Xin)
        Xall = self.imputer.transform(Xin)
        self.stop_record = dict(allocation=f"sha256('{STOP_TAG}'+match_id)[:8] mod {STOP_MOD} == 0 -> stop10",
                                fit90=weight_record(g[fit90], w_fit), stop10=weight_record(g[stop], w_stop), full=weight_record(g, w_all),
                                fit90_stop10_match_overlap=int(len(set(g[fit90].tolist()) & set(g[stop].tolist()))),
                                imputer_fit90_n_features=int(imp90.n_features_in_), imputer_full_rows=int(len(Xin)),
                                evaluator='explicit match-weighted Brier on stop10 (metric=None disables built-in metrics)',
                                patience=LGBM_PATIENCE, seeds={})
        self.models = []
        for seed in self.seeds:
            t0 = time.time()
            rec_eval = {}
            est = LGBMClassifier(random_state=seed, **self.params)
            est.fit(X90, y[fit90], sample_weight=w_fit, eval_set=[(Xst, y[stop])], eval_sample_weight=[w_stop],
                    eval_metric=brier_evaluator(w_stop),
                    callbacks=[lgb.early_stopping(LGBM_PATIENCE, first_metric_only=False, verbose=False, min_delta=0.0),
                               lgb.record_evaluation(rec_eval)])
            metrics_seen = sorted(rec_eval.get('valid_0', {}).keys())
            curve = rec_eval['valid_0']['weighted_brier']
            best = int(est.best_iteration_)
            stop_brier_at_best = float(np.sum(w_stop * (est.predict_proba(Xst, num_iteration=best)[:, 1] - y[stop]) ** 2) / w_stop.sum())
            stop_s = time.time() - t0
            t1 = time.time()
            final = LGBMClassifier(random_state=seed, **dict(self.params, n_estimators=best))
            final.fit(Xall, y, sample_weight=w_all)
            self.models.append(final)
            self.stop_record['seeds'][str(seed)] = dict(
                best_iteration=best, first_argmin_iteration=first_argmin_iteration(curve), rounds_evaluated=len(curve),
                cap_reached=bool(len(curve) >= self.params['n_estimators'] and len(curve) - best < LGBM_PATIENCE),
                early_stopped=bool(len(curve) < self.params['n_estimators']),
                best_stop_weighted_brier=float(curve[best - 1]), stop_brier_recomputed_from_predict=stop_brier_at_best,
                evaluator_metrics_seen=metrics_seen, stop_curve=[float(v) for v in curve],
                final_n_estimators=int(final.n_estimators), final_num_trees=int(final.booster_.num_trees()),
                stop_seconds=round(stop_s, 2), refit_seconds=round(time.time() - t1, 2))
            if log:
                log(f'{self.config} seed {seed}: best_iteration={best} rounds={len(curve)} stop_brier={curve[best - 1]:.6f} '
                    f'stop {stop_s:.1f}s refit {time.time() - t1:.1f}s')
        return self

    def seed_raw(self, Xin):
        Xi = self.imputer.transform(np.asarray(Xin, dtype=np.float64))
        return np.stack([m.predict_proba(Xi)[:, 1] for m in self.models])

    def raw(self, Xin):
        return self.seed_raw(Xin).mean(axis=0)


# ------------------------------------------------------------------ bundles
def input_columns(names, family, ridge):
    return [names.index(n) for n in (PT_INPUTS if family == 'pt' else ridge)]


def make_bundle(family, config, base, calibrators, input_names_all, ridge, manifest):
    return dict(version=VERSION, family=family, config=config, params=config_params(family, config),
                input_names_all_sha256=C.sha256_json(list(input_names_all)),
                input_names=list(PT_INPUTS if family == 'pt' else ridge),
                input_columns=input_columns(list(input_names_all), family, ridge),
                base=base, calibrators=calibrators, manifest=manifest)


def bundle_raw(b, X_all, names_all=None):
    if names_all is not None and C.sha256_json(list(names_all)) != b['input_names_all_sha256']:
        raise ValueError('input schema differs from the bundle schema')
    return b['base'].raw(np.asarray(X_all)[:, b['input_columns']])


def calibrate(b, cal, raw):
    return K.apply_calibration(cal, None if cal == 'raw' else b['calibrators'][cal], raw)


def bundle_predict(b, cal, X_all, names_all=None):
    return calibrate(b, cal, bundle_raw(b, X_all, names_all))


def calibrator_record(cals, raw_cal):
    sg, iso = cals['sigmoid'], cals['isotonic']
    return dict(sigmoid=dict(kind='LogisticRegression(C=1e6, max_iter=1000) on clip(raw,1e-8) logit (cr20260915_common.fit_calibrators); '
                                  'no positive-slope constraint', slope=float(sg.coef_[0, 0]), intercept=float(sg.intercept_[0]),
                             logit_clip=1e-8, n_iter=int(np.max(sg.n_iter_)), slope_positive=bool(sg.coef_[0, 0] > 0)),
                isotonic=dict(kind="IsotonicRegression(out_of_bounds='clip')", thresholds=int(len(iso.X_thresholds_)),
                              min_output=float(np.min(iso.y_thresholds_)), max_output=float(np.max(iso.y_thresholds_))),
                raw_cal_min=float(np.min(raw_cal)), raw_cal_max=float(np.max(raw_cal)))


# ------------------------------------------------------------------ evaluation cells
def time_bin_masks(t):
    t = np.asarray(t, dtype=np.float64)
    return {f'time_{int(lo)}_{"inf" if not np.isfinite(hi) else int(hi)}': (t >= lo) & (t < hi) for lo, hi in TIME_BINS}


def balanced_masks(p):
    p = np.asarray(p, dtype=np.float64)
    return {k: (p >= lo) & (p <= hi) for k, (lo, hi) in BALANCED.items()}


def cell_masks(p_pre, t):
    """Common evaluation cells from frozen p_pre and pre time only (independent of new q and labels)."""
    n = len(p_pre)
    cells = {'all': np.ones(n, dtype=bool)}
    bal = balanced_masks(p_pre)
    tb = time_bin_masks(t)
    cells.update(bal)
    cells.update(tb)
    for k, m in tb.items():
        cells[f'B40_x_{k}'] = bal['B40'] & m
    return cells


def reliability_fixed10(y, p, w):
    edges = np.linspace(0.0, 1.0, 11)
    out = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (p >= lo) & ((p < hi) if i < 9 else (p <= hi))
        out.append(dict(lo=float(lo), hi=float(hi), rows=int(m.sum()), weight=float(w[m].sum()) if m.any() else 0.0,
                        predicted=float(np.average(p[m], weights=w[m])) if m.any() else None,
                        observed=float(np.average(y[m], weights=w[m])) if m.any() else None))
    return out


def cell_metrics(y, p, g, bins=True):
    """Match-weighted metrics inside one cell (weights recomputed within the cell)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    g = np.asarray(g).astype(str)
    n = len(y)
    base = dict(rows=int(n), matches=int(len(np.unique(g))), positives=int(y.sum()) if n else 0,
                positive_rate_rows=float(y.mean()) if n else None,
                exact_prob_0=int(np.sum(p == 0)), exact_prob_1=int(np.sum(p == 1)),
                exact_0_or_1_opposite_label=int(np.sum(((p == 1) & (y == 0)) | ((p == 0) & (y == 1)))),
                sparse_lt30_matches=bool(len(np.unique(g)) < SPARSE_MATCHES), single_class=bool(n > 0 and len(np.unique(y)) < 2))
    if n == 0:
        base.update(empty=True, brier=None, logloss=None, auc=None, auc_na_reason='empty', intercept=None, slope=None, ece_10bin=None)
        return base
    ev = C.evaluate(y, p, g, bins=False)
    w = weights(g)
    base.update(empty=False, brier=ev['brier'], logloss=ev['logloss'], auc=ev['auc'], auc_na_reason=ev['auc_na_reason'],
                positive_rate_match_weighted=float(np.average(y, weights=w)), mean_prediction_match_weighted=float(np.average(p, weights=w)),
                intercept=ev['intercept'], slope=ev['slope'], calibration_na_reason=ev.get('calibration_na_reason'),
                citl_intercept_offset=ev['citl_intercept_offset'], ece_10bin=ev['ece_10bin'],
                calibration_fit_note='descriptive recalibration fits only; never applied to evaluation predictions')
    if bins:
        base['reliability_fixed10'] = reliability_fixed10(y, p, w)
    return base


def paired_bootstrap(y, preds, g, pairs, reps=BOOT_REPS, seed=BOOT_SEED):
    """Paired match bootstrap for fixed models (same multiplicities for every model in the cell).

    Row weight = equal-per-match weight of the cell x match multiplicity. Brier / log loss from per-match weighted loss
    sums; AUC exact weighted Mann-Whitney (ties 1/2, cr20260915_common._FastAUC). Degenerate (single-class) replicates
    give NaN AUC and are counted; percentile intervals use the finite replicates.
    """
    y = np.asarray(y).astype(np.float64)
    g = np.asarray(g).astype(str)
    u, inv = np.unique(g, return_inverse=True)
    nm = len(u)
    w = weights(g)
    eps = np.finfo(np.float64).eps
    models = sorted({m for pr in pairs for m in pr[:2]})
    Wm = np.bincount(inv, weights=w, minlength=nm)
    S, A = {}, {}
    for m in models:
        p = np.asarray(preds[m], dtype=np.float64)
        q = np.clip(p, eps, 1 - eps)
        S[m] = dict(brier=np.bincount(inv, weights=w * (p - y) ** 2, minlength=nm),
                    logloss=np.bincount(inv, weights=w * -(y * np.log(q) + (1 - y) * np.log(1 - q)), minlength=nm))
        A[m] = K._FastAUC(y, p)
    point = {m: dict(auc=A[m](w), brier=float(S[m]['brier'].sum() / Wm.sum()), logloss=float(S[m]['logloss'].sum() / Wm.sum())) for m in models}
    rng = np.random.default_rng(seed)
    draws = {m: {k: np.full(reps, np.nan) for k in ('auc', 'brier', 'logloss')} for m in models}
    degenerate = 0
    for r in range(reps):
        mult = np.bincount(rng.integers(0, nm, size=nm), minlength=nm).astype(np.float64)
        rw = w * mult[inv]
        den = mult @ Wm
        pos_w = float(np.sum(rw * y))
        if pos_w <= 0 or pos_w >= float(rw.sum()):
            degenerate += 1
        for m in models:
            draws[m]['brier'][r] = float(mult @ S[m]['brier'] / den)
            draws[m]['logloss'][r] = float(mult @ S[m]['logloss'] / den)
            draws[m]['auc'][r] = A[m](rw)

    def ci(a):
        ok = np.isfinite(a)
        return np.quantile(a[ok], [0.025, 0.975]).tolist() if ok.any() else None

    out = dict(replicates=int(reps), seed=int(seed), matches=int(nm), rows=int(len(y)), degenerate_single_class_replicates=int(degenerate),
               unit='match resampled with replacement; row weight = equal-per-match weight of the cell x multiplicity',
               scope='fixed-model evaluation-sample uncertainty only; no training/calibration/selection uncertainty; no multiplicity adjustment',
               model_ci={m: {k: dict(estimate=point[m][k], ci95=ci(draws[m][k]), finite_replicates=int(np.isfinite(draws[m][k]).sum()))
                             for k in ('auc', 'brier', 'logloss')} for m in models}, pairs=[])
    for pr in pairs:
        a, b = pr[0], pr[1]
        d = {}
        for k in ('auc', 'brier', 'logloss'):
            diff = draws[a][k] - draws[b][k]
            ok = np.isfinite(diff)
            est = None if point[a][k] is None or point[b][k] is None or not np.isfinite(point[a][k]) or not np.isfinite(point[b][k]) else point[a][k] - point[b][k]
            d[k] = dict(estimate=est, ci95=ci(diff), finite_replicates=int(ok.sum()),
                        fraction_replicates_a_better=(float(np.mean(diff[ok] > 0)) if k == 'auc' else float(np.mean(diff[ok] < 0))) if ok.any() else None)
        out['pairs'].append(dict(a=a, b=b, label=pr[2] if len(pr) > 2 else None, a_minus_b=d))
    return out


def key_join(g, s, ag, a_s):
    """Index ix with (ag[ix], a_s[ix]) == (g, s) row by row; requires unique keys on both sides and complete 1:1 coverage."""
    g, ag = np.asarray(g).astype(str), np.asarray(ag).astype(str)
    s, a_s = np.asarray(s).astype(np.int64), np.asarray(a_s).astype(np.int64)
    left = list(zip(g.tolist(), s.tolist()))
    right = list(zip(ag.tolist(), a_s.tolist()))
    if len(set(left)) != len(left):
        raise ValueError('duplicate (match, s) keys in evaluation rows')
    if len(set(right)) != len(right):
        raise ValueError('duplicate (match, s) keys in reference rows')
    if len(left) != len(right):
        raise ValueError(f'row count differs: {len(left)} vs {len(right)}')
    if left == right:
        return np.arange(len(left)), 'identical row order'
    pos = {k: i for i, k in enumerate(right)}
    missing = [k for k in left if k not in pos]
    if missing:
        raise ValueError(f'{len(missing)} evaluation keys missing from reference')
    return np.asarray([pos[k] for k in left], dtype=np.int64), 'exact key map'


# ------------------------------------------------------------------ parent data (read-only) with access logging here
def append_jsonl(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(C.clean(rec), ensure_ascii=False) + '\n')


def frozen_path(base=OUT):
    return Path(base) / 'frozen_manifest.json'


def log_access(base, purpose, set_name, kind, sealed):
    """Append-only access record in THIS run; sealed sets (TEST/external, parent TEST predictions) need the freeze."""
    fz = frozen_path(OUT)
    if sealed and not fz.exists():
        raise PermissionError(f'{purpose}: {set_name} is sealed until {fz} exists')
    append_jsonl(Path(base) / 'access_log.jsonl', dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(), purpose=purpose,
                                                       set=set_name, kind=kind, sealed=bool(sealed), frozen_manifest_exists=fz.exists(),
                                                       frozen_manifest_sha256=C.sha256_file(fz) if fz.exists() else None))


LABEL_KEYS = ('match', 's', 'sub_role', 'p_pre', 'valid_h90', 'Y_h90', 'adapter_id', 'adapter_sha256')


def load_parent_set(name, base, purpose):
    """Features (pre-only X_input), h90 labels and cohort arrays of one parent set with exact row-order checks.

    Equivalent to cr20260915_data._load_labeled (same files and checks) plus adapter fields, WITHOUT parent gates/logs.
    """
    if name.startswith('MAIN_TRAIN'):
        sealed = False
    elif name == 'MAIN_VALIDATION':
        sealed = False
    elif name in SEALED_SETS:
        sealed = True
    else:
        raise ValueError(name)
    log_access(base, purpose, name, 'parent features/labels/cohorts npz', sealed)
    with np.load(FC / 'labels' / f'{name}_features_pre_only.npz', allow_pickle=False) as z:
        F = {k: z[k] for k in ('X_input', 'input_names', 'match', 's_ms', 'sub_role', 'pre_ok')}
    with np.load(FC / 'labels' / f'{name}_labels.npz', allow_pickle=False) as z:
        Lb = {k: z[k] for k in LABEL_KEYS if k in z.files}
    with np.load(CR / 'cohorts' / f'{name}_cohort.npz', allow_pickle=False) as z:
        Co = {k: z[k] for k in ('match', 's', 'cohort', 'valid_h90', 'scale_known')}
    checks = dict(
        feature_label_row_order=bool(np.array_equal(F['match'], Lb['match']) and np.array_equal(F['s_ms'], Lb['s'])),
        cohort_label_row_order=bool(np.array_equal(Co['match'].astype(str), Lb['match'].astype(str)) and np.array_equal(Co['s'], Lb['s'])),
        cohort_valid_equals_label_valid=bool(np.array_equal(Co['valid_h90'].astype(int), Lb['valid_h90'].astype(int))))
    ok = F['pre_ok'] == 1
    checks['p_pre_V_column_equals_label_p_pre'] = bool(np.array_equal(F['X_input'][ok, -1], Lb['p_pre'][ok]))
    if not all(checks.values()):
        raise ValueError(f'{name}: parent row contract failed {checks}')
    return F, Lb, Co, checks


def load_trainval(base, smoke=False, cohort=None, need_validation=True):
    """TRAIN (+ VALIDATION Q_CAL/Q_SELECT) h90-valid rows of one cohort (or both) with role labels.

    smoke: TRAIN only; pseudo roles fold0-2 -> TRAIN, fold3 -> Q_CAL, fold4 -> Q_SELECT; hash subset (1/8 of matches).
    """
    parts = ['MAIN_TRAIN'] if smoke or not need_validation else ['MAIN_TRAIN', 'MAIN_VALIDATION']
    D = dict(parts=parts, smoke=bool(smoke), checks={})
    Xs, gs, ss, rs, ys, ps, cs, aids, ashs = [], [], [], [], [], [], [], [], []
    names = None
    for part in parts:
        F, Lb, Co, ch = load_parent_set(part, base, f'fit data {part} ({"smoke TRAIN-only" if smoke else "full"})')
        D['checks'][part] = ch
        nm = [str(x) for x in F['input_names']]
        if names is None:
            names = nm
        elif nm != names:
            raise ValueError('input schema differs between TRAIN and VALIDATION')
        sr = F['sub_role'].astype(str)
        keep = (Lb['valid_h90'] == 1) & (Co['cohort'] >= 0)
        if part == 'MAIN_VALIDATION':
            keep &= np.isin(sr, ['Q_CAL', 'Q_SELECT'])
        if cohort is not None:
            keep &= Co['cohort'] == COHORT_CODE[cohort]
        if not np.all(F['pre_ok'][keep] == 1):
            raise ValueError('valid rows without pre inputs')
        Xs.append(F['X_input'][keep])
        gs.append(F['match'][keep].astype(str))
        ss.append(F['s_ms'][keep])
        rs.append(sr[keep])
        ys.append(Lb['Y_h90'][keep].astype(np.int64))
        ps.append(Lb['p_pre'][keep])
        cs.append(Co['cohort'][keep])
        aids.append(Lb['adapter_id'][keep].astype(str))
        ashs.append(Lb['adapter_sha256'][keep].astype(str))
        F = Lb = Co = None
    D['names'] = names
    D['X'] = np.vstack(Xs)
    D['g'] = np.concatenate(gs)
    D['s_ms'] = np.concatenate(ss)
    D['sr_true'] = np.concatenate(rs)
    D['y'] = np.concatenate(ys)
    D['p_pre'] = np.concatenate(ps)
    D['cohort'] = np.concatenate(cs)
    D['adapter_id'] = np.concatenate(aids)
    D['adapter_sha256'] = np.concatenate(ashs)
    if smoke:
        f = np.asarray([C.train_fold(m) for m in D['g'].tolist()])
        D['role'] = np.where(f <= 2, 'TRAIN', np.where(f == 3, 'Q_CAL', 'Q_SELECT'))
        keep = smoke_keep(D['g'])
        for k in ('X', 'g', 's_ms', 'sr_true', 'y', 'p_pre', 'cohort', 'adapter_id', 'adapter_sha256', 'role'):
            D[k] = D[k][keep]
    else:
        D['role'] = np.where(np.char.startswith(D['sr_true'], 'fold'), 'TRAIN', D['sr_true'])
    return D


def oof_provenance(g, sub_role, adapter_id, adapter_sha256, vman):
    """TRAIN rows: adapter must be the own held-out fold OOF V (fold = train_fold(match)) with the manifest hash."""
    g = np.asarray(g).astype(str)
    u, inv = np.unique(g, return_inverse=True)
    fold_u = np.asarray([C.train_fold(m) for m in u.tolist()])
    fold = fold_u[inv]
    exp_id = np.char.add('oof_fold', fold.astype(str))
    exp_sha = np.asarray([vman['oof_sha256'][f'fold{k}'] for k in range(C.N_FOLDS)])[fold]
    sr_fold = np.asarray(sub_role).astype(str)
    return dict(rows=int(len(g)),
                adapter_id_equals_own_heldout_fold=int(np.sum(np.asarray(adapter_id).astype(str) == exp_id)),
                adapter_sha_equals_manifest_oof_hash=int(np.sum(np.asarray(adapter_sha256).astype(str) == exp_sha)),
                sub_role_equals_hash_fold=int(np.sum(sr_fold == np.char.add('fold', fold.astype(str)))))


# ------------------------------------------------------------------ status / logging (this root only)
class Status:
    def __init__(self, group, out=OUT):
        self.out = Path(out)
        self.group = group
        self.t0 = time.time()
        (self.out / 'status').mkdir(parents=True, exist_ok=True)
        (self.out / 'logs').mkdir(parents=True, exist_ok=True)
        self.log_path = self.out / 'logs' / f'{group}.log'
        self.history = []

    def log(self, msg):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")} +{time.time() - self.t0:8.1f}s] [{self.group}] {msg}'
        print(line, flush=True)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    def update(self, state, stage, processed=None, total=None, error=None, next_step=None, **extra):
        rec = dict(group=self.group, state=state, stage=stage, pid=os.getpid(), updated=time.strftime('%Y-%m-%d %H:%M:%S'),
                   elapsed_s=round(time.time() - self.t0, 1), processed=processed, total=total, error=error,
                   next_step=next_step, **extra)
        if not self.history or self.history[-1]['stage'] != stage or self.history[-1]['state'] != state:
            self.history.append(dict(stage=stage, state=state, at=rec['updated']))
        rec['history'] = self.history[-100:]
        C.write_json(self.out / 'status' / f'{self.group}.json', rec)
        aggregate_status(self.out)
        self.log(f'STATUS {state} {stage} processed={processed} total={total} error={error} next={next_step} '
                 + json.dumps(C.clean(extra), ensure_ascii=False)[:600])


def aggregate_status(out=OUT):
    out = Path(out)
    groups = {}
    for p in sorted((out / 'status').glob('*.json')):
        try:
            groups[p.stem] = json.loads(p.read_text(encoding='utf-8'))
        except Exception as exc:  # concurrent writer; next update refreshes
            groups[p.stem] = {'read_error': repr(exc)}
    agg = dict(version=VERSION, role=ROLE_TAG, updated=time.strftime('%Y-%m-%d %H:%M:%S'),
               running_groups=[g for g, r in groups.items() if r.get('state') == 'running'],
               failed_groups=[g for g, r in groups.items() if r.get('state') == 'failed'],
               groups={g: {k: r.get(k) for k in ('state', 'stage', 'processed', 'total', 'pid', 'elapsed_s', 'error', 'next_step', 'updated')}
                       for g, r in groups.items()},
               state_vocabulary='running; complete = stage finished; failed (attempt logs retained in logs/ and failures.jsonl); '
                                'verified only via validation.json',
               failed_run_history='logs/*.log, commands.txt and failures.jsonl keep every attempt')
    try:
        C.write_json(out / 'status.json', agg)
    except PermissionError:
        pass


def log_command(argv=None, note=''):
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    line = ' '.join([PYTHON, '-B'] + list(argv if argv is not None else sys.argv))
    with open(OUT / 'commands.txt', 'a', encoding='utf-8') as f:
        f.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} pid={os.getpid()} {line} {note}'.rstrip() + '\n')


def log_failure(stage, error, **info):
    append_jsonl(OUT / 'failures.jsonl', dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(), stage=stage, error=str(error)[:4000], **info))


def environment():
    import platform
    import sys
    import joblib
    import lightgbm
    import scipy
    import sklearn
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(), numpy=np.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__, lightgbm=lightgbm.__version__, joblib=joblib.__version__,
                threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                dont_write_bytecode=bool(sys.dont_write_bytecode), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def own_source_hashes():
    files = sorted((ROOT / 'scripts').glob('iq20260915_*.py')) + sorted((ROOT / 'tests').glob('test_iq20260915_*.py'))
    reused = [ROOT / 'scripts' / 'fc20260915_common.py', ROOT / 'scripts' / 'cr20260915_common.py', SPEC, *PROVENANCE_DOCS]
    return {p.relative_to(ROOT).as_posix(): C.sha256_file(p) for p in files + reused if p.exists()}


def threads_env(n=4):
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[v] = str(n)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
