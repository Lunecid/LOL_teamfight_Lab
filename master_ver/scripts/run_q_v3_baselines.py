"""P4: pre-engagement q baselines and explanation on P3 candidate labels (exploratory).

Codex design: docs/CLAUDE_EXECUTE_P4_Q_BASELINES.md. Claude Opus 5 implementation and execution.

q(X_pre) = P(delta_A_h > 0 | pre-state, retrospectively selected engagement cohort). Labels are outcomes
defined by frozen win-probability models, not observed engagement success. Outputs go to
outputs/q_v3_baselines only.

scripts/run_validation_suite.py is NOT imported: importing it creates its old output directory. The
helpers reused from it (weights, score, logit, calibration, the economic name rule and the SHAP grouping)
are restored verbatim below, with only the documented name mapping applied.
"""
import os

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'  # BLAS single-threaded; LightGBM alone uses n_jobs=4 (no nesting)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import hashlib
import json
import math
import platform
import re
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT / 'outputs' / 'engagement_labels_v3_sensitivity'
VS = ROOT / 'outputs' / 'validation_suite_20260914'
P1 = ROOT / 'outputs' / 'independent_v2_participant_order'
P2 = ROOT / 'outputs' / 'temporal_winprob_v3'
PA = ROOT / 'outputs' / 'position_ablation_20260914'
WT = ROOT / 'worktrees' / 'engagement-state-value'
VD = WT / 'outputs' / 'temporal_winprob_v3_buckets'
OLD_EVAL = WT / 'outputs' / 'state_value_main_50k_eval'
OUT_BASE = ROOT / 'outputs' / 'q_v3_baselines'
SPEC = ROOT / 'docs' / 'CLAUDE_EXECUTE_P4_Q_BASELINES.md'
REF_REGISTRY = ROOT / 'docs' / 'LITERATURE_GUIDED_NEXT_STEPS_20260914.md'

VERSION = 'q_v3_baselines_p4'
ROLE = 'EXPLORATORY_Q_BASELINES_ON_CANDIDATE_LABELS_NOT_GROUND_TRUTH_NOT_PRODUCTION'
TARGETS = {'A90': ('A', 90), 'A60': ('A', 60), 'A120': ('A', 120), 'B90': ('B', 90)}
PRIMARY = 'A90'
SPLITS = ('train', 'calibrate', 'select', 'test')
EXPECTED = {'train': (9228, 3218), 'calibrate': (4879, 1669), 'select': (4922, 1656), 'test': (7664, 2655)}
SPLIT_PATCH = {'train': '15.14', 'calibrate': '15.15', 'select': '15.15', 'test': '15.16'}
OLD_SPLIT_NAMES = {'train': 'train', 'calibrate': 'calibration_fit', 'select': 'development',
                   'test': 'test_exploratory'}
CANDIDATES = ('constant', 'p_pre_logistic', 'p_pre_spline', 'ridge_raw', 'ridge_sigmoid', 'ridge_isotonic',
              'economic_raw', 'economic_sigmoid', 'economic_isotonic')
ECON_SEEDS = (7, 42, 123)
LGBM_PARAMS = dict(n_estimators=250, num_leaves=15, learning_rate=.04, min_child_samples=100, reg_lambda=1,
                   n_jobs=4, verbosity=-1, colsample_bytree=.9)
BOOT_SEED = 20260915
P_PRE = 'p_pre_A'
TIME_BANDS = ((0, 2), (2, 10), (10, 20), (20, 30), (30, 1000))  # P3 start_time_bands_minutes
PRE_OBJECTIVES = ('baron', 'elder', 'herald', 'horde', 'atakhan', 'dragon', 'soul',
                  'dragon_AIR', 'dragon_EARTH', 'dragon_FIRE', 'dragon_WATER', 'dragon_HEXTECH',
                  'dragon_CHEMTECH', 'dragon_OTHER')
P3_OBJECTIVES = ('baron', 'dragon', 'elder', 'herald', 'horde', 'atakhan', 'soul_owned', 'soul_teamid0_unassigned',
                 'dragon_AIR', 'dragon_EARTH', 'dragon_FIRE', 'dragon_WATER', 'dragon_HEXTECH', 'dragon_CHEMTECH',
                 'dragon_OTHER', 'tower', 'inhibitor', 'plate')
FORBIDDEN_TOKENS = ('snapshot_age', 'p_post', 'delta', 'endpoint', 'duration', 'label', 'winner', 'game_end',
                    'next_kill', 'next_start', 'reason', 'horizon', 'during_', 'after_last', 'position', 'xy_',
                    'champion_id', 'match', 'patch', 'p_pre_B', 'Y_A', 'Y_B', 'L_ms', 'first_kill')


# ----------------------------------------------------------------------------------------------------------
# Restored verbatim from scripts/run_validation_suite.py (not imported; see module docstring).
def weights(g):
    _, ix, c = np.unique(g, return_inverse=True, return_counts=True)
    w = 1 / c[ix]; return w / w.mean()


def score(y, p, w):
    return dict(auc=float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) > 1 else None,
                brier=float(brier_score_loss(y, p, sample_weight=w)),
                logloss=float(log_loss(y, p, labels=[0, 1], sample_weight=w)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8); return np.log(p / (1 - p)).reshape(-1, 1)


def calibration(y, p, w):
    out = score(y, p, w); out.update(n=len(y), observed=float(np.average(y, weights=w)),
                                     predicted=float(np.average(p, weights=w)))
    if len(np.unique(y)) > 1 and np.std(p) > 1e-8:
        m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p), y, sample_weight=w)
        out.update(intercept=float(m.intercept_[0]), slope=float(m.coef_[0, 0]))
    bins = []
    for lo in np.arange(0, 1, .1):
        a = (p >= lo) & (p < lo + .1 if lo < .85 else p <= 1)
        if a.any(): bins.append(dict(lo=float(lo), n=int(a.sum()), predicted=float(np.average(p[a], weights=w[a])),
                                     observed=float(np.average(y[a], weights=w[a]))))
    out['bins'] = bins; return out


ECON_SLOT_SUFFIXES = ['totalGold_norm', 'curGold_norm', 'level_norm', 'xp_norm', 'alive', 'laneCS_norm', 'jgCS_norm',
                      'kills', 'deaths']


def _team_aggregate_rule(n):
    return ((n.startswith('blue_') or n.startswith('red_')) and not n.endswith('_x_time') and 'age_' not in n
            and 'last_' not in n and not n.endswith('_ever'))


def old_econ_rule(n):
    """Historic selection (run_validation_suite.q_experiment), old 'slotK_' names, includes snapshot_age_s."""
    return (n in ['time_minutes', 'snapshot_age_s']
            or (n.startswith('slot') and any(n.endswith(a) for a in ECON_SLOT_SUFFIXES)) or _team_aggregate_rule(n))


def econ_rule(n):
    """Same rule mapped to 'participant_slotK_' names with snapshot_age_s removed."""
    return (n == 'time_minutes'
            or (n.startswith('participant_slot') and any(n.endswith(a) for a in ECON_SLOT_SUFFIXES))
            or _team_aggregate_rule(n))


def shap_group(n):
    """Historic grouping (run_validation_suite.shap_audit); p_pre -> p_pre_A, snapshot_age_s absent."""
    if n == P_PRE: return 'prior_win_probability'
    if n.startswith('time_') or n == 'snapshot_age_s': return 'time_and_observation_age'
    if any(k in n for k in ['baron', 'elder', 'dragon', 'soul', 'herald', 'horde', 'atakhan']): return 'objectives'
    if any(k in n for k in ['Gold', 'xp_', 'level_', 'CS_']): return 'economy_and_experience'
    if any(k in n for k in ['kills', 'deaths', 'death_', 'alive']): return 'combat_and_survival'
    if any(k in n for k in ['tower', 'inhibitor', 'plates']): return 'structures'
    if 'champion_id' in n: return 'champion_identity'
    return 'health_mana_other'
# ----------------------------------------------------------------------------------------------------------


def feature_unit(n):
    """Units from worktrees/engagement-state-value/gameplay/state_value.py (StateBuilder.at)."""
    base = n[:-len('_x_time')] if n.endswith('_x_time') else n
    suffix = ' multiplied by query_minutes/30 (phase interaction)' if n.endswith('_x_time') else ''
    if n == P_PRE: return 'probability of final Blue win from frozen P1 model A at q_pre = s-1 ms'
    if n == 'time_minutes': return 'minutes (q_pre / 60000)'
    if n == 'time_minutes_sq': return 'minutes squared'
    if n == 'unknown_objective_team_count': return 'count of objective/structure/soul events without a Blue/Red team up to q_pre'
    m = re.match(r'participant_slot(\d)_(.+)$', n)
    if m:
        f = m.group(2)
        if f.endswith('_norm'): return 'normalized cache value at last frame <= q_pre (unitless)'
        if f.endswith('_pct'): return 'fraction [0, 1] at last frame <= q_pre'
        if f == 'alive': return 'binary at last frame <= q_pre'
        if f in ('kills', 'deaths'): return 'count up to q_pre'
        if f == 'death_age_minutes': return 'minutes since last death, capped at 10 (10 if none)'
        if f == 'champion_id': return 'categorical identifier (excluded)'
        return 'binary indicator up to q_pre'
    if base.endswith('_age_minutes'): return 'minutes since last acquisition, capped at 10' + suffix
    if '_acquired_last_' in base or base.endswith('_ever') or '_soul_' in base: return 'binary indicator' + suffix
    return 'count up to q_pre' + suffix


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''): h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def rank_key(tag, match, s):
    return hashlib.sha256(f'{tag}:{match}:{int(s)}'.encode()).hexdigest()


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o); return v if math.isfinite(v) else None
    if isinstance(o, Path): return str(o)
    return o


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(clean(obj), indent=1, ensure_ascii=False, allow_nan=False).encode('utf-8')
    path.write_bytes(data)  # bytes: no Windows newline translation, so the returned hash equals the file hash
    return sha256_bytes(data)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class Run:
    def __init__(self, out, smoke):
        self.out = out; self.smoke = smoke; self.t0 = time.time(); self.history = []
        (out / 'logs').mkdir(parents=True, exist_ok=True)
        self.log_path = out / 'logs' / 'run.log'
        self.test_prediction_calls = 0

    def log(self, msg):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")} +{time.time() - self.t0:7.1f}s] {msg}'
        print(line, flush=True)
        with open(self.log_path, 'a', encoding='utf-8') as f: f.write(line + '\n')

    def status(self, state, stage, **extra):
        self.history.append(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), elapsed_s=round(time.time() - self.t0, 1),
                                 state=state, stage=stage, **extra))
        write_json(self.out / 'status.json', dict(version=VERSION, role=ROLE, smoke=self.smoke, state=state,
                                                  stage=stage, history=self.history))
        self.log(f'STATUS {state} {stage} {extra if extra else ""}')


# ----------------------------------------------------------------------------------------------------------
# Frozen inputs
def frozen_files():
    files = []
    for d in (P3, VS):
        files += sorted(p for p in d.iterdir() if p.is_file())
    files += sorted(p for p in (P3 / 'audit_only').iterdir() if p.is_file())
    files += sorted(p for p in (P1 / 'models').rglob('*') if p.is_file())
    files += [P1 / 'protocol.json', P1 / 'selection.json', P1 / 'validation.json', P1 / 'results.json',
              P2 / 'protocol.json', P2 / 'selection.json', VD / 'protocol.json', OLD_EVAL / 'match_splits.json',
              PA / 'feature_names.json', ROOT / 'scripts' / 'run_validation_suite.py',
              ROOT / 'scripts' / 'run_engagement_labels_v3_sensitivity.py',
              ROOT / 'scripts' / 'engagement_labels_v3_rules.py', ROOT / 'scripts' / 'train_independent_v2.py',
              WT / 'gameplay' / 'state_value.py', SPEC, REF_REGISTRY]
    return [p for p in files if p.exists()]


def hash_frozen():
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha256_file(p) for p in frozen_files()}


# ----------------------------------------------------------------------------------------------------------
# Feature sets
def feature_sets(names, categorical):
    names = list(names); categorical = set(categorical)
    ridge = [n for n in names if not n.endswith('champion_id') and n not in categorical]
    econ = [n for n in names if econ_rule(n)]
    return {'p_pre': [P_PRE], 'ridge': ridge, 'economic': econ}


def schema_mapping_checks(names, sets):
    old_names = read_json(PA / 'feature_names.json')['baseline']
    q_old = read_json(VS / 'q_results.json')
    to_new = lambda n: re.sub(r'^slot(\d)_', r'participant_slot\1_', n)
    old_econ = [n for n in old_names if old_econ_rule(n)]
    mapped_econ = [to_new(n) for n in old_econ if n != 'snapshot_age_s']
    mapped_ridge = [P_PRE if n == 'p_pre' else to_new(n) for n in q_old['ridge_names'] if n != 'snapshot_age_s']
    mapped_all = [P_PRE if n == 'p_pre' else to_new(n) for n in old_names if n != 'snapshot_age_s']
    out = dict(
        old_baseline_names=len(old_names), new_input_names=len(names),
        old_econ_rule_reproduces_saved_economic_names=old_econ == q_old['economic_names'],
        old_economic_names=len(q_old['economic_names']), new_economic_names=len(sets['economic']),
        economic_equals_old_mapped_minus_snapshot_age=mapped_econ == sets['economic'],
        old_ridge_names=len(q_old['ridge_names']), new_ridge_names=len(sets['ridge']),
        ridge_equals_old_mapped_minus_snapshot_age=mapped_ridge == sets['ridge'],
        all_inputs_equal_old_baseline_mapped_minus_snapshot_age=mapped_all == list(names),
        snapshot_age_in_old_economic='snapshot_age_s' in q_old['economic_names'],
        snapshot_age_in_old_ridge='snapshot_age_s' in q_old['ridge_names'],
        old_minus_new=[n for n in old_names if (P_PRE if n == 'p_pre' else to_new(n)) not in names],
        new_minus_old=[n for n in names if n not in [(P_PRE if m == 'p_pre' else to_new(m)) for m in old_names]],
        old_p_pre_source='V1 expanded (worktree temporal_winprob_v3_buckets) at s-1',
        new_p_pre_source='P1 model A (independent_v2_participant_order expanded_model_v2.joblib) at s-1')
    out['pass'] = bool(out['old_econ_rule_reproduces_saved_economic_names']
                       and out['economic_equals_old_mapped_minus_snapshot_age']
                       and out['ridge_equals_old_mapped_minus_snapshot_age'])
    return out


# ----------------------------------------------------------------------------------------------------------
# Data
LABEL_KEYS = ('row_index', 'match', 'patch', 'h_s', 's_ms', 'q_pre_ms', 'L_ms', 'endpoint_ms', 'valid',
              'pre_snapshot_ms', 'pre_snapshot_age_s', 'p_pre_A', 'p_post_A', 'delta_A', 'Y_A', 'delta_A_exact_zero',
              'p_pre_B', 'p_post_B', 'delta_B', 'Y_B', 'delta_B_exact_zero', 'model_A_sha256',
              'model_B_bundle_sha256', 'model_A_version', 'model_B_version', 'history_version', 'state_version',
              'source_cache_files_sha256', 'role')


def load_data():
    with np.load(P3 / 'features_pre_only.npz', allow_pickle=False) as z:
        F = {k: z[k] for k in z.files}
    with np.load(P3 / 'labels_long.npz', allow_pickle=False) as z:
        keys = list(LABEL_KEYS) + [k for k in z.files if k.startswith('during_') or k.startswith('after_last_kill_')]
        Lb = {k: z[k] for k in keys}
    return F, Lb


def split_masks(match, patch):
    half = np.array([int(hashlib.sha256(('calibration17:' + m).encode()).hexdigest()[:8], 16) % 2 for m in match])
    va = patch == '15.15'
    return {'train': patch == '15.14', 'calibrate': va & (half == 0), 'select': va & (half == 1),
            'test': patch == '15.16'}


def verify_data(run, F, Lb, sets):
    checks = {}; info = {}
    n = len(F['row_index']); names = [str(x) for x in F['input_names']]
    match = F['match'].astype(str); s = F['s_ms']; patch = F['patch_split_metadata'].astype(str)
    X = F['X_input']
    checks['features_rows_26693'] = n == 26693
    checks['features_row_index_is_order'] = bool(np.array_equal(F['row_index'], np.arange(n)))
    checks['features_pre_query_is_s_minus_1'] = bool(np.array_equal(F['pre_query_ms'], s - 1))
    checks['features_role_pre_only'] = str(F['role']) == 'PRE_ONLY_INPUTS_EXPLORATORY'
    checks['features_finite'] = bool(np.isfinite(X).all())
    schema = read_json(P3 / 'feature_schema.json')
    cols = schema['files']['features_pre_only.npz']['X_input']['columns']
    checks['schema_names_equal_npz'] = [c['name'] for c in cols] == names
    checks['schema_feature_guards_pass'] = bool(schema['feature_guards']['pass'])
    checks['schema_snapshot_age_not_in_inputs'] = schema['feature_guards']['snapshot_age_s_in_inputs'] is False
    cat = [c['name'] for c in cols if c['role'] == 'input_categorical']
    checks['schema_categorical_equals_npz'] = cat == [str(x) for x in F['categorical_input_names']]
    # Label rows per horizon: exact P3 order.
    per_h = {}
    for h in (60, 90, 120):
        sel = np.flatnonzero(Lb['h_s'] == h)
        per_h[h] = sel
        checks[f'labels_h{h}_row_index_order'] = bool(np.array_equal(Lb['row_index'][sel], np.arange(n)))
        checks[f'labels_h{h}_match_equal'] = bool(np.array_equal(Lb['match'][sel].astype(str), match))
        checks[f'labels_h{h}_s_equal'] = bool(np.array_equal(Lb['s_ms'][sel], s))
        checks[f'labels_h{h}_patch_equal'] = bool(np.array_equal(Lb['patch'][sel].astype(str), patch))
        checks[f'labels_h{h}_valid_all'] = bool((Lb['valid'][sel] == 1).all())
        for mdl in ('A', 'B'):
            d = Lb['delta_' + mdl][sel]; yy = Lb['Y_' + mdl][sel]
            checks[f'labels_h{h}_{mdl}_Y_equals_delta_gt_0'] = bool(np.array_equal(yy, (d > 0).astype(yy.dtype)))
            checks[f'labels_h{h}_{mdl}_exact_zero_flag_consistent'] = bool(
                np.array_equal(Lb[f'delta_{mdl}_exact_zero'][sel], (d == 0).astype(int)))
            info[f'h{h}_{mdl}_exact_zero_rows'] = int((d == 0).sum())
            info[f'h{h}_{mdl}_exact_zero_labelled_0'] = int(((d == 0) & (yy == 0)).sum())
            checks[f'labels_h{h}_{mdl}_finite'] = bool(np.isfinite(d).all())
        checks[f'labels_h{h}_pre_snapshot_not_future'] = bool((Lb['pre_snapshot_ms'][sel] <= Lb['q_pre_ms'][sel]).all())
        checks[f'labels_h{h}_p_pre_A_equals_X_column'] = bool(np.array_equal(Lb['p_pre_A'][sel], X[:, names.index(P_PRE)]))
    # Round-trip CSV cross-check of the used label columns.
    usecols = ['row_index', 'match', 'h_s', 's_ms', 'delta_A', 'Y_A', 'delta_B', 'Y_B', 'p_pre_A']
    csv = pd.read_csv(P3 / 'labels_long.csv', usecols=usecols, float_precision='round_trip', dtype={'match': str})
    checks['labels_csv_rows_equal_npz'] = len(csv) == len(Lb['row_index'])
    if checks['labels_csv_rows_equal_npz']:
        for c in usecols:
            a = csv[c].to_numpy(); b = Lb[c]
            checks[f'labels_csv_roundtrip_{c}_equal'] = bool(np.array_equal(a.astype(str), b.astype(str)) if c == 'match'
                                                             else np.array_equal(a, b))
    csv = None
    # Old validation-suite rows and q predictions: exact keys/patches/order.
    old_rows = pd.read_csv(VS / 'rows.csv', dtype={'patch': str, 'match': str})
    old_q = pd.read_csv(VS / 'q_predictions.csv', dtype={'patch': str, 'match': str})
    for key, df in (('rows_csv', old_rows), ('q_predictions_csv', old_q)):
        checks[f'old_{key}_rows_equal'] = len(df) == n
        checks[f'old_{key}_match_order_equal'] = bool(len(df) == n and np.array_equal(df.match.to_numpy().astype(str), match))
        checks[f'old_{key}_s_order_equal'] = bool(len(df) == n and np.array_equal(df.s.to_numpy(), s))
        checks[f'old_{key}_patch_equal'] = bool(len(df) == n and np.array_equal(df.patch.to_numpy().astype(str), patch))
        checks[f'old_{key}_keys_unique'] = not bool(df.duplicated(['match', 's']).any())
    checks['old_rows_and_q_predictions_y_equal'] = bool(np.array_equal(old_rows.y.to_numpy(), old_q.y.to_numpy()))
    # Split recovery.
    M = split_masks(match, patch)
    M_old = split_masks(old_q.match.to_numpy().astype(str), old_q.patch.to_numpy().astype(str))
    q_old = read_json(VS / 'q_results.json')
    counts = {k: dict(rows=int(M[k].sum()), matches=int(len(set(match[M[k]])))) for k in SPLITS}
    info['split_counts'] = counts
    for k in SPLITS:
        checks[f'split_{k}_counts_expected'] = (counts[k]['rows'], counts[k]['matches']) == EXPECTED[k]
        checks[f'split_{k}_equals_old_q_predictions_split'] = bool(np.array_equal(M[k], M_old[k]))
        olds = [m for m in q_old['metrics'] if m['model'] == 'constant' and m['split'] == OLD_SPLIT_NAMES[k]]
        checks[f'split_{k}_rows_equal_saved_q_results_metrics_n'] = len(olds) == 1 and olds[0]['n'] == counts[k]['rows']
        checks[f'split_{k}_patch_membership'] = bool((patch[M[k]] == SPLIT_PATCH[k]).all())
    for k, old in (('calibrate', 'calibration_fit'), ('select', 'selection')):
        checks[f'split_{k}_equals_saved_development_partition'] = (
            q_old['development_partition'][old] == counts[k])
    cover = sum(M[k].astype(int) for k in SPLITS)
    checks['splits_cover_all_rows_once'] = bool((cover == 1).all())
    msets = {k: set(match[M[k]]) for k in SPLITS}
    inter = {f'{a}&{b}': len(msets[a] & msets[b]) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}
    info['q_partition_match_intersections'] = inter
    checks['q_partitions_mutually_disjoint_by_match'] = all(v == 0 for v in inter.values())
    # Zero overlap with every V partition.
    allq = set(match)
    vparts = {}
    for name, path in (('v1_buckets', VD / 'protocol.json'), ('p1', P1 / 'protocol.json'), ('p2', P2 / 'protocol.json')):
        sp = read_json(path)['splits']
        vparts[name] = {k: len(allq & set(sp[k])) for k in ('fit', 'calibrate', 'select', 'test')}
        checks[f'q_matches_subset_of_{name}_engagement_split'] = allq <= set(sp['engagement'])
    ev = read_json(OLD_EVAL / 'match_splits.json')
    vparts['v1_eval'] = {k: len(allq & set(ev[k])) for k in ('value_train', 'value_validation', 'predict_train')}
    info['v_partition_overlap'] = vparts
    checks['zero_match_overlap_with_all_V_partitions'] = all(v == 0 for d in vparts.values() for v in d.values())
    # Predictor guards.
    pred_names = sorted(set(sets['ridge']) | set(sets['economic']) | set(sets['p_pre']))
    checks['predictors_subset_of_p3_inputs'] = set(pred_names) <= set(names)
    checks['predictors_no_forbidden_tokens'] = not [n_ for n_ in pred_names if any(t in n_ for t in FORBIDDEN_TOKENS)]
    checks['predictors_no_categorical_ids'] = not (set(pred_names) & set(cat))
    checks['predictors_no_snapshot_age'] = not [n_ for n_ in pred_names if 'snapshot_age' in n_]
    idx = [names.index(n_) for n_ in pred_names]
    leak = []
    age = Lb['pre_snapshot_age_s'][per_h[90]]
    for h in (60, 90, 120):
        sel = per_h[h]
        for tgt in ('delta_A', 'Y_A', 'delta_B', 'Y_B', 'p_post_A', 'L_ms', 'endpoint_ms'):
            v = Lb[tgt][sel].astype(float)
            leak += [f'{names[j]}=={tgt}@{h}' for j in idx if np.array_equal(X[:, j], v)]
    leak += [f'{names[j]}==pre_snapshot_age_s' for j in idx if np.array_equal(X[:, j], age)]
    info['predictor_columns_identical_to_label_or_future_columns'] = leak
    checks['no_predictor_column_equals_label_future_or_snapshot_age_column'] = not leak
    info['old_y_vs_new_labels'] = {}
    for t, (mdl, h) in TARGETS.items():
        yy = Lb['Y_' + mdl][per_h[h]]
        oy = old_q.y.to_numpy()
        info['old_y_vs_new_labels'][t] = dict(
            agreement_all=float(np.mean(oy == yy)),
            agreement_by_split={k: float(np.mean(oy[M[k]] == yy[M[k]])) for k in SPLITS},
            new_positive_rate=float(yy.mean()), old_positive_rate=float(oy.mean()))
    failed = [k for k, v in checks.items() if not v]
    return dict(checks=checks, failed=failed, info=info, pass_=not failed), M, per_h, old_q


# ----------------------------------------------------------------------------------------------------------
# Models
def col_index(names, sets):
    return {k: [names.index(n) for n in v] for k, v in sets.items()}


def predict_bundle(b, X):
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
        if kind == 'economic_raw': return raw
    else:
        raise ValueError(kind)
    if kind.endswith('_sigmoid'): return b['calibrator'].predict_proba(logit(raw))[:, 1]
    if kind.endswith('_isotonic'): return b['calibrator'].predict(raw)
    raise ValueError(kind)


def manifest(target, sets_sha, schema_sha, labels_sha, A_sha, B_sha):
    mdl, h = TARGETS[target]
    return dict(role=ROLE, q_version=VERSION, output_is='q probability of candidate label Y (NOT a V win probability)',
                target=target, label_definition=f'Y_{mdl}_h{h} = 1(delta_{mdl} > 0 at B-rule endpoint h={h}s); exact zero -> 0',
                label_source='outputs/engagement_labels_v3_sensitivity/labels_long.npz', label_source_sha256=labels_sha,
                label_model=mdl, horizon_s=h,
                label_model_sha256=A_sha if mdl == 'A' else B_sha,
                p_pre_A_input_model_sha256=A_sha,
                input_features='outputs/engagement_labels_v3_sensitivity/features_pre_only.npz',
                input_schema_sha256=schema_sha, predictor_sets_sha256=sets_sha)


def fit_target(run, target, X, y, g, M, cix, sets, man):
    tr, ca = M['train'], M['calibrate']
    w, cw = weights(g[tr]), weights(g[ca])
    B = {}; fitlog = {}
    common = lambda cand, key: dict(candidate=cand, manifest=man, input_set=key, input_columns=cix[key],
                                    input_names=sets[key])
    B['constant'] = dict(candidate='constant', manifest=man, input_set=None, input_columns=[], input_names=[],
                         prior=float(np.average(y[tr], weights=w)))
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always')
        for cand, pipe in (('p_pre_logistic', make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000))),
                           ('p_pre_spline', make_pipeline(SplineTransformer(n_knots=5, degree=3), StandardScaler(),
                                                          LogisticRegression(C=1, max_iter=2000)))):
            pipe.fit(X[tr][:, cix['p_pre']], y[tr], logisticregression__sample_weight=w)
            B[cand] = dict(common(cand, 'p_pre'), pipeline=pipe, fitted_on='train')
        ridge = make_pipeline(StandardScaler(), LogisticRegression(C=.01, max_iter=3000))
        ridge.fit(X[tr][:, cix['ridge']], y[tr], logisticregression__sample_weight=w)
        B['ridge_raw'] = dict(common('ridge_raw', 'ridge'), pipeline=ridge, fitted_on='train')
        from lightgbm import LGBMClassifier
        seeds = []
        for seed in ECON_SEEDS:
            m = LGBMClassifier(random_state=seed, **LGBM_PARAMS)
            m.fit(X[tr][:, cix['economic']], y[tr], sample_weight=w); seeds.append(m)
        B['economic_raw'] = dict(common('economic_raw', 'economic'), seed_models=seeds, seeds=list(ECON_SEEDS),
                                 fitted_on='train')
        raw_ca = {'ridge': predict_bundle(B['ridge_raw'], X[ca]), 'economic': predict_bundle(B['economic_raw'], X[ca])}
        for base in ('ridge', 'economic'):
            p = raw_ca[base]
            sg = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p), y[ca], sample_weight=cw)
            iso = IsotonicRegression(out_of_bounds='clip').fit(p, y[ca], sample_weight=cw)
            extra = dict(base_pipeline=ridge) if base == 'ridge' else dict(seed_models=seeds, seeds=list(ECON_SEEDS))
            B[base + '_sigmoid'] = dict(common(base + '_sigmoid', base), calibrator=sg, **extra,
                                        fitted_on='base: train; calibrator: calibrate')
            B[base + '_isotonic'] = dict(common(base + '_isotonic', base), calibrator=iso, **extra,
                                         fitted_on='base: train; calibrator: calibrate')
    fitlog['warnings'] = sorted(set(f'{x.category.__name__}: {str(x.message)[:200]}' for x in wlist))
    tot = np.bincount(np.unique(g[tr], return_inverse=True)[1], weights=w)
    fitlog['train_weight'] = dict(rows=int(tr.sum()), sum=float(w.sum()), mean=float(w.mean()),
                                  per_match_total_min=float(tot.min()), per_match_total_max=float(tot.max()))
    fitlog['calibrate_weight'] = dict(rows=int(ca.sum()), sum=float(cw.sum()), mean=float(cw.mean()))
    fitlog['ridge_n_iter'] = int(ridge[-1].n_iter_[0])
    fitlog['p_pre_logistic_n_iter'] = int(B['p_pre_logistic']['pipeline'][-1].n_iter_[0])
    fitlog['p_pre_spline_n_iter'] = int(B['p_pre_spline']['pipeline'][-1].n_iter_[0])
    fitlog['constant_prior'] = B['constant']['prior']
    return B, fitlog


def evaluate(y, p, g, bins=True):
    if len(y) == 0:
        return dict(n=0, matches=0, auc=None, brier=None, logloss=None, auc_na_reason='empty')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = calibration(y, p, weights(g))
    if not bins: out.pop('bins')
    out['matches'] = int(len(np.unique(g))); out['positives'] = int(y.sum())
    out['auc_na_reason'] = None if out['auc'] is not None else 'single class'
    if 'slope' not in out:
        out['intercept'] = None; out['slope'] = None
        out['calibration_na_reason'] = 'single class' if len(np.unique(y)) < 2 else 'constant prediction'
    out['citl_intercept_offset'] = citl(y, p, weights(g))
    out['sparse_lt30_matches'] = out['matches'] < 30
    return out


def citl(y, p, w):
    """Calibration-in-the-large: intercept with logit(p) as offset (slope fixed at 1); weighted Newton."""
    if len(np.unique(y)) < 2: return None
    o = logit(p).ravel(); a = 0.0
    for _ in range(200):
        mu = 1 / (1 + np.exp(-(a + o))); grad = np.sum(w * (y - mu)); hess = np.sum(w * mu * (1 - mu))
        if hess <= 0: return None
        step = grad / hess; a += step
        if abs(step) < 1e-13: break
    return float(a)


# Independent metric reconstruction (no sklearn metric helpers).
def auc_direct(y, p, w):
    pos = y == 1
    if pos.all() or not pos.any(): return None
    u, inv = np.unique(p, return_inverse=True)
    wp = np.bincount(inv, weights=w * pos, minlength=len(u)); wn = np.bincount(inv, weights=w * ~pos, minlength=len(u))
    below = np.cumsum(wn) - wn
    return float(np.sum(wp * (below + 0.5 * wn)) / (wp.sum() * wn.sum()))


def brier_direct(y, p, w):
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def logloss_direct(y, p, w):
    eps = np.finfo(np.float64).eps; q = np.clip(p, eps, 1 - eps)
    return float(-np.sum(w * (y * np.log(q) + (1 - y) * np.log(1 - q))) / np.sum(w))


# ----------------------------------------------------------------------------------------------------------
# Strata on TEST (weights renormalized within stratum, as the existing V audit convention).
def strata(F_names, X, Lb_h, s_ms, mask):
    names = F_names; out = [('overall', mask.copy())]
    minutes = s_ms / 60000.
    for lo, hi in TIME_BANDS:
        out.append((f'start_minutes_{lo}_{hi}', mask & (minutes >= lo) & (minutes < hi)))
    def col(n): return X[:, names.index(n)]
    for obj in PRE_OBJECTIVES:
        if obj == 'dragon': v = col('blue_dragons') + col('red_dragons')
        elif obj == 'soul': v = col('blue_soul_event_recorded') + col('red_soul_event_recorded')
        else: v = col('blue_' + obj) + col('red_' + obj)
        for has in (False, True):
            out.append((f'pre_history_{obj}_{"acquired" if has else "not_acquired"}', mask & ((v > 0) == has)))
    for window, prefix in (('during', 'during_'), ('after_last_kill', 'after_last_kill_')):
        for obj in P3_OBJECTIVES:
            v = Lb_h[prefix + obj]
            for has in (False, True):
                out.append((f'posthoc_{window}_{obj}_{"present" if has else "absent"}', mask & ((v > 0) == has)))
    return out


def paired_bootstrap(y, P, g, pairs, reps, seed, models_extra=()):
    tw = weights(g); u, ix = np.unique(g, return_inverse=True); rng = np.random.default_rng(seed)
    models = sorted(set([m for pr in pairs for m in pr] + list(models_extra)))
    samples = {m: {k: [] for k in ('auc', 'brier', 'logloss')} for m in models}
    for _ in range(reps):
        counts = np.bincount(rng.integers(len(u), size=len(u)), minlength=len(u)); bw = tw * counts[ix]; keep = bw > 0
        for m in models:
            sc = score(y[keep], P[m][keep], bw[keep])
            for k in sc: samples[m][k].append(np.nan if sc[k] is None else sc[k])
    point = {m: score(y, P[m], tw) for m in models}
    model_ci = {m: {k: dict(estimate=point[m][k], ci95=np.nanquantile(np.array(samples[m][k]), [.025, .975]).tolist())
                    for k in point[m]} for m in models}
    res = []
    for a, b in pairs:
        d = {}
        for k in ('auc', 'brier', 'logloss'):
            diff = np.array(samples[a][k]) - np.array(samples[b][k])
            d[k] = dict(estimate=point[a][k] - point[b][k], ci95=np.nanquantile(diff, [.025, .975]).tolist(),
                        fraction_replicates_a_better=float(np.nanmean(diff > 0) if k == 'auc' else np.nanmean(diff < 0)))
        res.append(dict(a=a, b=b, delta_a_minus_b=d))
    return dict(replicates=reps, seed=seed, unit='match (resampled with replacement; weights = existing weights() x count)',
                interval='percentile 95%', fixed_models=True, pairs=res, model_ci=model_ci,
                matches=int(len(u)), rows=int(len(y)))


# ----------------------------------------------------------------------------------------------------------
# SHAP
def group_exact_shapley(f, xrow, Bg, groups):
    """Exact interventional Shapley over feature GROUPS: v(S) = mean_b f(x_S, b_notS). 2^G coalitions."""
    G = len(groups); nb, d = Bg.shape; K = 1 << G
    colmask = np.zeros((K, d), dtype=bool)
    for k, idx in enumerate(groups):
        on = ((np.arange(K) >> k) & 1).astype(bool)
        colmask[np.ix_(on, idx)] = True
    Z = np.where(colmask[:, None, :], xrow[None, None, :], Bg[None, :, :]).reshape(K * nb, d)
    v = f(Z).reshape(K, nb).mean(axis=1)
    fact = [math.factorial(i) for i in range(G + 1)]
    phi = np.zeros(G)
    for S in range(K):
        size = bin(S).count('1')
        for k in range(G):
            if not (S >> k) & 1:
                phi[k] += fact[size] * fact[G - size - 1] / fact[G] * (v[S | (1 << k)] - v[S])
    return phi, float(v[0]), float(v[K - 1])


def linear_parts(b):
    """Return (coef on raw input units, intercept, outer scale a, outer intercept c) for log-odds output."""
    kind = b['candidate']
    pipe = b['pipeline'] if kind in ('ridge_raw', 'p_pre_logistic') else b['base_pipeline']
    sc, lr = pipe[0], pipe[-1]
    beta = lr.coef_[0] / sc.scale_; b0 = lr.intercept_[0] - np.sum(lr.coef_[0] * sc.mean_ / sc.scale_)
    if kind == 'ridge_sigmoid':
        return beta, b0, float(b['calibrator'].coef_[0, 0]), float(b['calibrator'].intercept_[0]), pipe
    return beta, b0, 1.0, 0.0, pipe


def output_fn(b, scale):
    """Actual prediction function on the declared output scale."""
    kind = b['candidate']; cols = b['input_columns']
    if scale == 'probability':
        return lambda Z: predict_bundle(b, Z)
    if kind in ('ridge_raw', 'p_pre_logistic', 'p_pre_spline'):
        return lambda Z: b['pipeline'].decision_function(Z[:, cols])
    if kind == 'ridge_sigmoid':
        return lambda Z: b['calibrator'].decision_function(logit(b['base_pipeline'].predict_proba(Z[:, cols])[:, 1]))
    if kind == 'economic_sigmoid':
        return lambda Z: b['calibrator'].decision_function(logit(np.mean([m.predict_proba(Z[:, cols])[:, 1] for m in b['seed_models']], axis=0)))
    raise ValueError((kind, scale))


def run_shap(run, target, b, X, y, g, M, F, names, n_explain, n_bg, out_dir):
    match = F['match'].astype(str); s = F['s_ms']; kind = b['candidate']
    te_idx = np.flatnonzero(M['test']); tr_idx = np.flatnonzero(M['train'])
    ex = np.array(sorted(te_idx, key=lambda i: rank_key('p4_shap_explain', match[i], s[i]))[:n_explain])
    bg = np.array(sorted(tr_idx, key=lambda i: rank_key('p4_shap_background', match[i], s[i]))[:n_bg])
    cases = np.array(sorted(ex, key=lambda i: rank_key('p4_shap_case', match[i], s[i]))[:8])
    cols = b['input_columns']; in_names = b['input_names']
    summary = dict(target=target, model=kind, manifest=b['manifest'], explained_rows=int(len(ex)),
                   explained_matches=int(len(set(match[ex]))), background_rows=int(len(bg)),
                   background_matches=int(len(set(match[bg]))), background_split='train (15.14) only',
                   explain_split='test (15.16)',
                   selection_rule='sha256("p4_shap_explain:<match>:<s>") ascending (test); '
                                  'sha256("p4_shap_background:<match>:<s>") ascending (train); '
                                  'local cases sha256("p4_shap_case:<match>:<s>") ascending among explained rows',
                   reference='Lundberg & Lee (2017), A Unified Approach to Interpreting Model Predictions, NeurIPS '
                             '(arXiv:1705.07874) - registry docs/LITERATURE_GUIDED_NEXT_STEPS_20260914.md section 5',
                   not_causal=True, not_player_skill=True)
    res = {}
    groups_all = sorted(set(shap_group(n) for n in in_names))
    gidx = [[j for j, n in enumerate(in_names) if shap_group(n) == gr] for gr in groups_all]
    summary['grouping'] = {gr: [in_names[j] for j in idx] for gr, idx in zip(groups_all, gidx)}
    summary['grouping_rule'] = 'historic run_validation_suite.shap_audit group() with p_pre -> p_pre_A'
    summary['feature_units'] = {n: feature_unit(n) for n in in_names}
    phi = None; base = None
    if kind == 'constant':
        summary.update(method='trivial: constant output has zero attribution for every feature',
                       scale='log-odds', base_value=float(np.log(b['prior'] / (1 - b['prior']))), exact_final_output=True)
        phi = np.zeros((len(ex), 0))
    elif kind in ('ridge_raw', 'ridge_sigmoid', 'ridge_isotonic', 'p_pre_logistic'):
        beta, b0, a, c, pipe = linear_parts(b)
        Xe = X[ex][:, cols]; Xb = X[bg][:, cols]; mu_bg = Xb.mean(axis=0)
        phi = a * beta[None, :] * (Xe - mu_bg[None, :])
        base = float(c + a * (b0 + np.sum(beta * mu_bg)))
        f_lin = c + a * (b0 + Xe @ beta)
        dec = pipe.decision_function(Xe)
        if kind == 'ridge_sigmoid':
            f_actual = b['calibrator'].decision_function(logit(pipe.predict_proba(Xe)[:, 1]))
            f_bg_actual = b['calibrator'].decision_function(logit(pipe.predict_proba(Xb)[:, 1]))
        else:
            f_actual = dec; f_bg_actual = pipe.decision_function(Xb)
        p_raw_all = pipe.predict_proba(np.vstack([Xe, Xb]))[:, 1]
        summary.update(
            method='exact linear interventional SHAP: phi_j = a * beta_j * (x_j - mean_background_j), '
                   'beta_j = coef_j / scaler.scale_j (standardization preserved in raw feature units)',
            scale=('calibrated log-odds (logit of final q; sigmoid calibration is affine in raw log-odds)'
                   if kind == 'ridge_sigmoid' else
                   'raw ridge log-odds = BASE SCORE, NOT the final isotonic-calibrated probability'
                   if kind == 'ridge_isotonic' else 'log-odds of final q'),
            exact_final_output=kind != 'ridge_isotonic',
            base_value=base, background_mean_of_actual_output=float(np.mean(f_bg_actual)),
            calibrator_slope=a, calibrator_intercept=c,
            additivity_max_abs_residual_vs_linear_reconstruction=float(np.max(np.abs(phi.sum(1) + base - f_lin))),
            additivity_max_abs_residual_vs_actual_pipeline_output=float(np.max(np.abs(phi.sum(1) + base - f_actual))),
            base_minus_background_mean_actual_output=float(base - np.mean(f_bg_actual)),
            logit_clip_active_rows=int(np.sum((p_raw_all < 1e-8) | (p_raw_all > 1 - 1e-8))),
            scaler_mean=pipe[0].mean_.tolist(), scaler_scale=pipe[0].scale_.tolist(),
            coef_standardized=pipe[-1].coef_[0].tolist(), intercept_standardized=float(pipe[-1].intercept_[0]))
    elif kind == 'p_pre_spline':
        fn = output_fn(b, 'log-odds')
        fe = fn(X[ex]); fb = fn(X[bg]); base = float(np.mean(fb))
        phi = (fe - base)[:, None]
        summary.update(method='single-input model: exact interventional SHAP phi = f(x) - mean_background f',
                       scale='log-odds of final q', exact_final_output=True, base_value=base,
                       additivity_max_abs_residual_vs_actual_pipeline_output=float(np.max(np.abs(phi.sum(1) + base - fe))))
    elif kind.startswith('economic_'):
        Xe = X[ex][:, cols]
        contribs = [m.booster_.predict(Xe, pred_contrib=True, num_threads=4) for m in b['seed_models']]
        margins = [m.booster_.predict(Xe, raw_score=True, num_threads=4) for m in b['seed_models']]
        C = np.mean(contribs, axis=0); phi = C[:, :-1]; base_rows = C[:, -1]; base = float(base_rows[0])
        summary.update(
            method='LightGBM TreeSHAP pred_contrib (tree_path_dependent; reference = training cover, NOT the train '
                   'background sample), averaged over seeds 7/42/123',
            scale='mean per-seed raw tree log-odds = BASE SCORE, NOT the final '
                  + ('mean-probability output' if kind == 'economic_raw' else 'calibrated probability'),
            exact_final_output=False, base_value=base,
            per_seed_additivity_max_abs_residual=[float(np.max(np.abs(cc.sum(1) - mm))) for cc, mm in zip(contribs, margins)],
            additivity_max_abs_residual_vs_mean_raw_margin=float(np.max(np.abs(C.sum(1) - np.mean(margins, axis=0)))),
            base_value_constant_across_rows=bool(np.all(base_rows == base_rows[0])))
    # Group-level exact interventional Shapley on the actual final output (declared budget).
    if kind != 'constant':
        scale = 'probability' if kind.endswith('_isotonic') or kind == 'economic_raw' else 'log-odds'
        fn = output_fn(b, scale)
        Xfull_bg = X[bg]
        gcols = [[cols[j] for j in idx] for idx in gidx]
        t0 = time.time(); G_phi = np.zeros((len(ex), len(groups_all))); v0 = []; vall = []
        for r_i, i in enumerate(ex):
            ph, a0, a1 = group_exact_shapley(fn, X[i], Xfull_bg, gcols)
            G_phi[r_i] = ph; v0.append(a0); vall.append(a1)
            if (r_i + 1) % 100 == 0: run.log(f'  group Shapley {target} {r_i + 1}/{len(ex)} {time.time() - t0:.1f}s')
        f_ex = fn(X[ex])
        gsum = dict(method='exact interventional Shapley over feature groups by full enumeration of 2^G coalitions; '
                           'v(S) = mean over train background rows of f(x_S, background_notS)',
                    scale=scale + ' of the final selected q' + (' (calibrated)' if 'sigmoid' in kind else ''),
                    groups=groups_all, coalitions=1 << len(groups_all), background_rows=int(len(bg)),
                    model_evaluations=int(len(ex) * (1 << len(groups_all)) * len(bg)),
                    runtime_s=round(time.time() - t0, 1),
                    additivity_max_abs_residual=float(np.max(np.abs(G_phi.sum(1) + np.array(v0) - f_ex))),
                    full_coalition_equals_output_max_abs=float(np.max(np.abs(np.array(vall) - f_ex))),
                    empty_coalition_value_unique=bool(np.ptp(v0) < 1e-12), empty_coalition_value=float(v0[0]),
                    budget='G <= 8 groups, <= 512 explained rows, <= 256 background rows, declared in protocol')
        if phi is not None and phi.shape[1] == len(in_names) and kind in ('ridge_raw', 'ridge_sigmoid', 'p_pre_logistic', 'p_pre_spline'):
            grouped = np.stack([phi[:, idx].sum(1) for idx in gidx], axis=1)
            gsum['max_abs_diff_vs_grouped_feature_level_exact_shap'] = float(np.max(np.abs(grouped - G_phi)))
        if kind == 'ridge_isotonic':
            # In-run verification of the enumeration on this model: on the linear BASE score it must equal grouped linear SHAP.
            pipe = b['base_pipeline']; fb = lambda Z: pipe.decision_function(Z[:, cols])
            nv = min(64, len(ex)); Vphi = np.stack([group_exact_shapley(fb, X[i], Xfull_bg, gcols)[0] for i in ex[:nv]])
            grouped = np.stack([phi[:nv, idx].sum(1) for idx in gidx], axis=1)
            gsum['implementation_check_on_raw_base_score'] = dict(
                rows=int(nv), max_abs_diff_vs_grouped_linear_shap=float(np.max(np.abs(Vphi - grouped))))
            fin = predict_bundle(b, X[ex]); raw = pipe.predict_proba(X[ex][:, cols])[:, 1]
            gsum['isotonic_notes'] = dict(
                final_is_monotone_step_function_of_raw_ridge_probability=True,
                explained_rows_final_exact_0_or_1=int(np.sum((fin == 0) | (fin == 1))),
                explained_rows_distinct_final_values=int(len(np.unique(fin))),
                spearman_final_vs_raw_explained=float(scipy.stats.spearmanr(fin, raw).statistic))
        summary['group_exact_final_output'] = gsum
        res['group_phi'] = G_phi
    # Global and local summaries.
    wex = weights(g[ex])
    if phi is not None and phi.shape[1] == len(in_names) and len(in_names):
        glob = pd.DataFrame(dict(feature=in_names, group=[shap_group(n) for n in in_names],
                                 unit=[feature_unit(n) for n in in_names],
                                 mean_abs_shap=np.mean(np.abs(phi), axis=0),
                                 mean_abs_shap_match_weighted=np.average(np.abs(phi), axis=0, weights=wex),
                                 mean_signed_shap=np.mean(phi, axis=0),
                                 background_mean_raw=X[bg][:, cols].mean(axis=0)))
        glob = glob.sort_values(['mean_abs_shap', 'feature'], ascending=[False, True]).reset_index(drop=True)
        glob.to_csv(out_dir / 'global_feature_mean_abs_shap.csv', index=False, float_format='%.17g')
        grp = pd.DataFrame(dict(group=groups_all,
                                n_features=[len(idx) for idx in gidx],
                                mean_abs_sum_of_feature_shap=[float(np.mean(np.abs(phi[:, idx].sum(1)))) for idx in gidx],
                                mean_abs_group_exact_shapley_final_output=(np.mean(np.abs(res['group_phi']), axis=0)
                                                                           if 'group_phi' in res else np.nan)))
        grp.to_csv(out_dir / 'global_group_shap.csv', index=False, float_format='%.17g')
        # Correlation caution among the top features (train rows, unweighted Pearson).
        top = [in_names.index(f) for f in glob.feature[:30]]
        Xt = X[M['train']][:, [cols[j] for j in top]]
        sd = Xt.std(axis=0); ok = sd > 0
        R = np.corrcoef(Xt[:, ok], rowvar=False) if ok.sum() > 1 else np.zeros((0, 0))
        tn = [in_names[j] for j, k in zip(top, ok) if k]
        pairs = [dict(a=tn[i], b=tn[k], pearson_r_train=float(R[i, k])) for i in range(len(tn)) for k in range(i + 1, len(tn))
                 if abs(R[i, k]) >= 0.8]
        pairs.sort(key=lambda d: (-abs(d['pearson_r_train']), d['a'], d['b']))
        summary['correlated_top30_pairs_abs_r_ge_0_8_count'] = len(pairs)
        summary['correlated_top30_pairs_abs_r_ge_0_8_strongest40'] = pairs[:40]
        summary['correlated_top30_features_involved'] = sorted(set([d['a'] for d in pairs] + [d['b'] for d in pairs]))
        summary['top_features'] = glob.head(20)[['feature', 'group', 'unit', 'mean_abs_shap', 'mean_signed_shap']].to_dict('records')
        summary['group_table'] = grp.to_dict('records')
    local = []
    f_final = predict_bundle(b, X[ex])
    for i in cases:
        r_i = int(np.flatnonzero(ex == i)[0])
        rec = dict(row_index=int(i), y=int(y[i]), q_final=float(f_final[r_i]), base_value=base)
        if phi is not None and phi.shape[1]:
            order = np.argsort(-np.abs(phi[r_i]), kind='stable')[:12]
            rec['top_features'] = [dict(feature=in_names[j], group=shap_group(in_names[j]), unit=feature_unit(in_names[j]),
                                        value=float(X[i, cols[j]]), background_mean=float(X[bg][:, cols[j]].mean()),
                                        shap=float(phi[r_i, j])) for j in order]
            rec['sum_shap_plus_base'] = float(phi[r_i].sum() + (base or 0.0))
        if 'group_phi' in res:
            rec['group_exact_final_output'] = {gr: float(v) for gr, v in zip(groups_all, res['group_phi'][r_i])}
        local.append(rec)
    summary['local_cases'] = local
    np.savez_compressed(out_dir / 'shap_values.npz', row_index=ex, background_row_index=bg, case_row_index=cases,
                        feature_names=np.array(in_names), shap=phi if phi is not None else np.zeros((len(ex), 0)),
                        base_value=np.array(base if base is not None else np.nan),
                        group_names=np.array(groups_all),
                        group_exact=res.get('group_phi', np.zeros((len(ex), 0))), q_final=f_final)
    return summary, dict(ex=ex, bg=bg, cases=cases, phi=phi, group_phi=res.get('group_phi'))


# ----------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-dir', default='smoke_run')
    args = ap.parse_args()
    ws = os.environ.get('TEAMFIGHT_WORKSPACE')
    if ws and not os.path.samefile(ws, ROOT):
        raise SystemExit('script root differs from TEAMFIGHT_WORKSPACE')
    out = OUT_BASE / args.smoke_dir if args.smoke else OUT_BASE
    if (out / 'validation.json').exists():
        raise SystemExit(f'{out} already has validation.json; refusing to overwrite')
    out.mkdir(parents=True, exist_ok=True)
    run = Run(out, args.smoke)
    reps_primary = 60 if args.smoke else 2000
    reps_supp = 40 if args.smoke else 1000
    n_explain, n_bg = (24, 32) if args.smoke else (512, 256)
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    try:
        run.status('running', 'start', pid=os.getpid())
        src = Path(__file__).resolve()
        snap = out / 'source_snapshot'; snap.mkdir(exist_ok=True)
        shutil.copy2(src, snap / src.name)
        import lightgbm
        provenance = dict(version=VERSION, role=ROLE, executor='Claude Code CLI, model claude-opus-5',
                          design='Codex (docs/CLAUDE_EXECUTE_P4_Q_BASELINES.md)', spec_sha256=sha256_file(SPEC),
                          runner_sha256=sha256_file(src), python=sys.version, platform=platform.platform(),
                          versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                                        scipy=scipy.__version__, lightgbm=lightgbm.__version__),
                          threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                          lightgbm_n_jobs=4, gpu='hidden (CUDA_VISIBLE_DEVICES empty)', shap_library='not installed; not used')
        write_json(out / 'provenance.json', provenance)
        run.log('hashing frozen inputs (before)')
        hb = hash_frozen(); write_json(out / 'hashes_before.json', hb)
        run.status('running', 'hashes_before', files=len(hb))

        # ---- names, predictor sets, protocol (before any fitting) ----
        with np.load(P3 / 'features_pre_only.npz', allow_pickle=False) as z:
            names = [str(x) for x in z['input_names']]; categorical = [str(x) for x in z['categorical_input_names']]
        sets = feature_sets(names, categorical)
        mapping = schema_mapping_checks(names, sets)
        write_json(out / 'schema_mapping_checks.json', mapping)
        sets_sha = sha256_bytes(json.dumps(sets, sort_keys=True).encode())
        schema_sha = sha256_file(P3 / 'feature_schema.json'); labels_sha = sha256_file(P3 / 'labels_long.npz')
        features_sha = sha256_file(P3 / 'features_pre_only.npz')
        p3proto = read_json(P3 / 'protocol.json')
        A_sha = p3proto['frozen_models']['a_sha256']; B_sha = p3proto['frozen_models']['b_sha256']
        schema_out = dict(role=ROLE, version=VERSION, source='outputs/engagement_labels_v3_sensitivity/features_pre_only.npz',
                          source_sha256=features_sha, source_schema_sha256=schema_sha, input_names_all=names,
                          categorical_excluded=categorical, predictor_sets=sets, predictor_sets_sha256=sets_sha,
                          feature_units={n: feature_unit(n) for n in names if n not in categorical},
                          shap_groups={n: shap_group(n) for n in names},
                          ids_for_grouping_only=['match', 's_ms', 'row_index', 'patch_split_metadata'],
                          excluded=['snapshot_age_s (audit only; absent from P3 X_input)', 'champion IDs (categorical)',
                                    'positions', 'post-state, L, duration, endpoint/reason, future membership, final outcome, '
                                    'event counts after q_pre, target-source diagnostics, match IDs'])
        write_json(out / 'pre_only_schema.json', schema_out)
        protocol = dict(
            version=VERSION, role=ROLE, smoke=args.smoke, written_before_fitting=True,
            written_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            design='Codex design (docs/CLAUDE_EXECUTE_P4_Q_BASELINES.md); Claude Opus 5 implementation and execution',
            spec_sha256=sha256_file(SPEC),
            estimand='q(X_pre) = P(delta_A_h > 0 | pre-state, retrospectively selected engagement cohort); '
                     'labels are model-defined outcomes (frozen V), not observed engagement success',
            primary=dict(target='A90', rule='horizon 90 s fixed regardless of results; never selected on TEST'),
            sensitivity=dict(A60='identical pipeline, independent fit/selection', A120='identical pipeline, independent fit/selection',
                             B90='same pre-only inputs incl. p_pre_A; fixed inputs isolate the target choice; frozen B not repaired',
                             cross_target='frozen A90 selected q scored against Y_B90 on TEST: disagreement diagnostic, not accuracy against truth'),
            labels=dict(source='outputs/engagement_labels_v3_sensitivity/labels_long.npz', sha256=labels_sha,
                        exact_zero='delta == 0 -> Y = 0', model_A_sha256=A_sha, model_B_bundle_sha256=B_sha),
            inputs=dict(source='features_pre_only.npz', sha256=features_sha, schema_sha256=schema_sha,
                        predictor_sets_sha256=sets_sha, p_pre_A='frozen P1 model A at q_pre, trained on disjoint matches',
                        snapshot_age_s='excluded from predictors (audit only). The old q experiment (validation_suite_20260914) '
                                       'INCLUDED snapshot_age_s in economic and ridge inputs',
                        champion_ids='excluded (categorical; not used as numeric ranks)', positions='excluded',
                        economic_names=sets['economic'], economic_count=len(sets['economic']),
                        economic_rule='historic run_validation_suite econ selection with slotK_ -> participant_slotK_ and snapshot_age_s removed; '
                                      'includes team aggregate objective/structure counts -> called economic/aggregate baseline, NOT gold-only',
                        ridge_names=sets['ridge'], ridge_count=len(sets['ridge']),
                        ridge_rule='all P3 pre numeric inputs except champion IDs (snapshot_age_s already absent) plus p_pre_A',
                        schema_mapping_checks=mapping),
            split=dict(train='patch 15.14', calibrate="patch 15.15 and int(sha256('calibration17:'+match)[:8],16) % 2 == 0",
                       select="patch 15.15 and ... % 2 == 1", test='patch 15.16',
                       expected={k: dict(rows=v[0], matches=v[1]) for k, v in EXPECTED.items()},
                       rule='exact keys/patches/order of validation_suite_20260914 rows.csv and q_predictions.csv joined to P3 rows; '
                            'any material mismatch blocks fitting'),
            weights="existing weights(g): w = (1/count_per_match)/mean, recomputed within each fitted/evaluated subset "
                    "(equal total weight per match; mean weight 1)",
            preprocessing='StandardScaler/SplineTransformer fitted on TRAIN rows only (inside pipelines, unweighted as before); '
                          'LogisticRegression/LGBM with TRAIN match weights',
            candidates={
                'constant': 'TRAIN match-weighted positive rate',
                'p_pre_logistic': 'StandardScaler + LogisticRegression(C=1, max_iter=2000) on p_pre_A; uncalibrated',
                'p_pre_spline': 'SplineTransformer(n_knots=5, degree=3) + StandardScaler + LogisticRegression(C=1, max_iter=2000); uncalibrated',
                'ridge_raw': 'StandardScaler + LogisticRegression(C=0.01, max_iter=3000) on ridge_names',
                'ridge_sigmoid': 'LogisticRegression(C=1e6, max_iter=1000) on logit(clip(ridge_raw,1e-8)) fitted on CALIBRATE with match weights',
                'ridge_isotonic': "IsotonicRegression(out_of_bounds='clip') on ridge_raw fitted on CALIBRATE with match weights",
                'economic_raw': 'mean probability of LGBMClassifier seeds 7/42/123 (250 trees, 15 leaves, lr .04, min_child_samples 100, '
                                'reg_lambda 1, colsample_bytree .9, n_jobs 4; no early stopping/search) on economic_names',
                'economic_sigmoid': 'as ridge_sigmoid on economic_raw', 'economic_isotonic': 'as ridge_isotonic on economic_raw'},
            candidate_pool_note='Reduced pool: old full/position LightGBM candidates (old schema/target) are excluded. '
                                'The old reported q score (ridge_sigmoid, V1 labels) is context only, not an apples-to-apples baseline.',
            selection='lowest SELECT match-weighted Brier, then log loss, then candidate name; saved (with bundle hashes) '
                      'for every target before ANY test prediction',
            evaluation=dict(metrics='AUC, Brier, log loss (sklearn, restored score()); logistic recalibration intercept/slope '
                                    '(restored calibration(): joint fit of y on logit(p)); calibration-in-the-large intercept with '
                                    'logit(p) offset (added diagnostic); 10 equal-width reliability bins; all splits',
                            optimism='train minus test and select minus test per candidate (sign: metric difference)',
                            subgroups='TEST: overall; P3 start time bands (minutes of s) 0-2/2-10/10-20/20-30/30+; patch; pre-query '
                                      'objective acquisition history (either team, from X_pre); P3 during (q_pre, L] and '
                                      'after-last-kill (L, endpoint_h] objective events as POST-HOC descriptive strata only; '
                                      'weights renormalized within stratum; AUC NA if one class; <30 matches flagged sparse',
                            bootstrap=dict(primary_pairs='chosen vs constant, p_pre_logistic, p_pre_spline, selected economic '
                                                         'variant, selected ridge variant (identical pairs skipped)',
                                           replicates_primary=reps_primary, replicates_supplementary_chosen_vs_constant=reps_supp,
                                           seed=BOOT_SEED, unit='match', models='fixed; no refitting'),
                            independent_reconstruction='direct numpy weighted AUC (Mann-Whitney with half ties), Brier, log loss '
                                                       '(eps clipping as sklearn) for every candidate x split'),
            shap=dict(target='A90 selected q only', explained_rows=f'up to {n_explain} TEST rows by sha256 rank',
                      background_rows=f'up to {n_bg} TRAIN rows by sha256 rank (uniform)',
                      plans={'ridge_raw/p_pre_logistic': 'exact linear interventional SHAP on log-odds of final q',
                             'ridge_sigmoid': 'exact linear interventional SHAP on calibrated log-odds (sigmoid calibration is affine)',
                             'ridge_isotonic': 'exact linear SHAP on raw ridge log-odds labelled BASE SCORE (not final probability)',
                             'p_pre_spline': 'single-input exact phi = f(x) - mean_bg f on log-odds',
                             'constant': 'trivial zero attributions',
                             'economic_*': 'LightGBM TreeSHAP (path-dependent) on mean per-seed raw log-odds labelled BASE SCORE, '
                                           'not the final calibrated/averaged probability',
                             'all non-constant': 'plus exact group-level interventional Shapley on the actual final output '
                                                 '(2^G coalitions, G <= 8 historic groups, background above); for linear winners '
                                                 'this must equal grouped feature-level SHAP'},
                      reference='Lundberg & Lee (2017), NeurIPS, arXiv:1705.07874 (registry docs/LITERATURE_GUIDED_NEXT_STEPS_20260914.md)',
                      library='shap not installed and not installed by this run; closed forms / LightGBM pred_contrib / explicit enumeration'),
            checks=['exact row order and labels vs P3 (npz and round-trip CSV)', 'exact split membership vs saved q results',
                    'q partitions mutually disjoint; zero overlap with all V partitions', 'no future predictors / forbidden tokens',
                    'finite data and probabilities', 'transformer fitting origins', 'categorical IDs excluded',
                    'model serialization identity', 'selection before test', 'frozen hashes before == after',
                    'independent metric reconstruction', 'SHAP additivity and deterministic examples'],
            resources=dict(cpu='<= 4 threads (LightGBM n_jobs 4; BLAS 1 thread)', gpu='not used; user GPU queue untouched'),
            outputs='outputs/q_v3_baselines only',
            references={'Lundberg & Lee (2017), arXiv:1705.07874': 'SHAP: additive attribution of model predictions; not causal effects',
                        'Van Calster et al. (2019), DOI 10.1186/s12916-019-1466-7': 'calibration assessment (intercept/slope/curves); '
                                                                                     'medical sample-size rules not transferred',
                        'Maymin (2021), DOI 10.1515/jqas-2019-0096': 'state win-probability change as action value rationale; '
                                                                     'does not justify our endpoint rule or q performance',
                        'Decroos et al. (2019), DOI 10.1145/3292500.3330758': 'separating component probability, design choices and cases; '
                                                                              'not a justification of 90 s'},
            claims_not_made=['labels are not ground truth', 'no causal/intervention effects', 'no player-skill claims',
                             'no all-patch generalization; 15.16 was already observed in earlier work (not an untouched test)',
                             'not production ready'])
        protocol['protocol_sha256'] = sha256_bytes(json.dumps(clean(protocol), sort_keys=True).encode())
        write_json(out / 'protocol.json', protocol)
        run.status('running', 'protocol_written', protocol_sha256=protocol['protocol_sha256'],
                   schema_mapping_pass=mapping['pass'])
        if not mapping['pass']:
            run.status('blocked', 'schema_mapping_mismatch'); return 2

        # ---- data verification (blocking) ----
        F, Lb = load_data()
        dchk, M, per_h, old_q = verify_data(run, F, Lb, sets)
        write_json(out / 'data_checks.json', dict(pass_=dchk['pass_'], failed=dchk['failed'], checks=dchk['checks'], info=dchk['info']))
        run.status('running' if dchk['pass_'] else 'blocked', 'data_verification', failed=dchk['failed'])
        if not dchk['pass_']:
            return 2
        X = F['X_input']; g = F['match'].astype(str); s_ms = F['s_ms']; cix = col_index(names, sets)
        ids = pd.DataFrame(dict(row_index=F['row_index'], match=g, s_ms=s_ms, patch=F['patch_split_metadata'].astype(str)))
        split_name = np.empty(len(g), dtype=object)
        for k in SPLITS: split_name[M[k]] = k
        ids['split'] = split_name
        (out / 'predictions').mkdir(exist_ok=True)
        ids.to_csv(out / 'predictions' / 'ids.csv', index=False)
        Y = {t: Lb['Y_' + mdl][per_h[h]].astype(int) for t, (mdl, h) in TARGETS.items()}

        # ---- phase 1: fit + non-test predictions + SELECT for every target ----
        bundles, preds, metrics, selections, fitlogs, bundle_hashes = {}, {}, {}, {}, {}, {}
        for t in TARGETS:
            man = manifest(t, sets_sha, schema_sha, labels_sha, A_sha, B_sha)
            B, fl = fit_target(run, t, X, Y[t], g, M, cix, sets, man)
            bundles[t] = B; fitlogs[t] = fl
            mdir = out / 'models' / t; mdir.mkdir(parents=True, exist_ok=True)
            bundle_hashes[t] = {}
            for cand in CANDIDATES:
                joblib.dump(B[cand], mdir / f'{cand}.joblib'); bundle_hashes[t][cand] = sha256_file(mdir / f'{cand}.joblib')
            P = {c: np.full(len(g), np.nan) for c in CANDIDATES}
            for k in ('train', 'calibrate', 'select'):
                for c in CANDIDATES: P[c][M[k]] = predict_bundle(B[c], X[M[k]])
            preds[t] = P
            metrics[t] = {c: {k: evaluate(Y[t][M[k]], P[c][M[k]], g[M[k]]) for k in ('train', 'calibrate', 'select')}
                          for c in CANDIDATES}
            sel_rows = [(metrics[t][c]['select']['brier'], metrics[t][c]['select']['logloss'], c) for c in CANDIDATES]
            chosen = min(sel_rows)[2]
            sel_ridge = min(r for r in sel_rows if r[2].startswith('ridge_'))[2]
            sel_econ = min(r for r in sel_rows if r[2].startswith('economic_'))[2]
            selection = dict(role=ROLE, manifest=man, target=t, rule=protocol['selection'], chosen=chosen,
                             selected_ridge_variant=sel_ridge, selected_economic_variant=sel_econ,
                             select_split=dict(rows=int(M['select'].sum()), matches=int(len(set(g[M['select']])))),
                             select_metrics={c: {k: metrics[t][c]['select'][k] for k in ('brier', 'logloss', 'auc')} for c in CANDIDATES},
                             ranking=[r[2] for r in sorted(sel_rows)], bundle_sha256=bundle_hashes[t],
                             test_prediction_calls_so_far=run.test_prediction_calls,
                             written_at=time.strftime('%Y-%m-%d %H:%M:%S'), written_before_any_test_prediction=run.test_prediction_calls == 0)
            h_sel = write_json(out / 'selection' / f'{t}.json', selection)
            selections[t] = dict(selection, file_sha256=h_sel)
            run.status('running', f'phase1_selected_{t}', chosen=chosen, ridge=sel_ridge, economic=sel_econ,
                       selection_sha256=h_sel, warnings=len(fl['warnings']))
        write_json(out / 'fit_logs.json', fitlogs)
        phase1_end = time.time()

        # ---- phase 2: reload serialized bundles, identity check, TEST predictions ----
        identity = {}
        for t in TARGETS:
            identity[t] = {}
            for cand in CANDIDATES:
                path = out / 'models' / t / f'{cand}.joblib'
                h_now = sha256_file(path)
                b = joblib.load(path)
                same = all(np.array_equal(predict_bundle(b, X[M[k]]), preds[t][cand][M[k]]) for k in ('train', 'calibrate', 'select'))
                run.test_prediction_calls += 1
                preds[t][cand][M['test']] = predict_bundle(b, X[M['test']])
                identity[t][cand] = dict(sha256_at_selection=bundle_hashes[t][cand], sha256_at_test=h_now,
                                         hash_unchanged=h_now == bundle_hashes[t][cand], reloaded_predictions_bitwise_equal=same,
                                         manifest_target=b['manifest']['target'] == t)
                bundles[t][cand] = b
                for k in ('test',):
                    metrics[t][cand][k] = evaluate(Y[t][M[k]], preds[t][cand][M[k]], g[M[k]])
            run.status('running', f'phase2_test_{t}')
        write_json(out / 'serialization_identity.json', identity)

        # ---- predictions (IDs separate) ----
        for t in TARGETS:
            arr = {c: preds[t][c] for c in CANDIDATES}
            np.savez_compressed(out / 'predictions' / f'{t}.npz', row_index=F['row_index'], y=Y[t], split=split_name.astype(str), **arr)
            df = pd.DataFrame(dict(row_index=F['row_index'], split=split_name, y=Y[t], **arr))
            df.to_csv(out / 'predictions' / f'{t}.csv', index=False, float_format='%.17g')
        write_json(out / 'predictions' / 'manifest.json', dict(
            role=ROLE, note='q probabilities of candidate labels; NOT V win probabilities. IDs in ids.csv (grouping only).',
            manifests={t: manifest(t, sets_sha, schema_sha, labels_sha, A_sha, B_sha) for t in TARGETS},
            files={p.name: sha256_file(p) for p in sorted((out / 'predictions').iterdir()) if p.suffix in ('.npz', '.csv')}))

        # ---- optimism, subgroups ----
        optimism = {}
        for t in TARGETS:
            optimism[t] = {}
            for c in CANDIDATES:
                te_m = metrics[t][c]['test']
                optimism[t][c] = {f'{k}_minus_test': {m: (None if metrics[t][c][k][m] is None or te_m[m] is None
                                                          else metrics[t][c][k][m] - te_m[m]) for m in ('auc', 'brier', 'logloss')}
                                  for k in ('train', 'select')}
        subgroups = {}
        for t, (mdl, h) in TARGETS.items():
            Lh = {k: v[per_h[h]] for k, v in Lb.items() if k.startswith('during_') or k.startswith('after_last_kill_')}
            st = strata(names, X, Lh, s_ms, M['test'])
            subgroups[t] = []
            for name, mask in st:
                row = dict(stratum=name, rows=int(mask.sum()), matches=int(len(set(g[mask]))))
                row['by_model'] = ({c: evaluate(Y[t][mask], preds[t][c][mask], g[mask], bins=name == 'overall')
                                    for c in CANDIDATES} if mask.any() else {})
                subgroups[t].append(row)
            run.status('running', f'subgroups_{t}', strata=len(st))
        patch_table = {t: {SPLIT_PATCH[k] + ':' + k: {c: {m: metrics[t][c][k][m] for m in ('n', 'matches', 'auc', 'brier', 'logloss')}
                                                     for c in CANDIDATES} for k in SPLITS} for t in TARGETS}

        # ---- bootstrap ----
        te = M['test']
        boot = {}
        sel = selections[PRIMARY]; ch = sel['chosen']
        pairs, skipped = [], []
        for b_ in ('constant', 'p_pre_logistic', 'p_pre_spline', sel['selected_economic_variant'], sel['selected_ridge_variant']):
            (skipped if b_ == ch else pairs).append((ch, b_))
        run.log(f'bootstrap primary {pairs} reps={reps_primary}')
        Pte = {c: preds[PRIMARY][c][te] for c in CANDIDATES}
        boot[PRIMARY] = paired_bootstrap(Y[PRIMARY][te], Pte, g[te], pairs, reps_primary, BOOT_SEED)
        boot[PRIMARY]['skipped_identical_pairs'] = skipped
        for t in TARGETS:
            if t == PRIMARY: continue
            c_t = selections[t]['chosen']
            if c_t == 'constant':
                boot[t] = dict(skipped='chosen is constant'); continue
            Pt = {c: preds[t][c][te] for c in CANDIDATES}
            boot[t] = paired_bootstrap(Y[t][te], Pt, g[te], [(c_t, 'constant')], reps_supp, BOOT_SEED)
            boot[t]['role'] = 'supplementary sensitivity: chosen vs constant only'
        write_json(out / 'bootstrap.json', dict(role=ROLE, manifests={t: manifest(t, sets_sha, schema_sha, labels_sha, A_sha, B_sha) for t in TARGETS},
                                                results=boot, not_claimed='no causal, all-patch or untouched-test claims; model/selection uncertainty excluded'))
        run.status('running', 'bootstrap_done')

        # ---- cross-target diagnostic and old-experiment context ----
        cA = selections['A90']['chosen']; cB = selections['B90']['chosen']
        tw = weights(g[te])
        cross = dict(
            role='disagreement diagnostic; neither label is truth',
            a90_selected=cA, b90_selected=cB,
            a90_q_vs_Y_A90=evaluate(Y['A90'][te], preds['A90'][cA][te], g[te]),
            a90_q_vs_Y_B90=evaluate(Y['B90'][te], preds['A90'][cA][te], g[te]),
            b90_q_vs_Y_B90=evaluate(Y['B90'][te], preds['B90'][cB][te], g[te]),
            label_disagreement_A90_vs_B90_test=dict(rows=float(np.mean(Y['A90'][te] != Y['B90'][te])),
                                                    match_weighted=float(np.average(Y['A90'][te] != Y['B90'][te], weights=tw))),
            spearman_a90_q_vs_b90_q_test=float(scipy.stats.spearmanr(preds['A90'][cA][te], preds['B90'][cB][te]).statistic)
            if np.std(preds['A90'][cA][te]) > 0 and np.std(preds['B90'][cB][te]) > 0 else None)
        q_old = read_json(VS / 'q_results.json')
        old_ctx = dict(
            role='context only; different labels (V1 expanded delta), different p_pre source, snapshot_age_s included, '
                 'larger candidate pool (full/position LightGBM); not an apples-to-apples baseline',
            old_chosen=q_old['chosen_by_development'], old_selected_calibrations=q_old['selected_calibrations'],
            old_test={m['model']: {k: m[k] for k in ('auc', 'brier', 'logloss')} for m in q_old['metrics']
                      if m['split'] == 'test_exploratory' and m['model'] in ('constant', 'p_pre_logistic', 'p_pre_spline', 'full_ridge',
                                                                             'ridge_sigmoid', 'economic_raw', 'economic_sigmoid')},
            old_vs_new_label_agreement=dchk['info']['old_y_vs_new_labels'],
            schema_differences=mapping)
        try:
            old_shap = pd.read_csv(VS / 'selected_model_shap.csv')
            if 'feature' in old_shap.columns:
                num = old_shap.select_dtypes('number').columns.tolist()
                old_ctx['old_selected_shap_snapshot_age_row'] = old_shap[old_shap.feature == 'snapshot_age_s'].to_dict('records')
                old_ctx['old_selected_shap_columns'] = old_shap.columns.tolist()
                if num:
                    k0 = num[0]; ranks = old_shap.sort_values(k0, ascending=False).reset_index(drop=True)
                    hit = ranks.index[ranks.feature == 'snapshot_age_s'].tolist()
                    old_ctx['old_selected_shap_snapshot_age_rank_by_' + k0] = (hit[0] + 1) if hit else None
                    old_ctx['old_selected_shap_features'] = len(ranks)
        except Exception as exc:  # context only
            old_ctx['old_selected_shap_read_error'] = repr(exc)

        # ---- SHAP for A90 selected ----
        shap_dir = out / 'shap'; shap_dir.mkdir(exist_ok=True)
        run.status('running', 'shap_start', model=cA)
        try:
            shap_summary, shap_arr = run_shap(run, PRIMARY, bundles[PRIMARY][cA], X, Y[PRIMARY], g, M, F, names,
                                              n_explain, n_bg, shap_dir)
            # Determinism: recompute from a fresh reload for the local cases.
            b2 = joblib.load(out / 'models' / PRIMARY / f'{cA}.joblib')
            ex_again = np.array(sorted(np.flatnonzero(M['test']), key=lambda i: rank_key('p4_shap_explain', g[i], s_ms[i]))[:n_explain])
            det = dict(explained_rows_reselected_equal=bool(np.array_equal(ex_again, shap_arr['ex'])))
            if shap_arr['phi'] is not None and shap_arr['phi'].shape[1] and cA in ('ridge_raw', 'ridge_sigmoid', 'ridge_isotonic', 'p_pre_logistic'):
                beta, b0, a, c, pipe = linear_parts(b2)
                mu_bg = X[shap_arr['bg']][:, b2['input_columns']].mean(axis=0)
                cases_pos = [int(np.flatnonzero(shap_arr['ex'] == i)[0]) for i in shap_arr['cases']]
                phi2 = a * beta[None, :] * (X[shap_arr['cases']][:, b2['input_columns']] - mu_bg[None, :])
                det['case_shap_bitwise_equal_after_reload'] = bool(np.array_equal(phi2, shap_arr['phi'][cases_pos]))
            elif shap_arr['phi'] is not None and shap_arr['phi'].shape[1] and cA.startswith('economic_'):
                cases_pos = [int(np.flatnonzero(shap_arr['ex'] == i)[0]) for i in shap_arr['cases']]
                Xc = X[shap_arr['cases']][:, b2['input_columns']]
                phi2 = np.mean([m.booster_.predict(Xc, pred_contrib=True, num_threads=4) for m in b2['seed_models']], axis=0)[:, :-1]
                det['case_treeshap_max_abs_diff_after_reload'] = float(np.max(np.abs(phi2 - shap_arr['phi'][cases_pos])))
                det['case_treeshap_equal_after_reload_1e-12'] = det.pop('case_treeshap_max_abs_diff_after_reload') <= 1e-12
            elif shap_arr['phi'] is not None and shap_arr['phi'].shape[1] and cA == 'p_pre_spline':
                cases_pos = [int(np.flatnonzero(shap_arr['ex'] == i)[0]) for i in shap_arr['cases']]
                fn = output_fn(b2, 'log-odds')
                phi2 = fn(X[shap_arr['cases']]) - np.mean(fn(X[shap_arr['bg']]))
                det['case_shap_bitwise_equal_after_reload'] = bool(np.array_equal(phi2, shap_arr['phi'][cases_pos, 0]))
            if shap_arr['group_phi'] is not None:
                fn2 = output_fn(b2, 'probability' if cA.endswith('_isotonic') or cA == 'economic_raw' else 'log-odds')
                in_names = b2['input_names']; groups_all = sorted(set(shap_group(n) for n in in_names))
                gcols = [[b2['input_columns'][j] for j, n in enumerate(in_names) if shap_group(n) == gr] for gr in groups_all]
                i0 = shap_arr['cases'][:3]; pos = [int(np.flatnonzero(shap_arr['ex'] == i)[0]) for i in i0]
                again = np.stack([group_exact_shapley(fn2, X[i], X[shap_arr['bg']], gcols)[0] for i in i0])
                det['group_exact_first3_cases_bitwise_equal_after_reload'] = bool(np.array_equal(again, shap_arr['group_phi'][pos]))
            shap_summary['determinism'] = det
            write_json(shap_dir / 'shap_summary.json', shap_summary)
            shap_status = dict(status='complete', model=cA)
        except Exception as exc:
            shap_status = dict(status='failed', model=cA, error=repr(exc), traceback=traceback.format_exc())
            write_json(shap_dir / 'shap_failure.json', shap_status)
            shap_summary = None
        run.status('running', 'shap_done', **{k: v for k, v in shap_status.items() if k != 'traceback'})

        # ---- independent metric reconstruction and fitting-origin checks ----
        recon = dict(max_abs_diff={'auc': 0.0, 'brier': 0.0, 'logloss': 0.0}, compared=0)
        for t in TARGETS:
            for c in CANDIDATES:
                for k in SPLITS:
                    m_ = metrics[t][c][k]; yy = Y[t][M[k]]; pp = preds[t][c][M[k]]; w_ = weights(g[M[k]])
                    for key, fn in (('auc', auc_direct), ('brier', brier_direct), ('logloss', logloss_direct)):
                        v = fn(yy, pp, w_)
                        if (v is None) != (m_[key] is None): recon['max_abs_diff'][key] = float('inf'); continue
                        if v is not None: recon['max_abs_diff'][key] = max(recon['max_abs_diff'][key], abs(v - m_[key]))
                    recon['compared'] += 1
        # Weight normalization check.
        wchk = {}
        for k in SPLITS:
            w_ = weights(g[M[k]]); _, inv = np.unique(g[M[k]], return_inverse=True); tot = np.bincount(inv, weights=w_)
            wchk[k] = dict(mean=float(w_.mean()), sum=float(w_.sum()), per_match_total_min=float(tot.min()),
                           per_match_total_max=float(tot.max()), rows_over_matches=float(M[k].sum() / len(tot)))
        recon['weights'] = wchk
        origins = {}
        Xtr = X[M['train']]; Xca = X[M['calibrate']]
        for t in TARGETS:
            B = bundles[t]; o = {}
            for cand, key in (('p_pre_logistic', 'p_pre'), ('ridge_raw', 'ridge')):
                sc = B[cand]['pipeline'][0]; ref = StandardScaler().fit(Xtr[:, cix[key]])
                o[cand + '_scaler_equals_train_refit'] = bool(np.array_equal(sc.mean_, ref.mean_) and np.array_equal(sc.scale_, ref.scale_))
                o[cand + '_scaler_n_samples_seen'] = int(sc.n_samples_seen_)
                o[cand + '_scaler_n_samples_seen_equals_train_rows'] = int(sc.n_samples_seen_) == int(M['train'].sum())
                alt = StandardScaler().fit(X[M['train'] | M['calibrate']][:, cix[key]])
                o[cand + '_scaler_differs_from_train_plus_calibrate_fit'] = not np.array_equal(sc.mean_, alt.mean_)
            sp = B['p_pre_spline']['pipeline'][0]; ref = SplineTransformer(n_knots=5, degree=3).fit(Xtr[:, cix['p_pre']])
            o['spline_knots_equal_train_refit'] = bool(np.array_equal(sp.bsplines_[0].t, ref.bsplines_[0].t))
            o['spline_scaler_n_samples_seen'] = int(B['p_pre_spline']['pipeline'][1].n_samples_seen_)
            cw = weights(g[M['calibrate']])
            for base in ('ridge', 'economic'):
                raw = predict_bundle(dict(B[base + '_sigmoid'], candidate=base + '_raw', pipeline=B['ridge_raw']['pipeline'])
                                     if base == 'ridge' else dict(B['economic_raw']), Xca)
                sg = LogisticRegression(C=1e6, max_iter=1000).fit(logit(raw), Y[t][M['calibrate']], sample_weight=cw)
                o[base + '_sigmoid_equals_calibrate_refit'] = bool(np.array_equal(sg.coef_, B[base + '_sigmoid']['calibrator'].coef_)
                                                                   and np.array_equal(sg.intercept_, B[base + '_sigmoid']['calibrator'].intercept_))
                iso = IsotonicRegression(out_of_bounds='clip').fit(raw, Y[t][M['calibrate']], sample_weight=cw)
                o[base + '_isotonic_equals_calibrate_refit'] = bool(np.array_equal(iso.X_thresholds_, B[base + '_isotonic']['calibrator'].X_thresholds_)
                                                                    and np.array_equal(iso.y_thresholds_, B[base + '_isotonic']['calibrator'].y_thresholds_))
            origins[t] = o
        from lightgbm import LGBMClassifier
        m7 = LGBMClassifier(random_state=7, **LGBM_PARAMS).fit(Xtr[:, cix['economic']], Y[PRIMARY][M['train']], sample_weight=weights(g[M['train']]))
        d7 = float(np.max(np.abs(m7.predict_proba(X[:, cix['economic']])[:, 1] - bundles[PRIMARY]['economic_raw']['seed_models'][0].predict_proba(X[:, cix['economic']])[:, 1])))
        origins['lightgbm_seed7_A90_train_refit_max_abs_pred_diff'] = d7
        run.status('running', 'reconstruction_done', recon=recon['max_abs_diff'])

        # ---- probabilities finite ----
        prob = {t: {c: dict(finite=bool(np.isfinite(preds[t][c]).all()), in_0_1=bool(((preds[t][c] >= 0) & (preds[t][c] <= 1)).all()),
                            exact_0_or_1=int(np.sum((preds[t][c] == 0) | (preds[t][c] == 1))),
                            min=float(np.min(preds[t][c])), max=float(np.max(preds[t][c]))) for c in CANDIDATES} for t in TARGETS}

        results = dict(role=ROLE, version=VERSION, smoke=args.smoke, protocol_sha256=protocol['protocol_sha256'],
                       manifests={t: manifest(t, sets_sha, schema_sha, labels_sha, A_sha, B_sha) for t in TARGETS},
                       selections={t: {k: selections[t][k] for k in ('chosen', 'selected_ridge_variant', 'selected_economic_variant', 'ranking', 'file_sha256')} for t in TARGETS},
                       metrics=metrics, optimism=optimism, patch_table=patch_table, cross_target=cross, old_experiment_context=old_ctx,
                       probability_ranges=prob, fit_logs=fitlogs, shap_status={k: v for k, v in shap_status.items() if k != 'traceback'},
                       split_counts=dchk['info']['split_counts'], v_partition_overlap=dchk['info']['v_partition_overlap'],
                       label_exact_zero=dict((k, v) for k, v in dchk['info'].items() if 'exact_zero' in k))
        write_json(out / 'results.json', results)
        write_json(out / 'subgroups_test.json', dict(role=ROLE, note='TEST only; weights renormalized within stratum; posthoc_* strata use '
                                                                    'events after q_pre and are descriptive only (not available to q)',
                                                     manifests=results['manifests'], subgroups=subgroups))
        run.status('running', 'results_written')

        # ---- validation ----
        ha = hash_frozen(); write_json(out / 'hashes_after.json', ha)
        changed = [k for k in hb if hb[k] != ha.get(k)]; added = [k for k in ha if k not in hb]
        sel_files_ok = all((out / 'selection' / f'{t}.json').exists() and sha256_file(out / 'selection' / f'{t}.json') == selections[t]['file_sha256'] for t in TARGETS)
        checks = dict(
            data_checks_pass=dchk['pass_'],
            schema_mapping_pass=mapping['pass'],
            exact_split_membership=all(dchk['checks'][f'split_{k}_counts_expected'] and dchk['checks'][f'split_{k}_equals_old_q_predictions_split'] for k in SPLITS),
            q_partitions_disjoint=dchk['checks']['q_partitions_mutually_disjoint_by_match'],
            zero_V_overlap=dchk['checks']['zero_match_overlap_with_all_V_partitions'],
            no_future_predictors=all(dchk['checks'][k] for k in ('predictors_no_forbidden_tokens', 'predictors_no_snapshot_age',
                                                                 'no_predictor_column_equals_label_future_or_snapshot_age_column', 'features_role_pre_only')),
            categorical_ids_excluded=dchk['checks']['predictors_no_categorical_ids'],
            finite_data=dchk['checks']['features_finite'],
            finite_probabilities=all(v['finite'] and v['in_0_1'] for d in prob.values() for v in d.values()),
            transformer_fitting_origins=all(v for t in TARGETS for k, v in origins[t].items() if isinstance(v, bool)),
            model_serialization_identity=all(v['hash_unchanged'] and v['reloaded_predictions_bitwise_equal'] and v['manifest_target']
                                             for d in identity.values() for v in d.values()),
            selection_before_test=all(selections[t]['written_before_any_test_prediction'] for t in TARGETS) and sel_files_ok,
            frozen_hashes_unchanged=not changed,
            independent_metric_reconstruction=all(v < 1e-10 for v in recon['max_abs_diff'].values()),
            shap_complete=shap_status['status'] == 'complete')
        if shap_summary is not None:
            add = []
            for key in ('additivity_max_abs_residual_vs_linear_reconstruction', 'additivity_max_abs_residual_vs_mean_raw_margin',
                        'additivity_max_abs_residual_vs_actual_pipeline_output'):
                if key in shap_summary: add.append(shap_summary[key])
            gx = shap_summary.get('group_exact_final_output', {})
            if gx: add.append(gx['additivity_max_abs_residual'])
            checks['shap_additivity'] = bool(add) and max(add) < 1e-8
            if 'max_abs_diff_vs_grouped_feature_level_exact_shap' in gx:
                checks['shap_group_exact_matches_feature_level'] = gx['max_abs_diff_vs_grouped_feature_level_exact_shap'] < 1e-8
            if 'implementation_check_on_raw_base_score' in gx:
                checks['shap_group_exact_verified_on_base_score'] = gx['implementation_check_on_raw_base_score']['max_abs_diff_vs_grouped_linear_shap'] < 1e-8
            checks['shap_deterministic_examples'] = all(shap_summary['determinism'].values())
        failed = [k for k, v in checks.items() if not v]
        validation = dict(role=ROLE, version=VERSION, smoke=args.smoke, status='pass' if not failed else 'fail', failed_checks=failed,
                          checks=checks, data_checks=dchk['checks'], transformer_origins=origins, metric_reconstruction=recon,
                          serialization_identity=identity, frozen_changed=changed, frozen_added_not_by_this_run=added,
                          selection_times=dict(phase1_end_epoch=phase1_end),
                          manifests=results['manifests'])
        write_json(out / 'validation.json', validation)
        run.status('complete' if not failed else 'complete_with_failed_checks', 'validation', failed=failed)
        return 0 if not failed else 1
    except Exception as exc:
        run.log(traceback.format_exc())
        run.status('failed', 'exception', error=repr(exc))
        return 3


if __name__ == '__main__':
    sys.exit(main())
