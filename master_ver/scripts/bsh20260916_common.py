"""Balanced-state SHAP 2026-09-16: shared helpers (pure; importing has no side effects beyond environment defaults).

Design: docs/CLAUDE_BALANCED_SHAP_20260916.md. Everything written by this run lands in outputs/balanced_shap_20260916.
Parents are READ ONLY; parent writers are never called. Reused pure code: fc20260915_shap.group_exact_shapley, the iq / ta
bundle classes and calibrate helper. No fitting happens in this run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MKL_CBWR', 'AVX2,STRICT')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np  # noqa: E402

import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402
import iq20260915_common as Q  # noqa: E402
import ta20260916_common as T  # noqa: E402

ROOT = C.ROOT
OUT = ROOT / 'outputs' / 'balanced_shap_20260916'
SMOKE = OUT / 'smoke_train_only'
FC = C.OUT
CR = K.OUT
IQ = Q.OUT
TA = T.OUT
SPEC = ROOT / 'docs' / 'CLAUDE_BALANCED_SHAP_20260916.md'
PROVENANCE_DOCS = (Q.SPEC, T.SPEC, *Q.PROVENANCE_DOCS)
PYTHON = Q.PYTHON
VERSION = 'balanced_shap_20260916'
ROLE_TAG = 'EXPLORATORY_FOLLOWUP_AFTER_PRIOR_TEST_EXPOSURE_MODEL_DEFINED_LABELS_NOT_CONFIRMATORY'

H = 90
COHORTS = Q.COHORTS
COHORT_CODE = Q.COHORT_CODE
EXPECTED_COUNTS = Q.EXPECTED_COUNTS
EXPECTED_RIDGE_COUNT = Q.EXPECTED_RIDGE_COUNT
MODELS = ('final_q', 'plain_mlp')  # final_q = iq LightGBM winner (five-family Q_SELECT winner); plain_mlp = Track A plain MLP winner
MODEL_SOURCE = {'final_q': ('iq', 'lgbm'), 'plain_mlp': ('ta', 'mlp')}
CELLS = ('all', 'B40')
B40 = (0.40, 0.60)
GROUPS = ('prior_win_probability', 'time_and_observation_age', 'economy_and_experience', 'combat_and_survival', 'objectives', 'structures', 'health_mana_other')
EXPECTED_GROUP_SIZES = {'prior_win_probability': 1, 'time_and_observation_age': 2, 'economy_and_experience': 60, 'combat_and_survival': 68,
                        'objectives': 176, 'structures': 24, 'health_mana_other': 21}
N_EXPLAIN, N_BG, N_CASES = 256, 128, 8
SMOKE_EXPLAIN, SMOKE_BG = 12, 8
BOOT_REPS = Q.BOOT_REPS
BOOT_SEED = Q.BOOT_SEED
BANDS = ((0, 10), (10, 20), (20, 30), (30, 1000))
TAG_EXPLAIN, TAG_BG, TAG_CASE = 'bsh20260916_shap_explain', 'bsh20260916_shap_background', 'bsh20260916_shap_case'
weights = Q.weights
key_join = Q.key_join


def rank_key(tag, *parts):
    return hashlib.sha256(':'.join([tag] + [str(p) for p in parts]).encode()).hexdigest()


def select_rows(tag, rows, match, s_ms, n, *ctx):
    """First n row indices of `rows` by ascending sha256(tag:ctx...:match:s) (independent of any model output)."""
    return np.asarray(sorted(rows, key=lambda i: rank_key(tag, *ctx, match[i], int(s_ms[i])))[:n], dtype=np.int64)


def group_index(names, ridge, shap_groups):
    """Column indices (into the 352 ridge matrix) per group in the fixed GROUPS order; asserts a partition of the 352."""
    pos = {n: j for j, n in enumerate(ridge)}
    gidx = [[pos[n] for n in ridge if shap_groups[n] == g] for g in GROUPS]
    flat = sorted(sum(gidx, []))
    if flat != list(range(len(ridge))):
        raise ValueError('groups do not partition the ridge inputs')
    sizes = {g: len(ix) for g, ix in zip(GROUPS, gidx)}
    if sizes != EXPECTED_GROUP_SIZES:
        raise ValueError(f'group sizes {sizes} differ from the protocol')
    return gidx


def group_map_record(ridge, shap_groups):
    m = {n: shap_groups[n] for n in ridge}
    return dict(order=list(GROUPS), sizes=EXPECTED_GROUP_SIZES, mapping_sha256=C.sha256_json(m), n_inputs=len(ridge))


def model_fn(bundle, calibration):
    """f: 352-column ridge matrix -> final calibrated probability of the frozen bundle (its own prediction definition)."""
    def f(Z):
        return Q.calibrate(bundle, calibration, bundle['base'].raw(np.asarray(Z, dtype=np.float64)))
    return f


def parent_winner(model, coh):
    """(bundle path, calibration, bundle sha256 recorded in the parent freeze, candidate name) of an explained model."""
    src, fam = MODEL_SOURCE[model]
    if src == 'iq':
        fz = C.read_json(Q.frozen_path(IQ))
        w = fz['family_winners'][f'{fam}_{coh}']
        return IQ / w['bundle'], w['calibration'], w['bundle_sha256'], w['chosen']
    fz = C.read_json(T.frozen_path(TA))
    w = fz['family_winners'][f'{fam}_{coh}']
    return TA / w['bundle'], w['calibration'], w['bundle_sha256'], w['chosen']


def parent_eval_predictions(model, coh):
    src, fam = MODEL_SOURCE[model]
    base = IQ if src == 'iq' else TA
    return base / 'eval' / 'predictions' / f'MAIN_TEST_h90_{coh}.npz', f'named__{fam}_winner'


def b40_mask(p_pre):
    p = np.asarray(p_pre, dtype=np.float64)
    return (p >= B40[0]) & (p <= B40[1])


# ------------------------------------------------------------------ parent data (read-only) with access logging in THIS root
def frozen_path(base=OUT):
    return Path(base) / 'frozen_manifest.json'


def log_access(base, purpose, set_name, kind, sealed):
    fz = frozen_path(OUT)
    if sealed and not fz.exists():
        raise PermissionError(f'{purpose}: {set_name} is sealed until {fz} exists')
    Q.append_jsonl(Path(base) / 'access_log.jsonl', dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(), purpose=purpose,
                                                         set=set_name, kind=kind, sealed=bool(sealed), frozen_manifest_exists=fz.exists(),
                                                         frozen_manifest_sha256=C.sha256_file(fz) if fz.exists() else None))


def load_parent_set(name, base, purpose):
    """Features, h90 labels and cohort arrays of MAIN_TRAIN or MAIN_TEST with exact row-order checks (this run's sealed gate)."""
    if name == 'MAIN_TRAIN':
        sealed = False
    elif name == 'MAIN_TEST':
        sealed = True
    else:
        raise ValueError(name)
    log_access(base, purpose, name, 'parent features/labels/cohorts npz', sealed)
    with np.load(FC / 'labels' / f'{name}_features_pre_only.npz', allow_pickle=False) as z:
        F = {k: z[k] for k in ('X_input', 'input_names', 'match', 's_ms', 'sub_role', 'pre_ok')}
    with np.load(FC / 'labels' / f'{name}_labels.npz', allow_pickle=False) as z:
        Lb = {k: z[k] for k in ('match', 's', 'p_pre', 'valid_h90', 'Y_h90')}
    with np.load(CR / 'cohorts' / f'{name}_cohort.npz', allow_pickle=False) as z:
        Co = {k: z[k] for k in ('match', 's', 'cohort', 'valid_h90')}
    checks = dict(feature_label_row_order=bool(np.array_equal(F['match'], Lb['match']) and np.array_equal(F['s_ms'], Lb['s'])),
                  cohort_label_row_order=bool(np.array_equal(Co['match'].astype(str), Lb['match'].astype(str)) and np.array_equal(Co['s'], Lb['s'])),
                  cohort_valid_equals_label_valid=bool(np.array_equal(Co['valid_h90'].astype(int), Lb['valid_h90'].astype(int))))
    ok = F['pre_ok'] == 1
    checks['p_pre_V_column_equals_label_p_pre'] = bool(np.array_equal(F['X_input'][ok, -1], Lb['p_pre'][ok]))
    if not all(checks.values()):
        raise ValueError(f'{name}: parent row contract failed {checks}')
    return F, Lb, Co, checks


# ------------------------------------------------------------------ status / logging (this root only)
class Status(T.Status):
    def __init__(self, group, out=OUT):
        Q.Status.__init__(self, group, out=out)

    def update(self, state, stage, processed=None, total=None, error=None, next_step=None, **extra):
        rec = dict(group=self.group, state=state, stage=stage, pid=os.getpid(), updated=time.strftime('%Y-%m-%d %H:%M:%S'),
                   elapsed_s=round(time.time() - self.t0, 1), processed=processed, total=total, error=error, next_step=next_step, **extra)
        if not self.history or self.history[-1]['stage'] != stage or self.history[-1]['state'] != state:
            self.history.append(dict(stage=stage, state=state, at=rec['updated']))
        rec['history'] = self.history[-100:]
        C.write_json(self.out / 'status' / f'{self.group}.json', rec)
        aggregate_status(self.out)
        self.log(f'STATUS {state} {stage} processed={processed} total={total} error={error} next={next_step} ' + json.dumps(C.clean(extra), ensure_ascii=False)[:600])


def aggregate_status(out=OUT):
    out = Path(out)
    groups = {}
    for p in sorted((out / 'status').glob('*.json')):
        try:
            groups[p.stem] = json.loads(p.read_text(encoding='utf-8'))
        except Exception as exc:
            groups[p.stem] = {'read_error': repr(exc)}
    agg = dict(version=VERSION, role=ROLE_TAG, updated=time.strftime('%Y-%m-%d %H:%M:%S'),
               running_groups=[g for g, r in groups.items() if r.get('state') == 'running'], failed_groups=[g for g, r in groups.items() if r.get('state') == 'failed'],
               groups={g: {k: r.get(k) for k in ('state', 'stage', 'processed', 'total', 'pid', 'elapsed_s', 'error', 'next_step', 'updated')} for g, r in groups.items()})
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
    Q.append_jsonl(OUT / 'failures.jsonl', dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(), stage=stage, error=str(error)[:4000], **info))


def environment():
    return T.environment()


def own_source_hashes():
    files = sorted((ROOT / 'scripts').glob('bsh20260916_*.py')) + sorted((ROOT / 'tests').glob('test_bsh20260916_*.py'))
    reused = [ROOT / 'scripts' / n for n in ('fc20260915_common.py', 'fc20260915_shap.py', 'cr20260915_common.py', 'iq20260915_common.py', 'ta20260916_common.py')] + [SPEC, *PROVENANCE_DOCS]
    return {p.relative_to(ROOT).as_posix(): C.sha256_file(p) for p in files + reused if p.exists()}
