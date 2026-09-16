"""Horizon sensitivity 2026-09-16: shared helpers (pure; importing has no side effects beyond environment defaults).

Design: docs/CLAUDE_HORIZON_SENSITIVITY_20260916.md. Everything written by this run lands in
outputs/horizon_sensitivity_20260916. Parents are READ ONLY; parent writers are never called. Reused pure classes:
iq20260915_common.LinearQBase / LgbmQBase (PT, logistic, LightGBM) and ta20260916_common.MlpQBase (plain / residual MLP).
Bundles are joblib-serialised, so scripts/ must be on sys.path and iq20260915_common / ta20260916_common importable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MKL_CBWR', 'AVX2,STRICT')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np  # noqa: E402

import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402
import iq20260915_common as Q  # noqa: E402
import ta20260916_common as T  # noqa: E402

ROOT = C.ROOT
OUT = ROOT / 'outputs' / 'horizon_sensitivity_20260916'
SMOKE = OUT / 'smoke_train_only'
FC = C.OUT
CR = K.OUT
IQ = Q.OUT
TA = T.OUT
SPEC = ROOT / 'docs' / 'CLAUDE_HORIZON_SENSITIVITY_20260916.md'
PROVENANCE_DOCS = (Q.SPEC, T.SPEC, *Q.PROVENANCE_DOCS)
PYTHON = Q.PYTHON
VERSION = 'horizon_sensitivity_20260916'
ROLE_TAG = 'EXPLORATORY_FOLLOWUP_AFTER_PRIOR_TEST_EXPOSURE_MODEL_DEFINED_LABELS_NOT_CONFIRMATORY'

HORIZONS = (60, 120)
H90 = 90
COHORTS = Q.COHORTS
COHORT_CODE = Q.COHORT_CODE
EXT = Q.EXT
EVAL_SETS = Q.EVAL_SETS
SEALED_SETS = Q.SEALED_SETS
EXPECTED_COUNTS = Q.EXPECTED_COUNTS  # identical at h60 / h90 / h120 (verified against the cohort manifest in the protocol stage)
EXPECTED_RIDGE_COUNT = Q.EXPECTED_RIDGE_COUNT
FAMILIES = ('pt', 'logit', 'lgbm', 'mlp', 'resmlp')
IQ_FAMILIES = ('pt', 'logit', 'lgbm')
TA_FAMILIES = ('mlp', 'resmlp')
CALS = K.CALS
SEEDED = ('lgbm', 'mlp', 'resmlp')
SEEDS = {'lgbm': Q.LGBM_SEEDS, 'mlp': T.SEEDS, 'resmlp': T.SEEDS}
BOOT_REPS = Q.BOOT_REPS
BOOT_SEED = Q.BOOT_SEED
SPARSE_MATCHES = Q.SPARSE_MATCHES
LEGACY = Q.LEGACY  # evaluation name -> key in cohort-role A_<set>_h<h>_<cohort>.npz (None -> spec_<specialist_chosen>)
WINNERS = tuple(f'{f}_winner' for f in FAMILIES)
NAMED = WINNERS + ('overall_winner',) + tuple(LEGACY)
BOOT_MODELS = WINNERS
CONTRASTS = (
    ('lgbm_winner', 'pt_winner', 'PRIMARY: full LightGBM - PT baseline'),
    ('logit_winner', 'pt_winner', 'planned: full logistic - PT baseline'),
    ('lgbm_winner', 'logit_winner', 'planned: full LightGBM - full logistic'),
    ('mlp_winner', 'lgbm_winner', 'planned: plain MLP - full LightGBM'),
    ('resmlp_winner', 'lgbm_winner', 'planned: residual MLP - full LightGBM'),
    ('mlp_winner', 'logit_winner', 'planned: plain MLP - full logistic'))

split_candidate = Q.split_candidate
select_rule = Q.select_rule
weights = Q.weights
weight_record = Q.weight_record
membership_sha = Q.membership_sha
smoke_keep = Q.smoke_keep
cell_masks = Q.cell_masks
cell_metrics = Q.cell_metrics
paired_bootstrap = Q.paired_bootstrap
key_join = Q.key_join
oof_provenance = Q.oof_provenance
calibrate = Q.calibrate
bundle_raw = Q.bundle_raw
bundle_predict = Q.bundle_predict
calibrator_record = Q.calibrator_record


# ------------------------------------------------------------------ fixed configurations (h90 winners of iq and ta)
def h90_registry():
    """{cohort: {family: config}} read from the iq and ta frozen manifests (their h90 family winners)."""
    iq_fz = C.read_json(Q.frozen_path(IQ))
    ta_fz = C.read_json(T.frozen_path(TA))
    reg = {}
    for coh in COHORTS:
        reg[coh] = {}
        for f in IQ_FAMILIES:
            reg[coh][f] = iq_fz['family_winners'][f'{f}_{coh}']['config']
        for f in TA_FAMILIES:
            reg[coh][f] = ta_fz['family_winners'][f'{f}_{coh}']['config']
    return reg, dict(iq_frozen_manifest_sha256=C.sha256_file(Q.frozen_path(IQ)), ta_frozen_manifest_sha256=C.sha256_file(T.frozen_path(TA)))


def config_params(family, config):
    if family in IQ_FAMILIES:
        return Q.config_params(family, config)
    return T.config_params(family, config)


def candidate_names(config):
    """3 joint candidates '<config>__<calibration>' in lexical tie order (calibration-only selection)."""
    return sorted(f'{config}__{cal}' for cal in CALS)


def family_inputs(family, ridge):
    return list(Q.PT_INPUTS) if family == 'pt' else list(ridge)


def input_columns(names, family, ridge):
    return [names.index(n) for n in family_inputs(family, ridge)]


def fit_family(family, config, input_names, Xtr, y, g, w, device='cuda', log=None):
    """Dispatch to the reused pure classes. Returns (base, eligible, fit_record, stop_record_or_None)."""
    if family in ('pt', 'logit'):
        base = Q.LinearQBase(family, config, input_names).fit(Xtr, y, w)
        rec = dict(attempts=base.attempts, eligible_converged=base.eligible, design=base.design_record, params=dict(C=base.C_value))
        return base, bool(base.eligible), rec, None
    if family == 'lgbm':
        base = Q.LgbmQBase(config, input_names).fit(Xtr, y, g, log=log)
        rec = dict(params=base.params, seeds=list(base.seeds),
                   seeds_summary={s: {k: v for k, v in r.items() if k != 'stop_curve'} for s, r in base.stop_record['seeds'].items()},
                   imputer_full_n_features=int(base.imputer.n_features_in_))
        return base, True, rec, base.stop_record
    if family in TA_FAMILIES:
        base = T.MlpQBase(family, config, input_names, device=device).fit(Xtr, y, g, log=log)
        rec = dict(params=base.params, seeds=list(base.seeds), arch=base.arch,
                   seeds_summary={s: {k: v for k, v in r.items() if k != 'stop_curve'} for s, r in base.stop_record['seeds'].items()},
                   imputer_full_n_features=int(base.imputer.n_features_in_), scaler_n_samples_seen=int(np.max(base.scaler.n_samples_seen_)))
        return base, True, rec, base.stop_record
    raise ValueError(family)


def make_bundle(family, config, horizon, base, calibrators, input_names_all, ridge, manifest):
    return dict(version=VERSION, family=family, config=config, horizon_s=int(horizon), params=config_params(family, config),
                input_names_all_sha256=C.sha256_json(list(input_names_all)), input_names=family_inputs(family, ridge),
                input_columns=input_columns(list(input_names_all), family, ridge), base=base, calibrators=calibrators, manifest=manifest)


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


def label_keys(h):
    return ('match', 's', 'sub_role', 'p_pre', f'valid_h{h}', f'Y_h{h}', 'adapter_id', 'adapter_sha256')


def load_parent_set(name, base, purpose, h):
    """Features (pre-only X_input), h-labels and cohort arrays of one parent set with exact row-order checks."""
    if name.startswith('MAIN_TRAIN') or name == 'MAIN_VALIDATION':
        sealed = False
    elif name in SEALED_SETS:
        sealed = True
    else:
        raise ValueError(name)
    log_access(base, purpose, name, f'parent features/labels/cohorts npz (h{h})', sealed)
    with np.load(FC / 'labels' / f'{name}_features_pre_only.npz', allow_pickle=False) as z:
        F = {k: z[k] for k in ('X_input', 'input_names', 'match', 's_ms', 'sub_role', 'pre_ok')}
    with np.load(FC / 'labels' / f'{name}_labels.npz', allow_pickle=False) as z:
        Lb = {k: z[k] for k in label_keys(h) if k in z.files}
    with np.load(CR / 'cohorts' / f'{name}_cohort.npz', allow_pickle=False) as z:
        Co = {k: z[k] for k in ('match', 's', 'cohort', f'valid_h{h}', 'scale_known')}
    checks = dict(
        feature_label_row_order=bool(np.array_equal(F['match'], Lb['match']) and np.array_equal(F['s_ms'], Lb['s'])),
        cohort_label_row_order=bool(np.array_equal(Co['match'].astype(str), Lb['match'].astype(str)) and np.array_equal(Co['s'], Lb['s'])),
        cohort_valid_equals_label_valid=bool(np.array_equal(Co[f'valid_h{h}'].astype(int), Lb[f'valid_h{h}'].astype(int))))
    ok = F['pre_ok'] == 1
    checks['p_pre_V_column_equals_label_p_pre'] = bool(np.array_equal(F['X_input'][ok, -1], Lb['p_pre'][ok]))
    if not all(checks.values()):
        raise ValueError(f'{name} h{h}: parent row contract failed {checks}')
    return F, Lb, Co, checks


def load_trainval(base, h, smoke=False, cohort=None):
    """TRAIN (+ VALIDATION Q_CAL/Q_SELECT) h-valid rows of one cohort with role labels. smoke: TRAIN only, pseudo roles, 1/8 matches."""
    parts = ['MAIN_TRAIN'] if smoke else ['MAIN_TRAIN', 'MAIN_VALIDATION']
    D = dict(parts=parts, smoke=bool(smoke), horizon=int(h), checks={})
    Xs, gs, ss, rs, ys, ps, cs, aids, ashs = [], [], [], [], [], [], [], [], []
    names = None
    for part in parts:
        F, Lb, Co, ch = load_parent_set(part, base, f'fit data {part} h{h} ({"smoke TRAIN-only" if smoke else "full"})', h)
        D['checks'][part] = ch
        nm = [str(x) for x in F['input_names']]
        if names is None:
            names = nm
        elif nm != names:
            raise ValueError('input schema differs between TRAIN and VALIDATION')
        sr = F['sub_role'].astype(str)
        keep = (Lb[f'valid_h{h}'] == 1) & (Co['cohort'] >= 0)
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
        ys.append(Lb[f'Y_h{h}'][keep].astype(np.int64))
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


def legacy_path(set_name, h, coh):
    return CR / 'eval' / 'predictions' / f'A_{set_name}_h{h}_{coh}.npz'


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
        self.log(f'STATUS {state} {stage} processed={processed} total={total} error={error} next={next_step} '
                 + json.dumps(C.clean(extra), ensure_ascii=False)[:600])


def aggregate_status(out=OUT):
    out = Path(out)
    groups = {}
    for p in sorted((out / 'status').glob('*.json')):
        try:
            groups[p.stem] = json.loads(p.read_text(encoding='utf-8'))
        except Exception as exc:
            groups[p.stem] = {'read_error': repr(exc)}
    agg = dict(version=VERSION, role=ROLE_TAG, updated=time.strftime('%Y-%m-%d %H:%M:%S'),
               running_groups=[g for g, r in groups.items() if r.get('state') == 'running'],
               failed_groups=[g for g, r in groups.items() if r.get('state') == 'failed'],
               groups={g: {k: r.get(k) for k in ('state', 'stage', 'processed', 'total', 'pid', 'elapsed_s', 'error', 'next_step', 'updated')}
                       for g, r in groups.items()},
               state_vocabulary='running; complete = stage finished; failed (attempt logs retained); verified only via validation.json')
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
    files = sorted((ROOT / 'scripts').glob('hs20260916_*.py')) + sorted((ROOT / 'tests').glob('test_hs20260916_*.py'))
    reused = [ROOT / 'scripts' / n for n in ('fc20260915_common.py', 'cr20260915_common.py', 'iq20260915_common.py', 'ta20260916_common.py')] + [SPEC, *PROVENANCE_DOCS]
    return {p.relative_to(ROOT).as_posix(): C.sha256_file(p) for p in files + reused if p.exists()}
