"""Development-only definition 2026-09-16: shared helpers (pure; importing sets only environment defaults).

Design: docs/CLAUDE_DEFINITION_DEV_20260916.md. Everything written by this run lands in outputs/definition_dev_20260916.
Parents (full corpus, cohort-role, incremental q, parent exposures, caches, original repository) are READ ONLY.
"""
from __future__ import annotations

import csv
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

ROOT = C.ROOT
OUT = ROOT / 'outputs' / 'definition_dev_20260916'
FC = C.OUT
CR = K.OUT
IQ = Q.OUT
REPO = C.REPO
SPEC = ROOT / 'docs' / 'CLAUDE_DEFINITION_DEV_20260916.md'
PROVENANCE_DOCS = (ROOT / 'docs' / 'TOG_END_TO_END_READINESS_AUDIT_20260915.md', ROOT / 'docs' / 'COLLABORATOR_CRITIQUE_RESPONSE_20260915.md')
DEV_SPEC = REPO / 'config' / 'fight_boundary' / 'spec_15.14.json'
POOLED_SPEC = REPO / 'config' / 'fight_boundary' / 'spec_pooled.json'
PYTHON = Q.PYTHON
VERSION = 'definition_dev_20260916'
ROLE_TAG = Q.ROLE_TAG
H = Q.H
HS = C.HORIZONS_S
COHORTS = Q.COHORTS
COHORT_CODE = Q.COHORT_CODE
EXT = Q.EXT
EVAL_SETS = Q.EVAL_SETS
SEALED_SETS = Q.SEALED_SETS
MAIN_SETS = ('MAIN_TRAIN', 'MAIN_VALIDATION', 'MAIN_TEST')
ALL_SETS = MAIN_SETS + tuple(f'EXT_{s}' for s in EXT)
EXPECTED_COUNTS_OLD = Q.EXPECTED_COUNTS
EXPECTED_RIDGE_COUNT = Q.EXPECTED_RIDGE_COUNT
FAMILIES = ('logit', 'lgbm')
CALS = K.CALS
SEEDED = ('lgbm',)
SEEDS = {'lgbm': Q.LGBM_SEEDS}
BOOT_REPS = Q.BOOT_REPS
BOOT_SEED = Q.BOOT_SEED
SPARSE_MATCHES = Q.SPARSE_MATCHES
EXPO_FIELDS = ('match', 'patch', 's', 'L', 'next_start', 'end', 'same_match_overlap', 'end_observed')
DET_KEYS = ('det_cluster_blue', 'det_cluster_red', 'det_present_blue', 'det_present_red')
DEFINITIONS = {'frozen': dict(TF2_KILL_CLUSTER_GAP_MS=13700, CLUSTER_MAX_DIAMETER=4264.0),
               'dev': dict(TF2_KILL_CLUSTER_GAP_MS=14000, CLUSTER_MAX_DIAMETER=4285.0)}
DEV_ROUNDING = 'G rounded to 0.1 s (13.9637 -> 14.0 s), D to 1 u (4285.26 -> 4285), as the frozen preset rounds the pooled estimates (13.7246 -> 13.7 s, 4263.87 -> 4264)'
NAMED = ('refit_logit', 'refit_lgbm', 'frozen_logit', 'frozen_lgbm', 'frozen_pt')
BOOT_MODELS = NAMED
CONTRASTS = (('refit_lgbm', 'frozen_lgbm', 'PRIMARY: refit LightGBM - frozen parent LightGBM (dev population)'),
             ('refit_logit', 'frozen_logit', 'planned: refit logistic - frozen parent logistic (dev population)'),
             ('frozen_lgbm', 'frozen_pt', 'planned: frozen LightGBM - frozen PT (dev population)'),
             ('refit_lgbm', 'frozen_pt', 'planned: refit LightGBM - frozen PT (dev population)'))
FIXTURE_MATCHES = 100
FROZEN_CHECK_MATCHES = 2000

split_candidate = Q.split_candidate
select_rule = Q.select_rule
weights = Q.weights
weight_record = Q.weight_record
membership_sha = Q.membership_sha
cell_masks = Q.cell_masks
cell_metrics = Q.cell_metrics
paired_bootstrap = Q.paired_bootstrap
key_join = Q.key_join
oof_provenance = Q.oof_provenance
calibrate = Q.calibrate
calibrator_record = Q.calibrator_record
config_params = Q.config_params


def ts(s):
    return time.mktime(time.strptime(s, '%Y-%m-%d %H:%M:%S'))


def candidate_names(config):
    return sorted(f'{config}__{cal}' for cal in CALS)


def registry():
    iq_fz = C.read_json(Q.frozen_path(IQ))
    reg = {coh: {f: iq_fz['family_winners'][f'{f}_{coh}']['config'] for f in FAMILIES} for coh in COHORTS}
    winners = {k: dict(chosen=v['chosen'], bundle=v['bundle'], bundle_sha256=v['bundle_sha256'], calibration=v.get('calibration', split_candidate(v['chosen'])[1]))
               for k, v in iq_fz['family_winners'].items()}
    return reg, winners, dict(iq_frozen_manifest_sha256=C.sha256_file(Q.frozen_path(IQ)))


# ------------------------------------------------------------------ detector (parent code path, definition switch by env)
def set_definition_env(definition, cache_dirname):
    if definition not in DEFINITIONS:
        raise ValueError(definition)
    os.environ['LOL_CFG_PRESET'] = 'v3.3'
    os.environ['LOL_CFG_OVERRIDES'] = json.dumps({'CACHE_DIRNAME': str(cache_dirname), 'FIGHT_INDEX_CACHE_ENABLED': False, 'FIGHT_INDEX_NUM_WORKERS': 1,
                                                  'DUMP_FIGHTS': False, 'CACHE_IN_RAM': False, **DEFINITIONS[definition]})
    os.environ['LOL_OUTPUT_ROOT'] = str(OUT / 'runtime')
    os.environ['DD_DEFINITION'] = definition


_CTX = {}


def detector_context(definition):
    """Import the original repository detector with the definition's constants (must be the first repo import of the process)."""
    if _CTX:
        if _CTX['definition'] != definition:
            raise RuntimeError(f'detector already imported with definition {_CTX["definition"]}')
        return _CTX
    set_definition_env(definition, OUT / 'runtime' / '_unused_cache_dir')
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if str(ROOT / 'scripts') not in sys.path:
        sys.path.insert(0, str(ROOT / 'scripts'))
    from core.config import cfg
    import data.cache_io as cio
    from gameplay.fights import detect_fights
    from data.index_split import _fight_to_ref_row
    import measure_postkill_objective_delay as MPD
    cfg.CACHE_IN_RAM = False
    cfg.DUMP_FIGHTS = False
    got = {k: getattr(cfg, k, None) for k in DEFINITIONS[definition]}
    if int(got['TF2_KILL_CLUSTER_GAP_MS']) != DEFINITIONS[definition]['TF2_KILL_CLUSTER_GAP_MS'] or float(got['CLUSTER_MAX_DIAMETER']) != DEFINITIONS[definition]['CLUSTER_MAX_DIAMETER']:
        raise RuntimeError(f'detector constants not applied: {got}')
    _CTX.update(definition=definition, cfg=cfg, cio=cio, detect=detect_fights, ref=_fight_to_ref_row, MPD=MPD)
    return _CTX


def detector_settings(definition):
    cfg = detector_context(definition)['cfg']
    keys = ('TF2_KILL_CLUSTER_GAP_MS', 'CLUSTER_MAX_DIAMETER', 'TF2_VALIDITY_RADIUS', 'TF2_ENGAGE_PRE_KILL_MS', 'TF2_MIN_PER_TEAM',
            'TF2_INTERACTION_RADIUS', 'TF2_EXCLUDE_SHOP_INTERACTIONS', 'FIGHT_HORIZON_SEC', 'FEATURE_VERSION', 'FRAME_MS', 'PATCH_LEVEL')
    return {k: getattr(cfg, k, None) for k in keys}


def detect_chunk(task):
    """Re-run the detector on one list of matches of one cache directory under task['definition']; per exposure row the 8 parent
    fields plus the raw cluster / presence counts of the fight that produced it."""
    ctx = detector_context(task['definition'])
    cio = ctx['cio']
    cio.CACHE_DIR = Path(task['cache_dir'])
    ctx['MPD'].CACHE = Path(task['cache_dir'])
    out = []
    for mid in task['matches']:
        rec = dict(match=mid, loaded=0, error='', rows=[])
        try:
            pack = cio.load_match_cache(mid)
            if pack is None:
                rec['error'] = 'cache_not_loaded'
                out.append(rec)
                continue
            rec['loaded'] = 1
            patch = pack['meta']['patch']
            fights = ctx['detect'](pack, pack['meta']['team_map'])
            refs, counts = [], {}
            for f in fights:
                r = ctx['ref'](f, pack['minute_ts'], match_id=mid, patch=patch)
                if r is None:
                    continue
                refs.append(dict(patch=patch, s=int(r['t_start_ts']), L=int(r['last_kill_ts']), x=r['anchor_x'], y=r['anchor_y']))
                counts[(int(r['t_start_ts']), int(r['last_kill_ts']))] = dict(
                    cluster_blue=int(f['det_cluster_blue']) if isinstance(f.get('det_cluster_blue'), (int, np.integer)) else -1,
                    cluster_red=int(f['det_cluster_red']) if isinstance(f.get('det_cluster_red'), (int, np.integer)) else -1,
                    present_blue=int(f['det_present_blue']) if isinstance(f.get('det_present_blue'), (int, np.integer)) else -1,
                    present_red=int(f['det_present_red']) if isinstance(f.get('det_present_red'), (int, np.integer)) else -1,
                    ref_cluster_blue=int(r['det_cluster_blue']), ref_cluster_red=int(r['det_cluster_red']))
            _pairs, ex, _stats = ctx['MPD'].analyse_match((mid, refs, float('inf'), 4264))
            for e in ex:
                c = counts.get((int(e['s']), int(e['L'])), {})
                rec['rows'].append(dict({k: str(e[k]) for k in EXPO_FIELDS}, **{k: int(c.get(k, -1)) for k in ('cluster_blue', 'cluster_red', 'present_blue', 'present_red', 'ref_cluster_blue', 'ref_cluster_red')}))
        except Exception as exc:
            rec['error'] = f'{type(exc).__name__}:{exc}'[:300]
        out.append(rec)
    return out


# ------------------------------------------------------------------ parent tables (read-only)
def main_manifest():
    rows = list(csv.DictReader(open(C.MAIN_MANIFEST, encoding='utf-8', newline='')))
    if len(rows) != C.MAIN_TOTAL:
        raise ValueError('main manifest total mismatch')
    return rows


def set_of_role(role):
    return {'TRAIN': 'MAIN_TRAIN', 'VALIDATION': 'MAIN_VALIDATION', 'TEST': 'MAIN_TEST'}[role]


def sub_role_of(role, mid):
    if role == 'TRAIN':
        return f'fold{C.train_fold(mid)}'
    if role == 'VALIDATION':
        return C.validation_role(mid)
    return role


def external_dir(set_name):
    return FC / 'external' / set_name.replace('EXT_', '')


def parent_exposures(set_name):
    """{match: [8-tuple of str, ...]} of the parent exposures of one set (MAIN_* share the parent csv)."""
    path = C.EXPOSURES if set_name.startswith('MAIN') else external_dir(set_name) / 'exposures.csv'
    by = {}
    with open(path, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            by.setdefault(r['match'], []).append(tuple(r[k] for k in EXPO_FIELDS))
    return by, path


def set_matches(set_name):
    """[(match_id, role, sub_role, patch, cache_dir)] of one set in parent manifest order."""
    if set_name.startswith('MAIN'):
        role = {'MAIN_TRAIN': 'TRAIN', 'MAIN_VALIDATION': 'VALIDATION', 'MAIN_TEST': 'TEST'}[set_name]
        return [(r['match_id'], role, sub_role_of(role, r['match_id']), r['patch'], str(C.CACHE_MAIN)) for r in main_manifest() if r['role'] == role]
    d = external_dir(set_name)
    man = C.read_json(d / 'prepared_manifest.json')
    if man['status'] != 'complete':
        raise ValueError('external preparation incomplete')
    return [(r['match_id'], set_name, 'EXTERNAL', r['api_patch'], str(d / 'cache')) for r in man['matches'] if r['cache_status'] == 'written']


def redetect_path(set_name, definition):
    return OUT / 'redetect' / f'{set_name}_{definition}.csv'


def read_redetect(set_name, definition):
    by = {}
    with open(redetect_path(set_name, definition), encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            by.setdefault(r['match'], []).append(r)
    return by


# ------------------------------------------------------------------ populations (this root) with access logging
def frozen_path(base=OUT):
    return Path(base) / 'frozen_manifest.json'


def log_access(base, purpose, set_name, kind, sealed):
    fz = frozen_path(OUT)
    if sealed and not fz.exists():
        raise PermissionError(f'{purpose}: {set_name} is sealed until {fz} exists')
    Q.append_jsonl(Path(base) / 'access_log.jsonl', dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(), purpose=purpose, set=set_name, kind=kind,
                                                         sealed=bool(sealed), frozen_manifest_exists=fz.exists(), frozen_manifest_sha256=C.sha256_file(fz) if fz.exists() else None))


def is_sealed(set_name):
    return set_name in SEALED_SETS


def load_parent_set(name, base, purpose):
    """Parent (old-definition) features, labels and cohorts of one set (exact copy of the iq loader; own access log)."""
    log_access(base, purpose, name, 'parent features/labels/cohorts npz (old definition)', is_sealed(name))
    with np.load(FC / 'labels' / f'{name}_features_pre_only.npz', allow_pickle=False) as z:
        F = {k: z[k] for k in ('X_input', 'input_names', 'match', 's_ms', 'sub_role', 'pre_ok')}
    with np.load(FC / 'labels' / f'{name}_labels.npz', allow_pickle=False) as z:
        Lb = {k: z[k] for k in z.files if k in Q.LABEL_KEYS or k.startswith(('p_post_h', 'valid_h', 'Y_h', 'endpoint_h'))}
    with np.load(CR / 'cohorts' / f'{name}_cohort.npz', allow_pickle=False) as z:
        Co = {k: z[k] for k in ('match', 's', 'cohort', 'valid_h90', 'scale_known', 'n_min')}
    ok = (np.array_equal(F['match'], Lb['match']) and np.array_equal(F['s_ms'], Lb['s']) and np.array_equal(Co['match'].astype(str), Lb['match'].astype(str))
          and np.array_equal(Co['s'], Lb['s']) and np.array_equal(Co['valid_h90'].astype(int), Lb['valid_h90'].astype(int)))
    if not ok:
        raise ValueError(f'{name}: parent row contract failed')
    return F, Lb, Co


def load_dev_set(name, base, purpose):
    """Dev-population features, labels and cohorts of one set written by the rebuild stage (exact row-order checks)."""
    log_access(base, purpose, name, 'dev population features/labels/cohorts npz', is_sealed(name))
    with np.load(OUT / 'labels' / f'{name}_features_pre_only.npz', allow_pickle=False) as z:
        F = {k: z[k] for k in ('X_input', 'input_names', 'match', 's_ms', 'sub_role', 'pre_ok', 'row_source')}
    with np.load(OUT / 'labels' / f'{name}_labels.npz', allow_pickle=False) as z:
        Lb = {k: z[k] for k in z.files}
    with np.load(OUT / 'cohorts' / f'{name}_cohort.npz', allow_pickle=False) as z:
        Co = {k: z[k] for k in z.files}
    ok = (np.array_equal(F['match'], Lb['match']) and np.array_equal(F['s_ms'], Lb['s']) and np.array_equal(Co['match'], Lb['match']) and np.array_equal(Co['s'], Lb['s'])
          and np.array_equal(Co['valid_h90'].astype(int), Lb['valid_h90'].astype(int)))
    okp = F['pre_ok'] == 1
    ok = ok and np.array_equal(F['X_input'][okp, -1], Lb['p_pre'][okp])
    if not ok:
        raise ValueError(f'{name}: dev row contract failed')
    return F, Lb, Co


def load_trainval(base, cohort):
    """Dev TRAIN (+ VALIDATION Q_CAL / Q_SELECT) h90-valid rows of one cohort with role labels."""
    D = dict(checks={})
    Xs, gs, ss, rs, ys, ps, aids, ashs, srcs = [], [], [], [], [], [], [], [], []
    names = None
    for part in ('MAIN_TRAIN', 'MAIN_VALIDATION'):
        F, Lb, Co = load_dev_set(part, base, f'fit data {part} (dev population)')
        nm = [str(x) for x in F['input_names']]
        if names is None:
            names = nm
        elif nm != names:
            raise ValueError('input schema differs between TRAIN and VALIDATION')
        sr = F['sub_role'].astype(str)
        keep = (Lb['valid_h90'] == 1) & (Co['cohort'] >= 0) & (Co['cohort'] == COHORT_CODE[cohort])
        if part == 'MAIN_VALIDATION':
            keep &= np.isin(sr, ['Q_CAL', 'Q_SELECT'])
        if not np.all(F['pre_ok'][keep] == 1):
            raise ValueError('valid rows without pre inputs')
        Xs.append(F['X_input'][keep])
        gs.append(F['match'][keep].astype(str))
        ss.append(F['s_ms'][keep])
        rs.append(sr[keep])
        ys.append(Lb['Y_h90'][keep].astype(np.int64))
        ps.append(Lb['p_pre'][keep])
        aids.append(Lb['adapter_id'][keep].astype(str))
        ashs.append(Lb['adapter_sha256'][keep].astype(str))
        srcs.append(F['row_source'][keep].astype(str))
        F = Lb = Co = None
    D['names'] = names
    D['X'] = np.vstack(Xs)
    D['g'] = np.concatenate(gs)
    D['s_ms'] = np.concatenate(ss)
    D['sr_true'] = np.concatenate(rs)
    D['y'] = np.concatenate(ys)
    D['p_pre'] = np.concatenate(ps)
    D['adapter_id'] = np.concatenate(aids)
    D['adapter_sha256'] = np.concatenate(ashs)
    D['row_source'] = np.concatenate(srcs)
    D['role'] = np.where(np.char.startswith(D['sr_true'], 'fold'), 'TRAIN', D['sr_true'])
    return D


def fit_family(family, config, input_names, Xtr, y, g, w, log=None):
    if family == 'logit':
        base = Q.LinearQBase('logit', config, input_names).fit(Xtr, y, w)
        return base, bool(base.eligible), dict(attempts=base.attempts, eligible_converged=base.eligible, design=base.design_record, params=dict(C=base.C_value)), None
    if family == 'lgbm':
        base = Q.LgbmQBase(config, input_names).fit(Xtr, y, g, log=log)
        rec = dict(params=base.params, seeds=list(base.seeds), seeds_summary={s: {k: v for k, v in r.items() if k != 'stop_curve'} for s, r in base.stop_record['seeds'].items()},
                   imputer_full_n_features=int(base.imputer.n_features_in_))
        return base, True, rec, base.stop_record
    raise ValueError(family)


def make_bundle(family, config, base, calibrators, names_all, ridge, manifest):
    return dict(version=VERSION, family=family, config=config, arm='base', params=config_params(family, config), input_names_all_sha256=C.sha256_json(list(names_all)),
                input_names=list(ridge), input_columns=[list(names_all).index(n) for n in ridge], base=base, calibrators=calibrators, manifest=manifest)


def bundle_raw(b, X_all, names_all=None):
    if names_all is not None and C.sha256_json(list(names_all)) != b['input_names_all_sha256']:
        raise ValueError('input schema differs from the bundle schema')
    X = np.asarray(X_all, dtype=np.float64)
    cols = list(b['input_columns'])
    Xin = X if cols == list(range(X.shape[1])) else X[:, cols]
    return b['base'].raw(np.ascontiguousarray(Xin))


def bundle_predict(b, cal, X_all, names_all=None):
    return calibrate(b, cal, bundle_raw(b, X_all, names_all))


def parent_bundle_raw(b, X_all, names_all):
    """Raw prediction of a parent iq bundle (fancy-indexed columns as the parent did, so parent predictions reproduce bitwise)."""
    return Q.bundle_raw(b, X_all, names_all)


# ------------------------------------------------------------------ status / logging (this root only)
class Status(Q.Status):
    def __init__(self, group, out=OUT):
        super().__init__(group, out=out)

    def update(self, state, stage, processed=None, total=None, error=None, next_step=None, **extra):
        rec = dict(group=self.group, state=state, stage=stage, pid=os.getpid(), updated=time.strftime('%Y-%m-%d %H:%M:%S'), elapsed_s=round(time.time() - self.t0, 1),
                   processed=processed, total=total, error=error, next_step=next_step, **extra)
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
    agg = dict(version=VERSION, role=ROLE_TAG, updated=time.strftime('%Y-%m-%d %H:%M:%S'), running_groups=[g for g, r in groups.items() if r.get('state') == 'running'],
               failed_groups=[g for g, r in groups.items() if r.get('state') == 'failed'],
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
    return Q.environment()


def own_source_hashes():
    files = sorted((ROOT / 'scripts').glob('dd20260916_*.py')) + sorted((ROOT / 'tests').glob('test_dd20260916_*.py'))
    reused = [ROOT / 'scripts' / n for n in ('fc20260915_common.py', 'cr20260915_common.py', 'iq20260915_common.py', 'fc20260915_extract.py', 'measure_postkill_objective_delay.py')] + [SPEC, *PROVENANCE_DOCS]
    return {p.relative_to(ROOT).as_posix(): C.sha256_file(p) for p in files + reused if p.exists()}
