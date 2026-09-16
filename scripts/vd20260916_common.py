"""V mechanism 2026-09-16: shared helpers (pure; no side effects beyond environment defaults). Design: docs/CLAUDE_V_MECHANISM_20260916.md."""
from __future__ import annotations

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
OUT = ROOT / 'outputs' / 'v_mechanism_20260916'
SMOKE = OUT / 'smoke_validation_only'
FC = C.OUT
CR = K.OUT
SPEC = ROOT / 'docs' / 'CLAUDE_V_MECHANISM_20260916.md'
PROVENANCE_DOCS = (ROOT / 'docs' / 'COLLABORATOR_CRITIQUE_RESPONSE_20260915.md', ROOT / 'docs' / 'TOG_END_TO_END_READINESS_AUDIT_20260915.md')
PYTHON = Q.PYTHON
VERSION = 'v_mechanism_20260916'
ROLE_TAG = 'EXPLORATORY_FOLLOWUP_AFTER_PRIOR_TEST_EXPOSURE_MODEL_DEFINED_LABELS_NOT_CONFIRMATORY'
H = 90
COHORTS = Q.COHORTS
COHORT_CODE = Q.COHORT_CODE
EXPECTED_COUNTS = Q.EXPECTED_COUNTS
SETS = ('MAIN_VALIDATION', 'MAIN_TEST')
GROUPS = ('time_and_observation_age', 'economy_and_experience', 'combat_and_survival', 'objectives', 'structures', 'health_mana_other', 'champion_identity')
BANDS = ((0, 10), (10, 20), (20, 30), (30, 1000))
DELTA_STRATA = ((0.0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 10.0))
TOP_COLUMNS = 25
V_ADAPTER = FC / 'models' / 'v' / 'v_final_raw.joblib'
STATES_DIR = FC / 'extract' / 'MAIN' / 'states'
weights = Q.weights


def transformed_names(adapter):
    """Transformed column names of the frozen pipeline's preprocessing step (numeric__<name> / champions__<slot>_<id>)."""
    return [str(n) for n in adapter.base[:-1].get_feature_names_out(np.array(adapter.feature_names))]


def column_group(tname):
    if tname.startswith('champions__'):
        return 'champion_identity'
    name = tname.split('__', 1)[1]
    g = C.shap_group(name)
    if g not in GROUPS:
        raise ValueError(f'unexpected group {g} for {tname}')
    return g


def is_time_related(tname):
    name = tname.split('__', 1)[1]
    return name in ('time_minutes', 'time_minutes_sq') or name.endswith('_x_time')


def column_map(adapter):
    names = transformed_names(adapter)
    groups = [column_group(n) for n in names]
    gidx = {g: np.asarray([j for j, gg in enumerate(groups) if gg == g], dtype=np.int64) for g in GROUPS}
    if sorted(np.concatenate(list(gidx.values())).tolist()) != list(range(len(names))):
        raise ValueError('groups do not partition the transformed columns')
    tmask = np.asarray([is_time_related(n) for n in names], dtype=bool)
    return names, groups, gidx, tmask


def beta(adapter):
    lr = adapter.base.steps[-1][1]
    return np.asarray(lr.coef_[0], dtype=np.float64), float(lr.intercept_[0])


def transform(adapter, X_state):
    """State matrix (362 columns in adapter.state_names order) -> transformed 1,733-column dense float64 matrix."""
    X = np.asarray(X_state, dtype=np.float64)[:, adapter.keep]
    Z = adapter.base[:-1].transform(X)
    if hasattr(Z, 'toarray'):
        Z = Z.toarray()
    return np.asarray(Z, dtype=np.float64)


def decision(adapter, X_state):
    return adapter.base.decision_function(np.asarray(X_state, dtype=np.float64)[:, adapter.keep])


def delta_stratum(abs_delta):
    a = np.asarray(abs_delta, dtype=np.float64)
    out = np.full(len(a), -1, dtype=np.int64)
    for k, (lo, hi) in enumerate(DELTA_STRATA):
        out[(a > lo) & (a <= hi) if k else (a >= lo) & (a <= hi)] = k
    return out


def stratum_name(k):
    lo, hi = DELTA_STRATA[k]
    return f'{lo * 100:g}-{hi * 100:g}pp' if hi < 10 else f'>{lo * 100:g}pp'


def band_index(minutes):
    m = np.asarray(minutes, dtype=np.float64)
    out = np.full(len(m), -1, dtype=np.int64)
    for k, (lo, hi) in enumerate(BANDS):
        out[(m >= lo) & (m < hi)] = k
    return out


def logit(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(p) - np.log1p(-p)


# ------------------------------------------------------------------ parent data with access logging in THIS root
def frozen_path(base=OUT):
    return Path(base) / 'frozen_manifest.json'


def log_access(base, purpose, set_name, kind, sealed):
    fz = frozen_path(OUT)
    if sealed and not fz.exists():
        raise PermissionError(f'{purpose}: {set_name} is sealed until {fz} exists')
    Q.append_jsonl(Path(base) / 'access_log.jsonl', dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(), purpose=purpose, set=set_name, kind=kind,
                                                         sealed=bool(sealed), frozen_manifest_exists=fz.exists(), frozen_manifest_sha256=C.sha256_file(fz) if fz.exists() else None))


def load_labels(name, base, purpose):
    """Parent label arrays (h90) and cohort arrays of MAIN_VALIDATION (unsealed) or MAIN_TEST (sealed)."""
    if name == 'MAIN_VALIDATION':
        sealed = False
    elif name == 'MAIN_TEST':
        sealed = True
    else:
        raise ValueError(name)
    log_access(base, purpose, name, 'parent labels/cohorts npz', sealed)
    with np.load(FC / 'labels' / f'{name}_labels.npz', allow_pickle=False) as z:
        Lb = {k: z[k] for k in ('match', 's', 'sub_role', 'adapter_id', 'adapter_sha256', 'p_pre', 'pre_snapshot', 'valid_h90', 'Y_h90', 'p_post_h90', 'delta_h90',
                                'post_snapshot_h90', 'post_snapshot_after_L_h90', 'endpoint_h90', 'L')}
    with np.load(CR / 'cohorts' / f'{name}_cohort.npz', allow_pickle=False) as z:
        Co = {k: z[k] for k in ('match', 's', 'cohort', 'valid_h90')}
    if not (np.array_equal(Co['match'].astype(str), Lb['match'].astype(str)) and np.array_equal(Co['s'], Lb['s'])
            and np.array_equal(Co['valid_h90'].astype(int), Lb['valid_h90'].astype(int))):
        raise ValueError(f'{name}: cohort / label row contract failed')
    return Lb, Co


def chunk_files():
    return sorted(STATES_DIR.glob('chunk_*.npz'))


def state_access(base, purpose, sealed):
    log_access(base, purpose, 'extract/MAIN/states', 'parent extraction chunks (pre and post states of every set)', sealed)


# ------------------------------------------------------------------ status / logging (this root only)
class Status(T.Status):
    def __init__(self, group, out=OUT):
        Q.Status.__init__(self, group, out=out)

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
    with open(OUT / 'commands.txt', 'a', encoding='utf-8') as f:
        f.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} pid={os.getpid()} {" ".join([PYTHON, "-B"] + list(argv if argv is not None else sys.argv))} {note}'.rstrip() + '\n')


def log_failure(stage, error, **info):
    Q.append_jsonl(OUT / 'failures.jsonl', dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(), stage=stage, error=str(error)[:4000], **info))


def environment():
    return Q.environment()


def own_source_hashes():
    files = sorted((ROOT / 'scripts').glob('vd20260916_*.py')) + sorted((ROOT / 'tests').glob('test_vd20260916_*.py'))
    reused = [ROOT / 'scripts' / n for n in ('fc20260915_common.py', 'cr20260915_common.py', 'iq20260915_common.py', 'ta20260916_common.py')] + [SPEC, *PROVENANCE_DOCS]
    return {p.relative_to(ROOT).as_posix(): C.sha256_file(p) for p in files + reused if p.exists()}
