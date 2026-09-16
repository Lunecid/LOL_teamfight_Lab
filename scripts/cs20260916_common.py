"""Composite SHAP + Rashomon reliance 2026-09-16: shared helpers (pure; importing sets only environment defaults).

Design: docs/CLAUDE_COMPOSITE_SHAP_20260916.md. Everything written by this run lands in outputs/composite_shap_20260916.
Parents are READ ONLY. Reused pure code: fc20260915_shap.group_exact_shapley, bsh20260916_common.select_rows/rank_key,
iq20260915_common / ta20260916_common model classes (bundles are joblib-serialised).
"""
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
import bsh20260916_common as B  # noqa: E402

ROOT = C.ROOT
OUT = ROOT / 'outputs' / 'composite_shap_20260916'
SMOKE = OUT / 'smoke_train_only'
FC = C.OUT
CR = K.OUT
IQ = Q.OUT
TA = T.OUT
CC = ROOT / 'outputs' / 'champion_class_20260916'
BSH = B.OUT
SPEC = ROOT / 'docs' / 'CLAUDE_COMPOSITE_SHAP_20260916.md'
PROVENANCE_DOCS = (B.SPEC, Q.SPEC, T.SPEC)
PYTHON = Q.PYTHON
VERSION = 'composite_shap_20260916'
ROLE_TAG = Q.ROLE_TAG
H = 90
COHORTS = Q.COHORTS
COHORT_CODE = Q.COHORT_CODE
EXPECTED_COUNTS = Q.EXPECTED_COUNTS
V_ADAPTER = FC / 'models' / 'v' / 'v_final_raw.joblib'

# Part A
MODELS = B.MODELS                   # ('final_q', 'plain_mlp')
CELLS = B.CELLS                     # ('all', 'B40')
SHAP_COHORTS = ('T',)
N_EXPLAIN, N_BG = B.N_EXPLAIN, B.N_BG
SMOKE_EXPLAIN, SMOKE_BG = B.SMOKE_EXPLAIN, B.SMOKE_BG
BASE_GROUPS = ('time', 'economy_and_experience', 'combat_and_survival', 'objectives', 'structures', 'health_mana_other', 'champion_identity')
INPUT_GROUPS = B.GROUPS             # 7 groups of the q input space (prior_win_probability first)
X_TIME_DIVISOR = 30.0
BOOT_REPS = Q.BOOT_REPS
BOOT_SEED = Q.BOOT_SEED

# Part B
EPSILONS = (0.0005, 0.001)
PRIMARY_EPS = 0.0005
PERM_SEEDS = (11, 22, 33, 44, 55)
CALS = K.CALS
CANDIDATE_SOURCES = (('iq', 'logit'), ('iq', 'lgbm'), ('ta', 'mlp'), ('ta', 'resmlp'), ('cc', 'logit'), ('cc', 'lgbm'))

weights = Q.weights
select_rows = B.select_rows
rank_key = B.rank_key


def ts(s):
    return time.mktime(time.strptime(s, '%Y-%m-%d %H:%M:%S'))


# ------------------------------------------------------------------ base state, derived columns, composite function
class StateMap:
    """Column bookkeeping between the 362-name adapter state, the 266 base columns and the 352 q ridge inputs."""

    def __init__(self, adapter, input_names_all, ridge):
        self.state_names = list(adapter.state_names)
        self.input_names = list(input_names_all)
        self.ridge = list(ridge)
        if self.input_names[-1] != C.P_PRE or self.input_names[:-1] != [n for n in self.state_names if n != 'snapshot_age_s']:
            raise ValueError('q input names are not the adapter state minus snapshot_age_s plus p_pre_V')
        self.derived = {}
        for j, n in enumerate(self.state_names):
            if n.endswith('_x_time'):
                self.derived[j] = ('x_time', self.state_names.index(n[:-len('_x_time')]))
            elif n == 'time_minutes_sq':
                self.derived[j] = ('sq', self.state_names.index('time_minutes'))
        self.time_j = self.state_names.index('time_minutes')
        self.age_j = self.state_names.index('snapshot_age_s')
        self.base_state_cols = [j for j, n in enumerate(self.state_names) if j not in self.derived and n != 'snapshot_age_s']
        self.base_names = [self.state_names[j] for j in self.base_state_cols]
        self.q_state_cols = [self.state_names.index(n) for n in self.ridge if n != C.P_PRE]
        self.q_ppre_pos = self.ridge.index(C.P_PRE)
        self.input_pos = {n: i for i, n in enumerate(self.input_names)}
        if len(self.base_names) != 266 or len(self.derived) != 95:
            raise ValueError(f'unexpected base/derived counts {len(self.base_names)}/{len(self.derived)}')

    def base_groups(self):
        """[[base column indices] per BASE_GROUPS] (partition of the 266 base columns)."""
        g = {k: [] for k in BASE_GROUPS}
        for i, n in enumerate(self.base_names):
            if n == 'time_minutes':
                g['time'].append(i)
            elif n.endswith('champion_id'):
                g['champion_identity'].append(i)
            else:
                s = C.shap_group(n)
                if s not in g or s in ('time', 'champion_identity'):
                    raise ValueError(f'unexpected group {s} for base column {n}')
                g[s].append(i)
        idx = [g[k] for k in BASE_GROUPS]
        if sorted(sum(idx, [])) != list(range(len(self.base_names))):
            raise ValueError('base groups do not partition the base columns')
        return idx

    def base_from_inputs(self, Xin):
        """(n, 266) base matrix from the q input matrix (n, 362 incl. p_pre_V) — snapshot_age_s is absent from the inputs."""
        Xin = np.asarray(Xin, dtype=np.float64)
        return np.ascontiguousarray(Xin[:, [self.input_pos[n] for n in self.base_names]])

    def state_from_base(self, Zb):
        """(n, 362) adapter-order state from the base matrix: derived columns recomputed, snapshot_age_s = 0."""
        Zb = np.asarray(Zb, dtype=np.float64)
        S = np.zeros((len(Zb), len(self.state_names)), dtype=np.float64)
        S[:, self.base_state_cols] = Zb
        t = S[:, self.time_j]
        for j, (kind, src) in self.derived.items():
            S[:, j] = S[:, src] * t / X_TIME_DIVISOR if kind == 'x_time' else t * t
        return S

    def q_inputs(self, S, p_pre):
        Xq = np.empty((len(S), len(self.ridge)), dtype=np.float64)
        k = 0
        for i, n in enumerate(self.ridge):
            if n == C.P_PRE:
                Xq[:, i] = p_pre
            else:
                Xq[:, i] = S[:, self.q_state_cols[k]]
                k += 1
        return Xq


def composite_fn(adapter, smap, bundles, chunk=65536):
    """Zb (m, 266) -> {model: calibrated q prediction (m,)} through V and the frozen bundles (both models share one V pass)."""
    def f(Zb):
        Zb = np.asarray(Zb, dtype=np.float64)
        out = {m: np.empty(len(Zb)) for m in bundles}
        for s0 in range(0, len(Zb), chunk):
            S = smap.state_from_base(Zb[s0:s0 + chunk])
            p = adapter.predict_matrix(S, smap.state_names, adapter.state_version)
            Xq = smap.q_inputs(S, p)
            for m, (b, cal) in bundles.items():
                out[m][s0:s0 + chunk] = Q.calibrate(b, cal, b['base'].raw(Xq))
        return out
    return f


def exact_group_shapley_multi(f_multi, models, xrow, Bg, groups):
    """Exact Shapley over 2^G coalitions for several models sharing the same masked inputs (mirror of fc20260915_shap.group_exact_shapley)."""
    import math
    G = len(groups)
    nb, d = Bg.shape
    Kc = 1 << G
    colmask = np.zeros((Kc, d), dtype=bool)
    for k, idx in enumerate(groups):
        on = ((np.arange(Kc) >> k) & 1).astype(bool)
        colmask[np.ix_(on, idx)] = True
    Z = np.where(colmask[:, None, :], xrow[None, None, :], Bg[None, :, :]).reshape(Kc * nb, d)
    vals = f_multi(Z)
    fact = [math.factorial(i) for i in range(G + 1)]
    res = {}
    for m in models:
        v = vals[m].reshape(Kc, nb).mean(axis=1)
        phi = np.zeros(G)
        for S in range(Kc):
            size = bin(S).count('1')
            for k in range(G):
                if not (S >> k) & 1:
                    phi[k] += fact[size] * fact[G - size - 1] / fact[G] * (v[S | (1 << k)] - v[S])
        res[m] = (phi, float(v[0]), float(v[Kc - 1]))
    return res


# ------------------------------------------------------------------ parent bundles and predictions
def parent_winner(model, coh):
    return B.parent_winner(model, coh)


def candidate_bundles(coh):
    """[(source, family, config, bundle path)] of every saved same-input candidate of one cohort."""
    out = []
    for src, fam in CANDIDATE_SOURCES:
        if src == 'iq':
            paths = sorted((IQ / 'models' / coh / fam).glob('*.joblib'))
        elif src == 'ta':
            paths = sorted((TA / 'models' / coh / fam).glob('*.joblib'))
        else:
            paths = sorted((CC / 'models' / coh / fam / 'base').glob('*.joblib'))
        for p in paths:
            out.append((src, fam, p.stem, p))
    return out


def candidate_predictions(src, fam, coh):
    """Saved trainval prediction file of a candidate family (columns '<config>__<cal>' and 'role')."""
    if src == 'iq':
        return IQ / 'predictions' / f'{fam}_{coh}_trainval.npz'
    if src == 'ta':
        return TA / 'predictions' / f'{fam}_{coh}_trainval.npz'
    return CC / 'predictions' / f'{fam}_base_{coh}_trainval.npz'


# ------------------------------------------------------------------ parent data (read-only) with access logging in THIS root
def frozen_path(base=OUT):
    return Path(base) / 'frozen_manifest.json'


def log_access(base, purpose, set_name, kind, sealed):
    fz = frozen_path(OUT)
    if sealed and not fz.exists():
        raise PermissionError(f'{purpose}: {set_name} is sealed until {fz} exists')
    Q.append_jsonl(Path(base) / 'access_log.jsonl', dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(), purpose=purpose, set=set_name, kind=kind,
                                                         sealed=bool(sealed), frozen_manifest_exists=fz.exists(), frozen_manifest_sha256=C.sha256_file(fz) if fz.exists() else None))


def load_parent_set(name, base, purpose):
    sealed = name in Q.SEALED_SETS
    if not (name.startswith('MAIN_TRAIN') or name == 'MAIN_VALIDATION' or sealed):
        raise ValueError(name)
    log_access(base, purpose, name, 'parent features/labels/cohorts npz (h90)', sealed)
    with np.load(FC / 'labels' / f'{name}_features_pre_only.npz', allow_pickle=False) as z:
        F = {k: z[k] for k in ('X_input', 'input_names', 'match', 's_ms', 'sub_role', 'pre_ok')}
    with np.load(FC / 'labels' / f'{name}_labels.npz', allow_pickle=False) as z:
        Lb = {k: z[k] for k in Q.LABEL_KEYS if k in z.files}
    with np.load(CR / 'cohorts' / f'{name}_cohort.npz', allow_pickle=False) as z:
        Co = {k: z[k] for k in ('match', 's', 'cohort', 'valid_h90', 'scale_known')}
    ok = (np.array_equal(F['match'], Lb['match']) and np.array_equal(F['s_ms'], Lb['s']) and np.array_equal(Co['match'].astype(str), Lb['match'].astype(str))
          and np.array_equal(Co['s'], Lb['s']) and np.array_equal(Co['valid_h90'].astype(int), Lb['valid_h90'].astype(int)))
    okp = F['pre_ok'] == 1
    ok = ok and np.array_equal(F['X_input'][okp, -1], Lb['p_pre'][okp])
    if not ok:
        raise ValueError(f'{name}: parent row contract failed')
    return F, Lb, Co


def cohort_rows(F, Lb, Co, coh, sub_roles=None):
    m = (Lb['valid_h90'] == 1) & (Co['cohort'] == COHORT_CODE[coh]) & (F['pre_ok'] == 1)
    if sub_roles is not None:
        m &= np.isin(F['sub_role'].astype(str), list(sub_roles))
    return np.flatnonzero(m)


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
    return T.environment()


def own_source_hashes():
    files = sorted((ROOT / 'scripts').glob('cs20260916_*.py')) + sorted((ROOT / 'tests').glob('test_cs20260916_*.py'))
    reused = [ROOT / 'scripts' / n for n in ('fc20260915_common.py', 'cr20260915_common.py', 'iq20260915_common.py', 'ta20260916_common.py', 'bsh20260916_common.py', 'fc20260915_shap.py')] + [SPEC, *PROVENANCE_DOCS]
    return {p.relative_to(ROOT).as_posix(): C.sha256_file(p) for p in files + reused if p.exists()}
