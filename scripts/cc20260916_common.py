"""Champion-class features 2026-09-16: shared helpers (pure; importing has no side effects beyond environment defaults).

Design: docs/CLAUDE_CHAMPION_CLASS_20260916.md. Everything written by this run lands in outputs/champion_class_20260916.
Parents are READ ONLY; parent writers are never called. Reused pure classes: iq20260915_common.LinearQBase / LgbmQBase.
Bundles are joblib-serialised, so scripts/ must be on sys.path and iq20260915_common importable when bundles are loaded.
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

ROOT = C.ROOT
OUT = ROOT / 'outputs' / 'champion_class_20260916'
SMOKE = OUT / 'smoke_train_only'
DDRAGON = OUT / 'ddragon'
TAG_TABLE = DDRAGON / 'tag_table.json'
FC = C.OUT
CR = K.OUT
IQ = Q.OUT
SPEC = ROOT / 'docs' / 'CLAUDE_CHAMPION_CLASS_20260916.md'
PROVENANCE_DOCS = (Q.SPEC, *Q.PROVENANCE_DOCS)
PYTHON = Q.PYTHON
VERSION = 'champion_class_20260916'
ROLE_TAG = Q.ROLE_TAG

H = Q.H
COHORTS = Q.COHORTS
COHORT_CODE = Q.COHORT_CODE
EXT = Q.EXT
EVAL_SETS = Q.EVAL_SETS
SEALED_SETS = Q.SEALED_SETS
EXPECTED_COUNTS = Q.EXPECTED_COUNTS
EXPECTED_RIDGE_COUNT = Q.EXPECTED_RIDGE_COUNT
FAMILIES = ('logit', 'lgbm')
CALS = K.CALS
SEEDED = ('lgbm',)
SEEDS = {'lgbm': Q.LGBM_SEEDS}
BOOT_REPS = Q.BOOT_REPS
BOOT_SEED = Q.BOOT_SEED
SPARSE_MATCHES = Q.SPARSE_MATCHES

TAGS = ('Assassin', 'Fighter', 'Mage', 'Marksman', 'Support', 'Tank')
POSITIONS = ('TOP', 'JG', 'MID', 'BOT', 'SUP')  # parent slot order 0-4 blue, 5-9 red (class mix per slot in the spec)
TEAMS = (('blue', (0, 1, 2, 3, 4)), ('red', (5, 6, 7, 8, 9)))
SLOT_STATS = ('totalGold_norm', 'level_norm', 'kills', 'deaths', 'alive', 'hp_pct')
AGG_NAMES = ('totalGold_norm_sum', 'level_norm_sum', 'kills_sum', 'deaths_sum', 'alive_sum', 'hp_pct_mean')
SAME_TEAM_PAIRS = (('TOP', 'JG'), ('JG', 'MID'), ('BOT', 'SUP'))
BLOCKS = ('tags', 'class_count', 'class_agg', 'pairs', 'identity')
BLOCK_WIDTH = {'tags': 120, 'class_count': 18, 'class_agg': 108, 'pairs': 396}  # identity: 10 x len(vocab)
ARM_BLOCKS = {'base': (), 'tags': ('tags',), 'class_state': ('tags', 'class_count', 'class_agg'),
              'class_pairs': ('tags', 'class_count', 'class_agg', 'pairs'), 'identity': ('identity',),
              'class_pairs_identity': ('tags', 'class_count', 'class_agg', 'pairs', 'identity'),
              'draft_class': ('tags', 'class_count', 'pairs'), 'draft_identity': ('identity',)}
ARMS = tuple(ARM_BLOCKS)
DRAFT_ARMS = ('draft_class', 'draft_identity')  # no state inputs (ridge excluded)
SET_VERSION = {'MAIN_TRAIN': '15.14.1', 'MAIN_VALIDATION': '15.15.1', 'MAIN_TEST': '15.16.1', 'EXT_KR_16.13': '16.13.1',
               'EXT_KR_16.14_pilot': '16.14.1', 'EXT_KR_16.15': '16.15.1', 'EXT_NA1_16.13': '16.13.1'}
NAMED = tuple(f'{f}_{a}' for f in FAMILIES for a in ARMS)
PARENT_NAMED = ('parent_lgbm_winner', 'parent_logit_winner')
BOOT_MODELS = NAMED


def _contrasts():
    out = []
    for f in FAMILIES:
        for a in ARMS:
            if a == 'base':
                continue
            lab = 'PRIMARY' if (f == 'lgbm' and a == 'class_pairs') else 'planned'
            out.append((f'{f}_{a}', f'{f}_base', f'{lab}: {f} {a} - base'))
        out.append((f'{f}_class_pairs', f'{f}_identity', f'planned: {f} class_pairs - identity'))
        out.append((f'{f}_class_pairs_identity', f'{f}_identity', f'planned: {f} class_pairs_identity - identity'))
        out.append((f'{f}_class_pairs_identity', f'{f}_class_pairs', f'planned: {f} class_pairs_identity - class_pairs'))
    return tuple(out)


CONTRASTS = _contrasts()
PRIMARY = ('lgbm_class_pairs', 'lgbm_base', 'T', 'all', 'brier')

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
calibrator_record = Q.calibrator_record


def bundle_raw(b, X_all, names_all=None):
    """Raw prediction of a bundle on its arm matrix. The base always receives a C-contiguous float64 array: column fancy
    indexing (X[:, cols]) returns an F-ordered array whose BLAS path differs in the last bits (3e-16 seen in the smoke),
    which would break the bitwise reload-identity gates."""
    if names_all is not None and C.sha256_json(list(names_all)) != b['input_names_all_sha256']:
        raise ValueError('input schema differs from the bundle schema')
    X = np.asarray(X_all, dtype=np.float64)
    cols = list(b['input_columns'])
    Xin = X if cols == list(range(X.shape[1])) else X[:, cols]
    return b['base'].raw(np.ascontiguousarray(Xin))


def bundle_predict(b, cal, X_all, names_all=None):
    return calibrate(b, cal, bundle_raw(b, X_all, names_all))
config_params = Q.config_params


def ts(s):
    return time.mktime(time.strptime(s, '%Y-%m-%d %H:%M:%S'))


# ------------------------------------------------------------------ fixed configurations (h90 winners of iq)
def registry():
    """{cohort: {family: config}} read from the iq frozen manifest (h90 family winners)."""
    iq_fz = C.read_json(Q.frozen_path(IQ))
    reg = {coh: {f: iq_fz['family_winners'][f'{f}_{coh}']['config'] for f in FAMILIES} for coh in COHORTS}
    return reg, dict(iq_frozen_manifest_sha256=C.sha256_file(Q.frozen_path(IQ)))


def candidate_names(config):
    """3 joint candidates '<config>__<calibration>' in lexical tie order (calibration-only selection)."""
    return sorted(f'{config}__{cal}' for cal in CALS)


# ------------------------------------------------------------------ champion classes
def load_tag_table():
    return C.read_json(TAG_TABLE)


def tag_lookup(table, version):
    """{champion_id: (primary_index, secondary_index or -1)} for one Data Dragon version."""
    rows = table['table'][version]
    return {int(k): (TAGS.index(v['primary']), TAGS.index(v['secondary']) if v['secondary'] else -1) for k, v in rows.items()}


def slot_columns(names):
    names = list(names)
    ch = [names.index(f'participant_slot{i}_champion_id') for i in range(10)]
    st = {s: [names.index(f'participant_slot{i}_{s}') for i in range(10)] for s in SLOT_STATS}
    return ch, st


def champion_ids(X, names):
    ch, _ = slot_columns(names)
    return np.rint(np.asarray(X)[:, ch]).astype(np.int64)


def train_identity_vocab(X, names):
    ids = champion_ids(X, names)
    return sorted(int(v) for v in np.unique(ids) if v > 0)


def class_arrays(champ, lookup):
    """champ (n, 10) int -> (primary, secondary) (n, 10) with tag indices 0..5 or -1."""
    champ = np.asarray(champ, dtype=np.int64)
    maxid = int(max([max(lookup) if lookup else 0, int(champ.max()) if champ.size else 0])) + 1
    pa = np.full(maxid, -1, dtype=np.int64)
    sa = np.full(maxid, -1, dtype=np.int64)
    for i, (p, s) in lookup.items():
        if 0 <= i < maxid:
            pa[i], sa[i] = p, s
    safe = np.clip(champ, 0, maxid - 1)
    prim = np.where(champ > 0, pa[safe], -1)
    sec = np.where(champ > 0, sa[safe], -1)
    return prim, sec


def extra_blocks(X, names, version, vocab, table=None, blocks=BLOCKS):
    """{block: (names, float64 matrix)} for the requested blocks (definitions in the specification)."""
    table = load_tag_table() if table is None else table
    lk = tag_lookup(table, version)
    names = list(names)
    ch_ix, st_ix = slot_columns(names)
    X = np.asarray(X)
    n = len(X)
    champ = np.rint(X[:, ch_ix]).astype(np.int64)
    prim, sec = class_arrays(champ, lk)
    out = {}
    if 'tags' in blocks:
        cols, nm = [], []
        for i in range(10):
            for kind, arr in (('primary', prim), ('secondary', sec)):
                for t, tag in enumerate(TAGS):
                    cols.append((arr[:, i] == t).astype(np.float64))
                    nm.append(f'cc_slot{i}_{kind}_{tag}')
        out['tags'] = (nm, np.column_stack(cols) if n else np.zeros((0, len(nm))))
    if 'class_count' in blocks or 'class_agg' in blocks:
        cnt, agg = {}, {}
        for team, slots in TEAMS:
            P = prim[:, list(slots)]
            for t, tag in enumerate(TAGS):
                m = (P == t).astype(np.float64)
                c = m.sum(axis=1)
                cnt[(team, tag)] = c
                for s in SLOT_STATS:
                    S = X[:, [st_ix[s][j] for j in slots]].astype(np.float64)
                    tot = (m * S).sum(axis=1)
                    agg[(team, tag, s)] = np.where(c > 0, tot / np.maximum(c, 1.0), 0.0) if s == 'hp_pct' else tot
        if 'class_count' in blocks:
            nm, cols = [], []
            for team, _ in TEAMS:
                for tag in TAGS:
                    nm.append(f'cc_{team}_count_{tag}')
                    cols.append(cnt[(team, tag)])
            for tag in TAGS:
                nm.append(f'cc_diff_count_{tag}')
                cols.append(cnt[('blue', tag)] - cnt[('red', tag)])
            out['class_count'] = (nm, np.column_stack(cols) if n else np.zeros((0, len(nm))))
        if 'class_agg' in blocks:
            nm, cols = [], []
            for team, _ in TEAMS:
                for tag in TAGS:
                    for s, an in zip(SLOT_STATS, AGG_NAMES):
                        nm.append(f'cc_{team}_{tag}_{an}')
                        cols.append(agg[(team, tag, s)])
            for tag in TAGS:
                for s, an in zip(SLOT_STATS, AGG_NAMES):
                    nm.append(f'cc_diff_{tag}_{an}')
                    cols.append(agg[('blue', tag, s)] - agg[('red', tag, s)])
            out['class_agg'] = (nm, np.column_stack(cols) if n else np.zeros((0, len(nm))))
    if 'pairs' in blocks:
        nm, cols = [], []

        def onehot_pair(a, b, prefix):
            for ta, tag_a in enumerate(TAGS):
                for tb, tag_b in enumerate(TAGS):
                    cols.append(((a == ta) & (b == tb)).astype(np.float64))
                    nm.append(f'{prefix}_{tag_a}x{tag_b}')
        for team, slots in TEAMS:
            for pa_, pb_ in SAME_TEAM_PAIRS:
                onehot_pair(prim[:, slots[POSITIONS.index(pa_)]], prim[:, slots[POSITIONS.index(pb_)]], f'cc_{team}_pair_{pa_}_{pb_}')
        for k, pos in enumerate(POSITIONS):
            onehot_pair(prim[:, k], prim[:, 5 + k], f'cc_matchup_{pos}')
        out['pairs'] = (nm, np.column_stack(cols) if n else np.zeros((0, len(nm))))
    if 'identity' in blocks:
        vocab = np.asarray(sorted(int(v) for v in vocab), dtype=np.int64)
        nv = len(vocab)
        M = np.zeros((n, 10 * nv), dtype=np.float64)
        nm = [f'cc_slot{i}_champ_{int(v)}' for i in range(10) for v in vocab]
        for i in range(10):
            c = champ[:, i]
            j = np.searchsorted(vocab, c)
            jj = np.minimum(j, max(nv - 1, 0))
            ok = (c > 0) & (vocab[jj] == c) if nv else np.zeros(n, dtype=bool)
            r = np.nonzero(ok)[0]
            M[r, i * nv + jj[r]] = 1.0
        out['identity'] = (nm, M)
    for b, (bn, bm) in out.items():
        if bn != block_names(b, vocab) or bm.shape[1] != len(bn):
            raise ValueError(f'block {b}: names differ from block_names()')
    return out


def block_names(block, vocab=()):
    """Column names of one block in the order extra_blocks() produces them."""
    if block == 'tags':
        return [f'cc_slot{i}_{kind}_{tag}' for i in range(10) for kind in ('primary', 'secondary') for tag in TAGS]
    if block == 'class_count':
        return [f'cc_{team}_count_{tag}' for team, _ in TEAMS for tag in TAGS] + [f'cc_diff_count_{tag}' for tag in TAGS]
    if block == 'class_agg':
        return ([f'cc_{team}_{tag}_{an}' for team, _ in TEAMS for tag in TAGS for an in AGG_NAMES]
                + [f'cc_diff_{tag}_{an}' for tag in TAGS for an in AGG_NAMES])
    if block == 'pairs':
        nm = []
        for team, _ in TEAMS:
            for a, b in SAME_TEAM_PAIRS:
                nm += [f'cc_{team}_pair_{a}_{b}_{ta}x{tb}' for ta in TAGS for tb in TAGS]
        for pos in POSITIONS:
            nm += [f'cc_matchup_{pos}_{ta}x{tb}' for ta in TAGS for tb in TAGS]
        return nm
    if block == 'identity':
        return [f'cc_slot{i}_champ_{int(v)}' for i in range(10) for v in sorted(int(x) for x in vocab)]
    raise ValueError(block)


def arm_names(arm, ridge, vocab):
    nm = [] if arm in DRAFT_ARMS else list(ridge)
    for b in ARM_BLOCKS[arm]:
        nm += block_names(b, vocab)
    return nm


def arm_width(arm, n_vocab, n_ridge=EXPECTED_RIDGE_COUNT):
    w = 0 if arm in DRAFT_ARMS else int(n_ridge)
    return w + sum(10 * n_vocab if b == 'identity' else BLOCK_WIDTH[b] for b in ARM_BLOCKS[arm])


def identity_vocab_from_train(base, purpose):
    """Sorted champion ids (> 0) of the MAIN_TRAIN h90-valid cohort rows (both cohorts): the identity one-hot vocabulary."""
    F, Lb, Co, _ = load_parent_set('MAIN_TRAIN', base, purpose)
    keep = (Lb['valid_h90'] == 1) & (Co['cohort'] >= 0) & (F['pre_ok'] == 1)
    return train_identity_vocab(F['X_input'][keep], [str(x) for x in F['input_names']])


def arm_matrix(X, names, arm, ridge, versions, vocab, table=None):
    """(matrix float64, input names) of one arm; rows may come from several patch tables (versions per row)."""
    names = list(names)
    X = np.asarray(X)
    n = len(X)
    versions = np.asarray(versions).astype(str)
    if len(versions) != n:
        raise ValueError('versions length differs from rows')
    table = load_tag_table() if table is None else table
    blocks = ARM_BLOCKS[arm]
    nm = [] if arm in DRAFT_ARMS else list(ridge)
    width = arm_width(arm, len(vocab), len(ridge))
    out = np.empty((n, width), dtype=np.float64)
    col = 0
    if arm not in DRAFT_ARMS:
        ix = [names.index(r) for r in ridge]
        out[:, :len(ix)] = X[:, ix]
        col = len(ix)
    if blocks:
        first = True
        for v in np.unique(versions):
            m = versions == v
            ex = extra_blocks(X[m], names, str(v), vocab, table, blocks=blocks)
            c = col
            for b in blocks:
                bn, bm = ex[b]
                out[m, c:c + bm.shape[1]] = bm
                c += bm.shape[1]
                if first:
                    nm += bn
            first = False
            if c != width:
                raise ValueError(f'arm width mismatch {c} vs {width}')
    if len(nm) != width:
        raise ValueError('name count differs from width')
    return out, nm


def matrix_sha(M):
    return C.sha256_bytes(np.ascontiguousarray(np.asarray(M, dtype=np.float64)).tobytes())


# ------------------------------------------------------------------ families and bundles
def fit_family(family, config, input_names, Xtr, y, g, w, log=None):
    """Dispatch to the reused pure classes. Returns (base, eligible, fit_record, stop_record_or_None)."""
    if family == 'logit':
        base = Q.LinearQBase('logit', config, input_names).fit(Xtr, y, w)
        rec = dict(attempts=base.attempts, eligible_converged=base.eligible, design=base.design_record, params=dict(C=base.C_value))
        return base, bool(base.eligible), rec, None
    if family == 'lgbm':
        base = Q.LgbmQBase(config, input_names).fit(Xtr, y, g, log=log)
        rec = dict(params=base.params, seeds=list(base.seeds),
                   seeds_summary={s: {k: v for k, v in r.items() if k != 'stop_curve'} for s, r in base.stop_record['seeds'].items()},
                   imputer_full_n_features=int(base.imputer.n_features_in_))
        return base, True, rec, base.stop_record
    raise ValueError(family)


def make_bundle(family, config, arm, base, calibrators, names_arm, manifest):
    return dict(version=VERSION, family=family, config=config, arm=arm, blocks=list(ARM_BLOCKS[arm]), params=config_params(family, config),
                input_names_all_sha256=C.sha256_json(list(names_arm)), input_names=list(names_arm), input_columns=list(range(len(names_arm))),
                base=base, calibrators=calibrators, manifest=manifest)


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


def parent_pred_path(set_name, coh):
    return IQ / 'eval' / 'predictions' / f'{set_name}_h{H}_{coh}.npz'


def load_parent_set(name, base, purpose):
    """Features (pre-only X_input), h90 labels and cohort arrays of one parent set with exact row-order checks."""
    if name.startswith('MAIN_TRAIN') or name == 'MAIN_VALIDATION':
        sealed = False
    elif name in SEALED_SETS:
        sealed = True
    else:
        raise ValueError(name)
    log_access(base, purpose, name, 'parent features/labels/cohorts npz (h90)', sealed)
    with np.load(FC / 'labels' / f'{name}_features_pre_only.npz', allow_pickle=False) as z:
        F = {k: z[k] for k in ('X_input', 'input_names', 'match', 's_ms', 'sub_role', 'pre_ok')}
    with np.load(FC / 'labels' / f'{name}_labels.npz', allow_pickle=False) as z:
        Lb = {k: z[k] for k in Q.LABEL_KEYS if k in z.files}
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


def load_trainval(base, smoke=False, cohort=None):
    """TRAIN (+ VALIDATION Q_CAL/Q_SELECT) h90-valid rows of one cohort with role labels and the patch version per row.
    smoke: TRAIN only, pseudo roles (fold0-2 TRAIN, fold3 Q_CAL, fold4 Q_SELECT), 1/8 of matches."""
    parts = ['MAIN_TRAIN'] if smoke else ['MAIN_TRAIN', 'MAIN_VALIDATION']
    D = dict(parts=parts, smoke=bool(smoke), checks={})
    Xs, gs, ss, rs, ys, ps, cs, aids, ashs, vs = [], [], [], [], [], [], [], [], [], []
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
        vs.append(np.full(int(keep.sum()), SET_VERSION[part]))
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
    D['version'] = np.concatenate(vs)
    if smoke:
        f = np.asarray([C.train_fold(m) for m in D['g'].tolist()])
        D['role'] = np.where(f <= 2, 'TRAIN', np.where(f == 3, 'Q_CAL', 'Q_SELECT'))
        keep = smoke_keep(D['g'])
        for k in ('X', 'g', 's_ms', 'sr_true', 'y', 'p_pre', 'cohort', 'adapter_id', 'adapter_sha256', 'version', 'role'):
            D[k] = D[k][keep]
    else:
        D['role'] = np.where(np.char.startswith(D['sr_true'], 'fold'), 'TRAIN', D['sr_true'])
    return D


# ------------------------------------------------------------------ status / logging (this root only)
class Status(Q.Status):
    def __init__(self, group, out=OUT):
        super().__init__(group, out=out)

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
    return Q.environment()


def own_source_hashes():
    files = sorted((ROOT / 'scripts').glob('cc20260916_*.py')) + sorted((ROOT / 'tests').glob('test_cc20260916_*.py'))
    reused = [ROOT / 'scripts' / n for n in ('fc20260915_common.py', 'cr20260915_common.py', 'iq20260915_common.py')] + [SPEC, *PROVENANCE_DOCS]
    return {p.relative_to(ROOT).as_posix(): C.sha256_file(p) for p in files + reused if p.exists()}
