"""Cohort (teamfight / non-teamfight) q specialists and role-aware q representation, 2026-09-15: shared helpers.

Design: docs/CLAUDE_COHORT_ROLE_TRAIN_20260915.md (Codex). Implementation/execution: Claude Opus 5.
Everything written lands in outputs/cohort_role_training_20260915. The completed full-corpus run
(outputs/full_corpus_training_20260915) is read-only: pre states, keys, labels, V and the pooled q come from there.

Pure module (importing has no side effects). Classes serialized with joblib (DraftEncoder, RoleModel, role arm
bundles) require scripts/ on sys.path and this module importable as cr20260915_common.
"""
from __future__ import annotations

from itertools import permutations
import json
import os
from pathlib import Path
import time

import numpy as np

import fc20260915_common as C

ROOT = C.ROOT
OUT = ROOT / 'outputs' / 'cohort_role_training_20260915'
FC = C.OUT                      # prior full-corpus run (read-only)
SPEC = ROOT / 'docs' / 'CLAUDE_COHORT_ROLE_TRAIN_20260915.md'
SHARDS = Path('D:/LOL_Project/fusion_2615/corpus_shards_v33')
REPO = C.REPO
WT = C.WT
VERSION = 'cohort_role_training_20260915'
ROLE_TAG = 'EXPLORATORY_FOLLOWUP_MODEL_DEFINED_LABELS_PRIOR_TEST_EXPOSURE'
EXT = ('KR_16.13', 'KR_16.14_pilot', 'KR_16.15', 'NA1_16.13')
HS = C.HORIZONS_S
PRIMARY_H = C.PRIMARY_H
BOOT_REPS = 1000
BOOT_SEED = 20260915

# ------------------------------------------------------------------ scale definition (v3.3 manifest scale block)
TEAMFIGHT_MIN = 4       # manifest scale.teamfight_min
PICK_MAX = 1            # manifest scale.pick_max
SKIRMISH_MIN = 2        # manifest scale.skirmish_min
COHORTS = ('T', 'N')
COHORT_CODE = {'T': 1, 'N': 0, 'UNKNOWN': -1}


def scale_classes(blue, red, negative_is_zero):
    """Stored participation counts -> (n_min, known, cohort code, fine class code).

    negative_is_zero: boolean per row; True only where provenance proved -1 = int(0 or -1).
    cohort: 1 = T (min >= 4), 0 = N (known, min < 4), -1 unknown. fine: 0 pick (<=1), 1 skirmish (2..3), 2 teamfight, -1 unknown.
    """
    b = np.asarray(blue).astype(np.int64)
    r = np.asarray(red).astype(np.int64)
    nz = np.asarray(negative_is_zero, dtype=bool)
    bb = np.where((b < 0) & nz, 0, b)
    rr = np.where((r < 0) & nz, 0, r)
    known = (bb >= 0) & (rr >= 0)
    n_min = np.where(known, np.minimum(bb, rr), -1)
    cohort = np.where(~known, -1, np.where(n_min >= TEAMFIGHT_MIN, 1, 0))
    fine = np.where(~known, -1, np.where(n_min <= PICK_MAX, 0, np.where(n_min < TEAMFIGHT_MIN, 1, 2)))
    return n_min, known, cohort, fine


# ------------------------------------------------------------------ status / io
class Status:
    """Per-stage status file plus aggregated status.json for this output root."""

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
        rec['history'] = self.history[-80:]
        C.write_json(self.out / 'status' / f'{self.group}.json', rec)
        aggregate_status(self.out)
        self.log(f'STATUS {state} {stage} processed={processed} total={total} error={error} next={next_step} '
                 + json.dumps(C.clean(extra), ensure_ascii=False)[:500])


def aggregate_status(out=OUT):
    out = Path(out)
    groups = {}
    for p in sorted((out / 'status').glob('*.json')):
        try:
            groups[p.stem] = json.loads(p.read_text(encoding='utf-8'))
        except Exception as exc:
            groups[p.stem] = {'read_error': repr(exc)}
    running = [g for g, r in groups.items() if r.get('state') == 'running']
    agg = dict(version=VERSION, role=ROLE_TAG, updated=time.strftime('%Y-%m-%d %H:%M:%S'), running_groups=running,
               stage_states={g: r.get('state') for g, r in groups.items()},
               groups={g: {k: r.get(k) for k in ('state', 'stage', 'processed', 'total', 'pid', 'elapsed_s', 'error',
                                                  'next_step', 'updated')} for g, r in groups.items()},
               state_vocabulary='implemented = code written; running; executed = stage finished; verified = post-run checks passed '
                                '(see validation.json); failed')
    try:
        C.write_json(out / 'status.json', agg)
    except PermissionError:
        pass


def log_command(line):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'commands.txt', 'a', encoding='utf-8') as f:
        f.write(time.strftime('%Y-%m-%dT%H:%M:%S') + ' ' + line + '\n')


def outcome_gate(purpose, sets, frozen_required):
    """Append-only access log for generated labels Y of non-TRAIN/VALIDATION sets; refuses before freeze if required."""
    fz = OUT / 'frozen_manifest.json'
    if frozen_required and not fz.exists():
        raise PermissionError(f'{purpose}: TEST/external labels are sealed until {fz.name} exists')
    OUT.mkdir(parents=True, exist_ok=True)
    rec = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(), purpose=purpose, sets=list(sets),
               frozen_manifest_exists=fz.exists(),
               frozen_manifest_sha256=C.sha256_file(fz) if fz.exists() else None)
    with open(OUT / 'label_access_log.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\n')


# ------------------------------------------------------------------ paired match bootstrap (fixed models)
class _FastAUC:
    """Weighted Mann-Whitney AUC for one fixed score vector under changing row weights (ties count 1/2)."""

    def __init__(self, y, p):
        self.order = np.argsort(np.asarray(p, dtype=np.float64), kind='mergesort')
        s = np.asarray(p, dtype=np.float64)[self.order]
        new = np.ones(len(s), dtype=bool)
        new[1:] = s[1:] != s[:-1]
        self.gid = np.cumsum(new) - 1
        self.ng = int(self.gid[-1]) + 1 if len(s) else 0
        self.pos = (np.asarray(y)[self.order] == 1).astype(np.float64)

    def __call__(self, rw):
        r = rw[self.order]
        wp = np.bincount(self.gid, weights=r * self.pos, minlength=self.ng)
        wn = np.bincount(self.gid, weights=r * (1 - self.pos), minlength=self.ng)
        P, N = wp.sum(), wn.sum()
        if P <= 0 or N <= 0:
            return np.nan
        below = np.cumsum(wn) - wn
        return float(np.sum(wp * (below + 0.5 * wn)) / (P * N))


def paired_bootstrap(y, preds, g, pairs, reps=BOOT_REPS, seed=BOOT_SEED):
    """Match bootstrap with fixed models: equal-per-match weights x multiplicity; AUC / Brier / log loss and paired a-b."""
    y = np.asarray(y).astype(float)
    g = np.asarray(g)
    u, inv = np.unique(g, return_inverse=True)
    nm = len(u)
    w = C.weights(g)
    eps = np.finfo(np.float64).eps
    models = sorted({m for pr in pairs for m in pr})
    Wm = np.bincount(inv, weights=w, minlength=nm)
    S, A = {}, {}
    for m in models:
        p = np.asarray(preds[m], dtype=float)
        q = np.clip(p, eps, 1 - eps)
        S[m] = dict(brier=np.bincount(inv, weights=w * (p - y) ** 2, minlength=nm),
                    logloss=np.bincount(inv, weights=w * -(y * np.log(q) + (1 - y) * np.log(1 - q)), minlength=nm))
        A[m] = _FastAUC(y, p)
    point = {m: dict(auc=A[m](w), brier=float(S[m]['brier'].sum() / Wm.sum()), logloss=float(S[m]['logloss'].sum() / Wm.sum()))
             for m in models}
    rng = np.random.default_rng(seed)
    draws = {m: {k: np.full(reps, np.nan) for k in ('auc', 'brier', 'logloss')} for m in models}
    for r in range(reps):
        mult = np.bincount(rng.integers(0, nm, size=nm), minlength=nm).astype(float)
        rw = w * mult[inv]
        den = mult @ Wm
        for m in models:
            draws[m]['brier'][r] = float(mult @ S[m]['brier'] / den)
            draws[m]['logloss'][r] = float(mult @ S[m]['logloss'] / den)
            draws[m]['auc'][r] = A[m](rw)
    out = dict(replicates=reps, seed=seed, unit='match resampled with replacement; weights = equal-per-match x multiplicity',
               fixed_models=True, matches=int(nm), rows=int(len(y)),
               scope='test-sample uncertainty only; training, calibration, selection and role-model variability not included',
               model_ci={m: {k: dict(estimate=point[m][k], ci95=np.nanquantile(draws[m][k], [.025, .975]).tolist()) for k in ('auc', 'brier', 'logloss')}
                         for m in models}, pairs=[])
    for a, b in pairs:
        d = {}
        for k in ('auc', 'brier', 'logloss'):
            diff = draws[a][k] - draws[b][k]
            ok = np.isfinite(diff)
            d[k] = dict(estimate=point[a][k] - point[b][k], ci95=np.quantile(diff[ok], [.025, .975]).tolist() if ok.any() else None,
                        fraction_replicates_a_better=float(np.mean(diff[ok] > 0) if k == 'auc' else np.mean(diff[ok] < 0)) if ok.any() else None)
        out['pairs'].append(dict(a=a, b=b, a_minus_b=d))
    return out


# ------------------------------------------------------------------ role vocabulary and draft encoding
ROLES = ('TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY')
ROLE_INDEX = {r: i for i, r in enumerate(ROLES)}
PERMS = np.asarray(list(permutations(range(5))), dtype=np.int64)          # (120, 5): participant i -> role PERMS[k, i]
PERM_ONEHOT = np.zeros((120, 5, 5))
PERM_ONEHOT[np.arange(120)[:, None], np.arange(5)[None, :], PERMS] = 1.0
PROB_CLIP = 1e-12


class DraftEncoder:
    """Pregame draft fields -> sparse indicators.

    Input rows: (champion_id, spell_a, spell_b) integers, 0 / negative = missing. Champion: OneHotEncoder
    (handle_unknown='ignore') fit on TRAIN champion IDs (missing excluded from the vocabulary, encoded all-zero).
    Spells: unordered multi-hot; one OneHotEncoder(handle_unknown='ignore') is fit on the union of both spell
    columns and the two one-hot rows are OR-ed. Unknown or missing categories are all-zero rows.
    """

    def fit(self, D):
        from sklearn.preprocessing import OneHotEncoder
        D = np.asarray(D, dtype=np.int64)
        champ = D[:, :1]
        spells = D[:, 1:3].reshape(-1, 1)
        self.champion_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64)
        self.champion_encoder.fit(champ[champ[:, 0] > 0])
        self.spell_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64)
        self.spell_encoder.fit(spells[spells[:, 0] > 0])
        self.champion_vocabulary = [int(x) for x in self.champion_encoder.categories_[0]]
        self.spell_vocabulary = [int(x) for x in self.spell_encoder.categories_[0]]
        return self

    def transform(self, D):
        import scipy.sparse as sp
        D = np.asarray(D, dtype=np.int64)
        c = np.where(D[:, :1] > 0, D[:, :1], -999)
        Xc = self.champion_encoder.transform(c)
        s1 = self.spell_encoder.transform(np.where(D[:, 1:2] > 0, D[:, 1:2], -999))
        s2 = self.spell_encoder.transform(np.where(D[:, 2:3] > 0, D[:, 2:3], -999))
        Xs = (s1 + s2)
        Xs.data = np.minimum(Xs.data, 1.0)
        return sp.hstack([Xc, Xs], format='csr')

    def flags(self, D):
        D = np.asarray(D, dtype=np.int64)
        cv = set(self.champion_vocabulary)
        sv = set(self.spell_vocabulary)
        champ_missing = D[:, 0] <= 0
        champ_unseen = (~champ_missing) & np.asarray([int(x) not in cv for x in D[:, 0]])
        spell_missing = (D[:, 1] <= 0) | (D[:, 2] <= 0)
        spell_unseen = np.asarray([(a > 0 and int(a) not in sv) or (b > 0 and int(b) not in sv) for a, b in D[:, 1:3]])
        return dict(champion_missing=champ_missing, champion_unseen=champ_unseen, spell_missing=spell_missing,
                    spell_unseen=spell_unseen)

    @property
    def feature_names(self):
        return [f'champion={c}' for c in self.champion_vocabulary] + [f'spell={s}' for s in self.spell_vocabulary]


class RoleModel:
    """Auxiliary meta-role classifier: DraftEncoder -> multinomial LogisticRegression(C=1, lbfgs).

    Inference reads ONLY (champion_id, spell_a, spell_b) per participant. fit_record documents fitting membership.
    """

    FIELDS = ('champion_id', 'spell_a', 'spell_b')

    def __init__(self, C_value=1.0, max_iter=2000):
        self.C_value = float(C_value)
        self.max_iter = int(max_iter)
        self.encoder = None
        self.clf = None
        self.fit_record = {}

    def fit(self, D, y, w, fit_record):
        from sklearn.linear_model import LogisticRegression
        import warnings
        self.encoder = DraftEncoder().fit(D)
        X = self.encoder.transform(D)
        self.clf = LogisticRegression(C=self.C_value, solver='lbfgs', max_iter=self.max_iter)
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            self.clf.fit(X, np.asarray(y, dtype=np.int64), sample_weight=np.asarray(w, dtype=np.float64))
        if [int(c) for c in self.clf.classes_] != list(range(5)):
            raise ValueError('role classes must be 0..4 (all five roles present)')
        self.fit_record = dict(fit_record, n_iter=int(np.max(self.clf.n_iter_)), max_iter=self.max_iter,
                               converged=bool(np.max(self.clf.n_iter_) < self.max_iter), C=self.C_value,
                               warnings=sorted(set(f'{x.category.__name__}: {str(x.message)[:160]}' for x in wl)),
                               n_rows=int(len(y)), weight_sum=float(np.sum(w)),
                               champion_vocabulary_size=len(self.encoder.champion_vocabulary),
                               spell_vocabulary_size=len(self.encoder.spell_vocabulary),
                               multi_class='multinomial (sklearn lbfgs default for >2 classes)')
        return self

    def predict_proba(self, D):
        D = np.asarray(D, dtype=np.int64)
        if D.ndim != 2 or D.shape[1] != 3:
            raise ValueError('draft matrix must be (n, 3): champion_id, spell_a, spell_b')
        return self.clf.predict_proba(self.encoder.transform(D))


def team_role_posterior(P):
    """P: (n_teams, 5, 5) participant-by-role probabilities -> (marginals W (n,5,5), assignment entropy (n,), log Z).

    Weight of assignment pi proportional to prod_i p_i(pi_i) over the 120 one-to-one assignments; normalized by
    logsumexp; W[t, i, r] = posterior probability that participant i holds role r. Probabilities are clipped at
    1e-12 only inside the logarithm. Rows and columns of W sum to one.
    """
    from scipy.special import logsumexp
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 3 or P.shape[1:] != (5, 5):
        raise ValueError('P must be (n, 5, 5)')
    logp = np.log(np.clip(P, PROB_CLIP, 1.0))
    out_W = np.empty_like(P)
    out_H = np.empty(len(P))
    out_Z = np.empty(len(P))
    for a in range(0, len(P), 20000):
        lp = logp[a:a + 20000]
        s = lp[:, np.arange(5)[None, :], PERMS].sum(axis=2)          # (n, 120)
        z = logsumexp(s, axis=1)
        post = np.exp(s - z[:, None])
        out_W[a:a + 20000] = np.einsum('nk,kir->nir', post, PERM_ONEHOT)
        out_H[a:a + 20000] = -np.sum(np.where(post > 0, post * np.log(np.where(post > 0, post, 1.0)), 0.0), axis=1)
        out_Z[a:a + 20000] = z
    return out_W, out_H, out_Z


# ------------------------------------------------------------------ participant features and role representation
PART_FEATURES = ('totalGold_norm', 'curGold_norm', 'level_norm', 'xp_norm', 'hp_pct', 'mp_pct', 'alive', 'laneCS_norm',
                 'jgCS_norm', 'kills', 'deaths', 'death_since_snapshot', 'death_last_30s', 'death_age_minutes',
                 'baron_death_since_acquisition', 'elder_death_since_acquisition')
PHASE_FEATURES = ('totalGold_norm', 'xp_norm', 'level_norm')
UNCERTAINTY_FEATURES = ('assignment_entropy', 'mean_max_marginal', 'min_max_marginal', 'champion_missing_count',
                        'champion_unseen_count', 'spell_missing_count', 'spell_unseen_count')


def participant_columns(names):
    """Indices of participant_slot{0..9}_{f} for f in PART_FEATURES: (10, 16) and champion columns (10,).

    StateV2 slot k = k-th participant sorted by (teamId, participantId): slots 0-4 Blue (100), 5-9 Red (200).
    """
    ix = {n: i for i, n in enumerate(names)}
    P = np.asarray([[ix[f'participant_slot{s}_{f}'] for f in PART_FEATURES] for s in range(10)], dtype=np.int64)
    champ = np.asarray([ix[f'participant_slot{s}_champion_id'] for s in range(10)], dtype=np.int64)
    return P, champ


def global_names(names):
    return [n for n in names if not n.startswith('participant_slot')]


def role_feature_names():
    out = []
    for team in ('blue', 'red'):
        for r in ROLES:
            out += [f'role_{team}_{r}_{f}' for f in PART_FEATURES]
    for r in ROLES:
        out += [f'role_diff_{r}_{f}' for f in PART_FEATURES]
    for r in ROLES:
        out += [f'role_diff_{r}_{f}_x_time_minutes' for f in PHASE_FEATURES]
    for team in ('blue', 'red'):
        out += [f'roleunc_{team}_{u}' for u in UNCERTAINTY_FEATURES]
    return out


def role_group(n):
    """7 explanation groups for the role arm: 5 roles, global context, role uncertainty."""
    if n.startswith('roleunc_'):
        return 'role_uncertainty'
    for r in ROLES:
        if n.startswith(f'role_blue_{r}_') or n.startswith(f'role_red_{r}_') or n.startswith(f'role_diff_{r}_'):
            return f'role_{r}'
    return 'global_context'


ROLE_GROUPS = tuple(f'role_{r}' for r in ROLES) + ('global_context', 'role_uncertainty')


def build_role_matrix(X, names, W_rows, U_rows):
    """Role-aware q inputs for rows of an existing pre-state matrix.

    X: (n, len(names)) pre-only inputs (participant slots in StateV2 order: team then participant ID).
    W_rows: (n, 10, 5) role marginals per slot (slots 0-4 blue, 5-9 red; each team block row/col sums 1).
    U_rows: (n, 2, 7) team uncertainty summaries (UNCERTAINTY_FEATURES order).
    Returns (Z, z_names): global features (all non-participant inputs incl. p_pre_V), then role block.
    """
    X = np.asarray(X, dtype=np.float64)
    P, _champ = participant_columns(names)
    gn = global_names(names)
    gix = [names.index(n) for n in gn]
    x_part = X[:, P]                                                     # (n, 10, 16)
    blue = np.einsum('nir,nif->nrf', W_rows[:, :5, :], x_part[:, :5, :])  # (n, 5, 16)
    red = np.einsum('nir,nif->nrf', W_rows[:, 5:, :], x_part[:, 5:, :])
    diff = blue - red
    t = X[:, names.index('time_minutes')]
    ph = [PART_FEATURES.index(f) for f in PHASE_FEATURES]
    inter = diff[:, :, ph] * t[:, None, None]
    Z = np.concatenate([X[:, gix], blue.reshape(len(X), -1), red.reshape(len(X), -1), diff.reshape(len(X), -1),
                        inter.reshape(len(X), -1), U_rows.reshape(len(X), -1)], axis=1)
    return Z, gn + role_feature_names()


# ------------------------------------------------------------------ controlled h90 arms (C)
ARMS = ('participant', 'draft', 'role')
ARM_BASES = ('ridge', 'lgbm')
CALS = ('raw', 'sigmoid', 'isotonic')
RIDGE_C = 0.01
RIDGE_MAX_ITER = 3000


def arm_candidates(arm):
    return [f'{arm}_{b}_{c}' for b in ARM_BASES for c in CALS]


class DraftSlotEncoder:
    """Per-participant-slot draft indicators fit on cohort TRAIN rows only.

    champion: OneHotEncoder(categories = per-slot positive TRAIN champion IDs, handle_unknown='ignore');
    spells: per-slot unordered multi-hot over the slot's positive TRAIN spell IDs. Missing / unseen -> all zero.
    """

    def fit(self, champ, spa, spb):
        from sklearn.preprocessing import OneHotEncoder
        champ, spa, spb = (np.asarray(a, dtype=np.int64) for a in (champ, spa, spb))
        self.champ_cats = [np.unique(champ[:, j][champ[:, j] > 0]) for j in range(10)]
        self.spell_cats = [np.unique(np.concatenate([spa[:, j], spb[:, j]])[np.concatenate([spa[:, j], spb[:, j]]) > 0]) for j in range(10)]
        self.champ_enc = OneHotEncoder(categories=self.champ_cats, handle_unknown='ignore', sparse_output=True, dtype=np.float64)
        self.champ_enc.fit(np.stack([c[:1] if len(c) else np.array([1]) for c in self.champ_cats], axis=1))
        self.spell_enc = OneHotEncoder(categories=self.spell_cats, handle_unknown='ignore', sparse_output=True, dtype=np.float64)
        self.spell_enc.fit(np.stack([c[:1] if len(c) else np.array([1]) for c in self.spell_cats], axis=1))
        return self

    def transform(self, champ, spa, spb):
        import scipy.sparse as sp
        import warnings
        champ, spa, spb = (np.asarray(a, dtype=np.int64) for a in (champ, spa, spb))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')          # unknown categories are encoded as all zeros by design
            Xc = self.champ_enc.transform(champ)
            S = self.spell_enc.transform(spa) + self.spell_enc.transform(spb)
        S.data = np.minimum(S.data, 1.0)
        return sp.hstack([Xc, S], format='csr')

    @property
    def feature_names(self):
        out = []
        for j in range(10):
            out += [f'draft_slot{j}_champion={int(c)}' for c in self.champ_cats[j]]
        for j in range(10):
            out += [f'draft_slot{j}_spell={int(s)}' for s in self.spell_cats[j]]
        return out


class ArmFeaturizer:
    """inputs dict (X, names, champ, spa, spb, W, U) -> design matrix of one arm; fitted parts see cohort TRAIN only."""

    def __init__(self, arm, names, drop_groups=()):
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.names = list(names)
        self.drop_groups = tuple(drop_groups)
        ridge = C.q_feature_sets(self.names)['ridge']
        self.numeric_cols = [self.names.index(n) for n in ridge]
        self.numeric_names = list(ridge)
        self.draft_encoder = None
        if arm == 'role':
            self.role_names = global_names(self.names) + role_feature_names()
            keep = [i for i, n in enumerate(self.role_names) if role_group(n) not in self.drop_groups]
            self.role_keep = keep
        elif self.drop_groups:
            raise ValueError('drop_groups only for the role arm')

    def fit(self, inputs, rows):
        if self.arm == 'draft':
            self.draft_encoder = DraftSlotEncoder().fit(inputs['champ'][rows], inputs['spa'][rows], inputs['spb'][rows])
        return self

    @property
    def feature_names(self):
        if self.arm == 'participant':
            return self.numeric_names
        if self.arm == 'draft':
            return self.numeric_names + self.draft_encoder.feature_names
        return [self.role_names[i] for i in self.role_keep]

    @property
    def n_scaled(self):
        return len(self.numeric_names) if self.arm == 'draft' else len(self.feature_names)

    def transform(self, inputs, rows=None):
        sl = slice(None) if rows is None else rows
        X = inputs['X'][sl]
        forbidden = [n for n in self.feature_names if any(t in n for t in FORBIDDEN_ARM_TOKENS)]
        if forbidden:
            raise ValueError(f'forbidden arm inputs {forbidden[:5]}')
        if self.arm == 'participant':
            return X[:, self.numeric_cols]
        if self.arm == 'draft':
            import scipy.sparse as sp
            D = self.draft_encoder.transform(inputs['champ'][sl], inputs['spa'][sl], inputs['spb'][sl])
            return sp.hstack([sp.csr_matrix(X[:, self.numeric_cols]), D], format='csr')
        if list(inputs['names']) != self.names:
            raise ValueError('names differ')
        Z, zn = build_role_matrix(X, self.names, inputs['W'][sl], inputs['U'][sl])
        if zn != self.role_names:
            raise ValueError('role schema drift')
        return Z[:, self.role_keep]


FORBIDDEN_ARM_TOKENS = ('cohort', 'cluster', 'present_', 'n_min', 'teamPosition', 'role_slot', 'weak_role', 'snapshot_age', 'p_post',
                        'delta', 'endpoint', 'duration', 'label', 'winner', 'game_end', 'next_kill', 'next_start', 'L_ms',
                        'match', 'patch', 'fold', 'champion_id')


class RidgeBase:
    """StandardScaler on the first n_scaled columns (dense), remaining sparse indicators unscaled -> LogisticRegression(C=.01)."""

    def fit(self, M, y, w, n_scaled):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        import scipy.sparse as sp
        self.n_scaled = int(n_scaled)
        self.sparse = sp.issparse(M)
        if not self.sparse:
            self.pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=RIDGE_C, max_iter=RIDGE_MAX_ITER))
            self.pipeline.fit(M, y, logisticregression__sample_weight=w)
            self.n_iter = int(self.pipeline[-1].n_iter_[0])
        else:
            self.scaler = StandardScaler().fit(M[:, :self.n_scaled].toarray())
            self.lr = LogisticRegression(C=RIDGE_C, max_iter=RIDGE_MAX_ITER).fit(self._design(M), y, sample_weight=w)
            self.n_iter = int(self.lr.n_iter_[0])
        self.converged = self.n_iter < RIDGE_MAX_ITER
        return self

    def _design(self, M):
        import scipy.sparse as sp
        num = sp.csr_matrix(self.scaler.transform(M[:, :self.n_scaled].toarray()))
        return sp.hstack([num, M[:, self.n_scaled:]], format='csr')

    def decision(self, M):
        return self.pipeline.decision_function(M) if not self.sparse else self.lr.decision_function(self._design(M))

    def raw(self, M):
        return self.pipeline.predict_proba(M)[:, 1] if not self.sparse else self.lr.predict_proba(self._design(M))[:, 1]

    def linear_parts(self):
        """beta (per input column, on raw input scale for scaled columns) and intercept of the log-odds."""
        if self.sparse:
            coef = self.lr.coef_[0].copy()
            b0 = self.lr.intercept_[0] - np.sum(coef[:self.n_scaled] * self.scaler.mean_ / self.scaler.scale_)
            coef[:self.n_scaled] = coef[:self.n_scaled] / self.scaler.scale_
            return coef, float(b0)
        sc, lr = self.pipeline[0], self.pipeline[-1]
        return lr.coef_[0] / sc.scale_, float(lr.intercept_[0] - np.sum(lr.coef_[0] * sc.mean_ / sc.scale_))


class LgbmBase:
    def fit(self, M, y, w):
        from lightgbm import LGBMClassifier
        self.seeds = list(C.ECON_SEEDS)
        self.models = []
        for seed in self.seeds:
            m = LGBMClassifier(random_state=seed, **C.LGBM_PARAMS)
            m.fit(M, y, sample_weight=w)
            self.models.append(m)
        return self

    def raw(self, M):
        return np.mean([m.predict_proba(M)[:, 1] for m in self.models], axis=0)


def fit_calibrators(raw_cal, y_cal, w_cal):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    sg = LogisticRegression(C=1e6, max_iter=1000).fit(C.logit8(raw_cal), y_cal, sample_weight=w_cal)
    iso = IsotonicRegression(out_of_bounds='clip').fit(raw_cal, y_cal, sample_weight=w_cal)
    return dict(sigmoid=sg, isotonic=iso)


def apply_calibration(cal_kind, calibrator, raw):
    if cal_kind == 'raw':
        return raw
    if cal_kind == 'sigmoid':
        return calibrator.predict_proba(C.logit8(raw))[:, 1]
    if cal_kind == 'isotonic':
        return calibrator.predict(raw)
    raise ValueError(cal_kind)


def predict_arm_bundle(b, inputs, rows=None, M=None):
    """Final calibrated probability of an arm candidate bundle."""
    if M is None:
        M = b['featurizer'].transform(inputs, rows)
    raw = b['base_model'].raw(M)
    return apply_calibration(b['calibration'], b.get('calibrator'), raw)
