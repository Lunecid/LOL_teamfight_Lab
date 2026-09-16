"""Track A MLP 2026-09-16: shared helpers (pure; importing has no side effects beyond environment defaults).

Design: docs/CLAUDE_TRACK_A_MLP_20260916.md. Everything written by this run lands in outputs/track_a_mlp_20260916.
Parents (full corpus, cohort/role, incremental q, label validity, objective ablation, critique audit) are READ ONLY;
parent helpers that write logs or gates (iq20260915_common.Status/log_command/log_failure/log_access/load_parent_set/
load_trainval, cr20260915_data.*, cr20260915_common.outcome_gate/Status/log_command) are never called.

Reused pure helpers: iq20260915_common (config-free): stop_mask, weights, weight_record, candidate helpers, select_rule,
calibrate/bundle_raw/bundle_predict/calibrator_record, cell_masks, cell_metrics, paired_bootstrap, key_join,
oof_provenance, environment (extended here), smoke_keep; fc20260915_common: evaluate/brier_direct/logloss_direct/
sha256_*/write_json/read_json/save_npz/clean/q_feature_sets/train_fold; cr20260915_common: fit_calibrators/CALS.
Model class MlpQBase is serialised with joblib (state dicts as numpy arrays, no torch objects), so scripts/ must be on
sys.path and this module importable as ta20260916_common when bundles are loaded.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # deterministic cuBLAS (must precede CUDA initialisation)
os.environ.setdefault('MKL_CBWR', 'AVX2,STRICT')  # reproducible MKL code path for the CPU float64 prediction definition
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np  # noqa: E402

import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402
import iq20260915_common as Q  # noqa: E402

ROOT = C.ROOT
OUT = ROOT / 'outputs' / 'track_a_mlp_20260916'
SMOKE = OUT / 'smoke_train_only'
FC = C.OUT
CR = K.OUT
IQ = Q.OUT
SPEC = ROOT / 'docs' / 'CLAUDE_TRACK_A_MLP_20260916.md'
PROVENANCE_DOCS = (Q.SPEC, *Q.PROVENANCE_DOCS, ROOT / 'docs' / 'COG_MODEL_COVERAGE_DELTA_Q_20260915.md')
PYTHON = Q.PYTHON
VERSION = 'track_a_mlp_20260916'
ROLE_TAG = 'EXPLORATORY_FOLLOWUP_AFTER_PRIOR_TEST_EXPOSURE_MODEL_DEFINED_LABELS_NOT_CONFIRMATORY'

H = Q.H
COHORTS = Q.COHORTS
COHORT_CODE = Q.COHORT_CODE
EXT = Q.EXT
EVAL_SETS = Q.EVAL_SETS
SEALED_SETS = Q.SEALED_SETS
EXPECTED_COUNTS = Q.EXPECTED_COUNTS
EXPECTED_RIDGE_COUNT = Q.EXPECTED_RIDGE_COUNT

# ------------------------------------------------------------------ fixed search spaces (spec "Two newly fitted families")
FAMILIES = ('mlp', 'resmlp')
WIDTHS = (128, 256, 512)
DROPOUTS = (0.1, 0.3)
GRID = tuple((w, d) for w in WIDTHS for d in DROPOUTS)
SEEDS = (7, 42, 123)
OPT = dict(optimizer='AdamW', lr=1e-3, weight_decay=1e-4, weight_decay_scope='all parameters (AdamW default)',
           batch_size=512, max_epochs=100, patience=10, dtype_train='float32', amp=False, grad_clip=None,
           init='PyTorch default (kaiming_uniform for Linear)', shuffle='uniform permutation per epoch from a CPU torch.Generator seeded with the seed',
           last_batch='kept (partial)')
PRED_CHUNK = 8192
CPU_THREADS = 4
PRED_THREADS = 1  # inference on CPU float64 with one thread: device- and thread-count-independent prediction definition
STOP_TAG = Q.STOP_TAG  # the iq allocation -> same stop10 matches as the LightGBM arm
CALS = K.CALS
BOOT_REPS = Q.BOOT_REPS
BOOT_SEED = Q.BOOT_SEED
SPARSE_MATCHES = Q.SPARSE_MATCHES
TIME_BINS = Q.TIME_BINS
BALANCED = Q.BALANCED
IQ_FAMILIES = Q.FAMILIES  # ('pt', 'logit', 'lgbm')
IQ_WINNERS = Q.NEW_WINNERS  # ('pt_winner', 'logit_winner', 'lgbm_winner')
IQ_LEGACY = tuple(Q.LEGACY)  # old_A_specialist, old_pooled, old_p_pre_spline, old_p_pre_logistic, old_constant
NEW_WINNERS = ('mlp_winner', 'resmlp_winner')
NAMED = NEW_WINNERS + ('overall_winner_mlp_families',) + IQ_WINNERS + IQ_LEGACY
BOOT_MODELS = NEW_WINNERS + IQ_WINNERS
CONTRASTS = (  # (a, b, label); a - b; negative Brier / log loss = a better
    ('resmlp_winner', 'lgbm_winner', 'PRIMARY: residual MLP - full LightGBM'),
    ('mlp_winner', 'logit_winner', 'planned: plain MLP - full logistic'),
    ('resmlp_winner', 'mlp_winner', 'planned: residual MLP - plain MLP'),
    ('mlp_winner', 'lgbm_winner', 'planned: plain MLP - full LightGBM'),
    ('resmlp_winner', 'pt_winner', 'planned: residual MLP - PT baseline'),
    ('mlp_winner', 'pt_winner', 'planned: plain MLP - PT baseline'))


def config_names(family):
    if family not in FAMILIES:
        raise ValueError(family)
    return [f'{family}_W{w}_D{d:g}' for w, d in GRID]


def config_params(family, config):
    i = config_names(family).index(config)
    return dict(width=GRID[i][0], dropout=GRID[i][1])


def candidate_names(family):
    """18 joint candidates '<config>__<calibration>' in lexical tie order."""
    return sorted(f'{cfg}__{cal}' for cfg in config_names(family) for cal in CALS)


split_candidate = Q.split_candidate
select_rule = Q.select_rule
stop_mask = Q.stop_mask
weights = Q.weights
weight_record = Q.weight_record
membership_sha = Q.membership_sha
smoke_keep = Q.smoke_keep
first_argmin = Q.first_argmin_iteration
cell_masks = Q.cell_masks
cell_metrics = Q.cell_metrics
paired_bootstrap = Q.paired_bootstrap
key_join = Q.key_join
oof_provenance = Q.oof_provenance
calibrate = Q.calibrate
bundle_raw = Q.bundle_raw
bundle_predict = Q.bundle_predict
calibrator_record = Q.calibrator_record


# ------------------------------------------------------------------ torch models
def torch_import():
    import torch
    return torch


def build_module(family, n_in, width, dropout):
    """mlp: Linear-ReLU-Dropout-Linear-ReLU-Dropout-Linear(1). resmlp: projection, two residual blocks, LN-ReLU-Linear(1)."""
    import torch.nn as nn
    if family == 'mlp':
        return nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Dropout(dropout),
                             nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
                             nn.Linear(width, 1))
    if family == 'resmlp':
        return _res_classes()[1](n_in, width, dropout)
    raise ValueError(family)


_RES_CACHE = {}


def _res_classes():
    """(ResBlock, ResMLP) built once per process (torch imported lazily)."""
    if _RES_CACHE:
        return _RES_CACHE['ResBlock'], _RES_CACHE['ResMLP']
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, width, dropout):
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.lin1 = nn.Linear(width, 2 * width)
            self.act = nn.ReLU()
            self.drop1 = nn.Dropout(dropout)
            self.lin2 = nn.Linear(2 * width, width)
            self.drop2 = nn.Dropout(dropout)

        def forward(self, x):
            return x + self.drop2(self.lin2(self.drop1(self.act(self.lin1(self.norm(x))))))

    class ResMLP(nn.Module):
        def __init__(self, n_in, width, dropout):
            super().__init__()
            self.proj = nn.Linear(n_in, width)
            self.blocks = nn.ModuleList([ResBlock(width, dropout), ResBlock(width, dropout)])
            self.norm = nn.LayerNorm(width)
            self.act = nn.ReLU()
            self.head = nn.Linear(width, 1)

        def forward(self, x):
            h = self.proj(x)
            for b in self.blocks:
                h = b(h)
            return self.head(self.act(self.norm(h)))

    _RES_CACHE.update(ResBlock=ResBlock, ResMLP=ResMLP)
    return ResBlock, ResMLP


def param_count(module):
    return int(sum(p.numel() for p in module.parameters()))


def layer_shapes(module):
    return {n: list(p.shape) for n, p in module.named_parameters()}


def torch_setup(device, seed):
    """Deterministic torch state for one training run (seed -> init, dropout masks and minibatch order)."""
    torch = torch_import()
    torch.manual_seed(int(seed))
    torch.set_num_threads(CPU_THREADS)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('cuda requested but not available')
    return torch


def weighted_bce_mean(logits, y, w):
    """Minibatch loss: mean(w_i * BCEWithLogits_i) (no per-batch renormalisation)."""
    import torch.nn.functional as F
    return (w * F.binary_cross_entropy_with_logits(logits, y, reduction='none')).mean()


def population_weighted_bce(logits, y, w):
    """Population loss sum(w_i * BCE_i) / sum(w_i) (numpy, float64)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    bce = np.logaddexp(0.0, -z) * y + np.logaddexp(0.0, z) * (1.0 - y)
    return float(np.sum(w * bce) / np.sum(w))


def _stop_brier_gpu(torch, module, Xs, ys, ws):
    module.eval()
    with torch.no_grad():
        out = []
        for i in range(0, len(ys), PRED_CHUNK):
            out.append(torch.sigmoid(module(Xs[i:i + PRED_CHUNK]).squeeze(1)))
        p = torch.cat(out)
        return float((ws * (p - ys) ** 2).sum() / ws.sum())


def run_training(family, width, dropout, Z, y, w, seed, device, epochs, stop=None, patience=None, log=None):
    """Train from scratch. stop=(Zs, ys, ws) enables early stopping on the match-weighted stop Brier (strict improvement,
    first minimum kept, patience epochs). Returns (module, curve, best_epoch, epochs_run)."""
    torch = torch_setup(device, seed)
    module = build_module(family, Z.shape[1], width, dropout).to(device)
    opt = torch.optim.AdamW(module.parameters(), lr=OPT['lr'], weight_decay=OPT['weight_decay'])
    Zg = torch.as_tensor(np.asarray(Z, dtype=np.float32), device=device)
    yg = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device)
    wg = torch.as_tensor(np.asarray(w, dtype=np.float32), device=device)
    if stop is not None:
        Zs = torch.as_tensor(np.asarray(stop[0], dtype=np.float32), device=device)
        ys = torch.as_tensor(np.asarray(stop[1], dtype=np.float32), device=device)
        ws = torch.as_tensor(np.asarray(stop[2], dtype=np.float32), device=device)
    gen = torch.Generator(device='cpu').manual_seed(int(seed))
    n, bs = len(y), OPT['batch_size']
    curve, best, best_epoch, wait, ep = [], None, 0, 0, 0
    for ep in range(1, int(epochs) + 1):
        module.train()
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = weighted_bce_mean(module(Zg[idx]).squeeze(1), yg[idx], wg[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if stop is not None:
            b = _stop_brier_gpu(torch, module, Zs, ys, ws)
            curve.append(b)
            if best is None or b < best:
                best, best_epoch, wait = b, ep, 0
            else:
                wait += 1
            if log and (ep % 10 == 0 or ep == 1):
                log(f'    epoch {ep} stop_brier={b:.6f} best_epoch={best_epoch}')
            if wait >= int(patience):
                break
    return module, curve, best_epoch, ep


def state_to_numpy(module):
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in module.state_dict().items()}


def predict_cpu64(family, width, dropout, state, Z):
    """Prediction definition: saved float32 weights evaluated on CPU in float64, one thread, fixed chunk size."""
    torch = torch_import()
    torch.set_num_threads(PRED_THREADS)
    module = build_module(family, Z.shape[1], width, dropout).double()
    module.load_state_dict({k: torch.from_numpy(np.asarray(v, dtype=np.float32)).double() for k, v in state.items()})
    module.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Z), PRED_CHUNK):
            zc = torch.from_numpy(np.ascontiguousarray(Z[i:i + PRED_CHUNK], dtype=np.float64))
            out.append(torch.sigmoid(module(zc).squeeze(1)).numpy())
    return np.concatenate(out) if out else np.zeros(0)


class MlpQBase:
    """Three-seed MLP family on the 352 ridge inputs (SimpleImputer(median) -> StandardScaler, both TRAIN-fitted).

    Per seed: stop phase (fit90 rows -> preprocessing fit on fit90, early stopping on stop10 match-weighted Brier,
    patience 10, ties earlier, cap 100) then a from-scratch refit on ALL cohort TRAIN rows (preprocessing refit there)
    for exactly the chosen number of epochs. Candidate raw probability = mean of the three seed probabilities, each
    computed on CPU in float64 from the saved float32 weights."""

    def __init__(self, family, config, input_names, seeds=SEEDS, device='cuda', max_epochs=None, patience=None):
        if family not in FAMILIES:
            raise ValueError(family)
        self.family = family
        self.config = config
        self.params = config_params(family, config)
        self.input_names = list(input_names)
        self.seeds = tuple(seeds)
        self.device = device
        self.max_epochs = int(OPT['max_epochs'] if max_epochs is None else max_epochs)
        self.patience = int(OPT['patience'] if patience is None else patience)

    def _pre(self, X):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        imp = SimpleImputer(strategy='median').fit(X)
        sc = StandardScaler().fit(imp.transform(X))
        return imp, sc

    def fit(self, Xin, y, g, log=None):
        Xin = np.asarray(Xin, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        g = np.asarray(g).astype(str)
        if not np.isfinite(Xin).all():
            raise ValueError('non-finite TRAIN inputs (parent schema contract violated)')
        stop = stop_mask(g)
        fit90 = ~stop
        w_fit, w_stop, w_all = weights(g[fit90]), weights(g[stop]), weights(g)
        imp90, sc90 = self._pre(Xin[fit90])
        Z90 = sc90.transform(imp90.transform(Xin[fit90]))
        Zst = sc90.transform(imp90.transform(Xin[stop]))
        self.imputer, self.scaler = self._pre(Xin)
        Zall = self.scaler.transform(self.imputer.transform(Xin))
        width, dropout = self.params['width'], self.params['dropout']
        self.stop_record = dict(allocation=f"sha256('{STOP_TAG}'+match_id)[:8] mod {Q.STOP_MOD} == 0 -> stop10 (iq allocation)",
                                fit90=weight_record(g[fit90], w_fit), stop10=weight_record(g[stop], w_stop), full=weight_record(g, w_all),
                                fit90_stop10_match_overlap=int(len(set(g[fit90].tolist()) & set(g[stop].tolist()))),
                                preprocessing_stop_phase='median imputer + StandardScaler fit on fit90 rows only',
                                preprocessing_full='median imputer + StandardScaler refit on all cohort TRAIN rows',
                                evaluator='match-weighted Brier on stop10 (weights recomputed inside stop10), GPU float32 inference, eval mode',
                                patience=self.patience, max_epochs=self.max_epochs, device=self.device, optimizer=OPT, seeds={})
        self.states, self.arch = [], None
        for seed in self.seeds:
            t0 = time.time()
            m_stop, curve, best, ran = run_training(self.family, width, dropout, Z90, y[fit90], w_fit, seed, self.device,
                                                    self.max_epochs, stop=(Zst, y[stop], w_stop), patience=self.patience, log=log)
            if best < 1:
                raise RuntimeError('no epoch recorded in the stop phase')
            stop_s = time.time() - t0
            t1 = time.time()
            m_final, _, _, ran_final = run_training(self.family, width, dropout, Zall, y, w_all, seed, self.device, best)
            self.states.append(state_to_numpy(m_final))
            if self.arch is None:
                self.arch = dict(n_params=param_count(m_final), layers=layer_shapes(m_final), repr=str(m_final))
            self.stop_record['seeds'][str(seed)] = dict(
                best_epoch=int(best), first_argmin_epoch=first_argmin(curve), epochs_evaluated=len(curve),
                cap_reached=bool(len(curve) >= self.max_epochs and best == self.max_epochs),
                early_stopped=bool(len(curve) < self.max_epochs), best_stop_weighted_brier=float(curve[best - 1]),
                stop_curve=[float(v) for v in curve], refit_epochs=int(ran_final),
                stop_seconds=round(stop_s, 2), refit_seconds=round(time.time() - t1, 2))
            if log:
                log(f'{self.config} seed {seed}: best_epoch={best} epochs_evaluated={len(curve)} stop_brier={curve[best - 1]:.6f} '
                    f'stop {stop_s:.1f}s refit {time.time() - t1:.1f}s')
            m_stop = m_final = None
        return self

    def design(self, Xin):
        Xin = np.asarray(Xin, dtype=np.float64)
        if Xin.shape[1] != len(self.input_names):
            raise ValueError('input width mismatch')
        return self.scaler.transform(self.imputer.transform(Xin))

    def seed_raw(self, Xin):
        Z = self.design(Xin)
        return np.stack([predict_cpu64(self.family, self.params['width'], self.params['dropout'], st, Z) for st in self.states])

    def raw(self, Xin):
        return self.seed_raw(Xin).mean(axis=0)


# ------------------------------------------------------------------ bundles
def input_columns(names, ridge):
    return [names.index(n) for n in ridge]


def make_bundle(family, config, base, calibrators, input_names_all, ridge, manifest):
    return dict(version=VERSION, family=family, config=config, params=config_params(family, config),
                input_names_all_sha256=C.sha256_json(list(input_names_all)), input_names=list(ridge),
                input_columns=input_columns(list(input_names_all), ridge), base=base, calibrators=calibrators, manifest=manifest)


# ------------------------------------------------------------------ parent data (read-only) with access logging in THIS root
def append_jsonl(path, rec):
    Q.append_jsonl(path, rec)


def frozen_path(base=OUT):
    return Path(base) / 'frozen_manifest.json'


def log_access(base, purpose, set_name, kind, sealed):
    """Append-only access record in THIS run; sealed sets (TEST/external, iq/parent TEST predictions) need THIS run's freeze
    (frozen_manifest.json under OUT), regardless of the iq freeze."""
    fz = frozen_path(OUT)
    if sealed and not fz.exists():
        raise PermissionError(f'{purpose}: {set_name} is sealed until {fz} exists')
    append_jsonl(Path(base) / 'access_log.jsonl', dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(), purpose=purpose,
                                                       set=set_name, kind=kind, sealed=bool(sealed), frozen_manifest_exists=fz.exists(),
                                                       frozen_manifest_sha256=C.sha256_file(fz) if fz.exists() else None))


LABEL_KEYS = Q.LABEL_KEYS


def load_parent_set(name, base, purpose):
    """Features (pre-only X_input), h90 labels and cohort arrays of one parent set with exact row-order checks
    (iq20260915_common.load_parent_set with this run's sealed gate)."""
    if name.startswith('MAIN_TRAIN') or name == 'MAIN_VALIDATION':
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
    """TRAIN (+ VALIDATION Q_CAL/Q_SELECT) h90-valid rows of one cohort with role labels (iq20260915_common.load_trainval
    with this run's access log). smoke: TRAIN only; pseudo roles fold0-2 TRAIN, fold3 Q_CAL, fold4 Q_SELECT; 1/8 of matches."""
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


# ------------------------------------------------------------------ status / logging (this root only)
class Status(Q.Status):
    def __init__(self, group, out=OUT):
        super().__init__(group, out=out)

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
    env = Q.environment()
    try:
        import torch
        env.update(torch=torch.__version__, cuda=torch.version.cuda, cuda_available=bool(torch.cuda.is_available()),
                   cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   cudnn=torch.backends.cudnn.version(), cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                   mkl_cbwr=os.environ.get('MKL_CBWR'), torch_mkl=bool(torch.backends.mkl.is_available()),
                   torch_num_threads=int(torch.get_num_threads()), pred_threads=PRED_THREADS, pred_chunk=PRED_CHUNK)
    except Exception as exc:  # torch missing: recorded, fitting will fail loudly
        env.update(torch=f'unavailable: {exc!r}')
    return env


def own_source_hashes():
    files = sorted((ROOT / 'scripts').glob('ta20260916_*.py')) + sorted((ROOT / 'tests').glob('test_ta20260916_*.py'))
    reused = [ROOT / 'scripts' / 'fc20260915_common.py', ROOT / 'scripts' / 'cr20260915_common.py', ROOT / 'scripts' / 'iq20260915_common.py',
              SPEC, *PROVENANCE_DOCS]
    return {p.relative_to(ROOT).as_posix(): C.sha256_file(p) for p in files + reused if p.exists()}


def threads_env(n=CPU_THREADS, cuda=True):
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[v] = str(n)
    if not cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('MKL_CBWR', 'AVX2,STRICT')


def iq_frozen():
    return C.read_json(Q.frozen_path(IQ))


def iq_eval_predictions_path(set_name, coh):
    return IQ / 'eval' / 'predictions' / f'{set_name}_h90_{coh}.npz'
