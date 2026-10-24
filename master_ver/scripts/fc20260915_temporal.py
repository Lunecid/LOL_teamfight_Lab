"""Full-corpus comparator stage R: temporal SimpleRNN V (P2 CandidateB architecture) retrained on full TRAIN keys.

Lowest priority; never a prerequisite and never replaces the primary labeler. Uses the SAME full TRAIN bucket query
keys as primary V and the P2 causal history (nine one-minute positions t-8min..t, StateV2 at each position, positions
before the first frame zero-padded and masked). Preprocessing (numeric mean/std, per-slot champion vocabularies) is fit
on TRAIN history states only. Architecture/training from P2: SimpleRNN 8 tanh units, input dropout .25, RMSprop lr .001,
batch 64, 50 epochs, seeds 17/29/43, CPU torch 4 threads. Loss weights: equal total weight per match (the full-corpus V
fitting rule; P2 itself was unweighted -- disclosed). Per seed: raw and positive-slope sigmoid fitted on V_CAL, chosen by
V_SELECT match-weighted log loss; fixed arithmetic mean ensemble. Evaluated after the primary freeze on TEST bucket
queries and full minute trajectories against the frozen primary V.

Subcommands: extract (history state tables), train, evaluate.
"""
from __future__ import annotations

import os

_WORKER = os.environ.get('FC_WORKER') == '1'
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import fc20260915_data as D  # noqa: E402

sys.path.insert(0, str(C.WT))
import json  # noqa: E402

os.environ['LOL_OUTPUT_ROOT'] = str(C.OUT / 'runtime')
os.environ['LOL_CFG_PRESET'] = 'v3.3'
os.environ['LOL_CFG_OVERRIDES'] = json.dumps({
    'CACHE_DIRNAME': str(C.CACHE_MAIN), 'FIGHT_INDEX_CACHE_ENABLED': False,
    'FIGHT_INDEX_NUM_WORKERS': 1, 'DUMP_FIGHTS': False, 'CACHE_IN_RAM': False})

import argparse  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

SEEDS = (17, 29, 43)
UNITS, DROPOUT, LR, BATCH, EPOCHS, RHO, EPS = 8, .25, 1e-3, 64, int(os.environ.get('FC_TEMPORAL_SMOKE_EPOCHS', 50)), .9, 1e-7
STEP, NPOS = 60_000, 9
ROLES = ('TRAIN', 'V_CAL', 'V_SELECT', 'TEST')


def tdir():
    return Path(os.environ.get('FC_TEMPORAL_OUT', str(C.OUT / 'temporal_comparator')))


def hdir():
    return Path(os.environ.get('FC_TEMPORAL_HISTORY', str(C.OUT / 'temporal_comparator' / 'history')))


# ------------------------------------------------------------------ extraction of minute-position states
def history_chunk(task):
    import fc20260915_extract as X
    ctx = X._context()
    cio = ctx['cio']
    cio.CACHE_DIR = C.CACHE_MAIN
    names = task['names']
    out = dict(match=[], pos_ms=[], snapshot_ms=[], X=[], support=[], q_match=[], q_ms=[], q_bucket=[], check_equal=0, check_rows=0)
    for mid, queries, bucket_flags, bucket_queries, vstates in task['items']:
        pack = cio.load_match_cache(mid)
        b = ctx['V2'](pack, ctx['node_names'])
        ts = np.asarray(pack['minute_ts'], dtype=np.int64)
        support = max(0, int(ts[0]))
        top = max(queries)
        positions = [p for p in range(0, top + 1, STEP) if p >= support]
        sts = [b.at(p) for p in positions]
        Xs = ctx['state_matrix'](sts, names)
        pos_index = {p: i for i, p in enumerate(positions)}
        for q, v in zip(bucket_queries, vstates):
            out['check_rows'] += 1
            out['check_equal'] += int(np.array_equal(Xs[pos_index[q]], v))
        out['match'] += [mid] * len(positions)
        out['pos_ms'] += positions
        out['snapshot_ms'] += [s.snapshot_ms for s in sts]
        out['X'].append(Xs.astype(np.float32))
        out['support'] += [support] * len(positions)
        out['q_match'] += [mid] * len(queries)
        out['q_ms'] += list(queries)
        out['q_bucket'] += list(bucket_flags)
    C.save_npz(Path(task['path']), match=np.asarray(out['match'], dtype='U24'), pos_ms=np.asarray(out['pos_ms'], dtype=np.int64),
               snapshot_ms=np.asarray(out['snapshot_ms'], dtype=np.int64), X=np.concatenate(out['X']),
               support_ms=np.asarray(out['support'], dtype=np.int64), q_match=np.asarray(out['q_match'], dtype='U24'),
               q_ms=np.asarray(out['q_ms'], dtype=np.int64), q_is_bucket=np.asarray(out['q_bucket'], dtype=np.int8),
               plan_sha256=np.asarray(task['plan_sha256']), dtype_note=np.asarray('float32 copies of float64 StateV2 values'))
    return dict(chunk=task['chunk_id'], rows=len(out['match']), check_rows=out['check_rows'], check_equal=out['check_equal'])


def cmd_extract(args, st):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    L = D.Layout(False)
    man = L.manifest('MAIN')
    names = man['names']
    items_by_role = {}
    for sp, _op, cm in D.chunk_paths(L, 'MAIN'):
        with np.load(sp, allow_pickle=False) as z:
            role = dict(zip(z['m_match'].tolist(), z['m_sub_role'].tolist()))
            vm = z['v_match'].tolist()
            if not vm:
                continue
            rr = np.asarray(['TRAIN' if role[m].startswith('fold') else role[m] for m in vm])
            keep = np.isin(rr, ROLES)
            if not keep.any():
                continue
            qs, bf = z['v_query_ms'][keep], z['v_is_bucket_sample'][keep]
            X = z['v_X'][keep & (z['v_is_bucket_sample'] == 1)]
            mm = np.asarray(vm)[keep]
            mb = np.asarray(vm)[keep & (z['v_is_bucket_sample'] == 1)]
            for m in dict.fromkeys(mm.tolist()):
                sel = mm == m
                r = rr[keep][sel][0]
                items_by_role.setdefault(r, []).append((m, qs[sel].tolist(), bf[sel].tolist(), qs[sel][bf[sel] == 1].tolist(), X[mb == m]))
    (tdir() / 'history').mkdir(parents=True, exist_ok=True)
    tasks = []
    for r in ROLES:
        items = items_by_role.get(r, [])
        for c, s0 in enumerate(range(0, len(items), 300)):
            part = items[s0:s0 + 300]
            plan = C.sha256_json(dict(role=r, keys=[(m, q) for m, q, _, _, _ in part], names=names, extract=C.sha256_file(C.OUT / 'extract/MAIN/extraction_manifest.json')))
            path = tdir() / 'history' / f'{r}_{c:04d}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    if str(z['plan_sha256']) == plan:
                        continue
                raise SystemExit(f'stale history chunk {path.name}')
            tasks.append(dict(chunk_id=f'{r}_{c:04d}', items=part, names=names, plan_sha256=plan, path=str(path)))
    total = sum(len(t['items']) for t in tasks)
    done, eq, rows = 0, 0, 0
    st.update('running', 'history_extract', processed=0, total=total, next_step='train')
    os.environ['FC_WORKER'] = '1'
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as ex:
        futs = {ex.submit(history_chunk, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            done += len(futs[fut]['items'])
            eq += r['check_equal']
            rows += r['check_rows']
            st.update('running', 'history_extract', processed=done, total=total, query_state_equal=eq, query_rows=rows,
                      rate_per_s=round(done / max(1e-9, time.time() - t0), 2), next_step='train')
    if eq != rows:
        raise SystemExit(f'history query-position states differ from primary V states: {rows - eq} rows')
    C.write_json(tdir() / 'history_manifest.json', dict(names=names, chunks=sorted(str(p.name) for p in (tdir() / 'history').glob('*.npz')),
                                                        query_rows_checked=rows, query_state_equal_float32=eq))


# ------------------------------------------------------------------ table assembly
def load_role(role, pre=None):
    import train.temporal_history_winprob_v3 as T
    parts = sorted(hdir().glob(f'{role}_*.npz'))
    if os.environ.get('FC_TEMPORAL_MAX_CHUNKS'):
        parts = parts[:int(os.environ['FC_TEMPORAL_MAX_CHUNKS'])]
    tabs = []
    for p in parts:
        with np.load(p, allow_pickle=False) as z:
            tabs.append({k: z[k] for k in ('match', 'pos_ms', 'X', 'support_ms', 'q_match', 'q_ms', 'q_is_bucket')})
    match = np.concatenate([t['match'] for t in tabs])
    pos = np.concatenate([t['pos_ms'] for t in tabs])
    key = {(m, int(p)): i for i, (m, p) in enumerate(zip(match.tolist(), pos.tolist()))}
    support = {}
    for t in tabs:
        for m, s in zip(t['match'].tolist(), t['support_ms'].tolist()):
            support[m] = s
    qm = np.concatenate([t['q_match'] for t in tabs])
    qt = np.concatenate([t['q_ms'] for t in tabs])
    qb = np.concatenate([t['q_is_bucket'] for t in tabs])
    idx = np.zeros((len(qm), NPOS), dtype=np.int64)
    mask = np.zeros((len(qm), NPOS), dtype=bool)
    for i, (m, q) in enumerate(zip(qm.tolist(), qt.tolist())):
        for k in range(NPOS):
            h = q - (NPOS - 1 - k) * STEP
            if h >= support[m]:
                idx[i, k] = key[(m, h)]
                mask[i, k] = True
    if not mask[:, -1].all():
        raise ValueError('query position unsupported')
    Xall = np.concatenate([t['X'] for t in tabs])
    return dict(match=match, X=Xall, q_match=qm, q_ms=qt, q_bucket=qb, index=idx, mask=mask)


def encode_float32(pre, X):
    Z = ((X[:, pre.numeric_index].astype(np.float64) - pre.numeric_mean) / pre.numeric_scale).astype(np.float32)
    ids = X[:, pre.champion_index].astype(np.int64)
    Cm = np.full(ids.shape, -1, dtype=np.int64)
    for j, vocab in enumerate(pre.champion_vocab):
        p = np.searchsorted(vocab, ids[:, j])
        f = (p < len(vocab)) & (vocab[np.minimum(p, len(vocab) - 1)] == ids[:, j])
        Cm[f, j] = pre.champion_offsets[j] + p[f]
    return Z, Cm


def fit_preprocessor(names, X32):
    import train.temporal_history_winprob_v3 as T
    num = [i for i, n in enumerate(names) if n not in T.EXCLUDED_PREDICTORS and n not in T.CHAMPION_COLUMNS]
    mean = np.zeros(len(num))
    sq = np.zeros(len(num))
    n = 0
    for s in range(0, len(X32), 200000):
        B = X32[s:s + 200000][:, num].astype(np.float64)
        mean += B.sum(axis=0)
        sq += (B ** 2).sum(axis=0)
        n += len(B)
    mean /= n
    std = np.sqrt(np.maximum(sq / n - mean ** 2, 0))
    scale = np.where(std > 1e-12, std, 1.)
    vocab = [np.unique(X32[:, names.index(c)].astype(np.int64)) for c in T.CHAMPION_COLUMNS]
    return T.HistoryPreprocessor(names, mean, scale, vocab)


def train_weighted(Z, Cm, idx, mask, y, w, width, seed, on_epoch):
    """P2 _train_torch with per-row loss weights (equal total weight per match); float32 tables without extra copies."""
    import torch
    import torch.nn.functional as F
    import train.temporal_history_winprob_v3 as T
    torch.set_num_threads(int(os.environ.get('FC_TORCH_THREADS', 4)))
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    D_ = Z.shape[1]
    pad = len(Z) - 1                    # caller appended a zero pad row
    Zt, Ct = torch.from_numpy(Z), torch.from_numpy(Cm)
    it = torch.from_numpy(np.where(mask, idx, pad).astype(np.int64))
    Mt = torch.from_numpy(mask)
    yt = torch.from_numpy(y.astype(np.float32))
    wt = torch.from_numpy(w.astype(np.float32))
    limit = math.sqrt(6. / (width + UNITS))
    kernel = (torch.rand((UNITS, width), generator=gen) * 2 - 1) * limit
    q, r = torch.linalg.qr(torch.randn((UNITS, UNITS), generator=gen))
    recurrent = q * torch.sign(torch.diagonal(r))[None, :]
    out_k = (torch.rand((UNITS,), generator=gen) * 2 - 1) * math.sqrt(6. / (UNITS + 1))
    P = [kernel, recurrent, torch.zeros(UNITS), out_k, torch.zeros(())]
    for t in P:
        t.requires_grad_(True)
    opt = torch.optim.RMSprop(P, lr=LR, alpha=RHO, eps=EPS)
    keep = 1. - DROPOUT
    N = len(y)
    hist = []
    for epoch in range(EPOCHS):
        t0 = time.time()
        perm = rng.permutation(N)
        tot, seen = 0., 0.
        for s in range(0, N, BATCH):
            b = torch.from_numpy(perm[s:s + BATCH])
            ib = it[b]
            logit = T.torch_forward(torch, P, Zt[ib], Ct[ib], Mt[b], keep=keep, gen=gen)
            wb = wt[b]
            loss = (F.binary_cross_entropy_with_logits(logit, yt[b], reduction='none') * wb).sum() / wb.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * float(wb.sum())
            seen += float(wb.sum())
        rec = dict(epoch=epoch + 1, train_weighted_bce_with_dropout=tot / seen, seconds=round(time.time() - t0, 2),
                   finite=bool(all(torch.isfinite(t).all() for t in P)))
        hist.append(rec)
        on_epoch(rec)
        if not rec['finite']:
            raise FloatingPointError('non-finite parameters')
    params = {nm: t.detach().to(torch.float64).numpy().copy() for nm, t in zip(T.PARAM_NAMES, P)}
    return params, hist


def rnn_predict(params, Z, Cm, idx, mask, bs=20000):
    import train.temporal_history_winprob_v3 as T
    out = []
    for s in range(0, len(idx), bs):
        ii, mm = idx[s:s + bs], mask[s:s + bs]
        safe = np.where(mm, ii, 0)
        Xn = Z[safe].astype(np.float64) * mm[..., None]
        Cc = np.where(mm[..., None], Cm[safe], -1)
        out.append(T.simplernn_forward(params, Xn, Cc, mm)[0])
    return np.concatenate(out)


def cmd_train(args, st):
    import train.temporal_history_winprob_v3 as T
    L = D.Layout(False)
    names = C.read_json(C.OUT / 'temporal_comparator' / 'history_manifest.json')['names'] if not os.environ.get('FC_TEMPORAL_OUT') else C.read_json(C.OUT / 'extract/MAIN/extraction_manifest.json')['names']
    tr = load_role('TRAIN')
    keepb = tr['q_bucket'] == 1
    pre = fit_preprocessor(names, tr['X'])
    Z, Cm = encode_float32(pre, tr['X'])
    tr['X'] = None
    Z = np.vstack([Z, np.zeros((1, Z.shape[1]), dtype=np.float32)])
    Cm = np.vstack([Cm, np.full((1, Cm.shape[1]), -1, dtype=np.int64)])
    W = D.load_outcomes(L, 'MAIN', [f'fold{k}' for k in range(C.N_FOLDS)] + ['V_CAL', 'V_SELECT'], purpose='temporal comparator fit/calibration/selection')
    y = np.asarray([W[m][0] for m in tr['q_match'][keepb].tolist()])
    w = C.weights(tr['q_match'][keepb])
    (tdir() / 'models').mkdir(parents=True, exist_ok=True)
    params = {}
    for seed in SEEDS:
        ck = tdir() / 'models' / f'seed_{seed}.npz'
        if ck.exists():
            with np.load(ck, allow_pickle=False) as z:
                params[seed] = {k: z[k].copy() for k in T.PARAM_NAMES}
            continue
        started = time.time()
        pr, hist = train_weighted(Z, Cm, tr['index'][keepb], tr['mask'][keepb], y, w, pre.input_width, seed,
                                  lambda rec, seed=seed: st.update('running', f'train_seed{seed}', processed=rec['epoch'], total=EPOCHS,
                                                                    bce=round(rec['train_weighted_bce_with_dropout'], 5), epoch_seconds=rec['seconds'],
                                                                    next_step='next seed / calibration'))
        C.save_npz(ck, **pr)
        C.write_json(tdir() / 'models' / f'seed_{seed}_log.json', dict(seed=seed, epochs=hist, seconds=round(time.time() - started, 1)))
        params[seed] = pr
    np.savez(tdir() / 'models' / 'preprocessing.npz', **pre.arrays())
    Z = Cm = None
    res = {}
    preds = {}
    for role in ('V_CAL', 'V_SELECT'):
        R = load_role(role)
        Zr, Cr = encode_float32(pre, R['X'])
        b = R['q_bucket'] == 1
        preds[role] = dict(match=R['q_match'][b], y=np.asarray([W[m][0] for m in R['q_match'][b].tolist()]),
                           raw={s: rnn_predict(params[s], Zr, Cr, R['index'][b], R['mask'][b]) for s in SEEDS})
    members = {}
    for s in SEEDS:
        cal = preds['V_CAL']
        sig = C.PositiveSlopeSigmoid().fit(cal['raw'][s], cal['y'], C.weights(cal['match']))
        se = preds['V_SELECT']
        cand = {'raw': se['raw'][s], 'sigmoid_pos': sig.predict(se['raw'][s])}
        sc = {k: C.score(se['y'], p, C.weights(se['match'])) for k, p in cand.items()}
        ch = min(cand, key=lambda k: (sc[k]['logloss'], sc[k]['brier'], k))
        members[s] = dict(chosen=ch, select_scores=sc, sigmoid=sig.describe())
    p_ens = np.mean([members_apply(members[s], preds['V_SELECT']['raw'][s]) for s in SEEDS], axis=0)
    se = preds['V_SELECT']
    res = dict(members=members, ensemble_v_select=C.evaluate(se['y'], p_ens, se['match']),
               preprocessing_fit='TRAIN history states only (float32 copies; float64 mean/std)', loss_weights='equal total per match',
               selection_written_before_test=True, written_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    res['model_sha256'] = {f'seed_{s}.npz': C.sha256_file(tdir() / 'models' / f'seed_{s}.npz') for s in SEEDS}
    res['model_sha256']['preprocessing.npz'] = C.sha256_file(tdir() / 'models' / 'preprocessing.npz')
    res['data'] = dict(train_queries=int(keepb.sum()), train_matches=int(len(np.unique(tr['q_match'][keepb]))),
                       history_chunks_max=os.environ.get('FC_TEMPORAL_MAX_CHUNKS'), epochs=EPOCHS, torch_threads=int(os.environ.get('FC_TORCH_THREADS', 4)))
    C.write_json(tdir() / 'selection_temporal.json', res)
    st.update('running', 'temporal_selected', ensemble_v_select_logloss=res['ensemble_v_select']['logloss'], next_step='evaluate on TEST')


def members_apply(m, p):
    if m['chosen'] == 'raw':
        return p
    s = m['sigmoid']
    return 1 / (1 + np.exp(-(s['intercept_a'] + s['slope_b'] * C.clipped_logit(p))))


def cmd_evaluate(args, st):
    import train.temporal_history_winprob_v3 as T
    L = D.Layout(False)
    if not L.frozen_manifest.exists():
        raise SystemExit('primary frozen manifest required')
    sel = C.read_json(tdir() / 'selection_temporal.json')
    changed = {k: v for k, v in sel['model_sha256'].items() if C.sha256_file(tdir() / 'models' / k) != v}
    if changed:
        raise SystemExit(f'temporal model files changed after selection: {changed}')
    names = C.read_json(tdir() / 'history_manifest.json')['names']
    with np.load(tdir() / 'models' / 'preprocessing.npz', allow_pickle=False) as z:
        pre = T.HistoryPreprocessor.from_arrays(z)
    params = {}
    for s in SEEDS:
        with np.load(tdir() / 'models' / f'seed_{s}.npz', allow_pickle=False) as z:
            params[s] = {k: z[k].copy() for k in T.PARAM_NAMES}
    R = load_role('TEST')
    Zr, Cr = encode_float32(pre, R['X'])
    R['X'] = None
    W = D.load_outcomes(L, 'MAIN', ['TEST'], purpose='temporal comparator TEST evaluation (after both selections)')
    y = np.asarray([W[m][0] for m in R['q_match'].tolist()])
    p = np.mean([members_apply(sel['members'][str(s)] if str(s) in sel['members'] else sel['members'][s],
                               rnn_predict(params[s], Zr, Cr, R['index'], R['mask'])) for s in SEEDS], axis=0)
    with np.load(L.base / 'eval' / 'predictions' / 'v_MAIN_TEST.npz', allow_pickle=False) as z:
        key = {(m, int(q)): i for i, (m, q) in enumerate(zip(z['match'].tolist(), z['query_ms'].tolist()))}
        chosen = str(z['chosen'])
        pv = z[f'p_{chosen}']
    j = np.asarray([key[(m, int(q))] for m, q in zip(R['q_match'].tolist(), R['q_ms'].tolist())])
    prim = pv[j]
    b = R['q_bucket'] == 1
    from fc20260915_evaluate import bootstrap, v_cells
    out = dict(bucket=dict(temporal=v_cells(y[b], p[b], R['q_match'][b], R['q_ms'][b]), primary=v_cells(y[b], prim[b], R['q_match'][b], R['q_ms'][b]),
                           bootstrap=bootstrap(y[b], {'temporal_rnn': p[b], 'primary_v': prim[b]}, R['q_match'][b], [('temporal_rnn', 'primary_v')])),
               full_trajectory=dict(temporal=v_cells(y, p, R['q_match'], R['q_ms']), primary=v_cells(y, prim, R['q_match'], R['q_ms'])),
               agreement=dict(mean_abs_diff=float(np.mean(np.abs(p - prim))), pearson=float(np.corrcoef(p, prim)[0, 1])),
               role='comparator only; never replaces the primary labeler')
    C.save_npz(tdir() / 'test_predictions.npz', match=R['q_match'], query_ms=R['q_ms'], is_bucket=R['q_bucket'], p_temporal=p, p_primary=prim)
    C.write_json(tdir() / 'results_temporal.json', out)
    st.update('complete', 'temporal_evaluate', temporal_logloss=out['bucket']['temporal']['overall']['logloss'],
              primary_logloss=out['bucket']['primary']['overall']['logloss'], next_step='report')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('extract', 'train', 'evaluate'))
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    st = C.Status(tdir() if os.environ.get('FC_TEMPORAL_OUT') else C.OUT, f'temporal_{args.cmd}')
    try:
        {'extract': cmd_extract, 'train': cmd_train, 'evaluate': cmd_evaluate}[args.cmd](args, st)
        if args.cmd != 'evaluate':
            st.update('complete', f'temporal_{args.cmd}', next_step={'extract': 'train', 'train': 'evaluate'}[args.cmd])
        return 0
    except SystemExit as exc:
        st.update('failed', args.cmd, error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', args.cmd, error=repr(exc), next_step='inspect, fix, resume')
        return 3


if __name__ == '__main__':
    sys.exit(main())
