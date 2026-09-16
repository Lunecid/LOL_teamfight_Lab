"""Stage B: rebuild the changed matches of one set under the dev definition with the PARENT extraction function
(fc20260915_extract.extract_chunk), label them with the frozen V adapters, build the pre-only q inputs and the cohort from the
detector's raw cluster counts, and assemble the dev population = parent rows of unchanged matches (copied) + rebuilt rows.
--fixture N: rebuild the first N TRAIN matches with the PARENT exposures under the frozen constants and compare with the parent
rows bitwise (writes rebuild/fixture_check.json; no population file). Sealed sets (TEST / external) require this run's freeze."""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
os.environ.setdefault('MKL_CBWR', 'AVX2,STRICT')

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.dont_write_bytecode = True
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
import dd20260916_common as DD  # noqa: E402
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

E_SCALAR = ('s', 'L', 'q_pre', 'pre_ok', 'pre_snapshot', 'pre_reason')
E_H = ('endpoint', 'valid', 'post_snapshot')


def _extraction_env(definition):
    """Environment for the parent extraction function: its own preset/overrides plus the definition constants, runtime under this root."""
    DD.set_definition_env(definition, C.CACHE_MAIN)
    import fc20260915_extract as X  # sets LOL_CFG_PRESET/OVERRIDES and LOL_OUTPUT_ROOT (parent runtime) at import
    os.environ['LOL_OUTPUT_ROOT'] = str(DD.OUT / 'runtime')
    os.environ['LOL_CFG_OVERRIDES'] = json.dumps({**json.loads(os.environ['LOL_CFG_OVERRIDES']), **DD.DEFINITIONS[definition]})
    return X


def dd_extract_chunk(task):
    X = _extraction_env(task['definition'])
    return X.extract_chunk(task)


def run_extraction(items, cache_dir, out_dir, set_id, definition, workers, st, plan_tag):
    X = _extraction_env(definition)
    names = X.expected_state_names()
    (out_dir / 'states').mkdir(parents=True, exist_ok=True)
    (out_dir / 'outcomes_SEALED').mkdir(parents=True, exist_ok=True)
    tasks, meta = [], []
    for c, s0 in enumerate(range(0, len(items), 200)):
        part = items[s0:s0 + 200]
        plan_sha = C.sha256_json(dict(tag=plan_tag, definition=definition, cache_dir=str(cache_dir), names=names, items=part, set=set_id))
        path, opath = out_dir / 'states' / f'chunk_{c:05d}.npz', out_dir / 'outcomes_SEALED' / f'chunk_{c:05d}.npz'
        meta.append(dict(chunk=c, matches=len(part), plan_sha256=plan_sha, path=str(path)))
        if path.exists() and opath.exists():
            with np.load(path, allow_pickle=False) as z:
                if str(z['plan_sha256']) != plan_sha:
                    raise SystemExit(f'stale checkpoint {path.name}')
            continue
        tasks.append(dict(chunk_id=c, set_id=set_id, cache_dir=str(cache_dir), items=part, names=names, plan_sha256=plan_sha, path=str(path), outcome_path=str(opath), definition=definition))
    t0 = time.time()
    if tasks:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        os.environ['FC_WORKER'] = '1'
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as ex:
            futs = {ex.submit(dd_extract_chunk, t): t for t in tasks}
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                if r.get('errors'):
                    raise SystemExit(f'extraction worker errors: {list(r["errors"].items())[:3]}')
                done += 1
                st.update('running', 'extract', processed=done, total=len(tasks), seconds=round(time.time() - t0, 1), next_step='labels')
    return names, meta


def load_rows(meta, names):
    acc = {}

    def add(k, v):
        acc.setdefault(k, []).append(v)
    for cm in meta:
        with np.load(cm['path'], allow_pickle=False) as z:
            if [str(x) for x in z['names']] != names:
                raise SystemExit('state names differ between chunks')
            role_of = dict(zip(z['m_match'].tolist(), z['m_sub_role'].tolist()))
            em = z['e_match']
            add('match', em.astype(str))
            add('sub_role', np.asarray([role_of[m] for m in em.tolist()], dtype='U16') if len(em) else np.zeros(0, dtype='U16'))
            for k in E_SCALAR:
                add(k, z['e_' + k])
            add('X_pre', z['e_X_pre'])
            for h in DD.HS:
                for k in E_H:
                    add(f'{k}_h{h}', z[f'e_{k}_h{h}'])
                add(f'X_post_h{h}', z[f'e_X_post_h{h}'])
    return {k: np.concatenate(v) for k, v in acc.items()}


def adapter_for(sub_role):
    return f'oof_{sub_role}' if sub_role.startswith('fold') else 'final'


def label_rows(E, names, vman, st):
    n = len(E['match'])
    adapter_id = np.empty(n, dtype='U16')
    adapter_sha = np.empty(n, dtype='U64')
    p_pre = np.full(n, np.nan)
    p_post = {h: np.full(n, np.nan) for h in DD.HS}
    for aid in sorted(set(adapter_for(sr) for sr in np.unique(E['sub_role']).tolist())):
        path, sha = (vman['final']['path'], vman['final']['sha256']) if aid == 'final' else (vman['oof_paths'][aid.replace('oof_', '')], vman['oof_sha256'][aid.replace('oof_', '')])
        ad = C.load_v_adapter(DD.FC / path, sha)
        rows = np.asarray([adapter_for(sr) == aid for sr in E['sub_role'].tolist()])
        adapter_id[rows] = aid
        adapter_sha[rows] = sha
        ok = rows & (E['pre_ok'] == 1)
        if ok.any():
            p_pre[ok] = ad.predict_matrix(E['X_pre'][ok], names, C.STATE_VERSION)
        for h in DD.HS:
            v = rows & (E[f'valid_h{h}'] == 1)
            if v.any():
                p_post[h][v] = ad.predict_matrix(E[f'X_post_h{h}'][v], names, C.STATE_VERSION)
        st.log(f'labels: adapter {aid} scored {int(rows.sum())} rows')
    keep = [i for i, nm in enumerate(names) if nm != 'snapshot_age_s']
    input_names = [names[i] for i in keep] + [C.P_PRE]
    ok = E['pre_ok'] == 1
    Xin = np.full((n, len(input_names)), np.nan)
    Xin[ok, :-1] = E['X_pre'][ok][:, keep]
    Xin[ok, -1] = p_pre[ok]
    L = dict(match=E['match'], s=E['s'], sub_role=E['sub_role'], adapter_id=adapter_id, adapter_sha256=adapter_sha, p_pre=p_pre, pre_ok=E['pre_ok'])
    for h in DD.HS:
        v = E[f'valid_h{h}'] == 1
        delta = np.where(v, p_post[h] - p_pre, np.nan)
        L[f'valid_h{h}'] = E[f'valid_h{h}'].astype(np.int8)
        L[f'Y_h{h}'] = np.where(v, (delta > 0).astype(np.int8), -1).astype(np.int8)
        L[f'p_post_h{h}'] = p_post[h]
        L[f'endpoint_h{h}'] = E[f'endpoint_h{h}']
        if v.any() and not np.isfinite(delta[v]).all():
            raise SystemExit(f'non-finite delta at h{h}')
    F = dict(X_input=Xin, input_names=np.asarray(input_names), match=E['match'], s_ms=E['s'], sub_role=E['sub_role'], pre_ok=E['pre_ok'])
    return L, F


def items_for(set_name, exposures_by_match, only_matches=None):
    rows = DD.set_matches(set_name)
    out = []
    for mid, role, sub, patch, cache_dir in rows:
        if only_matches is not None and mid not in only_matches:
            continue
        ex = [dict(s=int(r['s']), L=int(r['L']), next_start=int(r['next_start']), end=int(r['end']), same_match_overlap=int(r['same_match_overlap']),
                   end_observed=int(r['end_observed']), patch=r['patch']) for r in exposures_by_match.get(mid, [])]
        out.append(dict(match_id=mid, role=role, sub_role=sub, patch_expected=patch, want_grid=False, exposures=ex))
    cache_dirs = sorted(set(r[4] for r in rows))
    return out, cache_dirs[0]


def fixture_check(n, out_dir, workers, st, access_base):
    """Rebuild the first n TRAIN matches with the PARENT exposures under the frozen constants and compare with the parent rows bitwise."""
    vman = C.read_json(DD.FC / 'v_models_manifest.json')
    schema = C.read_json(DD.FC / 'q_pre_only_schema.json')
    par, _ = DD.parent_exposures('MAIN_TRAIN')
    mids = [r[0] for r in DD.set_matches('MAIN_TRAIN')][:n]
    byrow = {m: [dict(zip(DD.EXPO_FIELDS, t)) for t in par.get(m, [])] for m in mids}
    items, cache_dir = items_for('MAIN_TRAIN', byrow, set(mids))
    names, meta = run_extraction(items, cache_dir, Path(out_dir), 'MAIN', 'frozen', workers, st, 'fixture')
    E = load_rows(meta, names)
    L, F = label_rows(E, names, vman, st)
    PF, PL, PC = DD.load_parent_set('MAIN_TRAIN', access_base, f'fixture comparison ({n} TRAIN matches)')
    pk = {(m, int(s)): i for i, (m, s) in enumerate(zip(PF['match'].astype(str).tolist(), PF['s_ms'].tolist()))}
    ix = np.asarray([pk[(m, int(s))] for m, s in zip(F['match'].tolist(), F['s_ms'].tolist())], dtype=np.int64)
    expected = int(np.isin(PF['match'].astype(str), mids).sum())
    chk = dict(rows=int(len(ix)), rows_expected=expected, keys_all_found=len(ix) == expected, input_names_equal=[str(x) for x in F['input_names']] == schema['input_names_all'],
               X_input_equal=bool(np.array_equal(F['X_input'], PF['X_input'][ix], equal_nan=True)), pre_ok_equal=bool(np.array_equal(F['pre_ok'], PF['pre_ok'][ix])),
               p_pre_equal=bool(np.array_equal(L['p_pre'], PL['p_pre'][ix], equal_nan=True)), sub_role_equal=bool(np.array_equal(L['sub_role'].astype(str), PL['sub_role'][ix].astype(str))),
               adapter_equal=bool(np.array_equal(L['adapter_id'].astype(str), PL['adapter_id'][ix].astype(str)) and np.array_equal(L['adapter_sha256'].astype(str), PL['adapter_sha256'][ix].astype(str))))
    for h in DD.HS:
        chk[f'valid_h{h}_equal'] = bool(np.array_equal(L[f'valid_h{h}'].astype(int), PL[f'valid_h{h}'][ix].astype(int)))
        chk[f'Y_h{h}_equal'] = bool(np.array_equal(L[f'Y_h{h}'].astype(int), PL[f'Y_h{h}'][ix].astype(int)))
        chk[f'p_post_h{h}_equal'] = bool(np.array_equal(L[f'p_post_h{h}'], PL[f'p_post_h{h}'][ix], equal_nan=True))
        chk[f'endpoint_h{h}_equal'] = bool(np.array_equal(L[f'endpoint_h{h}'], PL[f'endpoint_h{h}'][ix]))
    chk['all_equal'] = all(v for k, v in chk.items() if k.endswith('equal') or k == 'keys_all_found')
    return dict(matches=len(mids), definition='frozen', exposures='parent', checks=chk, names_sha256=C.sha256_json(names), written_at=time.strftime('%Y-%m-%d %H:%M:%S'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default=None)
    ap.add_argument('--fixture', type=int, default=0, help='rebuild N TRAIN matches with the parent exposures (frozen) and compare with the parent rows')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    DD.log_command()
    st = DD.Status(f'rebuild_{args.set}' if not args.fixture else f'rebuild_fixture{args.fixture}')
    try:
        if not (DD.OUT / 'protocol.json').exists():
            raise SystemExit('protocol.json must precede the rebuild')
        vman = C.read_json(DD.FC / 'v_models_manifest.json')
        schema = C.read_json(DD.FC / 'q_pre_only_schema.json')
        if args.fixture:
            res = fixture_check(args.fixture, DD.OUT / 'rebuild' / f'fixture{args.fixture}', args.workers, st, DD.OUT)
            C.write_json(DD.OUT / 'rebuild' / 'fixture_check.json', res)
            chk = res['checks']
            st.update('complete' if chk['all_equal'] else 'failed', 'fixture', checks=chk, next_step='full rebuild')
            return 0 if chk['all_equal'] else 1
        set_name = args.set
        sealed = DD.is_sealed(set_name)
        if sealed and not DD.frozen_path(DD.OUT).exists():
            raise SystemExit(f'{set_name} is sealed until this run freezes')
        fx = DD.OUT / 'rebuild' / 'fixture_check.json'
        if not fx.exists() or not C.read_json(fx)['checks']['all_equal']:
            raise SystemExit('fixture check must pass before any population rebuild')
        if (DD.OUT / 'labels' / f'{set_name}_labels.npz').exists():
            st.update('complete', 'resume_skip', note='population exists', next_step='next set')
            return 0
        ch = C.read_json(DD.OUT / 'changed' / f'{set_name}.json')
        changed = list(ch['matches'])
        st.update('running', 'items', changed_matches=len(changed), next_step='extract')
        items, cache_dir = items_for(set_name, ch['rows'], set(changed))
        set_id = 'MAIN' if set_name.startswith('MAIN') else set_name.replace('EXT_', '')
        names, meta = run_extraction(items, cache_dir, DD.OUT / 'extract' / set_name, set_id, 'dev', args.workers, st, f'rebuild_{set_name}') if items else (None, [])
        E = load_rows(meta, names) if meta else None
        if E is not None:
            L, F = label_rows(E, names, vman, st)
            if [str(x) for x in F['input_names']] != schema['input_names_all']:
                raise SystemExit('rebuilt input names differ from the parent schema')
            counts = {(m, int(r['s'])): (int(r['cluster_blue']), int(r['cluster_red'])) for m, rows in ch['rows'].items() for r in rows}
            cb = np.asarray([counts[(m, int(s))][0] for m, s in zip(L['match'].tolist(), L['s'].tolist())])
            cr = np.asarray([counts[(m, int(s))][1] for m, s in zip(L['match'].tolist(), L['s'].tolist())])
            n_min, known, cohort, fine = K.scale_classes(cb, cr, np.zeros(len(cb), dtype=bool))
            RC = dict(match=L['match'], s=L['s'], cohort=cohort.astype(np.int8), valid_h90=L['valid_h90'], scale_known=known.astype(np.int8), n_min=n_min.astype(np.int64))
            n_rebuilt = int(len(L['match']))
        else:
            n_rebuilt = 0
        PF, PL, PC = DD.load_parent_set(set_name, DD.OUT, f'assemble dev population {set_name} (parent rows of unchanged matches)')
        keep = ~np.isin(PF['match'].astype(str), changed)
        pm = PF['match'].astype(str)
        exposures_changed = sum(len(v) for v in ch['rows'].values())
        if E is not None and n_rebuilt != exposures_changed:
            raise SystemExit(f'rebuilt rows {n_rebuilt} != dev exposure rows of changed matches {exposures_changed}')

        def cat(a, b):
            return np.concatenate([a, b]) if b is not None else a
        src = np.concatenate([np.full(int(keep.sum()), 'parent'), np.full(n_rebuilt, 'rebuilt')]).astype('U8')
        feat = dict(X_input=cat(PF['X_input'][keep], F['X_input'] if E is not None else None), input_names=PF['input_names'], match=cat(pm[keep], F['match'] if E is not None else None),
                    s_ms=cat(PF['s_ms'][keep], F['s_ms'] if E is not None else None), sub_role=cat(PF['sub_role'][keep].astype('U16'), F['sub_role'].astype('U16') if E is not None else None),
                    pre_ok=cat(PF['pre_ok'][keep], F['pre_ok'] if E is not None else None), row_source=src)
        lab = dict(match=feat['match'], s=feat['s_ms'], sub_role=feat['sub_role'], adapter_id=cat(PL['adapter_id'][keep].astype('U16'), L['adapter_id'] if E is not None else None),
                   adapter_sha256=cat(PL['adapter_sha256'][keep].astype('U64'), L['adapter_sha256'] if E is not None else None), p_pre=cat(PL['p_pre'][keep], L['p_pre'] if E is not None else None),
                   row_source=src)
        for h in DD.HS:
            for k in ('valid', 'Y', 'p_post', 'endpoint'):
                key = f'{k}_h{h}'
                if key in PL:
                    lab[key] = cat(PL[key][keep], L[key] if E is not None else None)
        coh = dict(match=feat['match'], s=feat['s_ms'], cohort=cat(PC['cohort'][keep].astype(np.int8), RC['cohort'] if E is not None else None),
                   valid_h90=cat(PC['valid_h90'][keep].astype(np.int8), RC['valid_h90'] if E is not None else None), scale_known=cat(PC['scale_known'][keep].astype(np.int8), RC['scale_known'] if E is not None else None),
                   n_min=cat(PC['n_min'][keep].astype(np.int64), RC['n_min'] if E is not None else None), row_source=src)
        keys = list(zip(feat['match'].tolist(), feat['s_ms'].tolist()))
        if len(set(keys)) != len(keys):
            raise SystemExit('duplicate (match, s) keys in the dev population')
        fs = C.save_npz(DD.OUT / 'labels' / f'{set_name}_features_pre_only.npz', **feat)
        ls = C.save_npz(DD.OUT / 'labels' / f'{set_name}_labels.npz', **lab)
        cs = C.save_npz(DD.OUT / 'cohorts' / f'{set_name}_cohort.npz', **coh)
        v = lab['valid_h90'] == 1
        summary = dict(set=set_name, sealed=sealed, changed_matches=len(changed), parent_rows=int(len(pm)), parent_rows_copied=int(keep.sum()), parent_rows_dropped=int((~keep).sum()),
                       rebuilt_rows=n_rebuilt, dev_rows=int(len(feat['match'])), chunks=len(meta), names_sha256=C.sha256_json(names) if E is not None else None,
                       valid_h90=dict(parent_side=int((PL['valid_h90'] == 1).sum()), dev=int(v.sum()), by_cohort_dev={c: int((v & (coh['cohort'] == DD.COHORT_CODE[c])).sum()) for c in DD.COHORTS},
                                      by_cohort_parent={c: int(((PL['valid_h90'] == 1) & (PC['cohort'] == DD.COHORT_CODE[c])).sum()) for c in DD.COHORTS},
                                      rebuilt_valid={c: int((v & (src == 'rebuilt') & (coh['cohort'] == DD.COHORT_CODE[c])).sum()) for c in DD.COHORTS}),
                       cohort_unknown_rebuilt=int(((coh['cohort'] == -1) & (src == 'rebuilt')).sum()), files=dict(features=fs, labels=ls, cohorts=cs), written_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        C.write_json(DD.OUT / 'rebuild' / f'{set_name}.json', summary)
        st.update('complete', 'assembled', **{k: v for k, v in summary.items() if k in ('changed_matches', 'parent_rows_copied', 'rebuilt_rows', 'dev_rows')}, next_step='next set / fit')
        return 0
    except SystemExit as exc:
        DD.log_failure(st.group, exc)
        st.update('failed', 'rebuild', error=str(exc)[:1000], next_step='inspect')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        DD.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'rebuild', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
