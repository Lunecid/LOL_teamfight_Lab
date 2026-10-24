"""Stage T: frozen evaluation on MAIN TEST 15.16 and each external set separately, T and N h90-valid rows.
Models: the 16 frozen arm winners (+ parent iq winners joined by exact key, metrics only). Cells and paired bootstrap as in iq / Track A.
--smoke: pseudo Q_SELECT rows of the TRAIN-only smoke (no parent reference)."""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('MKL_CBWR', 'AVX2,STRICT')

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cc20260916_common as CC  # noqa: E402
import fc20260915_common as C  # noqa: E402

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402


def verify(base, fz):
    return {p: C.sha256_file((CC.OUT if p == 'protocol.json' else base) / p) == sha for p, sha in fz['frozen_files_sha256'].items()}


def parent_columns(set_name, coh, g, s, y, fz, base):
    path = CC.parent_pred_path(set_name, coh)
    CC.log_access(base, f'parent iq winner predictions {set_name} {coh}', f'parent {path.name}', 'parent frozen predictions incl. y', True)
    with np.load(path, allow_pickle=False) as z:
        A = {k: z[k] for k in ('match', 's_ms', 'y', 'named__lgbm_winner', 'named__logit_winner')}
    ag, a_s = A['match'].astype(str), A['s_ms']
    chk = dict(file_sha256=C.sha256_file(path), file_sha256_equals_frozen=C.sha256_file(path) == fz['parent_references']['parent_predictions_sha256'][f'{set_name}_{coh}'],
               rows_equal=len(ag) == len(g), parent_keys_unique=len(set(zip(ag.tolist(), a_s.tolist()))) == len(ag))
    try:
        ix, how = CC.key_join(g, s, ag, a_s)
    except ValueError as exc:
        raise SystemExit(f'parent join failed for {set_name} {coh}: {exc}')
    chk['join'] = f'{how} on (match, s_ms)'
    chk['every_row_joined_once'] = bool(len(np.unique(ix)) == len(g) == len(ag))
    chk['y_equal_parent_labels'] = bool(np.array_equal(A['y'][ix].astype(int), y.astype(int)))
    if not (chk['file_sha256_equals_frozen'] and chk['parent_keys_unique'] and chk['every_row_joined_once'] and chk['y_equal_parent_labels']):
        raise SystemExit(f'parent contract failed {set_name} {coh}: {chk}')
    cols = {'parent_lgbm_winner': A['named__lgbm_winner'][ix].astype(np.float64), 'parent_logit_winner': A['named__logit_winner'][ix].astype(np.float64)}
    return cols, chk


def evaluate_rows(set_name, coh, g, s, y, X, names, versions, bundles, fz, base, smoke, st, ridge, vocab, tab):
    rows = len(y)
    ridge_ix = [names.index(n) for n in ridge]
    checks = dict(rows=rows, keys_unique=len(set(zip(g.tolist(), s.tolist()))) == rows, nonfinite_inputs=int((~np.isfinite(X[:, ridge_ix])).sum()),
                  patch_versions={str(v): int(n) for v, n in zip(*np.unique(versions, return_counts=True))})
    if not checks['keys_unique'] or checks['nonfinite_inputs']:
        raise SystemExit(f'{set_name} {coh}: key/finite contract failed {checks}')
    p_pre = X[:, names.index('p_pre_V')]
    t = X[:, names.index('time_minutes')]
    t0 = time.time()
    P, raw, seed, msha = {}, {}, {}, {}
    for arm in CC.ARMS:
        Xa, names_arm = CC.arm_matrix(X, names, arm, ridge, versions, vocab, tab)
        if not np.isfinite(Xa).all():
            raise SystemExit(f'{set_name} {coh} {arm}: non-finite arm inputs')
        msha[arm] = CC.matrix_sha(Xa)
        for fam in CC.FAMILIES:
            b = bundles[coh][(fam, arm)]
            if C.sha256_json(names_arm) != b['input_names_all_sha256']:
                raise SystemExit(f'{fam} {arm}: arm input names differ from the bundle')
            nm = f'{fam}_{arm}'
            if fam in CC.SEEDED:
                seed[nm] = b['base'].seed_raw(Xa)
                raw[nm] = seed[nm].mean(axis=0)
            else:
                raw[nm] = b['base'].raw(Xa)
            for cal in CC.CALS:
                P[f'{nm}:{b["config"]}__{cal}'] = CC.calibrate(b, cal, raw[nm])
        Xa = None
    checks['seconds_predict'] = round(time.time() - t0, 1)
    checks['arm_matrix_sha256'] = msha
    named = {f'{fam}_{arm}': P[f'{fam}_{arm}:{fz["family_winners"][f"{fam}_{arm}_{coh}"]["chosen"]}'] for fam in CC.FAMILIES for arm in CC.ARMS}
    if smoke:
        pcols, pchk = {}, dict(note='SMOKE: no parent reference')
    else:
        pcols, pchk = parent_columns(set_name, coh, g, s, y, fz, base)
    named.update(pcols)
    checks['parent'] = pchk
    checks['parent_vs_base_max_abs_diff_DESCRIPTIVE'] = {f'{p}_minus_{fam}_base': float(np.max(np.abs(pcols[p] - named[f'{fam}_base']))) for p, fam in
                                                         (('parent_lgbm_winner', 'lgbm'), ('parent_logit_winner', 'logit')) if p in pcols}
    cells = CC.cell_masks(p_pre, t)
    tb = [k for k in cells if k.startswith('time_')]
    checks['time_bins_partition_rows'] = bool(np.all(np.sum([cells[k] for k in tb], axis=0) == 1)) if rows else True
    checks['B45_subset_B40'] = bool(np.all(~cells['B45'] | cells['B40']))
    model_names = list(CC.NAMED) + list(pcols)
    res = dict(set=set_name, cohort=coh, rows=rows, matches=int(len(np.unique(g))), checks=checks,
               winners={f'{f}_{a}': fz['family_winners'][f'{f}_{a}_{coh}']['chosen'] for f in CC.FAMILIES for a in CC.ARMS}, overall=fz['overall_winner'][coh]['chosen'],
               cells={k: dict(rows=int(m.sum()), matches=int(len(np.unique(g[m])))) for k, m in cells.items()})
    res['metrics_named'] = {c: {name: CC.cell_metrics(y[m], named[name][m], g[m], bins=True) for name in model_names} for c, m in cells.items()}
    res['metrics_all_candidates_DESCRIPTIVE'] = {c: {cand: CC.cell_metrics(y[cells[c]], P[cand][cells[c]], g[cells[c]], bins=False) for cand in P} for c in ('all', 'B40')}
    res['winner_seed_metrics_DESCRIPTIVE'] = {nm: {c: {str(sd): CC.cell_metrics(y[cells[c]], seed[nm][j][cells[c]], g[cells[c]], bins=False) for j, sd in enumerate(CC.SEEDS['lgbm'])}
                                                   for c in ('all', 'B40')} for nm in seed}
    res['bootstrap'] = {}
    for c in ('all', 'B40', 'B45'):
        m = cells[c]
        nm_ = int(len(np.unique(g[m])))
        classes = int(len(np.unique(y[m]))) if m.any() else 0
        if nm_ >= CC.SPARSE_MATCHES and classes == 2:
            t1 = time.time()
            res['bootstrap'][c] = CC.paired_bootstrap(y[m], {k: named[k][m] for k in CC.BOOT_MODELS}, g[m], CC.CONTRASTS)
            res['bootstrap'][c]['seconds'] = round(time.time() - t1, 1)
        else:
            res['bootstrap'][c] = dict(computed=False, reason=f'matches={nm_} (<{CC.SPARSE_MATCHES}) or label classes={classes}')
    res['time_cells_bootstrap'] = 'CI not computed for time and B40 x time cells (point estimates and reliability only, per specification)'
    arrays = dict(match=g, s_ms=s, y=y, p_pre=p_pre, time_minutes=t, **{f'cell__{k}': v for k, v in cells.items()},
                  **{f'named__{k}': v for k, v in named.items()}, **{f'cand__{k}': v for k, v in P.items()}, **{f'raw__{k}': v for k, v in raw.items()},
                  **{f'seedraw__{k}__{sd}': v[j] for k, v in seed.items() for j, sd in enumerate(CC.SEEDS['lgbm'])})
    res['predictions_file'] = f'eval/predictions/{set_name}_h{CC.H}_{coh}.npz'
    res['predictions_sha256'] = C.save_npz(base / 'eval' / 'predictions' / f'{set_name}_h{CC.H}_{coh}.npz', **arrays)
    a = res['metrics_named']['all']
    st.log(f'{set_name} {coh}: rows={rows} ' + ' '.join(f'{k}={a[k]["brier"]:.6f}' for k in model_names if a[k]['brier'] is not None))
    return res


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    CC.log_command()
    base = CC.SMOKE if args.smoke else CC.OUT
    st = CC.Status('evaluate' + ('_smoke' if args.smoke else ''))
    try:
        if not CC.frozen_path(base).exists():
            raise SystemExit('evaluation requires frozen_manifest.json')
        fz = C.read_json(CC.frozen_path(base))
        if bool(fz['smoke_train_only']) != bool(args.smoke):
            raise SystemExit('frozen manifest mode differs')
        before = verify(base, fz)
        if not all(before.values()):
            raise SystemExit(f'frozen hash mismatch before evaluation: {[k for k, v in before.items() if not v][:5]}')
        C.write_json(base / 'eval' / 'hashes_before.json', dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), results=before))
        proto = C.read_json(CC.OUT / 'protocol.json')
        vocab = list(proto['identity_vocab']['ids'])
        if C.sha256_file(CC.TAG_TABLE) != fz['tag_table_sha256'] or C.sha256_json(vocab) != fz['identity_vocab_sha256']:
            raise SystemExit('tag table or identity vocabulary differ from the freeze')
        tab = CC.load_tag_table()
        schema = C.read_json(CC.FC / 'q_pre_only_schema.json')
        ridge = list(schema['predictor_sets']['ridge'])
        bundles = {coh: {(fam, arm): joblib.load(base / fz['family_winners'][f'{fam}_{arm}_{coh}']['bundle']) for fam in CC.FAMILIES for arm in CC.ARMS} for coh in CC.COHORTS}
        results = {}
        sets = ('SMOKE_PSEUDO_Q_SELECT',) if args.smoke else CC.EVAL_SETS
        for k, set_name in enumerate(sets):
            st.update('running', f'evaluate_{set_name}', processed=k, total=len(sets), next_step='cohorts T then N')
            results[set_name] = {}
            if args.smoke:
                for coh in CC.COHORTS:
                    D = CC.load_trainval(base, smoke=True, cohort=coh)
                    m = D['role'] == 'Q_SELECT'
                    results[set_name][coh] = evaluate_rows(set_name, coh, D['g'][m], D['s_ms'][m], D['y'][m], D['X'][m], D['names'], D['version'][m], bundles, fz, base, True, st, ridge, vocab, tab)
                    D = None
            else:
                F, Lb, Co, ch = CC.load_parent_set(set_name, base, f'frozen evaluation {set_name}')
                names = [str(x) for x in F['input_names']]
                if names != schema['input_names_all']:
                    raise SystemExit('schema drift in evaluation set')
                for coh in CC.COHORTS:
                    m = (Lb['valid_h90'] == 1) & (Co['cohort'] == CC.COHORT_CODE[coh])
                    if int(m.sum()) != CC.EXPECTED_COUNTS[set_name][coh]:
                        raise SystemExit(f'count mismatch {set_name} {coh}: {int(m.sum())} vs {CC.EXPECTED_COUNTS[set_name][coh]}')
                    if not np.all(F['pre_ok'][m] == 1) or not np.array_equal(F['X_input'][m][:, -1], Lb['p_pre'][m]):
                        raise SystemExit('pre_ok / p_pre contract failed')
                    versions = np.full(int(m.sum()), CC.SET_VERSION[set_name])
                    r = evaluate_rows(set_name, coh, F['match'][m].astype(str), F['s_ms'][m], Lb['Y_h90'][m].astype(int), F['X_input'][m], names, versions, bundles, fz, base, False, st, ridge, vocab, tab)
                    r['checks']['parent_row_contract'] = ch
                    results[set_name][coh] = r
                F = Lb = Co = None
            C.write_json(base / 'eval' / 'results_partial.json', dict(role=CC.ROLE_TAG, complete=False, results=results))
        after = verify(base, fz)
        C.write_json(base / 'eval' / 'hashes_after.json', dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), results=after))
        ok = all(after.values())
        sha = C.write_json(base / 'eval' / 'results.json', dict(role=CC.ROLE_TAG, version=CC.VERSION, smoke_train_only=bool(args.smoke),
                                                                frozen_manifest_sha256=C.sha256_file(CC.frozen_path(base)), complete=True, frozen_hashes_unchanged=ok,
                                                                named_models=list(CC.NAMED) + ([] if args.smoke else list(CC.PARENT_NAMED)), bootstrap_models=list(CC.BOOT_MODELS),
                                                                contrasts=[dict(a=a, b=b, label=l) for a, b, l in CC.CONTRASTS],
                                                                primary=dict(a=CC.PRIMARY[0], b=CC.PRIMARY[1], cohort=CC.PRIMARY[2], cell=CC.PRIMARY[3], metric=CC.PRIMARY[4], set='MAIN_TEST'),
                                                                evaluated_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                                                                prior_exposure='old TEST/external results known before this run; exploratory follow-up', results=results))
        st.update('complete' if ok else 'failed', 'evaluate', results_sha256=sha, hashes_unchanged=ok, next_step='snapshot after, post-run checks, report')
        return 0 if ok else 1
    except SystemExit as exc:
        CC.log_failure(st.group, exc)
        st.update('failed', 'evaluate', error=str(exc)[:1000], next_step='inspect (failure retained)')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        CC.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'evaluate', error=repr(exc)[:1000], next_step='inspect, fix, rerun (read-only on frozen models)')
        return 3


if __name__ == '__main__':
    sys.exit(main())
