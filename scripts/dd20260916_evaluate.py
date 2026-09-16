"""Stage T: sealed evaluation on the dev MAIN TEST and external populations: refit winners and the frozen parent iq winners (pt, logit, lgbm)
on the same rows; identity of the frozen winners with the parent predictions on common rows; population census by cohort; cells; paired
bootstrap; old-population metrics quoted from the parent results for the side-by-side table."""
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
import dd20260916_common as DD  # noqa: E402
import fc20260915_common as C  # noqa: E402
import iq20260915_common as Q  # noqa: E402

import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

PARENT_FAMS = ('pt', 'logit', 'lgbm')


def verify(fz):
    return {p: C.sha256_file(DD.OUT / p) == sha for p, sha in fz['frozen_files_sha256'].items()}


def evaluate_rows(set_name, coh, g, s, y, X, names, src, refit, parent, fz, st, old):
    rows = len(y)
    checks = dict(rows=rows, keys_unique=len(set(zip(g.tolist(), s.tolist()))) == rows, nonfinite_inputs=int((~np.isfinite(X[:, refit['lgbm']['input_columns']])).sum()))
    if not checks['keys_unique'] or checks['nonfinite_inputs']:
        raise SystemExit(f'{set_name} {coh}: key/finite contract failed {checks}')
    p_pre, t = X[:, names.index('p_pre_V')], X[:, names.index('time_minutes')]
    named, raw = {}, {}
    for fam in DD.FAMILIES:
        b = refit[fam]
        raw[f'refit_{fam}'] = DD.bundle_raw(b, X, names)
        named[f'refit_{fam}'] = DD.calibrate(b, fz['family_winners'][f'{fam}_{coh}']['calibration'], raw[f'refit_{fam}'])
    for fam in PARENT_FAMS:
        b, cal = parent[fam]
        raw[f'frozen_{fam}'] = DD.parent_bundle_raw(b, X, names)
        named[f'frozen_{fam}'] = DD.calibrate(b, cal, raw[f'frozen_{fam}'])
    # identity with the parent predictions on common rows (parent eval arrays; sealed parent, after freeze)
    path = DD.IQ / 'eval' / 'predictions' / f'{set_name}_h{DD.H}_{coh}.npz'
    DD.log_access(DD.OUT, f'parent iq eval predictions {set_name} {coh} (identity on common rows)', f'parent {path.name}', 'parent frozen predictions incl. y', True)
    with np.load(path, allow_pickle=False) as z:
        A = {k: z[k] for k in ('match', 's_ms', 'y', 'named__pt_winner', 'named__logit_winner', 'named__lgbm_winner')}
    pk = {(m, int(v)): i for i, (m, v) in enumerate(zip(A['match'].astype(str).tolist(), A['s_ms'].tolist()))}
    common = src == 'parent'
    ix = np.asarray([pk.get((m, int(v)), -1) for m, v in zip(g[common].tolist(), s[common].tolist())])
    ident = dict(file_sha256_equals_frozen=C.sha256_file(path) == fz['parent_references']['parent_predictions_sha256'][f'{set_name}_{coh}'], common_rows=int(common.sum()),
                 common_rows_found_in_parent=int((ix >= 0).sum()), parent_rows_old=int(len(A['match'])))
    ok = ix >= 0
    ident['y_equal'] = bool(np.array_equal(A['y'][ix[ok]].astype(int), y[common][ok].astype(int)))
    for fam in PARENT_FAMS:
        ident[f'frozen_{fam}_equals_parent_on_common_rows'] = bool(np.array_equal(A[f'named__{fam}_winner'][ix[ok]], named[f'frozen_{fam}'][common][ok]))
    ident['old_rows_not_in_dev'] = int(len(A['match']) - int(ok.sum()))
    checks['identity'] = ident
    if not (ident['file_sha256_equals_frozen'] and ident['common_rows_found_in_parent'] == ident['common_rows'] and ident['y_equal'] and all(ident[f'frozen_{f}_equals_parent_on_common_rows'] for f in PARENT_FAMS)):
        raise SystemExit(f'identity with the parent failed {set_name} {coh}: {ident}')
    cells = DD.cell_masks(p_pre, t)
    res = dict(set=set_name, cohort=coh, rows=rows, matches=int(len(np.unique(g))), checks=checks,
               population=dict(dev_rows=rows, parent_source_rows=int(common.sum()), rebuilt_rows=int((~common).sum()), old_rows=int(len(A['match'])), old_rows_dropped=ident['old_rows_not_in_dev'],
                               old_rows_expected=DD.EXPECTED_COUNTS_OLD[set_name][coh]),
               winners={k: fz['family_winners'][f'{k}_{coh}']['chosen'] for k in DD.FAMILIES}, cells={k: dict(rows=int(m.sum()), matches=int(len(np.unique(g[m])))) for k, m in cells.items()},
               old_metrics_named=old)
    res['metrics_named'] = {c: {name: DD.cell_metrics(y[m], named[name][m], g[m], bins=True) for name in DD.NAMED} for c, m in cells.items()}
    res['metrics_by_row_source_DESCRIPTIVE'] = {srcname: {name: DD.cell_metrics(y[mm], named[name][mm], g[mm], bins=False) for name in DD.NAMED}
                                                for srcname, mm in (('parent', common), ('rebuilt', ~common)) if mm.any()}
    res['bootstrap'] = {}
    for c in ('all', 'B40', 'B45'):
        m = cells[c]
        nm = int(len(np.unique(g[m])))
        classes = int(len(np.unique(y[m]))) if m.any() else 0
        if nm >= DD.SPARSE_MATCHES and classes == 2:
            t1 = time.time()
            res['bootstrap'][c] = DD.paired_bootstrap(y[m], {k: named[k][m] for k in DD.BOOT_MODELS}, g[m], DD.CONTRASTS)
            res['bootstrap'][c]['seconds'] = round(time.time() - t1, 1)
        else:
            res['bootstrap'][c] = dict(computed=False, reason=f'matches={nm} (<{DD.SPARSE_MATCHES}) or label classes={classes}')
    arrays = dict(match=g, s_ms=s, y=y, p_pre=p_pre, time_minutes=t, row_source=src, **{f'cell__{k}': v for k, v in cells.items()}, **{f'named__{k}': v for k, v in named.items()},
                  **{f'raw__{k}': v for k, v in raw.items()})
    res['predictions_file'] = f'eval/predictions/{set_name}_h{DD.H}_{coh}.npz'
    res['predictions_sha256'] = C.save_npz(DD.OUT / 'eval' / 'predictions' / f'{set_name}_h{DD.H}_{coh}.npz', **arrays)
    a = res['metrics_named']['all']
    st.log(f'{set_name} {coh}: rows={rows} (rebuilt {int((~common).sum())}) ' + ' '.join(f'{k}={a[k]["brier"]:.6f}' for k in DD.NAMED) + f' | old lgbm {old["all"]["lgbm_winner"]["brier"]:.6f}')
    return res


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    DD.log_command()
    st = DD.Status('evaluate')
    try:
        if not DD.frozen_path(DD.OUT).exists():
            raise SystemExit('evaluation requires frozen_manifest.json')
        fz = C.read_json(DD.frozen_path(DD.OUT))
        for s in DD.EVAL_SETS:
            if not (DD.OUT / 'rebuild' / f'{s}.json').exists():
                raise SystemExit(f'rebuild of {s} must precede the evaluation')
        before = verify(fz)
        if not all(before.values()):
            raise SystemExit(f'frozen hash mismatch before evaluation: {[k for k, v in before.items() if not v][:5]}')
        C.write_json(DD.OUT / 'eval' / 'hashes_before.json', dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), results=before))
        schema = C.read_json(DD.FC / 'q_pre_only_schema.json')
        refit = {coh: {fam: joblib.load(DD.OUT / fz['family_winners'][f'{fam}_{coh}']['bundle']) for fam in DD.FAMILIES} for coh in DD.COHORTS}
        winners = fz['parent_references']['iq_winners']
        parent = {coh: {fam: (joblib.load(DD.IQ / winners[f'{fam}_{coh}']['bundle']), winners[f'{fam}_{coh}']['calibration']) for fam in PARENT_FAMS} for coh in DD.COHORTS}
        for coh in DD.COHORTS:
            for fam in PARENT_FAMS:
                if C.sha256_file(DD.IQ / winners[f'{fam}_{coh}']['bundle']) != winners[f'{fam}_{coh}']['bundle_sha256']:
                    raise SystemExit('parent bundle hash differs from the iq freeze')
        iq_res = C.read_json(DD.IQ / 'eval' / 'results.json')['results']
        results = {}
        for k, set_name in enumerate(DD.EVAL_SETS):
            st.update('running', f'evaluate_{set_name}', processed=k, total=len(DD.EVAL_SETS), next_step='cohorts T then N')
            F, Lb, Co = DD.load_dev_set(set_name, DD.OUT, f'frozen evaluation {set_name} (dev population)')
            names = [str(x) for x in F['input_names']]
            if names != schema['input_names_all']:
                raise SystemExit('schema drift in the dev population')
            results[set_name] = {}
            for coh in DD.COHORTS:
                m = (Lb['valid_h90'] == 1) & (Co['cohort'] == DD.COHORT_CODE[coh])
                if not np.all(F['pre_ok'][m] == 1):
                    raise SystemExit('valid rows without pre inputs')
                old = {c: {f'{fam}_winner': iq_res[set_name][coh]['metrics_named'][c][f'{fam}_winner'] for fam in PARENT_FAMS} for c in ('all', 'B40', 'B45')}
                results[set_name][coh] = evaluate_rows(set_name, coh, F['match'][m].astype(str), F['s_ms'][m], Lb['Y_h90'][m].astype(int), F['X_input'][m], names,
                                                       F['row_source'][m].astype(str), refit[coh], parent[coh], fz, st, old)
            F = Lb = Co = None
            C.write_json(DD.OUT / 'eval' / 'results_partial.json', dict(role=DD.ROLE_TAG, complete=False, results=results))
        after = verify(fz)
        C.write_json(DD.OUT / 'eval' / 'hashes_after.json', dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), results=after))
        ok = all(after.values())
        sha = C.write_json(DD.OUT / 'eval' / 'results.json', dict(role=DD.ROLE_TAG, version=DD.VERSION, frozen_manifest_sha256=C.sha256_file(DD.frozen_path(DD.OUT)), complete=True,
                                                                  frozen_hashes_unchanged=ok, named_models=list(DD.NAMED), bootstrap_models=list(DD.BOOT_MODELS),
                                                                  contrasts=[dict(a=a, b=b, label=l) for a, b, l in DD.CONTRASTS], definition=DD.DEFINITIONS['dev'],
                                                                  evaluated_at=time.strftime('%Y-%m-%d %H:%M:%S'), prior_exposure='old-population TEST/external results known before this run', results=results))
        st.update('complete' if ok else 'failed', 'evaluate', results_sha256=sha, hashes_unchanged=ok, next_step='snapshot after, post-run checks, report')
        return 0 if ok else 1
    except SystemExit as exc:
        DD.log_failure(st.group, exc)
        st.update('failed', 'evaluate', error=str(exc)[:1000], next_step='inspect')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        DD.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'evaluate', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
