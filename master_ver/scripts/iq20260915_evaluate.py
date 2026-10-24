"""Stage T: frozen evaluation (requires frozen_manifest.json; no refit / reselection) on MAIN TEST 15.16 and each external
set separately, for T (primary) and N (secondary) h90-valid rows.

Models: the three frozen family winners (plus the overall-winner alias), all 54 joint candidates (descriptive), and the
legacy references from the parent A_<set>_h90_<cohort>.npz (old A specialist, pooled q, cohort p_pre spline / logistic /
constant) joined by exact (match, s_ms) keys. Cells: all, B40, B45, pre-time bins and B40 x time from frozen p_pre and
pre time only. Paired match bootstrap (1000, seed 20260915) for the five planned contrasts on all / B40 / B45 cells with
>= 30 matches and both label classes. Frozen hashes verified before and after.
--smoke evaluates the TRAIN-only smoke freeze on its pseudo Q_SELECT rows with stand-in legacy columns (code path only).
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import sys  # noqa: E402

sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import iq20260915_common as Q  # noqa: E402

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

BOOT_MODELS = ('pt_winner', 'logit_winner', 'lgbm_winner', 'old_p_pre_spline', 'old_A_specialist')
NAMED = Q.NEW_WINNERS + ('overall_winner',) + tuple(Q.LEGACY)


def verify(base, fz):
    return {p: C.sha256_file((Q.OUT if p == 'protocol.json' else base) / p) == sha for p, sha in fz['frozen_files_sha256'].items()}


def load_bundles(base, fz):
    return {coh: {fam: {cfg: joblib.load(base / 'models' / coh / fam / f'{cfg}.joblib') for cfg in Q.config_names(fam)} for fam in Q.FAMILIES}
            for coh in Q.COHORTS}


def new_predictions(bundles, X, names):
    P, raw, seed = {}, {}, {}
    for fam, cfgs in bundles.items():
        for cfg, b in cfgs.items():
            raw[cfg] = Q.bundle_raw(b, X, names)
            if fam == 'lgbm':
                seed[cfg] = b['base'].seed_raw(np.asarray(X)[:, b['input_columns']])
                if not np.allclose(seed[cfg].mean(axis=0), raw[cfg], rtol=0, atol=1e-15):
                    raise SystemExit('seed mean differs from raw')
            for cal in Q.CALS:
                P[f'{cfg}__{cal}'] = Q.calibrate(b, cal, raw[cfg])
    return P, raw, seed


def legacy_columns(set_name, coh, g, s, y, fz, base):
    path = Q.CR / 'eval' / 'predictions' / f'A_{set_name}_h90_{coh}.npz'
    Q.log_access(base, f'legacy reference predictions {set_name} {coh}', f'parent A_{set_name}_h90_{coh}.npz', 'parent frozen predictions incl. y', True)
    with np.load(path, allow_pickle=False) as z:
        A = {k: z[k] for k in z.files}
    ag, a_s = A['match'].astype(str), A['s_ms']
    chk = dict(file_sha256=C.sha256_file(path), rows_equal=len(ag) == len(g),
               parent_keys_unique=len(set(zip(ag.tolist(), a_s.tolist()))) == len(ag))
    try:
        ix, how = Q.key_join(g, s, ag, a_s)
    except ValueError as exc:
        raise SystemExit(f'legacy join failed for {set_name} {coh}: {exc}')
    chk['join'] = f'{how} on (match, s_ms)'
    chk['every_row_joined_once'] = bool(len(np.unique(ix)) == len(g) == len(ag))
    chk['y_equal_parent_labels'] = bool(np.array_equal(A['y'][ix].astype(int), y.astype(int)))
    sch = str(A['specialist_chosen'])
    chk['specialist_chosen'] = sch
    chk['specialist_chosen_equals_parent_frozen'] = sch == fz['legacy_references']['A_specialist_h90_chosen'][coh]
    if not (chk['parent_keys_unique'] and chk['every_row_joined_once'] and chk['y_equal_parent_labels'] and chk['specialist_chosen_equals_parent_frozen']):
        raise SystemExit(f'legacy contract failed {set_name} {coh}: {chk}')
    cols = {name: A[f'spec_{sch}' if key is None else key][ix].astype(np.float64) for name, key in Q.LEGACY.items()}
    return cols, chk


def smoke_legacy(P, y):
    """Stand-in columns for the code path only (TRAIN-only smoke; not legacy models)."""
    return {'old_A_specialist': P['logit_C0.01__raw'], 'old_pooled': P['logit_C1__isotonic'], 'old_p_pre_spline': P['pt_C1__raw'],
            'old_p_pre_logistic': P['pt_C0.0001__raw'], 'old_constant': np.full(len(y), float(np.mean(y)))}


def evaluate_rows(set_name, coh, g, s, y, X, names, bundles, fz, base, smoke, st):
    ridge_ix = [names.index(n) for n in C.read_json(Q.FC / 'q_pre_only_schema.json')['predictor_sets']['ridge']]
    rows = len(y)
    checks = dict(rows=rows, keys_unique=len(set(zip(g.tolist(), s.tolist()))) == rows,
                  nonfinite_inputs=int((~np.isfinite(X[:, ridge_ix])).sum()))
    if not checks['keys_unique'] or checks['nonfinite_inputs']:
        raise SystemExit(f'{set_name} {coh}: key/finite contract failed {checks}')
    p_pre = X[:, names.index('p_pre_V')]
    t = X[:, names.index('time_minutes')]
    P, raw, seed = new_predictions(bundles[coh], X, names)
    named = {f'{fam}_winner': P[fz['family_winners'][f'{fam}_{coh}']['chosen']] for fam in Q.FAMILIES}
    named['overall_winner'] = P[fz['overall_winner'][coh]['chosen']]
    if smoke:
        leg, lchk = smoke_legacy(P, y), dict(note='SMOKE_STANDIN columns (not legacy models)')
    else:
        leg, lchk = legacy_columns(set_name, coh, g, s, y, fz, base)
    named.update(leg)
    checks['legacy'] = lchk
    cells = Q.cell_masks(p_pre, t)
    tb = [k for k in cells if k.startswith('time_')]
    checks['time_bins_partition_rows'] = bool(np.all(np.sum([cells[k] for k in tb], axis=0) == 1)) if rows else True
    checks['B45_subset_B40'] = bool(np.all(~cells['B45'] | cells['B40']))
    checks['B40_bounds_inclusive'] = bool(np.array_equal(cells['B40'], (p_pre >= 0.4) & (p_pre <= 0.6)))
    res = dict(set=set_name, cohort=coh, rows=rows, matches=int(len(np.unique(g))), checks=checks,
               winners={k: fz['family_winners'][f'{k}_{coh}']['chosen'] for k in Q.FAMILIES}, overall=fz['overall_winner'][coh]['chosen'],
               cells={k: dict(rows=int(m.sum()), matches=int(len(np.unique(g[m])))) for k, m in cells.items()})
    res['metrics_named'] = {c: {name: Q.cell_metrics(y[m], named[name][m], g[m], bins=True) for name in NAMED} for c, m in cells.items()}
    res['metrics_all_candidates_DESCRIPTIVE'] = {c: {cand: Q.cell_metrics(y[cells[c]], P[cand][cells[c]], g[cells[c]], bins=False) for cand in P}
                                                 for c in ('all', 'B40')}
    wcfg = fz['family_winners'][f'lgbm_{coh}']['config']
    res['lgbm_winner_config_seed_metrics_DESCRIPTIVE'] = {c: {str(sd): Q.cell_metrics(y[cells[c]], seed[wcfg][j][cells[c]], g[cells[c]], bins=False)
                                                              for j, sd in enumerate(Q.LGBM_SEEDS)} for c in ('all', 'B40')}
    res['bootstrap'] = {}
    for c in ('all', 'B40', 'B45'):
        m = cells[c]
        nm = int(len(np.unique(g[m])))
        classes = int(len(np.unique(y[m]))) if m.any() else 0
        if nm >= Q.SPARSE_MATCHES and classes == 2:
            t0 = time.time()
            res['bootstrap'][c] = Q.paired_bootstrap(y[m], {k: named[k][m] for k in BOOT_MODELS}, g[m], Q.CONTRASTS)
            res['bootstrap'][c]['seconds'] = round(time.time() - t0, 1)
        else:
            res['bootstrap'][c] = dict(computed=False, reason=f'matches={nm} (<{Q.SPARSE_MATCHES}) or label classes={classes}')
    res['time_cells_bootstrap'] = 'CI not computed for time and B40 x time cells (point estimates and reliability only, per specification)'
    arrays = dict(match=g, s_ms=s, y=y, p_pre=p_pre, time_minutes=t, **{f'cell__{k}': v for k, v in cells.items()},
                  **{f'named__{k}': v for k, v in named.items()}, **{f'cand__{k}': v for k, v in P.items()},
                  **{f'raw__{k}': v for k, v in raw.items()},
                  **{f'seedraw__{k}__{sd}': v[j] for k, v in seed.items() for j, sd in enumerate(Q.LGBM_SEEDS)})
    res['predictions_file'] = f'eval/predictions/{set_name}_h90_{coh}.npz'
    res['predictions_sha256'] = C.save_npz(base / 'eval' / 'predictions' / f'{set_name}_h90_{coh}.npz', **arrays)
    a, b40 = res['metrics_named']['all'], res['metrics_named']['B40']
    st.log(f'{set_name} {coh}: rows={rows} ' + ' '.join(f'{k}={a[k]["brier"]:.6f}' for k in NAMED if a[k]['brier'] is not None)
           + ' | B40 ' + ' '.join(f'{k}={b40[k]["brier"]:.6f}' for k in NAMED if b40[k]['brier'] is not None))
    return res


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    Q.log_command()
    base = Q.SMOKE if args.smoke else Q.OUT
    st = Q.Status('evaluate' + ('_smoke' if args.smoke else ''))
    try:
        if not Q.frozen_path(base).exists():
            raise SystemExit('evaluation requires frozen_manifest.json')
        fz = C.read_json(Q.frozen_path(base))
        if bool(fz['smoke_train_only']) != bool(args.smoke):
            raise SystemExit('frozen manifest mode differs')
        before = verify(base, fz)
        if not all(before.values()):
            raise SystemExit(f'frozen hash mismatch before evaluation: {[k for k, v in before.items() if not v][:5]}')
        C.write_json(base / 'eval' / 'hashes_before.json', dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), results=before))
        schema = C.read_json(Q.FC / 'q_pre_only_schema.json')
        bundles = load_bundles(base, fz)
        results = {}
        sets = ('SMOKE_PSEUDO_Q_SELECT',) if args.smoke else Q.EVAL_SETS
        for k, set_name in enumerate(sets):
            st.update('running', f'evaluate_{set_name}', processed=k, total=len(sets), next_step='cohorts T then N')
            results[set_name] = {}
            if args.smoke:
                for coh in Q.COHORTS:
                    D = Q.load_trainval(base, smoke=True, cohort=coh)
                    m = D['role'] == 'Q_SELECT'
                    results[set_name][coh] = evaluate_rows(set_name, coh, D['g'][m], D['s_ms'][m], D['y'][m], D['X'][m], D['names'], bundles, fz, base, True, st)
                    D = None
            else:
                F, Lb, Co, ch = Q.load_parent_set(set_name, base, f'frozen evaluation {set_name}')
                names = [str(x) for x in F['input_names']]
                if names != schema['input_names_all']:
                    raise SystemExit('schema drift in evaluation set')
                for coh in Q.COHORTS:
                    m = (Lb['valid_h90'] == 1) & (Co['cohort'] == Q.COHORT_CODE[coh])
                    if int(m.sum()) != Q.EXPECTED_COUNTS[set_name][coh]:
                        raise SystemExit(f'count mismatch {set_name} {coh}: {int(m.sum())} vs {Q.EXPECTED_COUNTS[set_name][coh]}')
                    if not np.all(F['pre_ok'][m] == 1) or not np.array_equal(F['X_input'][m][:, -1], Lb['p_pre'][m]):
                        raise SystemExit('pre_ok / p_pre contract failed')
                    r = evaluate_rows(set_name, coh, F['match'][m].astype(str), F['s_ms'][m], Lb['Y_h90'][m].astype(int), F['X_input'][m], names,
                                      bundles, fz, base, False, st)
                    r['checks']['parent_row_contract'] = ch
                    results[set_name][coh] = r
                    st.update('running', f'evaluated_{set_name}_{coh}', processed=k, total=len(sets),
                              all_brier={n: r['metrics_named']['all'][n]['brier'] for n in NAMED},
                              next_step='next cohort / set')
                F = Lb = Co = None
            C.write_json(base / 'eval' / 'results_partial.json', dict(role=Q.ROLE_TAG, frozen_manifest_sha256=C.sha256_file(Q.frozen_path(base)),
                                                                     complete=False, results=results))
        after = verify(base, fz)
        C.write_json(base / 'eval' / 'hashes_after.json', dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), results=after))
        ok = all(after.values())
        sha = C.write_json(base / 'eval' / 'results.json', dict(
            role=Q.ROLE_TAG, version=Q.VERSION, smoke_train_only=bool(args.smoke), frozen_manifest_sha256=C.sha256_file(Q.frozen_path(base)),
            complete=True, frozen_hashes_unchanged=ok, named_models=list(NAMED), bootstrap_models=list(BOOT_MODELS),
            contrasts=[dict(a=a, b=b, label=l) for a, b, l in Q.CONTRASTS], evaluated_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            prior_exposure='old TEST/external results known before this run; exploratory follow-up', results=results))
        st.update('complete' if ok else 'failed', 'evaluate', results_sha256=sha, hashes_unchanged=ok, next_step='snapshot after, post-run checks, report')
        return 0 if ok else 1
    except SystemExit as exc:
        Q.log_failure(st.group, exc)
        st.update('failed', 'evaluate', error=str(exc)[:1000], next_step='inspect (failure retained)')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        Q.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'evaluate', error=repr(exc)[:1000], next_step='inspect, fix, rerun (read-only on frozen models)')
        return 3


if __name__ == '__main__':
    sys.exit(main())
