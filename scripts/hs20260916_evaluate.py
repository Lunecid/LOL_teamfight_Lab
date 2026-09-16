"""Stage T: frozen evaluation per horizon on MAIN TEST 15.16 and each external set separately, T and N h-valid rows.
Models: five frozen family winners + overall alias + cohort-role legacy references at that horizon (exact key join). Cells and
paired bootstrap as in Track A. --smoke: h60 pseudo Q_SELECT rows with stand-in legacy columns."""
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
import hs20260916_common as HS  # noqa: E402
import fc20260915_common as C  # noqa: E402

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402


def verify(base, fz):
    return {p: C.sha256_file((HS.OUT if p == 'protocol.json' else base) / p) == sha for p, sha in fz['frozen_files_sha256'].items()}


def legacy_columns(set_name, h, coh, g, s, y, fz, base):
    path = HS.legacy_path(set_name, h, coh)
    HS.log_access(base, f'legacy reference predictions {set_name} h{h} {coh}', f'parent {path.name}', 'parent frozen predictions incl. y', True)
    with np.load(path, allow_pickle=False) as z:
        A = {k: z[k] for k in z.files}
    ag, a_s = A['match'].astype(str), A['s_ms']
    chk = dict(file_sha256=C.sha256_file(path), file_sha256_equals_frozen=C.sha256_file(path) == fz['parent_references']['legacy_predictions_sha256'][f'{set_name}_h{h}_{coh}'],
               rows_equal=len(ag) == len(g), parent_keys_unique=len(set(zip(ag.tolist(), a_s.tolist()))) == len(ag))
    try:
        ix, how = HS.key_join(g, s, ag, a_s)
    except ValueError as exc:
        raise SystemExit(f'legacy join failed for {set_name} h{h} {coh}: {exc}')
    chk['join'] = f'{how} on (match, s_ms)'
    chk['every_row_joined_once'] = bool(len(np.unique(ix)) == len(g) == len(ag))
    chk['y_equal_parent_labels'] = bool(np.array_equal(A['y'][ix].astype(int), y.astype(int)))
    sch = str(A['specialist_chosen'])
    chk['specialist_chosen'] = sch
    chk['specialist_chosen_equals_parent_frozen'] = sch == fz['parent_references']['legacy_A_specialists'][f'{coh}_h{h}']
    if not (chk['file_sha256_equals_frozen'] and chk['parent_keys_unique'] and chk['every_row_joined_once'] and chk['y_equal_parent_labels'] and chk['specialist_chosen_equals_parent_frozen']):
        raise SystemExit(f'legacy contract failed {set_name} h{h} {coh}: {chk}')
    cols = {name: A[f'spec_{sch}' if key is None else key][ix].astype(np.float64) for name, key in HS.LEGACY.items()}
    return cols, chk


def evaluate_rows(h, set_name, coh, g, s, y, X, names, bundles, fz, base, smoke, st):
    ridge_ix = HS.input_columns(names, 'logit', C.read_json(HS.FC / 'q_pre_only_schema.json')['predictor_sets']['ridge'])
    rows = len(y)
    checks = dict(rows=rows, keys_unique=len(set(zip(g.tolist(), s.tolist()))) == rows, nonfinite_inputs=int((~np.isfinite(X[:, ridge_ix])).sum()))
    if not checks['keys_unique'] or checks['nonfinite_inputs']:
        raise SystemExit(f'{set_name} h{h} {coh}: key/finite contract failed {checks}')
    p_pre = X[:, names.index('p_pre_V')]
    t = X[:, names.index('time_minutes')]
    t0 = time.time()
    P, raw, seed = {}, {}, {}
    for fam, b in bundles[f'h{h}'][coh].items():
        if C.sha256_json(list(names)) != b['input_names_all_sha256']:
            raise SystemExit('input schema differs from the bundle schema')
        Xin = np.asarray(X)[:, b['input_columns']]
        if fam in HS.SEEDED:
            seed[fam] = b['base'].seed_raw(Xin)
            raw[fam] = seed[fam].mean(axis=0)
        else:
            raw[fam] = b['base'].raw(Xin)
        for cal in HS.CALS:
            P[f'{fam}:{b["config"]}__{cal}'] = HS.calibrate(b, cal, raw[fam])
    checks['seconds_predict'] = round(time.time() - t0, 1)
    named = {f'{fam}_winner': P[f'{fam}:{fz["family_winners"][f"h{h}_{fam}_{coh}"]["chosen"]}'] for fam in HS.FAMILIES}
    named['overall_winner'] = P[fz['overall_winner'][f'h{h}_{coh}']['chosen']]
    if smoke:
        cands = sorted(P)
        cols = {'old_A_specialist': P[cands[0]], 'old_pooled': P[cands[1]], 'old_p_pre_spline': P[cands[2]], 'old_p_pre_logistic': P[cands[3]], 'old_constant': np.full(rows, float(np.mean(y)))}
        lchk = dict(note='SMOKE_STANDIN columns (not legacy models)')
    else:
        cols, lchk = legacy_columns(set_name, h, coh, g, s, y, fz, base)
    named.update(cols)
    checks['legacy'] = lchk
    cells = HS.cell_masks(p_pre, t)
    tb = [k for k in cells if k.startswith('time_')]
    checks['time_bins_partition_rows'] = bool(np.all(np.sum([cells[k] for k in tb], axis=0) == 1)) if rows else True
    checks['B45_subset_B40'] = bool(np.all(~cells['B45'] | cells['B40']))
    res = dict(horizon_s=h, set=set_name, cohort=coh, rows=rows, matches=int(len(np.unique(g))), checks=checks,
               winners={f: fz['family_winners'][f'h{h}_{f}_{coh}']['chosen'] for f in HS.FAMILIES}, overall=fz['overall_winner'][f'h{h}_{coh}']['chosen'],
               cells={k: dict(rows=int(m.sum()), matches=int(len(np.unique(g[m])))) for k, m in cells.items()})
    res['metrics_named'] = {c: {name: HS.cell_metrics(y[m], named[name][m], g[m], bins=True) for name in HS.NAMED} for c, m in cells.items()}
    res['metrics_all_candidates_DESCRIPTIVE'] = {c: {cand: HS.cell_metrics(y[cells[c]], P[cand][cells[c]], g[cells[c]], bins=False) for cand in P} for c in ('all', 'B40')}
    res['winner_seed_metrics_DESCRIPTIVE'] = {fam: {c: {str(sd): HS.cell_metrics(y[cells[c]], seed[fam][j][cells[c]], g[cells[c]], bins=False) for j, sd in enumerate(HS.SEEDS[fam])}
                                                    for c in ('all', 'B40')} for fam in seed}
    res['bootstrap'] = {}
    for c in ('all', 'B40', 'B45'):
        m = cells[c]
        nm = int(len(np.unique(g[m])))
        classes = int(len(np.unique(y[m]))) if m.any() else 0
        if nm >= HS.SPARSE_MATCHES and classes == 2:
            t1 = time.time()
            res['bootstrap'][c] = HS.paired_bootstrap(y[m], {k: named[k][m] for k in HS.BOOT_MODELS}, g[m], HS.CONTRASTS)
            res['bootstrap'][c]['seconds'] = round(time.time() - t1, 1)
        else:
            res['bootstrap'][c] = dict(computed=False, reason=f'matches={nm} (<{HS.SPARSE_MATCHES}) or label classes={classes}')
    res['time_cells_bootstrap'] = 'CI not computed for time and B40 x time cells (point estimates and reliability only, per specification)'
    arrays = dict(match=g, s_ms=s, y=y, p_pre=p_pre, time_minutes=t, **{f'cell__{k}': v for k, v in cells.items()},
                  **{f'named__{k}': v for k, v in named.items()}, **{f'cand__{k}': v for k, v in P.items()}, **{f'raw__{k}': v for k, v in raw.items()},
                  **{f'seedraw__{k}__{sd}': v[j] for k, v in seed.items() for j, sd in enumerate(HS.SEEDS[k])})
    res['predictions_file'] = f'eval/predictions/{set_name}_h{h}_{coh}.npz'
    res['predictions_sha256'] = C.save_npz(base / 'eval' / 'predictions' / f'{set_name}_h{h}_{coh}.npz', **arrays)
    a = res['metrics_named']['all']
    st.log(f'{set_name} h{h} {coh}: rows={rows} ' + ' '.join(f'{k}={a[k]["brier"]:.6f}' for k in HS.NAMED if a[k]['brier'] is not None))
    return res


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    HS.log_command()
    base = HS.SMOKE if args.smoke else HS.OUT
    st = HS.Status('evaluate' + ('_smoke' if args.smoke else ''))
    try:
        if not HS.frozen_path(base).exists():
            raise SystemExit('evaluation requires frozen_manifest.json')
        fz = C.read_json(HS.frozen_path(base))
        if bool(fz['smoke_train_only']) != bool(args.smoke):
            raise SystemExit('frozen manifest mode differs')
        before = verify(base, fz)
        if not all(before.values()):
            raise SystemExit(f'frozen hash mismatch before evaluation: {[k for k, v in before.items() if not v][:5]}')
        C.write_json(base / 'eval' / 'hashes_before.json', dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), results=before))
        schema = C.read_json(HS.FC / 'q_pre_only_schema.json')
        horizons = fz['horizons']
        bundles = {f'h{h}': {coh: {fam: joblib.load(base / fz['family_winners'][f'h{h}_{fam}_{coh}']['bundle']) for fam in HS.FAMILIES} for coh in HS.COHORTS} for h in horizons}
        results = {}
        sets = ('SMOKE_PSEUDO_Q_SELECT',) if args.smoke else HS.EVAL_SETS
        total = len(horizons) * len(sets)
        k = 0
        for h in horizons:
            results[f'h{h}'] = {}
            for set_name in sets:
                st.update('running', f'evaluate_h{h}_{set_name}', processed=k, total=total, next_step='cohorts T then N')
                results[f'h{h}'][set_name] = {}
                if args.smoke:
                    for coh in HS.COHORTS:
                        D = HS.load_trainval(base, h, smoke=True, cohort=coh)
                        m = D['role'] == 'Q_SELECT'
                        results[f'h{h}'][set_name][coh] = evaluate_rows(h, set_name, coh, D['g'][m], D['s_ms'][m], D['y'][m], D['X'][m], D['names'], bundles, fz, base, True, st)
                        D = None
                else:
                    F, Lb, Co, ch = HS.load_parent_set(set_name, base, f'frozen evaluation {set_name} h{h}', h)
                    names = [str(x) for x in F['input_names']]
                    if names != schema['input_names_all']:
                        raise SystemExit('schema drift in evaluation set')
                    for coh in HS.COHORTS:
                        m = (Lb[f'valid_h{h}'] == 1) & (Co['cohort'] == HS.COHORT_CODE[coh])
                        if int(m.sum()) != HS.EXPECTED_COUNTS[set_name][coh]:
                            raise SystemExit(f'count mismatch {set_name} h{h} {coh}: {int(m.sum())} vs {HS.EXPECTED_COUNTS[set_name][coh]}')
                        if not np.all(F['pre_ok'][m] == 1) or not np.array_equal(F['X_input'][m][:, -1], Lb['p_pre'][m]):
                            raise SystemExit('pre_ok / p_pre contract failed')
                        r = evaluate_rows(h, set_name, coh, F['match'][m].astype(str), F['s_ms'][m], Lb[f'Y_h{h}'][m].astype(int), F['X_input'][m], names, bundles, fz, base, False, st)
                        r['checks']['parent_row_contract'] = ch
                        results[f'h{h}'][set_name][coh] = r
                    F = Lb = Co = None
                k += 1
                C.write_json(base / 'eval' / 'results_partial.json', dict(role=HS.ROLE_TAG, complete=False, results=results))
        after = verify(base, fz)
        C.write_json(base / 'eval' / 'hashes_after.json', dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), results=after))
        ok = all(after.values())
        sha = C.write_json(base / 'eval' / 'results.json', dict(role=HS.ROLE_TAG, version=HS.VERSION, smoke_train_only=bool(args.smoke), horizons=list(horizons),
                                                                frozen_manifest_sha256=C.sha256_file(HS.frozen_path(base)), complete=True, frozen_hashes_unchanged=ok,
                                                                named_models=list(HS.NAMED), bootstrap_models=list(HS.BOOT_MODELS),
                                                                contrasts=[dict(a=a, b=b, label=l) for a, b, l in HS.CONTRASTS], evaluated_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                                                                prior_exposure='old TEST/external results known before this run; exploratory follow-up', results=results))
        st.update('complete' if ok else 'failed', 'evaluate', results_sha256=sha, hashes_unchanged=ok, next_step='snapshot after, post-run checks, report')
        return 0 if ok else 1
    except SystemExit as exc:
        HS.log_failure(st.group, exc)
        st.update('failed', 'evaluate', error=str(exc)[:1000], next_step='inspect (failure retained)')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        HS.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'evaluate', error=repr(exc)[:1000], next_step='inspect, fix, rerun (read-only on frozen models)')
        return 3


if __name__ == '__main__':
    sys.exit(main())
