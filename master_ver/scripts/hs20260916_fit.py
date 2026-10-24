"""Stage F: one horizon x one cohort x one family: the fixed h90 configuration refit on the horizon's TRAIN labels, calibrators on
cohort Q_CAL, calibration-only selection (3 candidates) on cohort Q_SELECT. --smoke: TRAIN-only pseudo roles under smoke_train_only/."""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '4'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MKL_CBWR', 'AVX2,STRICT')

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hs20260916_common as HS  # noqa: E402
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402


def gates(base, smoke):
    if not (HS.OUT / 'protocol.json').exists():
        raise SystemExit('protocol.json must precede every fit')
    if HS.frozen_path(base).exists():
        raise SystemExit('frozen: no fitting')
    if not smoke:
        ct = HS.OUT / 'contract_tests' / 'result.json'
        if not ct.exists() or not C.read_json(ct).get('passed'):
            raise SystemExit('contract tests must pass before full fits')
        if not (HS.frozen_path(HS.SMOKE).exists() and (HS.SMOKE / 'eval' / 'results.json').exists()):
            raise SystemExit('TRAIN-only smoke must complete before full fits')
        if (HS.OUT / 'eval').exists():
            raise SystemExit('eval/ exists before full fits')


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizon', type=int, choices=HS.HORIZONS, required=True)
    ap.add_argument('--cohort', choices=HS.COHORTS, required=True)
    ap.add_argument('--family', choices=HS.FAMILIES, required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    HS.log_command()
    base = HS.SMOKE if args.smoke else HS.OUT
    h, coh, fam = args.horizon, args.cohort, args.family
    key = f'h{h}_{fam}_{coh}'
    st = HS.Status(f'fit_{key}' + ('_smoke' if args.smoke else ''))
    sel_path = base / 'selection' / f'{key}.json'
    try:
        gates(base, args.smoke)
        if sel_path.exists():
            st.update('complete', 'resume_skip', note=f'{sel_path.name} exists; not refit', next_step='next')
            return 0
        mdir = base / 'models' / f'h{h}' / coh / fam
        partial = [p.relative_to(base).as_posix() for p in (sorted(mdir.glob('*.joblib')) if mdir.exists() else [])
                   + [q for q in (base / 'predictions' / f'{key}_trainval.npz', base / 'internal_stop' / f'{key}.json') if q.exists()]]
        if partial:
            raise SystemExit(f'partial artifacts exist without {sel_path.name}: {partial}; move them to a named attempt directory first')
        proto = C.read_json(HS.OUT / 'protocol.json')
        cfg = proto['fixed_configurations']['registry'][coh][fam]
        schema = C.read_json(HS.FC / 'q_pre_only_schema.json')
        ridge = list(schema['predictor_sets']['ridge'])
        st.update('running', 'load', config=cfg, next_step='fit')
        D = HS.load_trainval(base, h, smoke=args.smoke, cohort=coh)
        names = D['names']
        if names != schema['input_names_all'] or C.q_feature_sets(names)['ridge'] != ridge:
            raise SystemExit('input schema differs from parent q_pre_only_schema.json')
        X, y, g = D['X'], D['y'], D['g']
        keys = list(zip(g.tolist(), D['s_ms'].tolist()))
        if len(set(keys)) != len(keys):
            raise SystemExit('duplicated (match, s_ms) keys; fitting blocked')
        M = {k: D['role'] == k for k in ('TRAIN', 'Q_CAL', 'Q_SELECT')}
        counts = {k: dict(rows=int(m.sum()), matches=int(len(np.unique(g[m]))),
                          positive_rate_match_weighted=float(np.average(y[m], weights=HS.weights(g[m]))) if m.any() else None) for k, m in M.items()}
        if not args.smoke:
            bad = {k: (counts[k]['rows'], HS.EXPECTED_COUNTS[k][coh]) for k in M if counts[k]['rows'] != HS.EXPECTED_COUNTS[k][coh]}
            if bad:
                raise SystemExit(f'count mismatch blocks fitting: {bad}')
        if len(set(g[M['TRAIN']]) & set(g[M['Q_CAL']])) or len(set(g[M['TRAIN']]) & set(g[M['Q_SELECT']])) or len(set(g[M['Q_CAL']]) & set(g[M['Q_SELECT']])):
            raise SystemExit('TRAIN / Q_CAL / Q_SELECT match overlap')
        cols = HS.input_columns(names, fam, ridge)
        ridge_ix = HS.input_columns(names, 'logit', ridge)
        nonfinite = {k: int((~np.isfinite(X[m][:, ridge_ix])).sum()) for k, m in M.items()}
        if any(nonfinite.values()):
            raise SystemExit(f'non-finite inputs among valid rows: {nonfinite}')
        trainfold = np.char.startswith(D['sr_true'], 'fold')
        vman = C.read_json(HS.FC / 'v_models_manifest.json')
        prov = HS.oof_provenance(g[trainfold], D['sr_true'][trainfold], D['adapter_id'][trainfold], D['adapter_sha256'][trainfold], vman)
        if not (prov['rows'] == prov['adapter_id_equals_own_heldout_fold'] == prov['adapter_sha_equals_manifest_oof_hash'] == prov['sub_role_equals_hash_fold']):
            raise SystemExit(f'own-match OOF provenance failed: {prov}')
        W = {k: HS.weights(g[m]) for k, m in M.items()}
        wrec = {k: HS.weight_record(g[m], W[k]) for k, m in M.items()}
        man = dict(role=HS.ROLE_TAG, version=HS.VERSION, horizon_s=h, cohort=coh, family=fam, config=cfg, smoke_train_only=bool(args.smoke),
                   output_is=f'q probability of generated label Y_h{h} within the cohort (NOT a V win probability)',
                   label_definition=f'Y_h{h} = 1[V(endpoint_h{h}) - V(s-1) > 0] (parent labels unchanged; TRAIN OOF V)',
                   protocol_sha256=C.sha256_file(HS.OUT / 'protocol.json'), device=args.device)
        input_names = HS.family_inputs(fam, ridge)
        t0 = time.time()
        st.update('running', f'fit_{cfg}', next_step='calibrate / save / reload')
        basem, ok_fit, fitrec, stoprec = HS.fit_family(fam, cfg, input_names, X[M['TRAIN']][:, cols], y[M['TRAIN']], g[M['TRAIN']], W['TRAIN'], device=args.device, log=st.log)
        fit_s = time.time() - t0
        t1 = time.time()
        seedraw = {k: basem.seed_raw(X[m][:, cols]) for k, m in M.items()} if fam in HS.SEEDED else None
        raw = {k: seedraw[k].mean(axis=0) for k in M} if seedraw is not None else {k: basem.raw(X[m][:, cols]) for k, m in M.items()}
        pred_s = time.time() - t1
        cals = K.fit_calibrators(raw['Q_CAL'], y[M['Q_CAL']], W['Q_CAL'])
        calrec = HS.calibrator_record(cals, raw['Q_CAL'])
        bundle = HS.make_bundle(fam, cfg, h, basem, cals, names, ridge, dict(man, fitted_on='base: all cohort TRAIN rows; calibrators: cohort Q_CAL rows'))
        mdir.mkdir(parents=True, exist_ok=True)
        path = mdir / f'{cfg}.joblib'
        joblib.dump(bundle, path)
        bsha = C.sha256_file(path)
        P = {f'{cfg}__{cal}': {k: HS.calibrate(bundle, cal, raw[k]) for k in M} for cal in HS.CALS}
        b2 = joblib.load(path)
        ident = {cal: all(np.array_equal(HS.bundle_predict(b2, cal, X[m], names), P[f'{cfg}__{cal}'][k]) for k, m in M.items() if k != 'TRAIN') for cal in HS.CALS}
        if not all(ident.values()):
            raise SystemExit(f'reload identity failed: {ident}')
        cands = HS.candidate_names(cfg)
        eligible = {c: bool(ok_fit and all(np.isfinite(P[c][k]).all() for k in ('Q_CAL', 'Q_SELECT'))) for c in cands}
        met = {c: {k: C.evaluate(y[m], P[c][k], g[m], bins=(k == 'Q_SELECT')) for k, m in M.items()} for c in cands}
        elig = {c: (met[c]['Q_SELECT']['brier'], met[c]['Q_SELECT']['logloss']) for c in cands if eligible[c]}
        if not elig:
            raise SystemExit('no eligible candidate (documented failure)')
        chosen, ranking = HS.select_rule(elig)
        seed_met = {}
        if seedraw is not None:
            seed_met = {str(s): {k: C.evaluate(y[M[k]], seedraw[k][j], g[M[k]], bins=False) for k in ('Q_CAL', 'Q_SELECT')} for j, s in enumerate(HS.SEEDS[fam])}
        sel = dict(role=HS.ROLE_TAG, manifest=man, horizon_s=h, cohort=coh, family=fam, config=cfg,
                   rule='calibration-only selection: lowest cohort Q_SELECT match-weighted Brier, then log loss, then name; ineligible excluded',
                   candidates_lexical_tie_order=cands, eligible=eligible, chosen=chosen, chosen_config=cfg, chosen_calibration=HS.split_candidate(chosen)[1], ranking=ranking,
                   select_metrics={c: {k: met[c]['Q_SELECT'][k] for k in ('brier', 'logloss', 'auc', 'intercept', 'slope', 'ece_10bin')} for c in cands},
                   calibrate_metrics={c: {k: met[c]['Q_CAL'][k] for k in ('brier', 'logloss', 'auc')} for c in cands},
                   train_in_sample_metrics_DESCRIPTIVE={c: {k: met[c]['TRAIN'][k] for k in ('brier', 'logloss', 'auc')} for c in cands},
                   seed_metrics_DESCRIPTIVE=seed_met, split_counts=counts, weights=wrec, fit_record=dict(fitrec, seconds_fit=round(fit_s, 1), seconds_predict=round(pred_s, 1)),
                   calibrators=calrec, bundle_sha256={cfg: bsha}, reload_identity={cfg: ident}, oof_provenance=prov,
                   input_names_sha256=C.sha256_json(input_names), n_inputs=len(input_names), seconds_total=round(time.time() - t0, 1),
                   environment=HS.environment(), source_sha256=HS.own_source_hashes(), written_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        if stoprec is not None:
            C.write_json(base / 'internal_stop' / f'{key}.json', dict(role=HS.ROLE_TAG, horizon_s=h, cohort=coh, family=fam, config=cfg, stop_record=stoprec))
        arrays = dict(match=g, s_ms=D['s_ms'], role=D['role'], y=y, p_pre=D['p_pre'], time_minutes=X[:, names.index('time_minutes')])
        full = np.full(len(y), np.nan)
        for k, m in M.items():
            full[m] = raw[k]
        arrays[f'raw__{cfg}'] = full
        if seedraw is not None:
            for j, s in enumerate(HS.SEEDS[fam]):
                fs = np.full(len(y), np.nan)
                for k, m in M.items():
                    fs[m] = seedraw[k][j]
                arrays[f'seedraw__{cfg}__{s}'] = fs
        for c in cands:
            full = np.full(len(y), np.nan)
            for k, m in M.items():
                full[m] = P[c][k]
            arrays[c] = full
        sel['predictions_sha256'] = C.save_npz(base / 'predictions' / f'{key}_trainval.npz', **arrays)
        sha = C.write_json(sel_path, sel)
        st.update('complete', 'selected', chosen=chosen, selection_sha256=sha, seconds=sel['seconds_total'],
                  select_brier={c: round(met[c]['Q_SELECT']['brier'], 6) for c in cands}, next_step='next family / cohort / horizon, then freeze')
        return 0
    except SystemExit as exc:
        HS.log_failure(st.group, exc)
        st.update('failed', 'fit', error=str(exc)[:1000], next_step='inspect (failure retained)')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        HS.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'fit', error=repr(exc)[:1000], next_step='inspect, fix, rerun (failure retained)')
        return 3


if __name__ == '__main__':
    sys.exit(main())
