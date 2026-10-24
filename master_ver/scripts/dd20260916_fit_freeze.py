"""Stage F+Z: base-arm refit (logit, LightGBM; iq h90 configurations) on the dev TRAIN population per cohort, calibrators on dev Q_CAL,
calibration-only selection on dev Q_SELECT, then the freeze (before any sealed set is rebuilt or opened)."""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '4'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MKL_CBWR', 'AVX2,STRICT')

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dd20260916_common as DD  # noqa: E402
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402
import iq20260915_common as Q  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402


def fit_one(coh, fam, cfg, D, ridge, st, vman):
    names = D['names']
    X, y, g = D['X'], D['y'], D['g']
    M = {k: D['role'] == k for k in ('TRAIN', 'Q_CAL', 'Q_SELECT')}
    counts = {k: dict(rows=int(m.sum()), matches=int(len(np.unique(g[m])))) for k, m in M.items()}
    if len(set(g[M['TRAIN']]) & set(g[M['Q_CAL']])) or len(set(g[M['TRAIN']]) & set(g[M['Q_SELECT']])) or len(set(g[M['Q_CAL']]) & set(g[M['Q_SELECT']])):
        raise SystemExit('TRAIN / Q_CAL / Q_SELECT match overlap')
    ridge_ix = [names.index(n) for n in ridge]
    if any(int((~np.isfinite(X[m][:, ridge_ix])).sum()) for m in M.values()):
        raise SystemExit('non-finite inputs among valid rows')
    trainfold = np.char.startswith(D['sr_true'], 'fold')
    prov = DD.oof_provenance(g[trainfold], D['sr_true'][trainfold], D['adapter_id'][trainfold], D['adapter_sha256'][trainfold], vman)
    if not (prov['rows'] == prov['adapter_id_equals_own_heldout_fold'] == prov['adapter_sha_equals_manifest_oof_hash'] == prov['sub_role_equals_hash_fold']):
        raise SystemExit(f'own-match OOF provenance failed: {prov}')
    W = {k: DD.weights(g[m]) for k, m in M.items()}
    wrec = {k: DD.weight_record(g[m], W[k]) for k, m in M.items()}
    Xa = {k: np.ascontiguousarray(X[m][:, ridge_ix]) for k, m in M.items()}
    man = dict(role=DD.ROLE_TAG, version=DD.VERSION, cohort=coh, family=fam, config=cfg, definition=DD.DEFINITIONS['dev'], population='dev (rebuilt changed matches + parent rows)',
               output_is='q probability of generated label Y_h90 within the cohort (NOT a V win probability)', protocol_sha256=C.sha256_file(DD.OUT / 'protocol.json'),
               row_source_counts={k: {s: int(n) for s, n in zip(*np.unique(D['row_source'][m], return_counts=True))} for k, m in M.items()})
    t0 = time.time()
    basem, ok_fit, fitrec, stoprec = DD.fit_family(fam, cfg, list(ridge), Xa['TRAIN'], y[M['TRAIN']], g[M['TRAIN']], W['TRAIN'], log=st.log)
    fit_s = time.time() - t0
    seedraw = {k: basem.seed_raw(Xa[k]) for k in M} if fam in DD.SEEDED else None
    raw = {k: seedraw[k].mean(axis=0) for k in M} if seedraw is not None else {k: basem.raw(Xa[k]) for k in M}
    cals = K.fit_calibrators(raw['Q_CAL'], y[M['Q_CAL']], W['Q_CAL'])
    bundle = DD.make_bundle(fam, cfg, basem, cals, names, ridge, dict(man, fitted_on='base: all cohort dev TRAIN rows; calibrators: cohort dev Q_CAL rows'))
    mdir = DD.OUT / 'models' / coh / fam
    mdir.mkdir(parents=True, exist_ok=True)
    path = mdir / f'{cfg}.joblib'
    joblib.dump(bundle, path)
    P = {f'{cfg}__{cal}': {k: DD.calibrate(bundle, cal, raw[k]) for k in M} for cal in DD.CALS}
    b2 = joblib.load(path)
    ident = {cal: all(np.array_equal(DD.bundle_predict(b2, cal, X[m], names), P[f'{cfg}__{cal}'][k]) for k, m in M.items() if k != 'TRAIN') for cal in DD.CALS}
    if not all(ident.values()):
        raise SystemExit(f'reload identity failed: {ident}')
    cands = DD.candidate_names(cfg)
    eligible = {c: bool(ok_fit and all(np.isfinite(P[c][k]).all() for k in ('Q_CAL', 'Q_SELECT'))) for c in cands}
    met = {c: {k: C.evaluate(y[m], P[c][k], g[m], bins=(k == 'Q_SELECT')) for k, m in M.items()} for c in cands}
    elig = {c: (met[c]['Q_SELECT']['brier'], met[c]['Q_SELECT']['logloss']) for c in cands if eligible[c]}
    if not elig:
        raise SystemExit('no eligible candidate')
    chosen, ranking = DD.select_rule(elig)
    key = f'{fam}_{coh}'
    if stoprec is not None:
        C.write_json(DD.OUT / 'internal_stop' / f'{key}.json', dict(role=DD.ROLE_TAG, cohort=coh, family=fam, config=cfg, stop_record=stoprec))
    arrays = dict(match=g, s_ms=D['s_ms'], role=D['role'], y=y, p_pre=D['p_pre'], row_source=D['row_source'], time_minutes=X[:, names.index('time_minutes')])
    full = np.full(len(y), np.nan)
    for k, m in M.items():
        full[m] = raw[k]
    arrays[f'raw__{cfg}'] = full
    if seedraw is not None:
        for j, s in enumerate(DD.SEEDS[fam]):
            fs = np.full(len(y), np.nan)
            for k, m in M.items():
                fs[m] = seedraw[k][j]
            arrays[f'seedraw__{cfg}__{s}'] = fs
    for c in cands:
        full = np.full(len(y), np.nan)
        for k, m in M.items():
            full[m] = P[c][k]
        arrays[c] = full
    psha = C.save_npz(DD.OUT / 'predictions' / f'{key}_trainval.npz', **arrays)
    sel = dict(role=DD.ROLE_TAG, manifest=man, cohort=coh, family=fam, config=cfg, rule='calibration-only selection on dev Q_SELECT: Brier, then log loss, then name',
               candidates_lexical_tie_order=cands, eligible=eligible, chosen=chosen, chosen_calibration=DD.split_candidate(chosen)[1], ranking=ranking,
               select_metrics={c: {k: met[c]['Q_SELECT'][k] for k in ('brier', 'logloss', 'auc', 'intercept', 'slope', 'ece_10bin')} for c in cands},
               calibrate_metrics={c: {k: met[c]['Q_CAL'][k] for k in ('brier', 'logloss', 'auc')} for c in cands}, split_counts=counts, weights=wrec,
               fit_record=dict(fitrec, seconds_fit=round(fit_s, 1)), calibrators=DD.calibrator_record(cals, raw['Q_CAL']), bundle_sha256={cfg: C.sha256_file(path)},
               reload_identity={cfg: ident}, oof_provenance=prov, predictions_sha256=psha, input_names_sha256=C.sha256_json(list(ridge)), n_inputs=len(ridge),
               environment=DD.environment(), source_sha256=DD.own_source_hashes(), written_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    C.write_json(DD.OUT / 'selection' / f'{key}.json', sel)
    st.log(f'{key}: chosen {chosen} select Brier {met[chosen]["Q_SELECT"]["brier"]:.6f} rows {counts}')
    return sel


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    DD.log_command()
    st = DD.Status('fit_freeze')
    try:
        if not (DD.OUT / 'protocol.json').exists():
            raise SystemExit('protocol.json must precede fitting')
        if DD.frozen_path(DD.OUT).exists():
            raise SystemExit('frozen manifest exists (never overwritten)')
        for s in ('MAIN_TRAIN', 'MAIN_VALIDATION'):
            if not (DD.OUT / 'rebuild' / f'{s}.json').exists():
                raise SystemExit(f'rebuild of {s} must precede fitting')
        ct = DD.OUT / 'contract_tests' / 'result.json'
        if not ct.exists() or not C.read_json(ct).get('passed'):
            raise SystemExit('contract tests must pass before fitting')
        access = [json.loads(l) for l in (DD.OUT / 'access_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()] if (DD.OUT / 'access_log.jsonl').exists() else []
        if any(a.get('sealed') for a in access):
            raise SystemExit('sealed access before freeze')
        schema = C.read_json(DD.FC / 'q_pre_only_schema.json')
        ridge = list(schema['predictor_sets']['ridge'])
        proto = C.read_json(DD.OUT / 'protocol.json')
        reg = proto['fixed_configurations']['registry']
        vman = C.read_json(DD.FC / 'v_models_manifest.json')
        sels = {}
        for coh in DD.COHORTS:
            st.update('running', f'load_{coh}', next_step='fit')
            D = DD.load_trainval(DD.OUT, coh)
            if D['names'] != schema['input_names_all']:
                raise SystemExit('dev population schema differs from the parent q schema')
            keys = list(zip(D['g'].tolist(), D['s_ms'].tolist()))
            if len(set(keys)) != len(keys):
                raise SystemExit('duplicated keys')
            for fam in DD.FAMILIES:
                key = f'{fam}_{coh}'
                if (DD.OUT / 'selection' / f'{key}.json').exists():
                    sels[key] = C.read_json(DD.OUT / 'selection' / f'{key}.json')
                    continue
                st.update('running', f'fit_{key}', next_step='calibrate / select')
                sels[key] = fit_one(coh, fam, reg[coh][fam], D, ridge, st, vman)
            D = None
        # freeze
        st.update('running', 'freeze', next_step='sealed rebuild + evaluation')
        checks, selected, bundle_sha, pred_sha, stop_sha, selrec = {}, {}, {}, {}, {}, {}
        for coh in DD.COHORTS:
            D = DD.load_trainval(DD.OUT, coh)
            names = D['names']
            ms = D['role'] == 'Q_SELECT'
            ws = DD.weights(D['g'][ms])
            for fam in DD.FAMILIES:
                key = f'{fam}_{coh}'
                sel = sels[key]
                cfg = sel['config']
                p = DD.OUT / 'models' / coh / fam / f'{cfg}.joblib'
                bundle_sha[key] = C.sha256_file(p)
                pp = DD.OUT / 'predictions' / f'{key}_trainval.npz'
                pred_sha[key] = C.sha256_file(pp)
                with np.load(pp, allow_pickle=False) as z:
                    Z = {k: z[k] for k in z.files}
                ck = dict(bundle_hash=bundle_sha[key] == sel['bundle_sha256'][cfg], predictions_hash=pred_sha[key] == sel['predictions_sha256'],
                          rows=bool(np.array_equal(Z['match'].astype(str), D['g']) and np.array_equal(Z['s_ms'], D['s_ms']) and np.array_equal(Z['y'], D['y'])))
                direct = {c: (C.brier_direct(D['y'][ms], Z[c][ms], ws), C.logloss_direct(D['y'][ms], Z[c][ms], ws)) for c in DD.candidate_names(cfg) if sel['eligible'][c]}
                ck['selection_rule_reproduced'] = DD.select_rule(direct)[0] == sel['chosen']
                cal = DD.split_candidate(sel['chosen'])[1]
                b = joblib.load(p)
                for role in ('Q_CAL', 'Q_SELECT'):
                    m = D['role'] == role
                    ck[f'reload_selected_identical_{role}'] = bool(np.array_equal(DD.bundle_predict(b, cal, D['X'][m], names), Z[sel['chosen']][m]))
                b = Z = None
                checks[key] = ck
                selrec[key] = dict(sha256=C.sha256_file(DD.OUT / 'selection' / f'{key}.json'), chosen=sel['chosen'], written_at=sel['written_at'])
                selected[key] = dict(family=fam, cohort=coh, chosen=sel['chosen'], config=cfg, calibration=cal, bundle=f'models/{coh}/{fam}/{cfg}.joblib', bundle_sha256=bundle_sha[key],
                                     select_brier=sel['select_metrics'][sel['chosen']]['brier'])
                if (DD.OUT / 'internal_stop' / f'{key}.json').exists():
                    stop_sha[key] = C.sha256_file(DD.OUT / 'internal_stop' / f'{key}.json')
            D = None
        failed = [f'{k}:{n}' for k, ck in checks.items() for n, ok in ck.items() if not ok]
        if failed:
            raise SystemExit(f'pre-freeze checks failed: {failed}')
        reg_now, winners, reg_src = DD.registry()
        parent_sha = {f'{s}_{c}': C.sha256_file(DD.IQ / 'eval' / 'predictions' / f'{s}_h{DD.H}_{c}.npz') for s in DD.EVAL_SETS for c in DD.COHORTS}
        parent_bundles = {k: C.sha256_file(DD.IQ / v['bundle']) for k, v in winners.items()}
        for k, v in winners.items():
            if parent_bundles[k] != v['bundle_sha256']:
                raise SystemExit(f'parent bundle hash differs from the iq freeze: {k}')
        frozen = dict(role=DD.ROLE_TAG, version=DD.VERSION, protocol_sha256=C.sha256_file(DD.OUT / 'protocol.json'), contract_tests_result_sha256=C.sha256_file(ct),
                      census_sha256=C.sha256_file(DD.OUT / 'census.json'), rebuild_summaries_sha256={s: C.sha256_file(DD.OUT / 'rebuild' / f'{s}.json') for s in ('MAIN_TRAIN', 'MAIN_VALIDATION')},
                      fixture_check_sha256=C.sha256_file(DD.OUT / 'rebuild' / 'fixture_check.json'), frozen_check_sha256=C.sha256_file(DD.OUT / 'redetect' / 'frozen_check.json'),
                      selections=selrec, family_winners=selected, bundle_sha256=bundle_sha, trainval_predictions_sha256=pred_sha, internal_stop_sha256=stop_sha,
                      parent_references=dict(**reg_src, iq_winners=winners, iq_winner_bundle_sha256_now=parent_bundles, parent_predictions_sha256=parent_sha,
                                             cohort_role_frozen_manifest_sha256=C.sha256_file(DD.CR / 'frozen_manifest.json'), note='file hashes only; parent arrays are opened after this freeze'),
                      pre_freeze_checks=checks, sealed_accesses_before_freeze=len([a for a in access if a.get('sealed')]), source_sha256=DD.own_source_hashes(),
                      frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                      statement='Refit selections frozen on the dev TRAIN/VALIDATION population. TEST / external sets are rebuilt and opened only after this file exists; no refit or reselection afterwards.')
        frozen['frozen_files_sha256'] = {**{v['bundle']: v['bundle_sha256'] for v in selected.values()}, **{f'selection/{k}.json': v['sha256'] for k, v in selrec.items()},
                                        **{f'predictions/{k}_trainval.npz': v for k, v in pred_sha.items()}, **{f'internal_stop/{k}.json': v for k, v in stop_sha.items()},
                                        'protocol.json': frozen['protocol_sha256'], 'census.json': frozen['census_sha256']}
        sha = C.write_json(DD.frozen_path(DD.OUT), frozen)
        st.update('complete', 'frozen', frozen_manifest_sha256=sha, winners={k: v['chosen'] for k, v in selected.items()}, next_step='rebuild sealed sets, evaluate')
        return 0
    except SystemExit as exc:
        DD.log_failure(st.group, exc)
        st.update('failed', 'fit_freeze', error=str(exc)[:1000], next_step='inspect')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        DD.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'fit_freeze', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
