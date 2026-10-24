"""Stage R (Part B, unsealed): epsilon-Rashomon set of all saved same-input candidates on MAIN_VALIDATION Q_SELECT rows and the group
permutation reliance of every candidate; per group the range over the set (empirical model class reliance)."""
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
import cs20260916_common as CS  # noqa: E402
import fc20260915_common as C  # noqa: E402
import iq20260915_common as Q  # noqa: E402
import bsh20260916_common as B  # noqa: E402

import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    CS.log_command()
    st = CS.Status('rashomon')
    try:
        if not (CS.OUT / 'protocol.json').exists():
            raise SystemExit('protocol.json must precede Part B')
        proto = C.read_json(CS.OUT / 'protocol.json')
        schema = C.read_json(CS.FC / 'q_pre_only_schema.json')
        ridge, names = list(schema['predictor_sets']['ridge']), list(schema['input_names_all'])
        gidx = B.group_index(names, ridge, schema['shap_groups'])
        ridge_ix = [names.index(n) for n in ridge]
        F, Lb, Co = CS.load_parent_set('MAIN_VALIDATION', CS.OUT, 'Part B: Q_SELECT rows (unsealed)')
        out_all = {}
        for coh in CS.COHORTS:
            rows = CS.cohort_rows(F, Lb, Co, coh, sub_roles=('Q_SELECT',))
            if len(rows) != CS.EXPECTED_COUNTS['Q_SELECT'][coh]:
                raise SystemExit(f'{coh}: Q_SELECT rows {len(rows)} != {CS.EXPECTED_COUNTS["Q_SELECT"][coh]}')
            X = np.ascontiguousarray(F['X_input'][rows][:, ridge_ix])
            y = Lb['Y_h90'][rows].astype(int)
            g = F['match'][rows].astype(str)
            s = F['s_ms'][rows]
            w = CS.weights(g)
            perms = [np.random.default_rng(sd).permutation(len(rows)) for sd in CS.PERM_SEEDS]
            cands, t0 = [], time.time()
            for k, (src, fam, cfg, path) in enumerate(CS.candidate_bundles(coh)):
                st.update('running', f'{coh}_{src}_{fam}_{cfg}', processed=k, total=len(proto['part_b']['candidates'][coh]), next_step='permute groups')
                b = joblib.load(path)
                # iq/ta bundles index the 362-name schema; the champion-class base arms index their own 352-name arm matrix
                schema_ok = (b['input_names_all_sha256'] == C.sha256_json(names) and [names[i] for i in b['input_columns']] == ridge) or \
                            (b['input_names_all_sha256'] == C.sha256_json(ridge) and list(b['input_columns']) == list(range(len(ridge))))
                if not schema_ok or list(b['input_names']) != ridge:
                    raise SystemExit(f'{path.name}: input schema differs')
                raw0 = b['base'].raw(X)
                with np.load(CS.candidate_predictions(src, fam, coh), allow_pickle=False) as z:
                    zm, zs, zr = z['match'].astype(str), z['s_ms'], z['role'].astype(str)
                    saved = {cal: z[f'{cfg}__{cal}'] for cal in CS.CALS if f'{cfg}__{cal}' in z.files}
                pos = {kk: i for i, kk in enumerate(zip(zm.tolist(), zs.tolist()))}
                jx = np.asarray([pos[(m, int(v))] for m, v in zip(g.tolist(), s.tolist())])
                rel = {}
                for cal in CS.CALS:
                    p0 = Q.calibrate(b, cal, raw0)
                    ident = bool(cal in saved and np.array_equal(saved[cal][jx], p0))
                    b0 = C.brier_direct(y, p0, w)
                    rel[cal] = dict(brier=float(b0), saved_prediction_identical=ident, reliance={}, reliance_seeds={})
                for gi, gname in enumerate(CS.INPUT_GROUPS):
                    per_seed = {cal: [] for cal in CS.CALS}
                    for perm in perms:
                        Xp = X.copy()
                        Xp[:, gidx[gi]] = X[perm][:, gidx[gi]]
                        rawp = b['base'].raw(Xp)
                        for cal in CS.CALS:
                            per_seed[cal].append(C.brier_direct(y, Q.calibrate(b, cal, rawp), w) - rel[cal]['brier'])
                    for cal in CS.CALS:
                        rel[cal]['reliance'][gname] = float(np.mean(per_seed[cal]))
                        rel[cal]['reliance_seeds'][gname] = [float(v) for v in per_seed[cal]]
                for cal in CS.CALS:
                    cands.append(dict(source=src, family=fam, config=cfg, calibration=cal, name=f'{src}:{fam}:{cfg}__{cal}', bundle_sha256=C.sha256_file(path), **rel[cal]))
                b = None
                st.log(f'{coh} {src} {fam} {cfg}: ' + ' '.join(f'{cal}={rel[cal]["brier"]:.6f}' for cal in CS.CALS) + f' ({time.time() - t0:.0f}s)')
            best = min(c['brier'] for c in cands)
            sets = {}
            for eps in CS.EPSILONS:
                members = [c for c in cands if c['brier'] <= best + eps]
                rng = {}
                for gname in CS.INPUT_GROUPS:
                    vals = [c['reliance'][gname] for c in members]
                    rng[gname] = dict(min=float(min(vals)), max=float(max(vals)), argmin=members[int(np.argmin(vals))]['name'], argmax=members[int(np.argmax(vals))]['name'])
                top = {}
                for c in members:
                    tg = max(c['reliance'], key=c['reliance'].get)
                    top[tg] = top.get(tg, 0) + 1
                orders = [tuple(sorted(c['reliance'], key=c['reliance'].get, reverse=True)) for c in members]
                sets[str(eps)] = dict(epsilon=eps, threshold=float(best + eps), n_members=len(members), members=[c['name'] for c in members],
                                      families={f: sum(1 for c in members if c['family'] == f) for f in ('logit', 'lgbm', 'mlp', 'resmlp')},
                                      reliance_range=rng, most_relied_group_counts=top, distinct_orderings=len(set(orders)),
                                      p_pre_rank_1_fraction=float(np.mean([o[0] == 'prior_win_probability' for o in orders])) if members else None)
            not_ident = [c['name'] for c in cands if not c['saved_prediction_identical']]
            out_all[coh] = dict(cohort=coh, rows=int(len(rows)), matches=int(len(np.unique(g))), best_brier=float(best), best_candidate=min(cands, key=lambda c: c['brier'])['name'],
                                n_candidates=len(cands), candidates=cands, epsilon_sets=sets, saved_prediction_not_identical=not_ident, seconds=round(time.time() - t0, 1))
            C.write_json(CS.OUT / 'rashomon' / f'{coh}.json', dict(role=CS.ROLE_TAG, version=CS.VERSION, protocol_sha256=C.sha256_file(CS.OUT / 'protocol.json'), **out_all[coh],
                                                                 written_at=time.strftime('%Y-%m-%d %H:%M:%S')))
            st.log(f'{coh}: best {out_all[coh]["best_candidate"]} {best:.6f}; eps sets ' + ', '.join(f'{e}:{v["n_members"]}' for e, v in sets.items()) + f'; not identical {len(not_ident)}')
        st.update('complete', 'rashomon', next_step='freeze then Part A')
        return 0
    except SystemExit as exc:
        CS.log_failure(st.group, exc)
        st.update('failed', 'rashomon', error=str(exc)[:1000], next_step='inspect')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        CS.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'rashomon', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
