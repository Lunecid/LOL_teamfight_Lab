"""Stage A: stream the extraction chunks, decompose Delta-logit of the frozen V per row into group / time-related sums, verify the
identities against the parent labels, and aggregate by set, cohort and stratum. --smoke: MAIN_VALIDATION rows of the first 60 chunks
(unsealed) under smoke_validation_only/; full: MAIN_VALIDATION (secondary) and MAIN_TEST (sealed, primary) over all chunks."""
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
import vd20260916_common as V  # noqa: E402
import fc20260915_common as C  # noqa: E402

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402


def summarize(gsum, tsum, abscol, dl, absd):
    """Returns (agg, out): agg(mask, label) writes the statistics of one subset into out."""
    out = {}

    def agg(mask, label):
        n = int(mask.sum())
        if n == 0:
            out[label] = dict(n=0)
            return
        G = gsum[mask]
        d = dl[mask]
        absG = np.abs(G)
        tot = absG.sum(1)
        dominant = np.argmax(absG, axis=1)
        same_sign = np.sign(G[np.arange(n), dominant]) == np.sign(d)
        # sign-deciding block: scanning blocks by descending |g|, the first whose sign equals sign(Delta-logit)
        order = np.argsort(-absG, axis=1)
        match = np.take_along_axis((np.sign(G) == np.sign(d)[:, None]) & (G != 0), order, axis=1)
        has = match.any(axis=1)
        first = np.argmax(match, axis=1)
        decider = np.where(has, np.take_along_axis(order, first[:, None], axis=1)[:, 0], -1)
        single = np.abs(G[np.arange(n), dominant]) >= np.abs(d)
        out[label] = dict(n=n, mean_abs_delta_logit=float(np.mean(np.abs(d))), median_abs_delta_logit=float(np.median(np.abs(d))),
                          mean_abs_delta_V=float(np.mean(absd[mask])), positive_rate=float(np.mean(d > 0)),
                          mean_abs_group={g: float(absG[:, k].mean()) for k, g in enumerate(V.GROUPS)},
                          share_of_sum_abs={g: float(absG[:, k].sum() / max(tot.sum(), 1e-300)) for k, g in enumerate(V.GROUPS)},
                          mean_signed_group={g: float(G[:, k].mean()) for k, g in enumerate(V.GROUPS)},
                          time_related_share_of_sum_abs_columns=float(np.sum(tsum[mask]) / max(np.sum(abscol[mask]), 1e-300)),
                          dominant_group_frequency={g: float(np.mean(dominant == k)) for k, g in enumerate(V.GROUPS)},
                          dominant_group_sign_agrees_with_delta=float(np.mean(same_sign)),
                          sign_deciding_group_frequency={g: float(np.mean(decider == k)) for k, g in enumerate(V.GROUPS)},
                          single_group_exceeds_total_share=float(np.mean(single)))
    return agg, out


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    V.log_command()
    base = V.SMOKE if args.smoke else V.OUT
    st = V.Status('analyze' + ('_smoke' if args.smoke else ''))
    try:
        if not V.frozen_path(base).exists():
            raise SystemExit('frozen_manifest.json required')
        fz = C.read_json(V.frozen_path(base))
        if bool(fz['smoke']) != bool(args.smoke):
            raise SystemExit('frozen manifest mode differs')
        adapter = C.load_v_adapter(V.V_ADAPTER, fz['adapter_sha256'])
        b, b0 = V.beta(adapter)
        names, groups, gidx, tmask = V.column_map(adapter)
        if C.sha256_json(names) != fz['transformed_names_sha256']:
            raise SystemExit('transformed column names differ from the freeze')
        G = np.zeros((len(names), len(V.GROUPS)))
        for k, g in enumerate(V.GROUPS):
            G[gidx[g], k] = 1.0
        sets = ('MAIN_VALIDATION',) if args.smoke else V.SETS
        labels = {s: V.load_labels(s, base, f'labels {s}') for s in sets}
        keys = {}
        for s, (Lb, Co) in labels.items():
            valid = Lb['valid_h90'] == 1
            for i, (m, ss) in enumerate(zip(Lb['match'].astype(str).tolist(), Lb['s'].tolist())):
                if valid[i]:
                    keys[(m, int(ss))] = (s, i)
        V.state_access(base, 'stream pre/post states', sealed=not args.smoke)
        files = V.chunk_files()[370:430] if args.smoke else V.chunk_files()  # chunks are ordered TRAIN, VALIDATION, TEST; 370-429 hold VALIDATION matches
        coll = {s: dict(idx=[], gsum=[], tsum=[], abscol=[], dl=[], ppre=[], ppost=[]) for s in sets}
        colabs = np.zeros(len(names))
        colsigned = np.zeros(len(names))
        ncol = 0
        checks = dict(chunks=0, rows=0, names_equal=True, version_equal=True, p_pre_exact=0, p_post_exact=0, p_pre_rows=0, adapter_final_rows=0, max_abs_sum_minus_decision=0.0,
                      max_abs_champion=0.0, max_abs_logit_diff_minus_decision=0.0, sign_equals_Y=0, sign_rows=0)
        t0 = time.time()
        for fi, f in enumerate(files):
            with np.load(f, allow_pickle=False) as z:
                if str(z['set_id']) != 'MAIN':
                    continue
                checks['names_equal'] &= [str(n) for n in z['names']] == adapter.state_names
                checks['version_equal'] &= str(z['state_version']) == adapter.state_version
                em, es = z['e_match'].astype(str), z['e_s']
                hit = [(i, keys[(m, int(s))]) for i, (m, s) in enumerate(zip(em.tolist(), es.tolist())) if (m, int(s)) in keys]
                if not hit:
                    continue
                ix = np.asarray([h[0] for h in hit])
                Xp, Xe = z['e_X_pre'][ix], z['e_X_post_h90'][ix]
            checks['chunks'] += 1
            checks['rows'] += len(ix)
            Zp, Ze = V.transform(adapter, Xp), V.transform(adapter, Xe)
            cont = b[None, :] * (Ze - Zp)
            dl = V.decision(adapter, Xe) - V.decision(adapter, Xp)
            checks['max_abs_sum_minus_decision'] = max(checks['max_abs_sum_minus_decision'], float(np.max(np.abs(cont.sum(1) - dl))))
            checks['max_abs_champion'] = max(checks['max_abs_champion'], float(np.max(np.abs(cont[:, gidx['champion_identity']]))) if len(gidx['champion_identity']) else 0.0)
            p_pre = adapter.predict_matrix(Xp, adapter.state_names, adapter.state_version)
            p_post = adapter.predict_matrix(Xe, adapter.state_names, adapter.state_version)
            checks['max_abs_logit_diff_minus_decision'] = max(checks['max_abs_logit_diff_minus_decision'], float(np.max(np.abs((V.logit(p_post) - V.logit(p_pre)) - dl))))
            gs = cont @ G
            ts = np.abs(cont[:, tmask]).sum(1)
            ac = np.abs(cont).sum(1)
            colabs += np.abs(cont).sum(0)
            colsigned += cont.sum(0)
            ncol += len(ix)
            sets_of = [h[1][0] for h in hit]
            lis = np.asarray([h[1][1] for h in hit])
            for s in sets:
                m = np.asarray([x == s for x in sets_of])
                if not m.any():
                    continue
                Lb = labels[s][0]
                li = lis[m]
                checks['p_pre_rows'] += int(m.sum())
                checks['p_pre_exact'] += int(np.sum(p_pre[m] == Lb['p_pre'][li]))
                checks['p_post_exact'] += int(np.sum(p_post[m] == Lb['p_post_h90'][li]))
                checks['adapter_final_rows'] += int(np.sum(Lb['adapter_id'][li] == 'final'))
                checks['sign_equals_Y'] += int(np.sum((dl[m] > 0).astype(int) == Lb['Y_h90'][li].astype(int)))
                checks['sign_rows'] += int(m.sum())
                coll[s]['idx'].append(li)
                coll[s]['gsum'].append(gs[m])
                coll[s]['tsum'].append(ts[m])
                coll[s]['abscol'].append(ac[m])
                coll[s]['dl'].append(dl[m])
                coll[s]['ppre'].append(p_pre[m])
                coll[s]['ppost'].append(p_post[m])
            if (fi + 1) % 50 == 0:
                st.update('running', 'stream', processed=fi + 1, total=len(files), rows=checks['rows'], next_step='aggregate')
        checks['seconds_stream'] = round(time.time() - t0, 1)
        checks['pass'] = bool(checks['names_equal'] and checks['version_equal'] and checks['p_pre_exact'] == checks['p_post_exact'] == checks['p_pre_rows']
                              and checks['adapter_final_rows'] == checks['p_pre_rows'] and checks['max_abs_sum_minus_decision'] < 1e-9 and checks['max_abs_champion'] == 0.0
                              and checks['max_abs_logit_diff_minus_decision'] < 1e-8 and checks['sign_equals_Y'] == checks['sign_rows'])
        results = dict(role=V.ROLE_TAG, version=V.VERSION, smoke=bool(args.smoke), frozen_manifest_sha256=C.sha256_file(V.frozen_path(base)), adapter_sha256=fz['adapter_sha256'],
                       transformed_columns=len(names), groups=list(V.GROUPS), group_sizes={g: int(len(gidx[g])) for g in V.GROUPS}, time_related_columns=int(tmask.sum()),
                       checks=checks, sets={})
        order = np.argsort(-colabs / max(ncol, 1))
        results['top_columns_by_mean_abs_contribution'] = [dict(column=names[j], group=groups[j], time_related=bool(tmask[j]), mean_abs=float(colabs[j] / max(ncol, 1)),
                                                                mean_signed=float(colsigned[j] / max(ncol, 1)), beta=float(b[j])) for j in order[:V.TOP_COLUMNS]]
        for s in sets:
            if not coll[s]['idx']:
                results['sets'][s] = dict(rows=0)
                continue
            li = np.concatenate(coll[s]['idx'])
            gsum = np.vstack(coll[s]['gsum'])
            tsum = np.concatenate(coll[s]['tsum'])
            abscol = np.concatenate(coll[s]['abscol'])
            dl = np.concatenate(coll[s]['dl'])
            ppre, ppost = np.concatenate(coll[s]['ppre']), np.concatenate(coll[s]['ppost'])
            Lb, Co = labels[s]
            valid_rows = int((Lb['valid_h90'] == 1).sum())
            if not args.smoke and len(li) != valid_rows:
                raise SystemExit(f'{s}: {len(li)} rows collected but {valid_rows} valid rows expected')
            if len(set(li.tolist())) != len(li):
                raise SystemExit(f'{s}: duplicate rows collected')
            absd = np.abs(ppost - ppre)
            coh = Co['cohort'][li]
            same_frame = Lb['post_snapshot_h90'][li] == Lb['pre_snapshot'][li]
            band = V.band_index(Lb['s'][li] / 60000.)
            strat = V.delta_stratum(absd)
            agg, out = summarize(gsum, tsum, abscol, dl, absd)
            for coh_name in ('T', 'N'):
                cm = coh == V.COHORT_CODE[coh_name]
                agg(cm, f'{coh_name}:all')
                agg(cm & same_frame, f'{coh_name}:same_frame')
                agg(cm & ~same_frame, f'{coh_name}:new_frame')
                for k in range(len(V.DELTA_STRATA)):
                    agg(cm & (strat == k), f'{coh_name}:absdelta_{V.stratum_name(k)}')
                for k, (lo, hi) in enumerate(V.BANDS):
                    agg(cm & (band == k), f'{coh_name}:time_{lo}_{hi if hi < 1000 else "inf"}')
            agg(np.ones(len(li), dtype=bool), 'E:all')
            results['sets'][s] = dict(rows=int(len(li)), matches=int(len(np.unique(Lb['match'][li]))), valid_rows_expected=valid_rows,
                                      same_frame_share={c: float(np.mean(same_frame[coh == V.COHORT_CODE[c]])) for c in ('T', 'N')}, strata=out)
            C.save_npz(base / 'rows' / f'{s}_h90_decomposition.npz', label_row_index=li, match=Lb['match'][li], s_ms=Lb['s'][li], cohort=coh, same_frame=same_frame,
                       delta_logit=dl, group_sums=gsum, group_names=np.array(V.GROUPS), time_related_abs_sum=tsum, abs_column_sum=abscol, p_pre=ppre, p_post=ppost,
                       y=Lb['Y_h90'][li], abs_delta_V=absd, pre_minutes=Lb['s'][li] / 60000.)
            st.update('running', f'aggregated_{s}', rows=int(len(li)), next_step='next set')
        results['evaluated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        sha = C.write_json(base / 'results.json', results)
        st.update('complete' if checks['pass'] else 'failed', 'analyze', results_sha256=sha, checks=checks, next_step='snapshot after, post-run checks, report')
        return 0 if checks['pass'] else 1
    except SystemExit as exc:
        V.log_failure(st.group, exc)
        st.update('failed', 'analyze', error=str(exc)[:1000], next_step='inspect (failure retained)')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        V.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'analyze', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
