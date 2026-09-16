"""Stage D: re-run the parent detector code path on every match of a set under one definition (frozen | dev).
--definition frozen --limit N: reproduction check on the first N TRAIN matches against the parent exposures (must be exact).
Reads cache timelines only (no outcomes, labels or V outputs); writes redetect/<set>_<definition>.csv and a per-match json."""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dd20260916_common as DD  # noqa: E402
import fc20260915_common as C  # noqa: E402

import argparse  # noqa: E402
import csv  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

FIELDS = ('match', 'role', 'sub_role') + DD.EXPO_FIELDS + ('cluster_blue', 'cluster_red', 'present_blue', 'present_red', 'ref_cluster_blue', 'ref_cluster_red')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='MAIN', help='MAIN (all roles) or EXT_<name>')
    ap.add_argument('--definition', choices=sorted(DD.DEFINITIONS), required=True)
    ap.add_argument('--limit', type=int, default=0, help='frozen check: first N TRAIN matches only')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    DD.log_command()
    st = DD.Status(f'redetect_{args.set}_{args.definition}' + (f'_first{args.limit}' if args.limit else ''))
    try:
        if not (DD.OUT / 'protocol.json').exists():
            raise SystemExit('protocol.json must precede re-detection')
        out_csv = DD.redetect_path(args.set, args.definition)
        if out_csv.exists() and not args.limit:
            st.update('complete', 'resume_skip', note=f'{out_csv.name} exists', next_step='census')
            return 0
        if args.set == 'MAIN':
            rows = [(r['match_id'], r['role'], DD.sub_role_of(r['role'], r['match_id']), r['patch'], str(C.CACHE_MAIN)) for r in DD.main_manifest()]
            if args.limit:
                rows = [r for r in rows if r[1] == 'TRAIN'][:args.limit]
        else:
            rows = DD.set_matches(args.set)
            if args.limit:
                raise SystemExit('--limit is a TRAIN-only reproduction check (MAIN)')
        cache_dirs = sorted(set(r[4] for r in rows))
        if len(cache_dirs) != 1:
            raise SystemExit('one cache directory per set expected')
        cache_dir = cache_dirs[0]
        meta = {r[0]: r for r in rows}
        matches = [r[0] for r in rows]
        chunks = [matches[i:i + 200] for i in range(0, len(matches), 200)]
        st.update('running', 'detect', processed=0, total=len(matches), workers=args.workers, definition=DD.DEFINITIONS[args.definition], next_step='write csv')
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        recs = {}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as ex:
            futs = [ex.submit(DD.detect_chunk, dict(definition=args.definition, cache_dir=cache_dir, matches=c)) for c in chunks]
            done = 0
            for fut in as_completed(futs):
                for rec in fut.result():
                    recs[rec['match']] = rec
                done += 1
                if done % 20 == 0 or done == len(futs):
                    rate = len(recs) / max(1e-9, time.time() - t0)
                    st.update('running', 'detect', processed=len(recs), total=len(matches), rate_per_s=round(rate, 1), eta_s=round((len(matches) - len(recs)) / max(rate, 1e-9)), next_step='write csv')
        settings = DD.detector_settings(args.definition)
        target = out_csv if not args.limit else DD.OUT / 'redetect' / f'{args.set}_{args.definition}_first{args.limit}.csv'
        target.parent.mkdir(parents=True, exist_ok=True)
        n_rows, errors = 0, 0
        with open(target, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for mid in matches:
                rec = recs[mid]
                errors += bool(rec['error'])
                for x in rec['rows']:
                    row = {k: x[k] for k in FIELDS[3:]}
                    row.update(match=mid, role=meta[mid][1], sub_role=meta[mid][2])
                    w.writerow(row)
                    n_rows += 1
        summary = dict(set=args.set, definition=args.definition, constants=DD.DEFINITIONS[args.definition], detector_settings=settings, matches=len(matches),
                       loaded=sum(r['loaded'] for r in recs.values()), errors=errors, rows=n_rows, seconds=round(time.time() - t0, 1), cache_dir=cache_dir,
                       file=str(target.relative_to(DD.OUT)), file_sha256=C.sha256_file(target), per_match={m: dict(loaded=r['loaded'], error=r['error'], rows=len(r['rows'])) for m, r in recs.items() if r['error'] or not r['loaded']},
                       written_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        if args.limit:
            par, ppath = DD.parent_exposures('MAIN_TRAIN')
            mism = [m for m in matches if [tuple(x[k] for k in DD.EXPO_FIELDS) for x in recs[m]['rows']] != par.get(m, [])]
            summary.update(parent_exposures_sha256=C.sha256_file(ppath), matches_compared=len(matches), mismatching_matches=len(mism), mismatches=mism[:20],
                           exposure_rows_parent=sum(len(par.get(m, [])) for m in matches),
                           cluster_counts_unknown_rows=sum(1 for m in matches for x in recs[m]['rows'] if x['cluster_blue'] < 0 or x['cluster_red'] < 0))
            C.write_json(DD.OUT / 'redetect' / 'frozen_check.json', summary)
        else:
            C.write_json(DD.OUT / 'redetect' / f'{args.set}_{args.definition}.json', summary)
        st.update('complete', 'written', rows=n_rows, errors=errors, mismatching=summary.get('mismatching_matches'), seconds=summary['seconds'], next_step='census')
        return 0 if not summary.get('mismatching_matches') else 1
    except SystemExit as exc:
        DD.log_failure(st.group, exc)
        st.update('failed', 'redetect', error=str(exc)[:1000], next_step='inspect')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        DD.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'redetect', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
