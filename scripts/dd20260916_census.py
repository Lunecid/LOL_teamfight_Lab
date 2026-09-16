"""Stage C: compare the dev-definition exposures with the parent exposures per set (matches changed, rows common / removed / added),
the cohort composition of dev rows (detector counts) and, for the unsealed sets only, the affected h90-valid T / N parent rows.
Writes census.json and changed/<set>.json (dev exposure rows of every changed match)."""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dd20260916_common as DD  # noqa: E402
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402

import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402


def cohort_of(cb, cr):
    n_min, known, cohort, fine = K.scale_classes(np.asarray(cb), np.asarray(cr), np.zeros(len(cb), dtype=bool))
    return cohort


def main():
    DD.log_command()
    st = DD.Status('census')
    try:
        if not (DD.OUT / 'protocol.json').exists():
            raise SystemExit('protocol.json must precede the census')
        dev_main = DD.read_redetect('MAIN', 'dev')
        out = dict(version=DD.VERSION, definition=DD.DEFINITIONS['dev'], sets={}, written_at=None)
        for set_name in DD.ALL_SETS:
            st.update('running', f'census_{set_name}', next_step='next set')
            rows = DD.set_matches(set_name)
            ids = [r[0] for r in rows]
            par, ppath = DD.parent_exposures(set_name)
            dev = dev_main if set_name.startswith('MAIN') else DD.read_redetect(set_name, 'dev')
            changed, common, removed, added, n_par, n_dev = [], 0, 0, 0, 0, 0
            dev_rows_changed = {}
            dev_cohort = []
            missing_dev = 0
            for mid in ids:
                p = set(par.get(mid, []))
                d_rows = dev.get(mid)
                if d_rows is None:
                    missing_dev += 1
                    d_rows = []
                d = set(tuple(r[k] for k in DD.EXPO_FIELDS) for r in d_rows)
                n_par += len(p)
                n_dev += len(d)
                common += len(p & d)
                removed += len(p - d)
                added += len(d - p)
                for r in d_rows:
                    dev_cohort.append((int(r['cluster_blue']), int(r['cluster_red'])))
                if p != d:
                    changed.append(mid)
                    dev_rows_changed[mid] = [{k: r[k] for k in DD.EXPO_FIELDS + ('cluster_blue', 'cluster_red', 'present_blue', 'present_red')} for r in d_rows]
            dc = cohort_of([c[0] for c in dev_cohort], [c[1] for c in dev_cohort]) if dev_cohort else np.zeros(0)
            rec = dict(matches=len(ids), matches_changed=len(changed), matches_changed_frac=len(changed) / max(1, len(ids)), matches_missing_in_dev=missing_dev,
                       rows_parent=n_par, rows_dev=n_dev, rows_common=common, rows_removed=removed, rows_added=added,
                       rows_changed_frac_of_parent=(removed) / max(1, n_par), dev_rows_cohort=dict(T=int((dc == 1).sum()), N=int((dc == 0).sum()), unknown=int((dc == -1).sum())),
                       parent_exposures_sha256=C.sha256_file(ppath))
            if not DD.is_sealed(set_name):
                F, Lb, Co = DD.load_parent_set(set_name, DD.OUT, f'census: affected valid rows of {set_name} (unsealed)')
                valid = Lb['valid_h90'] == 1
                keys = set()
                for mid in changed:
                    for t in par.get(mid, []):
                        keys.add((t[0], int(t[2])))
                aff = np.asarray([(m, int(s)) in keys for m, s in zip(Lb['match'].astype(str).tolist(), Lb['s'].tolist())])
                for coh in DD.COHORTS:
                    m = valid & (Co['cohort'] == DD.COHORT_CODE[coh])
                    rec[f'parent_valid_{coh}_rows'] = int(m.sum())
                    rec[f'parent_valid_{coh}_rows_in_changed_matches'] = int((m & aff).sum())
                    rec[f'parent_valid_{coh}_frac_in_changed_matches'] = float((m & aff).sum() / max(1, m.sum()))
                # detector cluster counts vs parent n_min on the frozen check sample (TRAIN only)
                if set_name == 'MAIN_TRAIN' and (DD.OUT / 'redetect' / 'frozen_check.json').exists():
                    fc = C.read_json(DD.OUT / 'redetect' / 'frozen_check.json')
                    fr = {}
                    import csv
                    with open(DD.OUT / fc['file'], encoding='utf-8', newline='') as f:
                        for r in csv.DictReader(f):
                            fr[(r['match'], int(r['s']))] = min(int(r['cluster_blue']), int(r['cluster_red']))
                    nm = Co['n_min']
                    ix = [(i, fr[(m, int(s))]) for i, (m, s) in enumerate(zip(Lb['match'].astype(str).tolist(), Lb['s'].tolist())) if (m, int(s)) in fr]
                    known = [(i, v) for i, v in ix if Co['scale_known'][i] == 1]
                    rec['frozen_check_cluster_n_min_equals_parent'] = dict(rows=len(known), equal=int(sum(int(nm[i]) == v for i, v in known)))
                F = Lb = Co = None
            out['sets'][set_name] = rec
            C.write_json(DD.OUT / 'changed' / f'{set_name}.json', dict(set=set_name, matches=changed, rows=dev_rows_changed))
            st.log(f'{set_name}: matches {len(ids)} changed {len(changed)} ({rec["matches_changed_frac"]:.4f}); rows parent {n_par} dev {n_dev} common {common} removed {removed} added {added}')
        out['written_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        out['redetect_files_sha256'] = {p.name: C.sha256_file(p) for p in sorted((DD.OUT / 'redetect').glob('*_dev.csv'))}
        sha = C.write_json(DD.OUT / 'census.json', out)
        st.update('complete', 'census', census_sha256=sha, next_step='rebuild MAIN_TRAIN / MAIN_VALIDATION')
        return 0
    except SystemExit as exc:
        DD.log_failure(st.group, exc)
        st.update('failed', 'census', error=str(exc)[:1000], next_step='inspect')
        return 2
    except Exception as exc:
        st.log(traceback.format_exc())
        DD.log_failure(st.group, repr(exc), traceback=traceback.format_exc()[-3000:])
        st.update('failed', 'census', error=repr(exc)[:1000], next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
