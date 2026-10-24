"""Read-only integrity snapshot of parents this run must not modify (phase before / after; diff written after)."""
from __future__ import annotations

import os

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import iq20260915_snapshot as S  # noqa: E402
import vd20260916_common as V  # noqa: E402

OWN_DOCS = ('docs/CLAUDE_V_MECHANISM_20260916.md',)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('before', 'after', 'final'), required=True)
    args = ap.parse_args()
    if args.phase == 'before' and (V.OUT / 'integrity' / 'snapshot_before.json').exists():
        raise SystemExit('snapshot_before.json exists (never overwritten)')
    V.log_command()
    t0 = time.time()
    snap = dict(phase=args.phase, taken_at=time.strftime('%Y-%m-%d %H:%M:%S'), sha256_full_corpus_read_targets=S.fc_read_targets(),
                sha256_extract_states={p.name: C.sha256_file(p) for p in V.chunk_files()}, sha256_cohort_role_cohorts={p.name: C.sha256_file(p) for p in sorted((V.CR / 'cohorts').glob('*'))},
                inventory_full_corpus=S.inventory(V.FC),
                sha256_parent_scripts={p.name: C.sha256_file(p) for p in sorted((C.ROOT / 'scripts').glob('*.py')) if not p.name.startswith('vd20260916_')},
                sha256_parent_tests={p.name: C.sha256_file(p) for p in sorted((C.ROOT / 'tests').glob('*.py')) if not p.name.startswith('test_vd20260916_')},
                inventory_docs=S.inventory(C.ROOT / 'docs', skip=OWN_DOCS), pycache_scripts=S.inventory(C.ROOT / 'scripts' / '__pycache__'),
                pycache_tests=S.inventory(C.ROOT / 'tests' / '__pycache__'))
    snap['seconds'] = round(time.time() - t0, 1)
    snap['counts'] = {k: len(v) for k, v in snap.items() if isinstance(v, dict)}
    sha = C.write_json(V.OUT / 'integrity' / f'snapshot_{args.phase}.json', snap)
    print(f'{args.phase}: counts {snap["counts"]} in {snap["seconds"]} s sha256 {sha}')
    if args.phase in ('after', 'final'):
        before = C.read_json(V.OUT / 'integrity' / 'snapshot_before.json')
        diff = {}
        for key, a in before.items():
            if key in ('phase', 'taken_at', 'seconds', 'counts'):
                continue
            b = snap[key]
            diff[key] = dict(checked=len(a), changed=sorted(k for k in a if k in b and a[k] != b[k])[:50], removed=sorted(k for k in a if k not in b)[:50], added=sorted(k for k in b if k not in a)[:50])
            diff[key]['equal'] = not (diff[key]['changed'] or diff[key]['removed'] or diff[key]['added'])
        diff['all_equal'] = all(v['equal'] for v in diff.values() if isinstance(v, dict))
        C.write_json(V.OUT / 'integrity' / ('snapshot_diff.json' if args.phase == 'after' else 'snapshot_final_diff.json'), diff)
        print({k: (v['equal'] if isinstance(v, dict) else v) for k, v in diff.items()})


if __name__ == '__main__':
    main()
