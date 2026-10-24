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
import iq20260915_snapshot as S  # noqa: E402  (pure functions)
import dd20260916_common as DD  # noqa: E402

OWN_DOCS = ('docs/CLAUDE_DEFINITION_DEV_20260916.md',)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('before', 'after', 'final'), required=True)
    args = ap.parse_args()
    if args.phase == 'before' and (DD.OUT / 'integrity' / 'snapshot_before.json').exists():
        raise SystemExit('snapshot_before.json exists (never overwritten)')
    DD.log_command()
    t0 = time.time()
    ext = {}
    for s in DD.EXT:
        d = DD.external_dir(f'EXT_{s}')
        for n in ('exposures.csv', 'prepared_manifest.json'):
            if (d / n).exists():
                ext[(d / n).relative_to(C.ROOT).as_posix()] = C.sha256_file(d / n)
    snap = dict(phase=args.phase, taken_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                sha256_incremental_q=S.sha_tree(DD.IQ), sha256_cohort_role=S.sha_tree(DD.CR), sha256_full_corpus_read_targets=S.fc_read_targets(),
                sha256_parent_exposures={C.EXPOSURES.relative_to(C.ROOT).as_posix(): C.sha256_file(C.EXPOSURES), **ext},
                inventory_full_corpus=S.inventory(DD.FC),
                sha256_parent_scripts={p.name: C.sha256_file(p) for p in sorted((C.ROOT / 'scripts').glob('*.py')) if not p.name.startswith('dd20260916_')},
                sha256_parent_tests={p.name: C.sha256_file(p) for p in sorted((C.ROOT / 'tests').glob('*.py')) if not p.name.startswith('test_dd20260916_')},
                inventory_docs=S.inventory(C.ROOT / 'docs', skip=OWN_DOCS), inventory_deliverables=S.inventory(C.ROOT / 'deliverables'),
                pycache_scripts=S.inventory(C.ROOT / 'scripts' / '__pycache__'), pycache_tests=S.inventory(C.ROOT / 'tests' / '__pycache__'),
                original_repo_py_sha256=S.py_hashes(C.REPO, ['core', 'data', 'gameplay', 'scripts', 'train']),
                sha256_boundary_specs={p.name: C.sha256_file(p) for p in sorted((C.REPO / 'config' / 'fight_boundary').glob('*.json'))},
                shards_inventory=S.inventory(Path('D:/LOL_Project/fusion_2615/corpus_shards_v33')), main_cache_dir_stat=S.dir_stat(C.CACHE_MAIN))
    snap['seconds'] = round(time.time() - t0, 1)
    snap['counts'] = {k: len(v) for k, v in snap.items() if isinstance(v, dict)}
    sha = C.write_json(DD.OUT / 'integrity' / f'snapshot_{args.phase}.json', snap)
    print(f'{args.phase}: counts {snap["counts"]} in {snap["seconds"]} s sha256 {sha}')
    if args.phase in ('after', 'final'):
        before = C.read_json(DD.OUT / 'integrity' / 'snapshot_before.json')
        diff = {}
        for key, a in before.items():
            if key in ('phase', 'taken_at', 'seconds', 'counts'):
                continue
            b = snap[key]
            if key == 'main_cache_dir_stat':
                diff[key] = dict(equal=a == b, before=a, after=b)
                continue
            diff[key] = dict(checked=len(a), changed=sorted(k for k in a if k in b and a[k] != b[k])[:50], removed=sorted(k for k in a if k not in b)[:50], added=sorted(k for k in b if k not in a)[:50])
            diff[key]['equal'] = not (diff[key]['changed'] or diff[key]['removed'] or diff[key]['added'])
        diff['all_equal'] = all(v['equal'] for v in diff.values() if isinstance(v, dict))
        C.write_json(DD.OUT / 'integrity' / ('snapshot_diff.json' if args.phase == 'after' else 'snapshot_final_diff.json'), diff)
        print({k: (v['equal'] if isinstance(v, dict) else v) for k, v in diff.items()})


if __name__ == '__main__':
    main()
