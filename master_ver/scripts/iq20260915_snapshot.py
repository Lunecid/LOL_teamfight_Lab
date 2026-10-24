"""Read-only integrity snapshot of parents this run must not modify (phase before / after; diff written after).

* sha256: every file of cohort_role_training, label_validity_full, objective_channel_ablation and collaborator critique
  audit outputs; the full-corpus files this run reads (labels, schema, manifests, V manifest, q models) and all parent
  scripts/*.py outside the iq20260915_ prefix; the original repository and worktree tracked-code .py files.
* (size, mtime_ns) inventory: every file of the 16 GB full-corpus run, deliverables/, reports/, docs/ (excluding this
  run's own new docs), the v3.3 shard directory and the main cache directory statistics; scripts/ and tests/
  __pycache__ listings (imports must not create bytecode).
Usage: python -B scripts/iq20260915_snapshot.py --phase before|after
"""
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
import iq20260915_common as Q  # noqa: E402

OWN_DOCS = ('docs/CLAUDE_INCREMENTAL_Q_TRAIN_20260915.md',)


def sha_tree(root):
    root = Path(root)
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            out[p.relative_to(C.ROOT).as_posix() if C.ROOT in p.parents else p.as_posix()] = C.sha256_file(p)
    return out


def inventory(root, skip=()):
    root = Path(root)
    out = {}
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            rel = p.relative_to(C.ROOT).as_posix() if C.ROOT in p.parents else p.as_posix()
            if any(rel.startswith(s) for s in skip):
                continue
            try:
                s = p.stat()
                out[rel] = [s.st_size, s.st_mtime_ns]
            except OSError as exc:
                out[rel] = ['error', repr(exc)]
    return out


def fc_read_targets():
    fc = Q.FC
    files = sorted(p for p in (fc / 'labels').glob('*') if p.is_file())
    files += sorted(p for p in (fc / 'models' / 'q').rglob('*') if p.is_file())
    for fn in ('q_pre_only_schema.json', 'v_models_manifest.json', 'frozen_manifest.json', 'protocol.json', 'selection_v.json',
               'validation.json', 'outcome_access_log.jsonl', 'eval/results_q.json'):
        files.append(fc / fn)
    return {p.relative_to(C.ROOT).as_posix(): C.sha256_file(p) for p in files if p.exists()}


def py_hashes(root, subdirs):
    out = {}
    for sd in subdirs:
        base = Path(root) / sd
        if base.exists():
            for p in sorted(base.rglob('*.py')):
                if '__pycache__' not in p.parts:
                    out[p.relative_to(root).as_posix()] = C.sha256_file(p)
    return out


def dir_stat(root):
    n, size, mt = 0, 0, 0
    root = Path(root)
    if not root.exists():
        return dict(exists=False)
    with os.scandir(root) as it:
        for e in it:
            if e.is_file(follow_symlinks=False):
                s = e.stat(follow_symlinks=False)
                n += 1
                size += s.st_size
                mt = max(mt, s.st_mtime_ns)
    return dict(files=n, bytes=size, max_mtime_ns=mt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('before', 'after', 'final'), required=True)
    args = ap.parse_args()
    if args.phase == 'before' and (Q.OUT / 'integrity' / 'snapshot_before.json').exists():
        raise SystemExit('snapshot_before.json exists (never overwritten)')
    Q.log_command()
    t0 = time.time()
    snap = dict(phase=args.phase, taken_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                sha256_cohort_role=sha_tree(Q.CR), sha256_label_validity=sha_tree(Q.LVO), sha256_objective_ablation=sha_tree(Q.OCO),
                sha256_critique_audit=sha_tree(Q.AUDIT), sha256_full_corpus_read_targets=fc_read_targets(),
                inventory_full_corpus=inventory(Q.FC),
                sha256_parent_scripts={p.name: C.sha256_file(p) for p in sorted((C.ROOT / 'scripts').glob('*.py')) if not p.name.startswith('iq20260915_')},
                sha256_parent_tests={p.name: C.sha256_file(p) for p in sorted((C.ROOT / 'tests').glob('*.py')) if not p.name.startswith('test_iq20260915_')},
                inventory_docs=inventory(C.ROOT / 'docs', skip=OWN_DOCS), inventory_deliverables=inventory(C.ROOT / 'deliverables'),
                inventory_reports=inventory(C.ROOT / 'reports'),
                pycache_scripts=inventory(C.ROOT / 'scripts' / '__pycache__'), pycache_tests=inventory(C.ROOT / 'tests' / '__pycache__'),
                original_repo_py_sha256=py_hashes(C.REPO, ['core', 'data', 'gameplay', 'scripts', 'train']),
                worktree_py_sha256=py_hashes(C.WT, ['core', 'data', 'gameplay', 'scripts', 'train']),
                shards_inventory=inventory(Path('D:/LOL_Project/fusion_2615/corpus_shards_v33')),
                main_cache_dir_stat=dir_stat(C.CACHE_MAIN))
    snap['seconds'] = round(time.time() - t0, 1)
    snap['counts'] = {k: len(v) for k, v in snap.items() if isinstance(v, dict)}
    sha = C.write_json(Q.OUT / 'integrity' / f'snapshot_{args.phase}.json', snap)
    print(f'{args.phase}: counts {snap["counts"]} in {snap["seconds"]} s sha256 {sha}')
    if args.phase in ('after', 'final'):
        before = C.read_json(Q.OUT / 'integrity' / 'snapshot_before.json')
        diff = {}
        for key, a in before.items():
            if key in ('phase', 'taken_at', 'seconds', 'counts'):
                continue
            b = snap[key]
            if key == 'main_cache_dir_stat':
                diff[key] = dict(equal=a == b, before=a, after=b)
                continue
            diff[key] = dict(checked=len(a), changed=sorted(k for k in a if k in b and a[k] != b[k])[:50],
                             removed=sorted(k for k in a if k not in b)[:50], added=sorted(k for k in b if k not in a)[:50])
            diff[key]['equal'] = not (diff[key]['changed'] or diff[key]['removed'] or diff[key]['added'])
        diff['all_equal'] = all(v['equal'] for v in diff.values() if isinstance(v, dict))
        C.write_json(Q.OUT / 'integrity' / ('snapshot_diff.json' if args.phase == 'after' else 'snapshot_final_diff.json'), diff)
        print({k: (v['equal'] if isinstance(v, dict) else v) for k, v in diff.items()})


if __name__ == '__main__':
    main()
