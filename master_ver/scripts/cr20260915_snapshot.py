"""Read-only integrity snapshot of inputs this task must not modify (before / after the run).

* prior full-corpus run outputs/full_corpus_training_20260915: sha256 of every file except the adapted external caches,
  for which (size, mtime_ns) of every file is recorded (their per-match sha256 are already in prepared_manifest.json);
* original repo C:/Users/todtj/PycharmProjects/LOL_teamfight: sha256 of every tracked-code .py under core/, data/,
  gameplay/, scripts/ (read-only imports of the frozen detector);
* worktree sources used by the full run (scripts/, gameplay/, core/, data/ .py);
* main cache directory: file count, total bytes and max mtime (stat only);
* v3.3 shards: sha256 (also recorded by the cohort stage).
Usage: python scripts/cr20260915_snapshot.py --phase before|after
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402


def tree_hashes(root, skip_prefix=None):
    out, stat_only = {}, {}
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            rel = p.relative_to(root).as_posix()
            if skip_prefix and any(rel.startswith(sp) and '/cache/' in rel for sp in skip_prefix):
                s = p.stat()
                stat_only[rel] = [s.st_size, s.st_mtime_ns]
            else:
                out[rel] = C.sha256_file(p)
    return out, stat_only


def py_hashes(root, subdirs):
    out = {}
    for sd in subdirs:
        base = Path(root) / sd
        if not base.exists():
            continue
        for p in sorted(base.rglob('*.py')):
            if '__pycache__' in p.parts:
                continue
            out[p.relative_to(root).as_posix()] = C.sha256_file(p)
    return out


def dir_stat(root):
    n, size, mt = 0, 0, 0
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
    ap.add_argument('--phase', choices=('before', 'after'), required=True)
    args = ap.parse_args()
    t0 = time.time()
    fc_h, fc_stat = tree_hashes(K.FC, skip_prefix=['external/'])
    snap = dict(phase=args.phase, taken_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                prior_full_run_sha256=fc_h, prior_full_run_external_cache_stat=fc_stat,
                original_repo_py_sha256=py_hashes(K.REPO, ['core', 'data', 'gameplay', 'scripts']),
                worktree_py_sha256=py_hashes(K.WT, ['core', 'data', 'gameplay', 'scripts']),
                workspace_prior_scripts_sha256={p.name: C.sha256_file(p) for p in sorted((K.ROOT / 'scripts').glob('fc20260915_*.py'))},
                main_cache_dir_stat=dir_stat(C.CACHE_MAIN),
                shard_sha256={p.name: C.sha256_file(p) for p in sorted(K.SHARDS.glob('*'))})
    snap['seconds'] = round(time.time() - t0, 1)
    sha = C.write_json(K.OUT / 'integrity' / f'snapshot_{args.phase}.json', snap)
    print(f'{args.phase}: {len(fc_h)} hashed, {len(fc_stat)} stat-only, repo {len(snap["original_repo_py_sha256"])} py, '
          f'{snap["seconds"]} s, sha256 {sha}')
    if args.phase == 'after':
        before = C.read_json(K.OUT / 'integrity' / 'snapshot_before.json')
        diff = {}
        for key in ('prior_full_run_sha256', 'prior_full_run_external_cache_stat', 'original_repo_py_sha256', 'worktree_py_sha256',
                    'workspace_prior_scripts_sha256', 'main_cache_dir_stat', 'shard_sha256'):
            a, b = before[key], snap[key]
            if isinstance(a, dict) and all(isinstance(v, (str, list)) for v in a.values()):
                diff[key] = dict(changed=sorted(k for k in a if k in b and a[k] != b[k])[:50],
                                 removed=sorted(k for k in a if k not in b)[:50], added=sorted(k for k in b if k not in a)[:50])
            else:
                diff[key] = dict(equal=a == b, before=a, after=b)
        C.write_json(K.OUT / 'integrity' / 'snapshot_diff.json', diff)
        print(diff)


if __name__ == '__main__':
    main()
