"""Probe (read-only): v3.3 shard scale-count arrays vs the parent exposure population (no X, no labels read).

Writes outputs/cohort_role_training_20260915/probes/shard_probe.json. Descriptive; nothing is fitted.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'cohort_role_training_20260915'
SHARDS = Path('D:/LOL_Project/fusion_2615/corpus_shards_v33')
EXPOSURES = ROOT / 'outputs' / 'postkill_objective_delay_full' / 'exposures.csv'
KEYS = ('groups', 'patch', 'engage_ts', 'cluster_blue', 'cluster_red', 'present_blue', 'present_red', 'y_market_event')


def main():
    t0 = time.time()
    parts = {k: [] for k in KEYS}
    files = {}
    for p in sorted(SHARDS.glob('shard_*.npz')):
        with np.load(p, allow_pickle=False) as z:
            files[p.name] = sorted(z.files)
            for k in KEYS:
                parts[k].append(np.asarray(z[k]))
    A = {k: np.concatenate(v) for k, v in parts.items()}
    n = len(A['groups'])
    key = np.char.add(np.char.add(A['groups'].astype(str), ':'), A['engage_ts'].astype(str))
    uk, cnt = np.unique(key, return_counts=True)
    ex = {}
    with open(EXPOSURES, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            ex.setdefault(f"{r['match']}:{r['s']}", []).append(int(r['L']))
    shard_keys = set(uk.tolist())
    ex_keys = set(ex)
    res = dict(
        shard_files=len(files), shard_member_names=sorted(set(k for v in files.values() for k in v)), rows=int(n),
        matches=int(len(np.unique(A['groups']))), patches=dict(Counter(A['patch'].astype(str).tolist())),
        duplicate_keys=int((cnt > 1).sum()), duplicate_key_rows=int(cnt[cnt > 1].sum()),
        exposure_rows=int(sum(len(v) for v in ex.values())), exposure_unique_keys=len(ex_keys),
        exposure_keys_with_multiple_L=int(sum(1 for v in ex.values() if len(v) > 1)),
        exposure_keys_in_shards=len(ex_keys & shard_keys), exposure_keys_missing_from_shards=len(ex_keys - shard_keys),
        shard_keys_not_in_exposures=len(shard_keys - ex_keys),
        missing_matches=len({k.split(':')[0] for k in ex_keys - shard_keys}),
        exposure_matches=len({k.split(':')[0] for k in ex_keys}),
        matches_all_exposures_missing=len({k.split(':')[0] for k in ex_keys - shard_keys} - {k.split(':')[0] for k in ex_keys & shard_keys}),
        cluster_negative_rows=int(((A['cluster_blue'] < 0) | (A['cluster_red'] < 0)).sum()),
        cluster_blue_values=dict(Counter(A['cluster_blue'].astype(int).tolist())),
        cluster_red_values=dict(Counter(A['cluster_red'].astype(int).tolist())),
        present_negative_rows=int(((A['present_blue'] < 0) | (A['present_red'] < 0)).sum()),
        y_market_event_labelled=int((A['y_market_event'] >= 0).sum()),
        example_missing=sorted(ex_keys - shard_keys)[:10], example_extra=sorted(shard_keys - ex_keys)[:10],
        seconds=round(time.time() - t0, 1))
    (OUT / 'probes').mkdir(parents=True, exist_ok=True)
    (OUT / 'probes' / 'shard_probe.json').write_bytes(json.dumps(res, indent=1).encode('utf-8'))
    print(json.dumps({k: v for k, v in res.items() if k not in ('cluster_blue_values', 'cluster_red_values')}, indent=1))


if __name__ == '__main__':
    main()
