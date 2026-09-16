"""P3 post-run spot checks (run AFTER validation; read-only over stored artifacts; recomputes no labels).

1. labels_long.csv float round-trip vs labels_long.npz (reader caveat) and label stability under the default parser.
2. B90 npz values vs P2 endpoint diagnostics (bitwise), independent of the runner code.
3. Model-B output plateaus: probability values repeated across many distinct matches (pre and post queries),
   near-zero / exact-zero deltas, per-seed components, and the same values in P2's stored test predictions.
4. Source hashes now vs the hashes recorded by the run (which new scripts changed after the run).
Writes outputs/engagement_labels_v3_sensitivity/post_run_spot_checks.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / 'outputs' / 'engagement_labels_v3_sensitivity'
P2 = ROOT / 'outputs' / 'temporal_winprob_v3'
HS = (60, 90, 120)
FLOATS = ['p_pre_A', 'p_post_A', 'delta_A', 'p_pre_B', 'p_post_B', 'delta_B', 'p_pre_B_seed17', 'p_post_B_seed17',
          'p_pre_B_seed29', 'p_post_B_seed29', 'p_pre_B_seed43', 'p_post_B_seed43', 'pre_snapshot_age_s', 'post_snapshot_age_s',
          'duration_after_last_kill_s']


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT_BASE))
    o = Path(ap.parse_args().out)
    res = {'role': 'post-run spot checks after validation; descriptive; no labels recomputed or changed'}
    Z = np.load(o / 'labels_long.npz', allow_pickle=False)
    default = pd.read_csv(o / 'labels_long.csv', usecols=FLOATS)
    exact = pd.read_csv(o / 'labels_long.csv', usecols=FLOATS, float_precision='round_trip')
    csv = {}
    for c in FLOATS:
        csv[c] = {'round_trip_parser_bitwise_equal_npz': bool(np.array_equal(exact[c].to_numpy(), Z[c])),
                  'default_parser_bitwise_equal_npz': bool(np.array_equal(default[c].to_numpy(), Z[c])),
                  'default_parser_max_abs_diff': float(np.abs(default[c].to_numpy() - Z[c]).max())}
    for m in 'AB':
        csv[f'Y_{m}_label_changes_if_default_parsed_delta_used'] = int(((default[f'delta_{m}'].to_numpy() > 0) != (Z[f'delta_{m}'] > 0)).sum())
        rec = default[f'p_post_{m}'].to_numpy() - default[f'p_pre_{m}'].to_numpy()
        csv[f'Y_{m}_label_changes_if_recomputed_from_default_parsed_probabilities'] = int(((rec > 0) != (Z[f'delta_{m}'] > 0)).sum())
    csv['guidance'] = "use labels_long.npz, or pandas.read_csv(..., float_precision='round_trip'), for bitwise values"
    res['csv_float_round_trip'] = csv
    m90 = Z['h_s'] == 90
    order = np.argsort(Z['row_index'][m90], kind='mergesort')
    with np.load(P2 / 'endpoint_diagnostics' / 'endpoint_rows_DIAGNOSTIC.npz', allow_pickle=False) as e:
        res['npz_b90_bitwise_equal_p2'] = {k: bool(np.array_equal(Z[k][m90][order], e[k])) for k in ('p_pre_A', 'p_post_A', 'delta_A', 'p_pre_B', 'p_post_B', 'delta_B')}
    f = np.load(o / 'features_pre_only.npz', allow_pickle=False)
    res['features_p_pre_A_bitwise_equal_npz'] = bool(np.array_equal(f['X_input'][:, -1], Z['p_pre_A'][m90][order]))
    # Model-B plateaus: identical (12-decimal) probabilities shared by many distinct matches.
    match = Z['match']
    q = []
    q.append(pd.DataFrame({'kind': 'pre', 'match': match[m90], 'p_B': Z['p_pre_B'][m90], 'p_A': Z['p_pre_A'][m90]}))
    for h in HS:
        mh = Z['h_s'] == h
        q.append(pd.DataFrame({'kind': f'post_h{h}', 'match': match[mh], 'p_B': Z['p_post_B'][mh], 'p_A': Z['p_post_A'][mh]}))
    Q = pd.concat(q, ignore_index=True)
    Q['p_B_r12'] = Q.p_B.round(12)
    agg = Q.groupby('p_B_r12').agg(queries=('p_B', 'size'), matches=('match', 'nunique'), p_A_min=('p_A', 'min'), p_A_max=('p_A', 'max'))
    plate = agg[agg.matches >= 3].sort_values('matches', ascending=False)
    plateaus = []
    for val, row in plate.iterrows():
        sel = (np.abs(Q.p_B.to_numpy() - val) < 1e-9)
        entry = {'p_B_rounded_12': float(val), 'queries_within_1e-9': int(sel.sum()), 'distinct_matches': int(row.matches),
                 'queries_by_kind': Q[sel].kind.value_counts().to_dict(), 'p_A_range_on_same_queries': [float(row.p_A_min), float(row.p_A_max)]}
        seeds = {}
        for s in (17, 29, 43):
            vals = np.concatenate([Z[f'p_pre_B_seed{s}'][m90][np.abs(Z['p_pre_B'][m90] - val) < 1e-9]] +
                                  [Z[f'p_post_B_seed{s}'][Z['h_s'] == h][np.abs(Z['p_post_B'][Z['h_s'] == h] - val) < 1e-9] for h in HS])
            seeds[str(s)] = [float(vals.min()), float(vals.max())] if len(vals) else None
        entry['seed_component_ranges'] = seeds
        with np.load(P2 / 'predictions' / 'test_grid_A_B.npz', allow_pickle=False) as g:
            entry['p2_test_grid_queries_within_1e-9'] = int((np.abs(g['p_B'] - val) < 1e-9).sum())
            entry['p2_test_grid_queries_total'] = int(g['p_B'].size)
        with np.load(P2 / 'predictions' / 'test_one_minute_A_B.npz', allow_pickle=False) as t:
            entry['p2_test_one_minute_queries_within_1e-9'] = int((np.abs(t['p_B'] - val) < 1e-9).sum())
        plateaus.append(entry)
    res['model_B_repeated_value_plateaus (>=3 distinct matches)'] = plateaus
    res['model_B_plateau_note'] = ('On each plateau all three B members return near-constant values across distinct matches and game states. Model A on the same '
                                   'queries is given per plateau (p_A_range_on_same_queries): wide for some plateaus, extreme (near 0 or 1) for others. '
                                   'This is consistent with saturation of the recurrent hidden state; the mechanism was not verified here. '
                                   'B is a diagnostic comparator only.')
    near = {}
    for m in 'AB':
        d = Z[f'delta_{m}']
        near[m] = {'min_abs_delta': float(np.abs(d).min()), 'exact_zero_rows': int((d == 0).sum()), 'abs_delta_lt_1e-12_rows': int((np.abs(d) < 1e-12).sum()),
                   'abs_delta_lt_1e-12_engagements': int(len(set(zip(match[np.abs(d) < 1e-12].tolist(), Z['s_ms'][np.abs(d) < 1e-12].tolist()))))}
    res['near_zero_deltas'] = near
    run_hashes = json.loads((o / 'source_provenance.json').read_text(encoding='utf-8'))['new_sources_sha256']
    now = {p: sha256_file(ROOT / p) for p in run_hashes}
    res['new_sources_changed_after_run'] = {p: {'run': run_hashes[p], 'now': now[p]} for p in run_hashes if run_hashes[p] != now[p]}
    res['new_sources_unchanged_after_run'] = [p for p in run_hashes if run_hashes[p] == now[p]]
    res['this_script_sha256'] = sha256_file(Path(__file__))
    (o / 'post_run_spot_checks.json').write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in res.items() if k != 'csv_float_round_trip'}, indent=1)[:3000])


if __name__ == '__main__':
    main()
