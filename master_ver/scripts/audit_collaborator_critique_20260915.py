"""Post-hoc arithmetic on immutable predictions; no model fitting or label changes."""
from pathlib import Path
import gc
import hashlib
import json
import platform

import numpy as np
import sklearn
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / 'outputs/full_corpus_training_20260915'
COHORT = ROOT / 'outputs/cohort_role_training_20260915'
OUT = ROOT / 'outputs/collaborator_critique_audit_20260915'
SEED = 20260915
REPS = 1000
SOURCES = {}


def record(path):
    key = str(path.relative_to(ROOT))
    if key not in SOURCES:
        with path.open('rb') as handle:
            SOURCES[key] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return path


def weights(groups):
    unique, inv, count = np.unique(groups, return_inverse=True, return_counts=True)
    return unique, inv, count, 1.0 / count[inv]


def score(y, p, groups):
    unique, _, _, w = weights(groups)
    if len(y) == 0:
        return {'rows': 0, 'matches': 0}
    bins = []
    for lo in np.arange(0, 1, .1):
        mask = (p >= lo) & ((p < lo + .1) if lo < .9 else (p <= 1))
        if mask.any():
            bins.append({'lo': float(lo), 'rows': int(mask.sum()),
                         'predicted': float(np.average(p[mask], weights=w[mask])),
                         'observed': float(np.average(y[mask], weights=w[mask]))})
    return {'rows': len(y), 'matches': len(unique),
            'positive_rate': float(np.average(y, weights=w)),
            'mean_prediction': float(np.average(p, weights=w)),
            'auc': float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) == 2 else None,
            'brier': float(np.average((p - y) ** 2, weights=w)),
            'logloss': float(log_loss(y, p, sample_weight=w, labels=[0, 1])),
            'calibration_bins': bins}


def ages(a):
    assert np.isfinite(a).all() and (a >= 0).all()
    return {'rows': len(a), 'mean_s': float(np.mean(a)),
            'quantiles_s': dict(zip(['p0', 'p50', 'p90', 'p99', 'p100'],
                                   np.quantile(a, [0, .5, .9, .99, 1]).tolist())),
            'fraction_zero': float(np.mean(a == 0)),
            'fraction_ge30s': float(np.mean(a >= 30)),
            'fraction_ge60s': float(np.mean(a >= 60))}


def paired(y, a, b, groups):
    unique, inv, count, _ = weights(groups)
    eps = np.finfo(float).eps
    pa, pb = np.clip(a, eps, 1 - eps), np.clip(b, eps, 1 - eps)
    row_diffs = [(a-y)**2 - (b-y)**2,
                 -y*np.log(pa)-(1-y)*np.log1p(-pa)+y*np.log(pb)+(1-y)*np.log1p(-pb)]
    per_match = np.stack([np.bincount(inv, weights=d) / count for d in row_diffs])
    rng = np.random.default_rng(SEED)
    draws = np.stack([per_match[:, rng.integers(len(unique), size=len(unique))].mean(axis=1)
                      for _ in range(REPS)])
    return {name: {'difference': float(per_match[i].mean()),
                   'ci95': np.quantile(draws[:, i], [.025, .975]).tolist()}
            for i, name in enumerate(['brier', 'logloss'])}


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'results.json').exists():
        raise RuntimeError('Audit exists; do not overwrite a completed result')
    result = {'scope': 'Post-hoc frozen-prediction audit, not new training or untouched confirmation',
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'sklearn': sklearn.__version__},
              'weighting': 'each represented match has equal total weight within each reported cell',
              'bootstrap': {'replicates': REPS, 'seed': SEED, 'unit': 'match',
                            'scope': 'fixed predictions; evaluation-sample uncertainty only',
                            'multiple_comparison_adjustment': False},
              'sets': {}}
    manifest = json.loads(record(FULL/'extract/MAIN/extraction_manifest.json').read_text())
    train_age = []
    for chunk in manifest['chunks']:
        path = FULL / chunk['path']
        # Hashes are recorded for every stored chunk actually read; no X matrices are loaded.
        with np.load(record(path), allow_pickle=False) as z:
            roles = dict(zip(z['m_match'].tolist(), z['m_sub_role'].tolist()))
            take = np.asarray([roles[m].startswith('fold') for m in z['v_match'].tolist()])
            take &= z['v_is_bucket_sample'] == 1
            train_age.append((z['v_query_ms'][take]-z['v_snapshot_ms'][take])/1000)
    result['train_value_query_ages'] = ages(np.concatenate(train_age))
    assert result['train_value_query_ages']['rows'] == 424160
    for name in ['MAIN_TEST', 'EXT_KR_16.13', 'EXT_KR_16.14_pilot', 'EXT_KR_16.15', 'EXT_NA1_16.13']:
        print('Auditing', name, flush=True)
        with np.load(record(FULL/'eval/predictions'/f'v_{name}.npz'), allow_pickle=False) as v:
            match, winner = v['match'], v['winner_blue']
            unique, first, inv = np.unique(match, return_index=True, return_inverse=True)
            assert np.array_equal(winner, winner[first][inv])
            outcome = dict(zip(unique.tolist(), winner[first].tolist()))
            bucket = v['is_bucket_sample'] == 1
            p = v['p_'+str(v['chosen'])]
            cell = {'value_grid': score(winner[bucket], p[bucket], match[bucket]),
                    'value_grid_ages': ages((v['query_ms'][bucket]-v['snapshot_ms'][bucket])/1000),
                    'cohorts': {}}
        with np.load(record(FULL/'labels'/f'{name}_labels.npz'), allow_pickle=False) as lab:
            key = dict(zip(zip(lab['match'].tolist(), lab['s'].tolist()), range(len(lab['s']))))
            assert len(key) == len(lab['s'])
            for cohort in ['T', 'N']:
                with np.load(record(COHORT/'eval/predictions'/f'A_{name}_h90_{cohort}.npz'), allow_pickle=False) as q:
                    g, y = q['match'], q['y']
                    idx = np.asarray([key[k] for k in zip(g.tolist(), q['s_ms'].tolist())])
                    assert (lab['valid_h90'][idx] == 1).all()
                    assert np.array_equal(y, lab['Y_h90'][idx])
                    chosen = str(q['specialist_chosen']); pred=q['spec_'+chosen]; baseline=q['spec_p_pre_spline']
                    pp = lab['p_pre'][idx]; post = lab['p_post_h90'][idx]
                    cr = {'chosen_frozen_specialist': chosen, 'comparisons': {}, 'value_endpoints': {}}
                    for subset, mask in [('all', np.ones(len(y), dtype=bool)),
                                         ('B40', (pp >= .4) & (pp <= .6)),
                                         ('B45', (pp >= .45) & (pp <= .55))]:
                        if not mask.any():
                            cr['comparisons'][subset] = {'rows': 0}; continue
                        a = score(y[mask], pred[mask], g[mask]); b=score(y[mask], baseline[mask], g[mask])
                        cr['comparisons'][subset]={'specialist': a, 'p_pre_spline': b,
                            'specialist_minus_spline': paired(y[mask], pred[mask], baseline[mask], g[mask]),
                            'sparse': a['matches'] < 30 or len(np.unique(y[mask])) < 2}
                    covered = np.asarray([m in outcome for m in g.tolist()])
                    cr['outcome_coverage']={'covered_rows':int(covered.sum()), 'missing_rows':int((~covered).sum()),
                        'note':'W joined from saved V grid predictions; matches absent from this source excluded only from endpoint V scores'}
                    wg=np.asarray([outcome.get(m, -1) for m in g.tolist()])
                    for endpoint, vp, at in [('pre', pp, lab['pre_snapshot_age_s_AUDIT'][idx]),
                                            ('post', post, lab['post_snapshot_age_s_h90'][idx])]:
                        er={'ages_all_rows': ages(at), 'vs_final_W':score(wg[covered],vp[covered],g[covered]), 'age_strata': {}}
                        for lo,hi in [(0,15),(15,30),(30,45),(45,60),(60,float('inf'))]:
                            mask = covered & (at>=lo) & (at<hi)
                            if mask.any():
                                er['age_strata'][f'{lo}_{hi}']={'value_vs_W':score(wg[mask],vp[mask],g[mask]),
                                    'q_vs_Y':score(y[mask],pred[mask],g[mask]),
                                    'delta_abs_le_1pp':float(np.mean(abs(post[mask]-pp[mask])<=.01))}
                        cr['value_endpoints'][endpoint]=er
                    cell['cohorts'][cohort]=cr
        result['sets'][name]=cell
        gc.collect()
    for path, digest in SOURCES.items():
        with (ROOT/path).open('rb') as handle:
            assert hashlib.file_digest(handle,'sha256').hexdigest()==digest,path
    result['source_hashes']=SOURCES
    result['source_preservation_checked']=True
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Audit complete', flush=True)


if __name__ == '__main__':
    main()
