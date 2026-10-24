"""Codex saved-array audit, independent of executor metric and bootstrap helpers."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/objective_channel_ablation_20260915'
FC = ROOT/'outputs/full_corpus_training_20260915'
CR = ROOT/'outputs/cohort_role_training_20260915'

def weights(g):
    _, ix, n = np.unique(g, return_inverse=True, return_counts=True)
    return 1/n[ix]

def scores(y,p,g):
    w=weights(g)
    return dict(rows=len(y), matches=len(np.unique(g)),
        auc=float(roc_auc_score(y,p,sample_weight=w)),
        brier=float(brier_score_loss(y,p,sample_weight=w)),
        logloss=float(log_loss(y,p,sample_weight=w)))

def mean_bootstrap(values,g):
    _,ix,n=np.unique(g,return_inverse=True,return_counts=True)
    means=np.bincount(ix,weights=values)/n
    rng=np.random.default_rng(20260915)
    draws=[means[rng.integers(len(n),size=len(n))].mean() for _ in range(1000)]
    return dict(point=float(means.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())

def verify_reported(result):
    for name,cells in result['sets'].items():
        reported=json.loads((OUT/'results/per_set'/f'{name}.json').read_text(encoding='utf8'))
        for key,cell in cells.items():
            h,co=key.split('/')
            ref=reported['agreement'][h][co]
            assert abs(cell['row_disagreement']-ref['disagreement_row'])<1e-12
            assert abs(cell['match_disagreement']-ref['disagreement_match_weighted'])<1e-12
            if 'bootstrap' in cell:
                ref=reported['bootstrap_h90'][co]['estimates']['A_vs_B_noobj']['match_weighted']
                assert np.allclose(cell['bootstrap']['ci95'],ref['ci95'],atol=1e-12,rtol=0)
    w=json.loads((OUT/'results/w_eval.json').read_text(encoding='utf8'))['sets']
    for name,models in result['v'].items():
        for m,metrics in models.items():
            ref=w[name][f'{m}_raw']['match_weighted']
            for k in ('auc','brier','logloss'):
                assert abs(metrics[k]-ref[k])<1e-10,(name,m,k)
    for k,c in result['v_bootstrap_B_minus_A'].items():
        ref=w['MAIN_TEST']['bootstrap_B_minus_A']['results']['match_weighted'][k]
        assert abs(c['point']-ref['estimate'])<1e-12
        assert np.allclose(c['ci95'],ref['ci95'],atol=1e-12,rtol=0)
    q=json.loads((OUT/'results/q_target_dependence.json').read_text(encoding='utf8'))['cohorts']
    for co,models in result['q'].items():
        for m,metrics in models.items():
            for k in ('auc','brier','logloss'):
                assert abs(metrics[k]-q[co]['scores'][f'Y_{m}'][k])<1e-12
    result['executor_reported_metrics_and_CIs_match']=True
    (OUT/'codex_independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('All executor metrics and main match-bootstrap intervals match independent calculations.')

def main():
    import sys
    if '--verify-reported' in sys.argv:
        verify_reported(json.loads((OUT/'codex_independent_audit.json').read_text(encoding='utf8')))
        return
    assert (OUT/'frozen_manifest.json').exists(), 'Do not audit sealed outputs before freeze'
    files=sorted((OUT/'labels').glob('*_B_noobj_labels.npz'))
    assert len(files)==7, f'Expected 7 completed label sets, got {len(files)}'
    result={'scope':'Independent saved-array/key/metric audit; no fitting or human validation','sets':{}}
    for path in files:
        name=path.name.removesuffix('_B_noobj_labels.npz')
        with np.load(path,allow_pickle=False) as z, np.load(FC/'labels'/f'{name}_labels.npz',allow_pickle=False) as a, np.load(CR/'cohorts'/f'{name}_cohort.npz',allow_pickle=False) as c:
            for field in ('match','s'):
                assert np.array_equal(z[field],a[field]) and np.array_equal(z[field],c[field])
            models=[k.removeprefix('p_pre_') for k in z.files if k.startswith('p_pre_')]
            assert 'A' in models and 'B' in models, models
            cells={}
            for h in (60,90,120):
                valid=a[f'valid_h{h}'].astype(bool)
                for key in (f'valid_h{h}',f'endpoint_h{h}'):
                    assert np.array_equal(z[key],a[key])
                for source,target in [('p_pre_A','p_pre'),(f'p_post_h{h}_A',f'p_post_h{h}'),(f'delta_h{h}_A',f'delta_h{h}'),(f'Y_h{h}_A',f'Y_h{h}')]:
                    assert np.array_equal(z[source],a[target],equal_nan=True)
                for m in models:
                    d=z[f'delta_h{h}_{m}'];y=z[f'Y_h{h}_{m}']
                    pre=z[f'p_pre_{m}'];post=z[f'p_post_h{h}_{m}']
                    assert np.isfinite(d[valid]).all()
                    assert np.allclose(d[valid],(post-pre)[valid],atol=1e-14,rtol=0)
                    assert np.array_equal(y[valid],d[valid]>0)
                    assert ((pre[valid]>=0)&(pre[valid]<=1)&(post[valid]>=0)&(post[valid]<=1)).all()
                    assert np.all(y[~valid]==-1)
                for co,mask0 in [('E',np.ones(len(valid),bool)),('T',c['cohort']==1),('N',c['cohort']==0)]:
                    mask=valid&mask0;g=z['match'][mask]
                    d=z[f'Y_h{h}_A'][mask]!=z[f'Y_h{h}_B'][mask]
                    cell=dict(rows=int(mask.sum()),matches=len(np.unique(g)),row_disagreement=float(d.mean()),match_disagreement=float(np.average(d,weights=weights(g))))
                    if h==90 and name=='MAIN_TEST': cell['bootstrap']=mean_bootstrap(d,g)
                    cells[f'h{h}/{co}']=cell
            result['sets'][name]=cells
    with np.load(OUT/'labels/MAIN_TEST_B_noobj_labels.npz',allow_pickle=False) as z:
        ix_of={k:i for i,k in enumerate(zip(z['match'].tolist(),z['s'].tolist()))}
        result['q']={}
        for co in ('T','N'):
            with np.load(CR/'eval/predictions'/f'A_MAIN_TEST_h90_{co}.npz',allow_pickle=False) as p:
                ix=np.array([ix_of[k] for k in zip(p['match'].tolist(),p['s_ms'].tolist())])
                assert np.array_equal(p['y'],z['Y_h90_A'][ix])
                q=p['spec_'+str(p['specialist_chosen'])]
                result['q'][co]={m:scores(z[f'Y_h90_{m}'][ix],q,p['match']) for m in ('A','B')}
    result['v']={}
    for path in sorted((OUT/'eval/w_predictions').glob('w_*.npz')):
        name=path.stem.removeprefix('w_')
        with np.load(path,allow_pickle=False) as z:
            y=z['winner_blue'];g=z['match'];preds={m:z[f'p_{m}_raw'] for m in ('A','B_noobj')}
            result['v'][name]={m:scores(y,p,g) for m,p in preds.items()}
            if name=='MAIN_TEST':
                a,b=preds['A'],preds['B_noobj'];eps=np.finfo(float).eps
                a=np.clip(a,eps,1-eps);b=np.clip(b,eps,1-eps)
                losses=dict(brier=(preds['B_noobj']-y)**2-(preds['A']-y)**2,
                    logloss=(-y*np.log(b)-(1-y)*np.log(1-b))-(-y*np.log(a)-(1-y)*np.log(1-a)))
                result['v_bootstrap_B_minus_A']={m:mean_bootstrap(v,g) for m,v in losses.items()}
    assert 'MAIN_TEST' in result['v']
    result['passed']=True
    (OUT/'codex_independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'passed':True,'main_h90':{k:v for k,v in result['sets']['MAIN_TEST'].items() if k.startswith('h90')},'v':result['v']['MAIN_TEST'],'q':result['q']},indent=2))

if __name__=='__main__': main()
