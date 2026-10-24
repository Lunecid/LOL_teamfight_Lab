"""Codex independent arithmetic on Claude's saved sensitivity predictions; no fits."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/label_validity_full_20260915'
FC=ROOT/'outputs/full_corpus_training_20260915'
CR=ROOT/'outputs/cohort_role_training_20260915'
res={'scope':'Independent saved-array arithmetic and main-test q label-dependence; no raw re-extraction or human validation','sets':{}}
files=sorted((OUT/'alt_labels').glob('*_alt_labels.npz'))
assert len(files)==7, f'Expected all seven label sets, found {len(files)}'
for f in files:
    name=f.name.removesuffix('_alt_labels.npz')
    with np.load(f,allow_pickle=False) as z, np.load(FC/'labels'/f'{name}_labels.npz',allow_pickle=False) as a, np.load(CR/'cohorts'/f'{name}_cohort.npz',allow_pickle=False) as c:
        assert np.array_equal(z['match'],a['match']) and np.array_equal(z['s'],a['s'])
        assert np.array_equal(z['match'],c['match']) and np.array_equal(z['s'],c['s'])
        item={'rows':len(z['match']),'checks':{},'agreement':{}}
        for h in (60,90,120):
            v=a[f'valid_h{h}'].astype(bool)
            assert np.array_equal(z[f'valid_h{h}'],a[f'valid_h{h}'])
            assert np.array_equal(z[f'endpoint_h{h}'],a[f'endpoint_h{h}'])
            assert np.array_equal(z[f'Y_h{h}_A'],a[f'Y_h{h}'])
            for m in ('A','B_reg','B_econ'):
                d=z[f'delta_h{h}_{m}']; p=z[f'p_post_h{h}_{m}']-z[f'p_pre_{m}']; y=z[f'Y_h{h}_{m}']
                err=float(np.max(abs(d[v]-p[v])))
                assert err<1e-14 and np.isfinite(d[v]).all() and np.array_equal(y[v],d[v]>0)
                item['checks'][f'{m}_h{h}']={'valid':int(v.sum()),'delta_max_error':err,'exact_zero':int((d[v]==0).sum())}
            for co,cm in [('E',np.ones(len(v),bool)),('T',c['cohort']==1),('N',c['cohort']==0)]:
                mask=v&cm
                _,idx,n=np.unique(z['match'][mask],return_inverse=True,return_counts=True)
                weights=1/n[idx]
                for alt in ('B_reg','B_econ'):
                    diff=z[f'Y_h{h}_A'][mask]!=z[f'Y_h{h}_{alt}'][mask]
                    item['agreement'][f'h{h}/{co}/{alt}']={'rows':int(mask.sum()),'matches':len(n),'row_disagreement':float(diff.mean()),'match_weighted_disagreement':float(np.average(diff,weights=weights))}
        res['sets'][name]=item
        reported=json.loads((OUT/'results/per_set'/f'{name}.json').read_text(encoding='utf8'))
        for key,cell in item['agreement'].items():
            h,co,alt=key.split('/')
            expected=reported['agreement'][h][co][f'A_vs_{alt}']
            assert abs(cell['row_disagreement']-expected['disagreement_row'])<1e-12
            assert abs(cell['match_weighted_disagreement']-expected['disagreement_match_weighted'])<1e-12

with np.load(OUT/'alt_labels/MAIN_TEST_alt_labels.npz',allow_pickle=False) as z:
    lookup={k:i for i,k in enumerate(zip(z['match'].tolist(),z['s'].tolist()))}
    res['q_label_dependence']={}
    for co in ('T','N'):
        with np.load(CR/'eval/predictions'/f'A_MAIN_TEST_h90_{co}.npz',allow_pickle=False) as p:
            ix=np.asarray([lookup[k] for k in zip(p['match'].tolist(),p['s_ms'].tolist())])
            _,idx,n=np.unique(p['match'],return_inverse=True,return_counts=True); w=1/n[idx]
            q=p['spec_'+str(p['specialist_chosen'])]
            assert np.array_equal(p['y'],z['Y_h90_A'][ix])
            res['q_label_dependence'][co]={}
            for m in ('A','B_reg','B_econ'):
                y=z[f'Y_h90_{m}'][ix]
                res['q_label_dependence'][co][m]={'auc':float(roc_auc_score(y,q,sample_weight=w)),'brier':float(brier_score_loss(y,q,sample_weight=w)),'logloss':float(log_loss(y,q,sample_weight=w)),'rows':len(q)}
            if co=='T':
                # Independently reconstruct reported main T interval using direct
                # match resampling, rather than the executor's multiplicity dot products.
                bs={m:[] for m in ('B_reg','B_econ')}
                vals={m:np.bincount(idx,weights=z['Y_h90_A'][ix]!=z[f'Y_h90_{m}'][ix])/n for m in bs}
                rng=np.random.default_rng(20260915)
                for _ in range(1000):
                    pick=rng.integers(len(n),size=len(n))
                    for m in bs:bs[m].append(vals[m][pick].mean())
                rpt=json.loads((OUT/'results/per_set/MAIN_TEST.json').read_text(encoding='utf8'))
                res['main_T_disagreement_ci']={}
                for m in bs:
                    ci=np.quantile(bs[m],[.025,.975])
                    assert np.allclose(ci,rpt['bootstrap_h90']['T']['estimates'][f'A_vs_{m}']['match_weighted']['ci95'],atol=1e-12,rtol=0)
                    res['main_T_disagreement_ci'][m]=ci.tolist()
res['passed']=True
(OUT/'codex_independent_audit.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'passed':True,'main_h90':{n:{k:v for k,v in a['agreement'].items() if k.startswith('h90/')} for n,a in res['sets'].items() if n=='MAIN_TEST'},'q':res['q_label_dependence']},indent=2))
