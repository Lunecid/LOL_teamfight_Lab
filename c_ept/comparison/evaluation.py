"""Three matched model pairs, reusing the established conditional estimands."""
from __future__ import annotations

import numpy as np


def _aligned(left, right, fields):
    if len(left) != len(right):
        raise ValueError('paired output coverage differs')
    result=[]
    right_by_name={r['name']:r for r in right}
    if len(right_by_name)!=len(right) or len({r['name'] for r in left})!=len(left):
        raise ValueError('duplicate paired record name')
    if set(right_by_name)!={r['name'] for r in left}:
        raise ValueError('paired output names differ')
    for a in left:
        b=right_by_name[a['name']]
        for key in fields:
            if key not in a or key not in b or not np.array_equal(a[key],b[key]):
                raise ValueError('paired row/target/reference mismatch: '+key)
        result.append((a,b))
    return result


def neural_pair(ept_result, n2_result):
    """Keep complete rows; never inner-join mismatching targets/references."""
    endpoints=[]
    for a,b in _aligned(ept_result['endpoints'],n2_result['endpoints'],
                         ('match','row','y','p_b1')):
        for key in ('module','layer','horizon_s','target'):
            if a.get(key)!=b.get(key):
                raise ValueError('endpoint definition mismatch')
        out=dict(a,p_b1=b['p_ept'])
        for stem in ('p_draw','mc_se','mc_se_draw'):
            ka,kb=stem+'_ept',stem+'_b1'
            if ka in b:
                out[kb]=b[ka]
            elif kb in out:
                raise ValueError('paired MC/draw definition differs')
        endpoints.append(out)
    nll=[]
    for a,b in _aligned(ept_result['nll'],n2_result['nll'],('match','row','nll_b1')):
        if a.get('scale')!=b.get('scale') or a['common_head']!=b['common_head']:
            raise ValueError('paired structural scale/head mismatch')
        out=dict(a,nll_b1=b['nll_ept'])
        if 'nll_ept_normalized' in b:
            out['nll_b1_normalized']=b['nll_ept_normalized']
        nll.append(out)
    return {'endpoints':endpoints,'nll':nll,'provenance':{
        'model_labels':{'ept':'EPT','b1':'N2'},'shared_reference_checked':'B1',
        'interpretation':'single-seed diagnostic, not original confirmatory success'}}


def summarize_pairs(ept_result, n2_result, *, bootstrap=2000, seed=20261022, check=None):
    from c_ept.evaluation.metrics import summarize_endpoints
    paired=neural_pair(ept_result,n2_result)
    outputs={}
    for name,result,labels in (
        ('ept_vs_b1',ept_result,{'ept':'EPT','b1':'B1'}),
        ('n2_vs_b1',n2_result,{'ept':'N2','b1':'B1'}),
        ('ept_vs_n2',paired,{'ept':'EPT','b1':'N2'})):
        summary=summarize_endpoints(result['endpoints'],result['nll'],
                                     bootstrap=bootstrap,seed=seed,check=check)
        summary['model_labels']=labels
        summary['interpretation']='diagnostic_only'
        outputs[name]=summary
    return outputs


def run_three(bundle, ept, n2, stats, b1, *, check, device='cuda', batch_size=32,
              draws=1000, seed=20261022, scales=None, bootstrap=2000,
              bootstrap_seed=20261022):
    from c_ept.evaluation.adapters import run_inference
    common=dict(check=check,device=device,batch_size=batch_size,draws=draws,seed=seed,scales=scales)
    a=run_inference(bundle,ept,stats,b1,**common)
    b=run_inference(bundle,n2,stats,b1,**common)
    return summarize_pairs(a,b,bootstrap=bootstrap,seed=bootstrap_seed,check=check)
