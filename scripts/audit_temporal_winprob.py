"""Validate temporal WP artifacts and render source-backed example trajectories."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train.state_value_experiment import assert_disjoint, metrics
from train.temporal_winprob import random_minute_index, stable_int


def run(out, cache):
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    protocol, report = read(out/'protocol.json'), read(out/'results.json')
    assert report['status'] == read(out/'status.json')['stage'] == 'complete'
    splits = protocol['splits']
    assert_disjoint(*splits.values())
    for path,digest in protocol['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    sampled = read(out/'sampled_minutes.json')
    assert len(sampled)==len({r['match'] for r in sampled})
    assert {r['match'] for r in sampled}==set().union(*[set(splits[r]) for r in ('fit','calibrate','select','test')])
    assert not {r['match'] for r in sampled}&set(splits['engagement'])
    dataset=Path(protocol['source_dataset'])
    # Check all sampled row timestamps against the independent source grid.
    for row in sampled:
        with np.load(dataset/'matches'/(row['match']+'.npz'),allow_pickle=False) as z:
            t=z['value_times']
        assert row['time_ms']==int(t[random_minute_index(row['match'],len(t))])
    models={}
    for family in ('maymin','expanded'):
        path=out/(family+'_model.joblib')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==report['model_ids'][family]
        models[family]=joblib.load(path)
    selection=read(out/'selection.json')
    for family in models:
        assert report['chosen'][family]==min((k for k in selection['scores'] if k.startswith(family+'_')),
                                           key=lambda k:selection['scores'][k]['log_loss'])
    with np.load(out/'engagement_changes.npz',allow_pickle=False) as z:
        tab={k:z[k] for k in z.files}
    assert len(set(tab['id']))==len(tab['id'])==report['engagements']['n']
    assert set(tab['match']).issubset(splits['engagement'])
    valid=tab['query_ms']>=0
    assert np.all(tab['snapshot_ms'][valid]<=tab['query_ms'][valid])
    assert np.all(tab['query_ms'][:,0]<tab['query_ms'][:,1])
    for family in models:
        assert np.array_equal(np.isfinite(tab[family]),valid)
        assert ((tab[family][valid]>=0)&(tab[family][valid]<=1)).all()
        assert np.allclose(tab[family+'_delta'],tab[family][:,1:]-tab[family][:,:1],equal_nan=True)
    with np.load(out/'independent_time_curves.npz',allow_pickle=False) as z:
        curves={k:z[k] for k in z.files}
    assert set(curves['match'])==set(splits['test'])
    test_times={r['match']:r['time_ms'] for r in sampled if r['role']=='test'}
    selected=np.array([test_times[m]==t for m,t in zip(curves['match'],curves['time_ms'])])
    assert selected.sum()==len(splits['test'])
    for family in models:
        calc=metrics(curves['winner'][selected],curves[family][selected],curves['match'][selected])
        for key in ('auc','brier','log_loss'):
            assert abs(calc[key]-report['test_one_minute'][family][key])<1e-12

    os.environ['LOL_OUTPUT_ROOT']=str(out/'audit_runtime')
    os.environ['LOL_CFG_PRESET']='v3.3'
    os.environ['LOL_CFG_OVERRIDES']=json.dumps({'CACHE_DIRNAME':str(cache.resolve()),'FIGHT_INDEX_CACHE_ENABLED':False,'FIGHT_INDEX_NUM_WORKERS':1,'DUMP_FIGHTS':False})
    from core.config import NODE_FEATURE_NAMES
    from data.cache_io import load_match_cache
    from gameplay.state_value import StateBuilder
    # Display selection is based on event category and deterministic IDs, never delta magnitude.
    chosen=[]
    for i,label in enumerate(('Baron','Dragon','Elder')):
        indices=np.flatnonzero(np.any(tab['objectives'][:,i,:]>0,axis=1))
        if len(indices):
            chosen.append((min(indices,key=lambda k:stable_int(str(tab['id'][k]))),label))
    chosen.append((min(range(len(tab['id'])),key=lambda k:stable_int(str(tab['id'][k]))),'Unfiltered'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(chosen),1,figsize=(11,3.4*len(chosen)),squeeze=False)
    examples=[]
    clock_spreads=[]
    for axis,(i,label) in zip(axes[:,0],chosen):
        mid=str(tab['match'][i])
        builder=StateBuilder(load_match_cache(mid),NODE_FEATURE_NAMES)
        q=tab['query_ms'][i]
        present=q>=0
        states=[builder.at(int(t)) for t in q[present]]
        X=np.array([list(s.values.values()) for s in states])
        fixed=np.repeat(X[:1],3,axis=0)
        fixed[:,models['expanded'].state_names.index('snapshot_age_s')]=[0.,30.,60.]
        spread=float(np.ptp(models['expanded'].predict_proba(fixed)[:,1]))
        assert spread < 1e-12
        clock_spreads.append(spread)
        for family in models:
            assert np.allclose(models[family].predict_proba(X)[:,1],tab[family][i,present],atol=1e-12)
        with np.load(dataset/'matches'/(mid+'.npz'),allow_pickle=False) as z:
            grid,times=z['value_states'],z['value_times']
        # Plot actual causal state queries; do not connect minute predictions and
        # present interpolated lines as event-time win probabilities.
        fine_times=np.unique(np.concatenate([np.arange(times[0],times[-1]+1,5000),times,q[present]]))
        fine_states=np.array([list(builder.at(int(t)).values.values()) for t in fine_times])
        for family,color in (('maymin','#64748b'),('expanded','#087e8b')):
            axis.plot(fine_times/60000,models[family].predict_proba(fine_states)[:,1]*100,label=family,color=color)
        axis.scatter(q[present]/60000,tab['expanded'][i,present]*100,color='#c44536',zorder=4,label='pre / end proxy / +30s / +60s')
        axis.axvspan(q[0]/60000,q[1]/60000,alpha=.13,color='#e5a323',label='cutoff to last-kill proxy')
        axis.set(ylim=(0,100),ylabel='Blue win probability (%)',xlabel='Elapsed time (minutes)',
                 title=f'{label} example | {mid} | observed change {tab["expanded_delta"][i,0]*100:+.2f} pp')
        axis.grid(alpha=.2); axis.legend(loc='best',fontsize=8)
        nullable=lambda values:[float(x) if np.isfinite(x) else None for x in values]
        examples.append({'match':mid,'engagement_id':str(tab['id'][i]),'category':label,'query_ms':q.tolist(),
                         'expanded_probabilities':nullable(tab['expanded'][i]),'delta_pp':nullable(tab['expanded_delta'][i]*100)})
    fig.tight_layout()
    fig.savefig(out/'win_probability_examples.png',dpi=150)
    plt.close(fig)
    audit={'status':'passed','sampled_matches':len(sampled),'one_minute_per_match':True,
           'split_overlap':0,'source_and_model_hashes':'matched','saved_test_metrics':'recomputed',
           'engagement_rows':len(tab['id']),'source_rebuilt_examples':len(examples),
           'clock_age_only_max_probability_spread':max(clock_spreads),
           'display_selection':'deterministic IDs within event categories; illustrative, not a representative effect estimate',
           'examples':examples}
    (out/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--cache-dir',type=Path,required=True)
    a=ap.parse_args()
    run(a.out_dir.resolve(),a.cache_dir)
