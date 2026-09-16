import os,sys,json,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];WT=ROOT/'worktrees/engagement-state-value';CACHE=Path('D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13');OUT=ROOT/'outputs/position_ablation_20260914';VD=WT/'outputs/temporal_winprob_v3_buckets'
sys.path.insert(0,str(WT));sys.dont_write_bytecode=True;os.environ['LOL_OUTPUT_ROOT']=str(OUT/'runtime')
from core.config import NODE_FEATURE_NAMES
from gameplay.state_value import StateBuilder
def build(task):
    mid,rows=task
    with np.load(CACHE/(mid+'.npz'),allow_pickle=False) as z:pack={k:z[k] for k in z.files}
    pack['meta']=json.loads((CACHE/(mid+'.meta.json')).read_text());pack['events']=json.loads((CACHE/(mid+'.events.json')).read_text());b=StateBuilder(pack,NODE_FEATURE_NAMES);out=[]
    order=sorted(b.tm,key=b.slots.get)
    for r in rows:
        q=int(r['s'])-1;state=b.at(q);idx=np.searchsorted(b.ts,q,side='right')-1
        xy=pack['xy_raw_minute'][idx,np.array(order)-1,:].astype(float).reshape(-1)/15000.
        if not np.isfinite(xy).all():raise ValueError('nonfinite coordinates '+mid)
        out.append((r,state.values,xy,state.snapshot_ms))
    return out
def weights(g):
    _,ix,c=np.unique(g,return_inverse=True,return_counts=True);w=1/c[ix];return w/w.mean()
def metrics(y,p,w):
    from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
    return dict(auc=float(roc_auc_score(y,p,sample_weight=w)),brier=float(brier_score_loss(y,p,sample_weight=w)),logloss=float(log_loss(y,p,labels=[0,1],sample_weight=w)))
def main():
    from lightgbm import LGBMClassifier
    import joblib
    OUT.mkdir(parents=True,exist_ok=True);start=time.time()
    labels=pd.read_csv(ROOT/'outputs/b_boundary_60_90_120/dynamic_values.csv');labels=labels[(labels.h==90)&(labels.valid==1)]
    old=pd.read_csv(ROOT/'outputs/window_validation_20260914/labels.csv',dtype={'patch':str});pre=old[old.h==-1][['match','s','patch','expanded']].rename(columns={'expanded':'p_pre'})
    rows=labels.merge(pre,on=['match','s'],validate='one_to_one');rows['y']=(rows.p_post>rows.p_pre).astype(int)
    protocol=json.loads((VD/'protocol.json').read_text());used=set(rows.match);overlaps={k:len(used&set(protocol['splits'][k])) for k in ['fit','calibrate','select','test']};assert not any(overlaps.values())
    records=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,a in enumerate(pool.map(build,[(mid,g.to_dict('records')) for mid,g in rows.groupby('match')]),1):
            records.extend(a)
            if i%3000==0:print(f'features {i} matches {time.time()-start:.1f}s',flush=True)
    names=list(records[0][1]);X=np.array([[v[n] for n in names]+[r['p_pre']] for r,v,xy,snap in records],dtype=np.float32);names+=['p_pre'];coords=np.stack([xy for r,v,xy,snap in records]).astype(np.float32)
    meta=pd.DataFrame([dict(match=r['match'],s=r['s'],patch=r['patch'],y=r['y'],snapshot=snap,p_pre=r['p_pre']) for r,v,xy,snap in records]);meta.to_csv(OUT/'rows.csv',index=False)
    assert np.isfinite(X).all() and np.isfinite(coords).all();assert (meta.snapshot<meta.s).all()
    cats=[i for i,k in enumerate(names) if k.endswith('champion_id')];tr=(meta.patch=='15.14').to_numpy();va=(meta.patch=='15.15').to_numpy();te=(meta.patch=='15.16').to_numpy();y=meta.y.to_numpy();groups=meta.match.to_numpy();w=weights(groups[tr]);results=[];preds={}
    feature_names={'baseline':names,'position':names+[f'slot{i}_{axis}_last_observed' for i in range(10) for axis in ['x','y']]}
    (OUT/'feature_names.json').write_text(json.dumps(feature_names,indent=2))
    for condition,xx in [('baseline',X),('position',np.column_stack([X,coords]))]:
        ps=[]
        for seed in [7,42,123]:
            m=LGBMClassifier(n_estimators=250,num_leaves=15,learning_rate=.04,min_child_samples=100,reg_lambda=1.,random_state=seed,n_jobs=4,verbosity=-1,colsample_bytree=.9,subsample=1.)
            m.fit(xx[tr],y[tr],sample_weight=w,categorical_feature=cats)
            p=m.predict_proba(xx)[:,1];ps.append(p)
            for split,mask in [('development_1515',va),('test_1516',te)]:results.append(dict(condition=condition,seed=seed,split=split,**metrics(y[mask],p[mask],weights(groups[mask]))))
            joblib.dump(m,OUT/f'{condition}_{seed}.joblib');print(f'{condition} seed{seed} done {time.time()-start:.1f}s',flush=True)
        preds[condition]=np.mean(ps,axis=0)
        for split,mask in [('development_1515',va),('test_1516',te)]:results.append(dict(condition=condition,seed='ensemble',split=split,**metrics(y[mask],preds[condition][mask],weights(groups[mask]))))
    pm=meta.copy()
    for k,p in preds.items():pm[k]=p
    pm.to_csv(OUT/'predictions.csv',index=False)
    testg=groups[te];u,ix=np.unique(testg,return_inverse=True);basew=weights(testg);rng=np.random.default_rng(7);boot={k:[] for k in ['auc','brier','logloss']}
    for _ in range(500):
        c=np.bincount(rng.integers(0,len(u),len(u)),minlength=len(u));bw=basew*c[ix];keep=bw>0
        a=metrics(y[te][keep],preds['baseline'][te][keep],bw[keep]);b=metrics(y[te][keep],preds['position'][te][keep],bw[keep])
        for k in boot:boot[k].append(b[k]-a[k])
    testresults={k:metrics(y[te],p[te],weights(testg)) for k,p in preds.items()};delta={k:dict(difference=testresults['position'][k]-testresults['baseline'][k],ci95=np.quantile(boot[k],[.025,.975]).tolist()) for k in boot}
    out=dict(status='complete',rows=len(meta),features={k:len(v) for k,v in feature_names.items()},splits={p:dict(rows=int((meta.patch==p).sum()),matches=int(meta[meta.patch==p].match.nunique())) for p in sorted(meta.patch.unique())},value_partition_intersections=overlaps,future_snapshot_violations=int((meta.snapshot>=meta.s).sum()),results=results,paired_ensemble_test=delta,seconds=time.time()-start,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(test=testresults,delta=delta,splits=out['splits'],features=out['features'])),flush=True)
if __name__=='__main__':main()
