"""Frozen-model window sensitivity on disjoint engagement partition."""
import os,sys,json,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; WT=ROOT/'worktrees/engagement-state-value'
VD=WT/'outputs/temporal_winprob_v3_buckets';OUT=ROOT/'outputs/window_validation_20260914'
CACHE=Path('D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13')
sys.path.insert(0,str(WT));sys.dont_write_bytecode=True
os.environ['LOL_OUTPUT_ROOT']=str(OUT/'runtime');os.environ['OMP_NUM_THREADS']='1'
HS=[0,30,60,90,120,180]
def init():
    global models,names,StateBuilder
    import joblib
    from core.config import NODE_FEATURE_NAMES
    from gameplay.state_value import StateBuilder
    names=NODE_FEATURE_NAMES
    models={k:joblib.load(VD/(k+'_model.joblib')) for k in ['expanded','maymin']}
def one(task):
    mid,refs=task
    try:
        with np.load(CACHE/(mid+'.npz'),allow_pickle=False) as z:pack={k:z[k] for k in z.files}
        pack['events']=json.loads((CACHE/(mid+'.events.json')).read_text())
        pack['meta']=json.loads((CACHE/(mid+'.meta.json')).read_text())
        b=StateBuilder(pack,names);out=[];states=[];links=[]
        for r in refs:
            overlap=r['next_start']<=r['L']
            for h in [-1]+HS:
                endpoint=r['s']-1 if h==-1 else min(r['L']+h*1000,r['next_start']-1,r['end']-1)
                row=dict(match=mid,s=r['s'],L=r['L'],patch=r['patch'],h=h,endpoint=endpoint,overlap=int(overlap),valid=0,snapshot=-1,expanded=np.nan,maymin=np.nan)
                if not overlap and endpoint>=0 and endpoint<=int(b.ts[-1]):
                    st=b.at(endpoint);states.append(st);links.append(len(out));row.update(valid=1,snapshot=st.snapshot_ms)
                out.append(row)
        if states:
            for k,m in models.items():
                X=np.array([[s.values[n] for n in m.state_names] for s in states])
                ps=m.predict_proba(X)[:,1]
                for j,p in zip(links,ps):out[j][k]=float(p)
        return out,None
    except Exception as e:return [],mid+': '+repr(e)
def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((VD/'protocol.json').read_text()); target=set(protocol['splits']['engagement'])
    overlaps={k:len(target&set(protocol['splits'][k])) for k in ['fit','calibrate','select','test']}
    assert not any(overlaps.values()),overlaps
    ex=pd.read_csv(ROOT/'outputs/postkill_objective_delay_full/exposures.csv',dtype={'patch':str})
    chosen=ex[ex.match.isin(target)]
    tasks=[(m,g.to_dict('records')) for m,g in chosen.groupby('match')]
    if args.limit:tasks=tasks[:args.limit]
    results=[];errors=[];start=time.time()
    with ProcessPoolExecutor(max_workers=6,initializer=init) as pool:
        for i,(rows,err) in enumerate(pool.map(one,tasks,chunksize=16),1):
            results.extend(rows)
            if err:errors.append(err)
            if i%1000==0:print(f'{i}/{len(tasks)}, errors={len(errors)}, {time.time()-start:.1f}s',flush=True)
    df=pd.DataFrame(results);df.to_csv(OUT/'labels.csv',index=False)
    summary=[];flips=[];agreements=[]
    for model in ['expanded','maymin']:
        p=df.pivot(index=['match','s'],columns='h',values=model)
        delta=p[HS].subtract(p[-1],axis=0)
        for h in HS:
            d=delta[h].dropna();summary.append(dict(model=model,h=h,n=len(d),positive=float((d>0).mean()),zero=float((d==0).mean()),mean_abs=float(d.abs().mean())))
        for a in HS:
            for c in HS:
                if c<=a:continue
                good=delta[[a,c]].dropna();flips.append(dict(model=model,a=a,b=c,n=len(good),flip_rate=float(((good[a]>0)!=(good[c]>0)).mean())))
    for h in HS:
        ds=[]
        for m in ['expanded','maymin']:
            p=df.pivot(index=['match','s'],columns='h',values=m);ds.append(p[h]-p[-1])
        good=pd.concat(ds,axis=1).dropna();agreements.append(dict(h=h,n=len(good),agreement=float(((good.iloc[:,0]>0)==(good.iloc[:,1]>0)).mean())))
    full=[]
    first=pd.read_csv(ROOT/'outputs/postkill_objective_delay_full/first_acquisitions.csv',usecols=['match','s','mode','category','delay_s'])
    for h in HS[1:]:
        counts={mode:int(((first['mode']==mode)&(first.category=='any')&(first.delay_s<=h)).sum()) for mode in ['before_next','spatial_before_next']}
        follow=np.minimum(ex.next_start,ex.end)-ex.L
        full.append(dict(h=h,**counts,full_followup=int((follow>=h*1000).sum()),total=len(ex)))
    out=dict(status='complete',target_matches=len(target),processed_matches=len(tasks),engagements=int(df[['match','s']].drop_duplicates().shape[0]),partition_intersections=overlaps,errors=errors,seconds=time.time()-start,future_snapshot_violations=int(((df.valid==1)&(df.snapshot>df.endpoint)).sum()),summary=summary,flips=flips,model_agreement=agreements,full_corpus=full,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),VD/'expanded_model.joblib',VD/'maymin_model.joblib',WT/'gameplay/state_value.py',ROOT/'docs/WINDOW_VALIDATION_PROTOCOL_20260914.md']})
    (OUT/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='hashes'}),flush=True)
if __name__=='__main__':main()
