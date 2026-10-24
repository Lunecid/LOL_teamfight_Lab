from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import json,time,hashlib
ROOT=Path(__file__).resolve().parents[1];CACHE=Path('D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13');OUT=ROOT/'outputs/causal_position_20260914';INDEX=Path('D:/LOL_Project/fusion_2615/features/tog_revision/A3-temporal-windows/fight_index.npz')
TAUS=[5,15,30,60];QS=[.01,1,100];R=100.**2
def one(task):
    mid,patch=task
    try:
        with np.load(CACHE/(mid+'.npz'),allow_pickle=False) as z:t=z['minute_ts'].astype(float)/1000.;xy=z['xy_raw_minute'].astype(float)
        if xy.shape!=(len(t),10,2) or not np.isfinite(xy).all() or (np.diff(t)<=0).any():return [],dict(match=mid,error='invalid_schema_or_values')
        errs={k:[] for k in ['hold','constant_velocity']+[f'damped_{v}' for v in TAUS]+[f'kalman_{v}' for v in QS]};eligible=0
        states={q:np.stack([xy[0],np.zeros_like(xy[0])],axis=-1) for q in QS};covs={q:np.diag([R,100.**2]) for q in QS}
        for j in range(1,len(t)):
            dt=t[j]-t[j-1];F=np.array([[1.,dt],[0.,1.]])
            ok=j>=2 and t[j]>=180 and 45<=dt<=75 and 45<=t[j-1]-t[j-2]<=75
            if ok:
                eligible+=1;v=(xy[j-1]-xy[j-2])/(t[j-1]-t[j-2]);preds={'hold':xy[j-1],'constant_velocity':xy[j-1]+dt*v}
                for tau in TAUS:preds[f'damped_{tau}']=xy[j-1]+tau*(-np.expm1(-dt/tau))*v
                for k,p in preds.items():errs[k].extend(np.linalg.norm(p-xy[j],axis=1))
            for q in QS:
                xp=states[q]@F.T;P=F@covs[q]@F.T+q*np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]])
                if ok:errs[f'kalman_{q}'].extend(np.linalg.norm(xp[:,:,0]-xy[j],axis=1))
                K=P[:,0]/(P[0,0]+R);states[q]=xp+(xy[j]-xp[:,:,0])[:,:,None]*K
                A=np.eye(2)-np.outer(K,[1.,0.]);covs[q]=A@P@A.T+R*np.outer(K,K)
        rows=[]
        for k,values in errs.items():
            if not values:continue
            a=np.array(values);rows.append(dict(match=mid,patch=patch,method=k,n=len(a),mae=float(a.mean()),mse=float(np.mean(a*a)),within1600=float((a<=1600).mean())))
        return rows,dict(match=mid,frames=len(t),eligible_targets=eligible,excluded_targets=max(0,len(t)-2)-eligible)
    except Exception as e:return [],dict(match=mid,error=repr(e))
def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.time()
    with np.load(INDEX,allow_pickle=False) as z:tasks=sorted(set(zip(z['match_id'].astype(str),z['patch'].astype(str))))
    rows=[];audit=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,(a,b) in enumerate(pool.map(one,tasks),1):
            rows.extend(a);audit.append(b)
            if i%2000==0:print(f'{i}/{len(tasks)} {time.time()-start:.1f}s',flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'match_metrics.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'coverage.csv',index=False)
    summaries=[]
    for (patch,method),g in df.groupby(['patch','method']):summaries.append(dict(patch=patch,method=method,matches=len(g),points=int(g.n.sum()),mae=float(g.mae.mean()),rmse=float(np.sqrt(g.mse.mean())),within1600=float(g.within1600.mean())))
    dev=[x for x in summaries if x['patch']=='15.15'];selected={family:min([x for x in dev if x['method'].startswith(family)],key=lambda x:x['mae'])['method'] for family in ['damped','kalman']}
    test=df[df.patch=='15.16'].pivot(index='match',columns='method',values='mae');rng=np.random.default_rng(7);paired=[]
    for name in ['constant_velocity',*selected.values()]:
        delta=(test[name]-test['hold']).dropna().to_numpy();boot=[float(delta[rng.integers(0,len(delta),len(delta))].mean()) for _ in range(500)]
        paired.append(dict(method=name,mae_minus_hold=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    out=dict(status='complete',attempted_matches=len(tasks),errors=[a for a in audit if 'error'in a],summary=summaries,selected_on_1515=selected,paired_1516=paired,seconds=time.time()-start,index_sha256=hashlib.sha256(INDEX.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(selected=selected,paired=paired,test=[a for a in summaries if a['patch']=='15.16' and a['method']in ['hold','constant_velocity',*selected.values()]])),flush=True)
if __name__=='__main__':main()
