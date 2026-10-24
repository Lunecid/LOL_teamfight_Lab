import os,sys,json,time,hashlib
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];WT=ROOT/'worktrees/engagement-state-value';VD=WT/'outputs/temporal_winprob_v3_buckets';OUT=ROOT/'outputs/event_boundary_cif_20260914';CACHE=Path('D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13')
sys.path.insert(0,str(WT));sys.dont_write_bytecode=True;os.environ['LOL_OUTPUT_ROOT']=str(OUT/'runtime');os.environ['OMP_NUM_THREADS']='1'
HS=[60,90,120]
def init():
    global model,names,StateBuilder
    import joblib
    from core.config import NODE_FEATURE_NAMES
    from gameplay.state_value import StateBuilder
    names=NODE_FEATURE_NAMES;model=joblib.load(VD/'expanded_model.joblib')
def one(task):
    mid,refs=task
    try:
        ev=json.loads((CACHE/(mid+'.events.json')).read_text());meta=json.loads((CACHE/(mid+'.meta.json')).read_text())
        with np.load(CACHE/(mid+'.npz'),allow_pickle=False) as z:pack={k:z[k] for k in z.files}
        pack.update(events=ev,meta=meta);builder=StateBuilder(pack,names)
        kills=[e for e in ev if e.get('type')=='CHAMPION_KILL'];kts=sorted(int(e['timestamp']) for e in kills);kc=Counter(kts);km={int(e['timestamp']):e for e in kills}
        objs=[e for e in ev if e.get('type')=='ELITE_MONSTER_KILL' and e.get('monsterType') in ['DRAGON','BARON_NASHOR','RIFTHERALD','HORDE','ATAKHAN']]
        firsts=[];labels=[];states=[]
        for r in refs:
            L=r['L'];s=r['s'];kt=s+15000;pos=km[kt]['position'];amb=kc[kt]>1
            nk=next((t for t in kts if t>L),float('inf'))
            ne=r['next_start'] if r['next_start']<r['end'] else float('inf')
            boundary=min(nk,ne,r['end'])
            ot=min([int(o['timestamp']) for o in objs if int(o['timestamp'])>L and 'x' in (o.get('position') or {}) and 'y' in o['position'] and (o['position']['x']-pos['x'])**2+(o['position']['y']-pos['y'])**2<=4264**2],default=float('inf'))
            for mode,combat in [('A_next_engagement',ne),('B_next_kill',min(nk,ne))]:
                times={'objective':ot,'combat':combat,'game_end':r['end']};t=min(times.values());causes=[k for k,v in times.items() if v==t]
                cause=causes[0] if len(causes)==1 else 'simultaneous'
                if t>L+300000:t=L+300000;cause='censored'
                firsts.append(dict(match=mid,s=s,mode=mode,time_s=(t-L)/1000,cause=cause,anchor_ambiguous=int(amb)))
            for h in HS:
                ep=int(min(L+h*1000,boundary-1));row=dict(match=mid,s=s,h=h,endpoint=ep,duration_s=(ep-L)/1000,valid=0,p_post=np.nan,snapshot=-1,additional_kill_included=int(any(L<t<=ep for t in kts)))
                if L<=ep<=builder.ts[-1]:
                    st=builder.at(ep);row.update(valid=1,snapshot=st.snapshot_ms,index=len(states));states.append(st)
                labels.append(row)
        if states:
            X=np.array([[s.values[n] for n in model.state_names] for s in states]);ps=model.predict_proba(X)[:,1]
            for row in labels:
                if row['valid']:row['p_post']=float(ps[row.pop('index')])
        return firsts,labels,None
    except Exception as e:return [],[],mid+': '+repr(e)
def aj(a):
    g=a.groupby(['time_s','cause']).size().unstack(fill_value=0);risk=len(a);surv=1.;f={k:0. for k in ['objective','combat','game_end','simultaneous']};out=[]
    for t,row in g.iterrows():
        ds=sum(int(row.get(k,0)) for k in f)
        for k in f:f[k]+=surv*int(row.get(k,0))/risk
        surv*=1-ds/risk;risk-=int(row.sum());out.append(dict(time_s=t,survival=surv,**f))
    return pd.DataFrame(out)
def main():
    global OUT
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args();OUT=args.out
    OUT.mkdir(parents=True,exist_ok=True);start=time.time()
    old=pd.read_csv(ROOT/'outputs/window_validation_20260914/labels.csv');keys=old[(old.h==120)&(old.valid==1)][['match','s']]
    ex=pd.read_csv(ROOT/'outputs/postkill_objective_delay_full/exposures.csv');ex=ex.merge(keys,on=['match','s'],validate='one_to_one')
    rows=[];labs=[];errors=[]
    with ProcessPoolExecutor(max_workers=6,initializer=init) as pool:
        for i,(a,b,e) in enumerate(pool.map(one,[(m,g.to_dict('records')) for m,g in ex.groupby('match')],chunksize=16),1):
            rows.extend(a);labs.extend(b)
            if e:errors.append(e)
            if i%2000==0:print(f'{i} matches, {time.time()-start:.1f}s, errors{len(errors)}',flush=True)
    df=pd.DataFrame(rows);ld=pd.DataFrame(labs);df.to_csv(OUT/'first_events.csv',index=False);ld.to_csv(OUT/'dynamic_values.csv',index=False)
    sums=[];curves=[];maxerr=0
    for mode,a in df[df.anchor_ambiguous==0].groupby('mode'):
        c=aj(a);c['mode']=mode;curves.append(c)
        for h in [30,60,90,120,180,300]:
            v=c[c.time_s<=h].iloc[-1];rate=float(((a.cause=='objective')&(a.time_s<=h)).mean());maxerr=max(maxerr,abs(rate-v.objective))
            sums.append(dict(mode=mode,h=h,n=len(a),objective_cif=float(v.objective),combat_cif=float(v.combat),game_end_cif=float(v.game_end),simultaneous_cif=float(v.simultaneous),event_free=float(v.survival)))
    pd.concat(curves).to_csv(OUT/'cif_curves.csv',index=False)
    pre=old[old.h==-1][['match','s','expanded']].rename(columns={'expanded':'p_pre'})
    joined=ld.merge(pre,on=['match','s'],validate='many_to_one').merge(old[old.h.isin(HS)][['match','s','h','expanded']],on=['match','s','h'],validate='one_to_one');stats=[]
    for h,a in joined[joined.valid==1].groupby('h'):
        stats.append(dict(h=int(h),n=len(a),flip_vs_A=float(((a.p_post>a.p_pre)!=(a.expanded>a.p_pre)).mean()),positive=float((a.p_post>a.p_pre).mean()),median_duration_s=float(a.duration_s.median()),mean_duration_s=float(a.duration_s.mean()),full_horizon_pct=float((a.duration_s==h).mean()*100)))
    out=dict(matches=int(ex.match.nunique()),engagements=len(ex),errors=errors,anchor_excluded=int(df[df['mode']=='A_next_engagement'].anchor_ambiguous.sum()),cif=sums,dynamic_labels=stats,checks=dict(cif_empirical_maxerror=maxerr,additional_kill_included=int(ld.additional_kill_included.sum()),future_snapshot=int(((ld.valid==1)&(ld.snapshot>ld.endpoint)).sum()),invalid_labels=int((ld.valid==0).sum())),seconds=time.time()-start,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
