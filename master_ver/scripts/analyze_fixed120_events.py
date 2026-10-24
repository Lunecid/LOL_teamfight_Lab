"""Observed event associations and timestamp-local model response, fixed120."""
import os,sys,json,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];WT=ROOT/'worktrees/engagement-state-value'
VD=WT/'outputs/temporal_winprob_v3_buckets';OUT=ROOT/'outputs/fixed120_events_20260914'
CACHE=Path('D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13')
sys.path.insert(0,str(WT));sys.dont_write_bytecode=True
os.environ['LOL_OUTPUT_ROOT']=str(OUT/'runtime');os.environ['OMP_NUM_THREADS']='1'
CATS={'BARON_NASHOR':'baron','RIFTHERALD':'herald','HORDE':'horde','ATAKHAN':'atakhan'}
def init():
    global model,names,StateBuilder
    import joblib
    from core.config import NODE_FEATURE_NAMES
    from gameplay.state_value import StateBuilder
    names=NODE_FEATURE_NAMES;model=joblib.load(VD/'expanded_model.joblib')
def one(task):
    mid,refs=task
    try:
        with np.load(CACHE/(mid+'.npz'),allow_pickle=False) as z:pack={k:z[k] for k in z.files}
        pack['events']=json.loads((CACHE/(mid+'.events.json')).read_text());pack['meta']=json.loads((CACHE/(mid+'.meta.json')).read_text())
        events=pack['events'];b=StateBuilder(pack,names);tm={int(k):int(v) for k,v in pack['meta']['team_map'].items()}
        counts=Counter(int(e['timestamp']) for e in b.events);rows=[];responses={};states=[]
        for r in refs:
            row={k:r[k] for k in ['match','s','L','patch','endpoint','expanded','p_pre','p_last','snapshot','snapshot_last']}
            row.update(anchor_ambiguous=int(r['anchor_ambiguous']),local_objectives=0,remote_objectives=0,unknown_position_objectives=0,additional_kills=0,building_events=0)
            for ev in events:
                ts=int(ev['timestamp'])
                if not r['L']<ts<=r['endpoint']:continue
                typ=ev.get('type')
                if typ=='CHAMPION_KILL':row['additional_kills']+=1
                if typ in ['BUILDING_KILL','TURRET_PLATE_DESTROYED']:row['building_events']+=1
                if typ!='ELITE_MONSTER_KILL':continue
                mt=ev.get('monsterType');sub=ev.get('monsterSubType','')
                cat=('elder' if sub=='ELDER_DRAGON' else 'dragon') if mt=='DRAGON' else CATS.get(mt)
                if not cat:continue
                pos=ev.get('position') or {};xy=(pos.get('x'),pos.get('y'))
                local=None if None in xy or r['anchor_ambiguous'] else (xy[0]-r['anchor_x'])**2+(xy[1]-r['anchor_y'])**2<=4264**2
                row['unknown_position_objectives' if local is None else 'local_objectives' if local else 'remote_objectives']+=1
                team=int(ev.get('killerTeamId',0) or tm.get(int(ev.get('killerId',0)),0))
                key=(ts,cat,sub,team)
                if key in responses:continue
                rec=dict(match=mid,patch=r['patch'],timestamp=ts,category=cat,subtype=sub,team=team,same_timestamp_events=counts[ts],valid=0,response=np.nan,frame_changed=-1)
                if team in (100,200) and b.ts[0]<=ts-1 and ts<=b.ts[-1]:
                    before=b.at(ts-1);after=b.at(ts);j=len(states);states.extend([before,after]);rec.update(valid=1,frame_changed=int(before.snapshot_ms!=after.snapshot_ms),index=j)
                responses[key]=rec
            row['frame_changed']=int(r['snapshot']!=r['snapshot_last']);row['flip']=int((r['expanded']>r['p_pre'])!=(r['p_last']>r['p_pre']))
            rows.append(row)
        if states:
            X=np.array([[s.values[n] for n in model.state_names] for s in states]);p=model.predict_proba(X)[:,1]
            for rec in responses.values():
                if rec['valid']:
                    j=rec.pop('index');rec['response']=float((p[j+1]-p[j])*(1 if rec['team']==100 else -1))
        return rows,list(responses.values()),None
    except Exception as e:return [],[],mid+': '+repr(e)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/'outputs/window_validation_20260914/labels.csv',dtype={'patch':str})
    r=d[(d.h==120)&(d.valid==1)].copy()
    for h,col in [(-1,'p_pre'),(0,'p_last')]:r=r.merge(d[d.h==h][['match','s','expanded']].rename(columns={'expanded':col}),on=['match','s'],validate='one_to_one')
    r=r.merge(d[d.h==0][['match','s','snapshot']].rename(columns={'snapshot':'snapshot_last'}),on=['match','s'],validate='one_to_one')
    # Frozen A3 anchor index covers this independently sampled partition only partly;
    # recover exact anchors from the original first-kill position using s+15s.
    tasks=[]
    for mid,g in r.groupby('match'):
        events=json.loads((CACHE/(mid+'.events.json')).read_text());kills={int(e['timestamp']):e for e in events if e.get('type')=='CHAMPION_KILL'}
        killcounts=Counter(int(e['timestamp']) for e in events if e.get('type')=='CHAMPION_KILL')
        refs=g.to_dict('records')
        for ref in refs:
            ev=kills.get(int(ref['s'])+15000)
            if ev is None:raise ValueError('missing first kill anchor '+mid)
            pos=ev['position'];ref.update(anchor_x=pos['x'],anchor_y=pos['y'],anchor_ambiguous=killcounts[int(ref['s'])+15000]>1)
        tasks.append((mid,refs))
    rows=[];responses=[];errors=[];start=time.time()
    with ProcessPoolExecutor(max_workers=6,initializer=init) as pool:
        for n,(a,c,err) in enumerate(pool.map(one,tasks,chunksize=16),1):
            rows.extend(a);responses.extend(c)
            if err:errors.append(err)
            if n%2000==0:print(f'{n}/{len(tasks)} {time.time()-start:.1f}s errors={len(errors)}',flush=True)
    df=pd.DataFrame(rows);ev=pd.DataFrame(responses);df.to_csv(OUT/'engagements.csv',index=False);ev.to_csv(OUT/'event_responses.csv',index=False)
    groups=[]
    masks={'all':np.ones(len(df),bool),'local_objective':df.local_objectives>0,'remote_only_objective':(df.local_objectives==0)&(df.remote_objectives>0),'no_objective':(df.local_objectives+df.remote_objectives+df.unknown_position_objectives)==0,'additional_kills':df.additional_kills>0,'no_additional_kills':df.additional_kills==0,'frame_changed':df.frame_changed==1,'no_frame_change':df.frame_changed==0,'no_tracked_events':(df.local_objectives+df.remote_objectives+df.unknown_position_objectives+df.additional_kills+df.building_events)==0}
    for name,mask in masks.items():
        a=df[mask];groups.append(dict(group=name,n=len(a),flips=int(a.flip.sum()),flip_rate=float(a.flip.mean()) if len(a) else None))
    clean=ev[(ev.valid==1)&(ev.frame_changed==0)&(ev.same_timestamp_events==1)]
    stats=[]
    for cat,a in clean.groupby('category'):
        v=a.response;stats.append(dict(category=cat,n=len(a),mean_pp=float(v.mean()*100),median_pp=float(v.median()*100),negative_pct=float((v<0).mean()*100)))
    out=dict(status='complete',h=120,matches=len(tasks),engagements=len(df),errors=errors,groups=groups,event_response=stats,events=len(ev),isolated_events=len(clean),excluded_events=len(ev)-len(clean),seconds=time.time()-start,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
