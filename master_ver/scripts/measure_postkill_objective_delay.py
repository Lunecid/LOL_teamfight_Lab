"""Read-only development-patch diagnostic using the frozen v3.3 A3 index.
No model fitting or boundary selection. Horizon is an observation cap, not a label definition.
"""
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import argparse, csv, hashlib, json, time
import numpy as np

INDEX=Path('D:/LOL_Project/fusion_2615/features/tog_revision/A3-temporal-windows/fight_index.npz')
CACHE=Path('D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13')
CATS={'BARON_NASHOR':'baron','RIFTHERALD':'herald','HORDE':'horde','ATAKHAN':'atakhan'}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def analyse_match(task):
    mid, refs, horizon, radius = task
    events=json.loads((CACHE/(mid+'.events.json')).read_text(encoding='utf-8'))
    meta=json.loads((CACHE/(mid+'.meta.json')).read_text(encoding='utf-8'))
    tm={int(k):int(v) for k,v in meta['team_map'].items()}
    endings=[int(e['timestamp']) for e in events if e.get('type')=='GAME_END']
    end=max(endings) if endings else max(int(e['timestamp']) for e in events)
    objs=[]; stats=Counter(); seen=set(); stacks=Counter()
    for ev in sorted(events,key=lambda e:int(e['timestamp'])):
        if ev.get('type')!='ELITE_MONSTER_KILL': continue
        mt=ev.get('monsterType',''); sub=ev.get('monsterSubType','')
        cat=('elder' if sub=='ELDER_DRAGON' else 'dragon') if mt=='DRAGON' else CATS.get(mt)
        if cat is None: continue
        ts=int(ev['timestamp']); team=int(ev.get('killerTeamId',0) or tm.get(int(ev.get('killerId',0)),0))
        pos=ev.get('position') or {}; x=pos.get('x'); y=pos.get('y')
        key=(ts,cat,team,x,y,sub)
        if key in seen: stats['duplicate_source_events']+=1;continue
        seen.add(key)
        if team not in (100,200): stats['unknown_team']+=1; continue
        prior=stacks[team]
        if cat=='dragon':stacks[team]+=1
        objs.append(dict(ts=ts,cat=cat,sub=sub,team=team,x=x,y=y,prior=prior if cat=='dragon' else -1))
    refs=sorted(refs,key=lambda r:(r['s'],r['L']))
    pairs=[]; exposures=[]
    for i,r in enumerate(refs):
        nxt=refs[i+1]['s'] if i+1<len(refs) else end
        # Global next engagement is deliberately conservative: even a remote fight censors this diagnostic.
        exp=dict(match=mid,patch=r['patch'],s=r['s'],L=r['L'],next_start=nxt,end=end,
                 same_match_overlap=int(nxt<=r['L']),end_observed=int(bool(endings)))
        exposures.append(exp)
        for o in objs:
            delay=(o['ts']-r['L'])/1000
            if delay<0 or delay>horizon: continue
            spatial=(o['x'] is not None and o['y'] is not None and r['x']>=0 and r['y']>=0
                     and (o['x']-r['x'])**2+(o['y']-r['y'])**2<=radius**2)
            pairs.append(dict(match=mid,patch=r['patch'],s=r['s'],last_kill=r['L'],event_ts=o['ts'],
                category=o['cat'],subtype=o['sub'],team=o['team'],dragon_stacks_before=o['prior'],
                delay_s=delay,spatial=int(spatial),before_next=int(o['ts']<nxt),
                distance_u=round(((o['x']-r['x'])**2+(o['y']-r['y'])**2)**.5,3) if o['x'] is not None and o['y'] is not None else -1))
    return pairs, exposures, stats

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);start=time.time()
    with np.load(INDEX,allow_pickle=False) as z: data={k:z[k] for k in z.files}
    groups=defaultdict(list)
    for i,mid in enumerate(data['match_id']):
        if str(data['patch'][i]) not in ('15.14','15.15'):continue
        groups[str(mid)].append(dict(patch=str(data['patch'][i]),s=int(data['t_start_ts'][i]),L=int(data['last_kill_ts'][i]),x=float(data['anchor_x'][i]),y=float(data['anchor_y'][i])))
    tasks=[(m,r,300,4264) for m,r in sorted(groups.items())]
    if args.limit:tasks=tasks[:args.limit]
    pairs=[]; exposures=[];count=Counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,(p,e,c) in enumerate(pool.map(analyse_match,tasks),1):
            pairs.extend(p);exposures.extend(e);count.update(c)
            if n%1000==0:print(f'{n}/{len(tasks)} matches; {len(exposures)} engagements; {time.time()-start:.1f}s',flush=True)
    import pandas as pd
    df=pd.DataFrame(pairs); ex=pd.DataFrame(exposures)
    df.to_csv(args.out/'event_pairs.csv',index=False);ex.to_csv(args.out/'engagement_exposure.csv',index=False)
    out=dict(status='complete',source_index=str(INDEX),source_index_sha256=sha(INDEX),source_index_manifest=json.loads(INDEX.with_suffix('.json').read_text(encoding='utf-8')),
        source_cache=str(CACHE),script_sha256=sha(Path(__file__)),development_patches=['15.14','15.15'],
        sample='Development subset of the existing seeded 20,000-match A3 sample; all indexed engagements, not filtered by outcome',
        observation_cap_s=300,spatial_radius_u=4264,matches=len(tasks),engagements=len(ex),source_checks=dict(count),
        overlap_at_last_kill=int(ex.same_match_overlap.sum()),missing_game_end=int((ex.end_observed==0).sum()),
        caveats=['Temporal association, not causal attribution; neither acquisition team nor sign of predicted value selects events.',
        'Spatial filter reuses the existing first-kill anchor and attribution radius. It can exclude genuine rotations.',
        'Before-next filter censors at the next indexed engagement anywhere in the match, including remote fights.',
        'Means condition on an acquisition within the observation cap; no-acquisition is never encoded as zero delay.',
        'Soul is not counted as a separate acquisition to avoid duplicating dragon events; subtype/stacks retained. Herald use is not measured.',
        'This diagnostic does not choose the final grace period and does not inspect test-patch raw events.'],summaries=[])
    for patch in ['pooled','15.14','15.15']:
        d=df if patch=='pooled' else df[df.patch.astype(str)==patch]
        x=ex if patch=='pooled' else ex[ex.patch.astype(str)==patch]
        if len(x)==0: continue
        for mode in ['temporal','spatial','spatial_before_next']:
            m=d if mode=='temporal' else d[d.spatial==1]
            if mode=='spatial_before_next':m=m[m.before_next==1]
            for h in [30,60,90,120,180,300]:
                a=m[(m.delay_s>0)&(m.delay_s<=h)]
                for cat in ['any','baron','dragon','elder','herald','horde','atakhan']:
                    b=a if cat=='any' else a[a.category==cat]
                    first=b.sort_values('delay_s').drop_duplicates(['match','s'])
                    vals=first.delay_s.to_numpy()
                    eventkeys=b[['match','event_ts','category','team','subtype']].drop_duplicates()
                    # Counts use observed matches; incomplete follow-up is disclosed rather than called no event.
                    follow=np.minimum(x.end-x.L,h*1000)
                    if mode=='spatial_before_next':follow=np.minimum(follow,x.next_start-x.L)
                    out['summaries'].append(dict(patch=patch,mode=mode,horizon_s=h,category=cat,eligible_engagements=len(x),
                        observed_acquisition_engagements=len(first),observed_share=len(first)/len(x),
                        full_followup_engagements=int((follow>=h*1000).sum()),event_pairs=len(b),unique_events=len(eventkeys),
                        mean_s=float(vals.mean()) if len(vals) else None,median_s=float(np.median(vals)) if len(vals) else None,
                        p75_s=float(np.quantile(vals,.75)) if len(vals) else None,p90_s=float(np.quantile(vals,.9)) if len(vals) else None))
    out['same_timestamp_pairs']=int((df.delay_s==0).sum());out['seconds']=time.time()-start
    (args.out/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:out[k] for k in ['status','matches','engagements','overlap_at_last_kill','seconds']}),flush=True)
if __name__=='__main__':main()
