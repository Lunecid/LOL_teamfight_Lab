"""Full cached corpus, frozen v3.3 detector, first objectives before next engagement.
Descriptive all-patch analysis; no fitted label boundary or model.
"""
import os, sys, json, time, csv, hashlib
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
REPO=Path('C:/Users/todtj/PycharmProjects/LOL_teamfight')
CACHE=Path('D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13')
OUT=ROOT/'outputs/postkill_objective_delay_full'
os.environ['LOL_OUTPUT_ROOT']=str(OUT/'runtime')
os.environ['LOL_CFG_PRESET']='v3.3'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OMP_NUM_THREADS']='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(REPO))

def init():
    from core.config import cfg
    cfg.CACHE_IN_RAM=False
    cfg.DUMP_FIGHTS=False
    import data.cache_io as cio
    cio.CACHE_DIR=CACHE

def one(mid):
    from data.cache_io import load_match_cache
    from gameplay.fights import detect_fights
    from data.index_split import _fight_to_ref_row
    from measure_postkill_objective_delay import analyse_match
    pack=load_match_cache(mid)
    if pack is None:return [],[],{'unreadable_or_schema_rejected':1},mid
    try:
        patch=pack['meta']['patch']
        fights=detect_fights(pack,pack['meta']['team_map'])
        refs=[]
        for f in fights:
            r=_fight_to_ref_row(f,pack['minute_ts'],match_id=mid,patch=patch)
            if r is not None:
                refs.append(dict(patch=patch,s=r['t_start_ts'],L=r['last_kill_ts'],x=r['anchor_x'],y=r['anchor_y']))
        pairs,ex,stats=analyse_match((mid,refs,float('inf'),4264))
        # Store first per category and any, with no imposed elapsed-time cap.
        records=[]
        for mode in ('before_next','spatial_before_next'):
            selected={}
            for p in pairs:
                if not p['before_next'] or p['delay_s']<=0:continue
                if mode.startswith('spatial') and not p['spatial']:continue
                for cat in (p['category'],'any'):
                    key=(p['s'],cat)
                    if key not in selected or p['delay_s']<selected[key]['delay_s']:
                        selected[key]=dict(p,category=cat,mode=mode)
            records.extend(selected.values())
        stats['loaded_matches']=1;stats['patch_'+patch]=1
        stats['same_timestamp_pairs']=sum(p['delay_s']==0 for p in pairs)
        return records,ex,dict(stats),None
    except Exception as e:
        return [],[],{'errors':1},mid+': '+repr(e)

def summarize():
    import pandas as pd
    df=pd.read_csv(OUT/'first_acquisitions.csv',dtype={'patch':str})
    ex=pd.read_csv(OUT/'exposures.csv',dtype={'patch':str})
    rows=[]
    for patch in ['pooled']+sorted(ex.patch.unique()):
        d=df if patch=='pooled' else df[df.patch==patch]
        x=ex if patch=='pooled' else ex[ex.patch==patch]
        for (mode,cat),g in d.groupby(['mode','category']):
            v=g.delay_s
            rows.append(dict(patch=patch,mode=mode,category=cat,n=len(v),eligible=len(x),mean=float(v.mean()),median=float(v.median()),p90=float(v.quantile(.9)),p95=float(v.quantile(.95)),maximum=float(v.max()),within60=int((v<=60).sum()),after60=int((v>60).sum()),after60_pct=float((v>60).mean()*100),after120=int((v>120).sum()),after300=int((v>300).sum())))
    checks=dict(duplicate_first_rows=int(df.duplicated(['match','s','category','mode']).sum()),nonpositive_delays=int((df.delay_s<=0).sum()),spatial_violations=int(((df['mode']=='spatial_before_next')&(df.distance_u>4264.001)).sum()))
    details=json.loads((OUT/'run.json').read_text())
    details.update(engagements=len(ex),matches_with_engagement=int(ex.match.nunique()),patch_engagements=ex.patch.value_counts().to_dict(),checks=checks,summaries=rows)
    (OUT/'results.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    print(json.dumps({'engagements':len(ex),'checks':checks,'main':[r for r in rows if r['patch']=='pooled' and r['mode']=='spatial_before_next']}),flush=True)

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);ap.add_argument('--summarize',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.summarize:summarize();return
    mids=sorted(p.name.replace('.meta.json','') for p in CACHE.glob('*.meta.json'))
    if args.limit:mids=mids[:args.limit]
    start=time.time();counts=Counter();errors=[];writers={};handles={}
    try:
        with ProcessPoolExecutor(max_workers=8,initializer=init) as pool:
            for n,(records,ex,stats,error) in enumerate(pool.map(one,mids,chunksize=32),1):
                counts.update(stats)
                if error:errors.append(error)
                for name,rows in [('first_acquisitions',records),('exposures',ex)]:
                    if not rows:continue
                    if name not in writers:
                        handles[name]=(OUT/(name+'.csv')).open('w',newline='',encoding='utf-8')
                        writers[name]=csv.DictWriter(handles[name],fieldnames=list(rows[0]));writers[name].writeheader()
                    writers[name].writerows(rows)
                if n%5000==0:print(f'{n}/{len(mids)} matches, {time.time()-start:.1f}s, errors={len(errors)}',flush=True)
    finally:
        for f in handles.values():f.close()
    evidence={}
    for folder in ['core','data','gameplay']:
        for p in (REPO/folder).glob('*.py'):evidence[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(attempted_matches=len(mids),source=str(CACHE),preset='v3.3',all_patches=True,observation='until next detected engagement anywhere or game end; no fixed delay cap',stats=dict(counts),errors=errors,seconds=time.time()-start,source_hashes=evidence,match_list_sha256=hashlib.sha256('\n'.join(mids).encode()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'run.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    summarize()

if __name__=='__main__':main()
