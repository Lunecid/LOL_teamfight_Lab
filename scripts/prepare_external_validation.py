"""Freeze v3.3 detection on pre-existing 100-match replay validation sample."""
import os,sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/validation_suite_20260914/external';OUT.mkdir(parents=True,exist_ok=True)
REPO=Path('C:/Users/todtj/PycharmProjects/LOL_teamfight')
os.environ['LOL_CFG_PRESET']='v3.3';os.environ['LOL_OUTPUT_ROOT']=str(OUT/'runtime');sys.dont_write_bytecode=True;sys.path.insert(0,str(REPO))
from core.config import cfg,NODE_FEATURE_NAMES
from core.roles import get_role_slots_from_detail
from gameplay.pipeline_cache import parse_timeline_to_minute_cache
from gameplay.fights import detect_fights
from data.index_split import _fight_to_ref_row
cfg.DUMP_FIGHTS=False;cfg.CACHE_IN_RAM=False

def main():
    manifest=Path('D:/LOL_Project/validation/replay_goldset_2026_26_13/goldset_manifest.json')
    matches=json.loads(manifest.read_text())['matches'];raw=Path('D:/LOL_Project/data/raw/2026_current/kr')
    allrows=[];audit=[];cache=OUT/'cache';cache.mkdir(exist_ok=True)
    for i,r in enumerate(matches):
        mid=r['match_id'];dp=raw/'detail'/(mid+'.json');tp=raw/'timeline'/(mid+'.json')
        dh=hashlib.sha256(dp.read_bytes()).hexdigest();th=hashlib.sha256(tp.read_bytes()).hexdigest()
        assert dh==r['detail_sha256'] and th==r['timeline_sha256']
        d=json.loads(dp.read_text(encoding='utf-8'));tl=json.loads(tp.read_text(encoding='utf-8'));tm={int(p['participantId']):int(p['teamId']) for p in d['info']['participants']}
        p=parse_timeline_to_minute_cache(tl,tm,detail=d,cfg_in=cfg)
        p['meta']=dict(team_map=tm,role_slots=get_role_slots_from_detail(d),patch='16.13',anchor_is_norm=False,feature_version=cfg.FEATURE_VERSION)
        fights=detect_fights(p,tm);refs=[]
        for f in fights:
            a=_fight_to_ref_row(f,p['minute_ts'],match_id=mid,patch='16.13')
            if a is not None:refs.append(a)
        refs=sorted(refs,key=lambda a:a['t_start_ts']);kills=sorted(int(e['timestamp']) for e in p['events'] if e.get('type')=='CHAMPION_KILL')
        endevents=[e for e in p['events'] if e.get('type')=='GAME_END'];assert len(endevents)==1
        terminal=int(endevents[0]['timestamp']);winner=int(endevents[0]['winningTeam']==100)
        for j,a in enumerate(refs):
            s=int(a['t_start_ts']);L=int(a['last_kill_ts']);nxt=int(refs[j+1]['t_start_ts']) if j+1<len(refs) else terminal
            nk=next((t for t in kills if t>L),terminal);e=min(L+90000,nxt-1,nk-1,terminal-1)
            valid=(nxt>L and s>0 and e>=L and e<=int(p['minute_ts'][-1]))
            allrows.append(dict(match=mid,patch='16.13',s=s,L=L,endpoint=e,valid=int(valid),winner=winner,duration_s=(e-L)/1000))
        np.savez_compressed(cache/(mid+'.npz'),**{k:v for k,v in p.items() if isinstance(v,np.ndarray)})
        (cache/(mid+'.events.json')).write_text(json.dumps(p['events']),encoding='utf-8')
        (cache/(mid+'.meta.json')).write_text(json.dumps(p['meta']),encoding='utf-8')
        audit.append(dict(match=mid,detail_sha256=dh,timeline_sha256=th,fights=len(refs),winner=winner,last_snapshot=int(p['minute_ts'][-1])))
        if (i+1)%25==0:print('external prepared',i+1,flush=True)
    pd.DataFrame(allrows).to_csv(OUT/'engagements.csv',index=False)
    result=dict(matches=audit,node_names=NODE_FEATURE_NAMES,preset='v3.3',manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),config={k:v for k,v in vars(cfg).items() if isinstance(v,(str,int,float,bool,type(None)))},source_hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [REPO/'gameplay/fights.py',REPO/'gameplay/pipeline_cache.py',REPO/'data/index_split.py']})
    (OUT/'preparation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
