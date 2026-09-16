"""Expand to completed 26.13 corpus, excluding 100 pilot matches."""
import prepare_external_validation as p
import sqlite3,time
from concurrent.futures import ProcessPoolExecutor
OUT=p.ROOT/'outputs/validation_suite_20260914/external_full';CACHE=OUT/'cache'

def one(r):
    try:
        mid=r['match_id'];raw=p.Path('D:/LOL_Project/data/raw/2026_current/kr');dp=raw/'detail'/(mid+'.json');tp=raw/'timeline'/(mid+'.json')
        db=dp.read_bytes();tb=tp.read_bytes();dh=p.hashlib.sha256(db).hexdigest();th=p.hashlib.sha256(tb).hexdigest()
        assert dh==r['detail_sha256'] and th==r['timeline_sha256']
        d=p.json.loads(db);tl=p.json.loads(tb);tm={int(a['participantId']):int(a['teamId']) for a in d['info']['participants']}
        pack=p.parse_timeline_to_minute_cache(tl,tm,detail=d,cfg_in=p.cfg)
        pack['meta']=dict(team_map=tm,role_slots=p.get_role_slots_from_detail(d),patch='16.13',anchor_is_norm=False,feature_version=p.cfg.FEATURE_VERSION)
        fights=p.detect_fights(pack,tm);refs=[]
        for f in fights:
            a=p._fight_to_ref_row(f,pack['minute_ts'],match_id=mid,patch='16.13')
            if a is not None:refs.append(a)
        refs=sorted(refs,key=lambda a:a['t_start_ts']);kills=sorted(int(e['timestamp']) for e in pack['events'] if e.get('type')=='CHAMPION_KILL');ends=[e for e in pack['events'] if e.get('type')=='GAME_END'];assert len(ends)==1
        terminal=int(ends[0]['timestamp']);winner=int(ends[0]['winningTeam']==100);rows=[]
        for j,a in enumerate(refs):
            s=int(a['t_start_ts']);L=int(a['last_kill_ts']);nxt=int(refs[j+1]['t_start_ts']) if j+1<len(refs) else terminal;nk=next((t for t in kills if t>L),terminal);e=min(L+90000,nxt-1,nk-1,terminal-1)
            valid=nxt>L and s>0 and e>=L and e<=int(pack['minute_ts'][-1])
            rows.append(dict(match=mid,patch='16.13',s=s,L=L,endpoint=e,valid=int(valid),winner=winner,duration_s=(e-L)/1000,external_split=r['external_split']))
        p.np.savez_compressed(CACHE/(mid+'.npz'),**{k:v for k,v in pack.items() if isinstance(v,p.np.ndarray)})
        (CACHE/(mid+'.events.json')).write_text(p.json.dumps(pack['events']),encoding='utf-8');(CACHE/(mid+'.meta.json')).write_text(p.json.dumps(pack['meta']),encoding='utf-8')
        return rows,dict(match=mid,game_creation=r['game_creation'],external_split=r['external_split'],detail_sha256=dh,timeline_sha256=th,fights=len(refs),winner=winner,last_snapshot=int(pack['minute_ts'][-1])),None
    except Exception as e:return [],None,dict(match=r['match_id'],error=repr(e))

def main():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(exist_ok=True)
    c=sqlite3.connect('file:D:/LOL_Project/data/raw/2026_current/_collector_state/collector.sqlite3?mode=ro',uri=True);c.row_factory=sqlite3.Row
    rows=[dict(a) for a in c.execute("select match_id,game_creation,detail_sha256,timeline_sha256 from matches where status='complete' and api_patch='16.13' and queue_id=420 and map_id=11 order by game_creation,match_id")];c.close()
    pilot={a['match_id'] for a in p.json.loads(p.Path('D:/LOL_Project/validation/replay_goldset_2026_26_13/goldset_manifest.json').read_text())['matches']};total=len(rows);rows=[a for a in rows if a['match_id'] not in pilot];cut=int(len(rows)*.3)
    for i,r in enumerate(rows):r['external_split']='adaptation_early' if i<cut else 'evaluation_late'
    (OUT/'frozen_manifest.json').write_text(p.json.dumps(dict(total_complete=total,excluded_pilot=len(pilot),split='first 30 percent by creation for adaptation; remaining 70 percent evaluation',rows=rows),indent=2),encoding='utf-8')
    results=[];audit=[];errors=[];start=time.time()
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i,(rr,a,e) in enumerate(pool.map(one,rows,chunksize=8),1):
            results.extend(rr)
            if a:audit.append(a)
            if e:errors.append(e)
            if i%1000==0:print('full external',i,'seconds',round(time.time()-start,1),'errors',len(errors),flush=True)
    p.pd.DataFrame(results).to_csv(OUT/'engagements.csv',index=False)
    (OUT/'preparation.json').write_text(p.json.dumps(dict(matches=audit,node_names=p.NODE_FEATURE_NAMES,preset='v3.3',errors=errors,source_hashes={str(x.relative_to(p.REPO)):p.hashlib.sha256(x.read_bytes()).hexdigest() for x in [p.REPO/'gameplay/fights.py',p.REPO/'gameplay/pipeline_cache.py',p.REPO/'data/index_split.py']}),indent=2),encoding='utf-8')
    print('full preparation done',len(audit),len(results),len(errors),flush=True)

if __name__=='__main__':main()
