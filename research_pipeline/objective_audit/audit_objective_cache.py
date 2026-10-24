import os
"""Read-only, deterministic schema probe; no labels, fitting or population estimates."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json, io, datetime
import numpy as np

BASE = Path(__file__).resolve().parent
REPO = Path(os.environ["LOL_RESEARCH_ROOT"])
EXT = Path(os.environ["LOL_EXTERNAL_ROOT"])
SOURCES = {
 'main_kr': (Path(os.environ["LOL_LEGACY_CACHE_ROOT"]) / 'cache' / 'match_cache_fresh_v3_engage_status13_patch_index.json', Path(os.environ["LOL_LEGACY_CACHE_ROOT"]) / 'cache' / 'match_cache_fresh_v3_engage_status13'),
 'later_kr': (EXT/'match_cache_kr_15_18_22_patch_index.json', EXT/'cache_kr_15_18_22'),
 'kr_16': (EXT/'match_cache_kr_16_13_patch_index.json', REPO/'outputs/full_corpus_training_20260915/external/KR_16.13/cache'),
 'na_16': (EXT/'match_cache_na1_16_13_patch_index.json', EXT/'cache_na1_16_13'),
}
EXCLUDED='680e9e04eb2209561a1770da053e9188f8e382c5389ce63e052afd99c7f8c118'
N=64
def sha(b): return hashlib.sha256(b).hexdigest()
def selected(mid): return sha(('objective-schema-audit-v1:'+mid).encode())
def event_group(e):
 t=e.get('type','MISSING')
 return t+':'+str(e.get('monsterType','MISSING')) if t=='ELITE_MONSTER_KILL' else t
def normalize(o):
 if isinstance(o,Counter): return dict(sorted(o.items(),key=lambda kv:str(kv[0])))
 if isinstance(o,dict): return {str(k):normalize(v) for k,v in o.items()}
 if isinstance(o,list): return [normalize(v) for v in o]
 return o

out={'scope':'deterministic 64 matches per indexed region/patch; schema inspection, not full census',
 'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'n_per_group':N,
 'sampling':'ascending SHA256(objective-schema-audit-v1:match_id), excluding user-excluded canonical hash',
 'sources':{},'groups':{},'errors':[]}
ledger=[]
for source,(idxpath,root) in SOURCES.items():
 ib=idxpath.read_bytes(); idx=json.loads(ib)
 out['sources'][source]={'index':str(idxpath),'cache':str(root),'index_sha256':sha(ib),'indexed':len(idx)}
 bypatch=defaultdict(list)
 for mid,patch in idx.items():
  if sha(mid.encode())!=EXCLUDED: bypatch[patch].append(mid)
 for patch,ids in sorted(bypatch.items()):
  key=source+'/'+patch
  st={'selected':min(N,len(ids)), 'read':0, 'events':Counter(), 'field_presence':defaultdict(Counter),
      'values':defaultdict(Counter), 'monster_ownership':defaultdict(Counter), 'soul_records':[],
      'item_3513':Counter(), 'special_payload_keys':Counter(), 'damage_effect_tokens':Counter(),
      'frames':Counter(), 'patch_mismatch':0, 'npz_keys':Counter(), 'file_change':0,
      'duplicates_exact':0, 'errors':[], 'interesting_examples':{}}
  for mid in sorted(ids,key=selected)[:N]:
   mk=sha(mid.encode()); rec={'source':source,'patch':patch,'match_key_sha256':mk,'files':{}}
   try:
    data={}
    for ext in ['meta.json','events.json','npz']:
     p=root/(mid+'.'+ext); before=p.stat()
     if ext=='npz' and before.st_size>2_000_000: raise ValueError('NPZ exceeds small-file probe limit')
     b=p.read_bytes(); after=p.stat()
     stable=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
     st['file_change']+=int(not stable)
     rec['files'][ext]={'sha256':sha(b),'bytes':len(b),'stable_during_read':stable}
     data[ext]=b
    meta=json.loads(data['meta.json']); raw=json.loads(data['events.json'])
    es=raw if isinstance(raw,list) else raw['events']
    tm={int(k):int(v) for k,v in meta.get('team_map',{}).items()}
    st['patch_mismatch']+=int(meta.get('patch')!=patch)
    st['read']+=1
    seen=set(); dragons=Counter()
    for i,e in enumerate(es):
     group=event_group(e); typ=e.get('type'); st['events'][group]+=1
     fingerprint=json.dumps(e,sort_keys=True,separators=(',',':'))
     if fingerprint in seen: st['duplicates_exact']+=1
     seen.add(fingerprint)
     for field in e:
      st['field_presence'][group][field]+=1
      if any(s in field.lower() for s in ['gold','buff','petal','herald','bounty','health']): st['special_payload_keys'][field]+=1
     for field in ['monsterSubType','teamId','killerTeamId','bounty','shutdownBounty','goldGain','towerType','buildingType','name','featType','featValue']:
      if field in e: st['values'][group+'/'+field][str(e[field])]+=1
     if typ=='ELITE_MONSTER_KILL':
      kt=e.get('killerTeamId'); pid=e.get('killerId'); mapped=tm.get(pid)
      own=st['monster_ownership'][e.get('monsterType','MISSING')]
      own['n']+=1; own['killerTeam_valid']+=int(kt in (100,200)); own['killerId_mappable']+=int(mapped in (100,200))
      own['valid_team_disagreement']+=int(kt in (100,200) and mapped in (100,200) and kt!=mapped)
      own['invalid_team_with_valid_killer']+=int(kt not in (100,200) and mapped in (100,200))
      own['both_unresolved']+=int(kt not in (100,200) and mapped not in (100,200))
      if e.get('monsterType')=='DRAGON' and e.get('monsterSubType')!='ELDER_DRAGON':
       team=kt if kt in (100,200) else mapped
       dragons[team]+=1
     if typ=='DRAGON_SOUL_GIVEN':
      st['soul_records'].append({'match_key_sha256':mk,'timestamp':e.get('timestamp'),'teamId':e.get('teamId'),'name':e.get('name'),'preceding_elemental_counts':{str(k):v for k,v in dragons.items()}})
     if typ in ['ITEM_DESTROYED','ITEM_PURCHASED','ITEM_SOLD','ITEM_UNDO'] and 3513 in [e.get('itemId'),e.get('beforeId'),e.get('afterId')]:
      st['item_3513'][typ]+=1
     if typ=='CHAMPION_KILL':
      for field in ['victimDamageReceived','victimDamageDealt']:
       for hit in e.get(field,[]):
        for kk in ['name','spellName']:
         token=str(hit.get(kk,''))
         if any(s in token.lower() for s in ['elder','baron','herald','atakhan','petal','dragon','voidgrub']): st['damage_effect_tokens'][token]+=1
     if typ in ['ELITE_MONSTER_KILL','BUILDING_KILL','TURRET_PLATE_DESTROYED','DRAGON_SOUL_GIVEN','FEAT_UPDATE'] and group not in st['interesting_examples']:
      keep=['timestamp','type','killerId','killerTeamId','teamId','monsterType','monsterSubType','buildingType','towerType','bounty','shutdownBounty','goldGain','name','assistingParticipantIds']
      st['interesting_examples'][group]={k:e[k] for k in keep if k in e}
    with np.load(io.BytesIO(data['npz']),allow_pickle=False) as ar:
     for k in ar.files: st['npz_keys'][k]+=1
     ts=ar['minute_ts']; gold=ar['gold_team_minute']
     st['frames']['n']+=len(ts)
     st['frames']['nonfinite_team_gold']+=int((~np.isfinite(gold)).sum())
     st['frames']['gold_decreases']+=int((np.diff(gold,axis=0)<0).sum())
     for d in np.diff(ts): st['values']['frame_step_ms'][str(int(d))]+=1
     gameends=[e.get('timestamp') for e in es if e.get('type')=='GAME_END']
     if len(gameends)==1:
      gap=int(gameends[0]-ts[-1]); st['values']['terminal_minus_last_frame_ms'][str(gap)]+=1
    ledger.append(rec)
   except Exception as ex:
    err={'match_key_sha256':mk,'error':str(ex)}; st['errors'].append(err); out['errors'].append({'group':key,**err}); ledger.append(rec)
  out['groups'][key]=normalize(st)
  print(key, 'read=',st['read'],'events=',sum(st['events'].values()),'soul=',len(st['soul_records']),'errors=',len(st['errors']),flush=True)
out['script_sha256']=sha(Path(__file__).read_bytes())
(BASE/'audit_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
(BASE/'source_hashes.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
notebook={'cells':[{'cell_type':'markdown','metadata':{},'source':['# Objective cache schema probe\n','Bounded read-only inspection; not a representative effect estimate. Run the adjacent script to reproduce.']},
 {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':Path(__file__).read_text(encoding='utf-8').replace('BASE = Path(__file__).resolve().parent','BASE = Path.cwd()').replace("out['script_sha256']=sha(Path(__file__).read_bytes())","out['script_sha256']=sha((BASE/'audit_objective_cache.py').read_bytes())").splitlines(keepends=True)[:0]}], 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
# Notebook invokes the exact saved audit script, preserving one source of truth.
notebook['cells'][1]['source']=['from pathlib import Path\n','import runpy\n',"runpy.run_path(str(Path.cwd() / 'audit_objective_cache.py'), run_name='__main__')\n"]
(BASE/'audit_objective_cache.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
