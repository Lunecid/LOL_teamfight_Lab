"""Full FIT raw audit. Read-only source; explicit dispositions, no training ledger.
Three workers + parent aggregator, deterministic frozen hash split. No outcome trimming.
"""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import os, json, hashlib, re, math, time, argparse, sys
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RAW=Path(os.environ["LOL_RAW_MATCH_ROOT"])
R2=ROOT/'multiwindow_r2'
sys.path.insert(0,str(R2))
import producer
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'holdout_r3'))
from cohort_policy import load_policy, role_for
from split import bucket
POLICY=load_policy()
PATCHES=tuple(POLICY['patch_rules'])
FIELDS=producer.PLAYER
EVENTS=producer.EVENTS
_CATALOG={}
for _p in PATCHES:
 _d=json.loads((R2/f'knowledge_r2/{_p}/sources/championFull_{_p}.1.json').read_bytes())
 _CATALOG[_p]={int(v['key']):v for v in _d['data'].values()}
_FD=json.loads((ROOT/'literature_curation_gate1_20261018/FEATURE_DISTRIBUTIONS.json').read_bytes())
FEATURE_EDGES=[np.r_[-np.inf,np.array(_FD['histogram_data'][f]['edges']),np.inf] for f in FIELDS]
FEATURE_BINS=len(FEATURE_EDGES[1])-1
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(mid):return mid.upper() if '_' in mid else 'KR_'+mid
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def compress(indices):
 out=[]
 for i in sorted(set(map(int,indices))):
  if out and i==out[-1][1]+1:out[-1][1]=i
  else:out.append([i,i])
 return out
def base(mid):return {'match_id':mid,'bucket':bucket(mid),'patch':None,'hard_reasons':[],'population_reasons':[],'field_masks':{},'event_masks':{},'diagnostics':{},'metrics':{}}
def analyze(mid,detail,timeline,row=None):
 r=row or base(mid);hard=r['hard_reasons'];pop=r['population_reasons'];diag=Counter();fm=defaultdict(list);em=defaultdict(list)
 info=detail.get('info',{});ti=timeline.get('info',{});patch='.'.join(str(info.get('gameVersion','')).split('.')[:2]);r['patch']=patch
 if detail.get('metadata',{}).get('matchId')!=mid or timeline.get('metadata',{}).get('matchId')!=mid:hard.append('match_identity_conflict')
 if role_for(patch,bucket(mid),POLICY)[1]!='FIT':r['scope']='OTHER_ROLE';return finish(r,diag,fm,em),None
 if info.get('gameId')!=ti.get('gameId'):hard.append('numeric_game_identity_conflict')
 if any(info.get(k)!=v for k,v in [('queueId',420),('mapId',11),('gameMode','CLASSIC')]):pop.append('not_q420_map11_classic')
 pp=info.get('participants');frames=ti.get('frames')
 if not isinstance(pp,list) or len(pp)!=10 or any(not isinstance(p,dict) for p in pp):hard.append('invalid_participant_list');return finish(r,diag,fm,em),None
 team={p.get('participantId'):p.get('teamId') for p in pp}
 if set(team)!=set(range(1,11)) or Counter(team.values())!={100:5,200:5}:hard.append('invalid_core_team_membership');return finish(r,diag,fm,em),None
 # Compare private identifiers only in memory; never serialize them.
 dp={p.get('participantId'):p.get('puuid') for p in pp};tp=ti.get('participants')
 if isinstance(tp,list):
  tpmap={p.get('participantId'):p.get('puuid') for p in tp if isinstance(p,dict)}
  if set(tpmap)!=set(team) or any(dp[k]!=tpmap.get(k) for k in team):hard.append('participant_identity_conflict')
 else:diag['timeline_participant_identity_missing']+=1
 if not isinstance(frames,list) or not frames or any(not isinstance(f,dict) for f in frames):hard.append('invalid_frame_list');return finish(r,diag,fm,em),None
 ts=[f.get('timestamp') for f in frames]
 if any(type(t) is not int or t<0 for t in ts) or ts[0]!=0 or any(b<=a for a,b in zip(ts,ts[1:])):hard.append('invalid_frame_timestamps');return finish(r,diag,fm,em),None
 ts=np.array(ts,np.int64);raw_events=[]
 for fi,f in enumerate(frames):
  ee=f.get('events')
  if not isinstance(ee,list) or any(not isinstance(e,dict) for e in ee):hard.append('invalid_event_list');continue
  raw_events.extend(ee)
 if any(type(e.get('timestamp')) is not int or not 0<=e['timestamp']<=ts[-1] for e in raw_events):hard.append('invalid_event_timestamps');return finish(r,diag,fm,em),None
 et=np.array([e['timestamp'] for e in raw_events],np.int64)
 diag['raw_event_timestamp_reversals']=int((np.diff(et)<0).sum()) # stable-sortable; not corruption by itself.
 endings={(e['timestamp'],e.get('winningTeam')) for e in raw_events if e.get('type')=='GAME_END'}
 winners=[p.get('teamId') for p in info.get('teams',[]) if isinstance(p,dict) and p.get('win') is True]
 terminal_ok=False;end=int(ts[-1])
 if len(endings)>1 or len(winners)>1:hard.append('conflicting_terminal_evidence')
 elif len(endings)==1 and len(winners)==1:
  end,wt=next(iter(endings));terminal_ok=wt==winners[0] and wt in (100,200) and end==ts[-1]
  if not terminal_ok:hard.append('terminal_winner_or_time_conflict')
 else:diag['terminal_evidence_missing_mask_required']+=1
 duration=info.get('gameDuration')
 if terminal_ok:
  if end<300000:pop.append('existing_duration_lt_300s')
  if finite(duration) and abs(end/1000-duration)>2:diag['detail_duration_vs_terminal_over_2s']+=1
 elif finite(duration) and duration<300:pop.append('existing_duration_lt_300s')
 diag['frame_gap_over_65000ms']=int((np.diff(ts)>65000).sum())
 diag['nonterminal_frame_gap_not_60s']=int((np.abs(np.diff(ts)[:-1]-60000)>1000).sum())
 champ={p['participantId']:p.get('championId') for p in pp}
 for pid,c in champ.items():
  if c not in _CATALOG.get(patch,{}):diag['champion_catalog_unknown_slots']+=1
 r['champion_ids']=[champ[k] for k in range(1,11)]
 # sanitize malformed optional frame fields, retaining precise source coordinates.
 vv=[]
 for fi,f in enumerate(frames):
  pf=f.get('participantFrames');pf=pf if isinstance(pf,dict) else {}
  cooked={}
  for pid in range(1,11):
   cell=pf.get(str(pid));cell=dict(cell) if isinstance(cell,dict) else {}
   if cell.get('participantId',pid)!=pid:hard.append('frame_participant_identity_conflict')
   for key in ('championStats','position'):
    cell[key]=dict(cell[key]) if isinstance(cell.get(key),dict) else {}
   for key in FIELDS[:6]:
    if key in cell and not finite(cell[key]):fm[key+':invalid_numeric_type'].append(fi*10+pid-1);cell[key]=None
   for key in list(FIELDS[8:16])+['power','powerMax']:
    if key in cell['championStats'] and not finite(cell['championStats'][key]):
     name='power_fraction' if key in ('power','powerMax') else key
     fm[name+':invalid_numeric_type'].append(fi*10+pid-1);cell['championStats'][key]=None
   for key in ('x','y'):
    if key in cell['position'] and not finite(cell['position'][key]):fm[key+':invalid_numeric_type'].append(fi*10+pid-1);cell['position'][key]=None
   cooked[str(pid)]=cell
  vv.append(producer.player_frame({'participantFrames':cooked}))
 v=np.array(vv,np.float32);support=np.isfinite(v);valid=support.copy();nframes=len(v)
 for j,name in enumerate(FIELDS):
  missing=~support[:,:,j]
  fm[name+':missing_or_undefined'].extend(np.flatnonzero(missing))
  invalid=np.zeros((nframes,10),bool)
  if name=='level':invalid=support[:,:,j]&((v[:,:,j]<1)|(v[:,:,j]>18)|(v[:,:,j]!=np.floor(v[:,:,j])))
  elif name in ('health_fraction','power_fraction'):invalid=support[:,:,j]&((v[:,:,j]<0)|(v[:,:,j]>1.00001))
  elif name in ('xp','currentGold','totalGold','minionsKilled','jungleMinionsKilled','health','healthMax'):invalid=support[:,:,j]&(v[:,:,j]<0)
  if invalid.any():fm[name+':outside_hard_field_domain'].extend(np.flatnonzero(invalid));valid[:,:,j]&=~invalid
 for j,name in [(1,'xp'),(3,'totalGold'),(4,'minionsKilled'),(5,'jungleMinionsKilled')]:
  dec=(np.diff(v[:,:,j],axis=0)<0)&valid[1:,:,j]&valid[:-1,:,j]
  if dec.any():
   diag[name+'_decrease_transitions']=int(dec.sum())
   # A transition-level mask, not a claim that either endpoint is individually false.
   fm[name+':decrease_transition_ending_cell'].extend(np.flatnonzero(dec)+10)
 for j,name in [(12,'negative_armor_retained'),(13,'negative_magic_resist_retained')]:diag[name]=int((support[:,:,j]&(v[:,:,j]<0)).sum())
 # Source event arrays preserve r2 attribution semantics; unsupported events remain sparse coordinates.
 vals,bad=producer.event_arrays(raw_events,team)
 for ordinal,e in enumerate(raw_events):
  if e.get('type')=='CHAMPION_KILL' and 'assistingParticipantIds' not in e:bad[ordinal,:,1]=True
 for j,name in enumerate(EVENTS):
  ix=np.flatnonzero(bad[:,:,j].any(axis=1))
  if len(ix):em[name+':unattributed'].extend(ix)
 types=Counter(e.get('type','UNKNOWN') for e in raw_events)
 soul=[0,0];soul_time=None;last={};rule_flags=Counter();rule_coordinates=[];seen_kills={}
 ordered=sorted(enumerate(raw_events),key=lambda x:(x[1]['timestamp'],x[0]))
 for ordinal,e in ordered:
  typ=e.get('type');t=e['timestamp']
  if typ=='CHAMPION_KILL':
   if e.get('victimId') in team:
    sig=(t,e['victimId'])
    if sig in seen_kills:
     previous=seen_kills[sig];em['kills:ambiguous_same_timestamp_victim'].extend([previous,ordinal]);em['assists:ambiguous_same_timestamp_victim'].extend([previous,ordinal]);diag['ambiguous_kill_same_timestamp_victim']+=1
    seen_kills[sig]=ordinal
   else:em['kills:unknown_victim'].append(ordinal)
  if typ=='SKILL_LEVEL_UP' and (e.get('participantId') not in team or e.get('skillSlot') not in (1,2,3,4)):em['skill_level_up:invalid_actor_or_slot'].append(ordinal)
  if typ=='DRAGON_SOUL_GIVEN':
   tid=e.get('teamId')
   if tid==0:diag['team0_rift_announcement_ignored']+=1
   elif tid in (100,200):
    if soul[tid//100-1]<4:em['soul:inconsistent_prefix'].append(ordinal)
    if soul_time is None:soul_time=t
   else:em['soul:unknown_team'].append(ordinal)
  cats=np.flatnonzero(vals[ordinal].sum(axis=0)[2:8]) if typ=='ELITE_MONSTER_KILL' else []
  if len(cats)==1:
   col=int(cats[0])+2;name=EVENTS[col]
   if col==2:
    z=int(np.argmax(vals[ordinal,:,col]));soul[z]+=1
    if soul[z]==4 and soul_time is None:soul_time=t
   first={2:300000,3:None,4:1500000,5:1200000,6:900000,7:480000}[col]
   reasons=[]
   if first is not None and t<first:reasons.append(name+':before_inherited_first_spawn')
   if col==3 and (soul_time is None or t-soul_time<360000):reasons.append('elder:before_inherited_soul_delay')
   if col in (2,3,4) and col in last and t-last[col] < (300000 if col==2 else 360000):reasons.append(name+':short_inherited_respawn')
   for reason in reasons:
    rule_flags[reason]+=1;rule_coordinates.append({'reason':reason,'timestamp':t,'raw_ordinal':ordinal,'affected_field':'objective_availability_rule:'+name,'action':'derived_rule_unknown_from_this_prefix_not_acquisition_deletion'})
   last[col]=t
 if all(x>=4 for x in soul):em['soul:both_teams_have_four_dragons'].append(-1)
 # Patch rule contradiction masks rules, never silently deletes observed acquisitions.
 diag.update(rule_flags)
 if rule_flags:r['rule_masks']=rule_coordinates
 g=v[:,:,3];gv=valid[:,:,3]
 blue=[i-1 for i in range(1,11) if team[i]==100];red=[i-1 for i in range(1,11) if team[i]==200]
 dif=g[:,blue].sum(axis=1)-g[:,red].sum(axis=1);oks=gv.all(axis=1)
 deltas=np.diff(dif);deltaok=oks[1:]&oks[:-1]
 r['metrics']={'duration_s':float(end/1000) if terminal_ok else None,'last_observed_frame_s':float(ts[-1]/1000),'detail_duration_s':float(duration) if finite(duration) else None,'n_frames':nframes,'n_events':len(raw_events),'n_kills':types['CHAMPION_KILL'],
  'terminal_verified':bool(terminal_ok),'winning_team':winners[0] if len(winners)==1 else None,
  'final_gold_difference':float(dif[-1]) if oks[-1] else None,'max_abs_frame_gold_change':float(np.max(np.abs(deltas[deltaok]))) if deltaok.any() else None,
  'epic_acquisition_counts':vals.sum(axis=(0,1))[2:8].astype(int).tolist(),'unknown_event_counts':bad.any(axis=1).sum(axis=0).astype(int).tolist(),
  'player_cells_per_feature':nframes*10,'supported_player_cells':valid.sum(axis=(0,1)).astype(int).tolist(),
  'negative_armor_cells':diag['negative_armor_retained'],'negative_mr_cells':diag['negative_magic_resist_retained']}
 # Histograms include explicit underflow/overflow, never discard tails.
 hs=[];mins=[];maxs=[]
 for j in range(18):
  x=v[:,:,j][valid[:,:,j]];hs.append(np.histogram(x,bins=FEATURE_EDGES[j])[0]);mins.append(float(x.min()) if len(x) else None);maxs.append(float(x.max()) if len(x) else None)
 r['metrics']['feature_min']=mins;r['metrics']['feature_max']=maxs
 r=finish(r,diag,fm,em)
 return r,hs
def finish(r,diag,fm,em):
 r['hard_reasons']=sorted(set(r['hard_reasons']));r['population_reasons']=sorted(set(r['population_reasons']))
 r['diagnostics']={k:int(v) for k,v in diag.items() if v}
 r['field_masks']={k:compress(v) for k,v in fm.items() if v}
 r['event_masks']={k:compress(v) for k,v in em.items() if v}
 r['disposition']='QUARANTINE' if r['hard_reasons'] else ('EXCLUDE_POPULATION' if r['population_reasons'] else ('RETAIN_MASKED' if r['field_masks'] or r['event_masks'] or r.get('rule_masks') or diag.get('terminal_evidence_missing_mask_required') else 'RETAIN'))
 return r
def audit_file(job):
 if isinstance(job,str):mid=canonical(job);ds=[job];tls=[job]
 else:mid,ds,tls=job
 r=base(mid);r['source_files']={'detail':ds,'timeline':tls}
 if not ds:
  r['scope']='UNRESOLVED';r['hard_reasons'].append('missing_detail_scope_unresolved')
  r['orphan_timeline_hashes']={s:sha((RAW/'timeline'/f'{s}.json').read_bytes()) for s in tls}
  return finish(r,Counter(),{},{}),None
 p=RAW/'detail'/f'{ds[0]}.json'
 try:
  with p.open('rb') as f:
   before=os.fstat(f.fileno());head=f.read(8192)
   hit=re.search(rb'"gameVersion"\s*:\s*"([^"]+)"',head)
   if hit:
    patch='.'.join(hit[1].decode('ascii',errors='replace').split('.')[:2]);r['patch']=patch
    if patch not in PATCHES:return {'match_id':mid,'patch':patch,'scope':'OTHER_PATCH'},None
   blob=head+f.read();after=os.fstat(f.fileno())
  r.update(detail_sha256=sha(blob),detail_bytes=len(blob))
  for alias in ds[1:]:
   ab=(RAW/'detail'/f'{alias}.json').read_bytes()
   if sha(ab)!=r['detail_sha256']:r['hard_reasons'].append('conflicting_duplicate_detail')
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):r['hard_reasons'].append('detail_changed_during_read')
  detail=json.loads(blob)
  if not isinstance(detail,dict):raise ValueError('detail_not_object')
  actual='.'.join(str(detail.get('info',{}).get('gameVersion','')).split('.')[:2])
  if actual not in PATCHES:
   if not actual or actual in ('None',''):r['hard_reasons'].append('scope_unresolved_missing_patch');r['scope']='UNRESOLVED'
   else:return {'match_id':mid,'patch':actual,'scope':'OTHER_PATCH'},None
  else:
   r['patch']=actual;r['scope']='FIT_CANDIDATE';dr=detail.get('info',{}).get('gameDuration')
   r['metrics']['detail_duration_s']=float(dr) if finite(dr) else None
 except Exception as e:
  r['hard_reasons'].append('detail_read_or_parse_error:'+type(e).__name__);r['scope']='FIT_CANDIDATE' if r['patch'] in PATCHES else 'UNRESOLVED'
  return finish(r,Counter(),{},{}),None
 if r.get('scope')=='UNRESOLVED':return finish(r,Counter(),{},{}),None
 p=RAW/'timeline'/f'{tls[0] if tls else mid}.json'
 try:
  with p.open('rb') as f:
   before=os.fstat(f.fileno());blob=f.read();after=os.fstat(f.fileno())
  r.update(timeline_sha256=sha(blob),timeline_bytes=len(blob))
  for alias in tls[1:]:
   if sha((RAW/'timeline'/f'{alias}.json').read_bytes())!=r['timeline_sha256']:r['hard_reasons'].append('conflicting_duplicate_timeline')
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):r['hard_reasons'].append('timeline_changed_during_read')
  timeline=json.loads(blob)
  if not isinstance(timeline,dict):raise ValueError('timeline_not_object')
 except Exception as e:
  r['hard_reasons'].append('timeline_read_or_parse_error:'+type(e).__name__);return finish(r,Counter(),{},{}),None
 try:return analyze(mid,detail,timeline,r)
 except Exception as e:
  r['auditor_error']=type(e).__name__+':'+str(e)[:160];r['disposition']='AUDITOR_ERROR'
  return r,None
def batch(mids):
 rows=[];scope=Counter();hpre=[np.zeros(len(e)-1,np.int64) for e in FEATURE_EDGES];hpost=[x.copy() for x in hpre]
 for mid in mids:
  r,hs=audit_file(mid)
  if r.get('scope')=='OTHER_PATCH':scope['OTHER_PATCH:'+r['patch']]+=1;continue
  rows.append(r)
  if hs is not None:
   for j,h in enumerate(hs):
    hpre[j]+=h
    if r['disposition'].startswith('RETAIN'):hpost[j]+=h
 return rows,dict(scope),hpre,hpost
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--limit',type=int);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
 out=Path(args.output);out.mkdir(exist_ok=False,parents=True)
 start=time.time();names={}
 for kind in ('detail','timeline'):
  with os.scandir(RAW/kind) as it:names[kind]=sorted(e.name[:-5] for e in it if e.is_file() and e.name.endswith('.json'))
 groups=defaultdict(lambda:{'detail':[],'timeline':[]})
 for kind in names:
  for stem in names[kind]:groups[canonical(stem)][kind].append(stem)
 mids=[(m,groups[m]['detail'],groups[m]['timeline']) for m in sorted(groups) if any(role_for(p,bucket(m),POLICY)[1]=='FIT' for p in PATCHES)]
 total_candidates=len(mids)
 inv={'raw_root':str(RAW),'detail_files':len(names['detail']),'timeline_files':len(names['timeline']),'detail_name_list_sha256':sha('\n'.join(names['detail']).encode()),'timeline_name_list_sha256':sha('\n'.join(names['timeline']).encode()),'detail_without_timeline':sorted(set(names['detail'])-set(names['timeline'])),'timeline_without_detail':sorted(set(names['timeline'])-set(names['detail'])),'hash_eligible_unique_keys':len(mids),'hash_outside_fit_bucket':len(groups)-len(mids),'canonical_duplicate_groups':sum(any(len(g[k])>1 for k in ('detail','timeline')) for g in groups.values()),'split_policy':'operator-supplied local cohort policy','limit':args.limit,'workers':args.workers}
 dump(out/'INVENTORY.json',inv)
 if args.limit:mids=mids[:args.limit]
 done=0;nr=0;status=Counter();scopes=Counter();histpre=[np.zeros(len(e)-1,np.int64) for e in FEATURE_EDGES];histpost=[x.copy() for x in histpre]
 jobs=[mids[i:i+64] for i in range(0,len(mids),64)]
 with (out/'MATCH_AUDIT.jsonl').open('w',encoding='utf-8') as log,ProcessPoolExecutor(max_workers=args.workers) as pool:
  for i,(rows,sc,hp,hq) in enumerate(pool.map(batch,jobs,chunksize=1)):
   for r in rows:log.write(json.dumps(r,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n');status[r['disposition']]+=1
   scopes.update(sc);nr+=len(rows);done+=len(jobs[i])
   for j in range(18):histpre[j]+=hp[j];histpost[j]+=hq[j]
   if i%20==0 or i==len(jobs)-1:
    log.flush();progress={'scanned_hash_eligible':done,'total_hash_eligible':len(mids),'audited_fit_or_unresolved':nr,'dispositions':dict(status),'other_patches':dict(scopes),'elapsed_s':round(time.time()-start,1)};dump(out/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
 np.savez_compressed(out/'FEATURE_HISTOGRAMS.npz',**{f'{prefix}_{j}':a for prefix,ls in [('before',histpre),('after',histpost),('edges',FEATURE_EDGES)] for j,a in enumerate(ls)})
 dump(out/'RUN_COMPLETE.json',{'completed':status['AUDITOR_ERROR']==0,'auditor_error_count':status['AUDITOR_ERROR'],'limited_smoke':args.limit is not None,'elapsed_s':round(time.time()-start,1),'python':sys.version,'numpy':np.__version__,'script_sha256':sha(Path(__file__).read_bytes()),'r2_producer_sha256':sha((R2/'producer.py').read_bytes()),'all_hash_eligible_scanned':done==total_candidates and args.limit is None,'masks_encoding':'field ranges inclusive over frame_index*10+(participant_id-1); transition masks name ending cell. Event ranges inclusive raw flattened ordinal; -1 means global contradiction. Rule masks carry timestamp and ordinal. Apply prefix-bounded at query time.','new_training_ledger_created':False})
if __name__=='__main__':main()
