import os
"""Independent raw-window reference: no production prefix sums or endpoint helper."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
RAW=Path(os.environ["LOL_RAW_MATCH_ROOT"])
FIELDS=['level','xp','currentGold','totalGold','minionsKilled','jungleMinionsKilled','health_fraction','power_fraction','health','healthMax','attackDamage','abilityPower','armor','magicResist','attackSpeed','movementSpeed','x','y']
EVENTS=['kills','assists','elem_dragon','elder_dragon','baron','atakhan','herald','horde','outer_turret','inner_turret','base_turret','nexus_turret','inhibitor','plate','item_purchased','item_sold','item_undo','ward_placed','ward_kill','level_up']

def raw_arrays(detail,tl,audit):
 team={p['participantId']:p['teamId'] for p in detail['info']['participants']};frames=tl['info']['frames'];events=[e for f in frames for e in f['events']];ts=np.array([f['timestamp'] for f in frames]);et=np.array([e['timestamp'] for e in events]);v=np.zeros((len(events),2,20),np.int64);bad=np.zeros_like(v,bool)
 for i,e in enumerate(events):
  ty=e['type'];credits=[]
  if ty=='CHAMPION_KILL':
   assists=e.get('assistingParticipantIds');assists=assists if isinstance(assists,list) else []
   credits=[(0,team.get(e.get('killerId')))]+[(1,team.get(pid)) for pid in set(assists)]
  elif ty=='ELITE_MONSTER_KILL':
   kind=e.get('monsterType');col={'BARON_NASHOR':4,'ATAKHAN':5,'RIFTHERALD':6,'HORDE':7}.get(kind)
   if kind=='DRAGON':col=3 if e.get('monsterSubType')=='ELDER_DRAGON' else (2 if str(e.get('monsterSubType','')).endswith('_DRAGON') else None)
   if col is not None:credits=[(col,e.get('killerTeamId') if e.get('killerTeamId') is not None else team.get(e.get('killerId')))]
  elif ty in ('BUILDING_KILL','TURRET_PLATE_DESTROYED'):
   col=13 if ty=='TURRET_PLATE_DESTROYED' else (12 if e.get('buildingType')=='INHIBITOR_BUILDING' else {'OUTER_TURRET':8,'INNER_TURRET':9,'BASE_TURRET':10,'NEXUS_TURRET':11}.get(e.get('towerType')))
   if col is not None:credits=[(col,300-e['teamId'] if e.get('teamId') in (100,200) else None)]
  else:
   rule={'ITEM_PURCHASED':(14,'participantId'),'ITEM_SOLD':(15,'participantId'),'ITEM_UNDO':(16,'participantId'),'WARD_PLACED':(17,'creatorId'),'WARD_KILL':(18,'killerId'),'LEVEL_UP':(19,'participantId')}.get(ty)
   if rule:credits=[(rule[0],team.get(e.get(rule[1])))]
  for col,tid in credits:
   if tid in (100,200):v[i,tid//100-1,col]+=1
 for key,ranges in audit['event_masks'].items():
  name=key.split(':')[0]
  if name in EVENTS:
   for a,b in ranges:bad[a:b+1,:,EVENTS.index(name)]=True
 pf=np.full((len(frames),10,18),np.nan,np.float32)
 for fi,f in enumerate(frames):
  for pid in range(1,11):
   d=f.get('participantFrames',{}).get(str(pid),{});stats=d.get('championStats',{});xy=d.get('position',{})
   data=[d.get(k,np.nan) for k in FIELDS[:6]]
   for a,b in [('health','healthMax'),('power','powerMax')]:
    x,y=stats.get(a),stats.get(b);data.append(x/y if isinstance(x,(int,float)) and isinstance(y,(int,float)) and y>0 else np.nan)
   data.extend(stats.get(k,np.nan) for k in FIELDS[8:16]);data.extend(xy.get(k,np.nan) for k in ('x','y'))
   pf[fi,pid-1]=[x if isinstance(x,(int,float)) else np.nan for x in data]
 transitions=[]
 for key,ranges in audit['field_masks'].items():
  name=key.split(':')[0];col=FIELDS.index(name)
  for a,b in ranges:
   for cell in range(a,b+1):
    fi,pid0=divmod(cell,10);pf[fi,pid0,col]=np.nan
    if name=='totalGold' and 'decrease_transition' in key:transitions.append((int(ts[fi]),team[pid0+1]//100-1))
 gold=np.column_stack([pf[:,[p-1 for p in team if team[p]==tid],3].sum(1) for tid in (100,200)])
 return team,frames,events,ts,et,v,bad,pf,gold,transitions

def equal(a,b,name):
 if np.asarray(a).dtype.kind in 'fc' or np.asarray(b).dtype.kind in 'fc':np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-5,equal_nan=True,err_msg=name)
 else:np.testing.assert_array_equal(a,b,err_msg=name)

def validate_match(records,audit):
 mid=audit['match_id'];db=(RAW/'detail'/f'{mid}.json').read_bytes();tb=(RAW/'timeline'/f'{mid}.json').read_bytes()
 assert hashlib.sha256(db).hexdigest()==audit['detail_sha256'] and hashlib.sha256(tb).hexdigest()==audit['timeline_sha256'],mid
 detail=json.loads(db);tl=json.loads(tb);team,frames,events,ts,et,v,bad,pf,gold,transitions=raw_arrays(detail,tl,audit)
 endings={(e['timestamp'],e.get('winningTeam')) for e in events if e['type']=='GAME_END'};winners=[x['teamId'] for x in detail['info']['teams'] if x.get('win') is True]
 terminal_ok=len(endings)==1 and len(winners)==1
 end,win=next(iter(endings)) if terminal_ok else (-1,-1);terminal_ok=terminal_ok and win==winners[0] and end==ts[-1]
 if not terminal_ok:end=win=-1
 ambiguous={i for key,rr in audit['event_masks'].items() if key.startswith('kills:ambiguous') for a,b in rr for i in range(a,b+1)}
 checks=Counter()
 for r in records:
  t0=int(r['t0_ms']);fi=list(ts).index(t0);tau=int(r['tau_ms']);members=json.loads(str(r['focal_member_ordinals_json']));first=int(r['first_kill_ms']);last=int(r['last_kill_ms'])
  assert tau==first-10000 and 0<=tau-t0<=10000 and tau>=120000
  assert members and min(et[members])==first and max(et[members])==last and len(members)==len(set(members))
  sel=not any(tau<=et[i]<=last or i in members for i in ambiguous);equal(sel,r['selection_supported'],'selection')
  hist=np.full((5,10,18),np.nan,np.float32);ht=np.full(5,-1,np.int64);take=min(5,fi+1);hist[-take:]=pf[fi-take+1:fi+1];ht[-take:]=ts[fi-take+1:fi+1]
  equal(hist,r['player_raw'],'history_values');equal(ht,r['history_ms'],'history_time');equal(np.isfinite(hist),r['player_support'],'history_support')
  equal([team[i] for i in range(1,11)],r['team_id'],'team');prefix=et<=t0;ps=v[prefix].sum(0).astype(float);psup=~bad[prefix].any(0);ps[~psup]=np.nan
  equal(ps,r['prefix_team_events'],'prefix');equal(psup,r['prefix_team_events_support'],'prefix_support')
  ranks=np.zeros((10,4),float);rank_ok=True;so=np.zeros(2);status=np.full(2,2);dragons=[0,0];conflict=False
  for idx in sorted(np.flatnonzero(prefix),key=lambda i:(et[i],i)):
   e=events[idx]
   if e['type']=='SKILL_LEVEL_UP':
    pid,slot=e.get('participantId'),e.get('skillSlot')
    if pid in team and slot in (1,2,3,4):ranks[pid-1,slot-1]+=1
    else:rank_ok=False
   if v[idx,:,2].sum():
    z=int(v[idx,:,2].argmax());dragons[z]+=1
    if dragons[z]==4:so[z]=1
   if e['type']=='DRAGON_SOUL_GIVEN' and e.get('teamId') in (100,200):
    z=e['teamId']//100-1;conflict|=dragons[z]<4;so[z]=1;status[z]=1
  if not rank_ok:ranks[:]=np.nan
  equal(ranks,r['skill_level_up_count'],'prefix_skill');equal(np.full((10,4),rank_ok),r['skill_level_up_count_support'],'prefix_skill_support')
  if not psup[:,2].all() or conflict or so.sum()>1:so[:]=np.nan;status[:]=0
  if any(key.startswith('soul:') and any(a<0 or any(et[i]<=t0 for i in range(a,b+1)) for a,b in rr) for key,rr in audit['event_masks'].items()):so[:]=np.nan;status[:]=0
  equal(so,r['soul_owned'],'prefix_soul');equal(status,r['soul_source_status'],'prefix_soul_support')
  endpoint=[]
  for seconds in (60,120,180):
   nominal=t0+seconds*1000
   if terminal_ok and end<=nominal:e=end
   else:
    future=ts[ts>=nominal];e=int(future[0]) if len(future) and future[0]-nominal<=5000 else -1
   endpoint.append(e)
  equal(endpoint,r['endpoint_ms'],'endpoints');equal(np.array(endpoint)>=0,r['endpoint_supported'],'endpoint_support')
  gv=np.full((4,2),np.nan)
  for k,t in enumerate([t0]+endpoint):
   found=np.flatnonzero(ts==t)
   if t>=0 and len(found):gv[k]=gold[found[0]]
  equal(gv,r['team_gold_endpoints'],'endpoint_gold')
  for k,e in enumerate(endpoint):
   for cumulative in (True,False):
    start=t0 if cumulative else ([t0]+endpoint)[k];mask=(et>start)&(et<=e);support=(~bad[mask].any(0))&(start>=0)&(e>=0)&sel;counts=v[mask].sum(0).astype(float);counts[~support]=np.nan
    label='cum' if cumulative else 'slice';equal(counts,r[label+'_team_events'][k],label+' events');equal(support,r[label+'_event_supported'][k],label+' event support')
    g=gv[k+1]-gv[0 if cumulative else k]
    for t,z in transitions:
     if start>=0 and start<t<=e:g[z]=np.nan
    if not sel:g[:]=np.nan
    equal(g,r[label+'_team_gold'][k],label+' gold');equal(np.isfinite(g),r['gold_supported' if cumulative else 'slice_gold_supported'][k],label+' gold support')
   terminal=win//100 if terminal_ok and e>=end else 0
   if not terminal_ok or e<0 or not sel:terminal=-1
   equal(terminal,r['terminal_class'][k],'terminal_class');equal(terminal>=0,r['terminal_supported'][k],'terminal_support')
   previous=([t0]+endpoint)[k];at_risk=terminal_ok and 0<=previous<end and e>=0 and sel;equal(at_risk,r['hazard_at_risk_supported'][k],'hazard_support');equal(terminal if at_risk else -1,r['hazard_class'][k],'hazard_class')
  focal=np.zeros(2,float);fs=sel
  for i in members:
   assert events[i]['type']=='CHAMPION_KILL';pid=events[i].get('killerId');tid=team.get(pid)
   if tid in (100,200):focal[tid//100-1]+=1
   elif pid!=0:fs=False
  if not fs:focal[:]=np.nan
  equal(focal,r['focal_team_kills'],'focal');equal([fs,fs],r['focal_kills_supported'],'focal_support')
  net=r['cum_team_gold'][:,0]-r['cum_team_gold'][:,1];equal(net,r['net_gold'],'net');equal(np.where(np.isfinite(net),np.sign(net),-9).astype(np.int8),r['economic_sign'],'economic_sign')
  for k in range(3):
   x=np.r_[net[k],r['cum_team_events'][k,0,2:14]-r['cum_team_events'][k,1,2:14]];klass=-1
   if np.isfinite(x).all():klass=3 if np.all(x==0) else 0 if np.all(x>=0) else 1 if np.all(x<=0) else 2
   equal(klass,r['pareto_class'][k],'pareto');equal(klass>=0,r['pareto_supported'][k],'pareto_support')
  checks['raw_query_recalculations']+=1;checks['raw_horizon_recalculations']+=3
 return dict(checks)
