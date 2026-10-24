from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib

BASE=Path(__file__).resolve().parent
d=json.loads((BASE/'audit_results.json').read_text(encoding='utf-8'))
out={'scope':d['scope'],'groups':{},'unresolved_herald_examples':[],'item3513_examples':[]}
def sh(s):return hashlib.sha256(s.encode()).hexdigest()
for source,src in d['sources'].items():
 idx=json.loads(Path(src['index']).read_text()); root=Path(src['cache']); groups=defaultdict(list)
 for mid,p in idx.items():
  if sh(mid)!='680e9e04eb2209561a1770da053e9188f8e382c5389ce63e052afd99c7f8c118':groups[p].append(mid)
 for patch,ids in groups.items():
  st={'duplicate_types':Counter(),'soul_checks':Counter(),'3513_checks':Counter(),'plate':Counter(),'gold_keys':Counter(),'damage_tokens':Counter(),'herald_unknown_times':Counter(),'objective_bounty_keys':Counter()}
  for mid in sorted(ids,key=lambda m:sh('objective-schema-audit-v1:'+m))[:64]:
   es=json.loads((root/(mid+'.events.json')).read_text()); seen=set()
   elite=[e for e in es if e['type']=='ELITE_MONSTER_KILL']
   souls=[e for e in es if e['type']=='DRAGON_SOUL_GIVEN']
   items=[e for e in es if e['type']=='ITEM_DESTROYED' and e.get('itemId')==3513]
   for e in es:
    t=e['type']; sig=json.dumps(e,sort_keys=True)
    if sig in seen:st['duplicate_types'][t]+=1
    seen.add(sig)
    if t=='TURRET_PLATE_DESTROYED':
     st['plate']['n']+=1;st['plate']['after_14m']+=int(e['timestamp']>=840000)
     st['plate']['killer_zero']+=int(e.get('killerId')==0)
    for k in e:
     if any(q in k.lower() for q in ['gold','petal','buff']):st['gold_keys'][t+'/'+k]+=1
    if t.startswith('OBJECTIVE_BOUNTY'):st['objective_bounty_keys'][t+':'+','.join(sorted(e))]+=1
    if t=='CHAMPION_KILL':
     # Received damage only; token counts are not buff proc or kill counts.
     for hit in e.get('victimDamageReceived',[]):
      tok=str(hit.get('spellName',''))
      if tok in ['elderdragonbuff','srt_2025_atakhan_roseboundreward','srx_dragonsoulbuffhextech','srx_dragonsoulbuffinfernal']:st['damage_tokens'][tok]+=1
   for e in souls:
    prev=[x for x in elite if x.get('monsterType')=='DRAGON' and x.get('monsterSubType')!='ELDER_DRAGON' and x['timestamp']<=e['timestamp']]
    n=sum(x.get('killerTeamId')==e['teamId'] for x in prev)
    st['soul_checks']['n']+=1
    if e['teamId']==0:
     st['soul_checks']['team0']+=1;st['soul_checks']['team0_after_exactly_2_elementals']+=int(len(prev)==2)
    else:
     st['soul_checks']['team_valid']+=int(e['teamId'] in [100,200]); st['soul_checks']['team_valid_fourth']+=int(n==4)
   for e in items:
    prev=[x for x in elite if x.get('monsterType')=='RIFTHERALD' and x['timestamp']<=e['timestamp']]
    st['3513_checks']['destroyed']+=1;st['3513_checks']['after_herald']+=int(bool(prev))
    if len(out['item3513_examples'])<3:out['item3513_examples'].append({'source':source,'patch':patch,'event':e,'herald_to_destroy_ms':e['timestamp']-prev[-1]['timestamp'] if prev else None})
   for e in elite:
    if e.get('monsterType')=='RIFTHERALD' and e.get('killerTeamId') not in [100,200]:
     st['herald_unknown_times'][str(e['timestamp'])]+=1
     out['unresolved_herald_examples'].append({'source':source,'patch':patch,'event':e,'other_heralds':[{k:x.get(k) for k in ['timestamp','killerId','killerTeamId']} for x in elite if x.get('monsterType')=='RIFTHERALD'],'eye_destroy_times':[x['timestamp'] for x in items]})
  out['groups'][source+'/'+patch]=st
(BASE/'semantic_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
for k,s in out['groups'].items():print(k,json.dumps(s,ensure_ascii=False))
print('HERALD_EXAMPLES',json.dumps(out['unresolved_herald_examples'][:4],ensure_ascii=False))
print('EYE_EXAMPLES',json.dumps(out['item3513_examples'],ensure_ascii=False))
