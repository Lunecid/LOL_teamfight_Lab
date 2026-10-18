"""Raw-backed audited FIT MW60_FRAME_r3. No estimator fitting; original artifacts are read-only.

Run with --output <new directory>. Existing output directories are refused.
The bank supplies cohort identity/component membership only, never missing raw values.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
RESEARCH = Path(os.environ["LOL_RESEARCH_ROOT"])
sys.path.insert(0,str(RESEARCH))
from studies.observable_state_20261002.population import build_population
AUDIT = HERE/'audit_01/MATCH_AUDIT.jsonl'
from split import roles, bucket, policy
PREP = ROOT / 'multipatch_v5_prep'
BANK = Path(os.environ["LOL_EXTERNAL_ROOT"]) / 'bank' / 'kr_15_18_22'
RAW = Path(os.environ["LOL_RAW_MATCH_ROOT"])
CURATION = Path(os.environ['LOL_SPLIT_ASSIGNMENTS_PATH'])
QUERY = PREP / 'window_evidence_v5/queries_by_objective/Q_ALL_PRIMARY_TIMELINES.npz'
EVENTS = ['kills','assists','elem_dragon','elder_dragon','baron','atakhan','herald','horde',
          'outer_turret','inner_turret','base_turret','nexus_turret','inhibitor','plate',
          'item_purchased','item_sold','item_undo','ward_placed','ward_kill','level_up']
PLAYER = ['level','xp','currentGold','totalGold','minionsKilled','jungleMinionsKilled',
          'health_fraction','power_fraction','health','healthMax','attackDamage','abilityPower',
          'armor','magicResist','attackSpeed','movementSpeed','x','y']
OBJECTIVES = ['dragon','elder','horde','herald','atakhan','baron']
EVENT_OBJECTIVE = [2,3,7,6,5,4]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def dump(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def read(p):
    b=Path(p).read_bytes()
    return json.loads(b),hashlib.sha256(b).hexdigest()

def member_registry(entries, wanted):
    result={}; groups=defaultdict(list)
    for e in entries:
        if e['match_id'] in wanted: groups[e['shard']].append(e)
    for shard, rows in groups.items():
        with np.load(BANK/shard/'BANK.npz',allow_pickle=False) as b:
            keys=['component_values','component_member_offsets','component_member_ordinals']
            a={k:b[k] for k in keys+[('offset_'+k) for k in keys]}
        for e in rows:
            i=e['local_match']; v={}
            for k in keys:
                o=a['offset_'+k];v[k]=a[k][o[i]:o[i+1]]
            comp={};o=v['component_member_offsets']
            for j,row in enumerate(v['component_values']):
                comp[int(row[0])]={'members':v['component_member_ordinals'][o[j]:o[j+1]].tolist(),
                                  'f':int(row[3]),'last':int(row[4]),'t0':int(row[6])}
            result[e['match_id']]=comp
    return result

def event_arrays(events,team):
    n=len(events); val=np.zeros((n,2,len(EVENTS)),np.int32); bad=np.zeros_like(val,bool)
    def credit(i,col,pid=None,tid=None,zero_ok=False):
        if tid is None: tid=team.get(pid)
        if tid in (100,200):val[i,tid//100-1,col]+=1
        elif not zero_ok:bad[i,:,col]=True
    for i,e in enumerate(events):
        ty=e.get('type'); pid=e.get('killerId'); tid=e.get('killerTeamId')
        if ty=='CHAMPION_KILL':
            credit(i,0,pid,zero_ok=(pid==0))
            ass=e.get('assistingParticipantIds',[])
            if isinstance(ass,list):
                for p in set(ass): credit(i,1,p)
            else: bad[i,:,1]=True
        elif ty=='ELITE_MONSTER_KILL':
            mt=e.get('monsterType'); st=e.get('monsterSubType')
            col={'BARON_NASHOR':4,'ATAKHAN':5,'RIFTHERALD':6,'HORDE':7}.get(mt)
            if mt=='DRAGON': col=3 if st=='ELDER_DRAGON' else (2 if st and st.endswith('_DRAGON') else None)
            if col is None: bad[i,:,2:8]=True
            else: credit(i,col,pid,tid)
        elif ty in ('BUILDING_KILL','TURRET_PLATE_DESTROYED'):
            owner=e.get('teamId'); acq=300-owner if owner in (100,200) else None
            col=13 if ty=='TURRET_PLATE_DESTROYED' else (12 if e.get('buildingType')=='INHIBITOR_BUILDING' else {'OUTER_TURRET':8,'INNER_TURRET':9,'BASE_TURRET':10,'NEXUS_TURRET':11}.get(e.get('towerType')))
            if col is not None:credit(i,col,tid=acq)
            elif e.get('buildingType')!='NEXUS_BUILDING':bad[i,:,8:13]=True
        elif ty in ('ITEM_PURCHASED','ITEM_SOLD','ITEM_UNDO','LEVEL_UP'):
            credit(i,{'ITEM_PURCHASED':14,'ITEM_SOLD':15,'ITEM_UNDO':16,'LEVEL_UP':19}[ty],e.get('participantId'))
        elif ty=='WARD_PLACED':credit(i,17,e.get('creatorId'))
        elif ty=='WARD_KILL':credit(i,18,pid)
    return val,bad

def endpoints(ts,t0,end):
    out=[]
    for k in (1,2,3):
        b=t0+k*60000
        if end>=0 and end<=b:out.append(end)
        else:
            j=int(np.searchsorted(ts,b));out.append(int(ts[j]) if j<len(ts) and ts[j]-b<=5000 else -1)
    return np.array(out,np.int64)

def player_frame(frame):
    v=np.full((10,18),np.nan,np.float32)
    for p in range(1,11):
        d=frame.get('participantFrames',{}).get(str(p),{});s=d.get('championStats',{});xy=d.get('position',{})
        for j,k in enumerate(PLAYER[:6]):
            if isinstance(d.get(k),(int,float)):v[p-1,j]=d[k]
        for j,(a,b) in enumerate([('health','healthMax'),('power','powerMax')],6):
            if isinstance(s.get(a),(int,float)) and isinstance(s.get(b),(int,float)) and s[b]>0:v[p-1,j]=s[a]/s[b]
        for j,k in enumerate(PLAYER[8:16],8):
            if isinstance(s.get(k),(int,float)):v[p-1,j]=s[k]
        for j,k in enumerate(('x','y'),16):
            if isinstance(xy.get(k),(int,float)):v[p-1,j]=xy[k]
    return v

def availability(t,events,counts,valid,soul,soul_ms,patch):
    """Rule-derived candidate availability, not directly observed monster life.
    Uncertain despawn transition bands and unverified patches remain UNKNOWN.
    """
    value=np.full((6,2),np.nan,np.float32);status=np.zeros((6,2),np.uint8)
    if patch not in policy()['patch_rules']:return value,status
    last={c:max([e['timestamp'] for e in events if e.get('_category')==c],default=-1) for c in EVENT_OBJECTIVE}
    for j,c in enumerate(EVENT_OBJECTIVE):
        if not valid[:,c].all():continue
        if j in (0,1) and not np.isfinite(soul).all():continue
        has=bool(np.nansum(soul)>0)
        if (j==0 and has) or (j==1 and not has):
            value[j]=0;status[j]=2;continue
        if j==0:spawn=300000 if last[c]<0 else last[c]+300000
        elif j==1:spawn=max(soul_ms,last[c])+360000
        elif j==2:
            if counts[:,c].sum()>=3 or t>=900000:value[j]=0;status[j]=2;continue
            if t>=885000:continue
            spawn=480000
        elif j==3:
            if counts[:,c].sum()>=1 or t>=1500000:value[j]=0;status[j]=2;continue
            if t>=1485000:continue
            spawn=900000
        elif j==4:
            if counts[:,c].sum()>=1:value[j]=0;status[j]=2;continue
            spawn=1200000
        else:spawn=1500000 if last[c]<0 else last[c]+360000
        value[j]=[float(t>=spawn),float(t<spawn<=t+30000)];status[j]=2
    return value,status

def process(job):
    row=job;mid=row['match_id']
    detail,dh=read(RAW/'detail'/f'{mid}.json');timeline,th=read(RAW/'timeline'/f'{mid}.json')
    assert dh==row['detail_sha256'] and th==row['timeline_sha256'],mid+' source changed'
    info=detail['info'];frames=timeline['info']['frames']; raw_events=[e for f in frames for e in f.get('events',[])]
    team={p['participantId']:p['teamId'] for p in info['participants']}
    assert set(team)==set(range(1,11)) and Counter(team.values())=={100:5,200:5},mid
    assert detail['metadata']['matchId']==mid and timeline['metadata']['matchId']==mid,mid
    patch='.'.join(info['gameVersion'].split('.')[:2]); assert patch==row['patch'],mid
    assert info['mapId']==11 and info['queueId']==420,mid
    ts=np.array([f['timestamp'] for f in frames],np.int64);assert ts[0]==0 and np.all(np.diff(ts)>0),mid
    assert all(type(e.get('timestamp')) is int and 0<=e['timestamp']<=ts[-1] for e in raw_events),mid
    endings={(e['timestamp'],e.get('winningTeam')) for e in raw_events if e.get('type')=='GAME_END'}
    winners=[x['teamId'] for x in info['teams'] if x.get('win') is True]
    terminal_ok=len(endings)==1 and len(winners)==1
    end,winning=next(iter(endings)) if terminal_ok else (-1,-1)
    terminal_ok=terminal_ok and winning==winners[0] and winning in (100,200) and end==ts[-1]
    if not terminal_ok:end=winning=-1
    assert terminal_ok==row['metrics']['terminal_verified'],mid
    if terminal_ok:assert end>=300000,mid
    pf,transitions=masked_frames(frames,row,team)
    adapter=SimpleNamespace(events=raw_events,meta={'canonical_match_key':hashlib.sha256(mid.encode()).hexdigest(),'patch':patch},team_ids=np.array([team[i] for i in range(1,11)]),minute_ts=ts,xy=pf[:,:,16:18],end_ms=end if terminal_ok else None,winning_team=winning if terminal_ok else None)
    population=build_population(adapter)
    queries,components=make_queries(mid,population,raw_events,row)
    order=sorted(range(len(raw_events)),key=lambda i:(raw_events[i]['timestamp'],i)); events=[dict(raw_events[i]) for i in order]
    et=np.array([e['timestamp'] for e in events],np.int64)
    val,bad=event_arrays(events,team)
    inverse=np.empty(len(order),int);inverse[order]=np.arange(len(order))
    for key,rr in row['event_masks'].items():
        name=key.split(':')[0]
        if name in EVENTS:
            for a,b in rr:
                assert a>=0
                bad[inverse[a:b+1],:,EVENTS.index(name)]=True
    pref=np.concatenate([np.zeros((1,2,20),np.int64),np.cumsum(val,axis=0)])
    bp=np.concatenate([np.zeros((1,2,20),np.int64),np.cumsum(bad,axis=0)])
    for i,e in enumerate(events):
        cats=np.flatnonzero(val[i].sum(axis=0)[2:8]);e['_category']=int(cats[0]+2) if len(cats)==1 else -1
    gold=np.full((len(ts),2),np.nan)
    for z,tid in enumerate((100,200)):
        pids=[p-1 for p in sorted(team) if team[p]==tid];gold[:,z]=pf[:,pids,3].sum(axis=1)
    soul_ms=[-1,-1];soul_events=[];elemental=[0,0];soul_conflicts=0;rift_announcements=0
    for i,e in enumerate(events):
        if e['_category']==2:
            z=int(np.argmax(val[i,:,2]));elemental[z]+=1
            if elemental[z]==4 and soul_ms[z]<0:soul_ms[z]=e['timestamp'];soul_events.append((e['timestamp'],z,2))
        if e.get('type')=='DRAGON_SOUL_GIVEN':
            tid=e.get('teamId')
            if tid==0:rift_announcements+=1
            elif tid in (100,200):
                z=tid//100-1
                if elemental[z]<4:soul_conflicts+=1
                if soul_ms[z]<0:soul_ms[z]=e['timestamp'];soul_events.append((e['timestamp'],z,1))
    if sum(x>=0 for x in soul_ms)>1:soul_conflicts+=1
    records=[];timer_violations=Counter()
    # Audit kills against the rules using strict pre-event history; never use this future
    # audit to set a query's input values or input support mask.
    for i,e in enumerate(events):
        if e['_category'] not in EVENT_OBJECTIVE:continue
        t=e['timestamp'];so=np.array([float(0<=x<t) for x in soul_ms]);sm=max([x for x in soul_ms if 0<=x<t],default=-1)
        av,st=availability(t,events[:i],pref[i],bp[i]==0,so,sm,patch)
        j=EVENT_OBJECTIVE.index(e['_category'])
        if st[j,0] and av[j,0]!=1:timer_violations[OBJECTIVES[j]]+=1
    for q in queries:
        t0=int(q['t0_ms']);fi=int(np.searchsorted(ts,t0));assert fi<len(ts) and ts[fi]==t0
        assert t0<=q['tau_ms']==q['first_kill_ms']-10000 and 0<=q['tau_ms']-t0<=10000
        ep=endpoints(ts,t0,end);points=np.r_[t0,ep];eventidx=np.searchsorted(et,points,side='right')
        pointok=points>=0;tot=pref[eventidx];errs=bp[eventidx]
        counts=tot[1:]-tot[0];support=(errs[1:]-errs[0]==0)&pointok[1:,None,None]
        slices=tot[1:]-tot[:-1];ss=(errs[1:]-errs[:-1]==0)&(pointok[1:]&pointok[:-1])[:,None,None]
        counts=counts.astype(np.float32);slices=slices.astype(np.float32);counts[~support]=np.nan;slices[~ss]=np.nan
        gi=np.searchsorted(ts,points);gv=np.full((4,2),np.nan)
        for k,x in enumerate(points):
            if x>=0 and gi[k]<len(ts) and ts[gi[k]]==x:gv[k]=gold[gi[k]]
        gc=gv[1:]-gv[0];gs=gv[1:]-gv[:-1]
        for k in range(3):
            for ft,z in transitions:
                if t0<ts[ft]<=ep[k]:gc[k,z]=np.nan
                if points[k]>=0 and points[k]<ts[ft]<=ep[k]:gs[k,z]=np.nan
        hist=np.full((5,10,18),np.nan,np.float32);ht=np.full(5,-1,np.int64);take=min(5,fi+1)
        hist[-take:]=pf[fi-take+1:fi+1];ht[-take:]=ts[fi-take+1:fi+1]
        cut=int(eventidx[0]);pre=events[:cut];ps=pref[cut].astype(np.float32);psup=bp[cut]==0;ps[~psup]=np.nan
        # Prefix-only soul inference: no end-of-match contradiction flag may enter X.
        so=np.zeros(2,np.float32);sos=np.ones(2,np.uint8)*2;smt=-1
        local_counts=[0,0];local_conflict=False
        for i,e in enumerate(pre):
            if e['_category']==2:
                z=int(np.argmax(val[i,:,2]));local_counts[z]+=1
                if local_counts[z]==4:so[z]=1;smt=max(smt,e['timestamp'])
            if e.get('type')=='DRAGON_SOUL_GIVEN' and e.get('teamId') in (100,200):
                z=e['teamId']//100-1
                if local_counts[z]<4:local_conflict=True
                so[z]=1;sos[z]=1;smt=max(smt,e['timestamp'])
        if not psup[:,2].all() or local_conflict or so.sum()>1:so[:]=np.nan;sos[:]=0
        if any(key.startswith('soul:') and any(a<0 or any(raw_events[i]['timestamp']<=t0 for i in range(a,b+1)) for a,b in rr) for key,rr in row['event_masks'].items()):so[:]=np.nan;sos[:]=0
        av,avs=availability(t0,pre,pref[cut],psup,so,smt,patch)
        for rm in row.get('rule_masks',[]):
            if rm['timestamp']<=t0:
                name=rm['affected_field'].split(':')[-1]
                if name in EVENTS and EVENTS.index(name) in EVENT_OBJECTIVE:
                    j=EVENT_OBJECTIVE.index(EVENTS.index(name));av[j]=np.nan;avs[j]=0
        ranks=np.zeros((10,4),np.float32);rank_sup=np.ones((10,4),bool)
        for e in pre:
            if e.get('type')=='SKILL_LEVEL_UP':
                p=e.get('participantId');slot=e.get('skillSlot')
                if p in team and slot in (1,2,3,4):ranks[p-1,slot-1]+=1
                else:rank_sup[:]=False
        ranks[~rank_sup]=np.nan
        comp=components[q['component_id']];members=comp['members']
        focal=np.zeros(2,np.float32);focalok=comp['f']==q['first_kill_ms'] and comp['last']==q['last_kill_ms'] and comp['t0']==t0
        selected=[]
        for ordinal in members:
            if ordinal>=len(raw_events) or raw_events[ordinal].get('type')!='CHAMPION_KILL':focalok=False;continue
            e=raw_events[ordinal];selected.append(e['timestamp']);tid=team.get(e.get('killerId'))
            if tid in (100,200):focal[tid//100-1]+=1
            elif e.get('killerId')!=0:focalok=False
        focalok=focalok and bool(selected) and min(selected)==q['first_kill_ms'] and max(selected)==q['last_kill_ms'] and len(set(members))==len(members)
        focalok=focalok and q['selection_supported']
        if not focalok:focal[:]=np.nan
        terminal=np.where(ep>=end,winning//100,0).astype(np.int8);terminal[(ep<0)|(~np.repeat(terminal_ok,3))]=-1
        hazard_sup=terminal_ok&(points[:-1]>=0)&(points[:-1]<end)&(ep>=0);hazard=np.where(hazard_sup,terminal,-1).astype(np.int8)
        if not q['selection_supported']:
            gc[:]=np.nan;gs[:]=np.nan;counts[:]=np.nan;slices[:]=np.nan;support[:]=False;ss[:]=False;terminal[:]=-1;hazard[:]=-1;hazard_sup[:]=False
        net=gc[:,0]-gc[:,1];economic=np.where(np.isfinite(net),np.sign(net),-9).astype(np.int8)
        axes=np.column_stack([net,counts[:,0,2:14]-counts[:,1,2:14]])
        pareto=np.full(3,-1,np.int8)
        for k,x in enumerate(axes):
            if np.isfinite(x).all():pareto[k]=3 if np.all(x==0) else (0 if np.all(x>=0) else (1 if np.all(x<=0) else 2))
        recent=np.zeros((2,2),np.float32)
        for z in (0,1):
            for j,(col,dur) in enumerate(((4,180000),(3,150000))):
                recent[z,j]=float(any(t0-dur<e['timestamp']<=t0 and val[i,z,col]>0 for i,e in enumerate(pre))) if psup[z,col] else np.nan
        records.append(dict(row_id=q['row_id'],match_index=row['row_index'],component_id=q['component_id'],match_id=mid,patch=patch,
          split_role=roles(patch,bucket(mid))[0],macro_role=roles(patch,bucket(mid))[1],selection_supported=q['selection_supported'],focal_member_ordinals_json=json.dumps(members,separators=(',',':')),t0_ms=t0,tau_ms=q['tau_ms'],first_kill_ms=q['first_kill_ms'],last_kill_ms=q['last_kill_ms'],
          history_ms=ht,player_raw=hist,player_support=np.isfinite(hist),champion_id=np.array([next(p['championId'] for p in info['participants'] if p['participantId']==i) for i in range(1,11)],np.int32),
          team_id=np.array([team[i] for i in range(1,11)],np.int16),summoner_ids=np.array([[next(p[key] for p in info['participants'] if p['participantId']==i) for key in ('summoner1Id','summoner2Id')] for i in range(1,11)],np.int32),
          prefix_team_events=ps,prefix_team_events_support=psup,skill_level_up_count=ranks,skill_level_up_count_support=rank_sup,
          soul_owned=so,soul_source_status=sos,objective_availability_rule=av,objective_availability_status=avs,
          recent_acquisition=recent,actual_buff_holder_supported=np.zeros((10,2),bool),
          endpoint_ms=ep,endpoint_supported=pointok[1:],actual_elapsed_ms=np.where(ep>=0,ep-t0,-1),terminal_boundary_shift=(ep==end)&(end>t0+np.arange(1,4)*60000),
          team_gold_endpoints=gv,cum_team_gold=gc,slice_team_gold=gs,gold_supported=np.isfinite(gc),slice_gold_supported=np.isfinite(gs),
          cum_team_events=counts,cum_event_supported=support,slice_team_events=slices,slice_event_supported=ss,
          end_ms=end,winning_team=winning,terminal_supported=np.repeat(terminal_ok and q['selection_supported'],3)&pointok[1:],terminal_class=terminal,
          hazard_class=hazard,hazard_at_risk_supported=hazard_sup,focal_team_kills=focal,focal_kills_supported=np.repeat(focalok,2),
          net_gold=net,economic_sign=economic,pareto_class=pareto,pareto_supported=pareto>=0,
          direct_reward_supported=np.zeros((3,2),bool),remaining_value_supported=np.zeros(3,bool)))
    source={'match_id':mid,'row_index':row['row_index'],'patch':patch,'split_role':roles(patch,bucket(mid))[0],'macro_role':roles(patch,bucket(mid))[1],'detail_sha256':dh,'timeline_sha256':th,'n_queries':len(records),
            'n_components':len(population),'n_primary_candidates':len(queries),'n_selection_supported':sum(q['selection_supported'] for q in queries),'population_reason_counts':dict(Counter(reason for c in population for reason in c['all_reasons'])),'n_frames':len(frames),'n_events':len(events),'terminal_verified':bool(terminal_ok),'end_ms':end,'team0_rift_announcements':rift_announcements,
            'soul_conflicts':soul_conflicts,'timer_rule_contradictions':dict(timer_violations),
            'raw_ward_placed':sum(e.get('type')=='WARD_PLACED' for e in events),
            'unattributed_by_event':dict(zip(EVENTS,bad.any(axis=1).sum(axis=0).astype(int).tolist()))}
    return records,source

INPUT_KEYS={'row_id','match_index','component_id','match_id','patch','split_role','macro_role','t0_ms','history_ms','player_raw','player_support','champion_id','team_id','summoner_ids','prefix_team_events','prefix_team_events_support','skill_level_up_count','skill_level_up_count_support','soul_owned','soul_source_status','objective_availability_rule','objective_availability_status','recent_acquisition','actual_buff_holder_supported'}
META_KEYS={'row_id','match_index','component_id','match_id','patch','split_role','macro_role','t0_ms'}


def masked_frames(frames,row,team):
    pf=np.array([player_frame(f) for f in frames]);transitions=[]
    for key,rr in row['field_masks'].items():
        name=key.split(':')[0];j=PLAYER.index(name)
        for a,b in rr:
            for cell in range(a,b+1):
                fi,p=divmod(cell,10);pf[fi,p,j]=np.nan
                if name=='totalGold' and 'decrease_transition' in key:transitions.append((fi,team[p+1]//100-1))
    return pf,transitions

def make_queries(mid,population,events,row):
    ambiguous={i for key,rr in row['event_masks'].items() if key.startswith('kills:ambiguous') for a,b in rr for i in range(a,b+1)}
    queries=[];components={}
    for c in population:
        if not c['primary_candidate_eligible']:continue
        cid=c['component_id'];members=list(c['member_source_ordinals'])
        supported=not any(c['pre']<=events[i]['timestamp']<=c['L'] or i in members for i in ambiguous)
        queries.append(dict(row_id=f'{mid}:{cid}',component_id=cid,t0_ms=c['F_pre_ms'],tau_ms=c['pre'],first_kill_ms=c['f'],last_kill_ms=c['L'],selection_supported=supported))
        components[cid]={'members':members,'f':c['f'],'last':c['L'],'t0':c['F_pre_ms']}
    return queries,components

def source_snapshot():
    paths=[p for p in (ROOT/'multiwindow_r2').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [p for p in (ROOT/'multiwindow_r3').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [AUDIT,ROOT/'full_fit_raw_audit_20261018/review_02/VALIDATION.json',ROOT/'full_fit_raw_audit_20261018/review_02/ARTIFACT_MANIFEST.json',RESEARCH/'studies/observable_state_20261002/population.py',RESEARCH/'worktrees/engagement-state-value/studies/st_dbscan_20261001/kernel.py']
    return {str(p):sha(p) for p in sorted(set(paths))}

def process_batch(rows):
    records=[];sources=[]
    for row in rows:
        rr,ss=process(row);records.extend(rr);sources.append(ss)
    return records,sources

def build(output,limit=None,workers=3):
    assert 1<=workers<=3
    manifest=json.loads((HERE/'audit_01/VALIDATION.json').read_text(encoding='utf-8'))
    assert manifest['passed'] and sha(AUDIT)==manifest['audit_sha256']
    output.mkdir(parents=True,exist_ok=False);(output/'shards').mkdir()
    (output/'PRODUCER_EXECUTED.py').write_bytes(Path(__file__).read_bytes())
    dump(output/'ENVIRONMENT.json',{'python':sys.version,'numpy':np.__version__,'workers':workers,'cpu_budget':4,'windows_ms':[60000,120000,180000],'input_history_max_frames':5,'input_history_default_frames':3})
    before=source_snapshot();dump(output/'ORIGINAL_HASHES_BEFORE.json',before)
    assert manifest['passed']
    rows=[];excluded=Counter()
    with (output/'DISPOSITIONS.jsonl').open('w',encoding='utf-8') as log:
        for line in AUDIT.open(encoding='utf-8'):
            r=json.loads(line)
            log.write(json.dumps({k:r.get(k) for k in ['match_id','patch','scope','disposition','hard_reasons','population_reasons']},ensure_ascii=False)+'\n')
            if r['disposition'].startswith('RETAIN'):
                r['row_index']=len(rows);rows.append(r)
            else:excluded[r['disposition']]+=1
    total=len(rows)
    if limit is not None:rows=rows[:limit]
    tasks=[rows[i:i+128] for i in range(0,len(rows),128)]
    counts=Counter();shards=[];start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool,(output/'SOURCE_LEDGER.jsonl').open('w',encoding='utf-8') as log:
        for batch,(records,sources) in enumerate(pool.map(process_batch,tasks,chunksize=1)):
            for src in sources:
                log.write(json.dumps(src,ensure_ascii=False,separators=(',',':'))+'\n')
                counts['matches_processed']+=1;counts['matches_with_queries']+=int(src['n_queries']>0);counts['queries']+=src['n_queries'];counts['selection_supported_queries']+=src['n_selection_supported'];counts['components']+=src['n_components']
            if records:
                arrays={k:np.asarray([r[k] for r in records]) for k in records[0]}
                dst=output/'shards'/f'{batch:04d}.npz';np.savez_compressed(dst,**arrays);shards.append(dst.name)
            if batch%10==0 or batch==len(tasks)-1:
                log.flush();progress={**counts,'total_retained_matches':len(rows),'elapsed_s':round(time.time()-start,1)};dump(output/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    changed=[p for p,h in before.items() if sha(p)!=h];dump(output/'ORIGINAL_PRESERVATION.json',{'pass':not changed,'changed':changed,'n_files':len(before)});assert not changed
    dump(output/'BUILD_SUMMARY.json',{'profile':'MW60_FRAME_r3_holdout','full_population':limit is None,'audit_retained_matches':total,'n_matches_with_queries':counts['matches_with_queries'],'n_queries':counts['queries'],'selection_supported_queries':counts['selection_supported_queries'],'counts':dict(counts),'shards':shards,'event_fields':EVENTS,'player_fields':PLAYER,'objective_order':OBJECTIVES,'input_keys':sorted(INPUT_KEYS),'meta_keys':sorted(META_KEYS|{'selection_supported','focal_member_ordinals_json'}),'source_audit_sha256':sha(AUDIT),'producer_sha256':sha(Path(__file__)),'training_performed':False,'U_enabled':False,'elapsed_s':round(time.time()-start,1),'source_status_codes':{'0':'UNKNOWN','1':'EVENT_EXACT','2':'RULE_DERIVED','3':'STRUCTURAL_NA'},'terminal_class_codes':{'-1':'UNKNOWN','0':'ONGOING','1':'BLUE_WIN','2':'RED_WIN'},'pareto_codes':{'-1':'UNKNOWN','0':'BLUE_DOMINATES','1':'RED_DOMINATES','2':'TRADEOFF','3':'NO_DIFFERENCE'}})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--limit',type=int);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args();build(args.output,args.limit,args.workers)
