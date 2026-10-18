"""DB-indexed shared observed-prefix features. Never reads target fields into X."""
from collections import defaultdict,Counter
import json,copy
import numpy as np
import joblib
from common import *

TARGET_KEYS=('slice_team_gold','slice_gold_supported','slice_team_events','slice_event_supported','cum_team_gold','gold_supported','cum_team_events','cum_event_supported','net_gold','hazard_class','hazard_at_risk_supported','terminal_class','terminal_supported','selection_supported')
BASE=['level','xp','currentGold','totalGold','minionsKilled','jungleMinionsKilled','health_fraction','power_fraction','health','healthMax','attackDamage','abilityPower','armor','magicResist','attackSpeed','movementSpeed','x','y']
RIFTS=['Infernal','Ocean','Mountain','Cloud','Hextech','Chemtech']

def select():
    assert db.sql(f"SELECT to_regclass('{SCHEMA}.source_lookup') IS NOT NULL")=='t', 'Required indexed catalog view is missing'
    c=read(HERE/'CARD.json');assert set(c['roles'])=={'FIT_15.18','FIT_15.19','FIT_15.20','STOP_15.20'}
    parts=[f"(SELECT match_id FROM {SCHEMA}.source_lookup WHERE split_role='{role}' AND disposition='RETAIN_MASKED' AND n_queries>0 ORDER BY md5('common-r1-smoke:'||match_id) LIMIT {int(n)})" for role,n in c['roles'].items()]
    query=f"""WITH chosen AS ({' UNION ALL '.join(parts)})
    SELECT json_agg(t) FROM (SELECT q.row_id,q.match_id,q.family,q.shard_row,q.t0_ms,
    m.patch,m.split_role,m.macro_role,s.absolute_path AS shard_path,s.sha256 AS shard_sha256,
    f.detail_path,f.timeline_path,f.detail_audit_sha256,f.timeline_audit_sha256,
    encode(q.player_support,'hex') AS player_mask,
    (SELECT json_agg(json_build_object('h',h.horizon_s,'event_mask',encode(h.event_support,'hex'),'gold_mask',encode(h.gold_support,'hex'),'economic_supported',h.economic_supported) ORDER BY h.horizon_s) FROM {SCHEMA}.horizons h WHERE h.row_id=q.row_id) AS horizon_masks
    FROM chosen c JOIN {SCHEMA}.queries q USING(match_id) JOIN {SCHEMA}.matches m USING(match_id)
    JOIN {SCHEMA}.shards s ON s.family=q.family AND s.shard=q.shard
    JOIN {SCHEMA}.source_lookup f USING(match_id) ORDER BY q.match_id,q.row_id) t"""
    # Explicit joins avoid duplicate match_id names introduced by the shard join.
    query=query.replace('JOIN '+SCHEMA+'.source_lookup f USING(match_id)','JOIN '+SCHEMA+'.source_lookup f ON f.match_id=q.match_id')
    rows=json.loads(db.sql(query));assert rows
    for role,n in c['roles'].items():assert len({r['match_id'] for r in rows if r['split_role']==role})==n
    dump(HERE/'DB_SELECTION.json',{'schema':SCHEMA,'card_sha256':sha(HERE/'CARD.json'),'rows':rows,'sql':query,'read_only':True})
    return rows

def load_rows(selection):
    grouped=defaultdict(list)
    for r in selection:grouped[r['shard_path']].append(r)
    output={};hashes={}
    inputs=['player_raw','player_support','history_ms','champion_id','summoner_ids','team_id','t0_ms']
    for path,rr in grouped.items():
        digest=sha(path);assert all(digest==r['shard_sha256'] for r in rr);hashes[path]=digest
        with np.load(path,allow_pickle=False) as z:
            a={k:z[k] for k in inputs+list(TARGET_KEYS)+['row_id','match_id']}
            for r in rr:
                j=r['shard_row'];assert str(a['row_id'][j])==r['row_id'] and str(a['match_id'][j])==r['match_id']
                assert np.packbits(a['player_support'][j].reshape(-1),bitorder='little').tobytes().hex()==r['player_mask']
                for h,m in enumerate(r['horizon_masks']):
                    assert m['h']==(h+1)*60
                    for key,col in [('cum_event_supported','event_mask'),('gold_supported','gold_mask')]:assert np.packbits(a[key][j,h].reshape(-1),bitorder='little').tobytes().hex()==m[col]
                    assert bool(np.isfinite(a['net_gold'][j,h]))==m['economic_supported']
                output[r['row_id']]={k:a[k][j] for k in inputs+list(TARGET_KEYS)}
    return output,hashes

def canonical_order(frames,participants):
    """Freeze tactical proxy at <=90 seconds, safely before all selected t0s."""
    anchor=max((f for f in frames if f['timestamp']<=90000),key=lambda f:f['timestamp'])
    keys={}
    for p in participants:
        pid=p['participantId'];f=anchor.get('participantFrames',{}).get(str(pid),{});xy=f.get('position',{})
        spells=sorted([p['summoner1Id'],p['summoner2Id']]);x=xy.get('x');y=xy.get('y')
        if p['teamId']==200 and x is not None:x,y=-x,-y
        if 11 in spells:role=1
        elif x is None or y is None:role=5
        elif y-x>2500:role=0
        elif x-y>2500:role=3
        else:role=2
        keys[pid]=(role,-f.get('minionsKilled',0),p['championId'],*spells)
    order=[]
    for team in (100,200):
        players=[p for p in participants if p['teamId']==team];assert len(players)==5
        assert len({p['championId'] for p in players})==5,'Ambiguous canonical roster'
        order+=sorted([p['participantId'] for p in players],key=lambda pid:keys[pid])
    return np.array(order)-1,anchor['timestamp']

def snapshots(timeline,participants,times):
    """All updates are prefix-local. Unknown onset never propagates backwards."""
    team={p['participantId']:p['teamId']//100-1 for p in participants}
    events=[(e.get('timestamp',-1),i,e) for i,e in enumerate(e for f in timeline['info']['frames'] for e in f.get('events',[]))]
    events.sort(key=lambda e:(e[0],e[1]));events=[e for e in events if e[0]>=0]
    kda=np.zeros((10,3));kda_ok=np.ones((10,3),bool);ranks=np.zeros((10,4));rank_ok=np.ones((10,4),bool)
    death=np.full(10,-1.);items=[Counter() for _ in range(10)];item_ok=np.ones(10,bool)
    counts=np.zeros((2,20));count_ok=np.ones((2,20),bool);last=np.full((2,20),-1.)
    struct=np.zeros((2,14));struct_ok=np.ones((2,14),bool);struct_last=np.full((2,14),-1.)
    announcement=0;announce_last=-1.;rift=np.zeros(6);cursor=0;out={}
    def credit(col,t,pid=None,tid=None):
        z=team.get(pid) if tid is None else ({100:0,200:1}.get(tid))
        if z is None:count_ok[:,col]=False
        else:counts[z,col]+=1;last[z,col]=t
    for cutoff in sorted(set(int(t) for t in times if t>=0)):
        while cursor<len(events) and events[cursor][0]<=cutoff:
            t,ordinal,e=events[cursor];cursor+=1;ty=e.get('type');pid=e.get('participantId')
            if ty=='CHAMPION_KILL':
                killer=e.get('killerId');victim=e.get('victimId');ass=e.get('assistingParticipantIds')
                if killer in team:kda[killer-1,0]+=1;credit(0,t,killer)
                elif killer!=0:kda_ok[:,0]=False;count_ok[:,0]=False
                if victim in team:kda[victim-1,1]+=1;death[victim-1]=t
                else:kda_ok[:,1]=False;death[:]=np.nan
                if isinstance(ass,list) and all(p in team for p in ass):
                    for p in set(ass):kda[p-1,2]+=1;credit(1,t,p)
                else:kda_ok[:,2]=False;count_ok[:,1]=False
            elif ty=='SKILL_LEVEL_UP':
                slot=e.get('skillSlot')
                if pid in team and slot in (1,2,3,4):ranks[pid-1,slot-1]+=1
                elif pid in team:rank_ok[pid-1]=False
                else:rank_ok[:]=False
            elif ty in ('ITEM_PURCHASED','ITEM_SOLD','ITEM_DESTROYED','ITEM_UNDO'):
                if ty!='ITEM_DESTROYED':credit({'ITEM_PURCHASED':14,'ITEM_SOLD':15,'ITEM_UNDO':16}[ty],t,pid)
                if pid not in team:item_ok[:]=False
                elif ty=='ITEM_UNDO':item_ok[pid-1]=False
                else:
                    iid=e.get('itemId')
                    if not isinstance(iid,int) or iid<=0:item_ok[pid-1]=False
                    else:
                        items[pid-1][iid]+=1 if ty=='ITEM_PURCHASED' else -1
                        if items[pid-1][iid]<0:item_ok[pid-1]=False
            elif ty=='ELITE_MONSTER_KILL':
                mt=e.get('monsterType');st=e.get('monsterSubType');col={'BARON_NASHOR':4,'ATAKHAN':5,'RIFTHERALD':6,'HORDE':7}.get(mt)
                if mt=='DRAGON':col=3 if st=='ELDER_DRAGON' else (2 if st and st.endswith('_DRAGON') else None)
                if col is None:count_ok[:,2:8]=False
                else:credit(col,t,e.get('killerId'),e.get('killerTeamId'))
            elif ty in ('BUILDING_KILL','TURRET_PLATE_DESTROYED'):
                owner=e.get('teamId');lane={'TOP_LANE':0,'MID_LANE':1,'BOT_LANE':2}.get(e.get('laneType'));tower=e.get('towerType')
                col=13 if ty=='TURRET_PLATE_DESTROYED' else (12 if e.get('buildingType')=='INHIBITOR_BUILDING' else {'OUTER_TURRET':8,'INNER_TURRET':9,'BASE_TURRET':10,'NEXUS_TURRET':11}.get(tower))
                if col is not None:credit(col,t,tid=300-owner if owner in (100,200) else None)
                elif e.get('buildingType')!='NEXUS_BUILDING':count_ok[:,8:13]=False
                if ty=='BUILDING_KILL' and e.get('buildingType')!='NEXUS_BUILDING':
                    z={100:0,200:1}.get(owner);s=None
                    if col==12 and lane is not None:s=11+lane
                    elif tower in ('OUTER_TURRET','INNER_TURRET','BASE_TURRET') and lane is not None:s=lane*3+{'OUTER_TURRET':0,'INNER_TURRET':1,'BASE_TURRET':2}[tower]
                    elif tower=='NEXUS_TURRET':
                        xy=e.get('position',{})
                        if 'x' in xy and 'y' in xy:s=9+int((xy['x']-xy['y'])*(1 if owner==100 else -1)>0)
                    if z is None or s is None:struct_ok[:]=False
                    else:struct[z,s]+=1;struct_last[z,s]=t
            elif ty=='DRAGON_SOUL_GIVEN' and e.get('teamId')==0:
                announcement+=1;announce_last=t
                name=e.get('name',e.get('dragonSoulType',e.get('monsterSubType')))
                if name in RIFTS:rift[RIFTS.index(name)]+=1
                else:rift[:]=np.nan
            elif ty=='WARD_PLACED':credit(17,t,e.get('creatorId'))
            elif ty=='WARD_KILL':credit(18,t,e.get('killerId'))
            elif ty=='LEVEL_UP':credit(19,t,pid)
        age=np.where(death>=0,(cutoff-death)/1000,np.where(np.isfinite(death),0,np.nan))
        personal=np.c_[np.where(kda_ok,kda,np.nan),np.where(rank_ok,ranks,np.nan),age,(death<0).astype(float)]
        event_age=np.where(last>=0,(cutoff-last)/1000,-1.)
        structure_age=np.where(struct_last>=0,(cutoff-struct_last)/1000,-1.)
        soul=np.where(count_ok[:,2],(counts[:,2]>=4).astype(float),np.nan)
        if np.nansum(soul)>1:soul[:]=np.nan
        team_extra=np.c_[np.where(count_ok,counts,np.nan),np.where(count_ok,event_age,np.nan),np.where(struct_ok,struct,np.nan),np.where(struct_ok,structure_age,np.nan),soul]
        out[cutoff]={'personal':personal.copy(),'items':copy.deepcopy(items),'item_ok':item_ok.copy(),'team_extra':team_extra.copy(),'rift':np.r_[cutoff/1000,announcement,(cutoff-announce_last)/1000 if announce_last>=0 else -1,rift]}
    return out

def build_observed(selection,loaded):
    grouped=defaultdict(list)
    for r in selection:grouped[r['match_id']].append(r)
    records={};raw_hashes={};anchors={}
    for i,(mid,rows) in enumerate(grouped.items()):
        r=rows[0];dp=r['detail_path'];tp=r['timeline_path']
        assert sha(dp)==r['detail_audit_sha256'];assert sha(tp)==r['timeline_audit_sha256'];raw_hashes[dp]=sha(dp);raw_hashes[tp]=sha(tp)
        detail=read(dp);timeline=read(tp);participants=detail['info']['participants'];order,anchor=canonical_order(timeline['info']['frames'],participants)
        times=np.concatenate([loaded[r['row_id']]['history_ms'] for r in rows]);snaps=snapshots(timeline,participants,times);anchors[mid]={'anchor_ms':anchor,'order_pid':(order+1).tolist()}
        for r in rows:
            a=loaded[r['row_id']];assert a['t0_ms']>=anchor;ht=a['history_ms'];base=a['player_raw'].copy();base[~a['player_support']]=np.nan
            team_id=a['team_id'];player=[];team=[];rift=[];relation=[];itemrows=[]
            for j,t in enumerate(ht):
                if t<0:
                    player.append(np.full((10,27),np.nan));team.append(np.full((2,77),np.nan));rift.append(np.full(10,np.nan));relation.append(np.full((10,10,6),np.nan));itemrows.append(None);continue
                ss=snaps[int(t)];pp=np.c_[base[j],ss['personal']];player.append(pp[order]);itemrows.append((ss['items'],ss['item_ok'],order))
                obs=[]
                for z in (100,200):
                    ids=np.where(team_id==z)[0];v=np.c_[base[j,ids,3],base[j,ids,1],base[j,ids,0],ss['personal'][ids,:3],base[j,ids,4:6]]
                    obs.append(np.where(np.isfinite(v).all(0),np.nansum(v,axis=0),np.nan))
                team.append(np.c_[np.array(obs),ss['team_extra']]);rift.append(np.r_[ss['rift'],(a['t0_ms']-t)/1000])
                xy=base[j,:,16:18][order];hp=base[j,:,8][order];active=np.isfinite(xy).all(1)&np.isfinite(hp)&(hp>0)
                d=xy[None,:,:]-xy[:,None,:];valid=active[:,None]&active[None,:]
                rel=np.stack([d[:,:,0],d[:,:,1],np.linalg.norm(d,axis=-1),team_id[order,None]==team_id[order][None,:],np.eye(10,dtype=bool),valid],axis=-1).astype(float)
                rel[:,:,:3]=np.where(valid[:,:,None],rel[:,:,:3],np.nan);relation.append(rel)
            records[r['row_id']]={'player':np.array(player),'team':np.array(team),'rift':np.array(rift),'relation':np.array(relation),'items':itemrows,'champion':a['champion_id'][order],'spells':np.sort(a['summoner_ids'][order],axis=-1),'present':ht>=0,'history_ms':ht,'t0_ms':int(a['t0_ms'])}
        if i%64==0:progress('prefix',matches=i+1)
    dump(HERE/'RAW_SOURCE_HASHES.json',raw_hashes);dump(HERE/'CANONICAL_ANCHORS.json',anchors)
    return records

class Adapter:
    def fit(self,records,roles):
        assert all(r.startswith('FIT_') for r in roles),'Fit-only transformer'
        self.champs=sorted({int(x) for r in records for x in r['champion']});self.spells=sorted({int(x) for r in records for x in r['spells'].ravel()})
        self.items=sorted({int(k) for r in records for entry in r['items'] if entry for pid in range(10) for k,v in entry[0][pid].items() if v>0 and entry[1][pid]})
        self.stats={};raw=self.numeric(records)
        for k,a in raw.items():
            if k=='present':continue
            b=a.reshape(-1,a.shape[-1]);center=[];scale=[]
            for j in range(b.shape[1]):
                v=b[np.isfinite(b[:,j]),j];med=float(np.median(v)) if len(v) else 0;q=np.percentile(v,[25,75]) if len(v) else [0,1]
                center.append(med);scale.append(max(float(q[1]-q[0]),1.))
            self.stats[k]=(np.array(center),np.array(scale))
        self.fit_roles=sorted(set(roles));return self
    def numeric(self,records):
        out={k:[] for k in ('player','team','rift','relation','present')};iv={x:i+1 for i,x in enumerate(self.items)}
        for r in records:
            inventories=[]
            for entry in r['items']:
                v=np.full((10,len(iv)+1),np.nan)
                if entry:
                    balances,ok,order=entry
                    for j,pid in enumerate(order):
                        if ok[pid]:
                            v[j]=0
                            for item,count in balances[pid].items():v[j,iv.get(item,0)]+=count
                inventories.append(v)
            out['player'].append(np.concatenate([r['player'],np.array(inventories)],axis=-1))
            for k in ('team','rift','relation','present'):out[k].append(r[k])
        return {k:np.array(v) for k,v in out.items()}
    def transform(self,records,L):
        assert L in range(1,6);raw=self.numeric(records);out={}
        for k,a in raw.items():
            if k=='present':out[k]=a[:,-L:].astype(np.float32);continue
            valid=np.isfinite(a);center,scale=self.stats[k]
            normalized=np.zeros_like(a,dtype=np.float32);np.divide(a-center,scale,out=normalized,where=valid)
            out[k]=np.concatenate([normalized,valid.astype(np.float32)],axis=-1)[:,-L:]
        cats=[]
        for r in records:
            values=[]
            for pid in range(10):
                row=[]
                for value,vocab in [(r['champion'][pid],self.champs),*[(s,self.spells) for s in r['spells'][pid]]]:
                    bits=np.zeros(len(vocab)+1);bits[vocab.index(int(value))+1 if value in vocab else 0]=1;row.extend(bits)
                values.append(row)
            cats.append(values)
        cat=np.repeat(np.array(cats,dtype=np.float32)[:,None],L,axis=1)*out['present'][:,:,None,None]
        out['player']=np.concatenate([out['player'],cat],axis=-1)
        # Padded frames have no values, support flags, or categorical tokens.
        for k in ('player','team','rift','relation'):out[k]*=out['present'].reshape(out['present'].shape+(1,)*(out[k].ndim-2))
        assert all(np.isfinite(v).all() for v in out.values());return out

def main():
    if (HERE/'DATA.joblib').exists():raise RuntimeError('Refuse to replace prepared data')
    selection=select();loaded,shards=load_rows(selection);observed=build_observed(selection,loaded)
    rows=[observed[r['row_id']] for r in selection];roles=np.array([r['split_role'] for r in selection]);fit=np.char.startswith(roles,'FIT_');adapter=Adapter().fit([rows[i] for i in np.where(fit)[0]],roles[fit])
    x=adapter.transform(rows,read(HERE/'CARD.json')['L']);y={k:np.array([loaded[r['row_id']][k] for r in selection]) for k in TARGET_KEYS}
    meta={k:np.array([r[k] for r in selection]) for k in ('row_id','match_id','patch','split_role','macro_role')}
    joblib.dump({'x':x,'y':y,'meta':meta,'observed':rows,'adapter':adapter},HERE/'DATA.joblib',compress=3)
    dump(HERE/'DATA_MANIFEST.json',{'rows':len(rows),'matches':len(set(meta['match_id'])),'roles':dict(Counter(roles.tolist())),'input_shapes':{k:list(v.shape) for k,v in x.items()},'data_sha256':sha(HERE/'DATA.joblib'),'shard_hashes':shards,'source_hashes_sha256':sha(HERE/'RAW_SOURCE_HASHES.json'),'db_selection_sha256':sha(HERE/'DB_SELECTION.json'),'card_sha256':sha(HERE/'CARD.json'),'fit_vocabs':{'champions':len(adapter.champs),'spells':len(adapter.spells),'items':len(adapter.items)},'fit_roles':adapter.fit_roles,'model_input_keys':list(x),'provenance':{'observed_at_ms':'Each history_ms actual frame; all prefix events <= that timestamp.','age_ms':'t0-history_ms; last-event ages are explicit features.','source_status':'Finite raw frame values=observed; prefix reconstruction=event-derived; invalid fields=unknown.','structural_na':'No disabled K3/terrain/buff/respawn feature emitted; fitted patches all retain Atakhan.','source_refs':'Selected raw hashes + shard hashes + row IDs in separate metadata; not model features.'}})
    progress('data_ready',rows=len(rows),matches=len(set(meta['match_id'])))
if __name__=='__main__':
    import data
    data.main()
