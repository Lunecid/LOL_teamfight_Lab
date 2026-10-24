"""Regenerate corrected states, NOT labels from incompatible frozen V1 models."""
import run_validation_suite as old
from gameplay.state_value_v2 import StateBuilder, STATE_VERSION, state_matrix
import numpy as np,pandas as pd,json,time,hashlib
from concurrent.futures import ThreadPoolExecutor
OUT=old.ROOT/'outputs/state_value_v2_fix';OUT.mkdir(parents=True,exist_ok=True)

def one(task):
    mid,rows=task;p=old.pack(mid);b=StateBuilder(p,old.NODE_FEATURE_NAMES);out=[]
    original_order=tuple(sorted((int(k) for k in p['meta']['role_slots']),key=lambda pid:int(p['meta']['role_slots'].get(str(pid),p['meta']['role_slots'].get(pid)))))
    for r in rows:
        pre=b.at(int(r['s'])-1);end=b.at(int(r['endpoint']));out.append((r,pre,end,original_order!=b.participant_order))
    return out

def main():
    source=old.OUT/'rows.csv';rows=pd.read_csv(source,dtype={'patch':str});records=[];start=time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,a in enumerate(pool.map(one,[(m,g.to_dict('records')) for m,g in rows.groupby('match')]),1):
            records.extend(a)
            if i%2000==0:print('v2 states',i,round(time.time()-start,1),flush=True)
    names=list(records[0][1].values);X=state_matrix([pre for r,pre,end,reorder in records],names);E=state_matrix([end for r,pre,end,reorder in records],names)
    meta=pd.DataFrame([dict(match=r['match'],s=r['s'],endpoint=r['endpoint'],patch=r['patch'],pre_snapshot=pre.snapshot_ms,post_snapshot=end.snapshot_ms,pre_unassigned_soul_events=pre.unassigned_soul_events,post_unassigned_soul_events=end.unassigned_soul_events,reordered=int(reorder)) for r,pre,end,reorder in records]);assert not meta.duplicated(['match','s']).any();assert (meta.pre_snapshot<meta.s).all() and (meta.post_snapshot<=meta.endpoint).all()
    meta.to_csv(OUT/'rows.csv',index=False);np.savez_compressed(OUT/'states.npz',pre=X,post=E,state_version=STATE_VERSION,names=np.asarray(names))
    # Demonstrate mutation invariance on real cache data, independently of fixtures.
    checks=[]
    for mid,g in list(rows.groupby('match'))[:100]:
        pack=old.pack(mid);q=int(g.iloc[0].s)-1;a=StateBuilder(pack,old.NODE_FEATURE_NAMES).at(q)
        pack['meta']=dict(pack['meta']);pack['meta'].pop('role_slots',None);b=StateBuilder(pack,old.NODE_FEATURE_NAMES).at(q)
        checks.append(a.values==b.values)
    assert all(checks)
    legacy_path=old.WT/'gameplay/state_value.py';legacy_hash=hashlib.sha256(legacy_path.read_bytes()).hexdigest();previous=json.loads((old.OUT/'provenance.json').read_text());key=str(legacy_path.relative_to(old.ROOT));assert previous[key]==legacy_hash
    result=dict(state_version=STATE_VERSION,rows=len(meta),matches=int(meta.match.nunique()),features=len(names),pre_nonzero_soul_columns={n:int(np.count_nonzero(X[:,names.index(n)])) for n in names if '_soul_' in n and not n.endswith('_x_time')},reordered_matches=int(meta.loc[meta.reordered==1,'match'].nunique()),reordered_rows=int(meta.reordered.sum()),unassigned_events_exposed_as_diagnostic_only=True,role_deletion_real_cache_checks=len(checks),role_deletion_real_cache_failures=0,legacy_module_hash_unchanged=True,source_rows_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),module_sha256=hashlib.sha256((old.WT/'gameplay/state_value_v2.py').read_bytes()).hexdigest(),new_V_trained=False,new_labels_generated=False,old_probability_fields_copied=False,note='New feature names reject legacy role-based model schema. p_pre and Y must be regenerated after fitting independent V2. Population/endpoints unchanged.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
