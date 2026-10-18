"""Full held-out raw audit; read-only raw sources, identical FIT policy."""
from pathlib import Path
import os,json,time,argparse,hashlib
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from split import roles,role_names,bucket
import audit_core as a
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent
def run(out,limit=None):
    out.mkdir(exist_ok=False);start=time.time()
    names={k:sorted(e.name[:-5] for e in os.scandir(a.RAW/k) if e.is_file() and e.name.endswith('.json')) for k in ('detail','timeline')}
    groups=defaultdict(lambda:{'detail':[],'timeline':[]})
    for k,nn in names.items():
        for n in nn:groups[a.canonical(n)][k].append(n)
    aliases={m:g for m,g in groups.items() if any(len(x)>1 for x in g.values())}
    assert not aliases,'Canonical aliases require conflict-safe individual scope audit before proceeding'
    prior=[json.loads(s) for s in (R/'full_fit_raw_audit_20261018/full_02/MATCH_AUDIT.jsonl').open(encoding='utf-8')]
    knownfit={r['match_id'] for r in prior if roles(r['patch'],bucket(r['match_id']))[1]=='FIT'}
    # Only skip established FIT identities, never treat unresolved headers as held-out.
    jobs=[(m,g['detail'],g['timeline']) for m,g in sorted(groups.items()) if m not in knownfit]
    total=len(jobs)
    if limit is not None:jobs=jobs[:limit]
    inventory={'raw_root':str(a.RAW),'enumeration':'filesystem inventory; PostgreSQL crosscheck recorded separately','detail_files':len(names['detail']),'timeline_files':len(names['timeline']),'canonical_aliases':len(aliases),'known_fit_excluded_from_rescan':len(knownfit),'candidates':len(jobs),'full_candidates':total,'file_list_sha256':{k:hashlib.sha256('\n'.join(n).encode()).hexdigest() for k,n in names.items()},'frozen_roles':sorted(role_names()),'limit':limit}
    a.dump(out/'INVENTORY.json',inventory)
    (out/'AUDITOR_EXECUTED.py').write_bytes((H/'audit_core.py').read_bytes())
    counts=Counter();scope=Counter();rolecounts=Counter();done=0
    hp=[np.zeros(len(e)-1,np.int64) for e in a.FEATURE_EDGES];hq=[x.copy() for x in hp]
    batches=[jobs[i:i+64] for i in range(0,len(jobs),64)]
    with (out/'MATCH_AUDIT.jsonl').open('w',encoding='utf-8') as log,ProcessPoolExecutor(max_workers=3) as pool:
        for n,(rows,sc,pre,post) in enumerate(pool.map(a.batch,batches,chunksize=1)):
            for row in rows:
                row['split_role'],row['macro_role']=roles(row.get('patch'),row['bucket'])
                assert row['split_role'] in role_names() or row.get('scope')=='UNRESOLVED',row
                log.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
                counts[row['disposition']]+=1;rolecounts[row['split_role']+':'+row['disposition']]+=1
            scope.update(sc);done+=len(batches[n])
            for j in range(18):hp[j]+=pre[j];hq[j]+=post[j]
            if n%20==0 or n==len(batches)-1:
                log.flush();p={'processed':done,'total':len(jobs),'dispositions':dict(counts),'elapsed_s':round(time.time()-start,1)};a.dump(out/'PROGRESS.json',p);print(json.dumps(p),flush=True)
    np.savez_compressed(out/'FEATURE_HISTOGRAMS.npz',**{f'{p}_{j}':v for p,vs in [('before',hp),('after',hq),('edges',a.FEATURE_EDGES)] for j,v in enumerate(vs)})
    auditrows=[json.loads(s) for s in (out/'MATCH_AUDIT.jsonl').open(encoding='utf-8')]
    ids=[r['match_id'] for r in auditrows];assert len(ids)==len(set(ids))
    assert not set(ids)&knownfit
    assert counts['AUDITOR_ERROR']==0,counts
    for row in auditrows:
        if row['disposition'].startswith('RETAIN'):
            assert row['split_role'] in role_names() and row.get('timeline_sha256') and row.get('detail_sha256')
        for k,ranges in row['event_masks'].items():assert all(x>=0 and y>=x for x,y in ranges),(row['match_id'],k)
    receipt={'passed':True,'full_population':limit is None,'audit_sha256':hashlib.sha256((out/'MATCH_AUDIT.jsonl').read_bytes()).hexdigest(),'counts':dict(counts),'roles':dict(rolecounts),'outside_scope':dict(scope),'raw_records_audited':len(auditrows),'known_fit_overlap':len(set(ids)&knownfit),'ambiguous_scope_separate':sum(r['split_role']=='OTHER' for r in auditrows),'elapsed_s':round(time.time()-start,1),'training':False,'rules_identical_to_fit_except_prefix_safe_soul_conflict_fix':True}
    a.dump(out/'VALIDATION.json',receipt);a.dump(out/'RUN_COMPLETE.json',receipt)
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);x=p.parse_args();run(x.output,x.limit)
