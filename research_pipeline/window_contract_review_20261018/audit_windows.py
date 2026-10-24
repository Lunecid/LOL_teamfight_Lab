"""Inspect saved window artifacts only; no fits or outcome regeneration."""
import json, hashlib, subprocess, sys, os
from pathlib import Path
from collections import Counter
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
PREP=ROOT/'multipatch_v5_prep'
WIN=PREP/'window_evidence_v5'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''): h.update(x)
    return h.hexdigest()
def stats(a):
    a=np.asarray(a)
    return {'min':float(a.min()),'median':float(np.median(a)), 'p99':float(np.quantile(a,.99)), 'max':float(a.max())}

res={'scope':'Saved arrays, source code and manifests; not raw-event recertification',
     'training_executed_this_audit':False,'originals_modified':False}
manifest=read(WIN/'queries_by_objective/MANIFEST_QUERIES_BY_OBJECTIVE.json')
summary=read(WIN/'multiwindow_outcome_ledger/MULTIWINDOW_OUTCOME_LEDGER_SUMMARY_V5.json')
entries=[*manifest['Saved_Query_Archives'],summary['ledger_archive']]
res['archive_hashes']=[{'path':e['path'],'matches':sha(PREP/e['path'])==e['sha256']} for e in entries]
q=np.load(WIN/'queries_by_objective/Q_ALL_PRIMARY_TIMELINES.npz',allow_pickle=False)
l=np.load(PREP/summary['ledger_archive']['path'],allow_pickle=False)
res['array_schema']={k:{'shape':list(l[k].shape),'dtype':str(l[k].dtype)} for k in l.files}
res['rows']=len(q['t0_ms'])
res['same_query_keys']=all(np.array_equal(q[k],l[k]) for k in ['match_index','component_id','patch','split_role'])
res['unique_query_keys']=len(set(zip(q['match_index'].tolist(),q['component_id'].tolist())))
res['rows_by_role']=dict(Counter(q['macro_role'].tolist()))
cur=[json.loads(x) for x in Path(os.environ['LOL_SPLIT_ASSIGNMENTS_PATH']).read_text(encoding='utf-8').splitlines() if x.strip()]
flags=np.array([r['is_clean_combat_trainable'] for r in cur])
bad=~flags[q['match_index']]
res['queries_from_excluded_matches']={'rows':int(bad.sum()),'matches':len(np.unique(q['match_index'][bad])),
                                     'roles':dict(Counter(q['macro_role'][bad].tolist()))}
ts=l['frame_ts_4']; t0=q['t0_ms']; end=q['end_ms']
res['t0_equals_first_frame']=bool(np.array_equal(ts[:,0],t0))
res['tau_minus_t0_ms']=stats(q['tau_ms']-t0)
res['tau_minus_t0_nonzero']=int(np.sum(q['tau_ms']!=t0))
res['history_future_cells']=int(np.sum(q['hist_frame_ms']>t0[:,None]))
res['history_missing_cells_by_lag']=np.sum(q['hist_frame_ms']<0,axis=0).tolist()
res['endpoint_by_horizon']={}
for i,h in enumerate([60,120,180],1):
    target=t0+h*1000; nonterminal=end>target
    err=ts[:,i]-target
    res['endpoint_by_horizon'][h]={'nonterminal_rows':int(nonterminal.sum()),
        'nonterminal_error_ms':stats(err[nonterminal]),
        'nonterminal_error_over_5s':int(np.sum(nonterminal & (np.abs(err)>5000))),
        'terminal_rows':int(np.sum(~nonterminal)),
        'last_frame_before_game_end_on_terminal_rows':int(np.sum((~nonterminal)&(ts[:,i]<end))),
        'events_boundary_vs_nominal_ms_nonzero':int(np.sum(err!=0))}
res['repeated_future_frame_cells']=int(np.sum(np.diff(ts,axis=1)==0))
se=l['slice_team_events']; ce=l['cum_team_events']
res['additivity']={'events_exact':bool(np.array_equal(np.cumsum(se,axis=1),ce)),
    'gold_components_exact':bool(np.array_equal(np.cumsum(l['slice_team_gold_comps'],axis=1),l['cum_team_gold_comps'])),
    'gold_category_max_error':float(np.max(np.abs(np.cumsum(l['slice_gold_cats'],axis=1)-l['cum_gold_cats'])))}
res['ward_placed_total']=int(se[:,:,:,17].sum())
res['negative_residual_team_window_cells']=int(np.sum(l['slice_gold_cats'][:,:,:,4]<0))
st=l['window_end_states']
res['elder_available_but_no_team_soul_t0']=int(np.sum(q['avail_t0'][:,1]&~np.any(st[:,0,:,1]>0,axis=1)))
res['missing_target_fields']=[k for k in ['target_support','source_status','end_ms','winning_team','terminal_class'] if k not in l.files]
res['pareto_no_difference_by_horizon']=np.sum(l['cum_verdict_v1']==3,axis=0).tolist()

# Compare retained champion growth/item values with target-patch source catalogs again.
idx=read(PREP/'knowledge_multipatch_v5/PATCH_KNOWLEDGE_INDEX_V5.json')
res['knowledge_current_check']={}
for p in ['15.18','15.19','15.20','15.21','15.22']:
    b=read(PREP/f'knowledge_multipatch_v5/bundles/bundle_{p.replace(".","_")}_v2.json')
    src=idx['ddragon_sources_by_patch'][p]
    dd=read(Path(src['championFull_path']))['data']
    mism=[]
    for c in b['champions']:
        for f in c['growth_facts']:
            key=f['key'].removeprefix('raw.')
            if key in dd[c['canonical_name']]['stats'] and f['value']!=dd[c['canonical_name']]['stats'][key]:
                mism.append({'champion':c['canonical_name'],'field':key,'saved':f['value'],'source':dd[c['canonical_name']]['stats'][key]})
    res['knowledge_current_check'][p]={'growth_mismatches':len(mism),'example':mism[:1],
        'source_builds':dict(Counter(str(s.get('build')) for s in b['sources']))}
proc=subprocess.run([sys.executable,'-X','utf8','-B',str(PREP/'validate_multipatch_prep_v5.py')],capture_output=True,text=True,encoding='utf-8')
(OUT/'PREP_VALIDATOR.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
res['prep_validator']={'returncode':proc.returncode,'stdout':proc.stdout.strip()}
res['code_bindings']={n:sha(PREP/n) for n in ['build_window_evidence_v5.py','build_multiwindow_outcome_ledger_v5.py','build_multipatch_prep_v5.py']}
(OUT/'AUDIT_RESULTS.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in res.items() if k not in ['array_schema','archive_hashes','code_bindings','knowledge_current_check']},ensure_ascii=False,indent=2))
print('Archive hashes:',all(x['matches'] for x in res['archive_hashes']))
print('Growth mismatch counts:',{p:v['growth_mismatches'] for p,v in res['knowledge_current_check'].items()})
