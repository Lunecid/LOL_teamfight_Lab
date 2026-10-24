"""Read-only inventory: full index and filenames; bounded metadata probes. No training."""
import collections, datetime, hashlib, json, os, pathlib, re
ROOT=pathlib.Path(__file__).resolve().parent
PARENT=ROOT.parent
REPO=pathlib.Path(os.environ["LOL_RESEARCH_ROOT"])
PRIOR=REPO/'reports/journal_tog_20261005/evidence/EXTERNAL_DATA_INVENTORY_20261009.json'
SAMPLE=PARENT/'objective_audit/audit_results.json'
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def listing(p):
    if not p.is_dir():return set()
    return {e.name for e in os.scandir(p) if e.is_file()}
def sethash(values):return sha(('\n'.join(sorted(values))+'\n').encode())
prior=load(PRIOR);sample=load(SAMPLE)
out={'version':'v5-multipatch-inventory-r1','captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'All entries in four current cache indices, all filenames in their directories and nine known raw source pairs; three metadata files per indexed patch; existing schema evidence reused, not rerun','no_training':True,'no_new_labels':True,'raw_payloads_opened':0,'npz_payloads_opened':0,'sources':[{'path':str(p),'sha256':sha(p.read_bytes())} for p in [PRIOR,SAMPLE]],'indexed_caches':{},'raw_directories':{},'overlaps':[],'limitations':['Filename presence is not payload integrity or label eligibility','Full schema/state/rank/terrain/endpoint acceptance remains pending','Prior raw per-patch counts are dated 2026-10-09; current file totals do not revalidate every raw payload','Known directories only; no claim to have searched every disk','Previous use of held-out patches prevents a pristine-test claim']}
index_maps={};file_sets={}
for name,s in sample['sources'].items():
    idxpath=pathlib.Path(s['index']);cache=pathlib.Path(s['cache'])
    b=idxpath.read_bytes();idx=json.loads(b);index_maps[name]=idx
    names=listing(cache);file_sets[name]=names
    groups={}
    for patch in sorted(set(idx.values())):
        ids=[k for k,v in idx.items() if v==patch]
        complete=[k for k in ids if all(k+'.'+ext in names for ext in ['meta.json','events.json','npz'])]
        g={'indexed_matches':len(ids),'complete_file_triplets':len(complete),'missing_triplets':len(ids)-len(complete),'match_set_sha256':sethash(ids),'metadata_probe_n':0,'metadata_patch_mismatch':0,'metadata_errors':0,'metadata_keys':set()}
        for mid in sorted(ids,key=lambda k:sha(('v5-mp-meta:'+k).encode()))[:3]:
            try:
                meta=load(cache/(mid+'.meta.json'));g['metadata_probe_n']+=1;g['metadata_patch_mismatch']+=meta.get('patch')!=patch;g['metadata_keys'].update(meta)
            except Exception:g['metadata_errors']+=1
        g['metadata_keys']=sorted(g['metadata_keys'])
        evidence=sample['groups'].get(name+'/'+patch,{})
        g['prior_schema_probe']={'n':evidence.get('read'),'scope':sample['scope'],'patch_mismatch':evidence.get('patch_mismatch'),'errors':len(evidence.get('errors',[])),'npz_keys':sorted(evidence.get('npz_keys',{})),'SKILL_LEVEL_UP_records':evidence.get('events',{}).get('SKILL_LEVEL_UP',0)}
        groups[patch]=g
    out['indexed_caches'][name]={'index_path':str(idxpath),'cache_path':str(cache),'index_sha256':sha(b),'index_matches_previous_audit':sha(b)==s['index_sha256'],'index_stable_during_read':sha(b)==sha(idxpath.read_bytes()),'directory_files':len(names),'groups':groups}
for i,a in enumerate(index_maps):
    for b in list(index_maps)[i+1:]:out['overlaps'].append({'a':a,'b':b,'match_id_intersection':len(set(index_maps[a])&set(index_maps[b]))})
raw_sets={}
for name,v in prior['inventory']['candidates'].items():
    dirs=[pathlib.Path(v['detail_dir']),pathlib.Path(v['timeline_dir'])]
    d,t=[{x[:-5] for x in listing(p) if re.fullmatch(r'[A-Z0-9]+_\d+\.json',x)} for p in dirs]
    raw_sets[name]=d&t
    out['raw_directories'][name]={'detail_path':str(dirs[0]),'timeline_path':str(dirs[1]),'current_detail_files':len(d),'current_timeline_files':len(t),'current_pairs':len(d&t),'detail_only':len(d-t),'timeline_only':len(t-d),'pair_set_sha256':sethash(d&t),'totals_equal_prior':len(d)==v['detail_files'] and len(t)==v['timeline_files'],'prior_per_patch_counts':{p:info['detail_files_parsed'] for p,info in v['by_patch'].items()},'prior_scope':{'generated_utc':prior['inventory_run']['utc'],'detail':'full prior census','timeline':'2 percent sample for large KR 15.x; full for other candidates'},'prior_parse_errors':v['parse_errors']}
for name,rawname in [('later_kr','Lol_project/matches/kr'),('kr_16','2026_current/kr'),('na_16','2026_current/na1')]:
    out['indexed_caches'][name]['indexed_matches_with_current_raw_pair']=len(set(index_maps[name])&raw_sets[rawname])
out['raw_duplicates']={'fusion_2615_pairs':len(raw_sets['fusion_2615/raw/kr']),'fusion_already_in_current_kr':len(raw_sets['fusion_2615/raw/kr']&raw_sets['2026_current/kr']),'top_level_pairs':len(raw_sets['Lol_project/matches+timelines(top-level)']),'top_level_overlap_with_main_kr':len(raw_sets['Lol_project/matches+timelines(top-level)']&raw_sets['Lol_project/matches/kr'])}
out['prior_exposure']={k:v['prior_exposure'] for k,v in prior['candidate_summary'].items() if isinstance(v,dict) and 'prior_exposure' in v}
out['readiness']={'15.14_15.16':'LEGACY_CACHE_ONLY_CONDITIONAL; no source-time/missingness certification; retain legacy benchmark and optional source-sensitivity extension','15.18_15.20':'PRIMARY_TRAIN_CANDIDATES_RAW_BACKED; cache subset available; full producer and knowledge gates pending','15.21':'VALIDATION_CANDIDATE; never fit main predictor here','15.22':'RESERVED_NEW_PROTOCOL_BENCHMARK_TEST_WITH_PRIOR_EXPOSURE; not pristine','16.13_KR_NA':'EXTERNAL_DIAGNOSTICS_WITH_PRIOR_EXPOSURE; do not train main 15.x model','16.14_16.15':'SMALL_EXTERNAL_DIAGNOSTICS; previous use; not main train','15.17_or_pre_15.14':'NOT_FOUND_IN_AUDITED_INVENTORIES','ready_for_full_context_training':False}
(ROOT/'MULTIPATCH_DATA_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for name,v in out['indexed_caches'].items():print(name,{p:(g['indexed_matches'],g['complete_file_triplets']) for p,g in v['groups'].items()})
print('raw pairs',{k:v['current_pairs'] for k,v in out['raw_directories'].items()})
print('raw duplicates',out['raw_duplicates'])
print('No NPZ/raw payload read, labels or training. Saved MULTIPATCH_DATA_AUDIT.json')
