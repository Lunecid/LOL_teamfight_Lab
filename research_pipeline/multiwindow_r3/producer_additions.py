
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
    paths += [AUDIT,ROOT/'full_fit_raw_audit_20261018/review_02/VALIDATION.json',ROOT/'full_fit_raw_audit_20261018/review_02/ARTIFACT_MANIFEST.json',RESEARCH/'studies/observable_state_20261002/population.py',RESEARCH/'worktrees/engagement-state-value/studies/st_dbscan_20261001/kernel.py']
    return {str(p):sha(p) for p in sorted(set(paths))}

def process_batch(rows):
    records=[];sources=[]
    for row in rows:
        rr,ss=process(row);records.extend(rr);sources.append(ss)
    return records,sources

def build(output,limit=None,workers=3):
    assert 1<=workers<=3
    manifest=json.loads((ROOT/'full_fit_raw_audit_20261018/review_02/ARTIFACT_MANIFEST.json').read_text(encoding='utf-8'))
    assert sha(AUDIT)==manifest['files'][str(Path('full_02')/'MATCH_AUDIT.jsonl')]['sha256']
    output.mkdir(parents=True,exist_ok=False);(output/'shards').mkdir()
    (output/'PRODUCER_EXECUTED.py').write_bytes(Path(__file__).read_bytes())
    dump(output/'ENVIRONMENT.json',{'python':sys.version,'numpy':np.__version__,'workers':workers,'cpu_budget':4,'windows_ms':[60000,120000,180000],'input_history_max_frames':5,'input_history_default_frames':3})
    before=source_snapshot();dump(output/'ORIGINAL_HASHES_BEFORE.json',before)
    audit_validation=json.loads((ROOT/'full_fit_raw_audit_20261018/review_02/VALIDATION.json').read_text(encoding='utf-8'));assert audit_validation['passed']
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
    dump(output/'BUILD_SUMMARY.json',{'profile':'MW60_FRAME_r3','full_population':limit is None,'audit_retained_matches':total,'n_matches_with_queries':counts['matches_with_queries'],'n_queries':counts['queries'],'selection_supported_queries':counts['selection_supported_queries'],'counts':dict(counts),'shards':shards,'event_fields':EVENTS,'player_fields':PLAYER,'objective_order':OBJECTIVES,'input_keys':sorted(INPUT_KEYS),'meta_keys':sorted(META_KEYS|{'selection_supported','focal_member_ordinals_json'}),'source_audit_sha256':sha(AUDIT),'producer_sha256':sha(Path(__file__)),'training_performed':False,'U_enabled':False,'elapsed_s':round(time.time()-start,1),'source_status_codes':{'0':'UNKNOWN','1':'EVENT_EXACT','2':'RULE_DERIVED','3':'STRUCTURAL_NA'},'terminal_class_codes':{'-1':'UNKNOWN','0':'ONGOING','1':'BLUE_WIN','2':'RED_WIN'},'pareto_codes':{'-1':'UNKNOWN','0':'BLUE_DOMINATES','1':'RED_DOMINATES','2':'TRADEOFF','3':'NO_DIFFERENCE'}})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--limit',type=int);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args();build(args.output,args.limit,args.workers)
