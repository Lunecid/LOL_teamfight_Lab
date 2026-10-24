"""Append isolated raw-current catalog; all source datasets remain immutable.

Credentials are read by libpq through PGPASSFILE supplied by the caller.
prepare: source-backed CSV staging, hashes and per-target expected aggregates.
load: atomic new-schema creation with COPY; existing schema fails closed.
verify: full ordered roundtrip of every stored source column plus SQL aggregates.
"""
from pathlib import Path
from collections import Counter,defaultdict
from contextlib import ExitStack
import csv,hashlib,json,os,subprocess,sys,time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT/'holdout_r3'))
from split import roles,bucket
RAW=Path(os.environ["LOL_RAW_MATCH_ROOT"])
SCHEMA='raw_current_r3_20261018'
PSQL=os.environ.get("LOL_PSQL_BIN", "psql")
CONN=['-X','-w','-h',os.environ.get('LOL_PGHOST','127.0.0.1'),'-p',os.environ.get('LOL_PGPORT','5432'),'-U',os.environ.get('LOL_PGUSER','postgres'),'-d',os.environ.get('LOL_PGDATABASE','lol_project_raw_catalog_20261018')]
AUDITS={'FIT':ROOT/'full_fit_raw_audit_20261018/full_02/MATCH_AUDIT.jsonl','HOLDOUT':ROOT/'holdout_r3/audit_01/MATCH_AUDIT.jsonl'}
FAMILIES={'FIT':ROOT/'multiwindow_r3/production_01','HOLDOUT':ROOT/'holdout_r3/production_01'}
TABLES={
 'audits':('family,match_id,record_text','family,match_id'),
 'matches':('match_id,patch,split_role,macro_role,bucket,disposition,audit_family','match_id'),
 'files':('match_id,file_kind,absolute_path,present,size_bytes,mtime_ns,audit_sha256,audit_size_bytes','match_id,file_kind'),
 'sources':('match_id,family,n_queries,record_text','match_id'),
 'shards':('family,shard,absolute_path,sha256,n_queries','family,shard'),
 'queries':('row_id,match_id,family,shard,shard_row,t0_ms,tau_ms,selection_supported,history_ms_text,player_support,prefix_support,skill_support,buff_support,focal_support,soul_status_text,availability_status_text','row_id'),
 'horizons':('row_id,horizon_s,endpoint_ms,endpoint_supported,actual_elapsed_ms,terminal_boundary_shift,gold_support,slice_gold_support,event_support,slice_event_support,economic_supported,economic_sign,terminal_class,terminal_supported,hazard_class,hazard_supported,pareto_class,pareto_supported,direct_reward_support,remaining_value_supported','row_id,horizon_s')}

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def env(readonly=True):
    e=os.environ.copy();e.update(PGCLIENTENCODING='UTF8',PGOPTIONS=f'-c default_transaction_read_only={"on" if readonly else "off"} -c statement_timeout=0 -c lock_timeout=10000')
    return e
def sql(query):
    p=subprocess.run([PSQL,*CONN,'-qAt','-v','ON_ERROR_STOP=1','-c',query],capture_output=True,env=env())
    if p.returncode:raise RuntimeError('Read-only SQL failed: '+p.stderr.decode('utf-8',errors='replace'))
    return p.stdout.decode('utf-8').strip()
def bithex(x):return '\\x'+np.packbits(np.asarray(x,dtype=np.uint8).reshape(-1),bitorder='little').tobytes().hex()
def val(x):
    if x is None:return '\\N'
    if isinstance(x,(bool,np.bool_)):return 't' if x else 'f'
    return str(x)
def rowhash(row):return (json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8')
def progress(phase,**kw):
    dump(HERE/'PROGRESS.json',{'phase':phase,**kw});print(json.dumps({'phase':phase,**kw}),flush=True)

def prepare():
    if (HERE/'STAGING.json').exists():raise RuntimeError('Staging already exists; reuse immutable preparation')
    stage=HERE/'staging';stage.mkdir(exist_ok=True)
    manifests={}; canonical={}; expected_counts=Counter(); aggregates=defaultdict(Counter)
    for f,p in AUDITS.items():manifests[str(p)]=sha(p)
    for f,p in FAMILIES.items():
        for name in ('SOURCE_LEDGER.jsonl','BUILD_SUMMARY.json','VALIDATION.json','DISPOSITIONS.jsonl'):
            manifests[str(p/name)]=sha(p/name)
        v=read(p/'VALIDATION.json');b=read(p/'BUILD_SUMMARY.json')
        assert v['pass'] and v['full_population'] and sha(p/'BUILD_SUMMARY.json')==v['build_summary_sha256']
        assert b['source_audit_sha256']==manifests[str(AUDITS[f])]
    with ExitStack() as stack:
        writers={}
        for table,(cols,_) in TABLES.items():
            stream=stack.enter_context((stage/(table+'.csv')).open('w',encoding='utf-8',newline=''))
            writer=csv.writer(stream,lineterminator='\n');writer.writerow(cols.split(','));writers[table]=writer
        def emit(table,*values):writers[table].writerow([val(v) for v in values]);expected_counts[table]+=1
        audit_rows={}
        for family,path in AUDITS.items():
            with path.open(encoding='utf-8') as f:
                for line in f:
                    r=json.loads(line);mid=r['match_id'];assert bucket(mid)==r['bucket']
                    emit('audits',family,mid,line.rstrip('\r\n'))
                    if mid in canonical:
                        old=canonical[mid][1]
                        assert old['scope']==r['scope']=='UNRESOLVED' and old['disposition']==r['disposition']=='QUARANTINE'
                    canonical[mid]=(family,r)
        progress('audits',audit_records=expected_counts['audits'],unique_matches=len(canonical))
        disposition_counts=Counter(); file_missing=0; unaudited_hash=0
        for mid,(family,r) in canonical.items():
            role,macro=roles(r.get('patch'),r['bucket'])
            if r['scope']=='UNRESOLVED':role=None;macro='UNRESOLVED'
            assert macro not in ('OTHER','UNUSED')
            if r.get('split_role') and r['scope']!='UNRESOLVED':assert role==r['split_role']
            emit('matches',mid,r.get('patch'),role,macro,r['bucket'],r['disposition'],family)
            disposition_counts[(macro,r.get('patch'),r['disposition'])]+=1
            for kind in ('detail','timeline'):
                stems=r.get('source_files',{}).get(kind,[]);assert len(stems)<=1
                p=RAW/kind/((stems[0] if stems else mid)+'.json')
                assert p.parent==RAW/kind
                present=p.is_file();stat=p.stat() if present else None
                audited_bytes=r.get(kind+'_bytes');audited_hash=r.get(kind+'_sha256')
                if audited_bytes is not None:assert present and stat.st_size==audited_bytes,(mid,kind,'size changed')
                if audited_hash:assert present and len(audited_hash)==64
                file_missing+=not present;unaudited_hash+=present and audited_hash is None
                emit('files',mid,kind,str(p),present,stat.st_size if stat else None,stat.st_mtime_ns if stat else None,audited_hash,audited_bytes)
        retained={};queries_by_match=Counter()
        for family,p in FAMILIES.items():
            with (p/'SOURCE_LEDGER.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    r=json.loads(line);mid=r['match_id'];a=canonical[mid][1]
                    assert mid not in retained and a['disposition'].startswith('RETAIN')
                    assert all(a[k]==r[k] for k in ('patch','detail_sha256','timeline_sha256'))
                    retained[mid]=(family,r['n_queries']);emit('sources',mid,family,r['n_queries'],line.rstrip('\r\n'))
        assert set(retained)=={m for m,(_,r) in canonical.items() if r['disposition'].startswith('RETAIN')}
        progress('files',files=expected_counts['files'],missing=file_missing,retained=len(retained))
        schemas={};shard_sources={}
        for family,p in FAMILIES.items():
            b=read(p/'BUILD_SUMMARY.json');v=read(p/'VALIDATION.json');schemas[family]={'event_fields':b['event_fields'],'player_fields':b['player_fields'],'objective_order':b['objective_order']}
            for sn in b['shards']:
                path=p/'shards'/sn;digest=sha(path);assert digest==v['shard_sha256'][sn]
                shard_sources[str(path)]=digest
                with np.load(path,allow_pickle=False) as z:
                    keys=['row_id','match_id','patch','t0_ms','tau_ms','selection_supported','history_ms','player_support','prefix_team_events_support','skill_level_up_count_support','actual_buff_holder_supported','focal_kills_supported','soul_source_status','objective_availability_status','endpoint_ms','endpoint_supported','actual_elapsed_ms','terminal_boundary_shift','gold_supported','slice_gold_supported','cum_event_supported','slice_event_supported','net_gold','economic_sign','terminal_class','terminal_supported','hazard_class','hazard_at_risk_supported','pareto_class','pareto_supported','direct_reward_supported','remaining_value_supported']
                    a={k:z[k] for k in keys}
                    n=len(a['row_id']);emit('shards',family,sn,str(path),digest,n)
                    for i in range(n):
                        mid=str(a['match_id'][i]);rid=str(a['row_id'][i]);r=canonical[mid][1]
                        assert retained[mid][0]==family and r['patch']==str(a['patch'][i])
                        role,macro=roles(r['patch'],r['bucket']);queries_by_match[mid]+=1
                        emit('queries',rid,mid,family,sn,i,a['t0_ms'][i],a['tau_ms'][i],a['selection_supported'][i],json.dumps(a['history_ms'][i].tolist()),bithex(a['player_support'][i]),bithex(a['prefix_team_events_support'][i]),bithex(a['skill_level_up_count_support'][i]),bithex(a['actual_buff_holder_supported'][i]),bithex(a['focal_kills_supported'][i]),json.dumps(a['soul_source_status'][i].tolist()),json.dumps(a['objective_availability_status'][i].tolist()))
                        for j,h in enumerate((60,120,180)):
                            econ=bool(np.isfinite(a['net_gold'][i,j]));ev=a['cum_event_supported'][i,j]
                            emit('horizons',rid,h,a['endpoint_ms'][i,j],a['endpoint_supported'][i,j],a['actual_elapsed_ms'][i,j],a['terminal_boundary_shift'][i,j],bithex(a['gold_supported'][i,j]),bithex(a['slice_gold_supported'][i,j]),bithex(ev),bithex(a['slice_event_supported'][i,j]),econ,a['economic_sign'][i,j],a['terminal_class'][i,j],a['terminal_supported'][i,j],a['hazard_class'][i,j],a['hazard_at_risk_supported'][i,j],a['pareto_class'][i,j],a['pareto_supported'][i,j],bithex(a['direct_reward_supported'][i,j]),a['remaining_value_supported'][i,j])
                            c=aggregates[(role,r['patch'],h)];c['queries']+=1;c['selection_supported']+=int(a['selection_supported'][i]);c['economic_supported']+=econ;c['terminal_supported']+=int(a['terminal_supported'][i,j]);c['pareto_supported']+=int(a['pareto_supported'][i,j]);c['all_event_supported']+=int(ev.all());c['all_epic_supported']+=int(ev[:,2:8].all())
                            for team in range(2):
                                for k,field in enumerate(b['event_fields']):c[f'event_{team}_{field}']+=int(ev[team,k])
                if int(sn.split('.')[0])%100==0:progress('shards',family=family,shard=sn,queries=expected_counts['queries'])
        assert all(queries_by_match[m]==n for m,(_,n) in retained.items())
        assert expected_counts['queries']==sum(read(p/'BUILD_SUMMARY.json')['n_queries'] for p in FAMILIES.values())
    # Hash source control files once more: staging must not span a changed ledger.
    assert all(sha(Path(p))==digest for p,digest in manifests.items())
    dump(HERE/'SOURCE_MANIFEST.json',{'control_files':manifests,'shards':shard_sources})
    dump(HERE/'FILE_EXPECTED.json',{'counts':dict(expected_counts),'dispositions':[{'macro_role':k[0],'patch':k[1],'disposition':k[2],'n':n} for k,n in disposition_counts.items()], 'support':[{'split_role':k[0],'patch':k[1],'horizon_s':k[2],**dict(v)} for k,v in sorted(aggregates.items())],'missing_files':file_missing,'present_files_without_audit_hash':unaudited_hash})
    dump(HERE/'MASK_LAYOUT.json',{'encoding':'PostgreSQL bytea; numpy.packbits bitorder=little; C flatten; PostgreSQL get_bit index matches logical index; zero padding bits', 'player_support':[5,10,18],'prefix_support':[2,20],'skill_support':[10,4],'buff_support':[10,2],'focal_support':[2],'gold_support':[2],'event_support':[2,20],'schemas':schemas,'raw_audit_masks':'audits.record.field_masks/event_masks retain exact inclusive source coordinates; rule_masks retain onset metadata if present','scope':'Audit dispositions can use full match. They are not per-query model inputs. Selection and target masks stay metadata/targets.'})
    dump(HERE/'STAGING.json',{'schema':SCHEMA,'tables':dict(expected_counts),'csv_sha256':{t:sha(stage/(t+'.csv')) for t in TABLES},'source_manifest_sha256':sha(HERE/'SOURCE_MANIFEST.json'),'raw_root':str(RAW),'preserved_reserved_population':26415,'raw_hash_scope':'Historical audited hashes; current existence and audited size checked; raw content not rehashed in this indexing task.','training':False})
    progress('prepared',**dict(expected_counts))

def old_snapshot(destination):
    query="BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY; COPY (SELECT * FROM raw_cache.catalog_scans ORDER BY scan_id) TO STDOUT WITH CSV HEADER; COPY (SELECT * FROM raw_cache.files ORDER BY scan_id,relative_path) TO STDOUT WITH CSV HEADER; COMMIT;"
    with destination.open('wb') as out:
        p=subprocess.run([PSQL,*CONN,'-qAt','-v','ON_ERROR_STOP=1','-c',query],stdout=out,stderr=subprocess.PIPE,env=env())
    if p.returncode:raise RuntimeError('Existing schema snapshot failed')
    return sha(destination)

def load():
    s=read(HERE/'STAGING.json');assert s['schema']==SCHEMA
    layouts=read(HERE/'MASK_LAYOUT.json')['schemas'];assert layouts['FIT']==layouts['HOLDOUT']
    assert all(sha(HERE/'staging'/(t+'.csv'))==digest for t,digest in s['csv_sha256'].items())
    assert not sql(f"SELECT schema_name FROM information_schema.schemata WHERE schema_name='{SCHEMA}'"),'Schema already exists; refuse to replace'
    before=old_snapshot(HERE/'EXISTING_RAW_CACHE_BEFORE.csv');dump(HERE/'EXISTING_RAW_CACHE_BEFORE.json',{'sha256':before})
    metadata={'staging':s,'source_manifest':read(HERE/'SOURCE_MANIFEST.json'),'mask_layout':read(HERE/'MASK_LAYOUT.json'),'registration_policy':{'old_schema':'raw_cache','raw_contents_rehashed':False,'existing_file_loader_preserved':True,'training':False,'schema_name':SCHEMA,'raw_root':str(RAW)}}
    dump(HERE/'CATALOG_METADATA.json',metadata)
    with (HERE/'catalog_info.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['key','payload_text'])
        for key,value in metadata.items():w.writerow([key,json.dumps(value,ensure_ascii=False,separators=(',',':'))])
    commands=['\\set ON_ERROR_STOP on','BEGIN;','SET LOCAL synchronous_commit=on;', (HERE/'schema.sql').read_text(encoding='utf-8')]
    commands.append(f"\\copy {SCHEMA}.catalog_info (key,payload_text) FROM '{(HERE/'catalog_info.csv').as_posix()}' WITH (FORMAT csv, HEADER true)")
    for table,(columns,_) in TABLES.items():
        path=(HERE/'staging'/(table+'.csv')).as_posix()
        commands.append(f"\\copy {SCHEMA}.{table} ({columns}) FROM '{path}' WITH (FORMAT csv, HEADER true, NULL '\\N')")
    for table in TABLES:commands.append(f'ANALYZE {SCHEMA}.{table};')
    for table,n in s['tables'].items():commands.append(f"DO $$ BEGIN IF (SELECT count(*) FROM {SCHEMA}.{table}) <> {n} THEN RAISE EXCEPTION 'count mismatch: {table}'; END IF; END $$;")
    commands.append((HERE/'checks.sql').read_text(encoding='utf-8'))
    commands+=['COMMIT;']
    p=HERE/'IMPORT.sql';p.write_text('\n'.join(commands)+'\n',encoding='utf-8')
    with (HERE/'IMPORT.log').open('wb') as log:
        run=subprocess.run([PSQL,*CONN,'-v','ON_ERROR_STOP=1','-f',str(p)],stdout=log,stderr=subprocess.STDOUT,env=env(False))
    if run.returncode:raise RuntimeError('Atomic import failed; inspect IMPORT.log. No schema committed.')
    dump(HERE/'IMPORT_COMPLETE.json',{'committed':True,'schema':SCHEMA,'staging_sha256':sha(HERE/'STAGING.json'),'schema_sha256':sha(HERE/'schema.sql'),'checks_sha256':sha(HERE/'checks.sql')})
    progress('imported')

def digest_csv(path,order):
    # Full byte-column comparison is carried out after externally ordering staging
    # through the read-only DB COPY below; per-row sha sums are order independent.
    total=0;n=0
    with path.open(encoding='utf-8',newline='') as f:
        reader=csv.reader(f);header=next(reader)
        for row in reader:total=(total+int.from_bytes(hashlib.sha256(rowhash(row)).digest(),'big'))%(1<<256);n+=1
    return {'n':n,'column_names':header,'sha256_multiset_sum':f'{total:064x}'}

def verify():
    assert read(HERE/'IMPORT_COMPLETE.json')['committed']
    expected=read(HERE/'FILE_EXPECTED.json');checks={}
    db_meta=json.loads(sql(f"SELECT json_object_agg(key,payload) FROM {SCHEMA}.catalog_info"))
    assert db_meta==read(HERE/'CATALOG_METADATA.json')
    exports=HERE/'roundtrip';exports.mkdir(exist_ok=True)
    for table,(cols,order) in TABLES.items():
        dst=exports/(table+'.csv')
        query=f"COPY (SELECT {cols} FROM {SCHEMA}.{table} ORDER BY {order}) TO STDOUT WITH (FORMAT csv, HEADER true, NULL '\\N')"
        with dst.open('wb') as out:
            p=subprocess.run([PSQL,*CONN,'-qAt','-v','ON_ERROR_STOP=1','-c',query],stdout=out,stderr=subprocess.PIPE,env=env())
        if p.returncode:raise RuntimeError('Roundtrip failed: '+table)
        a=digest_csv(HERE/'staging'/(table+'.csv'),order);b=digest_csv(dst,order)
        assert a==b,(table,a,b);checks[table]=b;progress('roundtrip_verified',table=table,rows=b['n'])
    actual=json.loads(sql(f"SELECT json_agg(t) FROM (SELECT * FROM {SCHEMA}.support_summary ORDER BY split_role,patch,horizon_s) t"))
    assert actual==expected['support'],'SQL target support aggregates differ from NumPy'
    prior_path=ROOT/'holdout_r3/production_01/review/SUPPORT_BY_ROLE_PATCH_HORIZON.csv'
    prior_hash=sha(prior_path)
    hazard=json.loads(sql(f"SELECT json_agg(t) FROM (SELECT m.split_role,h.horizon_s,count(*) FILTER(WHERE hazard_supported) AS n FROM {SCHEMA}.horizons h JOIN {SCHEMA}.queries q USING(row_id) JOIN {SCHEMA}.matches m USING(match_id) GROUP BY 1,2) t"))
    hmap={(r['split_role'],r['horizon_s']):r['n'] for r in hazard}
    target_map={'gold_both':'economic_supported','epics_all':'all_epic_supported','all_20_events':'all_event_supported','terminal':'terminal_supported','pareto':'pareto_supported'}
    prior_checked=0
    with prior_path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            h=int(r['horizon_s']);group=r['group']
            selected=[a for a in actual if a['horizon_s']==h and (a['split_role']==group if '_' in group else a['split_role'].split('_')[0]==group)]
            assert selected
            numerator=sum(hmap[(a['split_role'],h)] if r['target']=='hazard_at_risk' else a[target_map[r['target']]] for a in selected)
            assert sum(a['queries'] for a in selected)==int(r['denominator'])
            assert numerator==int(r['supported']),(group,h,r['target'],numerator,r['supported'])
            prior_checked+=1
    assert sha(prior_path)==prior_hash
    dispositions=json.loads(sql(f"SELECT json_agg(t) FROM (SELECT macro_role,patch,disposition,count(*) AS n FROM {SCHEMA}.matches GROUP BY 1,2,3 ORDER BY 1,2,3) t"))
    key=lambda r:(r['macro_role'],str(r['patch']),r['disposition'])
    assert sorted(dispositions,key=key)==sorted(expected['dispositions'],key=key)
    # Independently validate frozen role mapping in SQL; bucket verified from original salt during preparation.
    split_bad=int(sql(f"SELECT count(*) FROM {SCHEMA}.matches WHERE split_role IS DISTINCT FROM CASE WHEN patch IN ('15.18','15.19') AND bucket<80 THEN 'FIT_'||patch WHEN patch='15.20' THEN CASE WHEN bucket<80 THEN 'FIT_15.20' ELSE 'STOP_15.20' END WHEN patch='15.21' THEN CASE WHEN bucket<50 THEN 'DEV_15.21' WHEN bucket<75 THEN 'CAL_15.21' ELSE 'AUDIT_15.21' END WHEN patch='15.22' THEN 'TEST_15.22' ELSE NULL END"));assert split_bad==0
    after=old_snapshot(HERE/'EXISTING_RAW_CACHE_AFTER.csv');assert after==read(HERE/'EXISTING_RAW_CACHE_BEFORE.json')['sha256']
    old_indexes=read(ROOT/'holdout_r3/DB_SCHEMA.json')['indexes']
    current_indexes=json.loads(sql("SELECT json_agg(t) FROM (SELECT schemaname,tablename,indexname,indexdef FROM pg_indexes WHERE schemaname='raw_cache') t"))
    assert sorted(current_indexes,key=lambda x:x['indexname'])==sorted(old_indexes,key=lambda x:x['indexname'])
    manifest=read(HERE/'SOURCE_MANIFEST.json')
    assert all(sha(Path(p))==digest for p,digest in manifest['control_files'].items())
    assert all(sha(Path(p))==digest for p,digest in manifest['shards'].items())
    result={'pass':True,'schema':SCHEMA,'tables':checks,'support_groups':len(actual),'sql_support_equals_numpy':True,'prior_published_support_rows_matched':prior_checked,'prior_published_support_sha256':prior_hash,'dispositions_equal':True,'split_mismatches':split_bad,'old_raw_cache_sha256_before_and_after':after,'old_raw_cache_unchanged':True,'old_raw_cache_indexes_unchanged':True,'source_control_files_and_shards_unchanged':True,'raw_content_rehashed':False,'training':False}
    dump(HERE/'SQL_SUPPORT_SUMMARY.json',actual);dump(HERE/'VERIFICATION.json',result);progress('verified',support_groups=len(actual))

if __name__=='__main__':{'prepare':prepare,'load':load,'verify':verify}[sys.argv[1]]()
