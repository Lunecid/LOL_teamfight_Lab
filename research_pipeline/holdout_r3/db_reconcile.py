"""Read-only catalog snapshot and exact ID reconciliation. Credentials via libpq only."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, json, os, subprocess, hashlib, re, sys
from datetime import datetime, timezone
from db_probe import PSQL, CONN

H=Path(__file__).resolve().parent
RAW=Path(os.environ["LOL_RAW_MATCH_ROOT"])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def canonical(x):
    x=x.upper()
    return x if '_' in x else 'KR_'+x
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    target=H/'DB_METADATA_SNAPSHOT.txt'
    sql="""BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;
    SELECT json_build_object('captured_at',now(),'snapshot',pg_current_snapshot()::text,
      'read_only',current_setting('transaction_read_only'),
      'scans',(SELECT json_agg(t) FROM (SELECT * FROM raw_cache.catalog_scans ORDER BY scan_id) t));
    COPY (SELECT scan_id,relative_path,match_id,file_kind,size_bytes,mtime_utc
          FROM raw_cache.files ORDER BY scan_id,match_id,file_kind,relative_path)
    TO STDOUT WITH CSV HEADER;
    COMMIT;"""
    env=os.environ.copy()
    env.update(PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=120000',PGCLIENTENCODING='UTF8')
    if '--reuse-snapshot' not in sys.argv:
        with target.open('wb') as out:
            proc=subprocess.run([PSQL,*CONN,'-qAt','-v','ON_ERROR_STOP=1','-c',sql],stdout=out,stderr=subprocess.PIPE,env=env)
        if proc.returncode:raise RuntimeError('Read-only snapshot failed; no credentials logged')
    ids=set(); kinds=Counter(); groups=defaultdict(Counter); n=0; total_bytes=0
    scan_counts=Counter(); scan_ids=defaultdict(set); path_errors=0; scan_bytes=Counter(); invalid_db_ids=0
    with target.open(encoding='utf-8',newline='') as f:
        meta=json.loads(next(f)); assert meta['read_only']=='on'
        for row in csv.DictReader(f):
            mid=canonical(row['match_id']);ids.add(mid);n+=1
            invalid_db_ids+=not bool(re.fullmatch(r'KR_\d+',row['match_id']))
            kinds[row['file_kind']]+=1;groups[(row['scan_id'],mid)][row['file_kind']]+=1
            scan_counts[row['scan_id']]+=1;scan_ids[row['scan_id']].add(mid)
            total_bytes+=int(row['size_bytes'])
            scan_bytes[row['scan_id']]+=int(row['size_bytes'])
            path=Path(row['relative_path'])
            if path.is_absolute() or '..' in path.parts:path_errors+=1
    assert all(scan_counts[s['scan_id']]==s['file_count'] and len(scan_ids[s['scan_id']])==s['unique_match_count'] for s in meta['scans'])
    assert set(scan_counts)=={s['scan_id'] for s in meta['scans']}
    assert all(scan_bytes[s['scan_id']]==s['total_size_bytes'] for s in meta['scans'])
    expected=Counter({'events_json':1,'meta_json':1,'npz':1})
    group_bad=sum(g!=expected for g in groups.values())
    details={canonical(p.stem) for p in (RAW/'detail').glob('*.json')}
    timelines={canonical(p.stem) for p in (RAW/'timeline').glob('*.json')}
    invalid_raw_ids=sum(not bool(re.fullmatch(r'KR_\d+',p.stem)) for kind in ('detail','timeline') for p in (RAW/kind).glob('*.json'))
    assert invalid_db_ids==invalid_raw_ids==0
    sources=[('FIT',H.parent/'multiwindow_r3/production_01/SOURCE_LEDGER.jsonl'),('HELDOUT',H/'production_01/SOURCE_LEDGER.jsonl')]
    coverage=defaultdict(Counter); ledger_ids=set(); hashes={}
    with (H/'DB_LEDGER_ID_COVERAGE.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['match_id','role','patch','query_rows','db_present'])
        for family,path in sources:
            hashes[str(path.relative_to(H.parent))]=sha(path)
            with path.open(encoding='utf-8') as src:
                for line in src:
                    r=json.loads(line);mid=canonical(r['match_id'])
                    assert re.fullmatch(r'KR_\d+',r['match_id'])
                    assert mid not in ledger_ids;ledger_ids.add(mid)
                    role=r.get('macro_role',family);key=role+'_'+r['patch'];c=coverage[key]
                    present=mid in ids;q=r['n_queries']
                    c['retained_matches']+=1;c['indexed_matches']+=int(present)
                    c['query_rows']+=q;c['indexed_query_rows']+=q*int(present)
                    writer.writerow([mid,role,r['patch'],q,int(present)])
    extra=ids-details
    (H/'DB_IDS_OUTSIDE_CURRENT_RAW.txt').write_text('\n'.join(sorted(extra))+'\n',encoding='utf-8')
    db_intersect_detail=len(ids&details);db_intersect_ledger=len(ids&ledger_ids)
    result={'status':'RECONCILED_DIFFERENT_CORPUS' if db_intersect_detail==0 and db_intersect_ledger==0 else 'RECONCILED_OVERLAPPING_CORPUS',
      'database':os.environ.get('LOL_PGDATABASE','lol_project_raw_catalog_20261018'),
      'read_only':True,'used_for_index_selection':False,'used_for_independent_id_reconciliation':True,
      'source_of_ledger':'read-only raw filesystem inventory with immutable source hashes',
      'snapshot':meta,'db_files':n,'db_unique_matches':len(ids),'db_file_kinds':dict(kinds),
      'db_total_bytes':total_bytes,'incomplete_or_duplicate_file_kind_groups':group_bad,'unsafe_relative_paths':path_errors,
      'id_namespace_verified':invalid_db_ids==invalid_raw_ids==0,
      'catalog_scan_ids_counts_and_bytes_match':True,
      'current_raw_root':str(RAW),'raw_detail_ids':len(details),'raw_timeline_ids':len(timelines),
      'db_intersect_raw_detail':db_intersect_detail,'db_outside_raw_detail':len(extra),
      'raw_detail_absent_from_db':len(details-ids),'retained_ledger_matches':len(ledger_ids),
      'db_intersect_retained_ledger':db_intersect_ledger,'coverage':dict(sorted(coverage.items())),
      'limitations':['Catalog indexes derived cache events/meta/npz, not current raw detail/timeline.',
        'No patch or content hashes in DB: ID overlap does not establish identical content or lineage.',
        'DB index absence is not a quarantine criterion; original splits, masks and ledgers unchanged.',
        'Filesystem cache size/mtime not revalidated by this metadata and ID comparison.'],
      'source_ledger_sha256':hashes,'snapshot_sha256':sha(target),'coverage_sha256':sha(H/'DB_LEDGER_ID_COVERAGE.csv'),
      'training_performed':False}
    dump(H/'DB_INDEX_RECONCILIATION.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['snapshot','source_ledger_sha256','limitations']},ensure_ascii=False))
if __name__=='__main__':main()
