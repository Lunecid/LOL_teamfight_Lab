"""Read-only PostgreSQL metadata; libpq handles local credentials, never printed."""
from pathlib import Path
import subprocess,os,json
H=Path(__file__).resolve().parent
PSQL=os.environ.get("LOL_PSQL_BIN", "psql")
CONN=['-X','-w','-h',os.environ.get('LOL_PGHOST','127.0.0.1'),'-p',os.environ.get('LOL_PGPORT','5432'),'-U',os.environ.get('LOL_PGUSER','postgres'),'-d',os.environ.get('LOL_PGDATABASE','lol_project_raw_catalog_20261018')]
def query(sql):
 env=os.environ.copy();env['PGOPTIONS']='-c default_transaction_read_only=on -c statement_timeout=60000';env['PGCLIENTENCODING']='UTF8'
 p=subprocess.run([PSQL,*CONN,'-At','-c',sql],capture_output=True,text=True,encoding='utf-8',env=env)
 if p.returncode:raise RuntimeError(p.stderr.strip())
 return p.stdout.strip()
if __name__=='__main__':
 sql="""SELECT json_build_object('database',current_database(),'server_version',current_setting('server_version'),'read_only',current_setting('default_transaction_read_only'),'tables',(SELECT json_agg(t) FROM (SELECT table_schema,table_name,json_agg(json_build_object('column',column_name,'type',data_type) ORDER BY ordinal_position) AS columns FROM information_schema.columns WHERE table_schema NOT IN ('pg_catalog','information_schema') GROUP BY table_schema,table_name ORDER BY table_schema,table_name) t),'indexes',(SELECT json_agg(i) FROM (SELECT schemaname,tablename,indexname,indexdef FROM pg_indexes WHERE schemaname NOT IN ('pg_catalog','information_schema')) i));"""
 data=json.loads(query(sql));(H/'DB_SCHEMA.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'database':data['database'],'read_only':data['read_only'],'tables':data['tables']},ensure_ascii=False))
