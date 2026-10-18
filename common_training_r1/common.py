from pathlib import Path
import hashlib,json,os,subprocess
import re
import numpy as np
HERE=Path(__file__).resolve().parent
SCHEMA=os.environ.get('LOL_CATALOG_SCHEMA','research_catalog')
if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',SCHEMA):
    raise ValueError('LOL_CATALOG_SCHEMA must be a single PostgreSQL identifier')

class db:
    """Minimal read-only psql client; authentication is delegated to libpq/PGPASSFILE."""
    @staticmethod
    def sql(query):
        args=[os.environ.get('LOL_PSQL_BIN','psql'),'-X','-w','-h',os.environ.get('LOL_PGHOST','127.0.0.1'),
              '-p',os.environ.get('LOL_PGPORT','5432'),'-U',os.environ.get('LOL_PGUSER','postgres'),
              '-d',os.environ.get('LOL_PGDATABASE','postgres'),
              '-qAt','-v','ON_ERROR_STOP=1','-c',query]
        env=os.environ.copy()
        env['PGCLIENTENCODING']='UTF8'
        env['PGOPTIONS']='-c default_transaction_read_only=on -c statement_timeout=0 -c lock_timeout=10000'
        result=subprocess.run(args,capture_output=True,env=env)
        if result.returncode:
            raise RuntimeError('Read-only PostgreSQL query failed: '+result.stderr.decode('utf-8',errors='replace'))
        return result.stdout.decode('utf-8').strip()
EVENTS=['kills','assists','elem_dragon','elder_dragon','baron','atakhan','herald','horde','outer_turret','inner_turret','base_turret','nexus_turret','inhibitor','plate']
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stable(x):return hashlib.sha256(x.encode()).hexdigest()
def progress(phase,**kw):dump(HERE/'PROGRESS.json',{'phase':phase,**kw});print(phase,kw,flush=True)
def weights(mask,match,patch):
    """Each active patch, then match, then supported query has equal mass."""
    mask=np.asarray(mask,bool);w=np.zeros(mask.shape,float)
    for j in range(mask.shape[1]):
        ps=np.unique(patch[mask[:,j]])
        for p in ps:
            ms=np.unique(match[(patch==p)&mask[:,j]])
            for m in ms:
                ix=(match==m)&(patch==p)&mask[:,j];w[ix,j]=1/(len(ps)*len(ms)*ix.sum())
    return w
def flat(x):return np.concatenate([x[k].reshape(len(x[k]),-1) for k in ('player','team','rift','relation','present')],axis=1).astype(np.float32)
def unflat(a,shapes):
    out={};start=0
    for k,s in shapes.items():n=int(np.prod(s));out[k]=a[:,start:start+n].reshape((-1,*s));start+=n
    assert start==a.shape[1];return out
def swap(x):
    order=np.r_[5:10,0:5]
    return {'player':x['player'][:,:,order], 'team':x['team'][:,:,::-1], 'rift':x['rift'].copy(), 'relation':x['relation'][:,:,order][:,:,:,order], 'present':x['present'].copy()}
def subset(x,ix):return {k:v[ix] for k,v in x.items()}

