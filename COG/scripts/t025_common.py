"""Isolated thesis follow-up helpers. No existing artifacts are written."""
import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
REPO=Path(__file__).resolve().parents[1]
DATA=Path(r'C:\Users\todtj\문서\LOL_Teamfight')
OUT=REPO/'outputs/thesis_followup_20260923'
OUT.mkdir(parents=True,exist_ok=True)
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(k,'2')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
os.environ['LOL_OUTPUT_ROOT']=str(OUT/'runtime')
sys.path[:0]=[str(DATA/'worktrees/engagement-state-value'),str(DATA/'scripts'),str(REPO/'scripts'),str(REPO)]

def dump(name,obj):
    def conv(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=conv)+'\n',encoding='utf-8')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def weights(g):
    _,idx,n=np.unique(g,return_inverse=True,return_counts=True)
    return 1/n[idx]

def scores(y,p,g):
    from sklearn.metrics import roc_auc_score
    w=weights(g);p=np.asarray(p,float);y=np.asarray(y,float)
    return dict(n=len(y),matches=len(np.unique(g)),brier=float(np.average((y-p)**2,weights=w)),
                auc=float(roc_auc_score(y,p,sample_weight=w)) if len(np.unique(y))==2 else None)

def contrast(y,a,b,g,reps=2000):
    if len(y)==0:return dict(n=0,status='EMPTY')
    u,i,n=np.unique(g,return_inverse=True,return_counts=True)
    dm=np.bincount(i,weights=(np.asarray(a)-y)**2-(np.asarray(b)-y)**2)/n
    rng=np.random.default_rng(7)
    boot=np.array([dm[rng.integers(0,len(u),len(u))].mean() for _ in range(reps)])
    return dict(n=len(y),matches=len(u),estimate=float(dm.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),reps=reps,seed=7)

def lab_dir(cohort):
    return REPO/('outputs/q_newv_fit85_20260920'+('_S' if cohort=='S' else ''))/'labels'

def pred_path(cohort):
    return REPO/('outputs/review_response_rr12_20260920'+('_S_qS_id' if cohort=='S' else ''))/'prediction_table.npz'

def load_packs(cohort):
    """Single extraction scan, pre-state only; strict key coverage and order checks."""
    import fc20260915_data as D
    packs={}
    for role,file in [('TR','TRAIN_oof_h90'),('CA','Q_CAL_h90'),('SE','Q_SELECT_h90'),('TE','TEST_h90')]:
        with np.load(lab_dir(cohort)/(file+'.npz'),allow_pickle=False) as z:
            ok=z['Y_SVI']>=0
            if 'missing_score' in z.files:ok &= z['missing_score']==0
            g=z['match'][ok].astype(str);s=z['s'][ok].astype(np.int64)
            keys={(a,int(b)):i for i,(a,b) in enumerate(zip(g,s))}
            assert len(keys)==len(g),'duplicate label keys'
            packs[role]=dict(g=g,s=s,p_pre=z['p_pre'][ok].astype(float),y=z['Y_SVI'][ok],keys=keys,seen=np.zeros(len(g),int))
    names=None
    for ci,(sp,_,_) in enumerate(D.chunk_paths(D.Layout(False),'MAIN')):
        with np.load(sp,allow_pickle=False) as z:
            nm=z['names'].tolist()
            if names is None:
                names=nm
                for p in packs.values():p['X']=np.empty((len(p['y']),len(names)),dtype=np.float64)
            assert names==nm
            keys=list(zip(z['e_match'].astype(str),map(int,z['e_s'])))
            selections={}
            for role,p in packs.items():
                sel=[(j,p['keys'][k]) for j,k in enumerate(keys) if k in p['keys']]
                if sel:selections[role]=np.asarray(sel,int)
            if selections:
                X=z['e_X_pre']
                for role,ix in selections.items():
                    p=packs[role];p['X'][ix[:,1]]=X[ix[:,0]];p['seen'][ix[:,1]]+=1
        if ci%10==0:print('read extraction chunks',ci,flush=True)
    keep=[i for i,n in enumerate(names) if n!='snapshot_age_s']
    cols=[names[i] for i in keep]
    for p in packs.values():
        assert np.all(p['seen']==1),'missing or duplicate extraction keys'
        p['X']=p['X'][:,keep];p['cols']=cols
        del p['keys'],p['seen']
    return packs,cols

def provenance():
    return dict(task='T025',generated=datetime.now(timezone.utc).isoformat(),epistemic='exploratory_after_TEST_exposure')
