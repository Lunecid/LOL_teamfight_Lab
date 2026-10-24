"""Synthetic reference checks; no dataset fitting or production certification."""
import json, hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
doc=json.loads((ROOT/'MULTIWINDOW_CONTRACT.json').read_text(encoding='utf-8'))
checks=[]
def check(name,condition):
    checks.append({'name':name,'passed':bool(condition)})

def cumulative_terminal(hazards):
    h=np.asarray(hazards,dtype=float)
    survival=1.;blue=0.;red=0.;out=[]
    for ongoing,b,r in h:
        blue+=survival*b
        red+=survival*r
        survival*=ongoing
        out.append([survival,blue,red])
    return np.array(out)

def resolve_endpoint(frames,nominal,end=None,terminal_frame_verified=False,tol=5000):
    frames=np.asarray(frames)
    if end is not None and end<=nominal:
        return int(end) if terminal_frame_verified and end in frames else None
    at=np.searchsorted(frames,nominal,side='left')
    if at==len(frames) or frames[at]-nominal>tol:
        return None
    return int(frames[at])

check('base18 order and count preserved',len(doc['blocks'][0]['ordered_fields'])==18)
check('5 L models share three horizons',doc['model']['L_grid']==[1,2,3,4,5] and doc['time']['horizons_seconds']==[60,120,180])
check('14 count fields',len(doc['targets']['counts']['ordered_fields'])==14)
check('no production ready claim',not doc['production_training_ready'])
check('legacy preserved',not doc['replaces_parent'])
check('U disabled',doc['change_log']['U']=='disabled')
check('future cutoff fields forbidden',{'tau_ms','first_kill_ms','last_kill_ms','end_ms','winning_team','target_support'}<=set(doc['feature_controls']['forbidden']))
h=np.array([[.7,.2,.1],[.6,.1,.3],[.5,.4,.1]])
p=cumulative_terminal(h)
check('terminal probabilities sum to one',np.allclose(p.sum(1),1))
check('terminal cumulative monotonicity',np.all(np.diff(p[:,1:],axis=0)>=0))
check('survival decreases',np.all(np.diff(p[:,0])<=0))
check('team swap commutes',np.allclose(cumulative_terminal(h[:,[0,2,1]]),p[:,[0,2,1]]))
check('certain terminal remains absorbed',np.allclose(cumulative_terminal([[0,1,0],[.1,.2,.7]])[:,1],[1,1]))
check('actual timestamp alignment',resolve_endpoint([0,60017,120034],60000)==60017)
check('missing future frame stays unknown',resolve_endpoint([0,60017],120000) is None)
check('endpoint outside tolerance rejected',resolve_endpoint([0,70000],60000) is None)
check('verified terminal can absorb',resolve_endpoint([0,60017,90000],120000,90000,True)==90000)
check('missing terminal gold not last-observation zero',resolve_endpoint([0,60017],120000,90000,False) is None)
g=np.array([[100,80],[30,60],[10,20]])
check('gold team difference telescopes',np.array_equal(np.cumsum(g,axis=0)[:,0]-np.cumsum(g,axis=0)[:,1],np.cumsum(g[:,0]-g[:,1])))
check('nonadditive statuses excluded from additive heads',not {'buff_active','soul_owner','terminal'}&set(doc['targets']['counts']['ordered_fields']))
result={'checks':checks,'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),
        'scope':'synthetic reference math and contract checks only','production_validated':False,'models_trained':0}
(ROOT/'CONTRACT_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
manifest={'scope':'audit and implementation-proposal files', 'files':[]}
for pth in sorted(ROOT.iterdir()):
    if pth.is_file() and pth.name!='ARTIFACT_MANIFEST.json':
        manifest['files'].append({'path':pth.name,'bytes':pth.stat().st_size,'sha256':hashlib.sha256(pth.read_bytes()).hexdigest()})
(ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(f"{result['passed']} reference checks passed; {result['failed']} failed. Production gates remain pending.")
raise SystemExit(1 if result['failed'] else 0)
