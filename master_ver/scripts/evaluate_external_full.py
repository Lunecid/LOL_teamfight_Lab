import run_validation_suite as s
from concurrent.futures import ThreadPoolExecutor
from train.temporal_winprob import random_minute_index
import numpy as np,pandas as pd,json,joblib,time
from sklearn.linear_model import LogisticRegression
OUT=s.OUT/'external_full'

def build(task):
    a,refs=task;mid=a['match'];p=s.pack(mid);b=s.StateBuilder(p,s.NODE_FEATURE_NAMES);out=[]
    for r in refs:out.append((r,b.at(int(r['s'])-1).values,b.at(int(r['endpoint'])).values))
    ts=np.arange(120000,int(p['minute_ts'][-1])+1,60000)
    v=None
    if len(ts):
        t=int(ts[random_minute_index(mid,len(ts))]);v=(dict(match=mid,time_ms=t,winner=a['winner'],external_split=a['external_split']),b.at(t).values)
    return out,v

def paired(y,p,base,g):
    w=s.weights(g);u,ix=np.unique(g,return_inverse=True);rng=np.random.default_rng(17);vals={k:[] for k in ['auc','brier','logloss']}
    for _ in range(500):
        count=np.bincount(rng.integers(len(u),size=len(u)),minlength=len(u));bw=w*count[ix];keep=bw>0
        a=s.score(y[keep],p[keep],bw[keep]);b=s.score(y[keep],base[keep],bw[keep])
        for k in vals:vals[k].append(a[k]-b[k])
    a=s.score(y,p,w);b=s.score(y,base,w)
    return {k:dict(estimate=a[k]-b[k],ci95=np.quantile(vals[k],[.025,.975]).tolist()) for k in vals}

def main():
    s.CACHE=OUT/'cache';prep=json.loads((OUT/'preparation.json').read_text());assert prep['node_names']==s.NODE_FEATURE_NAMES
    assert not prep['errors'],prep['errors'][:3]
    data=pd.read_csv(OUT/'engagements.csv',dtype={'patch':str});valid=data[data.valid==1];lookup={m:a.to_dict('records') for m,a in valid.groupby('match')}
    old=json.loads((s.VD/'protocol.json').read_text())['splits'];mids={a['match'] for a in prep['matches']};assert all(not mids&set(v) for v in old.values())
    cutoff=min(a['game_creation'] for a in prep['matches'] if a['external_split']=='evaluation_late')
    purged={a['match'] for a in prep['matches'] if a['external_split']=='adaptation_early' and a['game_creation']+a['last_snapshot']+60000>=cutoff}
    records=[];vrecords=[];start=time.time()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,(r,v) in enumerate(pool.map(build,[(a,lookup.get(a['match'],[])) for a in prep['matches']]),1):
            records.extend(r)
            if v:vrecords.append(v)
            if i%2000==0:print('external evaluation states',i,round(time.time()-start,1),flush=True)
    names=json.loads((s.OLD/'feature_names.json').read_text())['baseline'];qinfo=json.loads((s.OUT/'q_results.json').read_text());assert qinfo['chosen_by_development']=='ridge_sigmoid'
    model=joblib.load(s.VD/'expanded_model.joblib');r=pd.DataFrame([a for a,st,en in records]);X=np.array([[st[n] for n in names[:-1]] for a,st,en in records]);E=np.array([[en[n] for n in names[:-1]] for a,st,en in records])
    pre=model.predict_proba(X)[:,1];post=model.predict_proba(E)[:,1];r['p_pre']=pre;r['p_post']=post;r['y']=(post>pre).astype(int);X=np.column_stack([X,pre]).astype('float32')
    ridge=joblib.load(s.OUT/'full_ridge.joblib');raw=ridge.predict_proba(X[:,[names.index(n) for n in qinfo['ridge_names']]])[:,1];cal=joblib.load(s.OUT/'ridge_sigmoid.joblib');r['frozen']=cal.predict_proba(s.logit(raw))[:,1]
    ca=(r.external_split=='adaptation_early')&~r.match.isin(purged);te=r.external_split=='evaluation_late';qcal=LogisticRegression(C=1e6,max_iter=1000).fit(s.logit(r.loc[ca,'frozen']),r.loc[ca,'y'],sample_weight=s.weights(r.loc[ca,'match'].to_numpy()));r['adapted']=qcal.predict_proba(s.logit(r.frozen))[:,1];joblib.dump(qcal,OUT/'q_early_sigmoid.joblib')
    orig=pd.read_csv(s.OLD/'predictions.csv',dtype={'patch':str});tr=orig[orig.patch=='15.14'];prior=np.average(tr.y,weights=s.weights(tr.match.to_numpy()));r['constant_2025']=prior;r['constant_early2026']=np.average(r.loc[ca,'y'],weights=s.weights(r.loc[ca,'match'].to_numpy()))
    r.to_csv(OUT/'q_predictions.csv',index=False)
    vr=pd.DataFrame([a for a,st in vrecords]);vx=np.array([[st[n] for n in names[:-1]] for a,st in vrecords]);vr['frozen']=model.predict_proba(vx)[:,1];vc=(vr.external_split=='adaptation_early')&~vr.match.isin(purged);vt=vr.external_split=='evaluation_late';vcal=LogisticRegression(C=1e6,max_iter=1000).fit(s.logit(vr.loc[vc,'frozen']),vr.loc[vc,'winner']);vr['adapted']=vcal.predict_proba(s.logit(vr.frozen))[:,1];joblib.dump(vcal,OUT/'v_early_sigmoid.joblib');vr.to_csv(OUT/'v_predictions.csv',index=False)
    qmetrics={k:s.calibration(r.loc[te,'y'].to_numpy(),r.loc[te,k].to_numpy(),s.weights(r.loc[te,'match'].to_numpy())) for k in ['frozen','adapted','constant_2025','constant_early2026']}
    vmetrics={k:s.calibration(vr.loc[vt,'winner'].to_numpy(),vr.loc[vt,k].to_numpy(),np.ones(int(vt.sum()))) for k in ['frozen','adapted']}
    pairs={key:paired(r.loc[te,'y'].to_numpy(),r.loc[te,a].to_numpy(),r.loc[te,b].to_numpy(),r.loc[te,'match'].to_numpy()) for key,a,b in [('frozen_vs_constant','frozen','constant_2025'),('adapted_vs_frozen','adapted','frozen')]}
    vpair=paired(vr.loc[vt,'winner'].to_numpy(),vr.loc[vt,'adapted'].to_numpy(),vr.loc[vt,'frozen'].to_numpy(),vr.loc[vt,'match'].to_numpy())
    result=dict(total_matches=len(mids),valid_engagements=len(r),invalid_engagements=int((data.valid!=1).sum()),short_matches_without_V_query=len(mids)-len(vr),purged_adaptation_matches=sorted(purged),q_adaptation_matches=int(r.loc[ca,'match'].nunique()),q_evaluation_matches=int(r.loc[te,'match'].nunique()),q_evaluation_rows=int(te.sum()),v_adaptation_matches=int(vc.sum()),v_evaluation_matches=int(vt.sum()),q=qmetrics,v=vmetrics,q_paired=pairs,v_adapted_vs_frozen=vpair,model='ridge_sigmoid frozen before external expansion',labels='Frozen original expanded V, unmodified by external adaptation',source_overlap=0,cutoff_creation_ms=int(cutoff),calibrator_parameters=dict(q_slope=float(qcal.coef_[0,0]),q_intercept=float(qcal.intercept_[0]),v_slope=float(vcal.coef_[0,0]),v_intercept=float(vcal.intercept_[0])))
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('full external evaluation complete',result['q_evaluation_rows'],result['q_evaluation_matches'],flush=True)

if __name__=='__main__':main()
