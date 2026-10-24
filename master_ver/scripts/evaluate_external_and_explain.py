import run_validation_suite as s
import numpy as np
import pandas as pd
import json,joblib
from pathlib import Path
from train.temporal_winprob import random_minute_index

def explain_selected():
    z=np.load(s.OUT/'features.npz');r=pd.read_csv(s.OUT/'rows.csv',dtype={'patch':str});info=json.loads((s.OUT/'q_results.json').read_text());names=info['ridge_names'];fullnames=json.loads((s.OLD/'feature_names.json').read_text())['baseline'];xx=z['X'][:,[fullnames.index(n) for n in names]]
    pipe=joblib.load(s.OUT/'full_ridge.joblib');sc,lr=pipe.steps[0][1],pipe.steps[1][1];x=sc.transform(xx)
    train=(r.patch=='15.14').to_numpy();background=np.average(x[train],axis=0,weights=s.weights(r.loc[train,'match'].to_numpy()))
    intercept=float(lr.intercept_[0]+background@lr.coef_[0]);contrib=(x-background)*lr.coef_[0]
    # Shared affine sigmoid calibration maps raw log odds to calibrated log odds.
    cal=joblib.load(s.OUT/'ridge_sigmoid.joblib');a=float(cal.coef_[0,0]);b=float(cal.intercept_[0]);contrib*=a;expected=intercept*a+b
    prediction=cal.decision_function(pipe.decision_function(xx).reshape(-1,1));error=float(np.max(np.abs(contrib.sum(axis=1)+expected-prediction)));assert error<1e-8
    records=[]
    for patch in ['15.15','15.16']:
        mask=(r.patch==patch).to_numpy();w=s.weights(r.loc[mask,'match'].to_numpy());imp=np.average(np.abs(contrib[mask]),axis=0,weights=w)
        records.extend(dict(patch=patch,feature=n,mean_abs_shap=float(v)) for n,v in zip(names,imp))
    pd.DataFrame(records).to_csv(s.OUT/'selected_model_shap.csv',index=False)
    te=np.flatnonzero((r.patch=='15.16').to_numpy())[:100];np.savez_compressed(s.OUT/'selected_model_shap_cases.npz',row_indices=te,contributions=contrib[te],expected_value=expected,calibrated_log_odds=prediction[te])
    s.save('selected_model_shap.json',dict(model='ridge_sigmoid',method='Exact interventional linear Shapley values with independent-feature background convention',background='15.14 match-weighted means; correlations deliberately not conditionally modeled',scale='calibrated log-odds',max_additivity_error=error,expected_value=expected,calibrator_slope=a,reference='Lundberg and Lee 2017; linear additive model closed form',not_causal=True))

def external():
    ext=s.OUT/'external';s.CACHE=ext/'cache';prep=json.loads((ext/'preparation.json').read_text());assert prep['node_names']==s.NODE_FEATURE_NAMES
    labels=pd.read_csv(ext/'engagements.csv',dtype={'patch':str});valid=labels[labels.valid==1];names=json.loads((s.OLD/'feature_names.json').read_text())['baseline'];vm={k:joblib.load(s.VD/(k+'_model.joblib')) for k in ['expanded','maymin']}
    existing=json.loads((s.VD/'protocol.json').read_text())['splits'];mids={a['match'] for a in prep['matches']};assert all(not mids&set(v) for v in existing.values())
    # Extract with the same frozen StateBuilder used in 2025.
    records=[];vrows=[]
    for mid,rr in valid.groupby('match'):
        p=s.pack(mid);builder=s.StateBuilder(p,s.NODE_FEATURE_NAMES)
        records.extend(s.build_one((mid,rr.to_dict('records'))))
    # V evaluation includes all 100 matches, regardless of engagement detection.
    for audit in prep['matches']:
        mid=audit['match'];p=s.pack(mid);builder=s.StateBuilder(p,s.NODE_FEATURE_NAMES)
        times=np.arange(120000,int(p['minute_ts'][-1])+1,60000)
        t=int(times[random_minute_index(mid,len(times))]);st=builder.at(t)
        xx=np.array([[st.values[n] for n in names[:-1]]])
        vrows.append(dict(match=mid,time_ms=t,winner=audit['winner'],**{k:float(m.predict_proba(xx)[0,1]) for k,m in vm.items()}))
    r=pd.DataFrame([a for a,st,en,xy,sn in records]);X=np.array([[st[n] for n in names[:-1]] for a,st,en,xy,sn in records]);E=np.array([[en[n] for n in names[:-1]] for a,st,en,xy,sn in records]);C=np.stack([xy for a,st,en,xy,sn in records]);pp=vm['expanded'].predict_proba(X)[:,1];post=vm['expanded'].predict_proba(E)[:,1];X=np.column_stack([X,pp]).astype('float32');y=(post>pp).astype(int);r['y']=y;r['p_pre']=pp;r['p_post']=post
    info=json.loads((s.OUT/'q_results.json').read_text());ridge_ids=[names.index(n) for n in info['ridge_names']];econ_ids=[names.index(n) for n in info['economic_names']]
    oldp=pd.read_csv(s.OLD/'predictions.csv',dtype={'patch':str});tr=oldp[oldp.patch=='15.14'];prior=np.average(tr.y,weights=s.weights(tr.match.to_numpy()));preds={'constant':np.full(len(r),prior)}
    for key,xx in [('p_pre_logistic',pp.reshape(-1,1)),('p_pre_spline',pp.reshape(-1,1)),('full_ridge',X[:,ridge_ids])]:preds[key]=joblib.load(s.OUT/(key+'.joblib')).predict_proba(xx)[:,1]
    for key,xx,folder in [('full',X,s.OLD),('position',np.column_stack([X,C]),s.OLD),('economic',X[:,econ_ids],s.OUT)]:
        filename='baseline' if key=='full' else key
        preds[key+'_raw']=np.mean([joblib.load(folder/f'{filename}_{seed}.joblib').booster_.predict(xx,num_threads=4) for seed in [7,42,123]],axis=0)
    for key in ['full','position','economic','ridge']:
        pp0=preds['full_ridge' if key=='ridge' else key+'_raw'];preds[key+'_sigmoid']=joblib.load(s.OUT/(key+'_sigmoid.joblib')).predict_proba(s.logit(pp0))[:,1]
    w=s.weights(r.match.to_numpy());results={k:s.calibration(y,p,w) for k,p in preds.items()}
    for k,p in preds.items():r[k]=p
    r.to_csv(ext/'q_predictions.csv',index=False);v=pd.DataFrame(vrows);v.to_csv(ext/'v_predictions.csv',index=False)
    vscores={k:s.calibration(v.winner.to_numpy(),v[k].to_numpy(),np.ones(len(v))) for k in vm}
    chosen=info['chosen_by_development'];rng=np.random.default_rng(17);u,ix=np.unique(r.match,return_inverse=True);vals={k:[] for k in ['auc','brier','logloss']}
    for _ in range(500):
        c=np.bincount(rng.integers(len(u),size=len(u)),minlength=len(u));bw=w*c[ix];mask=bw>0
        a=s.score(y[mask],preds[chosen][mask],bw[mask]);b=s.score(y[mask],preds['constant'][mask],bw[mask])
        if a['auc'] is None:continue
        for k in vals:vals[k].append(a[k]-b[k])
    diff={k:dict(estimate=results[chosen][k]-results['constant'][k],ci95=np.quantile(vals[k],[.025,.975]).tolist()) for k in vals}
    outcome=dict(sample_matches=len(mids),valid_engagements=len(r),engagement_matches=int(r.match.nunique()),invalid_engagements=int((labels.valid!=1).sum()),value_partition_overlap=0,chosen_before_external=chosen,q=results,v=vscores,selected_minus_constant=diff,patch_api='16.13',patch_public='26.13',prior_use='100 matches previously sampled for replay/detector technical review; not an entirely untouched dataset',fitting_on_external=False)
    (ext/'results.json').write_text(json.dumps(outcome,indent=2),encoding='utf-8');print(json.dumps(outcome)[:2000],flush=True)

if __name__=='__main__':explain_selected();external()
