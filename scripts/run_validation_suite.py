"""Frozen-data audit and development-only q calibration. See BEFORE protocol."""
import os, sys, json, hashlib, time, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
WT=ROOT/'worktrees/engagement-state-value'; VD=WT/'outputs/temporal_winprob_v3_buckets'
CACHE=Path('D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13')
OUT=ROOT/'outputs/validation_suite_20260914'; OLD=ROOT/'outputs/position_ablation_20260914'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(WT));sys.dont_write_bytecode=True
os.environ['LOL_OUTPUT_ROOT']=str(OUT/'runtime')
from core.config import NODE_FEATURE_NAMES
from gameplay.state_value import StateBuilder
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
import joblib

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def weights(g):
    _,ix,c=np.unique(g,return_inverse=True,return_counts=True)
    w=1/c[ix];return w/w.mean()
def score(y,p,w):
    return dict(auc=float(roc_auc_score(y,p,sample_weight=w)) if len(np.unique(y))>1 else None,
                brier=float(brier_score_loss(y,p,sample_weight=w)),logloss=float(log_loss(y,p,labels=[0,1],sample_weight=w)))
def logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-8,1-1e-8);return np.log(p/(1-p)).reshape(-1,1)
def calibration(y,p,w):
    out=score(y,p,w);out.update(n=len(y),observed=float(np.average(y,weights=w)),predicted=float(np.average(p,weights=w)))
    if len(np.unique(y))>1 and np.std(p)>1e-8:
        m=LogisticRegression(C=1e6,max_iter=1000).fit(logit(p),y,sample_weight=w)
        out.update(intercept=float(m.intercept_[0]),slope=float(m.coef_[0,0]))
    bins=[]
    for lo in np.arange(0,1,.1):
        a=(p>=lo)&(p<lo+.1 if lo<.85 else p<=1)
        if a.any():bins.append(dict(lo=float(lo),n=int(a.sum()),predicted=float(np.average(p[a],weights=w[a])),observed=float(np.average(y[a],weights=w[a]))))
    out['bins']=bins;return out
def pack(mid):
    with np.load(CACHE/(mid+'.npz'),allow_pickle=False) as z:p={k:z[k] for k in z.files}
    p['meta']=json.loads((CACHE/(mid+'.meta.json')).read_text());p['events']=json.loads((CACHE/(mid+'.events.json')).read_text())
    return p
def build_one(task):
    mid,rows=task;p=pack(mid);b=StateBuilder(p,NODE_FEATURE_NAMES);out=[]
    order=sorted(b.tm,key=b.slots.get)
    for r in rows:
        st=b.at(int(r['s'])-1);end=b.at(int(r['endpoint']))
        idx=np.searchsorted(b.ts,int(r['s'])-1,side='right')-1
        xy=p['xy_raw_minute'][idx,np.array(order)-1,:].astype(float).reshape(-1)/15000
        out.append((r,st.values,end.values,xy,st.snapshot_ms))
    return out
def features():
    if (OUT/'features.npz').exists():return
    labels=pd.read_csv(ROOT/'outputs/b_boundary_60_90_120/dynamic_values.csv')
    labels=labels[(labels.h==90)&(labels.valid==1)][['match','s','endpoint','p_post','duration_s']]
    rows=pd.read_csv(OLD/'predictions.csv',dtype={'patch':str}).merge(labels,on=['match','s'],validate='one_to_one')
    records=[];start=time.time()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,a in enumerate(pool.map(build_one,[(m,g.to_dict('records')) for m,g in rows.groupby('match')]),1):
            records.extend(a)
            if i%1500==0:print('features',i,round(time.time()-start,1),flush=True)
    names=json.loads((OLD/'feature_names.json').read_text())['baseline'];state_names=names[:-1]
    X=np.array([[st[n] for n in state_names]+[r['p_pre']] for r,st,en,xy,sn in records],dtype=np.float64)
    E=np.array([[en[n] for n in state_names] for r,st,en,xy,sn in records],dtype=np.float64)
    C=np.stack([xy for r,st,en,xy,sn in records])
    meta=pd.DataFrame([r for r,st,en,xy,sn in records]);meta.to_csv(OUT/'rows.csv',index=False)
    assert np.isfinite(X).all() and np.isfinite(E).all() and np.isfinite(C).all()
    assert all(sn<int(r['s']) for r,st,en,xy,sn in records)
    models={k:joblib.load(VD/(k+'_model.joblib')) for k in ['expanded','maymin']}
    pp={}
    for k,m in models.items():
        assert m.state_names==state_names
        pp[k+'_pre']=m.predict_proba(X[:,:-1])[:,1];pp[k+'_post']=m.predict_proba(E)[:,1]
    checks={k:float(np.max(np.abs(pp['expanded_'+k]-meta[col]))) for k,col in [('pre','p_pre'),('post','p_post')]}
    assert max(checks.values())<1e-10,checks
    np.savez_compressed(OUT/'features.npz',X=X.astype('float32'),E=E.astype('float32'),C=C.astype('float32'),**pp)
    save('feature_checks.json',dict(value_reproduction=checks,rows=len(meta),future_violations=0,names=names))
    print('feature extraction complete',flush=True)

def q_experiment():
    from lightgbm import LGBMClassifier
    z=np.load(OUT/'features.npz');X=z['X'];C=z['C'];r=pd.read_csv(OUT/'rows.csv',dtype={'patch':str});names=json.loads((OLD/'feature_names.json').read_text())['baseline']
    y=r.y.to_numpy();g=r.match.to_numpy();tr=(r.patch=='15.14').to_numpy();va=(r.patch=='15.15').to_numpy();te=(r.patch=='15.16').to_numpy();w=weights(g[tr]);vw=weights(g[va]);tw=weights(g[te])
    # Deterministic disjoint development halves: calibrator fitting vs selection.
    half=np.array([int(hashlib.sha256(('calibration17:'+m).encode()).hexdigest()[:8],16)%2 for m in g])
    ca=va&(half==0);sel=va&(half==1);cw=weights(g[ca])
    preds={'constant':np.full(len(r),np.average(y[tr],weights=w)), 'full_raw':r.baseline.to_numpy(),'position_raw':r.position.to_numpy()}; fitted={}
    for key,xx,m in [
        ('p_pre_logistic',r[['p_pre']].to_numpy(),make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000))),
        ('p_pre_spline',r[['p_pre']].to_numpy(),make_pipeline(SplineTransformer(n_knots=5,degree=3),StandardScaler(),LogisticRegression(C=1,max_iter=2000)))]:
        m.fit(xx[tr],y[tr],logisticregression__sample_weight=w);preds[key]=m.predict_proba(xx)[:,1];fitted[key]=m
    econ=[i for i,n in enumerate(names) if n in ['time_minutes','snapshot_age_s'] or (n.startswith('slot') and any(n.endswith(a) for a in ['totalGold_norm','curGold_norm','level_norm','xp_norm','alive','laneCS_norm','jgCS_norm','kills','deaths'])) or ((n.startswith('blue_') or n.startswith('red_')) and not n.endswith('_x_time') and 'age_' not in n and 'last_' not in n and not n.endswith('_ever'))]
    numeric=[i for i,n in enumerate(names) if not n.endswith('champion_id')]
    m=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=3000))
    m.fit(X[tr][:,numeric],y[tr],logisticregression__sample_weight=w);preds['full_ridge']=m.predict_proba(X[:,numeric])[:,1];fitted['full_ridge']=m
    ps=[]
    for seed in [7,42,123]:
        m=LGBMClassifier(n_estimators=250,num_leaves=15,learning_rate=.04,min_child_samples=100,reg_lambda=1,random_state=seed,n_jobs=4,verbosity=-1,colsample_bytree=.9)
        m.fit(X[tr][:,econ],y[tr],sample_weight=w);ps.append(m.predict_proba(X[:,econ])[:,1]);joblib.dump(m,OUT/f'economic_{seed}.joblib')
    preds['economic_raw']=np.mean(ps,axis=0)
    for base in ['full','position','economic','ridge']:
        p=preds['full_ridge' if base=='ridge' else base+'_raw'];sg=LogisticRegression(C=1e6,max_iter=1000).fit(logit(p[ca]),y[ca],sample_weight=cw)
        iso=IsotonicRegression(out_of_bounds='clip').fit(p[ca],y[ca],sample_weight=cw)
        preds[base+'_sigmoid']=sg.predict_proba(logit(p))[:,1];preds[base+'_isotonic']=iso.predict(p)
        fitted[base+'_sigmoid']=sg;fitted[base+'_isotonic']=iso
    metrics=[]
    for key,p in preds.items():
        for split,mask in [('train',tr),('calibration_fit',ca),('development',sel),('test_exploratory',te)]:
            metrics.append(dict(model=key,split=split,**calibration(y[mask],p[mask],weights(g[mask]))))
    dev=[a for a in metrics if a['split']=='development'];chosen=min(dev,key=lambda a:(a['brier'],a['logloss']))['model']
    selected={base:min([a for a in dev if a['model'].startswith(base+'_') and a['model'] in [base+'_raw',base+'_sigmoid',base+'_isotonic']],key=lambda a:(a['brier'],a['logloss']))['model'] for base in ['full','position','economic']}
    for k,m in fitted.items():joblib.dump(m,OUT/(k+'.joblib'))
    pred=r.copy()
    for k,p in preds.items():pred[k]=p
    pred.to_csv(OUT/'q_predictions.csv',index=False)
    bootpairs=[(chosen,'constant'),(chosen,'full_raw'),(selected['full'],'full_raw'),(selected['position'],selected['full'])]
    u,ix=np.unique(g[te],return_inverse=True);rng=np.random.default_rng(17);samples={str(i):{k:[] for k in ['auc','brier','logloss']} for i in range(len(bootpairs))}
    for _ in range(500):
        counts=np.bincount(rng.integers(len(u),size=len(u)),minlength=len(u));bw=tw*counts[ix];keep=bw>0
        ms={k:score(y[te][keep],preds[k][te][keep],bw[keep]) for k in set(sum([list(a) for a in bootpairs],[]))}
        for i,(a,b) in enumerate(bootpairs):
            for k in samples[str(i)]:samples[str(i)][k].append(ms[a][k]-ms[b][k])
    pairs=[]
    for i,(a,b) in enumerate(bootpairs):
        aa=score(y[te],preds[a][te],tw);bb=score(y[te],preds[b][te],tw)
        pairs.append(dict(a=a,b=b,delta={k:dict(estimate=aa[k]-bb[k],ci95=np.quantile(samples[str(i)][k],[.025,.975]).tolist()) for k in aa}))
    save('q_results.json',dict(chosen_by_development=chosen,selected_calibrations=selected,development_partition={key:dict(rows=int(mask.sum()),matches=len(set(g[mask]))) for key,mask in [('calibration_fit',ca),('selection',sel)]},metrics=metrics,paired_bootstrap=pairs,economic_names=[names[i] for i in econ],ridge_names=[names[i] for i in numeric]))
    print('q complete',chosen,selected,flush=True)

def objective_rows(task):
    mid,times=task;ev=json.loads((CACHE/(mid+'.events.json')).read_text());out={'match':mid,'patch':str(json.loads((CACHE/(mid+'.meta.json')).read_text()).get('patch','unknown'))}
    d={k:[] for k in ['baron','elder','herald','horde','atakhan','dragon','soul_team_recorded','soul_unassigned_event']+[f'dragon_{k}' for k in ['AIR','EARTH','FIRE','WATER','HEXTECH','CHEMTECH','OTHER']]}
    for e in ev:
        if e.get('type')=='DRAGON_SOUL_GIVEN':
            key='soul_team_recorded' if int(e.get('teamId',0) or 0) in [100,200] else 'soul_unassigned_event'
            d[key].append(int(e['timestamp']))
        if e.get('type')!='ELITE_MONSTER_KILL':continue
        monster=e.get('monsterType','');sub=e.get('monsterSubType','')
        key={'BARON_NASHOR':'baron','RIFTHERALD':'herald','HORDE':'horde','ATAKHAN':'atakhan','DRAGON':'elder' if sub=='ELDER_DRAGON' else 'dragon'}.get(monster)
        if key:d[key].append(int(e['timestamp']))
        if key=='dragon':
            kind=sub.replace('_DRAGON','');kind=kind if 'dragon_'+kind in d else 'OTHER';d['dragon_'+kind].append(int(e['timestamp']))
    return pd.DataFrame([dict(**out,time_ms=int(t),**{k:int(np.searchsorted(sorted(ts),t,side='right')) for k,ts in d.items()}) for t in times])
def value_audit():
    z=np.load(VD/'independent_time_curves.npz');r=pd.DataFrame({k:z[k] for k in z.files})
    protocol=json.loads((VD/'protocol.json').read_text());assert set(r.match)==set(protocol['splits']['test'])
    for k in ['fit','calibrate','select','engagement']:assert not set(r.match)&set(protocol['splits'][k])
    with ThreadPoolExecutor(max_workers=6) as pool:
        obj=pd.concat(pool.map(objective_rows,[(m,g.time_ms.to_numpy()) for m,g in r.groupby('match')]),ignore_index=True)
    r=r.merge(obj,on=['match','time_ms'],validate='one_to_one');r.to_csv(OUT/'v_audit_rows.csv',index=False)
    sampled=pd.DataFrame(json.loads((VD/'sampled_minutes.json').read_text()));sampled=sampled[sampled.role=='test'][['match','time_ms']]
    one=sampled.merge(r,on=['match','time_ms'],validate='one_to_one');assert len(one)==4986
    results=[]
    strata=[('original_one_per_match',one),('all_times_equal_match',r)]
    for lo,hi in [(2,10),(10,20),(20,30),(30,1000)]:strata.append((f'minutes_{lo}_{hi}',r[(r.time_ms>=lo*60000)&(r.time_ms<hi*60000)]))
    for k in ['baron','elder','herald','horde','atakhan','dragon','soul_team_recorded','soul_unassigned_event']+[f'dragon_{k}' for k in ['AIR','EARTH','FIRE','WATER','HEXTECH','CHEMTECH','OTHER']]:
        for has in [False,True]:strata.append((k+('_acquired' if has else '_not_acquired'),r[(r[k]>0)==has]))
    for name,a in strata:
        if len(a)==0:results.append(dict(stratum=name,n=0));continue
        for model in ['expanded','maymin']:
            # Equal-match Brier equals the average within-match squared error.
            loss=(a[model]-a.winner)**2;means=loss.groupby(a.match).mean().to_numpy();rng=np.random.default_rng(17)
            boot=[float(np.mean(means[rng.integers(len(means),size=len(means))])) for _ in range(500)]
            results.append(dict(stratum=name,model=model,matches=int(a.match.nunique()),brier_ci95=np.quantile(boot,[.025,.975]).tolist(),**calibration(a.winner.to_numpy(),a[model].to_numpy(),weights(a.match.to_numpy()))))
    old=json.loads((VD/'results.json').read_text())['test_one_minute'];checks={}
    for k in ['expanded','maymin']:
        s=score(one.winner,one[k],np.ones(len(one)));checks[k]={n:abs(s[n]-old[k]['log_loss' if n=='logloss' else n]) for n in s}
    assert max(v for a in checks.values() for v in a.values())<1e-10
    save('v_results.json',dict(reproduction=checks,results=results,patches=r.groupby('patch').match.nunique().to_dict(),note='Acquisition history at or before query, not active buff or causal effect; subgroup weights normalized within subgroup.'))
    print('V audit complete',flush=True)

def label_audit():
    z=np.load(OUT/'features.npz');r=pd.read_csv(OUT/'rows.csv',dtype={'patch':str});names=json.loads((OLD/'feature_names.json').read_text())['baseline'];X=z['X'];E=z['E']
    r['delta']=z['expanded_post']-z['expanded_pre'];r['maymin_delta']=z['maymin_post']-z['maymin_pre'];r['value_disagreement']=(r.delta>0)!=(r.maymin_delta>0)
    for key,cols in [('gold',[n for n in names if n.endswith('totalGold_norm')]),('kills',['blue_kills','red_kills'])]:
        if key=='gold':blue=cols[:5];red=cols[5:]
        else:blue=cols[:1];red=cols[1:]
        ids=lambda ns:[names.index(n) for n in ns]
        r[key+'_balance_change']=(E[:,ids(blue)].sum(1)-E[:,ids(red)].sum(1))-(X[:,ids(blue)].sum(1)-X[:,ids(red)].sum(1))
    r['gold_disagreement']=(r.gold_balance_change!=0)&((r.gold_balance_change>0)!=(r.delta>0))
    r['kills_disagreement']=(r.kills_balance_change!=0)&((r.kills_balance_change>0)!=(r.delta>0))
    for k in ['baron','elder','herald','horde','atakhan','dragons']:
        cols=[names.index(t+'_'+k) for t in ['blue','red']];r[k+'_acquired_in_interval']=(E[:,cols]-X[:,cols]).sum(1)>0
    r.to_csv(OUT/'label_audit_rows.csv',index=False)
    summary={k:float(r[k].mean()) for k in ['value_disagreement','gold_disagreement','kills_disagreement']}
    summary['small_delta']={str(t):dict(n=int((r.delta.abs()<=t).sum()),fraction=float((r.delta.abs()<=t).mean())) for t in [0,.001,.005,.01,.02]}
    summary['by_patch']={str(k):dict(n=len(a),value_disagreement=float(a.value_disagreement.mean())) for k,a in r.groupby('patch')}
    summary['by_objective']={k:dict(n=int(r[k+'_acquired_in_interval'].sum()),value_disagreement=float(r.loc[r[k+'_acquired_in_interval'],'value_disagreement'].mean()) if r[k+'_acquired_in_interval'].any() else None) for k in ['baron','elder','herald','horde','atakhan','dragons']}
    selected=[]
    for key,mask in [('random',np.ones(len(r),dtype=bool)),('value_disagreement',r.value_disagreement),('gold_disagreement',r.gold_disagreement),('kills_disagreement',r.kills_disagreement)]+[(k,r[k+'_acquired_in_interval']) for k in ['baron','elder','herald','horde','atakhan','dragons']]:
        a=r[mask].sample(n=min(12,int(np.sum(mask))),random_state=17).copy();a['sampling_stratum']=key;selected.append(a)
    cases=pd.concat(selected).drop_duplicates(['match','s']).reset_index(drop=True);cases['case_id']=['case_%03d'%i for i in range(len(cases))];cases.to_csv(OUT/'review_answer_key.csv',index=False)
    blind=cases[['case_id','match','s','endpoint']].copy()
    for k in ['reviewer_id','blue_advantage_change','boundary_appropriate','continuation_or_new_fight','confidence','reason']:blind[k]=''
    blind.to_csv(OUT/'review_blinded_form.csv',index=False)
    # Event-only records omit model predictions, future final outcome and selection reason.
    streams=[]
    for mid,a in cases.groupby('match'):
        ev=json.loads((CACHE/(mid+'.events.json')).read_text())
        for c in a.to_dict('records'):
            streams.append(dict(case_id=c['case_id'],events=[e for e in ev if c['s']-15000<=int(e.get('timestamp',-1))<=c['endpoint']+15000 and e.get('type') in ['CHAMPION_KILL','ELITE_MONSTER_KILL','BUILDING_KILL','DRAGON_SOUL_GIVEN']]))
    save('review_blinded_events.json',streams);summary['review_cases']=len(cases);summary['human_review_completed']=False
    ex=pd.read_csv(ROOT/'outputs/postkill_objective_delay_full/exposures.csv',usecols=['match','patch'],dtype={'patch':str}).drop_duplicates()
    summary['available_engagement_patches']=ex.groupby('patch').match.nunique().to_dict()
    save('label_results.json',summary);print('label audit complete',flush=True)

def shap_audit():
    from scipy.stats import spearmanr
    z=np.load(OUT/'features.npz');r=pd.read_csv(OUT/'rows.csv',dtype={'patch':str});names=json.loads((OLD/'feature_names.json').read_text())['baseline'];X=z['X']
    def group(n):
        if n=='p_pre':return 'prior_win_probability'
        if n.startswith('time_') or n=='snapshot_age_s':return 'time_and_observation_age'
        if any(k in n for k in ['baron','elder','dragon','soul','herald','horde','atakhan']):return 'objectives'
        if any(k in n for k in ['Gold','xp_','level_','CS_']):return 'economy_and_experience'
        if any(k in n for k in ['kills','deaths','death_','alive']):return 'combat_and_survival'
        if any(k in n for k in ['tower','inhibitor','plates']):return 'structures'
        if 'champion_id' in n:return 'champion_identity'
        return 'health_mana_other'
    groups=np.array([group(n) for n in names]);records=[];vectors={};errors=[];top=[]
    for patch in ['15.15','15.16']:
        mask=(r.patch==patch).to_numpy();xx=X[mask];w=weights(r.loc[mask,'match'].to_numpy())
        for seed in [7,42,123]:
            m=joblib.load(OLD/f'baseline_{seed}.joblib');values=m.booster_.predict(xx,pred_contrib=True,num_threads=4)
            margin=m.booster_.predict(xx,raw_score=True,num_threads=4);err=float(np.max(np.abs(values.sum(axis=1)-margin)));assert err<1e-8;errors.append(err)
            imp=np.average(np.abs(values[:,:-1]),axis=0,weights=w);vectors[f'{patch}_{seed}']=imp
            for n,v in zip(names,imp):records.append(dict(patch=patch,seed=seed,feature=n,group=group(n),mean_abs_shap=float(v)))
            for gr in sorted(set(groups)):
                # Sum signed contributions per row before absolute value.
                top.append(dict(patch=patch,seed=seed,group=gr,mean_abs_group_shap=float(np.average(np.abs(values[:,:-1][:,groups==gr].sum(axis=1)),weights=w))))
            if patch=='15.16' and seed==7:np.savez_compressed(OUT/'shap_cases.npz',contributions=values[:100],row_indices=np.flatnonzero(mask)[:100])
    pd.DataFrame(records).to_csv(OUT/'shap_features.csv',index=False);pd.DataFrame(top).to_csv(OUT/'shap_groups.csv',index=False)
    correlations=[]
    keys=list(vectors)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:correlations.append(dict(a=a,b=b,spearman=float(spearmanr(vectors[a],vectors[b]).statistic),top20_overlap=len(set(np.argsort(vectors[a])[-20:])&set(np.argsort(vectors[b])[-20:]))))
    save('shap_results.json',dict(model='Existing uncalibrated full LightGBM, each seed separately',scale='raw log-odds',method='LightGBM TreeSHAP pred_contrib; tree-path training cover reference',max_additivity_error=max(errors),stability=correlations,not_causal=True))
    print('SHAP complete',flush=True)

if __name__=='__main__':
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['features','q','v','labels','shap','all']);stage=ap.parse_args().stage
    for key,fn in [('features',features),('q',q_experiment),('v',value_audit),('labels',label_audit),('shap',shap_audit)]:
        if stage in [key,'all']:fn()
