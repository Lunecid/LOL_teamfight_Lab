import run_validation_suite as s
from evaluate_external_full import paired
import numpy as np,pandas as pd,json,hashlib,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def main():
    z=np.load(s.OUT/'features.npz');r=pd.read_csv(s.OUT/'q_predictions.csv',dtype={'patch':str});info=json.loads((s.OUT/'q_results.json').read_text());allnames=json.loads((s.OLD/'feature_names.json').read_text())['baseline'];names=info['ridge_names'];xx=z['X'][:,[allnames.index(n) for n in names]];y=r.y.to_numpy();g=r.match.to_numpy();tr=(r.patch=='15.14').to_numpy();half=np.array([int(hashlib.sha256(('calibration17:'+m).encode()).hexdigest()[:8],16)%2 for m in g]);ca=(r.patch=='15.15').to_numpy()&(half==0);te=(r.patch=='15.16').to_numpy()
    comparisons={key:paired(y[te],r.ridge_sigmoid.to_numpy()[te],r[key].to_numpy()[te],g[te]) for key in ['p_pre_logistic','p_pre_spline','economic_sigmoid']}
    s.save('incremental_baselines.json',comparisons)
    tests={'direct_objectives':lambda n:any(t in n for t in ['baron','elder','dragon','soul','herald','horde','atakhan']), 'prior_probability':lambda n:n=='p_pre','economy_experience':lambda n:any(t in n for t in ['Gold','xp_','level_','CS_']),'combat_survival':lambda n:any(t in n for t in ['kills','deaths','death_','alive'])}
    results=[];pred=r[['match','s','patch','y']].copy()
    for key,rule in tests.items():
        keep=[i for i,n in enumerate(names) if not rule(n)];m=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=3000));m.fit(xx[tr][:,keep],y[tr],logisticregression__sample_weight=s.weights(g[tr]));p=m.predict_proba(xx[:,keep])[:,1];cal=LogisticRegression(C=1e6,max_iter=1000).fit(s.logit(p[ca]),y[ca],sample_weight=s.weights(g[ca]));p=cal.predict_proba(s.logit(p))[:,1];pred[key]=p
        results.append(dict(removed=key,removed_names=[n for n in names if rule(n)],remaining_features=len(keep),test=s.calibration(y[te],p[te],s.weights(g[te])),minus_full=paired(y[te],p[te],r.ridge_sigmoid.to_numpy()[te],g[te])))
        joblib.dump(dict(model=m,calibrator=cal,feature_names=[names[i] for i in keep]),s.OUT/('drop_'+key+'.joblib'))
    pred.to_csv(s.OUT/'group_ablation_predictions.csv',index=False);s.save('group_ablation.json',dict(status='exploratory post-result diagnostic; no primary model reselection',results=results,note='Dropping direct objectives retains their possible indirect information in p_pre, gold, etc. This is not removal of all objective influence.'))
    # Independent direct formulas for reported Brier/log loss; no metrics helper.
    a=r[te];counts=a.groupby('match').match.transform('size').to_numpy();w=1/counts;w/=w.sum();p=a.ridge_sigmoid.to_numpy();yy=a.y.to_numpy();direct_brier=float(np.sum(w*(p-yy)**2));direct_logloss=float(-np.sum(w*(yy*np.log(p)+(1-yy)*np.log1p(-p))))
    target=next(t for t in info['metrics'] if t['model']=='ridge_sigmoid' and t['split']=='test_exploratory');assert abs(direct_brier-target['brier'])<1e-12 and abs(direct_logloss-target['logloss'])<1e-12
    s.save('independent_numerical_checks.json',dict(brier_direct=direct_brier,logloss_direct=direct_logloss,duplicate_rows=int(r.duplicated(['match','s']).sum()),split_overlap=len(set(g[tr])&set(g[te])),calibration_selection_overlap=0,all_probabilities_finite=bool(np.isfinite(r.ridge_sigmoid).all()),bins_partition_checked=all(sum(b['n'] for b in t['bins'])==t['n'] for t in info['metrics'])))
    print('supplements complete',flush=True)

if __name__=='__main__':main()
