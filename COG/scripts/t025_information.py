"""Primary logistic F0-F3 plus matched numeric-only tree comparison (T025)."""
from t025_common import *
import argparse,time,warnings
ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['T','S'],required=True)
a=ap.parse_args()
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import joblib
t0=time.time();packs,cols=load_packs(a.cohort)
groups=json.loads((REPO/'docs/supplementary_e1_20260921/feature_groups.json').read_text(encoding='utf-8'))
# Preserve original numeric column order within every subset for reproducible optimization.
for spec in groups['arms'].values():
    wanted=set(spec['columns_from_X']);spec['columns_from_X']=[c for c in cols if c in wanted]
out=provenance();out.update(cohort=a.cohort,roles={k:dict(n=len(p['y']),matches=len(np.unique(p['g']))) for k,p in packs.items()},arms={})
pred={};models={}
for arm,spec in groups['arms'].items():
    ix=[cols.index(n) for n in spec['columns_from_X']]
    matrices={k:np.column_stack([p['X'][:,ix],p['p_pre']]) for k,p in packs.items()}
    print('fit',a.cohort,arm,'columns',len(ix)+1,flush=True)
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000))
    with threadpool_limits(limits=2),warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always',ConvergenceWarning)
        model.fit(matrices['TR'],packs['TR']['y'],logisticregression__sample_weight=weights(packs['TR']['g']))
        predictions={k:model.predict_proba(x)[:,1] for k,x in matrices.items() if k in ['SE','TE']}
    row=dict(columns=len(ix)+1,train_weight='1/n_match',C=.1,calibration='identity',
             select=scores(packs['SE']['y'],predictions['SE'],packs['SE']['g']),
             test=scores(packs['TE']['y'],predictions['TE'],packs['TE']['g']),
             converged=not any(issubclass(w.category,ConvergenceWarning) for w in ws),iterations=model[-1].n_iter_.tolist())
    out['arms'][arm]=row;pred[arm]=predictions['TE'];models[arm]=model
    joblib.dump(dict(model=model,columns=spec['columns_from_X']+['p_pre']),OUT/f'logit_{a.cohort}_{arm}.joblib')
    dump(f'information_{a.cohort}.json',out)
keys=list(pred);f0,f1,f2,f3=keys
y=packs['TE']['y'];g=packs['TE']['g']
out['contrasts']={name:contrast(y,pred[x],pred[z],g) for name,x,z in [('E_given_F',f3,f1),('F_given_E',f3,f2),('full_minus_base',f3,f0)]}
# Compare refit full arm with archived q on the same keys, independently of row order.
z=np.load(pred_path(a.cohort),allow_pickle=False)
ki={(str(m),int(s)):i for i,(m,s) in enumerate(zip(z['match'],z['s']))}
ii=np.array([ki[(str(m),int(s))] for m,s in zip(g,packs['TE']['s'])])
out['full_arm_archived_q_max_abs_diff']=float(np.max(np.abs(pred[f3]-z['p_q_base'][ii])))
out['full_vs_frozen_PT']=contrast(y,pred[f3],z['p_PT_flex'][ii],g)
dump(f'information_{a.cohort}.json',out)
np.savez_compressed(OUT/f'information_{a.cohort}_predictions.npz',match=g,s=packs['TE']['s'],y=y,**pred)
# Same 352 numeric inputs, scaling, fit weights, and partitions. Select tree complexity only on Q_SELECT.
import lightgbm as lgb
ix=[cols.index(n) for n in groups['arms'][f3]['columns_from_X']]
matrices={k:np.column_stack([p['X'][:,ix],p['p_pre']]) for k,p in packs.items()}
scaler=StandardScaler().fit(matrices['TR'])
matrices={k:scaler.transform(x) for k,x in matrices.items()}
tree_rows=[];best=None
for leaves,child in [(15,100),(15,300),(31,100),(31,300)]:
    print('tree',a.cohort,leaves,child,flush=True)
    model=lgb.LGBMClassifier(n_estimators=2000,learning_rate=.03,num_leaves=leaves,min_child_samples=child,
              reg_lambda=5.,colsample_bytree=.8,subsample=.8,subsample_freq=1,random_state=7,n_jobs=2,verbosity=-1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(matrices['TR'],packs['TR']['y'],sample_weight=weights(packs['TR']['g']),
                  eval_set=[(matrices['CA'],packs['CA']['y'])],eval_sample_weight=[weights(packs['CA']['g'])],
                  eval_metric='binary_logloss',callbacks=[lgb.early_stopping(50,verbose=False)])
        ps=model.predict_proba(matrices['SE'])[:,1]
    br=scores(packs['SE']['y'],ps,packs['SE']['g'])['brier']
    row=dict(leaves=leaves,min_child_samples=child,select_brier=br,best_iteration=model.best_iteration_)
    tree_rows.append(row)
    if best is None or br<best[0]:best=(br,model,row)
with warnings.catch_warnings():
    warnings.simplefilter('ignore');ptree=best[1].predict_proba(matrices['TE'])[:,1]
out['matched_tree']=dict(candidates=tree_rows,selected=best[2],test=scores(y,ptree,g),
       minus_logit=contrast(y,ptree,pred[f3],g),minus_PT=contrast(y,ptree,z['p_PT_flex'][ii],g),
       scope='Same inputs, partitions, fit weights; learner-specific optimization and tree tuning remain different.')
out['wall_s']=time.time()-t0
joblib.dump(dict(model=best[1],scaler=scaler,columns=groups['arms'][f3]['columns_from_X']+['p_pre']),OUT/f'matched_tree_{a.cohort}.joblib')
dump(f'information_{a.cohort}.json',out)
print('DONE',a.cohort,out['contrasts'],out['matched_tree']['minus_logit'],flush=True)
