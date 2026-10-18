import copy,time,os
from collections import defaultdict
import joblib,numpy as np,torch
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score
from common import *
from models import *

def concat(a,b):return {k:np.concatenate([a[k],b[k]]) for k in a}
def make_eval(model_predict,x):return [model_predict(x),model_predict(swap(x))]
def score(pred,dirs,y,meta,spec):
    match=meta['match_id'];patch=meta['patch'];rows=[]
    for h in range(3):
        mask=np.isfinite(y['net_gold'][:,h]);w=weights(mask[:,None],match,patch)[:,0];err=pred['net_gold'][mask,h]-y['net_gold'][mask,h]
        r={'horizon_s':60*(h+1),'net_gold_mae':float(np.sum(w[mask]*np.abs(err))),'net_gold_rmse':float(np.sqrt(np.sum(w[mask]*err**2))),'gold_supported_rows':int(mask.sum()),'all_rows':len(mask),'gold_supported_matches':len(np.unique(match[mask]))}
        team=[]
        for z in range(2):
            mask=y['gold_supported'][:,h,z];w=weights(mask[:,None],match,patch)[:,0];e=pred['cum_gold'][mask,h,z]-y['cum_team_gold'][mask,h,z];team.append({'mae':float(np.sum(w[mask]*np.abs(e))),'rmse':float(np.sqrt(np.sum(w[mask]*e**2)))})
        r['team_gold']=team;ev=[]
        for z in range(2):
            for k,name in enumerate(EVENTS):
                mask=y['cum_event_supported'][:,h,z,k];w=weights(mask[:,None],match,patch)[:,0];v=y['cum_team_events'][mask,h,z,k];prob=pred['count_positive'][mask,h,z,k];mu=pred['cum_count'][mask,h,z,k];ww=w[mask]
                sm=y['slice_event_supported'][:,h,z,k];sw=weights(sm[:,None],match,patch)[:,0];head=(h*2+z)*14+k;other=(h*2+1-z)*14+k
                losses=np.stack([-nb_nll(y['slice_team_events'][sm,h,z,k],dirs[0]['count'][sm,head],spec['theta'][head]),-nb_nll(y['slice_team_events'][sm,h,z,k],dirs[1]['count'][sm,other],spec['theta'][other])])
                nll=-logsumexp(losses,axis=0)+np.log(2)
                ev.append({'team':z,'event':name,'supported_rows':int(mask.sum()),'positive_rows':int((v>0).sum()),'positive_matches':len(np.unique(match[mask][v>0])),'mae':float(np.sum(ww*np.abs(mu-v))) if mask.any() else None,'brier':float(np.sum(ww*(prob-(v>0))**2)) if mask.any() else None,'ap':float(average_precision_score(v>0,prob,sample_weight=ww)) if (v>0).any() and (v==0).any() else None,'slice_nb_nll':float(np.sum(sw[sm]*nll)) if sm.any() else None})
        r['events']=ev
        mask=y['terminal_supported'][:,h];w=weights(mask[:,None],match,patch)[:,0];label=y['terminal_class'][mask,h].astype(int);prob=pred['terminal'][mask,h];r['terminal_logloss']=float(np.sum(w[mask]*-np.log(np.clip(prob[np.arange(len(label)),label],1e-12,1))));r['terminal_brier']=float(np.sum(w[mask]*np.sum((prob-np.eye(3)[label])**2,1)));r['terminal_supported_rows']=int(mask.sum());r['terminal_ended_rows']=int((label>0).sum())
        rows.append(r)
    return rows

def train_neural(xf,yf,mf,xs,ys,ms,spec,card):
    torch.manual_seed(card['seed']);np.random.seed(card['seed']);torch.set_num_threads(card['torch_threads']);torch.use_deterministic_algorithms(True)
    shapes={k:v.shape[1:] for k,v in xf.items()};net=Hierarchy(shapes,spec)
    xa=concat(xf,swap(xf));y,mask,w,_=setup_targets(yf,mf);tx=tensor_x(xa);ty={k:torch.from_numpy(v).float() for k,v in y.items()};tm={k:torch.from_numpy(v) for k,v in mask.items()};tw={k:torch.from_numpy(v).float() for k,v in w.items()}
    st,sm=target_arrays(ys);sst,ssm=swap_targets(st,sm);sy=concat(st,sst);sma=concat(sm,ssm);stopmatch=np.tile(ms['match_id'],2);stoppatch=np.tile(ms['patch'],2)
    optimizer=torch.optim.Adam(net.parameters(),lr=card['learning_rate'],weight_decay=1e-5)
    trace=[];best=float('inf');best_state=None;best_epoch=None;n=len(xa['present']);rng=np.random.default_rng(card['seed']);start=time.time()
    for epoch in range(card['epochs']):
        net.train();perm=rng.permutation(n);losses=[]
        for s in range(0,n,card['batch_size']):
            idx=perm[s:s+card['batch_size']];pred=net({k:v[idx] for k,v in tx.items()});loss=torch_loss(pred,{k:v[idx] for k,v in ty.items()},{k:v[idx] for k,v in tm.items()},{k:v[idx] for k,v in tw.items()},spec)*n/len(idx)
            assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();losses.append(loss.item())
        dirs=make_eval(lambda x:neural_predict(net,x),xs);val,groups=numpy_loss(concat(dirs[0],dirs[1]),sy,sma,stopmatch,stoppatch,spec)
        trace.append({'epoch':epoch+1,'training_minibatch_objective':float(np.mean(losses)),'stop_objective':val,'stop_groups':groups,'elapsed_s':time.time()-start})
        if val<best:best=val;best_state=copy.deepcopy(net.state_dict());best_epoch=epoch+1
        progress('neural_epoch',epoch=epoch+1,stop_objective=val)
    net.load_state_dict(best_state);net.eval();torch.save({'state_dict':best_state,'shapes':shapes,'spec':spec,'best_epoch':best_epoch,'card_sha256':sha(HERE/'CARD.json')},HERE/'hierarchy.pt')
    dump(HERE/'TRAIN_TRACE.json',trace)
    return net,{'best_epoch':best_epoch,'parameters':sum(p.numel() for p in net.parameters()),'elapsed_s':time.time()-start}

def main():
    assert read(HERE/'TEST_RESULTS.json')['passed'];assert not (HERE/'RESULTS.json').exists(),'Refuse to overwrite completed run'
    card=read(HERE/'CARD.json');assert not card['full_training'];assert sha(HERE/'DATA.joblib')==read(HERE/'DATA_MANIFEST.json')['data_sha256']
    data=joblib.load(HERE/'DATA.joblib');x,y,m=data['x'],data['y'],data['meta'];fit=m['macro_role']=='FIT';stop=m['macro_role']=='STOP';assert (fit|stop).all()
    xf,xs=subset(x,fit),subset(x,stop);yf,ys=subset(y,fit),subset(y,stop);mf,ms=subset(m,fit),subset(m,stop)
    ta,mask,w,spec=setup_targets(yf,mf);dump(HERE/'LOSS_SPEC.json',spec)
    progress('train_start',fit_rows=int(fit.sum()),stop_rows=int(stop.sum()))
    net,ninfo=train_neural(xf,yf,mf,xs,ys,ms,spec,card);dirs_nn=make_eval(lambda a:neural_predict(net,a),xs);pred_nn=derived(dirs_nn,spec)
    start=time.time();tree=tree_fit(flat(concat(xf,swap(xf))),ta,mask,w,spec,card);joblib.dump(tree,HERE/'tree.joblib',compress=3)
    dirs_tree=make_eval(lambda a:tree_predict(tree,flat(a)),xs);pred_tree=derived(dirs_tree,spec);tree_time=time.time()-start
    # Reload both independently; acceptance requires identical held-out predictions.
    checkpoint=torch.load(HERE/'hierarchy.pt',map_location='cpu',weights_only=True);reloaded=Hierarchy(checkpoint['shapes'],checkpoint['spec']);reloaded.load_state_dict(checkpoint['state_dict']);again=derived(make_eval(lambda a:neural_predict(reloaded,a),xs),spec)
    reload_errors={'hierarchy':max(float(np.max(np.abs(pred_nn[k]-again[k]))) for k in pred_nn)}
    again=derived(make_eval(lambda a:tree_predict(joblib.load(HERE/'tree.joblib'),flat(a)),xs),spec);reload_errors['tree']=max(float(np.max(np.abs(pred_tree[k]-again[k]))) for k in pred_tree);assert max(reload_errors.values())<1e-6
    # Paired team orientation symmetry and probability/coherence gates.
    checks={}
    for name,dirs,pred in [('hierarchy',dirs_nn,pred_nn),('tree',dirs_tree,pred_tree)]:
        reversed_=derived(dirs[::-1],spec)
        checks[name]={'swap_net_gold_max_abs':float(np.max(np.abs(pred['net_gold']+reversed_['net_gold']))),'terminal_probability_sum_max_abs':float(np.max(np.abs(pred['terminal'].sum(-1)-1))),'terminal_nonmonotone_cells':int((np.diff(pred['terminal'][:,:,1:],axis=1)<-1e-7).sum()),'count_probability_nonmonotone_cells':int((np.diff(pred['count_positive'],axis=1)<-1e-7).sum()),'negative_cumulative_team_gold_fraction':float((pred['cum_gold']<0).mean())}
        assert checks[name]['swap_net_gold_max_abs']<1e-5 and checks[name]['terminal_probability_sum_max_abs']<1e-5 and checks[name]['terminal_nonmonotone_cells']==0 and checks[name]['count_probability_nonmonotone_cells']==0
    results={'purpose':card['purpose'],'evaluation_role':'STOP_15.20','fit_rows':int(fit.sum()),'stop_rows':int(stop.sum()),'fit_matches':len(set(mf['match_id'])),'stop_matches':len(set(ms['match_id'])),'L':card['L'],'seed':card['seed'],'loss_spec_sha256':sha(HERE/'LOSS_SPEC.json'),'input_flat_dimension':flat(xf[:0] if isinstance(xf,np.ndarray) else subset(xf,slice(0,1))).shape[1],'hierarchy_info':ninfo,'tree_info':{'recipe_runs':1,'head_slots':93,'trained_estimators':sum(not h['constant'] for hs in tree['models'].values() for h in hs),'actual_trees':sum(h.get('trees',0) for hs in tree['models'].values() for h in hs),'elapsed_s':tree_time},'metrics':{'hierarchy':score(pred_nn,dirs_nn,ys,ms,spec),'tree':score(pred_tree,dirs_tree,ys,ms,spec)},'reload_errors':reload_errors,'coherence_checks':checks,'full_training':False,'training_allowed_roles':['FIT'],'selection_allowed_roles':['STOP'],'no_test_rows_loaded':True,'limitations':['Single small smoke and one seed/L: no architecture or L superiority conclusion.','Fixed 90s tactical sorting is a declared proxy; needs validation before final strong-tree claims.','Observed item-flow multiset masks undo/negative balances, not verified complete inventory.','Count interval independence and directional mixture do not enforce team exclusivity or terminal absorption of gold/count.','No K3, terrain geometry, exact respawn, buff ownership or U enabled.','Cumulative count NLL not reported; NB NLL is per disjoint interval; cumulative event Brier uses product P0 within each directional branch.']}
    np.savez_compressed(HERE/'PREDICTIONS.npz',row_id=ms['row_id'],match_id=ms['match_id'],**{name+'_'+k:v for name,p in [('tree',pred_tree),('hierarchy',pred_nn)] for k,v in p.items()})
    dump(HERE/'RESULTS.json',results)
    assert sha(HERE/'DATA.joblib')==read(HERE/'DATA_MANIFEST.json')['data_sha256']
    assert all(sha(p)==d for p,d in read(HERE/'RAW_SOURCE_HASHES.json').items())
    assert all(sha(p)==d for p,d in read(HERE/'DATA_MANIFEST.json')['shard_hashes'].items())
    dump(HERE/'ACCEPTANCE.json',{'pass':True,'unit_gates':read(HERE/'TEST_RESULTS.json'),'reload_errors':reload_errors,'source_files_and_shards_unchanged':True,'data_unchanged':True,'results_sha256':sha(HERE/'RESULTS.json'),'predictions_sha256':sha(HERE/'PREDICTIONS.npz'),'full_training':False,'db_read_only':True})
    progress('smoke_complete')
if __name__=='__main__':main()
