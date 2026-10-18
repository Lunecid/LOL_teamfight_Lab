import math,time
import numpy as np
import torch
from torch import nn
import lightgbm as lgb
from scipy.special import gammaln,softmax
from common import *

def target_arrays(y):
    return {'gold':y['slice_team_gold'].reshape(-1,6), 'count':y['slice_team_events'][...,:14].reshape(-1,84),'hazard':y['hazard_class'].astype(int)}, {'gold':y['slice_gold_supported'].reshape(-1,6),'count':y['slice_event_supported'][...,:14].reshape(-1,84),'hazard':y['hazard_at_risk_supported']}
def swap_targets(y,m):
    yy={};mm={}
    for k,width in [('gold',1),('count',14)]:
        yy[k]=y[k].reshape(-1,3,2,width)[:,:,::-1].reshape(y[k].shape);mm[k]=m[k].reshape(-1,3,2,width)[:,:,::-1].reshape(m[k].shape)
    yy['hazard']=np.where(y['hazard']==1,2,np.where(y['hazard']==2,1,y['hazard']));mm['hazard']=m['hazard'].copy();return yy,mm
def setup_targets(y,meta):
    t,m=target_arrays(y);tt,mm=swap_targets(t,m)
    a={k:np.concatenate([t[k],tt[k]]) for k in t};mask={k:np.concatenate([m[k],mm[k]]) for k in m}
    match=np.tile(meta['match_id'],2);patch=np.tile(meta['patch'],2)
    w={k:weights(mask[k],match,patch) for k in m}
    scale=max(float(np.median(np.abs(a['gold'][mask['gold']]))),100.)
    theta=[];initial={}
    initial['gold']=[float(np.sum(np.where(mask['gold'][:,j],np.nan_to_num(a['gold'][:,j]),0)*w['gold'][:,j])/scale) for j in range(6)]
    initial['count']=[]
    for j in range(84):
        ix=mask['count'][:,j];v=a['count'][ix,j];ww=w['count'][ix,j]
        mean=float(np.sum(v*ww));var=float(np.sum((v-mean)**2*ww));theta.append(float(np.clip(mean*mean/max(var-mean,1e-8),.1,1000)) if mean>0 else 1000.)
        initial['count'].append(math.log(max(mean,1e-5)))
    initial['hazard']=[]
    for j in range(3):
        ix=mask['hazard'][:,j];hist=np.bincount(a['hazard'][ix,j],weights=w['hazard'][ix,j],minlength=3)
        # Priors for initialization only; CE keeps the original labels and weights.
        initial['hazard'].extend(np.log((hist+1e-5)/(hist.sum()+3e-5)).tolist())
    assert np.allclose(np.array(theta).reshape(3,2,14)[:,0],np.array(theta).reshape(3,2,14)[:,1])
    spec={'gold_scale':scale,'theta':theta,'initial':initial,'active':{k:(v.sum(0)>0).tolist() for k,v in w.items()}}
    return a,mask,w,spec
def nb_nll(y,logmu,theta):
    mu=np.exp(np.clip(logmu,-12,8));theta=np.asarray(theta)
    return -(gammaln(y+theta)-gammaln(theta)-gammaln(y+1)+theta*(np.log(theta)-np.log(theta+mu))+y*(np.log(mu)-np.log(theta+mu)))
def nb_derivatives(logmu,y,r):
    mu=np.exp(np.clip(logmu,-12,8));return r*(mu-y)/(r+mu),r*mu*(r+y)/(r+mu)**2
def numpy_loss(pred,y,mask,match,patch,spec):
    w={k:weights(mask[k],match,patch) for k in mask};groups={}
    for k in y:
        values=[]
        for j in range(y[k].shape[1]):
            ix=mask[k][:,j]
            if not ix.any():continue
            v=y[k][ix,j];p=pred[k][ix,j]
            if k=='gold':d=np.abs(p-v/spec['gold_scale']);loss=np.where(d<=1,.5*d*d,d-.5)
            elif k=='count':loss=nb_nll(v,p,spec['theta'][j])
            else:loss=-np.log(np.clip(p[np.arange(len(p)),v.astype(int)],1e-12,1))
            values.append(float(np.sum(loss*w[k][ix,j])))
        if values:groups[k]=float(np.mean(values))
    return float(np.mean(list(groups.values()))),groups

class Hierarchy(nn.Module):
    def __init__(self,shapes,spec,use_player=True,use_team=True,use_rift=True,use_relation=True):
        super().__init__();self.flags=(use_player,use_team,use_rift,use_relation)
        self.edge=nn.Sequential(nn.Linear(shapes['relation'][-1],16),nn.ReLU())
        self.player=nn.Sequential(nn.Linear(shapes['player'][-1]+16,32),nn.ReLU())
        self.pg=nn.GRUCell(32,32);self.tg=nn.GRUCell(32,32);self.rg=nn.GRUCell(32,32)
        self.team=nn.Sequential(nn.Linear(shapes['team'][-1],32),nn.ReLU())
        self.rift=nn.Sequential(nn.Linear(shapes['rift'][-1],32),nn.ReLU())
        self.fuse=nn.Sequential(nn.Linear(160,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.output=nn.Linear(64,99)
        nn.init.zeros_(self.output.weight)
        self.output.bias.data.copy_(torch.tensor(spec['initial']['gold']+spec['initial']['count']+spec['initial']['hazard'],dtype=torch.float32))
    def forward(self,x):
        b,l=x['present'].shape;ph=x['player'].new_zeros(b*10,32);th=ph.new_zeros(b*2,32);rh=ph.new_zeros(b,32)
        for t in range(l):
            rel=x['relation'][:,t];active=rel[...,8:9] # support of distance, raw relation has 6 channels then 6 masks
            e=(self.edge(rel)*active).sum(2)/active.sum(2).clamp_min(1)
            if not self.flags[3]:e=e*0
            pp=self.player(torch.cat([x['player'][:,t],e],-1)).reshape(b*10,32)
            present=x['present'][:,t].bool()
            ph=torch.where(present.repeat_interleave(10)[:,None],self.pg(pp,ph),ph)
            th=torch.where(present.repeat_interleave(2)[:,None],self.tg(self.team(x['team'][:,t]).reshape(b*2,32),th),th)
            rh=torch.where(present[:,None],self.rg(self.rift(x['rift'][:,t]),rh),rh)
        pa=ph.reshape(b,2,5,32).mean(2).reshape(b,64);ta=th.reshape(b,64)
        fused=self.fuse(torch.cat([pa if self.flags[0] else pa*0,ta if self.flags[1] else ta*0,rh if self.flags[2] else rh*0],1))
        out=self.output(fused)
        return {'gold':out[:,:6],'count':out[:,6:90].clamp(-12,8),'hazard':out[:,90:].reshape(b,3,3)}
def torch_loss(pred,y,mask,w,spec):
    values=[];theta=torch.tensor(spec['theta'],dtype=torch.float32,device=pred['count'].device)
    for k in ('gold','count','hazard'):
        terms=[]
        for j,active in enumerate(spec['active'][k]):
            if not active:continue
            ix=mask[k][:,j]
            if not ix.any():terms.append(pred[k].sum()*0);continue
            p=pred[k][ix,j];v=y[k][ix,j];ww=w[k][ix,j]
            if k=='gold':loss=nn.functional.huber_loss(p,v/spec['gold_scale'],reduction='none',delta=1.)
            elif k=='count':
                mu=p.exp();r=theta[j]
                loss=-(torch.lgamma(v+r)-torch.lgamma(r)-torch.lgamma(v+1)+r*(torch.log(r)-torch.log(r+mu))+v*(p-torch.log(r+mu)))
            else:loss=nn.functional.cross_entropy(p,v.long(),reduction='none')
            terms.append((loss*ww).sum())
        if terms:values.append(torch.stack(terms).mean())
    return torch.stack(values).mean()
def tensor_x(x):return {k:torch.from_numpy(np.ascontiguousarray(v)).float() for k,v in x.items()}
def neural_predict(model,x,batch=256):
    model.eval();out=defaultdict(list)
    with torch.no_grad():
        for s in range(0,len(x['present']),batch):
            z=model(tensor_x(subset(x,slice(s,s+batch))))
            for k,v in z.items():out[k].append((v.softmax(-1) if k=='hazard' else v).numpy())
    return {k:np.concatenate(v) for k,v in out.items()}

def tree_fit(x,y,mask,w,spec,card):
    models={};timings={};start=time.time()
    for kind in ('gold','count','hazard'):
        fitted=[]
        for j in range(y[kind].shape[1]):
            ix=mask[kind][:,j];v=y[kind][ix,j];ww=w[kind][ix,j]
            if not len(v):fitted.append({'constant':True,'unsupported':True,'value':0 if kind!='hazard' else [1/3]*3});continue
            params={'verbosity':-1,'num_threads':card['torch_threads'],'learning_rate':card['tree_learning_rate'],'num_leaves':card['tree_leaves'],'min_data_in_leaf':card['tree_min_leaf'],'max_depth':5,'max_delta_step':1.,'seed':card['seed'],'feature_pre_filter':False,'force_col_wise':True,'deterministic':True,'lambda_l2':1.}
            if kind=='hazard':
                if len(np.unique(v))<2:
                    hist=np.bincount(v.astype(int),minlength=3)+.5;fitted.append({'constant':True,'value':(hist/hist.sum()).tolist()});continue
                params.update(objective='multiclass',num_class=3);base=None;target=v.astype(int);objective=None
            else:
                base=spec['initial'][kind][j];target=v/spec['gold_scale'] if kind=='gold' else v
                if np.all(v==v[0]):fitted.append({'constant':True,'value':float(target[0] if kind=='gold' else math.log(max(float(v[0]),1e-5)))});continue
                if kind=='gold':
                    def objective(pred,ds):
                        d=pred-ds.get_label();weights_=ds.get_weight();return np.clip(d,-1,1)*weights_,np.where(np.abs(d)<=1,1.,.01)*weights_
                else:
                    theta=spec['theta'][j]
                    def objective(pred,ds,r=theta):
                        grad,hess=nb_derivatives(pred,ds.get_label(),r);weights_=ds.get_weight();return grad*weights_,hess*weights_
                params['objective']=objective
            # Rescale the normalized target weights to mean 1 for tree regularization.
            ds=lgb.Dataset(x[ix],label=target,weight=ww*len(ww)/ww.sum(),init_score=np.full(len(v),base) if base is not None else None,free_raw_data=True)
            booster=lgb.train(params,ds,num_boost_round=card['tree_rounds'])
            fitted.append({'constant':False,'model':booster.model_to_string(),'base':base,'trees':booster.num_trees()})
        models[kind]=fitted;timings[kind]=time.time()-start;progress('tree_heads',group=kind,heads=len(fitted))
    return {'models':models,'timing_cumulative_s':timings}
def tree_predict(model,x):
    out={}
    for k,heads in model['models'].items():
        preds=[]
        for h in heads:
            if h['constant']:v=np.broadcast_to(np.array(h['value']), (len(x),3) if k=='hazard' else (len(x),)).copy()
            else:
                b=lgb.Booster(model_str=h['model']);v=b.predict(x,num_threads=4)
                if k!='hazard':v+=h['base']
            if k=='count':v=np.clip(v,-12,8)
            preds.append(v)
        out[k]=np.stack(preds,axis=1)
    return out

def derived(directional,spec):
    """Equal mixture after each directional distribution is accumulated."""
    branches=[]
    theta=np.array(spec['theta']).reshape(3,2,14)
    for flipped,p in enumerate(directional):
        g=p['gold'].reshape(-1,3,2)*spec['gold_scale'];mu=np.exp(p['count'].reshape(-1,3,2,14));h=p['hazard'].copy()
        if flipped:g=g[:,:,::-1];mu=mu[:,:,::-1];h=h[:,:,[0,2,1]]
        cumg=np.cumsum(g,axis=1);cumcount=np.cumsum(mu,axis=1)
        pzero=np.cumprod((theta/(theta+mu))**theta,axis=1)
        survival=np.ones(len(g));blue=np.zeros(len(g));red=np.zeros(len(g));terminal=[]
        for t in range(3):
            blue+=survival*h[:,t,1];red+=survival*h[:,t,2];survival=survival*h[:,t,0];terminal.append(np.stack([survival,blue,red],-1))
        branches.append({'cum_gold':cumg,'net_gold':cumg[:,:,0]-cumg[:,:,1],'cum_count':cumcount,'count_positive':1-pzero,'terminal':np.stack(terminal,1)})
    return {k:(branches[0][k]+branches[1][k])/2 for k in branches[0]}

from collections import defaultdict
