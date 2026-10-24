"""Frozen-model definition sensitivity, assist ablation, temporal invariance."""
from t025_common import *
import argparse,time,inspect,collections,copy,bisect
import joblib,torch
from rr20260920_review_response_rr12 import PTFlexFeatures
from t023_definition_oat import apply_oat,earliest_kill_key,current_params
from t023_e_fixed_probe import game_end_ms
from gameplay.state_value_v2 import StateBuilder,state_matrix
from core.config import cfg,NODE_FEATURE_NAMES
import gameplay.fights as F
import data.cache_io as cio,fc20260915_common as C
from v_redesign_evaluator_bundle import load_evaluator,predict_calibrated
from engagement_labels_v3_rules import endpoint_rule,endpoint_validity,next_kill_after
torch.set_num_threads(1)
ap=argparse.ArgumentParser();ap.add_argument('--matches',type=int,default=3000);ap.add_argument('--tag',default='definition')
a=ap.parse_args();t0=time.time()
cio.CACHE_DIR=Path(C.CACHE_MAIN)
names=json.loads((OUT/'feature_names.json').read_text(encoding='utf-8'))
arms={'ref':{},'R1400':{'R':1400},'R1800':{'R':1800},'I2000':{'I':2000},'I4264':{'I':4264},'no_shop':{'SHOPEX':True}}
arch={};mids=set()
for co in ['T','S']:
    with np.load(pred_path(co),allow_pickle=False) as z:
        arch[co]={(str(m),int(s)):dict(y=int(y),q=float(q),pt=float(pt),p=float(p)) for m,s,y,q,pt,p in zip(z['match'],z['s'],z['y'],z['p_q_base'],z['p_PT_flex'],z['p_pre'])}
        mids.update(z['match'].astype(str))
mids=sorted(mids,key=lambda m:hashlib.sha256(m.encode()).hexdigest())[:a.matches]
np.save(OUT/(a.tag+'_match_sample.npy'),np.array(mids))
rows={k:[] for k in arms};states=[];assist=[];inv=collections.Counter();errors=collections.Counter()
params={};keydup=0
orig_collect=F._collect_interactions_in_radius
def collect_assists(*args,**kwargs):
    result=orig_collect(*args,**kwargs)
    frame=inspect.currentframe().f_back
    c=frame.f_locals['cluster'];tm=frame.f_locals['tm'] if 'tm' in frame.f_locals else current_tm
    ka=set();direct=set()
    for k in c['kills']:
        ka.update(int(p) for p in k['assisting_ids'] if int(p) in tm)
        direct.update(int(k[x]) for x in ['killer_id','victim_id'] if int(k[x]) in tm)
    extras=set(result[1]);full=set(c['participants'])|extras;noassist=direct|extras;only=ka-direct
    n=lambda ps:min(sum(tm.get(p)==team for p in ps) for team in [100,200])
    sc=lambda n:'T' if n>=4 else 'S' if n>=2 else 'P'
    assist.append(dict(n_assist_only=len(only),n_recovered=len(only&noassist),
        jaccard=len(full&noassist)/len(full|noassist) if full|noassist else 1.,
        with_scale=sc(n(full)),without_scale=sc(n(noassist)),n_kill_actors=len(c['participants']),n_extra_only=len(extras-set(c['participants']))))
    return result

def raw_state(builder,q,cache):
    if q not in cache:
        cache[q]=len(states)
        states.append(state_matrix([builder.at(q)],names)[0])
    return cache[q]

for mi,mid in enumerate(mids):
    pack=cio.load_match_cache(mid)
    if pack is None:errors['missing_cache']+=1;continue
    current_tm={int(k):int(v) for k,v in pack['meta']['team_map'].items()}
    builder=StateBuilder(pack,list(NODE_FEATURE_NAMES));cache={}
    last=int(pack['minute_ts'][-1]);first=int(pack['minute_ts'][0]);gend=game_end_ms(pack)
    kills=sorted(int(e['timestamp']) for e in pack['events'] if e.get('type')=='CHAMPION_KILL')
    for arm,over in arms.items():
        apply_oat(cfg,over);params[arm]=current_params(cfg)
        if arm=='ref':F._collect_interactions_in_radius=collect_assists
        try:fs=F.detect_fights(pack,current_tm)
        finally:F._collect_interactions_in_radius=orig_collect
        fs=sorted(fs,key=lambda f:int(f['engage_ts']))
        for fi,f in enumerate(fs):
            s=int(f['engage_ts']);L=int(f['last_kill_ts']);q=s-1
            nm=min(int(f['det_cluster_blue']),int(f['det_cluster_red']))
            # Keep all scale classes for cutoff sensitivity; evaluate T/S later.
            ns=int(fs[fi+1]['engage_ts']) if fi+1<len(fs) else float('inf')
            ep=endpoint_rule(L,90,next_kill_after(kills,L),ns,gend)[0]
            flags=endpoint_validity(ep,L,q,first,last)
            if not all(flags.values()):errors[arm+'_invalid']+=1;continue
            anchor=earliest_kill_key(pack,f)
            if anchor is None:errors[arm+'_missing_anchor']+=1;continue
            pre=raw_state(builder,q,cache);post=raw_state(builder,ep,cache)
            rows[arm].append(dict(m=mid,s=s,L=L,ep=ep,nmin=nm,anchor=anchor,pre=pre,post=post))
    # Up to two archived query times per selected match; independent of new arm success.
    if mi<500:
        qs=sorted({s-1 for co in arch for m,s in arch[co] if m==mid})[:2]
        for q in qs:
            X=state_matrix([builder.at(q)],names)
            event_pack=dict(pack,events=[e for e in pack['events'] if int(e.get('timestamp',-1))<=q])
            xe=state_matrix([StateBuilder(event_pack,list(NODE_FEATURE_NAMES)).at(q)],names)
            inv['queries']+=1;inv['event_prefix_equal']+=int(np.array_equal(X,xe,equal_nan=True))
            future=np.asarray(pack['minute_ts'])>q
            changed=dict(pack)
            for key in ['node_minute','global_minute','gold_team_minute','xy_raw_minute']:
                if key in pack:
                    arr=pack[key].copy();arr[future]=0;changed[key]=arr
            xf=state_matrix([StateBuilder(changed,list(NODE_FEATURE_NAMES)).at(q)],names)
            inv['future_frame_values_equal']+=int(np.array_equal(X,xf,equal_nan=True))
            prefix=dict(event_pack)
            for key in ['minute_ts','node_minute','global_minute','gold_team_minute','xy_raw_minute']:
                if key in pack:prefix[key]=pack[key][~future]
            try:
                xp=state_matrix([StateBuilder(prefix,list(NODE_FEATURE_NAMES)).at(q)],names)
                inv['frame_prefix_accepted']+=1;inv['frame_prefix_equal']+=int(np.array_equal(X,xp,equal_nan=True))
            except ValueError:inv['frame_prefix_outside_observed_support']+=1
    if mi%100==0:
        print('definition matches',mi,'states',len(states),'rows',len(rows['ref']),'seconds',round(time.time()-t0),flush=True)
        dump(a.tag+'_progress.json',dict(matches_processed=mi,n_states=len(states),elapsed_s=time.time()-t0,errors=dict(errors)))

X=np.asarray(states);ev=load_evaluator(REPO/'outputs/v_redesign_wave4_corrected_20260919/evaluators/A_MLP_expanded_evaluator.joblib')
pv=predict_calibrated(ev,X)
keep=[i for i,n in enumerate(names) if n!='snapshot_age_s'];xq=X[:,keep]
models={};pts={}
for co in ['T','S']:
    models[co]=joblib.load(REPO/('outputs/q_newv_fit85_20260920'+('_S' if co=='S' else ''))/'models/logit_state.joblib')
    pts[co]=joblib.load(REPO/('outputs/review_response_rr12_20260920'+('_S_qS_id' if co=='S' else ''))/'models/PT_flex.joblib')

def score_rows(rr,co):
    if not rr:return dict(n=0),None
    pre=np.array([r['pre'] for r in rr]);post=np.array([r['post'] for r in rr]);g=np.array([r['m'] for r in rr])
    p=pv[pre];y=(pv[post]-p>0).astype(int)
    q=models[co]['pipe'].predict_proba(np.column_stack([xq[pre][:,models[co]['num_ix']],p]))[:,1]
    pt=pts[co].predict_proba(np.column_stack([p,np.array([r['s'] for r in rr])/60000]))[:,1]
    result=dict(q=scores(y,q,g),pt=scores(y,pt,g),delta=contrast(y,q,pt,g))
    return result,dict(y=y,q=q,pt=pt,p=p,g=g)

summary=provenance();summary.update(matches_requested=a.matches,matches_loaded=len(mids)-errors['missing_cache'],params=params,errors=dict(errors),arms={},parity={},invariance=dict(inv))
for arm,rr in rows.items():
    summary['arms'][arm]={}
    for co in ['T','S']:
        subset=[r for r in rr if (r['nmin']>=4 if co=='T' else 2<=r['nmin']<=3)]
        result,pr=score_rows(subset,co);summary['arms'][arm][co]=result
        if pr is None:continue
        # Stable first-kill anchors, same reference scale. This is conditional common-case evaluation.
        refkeys={(r['m'],tuple(r['anchor'])) for r in rows['ref'] if (r['nmin']>=4 if co=='T' else 2<=r['nmin']<=3)}
        common=np.array([(r['m'],tuple(r['anchor'])) in refkeys for r in subset])
        result['common_reference_scale']=contrast(pr['y'][common],pr['q'][common],pr['pt'][common],pr['g'][common])
        if arm=='ref':
            expected={k:v for k,v in arch[co].items() if k[0] in mids}
            found={(r['m'],r['s']):i for i,r in enumerate(subset)}
            shared=sorted(expected.keys()&found.keys())
            summary['parity'][co]=dict(archived_rows=len(expected),reconstructed_rows=len(found),common=len(shared),
                archived_only=len(expected.keys()-found.keys()),reconstructed_only=len(found.keys()-expected.keys()),
                label_mismatches=sum(int(pr['y'][found[k]])!=expected[k]['y'] for k in shared),
                max_abs_q=max(abs(pr['q'][found[k]]-expected[k]['q']) for k in shared),
                max_abs_pt=max(abs(pr['pt'][found[k]]-expected[k]['pt']) for k in shared),
                max_abs_p=max(abs(pr['p'][found[k]]-expected[k]['p']) for k in shared))
        # Local identifier-bearing rows stay in outputs only.
        np.savez_compressed(OUT/f'{a.tag}_{arm}_{co}.npz',match=pr['g'],s=np.array([r['s'] for r in subset]),y=pr['y'],q=pr['q'],pt=pr['pt'],anchor=np.array([r['anchor'] for r in subset]))
for threshold in [3,5]:
    tag=f'cutoff{threshold}';summary['arms'][tag]={}
    for co in ['T','S']:
        subset=[r for r in rows['ref'] if (r['nmin']>=threshold if co=='T' else 2<=r['nmin']<threshold)]
        summary['arms'][tag][co]=score_rows(subset,co)[0]
summary['assist']=dict(candidate_n=len(assist),assist_only_actors=sum(r['n_assist_only'] for r in assist),
    recovered_assist_only_actors=sum(r['n_recovered'] for r in assist),
    mean_jaccard=float(np.mean([r['jaccard'] for r in assist])),
    scale_changed=sum(r['with_scale']!=r['without_scale'] for r in assist),
    scale_transitions=dict(collections.Counter(r['with_scale']+'_'+r['without_scale'] for r in assist)),
    scope='Accepted pre-merge candidates; assistance erased from participant union only, boundary/presence unchanged. Not independent truth.')
summary['wall_s']=time.time()-t0
summary['modules']={m.__name__:dict(path=m.__file__,sha256=sha(m.__file__)) for m in [F,sys.modules['gameplay.state_value_v2'],sys.modules['gameplay.state_value']]}
dump(a.tag+'.json',summary)
print('DONE',a.tag,'parity',summary['parity'],'invariance',summary['invariance'],'assist',summary['assist'],flush=True)

