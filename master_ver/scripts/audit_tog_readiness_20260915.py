"""Read-only cross-artifact audit; does not fit/change any scientific model."""
from pathlib import Path
import collections
import hashlib
import json
import re
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / 'outputs/full_corpus_training_20260915'
CO = ROOT / 'outputs/cohort_role_training_20260915'
REPO = Path('C:/Users/todtj/PycharmProjects/LOL_teamfight')
OUT = ROOT / 'outputs/tog_readiness_audit_20260915'
OUT.mkdir(parents=True, exist_ok=True)
result = {'scope': 'Independent arithmetic on saved arrays, source/definition trace and manuscript inventory; not a raw-telemetry re-extraction or expert semantic validation', 'labels': {}, 'metrics': {}}

for f in sorted((FULL / 'labels').glob('*_labels.npz')):
    name = f.name.removesuffix('_labels.npz')
    z = np.load(f, allow_pickle=False)
    ids, s, last, pre = z['match'], z['s'], z['L'], z['p_pre']
    nextkill, nextstart = z['next_kill'], z['next_start_eff']
    cohort = np.load(CO / 'cohorts' / (name + '_cohort.npz'), allow_pickle=False)
    # Stored cohort encoding: 1=T, 0=N. Do not treat numeric codes as display names.
    c = np.where(cohort['cohort']==1, 'T', np.where(cohort['cohort']==0, 'N', 'unknown'))
    known, small = cohort['scale_known'].astype(bool), cohort['n_min']
    a = {'rows': len(ids), 'matches': len(np.unique(ids)), 'duplicate_keys': len(ids)-len(set(zip(ids.tolist(), s.tolist()))),
         'pre_time_errors': int(np.count_nonzero(z['q_pre'] != s-1)),
         'cohort_key_match': bool(np.array_equal(ids, cohort['match']) and np.array_equal(s, cohort['s']) and np.array_equal(last, cohort['L'])),
         'teamfight_rule_errors': int(np.count_nonzero((c=='T') != (known & (small>=4)))),
         'horizons': {}}
    for h in (60,90,120):
        m = z[f'valid_h{h}'].astype(bool)
        expected = np.minimum.reduce([last+1000*h, np.where(nextkill<0, np.iinfo(np.int64).max, nextkill-1), np.where(nextstart<0, np.iinfo(np.int64).max, nextstart-1), z['game_end']-1])
        endpoint, delta, y = z[f'endpoint_h{h}'], z[f'delta_h{h}'], z[f'Y_h{h}']
        err = delta[m]-(z[f'p_post_h{h}'][m]-pre[m])
        duration = (endpoint[m]-last[m])/1000
        a['horizons'][str(h)] = {
          'valid_rows': int(m.sum()), 'valid_matches': len(np.unique(ids[m])), 'excluded': int((~m).sum()),
          'T': int((m & (c=='T')).sum()), 'N': int((m & (c=='N')).sum()), 'unknown': int((m & ~known).sum()),
          'endpoint_formula_errors_all_rows': int((expected!=endpoint).sum()),
          'delta_formula_max_error': float(np.max(np.abs(err))), 'label_formula_errors': int((y[m]!=(delta[m]>0)).sum()),
          'future_pre_frames': int((z['pre_snapshot'][m]>z['q_pre'][m]).sum()),
          'future_post_frames': int((z[f'post_snapshot_h{h}'][m]>endpoint[m]).sum()),
          'extra_kills_recorded': int(z[f'after_raw_kills_h{h}'][m].sum()),
          'no_new_frame_after_last_kill_fraction': float(np.mean(z[f'post_snapshot_h{h}'][m]<=last[m])),
          'post_l_duration_mean_s': float(np.mean(duration)), 'post_l_duration_median_s': float(np.median(duration)),
          'horizon_reach_fraction': float(np.mean(endpoint[m]==last[m]+1000*h)),
          'near_zero_fraction': {str(b):float(np.mean(abs(delta[m])<=b)) for b in (.005,.01,.02)},
          'exact_zero': int((delta[m]==0).sum()), 'termination_counts': dict(collections.Counter(z[f'reasons_h{h}'][m].tolist()))}
    m = z['valid_h60'].astype(bool)&z['valid_h90'].astype(bool)&z['valid_h120'].astype(bool)
    a['horizon_sign_disagreement'] = {f'{i}_{j}': float(np.mean(z[f'Y_h{i}'][m]!=z[f'Y_h{j}'][m])) for i,j in ((60,90),(90,120),(60,120))}
    assert a['cohort_key_match'] and a['teamfight_rule_errors']==0
    assert all(v['T']+v['N']+v['unknown']==v['valid_rows'] for v in a['horizons'].values())
    result['labels'][name] = a
    z.close(); cohort.close()

for name in ('T','N'):
    f=CO/'eval/predictions'/f'A_MAIN_TEST_h90_{name}.npz'
    z=np.load(f,allow_pickle=False)
    ids, idx, cnt=np.unique(z['match'],return_inverse=True,return_counts=True)
    weights=1/cnt[idx]; y=z['y']
    cells={}
    for k in ['pooled','spec_constant','spec_p_pre_logistic','spec_p_pre_spline','spec_'+str(z['specialist_chosen'])]:
        p=z[k];cells[k]={'auc':float(roc_auc_score(y,p,sample_weight=weights)), 'brier':float(brier_score_loss(y,p,sample_weight=weights)), 'logloss':float(log_loss(y,p,sample_weight=weights))}
    result['metrics'][name]=cells
    # Audit-added post-hoc comparison, using existing predictions only.
    # One mean loss per match; resample matches, not correlated engagements.
    selected=z['spec_'+str(z['specialist_chosen'])]
    baseline=z['spec_p_pre_spline']
    differences={}
    rng=np.random.default_rng(20260915)
    for metric in ('brier','logloss'):
        if metric=='brier':
            d=(selected-y)**2-(baseline-y)**2
        else:
            p=np.clip(selected,1e-15,1-1e-15); b=np.clip(baseline,1e-15,1-1e-15)
            d=-y*np.log(p)-(1-y)*np.log1p(-p)+y*np.log(b)+(1-y)*np.log1p(-b)
        per_match=np.bincount(idx,weights=d)/cnt
        boot=np.array([per_match[rng.integers(len(ids),size=len(ids))].mean() for _ in range(1000)])
        differences[metric]={'difference':float(per_match.mean()),'percentile_95_ci':np.quantile(boot,[.025,.975]).tolist()}
    result.setdefault('posthoc_specialist_vs_p_pre_spline',{})[name]={'n_matches':len(ids),'bootstrap':1000,'seed':20260915,'multiple_comparison_adjustment':False,'losses':differences}
    if name=='T':
        labels=np.load(FULL/'labels/MAIN_TEST_labels.npz',allow_pickle=False)
        lookup=dict(zip(zip(labels['match'].tolist(),labels['s'].tolist()),labels['p_pre']))
        p_pre=np.array([lookup[k] for k in zip(z['match'].tolist(),z['s_ms'].tolist())])
        pred=z['spec_p_pre_logistic']; logits=np.log(pred/(1-pred))
        coef=np.linalg.lstsq(np.column_stack((np.ones(len(p_pre)),p_pre)),logits,rcond=None)[0]
        result['pre_probability_baseline_direction']={'scope':'Recover existing prediction function; no fitting to outcomes','intercept':float(coef[0]),'slope':float(coef[1]),'max_logit_reconstruction_error':float(np.max(abs(logits-(coef[0]+coef[1]*p_pre))))}
        labels.close()
    z.close()

main_ids={}
for n in ('TRAIN','VALIDATION','TEST'):
    with np.load(FULL/'labels'/f'MAIN_{n}_labels.npz',allow_pickle=False) as z:
        main_ids[n]=set(z['match'].tolist())
result['main_engagement_match_overlap']={f'{a}_{b}':len(main_ids[a]&main_ids[b]) for a,b in (('TRAIN','VALIDATION'),('TRAIN','TEST'),('VALIDATION','TEST'))}

manuscript=REPO/'docs/tog_manuscript'
pending=[]; pointers=[]
for f in sorted(manuscript.glob('*.tex')):
    for n,line in enumerate(f.read_text(encoding='utf-8').splitlines(),1):
        text=re.split(r'(?<!\\)%',line)[0]
        if '\\newcommand' not in text and '\\providecommand' not in text:
            pending += [{'file':f.name,'line':n,'key':k} for k in re.findall(r'\\pending\{([^}]+)\}',text)]
        if any(k in text for k in ('full_corpus_training_20260915','cohort_role_training_20260915')):
            pointers.append([f.name,n])
result['manuscript']={'pending_occurrences':len(pending),'pending_by_key':dict(collections.Counter(x['key'] for x in pending)), 'pending':pending, 'new_run_path_mentions_noncomment':pointers, 'conclusion_exists':(manuscript/'sec_conclusion.tex').exists(), 'not_compiled_this_audit':True}
current=REPO/'config/fight_boundary/spec_pooled.json'
result['definition_spec']=json.loads(current.read_text(encoding='utf-8'))
old=Path('D:/LOL_Project/fusion_2615/features/fight_boundary/spec_pooled.json')
result['older_similar_path_not_authoritative']={'path':str(old),'n_matches':json.loads(old.read_text())['n_matches']}
queue=Path('D:/LOL_Project/fusion_2615/features/tog_revision/queue_state')
result['legacy_queue']={key:{'ok':(queue/(key+'.ok')).exists(),'running_marker':(queue/(key+'.running')).exists()} for key in ('c_saint_d32_pretrain5_published_config','d1_hparam_search_lightgbm','d2_hparam_search_deep','d3_hparam_search_summary')}
sources=[current, REPO/'config/fight_boundary/drift.json', FULL/'protocol.json', FULL/'REPORT.md', FULL/'validation.json', CO/'protocol.json', CO/'REPORT.md',CO/'validation.json',manuscript/'main.tex',manuscript/'sec_label.tex',manuscript/'REVISION_TODO.md', Path(__file__)]
result['source_hashes']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
(OUT/'calculations.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'main':{n:{'rows':v['rows'],'h90':v['horizons']['90'],'flips':v['horizon_sign_disagreement']} for n,v in result['labels'].items() if n.startswith('MAIN')},'metrics':result['metrics'],'pending_by_key':result['manuscript']['pending_by_key']},ensure_ascii=False,indent=2))
