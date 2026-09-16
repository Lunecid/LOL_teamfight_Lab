import json
R=json.load(open('eval/results_C.json',encoding='utf-8'))
def f(x): return 'NA' if x is None else f'{x:.4f}'
for n,rc in R['results'].items():
  for c,cc in rc.items():
    aw=cc['arm_winners']; m=cc['metrics']
    print(n,c,cc['rows'],'matches',cc['matches'])
    for k,cand in [('role',aw['role']),('participant',aw['participant']),('draft',aw['draft']),('A','A_specialist'),('pooled','pooled'),('part_ridge','participant_ridge_raw'),('role_ridge','role_ridge_raw')]:
        print('    %-12s %-26s auc %s brier %s ll %s slope %s ece %s ex01 %d/%d'%(k,cand,f(m[cand]['auc']),f(m[cand]['brier']),f(m[cand]['logloss']),f(m[cand]['slope']),f(m[cand]['ece_10bin']),m[cand]['exact_0_or_1'],m[cand]['exact_0_or_1_opposite_label']))
    for p in cc.get('bootstrap',{}).get('pairs',[]):
        d=p['a_minus_b']; print('      %-28s - %-28s brier %+.5f [%+.5f,%+.5f] ll %+.5f [%+.5f,%+.5f] auc %+.4f [%+.4f,%+.4f]'%(p['a'],p['b'],d['brier']['estimate'],*d['brier']['ci95'],d['logloss']['estimate'],*d['logloss']['ci95'],d['auc']['estimate'],*d['auc']['ci95']))
    print('    ident ok', all(v['within_1e_12'] for v in cc['reload_identity_first_rows'].values()), max(v['max_abs_diff_own_bundle_on_first_rows'] for v in cc['reload_identity_first_rows'].values()))
print(json.dumps(R['role_outputs'])[:1800]); print(R['input_checks'])
