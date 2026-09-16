import json
S=json.load(open('shap/shap_summary.json',encoding='utf-8'))
print('all pass', S['all_checks_pass'])
for c,v in S['cohorts'].items():
    print(c, v['model'], v['overall_h90_winner'], v['explained_rows'], v['explained_matches'], v['background_rows'], v['coverage_note'], v['group_sizes'], 'base', round(v['base_value_mean'],4))
    print('  checks', v['checks'])
    for g in v['global_mean_abs']:
        print('   %-18s mean|phi| %.5f CI [%.5f, %.5f] signed %+.5f'%(g, v['global_mean_abs'][g], *v['global_mean_abs_bootstrap_ci95'][g], v['global_mean_signed'][g]))
    for b,d in v['time_bands'].items():
        print('   band',b,d['n'], {k:round(x,4) for k,x in (d['mean_abs'] or {}).items()})
    lc=v['local_cases'][0]
    print('   case', lc['match'][:6]+'...', lc['start_minute'], lc['y_h90'], round(lc['q_final'],4), {k:round(x,4) for k,x in lc['group_shapley'].items()})
    for t in lc['teams']:
        print('     ', t['team'], round(t['assignment_entropy'],3), [(p['champion'], p['estimated_role'], round(p['max_marginal'],3)) for p in t['participants']])
