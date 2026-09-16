"""Independent main TEST teamfight objective-subgroup arithmetic from saved counters."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/objective_channel_ablation_20260915'
FC=ROOT/'outputs/full_corpus_training_20260915'
reported=json.loads((OUT/'results/per_set/MAIN_TEST.json').read_text(encoding='utf8'))
checks={}
with np.load(FC/'labels/MAIN_TEST_labels.npz',allow_pickle=False) as p, np.load(OUT/'labels/MAIN_TEST_B_noobj_labels.npz',allow_pickle=False) as z:
    assert np.array_equal(p['match'],z['match']) and np.array_equal(p['s'],z['s'])
    cols=p['count_keys'].tolist()
    valid=(p['valid_h90']==1)&(z['cohort']==1)
    for window in ('full_(q_pre,e]','after_(L,e]'):
        counts=p['after_counts_h90'].copy()
        if window.startswith('full'): counts+=p['during_counts']
        checks[window]={}
        for obj in ('baron','dragon','elder','herald','horde','atakhan','soul_owned'):
            blue=counts[:,cols.index(obj+'_blue')]>0
            red=counts[:,cols.index(obj+'_red')]>0
            m=valid&(blue!=red)
            sign=np.where(blue[m],1,-1)
            g=z['match'][m]
            _,ix,n=np.unique(g,return_inverse=True,return_counts=True)
            w=1/n[ix]
            dis=z['Y_h90_A'][m]!=z['Y_h90_B'][m]
            oriented=sign*(z['delta_h90_A'][m]-z['delta_h90_B'][m])
            cell=dict(rows=int(m.sum()),matches=len(n),disagreement_row=float(dis.mean()),
                disagreement_match=float(np.average(dis,weights=w)),
                oriented_difference_row=float(oriented.mean()),oriented_difference_match=float(np.average(oriented,weights=w)))
            ref=reported['objectives_h90']['cohorts']['T'][window][obj]['single_team_oriented']
            assert cell['rows']==ref['rows'] and cell['matches']==ref['matches']
            for a,b in [('disagreement','disagreement'),('oriented_difference','oriented_delta_A_minus_B')]:
                assert abs(cell[a+'_row']-ref[b]['row'])<1e-12
                assert abs(cell[a+'_match']-ref[b]['match_weighted'])<1e-12
            checks[window][obj]=cell
result=dict(passed=True,scope='Main TEST T, seven owned objective types, both fixed intervals, single-acquiring-team rows; arithmetic validation, not causal values',checks=checks)
(OUT/'codex_objective_subgroup_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('Passed: main TEST T, 7 objective types x 2 intervals, subgroup sizes/rates/oriented differences.')
