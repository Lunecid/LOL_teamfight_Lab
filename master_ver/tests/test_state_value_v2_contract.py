import copy
import os
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'worktrees/engagement-state-value'))
os.environ['LOL_OUTPUT_ROOT'] = str(ROOT / 'outputs/state_value_v2_fix/runtime')
from gameplay.state_value import StateBuilder as Legacy, SNAPSHOT_FIELDS
from gameplay.state_value_v2 import StateBuilder, soul_element, state_matrix

def fixture():
    names = list(SNAPSHOT_FIELDS) + ['champion_id']
    node = np.zeros((3, 10, len(names)))
    for pid in range(1, 11):node[:, pid-1, :] = pid
    return dict(minute_ts=np.array([0, 60000, 120000]), node_minute=node,
                meta=dict(team_map={pid:100 if pid<=5 else 200 for pid in range(1,11)},
                          role_slots={pid:pid-1 for pid in range(1,11)}), events=[]), names

class ContractTests(unittest.TestCase):
    def test_all_observed_soul_names_and_legacy_aliases(self):
        for name,expected in [('Cloud','AIR'),('Mountain','EARTH'),('Infernal','FIRE'),('Ocean','WATER'),('Hextech','HEXTECH'),('Chemtech','CHEMTECH')]:
            for field in ['name','dragonSoul','soulType']:
                with self.subTest(name=name,field=field):
                    p,n=fixture();p['events']=[dict(type='DRAGON_SOUL_GIVEN',timestamp=60000,teamId=100,**{field:name})]
                    b=StateBuilder(p,n)
                    self.assertEqual(b.at(59999).values['blue_soul_'+expected],0)
                    self.assertEqual(b.at(60000).values['blue_soul_'+expected],1)
    def test_unassigned_is_not_team_acquisition(self):
        p,n=fixture();p['events']=[dict(type='DRAGON_SOUL_GIVEN',timestamp=60000,teamId=0,name='Cloud')]
        b=StateBuilder(p,n);a=b.at(60000)
        self.assertEqual(a.unassigned_soul_events,1)
        self.assertEqual(a.values['blue_soul_event_recorded']+a.values['red_soul_event_recorded'],0)
        self.assertEqual(a.values['unknown_objective_team_count'],0)
    def test_role_metadata_permutation_deletion_and_mutation_invariant(self):
        p,n=fixture();base=StateBuilder(p,n).at(90000).values
        for role in [None,{}, {pid:(4-(pid-1)) if pid<=5 else (14-pid) for pid in range(1,11)}, {'invalid':'future label'}]:
            q=copy.deepcopy(p);q['meta']['role_slots']=role
            self.assertEqual(StateBuilder(q,n).at(90000).values,base)
        self.assertEqual(base['participant_slot0_totalGold_norm'],1)
        self.assertFalse(any(k.startswith('slot') for k in base))
    def test_future_events_and_outcome_do_not_change_pre_state(self):
        p,n=fixture();before=StateBuilder(p,n).at(59999)
        p['events']=[dict(type='GAME_END',timestamp=120000,winningTeam=100),dict(type='DRAGON_SOUL_GIVEN',timestamp=90000,teamId=100,name='Cloud',dragonSoul='FIRE')]
        self.assertEqual(before.values,StateBuilder(p,n).at(59999).values)
        with self.assertRaises(ValueError):StateBuilder(p,n).at(90000)
    def test_source_unmodified_and_unknown_element_retained(self):
        p,n=fixture();p['events']=[dict(type='DRAGON_SOUL_GIVEN',timestamp=60000,teamId=200,name='NewElement')]
        meta=copy.deepcopy(p['meta']);events=copy.deepcopy(p['events']);a=StateBuilder(p,n).at(60000)
        self.assertEqual(a.values['red_soul_OTHER'],1);self.assertEqual(meta,p['meta']);self.assertEqual(events,p['events'])
    def test_schema_guard_rejects_legacy_and_wrong_order(self):
        p,n=fixture();a=StateBuilder(p,n).at(60000);keys=list(a.values)
        self.assertEqual(state_matrix([a],keys).shape,(1,len(keys)))
        with self.assertRaises(ValueError):state_matrix([Legacy(p,n).at(60000)],keys)
        with self.assertRaises(ValueError):state_matrix([a],list(reversed(keys)))

if __name__=='__main__':unittest.main()
