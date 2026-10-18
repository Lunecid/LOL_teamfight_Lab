import unittest,copy,json
from unittest.mock import patch
import numpy as np
import producer as p

BASE=json.loads(next(p.AUDIT.open(encoding='utf-8')));BASE['row_index']=0
D,DH=p.read(p.RAW/'detail'/f'{BASE["match_id"]}.json');T,TH=p.read(p.RAW/'timeline'/f'{BASE["match_id"]}.json')

class ProducerTests(unittest.TestCase):
 def runrow(self,row=None,detail=None,timeline=None):
  with patch.object(p,'read',side_effect=[(detail or D,DH),(timeline or T,TH)]):return p.process(copy.deepcopy(row or BASE))[0]
 def test_original_primary_query_parity(self):
  entries=[json.loads(s) for s in (p.BANK/'BANK_ENTRIES.jsonl').open()]
  lookup={e['match_id']:i for i,e in enumerate(entries)}
  row=next(r for s in p.AUDIT.open(encoding='utf-8') if (r:=json.loads(s))['match_id'] in lookup and r['disposition'].startswith('RETAIN'));row['row_index']=0;index=lookup[row['match_id']]
  with np.load(p.QUERY) as z:
   ids=np.flatnonzero(z['match_index']==index);expected={(int(z['component_id'][i]),int(z['t0_ms'][i]),int(z['first_kill_ms'][i]),int(z['last_kill_ms'][i])) for i in ids}
  actual={(r['component_id'],r['t0_ms'],r['first_kill_ms'],r['last_kill_ms']) for r in p.process(row)[0]};self.assertEqual(actual,expected);self.assertGreater(len(actual),0)
 def test_future_mask_does_not_enter_prefix(self):
  clean=self.runrow();r=copy.deepcopy(BASE);ev=[e for f in T['info']['frames'] for e in f['events']];last=max(i for i,e in enumerate(ev) if e['type']=='CHAMPION_KILL');r['event_masks']['kills:unknown_victim']=[[last,last]]
  changed=self.runrow(r)
  for a,b in zip(clean,changed):
   if a['t0_ms']<ev[last]['timestamp']:
    for k in ('player_raw','prefix_team_events','prefix_team_events_support','soul_owned'):np.testing.assert_equal(a[k],b[k])
 def test_ambiguous_boundary_masks_targets_not_input(self):
  clean=self.runrow();q=clean[0];ordinal=json.loads(q['focal_member_ordinals_json'])[0];r=copy.deepcopy(BASE);r['event_masks']['kills:ambiguous_same_timestamp_victim']=[[ordinal,ordinal]]
  changed=next(x for x in self.runrow(r) if x['row_id']==q['row_id']);self.assertFalse(changed['selection_supported'])
  for k in ('gold_supported','slice_gold_supported','cum_event_supported','slice_event_supported','terminal_supported','hazard_at_risk_supported','focal_kills_supported','pareto_supported'):self.assertFalse(changed[k].any(),k)
  for k in ('player_raw','prefix_team_events'):np.testing.assert_equal(q[k],changed[k])
 def test_unknown_terminal_not_fabricated(self):
  t=copy.deepcopy(T)
  for f in t['info']['frames']:f['events']=[e for e in f['events'] if e['type']!='GAME_END']
  r=copy.deepcopy(BASE);r['metrics']['terminal_verified']=False
  for x in self.runrow(r,timeline=t):
   self.assertFalse(x['terminal_supported'].any());self.assertTrue((x['terminal_class']==-1).all());self.assertFalse(x['hazard_at_risk_supported'].any())
 def test_field_mask_is_cell_local(self):
  r=copy.deepcopy(BASE);r['field_masks']={'armor:invalid_numeric_type':[[21,21]]};pf,tr=p.masked_frames(T['info']['frames'],r,{i:100 if i<6 else 200 for i in range(1,11)})
  self.assertTrue(np.isnan(pf[2,1,12]));self.assertTrue(np.isfinite(pf[2,2,12]))
 def test_gold_transition_crossing_later_endpoint(self):
  q=self.runrow()[0];ts=[f['timestamp'] for f in T['info']['frames']];fi=ts.index(int(q['endpoint_ms'][0]));r=copy.deepcopy(BASE);r['field_masks']={'totalGold:decrease_transition_ending_cell':[[fi*10,fi*10]]};x=next(x for x in self.runrow(r) if x['row_id']==q['row_id'])
  self.assertFalse(x['gold_supported'][:,0].any());self.assertTrue(x['gold_supported'][:,1].all())
 def test_terminal_endpoint_no_forward_fill(self):
  np.testing.assert_equal(p.endpoints(np.array([0,60000,120000,151000]),60000,151000),[120000,151000,151000])

if __name__=='__main__':unittest.main(verbosity=2)
