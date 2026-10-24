import copy,json,unittest,os
from pathlib import Path
from split import roles,bucket
import audit_core as a
class ScopeTests(unittest.TestCase):
 def test_all_frozen_assignments(self):
  n=0
  for line in Path(os.environ['LOL_SPLIT_ASSIGNMENTS_PATH']).read_text(encoding='utf-8').splitlines():
   r=json.loads(line);b=bucket(r['match_id']);self.assertEqual(b,r['v5_bucket'])
   actual=roles(r['patch'],b)
   self.assertEqual(actual[0],r['v5_split_role']);n+=1
  self.assertGreater(n,0)
 def test_boundaries(self):
  self.assertEqual(roles('15.20',79),('FIT_15.20','FIT'))
  self.assertEqual(roles('15.20',80),('STOP_15.20','STOP'))
  for b,role in [(49,'DEV'),(50,'CAL'),(74,'CAL'),(75,'AUDIT')]:self.assertEqual(roles('15.21',b)[1],role)
  self.assertEqual(roles('15.22',0)[1], 'TEST');self.assertEqual(roles('15.22',99)[1], 'TEST')
 def test_soul_conflict_has_prefix_onset(self):
  mid=os.environ["LOL_REVIEW_MATCH_ID"];d=json.loads((a.RAW/'detail'/f'{mid}.json').read_bytes());t=json.loads((a.RAW/'timeline'/f'{mid}.json').read_bytes())
  for f in t['info']['frames']:f['events']=[e for e in f['events'] if e['type'] not in ('ELITE_MONSTER_KILL','DRAGON_SOUL_GIVEN')]
  es=t['info']['frames'][2]['events']
  for z in (100,200):
   for j in range(4):es.append({'type':'ELITE_MONSTER_KILL','timestamp':100000+(z-100)*10+j,'killerTeamId':z,'killerId':1 if z==100 else 6,'monsterType':'DRAGON','monsterSubType':'FIRE_DRAGON','assistingParticipantIds':[]})
  row,_=a.analyze(mid,d,t)
  rr=row['event_masks']['soul:both_teams_have_four_dragons'];self.assertEqual(len(rr),1)
  i,j=rr[0];self.assertEqual(i,j);self.assertGreaterEqual(i,0)
  raw=[e for f in t['info']['frames'] for e in f['events']];self.assertEqual(raw[i]['timestamp'],101003)
if __name__=='__main__':unittest.main(verbosity=2)

