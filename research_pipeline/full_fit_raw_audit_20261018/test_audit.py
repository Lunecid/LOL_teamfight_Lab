import copy, json, unittest, os
from pathlib import Path
import audit_raw as a
MID=os.environ["LOL_REVIEW_MATCH_ID"]
D=json.loads((a.RAW/'detail'/f'{MID}.json').read_bytes())
T=json.loads((a.RAW/'timeline'/f'{MID}.json').read_bytes())
class TestAudit(unittest.TestCase):
 def run_case(self,d=None,t=None):return a.analyze(MID,copy.deepcopy(d or D),copy.deepcopy(t or T))[0]
 def test_valid(self):self.assertTrue(self.run_case()['disposition'].startswith('RETAIN'))
 def test_identity(self):
  t=copy.deepcopy(T);t['metadata']['matchId']='TEST_MATCH_1';self.assertIn('match_identity_conflict',self.run_case(t=t)['hard_reasons'])
 def test_order(self):
  t=copy.deepcopy(T);t['info']['frames'][2]['timestamp']=0;self.assertIn('invalid_frame_timestamps',self.run_case(t=t)['hard_reasons'])
 def test_missing_optional(self):
  t=copy.deepcopy(T);del t['info']['frames'][2]['participantFrames']['1']['championStats']['powerMax'];r=self.run_case(t=t);self.assertTrue(r['disposition'].startswith('RETAIN'));self.assertIn('power_fraction:missing_or_undefined',r['field_masks'])
 def test_extreme_and_negative_resistance_retained(self):
  t=copy.deepcopy(T);s=t['info']['frames'][2]['participantFrames']['1']['championStats'];s.update(armor=-123,magicResist=-321,movementSpeed=50000,healthMax=50000);r=self.run_case(t=t);self.assertTrue(r['disposition'].startswith('RETAIN'));self.assertGreater(r['diagnostics']['negative_armor_retained'],0)
 def test_unknown_ward_mask(self):
  t=copy.deepcopy(T);t['info']['frames'][2]['events'].append({'type':'WARD_PLACED','timestamp':120000,'creatorId':0});r=self.run_case(t=t);self.assertIn('ward_placed:unattributed',r['event_masks']);self.assertTrue(r['disposition'].startswith('RETAIN'))
 def test_team0_soul_not_ownership(self):
  t=copy.deepcopy(T);t['info']['frames'][2]['events'].append({'type':'DRAGON_SOUL_GIVEN','timestamp':120000,'teamId':0});r=self.run_case(t=t);self.assertNotIn('soul:inconsistent_prefix',r['event_masks']);self.assertGreater(r['diagnostics']['team0_rift_announcement_ignored'],0)
 def test_missing_terminal_masks(self):
  t=copy.deepcopy(T)
  for f in t['info']['frames']:f['events']=[e for e in f['events'] if e['type']!='GAME_END']
  r=self.run_case(t=t);self.assertEqual(r['disposition'],'RETAIN_MASKED');self.assertFalse(r['metrics']['terminal_verified'])
 def test_winner_conflict(self):
  t=copy.deepcopy(T)
  for f in t['info']['frames']:
   for e in f['events']:
    if e['type']=='GAME_END':e['winningTeam']=300-e['winningTeam']
  self.assertIn('terminal_winner_or_time_conflict',self.run_case(t=t)['hard_reasons'])
 def test_existing_split_parity(self):
  p=Path(os.environ['LOL_SPLIT_ASSIGNMENTS_PATH'])
  for line in p.read_text(encoding='utf-8').splitlines():
   r=json.loads(line);self.assertEqual(a.bucket(r['match_id']),r['v5_bucket'])
 def test_masks_source_coordinates(self):self.assertEqual(a.compress([11,10,12,25,25]),[[10,12],[25,25]])
 def test_numeric_id_canonicalization(self):self.assertEqual(a.bucket('7806915326'),a.bucket(MID))
 def test_boolean_is_not_observed_number(self):
  t=copy.deepcopy(T);t['info']['frames'][2]['participantFrames']['1']['totalGold']=False;r=self.run_case(t=t);self.assertIn('totalGold:invalid_numeric_type',r['field_masks']);self.assertTrue(r['disposition'].startswith('RETAIN'))
 def test_missing_assist_is_unknown(self):
  t=copy.deepcopy(T)
  event=next(e for f in t['info']['frames'] for e in f['events'] if e['type']=='CHAMPION_KILL');event.pop('assistingParticipantIds',None)
  self.assertIn('assists:unattributed',self.run_case(t=t)['event_masks'])
 def test_unknown_victims_are_not_duplicates(self):
  t=copy.deepcopy(T)
  for killer in (1,2):t['info']['frames'][2]['events'].append({'type':'CHAMPION_KILL','timestamp':120000,'killerId':killer,'assistingParticipantIds':[]})
  self.assertNotIn('duplicate_victim_kill_same_timestamp',self.run_case(t=t)['hard_reasons'])
 def test_known_victim_collision_masked_not_quarantined(self):
  t=copy.deepcopy(T)
  for killer in (1,2):t['info']['frames'][2]['events'].append({'type':'CHAMPION_KILL','timestamp':120000,'killerId':killer,'victimId':6,'assistingParticipantIds':[]})
  r=self.run_case(t=t);self.assertTrue(r['disposition'].startswith('RETAIN'));self.assertIn('kills:ambiguous_same_timestamp_victim',r['event_masks'])
 def test_rule_mask_has_time(self):
  t=copy.deepcopy(T);t['info']['frames'][2]['events'].append({'type':'ELITE_MONSTER_KILL','timestamp':120000,'killerId':1,'killerTeamId':100,'monsterType':'BARON_NASHOR'})
  r=self.run_case(t=t);self.assertTrue(all('timestamp' in x and 'raw_ordinal' in x for x in r['rule_masks']))
 def test_unknown_end_not_last_frame_duration(self):
  t=copy.deepcopy(T)
  for f in t['info']['frames']:f['events']=[e for e in f['events'] if e['type']!='GAME_END']
  r=self.run_case(t=t);self.assertIsNone(r['metrics']['duration_s']);self.assertIsNotNone(r['metrics']['last_observed_frame_s'])
if __name__=='__main__':unittest.main(verbosity=2)

