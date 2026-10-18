from t025_common import *
import collections
from t023_definition_oat import apply_oat
import fc20260915_common as C,data.cache_io as cio
from core.config import cfg
import gameplay.fights as F
cio.CACHE_DIR=Path(C.CACHE_MAIN);apply_oat(cfg,{})
lookup={};counts=collections.Counter()
for co in ['T','S']:
 with np.load(OUT/f'definition_ref_{co}.npz',allow_pickle=False) as z:
  lookup.update({(str(m),int(s)):(co,int(y)) for m,s,y in zip(z['match'],z['s'],z['y'])})
for mid in np.load(OUT/'definition_match_sample.npy'):
 p=cio.load_match_cache(str(mid));tm={int(k):int(v) for k,v in p['meta']['team_map'].items()}
 for f in F.detect_fights(p,tm):
  k=(str(mid),int(f['engage_ts']))
  if k not in lookup:continue
  co,y=lookup[k];ev=[e for e in p['events'] if e.get('type')=='CHAMPION_KILL' and int(f['first_kill_ts'])<=int(e['timestamp'])<=int(f['last_kill_ts'])]
  # All kills in temporal window, including simultaneous off-region events; explicitly a coarse proxy.
  kn=sum(1 if tm.get(int(e.get('killerId',0)))==100 else -1 if tm.get(int(e.get('killerId',0)))==200 else 0 for e in ev)
  aset={int(i) for e in ev for i in e.get('assistingParticipantIds',[])}
  an=sum(1 if tm.get(i)==100 else -1 if tm.get(i)==200 else 0 for i in aset)
  sign=1 if y else -1
  group='kill_tie' if kn==0 else 'kill_agree' if (kn>0)==bool(y) else 'kill_disagree'
  counts[(co,group,'n')]+=1
  counts[(co,group,'assist_tie' if an==0 else 'assist_agree' if (an>0)==bool(y) else 'assist_disagree')]+=1
out=provenance();out['scope']='Descriptive within-window association: unique assistant count difference vs SVI sign, grouped by kill-count direction. Same telemetry and temporal windows, not independent outcome truth; off-region kills in window can enter.'
out['cells']=[dict(cohort=c,kill_group=g,**{k:counts[c,g,k] for k in ['n','assist_tie','assist_agree','assist_disagree']}) for c in ['T','S'] for g in ['kill_agree','kill_disagree','kill_tie']]
dump('assist_outcome.json',out);print(out)
