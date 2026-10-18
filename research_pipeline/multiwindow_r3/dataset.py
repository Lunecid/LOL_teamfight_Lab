"""Safe shard reader. Raw-unit inputs; fitting scalers is the trainer's FIT-only job.

No automatic row dropping for short history. Optional rule features stay off.
Never feed meta fields (identity, split, component_id) into a model.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
from holdout_r3.split import fit_role_names

FEATURE_KEYS=('t0_ms','history_ms','player_raw','player_support','champion_id','team_id',
              'summoner_ids','prefix_team_events','prefix_team_events_support',
              'skill_level_up_count','skill_level_up_count_support','soul_owned','soul_source_status')
TARGET_KEYS=('cum_team_gold','gold_supported','slice_team_gold','slice_gold_supported','net_gold',
             'economic_sign','cum_team_events','cum_event_supported','slice_team_events',
             'slice_event_supported','terminal_class','terminal_supported','hazard_class',
             'hazard_at_risk_supported','focal_team_kills','focal_kills_supported','pareto_class','pareto_supported')
META_KEYS=('row_id','match_id','component_id','patch','split_role','macro_role','endpoint_ms','actual_elapsed_ms','selection_supported')

def iter_shards(directory,split_roles,L=3):
    if L not in range(1,6):raise ValueError('L must be 1..5')
    directory=Path(directory);validation=json.loads((directory/'VALIDATION.json').read_text(encoding='utf-8'))
    if not validation['pass'] or not validation['full_population']:raise ValueError('Full ledger validation required')
    if hashlib.sha256((directory/'BUILD_SUMMARY.json').read_bytes()).hexdigest()!=validation['build_summary_sha256']:raise ValueError('Build summary changed after validation')
    s=json.loads((directory/'BUILD_SUMMARY.json').read_text(encoding='utf-8'))
    requested=set(split_roles)
    if not requested or not requested<=fit_role_names():raise ValueError('This ledger reader accepts configured FIT roles only')
    for sn in s['shards']:
        if hashlib.sha256((directory/'shards'/sn).read_bytes()).hexdigest()!=validation['shard_sha256'][sn]:raise ValueError('Shard changed after validation: '+sn)
        with np.load(directory/'shards'/sn,allow_pickle=False) as z:
            keep=np.isin(z['split_role'],list(requested))
            if not keep.any():continue
            x={k:z[k][keep] for k in FEATURE_KEYS};y={k:z[k][keep] for k in TARGET_KEYS};meta={k:z[k][keep] for k in META_KEYS}
        for key in ('history_ms','player_raw','player_support'):x[key]=x[key][:,-L:]
        x['history_present']=x['history_ms']>=0
        y['economic_supported']=np.isfinite(y['net_gold'])
        y['economic_class']=np.where(y['economic_supported'],y['economic_sign']+1,-1).astype(np.int8)
        # Values remain NaN until the training pipeline fits its transformations.
        # Losses must select supported targets before arithmetic (NaN * 0 is NaN).
        yield x,y,meta
