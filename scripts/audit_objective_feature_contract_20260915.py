"""Independent schema/source inventory for the explicit objective-channel ablation."""
from pathlib import Path
import hashlib
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/label_validity_full_20260915/feature_lists.json'
TOKENS = ('baron', 'elder', 'dragon', 'soul', 'herald', 'horde', 'atakhan')
names = json.loads(SOURCE.read_text(encoding='utf8'))['state_names']
original = [n for n in names if n != 'snapshot_age_s']
removed = [n for n in original if any(t in n.lower() for t in TOKENS)]
retained = [n for n in original if n not in removed]
assert (len(original), len(removed), len(retained)) == (361, 176, 185)
assert len(set(original)) == len(original)
assert 'unknown_objective_team_count' in retained
assert sum(n.endswith('_champion_id') for n in retained) == 10
assert sum(n.startswith('participant_slot') for n in removed) == 20
assert all(n in retained for n in ('time_minutes', 'time_minutes_sq', 'blue_kills',
    'red_kills', 'blue_tower_OUTER_TURRET', 'red_inhibitor_kills'))
out = {
    'scope': 'schema contract only; no outcomes accessed, fits or validity claims',
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'original_count': len(original), 'removed_count': len(removed),
    'retained_count': len(retained), 'original': original, 'removed': removed,
    'retained': retained,
    'removed_by_keyword': dict(Counter(next(t for t in TOKENS if t in n.lower()) for n in removed)),
    'retained_mixed_diagnostic': 'unknown_objective_team_count: includes unknown elite-monster and structure teams; unassigned souls filtered by V2 before legacy builder. Not a pure missing-structure count.',
    'proxy_limit': 'Retained economy, growth, combat, structures and time can encode indirect objective information.',
    'passed': True,
}
base = ROOT / 'outputs/claude_dispatch_objective_ablation'
base.mkdir(parents=True, exist_ok=True)
(base / 'codex_feature_contract.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({k: out[k] for k in ('passed', 'original_count', 'removed_count', 'retained_count', 'removed_by_keyword')}))
