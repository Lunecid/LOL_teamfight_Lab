"""Snapshot/check immutable documentation and experiment evidence for integration."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ORIGINAL = Path('C:/Users/todtj/PycharmProjects/LOL_teamfight/docs/tog_manuscript')
RUNS = ['incremental_q_training_20260915', 'track_a_mlp_20260916',
        'horizon_sensitivity_20260916', 'balanced_shap_20260916',
        'v_mechanism_20260916', 'champion_class_20260916', 'definition_dev_20260916']

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def sources():
    files = set(ORIGINAL.rglob('*.tex')) | set(ORIGINAL.rglob('*.md')) | set(ORIGINAL.rglob('*.bib'))
    files.update(ROOT.joinpath('docs').glob('*20260915.md'))
    files.update(ROOT.joinpath('docs').glob('*FINDINGS_20260916.md'))
    files.update(ROOT.joinpath('scripts').glob('cc20260916_*.py'))
    files.add(ROOT / 'worktrees/engagement-state-value/gameplay/state_value_v2.py')
    files.add(ORIGINAL.parents[1] / 'config/fight_boundary/spec_pooled.json')
    for name in RUNS:
        folder = ROOT / 'outputs' / name
        for pattern in ('*.json', '*.md', 'eval/*.json', 'audit_followup/*.json'):
            files.update(folder.glob(pattern))
    return {str(p): digest(p) for p in sorted(files) if p.is_file()}

snapshot = OUT / 'source_snapshot.json'
if len(sys.argv) > 1 and sys.argv[1] == 'check':
    before = json.loads(snapshot.read_text(encoding='utf-8'))
    changed = [p for p, h in before.items() if not Path(p).is_file() or digest(Path(p)) != h]
    result = {'checked_files': len(before), 'unchanged': not changed, 'changed': changed,
              'scope': 'Named documentation, scripts and aggregate manifests/results; not a rehash of all raw data/models.'}
    (OUT / 'preservation_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    sys.exit(bool(changed))
else:
    if snapshot.exists():
        raise SystemExit('Refusing to overwrite initial snapshot')
    data = sources()
    snapshot.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Snapshotted {len(data)} evidence files')
