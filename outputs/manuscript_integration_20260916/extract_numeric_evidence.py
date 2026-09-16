"""Compact, lossless selected aggregate values with resolvable JSON pointers."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = ['incremental_q_training_20260915', 'track_a_mlp_20260916',
        'horizon_sensitivity_20260916', 'champion_class_20260916', 'definition_dev_20260916']
records = []

def walk(node, pointer, source):
    if not isinstance(node, dict):
        return
    if 'metrics_named' in node and 'cohort' in node:
        for cell in ['all', 'B40']:
            for model, metrics in node['metrics_named'].get(cell, {}).items():
                records.append(dict(source=source, pointer=pointer+['metrics_named', cell, model],
                                    values={k: metrics[k] for k in ['rows','matches','brier','auc','logloss'] if k in metrics}))
            boot = node.get('bootstrap', {}).get(cell, {})
            for i, pair in enumerate(boot.get('pairs', [])):
                records.append(dict(source=source, pointer=pointer+['bootstrap', cell, 'pairs', i],
                                    values={k: pair[k] for k in ['a','b','a_minus_b']}))
        return
    for key, value in node.items():
        walk(value, pointer+[key], source)

hashes = {}
for run in RUNS:
    path = ROOT / 'outputs' / run / 'eval/results.json'
    raw = path.read_bytes()
    source = path.relative_to(ROOT).as_posix()
    hashes[source] = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    walk(data['results'], ['results'], source)
(OUT/'numeric_evidence.json').write_text(json.dumps({'source_hashes':hashes,'records':records}, indent=2), encoding='utf-8')
lines = ['# Selected numeric evidence (generated from JSON)', '', 'Pointers in numeric_evidence.json retain exact full precision and source hashes.', '']
for row in records:
    if 'MAIN_TEST' not in row['pointer']:
        continue
    val = row['values']
    if 'brier' in val:
        text = f"Brier={val['brier']:.8f}; AUC={val['auc']:.8f}; rows={val['rows']}"
    else:
        b = val['a_minus_b']['brier']
        text = f"{val['a']}-{val['b']}: dBrier={b['estimate']:+.8f}; CI=[{b['ci95'][0]:+.8f},{b['ci95'][1]:+.8f}]"
    lines.append(f"- `{row['source']}` :: `{'/'.join(map(str,row['pointer']))}`: {text}")
(OUT/'numeric_evidence.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(f'Exported {len(records)} metric/contrast records; {len(lines)-4} MAIN_TEST lines')
