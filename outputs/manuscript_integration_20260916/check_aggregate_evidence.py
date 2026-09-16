"""Check retained aggregate provenance and numerical contrast consistency only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
checks = []

def record(name, ok, detail=None):
    checks.append({'name': name, 'passed': bool(ok), 'detail': detail})

runs = {'track_a_mlp':115, 'horizon_sensitivity':91, 'balanced_shap':36,
        'v_mechanism':24, 'champion_class':91, 'definition_dev':53}
for name, n in runs.items():
    folder = ROOT/'outputs'/f'{name}_20260916'
    report = json.loads((folder/'report_manifest.json').read_text(encoding='utf-8'))
    validation = json.loads((folder/'validation.json').read_text(encoding='utf-8'))
    record(name+'/saved_final_validation', validation.get('passed')==n and not validation.get('failed'),
           'Saved check outcome, not a rerun of the original experiment gates')
    result_file = folder/('shap/shap_summary.json' if name=='balanced_shap' else 'results.json' if name=='v_mechanism' else 'eval/results.json')
    key = 'shap_summary_sha256' if name=='balanced_shap' else 'results_sha256'
    record(name+'/reported_result_hash', hashlib.sha256(result_file.read_bytes()).hexdigest()==report[key])

index = json.loads((OUT/'numeric_evidence.json').read_text(encoding='utf-8'))
cache = {}
for source, expected in index['source_hashes'].items():
    raw = (ROOT/source).read_bytes()
    record(source+'/extraction_hash', hashlib.sha256(raw).hexdigest()==expected)
    cache[source] = json.loads(raw)
for i,row in enumerate(index['records']):
    node = cache[row['source']]
    for key in row['pointer']:
        node = node[key]
    record(f'pointer/{i}', all(node[k]==v for k,v in row['values'].items()))

def contrasts(node, where):
    if not isinstance(node, dict):
        return
    if 'metrics_named' in node:
        for cell, bootstrap in node.get('bootstrap',{}).items():
            metrics = node['metrics_named'].get(cell,{})
            for pair in bootstrap.get('pairs',[]):
                a,b = pair['a'],pair['b']
                if a not in metrics or b not in metrics:
                    continue
                for metric in ['brier','auc','logloss']:
                    x,y=metrics[a].get(metric),metrics[b].get(metric)
                    value=pair['a_minus_b'].get(metric,{}).get('estimate')
                    if x is not None and y is not None and value is not None:
                        record(f'{where}/{cell}/{a}-{b}/{metric}', abs((x-y)-value)<1e-9)
        return
    for key,val in node.items():
        contrasts(val, where+'/'+key)

for source,data in cache.items():
    contrasts(data['results'],source)
summary={'scope':'Read-only aggregate hash, JSON-pointer and contrast checks; no training or original gate rerun',
         'checks':len(checks),'passed':sum(c['passed'] for c in checks),'failed':[c for c in checks if not c['passed']],
         'recorded_original_checks':sum(runs.values())}
(OUT/'aggregate_verification.json').write_text(json.dumps({'summary':summary,'checks':checks},indent=2),encoding='utf-8')
print(json.dumps(summary))
raise SystemExit(bool(summary['failed']))
