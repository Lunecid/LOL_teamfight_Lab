"""Standard-library metadata gates. Check never opens model or dataset payloads."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

PYTHON = Path('C:/Users/todtj/anaconda3/python.exe')
TRAIN_COMMIT = 'ffdfa37c934d64002e2194e09e97c0e1bd9d46b6'
TRAIN_ID = '00af35b212db264ab32d484b3e5c93df374fc85678ba0c223176357bc75075e7'
TRAIN_CONFIG = '0eaa165f080cc7cff367bfcf2ec7dfc726aca56b1ad0b6a94db766b9e98536ec'
B1_COMMIT = '18bdea3a6e18525be8955f84f52a089980cbaef8'
SCHEMA = 'ept-dev-evaluation-v1'
FIXED = {'schema': SCHEMA, 'device': 'cuda', 'batch_size': 32, 'seed': 20261022,
         'bootstrap': 2000, 'draws': 1000, 'role': 'DEV', 'native_threads': 8}
PATHS = ('source', 'training_source', 'training_control', 'training_run', 's1', 's2', 's3', 'output')
FIELDS = set(FIXED) | set(PATHS) | {'reviewed_commit', 'definitions_sha256', 'final_audit_sha256', 'final_report_sha256'}


class EvaluationRefused(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise EvaluationRefused(message)


def unique(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'duplicate JSON key: ' + key)
        out[key] = value
    return out


def read_json(path):
    p = Path(path)
    require(p.stat().st_size <= 16 * 1024 * 1024, 'metadata exceeds bounded size')
    return json.loads(p.read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(EvaluationRefused('nonfinite JSON')))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def hash_string(value, length=64):
    require(type(value) is str and re.fullmatch('[0-9a-f]{%d}' % length, value), 'unbound/invalid hash')


def inside(root, relative):
    root, rel = Path(root).resolve(), Path(relative)
    require(type(relative) is str and not rel.is_absolute() and '..' not in rel.parts and ':' not in relative,
            'unsafe relative artifact path')
    p = root / rel
    require(p.resolve().is_relative_to(root) and not p.is_symlink(), 'artifact path escapes root')
    return p


def load_config(path):
    c = read_json(path)
    require(type(c) is dict and set(c) == FIELDS, 'evaluation config fields mismatch')
    for k, value in FIXED.items():
        require(type(c[k]) is type(value) and c[k] == value, 'fixed profile mismatch: ' + k)
    for k in PATHS:
        require(type(c[k]) is str and Path(c[k]).is_absolute(), 'absolute path required: ' + k)
        c[k] = str(Path(c[k]).resolve())
    for key in ('definitions_sha256', 'final_audit_sha256', 'final_report_sha256'):
        hash_string(c[key])
    hash_string(c['reviewed_commit'], 40)
    roots = [Path(c[k]) for k in PATHS if k != 'output']
    output = Path(c['output'])
    require(output.parent == Path('D:/LOL_Data/runs').resolve(), 'evaluation output must be a new direct runs-root child')
    require(output.name.startswith('ept_dev_'), 'evaluation output must start ept_dev_')
    require(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in roots), 'output overlaps protected source/input/run')
    require(c['source'] != c['training_source'], 'evaluation needs separate source')
    return c


def clean_source(path, expected):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    top = subprocess.run(['git', '-C', str(path), 'rev-parse', '--show-toplevel'], capture_output=True,
                         text=True, timeout=30, env=env)
    require(top.returncode == 0 and Path(top.stdout.strip()).resolve() == Path(path).resolve(),
            'source must be the exact checkout root')
    for args, want in ((['rev-parse', 'HEAD'], expected), (['status', '--porcelain=v1', '--untracked-files=all'], '')):
        p = subprocess.run(['git', '-C', str(path), *args], capture_output=True, text=True, timeout=30, env=env)
        require(p.returncode == 0 and p.stdout.strip() == want, 'source commit/whole-tree cleanliness failed')


def parent_metadata(c):
    # Run the fixed observer in its OWN interpreter. Its source-bound imports must
    # never contaminate the evaluation interpreter's c_ept module cache.
    script = """import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'tools'))
import ept_training_monitor as m
s,ct,r=map(Path,sys.argv[1:4]); cfg=m.load_config(ct/'operator-config.json')
report=m.inspect_run(r,s,cfg,control=ct)
m.require(report['state']=='completed','parent training is not normally completed; no DEV evaluation')
m.render_guard(report)
ctl=report['controller']; p=report['process']
m.require(ctl['state']=='completed' and type(ctl.get('exit_code')) is int and ctl['exit_code']==0 and type(p.get('exit_code')) is int and p['exit_code']==0,'parent exit failure')
m.require(not ctl.get('cleanup_error') and not ctl.get('manual_attention'),'parent cleanup unresolved')
folder,head,history=m.checkpoint_reader(s).latest_metadata(r/'train/S_cuda')
print(json.dumps({'report':report,'folder':str(folder),'head':head,'history':history},allow_nan=False))
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    for k in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(k, None)
    p = subprocess.run([str(PYTHON), '-B', '-c', script, c['training_source'], c['training_control'], c['training_run']],
                       cwd=c['training_source'], capture_output=True, text=True, encoding='utf-8', timeout=120, env=env)
    require(p.returncode == 0, 'parent metadata/termination guard refused: ' + p.stderr[-1600:])
    require(len(p.stdout) < 16 * 1024 * 1024, 'observer output too large')
    obj = json.loads(p.stdout)
    report = obj['report']
    require(report['code_commit'] == TRAIN_COMMIT and report['run_identity_sha256'] == TRAIN_ID
            and report['operator_config_sha256'] == TRAIN_CONFIG, 'unexpected parent identity')
    return obj


def best_reference(parent, root):
    head, hist = parent['head'], parent['history']
    require(head.get('done') is True, 'checkpoint not final')
    # History rows are the completed v6 log, not inferred from a folder timestamp.
    rows = hist['rows']
    require(len(rows) == head['epoch'] and rows, 'incomplete STOP history')
    values = [r['stop_loss'] for r in rows]
    require(all(type(v) in (int, float) and math.isfinite(v) for v in values), 'nonfinite STOP')
    best = min(range(len(values)), key=values.__getitem__)
    require(type(head.get('best_epoch')) is int and head['best_epoch'] == best, 'STOP best epoch mismatch')
    ref = head['best']
    require(ref['folder'] == f'epoch_{best+1:04d}', 'best pointer does not match earliest strict STOP minimum')
    hash_string(ref['sha256'])
    require(type(ref['bytes']) is int and ref['bytes'] > 0, 'invalid best size')
    file = inside(root, ref['folder'] + '/best.pt')
    return {'path': str(file), 'sha256': ref['sha256'], 'bytes': ref['bytes'], 'epoch': best + 1}


def verify_history_hashes(control):
    saved = read_json(Path(control) / 'previous-run-root-hashes.json')
    require(type(saved) is dict and len(saved) == 16, 'historical metadata manifest mismatch')
    allowed = {Path('D:/LOL_Data/runs/20261024_c_ept_s4a_1').resolve(),
               Path('D:/LOL_Data/runs/20261025_c_ept_s4a_1').resolve()}
    for path, expected in saved.items():
        p = Path(path).resolve()
        require(p.parent in allowed and p.suffix == '.json', 'unexpected historical metadata path')
        hash_string(expected)
        require(sha_file(p) == expected, 'historical metadata changed')
    return {'manifest_sha256': sha_file(Path(control) / 'previous-run-root-hashes.json'),
            'verified_count': len(saved)}


def input_metadata(c, parent):
    run = Path(c['training_run'])
    manifest = read_json(run / 'RUN.json')
    meta = manifest['identity']['inputs']
    for k in ('s1', 's2', 's3'):
        require(Path(meta['paths'][k]).resolve() == Path(c[k]), 'input differs from parent: ' + k)
    evidence = {}
    def bind(path, expected):
        hash_string(expected)
        require(sha_file(path) == expected, 'metadata hash mismatch: ' + str(path))
        evidence[str(Path(path).resolve())] = expected
    for key, path in (('S3RUN', Path(c['s3'])/'RUN.json'), ('S3STAGES',Path(c['s3'])/'STAGES.json'),
                      ('S1STORE',Path(c['s1'])/'STORE.json'), ('S2STORE',Path(c['s2'])/'STORE.json')):
        bind(path, meta['source_json_sha256'][key])
    s3 = read_json(Path(c['s3'])/'RUN.json')
    require(s3.get('code_commit') == B1_COMMIT and s3.get('code_dirty') is False and s3.get('smoke') is None,
            'B1 candidate source/scope mismatch')
    stages = read_json(Path(c['s3'])/'STAGES.json')['stages']
    require(stages['fit']['complete'] is True and stages['labels']['complete'] is True, 'S3 incomplete')
    b1path = Path(c['s3'])/'fit/b1/b1_heads.json'
    bind(b1path, stages['fit']['outputs']['fit/b1/b1_heads.json'])
    b1 = read_json(b1path)
    require(b1['fit_roles'] == ['FIT'] and b1['stop_roles'] == ['STOP'], 'B1 fit/STOP provenance mismatch')
    require(b1.get('schema') == 'b1_heads_v2', 'B1 manifest schema mismatch')
    for kind, group in b1['heads'].items():
        for name, entry in group.items():
            hp = inside(b1path.parent, entry['dir'] + '/head.json')
            bind(hp, entry['head_sha256'])
            doc = read_json(hp)
            require(doc.get('schema') == 'b1_head_v1' and doc.get('kind') == kind and doc.get('name') == name,
                    'B1 inner head identity mismatch')
            inner = doc['head']['files']
            require(set(inner) == set(entry['files']), 'B1 inner/outer model parts mismatch')
            for key, part in entry['files'].items():
                outer_path = inside(b1path.parent, part['file'])
                inner_path = inside(hp.parent, inner[key]['file'])
                hash_string(part['sha256'])
                require(outer_path == inner_path and part['sha256'] == inner[key]['sha256'],
                        'B1 inner/outer model binding mismatch')
    ledger = read_json(run/'STAGES.json')['stages']
    for stage, rel in (('stats','stats/STATS.json'), ('scales','B0_SCALES.json')):
        bind(run/rel, ledger[stage]['outputs'][rel])
    for k in ('prices', 'rules'):
        bind(meta['paths'][k], meta[k + '_sha256'])
    bind(Path(meta['paths']['labels'])/'LABELS.json', meta['label_provenance_sha256'])
    labelmeta = read_json(Path(meta['paths']['labels'])/'STORE.json')
    require(labelmeta.get('complete') is True and digest(labelmeta['files']) == meta['expected_provenance']['labels_store_sha256'],
            'labels metadata differs from bound S3')
    return {'inputs': meta, 'metadata_hashes': evidence, 'b1_manifest': str(b1path),
            'b1_source_commit': B1_COMMIT, 'b1_fit_scope': 'whole FIT of frozen S3; EPT uses 25%'}


def validate_final_audit(c):
    reports = Path(c['training_control'])/'reports'
    path = reports/'final-audit.json'
    require(sha_file(path) == c['final_audit_sha256'], 'final audit hash mismatch')
    audit = read_json(path)
    expected = {'schema': 'ept-pilot-final-audit-v1', 'state': 'completed',
                'code_commit': TRAIN_COMMIT, 'operator_config_sha256': TRAIN_CONFIG,
                'run_identity_sha256': TRAIN_ID, 'manual_visual_review': 'verified_against_json_csv'}
    require(type(audit) is dict, 'final audit must be an object')
    for key, value in expected.items():
        require(audit.get(key) == value, 'final audit semantic mismatch: ' + key)
    for key, config_key in (('run', 'training_run'), ('source', 'training_source')):
        require(type(audit.get(key)) is str and Path(audit[key]).resolve() == Path(c[config_key]),
                'final audit parent path mismatch: ' + key)
    require(audit.get('final_report_sha256') == c['final_report_sha256'] == sha_file(reports/'FINAL_REPORT_KO.md'),
            'final report is not bound to reviewed audit')
    artifacts = audit.get('reviewed_artifacts')
    require(type(artifacts) is dict and artifacts, 'reviewed numerical/visual artifacts required')
    suffixes = set()
    for relative, expected_hash in artifacts.items():
        artifact = inside(reports, relative)
        require(artifact.suffix in ('.png', '.csv', '.json') and artifact != path,
                'unexpected reviewed artifact')
        hash_string(expected_hash)
        require(sha_file(artifact) == expected_hash, 'reviewed artifact hash mismatch')
        suffixes.add(artifact.suffix)
    require({'.png', '.csv', '.json'} <= suffixes, 'PNG and original JSON/CSV review evidence required')
    return {'audit_sha256': c['final_audit_sha256'], 'reviewed_artifacts': artifacts}


def preflight(c, *, parent_probe=parent_metadata, source_probe=clean_source, allow_reserved=False):
    require(Path(__file__).resolve().parents[2] == Path(c['source']), 'evaluator imported from wrong source')
    source_probe(c['training_source'], TRAIN_COMMIT)
    source_probe(c['source'], c['reviewed_commit'])
    parent = parent_probe(c)  # fixed, clean observer; still before any inference imports
    if allow_reserved:
        marker = read_json(Path(c['output'])/'launch.once')
        require(marker == {'config_sha256': digest(c)}, 'reserved output launch binding mismatch')
        require(not (Path(c['output'])/'CHECK.json').exists(), 'preflight already attempted; no retry/resume')
    else:
        require(not Path(c['output']).exists(), 'fresh evaluation output required; no retry/resume')
    definitions = Path(c['source'])/'docs/EPT_DEV_DEFINITIONS_KO.md'
    require(sha_file(definitions) == c['definitions_sha256'], 'definition freeze mismatch')
    final_audit = validate_final_audit(c)
    best = best_reference(parent, Path(c['training_run'])/'train/S_cuda')
    return {'parent': parent, 'best': best, 'parent_final_audit': final_audit, **input_metadata(c,parent),
            'historical_roots': verify_history_hashes(c['training_control']), 'config_sha256': digest(c)}
