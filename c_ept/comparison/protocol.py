"""Stdlib-only configuration, provenance and decision contract; no model imports."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

SCHEMA = 'ept-comparison-operator-v1'
PROFILE = 'matched25_ept_b1_n2_v1'
PYTHON = Path('C:/Users/todtj/anaconda3/python.exe')
RUNS_ROOT = Path('D:/LOL_Data/runs')
MODELS = ('b1', 'ept', 'n2')
STAGES = ('prepare', 'fit', 'evaluate', 'finalize')
FIXED = {'share': .25, 'selection_seed': 20261022, 'neural_seed': 20261022,
         'b1_seed': 1, 'batch_size': 32, 'max_epochs': 30, 'patience': 2,
         'lr': .001, 'weight_decay': .01, 'warmup_steps': 10,
         'dtype': 'float32', 'tf32': False, 'amp': False, 'native_threads': 8,
         'mc_draws': 1000, 'mc_seed': 20261022, 'bootstrap': 2000,
         'bootstrap_seed': 20261022, 'prefetch': 2, 'n2_hidden': 512}
LIMITS = {'total_seconds': 104*3600, 'prepare_seconds': 4*3600,
          'fit_seconds': 72*3600, 'evaluate_seconds': 24*3600,
          'finalize_seconds': 4*3600, 'per_fit_seconds': 24*3600,
          'per_evaluate_seconds': 8*3600, 'memory_bytes': 40*1024**3}
PROTECTED = ('c_ept/cohort.py', 'research_pipeline/cohort_policy.py', 'research_pipeline/holdout_r3')
SOURCE_REQUIRED = ('c_ept/comparison/protocol.py', 'c_ept/comparison/operator.py',
                   'c_ept/comparison/runtime.py', 'c_ept/comparison/models.py',
                   'c_ept/comparison/data.py', 'c_ept/comparison/infrastructure.py',
                   'c_ept/comparison/evaluation.py', 'c_ept/comparison/reporting.py',
                   'c_ept/models/neural/team_set.py', 'c_ept/ept/train.py',
                   'c_ept/ept/data.py', 'c_ept/evaluation/adapters.py',
                   'c_ept/models/baselines_b1.py', 'tools/ept_comparison_controller.py',
                   'docs/EPT_MATCHED25_PROTOCOL_KO.md')
_HASH = re.compile(r'[0-9a-f]{64}')
_COMMIT = re.compile(r'[0-9a-f]{40}')


class ComparisonRefused(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ComparisonRefused(message)


def unique(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, f'duplicate JSON key: {k}')
        result[k] = v
    return result


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular JSON file required')
    require(path.stat().st_size <= 16*1024**2, 'metadata JSON too large')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ComparisonRefused('nonfinite JSON')))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha_file(path, *, check=None):
    if check is not None:check()
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            if check is not None:check()
            h.update(block)
    if check is not None:check()
    return h.hexdigest()


def hash_value(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def inside(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts and ':' not in str(rel), 'unsafe relative artifact path')
    target = root/rel
    require(target.resolve().is_relative_to(root) and not target.is_symlink(), 'artifact escapes root')
    return target


def validate_config(c):
    fields = {'schema', 'profile', 'source', 'source_commit', 'source_manifest',
              'source_manifest_sha256', 'python', 'output', 'inputs', 'settings',
              'limits', 'expected_scale_provenance', 'execution_authorized'}
    require(isinstance(c, dict) and set(c) == fields, 'comparison config fields mismatch')
    require(c['schema'] == SCHEMA and c['profile'] == PROFILE, 'comparison profile mismatch')
    require(c['settings'] == FIXED and all(type(c['settings'][k]) is type(v) for k,v in FIXED.items()),
            'fixed matched25 settings changed')
    require(c['limits'] == LIMITS and all(type(c['limits'][k]) is int for k in LIMITS), 'campaign limits changed')
    require(type(c['execution_authorized']) is bool, 'execution_authorized must be boolean')
    require(Path(c['python']) == PYTHON, 'fixed Python required')
    for key in ('source', 'source_manifest', 'output'):
        require(isinstance(c[key], str) and Path(c[key]).is_absolute(), f'{key} must be an explicit absolute path')
    require(c['source_commit'] is None or isinstance(c['source_commit'], str) and _COMMIT.fullmatch(c['source_commit']),
            'invalid source commit')
    require(c['source_manifest_sha256'] is None or hash_value(c['source_manifest_sha256']), 'invalid source manifest hash')
    require(isinstance(c['inputs'], dict) and set(c['inputs']) == {'s1','s2','labels','terrain','s3','prices'}, 'input identities required')
    for name, entry in c['inputs'].items():
        require(isinstance(entry, dict) and set(entry) == {'root','files'}, f'{name} input fields mismatch')
        require(isinstance(entry['root'], str) and Path(entry['root']).is_absolute(), 'explicit input root required')
        require(isinstance(entry['files'], dict) and entry['files'], 'input file identity map required')
        for rel, h in entry['files'].items():
            inside(entry['root'], rel)
            require(h is None or hash_value(h), 'invalid input hash')
    for name in ('s1','s2','labels'):
        require(set(c['inputs'][name]['files']) == {'STORE.json'}, 'store metadata identity required')
    require('B0_SCALES.json' in c['inputs']['s3']['files'], 'original S3 scale identity required')
    require({'RUN.json','STAGES.json'} <= set(c['inputs']['s3']['files']), 'S3 lineage metadata identities required')
    require(set(c['inputs']['terrain']['files']) == {'TERRAIN.json','terrain.npz','geodesic.npy'}, 'fixed terrain identities required')
    require(set(c['inputs']['prices']['files']) == {'PRICES.json'}, 'fixed prices identity required')
    require(isinstance(c['expected_scale_provenance'], dict), 'independent scale provenance required')
    return c


def load_config(path):
    return validate_config(read_json(path))


def clean_source(root, commit):
    def git(*args):
        r = subprocess.run(['git','-C',str(root),*args], capture_output=True, text=True,timeout=60)
        require(r.returncode == 0, 'source Git verification unavailable')
        return r.stdout.strip()
    require(git('rev-parse','HEAD') == commit, 'source commit mismatch')
    require(not git('status','--porcelain','--untracked-files=all'), 'clean execution checkout required')


def verify_source(c, *, source_probe=clean_source, check=None):
    if check is not None:check()
    require(c['source_commit'] is not None and c['source_manifest_sha256'] is not None, 'source bindings not finalized')
    source_probe(c['source'], c['source_commit'])
    if check is not None:check()
    require(sha_file(c['source_manifest'],check=check) == c['source_manifest_sha256'], 'reviewed source manifest differs')
    manifest = read_json(c['source_manifest'])
    files = manifest.get('files')
    require(isinstance(files, dict) and set(SOURCE_REQUIRED) <= set(files), 'incomplete comparison source closure')
    for rel, h in files.items():
        normalized=rel.replace('\\','/').casefold()
        require(not any(normalized == p or normalized.startswith(p+'/') for p in PROTECTED), 'protected source body must not be read')
        require(hash_value(h) and sha_file(inside(c['source'],rel),check=check) == h, 'reviewed source file differs: '+rel)


def preflight(c, *, source_probe=clean_source, runs_root=None, check=None):
    validate_config(c)
    verify_source(c, source_probe=source_probe,check=check)
    out = Path(c['output']).resolve()
    parent = Path(runs_root or RUNS_ROOT).resolve()
    require(out.parent == parent and re.fullmatch(r'\d{8}_c_ept_matched25_\d+',out.name), 'fresh canonical output required')
    require(not out.exists(), 'no comparison retry/resume or output reuse')
    require(not out.is_relative_to(Path(c['source']).resolve()), 'output inside source')
    for name, entry in c['inputs'].items():
        base = Path(entry['root']).resolve()
        require(not out.is_relative_to(base) and not base.is_relative_to(out), 'output/input roots overlap')
        for rel, h in entry['files'].items():
            path = inside(base, rel)
            require(hash_value(h) and path.is_file(), f'{name} input binding incomplete')
            # Check reads declared small metadata only. Payload integrity is
            # checked inside the separately authorized owned worker.
            if path.suffix.lower() == '.json':
                require(sha_file(path,check=check) == h, 'input metadata differs: '+name+'/'+rel)
    return {'status':'READY','config_sha256':digest(c), 'models':list(MODELS),
            'interpretation':'diagnostic_only', 'execution_authorized':c['execution_authorized']}


def pilot_decision(ept, comparator):
    require(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
                for v in (ept,comparator)), 'finite paired structural scores required')
    delta = float(ept)-float(comparator)
    return {'contrast':'EPT-minus-comparator','difference':delta,
            'status':'provisional_improvement' if delta < 0 else 'diagnose_before_expansion',
            'inference':'diagnostic_only','confirmed_success':False,
            'withdrawal':'no superiority claim from this single-seed pilot'}
