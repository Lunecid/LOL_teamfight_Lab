"""EPT-T002 metadata-only observer. Never reads model payloads or changes a run.

Root stage records and raw train/S_cuda sidecars remain the source of truth.
Reports live outside the run. Native plotting is imported only after termination.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys

MAX_BYTES = 4 * 1024 * 1024
SNAPSHOT_ATTEMPTS = 3
SNAPSHOT_BYTES = 32 * 1024 * 1024
STAGES = ['scales', 'stats', 'train_S_cuda']
RUN_SCHEMA = 'ept_training_run_v1'
PROCESS_SCHEMA = 'ept_training_process_v1'
REPORT_SCHEMA = 'ept-training-observation-v1'
PARAMETERS = {'size': 'S', 'device': 'cuda', 'max_epochs': 30, 'share': .25,
              'patience': 2, 'batch_size': 32, 'seed': 20261022,
              'lr': .001, 'weight_decay': .01, 'warmup_steps': 10}
CONFIG_FIELDS = {'schema', 'profile', 'python', 's1', 's2', 's3', 'runs_root',
                 'catalog_db', *PARAMETERS}
PROCESS_FIELDS = {'schema', 'operator_api_version', 'run_path', 'run_identity_sha256',
                  'code_commit', 'operator_config_sha256', 'pid', 'create_time',
                  'state', 'outcome', 'exit_code', 'updated_at'}
REF_FIELDS = {'run', 'code_commit', 'operator_config_sha256', 'run_identity_sha256'}


class MetadataError(ValueError):
    pass


class ObservationError(MetadataError):
    """Invalid/unconfirmed observation is not evidence that training failed."""
    def __init__(self, message, *, evidence, attempts, stable):
        super().__init__(message)
        self.evidence, self.attempts, self.stable = evidence, attempts, stable


class MetadataSnapshot:
    """Bounded optimistic read: fixed metadata bytes must survive acquisition.

    Each retry starts over, including RUN and the ledger. No writer/lock changes,
    payload reads, sleep loop, or fallback to an older completed checkpoint.
    """
    def __init__(self):
        self.files, self.directories, self.evidence = {}, {}, {}
        self.bytes_read = 0

    def raw(self, path):
        path = Path(path)
        if path not in self.files:
            try:
                raw = read_bytes(path)
            except FileNotFoundError:
                raw = None
            self.bytes_read += 0 if raw is None else len(raw)
            require(self.bytes_read <= SNAPSHOT_BYTES, 'metadata snapshot byte budget exceeded')
            self.files[path] = raw
        return self.files[path]

    def present(self, path):
        return self.raw(path) is not None

    def json(self, path):
        before = self.raw(path)
        require(before is not None, 'metadata missing: ' + Path(path).name)
        # Keep the same strict parser (and exact independent probe seam).
        doc = read_json(path)
        require(read_bytes(path) == before, 'metadata generation changed: ' + Path(path).name)
        return doc

    def listing(self, directory, pattern):
        key = (Path(directory), pattern)
        paths = sorted(key[0].glob(pattern))
        require(len(paths) <= 2 * PARAMETERS['max_epochs'] + 2, 'metadata directory entry budget exceeded')
        self.directories[key] = tuple(p.name for p in paths)
        return paths

    def stable(self):
        try:
            for (folder, pattern), names in self.directories.items():
                if tuple(p.name for p in sorted(folder.glob(pattern))) != names:
                    return False
            for path, before in self.files.items():
                try:
                    after = read_bytes(path)
                except FileNotFoundError:
                    after = None
                if before != after:
                    return False
        except (OSError, MetadataError):
            return False
        return True


def require(condition, message):
    if not condition:
        raise MetadataError(message)


def digest(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode('utf-8')).hexdigest()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _unique(pairs):
    doc = {}
    for key, value in pairs:
        require(key not in doc, 'duplicate JSON key')
        doc[key] = value
    return doc


def read_bytes(path):
    require(not Path(path).is_symlink(), 'metadata symlink refused: ' + Path(path).name)
    with Path(path).open('rb') as handle:
        raw = handle.read(MAX_BYTES + 1)
    require(len(raw) <= MAX_BYTES, 'metadata exceeds bounded read: ' + Path(path).name)
    return raw


def read_json(path):
    try:
        doc = json.loads(read_bytes(path).decode('utf-8-sig'), object_pairs_hook=_unique,
                         parse_constant=lambda value: (_ for _ in ()).throw(MetadataError('nonfinite JSON')))
        require(type(doc) is dict, 'metadata must be an object')
        # Also rejects overflowed floating literals such as 1e999.
        json.dumps(doc, allow_nan=False)
        return doc
    except (OSError, UnicodeError, ValueError) as exc:
        raise MetadataError(f'{Path(path).name}: metadata read failed: {exc}') from exc


def canonical(value):
    require(type(value) is str and Path(value).is_absolute(), 'canonical absolute path required')
    p = Path(value).resolve()
    require(str(p) == value, 'noncanonical path: ' + value)
    return p


def hex_field(value, length=64):
    require(type(value) is str and re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is not None,
            'invalid hash field')
    return value


def same_value(actual, expected):
    # Floats may be represented as integer literals only for numeric parameters.
    return (type(actual) is type(expected) or type(expected) is float and type(actual) in (int, float)) and actual == expected


def load_config(path):
    c = read_json(path)
    require(set(c) == CONFIG_FIELDS, 'operator config exact fields mismatch')
    require(c['schema'] == 'ept-training-operator-v1' and c['profile'] == 's25_early_stop', 'operator profile mismatch')
    require(all(same_value(c[k], v) for k, v in PARAMETERS.items()), 'training parameters outside authorized tuple')
    for k in ('python', 's1', 's2', 's3', 'runs_root', 'catalog_db'):
        canonical(c[k])
    return c


def validate_run(m, run, source, config):
    require(m.get('schema') == RUN_SCHEMA and m.get('stage') == 'ept_training_single'
            and type(m.get('operator_api_version')) is int and m['operator_api_version'] == 1, 'RUN schema/stage/API mismatch')
    identity = m.get('identity')
    require(type(identity) is dict and set(identity) == {'args', 'operator_config_sha256', 'code', 'runtime', 'inputs'}, 'RUN identity fields mismatch')
    require(all(type(identity[k]) is dict for k in ('args', 'code', 'runtime', 'inputs')), 'RUN identity object mismatch')
    code = identity['code']
    require(set(code) == {'code_commit', 'code_dirty', 'repo_root'} and code['code_dirty'] is False, 'RUN source is not clean')
    hex_field(code['code_commit'], 40)
    require(canonical(code['repo_root']) == source and m.get('code_dirty') is False
            and m.get('code_commit') == code['code_commit'], 'RUN code binding mismatch')
    require(m.get('operator_config_sha256') == identity['operator_config_sha256'] == digest(config), 'RUN config canonical hash mismatch')
    require(m.get('run_identity_sha256') == digest(identity) and m.get('runtime') == identity['runtime'], 'RUN canonical identity/runtime mismatch')
    args = identity['args']
    for k in ('s1', 's2', 's3', 'runs_root', 'catalog_db'):
        require(args.get(k) == config[k], 'RUN argument mismatch: ' + k)
    require(args == config and set(args) == CONFIG_FIELDS and digest(args) == digest(config), 'RUN arguments/config mismatch')
    require(run.parent == canonical(config['runs_root']), 'RUN outside configured root')
    require(m.get('stages') == STAGES and type(m.get('results')) is dict
            and set(m['results']) <= set(STAGES), 'RUN stage selection mismatch')
    return identity


def validate_ref(ref, config, source, run=None):
    require(set(ref) == REF_FIELDS, 'control run ref exact fields mismatch')
    path = canonical(ref['run'])
    require(re.fullmatch(r'\d{8}_c_ept_ept_pilot_\d+', path.name) is not None, 'control ref outside pilot run family')
    require(path.parent == canonical(config['runs_root']) and (run is None or path == run), 'control exact run ref mismatch')
    hex_field(ref['code_commit'], 40)
    hex_field(ref['run_identity_sha256'])
    require(ref['operator_config_sha256'] == digest(config), 'control config hash mismatch')
    return path


def observe_process(pid):
    # Process liveness uses psutil; metadata and checkpoint parsing use stdlib only.
    try:
        import psutil
        p = psutil.Process(pid)
        if not p.is_running() or p.status() == psutil.STATUS_ZOMBIE:
            return {'kind': 'missing'}
        return {'kind': 'observed', 'create_time': p.create_time()}
    except ImportError:
        return {'kind': 'unavailable'}
    except psutil.NoSuchProcess:
        return {'kind': 'missing'}
    except (psutil.Error, OSError, ValueError):
        return {'kind': 'unavailable'}


def generation_state(pid, created, probe=observe_process):
    require(type(pid) is int and pid > 0 and type(created) is float and math.isfinite(created) and created > 0, 'invalid PID generation')
    observed = probe(pid)
    if observed['kind'] == 'missing':
        return 'gone'
    if observed['kind'] == 'observed':
        return 'alive' if observed.get('create_time') == created else 'pid_reused'
    return 'unconfirmed'


def validate_process(p, run, m):
    require(set(p) == PROCESS_FIELDS and p.get('schema') == PROCESS_SCHEMA
            and type(p.get('operator_api_version')) is int and p['operator_api_version'] == 1, 'PROCESS exact fields/schema mismatch')
    require(canonical(p['run_path']) == run and all(p[k] == m[k] for k in REF_FIELDS - {'run'}), 'PROCESS binding mismatch')
    stamp = dt.datetime.fromisoformat(p['updated_at'])
    require(stamp.tzinfo is not None and stamp.utcoffset() == dt.timedelta(0), 'PROCESS timestamp must be UTC')
    state, outcome, code = p['state'], p['outcome'], p['exit_code']
    require((state == 'running' and outcome is None and code is None)
            or (state == 'finished' and outcome == 'completed' and type(code) is int and code == 0)
            or (state == 'failed' and outcome == 'failed' and (code is None or type(code) is int and code != 0)), 'PROCESS invalid terminal combination')


def checkpoint_reader(source):
    """Import the reviewed stdlib helper from this explicit source, never ambient cwd."""
    source = canonical(str(source))
    sys.path.insert(0, str(source))
    try:
        module = importlib.import_module('c_ept.ept.checkpoints')
        metrics = importlib.import_module('c_ept.ept.epoch_metrics')
    finally:
        sys.path.pop(0)
    require(Path(module.__file__).resolve() == source / 'c_ept/ept/checkpoints.py'
            and Path(metrics.__file__).resolve() == source / 'c_ept/ept/epoch_metrics.py', 'checkpoint helper source binding mismatch')
    return module


def validate_epoch_paths(train):
    """Keep the reviewed helper's fixed-name reads inside the explicit run."""
    require(train.resolve() == train, 'training metadata path escapes the bound run')
    folders = list(train.glob('epoch_[0-9][0-9][0-9][0-9]'))
    require(len(folders) <= PARAMETERS['max_epochs'], 'epoch folder count exceeds pilot horizon')
    for folder in folders:
        require(not folder.is_symlink() and folder.resolve().parent == train, 'epoch folder escape')
        for name in ('head.json', 'epoch_metrics.json', 'epoch_metrics.csv'):
            require(not (folder / name).is_symlink(), 'epoch metadata symlink refused')


def validate_stages(run, manifest, snapshot=None):
    read = read_json if snapshot is None else snapshot.json
    raw_read = read_bytes if snapshot is None else snapshot.raw
    path = run / 'STAGES.json'
    if not (path.exists() if snapshot is None else snapshot.present(path)):
        require(not manifest['results'], 'RUN results without stage ledger')
        return {}
    ledger = read(path)
    stages = ledger.get('stages')
    require(ledger.get('schema') == 's3_stages_v1' and type(stages) is dict
            and set(stages) <= set(STAGES) and type(ledger.get('calls')) is list, 'STAGES schema mismatch')
    gap = False
    for name in STAGES:
        record = stages.get(name)
        if record is None:
            gap = True
            continue
        require(type(record) is dict and type(record.get('complete')) is bool, 'stage completion flag mismatch')
        expected = digest({'identity': manifest['identity'], 'stage': name})
        require(record.get('inputs_sha256') == expected and record.get('inputs', {}).get('run_identity_sha256') == manifest['run_identity_sha256'], 'stage identity mismatch')
        if not record['complete']:
            gap = True
            continue
        require(not gap, 'completed stage follows incomplete stage')
        outputs = record.get('outputs')
        require(type(outputs) is dict and name + '.json' in outputs, 'stage root hash missing')
        # Validate paths/hash syntax only for payload entries: never open them.
        for rel, value in outputs.items():
            require(type(rel) is str and not Path(rel).is_absolute() and '..' not in Path(rel).parts and ':' not in rel, 'stage output path escape')
            hex_field(value)
        raw = raw_read(run / (name + '.json'))
        require(raw is not None, 'completed stage root missing')
        require(sha(raw) == outputs[name + '.json'], 'stage root SHA mismatch')
        doc = read(run / (name + '.json'))
        require(doc.get('schema') == 's4a_stage_v1' and doc.get('stage') == name
                and doc.get('inputs_sha256') == expected and doc == manifest['results'].get(name), 'stage/RUN results mismatch')
    return stages


def validate_training_identity(head, config):
    identity = head['identity']
    for key in ('seed', 'lr', 'weight_decay', 'warmup_steps', 'patience', 'device'):
        require(same_value(identity.get(key), config[key]), 'train identity parameter mismatch: ' + key)
    require(type(identity.get('epochs')) is int and identity['epochs'] == config['max_epochs'], 'train horizon mismatch')
    require(identity.get('dtype') == 'torch.float32' and identity.get('threads') == 8
            and identity.get('deterministic_algorithms') is True, 'train runtime identity mismatch')
    encoder = identity.get('encoder', {})
    require(encoder == {'d': 128, 'layers': 4, 'heads': 4, 'dropout': .1, 'kappas': 3, 'field_channels': 8}, 'train S encoder mismatch')
    for role in ('fit', 'stop'):
        rows = identity.get(role, {})
        require(rows.get('role') == role.upper() and rows.get('training') is (role == 'fit')
                and type(rows.get('batch_size')) is int and rows['batch_size'] == config['batch_size']
                and type(rows.get('seed')) is int and rows['seed'] == config['seed'], 'train FIT/STOP identity mismatch')


def validate_final_call(train, folder, head, snapshot=None):
    read = read_json if snapshot is None else snapshot.json
    raw_read = read_bytes if snapshot is None else snapshot.raw
    pending_path = train / 'calls/call_0001.pending.json'
    final_path = train / 'calls/call_0001.json'
    # A single fresh training call; any second marker proves a contract deviation.
    calls = train / 'calls'
    paths = calls.glob('call_*.json') if snapshot is None else snapshot.listing(calls, 'call_*.json')
    names = {p.name for p in paths}
    require(names == {pending_path.name, final_path.name}, 'final call missing or more than one attempt')
    pending, final = read(pending_path), read(final_path)
    identity_hash = sha(json.dumps(head['identity'], sort_keys=True).encode())
    for doc, kind in ((pending, 'pending'), (final, 'final')):
        require(doc.get('schema') == 'ept_train_checkpoint_v5' and doc.get('kind') == kind
                and type(doc.get('call')) is int and doc['call'] == 1
                and doc.get('identity_sha256') == identity_hash and doc.get('epoch_done_start') == 0, 'call identity/attempt mismatch')
    require(final.get('pending_sha256') == sha(raw_read(pending_path))
            and final.get('head_sha256_end') == sha(raw_read(folder / 'head.json'))
            and final.get('epoch_done_end') == head['epoch']
            and final.get('outcome') == 'completed' and final.get('status') == 'ok', 'final call/head/outcome mismatch')
    return final


def validate_controller(controller, source, config, commit=None):
    require(controller.get('schema') == 'ept-training-controller-v1'
            and controller.get('source') == str(source)
            and controller.get('operator_config_sha256') == digest(config), 'controller source/config/schema mismatch')
    hex_field(controller.get('code_commit'), 40)
    require(commit is None or controller['code_commit'] == commit, 'controller code mismatch')
    require(controller.get('state') in ('launching', 'running', 'completed', 'failed'), 'controller state mismatch')


def _inspect_run_once(run, source, config, snapshot, *, control=None, probe=observe_process):
    run, source = canonical(str(run)), canonical(str(source))
    m = snapshot.json(run / 'RUN.json')
    validate_run(m, run, source, config)
    controller = None
    if control is not None:
        ref = snapshot.json(control / 'run.json')
        validate_ref(ref, config, source, run)
        require(all(ref[k] == m[k] for k in REF_FIELDS - {'run'}), 'run ref identity mismatch')
        controller = snapshot.json(control / 'controller.json') if snapshot.present(control / 'controller.json') else None
        if controller is not None:
            validate_controller(controller, source, config, m['code_commit'])
    snapshot.evidence.update({k: m[k] for k in REF_FIELDS - {'run'}})
    snapshot.evidence['controller'] = controller
    p = snapshot.json(run / 'PROCESS.json') if snapshot.present(run / 'PROCESS.json') else None
    state, generation = 'bootstrapping', 'unconfirmed'
    if p is not None:
        validate_process(p, run, m)
        generation = generation_state(p['pid'], p['create_time'], probe)
        snapshot.evidence.update(process=p, process_generation=generation, actual_failed=p['state'] == 'failed')
        if p['state'] == 'running':
            state = 'running' if generation == 'alive' else 'interrupted_unconfirmed'
        else:
            state = 'completed' if p['state'] == 'finished' else 'failed'
            if generation == 'alive':
                state = 'terminal_process_alive'
            elif generation == 'unconfirmed':
                state = 'terminal_unconfirmed'
    if controller is not None and controller.get('state') == 'failed':
        state = 'failed'
        snapshot.evidence['actual_failed'] = True
    elif controller is not None and controller.get('state') in ('running', 'launching'):
        if generation_state(controller['controller_pid'], controller['controller_create_time'], probe) != 'alive':
            state = 'interrupted_unconfirmed'
    snapshot.evidence['observed_state'] = state
    stages = validate_stages(run, m, snapshot)
    validate_epoch_paths(run / 'train/S_cuda')
    folders = snapshot.listing(run / 'train/S_cuda', 'epoch_[0-9][0-9][0-9][0-9]')
    committed = [folder for folder in folders if snapshot.present(folder / 'head.json')]
    if committed:
        for name in ('epoch_metrics.json', 'epoch_metrics.csv'):
            snapshot.raw(committed[-1] / name)
    found = checkpoint_reader(source).latest_metadata(run / 'train/S_cuda')
    history, head, final = None, None, None
    if found is not None:
        folder, head, history = found
        require(head['schema'] == 'ept_train_checkpoint_v6' and history is not None, 'v6 epoch history required')
        validate_training_identity(head, config)
    if state == 'completed':
        require(found is not None and head.get('done') is True
                and all(stages.get(k, {}).get('complete') is True for k in STAGES), 'terminal success lacks completed stages/head')
        final = validate_final_call(run / 'train/S_cuda', folder, head, snapshot)
        outputs = stages['train_S_cuda']['outputs']
        for path in (folder / 'head.json', folder / 'epoch_metrics.json', folder / 'epoch_metrics.csv',
                     run / 'train/S_cuda/calls/call_0001.pending.json', run / 'train/S_cuda/calls/call_0001.json'):
            require(outputs.get(path.relative_to(run).as_posix()) == sha(snapshot.raw(path)), 'final stage/raw train metadata hash mismatch')
        data = m['results']['train_S_cuda']['data']
        require(data.get('size') == 'S' and data.get('device') == 'cuda'
                and data.get('seed') == config['seed'] and data.get('epochs_run') == head['epoch']
                and data.get('best_epoch') == head['best_epoch']
                and data.get('resumed') is False and data.get('recovered') is False, 'training result/head mismatch')
    return {'schema': REPORT_SCHEMA, 'run': str(run), 'source': str(source),
            'code_commit': m['code_commit'], 'operator_config_sha256': m['operator_config_sha256'],
            'run_identity_sha256': m['run_identity_sha256'], 'state': state,
            'process_generation': generation, 'process': p, 'controller': controller,
            'epoch': 0 if head is None else head['epoch'], 'history': history,
            'training_parameters': {**PARAMETERS, 'dtype': 'float32', 'optimizer': 'AdamW',
                                    'monitor': 'STOP strict improvement/min_delta=0', 'native_threads': 8},
            'final_call': final, 'stage_resources': {k: v.get('resources') for k, v in m['results'].items()},
            'api_local_tokens': None, 'cost': None,
            'limitations': 'Metadata verification only; no payload verification. Host private memory is sampled, not a continuous bound. FIT and STOP have different definitions.'}


def inspect_run(run, source, config, *, control=None, probe=observe_process):
    diagnostics = []
    for attempt in range(1, SNAPSHOT_ATTEMPTS + 1):
        snapshot = MetadataSnapshot()
        try:
            report = _inspect_run_once(run, source, config, snapshot, control=control, probe=probe)
            require(snapshot.stable(), 'metadata generation changed during snapshot acquisition')
            report['observation'] = {'attempts': attempt, 'consistent_snapshot': True}
            return report
        except (MetadataError, RuntimeError, OSError, KeyError, TypeError, ValueError) as exc:
            stable = snapshot.stable()
            diagnostics.append({'attempt': attempt, 'stable_bytes': stable, 'error': str(exc)})
    raise ObservationError(diagnostics[-1]['error'],
                           evidence={**snapshot.evidence, 'diagnostics': diagnostics},
                           attempts=SNAPSHOT_ATTEMPTS, stable=stable)


def atomic_json(path, doc):
    raw = (json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    atomic_bytes(path, raw)


def atomic_bytes(path, raw):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with tmp.open('wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def render_guard(report, probe=observe_process):
    require(not report.get('observation_error'), 'render refused: observation error is not terminal verification')
    require(report['state'] in ('completed', 'failed'), 'render requires verified terminal state')
    process = report.get('process')
    require(process is not None and generation_state(process['pid'], process['create_time'], probe) in ('gone', 'pid_reused'), 'render refused: training PID still alive or unconfirmed')
    controller = report.get('controller')
    require(controller is not None and controller.get('state') in ('completed', 'failed'), 'render requires terminal controller record')
    require(controller.get('schema') == 'ept-training-controller-v1', 'render controller schema mismatch')
    for field in ('tree_gone', 'job_gone_before_close', 'job_closed'):
        require(controller.get(field) is True, 'render refused: affirmative whole-job cleanup required: ' + field)
    ownership = controller.get('job_ownership')
    require(type(ownership) is dict and ownership.get('kind') == 'windows_job_kill_on_close'
            and ownership.get('assigned') is True and ownership.get('released') is True,
            'render refused: launch-owned Job containment binding required')
    for prefix in ('controller', 'wrapper'):
        pid, created = controller.get(prefix + '_pid'), controller.get(prefix + '_create_time')
        require(pid is not None, 'render requires controller/wrapper PID generation')
        require(generation_state(pid, created, probe) in ('gone', 'pid_reused'), 'render refused: controller/wrapper still alive or unconfirmed')
    for entry in controller.get('owned_generations', []):
        require(generation_state(entry['pid'], entry['create_time'], probe) in ('gone', 'pid_reused'), 'render refused: owned process tree still alive or unconfirmed')


def light_svg(rows):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="390" viewBox="0 0 1000 390"><rect width="1000" height="390" fill="white"/>']
    for offset, key, title in ((0, 'fit_batch_mean_loss', 'FIT diagnostic: mean training batch loss'), (500, 'stop_loss', 'STOP: pooled evaluation loss')):
        parts.append(f'<text x="{offset+35}" y="30" font-size="17">{title}</text>')
        vals = [r[key] for r in rows]
        if vals:
            lo, hi = min(vals), max(vals)
            span = hi-lo or max(abs(lo)*.05, 1.)
            points = [(offset+65+390*i/max(len(vals)-1, 1), 310-230*(v-lo)/span) for i, v in enumerate(vals)]
            parts.append('<polyline fill="none" stroke="#285e8e" stroke-width="2" points="' + ' '.join(f'{x:.2f},{y:.2f}' for x, y in points) + '"/>')
            for row, (x, y) in zip(rows, points):
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#285e8e"><title>epoch {row["epoch"]}: {row[key]:.9g}</title></circle>')
            parts.append(f'<text x="{offset+8}" y="80" font-size="12">{hi:.5g}</text><text x="{offset+8}" y="315" font-size="12">{lo:.5g}</text>')
            parts.append(f'<text x="{offset+65}" y="345" font-size="12">1</text><text x="{offset+440}" y="345" font-size="12">{len(vals)}</text>')
        else:
            parts.append(f'<text x="{offset+40}" y="180">No completed epoch available</text>')
        parts.append(f'<text x="{offset+200}" y="375" font-size="13">Completed epoch (1-based)</text>')
    parts.append('</svg>')
    return ''.join(parts).encode('utf-8')


def final_plot(report, out, probe=observe_process):
    render_guard(report, probe)
    require(report['history'] is not None, 'no numeric history available to render')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = report['history']
    rows = history['rows']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in zip(axes, ('fit_batch_mean_loss', 'stop_loss'), ('FIT training diagnostic', 'STOP pooled evaluation')):
        ax.plot([r['epoch'] for r in rows], [r[key] for r in rows], marker='o')
        ax.set(xlabel='Completed epoch (1-based)', ylabel=key, title=title)
        ax.grid(alpha=.2)
    axes[1].axvline(history['best_epoch'], color='grey', linestyle='--', label=f'Best STOP epoch {history["best_epoch"]}')
    axes[1].legend()
    reason = history['termination_reason'] if report['state'] == 'completed' else 'failed; last completed epoch only'
    fig.suptitle(f'S / CUDA / AdamW / float32 | {reason}')
    fig.text(.06, .04, 'FIT: arithmetic mean of training-mode batch losses. STOP: pooled scored-row/match/size evaluation.\nSeparate definitions; no generalization-gap inference. Selection: STOP strict minimum, patience 2, min_delta 0.', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .93))
    fig.savefig(out / 'loss-final.png', dpi=140)
    plt.close(fig)


def save_report(report, out, *, render=False, probe=observe_process):
    out = canonical(str(out))
    run = canonical(report['run']) if report.get('run') else None
    source = canonical(report['source'])
    require(not out.is_relative_to(source) and (run is None or not out.is_relative_to(run)), 'report must be outside source and run')
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / 'monitor-state.json'
    previous = read_json(state_path) if state_path.exists() else None
    binding = {k: report.get(k) for k in ('run', 'source', 'code_commit', 'operator_config_sha256', 'run_identity_sha256')}
    if previous:
        old_binding = previous.get('binding', {})
        initial_binding = (old_binding.get('run') is None and
                           all(old_binding.get(k) == binding[k] for k in ('source', 'operator_config_sha256')))
        initial_unverified = (old_binding.get('code_commit') is None and old_binding.get('run_identity_sha256') is None
                              and all(old_binding.get(k) == binding[k] for k in ('run', 'source', 'operator_config_sha256')))
        require(old_binding == binding or initial_binding or initial_unverified, 'monitor state belongs to a different run')
        require(report['epoch'] >= previous.get('epoch', 0), 'completed epoch regressed')
    if render:
        try:
            render_guard(report, probe)
        except MetadataError as exc:
            # Denial suppresses heavy work, not the lightweight failure evidence.
            report['render_refused'] = str(exc)
            render = False
    failure = report.get('error') or report.get('render_refused') or (report['state'] if report['state'] in ('failed', 'interrupted_unconfirmed', 'terminal_unconfirmed', 'terminal_process_alive') else None)
    token = {'epoch': report['epoch'], 'terminal': report['state'] if report['state'] in ('completed', 'failed') else None, 'failure': failure}
    if previous and all(previous.get(k) == v for k, v in token.items()) and (not render or previous.get('rendered')):
        if previous.get('status_snapshot') != report['state'] or previous.get('binding') != binding:
            atomic_json(state_path, {**previous, 'binding': binding, 'status_snapshot': report['state']})
        return False
    # No invented epoch-zero curves; transitions and errors still receive snapshots.
    if render:
        final_plot(report, out, probe)
    history = report.get('history')
    rows = [] if history is None else history['rows']
    atomic_json(out / 'report.json', report)
    if history:
        reader = checkpoint_reader(source)
        atomic_bytes(out / 'epoch_metrics.csv', reader.epoch_metrics.csv_bytes(rows))
        atomic_json(out / 'epoch_metrics.json', history)
    atomic_bytes(out / 'loss-live.svg', light_svg(rows))
    reason = history['termination_reason'] if history else 'unavailable'
    best = history['best_epoch'] if history else 'unavailable'
    text = f'# EPT 단일 파일럿 관측\n\n상태: {report["state"]}; 완료 에폭: {report["epoch"]}; STOP 선택 에폭(1 기반): {best}; 기록된 종료 사유: {reason}.\n\n'
    text += f'오류: {report.get("error", "없음")}\n\n'
    text += 'FIT는 학습 모드 배치 손실의 산술평균 진단값이며 STOP는 기존 pooled 평가 손실입니다. 두 정의로 일반화 차이·수렴·기준선 우위를 추론하지 않습니다.\n\n'
    text += '설정: ' + json.dumps(report.get('training_parameters', PARAMETERS), ensure_ascii=False, sort_keys=True) + '\n\n'
    text += '출처: ' + json.dumps(binding, ensure_ascii=False, sort_keys=True) + '\n\n'
    text += '메타데이터만 검증했으며 가중치를 검증하지 않았습니다. 메모리는 표본이며 연속 상한 보장이 아닙니다. 비용·토큰 미측정은 null입니다.\n'
    atomic_bytes(out / 'report.ko.md', text.encode('utf-8'))
    image = '<img src="loss-final.png" alt="Separate FIT and STOP loss panels">' if render or previous and previous.get('rendered') else ''
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>EPT pilot observation</title><style>body{max-width:1100px;margin:2em auto;font-family:system-ui}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>EPT pilot</h1><p>' + html.escape(report['state']) + '</p><img src="loss-live.svg" alt="Completed epoch diagnostics">' + image + '<pre>' + html.escape(text) + '</pre><p><a href="report.json">JSON</a> · <a href="epoch_metrics.csv">CSV</a> · <a href="report.ko.md">한국어 보고</a></p></html>'
    prior_snapshot = previous.get('snapshot') if previous else None
    event = f'epoch_{report["epoch"]:04d}_' + digest({**token, 'render': bool(render), 'previous_snapshot': prior_snapshot})[:16]
    snapshot = out / 'history' / event
    snapshot.mkdir(parents=True, exist_ok=True)
    names = ['report.json', 'report.ko.md', 'loss-live.svg']
    if history:
        names.extend(['epoch_metrics.json', 'epoch_metrics.csv'])
    if render:
        names.append('loss-final.png')
    hashes = {}
    for name in names:
        raw = read_bytes(out / name)
        target = snapshot / name
        if target.exists():
            require(read_bytes(target) == raw, 'immutable report snapshot conflict')
        else:
            atomic_bytes(target, raw)
        hashes[name] = sha(raw)
    snapshot_manifest = {'schema': 'ept-training-report-snapshot-v1', 'binding': binding, **token, 'previous_snapshot': prior_snapshot,
                         'rendered': bool(render), 'files_sha256': hashes}
    if (snapshot / 'snapshot.json').exists():
        require(read_json(snapshot / 'snapshot.json') == snapshot_manifest, 'immutable snapshot manifest conflict')
    else:
        atomic_json(snapshot / 'snapshot.json', snapshot_manifest)
    events = sorted((out / 'history').glob('epoch_*/snapshot.json'))
    links = ''.join('<li><a href="history/' + p.parent.name + '/report.json">' + p.parent.name + '</a></li>' for p in events)
    page = page.replace('</html>', '<h2>Retained observations</h2><ul>' + links + '</ul></html>')
    atomic_bytes(out / 'gallery.html', page.encode('utf-8'))
    atomic_json(state_path, {'schema': 'ept-training-monitor-state-v1', 'binding': binding, **token,
                            'rendered': bool(render), 'snapshot': snapshot.relative_to(out).as_posix(), 'status_snapshot': report['state']})
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--config', required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument('--run')
    selector.add_argument('--control')
    parser.add_argument('--out', required=True)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args(argv)
    source, out = canonical(args.source), canonical(args.out)
    config = load_config(canonical(args.config))
    control = canonical(args.control) if args.control else None
    run = canonical(args.run) if args.run else None
    try:
        if control:
            if (control / 'run.json').exists():
                run = validate_ref(read_json(control / 'run.json'), config, source)
            else:
                controller = read_json(control / 'controller.json') if (control / 'controller.json').exists() else None
                controller_generation, wrapper_generation = None, None
                state = 'bootstrapping'
                if controller:
                    validate_controller(controller, source, config)
                    controller_generation = generation_state(controller['controller_pid'], controller['controller_create_time'])
                    if controller.get('wrapper_pid') is not None:
                        wrapper_generation = generation_state(controller['wrapper_pid'], controller['wrapper_create_time'])
                    if controller['state'] == 'failed':
                        state = 'failed'
                    elif controller['state'] == 'completed':
                        state = 'observation_error'
                    elif controller_generation != 'alive':
                        state = 'interrupted_unconfirmed'
                report = {'schema': REPORT_SCHEMA, 'source': str(source), 'run': None,
                          'code_commit': controller.get('code_commit') if controller else None,
                          'operator_config_sha256': digest(config), 'run_identity_sha256': None,
                          'state': state, 'controller_generation': controller_generation,
                          'wrapper_generation': wrapper_generation,
                          'epoch': 0, 'history': None, 'controller': controller}
                if state in ('failed', 'observation_error'):
                    report['error'] = 'controller terminal before durable RUN/ref; no retry'
        if run is not None:
            report = inspect_run(run, source, config, control=control)
    except (MetadataError, RuntimeError, OSError, KeyError, TypeError, ValueError) as exc:
        evidence = exc.evidence if isinstance(exc, ObservationError) else {}
        report = {'schema': REPORT_SCHEMA, 'source': str(source), 'run': str(run) if run else None,
                  'code_commit': None, 'operator_config_sha256': digest(config), 'run_identity_sha256': None,
                  'state': 'observation_error', 'epoch': 0, 'history': None}
        # Preserve the last valid binding/history; an unreadable final is an error, never completion.
        prior = out / 'report.json'
        if prior.exists():
            old = read_json(prior)
            require(old.get('source') == str(source) and old.get('run') == report['run']
                    and old.get('operator_config_sha256') == digest(config), 'error report binding changed')
            report = {**old, 'last_verified_state': old.get('state'), 'state': 'observation_error'}
        report.update({k: evidence[k] for k in ('code_commit', 'operator_config_sha256', 'run_identity_sha256',
                                              'process', 'process_generation', 'controller') if k in evidence})
        report.update(state='failed' if evidence.get('actual_failed') and getattr(exc, 'stable', False) else 'observation_error', error=str(exc),
                      observation_error={'attempts': getattr(exc, 'attempts', 1),
                                         'stable_bytes': getattr(exc, 'stable', None),
                                         'observed_state': evidence.get('observed_state'),
                                         'diagnostics': evidence.get('diagnostics', [])})
    changed = save_report(report, out, render=args.render)
    if changed:
        print(json.dumps({'state': report['state'], 'epoch': report['epoch'], 'report': str(out / 'report.json')}, ensure_ascii=False))
    return 0 if not report.get('render_refused') and report['state'] not in ('failed', 'observation_error', 'interrupted_unconfirmed', 'terminal_unconfirmed', 'terminal_process_alive') else 1


if __name__ == '__main__':
    raise SystemExit(main())
